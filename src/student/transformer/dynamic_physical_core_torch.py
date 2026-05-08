from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class DynamicPhysicalCoreConfig:
    """
    Dynamic physical core configuration.

    当前支持刚度类全局参数：
        alpha_x  -> K_x_template
        alpha_xy -> K_xy_template

    不允许输出后修正位移；所有响应必须由 MCK / Newmark 方程解出。
    """

    dt: float = 0.01
    gamma: float = 0.5
    beta: float = 0.25
    dtype: torch.dtype = torch.float64
    linear_solve_mode: str = "solve"
    symmetrize_k_eff: bool = True
    precompute_newmark_matrices: bool = False


class DynamicPhysicalCoreTorch(nn.Module):
    """
    Full-order dynamic physical core.

    支持参数：
        alpha_x:
            theta slice shape = (1,)
            template shape    = (D, D)
            K contribution    = alpha_x * K_x_template

        alpha_xy:
            theta slice shape = (1,)
            template shape    = (D, D)
            K contribution    = alpha_xy * K_xy_template
    """

    PARAM_TO_TEMPLATE = {
        "alpha_x": "K_x_template",
        "alpha_xy": "K_xy_template",
    }

    def __init__(
        self,
        *,
        M0: np.ndarray | torch.Tensor,
        K0: np.ndarray | torch.Tensor,
        C0: np.ndarray | torch.Tensor,
        stiffness_templates: dict[str, np.ndarray | torch.Tensor],
        registry: Any,
        config: Optional[DynamicPhysicalCoreConfig] = None,
    ) -> None:
        super().__init__()

        self.config = config if config is not None else DynamicPhysicalCoreConfig()
        self.registry = registry
        self.dtype = self.config.dtype

        M0_t = self._as_matrix_tensor(M0, "M0")
        K0_t = self._as_matrix_tensor(K0, "K0")
        C0_t = self._as_matrix_tensor(C0, "C0")

        if M0_t.shape != K0_t.shape or K0_t.shape != C0_t.shape:
            raise ValueError(
                f"M0/K0/C0 shape mismatch: "
                f"M0={tuple(M0_t.shape)}, K0={tuple(K0_t.shape)}, C0={tuple(C0_t.shape)}"
            )

        self.n_dofs = int(M0_t.shape[0])
        self.register_buffer("M0", M0_t)
        self.register_buffer("K0", K0_t)
        self.register_buffer("C0", C0_t)

        unsupported = [name for name in self.registry.names if name not in self.PARAM_TO_TEMPLATE]
        if unsupported:
            raise ValueError(
                f"DynamicPhysicalCoreTorch does not support parameters: {unsupported}. "
                f"Supported: {list(self.PARAM_TO_TEMPLATE.keys())}."
            )

        self.enabled_template_names: dict[str, str] = {}
        for param_name in self.registry.names:
            template_name = self.PARAM_TO_TEMPLATE[param_name]
            if template_name not in stiffness_templates:
                raise KeyError(
                    f"Missing required stiffness template {template_name!r} for parameter {param_name!r}. "
                    f"Available templates: {list(stiffness_templates.keys())}"
                )
            tpl = self._as_matrix_tensor(stiffness_templates[template_name], template_name)
            if tpl.shape != K0_t.shape:
                raise ValueError(
                    f"{template_name} shape mismatch: expected {tuple(K0_t.shape)}, got {tuple(tpl.shape)}."
                )
            self.register_buffer(template_name, tpl)
            self.enabled_template_names[param_name] = template_name

        if bool(self.config.precompute_newmark_matrices):
            self.register_buffer(
                "newmark_A_base",
                self._build_newmark_A_base(),
                persistent=False,
            )
        else:
            self.newmark_A_base = None

    def _as_matrix_tensor(self, x: np.ndarray | torch.Tensor, name: str) -> torch.Tensor:
        out = torch.as_tensor(x, dtype=self.dtype)
        if out.ndim != 2:
            raise ValueError(f"{name} must be a 2D matrix, got shape={tuple(out.shape)}.")
        if out.shape[0] != out.shape[1]:
            raise ValueError(f"{name} must be square, got shape={tuple(out.shape)}.")
        return out

    def _normalize_theta(self, theta_t: torch.Tensor) -> torch.Tensor:
        theta = torch.as_tensor(theta_t, dtype=self.dtype, device=self.K0.device)
        if theta.ndim != 1:
            raise ValueError(
                f"theta_t must be 1D for one-step assembly, got shape={tuple(theta.shape)}."
            )
        expected = int(self.registry.total_dim)
        if int(theta.numel()) != expected:
            raise ValueError(
                f"theta_t dimension mismatch: expected {expected}, got {theta.numel()}."
            )
        return theta

    def assemble_stiffness(
        self,
        theta_t: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Assemble K_eff(theta_t).

        theta_t=None:
            K_eff = K0

        enabled params:
            K_eff = K0 + sum_p theta[p] * K_p_template
        """
        if theta_t is None:
            return self.K0, {}

        theta = self._normalize_theta(theta_t)
        theta_dict = self.registry.split_theta(theta)

        K_eff = self.K0
        dK_dict: dict[str, torch.Tensor] = {}
        for param_name in self.registry.names:
            value = theta_dict[param_name].reshape(-1)
            if value.numel() != 1:
                raise ValueError(f"{param_name} must have one value, got {value.numel()}.")
            template_name = self.enabled_template_names[param_name]
            template = getattr(self, template_name)
            dK = value[0] * template
            K_eff = K_eff + dK
            dK_dict[param_name] = dK

        if bool(self.config.symmetrize_k_eff):
            K_eff = 0.5 * (K_eff + K_eff.T)

        return K_eff, dK_dict

    def _newmark_constants(self) -> tuple[float, float, float, float, float, float]:
        """Return Newmark constants for the current fixed dt/gamma/beta.

        These constants do not depend on theta or on the current state, so the
        dynamic step can reuse them without changing the numerical scheme.
        """
        dt = float(self.config.dt)
        gamma = float(self.config.gamma)
        beta = float(self.config.beta)

        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}.")
        if beta <= 0.0:
            raise ValueError(f"Newmark beta must be positive, got {beta}.")

        a0 = 1.0 / (beta * dt * dt)
        a1 = gamma / (beta * dt)
        a2 = 1.0 / (beta * dt)
        a3 = 1.0 / (2.0 * beta) - 1.0
        a4 = gamma / beta - 1.0
        a5 = dt * (gamma / (2.0 * beta) - 1.0)
        return a0, a1, a2, a3, a4, a5

    def _base_stiffness_for_newmark(self) -> torch.Tensor:
        """Return the K0 part used in K_eff before adding dynamic templates."""
        if bool(self.config.symmetrize_k_eff):
            return 0.5 * (self.K0 + self.K0.T)
        return self.K0

    def _build_newmark_A_base(self) -> torch.Tensor:
        a0, a1, _, _, _, _ = self._newmark_constants()
        return self._base_stiffness_for_newmark() + a0 * self.M0 + a1 * self.C0

    def _assemble_dynamic_stiffness_delta(self, theta_t: Optional[torch.Tensor]) -> torch.Tensor:
        """Assemble only the theta-dependent part of K_eff.

        This is algebraically equivalent to ``assemble_stiffness(theta_t) - K0``
        under the same symmetrization rule, but avoids repeatedly adding K0
        when Newmark's constant effective matrix part is precomputed.
        """
        if theta_t is None:
            return torch.zeros_like(self.K0)

        theta = self._normalize_theta(theta_t)
        theta_dict = self.registry.split_theta(theta)

        dK_total = torch.zeros_like(self.K0)
        for param_name in self.registry.names:
            value = theta_dict[param_name].reshape(-1)
            if value.numel() != 1:
                raise ValueError(f"{param_name} must have one value, got {value.numel()}.")
            template_name = self.enabled_template_names[param_name]
            template = getattr(self, template_name)
            dK_total = dK_total + value[0] * template

        if bool(self.config.symmetrize_k_eff):
            dK_total = 0.5 * (dK_total + dK_total.T)
        return dK_total

    def _newmark_effective_matrix(self, theta_t: Optional[torch.Tensor]) -> torch.Tensor:
        """Build A = K_eff + a0 M + a1 C for one Newmark step.

        With ``precompute_newmark_matrices=True``, the constant part
        ``sym(K0) + a0 M + a1 C`` is reused and only the theta-dependent
        stiffness delta is assembled at every time step. This keeps the same
        equation and loss while reducing repeated matrix additions.
        """
        a0, a1, _, _, _, _ = self._newmark_constants()
        if bool(self.config.precompute_newmark_matrices):
            A_base = self.newmark_A_base
            if A_base is None:
                A_base = self._build_newmark_A_base()
            return A_base + self._assemble_dynamic_stiffness_delta(theta_t)

        K_eff, _ = self.assemble_stiffness(theta_t)
        return K_eff + a0 * self.M0 + a1 * self.C0

    def _linear_solve(self, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        mode = str(self.config.linear_solve_mode)
        if mode == "solve":
            return torch.linalg.solve(A, b)
        if mode == "cholesky":
            L = torch.linalg.cholesky(A)
            return torch.cholesky_solve(b.unsqueeze(-1), L).squeeze(-1)
        raise ValueError(f"Unsupported linear_solve_mode={mode!r}. Expected 'solve' or 'cholesky'.")

    def _newmark_step_single(
        self,
        *,
        u_t: torch.Tensor,
        v_t: torch.Tensor,
        a_t: torch.Tensor,
        F_t1: torch.Tensor,
        theta_t: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u_t = torch.as_tensor(u_t, dtype=self.dtype, device=self.K0.device)
        v_t = torch.as_tensor(v_t, dtype=self.dtype, device=self.K0.device)
        a_t = torch.as_tensor(a_t, dtype=self.dtype, device=self.K0.device)
        F_t1 = torch.as_tensor(F_t1, dtype=self.dtype, device=self.K0.device)

        if u_t.ndim != 1 or v_t.ndim != 1 or a_t.ndim != 1 or F_t1.ndim != 1:
            raise ValueError(
                "_newmark_step_single expects 1D tensors. "
                f"Got u={tuple(u_t.shape)}, v={tuple(v_t.shape)}, "
                f"a={tuple(a_t.shape)}, F={tuple(F_t1.shape)}."
            )

        D = int(self.n_dofs)
        if u_t.numel() != D or v_t.numel() != D or a_t.numel() != D or F_t1.numel() != D:
            raise ValueError(
                f"State/load dimension mismatch with n_dofs={D}: "
                f"u={u_t.numel()}, v={v_t.numel()}, a={a_t.numel()}, F={F_t1.numel()}."
            )

        dt = float(self.config.dt)
        gamma = float(self.config.gamma)
        a0, a1, a2, a3, a4, a5 = self._newmark_constants()

        M = self.M0
        C = self.C0

        A = self._newmark_effective_matrix(theta_t)
        rhs = (
            F_t1
            + M @ (a0 * u_t + a2 * v_t + a3 * a_t)
            + C @ (a1 * u_t + a4 * v_t + a5 * a_t)
        )

        u_t1 = self._linear_solve(A, rhs)
        a_t1 = a0 * (u_t1 - u_t) - a2 * v_t - a3 * a_t
        v_t1 = v_t + dt * ((1.0 - gamma) * a_t + gamma * a_t1)

        return u_t1, v_t1, a_t1

    def newmark_step(
        self,
        *,
        u_t: torch.Tensor,
        v_t: torch.Tensor,
        a_t: torch.Tensor,
        F_t1: torch.Tensor,
        theta_t: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u_t = torch.as_tensor(u_t, dtype=self.dtype, device=self.K0.device)
        v_t = torch.as_tensor(v_t, dtype=self.dtype, device=self.K0.device)
        a_t = torch.as_tensor(a_t, dtype=self.dtype, device=self.K0.device)
        F_t1 = torch.as_tensor(F_t1, dtype=self.dtype, device=self.K0.device)

        if u_t.ndim == 1 and v_t.ndim == 1 and a_t.ndim == 1 and F_t1.ndim == 1:
            return self._newmark_step_single(
                u_t=u_t,
                v_t=v_t,
                a_t=a_t,
                F_t1=F_t1,
                theta_t=theta_t,
            )

        if u_t.ndim != 2 or v_t.ndim != 2 or a_t.ndim != 2 or F_t1.ndim != 2:
            raise ValueError(
                "u_t, v_t, a_t, F_t1 must be either all 1D or all 2D tensors. "
                f"Got u={tuple(u_t.shape)}, v={tuple(v_t.shape)}, "
                f"a={tuple(a_t.shape)}, F={tuple(F_t1.shape)}."
            )

        B, D = int(u_t.shape[0]), int(u_t.shape[1])
        if int(v_t.shape[0]) != B or int(a_t.shape[0]) != B or int(F_t1.shape[0]) != B:
            raise ValueError("Batch size mismatch.")
        if int(v_t.shape[1]) != D or int(a_t.shape[1]) != D or int(F_t1.shape[1]) != D or D != int(self.n_dofs):
            raise ValueError(
                f"DOF dimension mismatch with n_dofs={self.n_dofs}: "
                f"u={tuple(u_t.shape)}, v={tuple(v_t.shape)}, a={tuple(a_t.shape)}, F={tuple(F_t1.shape)}."
            )

        if theta_t is None:
            theta_batch = [None for _ in range(B)]
        else:
            theta = torch.as_tensor(theta_t, dtype=self.dtype, device=self.K0.device)
            if theta.ndim == 1:
                theta_batch = [theta for _ in range(B)]
            elif theta.ndim == 2:
                if int(theta.shape[0]) != B:
                    raise ValueError(
                        f"Batch theta size mismatch: theta batch={theta.shape[0]}, state batch={B}."
                    )
                theta_batch = [theta[i] for i in range(B)]
            else:
                raise ValueError(f"theta_t must be None, 1D, or 2D, got shape={tuple(theta.shape)}.")

        u_out = []
        v_out = []
        a_out = []
        for b in range(B):
            u1, v1, a1 = self._newmark_step_single(
                u_t=u_t[b],
                v_t=v_t[b],
                a_t=a_t[b],
                F_t1=F_t1[b],
                theta_t=theta_batch[b],
            )
            u_out.append(u1)
            v_out.append(v1)
            a_out.append(a1)

        return torch.stack(u_out, dim=0), torch.stack(v_out, dim=0), torch.stack(a_out, dim=0)
