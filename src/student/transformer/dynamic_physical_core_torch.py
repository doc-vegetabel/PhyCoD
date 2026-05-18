from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

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

    PARAM_TO_STIFFNESS_TEMPLATE = {
        "alpha_x": "K_x_template",
        "alpha_xy": "K_xy_template",
    }
    PARAM_TO_DAMPING_TEMPLATE = {
        "beta_damp_x": "C_hf_x_template",
        "beta_damp_y": "C_hf_y_template",
        "beta_damp_hf_x": "C_hf_x_template",
        "beta_damp_hf_y": "C_hf_y_template",
    }
    PARAM_TO_TEMPLATE = PARAM_TO_STIFFNESS_TEMPLATE

    def __init__(
        self,
        *,
        M0: np.ndarray | torch.Tensor,
        K0: np.ndarray | torch.Tensor,
        C0: np.ndarray | torch.Tensor,
        stiffness_templates: dict[str, np.ndarray | torch.Tensor],
        damping_templates: dict[str, np.ndarray | torch.Tensor] | None = None,
        registry: Any,
        config: Optional[DynamicPhysicalCoreConfig] = None,
    ) -> None:
        super().__init__()

        self.config = config if config is not None else DynamicPhysicalCoreConfig()
        self.registry = registry
        self.dtype = self.config.dtype
        self._newmark_constants_cache = self._compute_newmark_constants()
        self._dt = float(self.config.dt)
        self._gamma = float(self.config.gamma)

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

        damping_templates = {} if damping_templates is None else dict(damping_templates)

        unsupported: list[str] = []
        for name in self.registry.names:
            spec = self.registry.get_spec(name)
            if spec.target == "K":
                if (spec.template_name or self.PARAM_TO_STIFFNESS_TEMPLATE.get(name)) is None:
                    unsupported.append(name)
            elif spec.target == "C":
                if (spec.template_name or self.PARAM_TO_DAMPING_TEMPLATE.get(name)) is None:
                    unsupported.append(name)
            else:
                unsupported.append(name)
        if unsupported:
            raise ValueError(
                f"DynamicPhysicalCoreTorch does not support parameters: {unsupported}. "
                f"Supported stiffness params: {list(self.PARAM_TO_STIFFNESS_TEMPLATE.keys())}; "
                f"supported damping params: {list(self.PARAM_TO_DAMPING_TEMPLATE.keys())}."
            )

        self.enabled_stiffness_template_names: dict[str, str] = {}
        self.enabled_damping_template_names: dict[str, str] = {}
        for param_name in self.registry.names:
            spec = self.registry.get_spec(param_name)
            if spec.target == "K":
                template_name = spec.template_name or self.PARAM_TO_STIFFNESS_TEMPLATE[param_name]
                template_source = stiffness_templates
                template_kind = "stiffness"
            elif spec.target == "C":
                template_name = spec.template_name or self.PARAM_TO_DAMPING_TEMPLATE[param_name]
                template_source = damping_templates
                template_kind = "damping"
            else:
                raise ValueError(f"Unsupported physical target {spec.target!r} for {param_name!r}.")

            if template_name not in template_source:
                raise KeyError(
                    f"Missing required {template_kind} template {template_name!r} "
                    f"for parameter {param_name!r}. Available templates: {list(template_source.keys())}"
                )
            tpl = self._as_matrix_tensor(template_source[template_name], template_name)
            if tpl.shape != K0_t.shape:
                raise ValueError(
                    f"{template_name} shape mismatch: expected {tuple(K0_t.shape)}, got {tuple(tpl.shape)}."
                )
            self.register_buffer(template_name, tpl)
            if spec.target == "K":
                self.enabled_stiffness_template_names[param_name] = template_name
            else:
                self.enabled_damping_template_names[param_name] = template_name

        self.enabled_param_names = tuple(self.registry.names)
        self.enabled_template_names = {
            **self.enabled_stiffness_template_names,
            **self.enabled_damping_template_names,
        }
        self.enabled_stiffness_template_buffer_names = tuple(
            self.enabled_stiffness_template_names[name]
            for name in self.enabled_param_names
            if name in self.enabled_stiffness_template_names
        )
        self.enabled_damping_template_buffer_names = tuple(
            self.enabled_damping_template_names[name]
            for name in self.enabled_param_names
            if name in self.enabled_damping_template_names
        )
        self.enabled_template_buffer_names = tuple(
            self.enabled_template_names[name]
            for name in self.enabled_param_names
            if name in self.enabled_template_names
        )
        slices = self.registry.slices
        self._scalar_param_indices: dict[str, int] = {}
        self._fast_scalar_theta = True
        for name in self.enabled_param_names:
            sl = slices[name]
            if int(sl.stop - sl.start) != 1:
                self._fast_scalar_theta = False
                break
            self._scalar_param_indices[name] = int(sl.start)

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

        enabled stiffness params:
            K_eff = K0 + sum_p theta[p] * K_p_template
        """
        if theta_t is None:
            return self.K0, {}

        theta = self._normalize_theta(theta_t)
        theta_dict = self.registry.split_theta(theta)

        K_eff = self.K0
        dK_dict: dict[str, torch.Tensor] = {}
        for param_name, template_name in self.enabled_stiffness_template_names.items():
            value = theta_dict[param_name].reshape(-1)
            if value.numel() != 1:
                raise ValueError(f"{param_name} must have one value, got {value.numel()}.")
            template = getattr(self, template_name)
            dK = value[0] * template
            K_eff = K_eff + dK
            dK_dict[param_name] = dK

        if bool(self.config.symmetrize_k_eff):
            K_eff = 0.5 * (K_eff + K_eff.T)

        return K_eff, dK_dict

    def assemble_damping(
        self,
        theta_t: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Assemble C_eff(theta_t).

        theta_t=None:
            C_eff = C0

        enabled damping params:
            C_eff = C0 + sum_p theta[p] * C_p_template
        """
        if theta_t is None:
            return self.C0, {}

        theta = self._normalize_theta(theta_t)
        theta_dict = self.registry.split_theta(theta)

        C_eff = self.C0
        dC_dict: dict[str, torch.Tensor] = {}
        for param_name, template_name in self.enabled_damping_template_names.items():
            value = theta_dict[param_name].reshape(-1)
            if value.numel() != 1:
                raise ValueError(f"{param_name} must have one value, got {value.numel()}.")
            template = getattr(self, template_name)
            dC = value[0] * template
            C_eff = C_eff + dC
            dC_dict[param_name] = dC

        if bool(self.config.symmetrize_k_eff):
            C_eff = 0.5 * (C_eff + C_eff.T)

        return C_eff, dC_dict

    def _compute_newmark_constants(self) -> tuple[float, float, float, float, float, float]:
        """Compute Newmark constants for the current fixed dt/gamma/beta."""
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

    def _newmark_constants(self) -> tuple[float, float, float, float, float, float]:
        """Return cached Newmark constants for the fixed dt/gamma/beta."""
        return self._newmark_constants_cache

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
        for param_name, template_name in self.enabled_stiffness_template_names.items():
            value = theta_dict[param_name].reshape(-1)
            if value.numel() != 1:
                raise ValueError(f"{param_name} must have one value, got {value.numel()}.")
            template = getattr(self, template_name)
            dK_total = dK_total + value[0] * template

        if bool(self.config.symmetrize_k_eff):
            dK_total = 0.5 * (dK_total + dK_total.T)
        return dK_total

    def _assemble_dynamic_stiffness_delta_fast(self, theta: torch.Tensor) -> torch.Tensor:
        """Fast theta-dependent stiffness assembly for already-normalized scalar theta."""
        if not self._fast_scalar_theta:
            return self._assemble_dynamic_stiffness_delta(theta)

        dK_total = None
        for param_name, template_name in self.enabled_stiffness_template_names.items():
            idx = self._scalar_param_indices[param_name]
            template = getattr(self, template_name)
            dK = theta[idx] * template
            dK_total = dK if dK_total is None else dK_total + dK

        if dK_total is None:
            dK_total = torch.zeros_like(self.K0)

        if bool(self.config.symmetrize_k_eff):
            dK_total = 0.5 * (dK_total + dK_total.T)
        return dK_total

    def _assemble_dynamic_damping_delta(self, theta_t: Optional[torch.Tensor]) -> torch.Tensor:
        """Assemble only the theta-dependent part of C_eff."""
        if theta_t is None:
            return torch.zeros_like(self.C0)

        theta = self._normalize_theta(theta_t)
        theta_dict = self.registry.split_theta(theta)

        dC_total = torch.zeros_like(self.C0)
        for param_name, template_name in self.enabled_damping_template_names.items():
            value = theta_dict[param_name].reshape(-1)
            if value.numel() != 1:
                raise ValueError(f"{param_name} must have one value, got {value.numel()}.")
            template = getattr(self, template_name)
            dC_total = dC_total + value[0] * template

        if bool(self.config.symmetrize_k_eff):
            dC_total = 0.5 * (dC_total + dC_total.T)
        return dC_total

    def _assemble_dynamic_damping_delta_fast(self, theta: torch.Tensor) -> torch.Tensor:
        """Fast theta-dependent damping assembly for already-normalized scalar theta."""
        if not self._fast_scalar_theta:
            return self._assemble_dynamic_damping_delta(theta)

        dC_total = None
        for param_name, template_name in self.enabled_damping_template_names.items():
            idx = self._scalar_param_indices[param_name]
            template = getattr(self, template_name)
            dC = theta[idx] * template
            dC_total = dC if dC_total is None else dC_total + dC

        if dC_total is None:
            dC_total = torch.zeros_like(self.C0)

        if bool(self.config.symmetrize_k_eff):
            dC_total = 0.5 * (dC_total + dC_total.T)
        return dC_total

    def _assemble_stiffness_fast(self, theta: torch.Tensor) -> torch.Tensor:
        """Fast K_eff assembly for already-normalized scalar theta."""
        if not self._fast_scalar_theta:
            K_eff, _ = self.assemble_stiffness(theta)
            return K_eff

        K_eff = self.K0
        for param_name, template_name in self.enabled_stiffness_template_names.items():
            idx = self._scalar_param_indices[param_name]
            template = getattr(self, template_name)
            K_eff = K_eff + theta[idx] * template

        if bool(self.config.symmetrize_k_eff):
            K_eff = 0.5 * (K_eff + K_eff.T)
        return K_eff

    def _assemble_damping_fast(self, theta: Optional[torch.Tensor]) -> torch.Tensor:
        """Fast C_eff assembly for theta tensors already on core device/dtype."""
        if theta is None:
            return self.C0
        if not self._fast_scalar_theta:
            C_eff, _ = self.assemble_damping(theta)
            return C_eff

        C_eff = self.C0
        for param_name, template_name in self.enabled_damping_template_names.items():
            idx = self._scalar_param_indices[param_name]
            template = getattr(self, template_name)
            C_eff = C_eff + theta[idx] * template

        if bool(self.config.symmetrize_k_eff):
            C_eff = 0.5 * (C_eff + C_eff.T)
        return C_eff

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
            return (
                A_base
                + self._assemble_dynamic_stiffness_delta(theta_t)
                + a1 * self._assemble_dynamic_damping_delta(theta_t)
            )

        K_eff, _ = self.assemble_stiffness(theta_t)
        C_eff, _ = self.assemble_damping(theta_t)
        return K_eff + a0 * self.M0 + a1 * C_eff

    def _newmark_effective_matrix_fast(self, theta_t: Optional[torch.Tensor]) -> torch.Tensor:
        """Fast A assembly for theta tensors already on core device/dtype."""
        a0, a1, _, _, _, _ = self._newmark_constants()
        if theta_t is None:
            if bool(self.config.precompute_newmark_matrices):
                A_base = self.newmark_A_base
                if A_base is None:
                    A_base = self._build_newmark_A_base()
                return A_base
            return self._base_stiffness_for_newmark() + a0 * self.M0 + a1 * self.C0

        if bool(self.config.precompute_newmark_matrices):
            A_base = self.newmark_A_base
            if A_base is None:
                A_base = self._build_newmark_A_base()
            return (
                A_base
                + self._assemble_dynamic_stiffness_delta_fast(theta_t)
                + a1 * self._assemble_dynamic_damping_delta_fast(theta_t)
            )

        K_eff = self._assemble_stiffness_fast(theta_t)
        C_eff = self._assemble_damping_fast(theta_t)
        return K_eff + a0 * self.M0 + a1 * C_eff

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
        C, _ = self.assemble_damping(theta_t)

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

    def _newmark_step_single_fast(
        self,
        *,
        u_t: torch.Tensor,
        v_t: torch.Tensor,
        a_t: torch.Tensor,
        F_t1: torch.Tensor,
        theta_t: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unchecked one-step Newmark update for tensors already on core device/dtype."""
        dt = self._dt
        gamma = self._gamma
        a0, a1, a2, a3, a4, a5 = self._newmark_constants()

        A = self._newmark_effective_matrix_fast(theta_t)
        C = self._assemble_damping_fast(theta_t)
        rhs = (
            F_t1
            + self.M0 @ (a0 * u_t + a2 * v_t + a3 * a_t)
            + C @ (a1 * u_t + a4 * v_t + a5 * a_t)
        )

        u_t1 = self._linear_solve(A, rhs)
        a_t1 = a0 * (u_t1 - u_t) - a2 * v_t - a3 * a_t
        v_t1 = v_t + dt * ((1.0 - gamma) * a_t + gamma * a_t1)

        return u_t1, v_t1, a_t1

    def _newmark_step_single_fast_timed(
        self,
        *,
        u_t: torch.Tensor,
        v_t: torch.Tensor,
        a_t: torch.Tensor,
        F_t1: torch.Tensor,
        theta_t: Optional[torch.Tensor] = None,
        time_fn: Callable[[], float],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
        """Timed variant of the unchecked fast Newmark update."""
        dt = self._dt
        gamma = self._gamma
        a0, a1, a2, a3, a4, a5 = self._newmark_constants()
        timing: dict[str, float] = {}

        t0 = time_fn()
        A = self._newmark_effective_matrix_fast(theta_t)
        timing["newmark_assemble_seconds"] = time_fn() - t0

        t0 = time_fn()
        rhs = (
            F_t1
            + self.M0 @ (a0 * u_t + a2 * v_t + a3 * a_t)
            + self._assemble_damping_fast(theta_t) @ (a1 * u_t + a4 * v_t + a5 * a_t)
        )
        timing["newmark_rhs_seconds"] = time_fn() - t0

        t0 = time_fn()
        u_t1 = self._linear_solve(A, rhs)
        timing["newmark_solve_seconds"] = time_fn() - t0

        t0 = time_fn()
        a_t1 = a0 * (u_t1 - u_t) - a2 * v_t - a3 * a_t
        v_t1 = v_t + dt * ((1.0 - gamma) * a_t + gamma * a_t1)
        timing["newmark_update_seconds"] = time_fn() - t0

        return u_t1, v_t1, a_t1, timing

    def newmark_step_fast(
        self,
        *,
        u_t: torch.Tensor,
        v_t: torch.Tensor,
        a_t: torch.Tensor,
        F_t1: torch.Tensor,
        theta_t: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fast Newmark step for the rollout hot path.

        The caller must pass tensors already converted to the physical-core
        device and dtype. This keeps the same Newmark equation and linear solve
        mode as ``newmark_step`` while avoiding repeated dtype/device wrapping,
        shape checks, and registry splitting inside the time loop.
        """
        if u_t.ndim == 1:
            return self._newmark_step_single_fast(
                u_t=u_t,
                v_t=v_t,
                a_t=a_t,
                F_t1=F_t1,
                theta_t=theta_t,
            )

        B = int(u_t.shape[0])
        if theta_t is None:
            theta_batch = [None for _ in range(B)]
        elif theta_t.ndim == 1:
            theta_batch = [theta_t for _ in range(B)]
        else:
            theta_batch = [theta_t[i] for i in range(B)]

        if B == 1:
            u1, v1, a1 = self._newmark_step_single_fast(
                u_t=u_t[0],
                v_t=v_t[0],
                a_t=a_t[0],
                F_t1=F_t1[0],
                theta_t=theta_batch[0],
            )
            return u1.unsqueeze(0), v1.unsqueeze(0), a1.unsqueeze(0)

        u_out = []
        v_out = []
        a_out = []
        for b in range(B):
            u1, v1, a1 = self._newmark_step_single_fast(
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

    def newmark_step_fast_timed(
        self,
        *,
        u_t: torch.Tensor,
        v_t: torch.Tensor,
        a_t: torch.Tensor,
        F_t1: torch.Tensor,
        theta_t: Optional[torch.Tensor] = None,
        time_fn: Callable[[], float],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
        """
        Timed Newmark step for profiling only.

        This follows the same operations as ``newmark_step_fast`` but splits
        timing into effective-matrix assembly, RHS build, linear solve, and
        state update. It is intentionally used only when rollout profiling is
        enabled because synchronized timing can slow training.
        """
        if u_t.ndim == 1:
            return self._newmark_step_single_fast_timed(
                u_t=u_t,
                v_t=v_t,
                a_t=a_t,
                F_t1=F_t1,
                theta_t=theta_t,
                time_fn=time_fn,
            )

        B = int(u_t.shape[0])
        if theta_t is None:
            theta_batch = [None for _ in range(B)]
        elif theta_t.ndim == 1:
            theta_batch = [theta_t for _ in range(B)]
        else:
            theta_batch = [theta_t[i] for i in range(B)]

        timing = {
            "newmark_assemble_seconds": 0.0,
            "newmark_rhs_seconds": 0.0,
            "newmark_solve_seconds": 0.0,
            "newmark_update_seconds": 0.0,
        }

        if B == 1:
            u1, v1, a1, step_timing = self._newmark_step_single_fast_timed(
                u_t=u_t[0],
                v_t=v_t[0],
                a_t=a_t[0],
                F_t1=F_t1[0],
                theta_t=theta_batch[0],
                time_fn=time_fn,
            )
            for key, value in step_timing.items():
                timing[key] += float(value)
            return u1.unsqueeze(0), v1.unsqueeze(0), a1.unsqueeze(0), timing

        u_out = []
        v_out = []
        a_out = []
        for b in range(B):
            u1, v1, a1, step_timing = self._newmark_step_single_fast_timed(
                u_t=u_t[b],
                v_t=v_t[b],
                a_t=a_t[b],
                F_t1=F_t1[b],
                theta_t=theta_batch[b],
                time_fn=time_fn,
            )
            for key, value in step_timing.items():
                timing[key] += float(value)
            u_out.append(u1)
            v_out.append(v1)
            a_out.append(a1)

        return torch.stack(u_out, dim=0), torch.stack(v_out, dim=0), torch.stack(a_out, dim=0), timing

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
