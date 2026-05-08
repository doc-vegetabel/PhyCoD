from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class FullOrderCorrectedCoreTorchConfig:
    """
    Full-order corrected student core 配置。

    当前阶段目标：
    - 只复刻 direct student 的 full-order Newmark-beta 推进；
    - 所有 correction 接口先保留，但默认返回 0；
    - theta_full=None 或 theta_full=0 时，严格退回 direct student baseline。

    注意：
    - 这里的状态变量是 full-order 自由 DOF:
        u, v, a, F : shape = (..., n_dofs_full)
    - 不使用 q / qdot / qddot；
    - 不使用 ModalAdapter；
    - 不使用 reduced matrices。
    """
    gamma: float = 0.5
    beta: float = 0.25
    dtype: torch.dtype = torch.float64
    linear_solve_mode: str = "solve"  # "solve" or "inverse"


class FullOrderCorrectedCoreTorch(nn.Module):
    """
    Full-order corrected student core.

    当前第一版动力学形式：

        M u_ddot + C u_dot + K u = F

    Newmark-beta 单步推进与 src/student/dynamic_solver.py 中的
    NewmarkBetaSolver.solve(...) 对齐：

        K_eff = K + a1 M + a2 C

        F_eff = F_{t+1}
              + M @ (a1 u_t + a3 v_t + a4 a_t)
              + C @ (a2 u_t + a5 v_t + a6 a_t)

        u_{t+1} = K_eff^{-1} F_eff
        a_{t+1} = a1 (u_{t+1} - u_t) - a3 v_t - a4 a_t
        v_{t+1} = v_t + dt * ((1-gamma) a_t + gamma a_{t+1})

    预留的 correction 接口：

        build_delta_M(...)
        build_delta_C(...)
        build_delta_K_mat(...)
        build_delta_K_geo(...)
        build_force_correction(...)

    当前这些接口全部返回 0。
    后续阶段 3 再逐步加入物理修正项。
    """

    def __init__(
        self,
        M: np.ndarray | torch.Tensor,
        K: np.ndarray | torch.Tensor,
        C: Optional[np.ndarray | torch.Tensor] = None,
        dt: float = 0.01,
        config: Optional[FullOrderCorrectedCoreTorchConfig] = None,
    ):
        super().__init__()

        self.config = config or FullOrderCorrectedCoreTorchConfig()
        self.dt = float(dt)
        self.gamma = float(self.config.gamma)
        self.beta = float(self.config.beta)
        self.dtype = self.config.dtype

        if self.dt <= 0.0:
            raise ValueError(f"dt must be positive, got {self.dt}.")

        if self.config.linear_solve_mode not in {"solve", "inverse"}:
            raise ValueError(
                "linear_solve_mode must be 'solve' or 'inverse', "
                f"got {self.config.linear_solve_mode!r}."
            )

        M_t = self._as_matrix_tensor(M, name="M")
        K_t = self._as_matrix_tensor(K, name="K")

        if C is None:
            C_t = torch.zeros_like(M_t)
        else:
            C_t = self._as_matrix_tensor(C, name="C")

        if M_t.shape != K_t.shape or M_t.shape != C_t.shape:
            raise ValueError(
                f"M/K/C shape mismatch: M={tuple(M_t.shape)}, "
                f"K={tuple(K_t.shape)}, C={tuple(C_t.shape)}"
            )

        if M_t.shape[0] != M_t.shape[1]:
            raise ValueError(f"M must be square, got {tuple(M_t.shape)}.")

        self.n_dofs_full = int(M_t.shape[0])

        self.register_buffer("M_base", M_t)
        self.register_buffer("K_base", K_t)
        self.register_buffer("C_base", C_t)

        # Newmark coefficients; must match NewmarkBetaSolver.
        self.a1 = 1.0 / (self.beta * self.dt ** 2)
        self.a2 = self.gamma / (self.beta * self.dt)
        self.a3 = 1.0 / (self.beta * self.dt)
        self.a4 = 1.0 / (2.0 * self.beta) - 1.0
        self.a5 = self.gamma / self.beta - 1.0
        self.a6 = (self.dt / 2.0) * (self.gamma / self.beta - 2.0)

        K_eff_base = self.K_base + self.a1 * self.M_base + self.a2 * self.C_base
        self.register_buffer("K_eff_base", K_eff_base)

        if self.config.linear_solve_mode == "inverse":
            self.register_buffer("K_eff_base_inv", torch.linalg.inv(K_eff_base))
            self.register_buffer("M_base_inv", torch.linalg.inv(self.M_base))
        else:
            self.register_buffer("K_eff_base_inv", torch.empty(0, dtype=self.dtype))
            self.register_buffer("M_base_inv", torch.empty(0, dtype=self.dtype))

    def _as_matrix_tensor(
        self,
        arr: np.ndarray | torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        if torch.is_tensor(arr):
            out = arr.detach().clone().to(dtype=self.dtype)
        else:
            out = torch.as_tensor(np.asarray(arr), dtype=self.dtype)

        if out.ndim != 2:
            raise ValueError(f"{name} must be a 2D matrix, got shape={tuple(out.shape)}.")
        return out

    def _check_full_vector(self, x: torch.Tensor, name: str) -> None:
        if x.shape[-1] != self.n_dofs_full:
            raise ValueError(
                f"{name} last dim mismatch: expected {self.n_dofs_full}, "
                f"got {x.shape[-1]}."
            )

    def _to_batched_vector(
        self,
        x: np.ndarray | torch.Tensor,
        name: str,
    ) -> Tuple[torch.Tensor, bool]:
        """
        返回:
            x_batched: shape = (B, D)
            was_unbatched: 原始输入是否为 (D,)
        """
        if torch.is_tensor(x):
            out = x.to(device=self.M_base.device, dtype=self.dtype)
        else:
            out = torch.as_tensor(x, device=self.M_base.device, dtype=self.dtype)

        if out.ndim == 1:
            self._check_full_vector(out, name)
            return out.unsqueeze(0), True

        if out.ndim == 2:
            self._check_full_vector(out, name)
            return out, False

        raise ValueError(f"{name} must have shape (D,) or (B,D), got {tuple(out.shape)}.")

    def _to_batched_force_history(
        self,
        F_time: np.ndarray | torch.Tensor,
    ) -> Tuple[torch.Tensor, bool]:
        """
        返回:
            F_batched: shape = (B, T, D)
            was_unbatched: 原始输入是否为 (T, D)
        """
        if torch.is_tensor(F_time):
            out = F_time.to(device=self.M_base.device, dtype=self.dtype)
        else:
            out = torch.as_tensor(F_time, device=self.M_base.device, dtype=self.dtype)

        if out.ndim == 2:
            if out.shape[1] != self.n_dofs_full:
                raise ValueError(
                    f"F_time second dim mismatch: expected {self.n_dofs_full}, "
                    f"got {out.shape[1]}."
                )
            return out.unsqueeze(0), True

        if out.ndim == 3:
            if out.shape[2] != self.n_dofs_full:
                raise ValueError(
                    f"F_time last dim mismatch: expected {self.n_dofs_full}, "
                    f"got {out.shape[2]}."
                )
            return out, False

        raise ValueError(
            f"F_time must have shape (T,D) or (B,T,D), got {tuple(out.shape)}."
        )

    def _expand_base_matrix(self, B: int, mat: torch.Tensor) -> torch.Tensor:
        return mat.unsqueeze(0).expand(B, -1, -1)

    def _zero_matrix_batch(self, B: int) -> torch.Tensor:
        return torch.zeros(
            B,
            self.n_dofs_full,
            self.n_dofs_full,
            dtype=self.dtype,
            device=self.M_base.device,
        )

    def _zero_vector_batch(self, B: int) -> torch.Tensor:
        return torch.zeros(
            B,
            self.n_dofs_full,
            dtype=self.dtype,
            device=self.M_base.device,
        )

    def _theta_dict_value(
            self,
            theta_full: Optional[Any],
            key: str,
    ) -> Optional[Any]:
        """
        当前阶段的显式修正接口。

        约定：
        - theta_full=None：无修正，严格回退 baseline
        - theta_full=dict：允许显式传入少量可控修正项
        - 其他类型暂不支持，避免未来网络输出误接进来造成语义混乱
        """
        if theta_full is None:
            return None

        if not isinstance(theta_full, dict):
            raise TypeError(
                "当前阶段 theta_full 只支持 None 或 dict。"
                "后续网络输出物理参数场时，再单独定义正式数据结构。"
            )

        return theta_full.get(key, None)

    def _as_batched_theta_vector(
            self,
            value: np.ndarray | torch.Tensor,
            *,
            B: int,
            name: str,
    ) -> torch.Tensor:
        """
        将 theta_full 中的向量修正项整理成 shape=(B, D)。

        支持：
        - (D,)
        - (1, D)
        - (B, D)
        """
        if torch.is_tensor(value):
            out = value.to(device=self.M_base.device, dtype=self.dtype)
        else:
            out = torch.as_tensor(value, device=self.M_base.device, dtype=self.dtype)

        if out.ndim == 1:
            if out.shape[0] != self.n_dofs_full:
                raise ValueError(
                    f"{name} length mismatch: expected {self.n_dofs_full}, "
                    f"got {out.shape[0]}."
                )
            return out.unsqueeze(0).expand(B, -1)

        if out.ndim == 2:
            if out.shape[1] != self.n_dofs_full:
                raise ValueError(
                    f"{name} second dim mismatch: expected {self.n_dofs_full}, "
                    f"got {out.shape[1]}."
                )

            if out.shape[0] == B:
                return out

            if out.shape[0] == 1 and B > 1:
                return out.expand(B, -1)

            raise ValueError(
                f"{name} batch mismatch: expected batch {B}, got {out.shape[0]}."
            )

        raise ValueError(
            f"{name} must have shape (D,) or (B,D), got {tuple(out.shape)}."
        )

    def _as_batched_theta_matrix(
            self,
            value: np.ndarray | torch.Tensor,
            *,
            B: int,
            name: str,
    ) -> torch.Tensor:
        """
        将 theta_full 中的矩阵修正项整理成 shape=(B, D, D)。

        支持：
        - (D, D)
        - (1, D, D)
        - (B, D, D)
        """
        if torch.is_tensor(value):
            out = value.to(device=self.M_base.device, dtype=self.dtype)
        else:
            out = torch.as_tensor(value, device=self.M_base.device, dtype=self.dtype)

        if out.ndim == 2:
            if out.shape != (self.n_dofs_full, self.n_dofs_full):
                raise ValueError(
                    f"{name} shape mismatch: expected "
                    f"({self.n_dofs_full}, {self.n_dofs_full}), got {tuple(out.shape)}."
                )
            return out.unsqueeze(0).expand(B, -1, -1)

        if out.ndim == 3:
            if out.shape[1:] != (self.n_dofs_full, self.n_dofs_full):
                raise ValueError(
                    f"{name} matrix shape mismatch: expected trailing shape "
                    f"({self.n_dofs_full}, {self.n_dofs_full}), got {tuple(out.shape[1:])}."
                )

            if out.shape[0] == B:
                return out

            if out.shape[0] == 1 and B > 1:
                return out.expand(B, -1, -1)

            raise ValueError(
                f"{name} batch mismatch: expected batch {B}, got {out.shape[0]}."
            )

        raise ValueError(
            f"{name} must have shape (D,D) or (B,D,D), got {tuple(out.shape)}."
        )

    def _as_batched_diag_matrix(
            self,
            value: np.ndarray | torch.Tensor,
            *,
            B: int,
            name: str,
    ) -> torch.Tensor:
        """
        将向量修正项转成 batch diagonal matrix。

        输入：
            value: shape=(D,) or (B,D)

        输出：
            Delta: shape=(B,D,D)
        """
        vec = self._as_batched_theta_vector(value, B=B, name=name)
        return torch.diag_embed(vec)


    def _solve_linear(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
        *,
        use_nominal_inverse_if_possible: bool = False,
    ) -> torch.Tensor:
        """
        A: shape = (B, D, D)
        b: shape = (B, D)

        linear_solve_mode:
        - "solve":   torch.linalg.solve(A, b)
        - "inverse": inv(A) @ b

        use_nominal_inverse_if_possible=True 时，若 A 正是 base K_eff 或 M，
        可走预存逆矩阵，主要用于尽量贴近当前 numpy baseline 的写法。
        """
        B = b.shape[0]

        if self.config.linear_solve_mode == "inverse":
            if use_nominal_inverse_if_possible and B >= 1:
                # 调用处负责传入具体 inverse；这里保留统一接口。
                pass

            A_inv = torch.linalg.inv(A)
            return torch.matmul(A_inv, b.unsqueeze(-1)).squeeze(-1)

        return torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)

    # -------------------------------------------------------------------------
    # Correction builders.
    # 当前阶段全部返回 0。
    # 后续阶段可以在这些函数里逐步加入物理修正。
    # -------------------------------------------------------------------------
    def build_delta_M(
        self,
        B: int,
        theta_full: Optional[Any] = None,
    ) -> torch.Tensor:
        _ = theta_full
        return self._zero_matrix_batch(B)

    def build_delta_C(
        self,
        B: int,
        theta_full: Optional[Any] = None,
    ) -> torch.Tensor:
        _ = theta_full
        return self._zero_matrix_batch(B)

    def build_delta_K_mat(
            self,
            B: int,
            theta_full: Optional[Any] = None,
    ) -> torch.Tensor:
        """
        构造材料/线性刚度修正 DeltaK_mat。

        当前阶段只提供显式 sanity-check 接口，不接网络：

        1) theta_full=None
           -> 返回 0，严格回退 direct student baseline

        2) theta_full={"delta_K_mat": array}
           -> 直接传入完整 DeltaK 矩阵
              shape = (D,D) or (B,D,D)

        3) theta_full={"delta_K_diag": array}
           -> 传入绝对量纲的对角刚度修正
              shape = (D,) or (B,D)

        4) theta_full={"delta_K_relative_diag": array}
           -> 传入相对 K_base 对角线的比例修正
              DeltaK_diag = relative_diag * diag(K_base)
              shape = (D,) or (B,D)

        注意：
        - 这是阶段 1 的接口可用性验证，不代表最终物理参数化形式。
        - 后续阶段 3 应替换为更物理的截面参数修正或 corrected FEM builder。
        """
        delta_K_mat = self._theta_dict_value(theta_full, "delta_K_mat")
        if delta_K_mat is not None:
            return self._as_batched_theta_matrix(
                delta_K_mat,
                B=B,
                name="theta_full['delta_K_mat']",
            )

        delta_K_diag = self._theta_dict_value(theta_full, "delta_K_diag")
        if delta_K_diag is not None:
            return self._as_batched_diag_matrix(
                delta_K_diag,
                B=B,
                name="theta_full['delta_K_diag']",
            )

        delta_K_relative_diag = self._theta_dict_value(
            theta_full,
            "delta_K_relative_diag",
        )
        if delta_K_relative_diag is not None:
            rel_vec = self._as_batched_theta_vector(
                delta_K_relative_diag,
                B=B,
                name="theta_full['delta_K_relative_diag']",
            )
            base_diag = torch.diag(self.K_base).unsqueeze(0)
            delta_diag = rel_vec * base_diag
            return torch.diag_embed(delta_diag)

        return self._zero_matrix_batch(B)

    def build_delta_K_geo(
        self,
        u_t: torch.Tensor,
        v_t: Optional[torch.Tensor] = None,
        theta_full: Optional[Any] = None,
    ) -> torch.Tensor:
        _ = v_t, theta_full
        B = int(u_t.shape[0])
        return self._zero_matrix_batch(B)

    def build_force_correction(
        self,
        u_t: torch.Tensor,
        v_t: torch.Tensor,
        a_t: torch.Tensor,
        F_t1: torch.Tensor,
        theta_full: Optional[Any] = None,
    ) -> torch.Tensor:
        _ = u_t, v_t, a_t, F_t1, theta_full
        B = int(F_t1.shape[0])
        return self._zero_vector_batch(B)

    def build_corrected_matrices(
        self,
        u_t: torch.Tensor,
        v_t: Optional[torch.Tensor] = None,
        theta_full: Optional[Any] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        构造当前步使用的 corrected M/C/K。

        当前零修正版：
            M_corr = M_base
            C_corr = C_base
            K_corr = K_base

        返回均为 batch matrix:
            shape = (B, D, D)
        """
        B = int(u_t.shape[0])

        delta_M = self.build_delta_M(B=B, theta_full=theta_full)
        delta_C = self.build_delta_C(B=B, theta_full=theta_full)
        delta_K_mat = self.build_delta_K_mat(B=B, theta_full=theta_full)
        delta_K_geo = self.build_delta_K_geo(
            u_t=u_t,
            v_t=v_t,
            theta_full=theta_full,
        )

        M_corr = self._expand_base_matrix(B, self.M_base) + delta_M
        C_corr = self._expand_base_matrix(B, self.C_base) + delta_C
        K_corr = self._expand_base_matrix(B, self.K_base) + delta_K_mat + delta_K_geo

        return {
            "M_corr": M_corr,
            "C_corr": C_corr,
            "K_corr": K_corr,
            "delta_M": delta_M,
            "delta_C": delta_C,
            "delta_K_mat": delta_K_mat,
            "delta_K_geo": delta_K_geo,
        }

    # -------------------------------------------------------------------------
    # Dynamics.
    # -------------------------------------------------------------------------
    def compute_initial_acceleration(
        self,
        u0: np.ndarray | torch.Tensor,
        v0: np.ndarray | torch.Tensor,
        F0: np.ndarray | torch.Tensor,
        theta_full: Optional[Any] = None,
    ) -> torch.Tensor:
        """
        计算初始加速度：

            a0 = M^{-1} (F0 + F_corr0 - C v0 - K u0)

        当前 F_corr0 = 0。
        """
        u0_b, u_was_unbatched = self._to_batched_vector(u0, "u0")
        v0_b, _ = self._to_batched_vector(v0, "v0")
        F0_b, _ = self._to_batched_vector(F0, "F0")

        B = int(u0_b.shape[0])
        mats = self.build_corrected_matrices(
            u_t=u0_b,
            v_t=v0_b,
            theta_full=theta_full,
        )

        force_corr = self.build_force_correction(
            u_t=u0_b,
            v_t=v0_b,
            a_t=torch.zeros_like(u0_b),
            F_t1=F0_b,
            theta_full=theta_full,
        )

        rhs = (
            F0_b
            + force_corr
            - torch.matmul(mats["C_corr"], v0_b.unsqueeze(-1)).squeeze(-1)
            - torch.matmul(mats["K_corr"], u0_b.unsqueeze(-1)).squeeze(-1)
        )

        if self.config.linear_solve_mode == "inverse":
            M_inv = torch.linalg.inv(mats["M_corr"])
            a0 = torch.matmul(M_inv, rhs.unsqueeze(-1)).squeeze(-1)
        else:
            a0 = torch.linalg.solve(mats["M_corr"], rhs.unsqueeze(-1)).squeeze(-1)

        if u_was_unbatched:
            return a0.squeeze(0)
        return a0

    def step(
        self,
        u_t: np.ndarray | torch.Tensor,
        v_t: np.ndarray | torch.Tensor,
        a_t: np.ndarray | torch.Tensor,
        F_t1: np.ndarray | torch.Tensor,
        theta_full: Optional[Any] = None,
        return_debug: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        """
        单步 full-order Newmark 推进。

        输入:
            u_t, v_t, a_t : shape = (D,) or (B,D)
            F_t1          : shape = (D,) or (B,D)

        输出:
            u_t1, v_t1, a_t1，同输入 batch 形式一致。
        """
        u_b, was_unbatched = self._to_batched_vector(u_t, "u_t")
        v_b, _ = self._to_batched_vector(v_t, "v_t")
        a_b, _ = self._to_batched_vector(a_t, "a_t")
        F_b, _ = self._to_batched_vector(F_t1, "F_t1")

        mats = self.build_corrected_matrices(
            u_t=u_b,
            v_t=v_b,
            theta_full=theta_full,
        )

        M_corr = mats["M_corr"]
        C_corr = mats["C_corr"]
        K_corr = mats["K_corr"]

        force_corr = self.build_force_correction(
            u_t=u_b,
            v_t=v_b,
            a_t=a_b,
            F_t1=F_b,
            theta_full=theta_full,
        )

        K_eff = K_corr + self.a1 * M_corr + self.a2 * C_corr

        term_M = self.a1 * u_b + self.a3 * v_b + self.a4 * a_b
        term_C = self.a2 * u_b + self.a5 * v_b + self.a6 * a_b

        mass_term = torch.matmul(M_corr, term_M.unsqueeze(-1)).squeeze(-1)
        damp_term = torch.matmul(C_corr, term_C.unsqueeze(-1)).squeeze(-1)

        F_eff = F_b + force_corr + mass_term + damp_term

        if self.config.linear_solve_mode == "inverse":
            K_eff_inv = torch.linalg.inv(K_eff)
            u_t1 = torch.matmul(K_eff_inv, F_eff.unsqueeze(-1)).squeeze(-1)
        else:
            u_t1 = torch.linalg.solve(K_eff, F_eff.unsqueeze(-1)).squeeze(-1)

        a_t1 = self.a1 * (u_t1 - u_b) - self.a3 * v_b - self.a4 * a_b
        v_t1 = v_b + self.dt * ((1.0 - self.gamma) * a_b + self.gamma * a_t1)

        debug = {
            **mats,
            "K_eff": K_eff,
            "force_corr": force_corr,
            "F_eff": F_eff,
            "term_M": term_M,
            "term_C": term_C,
            "mass_term": mass_term,
            "damp_term": damp_term,
        }

        if was_unbatched:
            u_t1 = u_t1.squeeze(0)
            v_t1 = v_t1.squeeze(0)
            a_t1 = a_t1.squeeze(0)
            debug = {
                k: (v.squeeze(0) if torch.is_tensor(v) and v.shape[0] == 1 else v)
                for k, v in debug.items()
            }

        if return_debug:
            return u_t1, v_t1, a_t1, debug
        return u_t1, v_t1, a_t1

    def rollout(
        self,
        F_time: np.ndarray | torch.Tensor,
        u0: Optional[np.ndarray | torch.Tensor] = None,
        v0: Optional[np.ndarray | torch.Tensor] = None,
        theta_full: Optional[Any] = None,
        return_debug: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, Any],
    ]:
        """
        整段 full-order rollout。

        输入:
            F_time:
                shape = (T, D) or (B, T, D)

            u0 / v0:
                None, shape = (D,), or shape = (B,D)

        输出:
            若 F_time 为 (T,D):
                u, v, a shape = (T,D)
            若 F_time 为 (B,T,D):
                u, v, a shape = (B,T,D)
        """
        F_b, was_unbatched = self._to_batched_force_history(F_time)

        B, T, D = F_b.shape
        if T < 1:
            raise ValueError("F_time must contain at least one time step.")

        u = torch.zeros(B, T, D, dtype=self.dtype, device=self.M_base.device)
        v = torch.zeros(B, T, D, dtype=self.dtype, device=self.M_base.device)
        a = torch.zeros(B, T, D, dtype=self.dtype, device=self.M_base.device)

        if u0 is not None:
            u0_b, _ = self._to_batched_vector(u0, "u0")
            if u0_b.shape[0] == 1 and B > 1:
                u0_b = u0_b.expand(B, -1)
            if u0_b.shape[0] != B:
                raise ValueError(f"u0 batch mismatch: expected {B}, got {u0_b.shape[0]}.")
            u[:, 0, :] = u0_b

        if v0 is not None:
            v0_b, _ = self._to_batched_vector(v0, "v0")
            if v0_b.shape[0] == 1 and B > 1:
                v0_b = v0_b.expand(B, -1)
            if v0_b.shape[0] != B:
                raise ValueError(f"v0 batch mismatch: expected {B}, got {v0_b.shape[0]}.")
            v[:, 0, :] = v0_b

        a0 = self.compute_initial_acceleration(
            u0=u[:, 0, :],
            v0=v[:, 0, :],
            F0=F_b[:, 0, :],
            theta_full=theta_full,
        )
        a[:, 0, :] = a0

        step_debug = []

        for i in range(T - 1):
            if return_debug:
                u_next, v_next, a_next, dbg = self.step(
                    u_t=u[:, i, :],
                    v_t=v[:, i, :],
                    a_t=a[:, i, :],
                    F_t1=F_b[:, i + 1, :],
                    theta_full=theta_full,
                    return_debug=True,
                )
                step_debug.append(dbg)
            else:
                u_next, v_next, a_next = self.step(
                    u_t=u[:, i, :],
                    v_t=v[:, i, :],
                    a_t=a[:, i, :],
                    F_t1=F_b[:, i + 1, :],
                    theta_full=theta_full,
                    return_debug=False,
                )

            u[:, i + 1, :] = u_next
            v[:, i + 1, :] = v_next
            a[:, i + 1, :] = a_next

        debug_out: Dict[str, Any] = {
            "n_steps": T,
            "n_dofs_full": D,
            "dt": self.dt,
            "gamma": self.gamma,
            "beta": self.beta,
            "linear_solve_mode": self.config.linear_solve_mode,
        }
        if return_debug:
            debug_out["step_debug"] = step_debug

        if was_unbatched:
            u = u.squeeze(0)
            v = v.squeeze(0)
            a = a.squeeze(0)

        if return_debug:
            return u, v, a, debug_out
        return u, v, a

    def summary(self) -> Dict[str, Any]:
        return {
            "core_type": "FullOrderCorrectedCoreTorch",
            "n_dofs_full": self.n_dofs_full,
            "dt": self.dt,
            "gamma": self.gamma,
            "beta": self.beta,
            "dtype": str(self.dtype),
            "linear_solve_mode": self.config.linear_solve_mode,
            "has_modal_adapter": False,
            "uses_q_state": False,
            "zero_correction_expected_to_match_direct_student": True,
            "supports_explicit_delta_K_mat": True,
            "supports_explicit_delta_K_diag": True,
            "supports_explicit_delta_K_relative_diag": True,
        }