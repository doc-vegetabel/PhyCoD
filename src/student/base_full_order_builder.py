from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.student.io import load_student_model_from_blade_master
from src.student.base_phi import build_base_phi_from_blade_csv, print_base_phi_summary
from src.student.coupled_fem_builder import build_coupled_fem_matrices_6dof_degrees


def selected_kappa_y_dof_indices(
    n_dofs: int,
    scale_mode: str = "y_bending",
) -> np.ndarray:
    """
    选择全局 kappa_y_scale 作用的 DOF。

    当前 full-order 自由节点 DOF 排列假定为：
        [ux, uy, uz, rx, ry, rz] per free node

    scale_mode:
        - "uy_only":
            只缩放 uy 平移自由度。
        - "y_bending":
            缩放 uy 和 rx，自由度选择方式与
            scripts/train_kappa_y_global_torch.py 中的 y_bending 模式保持一致。
    """
    if n_dofs % 6 != 0:
        raise ValueError(f"n_dofs must be divisible by 6, got {n_dofs}")

    n_nodes = n_dofs // 6
    uy = np.array([i * 6 + 1 for i in range(n_nodes)], dtype=np.int64)

    if scale_mode == "uy_only":
        return uy

    if scale_mode == "y_bending":
        rx = np.array([i * 6 + 3 for i in range(n_nodes)], dtype=np.int64)
        return np.unique(np.concatenate([uy, rx])).astype(np.int64)

    raise ValueError(
        f"Unsupported kappa_y scale_mode={scale_mode!r}. "
        "Expected one of: 'uy_only', 'y_bending'."
    )


def build_kappa_y_global_scale_vector(
    *,
    n_dofs: int,
    kappa_y_scale: float = 0.952,
    scale_mode: str = "y_bending",
) -> tuple[np.ndarray, np.ndarray]:
    """
    构造对称刚度缩放向量 s，用于：

        K_scaled = diag(s) @ K @ diag(s)

    对被选中的 y-bending 相关 DOF，有：

        s = sqrt(kappa_y_scale)

    因此 selected-selected 对角刚度项会被缩放为原来的 kappa_y_scale。
    """
    scale = float(kappa_y_scale)

    if not np.isfinite(scale):
        raise ValueError(f"kappa_y_scale must be finite, got {kappa_y_scale}")

    if scale <= 0.0:
        raise ValueError(f"kappa_y_scale must be positive, got {kappa_y_scale}")

    scaled_dof_idx = selected_kappa_y_dof_indices(
        n_dofs=n_dofs,
        scale_mode=scale_mode,
    )

    s = np.ones(int(n_dofs), dtype=np.float64)
    s[scaled_dof_idx] = np.sqrt(scale)

    return s, scaled_dof_idx


def apply_global_kappa_y_scale_to_k(
    K: np.ndarray,
    *,
    kappa_y_scale: float = 0.952,
    scale_mode: str = "y_bending",
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    将固定的全局 y-bending 刚度缩放接入 full-order K 矩阵。

    注意：
        这是 base student 的静态物理修正。
        它不是 Transformer 输出。
        它不是 theta_full 里的临时 delta_K。
        它也不是输出位移后修正。

    数学形式：

        K_base = S_y K_phi S_y

    其中 S_y 是对角矩阵，被选中的 y-bending 相关 DOF 取 sqrt(kappa_y_scale)。
    """
    K = np.asarray(K, dtype=np.float64)

    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be a square 2D matrix, got shape={K.shape}")

    s, scaled_dof_idx = build_kappa_y_global_scale_vector(
        n_dofs=K.shape[0],
        kappa_y_scale=float(kappa_y_scale),
        scale_mode=str(scale_mode),
    )

    K_scaled = (s[:, None] * K) * s[None, :]
    K_scaled = 0.5 * (K_scaled + K_scaled.T)

    info = {
        "enabled": True,
        "kappa_y_scale": float(kappa_y_scale),
        "scale_mode": str(scale_mode),
        "num_scaled_dofs": int(scaled_dof_idx.size),
        "scaled_dof_idx": scaled_dof_idx.astype(np.int64),
        "scale_vector": s,
    }

    return K_scaled, info


def build_base_student_full_order_mk(
    *,
    blade_csv: str | Path,
    model_name: str = "base_student_with_initial_twist_phi",
    alpha_flap: float = 1.0,
    alpha_edge: float = 1.0,
    alpha_torsion: float = 1.0,
    twist_column: str = "initial_twist_deg",
    phi_sign: float = -1.0,
    rotate_mass: bool = False,
    use_kappa_y_global_scale: bool = True,
    kappa_y_global_scale: float = 0.952,
    kappa_y_scale_mode: str = "y_bending",
    verbose: bool = True,
) -> dict[str, Any]:
    """
    构造当前正式 base student 的 full-order M/K 矩阵。

    当前 base 定义：

        Phi_base(s) = - initial_twist_deg(s)
        rotate_mass = False
        K_base = S_y(0.952) K_phi S_y(0.952)

    其中：

        K_phi:
            使用 Phi_base(s) 后装配得到的 coupled FEM 刚度矩阵。

        kappa_y_global_scale:
            当前固定为 0.952 的全局 y-bending 等效刚度缩放。

    注意：
        Phi_base 和 kappa_y_global_scale 都是当前 base student 的固定构型参数，
        后续 Transformer 只应该学习它们之后的动态物理残差。
    """
    model = load_student_model_from_blade_master(
        csv_path=str(blade_csv),
        model_name=model_name,
    )

    phi_info = build_base_phi_from_blade_csv(
        blade_csv=blade_csv,
        twist_column=twist_column,
        sign=phi_sign,
    )

    if verbose:
        print_base_phi_summary(
            phi_info,
            prefix="[Base Student Full-Order Builder Phi]",
        )

    phi_element_deg = np.asarray(phi_info["phi_element_deg"], dtype=np.float64)

    M, K_phi, fem_info = build_coupled_fem_matrices_6dof_degrees(
        model,
        section_params=None,
        phi_deg=phi_element_deg,
        alpha_flap=alpha_flap,
        alpha_edge=alpha_edge,
        alpha_torsion=alpha_torsion,
        rotate_mass=rotate_mass,
        return_full=True,
    )

    M = np.asarray(M, dtype=np.float64)
    K_phi = np.asarray(K_phi, dtype=np.float64)

    if bool(use_kappa_y_global_scale):
        K, kappa_y_info = apply_global_kappa_y_scale_to_k(
            K_phi,
            kappa_y_scale=float(kappa_y_global_scale),
            scale_mode=str(kappa_y_scale_mode),
        )
    else:
        K = K_phi.copy()
        kappa_y_info = {
            "enabled": False,
            "kappa_y_scale": 1.0,
            "scale_mode": str(kappa_y_scale_mode),
            "num_scaled_dofs": 0,
            "scaled_dof_idx": np.array([], dtype=np.int64),
            "scale_vector": np.ones(K_phi.shape[0], dtype=np.float64),
        }

    if verbose:
        print("[Base Student Full-Order Builder Kappa-Y]")
        print(f"  enabled = {bool(kappa_y_info['enabled'])}")
        print(f"  kappa_y_global_scale = {float(kappa_y_info['kappa_y_scale']):.8f}")
        print(f"  scale_mode = {kappa_y_info['scale_mode']}")
        print(f"  num_scaled_dofs = {int(kappa_y_info['num_scaled_dofs'])}")

    return {
        "model": model,
        "M": M,
        "K": K,
        "K_phi_unscaled": K_phi,
        "fem_info": fem_info,
        "phi_info": phi_info,
        "phi_element_deg": phi_element_deg,
        "kappa_y_info": kappa_y_info,
    }