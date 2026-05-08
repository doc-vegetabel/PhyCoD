from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.student.model import StudentBeamModel
from src.student.section_parameterization import (
    StudentSectionParameters,
    build_baseline_section_parameters,
    build_element_section_table,
)
from src.student.corrected_fem_builder import (
    build_6dof_euler_bernoulli_element_matrices,
    apply_cantilever_root_boundary_6dof,
)


def _as_phi_element_array(
    phi_rad: float | np.ndarray,
    *,
    n_stations: int,
    n_elements: int,
) -> np.ndarray:
    """
    将 phi_rad 整理成 element-level array。

    支持：
    - scalar: 所有单元统一 phi
    - shape=(n_stations,): 相邻 station 平均成 element phi
    - shape=(n_elements,): 直接作为 element phi

    单位：radian。
    """
    arr = np.asarray(phi_rad, dtype=np.float64)

    if arr.ndim == 0:
        return np.full(n_elements, float(arr), dtype=np.float64)

    arr = arr.reshape(-1)

    if arr.size == n_elements:
        out = arr.astype(np.float64, copy=True)
    elif arr.size == n_stations:
        out = 0.5 * (arr[:-1] + arr[1:])
    else:
        raise ValueError(
            f"phi_rad must be scalar, shape=({n_stations},), "
            f"or shape=({n_elements},), got shape={arr.shape}."
        )

    if not np.all(np.isfinite(out)):
        raise ValueError("phi_rad contains non-finite values.")

    return out


def make_principal_axis_rotation_transform_6dof(phi_rad: float) -> np.ndarray:
    """
    构造 12x12 单元 DOF 坐标转换矩阵 T。

    约定：
        q_local = T @ q_global
        K_global = T.T @ K_local @ T

    每个节点 DOF:
        [ux, uy, uz, theta_x, theta_y, theta_z]

    这里仅旋转 x/y 平移分量和 theta_x/theta_y 转角分量，
    uz 与 theta_z 保持不变。

    phi = 0 时，T 为单位阵。
    """
    phi = float(phi_rad)
    c = float(np.cos(phi))
    s = float(np.sin(phi))

    T = np.eye(12, dtype=np.float64)

    for base in (0, 6):
        # translations: [ux_local, uy_local]^T = R^T [ux_global, uy_global]^T
        T[base + 0, base + 0] = c
        T[base + 0, base + 1] = s
        T[base + 1, base + 0] = -s
        T[base + 1, base + 1] = c

        # rotations: [theta_x_local, theta_y_local]^T = R^T [theta_x_global, theta_y_global]^T
        T[base + 3, base + 3] = c
        T[base + 3, base + 4] = s
        T[base + 4, base + 3] = -s
        T[base + 4, base + 4] = c

    return T


def rotate_element_stiffness_by_principal_axis(
    Ke_local: np.ndarray,
    *,
    phi_rad: float,
) -> np.ndarray:
    """
    用截面主轴旋转角 phi 将局部未耦合弯曲刚度矩阵旋转到全局 x/y 坐标。

    phi = 0 时返回与 Ke_local 数值一致的矩阵。
    """
    Ke_local = np.asarray(Ke_local, dtype=np.float64)
    if Ke_local.shape != (12, 12):
        raise ValueError(f"Ke_local must have shape (12,12), got {Ke_local.shape}.")

    T = make_principal_axis_rotation_transform_6dof(phi_rad)
    Ke_global = T.T @ Ke_local @ T

    # 清理极小数值非对称
    Ke_global = 0.5 * (Ke_global + Ke_global.T)
    return Ke_global


def build_coupled_fem_matrices_6dof(
    model: StudentBeamModel,
    section_params: Optional[StudentSectionParameters] = None,
    *,
    phi_rad: float | np.ndarray = 0.0,
    alpha_flap: float = 1.0,
    alpha_edge: float = 1.0,
    alpha_torsion: float = 1.0,
    rotate_mass: bool = False,
    return_full: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    构造带 flap-edge bending coupling 的 6DOF FEM M/K。

    核心思想：
    - 当前 baseline 的 EI_flap / EI_edge 是局部主弯曲刚度；
    - 若截面主轴相对 student 全局 x/y 轴旋转 phi，则在全局坐标下会出现 x-y 弯曲耦合；
    - phi=0 时严格回退当前 corrected_fem_builder baseline。

    参数：
        phi_rad:
            scalar / station-level / element-level principal-axis rotation angle，单位 rad。

        rotate_mass:
            默认 False。当前只旋转刚度矩阵 K，不旋转质量矩阵 M。
            这样可以把本阶段目标聚焦在 bending stiffness coupling。
            后续如需惯性主轴耦合，可单独引入。
    """
    model.validate()

    if section_params is None:
        section_params = build_baseline_section_parameters(model)

    element_table = build_element_section_table(model, section_params)

    n_nodes = int(model.n_stations)
    n_elements = n_nodes - 1
    n_dofs = 6 * n_nodes

    phi_elem = _as_phi_element_array(
        phi_rad,
        n_stations=n_nodes,
        n_elements=n_elements,
    )

    M_global = np.zeros((n_dofs, n_dofs), dtype=np.float64)
    K_global = np.zeros((n_dofs, n_dofs), dtype=np.float64)

    for i in range(n_elements):
        L = float(element_table["L_m"][i])
        if L <= 0.0:
            continue

        m_avg = float(element_table["m_avg_kgpm"][i])
        EA = float(element_table["EA_avg_N"][i])
        EI_flap = float(element_table["EI_flap_avg_Nm2"][i]) * float(alpha_flap)
        EI_edge = float(element_table["EI_edge_avg_Nm2"][i]) * float(alpha_edge)
        GJ = float(element_table["GJ_avg_Nm2"][i]) * float(alpha_torsion)
        J_rho = float(element_table["J_rho_avg_kgm"][i])
        phi_i = float(phi_elem[i])

        Me_local, Ke_local = build_6dof_euler_bernoulli_element_matrices(
            L=L,
            m_avg=m_avg,
            EA=EA,
            EI_flap=EI_flap,
            EI_edge=EI_edge,
            GJ=GJ,
            J_rho=J_rho,
        )

        Ke = rotate_element_stiffness_by_principal_axis(
            Ke_local,
            phi_rad=phi_i,
        )

        if rotate_mass:
            T = make_principal_axis_rotation_transform_6dof(phi_i)
            Me = T.T @ Me_local @ T
            Me = 0.5 * (Me + Me.T)
        else:
            Me = Me_local

        idx = 6 * i
        M_global[idx : idx + 12, idx : idx + 12] += Me
        K_global[idx : idx + 12, idx : idx + 12] += Ke

    M_free, K_free = apply_cantilever_root_boundary_6dof(M_global, K_global)

    if not return_full:
        return M_free, K_free

    debug_info: Dict[str, Any] = {
        "builder": "build_coupled_fem_matrices_6dof",
        "n_nodes": n_nodes,
        "n_free_nodes": n_nodes - 1,
        "n_dofs_global": int(M_global.shape[0]),
        "n_dofs_free": int(M_free.shape[0]),
        "alpha_flap": float(alpha_flap),
        "alpha_edge": float(alpha_edge),
        "alpha_torsion": float(alpha_torsion),
        "rotate_mass": bool(rotate_mass),
        "section_params_source": section_params.source,
        "phi_element_rad": phi_elem,
        "phi_element_deg": np.rad2deg(phi_elem),
        "element_table": element_table,
        "M_global": M_global,
        "K_global": K_global,
    }

    return M_free, K_free, debug_info


def build_coupled_fem_matrices_6dof_degrees(
    model: StudentBeamModel,
    section_params: Optional[StudentSectionParameters] = None,
    *,
    phi_deg: float | np.ndarray = 0.0,
    alpha_flap: float = 1.0,
    alpha_edge: float = 1.0,
    alpha_torsion: float = 1.0,
    rotate_mass: bool = False,
    return_full: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    degree 版本入口，方便手动扫描。
    """
    phi_rad = np.deg2rad(np.asarray(phi_deg, dtype=np.float64))
    return build_coupled_fem_matrices_6dof(
        model,
        section_params=section_params,
        phi_rad=phi_rad,
        alpha_flap=alpha_flap,
        alpha_edge=alpha_edge,
        alpha_torsion=alpha_torsion,
        rotate_mass=rotate_mass,
        return_full=return_full,
    )


def bending_xy_coupling_norms_6dof(K_free: np.ndarray) -> Dict[str, float]:
    """
    计算 free-DOF 刚度矩阵中的 x/y bending coupling 子块范数。

    free node DOF:
        [ux, uy, uz, theta_x, theta_y, theta_z]

    x bending channel 近似取:
        ux, theta_y

    y bending channel 近似取:
        uy, theta_x
    """
    K = np.asarray(K_free, dtype=np.float64)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K_free must be square, got {K.shape}.")
    if K.shape[0] % 6 != 0:
        raise ValueError(f"K_free dimension must be divisible by 6, got {K.shape[0]}.")

    n_free_nodes = K.shape[0] // 6

    x_ids = []
    y_ids = []
    for i in range(n_free_nodes):
        base = 6 * i
        x_ids.extend([base + 0, base + 4])  # ux, theta_y
        y_ids.extend([base + 1, base + 3])  # uy, theta_x

    K_xy = K[np.ix_(x_ids, y_ids)]
    K_xx = K[np.ix_(x_ids, x_ids)]
    K_yy = K[np.ix_(y_ids, y_ids)]

    xy_fro = float(np.linalg.norm(K_xy, ord="fro"))
    xx_fro = float(np.linalg.norm(K_xx, ord="fro"))
    yy_fro = float(np.linalg.norm(K_yy, ord="fro"))

    return {
        "K_xy_fro": xy_fro,
        "K_xx_fro": xx_fro,
        "K_yy_fro": yy_fro,
        "K_xy_over_K_xx": xy_fro / max(xx_fro, 1e-30),
        "K_xy_over_K_yy": xy_fro / max(yy_fro, 1e-30),
        "K_symmetry_max_abs": float(np.max(np.abs(K - K.T))),
    }