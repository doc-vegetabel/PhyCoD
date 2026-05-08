from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.student.model import StudentBeamModel
from src.student.section_parameterization import (
    StudentSectionParameters,
    build_baseline_section_parameters,
    build_element_section_table,
)


def build_6dof_euler_bernoulli_element_matrices(
    *,
    L: float,
    m_avg: float,
    EA: float,
    EI_flap: float,
    EI_edge: float,
    GJ: float,
    J_rho: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    构造单个 3D Euler-Bernoulli 梁单元的 12x12 质量矩阵和刚度矩阵。

    DOF 排布与当前 src/student/fem_builder.py 完全一致：

        node i     : [ux, uy, uz, theta_x, theta_y, theta_z]
        node i + 1 : [ux, uy, uz, theta_x, theta_y, theta_z]

    局部 12DOF 索引：
        0  ux_i
        1  uy_i
        2  uz_i
        3  theta_x_i
        4  theta_y_i
        5  theta_z_i
        6  ux_j
        7  uy_j
        8  uz_j
        9  theta_x_j
        10 theta_y_j
        11 theta_z_j

    当前目标：
    - 严格复现 frozen baseline 的 fem_builder.py；
    - 暂不引入任何新的物理修正；
    - 后续 corrected FEM 的修正项会在 section parameters 或 element table 层面进入。
    """
    L = float(L)
    m_avg = float(m_avg)
    EA = float(EA)
    EI_flap = float(EI_flap)
    EI_edge = float(EI_edge)
    GJ = float(GJ)
    J_rho = float(J_rho)

    if L <= 0.0:
        raise ValueError(f"Element length L must be positive, got {L}.")

    Ke = np.zeros((12, 12), dtype=np.float64)
    Me = np.zeros((12, 12), dtype=np.float64)

    # ------------------------------------------------------------------
    # 1. Axial: uz, indices 2 and 8
    # ------------------------------------------------------------------
    Ke[2, 2] = Ke[8, 8] = EA / L
    Ke[2, 8] = Ke[8, 2] = -EA / L

    Me[2, 2] = Me[8, 8] = m_avg * L / 3.0
    Me[2, 8] = Me[8, 2] = m_avg * L / 6.0

    # ------------------------------------------------------------------
    # 2. Torsion: theta_z, indices 5 and 11
    # ------------------------------------------------------------------
    Ke[5, 5] = Ke[11, 11] = GJ / L
    Ke[5, 11] = Ke[11, 5] = -GJ / L

    Me[5, 5] = Me[11, 11] = J_rho * L / 3.0
    Me[5, 11] = Me[11, 5] = J_rho * L / 6.0

    # ------------------------------------------------------------------
    # 3. Flapwise: ux and theta_y, indices 0, 4, 6, 10
    #    与旧 fem_builder.py 一致：EI_flap 对应 ux 弯曲通道。
    # ------------------------------------------------------------------
    kf = EI_flap / L**3

    Ke[0, 0] = 12.0 * kf
    Ke[0, 4] = 6.0 * kf * L
    Ke[0, 6] = -12.0 * kf
    Ke[0, 10] = 6.0 * kf * L

    Ke[4, 0] = 6.0 * kf * L
    Ke[4, 4] = 4.0 * kf * L**2
    Ke[4, 6] = -6.0 * kf * L
    Ke[4, 10] = 2.0 * kf * L**2

    Ke[6, 0] = -12.0 * kf
    Ke[6, 4] = -6.0 * kf * L
    Ke[6, 6] = 12.0 * kf
    Ke[6, 10] = -6.0 * kf * L

    Ke[10, 0] = 6.0 * kf * L
    Ke[10, 4] = 2.0 * kf * L**2
    Ke[10, 6] = -6.0 * kf * L
    Ke[10, 10] = 4.0 * kf * L**2

    mf = m_avg * L / 420.0

    Me[0, 0] = 156.0 * mf
    Me[0, 4] = 22.0 * L * mf
    Me[0, 6] = 54.0 * mf
    Me[0, 10] = -13.0 * L * mf

    Me[4, 0] = 22.0 * L * mf
    Me[4, 4] = 4.0 * L**2 * mf
    Me[4, 6] = 13.0 * L * mf
    Me[4, 10] = -3.0 * L**2 * mf

    Me[6, 0] = 54.0 * mf
    Me[6, 4] = 13.0 * L * mf
    Me[6, 6] = 156.0 * mf
    Me[6, 10] = -22.0 * L * mf

    Me[10, 0] = -13.0 * L * mf
    Me[10, 4] = -3.0 * L**2 * mf
    Me[10, 6] = -22.0 * L * mf
    Me[10, 10] = 4.0 * L**2 * mf

    # ------------------------------------------------------------------
    # 4. Edgewise: uy and theta_x, indices 1, 3, 7, 9
    #    与旧 fem_builder.py 一致：交叉项符号与 flapwise 通道相反。
    # ------------------------------------------------------------------
    ke = EI_edge / L**3

    Ke[1, 1] = 12.0 * ke
    Ke[1, 3] = -6.0 * ke * L
    Ke[1, 7] = -12.0 * ke
    Ke[1, 9] = -6.0 * ke * L

    Ke[3, 1] = -6.0 * ke * L
    Ke[3, 3] = 4.0 * ke * L**2
    Ke[3, 7] = 6.0 * ke * L
    Ke[3, 9] = 2.0 * ke * L**2

    Ke[7, 1] = -12.0 * ke
    Ke[7, 3] = 6.0 * ke * L
    Ke[7, 7] = 12.0 * ke
    Ke[7, 9] = 6.0 * ke * L

    Ke[9, 1] = -6.0 * ke * L
    Ke[9, 3] = 2.0 * ke * L**2
    Ke[9, 7] = 6.0 * ke * L
    Ke[9, 9] = 4.0 * ke * L**2

    me = m_avg * L / 420.0

    Me[1, 1] = 156.0 * me
    Me[1, 3] = -22.0 * L * me
    Me[1, 7] = 54.0 * me
    Me[1, 9] = 13.0 * L * me

    Me[3, 1] = -22.0 * L * me
    Me[3, 3] = 4.0 * L**2 * me
    Me[3, 7] = -13.0 * L * me
    Me[3, 9] = -3.0 * L**2 * me

    Me[7, 1] = 54.0 * me
    Me[7, 3] = -13.0 * L * me
    Me[7, 7] = 156.0 * me
    Me[7, 9] = 22.0 * L * me

    Me[9, 1] = 13.0 * L * me
    Me[9, 3] = -3.0 * L**2 * me
    Me[9, 7] = 22.0 * L * me
    Me[9, 9] = 4.0 * L**2 * me

    return Me, Ke


def assemble_global_matrices_from_element_table(
    *,
    n_nodes: int,
    element_table: Dict[str, np.ndarray],
    alpha_flap: float = 1.0,
    alpha_edge: float = 1.0,
    alpha_torsion: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据 element section table 装配 full 6DOF 全局 M/K。

    当前 alpha 语义与旧 fem_builder.py 一致：
    - alpha_flap   乘到 EI_flap
    - alpha_edge   乘到 EI_edge
    - alpha_torsion 乘到 GJ
    """
    n_nodes = int(n_nodes)
    if n_nodes < 2:
        raise ValueError(f"n_nodes must be >= 2, got {n_nodes}.")

    n_dofs = 6 * n_nodes
    M_global = np.zeros((n_dofs, n_dofs), dtype=np.float64)
    K_global = np.zeros((n_dofs, n_dofs), dtype=np.float64)

    n_elements = int(np.asarray(element_table["L_m"]).size)
    if n_elements != n_nodes - 1:
        raise ValueError(
            f"element count mismatch: expected {n_nodes - 1}, got {n_elements}."
        )

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

        Me, Ke = build_6dof_euler_bernoulli_element_matrices(
            L=L,
            m_avg=m_avg,
            EA=EA,
            EI_flap=EI_flap,
            EI_edge=EI_edge,
            GJ=GJ,
            J_rho=J_rho,
        )

        idx = 6 * i
        M_global[idx : idx + 12, idx : idx + 12] += Me
        K_global[idx : idx + 12, idx : idx + 12] += Ke

    return M_global, K_global


def apply_cantilever_root_boundary_6dof(
    M_global: np.ndarray,
    K_global: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    施加与旧 fem_builder.py 一致的悬臂梁根部固定边界条件：

        root node 的 6 个自由度全部固定
        M_free = M_global[6:, 6:]
        K_free = K_global[6:, 6:]
    """
    M_global = np.asarray(M_global, dtype=np.float64)
    K_global = np.asarray(K_global, dtype=np.float64)

    if M_global.ndim != 2 or K_global.ndim != 2:
        raise ValueError("M_global and K_global must be 2D matrices.")
    if M_global.shape != K_global.shape:
        raise ValueError(
            f"M_global/K_global shape mismatch: {M_global.shape} vs {K_global.shape}."
        )
    if M_global.shape[0] != M_global.shape[1]:
        raise ValueError(f"Global matrices must be square, got {M_global.shape}.")
    if M_global.shape[0] < 12:
        raise ValueError("Global matrices are too small for a 6DOF cantilever beam.")

    return M_global[6:, 6:].copy(), K_global[6:, 6:].copy()


def build_corrected_fem_matrices_6dof(
    model: StudentBeamModel,
    section_params: Optional[StudentSectionParameters] = None,
    *,
    alpha_flap: float = 1.0,
    alpha_edge: float = 1.0,
    alpha_torsion: float = 1.0,
    return_full: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    新的 corrected FEM builder 入口。

    当前阶段：
    - 不引入真实修正；
    - 通过 section_parameterization.py 提供的参数表装配 M/K；
    - 在 section_params=None 时，使用 build_baseline_section_parameters(model)，
      应严格复现 build_fem_matrices_6dof(...)。

    参数：
        model:
            StudentBeamModel。

        section_params:
            截面参数集合。None 时使用 frozen baseline compatible 参数。

        alpha_flap / alpha_edge / alpha_torsion:
            与旧 fem_builder.py 一致的刚度倍率。

        return_full:
            False:
                返回 M_free, K_free。

            True:
                返回 M_free, K_free, debug_info。
    """
    model.validate()

    if section_params is None:
        section_params = build_baseline_section_parameters(model)

    element_table = build_element_section_table(model, section_params)

    M_global, K_global = assemble_global_matrices_from_element_table(
        n_nodes=model.n_stations,
        element_table=element_table,
        alpha_flap=alpha_flap,
        alpha_edge=alpha_edge,
        alpha_torsion=alpha_torsion,
    )

    M_free, K_free = apply_cantilever_root_boundary_6dof(M_global, K_global)

    if not return_full:
        return M_free, K_free

    debug_info: Dict[str, Any] = {
        "builder": "build_corrected_fem_matrices_6dof",
        "n_nodes": int(model.n_stations),
        "n_free_nodes": int(model.n_stations - 1),
        "n_dofs_global": int(M_global.shape[0]),
        "n_dofs_free": int(M_free.shape[0]),
        "alpha_flap": float(alpha_flap),
        "alpha_edge": float(alpha_edge),
        "alpha_torsion": float(alpha_torsion),
        "section_params_source": section_params.source,
        "element_table": element_table,
        "M_global": M_global,
        "K_global": K_global,
    }

    return M_free, K_free, debug_info


def summarize_fem_matrices(
    M: np.ndarray,
    K: np.ndarray,
) -> Dict[str, Any]:
    """
    生成 M/K 的简单数值摘要，便于测试和日志记录。
    """
    M = np.asarray(M, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)

    def _matrix_summary(A: np.ndarray) -> Dict[str, float]:
        return {
            "shape_0": int(A.shape[0]),
            "shape_1": int(A.shape[1]),
            "min": float(np.min(A)),
            "max": float(np.max(A)),
            "mean": float(np.mean(A)),
            "fro_norm": float(np.linalg.norm(A, ord="fro")),
            "symmetry_max_abs": float(np.max(np.abs(A - A.T))),
        }

    return {
        "M": _matrix_summary(M),
        "K": _matrix_summary(K),
    }