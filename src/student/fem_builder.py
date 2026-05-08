import numpy as np
from typing import Tuple
from src.student.model import StudentBeamModel


def build_fem_matrices(model: StudentBeamModel, direction: str = "flapwise", alpha: float = 1.0) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    (保留用于兼容之前的单向 2-DOF 测试)
    每个节点 2 个自由度: [v (平移), theta (转角)]
    """
    span_stations = model.span_stations_m
    mass_dist = model.mass_per_length_kgpm

    if direction == "flapwise":
        ei_dist = model.flapwise_ei_nm2
    elif direction == "edgewise":
        if model.edgewise_ei_nm2 is None:
            raise ValueError("edgewise_ei_nm2 is not set.")
        ei_dist = model.edgewise_ei_nm2
    else:
        raise ValueError(f"Unknown direction: {direction}")

    n_nodes = model.n_stations
    n_dofs = 2 * n_nodes
    M_global = np.zeros((n_dofs, n_dofs))
    K_global = np.zeros((n_dofs, n_dofs))

    for i in range(n_nodes - 1):
        L = span_stations[i + 1] - span_stations[i]
        if L <= 0: continue

        m_avg = 0.5 * (mass_dist[i] + mass_dist[i + 1])
        ei_avg = 0.5 * (ei_dist[i] + ei_dist[i + 1]) * alpha

        coeff_m = (m_avg * L) / 420.0
        Me = coeff_m * np.array([
            [156.0, 22.0 * L, 54.0, -13.0 * L],
            [22.0 * L, 4.0 * L ** 2, 13.0 * L, -3.0 * L ** 2],
            [54.0, 13.0 * L, 156.0, -22.0 * L],
            [-13.0 * L, -3.0 * L ** 2, -22.0 * L, 4.0 * L ** 2]
        ])

        coeff_k = ei_avg / (L ** 3)
        Ke = coeff_k * np.array([
            [12.0, 6.0 * L, -12.0, 6.0 * L],
            [6.0 * L, 4.0 * L ** 2, -6.0 * L, 2.0 * L ** 2],
            [-12.0, -6.0 * L, 12.0, -6.0 * L],
            [6.0 * L, 2.0 * L ** 2, -6.0 * L, 4.0 * L ** 2]
        ])

        idx = 2 * i
        M_global[idx:idx + 4, idx:idx + 4] += Me
        K_global[idx:idx + 4, idx:idx + 4] += Ke

    M_free = M_global[2:, 2:]
    K_free = K_global[2:, 2:]
    return M_free, K_free


def build_fem_matrices_6dof(model: StudentBeamModel, alpha_flap: float = 1.0,
                            alpha_edge: float = 1.0, alpha_torsion: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    [全新升级] 3D Euler-Bernoulli 梁单元，为 Student 模型构建 6-DOF 全局质量与刚度矩阵。
    每个节点具有 6 个自由度: [ux, uy, uz, theta_x, theta_y, theta_z]
    - ux: Flapwise displacement
    - uy: Edgewise displacement
    - uz: Spanwise (Axial) displacement
    """
    span_stations = model.span_stations_m
    mass_dist = model.mass_per_length_kgpm

    ei_flap_dist = model.flapwise_ei_nm2
    ei_edge_dist = model.edgewise_ei_nm2 if model.edgewise_ei_nm2 is not None else model.flapwise_ei_nm2
    gj_dist = model.torsional_gj_nm2 if model.torsional_gj_nm2 is not None else [1e9] * len(span_stations)

    n_nodes = model.n_stations
    n_dofs = 6 * n_nodes

    M_global = np.zeros((n_dofs, n_dofs))
    K_global = np.zeros((n_dofs, n_dofs))

    for i in range(n_nodes - 1):
        L = span_stations[i + 1] - span_stations[i]
        if L <= 0: continue

        m_avg = 0.5 * (mass_dist[i] + mass_dist[i + 1])
        ei_y = 0.5 * (ei_flap_dist[i] + ei_flap_dist[i + 1]) * alpha_flap  # Flapwise (弯曲绕Y轴)
        ei_x = 0.5 * (ei_edge_dist[i] + ei_edge_dist[i + 1]) * alpha_edge  # Edgewise (弯曲绕X轴)
        gj = 0.5 * (gj_dist[i] + gj_dist[i + 1]) * alpha_torsion  # 扭转 (绕Z轴)

        # 占位参数: 极大的轴向刚度 EA 和 截面极惯性矩 J_rho 占位
        ea = 1e10
        j_rho = m_avg * 1.0

        Ke = np.zeros((12, 12))
        Me = np.zeros((12, 12))

        # --- 1. Axial (Z方向平移: 索引 2, 8) ---
        Ke[2, 2] = Ke[8, 8] = ea / L
        Ke[2, 8] = Ke[8, 2] = -ea / L
        Me[2, 2] = Me[8, 8] = m_avg * L / 3.0
        Me[2, 8] = Me[8, 2] = m_avg * L / 6.0

        # --- 2. Torsion (Z方向扭转: 索引 5, 11) ---
        Ke[5, 5] = Ke[11, 11] = gj / L
        Ke[5, 11] = Ke[11, 5] = -gj / L
        Me[5, 5] = Me[11, 11] = j_rho * L / 3.0
        Me[5, 11] = Me[11, 5] = j_rho * L / 6.0

        # --- 3. Flapwise (X方向平移 ux, 绕Y转角 theta_y: 索引 0, 4, 6, 10) ---
        kf = ei_y / L ** 3
        Ke[0, 0] = 12 * kf;
        Ke[0, 4] = 6 * kf * L;
        Ke[0, 6] = -12 * kf;
        Ke[0, 10] = 6 * kf * L
        Ke[4, 0] = 6 * kf * L;
        Ke[4, 4] = 4 * kf * L ** 2;
        Ke[4, 6] = -6 * kf * L;
        Ke[4, 10] = 2 * kf * L ** 2
        Ke[6, 0] = -12 * kf;
        Ke[6, 4] = -6 * kf * L;
        Ke[6, 6] = 12 * kf;
        Ke[6, 10] = -6 * kf * L
        Ke[10, 0] = 6 * kf * L;
        Ke[10, 4] = 2 * kf * L ** 2;
        Ke[10, 6] = -6 * kf * L;
        Ke[10, 10] = 4 * kf * L ** 2

        mf = m_avg * L / 420.0
        Me[0, 0] = 156 * mf;
        Me[0, 4] = 22 * L * mf;
        Me[0, 6] = 54 * mf;
        Me[0, 10] = -13 * L * mf
        Me[4, 0] = 22 * L * mf;
        Me[4, 4] = 4 * L ** 2 * mf;
        Me[4, 6] = 13 * L * mf;
        Me[4, 10] = -3 * L ** 2 * mf
        Me[6, 0] = 54 * mf;
        Me[6, 4] = 13 * L * mf;
        Me[6, 6] = 156 * mf;
        Me[6, 10] = -22 * L * mf
        Me[10, 0] = -13 * L * mf;
        Me[10, 4] = -3 * L ** 2 * mf;
        Me[10, 6] = -22 * L * mf;
        Me[10, 10] = 4 * L ** 2 * mf

        # --- 4. Edgewise (Y方向平移 uy, 绕X转角 theta_x: 索引 1, 3, 7, 9) ---
        # 注意：右手法则下，绕X轴正向旋转会导致Y方向负位移，因此交叉项符号与Flapwise相反
        ke = ei_x / L ** 3
        Ke[1, 1] = 12 * ke;
        Ke[1, 3] = -6 * ke * L;
        Ke[1, 7] = -12 * ke;
        Ke[1, 9] = -6 * ke * L
        Ke[3, 1] = -6 * ke * L;
        Ke[3, 3] = 4 * ke * L ** 2;
        Ke[3, 7] = 6 * ke * L;
        Ke[3, 9] = 2 * ke * L ** 2
        Ke[7, 1] = -12 * ke;
        Ke[7, 3] = 6 * ke * L;
        Ke[7, 7] = 12 * ke;
        Ke[7, 9] = 6 * ke * L
        Ke[9, 1] = -6 * ke * L;
        Ke[9, 3] = 2 * ke * L ** 2;
        Ke[9, 7] = 6 * ke * L;
        Ke[9, 9] = 4 * ke * L ** 2

        me = m_avg * L / 420.0
        Me[1, 1] = 156 * me;
        Me[1, 3] = -22 * L * me;
        Me[1, 7] = 54 * me;
        Me[1, 9] = 13 * L * me
        Me[3, 1] = -22 * L * me;
        Me[3, 3] = 4 * L ** 2 * me;
        Me[3, 7] = -13 * L * me;
        Me[3, 9] = -3 * L ** 2 * me
        Me[7, 1] = 54 * me;
        Me[7, 3] = -13 * L * me;
        Me[7, 7] = 156 * me;
        Me[7, 9] = 22 * L * me
        Me[9, 1] = 13 * L * me;
        Me[9, 3] = -3 * L ** 2 * me;
        Me[9, 7] = 22 * L * me;
        Me[9, 9] = 4 * L ** 2 * me

        # 组装到全局矩阵
        idx = 6 * i
        M_global[idx:idx + 12, idx:idx + 12] += Me
        K_global[idx:idx + 12, idx:idx + 12] += Ke

    # 施加悬臂梁边界条件：固定节点0的所有 6 个自由度
    M_free = M_global[6:, 6:]
    K_free = K_global[6:, 6:]

    return M_free, K_free