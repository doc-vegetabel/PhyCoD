import numpy as np
from scipy.linalg import eigh
from src.student.fem_builder import build_fem_matrices_6dof


class NewmarkBetaSolver:
    """
    底层通用结构动力学 Newmark-beta 时域求解器
    """

    def __init__(self, M: np.ndarray, K: np.ndarray, C: np.ndarray = None,
                 dt: float = 0.002, gamma: float = 0.5, beta: float = 0.25):
        self.M = M
        self.K = K
        self.n_dofs = M.shape[0]
        self.C = np.zeros_like(M) if C is None else C
        self.dt = dt
        self.gamma = gamma
        self.beta = beta

        self.a1 = 1.0 / (beta * dt ** 2)
        self.a2 = gamma / (beta * dt)
        self.a3 = 1.0 / (beta * dt)
        self.a4 = 1.0 / (2.0 * beta) - 1.0
        self.a5 = gamma / beta - 1.0
        self.a6 = (dt / 2.0) * (gamma / beta - 2.0)

        self.K_eff = self.K + self.a1 * self.M + self.a2 * self.C
        self.K_eff_inv = np.linalg.inv(self.K_eff)

    def solve(self, F_time: np.ndarray, u0: np.ndarray = None, v0: np.ndarray = None):
        n_steps = F_time.shape[0]
        u = np.zeros((n_steps, self.n_dofs))
        v = np.zeros((n_steps, self.n_dofs))
        a = np.zeros((n_steps, self.n_dofs))

        if u0 is not None: u[0] = u0
        if v0 is not None: v[0] = v0

        F0_net = F_time[0] - self.C @ v[0] - self.K @ u[0]
        a[0] = np.linalg.inv(self.M) @ F0_net

        for i in range(n_steps - 1):
            F_eff = F_time[i + 1] + \
                    self.M @ (self.a1 * u[i] + self.a3 * v[i] + self.a4 * a[i]) + \
                    self.C @ (self.a2 * u[i] + self.a5 * v[i] + self.a6 * a[i])
            u[i + 1] = self.K_eff_inv @ F_eff
            a[i + 1] = self.a1 * (u[i + 1] - u[i]) - self.a3 * v[i] - self.a4 * a[i]
            v[i + 1] = v[i] + self.dt * ((1.0 - self.gamma) * a[i] + self.gamma * a[i + 1])

        return u, v, a


class WindBladeDynamicSystem:
    """
    顶层风机叶片动力学系统，封装了 6-DOF 矩阵生成、频率分析与时域求解
    """

    def __init__(self, model, alpha_flap: float = 1.0, alpha_edge: float = 1.0, alpha_torsion: float = 1.0):
        self.model = model
        # 默认使用 6-DOF 三维空间矩阵
        self.M, self.K = build_fem_matrices_6dof(model, alpha_flap, alpha_edge, alpha_torsion)
        self.n_dofs = self.M.shape[0]

    def get_natural_frequencies(self, num_modes: int = 5) -> np.ndarray:
        """求解无阻尼固有频率 (Hz)"""
        eigenvalues, _ = eigh(self.K, self.M)
        valid = eigenvalues > 0
        freqs = np.sqrt(eigenvalues[valid]) / (2.0 * np.pi)
        return freqs[:num_modes]

    def solve_step_response(self, F_time: np.ndarray, dt: float,
                            zeta_structural: float = 0.0, ref_freq_hz: float = None):
        """执行带 Rayleigh 阻尼的时域积分"""
        if zeta_structural > 0.0:
            if ref_freq_hz is None:
                ref_freq_hz = self.get_natural_frequencies(1)[0]
            beta_damp = 2.0 * zeta_structural / (2.0 * np.pi * ref_freq_hz)
            C = beta_damp * self.K
        else:
            C = np.zeros_like(self.M)

        solver = NewmarkBetaSolver(self.M, self.K, C=C, dt=dt)
        return solver.solve(F_time)