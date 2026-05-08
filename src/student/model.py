from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def _trapz(x: List[float], y: List[float]) -> float:
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    if len(x) < 2:
        return 0.0

    total = 0.0
    for i in range(len(x) - 1):
        dx = x[i + 1] - x[i]
        total += 0.5 * dx * (y[i] + y[i + 1])
    return total


@dataclass
class StudentBeamModel:
    """
    阶段1的最小学生结构模型数据容器。

    当前先统一这些最基础字段：
    - span_m: 叶片/梁总长
    - eta: 归一化展向坐标, [0, 1]
    - mass_per_length_kgpm: 单位长度质量分布
    - flapwise_ei_nm2: flapwise 等效弯曲刚度分布

    额外字段 edgewise_ei_nm2 / torsional_gj_nm2 先预留，
    下一阶段做更完整结构模型时直接复用。
    """
    model_name: str
    span_m: float
    eta: List[float]
    mass_per_length_kgpm: List[float]
    flapwise_ei_nm2: List[float]
    edgewise_ei_nm2: Optional[List[float]] = None
    torsional_gj_nm2: Optional[List[float]] = None
    notes: str = ""

    def validate(self) -> None:
        if self.span_m <= 0.0:
            raise ValueError(f"span_m must be positive, got {self.span_m}.")

        n = len(self.eta)
        if n < 2:
            raise ValueError("At least two span stations are required.")

        if len(self.mass_per_length_kgpm) != n:
            raise ValueError("mass_per_length_kgpm length must match eta length.")

        if len(self.flapwise_ei_nm2) != n:
            raise ValueError("flapwise_ei_nm2 length must match eta length.")

        if self.edgewise_ei_nm2 is not None and len(self.edgewise_ei_nm2) != n:
            raise ValueError("edgewise_ei_nm2 length must match eta length.")

        if self.torsional_gj_nm2 is not None and len(self.torsional_gj_nm2) != n:
            raise ValueError("torsional_gj_nm2 length must match eta length.")

        if abs(self.eta[0] - 0.0) > 1e-9:
            raise ValueError(f"eta must start at 0.0, got eta[0]={self.eta[0]}.")

        if abs(self.eta[-1] - 1.0) > 1e-9:
            raise ValueError(f"eta must end at 1.0, got eta[-1]={self.eta[-1]}.")

        for i in range(n - 1):
            if self.eta[i + 1] <= self.eta[i]:
                raise ValueError("eta must be strictly increasing.")

        for value in self.mass_per_length_kgpm:
            if value < 0.0:
                raise ValueError("mass_per_length_kgpm must be non-negative.")

        for value in self.flapwise_ei_nm2:
            if value <= 0.0:
                raise ValueError("flapwise_ei_nm2 must be positive.")

        if self.edgewise_ei_nm2 is not None:
            for value in self.edgewise_ei_nm2:
                if value <= 0.0:
                    raise ValueError("edgewise_ei_nm2 must be positive.")

        if self.torsional_gj_nm2 is not None:
            for value in self.torsional_gj_nm2:
                if value <= 0.0:
                    raise ValueError("torsional_gj_nm2 must be positive.")

    @property
    def n_stations(self) -> int:
        return len(self.eta)

    @property
    def span_stations_m(self) -> List[float]:
        return [self.span_m * e for e in self.eta]

    @property
    def estimated_total_mass_kg(self) -> float:
        return _trapz(self.span_stations_m, self.mass_per_length_kgpm)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "span_m": self.span_m,
            "n_stations": self.n_stations,
            "eta": self.eta,
            "mass_per_length_kgpm": self.mass_per_length_kgpm,
            "flapwise_ei_nm2": self.flapwise_ei_nm2,
            "edgewise_ei_nm2": self.edgewise_ei_nm2,
            "torsional_gj_nm2": self.torsional_gj_nm2,
            "notes": self.notes,
            "estimated_total_mass_kg": self.estimated_total_mass_kg,
        }



    # === 在 StudentBeamModel 类的最下方修改这两个方法 ===
    def get_natural_frequencies(self, alpha_flap=1.01, alpha_edge=0.925, alpha_torsion=1.0, num_modes=5):
        """快捷获取叶片的 6-DOF 三维无阻尼固有频率 (Hz)"""
        # [修改点]：把 import 放在函数里面！
        from .dynamic_solver import WindBladeDynamicSystem

        dyn_sys = WindBladeDynamicSystem(self, alpha_flap, alpha_edge, alpha_torsion)
        return dyn_sys.get_natural_frequencies(num_modes)

    def solve_dynamics(self, F_time, dt, alpha_flap=1.01, alpha_edge=0.925, alpha_torsion=1.0, zeta_structural=0.015):
        """
        快捷执行 6-DOF 时域动力学响应积分
        """
        # [修改点]：把 import 放在函数里面！
        from .dynamic_solver import WindBladeDynamicSystem

        dyn_sys = WindBladeDynamicSystem(self, alpha_flap, alpha_edge, alpha_torsion)
        f1 = dyn_sys.get_natural_frequencies(1)[0]
        return dyn_sys.solve_step_response(F_time, dt, zeta_structural=zeta_structural, ref_freq_hz=f1)