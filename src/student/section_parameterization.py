from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from src.student.model import StudentBeamModel


def _as_1d_float_array(values: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _validate_eta(eta: np.ndarray) -> None:
    if eta.ndim != 1:
        raise ValueError("eta must be a 1D array.")
    if eta.size < 2:
        raise ValueError("eta must contain at least two stations.")
    if abs(float(eta[0]) - 0.0) > 1e-9:
        raise ValueError(f"eta must start at 0.0, got {eta[0]}.")
    if abs(float(eta[-1]) - 1.0) > 1e-9:
        raise ValueError(f"eta must end at 1.0, got {eta[-1]}.")
    if np.any(np.diff(eta) <= 0.0):
        raise ValueError("eta must be strictly increasing.")
    if not np.all(np.isfinite(eta)):
        raise ValueError("eta contains non-finite values.")


@dataclass(frozen=True)
class SpanwiseSectionField:
    """
    一个沿展向 eta 定义的截面参数场。

    例如：
    - mass_per_length_kgpm
    - EA_N
    - EI_flap_Nm2
    - EI_edge_Nm2
    - GJ_Nm2
    - J_rho_kgm
    """

    name: str
    eta: np.ndarray
    values: np.ndarray
    unit: str = ""

    def __post_init__(self) -> None:
        eta = _as_1d_float_array(self.eta, name=f"{self.name}.eta")
        values = _as_1d_float_array(self.values, name=f"{self.name}.values")
        _validate_eta(eta)

        if eta.shape != values.shape:
            raise ValueError(
                f"{self.name}: eta and values must have the same shape, "
                f"got eta={eta.shape}, values={values.shape}."
            )

        object.__setattr__(self, "eta", eta.copy())
        object.__setattr__(self, "values", values.copy())

    @property
    def n_stations(self) -> int:
        return int(self.values.size)

    def element_average(self) -> np.ndarray:
        """
        返回每个相邻 station 之间的单元平均值。

        shape:
            (n_stations - 1,)
        """
        return 0.5 * (self.values[:-1] + self.values[1:])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "unit": self.unit,
            "eta": self.eta.tolist(),
            "values": self.values.tolist(),
            "n_stations": self.n_stations,
        }


@dataclass(frozen=True)
class StudentSectionParameters:
    """
    Student 梁模型的截面参数集合。

    当前阶段目标：
    - 复现 frozen baseline 中 fem_builder.py 的参数语义；
    - 先集中管理 m / EA / EI / GJ / J_rho；
    - 暂不改变 run_student_case(...) 的任何数值结果。

    当前 baseline 约定：
    - EA 使用 fem_builder.py 中的占位常数 1e10；
    - J_rho 使用 fem_builder.py 中的占位规则 j_rho = m_avg * 1.0；
      因此这里 station-level J_rho 取 mass_per_length * 1.0，
      element average 后正好复现 m_avg * 1.0。
    """

    eta: np.ndarray
    mass_per_length_kgpm: SpanwiseSectionField
    EA_N: SpanwiseSectionField
    EI_flap_Nm2: SpanwiseSectionField
    EI_edge_Nm2: SpanwiseSectionField
    GJ_Nm2: SpanwiseSectionField
    J_rho_kgm: SpanwiseSectionField
    source: str = "baseline"

    def __post_init__(self) -> None:
        eta = _as_1d_float_array(self.eta, name="StudentSectionParameters.eta")
        _validate_eta(eta)
        object.__setattr__(self, "eta", eta.copy())

        fields = [
            self.mass_per_length_kgpm,
            self.EA_N,
            self.EI_flap_Nm2,
            self.EI_edge_Nm2,
            self.GJ_Nm2,
            self.J_rho_kgm,
        ]

        for field in fields:
            if field.n_stations != eta.size:
                raise ValueError(
                    f"{field.name}: station count mismatch, "
                    f"expected {eta.size}, got {field.n_stations}."
                )
            if not np.allclose(field.eta, eta, rtol=0.0, atol=1e-12):
                raise ValueError(f"{field.name}: eta grid mismatch.")

        self._validate_positive_fields()

    def _validate_positive_fields(self) -> None:
        strictly_positive = [
            self.EA_N,
            self.EI_flap_Nm2,
            self.EI_edge_Nm2,
            self.GJ_Nm2,
        ]

        for field in strictly_positive:
            if np.any(field.values <= 0.0):
                raise ValueError(f"{field.name} must be strictly positive.")

        non_negative = [
            self.mass_per_length_kgpm,
            self.J_rho_kgm,
        ]

        for field in non_negative:
            if np.any(field.values < 0.0):
                raise ValueError(f"{field.name} must be non-negative.")

    @property
    def n_stations(self) -> int:
        return int(self.eta.size)

    @property
    def n_elements(self) -> int:
        return int(self.eta.size - 1)

    def span_stations_m(self, span_m: float) -> np.ndarray:
        span_m = float(span_m)
        if span_m <= 0.0:
            raise ValueError(f"span_m must be positive, got {span_m}.")
        return span_m * self.eta

    def element_lengths_m(self, span_m: float) -> np.ndarray:
        s = self.span_stations_m(span_m)
        L = np.diff(s)
        if np.any(L <= 0.0):
            raise ValueError("element lengths must be positive.")
        return L

    def element_averages(self, span_m: float) -> Dict[str, np.ndarray]:
        """
        返回与当前 fem_builder.py 单元循环一致的 element-average 参数。

        输出字段：
        - L_m
        - m_avg_kgpm
        - EA_avg_N
        - EI_flap_avg_Nm2
        - EI_edge_avg_Nm2
        - GJ_avg_Nm2
        - J_rho_avg_kgm
        """
        return {
            "L_m": self.element_lengths_m(span_m),
            "m_avg_kgpm": self.mass_per_length_kgpm.element_average(),
            "EA_avg_N": self.EA_N.element_average(),
            "EI_flap_avg_Nm2": self.EI_flap_Nm2.element_average(),
            "EI_edge_avg_Nm2": self.EI_edge_Nm2.element_average(),
            "GJ_avg_Nm2": self.GJ_Nm2.element_average(),
            "J_rho_avg_kgm": self.J_rho_kgm.element_average(),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "n_stations": self.n_stations,
            "n_elements": self.n_elements,
            "eta": self.eta.tolist(),
            "fields": {
                "mass_per_length_kgpm": self.mass_per_length_kgpm.to_dict(),
                "EA_N": self.EA_N.to_dict(),
                "EI_flap_Nm2": self.EI_flap_Nm2.to_dict(),
                "EI_edge_Nm2": self.EI_edge_Nm2.to_dict(),
                "GJ_Nm2": self.GJ_Nm2.to_dict(),
                "J_rho_kgm": self.J_rho_kgm.to_dict(),
            },
        }


def build_baseline_section_parameters(
    model: StudentBeamModel,
    *,
    ea_default: float = 1.0e10,
    gj_default: float = 1.0e9,
    j_rho_scale: float = 1.0,
) -> StudentSectionParameters:
    """
    从 StudentBeamModel 构造当前 frozen baseline 对应的截面参数集合。

    该函数必须复现当前 fem_builder.py 的默认语义：

    - mass_per_length 来自 model.mass_per_length_kgpm
    - EI_flap 来自 model.flapwise_ei_nm2
    - EI_edge 若存在则来自 model.edgewise_ei_nm2，否则退回 EI_flap
    - GJ 若存在则来自 model.torsional_gj_nm2，否则使用 gj_default
    - EA 当前使用 ea_default = 1e10
    - J_rho 当前使用 mass_per_length * j_rho_scale，
      使 element average 后等价于 fem_builder.py 中的 j_rho = m_avg * 1.0
    """
    model.validate()

    eta = _as_1d_float_array(model.eta, name="model.eta")
    _validate_eta(eta)

    mass = _as_1d_float_array(
        model.mass_per_length_kgpm,
        name="model.mass_per_length_kgpm",
    )
    ei_flap = _as_1d_float_array(
        model.flapwise_ei_nm2,
        name="model.flapwise_ei_nm2",
    )

    if model.edgewise_ei_nm2 is None:
        ei_edge = ei_flap.copy()
    else:
        ei_edge = _as_1d_float_array(
            model.edgewise_ei_nm2,
            name="model.edgewise_ei_nm2",
        )

    if model.torsional_gj_nm2 is None:
        gj = np.full_like(eta, float(gj_default), dtype=np.float64)
    else:
        gj = _as_1d_float_array(
            model.torsional_gj_nm2,
            name="model.torsional_gj_nm2",
        )

    ea = np.full_like(eta, float(ea_default), dtype=np.float64)
    j_rho = mass * float(j_rho_scale)

    return StudentSectionParameters(
        eta=eta,
        mass_per_length_kgpm=SpanwiseSectionField(
            name="mass_per_length_kgpm",
            eta=eta,
            values=mass,
            unit="kg/m",
        ),
        EA_N=SpanwiseSectionField(
            name="EA_N",
            eta=eta,
            values=ea,
            unit="N",
        ),
        EI_flap_Nm2=SpanwiseSectionField(
            name="EI_flap_Nm2",
            eta=eta,
            values=ei_flap,
            unit="N*m^2",
        ),
        EI_edge_Nm2=SpanwiseSectionField(
            name="EI_edge_Nm2",
            eta=eta,
            values=ei_edge,
            unit="N*m^2",
        ),
        GJ_Nm2=SpanwiseSectionField(
            name="GJ_Nm2",
            eta=eta,
            values=gj,
            unit="N*m^2",
        ),
        J_rho_kgm=SpanwiseSectionField(
            name="J_rho_kgm",
            eta=eta,
            values=j_rho,
            unit="kg*m",
        ),
        source="current_fem_builder_baseline_compatible",
    )


def build_element_section_table(
    model: StudentBeamModel,
    params: Optional[StudentSectionParameters] = None,
) -> Dict[str, np.ndarray]:
    """
    构造单元级截面参数表。

    当前主要用于：
    - 测试 section_parameterization 是否复现 fem_builder.py 的单元平均逻辑；
    - 后续 corrected_fem_builder.py 的输入准备。

    输出均为 numpy array。
    """
    if params is None:
        params = build_baseline_section_parameters(model)

    span_m = float(model.span_m)
    station_s = params.span_stations_m(span_m)
    elem = params.element_averages(span_m)

    n_elem = params.n_elements
    element_index = np.arange(n_elem, dtype=np.int64)

    return {
        "element_index": element_index,
        "eta0": params.eta[:-1].copy(),
        "eta1": params.eta[1:].copy(),
        "s0_m": station_s[:-1].copy(),
        "s1_m": station_s[1:].copy(),
        "L_m": elem["L_m"].copy(),
        "m_avg_kgpm": elem["m_avg_kgpm"].copy(),
        "EA_avg_N": elem["EA_avg_N"].copy(),
        "EI_flap_avg_Nm2": elem["EI_flap_avg_Nm2"].copy(),
        "EI_edge_avg_Nm2": elem["EI_edge_avg_Nm2"].copy(),
        "GJ_avg_Nm2": elem["GJ_avg_Nm2"].copy(),
        "J_rho_avg_kgm": elem["J_rho_avg_kgm"].copy(),
    }


def summarize_section_parameters(
    params: StudentSectionParameters,
    *,
    span_m: float,
) -> Dict[str, Any]:
    """
    生成简要摘要，便于测试脚本或日志打印。
    """
    elem = params.element_averages(span_m)

    def _field_summary(arr: np.ndarray) -> Dict[str, float]:
        arr = np.asarray(arr, dtype=np.float64)
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
        }

    return {
        "source": params.source,
        "n_stations": params.n_stations,
        "n_elements": params.n_elements,
        "span_m": float(span_m),
        "station_fields": {
            "mass_per_length_kgpm": _field_summary(params.mass_per_length_kgpm.values),
            "EA_N": _field_summary(params.EA_N.values),
            "EI_flap_Nm2": _field_summary(params.EI_flap_Nm2.values),
            "EI_edge_Nm2": _field_summary(params.EI_edge_Nm2.values),
            "GJ_Nm2": _field_summary(params.GJ_Nm2.values),
            "J_rho_kgm": _field_summary(params.J_rho_kgm.values),
        },
        "element_fields": {
            key: _field_summary(value)
            for key, value in elem.items()
            if key != "L_m"
        },
        "element_length_m": _field_summary(elem["L_m"]),
    }

@dataclass(frozen=True)
class SectionParameterCorrection:
    """
    截面参数修正项。

    当前阶段只支持手动显式修正，不接网络。

    约定：
    corrected_value = baseline_value * (1 + relative_delta) + absolute_delta

    支持字段：
    - EA_N
    - EI_flap_Nm2
    - EI_edge_Nm2
    - GJ_Nm2
    - J_rho_kgm

    注意：
    - 当前不修改 mass_per_length_kgpm；
    - mass_per_length 会影响平动质量矩阵，暂时保持 baseline；
    - J_rho 单独影响扭转惯性；
    - 所有 delta 均可为 None / scalar / shape=(n_stations,)。
    """

    delta_EA_relative: Optional[Any] = None
    delta_EA_absolute: Optional[Any] = None

    delta_EI_flap_relative: Optional[Any] = None
    delta_EI_flap_absolute: Optional[Any] = None

    delta_EI_edge_relative: Optional[Any] = None
    delta_EI_edge_absolute: Optional[Any] = None

    delta_GJ_relative: Optional[Any] = None
    delta_GJ_absolute: Optional[Any] = None

    delta_J_rho_relative: Optional[Any] = None
    delta_J_rho_absolute: Optional[Any] = None


def _as_station_delta(
    values: Optional[Any],
    *,
    n_stations: int,
    name: str,
) -> np.ndarray:
    """
    将修正项整理为 station-level array。

    支持：
    - None: 全零
    - scalar: 广播到全部 station
    - shape=(n_stations,)
    """
    if values is None:
        return np.zeros(n_stations, dtype=np.float64)

    arr = np.asarray(values, dtype=np.float64)

    if arr.ndim == 0:
        return np.full(n_stations, float(arr), dtype=np.float64)

    arr = arr.reshape(-1)
    if arr.size != n_stations:
        raise ValueError(
            f"{name} must be None, scalar, or shape=({n_stations},), "
            f"got shape={arr.shape}."
        )

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")

    return arr.astype(np.float64, copy=True)


def _apply_relative_absolute_correction(
    baseline: np.ndarray,
    *,
    relative_delta: Optional[Any],
    absolute_delta: Optional[Any],
    n_stations: int,
    name: str,
) -> np.ndarray:
    """
    corrected = baseline * (1 + relative_delta) + absolute_delta
    """
    base = np.asarray(baseline, dtype=np.float64).reshape(-1)
    if base.size != n_stations:
        raise ValueError(
            f"{name} baseline size mismatch: expected {n_stations}, got {base.size}."
        )

    rel = _as_station_delta(
        relative_delta,
        n_stations=n_stations,
        name=f"{name}.relative_delta",
    )
    abs_delta = _as_station_delta(
        absolute_delta,
        n_stations=n_stations,
        name=f"{name}.absolute_delta",
    )

    corrected = base * (1.0 + rel) + abs_delta

    if not np.all(np.isfinite(corrected)):
        raise ValueError(f"{name} corrected values contain non-finite values.")

    return corrected


def apply_section_parameter_corrections(
    baseline_params: StudentSectionParameters,
    correction: Optional[SectionParameterCorrection] = None,
    *,
    source_suffix: str = "manual_section_correction",
) -> StudentSectionParameters:
    """
    在 baseline section parameters 上应用显式截面参数修正。

    correction=None 时严格返回与 baseline 数值一致的新 StudentSectionParameters。

    当前用于：
    - 阶段 2 / 阶段 3 交界处的手动物理参数 sanity test；
    - 后续 corrected_fem_builder.py 的参数修正入口；
    - 暂不作为网络输出格式。
    """
    if correction is None:
        correction = SectionParameterCorrection()

    eta = baseline_params.eta.copy()
    n = baseline_params.n_stations

    EA = _apply_relative_absolute_correction(
        baseline_params.EA_N.values,
        relative_delta=correction.delta_EA_relative,
        absolute_delta=correction.delta_EA_absolute,
        n_stations=n,
        name="EA_N",
    )

    EI_flap = _apply_relative_absolute_correction(
        baseline_params.EI_flap_Nm2.values,
        relative_delta=correction.delta_EI_flap_relative,
        absolute_delta=correction.delta_EI_flap_absolute,
        n_stations=n,
        name="EI_flap_Nm2",
    )

    EI_edge = _apply_relative_absolute_correction(
        baseline_params.EI_edge_Nm2.values,
        relative_delta=correction.delta_EI_edge_relative,
        absolute_delta=correction.delta_EI_edge_absolute,
        n_stations=n,
        name="EI_edge_Nm2",
    )

    GJ = _apply_relative_absolute_correction(
        baseline_params.GJ_Nm2.values,
        relative_delta=correction.delta_GJ_relative,
        absolute_delta=correction.delta_GJ_absolute,
        n_stations=n,
        name="GJ_Nm2",
    )

    J_rho = _apply_relative_absolute_correction(
        baseline_params.J_rho_kgm.values,
        relative_delta=correction.delta_J_rho_relative,
        absolute_delta=correction.delta_J_rho_absolute,
        n_stations=n,
        name="J_rho_kgm",
    )

    return StudentSectionParameters(
        eta=eta,
        mass_per_length_kgpm=SpanwiseSectionField(
            name="mass_per_length_kgpm",
            eta=eta,
            values=baseline_params.mass_per_length_kgpm.values.copy(),
            unit=baseline_params.mass_per_length_kgpm.unit,
        ),
        EA_N=SpanwiseSectionField(
            name="EA_N",
            eta=eta,
            values=EA,
            unit=baseline_params.EA_N.unit,
        ),
        EI_flap_Nm2=SpanwiseSectionField(
            name="EI_flap_Nm2",
            eta=eta,
            values=EI_flap,
            unit=baseline_params.EI_flap_Nm2.unit,
        ),
        EI_edge_Nm2=SpanwiseSectionField(
            name="EI_edge_Nm2",
            eta=eta,
            values=EI_edge,
            unit=baseline_params.EI_edge_Nm2.unit,
        ),
        GJ_Nm2=SpanwiseSectionField(
            name="GJ_Nm2",
            eta=eta,
            values=GJ,
            unit=baseline_params.GJ_Nm2.unit,
        ),
        J_rho_kgm=SpanwiseSectionField(
            name="J_rho_kgm",
            eta=eta,
            values=J_rho,
            unit=baseline_params.J_rho_kgm.unit,
        ),
        source=f"{baseline_params.source}+{source_suffix}",
    )


def build_corrected_section_parameters_from_model(
    model: StudentBeamModel,
    *,
    correction: Optional[SectionParameterCorrection] = None,
    ea_default: float = 1.0e10,
    gj_default: float = 1.0e9,
    j_rho_scale: float = 1.0,
) -> StudentSectionParameters:
    """
    从 StudentBeamModel 出发，构造 baseline-compatible section parameters，
    然后应用手动截面参数修正。

    correction=None 时应严格复现当前 frozen baseline section parameters。
    """
    baseline_params = build_baseline_section_parameters(
        model,
        ea_default=ea_default,
        gj_default=gj_default,
        j_rho_scale=j_rho_scale,
    )

    return apply_section_parameter_corrections(
        baseline_params,
        correction=correction,
    )


def make_uniform_section_correction(
    *,
    delta_EA_relative: Optional[float] = None,
    delta_EI_flap_relative: Optional[float] = None,
    delta_EI_edge_relative: Optional[float] = None,
    delta_GJ_relative: Optional[float] = None,
    delta_J_rho_relative: Optional[float] = None,
) -> SectionParameterCorrection:
    """
    便捷构造全展向 uniform relative correction。

    主要用于 sanity test。
    """
    return SectionParameterCorrection(
        delta_EA_relative=delta_EA_relative,
        delta_EI_flap_relative=delta_EI_flap_relative,
        delta_EI_edge_relative=delta_EI_edge_relative,
        delta_GJ_relative=delta_GJ_relative,
        delta_J_rho_relative=delta_J_rho_relative,
    )