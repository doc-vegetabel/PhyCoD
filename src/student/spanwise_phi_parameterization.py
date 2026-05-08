from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np


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


def get_eta_array(model_or_eta: Any) -> np.ndarray:
    """
    从 StudentBeamModel 或 eta array 中读取 eta。

    支持：
    - StudentBeamModel-like object with `.eta`
    - array-like eta
    """
    if hasattr(model_or_eta, "eta"):
        eta = _as_1d_float_array(model_or_eta.eta, name="model.eta")
    else:
        eta = _as_1d_float_array(model_or_eta, name="eta")

    _validate_eta(eta)
    return eta


@dataclass(frozen=True)
class SpanwisePhiProfile:
    """
    展向主轴旋转角 phi(s) profile。

    当前约定：
    - eta: station-level eta, shape=(n_stations,)
    - phi_deg: station-level phi in degrees, shape=(n_stations,)
    - coupled_fem_builder 会将 station-level phi 平均到 element-level phi
    """

    name: str
    eta: np.ndarray
    phi_deg: np.ndarray
    source: str = ""

    def __post_init__(self) -> None:
        eta = _as_1d_float_array(self.eta, name=f"{self.name}.eta")
        phi_deg = _as_1d_float_array(self.phi_deg, name=f"{self.name}.phi_deg")

        _validate_eta(eta)

        if eta.shape != phi_deg.shape:
            raise ValueError(
                f"{self.name}: eta and phi_deg shape mismatch: "
                f"eta={eta.shape}, phi_deg={phi_deg.shape}."
            )

        object.__setattr__(self, "eta", eta.copy())
        object.__setattr__(self, "phi_deg", phi_deg.copy())

    @property
    def n_stations(self) -> int:
        return int(self.eta.size)

    @property
    def n_elements(self) -> int:
        return int(self.eta.size - 1)

    @property
    def phi_rad(self) -> np.ndarray:
        return np.deg2rad(self.phi_deg)

    def element_phi_deg(self) -> np.ndarray:
        return 0.5 * (self.phi_deg[:-1] + self.phi_deg[1:])

    def element_phi_rad(self) -> np.ndarray:
        return np.deg2rad(self.element_phi_deg())

    def summary(self) -> Dict[str, Any]:
        elem_phi = self.element_phi_deg()
        return {
            "name": self.name,
            "source": self.source,
            "n_stations": self.n_stations,
            "n_elements": self.n_elements,
            "station_phi_deg_min": float(np.min(self.phi_deg)),
            "station_phi_deg_max": float(np.max(self.phi_deg)),
            "station_phi_deg_mean": float(np.mean(self.phi_deg)),
            "element_phi_deg_min": float(np.min(elem_phi)),
            "element_phi_deg_max": float(np.max(elem_phi)),
            "element_phi_deg_mean": float(np.mean(elem_phi)),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "eta": self.eta.tolist(),
            "phi_deg": self.phi_deg.tolist(),
            "element_phi_deg": self.element_phi_deg().tolist(),
            "summary": self.summary(),
        }


def build_uniform_phi_profile(
    model_or_eta: Any,
    *,
    phi_deg: float,
    name: str = "uniform_phi",
) -> SpanwisePhiProfile:
    """
    构造 uniform phi profile。

    phi_deg 为所有 station 的统一角度。
    """
    eta = get_eta_array(model_or_eta)
    phi = np.full_like(eta, float(phi_deg), dtype=np.float64)

    return SpanwisePhiProfile(
        name=name,
        eta=eta,
        phi_deg=phi,
        source=f"uniform_phi_deg={float(phi_deg):.6g}",
    )


def build_linear_phi_profile(
    model_or_eta: Any,
    *,
    root_phi_deg: float,
    tip_phi_deg: float,
    name: str = "linear_phi",
) -> SpanwisePhiProfile:
    """
    构造 root-to-tip 线性变化的 phi profile。
    """
    eta = get_eta_array(model_or_eta)
    root_phi_deg = float(root_phi_deg)
    tip_phi_deg = float(tip_phi_deg)

    phi = root_phi_deg + (tip_phi_deg - root_phi_deg) * eta

    return SpanwisePhiProfile(
        name=name,
        eta=eta,
        phi_deg=phi,
        source=f"linear_root={root_phi_deg:.6g}_tip={tip_phi_deg:.6g}",
    )


def build_piecewise_constant_phi_profile(
    model_or_eta: Any,
    *,
    breakpoints: Sequence[float],
    values_deg: Sequence[float],
    name: str = "piecewise_phi",
) -> SpanwisePhiProfile:
    """
    构造分段常数 phi profile。

    参数：
        breakpoints:
            长度 = n_segments + 1。
            例如 [0.0, 0.25, 0.5, 0.75, 1.0]

        values_deg:
            长度 = n_segments。
            例如 [0.0, 2.0, 5.0, 8.0]

    赋值规则：
        eta in [breakpoints[i], breakpoints[i+1]) -> values_deg[i]
        eta == breakpoints[-1] -> values_deg[-1]
    """
    eta = get_eta_array(model_or_eta)

    bp = _as_1d_float_array(breakpoints, name="breakpoints")
    values = _as_1d_float_array(values_deg, name="values_deg")

    if bp.size < 2:
        raise ValueError("breakpoints must contain at least two values.")
    if values.size != bp.size - 1:
        raise ValueError(
            "values_deg length must equal len(breakpoints)-1, "
            f"got values={values.size}, breakpoints={bp.size}."
        )
    if np.any(np.diff(bp) <= 0.0):
        raise ValueError("breakpoints must be strictly increasing.")
    if bp[0] > eta[0] + 1e-12:
        raise ValueError(
            f"breakpoints[0] must be <= eta[0]={eta[0]}, got {bp[0]}."
        )
    if bp[-1] < eta[-1] - 1e-12:
        raise ValueError(
            f"breakpoints[-1] must be >= eta[-1]={eta[-1]}, got {bp[-1]}."
        )

    # search among internal breakpoints.
    # eta exactly equal to an internal breakpoint enters the right segment.
    segment_ids = np.searchsorted(bp[1:-1], eta, side="right")
    phi = values[segment_ids]

    return SpanwisePhiProfile(
        name=name,
        eta=eta,
        phi_deg=phi,
        source=(
            "piecewise_constant:"
            f"breakpoints={bp.tolist()},values_deg={values.tolist()}"
        ),
    )


def build_control_point_phi_profile(
    model_or_eta: Any,
    *,
    control_eta: Sequence[float],
    control_phi_deg: Sequence[float],
    name: str = "control_point_phi",
) -> SpanwisePhiProfile:
    """
    通过 control points 线性插值得到 station-level phi(s)。

    例如：
        control_eta = [0, 0.25, 0.5, 0.75, 1]
        control_phi_deg = [0, 2, 5, 8, 10]
    """
    eta = get_eta_array(model_or_eta)

    c_eta = _as_1d_float_array(control_eta, name="control_eta")
    c_phi = _as_1d_float_array(control_phi_deg, name="control_phi_deg")

    if c_eta.size != c_phi.size:
        raise ValueError(
            f"control_eta/control_phi_deg size mismatch: "
            f"{c_eta.size} vs {c_phi.size}."
        )
    if c_eta.size < 2:
        raise ValueError("At least two control points are required.")
    if np.any(np.diff(c_eta) <= 0.0):
        raise ValueError("control_eta must be strictly increasing.")
    if c_eta[0] > eta[0] + 1e-12:
        raise ValueError(
            f"control_eta[0] must be <= eta[0]={eta[0]}, got {c_eta[0]}."
        )
    if c_eta[-1] < eta[-1] - 1e-12:
        raise ValueError(
            f"control_eta[-1] must be >= eta[-1]={eta[-1]}, got {c_eta[-1]}."
        )

    phi = np.interp(eta, c_eta, c_phi)

    return SpanwisePhiProfile(
        name=name,
        eta=eta,
        phi_deg=phi,
        source=(
            "control_point_linear:"
            f"control_eta={c_eta.tolist()},control_phi_deg={c_phi.tolist()}"
        ),
    )


def parse_float_list(text: str) -> list[float]:
    """
    解析逗号分隔浮点数列表。
    """
    out: list[float] = []
    for item in str(text).split(","):
        item = item.strip()
        if item == "":
            continue
        out.append(float(item))
    if len(out) == 0:
        raise ValueError("Parsed float list is empty.")
    return out


def profile_similarity_metrics(
    profile_a: SpanwisePhiProfile,
    profile_b: SpanwisePhiProfile,
) -> Dict[str, float]:
    """
    比较两个 station-level phi profile。
    """
    if profile_a.eta.shape != profile_b.eta.shape:
        raise ValueError("eta shape mismatch.")
    if not np.allclose(profile_a.eta, profile_b.eta, rtol=0.0, atol=1e-12):
        raise ValueError("eta grid mismatch.")

    diff = profile_a.phi_deg - profile_b.phi_deg

    return {
        "mae_deg": float(np.mean(np.abs(diff))),
        "max_abs_deg": float(np.max(np.abs(diff))),
        "rmse_deg": float(np.sqrt(np.mean(diff**2))),
    }