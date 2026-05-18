from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

PhysicalTarget = Literal["K", "C", "F", "M", "G", "Q"]


@dataclass(frozen=True)
class PhysicalParameterSpec:
    """
    单个可学习物理参数的定义。

    当前阶段新增 alpha_x，用于先单独验证 x-bending 动态刚度残差
    是否能够修正复杂工况下的 x 方向相位漂移。

    约束：
    - Transformer 不直接输出位移修正；
    - Transformer 只输出这里注册过的低维物理参数；
    - 参数必须进入 full-order MCK / Newmark 方程；
    - theta=0 时必须回退 static corrected student。
    """

    name: str
    dim: int
    target: PhysicalTarget
    max_abs: float
    template_name: str | None
    description: str
    enabled_by_default: bool = False
    smooth_weight: float = 1.0
    amplitude_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError(f"Parameter {self.name!r} dim must be positive, got {self.dim}.")
        if self.max_abs <= 0.0:
            raise ValueError(f"Parameter {self.name!r} max_abs must be positive, got {self.max_abs}.")


def build_default_physical_parameter_specs() -> dict[str, PhysicalParameterSpec]:
    """
    当前支持两个全局刚度类动态物理参数：

    alpha_x:
        x-bending 主方向动态刚度残差。当前第一步默认只启用它，
        用于验证是否能降低 x 方向相位漂移。

    alpha_xy:
        x/y 几何耦合刚度残差。保留为后续联合训练参数，
        但当前 alpha_x 验证阶段默认不启用。

    base K0 已经包含：
        Phi_base(s) = -initial_twist_deg(s)
        kappa_y_global_scale = 0.952
    """
    return {
        "alpha_x": PhysicalParameterSpec(
            name="alpha_x",
            dim=1,
            target="K",
            max_abs=0.15,
            template_name="K_x_template",
            description=(
                "x-bending 主方向动态刚度残差。它作用在 static corrected student 的 K0 上，"
                "主要通过 ux/ry 相关自由度的等效刚度模板修正 x 方向主响应的频率/相位。"
            ),
            enabled_by_default=True,
            smooth_weight=1.0,
            amplitude_weight=1.0,
        ),
        "alpha_xy": PhysicalParameterSpec(
            name="alpha_xy",
            dim=1,
            target="K",
            max_abs=0.06,
            template_name="K_xy_template",
            description=(
                "全局 x/y 几何耦合刚度残差。它表示固定 Phi_base(s)=-initial_twist_deg(s) "
                "和 static kappa_y_global_scale=0.952 之后的 coupling residual。"
                "当前 alpha_x 验证阶段默认关闭，后续可与 alpha_x 联合训练。"
            ),
            enabled_by_default=False,
            smooth_weight=1.0,
            amplitude_weight=1.0,
        ),
        "beta_force_x": PhysicalParameterSpec(
            name="beta_force_x",
            dim=1,
            target="F",
            max_abs=0.50,
            template_name="force_x",
            description=(
                "Dimensionless x-load amplitude correction. It does not change M, C, or K; "
                "the Newmark core uses F_eff(t)=F(t)+beta_force_x(t)*P_x*F(t), where P_x "
                "selects nodal Fx degrees of freedom. This targets equivalent load-projection "
                "and geometric-nonlinear amplitude residuals while preserving the alpha stiffness "
                "frequency/phase mechanism."
            ),
            enabled_by_default=False,
            smooth_weight=1.0,
            amplitude_weight=1.0,
        ),
        "beta_force_y": PhysicalParameterSpec(
            name="beta_force_y",
            dim=1,
            target="F",
            max_abs=0.50,
            template_name="force_y",
            description=(
                "Dimensionless y-load amplitude correction. It does not change M, C, or K; "
                "the Newmark core uses F_eff(t)=F(t)+beta_force_y(t)*P_y*F(t), where P_y "
                "selects nodal Fy degrees of freedom. This gives beta a direct amplitude lever "
                "for Euler-Bernoulli versus geometrically exact load/response mismatch."
            ),
            enabled_by_default=False,
            smooth_weight=1.0,
            amplitude_weight=1.0,
        ),
    }


def _split_enabled_params(value: str) -> list[str]:
    """支持逗号和空格两种写法。"""
    items: list[str] = []
    for chunk in value.replace(",", " ").split():
        name = chunk.strip()
        if name:
            items.append(name)
    return items


def _normalize_enabled_params(
    enabled_params: str | Sequence[str] | None,
    *,
    all_specs: dict[str, PhysicalParameterSpec],
) -> list[str]:
    if enabled_params is None:
        names = [name for name, spec in all_specs.items() if spec.enabled_by_default]
    elif isinstance(enabled_params, str):
        names = _split_enabled_params(enabled_params)
    else:
        names = []
        for item in enabled_params:
            names.extend(_split_enabled_params(str(item)))

    if not names:
        raise ValueError(
            f"enabled_params is empty. Supported parameters: {list(all_specs.keys())}."
        )

    unknown = [name for name in names if name not in all_specs]
    if unknown:
        raise KeyError(
            f"Unknown physical parameter(s): {unknown}. "
            f"Supported parameters: {list(all_specs.keys())}. "
            "Use --enabled-params alpha_x,alpha_xy,beta_force_x,beta_force_y "
            "for the force-amplitude beta experiment."
        )

    out: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name not in seen:
            out.append(name)
            seen.add(name)

    return out


class PhysicalParameterRegistry:
    """
    可学习物理参数注册表。

    当前 alpha_x 验证阶段推荐：
        enabled_params = ["alpha_x"]
        total_dim = 1

    后续可启用：
        enabled_params = ["alpha_x", "alpha_xy"]
    """

    def __init__(
        self,
        enabled_params: str | Sequence[str] | None = None,
        specs: dict[str, PhysicalParameterSpec] | None = None,
    ) -> None:
        self.specs_all = specs or build_default_physical_parameter_specs()
        self.enabled_names = _normalize_enabled_params(enabled_params, all_specs=self.specs_all)

        self.specs_enabled: dict[str, PhysicalParameterSpec] = {
            name: self.specs_all[name]
            for name in self.enabled_names
        }

        self._slices: dict[str, slice] = {}
        start = 0
        for name in self.enabled_names:
            dim = int(self.specs_all[name].dim)
            self._slices[name] = slice(start, start + dim)
            start += dim

        self.total_dim = int(start)

    @property
    def names(self) -> list[str]:
        return list(self.enabled_names)

    @property
    def slices(self) -> dict[str, slice]:
        return dict(self._slices)

    def get_spec(self, name: str) -> PhysicalParameterSpec:
        if name not in self.specs_enabled:
            raise KeyError(
                f"Parameter {name!r} is not enabled. Enabled parameters: {self.enabled_names}"
            )
        return self.specs_enabled[name]

    def max_abs_list(self) -> list[float]:
        values: list[float] = []
        for name in self.enabled_names:
            spec = self.specs_enabled[name]
            values.extend([float(spec.max_abs)] * int(spec.dim))
        return values

    def template_names(self) -> list[str]:
        out: list[str] = []
        for name in self.enabled_names:
            template_name = self.specs_enabled[name].template_name
            if template_name is not None:
                out.append(template_name)
        return out

    def split_theta(self, theta: Any) -> dict[str, Any]:
        if theta.shape[-1] != self.total_dim:
            raise ValueError(
                f"theta last dim mismatch: expected {self.total_dim}, got {theta.shape[-1]}."
            )
        return {name: theta[..., sl] for name, sl in self._slices.items()}

    def summary(self) -> dict[str, Any]:
        return {
            "enabled_params": self.enabled_names,
            "total_dim": self.total_dim,
            "params": {
                name: {
                    "dim": spec.dim,
                    "target": spec.target,
                    "max_abs": spec.max_abs,
                    "template_name": spec.template_name,
                    "description": spec.description,
                    "smooth_weight": spec.smooth_weight,
                    "amplitude_weight": spec.amplitude_weight,
                    "slice": [self._slices[name].start, self._slices[name].stop],
                }
                for name, spec in self.specs_enabled.items()
            },
        }


def build_physical_parameter_registry(
    enabled_params: str | Sequence[str] | None = None,
) -> PhysicalParameterRegistry:
    return PhysicalParameterRegistry(enabled_params=enabled_params)
