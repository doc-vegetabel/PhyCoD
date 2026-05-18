from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from src.student.base_full_order_builder import (
    apply_global_kappa_y_scale_to_k,
    build_base_student_full_order_mk,
)
from src.student.coupled_fem_builder import build_coupled_fem_matrices_6dof_degrees

XYTemplateMode = Literal["uniform", "root_to_tip", "tip_to_root"]


@dataclass
class PhysicalTemplateConfig:
    """
    动态物理模板构造配置。

    当前支持：
        K_x_template:
            alpha_x 对应的 x-bending 主方向动态刚度残差模板。
            作用在 ux/ry 相关自由度。

        K_xy_template:
            alpha_xy 对应的 x/y coupling residual 模板。
            当前 alpha_x-only 验证阶段默认不启用，但保留以便后续联合训练。

    base K0 仍然是 static corrected student：
        Phi_base(s) = -initial_twist_deg(s)
        kappa_y_global_scale = 0.952
    """

    blade_csv: str
    alpha_flap: float = 1.0
    alpha_edge: float = 1.0
    alpha_torsion: float = 1.0

    twist_column: str = "initial_twist_deg"
    phi_sign: float = -1.0
    rotate_mass: bool = False

    kappa_y_static_scale: float = 0.952
    kappa_y_scale_mode: str = "y_bending"

    # alpha_x template finite difference:
    # K_x_template = (S_x(1+delta) K0 S_x(1+delta) - K0) / delta
    x_delta_scale: float = 0.01
    x_scale_mode: str = "x_bending"

    # K_y_template is kept as a diagnostic directional shape. Force beta does
    # not use it directly; beta_force_y acts on physical Fy load components.
    y_delta_scale: float = 0.01
    y_scale_mode: str = "y_bending"

    # alpha_xy residual phi finite difference
    xy_template_mode: XYTemplateMode = "root_to_tip"
    xy_delta_phi_deg: float = 1.0

    enabled_params: str = "alpha_x"

    verbose: bool = True


@dataclass
class PhysicalTemplateBundle:
    """
    物理模板包。

    M0, K0:
        static corrected student 的 base matrices。
        K0 已包含 Phi_base 和 static kappa_y_global_scale=0.952。

    K_x_template:
        alpha_x 对应的 x-bending 主方向动态刚度残差模板。

    K_xy_template:
        alpha_xy 对应的 x/y coupling residual 刚度模板。
    """

    M0: np.ndarray
    K0: np.ndarray
    K_phi_unscaled: np.ndarray
    K_x_template: np.ndarray
    K_y_template: np.ndarray
    K_xy_template: np.ndarray
    phi_element_deg: np.ndarray
    metadata: dict[str, Any]

    def stiffness_template_dict(self) -> dict[str, np.ndarray]:
        return {
            "K_x_template": self.K_x_template,
            "K_xy_template": self.K_xy_template,
        }

    def summary(self) -> dict[str, Any]:
        shapes = {
            "M0": list(self.M0.shape),
            "K0": list(self.K0.shape),
            "K_phi_unscaled": list(self.K_phi_unscaled.shape),
            "K_x_template": list(self.K_x_template.shape),
            "K_y_template": list(self.K_y_template.shape),
            "K_xy_template": list(self.K_xy_template.shape),
            "phi_element_deg": list(self.phi_element_deg.shape),
        }

        norms = {
            "M0_fro": float(np.linalg.norm(self.M0)),
            "K0_fro": float(np.linalg.norm(self.K0)),
            "K_phi_unscaled_fro": float(np.linalg.norm(self.K_phi_unscaled)),
            "K_x_template_fro": float(np.linalg.norm(self.K_x_template)),
            "K_y_template_fro": float(np.linalg.norm(self.K_y_template)),
            "K_xy_template_fro": float(np.linalg.norm(self.K_xy_template)),
        }

        symmetry = {
            "K0_sym_max_abs": float(np.max(np.abs(self.K0 - self.K0.T))),
            "K_x_template_sym_max_abs": float(
                np.max(np.abs(self.K_x_template - self.K_x_template.T))
            ),
            "K_y_template_sym_max_abs": float(
                np.max(np.abs(self.K_y_template - self.K_y_template.T))
            ),
            "K_xy_template_sym_max_abs": float(
                np.max(np.abs(self.K_xy_template - self.K_xy_template.T))
            ),
        }

        return {
            "shapes": shapes,
            "norms": norms,
            "symmetry": symmetry,
            "metadata": self.metadata,
        }


def _as_float_matrix(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape={arr.shape}.")
    return arr


def _symmetrize(K: np.ndarray) -> np.ndarray:
    return 0.5 * (K + K.T)


def _split_enabled_params(value: str) -> list[str]:
    return [p.strip() for p in str(value).replace(",", " ").split() if p.strip()]


def _validate_enabled_params(enabled_params: str) -> list[str]:
    names = _split_enabled_params(enabled_params)
    if not names:
        names = ["alpha_x"]
    allowed = {
        "alpha_x",
        "alpha_xy",
        "beta_force_x",
        "beta_force_y",
    }
    unknown = [n for n in names if n not in allowed]
    if unknown:
        raise ValueError(
            f"Unsupported enabled_params={names}. Supported templates are {sorted(allowed)}."
        )
    return names


def _build_direction_scale_vector(
    *,
    n_dofs: int,
    scale: float,
    mode: str,
) -> tuple[np.ndarray, list[int]]:
    """
    Build S diagonal for directional bending stiffness scaling.

    DOF convention per free node:
        [ux, uy, uz, rx, ry, rz]

    x_bending uses ux and ry.
    y_bending uses uy and rx.
    """
    if n_dofs % 6 != 0:
        raise ValueError(f"n_dofs must be divisible by 6, got {n_dofs}.")
    if scale <= 0.0 or not np.isfinite(scale):
        raise ValueError(f"scale must be positive finite, got {scale}.")

    n_nodes = n_dofs // 6
    mode = str(mode).lower()
    if mode == "x_bending":
        offsets = [0, 4]  # ux, ry
    elif mode == "ux_only":
        offsets = [0]
    elif mode == "y_bending":
        offsets = [1, 3]  # uy, rx
    elif mode == "uy_only":
        offsets = [1]
    else:
        raise ValueError(
            f"Unsupported scale mode={mode!r}. Expected x_bending, ux_only, y_bending, or uy_only."
        )

    indices: list[int] = []
    for i in range(n_nodes):
        for off in offsets:
            indices.append(6 * i + off)

    s = np.ones(n_dofs, dtype=np.float64)
    s[np.asarray(indices, dtype=np.int64)] = np.sqrt(float(scale))
    return s, indices


def _apply_directional_scale_to_k(
    K: np.ndarray,
    *,
    scale: float,
    mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    K = _as_float_matrix(K, "K")
    s, indices = _build_direction_scale_vector(n_dofs=K.shape[0], scale=float(scale), mode=str(mode))
    K_scaled = (s[:, None] * K) * s[None, :]
    info = {
        "enabled": True,
        "scale": float(scale),
        "scale_mode": str(mode),
        "num_scaled_dofs": int(len(indices)),
        "scaled_dof_indices": [int(i) for i in indices],
    }
    return _symmetrize(K_scaled), info


def _build_xy_delta_phi_profile_deg(
    *,
    n_elements: int,
    mode: XYTemplateMode,
    delta_phi_deg: float,
) -> np.ndarray:
    if n_elements <= 0:
        raise ValueError(f"n_elements must be positive, got {n_elements}.")

    delta = float(delta_phi_deg)
    if not np.isfinite(delta) or abs(delta) <= 0.0:
        raise ValueError(f"delta_phi_deg must be non-zero finite, got {delta_phi_deg}.")

    if mode == "uniform":
        return np.full(n_elements, delta, dtype=np.float64)
    if mode == "root_to_tip":
        return np.linspace(delta, 0.0, n_elements, dtype=np.float64)
    if mode == "tip_to_root":
        return np.linspace(0.0, delta, n_elements, dtype=np.float64)

    raise ValueError(
        f"Unsupported xy_template_mode={mode!r}. "
        "Expected one of: 'uniform', 'root_to_tip', 'tip_to_root'."
    )


def build_dynamic_stiffness_templates(cfg: PhysicalTemplateConfig) -> PhysicalTemplateBundle:
    """
    构造 Transformer 动态物理参数需要的刚度模板。

    输出：
        K0:
            static corrected student 刚度矩阵，已包含：
                Phi_base(s)=-initial_twist_deg(s)
                kappa_y_global_scale=0.952

        K_x_template:
            alpha_x 对应模板：
                K_eff(t) = K0 + alpha_x(t) * K_x_template

        K_xy_template:
            alpha_xy 对应模板：
                K_eff(t) = K0 + alpha_xy(t) * K_xy_template
    """
    enabled_names = _validate_enabled_params(cfg.enabled_params)

    blade_csv = Path(cfg.blade_csv).resolve()
    if not blade_csv.exists():
        raise FileNotFoundError(f"blade_csv not found: {blade_csv}")

    if cfg.kappa_y_static_scale <= 0.0:
        raise ValueError(
            f"kappa_y_static_scale must be positive, got {cfg.kappa_y_static_scale}."
        )

    if abs(cfg.x_delta_scale) <= 0.0:
        raise ValueError(f"x_delta_scale must be non-zero, got {cfg.x_delta_scale}.")
    if 1.0 + float(cfg.x_delta_scale) <= 0.0:
        raise ValueError(f"1+x_delta_scale must be positive, got {1.0 + float(cfg.x_delta_scale)}.")
    if abs(cfg.y_delta_scale) <= 0.0:
        raise ValueError(f"y_delta_scale must be non-zero, got {cfg.y_delta_scale}.")
    if 1.0 + float(cfg.y_delta_scale) <= 0.0:
        raise ValueError(f"1+y_delta_scale must be positive, got {1.0 + float(cfg.y_delta_scale)}.")
    if abs(cfg.xy_delta_phi_deg) <= 0.0:
        raise ValueError(f"xy_delta_phi_deg must be non-zero, got {cfg.xy_delta_phi_deg}.")

    if bool(cfg.verbose):
        templates = []
        if "alpha_x" in enabled_names:
            templates.append("K_x_template")
        if "alpha_xy" in enabled_names:
            templates.append("K_xy_template")
        if "beta_force_x" in enabled_names:
            templates.append("force_x")
        if "beta_force_y" in enabled_names:
            templates.append("force_y")
        print()
        print("[Physical Templates Enabled Params]")
        print(f"  enabled_param_names = {enabled_names}")
        print(f"  dynamic templates   = {templates}")

    # ------------------------------------------------------------------
    # 1. Static corrected student:
    #       K0 = S_y(0.952) K_phi S_y(0.952)
    # ------------------------------------------------------------------
    base = build_base_student_full_order_mk(
        blade_csv=blade_csv,
        model_name="physical_template_static_corrected_student_alpha_x_alpha_xy",
        alpha_flap=float(cfg.alpha_flap),
        alpha_edge=float(cfg.alpha_edge),
        alpha_torsion=float(cfg.alpha_torsion),
        twist_column=str(cfg.twist_column),
        phi_sign=float(cfg.phi_sign),
        rotate_mass=bool(cfg.rotate_mass),
        use_kappa_y_global_scale=True,
        kappa_y_global_scale=float(cfg.kappa_y_static_scale),
        kappa_y_scale_mode=str(cfg.kappa_y_scale_mode),
        verbose=bool(cfg.verbose),
    )

    model = base["model"]
    M0 = _as_float_matrix(base["M"], "M0")
    K0 = _as_float_matrix(base["K"], "K0")
    phi_element_deg = np.asarray(base["phi_element_deg"], dtype=np.float64)

    if "K_phi_unscaled" not in base:
        raise KeyError(
            "build_base_student_full_order_mk(...) must return 'K_phi_unscaled'. "
            "请确认 base_full_order_builder.py 仍保留 static kappa_y_global_scale 支持。"
        )

    K_phi_unscaled = _as_float_matrix(base["K_phi_unscaled"], "K_phi_unscaled")

    # ------------------------------------------------------------------
    # 2. K_x_template: x-bending directional scale finite difference around K0.
    # ------------------------------------------------------------------
    x_delta = float(cfg.x_delta_scale)
    K_x_perturbed, x_scale_info = _apply_directional_scale_to_k(
        K0,
        scale=1.0 + x_delta,
        mode=str(cfg.x_scale_mode),
    )
    K_x_template = (K_x_perturbed - K0) / x_delta
    K_x_template = _symmetrize(K_x_template)

    # ------------------------------------------------------------------
    # 2b. K_y_template: y-bending stiffness-shape finite difference.
    #     It is not used as an alpha stiffness correction in the first beta
    #     experiment; it only supplies the physical shape for C_y_template.
    # ------------------------------------------------------------------
    y_delta = float(cfg.y_delta_scale)
    K_y_perturbed, y_scale_info = _apply_directional_scale_to_k(
        K0,
        scale=1.0 + y_delta,
        mode=str(cfg.y_scale_mode),
    )
    K_y_template = (K_y_perturbed - K0) / y_delta
    K_y_template = _symmetrize(K_y_template)

    # ------------------------------------------------------------------
    # 3. K_xy_template: residual phi profile finite difference around K0.
    # ------------------------------------------------------------------
    n_elements = int(phi_element_deg.shape[0])
    delta_phi_profile_deg = _build_xy_delta_phi_profile_deg(
        n_elements=n_elements,
        mode=cfg.xy_template_mode,
        delta_phi_deg=float(cfg.xy_delta_phi_deg),
    )

    phi_perturbed_deg = phi_element_deg + delta_phi_profile_deg

    _M_xy, K_phi_xy_perturbed, fem_info_xy = build_coupled_fem_matrices_6dof_degrees(
        model,
        section_params=None,
        phi_deg=phi_perturbed_deg,
        alpha_flap=float(cfg.alpha_flap),
        alpha_edge=float(cfg.alpha_edge),
        alpha_torsion=float(cfg.alpha_torsion),
        rotate_mass=bool(cfg.rotate_mass),
        return_full=True,
    )

    K_phi_xy_perturbed = _as_float_matrix(K_phi_xy_perturbed, "K_phi_xy_perturbed")

    # 注意：扰动后的 K 也要经过相同 static kappa_y scale，保证模板围绕 K0 构造。
    K_xy_perturbed, kappa_y_xy_info = apply_global_kappa_y_scale_to_k(
        K_phi_xy_perturbed,
        kappa_y_scale=float(cfg.kappa_y_static_scale),
        scale_mode=str(cfg.kappa_y_scale_mode),
    )
    K_xy_perturbed = _as_float_matrix(K_xy_perturbed, "K_xy_perturbed")

    delta_phi_rad = np.deg2rad(float(cfg.xy_delta_phi_deg))
    K_xy_template = (K_xy_perturbed - K0) / float(delta_phi_rad)
    K_xy_template = _symmetrize(K_xy_template)

    metadata: dict[str, Any] = {
        "template_type": "dynamic_stiffness_templates_alpha_x_alpha_xy",
        "config": asdict(cfg),
        "enabled_param_names_resolved": enabled_names,
        "blade_csv": str(blade_csv),
        "n_dofs": int(K0.shape[0]),
        "n_elements": n_elements,
        "base_definition": {
            "Phi_base": "-initial_twist_deg",
            "phi_sign": float(cfg.phi_sign),
            "rotate_mass": bool(cfg.rotate_mass),
            "kappa_y_global_scale": float(cfg.kappa_y_static_scale),
            "kappa_y_scale_mode": str(cfg.kappa_y_scale_mode),
            "note": "kappa_y_global_scale is static base correction, not Transformer dynamic alpha_y.",
        },
        "K_x_template": {
            "meaning": (
                "dK / d alpha_x, where alpha_x is an x-bending directional stiffness residual; "
                "K_eff(t)=K0+alpha_x(t)*K_x_template. Positive alpha_x increases ux/ry effective stiffness."
            ),
            "x_delta_scale": float(cfg.x_delta_scale),
            "x_scale_mode": str(cfg.x_scale_mode),
            "x_scale_info": x_scale_info,
        },
        "K_y_template": {
            "meaning": (
                "y-bending stiffness-shape diagnostic template. Force beta does not use it directly; "
                "beta_force_y scales physical Fy load components in the Newmark right-hand side."
            ),
            "y_delta_scale": float(cfg.y_delta_scale),
            "y_scale_mode": str(cfg.y_scale_mode),
            "y_scale_info": y_scale_info,
        },
        "force_beta": {
            "meaning": (
                "beta_force_x/y are dimensionless equivalent-load amplitude corrections. "
                "They scale only physical Fx/Fy nodal load components in the Newmark right-hand side "
                "and do not create dynamic damping templates."
            ),
        },
        "K_xy_template": {
            "meaning": (
                "dK / d alpha_xy, where alpha_xy is global residual phi amplitude "
                "in radians; K_eff(t)=K0+alpha_xy(t)*K_xy_template"
            ),
            "xy_template_mode": str(cfg.xy_template_mode),
            "xy_delta_phi_deg": float(cfg.xy_delta_phi_deg),
            "xy_delta_phi_rad": float(delta_phi_rad),
            "delta_phi_profile_deg_min": float(np.min(delta_phi_profile_deg)),
            "delta_phi_profile_deg_max": float(np.max(delta_phi_profile_deg)),
            "fem_info_xy": fem_info_xy,
            "kappa_y_xy_info": kappa_y_xy_info,
        },
    }

    return PhysicalTemplateBundle(
        M0=M0,
        K0=K0,
        K_phi_unscaled=K_phi_unscaled,
        K_x_template=K_x_template,
        K_y_template=K_y_template,
        K_xy_template=K_xy_template,
        phi_element_deg=phi_element_deg,
        metadata=metadata,
    )
