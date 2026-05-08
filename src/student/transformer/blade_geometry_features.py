from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.student.io import load_student_model_from_blade_master
from src.student.base_phi import build_base_phi_from_blade_csv
from src.student.section_parameterization import build_baseline_section_parameters


@dataclass
class BladeGeometryFeatureConfig:
    """
    叶片节点级几何/结构特征构造配置。

    当前第一版目标：
    - 为 Transformer 的 geometry branch 提供每个自由节点的静态构型特征；
    - 特征维度不要太大，先保证稳定、可解释；
    - 默认输出 48 个自由节点，对应 teacher/student 对齐中的 N002~N049。

    注意：
    - 当前不直接输出节点坐标 x/y/z，因为当前 student 模型主要是一维展向梁；
    - 后续如果有真实三维参考轴坐标，可以再加入 x0,y0,z0。
    """

    blade_csv: str
    twist_column: str = "initial_twist_deg"
    phi_sign: float = -1.0

    # 是否使用 log10 后标准化的结构参数，建议打开，避免 EI/GJ 数值尺度过大。
    use_log_structural_features: bool = True

    # 是否加入原始 eta / s/L / phi 等构型特征。
    include_span_position: bool = True
    include_phi_features: bool = True
    include_structural_features: bool = True

    # root node 是固定边界，不进入 288 DOF 自由响应，因此默认排除 station 0。
    exclude_root_station: bool = True

    eps: float = 1.0e-12


@dataclass
class BladeGeometryFeatureBundle:
    """
    叶片几何特征包。

    features:
        shape = (N_free_nodes, feature_dim)

    feature_names:
        每一列的名称。

    raw_fields:
        未标准化或部分原始字段，便于 debug / 可视化。

    metadata:
        构造配置和统计信息。
    """

    features: np.ndarray
    feature_names: list[str]
    raw_fields: dict[str, np.ndarray]
    metadata: dict[str, Any]

    @property
    def n_nodes(self) -> int:
        return int(self.features.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    def summary(self) -> dict[str, Any]:
        return {
            "n_nodes": self.n_nodes,
            "feature_dim": self.feature_dim,
            "feature_names": list(self.feature_names),
            "feature_stats": {
                name: {
                    "min": float(np.min(self.features[:, j])),
                    "max": float(np.max(self.features[:, j])),
                    "mean": float(np.mean(self.features[:, j])),
                    "std": float(np.std(self.features[:, j])),
                }
                for j, name in enumerate(self.feature_names)
            },
            "metadata": self.metadata,
        }


def _as_1d_float(x: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D.")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _standardize(x: np.ndarray, *, eps: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mean = float(np.mean(x))
    std = float(np.std(x))
    if std < eps:
        return np.zeros_like(x, dtype=np.float64)
    return (x - mean) / (std + eps)


def _safe_log10_standardize(x: np.ndarray, *, eps: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if np.any(x <= 0.0):
        raise ValueError("Structural feature must be positive before log10 transform.")
    return _standardize(np.log10(x + eps), eps=eps)


def build_blade_geometry_features(
    cfg: BladeGeometryFeatureConfig,
) -> BladeGeometryFeatureBundle:
    """
    构造自由节点级叶片几何/结构特征。

    默认输出节点：
        station 1 ... station 48

    对应响应向量中的 48 个自由节点：
        each node has [ux, uy, uz, rx, ry, rz]

    第一版默认特征：
        eta
        s_over_L
        phi_deg_scaled
        sin(phi)
        cos(phi)
        log_mass_z
        log_EI_flap_z
        log_EI_edge_z
        log_GJ_z
        log_EA_z
        log_J_rho_z
    """
    blade_csv = Path(cfg.blade_csv).resolve()
    if not blade_csv.exists():
        raise FileNotFoundError(f"blade_csv not found: {blade_csv}")

    model = load_student_model_from_blade_master(
        csv_path=str(blade_csv),
        model_name="blade_geometry_feature_model",
    )
    model.validate()

    phi_info = build_base_phi_from_blade_csv(
        blade_csv=blade_csv,
        twist_column=str(cfg.twist_column),
        sign=float(cfg.phi_sign),
        expected_n_stations=model.n_stations,
    )

    section_params = build_baseline_section_parameters(model)

    eta_station = _as_1d_float(model.eta, name="model.eta")
    span_station_m = np.asarray(model.span_stations_m, dtype=np.float64)

    phi_station_deg = _as_1d_float(phi_info["phi_station_deg"], name="phi_station_deg")
    phi_element_deg = _as_1d_float(phi_info["phi_element_deg"], name="phi_element_deg")

    # 对自由节点 i=1..48，使用“以该节点为末端的 element phi”作为节点 phi。
    # 因此 station 1 对应 element 0，station 48 对应 element 47。
    if cfg.exclude_root_station:
        node_slice = slice(1, None)
        phi_node_deg = phi_element_deg.copy()
    else:
        node_slice = slice(None)
        # 若包含 root，则 root phi 用 station phi，其余仍使用 element-ending 约定。
        phi_node_deg = np.concatenate([phi_station_deg[:1], phi_element_deg], axis=0)

    eta_node = eta_station[node_slice]
    span_node_m = span_station_m[node_slice]
    s_over_L = span_node_m / float(model.span_m)

    if phi_node_deg.shape[0] != eta_node.shape[0]:
        raise ValueError(
            f"phi_node_deg and eta_node size mismatch: "
            f"{phi_node_deg.shape[0]} vs {eta_node.shape[0]}"
        )

    mass = section_params.mass_per_length_kgpm.values[node_slice]
    EA = section_params.EA_N.values[node_slice]
    EI_flap = section_params.EI_flap_Nm2.values[node_slice]
    EI_edge = section_params.EI_edge_Nm2.values[node_slice]
    GJ = section_params.GJ_Nm2.values[node_slice]
    J_rho = section_params.J_rho_kgm.values[node_slice]

    features: list[np.ndarray] = []
    feature_names: list[str] = []

    if cfg.include_span_position:
        features.append(eta_node)
        feature_names.append("eta")

        features.append(s_over_L)
        feature_names.append("s_over_L")

    if cfg.include_phi_features:
        phi_rad = np.deg2rad(phi_node_deg)

        # 20 deg 只是为了把角度缩放到 O(1)，不是物理参数。
        features.append(phi_node_deg / 20.0)
        feature_names.append("phi_deg_div20")

        features.append(np.sin(phi_rad))
        feature_names.append("sin_phi")

        features.append(np.cos(phi_rad))
        feature_names.append("cos_phi")

    if cfg.include_structural_features:
        structural_raw = {
            "mass": mass,
            "EA": EA,
            "EI_flap": EI_flap,
            "EI_edge": EI_edge,
            "GJ": GJ,
            "J_rho": J_rho,
        }

        for name, arr in structural_raw.items():
            arr = _as_1d_float(arr, name=name)

            if cfg.use_log_structural_features:
                feat = _safe_log10_standardize(arr, eps=float(cfg.eps))
                feat_name = f"log10_{name}_z"
            else:
                feat = _standardize(arr, eps=float(cfg.eps))
                feat_name = f"{name}_z"

            features.append(feat)
            feature_names.append(feat_name)

    if not features:
        raise ValueError("No geometry features were enabled.")

    feature_matrix = np.stack(features, axis=-1).astype(np.float64)

    if not np.all(np.isfinite(feature_matrix)):
        raise ValueError("geometry feature_matrix contains non-finite values.")

    raw_fields = {
        "eta_node": eta_node,
        "span_node_m": span_node_m,
        "s_over_L": s_over_L,
        "phi_node_deg": phi_node_deg,
        "mass_per_length_kgpm": mass,
        "EA_N": EA,
        "EI_flap_Nm2": EI_flap,
        "EI_edge_Nm2": EI_edge,
        "GJ_Nm2": GJ,
        "J_rho_kgm": J_rho,
    }

    metadata = {
        "config": asdict(cfg),
        "blade_csv": str(blade_csv),
        "model_name": model.model_name,
        "span_m": float(model.span_m),
        "n_stations": int(model.n_stations),
        "n_free_nodes": int(feature_matrix.shape[0]),
        "feature_dim": int(feature_matrix.shape[1]),
        "feature_names": list(feature_names),
        "node_mapping": (
            "features row i corresponds to free node station i+1 "
            "when exclude_root_station=True; each node has 6 DOF in response vector."
        ),
    }

    return BladeGeometryFeatureBundle(
        features=feature_matrix,
        feature_names=feature_names,
        raw_fields=raw_fields,
        metadata=metadata,
    )