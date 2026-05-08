from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class TrainablePhiVectorConfig:
    """
    48 个 element-level phi_e 的参数配置。

    注意：
    - phi_e 属于 κ 组截面耦合刚度修正；
    - phi_e 不依赖 u / u_dot / u_ddot；
    - phi_e 先进入 FEM 装配得到 K(phi)，响应仍由 PDE / Newmark 解出。
    """

    n_elements: int = 48
    phi_max_deg: float = 15.0
    raw_clip: float = 4.0


def safe_atanh(x: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -1.0 + eps, 1.0 - eps)
    return np.arctanh(x)


def raw_to_phi_deg(
    raw_phi: np.ndarray,
    *,
    phi_max_deg: float,
) -> np.ndarray:
    """
    raw_phi -> bounded phi_deg

    phi_deg = phi_max_deg * tanh(raw_phi)
    """
    raw_phi = np.asarray(raw_phi, dtype=np.float64)
    return float(phi_max_deg) * np.tanh(raw_phi)


def phi_deg_to_raw(
    phi_deg: np.ndarray,
    *,
    phi_max_deg: float,
) -> np.ndarray:
    """
    bounded phi_deg -> raw_phi initialization.
    """
    phi_deg = np.asarray(phi_deg, dtype=np.float64)
    ratio = phi_deg / float(phi_max_deg)
    return safe_atanh(ratio)


def make_initial_phi_deg(
    *,
    n_elements: int,
    mode: str = "uniform_neg7p5",
    uniform_phi_deg: float = -7.5,
    root_phi_deg: float = -10.0,
    tip_phi_deg: float = 0.0,
) -> np.ndarray:
    """
    构造初始 element-level phi_e，单位 degree。

    mode:
        zeros:
            全 0，用于 baseline 回退测试。

        uniform:
            全部设为 uniform_phi_deg。

        uniform_neg7p5:
            全部设为 -7.5°，来自 signed scan 的最佳时间序列初值。

        linear_root_to_tip:
            root_phi_deg -> tip_phi_deg 线性分布。
            默认 -10° -> 0°，来自 ratio scan 的 root-heavy 最佳候选。
    """
    n_elements = int(n_elements)
    if n_elements <= 0:
        raise ValueError(f"n_elements must be positive, got {n_elements}.")

    mode = str(mode).strip().lower()

    if mode == "zeros":
        return np.zeros(n_elements, dtype=np.float64)

    if mode == "uniform":
        return np.full(n_elements, float(uniform_phi_deg), dtype=np.float64)

    if mode == "uniform_neg7p5":
        return np.full(n_elements, -7.5, dtype=np.float64)

    if mode == "linear_root_to_tip":
        eta_elem = np.linspace(0.0, 1.0, n_elements)
        return float(root_phi_deg) + (float(tip_phi_deg) - float(root_phi_deg)) * eta_elem

    raise ValueError(
        f"Unsupported init mode={mode!r}. "
        "Use zeros, uniform, uniform_neg7p5, or linear_root_to_tip."
    )


def make_initial_raw_phi(
    *,
    n_elements: int,
    phi_max_deg: float,
    mode: str = "uniform_neg7p5",
    uniform_phi_deg: float = -7.5,
    root_phi_deg: float = -10.0,
    tip_phi_deg: float = 0.0,
) -> np.ndarray:
    phi_deg = make_initial_phi_deg(
        n_elements=n_elements,
        mode=mode,
        uniform_phi_deg=uniform_phi_deg,
        root_phi_deg=root_phi_deg,
        tip_phi_deg=tip_phi_deg,
    )
    return phi_deg_to_raw(phi_deg, phi_max_deg=phi_max_deg)


def phi_regularization_metrics(
    phi_deg: np.ndarray,
    *,
    phi_max_deg: float,
) -> Dict[str, float]:
    """
    返回用于训练监控和正则项的指标。

    所有 smooth / curvature / magnitude 都做了无量纲化。
    """
    phi_deg = np.asarray(phi_deg, dtype=np.float64).reshape(-1)
    scale = max(abs(float(phi_max_deg)), 1e-12)

    phi_scaled = phi_deg / scale

    if phi_scaled.size >= 2:
        first_diff = np.diff(phi_scaled)
        smooth_l2 = float(np.mean(first_diff**2))
        smooth_l1 = float(np.mean(np.abs(first_diff)))
    else:
        smooth_l2 = 0.0
        smooth_l1 = 0.0

    if phi_scaled.size >= 3:
        second_diff = phi_scaled[2:] - 2.0 * phi_scaled[1:-1] + phi_scaled[:-2]
        curvature_l2 = float(np.mean(second_diff**2))
        curvature_l1 = float(np.mean(np.abs(second_diff)))
    else:
        curvature_l2 = 0.0
        curvature_l1 = 0.0

    magnitude_l2 = float(np.mean(phi_scaled**2))
    magnitude_l1 = float(np.mean(np.abs(phi_scaled)))

    return {
        "phi_min_deg": float(np.min(phi_deg)),
        "phi_max_deg": float(np.max(phi_deg)),
        "phi_mean_deg": float(np.mean(phi_deg)),
        "phi_std_deg": float(np.std(phi_deg)),
        "smooth_l2": smooth_l2,
        "smooth_l1": smooth_l1,
        "curvature_l2": curvature_l2,
        "curvature_l1": curvature_l1,
        "magnitude_l2": magnitude_l2,
        "magnitude_l1": magnitude_l1,
    }


def phi_regularization_loss(
    phi_deg: np.ndarray,
    *,
    phi_max_deg: float,
    lambda_smooth: float = 0.1,
    lambda_curvature: float = 0.01,
    lambda_magnitude: float = 0.001,
) -> tuple[float, Dict[str, float]]:
    metrics = phi_regularization_metrics(phi_deg, phi_max_deg=phi_max_deg)

    loss = (
        float(lambda_smooth) * metrics["smooth_l2"]
        + float(lambda_curvature) * metrics["curvature_l2"]
        + float(lambda_magnitude) * metrics["magnitude_l2"]
    )

    return float(loss), metrics


def clip_raw_phi(
    raw_phi: np.ndarray,
    *,
    raw_clip: float,
) -> np.ndarray:
    raw_phi = np.asarray(raw_phi, dtype=np.float64)
    return np.clip(raw_phi, -float(raw_clip), float(raw_clip))


def save_phi_profile_npz(
    path: str | Path,
    *,
    raw_phi: np.ndarray,
    phi_deg: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    raw_phi = np.asarray(raw_phi, dtype=np.float64).reshape(-1)
    phi_deg = np.asarray(phi_deg, dtype=np.float64).reshape(-1)

    if raw_phi.shape != phi_deg.shape:
        raise ValueError(f"raw_phi/phi_deg shape mismatch: {raw_phi.shape} vs {phi_deg.shape}")

    meta_obj = np.array([metadata or {}], dtype=object)

    np.savez_compressed(
        path,
        raw_phi=raw_phi,
        phi_deg=phi_deg,
        metadata=meta_obj,
    )


def load_phi_profile_npz(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    data = np.load(path, allow_pickle=True)

    metadata = {}
    if "metadata" in data:
        arr = data["metadata"]
        if arr.size > 0:
            metadata = dict(arr.reshape(-1)[0])

    return {
        "raw_phi": np.asarray(data["raw_phi"], dtype=np.float64),
        "phi_deg": np.asarray(data["phi_deg"], dtype=np.float64),
        "metadata": metadata,
    }