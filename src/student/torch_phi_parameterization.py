from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class TrainableElementPhiConfig:
    n_elements: int = 48
    phi_max_deg: float = 15.0

    # 默认采用 root-heavy 主轴旋转初始化：
    # root = -15°, tip = 0°
    init_mode: str = "linear_root_to_tip"
    init_uniform_phi_deg: float = -7.5
    init_root_phi_deg: float = -15.0
    init_tip_phi_deg: float = 0.0


def safe_atanh_np(x: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -1.0 + eps, 1.0 - eps)
    return np.arctanh(x)


def make_initial_phi_deg(
    *,
    n_elements: int,
    mode: str,
    uniform_phi_deg: float = -7.5,
    root_phi_deg: float = -10.0,
    tip_phi_deg: float = 0.0,
) -> np.ndarray:
    n_elements = int(n_elements)
    if n_elements <= 0:
        raise ValueError(f"n_elements must be positive, got {n_elements}")

    mode = str(mode).strip().lower()

    if mode == "zeros":
        return np.zeros(n_elements, dtype=np.float64)

    if mode == "uniform":
        return np.full(n_elements, float(uniform_phi_deg), dtype=np.float64)

    if mode == "uniform_neg7p5":
        return np.full(n_elements, -7.5, dtype=np.float64)

    if mode == "linear_root_to_tip":
        eta = np.linspace(0.0, 1.0, n_elements)
        return float(root_phi_deg) + (float(tip_phi_deg) - float(root_phi_deg)) * eta

    raise ValueError(
        f"Unsupported init_mode={mode!r}. "
        "Use zeros, uniform, uniform_neg7p5, or linear_root_to_tip."
    )


def phi_deg_to_raw_np(phi_deg: np.ndarray, *, phi_max_deg: float) -> np.ndarray:
    phi_deg = np.asarray(phi_deg, dtype=np.float64)
    ratio = phi_deg / float(phi_max_deg)
    return safe_atanh_np(ratio)


def raw_to_phi_deg_torch(raw_phi: torch.Tensor, *, phi_max_deg: float) -> torch.Tensor:
    return float(phi_max_deg) * torch.tanh(raw_phi)


class TrainableElementPhi(nn.Module):
    """
    48 个 element-level phi_e 参数。

    phi_e 属于 κ 组截面耦合刚度修正。
    它不直接依赖 u / u_dot / u_ddot；
    它通过 K(phi) 改变 full-order PDE 的刚度矩阵。
    """

    def __init__(
        self,
        *,
        n_elements: int = 48,
        phi_max_deg: float = 15.0,
        init_mode: str = "uniform_neg7p5",
        init_uniform_phi_deg: float = -7.5,
        init_root_phi_deg: float = -10.0,
        init_tip_phi_deg: float = 0.0,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()

        self.n_elements = int(n_elements)
        self.phi_max_deg = float(phi_max_deg)

        phi_init = make_initial_phi_deg(
            n_elements=self.n_elements,
            mode=init_mode,
            uniform_phi_deg=init_uniform_phi_deg,
            root_phi_deg=init_root_phi_deg,
            tip_phi_deg=init_tip_phi_deg,
        )
        raw_init = phi_deg_to_raw_np(phi_init, phi_max_deg=self.phi_max_deg)

        raw_tensor = torch.as_tensor(raw_init, dtype=dtype, device=device)
        self.raw_phi = nn.Parameter(raw_tensor)

    def phi_deg(self) -> torch.Tensor:
        return raw_to_phi_deg_torch(self.raw_phi, phi_max_deg=self.phi_max_deg)

    def phi_rad(self) -> torch.Tensor:
        return self.phi_deg() * (torch.pi / 180.0)

    def regularization(
        self,
        *,
        lambda_smooth: float = 0.1,
        lambda_curvature: float = 0.01,
        lambda_magnitude: float = 0.001,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        phi = self.phi_deg()
        scale = max(abs(self.phi_max_deg), 1e-12)
        phi_scaled = phi / scale

        if phi_scaled.numel() >= 2:
            d1 = phi_scaled[1:] - phi_scaled[:-1]
            smooth_l2 = torch.mean(d1**2)
        else:
            smooth_l2 = torch.zeros((), dtype=phi.dtype, device=phi.device)

        if phi_scaled.numel() >= 3:
            d2 = phi_scaled[2:] - 2.0 * phi_scaled[1:-1] + phi_scaled[:-2]
            curvature_l2 = torch.mean(d2**2)
        else:
            curvature_l2 = torch.zeros((), dtype=phi.dtype, device=phi.device)

        magnitude_l2 = torch.mean(phi_scaled**2)

        loss = (
            float(lambda_smooth) * smooth_l2
            + float(lambda_curvature) * curvature_l2
            + float(lambda_magnitude) * magnitude_l2
        )

        with torch.no_grad():
            metrics = {
                "phi_min_deg": float(torch.min(phi).detach().cpu()),
                "phi_max_deg": float(torch.max(phi).detach().cpu()),
                "phi_mean_deg": float(torch.mean(phi).detach().cpu()),
                "phi_std_deg": float(torch.std(phi).detach().cpu()),
                "smooth_l2": float(smooth_l2.detach().cpu()),
                "curvature_l2": float(curvature_l2.detach().cpu()),
                "magnitude_l2": float(magnitude_l2.detach().cpu()),
            }

        return loss, metrics

    def summary(self) -> Dict[str, float]:
        with torch.no_grad():
            phi = self.phi_deg().detach().cpu()
            return {
                "n_elements": self.n_elements,
                "phi_max_deg_limit": self.phi_max_deg,
                "phi_min_deg": float(torch.min(phi)),
                "phi_max_deg": float(torch.max(phi)),
                "phi_mean_deg": float(torch.mean(phi)),
                "phi_std_deg": float(torch.std(phi)),
            }


def save_phi_module_npz(
    path: str | Path,
    *,
    phi_module: TrainableElementPhi,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        raw_phi = phi_module.raw_phi.detach().cpu().numpy()
        phi_deg = phi_module.phi_deg().detach().cpu().numpy()

    np.savez_compressed(
        path,
        raw_phi=raw_phi,
        phi_deg=phi_deg,
        metadata=np.array([metadata or {}], dtype=object),
    )


def load_phi_npz(path: str | Path) -> Dict[str, Any]:
    data = np.load(Path(path), allow_pickle=True)
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