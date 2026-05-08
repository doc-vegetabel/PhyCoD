from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn


from src.student.coupled_fem_builder import build_coupled_fem_matrices_6dof_degrees


@dataclass(frozen=True)
class TorchPhiCouplingBasisConfig:
    phi_basis_eps_deg: float = 5.0
    alpha_flap: float = 1.0
    alpha_edge: float = 1.0
    alpha_torsion: float = 1.0
    rotate_mass: bool = False


def _to_numpy_matrix(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def build_numpy_trig_phi_coupling_basis(
    model,
    *,
    config: TorchPhiCouplingBasisConfig,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    为每个 element 构造可微 trig basis。

    假设每个单元的主轴旋转效应可以表示为：

        K(phi) = K0
               + A_e * (cos(2 phi_e) - 1)
               + B_e * sin(2 phi_e)

    这与二维主轴旋转刚度矩阵的形式一致。
    phi=0 时严格回退 K0。
    """

    n_elements = int(model.n_stations - 1)
    eps_deg = float(config.phi_basis_eps_deg)
    eps_rad = np.deg2rad(eps_deg)

    if abs(np.sin(2.0 * eps_rad)) < 1e-12:
        raise ValueError("phi_basis_eps_deg leads to near-zero sin(2eps).")

    phi_zero = np.zeros(n_elements, dtype=np.float64)

    M0, K0, debug0 = build_coupled_fem_matrices_6dof_degrees(
        model,
        section_params=None,
        phi_deg=phi_zero,
        alpha_flap=config.alpha_flap,
        alpha_edge=config.alpha_edge,
        alpha_torsion=config.alpha_torsion,
        rotate_mass=config.rotate_mass,
        return_full=True,
    )

    M0 = _to_numpy_matrix(M0)
    K0 = _to_numpy_matrix(K0)

    n_dofs = int(K0.shape[0])
    A = np.zeros((n_elements, n_dofs, n_dofs), dtype=np.float64)
    B = np.zeros((n_elements, n_dofs, n_dofs), dtype=np.float64)

    denom_A = 2.0 * (np.cos(2.0 * eps_rad) - 1.0)
    denom_B = 2.0 * np.sin(2.0 * eps_rad)

    for e in range(n_elements):
        if verbose:
            print(f"  [basis] element {e + 1:02d}/{n_elements:02d}")

        phi_p = np.zeros(n_elements, dtype=np.float64)
        phi_m = np.zeros(n_elements, dtype=np.float64)
        phi_p[e] = eps_deg
        phi_m[e] = -eps_deg

        _, Kp, _ = build_coupled_fem_matrices_6dof_degrees(
            model,
            section_params=None,
            phi_deg=phi_p,
            alpha_flap=config.alpha_flap,
            alpha_edge=config.alpha_edge,
            alpha_torsion=config.alpha_torsion,
            rotate_mass=config.rotate_mass,
            return_full=True,
        )
        _, Km, _ = build_coupled_fem_matrices_6dof_degrees(
            model,
            section_params=None,
            phi_deg=phi_m,
            alpha_flap=config.alpha_flap,
            alpha_edge=config.alpha_edge,
            alpha_torsion=config.alpha_torsion,
            rotate_mass=config.rotate_mass,
            return_full=True,
        )

        Kp = _to_numpy_matrix(Kp)
        Km = _to_numpy_matrix(Km)

        A[e] = (Kp + Km - 2.0 * K0) / denom_A
        B[e] = (Kp - Km) / denom_B

    return {
        "M0": M0,
        "K0": K0,
        "A": A,
        "B": B,
        "n_elements": n_elements,
        "n_dofs": n_dofs,
        "config": {
            "phi_basis_eps_deg": config.phi_basis_eps_deg,
            "alpha_flap": config.alpha_flap,
            "alpha_edge": config.alpha_edge,
            "alpha_torsion": config.alpha_torsion,
            "rotate_mass": config.rotate_mass,
        },
        "debug0": debug0,
    }


def save_phi_coupling_basis_npz(path: str | Path, basis: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        path,
        M0=basis["M0"],
        K0=basis["K0"],
        A=basis["A"],
        B=basis["B"],
        n_elements=np.array([basis["n_elements"]], dtype=np.int64),
        n_dofs=np.array([basis["n_dofs"]], dtype=np.int64),
        config=np.array([basis.get("config", {})], dtype=object),
    )


def load_phi_coupling_basis_npz(path: str | Path) -> Dict[str, Any]:
    data = np.load(Path(path), allow_pickle=True)

    config = {}
    if "config" in data:
        arr = data["config"]
        if arr.size > 0:
            config = dict(arr.reshape(-1)[0])

    return {
        "M0": np.asarray(data["M0"], dtype=np.float64),
        "K0": np.asarray(data["K0"], dtype=np.float64),
        "A": np.asarray(data["A"], dtype=np.float64),
        "B": np.asarray(data["B"], dtype=np.float64),
        "n_elements": int(np.asarray(data["n_elements"]).reshape(-1)[0]),
        "n_dofs": int(np.asarray(data["n_dofs"]).reshape(-1)[0]),
        "config": config,
    }


class TorchCoupledFEMBuilder(nn.Module):
    """
    可微 K(phi) builder。

    M 暂时保持 baseline M0。
    K(phi) 使用 trig basis 构造：

        K = K0 + sum_e A_e * (cos(2 phi_e)-1) + B_e * sin(2 phi_e)

    phi_e 为 element-level，shape=(48,)。
    """

    def __init__(
        self,
        *,
        M0: np.ndarray,
        K0: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()

        M0_t = torch.as_tensor(M0, dtype=dtype, device=device)
        K0_t = torch.as_tensor(K0, dtype=dtype, device=device)
        A_t = torch.as_tensor(A, dtype=dtype, device=device)
        B_t = torch.as_tensor(B, dtype=dtype, device=device)

        if A_t.shape != B_t.shape:
            raise ValueError(f"A/B shape mismatch: {A_t.shape} vs {B_t.shape}")
        if A_t.ndim != 3:
            raise ValueError(f"A must have shape (n_elements,n_dofs,n_dofs), got {A_t.shape}")
        if K0_t.shape != A_t.shape[1:]:
            raise ValueError(f"K0 shape {K0_t.shape} incompatible with A {A_t.shape}")

        self.register_buffer("M0", M0_t)
        self.register_buffer("K0", K0_t)
        self.register_buffer("A", A_t)
        self.register_buffer("B", B_t)

    @classmethod
    def from_basis_npz(
        cls,
        path: str | Path,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ) -> "TorchCoupledFEMBuilder":
        basis = load_phi_coupling_basis_npz(path)
        return cls(
            M0=basis["M0"],
            K0=basis["K0"],
            A=basis["A"],
            B=basis["B"],
            dtype=dtype,
            device=device,
        )

    @property
    def n_elements(self) -> int:
        return int(self.A.shape[0])

    @property
    def n_dofs(self) -> int:
        return int(self.K0.shape[0])

    def assemble(self, phi_deg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if phi_deg.ndim != 1:
            raise ValueError(f"phi_deg must be 1D, got {phi_deg.shape}")
        if int(phi_deg.numel()) != self.n_elements:
            raise ValueError(
                f"phi_deg length mismatch: got {phi_deg.numel()}, expected {self.n_elements}"
            )

        phi_rad = phi_deg * (torch.pi / 180.0)

        cos_term = torch.cos(2.0 * phi_rad) - 1.0
        sin_term = torch.sin(2.0 * phi_rad)

        K = (
            self.K0
            + torch.einsum("e,eij->ij", cos_term, self.A)
            + torch.einsum("e,eij->ij", sin_term, self.B)
        )

        M = self.M0
        return M, K

    def summary(self) -> Dict[str, Any]:
        return {
            "builder": "TorchCoupledFEMBuilder",
            "n_elements": self.n_elements,
            "n_dofs": self.n_dofs,
            "basis_type": "trig_phi_basis",
        }


def get_or_build_phi_coupling_basis(
    *,
    model,
    cache_path: str | Path,
    config: TorchPhiCouplingBasisConfig,
    rebuild: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    cache_path = Path(cache_path)

    if cache_path.exists() and not rebuild:
        if verbose:
            print(f"[Load Torch Phi Basis] {cache_path}")
        return load_phi_coupling_basis_npz(cache_path)

    if verbose:
        print(f"[Build Torch Phi Basis] {cache_path}")

    basis = build_numpy_trig_phi_coupling_basis(
        model,
        config=config,
        verbose=verbose,
    )
    save_phi_coupling_basis_npz(cache_path, basis)
    return basis