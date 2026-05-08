from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from src.student.io import load_student_model_from_blade_master  # noqa: E402
from src.student.coupled_fem_builder import build_coupled_fem_matrices_6dof_degrees  # noqa: E402
from src.student.torch_coupled_fem_builder import (  # noqa: E402
    TorchCoupledFEMBuilder,
    TorchPhiCouplingBasisConfig,
    get_or_build_phi_coupling_basis,
)


@dataclass
class TorchCoupledFEMEquivalenceConfig:
    blade_csv: str = str(PROJECT_ROOT / "data" / "raw" / "nrel5mw" / "blade_master.csv")
    output_dir: str = str(PROJECT_ROOT / "results" / "student" / "torch_coupled_fem_builder_test")
    basis_cache_name: str = "torch_phi_trig_basis.npz"

    model_name: str = "torch_coupled_fem_equivalence_test"

    alpha_flap: float = 1.0
    alpha_edge: float = 1.0
    alpha_torsion: float = 1.0
    phi_basis_eps_deg: float = 5.0
    test_phi_deg: float = -7.5

    torch_dtype: str = "float64"
    device: str = "cpu"

    atol_M_zero: float = 1e-12
    atol_K_zero: float = 1e-8
    rtol_K_phi: float = 1e-4

    rebuild_basis: bool = False
    save_report: bool = True


def parse_args() -> TorchCoupledFEMEquivalenceConfig:
    d = TorchCoupledFEMEquivalenceConfig()
    parser = argparse.ArgumentParser(description="Test differentiable Torch K(phi) builder against NumPy coupled FEM builder.")

    parser.add_argument("--blade-csv", type=str, default=d.blade_csv)
    parser.add_argument("--output-dir", type=str, default=d.output_dir)
    parser.add_argument("--basis-cache-name", type=str, default=d.basis_cache_name)
    parser.add_argument("--model-name", type=str, default=d.model_name)

    parser.add_argument("--alpha-flap", type=float, default=d.alpha_flap)
    parser.add_argument("--alpha-edge", type=float, default=d.alpha_edge)
    parser.add_argument("--alpha-torsion", type=float, default=d.alpha_torsion)
    parser.add_argument("--phi-basis-eps-deg", type=float, default=d.phi_basis_eps_deg)
    parser.add_argument("--test-phi-deg", type=float, default=d.test_phi_deg)

    parser.add_argument("--torch-dtype", type=str, default=d.torch_dtype, choices=["float64", "float32"])
    parser.add_argument("--device", type=str, default=d.device)

    parser.add_argument("--atol-M-zero", type=float, default=d.atol_M_zero)
    parser.add_argument("--atol-K-zero", type=float, default=d.atol_K_zero)
    parser.add_argument("--rtol-K-phi", type=float, default=d.rtol_K_phi)

    parser.add_argument("--rebuild-basis", action="store_true", default=d.rebuild_basis)

    save_group = parser.add_mutually_exclusive_group()
    save_group.add_argument("--save-report", dest="save_report", action="store_true")
    save_group.add_argument("--no-save-report", dest="save_report", action="store_false")
    parser.set_defaults(save_report=d.save_report)

    args = parser.parse_args()

    return TorchCoupledFEMEquivalenceConfig(
        blade_csv=args.blade_csv,
        output_dir=args.output_dir,
        basis_cache_name=args.basis_cache_name,
        model_name=args.model_name,
        alpha_flap=args.alpha_flap,
        alpha_edge=args.alpha_edge,
        alpha_torsion=args.alpha_torsion,
        phi_basis_eps_deg=args.phi_basis_eps_deg,
        test_phi_deg=args.test_phi_deg,
        torch_dtype=args.torch_dtype,
        device=args.device,
        atol_M_zero=args.atol_M_zero,
        atol_K_zero=args.atol_K_zero,
        rtol_K_phi=args.rtol_K_phi,
        rebuild_basis=args.rebuild_basis,
        save_report=args.save_report,
    )


def get_dtype(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def matrix_stats(A: np.ndarray, B: np.ndarray) -> dict[str, float]:
    diff = np.asarray(A) - np.asarray(B)
    denom = max(float(np.linalg.norm(B, ord="fro")), 1e-30)
    return {
        "mae": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "relative_fro": float(np.linalg.norm(diff, ord="fro") / denom),
    }


def save_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def safe(x):
        if isinstance(x, dict):
            return {str(k): safe(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [safe(v) for v in x]
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        return x

    with open(path, "w", encoding="utf-8") as f:
        json.dump(safe(obj), f, indent=2, ensure_ascii=False)


def main() -> None:
    cfg = parse_args()

    output_dir = Path(cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    basis_cache_path = output_dir / cfg.basis_cache_name

    print()
    print("[Torch Coupled FEM Builder Equivalence Test]")
    print()
    print("[Config]")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")

    print()
    print("[1/4] Loading StudentBeamModel")
    model = load_student_model_from_blade_master(
        csv_path=cfg.blade_csv,
        model_name=cfg.model_name,
    )
    n_elements = int(model.n_stations - 1)
    print(f"  n_stations = {model.n_stations}")
    print(f"  n_elements = {n_elements}")

    print()
    print("[2/4] Building/loading Torch trig phi basis")
    basis = get_or_build_phi_coupling_basis(
        model=model,
        cache_path=basis_cache_path,
        config=TorchPhiCouplingBasisConfig(
            phi_basis_eps_deg=cfg.phi_basis_eps_deg,
            alpha_flap=cfg.alpha_flap,
            alpha_edge=cfg.alpha_edge,
            alpha_torsion=cfg.alpha_torsion,
            rotate_mass=False,
        ),
        rebuild=cfg.rebuild_basis,
        verbose=True,
    )

    builder = TorchCoupledFEMBuilder(
        M0=basis["M0"],
        K0=basis["K0"],
        A=basis["A"],
        B=basis["B"],
        dtype=get_dtype(cfg.torch_dtype),
        device=torch.device(cfg.device),
    )

    print()
    print("[3/4] Matrix equivalence checks")

    phi_zero_np = np.zeros(n_elements, dtype=np.float64)
    M_np_zero, K_np_zero, _ = build_coupled_fem_matrices_6dof_degrees(
        model,
        section_params=None,
        phi_deg=phi_zero_np,
        alpha_flap=cfg.alpha_flap,
        alpha_edge=cfg.alpha_edge,
        alpha_torsion=cfg.alpha_torsion,
        rotate_mass=False,
        return_full=True,
    )

    phi_zero_t = torch.zeros(n_elements, dtype=get_dtype(cfg.torch_dtype), device=cfg.device)
    M_t_zero, K_t_zero = builder.assemble(phi_zero_t)

    M_zero_stats = matrix_stats(M_t_zero.detach().cpu().numpy(), M_np_zero)
    K_zero_stats = matrix_stats(K_t_zero.detach().cpu().numpy(), K_np_zero)

    phi_test_np = np.full(n_elements, cfg.test_phi_deg, dtype=np.float64)
    _, K_np_phi, _ = build_coupled_fem_matrices_6dof_degrees(
        model,
        section_params=None,
        phi_deg=phi_test_np,
        alpha_flap=cfg.alpha_flap,
        alpha_edge=cfg.alpha_edge,
        alpha_torsion=cfg.alpha_torsion,
        rotate_mass=False,
        return_full=True,
    )

    phi_test_t = torch.full(
        (n_elements,),
        cfg.test_phi_deg,
        dtype=get_dtype(cfg.torch_dtype),
        device=cfg.device,
        requires_grad=True,
    )
    _, K_t_phi = builder.assemble(phi_test_t)
    K_phi_stats = matrix_stats(K_t_phi.detach().cpu().numpy(), K_np_phi)

    print("[phi=0]")
    print(f"  M max_abs = {M_zero_stats['max_abs']:.12e}")
    print(f"  K max_abs = {K_zero_stats['max_abs']:.12e}")

    print(f"[phi={cfg.test_phi_deg} deg]")
    print(f"  K mae          = {K_phi_stats['mae']:.12e}")
    print(f"  K max_abs      = {K_phi_stats['max_abs']:.12e}")
    print(f"  K relative_fro = {K_phi_stats['relative_fro']:.12e}")

    print()
    print("[4/4] Autograd smoke check")
    loss = torch.mean(K_t_phi**2)
    loss.backward()

    grad = phi_test_t.grad.detach().cpu().numpy()
    grad_finite = bool(np.all(np.isfinite(grad)))
    grad_norm = float(np.linalg.norm(grad))

    print(f"  grad finite = {grad_finite}")
    print(f"  grad norm   = {grad_norm:.12e}")

    checks = {
        "M_zero_matches": M_zero_stats["max_abs"] <= cfg.atol_M_zero,
        "K_zero_matches": K_zero_stats["max_abs"] <= cfg.atol_K_zero,
        "K_phi_close": K_phi_stats["relative_fro"] <= cfg.rtol_K_phi,
        "grad_finite": grad_finite,
        "grad_nonzero": grad_norm > 0.0,
    }

    print()
    print("[Checks]")
    for k, v in checks.items():
        print(f"  {k:<24s}: {'PASS' if v else 'FAIL'}")

    report = {
        "passed": bool(all(checks.values())),
        "config": asdict(cfg),
        "basis_cache_path": str(basis_cache_path),
        "M_zero_stats": M_zero_stats,
        "K_zero_stats": K_zero_stats,
        "K_phi_stats": K_phi_stats,
        "grad_norm": grad_norm,
        "checks": checks,
    }

    if cfg.save_report:
        report_path = output_dir / "torch_coupled_fem_builder_equivalence_report.json"
        save_json(report_path, report)
        print()
        print(f"[Saved Report] {report_path}")

    if not all(checks.values()):
        print()
        print("❌ FAIL: Torch coupled FEM builder equivalence test failed.")
        raise SystemExit(1)

    print()
    print("✅ PASS: Torch coupled FEM builder matches baseline and supports autograd.")


if __name__ == "__main__":
    main()