from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.student.io import load_student_model_from_blade_master  # noqa: E402
from src.student.fem_builder import build_fem_matrices_6dof  # noqa: E402
from src.student.corrected_fem_builder import (  # noqa: E402
    build_corrected_fem_matrices_6dof,
    summarize_fem_matrices,
)


@dataclass
class CorrectedFemBuilderEquivalenceConfig:
    blade_csv: str = str(PROJECT_ROOT / "data" / "raw" / "nrel5mw" / "blade_master.csv")
    output_dir: str = str(PROJECT_ROOT / "results" / "student" / "corrected_fem_builder_test")
    model_name: str = "corrected_fem_builder_baseline_test"

    alpha_flap: float = 1.0
    alpha_edge: float = 1.0
    alpha_torsion: float = 1.0

    atol_M: float = 1e-12
    rtol_M: float = 1e-12
    atol_K: float = 1e-8
    rtol_K: float = 1e-12

    save_report: bool = True


def parse_args() -> CorrectedFemBuilderEquivalenceConfig:
    default = CorrectedFemBuilderEquivalenceConfig()

    parser = argparse.ArgumentParser(
        description=(
            "Test whether corrected_fem_builder.py reproduces "
            "src/student/fem_builder.py baseline M/K matrices."
        )
    )

    parser.add_argument("--blade-csv", type=str, default=default.blade_csv)
    parser.add_argument("--output-dir", type=str, default=default.output_dir)
    parser.add_argument("--model-name", type=str, default=default.model_name)

    parser.add_argument("--alpha-flap", type=float, default=default.alpha_flap)
    parser.add_argument("--alpha-edge", type=float, default=default.alpha_edge)
    parser.add_argument("--alpha-torsion", type=float, default=default.alpha_torsion)

    parser.add_argument("--atol-M", type=float, default=default.atol_M)
    parser.add_argument("--rtol-M", type=float, default=default.rtol_M)
    parser.add_argument("--atol-K", type=float, default=default.atol_K)
    parser.add_argument("--rtol-K", type=float, default=default.rtol_K)

    save_group = parser.add_mutually_exclusive_group()
    save_group.add_argument("--save-report", dest="save_report", action="store_true")
    save_group.add_argument("--no-save-report", dest="save_report", action="store_false")
    parser.set_defaults(save_report=default.save_report)

    args = parser.parse_args()

    return CorrectedFemBuilderEquivalenceConfig(
        blade_csv=args.blade_csv,
        output_dir=args.output_dir,
        model_name=args.model_name,
        alpha_flap=args.alpha_flap,
        alpha_edge=args.alpha_edge,
        alpha_torsion=args.alpha_torsion,
        atol_M=args.atol_M,
        rtol_M=args.rtol_M,
        atol_K=args.atol_K,
        rtol_K=args.rtol_K,
        save_report=args.save_report,
    )


def make_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    return obj


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(obj), f, indent=2, ensure_ascii=False)


def compare_matrix(
    name: str,
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> Tuple[bool, Dict[str, float]]:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)

    if actual.shape != expected.shape:
        return False, {
            "shape_match": 0.0,
            "actual_shape_0": float(actual.shape[0]),
            "actual_shape_1": float(actual.shape[1]),
            "expected_shape_0": float(expected.shape[0]),
            "expected_shape_1": float(expected.shape[1]),
            "mae": float("inf"),
            "max_abs": float("inf"),
            "relative_fro": float("inf"),
        }

    diff = actual - expected
    abs_diff = np.abs(diff)

    mae = float(np.mean(abs_diff))
    max_abs = float(np.max(abs_diff))
    denom = float(np.linalg.norm(expected, ord="fro"))
    relative_fro = float(np.linalg.norm(diff, ord="fro") / max(denom, 1e-30))

    passed = bool(np.allclose(actual, expected, atol=atol, rtol=rtol))

    return passed, {
        "shape_match": 1.0,
        "mae": mae,
        "max_abs": max_abs,
        "relative_fro": relative_fro,
    }


def main() -> None:
    cfg = parse_args()

    blade_csv = Path(cfg.blade_csv).resolve()
    output_dir = Path(cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not blade_csv.exists():
        raise FileNotFoundError(f"blade_csv not found: {blade_csv}")

    print()
    print("[Corrected FEM Builder Baseline Equivalence Test]")
    print()
    print("[Config]")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")

    print()
    print("[1/4] Loading StudentBeamModel")
    model = load_student_model_from_blade_master(
        csv_path=str(blade_csv),
        model_name=cfg.model_name,
    )

    print(f"  model_name = {model.model_name}")
    print(f"  span_m = {model.span_m}")
    print(f"  n_stations = {model.n_stations}")
    print(f"  n_free_nodes = {model.n_stations - 1}")
    print(f"  expected free dofs = {(model.n_stations - 1) * 6}")

    print()
    print("[2/4] Building old baseline M/K via fem_builder.py")
    M_old, K_old = build_fem_matrices_6dof(
        model,
        alpha_flap=cfg.alpha_flap,
        alpha_edge=cfg.alpha_edge,
        alpha_torsion=cfg.alpha_torsion,
    )

    print(f"  M_old shape = {M_old.shape}")
    print(f"  K_old shape = {K_old.shape}")

    print()
    print("[3/4] Building new M/K via corrected_fem_builder.py")
    M_new, K_new, debug_info = build_corrected_fem_matrices_6dof(
        model,
        section_params=None,
        alpha_flap=cfg.alpha_flap,
        alpha_edge=cfg.alpha_edge,
        alpha_torsion=cfg.alpha_torsion,
        return_full=True,
    )

    print(f"  M_new shape = {M_new.shape}")
    print(f"  K_new shape = {K_new.shape}")
    print(f"  section_params_source = {debug_info['section_params_source']}")

    print()
    print("[4/4] Comparing M/K matrices")
    M_passed, M_metrics = compare_matrix(
        "M",
        actual=M_new,
        expected=M_old,
        atol=cfg.atol_M,
        rtol=cfg.rtol_M,
    )
    K_passed, K_metrics = compare_matrix(
        "K",
        actual=K_new,
        expected=K_old,
        atol=cfg.atol_K,
        rtol=cfg.rtol_K,
    )

    print(
        f"  M {'PASS' if M_passed else 'FAIL'} | "
        f"mae={M_metrics['mae']:.12e}, "
        f"max_abs={M_metrics['max_abs']:.12e}, "
        f"relative_fro={M_metrics['relative_fro']:.12e}"
    )
    print(
        f"  K {'PASS' if K_passed else 'FAIL'} | "
        f"mae={K_metrics['mae']:.12e}, "
        f"max_abs={K_metrics['max_abs']:.12e}, "
        f"relative_fro={K_metrics['relative_fro']:.12e}"
    )

    all_passed = bool(M_passed and K_passed)

    matrix_summary = {
        "old": summarize_fem_matrices(M_old, K_old),
        "new": summarize_fem_matrices(M_new, K_new),
    }

    report = {
        "passed": all_passed,
        "checks": {
            "M": bool(M_passed),
            "K": bool(K_passed),
        },
        "metrics": {
            "M": M_metrics,
            "K": K_metrics,
        },
        "config": asdict(cfg),
        "blade_csv": str(blade_csv),
        "model_info": {
            "model_name": model.model_name,
            "span_m": float(model.span_m),
            "n_stations": int(model.n_stations),
            "n_free_nodes": int(model.n_stations - 1),
            "n_dofs_free": int(M_old.shape[0]),
        },
        "matrix_summary": matrix_summary,
        "debug_summary": {
            "builder": debug_info["builder"],
            "n_nodes": debug_info["n_nodes"],
            "n_free_nodes": debug_info["n_free_nodes"],
            "n_dofs_global": debug_info["n_dofs_global"],
            "n_dofs_free": debug_info["n_dofs_free"],
            "alpha_flap": debug_info["alpha_flap"],
            "alpha_edge": debug_info["alpha_edge"],
            "alpha_torsion": debug_info["alpha_torsion"],
            "section_params_source": debug_info["section_params_source"],
        },
    }

    if cfg.save_report:
        report_path = output_dir / "corrected_fem_builder_baseline_equivalence_report.json"
        save_json(report_path, report)
        print()
        print(f"[Saved Report] {report_path}")

    print()
    if all_passed:
        print("✅ PASS: corrected_fem_builder.py reproduces fem_builder.py baseline M/K.")
    else:
        print("❌ FAIL: corrected_fem_builder.py does not match fem_builder.py baseline M/K.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()