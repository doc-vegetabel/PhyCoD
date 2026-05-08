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
from src.student.corrected_fem_builder import (  # noqa: E402
    build_corrected_fem_matrices_6dof,
)
from src.student.section_parameterization import (  # noqa: E402
    build_baseline_section_parameters,
    apply_section_parameter_corrections,
    make_uniform_section_correction,
)


@dataclass
class SectionCorrectionSanityConfig:
    blade_csv: str = str(PROJECT_ROOT / "data" / "raw" / "nrel5mw" / "blade_master.csv")
    output_dir: str = str(PROJECT_ROOT / "results" / "student" / "section_correction_sanity")
    model_name: str = "section_correction_sanity_test"

    alpha_flap: float = 1.0
    alpha_edge: float = 1.0
    alpha_torsion: float = 1.0

    # 只用于 sanity，不代表后续真实修正尺度
    delta_EI_flap_relative: float = 0.01
    delta_EI_edge_relative: float = 0.01
    delta_GJ_relative: float = 0.01
    delta_J_rho_relative: float = 0.01

    zero_atol_M: float = 1e-12
    zero_atol_K: float = 1e-8

    min_matrix_change: float = 1e-9

    save_report: bool = True


def parse_args() -> SectionCorrectionSanityConfig:
    default = SectionCorrectionSanityConfig()

    parser = argparse.ArgumentParser(
        description=(
            "Sanity test physical section-parameter corrections "
            "through corrected_fem_builder.py."
        )
    )

    parser.add_argument("--blade-csv", type=str, default=default.blade_csv)
    parser.add_argument("--output-dir", type=str, default=default.output_dir)
    parser.add_argument("--model-name", type=str, default=default.model_name)

    parser.add_argument("--alpha-flap", type=float, default=default.alpha_flap)
    parser.add_argument("--alpha-edge", type=float, default=default.alpha_edge)
    parser.add_argument("--alpha-torsion", type=float, default=default.alpha_torsion)

    parser.add_argument(
        "--delta-EI-flap-relative",
        type=float,
        default=default.delta_EI_flap_relative,
    )
    parser.add_argument(
        "--delta-EI-edge-relative",
        type=float,
        default=default.delta_EI_edge_relative,
    )
    parser.add_argument(
        "--delta-GJ-relative",
        type=float,
        default=default.delta_GJ_relative,
    )
    parser.add_argument(
        "--delta-J-rho-relative",
        type=float,
        default=default.delta_J_rho_relative,
    )

    parser.add_argument("--zero-atol-M", type=float, default=default.zero_atol_M)
    parser.add_argument("--zero-atol-K", type=float, default=default.zero_atol_K)
    parser.add_argument("--min-matrix-change", type=float, default=default.min_matrix_change)

    save_group = parser.add_mutually_exclusive_group()
    save_group.add_argument("--save-report", dest="save_report", action="store_true")
    save_group.add_argument("--no-save-report", dest="save_report", action="store_false")
    parser.set_defaults(save_report=default.save_report)

    args = parser.parse_args()

    return SectionCorrectionSanityConfig(
        blade_csv=args.blade_csv,
        output_dir=args.output_dir,
        model_name=args.model_name,
        alpha_flap=args.alpha_flap,
        alpha_edge=args.alpha_edge,
        alpha_torsion=args.alpha_torsion,
        delta_EI_flap_relative=args.delta_EI_flap_relative,
        delta_EI_edge_relative=args.delta_EI_edge_relative,
        delta_GJ_relative=args.delta_GJ_relative,
        delta_J_rho_relative=args.delta_J_rho_relative,
        zero_atol_M=args.zero_atol_M,
        zero_atol_K=args.zero_atol_K,
        min_matrix_change=args.min_matrix_change,
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


def matrix_diff_stats(A: np.ndarray, B: np.ndarray) -> Dict[str, float]:
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    diff = A - B
    abs_diff = np.abs(diff)

    denom = float(np.linalg.norm(B, ord="fro"))
    return {
        "mae": float(np.mean(abs_diff)),
        "max_abs": float(np.max(abs_diff)),
        "relative_fro": float(np.linalg.norm(diff, ord="fro") / max(denom, 1e-30)),
    }


def build_mk(model, section_params, cfg: SectionCorrectionSanityConfig) -> Tuple[np.ndarray, np.ndarray]:
    return build_corrected_fem_matrices_6dof(
        model,
        section_params=section_params,
        alpha_flap=cfg.alpha_flap,
        alpha_edge=cfg.alpha_edge,
        alpha_torsion=cfg.alpha_torsion,
        return_full=False,
    )


def print_case_result(
    name: str,
    M_stats: Dict[str, float],
    K_stats: Dict[str, float],
    checks: Dict[str, bool],
) -> None:
    print()
    print(f"[{name}]")
    print(
        f"  M diff: mae={M_stats['mae']:.12e}, "
        f"max_abs={M_stats['max_abs']:.12e}, "
        f"relative_fro={M_stats['relative_fro']:.12e}"
    )
    print(
        f"  K diff: mae={K_stats['mae']:.12e}, "
        f"max_abs={K_stats['max_abs']:.12e}, "
        f"relative_fro={K_stats['relative_fro']:.12e}"
    )
    for k, v in checks.items():
        print(f"  {k:<32s}: {'PASS' if v else 'FAIL'}")


def main() -> None:
    cfg = parse_args()

    blade_csv = Path(cfg.blade_csv).resolve()
    output_dir = Path(cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not blade_csv.exists():
        raise FileNotFoundError(f"blade_csv not found: {blade_csv}")

    print()
    print("[Section Correction Sanity Test]")
    print()
    print("[Config]")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")

    print()
    print("[1/5] Loading model")
    model = load_student_model_from_blade_master(
        csv_path=str(blade_csv),
        model_name=cfg.model_name,
    )
    print(f"  span_m = {model.span_m}")
    print(f"  n_stations = {model.n_stations}")
    print(f"  n_free_nodes = {model.n_stations - 1}")

    print()
    print("[2/5] Building baseline section parameters and baseline M/K")
    baseline_params = build_baseline_section_parameters(model)
    M_base, K_base = build_mk(model, baseline_params, cfg)

    print(f"  M_base shape = {M_base.shape}")
    print(f"  K_base shape = {K_base.shape}")
    print(f"  baseline source = {baseline_params.source}")

    results: Dict[str, Any] = {}

    print()
    print("[3/5] Zero correction check")
    zero_params = apply_section_parameter_corrections(
        baseline_params,
        correction=None,
    )
    M_zero, K_zero = build_mk(model, zero_params, cfg)

    M_zero_stats = matrix_diff_stats(M_zero, M_base)
    K_zero_stats = matrix_diff_stats(K_zero, K_base)
    zero_checks = {
        "M_unchanged": M_zero_stats["max_abs"] <= cfg.zero_atol_M,
        "K_unchanged": K_zero_stats["max_abs"] <= cfg.zero_atol_K,
    }
    print_case_result("zero_correction", M_zero_stats, K_zero_stats, zero_checks)

    results["zero_correction"] = {
        "M_stats": M_zero_stats,
        "K_stats": K_zero_stats,
        "checks": zero_checks,
        "source": zero_params.source,
    }

    print()
    print("[4/5] Stiffness correction checks")

    stiffness_cases = {
        "delta_EI_flap": make_uniform_section_correction(
            delta_EI_flap_relative=cfg.delta_EI_flap_relative,
        ),
        "delta_EI_edge": make_uniform_section_correction(
            delta_EI_edge_relative=cfg.delta_EI_edge_relative,
        ),
        "delta_GJ": make_uniform_section_correction(
            delta_GJ_relative=cfg.delta_GJ_relative,
        ),
    }

    for case_name, correction in stiffness_cases.items():
        params = apply_section_parameter_corrections(
            baseline_params,
            correction=correction,
            source_suffix=case_name,
        )
        M_corr, K_corr = build_mk(model, params, cfg)

        M_stats = matrix_diff_stats(M_corr, M_base)
        K_stats = matrix_diff_stats(K_corr, K_base)

        checks = {
            "M_unchanged": M_stats["max_abs"] <= cfg.zero_atol_M,
            "K_changed": K_stats["max_abs"] > cfg.min_matrix_change,
            "K_finite": np.isfinite(K_stats["max_abs"]),
        }

        print_case_result(case_name, M_stats, K_stats, checks)

        results[case_name] = {
            "M_stats": M_stats,
            "K_stats": K_stats,
            "checks": checks,
            "source": params.source,
        }

    print()
    print("[5/5] Inertia correction check")

    inertia_correction = make_uniform_section_correction(
        delta_J_rho_relative=cfg.delta_J_rho_relative,
    )
    inertia_params = apply_section_parameter_corrections(
        baseline_params,
        correction=inertia_correction,
        source_suffix="delta_J_rho",
    )
    M_inertia, K_inertia = build_mk(model, inertia_params, cfg)

    M_inertia_stats = matrix_diff_stats(M_inertia, M_base)
    K_inertia_stats = matrix_diff_stats(K_inertia, K_base)

    inertia_checks = {
        "M_changed": M_inertia_stats["max_abs"] > cfg.min_matrix_change,
        "K_unchanged": K_inertia_stats["max_abs"] <= cfg.zero_atol_K,
        "M_finite": np.isfinite(M_inertia_stats["max_abs"]),
    }

    print_case_result(
        "delta_J_rho",
        M_inertia_stats,
        K_inertia_stats,
        inertia_checks,
    )

    results["delta_J_rho"] = {
        "M_stats": M_inertia_stats,
        "K_stats": K_inertia_stats,
        "checks": inertia_checks,
        "source": inertia_params.source,
    }

    all_checks = []
    for item in results.values():
        all_checks.extend(bool(v) for v in item["checks"].values())

    all_passed = bool(all(all_checks))

    report = {
        "passed": all_passed,
        "config": asdict(cfg),
        "blade_csv": str(blade_csv),
        "model_info": {
            "model_name": model.model_name,
            "span_m": float(model.span_m),
            "n_stations": int(model.n_stations),
            "n_free_nodes": int(model.n_stations - 1),
            "n_dofs_free": int(M_base.shape[0]),
        },
        "results": results,
    }

    if cfg.save_report:
        report_path = output_dir / "section_correction_sanity_report.json"
        save_json(report_path, report)
        print()
        print(f"[Saved Report] {report_path}")

    print()
    if all_passed:
        print("✅ PASS: section-parameter correction sanity test passed.")
        print("   Zero correction preserves baseline; physical section corrections affect expected matrices.")
    else:
        print("❌ FAIL: section-parameter correction sanity test failed.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()