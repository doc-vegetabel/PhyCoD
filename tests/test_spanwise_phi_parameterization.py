from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from src.student.io import load_student_model_from_blade_master  # noqa: E402
from src.student.corrected_fem_builder import build_corrected_fem_matrices_6dof  # noqa: E402
from src.student.coupled_fem_builder import (  # noqa: E402
    build_coupled_fem_matrices_6dof_degrees,
    bending_xy_coupling_norms_6dof,
)
from src.student.spanwise_phi_parameterization import (  # noqa: E402
    build_uniform_phi_profile,
    build_linear_phi_profile,
    build_piecewise_constant_phi_profile,
    build_control_point_phi_profile,
    parse_float_list,
)


@dataclass
class SpanwisePhiParameterizationTestConfig:
    blade_csv: str = str(PROJECT_ROOT / "data" / "raw" / "nrel5mw" / "blade_master.csv")
    output_dir: str = str(PROJECT_ROOT / "results" / "student" / "spanwise_phi_parameterization_test")
    model_name: str = "spanwise_phi_parameterization_test"

    uniform_phi_deg: float = 5.0
    linear_root_phi_deg: float = 0.0
    linear_tip_phi_deg: float = 10.0

    piecewise_breakpoints: str = "0.0,0.25,0.5,0.75,1.0"
    piecewise_values_deg: str = "0.0,2.0,5.0,8.0"

    control_eta: str = "0.0,0.25,0.5,0.75,1.0"
    control_phi_deg: str = "0.0,2.0,5.0,8.0,10.0"

    alpha_flap: float = 1.0
    alpha_edge: float = 1.0
    alpha_torsion: float = 1.0
    rotate_mass: bool = False

    atol_M_zero: float = 1e-12
    atol_K_zero: float = 1e-8
    min_K_xy_fro: float = 1e-6

    save_report: bool = True


def parse_args() -> SpanwisePhiParameterizationTestConfig:
    default = SpanwisePhiParameterizationTestConfig()

    parser = argparse.ArgumentParser(
        description=(
            "Test spanwise phi(s) parameterization and its compatibility "
            "with coupled_fem_builder.py."
        )
    )

    parser.add_argument("--blade-csv", type=str, default=default.blade_csv)
    parser.add_argument("--output-dir", type=str, default=default.output_dir)
    parser.add_argument("--model-name", type=str, default=default.model_name)

    parser.add_argument("--uniform-phi-deg", type=float, default=default.uniform_phi_deg)
    parser.add_argument("--linear-root-phi-deg", type=float, default=default.linear_root_phi_deg)
    parser.add_argument("--linear-tip-phi-deg", type=float, default=default.linear_tip_phi_deg)

    parser.add_argument("--piecewise-breakpoints", type=str, default=default.piecewise_breakpoints)
    parser.add_argument("--piecewise-values-deg", type=str, default=default.piecewise_values_deg)

    parser.add_argument("--control-eta", type=str, default=default.control_eta)
    parser.add_argument("--control-phi-deg", type=str, default=default.control_phi_deg)

    parser.add_argument("--alpha-flap", type=float, default=default.alpha_flap)
    parser.add_argument("--alpha-edge", type=float, default=default.alpha_edge)
    parser.add_argument("--alpha-torsion", type=float, default=default.alpha_torsion)

    mass_group = parser.add_mutually_exclusive_group()
    mass_group.add_argument("--rotate-mass", dest="rotate_mass", action="store_true")
    mass_group.add_argument("--no-rotate-mass", dest="rotate_mass", action="store_false")
    parser.set_defaults(rotate_mass=default.rotate_mass)

    parser.add_argument("--atol-M-zero", type=float, default=default.atol_M_zero)
    parser.add_argument("--atol-K-zero", type=float, default=default.atol_K_zero)
    parser.add_argument("--min-K-xy-fro", type=float, default=default.min_K_xy_fro)

    save_group = parser.add_mutually_exclusive_group()
    save_group.add_argument("--save-report", dest="save_report", action="store_true")
    save_group.add_argument("--no-save-report", dest="save_report", action="store_false")
    parser.set_defaults(save_report=default.save_report)

    args = parser.parse_args()

    return SpanwisePhiParameterizationTestConfig(
        blade_csv=args.blade_csv,
        output_dir=args.output_dir,
        model_name=args.model_name,
        uniform_phi_deg=args.uniform_phi_deg,
        linear_root_phi_deg=args.linear_root_phi_deg,
        linear_tip_phi_deg=args.linear_tip_phi_deg,
        piecewise_breakpoints=args.piecewise_breakpoints,
        piecewise_values_deg=args.piecewise_values_deg,
        control_eta=args.control_eta,
        control_phi_deg=args.control_phi_deg,
        alpha_flap=args.alpha_flap,
        alpha_edge=args.alpha_edge,
        alpha_torsion=args.alpha_torsion,
        rotate_mass=args.rotate_mass,
        atol_M_zero=args.atol_M_zero,
        atol_K_zero=args.atol_K_zero,
        min_K_xy_fro=args.min_K_xy_fro,
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


def assert_existing_file(path: str | Path, label: str) -> Path:
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {p}")
    return p


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


def build_mk_with_profile(model, profile, cfg: SpanwisePhiParameterizationTestConfig):
    return build_coupled_fem_matrices_6dof_degrees(
        model,
        section_params=None,
        phi_deg=profile.phi_deg,
        alpha_flap=cfg.alpha_flap,
        alpha_edge=cfg.alpha_edge,
        alpha_torsion=cfg.alpha_torsion,
        rotate_mass=cfg.rotate_mass,
        return_full=True,
    )


def print_profile_summary(profile) -> None:
    s = profile.summary()
    print()
    print(f"[Profile] {profile.name}")
    print(f"  source = {profile.source}")
    print(f"  n_stations = {s['n_stations']}")
    print(f"  n_elements = {s['n_elements']}")
    print(
        f"  station phi deg: min={s['station_phi_deg_min']:.6f}, "
        f"max={s['station_phi_deg_max']:.6f}, "
        f"mean={s['station_phi_deg_mean']:.6f}"
    )
    print(
        f"  element phi deg: min={s['element_phi_deg_min']:.6f}, "
        f"max={s['element_phi_deg_max']:.6f}, "
        f"mean={s['element_phi_deg_mean']:.6f}"
    )


def main() -> None:
    cfg = parse_args()

    blade_csv = assert_existing_file(cfg.blade_csv, "blade_csv")
    output_dir = Path(cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("[Spanwise Phi Parameterization Test]")
    print()
    print("[Config]")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")

    print()
    print("[1/5] Loading StudentBeamModel")
    model = load_student_model_from_blade_master(
        csv_path=str(blade_csv),
        model_name=cfg.model_name,
    )

    print(f"  span_m = {model.span_m}")
    print(f"  n_stations = {model.n_stations}")
    print(f"  n_elements = {model.n_stations - 1}")

    print()
    print("[2/5] Building phi profiles")

    zero_profile = build_uniform_phi_profile(
        model,
        phi_deg=0.0,
        name="zero_uniform_phi",
    )
    uniform_profile = build_uniform_phi_profile(
        model,
        phi_deg=cfg.uniform_phi_deg,
        name="uniform_phi",
    )
    linear_profile = build_linear_phi_profile(
        model,
        root_phi_deg=cfg.linear_root_phi_deg,
        tip_phi_deg=cfg.linear_tip_phi_deg,
        name="linear_phi",
    )
    piecewise_profile = build_piecewise_constant_phi_profile(
        model,
        breakpoints=parse_float_list(cfg.piecewise_breakpoints),
        values_deg=parse_float_list(cfg.piecewise_values_deg),
        name="piecewise_phi",
    )
    control_profile = build_control_point_phi_profile(
        model,
        control_eta=parse_float_list(cfg.control_eta),
        control_phi_deg=parse_float_list(cfg.control_phi_deg),
        name="control_point_phi",
    )

    profiles = [
        zero_profile,
        uniform_profile,
        linear_profile,
        piecewise_profile,
        control_profile,
    ]

    for profile in profiles:
        print_profile_summary(profile)

    print()
    print("[3/5] Building baseline M/K")
    M_base, K_base = build_corrected_fem_matrices_6dof(
        model,
        section_params=None,
        alpha_flap=cfg.alpha_flap,
        alpha_edge=cfg.alpha_edge,
        alpha_torsion=cfg.alpha_torsion,
        return_full=False,
    )

    print(f"  M_base shape = {M_base.shape}")
    print(f"  K_base shape = {K_base.shape}")

    print()
    print("[4/5] Checking zero phi baseline equivalence")

    M_zero, K_zero, debug_zero = build_mk_with_profile(model, zero_profile, cfg)

    M_zero_diff = matrix_diff_stats(M_zero, M_base)
    K_zero_diff = matrix_diff_stats(K_zero, K_base)
    zero_coupling = bending_xy_coupling_norms_6dof(K_zero)

    zero_checks = {
        "M_zero_matches_baseline": M_zero_diff["max_abs"] <= cfg.atol_M_zero,
        "K_zero_matches_baseline": K_zero_diff["max_abs"] <= cfg.atol_K_zero,
        "K_zero_xy_near_zero": zero_coupling["K_xy_fro"] <= cfg.min_K_xy_fro,
    }

    print(f"  M_zero max_abs = {M_zero_diff['max_abs']:.12e}")
    print(f"  K_zero max_abs = {K_zero_diff['max_abs']:.12e}")
    print(f"  K_zero K_xy_fro = {zero_coupling['K_xy_fro']:.12e}")
    for k, v in zero_checks.items():
        print(f"  {k:<32s}: {'PASS' if v else 'FAIL'}")

    print()
    print("[5/5] Checking nonzero phi profiles produce K_xy coupling")

    profile_results: Dict[str, Any] = {
        "zero_uniform_phi": {
            "profile": zero_profile.to_dict(),
            "M_diff_vs_baseline": M_zero_diff,
            "K_diff_vs_baseline": K_zero_diff,
            "coupling": zero_coupling,
            "checks": zero_checks,
        }
    }

    nonzero_pass_flags: list[bool] = []

    for profile in [uniform_profile, linear_profile, piecewise_profile, control_profile]:
        M_phi, K_phi, debug_phi = build_mk_with_profile(model, profile, cfg)

        M_diff = matrix_diff_stats(M_phi, M_base)
        K_diff = matrix_diff_stats(K_phi, K_base)
        coupling = bending_xy_coupling_norms_6dof(K_phi)

        checks = {
            "profile_finite": bool(np.all(np.isfinite(profile.phi_deg))),
            "K_xy_nonzero": coupling["K_xy_fro"] > cfg.min_K_xy_fro,
            "K_symmetric": coupling["K_symmetry_max_abs"] <= cfg.atol_K_zero,
        }

        nonzero_pass_flags.extend(checks.values())

        print()
        print(f"[Nonzero profile check] {profile.name}")
        print(f"  K max_abs diff = {K_diff['max_abs']:.12e}")
        print(f"  K_xy_fro = {coupling['K_xy_fro']:.12e}")
        print(f"  K_xy_over_K_xx = {coupling['K_xy_over_K_xx']:.12e}")
        print(f"  K_xy_over_K_yy = {coupling['K_xy_over_K_yy']:.12e}")
        for k, v in checks.items():
            print(f"  {k:<32s}: {'PASS' if v else 'FAIL'}")

        profile_results[profile.name] = {
            "profile": profile.to_dict(),
            "M_diff_vs_baseline": M_diff,
            "K_diff_vs_baseline": K_diff,
            "coupling": coupling,
            "checks": checks,
            "debug_summary": {
                "builder": debug_phi["builder"],
                "rotate_mass": debug_phi["rotate_mass"],
                "phi_element_deg_head": debug_phi["phi_element_deg"][:10],
                "phi_element_deg_tail": debug_phi["phi_element_deg"][-10:],
            },
        }

    all_passed = bool(all(zero_checks.values()) and all(nonzero_pass_flags))

    report = {
        "passed": all_passed,
        "config": asdict(cfg),
        "model_info": {
            "model_name": model.model_name,
            "span_m": float(model.span_m),
            "n_stations": int(model.n_stations),
            "n_elements": int(model.n_stations - 1),
            "n_free_dofs": int(K_base.shape[0]),
        },
        "debug_zero_summary": {
            "builder": debug_zero["builder"],
            "rotate_mass": debug_zero["rotate_mass"],
            "phi_element_deg_head": debug_zero["phi_element_deg"][:10],
            "phi_element_deg_tail": debug_zero["phi_element_deg"][-10:],
        },
        "profile_results": profile_results,
    }

    if cfg.save_report:
        report_path = output_dir / "spanwise_phi_parameterization_report.json"
        save_json(report_path, report)
        print()
        print(f"[Saved Report] {report_path}")

    print()
    if all_passed:
        print("✅ PASS: spanwise phi parameterization test passed.")
        print("   phi(s)=0 recovers baseline; nonzero profiles create K_xy coupling.")
    else:
        print("❌ FAIL: spanwise phi parameterization test failed.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()