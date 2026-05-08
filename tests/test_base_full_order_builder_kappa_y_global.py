from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from src.student.base_full_order_builder import (  # noqa: E402
    build_base_student_full_order_mk,
    apply_global_kappa_y_scale_to_k,
    selected_kappa_y_dof_indices,
)


@dataclass
class BaseFullOrderKappaYGlobalTestConfig:
    blade_csv: str = str(PROJECT_ROOT / "data" / "raw" / "nrel5mw" / "blade_master.csv")
    output_dir: str = str(PROJECT_ROOT / "results" / "student" / "test_base_full_order_builder_kappa_y_global")

    alpha_flap: float = 1.0
    alpha_edge: float = 1.0
    alpha_torsion: float = 1.0

    twist_column: str = "initial_twist_deg"
    phi_sign: float = -1.0
    rotate_mass: bool = False

    kappa_y_global_scale: float = 0.952
    kappa_y_scale_mode: str = "y_bending"

    atol: float = 1.0e-8
    rtol: float = 1.0e-8
    diag_eps: float = 1.0e-12
    save_report: bool = True


def parse_args() -> BaseFullOrderKappaYGlobalTestConfig:
    d = BaseFullOrderKappaYGlobalTestConfig()

    parser = argparse.ArgumentParser(
        description=(
            "Test whether kappa_y_global_scale is correctly injected into "
            "build_base_student_full_order_mk(...) as a static base-student K scaling."
        )
    )

    parser.add_argument("--blade-csv", type=str, default=d.blade_csv)
    parser.add_argument("--output-dir", type=str, default=d.output_dir)

    parser.add_argument("--alpha-flap", type=float, default=d.alpha_flap)
    parser.add_argument("--alpha-edge", type=float, default=d.alpha_edge)
    parser.add_argument("--alpha-torsion", type=float, default=d.alpha_torsion)

    parser.add_argument("--twist-column", type=str, default=d.twist_column)
    parser.add_argument("--phi-sign", type=float, default=d.phi_sign)

    rotate_group = parser.add_mutually_exclusive_group()
    rotate_group.add_argument("--rotate-mass", dest="rotate_mass", action="store_true")
    rotate_group.add_argument("--no-rotate-mass", dest="rotate_mass", action="store_false")
    parser.set_defaults(rotate_mass=d.rotate_mass)

    parser.add_argument("--kappa-y-global-scale", type=float, default=d.kappa_y_global_scale)
    parser.add_argument(
        "--kappa-y-scale-mode",
        type=str,
        default=d.kappa_y_scale_mode,
        choices=["uy_only", "y_bending"],
    )

    parser.add_argument("--atol", type=float, default=d.atol)
    parser.add_argument("--rtol", type=float, default=d.rtol)
    parser.add_argument("--diag-eps", type=float, default=d.diag_eps)

    report_group = parser.add_mutually_exclusive_group()
    report_group.add_argument("--save-report", dest="save_report", action="store_true")
    report_group.add_argument("--no-save-report", dest="save_report", action="store_false")
    parser.set_defaults(save_report=d.save_report)

    args = parser.parse_args()

    return BaseFullOrderKappaYGlobalTestConfig(
        blade_csv=args.blade_csv,
        output_dir=args.output_dir,
        alpha_flap=args.alpha_flap,
        alpha_edge=args.alpha_edge,
        alpha_torsion=args.alpha_torsion,
        twist_column=args.twist_column,
        phi_sign=args.phi_sign,
        rotate_mass=args.rotate_mass,
        kappa_y_global_scale=args.kappa_y_global_scale,
        kappa_y_scale_mode=args.kappa_y_scale_mode,
        atol=args.atol,
        rtol=args.rtol,
        diag_eps=args.diag_eps,
        save_report=args.save_report,
    )


def assert_existing_file(path: str | Path, label: str) -> Path:
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {p}")
    return p


def relative_fro_error(a: np.ndarray, b: np.ndarray, eps: float = 1.0e-30) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + eps))


def make_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, Path):
        return str(obj)

    return obj


def compute_diag_ratio_error(
    *,
    K_ref: np.ndarray,
    K_test: np.ndarray,
    indices: np.ndarray,
    target_ratio: float,
    diag_eps: float,
) -> dict[str, Any]:
    diag_ref = np.diag(K_ref)
    diag_test = np.diag(K_test)

    indices = np.asarray(indices, dtype=np.int64)
    valid = indices[np.abs(diag_ref[indices]) > float(diag_eps)]

    if valid.size == 0:
        return {
            "num_valid": 0,
            "max_abs_error": None,
            "mean_abs_error": None,
        }

    ratio = diag_test[valid] / diag_ref[valid]
    abs_err = np.abs(ratio - float(target_ratio))

    return {
        "num_valid": int(valid.size),
        "max_abs_error": float(np.max(abs_err)),
        "mean_abs_error": float(np.mean(abs_err)),
    }


def main() -> None:
    cfg = parse_args()

    blade_csv = assert_existing_file(cfg.blade_csv, "blade_csv")
    output_dir = Path(cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 100)
    print("[Test] base_full_order_builder global kappa_y_scale")
    print("=" * 100)
    print(f"  blade_csv = {blade_csv}")
    print(f"  kappa_y_global_scale = {cfg.kappa_y_global_scale}")
    print(f"  kappa_y_scale_mode = {cfg.kappa_y_scale_mode}")

    print()
    print("[1/4] Build unscaled Phi-base M/K")

    unscaled = build_base_student_full_order_mk(
        blade_csv=blade_csv,
        model_name="test_base_phi_unscaled",
        alpha_flap=float(cfg.alpha_flap),
        alpha_edge=float(cfg.alpha_edge),
        alpha_torsion=float(cfg.alpha_torsion),
        twist_column=str(cfg.twist_column),
        phi_sign=float(cfg.phi_sign),
        rotate_mass=bool(cfg.rotate_mass),
        use_kappa_y_global_scale=False,
        kappa_y_global_scale=float(cfg.kappa_y_global_scale),
        kappa_y_scale_mode=str(cfg.kappa_y_scale_mode),
        verbose=True,
    )

    M_unscaled = np.asarray(unscaled["M"], dtype=np.float64)
    K_unscaled = np.asarray(unscaled["K"], dtype=np.float64)

    print()
    print("[2/4] Build scaled Phi-base M/K")

    scaled = build_base_student_full_order_mk(
        blade_csv=blade_csv,
        model_name="test_base_phi_scaled",
        alpha_flap=float(cfg.alpha_flap),
        alpha_edge=float(cfg.alpha_edge),
        alpha_torsion=float(cfg.alpha_torsion),
        twist_column=str(cfg.twist_column),
        phi_sign=float(cfg.phi_sign),
        rotate_mass=bool(cfg.rotate_mass),
        use_kappa_y_global_scale=True,
        kappa_y_global_scale=float(cfg.kappa_y_global_scale),
        kappa_y_scale_mode=str(cfg.kappa_y_scale_mode),
        verbose=True,
    )

    M_scaled = np.asarray(scaled["M"], dtype=np.float64)
    K_scaled = np.asarray(scaled["K"], dtype=np.float64)
    K_phi_unscaled_from_scaled_call = np.asarray(scaled["K_phi_unscaled"], dtype=np.float64)

    print()
    print("[3/4] Build expected K_scaled = S K S manually")

    K_expected, expected_info = apply_global_kappa_y_scale_to_k(
        K_unscaled,
        kappa_y_scale=float(cfg.kappa_y_global_scale),
        scale_mode=str(cfg.kappa_y_scale_mode),
    )

    scaled_dof_idx = selected_kappa_y_dof_indices(
        n_dofs=K_unscaled.shape[0],
        scale_mode=str(cfg.kappa_y_scale_mode),
    )

    all_idx = np.arange(K_unscaled.shape[0], dtype=np.int64)
    unscaled_dof_idx = np.setdiff1d(all_idx, scaled_dof_idx)

    print()
    print("[4/4] Check matrix consistency")

    M_rel_err = relative_fro_error(M_scaled, M_unscaled)
    K_rel_err = relative_fro_error(K_scaled, K_expected)
    K_phi_rel_err = relative_fro_error(K_phi_unscaled_from_scaled_call, K_unscaled)

    K_scaled_sym_max_abs = float(np.max(np.abs(K_scaled - K_scaled.T)))
    K_expected_sym_max_abs = float(np.max(np.abs(K_expected - K_expected.T)))

    M_unchanged = bool(np.allclose(M_scaled, M_unscaled, atol=cfg.atol, rtol=cfg.rtol))
    K_matches_expected = bool(np.allclose(K_scaled, K_expected, atol=cfg.atol, rtol=cfg.rtol))
    K_phi_consistent = bool(
        np.allclose(K_phi_unscaled_from_scaled_call, K_unscaled, atol=cfg.atol, rtol=cfg.rtol)
    )

    finite_ok = bool(
        np.all(np.isfinite(M_unscaled))
        and np.all(np.isfinite(K_unscaled))
        and np.all(np.isfinite(M_scaled))
        and np.all(np.isfinite(K_scaled))
    )

    K_changed = bool(not np.allclose(K_scaled, K_unscaled, atol=cfg.atol, rtol=cfg.rtol))

    selected_diag_error = compute_diag_ratio_error(
        K_ref=K_unscaled,
        K_test=K_scaled,
        indices=scaled_dof_idx,
        target_ratio=float(cfg.kappa_y_global_scale),
        diag_eps=float(cfg.diag_eps),
    )

    unselected_diag_error = compute_diag_ratio_error(
        K_ref=K_unscaled,
        K_test=K_scaled,
        indices=unscaled_dof_idx,
        target_ratio=1.0,
        diag_eps=float(cfg.diag_eps),
    )

    pass_flags = {
        "finite_ok": finite_ok,
        "M_unchanged": M_unchanged,
        "K_changed": K_changed,
        "K_matches_expected_sks": K_matches_expected,
        "K_phi_unscaled_consistent": K_phi_consistent,
        "K_scaled_symmetric": bool(K_scaled_sym_max_abs <= 10.0 * cfg.atol),
    }

    passed = bool(all(pass_flags.values()))

    report = {
        "config": asdict(cfg),
        "passed": passed,
        "pass_flags": pass_flags,
        "shapes": {
            "M_unscaled": list(M_unscaled.shape),
            "K_unscaled": list(K_unscaled.shape),
            "M_scaled": list(M_scaled.shape),
            "K_scaled": list(K_scaled.shape),
        },
        "errors": {
            "M_rel_err_scaled_vs_unscaled": M_rel_err,
            "K_rel_err_scaled_vs_expected_sks": K_rel_err,
            "K_phi_rel_err_scaled_call_vs_unscaled_call": K_phi_rel_err,
            "K_scaled_sym_max_abs": K_scaled_sym_max_abs,
            "K_expected_sym_max_abs": K_expected_sym_max_abs,
            "selected_diag_ratio_error_to_kappa_y_scale": selected_diag_error,
            "unselected_diag_ratio_error_to_1": unselected_diag_error,
        },
        "kappa_y_info_from_builder": scaled.get("kappa_y_info", {}),
        "expected_kappa_y_info": expected_info,
    }

    if bool(cfg.save_report):
        report_path = output_dir / "test_base_full_order_builder_kappa_y_global_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(make_json_safe(report), f, indent=2, ensure_ascii=False)
        print(f"  report = {report_path}")

    print()
    print("[Summary]")
    print(f"  finite_ok                 = {finite_ok}")
    print(f"  M_unchanged               = {M_unchanged}")
    print(f"  K_changed                 = {K_changed}")
    print(f"  K_matches_expected_sks    = {K_matches_expected}")
    print(f"  K_phi_unscaled_consistent = {K_phi_consistent}")
    print(f"  K_scaled_sym_max_abs      = {K_scaled_sym_max_abs:.6e}")
    print(f"  M_rel_err                 = {M_rel_err:.6e}")
    print(f"  K_rel_err                 = {K_rel_err:.6e}")

    if not passed:
        print()
        print("❌ FAIL: global kappa_y_scale was not injected consistently.")
        raise AssertionError(json.dumps(make_json_safe(report), indent=2, ensure_ascii=False))

    print()
    print("✅ PASS: global kappa_y_scale is now part of the static Phi-base student K matrix.")


if __name__ == "__main__":
    main()