from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from src.student.transformer.physical_parameter_registry import (
    build_physical_parameter_registry,
)
from src.student.transformer.physical_templates import (
    PhysicalTemplateConfig,
    build_dynamic_stiffness_templates,
)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sanity test for physical parameter registry and dynamic stiffness templates."
        )
    )

    parser.add_argument(
        "--blade-csv",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw" / "nrel5mw" / "blade_master.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "results" / "student" / "test_physical_parameter_registry_and_templates"),
    )

    parser.add_argument(
        "--enabled-params",
        type=str,
        default="alpha_y,alpha_xy",
        help="Comma-separated enabled physical parameters.",
    )

    parser.add_argument("--kappa-y-static-scale", type=float, default=0.952)
    parser.add_argument(
        "--kappa-y-scale-mode",
        type=str,
        default="y_bending",
        choices=["uy_only", "y_bending"],
    )
    parser.add_argument("--kappa-y-delta", type=float, default=1.0e-3)

    parser.add_argument(
        "--xy-template-mode",
        type=str,
        default="root_to_tip",
        choices=["uniform", "root_to_tip", "tip_to_root"],
    )
    parser.add_argument("--xy-delta-phi-deg", type=float, default=1.0)

    parser.add_argument("--sym-atol", type=float, default=1.0e-8)
    parser.add_argument("--min-template-norm", type=float, default=1.0e-12)

    return parser.parse_args()


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


def rel_fro_norm(a: np.ndarray, b: np.ndarray, eps: float = 1.0e-30) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + eps))


def main() -> None:
    args = parse_args()

    blade_csv = Path(args.blade_csv).resolve()
    if not blade_csv.exists():
        raise FileNotFoundError(f"blade_csv not found: {blade_csv}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 100)
    print("[Test] physical parameter registry + dynamic stiffness templates")
    print("=" * 100)
    print(f"  blade_csv = {blade_csv}")
    print(f"  enabled_params = {args.enabled_params}")
    print(f"  kappa_y_static_scale = {args.kappa_y_static_scale}")
    print(f"  xy_template_mode = {args.xy_template_mode}")
    print(f"  xy_delta_phi_deg = {args.xy_delta_phi_deg}")

    # ------------------------------------------------------------------
    # 1. Registry test
    # ------------------------------------------------------------------
    registry = build_physical_parameter_registry(
        enabled_params=args.enabled_params,
    )

    print()
    print("[1/3] Registry summary")
    reg_summary = registry.summary()
    print(json.dumps(make_json_safe(reg_summary), indent=2, ensure_ascii=False))

    theta_zero = np.zeros((registry.total_dim,), dtype=np.float64)
    theta_split = registry.split_theta(theta_zero)

    registry_ok = (
        registry.total_dim == len(registry.max_abs_list())
        and set(theta_split.keys()) == set(registry.names)
    )

    if not registry_ok:
        raise AssertionError("Registry split or max_abs_list check failed.")

    # ------------------------------------------------------------------
    # 2. Template construction test
    # ------------------------------------------------------------------
    print()
    print("[2/3] Build dynamic stiffness templates")

    cfg = PhysicalTemplateConfig(
        blade_csv=str(blade_csv),
        kappa_y_static_scale=float(args.kappa_y_static_scale),
        kappa_y_scale_mode=str(args.kappa_y_scale_mode),
        kappa_y_delta=float(args.kappa_y_delta),
        xy_template_mode=str(args.xy_template_mode),
        xy_delta_phi_deg=float(args.xy_delta_phi_deg),
        verbose=True,
    )

    bundle = build_dynamic_stiffness_templates(cfg)

    M0 = bundle.M0
    K0 = bundle.K0
    K_y_template = bundle.K_y_template
    K_xy_template = bundle.K_xy_template

    D = K0.shape[0]

    shape_ok = (
        M0.shape == (D, D)
        and K0.shape == (D, D)
        and K_y_template.shape == (D, D)
        and K_xy_template.shape == (D, D)
    )

    finite_ok = (
        np.all(np.isfinite(M0))
        and np.all(np.isfinite(K0))
        and np.all(np.isfinite(K_y_template))
        and np.all(np.isfinite(K_xy_template))
    )

    symmetry = {
        "K0": float(np.max(np.abs(K0 - K0.T))),
        "K_y_template": float(np.max(np.abs(K_y_template - K_y_template.T))),
        "K_xy_template": float(np.max(np.abs(K_xy_template - K_xy_template.T))),
    }

    symmetric_ok = all(v <= float(args.sym_atol) for v in symmetry.values())

    template_norms = {
        "K_y_template": float(np.linalg.norm(K_y_template)),
        "K_xy_template": float(np.linalg.norm(K_xy_template)),
    }

    template_nonzero_ok = all(
        v > float(args.min_template_norm)
        for v in template_norms.values()
    )

    if not shape_ok:
        raise AssertionError("Template shape check failed.")

    if not finite_ok:
        raise AssertionError("Template finite check failed.")

    if not symmetric_ok:
        raise AssertionError(f"Template symmetry check failed: {symmetry}")

    if not template_nonzero_ok:
        raise AssertionError(f"Template norm too small: {template_norms}")

    # ------------------------------------------------------------------
    # 3. Zero theta / nonzero theta stiffness assembly sanity
    # ------------------------------------------------------------------
    print()
    print("[3/3] Assemble K_eff sanity")

    alpha_y_zero = 0.0
    alpha_xy_zero = 0.0

    K_eff_zero = (
        K0
        + alpha_y_zero * K_y_template
        + alpha_xy_zero * K_xy_template
    )

    zero_theta_rel_err = rel_fro_norm(K_eff_zero, K0)

    alpha_y_test = 0.01
    alpha_xy_test = 0.01

    K_eff_y = K0 + alpha_y_test * K_y_template
    K_eff_xy = K0 + alpha_xy_test * K_xy_template
    K_eff_both = (
        K0
        + alpha_y_test * K_y_template
        + alpha_xy_test * K_xy_template
    )

    nonzero_change = {
        "alpha_y_rel_change": rel_fro_norm(K_eff_y, K0),
        "alpha_xy_rel_change": rel_fro_norm(K_eff_xy, K0),
        "both_rel_change": rel_fro_norm(K_eff_both, K0),
    }

    zero_theta_ok = bool(zero_theta_rel_err <= 1.0e-14)
    nonzero_theta_ok = all(v > 0.0 for v in nonzero_change.values())

    if not zero_theta_ok:
        raise AssertionError(
            f"Zero theta should return K0, got rel_err={zero_theta_rel_err:.6e}"
        )

    if not nonzero_theta_ok:
        raise AssertionError(f"Nonzero theta did not change K_eff: {nonzero_change}")

    report = {
        "passed": True,
        "registry": reg_summary,
        "template_summary": bundle.summary(),
        "checks": {
            "registry_ok": registry_ok,
            "shape_ok": shape_ok,
            "finite_ok": finite_ok,
            "symmetric_ok": symmetric_ok,
            "template_nonzero_ok": template_nonzero_ok,
            "zero_theta_ok": zero_theta_ok,
            "nonzero_theta_ok": nonzero_theta_ok,
        },
        "symmetry": symmetry,
        "template_norms": template_norms,
        "zero_theta_rel_err": zero_theta_rel_err,
        "nonzero_change": nonzero_change,
    }

    report_path = output_dir / "physical_parameter_registry_and_templates_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(report), f, indent=2, ensure_ascii=False)

    print()
    print("[Summary]")
    print(f"  registry_ok          = {registry_ok}")
    print(f"  shape_ok             = {shape_ok}")
    print(f"  finite_ok            = {finite_ok}")
    print(f"  symmetric_ok         = {symmetric_ok}")
    print(f"  template_nonzero_ok  = {template_nonzero_ok}")
    print(f"  zero_theta_rel_err   = {zero_theta_rel_err:.6e}")
    print(f"  nonzero_change       = {nonzero_change}")
    print(f"  report               = {report_path}")

    print()
    print("✅ PASS: physical parameter registry and stiffness templates are ready.")


if __name__ == "__main__":
    main()