from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.student.io import load_student_model_from_blade_master  # noqa: E402
from src.student.section_parameterization import (  # noqa: E402
    build_baseline_section_parameters,
    build_element_section_table,
    summarize_section_parameters,
)


@dataclass
class SectionParameterizationTestConfig:
    blade_csv: str = str(PROJECT_ROOT / "data" / "raw" / "nrel5mw" / "blade_master.csv")
    output_dir: str = str(PROJECT_ROOT / "results" / "student" / "section_parameterization_test")
    model_name: str = "section_parameterization_baseline_test"

    ea_default: float = 1.0e10
    gj_default: float = 1.0e9
    j_rho_scale: float = 1.0

    atol: float = 1e-12
    rtol: float = 1e-12

    save_report: bool = True


def parse_args() -> SectionParameterizationTestConfig:
    default = SectionParameterizationTestConfig()

    parser = argparse.ArgumentParser(
        description=(
            "Test whether section_parameterization.py reproduces the current "
            "fem_builder.py baseline section-parameter semantics."
        )
    )

    parser.add_argument("--blade-csv", type=str, default=default.blade_csv)
    parser.add_argument("--output-dir", type=str, default=default.output_dir)
    parser.add_argument("--model-name", type=str, default=default.model_name)

    parser.add_argument("--ea-default", type=float, default=default.ea_default)
    parser.add_argument("--gj-default", type=float, default=default.gj_default)
    parser.add_argument("--j-rho-scale", type=float, default=default.j_rho_scale)

    parser.add_argument("--atol", type=float, default=default.atol)
    parser.add_argument("--rtol", type=float, default=default.rtol)

    save_group = parser.add_mutually_exclusive_group()
    save_group.add_argument("--save-report", dest="save_report", action="store_true")
    save_group.add_argument("--no-save-report", dest="save_report", action="store_false")
    parser.set_defaults(save_report=default.save_report)

    args = parser.parse_args()

    return SectionParameterizationTestConfig(
        blade_csv=args.blade_csv,
        output_dir=args.output_dir,
        model_name=args.model_name,
        ea_default=args.ea_default,
        gj_default=args.gj_default,
        j_rho_scale=args.j_rho_scale,
        atol=args.atol,
        rtol=args.rtol,
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


def compare_array(
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
            "actual_size": float(actual.size),
            "expected_size": float(expected.size),
            "max_abs": float("inf"),
            "mae": float("inf"),
        }

    diff = actual - expected
    abs_diff = np.abs(diff)

    max_abs = float(np.max(abs_diff)) if abs_diff.size > 0 else 0.0
    mae = float(np.mean(abs_diff)) if abs_diff.size > 0 else 0.0

    passed = bool(np.allclose(actual, expected, atol=atol, rtol=rtol))

    return passed, {
        "shape_match": 1.0,
        "max_abs": max_abs,
        "mae": mae,
    }


def build_expected_old_fem_builder_element_values(
    model,
    *,
    ea_default: float,
    gj_default: float,
    j_rho_scale: float,
) -> Dict[str, np.ndarray]:
    """
    显式复刻当前 fem_builder.py 中 build_fem_matrices_6dof(...) 的
    单元参数取值方式，用于对比 section_parameterization.py。
    """
    span_stations = np.asarray(model.span_stations_m, dtype=np.float64)
    mass_dist = np.asarray(model.mass_per_length_kgpm, dtype=np.float64)
    ei_flap_dist = np.asarray(model.flapwise_ei_nm2, dtype=np.float64)

    if model.edgewise_ei_nm2 is None:
        ei_edge_dist = ei_flap_dist.copy()
    else:
        ei_edge_dist = np.asarray(model.edgewise_ei_nm2, dtype=np.float64)

    if model.torsional_gj_nm2 is None:
        gj_dist = np.full_like(span_stations, float(gj_default), dtype=np.float64)
    else:
        gj_dist = np.asarray(model.torsional_gj_nm2, dtype=np.float64)

    L_list = []
    m_avg_list = []
    ea_avg_list = []
    ei_flap_avg_list = []
    ei_edge_avg_list = []
    gj_avg_list = []
    j_rho_avg_list = []

    for i in range(model.n_stations - 1):
        L = span_stations[i + 1] - span_stations[i]
        if L <= 0.0:
            continue

        m_avg = 0.5 * (mass_dist[i] + mass_dist[i + 1])
        ei_flap_avg = 0.5 * (ei_flap_dist[i] + ei_flap_dist[i + 1])
        ei_edge_avg = 0.5 * (ei_edge_dist[i] + ei_edge_dist[i + 1])
        gj_avg = 0.5 * (gj_dist[i] + gj_dist[i + 1])

        # 当前 fem_builder.py 的占位规则
        ea = float(ea_default)
        j_rho = m_avg * float(j_rho_scale)

        L_list.append(L)
        m_avg_list.append(m_avg)
        ea_avg_list.append(ea)
        ei_flap_avg_list.append(ei_flap_avg)
        ei_edge_avg_list.append(ei_edge_avg)
        gj_avg_list.append(gj_avg)
        j_rho_avg_list.append(j_rho)

    return {
        "L_m": np.asarray(L_list, dtype=np.float64),
        "m_avg_kgpm": np.asarray(m_avg_list, dtype=np.float64),
        "EA_avg_N": np.asarray(ea_avg_list, dtype=np.float64),
        "EI_flap_avg_Nm2": np.asarray(ei_flap_avg_list, dtype=np.float64),
        "EI_edge_avg_Nm2": np.asarray(ei_edge_avg_list, dtype=np.float64),
        "GJ_avg_Nm2": np.asarray(gj_avg_list, dtype=np.float64),
        "J_rho_avg_kgm": np.asarray(j_rho_avg_list, dtype=np.float64),
    }


def main() -> None:
    cfg = parse_args()

    blade_csv = Path(cfg.blade_csv).resolve()
    output_dir = Path(cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not blade_csv.exists():
        raise FileNotFoundError(f"blade_csv not found: {blade_csv}")

    print()
    print("[Section Parameterization Baseline Equivalence Test]")
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
    print(f"  estimated_total_mass_kg = {model.estimated_total_mass_kg}")

    print()
    print("[2/4] Building baseline-compatible section parameters")
    params = build_baseline_section_parameters(
        model,
        ea_default=cfg.ea_default,
        gj_default=cfg.gj_default,
        j_rho_scale=cfg.j_rho_scale,
    )

    summary = summarize_section_parameters(params, span_m=model.span_m)
    print(f"  source = {params.source}")
    print(f"  n_stations = {params.n_stations}")
    print(f"  n_elements = {params.n_elements}")
    print("  station field summary:")
    for name, item in summary["station_fields"].items():
        print(
            f"    {name:<24s} "
            f"min={item['min']:.6e}, max={item['max']:.6e}, mean={item['mean']:.6e}"
        )

    print()
    print("[3/4] Building element section table")
    element_table = build_element_section_table(model, params)

    expected = build_expected_old_fem_builder_element_values(
        model,
        ea_default=cfg.ea_default,
        gj_default=cfg.gj_default,
        j_rho_scale=cfg.j_rho_scale,
    )

    print(f"  n_elements(table) = {element_table['element_index'].size}")
    print(f"  n_elements(expected) = {expected['L_m'].size}")

    print()
    print("[4/4] Comparing with current fem_builder.py semantics")

    compare_keys = [
        "L_m",
        "m_avg_kgpm",
        "EA_avg_N",
        "EI_flap_avg_Nm2",
        "EI_edge_avg_Nm2",
        "GJ_avg_Nm2",
        "J_rho_avg_kgm",
    ]

    checks: Dict[str, bool] = {}
    metrics: Dict[str, Dict[str, float]] = {}

    for key in compare_keys:
        passed, stat = compare_array(
            key,
            actual=element_table[key],
            expected=expected[key],
            atol=cfg.atol,
            rtol=cfg.rtol,
        )
        checks[key] = bool(passed)
        metrics[key] = stat

        flag = "PASS" if passed else "FAIL"
        print(
            f"  {key:<22s} {flag} | "
            f"mae={stat['mae']:.12e}, max_abs={stat['max_abs']:.12e}"
        )

    all_passed = bool(all(checks.values()))

    report = {
        "passed": all_passed,
        "checks": checks,
        "metrics": metrics,
        "config": asdict(cfg),
        "blade_csv": str(blade_csv),
        "model_info": {
            "model_name": model.model_name,
            "span_m": float(model.span_m),
            "n_stations": int(model.n_stations),
            "estimated_total_mass_kg": float(model.estimated_total_mass_kg),
        },
        "section_summary": summary,
    }

    if cfg.save_report:
        report_path = output_dir / "section_parameterization_baseline_equivalence_report.json"
        save_json(report_path, report)
        print()
        print(f"[Saved Report] {report_path}")

    print()
    if all_passed:
        print("✅ PASS: section_parameterization.py reproduces current fem_builder.py baseline semantics.")
    else:
        print("❌ FAIL: section_parameterization.py does not match current fem_builder.py baseline semantics.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()