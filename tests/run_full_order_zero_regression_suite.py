from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class RegressionSuiteConfig:
    blade_csv: str = str(PROJECT_ROOT / "data" / "raw" / "nrel5mw" / "blade_master.csv")
    output_dir: str = str(PROJECT_ROOT / "results" / "student" / "full_order_zero_regression")

    test_load_file: str = str(PROJECT_ROOT / "data" / "load" / "test_complex_case.dat")
    train_load_file: str = str(PROJECT_ROOT / "data" / "load" / "train_complex_case.dat")

    t_initial: float = 0.0
    t_final: float = 10.0
    dt: float = 0.01

    alpha_flap: float = 1.0
    alpha_edge: float = 1.0
    alpha_torsion: float = 1.0
    zeta_structural: float = 0.015
    ref_freq_hz: Optional[float] = None

    torch_dtype: str = "float64"
    device: str = "cpu"
    linear_solve_mode: str = "solve"

    stop_on_fail: bool = True


def parse_args() -> RegressionSuiteConfig:
    default = RegressionSuiteConfig()

    parser = argparse.ArgumentParser(
        description=(
            "Run zero-correction regression suite for "
            "FullOrderCorrectedCoreTorch against direct student baseline."
        )
    )

    parser.add_argument("--blade-csv", type=str, default=default.blade_csv)
    parser.add_argument("--output-dir", type=str, default=default.output_dir)

    parser.add_argument("--test-load-file", type=str, default=default.test_load_file)
    parser.add_argument("--train-load-file", type=str, default=default.train_load_file)

    parser.add_argument("--t-initial", type=float, default=default.t_initial)
    parser.add_argument("--t-final", type=float, default=default.t_final)
    parser.add_argument("--dt", type=float, default=default.dt)

    parser.add_argument("--alpha-flap", type=float, default=default.alpha_flap)
    parser.add_argument("--alpha-edge", type=float, default=default.alpha_edge)
    parser.add_argument("--alpha-torsion", type=float, default=default.alpha_torsion)
    parser.add_argument("--zeta-structural", type=float, default=default.zeta_structural)
    parser.add_argument("--ref-freq-hz", type=float, default=default.ref_freq_hz)

    parser.add_argument(
        "--torch-dtype",
        type=str,
        default=default.torch_dtype,
        choices=["float64", "float32"],
    )
    parser.add_argument("--device", type=str, default=default.device)
    parser.add_argument(
        "--linear-solve-mode",
        type=str,
        default=default.linear_solve_mode,
        choices=["solve", "inverse"],
    )

    stop_group = parser.add_mutually_exclusive_group()
    stop_group.add_argument("--stop-on-fail", dest="stop_on_fail", action="store_true")
    stop_group.add_argument("--no-stop-on-fail", dest="stop_on_fail", action="store_false")
    parser.set_defaults(stop_on_fail=default.stop_on_fail)

    args = parser.parse_args()

    return RegressionSuiteConfig(
        blade_csv=args.blade_csv,
        output_dir=args.output_dir,
        test_load_file=args.test_load_file,
        train_load_file=args.train_load_file,
        t_initial=args.t_initial,
        t_final=args.t_final,
        dt=args.dt,
        alpha_flap=args.alpha_flap,
        alpha_edge=args.alpha_edge,
        alpha_torsion=args.alpha_torsion,
        zeta_structural=args.zeta_structural,
        ref_freq_hz=args.ref_freq_hz,
        torch_dtype=args.torch_dtype,
        device=args.device,
        linear_solve_mode=args.linear_solve_mode,
        stop_on_fail=args.stop_on_fail,
    )


def _optional_str(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def build_case_commands(cfg: RegressionSuiteConfig) -> List[Dict[str, Any]]:
    compare_script = PROJECT_ROOT / "tests" / "compare_full_corrected_core_vs_student.py"

    if not compare_script.exists():
        raise FileNotFoundError(
            f"找不到 compare 脚本: {compare_script}\n"
            f"请先把 scripts/compare_full_corrected_core_vs_student.py 移动到 tests/ 下。"
        )

    blade_csv = Path(cfg.blade_csv).resolve()
    if not blade_csv.exists():
        raise FileNotFoundError(f"找不到 blade_csv: {blade_csv}")

    cases: List[Dict[str, Any]] = [
        {
            "name": "zero_load_check",
            "use_time_series": False,
            "load_file": None,
        },
        {
            "name": "test_complex_case_zero_correction",
            "use_time_series": True,
            "load_file": str(Path(cfg.test_load_file).resolve()),
        },
        {
            "name": "train_complex_case_zero_correction",
            "use_time_series": True,
            "load_file": str(Path(cfg.train_load_file).resolve()),
        },
    ]

    out: List[Dict[str, Any]] = []

    for case in cases:
        case_output_dir = Path(cfg.output_dir).resolve() / case["name"]

        cmd = [
            sys.executable,
            str(compare_script),
            "--blade-csv",
            str(blade_csv),
            "--output-dir",
            str(case_output_dir),
            "--case-name",
            case["name"],
            "--t-initial",
            str(cfg.t_initial),
            "--t-final",
            str(cfg.t_final),
            "--dt",
            str(cfg.dt),
            "--alpha-flap",
            str(cfg.alpha_flap),
            "--alpha-edge",
            str(cfg.alpha_edge),
            "--alpha-torsion",
            str(cfg.alpha_torsion),
            "--zeta-structural",
            str(cfg.zeta_structural),
            "--torch-dtype",
            cfg.torch_dtype,
            "--device",
            cfg.device,
            "--linear-solve-mode",
            cfg.linear_solve_mode,
            "--save-report",
        ]

        ref_freq = _optional_str(cfg.ref_freq_hz)
        if ref_freq is not None:
            cmd.extend(["--ref-freq-hz", ref_freq])

        if case["use_time_series"]:
            load_file = Path(case["load_file"]).resolve()
            if not load_file.exists():
                raise FileNotFoundError(f"找不到载荷文件: {load_file}")
            cmd.extend(
                [
                    "--use-time-series-load",
                    "--time-series-load-file",
                    str(load_file),
                ]
            )
        else:
            cmd.append("--no-time-series-load")

        out.append(
            {
                "case_name": case["name"],
                "command": cmd,
                "output_dir": str(case_output_dir),
            }
        )

    return out


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def run_one_case(case_info: Dict[str, Any]) -> Dict[str, Any]:
    case_name = case_info["case_name"]
    cmd = case_info["command"]

    print()
    print("=" * 100)
    print(f"[Regression Case] {case_name}")
    print("=" * 100)
    print("Command:")
    print(" ".join(cmd))
    print()

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=None,
        stderr=None,
    )

    passed = result.returncode == 0

    print()
    if passed:
        print(f"✅ CASE PASS: {case_name}")
    else:
        print(f"❌ CASE FAIL: {case_name}")

    return {
        "case_name": case_name,
        "passed": passed,
        "returncode": int(result.returncode),
        "output_dir": case_info["output_dir"],
        "command": cmd,
    }


def main() -> None:
    cfg = parse_args()

    output_dir = Path(cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("[Full-Order Zero-Correction Regression Suite]")
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print()
    print("[Config]")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")

    case_commands = build_case_commands(cfg)

    results: List[Dict[str, Any]] = []
    all_passed = True

    for case_info in case_commands:
        case_result = run_one_case(case_info)
        results.append(case_result)

        if not case_result["passed"]:
            all_passed = False
            if cfg.stop_on_fail:
                print()
                print("[Stopped] stop_on_fail=True，已在首个失败工况处停止。")
                break

    summary = {
        "passed": bool(all_passed),
        "config": asdict(cfg),
        "results": results,
    }

    summary_path = output_dir / "full_order_zero_regression_summary.json"
    save_json(summary_path, summary)

    print()
    print("=" * 100)
    print("[Regression Summary]")
    print("=" * 100)

    for r in results:
        flag = "PASS" if r["passed"] else "FAIL"
        print(f"  {r['case_name']:<40s} {flag}")

    print()
    print(f"Saved summary: {summary_path}")

    if all_passed:
        print()
        print("✅ ALL PASS: full-order zero-correction regression suite passed.")
    else:
        print()
        print("❌ REGRESSION FAILED: at least one zero-correction case failed.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()