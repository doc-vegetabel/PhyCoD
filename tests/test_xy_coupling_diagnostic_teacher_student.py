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


from scripts.run_student_cases import run_student_case  # noqa: E402
from src.teacher.beamdyn_teacher_adapter import (  # noqa: E402
    BeamDynTeacherCaseConfig,
    run_teacher_case_beamdyn,
    load_teacher_6dof_response,
    resample_response_to_time_grid,
    get_full_state_component,
    get_tip_component,
    get_last_k_component,
)


@dataclass
class XYCouplingDiagnosticConfig:
    teacher_exe: str = r"D:\openfast\openfast-main\openfast-main\build\modules\beamdyn\Release\beamdyn_driver.exe"
    template_inp: str = str(
        PROJECT_ROOT
        / "data"
        / "raw"
        / "reference_cases"
        / "beamdyn"
        / "nrel5mw"
        / "bd_driver_dynamic_nrel_5mw.inp"
    )
    blade_csv: str = str(PROJECT_ROOT / "data" / "raw" / "nrel5mw" / "blade_master.csv")

    # 如果你的文件名不同，运行时用 --x-load-file / --y-load-file 覆盖
    x_load_file: str = str(PROJECT_ROOT / "data" / "load" / "simple_tip_fx_case.dat")
    y_load_file: str = str(PROJECT_ROOT / "data" / "load" / "simple_tip_fy_case.dat")

    output_dir: str = str(PROJECT_ROOT / "results" / "student" / "xy_coupling_diagnostic")
    case_prefix: str = "xy_coupling_diagnostic"

    t_initial: float = 0.0
    t_final: float = 10.0
    dt: float = 0.01

    teacher_node_start: int = 2
    teacher_node_end: int = 49
    teacher_demean: bool = False

    alpha_flap: float = 1.0
    alpha_edge: float = 1.0
    alpha_torsion: float = 1.0
    zeta_structural: float = 0.015
    ref_freq_hz: Optional[float] = None

    # 动态响应诊断建议去掉初始帧偏置
    remove_initial_offset: bool = True

    # last-k 末端节点数量
    last_k: int = 5

    save_report: bool = True


def parse_args() -> XYCouplingDiagnosticConfig:
    default = XYCouplingDiagnosticConfig()

    parser = argparse.ArgumentParser(
        description=(
            "Diagnose x-y coupling mismatch between BeamDyn teacher and "
            "direct full-order student under x-only and y-only load cases."
        )
    )

    parser.add_argument("--teacher-exe", type=str, default=default.teacher_exe)
    parser.add_argument("--template-inp", type=str, default=default.template_inp)
    parser.add_argument("--blade-csv", type=str, default=default.blade_csv)

    parser.add_argument("--x-load-file", type=str, default=default.x_load_file)
    parser.add_argument("--y-load-file", type=str, default=default.y_load_file)

    parser.add_argument("--output-dir", type=str, default=default.output_dir)
    parser.add_argument("--case-prefix", type=str, default=default.case_prefix)

    parser.add_argument("--t-initial", type=float, default=default.t_initial)
    parser.add_argument("--t-final", type=float, default=default.t_final)
    parser.add_argument("--dt", type=float, default=default.dt)

    parser.add_argument("--teacher-node-start", type=int, default=default.teacher_node_start)
    parser.add_argument("--teacher-node-end", type=int, default=default.teacher_node_end)

    demean_group = parser.add_mutually_exclusive_group()
    demean_group.add_argument("--teacher-demean", dest="teacher_demean", action="store_true")
    demean_group.add_argument("--no-teacher-demean", dest="teacher_demean", action="store_false")
    parser.set_defaults(teacher_demean=default.teacher_demean)

    parser.add_argument("--alpha-flap", type=float, default=default.alpha_flap)
    parser.add_argument("--alpha-edge", type=float, default=default.alpha_edge)
    parser.add_argument("--alpha-torsion", type=float, default=default.alpha_torsion)
    parser.add_argument("--zeta-structural", type=float, default=default.zeta_structural)
    parser.add_argument("--ref-freq-hz", type=float, default=default.ref_freq_hz)

    offset_group = parser.add_mutually_exclusive_group()
    offset_group.add_argument(
        "--remove-initial-offset",
        dest="remove_initial_offset",
        action="store_true",
    )
    offset_group.add_argument(
        "--keep-initial-offset",
        dest="remove_initial_offset",
        action="store_false",
    )
    parser.set_defaults(remove_initial_offset=default.remove_initial_offset)

    parser.add_argument("--last-k", type=int, default=default.last_k)

    save_group = parser.add_mutually_exclusive_group()
    save_group.add_argument("--save-report", dest="save_report", action="store_true")
    save_group.add_argument("--no-save-report", dest="save_report", action="store_false")
    parser.set_defaults(save_report=default.save_report)

    args = parser.parse_args()

    return XYCouplingDiagnosticConfig(
        teacher_exe=args.teacher_exe,
        template_inp=args.template_inp,
        blade_csv=args.blade_csv,
        x_load_file=args.x_load_file,
        y_load_file=args.y_load_file,
        output_dir=args.output_dir,
        case_prefix=args.case_prefix,
        t_initial=args.t_initial,
        t_final=args.t_final,
        dt=args.dt,
        teacher_node_start=args.teacher_node_start,
        teacher_node_end=args.teacher_node_end,
        teacher_demean=args.teacher_demean,
        alpha_flap=args.alpha_flap,
        alpha_edge=args.alpha_edge,
        alpha_torsion=args.alpha_torsion,
        zeta_structural=args.zeta_structural,
        ref_freq_hz=args.ref_freq_hz,
        remove_initial_offset=args.remove_initial_offset,
        last_k=args.last_k,
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


def remove_initial_offset(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    if u.ndim != 2:
        raise ValueError(f"u must be 2D, got shape={u.shape}.")
    return u - u[:1, :]


def _safe_ratio(num: float, den: float) -> float:
    num = float(num)
    den = float(den)
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) < 1e-30:
        return float("nan")
    return num / den


def signal_stats(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    return {
        "mae": float(np.mean(np.abs(x))),
        "rms": float(np.sqrt(np.mean(x**2))),
        "max_abs": float(np.max(np.abs(x))),
    }


def component_stats(
    u: np.ndarray,
    *,
    component: str,
    last_k: int,
) -> Dict[str, Any]:
    full_field = get_full_state_component(u, component)
    tip_signal = get_tip_component(u, component)
    last_k_field = get_last_k_component(u, component, k=last_k)

    return {
        "full": signal_stats(full_field),
        "tip": signal_stats(tip_signal),
        "last_k": signal_stats(last_k_field),
    }


def coupling_metrics_for_response(
    u: np.ndarray,
    *,
    primary_component: str,
    cross_component: str,
    last_k: int,
) -> Dict[str, Any]:
    primary = component_stats(
        u,
        component=primary_component,
        last_k=last_k,
    )
    cross = component_stats(
        u,
        component=cross_component,
        last_k=last_k,
    )

    ratios = {
        "full_cross_to_primary_rms": _safe_ratio(
            cross["full"]["rms"],
            primary["full"]["rms"],
        ),
        "full_cross_to_primary_max_abs": _safe_ratio(
            cross["full"]["max_abs"],
            primary["full"]["max_abs"],
        ),
        "tip_cross_to_primary_rms": _safe_ratio(
            cross["tip"]["rms"],
            primary["tip"]["rms"],
        ),
        "tip_cross_to_primary_max_abs": _safe_ratio(
            cross["tip"]["max_abs"],
            primary["tip"]["max_abs"],
        ),
        "last_k_cross_to_primary_rms": _safe_ratio(
            cross["last_k"]["rms"],
            primary["last_k"]["rms"],
        ),
        "last_k_cross_to_primary_max_abs": _safe_ratio(
            cross["last_k"]["max_abs"],
            primary["last_k"]["max_abs"],
        ),
    }

    return {
        "primary_component": primary_component,
        "cross_component": cross_component,
        "primary": primary,
        "cross": cross,
        "ratios": ratios,
    }


def compare_teacher_student_coupling(
    teacher_metrics: Dict[str, Any],
    student_metrics: Dict[str, Any],
) -> Dict[str, float]:
    t_cross = teacher_metrics["cross"]
    s_cross = student_metrics["cross"]

    t_primary = teacher_metrics["primary"]
    s_primary = student_metrics["primary"]

    t_ratios = teacher_metrics["ratios"]
    s_ratios = student_metrics["ratios"]

    return {
        "student_full_cross_rms_over_teacher": _safe_ratio(
            s_cross["full"]["rms"],
            t_cross["full"]["rms"],
        ),
        "student_tip_cross_rms_over_teacher": _safe_ratio(
            s_cross["tip"]["rms"],
            t_cross["tip"]["rms"],
        ),
        "student_last_k_cross_rms_over_teacher": _safe_ratio(
            s_cross["last_k"]["rms"],
            t_cross["last_k"]["rms"],
        ),
        "teacher_full_cross_to_primary_rms": t_ratios["full_cross_to_primary_rms"],
        "student_full_cross_to_primary_rms": s_ratios["full_cross_to_primary_rms"],
        "teacher_tip_cross_to_primary_rms": t_ratios["tip_cross_to_primary_rms"],
        "student_tip_cross_to_primary_rms": s_ratios["tip_cross_to_primary_rms"],
        "teacher_last_k_cross_to_primary_rms": t_ratios["last_k_cross_to_primary_rms"],
        "student_last_k_cross_to_primary_rms": s_ratios["last_k_cross_to_primary_rms"],
        "full_ratio_gap_teacher_minus_student": (
            t_ratios["full_cross_to_primary_rms"]
            - s_ratios["full_cross_to_primary_rms"]
        ),
        "tip_ratio_gap_teacher_minus_student": (
            t_ratios["tip_cross_to_primary_rms"]
            - s_ratios["tip_cross_to_primary_rms"]
        ),
        "last_k_ratio_gap_teacher_minus_student": (
            t_ratios["last_k_cross_to_primary_rms"]
            - s_ratios["last_k_cross_to_primary_rms"]
        ),
        "teacher_full_primary_rms": t_primary["full"]["rms"],
        "student_full_primary_rms": s_primary["full"]["rms"],
        "teacher_full_cross_rms": t_cross["full"]["rms"],
        "student_full_cross_rms": s_cross["full"]["rms"],
    }


def run_teacher_and_student_case(
    *,
    case_name: str,
    load_file: Path,
    output_dir: Path,
    cfg: XYCouplingDiagnosticConfig,
    teacher_exe: Path,
    template_inp: Path,
    blade_csv: Path,
    primary_component: str,
    cross_component: str,
) -> Dict[str, Any]:
    print()
    print("=" * 100)
    print(f"[Case] {case_name}")
    print("=" * 100)
    print(f"  load_file = {load_file}")
    print(f"  primary_component = {primary_component}")
    print(f"  cross_component   = {cross_component}")

    print()
    print("[1/5] Running BeamDyn teacher")
    teacher_case_cfg = BeamDynTeacherCaseConfig(
        case_name=case_name,
        teacher_exe=teacher_exe,
        template_inp=template_inp,
        output_dir=output_dir / "teacher",
        use_time_series_load=True,
        t_initial=cfg.t_initial,
        t_final=cfg.t_final,
        dt=cfg.dt,
        time_series_load_file=load_file,
        keep_temp_inp=False,
    )

    teacher_result = run_teacher_case_beamdyn(teacher_case_cfg)
    teacher_out = teacher_result["out"]
    if teacher_out is None:
        raise RuntimeError(f"Teacher case {case_name} did not return .out path.")

    print(f"  teacher_out = {teacher_out}")

    print()
    print("[2/5] Loading teacher 6DOF response")
    time_teacher, u_teacher, teacher_columns = load_teacher_6dof_response(
        out_path=teacher_out,
        node_start=cfg.teacher_node_start,
        node_end=cfg.teacher_node_end,
        demean=cfg.teacher_demean,
    )

    print(f"  time_teacher shape = {time_teacher.shape}")
    print(f"  u_teacher shape    = {u_teacher.shape}")

    print()
    print("[3/5] Running direct student")
    student_params = {
        "alpha_flap": cfg.alpha_flap,
        "alpha_edge": cfg.alpha_edge,
        "alpha_torsion": cfg.alpha_torsion,
        "zeta_structural": cfg.zeta_structural,
        "ref_freq_hz": cfg.ref_freq_hz,
    }

    student_result = run_student_case(
        blade_csv=blade_csv,
        output_dir=output_dir / "student",
        case_name=case_name,
        use_time_series_load=True,
        t_initial=cfg.t_initial,
        t_final=cfg.t_final,
        dt=cfg.dt,
        time_series_load_file=load_file,
        student_params=student_params,
        u0_trans=None,
        v0_trans=None,
    )

    time_student = np.asarray(student_result["time"], dtype=np.float64)
    u_student = np.asarray(student_result["u_full"], dtype=np.float64)

    print(f"  time_student shape = {time_student.shape}")
    print(f"  u_student shape    = {u_student.shape}")

    print()
    print("[4/5] Resampling student to teacher time grid")
    u_student_rs = resample_response_to_time_grid(
        time_src=time_student,
        u_src=u_student,
        time_dst=time_teacher,
    )

    if u_student_rs.shape != u_teacher.shape:
        raise ValueError(
            f"student/teacher shape mismatch: {u_student_rs.shape} vs {u_teacher.shape}"
        )

    if cfg.remove_initial_offset:
        u_teacher_eval = remove_initial_offset(u_teacher)
        u_student_eval = remove_initial_offset(u_student_rs)
    else:
        u_teacher_eval = u_teacher
        u_student_eval = u_student_rs

    print(f"  remove_initial_offset = {cfg.remove_initial_offset}")

    print()
    print("[5/5] Computing coupling metrics")
    teacher_metrics = coupling_metrics_for_response(
        u_teacher_eval,
        primary_component=primary_component,
        cross_component=cross_component,
        last_k=cfg.last_k,
    )
    student_metrics = coupling_metrics_for_response(
        u_student_eval,
        primary_component=primary_component,
        cross_component=cross_component,
        last_k=cfg.last_k,
    )
    comparison = compare_teacher_student_coupling(
        teacher_metrics=teacher_metrics,
        student_metrics=student_metrics,
    )

    print_coupling_case_summary(
        case_name=case_name,
        teacher_metrics=teacher_metrics,
        student_metrics=student_metrics,
        comparison=comparison,
    )

    return {
        "case_name": case_name,
        "load_file": str(load_file),
        "primary_component": primary_component,
        "cross_component": cross_component,
        "teacher_result": teacher_result,
        "student_npz": str(student_result["npz"]),
        "student_yaml": str(student_result["yaml"]),
        "teacher_columns_head": teacher_columns[:12],
        "teacher_columns_tail": teacher_columns[-12:],
        "shapes": {
            "time_teacher": list(time_teacher.shape),
            "u_teacher": list(u_teacher.shape),
            "time_student": list(time_student.shape),
            "u_student": list(u_student.shape),
            "u_student_resampled": list(u_student_rs.shape),
        },
        "teacher_metrics": teacher_metrics,
        "student_metrics": student_metrics,
        "comparison": comparison,
    }


def print_coupling_case_summary(
    *,
    case_name: str,
    teacher_metrics: Dict[str, Any],
    student_metrics: Dict[str, Any],
    comparison: Dict[str, float],
) -> None:
    print()
    print(f"[Coupling Summary] {case_name}")

    print("  Teacher:")
    print(
        f"    full primary RMS = {teacher_metrics['primary']['full']['rms']:.12e}, "
        f"full cross RMS = {teacher_metrics['cross']['full']['rms']:.12e}, "
        f"cross/primary = {teacher_metrics['ratios']['full_cross_to_primary_rms']:.12e}"
    )
    print(
        f"    tip primary RMS  = {teacher_metrics['primary']['tip']['rms']:.12e}, "
        f"tip cross RMS  = {teacher_metrics['cross']['tip']['rms']:.12e}, "
        f"cross/primary = {teacher_metrics['ratios']['tip_cross_to_primary_rms']:.12e}"
    )
    print(
        f"    last-k primary RMS = {teacher_metrics['primary']['last_k']['rms']:.12e}, "
        f"last-k cross RMS = {teacher_metrics['cross']['last_k']['rms']:.12e}, "
        f"cross/primary = {teacher_metrics['ratios']['last_k_cross_to_primary_rms']:.12e}"
    )

    print("  Student:")
    print(
        f"    full primary RMS = {student_metrics['primary']['full']['rms']:.12e}, "
        f"full cross RMS = {student_metrics['cross']['full']['rms']:.12e}, "
        f"cross/primary = {student_metrics['ratios']['full_cross_to_primary_rms']:.12e}"
    )
    print(
        f"    tip primary RMS  = {student_metrics['primary']['tip']['rms']:.12e}, "
        f"tip cross RMS  = {student_metrics['cross']['tip']['rms']:.12e}, "
        f"cross/primary = {student_metrics['ratios']['tip_cross_to_primary_rms']:.12e}"
    )
    print(
        f"    last-k primary RMS = {student_metrics['primary']['last_k']['rms']:.12e}, "
        f"last-k cross RMS = {student_metrics['cross']['last_k']['rms']:.12e}, "
        f"cross/primary = {student_metrics['ratios']['last_k_cross_to_primary_rms']:.12e}"
    )

    print("  Teacher - Student ratio gap:")
    print(
        f"    full ratio gap   = {comparison['full_ratio_gap_teacher_minus_student']:.12e}"
    )
    print(
        f"    tip ratio gap    = {comparison['tip_ratio_gap_teacher_minus_student']:.12e}"
    )
    print(
        f"    last-k ratio gap = {comparison['last_k_ratio_gap_teacher_minus_student']:.12e}"
    )
    print("  Student cross amplitude / Teacher cross amplitude:")
    print(
        f"    full  = {comparison['student_full_cross_rms_over_teacher']:.12e}"
    )
    print(
        f"    tip   = {comparison['student_tip_cross_rms_over_teacher']:.12e}"
    )
    print(
        f"    last-k = {comparison['student_last_k_cross_rms_over_teacher']:.12e}"
    )


def main() -> None:
    cfg = parse_args()

    teacher_exe = assert_existing_file(cfg.teacher_exe, "teacher_exe")
    template_inp = assert_existing_file(cfg.template_inp, "template_inp")
    blade_csv = assert_existing_file(cfg.blade_csv, "blade_csv")
    x_load_file = assert_existing_file(cfg.x_load_file, "x_load_file")
    y_load_file = assert_existing_file(cfg.y_load_file, "y_load_file")

    output_dir = Path(cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("[XY Coupling Diagnostic: Teacher vs Student]")
    print()
    print("[Config]")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")

    x_case = run_teacher_and_student_case(
        case_name=f"{cfg.case_prefix}_x_only",
        load_file=x_load_file,
        output_dir=output_dir / "x_only",
        cfg=cfg,
        teacher_exe=teacher_exe,
        template_inp=template_inp,
        blade_csv=blade_csv,
        primary_component="x",
        cross_component="y",
    )

    y_case = run_teacher_and_student_case(
        case_name=f"{cfg.case_prefix}_y_only",
        load_file=y_load_file,
        output_dir=output_dir / "y_only",
        cfg=cfg,
        teacher_exe=teacher_exe,
        template_inp=template_inp,
        blade_csv=blade_csv,
        primary_component="y",
        cross_component="x",
    )

    report = {
        "passed": True,
        "config": asdict(cfg),
        "x_only": x_case,
        "y_only": y_case,
        "interpretation_hints": {
            "large_teacher_cross_ratio_small_student_cross_ratio": (
                "Teacher has x-y coupled response that the current student does not reproduce."
            ),
            "student_cross_amplitude_over_teacher_near_zero": (
                "Student cross-axis response is much weaker than teacher cross-axis response."
            ),
            "next_modeling_step": (
                "Consider adding a physical flap-edge coupling parameter, such as a spanwise "
                "principal-axis rotation phi(s), instead of only scaling EI_flap/EI_edge."
            ),
        },
    }

    if cfg.save_report:
        report_path = output_dir / "xy_coupling_diagnostic_teacher_student_report.json"
        save_json(report_path, report)
        print()
        print(f"[Saved Report] {report_path}")

    print()
    print("✅ PASS: x-y coupling diagnostic completed.")
    print("   Read the Teacher/Student cross-to-primary ratios to confirm coupling mismatch.")


if __name__ == "__main__":
    main()