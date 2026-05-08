from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

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
    compute_student_teacher_6dof_metrics,
)


@dataclass
class TeacherStudentIOSmokeConfig:
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
    time_series_load_file: str = str(PROJECT_ROOT / "data" / "load" / "test_complex_case.dat")

    output_dir: str = str(PROJECT_ROOT / "results" / "student" / "teacher_student_io_smoke")
    case_name: str = "teacher_student_io_smoke"

    t_initial: float = 0.0
    t_final: float = 2.0
    dt: float = 0.01

    teacher_node_start: int = 2
    teacher_node_end: int = 49
    teacher_demean: bool = False

    alpha_flap: float = 1.0
    alpha_edge: float = 1.0
    alpha_torsion: float = 1.0
    zeta_structural: float = 0.015
    ref_freq_hz: Optional[float] = None

    save_report: bool = True


def parse_args() -> TeacherStudentIOSmokeConfig:
    default = TeacherStudentIOSmokeConfig()

    parser = argparse.ArgumentParser(
        description=(
            "Smoke test teacher BeamDyn 6DOF output, direct student response, "
            "time-grid resampling, and student-teacher metric computation."
        )
    )

    parser.add_argument("--teacher-exe", type=str, default=default.teacher_exe)
    parser.add_argument("--template-inp", type=str, default=default.template_inp)
    parser.add_argument("--blade-csv", type=str, default=default.blade_csv)
    parser.add_argument("--time-series-load-file", type=str, default=default.time_series_load_file)

    parser.add_argument("--output-dir", type=str, default=default.output_dir)
    parser.add_argument("--case-name", type=str, default=default.case_name)

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

    save_group = parser.add_mutually_exclusive_group()
    save_group.add_argument("--save-report", dest="save_report", action="store_true")
    save_group.add_argument("--no-save-report", dest="save_report", action="store_false")
    parser.set_defaults(save_report=default.save_report)

    args = parser.parse_args()

    return TeacherStudentIOSmokeConfig(
        teacher_exe=args.teacher_exe,
        template_inp=args.template_inp,
        blade_csv=args.blade_csv,
        time_series_load_file=args.time_series_load_file,
        output_dir=args.output_dir,
        case_name=args.case_name,
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


def print_metrics(metrics: Dict[str, float]) -> None:
    print()
    print("[Student vs Teacher Metrics]")
    ordered_keys = [
        "all_mae",
        "all_rmse",
        "full_x_mae",
        "full_y_mae",
        "full_z_mae",
        "tip_x_mae",
        "tip_y_mae",
        "tip_z_mae",
        "last5_y_mae",
    ]
    for k in ordered_keys:
        if k in metrics:
            print(f"  {k:<18s}: {metrics[k]:.12e}")


def main() -> None:
    cfg = parse_args()

    teacher_exe = assert_existing_file(cfg.teacher_exe, "teacher_exe")
    template_inp = assert_existing_file(cfg.template_inp, "template_inp")
    blade_csv = assert_existing_file(cfg.blade_csv, "blade_csv")
    load_file = assert_existing_file(cfg.time_series_load_file, "time_series_load_file")

    output_dir = Path(cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("[Teacher-Student IO Alignment Smoke Test]")
    print()
    print("[Config]")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")

    print()
    print("[1/5] Running BeamDyn teacher case")
    teacher_case_cfg = BeamDynTeacherCaseConfig(
        case_name=cfg.case_name,
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
        raise RuntimeError("Teacher run did not return an .out path.")

    print(f"  teacher_out = {teacher_out}")

    print()
    print("[2/5] Loading teacher 6DOF fullfield response")
    time_teacher, u_teacher, teacher_columns = load_teacher_6dof_response(
        out_path=teacher_out,
        node_start=cfg.teacher_node_start,
        node_end=cfg.teacher_node_end,
        demean=cfg.teacher_demean,
    )

    print(f"  time_teacher shape = {time_teacher.shape}")
    print(f"  u_teacher shape    = {u_teacher.shape}")
    print(f"  n_teacher_columns  = {len(teacher_columns)}")
    print(f"  first columns      = {teacher_columns[:6]}")
    print(f"  last columns       = {teacher_columns[-6:]}")

    print()
    print("[3/5] Running direct student case")
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
        case_name=cfg.case_name,
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
    u_student_full = np.asarray(student_result["u_full"], dtype=np.float64)

    print(f"  time_student shape = {time_student.shape}")
    print(f"  u_student_full shape = {u_student_full.shape}")

    print()
    print("[4/5] Resampling student response to teacher time grid")
    u_student_rs = resample_response_to_time_grid(
        time_src=time_student,
        u_src=u_student_full,
        time_dst=time_teacher,
    )

    print(f"  u_student_rs shape = {u_student_rs.shape}")

    expected_dofs = (cfg.teacher_node_end - cfg.teacher_node_start + 1) * 6

    shape_checks = {
        "teacher_time_1d": time_teacher.ndim == 1,
        "teacher_response_2d": u_teacher.ndim == 2,
        "student_response_2d": u_student_rs.ndim == 2,
        "teacher_dofs_expected": u_teacher.shape[1] == expected_dofs,
        "student_teacher_shape_match": u_student_rs.shape == u_teacher.shape,
        "finite_teacher": bool(np.all(np.isfinite(u_teacher))),
        "finite_student": bool(np.all(np.isfinite(u_student_rs))),
    }

    print()
    print("[Shape / Finite Checks]")
    for k, v in shape_checks.items():
        print(f"  {k:<32s}: {'PASS' if v else 'FAIL'}")

    if not all(shape_checks.values()):
        raise RuntimeError("Teacher/student IO alignment checks failed.")

    print()
    print("[5/5] Computing baseline student-vs-teacher metrics")
    metrics = compute_student_teacher_6dof_metrics(
        u_student=u_student_rs,
        u_teacher=u_teacher,
    )
    print_metrics(metrics)

    report = {
        "passed": True,
        "config": asdict(cfg),
        "teacher_case_config": teacher_case_cfg.to_dict(),
        "teacher_result": teacher_result,
        "student_npz": str(student_result["npz"]),
        "student_yaml": str(student_result["yaml"]),
        "shape_checks": shape_checks,
        "metrics": metrics,
        "shapes": {
            "time_teacher": list(time_teacher.shape),
            "u_teacher": list(u_teacher.shape),
            "time_student": list(time_student.shape),
            "u_student_full": list(u_student_full.shape),
            "u_student_resampled": list(u_student_rs.shape),
        },
        "teacher_columns_head": teacher_columns[:12],
        "teacher_columns_tail": teacher_columns[-12:],
    }

    if cfg.save_report:
        report_path = output_dir / "teacher_student_io_alignment_smoke_report.json"
        save_json(report_path, report)
        print()
        print(f"[Saved Report] {report_path}")

    print()
    print("✅ PASS: teacher/student IO alignment smoke test passed.")
    print("   Teacher 6DOF .out response and direct student full-order response are shape-aligned.")


if __name__ == "__main__":
    main()