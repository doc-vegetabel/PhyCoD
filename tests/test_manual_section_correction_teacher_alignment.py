from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.linalg import eigh


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from scripts.run_student_cases import (  # noqa: E402
    run_student_case,
    _build_structural_damping_matrix,
)
from src.student.io import load_student_model_from_blade_master  # noqa: E402
from src.student.section_parameterization import (  # noqa: E402
    build_baseline_section_parameters,
    apply_section_parameter_corrections,
    make_uniform_section_correction,
)
from src.student.corrected_fem_builder import build_corrected_fem_matrices_6dof  # noqa: E402
from src.student.full_order_corrected_core_torch import (  # noqa: E402
    FullOrderCorrectedCoreTorch,
    FullOrderCorrectedCoreTorchConfig,
)
from src.teacher.beamdyn_teacher_adapter import (  # noqa: E402
    BeamDynTeacherCaseConfig,
    run_teacher_case_beamdyn,
    load_teacher_6dof_response,
    resample_response_to_time_grid,
    compute_student_teacher_6dof_metrics,
)


@dataclass
class ManualSectionCorrectionTeacherAlignmentConfig:
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

    output_dir: str = str(PROJECT_ROOT / "results" / "student" / "manual_section_correction_teacher_alignment")
    case_name: str = "manual_section_correction_teacher_alignment"

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

    torch_dtype: str = "float64"
    device: str = "cpu"
    linear_solve_mode: str = "solve"

    # 第一版只做 1D 扫描，不做二维组合，避免计算量和解释复杂度过高。
    scan_flap_values: str = "-0.05,-0.02,-0.01,0.0,0.01,0.02,0.05"
    scan_edge_values: str = "-0.05,-0.02,-0.01,0.0,0.01,0.02,0.05"

    # 默认关注 y 方向叶尖/末端区域，因为前一个 smoke test 显示 y 方向误差更明显。
    objective_metric: str = "last5_y_mae"

    save_report: bool = True


def parse_args() -> ManualSectionCorrectionTeacherAlignmentConfig:
    default = ManualSectionCorrectionTeacherAlignmentConfig()

    parser = argparse.ArgumentParser(
        description=(
            "Manual no-network section-parameter correction scan against BeamDyn teacher. "
            "This script scans delta_EI_flap and delta_EI_edge separately."
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

    parser.add_argument(
        "--scan-flap-values",
        type=str,
        default=default.scan_flap_values,
        help="Comma-separated relative delta values for EI_flap scan.",
    )
    parser.add_argument(
        "--scan-edge-values",
        type=str,
        default=default.scan_edge_values,
        help="Comma-separated relative delta values for EI_edge scan.",
    )
    parser.add_argument("--objective-metric", type=str, default=default.objective_metric)

    save_group = parser.add_mutually_exclusive_group()
    save_group.add_argument("--save-report", dest="save_report", action="store_true")
    save_group.add_argument("--no-save-report", dest="save_report", action="store_false")
    parser.set_defaults(save_report=default.save_report)

    args = parser.parse_args()

    return ManualSectionCorrectionTeacherAlignmentConfig(
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
        torch_dtype=args.torch_dtype,
        device=args.device,
        linear_solve_mode=args.linear_solve_mode,
        scan_flap_values=args.scan_flap_values,
        scan_edge_values=args.scan_edge_values,
        objective_metric=args.objective_metric,
        save_report=args.save_report,
    )


def parse_scan_values(text: str) -> List[float]:
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if item == "":
            continue
        values.append(float(item))
    if len(values) == 0:
        raise ValueError("scan values must not be empty.")
    return values


def get_torch_dtype(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


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
    if torch.is_tensor(obj):
        if obj.ndim == 0:
            return obj.detach().cpu().item()
        return obj.detach().cpu().tolist()
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


def compute_natural_frequencies_hz(
    M: np.ndarray,
    K: np.ndarray,
    *,
    num_modes: int = 10,
) -> np.ndarray:
    eigvals, _ = eigh(K, M)
    eigvals = np.asarray(eigvals, dtype=np.float64)
    valid = eigvals > 0.0
    freqs = np.sqrt(eigvals[valid]) / (2.0 * np.pi)
    return freqs[:num_modes]


def build_damping_like_student(
    M: np.ndarray,
    K: np.ndarray,
    *,
    zeta_structural: float,
    ref_freq_hz: Optional[float],
) -> Tuple[np.ndarray, Optional[float], np.ndarray]:
    natural_freqs = compute_natural_frequencies_hz(M, K, num_modes=10)
    C, ref_freq_used = _build_structural_damping_matrix(
        K=K,
        zeta_structural=float(zeta_structural),
        ref_freq_hz=ref_freq_hz,
        natural_freqs=natural_freqs,
    )
    return np.asarray(C, dtype=np.float64), ref_freq_used, natural_freqs


def as_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def rollout_with_mkc(
    *,
    M: np.ndarray,
    K: np.ndarray,
    C: np.ndarray,
    F_time: np.ndarray,
    u0: np.ndarray,
    v0: np.ndarray,
    dt: float,
    dtype: torch.dtype,
    device: torch.device,
    linear_solve_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    core = FullOrderCorrectedCoreTorch(
        M=M,
        K=K,
        C=C,
        dt=dt,
        config=FullOrderCorrectedCoreTorchConfig(
            gamma=0.5,
            beta=0.25,
            dtype=dtype,
            linear_solve_mode=linear_solve_mode,
        ),
    ).to(device)

    F_torch = torch.as_tensor(F_time, dtype=dtype, device=device)
    u0_torch = torch.as_tensor(u0, dtype=dtype, device=device)
    v0_torch = torch.as_tensor(v0, dtype=dtype, device=device)

    with torch.no_grad():
        u_t, v_t, a_t = core.rollout(
            F_time=F_torch,
            u0=u0_torch,
            v0=v0_torch,
            theta_full=None,
            return_debug=False,
        )

    return as_numpy(u_t), as_numpy(v_t), as_numpy(a_t), core.summary()


def build_corrected_mkc_from_correction(
    *,
    model,
    baseline_params,
    correction,
    cfg: ManualSectionCorrectionTeacherAlignmentConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[float], np.ndarray, str]:
    section_params = apply_section_parameter_corrections(
        baseline_params,
        correction=correction,
        source_suffix="manual_teacher_alignment",
    )

    M, K = build_corrected_fem_matrices_6dof(
        model,
        section_params=section_params,
        alpha_flap=cfg.alpha_flap,
        alpha_edge=cfg.alpha_edge,
        alpha_torsion=cfg.alpha_torsion,
        return_full=False,
    )

    C, ref_freq_used, natural_freqs = build_damping_like_student(
        M=M,
        K=K,
        zeta_structural=cfg.zeta_structural,
        ref_freq_hz=cfg.ref_freq_hz,
    )

    return M, K, C, ref_freq_used, natural_freqs, section_params.source


def percent_improvement(
    baseline_value: float,
    candidate_value: float,
) -> float:
    baseline_value = float(baseline_value)
    candidate_value = float(candidate_value)
    if not np.isfinite(baseline_value) or abs(baseline_value) <= 1e-30:
        return float("nan")
    return float((baseline_value - candidate_value) / baseline_value * 100.0)


def print_metric_subset(title: str, metrics: Dict[str, float]) -> None:
    keys = [
        "all_mae",
        "all_rmse",
        "full_x_mae",
        "full_y_mae",
        "tip_x_mae",
        "tip_y_mae",
        "last5_y_mae",
    ]
    print()
    print(title)
    for k in keys:
        if k in metrics:
            print(f"  {k:<18s}: {metrics[k]:.12e}")


def make_case_label(kind: str, value: float) -> str:
    sign = "p" if value >= 0 else "m"
    number = f"{abs(value):.6f}".replace(".", "p")
    return f"{kind}_{sign}{number}"


def main() -> None:
    cfg = parse_args()

    teacher_exe = assert_existing_file(cfg.teacher_exe, "teacher_exe")
    template_inp = assert_existing_file(cfg.template_inp, "template_inp")
    blade_csv = assert_existing_file(cfg.blade_csv, "blade_csv")
    load_file = assert_existing_file(cfg.time_series_load_file, "time_series_load_file")

    output_dir = Path(cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = get_torch_dtype(cfg.torch_dtype)
    device = torch.device(cfg.device)

    flap_values = parse_scan_values(cfg.scan_flap_values)
    edge_values = parse_scan_values(cfg.scan_edge_values)

    print()
    print("[Manual Section Correction Teacher Alignment]")
    print()
    print("[Config]")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")

    print()
    print(f"[Scan Values] flap = {flap_values}")
    print(f"[Scan Values] edge = {edge_values}")

    # ------------------------------------------------------------------
    # 1) Run teacher.
    # ------------------------------------------------------------------
    print()
    print("[1/7] Running BeamDyn teacher case")
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

    # ------------------------------------------------------------------
    # 2) Load teacher 6DOF.
    # ------------------------------------------------------------------
    print()
    print("[2/7] Loading teacher 6DOF response")
    time_teacher, u_teacher, teacher_columns = load_teacher_6dof_response(
        out_path=teacher_out,
        node_start=cfg.teacher_node_start,
        node_end=cfg.teacher_node_end,
        demean=cfg.teacher_demean,
    )

    print(f"  time_teacher shape = {time_teacher.shape}")
    print(f"  u_teacher shape    = {u_teacher.shape}")
    print(f"  first columns      = {teacher_columns[:6]}")
    print(f"  last columns       = {teacher_columns[-6:]}")

    # ------------------------------------------------------------------
    # 3) Run direct baseline student.
    # ------------------------------------------------------------------
    print()
    print("[3/7] Running direct baseline student")
    student_params = {
        "alpha_flap": cfg.alpha_flap,
        "alpha_edge": cfg.alpha_edge,
        "alpha_torsion": cfg.alpha_torsion,
        "zeta_structural": cfg.zeta_structural,
        "ref_freq_hz": cfg.ref_freq_hz,
    }

    student_result = run_student_case(
        blade_csv=blade_csv,
        output_dir=output_dir / "student_direct",
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
    F_time = np.asarray(student_result["F_time"], dtype=np.float64)
    u_direct = np.asarray(student_result["u_full"], dtype=np.float64)
    v0 = np.asarray(student_result["v_full"], dtype=np.float64)[0]
    u0 = u_direct[0].copy()

    u_direct_rs = resample_response_to_time_grid(
        time_src=time_student,
        u_src=u_direct,
        time_dst=time_teacher,
    )

    if u_direct_rs.shape != u_teacher.shape:
        raise ValueError(
            f"Direct student / teacher shape mismatch: "
            f"{u_direct_rs.shape} vs {u_teacher.shape}"
        )

    baseline_metrics = compute_student_teacher_6dof_metrics(
        u_student=u_direct_rs,
        u_teacher=u_teacher,
    )

    print_metric_subset("[Baseline Direct Student vs Teacher]", baseline_metrics)

    if cfg.objective_metric not in baseline_metrics:
        raise KeyError(
            f"objective_metric={cfg.objective_metric!r} not found in metrics. "
            f"Available keys: {list(baseline_metrics.keys())}"
        )

    baseline_objective = float(baseline_metrics[cfg.objective_metric])
    print()
    print(f"[Objective] {cfg.objective_metric} baseline = {baseline_objective:.12e}")

    # ------------------------------------------------------------------
    # 4) Build model and baseline section params.
    # ------------------------------------------------------------------
    print()
    print("[4/7] Loading model and baseline section parameters")
    model = load_student_model_from_blade_master(
        csv_path=str(blade_csv),
        model_name="manual_section_correction_teacher_alignment_model",
    )
    baseline_params = build_baseline_section_parameters(model)

    print(f"  model span_m = {model.span_m}")
    print(f"  n_stations = {model.n_stations}")
    print(f"  n_free_nodes = {model.n_stations - 1}")
    print(f"  section source = {baseline_params.source}")

    # ------------------------------------------------------------------
    # 5) Zero corrected-FEM guard.
    # ------------------------------------------------------------------
    print()
    print("[5/7] Zero correction guard via corrected_fem_builder + torch core")
    M_zero, K_zero, C_zero, ref_freq_zero, freqs_zero, section_source_zero = build_corrected_mkc_from_correction(
        model=model,
        baseline_params=baseline_params,
        correction=None,
        cfg=cfg,
    )

    u_zero, v_zero, a_zero, zero_core_summary = rollout_with_mkc(
        M=M_zero,
        K=K_zero,
        C=C_zero,
        F_time=F_time,
        u0=u0,
        v0=v0,
        dt=cfg.dt,
        dtype=dtype,
        device=device,
        linear_solve_mode=cfg.linear_solve_mode,
    )

    u_zero_rs = resample_response_to_time_grid(
        time_src=time_student,
        u_src=u_zero,
        time_dst=time_teacher,
    )

    zero_metrics = compute_student_teacher_6dof_metrics(
        u_student=u_zero_rs,
        u_teacher=u_teacher,
    )

    zero_vs_direct_diff = float(np.max(np.abs(u_zero - u_direct)))
    print(f"  zero ref_freq_used = {ref_freq_zero}")
    print(f"  zero natural_freqs[:5] = {freqs_zero[:5]}")
    print(f"  zero_vs_direct_u_max_abs = {zero_vs_direct_diff:.12e}")
    print_metric_subset("[Zero Corrected-FEM Student vs Teacher]", zero_metrics)

    # ------------------------------------------------------------------
    # 6) Manual 1D scans.
    # ------------------------------------------------------------------
    print()
    print("[6/7] Running manual 1D section-correction scans")

    scan_results: List[Dict[str, Any]] = []

    def run_one_scan_case(
        *,
        scan_family: str,
        delta_value: float,
    ) -> Dict[str, Any]:
        if scan_family == "delta_EI_flap":
            correction = make_uniform_section_correction(
                delta_EI_flap_relative=delta_value,
            )
        elif scan_family == "delta_EI_edge":
            correction = make_uniform_section_correction(
                delta_EI_edge_relative=delta_value,
            )
        else:
            raise ValueError(f"Unsupported scan_family={scan_family}")

        M, K, C, ref_freq_used, natural_freqs, section_source = build_corrected_mkc_from_correction(
            model=model,
            baseline_params=baseline_params,
            correction=correction,
            cfg=cfg,
        )

        u_corr, v_corr, a_corr, core_summary = rollout_with_mkc(
            M=M,
            K=K,
            C=C,
            F_time=F_time,
            u0=u0,
            v0=v0,
            dt=cfg.dt,
            dtype=dtype,
            device=device,
            linear_solve_mode=cfg.linear_solve_mode,
        )

        if not (
            np.all(np.isfinite(u_corr))
            and np.all(np.isfinite(v_corr))
            and np.all(np.isfinite(a_corr))
        ):
            raise FloatingPointError(
                f"Non-finite response detected for {scan_family}={delta_value}"
            )

        u_corr_rs = resample_response_to_time_grid(
            time_src=time_student,
            u_src=u_corr,
            time_dst=time_teacher,
        )

        metrics = compute_student_teacher_6dof_metrics(
            u_student=u_corr_rs,
            u_teacher=u_teacher,
        )

        objective_value = float(metrics[cfg.objective_metric])
        improvement_pct = percent_improvement(
            baseline_value=baseline_objective,
            candidate_value=objective_value,
        )

        response_change_vs_zero = {
            "u_max_abs": float(np.max(np.abs(u_corr - u_zero))),
            "u_mae": float(np.mean(np.abs(u_corr - u_zero))),
            "v_max_abs": float(np.max(np.abs(v_corr - v_zero))),
            "a_max_abs": float(np.max(np.abs(a_corr - a_zero))),
        }

        case_label = make_case_label(scan_family, delta_value)

        print(
            f"  {case_label:<32s} "
            f"{cfg.objective_metric}={objective_value:.12e} "
            f"improve={improvement_pct: .4f}% "
            f"resp_change_u_max={response_change_vs_zero['u_max_abs']:.3e}"
        )

        return {
            "case_label": case_label,
            "scan_family": scan_family,
            "delta_value": float(delta_value),
            "objective_metric": cfg.objective_metric,
            "objective_value": objective_value,
            "objective_improvement_pct_vs_baseline": improvement_pct,
            "metrics": metrics,
            "response_change_vs_zero": response_change_vs_zero,
            "ref_freq_used": ref_freq_used,
            "natural_freqs_hz": natural_freqs,
            "section_source": section_source,
            "core_summary": core_summary,
        }

    for value in flap_values:
        scan_results.append(
            run_one_scan_case(scan_family="delta_EI_flap", delta_value=value)
        )

    for value in edge_values:
        scan_results.append(
            run_one_scan_case(scan_family="delta_EI_edge", delta_value=value)
        )

    # ------------------------------------------------------------------
    # 7) Rank and report.
    # ------------------------------------------------------------------
    print()
    print("[7/7] Ranking scan results")

    ranked = sorted(
        scan_results,
        key=lambda item: float(item["objective_value"]),
    )

    best = ranked[0]
    baseline_rank_item = {
        "case_label": "baseline_direct_student",
        "scan_family": "baseline",
        "delta_value": 0.0,
        "objective_metric": cfg.objective_metric,
        "objective_value": baseline_objective,
        "objective_improvement_pct_vs_baseline": 0.0,
        "metrics": baseline_metrics,
    }

    print()
    print(f"[Best Candidate by {cfg.objective_metric}]")
    print(f"  case_label = {best['case_label']}")
    print(f"  scan_family = {best['scan_family']}")
    print(f"  delta_value = {best['delta_value']}")
    print(f"  objective_value = {best['objective_value']:.12e}")
    print(f"  improvement_pct = {best['objective_improvement_pct_vs_baseline']:.6f}%")
    print_metric_subset("[Best Candidate Metrics]", best["metrics"])

    print()
    print("[Top 10 Candidates]")
    print(
        f"  {'rank':>4s}  {'case_label':<32s} "
        f"{cfg.objective_metric:<18s} {'improve_pct':>12s} "
        f"{'full_x_mae':>14s} {'full_y_mae':>14s} {'tip_y_mae':>14s}"
    )
    for i, item in enumerate(ranked[:10], start=1):
        m = item["metrics"]
        print(
            f"  {i:4d}  {item['case_label']:<32s} "
            f"{item['objective_value']:<18.8e} "
            f"{item['objective_improvement_pct_vs_baseline']:>12.4f} "
            f"{m['full_x_mae']:>14.6e} "
            f"{m['full_y_mae']:>14.6e} "
            f"{m['tip_y_mae']:>14.6e}"
        )

    report = {
        "passed": True,
        "config": asdict(cfg),
        "teacher_case_config": teacher_case_cfg.to_dict(),
        "teacher_result": teacher_result,
        "student_npz": str(student_result["npz"]),
        "student_yaml": str(student_result["yaml"]),
        "teacher_columns_head": teacher_columns[:12],
        "teacher_columns_tail": teacher_columns[-12:],
        "shapes": {
            "time_teacher": list(time_teacher.shape),
            "u_teacher": list(u_teacher.shape),
            "time_student": list(time_student.shape),
            "u_direct": list(u_direct.shape),
            "u_direct_resampled": list(u_direct_rs.shape),
        },
        "baseline": baseline_rank_item,
        "zero_corrected_fem_guard": {
            "metrics": zero_metrics,
            "zero_vs_direct_u_max_abs": zero_vs_direct_diff,
            "ref_freq_used": ref_freq_zero,
            "natural_freqs_hz": freqs_zero,
            "section_source": section_source_zero,
            "core_summary": zero_core_summary,
        },
        "scan_results": scan_results,
        "ranked_results": ranked,
        "best": best,
    }

    if cfg.save_report:
        report_path = output_dir / "manual_section_correction_teacher_alignment_report.json"
        save_json(report_path, report)
        print()
        print(f"[Saved Report] {report_path}")

    print()
    print("✅ PASS: manual section correction teacher-alignment scan completed.")
    if best["objective_improvement_pct_vs_baseline"] > 0.0:
        print("   At least one manual physical section correction improved the objective metric.")
    else:
        print("   No scanned manual correction improved the objective metric; keep baseline or try wider/2D scans.")


if __name__ == "__main__":
    main()