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
from src.student.coupled_fem_builder import build_coupled_fem_matrices_6dof_degrees  # noqa: E402
from src.student.full_order_corrected_core_torch import (  # noqa: E402
    FullOrderCorrectedCoreTorch,
    FullOrderCorrectedCoreTorchConfig,
)
from src.student.spanwise_phi_parameterization import (  # noqa: E402
    SpanwisePhiProfile,
    build_uniform_phi_profile,
    build_linear_phi_profile,
    build_piecewise_constant_phi_profile,
    build_control_point_phi_profile,
    parse_float_list,
)
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
class SpanwisePhiTeacherAlignmentConfig:
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

    x_load_file: str = str(PROJECT_ROOT / "data" / "load" / "simple_tip_fx_case.dat")
    y_load_file: str = str(PROJECT_ROOT / "data" / "load" / "simple_tip_fy_case.dat")

    output_dir: str = str(PROJECT_ROOT / "results" / "student" / "spanwise_phi_teacher_alignment")
    case_prefix: str = "spanwise_phi_teacher_alignment"

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

    torch_dtype: str = "float64"
    device: str = "cpu"
    linear_solve_mode: str = "solve"

    rotate_mass: bool = False
    remove_initial_offset: bool = True
    last_k: int = 5

    # 用于构造 profile 的峰值角度，单位 degree
    profile_peak_values_deg: str = "-10,-7.5,-5,-2.5,0,2.5,5,7.5,10"

    # objective 用哪些 cross/primary ratio 参与平均误差
    ratio_keys: str = "full_cross_to_primary_rms,tip_cross_to_primary_rms,last_k_cross_to_primary_rms"

    save_report: bool = True


def parse_args() -> SpanwisePhiTeacherAlignmentConfig:
    default = SpanwisePhiTeacherAlignmentConfig()

    parser = argparse.ArgumentParser(
        description=(
            "Scan spanwise phi(s) profiles and compare corrected student "
            "x-y cross response ratios against BeamDyn teacher."
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

    mass_group = parser.add_mutually_exclusive_group()
    mass_group.add_argument("--rotate-mass", dest="rotate_mass", action="store_true")
    mass_group.add_argument("--no-rotate-mass", dest="rotate_mass", action="store_false")
    parser.set_defaults(rotate_mass=default.rotate_mass)

    offset_group = parser.add_mutually_exclusive_group()
    offset_group.add_argument("--remove-initial-offset", dest="remove_initial_offset", action="store_true")
    offset_group.add_argument("--keep-initial-offset", dest="remove_initial_offset", action="store_false")
    parser.set_defaults(remove_initial_offset=default.remove_initial_offset)

    parser.add_argument("--last-k", type=int, default=default.last_k)
    parser.add_argument("--profile-peak-values-deg", type=str, default=default.profile_peak_values_deg)
    parser.add_argument("--ratio-keys", type=str, default=default.ratio_keys)

    save_group = parser.add_mutually_exclusive_group()
    save_group.add_argument("--save-report", dest="save_report", action="store_true")
    save_group.add_argument("--no-save-report", dest="save_report", action="store_false")
    parser.set_defaults(save_report=default.save_report)

    args = parser.parse_args()

    return SpanwisePhiTeacherAlignmentConfig(
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
        torch_dtype=args.torch_dtype,
        device=args.device,
        linear_solve_mode=args.linear_solve_mode,
        rotate_mass=args.rotate_mass,
        remove_initial_offset=args.remove_initial_offset,
        last_k=args.last_k,
        profile_peak_values_deg=args.profile_peak_values_deg,
        ratio_keys=args.ratio_keys,
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


def get_torch_dtype(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def remove_initial_offset(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    return u - u[:1, :]


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x**2)))


def safe_ratio(num: float, den: float) -> float:
    num = float(num)
    den = float(den)
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) < 1e-30:
        return float("nan")
    return num / den


def response_coupling_ratios(
    u: np.ndarray,
    *,
    primary_component: str,
    cross_component: str,
    last_k: int,
) -> Dict[str, float]:
    primary_full = get_full_state_component(u, primary_component)
    cross_full = get_full_state_component(u, cross_component)

    primary_tip = get_tip_component(u, primary_component)
    cross_tip = get_tip_component(u, cross_component)

    primary_last_k = get_last_k_component(u, primary_component, k=last_k)
    cross_last_k = get_last_k_component(u, cross_component, k=last_k)

    primary_full_rms = rms(primary_full)
    cross_full_rms = rms(cross_full)
    primary_tip_rms = rms(primary_tip)
    cross_tip_rms = rms(cross_tip)
    primary_last_k_rms = rms(primary_last_k)
    cross_last_k_rms = rms(cross_last_k)

    return {
        "primary_component": primary_component,
        "cross_component": cross_component,

        "primary_full_rms": primary_full_rms,
        "cross_full_rms": cross_full_rms,
        "full_cross_to_primary_rms": safe_ratio(cross_full_rms, primary_full_rms),

        "primary_tip_rms": primary_tip_rms,
        "cross_tip_rms": cross_tip_rms,
        "tip_cross_to_primary_rms": safe_ratio(cross_tip_rms, primary_tip_rms),

        "primary_last_k_rms": primary_last_k_rms,
        "cross_last_k_rms": cross_last_k_rms,
        "last_k_cross_to_primary_rms": safe_ratio(cross_last_k_rms, primary_last_k_rms),
    }


def compute_natural_frequencies_hz(M: np.ndarray, K: np.ndarray, *, num_modes: int = 10) -> np.ndarray:
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    return (
        u_t.detach().cpu().numpy(),
        v_t.detach().cpu().numpy(),
        a_t.detach().cpu().numpy(),
    )


def run_teacher_and_prepare_student_load(
    *,
    case_name: str,
    load_file: Path,
    output_dir: Path,
    cfg: SpanwisePhiTeacherAlignmentConfig,
    teacher_exe: Path,
    template_inp: Path,
    blade_csv: Path,
    primary_component: str,
    cross_component: str,
) -> Dict[str, Any]:
    print()
    print("=" * 100)
    print(f"[Prepare Case] {case_name}")
    print("=" * 100)
    print(f"  load_file = {load_file}")
    print(f"  primary_component = {primary_component}")
    print(f"  cross_component   = {cross_component}")

    print()
    print("[1/4] Running BeamDyn teacher")
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
    print("[2/4] Loading teacher 6DOF response")
    time_teacher, u_teacher, teacher_columns = load_teacher_6dof_response(
        out_path=teacher_out,
        node_start=cfg.teacher_node_start,
        node_end=cfg.teacher_node_end,
        demean=cfg.teacher_demean,
    )

    print(f"  time_teacher shape = {time_teacher.shape}")
    print(f"  u_teacher shape    = {u_teacher.shape}")

    print()
    print("[3/4] Running direct student to get F_time and initial state")
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
    F_time = np.asarray(student_result["F_time"], dtype=np.float64)
    u_direct = np.asarray(student_result["u_full"], dtype=np.float64)
    v_direct = np.asarray(student_result["v_full"], dtype=np.float64)

    print(f"  time_student shape = {time_student.shape}")
    print(f"  F_time shape       = {F_time.shape}")
    print(f"  u_direct shape     = {u_direct.shape}")

    print()
    print("[4/4] Computing teacher target coupling ratios")
    if cfg.remove_initial_offset:
        u_teacher_eval = remove_initial_offset(u_teacher)
    else:
        u_teacher_eval = u_teacher

    teacher_ratios = response_coupling_ratios(
        u_teacher_eval,
        primary_component=primary_component,
        cross_component=cross_component,
        last_k=cfg.last_k,
    )

    print(f"  teacher full cross/primary   = {teacher_ratios['full_cross_to_primary_rms']:.12e}")
    print(f"  teacher tip cross/primary    = {teacher_ratios['tip_cross_to_primary_rms']:.12e}")
    print(f"  teacher last-k cross/primary = {teacher_ratios['last_k_cross_to_primary_rms']:.12e}")

    return {
        "case_name": case_name,
        "load_file": str(load_file),
        "primary_component": primary_component,
        "cross_component": cross_component,
        "teacher_result": teacher_result,
        "teacher_columns_head": teacher_columns[:12],
        "teacher_columns_tail": teacher_columns[-12:],
        "time_teacher": time_teacher,
        "u_teacher": u_teacher,
        "teacher_ratios": teacher_ratios,
        "student_npz": str(student_result["npz"]),
        "student_yaml": str(student_result["yaml"]),
        "time_student": time_student,
        "F_time": F_time,
        "u0": u_direct[0].copy(),
        "v0": v_direct[0].copy(),
    }


def build_profile_candidates(model, peak_values: List[float]) -> List[SpanwisePhiProfile]:
    profiles: List[SpanwisePhiProfile] = []

    # 避免重复 zero
    seen_names = set()

    def add(profile: SpanwisePhiProfile) -> None:
        if profile.name in seen_names:
            return
        profiles.append(profile)
        seen_names.add(profile.name)

    for peak in peak_values:
        label = format_phi_label(peak)
        add(build_uniform_phi_profile(model, phi_deg=peak, name=f"uniform_{label}"))

    for peak in peak_values:
        label = format_phi_label(peak)
        add(
            build_linear_phi_profile(
                model,
                root_phi_deg=0.0,
                tip_phi_deg=peak,
                name=f"linear_root0_tip_{label}",
            )
        )
        add(
            build_linear_phi_profile(
                model,
                root_phi_deg=peak,
                tip_phi_deg=0.0,
                name=f"linear_root_{label}_tip0",
            )
        )

    breakpoints = [0.0, 0.25, 0.5, 0.75, 1.0]

    for peak in peak_values:
        label = format_phi_label(peak)

        # tip-heavy: root 小，tip 大
        add(
            build_piecewise_constant_phi_profile(
                model,
                breakpoints=breakpoints,
                values_deg=[0.0, 0.25 * peak, 0.5 * peak, 0.8 * peak],
                name=f"piecewise_tip_heavy_{label}",
            )
        )

        # root-heavy: root 大，tip 小
        add(
            build_piecewise_constant_phi_profile(
                model,
                breakpoints=breakpoints,
                values_deg=[0.8 * peak, 0.5 * peak, 0.25 * peak, 0.0],
                name=f"piecewise_root_heavy_{label}",
            )
        )

        # control-point profiles
        add(
            build_control_point_phi_profile(
                model,
                control_eta=[0.0, 0.25, 0.5, 0.75, 1.0],
                control_phi_deg=[0.0, 0.2 * peak, 0.5 * peak, 0.8 * peak, peak],
                name=f"control_tip_heavy_{label}",
            )
        )

        add(
            build_control_point_phi_profile(
                model,
                control_eta=[0.0, 0.25, 0.5, 0.75, 1.0],
                control_phi_deg=[peak, 0.8 * peak, 0.5 * peak, 0.2 * peak, 0.0],
                name=f"control_root_heavy_{label}",
            )
        )

    return profiles


def format_phi_label(value: float) -> str:
    sign = "p" if float(value) >= 0 else "m"
    text = f"{abs(float(value)):.6f}".replace(".", "p")
    return f"{sign}{text}"


def evaluate_profile_on_case(
    *,
    profile: SpanwisePhiProfile,
    model,
    case_data: Dict[str, Any],
    cfg: SpanwisePhiTeacherAlignmentConfig,
    dtype: torch.dtype,
    device: torch.device,
    ratio_keys: List[str],
) -> Dict[str, Any]:
    M, K, debug = build_coupled_fem_matrices_6dof_degrees(
        model,
        section_params=None,
        phi_deg=profile.phi_deg,
        alpha_flap=cfg.alpha_flap,
        alpha_edge=cfg.alpha_edge,
        alpha_torsion=cfg.alpha_torsion,
        rotate_mass=cfg.rotate_mass,
        return_full=True,
    )

    C, ref_freq_used, natural_freqs = build_damping_like_student(
        M=M,
        K=K,
        zeta_structural=cfg.zeta_structural,
        ref_freq_hz=cfg.ref_freq_hz,
    )

    u, v, a = rollout_with_mkc(
        M=M,
        K=K,
        C=C,
        F_time=case_data["F_time"],
        u0=case_data["u0"],
        v0=case_data["v0"],
        dt=cfg.dt,
        dtype=dtype,
        device=device,
        linear_solve_mode=cfg.linear_solve_mode,
    )

    if cfg.remove_initial_offset:
        u_eval = remove_initial_offset(u)
    else:
        u_eval = u

    student_ratios = response_coupling_ratios(
        u_eval,
        primary_component=case_data["primary_component"],
        cross_component=case_data["cross_component"],
        last_k=cfg.last_k,
    )

    teacher_ratios = case_data["teacher_ratios"]

    ratio_errors = {}
    ratio_abs_errors = []
    for key in ratio_keys:
        err = float(student_ratios[key] - teacher_ratios[key])
        ratio_errors[key] = {
            "student": float(student_ratios[key]),
            "teacher": float(teacher_ratios[key]),
            "signed_error": err,
            "abs_error": abs(err),
        }
        ratio_abs_errors.append(abs(err))

    objective = float(np.mean(ratio_abs_errors))

    return {
        "case_name": case_data["case_name"],
        "primary_component": case_data["primary_component"],
        "cross_component": case_data["cross_component"],
        "objective": objective,
        "ratio_errors": ratio_errors,
        "student_ratios": student_ratios,
        "teacher_ratios": teacher_ratios,
        "ref_freq_used": ref_freq_used,
        "natural_freqs_hz": natural_freqs,
        "profile_summary": profile.summary(),
        "debug_summary": {
            "builder": debug["builder"],
            "rotate_mass": debug["rotate_mass"],
            "phi_element_deg_head": debug["phi_element_deg"][:10],
            "phi_element_deg_tail": debug["phi_element_deg"][-10:],
        },
        "finite": bool(
            np.all(np.isfinite(u))
            and np.all(np.isfinite(v))
            and np.all(np.isfinite(a))
        ),
    }


def main() -> None:
    cfg = parse_args()

    teacher_exe = assert_existing_file(cfg.teacher_exe, "teacher_exe")
    template_inp = assert_existing_file(cfg.template_inp, "template_inp")
    blade_csv = assert_existing_file(cfg.blade_csv, "blade_csv")
    x_load_file = assert_existing_file(cfg.x_load_file, "x_load_file")
    y_load_file = assert_existing_file(cfg.y_load_file, "y_load_file")

    output_dir = Path(cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = get_torch_dtype(cfg.torch_dtype)
    device = torch.device(cfg.device)

    peak_values = parse_float_list(cfg.profile_peak_values_deg)
    ratio_keys = [item.strip() for item in str(cfg.ratio_keys).split(",") if item.strip()]

    print()
    print("[Spanwise Phi Teacher Alignment Scan]")
    print()
    print("[Config]")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")

    print()
    print(f"[Profile Peak Values] {peak_values}")
    print(f"[Ratio Keys] {ratio_keys}")

    print()
    print("[1/5] Loading StudentBeamModel")
    model = load_student_model_from_blade_master(
        csv_path=str(blade_csv),
        model_name="spanwise_phi_teacher_alignment_model",
    )
    print(f"  span_m = {model.span_m}")
    print(f"  n_stations = {model.n_stations}")
    print(f"  n_elements = {model.n_stations - 1}")

    print()
    print("[2/5] Preparing x-only and y-only teacher/student cases")
    x_case = run_teacher_and_prepare_student_load(
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

    y_case = run_teacher_and_prepare_student_load(
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

    print()
    print("[3/5] Building profile candidates")
    profiles = build_profile_candidates(model, peak_values)
    print(f"  n_profiles = {len(profiles)}")
    for p in profiles[:10]:
        s = p.summary()
        print(
            f"  {p.name:<36s} "
            f"min={s['station_phi_deg_min']:.3f}, "
            f"max={s['station_phi_deg_max']:.3f}, "
            f"mean={s['station_phi_deg_mean']:.3f}"
        )
    if len(profiles) > 10:
        print(f"  ... {len(profiles) - 10} more profiles")

    print()
    print("[4/5] Evaluating profiles against teacher cross-response ratios")

    results = []

    for idx, profile in enumerate(profiles, start=1):
        x_eval = evaluate_profile_on_case(
            profile=profile,
            model=model,
            case_data=x_case,
            cfg=cfg,
            dtype=dtype,
            device=device,
            ratio_keys=ratio_keys,
        )
        y_eval = evaluate_profile_on_case(
            profile=profile,
            model=model,
            case_data=y_case,
            cfg=cfg,
            dtype=dtype,
            device=device,
            ratio_keys=ratio_keys,
        )

        combined_objective = float(0.5 * (x_eval["objective"] + y_eval["objective"]))

        item = {
            "profile_name": profile.name,
            "profile": profile.to_dict(),
            "combined_objective": combined_objective,
            "x_only_objective": x_eval["objective"],
            "y_only_objective": y_eval["objective"],
            "x_only": x_eval,
            "y_only": y_eval,
            "finite": bool(x_eval["finite"] and y_eval["finite"]),
        }
        results.append(item)

        print(
            f"  [{idx:03d}/{len(profiles):03d}] "
            f"{profile.name:<36s} "
            f"obj={combined_objective:.8e} "
            f"x_obj={x_eval['objective']:.8e} "
            f"y_obj={y_eval['objective']:.8e}"
        )

    print()
    print("[5/5] Ranking profiles")
    ranked = sorted(results, key=lambda item: float(item["combined_objective"]))
    best = ranked[0]

    print()
    print("[Best Profile]")
    print(f"  profile_name = {best['profile_name']}")
    print(f"  combined_objective = {best['combined_objective']:.12e}")
    print(f"  x_only_objective = {best['x_only_objective']:.12e}")
    print(f"  y_only_objective = {best['y_only_objective']:.12e}")
    print("  profile summary:")
    for k, v in best["profile"]["summary"].items():
        print(f"    {k}: {v}")

    print()
    print("[Best Profile Ratio Comparison]")
    for case_key in ["x_only", "y_only"]:
        case = best[case_key]
        print(f"  {case_key}:")
        for key in ratio_keys:
            e = case["ratio_errors"][key]
            print(
                f"    {key:<32s} "
                f"teacher={e['teacher']:.8e}, "
                f"student={e['student']:.8e}, "
                f"abs_err={e['abs_error']:.8e}"
            )

    print()
    print("[Top 10 Profiles]")
    print(
        f"  {'rank':>4s}  {'profile_name':<40s} "
        f"{'combined_obj':>14s} {'x_obj':>14s} {'y_obj':>14s} "
        f"{'phi_min':>10s} {'phi_max':>10s} {'phi_mean':>10s}"
    )
    for i, item in enumerate(ranked[:10], start=1):
        s = item["profile"]["summary"]
        print(
            f"  {i:4d}  {item['profile_name']:<40s} "
            f"{item['combined_objective']:>14.6e} "
            f"{item['x_only_objective']:>14.6e} "
            f"{item['y_only_objective']:>14.6e} "
            f"{s['station_phi_deg_min']:>10.3f} "
            f"{s['station_phi_deg_max']:>10.3f} "
            f"{s['station_phi_deg_mean']:>10.3f}"
        )

    report = {
        "passed": True,
        "config": asdict(cfg),
        "ratio_keys": ratio_keys,
        "teacher_targets": {
            "x_only": x_case["teacher_ratios"],
            "y_only": y_case["teacher_ratios"],
        },
        "case_info": {
            "x_only": {
                "case_name": x_case["case_name"],
                "load_file": x_case["load_file"],
                "teacher_result": x_case["teacher_result"],
                "student_npz": x_case["student_npz"],
                "student_yaml": x_case["student_yaml"],
            },
            "y_only": {
                "case_name": y_case["case_name"],
                "load_file": y_case["load_file"],
                "teacher_result": y_case["teacher_result"],
                "student_npz": y_case["student_npz"],
                "student_yaml": y_case["student_yaml"],
            },
        },
        "n_profiles": len(profiles),
        "results": results,
        "ranked_results": ranked,
        "best": best,
    }

    if cfg.save_report:
        report_path = output_dir / "spanwise_phi_teacher_alignment_report.json"
        save_json(report_path, report)
        print()
        print(f"[Saved Report] {report_path}")

    print()
    print("✅ PASS: spanwise phi teacher-alignment scan completed.")
    print("   已完成不同 phi(s) profile 对 teacher x-y 交叉响应比例的手动对齐扫描。")


if __name__ == "__main__":
    main()