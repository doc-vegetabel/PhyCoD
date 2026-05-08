from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
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
from src.teacher.beamdyn_teacher_adapter import (  # noqa: E402
    BeamDynTeacherCaseConfig,
    run_teacher_case_beamdyn,
    load_teacher_6dof_response,
    resample_response_to_time_grid,
    get_tip_component,
)


@dataclass
class KappaYScaleResponseCompareConfig:
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
    load_file: str = str(PROJECT_ROOT / "data" / "load" / "simple_tip_fy_case.dat")

    output_dir: str = str(PROJECT_ROOT / "results" / "student" / "kappa_y_scale_response_compare")
    case_name: str = "kappa_y_scale_response_compare"

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

    base_phi_twist_column: str = "initial_twist_deg"
    base_phi_sign: float = -1.0
    rotate_mass: bool = False

    kappa_y_scale: float = 0.96
    scale_mode: str = "y_bending"  # choices: uy_only, y_bending

    torch_dtype: str = "float64"
    device: str = "cpu"
    linear_solve_mode: str = "solve"

    remove_initial_offset: bool = True
    last_k: int = 5

    show_plot: bool = False
    save_report: bool = True


def parse_args() -> KappaYScaleResponseCompareConfig:
    d = KappaYScaleResponseCompareConfig()

    parser = argparse.ArgumentParser(
        description=(
            "Compare Teacher / base student / kappa-y-scaled student time histories "
            "under a y-only load case."
        )
    )

    parser.add_argument("--teacher-exe", type=str, default=d.teacher_exe)
    parser.add_argument("--template-inp", type=str, default=d.template_inp)
    parser.add_argument("--blade-csv", type=str, default=d.blade_csv)
    parser.add_argument("--load-file", type=str, default=d.load_file)

    parser.add_argument("--output-dir", type=str, default=d.output_dir)
    parser.add_argument("--case-name", type=str, default=d.case_name)

    parser.add_argument("--t-initial", type=float, default=d.t_initial)
    parser.add_argument("--t-final", type=float, default=d.t_final)
    parser.add_argument("--dt", type=float, default=d.dt)

    parser.add_argument("--teacher-node-start", type=int, default=d.teacher_node_start)
    parser.add_argument("--teacher-node-end", type=int, default=d.teacher_node_end)

    demean_group = parser.add_mutually_exclusive_group()
    demean_group.add_argument("--teacher-demean", dest="teacher_demean", action="store_true")
    demean_group.add_argument("--no-teacher-demean", dest="teacher_demean", action="store_false")
    parser.set_defaults(teacher_demean=d.teacher_demean)

    parser.add_argument("--alpha-flap", type=float, default=d.alpha_flap)
    parser.add_argument("--alpha-edge", type=float, default=d.alpha_edge)
    parser.add_argument("--alpha-torsion", type=float, default=d.alpha_torsion)
    parser.add_argument("--zeta-structural", type=float, default=d.zeta_structural)
    parser.add_argument("--ref-freq-hz", type=float, default=d.ref_freq_hz)

    parser.add_argument("--base-phi-twist-column", type=str, default=d.base_phi_twist_column)
    parser.add_argument("--base-phi-sign", type=float, default=d.base_phi_sign)

    rotate_group = parser.add_mutually_exclusive_group()
    rotate_group.add_argument("--rotate-mass", dest="rotate_mass", action="store_true")
    rotate_group.add_argument("--no-rotate-mass", dest="rotate_mass", action="store_false")
    parser.set_defaults(rotate_mass=d.rotate_mass)

    parser.add_argument("--kappa-y-scale", type=float, default=d.kappa_y_scale)
    parser.add_argument("--scale-mode", type=str, default=d.scale_mode, choices=["uy_only", "y_bending"])

    parser.add_argument("--torch-dtype", type=str, default=d.torch_dtype, choices=["float64", "float32"])
    parser.add_argument("--device", type=str, default=d.device)
    parser.add_argument("--linear-solve-mode", type=str, default=d.linear_solve_mode, choices=["solve", "inverse"])

    offset_group = parser.add_mutually_exclusive_group()
    offset_group.add_argument("--remove-initial-offset", dest="remove_initial_offset", action="store_true")
    offset_group.add_argument("--keep-initial-offset", dest="remove_initial_offset", action="store_false")
    parser.set_defaults(remove_initial_offset=d.remove_initial_offset)

    parser.add_argument("--last-k", type=int, default=d.last_k)

    show_group = parser.add_mutually_exclusive_group()
    show_group.add_argument("--show-plot", dest="show_plot", action="store_true")
    show_group.add_argument("--no-show-plot", dest="show_plot", action="store_false")
    parser.set_defaults(show_plot=d.show_plot)

    report_group = parser.add_mutually_exclusive_group()
    report_group.add_argument("--save-report", dest="save_report", action="store_true")
    report_group.add_argument("--no-save-report", dest="save_report", action="store_false")
    parser.set_defaults(save_report=d.save_report)

    args = parser.parse_args()

    return KappaYScaleResponseCompareConfig(
        teacher_exe=args.teacher_exe,
        template_inp=args.template_inp,
        blade_csv=args.blade_csv,
        load_file=args.load_file,
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
        base_phi_twist_column=args.base_phi_twist_column,
        base_phi_sign=args.base_phi_sign,
        rotate_mass=args.rotate_mass,
        kappa_y_scale=args.kappa_y_scale,
        scale_mode=args.scale_mode,
        torch_dtype=args.torch_dtype,
        device=args.device,
        linear_solve_mode=args.linear_solve_mode,
        remove_initial_offset=args.remove_initial_offset,
        last_k=args.last_k,
        show_plot=args.show_plot,
        save_report=args.save_report,
    )


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


def load_column_from_csv(csv_path: Path, column_name: str) -> np.ndarray:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in CSV: {csv_path}")
        if column_name not in reader.fieldnames:
            raise KeyError(
                f"Column '{column_name}' not found in {csv_path}. "
                f"Available columns: {reader.fieldnames}"
            )

        values = []
        for row in reader:
            values.append(float(row[column_name]))

    return np.asarray(values, dtype=np.float64)


def build_element_phi_from_initial_twist(
    *,
    blade_csv: Path,
    twist_column: str,
    sign: float,
    n_elements: int,
) -> np.ndarray:
    station_twist = load_column_from_csv(blade_csv, twist_column)

    if station_twist.size == n_elements + 1:
        element_twist = 0.5 * (station_twist[:-1] + station_twist[1:])
    elif station_twist.size == n_elements:
        element_twist = station_twist.copy()
    else:
        raise ValueError(
            f"Unexpected twist length. Got {station_twist.size}, "
            f"expected n_elements={n_elements} or n_elements+1={n_elements + 1}."
        )

    return float(sign) * element_twist


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


def selected_y_dof_indices(n_dofs: int, scale_mode: str) -> np.ndarray:
    if n_dofs % 6 != 0:
        raise ValueError(f"n_dofs must be divisible by 6, got {n_dofs}")

    n_nodes = n_dofs // 6

    uy = np.array([i * 6 + 1 for i in range(n_nodes)], dtype=int)

    if scale_mode == "uy_only":
        return uy

    if scale_mode == "y_bending":
        # 对沿 z 展向的梁，y 向弯曲通常主要涉及 uy 和绕 x 的截面转角 rx。
        rx = np.array([i * 6 + 3 for i in range(n_nodes)], dtype=int)
        return np.unique(np.concatenate([uy, rx]))

    raise ValueError(f"Unsupported scale_mode: {scale_mode}")


def apply_congruent_dof_stiffness_scale(
    K: np.ndarray,
    *,
    scale: float,
    dof_indices: np.ndarray,
) -> np.ndarray:
    if scale <= 0.0:
        raise ValueError(f"scale must be positive, got {scale}")

    K = np.asarray(K, dtype=np.float64)

    s = np.ones(K.shape[0], dtype=np.float64)
    s[dof_indices] = np.sqrt(float(scale))

    # K_scaled = S K S，保持刚度矩阵对称。
    K_scaled = (s[:, None] * K) * s[None, :]
    return 0.5 * (K_scaled + K_scaled.T)


def component_indices(n_nodes: int, component: str) -> np.ndarray:
    mapping = {
        "x": 0,
        "y": 1,
        "z": 2,
        "rx": 3,
        "ry": 4,
        "rz": 5,
    }
    if component not in mapping:
        raise ValueError(f"Unsupported component: {component}")
    return np.array([i * 6 + mapping[component] for i in range(n_nodes)], dtype=int)


def get_component_matrix(u_full: np.ndarray, component: str) -> np.ndarray:
    u_full = np.asarray(u_full, dtype=np.float64)
    if u_full.ndim != 2:
        raise ValueError(f"u_full must be 2D, got {u_full.shape}")

    n_dofs = u_full.shape[1]
    if n_dofs % 6 != 0:
        raise ValueError(f"n_dofs must be divisible by 6, got {n_dofs}")

    n_nodes = n_dofs // 6
    idx = component_indices(n_nodes, component)
    return u_full[:, idx]


def get_lastk_component_mean(
    u_full: np.ndarray,
    component: str,
    *,
    last_k: int,
) -> np.ndarray:
    comp = get_component_matrix(u_full, component)
    n_nodes = comp.shape[1]

    if last_k <= 0 or last_k > n_nodes:
        raise ValueError(f"Invalid last_k={last_k}, n_nodes={n_nodes}")

    return np.mean(comp[:, -last_k:], axis=1)


def compute_array_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}")

    err = pred - target
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))
    target_rms = float(np.sqrt(np.mean(target ** 2)))
    pred_rms = float(np.sqrt(np.mean(pred ** 2)))

    nrmse = rmse / (target_rms + 1.0e-12)
    rms_ratio = pred_rms / (target_rms + 1.0e-12)

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "target_rms": target_rms,
        "pred_rms": pred_rms,
        "nrmse_by_target_rms": float(nrmse),
        "pred_over_target_rms": float(rms_ratio),
    }


def compute_response_metrics(
    *,
    pred: np.ndarray,
    target: np.ndarray,
    last_k: int,
) -> Dict[str, Dict[str, float]]:
    return {
        "full_x": compute_array_metrics(
            get_component_matrix(pred, "x"),
            get_component_matrix(target, "x"),
        ),
        "full_y": compute_array_metrics(
            get_component_matrix(pred, "y"),
            get_component_matrix(target, "y"),
        ),
        "tip_x": compute_array_metrics(
            get_tip_component(pred, "x"),
            get_tip_component(target, "x"),
        ),
        "tip_y": compute_array_metrics(
            get_tip_component(pred, "y"),
            get_tip_component(target, "y"),
        ),
        f"last{last_k}_mean_x": compute_array_metrics(
            get_lastk_component_mean(pred, "x", last_k=last_k),
            get_lastk_component_mean(target, "x", last_k=last_k),
        ),
        f"last{last_k}_mean_y": compute_array_metrics(
            get_lastk_component_mean(pred, "y", last_k=last_k),
            get_lastk_component_mean(target, "y", last_k=last_k),
        ),
    }


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
) -> np.ndarray:
    core = FullOrderCorrectedCoreTorch(
        M=M,
        K=K,
        C=C,
        dt=float(dt),
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
        u_t, _, _ = core.rollout(
            F_time=F_torch,
            u0=u0_torch,
            v0=v0_torch,
            theta_full=None,
            return_debug=False,
        )

    return u_t.detach().cpu().numpy()


def plot_three_time_histories(
    *,
    time: np.ndarray,
    teacher_signal: np.ndarray,
    base_signal: np.ndarray,
    corrected_signal: np.ndarray,
    title: str,
    ylabel: str,
    corrected_label: str,
    save_path: Path,
    show_plot: bool,
) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(time, teacher_signal, label="Teacher")
    plt.plot(time, base_signal, label="Base student")
    plt.plot(time, corrected_signal, label=corrected_label)
    plt.xlabel("Time [s]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    if show_plot:
        plt.show()
    plt.close()


def print_metric_block(title: str, metrics: Dict[str, Dict[str, float]]) -> None:
    print()
    print(title)
    for key, m in metrics.items():
        print(
            f"  {key:<16s} "
            f"mae={m['mae']:.8e}, "
            f"rmse={m['rmse']:.8e}, "
            f"nrmse={m['nrmse_by_target_rms']:.8e}, "
            f"pred/target_rms={m['pred_over_target_rms']:.8e}"
        )


def main() -> None:
    cfg = parse_args()

    teacher_exe = assert_existing_file(cfg.teacher_exe, "teacher_exe")
    template_inp = assert_existing_file(cfg.template_inp, "template_inp")
    blade_csv = assert_existing_file(cfg.blade_csv, "blade_csv")
    load_file = assert_existing_file(cfg.load_file, "load_file")

    output_dir = Path(cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = get_torch_dtype(cfg.torch_dtype)
    device = torch.device(cfg.device)

    print()
    print("[Kappa-y Scale Response Compare]")
    print()
    print("[Config]")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")

    print()
    print("[1/6] Running BeamDyn teacher")
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
        raise RuntimeError("BeamDyn teacher did not return .out path.")
    print(f"  teacher_out = {teacher_out}")

    print()
    print("[2/6] Loading teacher response")
    time_teacher, u_teacher, _ = load_teacher_6dof_response(
        out_path=teacher_out,
        node_start=cfg.teacher_node_start,
        node_end=cfg.teacher_node_end,
        demean=cfg.teacher_demean,
    )
    print(f"  time_teacher shape = {time_teacher.shape}")
    print(f"  u_teacher shape    = {u_teacher.shape}")

    print()
    print("[3/6] Running base student")
    base_result = run_student_case(
        blade_csv=blade_csv,
        output_dir=output_dir / "base_student",
        case_name=f"{cfg.case_name}_base",
        use_time_series_load=True,
        t_initial=cfg.t_initial,
        t_final=cfg.t_final,
        dt=cfg.dt,
        time_series_load_file=load_file,
        student_params={
            "alpha_flap": cfg.alpha_flap,
            "alpha_edge": cfg.alpha_edge,
            "alpha_torsion": cfg.alpha_torsion,
            "zeta_structural": cfg.zeta_structural,
            "ref_freq_hz": cfg.ref_freq_hz,
        },
        u0_trans=None,
        v0_trans=None,
    )

    time_student = np.asarray(base_result["time"], dtype=np.float64)
    F_time = np.asarray(base_result["F_time"], dtype=np.float64)
    u_base = np.asarray(base_result["u_full"], dtype=np.float64)
    v_base = np.asarray(base_result["v_full"], dtype=np.float64)

    print(f"  time_student shape = {time_student.shape}")
    print(f"  F_time shape       = {F_time.shape}")
    print(f"  u_base shape       = {u_base.shape}")

    print()
    print("[4/6] Building kappa-y scaled student M/K/C and rollout")

    model = load_student_model_from_blade_master(
        csv_path=str(blade_csv),
        model_name="kappa_y_scale_response_compare_model",
    )

    n_elements = int(model.n_stations - 1)
    phi_element_deg = build_element_phi_from_initial_twist(
        blade_csv=blade_csv,
        twist_column=cfg.base_phi_twist_column,
        sign=cfg.base_phi_sign,
        n_elements=n_elements,
    )

    print(
        "  Phi_base element deg: "
        f"min={phi_element_deg.min():.6f}, "
        f"max={phi_element_deg.max():.6f}, "
        f"mean={phi_element_deg.mean():.6f}"
    )

    M_base, K_base, _ = build_coupled_fem_matrices_6dof_degrees(
        model,
        section_params=None,
        phi_deg=phi_element_deg,
        alpha_flap=cfg.alpha_flap,
        alpha_edge=cfg.alpha_edge,
        alpha_torsion=cfg.alpha_torsion,
        rotate_mass=cfg.rotate_mass,
        return_full=True,
    )

    M_base = np.asarray(M_base, dtype=np.float64)
    K_base = np.asarray(K_base, dtype=np.float64)

    scaled_dofs = selected_y_dof_indices(K_base.shape[0], cfg.scale_mode)
    K_scaled = apply_congruent_dof_stiffness_scale(
        K_base,
        scale=cfg.kappa_y_scale,
        dof_indices=scaled_dofs,
    )

    C_scaled, ref_freq_used, natural_freqs_scaled = build_damping_like_student(
        M=M_base,
        K=K_scaled,
        zeta_structural=cfg.zeta_structural,
        ref_freq_hz=cfg.ref_freq_hz,
    )

    print(f"  scale_mode       = {cfg.scale_mode}")
    print(f"  kappa_y_scale    = {cfg.kappa_y_scale}")
    print(f"  n_scaled_dofs    = {scaled_dofs.size}")
    print(f"  natural_freqs[:5] = {natural_freqs_scaled[:5]}")
    print(f"  ref_freq_used     = {ref_freq_used}")

    u_corrected = rollout_with_mkc(
        M=M_base,
        K=K_scaled,
        C=C_scaled,
        F_time=F_time,
        u0=u_base[0].copy(),
        v0=v_base[0].copy(),
        dt=cfg.dt,
        dtype=dtype,
        device=device,
        linear_solve_mode=cfg.linear_solve_mode,
    )

    print(f"  u_corrected shape = {u_corrected.shape}")

    print()
    print("[5/6] Resampling teacher and computing metrics")

    u_teacher_rs = resample_response_to_time_grid(
        time_src=time_teacher,
        u_src=u_teacher,
        time_dst=time_student,
    )

    if cfg.remove_initial_offset:
        u_teacher_eval = remove_initial_offset(u_teacher_rs)
        u_base_eval = remove_initial_offset(u_base)
        u_corrected_eval = remove_initial_offset(u_corrected)
    else:
        u_teacher_eval = u_teacher_rs
        u_base_eval = u_base
        u_corrected_eval = u_corrected

    base_metrics = compute_response_metrics(
        pred=u_base_eval,
        target=u_teacher_eval,
        last_k=cfg.last_k,
    )
    corrected_metrics = compute_response_metrics(
        pred=u_corrected_eval,
        target=u_teacher_eval,
        last_k=cfg.last_k,
    )

    print_metric_block("[Base Student vs Teacher]", base_metrics)
    print_metric_block("[Kappa-y Scaled Student vs Teacher]", corrected_metrics)

    print()
    print("[Metric Improvement: positive means corrected is better]")
    for key in ["full_y", "tip_y", f"last{cfg.last_k}_mean_y", "full_x", "tip_x", f"last{cfg.last_k}_mean_x"]:
        base_rmse = base_metrics[key]["rmse"]
        corr_rmse = corrected_metrics[key]["rmse"]
        improvement = (base_rmse - corr_rmse) / (base_rmse + 1.0e-12)
        print(f"  {key:<16s} rmse_improvement = {improvement:.8e}")

    print()
    print("[6/6] Saving figures and data")

    teacher_tip_y = get_tip_component(u_teacher_eval, "y")
    base_tip_y = get_tip_component(u_base_eval, "y")
    corrected_tip_y = get_tip_component(u_corrected_eval, "y")

    teacher_lastk_y = get_lastk_component_mean(u_teacher_eval, "y", last_k=cfg.last_k)
    base_lastk_y = get_lastk_component_mean(u_base_eval, "y", last_k=cfg.last_k)
    corrected_lastk_y = get_lastk_component_mean(u_corrected_eval, "y", last_k=cfg.last_k)

    teacher_tip_x = get_tip_component(u_teacher_eval, "x")
    base_tip_x = get_tip_component(u_base_eval, "x")
    corrected_tip_x = get_tip_component(u_corrected_eval, "x")

    tip_y_plot = output_dir / "tip_y_three_way_time_history.png"
    lastk_y_plot = output_dir / f"last{cfg.last_k}_mean_y_three_way_time_history.png"
    tip_x_guard_plot = output_dir / "tip_x_guard_three_way_time_history.png"

    corrected_label = f"Student kappa_y_scale={cfg.kappa_y_scale:g}"

    plot_three_time_histories(
        time=time_student,
        teacher_signal=teacher_tip_y,
        base_signal=base_tip_y,
        corrected_signal=corrected_tip_y,
        title=f"{cfg.case_name} - Tip y displacement",
        ylabel="Tip y displacement",
        corrected_label=corrected_label,
        save_path=tip_y_plot,
        show_plot=cfg.show_plot,
    )

    plot_three_time_histories(
        time=time_student,
        teacher_signal=teacher_lastk_y,
        base_signal=base_lastk_y,
        corrected_signal=corrected_lastk_y,
        title=f"{cfg.case_name} - Last-{cfg.last_k} mean y displacement",
        ylabel=f"Last-{cfg.last_k} mean y displacement",
        corrected_label=corrected_label,
        save_path=lastk_y_plot,
        show_plot=cfg.show_plot,
    )

    plot_three_time_histories(
        time=time_student,
        teacher_signal=teacher_tip_x,
        base_signal=base_tip_x,
        corrected_signal=corrected_tip_x,
        title=f"{cfg.case_name} - Tip x guard displacement",
        ylabel="Tip x displacement",
        corrected_label=corrected_label,
        save_path=tip_x_guard_plot,
        show_plot=cfg.show_plot,
    )

    data_path = output_dir / "kappa_y_scale_response_compare_data.npz"
    np.savez(
        data_path,
        time=time_student,
        u_teacher=u_teacher_eval,
        u_base=u_base_eval,
        u_corrected=u_corrected_eval,
        teacher_tip_y=teacher_tip_y,
        base_tip_y=base_tip_y,
        corrected_tip_y=corrected_tip_y,
        teacher_lastk_y=teacher_lastk_y,
        base_lastk_y=base_lastk_y,
        corrected_lastk_y=corrected_lastk_y,
        teacher_tip_x=teacher_tip_x,
        base_tip_x=base_tip_x,
        corrected_tip_x=corrected_tip_x,
        phi_element_deg=phi_element_deg,
        kappa_y_scale=np.array([cfg.kappa_y_scale], dtype=np.float64),
        natural_freqs_scaled=natural_freqs_scaled,
    )

    report = {
        "config": asdict(cfg),
        "files": {
            "teacher_out": str(teacher_out),
            "base_student_npz": str(base_result["npz"]),
            "base_student_yaml": str(base_result["yaml"]),
            "data_npz": str(data_path),
            "tip_y_plot": str(tip_y_plot),
            "lastk_y_plot": str(lastk_y_plot),
            "tip_x_guard_plot": str(tip_x_guard_plot),
        },
        "kappa_y_scaled_student": {
            "kappa_y_scale": float(cfg.kappa_y_scale),
            "scale_mode": cfg.scale_mode,
            "n_scaled_dofs": int(scaled_dofs.size),
            "natural_freqs_scaled_hz": [float(x) for x in natural_freqs_scaled.tolist()],
            "ref_freq_used": None if ref_freq_used is None else float(ref_freq_used),
        },
        "metrics": {
            "base_vs_teacher": base_metrics,
            "kappa_y_scaled_vs_teacher": corrected_metrics,
        },
        "rmse_improvement": {
            key: float((base_metrics[key]["rmse"] - corrected_metrics[key]["rmse"]) / (base_metrics[key]["rmse"] + 1.0e-12))
            for key in base_metrics.keys()
        },
        "interpretation": {
            "purpose": "Compare time histories under y-only load. This is a diagnostic test, not a training script.",
            "next_step": "If kappa_y_scale improves y time-history error without damaging x guard metrics, implement kappa_y_residual(s) in static training.",
        },
    }

    if cfg.save_report:
        report_path = output_dir / "kappa_y_scale_response_compare_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print()
        print("[Saved Report]")
        print(f"  {report_path}")

    print()
    print("[Saved Figures]")
    print(f"  tip y plot     = {tip_y_plot}")
    print(f"  last-k y plot  = {lastk_y_plot}")
    print(f"  tip x guard    = {tip_x_guard_plot}")

    print()
    print("[Saved Data]")
    print(f"  data npz = {data_path}")

    print()
    print("✅ PASS: kappa-y scale response comparison completed.")
    print("   已完成 Teacher / base student / kappa_y corrected student 三者在 y-only 载荷下的时历对比。")


if __name__ == "__main__":
    main()