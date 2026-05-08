from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.linalg import eigh


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# 你后续只需要改这里的载荷文件列表
TRAIN_LOAD_FILES = [
    str(PROJECT_ROOT / "data" / "load" / "simple_tip_fy_case.dat"),
]

VALID_LOAD_FILES = [
    # 可先留空；后续建议放独立 y-only 验证工况
    # str(PROJECT_ROOT / "data" / "load" / "test_complex_case.dat"),
]


from scripts.run_student_cases import (  # noqa: E402
    run_student_case,
    _build_structural_damping_matrix,
)
from src.student.io import load_student_model_from_blade_master  # noqa: E402
from src.student.coupled_fem_builder import build_coupled_fem_matrices_6dof_degrees  # noqa: E402
from src.teacher.beamdyn_teacher_adapter import (  # noqa: E402
    BeamDynTeacherCaseConfig,
    run_teacher_case_beamdyn,
    load_teacher_6dof_response,
    resample_response_to_time_grid,
    get_tip_component,
)


@dataclass
class TrainKappaYGlobalConfig:
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

    case_dir: str = str(PROJECT_ROOT / "results" / "student" / "kappa_y_global_training_cases")
    output_dir: str = str(PROJECT_ROOT / "results" / "student" / "train_kappa_y_global_torch")
    case_prefix: str = "train_kappa_y_global"

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

    scale_mode: str = "y_bending"  # choices: uy_only, y_bending
    scale_min: float = 0.90
    scale_max: float = 1.02
    init_scale: float = 0.96

    epochs: int = 80
    lr: float = 5.0e-2
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0

    w_full_y: float = 1.0
    w_tip_y: float = 1.0
    w_lastk_y: float = 1.0

    w_full_x_guard: float = 2.0
    w_tip_x_guard: float = 2.0
    x_guard_tol: float = 0.02

    w_scale_prior: float = 0.01
    prior_scale: float = 0.96

    remove_initial_offset: bool = True
    last_k: int = 5

    torch_dtype: str = "float64"
    device: str = "cpu"

    use_valid_for_best: bool = True
    early_stop_patience: int = 30
    early_stop_min_delta: float = 1.0e-6
    min_epochs: int = 10

    print_every: int = 1
    save_report: bool = True


@dataclass
class TrainingCase:
    name: str
    split: str
    load_file: str
    time: np.ndarray
    F_time: np.ndarray
    u_teacher: np.ndarray
    u_base: np.ndarray
    v_base: np.ndarray
    base_metrics: Dict[str, Dict[str, float]]


def parse_args() -> TrainKappaYGlobalConfig:
    d = TrainKappaYGlobalConfig()

    parser = argparse.ArgumentParser(
        description="Train one global static kappa_y_scale parameter for y-bending stiffness correction."
    )

    parser.add_argument("--teacher-exe", type=str, default=d.teacher_exe)
    parser.add_argument("--template-inp", type=str, default=d.template_inp)
    parser.add_argument("--blade-csv", type=str, default=d.blade_csv)

    parser.add_argument("--case-dir", type=str, default=d.case_dir)
    parser.add_argument("--output-dir", type=str, default=d.output_dir)
    parser.add_argument("--case-prefix", type=str, default=d.case_prefix)

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

    parser.add_argument("--scale-mode", type=str, default=d.scale_mode, choices=["uy_only", "y_bending"])
    parser.add_argument("--scale-min", type=float, default=d.scale_min)
    parser.add_argument("--scale-max", type=float, default=d.scale_max)
    parser.add_argument("--init-scale", type=float, default=d.init_scale)

    parser.add_argument("--epochs", type=int, default=d.epochs)
    parser.add_argument("--lr", type=float, default=d.lr)
    parser.add_argument("--weight-decay", type=float, default=d.weight_decay)
    parser.add_argument("--grad-clip-norm", type=float, default=d.grad_clip_norm)

    parser.add_argument("--w-full-y", type=float, default=d.w_full_y)
    parser.add_argument("--w-tip-y", type=float, default=d.w_tip_y)
    parser.add_argument("--w-lastk-y", type=float, default=d.w_lastk_y)

    parser.add_argument("--w-full-x-guard", type=float, default=d.w_full_x_guard)
    parser.add_argument("--w-tip-x-guard", type=float, default=d.w_tip_x_guard)
    parser.add_argument("--x-guard-tol", type=float, default=d.x_guard_tol)

    parser.add_argument("--w-scale-prior", type=float, default=d.w_scale_prior)
    parser.add_argument("--prior-scale", type=float, default=d.prior_scale)

    offset_group = parser.add_mutually_exclusive_group()
    offset_group.add_argument("--remove-initial-offset", dest="remove_initial_offset", action="store_true")
    offset_group.add_argument("--keep-initial-offset", dest="remove_initial_offset", action="store_false")
    parser.set_defaults(remove_initial_offset=d.remove_initial_offset)

    parser.add_argument("--last-k", type=int, default=d.last_k)

    parser.add_argument("--torch-dtype", type=str, default=d.torch_dtype, choices=["float64", "float32"])
    parser.add_argument("--device", type=str, default=d.device)

    valid_group = parser.add_mutually_exclusive_group()
    valid_group.add_argument("--use-valid-for-best", dest="use_valid_for_best", action="store_true")
    valid_group.add_argument("--use-train-for-best", dest="use_valid_for_best", action="store_false")
    parser.set_defaults(use_valid_for_best=d.use_valid_for_best)

    parser.add_argument("--early-stop-patience", type=int, default=d.early_stop_patience)
    parser.add_argument("--early-stop-min-delta", type=float, default=d.early_stop_min_delta)
    parser.add_argument("--min-epochs", type=int, default=d.min_epochs)

    parser.add_argument("--print-every", type=int, default=d.print_every)

    report_group = parser.add_mutually_exclusive_group()
    report_group.add_argument("--save-report", dest="save_report", action="store_true")
    report_group.add_argument("--no-save-report", dest="save_report", action="store_false")
    parser.set_defaults(save_report=d.save_report)

    args = parser.parse_args()

    return TrainKappaYGlobalConfig(
        teacher_exe=args.teacher_exe,
        template_inp=args.template_inp,
        blade_csv=args.blade_csv,
        case_dir=args.case_dir,
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
        base_phi_twist_column=args.base_phi_twist_column,
        base_phi_sign=args.base_phi_sign,
        rotate_mass=args.rotate_mass,
        scale_mode=args.scale_mode,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        init_scale=args.init_scale,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        w_full_y=args.w_full_y,
        w_tip_y=args.w_tip_y,
        w_lastk_y=args.w_lastk_y,
        w_full_x_guard=args.w_full_x_guard,
        w_tip_x_guard=args.w_tip_x_guard,
        x_guard_tol=args.x_guard_tol,
        w_scale_prior=args.w_scale_prior,
        prior_scale=args.prior_scale,
        remove_initial_offset=args.remove_initial_offset,
        last_k=args.last_k,
        torch_dtype=args.torch_dtype,
        device=args.device,
        use_valid_for_best=args.use_valid_for_best,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        min_epochs=args.min_epochs,
        print_every=args.print_every,
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
    raise ValueError(f"Unsupported dtype: {name}")


def remove_initial_offset_np(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    return u - u[:1, :]


def remove_initial_offset_torch(u: torch.Tensor) -> torch.Tensor:
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
        values = [float(row[column_name]) for row in reader]
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
            f"expected {n_elements} or {n_elements + 1}."
        )

    return float(sign) * element_twist


def compute_natural_frequencies_hz(M: np.ndarray, K: np.ndarray, *, num_modes: int = 10) -> np.ndarray:
    eigvals, _ = eigh(K, M)
    eigvals = np.asarray(eigvals, dtype=np.float64)
    eigvals = eigvals[eigvals > 0.0]
    return np.sqrt(eigvals)[:num_modes] / (2.0 * np.pi)


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


def selected_y_dof_indices(n_dofs: int, scale_mode: str) -> np.ndarray:
    if n_dofs % 6 != 0:
        raise ValueError(f"n_dofs must be divisible by 6, got {n_dofs}")

    n_nodes = n_dofs // 6
    uy = np.array([i * 6 + 1 for i in range(n_nodes)], dtype=int)

    if scale_mode == "uy_only":
        return uy

    if scale_mode == "y_bending":
        rx = np.array([i * 6 + 3 for i in range(n_nodes)], dtype=int)
        return np.unique(np.concatenate([uy, rx]))

    raise ValueError(f"Unsupported scale_mode: {scale_mode}")


def get_component_np(u_full: np.ndarray, component: str) -> np.ndarray:
    n_dofs = u_full.shape[1]
    n_nodes = n_dofs // 6
    idx = component_indices(n_nodes, component)
    return u_full[:, idx]


def get_component_torch(u_full: torch.Tensor, component: str) -> torch.Tensor:
    n_dofs = u_full.shape[1]
    n_nodes = n_dofs // 6
    idx = component_indices(n_nodes, component)
    idx_t = torch.as_tensor(idx, dtype=torch.long, device=u_full.device)
    return u_full.index_select(dim=1, index=idx_t)


def get_tip_component_torch(u_full: torch.Tensor, component: str) -> torch.Tensor:
    comp = get_component_torch(u_full, component)
    return comp[:, -1]


def get_lastk_component_mean_torch(u_full: torch.Tensor, component: str, *, last_k: int) -> torch.Tensor:
    comp = get_component_torch(u_full, component)
    return comp[:, -last_k:].mean(dim=1)


def compute_mse_np(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return float(np.mean((pred - target) ** 2))


def compute_case_base_metrics(
    *,
    u_base: np.ndarray,
    u_teacher: np.ndarray,
    last_k: int,
) -> Dict[str, Dict[str, float]]:
    base_full_x = compute_mse_np(get_component_np(u_base, "x"), get_component_np(u_teacher, "x"))
    base_full_y = compute_mse_np(get_component_np(u_base, "y"), get_component_np(u_teacher, "y"))

    base_tip_x = compute_mse_np(get_tip_component_np(u_base, "x"), get_tip_component_np(u_teacher, "x"))
    base_tip_y = compute_mse_np(get_tip_component_np(u_base, "y"), get_tip_component_np(u_teacher, "y"))

    base_lastk_y = compute_mse_np(
        get_lastk_component_mean_np(u_base, "y", last_k=last_k),
        get_lastk_component_mean_np(u_teacher, "y", last_k=last_k),
    )

    return {
        "full_x": {"mse": base_full_x},
        "full_y": {"mse": base_full_y},
        "tip_x": {"mse": base_tip_x},
        "tip_y": {"mse": base_tip_y},
        f"last{last_k}_mean_y": {"mse": base_lastk_y},
    }


def get_tip_component_np(u_full: np.ndarray, component: str) -> np.ndarray:
    return get_component_np(u_full, component)[:, -1]


def get_lastk_component_mean_np(u_full: np.ndarray, component: str, *, last_k: int) -> np.ndarray:
    comp = get_component_np(u_full, component)
    return comp[:, -last_k:].mean(axis=1)


def safe_denominator(value: float) -> float:
    return float(value) + 1.0e-18


def prepare_one_case(
    *,
    cfg: TrainKappaYGlobalConfig,
    teacher_exe: Path,
    template_inp: Path,
    blade_csv: Path,
    load_file: Path,
    split: str,
    index: int,
) -> TrainingCase:
    stem = load_file.stem
    case_name = f"{cfg.case_prefix}_{split}_{index:02d}_{stem}"
    case_dir = Path(cfg.case_dir).resolve() / split / stem
    case_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 100)
    print(f"[Prepare Case] {case_name}")
    print("=" * 100)
    print(f"  split     = {split}")
    print(f"  load_file = {load_file}")
    print(f"  case_dir  = {case_dir}")

    print()
    print("[1/4] Running BeamDyn teacher")
    teacher_case_cfg = BeamDynTeacherCaseConfig(
        case_name=case_name,
        teacher_exe=teacher_exe,
        template_inp=template_inp,
        output_dir=case_dir / "teacher",
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
        raise RuntimeError("Teacher did not return .out path.")
    print(f"  teacher_out = {teacher_out}")

    print()
    print("[2/4] Loading teacher response")
    time_teacher, u_teacher_raw, _ = load_teacher_6dof_response(
        out_path=teacher_out,
        node_start=cfg.teacher_node_start,
        node_end=cfg.teacher_node_end,
        demean=cfg.teacher_demean,
    )
    print(f"  time_teacher shape = {time_teacher.shape}")
    print(f"  u_teacher shape    = {u_teacher_raw.shape}")

    print()
    print("[3/4] Running base student")
    student_result = run_student_case(
        blade_csv=blade_csv,
        output_dir=case_dir / "student_base",
        case_name=case_name,
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

    time_student = np.asarray(student_result["time"], dtype=np.float64)
    F_time = np.asarray(student_result["F_time"], dtype=np.float64)
    u_base_raw = np.asarray(student_result["u_full"], dtype=np.float64)
    v_base = np.asarray(student_result["v_full"], dtype=np.float64)

    print(f"  time_student shape = {time_student.shape}")
    print(f"  F_time shape       = {F_time.shape}")
    print(f"  u_base shape       = {u_base_raw.shape}")

    print()
    print("[4/4] Resampling teacher to student grid and computing base metrics")
    u_teacher_rs = resample_response_to_time_grid(
        time_src=time_teacher,
        u_src=u_teacher_raw,
        time_dst=time_student,
    )

    if cfg.remove_initial_offset:
        u_teacher = remove_initial_offset_np(u_teacher_rs)
        u_base = remove_initial_offset_np(u_base_raw)
    else:
        u_teacher = u_teacher_rs
        u_base = u_base_raw

    base_metrics = compute_case_base_metrics(
        u_base=u_base,
        u_teacher=u_teacher,
        last_k=cfg.last_k,
    )

    print("[Base Metrics]")
    for k, v in base_metrics.items():
        print(f"  {k:<16s} mse={v['mse']:.8e}")

    return TrainingCase(
        name=case_name,
        split=split,
        load_file=str(load_file),
        time=time_student,
        F_time=F_time,
        u_teacher=u_teacher,
        u_base=u_base,
        v_base=v_base,
        base_metrics=base_metrics,
    )


def prepare_cases(
    *,
    cfg: TrainKappaYGlobalConfig,
    teacher_exe: Path,
    template_inp: Path,
    blade_csv: Path,
) -> Tuple[List[TrainingCase], List[TrainingCase]]:
    train_cases = []
    valid_cases = []

    for i, file in enumerate(TRAIN_LOAD_FILES):
        load_file = assert_existing_file(file, f"TRAIN_LOAD_FILES[{i}]")
        train_cases.append(
            prepare_one_case(
                cfg=cfg,
                teacher_exe=teacher_exe,
                template_inp=template_inp,
                blade_csv=blade_csv,
                load_file=load_file,
                split="train",
                index=i,
            )
        )

    for i, file in enumerate(VALID_LOAD_FILES):
        load_file = assert_existing_file(file, f"VALID_LOAD_FILES[{i}]")
        valid_cases.append(
            prepare_one_case(
                cfg=cfg,
                teacher_exe=teacher_exe,
                template_inp=template_inp,
                blade_csv=blade_csv,
                load_file=load_file,
                split="valid",
                index=i,
            )
        )

    return train_cases, valid_cases


def build_base_mk(
    *,
    cfg: TrainKappaYGlobalConfig,
    blade_csv: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model = load_student_model_from_blade_master(
        csv_path=str(blade_csv),
        model_name="train_kappa_y_global_model",
    )

    n_elements = int(model.n_stations - 1)
    phi_element_deg = build_element_phi_from_initial_twist(
        blade_csv=blade_csv,
        twist_column=cfg.base_phi_twist_column,
        sign=cfg.base_phi_sign,
        n_elements=n_elements,
    )

    print()
    print("[Build Base M/K]")
    print(f"  Phi_base = {cfg.base_phi_sign:+.1f} * {cfg.base_phi_twist_column}(s)")
    print(
        "  phi_element_deg: "
        f"min={phi_element_deg.min():.6f}, "
        f"max={phi_element_deg.max():.6f}, "
        f"mean={phi_element_deg.mean():.6f}"
    )

    M, K, _ = build_coupled_fem_matrices_6dof_degrees(
        model,
        section_params=None,
        phi_deg=phi_element_deg,
        alpha_flap=cfg.alpha_flap,
        alpha_edge=cfg.alpha_edge,
        alpha_torsion=cfg.alpha_torsion,
        rotate_mass=cfg.rotate_mass,
        return_full=True,
    )

    M = np.asarray(M, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)

    natural_freqs = compute_natural_frequencies_hz(M, K, num_modes=10)
    C_base, ref_freq_used = _build_structural_damping_matrix(
        K=K,
        zeta_structural=cfg.zeta_structural,
        ref_freq_hz=cfg.ref_freq_hz,
        natural_freqs=natural_freqs,
    )

    print(f"  M shape = {M.shape}")
    print(f"  K shape = {K.shape}")
    print(f"  natural_freqs[:5] = {natural_freqs[:5]}")
    print(f"  ref_freq_used = {ref_freq_used}")

    if cfg.zeta_structural > 0:
        if ref_freq_used is None:
            raise RuntimeError("ref_freq_used is None although zeta_structural > 0.")
        beta_damp = 2.0 * cfg.zeta_structural / (2.0 * np.pi * float(ref_freq_used))
    else:
        beta_damp = 0.0

    print(f"  fixed damping beta = {beta_damp:.12e}")
    print("  note: during training C(kappa) = beta_damp * K(kappa)")

    return M, K, C_base, np.array([beta_damp], dtype=np.float64)


def init_raw_from_scale(scale: float, scale_min: float, scale_max: float) -> float:
    if not (scale_min < scale < scale_max):
        raise ValueError(
            f"init_scale must satisfy scale_min < init_scale < scale_max, got "
            f"{scale_min}, {scale}, {scale_max}"
        )
    p = (scale - scale_min) / (scale_max - scale_min)
    return float(math.log(p / (1.0 - p)))


def scale_from_raw(
    raw: torch.Tensor,
    *,
    scale_min: float,
    scale_max: float,
) -> torch.Tensor:
    return scale_min + (scale_max - scale_min) * torch.sigmoid(raw)


def build_scaled_k_torch(
    *,
    K_base: torch.Tensor,
    scale: torch.Tensor,
    scaled_dof_idx: torch.Tensor,
) -> torch.Tensor:
    n = K_base.shape[0]
    s = torch.ones(n, dtype=K_base.dtype, device=K_base.device)
    s[scaled_dof_idx] = torch.sqrt(scale)
    K_scaled = (s[:, None] * K_base) * s[None, :]
    return 0.5 * (K_scaled + K_scaled.transpose(0, 1))


def newmark_rollout_torch(
    *,
    M: torch.Tensor,
    K: torch.Tensor,
    C: torch.Tensor,
    F_time: torch.Tensor,
    u0: torch.Tensor,
    v0: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    gamma = 0.5
    beta = 0.25

    dt_t = torch.as_tensor(dt, dtype=M.dtype, device=M.device)

    A = M + gamma * dt_t * C + beta * dt_t * dt_t * K

    a = torch.linalg.solve(M, F_time[0] - C @ v0 - K @ u0)
    u = u0
    v = v0

    states = [u]

    for i in range(1, F_time.shape[0]):
        u_pred = u + dt_t * v + dt_t * dt_t * (0.5 - beta) * a
        v_pred = v + dt_t * (1.0 - gamma) * a

        rhs = F_time[i] - C @ v_pred - K @ u_pred
        a_new = torch.linalg.solve(A, rhs)

        u_new = u_pred + beta * dt_t * dt_t * a_new
        v_new = v_pred + gamma * dt_t * a_new

        u = u_new
        v = v_new
        a = a_new

        states.append(u)

    return torch.stack(states, dim=0)


def compute_case_loss(
    *,
    case: TrainingCase,
    M_t: torch.Tensor,
    K_base_t: torch.Tensor,
    beta_damp_t: torch.Tensor,
    raw_scale: torch.Tensor,
    scaled_dof_idx_t: torch.Tensor,
    cfg: TrainKappaYGlobalConfig,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    scale = scale_from_raw(
        raw_scale,
        scale_min=cfg.scale_min,
        scale_max=cfg.scale_max,
    )

    K_scaled = build_scaled_k_torch(
        K_base=K_base_t,
        scale=scale,
        scaled_dof_idx=scaled_dof_idx_t,
    )
    C_scaled = beta_damp_t.reshape(()) * K_scaled

    F_time = torch.as_tensor(case.F_time, dtype=dtype, device=device)
    u0 = torch.as_tensor(case.u_base[0], dtype=dtype, device=device)
    v0 = torch.as_tensor(case.v_base[0], dtype=dtype, device=device)
    target = torch.as_tensor(case.u_teacher, dtype=dtype, device=device)

    pred_raw = newmark_rollout_torch(
        M=M_t,
        K=K_scaled,
        C=C_scaled,
        F_time=F_time,
        u0=u0,
        v0=v0,
        dt=cfg.dt,
    )

    if cfg.remove_initial_offset:
        pred = remove_initial_offset_torch(pred_raw)
    else:
        pred = pred_raw

    pred_full_y = get_component_torch(pred, "y")
    target_full_y = get_component_torch(target, "y")
    pred_tip_y = get_tip_component_torch(pred, "y")
    target_tip_y = get_tip_component_torch(target, "y")
    pred_lastk_y = get_lastk_component_mean_torch(pred, "y", last_k=cfg.last_k)
    target_lastk_y = get_lastk_component_mean_torch(target, "y", last_k=cfg.last_k)

    pred_full_x = get_component_torch(pred, "x")
    target_full_x = get_component_torch(target, "x")
    pred_tip_x = get_tip_component_torch(pred, "x")
    target_tip_x = get_tip_component_torch(target, "x")

    mse_full_y = torch.mean((pred_full_y - target_full_y) ** 2)
    mse_tip_y = torch.mean((pred_tip_y - target_tip_y) ** 2)
    mse_lastk_y = torch.mean((pred_lastk_y - target_lastk_y) ** 2)
    mse_full_x = torch.mean((pred_full_x - target_full_x) ** 2)
    mse_tip_x = torch.mean((pred_tip_x - target_tip_x) ** 2)

    base_full_y = safe_denominator(case.base_metrics["full_y"]["mse"])
    base_tip_y = safe_denominator(case.base_metrics["tip_y"]["mse"])
    base_lastk_y = safe_denominator(case.base_metrics[f"last{cfg.last_k}_mean_y"]["mse"])
    base_full_x = safe_denominator(case.base_metrics["full_x"]["mse"])
    base_tip_x = safe_denominator(case.base_metrics["tip_x"]["mse"])

    ratio_full_y = mse_full_y / base_full_y
    ratio_tip_y = mse_tip_y / base_tip_y
    ratio_lastk_y = mse_lastk_y / base_lastk_y
    ratio_full_x = mse_full_x / base_full_x
    ratio_tip_x = mse_tip_x / base_tip_x

    x_guard_limit = 1.0 + cfg.x_guard_tol
    full_x_guard = torch.relu(ratio_full_x - x_guard_limit) ** 2
    tip_x_guard = torch.relu(ratio_tip_x - x_guard_limit) ** 2

    y_loss = (
        cfg.w_full_y * ratio_full_y
        + cfg.w_tip_y * ratio_tip_y
        + cfg.w_lastk_y * ratio_lastk_y
    ) / max(cfg.w_full_y + cfg.w_tip_y + cfg.w_lastk_y, 1.0e-12)

    x_guard = (
        cfg.w_full_x_guard * full_x_guard
        + cfg.w_tip_x_guard * tip_x_guard
    )

    scale_prior = (scale - cfg.prior_scale) ** 2

    total = y_loss + x_guard + cfg.w_scale_prior * scale_prior

    stats = {
        "scale": float(scale.detach().cpu().item()),
        "y_loss": float(y_loss.detach().cpu().item()),
        "full_y_ratio": float(ratio_full_y.detach().cpu().item()),
        "tip_y_ratio": float(ratio_tip_y.detach().cpu().item()),
        f"last{cfg.last_k}_y_ratio": float(ratio_lastk_y.detach().cpu().item()),
        "full_x_ratio": float(ratio_full_x.detach().cpu().item()),
        "tip_x_ratio": float(ratio_tip_x.detach().cpu().item()),
        "x_guard": float(x_guard.detach().cpu().item()),
        "scale_prior": float(scale_prior.detach().cpu().item()),
        "total": float(total.detach().cpu().item()),
    }

    return total, stats


def evaluate_cases(
    *,
    cases: List[TrainingCase],
    M_t: torch.Tensor,
    K_base_t: torch.Tensor,
    beta_damp_t: torch.Tensor,
    raw_scale: torch.Tensor,
    scaled_dof_idx_t: torch.Tensor,
    cfg: TrainKappaYGlobalConfig,
    dtype: torch.dtype,
    device: torch.device,
) -> Dict[str, float]:
    if len(cases) == 0:
        return {
            "total": float("nan"),
            "scale": float(scale_from_raw(raw_scale, scale_min=cfg.scale_min, scale_max=cfg.scale_max).detach().cpu().item()),
            "y_loss": float("nan"),
            "full_y_ratio": float("nan"),
            "tip_y_ratio": float("nan"),
            f"last{cfg.last_k}_y_ratio": float("nan"),
            "full_x_ratio": float("nan"),
            "tip_x_ratio": float("nan"),
            "x_guard": float("nan"),
        }

    totals = []
    aggregate: Dict[str, List[float]] = {}

    with torch.no_grad():
        for case in cases:
            loss, stats = compute_case_loss(
                case=case,
                M_t=M_t,
                K_base_t=K_base_t,
                beta_damp_t=beta_damp_t,
                raw_scale=raw_scale,
                scaled_dof_idx_t=scaled_dof_idx_t,
                cfg=cfg,
                dtype=dtype,
                device=device,
            )
            totals.append(float(loss.detach().cpu().item()))
            for k, v in stats.items():
                aggregate.setdefault(k, []).append(float(v))

    result = {k: float(np.mean(v)) for k, v in aggregate.items()}
    result["total"] = float(np.mean(totals))
    return result


def save_history_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    cfg = parse_args()

    teacher_exe = assert_existing_file(cfg.teacher_exe, "teacher_exe")
    template_inp = assert_existing_file(cfg.template_inp, "template_inp")
    blade_csv = assert_existing_file(cfg.blade_csv, "blade_csv")

    output_dir = Path(cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = get_torch_dtype(cfg.torch_dtype)
    device = torch.device(cfg.device)

    print()
    print("[Train Kappa-y Global Torch]")
    print()
    print("[Config]")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")

    print()
    print("[Load Files]")
    print("  train:")
    for f in TRAIN_LOAD_FILES:
        print(f"    - {f}")
    print("  valid:")
    if VALID_LOAD_FILES:
        for f in VALID_LOAD_FILES:
            print(f"    - {f}")
    else:
        print("    - <empty>")

    print()
    print("[1/5] Preparing train/valid cases")
    train_cases, valid_cases = prepare_cases(
        cfg=cfg,
        teacher_exe=teacher_exe,
        template_inp=template_inp,
        blade_csv=blade_csv,
    )

    if len(train_cases) == 0:
        raise RuntimeError("No train cases provided.")

    if len(valid_cases) == 0:
        print()
        print("[Warning] No validation cases. Best model will be selected by train score.")

    print()
    print("[2/5] Building base M/K")
    M_base, K_base, _, beta_damp = build_base_mk(
        cfg=cfg,
        blade_csv=blade_csv,
    )

    scaled_dof_idx = selected_y_dof_indices(K_base.shape[0], cfg.scale_mode)
    print()
    print("[Scaled DOFs]")
    print(f"  scale_mode = {cfg.scale_mode}")
    print(f"  n_scaled_dofs = {scaled_dof_idx.size}")

    M_t = torch.as_tensor(M_base, dtype=dtype, device=device)
    K_base_t = torch.as_tensor(K_base, dtype=dtype, device=device)
    beta_damp_t = torch.as_tensor(beta_damp, dtype=dtype, device=device)
    scaled_dof_idx_t = torch.as_tensor(scaled_dof_idx, dtype=torch.long, device=device)

    print()
    print("[3/5] Initializing trainable global kappa_y_scale")
    raw_init = init_raw_from_scale(cfg.init_scale, cfg.scale_min, cfg.scale_max)
    raw_scale = torch.nn.Parameter(
        torch.as_tensor(raw_init, dtype=dtype, device=device)
    )

    print(f"  scale range = [{cfg.scale_min:.6f}, {cfg.scale_max:.6f}]")
    print(f"  init_scale  = {cfg.init_scale:.6f}")
    print(f"  raw_init    = {raw_init:.8e}")

    optimizer = torch.optim.Adam(
        [raw_scale],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    print()
    print("[4/5] Training")
    print("  active parameter = global kappa_y_scale")
    print("  no f_static is trained in this script")

    best_score = float("inf")
    best_epoch = 0
    best_raw_scale = None
    no_improve_epochs = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, cfg.epochs + 1):
        optimizer.zero_grad(set_to_none=True)

        train_losses = []
        for case in train_cases:
            loss, _ = compute_case_loss(
                case=case,
                M_t=M_t,
                K_base_t=K_base_t,
                beta_damp_t=beta_damp_t,
                raw_scale=raw_scale,
                scaled_dof_idx_t=scaled_dof_idx_t,
                cfg=cfg,
                dtype=dtype,
                device=device,
            )
            train_losses.append(loss)

        train_loss = torch.stack(train_losses).mean()
        train_loss.backward()

        if cfg.grad_clip_norm is not None and cfg.grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_([raw_scale], max_norm=cfg.grad_clip_norm)

        optimizer.step()

        train_stats = evaluate_cases(
            cases=train_cases,
            M_t=M_t,
            K_base_t=K_base_t,
            beta_damp_t=beta_damp_t,
            raw_scale=raw_scale,
            scaled_dof_idx_t=scaled_dof_idx_t,
            cfg=cfg,
            dtype=dtype,
            device=device,
        )

        valid_stats = evaluate_cases(
            cases=valid_cases,
            M_t=M_t,
            K_base_t=K_base_t,
            beta_damp_t=beta_damp_t,
            raw_scale=raw_scale,
            scaled_dof_idx_t=scaled_dof_idx_t,
            cfg=cfg,
            dtype=dtype,
            device=device,
        )

        if len(valid_cases) > 0 and cfg.use_valid_for_best:
            score = valid_stats["total"]
        else:
            score = train_stats["total"]

        improved = score < best_score - cfg.early_stop_min_delta
        if improved:
            best_score = score
            best_epoch = epoch
            best_raw_scale = raw_scale.detach().cpu().clone()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        row = {
            "epoch": epoch,
            "scale": train_stats["scale"],
            "train_total": train_stats["total"],
            "train_y_loss": train_stats["y_loss"],
            "train_full_y_ratio": train_stats["full_y_ratio"],
            "train_tip_y_ratio": train_stats["tip_y_ratio"],
            f"train_last{cfg.last_k}_y_ratio": train_stats[f"last{cfg.last_k}_y_ratio"],
            "train_full_x_ratio": train_stats["full_x_ratio"],
            "train_tip_x_ratio": train_stats["tip_x_ratio"],
            "train_x_guard": train_stats["x_guard"],
            "valid_total": valid_stats["total"],
            "valid_y_loss": valid_stats["y_loss"],
            "valid_full_y_ratio": valid_stats["full_y_ratio"],
            "valid_tip_y_ratio": valid_stats["tip_y_ratio"],
            f"valid_last{cfg.last_k}_y_ratio": valid_stats[f"last{cfg.last_k}_y_ratio"],
            "valid_full_x_ratio": valid_stats["full_x_ratio"],
            "valid_tip_x_ratio": valid_stats["tip_x_ratio"],
            "valid_x_guard": valid_stats["x_guard"],
            "best_score": best_score,
            "is_best": bool(improved),
        }
        history.append(row)

        if epoch % cfg.print_every == 0 or epoch == 1 or improved:
            best_mark = " BEST" if improved else ""
            print(
                f"[Epoch {epoch:04d}] "
                f"scale={train_stats['scale']:.8f} "
                f"train_total={train_stats['total']:.8e} "
                f"train_y={train_stats['y_loss']:.6f} "
                f"train_full_y={train_stats['full_y_ratio']:.6f} "
                f"train_tip_y={train_stats['tip_y_ratio']:.6f} "
                f"train_last{cfg.last_k}_y={train_stats[f'last{cfg.last_k}_y_ratio']:.6f} "
                f"train_full_x={train_stats['full_x_ratio']:.6f} "
                f"train_tip_x={train_stats['tip_x_ratio']:.6f} "
                f"valid_total={valid_stats['total']:.8e} "
                f"valid_y={valid_stats['y_loss']:.6f} "
                f"valid_full_x={valid_stats['full_x_ratio']:.6f} "
                f"valid_tip_x={valid_stats['tip_x_ratio']:.6f}"
                f"{best_mark}"
            )

        if (
            epoch >= cfg.min_epochs
            and no_improve_epochs >= cfg.early_stop_patience
        ):
            print()
            print("[Early Stop]")
            print(f"  epoch = {epoch}")
            print(f"  best_epoch = {best_epoch}")
            print(f"  best_score = {best_score:.8e}")
            print(f"  no_improve_epochs = {no_improve_epochs}")
            break

    if best_raw_scale is None:
        best_raw_scale = raw_scale.detach().cpu().clone()

    print()
    print("[5/5] Saving results")

    best_scale = scale_from_raw(
        best_raw_scale.to(dtype=dtype, device=device),
        scale_min=cfg.scale_min,
        scale_max=cfg.scale_max,
    ).detach().cpu().item()

    final_scale = scale_from_raw(
        raw_scale.detach(),
        scale_min=cfg.scale_min,
        scale_max=cfg.scale_max,
    ).detach().cpu().item()

    best_params_path = output_dir / "best_kappa_y_global_params.npz"
    final_params_path = output_dir / "final_kappa_y_global_params.npz"
    history_path = output_dir / "training_history.csv"
    report_path = output_dir / "train_kappa_y_global_report.json"

    np.savez(
        best_params_path,
        kappa_y_scale=np.array([best_scale], dtype=np.float64),
        raw_scale=np.array([float(best_raw_scale.item())], dtype=np.float64),
        scale_min=np.array([cfg.scale_min], dtype=np.float64),
        scale_max=np.array([cfg.scale_max], dtype=np.float64),
        best_epoch=np.array([best_epoch], dtype=np.int64),
        best_score=np.array([best_score], dtype=np.float64),
        scale_mode=np.array([cfg.scale_mode]),
    )

    np.savez(
        final_params_path,
        kappa_y_scale=np.array([final_scale], dtype=np.float64),
        raw_scale=np.array([float(raw_scale.detach().cpu().item())], dtype=np.float64),
        scale_min=np.array([cfg.scale_min], dtype=np.float64),
        scale_max=np.array([cfg.scale_max], dtype=np.float64),
        scale_mode=np.array([cfg.scale_mode]),
    )

    save_history_csv(history_path, history)

    final_train_stats = evaluate_cases(
        cases=train_cases,
        M_t=M_t,
        K_base_t=K_base_t,
        beta_damp_t=beta_damp_t,
        raw_scale=raw_scale,
        scaled_dof_idx_t=scaled_dof_idx_t,
        cfg=cfg,
        dtype=dtype,
        device=device,
    )
    final_valid_stats = evaluate_cases(
        cases=valid_cases,
        M_t=M_t,
        K_base_t=K_base_t,
        beta_damp_t=beta_damp_t,
        raw_scale=raw_scale,
        scaled_dof_idx_t=scaled_dof_idx_t,
        cfg=cfg,
        dtype=dtype,
        device=device,
    )

    report = {
        "config": asdict(cfg),
        "train_load_files": TRAIN_LOAD_FILES,
        "valid_load_files": VALID_LOAD_FILES,
        "best": {
            "best_epoch": int(best_epoch),
            "best_score": float(best_score),
            "best_kappa_y_scale": float(best_scale),
        },
        "final": {
            "final_kappa_y_scale": float(final_scale),
            "train_stats": final_train_stats,
            "valid_stats": final_valid_stats,
        },
        "saved_files": {
            "best_params": str(best_params_path),
            "final_params": str(final_params_path),
            "history": str(history_path),
            "report": str(report_path),
        },
        "note": (
            "This script trains only one static global kappa_y_scale parameter. "
            "No f_static/load mapping parameter is trained here."
        ),
    }

    if cfg.save_report:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"  best params  = {best_params_path}")
    print(f"  final params = {final_params_path}")
    print(f"  history      = {history_path}")
    print(f"  report       = {report_path}")

    print()
    print("[Best Result]")
    print(f"  best_epoch = {best_epoch}")
    print(f"  best_score = {best_score:.8e}")
    print(f"  best_kappa_y_scale = {best_scale:.8f}")

    print()
    print("[Final Result]")
    print(f"  final_kappa_y_scale = {final_scale:.8f}")
    print(f"  train_y_loss = {final_train_stats['y_loss']:.8f}")
    if len(valid_cases) > 0:
        print(f"  valid_y_loss = {final_valid_stats['y_loss']:.8f}")

    print()
    print("✅ PASS: global kappa_y_scale training completed.")
    print("   当前脚本只训练 κ：y-bending 全局静态刚度缩放；不包含 f_static。")


if __name__ == "__main__":
    main()