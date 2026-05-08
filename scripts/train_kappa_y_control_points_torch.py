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
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# 你后续只需要改这里的载荷文件列表
TRAIN_LOAD_FILES = [
    str(PROJECT_ROOT / "data" / "load" / "simple_tip_fy_case.dat"),
]

VALID_LOAD_FILES = [
    # 建议后续加入独立 y-only 或 mixed load 验证工况
    str(PROJECT_ROOT / "data" / "load" / "test_complex_case.dat"),
]


from scripts.train_kappa_y_global_torch import (  # noqa: E402
    TrainingCase,
    assert_existing_file,
    get_torch_dtype,
    prepare_one_case,
    build_base_mk,
    get_component_torch,
    get_tip_component_torch,
    get_lastk_component_mean_torch,
    remove_initial_offset_torch,
    safe_denominator,
    save_history_csv,
)


@dataclass
class TrainKappaYControlPointsConfig:
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

    case_dir: str = str(PROJECT_ROOT / "results" / "student" / "kappa_y_cp_training_cases")
    output_dir: str = str(PROJECT_ROOT / "results" / "student" / "train_kappa_y_cp_torch")
    case_prefix: str = "train_kappa_y_cp"

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

    n_control_points: int = 6
    scale_mode: str = "y_bending"  # choices: uy_only, y_bending
    scale_min: float = 0.88
    scale_max: float = 1.05
    init_scale: float = 0.952

    epochs: int = 100
    lr: float = 1.0e-2
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0

    w_full_y: float = 1.0
    w_tip_y: float = 1.0
    w_lastk_y: float = 1.0

    w_full_x_guard: float = 2.0
    w_tip_x_guard: float = 2.0
    x_guard_tol: float = 0.02

    w_scale_prior: float = 0.01
    prior_scale: float = 0.952
    w_smooth: float = 0.05
    w_curvature: float = 0.05

    remove_initial_offset: bool = True
    last_k: int = 5

    torch_dtype: str = "float64"
    device: str = "cpu"

    use_valid_for_best: bool = True
    early_stop_patience: int = 40
    early_stop_min_delta: float = 1.0e-6
    min_epochs: int = 20

    print_every: int = 1
    save_report: bool = True


def parse_args() -> TrainKappaYControlPointsConfig:
    d = TrainKappaYControlPointsConfig()

    parser = argparse.ArgumentParser(
        description="Train 6-control-point static kappa_y_scale(s) for y-bending stiffness correction."
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

    parser.add_argument("--n-control-points", type=int, default=d.n_control_points)
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
    parser.add_argument("--w-smooth", type=float, default=d.w_smooth)
    parser.add_argument("--w-curvature", type=float, default=d.w_curvature)

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

    return TrainKappaYControlPointsConfig(
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
        n_control_points=args.n_control_points,
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
        w_smooth=args.w_smooth,
        w_curvature=args.w_curvature,
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


def init_raw_from_scale(scale: float, scale_min: float, scale_max: float) -> float:
    if not (scale_min < scale < scale_max):
        raise ValueError(
            f"init_scale must satisfy scale_min < init_scale < scale_max, got "
            f"{scale_min}, {scale}, {scale_max}"
        )
    p = (scale - scale_min) / (scale_max - scale_min)
    return float(math.log(p / (1.0 - p)))


def control_scales_from_raw(
    raw_cp: torch.Tensor,
    *,
    scale_min: float,
    scale_max: float,
) -> torch.Tensor:
    return scale_min + (scale_max - scale_min) * torch.sigmoid(raw_cp)


def interpolate_control_points_to_nodes(
    cp_scale: torch.Tensor,
    *,
    n_nodes: int,
) -> torch.Tensor:
    if cp_scale.ndim != 1:
        raise ValueError(f"cp_scale must be 1D, got {cp_scale.shape}")
    if cp_scale.numel() < 2:
        raise ValueError("At least 2 control points are required.")

    x = cp_scale.reshape(1, 1, -1)
    y = F.interpolate(
        x,
        size=n_nodes,
        mode="linear",
        align_corners=True,
    )
    return y.reshape(-1)


def build_node_scaled_k_torch(
    *,
    K_base: torch.Tensor,
    node_scale: torch.Tensor,
    scale_mode: str,
) -> torch.Tensor:
    n_dofs = K_base.shape[0]
    if n_dofs % 6 != 0:
        raise ValueError(f"n_dofs must be divisible by 6, got {n_dofs}")

    n_nodes = n_dofs // 6
    if node_scale.numel() != n_nodes:
        raise ValueError(
            f"node_scale length mismatch: expected {n_nodes}, got {node_scale.numel()}"
        )

    s = torch.ones(n_dofs, dtype=K_base.dtype, device=K_base.device)
    sqrt_scale = torch.sqrt(node_scale)

    uy_idx = torch.arange(n_nodes, dtype=torch.long, device=K_base.device) * 6 + 1
    s[uy_idx] = sqrt_scale

    if scale_mode == "y_bending":
        rx_idx = torch.arange(n_nodes, dtype=torch.long, device=K_base.device) * 6 + 3
        s[rx_idx] = sqrt_scale
    elif scale_mode == "uy_only":
        pass
    else:
        raise ValueError(f"Unsupported scale_mode: {scale_mode}")

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


def regularization_terms(
    *,
    cp_scale: torch.Tensor,
    node_scale: torch.Tensor,
    prior_scale: float,
) -> Dict[str, torch.Tensor]:
    scale_prior = torch.mean((node_scale - prior_scale) ** 2)

    if cp_scale.numel() >= 2:
        d1 = cp_scale[1:] - cp_scale[:-1]
        smooth = torch.mean(d1 ** 2)
    else:
        smooth = torch.zeros((), dtype=cp_scale.dtype, device=cp_scale.device)

    if cp_scale.numel() >= 3:
        d2 = cp_scale[2:] - 2.0 * cp_scale[1:-1] + cp_scale[:-2]
        curvature = torch.mean(d2 ** 2)
    else:
        curvature = torch.zeros((), dtype=cp_scale.dtype, device=cp_scale.device)

    return {
        "scale_prior": scale_prior,
        "smooth": smooth,
        "curvature": curvature,
    }


def compute_case_loss(
    *,
    case: TrainingCase,
    M_t: torch.Tensor,
    K_base_t: torch.Tensor,
    beta_damp_t: torch.Tensor,
    raw_cp: torch.Tensor,
    cfg: TrainKappaYControlPointsConfig,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    n_nodes = K_base_t.shape[0] // 6

    cp_scale = control_scales_from_raw(
        raw_cp,
        scale_min=cfg.scale_min,
        scale_max=cfg.scale_max,
    )
    node_scale = interpolate_control_points_to_nodes(
        cp_scale,
        n_nodes=n_nodes,
    )

    K_scaled = build_node_scaled_k_torch(
        K_base=K_base_t,
        node_scale=node_scale,
        scale_mode=cfg.scale_mode,
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

    y_loss = (
        cfg.w_full_y * ratio_full_y
        + cfg.w_tip_y * ratio_tip_y
        + cfg.w_lastk_y * ratio_lastk_y
    ) / max(cfg.w_full_y + cfg.w_tip_y + cfg.w_lastk_y, 1.0e-12)

    x_guard_limit = 1.0 + cfg.x_guard_tol
    full_x_guard = torch.relu(ratio_full_x - x_guard_limit) ** 2
    tip_x_guard = torch.relu(ratio_tip_x - x_guard_limit) ** 2
    x_guard = (
        cfg.w_full_x_guard * full_x_guard
        + cfg.w_tip_x_guard * tip_x_guard
    )

    regs = regularization_terms(
        cp_scale=cp_scale,
        node_scale=node_scale,
        prior_scale=cfg.prior_scale,
    )

    reg_loss = (
        cfg.w_scale_prior * regs["scale_prior"]
        + cfg.w_smooth * regs["smooth"]
        + cfg.w_curvature * regs["curvature"]
    )

    total = y_loss + x_guard + reg_loss

    stats = {
        "total": float(total.detach().cpu().item()),
        "y_loss": float(y_loss.detach().cpu().item()),
        "full_y_ratio": float(ratio_full_y.detach().cpu().item()),
        "tip_y_ratio": float(ratio_tip_y.detach().cpu().item()),
        f"last{cfg.last_k}_y_ratio": float(ratio_lastk_y.detach().cpu().item()),
        "full_x_ratio": float(ratio_full_x.detach().cpu().item()),
        "tip_x_ratio": float(ratio_tip_x.detach().cpu().item()),
        "x_guard": float(x_guard.detach().cpu().item()),
        "reg_loss": float(reg_loss.detach().cpu().item()),
        "scale_prior": float(regs["scale_prior"].detach().cpu().item()),
        "smooth": float(regs["smooth"].detach().cpu().item()),
        "curvature": float(regs["curvature"].detach().cpu().item()),
        "cp_min": float(cp_scale.min().detach().cpu().item()),
        "cp_max": float(cp_scale.max().detach().cpu().item()),
        "cp_mean": float(cp_scale.mean().detach().cpu().item()),
        "node_min": float(node_scale.min().detach().cpu().item()),
        "node_max": float(node_scale.max().detach().cpu().item()),
        "node_mean": float(node_scale.mean().detach().cpu().item()),
    }

    return total, stats


def evaluate_cases(
    *,
    cases: List[TrainingCase],
    M_t: torch.Tensor,
    K_base_t: torch.Tensor,
    beta_damp_t: torch.Tensor,
    raw_cp: torch.Tensor,
    cfg: TrainKappaYControlPointsConfig,
    dtype: torch.dtype,
    device: torch.device,
) -> Dict[str, float]:
    if len(cases) == 0:
        cp_scale = control_scales_from_raw(
            raw_cp,
            scale_min=cfg.scale_min,
            scale_max=cfg.scale_max,
        ).detach().cpu().numpy()
        return {
            "total": float("nan"),
            "y_loss": float("nan"),
            "full_y_ratio": float("nan"),
            "tip_y_ratio": float("nan"),
            f"last{cfg.last_k}_y_ratio": float("nan"),
            "full_x_ratio": float("nan"),
            "tip_x_ratio": float("nan"),
            "x_guard": float("nan"),
            "reg_loss": float("nan"),
            "cp_min": float(np.min(cp_scale)),
            "cp_max": float(np.max(cp_scale)),
            "cp_mean": float(np.mean(cp_scale)),
            "node_min": float("nan"),
            "node_max": float("nan"),
            "node_mean": float("nan"),
        }

    aggregate: Dict[str, List[float]] = {}

    with torch.no_grad():
        for case in cases:
            _, stats = compute_case_loss(
                case=case,
                M_t=M_t,
                K_base_t=K_base_t,
                beta_damp_t=beta_damp_t,
                raw_cp=raw_cp,
                cfg=cfg,
                dtype=dtype,
                device=device,
            )
            for k, v in stats.items():
                aggregate.setdefault(k, []).append(float(v))

    return {k: float(np.mean(v)) for k, v in aggregate.items()}


def prepare_cases(
    *,
    cfg: TrainKappaYControlPointsConfig,
    teacher_exe: Path,
    template_inp: Path,
    blade_csv: Path,
) -> Tuple[List[TrainingCase], List[TrainingCase]]:
    train_cases: List[TrainingCase] = []
    valid_cases: List[TrainingCase] = []

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


def save_cp_params(
    *,
    path: Path,
    raw_cp: torch.Tensor,
    K_base_t: torch.Tensor,
    cfg: TrainKappaYControlPointsConfig,
    epoch: int,
    score: float,
) -> None:
    with torch.no_grad():
        cp_scale_t = control_scales_from_raw(
            raw_cp,
            scale_min=cfg.scale_min,
            scale_max=cfg.scale_max,
        )
        node_scale_t = interpolate_control_points_to_nodes(
            cp_scale_t,
            n_nodes=K_base_t.shape[0] // 6,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        cp_scale=cp_scale_t.detach().cpu().numpy().astype(np.float64),
        node_scale=node_scale_t.detach().cpu().numpy().astype(np.float64),
        raw_cp=raw_cp.detach().cpu().numpy().astype(np.float64),
        n_control_points=np.array([cfg.n_control_points], dtype=np.int64),
        scale_min=np.array([cfg.scale_min], dtype=np.float64),
        scale_max=np.array([cfg.scale_max], dtype=np.float64),
        init_scale=np.array([cfg.init_scale], dtype=np.float64),
        prior_scale=np.array([cfg.prior_scale], dtype=np.float64),
        epoch=np.array([epoch], dtype=np.int64),
        score=np.array([score], dtype=np.float64),
        scale_mode=np.array([cfg.scale_mode]),
    )


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
    print("[Train Kappa-y Control Points Torch]")
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

    M_t = torch.as_tensor(M_base, dtype=dtype, device=device)
    K_base_t = torch.as_tensor(K_base, dtype=dtype, device=device)
    beta_damp_t = torch.as_tensor(beta_damp, dtype=dtype, device=device)

    print()
    print("[3/5] Initializing trainable kappa_y control points")
    if cfg.n_control_points < 2:
        raise ValueError("n_control_points must be >= 2.")

    raw_init = init_raw_from_scale(cfg.init_scale, cfg.scale_min, cfg.scale_max)
    raw_cp = torch.nn.Parameter(
        torch.full(
            (cfg.n_control_points,),
            fill_value=raw_init,
            dtype=dtype,
            device=device,
        )
    )

    print(f"  n_control_points = {cfg.n_control_points}")
    print(f"  scale range      = [{cfg.scale_min:.6f}, {cfg.scale_max:.6f}]")
    print(f"  init_scale       = {cfg.init_scale:.6f}")
    print(f"  raw_init         = {raw_init:.8e}")
    print(f"  scale_mode       = {cfg.scale_mode}")
    print("  no f_static is trained in this script")

    optimizer = torch.optim.Adam(
        [raw_cp],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    print()
    print("[4/5] Training")
    print("  active parameter = kappa_y_scale(s) with control points")

    best_score = float("inf")
    best_epoch = 0
    best_raw_cp = None
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
                raw_cp=raw_cp,
                cfg=cfg,
                dtype=dtype,
                device=device,
            )
            train_losses.append(loss)

        train_loss = torch.stack(train_losses).mean()
        train_loss.backward()

        if cfg.grad_clip_norm is not None and cfg.grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_([raw_cp], max_norm=cfg.grad_clip_norm)

        optimizer.step()

        train_stats = evaluate_cases(
            cases=train_cases,
            M_t=M_t,
            K_base_t=K_base_t,
            beta_damp_t=beta_damp_t,
            raw_cp=raw_cp,
            cfg=cfg,
            dtype=dtype,
            device=device,
        )
        valid_stats = evaluate_cases(
            cases=valid_cases,
            M_t=M_t,
            K_base_t=K_base_t,
            beta_damp_t=beta_damp_t,
            raw_cp=raw_cp,
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
            best_raw_cp = raw_cp.detach().cpu().clone()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        row = {
            "epoch": epoch,
            "train_total": train_stats["total"],
            "train_y_loss": train_stats["y_loss"],
            "train_full_y_ratio": train_stats["full_y_ratio"],
            "train_tip_y_ratio": train_stats["tip_y_ratio"],
            f"train_last{cfg.last_k}_y_ratio": train_stats[f"last{cfg.last_k}_y_ratio"],
            "train_full_x_ratio": train_stats["full_x_ratio"],
            "train_tip_x_ratio": train_stats["tip_x_ratio"],
            "train_x_guard": train_stats["x_guard"],
            "train_reg_loss": train_stats["reg_loss"],
            "train_cp_min": train_stats["cp_min"],
            "train_cp_max": train_stats["cp_max"],
            "train_cp_mean": train_stats["cp_mean"],
            "train_node_min": train_stats["node_min"],
            "train_node_max": train_stats["node_max"],
            "train_node_mean": train_stats["node_mean"],
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
                f"train_total={train_stats['total']:.8e} "
                f"train_y={train_stats['y_loss']:.6f} "
                f"full_y={train_stats['full_y_ratio']:.6f} "
                f"tip_y={train_stats['tip_y_ratio']:.6f} "
                f"last{cfg.last_k}_y={train_stats[f'last{cfg.last_k}_y_ratio']:.6f} "
                f"full_x={train_stats['full_x_ratio']:.6f} "
                f"tip_x={train_stats['tip_x_ratio']:.6f} "
                f"cp=[{train_stats['cp_min']:.5f},{train_stats['cp_max']:.5f}] "
                f"mean={train_stats['cp_mean']:.5f} "
                f"reg={train_stats['reg_loss']:.3e} "
                f"valid_total={valid_stats['total']:.8e} "
                f"valid_y={valid_stats['y_loss']:.6f}"
                f"{best_mark}"
            )

        if epoch >= cfg.min_epochs and no_improve_epochs >= cfg.early_stop_patience:
            print()
            print("[Early Stop]")
            print(f"  epoch = {epoch}")
            print(f"  best_epoch = {best_epoch}")
            print(f"  best_score = {best_score:.8e}")
            print(f"  no_improve_epochs = {no_improve_epochs}")
            break

    if best_raw_cp is None:
        best_raw_cp = raw_cp.detach().cpu().clone()

    print()
    print("[5/5] Saving results")

    best_params_path = output_dir / "best_kappa_y_cp_params.npz"
    final_params_path = output_dir / "final_kappa_y_cp_params.npz"
    history_path = output_dir / "training_history.csv"
    report_path = output_dir / "train_kappa_y_cp_report.json"

    save_cp_params(
        path=best_params_path,
        raw_cp=best_raw_cp.to(dtype=dtype, device=device),
        K_base_t=K_base_t,
        cfg=cfg,
        epoch=best_epoch,
        score=best_score,
    )
    save_cp_params(
        path=final_params_path,
        raw_cp=raw_cp.detach(),
        K_base_t=K_base_t,
        cfg=cfg,
        epoch=history[-1]["epoch"],
        score=history[-1]["train_total"],
    )
    save_history_csv(history_path, history)

    with torch.no_grad():
        best_cp = control_scales_from_raw(
            best_raw_cp.to(dtype=dtype, device=device),
            scale_min=cfg.scale_min,
            scale_max=cfg.scale_max,
        )
        best_node = interpolate_control_points_to_nodes(
            best_cp,
            n_nodes=K_base_t.shape[0] // 6,
        )

        final_cp = control_scales_from_raw(
            raw_cp.detach(),
            scale_min=cfg.scale_min,
            scale_max=cfg.scale_max,
        )
        final_node = interpolate_control_points_to_nodes(
            final_cp,
            n_nodes=K_base_t.shape[0] // 6,
        )

    final_train_stats = evaluate_cases(
        cases=train_cases,
        M_t=M_t,
        K_base_t=K_base_t,
        beta_damp_t=beta_damp_t,
        raw_cp=raw_cp,
        cfg=cfg,
        dtype=dtype,
        device=device,
    )
    final_valid_stats = evaluate_cases(
        cases=valid_cases,
        M_t=M_t,
        K_base_t=K_base_t,
        beta_damp_t=beta_damp_t,
        raw_cp=raw_cp,
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
            "best_cp_scale": best_cp.detach().cpu().numpy().astype(float).tolist(),
            "best_node_scale_min": float(best_node.min().detach().cpu().item()),
            "best_node_scale_max": float(best_node.max().detach().cpu().item()),
            "best_node_scale_mean": float(best_node.mean().detach().cpu().item()),
        },
        "final": {
            "final_cp_scale": final_cp.detach().cpu().numpy().astype(float).tolist(),
            "final_node_scale_min": float(final_node.min().detach().cpu().item()),
            "final_node_scale_max": float(final_node.max().detach().cpu().item()),
            "final_node_scale_mean": float(final_node.mean().detach().cpu().item()),
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
            "This script trains static kappa_y_scale(s) using low-dimensional control points. "
            "No f_static/load mapping and no Transformer/dynamic parameter are trained here."
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
    print(f"  best_cp_scale = {best_cp.detach().cpu().numpy()}")

    print()
    print("[Final Result]")
    print(f"  final_cp_scale = {final_cp.detach().cpu().numpy()}")
    print(f"  final_train_y_loss = {final_train_stats['y_loss']:.8f}")
    if len(valid_cases) > 0:
        print(f"  final_valid_y_loss = {final_valid_stats['y_loss']:.8f}")

    print()
    print("✅ PASS: kappa_y control-points training completed.")
    print("   当前脚本只训练 κ：6 个展向控制点形式的 y-bending 静态刚度缩放；不包含 f_static。")


if __name__ == "__main__":
    main()