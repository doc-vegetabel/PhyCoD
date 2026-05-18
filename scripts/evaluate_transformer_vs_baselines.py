from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml
from scipy.linalg import eigh

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from src.teacher.beamdyn_teacher_adapter import (  # noqa: E402
    BeamDynTeacherCaseConfig,
    run_teacher_case_beamdyn,
    load_teacher_6dof_response,
    resample_response_to_time_grid,
)
from src.student.base_full_order_builder import build_base_student_full_order_mk  # noqa: E402
from src.student.dynamic_solver import NewmarkBetaSolver  # noqa: E402
from src.student.load_adapter import build_student_force_history_from_case  # noqa: E402
from src.student.transformer.blade_geometry_features import (  # noqa: E402
    BladeGeometryFeatureConfig,
    build_blade_geometry_features,
)
from src.student.transformer.physical_parameter_registry import (  # noqa: E402
    build_physical_parameter_registry,
)
from scripts.train_transformer_physical_params_torch import (  # noqa: E402
    TransformerPhysicalTrainConfig,
    build_training_model,
    _get_torch_dtype,
)
from src.student.transformer.spatiotemporal_physics_encoder import (  # noqa: E402
    compute_causal_load_spectral_features_from_force,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate trained Transformer against teacher, no-scale baseline, "
            "and static kappa-y baseline."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the primary Transformer checkpoint, e.g. alpha+beta best_transformer_physical_params.pt.",
    )
    parser.add_argument(
        "--alpha-checkpoint",
        type=str,
        default=None,
        help="Optional alpha-only checkpoint to plot/evaluate alongside the primary checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-label",
        type=str,
        default=None,
        help="Plot/metric label for --checkpoint. Defaults to alpha_beta when --alpha-checkpoint is set.",
    )
    parser.add_argument(
        "--alpha-label",
        type=str,
        default="alpha_only",
        help="Plot/metric label for --alpha-checkpoint.",
    )
    parser.add_argument(
        "--load-file",
        type=str,
        default=str(PROJECT_ROOT / "data" / "load" / "test_case_1.dat"),
        help="Test load .dat file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "results" / "transformer" / "eval_transformer_vs_baselines"),
    )

    parser.add_argument("--teacher-exe", type=str, default=None)
    parser.add_argument("--template-inp", type=str, default=None)
    parser.add_argument("--blade-csv", type=str, default=None)

    parser.add_argument("--case-name", type=str, default="eval_test_case")
    parser.add_argument("--t-initial", type=float, default=None)
    parser.add_argument("--t-final", type=float, default=None)
    parser.add_argument("--dt", type=float, default=None)

    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--teacher-node-start", type=int, default=None)
    parser.add_argument("--teacher-node-end", type=int, default=None)
    parser.add_argument("--teacher-demean", action="store_true", default=False)

    parser.add_argument("--remove-initial-offset", action="store_true", default=True)
    parser.add_argument("--keep-initial-offset", dest="remove_initial_offset", action="store_false")

    parser.add_argument("--kappa-y-static-scale", type=float, default=None)
    parser.add_argument(
        "--kappa-y-scale-mode",
        type=str,
        default=None,
        choices=["uy_only", "y_bending"],
    )

    parser.add_argument("--save-npz", action="store_true", default=True)
    parser.add_argument("--no-save-npz", dest="save_npz", action="store_false")

    return parser.parse_args()


def make_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if torch.is_tensor(obj):
        return obj.detach().cpu().numpy().tolist()

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, Path):
        return str(obj)

    return obj


def save_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(make_json_safe(payload), f, sort_keys=False, allow_unicode=True)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(payload), f, indent=2, ensure_ascii=False)


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    keys = list(rows[0].keys())
    seen = set(keys)
    for row in rows[1:]:
        for key in row.keys():
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def config_from_checkpoint(
    ckpt: dict[str, Any],
    args: argparse.Namespace,
) -> TransformerPhysicalTrainConfig:
    default_cfg = TransformerPhysicalTrainConfig()
    cfg_dict = asdict(default_cfg)

    ckpt_cfg = ckpt.get("cfg", {})
    if ckpt_cfg is not None:
        cfg_dict.update(ckpt_cfg)

    if args.teacher_exe is not None:
        cfg_dict["teacher_exe"] = args.teacher_exe
    if args.template_inp is not None:
        cfg_dict["template_inp"] = args.template_inp
    if args.blade_csv is not None:
        cfg_dict["blade_csv"] = args.blade_csv

    if args.t_initial is not None:
        cfg_dict["t_initial"] = float(args.t_initial)
    if args.t_final is not None:
        cfg_dict["t_final"] = float(args.t_final)
    if args.dt is not None:
        cfg_dict["dt"] = float(args.dt)

    if args.teacher_node_start is not None:
        cfg_dict["teacher_node_start"] = int(args.teacher_node_start)
    if args.teacher_node_end is not None:
        cfg_dict["teacher_node_end"] = int(args.teacher_node_end)

    cfg_dict["teacher_demean"] = bool(args.teacher_demean)

    if args.kappa_y_static_scale is not None:
        cfg_dict["kappa_y_static_scale"] = float(args.kappa_y_static_scale)
    if args.kappa_y_scale_mode is not None:
        cfg_dict["kappa_y_scale_mode"] = str(args.kappa_y_scale_mode)

    cfg_dict["device"] = str(args.device)

    valid_names = {f.name for f in fields(TransformerPhysicalTrainConfig)}
    cfg_dict = {k: v for k, v in cfg_dict.items() if k in valid_names}

    return TransformerPhysicalTrainConfig(**cfg_dict)


def assert_file(path: str | Path, name: str) -> Path:
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"{name} not found: {p}")
    return p


def remove_initial_offset(u: np.ndarray) -> np.ndarray:
    return np.asarray(u, dtype=np.float64) - np.asarray(u[:1, :], dtype=np.float64)


def compute_natural_frequencies_hz(
    M: np.ndarray,
    K: np.ndarray,
    *,
    num_modes: int = 10,
) -> np.ndarray:
    eigvals, _ = eigh(K, M)
    eigvals = np.asarray(eigvals, dtype=np.float64)
    eigvals = eigvals[eigvals > 0.0]
    if eigvals.size == 0:
        raise ValueError("No positive eigenvalues found.")
    return np.sqrt(eigvals[:num_modes]) / (2.0 * np.pi)


def build_structural_damping_matrix(
    *,
    K: np.ndarray,
    zeta_structural: float,
    ref_freq_hz: Optional[float],
    natural_freqs_hz: np.ndarray,
) -> tuple[np.ndarray, Optional[float]]:
    if zeta_structural <= 0.0:
        return np.zeros_like(K), ref_freq_hz

    if ref_freq_hz is None:
        ref_freq_hz = float(natural_freqs_hz[0])

    damping_coeff = 2.0 * float(zeta_structural) / (2.0 * np.pi * float(ref_freq_hz))
    C = damping_coeff * K
    return np.asarray(C, dtype=np.float64), float(ref_freq_hz)


def component_indices(n_nodes: int, component: str) -> np.ndarray:
    offsets = {
        "x": 0,
        "y": 1,
        "z": 2,
        "rx": 3,
        "ry": 4,
        "rz": 5,
    }
    if component not in offsets:
        raise ValueError(f"Unknown component: {component}")
    off = offsets[component]
    return np.asarray([i * 6 + off for i in range(n_nodes)], dtype=np.int64)


def tip_component_index(n_nodes: int, component: str) -> int:
    return int(component_indices(n_nodes, component)[-1])


def last_k_component_indices(n_nodes: int, component: str, last_k: int = 5) -> np.ndarray:
    idx = component_indices(n_nodes, component)
    return idx[-int(last_k):]


def compute_metrics(
    *,
    name: str,
    u: np.ndarray,
    u_teacher: np.ndarray,
    n_nodes: int,
) -> dict[str, float]:
    diff = np.asarray(u, dtype=np.float64) - np.asarray(u_teacher, dtype=np.float64)

    x_idx = component_indices(n_nodes, "x")
    y_idx = component_indices(n_nodes, "y")
    z_idx = component_indices(n_nodes, "z")

    tip_x = tip_component_index(n_nodes, "x")
    tip_y = tip_component_index(n_nodes, "y")
    tip_z = tip_component_index(n_nodes, "z")
    last5_y = last_k_component_indices(n_nodes, "y", last_k=5)

    def mse(a: np.ndarray) -> float:
        return float(np.mean(a ** 2))

    def mae(a: np.ndarray) -> float:
        return float(np.mean(np.abs(a)))

    out = {
        "model": name,
        "all_mse": mse(diff),
        "all_mae": mae(diff),
        "all_rmse": float(np.sqrt(mse(diff))),

        "full_x_mse": mse(diff[:, x_idx]),
        "full_x_mae": mae(diff[:, x_idx]),
        "full_x_rmse": float(np.sqrt(mse(diff[:, x_idx]))),

        "full_y_mse": mse(diff[:, y_idx]),
        "full_y_mae": mae(diff[:, y_idx]),
        "full_y_rmse": float(np.sqrt(mse(diff[:, y_idx]))),

        "full_z_mse": mse(diff[:, z_idx]),
        "full_z_mae": mae(diff[:, z_idx]),
        "full_z_rmse": float(np.sqrt(mse(diff[:, z_idx]))),

        "tip_x_mse": mse(diff[:, tip_x]),
        "tip_x_mae": mae(diff[:, tip_x]),
        "tip_x_rmse": float(np.sqrt(mse(diff[:, tip_x]))),

        "tip_y_mse": mse(diff[:, tip_y]),
        "tip_y_mae": mae(diff[:, tip_y]),
        "tip_y_rmse": float(np.sqrt(mse(diff[:, tip_y]))),

        "tip_z_mse": mse(diff[:, tip_z]),
        "tip_z_mae": mae(diff[:, tip_z]),
        "tip_z_rmse": float(np.sqrt(mse(diff[:, tip_z]))),

        "last5_y_mse": mse(diff[:, last5_y]),
        "last5_y_mae": mae(diff[:, last5_y]),
        "last5_y_rmse": float(np.sqrt(mse(diff[:, last5_y]))),
    }

    return out


def _series_for_indices(u: np.ndarray, idx: int | np.ndarray) -> np.ndarray:
    arr = np.asarray(u, dtype=np.float64)
    if np.isscalar(idx):
        return arr[:, int(idx)]
    idx_arr = np.asarray(idx, dtype=np.int64)
    return np.mean(arr[:, idx_arr], axis=1)


def _demeaned_rms(signal: np.ndarray) -> float:
    x = np.asarray(signal, dtype=np.float64)
    x = x - np.mean(x)
    return float(np.sqrt(np.mean(x * x)))


def compute_alpha_beta_diagnostics(
    *,
    time: np.ndarray,
    u_alpha: np.ndarray,
    u_beta: np.ndarray,
    u_teacher: np.ndarray,
    n_nodes: int,
    beta_theta_dict: dict[str, np.ndarray],
) -> dict[str, float]:
    alpha = np.asarray(u_alpha, dtype=np.float64)
    beta = np.asarray(u_beta, dtype=np.float64)
    teacher = np.asarray(u_teacher, dtype=np.float64)
    delta = beta - alpha
    alpha_error = alpha - teacher
    eps = 1.0e-12

    tip_x = tip_component_index(n_nodes, "x")
    tip_y = tip_component_index(n_nodes, "y")
    last5_x = last_k_component_indices(n_nodes, "x", last_k=5)
    last5_y = last_k_component_indices(n_nodes, "y", last_k=5)
    groups: dict[str, int | np.ndarray] = {
        "tip_x": tip_x,
        "tip_y": tip_y,
        "last5_x": last5_x,
        "last5_y": last5_y,
    }

    out: dict[str, float] = {
        "alpha_beta_delta_rms": float(np.sqrt(np.mean(delta * delta))),
        "alpha_beta_delta_max_abs": float(np.max(np.abs(delta))),
        "alpha_beta_delta_to_alpha_error_ratio": float(
            np.sqrt(np.mean(delta * delta)) / max(float(np.sqrt(np.mean(alpha_error * alpha_error))), eps)
        ),
    }

    late_mask = np.asarray(time, dtype=np.float64) >= 0.5 * (float(time[0]) + float(time[-1]))
    if not np.any(late_mask):
        late_mask = np.ones_like(np.asarray(time), dtype=bool)

    for name, idx in groups.items():
        d = _series_for_indices(delta, idx)
        e = _series_for_indices(alpha_error, idx)
        out[f"{name}_alpha_beta_delta_rms"] = _demeaned_rms(d)
        out[f"{name}_alpha_beta_delta_max_abs"] = float(np.max(np.abs(d)))
        out[f"{name}_alpha_beta_delta_to_alpha_error_ratio"] = _demeaned_rms(d) / max(_demeaned_rms(e), eps)

        alpha_late = _series_for_indices(alpha[late_mask], idx)
        beta_late = _series_for_indices(beta[late_mask], idx)
        teacher_late = _series_for_indices(teacher[late_mask], idx)
        teacher_rms = max(_demeaned_rms(teacher_late), eps)
        alpha_ratio = _demeaned_rms(alpha_late) / teacher_rms
        beta_ratio = _demeaned_rms(beta_late) / teacher_rms
        out[f"late_{name}_rms_alpha_to_teacher"] = alpha_ratio
        out[f"late_{name}_rms_beta_to_teacher"] = beta_ratio
        out[f"late_{name}_rms_beta_to_alpha"] = beta_ratio / max(alpha_ratio, eps)

    for name in ("beta_force_x", "beta_force_y"):
        if name not in beta_theta_dict:
            continue
        arr = np.asarray(beta_theta_dict[name], dtype=np.float64).reshape(-1)
        if arr.size == 0:
            continue
        out[f"{name}_theta_mean"] = float(np.mean(arr))
        out[f"{name}_theta_rms"] = float(np.sqrt(np.mean(arr * arr)))
        out[f"{name}_theta_min"] = float(np.min(arr))
        out[f"{name}_theta_max"] = float(np.max(arr))
        out[f"{name}_theta_std"] = float(np.std(arr))

    return out


def run_teacher_response(
    *,
    cfg: TransformerPhysicalTrainConfig,
    load_file: Path,
    output_dir: Path,
    case_name: str,
) -> tuple[np.ndarray, np.ndarray, list[str] | None, Path]:
    teacher_cfg = BeamDynTeacherCaseConfig(
        case_name=case_name,
        teacher_exe=Path(cfg.teacher_exe).resolve(),
        template_inp=Path(cfg.template_inp).resolve(),
        output_dir=output_dir / "teacher",
        use_time_series_load=True,
        t_initial=float(cfg.t_initial),
        t_final=float(cfg.t_final),
        dt=float(cfg.dt),
        time_series_load_file=load_file,
        keep_temp_inp=False,
    )

    teacher_result = run_teacher_case_beamdyn(teacher_cfg)
    teacher_out = Path(teacher_result["out"]).resolve()

    time_teacher, u_teacher_raw, teacher_columns = load_teacher_6dof_response(
        out_path=teacher_out,
        node_start=int(cfg.teacher_node_start),
        node_end=int(cfg.teacher_node_end),
        demean=bool(cfg.teacher_demean),
    )

    return (
        np.asarray(time_teacher, dtype=np.float64),
        np.asarray(u_teacher_raw, dtype=np.float64),
        teacher_columns,
        teacher_out,
    )


def run_base_student_response(
    *,
    cfg: TransformerPhysicalTrainConfig,
    load_file: Path,
    output_dir: Path,
    case_name: str,
    use_kappa_y_global_scale: bool,
    kappa_y_global_scale: float,
    kappa_y_scale_mode: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    base_result = build_base_student_full_order_mk(
        blade_csv=Path(cfg.blade_csv).resolve(),
        model_name=f"eval_{case_name}",
        alpha_flap=float(cfg.alpha_flap),
        alpha_edge=float(cfg.alpha_edge),
        alpha_torsion=float(cfg.alpha_torsion),
        twist_column=str(cfg.base_phi_twist_column),
        phi_sign=float(cfg.base_phi_sign),
        rotate_mass=bool(cfg.rotate_mass),
        use_kappa_y_global_scale=bool(use_kappa_y_global_scale),
        kappa_y_global_scale=float(kappa_y_global_scale),
        kappa_y_scale_mode=str(kappa_y_scale_mode),
        verbose=True,
    )

    student_model = base_result["model"]
    M = np.asarray(base_result["M"], dtype=np.float64)
    K = np.asarray(base_result["K"], dtype=np.float64)

    natural_freqs_hz = compute_natural_frequencies_hz(M, K, num_modes=10)
    C, ref_freq_used = build_structural_damping_matrix(
        K=K,
        zeta_structural=float(cfg.zeta_structural),
        ref_freq_hz=cfg.ref_freq_hz,
        natural_freqs_hz=natural_freqs_hz,
    )

    n_dofs = int(M.shape[0])

    case = {
        "case_name": case_name,
        "use_time_series_load": True,
        "t_initial": float(cfg.t_initial),
        "t_final": float(cfg.t_final),
        "dt": float(cfg.dt),
        "time_series_load_file": str(load_file),
    }

    time, F_time, eta_points = build_student_force_history_from_case(
        case=case,
        student_model=student_model,
        n_dofs_full=n_dofs,
    )

    solver = NewmarkBetaSolver(
        M=M,
        K=K,
        C=C,
        dt=float(cfg.dt),
    )

    u0 = np.zeros(n_dofs, dtype=np.float64)
    v0 = np.zeros(n_dofs, dtype=np.float64)

    u_full, v_full, a_full = solver.solve(
        F_time=np.asarray(F_time, dtype=np.float64),
        u0=u0,
        v0=v0,
    )

    return {
        "time": np.asarray(time, dtype=np.float64),
        "F_time": np.asarray(F_time, dtype=np.float64),
        "u_full": np.asarray(u_full, dtype=np.float64),
        "v_full": np.asarray(v_full, dtype=np.float64),
        "a_full": np.asarray(a_full, dtype=np.float64),
        "M": M,
        "K": K,
        "C": C,
        "natural_freqs_hz": natural_freqs_hz,
        "ref_freq_used": ref_freq_used,
        "eta_points": np.asarray(eta_points, dtype=np.float64),
        "base_model_info": base_result.get("base_model_info", {}),
        "phi_element_deg": np.asarray(base_result.get("phi_element_deg", []), dtype=np.float64),
    }


def load_transformer_model(
    *,
    cfg: TransformerPhysicalTrainConfig,
    ckpt: dict[str, Any],
    geometry_dim: int,
    device: torch.device,
) -> torch.nn.Module:
    registry = build_physical_parameter_registry(
        enabled_params=str(cfg.enabled_params),
    )

    model, _ = build_training_model(
        cfg=cfg,
        registry=registry,
        geometry_dim=int(geometry_dim),
        dtype_core=_get_torch_dtype(cfg.core_dtype),
        device=device,
    )

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    elif "encoder_state_dict" in ckpt:
        model.encoder.load_state_dict(ckpt["encoder_state_dict"], strict=True)
    else:
        raise KeyError("Checkpoint does not contain model_state_dict or encoder_state_dict.")

    model.eval()
    return model


def _safe_label(label: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(label).strip())
    return out or "model"


def _extract_theta_series(result: dict[str, Any]) -> dict[str, np.ndarray]:
    theta_series: dict[str, np.ndarray] = {}
    theta_dict = result.get("theta_dict", {})
    for name, arr in theta_dict.items():
        arr_np = np.asarray(arr)
        if arr_np.ndim == 2 and arr_np.shape[1] == 1:
            theta_series[name] = arr_np[:, 0]
        else:
            theta_series[name] = arr_np.reshape((arr_np.shape[0], -1))[:, 0]

    param_names = list(theta_dict.keys())
    for aux_name, arr in result.get("theta_aux", {}).items():
        arr_np = np.asarray(arr)
        if arr_np.ndim == 1:
            theta_series[aux_name] = arr_np
        elif arr_np.ndim == 2 and arr_np.shape[1] == 1:
            theta_series[aux_name] = arr_np[:, 0]
        elif arr_np.ndim == 2 and arr_np.shape[1] == len(param_names):
            suffix = aux_name.replace("theta_", "")
            for j, param_name in enumerate(param_names):
                theta_series[f"{param_name}_{suffix}"] = arr_np[:, j]
        elif arr_np.ndim >= 2:
            flat = arr_np.reshape((arr_np.shape[0], -1))
            for j in range(flat.shape[1]):
                theta_series[f"{aux_name}_{j}"] = flat[:, j]
    return theta_series




def _infer_load_spectral_window_size(cfg: TransformerPhysicalTrainConfig) -> Optional[int]:
    """
    Match the training-side load spectral feature window.

    Priority:
      1. cfg.load_spectral_window_size
      2. cfg.temporal_window_size
      3. full available history inside encoder/helper
    """
    if getattr(cfg, "load_spectral_window_size", None) is not None:
        return int(cfg.load_spectral_window_size)
    if getattr(cfg, "temporal_window_size", None) is not None:
        return int(cfg.temporal_window_size)
    return None


def compute_eval_load_spectral_features(
    *,
    cfg: TransformerPhysicalTrainConfig,
    F_time: np.ndarray,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Compute causal load spectral features during evaluation from the already aligned
    physical force history F_time.

    This mirrors the training-time design:
      - F_time itself remains the physical force for Newmark/MCK rollout.
      - F_spectral is only an auxiliary input to the load branch.
      - No extra .npz feature file is required.
    """
    if not bool(getattr(cfg, "use_load_spectral_features", False)):
        return None

    F_np = np.asarray(F_time, dtype=np.float64)
    if F_np.ndim != 2:
        raise ValueError(f"F_time must have shape (T,D), got {F_np.shape}.")
    if F_np.shape[1] % 6 != 0:
        raise ValueError(f"F_time last dim must be divisible by 6, got {F_np.shape[1]}.")

    n_nodes = int(F_np.shape[1] // 6)
    F_tensor = torch.as_tensor(F_np, dtype=torch.float32, device=device)

    F_spec = compute_causal_load_spectral_features_from_force(
        F=F_tensor,
        n_nodes=n_nodes,
        dof_per_node=6,
        window_size=_infer_load_spectral_window_size(cfg),
        dt=float(cfg.dt),
        freq_min=float(getattr(cfg, "load_spectral_freq_min", 0.05)),
        freq_max=float(getattr(cfg, "load_spectral_freq_max", 5.0)),
        bands=str(getattr(cfg, "load_spectral_bands", "0.05-0.5,0.5-1.5,1.5-5.0")),
        observations=str(getattr(cfg, "load_spectral_observations", "tip,last5,mean")),
        last_k=int(getattr(cfg, "load_spectral_last_k", 5)),
        active_rel_threshold=float(getattr(cfg, "load_spectral_active_rel_threshold", 1.0e-3)),
        active_abs_threshold=float(getattr(cfg, "load_spectral_active_abs_threshold", 1.0e-12)),
    )

    # Model/encoder expects (B,T,D_spec).
    return F_spec.to(device=device, dtype=_get_torch_dtype(cfg.encoder_dtype)).unsqueeze(0)

def run_transformer_response(
    *,
    model: torch.nn.Module,
    cfg: TransformerPhysicalTrainConfig,
    geometry: torch.Tensor,
    F_time: np.ndarray,
    u_static: np.ndarray,
    v_static: np.ndarray,
    a_static: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    dtype_core = _get_torch_dtype(cfg.core_dtype)

    F = torch.as_tensor(F_time, dtype=dtype_core, device=device).unsqueeze(0)
    u_static_t = torch.as_tensor(u_static, dtype=dtype_core, device=device).unsqueeze(0)
    v_static_t = torch.as_tensor(v_static, dtype=dtype_core, device=device).unsqueeze(0)
    a_static_t = torch.as_tensor(a_static, dtype=dtype_core, device=device).unsqueeze(0)

    F_spectral_t = compute_eval_load_spectral_features(
        cfg=cfg,
        F_time=F_time,
        device=device,
    )

    u0 = u_static_t[:, 0, :].to(dtype=torch.float32)
    v0 = v_static_t[:, 0, :].to(dtype=torch.float32)
    a0 = a_static_t[:, 0, :].to(dtype=torch.float32)

    with torch.no_grad():
        out = model(
            u_static=u_static_t,
            v_static=v_static_t,
            a_static=a_static_t,
            F=F,
            load_spectral_features=F_spectral_t,
            geometry_features=geometry,
            u0=u0,
            v0=v0,
            a0=a0,
        )

    theta = out.theta.detach().cpu().numpy()
    theta_dict = {
        k: v.detach().cpu().numpy()
        for k, v in out.theta_dict.items()
    }
    theta_aux = {
        k: v.detach().cpu().numpy()
        for k, v in (out.theta_aux or {}).items()
    }

    return {
        "u_full": out.u_pred[0].detach().cpu().numpy(),
        "v_full": out.v_pred[0].detach().cpu().numpy(),
        "a_full": out.a_pred[0].detach().cpu().numpy(),
        "theta": theta[0],
        "theta_dict": {k: v[0] for k, v in theta_dict.items()},
        "theta_aux": {k: v[0] for k, v in theta_aux.items()},
        "load_spectral_features": (
            None if F_spectral_t is None else F_spectral_t[0].detach().cpu().numpy()
        ),
    }


def plot_timeseries(
    *,
    time: np.ndarray,
    teacher: np.ndarray,
    no_scale: np.ndarray,
    static_scale: np.ndarray,
    transformer: np.ndarray,
    idx: int,
    title: str,
    ylabel: str,
    path: Path,
) -> None:
    """
    Plot four response histories.

    注意：
        在 tip_x 等分量上，不同模型的曲线可能高度重合。
        因此这里使用不同 linestyle / marker / zorder，
        避免图中看起来像“只有两条线”。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    markevery = max(1, len(time) // 25)

    plt.figure(figsize=(10, 5))

    plt.plot(
        time,
        teacher[:, idx],
        label="teacher",
        linewidth=2.2,
        linestyle="-",
        marker=None,
        zorder=4,
    )

    plt.plot(
        time,
        no_scale[:, idx],
        label="base no-scale",
        linewidth=1.4,
        linestyle="--",
        marker="o",
        markersize=3,
        markevery=markevery,
        zorder=1,
    )

    plt.plot(
        time,
        static_scale[:, idx],
        label="base kappa-scale",
        linewidth=1.6,
        linestyle="-.",
        marker="s",
        markersize=3,
        markevery=markevery,
        zorder=2,
    )

    plt.plot(
        time,
        transformer[:, idx],
        label="Transformer",
        linewidth=1.8,
        linestyle=":",
        marker="^",
        markersize=3,
        markevery=markevery,
        zorder=3,
    )

    plt.xlabel("time [s]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_response_series(
    *,
    time: np.ndarray,
    series: dict[str, np.ndarray],
    idx: int | np.ndarray,
    title: str,
    ylabel: str,
    path: Path,
    error_to: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    styles = {
        "teacher": {"linewidth": 2.3, "linestyle": "-", "zorder": 5},
        "base_student": {"linewidth": 1.6, "linestyle": "--", "zorder": 2},
        "alpha_only": {"linewidth": 1.8, "linestyle": "-.", "zorder": 3},
        "alpha_beta": {"linewidth": 2.0, "linestyle": ":", "zorder": 4},
        "transformer": {"linewidth": 2.0, "linestyle": ":", "zorder": 4},
    }

    reference = None
    if error_to is not None:
        if error_to not in series:
            raise KeyError(f"error_to={error_to!r} not found in series keys={list(series)}.")
        reference = np.asarray(series[error_to], dtype=np.float64)

    plt.figure(figsize=(10, 5))
    if reference is not None:
        plt.axhline(0.0, linewidth=1.0, linestyle="-", zorder=0)

    for label, values in series.items():
        arr = np.asarray(values, dtype=np.float64)
        if isinstance(idx, np.ndarray):
            y = np.mean(arr[:, idx], axis=1)
        else:
            y = arr[:, int(idx)]
        if reference is not None:
            if isinstance(idx, np.ndarray):
                y_ref = np.mean(reference[:, idx], axis=1)
            else:
                y_ref = reference[:, int(idx)]
            y = y - y_ref
            if label == error_to:
                continue

        style = styles.get(label, {"linewidth": 1.5, "linestyle": "-", "zorder": 1})
        plt.plot(time, y, label=label.replace("_", " "), **style)

    plt.xlabel("time [s]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_timeseries_error_to_teacher(
    *,
    time: np.ndarray,
    teacher: np.ndarray,
    no_scale: np.ndarray,
    static_scale: np.ndarray,
    transformer: np.ndarray,
    idx: int,
    title: str,
    ylabel: str,
    path: Path,
) -> None:
    """
    Plot response error relative to teacher.

    这个图比原始时历更适合判断：
        no-scale / static-scale / Transformer
    谁离 teacher 更近。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    markevery = max(1, len(time) // 25)

    plt.figure(figsize=(10, 5))

    plt.axhline(0.0, linewidth=1.0, linestyle="-", zorder=0)

    plt.plot(
        time,
        no_scale[:, idx] - teacher[:, idx],
        label="base no-scale - teacher",
        linewidth=1.4,
        linestyle="--",
        marker="o",
        markersize=3,
        markevery=markevery,
        zorder=1,
    )

    plt.plot(
        time,
        static_scale[:, idx] - teacher[:, idx],
        label="base kappa-scale - teacher",
        linewidth=1.6,
        linestyle="-.",
        marker="s",
        markersize=3,
        markevery=markevery,
        zorder=2,
    )

    plt.plot(
        time,
        transformer[:, idx] - teacher[:, idx],
        label="Transformer - teacher",
        linewidth=1.8,
        linestyle=":",
        marker="^",
        markersize=3,
        markevery=markevery,
        zorder=3,
    )

    plt.xlabel("time [s]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_mean_last5_y(
    *,
    time: np.ndarray,
    teacher: np.ndarray,
    no_scale: np.ndarray,
    static_scale: np.ndarray,
    transformer: np.ndarray,
    last5_y_idx: np.ndarray,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(time, np.mean(teacher[:, last5_y_idx], axis=1), label="teacher", linewidth=2.0)
    plt.plot(time, np.mean(no_scale[:, last5_y_idx], axis=1), label="base no-scale", linewidth=1.4)
    plt.plot(time, np.mean(static_scale[:, last5_y_idx], axis=1), label="base kappa-scale", linewidth=1.4)
    plt.plot(time, np.mean(transformer[:, last5_y_idx], axis=1), label="Transformer", linewidth=1.4)
    plt.xlabel("time [s]")
    plt.ylabel("mean last5 uy")
    plt.title("Mean last-5 nodes y displacement")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_theta(
    *,
    time: np.ndarray,
    theta_dict: dict[str, np.ndarray],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    for name, arr in theta_dict.items():
        arr = np.asarray(arr)
        if arr.ndim == 2 and arr.shape[1] == 1:
            plt.plot(time, arr[:, 0], label=name, linewidth=1.6)
        else:
            for j in range(arr.shape[-1]):
                plt.plot(time, arr[:, j], label=f"{name}[{j}]", linewidth=1.2)

    plt.xlabel("time [s]")
    plt.ylabel("theta")
    plt.title("Transformer physical parameters")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()

    checkpoint = assert_file(args.checkpoint, "checkpoint")
    alpha_checkpoint = (
        assert_file(args.alpha_checkpoint, "alpha_checkpoint")
        if args.alpha_checkpoint is not None and str(args.alpha_checkpoint).strip()
        else None
    )
    load_file = assert_file(args.load_file, "load_file")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    print()
    print("=" * 100)
    print("[Evaluate] Transformer vs baselines")
    print("=" * 100)
    print(f"  checkpoint = {checkpoint}")
    if alpha_checkpoint is not None:
        print(f"  alpha_checkpoint = {alpha_checkpoint}")
    print(f"  load_file  = {load_file}")
    print(f"  output_dir = {output_dir}")

    ckpt = torch.load(checkpoint, map_location=device)
    cfg = config_from_checkpoint(ckpt, args)
    alpha_ckpt = None
    alpha_cfg = None
    if alpha_checkpoint is not None:
        alpha_ckpt = torch.load(alpha_checkpoint, map_location=device)
        alpha_cfg = config_from_checkpoint(alpha_ckpt, args)

    print(f"  enabled_params = {cfg.enabled_params}")
    if alpha_cfg is not None:
        print(f"  alpha_enabled_params = {alpha_cfg.enabled_params}")
    print(f"  use_phase_gated_decomposition = {getattr(cfg, 'use_phase_gated_decomposition', False)}")
    print(f"  use_load_spectral_features = {getattr(cfg, 'use_load_spectral_features', False)}")
    if bool(getattr(cfg, 'use_load_spectral_features', False)):
        print(f"  load_spectral_feature_dim = {getattr(cfg, 'load_spectral_feature_dim', None)}")
        print(f"  load_spectral_window_size = {_infer_load_spectral_window_size(cfg)}")

    cfg.output_dir = str(output_dir)
    cfg.device = str(device)
    if alpha_cfg is not None:
        alpha_cfg.output_dir = str(output_dir)
        alpha_cfg.device = str(device)

    assert_file(cfg.teacher_exe, "teacher_exe")
    assert_file(cfg.template_inp, "template_inp")
    assert_file(cfg.blade_csv, "blade_csv")

    save_yaml(
        output_dir / "eval_config.yaml",
        {
            "args": vars(args),
            "cfg": asdict(cfg),
            "checkpoint_epoch": ckpt.get("epoch", None),
            "checkpoint_best_score": ckpt.get("best_score", None),
            "alpha_checkpoint": None if alpha_checkpoint is None else str(alpha_checkpoint),
            "alpha_checkpoint_epoch": None if alpha_ckpt is None else alpha_ckpt.get("epoch", None),
            "alpha_checkpoint_best_score": None if alpha_ckpt is None else alpha_ckpt.get("best_score", None),
        },
    )

    # ------------------------------------------------------------
    # 1. Run teacher
    # ------------------------------------------------------------
    print()
    print("[1/6] Run BeamDyn teacher")

    teacher_case_name = f"{args.case_name}_teacher"
    time_teacher_raw, u_teacher_raw, teacher_columns, teacher_out = run_teacher_response(
        cfg=cfg,
        load_file=load_file,
        output_dir=output_dir,
        case_name=teacher_case_name,
    )

    # ------------------------------------------------------------
    # 2. Run no-scale baseline
    # ------------------------------------------------------------
    print()
    print("[2/6] Run base student no-scale")

    no_scale = run_base_student_response(
        cfg=cfg,
        load_file=load_file,
        output_dir=output_dir / "base_no_scale",
        case_name=f"{args.case_name}_base_no_scale",
        use_kappa_y_global_scale=False,
        kappa_y_global_scale=float(cfg.kappa_y_static_scale),
        kappa_y_scale_mode=str(cfg.kappa_y_scale_mode),
    )

    # ------------------------------------------------------------
    # 3. Run static kappa-y baseline
    # ------------------------------------------------------------
    print()
    print("[3/6] Run base student static kappa-y scale")

    static_scale = run_base_student_response(
        cfg=cfg,
        load_file=load_file,
        output_dir=output_dir / "base_static_kappa_y",
        case_name=f"{args.case_name}_base_static_kappa_y",
        use_kappa_y_global_scale=True,
        kappa_y_global_scale=float(cfg.kappa_y_static_scale),
        kappa_y_scale_mode=str(cfg.kappa_y_scale_mode),
    )

    time = static_scale["time"]
    F_time = static_scale["F_time"]

    if not np.allclose(time, no_scale["time"]):
        raise ValueError("no-scale and static-scale time grids mismatch.")

    u_teacher = resample_response_to_time_grid(
        time_src=time_teacher_raw,
        u_src=u_teacher_raw,
        time_dst=time,
    )
    u_teacher = np.asarray(u_teacher, dtype=np.float64)

    # ------------------------------------------------------------
    # 4. Slice max steps
    # ------------------------------------------------------------
    print()
    print("[4/6] Slice time window")

    T_total = int(time.shape[0])
    if args.max_steps is None:
        T_use = T_total
    else:
        T_use = min(int(args.max_steps), T_total)

    if T_use < 2:
        raise ValueError(f"Need at least 2 steps, got T_use={T_use}.")

    sl = slice(0, T_use)

    time = time[sl]
    F_time = F_time[sl]
    u_teacher = u_teacher[sl]

    u_no_scale = no_scale["u_full"][sl]
    u_static = static_scale["u_full"][sl]
    v_static = static_scale["v_full"][sl]
    a_static = static_scale["a_full"][sl]

    if bool(args.remove_initial_offset):
        u_teacher_cmp = remove_initial_offset(u_teacher)
        u_no_scale_cmp = remove_initial_offset(u_no_scale)
        u_static_cmp = remove_initial_offset(u_static)
    else:
        u_teacher_cmp = u_teacher.copy()
        u_no_scale_cmp = u_no_scale.copy()
        u_static_cmp = u_static.copy()

    print(f"  T_total = {T_total}")
    print(f"  T_use   = {T_use}")
    print(f"  t range = [{time[0]:.6f}, {time[-1]:.6f}]")

    # ------------------------------------------------------------
    # 5. Run Transformer checkpoints
    # ------------------------------------------------------------
    print()
    print("[5/6] Run trained Transformer checkpoint(s)")

    geo_bundle = build_blade_geometry_features(
        BladeGeometryFeatureConfig(
            blade_csv=str(cfg.blade_csv),
            twist_column=str(cfg.base_phi_twist_column),
            phi_sign=float(cfg.base_phi_sign),
            exclude_root_station=True,
        )
    )

    geometry = torch.as_tensor(
        geo_bundle.features,
        dtype=_get_torch_dtype(cfg.encoder_dtype),
        device=device,
    )

    primary_label = (
        str(args.checkpoint_label).strip()
        if args.checkpoint_label is not None and str(args.checkpoint_label).strip()
        else ("alpha_beta" if alpha_ckpt is not None else "transformer")
    )
    transformer_specs: list[tuple[str, TransformerPhysicalTrainConfig, dict[str, Any], Path]] = []
    if alpha_ckpt is not None and alpha_cfg is not None and alpha_checkpoint is not None:
        transformer_specs.append((str(args.alpha_label), alpha_cfg, alpha_ckpt, alpha_checkpoint))
    transformer_specs.append((primary_label, cfg, ckpt, checkpoint))

    transformer_results: dict[str, dict[str, Any]] = {}
    transformer_responses: dict[str, np.ndarray] = {}
    transformer_checkpoint_paths: dict[str, str] = {}

    for label, model_cfg, model_ckpt, model_checkpoint in transformer_specs:
        safe_label = _safe_label(label)
        print(f"  [{label}] checkpoint = {model_checkpoint}")
        model = load_transformer_model(
            cfg=model_cfg,
            ckpt=model_ckpt,
            geometry_dim=int(geo_bundle.feature_dim),
            device=device,
        )
        result = run_transformer_response(
            model=model,
            cfg=model_cfg,
            geometry=geometry,
            F_time=F_time,
            u_static=u_static_cmp,
            v_static=v_static,
            a_static=a_static,
            device=device,
        )
        transformer_results[label] = result
        transformer_responses[label] = result["u_full"]
        transformer_checkpoint_paths[label] = str(model_checkpoint)
        plot_theta(
            time=time,
            theta_dict=result["theta_dict"],
            path=output_dir / "plots" / f"theta_timeseries_{safe_label}.png",
        )

    # ------------------------------------------------------------
    # 6. Metrics + plots
    # ------------------------------------------------------------
    print()
    print("[6/6] Save metrics and plots")

    n_dofs = int(u_teacher_cmp.shape[1])
    n_nodes = n_dofs // 6

    metrics = [
        compute_metrics(
            name="base_no_scale",
            u=u_no_scale_cmp,
            u_teacher=u_teacher_cmp,
            n_nodes=n_nodes,
        ),
        compute_metrics(
            name="base_static_kappa_y",
            u=u_static_cmp,
            u_teacher=u_teacher_cmp,
            n_nodes=n_nodes,
        ),
    ]
    for label, response in transformer_responses.items():
        metrics.append(
            compute_metrics(
                name=label,
                u=response,
                u_teacher=u_teacher_cmp,
                n_nodes=n_nodes,
            )
        )

    static_metric = metrics[1]
    for m in metrics:
        for key in [
            "full_x_mse",
            "full_y_mse",
            "tip_y_mse",
            "last5_y_mse",
        ]:
            denom = static_metric[key]
            ratio_key = f"{key}_ratio_to_static"
            m[ratio_key] = float(m[key] / denom) if denom > 0.0 else np.nan

    alpha_beta_diagnostics: dict[str, float] = {}
    alpha_label_for_diag = str(args.alpha_label)
    if alpha_label_for_diag in transformer_responses and primary_label in transformer_responses:
        alpha_beta_diagnostics = compute_alpha_beta_diagnostics(
            time=time,
            u_alpha=transformer_responses[alpha_label_for_diag],
            u_beta=transformer_responses[primary_label],
            u_teacher=u_teacher_cmp,
            n_nodes=n_nodes,
            beta_theta_dict=transformer_results[primary_label].get("theta_dict", {}),
        )
        for m in metrics:
            for key in alpha_beta_diagnostics:
                m.setdefault(key, np.nan)
            if m.get("model") == primary_label:
                m.update(alpha_beta_diagnostics)

    save_csv(output_dir / "metrics.csv", metrics)

    metrics_summary = {
        "checkpoint": str(checkpoint),
        "transformer_checkpoints": transformer_checkpoint_paths,
        "load_file": str(load_file),
        "teacher_out": str(teacher_out),
        "T_use": T_use,
        "time_start": float(time[0]),
        "time_end": float(time[-1]),
        "remove_initial_offset": bool(args.remove_initial_offset),
        "metrics": metrics,
        "alpha_beta_diagnostics": alpha_beta_diagnostics,
        "theta_stats": {
            label: {
                name: {
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "max_abs": float(np.max(np.abs(arr))),
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                }
                for name, arr in result["theta_dict"].items()
            }
            for label, result in transformer_results.items()
        },
        "theta_aux_stats": {
            label: {
                name: {
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "max_abs": float(np.max(np.abs(arr))),
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                }
                for name, arr in result.get("theta_aux", {}).items()
            }
            for label, result in transformer_results.items()
        },
        "load_spectral_features": {
            "enabled": bool(getattr(cfg, "use_load_spectral_features", False)),
            "feature_dim": getattr(cfg, "load_spectral_feature_dim", None),
            "window_size": _infer_load_spectral_window_size(cfg),
            "freq_min": getattr(cfg, "load_spectral_freq_min", None),
            "freq_max": getattr(cfg, "load_spectral_freq_max", None),
            "bands": getattr(cfg, "load_spectral_bands", None),
            "observations": getattr(cfg, "load_spectral_observations", None),
            "normalize": bool(getattr(cfg, "load_spectral_normalize", False)),
        },
        "baseline_definitions": {
            "base_no_scale": {
                "Phi_base": "-initial_twist_deg",
                "use_kappa_y_global_scale": False,
            },
            "base_static_kappa_y": {
                "Phi_base": "-initial_twist_deg",
                "use_kappa_y_global_scale": True,
                "kappa_y_global_scale": float(cfg.kappa_y_static_scale),
                "kappa_y_scale_mode": str(cfg.kappa_y_scale_mode),
            },
            **{
                label: {
                    "base": "base_static_kappa_y",
                    "checkpoint": transformer_checkpoint_paths[label],
                    "enabled_params": str(model_cfg.enabled_params),
                }
                for label, model_cfg, _model_ckpt, _model_checkpoint in transformer_specs
            },
        },
    }

    save_json(output_dir / "metrics_summary.json", metrics_summary)

    tip_x = tip_component_index(n_nodes, "x")
    tip_y = tip_component_index(n_nodes, "y")
    last5_x = last_k_component_indices(n_nodes, "x", last_k=5)
    last5_y = last_k_component_indices(n_nodes, "y", last_k=5)

    comparison_series = {
        "teacher": u_teacher_cmp,
        "base_student": u_static_cmp,
        **transformer_responses,
    }

    plot_response_series(
        time=time,
        series=comparison_series,
        idx=tip_y,
        title="Tip y displacement comparison",
        ylabel="tip uy",
        path=output_dir / "plots" / "tip_y_timeseries.png",
    )
    plot_response_series(
        time=time,
        series=comparison_series,
        idx=tip_x,
        title="Tip x displacement comparison",
        ylabel="tip ux",
        path=output_dir / "plots" / "tip_x_timeseries.png",
    )
    plot_response_series(
        time=time,
        series=comparison_series,
        idx=last5_y,
        title="Mean last-5 nodes y displacement",
        ylabel="mean last5 uy",
        path=output_dir / "plots" / "last5_y_mean_timeseries.png",
    )
    plot_response_series(
        time=time,
        series=comparison_series,
        idx=last5_x,
        title="Mean last-5 nodes x displacement",
        ylabel="mean last5 ux",
        path=output_dir / "plots" / "last5_x_mean_timeseries.png",
    )
    plot_response_series(
        time=time,
        series=comparison_series,
        idx=tip_y,
        title="Tip y error relative to teacher",
        ylabel="tip uy error",
        path=output_dir / "plots" / "tip_y_error_to_teacher.png",
        error_to="teacher",
    )
    plot_response_series(
        time=time,
        series=comparison_series,
        idx=tip_x,
        title="Tip x error relative to teacher",
        ylabel="tip ux error",
        path=output_dir / "plots" / "tip_x_error_to_teacher.png",
        error_to="teacher",
    )
    plot_response_series(
        time=time,
        series=comparison_series,
        idx=last5_y,
        title="Mean last-5 y error relative to teacher",
        ylabel="mean last5 uy error",
        path=output_dir / "plots" / "last5_y_error_to_teacher.png",
        error_to="teacher",
    )
    plot_response_series(
        time=time,
        series=comparison_series,
        idx=last5_x,
        title="Mean last-5 x error relative to teacher",
        ylabel="mean last5 ux error",
        path=output_dir / "plots" / "last5_x_error_to_teacher.png",
        error_to="teacher",
    )

    theta_series_by_label = {
        label: _extract_theta_series(result)
        for label, result in transformer_results.items()
    }

    rows = []

    for i in range(T_use):
        row = {
            "time": float(time[i]),

            "teacher_tip_x": float(u_teacher_cmp[i, tip_x]),
            "no_scale_tip_x": float(u_no_scale_cmp[i, tip_x]),
            "base_student_tip_x": float(u_static_cmp[i, tip_x]),

            "teacher_tip_y": float(u_teacher_cmp[i, tip_y]),
            "no_scale_tip_y": float(u_no_scale_cmp[i, tip_y]),
            "base_student_tip_y": float(u_static_cmp[i, tip_y]),

            "teacher_last5_x_mean": float(np.mean(u_teacher_cmp[i, last5_x])),
            "no_scale_last5_x_mean": float(np.mean(u_no_scale_cmp[i, last5_x])),
            "base_student_last5_x_mean": float(np.mean(u_static_cmp[i, last5_x])),

            "teacher_last5_y_mean": float(np.mean(u_teacher_cmp[i, last5_y])),
            "no_scale_last5_y_mean": float(np.mean(u_no_scale_cmp[i, last5_y])),
            "base_student_last5_y_mean": float(np.mean(u_static_cmp[i, last5_y])),
        }

        for label, response in transformer_responses.items():
            safe_label = _safe_label(label)
            row[f"{safe_label}_tip_x"] = float(response[i, tip_x])
            row[f"{safe_label}_tip_y"] = float(response[i, tip_y])
            row[f"{safe_label}_last5_x_mean"] = float(np.mean(response[i, last5_x]))
            row[f"{safe_label}_last5_y_mean"] = float(np.mean(response[i, last5_y]))

        for label, theta_series in theta_series_by_label.items():
            safe_label = _safe_label(label)
            for name, arr in theta_series.items():
                row[f"{safe_label}_{name}"] = float(arr[i])

        rows.append(row)

    save_csv(output_dir / "selected_timeseries.csv", rows)

    if bool(args.save_npz):
        response_payload = {
            f"u_{_safe_label(label)}": response
            for label, response in transformer_responses.items()
        }
        theta_payload: dict[str, np.ndarray] = {}
        for label, result in transformer_results.items():
            safe_label = _safe_label(label)
            theta_payload[f"theta_{safe_label}"] = result["theta"]
            for name, arr in result["theta_dict"].items():
                theta_payload[f"theta_{safe_label}_{name}"] = arr
            for name, arr in result.get("theta_aux", {}).items():
                theta_payload[f"theta_aux_{safe_label}_{name}"] = arr

        primary_result = transformer_results[primary_label]
        np.savez(
            output_dir / "comparison_responses.npz",
            time=time,
            F_time=F_time,
            u_teacher=u_teacher_cmp,
            u_base_no_scale=u_no_scale_cmp,
            u_base_static_kappa_y=u_static_cmp,
            **response_payload,
            theta=primary_result["theta"],
            load_spectral_features=(
                np.empty((0, 0), dtype=np.float64)
                if primary_result.get("load_spectral_features") is None
                else primary_result["load_spectral_features"]
            ),
            **theta_payload,
        )

    print()
    print("[Summary]")
    for m in metrics:
        print(
            f"  {m['model']:<24s} "
            f"full_y_mse={m['full_y_mse']:.6e} "
            f"ratio_to_static={m['full_y_mse_ratio_to_static']:.6f} "
            f"full_x_ratio_to_static={m['full_x_mse_ratio_to_static']:.6f}"
        )

    print()
    print(f"  metrics_csv   = {output_dir / 'metrics.csv'}")
    print(f"  summary_json  = {output_dir / 'metrics_summary.json'}")
    print(f"  selected_csv  = {output_dir / 'selected_timeseries.csv'}")
    print(f"  plots_dir     = {output_dir / 'plots'}")
    print()
    print("✅ PASS: evaluation completed.")


if __name__ == "__main__":
    main()
