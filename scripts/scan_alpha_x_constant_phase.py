from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.evaluate_transformer_vs_baselines import (  # noqa: E402
    run_teacher_response,
    run_base_student_response,
    compute_metrics,
    resample_response_to_time_grid,
    remove_initial_offset,
    component_indices,
    tip_component_index,
    last_k_component_indices,
    plot_timeseries,
    plot_timeseries_error_to_teacher,
    save_json,
    save_csv,
    make_json_safe,
)
from scripts.train_transformer_physical_params_torch import (  # noqa: E402
    TransformerPhysicalTrainConfig,
    _get_torch_dtype,
    _compute_natural_frequencies_hz,
    _build_structural_damping_matrix,
)
from src.student.transformer.physical_parameter_registry import build_physical_parameter_registry  # noqa: E402
from src.student.transformer.physical_templates import PhysicalTemplateConfig, build_dynamic_stiffness_templates  # noqa: E402
from src.student.transformer.dynamic_physical_core_torch import DynamicPhysicalCoreConfig, DynamicPhysicalCoreTorch  # noqa: E402


def parse_float_list(text: str) -> list[float]:
    values = []
    for item in str(text).replace(";", ",").split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError("alpha list is empty.")
    return values



def save_csv_union(path: Path, rows: list[dict[str, Any]]) -> None:
    """Save heterogeneous metric rows using the union of all keys."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    preferred = [
        "model",
        "alpha_x",
        "full_x_mse",
        "full_y_mse",
        "full_x_mse_ratio_to_static",
        "full_y_mse_ratio_to_static",
        "tip_x_mse",
        "tip_y_mse",
        "tip_x_mse_ratio_to_static",
        "tip_y_mse_ratio_to_static",
        "last5_y_mse",
        "last5_y_mse_ratio_to_static",
    ]
    fieldnames = []
    seen = set()
    for key in preferred:
        if any(key in row for row in rows):
            fieldnames.append(key)
            seen.add(key)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    safe_rows = [make_json_safe(row) for row in rows]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in safe_rows:
            writer.writerow(row)
def parse_args() -> argparse.Namespace:
    d = TransformerPhysicalTrainConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Constant alpha_x sensitivity scan. It runs teacher/static baseline once, then "
            "rolls out K_eff = K0 + alpha_x * K_x_template for several constant alpha_x values."
        )
    )
    parser.add_argument("--load-file", type=str, default=str(PROJECT_ROOT / "data" / "load" / "test" / "test_complex_case_1.dat"))
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "results" / "transformer" / "alpha_x_constant_scan"))
    parser.add_argument("--case-name", type=str, default="alpha_x_constant_scan")
    parser.add_argument("--alpha-x-values", type=str, default="-0.10,-0.05,-0.03,0.0,0.03,0.05,0.07,0.10,0.15")
    parser.add_argument("--max-steps", type=int, default=1001)
    parser.add_argument("--device", type=str, default=d.device)
    parser.add_argument("--core-dtype", type=str, default="float64", choices=["float32", "float64"])

    parser.add_argument("--teacher-exe", type=str, default=d.teacher_exe)
    parser.add_argument("--template-inp", type=str, default=d.template_inp)
    parser.add_argument("--blade-csv", type=str, default=d.blade_csv)
    parser.add_argument("--dt", type=float, default=d.dt)
    parser.add_argument("--t-initial", type=float, default=d.t_initial)
    parser.add_argument("--t-final", type=float, default=d.t_final)
    parser.add_argument("--teacher-node-start", type=int, default=d.teacher_node_start)
    parser.add_argument("--teacher-node-end", type=int, default=d.teacher_node_end)
    parser.add_argument("--teacher-demean", action="store_true", default=d.teacher_demean)

    parser.add_argument("--kappa-y-static-scale", type=float, default=d.kappa_y_static_scale)
    parser.add_argument("--kappa-y-scale-mode", type=str, default=d.kappa_y_scale_mode, choices=["uy_only", "y_bending"])
    parser.add_argument("--zeta-structural", type=float, default=d.zeta_structural)
    parser.add_argument("--ref-freq-hz", type=float, default=d.ref_freq_hz)

    parser.add_argument("--remove-initial-offset", action="store_true", default=True)
    parser.add_argument("--keep-initial-offset", dest="remove_initial_offset", action="store_false")

    parser.add_argument("--phase-start", type=float, default=0.0, help="Start time for lag/correlation metrics.")
    parser.add_argument("--phase-end", type=float, default=None, help="End time for lag/correlation metrics. Default uses sequence end.")
    parser.add_argument("--lag-window-seconds", type=float, default=2.56)
    parser.add_argument("--lag-stride-seconds", type=float, default=1.28)
    parser.add_argument("--max-lag-seconds", type=float, default=0.80)
    parser.add_argument("--last-k", type=int, default=5)
    parser.add_argument("--save-plots", action="store_true", default=True)
    parser.add_argument("--no-save-plots", dest="save_plots", action="store_false")
    return parser.parse_args()


def build_cfg(args: argparse.Namespace) -> TransformerPhysicalTrainConfig:
    cfg = TransformerPhysicalTrainConfig()
    cfg.teacher_exe = str(args.teacher_exe)
    cfg.template_inp = str(args.template_inp)
    cfg.blade_csv = str(args.blade_csv)
    cfg.dt = float(args.dt)
    cfg.t_initial = float(args.t_initial)
    cfg.t_final = float(args.t_final)
    cfg.teacher_node_start = int(args.teacher_node_start)
    cfg.teacher_node_end = int(args.teacher_node_end)
    cfg.teacher_demean = bool(args.teacher_demean)
    cfg.kappa_y_static_scale = float(args.kappa_y_static_scale)
    cfg.kappa_y_scale_mode = str(args.kappa_y_scale_mode)
    cfg.zeta_structural = float(args.zeta_structural)
    cfg.ref_freq_hz = args.ref_freq_hz
    cfg.enabled_params = "alpha_x"
    cfg.device = str(args.device)
    cfg.core_dtype = str(args.core_dtype)
    cfg.encoder_dtype = "float32"
    cfg.remove_initial_offset = bool(args.remove_initial_offset)
    return cfg


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def run_constant_alpha_x_rollout(
    *,
    core: DynamicPhysicalCoreTorch,
    F_time: np.ndarray,
    u0: np.ndarray,
    v0: np.ndarray,
    a0: np.ndarray,
    alpha_x: float,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, np.ndarray]:
    F = torch.as_tensor(F_time, dtype=dtype, device=device)
    u_t = torch.as_tensor(u0, dtype=dtype, device=device).unsqueeze(0)
    v_t = torch.as_tensor(v0, dtype=dtype, device=device).unsqueeze(0)
    a_t = torch.as_tensor(a0, dtype=dtype, device=device).unsqueeze(0)
    theta = torch.as_tensor([float(alpha_x)], dtype=dtype, device=device)

    u_list = [u_t]
    v_list = [v_t]
    a_list = [a_t]

    with torch.no_grad():
        for t in range(int(F.shape[0]) - 1):
            u_t, v_t, a_t = core.newmark_step(
                u_t=u_t,
                v_t=v_t,
                a_t=a_t,
                F_t1=F[t + 1:t + 2, :],
                theta_t=theta,
            )
            u_list.append(u_t)
            v_list.append(v_t)
            a_list.append(a_t)

    return {
        "u_full": tensor_to_numpy(torch.cat(u_list, dim=0)),
        "v_full": tensor_to_numpy(torch.cat(v_list, dim=0)),
        "a_full": tensor_to_numpy(torch.cat(a_list, dim=0)),
        "theta": np.full((int(F.shape[0]), 1), float(alpha_x), dtype=np.float64),
    }


def signal_for_observation(u: np.ndarray, *, n_nodes: int, component: str, observation: str, last_k: int = 5) -> np.ndarray:
    observation = str(observation).lower()
    if observation == "tip":
        return np.asarray(u[:, tip_component_index(n_nodes, component)], dtype=np.float64)
    if observation == "last5" or observation == "lastk":
        idx = last_k_component_indices(n_nodes, component, last_k=last_k)
        return np.mean(np.asarray(u[:, idx], dtype=np.float64), axis=1)
    if observation == "mean" or observation == "all":
        idx = component_indices(n_nodes, component)
        return np.mean(np.asarray(u[:, idx], dtype=np.float64), axis=1)
    raise ValueError(f"Unsupported observation={observation!r}.")


def _safe_corr(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - np.mean(a)
    b = b - np.mean(b)
    den = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if den < eps:
        return 0.0
    return float(np.sum(a * b) / den)


def best_lag_for_window(pred: np.ndarray, target: np.ndarray, *, max_lag_steps: int) -> tuple[int, float, float]:
    """
    Returns (best_lag_steps, best_corr, corr_at_zero).

    Convention:
      lag > 0 means pred is best aligned by shifting pred earlier relative to target,
      i.e. pred tends to lag behind target.
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    L = int(pred.shape[0])
    max_lag_steps = min(int(max_lag_steps), max(0, L // 2 - 1))

    best_lag = 0
    best_corr = -np.inf
    corr0 = _safe_corr(pred, target)

    for lag in range(-max_lag_steps, max_lag_steps + 1):
        if lag < 0:
            # pred leads target: compare pred[-lag:] with target[:L+lag]
            p = pred[-lag:]
            q = target[: L + lag]
        elif lag > 0:
            # pred lags target: compare pred[:L-lag] with target[lag:]
            p = pred[: L - lag]
            q = target[lag:]
        else:
            p = pred
            q = target
        if p.size < 4:
            continue
        c = _safe_corr(p, q)
        if c > best_corr:
            best_corr = c
            best_lag = lag

    return int(best_lag), float(best_corr), float(corr0)


def local_lag_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    time: np.ndarray,
    start_time: float,
    end_time: Optional[float],
    window_seconds: float,
    stride_seconds: float,
    max_lag_seconds: float,
) -> dict[str, float]:
    dt = float(np.median(np.diff(time)))
    T = int(len(time))
    start_idx = int(np.searchsorted(time, float(start_time), side="left"))
    if end_time is None:
        end_idx = T
    else:
        end_idx = int(np.searchsorted(time, float(end_time), side="right"))
    end_idx = min(max(end_idx, start_idx + 2), T)

    win = max(8, int(round(float(window_seconds) / dt)))
    stride = max(1, int(round(float(stride_seconds) / dt)))
    max_lag_steps = max(1, int(round(float(max_lag_seconds) / dt)))

    lags = []
    best_corrs = []
    corr0s = []
    for i0 in range(start_idx, max(start_idx, end_idx - win) + 1, stride):
        i1 = i0 + win
        if i1 > end_idx:
            break
        lag, best_corr, corr0 = best_lag_for_window(pred[i0:i1], target[i0:i1], max_lag_steps=max_lag_steps)
        lags.append(lag * dt)
        best_corrs.append(best_corr)
        corr0s.append(corr0)

    if not lags:
        return {
            "n_windows": 0,
            "mean_abs_lag_s": float("nan"),
            "rms_lag_s": float("nan"),
            "mean_lag_s": float("nan"),
            "mean_best_corr": float("nan"),
            "mean_corr0": float("nan"),
            "corr0_gap": float("nan"),
        }

    lags_arr = np.asarray(lags, dtype=np.float64)
    best_corrs_arr = np.asarray(best_corrs, dtype=np.float64)
    corr0s_arr = np.asarray(corr0s, dtype=np.float64)
    return {
        "n_windows": int(len(lags)),
        "mean_abs_lag_s": float(np.mean(np.abs(lags_arr))),
        "rms_lag_s": float(np.sqrt(np.mean(lags_arr ** 2))),
        "mean_lag_s": float(np.mean(lags_arr)),
        "mean_best_corr": float(np.mean(best_corrs_arr)),
        "mean_corr0": float(np.mean(corr0s_arr)),
        "corr0_gap": float(np.mean(best_corrs_arr - corr0s_arr)),
    }


def add_phase_metrics(
    row: dict[str, Any],
    *,
    u_model: np.ndarray,
    u_teacher: np.ndarray,
    time: np.ndarray,
    n_nodes: int,
    args: argparse.Namespace,
    observation: str,
    component: str,
) -> None:
    pred_sig = signal_for_observation(u_model, n_nodes=n_nodes, component=component, observation=observation, last_k=int(args.last_k))
    teacher_sig = signal_for_observation(u_teacher, n_nodes=n_nodes, component=component, observation=observation, last_k=int(args.last_k))
    m = local_lag_metrics(
        pred_sig,
        teacher_sig,
        time=time,
        start_time=float(args.phase_start),
        end_time=args.phase_end,
        window_seconds=float(args.lag_window_seconds),
        stride_seconds=float(args.lag_stride_seconds),
        max_lag_seconds=float(args.max_lag_seconds),
    )
    prefix = f"{observation}_{component}"
    for k, v in m.items():
        row[f"{prefix}_{k}"] = v


def main() -> None:
    args = parse_args()
    cfg = build_cfg(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"

    load_file = Path(args.load_file).resolve()
    if not load_file.exists():
        raise FileNotFoundError(f"load_file not found: {load_file}")

    device = torch.device(args.device)
    dtype_core = _get_torch_dtype(args.core_dtype)
    alpha_values = parse_float_list(args.alpha_x_values)

    print()
    print("=" * 100)
    print("[Scan] Constant alpha_x phase sensitivity")
    print("=" * 100)
    print(f"  load_file  = {load_file}")
    print(f"  output_dir = {output_dir}")
    print(f"  alpha_x_values = {alpha_values}")
    print(f"  max_steps = {args.max_steps}")
    print(f"  phase window = [{args.phase_start}, {args.phase_end if args.phase_end is not None else 'end'}]")
    print(f"  device/core_dtype = {device}/{args.core_dtype}")

    # 1. Teacher
    print("\n[1/5] Run BeamDyn teacher")
    time_teacher_raw, u_teacher_raw, _, _ = run_teacher_response(
        cfg=cfg,
        load_file=load_file,
        output_dir=output_dir,
        case_name=f"{args.case_name}_teacher",
    )

    # 2. Static baseline to get time/load/state
    print("\n[2/5] Run static kappa-y baseline")
    static_scale = run_base_student_response(
        cfg=cfg,
        load_file=load_file,
        output_dir=output_dir / "base_static_kappa_y",
        case_name=f"{args.case_name}_base_static_kappa_y",
        use_kappa_y_global_scale=True,
        kappa_y_global_scale=float(cfg.kappa_y_static_scale),
        kappa_y_scale_mode=str(cfg.kappa_y_scale_mode),
    )

    time = np.asarray(static_scale["time"], dtype=np.float64)
    F_time = np.asarray(static_scale["F_time"], dtype=np.float64)
    u_static = np.asarray(static_scale["u_full"], dtype=np.float64)
    v_static = np.asarray(static_scale["v_full"], dtype=np.float64)
    a_static = np.asarray(static_scale["a_full"], dtype=np.float64)

    u_teacher = resample_response_to_time_grid(time_src=time_teacher_raw, u_src=u_teacher_raw, time_dst=time)
    u_teacher = np.asarray(u_teacher, dtype=np.float64)

    T_use = min(int(args.max_steps), int(time.shape[0]))
    if T_use < 2:
        raise ValueError(f"Need at least 2 time steps, got T_use={T_use}.")
    sl = slice(0, T_use)
    time = time[sl]
    F_time = F_time[sl]
    u_static = u_static[sl]
    v_static = v_static[sl]
    a_static = a_static[sl]
    u_teacher = u_teacher[sl]

    if bool(args.remove_initial_offset):
        u_teacher_cmp = remove_initial_offset(u_teacher)
        u_static_cmp = remove_initial_offset(u_static)
    else:
        u_teacher_cmp = u_teacher.copy()
        u_static_cmp = u_static.copy()

    n_dofs = int(u_static.shape[1])
    n_nodes = n_dofs // 6
    print(f"  T_use = {T_use}, t=[{time[0]:.4f}, {time[-1]:.4f}], n_nodes={n_nodes}, n_dofs={n_dofs}")

    # 3. Build alpha_x physical core
    print("\n[3/5] Build alpha_x physical core")
    registry = build_physical_parameter_registry(enabled_params="alpha_x")
    template_bundle = build_dynamic_stiffness_templates(
        PhysicalTemplateConfig(
            blade_csv=str(cfg.blade_csv),
            alpha_flap=float(cfg.alpha_flap),
            alpha_edge=float(cfg.alpha_edge),
            alpha_torsion=float(cfg.alpha_torsion),
            twist_column=str(cfg.base_phi_twist_column),
            phi_sign=float(cfg.base_phi_sign),
            rotate_mass=bool(cfg.rotate_mass),
            kappa_y_static_scale=float(cfg.kappa_y_static_scale),
            kappa_y_scale_mode=str(cfg.kappa_y_scale_mode),
            xy_template_mode="root_to_tip",
            xy_delta_phi_deg=1.0,
            enabled_params="alpha_x",
            verbose=True,
        )
    )
    M0 = np.asarray(template_bundle.M0, dtype=np.float64)
    K0 = np.asarray(template_bundle.K0, dtype=np.float64)
    freqs = _compute_natural_frequencies_hz(M0, K0, num_modes=10)
    C0, ref_freq_used = _build_structural_damping_matrix(
        K=K0,
        zeta_structural=float(cfg.zeta_structural),
        ref_freq_hz=cfg.ref_freq_hz,
        natural_freqs_hz=freqs,
    )
    core = DynamicPhysicalCoreTorch(
        M0=M0,
        K0=K0,
        C0=C0,
        stiffness_templates=template_bundle.stiffness_template_dict(),
        registry=registry,
        config=DynamicPhysicalCoreConfig(
            dt=float(cfg.dt),
            gamma=0.5,
            beta=0.25,
            dtype=dtype_core,
            linear_solve_mode="solve",
            symmetrize_k_eff=True,
        ),
    ).to(device)

    # 4. Scan
    print("\n[4/5] Scan constant alpha_x")
    rows: list[dict[str, Any]] = []
    static_metrics = compute_metrics(name="base_static_kappa_y", u=u_static_cmp, u_teacher=u_teacher_cmp, n_nodes=n_nodes)
    static_row: dict[str, Any] = {"model": "base_static_kappa_y", "alpha_x": 0.0, **static_metrics}
    for obs in ["tip", "last5"]:
        for comp in ["x", "y"]:
            add_phase_metrics(static_row, u_model=u_static_cmp, u_teacher=u_teacher_cmp, time=time, n_nodes=n_nodes, args=args, observation=obs, component=comp)
    rows.append(static_row)

    u_scan_cache: dict[float, np.ndarray] = {}
    for alpha in alpha_values:
        print(f"  alpha_x={alpha:+.6f}")
        out = run_constant_alpha_x_rollout(
            core=core,
            F_time=F_time,
            u0=u_static[0],
            v0=v_static[0],
            a0=a_static[0],
            alpha_x=float(alpha),
            device=device,
            dtype=dtype_core,
        )
        u_model = out["u_full"]
        if bool(args.remove_initial_offset):
            u_cmp = remove_initial_offset(u_model)
        else:
            u_cmp = u_model.copy()
        u_scan_cache[float(alpha)] = u_cmp

        metrics = compute_metrics(name=f"alpha_x_{alpha:+.6f}", u=u_cmp, u_teacher=u_teacher_cmp, n_nodes=n_nodes)
        row: dict[str, Any] = {"model": f"alpha_x_{alpha:+.6f}", "alpha_x": float(alpha), **metrics}

        # Ratios against static baseline.
        for key in ["full_x_mse", "full_y_mse", "tip_x_mse", "tip_y_mse", "last5_y_mse"]:
            denom = float(static_metrics.get(key, np.nan))
            row[f"{key}_ratio_to_static"] = float(metrics[key] / denom) if np.isfinite(denom) and denom > 0 else float("nan")

        for obs in ["tip", "last5"]:
            for comp in ["x", "y"]:
                add_phase_metrics(row, u_model=u_cmp, u_teacher=u_teacher_cmp, time=time, n_nodes=n_nodes, args=args, observation=obs, component=comp)
        rows.append(row)

    # 5. Save
    print("\n[5/5] Save artifacts")
    save_csv_union(output_dir / "alpha_x_constant_scan_metrics.csv", rows)
    save_json(
        output_dir / "alpha_x_constant_scan_summary.json",
        {
            "args": vars(args),
            "cfg": cfg.__dict__,
            "natural_freqs_hz": freqs.tolist(),
            "ref_freq_used": ref_freq_used,
            "static_metrics": static_metrics,
            "rows": rows,
        },
    )

    # Save compact response npz.
    if u_scan_cache:
        np.savez_compressed(
            output_dir / "alpha_x_constant_scan_responses.npz",
            time=time,
            u_teacher=u_teacher_cmp,
            u_static=u_static_cmp,
            alpha_values=np.asarray(alpha_values, dtype=np.float64),
            **{f"u_alpha_{i}": u for i, (a, u) in enumerate(u_scan_cache.items())},
        )

    if bool(args.save_plots):
        tip_x = tip_component_index(n_nodes, "x")
        tip_y = tip_component_index(n_nodes, "y")
        selected_alphas = [alpha_values[0], 0.0, alpha_values[-1]]
        for alpha in selected_alphas:
            if float(alpha) not in u_scan_cache:
                continue
            u_alpha = u_scan_cache[float(alpha)]
            safe = str(alpha).replace("-", "neg").replace("+", "pos").replace(".", "p")
            plot_timeseries(
                time=time,
                teacher=u_teacher_cmp,
                no_scale=u_static_cmp,
                static_scale=u_static_cmp,
                transformer=u_alpha,
                idx=tip_x,
                title=f"Tip x displacement: constant alpha_x={alpha:+.4f}",
                ylabel="tip ux",
                path=plots_dir / f"tip_x_alpha_{safe}.png",
            )
            plot_timeseries_error_to_teacher(
                time=time,
                teacher=u_teacher_cmp,
                no_scale=u_static_cmp,
                static_scale=u_static_cmp,
                transformer=u_alpha,
                idx=tip_x,
                title=f"Tip x error: constant alpha_x={alpha:+.4f}",
                ylabel="tip ux error",
                path=plots_dir / f"tip_x_error_alpha_{safe}.png",
            )
            plot_timeseries(
                time=time,
                teacher=u_teacher_cmp,
                no_scale=u_static_cmp,
                static_scale=u_static_cmp,
                transformer=u_alpha,
                idx=tip_y,
                title=f"Tip y displacement: constant alpha_x={alpha:+.4f}",
                ylabel="tip uy",
                path=plots_dir / f"tip_y_alpha_{safe}.png",
            )
            plot_timeseries_error_to_teacher(
                time=time,
                teacher=u_teacher_cmp,
                no_scale=u_static_cmp,
                static_scale=u_static_cmp,
                transformer=u_alpha,
                idx=tip_y,
                title=f"Tip y error: constant alpha_x={alpha:+.4f}",
                ylabel="tip uy error",
                path=plots_dir / f"tip_y_error_alpha_{safe}.png",
            )

    # Print short ranking.
    print()
    print("[Summary: sorted by tip_y mean_abs_lag_s then full_y_mse]")
    scan_rows = [r for r in rows if r["model"] != "base_static_kappa_y"]
    sort_key = lambda r: (
        float(r.get("tip_y_mean_abs_lag_s", np.inf)) if np.isfinite(float(r.get("tip_y_mean_abs_lag_s", np.inf))) else np.inf,
        float(r.get("full_y_mse", np.inf)),
    )
    for r in sorted(scan_rows, key=sort_key)[:10]:
        print(
            f"  alpha={float(r['alpha_x']):+.5f} "
            f"full_x_ratio={float(r.get('full_x_mse_ratio_to_static', np.nan)):.6f} "
            f"full_y_ratio={float(r.get('full_y_mse_ratio_to_static', np.nan)):.6f} "
            f"tip_x_lag={float(r.get('tip_x_mean_abs_lag_s', np.nan)):.4f}s "
            f"tip_y_lag={float(r.get('tip_y_mean_abs_lag_s', np.nan)):.4f}s "
            f"tip_x_corr0={float(r.get('tip_x_mean_corr0', np.nan)):.4f} "
            f"tip_y_corr0={float(r.get('tip_y_mean_corr0', np.nan)):.4f}"
        )
    print(f"\nmetrics_csv = {output_dir / 'alpha_x_constant_scan_metrics.csv'}")
    print(f"summary_json = {output_dir / 'alpha_x_constant_scan_summary.json'}")
    print("✅ PASS: constant alpha_x scan completed.")


if __name__ == "__main__":
    main()
