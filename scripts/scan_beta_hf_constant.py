from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.evaluate_transformer_vs_baselines import (  # noqa: E402
    compute_alpha_beta_diagnostics,
    compute_metrics,
    load_transformer_model,
    remove_initial_offset,
    run_base_student_response,
    run_teacher_response,
    save_csv,
    save_json,
)
from scripts.train_transformer_physical_params_torch import (  # noqa: E402
    TransformerPhysicalTrainConfig,
    _get_torch_dtype,
    build_training_model,
)
from src.student.transformer.blade_geometry_features import (  # noqa: E402
    BladeGeometryFeatureConfig,
    build_blade_geometry_features,
)
from src.teacher.beamdyn_teacher_adapter import resample_response_to_time_grid  # noqa: E402
from src.student.transformer.physical_parameter_registry import build_physical_parameter_registry  # noqa: E402


def parse_float_list(text: str) -> list[float]:
    values = []
    for item in str(text).replace(";", ",").split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError("empty float list")
    return values


def cfg_from_checkpoint(ckpt: dict[str, Any], args: argparse.Namespace) -> TransformerPhysicalTrainConfig:
    cfg_dict = asdict(TransformerPhysicalTrainConfig())
    cfg_dict.update(ckpt.get("cfg", {}) or {})
    for attr in [
        "teacher_exe",
        "template_inp",
        "blade_csv",
        "dt",
        "t_initial",
        "t_final",
        "teacher_node_start",
        "teacher_node_end",
        "kappa_y_static_scale",
        "kappa_y_scale_mode",
    ]:
        value = getattr(args, attr, None)
        if value is not None:
            cfg_dict[attr] = value
    cfg_dict["teacher_demean"] = bool(args.teacher_demean)
    cfg_dict["device"] = str(args.device)
    valid = {f.name for f in fields(TransformerPhysicalTrainConfig)}
    return TransformerPhysicalTrainConfig(**{k: v for k, v in cfg_dict.items() if k in valid})


def beta_theta_sequence(
    *,
    alpha_theta_dict: dict[str, np.ndarray],
    registry: Any,
    beta_x: float,
    beta_y: float,
) -> np.ndarray:
    first = next(iter(alpha_theta_dict.values()))
    T = int(np.asarray(first).shape[0])
    theta = np.zeros((1, T, int(registry.total_dim)), dtype=np.float64)
    for name in ("alpha_x", "alpha_xy"):
        if name in registry.slices and name in alpha_theta_dict:
            sl = registry.slices[name]
            theta[0, :, sl] = np.asarray(alpha_theta_dict[name], dtype=np.float64).reshape(T, -1)
    for name, value in (("beta_damp_hf_x", beta_x), ("beta_damp_hf_y", beta_y)):
        if name in registry.slices:
            sl = registry.slices[name]
            theta[0, :, sl] = float(value)
    return theta


def add_static_ratios(rows: list[dict[str, Any]]) -> None:
    static = next(r for r in rows if r["model"] == "base_static_kappa_y")
    for row in rows:
        for key in ["full_x_mse", "full_y_mse", "tip_y_mse", "last5_y_mse"]:
            denom = float(static[key])
            row[f"{key}_ratio_to_static"] = float(row[key]) / denom if denom > 0.0 else np.nan


def main() -> None:
    d = TransformerPhysicalTrainConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Scan constant beta_damp_hf_x/y on top of alpha checkpoint theta(t). "
            "This tests the upper bound of the high-frequency damping channel without training."
        )
    )
    parser.add_argument("--alpha-checkpoint", required=True)
    parser.add_argument("--load-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--case-name", default="beta_hf_constant_scan")
    parser.add_argument("--beta-x-values", default="0.0,0.02,0.05,0.10,0.20,0.40")
    parser.add_argument("--beta-y-values", default="0.0,0.02,0.05,0.10,0.20,0.40")
    parser.add_argument("--beta-hf-damp-scale-x", type=float, default=0.20)
    parser.add_argument("--beta-hf-damp-scale-y", type=float, default=0.15)
    parser.add_argument("--max-steps", type=int, default=1001)
    parser.add_argument("--device", default=d.device)
    parser.add_argument("--teacher-exe", default=None)
    parser.add_argument("--template-inp", default=None)
    parser.add_argument("--blade-csv", default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--t-initial", type=float, default=None)
    parser.add_argument("--t-final", type=float, default=None)
    parser.add_argument("--teacher-node-start", type=int, default=None)
    parser.add_argument("--teacher-node-end", type=int, default=None)
    parser.add_argument("--teacher-demean", action="store_true", default=False)
    parser.add_argument("--kappa-y-static-scale", type=float, default=None)
    parser.add_argument("--kappa-y-scale-mode", default=None)
    parser.add_argument("--remove-initial-offset", action="store_true", default=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    load_file = Path(args.load_file)

    alpha_ckpt = torch.load(args.alpha_checkpoint, map_location=device)
    alpha_cfg = cfg_from_checkpoint(alpha_ckpt, args)
    beta_cfg = replace(
        alpha_cfg,
        enabled_params="alpha_x,alpha_xy,beta_damp_hf_x,beta_damp_hf_y",
        beta_hf_damp_scale_x=float(args.beta_hf_damp_scale_x),
        beta_hf_damp_scale_y=float(args.beta_hf_damp_scale_y),
        output_dir=str(output_dir),
    )

    print("[1/5] teacher/static responses")
    time_teacher, u_teacher_raw, _cols, teacher_out = run_teacher_response(
        cfg=alpha_cfg,
        load_file=load_file,
        output_dir=output_dir,
        case_name=args.case_name,
    )
    static_result = run_base_student_response(
        cfg=alpha_cfg,
        load_file=load_file,
        output_dir=output_dir / "base_static_kappa_y",
        case_name=args.case_name,
        use_kappa_y_global_scale=True,
        kappa_y_global_scale=float(alpha_cfg.kappa_y_static_scale),
        kappa_y_scale_mode=str(alpha_cfg.kappa_y_scale_mode),
    )
    time = static_result["time"]
    T = min(int(args.max_steps), int(time.shape[0]))
    time = time[:T]
    u_teacher = resample_response_to_time_grid(time_teacher, u_teacher_raw, time)
    u_static = static_result["u_full"][:T]
    v_static = static_result["v_full"][:T]
    a_static = static_result["a_full"][:T]
    F_time = static_result["F_time"][:T]
    if bool(args.remove_initial_offset):
        u_teacher = remove_initial_offset(u_teacher)
        u_static = remove_initial_offset(u_static)

    print("[2/5] alpha checkpoint rollout")
    geo_bundle = build_blade_geometry_features(
        BladeGeometryFeatureConfig(
            blade_csv=str(alpha_cfg.blade_csv),
            twist_column=str(alpha_cfg.base_phi_twist_column),
            phi_sign=float(alpha_cfg.base_phi_sign),
            exclude_root_station=True,
        )
    )
    geometry = torch.as_tensor(
        geo_bundle.features,
        dtype=_get_torch_dtype(alpha_cfg.encoder_dtype),
        device=device,
    )
    alpha_model = load_transformer_model(
        cfg=alpha_cfg,
        ckpt=alpha_ckpt,
        geometry_dim=int(geo_bundle.feature_dim),
        device=device,
    )
    from scripts.evaluate_transformer_vs_baselines import run_transformer_response

    alpha_result = run_transformer_response(
        model=alpha_model,
        cfg=alpha_cfg,
        geometry=geometry,
        F_time=F_time,
        u_static=u_static,
        v_static=v_static,
        a_static=a_static,
        device=device,
    )
    u_alpha = alpha_result["u_full"]

    print("[3/5] beta high-frequency core")
    registry = build_physical_parameter_registry(enabled_params=beta_cfg.enabled_params)
    beta_model, _metadata = build_training_model(
        cfg=beta_cfg,
        registry=registry,
        geometry_dim=int(geo_bundle.feature_dim),
        dtype_core=_get_torch_dtype(beta_cfg.core_dtype),
        device=device,
    )
    beta_model.eval()

    F = torch.as_tensor(F_time, dtype=_get_torch_dtype(beta_cfg.core_dtype), device=device).unsqueeze(0)
    u_static_t = torch.as_tensor(u_static, dtype=_get_torch_dtype(beta_cfg.core_dtype), device=device).unsqueeze(0)
    v_static_t = torch.as_tensor(v_static, dtype=_get_torch_dtype(beta_cfg.core_dtype), device=device).unsqueeze(0)
    a_static_t = torch.as_tensor(a_static, dtype=_get_torch_dtype(beta_cfg.core_dtype), device=device).unsqueeze(0)
    u0 = u_static_t[:, 0, :]
    v0 = v_static_t[:, 0, :]
    a0 = a_static_t[:, 0, :]

    print("[4/5] scan constant beta_hf values")
    rows: list[dict[str, Any]] = [
        compute_metrics(
            name="base_static_kappa_y",
            u=u_static,
            u_teacher=u_teacher,
            n_nodes=u_teacher.shape[1] // 6,
        ),
        compute_metrics(
            name="alpha_only",
            u=u_alpha,
            u_teacher=u_teacher,
            n_nodes=u_teacher.shape[1] // 6,
        ),
    ]
    response_payload: dict[str, np.ndarray] = {
        "u_teacher": u_teacher,
        "u_static": u_static,
        "u_alpha_only": u_alpha,
    }
    beta_x_values = parse_float_list(args.beta_x_values)
    beta_y_values = parse_float_list(args.beta_y_values)
    for beta_x in beta_x_values:
        for beta_y in beta_y_values:
            theta_np = beta_theta_sequence(
                alpha_theta_dict=alpha_result["theta_dict"],
                registry=registry,
                beta_x=float(beta_x),
                beta_y=float(beta_y),
            )
            theta = torch.as_tensor(theta_np, dtype=_get_torch_dtype(beta_cfg.core_dtype), device=device)
            with torch.no_grad():
                u_pred, _v_pred, _a_pred = beta_model.rollout_with_theta_sequence(
                    theta_seq=theta,
                    u_static=u_static_t,
                    v_static=v_static_t,
                    a_static=a_static_t,
                    F=F,
                    u0=u0,
                    v0=v0,
                    a0=a0,
                )
            u_beta = u_pred[0].detach().cpu().numpy()
            model_name = f"beta_hf_x_{beta_x:+.4f}_y_{beta_y:+.4f}"
            row = compute_metrics(
                name=model_name,
                u=u_beta,
                u_teacher=u_teacher,
                n_nodes=u_teacher.shape[1] // 6,
            )
            row["beta_hf_x"] = float(beta_x)
            row["beta_hf_y"] = float(beta_y)
            row.update(
                compute_alpha_beta_diagnostics(
                    time=time,
                    u_alpha=u_alpha,
                    u_beta=u_beta,
                    u_teacher=u_teacher,
                    n_nodes=u_teacher.shape[1] // 6,
                    beta_theta_dict={
                        "beta_damp_hf_x": np.full((T, 1), float(beta_x)),
                        "beta_damp_hf_y": np.full((T, 1), float(beta_y)),
                    },
                )
            )
            rows.append(row)
            response_payload[f"u_scan_x_{beta_x:+.4f}_y_{beta_y:+.4f}"] = u_beta

    add_static_ratios(rows)
    scan_rows = [r for r in rows if str(r["model"]).startswith("beta_hf_")]
    best_x = min(scan_rows, key=lambda r: float(r["full_x_mse_ratio_to_static"]))
    best_y = min(scan_rows, key=lambda r: float(r["full_y_mse_ratio_to_static"]))
    best_last5_y = min(scan_rows, key=lambda r: float(r["last5_y_mse_ratio_to_static"]))

    print("[5/5] save results")
    save_csv(output_dir / "beta_hf_constant_scan_metrics.csv", rows)
    save_json(
        output_dir / "beta_hf_constant_scan_summary.json",
        {
            "alpha_checkpoint": str(Path(args.alpha_checkpoint).resolve()),
            "load_file": str(load_file.resolve()),
            "teacher_out": str(teacher_out),
            "beta_hf_damp_scale_x": float(args.beta_hf_damp_scale_x),
            "beta_hf_damp_scale_y": float(args.beta_hf_damp_scale_y),
            "best_full_x": best_x,
            "best_full_y": best_y,
            "best_last5_y": best_last5_y,
            "metrics_csv": str(output_dir / "beta_hf_constant_scan_metrics.csv"),
        },
    )
    np.savez(output_dir / "beta_hf_constant_scan_responses.npz", time=time, F_time=F_time, **response_payload)
    print(f"metrics_csv = {output_dir / 'beta_hf_constant_scan_metrics.csv'}")
    print(f"summary_json = {output_dir / 'beta_hf_constant_scan_summary.json'}")
    print("best_full_x:", best_x)
    print("best_full_y:", best_y)
    print("best_last5_y:", best_last5_y)
    print("PASS")


if __name__ == "__main__":
    main()
