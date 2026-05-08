from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.run_student_cases import (  # noqa: E402
    run_student_case,
    _build_structural_damping_matrix,
)
from src.student.io import load_student_model_from_blade_master  # noqa: E402
from src.student.dynamic_solver import WindBladeDynamicSystem  # noqa: E402
from src.student.full_order_corrected_core_torch import (  # noqa: E402
    FullOrderCorrectedCoreTorch,
    FullOrderCorrectedCoreTorchConfig,
)


@dataclass
class DeltaKSanityConfig:
    blade_csv: str = str(PROJECT_ROOT / "data" / "raw" / "nrel5mw" / "blade_master.csv")
    output_dir: str = str(PROJECT_ROOT / "results" / "student" / "full_order_delta_k_sanity")
    case_name: str = "delta_k_sanity_test_complex"

    time_series_load_file: str = str(PROJECT_ROOT / "data" / "load" / "test_complex_case.dat")
    t_initial: float = 0.0
    t_final: float = 2.0
    dt: float = 0.01

    alpha_flap: float = 1.0
    alpha_edge: float = 1.0
    alpha_torsion: float = 1.0
    zeta_structural: float = 0.015
    ref_freq_hz: Optional[float] = None

    torch_dtype: str = "float64"
    device: str = "cpu"
    linear_solve_mode: str = "solve"

    # zero-correction 对齐阈值
    atol_u_mae: float = 1e-10
    atol_u_max: float = 1e-8
    atol_v_mae: float = 1e-10
    atol_v_max: float = 1e-8
    atol_a_mae: float = 1e-8
    atol_a_max: float = 1e-6

    # 非零 delta_K sanity 阈值：只要响应发生明显变化即可
    delta_k_relative_scale_x: float = 0.02
    delta_k_relative_scale_y: float = 0.02
    min_u_change_max: float = 1e-9

    save_report: bool = True


def parse_args() -> DeltaKSanityConfig:
    default = DeltaKSanityConfig()

    parser = argparse.ArgumentParser(
        description=(
            "Sanity check explicit delta_K interface in "
            "FullOrderCorrectedCoreTorch."
        )
    )

    parser.add_argument("--blade-csv", type=str, default=default.blade_csv)
    parser.add_argument("--output-dir", type=str, default=default.output_dir)
    parser.add_argument("--case-name", type=str, default=default.case_name)
    parser.add_argument("--time-series-load-file", type=str, default=default.time_series_load_file)

    parser.add_argument("--t-initial", type=float, default=default.t_initial)
    parser.add_argument("--t-final", type=float, default=default.t_final)
    parser.add_argument("--dt", type=float, default=default.dt)

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

    parser.add_argument("--atol-u-mae", type=float, default=default.atol_u_mae)
    parser.add_argument("--atol-u-max", type=float, default=default.atol_u_max)
    parser.add_argument("--atol-v-mae", type=float, default=default.atol_v_mae)
    parser.add_argument("--atol-v-max", type=float, default=default.atol_v_max)
    parser.add_argument("--atol-a-mae", type=float, default=default.atol_a_mae)
    parser.add_argument("--atol-a-max", type=float, default=default.atol_a_max)

    parser.add_argument(
        "--delta-k-relative-scale-x",
        type=float,
        default=default.delta_k_relative_scale_x,
    )
    parser.add_argument(
        "--delta-k-relative-scale-y",
        type=float,
        default=default.delta_k_relative_scale_y,
    )
    parser.add_argument("--min-u-change-max", type=float, default=default.min_u_change_max)

    save_group = parser.add_mutually_exclusive_group()
    save_group.add_argument("--save-report", dest="save_report", action="store_true")
    save_group.add_argument("--no-save-report", dest="save_report", action="store_false")
    parser.set_defaults(save_report=default.save_report)

    args = parser.parse_args()

    return DeltaKSanityConfig(
        blade_csv=args.blade_csv,
        output_dir=args.output_dir,
        case_name=args.case_name,
        time_series_load_file=args.time_series_load_file,
        t_initial=args.t_initial,
        t_final=args.t_final,
        dt=args.dt,
        alpha_flap=args.alpha_flap,
        alpha_edge=args.alpha_edge,
        alpha_torsion=args.alpha_torsion,
        zeta_structural=args.zeta_structural,
        ref_freq_hz=args.ref_freq_hz,
        torch_dtype=args.torch_dtype,
        device=args.device,
        linear_solve_mode=args.linear_solve_mode,
        atol_u_mae=args.atol_u_mae,
        atol_u_max=args.atol_u_max,
        atol_v_mae=args.atol_v_mae,
        atol_v_max=args.atol_v_max,
        atol_a_mae=args.atol_a_mae,
        atol_a_max=args.atol_a_max,
        delta_k_relative_scale_x=args.delta_k_relative_scale_x,
        delta_k_relative_scale_y=args.delta_k_relative_scale_y,
        min_u_change_max=args.min_u_change_max,
        save_report=args.save_report,
    )


def get_torch_dtype(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def build_student_mkc_like_run_student_case(
    *,
    blade_csv: str | Path,
    alpha_flap: float,
    alpha_edge: float,
    alpha_torsion: float,
    zeta_structural: float,
    ref_freq_hz: Optional[float],
) -> Dict[str, Any]:
    student_model = load_student_model_from_blade_master(
        str(Path(blade_csv).resolve()),
        model_name="student_delta_k_sanity",
    )

    blade_sys = WindBladeDynamicSystem(
        student_model,
        alpha_flap=float(alpha_flap),
        alpha_edge=float(alpha_edge),
        alpha_torsion=float(alpha_torsion),
    )

    M = np.asarray(blade_sys.M, dtype=np.float64)
    K = np.asarray(blade_sys.K, dtype=np.float64)

    natural_freqs = np.asarray(
        blade_sys.get_natural_frequencies(num_modes=10),
        dtype=np.float64,
    )

    C, ref_freq_used = _build_structural_damping_matrix(
        K=K,
        zeta_structural=float(zeta_structural),
        ref_freq_hz=ref_freq_hz,
        natural_freqs=natural_freqs,
    )

    return {
        "student_model": student_model,
        "M": M,
        "K": K,
        "C": np.asarray(C, dtype=np.float64),
        "natural_freqs_hz": natural_freqs,
        "ref_freq_used": ref_freq_used,
        "n_dofs_full": int(M.shape[0]),
        "n_free_nodes": int(len(student_model.eta) - 1),
    }


def as_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def abs_stats(pred: np.ndarray, ref: np.ndarray) -> Tuple[float, float]:
    diff = np.asarray(pred) - np.asarray(ref)
    return float(np.mean(np.abs(diff))), float(np.max(np.abs(diff)))


def component_field(x: np.ndarray, component: str) -> np.ndarray:
    comp_to_idx = {
        "x": 0,
        "y": 1,
        "z": 2,
        "rx": 3,
        "ry": 4,
        "rz": 5,
    }
    if component not in comp_to_idx:
        raise ValueError(f"Unsupported component: {component}")
    return x[..., comp_to_idx[component]::6]


def compute_zero_correction_metrics(
    *,
    u_zero: np.ndarray,
    v_zero: np.ndarray,
    a_zero: np.ndarray,
    u_ref: np.ndarray,
    v_ref: np.ndarray,
    a_ref: np.ndarray,
) -> Dict[str, float]:
    u_mae, u_max = abs_stats(u_zero, u_ref)
    v_mae, v_max = abs_stats(v_zero, v_ref)
    a_mae, a_max = abs_stats(a_zero, a_ref)

    return {
        "u_mae": u_mae,
        "u_max_abs": u_max,
        "v_mae": v_mae,
        "v_max_abs": v_max,
        "a_mae": a_mae,
        "a_max_abs": a_max,
    }


def compute_active_delta_k_metrics(
    *,
    u_corr: np.ndarray,
    v_corr: np.ndarray,
    a_corr: np.ndarray,
    u_zero: np.ndarray,
    v_zero: np.ndarray,
    a_zero: np.ndarray,
) -> Dict[str, float]:
    u_change_mae, u_change_max = abs_stats(u_corr, u_zero)
    v_change_mae, v_change_max = abs_stats(v_corr, v_zero)
    a_change_mae, a_change_max = abs_stats(a_corr, a_zero)

    full_x_change_mae, full_x_change_max = abs_stats(
        component_field(u_corr, "x"),
        component_field(u_zero, "x"),
    )
    full_y_change_mae, full_y_change_max = abs_stats(
        component_field(u_corr, "y"),
        component_field(u_zero, "y"),
    )

    tip_x_change_mae, tip_x_change_max = abs_stats(
        component_field(u_corr, "x")[..., -1],
        component_field(u_zero, "x")[..., -1],
    )
    tip_y_change_mae, tip_y_change_max = abs_stats(
        component_field(u_corr, "y")[..., -1],
        component_field(u_zero, "y")[..., -1],
    )

    return {
        "u_change_mae": u_change_mae,
        "u_change_max_abs": u_change_max,
        "v_change_mae": v_change_mae,
        "v_change_max_abs": v_change_max,
        "a_change_mae": a_change_mae,
        "a_change_max_abs": a_change_max,
        "full_x_change_mae": full_x_change_mae,
        "full_x_change_max_abs": full_x_change_max,
        "full_y_change_mae": full_y_change_mae,
        "full_y_change_max_abs": full_y_change_max,
        "tip_x_change_mae": tip_x_change_mae,
        "tip_x_change_max_abs": tip_x_change_max,
        "tip_y_change_mae": tip_y_change_mae,
        "tip_y_change_max_abs": tip_y_change_max,
    }


def check_zero_pass(metrics: Dict[str, float], cfg: DeltaKSanityConfig) -> Tuple[bool, Dict[str, bool]]:
    checks = {
        "u_mae": metrics["u_mae"] <= cfg.atol_u_mae,
        "u_max_abs": metrics["u_max_abs"] <= cfg.atol_u_max,
        "v_mae": metrics["v_mae"] <= cfg.atol_v_mae,
        "v_max_abs": metrics["v_max_abs"] <= cfg.atol_v_max,
        "a_mae": metrics["a_mae"] <= cfg.atol_a_mae,
        "a_max_abs": metrics["a_max_abs"] <= cfg.atol_a_max,
    }
    return all(checks.values()), checks


def check_active_delta_k_pass(
    metrics: Dict[str, float],
    cfg: DeltaKSanityConfig,
) -> Tuple[bool, Dict[str, bool]]:
    checks = {
        "u_change_max_abs_is_nonzero": metrics["u_change_max_abs"] > cfg.min_u_change_max,
        "u_change_is_finite": np.isfinite(metrics["u_change_max_abs"]),
        "v_change_is_finite": np.isfinite(metrics["v_change_max_abs"]),
        "a_change_is_finite": np.isfinite(metrics["a_change_max_abs"]),
    }
    return all(checks.values()), checks


def build_delta_k_relative_diag(
    *,
    n_dofs_full: int,
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    """
    构造一个只用于 sanity check 的相对对角刚度修正。

    每个自由节点 6DOF:
        [ux, uy, uz, theta_x, theta_y, theta_z]

    这里仅对 ux / uy 对应的对角刚度施加小比例修正。
    这不是最终物理参数化，只用于验证 delta_K 是否进入 PDE。
    """
    rel = np.zeros(n_dofs_full, dtype=np.float64)
    rel[0::6] = float(scale_x)
    rel[1::6] = float(scale_y)
    return rel


def print_named_metrics(title: str, metrics: Dict[str, float], checks: Optional[Dict[str, bool]] = None) -> None:
    print()
    print(title)
    for k, v in metrics.items():
        flag = ""
        if checks is not None and k in checks:
            flag = "  PASS" if checks[k] else "  FAIL"
        print(f"  {k:<30s}: {v:.12e}{flag}")


def print_checks(title: str, checks: Dict[str, bool]) -> None:
    print()
    print(title)
    for k, v in checks.items():
        print(f"  {k:<30s}: {'PASS' if v else 'FAIL'}")


def make_json_safe(obj: Any) -> Any:
    """
    将 numpy / torch / Path 等对象递归转换为 json.dump 可序列化对象。

    主要解决：
    - np.bool_  -> bool
    - np.integer -> int
    - np.floating -> float
    - np.ndarray -> list
    - torch.Tensor -> list / scalar
    - Path -> str
    """
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


def main() -> None:
    cfg = parse_args()

    blade_csv = Path(cfg.blade_csv).resolve()
    output_dir = Path(cfg.output_dir).resolve()
    load_file = Path(cfg.time_series_load_file).resolve()

    if not blade_csv.exists():
        raise FileNotFoundError(f"blade_csv not found: {blade_csv}")
    if not load_file.exists():
        raise FileNotFoundError(f"time_series_load_file not found: {load_file}")

    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = get_torch_dtype(cfg.torch_dtype)
    device = torch.device(cfg.device)

    student_params = {
        "alpha_flap": cfg.alpha_flap,
        "alpha_edge": cfg.alpha_edge,
        "alpha_torsion": cfg.alpha_torsion,
        "zeta_structural": cfg.zeta_structural,
        "ref_freq_hz": cfg.ref_freq_hz,
    }

    print()
    print("[DeltaK Sanity Config]")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")

    print()
    print("[1/5] Running direct student baseline")
    baseline = run_student_case(
        blade_csv=blade_csv,
        output_dir=output_dir / "direct_student",
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

    F_time = np.asarray(baseline["F_time"], dtype=np.float64)
    u_ref = np.asarray(baseline["u_full"], dtype=np.float64)
    v_ref = np.asarray(baseline["v_full"], dtype=np.float64)
    a_ref = np.asarray(baseline["a_full"], dtype=np.float64)

    print(f"  F_time shape = {F_time.shape}")
    print(f"  u_ref shape  = {u_ref.shape}")

    print()
    print("[2/5] Rebuilding full-order M/K/C")
    mkc = build_student_mkc_like_run_student_case(
        blade_csv=blade_csv,
        alpha_flap=cfg.alpha_flap,
        alpha_edge=cfg.alpha_edge,
        alpha_torsion=cfg.alpha_torsion,
        zeta_structural=cfg.zeta_structural,
        ref_freq_hz=cfg.ref_freq_hz,
    )

    M = mkc["M"]
    K = mkc["K"]
    C = mkc["C"]

    print(f"  M shape = {M.shape}")
    print(f"  K shape = {K.shape}")
    print(f"  C shape = {C.shape}")
    print(f"  natural_freqs[:5] = {mkc['natural_freqs_hz'][:5]}")
    print(f"  ref_freq_used = {mkc['ref_freq_used']}")

    print()
    print("[3/5] Initializing FullOrderCorrectedCoreTorch")
    core = FullOrderCorrectedCoreTorch(
        M=M,
        K=K,
        C=C,
        dt=cfg.dt,
        config=FullOrderCorrectedCoreTorchConfig(
            gamma=0.5,
            beta=0.25,
            dtype=dtype,
            linear_solve_mode=cfg.linear_solve_mode,
        ),
    ).to(device)

    for k, v in core.summary().items():
        print(f"  {k}: {v}")

    F_torch = torch.as_tensor(F_time, dtype=dtype, device=device)
    u0_torch = torch.as_tensor(u_ref[0], dtype=dtype, device=device)
    v0_torch = torch.as_tensor(v_ref[0], dtype=dtype, device=device)

    print()
    print("[4/5] Running zero-correction rollout")
    with torch.no_grad():
        u_zero_t, v_zero_t, a_zero_t = core.rollout(
            F_time=F_torch,
            u0=u0_torch,
            v0=v0_torch,
            theta_full=None,
            return_debug=False,
        )

    u_zero = as_numpy(u_zero_t)
    v_zero = as_numpy(v_zero_t)
    a_zero = as_numpy(a_zero_t)

    zero_metrics = compute_zero_correction_metrics(
        u_zero=u_zero,
        v_zero=v_zero,
        a_zero=a_zero,
        u_ref=u_ref,
        v_ref=v_ref,
        a_ref=a_ref,
    )
    zero_passed, zero_checks = check_zero_pass(zero_metrics, cfg)

    print_named_metrics("[Zero-Correction Metrics]", zero_metrics, zero_checks)

    print()
    print("[5/5] Running active delta_K rollout")

    rel_diag = build_delta_k_relative_diag(
        n_dofs_full=F_time.shape[1],
        scale_x=cfg.delta_k_relative_scale_x,
        scale_y=cfg.delta_k_relative_scale_y,
    )

    theta_full = {
        "delta_K_relative_diag": rel_diag,
    }

    with torch.no_grad():
        u_corr_t, v_corr_t, a_corr_t = core.rollout(
            F_time=F_torch,
            u0=u0_torch,
            v0=v0_torch,
            theta_full=theta_full,
            return_debug=False,
        )

    u_corr = as_numpy(u_corr_t)
    v_corr = as_numpy(v_corr_t)
    a_corr = as_numpy(a_corr_t)

    active_metrics = compute_active_delta_k_metrics(
        u_corr=u_corr,
        v_corr=v_corr,
        a_corr=a_corr,
        u_zero=u_zero,
        v_zero=v_zero,
        a_zero=a_zero,
    )
    active_passed, active_checks = check_active_delta_k_pass(active_metrics, cfg)

    print_named_metrics("[Active DeltaK Change Metrics]", active_metrics)
    print_checks("[Active DeltaK Checks]", active_checks)

    all_passed = bool(zero_passed and active_passed)

    print()
    if all_passed:
        print("✅ PASS: delta_K interface sanity check passed.")
        print("   zero correction still matches baseline, and nonzero delta_K changes response.")
    else:
        print("❌ FAIL: delta_K interface sanity check failed.")
        if not zero_passed:
            print("   zero-correction baseline alignment failed.")
        if not active_passed:
            print("   nonzero delta_K did not produce a detectable response change.")

    report = {
        "passed": all_passed,
        "zero_passed": bool(zero_passed),
        "active_passed": bool(active_passed),
        "zero_checks": zero_checks,
        "active_checks": active_checks,
        "zero_metrics": zero_metrics,
        "active_metrics": active_metrics,
        "config": asdict(cfg),
        "baseline_npz": str(baseline["npz"]),
        "baseline_yaml": str(baseline["yaml"]),
        "core_summary": core.summary(),
        "mkc_summary": {
            "M_shape": list(M.shape),
            "K_shape": list(K.shape),
            "C_shape": list(C.shape),
            "natural_freqs_hz": mkc["natural_freqs_hz"].tolist(),
            "ref_freq_used": mkc["ref_freq_used"],
            "n_dofs_full": mkc["n_dofs_full"],
            "n_free_nodes": mkc["n_free_nodes"],
        },
    }

    if cfg.save_report:
        report_path = output_dir / f"{cfg.case_name}_delta_k_sanity_report.json"
        save_json(report_path, report)
        print()
        print(f"[Saved Report] {report_path}")

    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()