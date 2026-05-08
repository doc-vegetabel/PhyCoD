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
class CompareConfig:
    blade_csv: str = str(PROJECT_ROOT / "data" / "raw" / "nrel5mw" / "blade_master.csv")
    output_dir: str = str(PROJECT_ROOT / "results" / "student" / "full_order_core_compare")
    case_name: str = "compare_full_order_zero_correction"

    use_time_series_load: bool = True
    time_series_load_file: str = str(PROJECT_ROOT / "data" / "load" / "test_complex_case.dat")

    t_initial: float = 0.0
    t_final: float = 10.0
    dt: float = 0.01

    alpha_flap: float = 1.0
    alpha_edge: float = 1.0
    alpha_torsion: float = 1.0
    zeta_structural: float = 0.015
    ref_freq_hz: Optional[float] = None

    torch_dtype: str = "float64"
    device: str = "cpu"
    linear_solve_mode: str = "solve"  # "solve" or "inverse"

    atol_u_mae: float = 1e-10
    atol_u_max: float = 1e-8
    atol_v_mae: float = 1e-10
    atol_v_max: float = 1e-8
    atol_a_mae: float = 1e-8
    atol_a_max: float = 1e-6

    save_report: bool = True


def parse_args() -> CompareConfig:
    default = CompareConfig()

    parser = argparse.ArgumentParser(
        description=(
            "Compare FullOrderCorrectedCoreTorch(theta=0) against "
            "direct run_student_case(...) baseline."
        )
    )

    parser.add_argument("--blade-csv", type=str, default=default.blade_csv)
    parser.add_argument("--output-dir", type=str, default=default.output_dir)
    parser.add_argument("--case-name", type=str, default=default.case_name)

    load_group = parser.add_mutually_exclusive_group()
    load_group.add_argument(
        "--use-time-series-load",
        dest="use_time_series_load",
        action="store_true",
        help="Use teacher-style multi-point time-series load file.",
    )
    load_group.add_argument(
        "--no-time-series-load",
        dest="use_time_series_load",
        action="store_false",
        help="Use zero load.",
    )
    parser.set_defaults(use_time_series_load=default.use_time_series_load)

    parser.add_argument(
        "--time-series-load-file",
        type=str,
        default=default.time_series_load_file,
    )

    parser.add_argument("--t-initial", type=float, default=default.t_initial)
    parser.add_argument("--t-final", type=float, default=default.t_final)
    parser.add_argument("--dt", type=float, default=default.dt)

    parser.add_argument("--alpha-flap", type=float, default=default.alpha_flap)
    parser.add_argument("--alpha-edge", type=float, default=default.alpha_edge)
    parser.add_argument("--alpha-torsion", type=float, default=default.alpha_torsion)
    parser.add_argument("--zeta-structural", type=float, default=default.zeta_structural)
    parser.add_argument(
        "--ref-freq-hz",
        type=float,
        default=default.ref_freq_hz,
        help="If omitted, use first natural frequency, same as run_student_case(...).",
    )

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

    save_group = parser.add_mutually_exclusive_group()
    save_group.add_argument("--save-report", dest="save_report", action="store_true")
    save_group.add_argument("--no-save-report", dest="save_report", action="store_false")
    parser.set_defaults(save_report=default.save_report)

    args = parser.parse_args()
    return CompareConfig(
        blade_csv=args.blade_csv,
        output_dir=args.output_dir,
        case_name=args.case_name,
        use_time_series_load=args.use_time_series_load,
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
    dt: float,
    alpha_flap: float,
    alpha_edge: float,
    alpha_torsion: float,
    zeta_structural: float,
    ref_freq_hz: Optional[float],
) -> Dict[str, Any]:
    """
    使用与 run_student_case(...) 一致的逻辑重建 M/K/C。

    这样 compare 的 core 初始化不会经过 NominalStudentCore / ModalAdapter，
    但仍能保证 M/K/C 与 direct student baseline 一致。
    """
    _ = dt

    student_model = load_student_model_from_blade_master(
        str(Path(blade_csv).resolve()),
        model_name="student_full_order_compare",
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
    idx = comp_to_idx[component]
    return x[..., idx::6]


def last_k_component(x: np.ndarray, component: str, k: int = 5) -> np.ndarray:
    field = component_field(x, component)
    if field.shape[-1] <= k:
        return field
    return field[..., -k:]


def compute_compare_metrics(
    *,
    u_pred: np.ndarray,
    v_pred: np.ndarray,
    a_pred: np.ndarray,
    u_ref: np.ndarray,
    v_ref: np.ndarray,
    a_ref: np.ndarray,
) -> Dict[str, float]:
    u_mae, u_max = abs_stats(u_pred, u_ref)
    v_mae, v_max = abs_stats(v_pred, v_ref)
    a_mae, a_max = abs_stats(a_pred, a_ref)

    full_x_mae, full_x_max = abs_stats(
        component_field(u_pred, "x"),
        component_field(u_ref, "x"),
    )
    full_y_mae, full_y_max = abs_stats(
        component_field(u_pred, "y"),
        component_field(u_ref, "y"),
    )

    tip_x_mae, tip_x_max = abs_stats(
        component_field(u_pred, "x")[..., -1],
        component_field(u_ref, "x")[..., -1],
    )
    tip_y_mae, tip_y_max = abs_stats(
        component_field(u_pred, "y")[..., -1],
        component_field(u_ref, "y")[..., -1],
    )

    last5_y_mae, last5_y_max = abs_stats(
        last_k_component(u_pred, "y", k=5),
        last_k_component(u_ref, "y", k=5),
    )

    return {
        "u_mae": u_mae,
        "u_max_abs": u_max,
        "v_mae": v_mae,
        "v_max_abs": v_max,
        "a_mae": a_mae,
        "a_max_abs": a_max,
        "full_x_mae": full_x_mae,
        "full_x_max_abs": full_x_max,
        "full_y_mae": full_y_mae,
        "full_y_max_abs": full_y_max,
        "tip_x_mae": tip_x_mae,
        "tip_x_max_abs": tip_x_max,
        "tip_y_mae": tip_y_mae,
        "tip_y_max_abs": tip_y_max,
        "last5_y_mae": last5_y_mae,
        "last5_y_max_abs": last5_y_max,
    }


def check_pass(metrics: Dict[str, float], cfg: CompareConfig) -> Tuple[bool, Dict[str, bool]]:
    checks = {
        "u_mae": metrics["u_mae"] <= cfg.atol_u_mae,
        "u_max_abs": metrics["u_max_abs"] <= cfg.atol_u_max,
        "v_mae": metrics["v_mae"] <= cfg.atol_v_mae,
        "v_max_abs": metrics["v_max_abs"] <= cfg.atol_v_max,
        "a_mae": metrics["a_mae"] <= cfg.atol_a_mae,
        "a_max_abs": metrics["a_max_abs"] <= cfg.atol_a_max,
    }
    return all(checks.values()), checks


def print_metrics(metrics: Dict[str, float], checks: Dict[str, bool]) -> None:
    print()
    print("[Zero-Correction Compare Metrics]")
    ordered_keys = [
        "u_mae",
        "u_max_abs",
        "v_mae",
        "v_max_abs",
        "a_mae",
        "a_max_abs",
        "full_x_mae",
        "full_x_max_abs",
        "full_y_mae",
        "full_y_max_abs",
        "tip_x_mae",
        "tip_x_max_abs",
        "tip_y_mae",
        "tip_y_max_abs",
        "last5_y_mae",
        "last5_y_max_abs",
    ]

    for k in ordered_keys:
        flag = ""
        if k in checks:
            flag = "  PASS" if checks[k] else "  FAIL"
        print(f"  {k:<18s}: {metrics[k]:.12e}{flag}")


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main() -> None:
    cfg = parse_args()

    blade_csv = Path(cfg.blade_csv).resolve()
    output_dir = Path(cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not blade_csv.exists():
        raise FileNotFoundError(f"blade_csv not found: {blade_csv}")

    time_series_load_file: Optional[Path]
    if cfg.use_time_series_load:
        time_series_load_file = Path(cfg.time_series_load_file).resolve()
        if not time_series_load_file.exists():
            raise FileNotFoundError(
                f"time_series_load_file not found: {time_series_load_file}"
            )
    else:
        time_series_load_file = None

    device = torch.device(cfg.device)
    dtype = get_torch_dtype(cfg.torch_dtype)

    student_params = {
        "alpha_flap": cfg.alpha_flap,
        "alpha_edge": cfg.alpha_edge,
        "alpha_torsion": cfg.alpha_torsion,
        "zeta_structural": cfg.zeta_structural,
        "ref_freq_hz": cfg.ref_freq_hz,
    }

    print()
    print("[Compare Config]")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")

    # ------------------------------------------------------------------
    # 1) Run direct student baseline.
    # ------------------------------------------------------------------
    print()
    print("[1/4] Running direct student baseline via run_student_case(...)")

    baseline = run_student_case(
        blade_csv=blade_csv,
        output_dir=output_dir / "direct_student",
        case_name=cfg.case_name,
        use_time_series_load=cfg.use_time_series_load,
        t_initial=cfg.t_initial,
        t_final=cfg.t_final,
        dt=cfg.dt,
        time_series_load_file=time_series_load_file,
        student_params=student_params,
        u0_trans=None,
        v0_trans=None,
    )

    time = np.asarray(baseline["time"], dtype=np.float64)
    F_time = np.asarray(baseline["F_time"], dtype=np.float64)

    u_ref = np.asarray(baseline["u_full"], dtype=np.float64)
    v_ref = np.asarray(baseline["v_full"], dtype=np.float64)
    a_ref = np.asarray(baseline["a_full"], dtype=np.float64)

    print(f"  time shape   = {time.shape}")
    print(f"  F_time shape = {F_time.shape}")
    print(f"  u_ref shape  = {u_ref.shape}")
    print(f"  v_ref shape  = {v_ref.shape}")
    print(f"  a_ref shape  = {a_ref.shape}")

    # ------------------------------------------------------------------
    # 2) Rebuild full-order M/K/C without modal reduction.
    # ------------------------------------------------------------------
    print()
    print("[2/4] Rebuilding full-order M/K/C like run_student_case(...)")

    mkc = build_student_mkc_like_run_student_case(
        blade_csv=blade_csv,
        dt=cfg.dt,
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

    if M.shape[0] != F_time.shape[1]:
        raise ValueError(
            f"M/F_time DOF mismatch: M.shape[0]={M.shape[0]}, "
            f"F_time.shape[1]={F_time.shape[1]}"
        )

    # ------------------------------------------------------------------
    # 3) Run FullOrderCorrectedCoreTorch with zero correction.
    # ------------------------------------------------------------------
    print()
    print("[3/4] Running FullOrderCorrectedCoreTorch(theta_full=None)")

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

    print("  core summary:")
    for k, v in core.summary().items():
        print(f"    {k}: {v}")

    F_torch = torch.as_tensor(F_time, dtype=dtype, device=device)

    # 使用 direct baseline 的初始 full state，确保 compare 不引入初始条件差异。
    u0_torch = torch.as_tensor(u_ref[0], dtype=dtype, device=device)
    v0_torch = torch.as_tensor(v_ref[0], dtype=dtype, device=device)

    with torch.no_grad():
        u_pred_t, v_pred_t, a_pred_t = core.rollout(
            F_time=F_torch,
            u0=u0_torch,
            v0=v0_torch,
            theta_full=None,
            return_debug=False,
        )

    u_pred = as_numpy(u_pred_t)
    v_pred = as_numpy(v_pred_t)
    a_pred = as_numpy(a_pred_t)

    # ------------------------------------------------------------------
    # 4) Compare.
    # ------------------------------------------------------------------
    print()
    print("[4/4] Comparing torch full-order core against direct student baseline")

    metrics = compute_compare_metrics(
        u_pred=u_pred,
        v_pred=v_pred,
        a_pred=a_pred,
        u_ref=u_ref,
        v_ref=v_ref,
        a_ref=a_ref,
    )

    passed, checks = check_pass(metrics, cfg)
    print_metrics(metrics, checks)

    print()
    if passed:
        print("✅ PASS: zero-correction full-order torch core matches direct student baseline.")
    else:
        print("❌ FAIL: zero-correction full-order torch core does NOT match direct student baseline.")
        print("   建议优先检查 dtype、C 矩阵、初始加速度、F_time 和 Newmark 系数。")

    report = {
        "passed": bool(passed),
        "checks": checks,
        "metrics": metrics,
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
        report_path = output_dir / f"{cfg.case_name}_full_order_zero_compare_report.json"
        save_json(report_path, report)
        print()
        print(f"[Saved Report] {report_path}")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()