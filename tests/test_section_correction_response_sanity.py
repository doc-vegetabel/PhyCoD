from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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
from src.student.corrected_fem_builder import build_corrected_fem_matrices_6dof  # noqa: E402
from src.student.section_parameterization import (  # noqa: E402
    build_baseline_section_parameters,
    apply_section_parameter_corrections,
    make_uniform_section_correction,
)
from src.student.full_order_corrected_core_torch import (  # noqa: E402
    FullOrderCorrectedCoreTorch,
    FullOrderCorrectedCoreTorchConfig,
)


@dataclass
class SectionCorrectionResponseSanityConfig:
    blade_csv: str = str(PROJECT_ROOT / "data" / "raw" / "nrel5mw" / "blade_master.csv")
    output_dir: str = str(PROJECT_ROOT / "results" / "student" / "section_correction_response_sanity")
    case_name: str = "section_correction_response_sanity"

    time_series_load_file: str = str(PROJECT_ROOT / "data" / "load" / "test_complex_case.dat")

    # 先用 2s 做 sanity，避免测试太慢
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

    # 手动物理修正幅值，只用于 sanity test，不代表真实训练尺度
    delta_EI_flap_relative: float = 0.01
    delta_EI_edge_relative: float = 0.01
    delta_GJ_relative: float = 0.01
    delta_J_rho_relative: float = 0.01

    # 非零修正响应变化阈值
    min_response_change: float = 1e-10

    save_report: bool = True


def parse_args() -> SectionCorrectionResponseSanityConfig:
    default = SectionCorrectionResponseSanityConfig()

    parser = argparse.ArgumentParser(
        description=(
            "Response-level sanity test for physical section-parameter corrections "
            "through corrected_fem_builder.py and FullOrderCorrectedCoreTorch."
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
        "--delta-EI-flap-relative",
        type=float,
        default=default.delta_EI_flap_relative,
    )
    parser.add_argument(
        "--delta-EI-edge-relative",
        type=float,
        default=default.delta_EI_edge_relative,
    )
    parser.add_argument(
        "--delta-GJ-relative",
        type=float,
        default=default.delta_GJ_relative,
    )
    parser.add_argument(
        "--delta-J-rho-relative",
        type=float,
        default=default.delta_J_rho_relative,
    )
    parser.add_argument("--min-response-change", type=float, default=default.min_response_change)

    save_group = parser.add_mutually_exclusive_group()
    save_group.add_argument("--save-report", dest="save_report", action="store_true")
    save_group.add_argument("--no-save-report", dest="save_report", action="store_false")
    parser.set_defaults(save_report=default.save_report)

    args = parser.parse_args()

    return SectionCorrectionResponseSanityConfig(
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
        delta_EI_flap_relative=args.delta_EI_flap_relative,
        delta_EI_edge_relative=args.delta_EI_edge_relative,
        delta_GJ_relative=args.delta_GJ_relative,
        delta_J_rho_relative=args.delta_J_rho_relative,
        min_response_change=args.min_response_change,
        save_report=args.save_report,
    )


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
    """
    使用与 run_student_case(...) 一致的刚度比例阻尼逻辑：

        C = beta_damp * K
        beta_damp = 2*zeta / (2*pi*f_ref)

    若 ref_freq_hz=None，则根据当前 M/K 的第一阶频率确定。
    """
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


def abs_stats(pred: np.ndarray, ref: np.ndarray) -> Tuple[float, float]:
    diff = np.asarray(pred, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
    return float(np.mean(np.abs(diff))), float(np.max(np.abs(diff)))


def response_diff_metrics(
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

    return {
        "u_mae": u_mae,
        "u_max_abs": u_max,
        "v_mae": v_mae,
        "v_max_abs": v_max,
        "a_mae": a_mae,
        "a_max_abs": a_max,
    }


def matrix_diff_metrics(A: np.ndarray, B: np.ndarray) -> Dict[str, float]:
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    diff = A - B
    abs_diff = np.abs(diff)
    denom = float(np.linalg.norm(B, ord="fro"))

    return {
        "mae": float(np.mean(abs_diff)),
        "max_abs": float(np.max(abs_diff)),
        "relative_fro": float(np.linalg.norm(diff, ord="fro") / max(denom, 1e-30)),
    }


def is_response_finite(u: np.ndarray, v: np.ndarray, a: np.ndarray) -> bool:
    return bool(
        np.all(np.isfinite(u))
        and np.all(np.isfinite(v))
        and np.all(np.isfinite(a))
    )


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


def build_corrected_mkc_from_section_params(
    *,
    model,
    section_params,
    cfg: SectionCorrectionResponseSanityConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[float], np.ndarray]:
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

    return M, K, C, ref_freq_used, natural_freqs


def print_response_metrics(
    title: str,
    metrics: Dict[str, float],
    checks: Optional[Dict[str, bool]] = None,
) -> None:
    print()
    print(title)
    for k, v in metrics.items():
        flag = ""
        if checks is not None and k in checks:
            flag = "  PASS" if checks[k] else "  FAIL"
        print(f"  {k:<18s}: {v:.12e}{flag}")


def print_checks(title: str, checks: Dict[str, bool]) -> None:
    print()
    print(title)
    for k, v in checks.items():
        print(f"  {k:<32s}: {'PASS' if v else 'FAIL'}")


def check_zero_response(metrics: Dict[str, float], cfg: SectionCorrectionResponseSanityConfig) -> Tuple[bool, Dict[str, bool]]:
    checks = {
        "u_mae": metrics["u_mae"] <= cfg.atol_u_mae,
        "u_max_abs": metrics["u_max_abs"] <= cfg.atol_u_max,
        "v_mae": metrics["v_mae"] <= cfg.atol_v_mae,
        "v_max_abs": metrics["v_max_abs"] <= cfg.atol_v_max,
        "a_mae": metrics["a_mae"] <= cfg.atol_a_mae,
        "a_max_abs": metrics["a_max_abs"] <= cfg.atol_a_max,
    }
    return bool(all(checks.values())), checks


def check_active_response(metrics: Dict[str, float], finite: bool, cfg: SectionCorrectionResponseSanityConfig) -> Tuple[bool, Dict[str, bool]]:
    max_change = max(
        float(metrics["u_max_abs"]),
        float(metrics["v_max_abs"]),
        float(metrics["a_max_abs"]),
    )

    checks = {
        "response_changed": max_change > cfg.min_response_change,
        "response_finite": bool(finite),
    }
    return bool(all(checks.values())), checks


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

    print()
    print("[Section Correction Response Sanity Test]")
    print()
    print("[Config]")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")

    student_params = {
        "alpha_flap": cfg.alpha_flap,
        "alpha_edge": cfg.alpha_edge,
        "alpha_torsion": cfg.alpha_torsion,
        "zeta_structural": cfg.zeta_structural,
        "ref_freq_hz": cfg.ref_freq_hz,
    }

    print()
    print("[1/6] Running direct student baseline via run_student_case(...)")
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

    u0 = u_ref[0].copy()
    v0 = v_ref[0].copy()

    print(f"  F_time shape = {F_time.shape}")
    print(f"  u_ref shape  = {u_ref.shape}")
    print(f"  v_ref shape  = {v_ref.shape}")
    print(f"  a_ref shape  = {a_ref.shape}")

    print()
    print("[2/6] Loading model and baseline section parameters")
    model = load_student_model_from_blade_master(
        csv_path=str(blade_csv),
        model_name="section_correction_response_sanity_model",
    )
    baseline_params = build_baseline_section_parameters(model)

    print(f"  model span_m = {model.span_m}")
    print(f"  n_stations = {model.n_stations}")
    print(f"  n_free_nodes = {model.n_stations - 1}")
    print(f"  section source = {baseline_params.source}")

    print()
    print("[3/6] Zero section correction rollout")
    zero_params = apply_section_parameter_corrections(
        baseline_params,
        correction=None,
        source_suffix="zero_response_check",
    )
    M_zero, K_zero, C_zero, ref_freq_zero, freqs_zero = build_corrected_mkc_from_section_params(
        model=model,
        section_params=zero_params,
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

    zero_metrics = response_diff_metrics(
        u_pred=u_zero,
        v_pred=v_zero,
        a_pred=a_zero,
        u_ref=u_ref,
        v_ref=v_ref,
        a_ref=a_ref,
    )
    zero_passed, zero_checks = check_zero_response(zero_metrics, cfg)

    print(f"  zero ref_freq_used = {ref_freq_zero}")
    print(f"  zero natural_freqs[:5] = {freqs_zero[:5]}")
    print_response_metrics("[Zero Section Correction vs Direct Baseline]", zero_metrics, zero_checks)

    print()
    print("[4/6] Building active section correction cases")
    correction_cases = {
        "delta_EI_flap": make_uniform_section_correction(
            delta_EI_flap_relative=cfg.delta_EI_flap_relative,
        ),
        "delta_EI_edge": make_uniform_section_correction(
            delta_EI_edge_relative=cfg.delta_EI_edge_relative,
        ),
        "delta_GJ": make_uniform_section_correction(
            delta_GJ_relative=cfg.delta_GJ_relative,
        ),
        "delta_J_rho": make_uniform_section_correction(
            delta_J_rho_relative=cfg.delta_J_rho_relative,
        ),
    }

    results: Dict[str, Any] = {
        "zero_correction": {
            "passed": zero_passed,
            "checks": zero_checks,
            "response_metrics_vs_direct_baseline": zero_metrics,
            "ref_freq_used": ref_freq_zero,
            "natural_freqs_hz": freqs_zero,
            "core_summary": zero_core_summary,
            "section_source": zero_params.source,
        }
    }

    print()
    print("[5/6] Active section correction response rollouts")
    active_pass_flags = []

    for case_name, correction in correction_cases.items():
        print()
        print(f"--- Active case: {case_name} ---")

        corrected_params = apply_section_parameter_corrections(
            baseline_params,
            correction=correction,
            source_suffix=f"{case_name}_response_check",
        )

        M_corr, K_corr, C_corr, ref_freq_corr, freqs_corr = build_corrected_mkc_from_section_params(
            model=model,
            section_params=corrected_params,
            cfg=cfg,
        )

        M_change = matrix_diff_metrics(M_corr, M_zero)
        K_change = matrix_diff_metrics(K_corr, K_zero)
        C_change = matrix_diff_metrics(C_corr, C_zero)

        u_corr, v_corr, a_corr, core_summary = rollout_with_mkc(
            M=M_corr,
            K=K_corr,
            C=C_corr,
            F_time=F_time,
            u0=u0,
            v0=v0,
            dt=cfg.dt,
            dtype=dtype,
            device=device,
            linear_solve_mode=cfg.linear_solve_mode,
        )

        response_change_metrics = response_diff_metrics(
            u_pred=u_corr,
            v_pred=v_corr,
            a_pred=a_corr,
            u_ref=u_zero,
            v_ref=v_zero,
            a_ref=a_zero,
        )

        finite = is_response_finite(u_corr, v_corr, a_corr)
        active_passed, active_checks = check_active_response(
            response_change_metrics,
            finite=finite,
            cfg=cfg,
        )
        active_pass_flags.append(active_passed)

        print(f"  ref_freq_used = {ref_freq_corr}")
        print(f"  natural_freqs[:5] = {freqs_corr[:5]}")
        print(
            f"  M change: max_abs={M_change['max_abs']:.12e}, "
            f"relative_fro={M_change['relative_fro']:.12e}"
        )
        print(
            f"  K change: max_abs={K_change['max_abs']:.12e}, "
            f"relative_fro={K_change['relative_fro']:.12e}"
        )
        print(
            f"  C change: max_abs={C_change['max_abs']:.12e}, "
            f"relative_fro={C_change['relative_fro']:.12e}"
        )

        print_response_metrics(
            "[Response Change vs Zero Section Correction]",
            response_change_metrics,
        )
        print_checks("[Active Response Checks]", active_checks)

        results[case_name] = {
            "passed": active_passed,
            "checks": active_checks,
            "response_change_vs_zero": response_change_metrics,
            "finite": finite,
            "M_change_vs_zero": M_change,
            "K_change_vs_zero": K_change,
            "C_change_vs_zero": C_change,
            "ref_freq_used": ref_freq_corr,
            "natural_freqs_hz": freqs_corr,
            "core_summary": core_summary,
            "section_source": corrected_params.source,
        }

    print()
    print("[6/6] Summary")
    all_passed = bool(zero_passed and all(active_pass_flags))

    for name, item in results.items():
        print(f"  {name:<24s}: {'PASS' if item['passed'] else 'FAIL'}")

    report = {
        "passed": all_passed,
        "config": asdict(cfg),
        "blade_csv": str(blade_csv),
        "time_series_load_file": str(load_file),
        "baseline_npz": str(baseline["npz"]),
        "baseline_yaml": str(baseline["yaml"]),
        "model_info": {
            "model_name": model.model_name,
            "span_m": float(model.span_m),
            "n_stations": int(model.n_stations),
            "n_free_nodes": int(model.n_stations - 1),
            "n_dofs_free": int(F_time.shape[1]),
        },
        "results": results,
    }

    if cfg.save_report:
        report_path = output_dir / "section_correction_response_sanity_report.json"
        save_json(report_path, report)
        print()
        print(f"[Saved Report] {report_path}")

    print()
    if all_passed:
        print("✅ PASS: section correction response sanity test passed.")
        print("   Zero correction matches direct baseline; nonzero physical section corrections change response.")
    else:
        print("❌ FAIL: section correction response sanity test failed.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()