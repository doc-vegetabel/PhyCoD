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
from src.student.coupled_fem_builder import (  # noqa: E402
    build_coupled_fem_matrices_6dof_degrees,
    bending_xy_coupling_norms_6dof,
)
from src.student.full_order_corrected_core_torch import (  # noqa: E402
    FullOrderCorrectedCoreTorch,
    FullOrderCorrectedCoreTorchConfig,
)
from src.teacher.beamdyn_teacher_adapter import (  # noqa: E402
    get_full_state_component,
    get_tip_component,
    get_last_k_component,
)


@dataclass
class PrincipalAxisRotationCouplingSanityConfig:
    blade_csv: str = str(PROJECT_ROOT / "data" / "raw" / "nrel5mw" / "blade_master.csv")
    x_load_file: str = str(PROJECT_ROOT / "data" / "load" / "simple_tip_fx_case.dat")
    y_load_file: str = str(PROJECT_ROOT / "data" / "load" / "simple_tip_fy_case.dat")

    output_dir: str = str(PROJECT_ROOT / "results" / "student" / "principal_axis_rotation_coupling_sanity")
    case_prefix: str = "principal_axis_rotation_coupling_sanity"

    t_initial: float = 0.0
    t_final: float = 10.0
    dt: float = 0.01

    alpha_flap: float = 1.0
    alpha_edge: float = 1.0
    alpha_torsion: float = 1.0
    zeta_structural: float = 0.015
    ref_freq_hz: Optional[float] = None

    phi_deg: float = 5.0
    rotate_mass: bool = False

    torch_dtype: str = "float64"
    device: str = "cpu"
    linear_solve_mode: str = "solve"

    remove_initial_offset: bool = True
    last_k: int = 5

    atol_M_zero: float = 1e-12
    atol_K_zero: float = 1e-8
    min_K_xy_fro: float = 1e-6
    min_cross_response_rms: float = 1e-12

    save_report: bool = True


def parse_args() -> PrincipalAxisRotationCouplingSanityConfig:
    default = PrincipalAxisRotationCouplingSanityConfig()

    parser = argparse.ArgumentParser(
        description=(
            "Sanity test principal-axis rotation phi as a physical x-y "
            "bending coupling correction for the full-order student FEM."
        )
    )

    parser.add_argument("--blade-csv", type=str, default=default.blade_csv)
    parser.add_argument("--x-load-file", type=str, default=default.x_load_file)
    parser.add_argument("--y-load-file", type=str, default=default.y_load_file)

    parser.add_argument("--output-dir", type=str, default=default.output_dir)
    parser.add_argument("--case-prefix", type=str, default=default.case_prefix)

    parser.add_argument("--t-initial", type=float, default=default.t_initial)
    parser.add_argument("--t-final", type=float, default=default.t_final)
    parser.add_argument("--dt", type=float, default=default.dt)

    parser.add_argument("--alpha-flap", type=float, default=default.alpha_flap)
    parser.add_argument("--alpha-edge", type=float, default=default.alpha_edge)
    parser.add_argument("--alpha-torsion", type=float, default=default.alpha_torsion)
    parser.add_argument("--zeta-structural", type=float, default=default.zeta_structural)
    parser.add_argument("--ref-freq-hz", type=float, default=default.ref_freq_hz)

    parser.add_argument("--phi-deg", type=float, default=default.phi_deg)

    mass_group = parser.add_mutually_exclusive_group()
    mass_group.add_argument("--rotate-mass", dest="rotate_mass", action="store_true")
    mass_group.add_argument("--no-rotate-mass", dest="rotate_mass", action="store_false")
    parser.set_defaults(rotate_mass=default.rotate_mass)

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

    offset_group = parser.add_mutually_exclusive_group()
    offset_group.add_argument("--remove-initial-offset", dest="remove_initial_offset", action="store_true")
    offset_group.add_argument("--keep-initial-offset", dest="remove_initial_offset", action="store_false")
    parser.set_defaults(remove_initial_offset=default.remove_initial_offset)

    parser.add_argument("--last-k", type=int, default=default.last_k)

    parser.add_argument("--atol-M-zero", type=float, default=default.atol_M_zero)
    parser.add_argument("--atol-K-zero", type=float, default=default.atol_K_zero)
    parser.add_argument("--min-K-xy-fro", type=float, default=default.min_K_xy_fro)
    parser.add_argument("--min-cross-response-rms", type=float, default=default.min_cross_response_rms)

    save_group = parser.add_mutually_exclusive_group()
    save_group.add_argument("--save-report", dest="save_report", action="store_true")
    save_group.add_argument("--no-save-report", dest="save_report", action="store_false")
    parser.set_defaults(save_report=default.save_report)

    args = parser.parse_args()

    return PrincipalAxisRotationCouplingSanityConfig(
        blade_csv=args.blade_csv,
        x_load_file=args.x_load_file,
        y_load_file=args.y_load_file,
        output_dir=args.output_dir,
        case_prefix=args.case_prefix,
        t_initial=args.t_initial,
        t_final=args.t_final,
        dt=args.dt,
        alpha_flap=args.alpha_flap,
        alpha_edge=args.alpha_edge,
        alpha_torsion=args.alpha_torsion,
        zeta_structural=args.zeta_structural,
        ref_freq_hz=args.ref_freq_hz,
        phi_deg=args.phi_deg,
        rotate_mass=args.rotate_mass,
        torch_dtype=args.torch_dtype,
        device=args.device,
        linear_solve_mode=args.linear_solve_mode,
        remove_initial_offset=args.remove_initial_offset,
        last_k=args.last_k,
        atol_M_zero=args.atol_M_zero,
        atol_K_zero=args.atol_K_zero,
        min_K_xy_fro=args.min_K_xy_fro,
        min_cross_response_rms=args.min_cross_response_rms,
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


def matrix_diff_stats(A: np.ndarray, B: np.ndarray) -> Dict[str, float]:
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

    return as_numpy(u_t), as_numpy(v_t), as_numpy(a_t)


def remove_initial_offset(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    return u - u[:1, :]


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x**2)))


def response_component_summary(
    u: np.ndarray,
    *,
    primary_component: str,
    cross_component: str,
    last_k: int,
) -> Dict[str, Any]:
    primary_full = get_full_state_component(u, primary_component)
    cross_full = get_full_state_component(u, cross_component)

    primary_tip = get_tip_component(u, primary_component)
    cross_tip = get_tip_component(u, cross_component)

    primary_last_k = get_last_k_component(u, primary_component, k=last_k)
    cross_last_k = get_last_k_component(u, cross_component, k=last_k)

    def _ratio(num: float, den: float) -> float:
        return float(num / max(abs(den), 1e-30))

    out = {
        "primary_component": primary_component,
        "cross_component": cross_component,
        "primary_full_rms": rms(primary_full),
        "cross_full_rms": rms(cross_full),
        "primary_tip_rms": rms(primary_tip),
        "cross_tip_rms": rms(cross_tip),
        "primary_last_k_rms": rms(primary_last_k),
        "cross_last_k_rms": rms(cross_last_k),
    }

    out["full_cross_to_primary_rms"] = _ratio(out["cross_full_rms"], out["primary_full_rms"])
    out["tip_cross_to_primary_rms"] = _ratio(out["cross_tip_rms"], out["primary_tip_rms"])
    out["last_k_cross_to_primary_rms"] = _ratio(out["cross_last_k_rms"], out["primary_last_k_rms"])

    return out


def run_load_case_with_matrices(
    *,
    case_name: str,
    load_file: Path,
    M_uncoupled: np.ndarray,
    K_uncoupled: np.ndarray,
    M_coupled: np.ndarray,
    K_coupled: np.ndarray,
    cfg: PrincipalAxisRotationCouplingSanityConfig,
    dtype: torch.dtype,
    device: torch.device,
    primary_component: str,
    cross_component: str,
) -> Dict[str, Any]:
    print()
    print("=" * 100)
    print(f"[Response Case] {case_name}")
    print("=" * 100)
    print(f"  load_file = {load_file}")
    print(f"  primary_component = {primary_component}")
    print(f"  cross_component   = {cross_component}")

    student_params = {
        "alpha_flap": cfg.alpha_flap,
        "alpha_edge": cfg.alpha_edge,
        "alpha_torsion": cfg.alpha_torsion,
        "zeta_structural": cfg.zeta_structural,
        "ref_freq_hz": cfg.ref_freq_hz,
    }

    direct = run_student_case(
        blade_csv=Path(cfg.blade_csv).resolve(),
        output_dir=Path(cfg.output_dir).resolve() / case_name / "direct_student",
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

    F_time = np.asarray(direct["F_time"], dtype=np.float64)
    u_direct = np.asarray(direct["u_full"], dtype=np.float64)
    v_direct = np.asarray(direct["v_full"], dtype=np.float64)

    u0 = u_direct[0].copy()
    v0 = v_direct[0].copy()

    C_uncoupled, ref_uncoupled, freqs_uncoupled = build_damping_like_student(
        M_uncoupled,
        K_uncoupled,
        zeta_structural=cfg.zeta_structural,
        ref_freq_hz=cfg.ref_freq_hz,
    )
    C_coupled, ref_coupled, freqs_coupled = build_damping_like_student(
        M_coupled,
        K_coupled,
        zeta_structural=cfg.zeta_structural,
        ref_freq_hz=cfg.ref_freq_hz,
    )

    u_unc, v_unc, a_unc = rollout_with_mkc(
        M=M_uncoupled,
        K=K_uncoupled,
        C=C_uncoupled,
        F_time=F_time,
        u0=u0,
        v0=v0,
        dt=cfg.dt,
        dtype=dtype,
        device=device,
        linear_solve_mode=cfg.linear_solve_mode,
    )

    u_cpl, v_cpl, a_cpl = rollout_with_mkc(
        M=M_coupled,
        K=K_coupled,
        C=C_coupled,
        F_time=F_time,
        u0=u0,
        v0=v0,
        dt=cfg.dt,
        dtype=dtype,
        device=device,
        linear_solve_mode=cfg.linear_solve_mode,
    )

    if cfg.remove_initial_offset:
        u_unc_eval = remove_initial_offset(u_unc)
        u_cpl_eval = remove_initial_offset(u_cpl)
    else:
        u_unc_eval = u_unc
        u_cpl_eval = u_cpl

    unc_summary = response_component_summary(
        u_unc_eval,
        primary_component=primary_component,
        cross_component=cross_component,
        last_k=cfg.last_k,
    )
    cpl_summary = response_component_summary(
        u_cpl_eval,
        primary_component=primary_component,
        cross_component=cross_component,
        last_k=cfg.last_k,
    )

    response_change = {
        "u_max_abs": float(np.max(np.abs(u_cpl - u_unc))),
        "u_mae": float(np.mean(np.abs(u_cpl - u_unc))),
        "v_max_abs": float(np.max(np.abs(v_cpl - v_unc))),
        "a_max_abs": float(np.max(np.abs(a_cpl - a_unc))),
    }

    print()
    print("[Uncoupled response]")
    print(f"  full cross RMS = {unc_summary['cross_full_rms']:.12e}")
    print(f"  tip  cross RMS = {unc_summary['cross_tip_rms']:.12e}")
    print(f"  last-k cross RMS = {unc_summary['cross_last_k_rms']:.12e}")
    print(f"  full cross/primary = {unc_summary['full_cross_to_primary_rms']:.12e}")
    print(f"  tip  cross/primary = {unc_summary['tip_cross_to_primary_rms']:.12e}")
    print(f"  last-k cross/primary = {unc_summary['last_k_cross_to_primary_rms']:.12e}")

    print()
    print("[Coupled response]")
    print(f"  full cross RMS = {cpl_summary['cross_full_rms']:.12e}")
    print(f"  tip  cross RMS = {cpl_summary['cross_tip_rms']:.12e}")
    print(f"  last-k cross RMS = {cpl_summary['cross_last_k_rms']:.12e}")
    print(f"  full cross/primary = {cpl_summary['full_cross_to_primary_rms']:.12e}")
    print(f"  tip  cross/primary = {cpl_summary['tip_cross_to_primary_rms']:.12e}")
    print(f"  last-k cross/primary = {cpl_summary['last_k_cross_to_primary_rms']:.12e}")

    checks = {
        "uncoupled_cross_near_zero": unc_summary["cross_full_rms"] <= cfg.min_cross_response_rms,
        "coupled_cross_nonzero": cpl_summary["cross_full_rms"] > cfg.min_cross_response_rms,
        "coupled_response_finite": bool(
            np.all(np.isfinite(u_cpl))
            and np.all(np.isfinite(v_cpl))
            and np.all(np.isfinite(a_cpl))
        ),
    }

    print()
    print("[Response checks]")
    for k, v in checks.items():
        print(f"  {k:<32s}: {'PASS' if v else 'FAIL'}")

    return {
        "case_name": case_name,
        "load_file": str(load_file),
        "primary_component": primary_component,
        "cross_component": cross_component,
        "ref_freq_uncoupled": ref_uncoupled,
        "ref_freq_coupled": ref_coupled,
        "natural_freqs_uncoupled": freqs_uncoupled,
        "natural_freqs_coupled": freqs_coupled,
        "uncoupled_summary": unc_summary,
        "coupled_summary": cpl_summary,
        "response_change": response_change,
        "checks": checks,
    }


def main() -> None:
    cfg = parse_args()

    blade_csv = assert_existing_file(cfg.blade_csv, "blade_csv")
    x_load_file = assert_existing_file(cfg.x_load_file, "x_load_file")
    y_load_file = assert_existing_file(cfg.y_load_file, "y_load_file")

    output_dir = Path(cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = get_torch_dtype(cfg.torch_dtype)
    device = torch.device(cfg.device)

    print()
    print("[Principal Axis Rotation Coupling Sanity Test]")
    print()
    print("[Config]")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")

    print()
    print("[1/4] Loading StudentBeamModel")
    model = load_student_model_from_blade_master(
        csv_path=str(blade_csv),
        model_name="principal_axis_rotation_coupling_sanity_model",
    )
    print(f"  span_m = {model.span_m}")
    print(f"  n_stations = {model.n_stations}")
    print(f"  n_free_nodes = {model.n_stations - 1}")

    print()
    print("[2/4] Matrix equivalence and coupling checks")

    M_base, K_base = build_corrected_fem_matrices_6dof(
        model,
        section_params=None,
        alpha_flap=cfg.alpha_flap,
        alpha_edge=cfg.alpha_edge,
        alpha_torsion=cfg.alpha_torsion,
        return_full=False,
    )

    M_phi0, K_phi0, debug_phi0 = build_coupled_fem_matrices_6dof_degrees(
        model,
        section_params=None,
        phi_deg=0.0,
        alpha_flap=cfg.alpha_flap,
        alpha_edge=cfg.alpha_edge,
        alpha_torsion=cfg.alpha_torsion,
        rotate_mass=cfg.rotate_mass,
        return_full=True,
    )

    M_phi, K_phi, debug_phi = build_coupled_fem_matrices_6dof_degrees(
        model,
        section_params=None,
        phi_deg=cfg.phi_deg,
        alpha_flap=cfg.alpha_flap,
        alpha_edge=cfg.alpha_edge,
        alpha_torsion=cfg.alpha_torsion,
        rotate_mass=cfg.rotate_mass,
        return_full=True,
    )

    M_zero_diff = matrix_diff_stats(M_phi0, M_base)
    K_zero_diff = matrix_diff_stats(K_phi0, K_base)

    M_phi_diff = matrix_diff_stats(M_phi, M_base)
    K_phi_diff = matrix_diff_stats(K_phi, K_base)

    base_coupling = bending_xy_coupling_norms_6dof(K_base)
    phi0_coupling = bending_xy_coupling_norms_6dof(K_phi0)
    phi_coupling = bending_xy_coupling_norms_6dof(K_phi)

    matrix_checks = {
        "phi0_M_matches_baseline": M_zero_diff["max_abs"] <= cfg.atol_M_zero,
        "phi0_K_matches_baseline": K_zero_diff["max_abs"] <= cfg.atol_K_zero,
        "baseline_K_xy_near_zero": base_coupling["K_xy_fro"] <= cfg.min_K_xy_fro,
        "phi_K_xy_nonzero": phi_coupling["K_xy_fro"] > cfg.min_K_xy_fro,
        "phi_K_symmetric": phi_coupling["K_symmetry_max_abs"] <= cfg.atol_K_zero,
    }

    print()
    print("[phi=0 vs baseline]")
    print(f"  M max_abs = {M_zero_diff['max_abs']:.12e}")
    print(f"  K max_abs = {K_zero_diff['max_abs']:.12e}")

    print()
    print(f"[phi={cfg.phi_deg} deg vs baseline]")
    print(f"  M max_abs = {M_phi_diff['max_abs']:.12e}")
    print(f"  K max_abs = {K_phi_diff['max_abs']:.12e}")
    print(f"  K_xy_fro = {phi_coupling['K_xy_fro']:.12e}")
    print(f"  K_xy_over_K_xx = {phi_coupling['K_xy_over_K_xx']:.12e}")
    print(f"  K_xy_over_K_yy = {phi_coupling['K_xy_over_K_yy']:.12e}")

    print()
    print("[Matrix checks]")
    for k, v in matrix_checks.items():
        print(f"  {k:<32s}: {'PASS' if v else 'FAIL'}")

    print()
    print("[3/4] x-only response coupling check")
    x_case = run_load_case_with_matrices(
        case_name=f"{cfg.case_prefix}_x_only_phi_{cfg.phi_deg:g}deg",
        load_file=x_load_file,
        M_uncoupled=M_base,
        K_uncoupled=K_base,
        M_coupled=M_phi,
        K_coupled=K_phi,
        cfg=cfg,
        dtype=dtype,
        device=device,
        primary_component="x",
        cross_component="y",
    )

    print()
    print("[4/4] y-only response coupling check")
    y_case = run_load_case_with_matrices(
        case_name=f"{cfg.case_prefix}_y_only_phi_{cfg.phi_deg:g}deg",
        load_file=y_load_file,
        M_uncoupled=M_base,
        K_uncoupled=K_base,
        M_coupled=M_phi,
        K_coupled=K_phi,
        cfg=cfg,
        dtype=dtype,
        device=device,
        primary_component="y",
        cross_component="x",
    )

    all_checks = list(matrix_checks.values())
    all_checks.extend(x_case["checks"].values())
    all_checks.extend(y_case["checks"].values())

    all_passed = bool(all(bool(v) for v in all_checks))

    report = {
        "passed": all_passed,
        "config": asdict(cfg),
        "matrix_checks": matrix_checks,
        "matrix_metrics": {
            "M_zero_diff": M_zero_diff,
            "K_zero_diff": K_zero_diff,
            "M_phi_diff": M_phi_diff,
            "K_phi_diff": K_phi_diff,
            "base_coupling": base_coupling,
            "phi0_coupling": phi0_coupling,
            "phi_coupling": phi_coupling,
        },
        "debug_summary": {
            "phi0_builder": debug_phi0["builder"],
            "phi_builder": debug_phi["builder"],
            "phi_element_deg_head": debug_phi["phi_element_deg"][:10],
            "rotate_mass": debug_phi["rotate_mass"],
        },
        "x_only": x_case,
        "y_only": y_case,
    }

    if cfg.save_report:
        report_path = output_dir / "principal_axis_rotation_coupling_sanity_report.json"
        save_json(report_path, report)
        print()
        print(f"[Saved Report] {report_path}")

    print()
    if all_passed:
        print("✅ PASS: principal-axis rotation coupling sanity test passed.")
        print("   phi=0 recovers baseline; phi!=0 creates K_xy coupling and cross-axis response.")
    else:
        print("❌ FAIL: principal-axis rotation coupling sanity test failed.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()