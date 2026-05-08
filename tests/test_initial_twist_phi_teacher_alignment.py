from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
class InitialTwistPhiPlotCompareConfig:
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

    output_dir: str = str(PROJECT_ROOT / "results" / "student" / "initial_twist_phi_plot_compare")
    case_name: str = "initial_twist_phi_plot_compare"

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

    # baseline: phi = 0
    phi0_deg: float = 0.0

    # comparison: phi = sign * initial_twist_deg(s)
    phi1_twist_column: str = "initial_twist_deg"
    phi1_sign: float = -1.0

    torch_dtype: str = "float64"
    device: str = "cpu"
    linear_solve_mode: str = "solve"

    remove_initial_offset: bool = True
    show_plot: bool = False


def parse_args() -> InitialTwistPhiPlotCompareConfig:
    d = InitialTwistPhiPlotCompareConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Compare teacher / student(phi=0) / student(phi=initial_twist_deg from CSV) "
            "tip displacement time histories and save x/y plots."
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

    parser.add_argument("--phi0-deg", type=float, default=d.phi0_deg)
    parser.add_argument("--phi1-twist-column", type=str, default=d.phi1_twist_column)
    parser.add_argument("--phi1-sign", type=float, default=d.phi1_sign)

    parser.add_argument("--torch-dtype", type=str, default=d.torch_dtype, choices=["float64", "float32"])
    parser.add_argument("--device", type=str, default=d.device)
    parser.add_argument("--linear-solve-mode", type=str, default=d.linear_solve_mode, choices=["solve", "inverse"])

    offset_group = parser.add_mutually_exclusive_group()
    offset_group.add_argument("--remove-initial-offset", dest="remove_initial_offset", action="store_true")
    offset_group.add_argument("--keep-initial-offset", dest="remove_initial_offset", action="store_false")
    parser.set_defaults(remove_initial_offset=d.remove_initial_offset)

    show_group = parser.add_mutually_exclusive_group()
    show_group.add_argument("--show-plot", dest="show_plot", action="store_true")
    show_group.add_argument("--no-show-plot", dest="show_plot", action="store_false")
    parser.set_defaults(show_plot=d.show_plot)

    args = parser.parse_args()

    return InitialTwistPhiPlotCompareConfig(
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
        phi0_deg=args.phi0_deg,
        phi1_twist_column=args.phi1_twist_column,
        phi1_sign=args.phi1_sign,
        torch_dtype=args.torch_dtype,
        device=args.device,
        linear_solve_mode=args.linear_solve_mode,
        remove_initial_offset=args.remove_initial_offset,
        show_plot=args.show_plot,
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


def build_element_phi_from_initial_twist_csv(
    *,
    blade_csv: Path,
    twist_column: str = "initial_twist_deg",
    sign: float = 1.0,
) -> np.ndarray:
    """
    从 blade_master.csv 读取 49 个截面 initial_twist_deg，
    并通过相邻截面平均构造 48 个 element-level phi。

    返回:
        phi_element_deg: shape = (48,)
    """
    df = pd.read_csv(blade_csv)

    if twist_column not in df.columns:
        raise KeyError(
            f"Column '{twist_column}' not found in {blade_csv}. "
            f"Available columns: {list(df.columns)}"
        )

    phi_station_deg = float(sign) * df[twist_column].to_numpy(dtype=np.float64)

    if phi_station_deg.ndim != 1:
        raise ValueError(f"{twist_column} must be a 1D column.")

    if len(phi_station_deg) != 49:
        print(
            f"[Warning] Expected 49 station twist values, "
            f"but got {len(phi_station_deg)}. "
            f"Will still use adjacent averaging."
        )

    if len(phi_station_deg) < 2:
        raise ValueError("Need at least 2 station twist values to build element phi.")

    phi_element_deg = 0.5 * (phi_station_deg[:-1] + phi_station_deg[1:])

    print()
    print("[Initial Twist Phi]")
    print(f"  source column       = {twist_column}")
    print(f"  sign                = {float(sign):+.3f}")
    print(f"  station phi shape   = {phi_station_deg.shape}")
    print(f"  element phi shape   = {phi_element_deg.shape}")
    print(
        "  station phi deg     "
        f"min={phi_station_deg.min():.6f}, "
        f"max={phi_station_deg.max():.6f}, "
        f"mean={phi_station_deg.mean():.6f}"
    )
    print(
        "  element phi deg     "
        f"min={phi_element_deg.min():.6f}, "
        f"max={phi_element_deg.max():.6f}, "
        f"mean={phi_element_deg.mean():.6f}"
    )

    return phi_element_deg


def rollout_with_phi(
    *,
    model,
    phi_deg: float | np.ndarray,
    F_time: np.ndarray,
    u0: np.ndarray,
    v0: np.ndarray,
    cfg: InitialTwistPhiPlotCompareConfig,
    dtype: torch.dtype,
    device: torch.device,
) -> np.ndarray:
    """
    支持两种 phi 输入：
    1. scalar: uniform phi，例如 0.0
    2. ndarray: element-level phi，例如 shape=(48,) 的 initial_twist phi
    """
    if np.ndim(phi_deg) == 0:
        phi_arg = float(phi_deg)
        print(f"  rollout phi scalar = {phi_arg:.6f} deg")
    else:
        phi_arg = np.asarray(phi_deg, dtype=np.float64)
        print(
            "  rollout phi array  "
            f"shape={phi_arg.shape}, "
            f"min={phi_arg.min():.6f}, "
            f"max={phi_arg.max():.6f}, "
            f"mean={phi_arg.mean():.6f} deg"
        )

    M, K, _ = build_coupled_fem_matrices_6dof_degrees(
        model,
        section_params=None,
        phi_deg=phi_arg,
        alpha_flap=cfg.alpha_flap,
        alpha_edge=cfg.alpha_edge,
        alpha_torsion=cfg.alpha_torsion,
        rotate_mass=False,
        return_full=True,
    )

    C, ref_freq_used, natural_freqs = build_damping_like_student(
        M=M,
        K=K,
        zeta_structural=cfg.zeta_structural,
        ref_freq_hz=cfg.ref_freq_hz,
    )

    print(f"  natural_freqs[:5] = {natural_freqs[:5]}")
    print(f"  ref_freq_used     = {ref_freq_used}")

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


def plot_one_direction(
    *,
    time: np.ndarray,
    teacher_signal: np.ndarray,
    phi0_signal: np.ndarray,
    phi1_signal: np.ndarray,
    direction: str,
    phi0_label: str,
    phi1_label: str,
    save_path: Path,
    case_name: str,
) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(time, teacher_signal, label="Teacher")
    plt.plot(time, phi0_signal, label=f"Student {phi0_label}")
    plt.plot(time, phi1_signal, label=f"Student {phi1_label}")
    plt.xlabel("Time [s]")
    plt.ylabel(f"Tip displacement in {direction} direction")
    plt.title(f"{case_name} - Tip {direction}-direction displacement time history")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


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
    print("[Initial Twist Phi Plot Compare]")
    print()
    print("[Config]")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")

    print()
    print("[1/4] Running BeamDyn teacher")
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
    print("[2/4] Loading teacher response")
    time_teacher, u_teacher, _ = load_teacher_6dof_response(
        out_path=teacher_out,
        node_start=cfg.teacher_node_start,
        node_end=cfg.teacher_node_end,
        demean=cfg.teacher_demean,
    )
    print(f"  time_teacher shape = {time_teacher.shape}")
    print(f"  u_teacher shape    = {u_teacher.shape}")

    print()
    print("[3/4] Running direct student once to get force history and initial state")
    direct_result = run_student_case(
        blade_csv=blade_csv,
        output_dir=output_dir / "direct_student",
        case_name=cfg.case_name,
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

    time_student = np.asarray(direct_result["time"], dtype=np.float64)
    F_time = np.asarray(direct_result["F_time"], dtype=np.float64)
    u_direct = np.asarray(direct_result["u_full"], dtype=np.float64)
    v_direct = np.asarray(direct_result["v_full"], dtype=np.float64)

    print(f"  time_student shape = {time_student.shape}")
    print(f"  F_time shape       = {F_time.shape}")
    print(f"  u_direct shape     = {u_direct.shape}")
    print(f"  v_direct shape     = {v_direct.shape}")

    print()
    print("[4/4] Running student with phi=0 and phi=initial_twist_deg from CSV")
    model = load_student_model_from_blade_master(
        csv_path=str(blade_csv),
        model_name="initial_twist_phi_plot_compare_model",
    )

    phi1_element_deg = build_element_phi_from_initial_twist_csv(
        blade_csv=blade_csv,
        twist_column=cfg.phi1_twist_column,
        sign=cfg.phi1_sign,
    )

    phi0_label = f"phi={cfg.phi0_deg:g}°"
    phi1_label = f"phi={cfg.phi1_sign:+g}*{cfg.phi1_twist_column}"

    print()
    print("[Student rollout] phi=0")
    u_phi0 = rollout_with_phi(
        model=model,
        phi_deg=cfg.phi0_deg,
        F_time=F_time,
        u0=u_direct[0].copy(),
        v0=v_direct[0].copy(),
        cfg=cfg,
        dtype=dtype,
        device=device,
    )

    print()
    print("[Student rollout] phi=initial_twist_deg")
    u_phi1 = rollout_with_phi(
        model=model,
        phi_deg=phi1_element_deg,
        F_time=F_time,
        u0=u_direct[0].copy(),
        v0=v_direct[0].copy(),
        cfg=cfg,
        dtype=dtype,
        device=device,
    )

    u_phi0_rs = resample_response_to_time_grid(
        time_src=time_student,
        u_src=u_phi0,
        time_dst=time_teacher,
    )
    u_phi1_rs = resample_response_to_time_grid(
        time_src=time_student,
        u_src=u_phi1,
        time_dst=time_teacher,
    )

    if cfg.remove_initial_offset:
        u_teacher_eval = remove_initial_offset(u_teacher)
        u_phi0_eval = remove_initial_offset(u_phi0_rs)
        u_phi1_eval = remove_initial_offset(u_phi1_rs)
    else:
        u_teacher_eval = u_teacher
        u_phi0_eval = u_phi0_rs
        u_phi1_eval = u_phi1_rs

    teacher_tip_x = get_tip_component(u_teacher_eval, "x")
    teacher_tip_y = get_tip_component(u_teacher_eval, "y")

    phi0_tip_x = get_tip_component(u_phi0_eval, "x")
    phi0_tip_y = get_tip_component(u_phi0_eval, "y")

    phi1_tip_x = get_tip_component(u_phi1_eval, "x")
    phi1_tip_y = get_tip_component(u_phi1_eval, "y")

    x_plot_path = output_dir / "x_direction_timeseries.png"
    y_plot_path = output_dir / "y_direction_timeseries.png"

    np.savez_compressed(
        output_dir / "initial_twist_phi_used.npz",
        phi_element_deg=phi1_element_deg,
        phi_sign=cfg.phi1_sign,
        twist_column=cfg.phi1_twist_column,
    )

    np.savez_compressed(
        output_dir / "timeseries_compare_data.npz",
        time_teacher=time_teacher,
        time_student=time_student,
        u_teacher=u_teacher,
        u_teacher_eval=u_teacher_eval,
        u_phi0=u_phi0,
        u_phi1=u_phi1,
        u_phi0_resampled=u_phi0_rs,
        u_phi1_resampled=u_phi1_rs,
        u_phi0_eval=u_phi0_eval,
        u_phi1_eval=u_phi1_eval,
        teacher_tip_x=teacher_tip_x,
        teacher_tip_y=teacher_tip_y,
        phi0_tip_x=phi0_tip_x,
        phi0_tip_y=phi0_tip_y,
        phi1_tip_x=phi1_tip_x,
        phi1_tip_y=phi1_tip_y,
        F_time=F_time,
        phi1_element_deg=phi1_element_deg,
    )

    plot_one_direction(
        time=time_teacher,
        teacher_signal=teacher_tip_x,
        phi0_signal=phi0_tip_x,
        phi1_signal=phi1_tip_x,
        direction="x",
        phi0_label=phi0_label,
        phi1_label=phi1_label,
        save_path=x_plot_path,
        case_name=cfg.case_name,
    )

    plot_one_direction(
        time=time_teacher,
        teacher_signal=teacher_tip_y,
        phi0_signal=phi0_tip_y,
        phi1_signal=phi1_tip_y,
        direction="y",
        phi0_label=phi0_label,
        phi1_label=phi1_label,
        save_path=y_plot_path,
        case_name=cfg.case_name,
    )

    print()
    print("[Saved Figures]")
    print(f"  x plot: {x_plot_path}")
    print(f"  y plot: {y_plot_path}")
    print()
    print("[Saved Data]")
    print(f"  phi used: {output_dir / 'initial_twist_phi_used.npz'}")
    print(f"  timeseries data: {output_dir / 'timeseries_compare_data.npz'}")

    if cfg.show_plot:
        plt.figure(figsize=(12, 6))
        plt.plot(time_teacher, teacher_tip_x, label="Teacher")
        plt.plot(time_teacher, phi0_tip_x, label=f"Student {phi0_label}")
        plt.plot(time_teacher, phi1_tip_x, label=f"Student {phi1_label}")
        plt.xlabel("Time [s]")
        plt.ylabel("Tip displacement in x direction")
        plt.title(f"{cfg.case_name} - Tip x-direction displacement time history")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(time_teacher, teacher_tip_y, label="Teacher")
        plt.plot(time_teacher, phi0_tip_y, label=f"Student {phi0_label}")
        plt.plot(time_teacher, phi1_tip_y, label=f"Student {phi1_label}")
        plt.xlabel("Time [s]")
        plt.ylabel("Tip displacement in y direction")
        plt.title(f"{cfg.case_name} - Tip y-direction displacement time history")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    print()
    print("✅ PASS: 已输出 x / y 两个方向的三曲线时历图。")
    print("   三条曲线分别为：Teacher、Student phi=0、Student phi=initial_twist_deg(s)。")


if __name__ == "__main__":
    main()