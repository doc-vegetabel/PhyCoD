from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
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
class UniformPhiPlotCompareConfig:
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
    load_file: str = str(PROJECT_ROOT / "data" / "load" / "test_complex_case.dat")

    output_dir: str = str(PROJECT_ROOT / "results" / "student" / "uniform_phi_plot_compare")
    case_name: str = "uniform_phi_plot_compare"

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

    phi0_deg: float = 0.0
    phi1_deg: float = -7.5

    torch_dtype: str = "float64"
    device: str = "cpu"
    linear_solve_mode: str = "solve"

    remove_initial_offset: bool = True
    show_plot: bool = False


def parse_args() -> UniformPhiPlotCompareConfig:
    d = UniformPhiPlotCompareConfig()
    parser = argparse.ArgumentParser(
        description="Compare teacher / student(phi=0) / student(phi=-7.5) tip displacement time histories and save x/y plots."
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
    parser.add_argument("--phi1-deg", type=float, default=d.phi1_deg)

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

    return UniformPhiPlotCompareConfig(
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
        phi1_deg=args.phi1_deg,
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


def rollout_with_uniform_phi(
    *,
    model,
    phi_deg: float,
    F_time: np.ndarray,
    u0: np.ndarray,
    v0: np.ndarray,
    cfg: UniformPhiPlotCompareConfig,
    dtype: torch.dtype,
    device: torch.device,
) -> np.ndarray:
    M, K, _ = build_coupled_fem_matrices_6dof_degrees(
        model,
        section_params=None,
        phi_deg=float(phi_deg),
        alpha_flap=cfg.alpha_flap,
        alpha_edge=cfg.alpha_edge,
        alpha_torsion=cfg.alpha_torsion,
        rotate_mass=False,
        return_full=True,
    )

    C, _, _ = build_damping_like_student(
        M=M,
        K=K,
        zeta_structural=cfg.zeta_structural,
        ref_freq_hz=cfg.ref_freq_hz,
    )

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
    phi0_deg: float,
    phi1_deg: float,
    save_path: Path,
    case_name: str,
) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(time, teacher_signal, label="Teacher")
    plt.plot(time, phi0_signal, label=f"Student phi={phi0_deg:g}°")
    plt.plot(time, phi1_signal, label=f"Student phi={phi1_deg:g}°")
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
    print("[Uniform Phi Plot Compare]")
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

    print()
    print("[4/4] Running student with phi=0 and phi=-7.5")
    model = load_student_model_from_blade_master(
        csv_path=str(blade_csv),
        model_name="uniform_phi_plot_compare_model",
    )

    u_phi0 = rollout_with_uniform_phi(
        model=model,
        phi_deg=cfg.phi0_deg,
        F_time=F_time,
        u0=u_direct[0].copy(),
        v0=v_direct[0].copy(),
        cfg=cfg,
        dtype=dtype,
        device=device,
    )
    u_phi1 = rollout_with_uniform_phi(
        model=model,
        phi_deg=cfg.phi1_deg,
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

    plot_one_direction(
        time=time_teacher,
        teacher_signal=teacher_tip_x,
        phi0_signal=phi0_tip_x,
        phi1_signal=phi1_tip_x,
        direction="x",
        phi0_deg=cfg.phi0_deg,
        phi1_deg=cfg.phi1_deg,
        save_path=x_plot_path,
        case_name=cfg.case_name,
    )

    plot_one_direction(
        time=time_teacher,
        teacher_signal=teacher_tip_y,
        phi0_signal=phi0_tip_y,
        phi1_signal=phi1_tip_y,
        direction="y",
        phi0_deg=cfg.phi0_deg,
        phi1_deg=cfg.phi1_deg,
        save_path=y_plot_path,
        case_name=cfg.case_name,
    )

    print()
    print("[Saved Figures]")
    print(f"  x plot: {x_plot_path}")
    print(f"  y plot: {y_plot_path}")

    if cfg.show_plot:
        plt.figure(figsize=(12, 6))
        plt.plot(time_teacher, teacher_tip_x, label="Teacher")
        plt.plot(time_teacher, phi0_tip_x, label=f"Student phi={cfg.phi0_deg:g}°")
        plt.plot(time_teacher, phi1_tip_x, label=f"Student phi={cfg.phi1_deg:g}°")
        plt.xlabel("Time [s]")
        plt.ylabel("Tip displacement in x direction")
        plt.title(f"{cfg.case_name} - Tip x-direction displacement time history")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(time_teacher, teacher_tip_y, label="Teacher")
        plt.plot(time_teacher, phi0_tip_y, label=f"Student phi={cfg.phi0_deg:g}°")
        plt.plot(time_teacher, phi1_tip_y, label=f"Student phi={cfg.phi1_deg:g}°")
        plt.xlabel("Time [s]")
        plt.ylabel("Tip displacement in y direction")
        plt.title(f"{cfg.case_name} - Tip y-direction displacement time history")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    print()
    print("✅ PASS: 已输出 x / y 两个方向的三曲线时历图。")


if __name__ == "__main__":
    main()