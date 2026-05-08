from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
class KappaYScaleFrequencyScanConfig:
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

    output_dir: str = str(PROJECT_ROOT / "results" / "student" / "kappa_y_stiffness_scale_frequency_scan")
    case_name: str = "kappa_y_stiffness_scale_frequency_scan"

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

    scale_values: str = "1.00,0.98,0.96,0.94,0.92,0.90"
    scale_mode: str = "y_bending"  # choices: uy_only, y_bending

    torch_dtype: str = "float64"
    device: str = "cpu"
    linear_solve_mode: str = "solve"

    remove_initial_offset: bool = True
    fft_t_start: float = 0.5
    fft_t_end: Optional[float] = None
    min_freq_hz: float = 0.05
    max_freq_hz: float = 5.0
    last_k: int = 5

    show_plot: bool = False
    save_report: bool = True


def parse_args() -> KappaYScaleFrequencyScanConfig:
    d = KappaYScaleFrequencyScanConfig()

    parser = argparse.ArgumentParser(
        description=(
            "Scan static y-direction stiffness scaling and diagnose whether reducing y stiffness "
            "moves current base student y dominant frequency toward BeamDyn teacher."
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

    parser.add_argument("--base-phi-twist-column", type=str, default=d.base_phi_twist_column)
    parser.add_argument("--base-phi-sign", type=float, default=d.base_phi_sign)

    rotate_group = parser.add_mutually_exclusive_group()
    rotate_group.add_argument("--rotate-mass", dest="rotate_mass", action="store_true")
    rotate_group.add_argument("--no-rotate-mass", dest="rotate_mass", action="store_false")
    parser.set_defaults(rotate_mass=d.rotate_mass)

    parser.add_argument("--scale-values", type=str, default=d.scale_values)
    parser.add_argument("--scale-mode", type=str, default=d.scale_mode, choices=["uy_only", "y_bending"])

    parser.add_argument("--torch-dtype", type=str, default=d.torch_dtype, choices=["float64", "float32"])
    parser.add_argument("--device", type=str, default=d.device)
    parser.add_argument("--linear-solve-mode", type=str, default=d.linear_solve_mode, choices=["solve", "inverse"])

    offset_group = parser.add_mutually_exclusive_group()
    offset_group.add_argument("--remove-initial-offset", dest="remove_initial_offset", action="store_true")
    offset_group.add_argument("--keep-initial-offset", dest="remove_initial_offset", action="store_false")
    parser.set_defaults(remove_initial_offset=d.remove_initial_offset)

    parser.add_argument("--fft-t-start", type=float, default=d.fft_t_start)
    parser.add_argument("--fft-t-end", type=float, default=d.fft_t_end)
    parser.add_argument("--min-freq-hz", type=float, default=d.min_freq_hz)
    parser.add_argument("--max-freq-hz", type=float, default=d.max_freq_hz)
    parser.add_argument("--last-k", type=int, default=d.last_k)

    show_group = parser.add_mutually_exclusive_group()
    show_group.add_argument("--show-plot", dest="show_plot", action="store_true")
    show_group.add_argument("--no-show-plot", dest="show_plot", action="store_false")
    parser.set_defaults(show_plot=d.show_plot)

    report_group = parser.add_mutually_exclusive_group()
    report_group.add_argument("--save-report", dest="save_report", action="store_true")
    report_group.add_argument("--no-save-report", dest="save_report", action="store_false")
    parser.set_defaults(save_report=d.save_report)

    args = parser.parse_args()

    return KappaYScaleFrequencyScanConfig(
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
        base_phi_twist_column=args.base_phi_twist_column,
        base_phi_sign=args.base_phi_sign,
        rotate_mass=args.rotate_mass,
        scale_values=args.scale_values,
        scale_mode=args.scale_mode,
        torch_dtype=args.torch_dtype,
        device=args.device,
        linear_solve_mode=args.linear_solve_mode,
        remove_initial_offset=args.remove_initial_offset,
        fft_t_start=args.fft_t_start,
        fft_t_end=args.fft_t_end,
        min_freq_hz=args.min_freq_hz,
        max_freq_hz=args.max_freq_hz,
        last_k=args.last_k,
        show_plot=args.show_plot,
        save_report=args.save_report,
    )


def assert_existing_file(path: str | Path, label: str) -> Path:
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {p}")
    return p


def parse_scale_values(text: str) -> List[float]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError("scale_values is empty.")
    return values


def get_torch_dtype(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def remove_initial_offset(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    return u - u[:1, :]


def load_column_from_csv(csv_path: Path, column_name: str) -> np.ndarray:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in CSV: {csv_path}")
        if column_name not in reader.fieldnames:
            raise KeyError(
                f"Column '{column_name}' not found in {csv_path}. "
                f"Available columns: {reader.fieldnames}"
            )

        values = []
        for row in reader:
            values.append(float(row[column_name]))

    return np.asarray(values, dtype=np.float64)


def build_element_phi_from_initial_twist(
    *,
    blade_csv: Path,
    twist_column: str,
    sign: float,
    n_elements: int,
) -> np.ndarray:
    station_twist = load_column_from_csv(blade_csv, twist_column)

    if station_twist.size == n_elements + 1:
        element_twist = 0.5 * (station_twist[:-1] + station_twist[1:])
    elif station_twist.size == n_elements:
        element_twist = station_twist.copy()
    else:
        raise ValueError(
            f"Unexpected twist length. Got {station_twist.size}, "
            f"expected n_elements={n_elements} or n_elements+1={n_elements + 1}."
        )

    return float(sign) * element_twist


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


def selected_y_dof_indices(n_dofs: int, scale_mode: str) -> np.ndarray:
    if n_dofs % 6 != 0:
        raise ValueError(f"n_dofs must be divisible by 6, got {n_dofs}")

    n_nodes = n_dofs // 6
    uy = np.array([i * 6 + 1 for i in range(n_nodes)], dtype=int)

    if scale_mode == "uy_only":
        return uy

    if scale_mode == "y_bending":
        # 对沿 z 展向的梁，y 向平动通常和绕 x 的截面转角共同描述 y 弯曲。
        # 这里同时缩放 uy 和 rx，用于诊断 y 主导弯曲刚度。
        rx = np.array([i * 6 + 3 for i in range(n_nodes)], dtype=int)
        return np.unique(np.concatenate([uy, rx]))

    raise ValueError(f"Unsupported scale_mode: {scale_mode}")


def apply_congruent_dof_stiffness_scale(
    K: np.ndarray,
    *,
    scale: float,
    dof_indices: np.ndarray,
) -> np.ndarray:
    if scale <= 0.0:
        raise ValueError(f"scale must be positive, got {scale}")

    K = np.asarray(K, dtype=np.float64)
    s = np.ones(K.shape[0], dtype=np.float64)
    s[dof_indices] = np.sqrt(float(scale))

    # K_scaled = S K S，保持对称性。
    K_scaled = (s[:, None] * K) * s[None, :]
    return 0.5 * (K_scaled + K_scaled.T)


def component_indices(n_nodes: int, component: str) -> np.ndarray:
    mapping = {
        "x": 0,
        "y": 1,
        "z": 2,
        "rx": 3,
        "ry": 4,
        "rz": 5,
    }
    if component not in mapping:
        raise ValueError(f"Unsupported component: {component}")
    return np.array([i * 6 + mapping[component] for i in range(n_nodes)], dtype=int)


def get_lastk_component_mean(
    u_full: np.ndarray,
    component: str,
    *,
    last_k: int,
) -> np.ndarray:
    u_full = np.asarray(u_full, dtype=np.float64)
    n_dofs = u_full.shape[1]
    if n_dofs % 6 != 0:
        raise ValueError(f"n_dofs must be divisible by 6, got {n_dofs}")

    n_nodes = n_dofs // 6
    if last_k <= 0 or last_k > n_nodes:
        raise ValueError(f"Invalid last_k={last_k}, n_nodes={n_nodes}")

    idx = component_indices(n_nodes, component)
    comp = u_full[:, idx]
    return np.mean(comp[:, -last_k:], axis=1)


def select_time_window(
    time: np.ndarray,
    signal: np.ndarray,
    *,
    t_start: float,
    t_end: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    time = np.asarray(time, dtype=np.float64).reshape(-1)
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)

    if t_end is None:
        mask = time >= float(t_start)
    else:
        mask = (time >= float(t_start)) & (time <= float(t_end))

    if np.count_nonzero(mask) < 8:
        raise ValueError("FFT window too short.")

    return time[mask], signal[mask]


def estimate_dominant_frequency_fft(
    time: np.ndarray,
    signal: np.ndarray,
    *,
    min_freq_hz: float,
    max_freq_hz: float,
    t_start: float,
    t_end: Optional[float],
) -> Dict[str, Any]:
    time_w, sig_w = select_time_window(
        time=time,
        signal=signal,
        t_start=t_start,
        t_end=t_end,
    )

    dt = float(np.median(np.diff(time_w)))
    sig = np.asarray(sig_w, dtype=np.float64)
    sig = sig - np.mean(sig)

    rms = float(np.sqrt(np.mean(sig ** 2)))
    freqs = np.fft.rfftfreq(sig.size, d=dt)

    if rms <= 1.0e-14:
        return {
            "dominant_freq_hz": float("nan"),
            "dominant_amp": 0.0,
            "rms": rms,
            "freqs": freqs,
            "spectrum": np.zeros_like(freqs),
            "top_freqs_hz": [],
            "top_amps": [],
        }

    window = np.hanning(sig.size)
    spectrum = np.abs(np.fft.rfft(sig * window))

    mask = (freqs >= float(min_freq_hz)) & (freqs <= float(max_freq_hz))
    if not np.any(mask):
        raise ValueError(f"No FFT bins in [{min_freq_hz}, {max_freq_hz}] Hz.")

    candidate_indices = np.where(mask)[0]
    local_spectrum = spectrum[candidate_indices]
    peak_idx = int(candidate_indices[int(np.argmax(local_spectrum))])

    freq_est = float(freqs[peak_idx])
    amp_est = float(spectrum[peak_idx])

    if 1 <= peak_idx < len(spectrum) - 1:
        y0 = float(spectrum[peak_idx - 1])
        y1 = float(spectrum[peak_idx])
        y2 = float(spectrum[peak_idx + 1])
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1.0e-30:
            delta = 0.5 * (y0 - y2) / denom
            df = float(freqs[1] - freqs[0])
            if abs(delta) <= 1.0:
                freq_est = float(freqs[peak_idx] + delta * df)

    top_n = min(5, candidate_indices.size)
    sorted_local = np.argsort(local_spectrum)[::-1][:top_n]
    top_indices = candidate_indices[sorted_local]

    return {
        "dominant_freq_hz": freq_est,
        "dominant_amp": amp_est,
        "rms": rms,
        "freqs": freqs,
        "spectrum": spectrum,
        "top_freqs_hz": [float(freqs[i]) for i in top_indices],
        "top_amps": [float(spectrum[i]) for i in top_indices],
    }


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
) -> np.ndarray:
    core = FullOrderCorrectedCoreTorch(
        M=M,
        K=K,
        C=C,
        dt=float(dt),
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
        u_t, _, _ = core.rollout(
            F_time=F_torch,
            u0=u0_torch,
            v0=v0_torch,
            theta_full=None,
            return_debug=False,
        )

    return u_t.detach().cpu().numpy()


def plot_scan(
    *,
    scales: List[float],
    freqs: List[float],
    teacher_freq: float,
    save_path: Path,
    show_plot: bool,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(scales, freqs, marker="o", label="Student scaled y frequency")
    plt.axhline(teacher_freq, linestyle="--", label=f"Teacher {teacher_freq:.6f} Hz")
    plt.xlabel("y stiffness scale")
    plt.ylabel("Dominant y frequency [Hz]")
    plt.title("y stiffness scale vs dominant y frequency")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    if show_plot:
        plt.show()
    plt.close()


def to_jsonable_fft(x: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "dominant_freq_hz": float(x["dominant_freq_hz"]),
        "dominant_amp": float(x["dominant_amp"]),
        "rms": float(x["rms"]),
        "top_freqs_hz": [float(v) for v in x["top_freqs_hz"]],
        "top_amps": [float(v) for v in x["top_amps"]],
    }


def main() -> None:
    cfg = parse_args()

    teacher_exe = assert_existing_file(cfg.teacher_exe, "teacher_exe")
    template_inp = assert_existing_file(cfg.template_inp, "template_inp")
    blade_csv = assert_existing_file(cfg.blade_csv, "blade_csv")
    load_file = assert_existing_file(cfg.load_file, "load_file")

    output_dir = Path(cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scale_values = parse_scale_values(cfg.scale_values)

    dtype = get_torch_dtype(cfg.torch_dtype)
    device = torch.device(cfg.device)

    print()
    print("[Kappa-y Stiffness Scale Frequency Scan]")
    print()
    print("[Config]")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")
    print(f"  parsed_scale_values: {scale_values}")

    print()
    print("[1/7] Running BeamDyn teacher")
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
    print("[2/7] Loading teacher response")
    time_teacher, u_teacher, _ = load_teacher_6dof_response(
        out_path=teacher_out,
        node_start=cfg.teacher_node_start,
        node_end=cfg.teacher_node_end,
        demean=cfg.teacher_demean,
    )
    print(f"  time_teacher shape = {time_teacher.shape}")
    print(f"  u_teacher shape    = {u_teacher.shape}")

    print()
    print("[3/7] Running current base student once to get F_time and initial state")
    base_result = run_student_case(
        blade_csv=blade_csv,
        output_dir=output_dir / "student_base",
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

    time_student = np.asarray(base_result["time"], dtype=np.float64)
    F_time = np.asarray(base_result["F_time"], dtype=np.float64)
    u_base = np.asarray(base_result["u_full"], dtype=np.float64)
    v_base = np.asarray(base_result["v_full"], dtype=np.float64)

    print(f"  time_student shape = {time_student.shape}")
    print(f"  F_time shape       = {F_time.shape}")
    print(f"  u_base shape       = {u_base.shape}")

    print()
    print("[4/7] Building base M/K using Phi_base = -initial_twist_deg(s)")
    model = load_student_model_from_blade_master(
        csv_path=str(blade_csv),
        model_name="kappa_y_scale_frequency_scan_model",
    )

    n_elements = int(model.n_stations - 1)
    phi_element_deg = build_element_phi_from_initial_twist(
        blade_csv=blade_csv,
        twist_column=cfg.base_phi_twist_column,
        sign=cfg.base_phi_sign,
        n_elements=n_elements,
    )

    print(f"  n_elements = {n_elements}")
    print(
        "  phi_element_deg: "
        f"min={phi_element_deg.min():.6f}, "
        f"max={phi_element_deg.max():.6f}, "
        f"mean={phi_element_deg.mean():.6f}"
    )

    M_base, K_base, _ = build_coupled_fem_matrices_6dof_degrees(
        model,
        section_params=None,
        phi_deg=phi_element_deg,
        alpha_flap=cfg.alpha_flap,
        alpha_edge=cfg.alpha_edge,
        alpha_torsion=cfg.alpha_torsion,
        rotate_mass=cfg.rotate_mass,
        return_full=True,
    )

    M_base = np.asarray(M_base, dtype=np.float64)
    K_base = np.asarray(K_base, dtype=np.float64)

    print(f"  M_base shape = {M_base.shape}")
    print(f"  K_base shape = {K_base.shape}")

    y_dof_idx = selected_y_dof_indices(K_base.shape[0], cfg.scale_mode)
    print(f"  scale_mode = {cfg.scale_mode}")
    print(f"  n_scaled_dofs = {y_dof_idx.size}")

    print()
    print("[5/7] Preparing teacher target y frequency")
    u_teacher_rs = resample_response_to_time_grid(
        time_src=time_teacher,
        u_src=u_teacher,
        time_dst=time_student,
    )

    if cfg.remove_initial_offset:
        u_teacher_eval = remove_initial_offset(u_teacher_rs)
    else:
        u_teacher_eval = u_teacher_rs

    teacher_tip_y = get_tip_component(u_teacher_eval, "y")
    teacher_lastk_y = get_lastk_component_mean(
        u_teacher_eval,
        "y",
        last_k=cfg.last_k,
    )

    teacher_tip_fft = estimate_dominant_frequency_fft(
        time=time_student,
        signal=teacher_tip_y,
        min_freq_hz=cfg.min_freq_hz,
        max_freq_hz=cfg.max_freq_hz,
        t_start=cfg.fft_t_start,
        t_end=cfg.fft_t_end,
    )
    teacher_lastk_fft = estimate_dominant_frequency_fft(
        time=time_student,
        signal=teacher_lastk_y,
        min_freq_hz=cfg.min_freq_hz,
        max_freq_hz=cfg.max_freq_hz,
        t_start=cfg.fft_t_start,
        t_end=cfg.fft_t_end,
    )

    teacher_tip_freq = float(teacher_tip_fft["dominant_freq_hz"])
    teacher_lastk_freq = float(teacher_lastk_fft["dominant_freq_hz"])

    print(f"  teacher tip y freq       = {teacher_tip_freq:.8e} Hz")
    print(f"  teacher last-{cfg.last_k} y freq = {teacher_lastk_freq:.8e} Hz")

    print()
    print("[6/7] Scanning y stiffness scales")

    scan_results: List[Dict[str, Any]] = []

    for scale in scale_values:
        print()
        print(f"[Scale] y stiffness scale = {scale:.6f}")

        K_scaled = apply_congruent_dof_stiffness_scale(
            K_base,
            scale=scale,
            dof_indices=y_dof_idx,
        )

        C_scaled, ref_freq_used, natural_freqs = build_damping_like_student(
            M=M_base,
            K=K_scaled,
            zeta_structural=cfg.zeta_structural,
            ref_freq_hz=cfg.ref_freq_hz,
        )

        u_scaled = rollout_with_mkc(
            M=M_base,
            K=K_scaled,
            C=C_scaled,
            F_time=F_time,
            u0=u_base[0].copy(),
            v0=v_base[0].copy(),
            dt=cfg.dt,
            dtype=dtype,
            device=device,
            linear_solve_mode=cfg.linear_solve_mode,
        )

        if cfg.remove_initial_offset:
            u_eval = remove_initial_offset(u_scaled)
        else:
            u_eval = u_scaled

        student_tip_y = get_tip_component(u_eval, "y")
        student_lastk_y = get_lastk_component_mean(
            u_eval,
            "y",
            last_k=cfg.last_k,
        )

        tip_fft = estimate_dominant_frequency_fft(
            time=time_student,
            signal=student_tip_y,
            min_freq_hz=cfg.min_freq_hz,
            max_freq_hz=cfg.max_freq_hz,
            t_start=cfg.fft_t_start,
            t_end=cfg.fft_t_end,
        )
        lastk_fft = estimate_dominant_frequency_fft(
            time=time_student,
            signal=student_lastk_y,
            min_freq_hz=cfg.min_freq_hz,
            max_freq_hz=cfg.max_freq_hz,
            t_start=cfg.fft_t_start,
            t_end=cfg.fft_t_end,
        )

        tip_freq = float(tip_fft["dominant_freq_hz"])
        lastk_freq = float(lastk_fft["dominant_freq_hz"])

        tip_ratio = tip_freq / teacher_tip_freq
        lastk_ratio = lastk_freq / teacher_lastk_freq

        print(f"  natural_freqs[:5] = {natural_freqs[:5]}")
        print(f"  ref_freq_used     = {ref_freq_used}")
        print(f"  tip_y freq        = {tip_freq:.8e} Hz, ratio={tip_ratio:.8e}")
        print(f"  last{cfg.last_k}_y freq    = {lastk_freq:.8e} Hz, ratio={lastk_ratio:.8e}")

        scan_results.append(
            {
                "scale": float(scale),
                "ref_freq_used": None if ref_freq_used is None else float(ref_freq_used),
                "natural_freqs_hz": [float(x) for x in natural_freqs.tolist()],
                "tip_y": {
                    "student_fft": to_jsonable_fft(tip_fft),
                    "teacher_freq_hz": teacher_tip_freq,
                    "student_over_teacher": float(tip_ratio),
                    "relative_error": float(tip_ratio - 1.0),
                    "abs_error_hz": float(tip_freq - teacher_tip_freq),
                },
                "lastk_mean_y": {
                    "last_k": int(cfg.last_k),
                    "student_fft": to_jsonable_fft(lastk_fft),
                    "teacher_freq_hz": teacher_lastk_freq,
                    "student_over_teacher": float(lastk_ratio),
                    "relative_error": float(lastk_ratio - 1.0),
                    "abs_error_hz": float(lastk_freq - teacher_lastk_freq),
                },
            }
        )

    print()
    print("[7/7] Ranking scales by last-k y frequency error")
    ranked = sorted(
        scan_results,
        key=lambda x: abs(float(x["lastk_mean_y"]["relative_error"])),
    )

    print()
    print("[Ranking]")
    print("  rank  scale      tip_ratio     lastk_ratio   lastk_abs_err_hz")
    for i, item in enumerate(ranked, start=1):
        print(
            f"  {i:4d}  "
            f"{item['scale']:8.4f}  "
            f"{item['tip_y']['student_over_teacher']:12.8f}  "
            f"{item['lastk_mean_y']['student_over_teacher']:12.8f}  "
            f"{item['lastk_mean_y']['abs_error_hz']:16.8e}"
        )

    best = ranked[0]

    print()
    print("[Best Scale by Last-k y Frequency]")
    print(f"  scale = {best['scale']:.6f}")
    print(f"  tip student/teacher   = {best['tip_y']['student_over_teacher']:.8e}")
    print(f"  last-k student/teacher = {best['lastk_mean_y']['student_over_teacher']:.8e}")

    scale_plot_path = output_dir / "scale_vs_y_frequency.png"
    plot_scan(
        scales=[x["scale"] for x in scan_results],
        freqs=[x["lastk_mean_y"]["student_fft"]["dominant_freq_hz"] for x in scan_results],
        teacher_freq=teacher_lastk_freq,
        save_path=scale_plot_path,
        show_plot=cfg.show_plot,
    )

    report = {
        "config": asdict(cfg),
        "teacher": {
            "teacher_out": str(teacher_out),
            "tip_y_fft": to_jsonable_fft(teacher_tip_fft),
            "lastk_mean_y_fft": to_jsonable_fft(teacher_lastk_fft),
        },
        "base_student": {
            "student_npz": str(base_result["npz"]),
            "student_yaml": str(base_result["yaml"]),
            "phi_element_deg_min": float(phi_element_deg.min()),
            "phi_element_deg_max": float(phi_element_deg.max()),
            "phi_element_deg_mean": float(phi_element_deg.mean()),
        },
        "scan_results": scan_results,
        "best_by_lastk_y_frequency": best,
        "figures": {
            "scale_vs_y_frequency": str(scale_plot_path),
        },
        "interpretation": {
            "purpose": "Diagnostic only. This script uses K_scaled = S K S for symmetric DOF-level stiffness scaling.",
            "if_best_scale_less_than_1": "Reducing y-bending stiffness moves student y frequency toward teacher.",
            "next_step": "If scan is monotonic and best scale is around expected value, implement section/element-level kappa_y_residual(s) in static training.",
        },
    }

    if cfg.save_report:
        report_path = output_dir / "kappa_y_stiffness_scale_frequency_scan_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print()
        print("[Saved Report]")
        print(f"  {report_path}")

    print()
    print("[Saved Figures]")
    print(f"  scale plot = {scale_plot_path}")

    print()
    print("✅ PASS: kappa-y stiffness scale frequency scan completed.")
    print("   请查看 best scale 是否低于 1；若是，说明降低 y 主导刚度确实能把 y 主频拉近 teacher。")


if __name__ == "__main__":
    main()