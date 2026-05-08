from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from scripts.run_student_cases import run_student_case  # noqa: E402
from src.teacher.beamdyn_teacher_adapter import (  # noqa: E402
    BeamDynTeacherCaseConfig,
    run_teacher_case_beamdyn,
    load_teacher_6dof_response,
    resample_response_to_time_grid,
    get_tip_component,
)


@dataclass
class YFrequencyDiagnosticConfig:
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

    # 默认先放 simple_tip_fy_case；你可以运行时用 --load-file 换成你的 y 初始载荷 dat
    load_file: str = str(PROJECT_ROOT / "data" / "load" / "simple_tip_fy_case.dat")

    output_dir: str = str(PROJECT_ROOT / "results" / "student" / "y_frequency_teacher_student_diagnostic")
    case_name: str = "y_frequency_teacher_student_diagnostic"

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

    remove_initial_offset: bool = True

    # 频率分析窗口。若是冲击/初始载荷，可从 0.2~0.5s 后开始，减少载荷突变影响。
    fft_t_start: float = 0.5
    fft_t_end: Optional[float] = None
    min_freq_hz: float = 0.05
    max_freq_hz: float = 5.0
    last_k: int = 5

    show_plot: bool = False
    save_report: bool = True


def parse_args() -> YFrequencyDiagnosticConfig:
    d = YFrequencyDiagnosticConfig()

    parser = argparse.ArgumentParser(
        description=(
            "Diagnose dominant y-direction response frequency of BeamDyn teacher "
            "and current base student under a y-only initial/load dat file."
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

    return YFrequencyDiagnosticConfig(
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


def remove_initial_offset(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    return u - u[:1, :]


def component_indices(n_nodes: int, component: str) -> np.ndarray:
    component_to_offset = {
        "x": 0,
        "y": 1,
        "z": 2,
        "rx": 3,
        "ry": 4,
        "rz": 5,
    }
    if component not in component_to_offset:
        raise ValueError(f"Unsupported component: {component}")
    offset = component_to_offset[component]
    return np.array([i * 6 + offset for i in range(n_nodes)], dtype=int)


def get_lastk_component_mean(
    u_full: np.ndarray,
    component: str,
    *,
    last_k: int,
) -> np.ndarray:
    u_full = np.asarray(u_full, dtype=np.float64)
    if u_full.ndim != 2:
        raise ValueError(f"u_full must be 2D, got shape={u_full.shape}")

    n_dofs = u_full.shape[1]
    if n_dofs % 6 != 0:
        raise ValueError(f"n_dofs must be divisible by 6, got {n_dofs}")

    n_nodes = n_dofs // 6
    last_k = int(last_k)
    if last_k <= 0:
        raise ValueError("last_k must be positive.")
    if last_k > n_nodes:
        raise ValueError(f"last_k={last_k} is larger than n_nodes={n_nodes}")

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

    if time.shape[0] != signal.shape[0]:
        raise ValueError(f"time/signal length mismatch: {time.shape[0]} vs {signal.shape[0]}")

    if t_end is None:
        mask = time >= float(t_start)
    else:
        mask = (time >= float(t_start)) & (time <= float(t_end))

    if np.count_nonzero(mask) < 8:
        raise ValueError(
            f"FFT window too short: only {np.count_nonzero(mask)} samples. "
            f"Check fft_t_start={t_start}, fft_t_end={t_end}."
        )

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
    if dt <= 0.0:
        raise ValueError(f"Invalid dt estimated from time window: {dt}")

    sig = np.asarray(sig_w, dtype=np.float64)
    sig = sig - np.mean(sig)

    rms = float(np.sqrt(np.mean(sig ** 2)))
    if rms <= 1.0e-14:
        freqs = np.fft.rfftfreq(sig.size, d=dt)
        return {
            "dominant_freq_hz": float("nan"),
            "dominant_amp": 0.0,
            "rms": rms,
            "dt": dt,
            "n_samples": int(sig.size),
            "freqs": freqs,
            "spectrum": np.zeros_like(freqs),
            "top_freqs_hz": [],
            "top_amps": [],
        }

    window = np.hanning(sig.size)
    sig_windowed = sig * window

    freqs = np.fft.rfftfreq(sig_windowed.size, d=dt)
    spectrum = np.abs(np.fft.rfft(sig_windowed))

    mask = (freqs >= float(min_freq_hz)) & (freqs <= float(max_freq_hz))
    if not np.any(mask):
        raise ValueError(
            f"No FFT bins in frequency range [{min_freq_hz}, {max_freq_hz}] Hz. "
            f"Try increasing t_final or changing frequency range."
        )

    candidate_indices = np.where(mask)[0]
    local_spectrum = spectrum[candidate_indices]
    peak_local = int(np.argmax(local_spectrum))
    peak_idx = int(candidate_indices[peak_local])

    freq_est = float(freqs[peak_idx])
    amp_est = float(spectrum[peak_idx])

    # 简单三点抛物线插值，提高频率估计精度
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

    # 输出前 5 个较大频率峰，辅助判断是否存在多模态混叠
    top_n = min(5, candidate_indices.size)
    sorted_local = np.argsort(local_spectrum)[::-1][:top_n]
    top_indices = candidate_indices[sorted_local]
    top_freqs = [float(freqs[i]) for i in top_indices]
    top_amps = [float(spectrum[i]) for i in top_indices]

    return {
        "dominant_freq_hz": freq_est,
        "dominant_amp": amp_est,
        "rms": rms,
        "dt": dt,
        "n_samples": int(sig.size),
        "freqs": freqs,
        "spectrum": spectrum,
        "top_freqs_hz": top_freqs,
        "top_amps": top_amps,
    }


def compare_frequency(
    teacher_freq: float,
    student_freq: float,
) -> Dict[str, float]:
    if not np.isfinite(teacher_freq) or abs(teacher_freq) < 1.0e-14:
        return {
            "teacher_freq_hz": float(teacher_freq),
            "student_freq_hz": float(student_freq),
            "student_over_teacher": float("nan"),
            "relative_error": float("nan"),
            "absolute_error_hz": float("nan"),
        }

    return {
        "teacher_freq_hz": float(teacher_freq),
        "student_freq_hz": float(student_freq),
        "student_over_teacher": float(student_freq / teacher_freq),
        "relative_error": float((student_freq - teacher_freq) / teacher_freq),
        "absolute_error_hz": float(student_freq - teacher_freq),
    }


def plot_time_history(
    *,
    time: np.ndarray,
    teacher_signal: np.ndarray,
    student_signal: np.ndarray,
    title: str,
    ylabel: str,
    save_path: Path,
    show_plot: bool,
) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(time, teacher_signal, label="Teacher")
    plt.plot(time, student_signal, label="Student base")
    plt.xlabel("Time [s]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    if show_plot:
        plt.show()
    plt.close()


def plot_spectrum(
    *,
    teacher_fft: Dict[str, Any],
    student_fft: Dict[str, Any],
    title: str,
    save_path: Path,
    show_plot: bool,
    min_freq_hz: float,
    max_freq_hz: float,
) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(teacher_fft["freqs"], teacher_fft["spectrum"], label="Teacher spectrum")
    plt.plot(student_fft["freqs"], student_fft["spectrum"], label="Student spectrum")
    plt.axvline(
        teacher_fft["dominant_freq_hz"],
        linestyle="--",
        label=f"Teacher peak {teacher_fft['dominant_freq_hz']:.4f} Hz",
    )
    plt.axvline(
        student_fft["dominant_freq_hz"],
        linestyle=":",
        label=f"Student peak {student_fft['dominant_freq_hz']:.4f} Hz",
    )
    plt.xlim(min_freq_hz, max_freq_hz)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("FFT amplitude")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    if show_plot:
        plt.show()
    plt.close()


def to_jsonable_fft(fft_result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "dominant_freq_hz": float(fft_result["dominant_freq_hz"]),
        "dominant_amp": float(fft_result["dominant_amp"]),
        "rms": float(fft_result["rms"]),
        "dt": float(fft_result["dt"]),
        "n_samples": int(fft_result["n_samples"]),
        "top_freqs_hz": [float(x) for x in fft_result["top_freqs_hz"]],
        "top_amps": [float(x) for x in fft_result["top_amps"]],
    }


def main() -> None:
    cfg = parse_args()

    teacher_exe = assert_existing_file(cfg.teacher_exe, "teacher_exe")
    template_inp = assert_existing_file(cfg.template_inp, "template_inp")
    blade_csv = assert_existing_file(cfg.blade_csv, "blade_csv")
    load_file = assert_existing_file(cfg.load_file, "load_file")

    output_dir = Path(cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("[Y Frequency Teacher-Student Diagnostic]")
    print()
    print("[Config]")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")

    print()
    print("[1/6] Running BeamDyn teacher")
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
    print("[2/6] Loading teacher response")
    time_teacher, u_teacher, teacher_columns = load_teacher_6dof_response(
        out_path=teacher_out,
        node_start=cfg.teacher_node_start,
        node_end=cfg.teacher_node_end,
        demean=cfg.teacher_demean,
    )
    print(f"  time_teacher shape = {time_teacher.shape}")
    print(f"  u_teacher shape    = {u_teacher.shape}")
    print(f"  first columns      = {teacher_columns[:6]}")
    print(f"  last columns       = {teacher_columns[-6:]}")

    print()
    print("[3/6] Running current base student")
    student_result = run_student_case(
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

    time_student = np.asarray(student_result["time"], dtype=np.float64)
    u_student = np.asarray(student_result["u_full"], dtype=np.float64)

    print(f"  time_student shape = {time_student.shape}")
    print(f"  u_student shape    = {u_student.shape}")

    print()
    print("[4/6] Resampling teacher response to student time grid")
    u_teacher_rs = resample_response_to_time_grid(
        time_src=time_teacher,
        u_src=u_teacher,
        time_dst=time_student,
    )

    if cfg.remove_initial_offset:
        u_teacher_eval = remove_initial_offset(u_teacher_rs)
        u_student_eval = remove_initial_offset(u_student)
    else:
        u_teacher_eval = u_teacher_rs
        u_student_eval = u_student

    print(f"  u_teacher_rs shape = {u_teacher_rs.shape}")
    print(f"  remove_initial_offset = {cfg.remove_initial_offset}")

    print()
    print("[5/6] Extracting y-direction signals")

    teacher_tip_y = get_tip_component(u_teacher_eval, "y")
    student_tip_y = get_tip_component(u_student_eval, "y")

    teacher_lastk_y = get_lastk_component_mean(
        u_teacher_eval,
        "y",
        last_k=cfg.last_k,
    )
    student_lastk_y = get_lastk_component_mean(
        u_student_eval,
        "y",
        last_k=cfg.last_k,
    )

    print(f"  tip_y signal length      = {teacher_tip_y.size}")
    print(f"  last{cfg.last_k}_y signal length = {teacher_lastk_y.size}")

    print()
    print("[6/6] Estimating dominant frequencies by FFT")

    teacher_tip_fft = estimate_dominant_frequency_fft(
        time=time_student,
        signal=teacher_tip_y,
        min_freq_hz=cfg.min_freq_hz,
        max_freq_hz=cfg.max_freq_hz,
        t_start=cfg.fft_t_start,
        t_end=cfg.fft_t_end,
    )
    student_tip_fft = estimate_dominant_frequency_fft(
        time=time_student,
        signal=student_tip_y,
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
    student_lastk_fft = estimate_dominant_frequency_fft(
        time=time_student,
        signal=student_lastk_y,
        min_freq_hz=cfg.min_freq_hz,
        max_freq_hz=cfg.max_freq_hz,
        t_start=cfg.fft_t_start,
        t_end=cfg.fft_t_end,
    )

    tip_compare = compare_frequency(
        teacher_freq=teacher_tip_fft["dominant_freq_hz"],
        student_freq=student_tip_fft["dominant_freq_hz"],
    )
    lastk_compare = compare_frequency(
        teacher_freq=teacher_lastk_fft["dominant_freq_hz"],
        student_freq=student_lastk_fft["dominant_freq_hz"],
    )

    print()
    print("[Dominant Frequency Comparison: Tip y]")
    print(f"  teacher freq = {tip_compare['teacher_freq_hz']:.8e} Hz")
    print(f"  student freq = {tip_compare['student_freq_hz']:.8e} Hz")
    print(f"  student/teacher = {tip_compare['student_over_teacher']:.8e}")
    print(f"  relative error  = {tip_compare['relative_error']:.8e}")
    print(f"  absolute error  = {tip_compare['absolute_error_hz']:.8e} Hz")
    print(f"  teacher top freq bins = {teacher_tip_fft['top_freqs_hz']}")
    print(f"  student top freq bins = {student_tip_fft['top_freqs_hz']}")

    print()
    print(f"[Dominant Frequency Comparison: Last-{cfg.last_k} mean y]")
    print(f"  teacher freq = {lastk_compare['teacher_freq_hz']:.8e} Hz")
    print(f"  student freq = {lastk_compare['student_freq_hz']:.8e} Hz")
    print(f"  student/teacher = {lastk_compare['student_over_teacher']:.8e}")
    print(f"  relative error  = {lastk_compare['relative_error']:.8e}")
    print(f"  absolute error  = {lastk_compare['absolute_error_hz']:.8e} Hz")
    print(f"  teacher top freq bins = {teacher_lastk_fft['top_freqs_hz']}")
    print(f"  student top freq bins = {student_lastk_fft['top_freqs_hz']}")

    tip_time_plot = output_dir / "tip_y_time_history.png"
    tip_spectrum_plot = output_dir / "tip_y_fft_spectrum.png"
    lastk_time_plot = output_dir / f"last{cfg.last_k}_mean_y_time_history.png"
    lastk_spectrum_plot = output_dir / f"last{cfg.last_k}_mean_y_fft_spectrum.png"

    plot_time_history(
        time=time_student,
        teacher_signal=teacher_tip_y,
        student_signal=student_tip_y,
        title=f"{cfg.case_name} - Tip y displacement time history",
        ylabel="Tip y displacement",
        save_path=tip_time_plot,
        show_plot=cfg.show_plot,
    )
    plot_spectrum(
        teacher_fft=teacher_tip_fft,
        student_fft=student_tip_fft,
        title=f"{cfg.case_name} - Tip y FFT spectrum",
        save_path=tip_spectrum_plot,
        show_plot=cfg.show_plot,
        min_freq_hz=cfg.min_freq_hz,
        max_freq_hz=cfg.max_freq_hz,
    )
    plot_time_history(
        time=time_student,
        teacher_signal=teacher_lastk_y,
        student_signal=student_lastk_y,
        title=f"{cfg.case_name} - Last-{cfg.last_k} mean y displacement time history",
        ylabel=f"Last-{cfg.last_k} mean y displacement",
        save_path=lastk_time_plot,
        show_plot=cfg.show_plot,
    )
    plot_spectrum(
        teacher_fft=teacher_lastk_fft,
        student_fft=student_lastk_fft,
        title=f"{cfg.case_name} - Last-{cfg.last_k} mean y FFT spectrum",
        save_path=lastk_spectrum_plot,
        show_plot=cfg.show_plot,
        min_freq_hz=cfg.min_freq_hz,
        max_freq_hz=cfg.max_freq_hz,
    )

    report = {
        "config": asdict(cfg),
        "files": {
            "teacher_out": str(teacher_out),
            "student_npz": str(student_result["npz"]),
            "student_yaml": str(student_result["yaml"]),
            "tip_y_time_plot": str(tip_time_plot),
            "tip_y_spectrum_plot": str(tip_spectrum_plot),
            "lastk_y_time_plot": str(lastk_time_plot),
            "lastk_y_spectrum_plot": str(lastk_spectrum_plot),
        },
        "tip_y": {
            "teacher_fft": to_jsonable_fft(teacher_tip_fft),
            "student_fft": to_jsonable_fft(student_tip_fft),
            "comparison": tip_compare,
        },
        "lastk_mean_y": {
            "last_k": int(cfg.last_k),
            "teacher_fft": to_jsonable_fft(teacher_lastk_fft),
            "student_fft": to_jsonable_fft(student_lastk_fft),
            "comparison": lastk_compare,
        },
        "interpretation_hint": {
            "student_over_teacher_gt_1": "student y dominant frequency is higher than teacher; y effective stiffness may be too high if mass is trusted.",
            "student_over_teacher_lt_1": "student y dominant frequency is lower than teacher; y effective stiffness may be too low if mass is trusted.",
        },
    }

    if cfg.save_report:
        report_path = output_dir / "y_frequency_teacher_student_diagnostic_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print()
        print("[Saved Report]")
        print(f"  {report_path}")

    print()
    print("[Saved Figures]")
    print(f"  tip y time      = {tip_time_plot}")
    print(f"  tip y spectrum  = {tip_spectrum_plot}")
    print(f"  last-k time     = {lastk_time_plot}")
    print(f"  last-k spectrum = {lastk_spectrum_plot}")

    print()
    print("✅ PASS: y-direction frequency diagnostic completed.")
    print("   请重点查看 student/teacher 主频比值，用于判断下一步 κ_y_residual(s) 应该增大还是减小 y 向等效刚度。")


if __name__ == "__main__":
    main()