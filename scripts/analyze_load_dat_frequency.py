#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze time-domain and frequency-domain features of blade load .dat files.

V2 adds time-domain / autocorrelation features so a short pulse is not mistaken
for a sustained periodic load just because its FFT has a strong low-frequency bin.
It can also export causal sliding-window load spectral features for later model input.

Expected .dat format:
  line 1: number of load nodes, e.g. 48
  line 2: span positions for each node
  line 3: column names: time Fx1 Fy1 Fz1 Mx1 My1 Mz1 Fx2 ...
  remaining lines: numeric time series
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

EPS = 1e-30


def parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_bands(text: str) -> List[Tuple[float, float]]:
    bands: List[Tuple[float, float]] = []
    if not text:
        return bands
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" not in item:
            raise ValueError(f"Invalid band '{item}'. Expected format like 0.05-0.5")
        a, b = item.split("-", 1)
        lo = float(a)
        hi = float(b)
        if hi <= lo:
            raise ValueError(f"Invalid band '{item}': upper bound must be > lower bound.")
        bands.append((lo, hi))
    return bands


def sanitize_name(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_").replace(".", "p").replace("-", "m")


def read_load_dat(path: Path) -> Tuple[int, np.ndarray, List[str], np.ndarray]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        raw_lines = [line.strip() for line in f if line.strip()]
    if len(raw_lines) < 4:
        raise ValueError(f"File too short: {path}")
    n_nodes = int(raw_lines[0].split()[0])
    span_positions = np.array([float(x) for x in raw_lines[1].split()], dtype=np.float64)
    headers = raw_lines[2].split()
    if len(span_positions) != n_nodes:
        raise ValueError(f"n_nodes={n_nodes}, but got {len(span_positions)} span positions")
    expected_cols = len(headers)
    rows: List[List[float]] = []
    for lineno, line in enumerate(raw_lines[3:], start=4):
        values = line.split()
        if len(values) != expected_cols:
            raise ValueError(f"Line {lineno}: expected {expected_cols} columns, got {len(values)}")
        rows.append([float(x) for x in values])
    return n_nodes, span_positions, headers, np.asarray(rows, dtype=np.float64)


def column_index(headers: List[str], name: str) -> int:
    try:
        return headers.index(name)
    except ValueError as exc:
        raise KeyError(f"Column '{name}' not found in dat header.") from exc


def get_component_matrix(headers: List[str], data: np.ndarray, component: str, n_nodes: int) -> np.ndarray:
    cols = [column_index(headers, f"{component}{i}") for i in range(1, n_nodes + 1)]
    return data[:, cols]


def build_observation_signals(
    headers: List[str],
    data: np.ndarray,
    n_nodes: int,
    components: List[str],
    observations: List[str],
    last_k: int,
) -> Dict[str, np.ndarray]:
    signals: Dict[str, np.ndarray] = {}
    comp_mats: Dict[str, np.ndarray] = {c: get_component_matrix(headers, data, c, n_nodes) for c in components}
    last_k = max(1, min(last_k, n_nodes))
    for comp, mat in comp_mats.items():
        for obs in observations:
            obs_l = obs.lower()
            if obs_l == "tip":
                signals[f"{comp}_tip"] = mat[:, -1]
            elif obs_l == "root":
                signals[f"{comp}_root"] = mat[:, 0]
            elif obs_l == "mean":
                signals[f"{comp}_mean"] = mat.mean(axis=1)
            elif obs_l == "last5":
                signals[f"{comp}_last{last_k}"] = mat[:, -last_k:].mean(axis=1)
            elif obs_l == "rms":
                signals[f"{comp}_rms"] = np.sqrt(np.mean(mat ** 2, axis=1))
            elif obs_l.startswith("node"):
                node_idx = int(obs_l.replace("node", ""))
                if node_idx < 1 or node_idx > n_nodes:
                    raise ValueError(f"{obs} out of range 1..{n_nodes}")
                signals[f"{comp}_node{node_idx}"] = mat[:, node_idx - 1]
            else:
                raise ValueError(f"Unknown observation '{obs}'. Supported: tip, root, mean, last5, rms, nodeK")

    lower = {c.lower() for c in components}
    if "fx" in lower and "fy" in lower:
        fx = comp_mats[next(c for c in components if c.lower() == "fx")]
        fy = comp_mats[next(c for c in components if c.lower() == "fy")]
        fxy = np.sqrt(fx ** 2 + fy ** 2)
        for obs in observations:
            obs_l = obs.lower()
            if obs_l == "tip":
                signals["Fxy_tip_resultant"] = fxy[:, -1]
            elif obs_l == "mean":
                signals["Fxy_mean_resultant"] = fxy.mean(axis=1)
            elif obs_l == "last5":
                signals[f"Fxy_last{last_k}_resultant"] = fxy[:, -last_k:].mean(axis=1)
            elif obs_l == "root":
                signals["Fxy_root_resultant"] = fxy[:, 0]
            elif obs_l == "rms":
                signals["Fxy_rms_resultant"] = np.sqrt(np.mean(fxy ** 2, axis=1))
    return signals


def estimate_dt(t: np.ndarray) -> float:
    if len(t) < 2:
        raise ValueError("Need at least two time samples")
    dt_arr = np.diff(t)
    dt = float(np.median(dt_arr))
    if not np.allclose(dt_arr, dt, rtol=1e-3, atol=1e-8):
        print("[WARN] Non-uniform time step detected; using median dt.")
    return dt


def hann_power_spectrum(t: np.ndarray, y: np.ndarray, demean: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    if len(t) < 4:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    dt = estimate_dt(t)
    yy = np.asarray(y, dtype=np.float64)
    if demean:
        yy = yy - np.mean(yy)
    window = np.hanning(len(yy))
    yyw = yy * window
    fft = np.fft.rfft(yyw)
    freq = np.fft.rfftfreq(len(yyw), d=dt)
    power = (np.abs(fft) ** 2) / max(np.sum(window ** 2), EPS)
    return freq, power


def frequency_stats(
    freq: np.ndarray,
    power: np.ndarray,
    freq_min: float,
    freq_max: float,
    bands: List[Tuple[float, float]],
    top_k: int,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    mask = (freq >= freq_min) & (freq <= freq_max) & (freq > 0.0)
    f = freq[mask]
    p = power[mask]
    stats: Dict[str, float] = {}
    if len(f) == 0 or float(np.sum(p)) <= EPS:
        stats.update({
            "dominant_freq_hz": float("nan"),
            "dominant_power": 0.0,
            "total_power": 0.0,
            "spectral_centroid_hz": float("nan"),
            "spectral_entropy": float("nan"),
            "peak_sharpness": float("nan"),
            "periodicity_index_fft": float("nan"),
        })
        for lo, hi in bands:
            stats[f"band_{lo:g}_{hi:g}_power_ratio"] = float("nan")
        return stats, []

    total_power = float(np.sum(p))
    p_norm = p / max(total_power, EPS)
    imax = int(np.argmax(p))
    dominant_freq = float(f[imax])
    dominant_power = float(p[imax])
    centroid = float(np.sum(f * p_norm))
    entropy = float(-np.sum(p_norm * np.log(p_norm + EPS)) / math.log(max(len(p_norm), 2)))
    sharp = float(dominant_power / max(total_power, EPS))
    periodicity_fft = float(dominant_power / (float(np.median(p)) + EPS))
    stats.update({
        "dominant_freq_hz": dominant_freq,
        "dominant_power": dominant_power,
        "total_power": total_power,
        "spectral_centroid_hz": centroid,
        "spectral_entropy": entropy,
        "peak_sharpness": sharp,
        "periodicity_index_fft": periodicity_fft,
    })
    for lo, hi in bands:
        bmask = (f >= lo) & (f < hi)
        stats[f"band_{lo:g}_{hi:g}_power_ratio"] = float(np.sum(p[bmask])) / max(total_power, EPS)

    peak_indices: List[int] = []
    if len(p) >= 3:
        for i in range(1, len(p) - 1):
            if p[i] >= p[i - 1] and p[i] >= p[i + 1]:
                peak_indices.append(i)
    if not peak_indices:
        peak_indices = list(np.argsort(p)[::-1][:top_k])
    else:
        peak_indices = sorted(peak_indices, key=lambda i: p[i], reverse=True)[:top_k]
    peaks = []
    for rank, i in enumerate(peak_indices, start=1):
        peaks.append({"rank": rank, "freq_hz": float(f[i]), "power": float(p[i]), "power_ratio": float(p[i] / max(total_power, EPS))})
    return stats, peaks


def active_mask_from_signal(y: np.ndarray, active_rel_threshold: float, active_abs_threshold: float) -> Tuple[np.ndarray, float]:
    amp = float(np.max(np.abs(y))) if len(y) else 0.0
    threshold = max(active_abs_threshold, active_rel_threshold * amp)
    if amp <= EPS:
        return np.zeros_like(y, dtype=bool), threshold
    return np.abs(y) >= threshold, threshold


def count_segments(mask: np.ndarray) -> Tuple[int, int]:
    n_segments = 0
    longest = 0
    current = 0
    was_active = False
    for v in mask:
        if v:
            current += 1
            if not was_active:
                n_segments += 1
            was_active = True
        else:
            longest = max(longest, current)
            current = 0
            was_active = False
    longest = max(longest, current)
    return n_segments, longest


def zero_crossing_count(y: np.ndarray, threshold: float) -> int:
    if len(y) < 2:
        return 0
    yy = y - np.mean(y)
    yy[np.abs(yy) < threshold] = 0.0
    signs = np.sign(yy)
    nz = signs[signs != 0]
    if len(nz) < 2:
        return 0
    return int(np.sum(nz[1:] * nz[:-1] < 0))


def normalized_autocorr(y: np.ndarray) -> np.ndarray:
    if len(y) < 4:
        return np.array([], dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64) - np.mean(y)
    denom = float(np.sum(yy ** 2))
    if denom <= EPS:
        return np.array([], dtype=np.float64)
    corr = np.correlate(yy, yy, mode="full")
    return corr[len(yy) - 1:] / denom


def autocorr_features(t: np.ndarray, y: np.ndarray, freq_min: float, freq_max: float) -> Dict[str, float]:
    if len(t) < 8:
        return {"autocorr_peak": float("nan"), "autocorr_period_s": float("nan"), "autocorr_freq_hz": float("nan")}
    dt = estimate_dt(t)
    ac = normalized_autocorr(y)
    if len(ac) == 0:
        return {"autocorr_peak": float("nan"), "autocorr_period_s": float("nan"), "autocorr_freq_hz": float("nan")}
    min_lag = max(1, int(round((1.0 / max(freq_max, EPS)) / dt)))
    max_lag = min(len(ac) - 1, int(round((1.0 / max(freq_min, EPS)) / dt)))
    if max_lag <= min_lag:
        return {"autocorr_peak": float("nan"), "autocorr_period_s": float("nan"), "autocorr_freq_hz": float("nan")}
    seg = ac[min_lag:max_lag + 1]
    lag_idx = min_lag + int(np.argmax(seg))
    period_s = float(lag_idx * dt)
    return {"autocorr_peak": float(ac[lag_idx]), "autocorr_period_s": period_s, "autocorr_freq_hz": float(1.0 / period_s) if period_s > EPS else float("nan")}


def time_domain_features(
    t: np.ndarray,
    y: np.ndarray,
    dominant_freq_hz: float,
    active_rel_threshold: float,
    active_abs_threshold: float,
) -> Dict[str, float]:
    dt = estimate_dt(t) if len(t) >= 2 else float("nan")
    total_duration = float(t[-1] - t[0]) if len(t) >= 2 else 0.0
    mask, threshold = active_mask_from_signal(y, active_rel_threshold, active_abs_threshold)
    active_fraction = float(np.mean(mask)) if len(mask) else 0.0
    n_segments, longest_run = count_segments(mask)
    longest_active_duration = float(longest_run * dt) if np.isfinite(dt) else 0.0
    if np.any(mask):
        active_indices = np.where(mask)[0]
        first_active_time = float(t[active_indices[0]])
        last_active_time = float(t[active_indices[-1]])
        active_span_duration = float(last_active_time - first_active_time + dt)
    else:
        first_active_time = float("nan")
        last_active_time = float("nan")
        active_span_duration = 0.0
    zc_threshold = max(active_abs_threshold, 0.02 * float(np.max(np.abs(y))) if len(y) else 0.0)
    zc = zero_crossing_count(y, threshold=zc_threshold)
    if dominant_freq_hz is None or not np.isfinite(dominant_freq_hz):
        cycles_total = float("nan")
        cycles_active = float("nan")
    else:
        cycles_total = float(dominant_freq_hz * total_duration)
        cycles_active = float(dominant_freq_hz * active_span_duration)
    rms = float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0
    std = float(np.std(y)) if len(y) else 0.0
    mean_abs = float(np.mean(np.abs(y))) if len(y) else 0.0
    max_abs = float(np.max(np.abs(y))) if len(y) else 0.0

    pulse_like_score = 0.0
    if active_fraction < 0.35:
        pulse_like_score += 0.45
    if n_segments <= 2:
        pulse_like_score += 0.25
    if np.isfinite(cycles_active) and cycles_active < 1.5:
        pulse_like_score += 0.20
    if zc <= 2:
        pulse_like_score += 0.10
    pulse_like_score = float(min(1.0, pulse_like_score))

    sustained_oscillation_score = 0.0
    if active_fraction > 0.65:
        sustained_oscillation_score += 0.30
    if zc >= 4:
        sustained_oscillation_score += 0.25
    if np.isfinite(cycles_total) and cycles_total >= 2.0:
        sustained_oscillation_score += 0.25
    if n_segments >= 1 and active_span_duration > 0.6 * total_duration:
        sustained_oscillation_score += 0.20
    sustained_oscillation_score = float(min(1.0, sustained_oscillation_score))

    return {
        "mean_abs": mean_abs,
        "rms": rms,
        "std": std,
        "max_abs": max_abs,
        "active_threshold": float(threshold),
        "active_fraction": active_fraction,
        "active_segments": int(n_segments),
        "longest_active_duration_s": longest_active_duration,
        "first_active_time_s": first_active_time,
        "last_active_time_s": last_active_time,
        "active_span_duration_s": active_span_duration,
        "zero_crossings_demeaned": int(zc),
        "effective_cycles_total": cycles_total,
        "effective_cycles_active_span": cycles_active,
        "pulse_like_score": pulse_like_score,
        "sustained_oscillation_score": sustained_oscillation_score,
    }


def classify_signal(row: Dict[str, float]) -> str:
    max_abs = float(row.get("max_abs", 0.0) or 0.0)
    if max_abs <= EPS:
        return "zero_or_near_zero"
    pulse = float(row.get("pulse_like_score", 0.0) or 0.0)
    osc = float(row.get("sustained_oscillation_score", 0.0) or 0.0)
    ac_peak = float(row.get("autocorr_peak", 0.0) or 0.0)
    sharp = float(row.get("peak_sharpness", 0.0) or 0.0)
    active_fraction = float(row.get("active_fraction", 0.0) or 0.0)
    if pulse >= 0.65 and active_fraction < 0.45:
        return "short_pulse_or_transient"
    if osc >= 0.65 and ac_peak >= 0.25 and sharp >= 0.15:
        return "sustained_periodic"
    if active_fraction >= 0.65 and sharp < 0.20:
        return "broadband_or_irregular"
    if osc >= 0.45:
        return "weak_or_mixed_periodic"
    return "transient_or_low_confidence"


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def compute_signal_features(
    t: np.ndarray,
    y: np.ndarray,
    freq_min: float,
    freq_max: float,
    bands: List[Tuple[float, float]],
    top_k: int,
    active_rel_threshold: float,
    active_abs_threshold: float,
) -> Tuple[Dict[str, float], List[Dict[str, float]], np.ndarray, np.ndarray]:
    freq, power = hann_power_spectrum(t, y, demean=True)
    fstats, peaks = frequency_stats(freq, power, freq_min, freq_max, bands, top_k)
    tstats = time_domain_features(t, y, float(fstats.get("dominant_freq_hz", float("nan"))), active_rel_threshold, active_abs_threshold)
    astats = autocorr_features(t, y, freq_min, freq_max)
    row = {**tstats, **fstats, **astats}
    row["signal_class"] = classify_signal(row)
    return row, peaks, freq, power


def make_window_features(
    t: np.ndarray,
    signals: Dict[str, np.ndarray],
    freq_min: float,
    freq_max: float,
    bands: List[Tuple[float, float]],
    top_k: int,
    active_rel_threshold: float,
    active_abs_threshold: float,
    window_seconds: float,
    feature_stride_steps: int,
    min_window_samples: int,
) -> Tuple[List[Dict[str, object]], np.ndarray, List[str]]:
    dt = estimate_dt(t)
    window_steps = max(min_window_samples, int(round(window_seconds / dt)))
    stride = max(1, int(feature_stride_steps))
    rows: List[Dict[str, object]] = []
    feature_names: List[str] = []
    feature_matrix: List[List[float]] = []
    numeric_keys = [
        "dominant_freq_hz", "spectral_centroid_hz", "spectral_entropy", "peak_sharpness",
        "periodicity_index_fft", "autocorr_peak", "autocorr_freq_hz", "active_fraction",
        "active_segments", "active_span_duration_s", "zero_crossings_demeaned",
        "effective_cycles_total", "effective_cycles_active_span", "pulse_like_score",
        "sustained_oscillation_score",
    ]
    for lo, hi in bands:
        numeric_keys.append(f"band_{lo:g}_{hi:g}_power_ratio")
    ordered_signal_names = list(signals.keys())
    for sig in ordered_signal_names:
        for key in numeric_keys:
            feature_names.append(f"{sanitize_name(sig)}__{key}")
    for end_idx in range(0, len(t), stride):
        start_idx = max(0, end_idx - window_steps + 1)
        if end_idx - start_idx + 1 < min_window_samples:
            continue
        tw = t[start_idx:end_idx + 1]
        row: Dict[str, object] = {
            "time": float(t[end_idx]), "start_time": float(tw[0]), "end_time": float(tw[-1]),
            "start_index": int(start_idx), "end_index": int(end_idx),
        }
        feat_vec: List[float] = []
        for sig in ordered_signal_names:
            yw = signals[sig][start_idx:end_idx + 1]
            feat, _, _, _ = compute_signal_features(tw, yw, freq_min, freq_max, bands, top_k, active_rel_threshold, active_abs_threshold)
            for key in numeric_keys:
                val = feat.get(key, float("nan"))
                v = float(val) if isinstance(val, (int, float, np.integer, np.floating)) else float("nan")
                row[f"{sanitize_name(sig)}__{key}"] = v
                feat_vec.append(v)
        rows.append(row)
        feature_matrix.append(feat_vec)
    return rows, np.asarray(feature_matrix, dtype=np.float64), feature_names


def maybe_plot_spectra(out_dir: Path, spectra_rows: List[Dict[str, object]], summary_rows: List[Dict[str, object]], freq_min: float, freq_max: float, max_plots: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable; skip plots. Error: {exc}")
        return
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    selected = [str(r["signal"]) for r in sorted(summary_rows, key=lambda r: float(r.get("total_power", 0.0) or 0.0), reverse=True)[:max_plots]]
    by_signal: Dict[str, List[Dict[str, object]]] = {}
    for row in spectra_rows:
        sig = str(row["signal"])
        if sig in selected:
            by_signal.setdefault(sig, []).append(row)
    for sig, rows in by_signal.items():
        rows_sorted = sorted(rows, key=lambda r: float(r["freq_hz"]))
        freq = np.array([float(r["freq_hz"]) for r in rows_sorted])
        power = np.array([float(r["power"]) for r in rows_sorted])
        mask = (freq >= freq_min) & (freq <= freq_max)
        if not np.any(mask):
            continue
        fig = plt.figure(figsize=(8, 4.5))
        plt.plot(freq[mask], power[mask])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power")
        plt.title(f"Power spectrum: {sig}")
        plt.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{sanitize_name(sig)}_spectrum.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--components", default="Fx,Fy")
    parser.add_argument("--observations", default="tip,last5,mean")
    parser.add_argument("--last-k", type=int, default=5)
    parser.add_argument("--freq-min", type=float, default=0.05)
    parser.add_argument("--freq-max", type=float, default=5.0)
    parser.add_argument("--bands", default="0.05-0.5,0.5-1.5,1.5-5.0")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--active-rel-threshold", type=float, default=1e-3)
    parser.add_argument("--active-abs-threshold", type=float, default=1e-12)
    parser.add_argument("--max-plots", type=int, default=12)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--make-window-features", action="store_true")
    parser.add_argument("--window-seconds", type=float, default=1.92)
    parser.add_argument("--feature-stride-steps", type=int, default=1)
    parser.add_argument("--min-window-samples", type=int, default=64)
    args = parser.parse_args()

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    components = parse_csv_list(args.components)
    observations = parse_csv_list(args.observations)
    bands = parse_bands(args.bands)
    n_nodes, span_positions, headers, data = read_load_dat(args.load_file)
    t = data[:, column_index(headers, "time")]
    dt = estimate_dt(t)
    duration = float(t[-1] - t[0])
    fs = 1.0 / dt

    print("\n" + "=" * 100)
    print("[Load Time/Frequency Analysis V2]")
    print("=" * 100)
    print(f"  load_file     = {args.load_file.resolve()}")
    print(f"  output_dir    = {out_dir.resolve()}")
    print(f"  n_nodes       = {n_nodes}")
    print(f"  n_steps       = {len(t)}")
    print(f"  dt            = {dt:.8f} s")
    print(f"  fs            = {fs:.6f} Hz")
    print(f"  duration      = {duration:.6f} s")
    print(f"  freq_window   = [{args.freq_min}, {args.freq_max}] Hz")
    print(f"  components    = {components}")
    print(f"  observations  = {observations}")

    signals = build_observation_signals(headers, data, n_nodes, components, observations, args.last_k)
    summary_rows: List[Dict[str, object]] = []
    peak_rows: List[Dict[str, object]] = []
    spectra_rows: List[Dict[str, object]] = []
    for signal_name, y in signals.items():
        feat, peaks, freq, power = compute_signal_features(t, y, args.freq_min, args.freq_max, bands, args.top_k, args.active_rel_threshold, args.active_abs_threshold)
        summary_rows.append({"load_file": str(args.load_file), "signal": signal_name, "freq_min": args.freq_min, "freq_max": args.freq_max, **feat})
        for peak in peaks:
            peak_rows.append({"load_file": str(args.load_file), "signal": signal_name, **peak})
        mask = (freq >= args.freq_min) & (freq <= args.freq_max)
        for f, p in zip(freq[mask], power[mask]):
            spectra_rows.append({"load_file": str(args.load_file), "signal": signal_name, "freq_hz": float(f), "power": float(p)})
    summary_rows = sorted(summary_rows, key=lambda r: float(r.get("total_power", 0.0) or 0.0), reverse=True)
    write_csv(out_dir / "load_signal_summary_v2.csv", summary_rows)
    write_csv(out_dir / "load_frequency_top_peaks_v2.csv", peak_rows)
    write_csv(out_dir / "load_power_spectra_v2.csv", spectra_rows)

    metadata = {
        "load_file": str(args.load_file.resolve()), "n_nodes": n_nodes, "span_positions": span_positions.tolist(),
        "n_steps": int(len(t)), "dt": dt, "sampling_frequency_hz": fs, "duration_seconds": duration,
        "components": components, "observations": observations, "freq_min": args.freq_min, "freq_max": args.freq_max,
        "bands": bands, "active_rel_threshold": args.active_rel_threshold, "active_abs_threshold": args.active_abs_threshold,
    }

    if args.make_window_features:
        print("\n[Windowed spectral features]")
        print(f"  window_seconds       = {args.window_seconds}")
        print(f"  feature_stride_steps = {args.feature_stride_steps}")
        window_rows, feature_matrix, feature_names = make_window_features(
            t, signals, args.freq_min, args.freq_max, bands, args.top_k, args.active_rel_threshold,
            args.active_abs_threshold, args.window_seconds, args.feature_stride_steps, args.min_window_samples,
        )
        write_csv(out_dir / "load_window_spectral_features.csv", window_rows)
        np.savez_compressed(
            out_dir / "load_window_spectral_features.npz",
            features=feature_matrix,
            feature_names=np.asarray(feature_names, dtype=object),
            time=np.asarray([float(r["time"]) for r in window_rows], dtype=np.float64),
            start_index=np.asarray([int(r["start_index"]) for r in window_rows], dtype=np.int64),
            end_index=np.asarray([int(r["end_index"]) for r in window_rows], dtype=np.int64),
        )
        metadata["window_features"] = {
            "enabled": True,
            "window_seconds": args.window_seconds,
            "feature_stride_steps": args.feature_stride_steps,
            "min_window_samples": args.min_window_samples,
            "n_feature_rows": int(feature_matrix.shape[0]),
            "n_features": int(feature_matrix.shape[1]) if feature_matrix.ndim == 2 else 0,
            "feature_names": feature_names,
        }
    else:
        metadata["window_features"] = {"enabled": False}

    (out_dir / "load_frequency_metadata_v2.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if not args.no_plots:
        maybe_plot_spectra(out_dir, spectra_rows, summary_rows, args.freq_min, args.freq_max, args.max_plots)

    print("\n[Top signals by total spectral power]")
    for row in summary_rows[: min(12, len(summary_rows))]:
        print(
            f"  {str(row['signal']):<24s} class={str(row['signal_class']):<26s} "
            f"dom_f={float(row['dominant_freq_hz']):.4f} Hz  "
            f"sharp={float(row['peak_sharpness']):.4f}  "
            f"active={float(row['active_fraction']):.3f}  "
            f"cycles_active={float(row['effective_cycles_active_span']):.3f}  "
            f"pulse={float(row['pulse_like_score']):.2f}  "
            f"osc={float(row['sustained_oscillation_score']):.2f}"
        )
    print("\n[Saved]")
    print(f"  summary_csv = {out_dir / 'load_signal_summary_v2.csv'}")
    print(f"  peaks_csv   = {out_dir / 'load_frequency_top_peaks_v2.csv'}")
    print(f"  spectra_csv = {out_dir / 'load_power_spectra_v2.csv'}")
    if args.make_window_features:
        print(f"  window_csv  = {out_dir / 'load_window_spectral_features.csv'}")
        print(f"  window_npz  = {out_dir / 'load_window_spectral_features.npz'}")
    if not args.no_plots:
        print(f"  plots_dir   = {out_dir / 'plots'}")
    print("✅ PASS: load time/frequency analysis V2 completed.")


if __name__ == "__main__":
    main()
