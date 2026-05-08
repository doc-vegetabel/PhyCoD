#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate frequency-sweep blade load .dat files for alpha_x phase/frequency learning.

Default design:
  - 3 frequency bands: low / mid / high
  - train: 6 cases per band = 18 cases
  - valid: 1 case per band = 3 cases
  - test : 1 case per band = 3 cases

Output format:
  line 1: n_nodes
  line 2: span positions
  line 3: time Fx1 Fy1 Fz1 Mx1 My1 Mz1 ... Fx48 Fy48 Fz48 Mx48 My48 Mz48
  following lines: load time history

Only Fx/Fy are non-zero by default. Fz and moments are zero.

Example:
  python scripts/generate_frequency_sweep_loads.py ^
    --output-root data/load/frequency_sweep_alpha_x ^
    --duration 10.0 ^
    --dt 0.01 ^
    --fx-amp 1.0 ^
    --fy-amp 1.0 ^
    --ramp-seconds 0.5 ^
    --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


COMPONENTS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")


@dataclass
class LoadCaseSpec:
    split: str
    band: str
    case_id: int
    case_name: str
    pattern: str
    fx_freq_hz: float
    fy_freq_hz: float
    fx_amp_scale: float
    fy_amp_scale: float
    fx_phase_rad: float
    fy_phase_rad: float
    spatial_profile: str
    description: str


def default_span_positions(n_nodes: int = 48) -> np.ndarray:
    # Matches the header provided by the user: 1/48, 2/48, ..., 48/48.
    return np.arange(1, n_nodes + 1, dtype=np.float64) / float(n_nodes)


def make_headers(n_nodes: int) -> List[str]:
    headers = ["time"]
    for i in range(1, n_nodes + 1):
        for comp in COMPONENTS:
            headers.append(f"{comp}{i}")
    return headers


def cosine_ramp_envelope(t: np.ndarray, duration: float, ramp_seconds: float) -> np.ndarray:
    if ramp_seconds <= 0.0:
        return np.ones_like(t, dtype=np.float64)

    ramp = min(ramp_seconds, 0.5 * duration)
    env = np.ones_like(t, dtype=np.float64)

    m_in = t < ramp
    env[m_in] = 0.5 * (1.0 - np.cos(np.pi * t[m_in] / ramp))

    m_out = t > (duration - ramp)
    tau = duration - t[m_out]
    env[m_out] = 0.5 * (1.0 - np.cos(np.pi * tau / ramp))

    return env


def spatial_weights(n_nodes: int, profile: str) -> np.ndarray:
    x = default_span_positions(n_nodes)
    w = np.zeros(n_nodes, dtype=np.float64)
    p = profile.lower()

    if p == "tip":
        w[-1] = 1.0
    elif p == "last5":
        w[-5:] = 1.0
    elif p == "last10":
        w[-10:] = 1.0
    elif p == "midspan":
        center = 0.60
        width = 0.10
        w = np.exp(-0.5 * ((x - center) / width) ** 2)
        w /= max(float(w.max()), 1e-12)
    elif p == "fullspan_linear":
        w = x.copy()
        w /= max(float(w.max()), 1e-12)
    elif p == "fullspan_quadratic":
        w = x ** 2
        w /= max(float(w.max()), 1e-12)
    elif p == "fullspan_smooth":
        w = np.sin(0.5 * np.pi * x) ** 2
        w /= max(float(w.max()), 1e-12)
    else:
        raise ValueError(
            f"Unknown spatial profile '{profile}'. "
            "Supported: tip,last5,last10,midspan,fullspan_linear,fullspan_quadratic,fullspan_smooth"
        )

    return w


def sinusoid(t: np.ndarray, freq_hz: float, phase_rad: float) -> np.ndarray:
    return np.sin(2.0 * np.pi * freq_hz * t + phase_rad)


def build_case_specs() -> List[LoadCaseSpec]:
    specs: List[LoadCaseSpec] = []

    train_freqs = {
        "low":  [0.06, 0.08, 0.10, 0.12, 0.14, 0.16],
        "mid":  [0.20, 0.22, 0.25, 0.28, 0.30, 0.33],
        "high": [0.40, 0.45, 0.50, 0.55, 0.60, 0.70],
    }
    valid_freqs = {
        "low": 0.11,
        "mid": 0.27,
        "high": 0.52,
    }
    test_freqs = {
        "low": 0.15,
        "mid": 0.31,
        "high": 0.65,
    }

    train_patterns = [
        ("tip_fx",           1.0, 0.0, 0.0, 0.0,              "tip",                1.0, "tip single-direction Fx sinusoid"),
        ("tip_fy",           0.0, 1.0, 0.0, 0.0,              "tip",                1.0, "tip single-direction Fy sinusoid"),
        ("tip_xy_phase",     1.0, 0.7, 0.0, 0.5 * math.pi,    "tip",                1.0, "tip Fx/Fy same frequency with 90deg phase shift"),
        ("last5_fx",         0.8, 0.0, 0.0, 0.0,              "last5",              1.0, "last5 distributed Fx sinusoid"),
        ("last5_xy_phase",   0.8, 0.6, 0.0, 0.25 * math.pi,   "last5",              1.0, "last5 distributed Fx/Fy same frequency"),
        ("fullspan_xy_mix",  0.7, 0.5, 0.0, 0.35 * math.pi,   "fullspan_smooth",    1.6, "fullspan Fx/Fy mixed frequencies"),
    ]

    for band, freqs in train_freqs.items():
        for i, f in enumerate(freqs, start=1):
            pattern, fx_s, fy_s, phx, phy, prof, fy_mult, desc = train_patterns[i - 1]
            fy_f = f * fy_mult if fy_s != 0.0 else f
            specs.append(
                LoadCaseSpec(
                    split="train",
                    band=band,
                    case_id=i,
                    case_name=f"freq_{band}_train_{i:02d}_{pattern}_f{f:.3f}".replace(".", "p"),
                    pattern=pattern,
                    fx_freq_hz=f,
                    fy_freq_hz=fy_f,
                    fx_amp_scale=fx_s,
                    fy_amp_scale=fy_s,
                    fx_phase_rad=phx,
                    fy_phase_rad=phy,
                    spatial_profile=prof,
                    description=desc,
                )
            )

    for band, f in valid_freqs.items():
        specs.append(
            LoadCaseSpec(
                split="valid",
                band=band,
                case_id=1,
                case_name=f"freq_{band}_valid_01_last5_xy_holdout_f{f:.3f}".replace(".", "p"),
                pattern="last5_xy_holdout",
                fx_freq_hz=f,
                fy_freq_hz=1.4 * f,
                fx_amp_scale=0.8,
                fy_amp_scale=0.6,
                fx_phase_rad=0.0,
                fy_phase_rad=0.3 * math.pi,
                spatial_profile="last5",
                description="validation held-out frequency, last5 Fx/Fy mixed load",
            )
        )

    for band, f in test_freqs.items():
        specs.append(
            LoadCaseSpec(
                split="test",
                band=band,
                case_id=1,
                case_name=f"freq_{band}_test_01_fullspan_xy_holdout_f{f:.3f}".replace(".", "p"),
                pattern="fullspan_xy_holdout",
                fx_freq_hz=f,
                fy_freq_hz=1.3 * f,
                fx_amp_scale=0.7,
                fy_amp_scale=0.5,
                fx_phase_rad=0.0,
                fy_phase_rad=0.4 * math.pi,
                spatial_profile="fullspan_smooth",
                description="test held-out frequency, fullspan Fx/Fy mixed load",
            )
        )

    return specs


def generate_load_array(
    spec: LoadCaseSpec,
    n_nodes: int,
    duration: float,
    dt: float,
    fx_amp: float,
    fy_amp: float,
    ramp_seconds: float,
    profile_normalization: str,
) -> Tuple[np.ndarray, np.ndarray]:
    n_steps = int(round(duration / dt)) + 1
    t = np.linspace(0.0, duration, n_steps, dtype=np.float64)

    w = spatial_weights(n_nodes, spec.spatial_profile)

    norm = profile_normalization.lower()
    if norm == "max":
        w = w / max(float(np.max(np.abs(w))), 1e-12)
    elif norm == "sum":
        w = w / max(float(np.sum(np.abs(w))), 1e-12)
    elif norm == "l2":
        w = w / max(float(np.sqrt(np.sum(w ** 2))), 1e-12)
    else:
        raise ValueError("--profile-normalization must be one of: max,sum,l2")

    env = cosine_ramp_envelope(t, duration=duration, ramp_seconds=ramp_seconds)

    fx_time = fx_amp * spec.fx_amp_scale * sinusoid(t, spec.fx_freq_hz, spec.fx_phase_rad) * env
    fy_time = fy_amp * spec.fy_amp_scale * sinusoid(t, spec.fy_freq_hz, spec.fy_phase_rad) * env

    data = np.zeros((len(t), 1 + n_nodes * 6), dtype=np.float64)
    data[:, 0] = t

    for node_idx in range(n_nodes):
        base = 1 + node_idx * 6
        data[:, base + 0] = fx_time * w[node_idx]
        data[:, base + 1] = fy_time * w[node_idx]

    return t, data


def write_load_dat(path: Path, data: np.ndarray, span_positions: np.ndarray, precision: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_nodes = len(span_positions)
    headers = make_headers(n_nodes)

    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"{n_nodes}\n")
        f.write(" ".join(f"{x:.4f}" for x in span_positions) + "\n")
        f.write(" ".join(headers) + "\n")

        fmt = f"{{:.{precision}e}}"
        time_fmt = "{:.6f}"
        for row in data:
            values = [time_fmt.format(row[0])]
            values.extend(fmt.format(v) for v in row[1:])
            f.write(" ".join(values) + "\n")


def write_manifest(output_root: Path, specs: List[LoadCaseSpec], extra_rows: List[Dict[str, object]]) -> None:
    rows: List[Dict[str, object]] = []
    extra_by_name = {r["case_name"]: r for r in extra_rows}

    for spec in specs:
        row = asdict(spec)
        row.update(extra_by_name.get(spec.case_name, {}))
        rows.append(row)

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    with (output_root / "frequency_sweep_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with (output_root / "frequency_sweep_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def build_train_valid_test_arg_files(output_root: Path, specs: List[LoadCaseSpec]) -> None:
    for split in ("train", "valid", "test"):
        paths = [
            str((output_root / spec.split / f"{spec.case_name}.dat").as_posix())
            for spec in specs
            if spec.split == split
        ]
        (output_root / f"{split}_load_files.txt").write_text(",".join(paths), encoding="utf-8")
        (output_root / f"{split}_load_files_lines.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("data/load/frequency_sweep_alpha_x"))
    parser.add_argument("--n-nodes", type=int, default=48)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--fx-amp", type=float, default=1.0)
    parser.add_argument("--fy-amp", type=float, default=1.0)
    parser.add_argument("--ramp-seconds", type=float, default=0.5)
    parser.add_argument("--precision", type=int, default=8)
    parser.add_argument(
        "--profile-normalization",
        type=str,
        default="max",
        choices=["max", "sum", "l2"],
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    span = default_span_positions(args.n_nodes)
    specs = build_case_specs()
    extra_rows: List[Dict[str, object]] = []

    print("\n" + "=" * 100)
    print("[Generate Frequency-Sweep Load DAT Files]")
    print("=" * 100)
    print(f"  output_root = {output_root.resolve()}")
    print(f"  n_nodes     = {args.n_nodes}")
    print(f"  duration    = {args.duration}")
    print(f"  dt          = {args.dt}")
    print(f"  fx_amp      = {args.fx_amp}")
    print(f"  fy_amp      = {args.fy_amp}")
    print(f"  ramp        = {args.ramp_seconds}")
    print(f"  n_train     = {sum(1 for s in specs if s.split == 'train')}")
    print(f"  n_valid     = {sum(1 for s in specs if s.split == 'valid')}")
    print(f"  n_test      = {sum(1 for s in specs if s.split == 'test')}")

    for spec in specs:
        out_path = output_root / spec.split / f"{spec.case_name}.dat"
        if out_path.exists() and not args.overwrite:
            print(f"  [Skip existing] {out_path}")
            continue

        t, data = generate_load_array(
            spec=spec,
            n_nodes=args.n_nodes,
            duration=args.duration,
            dt=args.dt,
            fx_amp=args.fx_amp,
            fy_amp=args.fy_amp,
            ramp_seconds=args.ramp_seconds,
            profile_normalization=args.profile_normalization,
        )
        write_load_dat(out_path, data, span, precision=args.precision)

        fx_cols = []
        fy_cols = []
        for i in range(args.n_nodes):
            base = 1 + i * 6
            fx_cols.append(base + 0)
            fy_cols.append(base + 1)
        fx = data[:, fx_cols]
        fy = data[:, fy_cols]

        extra_rows.append({
            "case_name": spec.case_name,
            "path": str(out_path.as_posix()),
            "n_steps": int(len(t)),
            "time_start": float(t[0]),
            "time_end": float(t[-1]),
            "dt": float(args.dt),
            "fx_max_abs": float(np.max(np.abs(fx))),
            "fy_max_abs": float(np.max(np.abs(fy))),
            "fx_rms_all_nodes": float(np.sqrt(np.mean(fx ** 2))),
            "fy_rms_all_nodes": float(np.sqrt(np.mean(fy ** 2))),
        })

        print(
            f"  [{spec.split:<5s}] {spec.case_name:<55s} "
            f"band={spec.band:<4s} pattern={spec.pattern:<18s} "
            f"fx_f={spec.fx_freq_hz:.3f} fy_f={spec.fy_freq_hz:.3f}"
        )

    write_manifest(output_root, specs, extra_rows)
    build_train_valid_test_arg_files(output_root, specs)

    print("\n[Saved]")
    print(f"  manifest_csv  = {output_root / 'frequency_sweep_manifest.csv'}")
    print(f"  manifest_json = {output_root / 'frequency_sweep_manifest.json'}")
    print(f"  train list    = {output_root / 'train_load_files.txt'}")
    print(f"  valid list    = {output_root / 'valid_load_files.txt'}")
    print(f"  test list     = {output_root / 'test_load_files.txt'}")

    print("\n[PowerShell helper]")
    print("  $train_files = Get-Content " + str(output_root / "train_load_files.txt"))
    print("  $valid_files = Get-Content " + str(output_root / "valid_load_files.txt"))
    print("  $test_files  = Get-Content " + str(output_root / "test_load_files.txt"))
    print("\n✅ PASS: generated frequency-sweep load files.")


if __name__ == "__main__":
    main()
