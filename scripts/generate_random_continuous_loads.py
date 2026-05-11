#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate random continuous blade load .dat files for generalization training.

The script copies the first three header lines from a reference load file by
default, so the output format matches existing complex-case .dat files:

  line 1: n_nodes
  line 2: span positions
  line 3: time Fx1 Fy1 Fz1 Mx1 My1 Mz1 ...

Only Fx/Fy are non-zero by default. Each case samples a continuous mixture of
frequency components, phases, amplitudes, spatial profiles, envelopes, and an
optional chirp term. The resulting train/valid/test files are meant to replace
case-name-specific tuning with broader load-family coverage.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


COMPONENTS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")


@dataclass
class RandomLoadSpec:
    split: str
    case_id: int
    case_name: str
    seed: int
    family: str
    spatial_profile: str
    n_components: int
    fx_scale: float
    fy_scale: float
    chirp_weight: float
    burst_weight: float
    freq_min_hz: float
    freq_max_hz: float


def read_reference_header(path: Path) -> tuple[int, np.ndarray, list[str]]:
    with path.open("r", encoding="utf-8") as f:
        n_nodes = int(f.readline().strip())
        span_positions = np.asarray([float(v) for v in f.readline().split()], dtype=np.float64)
        headers = f.readline().split()
    if len(span_positions) != n_nodes:
        raise ValueError(
            f"Reference span count mismatch: n_nodes={n_nodes}, spans={len(span_positions)}"
        )
    expected_columns = 1 + n_nodes * 6
    if len(headers) != expected_columns:
        raise ValueError(
            f"Reference header column mismatch: expected {expected_columns}, got {len(headers)}"
        )
    return n_nodes, span_positions, headers


def default_headers(n_nodes: int) -> list[str]:
    headers = ["time"]
    for i in range(1, n_nodes + 1):
        headers.extend(f"{comp}{i}" for comp in COMPONENTS)
    return headers


def default_span_positions(n_nodes: int) -> np.ndarray:
    return np.arange(1, n_nodes + 1, dtype=np.float64) / float(n_nodes)


def split_counts(total: int, valid: int, test: int) -> dict[str, int]:
    if total < 0 or valid < 0 or test < 0:
        raise ValueError("case counts must be non-negative")
    return {"train": total, "valid": valid, "test": test}


def smooth_ramp(t: np.ndarray, duration: float, ramp_seconds: float) -> np.ndarray:
    if ramp_seconds <= 0.0:
        return np.ones_like(t)
    ramp = min(float(ramp_seconds), 0.5 * float(duration))
    env = np.ones_like(t)
    m0 = t < ramp
    env[m0] = 0.5 * (1.0 - np.cos(np.pi * t[m0] / ramp))
    m1 = t > duration - ramp
    tau = duration - t[m1]
    env[m1] = 0.5 * (1.0 - np.cos(np.pi * tau / ramp))
    return env


def spatial_profile(span: np.ndarray, name: str, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(span, dtype=np.float64)
    p = name.lower()
    w = np.zeros_like(x)
    if p == "tip":
        w[-1] = 1.0
    elif p == "last5":
        w[-min(5, len(w)):] = 1.0
    elif p == "last10":
        w[-min(10, len(w)):] = 1.0
    elif p == "fullspan_linear":
        w = x.copy()
    elif p == "fullspan_smooth":
        w = np.sin(0.5 * np.pi * x) ** 2
    elif p == "fullspan_quadratic":
        w = x ** 2
    elif p == "midspan_gaussian":
        center = rng.uniform(0.45, 0.75)
        width = rng.uniform(0.08, 0.18)
        w = np.exp(-0.5 * ((x - center) / width) ** 2)
    elif p == "random_smooth":
        c1 = rng.uniform(0.4, 1.2)
        c2 = rng.uniform(-0.35, 0.35)
        c3 = rng.uniform(-0.25, 0.25)
        w = c1 * x + c2 * np.sin(np.pi * x) + c3 * np.sin(2.0 * np.pi * x)
        w = np.maximum(w, 0.0)
    else:
        raise ValueError(f"Unsupported spatial profile: {name}")
    return w / max(float(np.max(np.abs(w))), 1.0e-12)


def random_envelope(
        t: np.ndarray,
        duration: float,
        rng: np.random.Generator,
        burst_weight: float,
) -> np.ndarray:
    env = np.ones_like(t)
    slow_freq = rng.uniform(0.03, 0.16)
    slow_phase = rng.uniform(0.0, 2.0 * np.pi)
    env *= 1.0 + rng.uniform(0.0, 0.35) * np.sin(2.0 * np.pi * slow_freq * t + slow_phase)
    env = np.maximum(env, 0.05)

    if burst_weight > 0.0:
        n_bursts = int(rng.integers(1, 4))
        for _ in range(n_bursts):
            center = rng.uniform(0.15 * duration, 0.90 * duration)
            width = rng.uniform(0.25, 1.25)
            amp = rng.uniform(0.15, 0.85) * float(burst_weight)
            env *= 1.0 + amp * np.exp(-0.5 * ((t - center) / width) ** 2)
    return env


def multisine(
        t: np.ndarray,
        rng: np.random.Generator,
        n_components: int,
        freq_min: float,
        freq_max: float,
) -> tuple[np.ndarray, list[float], list[float], list[float]]:
    y = np.zeros_like(t)
    freqs: list[float] = []
    amps: list[float] = []
    phases: list[float] = []
    for _ in range(int(n_components)):
        freq = float(rng.uniform(freq_min, freq_max))
        amp = float(rng.uniform(0.25, 1.0))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        y += amp * np.sin(2.0 * np.pi * freq * t + phase)
        freqs.append(freq)
        amps.append(amp)
        phases.append(phase)
    y /= max(float(np.max(np.abs(y))), 1.0e-12)
    return y, freqs, amps, phases


def chirp_signal(
        t: np.ndarray,
        rng: np.random.Generator,
        duration: float,
        freq_min: float,
        freq_max: float,
) -> tuple[np.ndarray, float, float]:
    f0 = float(rng.uniform(freq_min, 0.5 * (freq_min + freq_max)))
    f1 = float(rng.uniform(0.5 * (freq_min + freq_max), freq_max))
    phase0 = float(rng.uniform(0.0, 2.0 * np.pi))
    k = (f1 - f0) / max(float(duration), 1.0e-12)
    phase = 2.0 * np.pi * (f0 * t + 0.5 * k * t * t) + phase0
    y = np.sin(phase)
    return y, f0, f1


def sample_spec(
        split: str,
        case_id: int,
        seed: int,
        freq_min: float,
        freq_max: float,
        max_components: int,
        profile_names: list[str],
        family_names: list[str],
        chirp_weight_max: float,
        burst_weight_max: float,
        rng: np.random.Generator,
) -> RandomLoadSpec:
    family = str(rng.choice(family_names))
    n_components = 1 if family == "single" else int(rng.integers(2, max_components + 1))
    profile = str(rng.choice(profile_names))
    fx_scale = float(rng.uniform(0.25, 1.0))
    fy_scale = float(rng.uniform(0.25, 1.0))
    if rng.random() < 0.2:
        fx_scale *= 0.25
    if rng.random() < 0.2:
        fy_scale *= 0.25
    chirp_cap = float(chirp_weight_max) if family == "chirp_mix" else min(float(chirp_weight_max), 0.20)
    burst_cap = float(burst_weight_max) if family == "burst" else min(float(burst_weight_max), 0.35)
    chirp_weight = float(rng.uniform(0.0, max(0.0, chirp_cap)))
    burst_weight = float(rng.uniform(0.0, max(0.0, burst_cap)))
    name = f"rand_{split}_{case_id:04d}_{family}_{profile}".replace(".", "p")
    return RandomLoadSpec(
        split=split,
        case_id=case_id,
        case_name=name,
        seed=seed,
        family=family,
        spatial_profile=profile,
        n_components=n_components,
        fx_scale=fx_scale,
        fy_scale=fy_scale,
        chirp_weight=chirp_weight,
        burst_weight=burst_weight,
        freq_min_hz=float(freq_min),
        freq_max_hz=float(freq_max),
    )


def generate_case(
        spec: RandomLoadSpec,
        span: np.ndarray,
        duration: float,
        dt: float,
        fx_amp: float,
        fy_amp: float,
        static_fx_root: float,
        static_fx_tip: float,
        ramp_seconds: float,
        rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, object]]:
    n_nodes = len(span)
    n_steps = int(round(float(duration) / float(dt))) + 1
    t = np.linspace(0.0, float(duration), n_steps, dtype=np.float64)
    env = smooth_ramp(t, duration=duration, ramp_seconds=ramp_seconds)
    env *= random_envelope(t, duration=duration, rng=rng, burst_weight=spec.burst_weight)

    wx = spatial_profile(span, spec.spatial_profile, rng)
    wy = spatial_profile(span, spec.spatial_profile, rng)
    fx_base = static_fx_root + (static_fx_tip - static_fx_root) * span

    fx_sig, fx_freqs, fx_amps, fx_phases = multisine(
        t,
        rng,
        n_components=spec.n_components,
        freq_min=spec.freq_min_hz,
        freq_max=spec.freq_max_hz,
    )
    fy_sig, fy_freqs, fy_amps, fy_phases = multisine(
        t,
        rng,
        n_components=spec.n_components,
        freq_min=spec.freq_min_hz,
        freq_max=spec.freq_max_hz,
    )

    if spec.chirp_weight > 0.0:
        fx_chirp, fx_chirp_f0, fx_chirp_f1 = chirp_signal(
            t, rng, duration, spec.freq_min_hz, spec.freq_max_hz
        )
        fy_chirp, fy_chirp_f0, fy_chirp_f1 = chirp_signal(
            t, rng, duration, spec.freq_min_hz, spec.freq_max_hz
        )
        fx_sig = (1.0 - spec.chirp_weight) * fx_sig + spec.chirp_weight * fx_chirp
        fy_sig = (1.0 - spec.chirp_weight) * fy_sig + spec.chirp_weight * fy_chirp
    else:
        fx_chirp_f0 = fx_chirp_f1 = fy_chirp_f0 = fy_chirp_f1 = float("nan")

    fx_sig /= max(float(np.max(np.abs(fx_sig))), 1.0e-12)
    fy_sig /= max(float(np.max(np.abs(fy_sig))), 1.0e-12)

    data = np.zeros((n_steps, 1 + n_nodes * 6), dtype=np.float64)
    data[:, 0] = t
    for node_idx in range(n_nodes):
        base = 1 + 6 * node_idx
        data[:, base + 0] = fx_base[node_idx] + fx_amp * spec.fx_scale * fx_sig * env * wx[node_idx]
        data[:, base + 1] = fy_amp * spec.fy_scale * fy_sig * env * wy[node_idx]

    meta = {
        **asdict(spec),
        "n_steps": int(n_steps),
        "duration": float(duration),
        "dt": float(dt),
        "fx_freqs_hz": ";".join(f"{v:.6g}" for v in fx_freqs),
        "fy_freqs_hz": ";".join(f"{v:.6g}" for v in fy_freqs),
        "fx_component_amps": ";".join(f"{v:.6g}" for v in fx_amps),
        "fy_component_amps": ";".join(f"{v:.6g}" for v in fy_amps),
        "fx_component_phases": ";".join(f"{v:.6g}" for v in fx_phases),
        "fy_component_phases": ";".join(f"{v:.6g}" for v in fy_phases),
        "fx_chirp_f0_hz": fx_chirp_f0,
        "fx_chirp_f1_hz": fx_chirp_f1,
        "fy_chirp_f0_hz": fy_chirp_f0,
        "fy_chirp_f1_hz": fy_chirp_f1,
        "fx_max_abs": float(np.max(np.abs(data[:, 1::6]))),
        "fy_max_abs": float(np.max(np.abs(data[:, 2::6]))),
        "fx_dynamic_max_abs": float(np.max(np.abs(data[:, 1::6] - fx_base[None, :]))),
        "fy_dynamic_max_abs": float(np.max(np.abs(data[:, 2::6]))),
    }
    return data, meta


def write_dat(path: Path, data: np.ndarray, n_nodes: int, span: np.ndarray, headers: list[str], precision: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"{n_nodes}\n")
        f.write(" ".join(f"{v:.4f}" for v in span) + "\n")
        f.write(" ".join(headers) + "\n")
        value_fmt = f"{{:.{precision}e}}"
        for row in data:
            f.write(" ".join(["{:.3f}".format(row[0])] + [value_fmt.format(v) for v in row[1:]]) + "\n")


def write_manifest_and_lists(output_root: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with (output_root / "random_continuous_load_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (output_root / "random_continuous_load_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    for split in ("train", "valid", "test"):
        paths = [str(row["path"]) for row in rows if row["split"] == split]
        (output_root / f"{split}_load_files.txt").write_text(",".join(paths), encoding="utf-8")
        (output_root / f"{split}_load_files_lines.txt").write_text("\n".join(paths) + ("\n" if paths else ""), encoding="utf-8")


def parse_profiles(value: str) -> list[str]:
    profiles = [v.strip() for v in value.split(",") if v.strip()]
    if not profiles:
        raise ValueError("at least one spatial profile is required")
    return profiles


def parse_families(value: str) -> list[str]:
    families = [v.strip() for v in value.split(",") if v.strip()]
    allowed = {"single", "multi", "burst", "chirp_mix"}
    bad = [v for v in families if v not in allowed]
    if bad:
        raise ValueError(f"Unsupported families {bad}; allowed: {sorted(allowed)}")
    if not families:
        raise ValueError("at least one family is required")
    return families


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("data/load/random_continuous_alpha_v1"))
    parser.add_argument("--reference-load-file", type=Path, default=Path("data/load/test/test_complex_case_1.dat"))
    parser.add_argument("--n-train", type=int, default=96)
    parser.add_argument("--n-valid", type=int, default=24)
    parser.add_argument("--n-test", type=int, default=24)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--freq-min", type=float, default=0.05)
    parser.add_argument("--freq-max", type=float, default=0.80)
    parser.add_argument("--max-components", type=int, default=4)
    parser.add_argument("--fx-amp", type=float, default=1800.0)
    parser.add_argument("--fy-amp", type=float, default=1800.0)
    parser.add_argument("--static-fx-root", type=float, default=2000.0)
    parser.add_argument("--static-fx-tip", type=float, default=12000.0)
    parser.add_argument("--ramp-seconds", type=float, default=0.30)
    parser.add_argument("--precision", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--families",
        type=str,
        default="single,multi,burst,chirp_mix",
        help="Comma-separated load families: single,multi,burst,chirp_mix.",
    )
    parser.add_argument("--chirp-weight-max", type=float, default=0.45)
    parser.add_argument("--burst-weight-max", type=float, default=1.0)
    parser.add_argument(
        "--spatial-profiles",
        type=str,
        default="tip,last5,last10,fullspan_linear,fullspan_smooth,fullspan_quadratic,midspan_gaussian,random_smooth",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.reference_load_file.exists():
        n_nodes, span, headers = read_reference_header(args.reference_load_file)
    else:
        n_nodes = 48
        span = default_span_positions(n_nodes)
        headers = default_headers(n_nodes)

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    profiles = parse_profiles(args.spatial_profiles)
    families = parse_families(args.families)
    counts = split_counts(args.n_train, args.n_valid, args.n_test)
    root_rng = np.random.default_rng(int(args.seed))
    rows: list[dict[str, object]] = []

    print("[Generate Random Continuous Load DAT Files]")
    print(f"  output_root = {output_root.resolve()}")
    print(f"  reference   = {args.reference_load_file}")
    print(f"  n_nodes     = {n_nodes}")
    print(f"  counts      = train {args.n_train}, valid {args.n_valid}, test {args.n_test}")
    print(f"  freq_range  = [{args.freq_min}, {args.freq_max}] Hz")
    print(f"  families    = {','.join(families)}")

    for split, count in counts.items():
        for i in range(1, count + 1):
            case_seed = int(root_rng.integers(1, 2**31 - 1))
            rng = np.random.default_rng(case_seed)
            spec = sample_spec(
                split=split,
                case_id=i,
                seed=case_seed,
                freq_min=float(args.freq_min),
                freq_max=float(args.freq_max),
                max_components=int(args.max_components),
                profile_names=profiles,
                family_names=families,
                chirp_weight_max=float(args.chirp_weight_max),
                burst_weight_max=float(args.burst_weight_max),
                rng=rng,
            )
            out_path = output_root / split / f"{spec.case_name}.dat"
            if out_path.exists() and not args.overwrite:
                print(f"  [skip] {out_path}")
                continue
            data, meta = generate_case(
                spec=spec,
                span=span,
                duration=float(args.duration),
                dt=float(args.dt),
                fx_amp=float(args.fx_amp),
                fy_amp=float(args.fy_amp),
                static_fx_root=float(args.static_fx_root),
                static_fx_tip=float(args.static_fx_tip),
                ramp_seconds=float(args.ramp_seconds),
                rng=rng,
            )
            write_dat(out_path, data, n_nodes, span, headers, int(args.precision))
            meta["path"] = str(out_path.as_posix())
            rows.append(meta)
            print(
                f"  [{split:<5}] {spec.case_name:<48} "
                f"family={spec.family:<9} profile={spec.spatial_profile:<18} "
                f"n_comp={spec.n_components}"
            )

    write_manifest_and_lists(output_root, rows)
    print("[Saved]")
    print(f"  manifest = {output_root / 'random_continuous_load_manifest.csv'}")
    print(f"  train    = {output_root / 'train_load_files_lines.txt'}")
    print(f"  valid    = {output_root / 'valid_load_files_lines.txt'}")
    print(f"  test     = {output_root / 'test_load_files_lines.txt'}")


if __name__ == "__main__":
    main()
