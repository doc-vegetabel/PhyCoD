import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROLE_KEYWORDS = {
    "teacher": ["teacher", "target", "truth", "beamdyn", "reference", "ref"],
    "transformer": ["transformer", "pred", "prediction", "trained", "dynamic"],
    "static": ["static", "kappa", "base_static", "base_static_kappa_y"],
}


def norm_name(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def token_hit(name: str, token: str) -> bool:
    n = norm_name(name)
    parts = n.split("_")
    return token in parts or token in n


def dir_hit(name: str, direction: str) -> bool:
    n = norm_name(name)
    parts = n.split("_")
    if direction in parts:
        return True
    return bool(re.search(rf"(^|_){direction}($|_)", n)) or bool(re.search(rf"{direction}$", n))


def role_score(name: str, role: str) -> int:
    n = norm_name(name)
    score = 0
    for kw in ROLE_KEYWORDS.get(role, []):
        if kw in n:
            score += 10
    return score


def choose_time_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        n = norm_name(c)
        if n in ["time", "t", "time_s", "time_sec", "time_seconds"]:
            return c
    for c in df.columns:
        if "time" in norm_name(c):
            return c
    return df.columns[0]


def numeric_cols(df: pd.DataFrame, time_col: str):
    cols = []
    for c in df.columns:
        if c == time_col:
            continue
        converted = pd.to_numeric(df[c], errors="coerce")
        if converted.notna().sum() >= max(4, int(0.8 * len(converted))):
            df[c] = converted
            cols.append(c)
    return cols


def find_best_col(cols, role, direction, obs):
    best = None
    best_score = -10**9

    for c in cols:
        if obs and not token_hit(c, obs):
            continue

        sc = role_score(c, role)
        if sc <= 0:
            continue

        if dir_hit(c, direction):
            sc += 5
        else:
            sc -= 8

        n = norm_name(c)
        if any(bad in n for bad in ["error", "err", "force", "load", "theta", "alpha", "lag", "mse"]):
            sc -= 20

        if sc > best_score:
            best_score = sc
            best = c

    if best_score < 8:
        return None
    return best


def safe_rms_zero_mean(x):
    x = np.asarray(x, float)
    x = x - np.mean(x)
    return float(np.sqrt(np.mean(x * x)))


def finite_pair(t, ref, pred, base=None):
    arrays = [np.asarray(t, float), np.asarray(ref, float), np.asarray(pred, float)]
    if base is not None:
        arrays.append(np.asarray(base, float))

    mask = np.ones_like(arrays[0], dtype=bool)
    for a in arrays:
        mask &= np.isfinite(a)

    arrays = [a[mask] for a in arrays]
    if base is None:
        return arrays[0], arrays[1], arrays[2], None
    return arrays[0], arrays[1], arrays[2], arrays[3]


def corr_at_lag(ref, pred, lag):
    ref = np.asarray(ref, float)
    pred = np.asarray(pred, float)

    if lag > 0:
        a = ref[:-lag]
        b = pred[lag:]
    elif lag < 0:
        a = ref[-lag:]
        b = pred[:lag]
    else:
        a = ref
        b = pred

    if len(a) < 4:
        return np.nan

    a = a - np.mean(a)
    b = b - np.mean(b)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den <= 1e-30:
        return np.nan
    return float(np.dot(a, b) / den)


def best_lag_seconds(t, ref, pred, max_lag_seconds):
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return np.nan, np.nan

    max_lag = int(round(max_lag_seconds / dt))
    max_lag = max(0, min(max_lag, len(t) // 2))

    lags = np.arange(-max_lag, max_lag + 1, dtype=int)
    corrs = np.array([corr_at_lag(ref, pred, int(l)) for l in lags], dtype=float)

    if np.all(~np.isfinite(corrs)):
        return np.nan, np.nan

    idx = int(np.nanargmax(corrs))

    # 正值表示 transformer 相对 teacher 滞后。
    return float(lags[idx] * dt), float(corrs[idx])


def fft_features(t, ref, pred, freq_min, freq_max):
    t = np.asarray(t, float)
    ref = np.asarray(ref, float)
    pred = np.asarray(pred, float)

    dt = float(np.median(np.diff(t)))
    n = len(t)

    if n < 8 or not np.isfinite(dt) or dt <= 0:
        return {}

    window = np.hanning(n)
    ref_z = (ref - np.mean(ref)) * window
    pred_z = (pred - np.mean(pred)) * window

    freqs = np.fft.rfftfreq(n, dt)
    ref_fft = np.fft.rfft(ref_z)
    pred_fft = np.fft.rfft(pred_z)

    mag_ref = np.abs(ref_fft)
    mag_pred = np.abs(pred_fft)

    band = (freqs >= freq_min) & (freqs <= freq_max) & (freqs > 0)
    if not np.any(band):
        return {}

    idxs = np.where(band)[0]
    k = int(idxs[np.argmax(mag_ref[idxs])])

    f_peak = float(freqs[k])
    phase_ref = float(np.angle(ref_fft[k]))
    phase_pred = float(np.angle(pred_fft[k]))

    phase_diff = float(np.angle(np.exp(1j * (phase_pred - phase_ref))))
    amp_ratio_fft = float(mag_pred[k] / (mag_ref[k] + 1e-30))
    phase_time_equiv = float(phase_diff / (2.0 * math.pi * f_peak)) if f_peak > 0 else np.nan

    return {
        "fft_peak_freq_hz": f_peak,
        "fft_amp_ratio_at_teacher_peak": amp_ratio_fft,
        "fft_phase_diff_rad_pred_minus_teacher": phase_diff,
        "fft_phase_time_equiv_s_pred_minus_teacher": phase_time_equiv,
    }


def analyze_pair(t, ref, pred, base, label, out_dir, freq_min, freq_max, max_lag_seconds):
    t, ref, pred, base = finite_pair(t, ref, pred, base)

    err = pred - ref

    row = {
        "label": label,
        "n": len(t),
        "mse": float(np.mean(err * err)),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "mae": float(np.mean(np.abs(err))),
        "teacher_pp_amp": float(np.max(ref) - np.min(ref)),
        "pred_pp_amp": float(np.max(pred) - np.min(pred)),
        "pp_amp_ratio_pred_over_teacher": float(
            (np.max(pred) - np.min(pred)) / ((np.max(ref) - np.min(ref)) + 1e-30)
        ),
        "teacher_rms_zm": safe_rms_zero_mean(ref),
        "pred_rms_zm": safe_rms_zero_mean(pred),
        "rms_amp_ratio_pred_over_teacher": float(
            safe_rms_zero_mean(pred) / (safe_rms_zero_mean(ref) + 1e-30)
        ),
    }

    lag_s, lag_corr = best_lag_seconds(t, ref, pred, max_lag_seconds)
    row["best_corr_lag_s_positive_pred_lags"] = lag_s
    row["best_corr_value"] = lag_corr

    row.update(fft_features(t, ref, pred, freq_min, freq_max))

    if base is not None:
        berr = base - ref
        row["baseline_mse"] = float(np.mean(berr * berr))
        row["baseline_rmse"] = float(np.sqrt(np.mean(berr * berr)))
        row["baseline_mae"] = float(np.mean(np.abs(berr)))

        b_lag_s, b_lag_corr = best_lag_seconds(t, ref, base, max_lag_seconds)
        row["baseline_best_corr_lag_s_positive_base_lags"] = b_lag_s
        row["baseline_best_corr_value"] = b_lag_corr
        row["baseline_pp_amp_ratio_over_teacher"] = float(
            (np.max(base) - np.min(base)) / ((np.max(ref) - np.min(ref)) + 1e-30)
        )
        row["baseline_rms_amp_ratio_over_teacher"] = float(
            safe_rms_zero_mean(base) / (safe_rms_zero_mean(ref) + 1e-30)
        )

    safe_label = re.sub(r"[^a-zA-Z0-9_.-]+", "_", label)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(t, ref, label="teacher")
    plt.plot(t, pred, label="transformer")
    if base is not None:
        plt.plot(t, base, label="static_baseline")
    plt.xlabel("time [s]")
    plt.ylabel(label)
    plt.title(f"Time overlay: {label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{safe_label}_time_overlay.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(t, err, label="transformer - teacher")
    if base is not None:
        plt.plot(t, base - ref, label="static - teacher")
    plt.xlabel("time [s]")
    plt.ylabel("error")
    plt.title(f"Error: {label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{safe_label}_error.png", dpi=180)
    plt.close()

    dt = float(np.median(np.diff(t)))
    if len(t) >= 8 and np.isfinite(dt) and dt > 0:
        window = np.hanning(len(t))
        freqs = np.fft.rfftfreq(len(t), dt)

        ref_mag = np.abs(np.fft.rfft((ref - np.mean(ref)) * window))
        pred_mag = np.abs(np.fft.rfft((pred - np.mean(pred)) * window))
        band = (freqs >= freq_min) & (freqs <= freq_max)

        plt.figure(figsize=(10, 4))
        plt.plot(freqs[band], ref_mag[band], label="teacher")
        plt.plot(freqs[band], pred_mag[band], label="transformer")
        if base is not None:
            base_mag = np.abs(np.fft.rfft((base - np.mean(base)) * window))
            plt.plot(freqs[band], base_mag[band], label="static_baseline")
        plt.xlabel("frequency [Hz]")
        plt.ylabel("FFT magnitude")
        plt.title(f"Spectrum: {label}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{safe_label}_spectrum.png", dpi=180)
        plt.close()

    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--freq-min", type=float, default=0.05)
    ap.add_argument("--freq-max", type=float, default=5.0)
    ap.add_argument("--max-lag-seconds", type=float, default=0.80)
    ap.add_argument("--list-columns", action="store_true")
    ap.add_argument("--teacher-col", default=None)
    ap.add_argument("--pred-col", default=None)
    ap.add_argument("--baseline-col", default=None)
    ap.add_argument("--time-col", default=None)
    ap.add_argument("--obs-order", default="tip,last5,mean,full")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    if args.list_columns:
        print("[Available columns]")
        for c in df.columns:
            print(c)
        return

    time_col = args.time_col or choose_time_col(df)
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    cols = numeric_cols(df, time_col)

    t = df[time_col].to_numpy(dtype=float)
    rows = []

    if args.teacher_col and args.pred_col:
        base = df[args.baseline_col].to_numpy(dtype=float) if args.baseline_col else None
        rows.append(
            analyze_pair(
                t,
                df[args.teacher_col].to_numpy(dtype=float),
                df[args.pred_col].to_numpy(dtype=float),
                base,
                label=f"manual_{args.pred_col}_vs_{args.teacher_col}",
                out_dir=out_dir,
                freq_min=args.freq_min,
                freq_max=args.freq_max,
                max_lag_seconds=args.max_lag_seconds,
            )
        )
    else:
        obs_order = [x.strip() for x in args.obs_order.split(",") if x.strip()]

        for obs in obs_order:
            for direction in ["x", "y"]:
                teacher_col = find_best_col(cols, "teacher", direction, obs)
                pred_col = find_best_col(cols, "transformer", direction, obs)
                static_col = find_best_col(cols, "static", direction, obs)

                if teacher_col and pred_col:
                    label = f"{obs}_{direction}"
                    print(f"[Detect] {label}: teacher={teacher_col} transformer={pred_col} static={static_col}")

                    base = df[static_col].to_numpy(dtype=float) if static_col else None

                    rows.append(
                        analyze_pair(
                            t,
                            df[teacher_col].to_numpy(dtype=float),
                            df[pred_col].to_numpy(dtype=float),
                            base,
                            label=label,
                            out_dir=out_dir,
                            freq_min=args.freq_min,
                            freq_max=args.freq_max,
                            max_lag_seconds=args.max_lag_seconds,
                        )
                    )

    if not rows:
        print("[ERROR] No signal pair was detected.")
        print("Please inspect available columns with:")
        print(f'python tests/diagnose_selected_timeseries.py --csv "{csv_path}" --out-dir "{out_dir}" --list-columns')
        raise SystemExit(2)

    summary = pd.DataFrame(rows)
    summary_path = out_dir / "diagnostic_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("")
    print("[Saved]")
    print(f"summary_csv = {summary_path}")
    print(f"plots_dir   = {out_dir}")

    show_cols = [
        "label",
        "mse",
        "pp_amp_ratio_pred_over_teacher",
        "rms_amp_ratio_pred_over_teacher",
        "best_corr_lag_s_positive_pred_lags",
        "best_corr_value",
        "fft_peak_freq_hz",
        "fft_amp_ratio_at_teacher_peak",
        "fft_phase_diff_rad_pred_minus_teacher",
        "fft_phase_time_equiv_s_pred_minus_teacher",
    ]
    show_cols = [c for c in show_cols if c in summary.columns]

    print("")
    print("[Diagnostic Summary]")
    print(summary[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
