from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


PHASE_HISTORY_HINTS = (
    "phase_gate",
    "theta_fast",
    "theta_gated_fast",
    "theta_slow",
    "phase_reg",
)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def _to_float(value: Any, default: float = math.nan) -> float:
    if value is None:
        return default
    try:
        if isinstance(value, str) and value.strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _finite_array(values: list[Any]) -> np.ndarray:
    out = np.asarray([_to_float(v) for v in values], dtype=np.float64)
    return out[np.isfinite(out)]


def _column_array(rows: list[dict[str, str]], column: str) -> np.ndarray:
    return np.asarray([_to_float(row.get(column)) for row in rows], dtype=np.float64)


def _mean_or_nan(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else math.nan


def _max_or_nan(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.max(values)) if values.size else math.nan


def _rms_or_nan(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan
    return float(np.sqrt(np.mean(values * values)))


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size < 3:
        return math.nan
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if denom <= 0.0:
        return math.nan
    return float(np.sum(a * b) / denom)


def _lagged_pair(
    teacher: np.ndarray,
    pred: np.ndarray,
    lag_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    if lag_steps > 0:
        return teacher[:-lag_steps], pred[lag_steps:]
    if lag_steps < 0:
        return teacher[-lag_steps:], pred[:lag_steps]
    return teacher, pred


def _late_lag_metrics(
    *,
    rows: list[dict[str, str]],
    teacher_col: str,
    pred_col: str,
    late_start: float,
    late_end: float | None,
    max_lag_seconds: float,
) -> dict[str, float]:
    if not rows or teacher_col not in rows[0] or pred_col not in rows[0] or "time" not in rows[0]:
        return {
            "best_corr_lag_s": math.nan,
            "corr": math.nan,
            "rms_amp_ratio": math.nan,
            "late_window_n": 0.0,
        }

    time = _column_array(rows, "time")
    teacher = _column_array(rows, teacher_col)
    pred = _column_array(rows, pred_col)

    mask = np.isfinite(time) & np.isfinite(teacher) & np.isfinite(pred)
    mask = mask & (time >= float(late_start))
    if late_end is not None:
        mask = mask & (time <= float(late_end))

    time = time[mask]
    teacher = teacher[mask]
    pred = pred[mask]
    if time.size < 4:
        return {
            "best_corr_lag_s": math.nan,
            "corr": math.nan,
            "rms_amp_ratio": math.nan,
            "late_window_n": float(time.size),
        }

    dt_values = np.diff(time)
    dt_values = dt_values[np.isfinite(dt_values) & (dt_values > 0.0)]
    if dt_values.size == 0:
        dt = 1.0
    else:
        dt = float(np.median(dt_values))

    max_lag_steps = max(0, int(round(float(max_lag_seconds) / dt)))
    best_corr = -math.inf
    best_lag_steps = 0
    for lag_steps in range(-max_lag_steps, max_lag_steps + 1):
        teacher_lag, pred_lag = _lagged_pair(teacher, pred, lag_steps)
        corr = _pearson_corr(teacher_lag, pred_lag)
        if np.isfinite(corr) and corr > best_corr:
            best_corr = corr
            best_lag_steps = lag_steps

    teacher_centered = teacher - np.mean(teacher)
    pred_centered = pred - np.mean(pred)
    teacher_rms = float(np.sqrt(np.mean(teacher_centered * teacher_centered)))
    pred_rms = float(np.sqrt(np.mean(pred_centered * pred_centered)))
    amp_ratio = pred_rms / teacher_rms if teacher_rms > 0.0 else math.nan

    return {
        "best_corr_lag_s": float(best_lag_steps * dt),
        "corr": float(best_corr) if np.isfinite(best_corr) else math.nan,
        "rms_amp_ratio": float(amp_ratio),
        "late_window_n": float(time.size),
    }


def _history_summary(history_csv: Path) -> dict[str, Any]:
    rows = _read_csv_rows(history_csv)
    if not rows:
        return {
            "path": str(history_csv),
            "exists": history_csv.exists(),
            "num_rows": 0,
            "phase_columns": [],
            "final_row": {},
            "best_row": {},
        }

    columns = list(rows[0].keys())
    phase_columns = [
        col for col in columns
        if any(hint in col for hint in PHASE_HISTORY_HINTS)
    ]

    final_row = rows[-1]
    best_epoch = _to_float(final_row.get("best_epoch"))
    best_row = {}
    if np.isfinite(best_epoch):
        for row in rows:
            if _to_float(row.get("epoch")) == best_epoch:
                best_row = row
                break
    if not best_row:
        finite_score_rows = [
            row for row in rows
            if np.isfinite(_to_float(row.get("score")))
        ]
        if finite_score_rows:
            best_row = min(finite_score_rows, key=lambda row: _to_float(row.get("score")))

    return {
        "path": str(history_csv),
        "exists": True,
        "num_rows": len(rows),
        "phase_columns": phase_columns,
        "final_row": {k: final_row.get(k) for k in ["epoch", "score", "best_epoch", *phase_columns]},
        "best_row": {k: best_row.get(k) for k in ["epoch", "score", *phase_columns]} if best_row else {},
    }


def _discover_case_dirs(post_test_dir: Path) -> list[Path]:
    if not post_test_dir.exists():
        return []

    case_dirs: list[Path] = []
    if (post_test_dir / "metrics.csv").exists() or (post_test_dir / "selected_timeseries.csv").exists():
        case_dirs.append(post_test_dir)

    for child in sorted(post_test_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / "metrics.csv").exists() or (child / "selected_timeseries.csv").exists():
            case_dirs.append(child)

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in case_dirs:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def _metric_row_for_model(metrics_rows: list[dict[str, str]], model_name: str) -> dict[str, str]:
    for row in metrics_rows:
        if str(row.get("model", "")).strip().lower() == model_name:
            return row
    return {}


def _summarize_selected_timeseries(
    selected_rows: list[dict[str, str]],
    *,
    gate_active_threshold: float,
    late_start: float,
    late_end: float | None,
    max_lag_seconds: float,
) -> dict[str, float]:
    summary: dict[str, float] = {}

    if not selected_rows:
        for key in [
            "g_phase_mean",
            "g_phase_max",
            "g_phase_active_ratio",
            "alpha_x_gated_fast_rms",
            "alpha_x_gated_fast_abs_max",
            "alpha_xy_gated_fast_rms",
            "alpha_xy_gated_fast_abs_max",
        ]:
            summary[key] = math.nan
        return summary

    columns = list(selected_rows[0].keys())
    gate_cols = [
        col for col in columns
        if col == "g_phase" or col.startswith("g_phase_")
    ]
    gate_values = (
        np.concatenate([_column_array(selected_rows, col) for col in gate_cols])
        if gate_cols else np.asarray([], dtype=np.float64)
    )
    finite_gate = gate_values[np.isfinite(gate_values)]
    summary["g_phase_mean"] = _mean_or_nan(finite_gate)
    summary["g_phase_max"] = _max_or_nan(finite_gate)
    summary["g_phase_active_ratio"] = (
        float(np.mean(finite_gate > float(gate_active_threshold)))
        if finite_gate.size else math.nan
    )

    for param in ("alpha_x", "alpha_xy"):
        col = f"{param}_gated_fast"
        values = _column_array(selected_rows, col) if col in columns else np.asarray([], dtype=np.float64)
        summary[f"{param}_gated_fast_rms"] = _rms_or_nan(values)
        summary[f"{param}_gated_fast_abs_max"] = _max_or_nan(np.abs(values))

    for component in ("x", "y"):
        metrics = _late_lag_metrics(
            rows=selected_rows,
            teacher_col=f"teacher_tip_{component}",
            pred_col=f"transformer_tip_{component}",
            late_start=late_start,
            late_end=late_end,
            max_lag_seconds=max_lag_seconds,
        )
        for key, value in metrics.items():
            summary[f"tip_{component}_late_{key}"] = value

    return summary


def _summarize_case(
    case_dir: Path,
    *,
    post_test_dir: Path,
    history_info: dict[str, Any],
    gate_active_threshold: float,
    late_start: float,
    late_end: float | None,
    max_lag_seconds: float,
) -> dict[str, Any]:
    metrics_csv = case_dir / "metrics.csv"
    selected_csv = case_dir / "selected_timeseries.csv"
    metrics_summary_json = case_dir / "metrics_summary.json"

    metrics_rows = _read_csv_rows(metrics_csv)
    transformer_metrics = _metric_row_for_model(metrics_rows, "transformer")
    selected_rows = _read_csv_rows(selected_csv)
    metrics_summary = _read_json(metrics_summary_json)

    if case_dir.resolve() == post_test_dir.resolve():
        case_name = case_dir.name
    else:
        case_name = case_dir.relative_to(post_test_dir).as_posix()

    row: dict[str, Any] = {
        "case_name": case_name,
        "case_dir": str(case_dir),
        "load_file": metrics_summary.get("load_file", ""),
        "x_ratio_to_static": _to_float(transformer_metrics.get("full_x_mse_ratio_to_static")),
        "y_ratio_to_static": _to_float(transformer_metrics.get("full_y_mse_ratio_to_static")),
        "metrics_csv": str(metrics_csv) if metrics_csv.exists() else "",
        "selected_timeseries_csv": str(selected_csv) if selected_csv.exists() else "",
        "history_final_epoch": _to_float(history_info.get("final_row", {}).get("epoch")),
        "history_best_epoch": _to_float(history_info.get("final_row", {}).get("best_epoch")),
        "history_final_train_phase_gate_mean": _to_float(
            history_info.get("final_row", {}).get("train_phase_gate_mean")
        ),
        "history_final_valid_phase_gate_mean": _to_float(
            history_info.get("final_row", {}).get("valid_phase_gate_mean")
        ),
        "history_best_valid_phase_gate_mean": _to_float(
            history_info.get("best_row", {}).get("valid_phase_gate_mean")
        ),
    }

    row.update(
        _summarize_selected_timeseries(
            selected_rows,
            gate_active_threshold=gate_active_threshold,
            late_start=late_start,
            late_end=late_end,
            max_lag_seconds=max_lag_seconds,
        )
    )
    return row


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize phase-gated Transformer training history and post-test "
            "metrics/selected_timeseries outputs."
        )
    )
    parser.add_argument(
        "--run-dir",
        default=".",
        help="Training output directory containing training_history.csv and post_test/.",
    )
    parser.add_argument(
        "--history-csv",
        default=None,
        help="Optional explicit path to training_history.csv.",
    )
    parser.add_argument(
        "--post-test-dir",
        default=None,
        help="Optional explicit path to post_test output directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for phase_gated_diagnosis_summary.csv/json. Defaults to run-dir.",
    )
    parser.add_argument("--late-start", type=float, default=5.0)
    parser.add_argument("--late-end", type=float, default=None)
    parser.add_argument("--max-lag-seconds", type=float, default=0.50)
    parser.add_argument("--gate-active-threshold", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = Path(args.run_dir).resolve()
    history_csv = (
        Path(args.history_csv).resolve()
        if args.history_csv is not None
        else run_dir / "training_history.csv"
    )
    post_test_dir = (
        Path(args.post_test_dir).resolve()
        if args.post_test_dir is not None
        else run_dir / "post_test"
    )
    if not post_test_dir.exists() and (run_dir / "post_train_test").exists():
        post_test_dir = run_dir / "post_train_test"

    output_dir = Path(args.output_dir).resolve() if args.output_dir is not None else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    history_info = _history_summary(history_csv)
    case_dirs = _discover_case_dirs(post_test_dir)
    summaries = [
        _summarize_case(
            case_dir,
            post_test_dir=post_test_dir,
            history_info=history_info,
            gate_active_threshold=float(args.gate_active_threshold),
            late_start=float(args.late_start),
            late_end=args.late_end,
            max_lag_seconds=float(args.max_lag_seconds),
        )
        for case_dir in case_dirs
    ]

    csv_path = output_dir / "phase_gated_diagnosis_summary.csv"
    json_path = output_dir / "phase_gated_diagnosis_summary.json"

    _save_csv(csv_path, summaries)
    payload = {
        "run_dir": str(run_dir),
        "history": history_info,
        "post_test_dir": str(post_test_dir),
        "num_cases": len(summaries),
        "settings": {
            "late_start": float(args.late_start),
            "late_end": args.late_end,
            "max_lag_seconds": float(args.max_lag_seconds),
            "gate_active_threshold": float(args.gate_active_threshold),
        },
        "cases": summaries,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, ensure_ascii=False)

    print()
    print("[Phase-gated Diagnosis]")
    print(f"  history_csv = {history_csv}")
    print(f"  post_test_dir = {post_test_dir}")
    print(f"  cases = {len(summaries)}")
    print(f"  summary_csv = {csv_path}")
    print(f"  summary_json = {json_path}")


if __name__ == "__main__":
    main()
