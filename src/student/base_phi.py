from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_BASE_PHI_COLUMN = "initial_twist_deg"
DEFAULT_BASE_PHI_SIGN = -1.0


def build_base_phi_from_blade_csv(
    blade_csv: str | Path,
    *,
    twist_column: str = DEFAULT_BASE_PHI_COLUMN,
    sign: float = DEFAULT_BASE_PHI_SIGN,
    expected_n_stations: int | None = 49,
) -> dict[str, Any]:
    """
    Build fixed base-student principal-axis rotation Phi(s)
    from blade_master.csv.

    The raw blade file stores station-level initial twist.
    FEM assembly uses element-level Phi, so we use adjacent averaging.

    By current project convention:
        Phi_base(s) = - initial_twist_deg(s)

    Returns:
        {
            "phi_station_deg": shape (n_stations,),
            "phi_element_deg": shape (n_stations - 1,),
            "twist_station_deg": shape (n_stations,),
            "summary": dict,
        }
    """
    blade_csv = Path(blade_csv).resolve()
    if not blade_csv.exists():
        raise FileNotFoundError(f"blade_csv not found: {blade_csv}")

    df = pd.read_csv(blade_csv)

    if twist_column not in df.columns:
        raise KeyError(
            f"Column '{twist_column}' not found in {blade_csv}. "
            f"Available columns: {list(df.columns)}"
        )

    twist_station_deg = df[twist_column].to_numpy(dtype=np.float64)

    if twist_station_deg.ndim != 1:
        raise ValueError(f"{twist_column} must be a 1D column.")

    if expected_n_stations is not None and len(twist_station_deg) != expected_n_stations:
        raise ValueError(
            f"Expected {expected_n_stations} station twist values, "
            f"but got {len(twist_station_deg)} from {blade_csv}."
        )

    phi_station_deg = float(sign) * twist_station_deg
    phi_element_deg = 0.5 * (phi_station_deg[:-1] + phi_station_deg[1:])

    return {
        "blade_csv": str(blade_csv),
        "twist_column": twist_column,
        "sign": float(sign),
        "twist_station_deg": twist_station_deg,
        "phi_station_deg": phi_station_deg,
        "phi_element_deg": phi_element_deg,
        "summary": {
            "n_stations": int(len(phi_station_deg)),
            "n_elements": int(len(phi_element_deg)),
            "twist_min_deg": float(np.min(twist_station_deg)),
            "twist_max_deg": float(np.max(twist_station_deg)),
            "twist_mean_deg": float(np.mean(twist_station_deg)),
            "phi_station_min_deg": float(np.min(phi_station_deg)),
            "phi_station_max_deg": float(np.max(phi_station_deg)),
            "phi_station_mean_deg": float(np.mean(phi_station_deg)),
            "phi_element_min_deg": float(np.min(phi_element_deg)),
            "phi_element_max_deg": float(np.max(phi_element_deg)),
            "phi_element_mean_deg": float(np.mean(phi_element_deg)),
        },
    }


def get_base_phi_element_deg(
    blade_csv: str | Path,
    *,
    twist_column: str = DEFAULT_BASE_PHI_COLUMN,
    sign: float = DEFAULT_BASE_PHI_SIGN,
) -> np.ndarray:
    """
    Convenience wrapper returning only element-level Phi.
    """
    info = build_base_phi_from_blade_csv(
        blade_csv=blade_csv,
        twist_column=twist_column,
        sign=sign,
    )
    return np.asarray(info["phi_element_deg"], dtype=np.float64)


def print_base_phi_summary(phi_info: dict[str, Any], *, prefix: str = "[Base Phi]") -> None:
    summary = phi_info["summary"]

    print()
    print(prefix)
    print(f"  blade_csv             = {phi_info['blade_csv']}")
    print(f"  twist_column          = {phi_info['twist_column']}")
    print(f"  sign                  = {phi_info['sign']:+.3f}")
    print(f"  n_stations            = {summary['n_stations']}")
    print(f"  n_elements            = {summary['n_elements']}")
    print(
        "  twist_station_deg     "
        f"min={summary['twist_min_deg']:.6f}, "
        f"max={summary['twist_max_deg']:.6f}, "
        f"mean={summary['twist_mean_deg']:.6f}"
    )
    print(
        "  phi_element_deg       "
        f"min={summary['phi_element_min_deg']:.6f}, "
        f"max={summary['phi_element_max_deg']:.6f}, "
        f"mean={summary['phi_element_mean_deg']:.6f}"
    )