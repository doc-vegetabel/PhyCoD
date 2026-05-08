from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def _find_header_line(lines: List[str]) -> int:
    """
    找到以 'Time' 开头的表头行。
    """
    for i, line in enumerate(lines):
        if line.strip().startswith("Time"):
            return i
    raise ValueError("未在 .out 文件中找到以 'Time' 开头的表头行。")


def _build_6dof_columns(node_start: int, node_end: int) -> List[str]:
    """
    构建 6DOF 列名顺序：
      [ux, uy, uz, rx, ry, rz] * n_nodes
    对应 BeamDyn 输出列：
      TDxr, TDyr, TDzr, RDxr, RDyr, RDzr
    """
    cols = []
    for node in range(node_start, node_end + 1):
        prefix = f"N{node:03d}_"
        cols.extend(
            [
                prefix + "TDxr",
                prefix + "TDyr",
                prefix + "TDzr",
                prefix + "RDxr",
                prefix + "RDyr",
                prefix + "RDzr",
            ]
        )
    return cols


def load_teacher_fullfield_6dof_from_out(
    out_path: str | Path,
    node_start: int = 2,
    node_end: int = 49,
    demean: bool = False,
    return_time: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """
    从 BeamDyn .out 文件提取 6DOF teacher 响应，输出 shape:
      (T, n_nodes * 6)

    每个节点顺序:
      [ux, uy, uz, rx, ry, rz]

    这里会自动跳过表头下一行的单位行，例如:
      (s) (m) (m) (m) (rad) ...
    """
    out_path = Path(out_path)
    if not out_path.exists():
        raise FileNotFoundError(f".out 文件不存在: {out_path}")

    with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    header_idx = _find_header_line(lines)

    # 解析列名行
    header_cols = lines[header_idx].strip().split()
    if "Time" not in header_cols:
        raise ValueError(f"表头行解析失败，header_cols={header_cols[:10]}")

    # skiprows:
    #   header_idx      -> 这一行是列名
    #   header_idx + 1  -> 下一行通常是单位行，需要跳过
    df = pd.read_csv(
        out_path,
        sep=r"\s+",
        engine="python",
        skiprows=header_idx + 2,
        names=header_cols,
    )

    required_cols = _build_6dof_columns(node_start=node_start, node_end=node_end)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"teacher .out 缺少 6DOF 列。缺失列数量={len(missing)}，前10个缺失列={missing[:10]}"
        )

    # 强制数值化，若还有非数值内容会变成 NaN
    data_df = df[required_cols].apply(pd.to_numeric, errors="coerce")
    time_df = pd.to_numeric(df["Time"], errors="coerce")

    # 删除含 NaN 的坏行（通常不会有，除非文件里混入额外文本）
    time_valid = (~time_df.isna()).to_numpy(copy=True)
    data_valid = (~data_df.isna().any(axis=1)).to_numpy(copy=True)
    valid_mask = np.logical_and(time_valid, data_valid)

    time = time_df.to_numpy(dtype=np.float32, copy=True)[valid_mask]
    data = data_df.to_numpy(dtype=np.float32, copy=True)[valid_mask]
    if demean:
        data = data - data.mean(axis=0, keepdims=True)

    if return_time:
        return time, data

    return data


def get_teacher_6dof_column_names(
    node_start: int = 2,
    node_end: int = 49,
) -> List[str]:
    return _build_6dof_columns(node_start=node_start, node_end=node_end)