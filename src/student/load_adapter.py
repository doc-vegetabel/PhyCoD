from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def read_multi_timeseries_load_file(load_file: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    读取与当前 teacher (BeamDyn stand-alone driver) 相同格式的多节点时历载荷文件。

    文件格式:
        NPoints
        eta_1 eta_2 ... eta_N
        time Fx1 Fy1 Fz1 Mx1 My1 Mz1 Fx2 Fy2 Fz2 Mx2 My2 Mz2 ... FxN FyN FzN MxN MyN MzN
        t1   ...
        t2   ...
        ...

    返回:
        load_time   : shape = (n_steps,)
        eta_points  : shape = (n_points,)
        load_series : shape = (n_steps, n_points, 6)

    其中:
        load_series[it, ip, :] = [Fx, Fy, Fz, Mx, My, Mz]
    """
    load_file = Path(load_file).resolve()
    if not load_file.exists():
        raise FileNotFoundError(f"找不到时历载荷文件: {load_file}")

    with open(load_file, "r", encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    if len(raw_lines) < 4:
        raise ValueError(
            f"载荷文件内容不足，至少应包含 4 行（NPoints, eta行, header, 数据行）: {load_file}"
        )

    # 第1行: NPoints
    try:
        n_points = int(raw_lines[0].split()[0])
    except Exception as e:
        raise ValueError(f"无法解析 NPoints: {raw_lines[0]}") from e

    if n_points <= 0:
        raise ValueError(f"NPoints 必须为正数，当前为: {n_points}")

    # 第2行: eta 列表
    eta_points = np.fromstring(raw_lines[1], sep=" ", dtype=np.float64)
    if eta_points.size != n_points:
        raise ValueError(
            f"eta 数量与 NPoints 不一致: NPoints={n_points}, eta_count={eta_points.size}"
        )

    if np.any(eta_points < 0.0) or np.any(eta_points > 1.0):
        raise ValueError(f"eta 必须全部位于 [0,1] 内，当前 eta={eta_points}")

    if np.any(np.diff(eta_points) < 0.0):
        raise ValueError(f"eta 应按非递减顺序给出，当前 eta={eta_points}")

    # 第3行是 header，从第4行开始读数值
    data = np.loadtxt(load_file, skiprows=3)
    if data.ndim == 1:
        data = data[None, :]

    expected_cols = 1 + 6 * n_points
    if data.shape[1] != expected_cols:
        raise ValueError(
            f"载荷文件列数错误: 期望 {expected_cols}, 实际 {data.shape[1]}"
        )

    load_time = data[:, 0].astype(np.float64)

    if load_time.size == 0:
        raise ValueError("载荷文件中没有有效时间步数据。")

    if np.any(np.diff(load_time) < 0.0):
        raise ValueError("载荷时间 load_time 必须单调非递减。")

    # 余下列 reshape 成 (n_steps, n_points, 6)
    load_series = data[:, 1:].reshape(data.shape[0], n_points, 6).astype(np.float64)

    return load_time, eta_points, load_series


def interp_multi_timeseries_load(
    t: float,
    load_time: np.ndarray,
    load_series: np.ndarray,
) -> np.ndarray:
    """
    对多节点时历载荷做时间插值，规则与当前 teacher 保持一致:

    1) t <= 第一帧时间: 首值保持
    2) 中间时刻: 线性插值
    3) t >= 最后一帧时间: 尾值保持

    输入:
        t          : 当前仿真时刻
        load_time  : shape = (n_steps,)
        load_series: shape = (n_steps, n_points, 6)

    返回:
        cur_loads  : shape = (n_points, 6)
    """
    if load_time.ndim != 1:
        raise ValueError("load_time 必须是一维数组。")
    if load_series.ndim != 3:
        raise ValueError("load_series 必须是三维数组 (n_steps, n_points, 6)。")
    if load_series.shape[0] != load_time.shape[0]:
        raise ValueError("load_series 的时间维长度必须与 load_time 一致。")
    if load_series.shape[2] != 6:
        raise ValueError("load_series 最后一维必须为 6，对应 [Fx,Fy,Fz,Mx,My,Mz]。")

    if t <= load_time[0]:
        return load_series[0].copy()

    if t >= load_time[-1]:
        return load_series[-1].copy()

    hi = int(np.searchsorted(load_time, t, side="right"))
    lo = hi - 1

    t0 = load_time[lo]
    t1 = load_time[hi]

    if abs(t1 - t0) < 1e-14:
        return load_series[lo].copy()

    alpha = (t - t0) / (t1 - t0)
    return (1.0 - alpha) * load_series[lo] + alpha * load_series[hi]


def get_student_free_node_eta(student_model) -> np.ndarray:
    """
    从 student_model 中获取自由节点对应的 eta。

    说明:
    - student_model.eta 包含完整节点, 从 root(0.0) 到 tip(1.0)
    - 当前 student FEM 在 build_fem_matrices_6dof() 中固定了根部节点全部 6 个自由度:
          M_free = M_global[6:, 6:]
          K_free = K_global[6:, 6:]
    - 因此自由节点正好对应 model.eta[1:]

    返回:
        free_node_eta : shape = (n_free_nodes,)
    """
    if not hasattr(student_model, "eta"):
        raise AttributeError("student_model 没有 eta 属性。")

    eta_all = np.asarray(student_model.eta, dtype=np.float64).reshape(-1)
    if eta_all.size < 2:
        raise ValueError("student_model.eta 至少需要包含 root 和另一个节点。")

    if abs(eta_all[0] - 0.0) > 1e-9:
        raise ValueError(f"student_model.eta[0] 应为 0.0，当前为 {eta_all[0]}")
    if abs(eta_all[-1] - 1.0) > 1e-9:
        raise ValueError(f"student_model.eta[-1] 应为 1.0，当前为 {eta_all[-1]}")

    if np.any(np.diff(eta_all) <= 0.0):
        raise ValueError("student_model.eta 必须严格递增。")

    free_node_eta = eta_all[1:].copy()
    return free_node_eta


def distribute_point_load_by_eta(
    F_row: np.ndarray,
    point_load_6: np.ndarray,
    eta: float,
    free_node_eta: np.ndarray,
) -> None:
    """
    把一个 6分量点载荷按 eta 分配到 student 自由节点对应的 DOF 上。

    当前采取的映射策略（第一版，适合 teacher-student 对齐排错）：
    - 若 eta 落在两个自由节点之间，则按线性权重分配到相邻两个节点
    - 若 eta 小于第一个自由节点，则全部施加给第一个自由节点
    - 若 eta 大于最后一个自由节点，则全部施加给最后一个自由节点

    DOF 排布约定（与你当前 student/fem_builder.py 一致）：
        每个节点 6 个自由度:
        [ux, uy, uz, theta_x, theta_y, theta_z]
    因此 point_load_6 直接对应:
        [Fx, Fy, Fz, Mx, My, Mz]

    输入:
        F_row         : shape = (n_dofs_full,)
        point_load_6  : shape = (6,)
        eta           : 当前点载荷作用位置
        free_node_eta : shape = (n_free_nodes,)
    """
    if F_row.ndim != 1:
        raise ValueError("F_row 必须是一维数组。")

    point_load_6 = np.asarray(point_load_6, dtype=np.float64).reshape(-1)
    if point_load_6.size != 6:
        raise ValueError(f"point_load_6 长度必须为 6，当前为 {point_load_6.size}")

    free_node_eta = np.asarray(free_node_eta, dtype=np.float64).reshape(-1)
    n_free_nodes = free_node_eta.size
    if n_free_nodes < 1:
        raise ValueError("free_node_eta 至少需要包含一个自由节点。")

    expected_dofs = n_free_nodes * 6
    if F_row.size != expected_dofs:
        raise ValueError(
            f"F_row 长度与 free_node_eta 不匹配: "
            f"期望 {expected_dofs}, 实际 {F_row.size}"
        )

    if eta <= free_node_eta[0]:
        idx0 = idx1 = 0
        w0, w1 = 1.0, 0.0
    elif eta >= free_node_eta[-1]:
        idx0 = idx1 = n_free_nodes - 1
        w0, w1 = 1.0, 0.0
    else:
        idx1 = int(np.searchsorted(free_node_eta, eta, side="right"))
        idx0 = idx1 - 1

        e0 = free_node_eta[idx0]
        e1 = free_node_eta[idx1]

        if abs(e1 - e0) < 1e-14:
            w0, w1 = 1.0, 0.0
        else:
            w1 = (eta - e0) / (e1 - e0)
            w0 = 1.0 - w1

    # 节点 idx 的 6 个自由度切片
    s0 = idx0 * 6
    F_row[s0:s0 + 6] += w0 * point_load_6

    if idx1 != idx0:
        s1 = idx1 * 6
        F_row[s1:s1 + 6] += w1 * point_load_6


def build_student_force_history_from_multits(
    time_array: np.ndarray,
    n_dofs_full: int,
    free_node_eta: np.ndarray,
    load_file: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    根据 teacher / student 共用的多节点时历载荷文件，构造 student 使用的 F_time。

    输入:
        time_array     : student 仿真时间网格, shape = (n_steps,)
        n_dofs_full    : student 自由系统总 DOF 数
        free_node_eta  : student 自由节点 eta, shape = (n_free_nodes,)
        load_file      : 与 teacher 共用的 .dat 文件路径

    返回:
        F_time         : shape = (n_steps, n_dofs_full)
        eta_points     : shape = (n_points,)

    说明:
    - 每个仿真时刻 t，都会按 teacher 相同规则插值得到当前各加载点载荷
    - 然后把各点载荷按 eta 分配到 student 自由节点对应的 DOF 上
    """
    time_array = np.asarray(time_array, dtype=np.float64).reshape(-1)
    if time_array.size < 1:
        raise ValueError("time_array 不能为空。")

    free_node_eta = np.asarray(free_node_eta, dtype=np.float64).reshape(-1)
    if free_node_eta.size < 1:
        raise ValueError("free_node_eta 不能为空。")

    expected_dofs = free_node_eta.size * 6
    if n_dofs_full != expected_dofs:
        raise ValueError(
            f"n_dofs_full 与 free_node_eta 不匹配: "
            f"期望 {expected_dofs}, 实际 {n_dofs_full}"
        )

    load_time, eta_points, load_series = read_multi_timeseries_load_file(load_file)

    F_time = np.zeros((time_array.size, n_dofs_full), dtype=np.float64)

    for it, t in enumerate(time_array):
        cur_loads = interp_multi_timeseries_load(
            t=float(t),
            load_time=load_time,
            load_series=load_series,
        )  # shape = (n_points, 6)

        row = np.zeros(n_dofs_full, dtype=np.float64)

        for ip, eta in enumerate(eta_points):
            distribute_point_load_by_eta(
                F_row=row,
                point_load_6=cur_loads[ip],
                eta=float(eta),
                free_node_eta=free_node_eta,
            )

        F_time[it, :] = row

    return F_time, eta_points


def build_student_force_history_from_case(
    case: Dict[str, Any],
    student_model,
    n_dofs_full: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从 teacher 风格的 case 字典直接构造 student 的 time_array 与 F_time。

    case 预期字段:
        - t_initial
        - t_final
        - dt
        - use_time_series_load
        - time_series_load_file (当 use_time_series_load=True 时必须提供)

    返回:
        time_array  : shape = (n_steps,)
        F_time      : shape = (n_steps, n_dofs_full)
        eta_points  : shape = (n_points,)；若未使用时历载荷，则返回空数组
    """
    if "t_initial" not in case:
        raise KeyError("case 缺少 t_initial")
    if "t_final" not in case:
        raise KeyError("case 缺少 t_final")
    if "dt" not in case:
        raise KeyError("case 缺少 dt")

    t_initial = float(case["t_initial"])
    t_final = float(case["t_final"])
    dt = float(case["dt"])

    if dt <= 0.0:
        raise ValueError("dt 必须大于 0。")
    if t_final < t_initial:
        raise ValueError("t_final 必须大于或等于 t_initial。")

    n_steps = int(round((t_final - t_initial) / dt)) + 1
    time_array = t_initial + np.arange(n_steps, dtype=np.float64) * dt

    free_node_eta = get_student_free_node_eta(student_model)

    if n_dofs_full != free_node_eta.size * 6:
        raise ValueError(
            f"n_dofs_full 与 student 自由节点数量不一致: "
            f"n_dofs_full={n_dofs_full}, n_free_nodes={free_node_eta.size}"
        )

    use_ts = bool(case.get("use_time_series_load", False))
    if not use_ts:
        F_time = np.zeros((n_steps, n_dofs_full), dtype=np.float64)
        return time_array, F_time, np.array([], dtype=np.float64)

    ts_file = case.get("time_series_load_file", None)
    if ts_file is None:
        raise ValueError("use_time_series_load=True 时必须提供 time_series_load_file。")

    F_time, eta_points = build_student_force_history_from_multits(
        time_array=time_array,
        n_dofs_full=n_dofs_full,
        free_node_eta=free_node_eta,
        load_file=ts_file,
    )

    return time_array, F_time, eta_points