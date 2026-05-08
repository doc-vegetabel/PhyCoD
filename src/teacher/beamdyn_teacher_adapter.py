from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from scripts.run_new_beamdyn_cases import run_beamdyn_case
from src.teacher.sixdof_parser import (
    load_teacher_fullfield_6dof_from_out,
    get_teacher_6dof_column_names,
)


@dataclass(frozen=True)
class BeamDynTeacherCaseConfig:
    """
    BeamDyn teacher case 配置。

    这个 dataclass 是新的 teacher 调用入口配置，
    不依赖 pipeline_hybrid_transformer.py。
    """

    case_name: str
    teacher_exe: str | Path
    template_inp: str | Path
    output_dir: str | Path

    use_time_series_load: bool
    t_initial: float
    t_final: float
    dt: float
    time_series_load_file: Optional[str | Path] = None

    keep_temp_inp: bool = False

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        for k in ["teacher_exe", "template_inp", "output_dir", "time_series_load_file"]:
            if out.get(k, None) is not None:
                out[k] = str(Path(out[k]).resolve())
        return out


def run_teacher_case_beamdyn(
    cfg: BeamDynTeacherCaseConfig,
) -> Dict[str, Any]:
    """
    运行一个 BeamDyn teacher case。

    返回：
        {
            "case_name": ...,
            "out": Path(...),
            "ech": Path(...) or None,
            "stdout": Path(...),
            "stderr": Path(...),
        }

    注意：
    - 只调用 scripts.run_new_beamdyn_cases.run_beamdyn_case；
    - 不依赖旧 pipeline_hybrid_transformer.py。
    """
    result = run_beamdyn_case(
        exe_path=cfg.teacher_exe,
        template_inp=cfg.template_inp,
        output_dir=cfg.output_dir,
        case_name=cfg.case_name,
        use_time_series_load=cfg.use_time_series_load,
        t_initial=cfg.t_initial,
        t_final=cfg.t_final,
        dt=cfg.dt,
        time_series_load_file=cfg.time_series_load_file,
        keep_temp_inp=cfg.keep_temp_inp,
    )

    return {
        "case_name": str(result["case_name"]),
        "out": Path(result["out"]).resolve() if result.get("out") is not None else None,
        "ech": Path(result["ech"]).resolve() if result.get("ech") is not None else None,
        "stdout": Path(result["stdout"]).resolve() if result.get("stdout") is not None else None,
        "stderr": Path(result["stderr"]).resolve() if result.get("stderr") is not None else None,
    }


def load_teacher_6dof_response(
    out_path: str | Path,
    *,
    node_start: int = 2,
    node_end: int = 49,
    demean: bool = False,
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """
    从 BeamDyn .out 读取 teacher full-order 6DOF 响应。

    输出：
        time_teacher:
            shape = (T,)

        u_teacher_6dof:
            shape = (T, n_nodes * 6)

        column_names:
            长度 = n_nodes * 6

    每个节点 DOF 顺序：
        [ux, uy, uz, rx, ry, rz]
    """
    time_teacher, u_teacher_6dof = load_teacher_fullfield_6dof_from_out(
        out_path=out_path,
        node_start=node_start,
        node_end=node_end,
        demean=demean,
        return_time=True,
    )

    time_teacher = np.asarray(time_teacher, dtype=np.float64).reshape(-1)
    u_teacher_6dof = np.asarray(u_teacher_6dof, dtype=np.float64)

    if u_teacher_6dof.ndim != 2:
        raise ValueError(
            f"u_teacher_6dof must be 2D, got shape={u_teacher_6dof.shape}."
        )
    if u_teacher_6dof.shape[0] != time_teacher.size:
        raise ValueError(
            "Teacher response time dimension mismatch: "
            f"time={time_teacher.size}, response={u_teacher_6dof.shape[0]}"
        )

    column_names = get_teacher_6dof_column_names(
        node_start=node_start,
        node_end=node_end,
    )

    if len(column_names) != u_teacher_6dof.shape[1]:
        raise ValueError(
            f"column_names length mismatch: {len(column_names)} vs "
            f"{u_teacher_6dof.shape[1]}"
        )

    return time_teacher, u_teacher_6dof, column_names


def resample_response_to_time_grid(
    time_src: np.ndarray,
    u_src: np.ndarray,
    time_dst: np.ndarray,
) -> np.ndarray:
    """
    将二维响应 u_src: (T_src, D) 插值到 time_dst。

    该函数复制必要的线性插值逻辑，避免依赖旧 pipeline_hybrid_transformer.py。
    """
    time_src = np.asarray(time_src, dtype=np.float64).reshape(-1)
    time_dst = np.asarray(time_dst, dtype=np.float64).reshape(-1)
    u_src = np.asarray(u_src, dtype=np.float64)

    if time_src.ndim != 1:
        raise ValueError("time_src must be 1D.")
    if time_dst.ndim != 1:
        raise ValueError("time_dst must be 1D.")
    if u_src.ndim != 2:
        raise ValueError(f"u_src must be 2D, got shape={u_src.shape}.")
    if u_src.shape[0] != time_src.size:
        raise ValueError(
            f"u_src time dimension mismatch: {u_src.shape[0]} vs {time_src.size}."
        )

    if time_src.size == time_dst.size and np.allclose(time_src, time_dst):
        return u_src.copy()

    u_dst = np.zeros((time_dst.size, u_src.shape[1]), dtype=np.float64)

    for j in range(u_src.shape[1]):
        u_dst[:, j] = np.interp(
            time_dst,
            time_src,
            u_src[:, j],
            left=u_src[0, j],
            right=u_src[-1, j],
        )

    return u_dst


def get_full_state_component(
    u_full: np.ndarray,
    component: str,
) -> np.ndarray:
    """
    从 full 6DOF state 中抽取某个分量场。

    输入：
        u_full: shape = (..., n_nodes * 6)

    component:
        x, y, z, rx, ry, rz

    输出：
        shape = (..., n_nodes)
    """
    comp_to_idx = {
        "x": 0,
        "y": 1,
        "z": 2,
        "rx": 3,
        "ry": 4,
        "rz": 5,
    }

    component = component.lower().strip()
    if component not in comp_to_idx:
        raise ValueError(f"Unsupported component={component!r}.")

    u_full = np.asarray(u_full, dtype=np.float64)

    if u_full.shape[-1] % 6 != 0:
        raise ValueError(
            f"Last dimension must be divisible by 6, got {u_full.shape[-1]}."
        )

    return u_full[..., comp_to_idx[component] :: 6]


def get_tip_component(
    u_full: np.ndarray,
    component: str,
) -> np.ndarray:
    field = get_full_state_component(u_full, component=component)
    return field[..., -1]


def get_last_k_component(
    u_full: np.ndarray,
    component: str,
    *,
    k: int = 5,
) -> np.ndarray:
    field = get_full_state_component(u_full, component=component)
    if field.shape[-1] <= k:
        return field
    return field[..., -k:]


def _mae_rmse_max_abs(
    pred: np.ndarray,
    target: np.ndarray,
) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}")

    diff = pred - target
    abs_diff = np.abs(diff)

    return {
        "mae": float(np.mean(abs_diff)),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "max_abs": float(np.max(abs_diff)),
    }


def compute_student_teacher_6dof_metrics(
    *,
    u_student: np.ndarray,
    u_teacher: np.ndarray,
) -> Dict[str, float]:
    """
    计算 student full-order 6DOF 与 teacher full-order 6DOF 的误差指标。

    重点输出：
    - all_mae / all_rmse
    - full_x_mae / full_y_mae
    - tip_x_mae / tip_y_mae
    - last5_y_mae
    """
    u_student = np.asarray(u_student, dtype=np.float64)
    u_teacher = np.asarray(u_teacher, dtype=np.float64)

    if u_student.shape != u_teacher.shape:
        raise ValueError(
            f"student/teacher shape mismatch: {u_student.shape} vs {u_teacher.shape}"
        )

    all_stats = _mae_rmse_max_abs(u_student, u_teacher)

    full_x = _mae_rmse_max_abs(
        get_full_state_component(u_student, "x"),
        get_full_state_component(u_teacher, "x"),
    )
    full_y = _mae_rmse_max_abs(
        get_full_state_component(u_student, "y"),
        get_full_state_component(u_teacher, "y"),
    )
    full_z = _mae_rmse_max_abs(
        get_full_state_component(u_student, "z"),
        get_full_state_component(u_teacher, "z"),
    )

    tip_x = _mae_rmse_max_abs(
        get_tip_component(u_student, "x"),
        get_tip_component(u_teacher, "x"),
    )
    tip_y = _mae_rmse_max_abs(
        get_tip_component(u_student, "y"),
        get_tip_component(u_teacher, "y"),
    )
    tip_z = _mae_rmse_max_abs(
        get_tip_component(u_student, "z"),
        get_tip_component(u_teacher, "z"),
    )

    last5_y = _mae_rmse_max_abs(
        get_last_k_component(u_student, "y", k=5),
        get_last_k_component(u_teacher, "y", k=5),
    )

    return {
        "all_mae": all_stats["mae"],
        "all_rmse": all_stats["rmse"],
        "all_max_abs": all_stats["max_abs"],

        "full_x_mae": full_x["mae"],
        "full_x_rmse": full_x["rmse"],
        "full_x_max_abs": full_x["max_abs"],

        "full_y_mae": full_y["mae"],
        "full_y_rmse": full_y["rmse"],
        "full_y_max_abs": full_y["max_abs"],

        "full_z_mae": full_z["mae"],
        "full_z_rmse": full_z["rmse"],
        "full_z_max_abs": full_z["max_abs"],

        "tip_x_mae": tip_x["mae"],
        "tip_x_rmse": tip_x["rmse"],
        "tip_x_max_abs": tip_x["max_abs"],

        "tip_y_mae": tip_y["mae"],
        "tip_y_rmse": tip_y["rmse"],
        "tip_y_max_abs": tip_y["max_abs"],

        "tip_z_mae": tip_z["mae"],
        "tip_z_rmse": tip_z["rmse"],
        "tip_z_max_abs": tip_z["max_abs"],

        "last5_y_mae": last5_y["mae"],
        "last5_y_rmse": last5_y["rmse"],
        "last5_y_max_abs": last5_y["max_abs"],
    }