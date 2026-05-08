from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from scripts.run_student_cases import run_student_case  # noqa: E402
from src.teacher.beamdyn_teacher_adapter import (  # noqa: E402
    BeamDynTeacherCaseConfig,
    run_teacher_case_beamdyn,
    load_teacher_6dof_response,
    resample_response_to_time_grid,
)


# ============================================================
# 你以后主要只需要改这里
# ============================================================

TRAIN_LOAD_FILES = [
    str(PROJECT_ROOT / "data" / "load" / "test_complex_case.dat"),
]

VALID_LOAD_FILES = [
    # 以后有验证工况时在这里添加，例如：
    # str(PROJECT_ROOT / "data" / "load" / "valid_complex_case.dat"),
]


# ============================================================
# Config
# ============================================================

@dataclass
class PhysicalTrainingCasePrepConfig:
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

    output_dir: str = str(PROJECT_ROOT / "results" / "transformer" / "physical_training_cases_base_phi_static_kappa_y")
    case_prefix: str = "base_phi_training"

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

    # 当前新 base student 的定义
    use_base_initial_twist_phi: bool = True
    base_phi_twist_column: str = "initial_twist_deg"
    base_phi_sign: float = -1.0
    rotate_mass: bool = False

    remove_initial_offset: bool = True

    rebuild_cases: bool = False
    save_summary: bool = True


# ============================================================
# Utility
# ============================================================

def _assert_existing_file(path: str | Path, label: str) -> Path:
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {p}")
    return p


def _safe_case_stem(load_file: str | Path) -> str:
    stem = Path(load_file).stem
    safe = []
    for ch in stem:
        if ch.isalnum() or ch in ["_", "-", "."]:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def _parse_file_list(value: Optional[str], fallback: List[str]) -> List[str]:
    """
    支持两种输入：
        1. None：使用脚本顶部默认列表
        2. "a.dat,b.dat,c.dat"：命令行传入多个文件
    """
    if value is None:
        return list(fallback)

    value = value.strip()
    if len(value) == 0:
        return []

    return [item.strip() for item in value.split(",") if item.strip()]


def _remove_initial_offset(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    return u - u[:1, :]


def _component_indices(n_nodes: int, component: str) -> np.ndarray:
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
    off = component_to_offset[component]
    return np.asarray([i * 6 + off for i in range(n_nodes)], dtype=np.int64)


def _tip_component_index(n_nodes: int, component: str) -> int:
    return int(_component_indices(n_nodes, component)[-1])


def _last_k_component_indices(n_nodes: int, component: str, last_k: int = 5) -> np.ndarray:
    idx = _component_indices(n_nodes, component)
    return idx[-int(last_k):]


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.mean(diff ** 2))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.mean(np.abs(diff)))


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(_mse(a, b)))


def compute_baseline_metrics(
    *,
    u_base: np.ndarray,
    u_teacher: np.ndarray,
    last_k: int = 5,
) -> Dict[str, float]:
    """
    输入必须已经在同一时间网格、同一 DOF 顺序下。

    DOF layout:
        node 1: ux, uy, uz, rx, ry, rz
        node 2: ux, uy, uz, rx, ry, rz
        ...
    """
    u_base = np.asarray(u_base, dtype=np.float64)
    u_teacher = np.asarray(u_teacher, dtype=np.float64)

    if u_base.shape != u_teacher.shape:
        raise ValueError(
            f"u_base/u_teacher shape mismatch: "
            f"{u_base.shape} vs {u_teacher.shape}"
        )
    if u_base.ndim != 2:
        raise ValueError(f"Expected 2D response arrays, got {u_base.ndim}D")

    n_dofs = u_base.shape[1]
    if n_dofs % 6 != 0:
        raise ValueError(f"Expected n_dofs divisible by 6, got {n_dofs}")

    n_nodes = n_dofs // 6

    x_idx = _component_indices(n_nodes, "x")
    y_idx = _component_indices(n_nodes, "y")
    z_idx = _component_indices(n_nodes, "z")

    tip_x = _tip_component_index(n_nodes, "x")
    tip_y = _tip_component_index(n_nodes, "y")
    tip_z = _tip_component_index(n_nodes, "z")

    last_y_idx = _last_k_component_indices(n_nodes, "y", last_k=last_k)

    metrics = {
        "all_mse": _mse(u_base, u_teacher),
        "all_mae": _mae(u_base, u_teacher),
        "all_rmse": _rmse(u_base, u_teacher),

        "full_x_mse": _mse(u_base[:, x_idx], u_teacher[:, x_idx]),
        "full_x_mae": _mae(u_base[:, x_idx], u_teacher[:, x_idx]),
        "full_x_rmse": _rmse(u_base[:, x_idx], u_teacher[:, x_idx]),

        "full_y_mse": _mse(u_base[:, y_idx], u_teacher[:, y_idx]),
        "full_y_mae": _mae(u_base[:, y_idx], u_teacher[:, y_idx]),
        "full_y_rmse": _rmse(u_base[:, y_idx], u_teacher[:, y_idx]),

        "full_z_mse": _mse(u_base[:, z_idx], u_teacher[:, z_idx]),
        "full_z_mae": _mae(u_base[:, z_idx], u_teacher[:, z_idx]),
        "full_z_rmse": _rmse(u_base[:, z_idx], u_teacher[:, z_idx]),

        "tip_x_mse": _mse(u_base[:, tip_x], u_teacher[:, tip_x]),
        "tip_x_mae": _mae(u_base[:, tip_x], u_teacher[:, tip_x]),
        "tip_x_rmse": _rmse(u_base[:, tip_x], u_teacher[:, tip_x]),

        "tip_y_mse": _mse(u_base[:, tip_y], u_teacher[:, tip_y]),
        "tip_y_mae": _mae(u_base[:, tip_y], u_teacher[:, tip_y]),
        "tip_y_rmse": _rmse(u_base[:, tip_y], u_teacher[:, tip_y]),

        "tip_z_mse": _mse(u_base[:, tip_z], u_teacher[:, tip_z]),
        "tip_z_mae": _mae(u_base[:, tip_z], u_teacher[:, tip_z]),
        "tip_z_rmse": _rmse(u_base[:, tip_z], u_teacher[:, tip_z]),

        "last5_y_mse": _mse(u_base[:, last_y_idx], u_teacher[:, last_y_idx]),
        "last5_y_mae": _mae(u_base[:, last_y_idx], u_teacher[:, last_y_idx]),
        "last5_y_rmse": _rmse(u_base[:, last_y_idx], u_teacher[:, last_y_idx]),
    }

    return metrics


def _save_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(payload, f, sort_keys=False, allow_unicode=True)


# ============================================================
# Core prepare logic
# ============================================================

def prepare_one_physical_training_case(
    *,
    cfg: PhysicalTrainingCasePrepConfig,
    load_file: str | Path,
    split: str,
    rebuild: Optional[bool] = None,
) -> Path:
    """
    准备一个 physical training case。

    输出 .npz 中的核心字段：
        time
        F_raw
        u_teacher
        u_base
        v_base
        a_base
        natural_freqs_hz
        base_phi_element_deg
        baseline metrics

    注意：
        u_teacher 已经被重采样到 student 时间网格。
        u_base / v_base / a_base 来自当前新 base student。
    """
    teacher_exe = _assert_existing_file(cfg.teacher_exe, "teacher_exe")
    template_inp = _assert_existing_file(cfg.template_inp, "template_inp")
    blade_csv = _assert_existing_file(cfg.blade_csv, "blade_csv")
    load_file = _assert_existing_file(load_file, "load_file")

    output_dir = Path(cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if rebuild is None:
        rebuild = bool(cfg.rebuild_cases)

    case_stem = _safe_case_stem(load_file)
    case_name = f"{cfg.case_prefix}_{split}_{case_stem}"

    case_dir = output_dir / split / case_stem
    case_dir.mkdir(parents=True, exist_ok=True)

    npz_path = case_dir / f"{case_stem}_phi_base_training_case.npz"
    yaml_path = case_dir / f"{case_stem}_phi_base_training_case_summary.yaml"

    if npz_path.exists() and not rebuild:
        print()
        print(f"[Skip Existing Case] {case_name}")
        print(f"  split   = {split}")
        print(f"  load    = {load_file}")
        print(f"  npz     = {npz_path}")
        print("  reason  = cache exists and rebuild_cases=False")
        return npz_path

    print()
    print("=" * 100)
    print(f"[Prepare Physical Training Case] {case_name}")
    print("=" * 100)
    print(f"  split     = {split}")
    print(f"  load_file = {load_file}")
    print(f"  out_npz   = {npz_path}")

    # ------------------------------------------------------------
    # 1. Run BeamDyn teacher
    # ------------------------------------------------------------
    print()
    print("[1/5] Running BeamDyn teacher")

    teacher_cfg = BeamDynTeacherCaseConfig(
        case_name=case_name,
        teacher_exe=teacher_exe,
        template_inp=template_inp,
        output_dir=case_dir / "teacher",
        use_time_series_load=True,
        t_initial=float(cfg.t_initial),
        t_final=float(cfg.t_final),
        dt=float(cfg.dt),
        time_series_load_file=load_file,
        keep_temp_inp=False,
    )

    teacher_result = run_teacher_case_beamdyn(teacher_cfg)
    teacher_out = teacher_result.get("out", None)
    if teacher_out is None:
        raise RuntimeError("BeamDyn teacher did not return .out path.")

    teacher_out = Path(teacher_out).resolve()
    print(f"  teacher_out = {teacher_out}")

    # ------------------------------------------------------------
    # 2. Load teacher response
    # ------------------------------------------------------------
    print()
    print("[2/5] Loading teacher response")

    time_teacher, u_teacher_raw, teacher_columns = load_teacher_6dof_response(
        out_path=teacher_out,
        node_start=int(cfg.teacher_node_start),
        node_end=int(cfg.teacher_node_end),
        demean=bool(cfg.teacher_demean),
    )

    time_teacher = np.asarray(time_teacher, dtype=np.float64)
    u_teacher_raw = np.asarray(u_teacher_raw, dtype=np.float64)

    print(f"  time_teacher shape = {time_teacher.shape}")
    print(f"  u_teacher shape    = {u_teacher_raw.shape}")

    # ------------------------------------------------------------
    # 3. Run current base student
    # ------------------------------------------------------------
    print()
    print("[3/5] Running current base student")

    student_result = run_student_case(
        blade_csv=blade_csv,
        output_dir=case_dir / "student_base",
        case_name=case_name,
        use_time_series_load=True,
        t_initial=float(cfg.t_initial),
        t_final=float(cfg.t_final),
        dt=float(cfg.dt),
        time_series_load_file=load_file,
        student_params={
            "alpha_flap": float(cfg.alpha_flap),
            "alpha_edge": float(cfg.alpha_edge),
            "alpha_torsion": float(cfg.alpha_torsion),
            "zeta_structural": float(cfg.zeta_structural),
            "ref_freq_hz": cfg.ref_freq_hz,

            # 新 base student
            "use_base_initial_twist_phi": bool(cfg.use_base_initial_twist_phi),
            "base_phi_twist_column": str(cfg.base_phi_twist_column),
            "base_phi_sign": float(cfg.base_phi_sign),
            "rotate_mass": bool(cfg.rotate_mass),
        },
        u0_trans=None,
        v0_trans=None,
    )

    time_student = np.asarray(student_result["time"], dtype=np.float64)
    F_raw = np.asarray(student_result["F_time"], dtype=np.float64)

    u_base_raw = np.asarray(student_result["u_full"], dtype=np.float64)
    v_base = np.asarray(student_result["v_full"], dtype=np.float64)
    a_base = np.asarray(student_result["a_full"], dtype=np.float64)

    natural_freqs_hz = np.asarray(student_result["natural_freqs_hz"], dtype=np.float64)

    base_phi_element_deg = student_result.get("base_phi_element_deg", None)
    if base_phi_element_deg is None:
        base_phi_element_deg = np.asarray([], dtype=np.float64)
    else:
        base_phi_element_deg = np.asarray(base_phi_element_deg, dtype=np.float64)

    print(f"  time_student shape = {time_student.shape}")
    print(f"  F_raw shape        = {F_raw.shape}")
    print(f"  u_base shape       = {u_base_raw.shape}")
    print(f"  v_base shape       = {v_base.shape}")
    print(f"  a_base shape       = {a_base.shape}")

    # ------------------------------------------------------------
    # 4. Align teacher to student time grid
    # ------------------------------------------------------------
    print()
    print("[4/5] Resampling teacher response to student time grid")

    u_teacher_on_student_raw = resample_response_to_time_grid(
        time_src=time_teacher,
        u_src=u_teacher_raw,
        time_dst=time_student,
    )

    u_teacher_on_student_raw = np.asarray(u_teacher_on_student_raw, dtype=np.float64)

    print(f"  u_teacher_on_student_raw shape = {u_teacher_on_student_raw.shape}")

    if u_teacher_on_student_raw.shape != u_base_raw.shape:
        raise ValueError(
            "Aligned teacher and base student response shape mismatch: "
            f"teacher={u_teacher_on_student_raw.shape}, base={u_base_raw.shape}"
        )

    if bool(cfg.remove_initial_offset):
        u_teacher = _remove_initial_offset(u_teacher_on_student_raw)
        u_base = _remove_initial_offset(u_base_raw)
    else:
        u_teacher = u_teacher_on_student_raw.copy()
        u_base = u_base_raw.copy()

    # ------------------------------------------------------------
    # 5. Metrics and save
    # ------------------------------------------------------------
    print()
    print("[5/5] Computing baseline metrics and saving cache")

    baseline_metrics = compute_baseline_metrics(
        u_base=u_base,
        u_teacher=u_teacher,
        last_k=5,
    )

    print()
    print("[Baseline Metrics: base student vs teacher]")
    for key in [
        "all_mae",
        "all_rmse",
        "full_x_mae",
        "full_y_mae",
        "tip_x_mae",
        "tip_y_mae",
        "last5_y_mae",
    ]:
        print(f"  {key:<16}: {baseline_metrics[key]:.12e}")

    np.savez(
        npz_path,
        # common grid for training
        time=time_student,
        F_raw=F_raw,

        # aligned responses for training
        u_teacher=u_teacher,
        u_base=u_base,
        v_base=v_base,
        a_base=a_base,

        # raw / provenance responses
        time_teacher_raw=time_teacher,
        u_teacher_raw=u_teacher_raw,
        u_teacher_on_student_raw=u_teacher_on_student_raw,
        u_base_raw=u_base_raw,

        # model / base info
        natural_freqs_hz=natural_freqs_hz,
        base_phi_element_deg=base_phi_element_deg,

        # case metadata as arrays
        t_initial=np.asarray([cfg.t_initial], dtype=np.float64),
        t_final=np.asarray([cfg.t_final], dtype=np.float64),
        dt=np.asarray([cfg.dt], dtype=np.float64),
        remove_initial_offset=np.asarray([1 if cfg.remove_initial_offset else 0], dtype=np.int64),
        use_base_initial_twist_phi=np.asarray([1 if cfg.use_base_initial_twist_phi else 0], dtype=np.int64),
        base_phi_sign=np.asarray([cfg.base_phi_sign], dtype=np.float64),
        rotate_mass=np.asarray([1 if cfg.rotate_mass else 0], dtype=np.int64),

        # baseline metric scalars
        **{k: np.asarray([v], dtype=np.float64) for k, v in baseline_metrics.items()},
    )

    summary = {
        "case_name": case_name,
        "split": split,
        "load_file": str(load_file),
        "teacher_out": str(teacher_out),
        "npz_path": str(npz_path),
        "yaml_path": str(yaml_path),
        "config": asdict(cfg),
        "shapes": {
            "time_teacher": list(time_teacher.shape),
            "u_teacher_raw": list(u_teacher_raw.shape),
            "time_student": list(time_student.shape),
            "F_raw": list(F_raw.shape),
            "u_teacher": list(u_teacher.shape),
            "u_base": list(u_base.shape),
            "v_base": list(v_base.shape),
            "a_base": list(a_base.shape),
            "base_phi_element_deg": list(base_phi_element_deg.shape),
        },
        "base_student": {
            "use_base_initial_twist_phi": bool(cfg.use_base_initial_twist_phi),
            "base_phi_twist_column": str(cfg.base_phi_twist_column),
            "base_phi_sign": float(cfg.base_phi_sign),
            "rotate_mass": bool(cfg.rotate_mass),
            "natural_freqs_hz": natural_freqs_hz.tolist(),
            "student_params_used": student_result.get("student_params_used", {}),
            "base_model_info": student_result.get("base_model_info", {}),
        },
        "baseline_metrics": baseline_metrics,
        "teacher_columns_first_6": list(teacher_columns[:6]) if teacher_columns is not None else None,
        "teacher_columns_last_6": list(teacher_columns[-6:]) if teacher_columns is not None else None,
    }

    if bool(cfg.save_summary):
        _save_yaml(yaml_path, summary)

    print()
    print("[Saved]")
    print(f"  npz  = {npz_path}")
    if bool(cfg.save_summary):
        print(f"  yaml = {yaml_path}")

    return npz_path


def prepare_physical_training_cases(
    *,
    cfg: PhysicalTrainingCasePrepConfig,
    train_load_files: List[str | Path],
    valid_load_files: List[str | Path],
    rebuild: Optional[bool] = None,
) -> Dict[str, List[Path]]:
    """
    批量准备 train / valid cases。

    后续训练脚本可以直接 import 这个函数。
    """
    prepared: Dict[str, List[Path]] = {
        "train": [],
        "valid": [],
    }

    for load_file in train_load_files:
        p = prepare_one_physical_training_case(
            cfg=cfg,
            load_file=load_file,
            split="train",
            rebuild=rebuild,
        )
        prepared["train"].append(p)

    for load_file in valid_load_files:
        p = prepare_one_physical_training_case(
            cfg=cfg,
            load_file=load_file,
            split="valid",
            rebuild=rebuild,
        )
        prepared["valid"].append(p)

    return prepared


# ============================================================
# CLI
# ============================================================

def parse_args() -> tuple[PhysicalTrainingCasePrepConfig, List[str], List[str]]:
    d = PhysicalTrainingCasePrepConfig()

    parser = argparse.ArgumentParser(
        description=(
            "Prepare physical training cases with fixed base Phi(s) = -initial_twist_deg(s). "
            "This script automatically runs BeamDyn teacher and current base student."
        )
    )

    parser.add_argument("--teacher-exe", type=str, default=d.teacher_exe)
    parser.add_argument("--template-inp", type=str, default=d.template_inp)
    parser.add_argument("--blade-csv", type=str, default=d.blade_csv)

    parser.add_argument("--output-dir", type=str, default=d.output_dir)
    parser.add_argument("--case-prefix", type=str, default=d.case_prefix)

    parser.add_argument("--train-load-files", type=str, default=None)
    parser.add_argument("--valid-load-files", type=str, default=None)

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

    base_phi_group = parser.add_mutually_exclusive_group()
    base_phi_group.add_argument(
        "--use-base-initial-twist-phi",
        dest="use_base_initial_twist_phi",
        action="store_true",
    )
    base_phi_group.add_argument(
        "--no-base-initial-twist-phi",
        dest="use_base_initial_twist_phi",
        action="store_false",
    )
    parser.set_defaults(use_base_initial_twist_phi=d.use_base_initial_twist_phi)

    parser.add_argument("--base-phi-twist-column", type=str, default=d.base_phi_twist_column)
    parser.add_argument("--base-phi-sign", type=float, default=d.base_phi_sign)

    mass_group = parser.add_mutually_exclusive_group()
    mass_group.add_argument("--rotate-mass", dest="rotate_mass", action="store_true")
    mass_group.add_argument("--no-rotate-mass", dest="rotate_mass", action="store_false")
    parser.set_defaults(rotate_mass=d.rotate_mass)

    offset_group = parser.add_mutually_exclusive_group()
    offset_group.add_argument("--remove-initial-offset", dest="remove_initial_offset", action="store_true")
    offset_group.add_argument("--keep-initial-offset", dest="remove_initial_offset", action="store_false")
    parser.set_defaults(remove_initial_offset=d.remove_initial_offset)

    rebuild_group = parser.add_mutually_exclusive_group()
    rebuild_group.add_argument("--rebuild-cases", dest="rebuild_cases", action="store_true")
    rebuild_group.add_argument("--no-rebuild-cases", dest="rebuild_cases", action="store_false")
    parser.set_defaults(rebuild_cases=d.rebuild_cases)

    summary_group = parser.add_mutually_exclusive_group()
    summary_group.add_argument("--save-summary", dest="save_summary", action="store_true")
    summary_group.add_argument("--no-save-summary", dest="save_summary", action="store_false")
    parser.set_defaults(save_summary=d.save_summary)

    args = parser.parse_args()

    cfg = PhysicalTrainingCasePrepConfig(
        teacher_exe=args.teacher_exe,
        template_inp=args.template_inp,
        blade_csv=args.blade_csv,
        output_dir=args.output_dir,
        case_prefix=args.case_prefix,
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
        use_base_initial_twist_phi=args.use_base_initial_twist_phi,
        base_phi_twist_column=args.base_phi_twist_column,
        base_phi_sign=args.base_phi_sign,
        rotate_mass=args.rotate_mass,
        remove_initial_offset=args.remove_initial_offset,
        rebuild_cases=args.rebuild_cases,
        save_summary=args.save_summary,
    )

    train_load_files = _parse_file_list(args.train_load_files, TRAIN_LOAD_FILES)
    valid_load_files = _parse_file_list(args.valid_load_files, VALID_LOAD_FILES)

    return cfg, train_load_files, valid_load_files


def main() -> None:
    cfg, train_load_files, valid_load_files = parse_args()

    print()
    print("[Prepare Physical Training Cases: Base Phi]")
    print()
    print("[Config]")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")

    print()
    print("[Load Files]")
    print("  train:")
    for f in train_load_files:
        print(f"    - {f}")
    print("  valid:")
    if len(valid_load_files) == 0:
        print("    - <empty>")
    else:
        for f in valid_load_files:
            print(f"    - {f}")

    prepared = prepare_physical_training_cases(
        cfg=cfg,
        train_load_files=train_load_files,
        valid_load_files=valid_load_files,
        rebuild=cfg.rebuild_cases,
    )

    print()
    print("=" * 100)
    print("[Done]")
    print("=" * 100)
    print("train cases:")
    for p in prepared["train"]:
        print(f"  - {p}")
    print("valid cases:")
    if len(prepared["valid"]) == 0:
        print("  - <empty>")
    else:
        for p in prepared["valid"]:
            print(f"  - {p}")

    print()
    print("✅ PASS: physical training case preparation completed.")


if __name__ == "__main__":
    main()