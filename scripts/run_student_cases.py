from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Union, Any

import numpy as np
import yaml
from scipy.linalg import eigh

Number = Union[int, float]


def _to_str_path(p: str | Path | None) -> str | None:
    if p is None:
        return None
    return str(Path(p).resolve())


def _default_student_params() -> Dict[str, Any]:
    return {
        "alpha_flap": 1.0,
        "alpha_edge": 1.0,
        "alpha_torsion": 1.0,
        "zeta_structural": 0.015,
        "ref_freq_hz": None,

        # 新 base student 默认设置：
        # Phi_base(s) = - initial_twist_deg(s)
        "use_base_initial_twist_phi": True,
        "base_phi_twist_column": "initial_twist_deg",
        "base_phi_sign": -1.0,

        # 当前暂不学习质量/惯性，因此不旋转质量矩阵
        "rotate_mass": False,
    }


def _merge_student_params(student_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = _default_student_params()
    if student_params is not None:
        merged.update(student_params)
    return merged


def _compute_natural_frequencies_hz(
    M: np.ndarray,
    K: np.ndarray,
    *,
    num_modes: int = 10,
) -> np.ndarray:
    """
    根据 full-order M/K 计算固有频率。

    注意：
        当前 base student 已经包含固定 Phi_base(s)，
        因此 natural_freqs 也应从新的 M/K 重新计算。
    """
    eigvals, _ = eigh(K, M)
    eigvals = np.asarray(eigvals, dtype=np.float64)

    valid = eigvals > 0.0
    if not np.any(valid):
        raise ValueError("未找到正特征值，无法计算 natural frequencies。")

    freqs = np.sqrt(eigvals[valid]) / (2.0 * np.pi)
    return freqs[:num_modes]


def _build_structural_damping_matrix(
    K: np.ndarray,
    zeta_structural: float,
    ref_freq_hz: Optional[float],
    natural_freqs: np.ndarray,
) -> tuple[np.ndarray, Optional[float]]:
    """
    与 student.dynamic_solver.WindBladeDynamicSystem.solve_step_response 中相同思想：
    采用 C = beta_damp * K 的刚度比例阻尼。

    beta_damp = 2*zeta / (2*pi*f_ref)

    当前阶段：
        - zeta_structural 不学习
        - C 仍然根据新的 base K 重建
        - 这不属于阻尼学习，只是让固定阻尼与新的 base 刚度保持一致
    """
    if zeta_structural <= 0.0:
        return np.zeros_like(K), ref_freq_hz

    if ref_freq_hz is None:
        if natural_freqs.size < 1:
            raise ValueError("无法自动确定 ref_freq_hz：natural_freqs 为空。")
        ref_freq_hz = float(natural_freqs[0])

    beta_damp = 2.0 * float(zeta_structural) / (2.0 * np.pi * float(ref_freq_hz))
    C = beta_damp * K
    return C, float(ref_freq_hz)


def _build_trans_indices(n_free_nodes: int) -> np.ndarray:
    """
    平动自由度索引：[ux, uy, uz] for each free node
    """
    return np.array([i * 6 + j for i in range(n_free_nodes) for j in range(3)], dtype=int)


def _validate_case_inputs(
    case_name: str,
    use_time_series_load: bool,
    t_initial: Number,
    t_final: Number,
    dt: Number,
    time_series_load_file: Optional[str | Path],
) -> None:
    if not isinstance(case_name, str) or len(case_name.strip()) == 0:
        raise ValueError("case_name 必须是非空字符串。")

    if float(dt) <= 0.0:
        raise ValueError("dt 必须大于 0。")

    if float(t_final) < float(t_initial):
        raise ValueError("t_final 必须大于或等于 t_initial。")

    if use_time_series_load:
        if time_series_load_file is None:
            raise ValueError("use_time_series_load=True 时必须提供 time_series_load_file。")
        ts_file = Path(time_series_load_file).resolve()
        if not ts_file.exists():
            raise FileNotFoundError(f"找不到时历载荷文件: {ts_file}")


def _save_student_summary_yaml(
    save_path: Path,
    summary: Dict[str, Any],
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(summary, f, sort_keys=False, allow_unicode=True)


def _build_student_mk_and_model(
    *,
    blade_csv: Path,
    case_name: str,
    alpha_flap: float,
    alpha_edge: float,
    alpha_torsion: float,
    use_base_initial_twist_phi: bool,
    base_phi_twist_column: str,
    base_phi_sign: float,
    rotate_mass: bool,
) -> Dict[str, Any]:
    """
    统一构造 student_model, M, K。

    默认主线：
        use_base_initial_twist_phi=True

        base student = fixed Phi_base(s) student
        Phi_base(s) = - initial_twist_deg(s)

    旧 baseline 对照：
        use_base_initial_twist_phi=False

        使用 WindBladeDynamicSystem 的原始 uncoupled FEM baseline。
        这个分支只建议用于历史回归和对照，不建议作为后续训练主线。
    """
    if use_base_initial_twist_phi:
        from src.student.base_full_order_builder import build_base_student_full_order_mk

        base_result = build_base_student_full_order_mk(
            blade_csv=blade_csv,
            model_name=f"student_{case_name}",
            alpha_flap=alpha_flap,
            alpha_edge=alpha_edge,
            alpha_torsion=alpha_torsion,
            twist_column=base_phi_twist_column,
            phi_sign=base_phi_sign,
            rotate_mass=rotate_mass,
            verbose=True,
        )

        student_model = base_result["model"]
        M = np.asarray(base_result["M"], dtype=np.float64)
        K = np.asarray(base_result["K"], dtype=np.float64)

        phi_info = base_result.get("phi_info", None)
        phi_element_deg = np.asarray(base_result["phi_element_deg"], dtype=np.float64)

        fem_info = base_result.get("fem_info", {})

        base_model_info = {
            "base_type": "fixed_initial_twist_phi",
            "use_base_initial_twist_phi": True,
            "base_phi_twist_column": base_phi_twist_column,
            "base_phi_sign": float(base_phi_sign),
            "rotate_mass": bool(rotate_mass),
            "phi_summary": None if phi_info is None else phi_info.get("summary", None),
        }

        return {
            "student_model": student_model,
            "M": M,
            "K": K,
            "fem_info": fem_info,
            "phi_info": phi_info,
            "phi_element_deg": phi_element_deg,
            "base_model_info": base_model_info,
        }

    # 仅作为旧 baseline 对照保留
    from src.student.io import load_student_model_from_blade_master
    from src.student.dynamic_solver import WindBladeDynamicSystem

    student_model = load_student_model_from_blade_master(
        str(blade_csv),
        model_name=f"student_{case_name}",
    )

    blade_sys = WindBladeDynamicSystem(
        student_model,
        alpha_flap=alpha_flap,
        alpha_edge=alpha_edge,
        alpha_torsion=alpha_torsion,
    )

    M = np.asarray(blade_sys.M, dtype=np.float64)
    K = np.asarray(blade_sys.K, dtype=np.float64)

    base_model_info = {
        "base_type": "legacy_uncoupled_phi0",
        "use_base_initial_twist_phi": False,
        "base_phi_twist_column": None,
        "base_phi_sign": None,
        "rotate_mass": False,
        "phi_summary": None,
    }

    return {
        "student_model": student_model,
        "M": M,
        "K": K,
        "fem_info": {},
        "phi_info": None,
        "phi_element_deg": None,
        "base_model_info": base_model_info,
    }


def run_student_case(
    blade_csv: str | Path,
    output_dir: str | Path,
    case_name: str,
    use_time_series_load: bool,
    t_initial: Number,
    t_final: Number,
    dt: Number,
    time_series_load_file: Optional[str | Path] = None,
    student_params: Optional[Dict[str, Any]] = None,
    u0_trans: Optional[np.ndarray] = None,
    v0_trans: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    运行一个 student case，功能定位类似 teacher 侧的 run_beamdyn_case。

    当前主线定义：
        base student 已经包含固定构型角：
            Phi_base(s) = - initial_twist_deg(s)

        因此默认 direct student 不再是旧 phi=0 uncoupled baseline，
        而是 fixed initial-twist Phi coupled baseline。

    输入:
        blade_csv             : blade_master.csv 路径
        output_dir            : student 响应输出目录
        case_name             : 工况名
        use_time_series_load  : 是否使用与 teacher 共用的多节点时历载荷
        t_initial, t_final, dt: 仿真时域参数
        time_series_load_file : 多节点时历载荷文件路径
        student_params        : student 参数字典，支持字段：
                               - alpha_flap
                               - alpha_edge
                               - alpha_torsion
                               - zeta_structural
                               - ref_freq_hz

                               新增 base Phi 相关字段：
                               - use_base_initial_twist_phi
                                   默认 True
                               - base_phi_twist_column
                                   默认 "initial_twist_deg"
                               - base_phi_sign
                                   默认 -1.0
                               - rotate_mass
                                   默认 False

        u0_trans              : 初始平动位移，长度应为 n_free_nodes * 3
        v0_trans              : 初始平动速度，长度应为 n_free_nodes * 3

    输出:
        返回一个 dict，包含：
            - case_name
            - npz
            - yaml
            - time
            - u_full / v_full / a_full
            - u_trans / v_trans / a_trans
            - F_time
            - natural_freqs_hz
            - student_params_used
            - base_model_info
            - eta_points
    """
    blade_csv = Path(blade_csv).resolve()
    output_dir = Path(output_dir).resolve()
    # Ensure the output directory exists before saving response npz/yaml files.
    # This is especially important when prepare_physical_training_cases passes
    # nested directories such as <case_dir>/student_base during rebuild.
    output_dir.mkdir(parents=True, exist_ok=True)

    if not blade_csv.exists():
        raise FileNotFoundError(f"找不到 blade_csv: {blade_csv}")

    _validate_case_inputs(
        case_name=case_name,
        use_time_series_load=use_time_series_load,
        t_initial=t_initial,
        t_final=t_final,
        dt=dt,
        time_series_load_file=time_series_load_file,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # 延迟导入，避免循环或环境加载问题
    from src.student.dynamic_solver import NewmarkBetaSolver
    from src.student.load_adapter import build_student_force_history_from_case

    params = _merge_student_params(student_params)

    alpha_flap = float(params.get("alpha_flap", 1.0))
    alpha_edge = float(params.get("alpha_edge", 1.0))
    alpha_torsion = float(params.get("alpha_torsion", 1.0))
    zeta_structural = float(params.get("zeta_structural", 0.015))

    ref_freq_hz = params.get("ref_freq_hz", None)
    if ref_freq_hz is not None:
        ref_freq_hz = float(ref_freq_hz)

    use_base_initial_twist_phi = bool(params.get("use_base_initial_twist_phi", True))
    base_phi_twist_column = str(params.get("base_phi_twist_column", "initial_twist_deg"))
    base_phi_sign = float(params.get("base_phi_sign", -1.0))
    rotate_mass = bool(params.get("rotate_mass", False))

    case = {
        "case_name": case_name,
        "use_time_series_load": bool(use_time_series_load),
        "t_initial": float(t_initial),
        "t_final": float(t_final),
        "dt": float(dt),
        "time_series_load_file": _to_str_path(time_series_load_file),
    }

    print(f"\n[STUDENT RUN] {case_name}")
    print(f"  blade_csv = {blade_csv}")
    print(f"  output_dir = {output_dir}")
    print(f"  use_time_series_load = {use_time_series_load}")
    print(f"  t_initial = {t_initial}")
    print(f"  t_final   = {t_final}")
    print(f"  dt        = {dt}")
    if use_time_series_load:
        print(f"  ts_file   = {Path(time_series_load_file).resolve()}")

    print()
    print("[Student Base Definition]")
    print(f"  use_base_initial_twist_phi = {use_base_initial_twist_phi}")
    if use_base_initial_twist_phi:
        print(f"  Phi_base(s) = {base_phi_sign:+.1f} * {base_phi_twist_column}(s)")
        print(f"  rotate_mass = {rotate_mass}")
    else:
        print("  using legacy uncoupled phi=0 baseline")

    mk_result = _build_student_mk_and_model(
        blade_csv=blade_csv,
        case_name=case_name,
        alpha_flap=alpha_flap,
        alpha_edge=alpha_edge,
        alpha_torsion=alpha_torsion,
        use_base_initial_twist_phi=use_base_initial_twist_phi,
        base_phi_twist_column=base_phi_twist_column,
        base_phi_sign=base_phi_sign,
        rotate_mass=rotate_mass,
    )

    student_model = mk_result["student_model"]
    M = np.asarray(mk_result["M"], dtype=np.float64)
    K = np.asarray(mk_result["K"], dtype=np.float64)
    phi_element_deg = mk_result["phi_element_deg"]
    base_model_info = mk_result["base_model_info"]

    if M.shape != K.shape:
        raise ValueError(f"M/K shape mismatch: M={M.shape}, K={K.shape}")
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"M must be square 2D matrix, got {M.shape}")

    n_dofs_full = M.shape[0]

    natural_freqs = _compute_natural_frequencies_hz(M, K, num_modes=10)
    C, ref_freq_used = _build_structural_damping_matrix(
        K=K,
        zeta_structural=zeta_structural,
        ref_freq_hz=ref_freq_hz,
        natural_freqs=natural_freqs,
    )

    time_array, F_time, eta_points = build_student_force_history_from_case(
        case=case,
        student_model=student_model,
        n_dofs_full=n_dofs_full,
    )

    n_free_nodes = len(student_model.eta) - 1
    trans_indices = _build_trans_indices(n_free_nodes)

    U0_full = np.zeros(n_dofs_full, dtype=np.float64)
    V0_full = np.zeros(n_dofs_full, dtype=np.float64)

    if u0_trans is not None:
        u0_trans = np.asarray(u0_trans, dtype=np.float64).reshape(-1)
        if u0_trans.size != trans_indices.size:
            raise ValueError(
                f"u0_trans 长度不匹配：期望 {trans_indices.size}，实际 {u0_trans.size}"
            )
        U0_full[trans_indices] = u0_trans

    if v0_trans is not None:
        v0_trans = np.asarray(v0_trans, dtype=np.float64).reshape(-1)
        if v0_trans.size != trans_indices.size:
            raise ValueError(
                f"v0_trans 长度不匹配：期望 {trans_indices.size}，实际 {v0_trans.size}"
            )
        V0_full[trans_indices] = v0_trans

    solver = NewmarkBetaSolver(
        M=M,
        K=K,
        C=C,
        dt=float(dt),
    )

    u_full, v_full, a_full = solver.solve(
        F_time=F_time,
        u0=U0_full,
        v0=V0_full,
    )

    u_trans = u_full[:, trans_indices]
    v_trans = v_full[:, trans_indices]
    a_trans = a_full[:, trans_indices]

    npz_path = output_dir / f"{case_name}_student_response.npz"
    yaml_path = output_dir / f"{case_name}_student_summary.yaml"

    npz_payload = {
        "time": time_array,
        "u_full": u_full,
        "v_full": v_full,
        "a_full": a_full,
        "u_trans": u_trans,
        "v_trans": v_trans,
        "a_trans": a_trans,
        "F_time": F_time,
        "natural_freqs_hz": natural_freqs,
        "eta_points": eta_points,
        "alpha_flap": np.array([alpha_flap], dtype=np.float64),
        "alpha_edge": np.array([alpha_edge], dtype=np.float64),
        "alpha_torsion": np.array([alpha_torsion], dtype=np.float64),
        "zeta_structural": np.array([zeta_structural], dtype=np.float64),
        "ref_freq_hz": np.array(
            [np.nan if ref_freq_used is None else ref_freq_used],
            dtype=np.float64,
        ),
        "use_base_initial_twist_phi": np.array(
            [1 if use_base_initial_twist_phi else 0],
            dtype=np.int64,
        ),
        "base_phi_sign": np.array([base_phi_sign], dtype=np.float64),
        "rotate_mass": np.array([1 if rotate_mass else 0], dtype=np.int64),
    }

    if phi_element_deg is not None:
        npz_payload["base_phi_element_deg"] = np.asarray(phi_element_deg, dtype=np.float64)

    np.savez(
        npz_path,
        **npz_payload,
    )

    summary = {
        "case_name": case_name,
        "blade_csv": str(blade_csv),
        "output_dir": str(output_dir),
        "case": {
            "use_time_series_load": bool(use_time_series_load),
            "t_initial": float(t_initial),
            "t_final": float(t_final),
            "dt": float(dt),
            "time_series_load_file": _to_str_path(time_series_load_file),
        },
        "student_params_used": {
            "alpha_flap": alpha_flap,
            "alpha_edge": alpha_edge,
            "alpha_torsion": alpha_torsion,
            "zeta_structural": zeta_structural,
            "ref_freq_hz": ref_freq_used,
            "use_base_initial_twist_phi": use_base_initial_twist_phi,
            "base_phi_twist_column": base_phi_twist_column if use_base_initial_twist_phi else None,
            "base_phi_sign": base_phi_sign if use_base_initial_twist_phi else None,
            "rotate_mass": rotate_mass if use_base_initial_twist_phi else False,
        },
        "base_model_info": base_model_info,
        "model_info": {
            "model_name": student_model.model_name,
            "span_m": float(student_model.span_m),
            "n_stations": int(student_model.n_stations),
            "n_free_nodes": int(n_free_nodes),
            "n_dofs_full": int(n_dofs_full),
        },
        "matrix_info": {
            "M_shape": list(M.shape),
            "K_shape": list(K.shape),
            "C_shape": list(C.shape),
            "M_finite": bool(np.all(np.isfinite(M))),
            "K_finite": bool(np.all(np.isfinite(K))),
            "C_finite": bool(np.all(np.isfinite(C))),
        },
        "load_info": {
            "eta_points": eta_points.tolist(),
            "time_steps": int(time_array.size),
        },
        "natural_freqs_hz": natural_freqs.tolist(),
        "saved_files": {
            "npz": str(npz_path),
            "yaml": str(yaml_path),
        },
    }

    _save_student_summary_yaml(yaml_path, summary)

    print(f"  natural_freqs[:5] = {natural_freqs[:5]}")
    print(f"  saved npz  = {npz_path}")
    print(f"  saved yaml = {yaml_path}")

    return {
        "case_name": case_name,
        "npz": npz_path,
        "yaml": yaml_path,
        "time": time_array,
        "u_full": u_full,
        "v_full": v_full,
        "a_full": a_full,
        "u_trans": u_trans,
        "v_trans": v_trans,
        "a_trans": a_trans,
        "F_time": F_time,
        "natural_freqs_hz": natural_freqs,
        "student_params_used": {
            "alpha_flap": alpha_flap,
            "alpha_edge": alpha_edge,
            "alpha_torsion": alpha_torsion,
            "zeta_structural": zeta_structural,
            "ref_freq_hz": ref_freq_used,
            "use_base_initial_twist_phi": use_base_initial_twist_phi,
            "base_phi_twist_column": base_phi_twist_column if use_base_initial_twist_phi else None,
            "base_phi_sign": base_phi_sign if use_base_initial_twist_phi else None,
            "rotate_mass": rotate_mass if use_base_initial_twist_phi else False,
        },
        "base_model_info": base_model_info,
        "base_phi_element_deg": None if phi_element_deg is None else np.asarray(phi_element_deg, dtype=np.float64),
        "eta_points": eta_points,
    }


def run_student_suite(
    blade_csv: str | Path,
    output_dir: str | Path,
    cases: List[Dict[str, Any]],
    student_params: Optional[Dict[str, Any]] = None,
    u0_trans: Optional[np.ndarray] = None,
    v0_trans: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """
    连续运行多个 student case。

    每个 case 需要包含：
        - case_name
        - use_time_series_load
        - t_initial
        - t_final
        - dt
        - time_series_load_file，当 use_time_series_load=True 时必须提供

    默认情况下，所有 case 都使用新的 base student：
        Phi_base(s) = - initial_twist_deg(s)
    """
    results = []
    for case in cases:
        result = run_student_case(
            blade_csv=blade_csv,
            output_dir=output_dir,
            case_name=case["case_name"],
            use_time_series_load=case["use_time_series_load"],
            t_initial=case["t_initial"],
            t_final=case["t_final"],
            dt=case["dt"],
            time_series_load_file=case.get("time_series_load_file", None),
            student_params=student_params,
            u0_trans=u0_trans,
            v0_trans=v0_trans,
        )
        results.append(result)
    return results


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    blade_csv = PROJECT_ROOT / "data" / "raw" / "nrel5mw" / "blade_master.csv"
    output_dir = PROJECT_ROOT / "results" / "student" / "newdyn_test"
    ts_dat = PROJECT_ROOT / "data" / "load" / "train_complex_case.dat"

    cases = [
        {
            "case_name": "stu_ts_off_10s",
            "use_time_series_load": False,
            "t_initial": 0.0,
            "t_final": 10.0,
            "dt": 0.01,
        },
        {
            "case_name": "stu_ts_on_10s",
            "use_time_series_load": True,
            "t_initial": 0.0,
            "t_final": 10.0,
            "dt": 0.01,
            "time_series_load_file": ts_dat,
        },
    ]

    student_params = {
        "alpha_flap": 1.0,
        "alpha_edge": 1.0,
        "alpha_torsion": 1.0,
        "zeta_structural": 0.015,
        "ref_freq_hz": None,

        # 新 base student 默认构型
        "use_base_initial_twist_phi": True,
        "base_phi_twist_column": "initial_twist_deg",
        "base_phi_sign": -1.0,
        "rotate_mass": False,
    }

    results = run_student_suite(
        blade_csv=blade_csv,
        output_dir=output_dir,
        cases=cases,
        student_params=student_params,
        u0_trans=None,
        v0_trans=None,
    )

    print("\n全部完成：")
    for r in results:
        print(f"- {r['case_name']}:")
        print(f"    npz  = {r['npz']}")
        print(f"    yaml = {r['yaml']}")