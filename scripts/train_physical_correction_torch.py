from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy.linalg import eigh


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from scripts.prepare_physical_training_cases_base_phi import (  # noqa: E402
    PhysicalTrainingCasePrepConfig,
    prepare_physical_training_cases,
)
from src.student.base_full_order_builder import build_base_student_full_order_mk  # noqa: E402
from src.student.full_order_corrected_core_torch import (  # noqa: E402
    FullOrderCorrectedCoreTorch,
    FullOrderCorrectedCoreTorchConfig,
)


# ============================================================
# 以后主要只需要改这里
# ============================================================

TRAIN_LOAD_FILES = [
    str(PROJECT_ROOT / "data" / "load" / "train_complex_case.dat"),
]

VALID_LOAD_FILES = [
    # 建议后续放入独立验证工况，例如：
    str(PROJECT_ROOT / "data" / "load" / "test_complex_case.dat"),
]


# ============================================================
# Config
# ============================================================

@dataclass
class PhysicalCorrectionTrainConfig:
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

    training_case_dir: str = str(PROJECT_ROOT / "results" / "student" / "physical_training_cases_base_phi")
    output_dir: str = str(PROJECT_ROOT / "results" / "student" / "train_physical_correction_torch")

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

    use_base_initial_twist_phi: bool = True
    base_phi_twist_column: str = "initial_twist_deg"
    base_phi_sign: float = -1.0
    rotate_mass: bool = False

    prepare_cases: bool = True
    rebuild_cases: bool = False
    remove_initial_offset: bool = True

    epochs: int = 300
    lr: float = 2.0e-3
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0

    torch_dtype: str = "float64"
    device: str = "cpu"
    linear_solve_mode: str = "solve"

    force_scale_limit: float = 0.50
    force_cross_limit: float = 0.05
    moment_arm_limit_m: float = 2.00

    w_y: float = 1.0
    w_x_guard: float = 5.0
    x_guard_tol: float = 0.02

    w_force_scale_mag: float = 1.0e-3
    w_force_cross_mag: float = 1.0e-3
    w_moment_mag: float = 1.0e-3
    w_smooth: float = 1.0e-2
    w_curvature: float = 1.0e-2

    use_valid_for_best: bool = True
    early_stop_patience: int = 80
    early_stop_min_delta: float = 1.0e-5
    min_epochs: int = 30

    lr_plateau_patience: int = 30
    lr_plateau_factor: float = 0.5
    min_lr: float = 1.0e-5

    print_every: int = 1
    save_every: int = 25
    save_report: bool = True


# ============================================================
# Utilities
# ============================================================

def _parse_file_list(value: Optional[str], fallback: List[str]) -> List[str]:
    if value is None:
        return list(fallback)

    value = value.strip()
    if len(value) == 0:
        return []

    return [item.strip() for item in value.split(",") if item.strip()]


def _assert_existing_file(path: str | Path, label: str) -> Path:
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {p}")
    return p


def _get_torch_dtype(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def _safe_case_stem(load_file: str | Path) -> str:
    stem = Path(load_file).stem
    safe = []
    for ch in stem:
        if ch.isalnum() or ch in ["_", "-", "."]:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def _expected_case_path(
    *,
    training_case_dir: str | Path,
    split: str,
    load_file: str | Path,
) -> Path:
    stem = _safe_case_stem(load_file)
    return (
        Path(training_case_dir).resolve()
        / split
        / stem
        / f"{stem}_phi_base_training_case.npz"
    )


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

    offset = component_to_offset[component]
    return np.asarray([i * 6 + offset for i in range(n_nodes)], dtype=np.int64)


def _remove_initial_offset_torch(u: torch.Tensor) -> torch.Tensor:
    return u - u[:1, :]


def _compute_natural_frequencies_hz(
    M: np.ndarray,
    K: np.ndarray,
    *,
    num_modes: int = 10,
) -> np.ndarray:
    eigvals, _ = eigh(K, M)
    eigvals = np.asarray(eigvals, dtype=np.float64)
    valid = eigvals > 0.0
    if not np.any(valid):
        raise ValueError("No positive eigenvalues found when computing natural frequencies.")
    freqs = np.sqrt(eigvals[valid]) / (2.0 * np.pi)
    return freqs[:num_modes]


def _build_structural_damping_matrix(
    *,
    K: np.ndarray,
    zeta_structural: float,
    ref_freq_hz: Optional[float],
    natural_freqs_hz: np.ndarray,
) -> Tuple[np.ndarray, Optional[float]]:
    if zeta_structural <= 0.0:
        return np.zeros_like(K), ref_freq_hz

    if ref_freq_hz is None:
        if natural_freqs_hz.size < 1:
            raise ValueError("natural_freqs_hz is empty; cannot infer ref_freq_hz.")
        ref_freq_hz = float(natural_freqs_hz[0])

    beta_damp = 2.0 * float(zeta_structural) / (2.0 * np.pi * float(ref_freq_hz))
    C = beta_damp * K
    return np.asarray(C, dtype=np.float64), float(ref_freq_hz)


def _save_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(payload, f, sort_keys=False, allow_unicode=True)


def _save_history_csv(path: Path, history: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if len(history) == 0:
        return

    fieldnames = list(history[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


# ============================================================
# Training case cache
# ============================================================

@dataclass
class TrainingCase:
    path: Path
    name: str
    time: torch.Tensor
    F_raw: torch.Tensor
    u_teacher: torch.Tensor
    u_base: torch.Tensor
    base_x_mse: float
    base_y_mse: float
    remove_initial_offset: bool


def load_training_case(
    path: str | Path,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> TrainingCase:
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Training case not found: {path}")

    data = np.load(path)

    time = np.asarray(data["time"], dtype=np.float64)
    F_raw = np.asarray(data["F_raw"], dtype=np.float64)
    u_teacher = np.asarray(data["u_teacher"], dtype=np.float64)
    u_base = np.asarray(data["u_base"], dtype=np.float64)

    if F_raw.shape != u_teacher.shape:
        raise ValueError(
            f"F_raw/u_teacher shape mismatch in {path}: "
            f"{F_raw.shape} vs {u_teacher.shape}"
        )

    if u_base.shape != u_teacher.shape:
        raise ValueError(
            f"u_base/u_teacher shape mismatch in {path}: "
            f"{u_base.shape} vs {u_teacher.shape}"
        )

    base_x_mse = float(np.asarray(data["full_x_mse"]).reshape(-1)[0])
    base_y_mse = float(np.asarray(data["full_y_mse"]).reshape(-1)[0])
    remove_initial_offset = bool(int(np.asarray(data["remove_initial_offset"]).reshape(-1)[0]))

    return TrainingCase(
        path=path,
        name=path.stem,
        time=torch.as_tensor(time, dtype=dtype, device=device),
        F_raw=torch.as_tensor(F_raw, dtype=dtype, device=device),
        u_teacher=torch.as_tensor(u_teacher, dtype=dtype, device=device),
        u_base=torch.as_tensor(u_base, dtype=dtype, device=device),
        base_x_mse=base_x_mse,
        base_y_mse=base_y_mse,
        remove_initial_offset=remove_initial_offset,
    )


def load_training_cases(
    paths: List[str | Path],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> List[TrainingCase]:
    return [load_training_case(p, dtype=dtype, device=device) for p in paths]


# ============================================================
# Load mapping correction
# ============================================================

class LoadMappingCorrection(nn.Module):
    """
    第一版只训练 f：外载荷映射 / 等效载荷修正。

    DOF layout per node:
        [Fx, Fy, Fz, Mx, My, Mz]

    零初始化时严格满足：
        F_eff = F_raw
    """

    def __init__(
        self,
        *,
        n_nodes: int,
        force_scale_limit: float,
        force_cross_limit: float,
        moment_arm_limit_m: float,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()

        self.n_nodes = int(n_nodes)
        self.force_scale_limit = float(force_scale_limit)
        self.force_cross_limit = float(force_cross_limit)
        self.moment_arm_limit_m = float(moment_arm_limit_m)

        self.raw_force_scale = nn.Parameter(
            torch.zeros((self.n_nodes, 3), dtype=dtype, device=device)
        )
        self.raw_force_cross = nn.Parameter(
            torch.zeros((self.n_nodes, 2), dtype=dtype, device=device)
        )
        self.raw_moment_arm = nn.Parameter(
            torch.zeros((self.n_nodes, 4), dtype=dtype, device=device)
        )

    def force_scale(self) -> torch.Tensor:
        return 1.0 + self.force_scale_limit * torch.tanh(self.raw_force_scale)

    def force_cross(self) -> torch.Tensor:
        return self.force_cross_limit * torch.tanh(self.raw_force_cross)

    def moment_arm(self) -> torch.Tensor:
        return self.moment_arm_limit_m * torch.tanh(self.raw_moment_arm)

    def forward(self, F_raw: torch.Tensor) -> torch.Tensor:
        if F_raw.ndim != 2:
            raise ValueError(f"F_raw must be 2D [T, n_dofs], got {F_raw.shape}")

        T, n_dofs = F_raw.shape
        expected_n_dofs = self.n_nodes * 6
        if n_dofs != expected_n_dofs:
            raise ValueError(f"Expected n_dofs={expected_n_dofs}, got {n_dofs}")

        F = F_raw.reshape(T, self.n_nodes, 6)

        Fx = F[:, :, 0]
        Fy = F[:, :, 1]
        Fz = F[:, :, 2]
        Mx = F[:, :, 3]
        My = F[:, :, 4]
        Mz = F[:, :, 5]

        scale = self.force_scale()
        cross = self.force_cross()
        arm = self.moment_arm()

        sx = scale[:, 0]
        sy = scale[:, 1]
        sz = scale[:, 2]

        c_xy = cross[:, 0]
        c_yx = cross[:, 1]

        b_mx_from_fy = arm[:, 0]
        b_my_from_fx = arm[:, 1]
        b_mz_from_fx = arm[:, 2]
        b_mz_from_fy = arm[:, 3]

        Fx_eff = sx[None, :] * Fx + c_xy[None, :] * Fy
        Fy_eff = sy[None, :] * Fy + c_yx[None, :] * Fx
        Fz_eff = sz[None, :] * Fz

        Mx_eff = Mx + b_mx_from_fy[None, :] * Fy
        My_eff = My + b_my_from_fx[None, :] * Fx
        Mz_eff = Mz + b_mz_from_fx[None, :] * Fx + b_mz_from_fy[None, :] * Fy

        F_eff = torch.stack(
            [Fx_eff, Fy_eff, Fz_eff, Mx_eff, My_eff, Mz_eff],
            dim=-1,
        )

        return F_eff.reshape(T, n_dofs)

    def normalized_parameter_matrix(self) -> torch.Tensor:
        scale_norm = (self.force_scale() - 1.0) / max(self.force_scale_limit, 1.0e-12)
        cross_norm = self.force_cross() / max(self.force_cross_limit, 1.0e-12)
        arm_norm = self.moment_arm() / max(self.moment_arm_limit_m, 1.0e-12)
        return torch.cat([scale_norm, cross_norm, arm_norm], dim=-1)

    def regularization_terms(self) -> Dict[str, torch.Tensor]:
        P = self.normalized_parameter_matrix()

        force_scale_mag = torch.mean(
            ((self.force_scale() - 1.0) / max(self.force_scale_limit, 1.0e-12)) ** 2
        )
        force_cross_mag = torch.mean(
            (self.force_cross() / max(self.force_cross_limit, 1.0e-12)) ** 2
        )
        moment_mag = torch.mean(
            (self.moment_arm() / max(self.moment_arm_limit_m, 1.0e-12)) ** 2
        )

        if P.shape[0] >= 2:
            smooth = torch.mean((P[1:, :] - P[:-1, :]) ** 2)
        else:
            smooth = torch.zeros((), dtype=P.dtype, device=P.device)

        if P.shape[0] >= 3:
            curvature = torch.mean((P[2:, :] - 2.0 * P[1:-1, :] + P[:-2, :]) ** 2)
        else:
            curvature = torch.zeros((), dtype=P.dtype, device=P.device)

        return {
            "force_scale_mag": force_scale_mag,
            "force_cross_mag": force_cross_mag,
            "moment_mag": moment_mag,
            "smooth": smooth,
            "curvature": curvature,
        }

    def export_numpy(self) -> Dict[str, np.ndarray]:
        with torch.no_grad():
            return {
                "force_scale": self.force_scale().detach().cpu().numpy(),
                "force_cross": self.force_cross().detach().cpu().numpy(),
                "moment_arm": self.moment_arm().detach().cpu().numpy(),
                "raw_force_scale": self.raw_force_scale.detach().cpu().numpy(),
                "raw_force_cross": self.raw_force_cross.detach().cpu().numpy(),
                "raw_moment_arm": self.raw_moment_arm.detach().cpu().numpy(),
            }


# ============================================================
# Base core
# ============================================================

def _call_build_base_student_full_order_mk(
    *,
    cfg: PhysicalCorrectionTrainConfig,
) -> Dict[str, Any]:
    blade_csv = _assert_existing_file(cfg.blade_csv, "blade_csv")

    common_kwargs = {
        "blade_csv": blade_csv,
        "model_name": "physical_correction_base_student",
        "alpha_flap": float(cfg.alpha_flap),
        "alpha_edge": float(cfg.alpha_edge),
        "alpha_torsion": float(cfg.alpha_torsion),
        "rotate_mass": bool(cfg.rotate_mass),
    }

    try:
        result = build_base_student_full_order_mk(
            **common_kwargs,
            twist_column=str(cfg.base_phi_twist_column),
            phi_sign=float(cfg.base_phi_sign),
            verbose=True,
        )
    except TypeError:
        try:
            result = build_base_student_full_order_mk(
                **common_kwargs,
                base_phi_twist_column=str(cfg.base_phi_twist_column),
                base_phi_sign=float(cfg.base_phi_sign),
                verbose=True,
            )
        except TypeError:
            result = build_base_student_full_order_mk(
                **common_kwargs,
                twist_column=str(cfg.base_phi_twist_column),
                sign=float(cfg.base_phi_sign),
            )

    if not isinstance(result, dict):
        raise TypeError(
            "build_base_student_full_order_mk is expected to return a dict "
            "with at least keys 'M' and 'K'."
        )

    if "M" not in result or "K" not in result:
        raise KeyError("base_full_order_builder result must contain keys 'M' and 'K'.")

    return result


def build_base_core(
    *,
    cfg: PhysicalCorrectionTrainConfig,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[FullOrderCorrectedCoreTorch, Dict[str, Any]]:
    print()
    print("[Build Base Full-Order Core]")
    print(f"  Phi_base = {cfg.base_phi_sign:+.1f} * {cfg.base_phi_twist_column}(s)")
    print(f"  rotate_mass = {cfg.rotate_mass}")

    base_result = _call_build_base_student_full_order_mk(cfg=cfg)

    M = np.asarray(base_result["M"], dtype=np.float64)
    K = np.asarray(base_result["K"], dtype=np.float64)

    natural_freqs_hz = _compute_natural_frequencies_hz(M, K, num_modes=10)
    C, ref_freq_used = _build_structural_damping_matrix(
        K=K,
        zeta_structural=float(cfg.zeta_structural),
        ref_freq_hz=cfg.ref_freq_hz,
        natural_freqs_hz=natural_freqs_hz,
    )

    core = FullOrderCorrectedCoreTorch(
        M=M,
        K=K,
        C=C,
        dt=float(cfg.dt),
        config=FullOrderCorrectedCoreTorchConfig(
            gamma=0.5,
            beta=0.25,
            dtype=dtype,
            linear_solve_mode=cfg.linear_solve_mode,
        ),
    ).to(device)

    phi_element_deg = base_result.get("phi_element_deg", base_result.get("base_phi_element_deg", []))
    phi_element_deg = np.asarray(phi_element_deg, dtype=np.float64)

    info = {
        "M_shape": list(M.shape),
        "K_shape": list(K.shape),
        "C_shape": list(C.shape),
        "natural_freqs_hz": natural_freqs_hz.tolist(),
        "ref_freq_used": ref_freq_used,
        "phi_element_deg": phi_element_deg.tolist(),
    }

    print(f"  M shape = {M.shape}")
    print(f"  K shape = {K.shape}")
    print(f"  C shape = {C.shape}")
    print(f"  natural_freqs[:5] = {natural_freqs_hz[:5]}")
    print(f"  ref_freq_used = {ref_freq_used}")

    return core, info


# ============================================================
# Training logic
# ============================================================

def rollout_prediction(
    *,
    core: FullOrderCorrectedCoreTorch,
    correction: LoadMappingCorrection,
    case: TrainingCase,
    n_dofs: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    F_eff = correction(case.F_raw)

    u0 = torch.zeros((n_dofs,), dtype=dtype, device=device)
    v0 = torch.zeros((n_dofs,), dtype=dtype, device=device)

    u_pred_raw, _, _ = core.rollout(
        F_time=F_eff,
        u0=u0,
        v0=v0,
        theta_full=None,
        return_debug=False,
    )

    if case.remove_initial_offset:
        return _remove_initial_offset_torch(u_pred_raw)

    return u_pred_raw


def compute_case_loss(
    *,
    u_pred: torch.Tensor,
    case: TrainingCase,
    x_idx: torch.Tensor,
    y_idx: torch.Tensor,
    cfg: PhysicalCorrectionTrainConfig,
) -> Dict[str, torch.Tensor]:
    eps = torch.as_tensor(1.0e-12, dtype=u_pred.dtype, device=u_pred.device)

    diff = u_pred - case.u_teacher

    x_mse = torch.mean(diff[:, x_idx] ** 2)
    y_mse = torch.mean(diff[:, y_idx] ** 2)

    base_x_mse = torch.as_tensor(case.base_x_mse, dtype=u_pred.dtype, device=u_pred.device)
    base_y_mse = torch.as_tensor(case.base_y_mse, dtype=u_pred.dtype, device=u_pred.device)

    x_ratio = x_mse / torch.clamp(base_x_mse, min=eps)
    y_ratio = y_mse / torch.clamp(base_y_mse, min=eps)

    x_guard = torch.relu(x_ratio - (1.0 + float(cfg.x_guard_tol))) ** 2

    data_loss = float(cfg.w_y) * y_ratio + float(cfg.w_x_guard) * x_guard

    return {
        "data_loss": data_loss,
        "x_mse": x_mse,
        "y_mse": y_mse,
        "x_ratio": x_ratio,
        "y_ratio": y_ratio,
        "x_guard": x_guard,
    }


def evaluate_cases(
    *,
    cases: List[TrainingCase],
    core: FullOrderCorrectedCoreTorch,
    correction: LoadMappingCorrection,
    x_idx: torch.Tensor,
    y_idx: torch.Tensor,
    cfg: PhysicalCorrectionTrainConfig,
    dtype: torch.dtype,
    device: torch.device,
    require_grad: bool,
) -> Dict[str, torch.Tensor]:
    if len(cases) == 0:
        zero = torch.zeros((), dtype=dtype, device=device)
        return {
            "total_loss": zero,
            "data_loss": zero,
            "reg_loss": zero,
            "y_ratio": zero,
            "x_ratio": zero,
            "x_guard": zero,
        }

    n_dofs = int(len(x_idx) * 6)

    context = torch.enable_grad() if require_grad else torch.no_grad()

    with context:
        data_losses = []
        y_ratios = []
        x_ratios = []
        x_guards = []

        for case in cases:
            u_pred = rollout_prediction(
                core=core,
                correction=correction,
                case=case,
                n_dofs=n_dofs,
                dtype=dtype,
                device=device,
            )

            loss_dict = compute_case_loss(
                u_pred=u_pred,
                case=case,
                x_idx=x_idx,
                y_idx=y_idx,
                cfg=cfg,
            )

            data_losses.append(loss_dict["data_loss"])
            y_ratios.append(loss_dict["y_ratio"])
            x_ratios.append(loss_dict["x_ratio"])
            x_guards.append(loss_dict["x_guard"])

        data_loss = torch.stack(data_losses).mean()
        y_ratio = torch.stack(y_ratios).mean()
        x_ratio = torch.stack(x_ratios).mean()
        x_guard = torch.stack(x_guards).mean()

        reg_terms = correction.regularization_terms()

        reg_loss = (
            float(cfg.w_force_scale_mag) * reg_terms["force_scale_mag"]
            + float(cfg.w_force_cross_mag) * reg_terms["force_cross_mag"]
            + float(cfg.w_moment_mag) * reg_terms["moment_mag"]
            + float(cfg.w_smooth) * reg_terms["smooth"]
            + float(cfg.w_curvature) * reg_terms["curvature"]
        )

        total_loss = data_loss + reg_loss

        return {
            "total_loss": total_loss,
            "data_loss": data_loss,
            "reg_loss": reg_loss,
            "y_ratio": y_ratio,
            "x_ratio": x_ratio,
            "x_guard": x_guard,
            "reg_force_scale_mag": reg_terms["force_scale_mag"],
            "reg_force_cross_mag": reg_terms["force_cross_mag"],
            "reg_moment_mag": reg_terms["moment_mag"],
            "reg_smooth": reg_terms["smooth"],
            "reg_curvature": reg_terms["curvature"],
        }


# ============================================================
# Save helpers
# ============================================================

def save_checkpoint(
    *,
    path: Path,
    cfg: PhysicalCorrectionTrainConfig,
    correction: LoadMappingCorrection,
    epoch: int,
    score: float,
    train_summary: Dict[str, float],
    valid_summary: Dict[str, float],
    base_core_info: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": int(epoch),
            "score": float(score),
            "config": asdict(cfg),
            "correction_state_dict": correction.state_dict(),
            "train_summary": train_summary,
            "valid_summary": valid_summary,
            "base_core_info": base_core_info,
        },
        path,
    )


def save_parameter_npz(
    *,
    path: Path,
    correction: LoadMappingCorrection,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    params = correction.export_numpy()
    np.savez(path, **params)


# ============================================================
# Case preparation
# ============================================================

def ensure_training_cases(
    *,
    cfg: PhysicalCorrectionTrainConfig,
    train_load_files: List[str],
    valid_load_files: List[str],
) -> Dict[str, List[Path]]:
    if cfg.prepare_cases:
        prep_cfg = PhysicalTrainingCasePrepConfig(
            teacher_exe=cfg.teacher_exe,
            template_inp=cfg.template_inp,
            blade_csv=cfg.blade_csv,
            output_dir=cfg.training_case_dir,
            case_prefix="physical_correction",
            t_initial=cfg.t_initial,
            t_final=cfg.t_final,
            dt=cfg.dt,
            teacher_node_start=cfg.teacher_node_start,
            teacher_node_end=cfg.teacher_node_end,
            teacher_demean=cfg.teacher_demean,
            alpha_flap=cfg.alpha_flap,
            alpha_edge=cfg.alpha_edge,
            alpha_torsion=cfg.alpha_torsion,
            zeta_structural=cfg.zeta_structural,
            ref_freq_hz=cfg.ref_freq_hz,
            use_base_initial_twist_phi=cfg.use_base_initial_twist_phi,
            base_phi_twist_column=cfg.base_phi_twist_column,
            base_phi_sign=cfg.base_phi_sign,
            rotate_mass=cfg.rotate_mass,
            remove_initial_offset=cfg.remove_initial_offset,
            rebuild_cases=cfg.rebuild_cases,
            save_summary=True,
        )

        return prepare_physical_training_cases(
            cfg=prep_cfg,
            train_load_files=train_load_files,
            valid_load_files=valid_load_files,
            rebuild=cfg.rebuild_cases,
        )

    prepared = {
        "train": [
            _expected_case_path(
                training_case_dir=cfg.training_case_dir,
                split="train",
                load_file=f,
            )
            for f in train_load_files
        ],
        "valid": [
            _expected_case_path(
                training_case_dir=cfg.training_case_dir,
                split="valid",
                load_file=f,
            )
            for f in valid_load_files
        ],
    }

    for split, paths in prepared.items():
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(
                    f"{split} cache not found: {p}. "
                    f"Use --prepare-cases or run prepare script first."
                )

    return prepared


# ============================================================
# CLI
# ============================================================

def parse_args() -> Tuple[PhysicalCorrectionTrainConfig, List[str], List[str]]:
    d = PhysicalCorrectionTrainConfig()

    parser = argparse.ArgumentParser(
        description=(
            "Train physical correction parameters with fixed "
            "Phi_base(s) = -initial_twist_deg(s). "
            "Current version trains LoadMappingCorrection only."
        )
    )

    parser.add_argument("--teacher-exe", type=str, default=d.teacher_exe)
    parser.add_argument("--template-inp", type=str, default=d.template_inp)
    parser.add_argument("--blade-csv", type=str, default=d.blade_csv)

    parser.add_argument("--training-case-dir", type=str, default=d.training_case_dir)
    parser.add_argument("--output-dir", type=str, default=d.output_dir)

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
    base_phi_group.add_argument("--use-base-initial-twist-phi", dest="use_base_initial_twist_phi", action="store_true")
    base_phi_group.add_argument("--no-base-initial-twist-phi", dest="use_base_initial_twist_phi", action="store_false")
    parser.set_defaults(use_base_initial_twist_phi=d.use_base_initial_twist_phi)

    parser.add_argument("--base-phi-twist-column", type=str, default=d.base_phi_twist_column)
    parser.add_argument("--base-phi-sign", type=float, default=d.base_phi_sign)

    mass_group = parser.add_mutually_exclusive_group()
    mass_group.add_argument("--rotate-mass", dest="rotate_mass", action="store_true")
    mass_group.add_argument("--no-rotate-mass", dest="rotate_mass", action="store_false")
    parser.set_defaults(rotate_mass=d.rotate_mass)

    prepare_group = parser.add_mutually_exclusive_group()
    prepare_group.add_argument("--prepare-cases", dest="prepare_cases", action="store_true")
    prepare_group.add_argument("--no-prepare-cases", dest="prepare_cases", action="store_false")
    parser.set_defaults(prepare_cases=d.prepare_cases)

    rebuild_group = parser.add_mutually_exclusive_group()
    rebuild_group.add_argument("--rebuild-cases", dest="rebuild_cases", action="store_true")
    rebuild_group.add_argument("--no-rebuild-cases", dest="rebuild_cases", action="store_false")
    parser.set_defaults(rebuild_cases=d.rebuild_cases)

    offset_group = parser.add_mutually_exclusive_group()
    offset_group.add_argument("--remove-initial-offset", dest="remove_initial_offset", action="store_true")
    offset_group.add_argument("--keep-initial-offset", dest="remove_initial_offset", action="store_false")
    parser.set_defaults(remove_initial_offset=d.remove_initial_offset)

    parser.add_argument("--epochs", type=int, default=d.epochs)
    parser.add_argument("--lr", type=float, default=d.lr)
    parser.add_argument("--weight-decay", type=float, default=d.weight_decay)
    parser.add_argument("--grad-clip-norm", type=float, default=d.grad_clip_norm)

    parser.add_argument("--torch-dtype", type=str, default=d.torch_dtype, choices=["float64", "float32"])
    parser.add_argument("--device", type=str, default=d.device)
    parser.add_argument("--linear-solve-mode", type=str, default=d.linear_solve_mode, choices=["solve", "inverse"])

    parser.add_argument("--force-scale-limit", type=float, default=d.force_scale_limit)
    parser.add_argument("--force-cross-limit", type=float, default=d.force_cross_limit)
    parser.add_argument("--moment-arm-limit-m", type=float, default=d.moment_arm_limit_m)

    parser.add_argument("--w-y", type=float, default=d.w_y)
    parser.add_argument("--w-x-guard", type=float, default=d.w_x_guard)
    parser.add_argument("--x-guard-tol", type=float, default=d.x_guard_tol)

    parser.add_argument("--w-force-scale-mag", type=float, default=d.w_force_scale_mag)
    parser.add_argument("--w-force-cross-mag", type=float, default=d.w_force_cross_mag)
    parser.add_argument("--w-moment-mag", type=float, default=d.w_moment_mag)
    parser.add_argument("--w-smooth", type=float, default=d.w_smooth)
    parser.add_argument("--w-curvature", type=float, default=d.w_curvature)

    valid_group = parser.add_mutually_exclusive_group()
    valid_group.add_argument("--use-valid-for-best", dest="use_valid_for_best", action="store_true")
    valid_group.add_argument("--use-train-for-best", dest="use_valid_for_best", action="store_false")
    parser.set_defaults(use_valid_for_best=d.use_valid_for_best)

    parser.add_argument("--early-stop-patience", type=int, default=d.early_stop_patience)
    parser.add_argument("--early-stop-min-delta", type=float, default=d.early_stop_min_delta)
    parser.add_argument("--min-epochs", type=int, default=d.min_epochs)

    parser.add_argument("--lr-plateau-patience", type=int, default=d.lr_plateau_patience)
    parser.add_argument("--lr-plateau-factor", type=float, default=d.lr_plateau_factor)
    parser.add_argument("--min-lr", type=float, default=d.min_lr)

    parser.add_argument("--print-every", type=int, default=d.print_every)
    parser.add_argument("--save-every", type=int, default=d.save_every)

    summary_group = parser.add_mutually_exclusive_group()
    summary_group.add_argument("--save-report", dest="save_report", action="store_true")
    summary_group.add_argument("--no-save-report", dest="save_report", action="store_false")
    parser.set_defaults(save_report=d.save_report)

    args = parser.parse_args()

    cfg = PhysicalCorrectionTrainConfig(
        teacher_exe=args.teacher_exe,
        template_inp=args.template_inp,
        blade_csv=args.blade_csv,
        training_case_dir=args.training_case_dir,
        output_dir=args.output_dir,
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
        prepare_cases=args.prepare_cases,
        rebuild_cases=args.rebuild_cases,
        remove_initial_offset=args.remove_initial_offset,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        torch_dtype=args.torch_dtype,
        device=args.device,
        linear_solve_mode=args.linear_solve_mode,
        force_scale_limit=args.force_scale_limit,
        force_cross_limit=args.force_cross_limit,
        moment_arm_limit_m=args.moment_arm_limit_m,
        w_y=args.w_y,
        w_x_guard=args.w_x_guard,
        x_guard_tol=args.x_guard_tol,
        w_force_scale_mag=args.w_force_scale_mag,
        w_force_cross_mag=args.w_force_cross_mag,
        w_moment_mag=args.w_moment_mag,
        w_smooth=args.w_smooth,
        w_curvature=args.w_curvature,
        use_valid_for_best=args.use_valid_for_best,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        min_epochs=args.min_epochs,
        lr_plateau_patience=args.lr_plateau_patience,
        lr_plateau_factor=args.lr_plateau_factor,
        min_lr=args.min_lr,
        print_every=args.print_every,
        save_every=args.save_every,
        save_report=args.save_report,
    )

    train_load_files = _parse_file_list(args.train_load_files, TRAIN_LOAD_FILES)
    valid_load_files = _parse_file_list(args.valid_load_files, VALID_LOAD_FILES)

    return cfg, train_load_files, valid_load_files


# ============================================================
# Main
# ============================================================

def main() -> None:
    cfg, train_load_files, valid_load_files = parse_args()

    output_dir = Path(cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = _get_torch_dtype(cfg.torch_dtype)
    device = torch.device(cfg.device)

    print()
    print("[Train Physical Correction Torch]")
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

    if len(train_load_files) == 0:
        raise ValueError("No training load files provided.")

    print()
    print("[1/5] Preparing or locating training cases")
    prepared = ensure_training_cases(
        cfg=cfg,
        train_load_files=train_load_files,
        valid_load_files=valid_load_files,
    )

    print()
    print("[Prepared Cases]")
    print("  train:")
    for p in prepared["train"]:
        print(f"    - {p}")
    print("  valid:")
    if len(prepared["valid"]) == 0:
        print("    - <empty>")
    else:
        for p in prepared["valid"]:
            print(f"    - {p}")

    print()
    print("[2/5] Loading training case caches")
    train_cases = load_training_cases(prepared["train"], dtype=dtype, device=device)
    valid_cases = load_training_cases(prepared["valid"], dtype=dtype, device=device)

    if len(valid_cases) == 0:
        print()
        print("[Warning] No validation cases provided. Best model will be selected by train score.")
        use_valid_for_best = False
    else:
        use_valid_for_best = bool(cfg.use_valid_for_best)

    first_case = train_cases[0]
    n_dofs = int(first_case.F_raw.shape[1])
    if n_dofs % 6 != 0:
        raise ValueError(f"n_dofs must be divisible by 6, got {n_dofs}")

    n_nodes = n_dofs // 6

    print(f"  n_train_cases = {len(train_cases)}")
    print(f"  n_valid_cases = {len(valid_cases)}")
    print(f"  n_dofs        = {n_dofs}")
    print(f"  n_nodes       = {n_nodes}")

    x_idx = torch.as_tensor(_component_indices(n_nodes, "x"), dtype=torch.long, device=device)
    y_idx = torch.as_tensor(_component_indices(n_nodes, "y"), dtype=torch.long, device=device)

    print()
    print("[3/5] Building base full-order Newmark core")
    core, base_core_info = build_base_core(cfg=cfg, dtype=dtype, device=device)

    print()
    print("[4/5] Initializing LoadMappingCorrection")
    correction = LoadMappingCorrection(
        n_nodes=n_nodes,
        force_scale_limit=float(cfg.force_scale_limit),
        force_cross_limit=float(cfg.force_cross_limit),
        moment_arm_limit_m=float(cfg.moment_arm_limit_m),
        dtype=dtype,
        device=device,
    ).to(device)

    optimizer = torch.optim.Adam(
        correction.parameters(),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(cfg.lr_plateau_factor),
        patience=int(cfg.lr_plateau_patience),
        min_lr=float(cfg.min_lr),
    )

    best_score = float("inf")
    best_epoch = -1
    no_improve_epochs = 0
    history: List[Dict[str, float]] = []

    best_ckpt_path = output_dir / "best_load_mapping_correction.pt"
    final_ckpt_path = output_dir / "final_load_mapping_correction.pt"
    best_param_path = output_dir / "best_load_mapping_params.npz"
    final_param_path = output_dir / "final_load_mapping_params.npz"
    history_path = output_dir / "training_history.csv"
    report_path = output_dir / "train_physical_correction_report.yaml"

    print()
    print("[5/5] Training")
    print("  active correction module = LoadMappingCorrection")
    print("  initial correction       = zero, so F_eff == F_raw")

    for epoch in range(1, int(cfg.epochs) + 1):
        correction.train()
        optimizer.zero_grad(set_to_none=True)

        train_eval = evaluate_cases(
            cases=train_cases,
            core=core,
            correction=correction,
            x_idx=x_idx,
            y_idx=y_idx,
            cfg=cfg,
            dtype=dtype,
            device=device,
            require_grad=True,
        )

        train_loss = train_eval["total_loss"]
        train_loss.backward()

        if cfg.grad_clip_norm is not None and float(cfg.grad_clip_norm) > 0.0:
            torch.nn.utils.clip_grad_norm_(
                correction.parameters(),
                max_norm=float(cfg.grad_clip_norm),
            )

        optimizer.step()

        correction.eval()

        train_eval_detached = evaluate_cases(
            cases=train_cases,
            core=core,
            correction=correction,
            x_idx=x_idx,
            y_idx=y_idx,
            cfg=cfg,
            dtype=dtype,
            device=device,
            require_grad=False,
        )

        if len(valid_cases) > 0:
            valid_eval = evaluate_cases(
                cases=valid_cases,
                core=core,
                correction=correction,
                x_idx=x_idx,
                y_idx=y_idx,
                cfg=cfg,
                dtype=dtype,
                device=device,
                require_grad=False,
            )
        else:
            valid_eval = train_eval_detached

        train_total = float(train_eval_detached["total_loss"].detach().cpu())
        train_y = float(train_eval_detached["y_ratio"].detach().cpu())
        train_x = float(train_eval_detached["x_ratio"].detach().cpu())
        train_x_guard = float(train_eval_detached["x_guard"].detach().cpu())
        train_reg = float(train_eval_detached["reg_loss"].detach().cpu())

        valid_total = float(valid_eval["total_loss"].detach().cpu())
        valid_y = float(valid_eval["y_ratio"].detach().cpu())
        valid_x = float(valid_eval["x_ratio"].detach().cpu())
        valid_x_guard = float(valid_eval["x_guard"].detach().cpu())

        monitor_score = valid_total if use_valid_for_best else train_total

        scheduler.step(monitor_score)
        current_lr = float(optimizer.param_groups[0]["lr"])

        improved = monitor_score < (best_score - float(cfg.early_stop_min_delta))

        if improved:
            best_score = monitor_score
            best_epoch = epoch
            no_improve_epochs = 0

            train_summary = {
                "total_loss": train_total,
                "y_ratio": train_y,
                "x_ratio": train_x,
                "x_guard": train_x_guard,
                "reg_loss": train_reg,
            }
            valid_summary = {
                "total_loss": valid_total,
                "y_ratio": valid_y,
                "x_ratio": valid_x,
                "x_guard": valid_x_guard,
            }

            save_checkpoint(
                path=best_ckpt_path,
                cfg=cfg,
                correction=correction,
                epoch=epoch,
                score=best_score,
                train_summary=train_summary,
                valid_summary=valid_summary,
                base_core_info=base_core_info,
            )
            save_parameter_npz(
                path=best_param_path,
                correction=correction,
            )
        else:
            no_improve_epochs += 1

        row = {
            "epoch": float(epoch),
            "lr": current_lr,
            "train_total": train_total,
            "train_y_ratio": train_y,
            "train_x_ratio": train_x,
            "train_x_guard": train_x_guard,
            "train_reg": train_reg,
            "valid_total": valid_total,
            "valid_y_ratio": valid_y,
            "valid_x_ratio": valid_x,
            "valid_x_guard": valid_x_guard,
            "best_score": float(best_score),
            "best_epoch": float(best_epoch),
            "no_improve_epochs": float(no_improve_epochs),
        }
        history.append(row)

        if epoch % int(cfg.print_every) == 0 or epoch == 1 or improved:
            mark = " BEST" if improved else ""
            print(
                f"[Epoch {epoch:04d}] "
                f"train_total={train_total:.8e} "
                f"train_y={train_y:.6f} "
                f"train_x={train_x:.6f} "
                f"x_guard={train_x_guard:.3e} "
                f"valid_total={valid_total:.8e} "
                f"valid_y={valid_y:.6f} "
                f"valid_x={valid_x:.6f} "
                f"lr={current_lr:.2e}"
                f"{mark}"
            )

        if epoch % int(cfg.save_every) == 0:
            _save_history_csv(history_path, history)
            save_parameter_npz(
                path=output_dir / f"load_mapping_params_epoch_{epoch:04d}.npz",
                correction=correction,
            )

        if (
            epoch >= int(cfg.min_epochs)
            and int(cfg.early_stop_patience) > 0
            and no_improve_epochs >= int(cfg.early_stop_patience)
        ):
            print()
            print("[Early Stop]")
            print(f"  epoch = {epoch}")
            print(f"  best_epoch = {best_epoch}")
            print(f"  best_score = {best_score:.8e}")
            print(f"  no_improve_epochs = {no_improve_epochs}")
            break

    correction.eval()

    final_train_eval = evaluate_cases(
        cases=train_cases,
        core=core,
        correction=correction,
        x_idx=x_idx,
        y_idx=y_idx,
        cfg=cfg,
        dtype=dtype,
        device=device,
        require_grad=False,
    )

    final_valid_eval = evaluate_cases(
        cases=valid_cases if len(valid_cases) > 0 else train_cases,
        core=core,
        correction=correction,
        x_idx=x_idx,
        y_idx=y_idx,
        cfg=cfg,
        dtype=dtype,
        device=device,
        require_grad=False,
    )

    final_train_summary = {
        "total_loss": float(final_train_eval["total_loss"].detach().cpu()),
        "y_ratio": float(final_train_eval["y_ratio"].detach().cpu()),
        "x_ratio": float(final_train_eval["x_ratio"].detach().cpu()),
        "x_guard": float(final_train_eval["x_guard"].detach().cpu()),
        "reg_loss": float(final_train_eval["reg_loss"].detach().cpu()),
    }

    final_valid_summary = {
        "total_loss": float(final_valid_eval["total_loss"].detach().cpu()),
        "y_ratio": float(final_valid_eval["y_ratio"].detach().cpu()),
        "x_ratio": float(final_valid_eval["x_ratio"].detach().cpu()),
        "x_guard": float(final_valid_eval["x_guard"].detach().cpu()),
    }

    save_checkpoint(
        path=final_ckpt_path,
        cfg=cfg,
        correction=correction,
        epoch=int(history[-1]["epoch"]) if len(history) > 0 else 0,
        score=final_valid_summary["total_loss"] if use_valid_for_best else final_train_summary["total_loss"],
        train_summary=final_train_summary,
        valid_summary=final_valid_summary,
        base_core_info=base_core_info,
    )
    save_parameter_npz(path=final_param_path, correction=correction)
    _save_history_csv(history_path, history)

    report = {
        "config": asdict(cfg),
        "train_load_files": train_load_files,
        "valid_load_files": valid_load_files,
        "prepared_cases": {
            "train": [str(p) for p in prepared["train"]],
            "valid": [str(p) for p in prepared["valid"]],
        },
        "best": {
            "best_epoch": int(best_epoch),
            "best_score": float(best_score),
            "best_checkpoint": str(best_ckpt_path),
            "best_params": str(best_param_path),
        },
        "final": {
            "final_checkpoint": str(final_ckpt_path),
            "final_params": str(final_param_path),
            "train_summary": final_train_summary,
            "valid_summary": final_valid_summary,
        },
        "base_core_info": base_core_info,
        "history_csv": str(history_path),
    }

    if bool(cfg.save_report):
        _save_yaml(report_path, report)

    print()
    print("[Saved Results]")
    print(f"  best checkpoint  = {best_ckpt_path}")
    print(f"  best params      = {best_param_path}")
    print(f"  final checkpoint = {final_ckpt_path}")
    print(f"  final params     = {final_param_path}")
    print(f"  history          = {history_path}")
    if bool(cfg.save_report):
        print(f"  report           = {report_path}")

    print()
    print("[Best Result]")
    print(f"  best_epoch = {best_epoch}")
    print(f"  best_score = {best_score:.8e}")

    print()
    print("[Final Result]")
    print(f"  train_y_ratio = {final_train_summary['y_ratio']:.6f}")
    print(f"  train_x_ratio = {final_train_summary['x_ratio']:.6f}")
    print(f"  valid_y_ratio = {final_valid_summary['y_ratio']:.6f}")
    print(f"  valid_x_ratio = {final_valid_summary['x_ratio']:.6f}")

    print()
    print("✅ PASS: physical correction training completed.")
    print("   当前仅启用 LoadMappingCorrection，即只训练 f：外载荷映射参数组。")


if __name__ == "__main__":
    main()