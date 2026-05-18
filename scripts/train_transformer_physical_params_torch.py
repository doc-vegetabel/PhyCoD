from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from scipy.linalg import eigh

# =============================================================================
# Fast training utilities: NPZ cache + torch runtime setup
# =============================================================================

@dataclass
class CachedNpzFile:
    """A small in-memory replacement for numpy.lib.npyio.NpzFile.

    It supports the common interface used by np.load(...):
      - data["key"]
      - data.files
      - data.keys()
      - data.items()
      - data.values()
      - data.get("key", default)
      - "key" in data
      - context-manager usage: with fast_np_load(path) as data:

    Note:
        This class intentionally keeps arrays alive after __exit__ so the RAM
        cache remains reusable across epochs.
    """
    data: Dict[str, Any]
    path: str

    @property
    def files(self):
        return list(self.data.keys())

    def __getitem__(self, key):
        return self.data[key]

    def __contains__(self, key):
        return key in self.data

    def keys(self):
        return self.data.keys()

    def items(self):
        return self.data.items()

    def values(self):
        return self.data.values()

    def get(self, key, default=None):
        return self.data.get(key, default)

    def close(self):
        # Kept for compatibility with np.load(...).close()
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Do not clear cache when exiting a context manager.
        return False


_FAST_NPZ_CACHE: Dict[str, CachedNpzFile] = {}
_USE_FAST_NPZ_CACHE = False


def fast_np_load(file, *args, **kwargs):
    """Cached np.load for .npz training cases.

    For non-.npz files, or when cache is disabled, this falls back to np.load.
    This avoids repeatedly reading the same training/validation case from disk
    every epoch.
    """
    global _FAST_NPZ_CACHE, _USE_FAST_NPZ_CACHE

    if not _USE_FAST_NPZ_CACHE:
        return np.load(file, *args, **kwargs)

    path = os.fspath(file)
    if not path.lower().endswith(".npz"):
        return np.load(file, *args, **kwargs)

    abs_path = os.path.abspath(path)
    cached = _FAST_NPZ_CACHE.get(abs_path)
    if cached is not None:
        return cached

    with np.load(abs_path, *args, **kwargs) as z:
        # Copy arrays into normal memory so they remain valid after z is closed.
        data = {k: z[k].copy() for k in z.files}

    cached = CachedNpzFile(data=data, path=abs_path)
    _FAST_NPZ_CACHE[abs_path] = cached
    return cached


def configure_torch_fast_runtime(
    device: str,
    *,
    allow_tf32_encoder: bool = False,
    matmul_precision: str = "high",
):
    """Configure safe-ish PyTorch runtime acceleration.

    The full-order core can still use float64. TF32 only affects eligible
    float32 CUDA matmul kernels, mainly the neural encoder/MLP side.
    """
    try:
        torch.set_float32_matmul_precision(matmul_precision)
    except Exception as exc:
        print(f"[Fast Runtime] torch.set_float32_matmul_precision skipped: {exc}")

    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32_encoder)
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32_encoder)

        print("[Fast Runtime]")
        print(f"  cudnn.benchmark       = {torch.backends.cudnn.benchmark}")
        print(f"  allow_tf32_encoder    = {allow_tf32_encoder}")
        print(f"  matmul_precision      = {matmul_precision}")
        print(f"  cuda_device           = {torch.cuda.get_device_name(torch.cuda.current_device())}")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.prepare_physical_training_cases_base_phi import (  # noqa: E402
    PhysicalTrainingCasePrepConfig,
    prepare_physical_training_cases,
)
from src.student.transformer.blade_geometry_features import (  # noqa: E402
    BladeGeometryFeatureConfig,
    build_blade_geometry_features,
)
from src.student.transformer.dynamic_physical_core_torch import (  # noqa: E402
    DynamicPhysicalCoreConfig,
    DynamicPhysicalCoreTorch,
)
from src.student.transformer.physical_parameter_registry import (  # noqa: E402
    build_physical_parameter_registry,
)
from src.student.transformer.physical_templates import (  # noqa: E402
    PhysicalTemplateConfig,
    build_dynamic_stiffness_templates,
)
from src.student.transformer.frequency_losses import (  # noqa: E402
    adaptive_phase_window_loss,
    build_frequency_alignment_cache,
    build_peak_lag_alignment_cache,
    frequency_alignment_loss,
    frequency_alignment_loss_from_cache,
    local_band_phase_loss,
    local_continuous_phase_lag_loss,
    local_phase_increment_loss,
    local_phase_slope_loss,
    local_phase_correlation_loss,
    peak_and_lag_alignment_loss,
    peak_and_lag_alignment_loss_from_cache,
    phase_drift_rate_loss,
)
from src.student.transformer.spatiotemporal_physics_encoder import (  # noqa: E402
    SpatiotemporalPhysicsEncoder,
    SpatiotemporalPhysicsEncoderConfig,
    compute_causal_load_spectral_features_from_force,
    expected_load_spectral_feature_dim,
)
from src.student.transformer.transformer_rollout_torch import (  # noqa: E402
    TransformerPhysicalRolloutTorch,
    TransformerRolloutConfig,
    theta_amplitude_loss,
    theta_smoothness_loss,
)

# ============================================================
# 默认训练 / 验证 / 测试载荷文件
# ============================================================
#
# 日常使用时，优先只改这里，不需要在命令行里写一堆路径。
#
# 当前目录约定：
#   data/load/train/train_complex_case_1.dat ... train_complex_case_8.dat
#   data/load/val/val_complex_case_1.dat
#   data/load/test/test_complex_case_1.dat
#
# 如果后续增加训练/验证/测试工况，只需要改：
#   DEFAULT_N_TRAIN_CASES
#   DEFAULT_N_VALID_CASES
#   DEFAULT_N_TEST_CASES

LOAD_ROOT = PROJECT_ROOT / "data" / "load"

TRAIN_LOAD_DIR = LOAD_ROOT / "train"
VALID_LOAD_DIR = LOAD_ROOT / "val"
TEST_LOAD_DIR = LOAD_ROOT / "test"

DEFAULT_TRAIN_LOAD_PREFIX = "train_complex_case_"
DEFAULT_VALID_LOAD_PREFIX = "val_complex_case_"
DEFAULT_TEST_LOAD_PREFIX = "test_complex_case_"

DEFAULT_LOAD_SUFFIX = ".dat"
DEFAULT_LOAD_CASE_START_INDEX = 1

DEFAULT_N_TRAIN_CASES = 8
DEFAULT_N_VALID_CASES = 2
DEFAULT_N_TEST_CASES = 1


def build_default_indexed_load_files(
        *,
        load_dir: str | Path,
        prefix: str,
        n_cases: int,
        suffix: str = DEFAULT_LOAD_SUFFIX,
        start_index: int = DEFAULT_LOAD_CASE_START_INDEX,
) -> list[str]:
    """
    根据目录、文件名前缀和数量自动生成载荷文件列表。

    Example:
        load_dir = data/load/train
        prefix   = train_complex_case_
        n_cases  = 8

    Output:
        data/load/train/train_complex_case_1.dat
        ...
        data/load/train/train_complex_case_8.dat
    """
    load_dir = Path(load_dir)
    n_cases = int(n_cases)
    start_index = int(start_index)

    if n_cases < 0:
        raise ValueError(f"n_cases must be non-negative, got {n_cases}.")
    if start_index <= 0:
        raise ValueError(f"start_index must be positive, got {start_index}.")

    return [
        str(load_dir / f"{prefix}{idx}{suffix}")
        for idx in range(start_index, start_index + n_cases)
    ]


TRAIN_LOAD_FILES = build_default_indexed_load_files(
    load_dir=TRAIN_LOAD_DIR,
    prefix=DEFAULT_TRAIN_LOAD_PREFIX,
    n_cases=DEFAULT_N_TRAIN_CASES,
)

VALID_LOAD_FILES = build_default_indexed_load_files(
    load_dir=VALID_LOAD_DIR,
    prefix=DEFAULT_VALID_LOAD_PREFIX,
    n_cases=DEFAULT_N_VALID_CASES,
)

TEST_LOAD_FILES = build_default_indexed_load_files(
    load_dir=TEST_LOAD_DIR,
    prefix=DEFAULT_TEST_LOAD_PREFIX,
    n_cases=DEFAULT_N_TEST_CASES,
)


# ============================================================
# Config
# ============================================================

@dataclass
class TransformerPhysicalTrainConfig:
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

    training_case_dir: str = str(
        PROJECT_ROOT
        / "results"
        / "transformer"
        / "physical_training_cases_new_loads_static_kappa_y"
    )

    output_dir: str = str(PROJECT_ROOT / "results" / "student" / "train_transformer_physical_params_torch")

    # ------------------------------------------------------------
    # 默认 train / val / test 载荷文件配置
    # ------------------------------------------------------------
    # 日常建议直接在脚本顶部 DEFAULT_N_TRAIN_CASES 等常量处改数量；
    # 这里保存到 cfg 里，方便 checkpoint / yaml 记录实验配置。
    train_load_dir: str = str(TRAIN_LOAD_DIR)
    valid_load_dir: str = str(VALID_LOAD_DIR)
    test_load_dir: str = str(TEST_LOAD_DIR)

    train_load_prefix: str = DEFAULT_TRAIN_LOAD_PREFIX
    valid_load_prefix: str = DEFAULT_VALID_LOAD_PREFIX
    test_load_prefix: str = DEFAULT_TEST_LOAD_PREFIX

    load_suffix: str = DEFAULT_LOAD_SUFFIX
    load_case_start_index: int = DEFAULT_LOAD_CASE_START_INDEX
    n_train_cases: int = DEFAULT_N_TRAIN_CASES
    n_valid_cases: int = DEFAULT_N_VALID_CASES
    n_test_cases: int = DEFAULT_N_TEST_CASES

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
    beta_damp_template_gain_x: float = 1.0
    beta_damp_template_gain_y: float = 1.0

    use_base_initial_twist_phi: bool = True
    base_phi_twist_column: str = "initial_twist_deg"
    base_phi_sign: float = -1.0
    rotate_mass: bool = False

    # static corrected student
    kappa_y_static_scale: float = 0.952
    kappa_y_scale_mode: str = "y_bending"

    # 当前第一步先只训练 alpha_x：x-bending 主方向动态刚度残差。
    # alpha_xy 保留为后续联合训练参数，但当前默认关闭。
    enabled_params: str = "alpha_x"

    # 三个输入分支开关
    use_response_branch: bool = True
    use_load_branch: bool = False
    use_geometry_branch: bool = False
    condition_dynamic_branches_on_geometry: bool = False

    # Load branch 内部的在线外力频域/脉冲特征。
    # 开启后，训练 case loading 阶段会从 F_raw 自动计算 causal 局部特征；
    # F_raw 本身仍作为真实物理外力进入 Newmark/MCK core，不会被拼接或改写。
    use_load_spectral_features: bool = False
    load_spectral_window_size: Optional[int] = None
    load_spectral_feature_dim: Optional[int] = None
    load_spectral_freq_min: float = 0.05
    load_spectral_freq_max: float = 5.0
    load_spectral_bands: str = "0.05-0.5,0.5-1.5,1.5-5.0"
    load_spectral_observations: str = "tip,last5,mean"
    load_spectral_last_k: int = 5
    load_spectral_active_rel_threshold: float = 1.0e-3
    load_spectral_active_abs_threshold: float = 1.0e-12
    load_spectral_normalize: bool = True

    # Result-preserving acceleration caches.
    # F_spectral 只依赖 F_raw 和频谱配置；teacher alignment cache 只依赖 teacher 响应和 loss 配置。
    use_load_spectral_disk_cache: bool = True
    load_spectral_cache_dir: Optional[str] = None
    force_recompute_load_spectral_cache: bool = False
    use_cached_alignment_loss: bool = True

    # Newmark/core acceleration. Default keeps torch.linalg.solve for numerical comparability.
    fast_core_precompute_newmark: bool = True
    linear_solve_mode: str = "solve"

    # Transformer 网络参数
    d_model: int = 64
    n_spatial_heads: int = 4
    n_temporal_heads: int = 4
    n_temporal_layers: int = 2
    dropout: float = 0.0

    # None = 原来的 full-history causal attention
    # W    = 每个 theta_t 只看最近 W 个 static response/load 时间步
    temporal_window_size: Optional[int] = None

    # training case 准备
    prepare_cases: bool = False
    rebuild_cases: bool = False
    remove_initial_offset: bool = True

    # 如果调试很慢，可以先用前 N 个时间步。
    # None 表示使用完整时序。
    max_steps_per_case: Optional[int] = None

    # 训练结束后自动评估 best checkpoint。
    # 注意：test 只用于最终报告，不参与 best checkpoint 选择。
    run_test_after_training: bool = False
    test_load_files: str = ",".join(TEST_LOAD_FILES)
    test_max_steps: int = 500
    test_output_dir: Optional[str] = None
    test_case_name_prefix: str = "post_train_eval"

    # 训练参数
    epochs: int = 50
    lr: float = 1.0e-4
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    encoder_dtype: str = "float32"
    core_dtype: str = "float64"

    # runtime acceleration / I/O cache
    cache_npz_in_ram: bool = False
    allow_tf32_encoder: bool = False
    matmul_precision: str = "high"
    profile_train_timing: bool = False
    profile_timing_sync_cuda: bool = False

    # response-domain loss 权重。
    # alpha_x 第一阶段以 x/y 相位与频率对齐为主，response MSE 只作为弱约束。
    w_y: float = 0.05
    w_x: float = 0.05

    # frequency-domain loss 权重。
    w_spec_x: float = 1.0
    w_spec_y: float = 1.0
    w_peak_x: float = 0.2
    w_peak_y: float = 0.2
    freq_min: float = 0.05
    freq_max: float = 5.0
    freq_peak_temperature: float = 0.02

    # peak-time / local-lag alignment loss.
    # 这些项专门针对复杂工况后期峰值/谷值错位与局部相位滞后。
    alignment_observations: str = "tip,last5"
    alignment_last_k: int = 5

    w_peak_time_x: float = 0.02
    w_peak_time_y: float = 0.02
    peak_time_start: float = 0.0
    peak_time_end: Optional[float] = None
    peak_time_window_seconds: float = 0.35
    peak_time_temperature: float = 0.08
    peak_time_min_distance_seconds: float = 0.30
    peak_time_prominence_std: float = 0.15
    peak_time_max_events: int = 16

    w_lag_x: float = 0.05
    w_lag_y: float = 0.05
    lag_start: float = 0.0
    lag_end: Optional[float] = None
    lag_window_seconds: float = 2.56
    lag_stride_seconds: float = 1.28
    max_lag_seconds: float = 0.80
    lag_temperature: float = 0.05

    # Adaptive phase-window hard mining and complex spectrum phase loss.
    # Unlike peak_time_start/lag_start, this scans the full sequence by default
    # and lets high-score local phase-drift windows receive stronger loss.
    use_adaptive_phase_window_loss: bool = False
    phase_window_observations: str = "tip,last5"
    phase_window_last_k: int = 5
    phase_window_start: float = 0.0
    phase_window_end: Optional[float] = None
    phase_window_size_seconds: float = 1.92
    phase_window_stride_seconds: float = 0.64
    phase_window_top_k: int = 4
    phase_window_score_temperature: float = 0.25
    phase_window_gate_score_ref: float = 0.12
    phase_window_max_lag_seconds: float = 0.50
    phase_window_lag_temperature: float = 0.04
    phase_window_freq_min: float = 0.05
    phase_window_freq_max: Optional[float] = 5.0
    phase_window_amplitude_reference: float = 0.0
    phase_window_amplitude_reference_x: float = 0.0
    phase_window_amplitude_reference_y: float = 0.0
    phase_window_amplitude_weight: float = 0.0
    phase_window_amplitude_power: float = 1.0
    phase_window_amplitude_max_weight: float = 4.0
    phase_window_static_failure_weight: float = 0.0
    phase_window_static_failure_max_weight: float = 4.0
    w_adaptive_phase_x: float = 0.0
    w_adaptive_phase_y: float = 0.0
    w_complex_phase_x: float = 0.0
    w_complex_phase_y: float = 0.0
    w_complex_amp_guard_x: float = 0.0
    w_complex_amp_guard_y: float = 0.0
    w_phase_gate_align: float = 0.0
    use_phase_drift_rate_loss: bool = False
    w_phase_drift_lag_x: float = 0.0
    w_phase_drift_lag_y: float = 0.0
    w_phase_drift_rate_x: float = 0.0
    w_phase_drift_rate_y: float = 0.0
    phase_drift_observations: str = "tip,last5"
    phase_drift_last_k: int = 5
    phase_drift_start: float = 0.0
    phase_drift_end: Optional[float] = None
    phase_drift_window_seconds: float = 1.54
    phase_drift_stride_seconds: float = 0.32
    phase_drift_max_lag_seconds: float = 0.50
    phase_drift_lag_temperature: float = 0.04
    phase_drift_freq_min: float = 0.50
    phase_drift_freq_max: Optional[float] = 1.20
    phase_drift_high_power_threshold: float = 0.20
    phase_drift_high_power_temperature: float = 0.08
    phase_drift_amplitude_reference: float = 0.0
    phase_drift_amplitude_reference_x: float = 0.0
    phase_drift_amplitude_reference_y: float = 0.0
    phase_drift_amplitude_weight: float = 0.0
    phase_drift_amplitude_power: float = 1.0
    phase_drift_amplitude_max_weight: float = 4.0
    phase_drift_static_failure_weight: float = 0.0
    phase_drift_static_failure_max_weight: float = 4.0
    use_local_band_phase_loss: bool = False
    w_local_band_phase_x: float = 0.0
    w_local_band_phase_y: float = 0.0
    local_band_phase_observations: str = "tip,last5"
    local_band_phase_last_k: int = 5
    local_band_phase_start: float = 0.0
    local_band_phase_end: Optional[float] = None
    local_band_phase_window_seconds: float = 1.54
    local_band_phase_stride_seconds: float = 0.32
    local_band_phase_freq_min: float = 0.45
    local_band_phase_freq_max: Optional[float] = 1.50
    local_band_phase_high_power_threshold: float = 0.08
    local_band_phase_high_power_temperature: float = 0.04
    use_local_phase_corr_loss: bool = False
    w_local_phase_corr_x: float = 0.0
    w_local_phase_corr_y: float = 0.0
    local_phase_corr_observations: str = "tip,last5"
    local_phase_corr_last_k: int = 5
    local_phase_corr_start: float = 0.0
    local_phase_corr_end: Optional[float] = None
    local_phase_corr_window_seconds: float = 1.28
    local_phase_corr_stride_seconds: float = 0.32
    local_phase_corr_max_lag_seconds: float = 0.40
    local_phase_corr_lag_temperature: float = 0.04
    local_phase_corr_freq_min: float = 0.45
    local_phase_corr_freq_max: Optional[float] = 1.50
    local_phase_corr_base_weight: float = 1.0
    local_phase_corr_high_power_weight: float = 2.0
    local_phase_corr_high_power_threshold: float = 0.08
    local_phase_corr_high_power_temperature: float = 0.04
    local_phase_corr_static_failure_weight: float = 0.0
    local_phase_corr_static_failure_max_weight: float = 3.0
    local_phase_corr_corr_weight: float = 1.0
    local_phase_corr_corr_gap_weight: float = 1.0
    local_phase_corr_lag_weight: float = 0.25
    local_phase_corr_corr_gap_tol: float = 0.0
    use_local_phase_increment_loss: bool = False
    w_local_phase_absolute_x: float = 0.0
    w_local_phase_absolute_y: float = 0.0
    w_local_phase_increment_x: float = 0.0
    w_local_phase_increment_y: float = 0.0
    local_phase_increment_observations: str = "tip,last5"
    local_phase_increment_last_k: int = 5
    local_phase_increment_start: float = 0.0
    local_phase_increment_end: Optional[float] = None
    local_phase_increment_window_seconds: float = 1.54
    local_phase_increment_stride_seconds: float = 0.16
    local_phase_increment_freq_min: float = 0.45
    local_phase_increment_freq_max: Optional[float] = 1.50
    local_phase_increment_base_weight: float = 1.0
    local_phase_increment_high_power_weight: float = 2.0
    local_phase_increment_high_power_threshold: float = 0.08
    local_phase_increment_high_power_temperature: float = 0.04
    local_phase_increment_static_failure_weight: float = 0.0
    local_phase_increment_static_failure_max_weight: float = 3.0
    use_continuous_phase_lag_loss: bool = False
    w_continuous_phase_absolute_x: float = 0.0
    w_continuous_phase_absolute_y: float = 0.0
    w_continuous_phase_time_shift_x: float = 0.0
    w_continuous_phase_time_shift_y: float = 0.0
    continuous_phase_observations: str = "tip,last5"
    continuous_phase_last_k: int = 5
    continuous_phase_start: float = 0.0
    continuous_phase_end: Optional[float] = None
    continuous_phase_window_seconds: float = 1.28
    continuous_phase_stride_seconds: float = 0.16
    continuous_phase_freq_min: float = 0.45
    continuous_phase_freq_max: Optional[float] = 1.50
    continuous_phase_n_freq_bins: int = 64
    continuous_phase_frequency_temperature: float = 0.08
    continuous_phase_time_shift_scale_seconds: float = 0.02
    continuous_phase_base_weight: float = 0.0
    continuous_phase_high_power_weight: float = 3.0
    continuous_phase_high_power_threshold: float = 0.08
    continuous_phase_high_power_temperature: float = 0.04
    continuous_phase_static_failure_weight: float = 0.0
    continuous_phase_static_failure_max_weight: float = 3.0
    use_local_phase_slope_loss: bool = False
    w_local_phase_slope_x: float = 0.0
    w_local_phase_slope_y: float = 0.0
    local_phase_slope_observations: str = "tip,last5"
    local_phase_slope_last_k: int = 5
    local_phase_slope_start: float = 0.0
    local_phase_slope_end: Optional[float] = None
    local_phase_slope_window_seconds: float = 1.28
    local_phase_slope_stride_seconds: float = 0.16
    local_phase_slope_freq_min: float = 0.45
    local_phase_slope_freq_max: Optional[float] = 1.50
    local_phase_slope_n_freq_bins: int = 64
    local_phase_slope_frequency_temperature: float = 0.08
    local_phase_slope_time_shift_scale_seconds: float = 0.02
    local_phase_slope_base_weight: float = 0.0
    local_phase_slope_high_power_weight: float = 3.0
    local_phase_slope_high_power_threshold: float = 0.08
    local_phase_slope_high_power_temperature: float = 0.04
    local_phase_slope_static_failure_weight: float = 0.0
    local_phase_slope_static_failure_max_weight: float = 3.0
    use_slow_only_branch_diagnosis: bool = False
    w_slow_good_no_regression: float = 0.0
    w_slow_good_fast_suppress: float = 0.0
    w_slow_bad_phase: float = 0.0
    slow_good_response_ratio_limit: float = 1.02
    slow_good_corr_drop_tol: float = 0.01
    slow_good_amp_log_tol: float = 0.08
    slow_bad_weight_max: float = 3.0
    use_static_quality_gate_suppression: bool = False
    w_static_good_gate_l1: float = 0.0
    static_quality_observations: str = "tip,last5"
    static_quality_last_k: int = 5
    static_quality_start: float = 0.0
    static_quality_end: Optional[float] = None
    static_quality_window_seconds: float = 1.28
    static_quality_stride_seconds: float = 0.64
    static_quality_max_lag_seconds: float = 0.20
    static_quality_good_corr_threshold: float = 0.995
    static_quality_good_lag_seconds: float = 0.02
    static_quality_good_amp_log_tol: float = 0.08
    use_state_window_no_regression_guard: bool = False
    w_state_no_regression_response: float = 0.0
    w_state_no_regression_corr: float = 0.0
    w_state_no_regression_amp: float = 0.0
    state_no_regression_response_ratio_limit: float = 1.02
    state_no_regression_corr_drop_tol: float = 0.01
    state_no_regression_amp_log_tol: float = 0.08

    # Beta/amplitude training mode. In auto mode this is enabled only for
    # freeze-alpha beta runs, leaving alpha training behavior unchanged.
    beta_loss_mode: str = "auto"
    beta_amp_observations: str = "tip,last5,mean"
    beta_amp_last_k: int = 5
    beta_amp_start: float = 0.0
    beta_amp_end: Optional[float] = None
    beta_amp_window_seconds: float = 1.28
    beta_amp_stride_seconds: float = 0.32
    beta_amp_alpha_error_ref: float = 0.05
    beta_amp_alpha_error_max_weight: float = 4.0
    beta_amp_log_tol: float = 0.0
    beta_amp_improvement_margin: float = 0.0
    w_beta_amp_x: float = 0.20
    w_beta_amp_y: float = 0.20
    w_beta_amp_tip_y: float = 0.0
    w_beta_amp_last5_y: float = 0.20
    w_beta_amp_improvement: float = 0.0
    w_beta_damp_sign_x: float = 0.0
    w_beta_damp_sign_y: float = 0.0
    beta_damp_sign_min_alpha_error: float = 0.02
    w_beta_alpha_response_guard: float = 0.20
    w_beta_alpha_corr_guard: float = 0.20
    w_beta_alpha_amp_guard: float = 0.10
    beta_alpha_response_ratio_limit: float = 1.02
    beta_alpha_corr_drop_tol: float = 0.01
    beta_alpha_amp_worsen_tol: float = 0.02

    best_score_guard_weight: float = 1.0

    # Optional slow + phase-gated fast decomposition for alpha_x / alpha_xy.
    # False keeps old bounded theta head unchanged.
    use_phase_gated_decomposition: bool = False
    phase_slow_scale: float = 1.0
    phase_fast_scale: float = 0.5
    phase_gate_init_bias: float = -4.0
    phase_total_clip_scale: float = 1.0

    # Decomposition-specific regularization.
    # These are only active when use_phase_gated_decomposition=True.
    w_theta_slow_smooth: float = 2.0e-3
    w_theta_fast_amp: float = 1.0e-3
    w_theta_fast_smooth: float = 1.0e-4
    w_theta_fast_window_mean: float = 0.0
    w_theta_gated_fast_window_mean: float = 0.0
    theta_fast_window_mean_seconds: float = 1.54
    theta_fast_window_mean_stride_seconds: float = 0.32
    w_phase_gate_l1: float = 1.0e-3
    w_phase_gate_tv: float = 1.0e-3
    phase_gate_active_threshold: float = 0.2
    w_phase_gate_bootstrap: float = 0.0
    phase_gate_bootstrap_target: float = 0.0
    phase_gate_bootstrap_end_epoch: int = 0

    # Optional from-scratch curriculum. It leaves the model/core unchanged and
    # only ramps selected loss weights across epochs.
    use_loss_curriculum: bool = False
    curriculum_phase_start_epoch: int = 40
    curriculum_phase_full_epoch: int = 100
    curriculum_guard_start_epoch: int = 80
    curriculum_guard_full_epoch: int = 140
    curriculum_lag_start_scale: float = 0.25
    curriculum_phase_drift_start_scale: float = 0.0
    curriculum_adaptive_phase_start_scale: float = 0.0
    curriculum_gate_reg_start_scale: float = 0.05
    curriculum_static_good_gate_start_scale: float = 0.0
    curriculum_state_guard_start_scale: float = 0.0

    # best checkpoint selection: response / freq / mixed
    best_score_mode: str = "freq"
    best_start_epoch: int = 1

    # Hard constraint for best-checkpoint selection.
    # If enabled, an epoch can update best checkpoint only when
    # valid_x <= x_best_constraint_max.
    use_x_constraint_for_best: bool = False
    x_best_constraint_max: float = 1.0

    w_x_guard: float = 1.0
    x_guard_tol: float = 0.0
    w_theta_amp: float = 1.0e-3
    w_theta_smooth: float = 1.0e-3
    w_tip_y: float = 0.0
    w_last5_y: float = 0.0

    # checkpoint / early stop
    init_checkpoint: Optional[str] = None
    freeze_alpha_backbone_for_beta: bool = False
    use_valid_for_best: bool = True

    # validation frequency
    # 1 = old behavior: run validation every epoch
    # 5 = run validation at epoch 1, 5, 10, 15, ...
    valid_every: int = 5
    early_stop_patience: int = 30
    early_stop_min_delta: float = 1.0e-5
    min_epochs: int = 5

    lr_plateau_patience: int = 15
    lr_plateau_factor: float = 0.5
    min_lr: float = 1.0e-6

    print_every: int = 1
    save_every: int = 10
    seed: int = 1234


@dataclass
class TransformerTrainingCase:
    path: Path
    name: str
    time: torch.Tensor
    F_raw: torch.Tensor
    F_spectral: Optional[torch.Tensor]
    u_teacher: torch.Tensor
    u_static: torch.Tensor
    v_static: torch.Tensor
    a_static: torch.Tensor
    base_x_mse: float
    base_y_mse: float
    base_tip_y_mse: float
    base_last5_y_mse: float
    remove_initial_offset: bool
    loss_cache: Optional[dict[str, Any]] = None


# ============================================================
# Utility
# ============================================================

def _parse_file_list(value: Optional[str], fallback: list[str]) -> list[str]:
    if value is None:
        return list(fallback)

    value = value.strip()
    if len(value) == 0:
        return []

    return [item.strip() for item in value.split(",") if item.strip()]


def build_load_files_from_cfg(
        *,
        cfg: TransformerPhysicalTrainConfig,
        split: str,
) -> list[str]:
    """
    从 cfg 自动生成 train / valid / test load files。

    split:
        "train" / "valid" / "test"
    """
    if split == "train":
        return build_default_indexed_load_files(
            load_dir=cfg.train_load_dir,
            prefix=cfg.train_load_prefix,
            n_cases=cfg.n_train_cases,
            suffix=cfg.load_suffix,
            start_index=cfg.load_case_start_index,
        )

    if split == "valid":
        return build_default_indexed_load_files(
            load_dir=cfg.valid_load_dir,
            prefix=cfg.valid_load_prefix,
            n_cases=cfg.n_valid_cases,
            suffix=cfg.load_suffix,
            start_index=cfg.load_case_start_index,
        )

    if split == "test":
        return build_default_indexed_load_files(
            load_dir=cfg.test_load_dir,
            prefix=cfg.test_load_prefix,
            n_cases=cfg.n_test_cases,
            suffix=cfg.load_suffix,
            start_index=cfg.load_case_start_index,
        )

    raise ValueError(f"Unsupported split: {split}")


def _parse_path_list(value: Optional[str]) -> list[str]:
    if value is None:
        return []

    value = value.strip()
    if len(value) == 0:
        return []

    return [item.strip() for item in value.split(",") if item.strip()]


def _linear_curriculum_scale(
        *,
        epoch: int,
        start_epoch: int,
        full_epoch: int,
        start_scale: float,
) -> float:
    """Return a clipped linear scale from start_scale to 1.0."""
    start_epoch = int(start_epoch)
    full_epoch = int(full_epoch)
    start_scale = float(start_scale)
    if full_epoch <= start_epoch:
        return 1.0 if epoch >= start_epoch else start_scale
    if epoch <= start_epoch:
        return start_scale
    if epoch >= full_epoch:
        return 1.0
    frac = float(epoch - start_epoch) / float(full_epoch - start_epoch)
    return start_scale + frac * (1.0 - start_scale)


def _phase_gate_bootstrap_weight_for_epoch(
        cfg: TransformerPhysicalTrainConfig,
        epoch: int,
) -> float:
    """Return the optional early gate-bootstrap weight for this epoch."""
    weight = float(getattr(cfg, "w_phase_gate_bootstrap", 0.0))
    end_epoch = int(getattr(cfg, "phase_gate_bootstrap_end_epoch", 0))
    if end_epoch > 0 and int(epoch) > end_epoch:
        return 0.0
    return weight


def make_epoch_curriculum_cfg(
        cfg: TransformerPhysicalTrainConfig,
        epoch: int,
) -> tuple[TransformerPhysicalTrainConfig, dict[str, float]]:
    """
    Build an epoch-local config with scheduled loss weights.

    The schedule is intentionally loss-only: model parameters, registry, dynamic
    physical core, and final theta interface are untouched.
    """
    bootstrap_weight = _phase_gate_bootstrap_weight_for_epoch(cfg, epoch)
    if not bool(getattr(cfg, "use_loss_curriculum", False)):
        epoch_cfg = replace(cfg, w_phase_gate_bootstrap=bootstrap_weight)
        metrics = {
            "curriculum_enabled": 0.0,
            "curriculum_phase_scale": 1.0,
            "curriculum_adaptive_phase_scale": 1.0,
            "curriculum_lag_scale": 1.0,
            "curriculum_guard_scale": 1.0,
            "curriculum_static_good_gate_scale": 1.0,
            "curriculum_gate_reg_scale": 1.0,
            "effective_w_lag_x": float(cfg.w_lag_x),
            "effective_w_lag_y": float(cfg.w_lag_y),
            "effective_w_peak_time_x": float(cfg.w_peak_time_x),
            "effective_w_peak_time_y": float(cfg.w_peak_time_y),
            "effective_w_phase_drift_lag_x": float(cfg.w_phase_drift_lag_x),
            "effective_w_phase_drift_lag_y": float(cfg.w_phase_drift_lag_y),
            "effective_w_phase_drift_rate_x": float(cfg.w_phase_drift_rate_x),
            "effective_w_phase_drift_rate_y": float(cfg.w_phase_drift_rate_y),
            "effective_w_local_band_phase_x": float(cfg.w_local_band_phase_x),
            "effective_w_local_band_phase_y": float(cfg.w_local_band_phase_y),
            "effective_w_local_phase_corr_x": float(cfg.w_local_phase_corr_x),
            "effective_w_local_phase_corr_y": float(cfg.w_local_phase_corr_y),
            "effective_w_local_phase_increment_x": float(cfg.w_local_phase_increment_x),
            "effective_w_local_phase_increment_y": float(cfg.w_local_phase_increment_y),
            "effective_w_continuous_phase_time_shift_x": float(cfg.w_continuous_phase_time_shift_x),
            "effective_w_continuous_phase_time_shift_y": float(cfg.w_continuous_phase_time_shift_y),
            "effective_w_local_phase_slope_x": float(cfg.w_local_phase_slope_x),
            "effective_w_local_phase_slope_y": float(cfg.w_local_phase_slope_y),
            "effective_w_phase_gate_l1": float(cfg.w_phase_gate_l1),
            "effective_w_phase_gate_tv": float(cfg.w_phase_gate_tv),
            "effective_w_phase_gate_bootstrap": float(epoch_cfg.w_phase_gate_bootstrap),
            "effective_w_static_good_gate_l1": float(cfg.w_static_good_gate_l1),
            "effective_w_state_no_regression_response": float(cfg.w_state_no_regression_response),
            "effective_w_state_no_regression_corr": float(cfg.w_state_no_regression_corr),
            "effective_w_state_no_regression_amp": float(cfg.w_state_no_regression_amp),
        }
        return epoch_cfg, metrics

    phase_scale = _linear_curriculum_scale(
        epoch=epoch,
        start_epoch=int(cfg.curriculum_phase_start_epoch),
        full_epoch=int(cfg.curriculum_phase_full_epoch),
        start_scale=float(cfg.curriculum_phase_drift_start_scale),
    )
    adaptive_phase_scale = _linear_curriculum_scale(
        epoch=epoch,
        start_epoch=int(cfg.curriculum_phase_start_epoch),
        full_epoch=int(cfg.curriculum_phase_full_epoch),
        start_scale=float(cfg.curriculum_adaptive_phase_start_scale),
    )
    lag_scale = _linear_curriculum_scale(
        epoch=epoch,
        start_epoch=1,
        full_epoch=int(cfg.curriculum_phase_full_epoch),
        start_scale=float(cfg.curriculum_lag_start_scale),
    )
    guard_scale = _linear_curriculum_scale(
        epoch=epoch,
        start_epoch=int(cfg.curriculum_guard_start_epoch),
        full_epoch=int(cfg.curriculum_guard_full_epoch),
        start_scale=float(cfg.curriculum_state_guard_start_scale),
    )
    static_good_gate_scale = _linear_curriculum_scale(
        epoch=epoch,
        start_epoch=int(cfg.curriculum_guard_start_epoch),
        full_epoch=int(cfg.curriculum_guard_full_epoch),
        start_scale=float(cfg.curriculum_static_good_gate_start_scale),
    )
    gate_reg_scale = _linear_curriculum_scale(
        epoch=epoch,
        start_epoch=1,
        full_epoch=int(cfg.curriculum_guard_full_epoch),
        start_scale=float(cfg.curriculum_gate_reg_start_scale),
    )

    epoch_cfg = replace(
        cfg,
        w_peak_time_x=float(cfg.w_peak_time_x) * lag_scale,
        w_peak_time_y=float(cfg.w_peak_time_y) * lag_scale,
        w_lag_x=float(cfg.w_lag_x) * lag_scale,
        w_lag_y=float(cfg.w_lag_y) * lag_scale,
        w_adaptive_phase_x=float(cfg.w_adaptive_phase_x) * adaptive_phase_scale,
        w_adaptive_phase_y=float(cfg.w_adaptive_phase_y) * adaptive_phase_scale,
        w_complex_phase_x=float(cfg.w_complex_phase_x) * adaptive_phase_scale,
        w_complex_phase_y=float(cfg.w_complex_phase_y) * adaptive_phase_scale,
        w_complex_amp_guard_x=float(cfg.w_complex_amp_guard_x) * adaptive_phase_scale,
        w_complex_amp_guard_y=float(cfg.w_complex_amp_guard_y) * adaptive_phase_scale,
        w_phase_gate_align=float(cfg.w_phase_gate_align) * adaptive_phase_scale,
        w_phase_drift_lag_x=float(cfg.w_phase_drift_lag_x) * phase_scale,
        w_phase_drift_lag_y=float(cfg.w_phase_drift_lag_y) * phase_scale,
        w_phase_drift_rate_x=float(cfg.w_phase_drift_rate_x) * phase_scale,
        w_phase_drift_rate_y=float(cfg.w_phase_drift_rate_y) * phase_scale,
        w_local_band_phase_x=float(cfg.w_local_band_phase_x) * phase_scale,
        w_local_band_phase_y=float(cfg.w_local_band_phase_y) * phase_scale,
        w_local_phase_corr_x=float(cfg.w_local_phase_corr_x) * phase_scale,
        w_local_phase_corr_y=float(cfg.w_local_phase_corr_y) * phase_scale,
        w_local_phase_absolute_x=float(cfg.w_local_phase_absolute_x) * phase_scale,
        w_local_phase_absolute_y=float(cfg.w_local_phase_absolute_y) * phase_scale,
        w_local_phase_increment_x=float(cfg.w_local_phase_increment_x) * phase_scale,
        w_local_phase_increment_y=float(cfg.w_local_phase_increment_y) * phase_scale,
        w_continuous_phase_absolute_x=float(cfg.w_continuous_phase_absolute_x) * phase_scale,
        w_continuous_phase_absolute_y=float(cfg.w_continuous_phase_absolute_y) * phase_scale,
        w_continuous_phase_time_shift_x=float(cfg.w_continuous_phase_time_shift_x) * phase_scale,
        w_continuous_phase_time_shift_y=float(cfg.w_continuous_phase_time_shift_y) * phase_scale,
        w_local_phase_slope_x=float(cfg.w_local_phase_slope_x) * phase_scale,
        w_local_phase_slope_y=float(cfg.w_local_phase_slope_y) * phase_scale,
        w_phase_gate_l1=float(cfg.w_phase_gate_l1) * gate_reg_scale,
        w_phase_gate_tv=float(cfg.w_phase_gate_tv) * gate_reg_scale,
        w_phase_gate_bootstrap=bootstrap_weight,
        w_static_good_gate_l1=float(cfg.w_static_good_gate_l1) * static_good_gate_scale,
        w_state_no_regression_response=float(cfg.w_state_no_regression_response) * guard_scale,
        w_state_no_regression_corr=float(cfg.w_state_no_regression_corr) * guard_scale,
        w_state_no_regression_amp=float(cfg.w_state_no_regression_amp) * guard_scale,
    )
    metrics = {
        "curriculum_enabled": 1.0,
        "curriculum_phase_scale": float(phase_scale),
        "curriculum_adaptive_phase_scale": float(adaptive_phase_scale),
        "curriculum_lag_scale": float(lag_scale),
        "curriculum_guard_scale": float(guard_scale),
        "curriculum_static_good_gate_scale": float(static_good_gate_scale),
        "curriculum_gate_reg_scale": float(gate_reg_scale),
        "effective_w_lag_x": float(epoch_cfg.w_lag_x),
        "effective_w_lag_y": float(epoch_cfg.w_lag_y),
        "effective_w_peak_time_x": float(epoch_cfg.w_peak_time_x),
        "effective_w_peak_time_y": float(epoch_cfg.w_peak_time_y),
        "effective_w_phase_drift_lag_x": float(epoch_cfg.w_phase_drift_lag_x),
        "effective_w_phase_drift_lag_y": float(epoch_cfg.w_phase_drift_lag_y),
        "effective_w_phase_drift_rate_x": float(epoch_cfg.w_phase_drift_rate_x),
        "effective_w_phase_drift_rate_y": float(epoch_cfg.w_phase_drift_rate_y),
        "effective_w_local_band_phase_x": float(epoch_cfg.w_local_band_phase_x),
        "effective_w_local_band_phase_y": float(epoch_cfg.w_local_band_phase_y),
        "effective_w_local_phase_corr_x": float(epoch_cfg.w_local_phase_corr_x),
        "effective_w_local_phase_corr_y": float(epoch_cfg.w_local_phase_corr_y),
        "effective_w_local_phase_increment_x": float(epoch_cfg.w_local_phase_increment_x),
        "effective_w_local_phase_increment_y": float(epoch_cfg.w_local_phase_increment_y),
        "effective_w_continuous_phase_time_shift_x": float(epoch_cfg.w_continuous_phase_time_shift_x),
        "effective_w_continuous_phase_time_shift_y": float(epoch_cfg.w_continuous_phase_time_shift_y),
        "effective_w_local_phase_slope_x": float(epoch_cfg.w_local_phase_slope_x),
        "effective_w_local_phase_slope_y": float(epoch_cfg.w_local_phase_slope_y),
        "effective_w_phase_gate_l1": float(epoch_cfg.w_phase_gate_l1),
        "effective_w_phase_gate_tv": float(epoch_cfg.w_phase_gate_tv),
        "effective_w_phase_gate_bootstrap": float(epoch_cfg.w_phase_gate_bootstrap),
        "effective_w_static_good_gate_l1": float(epoch_cfg.w_static_good_gate_l1),
        "effective_w_state_no_regression_response": float(epoch_cfg.w_state_no_regression_response),
        "effective_w_state_no_regression_corr": float(epoch_cfg.w_state_no_regression_corr),
        "effective_w_state_no_regression_amp": float(epoch_cfg.w_state_no_regression_amp),
    }
    return epoch_cfg, metrics


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


def _assert_existing_file(path: str | Path, label: str) -> Path:
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {p}")
    return p


def _get_torch_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype: {name}")


def _component_indices(n_nodes: int, component: str) -> np.ndarray:
    offsets = {
        "x": 0,
        "y": 1,
        "z": 2,
        "rx": 3,
        "ry": 4,
        "rz": 5,
    }
    if component not in offsets:
        raise ValueError(f"Unsupported component: {component}")
    off = offsets[component]
    return np.asarray([i * 6 + off for i in range(n_nodes)], dtype=np.int64)


def _last_k_component_indices(n_nodes: int, component: str, last_k: int = 5) -> np.ndarray:
    idx = _component_indices(n_nodes, component)
    return idx[-int(last_k):]


def _tip_component_index(n_nodes: int, component: str) -> int:
    return int(_component_indices(n_nodes, component)[-1])


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
        raise ValueError("No positive eigenvalues found.")
    return np.sqrt(eigvals[valid])[:num_modes] / (2.0 * np.pi)


def _build_structural_damping_matrix(
        *,
        K: np.ndarray,
        zeta_structural: float,
        ref_freq_hz: Optional[float],
        natural_freqs_hz: np.ndarray,
) -> tuple[np.ndarray, Optional[float]]:
    if zeta_structural <= 0.0:
        return np.zeros_like(K), ref_freq_hz

    if ref_freq_hz is None:
        if natural_freqs_hz.size < 1:
            raise ValueError("Cannot infer ref_freq_hz from empty natural_freqs_hz.")
        ref_freq_hz = float(natural_freqs_hz[0])

    beta_damp = _structural_damping_scale(
        zeta_structural=float(zeta_structural),
        ref_freq_hz=float(ref_freq_hz),
    )
    C = beta_damp * K
    return np.asarray(C, dtype=np.float64), float(ref_freq_hz)


def _structural_damping_scale(
        *,
        zeta_structural: float,
        ref_freq_hz: Optional[float],
) -> float:
    if zeta_structural <= 0.0 or ref_freq_hz is None:
        return 0.0
    return 2.0 * float(zeta_structural) / (2.0 * np.pi * float(ref_freq_hz))


def _save_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(payload, f, sort_keys=False, allow_unicode=True)


def _save_history_csv(path: Path, history: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if len(history) == 0:
        return

    keys = list(history[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def _is_head_row_expansion_key(key: str) -> bool:
    return (
        key.startswith("head.")
        or ".head." in key
        or key.endswith(".head.max_abs")
        or key == "max_abs"
    )


def _can_partial_row_copy(key: str, source: torch.Tensor, target: torch.Tensor) -> bool:
    if not _is_head_row_expansion_key(key):
        return False
    if source.ndim == 0 or source.ndim != target.ndim:
        return False
    if int(source.shape[0]) > int(target.shape[0]):
        return False
    return all(int(source.shape[i]) == int(target.shape[i]) for i in range(1, source.ndim))


def _merge_partial_state_dict(
        module: torch.nn.Module,
        source_state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """
    Merge an alpha checkpoint into a possibly wider beta model.

    Exact-shape tensors are copied as usual. For the physical head, tensors that
    grew only in the output/parameter dimension copy their leading alpha rows and
    leave the new beta rows at the freshly initialized near-zero values.
    """
    target_state = module.state_dict()
    merged_state = {
        key: value.detach().clone()
        for key, value in target_state.items()
    }

    exact_loaded: list[str] = []
    partial_loaded: list[dict[str, Any]] = []
    skipped_shape: list[dict[str, Any]] = []
    unexpected: list[str] = []

    for key, source_value in source_state.items():
        if key not in target_state:
            unexpected.append(key)
            continue
        if not torch.is_tensor(source_value):
            skipped_shape.append({"key": key, "source_shape": None, "target_shape": None})
            continue

        target_value = target_state[key]
        source_tensor = source_value.detach().to(
            device=target_value.device,
            dtype=target_value.dtype,
        )

        if tuple(source_tensor.shape) == tuple(target_value.shape):
            merged_state[key] = source_tensor.clone()
            exact_loaded.append(key)
            continue

        if _can_partial_row_copy(key, source_tensor, target_value):
            out = target_value.detach().clone()
            slices = tuple(slice(0, int(dim)) for dim in source_tensor.shape)
            out[slices] = source_tensor
            merged_state[key] = out
            partial_loaded.append(
                {
                    "key": key,
                    "source_shape": list(source_tensor.shape),
                    "target_shape": list(target_value.shape),
                }
            )
            continue

        skipped_shape.append(
            {
                "key": key,
                "source_shape": list(source_tensor.shape),
                "target_shape": list(target_value.shape),
            }
        )

    load_result = module.load_state_dict(merged_state, strict=True)
    return {
        "exact_loaded": exact_loaded,
        "partial_loaded": partial_loaded,
        "skipped_shape": skipped_shape,
        "unexpected": unexpected,
        "missing_after_merge": list(getattr(load_result, "missing_keys", [])),
        "unexpected_after_merge": list(getattr(load_result, "unexpected_keys", [])),
    }


def _load_initial_checkpoint_partial(
        *,
        model: torch.nn.Module,
        checkpoint: Any,
) -> tuple[str, dict[str, Any]]:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return "model_state_dict", _merge_partial_state_dict(model, checkpoint["model_state_dict"])
    if isinstance(checkpoint, dict) and "encoder_state_dict" in checkpoint:
        return "encoder_state_dict", _merge_partial_state_dict(model.encoder, checkpoint["encoder_state_dict"])
    if isinstance(checkpoint, dict):
        return "raw_state_dict", _merge_partial_state_dict(model, checkpoint)
    raise TypeError(
        f"Unsupported checkpoint type {type(checkpoint)!r}; expected a state-dict-like mapping."
    )


def _beta_theta_indices(registry: Any) -> list[int]:
    indices: list[int] = []
    slices = registry.slices
    for name in registry.names:
        spec = registry.get_spec(name)
        if spec.target != "C":
            continue
        sl = slices[name]
        indices.extend(range(int(sl.start), int(sl.stop)))
    return indices


def _enabled_param_names_from_cfg(cfg: TransformerPhysicalTrainConfig) -> list[str]:
    return [
        item.strip()
        for item in str(getattr(cfg, "enabled_params", "")).replace(",", " ").split()
        if item.strip()
    ]


def _use_beta_alpha_reference_loss(cfg: TransformerPhysicalTrainConfig) -> bool:
    mode = str(getattr(cfg, "beta_loss_mode", "auto")).strip().lower()
    if mode in {"standard", "none", "off", "false", "0"}:
        return False
    if mode in {"alpha_relative", "alpha_relative_amp", "amplitude", "amp"}:
        return True
    if mode != "auto":
        raise ValueError(
            f"Unsupported beta_loss_mode={getattr(cfg, 'beta_loss_mode', None)!r}. "
            "Expected auto, standard, or alpha_relative_amp."
        )
    names = _enabled_param_names_from_cfg(cfg)
    return bool(getattr(cfg, "freeze_alpha_backbone_for_beta", False)) and any(
        name.startswith("beta_") for name in names
    )


def _alpha_reference_theta_sequence(
        theta: torch.Tensor,
        registry: Any,
) -> Optional[torch.Tensor]:
    beta_indices = _beta_theta_indices(registry)
    if not beta_indices:
        return None
    theta_ref = theta.detach().clone()
    theta_ref[..., beta_indices] = 0.0
    return theta_ref


def _mask_parameter_rows(param: torch.nn.Parameter, train_rows: list[int]) -> None:
    mask = torch.zeros_like(param.detach())
    row_index = torch.as_tensor(train_rows, dtype=torch.long, device=mask.device)
    if param.ndim == 1:
        mask.index_fill_(0, row_index, 1.0)
    elif param.ndim >= 2:
        mask.index_fill_(0, row_index, 1.0)
    else:
        raise ValueError("Cannot row-mask a scalar parameter.")

    param.register_hook(lambda grad, row_mask=mask: grad * row_mask.to(dtype=grad.dtype, device=grad.device))


def _configure_beta_head_only_training(
        *,
        model: torch.nn.Module,
        registry: Any,
) -> dict[str, Any]:
    """
    Freeze the alpha backbone and train only C-target beta rows of theta heads.

    This is intentionally conservative for the first amplitude run: alpha_x and
    alpha_xy rows stay bitwise untouched by gradients, while beta rows can learn
    damping residuals from the already-trained Transformer features.
    """
    beta_rows = _beta_theta_indices(registry)
    if not beta_rows:
        raise ValueError(
            "freeze_alpha_backbone_for_beta=True but registry has no C-target beta parameters."
        )

    for param in model.encoder.parameters():
        param.requires_grad_(False)

    head = model.encoder.head
    final_layers: list[torch.nn.Module] = []
    if hasattr(head, "slow_net") and hasattr(head, "fast_net"):
        final_layers.extend([head.slow_net[-1], head.fast_net[-1]])
    elif hasattr(head, "net"):
        final_layers.append(head.net[-1])
    else:
        raise TypeError(f"Unsupported physical head type for beta-only freezing: {type(head)!r}")

    masked_tensors: list[str] = []
    for layer_idx, layer in enumerate(final_layers):
        if not isinstance(layer, torch.nn.Linear):
            raise TypeError(f"Expected final theta head layer to be nn.Linear, got {type(layer)!r}.")
        for tensor_name in ("weight", "bias"):
            param = getattr(layer, tensor_name)
            param.requires_grad_(True)
            _mask_parameter_rows(param, beta_rows)
            masked_tensors.append(f"head_final_{layer_idx}.{tensor_name}")

    trainable_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    total_encoder_params = sum(p.numel() for p in model.encoder.parameters())
    return {
        "beta_rows": [int(i) for i in beta_rows],
        "masked_tensors": masked_tensors,
        "trainable_encoder_params": int(trainable_params),
        "total_encoder_params": int(total_encoder_params),
        "weight_decay_forced_to_zero": True,
    }


def _read_metrics_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"metrics.csv not found: {path}")

    rows: list[dict[str, Any]] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: dict[str, Any] = {}
            for k, v in row.items():
                if k == "model":
                    parsed[k] = v
                else:
                    try:
                        parsed[k] = float(v)
                    except Exception:
                        parsed[k] = v
            rows.append(parsed)
    return rows


def _print_post_train_test_summary(
        *,
        metrics_csv: Path,
        load_file: Path,
) -> None:
    rows = _read_metrics_csv(metrics_csv)

    print()
    print("[Post-train Test Summary]")
    print(f"  load_file = {load_file}")
    print(f"  metrics   = {metrics_csv}")
    print(
        "  "
        f"{'model':<24s} "
        f"{'full_x_mse':>14s} "
        f"{'full_y_mse':>14s} "
        f"{'x_ratio/static':>16s} "
        f"{'y_ratio/static':>16s}"
    )

    for r in rows:
        model = str(r.get("model", "unknown"))
        full_x = float(r.get("full_x_mse", float("nan")))
        full_y = float(r.get("full_y_mse", float("nan")))
        x_ratio = float(r.get("full_x_mse_ratio_to_static", float("nan")))
        y_ratio = float(r.get("full_y_mse_ratio_to_static", float("nan")))

        print(
            "  "
            f"{model:<24s} "
            f"{full_x:14.6e} "
            f"{full_y:14.6e} "
            f"{x_ratio:16.6f} "
            f"{y_ratio:16.6f}"
        )


def _make_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if torch.is_tensor(obj):
        return obj.detach().cpu().numpy().tolist()

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, Path):
        return str(obj)

    return obj



# ============================================================
# Online load spectral features
# ============================================================

def _infer_load_spectral_window_size(cfg: TransformerPhysicalTrainConfig) -> Optional[int]:
    if cfg.load_spectral_window_size is not None:
        return int(cfg.load_spectral_window_size)
    if cfg.temporal_window_size is not None:
        return int(cfg.temporal_window_size)
    return None


def _default_load_spectral_cache_dir(cfg: TransformerPhysicalTrainConfig) -> Path:
    if cfg.load_spectral_cache_dir:
        return Path(cfg.load_spectral_cache_dir).expanduser().resolve()
    return (Path(cfg.training_case_dir).expanduser().resolve() / "_load_spectral_cache")


def _load_spectral_cache_path(
        *,
        case_path: Path,
        F_raw: np.ndarray,
        cfg: TransformerPhysicalTrainConfig,
) -> Path:
    """Build a stable cache path for causal load spectral features.

    The cache key includes case file metadata, F_raw shape and all spectral
    feature parameters. If a case is rebuilt, mtime/size changes and a new cache
    file is used automatically.
    """
    stat = case_path.stat()
    payload = {
        "case_path": str(case_path.resolve()),
        "case_size": int(stat.st_size),
        "case_mtime_ns": int(stat.st_mtime_ns),
        "F_shape": tuple(int(x) for x in F_raw.shape),
        "F_dtype": str(F_raw.dtype),
        "dt": float(cfg.dt),
        "window_size": _infer_load_spectral_window_size(cfg),
        "freq_min": float(cfg.load_spectral_freq_min),
        "freq_max": float(cfg.load_spectral_freq_max),
        "bands": str(cfg.load_spectral_bands),
        "observations": str(cfg.load_spectral_observations),
        "last_k": int(cfg.load_spectral_last_k),
        "active_rel_threshold": float(cfg.load_spectral_active_rel_threshold),
        "active_abs_threshold": float(cfg.load_spectral_active_abs_threshold),
        "feature_version": 1,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()[:20]
    safe_stem = case_path.stem.replace("/", "_").replace("\\", "_")
    return _default_load_spectral_cache_dir(cfg) / f"{safe_stem}.load_spectral.{digest}.pt"


def _compute_case_load_spectral_features(
        *,
        F_raw: np.ndarray,
        case_path: Optional[Path],
        cfg: TransformerPhysicalTrainConfig,
        dtype: torch.dtype,
        device: torch.device,
) -> torch.Tensor:
    """
    从当前 training case 的 F_raw 直接计算 causal load spectral features。

    注意：这里不从 .dat 文件重新读取，而是使用已经和 teacher/static response 对齐并截断后的 F_raw，
    避免 .dat 与 .npz 之间的长度/offset 不一致。
    """
    n_dofs = int(F_raw.shape[1])
    if n_dofs % 6 != 0:
        raise ValueError(f"n_dofs must be divisible by 6, got {n_dofs}")
    n_nodes = n_dofs // 6

    cache_path: Optional[Path] = None
    if bool(cfg.use_load_spectral_disk_cache) and case_path is not None:
        cache_path = _load_spectral_cache_path(case_path=case_path, F_raw=F_raw, cfg=cfg)
        if cache_path.exists() and not bool(cfg.force_recompute_load_spectral_cache):
            cached = torch.load(cache_path, map_location="cpu")
            if isinstance(cached, dict) and "F_spectral" in cached:
                F_cached = cached["F_spectral"]
            else:
                F_cached = cached
            return torch.as_tensor(F_cached, dtype=dtype, device=device)

    # Compute on the target device to preserve the old numerical path as much as possible.
    # The saved cache is stored on CPU and then moved back to the requested device/dtype.
    F_tensor = torch.as_tensor(F_raw, dtype=torch.float32, device=device)
    F_spec = compute_causal_load_spectral_features_from_force(
        F=F_tensor,
        n_nodes=n_nodes,
        dof_per_node=6,
        window_size=_infer_load_spectral_window_size(cfg),
        dt=float(cfg.dt),
        freq_min=float(cfg.load_spectral_freq_min),
        freq_max=float(cfg.load_spectral_freq_max),
        bands=str(cfg.load_spectral_bands),
        observations=str(cfg.load_spectral_observations),
        last_k=int(cfg.load_spectral_last_k),
        active_rel_threshold=float(cfg.load_spectral_active_rel_threshold),
        active_abs_threshold=float(cfg.load_spectral_active_abs_threshold),
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"F_spectral": F_spec.detach().to(device="cpu", dtype=torch.float32)}, cache_path)

    return F_spec.to(dtype=dtype, device=device)


def fit_load_spectral_normalization(
        *,
        train_cases: list[TransformerTrainingCase],
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Fit train-set mean/std for raw load spectral features."""
    feats = [c.F_spectral for c in train_cases if c.F_spectral is not None]
    if len(feats) == 0:
        return None, None
    cat = torch.cat([x.reshape(-1, x.shape[-1]).detach().to(dtype=torch.float32) for x in feats], dim=0)
    mean = torch.nanmean(cat, dim=0)
    centered = torch.nan_to_num(cat - mean.view(1, -1), nan=0.0, posinf=0.0, neginf=0.0)
    std = torch.sqrt(torch.mean(centered ** 2, dim=0))
    std = torch.clamp(std, min=1.0e-6)
    return mean, std

# ============================================================
# Training case loading
# ============================================================

def load_transformer_training_case(
        path: str | Path,
        *,
        dtype: torch.dtype,
        device: torch.device,
        max_steps: Optional[int] = None,
        cfg: Optional[TransformerPhysicalTrainConfig] = None,
) -> TransformerTrainingCase:
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Training case not found: {path}")

    data = fast_np_load(path)

    time = np.asarray(data["time"], dtype=np.float64)
    F_raw = np.asarray(data["F_raw"], dtype=np.float64)
    u_teacher = np.asarray(data["u_teacher"], dtype=np.float64)
    u_base = np.asarray(data["u_base"], dtype=np.float64)
    v_base = np.asarray(data["v_base"], dtype=np.float64)
    a_base = np.asarray(data["a_base"], dtype=np.float64)

    if not (F_raw.shape == u_teacher.shape == u_base.shape == v_base.shape == a_base.shape):
        raise ValueError(
            f"Case arrays shape mismatch in {path}: "
            f"F={F_raw.shape}, teacher={u_teacher.shape}, "
            f"u_base={u_base.shape}, v_base={v_base.shape}, a_base={a_base.shape}"
        )

    if F_raw.ndim != 2:
        raise ValueError(f"Expected 2D arrays [T,D], got F_raw.ndim={F_raw.ndim}")

    if max_steps is not None:
        max_steps = int(max_steps)
        if max_steps < 2:
            raise ValueError(f"max_steps must be >=2, got {max_steps}")
        time = time[:max_steps]
        F_raw = F_raw[:max_steps]
        u_teacher = u_teacher[:max_steps]
        u_base = u_base[:max_steps]
        v_base = v_base[:max_steps]
        a_base = a_base[:max_steps]

    n_dofs = F_raw.shape[1]
    if n_dofs % 6 != 0:
        raise ValueError(f"n_dofs must be divisible by 6, got {n_dofs}")

    n_nodes = n_dofs // 6
    y_idx = _component_indices(n_nodes, "y")
    tip_y_idx = _tip_component_index(n_nodes, "y")
    last5_y_idx = _last_k_component_indices(n_nodes, "y", last_k=5)

    # 优先使用 cache 里的 baseline metric；如果是 max_steps 截断，则重新计算局部 baseline metric。
    if max_steps is None and "full_x_mse" in data and "full_y_mse" in data:
        base_x_mse = float(np.asarray(data["full_x_mse"]).reshape(-1)[0])
        base_y_mse = float(np.asarray(data["full_y_mse"]).reshape(-1)[0])
        base_tip_y_mse = float(np.asarray(data["tip_y_mse"]).reshape(-1)[0])
        base_last5_y_mse = float(np.asarray(data["last5_y_mse"]).reshape(-1)[0])
    else:
        diff_base = u_base - u_teacher
        x_idx = _component_indices(n_nodes, "x")
        base_x_mse = float(np.mean(diff_base[:, x_idx] ** 2))
        base_y_mse = float(np.mean(diff_base[:, y_idx] ** 2))
        base_tip_y_mse = float(np.mean(diff_base[:, tip_y_idx] ** 2))
        base_last5_y_mse = float(np.mean(diff_base[:, last5_y_idx] ** 2))

    remove_initial_offset = bool(
        int(np.asarray(data.get("remove_initial_offset", np.asarray([1]))).reshape(-1)[0])
    )

    F_spectral = None
    if cfg is not None and bool(cfg.use_load_spectral_features):
        F_spectral = _compute_case_load_spectral_features(
            F_raw=F_raw,
            case_path=path,
            cfg=cfg,
            dtype=dtype,
            device=device,
        )

    return TransformerTrainingCase(
        path=path,
        name=path.stem,
        time=torch.as_tensor(time, dtype=dtype, device=device),
        F_raw=torch.as_tensor(F_raw, dtype=dtype, device=device),
        F_spectral=F_spectral,
        u_teacher=torch.as_tensor(u_teacher, dtype=dtype, device=device),
        u_static=torch.as_tensor(u_base, dtype=dtype, device=device),
        v_static=torch.as_tensor(v_base, dtype=dtype, device=device),
        a_static=torch.as_tensor(a_base, dtype=dtype, device=device),
        base_x_mse=base_x_mse,
        base_y_mse=base_y_mse,
        base_tip_y_mse=base_tip_y_mse,
        base_last5_y_mse=base_last5_y_mse,
        remove_initial_offset=remove_initial_offset,
    )


def load_cases(
        paths: list[str | Path],
        *,
        dtype: torch.dtype,
        device: torch.device,
        max_steps: Optional[int],
        cfg: Optional[TransformerPhysicalTrainConfig] = None,
) -> list[TransformerTrainingCase]:
    return [
        load_transformer_training_case(
            p,
            dtype=dtype,
            device=device,
            max_steps=max_steps,
            cfg=cfg,
        )
        for p in paths
    ]


def build_case_alignment_loss_cache(
        *,
        case: TransformerTrainingCase,
        cfg: TransformerPhysicalTrainConfig,
        x_idx: torch.Tensor,
        y_idx: torch.Tensor,
) -> dict[str, Any]:
    """Precompute teacher-side fixed quantities for spectral/peak/lag losses."""
    teacher = case.u_teacher
    return {
        "freq_x": build_frequency_alignment_cache(
            teacher,
            dt=float(cfg.dt),
            dof_indices=x_idx,
            freq_min=float(cfg.freq_min),
            freq_max=float(cfg.freq_max),
            peak_temperature=float(cfg.freq_peak_temperature),
        ),
        "freq_y": build_frequency_alignment_cache(
            teacher,
            dt=float(cfg.dt),
            dof_indices=y_idx,
            freq_min=float(cfg.freq_min),
            freq_max=float(cfg.freq_max),
            peak_temperature=float(cfg.freq_peak_temperature),
        ),
        "align_x": build_peak_lag_alignment_cache(
            teacher,
            dt=float(cfg.dt),
            dof_indices=x_idx,
            observations=str(cfg.alignment_observations),
            last_k=int(cfg.alignment_last_k),
            peak_start_time=float(cfg.peak_time_start),
            peak_end_time=cfg.peak_time_end,
            peak_window_seconds=float(cfg.peak_time_window_seconds),
            peak_temperature=float(cfg.peak_time_temperature),
            peak_min_distance_seconds=float(cfg.peak_time_min_distance_seconds),
            peak_prominence_std=float(cfg.peak_time_prominence_std),
            peak_max_events=int(cfg.peak_time_max_events),
            lag_start_time=float(cfg.lag_start),
            lag_end_time=cfg.lag_end,
            lag_window_seconds=float(cfg.lag_window_seconds),
            lag_stride_seconds=float(cfg.lag_stride_seconds),
            max_lag_seconds=float(cfg.max_lag_seconds),
            lag_temperature=float(cfg.lag_temperature),
        ),
        "align_y": build_peak_lag_alignment_cache(
            teacher,
            dt=float(cfg.dt),
            dof_indices=y_idx,
            observations=str(cfg.alignment_observations),
            last_k=int(cfg.alignment_last_k),
            peak_start_time=float(cfg.peak_time_start),
            peak_end_time=cfg.peak_time_end,
            peak_window_seconds=float(cfg.peak_time_window_seconds),
            peak_temperature=float(cfg.peak_time_temperature),
            peak_min_distance_seconds=float(cfg.peak_time_min_distance_seconds),
            peak_prominence_std=float(cfg.peak_time_prominence_std),
            peak_max_events=int(cfg.peak_time_max_events),
            lag_start_time=float(cfg.lag_start),
            lag_end_time=cfg.lag_end,
            lag_window_seconds=float(cfg.lag_window_seconds),
            lag_stride_seconds=float(cfg.lag_stride_seconds),
            max_lag_seconds=float(cfg.max_lag_seconds),
            lag_temperature=float(cfg.lag_temperature),
        ),
    }


def prime_alignment_loss_caches(
        *,
        cases: list[TransformerTrainingCase],
        cfg: TransformerPhysicalTrainConfig,
        x_idx: torch.Tensor,
        y_idx: torch.Tensor,
) -> None:
    if not bool(cfg.use_cached_alignment_loss):
        return
    for case in cases:
        case.loss_cache = build_case_alignment_loss_cache(
            case=case,
            cfg=cfg,
            x_idx=x_idx,
            y_idx=y_idx,
        )


# ============================================================
# Loss & Evaluation
# ============================================================


def _split_keyword_list(value: str) -> list[str]:
    return [
        item.strip().lower()
        for item in str(value).replace(";", ",").split(",")
        if item.strip()
    ]

PHASE_GATED_LOG_METRIC_KEYS = [
    "phase_reg_loss",
    "theta_slow_smooth",
    "theta_fast_amp",
    "theta_fast_smooth",
    "theta_fast_window_mean",
    "theta_gated_fast_window_mean",
    "phase_gate_l1",
    "phase_gate_tv",
    "phase_gate_bootstrap_loss",
    "phase_gate_bootstrap_deficit",
    "phase_gate_mean",
    "phase_gate_max",
    "phase_gate_active_ratio",
    "theta_fast_abs_max",
    "theta_fast_window_mean_abs_max",
    "theta_gated_fast_window_mean_abs_max",
    "theta_gated_fast_rms",
    "theta_gated_fast_abs_max",
]

PHASE_GATED_MAX_METRIC_KEYS = {
    "phase_gate_max",
    "theta_fast_abs_max",
    "theta_fast_window_mean_abs_max",
    "theta_gated_fast_window_mean_abs_max",
    "theta_gated_fast_abs_max",
}

ADAPTIVE_PHASE_LOG_METRIC_KEYS = [
    "adaptive_phase_loss",
    "complex_phase_loss",
    "complex_amp_guard_loss",
    "phase_gate_align_loss",
    "adaptive_x_lag_loss",
    "adaptive_y_lag_loss",
    "adaptive_x_complex_phase_loss",
    "adaptive_y_complex_phase_loss",
    "adaptive_x_complex_amp_guard_loss",
    "adaptive_y_complex_amp_guard_loss",
    "adaptive_x_gate_align_loss",
    "adaptive_y_gate_align_loss",
    "adaptive_x_score_mean",
    "adaptive_y_score_mean",
    "adaptive_x_score_max",
    "adaptive_y_score_max",
    "adaptive_x_amplitude_weight_mean",
    "adaptive_y_amplitude_weight_mean",
    "adaptive_x_amplitude_weight_max",
    "adaptive_y_amplitude_weight_max",
    "adaptive_x_static_failure_weight_mean",
    "adaptive_y_static_failure_weight_mean",
    "adaptive_x_static_failure_weight_max",
    "adaptive_y_static_failure_weight_max",
    "adaptive_x_best_abs_lag_s_mean",
    "adaptive_y_best_abs_lag_s_mean",
    "adaptive_x_best_corr_mean",
    "adaptive_y_best_corr_mean",
    "adaptive_x_corr0_mean",
    "adaptive_y_corr0_mean",
    "adaptive_x_selected_t_start_mean",
    "adaptive_y_selected_t_start_mean",
    "adaptive_x_selected_t_start_min",
    "adaptive_y_selected_t_start_min",
    "adaptive_x_selected_t_start_max",
    "adaptive_y_selected_t_start_max",
    "adaptive_x_selected_gate_mean",
    "adaptive_y_selected_gate_mean",
    "adaptive_x_gate_target_mean",
    "adaptive_y_gate_target_mean",
    "adaptive_x_n_windows",
    "adaptive_y_n_windows",
    "adaptive_x_n_selected_windows",
    "adaptive_y_n_selected_windows",
    "phase_drift_loss",
    "phase_drift_x_lag_loss",
    "phase_drift_y_lag_loss",
    "phase_drift_x_rate_loss",
    "phase_drift_y_rate_loss",
    "phase_drift_x_mean_abs_lag_s",
    "phase_drift_y_mean_abs_lag_s",
    "phase_drift_x_mean_abs_dlag_s",
    "phase_drift_y_mean_abs_dlag_s",
    "phase_drift_x_high_weight_mean",
    "phase_drift_y_high_weight_mean",
    "phase_drift_x_amplitude_weight_mean",
    "phase_drift_y_amplitude_weight_mean",
    "phase_drift_x_static_failure_weight_mean",
    "phase_drift_y_static_failure_weight_mean",
    "phase_drift_x_combined_weight_mean",
    "phase_drift_y_combined_weight_mean",
    "phase_drift_x_n_windows",
    "phase_drift_y_n_windows",
    "local_band_phase_loss",
    "local_band_phase_x_loss",
    "local_band_phase_y_loss",
    "local_band_phase_x_raw_phase_loss",
    "local_band_phase_y_raw_phase_loss",
    "local_band_phase_x_high_weight_mean",
    "local_band_phase_y_high_weight_mean",
    "local_band_phase_x_phase_cos_mean",
    "local_band_phase_y_phase_cos_mean",
    "local_band_phase_x_n_windows",
    "local_band_phase_y_n_windows",
    "local_phase_corr_loss",
    "local_phase_corr_x_loss",
    "local_phase_corr_y_loss",
    "local_phase_corr_x_corr_loss",
    "local_phase_corr_y_corr_loss",
    "local_phase_corr_x_corr_gap_loss",
    "local_phase_corr_y_corr_gap_loss",
    "local_phase_corr_x_lag_loss",
    "local_phase_corr_y_lag_loss",
    "local_phase_corr_x_corr0_mean",
    "local_phase_corr_y_corr0_mean",
    "local_phase_corr_x_best_corr_mean",
    "local_phase_corr_y_best_corr_mean",
    "local_phase_corr_x_corr_gap_mean",
    "local_phase_corr_y_corr_gap_mean",
    "local_phase_corr_x_mean_abs_lag_s",
    "local_phase_corr_y_mean_abs_lag_s",
    "local_phase_corr_x_high_weight_mean",
    "local_phase_corr_y_high_weight_mean",
    "local_phase_corr_x_static_failure_weight_mean",
    "local_phase_corr_y_static_failure_weight_mean",
    "local_phase_corr_x_combined_weight_mean",
    "local_phase_corr_y_combined_weight_mean",
    "local_phase_corr_x_n_windows",
    "local_phase_corr_y_n_windows",
    "local_phase_increment_loss",
    "local_phase_increment_x_absolute_loss",
    "local_phase_increment_y_absolute_loss",
    "local_phase_increment_x_increment_loss",
    "local_phase_increment_y_increment_loss",
    "local_phase_increment_x_phase_cos_mean",
    "local_phase_increment_y_phase_cos_mean",
    "local_phase_increment_x_increment_cos_mean",
    "local_phase_increment_y_increment_cos_mean",
    "local_phase_increment_x_high_weight_mean",
    "local_phase_increment_y_high_weight_mean",
    "local_phase_increment_x_static_failure_weight_mean",
    "local_phase_increment_y_static_failure_weight_mean",
    "local_phase_increment_x_combined_weight_mean",
    "local_phase_increment_y_combined_weight_mean",
    "local_phase_increment_x_target_freq_hz_mean",
    "local_phase_increment_y_target_freq_hz_mean",
    "local_phase_increment_x_n_windows",
    "local_phase_increment_y_n_windows",
    "local_phase_increment_x_n_increments",
    "local_phase_increment_y_n_increments",
    "continuous_phase_loss",
    "continuous_phase_x_absolute_loss",
    "continuous_phase_y_absolute_loss",
    "continuous_phase_x_time_shift_loss",
    "continuous_phase_y_time_shift_loss",
    "continuous_phase_x_phase_cos_mean",
    "continuous_phase_y_phase_cos_mean",
    "continuous_phase_x_equivalent_abs_lag_s_mean",
    "continuous_phase_y_equivalent_abs_lag_s_mean",
    "continuous_phase_x_high_weight_mean",
    "continuous_phase_y_high_weight_mean",
    "continuous_phase_x_static_failure_weight_mean",
    "continuous_phase_y_static_failure_weight_mean",
    "continuous_phase_x_combined_weight_mean",
    "continuous_phase_y_combined_weight_mean",
    "continuous_phase_x_target_freq_hz_mean",
    "continuous_phase_y_target_freq_hz_mean",
    "continuous_phase_x_n_windows",
    "continuous_phase_y_n_windows",
    "local_phase_slope_loss",
    "local_phase_slope_x_loss",
    "local_phase_slope_y_loss",
    "local_phase_slope_x_phase_cos_mean",
    "local_phase_slope_y_phase_cos_mean",
    "local_phase_slope_x_equivalent_abs_dlag_s_mean",
    "local_phase_slope_y_equivalent_abs_dlag_s_mean",
    "local_phase_slope_x_high_weight_mean",
    "local_phase_slope_y_high_weight_mean",
    "local_phase_slope_x_static_failure_weight_mean",
    "local_phase_slope_y_static_failure_weight_mean",
    "local_phase_slope_x_combined_weight_mean",
    "local_phase_slope_y_combined_weight_mean",
    "local_phase_slope_x_target_freq_hz_mean",
    "local_phase_slope_y_target_freq_hz_mean",
    "local_phase_slope_x_n_windows",
    "local_phase_slope_y_n_windows",
    "local_phase_slope_x_n_slopes",
    "local_phase_slope_y_n_slopes",
    "slow_only_diagnosis_loss",
    "slow_good_no_regression_loss",
    "slow_good_fast_suppress_loss",
    "slow_bad_phase_loss",
    "slow_good_window_ratio",
    "slow_bad_window_ratio",
    "slow_quality_score_mean",
    "slow_quality_score_max",
    "slow_good_gate_mean",
    "slow_bad_gate_mean",
    "slow_bad_phase_mean_abs_lag_s",
    "slow_only_n_windows",
    "static_quality_gate_loss",
    "static_good_gate_mean",
    "static_bad_gate_mean",
    "static_gate_selectivity_gap",
    "static_good_window_ratio",
    "static_bad_window_ratio",
    "static_quality_score_mean",
    "static_quality_score_max",
    "static_quality_gate_mean",
    "static_quality_n_windows",
]

ADAPTIVE_PHASE_MAX_METRIC_KEYS = {
    "adaptive_x_score_max",
    "adaptive_y_score_max",
    "adaptive_x_amplitude_weight_max",
    "adaptive_y_amplitude_weight_max",
    "adaptive_x_static_failure_weight_max",
    "adaptive_y_static_failure_weight_max",
    "slow_quality_score_max",
    "adaptive_x_selected_t_start_max",
    "adaptive_y_selected_t_start_max",
    "static_quality_score_max",
}

ADAPTIVE_PHASE_MIN_METRIC_KEYS = {
    "adaptive_x_selected_t_start_min",
    "adaptive_y_selected_t_start_min",
}

NO_REGRESSION_LOG_METRIC_KEYS = [
    "state_no_regression_guard_loss",
    "state_no_regression_response_guard_loss",
    "state_no_regression_corr_guard_loss",
    "state_no_regression_amp_guard_loss",
    "state_no_regression_good_window_ratio",
    "state_no_regression_response_excess",
    "state_no_regression_corr_drop_excess",
    "state_no_regression_amp_log_excess",
    "state_no_regression_n_windows",
]

BETA_ALPHA_LOG_METRIC_KEYS = [
    "beta_alpha_loss_mode_active",
    "beta_amp_loss",
    "beta_amp_x_loss",
    "beta_amp_y_loss",
    "beta_amp_tip_y_loss",
    "beta_amp_last5_y_loss",
    "beta_amp_direct_loss",
    "beta_amp_improvement_loss",
    "beta_amp_x_improvement_loss",
    "beta_amp_y_improvement_loss",
    "beta_amp_tip_y_improvement_loss",
    "beta_amp_last5_y_improvement_loss",
    "beta_damp_sign_loss",
    "beta_damp_x_sign_loss",
    "beta_damp_y_sign_loss",
    "beta_amp_pred_log_error_mean",
    "beta_amp_alpha_log_error_mean",
    "beta_amp_improvement_margin_mean",
    "beta_amp_weight_mean",
    "beta_amp_n_windows",
    "beta_alpha_guard_loss",
    "beta_alpha_response_guard_loss",
    "beta_alpha_corr_guard_loss",
    "beta_alpha_amp_guard_loss",
    "beta_alpha_response_excess",
    "beta_alpha_corr_drop_excess",
    "beta_alpha_amp_worsen_excess",
    "beta_alpha_guard_n_windows",
    "x_ratio_to_alpha",
    "y_ratio_to_alpha",
    "tip_y_ratio_to_alpha",
    "last5_y_ratio_to_alpha",
]

TIMING_LOG_METRIC_KEYS = [
    "timing_total_seconds",
    "timing_model_forward_seconds",
    "timing_encoder_seconds",
    "timing_core_prepare_seconds",
    "timing_newmark_loop_seconds",
    "timing_newmark_assemble_seconds",
    "timing_newmark_rhs_seconds",
    "timing_newmark_solve_seconds",
    "timing_newmark_update_seconds",
    "timing_state_stack_seconds",
    "timing_loss_seconds",
    "timing_backward_seconds",
    "timing_metric_accum_seconds",
    "timing_grad_clip_seconds",
    "timing_optimizer_step_seconds",
]


def _metric_to_float(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key)
    if value is None:
        return float("nan")
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def _sync_for_timing(cfg: TransformerPhysicalTrainConfig, device: torch.device | None = None) -> None:
    if not bool(getattr(cfg, "profile_timing_sync_cuda", False)):
        return
    if not torch.cuda.is_available():
        return
    if device is not None and device.type != "cuda":
        return
    torch.cuda.synchronize(device)


def _time_now(cfg: TransformerPhysicalTrainConfig, device: torch.device | None = None) -> float:
    _sync_for_timing(cfg, device)
    return time.perf_counter()


def _theta_window_mean_loss(
        theta_seq: torch.Tensor,
        *,
        dt: float,
        window_seconds: float,
        stride_seconds: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean-square and max-abs of sliding-window theta means."""
    if theta_seq.ndim == 2:
        theta_seq = theta_seq.unsqueeze(0)
    if theta_seq.ndim != 3:
        raise ValueError(f"theta_seq must be (T,P) or (B,T,P), got {tuple(theta_seq.shape)}")

    zero = theta_seq.sum() * 0.0
    T = int(theta_seq.shape[1])
    win_steps = max(1, int(round(float(window_seconds) / float(dt))))
    stride_steps = max(1, int(round(float(stride_seconds) / float(dt))))
    if T <= 0:
        return zero, zero.detach()
    if T < win_steps:
        mean_values = torch.mean(theta_seq, dim=1)
        return torch.mean(mean_values ** 2), torch.max(torch.abs(mean_values)).detach()

    means = []
    for lo in range(0, T - win_steps + 1, stride_steps):
        means.append(torch.mean(theta_seq[:, lo:lo + win_steps, :], dim=1))
    if not means:
        mean_values = torch.mean(theta_seq, dim=1)
    else:
        mean_values = torch.stack(means, dim=1)
    return torch.mean(mean_values ** 2), torch.max(torch.abs(mean_values)).detach()


def phase_gated_decomposition_regularization_loss(
        *,
        theta_aux: Optional[dict[str, torch.Tensor]],
        cfg: TransformerPhysicalTrainConfig,
        dtype: torch.dtype,
        device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Regularize slow + phase-gated fast decomposition without changing K_eff interface.

    The final physical rollout still uses theta_total. These auxiliary losses only shape the
    internal decomposition:
      - slow branch: stronger smoothness;
      - fast branch: bounded amplitude and weak smoothness;
      - phase gate: sparse and temporally stable.
    """
    zero = torch.zeros((), dtype=dtype, device=device)
    if not bool(getattr(cfg, "use_phase_gated_decomposition", False)) or theta_aux is None:
        return {
            "phase_reg_loss": zero,
            "theta_slow_smooth": zero,
            "theta_fast_amp": zero,
            "theta_fast_smooth": zero,
            "theta_fast_window_mean": zero,
            "theta_gated_fast_window_mean": zero,
            "phase_gate_l1": zero,
            "phase_gate_tv": zero,
            "phase_gate_bootstrap_loss": zero,
            "phase_gate_bootstrap_deficit": zero,
            "phase_gate_mean": zero,
            "phase_gate_max": zero,
            "phase_gate_active_ratio": zero,
            "theta_fast_abs_max": zero,
            "theta_fast_window_mean_abs_max": zero,
            "theta_gated_fast_window_mean_abs_max": zero,
            "theta_gated_fast_rms": zero,
            "theta_gated_fast_abs_max": zero,
        }

    theta_slow = theta_aux.get("theta_slow")
    theta_fast = theta_aux.get("theta_fast")
    theta_gated_fast = theta_aux.get("theta_gated_fast")
    g_phase = theta_aux.get("g_phase")

    if theta_slow is None or theta_fast is None or g_phase is None:
        raise RuntimeError(
            "use_phase_gated_decomposition=True but theta_aux misses one of "
            "theta_slow/theta_fast/g_phase."
        )

    theta_slow = theta_slow.to(dtype=dtype, device=device)
    theta_fast = theta_fast.to(dtype=dtype, device=device)
    g_phase = g_phase.to(dtype=dtype, device=device)
    if theta_gated_fast is None:
        theta_gated_fast = g_phase * theta_fast
    else:
        theta_gated_fast = theta_gated_fast.to(dtype=dtype, device=device)

    slow_smooth = theta_smoothness_loss(theta_slow)
    fast_amp = theta_amplitude_loss(theta_fast)
    fast_smooth = theta_smoothness_loss(theta_fast)
    fast_window_mean, fast_window_mean_abs_max = _theta_window_mean_loss(
        theta_fast,
        dt=float(cfg.dt),
        window_seconds=float(cfg.theta_fast_window_mean_seconds),
        stride_seconds=float(cfg.theta_fast_window_mean_stride_seconds),
    )
    gated_fast_window_mean, gated_fast_window_mean_abs_max = _theta_window_mean_loss(
        theta_gated_fast,
        dt=float(cfg.dt),
        window_seconds=float(cfg.theta_fast_window_mean_seconds),
        stride_seconds=float(cfg.theta_fast_window_mean_stride_seconds),
    )
    gate_l1 = torch.mean(g_phase)
    if g_phase.ndim >= 3 and g_phase.shape[1] >= 2:
        gate_tv = torch.mean(torch.abs(g_phase[:, 1:, :] - g_phase[:, :-1, :]))
    else:
        gate_tv = zero
    gate_bootstrap_target = min(max(float(cfg.phase_gate_bootstrap_target), 0.0), 1.0)
    if gate_bootstrap_target > 0.0:
        target = torch.as_tensor(gate_bootstrap_target, dtype=dtype, device=device)
        gate_bootstrap_deficit = torch.relu(target - gate_l1)
        gate_bootstrap_loss = gate_bootstrap_deficit ** 2
    else:
        gate_bootstrap_deficit = zero
        gate_bootstrap_loss = zero

    phase_reg_loss = (
        float(cfg.w_theta_slow_smooth) * slow_smooth
        + float(cfg.w_theta_fast_amp) * fast_amp
        + float(cfg.w_theta_fast_smooth) * fast_smooth
        + float(cfg.w_theta_fast_window_mean) * fast_window_mean
        + float(cfg.w_theta_gated_fast_window_mean) * gated_fast_window_mean
        + float(cfg.w_phase_gate_l1) * gate_l1
        + float(cfg.w_phase_gate_tv) * gate_tv
        + float(cfg.w_phase_gate_bootstrap) * gate_bootstrap_loss
    )

    return {
        "phase_reg_loss": phase_reg_loss,
        "theta_slow_smooth": slow_smooth,
        "theta_fast_amp": fast_amp,
        "theta_fast_smooth": fast_smooth,
        "theta_fast_window_mean": fast_window_mean,
        "theta_gated_fast_window_mean": gated_fast_window_mean,
        "phase_gate_l1": gate_l1,
        "phase_gate_tv": gate_tv,
        "phase_gate_bootstrap_loss": gate_bootstrap_loss,
        "phase_gate_bootstrap_deficit": gate_bootstrap_deficit,
        "phase_gate_mean": torch.mean(g_phase),
        "phase_gate_max": torch.max(g_phase),
        "phase_gate_active_ratio": torch.mean(
            (g_phase > float(cfg.phase_gate_active_threshold)).to(dtype=dtype)
        ),
        "theta_fast_abs_max": torch.max(torch.abs(theta_fast)),
        "theta_fast_window_mean_abs_max": fast_window_mean_abs_max,
        "theta_gated_fast_window_mean_abs_max": gated_fast_window_mean_abs_max,
        "theta_gated_fast_rms": torch.sqrt(torch.mean(theta_gated_fast ** 2)),
        "theta_gated_fast_abs_max": torch.max(torch.abs(theta_gated_fast)),
    }


def adaptive_phase_window_training_loss(
        *,
        pred: torch.Tensor,
        teacher: torch.Tensor,
        static: torch.Tensor,
        theta_aux: Optional[dict[str, torch.Tensor]],
        cfg: TransformerPhysicalTrainConfig,
        x_idx: torch.Tensor,
        y_idx: torch.Tensor,
) -> dict[str, torch.Tensor]:
    zero = torch.zeros((), dtype=pred.dtype, device=pred.device)
    empty = {
        "adaptive_phase_loss": zero,
        "complex_phase_loss": zero,
        "complex_amp_guard_loss": zero,
        "phase_gate_align_loss": zero,
    }
    for direction in ("x", "y"):
        for key in (
            "lag_loss",
            "complex_phase_loss",
            "complex_amp_guard_loss",
            "gate_align_loss",
            "score_mean",
            "score_max",
            "amplitude_weight_mean",
            "amplitude_weight_max",
            "static_failure_weight_mean",
            "static_failure_weight_max",
            "best_abs_lag_s_mean",
            "best_corr_mean",
            "corr0_mean",
            "selected_t_start_mean",
            "selected_t_start_min",
            "selected_t_start_max",
            "selected_gate_mean",
            "gate_target_mean",
            "n_windows",
            "n_selected_windows",
        ):
            empty[f"adaptive_{direction}_{key}"] = zero

    if not bool(getattr(cfg, "use_adaptive_phase_window_loss", False)):
        return empty

    gate = None
    if theta_aux is not None:
        gate = theta_aux.get("g_phase")

    common_kwargs = dict(
        dt=float(cfg.dt),
        observations=str(cfg.phase_window_observations),
        last_k=int(cfg.phase_window_last_k),
        gate=gate,
        start_time=float(cfg.phase_window_start),
        end_time=cfg.phase_window_end,
        window_seconds=float(cfg.phase_window_size_seconds),
        stride_seconds=float(cfg.phase_window_stride_seconds),
        top_k=int(cfg.phase_window_top_k),
        score_temperature=float(cfg.phase_window_score_temperature),
        gate_target_score_ref=float(cfg.phase_window_gate_score_ref),
        max_lag_seconds=float(cfg.phase_window_max_lag_seconds),
        lag_temperature=float(cfg.phase_window_lag_temperature),
        freq_min=float(cfg.phase_window_freq_min),
        freq_max=cfg.phase_window_freq_max,
        amplitude_weight=float(cfg.phase_window_amplitude_weight),
        amplitude_power=float(cfg.phase_window_amplitude_power),
        amplitude_max_weight=float(cfg.phase_window_amplitude_max_weight),
        static_failure_weight=float(cfg.phase_window_static_failure_weight),
        static_failure_max_weight=float(cfg.phase_window_static_failure_max_weight),
        static_failure_corr_threshold=float(cfg.static_quality_good_corr_threshold),
        static_failure_lag_seconds=float(cfg.static_quality_good_lag_seconds),
        static_failure_amp_log_tol=float(cfg.static_quality_good_amp_log_tol),
        static_failure_max_lag_seconds=float(cfg.static_quality_max_lag_seconds),
    )
    amp_ref_x = (
        float(cfg.phase_window_amplitude_reference_x)
        if float(cfg.phase_window_amplitude_reference_x) > 0.0
        else float(cfg.phase_window_amplitude_reference)
    )
    amp_ref_y = (
        float(cfg.phase_window_amplitude_reference_y)
        if float(cfg.phase_window_amplitude_reference_y) > 0.0
        else float(cfg.phase_window_amplitude_reference)
    )

    adaptive_x = adaptive_phase_window_loss(
        pred,
        teacher,
        dof_indices=x_idx,
        static_reference=static,
        amplitude_reference=amp_ref_x,
        **common_kwargs,
    )
    adaptive_y = adaptive_phase_window_loss(
        pred,
        teacher,
        dof_indices=y_idx,
        static_reference=static,
        amplitude_reference=amp_ref_y,
        **common_kwargs,
    )

    adaptive_phase_loss = (
        float(cfg.w_adaptive_phase_x) * adaptive_x["lag_loss"]
        + float(cfg.w_adaptive_phase_y) * adaptive_y["lag_loss"]
    )
    complex_phase_loss = (
        float(cfg.w_complex_phase_x) * adaptive_x["complex_phase_loss"]
        + float(cfg.w_complex_phase_y) * adaptive_y["complex_phase_loss"]
    )
    complex_amp_guard_loss = (
        float(cfg.w_complex_amp_guard_x) * adaptive_x["complex_amp_guard_loss"]
        + float(cfg.w_complex_amp_guard_y) * adaptive_y["complex_amp_guard_loss"]
    )
    phase_gate_align_loss = float(cfg.w_phase_gate_align) * 0.5 * (
        adaptive_x["gate_align_loss"] + adaptive_y["gate_align_loss"]
    )

    out: dict[str, torch.Tensor] = {
        "adaptive_phase_loss": adaptive_phase_loss,
        "complex_phase_loss": complex_phase_loss,
        "complex_amp_guard_loss": complex_amp_guard_loss,
        "phase_gate_align_loss": phase_gate_align_loss,
    }

    for prefix, payload in (("adaptive_x", adaptive_x), ("adaptive_y", adaptive_y)):
        for key in (
            "lag_loss",
            "complex_phase_loss",
            "complex_amp_guard_loss",
            "gate_align_loss",
            "score_mean",
            "score_max",
            "amplitude_weight_mean",
            "amplitude_weight_max",
            "static_failure_weight_mean",
            "static_failure_weight_max",
            "best_abs_lag_s_mean",
            "best_corr_mean",
            "corr0_mean",
            "selected_t_start_mean",
            "selected_t_start_min",
            "selected_t_start_max",
            "selected_gate_mean",
            "gate_target_mean",
            "n_windows",
            "n_selected_windows",
        ):
            out[f"{prefix}_{key}"] = payload[key]

    return out


def phase_drift_rate_training_loss(
        *,
        pred: torch.Tensor,
        teacher: torch.Tensor,
        static: torch.Tensor,
        cfg: TransformerPhysicalTrainConfig,
        x_idx: torch.Tensor,
        y_idx: torch.Tensor,
) -> dict[str, torch.Tensor]:
    zero = torch.zeros((), dtype=pred.dtype, device=pred.device)
    empty = {
        "phase_drift_loss": zero,
        "phase_drift_x_lag_loss": zero,
        "phase_drift_y_lag_loss": zero,
        "phase_drift_x_rate_loss": zero,
        "phase_drift_y_rate_loss": zero,
        "phase_drift_x_mean_abs_lag_s": zero.detach(),
        "phase_drift_y_mean_abs_lag_s": zero.detach(),
        "phase_drift_x_mean_abs_dlag_s": zero.detach(),
        "phase_drift_y_mean_abs_dlag_s": zero.detach(),
        "phase_drift_x_high_weight_mean": zero.detach(),
        "phase_drift_y_high_weight_mean": zero.detach(),
        "phase_drift_x_amplitude_weight_mean": zero.detach(),
        "phase_drift_y_amplitude_weight_mean": zero.detach(),
        "phase_drift_x_static_failure_weight_mean": zero.detach(),
        "phase_drift_y_static_failure_weight_mean": zero.detach(),
        "phase_drift_x_combined_weight_mean": zero.detach(),
        "phase_drift_y_combined_weight_mean": zero.detach(),
        "phase_drift_x_n_windows": zero.detach(),
        "phase_drift_y_n_windows": zero.detach(),
    }
    if not bool(getattr(cfg, "use_phase_drift_rate_loss", False)):
        return empty

    common_kwargs = dict(
        dt=float(cfg.dt),
        observations=str(cfg.phase_drift_observations),
        last_k=int(cfg.phase_drift_last_k),
        start_time=float(cfg.phase_drift_start),
        end_time=cfg.phase_drift_end,
        window_seconds=float(cfg.phase_drift_window_seconds),
        stride_seconds=float(cfg.phase_drift_stride_seconds),
        max_lag_seconds=float(cfg.phase_drift_max_lag_seconds),
        lag_temperature=float(cfg.phase_drift_lag_temperature),
        freq_min=float(cfg.phase_drift_freq_min),
        freq_max=cfg.phase_drift_freq_max,
        high_power_threshold=float(cfg.phase_drift_high_power_threshold),
        high_power_temperature=float(cfg.phase_drift_high_power_temperature),
        amplitude_weight=float(cfg.phase_drift_amplitude_weight),
        amplitude_power=float(cfg.phase_drift_amplitude_power),
        amplitude_max_weight=float(cfg.phase_drift_amplitude_max_weight),
        static_failure_weight=float(cfg.phase_drift_static_failure_weight),
        static_failure_max_weight=float(cfg.phase_drift_static_failure_max_weight),
        static_failure_corr_threshold=float(cfg.static_quality_good_corr_threshold),
        static_failure_lag_seconds=float(cfg.static_quality_good_lag_seconds),
        static_failure_amp_log_tol=float(cfg.static_quality_good_amp_log_tol),
        static_failure_max_lag_seconds=float(cfg.static_quality_max_lag_seconds),
    )
    amp_ref_x = (
        float(cfg.phase_drift_amplitude_reference_x)
        if float(cfg.phase_drift_amplitude_reference_x) > 0.0
        else float(cfg.phase_drift_amplitude_reference)
    )
    amp_ref_y = (
        float(cfg.phase_drift_amplitude_reference_y)
        if float(cfg.phase_drift_amplitude_reference_y) > 0.0
        else float(cfg.phase_drift_amplitude_reference)
    )
    drift_x = phase_drift_rate_loss(
        pred,
        teacher,
        dof_indices=x_idx,
        static_reference=static,
        amplitude_reference=amp_ref_x,
        **common_kwargs,
    )
    drift_y = phase_drift_rate_loss(
        pred,
        teacher,
        dof_indices=y_idx,
        static_reference=static,
        amplitude_reference=amp_ref_y,
        **common_kwargs,
    )
    loss = (
        float(cfg.w_phase_drift_lag_x) * drift_x["lag_loss"]
        + float(cfg.w_phase_drift_lag_y) * drift_y["lag_loss"]
        + float(cfg.w_phase_drift_rate_x) * drift_x["drift_loss"]
        + float(cfg.w_phase_drift_rate_y) * drift_y["drift_loss"]
    )
    return {
        "phase_drift_loss": loss,
        "phase_drift_x_lag_loss": drift_x["lag_loss"],
        "phase_drift_y_lag_loss": drift_y["lag_loss"],
        "phase_drift_x_rate_loss": drift_x["drift_loss"],
        "phase_drift_y_rate_loss": drift_y["drift_loss"],
        "phase_drift_x_mean_abs_lag_s": drift_x["mean_abs_lag_s"],
        "phase_drift_y_mean_abs_lag_s": drift_y["mean_abs_lag_s"],
        "phase_drift_x_mean_abs_dlag_s": drift_x["mean_abs_dlag_s"],
        "phase_drift_y_mean_abs_dlag_s": drift_y["mean_abs_dlag_s"],
        "phase_drift_x_high_weight_mean": drift_x["high_weight_mean"],
        "phase_drift_y_high_weight_mean": drift_y["high_weight_mean"],
        "phase_drift_x_amplitude_weight_mean": drift_x["amplitude_weight_mean"],
        "phase_drift_y_amplitude_weight_mean": drift_y["amplitude_weight_mean"],
        "phase_drift_x_static_failure_weight_mean": drift_x["static_failure_weight_mean"],
        "phase_drift_y_static_failure_weight_mean": drift_y["static_failure_weight_mean"],
        "phase_drift_x_combined_weight_mean": drift_x["combined_weight_mean"],
        "phase_drift_y_combined_weight_mean": drift_y["combined_weight_mean"],
        "phase_drift_x_n_windows": drift_x["n_windows"],
        "phase_drift_y_n_windows": drift_y["n_windows"],
    }


def local_band_phase_training_loss(
        *,
        pred: torch.Tensor,
        teacher: torch.Tensor,
        cfg: TransformerPhysicalTrainConfig,
        x_idx: torch.Tensor,
        y_idx: torch.Tensor,
) -> dict[str, torch.Tensor]:
    zero = torch.zeros((), dtype=pred.dtype, device=pred.device)
    empty = {
        "local_band_phase_loss": zero,
        "local_band_phase_x_loss": zero,
        "local_band_phase_y_loss": zero,
        "local_band_phase_x_raw_phase_loss": zero.detach(),
        "local_band_phase_y_raw_phase_loss": zero.detach(),
        "local_band_phase_x_high_weight_mean": zero.detach(),
        "local_band_phase_y_high_weight_mean": zero.detach(),
        "local_band_phase_x_phase_cos_mean": zero.detach(),
        "local_band_phase_y_phase_cos_mean": zero.detach(),
        "local_band_phase_x_n_windows": zero.detach(),
        "local_band_phase_y_n_windows": zero.detach(),
    }
    if not bool(getattr(cfg, "use_local_band_phase_loss", False)):
        return empty

    common_kwargs = dict(
        dt=float(cfg.dt),
        observations=str(cfg.local_band_phase_observations),
        last_k=int(cfg.local_band_phase_last_k),
        start_time=float(cfg.local_band_phase_start),
        end_time=cfg.local_band_phase_end,
        window_seconds=float(cfg.local_band_phase_window_seconds),
        stride_seconds=float(cfg.local_band_phase_stride_seconds),
        freq_min=float(cfg.local_band_phase_freq_min),
        freq_max=cfg.local_band_phase_freq_max,
        high_power_threshold=float(cfg.local_band_phase_high_power_threshold),
        high_power_temperature=float(cfg.local_band_phase_high_power_temperature),
    )
    phase_x = local_band_phase_loss(
        pred,
        teacher,
        dof_indices=x_idx,
        **common_kwargs,
    )
    phase_y = local_band_phase_loss(
        pred,
        teacher,
        dof_indices=y_idx,
        **common_kwargs,
    )
    loss = (
        float(cfg.w_local_band_phase_x) * phase_x["loss"]
        + float(cfg.w_local_band_phase_y) * phase_y["loss"]
    )
    return {
        "local_band_phase_loss": loss,
        "local_band_phase_x_loss": phase_x["loss"],
        "local_band_phase_y_loss": phase_y["loss"],
        "local_band_phase_x_raw_phase_loss": phase_x["raw_phase_loss"],
        "local_band_phase_y_raw_phase_loss": phase_y["raw_phase_loss"],
        "local_band_phase_x_high_weight_mean": phase_x["high_weight_mean"],
        "local_band_phase_y_high_weight_mean": phase_y["high_weight_mean"],
        "local_band_phase_x_phase_cos_mean": phase_x["phase_cos_mean"],
        "local_band_phase_y_phase_cos_mean": phase_y["phase_cos_mean"],
        "local_band_phase_x_n_windows": phase_x["n_windows"],
        "local_band_phase_y_n_windows": phase_y["n_windows"],
    }


def local_phase_correlation_training_loss(
        *,
        pred: torch.Tensor,
        teacher: torch.Tensor,
        static: torch.Tensor,
        cfg: TransformerPhysicalTrainConfig,
        x_idx: torch.Tensor,
        y_idx: torch.Tensor,
) -> dict[str, torch.Tensor]:
    zero = torch.zeros((), dtype=pred.dtype, device=pred.device)
    empty = {
        "local_phase_corr_loss": zero,
        "local_phase_corr_x_loss": zero,
        "local_phase_corr_y_loss": zero,
        "local_phase_corr_x_corr_loss": zero,
        "local_phase_corr_y_corr_loss": zero,
        "local_phase_corr_x_corr_gap_loss": zero,
        "local_phase_corr_y_corr_gap_loss": zero,
        "local_phase_corr_x_lag_loss": zero,
        "local_phase_corr_y_lag_loss": zero,
        "local_phase_corr_x_corr0_mean": zero.detach(),
        "local_phase_corr_y_corr0_mean": zero.detach(),
        "local_phase_corr_x_best_corr_mean": zero.detach(),
        "local_phase_corr_y_best_corr_mean": zero.detach(),
        "local_phase_corr_x_corr_gap_mean": zero.detach(),
        "local_phase_corr_y_corr_gap_mean": zero.detach(),
        "local_phase_corr_x_mean_abs_lag_s": zero.detach(),
        "local_phase_corr_y_mean_abs_lag_s": zero.detach(),
        "local_phase_corr_x_high_weight_mean": zero.detach(),
        "local_phase_corr_y_high_weight_mean": zero.detach(),
        "local_phase_corr_x_static_failure_weight_mean": zero.detach(),
        "local_phase_corr_y_static_failure_weight_mean": zero.detach(),
        "local_phase_corr_x_combined_weight_mean": zero.detach(),
        "local_phase_corr_y_combined_weight_mean": zero.detach(),
        "local_phase_corr_x_n_windows": zero.detach(),
        "local_phase_corr_y_n_windows": zero.detach(),
    }
    if not bool(getattr(cfg, "use_local_phase_corr_loss", False)):
        return empty

    common_kwargs = dict(
        dt=float(cfg.dt),
        observations=str(cfg.local_phase_corr_observations),
        last_k=int(cfg.local_phase_corr_last_k),
        start_time=float(cfg.local_phase_corr_start),
        end_time=cfg.local_phase_corr_end,
        window_seconds=float(cfg.local_phase_corr_window_seconds),
        stride_seconds=float(cfg.local_phase_corr_stride_seconds),
        max_lag_seconds=float(cfg.local_phase_corr_max_lag_seconds),
        lag_temperature=float(cfg.local_phase_corr_lag_temperature),
        freq_min=float(cfg.local_phase_corr_freq_min),
        freq_max=cfg.local_phase_corr_freq_max,
        base_weight=float(cfg.local_phase_corr_base_weight),
        high_power_weight=float(cfg.local_phase_corr_high_power_weight),
        high_power_threshold=float(cfg.local_phase_corr_high_power_threshold),
        high_power_temperature=float(cfg.local_phase_corr_high_power_temperature),
        static_failure_weight=float(cfg.local_phase_corr_static_failure_weight),
        static_failure_max_weight=float(cfg.local_phase_corr_static_failure_max_weight),
        static_failure_corr_threshold=float(cfg.static_quality_good_corr_threshold),
        static_failure_lag_seconds=float(cfg.static_quality_good_lag_seconds),
        static_failure_amp_log_tol=float(cfg.static_quality_good_amp_log_tol),
        static_failure_max_lag_seconds=float(cfg.static_quality_max_lag_seconds),
        corr_weight=float(cfg.local_phase_corr_corr_weight),
        corr_gap_weight=float(cfg.local_phase_corr_corr_gap_weight),
        lag_weight=float(cfg.local_phase_corr_lag_weight),
        corr_gap_tol=float(cfg.local_phase_corr_corr_gap_tol),
    )
    corr_x = local_phase_correlation_loss(
        pred,
        teacher,
        static_reference=static,
        dof_indices=x_idx,
        **common_kwargs,
    )
    corr_y = local_phase_correlation_loss(
        pred,
        teacher,
        static_reference=static,
        dof_indices=y_idx,
        **common_kwargs,
    )
    loss = (
        float(cfg.w_local_phase_corr_x) * corr_x["loss"]
        + float(cfg.w_local_phase_corr_y) * corr_y["loss"]
    )
    return {
        "local_phase_corr_loss": loss,
        "local_phase_corr_x_loss": corr_x["loss"],
        "local_phase_corr_y_loss": corr_y["loss"],
        "local_phase_corr_x_corr_loss": corr_x["corr_loss"],
        "local_phase_corr_y_corr_loss": corr_y["corr_loss"],
        "local_phase_corr_x_corr_gap_loss": corr_x["corr_gap_loss"],
        "local_phase_corr_y_corr_gap_loss": corr_y["corr_gap_loss"],
        "local_phase_corr_x_lag_loss": corr_x["lag_loss"],
        "local_phase_corr_y_lag_loss": corr_y["lag_loss"],
        "local_phase_corr_x_corr0_mean": corr_x["corr0_mean"],
        "local_phase_corr_y_corr0_mean": corr_y["corr0_mean"],
        "local_phase_corr_x_best_corr_mean": corr_x["best_corr_mean"],
        "local_phase_corr_y_best_corr_mean": corr_y["best_corr_mean"],
        "local_phase_corr_x_corr_gap_mean": corr_x["corr_gap_mean"],
        "local_phase_corr_y_corr_gap_mean": corr_y["corr_gap_mean"],
        "local_phase_corr_x_mean_abs_lag_s": corr_x["mean_abs_lag_s"],
        "local_phase_corr_y_mean_abs_lag_s": corr_y["mean_abs_lag_s"],
        "local_phase_corr_x_high_weight_mean": corr_x["high_weight_mean"],
        "local_phase_corr_y_high_weight_mean": corr_y["high_weight_mean"],
        "local_phase_corr_x_static_failure_weight_mean": corr_x["static_failure_weight_mean"],
        "local_phase_corr_y_static_failure_weight_mean": corr_y["static_failure_weight_mean"],
        "local_phase_corr_x_combined_weight_mean": corr_x["combined_weight_mean"],
        "local_phase_corr_y_combined_weight_mean": corr_y["combined_weight_mean"],
        "local_phase_corr_x_n_windows": corr_x["n_windows"],
        "local_phase_corr_y_n_windows": corr_y["n_windows"],
    }


def local_phase_increment_training_loss(
        *,
        pred: torch.Tensor,
        teacher: torch.Tensor,
        static: torch.Tensor,
        cfg: TransformerPhysicalTrainConfig,
        x_idx: torch.Tensor,
        y_idx: torch.Tensor,
) -> dict[str, torch.Tensor]:
    zero = torch.zeros((), dtype=pred.dtype, device=pred.device)
    empty = {
        "local_phase_increment_loss": zero,
        "local_phase_increment_x_absolute_loss": zero,
        "local_phase_increment_y_absolute_loss": zero,
        "local_phase_increment_x_increment_loss": zero,
        "local_phase_increment_y_increment_loss": zero,
        "local_phase_increment_x_phase_cos_mean": zero.detach(),
        "local_phase_increment_y_phase_cos_mean": zero.detach(),
        "local_phase_increment_x_increment_cos_mean": zero.detach(),
        "local_phase_increment_y_increment_cos_mean": zero.detach(),
        "local_phase_increment_x_high_weight_mean": zero.detach(),
        "local_phase_increment_y_high_weight_mean": zero.detach(),
        "local_phase_increment_x_static_failure_weight_mean": zero.detach(),
        "local_phase_increment_y_static_failure_weight_mean": zero.detach(),
        "local_phase_increment_x_combined_weight_mean": zero.detach(),
        "local_phase_increment_y_combined_weight_mean": zero.detach(),
        "local_phase_increment_x_target_freq_hz_mean": zero.detach(),
        "local_phase_increment_y_target_freq_hz_mean": zero.detach(),
        "local_phase_increment_x_n_windows": zero.detach(),
        "local_phase_increment_y_n_windows": zero.detach(),
        "local_phase_increment_x_n_increments": zero.detach(),
        "local_phase_increment_y_n_increments": zero.detach(),
    }
    if not bool(getattr(cfg, "use_local_phase_increment_loss", False)):
        return empty

    common_kwargs = dict(
        dt=float(cfg.dt),
        observations=str(cfg.local_phase_increment_observations),
        last_k=int(cfg.local_phase_increment_last_k),
        start_time=float(cfg.local_phase_increment_start),
        end_time=cfg.local_phase_increment_end,
        window_seconds=float(cfg.local_phase_increment_window_seconds),
        stride_seconds=float(cfg.local_phase_increment_stride_seconds),
        freq_min=float(cfg.local_phase_increment_freq_min),
        freq_max=cfg.local_phase_increment_freq_max,
        base_weight=float(cfg.local_phase_increment_base_weight),
        high_power_weight=float(cfg.local_phase_increment_high_power_weight),
        high_power_threshold=float(cfg.local_phase_increment_high_power_threshold),
        high_power_temperature=float(cfg.local_phase_increment_high_power_temperature),
        static_failure_weight=float(cfg.local_phase_increment_static_failure_weight),
        static_failure_max_weight=float(cfg.local_phase_increment_static_failure_max_weight),
        static_failure_corr_threshold=float(cfg.static_quality_good_corr_threshold),
        static_failure_lag_seconds=float(cfg.static_quality_good_lag_seconds),
        static_failure_amp_log_tol=float(cfg.static_quality_good_amp_log_tol),
        static_failure_max_lag_seconds=float(cfg.static_quality_max_lag_seconds),
        lag_temperature=float(cfg.lag_temperature),
    )
    phase_x = local_phase_increment_loss(
        pred,
        teacher,
        static_reference=static,
        dof_indices=x_idx,
        **common_kwargs,
    )
    phase_y = local_phase_increment_loss(
        pred,
        teacher,
        static_reference=static,
        dof_indices=y_idx,
        **common_kwargs,
    )
    loss = (
        float(cfg.w_local_phase_absolute_x) * phase_x["absolute_phase_loss"]
        + float(cfg.w_local_phase_absolute_y) * phase_y["absolute_phase_loss"]
        + float(cfg.w_local_phase_increment_x) * phase_x["increment_phase_loss"]
        + float(cfg.w_local_phase_increment_y) * phase_y["increment_phase_loss"]
    )
    return {
        "local_phase_increment_loss": loss,
        "local_phase_increment_x_absolute_loss": phase_x["absolute_phase_loss"],
        "local_phase_increment_y_absolute_loss": phase_y["absolute_phase_loss"],
        "local_phase_increment_x_increment_loss": phase_x["increment_phase_loss"],
        "local_phase_increment_y_increment_loss": phase_y["increment_phase_loss"],
        "local_phase_increment_x_phase_cos_mean": phase_x["phase_cos_mean"],
        "local_phase_increment_y_phase_cos_mean": phase_y["phase_cos_mean"],
        "local_phase_increment_x_increment_cos_mean": phase_x["increment_cos_mean"],
        "local_phase_increment_y_increment_cos_mean": phase_y["increment_cos_mean"],
        "local_phase_increment_x_high_weight_mean": phase_x["high_weight_mean"],
        "local_phase_increment_y_high_weight_mean": phase_y["high_weight_mean"],
        "local_phase_increment_x_static_failure_weight_mean": phase_x["static_failure_weight_mean"],
        "local_phase_increment_y_static_failure_weight_mean": phase_y["static_failure_weight_mean"],
        "local_phase_increment_x_combined_weight_mean": phase_x["combined_weight_mean"],
        "local_phase_increment_y_combined_weight_mean": phase_y["combined_weight_mean"],
        "local_phase_increment_x_target_freq_hz_mean": phase_x["target_freq_hz_mean"],
        "local_phase_increment_y_target_freq_hz_mean": phase_y["target_freq_hz_mean"],
        "local_phase_increment_x_n_windows": phase_x["n_windows"],
        "local_phase_increment_y_n_windows": phase_y["n_windows"],
        "local_phase_increment_x_n_increments": phase_x["n_increments"],
        "local_phase_increment_y_n_increments": phase_y["n_increments"],
    }


def continuous_phase_lag_training_loss(
        *,
        pred: torch.Tensor,
        teacher: torch.Tensor,
        static: torch.Tensor,
        cfg: TransformerPhysicalTrainConfig,
        x_idx: torch.Tensor,
        y_idx: torch.Tensor,
) -> dict[str, torch.Tensor]:
    zero = torch.zeros((), dtype=pred.dtype, device=pred.device)
    empty = {
        "continuous_phase_loss": zero,
        "continuous_phase_x_absolute_loss": zero,
        "continuous_phase_y_absolute_loss": zero,
        "continuous_phase_x_time_shift_loss": zero,
        "continuous_phase_y_time_shift_loss": zero,
        "continuous_phase_x_phase_cos_mean": zero.detach(),
        "continuous_phase_y_phase_cos_mean": zero.detach(),
        "continuous_phase_x_equivalent_abs_lag_s_mean": zero.detach(),
        "continuous_phase_y_equivalent_abs_lag_s_mean": zero.detach(),
        "continuous_phase_x_high_weight_mean": zero.detach(),
        "continuous_phase_y_high_weight_mean": zero.detach(),
        "continuous_phase_x_static_failure_weight_mean": zero.detach(),
        "continuous_phase_y_static_failure_weight_mean": zero.detach(),
        "continuous_phase_x_combined_weight_mean": zero.detach(),
        "continuous_phase_y_combined_weight_mean": zero.detach(),
        "continuous_phase_x_target_freq_hz_mean": zero.detach(),
        "continuous_phase_y_target_freq_hz_mean": zero.detach(),
        "continuous_phase_x_n_windows": zero.detach(),
        "continuous_phase_y_n_windows": zero.detach(),
    }
    if not bool(getattr(cfg, "use_continuous_phase_lag_loss", False)):
        return empty

    common_kwargs = dict(
        dt=float(cfg.dt),
        observations=str(cfg.continuous_phase_observations),
        last_k=int(cfg.continuous_phase_last_k),
        start_time=float(cfg.continuous_phase_start),
        end_time=cfg.continuous_phase_end,
        window_seconds=float(cfg.continuous_phase_window_seconds),
        stride_seconds=float(cfg.continuous_phase_stride_seconds),
        freq_min=float(cfg.continuous_phase_freq_min),
        freq_max=cfg.continuous_phase_freq_max,
        n_freq_bins=int(cfg.continuous_phase_n_freq_bins),
        frequency_temperature=float(cfg.continuous_phase_frequency_temperature),
        time_shift_scale_seconds=float(cfg.continuous_phase_time_shift_scale_seconds),
        base_weight=float(cfg.continuous_phase_base_weight),
        high_power_weight=float(cfg.continuous_phase_high_power_weight),
        high_power_threshold=float(cfg.continuous_phase_high_power_threshold),
        high_power_temperature=float(cfg.continuous_phase_high_power_temperature),
        static_failure_weight=float(cfg.continuous_phase_static_failure_weight),
        static_failure_max_weight=float(cfg.continuous_phase_static_failure_max_weight),
        static_failure_corr_threshold=float(cfg.static_quality_good_corr_threshold),
        static_failure_lag_seconds=float(cfg.static_quality_good_lag_seconds),
        static_failure_amp_log_tol=float(cfg.static_quality_good_amp_log_tol),
        static_failure_max_lag_seconds=float(cfg.static_quality_max_lag_seconds),
        lag_temperature=float(cfg.lag_temperature),
    )
    phase_x = local_continuous_phase_lag_loss(
        pred,
        teacher,
        static_reference=static,
        dof_indices=x_idx,
        **common_kwargs,
    )
    phase_y = local_continuous_phase_lag_loss(
        pred,
        teacher,
        static_reference=static,
        dof_indices=y_idx,
        **common_kwargs,
    )
    loss = (
        float(cfg.w_continuous_phase_absolute_x) * phase_x["absolute_phase_loss"]
        + float(cfg.w_continuous_phase_absolute_y) * phase_y["absolute_phase_loss"]
        + float(cfg.w_continuous_phase_time_shift_x) * phase_x["time_shift_loss"]
        + float(cfg.w_continuous_phase_time_shift_y) * phase_y["time_shift_loss"]
    )
    return {
        "continuous_phase_loss": loss,
        "continuous_phase_x_absolute_loss": phase_x["absolute_phase_loss"],
        "continuous_phase_y_absolute_loss": phase_y["absolute_phase_loss"],
        "continuous_phase_x_time_shift_loss": phase_x["time_shift_loss"],
        "continuous_phase_y_time_shift_loss": phase_y["time_shift_loss"],
        "continuous_phase_x_phase_cos_mean": phase_x["phase_cos_mean"],
        "continuous_phase_y_phase_cos_mean": phase_y["phase_cos_mean"],
        "continuous_phase_x_equivalent_abs_lag_s_mean": phase_x["equivalent_abs_lag_s_mean"],
        "continuous_phase_y_equivalent_abs_lag_s_mean": phase_y["equivalent_abs_lag_s_mean"],
        "continuous_phase_x_high_weight_mean": phase_x["high_weight_mean"],
        "continuous_phase_y_high_weight_mean": phase_y["high_weight_mean"],
        "continuous_phase_x_static_failure_weight_mean": phase_x["static_failure_weight_mean"],
        "continuous_phase_y_static_failure_weight_mean": phase_y["static_failure_weight_mean"],
        "continuous_phase_x_combined_weight_mean": phase_x["combined_weight_mean"],
        "continuous_phase_y_combined_weight_mean": phase_y["combined_weight_mean"],
        "continuous_phase_x_target_freq_hz_mean": phase_x["target_freq_hz_mean"],
        "continuous_phase_y_target_freq_hz_mean": phase_y["target_freq_hz_mean"],
        "continuous_phase_x_n_windows": phase_x["n_windows"],
        "continuous_phase_y_n_windows": phase_y["n_windows"],
    }


def local_phase_slope_training_loss(
        *,
        pred: torch.Tensor,
        teacher: torch.Tensor,
        static: torch.Tensor,
        cfg: TransformerPhysicalTrainConfig,
        x_idx: torch.Tensor,
        y_idx: torch.Tensor,
) -> dict[str, torch.Tensor]:
    zero = torch.zeros((), dtype=pred.dtype, device=pred.device)
    empty = {
        "local_phase_slope_loss": zero,
        "local_phase_slope_x_loss": zero,
        "local_phase_slope_y_loss": zero,
        "local_phase_slope_x_phase_cos_mean": zero.detach(),
        "local_phase_slope_y_phase_cos_mean": zero.detach(),
        "local_phase_slope_x_equivalent_abs_dlag_s_mean": zero.detach(),
        "local_phase_slope_y_equivalent_abs_dlag_s_mean": zero.detach(),
        "local_phase_slope_x_high_weight_mean": zero.detach(),
        "local_phase_slope_y_high_weight_mean": zero.detach(),
        "local_phase_slope_x_static_failure_weight_mean": zero.detach(),
        "local_phase_slope_y_static_failure_weight_mean": zero.detach(),
        "local_phase_slope_x_combined_weight_mean": zero.detach(),
        "local_phase_slope_y_combined_weight_mean": zero.detach(),
        "local_phase_slope_x_target_freq_hz_mean": zero.detach(),
        "local_phase_slope_y_target_freq_hz_mean": zero.detach(),
        "local_phase_slope_x_n_windows": zero.detach(),
        "local_phase_slope_y_n_windows": zero.detach(),
        "local_phase_slope_x_n_slopes": zero.detach(),
        "local_phase_slope_y_n_slopes": zero.detach(),
    }
    if not bool(getattr(cfg, "use_local_phase_slope_loss", False)):
        return empty

    common_kwargs = dict(
        dt=float(cfg.dt),
        observations=str(cfg.local_phase_slope_observations),
        last_k=int(cfg.local_phase_slope_last_k),
        start_time=float(cfg.local_phase_slope_start),
        end_time=cfg.local_phase_slope_end,
        window_seconds=float(cfg.local_phase_slope_window_seconds),
        stride_seconds=float(cfg.local_phase_slope_stride_seconds),
        freq_min=float(cfg.local_phase_slope_freq_min),
        freq_max=cfg.local_phase_slope_freq_max,
        n_freq_bins=int(cfg.local_phase_slope_n_freq_bins),
        frequency_temperature=float(cfg.local_phase_slope_frequency_temperature),
        time_shift_scale_seconds=float(cfg.local_phase_slope_time_shift_scale_seconds),
        base_weight=float(cfg.local_phase_slope_base_weight),
        high_power_weight=float(cfg.local_phase_slope_high_power_weight),
        high_power_threshold=float(cfg.local_phase_slope_high_power_threshold),
        high_power_temperature=float(cfg.local_phase_slope_high_power_temperature),
        static_failure_weight=float(cfg.local_phase_slope_static_failure_weight),
        static_failure_max_weight=float(cfg.local_phase_slope_static_failure_max_weight),
        static_failure_corr_threshold=float(cfg.static_quality_good_corr_threshold),
        static_failure_lag_seconds=float(cfg.static_quality_good_lag_seconds),
        static_failure_amp_log_tol=float(cfg.static_quality_good_amp_log_tol),
        static_failure_max_lag_seconds=float(cfg.static_quality_max_lag_seconds),
        lag_temperature=float(cfg.lag_temperature),
    )
    phase_x = local_phase_slope_loss(
        pred,
        teacher,
        static_reference=static,
        dof_indices=x_idx,
        **common_kwargs,
    )
    phase_y = local_phase_slope_loss(
        pred,
        teacher,
        static_reference=static,
        dof_indices=y_idx,
        **common_kwargs,
    )
    loss = (
        float(cfg.w_local_phase_slope_x) * phase_x["slope_loss"]
        + float(cfg.w_local_phase_slope_y) * phase_y["slope_loss"]
    )
    return {
        "local_phase_slope_loss": loss,
        "local_phase_slope_x_loss": phase_x["slope_loss"],
        "local_phase_slope_y_loss": phase_y["slope_loss"],
        "local_phase_slope_x_phase_cos_mean": phase_x["slope_phase_cos_mean"],
        "local_phase_slope_y_phase_cos_mean": phase_y["slope_phase_cos_mean"],
        "local_phase_slope_x_equivalent_abs_dlag_s_mean": phase_x["equivalent_abs_dlag_s_mean"],
        "local_phase_slope_y_equivalent_abs_dlag_s_mean": phase_y["equivalent_abs_dlag_s_mean"],
        "local_phase_slope_x_high_weight_mean": phase_x["high_weight_mean"],
        "local_phase_slope_y_high_weight_mean": phase_y["high_weight_mean"],
        "local_phase_slope_x_static_failure_weight_mean": phase_x["static_failure_weight_mean"],
        "local_phase_slope_y_static_failure_weight_mean": phase_y["static_failure_weight_mean"],
        "local_phase_slope_x_combined_weight_mean": phase_x["combined_weight_mean"],
        "local_phase_slope_y_combined_weight_mean": phase_y["combined_weight_mean"],
        "local_phase_slope_x_target_freq_hz_mean": phase_x["target_freq_hz_mean"],
        "local_phase_slope_y_target_freq_hz_mean": phase_y["target_freq_hz_mean"],
        "local_phase_slope_x_n_windows": phase_x["n_windows"],
        "local_phase_slope_y_n_windows": phase_y["n_windows"],
        "local_phase_slope_x_n_slopes": phase_x["n_slopes"],
        "local_phase_slope_y_n_slopes": phase_y["n_slopes"],
    }


def _static_quality_observation_indices(
        *,
        dof_indices: torch.Tensor,
        observations: str,
        last_k: int,
) -> list[torch.Tensor]:
    obs = set(_split_keyword_list(observations))
    indices: list[torch.Tensor] = []
    if "tip" in obs and int(dof_indices.numel()) > 0:
        indices.append(dof_indices[-1:].to(dtype=torch.long))
    if "last5" in obs and int(dof_indices.numel()) > 0:
        k = max(1, min(int(last_k), int(dof_indices.numel())))
        indices.append(dof_indices[-k:].to(dtype=torch.long))
    if "mean" in obs and int(dof_indices.numel()) > 0:
        indices.append(dof_indices.to(dtype=torch.long))
    if not indices and int(dof_indices.numel()) > 0:
        indices.append(dof_indices[-1:].to(dtype=torch.long))
    return indices


def _best_window_lag_corr_amp(
        static_signal: torch.Tensor,
        teacher_signal: torch.Tensor,
        *,
        dt: float,
        max_lag_seconds: float,
        eps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_lag_steps = max(0, int(round(float(max_lag_seconds) / float(dt))))
    best_corr: Optional[torch.Tensor] = None
    best_abs_lag_s = torch.zeros((), dtype=static_signal.dtype, device=static_signal.device)
    for lag in range(-max_lag_steps, max_lag_steps + 1):
        if lag < 0:
            s = static_signal[-lag:]
            t = teacher_signal[: s.shape[0]]
        elif lag > 0:
            s = static_signal[:-lag]
            t = teacher_signal[lag:]
        else:
            s = static_signal
            t = teacher_signal
        if int(s.numel()) < 2:
            continue
        s0 = s - torch.mean(s)
        t0 = t - torch.mean(t)
        denom = torch.sqrt(torch.sum(s0 * s0).clamp_min(eps) * torch.sum(t0 * t0).clamp_min(eps))
        corr = torch.sum(s0 * t0) / denom.clamp_min(eps)
        if best_corr is None or bool((corr > best_corr).detach().cpu()):
            best_corr = corr
            best_abs_lag_s = torch.as_tensor(abs(lag) * float(dt), dtype=static_signal.dtype, device=static_signal.device)

    if best_corr is None:
        best_corr = torch.zeros((), dtype=static_signal.dtype, device=static_signal.device)

    static_rms = torch.sqrt(torch.mean((static_signal - torch.mean(static_signal)) ** 2).clamp_min(eps))
    teacher_rms = torch.sqrt(torch.mean((teacher_signal - torch.mean(teacher_signal)) ** 2).clamp_min(eps))
    amp_log_abs = torch.abs(torch.log(static_rms.clamp_min(eps) / teacher_rms.clamp_min(eps)))
    return best_abs_lag_s.detach(), best_corr.detach(), amp_log_abs.detach()


def _window_corr_and_amp_log_abs(
        signal: torch.Tensor,
        teacher_signal: torch.Tensor,
        eps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    s0 = signal - torch.mean(signal)
    t0 = teacher_signal - torch.mean(teacher_signal)
    denom = torch.sqrt(torch.sum(s0 * s0).clamp_min(eps) * torch.sum(t0 * t0).clamp_min(eps))
    corr = torch.sum(s0 * t0) / denom.clamp_min(eps)
    signal_rms = torch.sqrt(torch.mean(s0 * s0).clamp_min(eps))
    teacher_rms = torch.sqrt(torch.mean(t0 * t0).clamp_min(eps))
    amp_log_abs = torch.abs(torch.log(signal_rms.clamp_min(eps) / teacher_rms.clamp_min(eps)))
    return corr, amp_log_abs


def _window_amp_log_signed(
        signal: torch.Tensor,
        teacher_signal: torch.Tensor,
        eps: torch.Tensor,
) -> torch.Tensor:
    s0 = signal - torch.mean(signal)
    t0 = teacher_signal - torch.mean(teacher_signal)
    signal_rms = torch.sqrt(torch.mean(s0 * s0).clamp_min(eps))
    teacher_rms = torch.sqrt(torch.mean(t0 * t0).clamp_min(eps))
    return torch.log(signal_rms.clamp_min(eps) / teacher_rms.clamp_min(eps))


def _enabled_param_index(enabled_params: str, name: str) -> Optional[int]:
    names = [item.strip() for item in str(enabled_params).replace(",", " ").split() if item.strip()]
    try:
        return names.index(name)
    except ValueError:
        return None


def static_quality_gate_suppression_loss(
        *,
        static: torch.Tensor,
        teacher: torch.Tensor,
        theta_aux: Optional[dict[str, torch.Tensor]],
        cfg: TransformerPhysicalTrainConfig,
        x_idx: torch.Tensor,
        y_idx: torch.Tensor,
) -> dict[str, torch.Tensor]:
    zero = torch.zeros((), dtype=teacher.dtype, device=teacher.device)
    empty = {
        "static_quality_gate_loss": zero,
        "static_good_gate_mean": zero,
        "static_bad_gate_mean": zero,
        "static_gate_selectivity_gap": zero,
        "static_good_window_ratio": zero,
        "static_bad_window_ratio": zero,
        "static_quality_score_mean": zero,
        "static_quality_score_max": zero,
        "static_quality_gate_mean": zero,
        "static_quality_n_windows": zero,
    }
    if (
        not bool(getattr(cfg, "use_static_quality_gate_suppression", False))
        or theta_aux is None
        or theta_aux.get("g_phase") is None
    ):
        return empty

    gate = theta_aux["g_phase"].to(dtype=teacher.dtype, device=teacher.device)
    if gate.ndim == 3:
        gate = gate[0]
    if gate.ndim == 2:
        gate = torch.mean(gate, dim=-1)
    if gate.ndim != 1:
        return empty

    static = static.to(dtype=teacher.dtype, device=teacher.device).detach()
    teacher = teacher.detach()
    T = int(min(static.shape[0], teacher.shape[0], gate.shape[0]))
    if T < 2:
        return empty
    static = static[:T]
    teacher = teacher[:T]
    gate = gate[:T]

    dt = float(cfg.dt)
    start = max(0, int(round(float(cfg.static_quality_start) / dt)))
    end = T if cfg.static_quality_end is None else min(T, int(round(float(cfg.static_quality_end) / dt)))
    win = max(2, int(round(float(cfg.static_quality_window_seconds) / dt)))
    stride = max(1, int(round(float(cfg.static_quality_stride_seconds) / dt)))
    if end - start < win:
        return empty

    obs_indices = (
        _static_quality_observation_indices(
            dof_indices=x_idx,
            observations=str(cfg.static_quality_observations),
            last_k=int(cfg.static_quality_last_k),
        )
        + _static_quality_observation_indices(
            dof_indices=y_idx,
            observations=str(cfg.static_quality_observations),
            last_k=int(cfg.static_quality_last_k),
        )
    )
    if not obs_indices:
        return empty

    scores = []
    gate_means = []
    good_weights = []
    bad_weights = []
    for s0 in range(start, end - win + 1, stride):
        s1 = s0 + win
        signal_scores = []
        for idx in obs_indices:
            static_signal = torch.mean(static[s0:s1, idx], dim=-1)
            teacher_signal = torch.mean(teacher[s0:s1, idx], dim=-1)
            abs_lag_s, corr, amp_log_abs = _best_window_lag_corr_amp(
                static_signal,
                teacher_signal,
                dt=dt,
                max_lag_seconds=float(cfg.static_quality_max_lag_seconds),
                eps=torch.as_tensor(1.0e-12, dtype=teacher.dtype, device=teacher.device),
            )
            corr_excess = torch.relu(torch.as_tensor(float(cfg.static_quality_good_corr_threshold), dtype=teacher.dtype, device=teacher.device) - corr)
            lag_excess = torch.relu(abs_lag_s - float(cfg.static_quality_good_lag_seconds)) / max(float(cfg.static_quality_good_lag_seconds), 1.0e-12)
            amp_excess = torch.relu(amp_log_abs - float(cfg.static_quality_good_amp_log_tol)) / max(float(cfg.static_quality_good_amp_log_tol), 1.0e-12)
            signal_scores.append(corr_excess + lag_excess + amp_excess)
        score = torch.stack(signal_scores).mean().detach()
        gate_mean = torch.mean(gate[s0:s1])
        good = (score <= 1.0e-8).to(dtype=teacher.dtype)
        bad = (score > 1.0e-8).to(dtype=teacher.dtype)
        scores.append(score)
        gate_means.append(gate_mean)
        good_weights.append(good)
        bad_weights.append(bad)

    if not scores:
        return empty

    score_t = torch.stack(scores)
    gate_t = torch.stack(gate_means)
    good_t = torch.stack(good_weights)
    bad_t = torch.stack(bad_weights)
    good_count = torch.sum(good_t).clamp_min(1.0)
    bad_count = torch.sum(bad_t).clamp_min(1.0)
    good_gate = torch.sum(gate_t * good_t) / good_count
    bad_gate = torch.sum(gate_t * bad_t) / bad_count
    has_good = bool((torch.sum(good_t) > 0).detach().cpu())
    unweighted_loss = good_gate if has_good else zero
    weighted_loss = float(cfg.w_static_good_gate_l1) * unweighted_loss

    return {
        "static_quality_gate_loss": weighted_loss,
        "static_good_gate_mean": good_gate.detach(),
        "static_bad_gate_mean": bad_gate.detach(),
        "static_gate_selectivity_gap": (bad_gate - good_gate).detach(),
        "static_good_window_ratio": torch.mean(good_t).detach(),
        "static_bad_window_ratio": torch.mean(bad_t).detach(),
        "static_quality_score_mean": torch.mean(score_t).detach(),
        "static_quality_score_max": torch.max(score_t).detach(),
        "static_quality_gate_mean": torch.mean(gate_t).detach(),
        "static_quality_n_windows": torch.as_tensor(float(len(scores)), dtype=teacher.dtype, device=teacher.device),
    }


def state_window_no_regression_guard_loss(
        *,
        pred: torch.Tensor,
        static: torch.Tensor,
        teacher: torch.Tensor,
        cfg: TransformerPhysicalTrainConfig,
        x_idx: torch.Tensor,
        y_idx: torch.Tensor,
) -> dict[str, torch.Tensor]:
    zero = torch.zeros((), dtype=pred.dtype, device=pred.device)
    empty = {
        "state_no_regression_guard_loss": zero,
        "state_no_regression_response_guard_loss": zero,
        "state_no_regression_corr_guard_loss": zero,
        "state_no_regression_amp_guard_loss": zero,
        "state_no_regression_good_window_ratio": zero.detach(),
        "state_no_regression_response_excess": zero.detach(),
        "state_no_regression_corr_drop_excess": zero.detach(),
        "state_no_regression_amp_log_excess": zero.detach(),
        "state_no_regression_n_windows": zero.detach(),
    }
    if not bool(getattr(cfg, "use_state_window_no_regression_guard", False)):
        return empty

    pred = pred.to(dtype=teacher.dtype, device=teacher.device)
    static = static.to(dtype=teacher.dtype, device=teacher.device).detach()
    teacher = teacher.to(dtype=pred.dtype, device=pred.device)
    T = int(min(pred.shape[0], static.shape[0], teacher.shape[0]))
    if T < 2:
        return empty
    pred = pred[:T]
    static = static[:T]
    teacher = teacher[:T]

    dt = float(cfg.dt)
    start = max(0, int(round(float(cfg.static_quality_start) / dt)))
    end = T if cfg.static_quality_end is None else min(T, int(round(float(cfg.static_quality_end) / dt)))
    win = max(2, int(round(float(cfg.static_quality_window_seconds) / dt)))
    stride = max(1, int(round(float(cfg.static_quality_stride_seconds) / dt)))
    if end - start < win:
        return empty

    obs_indices = (
        _static_quality_observation_indices(
            dof_indices=x_idx,
            observations=str(cfg.static_quality_observations),
            last_k=int(cfg.static_quality_last_k),
        )
        + _static_quality_observation_indices(
            dof_indices=y_idx,
            observations=str(cfg.static_quality_observations),
            last_k=int(cfg.static_quality_last_k),
        )
    )
    if not obs_indices:
        return empty

    eps = torch.as_tensor(1.0e-12, dtype=pred.dtype, device=pred.device)
    response_terms = []
    corr_terms = []
    amp_terms = []
    good_flags = []
    response_excesses = []
    corr_excesses = []
    amp_excesses = []

    for s0 in range(start, end - win + 1, stride):
        s1 = s0 + win
        per_signal_response = []
        per_signal_corr = []
        per_signal_amp = []
        per_signal_good = []
        per_signal_response_excess = []
        per_signal_corr_excess = []
        per_signal_amp_excess = []

        for idx in obs_indices:
            static_signal = torch.mean(static[s0:s1, idx], dim=-1)
            pred_signal = torch.mean(pred[s0:s1, idx], dim=-1)
            teacher_signal = torch.mean(teacher[s0:s1, idx], dim=-1)

            abs_lag_s, static_best_corr, static_amp_log_abs = _best_window_lag_corr_amp(
                static_signal,
                teacher_signal,
                dt=dt,
                max_lag_seconds=float(cfg.static_quality_max_lag_seconds),
                eps=eps,
            )
            static_corr_excess = torch.relu(
                torch.as_tensor(float(cfg.static_quality_good_corr_threshold), dtype=pred.dtype, device=pred.device)
                - static_best_corr
            )
            static_lag_excess = torch.relu(abs_lag_s - float(cfg.static_quality_good_lag_seconds)) / max(
                float(cfg.static_quality_good_lag_seconds),
                1.0e-12,
            )
            static_amp_excess = torch.relu(static_amp_log_abs - float(cfg.static_quality_good_amp_log_tol)) / max(
                float(cfg.static_quality_good_amp_log_tol),
                1.0e-12,
            )
            good = ((static_corr_excess + static_lag_excess + static_amp_excess) <= 1.0e-8).to(dtype=pred.dtype)

            static_mse = torch.mean((static_signal - teacher_signal) ** 2).detach()
            pred_mse = torch.mean((pred_signal - teacher_signal) ** 2)
            response_ratio = pred_mse / static_mse.clamp_min(eps)
            response_excess = torch.relu(response_ratio - float(cfg.state_no_regression_response_ratio_limit))

            static_corr0, _ = _window_corr_and_amp_log_abs(static_signal, teacher_signal, eps)
            pred_corr0, pred_amp_log_abs = _window_corr_and_amp_log_abs(pred_signal, teacher_signal, eps)
            corr_drop = static_corr0.detach() - pred_corr0
            corr_excess = torch.relu(corr_drop - float(cfg.state_no_regression_corr_drop_tol))
            amp_excess = torch.relu(pred_amp_log_abs - float(cfg.state_no_regression_amp_log_tol))

            per_signal_response.append(good * response_excess ** 2)
            per_signal_corr.append(good * corr_excess ** 2)
            per_signal_amp.append(good * amp_excess ** 2)
            per_signal_good.append(good.detach())
            per_signal_response_excess.append((good * response_excess).detach())
            per_signal_corr_excess.append((good * corr_excess).detach())
            per_signal_amp_excess.append((good * amp_excess).detach())

        response_terms.append(torch.stack(per_signal_response).mean())
        corr_terms.append(torch.stack(per_signal_corr).mean())
        amp_terms.append(torch.stack(per_signal_amp).mean())
        good_flags.append(torch.stack(per_signal_good).mean())
        response_excesses.append(torch.stack(per_signal_response_excess).mean())
        corr_excesses.append(torch.stack(per_signal_corr_excess).mean())
        amp_excesses.append(torch.stack(per_signal_amp_excess).mean())

    if not response_terms:
        return empty

    response_guard = torch.stack(response_terms).mean()
    corr_guard = torch.stack(corr_terms).mean()
    amp_guard = torch.stack(amp_terms).mean()
    guard_loss = (
        float(cfg.w_state_no_regression_response) * response_guard
        + float(cfg.w_state_no_regression_corr) * corr_guard
        + float(cfg.w_state_no_regression_amp) * amp_guard
    )

    return {
        "state_no_regression_guard_loss": guard_loss,
        "state_no_regression_response_guard_loss": response_guard,
        "state_no_regression_corr_guard_loss": corr_guard,
        "state_no_regression_amp_guard_loss": amp_guard,
        "state_no_regression_good_window_ratio": torch.stack(good_flags).mean().detach(),
        "state_no_regression_response_excess": torch.stack(response_excesses).mean().detach(),
        "state_no_regression_corr_drop_excess": torch.stack(corr_excesses).mean().detach(),
        "state_no_regression_amp_log_excess": torch.stack(amp_excesses).mean().detach(),
        "state_no_regression_n_windows": torch.as_tensor(float(len(response_terms)), dtype=pred.dtype, device=pred.device),
    }


def _windowed_alpha_relative_amplitude_terms(
        *,
        pred: torch.Tensor,
        alpha_ref: torch.Tensor,
        teacher: torch.Tensor,
        index_groups: list[torch.Tensor],
        cfg: TransformerPhysicalTrainConfig,
        theta: Optional[torch.Tensor] = None,
        beta_index: Optional[int] = None,
        beta_sign_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    zero = torch.zeros((), dtype=pred.dtype, device=pred.device)
    empty = {
        "loss": zero,
        "direct_loss": zero,
        "improvement_loss": zero,
        "sign_loss": zero,
        "pred_log_error": zero.detach(),
        "alpha_log_error": zero.detach(),
        "improvement_margin": zero.detach(),
        "weight": zero.detach(),
        "n_windows": zero.detach(),
    }
    if not index_groups:
        return empty

    T = int(min(pred.shape[0], alpha_ref.shape[0], teacher.shape[0]))
    if T < 2:
        return empty
    pred = pred[:T]
    alpha_ref = alpha_ref[:T].to(dtype=pred.dtype, device=pred.device).detach()
    teacher = teacher[:T].to(dtype=pred.dtype, device=pred.device)

    dt = float(cfg.dt)
    start = max(0, int(round(float(cfg.beta_amp_start) / dt)))
    end = T if cfg.beta_amp_end is None else min(T, int(round(float(cfg.beta_amp_end) / dt)))
    win = max(2, int(round(float(cfg.beta_amp_window_seconds) / dt)))
    stride = max(1, int(round(float(cfg.beta_amp_stride_seconds) / dt)))
    if end - start < win:
        return empty

    eps = torch.as_tensor(1.0e-12, dtype=pred.dtype, device=pred.device)
    alpha_ref_error = max(float(cfg.beta_amp_alpha_error_ref), 1.0e-12)
    max_weight = max(float(cfg.beta_amp_alpha_error_max_weight), 0.0)
    log_tol = max(float(cfg.beta_amp_log_tol), 0.0)
    improvement_margin = max(float(cfg.beta_amp_improvement_margin), 0.0)
    improvement_weight = max(float(cfg.w_beta_amp_improvement), 0.0)
    beta_sign_weight = max(float(beta_sign_weight), 0.0)
    sign_min_alpha_error = max(float(cfg.beta_damp_sign_min_alpha_error), 0.0)
    theta_seq = None
    if theta is not None and beta_index is not None and beta_sign_weight > 0.0:
        theta_seq = theta[:T].to(dtype=pred.dtype, device=pred.device)
        if theta_seq.ndim != 2 or int(beta_index) < 0 or int(beta_index) >= int(theta_seq.shape[-1]):
            theta_seq = None

    losses: list[torch.Tensor] = []
    direct_losses: list[torch.Tensor] = []
    improvement_losses: list[torch.Tensor] = []
    sign_losses: list[torch.Tensor] = []
    pred_errors: list[torch.Tensor] = []
    alpha_errors: list[torch.Tensor] = []
    improvement_margins: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []

    for s0 in range(start, end - win + 1, stride):
        s1 = s0 + win
        for idx in index_groups:
            idx = idx.to(device=pred.device, dtype=torch.long)
            pred_signal = torch.mean(pred[s0:s1, idx], dim=-1)
            alpha_signal = torch.mean(alpha_ref[s0:s1, idx], dim=-1)
            teacher_signal = torch.mean(teacher[s0:s1, idx], dim=-1)

            pred_amp_log = _window_amp_log_signed(pred_signal, teacher_signal, eps)
            alpha_amp_log = _window_amp_log_signed(alpha_signal, teacher_signal, eps).detach()
            pred_amp_log_abs = torch.abs(pred_amp_log)
            alpha_amp_log_abs = torch.abs(alpha_amp_log)
            alpha_amp_log_abs = alpha_amp_log_abs.detach()
            weight = torch.clamp(alpha_amp_log_abs / alpha_ref_error, min=0.0, max=max_weight).detach()
            amp_error = torch.relu(pred_amp_log_abs - log_tol)
            direct_loss = weight * amp_error ** 2
            improvement_error = torch.relu(pred_amp_log_abs - alpha_amp_log_abs + improvement_margin)
            improvement_loss = weight * improvement_error ** 2
            sign_loss = zero
            if theta_seq is not None:
                desired_sign = torch.sign(alpha_amp_log).detach()
                active = (alpha_amp_log_abs > sign_min_alpha_error).to(dtype=pred.dtype)
                theta_mean = torch.mean(theta_seq[s0:s1, int(beta_index)])
                sign_loss = weight * active * torch.relu(-desired_sign * theta_mean) ** 2
            total = (
                direct_loss
                + improvement_weight * improvement_loss
                + beta_sign_weight * sign_loss
            )
            losses.append(total)
            direct_losses.append(direct_loss)
            improvement_losses.append(improvement_loss)
            sign_losses.append(sign_loss)
            pred_errors.append(pred_amp_log_abs.detach())
            alpha_errors.append(alpha_amp_log_abs)
            improvement_margins.append((alpha_amp_log_abs - pred_amp_log_abs).detach())
            weights.append(weight)

    if not losses:
        return empty

    return {
        "loss": torch.stack(losses).mean(),
        "direct_loss": torch.stack(direct_losses).mean(),
        "improvement_loss": torch.stack(improvement_losses).mean(),
        "sign_loss": torch.stack(sign_losses).mean(),
        "pred_log_error": torch.stack(pred_errors).mean().detach(),
        "alpha_log_error": torch.stack(alpha_errors).mean().detach(),
        "improvement_margin": torch.stack(improvement_margins).mean().detach(),
        "weight": torch.stack(weights).mean().detach(),
        "n_windows": torch.as_tensor(float(len(losses)), dtype=pred.dtype, device=pred.device),
    }


def beta_alpha_relative_amplitude_loss(
        *,
        pred: torch.Tensor,
        alpha_ref: Optional[torch.Tensor],
        teacher: torch.Tensor,
        theta: Optional[torch.Tensor],
        cfg: TransformerPhysicalTrainConfig,
        x_idx: torch.Tensor,
        y_idx: torch.Tensor,
        tip_y_idx: int,
        last5_y_idx: torch.Tensor,
) -> dict[str, torch.Tensor]:
    zero = torch.zeros((), dtype=pred.dtype, device=pred.device)
    empty = {
        "beta_amp_loss": zero,
        "beta_amp_x_loss": zero,
        "beta_amp_y_loss": zero,
        "beta_amp_tip_y_loss": zero,
        "beta_amp_last5_y_loss": zero,
        "beta_amp_direct_loss": zero,
        "beta_amp_improvement_loss": zero,
        "beta_amp_x_improvement_loss": zero,
        "beta_amp_y_improvement_loss": zero,
        "beta_amp_tip_y_improvement_loss": zero,
        "beta_amp_last5_y_improvement_loss": zero,
        "beta_damp_sign_loss": zero,
        "beta_damp_x_sign_loss": zero,
        "beta_damp_y_sign_loss": zero,
        "beta_amp_pred_log_error_mean": zero.detach(),
        "beta_amp_alpha_log_error_mean": zero.detach(),
        "beta_amp_improvement_margin_mean": zero.detach(),
        "beta_amp_weight_mean": zero.detach(),
        "beta_amp_n_windows": zero.detach(),
    }
    if alpha_ref is None:
        return empty

    alpha = alpha_ref.to(dtype=pred.dtype, device=pred.device)
    theta_seq = None
    if theta is not None:
        theta_seq = theta[: int(pred.shape[0])].to(dtype=pred.dtype, device=pred.device)
    beta_x_index = _enabled_param_index(str(cfg.enabled_params), "beta_damp_x")
    beta_y_index = _enabled_param_index(str(cfg.enabled_params), "beta_damp_y")
    x_terms = _windowed_alpha_relative_amplitude_terms(
        pred=pred,
        alpha_ref=alpha,
        teacher=teacher,
        index_groups=_static_quality_observation_indices(
            dof_indices=x_idx,
            observations=str(cfg.beta_amp_observations),
            last_k=int(cfg.beta_amp_last_k),
        ),
        cfg=cfg,
        theta=theta_seq,
        beta_index=beta_x_index,
        beta_sign_weight=float(cfg.w_beta_damp_sign_x),
    )
    y_terms = _windowed_alpha_relative_amplitude_terms(
        pred=pred,
        alpha_ref=alpha,
        teacher=teacher,
        index_groups=_static_quality_observation_indices(
            dof_indices=y_idx,
            observations=str(cfg.beta_amp_observations),
            last_k=int(cfg.beta_amp_last_k),
        ),
        cfg=cfg,
        theta=theta_seq,
        beta_index=beta_y_index,
        beta_sign_weight=float(cfg.w_beta_damp_sign_y),
    )
    tip_y_terms = _windowed_alpha_relative_amplitude_terms(
        pred=pred,
        alpha_ref=alpha,
        teacher=teacher,
        index_groups=[torch.as_tensor([int(tip_y_idx)], dtype=torch.long, device=pred.device)],
        cfg=cfg,
        theta=theta_seq,
        beta_index=beta_y_index,
        beta_sign_weight=0.0,
    )
    last5_y_terms = _windowed_alpha_relative_amplitude_terms(
        pred=pred,
        alpha_ref=alpha,
        teacher=teacher,
        index_groups=[last5_y_idx.to(dtype=torch.long, device=pred.device)],
        cfg=cfg,
        theta=theta_seq,
        beta_index=beta_y_index,
        beta_sign_weight=0.0,
    )

    loss = (
        float(cfg.w_beta_amp_x) * x_terms["loss"]
        + float(cfg.w_beta_amp_y) * y_terms["loss"]
        + float(cfg.w_beta_amp_tip_y) * tip_y_terms["loss"]
        + float(cfg.w_beta_amp_last5_y) * last5_y_terms["loss"]
    )
    pred_error_values = [
        x_terms["pred_log_error"],
        y_terms["pred_log_error"],
        tip_y_terms["pred_log_error"],
        last5_y_terms["pred_log_error"],
    ]
    alpha_error_values = [
        x_terms["alpha_log_error"],
        y_terms["alpha_log_error"],
        tip_y_terms["alpha_log_error"],
        last5_y_terms["alpha_log_error"],
    ]
    weight_values = [
        x_terms["weight"],
        y_terms["weight"],
        tip_y_terms["weight"],
        last5_y_terms["weight"],
    ]
    direct_loss = (
        float(cfg.w_beta_amp_x) * x_terms["direct_loss"]
        + float(cfg.w_beta_amp_y) * y_terms["direct_loss"]
        + float(cfg.w_beta_amp_tip_y) * tip_y_terms["direct_loss"]
        + float(cfg.w_beta_amp_last5_y) * last5_y_terms["direct_loss"]
    )
    improvement_loss = (
        float(cfg.w_beta_amp_x) * x_terms["improvement_loss"]
        + float(cfg.w_beta_amp_y) * y_terms["improvement_loss"]
        + float(cfg.w_beta_amp_tip_y) * tip_y_terms["improvement_loss"]
        + float(cfg.w_beta_amp_last5_y) * last5_y_terms["improvement_loss"]
    )
    sign_loss = (
        float(cfg.w_beta_amp_x) * x_terms["sign_loss"]
        + float(cfg.w_beta_amp_y) * y_terms["sign_loss"]
    )
    improvement_margin_values = [
        x_terms["improvement_margin"],
        y_terms["improvement_margin"],
        tip_y_terms["improvement_margin"],
        last5_y_terms["improvement_margin"],
    ]
    n_windows = (
        x_terms["n_windows"]
        + y_terms["n_windows"]
        + tip_y_terms["n_windows"]
        + last5_y_terms["n_windows"]
    )
    return {
        "beta_amp_loss": loss,
        "beta_amp_x_loss": x_terms["loss"],
        "beta_amp_y_loss": y_terms["loss"],
        "beta_amp_tip_y_loss": tip_y_terms["loss"],
        "beta_amp_last5_y_loss": last5_y_terms["loss"],
        "beta_amp_direct_loss": direct_loss,
        "beta_amp_improvement_loss": improvement_loss,
        "beta_amp_x_improvement_loss": x_terms["improvement_loss"],
        "beta_amp_y_improvement_loss": y_terms["improvement_loss"],
        "beta_amp_tip_y_improvement_loss": tip_y_terms["improvement_loss"],
        "beta_amp_last5_y_improvement_loss": last5_y_terms["improvement_loss"],
        "beta_damp_sign_loss": sign_loss,
        "beta_damp_x_sign_loss": x_terms["sign_loss"],
        "beta_damp_y_sign_loss": y_terms["sign_loss"],
        "beta_amp_pred_log_error_mean": torch.stack(pred_error_values).mean().detach(),
        "beta_amp_alpha_log_error_mean": torch.stack(alpha_error_values).mean().detach(),
        "beta_amp_improvement_margin_mean": torch.stack(improvement_margin_values).mean().detach(),
        "beta_amp_weight_mean": torch.stack(weight_values).mean().detach(),
        "beta_amp_n_windows": n_windows.detach(),
    }


def beta_alpha_no_regression_guard_loss(
        *,
        pred: torch.Tensor,
        alpha_ref: Optional[torch.Tensor],
        teacher: torch.Tensor,
        cfg: TransformerPhysicalTrainConfig,
        x_idx: torch.Tensor,
        y_idx: torch.Tensor,
) -> dict[str, torch.Tensor]:
    zero = torch.zeros((), dtype=pred.dtype, device=pred.device)
    empty = {
        "beta_alpha_guard_loss": zero,
        "beta_alpha_response_guard_loss": zero,
        "beta_alpha_corr_guard_loss": zero,
        "beta_alpha_amp_guard_loss": zero,
        "beta_alpha_response_excess": zero.detach(),
        "beta_alpha_corr_drop_excess": zero.detach(),
        "beta_alpha_amp_worsen_excess": zero.detach(),
        "beta_alpha_guard_n_windows": zero.detach(),
    }
    if alpha_ref is None:
        return empty

    pred = pred.to(dtype=teacher.dtype, device=teacher.device)
    alpha_ref = alpha_ref.to(dtype=teacher.dtype, device=teacher.device).detach()
    teacher = teacher.to(dtype=pred.dtype, device=pred.device)
    T = int(min(pred.shape[0], alpha_ref.shape[0], teacher.shape[0]))
    if T < 2:
        return empty
    pred = pred[:T]
    alpha_ref = alpha_ref[:T]
    teacher = teacher[:T]

    dt = float(cfg.dt)
    start = max(0, int(round(float(cfg.beta_amp_start) / dt)))
    end = T if cfg.beta_amp_end is None else min(T, int(round(float(cfg.beta_amp_end) / dt)))
    win = max(2, int(round(float(cfg.beta_amp_window_seconds) / dt)))
    stride = max(1, int(round(float(cfg.beta_amp_stride_seconds) / dt)))
    if end - start < win:
        return empty

    obs_indices = (
        _static_quality_observation_indices(
            dof_indices=x_idx,
            observations=str(cfg.beta_amp_observations),
            last_k=int(cfg.beta_amp_last_k),
        )
        + _static_quality_observation_indices(
            dof_indices=y_idx,
            observations=str(cfg.beta_amp_observations),
            last_k=int(cfg.beta_amp_last_k),
        )
    )
    if not obs_indices:
        return empty

    eps = torch.as_tensor(1.0e-12, dtype=pred.dtype, device=pred.device)
    response_terms: list[torch.Tensor] = []
    corr_terms: list[torch.Tensor] = []
    amp_terms: list[torch.Tensor] = []
    response_excesses: list[torch.Tensor] = []
    corr_excesses: list[torch.Tensor] = []
    amp_excesses: list[torch.Tensor] = []

    for s0 in range(start, end - win + 1, stride):
        s1 = s0 + win
        for idx in obs_indices:
            idx = idx.to(device=pred.device, dtype=torch.long)
            alpha_signal = torch.mean(alpha_ref[s0:s1, idx], dim=-1)
            pred_signal = torch.mean(pred[s0:s1, idx], dim=-1)
            teacher_signal = torch.mean(teacher[s0:s1, idx], dim=-1)

            alpha_mse = torch.mean((alpha_signal - teacher_signal) ** 2).detach()
            pred_mse = torch.mean((pred_signal - teacher_signal) ** 2)
            response_ratio = pred_mse / alpha_mse.clamp_min(eps)
            response_excess = torch.relu(response_ratio - float(cfg.beta_alpha_response_ratio_limit))

            alpha_corr0, alpha_amp_log_abs = _window_corr_and_amp_log_abs(alpha_signal, teacher_signal, eps)
            pred_corr0, pred_amp_log_abs = _window_corr_and_amp_log_abs(pred_signal, teacher_signal, eps)
            corr_excess = torch.relu(
                alpha_corr0.detach() - pred_corr0 - float(cfg.beta_alpha_corr_drop_tol)
            )
            amp_excess = torch.relu(
                pred_amp_log_abs - alpha_amp_log_abs.detach() - float(cfg.beta_alpha_amp_worsen_tol)
            )

            response_terms.append(response_excess ** 2)
            corr_terms.append(corr_excess ** 2)
            amp_terms.append(amp_excess ** 2)
            response_excesses.append(response_excess.detach())
            corr_excesses.append(corr_excess.detach())
            amp_excesses.append(amp_excess.detach())

    if not response_terms:
        return empty

    response_guard = torch.stack(response_terms).mean()
    corr_guard = torch.stack(corr_terms).mean()
    amp_guard = torch.stack(amp_terms).mean()
    guard_loss = (
        float(cfg.w_beta_alpha_response_guard) * response_guard
        + float(cfg.w_beta_alpha_corr_guard) * corr_guard
        + float(cfg.w_beta_alpha_amp_guard) * amp_guard
    )
    return {
        "beta_alpha_guard_loss": guard_loss,
        "beta_alpha_response_guard_loss": response_guard,
        "beta_alpha_corr_guard_loss": corr_guard,
        "beta_alpha_amp_guard_loss": amp_guard,
        "beta_alpha_response_excess": torch.stack(response_excesses).mean().detach(),
        "beta_alpha_corr_drop_excess": torch.stack(corr_excesses).mean().detach(),
        "beta_alpha_amp_worsen_excess": torch.stack(amp_excesses).mean().detach(),
        "beta_alpha_guard_n_windows": torch.as_tensor(float(len(response_terms)), dtype=pred.dtype, device=pred.device),
    }


def _soft_lag_loss_single_window(
        pred_signal: torch.Tensor,
        teacher_signal: torch.Tensor,
        *,
        dt: float,
        max_lag_seconds: float,
        temperature: float,
        eps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    p = (pred_signal - pred_signal.mean()) / pred_signal.std().clamp_min(eps)
    t = (teacher_signal - teacher_signal.mean()) / teacher_signal.std().clamp_min(eps)
    max_lag_steps = max(1, int(round(float(max_lag_seconds) / float(dt))))
    tau = max(float(temperature), 1.0e-8)
    corrs: list[torch.Tensor] = []
    valid_lags: list[int] = []
    for lag in range(-max_lag_steps, max_lag_steps + 1):
        if lag < 0:
            p_l = p[:lag]
            t_l = t[-lag:]
        elif lag > 0:
            p_l = p[lag:]
            t_l = t[:-lag]
        else:
            p_l = p
            t_l = t
        if p_l.numel() < 8:
            continue
        corrs.append(torch.mean(p_l * t_l))
        valid_lags.append(lag)
    if not corrs:
        zero = pred_signal.sum() * 0.0
        return zero, zero.detach()
    corr_tensor = torch.stack(corrs)
    lag_tensor = torch.as_tensor(valid_lags, dtype=pred_signal.dtype, device=pred_signal.device)
    weights = torch.softmax(corr_tensor / tau, dim=0)
    soft_lag_steps = torch.sum(weights * lag_tensor)
    loss = (soft_lag_steps / float(max_lag_steps)) ** 2
    return loss, (torch.abs(soft_lag_steps) * float(dt)).detach()


def slow_only_branch_diagnosis_loss(
        *,
        pred: torch.Tensor,
        slow: Optional[torch.Tensor],
        teacher: torch.Tensor,
        theta_aux: Optional[dict[str, torch.Tensor]],
        cfg: TransformerPhysicalTrainConfig,
        x_idx: torch.Tensor,
        y_idx: torch.Tensor,
) -> dict[str, torch.Tensor]:
    zero = torch.zeros((), dtype=pred.dtype, device=pred.device)
    empty = {
        "slow_only_diagnosis_loss": zero,
        "slow_good_no_regression_loss": zero,
        "slow_good_fast_suppress_loss": zero,
        "slow_bad_phase_loss": zero,
        "slow_good_window_ratio": zero.detach(),
        "slow_bad_window_ratio": zero.detach(),
        "slow_quality_score_mean": zero.detach(),
        "slow_quality_score_max": zero.detach(),
        "slow_good_gate_mean": zero.detach(),
        "slow_bad_gate_mean": zero.detach(),
        "slow_bad_phase_mean_abs_lag_s": zero.detach(),
        "slow_only_n_windows": zero.detach(),
    }
    if (
        not bool(getattr(cfg, "use_slow_only_branch_diagnosis", False))
        or slow is None
        or theta_aux is None
    ):
        return empty

    gate = theta_aux.get("g_phase")
    gated_fast = theta_aux.get("theta_gated_fast")
    if gate is None or gated_fast is None:
        return empty

    pred = pred.to(dtype=teacher.dtype, device=teacher.device)
    slow = slow.to(dtype=teacher.dtype, device=teacher.device).detach()
    teacher = teacher.to(dtype=pred.dtype, device=pred.device)
    gate = gate.to(dtype=pred.dtype, device=pred.device)
    gated_fast = gated_fast.to(dtype=pred.dtype, device=pred.device)
    if gate.ndim == 3:
        gate = gate[0]
    if gate.ndim == 2:
        gate = torch.mean(gate, dim=-1)
    if gated_fast.ndim == 3:
        gated_fast = gated_fast[0]
    if gate.ndim != 1 or gated_fast.ndim != 2:
        return empty

    T = int(min(pred.shape[0], slow.shape[0], teacher.shape[0], gate.shape[0], gated_fast.shape[0]))
    if T < 2:
        return empty
    pred = pred[:T]
    slow = slow[:T]
    teacher = teacher[:T]
    gate = gate[:T]
    gated_fast = gated_fast[:T]

    dt = float(cfg.dt)
    start = max(0, int(round(float(cfg.static_quality_start) / dt)))
    end = T if cfg.static_quality_end is None else min(T, int(round(float(cfg.static_quality_end) / dt)))
    win = max(8, int(round(float(cfg.static_quality_window_seconds) / dt)))
    stride = max(1, int(round(float(cfg.static_quality_stride_seconds) / dt)))
    if end - start < win:
        return empty

    obs_indices = (
        _static_quality_observation_indices(
            dof_indices=x_idx,
            observations=str(cfg.static_quality_observations),
            last_k=int(cfg.static_quality_last_k),
        )
        + _static_quality_observation_indices(
            dof_indices=y_idx,
            observations=str(cfg.static_quality_observations),
            last_k=int(cfg.static_quality_last_k),
        )
    )
    if not obs_indices:
        return empty

    eps = torch.as_tensor(1.0e-12, dtype=pred.dtype, device=pred.device)
    no_reg_terms: list[torch.Tensor] = []
    suppress_terms: list[torch.Tensor] = []
    bad_phase_terms: list[torch.Tensor] = []
    bad_abs_lags: list[torch.Tensor] = []
    good_flags: list[torch.Tensor] = []
    bad_flags: list[torch.Tensor] = []
    scores: list[torch.Tensor] = []
    gate_means: list[torch.Tensor] = []

    for s0 in range(start, end - win + 1, stride):
        s1 = s0 + win
        gate_mean = torch.mean(gate[s0:s1])
        fast_rms = torch.sqrt(torch.mean(gated_fast[s0:s1, :] ** 2).clamp_min(eps))
        window_scores = []
        window_no_reg = []
        window_bad_phase = []
        window_bad_abs_lag = []

        for idx in obs_indices:
            slow_signal = torch.mean(slow[s0:s1, idx], dim=-1)
            pred_signal = torch.mean(pred[s0:s1, idx], dim=-1)
            teacher_signal = torch.mean(teacher[s0:s1, idx], dim=-1)

            abs_lag_s, slow_best_corr, slow_amp_log_abs = _best_window_lag_corr_amp(
                slow_signal,
                teacher_signal,
                dt=dt,
                max_lag_seconds=float(cfg.static_quality_max_lag_seconds),
                eps=eps,
            )
            corr_excess = torch.relu(
                torch.as_tensor(float(cfg.static_quality_good_corr_threshold), dtype=pred.dtype, device=pred.device)
                - slow_best_corr
            ) / max(1.0 - float(cfg.static_quality_good_corr_threshold), 1.0e-6)
            lag_excess = torch.relu(abs_lag_s - float(cfg.static_quality_good_lag_seconds)) / max(
                float(cfg.static_quality_good_lag_seconds),
                1.0e-6,
            )
            amp_excess = torch.relu(slow_amp_log_abs - float(cfg.static_quality_good_amp_log_tol)) / max(
                float(cfg.static_quality_good_amp_log_tol),
                1.0e-6,
            )
            score = (corr_excess + lag_excess + amp_excess).detach()
            window_scores.append(score)

            slow_mse = torch.mean((slow_signal - teacher_signal) ** 2).detach()
            pred_mse = torch.mean((pred_signal - teacher_signal) ** 2)
            response_excess = torch.relu(
                pred_mse / slow_mse.clamp_min(eps) - float(cfg.slow_good_response_ratio_limit)
            )
            slow_corr0, _ = _window_corr_and_amp_log_abs(slow_signal, teacher_signal, eps)
            pred_corr0, pred_amp_log_abs = _window_corr_and_amp_log_abs(pred_signal, teacher_signal, eps)
            corr_drop = slow_corr0.detach() - pred_corr0
            corr_drop_excess = torch.relu(corr_drop - float(cfg.slow_good_corr_drop_tol))
            amp_excess_pred = torch.relu(pred_amp_log_abs - float(cfg.slow_good_amp_log_tol))
            window_no_reg.append(response_excess ** 2 + corr_drop_excess ** 2 + amp_excess_pred ** 2)

            lag_loss, abs_pred_lag_s = _soft_lag_loss_single_window(
                pred_signal,
                teacher_signal,
                dt=dt,
                max_lag_seconds=float(cfg.phase_window_max_lag_seconds),
                temperature=float(cfg.phase_window_lag_temperature),
                eps=eps,
            )
            bad_weight = torch.clamp(score, min=0.0, max=float(cfg.slow_bad_weight_max))
            window_bad_phase.append(bad_weight * lag_loss)
            window_bad_abs_lag.append(bad_weight * abs_pred_lag_s)

        score_window = torch.stack(window_scores).mean().detach()
        good = (score_window <= 1.0e-8).to(dtype=pred.dtype)
        bad = (score_window > 1.0e-8).to(dtype=pred.dtype)
        no_reg_terms.append(good * torch.stack(window_no_reg).mean())
        suppress_terms.append(good * (gate_mean ** 2 + fast_rms ** 2))
        bad_phase_terms.append(bad * torch.stack(window_bad_phase).mean())
        bad_abs_lags.append(bad * torch.stack(window_bad_abs_lag).mean().detach())
        good_flags.append(good.detach())
        bad_flags.append(bad.detach())
        scores.append(score_window)
        gate_means.append(gate_mean.detach())

    if not scores:
        return empty

    no_reg_loss = torch.stack(no_reg_terms).mean()
    suppress_loss = torch.stack(suppress_terms).mean()
    bad_phase_loss = torch.stack(bad_phase_terms).mean()
    total = (
        float(cfg.w_slow_good_no_regression) * no_reg_loss
        + float(cfg.w_slow_good_fast_suppress) * suppress_loss
        + float(cfg.w_slow_bad_phase) * bad_phase_loss
    )

    good_t = torch.stack(good_flags)
    bad_t = torch.stack(bad_flags)
    gate_t = torch.stack(gate_means)
    good_count = torch.sum(good_t).clamp_min(1.0)
    bad_count = torch.sum(bad_t).clamp_min(1.0)
    return {
        "slow_only_diagnosis_loss": total,
        "slow_good_no_regression_loss": no_reg_loss,
        "slow_good_fast_suppress_loss": suppress_loss,
        "slow_bad_phase_loss": bad_phase_loss,
        "slow_good_window_ratio": torch.mean(good_t).detach(),
        "slow_bad_window_ratio": torch.mean(bad_t).detach(),
        "slow_quality_score_mean": torch.stack(scores).mean().detach(),
        "slow_quality_score_max": torch.stack(scores).max().detach(),
        "slow_good_gate_mean": (torch.sum(gate_t * good_t) / good_count).detach(),
        "slow_bad_gate_mean": (torch.sum(gate_t * bad_t) / bad_count).detach(),
        "slow_bad_phase_mean_abs_lag_s": torch.stack(bad_abs_lags).mean().detach(),
        "slow_only_n_windows": torch.as_tensor(float(len(scores)), dtype=pred.dtype, device=pred.device),
    }


def _demeaned_rms_ratio(
        pred: torch.Tensor,
        target: torch.Tensor,
        dof_indices: torch.Tensor,
        eps: torch.Tensor,
) -> torch.Tensor:
    p = pred[:, dof_indices]
    t = target[:, dof_indices]
    p = p - torch.mean(p, dim=0, keepdim=True)
    t = t - torch.mean(t, dim=0, keepdim=True)
    p_rms = torch.sqrt(torch.mean(p * p).clamp_min(eps))
    t_rms = torch.sqrt(torch.mean(t * t).clamp_min(eps))
    return p_rms / t_rms.clamp_min(eps)


def compute_response_loss(
        *,
        u_pred: torch.Tensor,
        case: TransformerTrainingCase,
        cfg: TransformerPhysicalTrainConfig,
        x_idx: torch.Tensor,
        y_idx: torch.Tensor,
        tip_y_idx: int,
        last5_y_idx: torch.Tensor,
        theta: torch.Tensor,
        theta_aux: Optional[dict[str, torch.Tensor]] = None,
        u_slow: Optional[torch.Tensor] = None,
        u_alpha_ref: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """
    alpha_x / alpha_xy training loss.

    总目标：
        total_loss = response_loss + freq_loss + reg_loss

    response_loss 默认弱权重，freq_loss 默认主导，目的是先降低复杂工况下
    x/y 两个方向的频率 / 模态错位，而不是让模型优先追逐幅值。
    """
    eps = torch.as_tensor(1.0e-12, dtype=u_pred.dtype, device=u_pred.device)

    teacher = case.u_teacher.to(device=u_pred.device, dtype=u_pred.dtype)
    pred = u_pred[0]
    diff = pred - teacher

    x_mse = torch.mean(diff[:, x_idx] ** 2)
    y_mse = torch.mean(diff[:, y_idx] ** 2)
    tip_y_mse = torch.mean(diff[:, tip_y_idx] ** 2)
    last5_y_mse = torch.mean(diff[:, last5_y_idx] ** 2)

    base_x = torch.as_tensor(case.base_x_mse, dtype=u_pred.dtype, device=u_pred.device)
    base_y = torch.as_tensor(case.base_y_mse, dtype=u_pred.dtype, device=u_pred.device)
    base_tip_y = torch.as_tensor(case.base_tip_y_mse, dtype=u_pred.dtype, device=u_pred.device)
    base_last5_y = torch.as_tensor(case.base_last5_y_mse, dtype=u_pred.dtype, device=u_pred.device)

    x_ratio = x_mse / torch.clamp(base_x, min=eps)
    y_ratio = y_mse / torch.clamp(base_y, min=eps)
    tip_y_ratio = tip_y_mse / torch.clamp(base_tip_y, min=eps)
    last5_y_ratio = last5_y_mse / torch.clamp(base_last5_y, min=eps)

    beta_alpha_active = bool(u_alpha_ref is not None and _use_beta_alpha_reference_loss(cfg))
    nan_metric = torch.as_tensor(float("nan"), dtype=u_pred.dtype, device=u_pred.device)
    x_ratio_to_alpha = nan_metric
    y_ratio_to_alpha = nan_metric
    tip_y_ratio_to_alpha = nan_metric
    last5_y_ratio_to_alpha = nan_metric
    if u_alpha_ref is not None:
        alpha_ref = u_alpha_ref[0].to(device=u_pred.device, dtype=u_pred.dtype).detach()
        alpha_diff = alpha_ref - teacher
        alpha_x_mse = torch.mean(alpha_diff[:, x_idx] ** 2)
        alpha_y_mse = torch.mean(alpha_diff[:, y_idx] ** 2)
        alpha_tip_y_mse = torch.mean(alpha_diff[:, tip_y_idx] ** 2)
        alpha_last5_y_mse = torch.mean(alpha_diff[:, last5_y_idx] ** 2)
        x_ratio_to_alpha = x_mse / torch.clamp(alpha_x_mse, min=eps)
        y_ratio_to_alpha = y_mse / torch.clamp(alpha_y_mse, min=eps)
        tip_y_ratio_to_alpha = tip_y_mse / torch.clamp(alpha_tip_y_mse, min=eps)
        last5_y_ratio_to_alpha = last5_y_mse / torch.clamp(alpha_last5_y_mse, min=eps)

    x_guard = torch.relu(x_ratio - (1.0 + float(cfg.x_guard_tol))) ** 2

    response_loss = (
            float(cfg.w_y) * y_ratio
            + float(cfg.w_tip_y) * tip_y_ratio
            + float(cfg.w_last5_y) * last5_y_ratio
            + float(cfg.w_x_guard) * x_guard
            + float(cfg.w_x) * x_ratio
    )
    if beta_alpha_active:
        alpha_x_guard = torch.relu(
            x_ratio_to_alpha - float(cfg.beta_alpha_response_ratio_limit)
        ) ** 2
        response_loss = (
            float(cfg.w_y) * y_ratio_to_alpha
            + float(cfg.w_tip_y) * tip_y_ratio_to_alpha
            + float(cfg.w_last5_y) * last5_y_ratio_to_alpha
            + float(cfg.w_x_guard) * alpha_x_guard
            + float(cfg.w_x) * x_ratio_to_alpha
        )

    if bool(cfg.use_cached_alignment_loss) and case.loss_cache is not None:
        freq_x = frequency_alignment_loss_from_cache(pred=pred, cache=case.loss_cache["freq_x"])
        freq_y = frequency_alignment_loss_from_cache(pred=pred, cache=case.loss_cache["freq_y"])
    else:
        freq_x = frequency_alignment_loss(
            pred=pred,
            target=teacher,
            dt=float(cfg.dt),
            dof_indices=x_idx,
            freq_min=float(cfg.freq_min),
            freq_max=float(cfg.freq_max),
            peak_temperature=float(cfg.freq_peak_temperature),
        )
        freq_y = frequency_alignment_loss(
            pred=pred,
            target=teacher,
            dt=float(cfg.dt),
            dof_indices=y_idx,
            freq_min=float(cfg.freq_min),
            freq_max=float(cfg.freq_max),
            peak_temperature=float(cfg.freq_peak_temperature),
        )

    spectrum_loss = (
            float(cfg.w_spec_x) * freq_x["spec_loss"]
            + float(cfg.w_spec_y) * freq_y["spec_loss"]
            + float(cfg.w_peak_x) * freq_x["peak_loss"]
            + float(cfg.w_peak_y) * freq_y["peak_loss"]
    )

    if bool(cfg.use_cached_alignment_loss) and case.loss_cache is not None:
        align_x = peak_and_lag_alignment_loss_from_cache(pred=pred, cache=case.loss_cache["align_x"])
        align_y = peak_and_lag_alignment_loss_from_cache(pred=pred, cache=case.loss_cache["align_y"])
    else:
        align_x = peak_and_lag_alignment_loss(
            pred=pred,
            target=teacher,
            dt=float(cfg.dt),
            dof_indices=x_idx,
            observations=str(cfg.alignment_observations),
            last_k=int(cfg.alignment_last_k),
            peak_start_time=float(cfg.peak_time_start),
            peak_end_time=cfg.peak_time_end,
            peak_window_seconds=float(cfg.peak_time_window_seconds),
            peak_temperature=float(cfg.peak_time_temperature),
            peak_min_distance_seconds=float(cfg.peak_time_min_distance_seconds),
            peak_prominence_std=float(cfg.peak_time_prominence_std),
            peak_max_events=int(cfg.peak_time_max_events),
            lag_start_time=float(cfg.lag_start),
            lag_end_time=cfg.lag_end,
            lag_window_seconds=float(cfg.lag_window_seconds),
            lag_stride_seconds=float(cfg.lag_stride_seconds),
            max_lag_seconds=float(cfg.max_lag_seconds),
            lag_temperature=float(cfg.lag_temperature),
        )
        align_y = peak_and_lag_alignment_loss(
            pred=pred,
            target=teacher,
            dt=float(cfg.dt),
            dof_indices=y_idx,
            observations=str(cfg.alignment_observations),
            last_k=int(cfg.alignment_last_k),
            peak_start_time=float(cfg.peak_time_start),
            peak_end_time=cfg.peak_time_end,
            peak_window_seconds=float(cfg.peak_time_window_seconds),
            peak_temperature=float(cfg.peak_time_temperature),
            peak_min_distance_seconds=float(cfg.peak_time_min_distance_seconds),
            peak_prominence_std=float(cfg.peak_time_prominence_std),
            peak_max_events=int(cfg.peak_time_max_events),
            lag_start_time=float(cfg.lag_start),
            lag_end_time=cfg.lag_end,
            lag_window_seconds=float(cfg.lag_window_seconds),
            lag_stride_seconds=float(cfg.lag_stride_seconds),
            max_lag_seconds=float(cfg.max_lag_seconds),
            lag_temperature=float(cfg.lag_temperature),
        )

    peak_time_loss = (
            float(cfg.w_peak_time_x) * align_x["peak_time_loss"]
            + float(cfg.w_peak_time_y) * align_y["peak_time_loss"]
    )
    lag_loss = (
            float(cfg.w_lag_x) * align_x["lag_loss"]
            + float(cfg.w_lag_y) * align_y["lag_loss"]
    )

    adaptive_phase = adaptive_phase_window_training_loss(
        pred=pred,
        teacher=teacher,
        static=case.u_static.to(device=u_pred.device, dtype=u_pred.dtype),
        theta_aux=theta_aux,
        cfg=cfg,
        x_idx=x_idx,
        y_idx=y_idx,
    )
    phase_drift = phase_drift_rate_training_loss(
        pred=pred,
        teacher=teacher,
        static=case.u_static.to(device=u_pred.device, dtype=u_pred.dtype),
        cfg=cfg,
        x_idx=x_idx,
        y_idx=y_idx,
    )
    local_band_phase = local_band_phase_training_loss(
        pred=pred,
        teacher=teacher,
        cfg=cfg,
        x_idx=x_idx,
        y_idx=y_idx,
    )
    local_phase_corr = local_phase_correlation_training_loss(
        pred=pred,
        teacher=teacher,
        static=case.u_static.to(device=u_pred.device, dtype=u_pred.dtype),
        cfg=cfg,
        x_idx=x_idx,
        y_idx=y_idx,
    )
    local_phase_increment = local_phase_increment_training_loss(
        pred=pred,
        teacher=teacher,
        static=case.u_static.to(device=u_pred.device, dtype=u_pred.dtype),
        cfg=cfg,
        x_idx=x_idx,
        y_idx=y_idx,
    )
    continuous_phase = continuous_phase_lag_training_loss(
        pred=pred,
        teacher=teacher,
        static=case.u_static.to(device=u_pred.device, dtype=u_pred.dtype),
        cfg=cfg,
        x_idx=x_idx,
        y_idx=y_idx,
    )
    local_phase_slope = local_phase_slope_training_loss(
        pred=pred,
        teacher=teacher,
        static=case.u_static.to(device=u_pred.device, dtype=u_pred.dtype),
        cfg=cfg,
        x_idx=x_idx,
        y_idx=y_idx,
    )
    static_quality_gate = static_quality_gate_suppression_loss(
        static=case.u_static,
        teacher=teacher,
        theta_aux=theta_aux,
        cfg=cfg,
        x_idx=x_idx,
        y_idx=y_idx,
    )
    state_no_regression = state_window_no_regression_guard_loss(
        pred=pred,
        static=case.u_static,
        teacher=teacher,
        cfg=cfg,
        x_idx=x_idx,
        y_idx=y_idx,
    )
    beta_amp = beta_alpha_relative_amplitude_loss(
        pred=pred,
        alpha_ref=None if u_alpha_ref is None else u_alpha_ref[0],
        teacher=teacher,
        theta=theta[0] if theta.ndim == 3 else theta,
        cfg=cfg,
        x_idx=x_idx,
        y_idx=y_idx,
        tip_y_idx=tip_y_idx,
        last5_y_idx=last5_y_idx,
    )
    beta_alpha_guard = beta_alpha_no_regression_guard_loss(
        pred=pred,
        alpha_ref=None if u_alpha_ref is None else u_alpha_ref[0],
        teacher=teacher,
        cfg=cfg,
        x_idx=x_idx,
        y_idx=y_idx,
    )
    slow_only_diag = slow_only_branch_diagnosis_loss(
        pred=pred,
        slow=None if u_slow is None else u_slow[0],
        teacher=teacher,
        theta_aux=theta_aux,
        cfg=cfg,
        x_idx=x_idx,
        y_idx=y_idx,
    )

    # freq_loss 现在表示“频率/峰值/相位综合对齐损失”：
    # 全局频谱 + teacher-anchored peak-time + local cross-correlation lag
    # + adaptive phase-window hard mining + complex spectrum phase loss。
    freq_loss = (
        spectrum_loss
        + peak_time_loss
        + lag_loss
        + adaptive_phase["adaptive_phase_loss"]
        + adaptive_phase["complex_phase_loss"]
        + adaptive_phase["complex_amp_guard_loss"]
        + adaptive_phase["phase_gate_align_loss"]
        + phase_drift["phase_drift_loss"]
        + local_band_phase["local_band_phase_loss"]
        + local_phase_corr["local_phase_corr_loss"]
        + local_phase_increment["local_phase_increment_loss"]
        + continuous_phase["continuous_phase_loss"]
        + local_phase_slope["local_phase_slope_loss"]
        + slow_only_diag["slow_only_diagnosis_loss"]
    )

    amp = theta_amplitude_loss(theta.to(dtype=u_pred.dtype))
    smooth = theta_smoothness_loss(theta.to(dtype=u_pred.dtype))
    phase_reg = phase_gated_decomposition_regularization_loss(
        theta_aux=theta_aux,
        cfg=cfg,
        dtype=u_pred.dtype,
        device=u_pred.device,
    )
    base_reg_loss = float(cfg.w_theta_amp) * amp + float(cfg.w_theta_smooth) * smooth
    reg_loss = base_reg_loss + phase_reg["phase_reg_loss"] + static_quality_gate["static_quality_gate_loss"]

    if beta_alpha_active:
        total_loss = (
            response_loss
            + beta_amp["beta_amp_loss"]
            + reg_loss
            + beta_alpha_guard["beta_alpha_guard_loss"]
        )
    else:
        total_loss = (
            response_loss
            + freq_loss
            + reg_loss
            + state_no_regression["state_no_regression_guard_loss"]
        )

    result = {
        "total_loss": total_loss,
        "response_loss": response_loss,
        "data_loss": response_loss,
        "freq_loss": freq_loss,
        "spectrum_loss": spectrum_loss,
        "peak_time_loss": peak_time_loss,
        "lag_loss": lag_loss,
        "adaptive_phase_loss": adaptive_phase["adaptive_phase_loss"],
        "complex_phase_loss": adaptive_phase["complex_phase_loss"],
        "complex_amp_guard_loss": adaptive_phase["complex_amp_guard_loss"],
        "phase_gate_align_loss": adaptive_phase["phase_gate_align_loss"],
        "phase_drift_loss": phase_drift["phase_drift_loss"],
        "phase_drift_x_lag_loss": phase_drift["phase_drift_x_lag_loss"],
        "phase_drift_y_lag_loss": phase_drift["phase_drift_y_lag_loss"],
        "phase_drift_x_rate_loss": phase_drift["phase_drift_x_rate_loss"],
        "phase_drift_y_rate_loss": phase_drift["phase_drift_y_rate_loss"],
        "phase_drift_x_mean_abs_lag_s": phase_drift["phase_drift_x_mean_abs_lag_s"],
        "phase_drift_y_mean_abs_lag_s": phase_drift["phase_drift_y_mean_abs_lag_s"],
        "phase_drift_x_mean_abs_dlag_s": phase_drift["phase_drift_x_mean_abs_dlag_s"],
        "phase_drift_y_mean_abs_dlag_s": phase_drift["phase_drift_y_mean_abs_dlag_s"],
        "phase_drift_x_high_weight_mean": phase_drift["phase_drift_x_high_weight_mean"],
        "phase_drift_y_high_weight_mean": phase_drift["phase_drift_y_high_weight_mean"],
        "phase_drift_x_amplitude_weight_mean": phase_drift["phase_drift_x_amplitude_weight_mean"],
        "phase_drift_y_amplitude_weight_mean": phase_drift["phase_drift_y_amplitude_weight_mean"],
        "phase_drift_x_static_failure_weight_mean": phase_drift["phase_drift_x_static_failure_weight_mean"],
        "phase_drift_y_static_failure_weight_mean": phase_drift["phase_drift_y_static_failure_weight_mean"],
        "phase_drift_x_combined_weight_mean": phase_drift["phase_drift_x_combined_weight_mean"],
        "phase_drift_y_combined_weight_mean": phase_drift["phase_drift_y_combined_weight_mean"],
        "phase_drift_x_n_windows": phase_drift["phase_drift_x_n_windows"],
        "phase_drift_y_n_windows": phase_drift["phase_drift_y_n_windows"],
        "local_band_phase_loss": local_band_phase["local_band_phase_loss"],
        "local_band_phase_x_loss": local_band_phase["local_band_phase_x_loss"],
        "local_band_phase_y_loss": local_band_phase["local_band_phase_y_loss"],
        "local_band_phase_x_raw_phase_loss": local_band_phase["local_band_phase_x_raw_phase_loss"],
        "local_band_phase_y_raw_phase_loss": local_band_phase["local_band_phase_y_raw_phase_loss"],
        "local_band_phase_x_high_weight_mean": local_band_phase["local_band_phase_x_high_weight_mean"],
        "local_band_phase_y_high_weight_mean": local_band_phase["local_band_phase_y_high_weight_mean"],
        "local_band_phase_x_phase_cos_mean": local_band_phase["local_band_phase_x_phase_cos_mean"],
        "local_band_phase_y_phase_cos_mean": local_band_phase["local_band_phase_y_phase_cos_mean"],
        "local_band_phase_x_n_windows": local_band_phase["local_band_phase_x_n_windows"],
        "local_band_phase_y_n_windows": local_band_phase["local_band_phase_y_n_windows"],
        "local_phase_corr_loss": local_phase_corr["local_phase_corr_loss"],
        "local_phase_corr_x_loss": local_phase_corr["local_phase_corr_x_loss"],
        "local_phase_corr_y_loss": local_phase_corr["local_phase_corr_y_loss"],
        "local_phase_corr_x_corr_loss": local_phase_corr["local_phase_corr_x_corr_loss"],
        "local_phase_corr_y_corr_loss": local_phase_corr["local_phase_corr_y_corr_loss"],
        "local_phase_corr_x_corr_gap_loss": local_phase_corr["local_phase_corr_x_corr_gap_loss"],
        "local_phase_corr_y_corr_gap_loss": local_phase_corr["local_phase_corr_y_corr_gap_loss"],
        "local_phase_corr_x_lag_loss": local_phase_corr["local_phase_corr_x_lag_loss"],
        "local_phase_corr_y_lag_loss": local_phase_corr["local_phase_corr_y_lag_loss"],
        "local_phase_corr_x_corr0_mean": local_phase_corr["local_phase_corr_x_corr0_mean"],
        "local_phase_corr_y_corr0_mean": local_phase_corr["local_phase_corr_y_corr0_mean"],
        "local_phase_corr_x_best_corr_mean": local_phase_corr["local_phase_corr_x_best_corr_mean"],
        "local_phase_corr_y_best_corr_mean": local_phase_corr["local_phase_corr_y_best_corr_mean"],
        "local_phase_corr_x_corr_gap_mean": local_phase_corr["local_phase_corr_x_corr_gap_mean"],
        "local_phase_corr_y_corr_gap_mean": local_phase_corr["local_phase_corr_y_corr_gap_mean"],
        "local_phase_corr_x_mean_abs_lag_s": local_phase_corr["local_phase_corr_x_mean_abs_lag_s"],
        "local_phase_corr_y_mean_abs_lag_s": local_phase_corr["local_phase_corr_y_mean_abs_lag_s"],
        "local_phase_corr_x_high_weight_mean": local_phase_corr["local_phase_corr_x_high_weight_mean"],
        "local_phase_corr_y_high_weight_mean": local_phase_corr["local_phase_corr_y_high_weight_mean"],
        "local_phase_corr_x_static_failure_weight_mean": local_phase_corr["local_phase_corr_x_static_failure_weight_mean"],
        "local_phase_corr_y_static_failure_weight_mean": local_phase_corr["local_phase_corr_y_static_failure_weight_mean"],
        "local_phase_corr_x_combined_weight_mean": local_phase_corr["local_phase_corr_x_combined_weight_mean"],
        "local_phase_corr_y_combined_weight_mean": local_phase_corr["local_phase_corr_y_combined_weight_mean"],
        "local_phase_corr_x_n_windows": local_phase_corr["local_phase_corr_x_n_windows"],
        "local_phase_corr_y_n_windows": local_phase_corr["local_phase_corr_y_n_windows"],
        "local_phase_increment_loss": local_phase_increment["local_phase_increment_loss"],
        "local_phase_increment_x_absolute_loss": local_phase_increment["local_phase_increment_x_absolute_loss"],
        "local_phase_increment_y_absolute_loss": local_phase_increment["local_phase_increment_y_absolute_loss"],
        "local_phase_increment_x_increment_loss": local_phase_increment["local_phase_increment_x_increment_loss"],
        "local_phase_increment_y_increment_loss": local_phase_increment["local_phase_increment_y_increment_loss"],
        "local_phase_increment_x_phase_cos_mean": local_phase_increment["local_phase_increment_x_phase_cos_mean"],
        "local_phase_increment_y_phase_cos_mean": local_phase_increment["local_phase_increment_y_phase_cos_mean"],
        "local_phase_increment_x_increment_cos_mean": local_phase_increment["local_phase_increment_x_increment_cos_mean"],
        "local_phase_increment_y_increment_cos_mean": local_phase_increment["local_phase_increment_y_increment_cos_mean"],
        "local_phase_increment_x_high_weight_mean": local_phase_increment["local_phase_increment_x_high_weight_mean"],
        "local_phase_increment_y_high_weight_mean": local_phase_increment["local_phase_increment_y_high_weight_mean"],
        "local_phase_increment_x_static_failure_weight_mean": local_phase_increment["local_phase_increment_x_static_failure_weight_mean"],
        "local_phase_increment_y_static_failure_weight_mean": local_phase_increment["local_phase_increment_y_static_failure_weight_mean"],
        "local_phase_increment_x_combined_weight_mean": local_phase_increment["local_phase_increment_x_combined_weight_mean"],
        "local_phase_increment_y_combined_weight_mean": local_phase_increment["local_phase_increment_y_combined_weight_mean"],
        "local_phase_increment_x_target_freq_hz_mean": local_phase_increment["local_phase_increment_x_target_freq_hz_mean"],
        "local_phase_increment_y_target_freq_hz_mean": local_phase_increment["local_phase_increment_y_target_freq_hz_mean"],
        "local_phase_increment_x_n_windows": local_phase_increment["local_phase_increment_x_n_windows"],
        "local_phase_increment_y_n_windows": local_phase_increment["local_phase_increment_y_n_windows"],
        "local_phase_increment_x_n_increments": local_phase_increment["local_phase_increment_x_n_increments"],
        "local_phase_increment_y_n_increments": local_phase_increment["local_phase_increment_y_n_increments"],
        "continuous_phase_loss": continuous_phase["continuous_phase_loss"],
        "continuous_phase_x_absolute_loss": continuous_phase["continuous_phase_x_absolute_loss"],
        "continuous_phase_y_absolute_loss": continuous_phase["continuous_phase_y_absolute_loss"],
        "continuous_phase_x_time_shift_loss": continuous_phase["continuous_phase_x_time_shift_loss"],
        "continuous_phase_y_time_shift_loss": continuous_phase["continuous_phase_y_time_shift_loss"],
        "continuous_phase_x_phase_cos_mean": continuous_phase["continuous_phase_x_phase_cos_mean"],
        "continuous_phase_y_phase_cos_mean": continuous_phase["continuous_phase_y_phase_cos_mean"],
        "continuous_phase_x_equivalent_abs_lag_s_mean": continuous_phase["continuous_phase_x_equivalent_abs_lag_s_mean"],
        "continuous_phase_y_equivalent_abs_lag_s_mean": continuous_phase["continuous_phase_y_equivalent_abs_lag_s_mean"],
        "continuous_phase_x_high_weight_mean": continuous_phase["continuous_phase_x_high_weight_mean"],
        "continuous_phase_y_high_weight_mean": continuous_phase["continuous_phase_y_high_weight_mean"],
        "continuous_phase_x_static_failure_weight_mean": continuous_phase["continuous_phase_x_static_failure_weight_mean"],
        "continuous_phase_y_static_failure_weight_mean": continuous_phase["continuous_phase_y_static_failure_weight_mean"],
        "continuous_phase_x_combined_weight_mean": continuous_phase["continuous_phase_x_combined_weight_mean"],
        "continuous_phase_y_combined_weight_mean": continuous_phase["continuous_phase_y_combined_weight_mean"],
        "continuous_phase_x_target_freq_hz_mean": continuous_phase["continuous_phase_x_target_freq_hz_mean"],
        "continuous_phase_y_target_freq_hz_mean": continuous_phase["continuous_phase_y_target_freq_hz_mean"],
        "continuous_phase_x_n_windows": continuous_phase["continuous_phase_x_n_windows"],
        "continuous_phase_y_n_windows": continuous_phase["continuous_phase_y_n_windows"],
        "local_phase_slope_loss": local_phase_slope["local_phase_slope_loss"],
        "local_phase_slope_x_loss": local_phase_slope["local_phase_slope_x_loss"],
        "local_phase_slope_y_loss": local_phase_slope["local_phase_slope_y_loss"],
        "local_phase_slope_x_phase_cos_mean": local_phase_slope["local_phase_slope_x_phase_cos_mean"],
        "local_phase_slope_y_phase_cos_mean": local_phase_slope["local_phase_slope_y_phase_cos_mean"],
        "local_phase_slope_x_equivalent_abs_dlag_s_mean": local_phase_slope["local_phase_slope_x_equivalent_abs_dlag_s_mean"],
        "local_phase_slope_y_equivalent_abs_dlag_s_mean": local_phase_slope["local_phase_slope_y_equivalent_abs_dlag_s_mean"],
        "local_phase_slope_x_high_weight_mean": local_phase_slope["local_phase_slope_x_high_weight_mean"],
        "local_phase_slope_y_high_weight_mean": local_phase_slope["local_phase_slope_y_high_weight_mean"],
        "local_phase_slope_x_static_failure_weight_mean": local_phase_slope["local_phase_slope_x_static_failure_weight_mean"],
        "local_phase_slope_y_static_failure_weight_mean": local_phase_slope["local_phase_slope_y_static_failure_weight_mean"],
        "local_phase_slope_x_combined_weight_mean": local_phase_slope["local_phase_slope_x_combined_weight_mean"],
        "local_phase_slope_y_combined_weight_mean": local_phase_slope["local_phase_slope_y_combined_weight_mean"],
        "local_phase_slope_x_target_freq_hz_mean": local_phase_slope["local_phase_slope_x_target_freq_hz_mean"],
        "local_phase_slope_y_target_freq_hz_mean": local_phase_slope["local_phase_slope_y_target_freq_hz_mean"],
        "local_phase_slope_x_n_windows": local_phase_slope["local_phase_slope_x_n_windows"],
        "local_phase_slope_y_n_windows": local_phase_slope["local_phase_slope_y_n_windows"],
        "local_phase_slope_x_n_slopes": local_phase_slope["local_phase_slope_x_n_slopes"],
        "local_phase_slope_y_n_slopes": local_phase_slope["local_phase_slope_y_n_slopes"],
        "slow_only_diagnosis_loss": slow_only_diag["slow_only_diagnosis_loss"],
        "slow_good_no_regression_loss": slow_only_diag["slow_good_no_regression_loss"],
        "slow_good_fast_suppress_loss": slow_only_diag["slow_good_fast_suppress_loss"],
        "slow_bad_phase_loss": slow_only_diag["slow_bad_phase_loss"],
        "slow_good_window_ratio": slow_only_diag["slow_good_window_ratio"],
        "slow_bad_window_ratio": slow_only_diag["slow_bad_window_ratio"],
        "slow_quality_score_mean": slow_only_diag["slow_quality_score_mean"],
        "slow_quality_score_max": slow_only_diag["slow_quality_score_max"],
        "slow_good_gate_mean": slow_only_diag["slow_good_gate_mean"],
        "slow_bad_gate_mean": slow_only_diag["slow_bad_gate_mean"],
        "slow_bad_phase_mean_abs_lag_s": slow_only_diag["slow_bad_phase_mean_abs_lag_s"],
        "slow_only_n_windows": slow_only_diag["slow_only_n_windows"],
        "state_no_regression_guard_loss": state_no_regression["state_no_regression_guard_loss"],
        "state_no_regression_response_guard_loss": state_no_regression["state_no_regression_response_guard_loss"],
        "state_no_regression_corr_guard_loss": state_no_regression["state_no_regression_corr_guard_loss"],
        "state_no_regression_amp_guard_loss": state_no_regression["state_no_regression_amp_guard_loss"],
        "state_no_regression_good_window_ratio": state_no_regression["state_no_regression_good_window_ratio"],
        "state_no_regression_response_excess": state_no_regression["state_no_regression_response_excess"],
        "state_no_regression_corr_drop_excess": state_no_regression["state_no_regression_corr_drop_excess"],
        "state_no_regression_amp_log_excess": state_no_regression["state_no_regression_amp_log_excess"],
        "state_no_regression_n_windows": state_no_regression["state_no_regression_n_windows"],
        "beta_alpha_loss_mode_active": torch.as_tensor(
            1.0 if beta_alpha_active else 0.0,
            dtype=u_pred.dtype,
            device=u_pred.device,
        ),
        "beta_amp_loss": beta_amp["beta_amp_loss"],
        "beta_amp_x_loss": beta_amp["beta_amp_x_loss"],
        "beta_amp_y_loss": beta_amp["beta_amp_y_loss"],
        "beta_amp_tip_y_loss": beta_amp["beta_amp_tip_y_loss"],
        "beta_amp_last5_y_loss": beta_amp["beta_amp_last5_y_loss"],
        "beta_amp_direct_loss": beta_amp["beta_amp_direct_loss"],
        "beta_amp_improvement_loss": beta_amp["beta_amp_improvement_loss"],
        "beta_amp_x_improvement_loss": beta_amp["beta_amp_x_improvement_loss"],
        "beta_amp_y_improvement_loss": beta_amp["beta_amp_y_improvement_loss"],
        "beta_amp_tip_y_improvement_loss": beta_amp["beta_amp_tip_y_improvement_loss"],
        "beta_amp_last5_y_improvement_loss": beta_amp["beta_amp_last5_y_improvement_loss"],
        "beta_damp_sign_loss": beta_amp["beta_damp_sign_loss"],
        "beta_damp_x_sign_loss": beta_amp["beta_damp_x_sign_loss"],
        "beta_damp_y_sign_loss": beta_amp["beta_damp_y_sign_loss"],
        "beta_amp_pred_log_error_mean": beta_amp["beta_amp_pred_log_error_mean"],
        "beta_amp_alpha_log_error_mean": beta_amp["beta_amp_alpha_log_error_mean"],
        "beta_amp_improvement_margin_mean": beta_amp["beta_amp_improvement_margin_mean"],
        "beta_amp_weight_mean": beta_amp["beta_amp_weight_mean"],
        "beta_amp_n_windows": beta_amp["beta_amp_n_windows"],
        "beta_alpha_guard_loss": beta_alpha_guard["beta_alpha_guard_loss"],
        "beta_alpha_response_guard_loss": beta_alpha_guard["beta_alpha_response_guard_loss"],
        "beta_alpha_corr_guard_loss": beta_alpha_guard["beta_alpha_corr_guard_loss"],
        "beta_alpha_amp_guard_loss": beta_alpha_guard["beta_alpha_amp_guard_loss"],
        "beta_alpha_response_excess": beta_alpha_guard["beta_alpha_response_excess"],
        "beta_alpha_corr_drop_excess": beta_alpha_guard["beta_alpha_corr_drop_excess"],
        "beta_alpha_amp_worsen_excess": beta_alpha_guard["beta_alpha_amp_worsen_excess"],
        "beta_alpha_guard_n_windows": beta_alpha_guard["beta_alpha_guard_n_windows"],
        "reg_loss": reg_loss,
        "x_mse": x_mse,
        "y_mse": y_mse,
        "tip_y_mse": tip_y_mse,
        "last5_y_mse": last5_y_mse,
        "x_ratio": x_ratio,
        "y_ratio": y_ratio,
        "tip_y_ratio": tip_y_ratio,
        "last5_y_ratio": last5_y_ratio,
        "x_ratio_to_alpha": x_ratio_to_alpha,
        "y_ratio_to_alpha": y_ratio_to_alpha,
        "tip_y_ratio_to_alpha": tip_y_ratio_to_alpha,
        "last5_y_ratio_to_alpha": last5_y_ratio_to_alpha,
        "x_guard": x_guard,
        "freq_x_spec_loss": freq_x["spec_loss"],
        "freq_y_spec_loss": freq_y["spec_loss"],
        "freq_x_peak_loss": freq_x["peak_loss"],
        "freq_y_peak_loss": freq_y["peak_loss"],
        "freq_x_pred_peak_hz": freq_x["pred_peak_hz"],
        "freq_x_target_peak_hz": freq_x["target_peak_hz"],
        "freq_y_pred_peak_hz": freq_y["pred_peak_hz"],
        "freq_y_target_peak_hz": freq_y["target_peak_hz"],
        "align_x_peak_time_loss": align_x["peak_time_loss"],
        "align_y_peak_time_loss": align_y["peak_time_loss"],
        "align_x_lag_loss": align_x["lag_loss"],
        "align_y_lag_loss": align_y["lag_loss"],
        "align_x_mean_abs_peak_time_error_s": align_x["mean_abs_peak_time_error_s"],
        "align_y_mean_abs_peak_time_error_s": align_y["mean_abs_peak_time_error_s"],
        "align_x_mean_abs_lag_s": align_x["mean_abs_lag_s"],
        "align_y_mean_abs_lag_s": align_y["mean_abs_lag_s"],
        "align_x_n_peak_events": align_x["n_peak_events"],
        "align_y_n_peak_events": align_y["n_peak_events"],
        "align_x_n_lag_windows": align_x["n_lag_windows"],
        "align_y_n_lag_windows": align_y["n_lag_windows"],
        "theta_amp": amp,
        "theta_smooth": smooth,
        "phase_reg_loss": phase_reg["phase_reg_loss"],
        "theta_slow_smooth": phase_reg["theta_slow_smooth"],
        "theta_fast_amp": phase_reg["theta_fast_amp"],
        "theta_fast_smooth": phase_reg["theta_fast_smooth"],
        "theta_fast_window_mean": phase_reg["theta_fast_window_mean"],
        "theta_gated_fast_window_mean": phase_reg["theta_gated_fast_window_mean"],
        "phase_gate_l1": phase_reg["phase_gate_l1"],
        "phase_gate_tv": phase_reg["phase_gate_tv"],
        "phase_gate_bootstrap_loss": phase_reg["phase_gate_bootstrap_loss"],
        "phase_gate_bootstrap_deficit": phase_reg["phase_gate_bootstrap_deficit"],
        "phase_gate_mean": phase_reg["phase_gate_mean"],
        "phase_gate_max": phase_reg["phase_gate_max"],
        "phase_gate_active_ratio": phase_reg["phase_gate_active_ratio"],
        "theta_fast_abs_max": phase_reg["theta_fast_abs_max"],
        "theta_fast_window_mean_abs_max": phase_reg["theta_fast_window_mean_abs_max"],
        "theta_gated_fast_window_mean_abs_max": phase_reg["theta_gated_fast_window_mean_abs_max"],
        "theta_gated_fast_rms": phase_reg["theta_gated_fast_rms"],
        "theta_gated_fast_abs_max": phase_reg["theta_gated_fast_abs_max"],
        **static_quality_gate,
        **{
            key: adaptive_phase[key]
            for key in ADAPTIVE_PHASE_LOG_METRIC_KEYS
            if key in adaptive_phase and key not in {
                "adaptive_phase_loss",
                "complex_phase_loss",
                "complex_amp_guard_loss",
                "phase_gate_align_loss",
            }
        },
    }
    return result


def compute_slow_only_rollout_for_diagnosis(
        *,
        model: TransformerPhysicalRolloutTorch,
        out_theta_aux: Optional[dict[str, torch.Tensor]],
        cfg: TransformerPhysicalTrainConfig,
        u_static: torch.Tensor,
        v_static: torch.Tensor,
        a_static: torch.Tensor,
        F: torch.Tensor,
        u0: torch.Tensor,
        v0: torch.Tensor,
        a0: torch.Tensor,
) -> Optional[torch.Tensor]:
    if not bool(getattr(cfg, "use_slow_only_branch_diagnosis", False)):
        return None
    if out_theta_aux is None or out_theta_aux.get("theta_slow") is None:
        return None
    theta_slow = out_theta_aux["theta_slow"].detach()
    with torch.no_grad():
        u_slow, _, _ = model.rollout_with_theta_sequence(
            theta_seq=theta_slow,
            u_static=u_static,
            v_static=v_static,
            a_static=a_static,
            F=F,
            u0=u0,
            v0=v0,
            a0=a0,
        )
    return u_slow


def compute_alpha_reference_rollout_for_beta_loss(
        *,
        model: TransformerPhysicalRolloutTorch,
        theta: torch.Tensor,
        cfg: TransformerPhysicalTrainConfig,
        u_static: torch.Tensor,
        v_static: torch.Tensor,
        a_static: torch.Tensor,
        F: torch.Tensor,
        u0: torch.Tensor,
        v0: torch.Tensor,
        a0: torch.Tensor,
) -> Optional[torch.Tensor]:
    if not _use_beta_alpha_reference_loss(cfg):
        return None
    theta_ref = _alpha_reference_theta_sequence(theta, model.physical_core.registry)
    if theta_ref is None:
        return None
    with torch.no_grad():
        u_alpha, _, _ = model.rollout_with_theta_sequence(
            theta_seq=theta_ref,
            u_static=u_static,
            v_static=v_static,
            a_static=a_static,
            F=F,
            u0=u0,
            v0=v0,
            a0=a0,
        )
    return u_alpha.detach()


def evaluate_cases(
        *,
        model: TransformerPhysicalRolloutTorch,
        cases: list[TransformerTrainingCase],
        geometry: torch.Tensor,
        cfg: TransformerPhysicalTrainConfig,
        x_idx: torch.Tensor,
        y_idx: torch.Tensor,
        tip_y_idx: int,
        last5_y_idx: torch.Tensor,
        require_grad: bool,
) -> dict[str, Any]:
    if len(cases) == 0:
        zero = torch.zeros((), dtype=torch.float64)
        return {
            "total_loss": zero,
            "data_loss": zero,
            "reg_loss": zero,
            "x_ratio": zero,
            "y_ratio": zero,
            "tip_y_ratio": zero,
            "last5_y_ratio": zero,
            "theta_abs_max": zero,
            **{k: zero for k in PHASE_GATED_LOG_METRIC_KEYS},
            **{k: zero for k in ADAPTIVE_PHASE_LOG_METRIC_KEYS},
            **{k: zero for k in NO_REGRESSION_LOG_METRIC_KEYS},
            **{k: zero for k in BETA_ALPHA_LOG_METRIC_KEYS},
            **{k: 0.0 for k in TIMING_LOG_METRIC_KEYS},
            "num_cases": 0,
        }

    context = torch.enable_grad() if require_grad else torch.inference_mode()
    profile_timing = bool(getattr(cfg, "profile_train_timing", False))
    timing_sums: dict[str, float] = {k: 0.0 for k in TIMING_LOG_METRIC_KEYS}
    total_t0 = _time_now(cfg, next(model.parameters()).device) if profile_timing else 0.0

    with context:
        losses = []
        data_losses = []
        response_losses = []
        freq_losses = []
        spectrum_losses = []
        peak_time_losses = []
        lag_losses = []
        reg_losses = []
        freq_x_spec_losses = []
        freq_y_spec_losses = []
        freq_x_peak_losses = []
        freq_y_peak_losses = []
        freq_x_pred_peaks = []
        freq_x_target_peaks = []
        freq_y_pred_peaks = []
        freq_y_target_peaks = []
        align_x_peak_time_losses = []
        align_y_peak_time_losses = []
        align_x_lag_losses = []
        align_y_lag_losses = []
        align_x_peak_time_errors = []
        align_y_peak_time_errors = []
        align_x_lag_errors = []
        align_y_lag_errors = []
        align_x_n_peak_events = []
        align_y_n_peak_events = []
        align_x_n_lag_windows = []
        align_y_n_lag_windows = []
        x_ratios = []
        y_ratios = []
        tip_y_ratios = []
        last5_y_ratios = []
        x_guards = []
        theta_abs_max_values = []
        phase_metric_values: dict[str, list[torch.Tensor]] = {
            k: [] for k in PHASE_GATED_LOG_METRIC_KEYS
        }
        adaptive_metric_values: dict[str, list[torch.Tensor]] = {
            k: [] for k in ADAPTIVE_PHASE_LOG_METRIC_KEYS
        }
        no_regression_metric_values: dict[str, list[torch.Tensor]] = {
            k: [] for k in NO_REGRESSION_LOG_METRIC_KEYS
        }
        beta_alpha_metric_values: dict[str, list[torch.Tensor]] = {
            k: [] for k in BETA_ALPHA_LOG_METRIC_KEYS
        }

        for case in cases:
            # 当前训练逐 case 处理，batch size = 1。
            u_static = case.u_static.unsqueeze(0)
            v_static = case.v_static.unsqueeze(0)
            a_static = case.a_static.unsqueeze(0)
            F = case.F_raw.unsqueeze(0)
            F_spectral = case.F_spectral.unsqueeze(0) if case.F_spectral is not None else None

            u0 = case.u_static[:1, :].to(dtype=torch.float32)
            v0 = case.v_static[:1, :].to(dtype=torch.float32)
            a0 = case.a_static[:1, :].to(dtype=torch.float32)

            t0 = _time_now(cfg, next(model.parameters()).device) if profile_timing else 0.0
            out = model(
                u_static=u_static,
                v_static=v_static,
                a_static=a_static,
                F=F,
                geometry_features=geometry,
                load_spectral_features=F_spectral,
                u0=u0,
                v0=v0,
                a0=a0,
            )
            if profile_timing:
                timing_sums["timing_model_forward_seconds"] += _time_now(cfg, next(model.parameters()).device) - t0
                for src_key, dst_key in (
                    ("encoder_seconds", "timing_encoder_seconds"),
                    ("core_prepare_seconds", "timing_core_prepare_seconds"),
                    ("newmark_loop_seconds", "timing_newmark_loop_seconds"),
                    ("newmark_assemble_seconds", "timing_newmark_assemble_seconds"),
                    ("newmark_rhs_seconds", "timing_newmark_rhs_seconds"),
                    ("newmark_solve_seconds", "timing_newmark_solve_seconds"),
                    ("newmark_update_seconds", "timing_newmark_update_seconds"),
                    ("state_stack_seconds", "timing_state_stack_seconds"),
                ):
                    timing_sums[dst_key] += float(out.metadata.get(src_key, 0.0))

            t0 = _time_now(cfg, next(model.parameters()).device) if profile_timing else 0.0
            u_slow = compute_slow_only_rollout_for_diagnosis(
                model=model,
                out_theta_aux=out.theta_aux,
                cfg=cfg,
                u_static=u_static,
                v_static=v_static,
                a_static=a_static,
                F=F,
                u0=u0,
                v0=v0,
                a0=a0,
            )
            u_alpha_ref = compute_alpha_reference_rollout_for_beta_loss(
                model=model,
                theta=out.theta,
                cfg=cfg,
                u_static=u_static,
                v_static=v_static,
                a_static=a_static,
                F=F,
                u0=u0,
                v0=v0,
                a0=a0,
            )
            loss_dict = compute_response_loss(
                u_pred=out.u_pred,
                case=case,
                cfg=cfg,
                x_idx=x_idx,
                y_idx=y_idx,
                tip_y_idx=tip_y_idx,
                last5_y_idx=last5_y_idx,
                theta=out.theta,
                theta_aux=out.theta_aux,
                u_slow=u_slow,
                u_alpha_ref=u_alpha_ref,
            )
            if profile_timing:
                timing_sums["timing_loss_seconds"] += _time_now(cfg, next(model.parameters()).device) - t0

            t0 = _time_now(cfg, next(model.parameters()).device) if profile_timing else 0.0
            losses.append(loss_dict["total_loss"])
            data_losses.append(loss_dict["data_loss"])
            response_losses.append(loss_dict["response_loss"])
            freq_losses.append(loss_dict["freq_loss"])
            spectrum_losses.append(loss_dict["spectrum_loss"])
            peak_time_losses.append(loss_dict["peak_time_loss"])
            lag_losses.append(loss_dict["lag_loss"])
            reg_losses.append(loss_dict["reg_loss"])
            freq_x_spec_losses.append(loss_dict["freq_x_spec_loss"])
            freq_y_spec_losses.append(loss_dict["freq_y_spec_loss"])
            freq_x_peak_losses.append(loss_dict["freq_x_peak_loss"])
            freq_y_peak_losses.append(loss_dict["freq_y_peak_loss"])
            freq_x_pred_peaks.append(loss_dict["freq_x_pred_peak_hz"])
            freq_x_target_peaks.append(loss_dict["freq_x_target_peak_hz"])
            freq_y_pred_peaks.append(loss_dict["freq_y_pred_peak_hz"])
            freq_y_target_peaks.append(loss_dict["freq_y_target_peak_hz"])
            align_x_peak_time_losses.append(loss_dict["align_x_peak_time_loss"])
            align_y_peak_time_losses.append(loss_dict["align_y_peak_time_loss"])
            align_x_lag_losses.append(loss_dict["align_x_lag_loss"])
            align_y_lag_losses.append(loss_dict["align_y_lag_loss"])
            align_x_peak_time_errors.append(loss_dict["align_x_mean_abs_peak_time_error_s"])
            align_y_peak_time_errors.append(loss_dict["align_y_mean_abs_peak_time_error_s"])
            align_x_lag_errors.append(loss_dict["align_x_mean_abs_lag_s"])
            align_y_lag_errors.append(loss_dict["align_y_mean_abs_lag_s"])
            align_x_n_peak_events.append(loss_dict["align_x_n_peak_events"])
            align_y_n_peak_events.append(loss_dict["align_y_n_peak_events"])
            align_x_n_lag_windows.append(loss_dict["align_x_n_lag_windows"])
            align_y_n_lag_windows.append(loss_dict["align_y_n_lag_windows"])
            x_ratios.append(loss_dict["x_ratio"])
            y_ratios.append(loss_dict["y_ratio"])
            tip_y_ratios.append(loss_dict["tip_y_ratio"])
            last5_y_ratios.append(loss_dict["last5_y_ratio"])
            x_guards.append(loss_dict["x_guard"])
            theta_abs_max_values.append(torch.max(torch.abs(out.theta.to(dtype=loss_dict["total_loss"].dtype))))
            missing_metric_default = torch.zeros(
                (),
                dtype=loss_dict["total_loss"].dtype,
                device=loss_dict["total_loss"].device,
            )
            for k in PHASE_GATED_LOG_METRIC_KEYS:
                phase_metric_values[k].append(loss_dict.get(k, missing_metric_default))
            for k in ADAPTIVE_PHASE_LOG_METRIC_KEYS:
                adaptive_metric_values[k].append(loss_dict.get(k, missing_metric_default))
            for k in NO_REGRESSION_LOG_METRIC_KEYS:
                no_regression_metric_values[k].append(loss_dict.get(k, missing_metric_default))
            for k in BETA_ALPHA_LOG_METRIC_KEYS:
                beta_alpha_metric_values[k].append(loss_dict.get(k, missing_metric_default))
            if profile_timing:
                timing_sums["timing_metric_accum_seconds"] += _time_now(cfg, next(model.parameters()).device) - t0

        total_loss = torch.stack(losses).mean()
        data_loss = torch.stack(data_losses).mean()
        response_loss = torch.stack(response_losses).mean()
        freq_loss = torch.stack(freq_losses).mean()
        spectrum_loss = torch.stack(spectrum_losses).mean()
        peak_time_loss = torch.stack(peak_time_losses).mean()
        lag_loss = torch.stack(lag_losses).mean()
        reg_loss = torch.stack(reg_losses).mean()
        freq_x_spec_loss = torch.stack(freq_x_spec_losses).mean()
        freq_y_spec_loss = torch.stack(freq_y_spec_losses).mean()
        freq_x_peak_loss = torch.stack(freq_x_peak_losses).mean()
        freq_y_peak_loss = torch.stack(freq_y_peak_losses).mean()
        freq_x_pred_peak_hz = torch.stack(freq_x_pred_peaks).mean()
        freq_x_target_peak_hz = torch.stack(freq_x_target_peaks).mean()
        freq_y_pred_peak_hz = torch.stack(freq_y_pred_peaks).mean()
        freq_y_target_peak_hz = torch.stack(freq_y_target_peaks).mean()
        align_x_peak_time_loss = torch.stack(align_x_peak_time_losses).mean()
        align_y_peak_time_loss = torch.stack(align_y_peak_time_losses).mean()
        align_x_lag_loss = torch.stack(align_x_lag_losses).mean()
        align_y_lag_loss = torch.stack(align_y_lag_losses).mean()
        align_x_mean_abs_peak_time_error_s = torch.stack(align_x_peak_time_errors).mean()
        align_y_mean_abs_peak_time_error_s = torch.stack(align_y_peak_time_errors).mean()
        align_x_mean_abs_lag_s = torch.stack(align_x_lag_errors).mean()
        align_y_mean_abs_lag_s = torch.stack(align_y_lag_errors).mean()
        align_x_n_peak_events = torch.stack(align_x_n_peak_events).mean()
        align_y_n_peak_events = torch.stack(align_y_n_peak_events).mean()
        align_x_n_lag_windows = torch.stack(align_x_n_lag_windows).mean()
        align_y_n_lag_windows = torch.stack(align_y_n_lag_windows).mean()
        x_ratio = torch.stack(x_ratios).mean()
        y_ratio = torch.stack(y_ratios).mean()
        tip_y_ratio = torch.stack(tip_y_ratios).mean()
        last5_y_ratio = torch.stack(last5_y_ratios).mean()
        x_guard = torch.stack(x_guards).mean()
        theta_abs_max = torch.stack(theta_abs_max_values).max()
        phase_metrics = {}
        for k, values in phase_metric_values.items():
            stacked = torch.stack(values)
            if k in PHASE_GATED_MAX_METRIC_KEYS:
                phase_metrics[k] = stacked.max()
            else:
                phase_metrics[k] = stacked.mean()
        adaptive_metrics = {}
        for k, values in adaptive_metric_values.items():
            stacked = torch.stack(values)
            if k in ADAPTIVE_PHASE_MAX_METRIC_KEYS:
                adaptive_metrics[k] = stacked.max()
            elif k in ADAPTIVE_PHASE_MIN_METRIC_KEYS:
                adaptive_metrics[k] = stacked.min()
            else:
                adaptive_metrics[k] = stacked.mean()
        no_regression_metrics = {
            k: torch.stack(values).mean()
            for k, values in no_regression_metric_values.items()
        }
        beta_alpha_metrics = {
            k: torch.stack(values).mean()
            for k, values in beta_alpha_metric_values.items()
        }

    result = {
        "total_loss": total_loss,
        "data_loss": data_loss,
        "response_loss": response_loss,
        "freq_loss": freq_loss,
        "spectrum_loss": spectrum_loss,
        "peak_time_loss": peak_time_loss,
        "lag_loss": lag_loss,
        "reg_loss": reg_loss,
        "freq_x_spec_loss": freq_x_spec_loss,
        "freq_y_spec_loss": freq_y_spec_loss,
        "freq_x_peak_loss": freq_x_peak_loss,
        "freq_y_peak_loss": freq_y_peak_loss,
        "freq_x_pred_peak_hz": freq_x_pred_peak_hz,
        "freq_x_target_peak_hz": freq_x_target_peak_hz,
        "freq_y_pred_peak_hz": freq_y_pred_peak_hz,
        "freq_y_target_peak_hz": freq_y_target_peak_hz,
        "align_x_peak_time_loss": align_x_peak_time_loss,
        "align_y_peak_time_loss": align_y_peak_time_loss,
        "align_x_lag_loss": align_x_lag_loss,
        "align_y_lag_loss": align_y_lag_loss,
        "align_x_mean_abs_peak_time_error_s": align_x_mean_abs_peak_time_error_s,
        "align_y_mean_abs_peak_time_error_s": align_y_mean_abs_peak_time_error_s,
        "align_x_mean_abs_lag_s": align_x_mean_abs_lag_s,
        "align_y_mean_abs_lag_s": align_y_mean_abs_lag_s,
        "align_x_n_peak_events": align_x_n_peak_events,
        "align_y_n_peak_events": align_y_n_peak_events,
        "align_x_n_lag_windows": align_x_n_lag_windows,
        "align_y_n_lag_windows": align_y_n_lag_windows,
        "x_ratio": x_ratio,
        "y_ratio": y_ratio,
        "tip_y_ratio": tip_y_ratio,
        "last5_y_ratio": last5_y_ratio,
        "x_guard": x_guard,
        "theta_abs_max": theta_abs_max,
        **phase_metrics,
        **adaptive_metrics,
        **no_regression_metrics,
        **beta_alpha_metrics,
        **timing_sums,
        "num_cases": len(cases),
    }
    if profile_timing:
        result["timing_total_seconds"] = _time_now(cfg, next(model.parameters()).device) - total_t0
    return result


def train_cases_grad_accum(
        *,
        model: TransformerPhysicalRolloutTorch,
        cases: list[TransformerTrainingCase],
        geometry: torch.Tensor,
        cfg: TransformerPhysicalTrainConfig,
        x_idx: torch.Tensor,
        y_idx: torch.Tensor,
        tip_y_idx: int,
        last5_y_idx: torch.Tensor,
        optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    """
    Memory-friendly training evaluation.

    与原来的训练目标保持一致：
        loss = mean(loss_i over train cases)

    但实现方式改为：
        每个 case forward 后立即 backward(loss_i / n_cases)，
        不再一次性保留所有 case 的计算图。

    注意：
        optimizer.step() 仍然只在所有 case backward 完成后执行一次。
        因此这不是 per-case SGD，而是 gradient accumulation。
    """
    if len(cases) == 0:
        raise RuntimeError("No training cases for train_cases_grad_accum().")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    n_cases = len(cases)
    profile_timing = bool(getattr(cfg, "profile_train_timing", False))
    timing_device = next(model.parameters()).device
    timing_sums: dict[str, float] = {k: 0.0 for k in TIMING_LOG_METRIC_KEYS}
    total_t0 = _time_now(cfg, timing_device) if profile_timing else 0.0

    metric_keys = [
        "total_loss",
        "data_loss",
        "response_loss",
        "freq_loss",
        "spectrum_loss",
        "peak_time_loss",
        "lag_loss",
        "reg_loss",
        "freq_x_spec_loss",
        "freq_y_spec_loss",
        "freq_x_peak_loss",
        "freq_y_peak_loss",
        "freq_x_pred_peak_hz",
        "freq_x_target_peak_hz",
        "freq_y_pred_peak_hz",
        "freq_y_target_peak_hz",
        "align_x_peak_time_loss",
        "align_y_peak_time_loss",
        "align_x_lag_loss",
        "align_y_lag_loss",
        "align_x_mean_abs_peak_time_error_s",
        "align_y_mean_abs_peak_time_error_s",
        "align_x_mean_abs_lag_s",
        "align_y_mean_abs_lag_s",
        "align_x_n_peak_events",
        "align_y_n_peak_events",
        "align_x_n_lag_windows",
        "align_y_n_lag_windows",
        "x_ratio",
        "y_ratio",
        "tip_y_ratio",
        "last5_y_ratio",
        "x_guard",
    ]
    phase_metric_mean_keys = [
        k for k in PHASE_GATED_LOG_METRIC_KEYS
        if k not in PHASE_GATED_MAX_METRIC_KEYS
    ]
    phase_metric_max_keys = [
        k for k in PHASE_GATED_LOG_METRIC_KEYS
        if k in PHASE_GATED_MAX_METRIC_KEYS
    ]
    adaptive_metric_mean_keys = [
        k for k in ADAPTIVE_PHASE_LOG_METRIC_KEYS
        if k not in ADAPTIVE_PHASE_MAX_METRIC_KEYS
        and k not in ADAPTIVE_PHASE_MIN_METRIC_KEYS
    ]
    adaptive_metric_max_keys = [
        k for k in ADAPTIVE_PHASE_LOG_METRIC_KEYS
        if k in ADAPTIVE_PHASE_MAX_METRIC_KEYS
    ]
    adaptive_metric_min_keys = [
        k for k in ADAPTIVE_PHASE_LOG_METRIC_KEYS
        if k in ADAPTIVE_PHASE_MIN_METRIC_KEYS
    ]
    metric_keys = (
        metric_keys
        + phase_metric_mean_keys
        + adaptive_metric_mean_keys
        + list(NO_REGRESSION_LOG_METRIC_KEYS)
        + list(BETA_ALPHA_LOG_METRIC_KEYS)
    )

    # Keep logging reductions on device during the per-case backward loop.
    # Pulling every scalar to CPU after every case forces many CUDA sync points
    # and can dominate small/medium training runs without changing gradients.
    metric_sums: dict[str, Optional[torch.Tensor]] = {k: None for k in metric_keys}
    phase_metric_max_values: dict[str, Optional[torch.Tensor]] = {k: None for k in phase_metric_max_keys}
    adaptive_metric_max_values: dict[str, Optional[torch.Tensor]] = {k: None for k in adaptive_metric_max_keys}
    adaptive_metric_min_values: dict[str, Optional[torch.Tensor]] = {k: None for k in adaptive_metric_min_keys}
    theta_abs_max_value: Optional[torch.Tensor] = None
    for case in cases:
        u_static = case.u_static.unsqueeze(0)
        v_static = case.v_static.unsqueeze(0)
        a_static = case.a_static.unsqueeze(0)
        F = case.F_raw.unsqueeze(0)
        F_spectral = case.F_spectral.unsqueeze(0) if case.F_spectral is not None else None

        u0 = case.u_static[:1, :].to(dtype=torch.float32)
        v0 = case.v_static[:1, :].to(dtype=torch.float32)
        a0 = case.a_static[:1, :].to(dtype=torch.float32)

        t0 = _time_now(cfg, timing_device) if profile_timing else 0.0
        out = model(
            u_static=u_static,
            v_static=v_static,
            a_static=a_static,
            F=F,
            geometry_features=geometry,
            load_spectral_features=F_spectral,
            u0=u0,
            v0=v0,
            a0=a0,
        )
        if profile_timing:
            timing_sums["timing_model_forward_seconds"] += _time_now(cfg, timing_device) - t0
            for src_key, dst_key in (
                ("encoder_seconds", "timing_encoder_seconds"),
                ("core_prepare_seconds", "timing_core_prepare_seconds"),
                ("newmark_loop_seconds", "timing_newmark_loop_seconds"),
                ("newmark_assemble_seconds", "timing_newmark_assemble_seconds"),
                ("newmark_rhs_seconds", "timing_newmark_rhs_seconds"),
                ("newmark_solve_seconds", "timing_newmark_solve_seconds"),
                ("newmark_update_seconds", "timing_newmark_update_seconds"),
                ("state_stack_seconds", "timing_state_stack_seconds"),
            ):
                timing_sums[dst_key] += float(out.metadata.get(src_key, 0.0))

        t0 = _time_now(cfg, timing_device) if profile_timing else 0.0
        u_slow = compute_slow_only_rollout_for_diagnosis(
            model=model,
            out_theta_aux=out.theta_aux,
            cfg=cfg,
            u_static=u_static,
            v_static=v_static,
            a_static=a_static,
            F=F,
            u0=u0,
            v0=v0,
            a0=a0,
        )
        u_alpha_ref = compute_alpha_reference_rollout_for_beta_loss(
            model=model,
            theta=out.theta,
            cfg=cfg,
            u_static=u_static,
            v_static=v_static,
            a_static=a_static,
            F=F,
            u0=u0,
            v0=v0,
            a0=a0,
        )
        loss_dict = compute_response_loss(
            u_pred=out.u_pred,
            case=case,
            cfg=cfg,
            x_idx=x_idx,
            y_idx=y_idx,
            tip_y_idx=tip_y_idx,
            last5_y_idx=last5_y_idx,
                theta=out.theta,
                theta_aux=out.theta_aux,
                u_slow=u_slow,
                u_alpha_ref=u_alpha_ref,
            )

        # 关键：除以 n_cases，保证等价于 mean loss 再 backward。
        if profile_timing:
            timing_sums["timing_loss_seconds"] += _time_now(cfg, timing_device) - t0

        scaled_loss = loss_dict["total_loss"] / float(n_cases)
        t0 = _time_now(cfg, timing_device) if profile_timing else 0.0
        scaled_loss.backward()
        if profile_timing:
            timing_sums["timing_backward_seconds"] += _time_now(cfg, timing_device) - t0

        t0 = _time_now(cfg, timing_device) if profile_timing else 0.0
        missing_metric_default = torch.zeros(
            (),
            dtype=loss_dict["total_loss"].dtype,
            device=loss_dict["total_loss"].device,
        )
        for k in metric_keys:
            value = loss_dict.get(k, missing_metric_default).detach().to(dtype=torch.float64)
            if metric_sums[k] is None:
                metric_sums[k] = value.clone()
            else:
                metric_sums[k] = metric_sums[k] + value
        for k in phase_metric_max_keys:
            value = loss_dict.get(k, missing_metric_default).detach().to(dtype=torch.float64)
            phase_metric_max_values[k] = (
                value.clone()
                if phase_metric_max_values[k] is None
                else torch.maximum(phase_metric_max_values[k], value)
            )
        for k in adaptive_metric_max_keys:
            value = loss_dict.get(k, missing_metric_default).detach().to(dtype=torch.float64)
            adaptive_metric_max_values[k] = (
                value.clone()
                if adaptive_metric_max_values[k] is None
                else torch.maximum(adaptive_metric_max_values[k], value)
            )
        for k in adaptive_metric_min_keys:
            value = loss_dict.get(k, missing_metric_default).detach().to(dtype=torch.float64)
            adaptive_metric_min_values[k] = (
                value.clone()
                if adaptive_metric_min_values[k] is None
                else torch.minimum(adaptive_metric_min_values[k], value)
            )

        current_theta_abs_max = torch.max(torch.abs(out.theta.detach())).to(dtype=torch.float64)
        theta_abs_max_value = (
            current_theta_abs_max.clone()
            if theta_abs_max_value is None
            else torch.maximum(theta_abs_max_value, current_theta_abs_max)
        )

        # 显式删除大对象，帮助 CUDA 更早释放计算图引用。
        if profile_timing:
            timing_sums["timing_metric_accum_seconds"] += _time_now(cfg, timing_device) - t0

        del out
        del u_slow
        del u_alpha_ref
        del loss_dict
        del scaled_loss

    t0 = _time_now(cfg, timing_device) if profile_timing else 0.0
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.encoder.parameters(),
        max_norm=float(cfg.grad_clip_norm),
    )
    if profile_timing:
        timing_sums["timing_grad_clip_seconds"] = _time_now(cfg, timing_device) - t0

    t0 = _time_now(cfg, timing_device) if profile_timing else 0.0
    optimizer.step()
    if profile_timing:
        timing_sums["timing_optimizer_step_seconds"] = _time_now(cfg, timing_device) - t0

    dtype_out = torch.float64
    result: dict[str, Any] = {}

    for k in metric_keys:
        value = metric_sums[k]
        result[k] = (
            torch.as_tensor(float("nan"), dtype=dtype_out)
            if value is None
            else (value / float(n_cases)).to(dtype=dtype_out)
        )
    for k, value in phase_metric_max_values.items():
        result[k] = (
            torch.as_tensor(float("nan"), dtype=dtype_out)
            if value is None
            else value.to(dtype=dtype_out)
        )
    for k, value in adaptive_metric_max_values.items():
        result[k] = (
            torch.as_tensor(float("nan"), dtype=dtype_out)
            if value is None
            else value.to(dtype=dtype_out)
        )
    for k, value in adaptive_metric_min_values.items():
        result[k] = (
            torch.as_tensor(float("nan"), dtype=dtype_out)
            if value is None
            else value.to(dtype=dtype_out)
        )

    result["theta_abs_max"] = (
        torch.as_tensor(float("nan"), dtype=dtype_out)
        if theta_abs_max_value is None
        else theta_abs_max_value.to(dtype=dtype_out)
    )
    result["num_cases"] = n_cases
    result["grad_norm"] = (
        float(grad_norm.detach().cpu())
        if torch.is_tensor(grad_norm)
        else float(grad_norm)
    )
    for k, value in timing_sums.items():
        result[k] = float(value)
    if profile_timing:
        result["timing_total_seconds"] = _time_now(cfg, timing_device) - total_t0

    return result


def select_best_score(metrics: dict[str, Any], cfg: TransformerPhysicalTrainConfig) -> float:
    """Select checkpoint score according to cfg.best_score_mode."""
    mode = str(cfg.best_score_mode).lower().strip()
    def as_metric_tensor(
            key: str,
            default: float = 0.0,
            like: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        value = metrics.get(key, default)
        if torch.is_tensor(value):
            return value
        if like is not None:
            return torch.as_tensor(float(value), dtype=like.dtype, device=like.device)
        return torch.as_tensor(float(value), dtype=torch.float64)

    def finite_or(key: str, fallback_key: str) -> torch.Tensor:
        value = as_metric_tensor(key, float("nan"))
        fallback = as_metric_tensor(fallback_key, 0.0, like=value)
        return torch.where(torch.isfinite(value), value, fallback)

    if mode == "response":
        key = "response_loss" if "response_loss" in metrics else "data_loss"
        value = metrics[key]
    elif mode == "freq":
        key = "freq_loss"
        value = metrics[key]
    elif mode == "mixed":
        key = "total_loss"
        value = metrics[key]
    elif mode == "guarded_freq":
        freq_value = metrics["freq_loss"]
        guard_value = metrics.get("state_no_regression_guard_loss", 0.0)
        if torch.is_tensor(freq_value) or torch.is_tensor(guard_value):
            if not torch.is_tensor(freq_value):
                freq_value = torch.as_tensor(float(freq_value))
            if not torch.is_tensor(guard_value):
                guard_value = torch.as_tensor(float(guard_value), device=freq_value.device, dtype=freq_value.dtype)
            value = freq_value + float(cfg.best_score_guard_weight) * guard_value
        else:
            value = float(freq_value) + float(cfg.best_score_guard_weight) * float(guard_value)
    elif mode == "beta_amp":
        y_value = finite_or("y_ratio_to_alpha", "y_ratio")
        tip_y_value = finite_or("tip_y_ratio_to_alpha", "tip_y_ratio")
        last5_y_value = finite_or("last5_y_ratio_to_alpha", "last5_y_ratio")
        x_value = finite_or("x_ratio_to_alpha", "x_ratio")
        guard_value = as_metric_tensor("beta_alpha_guard_loss", 0.0, like=y_value)
        improvement_value = as_metric_tensor("beta_amp_improvement_loss", 0.0, like=y_value)
        sign_value = as_metric_tensor("beta_damp_sign_loss", 0.0, like=y_value)
        value = (
            0.35 * y_value
            + 0.20 * tip_y_value
            + 0.25 * last5_y_value
            + 0.20 * x_value
            + float(cfg.best_score_guard_weight) * guard_value
            + 0.25 * improvement_value
            + 0.10 * sign_value
        )
    else:
        raise ValueError(
            f"Unsupported best_score_mode={cfg.best_score_mode!r}. "
            "Expected one of: response, freq, mixed, guarded_freq, beta_amp."
        )
    return float(value.detach().cpu()) if torch.is_tensor(value) else float(value)


def can_update_best_checkpoint(
        *,
        valid_metrics: dict[str, float],
        cfg: TransformerPhysicalTrainConfig,
) -> tuple[bool, str]:
    """
    Decide whether current epoch is eligible to update best checkpoint.

    当前项目约束：
        x 方向是叶片挥舞主响应方向，不允许相对 static kappa-y baseline 恶化。
        因此当 use_x_constraint_for_best=True 时，只有 valid_x <= x_best_constraint_max
        的 epoch 才能成为 best checkpoint。

    Returns:
        eligible:
            True  -> this epoch can compete for best checkpoint
            False -> this epoch must not update best checkpoint
        reason:
            Human-readable reason for logging.
    """
    if not bool(cfg.use_x_constraint_for_best):
        return True, "x_constraint_disabled"

    if "x_ratio" in valid_metrics:
        valid_x = float(valid_metrics["x_ratio"])
    elif "valid_x" in valid_metrics:
        valid_x = float(valid_metrics["valid_x"])
    else:
        raise KeyError(
            "valid_metrics must contain 'x_ratio' or 'valid_x' when "
            "use_x_constraint_for_best=True."
        )

    max_allowed = float(cfg.x_best_constraint_max)

    if valid_x <= max_allowed:
        return True, f"x_constraint_passed(valid_x={valid_x:.6e} <= {max_allowed:.6e})"

    return False, f"x_constraint_failed(valid_x={valid_x:.6e} > {max_allowed:.6e})"


def run_post_training_tests(
        *,
        cfg: TransformerPhysicalTrainConfig,
        best_checkpoint: Path,
        output_dir: Path,
        test_load_files: list[str],
) -> list[dict[str, Any]]:
    """
    训练结束后，用 best checkpoint 自动跑独立 test 评估。

    注意：
        这里调用 scripts/evaluate_transformer_vs_baselines.py。
        test 结果只写入报告，不参与训练、不参与 best checkpoint 选择。
    """
    if not best_checkpoint.exists():
        raise FileNotFoundError(
            f"Best checkpoint not found, cannot run post-training test: {best_checkpoint}"
        )

    eval_script = PROJECT_ROOT / "scripts" / "evaluate_transformer_vs_baselines.py"
    if not eval_script.exists():
        raise FileNotFoundError(f"Evaluation script not found: {eval_script}")

    results: list[dict[str, Any]] = []

    for load_file_str in test_load_files:
        load_file = Path(load_file_str).resolve()
        if not load_file.exists():
            raise FileNotFoundError(f"Test load file not found: {load_file}")

        case_stem = _safe_case_stem(load_file)

        if cfg.test_output_dir is None or str(cfg.test_output_dir).strip() == "":
            eval_output_dir = output_dir / "post_train_test" / case_stem
        else:
            base_test_dir = Path(str(cfg.test_output_dir)).resolve()
            if len(test_load_files) == 1:
                eval_output_dir = base_test_dir
            else:
                eval_output_dir = base_test_dir / case_stem

        eval_case_name = f"{cfg.test_case_name_prefix}_{case_stem}"

        cmd = [
            sys.executable,
            str(eval_script),
            "--checkpoint",
            str(best_checkpoint),
            "--load-file",
            str(load_file),
            "--output-dir",
            str(eval_output_dir),
            "--case-name",
            str(eval_case_name),
            "--max-steps",
            str(int(cfg.test_max_steps)),
            "--device",
            str(cfg.device),
        ]

        print()
        print("[Post-train Test] Run evaluation")
        print("  " + " ".join(cmd))

        subprocess.run(cmd, check=True)

        metrics_csv = eval_output_dir / "metrics.csv"
        _print_post_train_test_summary(
            metrics_csv=metrics_csv,
            load_file=load_file,
        )

        results.append(
            {
                "load_file": str(load_file),
                "eval_output_dir": str(eval_output_dir),
                "metrics_csv": str(metrics_csv),
                "metrics": _read_metrics_csv(metrics_csv),
            }
        )

    return results


# ============================================================
# Build model
# ============================================================

def build_training_model(
        *,
        cfg: TransformerPhysicalTrainConfig,
        registry,
        geometry_dim: int,
        dtype_core: torch.dtype,
        device: torch.device,
) -> tuple[TransformerPhysicalRolloutTorch, dict[str, Any]]:
    template_cfg = PhysicalTemplateConfig(
        blade_csv=str(cfg.blade_csv),
        alpha_flap=float(cfg.alpha_flap),
        alpha_edge=float(cfg.alpha_edge),
        alpha_torsion=float(cfg.alpha_torsion),
        twist_column=str(cfg.base_phi_twist_column),
        phi_sign=float(cfg.base_phi_sign),
        rotate_mass=bool(cfg.rotate_mass),
        kappa_y_static_scale=float(cfg.kappa_y_static_scale),
        kappa_y_scale_mode=str(cfg.kappa_y_scale_mode),
        beta_damp_template_gain_x=float(cfg.beta_damp_template_gain_x),
        beta_damp_template_gain_y=float(cfg.beta_damp_template_gain_y),
        xy_template_mode="root_to_tip",
        xy_delta_phi_deg=1.0,
        enabled_params=str(cfg.enabled_params),  # 关键：显式传给 physical_templates
        verbose=True,
    )
    template_bundle = build_dynamic_stiffness_templates(template_cfg)

    M0 = np.asarray(template_bundle.M0, dtype=np.float64)
    K0 = np.asarray(template_bundle.K0, dtype=np.float64)

    natural_freqs_hz = _compute_natural_frequencies_hz(M0, K0, num_modes=10)
    C0, ref_freq_used = _build_structural_damping_matrix(
        K=K0,
        zeta_structural=float(cfg.zeta_structural),
        ref_freq_hz=cfg.ref_freq_hz,
        natural_freqs_hz=natural_freqs_hz,
    )
    damping_template_scale = _structural_damping_scale(
        zeta_structural=float(cfg.zeta_structural),
        ref_freq_hz=ref_freq_used,
    )
    damping_templates = template_bundle.damping_template_dict(
        damping_scale=damping_template_scale,
        beta_damp_template_gain_x=float(cfg.beta_damp_template_gain_x),
        beta_damp_template_gain_y=float(cfg.beta_damp_template_gain_y),
    )

    core = DynamicPhysicalCoreTorch(
        M0=M0,
        K0=K0,
        C0=C0,
        stiffness_templates=template_bundle.stiffness_template_dict(),
        damping_templates=damping_templates,
        registry=registry,
        config=DynamicPhysicalCoreConfig(
            dt=float(cfg.dt),
            gamma=0.5,
            beta=0.25,
            dtype=dtype_core,
            linear_solve_mode=str(cfg.linear_solve_mode),
            symmetrize_k_eff=True,
            precompute_newmark_matrices=bool(cfg.fast_core_precompute_newmark),
        ),
    ).to(device)

    encoder = SpatiotemporalPhysicsEncoder(
        geometry_dim=int(geometry_dim),
        registry=registry,
        config=SpatiotemporalPhysicsEncoderConfig(
            n_nodes=48,
            dof_per_node=6,
            d_model=int(cfg.d_model),
            n_spatial_heads=int(cfg.n_spatial_heads),
            n_temporal_heads=int(cfg.n_temporal_heads),
            n_temporal_layers=int(cfg.n_temporal_layers),
            temporal_ff_dim=2 * int(cfg.d_model),
            spatial_mlp_hidden_dim=2 * int(cfg.d_model),
            dropout=float(cfg.dropout),
            use_response_branch=bool(cfg.use_response_branch),
            use_load_branch=bool(cfg.use_load_branch),
            use_geometry_branch=bool(cfg.use_geometry_branch),
            use_load_spectral_features=bool(cfg.use_load_spectral_features),
            load_spectral_feature_dim=cfg.load_spectral_feature_dim,
            load_spectral_window_size=_infer_load_spectral_window_size(cfg),
            load_spectral_freq_min=float(cfg.load_spectral_freq_min),
            load_spectral_freq_max=float(cfg.load_spectral_freq_max),
            load_spectral_bands=str(cfg.load_spectral_bands),
            load_spectral_observations=str(cfg.load_spectral_observations),
            load_spectral_last_k=int(cfg.load_spectral_last_k),
            load_spectral_active_rel_threshold=float(cfg.load_spectral_active_rel_threshold),
            load_spectral_active_abs_threshold=float(cfg.load_spectral_active_abs_threshold),
            load_spectral_normalize=bool(cfg.load_spectral_normalize),
            condition_dynamic_branches_on_geometry=bool(cfg.condition_dynamic_branches_on_geometry),
            causal_temporal=True,
            temporal_window_size=cfg.temporal_window_size,
            use_temporal_transformer=True,
            use_phase_gated_decomposition=bool(cfg.use_phase_gated_decomposition),
            phase_slow_scale=float(cfg.phase_slow_scale),
            phase_fast_scale=float(cfg.phase_fast_scale),
            phase_gate_init_bias=float(cfg.phase_gate_init_bias),
            phase_total_clip_scale=float(cfg.phase_total_clip_scale),
        ),
    ).to(device)

    model = TransformerPhysicalRolloutTorch(
        encoder=encoder,
        physical_core=core,
        config=TransformerRolloutConfig(
            conditioning_mode="static",
            encoder_dtype=_get_torch_dtype(cfg.encoder_dtype),
            core_dtype=dtype_core,
            detach_static_conditioning=False,
            detach_rollout_state_each_step=False,
            profile_timing=bool(cfg.profile_train_timing),
            profile_timing_sync_cuda=bool(cfg.profile_timing_sync_cuda),
        ),
    ).to(device)

    info = {
        "template_summary": template_bundle.summary(),
        "natural_freqs_hz": natural_freqs_hz.tolist(),
        "ref_freq_used": ref_freq_used,
        "M0_shape": list(M0.shape),
        "K0_shape": list(K0.shape),
        "C0_shape": list(C0.shape),
        "damping_template_scale": float(damping_template_scale),
    }

    return model, info


# ============================================================
# CLI
# ============================================================

def parse_args() -> tuple[
    TransformerPhysicalTrainConfig,
    list[str],
    list[str],
    list[str],
    list[str],
    list[str],
]:
    d = TransformerPhysicalTrainConfig()

    parser = argparse.ArgumentParser(
        description=(
            "Train alpha_xy-only Transformer physical parameter with teacher-supervised static conditioning. "
            "Current first validation: enabled_params=alpha_x."
        )
    )

    parser.add_argument("--teacher-exe", type=str, default=d.teacher_exe)
    parser.add_argument("--template-inp", type=str, default=d.template_inp)
    parser.add_argument("--blade-csv", type=str, default=d.blade_csv)

    parser.add_argument("--training-case-dir", type=str, default=d.training_case_dir)
    parser.add_argument("--output-dir", type=str, default=d.output_dir)

    # ------------------------------------------------------------
    # Load file override
    # ------------------------------------------------------------
    # 日常不需要传这些参数。
    # 如果传了，则优先使用命令行手动列表；
    # 如果不传，则使用脚本顶部 DEFAULT_N_TRAIN_CASES 等配置自动生成。
    parser.add_argument("--train-load-files", type=str, default=None)
    parser.add_argument("--valid-load-files", type=str, default=None)
    parser.add_argument("--test-load-files", type=str, default=None)

    parser.add_argument("--train-case-paths", type=str, default=None)
    parser.add_argument("--valid-case-paths", type=str, default=None)

    parser.add_argument("--t-initial", type=float, default=d.t_initial)
    parser.add_argument("--t-final", type=float, default=d.t_final)
    parser.add_argument("--dt", type=float, default=d.dt)

    parser.add_argument("--teacher-node-start", type=int, default=d.teacher_node_start)
    parser.add_argument("--teacher-node-end", type=int, default=d.teacher_node_end)
    parser.add_argument("--teacher-demean", action="store_true", default=d.teacher_demean)

    parser.add_argument("--zeta-structural", type=float, default=d.zeta_structural)
    parser.add_argument("--ref-freq-hz", type=float, default=d.ref_freq_hz)
    parser.add_argument("--beta-damp-template-gain-x", type=float, default=d.beta_damp_template_gain_x)
    parser.add_argument("--beta-damp-template-gain-y", type=float, default=d.beta_damp_template_gain_y)

    parser.add_argument("--kappa-y-static-scale", type=float, default=d.kappa_y_static_scale)
    parser.add_argument("--kappa-y-scale-mode", type=str, default=d.kappa_y_scale_mode,
                        choices=["uy_only", "y_bending"])

    parser.add_argument("--enabled-params", type=str, default=d.enabled_params)

    parser.add_argument("--use-response-branch", action="store_true", default=d.use_response_branch)
    parser.add_argument("--no-response-branch", dest="use_response_branch", action="store_false")

    parser.add_argument("--use-load-branch", action="store_true", default=d.use_load_branch)
    parser.add_argument("--no-load-branch", dest="use_load_branch", action="store_false")

    parser.add_argument("--use-load-spectral-features", action="store_true", default=d.use_load_spectral_features)
    parser.add_argument("--no-load-spectral-features", dest="use_load_spectral_features", action="store_false")
    parser.add_argument("--load-spectral-window-size", type=int, default=d.load_spectral_window_size)
    parser.add_argument("--load-spectral-feature-dim", type=int, default=d.load_spectral_feature_dim)
    parser.add_argument("--load-spectral-freq-min", type=float, default=d.load_spectral_freq_min)
    parser.add_argument("--load-spectral-freq-max", type=float, default=d.load_spectral_freq_max)
    parser.add_argument("--load-spectral-bands", type=str, default=d.load_spectral_bands)
    parser.add_argument("--load-spectral-observations", type=str, default=d.load_spectral_observations)
    parser.add_argument("--load-spectral-last-k", type=int, default=d.load_spectral_last_k)
    parser.add_argument("--load-spectral-active-rel-threshold", type=float, default=d.load_spectral_active_rel_threshold)
    parser.add_argument("--load-spectral-active-abs-threshold", type=float, default=d.load_spectral_active_abs_threshold)
    parser.add_argument("--load-spectral-normalize", action="store_true", default=d.load_spectral_normalize)
    parser.add_argument("--no-load-spectral-normalize", dest="load_spectral_normalize", action="store_false")

    parser.add_argument("--use-load-spectral-disk-cache", dest="use_load_spectral_disk_cache",
                        action="store_true", default=d.use_load_spectral_disk_cache,
                        help="Cache fixed causal load spectral features on disk and reuse them across runs.")
    parser.add_argument("--no-load-spectral-disk-cache", dest="use_load_spectral_disk_cache", action="store_false")
    parser.add_argument("--load-spectral-cache-dir", type=str, default=d.load_spectral_cache_dir)
    parser.add_argument("--force-recompute-load-spectral-cache", action="store_true",
                        default=d.force_recompute_load_spectral_cache)

    parser.add_argument("--use-cached-alignment-loss", dest="use_cached_alignment_loss",
                        action="store_true", default=d.use_cached_alignment_loss,
                        help="Cache teacher-side spectral/peak/lag loss quantities per case.")
    parser.add_argument("--no-cached-alignment-loss", dest="use_cached_alignment_loss", action="store_false")

    parser.add_argument("--fast-core-precompute-newmark", dest="fast_core_precompute_newmark",
                        action="store_true", default=d.fast_core_precompute_newmark,
                        help="Precompute the constant Newmark effective matrix part inside the dynamic core.")
    parser.add_argument("--no-fast-core-precompute-newmark", dest="fast_core_precompute_newmark", action="store_false")
    parser.add_argument("--linear-solve-mode", type=str, default=d.linear_solve_mode, choices=["solve", "cholesky"])

    parser.add_argument("--use-geometry-branch", action="store_true", default=d.use_geometry_branch)
    parser.add_argument("--no-geometry-branch", dest="use_geometry_branch", action="store_false")

    parser.add_argument("--condition-dynamic-branches-on-geometry", action="store_true",
                        default=d.condition_dynamic_branches_on_geometry)
    parser.add_argument("--no-condition-dynamic-branches-on-geometry", dest="condition_dynamic_branches_on_geometry",
                        action="store_false")

    parser.add_argument("--d-model", type=int, default=d.d_model)
    parser.add_argument("--n-spatial-heads", type=int, default=d.n_spatial_heads)
    parser.add_argument("--n-temporal-heads", type=int, default=d.n_temporal_heads)
    parser.add_argument("--n-temporal-layers", type=int, default=d.n_temporal_layers)
    parser.add_argument("--dropout", type=float, default=d.dropout)

    parser.add_argument(
        "--temporal-window-size",
        type=int,
        default=d.temporal_window_size,
        help=(
            "Fixed causal lookback window for temporal attention. "
            "None means full-history causal attention. "
            "Example: 96 means theta_t attends only to the latest 96 steps."
        ),
    )

    parser.add_argument("--use-phase-gated-decomposition", action="store_true", default=d.use_phase_gated_decomposition)
    parser.add_argument("--no-phase-gated-decomposition", dest="use_phase_gated_decomposition", action="store_false")
    parser.add_argument("--phase-slow-scale", type=float, default=d.phase_slow_scale)
    parser.add_argument("--phase-fast-scale", type=float, default=d.phase_fast_scale)
    parser.add_argument("--phase-gate-init-bias", type=float, default=d.phase_gate_init_bias)
    parser.add_argument("--phase-total-clip-scale", type=float, default=d.phase_total_clip_scale)

    parser.add_argument("--prepare-cases", action="store_true", default=d.prepare_cases)
    parser.add_argument("--rebuild-cases", action="store_true", default=d.rebuild_cases)
    parser.add_argument("--max-steps-per-case", type=int, default=d.max_steps_per_case)

    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument(
        "--run-test-after-training",
        dest="run_test_after_training",
        action="store_true",
        help="After training, evaluate best checkpoint on test load files.",
    )
    test_group.add_argument(
        "--no-run-test-after-training",
        dest="run_test_after_training",
        action="store_false",
        help="Disable post-training test evaluation.",
    )
    parser.set_defaults(run_test_after_training=d.run_test_after_training)

    parser.add_argument(
        "--test-max-steps",
        type=int,
        default=d.test_max_steps,
        help="Max steps used by post-training evaluation.",
    )
    parser.add_argument(
        "--test-output-dir",
        type=str,
        default=d.test_output_dir,
        help=(
            "Output dir for post-training test. "
            "If omitted, use output_dir/post_train_test/<case_stem>."
        ),
    )
    parser.add_argument(
        "--test-case-name-prefix",
        type=str,
        default=d.test_case_name_prefix,
    )

    parser.add_argument("--epochs", type=int, default=d.epochs)
    parser.add_argument("--lr", type=float, default=d.lr)
    parser.add_argument("--weight-decay", type=float, default=d.weight_decay)
    parser.add_argument("--grad-clip-norm", type=float, default=d.grad_clip_norm)

    parser.add_argument("--device", type=str, default=d.device)
    parser.add_argument("--encoder-dtype", type=str, default=d.encoder_dtype, choices=["float32", "float64"])
    parser.add_argument("--core-dtype", type=str, default=d.core_dtype, choices=["float32", "float64"])

    parser.add_argument("--w-y", type=float, default=d.w_y)
    parser.add_argument("--w-x-guard", type=float, default=d.w_x_guard)
    parser.add_argument(
        "--w-x",
        type=float,
        default=d.w_x,
        help="Explicit full-x response loss weight. Default 0.0 keeps old behavior.",
    )
    parser.add_argument("--x-guard-tol", type=float, default=d.x_guard_tol)
    parser.add_argument("--w-theta-amp", type=float, default=d.w_theta_amp)
    parser.add_argument("--w-theta-smooth", type=float, default=d.w_theta_smooth)
    parser.add_argument("--w-theta-slow-smooth", type=float, default=d.w_theta_slow_smooth)
    parser.add_argument("--w-theta-fast-amp", type=float, default=d.w_theta_fast_amp)
    parser.add_argument("--w-theta-fast-smooth", type=float, default=d.w_theta_fast_smooth)
    parser.add_argument("--w-theta-fast-window-mean", type=float, default=d.w_theta_fast_window_mean)
    parser.add_argument("--w-theta-gated-fast-window-mean", type=float, default=d.w_theta_gated_fast_window_mean)
    parser.add_argument("--theta-fast-window-mean-seconds", type=float, default=d.theta_fast_window_mean_seconds)
    parser.add_argument("--theta-fast-window-mean-stride-seconds", type=float, default=d.theta_fast_window_mean_stride_seconds)
    parser.add_argument("--w-phase-gate-l1", type=float, default=d.w_phase_gate_l1)
    parser.add_argument("--w-phase-gate-tv", type=float, default=d.w_phase_gate_tv)
    parser.add_argument("--phase-gate-active-threshold", type=float, default=d.phase_gate_active_threshold)
    parser.add_argument("--w-phase-gate-bootstrap", type=float, default=d.w_phase_gate_bootstrap)
    parser.add_argument("--phase-gate-bootstrap-target", type=float, default=d.phase_gate_bootstrap_target)
    parser.add_argument("--phase-gate-bootstrap-end-epoch", type=int, default=d.phase_gate_bootstrap_end_epoch)
    parser.add_argument("--use-loss-curriculum", action="store_true", default=d.use_loss_curriculum)
    parser.add_argument("--no-loss-curriculum", dest="use_loss_curriculum", action="store_false")
    parser.add_argument("--curriculum-phase-start-epoch", type=int, default=d.curriculum_phase_start_epoch)
    parser.add_argument("--curriculum-phase-full-epoch", type=int, default=d.curriculum_phase_full_epoch)
    parser.add_argument("--curriculum-guard-start-epoch", type=int, default=d.curriculum_guard_start_epoch)
    parser.add_argument("--curriculum-guard-full-epoch", type=int, default=d.curriculum_guard_full_epoch)
    parser.add_argument("--curriculum-lag-start-scale", type=float, default=d.curriculum_lag_start_scale)
    parser.add_argument("--curriculum-phase-drift-start-scale", type=float,
                        default=d.curriculum_phase_drift_start_scale)
    parser.add_argument("--curriculum-adaptive-phase-start-scale", type=float,
                        default=d.curriculum_adaptive_phase_start_scale)
    parser.add_argument("--curriculum-gate-reg-start-scale", type=float,
                        default=d.curriculum_gate_reg_start_scale)
    parser.add_argument("--curriculum-static-good-gate-start-scale", type=float,
                        default=d.curriculum_static_good_gate_start_scale)
    parser.add_argument("--curriculum-state-guard-start-scale", type=float,
                        default=d.curriculum_state_guard_start_scale)
    parser.add_argument("--w-tip-y", type=float, default=d.w_tip_y)
    parser.add_argument("--w-last5-y", type=float, default=d.w_last5_y)

    parser.add_argument("--w-spec-x", type=float, default=d.w_spec_x)
    parser.add_argument("--w-spec-y", type=float, default=d.w_spec_y)
    parser.add_argument("--w-peak-x", type=float, default=d.w_peak_x)
    parser.add_argument("--w-peak-y", type=float, default=d.w_peak_y)
    parser.add_argument("--freq-min", type=float, default=d.freq_min)
    parser.add_argument("--freq-max", type=float, default=d.freq_max)
    parser.add_argument("--freq-peak-temperature", type=float, default=d.freq_peak_temperature)

    parser.add_argument("--alignment-observations", type=str, default=d.alignment_observations,
                        help="Observations for peak/lag loss, e.g. 'tip,last5' or 'tip,last5,mean'.")
    parser.add_argument("--alignment-last-k", type=int, default=d.alignment_last_k)

    parser.add_argument("--w-peak-time-x", type=float, default=d.w_peak_time_x)
    parser.add_argument("--w-peak-time-y", type=float, default=d.w_peak_time_y)
    parser.add_argument("--peak-time-start", type=float, default=d.peak_time_start)
    parser.add_argument("--peak-time-end", type=float, default=d.peak_time_end)
    parser.add_argument("--peak-time-window-seconds", type=float, default=d.peak_time_window_seconds)
    parser.add_argument("--peak-time-temperature", type=float, default=d.peak_time_temperature)
    parser.add_argument("--peak-time-min-distance-seconds", type=float, default=d.peak_time_min_distance_seconds)
    parser.add_argument("--peak-time-prominence-std", type=float, default=d.peak_time_prominence_std)
    parser.add_argument("--peak-time-max-events", type=int, default=d.peak_time_max_events)

    parser.add_argument("--w-lag-x", type=float, default=d.w_lag_x)
    parser.add_argument("--w-lag-y", type=float, default=d.w_lag_y)
    parser.add_argument("--lag-start", type=float, default=d.lag_start)
    parser.add_argument("--lag-end", type=float, default=d.lag_end)
    parser.add_argument("--lag-window-seconds", type=float, default=d.lag_window_seconds)
    parser.add_argument("--lag-stride-seconds", type=float, default=d.lag_stride_seconds)
    parser.add_argument("--max-lag-seconds", type=float, default=d.max_lag_seconds)
    parser.add_argument("--lag-temperature", type=float, default=d.lag_temperature)

    parser.add_argument("--use-adaptive-phase-window-loss", action="store_true",
                        default=d.use_adaptive_phase_window_loss)
    parser.add_argument("--no-adaptive-phase-window-loss", dest="use_adaptive_phase_window_loss",
                        action="store_false")
    parser.add_argument("--phase-window-observations", type=str, default=d.phase_window_observations)
    parser.add_argument("--phase-window-last-k", type=int, default=d.phase_window_last_k)
    parser.add_argument("--phase-window-start", type=float, default=d.phase_window_start)
    parser.add_argument("--phase-window-end", type=float, default=d.phase_window_end)
    parser.add_argument("--phase-window-size-seconds", type=float, default=d.phase_window_size_seconds)
    parser.add_argument("--phase-window-stride-seconds", type=float, default=d.phase_window_stride_seconds)
    parser.add_argument("--phase-window-top-k", type=int, default=d.phase_window_top_k)
    parser.add_argument("--phase-window-score-temperature", type=float, default=d.phase_window_score_temperature)
    parser.add_argument("--phase-window-gate-score-ref", type=float, default=d.phase_window_gate_score_ref)
    parser.add_argument("--phase-window-max-lag-seconds", type=float, default=d.phase_window_max_lag_seconds)
    parser.add_argument("--phase-window-lag-temperature", type=float, default=d.phase_window_lag_temperature)
    parser.add_argument("--phase-window-freq-min", type=float, default=d.phase_window_freq_min)
    parser.add_argument("--phase-window-freq-max", type=float, default=d.phase_window_freq_max)
    parser.add_argument("--phase-window-amplitude-reference", type=float,
                        default=d.phase_window_amplitude_reference)
    parser.add_argument("--phase-window-amplitude-reference-x", type=float,
                        default=d.phase_window_amplitude_reference_x)
    parser.add_argument("--phase-window-amplitude-reference-y", type=float,
                        default=d.phase_window_amplitude_reference_y)
    parser.add_argument("--phase-window-amplitude-weight", type=float,
                        default=d.phase_window_amplitude_weight)
    parser.add_argument("--phase-window-amplitude-power", type=float,
                        default=d.phase_window_amplitude_power)
    parser.add_argument("--phase-window-amplitude-max-weight", type=float,
                        default=d.phase_window_amplitude_max_weight)
    parser.add_argument("--phase-window-static-failure-weight", type=float,
                        default=d.phase_window_static_failure_weight)
    parser.add_argument("--phase-window-static-failure-max-weight", type=float,
                        default=d.phase_window_static_failure_max_weight)
    parser.add_argument("--w-adaptive-phase-x", type=float, default=d.w_adaptive_phase_x)
    parser.add_argument("--w-adaptive-phase-y", type=float, default=d.w_adaptive_phase_y)
    parser.add_argument("--w-complex-phase-x", type=float, default=d.w_complex_phase_x)
    parser.add_argument("--w-complex-phase-y", type=float, default=d.w_complex_phase_y)
    parser.add_argument("--w-complex-amp-guard-x", type=float, default=d.w_complex_amp_guard_x)
    parser.add_argument("--w-complex-amp-guard-y", type=float, default=d.w_complex_amp_guard_y)
    parser.add_argument("--w-phase-gate-align", type=float, default=d.w_phase_gate_align)
    parser.add_argument("--use-phase-drift-rate-loss", action="store_true",
                        default=d.use_phase_drift_rate_loss)
    parser.add_argument("--no-phase-drift-rate-loss", dest="use_phase_drift_rate_loss",
                        action="store_false")
    parser.add_argument("--w-phase-drift-lag-x", type=float, default=d.w_phase_drift_lag_x)
    parser.add_argument("--w-phase-drift-lag-y", type=float, default=d.w_phase_drift_lag_y)
    parser.add_argument("--w-phase-drift-rate-x", type=float, default=d.w_phase_drift_rate_x)
    parser.add_argument("--w-phase-drift-rate-y", type=float, default=d.w_phase_drift_rate_y)
    parser.add_argument("--phase-drift-observations", type=str, default=d.phase_drift_observations)
    parser.add_argument("--phase-drift-last-k", type=int, default=d.phase_drift_last_k)
    parser.add_argument("--phase-drift-start", type=float, default=d.phase_drift_start)
    parser.add_argument("--phase-drift-end", type=float, default=d.phase_drift_end)
    parser.add_argument("--phase-drift-window-seconds", type=float, default=d.phase_drift_window_seconds)
    parser.add_argument("--phase-drift-stride-seconds", type=float, default=d.phase_drift_stride_seconds)
    parser.add_argument("--phase-drift-max-lag-seconds", type=float, default=d.phase_drift_max_lag_seconds)
    parser.add_argument("--phase-drift-lag-temperature", type=float, default=d.phase_drift_lag_temperature)
    parser.add_argument("--phase-drift-freq-min", type=float, default=d.phase_drift_freq_min)
    parser.add_argument("--phase-drift-freq-max", type=float, default=d.phase_drift_freq_max)
    parser.add_argument("--phase-drift-high-power-threshold", type=float,
                        default=d.phase_drift_high_power_threshold)
    parser.add_argument("--phase-drift-high-power-temperature", type=float,
                        default=d.phase_drift_high_power_temperature)
    parser.add_argument("--phase-drift-amplitude-reference", type=float,
                        default=d.phase_drift_amplitude_reference)
    parser.add_argument("--phase-drift-amplitude-reference-x", type=float,
                        default=d.phase_drift_amplitude_reference_x)
    parser.add_argument("--phase-drift-amplitude-reference-y", type=float,
                        default=d.phase_drift_amplitude_reference_y)
    parser.add_argument("--phase-drift-amplitude-weight", type=float,
                        default=d.phase_drift_amplitude_weight)
    parser.add_argument("--phase-drift-amplitude-power", type=float,
                        default=d.phase_drift_amplitude_power)
    parser.add_argument("--phase-drift-amplitude-max-weight", type=float,
                        default=d.phase_drift_amplitude_max_weight)
    parser.add_argument("--phase-drift-static-failure-weight", type=float,
                        default=d.phase_drift_static_failure_weight)
    parser.add_argument("--phase-drift-static-failure-max-weight", type=float,
                        default=d.phase_drift_static_failure_max_weight)
    parser.add_argument("--use-local-band-phase-loss", action="store_true",
                        default=d.use_local_band_phase_loss)
    parser.add_argument("--no-local-band-phase-loss", dest="use_local_band_phase_loss",
                        action="store_false")
    parser.add_argument("--w-local-band-phase-x", type=float, default=d.w_local_band_phase_x)
    parser.add_argument("--w-local-band-phase-y", type=float, default=d.w_local_band_phase_y)
    parser.add_argument("--local-band-phase-observations", type=str,
                        default=d.local_band_phase_observations)
    parser.add_argument("--local-band-phase-last-k", type=int,
                        default=d.local_band_phase_last_k)
    parser.add_argument("--local-band-phase-start", type=float,
                        default=d.local_band_phase_start)
    parser.add_argument("--local-band-phase-end", type=float,
                        default=d.local_band_phase_end)
    parser.add_argument("--local-band-phase-window-seconds", type=float,
                        default=d.local_band_phase_window_seconds)
    parser.add_argument("--local-band-phase-stride-seconds", type=float,
                        default=d.local_band_phase_stride_seconds)
    parser.add_argument("--local-band-phase-freq-min", type=float,
                        default=d.local_band_phase_freq_min)
    parser.add_argument("--local-band-phase-freq-max", type=float,
                        default=d.local_band_phase_freq_max)
    parser.add_argument("--local-band-phase-high-power-threshold", type=float,
                        default=d.local_band_phase_high_power_threshold)
    parser.add_argument("--local-band-phase-high-power-temperature", type=float,
                        default=d.local_band_phase_high_power_temperature)
    parser.add_argument("--use-local-phase-corr-loss", action="store_true",
                        default=d.use_local_phase_corr_loss)
    parser.add_argument("--no-local-phase-corr-loss", dest="use_local_phase_corr_loss",
                        action="store_false")
    parser.add_argument("--w-local-phase-corr-x", type=float, default=d.w_local_phase_corr_x)
    parser.add_argument("--w-local-phase-corr-y", type=float, default=d.w_local_phase_corr_y)
    parser.add_argument("--local-phase-corr-observations", type=str,
                        default=d.local_phase_corr_observations)
    parser.add_argument("--local-phase-corr-last-k", type=int,
                        default=d.local_phase_corr_last_k)
    parser.add_argument("--local-phase-corr-start", type=float,
                        default=d.local_phase_corr_start)
    parser.add_argument("--local-phase-corr-end", type=float,
                        default=d.local_phase_corr_end)
    parser.add_argument("--local-phase-corr-window-seconds", type=float,
                        default=d.local_phase_corr_window_seconds)
    parser.add_argument("--local-phase-corr-stride-seconds", type=float,
                        default=d.local_phase_corr_stride_seconds)
    parser.add_argument("--local-phase-corr-max-lag-seconds", type=float,
                        default=d.local_phase_corr_max_lag_seconds)
    parser.add_argument("--local-phase-corr-lag-temperature", type=float,
                        default=d.local_phase_corr_lag_temperature)
    parser.add_argument("--local-phase-corr-freq-min", type=float,
                        default=d.local_phase_corr_freq_min)
    parser.add_argument("--local-phase-corr-freq-max", type=float,
                        default=d.local_phase_corr_freq_max)
    parser.add_argument("--local-phase-corr-base-weight", type=float,
                        default=d.local_phase_corr_base_weight)
    parser.add_argument("--local-phase-corr-high-power-weight", type=float,
                        default=d.local_phase_corr_high_power_weight)
    parser.add_argument("--local-phase-corr-high-power-threshold", type=float,
                        default=d.local_phase_corr_high_power_threshold)
    parser.add_argument("--local-phase-corr-high-power-temperature", type=float,
                        default=d.local_phase_corr_high_power_temperature)
    parser.add_argument("--local-phase-corr-static-failure-weight", type=float,
                        default=d.local_phase_corr_static_failure_weight)
    parser.add_argument("--local-phase-corr-static-failure-max-weight", type=float,
                        default=d.local_phase_corr_static_failure_max_weight)
    parser.add_argument("--local-phase-corr-corr-weight", type=float,
                        default=d.local_phase_corr_corr_weight)
    parser.add_argument("--local-phase-corr-corr-gap-weight", type=float,
                        default=d.local_phase_corr_corr_gap_weight)
    parser.add_argument("--local-phase-corr-lag-weight", type=float,
                        default=d.local_phase_corr_lag_weight)
    parser.add_argument("--local-phase-corr-corr-gap-tol", type=float,
                        default=d.local_phase_corr_corr_gap_tol)
    parser.add_argument("--use-local-phase-increment-loss", action="store_true",
                        default=d.use_local_phase_increment_loss)
    parser.add_argument("--no-local-phase-increment-loss", dest="use_local_phase_increment_loss",
                        action="store_false")
    parser.add_argument("--w-local-phase-absolute-x", type=float,
                        default=d.w_local_phase_absolute_x)
    parser.add_argument("--w-local-phase-absolute-y", type=float,
                        default=d.w_local_phase_absolute_y)
    parser.add_argument("--w-local-phase-increment-x", type=float,
                        default=d.w_local_phase_increment_x)
    parser.add_argument("--w-local-phase-increment-y", type=float,
                        default=d.w_local_phase_increment_y)
    parser.add_argument("--local-phase-increment-observations", type=str,
                        default=d.local_phase_increment_observations)
    parser.add_argument("--local-phase-increment-last-k", type=int,
                        default=d.local_phase_increment_last_k)
    parser.add_argument("--local-phase-increment-start", type=float,
                        default=d.local_phase_increment_start)
    parser.add_argument("--local-phase-increment-end", type=float,
                        default=d.local_phase_increment_end)
    parser.add_argument("--local-phase-increment-window-seconds", type=float,
                        default=d.local_phase_increment_window_seconds)
    parser.add_argument("--local-phase-increment-stride-seconds", type=float,
                        default=d.local_phase_increment_stride_seconds)
    parser.add_argument("--local-phase-increment-freq-min", type=float,
                        default=d.local_phase_increment_freq_min)
    parser.add_argument("--local-phase-increment-freq-max", type=float,
                        default=d.local_phase_increment_freq_max)
    parser.add_argument("--local-phase-increment-base-weight", type=float,
                        default=d.local_phase_increment_base_weight)
    parser.add_argument("--local-phase-increment-high-power-weight", type=float,
                        default=d.local_phase_increment_high_power_weight)
    parser.add_argument("--local-phase-increment-high-power-threshold", type=float,
                        default=d.local_phase_increment_high_power_threshold)
    parser.add_argument("--local-phase-increment-high-power-temperature", type=float,
                        default=d.local_phase_increment_high_power_temperature)
    parser.add_argument("--local-phase-increment-static-failure-weight", type=float,
                        default=d.local_phase_increment_static_failure_weight)
    parser.add_argument("--local-phase-increment-static-failure-max-weight", type=float,
                        default=d.local_phase_increment_static_failure_max_weight)
    parser.add_argument("--use-continuous-phase-lag-loss", action="store_true",
                        default=d.use_continuous_phase_lag_loss)
    parser.add_argument("--no-continuous-phase-lag-loss", dest="use_continuous_phase_lag_loss",
                        action="store_false")
    parser.add_argument("--w-continuous-phase-absolute-x", type=float,
                        default=d.w_continuous_phase_absolute_x)
    parser.add_argument("--w-continuous-phase-absolute-y", type=float,
                        default=d.w_continuous_phase_absolute_y)
    parser.add_argument("--w-continuous-phase-time-shift-x", type=float,
                        default=d.w_continuous_phase_time_shift_x)
    parser.add_argument("--w-continuous-phase-time-shift-y", type=float,
                        default=d.w_continuous_phase_time_shift_y)
    parser.add_argument("--continuous-phase-observations", type=str,
                        default=d.continuous_phase_observations)
    parser.add_argument("--continuous-phase-last-k", type=int,
                        default=d.continuous_phase_last_k)
    parser.add_argument("--continuous-phase-start", type=float,
                        default=d.continuous_phase_start)
    parser.add_argument("--continuous-phase-end", type=float,
                        default=d.continuous_phase_end)
    parser.add_argument("--continuous-phase-window-seconds", type=float,
                        default=d.continuous_phase_window_seconds)
    parser.add_argument("--continuous-phase-stride-seconds", type=float,
                        default=d.continuous_phase_stride_seconds)
    parser.add_argument("--continuous-phase-freq-min", type=float,
                        default=d.continuous_phase_freq_min)
    parser.add_argument("--continuous-phase-freq-max", type=float,
                        default=d.continuous_phase_freq_max)
    parser.add_argument("--continuous-phase-n-freq-bins", type=int,
                        default=d.continuous_phase_n_freq_bins)
    parser.add_argument("--continuous-phase-frequency-temperature", type=float,
                        default=d.continuous_phase_frequency_temperature)
    parser.add_argument("--continuous-phase-time-shift-scale-seconds", type=float,
                        default=d.continuous_phase_time_shift_scale_seconds)
    parser.add_argument("--continuous-phase-base-weight", type=float,
                        default=d.continuous_phase_base_weight)
    parser.add_argument("--continuous-phase-high-power-weight", type=float,
                        default=d.continuous_phase_high_power_weight)
    parser.add_argument("--continuous-phase-high-power-threshold", type=float,
                        default=d.continuous_phase_high_power_threshold)
    parser.add_argument("--continuous-phase-high-power-temperature", type=float,
                        default=d.continuous_phase_high_power_temperature)
    parser.add_argument("--continuous-phase-static-failure-weight", type=float,
                        default=d.continuous_phase_static_failure_weight)
    parser.add_argument("--continuous-phase-static-failure-max-weight", type=float,
                        default=d.continuous_phase_static_failure_max_weight)
    parser.add_argument("--use-local-phase-slope-loss", dest="use_local_phase_slope_loss",
                        action="store_true", default=d.use_local_phase_slope_loss)
    parser.add_argument("--no-local-phase-slope-loss", dest="use_local_phase_slope_loss",
                        action="store_false")
    parser.add_argument("--w-local-phase-slope-x", type=float,
                        default=d.w_local_phase_slope_x)
    parser.add_argument("--w-local-phase-slope-y", type=float,
                        default=d.w_local_phase_slope_y)
    parser.add_argument("--local-phase-slope-observations", type=str,
                        default=d.local_phase_slope_observations)
    parser.add_argument("--local-phase-slope-last-k", type=int,
                        default=d.local_phase_slope_last_k)
    parser.add_argument("--local-phase-slope-start", type=float,
                        default=d.local_phase_slope_start)
    parser.add_argument("--local-phase-slope-end", type=float,
                        default=d.local_phase_slope_end)
    parser.add_argument("--local-phase-slope-window-seconds", type=float,
                        default=d.local_phase_slope_window_seconds)
    parser.add_argument("--local-phase-slope-stride-seconds", type=float,
                        default=d.local_phase_slope_stride_seconds)
    parser.add_argument("--local-phase-slope-freq-min", type=float,
                        default=d.local_phase_slope_freq_min)
    parser.add_argument("--local-phase-slope-freq-max", type=float,
                        default=d.local_phase_slope_freq_max)
    parser.add_argument("--local-phase-slope-n-freq-bins", type=int,
                        default=d.local_phase_slope_n_freq_bins)
    parser.add_argument("--local-phase-slope-frequency-temperature", type=float,
                        default=d.local_phase_slope_frequency_temperature)
    parser.add_argument("--local-phase-slope-time-shift-scale-seconds", type=float,
                        default=d.local_phase_slope_time_shift_scale_seconds)
    parser.add_argument("--local-phase-slope-base-weight", type=float,
                        default=d.local_phase_slope_base_weight)
    parser.add_argument("--local-phase-slope-high-power-weight", type=float,
                        default=d.local_phase_slope_high_power_weight)
    parser.add_argument("--local-phase-slope-high-power-threshold", type=float,
                        default=d.local_phase_slope_high_power_threshold)
    parser.add_argument("--local-phase-slope-high-power-temperature", type=float,
                        default=d.local_phase_slope_high_power_temperature)
    parser.add_argument("--local-phase-slope-static-failure-weight", type=float,
                        default=d.local_phase_slope_static_failure_weight)
    parser.add_argument("--local-phase-slope-static-failure-max-weight", type=float,
                        default=d.local_phase_slope_static_failure_max_weight)
    parser.add_argument("--use-slow-only-branch-diagnosis", action="store_true",
                        default=d.use_slow_only_branch_diagnosis)
    parser.add_argument("--no-slow-only-branch-diagnosis", dest="use_slow_only_branch_diagnosis",
                        action="store_false")
    parser.add_argument("--w-slow-good-no-regression", type=float, default=d.w_slow_good_no_regression)
    parser.add_argument("--w-slow-good-fast-suppress", type=float, default=d.w_slow_good_fast_suppress)
    parser.add_argument("--w-slow-bad-phase", type=float, default=d.w_slow_bad_phase)
    parser.add_argument("--slow-good-response-ratio-limit", type=float, default=d.slow_good_response_ratio_limit)
    parser.add_argument("--slow-good-corr-drop-tol", type=float, default=d.slow_good_corr_drop_tol)
    parser.add_argument("--slow-good-amp-log-tol", type=float, default=d.slow_good_amp_log_tol)
    parser.add_argument("--slow-bad-weight-max", type=float, default=d.slow_bad_weight_max)
    parser.add_argument("--use-static-quality-gate-suppression", action="store_true",
                        default=d.use_static_quality_gate_suppression)
    parser.add_argument("--no-static-quality-gate-suppression", dest="use_static_quality_gate_suppression",
                        action="store_false")
    parser.add_argument("--w-static-good-gate-l1", type=float, default=d.w_static_good_gate_l1)
    parser.add_argument("--static-quality-observations", type=str, default=d.static_quality_observations)
    parser.add_argument("--static-quality-last-k", type=int, default=d.static_quality_last_k)
    parser.add_argument("--static-quality-start", type=float, default=d.static_quality_start)
    parser.add_argument("--static-quality-end", type=float, default=d.static_quality_end)
    parser.add_argument("--static-quality-window-seconds", type=float, default=d.static_quality_window_seconds)
    parser.add_argument("--static-quality-stride-seconds", type=float, default=d.static_quality_stride_seconds)
    parser.add_argument("--static-quality-max-lag-seconds", type=float, default=d.static_quality_max_lag_seconds)
    parser.add_argument("--static-quality-good-corr-threshold", type=float, default=d.static_quality_good_corr_threshold)
    parser.add_argument("--static-quality-good-lag-seconds", type=float, default=d.static_quality_good_lag_seconds)
    parser.add_argument("--static-quality-good-amp-log-tol", type=float, default=d.static_quality_good_amp_log_tol)
    parser.add_argument("--use-state-window-no-regression-guard", action="store_true",
                        default=d.use_state_window_no_regression_guard)
    parser.add_argument("--no-state-window-no-regression-guard",
                        dest="use_state_window_no_regression_guard",
                        action="store_false")
    parser.add_argument("--w-state-no-regression-response", type=float,
                        default=d.w_state_no_regression_response)
    parser.add_argument("--w-state-no-regression-corr", type=float,
                        default=d.w_state_no_regression_corr)
    parser.add_argument("--w-state-no-regression-amp", type=float,
                        default=d.w_state_no_regression_amp)
    parser.add_argument("--state-no-regression-response-ratio-limit", type=float,
                        default=d.state_no_regression_response_ratio_limit)
    parser.add_argument("--state-no-regression-corr-drop-tol", type=float,
                        default=d.state_no_regression_corr_drop_tol)
    parser.add_argument("--state-no-regression-amp-log-tol", type=float,
                        default=d.state_no_regression_amp_log_tol)

    parser.add_argument(
        "--beta-loss-mode",
        type=str,
        default=d.beta_loss_mode,
        choices=["auto", "standard", "alpha_relative_amp", "alpha_relative", "amplitude", "amp"],
        help=(
            "Beta training objective. auto enables alpha-relative amplitude loss "
            "when freeze-alpha beta training is active; standard keeps alpha-style loss."
        ),
    )
    parser.add_argument("--beta-amp-observations", type=str, default=d.beta_amp_observations)
    parser.add_argument("--beta-amp-last-k", type=int, default=d.beta_amp_last_k)
    parser.add_argument("--beta-amp-start", type=float, default=d.beta_amp_start)
    parser.add_argument("--beta-amp-end", type=float, default=d.beta_amp_end)
    parser.add_argument("--beta-amp-window-seconds", type=float, default=d.beta_amp_window_seconds)
    parser.add_argument("--beta-amp-stride-seconds", type=float, default=d.beta_amp_stride_seconds)
    parser.add_argument("--beta-amp-alpha-error-ref", type=float, default=d.beta_amp_alpha_error_ref)
    parser.add_argument("--beta-amp-alpha-error-max-weight", type=float,
                        default=d.beta_amp_alpha_error_max_weight)
    parser.add_argument("--beta-amp-log-tol", type=float, default=d.beta_amp_log_tol)
    parser.add_argument("--beta-amp-improvement-margin", type=float,
                        default=d.beta_amp_improvement_margin)
    parser.add_argument("--w-beta-amp-x", type=float, default=d.w_beta_amp_x)
    parser.add_argument("--w-beta-amp-y", type=float, default=d.w_beta_amp_y)
    parser.add_argument("--w-beta-amp-tip-y", type=float, default=d.w_beta_amp_tip_y)
    parser.add_argument("--w-beta-amp-last5-y", type=float, default=d.w_beta_amp_last5_y)
    parser.add_argument("--w-beta-amp-improvement", type=float,
                        default=d.w_beta_amp_improvement)
    parser.add_argument("--w-beta-damp-sign-x", type=float, default=d.w_beta_damp_sign_x)
    parser.add_argument("--w-beta-damp-sign-y", type=float, default=d.w_beta_damp_sign_y)
    parser.add_argument("--beta-damp-sign-min-alpha-error", type=float,
                        default=d.beta_damp_sign_min_alpha_error)
    parser.add_argument("--w-beta-alpha-response-guard", type=float,
                        default=d.w_beta_alpha_response_guard)
    parser.add_argument("--w-beta-alpha-corr-guard", type=float,
                        default=d.w_beta_alpha_corr_guard)
    parser.add_argument("--w-beta-alpha-amp-guard", type=float,
                        default=d.w_beta_alpha_amp_guard)
    parser.add_argument("--beta-alpha-response-ratio-limit", type=float,
                        default=d.beta_alpha_response_ratio_limit)
    parser.add_argument("--beta-alpha-corr-drop-tol", type=float,
                        default=d.beta_alpha_corr_drop_tol)
    parser.add_argument("--beta-alpha-amp-worsen-tol", type=float,
                        default=d.beta_alpha_amp_worsen_tol)

    parser.add_argument("--best-score-guard-weight", type=float, default=d.best_score_guard_weight)

    parser.add_argument("--best-score-mode", type=str, default=d.best_score_mode,
                        choices=["response", "freq", "mixed", "guarded_freq", "beta_amp"])
    parser.add_argument(
        "--best-start-epoch",
        type=int,
        default=d.best_start_epoch,
        help=(
            "Do not allow best checkpoint updates before this epoch. "
            "Use this with phase/gate curricula to avoid saving an early closed-gate model."
        ),
    )

    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=d.init_checkpoint,
        help=(
            "Optional checkpoint used to initialize model weights before training. "
            "Only model/encoder weights are loaded; optimizer state is intentionally "
            "not restored so LR and loss weights can be changed for follow-up runs."
        ),
    )
    parser.add_argument(
        "--freeze-alpha-backbone-for-beta",
        action="store_true",
        default=d.freeze_alpha_backbone_for_beta,
        help=(
            "Freeze the loaded encoder/Transformer/alpha rows and train only C-target "
            "beta rows of the final theta head. This is the conservative first beta mode."
        ),
    )
    parser.add_argument(
        "--no-freeze-alpha-backbone-for-beta",
        dest="freeze_alpha_backbone_for_beta",
        action="store_false",
    )

    best_constraint_group = parser.add_mutually_exclusive_group()
    best_constraint_group.add_argument(
        "--use-x-constraint-for-best",
        dest="use_x_constraint_for_best",
        action="store_true",
        help=(
            "If enabled, best checkpoint can be updated only when "
            "valid_x <= x_best_constraint_max."
        ),
    )
    best_constraint_group.add_argument(
        "--no-x-constraint-for-best",
        dest="use_x_constraint_for_best",
        action="store_false",
        help="Disable x hard constraint for best checkpoint selection.",
    )
    parser.set_defaults(use_x_constraint_for_best=d.use_x_constraint_for_best)

    parser.add_argument(
        "--x-best-constraint-max",
        type=float,
        default=d.x_best_constraint_max,
        help="Maximum allowed valid_x ratio for best checkpoint selection.",
    )

    parser.add_argument(
        "--valid-every",
        type=int,
        default=d.valid_every,
        help=(
            "Run validation every N epochs. "
            "1 keeps old behavior. Example: 5 runs validation at epoch 1, 5, 10, ..."
        ),
    )
    parser.add_argument("--early-stop-patience", type=int, default=d.early_stop_patience)
    parser.add_argument("--early-stop-min-delta", type=float, default=d.early_stop_min_delta)
    parser.add_argument("--min-epochs", type=int, default=d.min_epochs)

    parser.add_argument("--lr-plateau-patience", type=int, default=d.lr_plateau_patience)
    parser.add_argument("--lr-plateau-factor", type=float, default=d.lr_plateau_factor)
    parser.add_argument("--min-lr", type=float, default=d.min_lr)

    parser.add_argument("--print-every", type=int, default=d.print_every)
    parser.add_argument("--save-every", type=int, default=d.save_every)
    parser.add_argument("--seed", type=int, default=d.seed)

    parser.add_argument(
        "--cache-npz-in-ram",
        action="store_true",
        help="Cache .npz training/validation cases in RAM to avoid repeated disk I/O.",
    )
    parser.add_argument(
        "--allow-tf32-encoder",
        action="store_true",
        help="Allow TF32 for eligible float32 CUDA matmul kernels in encoder/MLP. Core float64 rollout is unchanged.",
    )
    parser.add_argument(
        "--matmul-precision",
        type=str,
        default=d.matmul_precision,
        choices=["highest", "high", "medium"],
        help="torch.set_float32_matmul_precision setting for float32 matmul kernels.",
    )
    parser.add_argument(
        "--profile-train-timing",
        action="store_true",
        help=(
            "Record per-epoch timing diagnostics in training_history.csv for model forward, "
            "encoder, Newmark loop, loss, backward, grad clip, and optimizer step."
        ),
    )
    parser.add_argument(
        "--profile-timing-sync-cuda",
        action="store_true",
        help=(
            "Synchronize CUDA before timing reads for more precise profiling. "
            "This slows training and should be used only for short diagnostic runs."
        ),
    )

    args = parser.parse_args()

    global _USE_FAST_NPZ_CACHE
    _USE_FAST_NPZ_CACHE = bool(args.cache_npz_in_ram)

    configure_torch_fast_runtime(
        device=args.device,
        allow_tf32_encoder=args.allow_tf32_encoder,
        matmul_precision=args.matmul_precision,
    )

    if args.cache_npz_in_ram:
        print("[Fast NPZ Cache] enabled: training/validation .npz files will be cached in RAM.")
    if args.profile_train_timing:
        print("[Timing Profile]")
        print("  enabled = True")
        print(f"  sync_cuda = {bool(args.profile_timing_sync_cuda)}")

    cfg = TransformerPhysicalTrainConfig(
        teacher_exe=args.teacher_exe,
        template_inp=args.template_inp,
        blade_csv=args.blade_csv,
        training_case_dir=args.training_case_dir,
        output_dir=args.output_dir,

        train_load_dir=str(TRAIN_LOAD_DIR),
        valid_load_dir=str(VALID_LOAD_DIR),
        test_load_dir=str(TEST_LOAD_DIR),
        train_load_prefix=DEFAULT_TRAIN_LOAD_PREFIX,
        valid_load_prefix=DEFAULT_VALID_LOAD_PREFIX,
        test_load_prefix=DEFAULT_TEST_LOAD_PREFIX,
        load_suffix=DEFAULT_LOAD_SUFFIX,
        load_case_start_index=DEFAULT_LOAD_CASE_START_INDEX,
        n_train_cases=DEFAULT_N_TRAIN_CASES,
        n_valid_cases=DEFAULT_N_VALID_CASES,
        n_test_cases=DEFAULT_N_TEST_CASES,

        t_initial=args.t_initial,
        t_final=args.t_final,
        dt=args.dt,
        teacher_node_start=args.teacher_node_start,
        teacher_node_end=args.teacher_node_end,
        teacher_demean=args.teacher_demean,
        zeta_structural=args.zeta_structural,
        ref_freq_hz=args.ref_freq_hz,
        beta_damp_template_gain_x=float(args.beta_damp_template_gain_x),
        beta_damp_template_gain_y=float(args.beta_damp_template_gain_y),
        kappa_y_static_scale=args.kappa_y_static_scale,
        kappa_y_scale_mode=args.kappa_y_scale_mode,
        enabled_params=args.enabled_params,
        use_response_branch=args.use_response_branch,
        use_load_branch=args.use_load_branch,
        use_geometry_branch=args.use_geometry_branch,
        use_load_spectral_features=bool(args.use_load_spectral_features),
        load_spectral_window_size=args.load_spectral_window_size,
        load_spectral_feature_dim=args.load_spectral_feature_dim,
        load_spectral_freq_min=float(args.load_spectral_freq_min),
        load_spectral_freq_max=float(args.load_spectral_freq_max),
        load_spectral_bands=str(args.load_spectral_bands),
        load_spectral_observations=str(args.load_spectral_observations),
        load_spectral_last_k=int(args.load_spectral_last_k),
        load_spectral_active_rel_threshold=float(args.load_spectral_active_rel_threshold),
        load_spectral_active_abs_threshold=float(args.load_spectral_active_abs_threshold),
        load_spectral_normalize=bool(args.load_spectral_normalize),
        use_load_spectral_disk_cache=bool(args.use_load_spectral_disk_cache),
        load_spectral_cache_dir=args.load_spectral_cache_dir,
        force_recompute_load_spectral_cache=bool(args.force_recompute_load_spectral_cache),
        use_cached_alignment_loss=bool(args.use_cached_alignment_loss),
        fast_core_precompute_newmark=bool(args.fast_core_precompute_newmark),
        linear_solve_mode=str(args.linear_solve_mode),
        condition_dynamic_branches_on_geometry=args.condition_dynamic_branches_on_geometry,
        d_model=args.d_model,
        n_spatial_heads=args.n_spatial_heads,
        n_temporal_heads=args.n_temporal_heads,
        n_temporal_layers=args.n_temporal_layers,
        dropout=args.dropout,
        temporal_window_size=args.temporal_window_size,
        use_phase_gated_decomposition=bool(args.use_phase_gated_decomposition),
        phase_slow_scale=float(args.phase_slow_scale),
        phase_fast_scale=float(args.phase_fast_scale),
        phase_gate_init_bias=float(args.phase_gate_init_bias),
        phase_total_clip_scale=float(args.phase_total_clip_scale),
        prepare_cases=args.prepare_cases,
        rebuild_cases=args.rebuild_cases,
        max_steps_per_case=args.max_steps_per_case,
        run_test_after_training=bool(args.run_test_after_training),
        test_max_steps=int(args.test_max_steps),
        test_output_dir=args.test_output_dir,
        test_case_name_prefix=str(args.test_case_name_prefix),
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        device=args.device,
        encoder_dtype=args.encoder_dtype,
        core_dtype=args.core_dtype,
        cache_npz_in_ram=bool(args.cache_npz_in_ram),
        allow_tf32_encoder=bool(args.allow_tf32_encoder),
        matmul_precision=str(args.matmul_precision),
        profile_train_timing=bool(args.profile_train_timing),
        profile_timing_sync_cuda=bool(args.profile_timing_sync_cuda),
        w_y=args.w_y,
        w_x=args.w_x,
        w_x_guard=args.w_x_guard,
        use_x_constraint_for_best=bool(args.use_x_constraint_for_best),
        x_best_constraint_max=float(args.x_best_constraint_max),
        x_guard_tol=args.x_guard_tol,
        w_theta_amp=args.w_theta_amp,
        w_theta_smooth=args.w_theta_smooth,
        w_theta_slow_smooth=args.w_theta_slow_smooth,
        w_theta_fast_amp=args.w_theta_fast_amp,
        w_theta_fast_smooth=args.w_theta_fast_smooth,
        w_theta_fast_window_mean=args.w_theta_fast_window_mean,
        w_theta_gated_fast_window_mean=args.w_theta_gated_fast_window_mean,
        theta_fast_window_mean_seconds=args.theta_fast_window_mean_seconds,
        theta_fast_window_mean_stride_seconds=args.theta_fast_window_mean_stride_seconds,
        w_phase_gate_l1=args.w_phase_gate_l1,
        w_phase_gate_tv=args.w_phase_gate_tv,
        phase_gate_active_threshold=float(args.phase_gate_active_threshold),
        w_phase_gate_bootstrap=float(args.w_phase_gate_bootstrap),
        phase_gate_bootstrap_target=float(args.phase_gate_bootstrap_target),
        phase_gate_bootstrap_end_epoch=int(args.phase_gate_bootstrap_end_epoch),
        use_loss_curriculum=bool(args.use_loss_curriculum),
        curriculum_phase_start_epoch=int(args.curriculum_phase_start_epoch),
        curriculum_phase_full_epoch=int(args.curriculum_phase_full_epoch),
        curriculum_guard_start_epoch=int(args.curriculum_guard_start_epoch),
        curriculum_guard_full_epoch=int(args.curriculum_guard_full_epoch),
        curriculum_lag_start_scale=float(args.curriculum_lag_start_scale),
        curriculum_phase_drift_start_scale=float(args.curriculum_phase_drift_start_scale),
        curriculum_adaptive_phase_start_scale=float(args.curriculum_adaptive_phase_start_scale),
        curriculum_gate_reg_start_scale=float(args.curriculum_gate_reg_start_scale),
        curriculum_static_good_gate_start_scale=float(args.curriculum_static_good_gate_start_scale),
        curriculum_state_guard_start_scale=float(args.curriculum_state_guard_start_scale),
        w_tip_y=args.w_tip_y,
        w_last5_y=args.w_last5_y,
        w_spec_x=args.w_spec_x,
        w_spec_y=args.w_spec_y,
        w_peak_x=args.w_peak_x,
        w_peak_y=args.w_peak_y,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        freq_peak_temperature=args.freq_peak_temperature,
        alignment_observations=args.alignment_observations,
        alignment_last_k=args.alignment_last_k,
        w_peak_time_x=args.w_peak_time_x,
        w_peak_time_y=args.w_peak_time_y,
        peak_time_start=args.peak_time_start,
        peak_time_end=args.peak_time_end,
        peak_time_window_seconds=args.peak_time_window_seconds,
        peak_time_temperature=args.peak_time_temperature,
        peak_time_min_distance_seconds=args.peak_time_min_distance_seconds,
        peak_time_prominence_std=args.peak_time_prominence_std,
        peak_time_max_events=args.peak_time_max_events,
        w_lag_x=args.w_lag_x,
        w_lag_y=args.w_lag_y,
        lag_start=args.lag_start,
        lag_end=args.lag_end,
        lag_window_seconds=args.lag_window_seconds,
        lag_stride_seconds=args.lag_stride_seconds,
        max_lag_seconds=args.max_lag_seconds,
        lag_temperature=args.lag_temperature,
        use_adaptive_phase_window_loss=bool(args.use_adaptive_phase_window_loss),
        phase_window_observations=str(args.phase_window_observations),
        phase_window_last_k=int(args.phase_window_last_k),
        phase_window_start=float(args.phase_window_start),
        phase_window_end=args.phase_window_end,
        phase_window_size_seconds=float(args.phase_window_size_seconds),
        phase_window_stride_seconds=float(args.phase_window_stride_seconds),
        phase_window_top_k=int(args.phase_window_top_k),
        phase_window_score_temperature=float(args.phase_window_score_temperature),
        phase_window_gate_score_ref=float(args.phase_window_gate_score_ref),
        phase_window_max_lag_seconds=float(args.phase_window_max_lag_seconds),
        phase_window_lag_temperature=float(args.phase_window_lag_temperature),
        phase_window_freq_min=float(args.phase_window_freq_min),
        phase_window_freq_max=args.phase_window_freq_max,
        phase_window_amplitude_reference=float(args.phase_window_amplitude_reference),
        phase_window_amplitude_reference_x=float(args.phase_window_amplitude_reference_x),
        phase_window_amplitude_reference_y=float(args.phase_window_amplitude_reference_y),
        phase_window_amplitude_weight=float(args.phase_window_amplitude_weight),
        phase_window_amplitude_power=float(args.phase_window_amplitude_power),
        phase_window_amplitude_max_weight=float(args.phase_window_amplitude_max_weight),
        phase_window_static_failure_weight=float(args.phase_window_static_failure_weight),
        phase_window_static_failure_max_weight=float(args.phase_window_static_failure_max_weight),
        w_adaptive_phase_x=float(args.w_adaptive_phase_x),
        w_adaptive_phase_y=float(args.w_adaptive_phase_y),
        w_complex_phase_x=float(args.w_complex_phase_x),
        w_complex_phase_y=float(args.w_complex_phase_y),
        w_complex_amp_guard_x=float(args.w_complex_amp_guard_x),
        w_complex_amp_guard_y=float(args.w_complex_amp_guard_y),
        w_phase_gate_align=float(args.w_phase_gate_align),
        use_phase_drift_rate_loss=bool(args.use_phase_drift_rate_loss),
        w_phase_drift_lag_x=float(args.w_phase_drift_lag_x),
        w_phase_drift_lag_y=float(args.w_phase_drift_lag_y),
        w_phase_drift_rate_x=float(args.w_phase_drift_rate_x),
        w_phase_drift_rate_y=float(args.w_phase_drift_rate_y),
        phase_drift_observations=str(args.phase_drift_observations),
        phase_drift_last_k=int(args.phase_drift_last_k),
        phase_drift_start=float(args.phase_drift_start),
        phase_drift_end=args.phase_drift_end,
        phase_drift_window_seconds=float(args.phase_drift_window_seconds),
        phase_drift_stride_seconds=float(args.phase_drift_stride_seconds),
        phase_drift_max_lag_seconds=float(args.phase_drift_max_lag_seconds),
        phase_drift_lag_temperature=float(args.phase_drift_lag_temperature),
        phase_drift_freq_min=float(args.phase_drift_freq_min),
        phase_drift_freq_max=args.phase_drift_freq_max,
        phase_drift_high_power_threshold=float(args.phase_drift_high_power_threshold),
        phase_drift_high_power_temperature=float(args.phase_drift_high_power_temperature),
        phase_drift_amplitude_reference=float(args.phase_drift_amplitude_reference),
        phase_drift_amplitude_reference_x=float(args.phase_drift_amplitude_reference_x),
        phase_drift_amplitude_reference_y=float(args.phase_drift_amplitude_reference_y),
        phase_drift_amplitude_weight=float(args.phase_drift_amplitude_weight),
        phase_drift_amplitude_power=float(args.phase_drift_amplitude_power),
        phase_drift_amplitude_max_weight=float(args.phase_drift_amplitude_max_weight),
        phase_drift_static_failure_weight=float(args.phase_drift_static_failure_weight),
        phase_drift_static_failure_max_weight=float(args.phase_drift_static_failure_max_weight),
        use_local_band_phase_loss=bool(args.use_local_band_phase_loss),
        w_local_band_phase_x=float(args.w_local_band_phase_x),
        w_local_band_phase_y=float(args.w_local_band_phase_y),
        local_band_phase_observations=str(args.local_band_phase_observations),
        local_band_phase_last_k=int(args.local_band_phase_last_k),
        local_band_phase_start=float(args.local_band_phase_start),
        local_band_phase_end=args.local_band_phase_end,
        local_band_phase_window_seconds=float(args.local_band_phase_window_seconds),
        local_band_phase_stride_seconds=float(args.local_band_phase_stride_seconds),
        local_band_phase_freq_min=float(args.local_band_phase_freq_min),
        local_band_phase_freq_max=args.local_band_phase_freq_max,
        local_band_phase_high_power_threshold=float(args.local_band_phase_high_power_threshold),
        local_band_phase_high_power_temperature=float(args.local_band_phase_high_power_temperature),
        use_local_phase_corr_loss=bool(args.use_local_phase_corr_loss),
        w_local_phase_corr_x=float(args.w_local_phase_corr_x),
        w_local_phase_corr_y=float(args.w_local_phase_corr_y),
        local_phase_corr_observations=str(args.local_phase_corr_observations),
        local_phase_corr_last_k=int(args.local_phase_corr_last_k),
        local_phase_corr_start=float(args.local_phase_corr_start),
        local_phase_corr_end=args.local_phase_corr_end,
        local_phase_corr_window_seconds=float(args.local_phase_corr_window_seconds),
        local_phase_corr_stride_seconds=float(args.local_phase_corr_stride_seconds),
        local_phase_corr_max_lag_seconds=float(args.local_phase_corr_max_lag_seconds),
        local_phase_corr_lag_temperature=float(args.local_phase_corr_lag_temperature),
        local_phase_corr_freq_min=float(args.local_phase_corr_freq_min),
        local_phase_corr_freq_max=args.local_phase_corr_freq_max,
        local_phase_corr_base_weight=float(args.local_phase_corr_base_weight),
        local_phase_corr_high_power_weight=float(args.local_phase_corr_high_power_weight),
        local_phase_corr_high_power_threshold=float(args.local_phase_corr_high_power_threshold),
        local_phase_corr_high_power_temperature=float(args.local_phase_corr_high_power_temperature),
        local_phase_corr_static_failure_weight=float(args.local_phase_corr_static_failure_weight),
        local_phase_corr_static_failure_max_weight=float(args.local_phase_corr_static_failure_max_weight),
        local_phase_corr_corr_weight=float(args.local_phase_corr_corr_weight),
        local_phase_corr_corr_gap_weight=float(args.local_phase_corr_corr_gap_weight),
        local_phase_corr_lag_weight=float(args.local_phase_corr_lag_weight),
        local_phase_corr_corr_gap_tol=float(args.local_phase_corr_corr_gap_tol),
        use_local_phase_increment_loss=bool(args.use_local_phase_increment_loss),
        w_local_phase_absolute_x=float(args.w_local_phase_absolute_x),
        w_local_phase_absolute_y=float(args.w_local_phase_absolute_y),
        w_local_phase_increment_x=float(args.w_local_phase_increment_x),
        w_local_phase_increment_y=float(args.w_local_phase_increment_y),
        local_phase_increment_observations=str(args.local_phase_increment_observations),
        local_phase_increment_last_k=int(args.local_phase_increment_last_k),
        local_phase_increment_start=float(args.local_phase_increment_start),
        local_phase_increment_end=args.local_phase_increment_end,
        local_phase_increment_window_seconds=float(args.local_phase_increment_window_seconds),
        local_phase_increment_stride_seconds=float(args.local_phase_increment_stride_seconds),
        local_phase_increment_freq_min=float(args.local_phase_increment_freq_min),
        local_phase_increment_freq_max=args.local_phase_increment_freq_max,
        local_phase_increment_base_weight=float(args.local_phase_increment_base_weight),
        local_phase_increment_high_power_weight=float(args.local_phase_increment_high_power_weight),
        local_phase_increment_high_power_threshold=float(args.local_phase_increment_high_power_threshold),
        local_phase_increment_high_power_temperature=float(args.local_phase_increment_high_power_temperature),
        local_phase_increment_static_failure_weight=float(args.local_phase_increment_static_failure_weight),
        local_phase_increment_static_failure_max_weight=float(args.local_phase_increment_static_failure_max_weight),
        use_continuous_phase_lag_loss=bool(args.use_continuous_phase_lag_loss),
        w_continuous_phase_absolute_x=float(args.w_continuous_phase_absolute_x),
        w_continuous_phase_absolute_y=float(args.w_continuous_phase_absolute_y),
        w_continuous_phase_time_shift_x=float(args.w_continuous_phase_time_shift_x),
        w_continuous_phase_time_shift_y=float(args.w_continuous_phase_time_shift_y),
        continuous_phase_observations=str(args.continuous_phase_observations),
        continuous_phase_last_k=int(args.continuous_phase_last_k),
        continuous_phase_start=float(args.continuous_phase_start),
        continuous_phase_end=args.continuous_phase_end,
        continuous_phase_window_seconds=float(args.continuous_phase_window_seconds),
        continuous_phase_stride_seconds=float(args.continuous_phase_stride_seconds),
        continuous_phase_freq_min=float(args.continuous_phase_freq_min),
        continuous_phase_freq_max=args.continuous_phase_freq_max,
        continuous_phase_n_freq_bins=int(args.continuous_phase_n_freq_bins),
        continuous_phase_frequency_temperature=float(args.continuous_phase_frequency_temperature),
        continuous_phase_time_shift_scale_seconds=float(args.continuous_phase_time_shift_scale_seconds),
        continuous_phase_base_weight=float(args.continuous_phase_base_weight),
        continuous_phase_high_power_weight=float(args.continuous_phase_high_power_weight),
        continuous_phase_high_power_threshold=float(args.continuous_phase_high_power_threshold),
        continuous_phase_high_power_temperature=float(args.continuous_phase_high_power_temperature),
        continuous_phase_static_failure_weight=float(args.continuous_phase_static_failure_weight),
        continuous_phase_static_failure_max_weight=float(args.continuous_phase_static_failure_max_weight),
        use_local_phase_slope_loss=bool(args.use_local_phase_slope_loss),
        w_local_phase_slope_x=float(args.w_local_phase_slope_x),
        w_local_phase_slope_y=float(args.w_local_phase_slope_y),
        local_phase_slope_observations=str(args.local_phase_slope_observations),
        local_phase_slope_last_k=int(args.local_phase_slope_last_k),
        local_phase_slope_start=float(args.local_phase_slope_start),
        local_phase_slope_end=args.local_phase_slope_end,
        local_phase_slope_window_seconds=float(args.local_phase_slope_window_seconds),
        local_phase_slope_stride_seconds=float(args.local_phase_slope_stride_seconds),
        local_phase_slope_freq_min=float(args.local_phase_slope_freq_min),
        local_phase_slope_freq_max=args.local_phase_slope_freq_max,
        local_phase_slope_n_freq_bins=int(args.local_phase_slope_n_freq_bins),
        local_phase_slope_frequency_temperature=float(args.local_phase_slope_frequency_temperature),
        local_phase_slope_time_shift_scale_seconds=float(args.local_phase_slope_time_shift_scale_seconds),
        local_phase_slope_base_weight=float(args.local_phase_slope_base_weight),
        local_phase_slope_high_power_weight=float(args.local_phase_slope_high_power_weight),
        local_phase_slope_high_power_threshold=float(args.local_phase_slope_high_power_threshold),
        local_phase_slope_high_power_temperature=float(args.local_phase_slope_high_power_temperature),
        local_phase_slope_static_failure_weight=float(args.local_phase_slope_static_failure_weight),
        local_phase_slope_static_failure_max_weight=float(args.local_phase_slope_static_failure_max_weight),
        use_slow_only_branch_diagnosis=bool(args.use_slow_only_branch_diagnosis),
        w_slow_good_no_regression=float(args.w_slow_good_no_regression),
        w_slow_good_fast_suppress=float(args.w_slow_good_fast_suppress),
        w_slow_bad_phase=float(args.w_slow_bad_phase),
        slow_good_response_ratio_limit=float(args.slow_good_response_ratio_limit),
        slow_good_corr_drop_tol=float(args.slow_good_corr_drop_tol),
        slow_good_amp_log_tol=float(args.slow_good_amp_log_tol),
        slow_bad_weight_max=float(args.slow_bad_weight_max),
        use_static_quality_gate_suppression=bool(args.use_static_quality_gate_suppression),
        w_static_good_gate_l1=float(args.w_static_good_gate_l1),
        static_quality_observations=str(args.static_quality_observations),
        static_quality_last_k=int(args.static_quality_last_k),
        static_quality_start=float(args.static_quality_start),
        static_quality_end=args.static_quality_end,
        static_quality_window_seconds=float(args.static_quality_window_seconds),
        static_quality_stride_seconds=float(args.static_quality_stride_seconds),
        static_quality_max_lag_seconds=float(args.static_quality_max_lag_seconds),
        static_quality_good_corr_threshold=float(args.static_quality_good_corr_threshold),
        static_quality_good_lag_seconds=float(args.static_quality_good_lag_seconds),
        static_quality_good_amp_log_tol=float(args.static_quality_good_amp_log_tol),
        use_state_window_no_regression_guard=bool(args.use_state_window_no_regression_guard),
        w_state_no_regression_response=float(args.w_state_no_regression_response),
        w_state_no_regression_corr=float(args.w_state_no_regression_corr),
        w_state_no_regression_amp=float(args.w_state_no_regression_amp),
        state_no_regression_response_ratio_limit=float(args.state_no_regression_response_ratio_limit),
        state_no_regression_corr_drop_tol=float(args.state_no_regression_corr_drop_tol),
        state_no_regression_amp_log_tol=float(args.state_no_regression_amp_log_tol),
        beta_loss_mode=str(args.beta_loss_mode),
        beta_amp_observations=str(args.beta_amp_observations),
        beta_amp_last_k=int(args.beta_amp_last_k),
        beta_amp_start=float(args.beta_amp_start),
        beta_amp_end=args.beta_amp_end,
        beta_amp_window_seconds=float(args.beta_amp_window_seconds),
        beta_amp_stride_seconds=float(args.beta_amp_stride_seconds),
        beta_amp_alpha_error_ref=float(args.beta_amp_alpha_error_ref),
        beta_amp_alpha_error_max_weight=float(args.beta_amp_alpha_error_max_weight),
        beta_amp_log_tol=float(args.beta_amp_log_tol),
        beta_amp_improvement_margin=float(args.beta_amp_improvement_margin),
        w_beta_amp_x=float(args.w_beta_amp_x),
        w_beta_amp_y=float(args.w_beta_amp_y),
        w_beta_amp_tip_y=float(args.w_beta_amp_tip_y),
        w_beta_amp_last5_y=float(args.w_beta_amp_last5_y),
        w_beta_amp_improvement=float(args.w_beta_amp_improvement),
        w_beta_damp_sign_x=float(args.w_beta_damp_sign_x),
        w_beta_damp_sign_y=float(args.w_beta_damp_sign_y),
        beta_damp_sign_min_alpha_error=float(args.beta_damp_sign_min_alpha_error),
        w_beta_alpha_response_guard=float(args.w_beta_alpha_response_guard),
        w_beta_alpha_corr_guard=float(args.w_beta_alpha_corr_guard),
        w_beta_alpha_amp_guard=float(args.w_beta_alpha_amp_guard),
        beta_alpha_response_ratio_limit=float(args.beta_alpha_response_ratio_limit),
        beta_alpha_corr_drop_tol=float(args.beta_alpha_corr_drop_tol),
        beta_alpha_amp_worsen_tol=float(args.beta_alpha_amp_worsen_tol),
        best_score_guard_weight=float(args.best_score_guard_weight),
        best_score_mode=args.best_score_mode,
        best_start_epoch=int(args.best_start_epoch),
        init_checkpoint=args.init_checkpoint,
        freeze_alpha_backbone_for_beta=bool(args.freeze_alpha_backbone_for_beta),
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        min_epochs=args.min_epochs,
        valid_every=args.valid_every,
        lr_plateau_patience=args.lr_plateau_patience,
        lr_plateau_factor=args.lr_plateau_factor,
        min_lr=args.min_lr,
        print_every=args.print_every,
        save_every=args.save_every,
        seed=args.seed,
    )

    default_train_load_files = build_load_files_from_cfg(cfg=cfg, split="train")
    default_valid_load_files = build_load_files_from_cfg(cfg=cfg, split="valid")
    default_test_load_files = build_load_files_from_cfg(cfg=cfg, split="test")

    # 如果命令行传了 --train-load-files / --valid-load-files / --test-load-files，
    # 则优先使用命令行；否则使用 cfg 自动生成的默认文件列表。
    train_load_files = _parse_file_list(args.train_load_files, default_train_load_files)
    valid_load_files = _parse_file_list(args.valid_load_files, default_valid_load_files)
    test_load_files = _parse_file_list(args.test_load_files, default_test_load_files)

    train_case_paths = _parse_path_list(args.train_case_paths)
    valid_case_paths = _parse_path_list(args.valid_case_paths)

    return (
        cfg,
        train_load_files,
        valid_load_files,
        test_load_files,
        train_case_paths,
        valid_case_paths,
    )


# ============================================================
# Main training
# ============================================================

def main() -> None:
    (
        cfg,
        train_load_files,
        valid_load_files,
        test_load_files,
        train_case_paths,
        valid_case_paths,
    ) = parse_args()

    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    output_dir = Path(cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device)
    dtype_case = _get_torch_dtype(cfg.core_dtype)
    dtype_core = _get_torch_dtype(cfg.core_dtype)

    print()
    print("=" * 100)
    print("[Train] Transformer physical parameters: static conditioning")
    print("=" * 100)
    print(f"  enabled_params = {cfg.enabled_params}")
    print(f"  use_phase_gated_decomposition = {getattr(cfg, 'use_phase_gated_decomposition', False)}")
    if bool(getattr(cfg, "use_phase_gated_decomposition", False)):
        print(f"  phase_gate_active_threshold = {cfg.phase_gate_active_threshold}")
        print(
            "  phase_gate_bootstrap = "
            f"w={cfg.w_phase_gate_bootstrap}, "
            f"target={cfg.phase_gate_bootstrap_target}, "
            f"end_epoch={cfg.phase_gate_bootstrap_end_epoch}"
        )
    print(f"  best_score_mode = {cfg.best_score_mode}")
    print(f"  best_start_epoch = {cfg.best_start_epoch}")
    print(f"  beta_loss_mode = {cfg.beta_loss_mode}")
    print(
        "  beta_damp_template_gain = "
        f"x:{cfg.beta_damp_template_gain_x}, y:{cfg.beta_damp_template_gain_y}"
    )
    print(
        "  beta_amp_improvement = "
        f"w:{cfg.w_beta_amp_improvement}, margin:{cfg.beta_amp_improvement_margin}, "
        f"sign_w_x/y:{cfg.w_beta_damp_sign_x}/{cfg.w_beta_damp_sign_y}"
    )
    print(f"  beta_alpha_reference_loss_active = {_use_beta_alpha_reference_loss(cfg)}")
    print(f"  use_loss_curriculum = {cfg.use_loss_curriculum}")
    if bool(cfg.use_loss_curriculum):
        print(
            "  loss_curriculum = "
            f"phase_start/full={cfg.curriculum_phase_start_epoch}/{cfg.curriculum_phase_full_epoch}, "
            f"guard_start/full={cfg.curriculum_guard_start_epoch}/{cfg.curriculum_guard_full_epoch}, "
            f"lag_start_scale={cfg.curriculum_lag_start_scale}, "
            f"phase_drift_start_scale={cfg.curriculum_phase_drift_start_scale}, "
            f"gate_reg_start_scale={cfg.curriculum_gate_reg_start_scale}"
        )
    print(
        f"  branches: response={cfg.use_response_branch}, load={cfg.use_load_branch}, geometry={cfg.use_geometry_branch}")
    print(f"  condition_dynamic_branches_on_geometry = {cfg.condition_dynamic_branches_on_geometry}")
    print(f"  temporal_window_size = {cfg.temporal_window_size}")
    print(f"  use_load_spectral_features = {cfg.use_load_spectral_features}")
    print(f"  use_load_spectral_disk_cache = {cfg.use_load_spectral_disk_cache}")
    print(f"  use_cached_alignment_loss = {cfg.use_cached_alignment_loss}")
    print(f"  use_adaptive_phase_window_loss = {cfg.use_adaptive_phase_window_loss}")
    if bool(cfg.use_adaptive_phase_window_loss):
        print(
            "  phase_window = "
            f"obs={cfg.phase_window_observations}, "
            f"start={cfg.phase_window_start}, end={cfg.phase_window_end}, "
            f"size={cfg.phase_window_size_seconds}, stride={cfg.phase_window_stride_seconds}, "
            f"top_k={cfg.phase_window_top_k}, "
            f"gate_score_ref={cfg.phase_window_gate_score_ref}, "
            f"freq=[{cfg.phase_window_freq_min}, {cfg.phase_window_freq_max}], "
            f"amp_ref={cfg.phase_window_amplitude_reference}, "
            f"amp_ref_xy=({cfg.phase_window_amplitude_reference_x}, {cfg.phase_window_amplitude_reference_y}), "
            f"amp_weight={cfg.phase_window_amplitude_weight}, "
            f"static_failure_weight={cfg.phase_window_static_failure_weight}, "
            f"static_failure_max_weight={cfg.phase_window_static_failure_max_weight}"
        )
    print(f"  use_phase_drift_rate_loss = {cfg.use_phase_drift_rate_loss}")
    if bool(cfg.use_phase_drift_rate_loss):
        print(
            "  phase_drift = "
            f"obs={cfg.phase_drift_observations}, "
            f"freq=[{cfg.phase_drift_freq_min}, {cfg.phase_drift_freq_max}], "
            f"amp_ref={cfg.phase_drift_amplitude_reference}, "
            f"amp_ref_xy=({cfg.phase_drift_amplitude_reference_x}, {cfg.phase_drift_amplitude_reference_y}), "
            f"amp_weight={cfg.phase_drift_amplitude_weight}, "
            f"amp_max_weight={cfg.phase_drift_amplitude_max_weight}, "
            f"static_failure_weight={cfg.phase_drift_static_failure_weight}, "
            f"static_failure_max_weight={cfg.phase_drift_static_failure_max_weight}"
        )
    print(f"  use_local_band_phase_loss = {cfg.use_local_band_phase_loss}")
    if bool(cfg.use_local_band_phase_loss):
        print(
            "  local_band_phase = "
            f"obs={cfg.local_band_phase_observations}, "
            f"freq=[{cfg.local_band_phase_freq_min}, {cfg.local_band_phase_freq_max}], "
            f"window={cfg.local_band_phase_window_seconds}, "
            f"stride={cfg.local_band_phase_stride_seconds}, "
            f"w_x={cfg.w_local_band_phase_x}, "
            f"w_y={cfg.w_local_band_phase_y}, "
            f"high_power_threshold={cfg.local_band_phase_high_power_threshold}, "
            f"high_power_temperature={cfg.local_band_phase_high_power_temperature}"
        )
    print(f"  use_local_phase_corr_loss = {cfg.use_local_phase_corr_loss}")
    if bool(cfg.use_local_phase_corr_loss):
        print(
            "  local_phase_corr = "
            f"obs={cfg.local_phase_corr_observations}, "
            f"freq=[{cfg.local_phase_corr_freq_min}, {cfg.local_phase_corr_freq_max}], "
            f"window={cfg.local_phase_corr_window_seconds}, "
            f"stride={cfg.local_phase_corr_stride_seconds}, "
            f"base_weight={cfg.local_phase_corr_base_weight}, "
            f"high_weight={cfg.local_phase_corr_high_power_weight}, "
            f"static_failure_weight={cfg.local_phase_corr_static_failure_weight}, "
            f"w_x={cfg.w_local_phase_corr_x}, "
            f"w_y={cfg.w_local_phase_corr_y}"
        )
    print(f"  use_local_phase_increment_loss = {cfg.use_local_phase_increment_loss}")
    if bool(cfg.use_local_phase_increment_loss):
        print(
            "  local_phase_increment = "
            f"obs={cfg.local_phase_increment_observations}, "
            f"freq=[{cfg.local_phase_increment_freq_min}, {cfg.local_phase_increment_freq_max}], "
            f"window={cfg.local_phase_increment_window_seconds}, "
            f"stride={cfg.local_phase_increment_stride_seconds}, "
            f"w_abs=({cfg.w_local_phase_absolute_x}, {cfg.w_local_phase_absolute_y}), "
            f"w_inc=({cfg.w_local_phase_increment_x}, {cfg.w_local_phase_increment_y}), "
            f"high_weight={cfg.local_phase_increment_high_power_weight}, "
            f"static_failure_weight={cfg.local_phase_increment_static_failure_weight}"
        )
    print(f"  use_continuous_phase_lag_loss = {cfg.use_continuous_phase_lag_loss}")
    if bool(cfg.use_continuous_phase_lag_loss):
        print(
            "  continuous_phase_lag = "
            f"obs={cfg.continuous_phase_observations}, "
            f"freq=[{cfg.continuous_phase_freq_min}, {cfg.continuous_phase_freq_max}], "
            f"n_freq_bins={cfg.continuous_phase_n_freq_bins}, "
            f"window={cfg.continuous_phase_window_seconds}, "
            f"stride={cfg.continuous_phase_stride_seconds}, "
            f"lag_scale={cfg.continuous_phase_time_shift_scale_seconds}, "
            f"w_abs=({cfg.w_continuous_phase_absolute_x}, {cfg.w_continuous_phase_absolute_y}), "
            f"w_shift=({cfg.w_continuous_phase_time_shift_x}, {cfg.w_continuous_phase_time_shift_y}), "
            f"base_weight={cfg.continuous_phase_base_weight}, "
            f"high_weight={cfg.continuous_phase_high_power_weight}, "
            f"static_failure_weight={cfg.continuous_phase_static_failure_weight}"
        )
    print(f"  use_local_phase_slope_loss = {cfg.use_local_phase_slope_loss}")
    if bool(cfg.use_local_phase_slope_loss):
        print(
            "  local_phase_slope = "
            f"obs={cfg.local_phase_slope_observations}, "
            f"freq=[{cfg.local_phase_slope_freq_min}, {cfg.local_phase_slope_freq_max}], "
            f"n_freq_bins={cfg.local_phase_slope_n_freq_bins}, "
            f"window={cfg.local_phase_slope_window_seconds}, "
            f"stride={cfg.local_phase_slope_stride_seconds}, "
            f"lag_slope_scale={cfg.local_phase_slope_time_shift_scale_seconds}, "
            f"w=({cfg.w_local_phase_slope_x}, {cfg.w_local_phase_slope_y}), "
            f"base_weight={cfg.local_phase_slope_base_weight}, "
            f"high_weight={cfg.local_phase_slope_high_power_weight}, "
            f"static_failure_weight={cfg.local_phase_slope_static_failure_weight}"
        )
    print(f"  use_slow_only_branch_diagnosis = {cfg.use_slow_only_branch_diagnosis}")
    if bool(cfg.use_slow_only_branch_diagnosis):
        print(
            "  slow_only_diagnosis = "
            f"w_good_no_reg={cfg.w_slow_good_no_regression}, "
            f"w_good_fast={cfg.w_slow_good_fast_suppress}, "
            f"w_bad_phase={cfg.w_slow_bad_phase}, "
            f"good_ratio_limit={cfg.slow_good_response_ratio_limit}, "
            f"good_corr_drop_tol={cfg.slow_good_corr_drop_tol}, "
            f"good_amp_log_tol={cfg.slow_good_amp_log_tol}, "
            f"bad_weight_max={cfg.slow_bad_weight_max}"
        )
    print(f"  use_static_quality_gate_suppression = {cfg.use_static_quality_gate_suppression}")
    if bool(cfg.use_static_quality_gate_suppression):
        print(
            "  static_quality_gate = "
            f"w_good_gate={cfg.w_static_good_gate_l1}, "
            f"obs={cfg.static_quality_observations}, "
            f"window={cfg.static_quality_window_seconds}, stride={cfg.static_quality_stride_seconds}, "
            f"good_corr>={cfg.static_quality_good_corr_threshold}, "
            f"good_lag<={cfg.static_quality_good_lag_seconds}, "
            f"good_amp_log_tol={cfg.static_quality_good_amp_log_tol}"
        )
    print(f"  fast_core_precompute_newmark = {cfg.fast_core_precompute_newmark}")
    print(f"  linear_solve_mode = {cfg.linear_solve_mode}")
    print(f"  output_dir = {output_dir}")
    print(f"  n_train_cases = {cfg.n_train_cases}")
    print(f"  n_valid_cases = {cfg.n_valid_cases}")
    print(f"  n_test_cases  = {cfg.n_test_cases}")

    _assert_existing_file(cfg.blade_csv, "blade_csv")

    # ------------------------------------------------------------
    # 1. Prepare / resolve case paths
    # ------------------------------------------------------------
    print()
    print("[1/6] Resolve training cases")

    print()
    print("[Load Files]")
    print("  train load files:")
    for p in train_load_files:
        print(f"    {p}")

    print("  valid load files:")
    for p in valid_load_files:
        print(f"    {p}")

    print("  test load files:")
    for p in test_load_files:
        print(f"    {p}")
    if bool(cfg.prepare_cases):
        prep_cfg = PhysicalTrainingCasePrepConfig(
            teacher_exe=str(cfg.teacher_exe),
            template_inp=str(cfg.template_inp),
            blade_csv=str(cfg.blade_csv),
            output_dir=str(cfg.training_case_dir),
            t_initial=float(cfg.t_initial),
            t_final=float(cfg.t_final),
            dt=float(cfg.dt),
            teacher_node_start=int(cfg.teacher_node_start),
            teacher_node_end=int(cfg.teacher_node_end),
            teacher_demean=bool(cfg.teacher_demean),
            zeta_structural=float(cfg.zeta_structural),
            ref_freq_hz=cfg.ref_freq_hz,
            use_base_initial_twist_phi=bool(cfg.use_base_initial_twist_phi),
            base_phi_twist_column=str(cfg.base_phi_twist_column),
            base_phi_sign=float(cfg.base_phi_sign),
            rotate_mass=bool(cfg.rotate_mass),
            remove_initial_offset=bool(cfg.remove_initial_offset),
            rebuild_cases=bool(cfg.rebuild_cases),
        )

        prepared = prepare_physical_training_cases(
            cfg=prep_cfg,
            train_load_files=train_load_files,
            valid_load_files=valid_load_files,
            rebuild=bool(cfg.rebuild_cases),
        )

        train_case_paths = [
            str(_expected_case_path(
                training_case_dir=cfg.training_case_dir,
                split="train",
                load_file=p,
            ))
            for p in train_load_files
        ]
        valid_case_paths = [str(p) for p in prepared["valid"]]

    if not train_case_paths:
        train_case_paths = [
            str(_expected_case_path(
                training_case_dir=cfg.training_case_dir,
                split="train",
                load_file=p,
            ))
            for p in train_load_files
        ]

    if not valid_case_paths:
        valid_case_paths = [
            str(_expected_case_path(
                training_case_dir=cfg.training_case_dir,
                split="valid",
                load_file=p,
            ))
            for p in valid_load_files
        ]

    print("  train cases:")
    for p in train_case_paths:
        print(f"    {p}")

    print("  valid cases:")
    for p in valid_case_paths:
        print(f"    {p}")

    train_cases = load_cases(
        train_case_paths,
        dtype=dtype_case,
        device=device,
        max_steps=cfg.max_steps_per_case,
        cfg=cfg,
    )

    valid_cases = load_cases(
        valid_case_paths,
        dtype=dtype_case,
        device=device,
        max_steps=cfg.max_steps_per_case,
        cfg=cfg,
    ) if len(valid_case_paths) > 0 else []

    if len(train_cases) == 0:
        raise RuntimeError("No training cases were loaded.")

    n_dofs = int(train_cases[0].F_raw.shape[-1])
    if n_dofs != 288:
        raise ValueError(f"Expected n_dofs=288, got {n_dofs}")

    n_nodes = n_dofs // 6

    x_idx_np = _component_indices(n_nodes, "x")
    y_idx_np = _component_indices(n_nodes, "y")
    tip_y_idx = _tip_component_index(n_nodes, "y")
    last5_y_idx_np = _last_k_component_indices(n_nodes, "y", last_k=5)

    x_idx = torch.as_tensor(x_idx_np, dtype=torch.long, device=device)
    y_idx = torch.as_tensor(y_idx_np, dtype=torch.long, device=device)
    last5_y_idx = torch.as_tensor(last5_y_idx_np, dtype=torch.long, device=device)

    if bool(cfg.use_cached_alignment_loss):
        print()
        print("[Alignment Loss Cache]")
        print("  building teacher-side spectral / peak / lag caches for train+valid cases ...")
        prime_alignment_loss_caches(cases=train_cases, cfg=cfg, x_idx=x_idx, y_idx=y_idx)
        prime_alignment_loss_caches(cases=valid_cases, cfg=cfg, x_idx=x_idx, y_idx=y_idx)
        print(f"  cached cases = {len(train_cases) + len(valid_cases)}")

    if bool(cfg.use_load_spectral_features):
        if not bool(cfg.use_load_branch):
            raise ValueError("--use-load-spectral-features requires --use-load-branch.")
        expected_dim = expected_load_spectral_feature_dim(
            observations=str(cfg.load_spectral_observations),
            bands=str(cfg.load_spectral_bands),
            include_fxy_resultant=True,
        )
        cfg.load_spectral_feature_dim = int(expected_dim)
        spectral_mean, spectral_std = fit_load_spectral_normalization(train_cases=train_cases)
        if spectral_mean is None or spectral_std is None:
            raise RuntimeError("use_load_spectral_features=True but no F_spectral was computed.")
        print()
        print("[Load Spectral Features]")
        print(f"  enabled = True")
        print(f"  feature_dim = {cfg.load_spectral_feature_dim}")
        print(f"  window_size = {_infer_load_spectral_window_size(cfg)}")
        print(f"  observations = {cfg.load_spectral_observations}")
        print(f"  bands = {cfg.load_spectral_bands}")
        if bool(cfg.use_load_spectral_disk_cache):
            print(f"  disk_cache_dir = {_default_load_spectral_cache_dir(cfg)}")
    else:
        spectral_mean = None
        spectral_std = None

    # ------------------------------------------------------------
    # 2. Geometry features
    # ------------------------------------------------------------
    print()
    print("[2/6] Build geometry features")

    geo_bundle = build_blade_geometry_features(
        BladeGeometryFeatureConfig(
            blade_csv=str(cfg.blade_csv),
            twist_column=str(cfg.base_phi_twist_column),
            phi_sign=float(cfg.base_phi_sign),
            exclude_root_station=True,
        )
    )

    geometry = torch.as_tensor(
        geo_bundle.features,
        dtype=_get_torch_dtype(cfg.encoder_dtype),
        device=device,
    )

    # ------------------------------------------------------------
    # 3. Registry + model
    # ------------------------------------------------------------
    print()
    print("[3/6] Build registry and model")

    registry = build_physical_parameter_registry(
        enabled_params=str(cfg.enabled_params),
    )

    model, model_info = build_training_model(
        cfg=cfg,
        registry=registry,
        geometry_dim=int(geo_bundle.feature_dim),
        dtype_core=dtype_core,
        device=device,
    )

    if bool(cfg.use_load_spectral_features) and spectral_mean is not None and spectral_std is not None:
        model.encoder.set_load_spectral_normalization(spectral_mean, spectral_std)

    init_checkpoint_report: Optional[dict[str, Any]] = None
    if cfg.init_checkpoint is not None and str(cfg.init_checkpoint).strip():
        init_checkpoint_path = Path(str(cfg.init_checkpoint)).expanduser()
        if not init_checkpoint_path.exists():
            raise FileNotFoundError(f"Initial checkpoint not found: {init_checkpoint_path}")

        checkpoint = torch.load(init_checkpoint_path, map_location=device)
        loaded_part, init_checkpoint_report = _load_initial_checkpoint_partial(
            model=model,
            checkpoint=checkpoint,
        )

        print()
        print("[Init Checkpoint]")
        print(f"  path        = {init_checkpoint_path}")
        print(f"  loaded_part = {loaded_part}")
        print(f"  exact_loaded   = {len(init_checkpoint_report['exact_loaded'])}")
        print(f"  partial_loaded = {len(init_checkpoint_report['partial_loaded'])}")
        if init_checkpoint_report["partial_loaded"]:
            for item in init_checkpoint_report["partial_loaded"][:12]:
                print(
                    "    partial "
                    f"{item['key']}: {item['source_shape']} -> {item['target_shape']}"
                )
        print(f"  skipped_shape  = {len(init_checkpoint_report['skipped_shape'])}")
        if init_checkpoint_report["skipped_shape"]:
            for item in init_checkpoint_report["skipped_shape"][:12]:
                print(
                    "    skipped "
                    f"{item['key']}: {item['source_shape']} -> {item['target_shape']}"
                )
        if init_checkpoint_report["unexpected"]:
            print(f"  unexpected     = {len(init_checkpoint_report['unexpected'])}")

    if init_checkpoint_report is not None:
        model_info["init_checkpoint_load"] = init_checkpoint_report

    beta_freeze_report: Optional[dict[str, Any]] = None
    if bool(cfg.freeze_alpha_backbone_for_beta):
        beta_freeze_report = _configure_beta_head_only_training(
            model=model,
            registry=registry,
        )
        model_info["beta_head_only_training"] = beta_freeze_report
        print()
        print("[Beta Head Only Training]")
        print(f"  beta_rows                = {beta_freeze_report['beta_rows']}")
        print(f"  masked_tensors           = {beta_freeze_report['masked_tensors']}")
        print(f"  trainable_encoder_params = {beta_freeze_report['trainable_encoder_params']}")
        print("  optimizer_weight_decay   = 0.0")

    optimizer_weight_decay = 0.0 if beta_freeze_report is not None else float(cfg.weight_decay)

    optimizer = torch.optim.AdamW(
        [p for p in model.encoder.parameters() if p.requires_grad],
        lr=float(cfg.lr),
        weight_decay=float(optimizer_weight_decay),
    )

    # ------------------------------------------------------------
    # 4. Save config
    # ------------------------------------------------------------
    run_config = {
        "cfg": asdict(cfg),
        "train_case_paths": train_case_paths,
        "valid_case_paths": valid_case_paths,
        "registry": registry.summary(),
        "geometry_summary": geo_bundle.summary(),
        "model_info": model_info,
    }

    _save_yaml(output_dir / "train_config.yaml", _make_json_safe(run_config))

    # ------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------
    print()
    print("[4/6] Training loop")

    history: list[dict[str, float]] = []

    best_score = float("inf")
    best_epoch = -1
    epochs_since_best = 0
    lr_plateau_count = 0

    best_path = output_dir / "best_transformer_physical_params.pt"
    last_path = output_dir / "last_transformer_physical_params.pt"

    last_valid_eval: Optional[dict[str, Any]] = None
    last_valid_epoch: int = -1

    for epoch in range(1, int(cfg.epochs) + 1):
        epoch_cfg, curriculum_metrics = make_epoch_curriculum_cfg(cfg, epoch)

        train_eval = train_cases_grad_accum(
            model=model,
            cases=train_cases,
            geometry=geometry,
            cfg=epoch_cfg,
            x_idx=x_idx,
            y_idx=y_idx,
            tip_y_idx=tip_y_idx,
            last5_y_idx=last5_y_idx,
            optimizer=optimizer,
        )

        grad_norm = train_eval["grad_norm"]

        model.eval()

        valid_every = max(1, int(cfg.valid_every))
        run_valid_this_epoch = (
                len(valid_cases) > 0
                and (
                        epoch == 1
                        or epoch % valid_every == 0
                        or epoch == int(cfg.epochs)
                )
        )

        if run_valid_this_epoch:
            valid_eval = evaluate_cases(
                model=model,
                cases=valid_cases,
                geometry=geometry,
                cfg=epoch_cfg,
                x_idx=x_idx,
                y_idx=y_idx,
                tip_y_idx=tip_y_idx,
                last5_y_idx=last5_y_idx,
                require_grad=False,
            )
            last_valid_eval = valid_eval
            last_valid_epoch = epoch
        else:
            valid_eval = last_valid_eval

        if bool(cfg.use_valid_for_best) and len(valid_cases) > 0:
            if valid_eval is None:
                score = float("inf")
            else:
                score = select_best_score(valid_eval, epoch_cfg)
        else:
            score = select_best_score(train_eval, epoch_cfg)

        # Construct metrics block for best checkpoint evaluation
        best_epoch_allowed = epoch >= int(cfg.best_start_epoch)
        if run_valid_this_epoch and valid_eval is not None:
            valid_x_for_best = float(valid_eval["x_ratio"].detach().cpu())
            valid_y_for_best = float(valid_eval["y_ratio"].detach().cpu())
            if _use_beta_alpha_reference_loss(epoch_cfg):
                alpha_x_for_best = _metric_to_float(valid_eval, "x_ratio_to_alpha")
                alpha_y_for_best = _metric_to_float(valid_eval, "y_ratio_to_alpha")
                if np.isfinite(alpha_x_for_best):
                    valid_x_for_best = alpha_x_for_best
                if np.isfinite(alpha_y_for_best):
                    valid_y_for_best = alpha_y_for_best
            valid_metrics_for_best = {
                "valid_y": valid_y_for_best,
                "valid_x": valid_x_for_best,
                "x_ratio": valid_x_for_best,
                "y_ratio": valid_y_for_best,
            }

            eligible_for_best, best_gate_reason = can_update_best_checkpoint(
                valid_metrics=valid_metrics_for_best,
                cfg=epoch_cfg,
            )
            if not best_epoch_allowed:
                eligible_for_best = False
                best_gate_reason = f"before_best_start_epoch({int(cfg.best_start_epoch)})"

            improved = bool(
                eligible_for_best
                and score < (best_score - float(cfg.early_stop_min_delta))
            )
        else:
            eligible_for_best = False
            best_gate_reason = f"validation_skipped(valid_every={int(cfg.valid_every)})"
            improved = False

        if improved:
            best_score = score
            best_epoch = epoch
            epochs_since_best = 0
            lr_plateau_count = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "encoder_state_dict": model.encoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "cfg": asdict(cfg),
                    "effective_cfg": asdict(epoch_cfg),
                    "curriculum_metrics": curriculum_metrics,
                    "registry": registry.summary(),
                    "best_score": best_score,
                    "best_gate_reason": str(best_gate_reason),
                    "train_eval": {
                        k: float(v.detach().cpu()) if torch.is_tensor(v) else v
                        for k, v in train_eval.items()
                        if k != "num_cases"
                    },
                    "valid_eval": {
                        k: float(v.detach().cpu()) if torch.is_tensor(v) else v
                        for k, v in valid_eval.items()
                        if k != "num_cases"
                    },
                },
                best_path,
            )
        else:
            # 只有真正做了 valid 检查的 epoch，才计入 early stop / LR plateau。
            # 否则 valid_every > 1 时，会因为跳过验证而过早触发 early stop。
            if run_valid_this_epoch and best_epoch_allowed:
                epochs_since_best += 1
                lr_plateau_count += 1

        if (
                int(cfg.lr_plateau_patience) > 0
                and lr_plateau_count >= int(cfg.lr_plateau_patience)
        ):
            for group in optimizer.param_groups:
                old_lr = float(group["lr"])
                new_lr = max(old_lr * float(cfg.lr_plateau_factor), float(cfg.min_lr))
                group["lr"] = new_lr
            lr_plateau_count = 0

        current_lr = float(optimizer.param_groups[0]["lr"])

        if valid_eval is None:
            valid_eval_for_log = {
                "total_loss": torch.as_tensor(float("nan")),
                "data_loss": torch.as_tensor(float("nan")),
                "response_loss": torch.as_tensor(float("nan")),
                "freq_loss": torch.as_tensor(float("nan")),
                "spectrum_loss": torch.as_tensor(float("nan")),
                "peak_time_loss": torch.as_tensor(float("nan")),
                "lag_loss": torch.as_tensor(float("nan")),
                "reg_loss": torch.as_tensor(float("nan")),
                "freq_x_spec_loss": torch.as_tensor(float("nan")),
                "freq_y_spec_loss": torch.as_tensor(float("nan")),
                "freq_x_peak_loss": torch.as_tensor(float("nan")),
                "freq_y_peak_loss": torch.as_tensor(float("nan")),
                "freq_x_pred_peak_hz": torch.as_tensor(float("nan")),
                "freq_x_target_peak_hz": torch.as_tensor(float("nan")),
                "freq_y_pred_peak_hz": torch.as_tensor(float("nan")),
                "freq_y_target_peak_hz": torch.as_tensor(float("nan")),
                "align_x_peak_time_loss": torch.as_tensor(float("nan")),
                "align_y_peak_time_loss": torch.as_tensor(float("nan")),
                "align_x_lag_loss": torch.as_tensor(float("nan")),
                "align_y_lag_loss": torch.as_tensor(float("nan")),
                "align_x_mean_abs_peak_time_error_s": torch.as_tensor(float("nan")),
                "align_y_mean_abs_peak_time_error_s": torch.as_tensor(float("nan")),
                "align_x_mean_abs_lag_s": torch.as_tensor(float("nan")),
                "align_y_mean_abs_lag_s": torch.as_tensor(float("nan")),
                "align_x_n_peak_events": torch.as_tensor(float("nan")),
                "align_y_n_peak_events": torch.as_tensor(float("nan")),
                "align_x_n_lag_windows": torch.as_tensor(float("nan")),
                "align_y_n_lag_windows": torch.as_tensor(float("nan")),
                "y_ratio": torch.as_tensor(float("nan")),
                "x_ratio": torch.as_tensor(float("nan")),
                "tip_y_ratio": torch.as_tensor(float("nan")),
                "last5_y_ratio": torch.as_tensor(float("nan")),
                "x_guard": torch.as_tensor(float("nan")),
                "theta_abs_max": torch.as_tensor(float("nan")),
            }
        else:
            valid_eval_for_log = valid_eval

        row = {
            "epoch": float(epoch),
            "lr": current_lr,
            "score": score,
            "best_score": best_score,
            "best_epoch": float(best_epoch),
            "best_eligible": int(eligible_for_best),
            "best_gate_reason": str(best_gate_reason),
            "best_start_epoch": float(cfg.best_start_epoch),
            "best_epoch_allowed": int(best_epoch_allowed),
            "best_score_mode": str(cfg.best_score_mode),
            "x_best_constraint_max": float(cfg.x_best_constraint_max),
            "use_x_constraint_for_best": int(bool(cfg.use_x_constraint_for_best)),
            **curriculum_metrics,

            "grad_norm": float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm),
            "valid_ran": int(bool(run_valid_this_epoch)),
            "last_valid_epoch": float(last_valid_epoch),

            "train_total_loss": float(train_eval["total_loss"].detach().cpu()),
            "train_data_loss": float(train_eval["data_loss"].detach().cpu()),
            "train_response_loss": float(train_eval["response_loss"].detach().cpu()),
            "train_freq_loss": float(train_eval["freq_loss"].detach().cpu()),
            "train_spectrum_loss": float(train_eval["spectrum_loss"].detach().cpu()),
            "train_peak_time_loss": float(train_eval["peak_time_loss"].detach().cpu()),
            "train_lag_loss": float(train_eval["lag_loss"].detach().cpu()),
            "train_reg_loss": float(train_eval["reg_loss"].detach().cpu()),
            "train_freq_x_spec_loss": float(train_eval["freq_x_spec_loss"].detach().cpu()),
            "train_freq_y_spec_loss": float(train_eval["freq_y_spec_loss"].detach().cpu()),
            "train_freq_x_peak_loss": float(train_eval["freq_x_peak_loss"].detach().cpu()),
            "train_freq_y_peak_loss": float(train_eval["freq_y_peak_loss"].detach().cpu()),
            "train_freq_x_pred_peak_hz": float(train_eval["freq_x_pred_peak_hz"].detach().cpu()),
            "train_freq_x_target_peak_hz": float(train_eval["freq_x_target_peak_hz"].detach().cpu()),
            "train_freq_y_pred_peak_hz": float(train_eval["freq_y_pred_peak_hz"].detach().cpu()),
            "train_freq_y_target_peak_hz": float(train_eval["freq_y_target_peak_hz"].detach().cpu()),
            "train_align_x_peak_time_loss": float(train_eval["align_x_peak_time_loss"].detach().cpu()),
            "train_align_y_peak_time_loss": float(train_eval["align_y_peak_time_loss"].detach().cpu()),
            "train_align_x_lag_loss": float(train_eval["align_x_lag_loss"].detach().cpu()),
            "train_align_y_lag_loss": float(train_eval["align_y_lag_loss"].detach().cpu()),
            "train_align_x_mean_abs_peak_time_error_s": float(train_eval["align_x_mean_abs_peak_time_error_s"].detach().cpu()),
            "train_align_y_mean_abs_peak_time_error_s": float(train_eval["align_y_mean_abs_peak_time_error_s"].detach().cpu()),
            "train_align_x_mean_abs_lag_s": float(train_eval["align_x_mean_abs_lag_s"].detach().cpu()),
            "train_align_y_mean_abs_lag_s": float(train_eval["align_y_mean_abs_lag_s"].detach().cpu()),
            "train_align_x_n_peak_events": float(train_eval["align_x_n_peak_events"].detach().cpu()),
            "train_align_y_n_peak_events": float(train_eval["align_y_n_peak_events"].detach().cpu()),
            "train_align_x_n_lag_windows": float(train_eval["align_x_n_lag_windows"].detach().cpu()),
            "train_align_y_n_lag_windows": float(train_eval["align_y_n_lag_windows"].detach().cpu()),
            "train_y_ratio": float(train_eval["y_ratio"].detach().cpu()),
            "train_x_ratio": float(train_eval["x_ratio"].detach().cpu()),
            "train_tip_y_ratio": float(train_eval["tip_y_ratio"].detach().cpu()),
            "train_last5_y_ratio": float(train_eval["last5_y_ratio"].detach().cpu()),
            "train_x_guard": float(train_eval["x_guard"].detach().cpu()),
            "train_theta_abs_max": float(train_eval["theta_abs_max"].detach().cpu()),

            "valid_total_loss": float(valid_eval_for_log["total_loss"].detach().cpu()),
            "valid_data_loss": float(valid_eval_for_log["data_loss"].detach().cpu()),
            "valid_response_loss": float(valid_eval_for_log["response_loss"].detach().cpu()),
            "valid_freq_loss": float(valid_eval_for_log["freq_loss"].detach().cpu()),
            "valid_spectrum_loss": float(valid_eval_for_log["spectrum_loss"].detach().cpu()),
            "valid_peak_time_loss": float(valid_eval_for_log["peak_time_loss"].detach().cpu()),
            "valid_lag_loss": float(valid_eval_for_log["lag_loss"].detach().cpu()),
            "valid_reg_loss": float(valid_eval_for_log["reg_loss"].detach().cpu()),
            "valid_freq_x_spec_loss": float(valid_eval_for_log["freq_x_spec_loss"].detach().cpu()),
            "valid_freq_y_spec_loss": float(valid_eval_for_log["freq_y_spec_loss"].detach().cpu()),
            "valid_freq_x_peak_loss": float(valid_eval_for_log["freq_x_peak_loss"].detach().cpu()),
            "valid_freq_y_peak_loss": float(valid_eval_for_log["freq_y_peak_loss"].detach().cpu()),
            "valid_freq_x_pred_peak_hz": float(valid_eval_for_log["freq_x_pred_peak_hz"].detach().cpu()),
            "valid_freq_x_target_peak_hz": float(valid_eval_for_log["freq_x_target_peak_hz"].detach().cpu()),
            "valid_freq_y_pred_peak_hz": float(valid_eval_for_log["freq_y_pred_peak_hz"].detach().cpu()),
            "valid_freq_y_target_peak_hz": float(valid_eval_for_log["freq_y_target_peak_hz"].detach().cpu()),
            "valid_align_x_peak_time_loss": float(valid_eval_for_log["align_x_peak_time_loss"].detach().cpu()),
            "valid_align_y_peak_time_loss": float(valid_eval_for_log["align_y_peak_time_loss"].detach().cpu()),
            "valid_align_x_lag_loss": float(valid_eval_for_log["align_x_lag_loss"].detach().cpu()),
            "valid_align_y_lag_loss": float(valid_eval_for_log["align_y_lag_loss"].detach().cpu()),
            "valid_align_x_mean_abs_peak_time_error_s": float(valid_eval_for_log["align_x_mean_abs_peak_time_error_s"].detach().cpu()),
            "valid_align_y_mean_abs_peak_time_error_s": float(valid_eval_for_log["align_y_mean_abs_peak_time_error_s"].detach().cpu()),
            "valid_align_x_mean_abs_lag_s": float(valid_eval_for_log["align_x_mean_abs_lag_s"].detach().cpu()),
            "valid_align_y_mean_abs_lag_s": float(valid_eval_for_log["align_y_mean_abs_lag_s"].detach().cpu()),
            "valid_align_x_n_peak_events": float(valid_eval_for_log["align_x_n_peak_events"].detach().cpu()),
            "valid_align_y_n_peak_events": float(valid_eval_for_log["align_y_n_peak_events"].detach().cpu()),
            "valid_align_x_n_lag_windows": float(valid_eval_for_log["align_x_n_lag_windows"].detach().cpu()),
            "valid_align_y_n_lag_windows": float(valid_eval_for_log["align_y_n_lag_windows"].detach().cpu()),
            "valid_y_ratio": float(valid_eval_for_log["y_ratio"].detach().cpu()),
            "valid_x_ratio": float(valid_eval_for_log["x_ratio"].detach().cpu()),
            "valid_tip_y_ratio": float(valid_eval_for_log["tip_y_ratio"].detach().cpu()),
            "valid_last5_y_ratio": float(valid_eval_for_log["last5_y_ratio"].detach().cpu()),
            "valid_x_guard": float(valid_eval_for_log["x_guard"].detach().cpu()),
            "valid_theta_abs_max": float(valid_eval_for_log["theta_abs_max"].detach().cpu()),
        }

        for phase_key in PHASE_GATED_LOG_METRIC_KEYS:
            row[f"train_{phase_key}"] = _metric_to_float(train_eval, phase_key)
            row[f"valid_{phase_key}"] = _metric_to_float(valid_eval_for_log, phase_key)
        for adaptive_key in ADAPTIVE_PHASE_LOG_METRIC_KEYS:
            row[f"train_{adaptive_key}"] = _metric_to_float(train_eval, adaptive_key)
            row[f"valid_{adaptive_key}"] = _metric_to_float(valid_eval_for_log, adaptive_key)
        for guard_key in NO_REGRESSION_LOG_METRIC_KEYS:
            row[f"train_{guard_key}"] = _metric_to_float(train_eval, guard_key)
            row[f"valid_{guard_key}"] = _metric_to_float(valid_eval_for_log, guard_key)
        for beta_key in BETA_ALPHA_LOG_METRIC_KEYS:
            row[f"train_{beta_key}"] = _metric_to_float(train_eval, beta_key)
            row[f"valid_{beta_key}"] = _metric_to_float(valid_eval_for_log, beta_key)
        for timing_key in TIMING_LOG_METRIC_KEYS:
            row[f"train_{timing_key}"] = _metric_to_float(train_eval, timing_key)
            row[f"valid_{timing_key}"] = _metric_to_float(valid_eval_for_log, timing_key)

        history.append(row)

        if epoch % int(cfg.print_every) == 0:
            use_alpha_print = (
                _use_beta_alpha_reference_loss(epoch_cfg)
                and np.isfinite(row.get("train_y_ratio_to_alpha", float("nan")))
                and np.isfinite(row.get("valid_y_ratio_to_alpha", float("nan")))
            )
            train_y_print = row["train_y_ratio_to_alpha"] if use_alpha_print else row["train_y_ratio"]
            train_x_print = row["train_x_ratio_to_alpha"] if use_alpha_print else row["train_x_ratio"]
            valid_y_print = row["valid_y_ratio_to_alpha"] if use_alpha_print else row["valid_y_ratio"]
            valid_x_print = row["valid_x_ratio_to_alpha"] if use_alpha_print else row["valid_x_ratio"]
            ratio_label = "_to_alpha" if use_alpha_print else ""
            print(
                f"[Epoch {epoch:04d}] "
                f"score={score:.6e} best={best_score:.6e}@{best_epoch} "
                f"train_freq={row['train_freq_loss']:.3e} valid_freq={row['valid_freq_loss']:.3e} "
                f"valid_peak={row['valid_peak_time_loss']:.3e} valid_lag={row['valid_lag_loss']:.3e} "
                f"train_y{ratio_label}={train_y_print:.6f} train_x{ratio_label}={train_x_print:.6f} "
                f"valid_y{ratio_label}={valid_y_print:.6f} valid_x{ratio_label}={valid_x_print:.6f} "
                f"best_mode={cfg.best_score_mode} best_gate={'PASS' if eligible_for_best else 'FAIL'} "
                f"theta_max={row['train_theta_abs_max']:.4e} "
                f"lr={current_lr:.3e}"
            )
            if bool(cfg.use_loss_curriculum):
                print(
                    "          "
                    f"curriculum phase={row['curriculum_phase_scale']:.3f} "
                    f"lag={row['curriculum_lag_scale']:.3f} "
                    f"guard={row['curriculum_guard_scale']:.3f} "
                    f"gate_reg={row['curriculum_gate_reg_scale']:.3f} "
                    f"w_drift_y=({row['effective_w_phase_drift_lag_y']:.3e},"
                    f"{row['effective_w_phase_drift_rate_y']:.3e}) "
                    f"w_gate_l1={row['effective_w_phase_gate_l1']:.3e} "
                    f"w_gate_boot={row['effective_w_phase_gate_bootstrap']:.3e}"
                )
            if bool(cfg.profile_train_timing):
                print(
                    "          "
                    f"timing_train_total={row['train_timing_total_seconds']:.2f}s "
                    f"forward={row['train_timing_model_forward_seconds']:.2f}s "
                    f"encoder={row['train_timing_encoder_seconds']:.2f}s "
                    f"newmark={row['train_timing_newmark_loop_seconds']:.2f}s "
                    f"assemble={row['train_timing_newmark_assemble_seconds']:.2f}s "
                    f"rhs={row['train_timing_newmark_rhs_seconds']:.2f}s "
                    f"solve={row['train_timing_newmark_solve_seconds']:.2f}s "
                    f"update={row['train_timing_newmark_update_seconds']:.2f}s "
                    f"loss={row['train_timing_loss_seconds']:.2f}s "
                    f"backward={row['train_timing_backward_seconds']:.2f}s "
                    f"valid_total={row['valid_timing_total_seconds']:.2f}s"
                )

        if int(cfg.save_every) > 0 and epoch % int(cfg.save_every) == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "encoder_state_dict": model.encoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "cfg": asdict(cfg),
                    "effective_cfg": asdict(epoch_cfg),
                    "curriculum_metrics": curriculum_metrics,
                    "registry": registry.summary(),
                    "score": score,
                },
                output_dir / f"checkpoint_epoch_{epoch:04d}.pt",
            )

        if (
                epoch >= int(cfg.min_epochs)
                and int(cfg.early_stop_patience) > 0
                and epochs_since_best >= int(cfg.early_stop_patience)
        ):
            print()
            print(
                f"[Early Stop] epoch={epoch}, best_epoch={best_epoch}, "
                f"best_score={best_score:.6e}"
            )
            break

        if current_lr <= float(cfg.min_lr) and epochs_since_best >= int(cfg.early_stop_patience):
            print()
            print("[Stop] learning rate reached min_lr and no improvement.")
            break

    # ------------------------------------------------------------
    # 6. Save final & Run Test
    # ------------------------------------------------------------
    print()
    print("[5/6] Save final artifacts")

    final_epoch_cfg, final_curriculum_metrics = make_epoch_curriculum_cfg(
        cfg,
        int(history[-1]["epoch"]) if history else 0,
    )
    torch.save(
        {
            "epoch": history[-1]["epoch"] if history else 0,
            "model_state_dict": model.state_dict(),
            "encoder_state_dict": model.encoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "cfg": asdict(cfg),
            "effective_cfg": asdict(final_epoch_cfg),
            "curriculum_metrics": final_curriculum_metrics,
            "registry": registry.summary(),
            "best_score": best_score,
            "best_epoch": best_epoch,
        },
        last_path,
    )

    _save_history_csv(output_dir / "training_history.csv", history)

    post_train_test_results: list[dict[str, Any]] = []

    if bool(cfg.run_test_after_training):
        print()
        print("[Post-train Test] Enabled")

        if best_epoch < 0 or not best_path.exists():
            print(
                "[Post-train Test] SKIP: no valid best checkpoint was saved. "
                "Check x-best constraint or valid metrics."
            )
        else:
            post_train_test_results = run_post_training_tests(
                cfg=cfg,
                best_checkpoint=best_path,
                output_dir=output_dir,
                test_load_files=test_load_files,
            )

            with open(output_dir / "post_train_test_summary.json", "w", encoding="utf-8") as f:
                json.dump(
                    _make_json_safe(
                        {
                            "best_checkpoint": str(best_path),
                            "best_epoch": best_epoch,
                            "test_max_steps": int(cfg.test_max_steps),
                            "test_load_files": test_load_files,
                            "results": post_train_test_results,
                        }
                    ),
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

    summary = {
        "best_score": best_score,
        "best_epoch": best_epoch,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "num_history_rows": len(history),
        "final_row": history[-1] if history else {},
        "use_x_constraint_for_best": bool(cfg.use_x_constraint_for_best),
        "x_best_constraint_max": float(cfg.x_best_constraint_max),
        "best_score_mode": str(cfg.best_score_mode),
        "best_selection_rule": (
            f"min {cfg.best_score_mode} score subject to valid_x <= x_best_constraint_max"
            if bool(cfg.use_x_constraint_for_best)
            else f"min {cfg.best_score_mode} score"
        ),
        "config": asdict(cfg),
        "registry": registry.summary(),
        "run_test_after_training": bool(cfg.run_test_after_training),
        "test_load_files": test_load_files,
        "test_max_steps": int(cfg.test_max_steps),
        "post_train_test_summary": (
            str(output_dir / "post_train_test_summary.json")
            if bool(cfg.run_test_after_training)
            else None
        ),
    }

    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(_make_json_safe(summary), f, indent=2, ensure_ascii=False)

    print()
    print("[6/6] Done")
    print(f"  best_epoch      = {best_epoch}")
    print(f"  best_score      = {best_score:.6e}")
    print(f"  best_checkpoint = {best_path}")
    print(f"  history_csv     = {output_dir / 'training_history.csv'}")
    print(f"  summary_json    = {output_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()
