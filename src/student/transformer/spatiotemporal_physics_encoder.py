from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from src.student.transformer.physical_parameter_heads import (
    BoundedPhysicalParameterHead,
    PhysicalParameterHeadConfig,
    PhaseGatedPhysicalParameterHead,
    PhaseGatedPhysicalParameterHeadConfig,
)
from src.student.transformer.physical_parameter_registry import (
    PhysicalParameterRegistry,
    build_physical_parameter_registry,
)
from src.student.transformer.spatial_attention_encoders import (
    GatedBranchFusion,
    GeometryBranchEncoder,
    LoadBranchEncoder,
    ResponseBranchEncoder,
    SpatialBranchEncoderConfig,
)


@dataclass
class SpatiotemporalPhysicsEncoderConfig:
    """
    第一阶段 Transformer 物理参数编码器配置。

    当前第一阶段推荐：
        enabled_params = "alpha_y"
        use_response_branch = True
        use_load_branch = False or True
        use_geometry_branch = False or True

    目标：
        根据 response/load/geometry 三类信息输出 theta_t。
    """

    n_nodes: int = 48
    dof_per_node: int = 6

    d_model: int = 64
    n_spatial_heads: int = 4
    spatial_mlp_hidden_dim: int = 128

    n_temporal_layers: int = 2
    n_temporal_heads: int = 4
    temporal_ff_dim: int = 128
    dropout: float = 0.0

    use_response_branch: bool = True
    use_load_branch: bool = False
    use_geometry_branch: bool = False

    # Load branch 内部可选的外力频域/脉冲特征分支。
    # 该分支不会改变 F 本身；F 仍然作为物理外力进入 Newmark/MCK core。
    # 若 forward 未显式传入 load_spectral_features，则 encoder 会从 F 在线计算 causal 局部特征。
    use_load_spectral_features: bool = False
    load_spectral_feature_dim: int | None = None
    load_spectral_window_size: int | None = None
    load_spectral_freq_min: float = 0.05
    load_spectral_freq_max: float = 5.0
    load_spectral_bands: str = "0.05-0.5,0.5-1.5,1.5-5.0"
    load_spectral_observations: str = "tip,last5,mean"
    load_spectral_last_k: int = 5
    load_spectral_active_rel_threshold: float = 1.0e-3
    load_spectral_active_abs_threshold: float = 1.0e-12
    load_spectral_normalize: bool = True

    # 如果 True，则 response/load branch 的 node token 会额外加入 geometry embedding。
    # 若想做严格 no-geometry ablation，应设为 False 且 use_geometry_branch=False。
    condition_dynamic_branches_on_geometry: bool = True

    causal_temporal: bool = True

    # 固定长度 temporal lookback window。
    # None 表示保持原来的 full-history causal attention。
    # 若设为 W，则 theta_t 只能 attend 到最近 W 个时间步：
    #   [max(0, t-W+1), ..., t]
    temporal_window_size: int | None = None

    use_temporal_transformer: bool = True

    # Optional slow + phase-gated fast decomposition head.
    # False keeps the original single bounded theta head and preserves old checkpoints/experiments.
    use_phase_gated_decomposition: bool = False
    phase_slow_scale: float = 1.0
    phase_fast_scale: float = 0.5
    phase_gate_init_bias: float = -4.0
    phase_total_clip_scale: float = 1.0


@dataclass
class SpatiotemporalPhysicsEncoderOutput:
    theta: torch.Tensor
    raw_theta: torch.Tensor
    theta_dict: dict[str, torch.Tensor]
    temporal_hidden: torch.Tensor
    fused_branch_embedding: torch.Tensor
    branch_embeddings: dict[str, torch.Tensor]
    spatial_attention_weights: dict[str, torch.Tensor]
    fusion_gate_weights: dict[str, torch.Tensor]
    theta_aux: dict[str, torch.Tensor] | None = None


class SinusoidalTemporalPositionalEncoding(nn.Module):
    """
    标准 sinusoidal 时间位置编码。

    输入：
        x: shape = (B,T,D)

    输出：
        x + pe[:T]
    """

    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()

        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}.")

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must have shape (B,T,D), got {tuple(x.shape)}.")

        T = int(x.shape[1])
        if T > self.pe.shape[1]:
            raise ValueError(
                f"Sequence length T={T} exceeds max_len={self.pe.shape[1]}."
            )

        return x + self.pe[:, :T, :].to(device=x.device, dtype=x.dtype)


def full_state_to_node_response_features(
    *,
    u: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    n_nodes: int = 48,
    dof_per_node: int = 6,
) -> torch.Tensor:
    """
    将 full-order flat response 转为节点级 response feature。

    输入：
        u, v, a: shape = (..., 288)

    输出：
        response_node_features: shape = (..., 48, 18)
        每个节点为 [u_i(6), v_i(6), a_i(6)]
    """
    if not (u.shape == v.shape == a.shape):
        raise ValueError(
            f"u/v/a shape mismatch: u={tuple(u.shape)}, "
            f"v={tuple(v.shape)}, a={tuple(a.shape)}."
        )

    expected_d = int(n_nodes) * int(dof_per_node)

    if u.shape[-1] != expected_d:
        raise ValueError(
            f"full state last dim mismatch: expected {expected_d}, got {u.shape[-1]}."
        )

    prefix = u.shape[:-1]

    u_node = u.reshape(*prefix, n_nodes, dof_per_node)
    v_node = v.reshape(*prefix, n_nodes, dof_per_node)
    a_node = a.reshape(*prefix, n_nodes, dof_per_node)

    return torch.cat([u_node, v_node, a_node], dim=-1)


def full_force_to_node_load_features(
    *,
    F: torch.Tensor,
    n_nodes: int = 48,
    dof_per_node: int = 6,
) -> torch.Tensor:
    """
    将 full-order flat force 转为节点级 load feature。

    输入：
        F: shape = (..., 288)

    输出：
        load_node_features: shape = (..., 48, 6)
    """
    expected_d = int(n_nodes) * int(dof_per_node)

    if F.shape[-1] != expected_d:
        raise ValueError(
            f"force last dim mismatch: expected {expected_d}, got {F.shape[-1]}."
        )

    prefix = F.shape[:-1]
    return F.reshape(*prefix, n_nodes, dof_per_node)


def _parse_frequency_bands(text: str) -> list[tuple[float, float]]:
    bands: list[tuple[float, float]] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        lo_s, hi_s = item.split("-", 1)
        lo = float(lo_s)
        hi = float(hi_s)
        if hi <= lo:
            raise ValueError(f"Invalid frequency band {item!r}.")
        bands.append((lo, hi))
    return bands


def _parse_observations(text: str) -> list[str]:
    return [x.strip().lower() for x in str(text).split(",") if x.strip()]


def expected_load_spectral_feature_dim(
    *,
    observations: str = "tip,last5,mean",
    bands: str = "0.05-0.5,0.5-1.5,1.5-5.0",
    include_fxy_resultant: bool = True,
) -> int:
    """Return the deterministic feature dimension used by the load spectral branch."""
    n_obs = len(_parse_observations(observations))
    n_signal = 2 * n_obs + (n_obs if include_fxy_resultant else 0)  # Fx, Fy, optional Fxy
    # dominant_freq, centroid, entropy, peak_sharpness,
    # active_fraction, active_segments, effective_cycles_active_span,
    # pulse_like_score, sustained_oscillation_score, plus band ratios.
    n_feat_per_signal = 9 + len(_parse_frequency_bands(bands))
    return int(n_signal * n_feat_per_signal)


def _safe_nan_to_num(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def compute_causal_load_spectral_features_from_force(
    F: torch.Tensor,
    *,
    n_nodes: int = 48,
    dof_per_node: int = 6,
    window_size: int | None = None,
    dt: float = 0.01,
    freq_min: float = 0.05,
    freq_max: float = 5.0,
    bands: str = "0.05-0.5,0.5-1.5,1.5-5.0",
    observations: str = "tip,last5,mean",
    last_k: int = 5,
    active_rel_threshold: float = 1.0e-3,
    active_abs_threshold: float = 1.0e-12,
) -> torch.Tensor:
    """
    从 full-order 外力 F 在线计算 causal 局部时域/频域特征。

    输入：
        F: shape = (B,T,N*6) 或 (T,N*6)

    输出：
        load_spectral_features: shape = (B,T,D_spec) 或 (T,D_spec)

    设计原则：
        1. 只使用 t 时刻之前的局部窗口 F[:, max(0,t-W+1):t+1]，不泄漏未来；
        2. 同时包含 FFT 频谱特征和 short-pulse / sustained-oscillation 时域判据；
        3. 该函数不需要梯度，建议在训练 case loading 阶段预先算一次。
    """
    squeeze_batch = False
    if F.ndim == 2:
        F = F.unsqueeze(0)
        squeeze_batch = True
    if F.ndim != 3:
        raise ValueError(f"F must have shape (B,T,D) or (T,D), got {tuple(F.shape)}.")

    B, T, D = int(F.shape[0]), int(F.shape[1]), int(F.shape[2])
    expected_d = int(n_nodes) * int(dof_per_node)
    if D != expected_d:
        raise ValueError(f"F last dim mismatch: expected {expected_d}, got {D}.")

    device = F.device
    dtype = torch.float32 if F.dtype in (torch.float16, torch.bfloat16) else F.dtype
    F_work = F.to(dtype=dtype)
    F_node = F_work.reshape(B, T, n_nodes, dof_per_node)
    Fx = F_node[..., 0]
    Fy = F_node[..., 1]
    Fxy = torch.sqrt(Fx ** 2 + Fy ** 2)

    obs_list = _parse_observations(observations)
    last_k = max(1, min(int(last_k), int(n_nodes)))

    signal_list: list[torch.Tensor] = []
    for comp in (Fx, Fy, Fxy):
        for obs in obs_list:
            if obs == "tip":
                signal_list.append(comp[:, :, -1])
            elif obs == "root":
                signal_list.append(comp[:, :, 0])
            elif obs == "mean":
                signal_list.append(comp.mean(dim=-1))
            elif obs == "last5":
                signal_list.append(comp[:, :, -last_k:].mean(dim=-1))
            elif obs == "rms":
                signal_list.append(torch.sqrt(torch.mean(comp ** 2, dim=-1)))
            elif obs.startswith("node"):
                node_idx = int(obs.replace("node", ""))
                if node_idx < 1 or node_idx > n_nodes:
                    raise ValueError(f"Observation {obs!r} out of range 1..{n_nodes}.")
                signal_list.append(comp[:, :, node_idx - 1])
            else:
                raise ValueError(f"Unsupported load spectral observation {obs!r}.")

    bands_parsed = _parse_frequency_bands(bands)
    n_feat_expected = expected_load_spectral_feature_dim(
        observations=observations,
        bands=bands,
        include_fxy_resultant=True,
    )

    W = int(window_size) if window_size is not None else int(T)
    W = max(4, min(W, T))
    dt = float(dt)

    all_t_features: list[torch.Tensor] = []
    eps = torch.as_tensor(1.0e-12, dtype=dtype, device=device)

    with torch.no_grad():
        for t in range(T):
            start = max(0, t - W + 1)
            step_features: list[torch.Tensor] = []
            for sig in signal_list:
                y = sig[:, start:t + 1]
                L = int(y.shape[1])

                max_abs = torch.amax(torch.abs(y), dim=1)
                active_threshold = torch.maximum(
                    torch.as_tensor(float(active_abs_threshold), dtype=dtype, device=device),
                    torch.as_tensor(float(active_rel_threshold), dtype=dtype, device=device) * max_abs,
                )
                active = torch.abs(y) >= active_threshold[:, None]
                active_fraction = active.to(dtype).mean(dim=1)

                # Segment count is approximated by active rising edges.
                if L >= 2:
                    prev = torch.zeros_like(active)
                    prev[:, 1:] = active[:, :-1]
                    active_segments = (active & (~prev)).to(dtype).sum(dim=1)
                else:
                    active_segments = active.to(dtype).sum(dim=1)

                if L >= 4 and torch.any(max_abs > eps):
                    y_demean = y - y.mean(dim=1, keepdim=True)
                    win = torch.hann_window(L, periodic=False, dtype=dtype, device=device).view(1, L)
                    yw = y_demean * win
                    fft = torch.fft.rfft(yw, dim=1)
                    freq = torch.fft.rfftfreq(L, d=dt).to(device=device, dtype=dtype)
                    power = (torch.abs(fft) ** 2) / torch.clamp(torch.sum(win ** 2), min=eps)

                    mask = (freq >= float(freq_min)) & (freq <= float(freq_max)) & (freq > 0.0)
                    if torch.any(mask):
                        f = freq[mask]
                        p = power[:, mask]
                        total_power = p.sum(dim=1)
                        valid_power = total_power > eps
                        p_norm = p / torch.clamp(total_power[:, None], min=eps)
                        imax = torch.argmax(p, dim=1)
                        dominant_freq = f[imax]
                        dominant_power = torch.gather(p, 1, imax[:, None]).squeeze(1)
                        centroid = (p_norm * f[None, :]).sum(dim=1)
                        import math
                        entropy_norm = max(math.log(float(max(len(f), 2))), 1.0e-12)
                        entropy = -(p_norm * torch.log(p_norm + eps)).sum(dim=1) / entropy_norm
                        peak_sharpness = dominant_power / torch.clamp(total_power, min=eps)
                        band_ratios = []
                        for lo, hi in bands_parsed:
                            bmask = (f >= float(lo)) & (f < float(hi))
                            if torch.any(bmask):
                                band_power = p[:, bmask].sum(dim=1)
                            else:
                                band_power = torch.zeros(B, dtype=dtype, device=device)
                            band_ratios.append(band_power / torch.clamp(total_power, min=eps))

                        dominant_freq = torch.where(valid_power, dominant_freq, torch.zeros_like(dominant_freq))
                        centroid = torch.where(valid_power, centroid, torch.zeros_like(centroid))
                        entropy = torch.where(valid_power, entropy, torch.zeros_like(entropy))
                        peak_sharpness = torch.where(valid_power, peak_sharpness, torch.zeros_like(peak_sharpness))
                    else:
                        dominant_freq = torch.zeros(B, dtype=dtype, device=device)
                        centroid = torch.zeros(B, dtype=dtype, device=device)
                        entropy = torch.zeros(B, dtype=dtype, device=device)
                        peak_sharpness = torch.zeros(B, dtype=dtype, device=device)
                        band_ratios = [torch.zeros(B, dtype=dtype, device=device) for _ in bands_parsed]
                else:
                    dominant_freq = torch.zeros(B, dtype=dtype, device=device)
                    centroid = torch.zeros(B, dtype=dtype, device=device)
                    entropy = torch.zeros(B, dtype=dtype, device=device)
                    peak_sharpness = torch.zeros(B, dtype=dtype, device=device)
                    band_ratios = [torch.zeros(B, dtype=dtype, device=device) for _ in bands_parsed]

                # Active span duration and effective cycles inside active span.
                time_idx = torch.arange(L, device=device, dtype=dtype).view(1, L)
                active_float = active.to(dtype)
                has_active = active_float.sum(dim=1) > 0
                first_idx = torch.argmax(active_float, dim=1).to(dtype)
                last_idx = (L - 1) - torch.argmax(torch.flip(active_float, dims=[1]), dim=1).to(dtype)
                active_span_duration = torch.where(
                    has_active,
                    (last_idx - first_idx + 1.0) * dt,
                    torch.zeros(B, dtype=dtype, device=device),
                )
                effective_cycles_active_span = dominant_freq * active_span_duration

                # Pulse / sustained oscillation heuristic scores in [0,1].
                pulse = torch.zeros(B, dtype=dtype, device=device)
                pulse = pulse + torch.where(active_fraction < 0.35, 0.45, 0.0)
                pulse = pulse + torch.where(active_segments <= 2.0, 0.25, 0.0)
                pulse = pulse + torch.where(effective_cycles_active_span < 1.5, 0.20, 0.0)
                # A cheap zero-crossing proxy: if no clear spectral peak and active is short, treat as pulse-like.
                pulse = torch.clamp(pulse + torch.where(peak_sharpness < 0.05, 0.10, 0.0), 0.0, 1.0)

                osc = torch.zeros(B, dtype=dtype, device=device)
                osc = osc + torch.where(active_fraction > 0.65, 0.30, 0.0)
                osc = osc + torch.where((dominant_freq * (float(L) * dt)) >= 2.0, 0.25, 0.0)
                osc = osc + torch.where(active_span_duration > 0.6 * float(L) * dt, 0.20, 0.0)
                osc = osc + torch.where(peak_sharpness > 0.15, 0.25, 0.0)
                osc = torch.clamp(osc, 0.0, 1.0)

                one_signal = [
                    dominant_freq,
                    centroid,
                    entropy,
                    peak_sharpness,
                    active_fraction,
                    active_segments,
                    effective_cycles_active_span,
                    pulse,
                    osc,
                    *band_ratios,
                ]
                step_features.extend(one_signal)

            step_feat = torch.stack([_safe_nan_to_num(x) for x in step_features], dim=1)
            all_t_features.append(step_feat)

        out = torch.stack(all_t_features, dim=1)

    if int(out.shape[-1]) != n_feat_expected:
        raise RuntimeError(
            f"load spectral feature dim mismatch: expected {n_feat_expected}, got {out.shape[-1]}."
        )

    if squeeze_batch:
        out = out.squeeze(0)
    return out


class SpatiotemporalPhysicsEncoder(nn.Module):
    """
    第一阶段 geometry-aware spatiotemporal physics encoder。

    输入可以是：
        1. 已经整理好的 node features:
            response_node_features: (B,T,N,18)
            load_node_features:     (B,T,N,6)

        2. 或者 full-order flat states:
            u, v, a, F: (B,T,288)

    输出：
        theta: (B,T,P)

    当前第一阶段建议 P=1:
        theta = [alpha_y]

    后续可通过 registry 扩展到：
        [alpha_y, alpha_xy]
    """

    def __init__(
        self,
        *,
        geometry_dim: int | None,
        registry: PhysicalParameterRegistry | None = None,
        enabled_params: str | list[str] | tuple[str, ...] | None = "alpha_y",
        config: SpatiotemporalPhysicsEncoderConfig | None = None,
    ):
        super().__init__()

        self.config = config or SpatiotemporalPhysicsEncoderConfig()
        self.registry = registry or build_physical_parameter_registry(
            enabled_params=enabled_params,
        )

        if not (
            self.config.use_response_branch
            or self.config.use_load_branch
            or self.config.use_geometry_branch
        ):
            raise ValueError(
                "At least one branch must be enabled among response/load/geometry."
            )

        if self.config.use_load_spectral_features and not self.config.use_load_branch:
            raise ValueError(
                "use_load_spectral_features=True requires use_load_branch=True, "
                "because spectral features are fused inside the load branch."
            )

        if geometry_dim is None:
            if self.config.use_geometry_branch or self.config.condition_dynamic_branches_on_geometry:
                raise ValueError(
                    "geometry_dim is required when geometry branch or geometry conditioning is enabled."
                )

        self.geometry_dim = geometry_dim

        branch_config_response = SpatialBranchEncoderConfig(
            input_dim=18,
            d_model=self.config.d_model,
            n_heads=self.config.n_spatial_heads,
            mlp_hidden_dim=self.config.spatial_mlp_hidden_dim,
            dropout=self.config.dropout,
        )

        branch_config_load = SpatialBranchEncoderConfig(
            input_dim=6,
            d_model=self.config.d_model,
            n_heads=self.config.n_spatial_heads,
            mlp_hidden_dim=self.config.spatial_mlp_hidden_dim,
            dropout=self.config.dropout,
        )

        branch_config_geometry = None
        if geometry_dim is not None:
            branch_config_geometry = SpatialBranchEncoderConfig(
                input_dim=int(geometry_dim),
                d_model=self.config.d_model,
                n_heads=self.config.n_spatial_heads,
                mlp_hidden_dim=self.config.spatial_mlp_hidden_dim,
                dropout=self.config.dropout,
            )

        self.response_branch = None
        if self.config.use_response_branch:
            self.response_branch = ResponseBranchEncoder(
                config=branch_config_response,
                geometry_dim=(
                    int(geometry_dim)
                    if self.config.condition_dynamic_branches_on_geometry and geometry_dim is not None
                    else None
                ),
            )

        self.load_branch = None
        if self.config.use_load_branch:
            self.load_branch = LoadBranchEncoder(
                config=branch_config_load,
                geometry_dim=(
                    int(geometry_dim)
                    if self.config.condition_dynamic_branches_on_geometry and geometry_dim is not None
                    else None
                ),
            )

        self.load_spectral_branch = None
        if self.config.use_load_spectral_features:
            feature_dim = self.config.load_spectral_feature_dim
            if feature_dim is None:
                feature_dim = expected_load_spectral_feature_dim(
                    observations=self.config.load_spectral_observations,
                    bands=self.config.load_spectral_bands,
                    include_fxy_resultant=True,
                )
                self.config.load_spectral_feature_dim = int(feature_dim)
            feature_dim = int(feature_dim)
            self.load_spectral_branch = nn.Sequential(
                nn.Linear(feature_dim, self.config.d_model),
                nn.GELU(),
                nn.LayerNorm(self.config.d_model),
                nn.Dropout(self.config.dropout),
            )
            self.register_buffer(
                "load_spectral_mean",
                torch.zeros(feature_dim, dtype=torch.float32),
                persistent=True,
            )
            self.register_buffer(
                "load_spectral_std",
                torch.ones(feature_dim, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.register_buffer("load_spectral_mean", torch.zeros(0, dtype=torch.float32), persistent=True)
            self.register_buffer("load_spectral_std", torch.ones(0, dtype=torch.float32), persistent=True)

        self.geometry_branch = None
        if self.config.use_geometry_branch:
            if branch_config_geometry is None:
                raise ValueError("geometry_dim is required for geometry branch.")
            self.geometry_branch = GeometryBranchEncoder(
                config=branch_config_geometry,
            )

        self.fusion = GatedBranchFusion(
            d_model=self.config.d_model,
            allowed_branches=("response", "load", "geometry"),
        )

        self.pos_encoding = SinusoidalTemporalPositionalEncoding(
            d_model=self.config.d_model,
        )

        if self.config.use_temporal_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.config.d_model,
                nhead=self.config.n_temporal_heads,
                dim_feedforward=self.config.temporal_ff_dim,
                dropout=self.config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )

            self.temporal_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=self.config.n_temporal_layers,
            )
        else:
            self.temporal_encoder = nn.Identity()

        if bool(self.config.use_phase_gated_decomposition):
            self.head = PhaseGatedPhysicalParameterHead(
                registry=self.registry,
                config=PhaseGatedPhysicalParameterHeadConfig(
                    d_model=self.config.d_model,
                    hidden_dim=self.config.temporal_ff_dim,
                    dropout=self.config.dropout,
                    use_layernorm=True,
                    slow_scale=float(self.config.phase_slow_scale),
                    fast_scale=float(self.config.phase_fast_scale),
                    gate_init_bias=float(self.config.phase_gate_init_bias),
                    total_clip_scale=float(self.config.phase_total_clip_scale),
                ),
            )
        else:
            self.head = BoundedPhysicalParameterHead(
                registry=self.registry,
                config=PhysicalParameterHeadConfig(
                    d_model=self.config.d_model,
                    hidden_dim=self.config.temporal_ff_dim,
                    dropout=self.config.dropout,
                    use_layernorm=True,
                ),
            )

    def set_load_spectral_normalization(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> None:
        """Set train-set normalization statistics for the load spectral branch."""
        if not self.config.use_load_spectral_features:
            return
        mean = mean.detach().to(device=self.load_spectral_mean.device, dtype=self.load_spectral_mean.dtype).reshape(-1)
        std = std.detach().to(device=self.load_spectral_std.device, dtype=self.load_spectral_std.dtype).reshape(-1)
        if mean.shape != self.load_spectral_mean.shape:
            raise ValueError(
                f"load spectral mean shape mismatch: expected {tuple(self.load_spectral_mean.shape)}, got {tuple(mean.shape)}."
            )
        if std.shape != self.load_spectral_std.shape:
            raise ValueError(
                f"load spectral std shape mismatch: expected {tuple(self.load_spectral_std.shape)}, got {tuple(std.shape)}."
            )
        std = torch.clamp(std, min=1.0e-6)
        self.load_spectral_mean.copy_(mean)
        self.load_spectral_std.copy_(std)

    def _make_causal_mask(
        self,
        T: int,
        *,
        device: torch.device,
    ) -> torch.Tensor | None:
        """
        构造 temporal attention mask。

        PyTorch TransformerEncoder 的 bool mask 语义：
            True  = 不允许 attention
            False = 允许 attention

        默认 full-history causal:
            第 t 步允许看 0...t。

        若 temporal_window_size = W:
            第 t 步只允许看 max(0, t-W+1)...t。
        """
        window_size = self.config.temporal_window_size

        if not self.config.causal_temporal and window_size is None:
            return None

        idx = torch.arange(T, device=device)
        query_t = idx[:, None]  # shape = (T,1)
        key_t = idx[None, :]    # shape = (1,T)

        mask = torch.zeros(T, T, dtype=torch.bool, device=device)

        if self.config.causal_temporal:
            # 不允许看未来：key > query
            mask = mask | (key_t > query_t)

        if window_size is not None:
            window_size = int(window_size)
            if window_size <= 0:
                raise ValueError(
                    f"temporal_window_size must be positive or None, got {window_size}."
                )

            if self.config.causal_temporal:
                # 不允许看太旧的历史：
                # query_t - key_t >= W 表示 key_t <= query_t - W
                mask = mask | ((query_t - key_t) >= window_size)
            else:
                # 非 causal 情况下退化为局部双向窗口。
                mask = mask | (torch.abs(query_t - key_t) >= window_size)

        return mask

    def forward(
        self,
        *,
        response_node_features: torch.Tensor | None = None,
        load_node_features: torch.Tensor | None = None,
        geometry_features: torch.Tensor | None = None,
        u: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        a: torch.Tensor | None = None,
        F: torch.Tensor | None = None,
        load_spectral_features: torch.Tensor | None = None,
    ) -> SpatiotemporalPhysicsEncoderOutput:
        """
        返回：
            theta: shape = (B,T,P)

        使用方式 1：
            直接传 response_node_features / load_node_features。

        使用方式 2：
            传 u,v,a,F 的 full-order flat state，由本模块 reshape。
        """
        if response_node_features is None and u is not None:
            if v is None or a is None:
                raise ValueError("When using flat state input, u/v/a must all be provided.")
            response_node_features = full_state_to_node_response_features(
                u=u,
                v=v,
                a=a,
                n_nodes=self.config.n_nodes,
                dof_per_node=self.config.dof_per_node,
            )

        if load_node_features is None and F is not None:
            load_node_features = full_force_to_node_load_features(
                F=F,
                n_nodes=self.config.n_nodes,
                dof_per_node=self.config.dof_per_node,
            )

        prefix: tuple[int, int] | None = None

        if self.config.use_response_branch:
            if response_node_features is None:
                raise ValueError("response branch is enabled but response_node_features is None.")
            if response_node_features.ndim != 4:
                raise ValueError(
                    "response_node_features must have shape (B,T,N,18), "
                    f"got {tuple(response_node_features.shape)}."
                )
            prefix = (int(response_node_features.shape[0]), int(response_node_features.shape[1]))

        if self.config.use_load_branch:
            if load_node_features is None:
                raise ValueError("load branch is enabled but load_node_features is None.")
            if load_node_features.ndim != 4:
                raise ValueError(
                    "load_node_features must have shape (B,T,N,6), "
                    f"got {tuple(load_node_features.shape)}."
                )

            load_prefix = (int(load_node_features.shape[0]), int(load_node_features.shape[1]))

            if prefix is None:
                prefix = load_prefix
            elif prefix != load_prefix:
                raise ValueError(
                    f"response/load prefix mismatch: response={prefix}, load={load_prefix}."
                )

        if prefix is None:
            # 只开 geometry branch 时，需要从 geometry_features 外部推断不了 B,T。
            # 第一阶段不建议只开 geometry branch 做时序参数预测。
            raise ValueError(
                "Cannot infer (B,T). Enable response/load branch or provide dynamic inputs."
            )

        B, T = prefix

        branch_embeddings: dict[str, torch.Tensor] = {}
        spatial_attention_weights: dict[str, torch.Tensor] = {}

        geometry_for_dynamic = None
        if self.config.condition_dynamic_branches_on_geometry:
            if geometry_features is None:
                raise ValueError(
                    "condition_dynamic_branches_on_geometry=True but geometry_features is None."
                )
            geometry_for_dynamic = geometry_features

        if self.config.use_response_branch:
            assert self.response_branch is not None
            h_response, attn_response = self.response_branch(
                response_node_features,
                geometry_features=geometry_for_dynamic,
            )
            branch_embeddings["response"] = h_response
            spatial_attention_weights["response"] = attn_response

        if self.config.use_load_branch:
            assert self.load_branch is not None
            h_load, attn_load = self.load_branch(
                load_node_features,
                geometry_features=geometry_for_dynamic,
            )

            if self.config.use_load_spectral_features:
                assert self.load_spectral_branch is not None
                if load_spectral_features is None:
                    if F is None:
                        raise ValueError(
                            "use_load_spectral_features=True but neither load_spectral_features nor F was provided."
                        )
                    window_size = self.config.load_spectral_window_size
                    if window_size is None:
                        window_size = self.config.temporal_window_size
                    load_spectral_features = compute_causal_load_spectral_features_from_force(
                        F=F,
                        n_nodes=self.config.n_nodes,
                        dof_per_node=self.config.dof_per_node,
                        window_size=window_size,
                        dt=0.01,
                        freq_min=float(self.config.load_spectral_freq_min),
                        freq_max=float(self.config.load_spectral_freq_max),
                        bands=str(self.config.load_spectral_bands),
                        observations=str(self.config.load_spectral_observations),
                        last_k=int(self.config.load_spectral_last_k),
                        active_rel_threshold=float(self.config.load_spectral_active_rel_threshold),
                        active_abs_threshold=float(self.config.load_spectral_active_abs_threshold),
                    )

                if load_spectral_features.ndim != 3:
                    raise ValueError(
                        "load_spectral_features must have shape (B,T,D_spec), "
                        f"got {tuple(load_spectral_features.shape)}."
                    )
                if tuple(load_spectral_features.shape[:2]) != (B, T):
                    raise ValueError(
                        f"load_spectral_features prefix mismatch: expected {(B,T)}, "
                        f"got {tuple(load_spectral_features.shape[:2])}."
                    )
                if int(load_spectral_features.shape[-1]) != int(self.config.load_spectral_feature_dim):
                    raise ValueError(
                        f"load_spectral_features dim mismatch: expected {self.config.load_spectral_feature_dim}, "
                        f"got {load_spectral_features.shape[-1]}."
                    )

                spec = load_spectral_features.to(device=h_load.device, dtype=h_load.dtype)
                if bool(self.config.load_spectral_normalize):
                    mean = self.load_spectral_mean.to(device=h_load.device, dtype=h_load.dtype).view(1, 1, -1)
                    std = self.load_spectral_std.to(device=h_load.device, dtype=h_load.dtype).view(1, 1, -1)
                    spec = (spec - mean) / torch.clamp(std, min=1.0e-6)
                spec = torch.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)
                h_spec = self.load_spectral_branch(spec)
                h_load = h_load + h_spec

            branch_embeddings["load"] = h_load
            spatial_attention_weights["load"] = attn_load

        if self.config.use_geometry_branch:
            if geometry_features is None:
                raise ValueError("geometry branch is enabled but geometry_features is None.")
            assert self.geometry_branch is not None
            h_geometry, attn_geometry = self.geometry_branch(
                geometry_features,
                prefix_shape=(B, T),
            )
            branch_embeddings["geometry"] = h_geometry
            spatial_attention_weights["geometry"] = attn_geometry

        fused, gate_weights = self.fusion(branch_embeddings)

        if fused.shape != (B, T, self.config.d_model):
            raise RuntimeError(
                f"fused embedding shape mismatch: expected {(B,T,self.config.d_model)}, "
                f"got {tuple(fused.shape)}."
            )

        temporal_input = self.pos_encoding(fused)

        if self.config.use_temporal_transformer:
            mask = self._make_causal_mask(
                T,
                device=temporal_input.device,
            )
            temporal_hidden = self.temporal_encoder(
                temporal_input,
                mask=mask,
            )
        else:
            temporal_hidden = self.temporal_encoder(temporal_input)

        head_out = self.head(temporal_hidden, return_dict=True)

        return SpatiotemporalPhysicsEncoderOutput(
            theta=head_out["theta"],
            raw_theta=head_out["raw_theta"],
            theta_dict=head_out["theta_dict"],
            temporal_hidden=temporal_hidden,
            fused_branch_embedding=fused,
            branch_embeddings=branch_embeddings,
            spatial_attention_weights=spatial_attention_weights,
            fusion_gate_weights=gate_weights,
            theta_aux=head_out.get("theta_aux"),
        )