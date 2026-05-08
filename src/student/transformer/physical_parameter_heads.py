from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from src.student.transformer.physical_parameter_registry import (
    PhysicalParameterRegistry,
    build_physical_parameter_registry,
)


@dataclass
class PhysicalParameterHeadConfig:
    """
    物理参数输出头配置。

    输入：
        hidden state h_t, shape = (..., d_model)

    输出：
        theta_t, shape = (..., total_dim)

    输出不会直接作为无界参数使用，而是经过：
        theta = max_abs * tanh(raw)

    关键初始化约束：
        初始 raw_theta 应接近 0，使 theta_t 接近 0。
        这样 Transformer 初始 rollout 退化为 static corrected student。
    """

    d_model: int = 64
    hidden_dim: int = 128
    dropout: float = 0.0
    use_layernorm: bool = True

    # 新增：让最后一层 theta head 初始输出接近 0。
    zero_init_last_bias: bool = True
    init_last_weight_std: float = 1.0e-5

    # 新增：可选整体缩放。默认 1.0 表示仍使用 registry 中的 max_abs。
    # 如果后续还不稳定，可以临时设为 0.2 或 0.1。
    theta_output_scale: float = 1.0



@dataclass
class PhaseGatedPhysicalParameterHeadConfig:
    """
    slow + phase-gated fast decomposition 物理参数输出头配置。

    对外仍输出 registry 顺序下的 theta_total，保持 DynamicPhysicalCoreTorch 接口不变：
        theta_total = theta_slow + g_phase * theta_fast

    其中：
        theta_slow: 更适合慢变刚度/耦合漂移；
        theta_fast: 快变残差候选；
        g_phase:   [0,1] 门控，控制 fast 分支何时启用。
    """

    d_model: int = 64
    hidden_dim: int = 128
    dropout: float = 0.0
    use_layernorm: bool = True

    zero_init_last_bias: bool = True
    init_last_weight_std: float = 1.0e-5
    theta_output_scale: float = 1.0

    slow_scale: float = 1.0
    fast_scale: float = 0.5
    gate_init_bias: float = -4.0
    total_clip_scale: float = 1.0


class PhaseGatedPhysicalParameterHead(nn.Module):
    """
    输出 phase-gated dynamic correction：
        theta_total = theta_slow + sigmoid(raw_gate) * theta_fast

    重要兼容性：
        - out["theta"] 仍是 shape (..., P) 的总物理参数；
        - out["theta_dict"] 仍按 registry 拆成 alpha_x / alpha_xy；
        - DynamicPhysicalCoreTorch 不需要知道 slow/fast/gate 的存在；
        - 额外可解释分量放在 out["theta_aux"] 中供正则、诊断和绘图使用。
    """

    def __init__(
        self,
        *,
        registry: PhysicalParameterRegistry | None = None,
        enabled_params: str | list[str] | tuple[str, ...] | None = None,
        config: PhaseGatedPhysicalParameterHeadConfig | None = None,
    ):
        super().__init__()

        self.config = config or PhaseGatedPhysicalParameterHeadConfig()
        self.registry = registry or build_physical_parameter_registry(
            enabled_params=enabled_params,
        )

        if self.config.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.config.d_model}.")
        if self.config.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.config.hidden_dim}.")
        if self.registry.total_dim <= 0:
            raise ValueError("registry.total_dim must be positive.")

        self.norm = (
            nn.LayerNorm(self.config.d_model)
            if self.config.use_layernorm
            else nn.Identity()
        )

        self.slow_net = self._make_head(self.registry.total_dim)
        self.fast_net = self._make_head(self.registry.total_dim)
        self.gate_net = self._make_head(1)

        self._init_last_layer(self.slow_net, bias_value=0.0)
        self._init_last_layer(self.fast_net, bias_value=0.0)
        self._init_last_layer(self.gate_net, bias_value=float(self.config.gate_init_bias))

        max_abs = torch.tensor(
            self.registry.max_abs_list(),
            dtype=torch.float32,
        )
        self.register_buffer("max_abs", max_abs)

    def _make_head(self, out_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self.config.d_model, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, int(out_dim)),
        )

    def _init_last_layer(self, net: nn.Sequential, *, bias_value: float) -> None:
        last = net[-1]
        if not isinstance(last, nn.Linear):
            raise TypeError(f"Expected last head layer to be nn.Linear, got {type(last)}.")

        nn.init.normal_(
            last.weight,
            mean=0.0,
            std=float(self.config.init_last_weight_std),
        )

        if bool(self.config.zero_init_last_bias):
            nn.init.constant_(last.bias, float(bias_value))

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        return_dict: bool = True,
    ) -> dict[str, Any]:
        if hidden.ndim < 2:
            raise ValueError(
                f"hidden must have shape (..., d_model), got {tuple(hidden.shape)}."
            )
        if hidden.shape[-1] != self.config.d_model:
            raise ValueError(
                f"hidden last dim mismatch: expected {self.config.d_model}, "
                f"got {hidden.shape[-1]}."
            )

        hidden = self.norm(hidden)

        raw_slow = self.slow_net(hidden)
        raw_fast = self.fast_net(hidden)
        raw_gate = self.gate_net(hidden)

        max_abs = self.max_abs.to(device=raw_slow.device, dtype=raw_slow.dtype)
        limit = max_abs * float(self.config.theta_output_scale) * float(self.config.total_clip_scale)

        theta_slow = (
            torch.tanh(raw_slow)
            * max_abs
            * float(self.config.theta_output_scale)
            * float(self.config.slow_scale)
        )
        theta_fast = (
            torch.tanh(raw_fast)
            * max_abs
            * float(self.config.theta_output_scale)
            * float(self.config.fast_scale)
        )
        g_phase = torch.sigmoid(raw_gate)
        theta_gated_fast = g_phase * theta_fast
        theta_pre_clip = theta_slow + theta_gated_fast

        # 保持总 theta 不超过旧 registry 对应的安全物理范围，避免 slow+fast 叠加后
        # 给 K_eff 引入比旧模型更大的无界扰动。
        theta = torch.maximum(torch.minimum(theta_pre_clip, limit), -limit)

        # raw_theta 保持旧接口 shape (..., P)。它不再是一一对应的线性 raw head，
        # 但可作为总参数的近似 raw 记录；可解释分量见 theta_aux。
        raw_theta = raw_slow + g_phase * raw_fast

        theta_aux: dict[str, torch.Tensor] = {
            "theta_slow": theta_slow,
            "theta_fast": theta_fast,
            "theta_gated_fast": theta_gated_fast,
            "theta_pre_clip": theta_pre_clip,
            "g_phase": g_phase,
            "raw_theta_slow": raw_slow,
            "raw_theta_fast": raw_fast,
            "raw_phase_gate": raw_gate,
        }

        out: dict[str, Any] = {
            "theta": theta,
            "raw_theta": raw_theta,
            "theta_aux": theta_aux,
        }

        if return_dict:
            out["theta_dict"] = self.registry.split_theta(theta)

        return out

class BoundedPhysicalParameterHead(nn.Module):
    """
    将 encoder hidden state 映射为有界物理参数 theta。

    当前第一阶段建议：
        enabled_params = "alpha_y"

    后续可扩展：
        enabled_params = "alpha_y,alpha_xy"

    注意：
        这个 head 不关心物理参数如何进入 MCK；
        它只负责按照 registry 输出有界 theta。
    """

    def __init__(
        self,
        *,
        registry: PhysicalParameterRegistry | None = None,
        enabled_params: str | list[str] | tuple[str, ...] | None = None,
        config: PhysicalParameterHeadConfig | None = None,
    ):
        super().__init__()

        self.config = config or PhysicalParameterHeadConfig()
        self.registry = registry or build_physical_parameter_registry(
            enabled_params=enabled_params,
        )

        if self.config.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.config.d_model}.")

        if self.config.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.config.hidden_dim}.")

        if self.registry.total_dim <= 0:
            raise ValueError("registry.total_dim must be positive.")

        layers: list[nn.Module] = [
            nn.Linear(self.config.d_model, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.registry.total_dim),
        ]

        self.net = nn.Sequential(*layers)

        # ------------------------------------------------------------
        # Critical initialization:
        #   初始 theta 必须接近 0，使初始 K_eff ≈ K0。
        #   不要把最后一层 weight 完全置 0，否则 trunk 初期几乎收不到梯度。
        #   使用极小随机权重 + 0 bias，更适合当前动态系统训练。
        # ------------------------------------------------------------
        last = self.net[-1]
        if not isinstance(last, nn.Linear):
            raise TypeError(f"Expected last head layer to be nn.Linear, got {type(last)}.")

        nn.init.normal_(
            last.weight,
            mean=0.0,
            std=float(self.config.init_last_weight_std),
        )

        if bool(self.config.zero_init_last_bias):
            nn.init.zeros_(last.bias)

        max_abs = torch.tensor(
            self.registry.max_abs_list(),
            dtype=torch.float32,
        )
        self.register_buffer("max_abs", max_abs)

        self.norm = (
            nn.LayerNorm(self.config.d_model)
            if self.config.use_layernorm
            else nn.Identity()
        )

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        return_dict: bool = True,
    ) -> dict[str, Any]:
        """
        hidden:
            shape = (..., d_model)

        return:
            {
                "theta": bounded theta, shape = (..., P),
                "raw_theta": raw output before tanh, shape = (..., P),
                "theta_dict": optional split dict
            }
        """
        if hidden.ndim < 2:
            raise ValueError(
                f"hidden must have shape (..., d_model), got {tuple(hidden.shape)}."
            )

        if hidden.shape[-1] != self.config.d_model:
            raise ValueError(
                f"hidden last dim mismatch: expected {self.config.d_model}, "
                f"got {hidden.shape[-1]}."
            )

        hidden = self.norm(hidden)
        raw_theta = self.net(hidden)

        max_abs = self.max_abs.to(device=raw_theta.device, dtype=raw_theta.dtype)
        theta = torch.tanh(raw_theta) * max_abs * float(self.config.theta_output_scale)
        out: dict[str, Any] = {
            "theta": theta,
            "raw_theta": raw_theta,
        }

        if return_dict:
            out["theta_dict"] = self.registry.split_theta(theta)

        return out