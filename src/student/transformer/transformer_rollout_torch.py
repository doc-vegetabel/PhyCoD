from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn

from src.student.transformer.dynamic_physical_core_torch import DynamicPhysicalCoreTorch
from src.student.transformer.spatiotemporal_physics_encoder import (
    SpatiotemporalPhysicsEncoder,
    SpatiotemporalPhysicsEncoderOutput,
)


ConditioningMode = Literal["static", "closed_loop"]


@dataclass
class TransformerRolloutConfig:
    """
    Transformer + DynamicPhysicalCore rollout 配置。

    当前第一阶段主线：
        conditioning_mode = "static"

    static conditioning:
        Transformer 输入来自 static corrected student 的 u_static/v_static/a_static。
        但预测响应 u_pred/v_pred/a_pred 由 DynamicPhysicalCoreTorch 闭环 rollout 得到。

    后续阶段：
        conditioning_mode = "closed_loop"
        Transformer 输入改为当前预测响应 u_pred/v_pred/a_pred。
    """

    conditioning_mode: ConditioningMode = "static"

    # encoder 通常用 float32；physical core 当前建议用 float64。
    encoder_dtype: torch.dtype = torch.float32
    core_dtype: torch.dtype = torch.float64

    detach_static_conditioning: bool = False

    # 当前测试阶段先不做 truncated BPTT。
    detach_rollout_state_each_step: bool = False
    profile_timing: bool = False
    profile_timing_sync_cuda: bool = False


@dataclass
class TransformerRolloutOutput:
    u_pred: torch.Tensor
    v_pred: torch.Tensor
    a_pred: torch.Tensor

    theta: torch.Tensor
    raw_theta: torch.Tensor
    theta_dict: dict[str, torch.Tensor]
    theta_aux: dict[str, torch.Tensor] | None

    encoder_output: SpatiotemporalPhysicsEncoderOutput

    metadata: dict[str, Any]


class TransformerPhysicalRolloutTorch(nn.Module):
    """
    将 SpatiotemporalPhysicsEncoder 与 DynamicPhysicalCoreTorch 串起来。

    当前 static-conditioning 流程：

        1. 输入 static corrected student trajectory:
              u_static, v_static, a_static, F

        2. Encoder 输出整条 theta 序列:
              theta_seq = encoder(u_static, v_static, a_static, F, geometry)

        3. DynamicPhysicalCoreTorch 从初始状态开始逐步 Newmark:
              theta_t -> K_eff_t -> u_{t+1}

        4. 返回 u_pred/v_pred/a_pred 和 theta_seq。

    重要约束：
        - Transformer 不输出位移修正；
        - theta 进入 K_eff；
        - 响应由 Newmark/MCK core 求解；
        - teacher 后续只用于 loss，不作为输入。
    """

    def __init__(
        self,
        *,
        encoder: SpatiotemporalPhysicsEncoder,
        physical_core: DynamicPhysicalCoreTorch,
        config: TransformerRolloutConfig | None = None,
    ):
        super().__init__()

        self.encoder = encoder
        self.physical_core = physical_core
        self.config = config or TransformerRolloutConfig()

        if self.config.conditioning_mode not in {"static", "closed_loop"}:
            raise ValueError(
                f"Unsupported conditioning_mode={self.config.conditioning_mode!r}."
            )

    @staticmethod
    def _sync_for_timing(enabled: bool, device: torch.device | None = None) -> None:
        if not enabled or not torch.cuda.is_available():
            return
        if device is not None and device.type != "cuda":
            return
        torch.cuda.synchronize(device)

    def _time_now(self, device: torch.device | None = None) -> float:
        self._sync_for_timing(bool(self.config.profile_timing_sync_cuda), device)
        return time.perf_counter()

    @staticmethod
    def _check_btd_tensor(
        x: torch.Tensor,
        *,
        name: str,
        expected_d: int | None = None,
    ) -> tuple[int, int, int]:
        if not torch.is_tensor(x):
            raise TypeError(f"{name} must be a torch.Tensor.")

        if x.ndim != 3:
            raise ValueError(f"{name} must have shape (B,T,D), got {tuple(x.shape)}.")

        B, T, D = (int(x.shape[0]), int(x.shape[1]), int(x.shape[2]))

        if T < 2:
            raise ValueError(f"{name} must have T>=2 for rollout, got T={T}.")

        if expected_d is not None and D != int(expected_d):
            raise ValueError(
                f"{name} last dim mismatch: expected {expected_d}, got {D}."
            )

        return B, T, D

    def _prepare_initial_state(
        self,
        *,
        u_static: torch.Tensor,
        v_static: torch.Tensor,
        a_static: torch.Tensor,
        u0: torch.Tensor | None,
        v0: torch.Tensor | None,
        a0: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        core_dtype = self.config.core_dtype
        device = self.physical_core.K0.device

        if u0 is None:
            u_init = u_static[:, 0, :]
        else:
            u_init = u0

        if v0 is None:
            v_init = v_static[:, 0, :]
        else:
            v_init = v0

        if a0 is None:
            a_init = a_static[:, 0, :]
        else:
            a_init = a0

        u_init = u_init.to(device=device, dtype=core_dtype)
        v_init = v_init.to(device=device, dtype=core_dtype)
        a_init = a_init.to(device=device, dtype=core_dtype)

        return u_init, v_init, a_init

    def _encode_static_conditioning(
        self,
        *,
        u_static: torch.Tensor,
        v_static: torch.Tensor,
        a_static: torch.Tensor,
        F: torch.Tensor,
        geometry_features: torch.Tensor | None,
        load_spectral_features: torch.Tensor | None = None,
    ) -> SpatiotemporalPhysicsEncoderOutput:
        encoder_dtype = self.config.encoder_dtype
        device = next(self.encoder.parameters()).device

        u_in = u_static
        v_in = v_static
        a_in = a_static
        F_in = F
        F_spec_in = load_spectral_features

        if self.config.detach_static_conditioning:
            u_in = u_in.detach()
            v_in = v_in.detach()
            a_in = a_in.detach()
            F_in = F_in.detach()
            if F_spec_in is not None:
                F_spec_in = F_spec_in.detach()

        u_in = u_in.to(device=device, dtype=encoder_dtype)
        v_in = v_in.to(device=device, dtype=encoder_dtype)
        a_in = a_in.to(device=device, dtype=encoder_dtype)
        F_in = F_in.to(device=device, dtype=encoder_dtype)
        if F_spec_in is not None:
            F_spec_in = F_spec_in.to(device=device, dtype=encoder_dtype)

        geometry_in = None
        if geometry_features is not None:
            geometry_in = geometry_features.to(device=device, dtype=encoder_dtype)

        return self.encoder(
            u=u_in,
            v=v_in,
            a=a_in,
            F=F_in,
            geometry_features=geometry_in,
            load_spectral_features=F_spec_in,
        )

    def forward_static_conditioning(
        self,
        *,
        u_static: torch.Tensor,
        v_static: torch.Tensor,
        a_static: torch.Tensor,
        F: torch.Tensor,
        geometry_features: torch.Tensor | None = None,
        load_spectral_features: torch.Tensor | None = None,
        u0: torch.Tensor | None = None,
        v0: torch.Tensor | None = None,
        a0: torch.Tensor | None = None,
    ) -> TransformerRolloutOutput:
        """
        static-conditioning rollout。

        输入：
            u_static, v_static, a_static, F:
                shape = (B,T,D)

        输出：
            u_pred, v_pred, a_pred:
                shape = (B,T,D)

            theta:
                shape = (B,T,P)

        使用方式：
            theta[:, t, :] 作用于从 t 到 t+1 的 Newmark step。
            因此 rollout 用到 t=0...T-2 的 theta。
        """
        B, T, D = self._check_btd_tensor(
            u_static,
            name="u_static",
            expected_d=self.physical_core.n_dofs,
        )

        for name, x in {
            "v_static": v_static,
            "a_static": a_static,
            "F": F,
        }.items():
            self._check_btd_tensor(
                x,
                name=name,
                expected_d=D,
            )

            if tuple(x.shape[:2]) != (B, T):
                raise ValueError(
                    f"{name} prefix shape mismatch: expected {(B,T)}, got {tuple(x.shape[:2])}."
                )

        if load_spectral_features is not None:
            self._check_btd_tensor(
                load_spectral_features,
                name="load_spectral_features",
                expected_d=None,
            )
            if tuple(load_spectral_features.shape[:2]) != (B, T):
                raise ValueError(
                    f"load_spectral_features prefix shape mismatch: expected {(B,T)}, "
                    f"got {tuple(load_spectral_features.shape[:2])}."
                )

        profile_timing = bool(self.config.profile_timing)
        timing: dict[str, float] = {}

        encoder_device = next(self.encoder.parameters()).device
        t0 = self._time_now(encoder_device) if profile_timing else 0.0
        encoder_out = self._encode_static_conditioning(
            u_static=u_static,
            v_static=v_static,
            a_static=a_static,
            F=F,
            geometry_features=geometry_features,
            load_spectral_features=load_spectral_features,
        )
        if profile_timing:
            timing["encoder_seconds"] = self._time_now(encoder_device) - t0

        theta_seq = encoder_out.theta

        if tuple(theta_seq.shape[:2]) != (B, T):
            raise RuntimeError(
                f"theta_seq prefix mismatch: expected {(B,T)}, got {tuple(theta_seq.shape[:2])}."
            )

        core_dtype = self.config.core_dtype
        core_device = self.physical_core.K0.device

        t0 = self._time_now(core_device) if profile_timing else 0.0
        F_core = F.to(device=core_device, dtype=core_dtype)
        theta_core = theta_seq.to(device=core_device, dtype=core_dtype)

        u_t, v_t, a_t = self._prepare_initial_state(
            u_static=u_static,
            v_static=v_static,
            a_static=a_static,
            u0=u0,
            v0=v0,
            a0=a0,
        )
        if profile_timing:
            timing["core_prepare_seconds"] = self._time_now(core_device) - t0

        u_list = [u_t]
        v_list = [v_t]
        a_list = [a_t]

        t0 = self._time_now(core_device) if profile_timing else 0.0
        newmark_detail_timing = {
            "newmark_assemble_seconds": 0.0,
            "newmark_rhs_seconds": 0.0,
            "newmark_solve_seconds": 0.0,
            "newmark_update_seconds": 0.0,
        }
        for t in range(T - 1):
            theta_t = theta_core[:, t, :]

            if self.config.detach_rollout_state_each_step:
                u_t = u_t.detach()
                v_t = v_t.detach()
                a_t = a_t.detach()

            if profile_timing:
                u_next, v_next, a_next, step_timing = self.physical_core.newmark_step_fast_timed(
                    u_t=u_t,
                    v_t=v_t,
                    a_t=a_t,
                    F_t1=F_core[:, t + 1, :],
                    theta_t=theta_t,
                    time_fn=lambda: self._time_now(core_device),
                )
                for key, value in step_timing.items():
                    newmark_detail_timing[key] += float(value)
            else:
                u_next, v_next, a_next = self.physical_core.newmark_step_fast(
                    u_t=u_t,
                    v_t=v_t,
                    a_t=a_t,
                    F_t1=F_core[:, t + 1, :],
                    theta_t=theta_t,
                )

            u_list.append(u_next)
            v_list.append(v_next)
            a_list.append(a_next)

            u_t, v_t, a_t = u_next, v_next, a_next
        if profile_timing:
            timing["newmark_loop_seconds"] = self._time_now(core_device) - t0
            timing.update(newmark_detail_timing)

        t0 = self._time_now(core_device) if profile_timing else 0.0
        u_pred = torch.stack(u_list, dim=1)
        v_pred = torch.stack(v_list, dim=1)
        a_pred = torch.stack(a_list, dim=1)
        if profile_timing:
            timing["state_stack_seconds"] = self._time_now(core_device) - t0

        return TransformerRolloutOutput(
            u_pred=u_pred,
            v_pred=v_pred,
            a_pred=a_pred,
            theta=theta_seq,
            raw_theta=encoder_out.raw_theta,
            theta_dict=encoder_out.theta_dict,
            theta_aux=encoder_out.theta_aux,
            encoder_output=encoder_out,
            metadata={
                "conditioning_mode": "static",
                "B": B,
                "T": T,
                "D": D,
                "theta_dim": int(theta_seq.shape[-1]),
                "theta_usage": "theta[:,t,:] is used for Newmark step t -> t+1",
                **timing,
            },
        )

    def forward(
        self,
        *,
        u_static: torch.Tensor,
        v_static: torch.Tensor,
        a_static: torch.Tensor,
        F: torch.Tensor,
        geometry_features: torch.Tensor | None = None,
        load_spectral_features: torch.Tensor | None = None,
        u0: torch.Tensor | None = None,
        v0: torch.Tensor | None = None,
        a0: torch.Tensor | None = None,
    ) -> TransformerRolloutOutput:
        if self.config.conditioning_mode == "static":
            return self.forward_static_conditioning(
                u_static=u_static,
                v_static=v_static,
                a_static=a_static,
                F=F,
                geometry_features=geometry_features,
                load_spectral_features=load_spectral_features,
                u0=u0,
                v0=v0,
                a0=a0,
            )

        raise NotImplementedError(
            "closed_loop conditioning will be added after static-conditioning training works."
        )


def response_mse_loss(
    *,
    u_pred: torch.Tensor,
    u_target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    简单响应 MSE。

    正式训练脚本后续会替换为：
        y/x normalized loss
        tip loss
        last-k loss
        x guard
        theta smoothness
    """
    if u_pred.shape != u_target.shape:
        raise ValueError(
            f"u_pred/u_target shape mismatch: "
            f"u_pred={tuple(u_pred.shape)}, u_target={tuple(u_target.shape)}"
        )

    diff = u_pred - u_target.to(device=u_pred.device, dtype=u_pred.dtype)
    loss = diff ** 2

    if reduction == "mean":
        return torch.mean(loss)

    if reduction == "sum":
        return torch.sum(loss)

    if reduction == "none":
        return loss

    raise ValueError(f"Unsupported reduction={reduction!r}.")


def theta_amplitude_loss(theta: torch.Tensor) -> torch.Tensor:
    return torch.mean(theta ** 2)


def theta_smoothness_loss(theta: torch.Tensor) -> torch.Tensor:
    """
    theta:
        shape = (B,T,P)
    """
    if theta.ndim != 3:
        raise ValueError(f"theta must have shape (B,T,P), got {tuple(theta.shape)}.")

    if theta.shape[1] < 2:
        return torch.zeros((), dtype=theta.dtype, device=theta.device)

    return torch.mean((theta[:, 1:, :] - theta[:, :-1, :]) ** 2)
