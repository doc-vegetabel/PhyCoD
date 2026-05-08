from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn


@dataclass
class SpatialBranchEncoderConfig:
    """
    节点级空间注意力分支配置。

    当前第一阶段目标：
    - response branch: 输入每节点 [u, v, a]，维度 18；
    - load branch: 输入每节点 F，维度 6；
    - geometry branch: 输入每节点 blade geometry feature；
    - 每个分支输出统一 d_model 维 embedding；
    - 后续再接 temporal transformer 和 physical parameter head。
    """

    input_dim: int
    d_model: int = 64
    n_heads: int = 4
    mlp_hidden_dim: int = 128
    dropout: float = 0.0
    use_layernorm: bool = True


class NodeMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        hidden_dim: int,
        dropout: float = 0.0,
        use_layernorm: bool = True,
    ):
        super().__init__()

        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        ]

        if use_layernorm:
            layers.append(nn.LayerNorm(d_model))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatialAttentionPool(nn.Module):
    """
    使用 learnable query 对 N 个节点 token 做 attention pooling。

    输入：
        x: shape = (B_flat, N, d_model)

    输出：
        pooled: shape = (B_flat, d_model)
        attn_weights: shape = (B_flat, n_heads, N)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        use_layernorm: bool = True,
    ):
        super().__init__()

        self.query = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.query, mean=0.0, std=0.02)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"x must have shape (B_flat,N,D), got {tuple(x.shape)}.")

        B = int(x.shape[0])
        query = self.query.expand(B, -1, -1)

        pooled, weights = self.attn(
            query=query,
            key=x,
            value=x,
            need_weights=True,
            average_attn_weights=False,
        )

        # pooled: (B,1,D)
        pooled = self.norm(pooled[:, 0, :])

        # weights: (B,n_heads,1,N) -> (B,n_heads,N)
        weights = weights[:, :, 0, :]

        return pooled, weights


def _flatten_node_tensor(
    x: torch.Tensor,
    *,
    expected_last_dim: int,
    name: str,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """
    将 (..., N, C) 展平为 (B_flat, N, C)。

    返回：
        flat_x
        prefix_shape
    """
    if x.ndim < 3:
        raise ValueError(
            f"{name} must have shape (...,N,C), got {tuple(x.shape)}."
        )

    if x.shape[-1] != expected_last_dim:
        raise ValueError(
            f"{name} last dim mismatch: expected {expected_last_dim}, got {x.shape[-1]}."
        )

    prefix_shape = tuple(int(v) for v in x.shape[:-2])
    N = int(x.shape[-2])
    C = int(x.shape[-1])

    flat = x.reshape(-1, N, C)
    return flat, prefix_shape


def _restore_prefix(
    x: torch.Tensor,
    prefix_shape: tuple[int, ...],
) -> torch.Tensor:
    return x.reshape(*prefix_shape, x.shape[-1])


def _expand_geometry_to_flat(
    geometry: torch.Tensor,
    *,
    prefix_shape: tuple[int, ...],
    n_nodes: int,
) -> torch.Tensor:
    """
    支持 geometry 输入：
        (N,G)
        (...,N,G)

    输出：
        (B_flat,N,G)
    """
    if geometry.ndim == 2:
        if geometry.shape[0] != n_nodes:
            raise ValueError(
                f"geometry node count mismatch: expected {n_nodes}, got {geometry.shape[0]}."
            )

        expanded = geometry.reshape(
            *([1] * len(prefix_shape)),
            geometry.shape[0],
            geometry.shape[1],
        ).expand(*prefix_shape, geometry.shape[0], geometry.shape[1])

        return expanded.reshape(-1, geometry.shape[0], geometry.shape[1])

    if geometry.ndim >= 3:
        if tuple(geometry.shape[:-2]) != prefix_shape:
            raise ValueError(
                f"geometry prefix mismatch: expected {prefix_shape}, "
                f"got {tuple(geometry.shape[:-2])}."
            )

        if geometry.shape[-2] != n_nodes:
            raise ValueError(
                f"geometry node count mismatch: expected {n_nodes}, got {geometry.shape[-2]}."
            )

        return geometry.reshape(-1, geometry.shape[-2], geometry.shape[-1])

    raise ValueError(
        f"geometry must have shape (N,G) or (...,N,G), got {tuple(geometry.shape)}."
    )


class ResponseBranchEncoder(nn.Module):
    """
    响应分支。

    输入：
        response_node_features: shape = (..., N, 18)

    每个节点 18 维：
        [u_i(6), v_i(6), a_i(6)]

    可选 geometry_features:
        shape = (N,G) 或 (...,N,G)
        若提供，则通过 geometry_mlp 映射后加到 response token 上。
    """

    def __init__(
        self,
        *,
        config: SpatialBranchEncoderConfig,
        geometry_dim: int | None = None,
    ):
        super().__init__()

        if config.input_dim != 18:
            raise ValueError(
                f"ResponseBranchEncoder expects input_dim=18, got {config.input_dim}."
            )

        self.config = config
        self.node_mlp = NodeMLP(
            input_dim=config.input_dim,
            d_model=config.d_model,
            hidden_dim=config.mlp_hidden_dim,
            dropout=config.dropout,
            use_layernorm=config.use_layernorm,
        )

        self.geometry_mlp = None
        if geometry_dim is not None:
            self.geometry_mlp = NodeMLP(
                input_dim=int(geometry_dim),
                d_model=config.d_model,
                hidden_dim=config.mlp_hidden_dim,
                dropout=config.dropout,
                use_layernorm=config.use_layernorm,
            )

        self.pool = SpatialAttentionPool(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
            use_layernorm=config.use_layernorm,
        )

    def forward(
        self,
        response_node_features: torch.Tensor,
        geometry_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_flat, prefix = _flatten_node_tensor(
            response_node_features,
            expected_last_dim=18,
            name="response_node_features",
        )

        node_tokens = self.node_mlp(x_flat)

        if geometry_features is not None:
            if self.geometry_mlp is None:
                raise ValueError(
                    "geometry_features were provided, but this ResponseBranchEncoder "
                    "was created with geometry_dim=None."
                )

            g_flat = _expand_geometry_to_flat(
                geometry_features.to(device=x_flat.device, dtype=x_flat.dtype),
                prefix_shape=prefix,
                n_nodes=int(x_flat.shape[1]),
            )
            node_tokens = node_tokens + self.geometry_mlp(g_flat)

        pooled_flat, attn_flat = self.pool(node_tokens)

        pooled = _restore_prefix(pooled_flat, prefix)
        attn = attn_flat.reshape(*prefix, attn_flat.shape[-2], attn_flat.shape[-1])

        return pooled, attn


class LoadBranchEncoder(nn.Module):
    """
    载荷分支。

    输入：
        load_node_features: shape = (..., N, 6)

    每个节点 6 维：
        [Fx, Fy, Fz, Mx, My, Mz]

    可选 geometry_features:
        shape = (N,G) 或 (...,N,G)
    """

    def __init__(
        self,
        *,
        config: SpatialBranchEncoderConfig,
        geometry_dim: int | None = None,
    ):
        super().__init__()

        if config.input_dim != 6:
            raise ValueError(
                f"LoadBranchEncoder expects input_dim=6, got {config.input_dim}."
            )

        self.config = config
        self.node_mlp = NodeMLP(
            input_dim=config.input_dim,
            d_model=config.d_model,
            hidden_dim=config.mlp_hidden_dim,
            dropout=config.dropout,
            use_layernorm=config.use_layernorm,
        )

        self.geometry_mlp = None
        if geometry_dim is not None:
            self.geometry_mlp = NodeMLP(
                input_dim=int(geometry_dim),
                d_model=config.d_model,
                hidden_dim=config.mlp_hidden_dim,
                dropout=config.dropout,
                use_layernorm=config.use_layernorm,
            )

        self.pool = SpatialAttentionPool(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
            use_layernorm=config.use_layernorm,
        )

    def forward(
        self,
        load_node_features: torch.Tensor,
        geometry_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_flat, prefix = _flatten_node_tensor(
            load_node_features,
            expected_last_dim=6,
            name="load_node_features",
        )

        node_tokens = self.node_mlp(x_flat)

        if geometry_features is not None:
            if self.geometry_mlp is None:
                raise ValueError(
                    "geometry_features were provided, but this LoadBranchEncoder "
                    "was created with geometry_dim=None."
                )

            g_flat = _expand_geometry_to_flat(
                geometry_features.to(device=x_flat.device, dtype=x_flat.dtype),
                prefix_shape=prefix,
                n_nodes=int(x_flat.shape[1]),
            )
            node_tokens = node_tokens + self.geometry_mlp(g_flat)

        pooled_flat, attn_flat = self.pool(node_tokens)

        pooled = _restore_prefix(pooled_flat, prefix)
        attn = attn_flat.reshape(*prefix, attn_flat.shape[-2], attn_flat.shape[-1])

        return pooled, attn


class GeometryBranchEncoder(nn.Module):
    """
    几何分支。

    输入：
        geometry_features:
            shape = (N,G) 或 (...,N,G)

    如果输入是 (N,G)，需要提供 prefix_shape，例如 (B,T)，
    输出 shape 就是 (B,T,d_model)。
    """

    def __init__(
        self,
        *,
        config: SpatialBranchEncoderConfig,
    ):
        super().__init__()

        self.config = config
        self.node_mlp = NodeMLP(
            input_dim=config.input_dim,
            d_model=config.d_model,
            hidden_dim=config.mlp_hidden_dim,
            dropout=config.dropout,
            use_layernorm=config.use_layernorm,
        )

        self.pool = SpatialAttentionPool(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
            use_layernorm=config.use_layernorm,
        )

    def forward(
        self,
        geometry_features: torch.Tensor,
        *,
        prefix_shape: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if geometry_features.ndim == 2:
            if prefix_shape is None:
                prefix_shape = tuple()

            n_nodes = int(geometry_features.shape[0])
            expanded = geometry_features.reshape(
                *([1] * len(prefix_shape)),
                n_nodes,
                geometry_features.shape[1],
            ).expand(*prefix_shape, n_nodes, geometry_features.shape[1])

            x_flat = expanded.reshape(-1, n_nodes, geometry_features.shape[1])
            prefix = prefix_shape

        else:
            x_flat, prefix = _flatten_node_tensor(
                geometry_features,
                expected_last_dim=int(self.config.input_dim),
                name="geometry_features",
            )

        node_tokens = self.node_mlp(x_flat)
        pooled_flat, attn_flat = self.pool(node_tokens)

        pooled = _restore_prefix(pooled_flat, prefix)
        attn = attn_flat.reshape(*prefix, attn_flat.shape[-2], attn_flat.shape[-1])

        return pooled, attn


class GatedBranchFusion(nn.Module):
    """
    可开关分支融合模块。

    输入：
        branch_embeddings = {
            "response": tensor(...,D),
            "load": tensor(...,D),
            "geometry": tensor(...,D),
        }

    可以只传其中一个或多个分支。

    输出：
        fused: shape = (...,D)
        gate_weights: dict[str, tensor(...)]
    """

    def __init__(
        self,
        *,
        d_model: int,
        allowed_branches: tuple[str, ...] = ("response", "load", "geometry"),
    ):
        super().__init__()

        self.d_model = int(d_model)
        self.allowed_branches = tuple(allowed_branches)

        self.score_layers = nn.ModuleDict(
            {
                name: nn.Linear(self.d_model, 1)
                for name in self.allowed_branches
            }
        )

    def forward(
        self,
        branch_embeddings: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not branch_embeddings:
            raise ValueError("branch_embeddings must not be empty.")

        names = list(branch_embeddings.keys())

        for name in names:
            if name not in self.allowed_branches:
                raise KeyError(
                    f"Unknown branch {name!r}. Allowed branches: {self.allowed_branches}"
                )

        ref_shape = branch_embeddings[names[0]].shape

        if ref_shape[-1] != self.d_model:
            raise ValueError(
                f"Branch {names[0]!r} last dim mismatch: "
                f"expected {self.d_model}, got {ref_shape[-1]}."
            )

        for name in names[1:]:
            if branch_embeddings[name].shape != ref_shape:
                raise ValueError(
                    f"All branch embeddings must have same shape. "
                    f"{names[0]}={tuple(ref_shape)}, "
                    f"{name}={tuple(branch_embeddings[name].shape)}."
                )

        if len(names) == 1:
            only = names[0]
            gate = torch.ones(
                ref_shape[:-1],
                device=branch_embeddings[only].device,
                dtype=branch_embeddings[only].dtype,
            )
            return branch_embeddings[only], {only: gate}

        scores = []
        embeddings = []

        for name in names:
            h = branch_embeddings[name]
            scores.append(self.score_layers[name](h))
            embeddings.append(h)

        score_tensor = torch.cat(scores, dim=-1)
        gate_tensor = torch.softmax(score_tensor, dim=-1)

        stacked = torch.stack(embeddings, dim=-2)
        fused = torch.sum(gate_tensor.unsqueeze(-1) * stacked, dim=-2)

        gate_weights = {
            name: gate_tensor[..., j]
            for j, name in enumerate(names)
        }

        return fused, gate_weights