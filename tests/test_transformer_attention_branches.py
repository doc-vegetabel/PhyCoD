from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from src.student.transformer.blade_geometry_features import (  # noqa: E402
    BladeGeometryFeatureConfig,
    build_blade_geometry_features,
)
from src.student.transformer.spatial_attention_encoders import (  # noqa: E402
    GatedBranchFusion,
    GeometryBranchEncoder,
    LoadBranchEncoder,
    ResponseBranchEncoder,
    SpatialBranchEncoderConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test switchable response/load/geometry attention branches."
    )

    parser.add_argument(
        "--blade-csv",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw" / "nrel5mw" / "blade_master.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "results" / "student" / "test_transformer_attention_branches"),
    )

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=5)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--mlp-hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)

    return parser.parse_args()


def make_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]

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


def check_finite_tensor(x: torch.Tensor, name: str) -> None:
    if not torch.all(torch.isfinite(x)).item():
        raise AssertionError(f"{name} contains non-finite values.")


def check_attn_weights(
    weights: torch.Tensor,
    *,
    name: str,
    expected_prefix: tuple[int, ...],
    expected_heads: int,
    expected_nodes: int,
    atol: float = 1.0e-5,
) -> dict[str, Any]:
    """
    weights shape = (*prefix, n_heads, N)
    """
    expected_shape = (*expected_prefix, expected_heads, expected_nodes)

    if tuple(weights.shape) != expected_shape:
        raise AssertionError(
            f"{name} attention weight shape mismatch: "
            f"expected {expected_shape}, got {tuple(weights.shape)}."
        )

    check_finite_tensor(weights, f"{name}.attn_weights")

    sums = torch.sum(weights, dim=-1)
    max_sum_err = float(torch.max(torch.abs(sums - 1.0)).detach().cpu())

    if max_sum_err > atol:
        raise AssertionError(
            f"{name} attention weights do not sum to 1 over nodes, "
            f"max_sum_err={max_sum_err:.6e}"
        )

    return {
        "shape": list(weights.shape),
        "max_sum_err": max_sum_err,
        "min": float(torch.min(weights).detach().cpu()),
        "max": float(torch.max(weights).detach().cpu()),
        "mean": float(torch.mean(weights).detach().cpu()),
    }


def main() -> None:
    args = parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    blade_csv = Path(args.blade_csv).resolve()
    if not blade_csv.exists():
        raise FileNotFoundError(f"blade_csv not found: {blade_csv}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 100)
    print("[Test] Transformer attention branches")
    print("=" * 100)
    print(f"  blade_csv = {blade_csv}")
    print(f"  batch_size = {args.batch_size}")
    print(f"  seq_len = {args.seq_len}")
    print(f"  d_model = {args.d_model}")
    print(f"  n_heads = {args.n_heads}")

    # ------------------------------------------------------------------
    # 1. Build blade geometry features
    # ------------------------------------------------------------------
    print()
    print("[1/4] Build blade geometry features")

    geo_bundle = build_blade_geometry_features(
        BladeGeometryFeatureConfig(
            blade_csv=str(blade_csv),
            twist_column="initial_twist_deg",
            phi_sign=-1.0,
            exclude_root_station=True,
        )
    )

    geometry_np = geo_bundle.features.astype(np.float32)

    N = int(geometry_np.shape[0])
    G = int(geometry_np.shape[1])
    B = int(args.batch_size)
    T = int(args.seq_len)

    if N != 48:
        raise AssertionError(f"Expected 48 free nodes, got N={N}.")

    geometry = torch.tensor(geometry_np, dtype=torch.float32)

    print(f"  n_nodes = {N}")
    print(f"  geometry_dim = {G}")
    print(f"  feature_names = {geo_bundle.feature_names}")

    # ------------------------------------------------------------------
    # 2. Build fake response/load node features
    # ------------------------------------------------------------------
    print()
    print("[2/4] Build random response/load features")

    response = torch.randn(B, T, N, 18, dtype=torch.float32)
    load = torch.randn(B, T, N, 6, dtype=torch.float32)

    # 模拟真实力尺度可能比响应大，测试 MLP 不应因尺度直接报错。
    load = load * 1.0e3

    prefix = (B, T)

    # ------------------------------------------------------------------
    # 3. Build branches
    # ------------------------------------------------------------------
    print()
    print("[3/4] Build branch encoders")

    response_branch = ResponseBranchEncoder(
        config=SpatialBranchEncoderConfig(
            input_dim=18,
            d_model=int(args.d_model),
            n_heads=int(args.n_heads),
            mlp_hidden_dim=int(args.mlp_hidden_dim),
            dropout=float(args.dropout),
        ),
        geometry_dim=G,
    )

    load_branch = LoadBranchEncoder(
        config=SpatialBranchEncoderConfig(
            input_dim=6,
            d_model=int(args.d_model),
            n_heads=int(args.n_heads),
            mlp_hidden_dim=int(args.mlp_hidden_dim),
            dropout=float(args.dropout),
        ),
        geometry_dim=G,
    )

    geometry_branch = GeometryBranchEncoder(
        config=SpatialBranchEncoderConfig(
            input_dim=G,
            d_model=int(args.d_model),
            n_heads=int(args.n_heads),
            mlp_hidden_dim=int(args.mlp_hidden_dim),
            dropout=float(args.dropout),
        )
    )

    fusion = GatedBranchFusion(
        d_model=int(args.d_model),
        allowed_branches=("response", "load", "geometry"),
    )

    # ------------------------------------------------------------------
    # 4. Forward all branch combinations
    # ------------------------------------------------------------------
    print()
    print("[4/4] Forward branch combinations")

    h_response, attn_response = response_branch(response, geometry_features=geometry)
    h_load, attn_load = load_branch(load, geometry_features=geometry)
    h_geometry, attn_geometry = geometry_branch(geometry, prefix_shape=prefix)

    expected_embedding_shape = (B, T, int(args.d_model))

    for name, h in {
        "response": h_response,
        "load": h_load,
        "geometry": h_geometry,
    }.items():
        if tuple(h.shape) != expected_embedding_shape:
            raise AssertionError(
                f"{name} embedding shape mismatch: "
                f"expected {expected_embedding_shape}, got {tuple(h.shape)}."
            )
        check_finite_tensor(h, f"{name}.embedding")

    attn_stats = {
        "response": check_attn_weights(
            attn_response,
            name="response",
            expected_prefix=prefix,
            expected_heads=int(args.n_heads),
            expected_nodes=N,
        ),
        "load": check_attn_weights(
            attn_load,
            name="load",
            expected_prefix=prefix,
            expected_heads=int(args.n_heads),
            expected_nodes=N,
        ),
        "geometry": check_attn_weights(
            attn_geometry,
            name="geometry",
            expected_prefix=prefix,
            expected_heads=int(args.n_heads),
            expected_nodes=N,
        ),
    }

    branch_outputs = {
        "response": h_response,
        "load": h_load,
        "geometry": h_geometry,
    }

    combos = [
        ("response",),
        ("load",),
        ("geometry",),
        ("response", "load"),
        ("response", "geometry"),
        ("load", "geometry"),
        ("response", "load", "geometry"),
    ]

    fusion_stats: dict[str, Any] = {}

    for combo in combos:
        combo_dict = {name: branch_outputs[name] for name in combo}
        fused, gates = fusion(combo_dict)

        combo_name = "+".join(combo)

        if tuple(fused.shape) != expected_embedding_shape:
            raise AssertionError(
                f"fused embedding shape mismatch for combo={combo_name}: "
                f"expected {expected_embedding_shape}, got {tuple(fused.shape)}."
            )

        check_finite_tensor(fused, f"fused.{combo_name}")

        gate_sum = None
        if len(gates) > 1:
            gate_tensor = torch.stack([gates[name] for name in combo], dim=-1)
            gate_sum = float(
                torch.max(torch.abs(torch.sum(gate_tensor, dim=-1) - 1.0)).detach().cpu()
            )

            if gate_sum > 1.0e-6:
                raise AssertionError(
                    f"fusion gates do not sum to 1 for combo={combo_name}, "
                    f"max_err={gate_sum:.6e}"
                )
        else:
            only_gate = next(iter(gates.values()))
            gate_sum = float(torch.max(torch.abs(only_gate - 1.0)).detach().cpu())

            if gate_sum > 1.0e-6:
                raise AssertionError(
                    f"single-branch fusion gate should be 1 for combo={combo_name}, "
                    f"max_err={gate_sum:.6e}"
                )

        fusion_stats[combo_name] = {
            "fused_shape": list(fused.shape),
            "gate_names": list(gates.keys()),
            "gate_sum_max_err": gate_sum,
            "fused_norm": float(torch.linalg.norm(fused).detach().cpu()),
        }

        print(f"  combo={combo_name:24s} -> fused shape={tuple(fused.shape)}")

    report = {
        "passed": True,
        "geometry_summary": geo_bundle.summary(),
        "config": {
            "batch_size": B,
            "seq_len": T,
            "n_nodes": N,
            "geometry_dim": G,
            "d_model": int(args.d_model),
            "n_heads": int(args.n_heads),
            "mlp_hidden_dim": int(args.mlp_hidden_dim),
            "dropout": float(args.dropout),
        },
        "embedding_shapes": {
            "response": list(h_response.shape),
            "load": list(h_load.shape),
            "geometry": list(h_geometry.shape),
        },
        "attn_stats": attn_stats,
        "fusion_stats": fusion_stats,
    }

    report_path = output_dir / "transformer_attention_branches_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(report), f, indent=2, ensure_ascii=False)

    print()
    print("[Summary]")
    print(f"  response embedding = {tuple(h_response.shape)}")
    print(f"  load embedding     = {tuple(h_load.shape)}")
    print(f"  geometry embedding = {tuple(h_geometry.shape)}")
    print(f"  report             = {report_path}")

    print()
    print("✅ PASS: response/load/geometry attention branches and gated fusion are ready.")


if __name__ == "__main__":
    main()