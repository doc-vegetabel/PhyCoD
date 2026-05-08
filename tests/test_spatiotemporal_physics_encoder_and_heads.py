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
from src.student.transformer.physical_parameter_registry import (  # noqa: E402
    build_physical_parameter_registry,
)
from src.student.transformer.spatiotemporal_physics_encoder import (  # noqa: E402
    SpatiotemporalPhysicsEncoder,
    SpatiotemporalPhysicsEncoderConfig,
    full_force_to_node_load_features,
    full_state_to_node_response_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test spatiotemporal physics encoder and bounded parameter heads."
    )

    parser.add_argument(
        "--blade-csv",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw" / "nrel5mw" / "blade_master.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "results" / "student" / "test_spatiotemporal_physics_encoder_and_heads"),
    )

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=6)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-spatial-heads", type=int, default=4)
    parser.add_argument("--n-temporal-heads", type=int, default=4)
    parser.add_argument("--n-temporal-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument(
        "--enabled-params",
        type=str,
        default="alpha_y",
        help="First-stage default should be alpha_y. Later can test alpha_y,alpha_xy.",
    )

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


def assert_finite(name: str, x: torch.Tensor) -> None:
    if not torch.all(torch.isfinite(x)).item():
        raise AssertionError(f"{name} contains non-finite values.")


def assert_theta_bounds(
    *,
    theta: torch.Tensor,
    registry,
    atol: float = 1.0e-6,
) -> dict[str, Any]:
    theta_dict = registry.split_theta(theta)
    stats: dict[str, Any] = {}

    for name, value in theta_dict.items():
        spec = registry.get_spec(name)
        max_abs = float(spec.max_abs)
        max_val = float(torch.max(value).detach().cpu())
        min_val = float(torch.min(value).detach().cpu())
        max_abs_observed = float(torch.max(torch.abs(value)).detach().cpu())

        if max_abs_observed > max_abs + atol:
            raise AssertionError(
                f"Parameter {name} exceeds bound {max_abs}: "
                f"observed max_abs={max_abs_observed}"
            )

        stats[name] = {
            "shape": list(value.shape),
            "min": min_val,
            "max": max_val,
            "max_abs": max_abs_observed,
            "bound": max_abs,
        }

    return stats


def check_gate_sum(gates: dict[str, torch.Tensor]) -> float:
    if len(gates) == 1:
        only = next(iter(gates.values()))
        err = torch.max(torch.abs(only - 1.0))
        return float(err.detach().cpu())

    stack = torch.stack([g for g in gates.values()], dim=-1)
    err = torch.max(torch.abs(torch.sum(stack, dim=-1) - 1.0))
    return float(err.detach().cpu())


def run_one_combo(
    *,
    combo_name: str,
    use_response: bool,
    use_load: bool,
    use_geometry: bool,
    condition_geometry: bool,
    registry,
    geometry: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    F: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, Any]:
    B = int(args.batch_size)
    T = int(args.seq_len)
    D_MODEL = int(args.d_model)

    model = SpatiotemporalPhysicsEncoder(
        geometry_dim=int(geometry.shape[-1]),
        registry=registry,
        config=SpatiotemporalPhysicsEncoderConfig(
            d_model=D_MODEL,
            n_spatial_heads=int(args.n_spatial_heads),
            n_temporal_heads=int(args.n_temporal_heads),
            n_temporal_layers=int(args.n_temporal_layers),
            temporal_ff_dim=2 * D_MODEL,
            spatial_mlp_hidden_dim=2 * D_MODEL,
            dropout=0.0,
            use_response_branch=bool(use_response),
            use_load_branch=bool(use_load),
            use_geometry_branch=bool(use_geometry),
            condition_dynamic_branches_on_geometry=bool(condition_geometry),
            causal_temporal=True,
            use_temporal_transformer=True,
        ),
    )

    out = model(
        u=u,
        v=v,
        a=a,
        F=F,
        geometry_features=geometry,
    )

    expected_theta_shape = (B, T, registry.total_dim)
    expected_hidden_shape = (B, T, D_MODEL)

    if tuple(out.theta.shape) != expected_theta_shape:
        raise AssertionError(
            f"{combo_name}: theta shape mismatch: "
            f"expected {expected_theta_shape}, got {tuple(out.theta.shape)}"
        )

    if tuple(out.temporal_hidden.shape) != expected_hidden_shape:
        raise AssertionError(
            f"{combo_name}: temporal_hidden shape mismatch: "
            f"expected {expected_hidden_shape}, got {tuple(out.temporal_hidden.shape)}"
        )

    assert_finite(f"{combo_name}.theta", out.theta)
    assert_finite(f"{combo_name}.raw_theta", out.raw_theta)
    assert_finite(f"{combo_name}.temporal_hidden", out.temporal_hidden)

    theta_stats = assert_theta_bounds(
        theta=out.theta,
        registry=registry,
    )

    expected_branches = []
    if use_response:
        expected_branches.append("response")
    if use_load:
        expected_branches.append("load")
    if use_geometry:
        expected_branches.append("geometry")

    if set(out.branch_embeddings.keys()) != set(expected_branches):
        raise AssertionError(
            f"{combo_name}: branch key mismatch: "
            f"expected {expected_branches}, got {list(out.branch_embeddings.keys())}"
        )

    gate_err = check_gate_sum(out.fusion_gate_weights)
    if gate_err > 1.0e-6:
        raise AssertionError(f"{combo_name}: fusion gates do not sum to 1, err={gate_err}")

    # 梯度测试：确认 theta 可反传到网络参数。
    loss = torch.mean(out.theta ** 2) + 1.0e-4 * torch.mean(out.temporal_hidden ** 2)
    loss.backward()

    grad_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm_sq += float(torch.sum(p.grad.detach() ** 2).cpu())

    grad_norm = grad_norm_sq ** 0.5

    if grad_norm <= 0.0:
        raise AssertionError(f"{combo_name}: gradient norm is zero.")

    return {
        "combo_name": combo_name,
        "theta_shape": list(out.theta.shape),
        "temporal_hidden_shape": list(out.temporal_hidden.shape),
        "branches": list(out.branch_embeddings.keys()),
        "theta_stats": theta_stats,
        "gate_sum_max_err": gate_err,
        "grad_norm": grad_norm,
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
    print("[Test] SpatiotemporalPhysicsEncoder + PhysicalParameterHead")
    print("=" * 100)
    print(f"  blade_csv = {blade_csv}")
    print(f"  enabled_params = {args.enabled_params}")
    print(f"  batch_size = {args.batch_size}")
    print(f"  seq_len = {args.seq_len}")
    print(f"  d_model = {args.d_model}")

    # ------------------------------------------------------------------
    # 1. Build geometry features
    # ------------------------------------------------------------------
    print()
    print("[1/4] Build geometry features")

    geo_bundle = build_blade_geometry_features(
        BladeGeometryFeatureConfig(
            blade_csv=str(blade_csv),
            twist_column="initial_twist_deg",
            phi_sign=-1.0,
            exclude_root_station=True,
        )
    )

    geometry = torch.tensor(geo_bundle.features, dtype=torch.float32)

    N = int(geo_bundle.n_nodes)
    G = int(geo_bundle.feature_dim)
    B = int(args.batch_size)
    T = int(args.seq_len)
    D = N * 6

    print(f"  n_nodes = {N}")
    print(f"  geometry_dim = {G}")
    print(f"  full_dofs = {D}")

    if N != 48:
        raise AssertionError(f"Expected 48 free nodes, got {N}.")

    # ------------------------------------------------------------------
    # 2. Build fake full-order states
    # ------------------------------------------------------------------
    print()
    print("[2/4] Build random full-order states")

    u = 1.0e-2 * torch.randn(B, T, D, dtype=torch.float32)
    v = 1.0e-1 * torch.randn(B, T, D, dtype=torch.float32)
    a = 1.0e0 * torch.randn(B, T, D, dtype=torch.float32)
    F = 1.0e3 * torch.randn(B, T, D, dtype=torch.float32)

    response_node = full_state_to_node_response_features(
        u=u,
        v=v,
        a=a,
        n_nodes=N,
        dof_per_node=6,
    )
    load_node = full_force_to_node_load_features(
        F=F,
        n_nodes=N,
        dof_per_node=6,
    )

    if tuple(response_node.shape) != (B, T, N, 18):
        raise AssertionError(f"response_node shape wrong: {tuple(response_node.shape)}")

    if tuple(load_node.shape) != (B, T, N, 6):
        raise AssertionError(f"load_node shape wrong: {tuple(load_node.shape)}")

    # ------------------------------------------------------------------
    # 3. Build registry
    # ------------------------------------------------------------------
    print()
    print("[3/4] Build registry")

    registry = build_physical_parameter_registry(
        enabled_params=str(args.enabled_params),
    )

    print(f"  enabled = {registry.names}")
    print(f"  theta_dim = {registry.total_dim}")

    # ------------------------------------------------------------------
    # 4. Test branch combinations
    # ------------------------------------------------------------------
    print()
    print("[4/4] Test branch combinations")

    combos = [
        {
            "name": "response_only",
            "use_response": True,
            "use_load": False,
            "use_geometry": False,
            "condition_geometry": False,
        },
        {
            "name": "response_load",
            "use_response": True,
            "use_load": True,
            "use_geometry": False,
            "condition_geometry": False,
        },
        {
            "name": "response_load_geometry_branch",
            "use_response": True,
            "use_load": True,
            "use_geometry": True,
            "condition_geometry": False,
        },
        {
            "name": "response_load_geometry_conditioned",
            "use_response": True,
            "use_load": True,
            "use_geometry": True,
            "condition_geometry": True,
        },
    ]

    combo_reports = {}

    for combo in combos:
        report = run_one_combo(
            combo_name=combo["name"],
            use_response=combo["use_response"],
            use_load=combo["use_load"],
            use_geometry=combo["use_geometry"],
            condition_geometry=combo["condition_geometry"],
            registry=registry,
            geometry=geometry,
            u=u,
            v=v,
            a=a,
            F=F,
            args=args,
        )
        combo_reports[combo["name"]] = report

        print(
            f"  {combo['name']:34s} "
            f"theta_shape={tuple(report['theta_shape'])}, "
            f"grad_norm={report['grad_norm']:.6e}"
        )

    final_report = {
        "passed": True,
        "enabled_params": str(args.enabled_params),
        "registry": registry.summary(),
        "geometry_summary": geo_bundle.summary(),
        "config": {
            "batch_size": B,
            "seq_len": T,
            "n_nodes": N,
            "full_dofs": D,
            "geometry_dim": G,
            "d_model": int(args.d_model),
            "n_spatial_heads": int(args.n_spatial_heads),
            "n_temporal_heads": int(args.n_temporal_heads),
            "n_temporal_layers": int(args.n_temporal_layers),
        },
        "combo_reports": combo_reports,
    }

    report_path = output_dir / "spatiotemporal_physics_encoder_and_heads_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(final_report), f, indent=2, ensure_ascii=False)

    print()
    print("[Summary]")
    print(f"  report = {report_path}")
    print()
    print("✅ PASS: SpatiotemporalPhysicsEncoder outputs bounded physical parameters with switchable branches.")


if __name__ == "__main__":
    main()