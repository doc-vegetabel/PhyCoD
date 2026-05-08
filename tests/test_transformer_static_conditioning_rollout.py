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
from src.student.transformer.spatiotemporal_physics_encoder import (  # noqa: E402
    SpatiotemporalPhysicsEncoder,
    SpatiotemporalPhysicsEncoderConfig,
)
from src.student.transformer.transformer_rollout_torch import (  # noqa: E402
    TransformerPhysicalRolloutTorch,
    TransformerRolloutConfig,
    response_mse_loss,
    theta_amplitude_loss,
    theta_smoothness_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test static-conditioning Transformer rollout through DynamicPhysicalCoreTorch."
    )

    parser.add_argument(
        "--blade-csv",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw" / "nrel5mw" / "blade_master.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "results" / "student" / "test_transformer_static_conditioning_rollout"),
    )

    parser.add_argument(
        "--enabled-params",
        type=str,
        default="alpha_y",
        help="First-stage default should be alpha_y.",
    )

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=6)
    parser.add_argument("--dt", type=float, default=0.01)

    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-spatial-heads", type=int, default=4)
    parser.add_argument("--n-temporal-heads", type=int, default=4)
    parser.add_argument("--n-temporal-layers", type=int, default=2)

    parser.add_argument("--use-response-branch", action="store_true", default=True)
    parser.add_argument("--no-response-branch", dest="use_response_branch", action="store_false")

    parser.add_argument("--use-load-branch", action="store_true", default=False)
    parser.add_argument("--no-load-branch", dest="use_load_branch", action="store_false")

    parser.add_argument("--use-geometry-branch", action="store_true", default=False)
    parser.add_argument("--no-geometry-branch", dest="use_geometry_branch", action="store_false")

    parser.add_argument("--condition-dynamic-branches-on-geometry", action="store_true", default=False)

    parser.add_argument("--kappa-y-static-scale", type=float, default=0.952)
    parser.add_argument("--kappa-y-scale-mode", type=str, default="y_bending", choices=["uy_only", "y_bending"])
    parser.add_argument("--xy-template-mode", type=str, default="root_to_tip", choices=["uniform", "root_to_tip", "tip_to_root"])
    parser.add_argument("--xy-delta-phi-deg", type=float, default=1.0)

    parser.add_argument("--force-scale", type=float, default=1.0e5)
    parser.add_argument("--static-state-scale", type=float, default=1.0e-3)
    parser.add_argument("--target-scale", type=float, default=0.95)

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


def grad_norm(module: torch.nn.Module) -> float:
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += float(torch.sum(p.grad.detach() ** 2).cpu())
    return total ** 0.5


def assert_finite(name: str, x: torch.Tensor) -> None:
    if not torch.all(torch.isfinite(x)).item():
        raise AssertionError(f"{name} contains non-finite values.")


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
    print("[Test] Transformer static-conditioning rollout")
    print("=" * 100)
    print(f"  blade_csv = {blade_csv}")
    print(f"  enabled_params = {args.enabled_params}")
    print(f"  branches: response={args.use_response_branch}, load={args.use_load_branch}, geometry={args.use_geometry_branch}")
    print(f"  condition_dynamic_branches_on_geometry = {args.condition_dynamic_branches_on_geometry}")

    # ------------------------------------------------------------------
    # 1. Registry
    # ------------------------------------------------------------------
    print()
    print("[1/6] Build physical parameter registry")

    registry = build_physical_parameter_registry(
        enabled_params=str(args.enabled_params),
    )

    print(f"  enabled = {registry.names}")
    print(f"  theta_dim = {registry.total_dim}")

    # ------------------------------------------------------------------
    # 2. Geometry
    # ------------------------------------------------------------------
    print()
    print("[2/6] Build geometry features")

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
    D = N * 6
    B = int(args.batch_size)
    T = int(args.seq_len)

    if N != 48 or D != 288:
        raise AssertionError(f"Expected N=48,D=288, got N={N}, D={D}")

    print(f"  n_nodes = {N}")
    print(f"  geometry_dim = {G}")
    print(f"  full_dofs = {D}")

    # ------------------------------------------------------------------
    # 3. Templates + DynamicPhysicalCore
    # ------------------------------------------------------------------
    print()
    print("[3/6] Build templates and dynamic physical core")

    template_cfg = PhysicalTemplateConfig(
        blade_csv=str(blade_csv),
        kappa_y_static_scale=float(args.kappa_y_static_scale),
        kappa_y_scale_mode=str(args.kappa_y_scale_mode),
        xy_template_mode=str(args.xy_template_mode),
        xy_delta_phi_deg=float(args.xy_delta_phi_deg),
        verbose=True,
    )

    template_bundle = build_dynamic_stiffness_templates(template_cfg)

    M0 = template_bundle.M0
    K0 = template_bundle.K0
    C0 = np.zeros_like(K0)

    core = DynamicPhysicalCoreTorch(
        M0=M0,
        K0=K0,
        C0=C0,
        stiffness_templates=template_bundle.stiffness_template_dict(),
        registry=registry,
        config=DynamicPhysicalCoreConfig(
            dt=float(args.dt),
            gamma=0.5,
            beta=0.25,
            dtype=torch.float64,
            linear_solve_mode="solve",
            symmetrize_k_eff=True,
        ),
    )

    # ------------------------------------------------------------------
    # 4. Encoder + Rollout model
    # ------------------------------------------------------------------
    print()
    print("[4/6] Build spatiotemporal encoder and rollout model")

    encoder = SpatiotemporalPhysicsEncoder(
        geometry_dim=G,
        registry=registry,
        config=SpatiotemporalPhysicsEncoderConfig(
            n_nodes=N,
            dof_per_node=6,
            d_model=int(args.d_model),
            n_spatial_heads=int(args.n_spatial_heads),
            n_temporal_heads=int(args.n_temporal_heads),
            n_temporal_layers=int(args.n_temporal_layers),
            temporal_ff_dim=2 * int(args.d_model),
            spatial_mlp_hidden_dim=2 * int(args.d_model),
            dropout=0.0,
            use_response_branch=bool(args.use_response_branch),
            use_load_branch=bool(args.use_load_branch),
            use_geometry_branch=bool(args.use_geometry_branch),
            condition_dynamic_branches_on_geometry=bool(args.condition_dynamic_branches_on_geometry),
            causal_temporal=True,
            use_temporal_transformer=True,
        ),
    )

    model = TransformerPhysicalRolloutTorch(
        encoder=encoder,
        physical_core=core,
        config=TransformerRolloutConfig(
            conditioning_mode="static",
            encoder_dtype=torch.float32,
            core_dtype=torch.float64,
            detach_static_conditioning=False,
            detach_rollout_state_each_step=False,
        ),
    )

    # ------------------------------------------------------------------
    # 5. Fake static conditioning data
    # ------------------------------------------------------------------
    print()
    print("[5/6] Build fake static trajectory and force")

    u_static = float(args.static_state_scale) * torch.randn(B, T, D, dtype=torch.float32)
    v_static = float(args.static_state_scale) * torch.randn(B, T, D, dtype=torch.float32)
    a_static = float(args.static_state_scale) * torch.randn(B, T, D, dtype=torch.float32)

    F = torch.zeros(B, T, D, dtype=torch.float32)

    # 给叶尖 uy 一个随时间变化的力，避免 rollout 全零。
    tip_uy_dof = D - 5
    time = torch.linspace(0.0, 1.0, T, dtype=torch.float32)
    force_series = float(args.force_scale) * torch.sin(2.0 * torch.pi * time)
    F[:, :, tip_uy_dof] = force_series.unsqueeze(0).expand(B, -1)

    # 初始状态统一置零，方便测试。
    u0 = torch.zeros(B, D, dtype=torch.float32)
    v0 = torch.zeros(B, D, dtype=torch.float32)
    a0 = torch.zeros(B, D, dtype=torch.float32)

    # ------------------------------------------------------------------
    # 6. Forward + loss + backward
    # ------------------------------------------------------------------
    print()
    print("[6/6] Forward rollout and backprop")

    out = model(
        u_static=u_static,
        v_static=v_static,
        a_static=a_static,
        F=F,
        geometry_features=geometry,
        u0=u0,
        v0=v0,
        a0=a0,
    )

    expected_u_shape = (B, T, D)
    expected_theta_shape = (B, T, registry.total_dim)

    if tuple(out.u_pred.shape) != expected_u_shape:
        raise AssertionError(
            f"u_pred shape mismatch: expected {expected_u_shape}, got {tuple(out.u_pred.shape)}"
        )

    if tuple(out.theta.shape) != expected_theta_shape:
        raise AssertionError(
            f"theta shape mismatch: expected {expected_theta_shape}, got {tuple(out.theta.shape)}"
        )

    assert_finite("u_pred", out.u_pred)
    assert_finite("v_pred", out.v_pred)
    assert_finite("a_pred", out.a_pred)
    assert_finite("theta", out.theta)

    # 构造一个简单 target。
    # 正式训练时这里会替换成 u_teacher。
    u_target = float(args.target_scale) * out.u_pred.detach()

    loss_resp = response_mse_loss(
        u_pred=out.u_pred,
        u_target=u_target,
        reduction="mean",
    )
    loss_amp = theta_amplitude_loss(out.theta.to(dtype=out.u_pred.dtype))
    loss_smooth = theta_smoothness_loss(out.theta.to(dtype=out.u_pred.dtype))

    loss = loss_resp + 1.0e-3 * loss_amp + 1.0e-3 * loss_smooth

    model.zero_grad(set_to_none=True)
    loss.backward()

    encoder_grad_norm = grad_norm(model.encoder)

    if encoder_grad_norm <= 0.0:
        raise AssertionError("Gradient did not flow back to encoder parameters.")

    theta_max_abs = float(torch.max(torch.abs(out.theta)).detach().cpu())
    u_pred_norm = float(torch.linalg.norm(out.u_pred).detach().cpu())
    v_pred_norm = float(torch.linalg.norm(out.v_pred).detach().cpu())
    a_pred_norm = float(torch.linalg.norm(out.a_pred).detach().cpu())

    report = {
        "passed": True,
        "registry": registry.summary(),
        "geometry_summary": geo_bundle.summary(),
        "template_summary": template_bundle.summary(),
        "config": {
            "B": B,
            "T": T,
            "D": D,
            "dt": float(args.dt),
            "enabled_params": str(args.enabled_params),
            "use_response_branch": bool(args.use_response_branch),
            "use_load_branch": bool(args.use_load_branch),
            "use_geometry_branch": bool(args.use_geometry_branch),
            "condition_dynamic_branches_on_geometry": bool(args.condition_dynamic_branches_on_geometry),
        },
        "shapes": {
            "u_pred": list(out.u_pred.shape),
            "v_pred": list(out.v_pred.shape),
            "a_pred": list(out.a_pred.shape),
            "theta": list(out.theta.shape),
        },
        "metrics": {
            "loss": float(loss.detach().cpu()),
            "loss_resp": float(loss_resp.detach().cpu()),
            "loss_amp": float(loss_amp.detach().cpu()),
            "loss_smooth": float(loss_smooth.detach().cpu()),
            "encoder_grad_norm": encoder_grad_norm,
            "theta_max_abs": theta_max_abs,
            "u_pred_norm": u_pred_norm,
            "v_pred_norm": v_pred_norm,
            "a_pred_norm": a_pred_norm,
        },
        "rollout_metadata": out.metadata,
    }

    report_path = output_dir / "transformer_static_conditioning_rollout_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(report), f, indent=2, ensure_ascii=False)

    print()
    print("[Summary]")
    print(f"  u_pred shape       = {tuple(out.u_pred.shape)}")
    print(f"  theta shape        = {tuple(out.theta.shape)}")
    print(f"  loss               = {float(loss.detach().cpu()):.6e}")
    print(f"  loss_resp          = {float(loss_resp.detach().cpu()):.6e}")
    print(f"  loss_amp           = {float(loss_amp.detach().cpu()):.6e}")
    print(f"  loss_smooth        = {float(loss_smooth.detach().cpu()):.6e}")
    print(f"  encoder_grad_norm  = {encoder_grad_norm:.6e}")
    print(f"  theta_max_abs      = {theta_max_abs:.6e}")
    print(f"  u_pred_norm        = {u_pred_norm:.6e}")
    print(f"  report             = {report_path}")

    print()
    print("✅ PASS: static-conditioning Transformer rollout can backpropagate through physical core.")


if __name__ == "__main__":
    main()