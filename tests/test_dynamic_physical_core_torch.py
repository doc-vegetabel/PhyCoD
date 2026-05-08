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


from src.student.transformer.alpha_y_control_points import (  # noqa: E402
    parse_alpha_y_cp_name,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sanity test for DynamicPhysicalCoreTorch."
    )

    parser.add_argument(
        "--blade-csv",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw" / "nrel5mw" / "blade_master.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "results" / "student" / "test_dynamic_physical_core_torch"),
    )

    parser.add_argument(
        "--enabled-params",
        type=str,
        default="alpha_y,alpha_xy",
    )

    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--damping-beta", type=float, default=0.0)

    parser.add_argument("--kappa-y-static-scale", type=float, default=0.952)
    parser.add_argument(
        "--kappa-y-scale-mode",
        type=str,
        default="y_bending",
        choices=["uy_only", "y_bending"],
    )
    parser.add_argument("--kappa-y-delta", type=float, default=1.0e-3)

    parser.add_argument(
        "--xy-template-mode",
        type=str,
        default="root_to_tip",
        choices=["uniform", "root_to_tip", "tip_to_root"],
    )
    parser.add_argument("--xy-delta-phi-deg", type=float, default=1.0)

    parser.add_argument("--alpha-y-test", type=float, default=0.01)
    parser.add_argument("--alpha-xy-test", type=float, default=0.01)

    # alpha_y_cpN 测试值。
    # 例如 alpha_y_cp3 时，会把 3 个控制点都设置成该值。
    parser.add_argument("--alpha-y-cp-test", type=float, default=0.01)

    parser.add_argument("--force-scale", type=float, default=1.0e5)

    parser.add_argument("--sym-atol", type=float, default=1.0e-8)
    parser.add_argument("--zero-atol", type=float, default=1.0e-10)

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


def rel_fro(a: torch.Tensor, b: torch.Tensor, eps: float = 1.0e-30) -> float:
    return float(
        torch.linalg.norm(a - b).detach().cpu()
        / (torch.linalg.norm(b).detach().cpu() + eps)
    )


def rel_vec(a: torch.Tensor, b: torch.Tensor, eps: float = 1.0e-30) -> float:
    return float(
        torch.linalg.norm(a - b).detach().cpu()
        / (torch.linalg.norm(b).detach().cpu() + eps)
    )


def _enabled_param_names(enabled_params: str) -> list[str]:
    return [s.strip() for s in str(enabled_params).split(",") if s.strip()]


def _has_alpha_y_cp(enabled_params: str) -> bool:
    for name in _enabled_param_names(enabled_params):
        if parse_alpha_y_cp_name(name) is not None:
            return True
    return False


def _fill_theta_for_enabled_params(
    *,
    theta: torch.Tensor,
    registry: Any,
    alpha_y_value: float,
    alpha_xy_value: float,
    alpha_y_cp_value: float,
) -> None:
    """
    根据 registry.slices 写入非零 theta。

    支持：
        alpha_y
        alpha_xy
        alpha_y_cp3 / alpha_y_cp4 / ...
    """
    for name in registry.names:
        if name not in registry.slices:
            raise KeyError(f"Registry missing slice for parameter: {name}")

        sl = registry.slices[name]

        if name == "alpha_y":
            theta[sl] = float(alpha_y_value)
            continue

        if name == "alpha_xy":
            theta[sl] = float(alpha_xy_value)
            continue

        n_cp = parse_alpha_y_cp_name(name)
        if n_cp is not None:
            theta[sl] = float(alpha_y_cp_value)
            continue


def main() -> None:
    args = parse_args()

    blade_csv = Path(args.blade_csv).resolve()
    if not blade_csv.exists():
        raise FileNotFoundError(f"blade_csv not found: {blade_csv}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.set_default_dtype(torch.float64)

    print()
    print("=" * 100)
    print("[Test] DynamicPhysicalCoreTorch")
    print("=" * 100)
    print(f"  blade_csv = {blade_csv}")
    print(f"  enabled_params = {args.enabled_params}")
    print(f"  dt = {args.dt}")
    print(f"  damping_beta = {args.damping_beta}")

    # ------------------------------------------------------------------
    # 1. Build registry + templates
    # ------------------------------------------------------------------
    print()
    print("[1/5] Build registry and physical templates")

    registry = build_physical_parameter_registry(
        enabled_params=args.enabled_params,
    )

    template_cfg = PhysicalTemplateConfig(
        blade_csv=str(blade_csv),
        kappa_y_static_scale=float(args.kappa_y_static_scale),
        kappa_y_scale_mode=str(args.kappa_y_scale_mode),
        kappa_y_delta=float(args.kappa_y_delta),
        xy_template_mode=str(args.xy_template_mode),
        xy_delta_phi_deg=float(args.xy_delta_phi_deg),

        # 关键修改：
        # 必须把命令行 enabled_params 传入 template config，
        # 否则 alpha_y_cp3 不会生成 K_y_cp_tpl。
        enabled_params=str(args.enabled_params),

        verbose=True,
    )

    bundle = build_dynamic_stiffness_templates(template_cfg)

    M0 = bundle.M0
    K0 = bundle.K0
    C0 = float(args.damping_beta) * K0

    stiffness_templates = bundle.stiffness_template_dict()

    print()
    print("[Template Dict]")
    print(f"  keys = {list(stiffness_templates.keys())}")

    if _has_alpha_y_cp(args.enabled_params) and "K_y_cp_tpl" not in stiffness_templates:
        raise KeyError(
            "alpha_y_cpN is enabled, but stiffness_templates has no 'K_y_cp_tpl'. "
            f"Available templates: {list(stiffness_templates.keys())}"
        )

    # ------------------------------------------------------------------
    # 2. Build core
    # ------------------------------------------------------------------
    print()
    print("[2/5] Build dynamic physical core")

    core = DynamicPhysicalCoreTorch(
        M0=M0,
        K0=K0,
        C0=C0,
        stiffness_templates=stiffness_templates,
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

    D = int(core.n_dofs)
    P = int(registry.total_dim)

    print(f"  n_dofs = {D}")
    print(f"  theta_dim = {P}")
    print(f"  enabled = {registry.names}")

    # ------------------------------------------------------------------
    # 3. K_eff zero / nonzero theta test
    # ------------------------------------------------------------------
    print()
    print("[3/5] Check K_eff assembly")

    theta_zero = torch.zeros(P, dtype=torch.float64)
    K_eff_zero = core.assemble_stiffness(theta_zero)[0]

    K0_t = core.K0
    zero_k_rel_err = rel_fro(K_eff_zero, K0_t)

    theta_nonzero = torch.zeros(P, dtype=torch.float64)

    _fill_theta_for_enabled_params(
        theta=theta_nonzero,
        registry=registry,
        alpha_y_value=float(args.alpha_y_test),
        alpha_xy_value=float(args.alpha_xy_test),
        alpha_y_cp_value=float(args.alpha_y_cp_test),
    )

    print()
    print("[Theta Test Values]")
    print(f"  theta_nonzero shape = {tuple(theta_nonzero.shape)}")
    print(f"  theta_nonzero       = {theta_nonzero.detach().cpu().numpy()}")

    K_eff_nonzero = core.assemble_stiffness(theta_nonzero)[0]
    nonzero_k_rel_change = rel_fro(K_eff_nonzero, K0_t)

    K_eff_sym = float(
        torch.max(torch.abs(K_eff_nonzero - K_eff_nonzero.T)).detach().cpu()
    )

    zero_k_ok = bool(zero_k_rel_err <= float(args.zero_atol))
    nonzero_k_ok = bool(nonzero_k_rel_change > 0.0)
    symmetry_ok = bool(K_eff_sym <= float(args.sym_atol))

    if not zero_k_ok:
        raise AssertionError(
            f"theta=0 should return K0, got rel_err={zero_k_rel_err:.6e}"
        )

    if not nonzero_k_ok:
        raise AssertionError("theta!=0 did not change K_eff.")

    if not symmetry_ok:
        raise AssertionError(f"K_eff is not symmetric enough: {K_eff_sym:.6e}")

    # ------------------------------------------------------------------
    # 4. One-step Newmark zero / nonzero theta response test
    # ------------------------------------------------------------------
    print()
    print("[4/5] Check one-step Newmark response")

    u0 = torch.zeros(D, dtype=torch.float64)
    v0 = torch.zeros(D, dtype=torch.float64)
    a0 = torch.zeros(D, dtype=torch.float64)

    F1 = torch.zeros(D, dtype=torch.float64)
    # 最后一个自由节点 uy DOF = D - 5，因为每节点 [ux, uy, uz, rx, ry, rz]
    F1[D - 5] = float(args.force_scale)

    u_none, v_none, a_none = core.newmark_step(
        u_t=u0,
        v_t=v0,
        a_t=a0,
        F_t1=F1,
        theta_t=None,
    )

    u_zero, v_zero, a_zero = core.newmark_step(
        u_t=u0,
        v_t=v0,
        a_t=a0,
        F_t1=F1,
        theta_t=theta_zero,
    )

    u_nonzero, v_nonzero, a_nonzero = core.newmark_step(
        u_t=u0,
        v_t=v0,
        a_t=a0,
        F_t1=F1,
        theta_t=theta_nonzero,
    )

    zero_step_u_rel_err = rel_vec(u_zero, u_none)
    nonzero_step_u_rel_change = rel_vec(u_nonzero, u_zero)

    zero_step_ok = bool(zero_step_u_rel_err <= float(args.zero_atol))
    nonzero_step_ok = bool(nonzero_step_u_rel_change > 0.0)

    if not zero_step_ok:
        raise AssertionError(
            "theta=0 and theta=None should produce the same one-step response, "
            f"got rel_err={zero_step_u_rel_err:.6e}"
        )

    if not nonzero_step_ok:
        raise AssertionError("theta!=0 did not change one-step response.")

    # ------------------------------------------------------------------
    # 5. Gradient test
    # ------------------------------------------------------------------
    print()
    print("[5/5] Check gradient from response back to theta")

    theta_train = theta_nonzero.clone().detach().requires_grad_(True)

    u_grad, v_grad, a_grad = core.newmark_step(
        u_t=u0,
        v_t=v0,
        a_t=a0,
        F_t1=F1,
        theta_t=theta_train,
    )

    loss = (
        torch.mean(u_grad ** 2)
        + 1.0e-4 * torch.mean(v_grad ** 2)
        + 1.0e-8 * torch.mean(a_grad ** 2)
    )

    loss.backward()

    grad = theta_train.grad
    grad_finite = bool(grad is not None and torch.all(torch.isfinite(grad)).item())
    grad_norm = float(torch.linalg.norm(grad).detach().cpu()) if grad is not None else 0.0
    grad_nonzero = bool(grad_norm > 0.0)

    if not grad_finite:
        raise AssertionError(f"theta gradient is not finite: {grad}")

    if not grad_nonzero:
        raise AssertionError("theta gradient norm is zero; response did not backprop to theta.")

    report = {
        "passed": True,
        "registry": registry.summary(),
        "template_summary": bundle.summary(),
        "core": {
            "n_dofs": D,
            "theta_dim": P,
            "dt": float(args.dt),
            "damping_beta": float(args.damping_beta),
        },
        "checks": {
            "zero_k_ok": zero_k_ok,
            "nonzero_k_ok": nonzero_k_ok,
            "symmetry_ok": symmetry_ok,
            "zero_step_ok": zero_step_ok,
            "nonzero_step_ok": nonzero_step_ok,
            "grad_finite": grad_finite,
            "grad_nonzero": grad_nonzero,
        },
        "metrics": {
            "zero_k_rel_err": zero_k_rel_err,
            "nonzero_k_rel_change": nonzero_k_rel_change,
            "K_eff_sym_max_abs": K_eff_sym,
            "zero_step_u_rel_err": zero_step_u_rel_err,
            "nonzero_step_u_rel_change": nonzero_step_u_rel_change,
            "grad_norm": grad_norm,
            "loss": float(loss.detach().cpu()),
        },
        "theta_test": {
            "theta_nonzero": theta_nonzero.detach().cpu().numpy(),
            "alpha_y_test": float(args.alpha_y_test),
            "alpha_xy_test": float(args.alpha_xy_test),
            "alpha_y_cp_test": float(args.alpha_y_cp_test),
        },
        "stiffness_template_keys": list(stiffness_templates.keys()),
    }

    report_path = output_dir / "dynamic_physical_core_torch_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(report), f, indent=2, ensure_ascii=False)

    print()
    print("[Summary]")
    print(f"  zero_k_rel_err             = {zero_k_rel_err:.6e}")
    print(f"  nonzero_k_rel_change       = {nonzero_k_rel_change:.6e}")
    print(f"  K_eff_sym_max_abs          = {K_eff_sym:.6e}")
    print(f"  zero_step_u_rel_err        = {zero_step_u_rel_err:.6e}")
    print(f"  nonzero_step_u_rel_change  = {nonzero_step_u_rel_change:.6e}")
    print(f"  grad_norm                  = {grad_norm:.6e}")
    print(f"  report                     = {report_path}")

    print()
    print("✅ PASS: DynamicPhysicalCoreTorch can assemble K_eff, run one Newmark step, and backpropagate to theta.")


if __name__ == "__main__":
    main()