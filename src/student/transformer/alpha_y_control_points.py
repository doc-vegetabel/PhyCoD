from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class AlphaYControlPointConfig:
    """
    Configuration for spanwise alpha_y control-point templates.

    alpha_y_cpN means:
        theta_t = [alpha_y_cp_0(t), ..., alpha_y_cp_{N-1}(t)]

    These control-point values are interpolated along blade span
    and converted into N stiffness residual templates:

        K_eff(t) = K0 + sum_i alpha_y_cp_i(t) * K_y_cp_tpl[i]

    The first implementation uses simple linear hat basis functions
    along the free-node span.
    """

    n_control_points: int
    scale_mode: str = "y_bending"
    finite_difference_delta: float = 1.0e-3


def parse_alpha_y_cp_name(name: str) -> int | None:
    """
    Parse parameter names like:
        alpha_y_cp3
        alpha_y_cp4
        alpha_y_cp6

    Returns:
        n_control_points if matched, otherwise None.
    """
    m = re.fullmatch(r"alpha_y_cp(\d+)", str(name).strip())
    if m is None:
        return None

    n_cp = int(m.group(1))
    if n_cp < 2:
        raise ValueError(
            f"alpha_y control points must be >= 2, got {n_cp} from {name}"
        )
    return n_cp


def build_linear_hat_control_point_weights(
    *,
    n_nodes: int,
    n_control_points: int,
) -> np.ndarray:
    """
    Build spanwise linear hat basis weights.

    Returns:
        weights: shape = (n_control_points, n_nodes)

    Each node has weights over control points, and approximately:
        sum_i weights[i, node] = 1
    """
    if n_nodes <= 1:
        raise ValueError(f"n_nodes must be > 1, got {n_nodes}")
    if n_control_points < 2:
        raise ValueError(
            f"n_control_points must be >= 2, got {n_control_points}"
        )

    eta_nodes = np.linspace(0.0, 1.0, int(n_nodes), dtype=np.float64)
    eta_cp = np.linspace(0.0, 1.0, int(n_control_points), dtype=np.float64)

    weights = np.zeros((int(n_control_points), int(n_nodes)), dtype=np.float64)

    for i, c in enumerate(eta_cp):
        if i == 0:
            right = eta_cp[i + 1]
            weights[i] = np.clip((right - eta_nodes) / max(right - c, 1.0e-12), 0.0, 1.0)
        elif i == n_control_points - 1:
            left = eta_cp[i - 1]
            weights[i] = np.clip((eta_nodes - left) / max(c - left, 1.0e-12), 0.0, 1.0)
        else:
            left = eta_cp[i - 1]
            right = eta_cp[i + 1]
            w_left = (eta_nodes - left) / max(c - left, 1.0e-12)
            w_right = (right - eta_nodes) / max(right - c, 1.0e-12)
            weights[i] = np.maximum(0.0, np.minimum(w_left, w_right))

    # Normalize to avoid tiny numerical deviation.
    denom = np.sum(weights, axis=0, keepdims=True)
    weights = weights / np.maximum(denom, 1.0e-12)

    return weights


def y_bending_dof_offsets_for_mode(scale_mode: str) -> Sequence[int]:
    """
    DOF layout per free node:
        ux, uy, uz, rx, ry, rz

    For y-bending response, the affected pair is usually:
        uy and rz

    This matches the existing y_bending scale behavior:
        48 free nodes * 2 dofs = 96 scaled dofs.
    """
    mode = str(scale_mode).strip().lower()

    if mode == "y_bending":
        return (1, 5)

    raise ValueError(
        f"Unsupported alpha_y_cp scale_mode: {scale_mode}. "
        f"Currently only 'y_bending' is supported."
    )


def build_control_point_dof_weights(
    *,
    n_dofs: int,
    n_control_points: int,
    scale_mode: str = "y_bending",
) -> np.ndarray:
    """
    Convert node-level control-point weights into full DOF-level weights.

    Returns:
        dof_weights: shape = (n_control_points, n_dofs)

    Only selected y-bending DOFs receive nonzero weights.
    """
    n_dofs = int(n_dofs)
    if n_dofs % 6 != 0:
        raise ValueError(f"Expected n_dofs divisible by 6, got {n_dofs}")

    n_nodes = n_dofs // 6
    node_weights = build_linear_hat_control_point_weights(
        n_nodes=n_nodes,
        n_control_points=int(n_control_points),
    )

    dof_weights = np.zeros((int(n_control_points), n_dofs), dtype=np.float64)
    offsets = y_bending_dof_offsets_for_mode(scale_mode)

    for node in range(n_nodes):
        for off in offsets:
            dof = node * 6 + int(off)
            dof_weights[:, dof] = node_weights[:, node]

    return dof_weights


def build_alpha_y_cp_stiffness_templates(
    *,
    K0: np.ndarray,
    n_control_points: int,
    scale_mode: str = "y_bending",
    finite_difference_delta: float = 1.0e-3,
) -> dict[str, np.ndarray]:
    """
    Build control-point stiffness residual templates around current K0.

    For each control point i, construct a small weighted congruence scaling:

        S_i = diag(1 + delta * w_i)

        K_i(delta) = S_i @ K0 @ S_i

        K_y_cp_tpl[i] = (K_i(delta) - K0) / delta

    Then dynamic assembly uses:

        K_eff(t) = K0 + sum_i alpha_i(t) * K_y_cp_tpl[i]

    This is a local linearized stiffness-residual basis, not output correction.
    It still modifies K and the response is solved by Newmark / MCK.
    """
    K0 = np.asarray(K0, dtype=np.float64)
    if K0.ndim != 2 or K0.shape[0] != K0.shape[1]:
        raise ValueError(f"K0 must be square 2D, got {K0.shape}")

    n_dofs = int(K0.shape[0])
    delta = float(finite_difference_delta)
    if delta <= 0.0:
        raise ValueError(f"finite_difference_delta must be > 0, got {delta}")

    dof_weights = build_control_point_dof_weights(
        n_dofs=n_dofs,
        n_control_points=int(n_control_points),
        scale_mode=scale_mode,
    )

    templates = []
    for i in range(int(n_control_points)):
        s = 1.0 + delta * dof_weights[i]
        K_scaled = (s[:, None] * K0) * s[None, :]
        K_tpl_i = (K_scaled - K0) / delta
        K_tpl_i = 0.5 * (K_tpl_i + K_tpl_i.T)
        templates.append(K_tpl_i)

    K_y_cp_tpl = np.stack(templates, axis=0)

    return {
        "K_y_cp_tpl": K_y_cp_tpl,
        "alpha_y_cp_dof_weights": dof_weights,
    }