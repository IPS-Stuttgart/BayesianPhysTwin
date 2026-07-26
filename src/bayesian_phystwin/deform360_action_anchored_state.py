"""Action-anchored endpoint-state estimates for sparse deformable graphs.

The observation channel and the measured controller channel play different
roles here.  Consecutive prefix geometries provide a noisy material velocity,
while registered controller motion provides hard values only at contact nodes.
A shared velocity bias is estimated from their disagreement at those nodes,
then the corrected field is smoothed on the graph with the controller values
held fixed.

This is a small Bayesian-PhysTwin adapter motivated by TrackDeform3D's
per-frame end-effector anchoring.  It is not an implementation or benchmark of
the upstream RGB-D tracker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class ActionAnchoredStateConfig:
    """Frozen controls for one graph-regularized endpoint-state estimate."""

    laplacian_weight: float = 4.0
    solve_ridge: float = 1e-9
    maximum_initial_speed_m_s: float = 1.5
    minimum_orientation_margin: float = 1e-6

    def __post_init__(self) -> None:
        _require(self.laplacian_weight >= 0.0, "Laplacian weight must be nonnegative")
        _require(self.solve_ridge > 0.0, "solve ridge must be positive")
        _require(
            self.maximum_initial_speed_m_s > 0.0,
            "maximum initial speed must be positive",
        )
        _require(
            self.minimum_orientation_margin >= 0.0,
            "minimum orientation margin must be nonnegative",
        )


@dataclass(frozen=True)
class ActionAnchoredStateEstimate:
    """Candidate velocity arms and their target-free diagnostics."""

    aligned_previous_positions_m: np.ndarray
    camera_velocity_m_s: np.ndarray
    camera_smoothed_velocity_m_s: np.ndarray
    action_harmonic_velocity_m_s: np.ndarray
    bias_corrected_action_velocity_m_s: np.ndarray
    anchor_velocity_m_s: np.ndarray
    shared_observation_bias_m_s: np.ndarray
    accepted: bool
    diagnostics: dict[str, Any]

    def __post_init__(self) -> None:
        arrays = {
            "aligned_previous_positions_m": self.aligned_previous_positions_m,
            "camera_velocity_m_s": self.camera_velocity_m_s,
            "camera_smoothed_velocity_m_s": self.camera_smoothed_velocity_m_s,
            "action_harmonic_velocity_m_s": self.action_harmonic_velocity_m_s,
            "bias_corrected_action_velocity_m_s": (
                self.bias_corrected_action_velocity_m_s
            ),
        }
        shape = np.asarray(self.camera_velocity_m_s).shape
        _require(
            len(shape) == 2 and shape[1] == 3,
            "state velocities must have shape (N,3)",
        )
        for name, values in arrays.items():
            array = np.asarray(values, dtype=np.float64)
            _require(array.shape == shape, f"{name} differs from the graph")
            _require(np.all(np.isfinite(array)), f"{name} contains nonfinite values")
            copied = array.copy()
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)
        anchor = np.asarray(self.anchor_velocity_m_s, dtype=np.float64)
        bias = np.asarray(self.shared_observation_bias_m_s, dtype=np.float64)
        _require(
            anchor.ndim == 2 and anchor.shape[1] == 3 and len(anchor) >= 1,
            "anchor velocities must have shape (A,3)",
        )
        _require(bias.shape == (3,), "shared observation bias must have shape (3,)")
        _require(
            np.all(np.isfinite(anchor)) and np.all(np.isfinite(bias)),
            "anchor velocity and shared bias must be finite",
        )
        for name, values in (
            ("anchor_velocity_m_s", anchor),
            ("shared_observation_bias_m_s", bias),
        ):
            copied = values.copy()
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)


def chain_laplacian(node_count: int) -> np.ndarray:
    """Return the unnormalized Laplacian of an ordered chain."""

    _require(node_count >= 2, "a chain needs at least two nodes")
    laplacian = np.zeros((node_count, node_count), dtype=np.float64)
    indices = np.arange(node_count - 1)
    laplacian[indices, indices] += 1.0
    laplacian[indices + 1, indices + 1] += 1.0
    laplacian[indices, indices + 1] -= 1.0
    laplacian[indices + 1, indices] -= 1.0
    return laplacian


def align_chain_orientation(
    previous_positions_m: np.ndarray,
    current_positions_m: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    """Align an ordered previous chain to the current chain without outcomes."""

    previous = np.asarray(previous_positions_m, dtype=np.float64)
    current = np.asarray(current_positions_m, dtype=np.float64)
    _require(
        previous.shape == current.shape
        and previous.ndim == 2
        and previous.shape[1] == 3,
        "prefix chains must have matching shape (N,3)",
    )
    _require(
        len(previous) >= 2
        and np.all(np.isfinite(previous))
        and np.all(np.isfinite(current)),
        "prefix chains must be finite and nonempty",
    )
    direct = float(np.mean(np.sum((previous - current) ** 2, axis=1)))
    reversed_cost = float(np.mean(np.sum((previous[::-1] - current) ** 2, axis=1)))
    use_reversed = reversed_cost < direct
    selected = reversed_cost if use_reversed else direct
    alternative = direct if use_reversed else reversed_cost
    margin = (alternative - selected) / max(alternative, np.finfo(float).eps)
    aligned = previous[::-1].copy() if use_reversed else previous.copy()
    return aligned, {
        "reversed": use_reversed,
        "direct_mean_squared_m2": direct,
        "reversed_mean_squared_m2": reversed_cost,
        "relative_orientation_margin": float(margin),
    }


def _validate_anchors(
    node_count: int,
    anchor_node_indices: np.ndarray,
    anchor_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(anchor_node_indices, dtype=np.int64)
    values = np.asarray(anchor_values, dtype=np.float64)
    _require(indices.ndim == 1 and len(indices) >= 1, "at least one anchor is required")
    _require(
        len(np.unique(indices)) == len(indices)
        and np.all(indices >= 0)
        and np.all(indices < node_count),
        "anchor indices must be unique and inside the graph",
    )
    _require(
        values.shape == (len(indices), 3) and np.all(np.isfinite(values)),
        "anchor values must have shape (A,3) and be finite",
    )
    return indices, values


def _smooth_field(
    raw_values: np.ndarray,
    laplacian: np.ndarray,
    *,
    weight: float,
    ridge: float,
    anchor_indices: np.ndarray | None = None,
    anchor_values: np.ndarray | None = None,
) -> np.ndarray:
    raw = np.asarray(raw_values, dtype=np.float64)
    node_count = len(raw)
    _require(raw.shape == (node_count, 3), "raw graph field must have shape (N,3)")
    _require(laplacian.shape == (node_count, node_count), "Laplacian shape differs")
    system = np.eye(node_count, dtype=np.float64) + weight * laplacian
    if anchor_indices is None:
        return np.linalg.solve(system + ridge * np.eye(node_count), raw)

    indices, values = _validate_anchors(node_count, anchor_indices, anchor_values)
    free = np.setdiff1d(np.arange(node_count), indices, assume_unique=False)
    output = np.empty_like(raw)
    output[indices] = values
    if len(free):
        free_system = system[np.ix_(free, free)]
        right = raw[free] - system[np.ix_(free, indices)] @ values
        output[free] = np.linalg.solve(
            free_system + ridge * np.eye(len(free)),
            right,
        )
    return output


def _harmonic_extension(
    node_count: int,
    laplacian: np.ndarray,
    anchor_indices: np.ndarray,
    anchor_values: np.ndarray,
    *,
    ridge: float,
) -> np.ndarray:
    indices, values = _validate_anchors(node_count, anchor_indices, anchor_values)
    free = np.setdiff1d(np.arange(node_count), indices, assume_unique=False)
    output = np.empty((node_count, 3), dtype=np.float64)
    output[indices] = values
    if not len(free):
        return output
    free_laplacian = laplacian[np.ix_(free, free)]
    right = -laplacian[np.ix_(free, indices)] @ values
    output[free] = np.linalg.solve(
        free_laplacian + ridge * np.eye(len(free)),
        right,
    )
    return output


def _speed_summary(values: np.ndarray) -> dict[str, float]:
    speed = np.linalg.norm(values, axis=1)
    return {
        "median_m_s": float(np.median(speed)),
        "p95_m_s": float(np.quantile(speed, 0.95)),
        "maximum_m_s": float(np.max(speed)),
    }


def estimate_action_anchored_chain_state(
    previous_positions_m: np.ndarray,
    current_positions_m: np.ndarray,
    previous_controller_positions_m: np.ndarray,
    current_controller_positions_m: np.ndarray,
    anchor_node_indices: np.ndarray,
    *,
    dt_seconds: float,
    config: ActionAnchoredStateConfig | None = None,
) -> ActionAnchoredStateEstimate:
    """Estimate four endpoint-velocity arms from two causal prefix states."""

    cfg = config or ActionAnchoredStateConfig()
    _require(dt_seconds > 0.0, "prefix interval must be positive")
    current = np.asarray(current_positions_m, dtype=np.float64)
    aligned_previous, orientation = align_chain_orientation(
        previous_positions_m,
        current,
    )
    _require(
        orientation["relative_orientation_margin"] >= cfg.minimum_orientation_margin,
        "chain orientation is ambiguous",
    )
    node_count = len(current)
    previous_controllers = np.asarray(
        previous_controller_positions_m,
        dtype=np.float64,
    )
    current_controllers = np.asarray(
        current_controller_positions_m,
        dtype=np.float64,
    )
    _require(
        previous_controllers.shape == current_controllers.shape,
        "controller prefix positions differ in shape",
    )
    indices, _ = _validate_anchors(
        node_count,
        anchor_node_indices,
        current_controllers,
    )
    _require(
        previous_controllers.shape == (len(indices), 3)
        and np.all(np.isfinite(previous_controllers)),
        "controller prefix positions must have shape (A,3)",
    )

    camera_velocity = (current - aligned_previous) / dt_seconds
    anchor_velocity = (current_controllers - previous_controllers) / dt_seconds
    shared_bias = np.median(
        camera_velocity[indices] - anchor_velocity,
        axis=0,
    )
    bias_corrected = camera_velocity - shared_bias
    laplacian = chain_laplacian(node_count)
    camera_smoothed = _smooth_field(
        camera_velocity,
        laplacian,
        weight=cfg.laplacian_weight,
        ridge=cfg.solve_ridge,
    )
    action_harmonic = _harmonic_extension(
        node_count,
        laplacian,
        indices,
        anchor_velocity,
        ridge=cfg.solve_ridge,
    )
    fused = _smooth_field(
        bias_corrected,
        laplacian,
        weight=cfg.laplacian_weight,
        ridge=cfg.solve_ridge,
        anchor_indices=indices,
        anchor_values=anchor_velocity,
    )
    anchor_error = fused[indices] - anchor_velocity
    maximum_speed = float(np.max(np.linalg.norm(fused, axis=1)))
    accepted = maximum_speed <= cfg.maximum_initial_speed_m_s
    diagnostics: dict[str, Any] = {
        "orientation": orientation,
        "anchor_node_indices": indices.astype(int).tolist(),
        "dt_seconds": float(dt_seconds),
        "laplacian_weight": cfg.laplacian_weight,
        "shared_observation_bias_m_s": shared_bias.tolist(),
        "shared_observation_bias_norm_m_s": float(np.linalg.norm(shared_bias)),
        "maximum_hard_anchor_error_m_s": float(
            np.max(np.linalg.norm(anchor_error, axis=1))
        ),
        "camera_velocity": _speed_summary(camera_velocity),
        "camera_smoothed_velocity": _speed_summary(camera_smoothed),
        "action_harmonic_velocity": _speed_summary(action_harmonic),
        "bias_corrected_action_velocity": _speed_summary(fused),
        "maximum_initial_speed_m_s": cfg.maximum_initial_speed_m_s,
        "accepted": accepted,
        "fallback_policy": (
            "use exact zero-velocity physical baseline when the target-free "
            "state gate rejects"
        ),
    }
    return ActionAnchoredStateEstimate(
        aligned_previous_positions_m=aligned_previous,
        camera_velocity_m_s=camera_velocity,
        camera_smoothed_velocity_m_s=camera_smoothed,
        action_harmonic_velocity_m_s=action_harmonic,
        bias_corrected_action_velocity_m_s=fused,
        anchor_velocity_m_s=anchor_velocity,
        shared_observation_bias_m_s=shared_bias,
        accepted=accepted,
        diagnostics=diagnostics,
    )


__all__ = [
    "ActionAnchoredStateConfig",
    "ActionAnchoredStateEstimate",
    "align_chain_orientation",
    "chain_laplacian",
    "estimate_action_anchored_chain_state",
]
