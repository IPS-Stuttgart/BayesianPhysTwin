"""Source-only paired error transport about a frozen learned readout.

The centering defect is a prior prediction, never an observed future residual.
The returned arrays are readouts, not guaranteed feasible physical trajectories.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from .deform_state_restart import (
    RestartConfig,
    RodState,
    paired_physical_readout,
    prediction_metrics,
    update_rod_state,
)

SCHEMA = "deform-reference-transport-source-v1"
ARMS = ("incumbent", "paired", "reference_initialized", "reference_centered")
PRIMARY = "reference_centered"
HORIZONS = {"all": (0, 120), "early": (0, 40), "middle": (40, 80), "late": (80, 120)}


def content_id(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def learned_reference_offsets(
    incumbent: np.ndarray, archived_physical: np.ndarray, config: RestartConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Use a backward difference of the already-predicted readout offset."""
    a, b = np.asarray(incumbent), np.asarray(archived_physical)
    shape = (config.forecast_end, config.node_count, 3)
    if (
        a.ndim != 4
        or a.shape[1:] != shape
        or b.shape != a.shape
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("aligned finite frozen forecast arrays required")
    offset = a.astype(np.float64) - b.astype(np.float64)
    if np.count_nonzero(offset[:, :, config.clamped_nodes]):
        raise ValueError("learned readout cannot move prescribed clamps")
    start = config.prefix_length - 1
    if start < 1:
        raise ValueError("backward velocity needs a preceding prefix frame")
    velocity = np.diff(offset[:, start - 1 :], axis=1) / config.dt_s
    return offset[:, start:].copy(), velocity


def _finite_state(state: RodState) -> None:
    for name in ("positions", "velocity", "previous_positions", "material_u0", "theta"):
        if not getattr(state, name).isfinite().all():
            raise ValueError("nonfinite native state")


def _translate(state: RodState, dx: Any, dv: Any, clamps: tuple[int, ...]) -> RodState:
    return update_rod_state(state, dx, dv, gain=1.0, clamped_nodes=clamps)


def transport_pair(
    *,
    advance: Callable[[RodState, np.ndarray], RodState],
    nominal_states: Sequence[RodState],
    future_actions: np.ndarray,
    incumbent: np.ndarray,
    pose_increment: Any,
    velocity_increment: Any,
    position_offsets: Any,
    velocity_offsets: Any,
    clamped_nodes: tuple[int, ...],
    mode: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Propagate the same state innovation with optional common re-centering.

    At each centering, both branches receive the same positional/velocity
    translation; their state difference and separate material memories survive.
    No future observation is accepted by this API.
    """
    if mode not in ("reference_initialized", "reference_centered"):
        raise ValueError("unknown frozen transport arm")
    if len(nominal_states) < 2:
        raise ValueError("endpoint plus future nominal states required")
    first = nominal_states[0]
    batch, nodes, axes = first.positions.shape
    steps = len(nominal_states) - 1
    expected_offsets = (batch, steps + 1, nodes, 3)
    if (
        axes != 3
        or incumbent.shape != (batch, steps, nodes, 3)
        or future_actions.shape != (batch, steps, len(clamped_nodes), 3)
        or position_offsets.shape != expected_offsets
        or velocity_offsets.shape != expected_offsets
        or not np.isfinite(incumbent).all()
        or not np.isfinite(future_actions).all()
        or not position_offsets.isfinite().all()
        or not velocity_offsets.isfinite().all()
    ):
        raise ValueError("transport state/time/identity arrays differ")
    if (
        position_offsets[:, :, clamped_nodes].count_nonzero()
        or velocity_offsets[:, :, clamped_nodes].count_nonzero()
    ):
        raise ValueError("reference offsets cannot alter clamps")
    for i, state in enumerate(nominal_states):
        _finite_state(state)
        if state.positions.shape != first.positions.shape:
            raise ValueError("nominal state geometry changed")
        if state.prediction_index != first.prediction_index + i:
            raise ValueError("nominal states must be consecutive")

    center = _translate(
        first, position_offsets[:, 0], velocity_offsets[:, 0], clamped_nodes
    )
    updated = _translate(center, pose_increment, velocity_increment, clamped_nodes)
    # Validate increments even on the exact no-update branch.
    if not pose_increment.count_nonzero() and not velocity_increment.count_nonzero():
        return incumbent, {}

    trace: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "center_before",
            "updated_before",
            "center_after",
            "updated_after",
            "center_velocity_before",
            "updated_velocity_before",
            "center_velocity_after",
            "updated_velocity_after",
            "centering_dx",
            "centering_dv",
        )
    }

    def record(name: str, tensor: Any) -> None:
        trace[name].append(tensor.detach().cpu().numpy().copy())

    for step in range(steps):
        record("center_before", center.positions)
        record("updated_before", updated.positions)
        record("center_velocity_before", center.velocity)
        record("updated_velocity_before", updated.velocity)
        center = advance(center, future_actions[:, step])
        updated = advance(updated, future_actions[:, step])
        _finite_state(center)
        _finite_state(updated)
        record("center_after", center.positions)
        record("updated_after", updated.positions)
        record("center_velocity_after", center.velocity)
        record("updated_velocity_after", updated.velocity)
        dx, dv = center.positions * 0, center.velocity * 0
        if mode == "reference_centered" and step + 1 < steps:
            target = nominal_states[step + 1]
            # Subtract before adding the offset: an exact zero defect stays zero.
            dx = target.positions - center.positions + position_offsets[:, step + 1]
            dv = target.velocity - center.velocity + velocity_offsets[:, step + 1]
            # Prescribed position controls are exact. Their velocity bookkeeping
            # is not an inferred state variable and remains on the native branch.
            if dx[:, clamped_nodes].count_nonzero():
                raise ValueError("native branches disagree on prescribed clamps")
            dv = dv.clone()
            dv[:, clamped_nodes] = 0
            center = _translate(center, dx, dv, clamped_nodes)
            updated = _translate(updated, dx, dv, clamped_nodes)
        record("centering_dx", dx)
        record("centering_dv", dv)
    arrays = {name: np.stack(values, axis=1) for name, values in trace.items()}
    prediction = paired_physical_readout(
        incumbent, arrays["center_after"], arrays["updated_after"]
    )
    return prediction, arrays


def source_decision(
    case_metrics: dict[str, dict[str, dict[str, dict[str, float]]]],
) -> dict[str, Any]:
    """Frozen finite-source gate; secondaries cannot rescue the primary."""
    if len(case_metrics) != 13 or "103.pkl" in case_metrics:
        raise ValueError("exactly thirteen non-design source cases required")
    names = sorted(case_metrics)
    means = {
        arm: {
            h: {
                metric: float(np.mean([case_metrics[n][arm][h][metric] for n in names]))
                for metric in ("coordinate_l1_mm", "point_rmse_mm", "fde_mm")
            }
            for h in HORIZONS
        }
        for arm in ARMS
    }
    base, candidate = means["paired"], means[PRIMARY]
    wins = sum(
        all(
            case_metrics[n][PRIMARY]["all"][m] < case_metrics[n]["paired"]["all"][m]
            for m in ("coordinate_l1_mm", "point_rmse_mm")
        )
        for n in names
    )
    worst = max(
        case_metrics[n][PRIMARY]["all"]["point_rmse_mm"]
        / max(case_metrics[n]["paired"]["all"]["point_rmse_mm"], 1e-12)
        for n in names
    )
    differences = np.array(
        [
            case_metrics[n][PRIMARY]["all"]["point_rmse_mm"]
            - case_metrics[n]["paired"]["all"]["point_rmse_mm"]
            for n in names
        ]
    )
    rng = np.random.default_rng(260929)
    draws = rng.integers(0, len(names), (10000, len(names)))
    bounds = np.asarray(np.quantile(differences[draws].mean(axis=1), [0.025, 0.975]))
    interval = [float(bounds[0]), float(bounds[1])]
    checks = {
        "two_percent_l1_gain": candidate["all"]["coordinate_l1_mm"]
        <= 0.98 * base["all"]["coordinate_l1_mm"],
        "two_percent_rmse_gain": candidate["all"]["point_rmse_mm"]
        <= 0.98 * base["all"]["point_rmse_mm"],
        "late_rmse_nonincreasing": candidate["late"]["point_rmse_mm"]
        <= base["late"]["point_rmse_mm"],
        "eight_of_thirteen_joint_wins": wins >= 8,
        "worst_rmse_ratio_at_most_1_05": worst <= 1.05,
        "rmse_bootstrap_upper_below_zero": interval[1] < 0,
    }
    return {
        "means": means,
        "checks": checks,
        "passed": all(checks.values()),
        "primary_joint_wins": wins,
        "primary_worst_case_rmse_ratio": worst,
        "paired_rmse_difference_95pct_mm": interval,
        "future_transfer_authorized": False,
        "incumbent_modified": False,
    }


def score_predictions(
    names: list[str],
    predictions: dict[str, np.ndarray],
    truth: np.ndarray,
    config: RestartConfig,
) -> dict[str, Any]:
    if len(names) != 14 or len(set(names)) != 14 or config.design_case not in names:
        raise ValueError("the complete fourteen-case source roster is required")
    expected = (14, 120, config.node_count, 3)
    if set(predictions) != set(ARMS) or truth.shape != expected:
        raise ValueError("all frozen arms and source truths must align")
    if any(p.shape != expected for p in predictions.values()):
        raise ValueError("prediction geometry changed")
    metrics = {}
    for i, name in enumerate(names):
        if name == config.design_case:
            continue
        metrics[name] = {
            arm: {
                h: prediction_metrics(
                    p[i, start:end][:, config.hidden_nodes],
                    truth[i, start:end][:, config.hidden_nodes],
                )
                for h, (start, end) in HORIZONS.items()
            }
            for arm, p in predictions.items()
        }
    return {"case_metrics": metrics, "decision": source_decision(metrics)}


def config_for_source() -> RestartConfig:
    return replace(RestartConfig(), seed=260929)
