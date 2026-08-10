"""Causal prefix fitting for graph-modal discrepancy dynamics."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._graph_dynamic_discrepancy_common import (
    GRAPH_DYNAMIC_DISCREPANCY_BOUNDARY,
    GRAPH_DYNAMIC_DISCREPANCY_SCHEMA,
    GRAPH_DYNAMIC_DISCREPANCY_VERSION,
    GraphDynamicDiscrepancyConfigV1,
    _real,
    _require,
    _state_array,
    _symmetric,
    _transition_and_noise,
    _validate_graph_basis,
)
from ._graph_dynamic_discrepancy_contract import (
    GraphDynamicDiscrepancyBeliefV1,
)
from ._graph_dynamic_discrepancy_observation import (
    _covariance_array,
    _frame_group_labels,
    _frame_update,
)


def fit_graph_dynamic_discrepancy(
    residual_m: np.ndarray,
    valid: np.ndarray,
    graph_basis: np.ndarray,
    *,
    frame_dt_s: float,
    observation_covariance_m2: np.ndarray | None = None,
    prior_reliability: np.ndarray | None = None,
    correlation_group_ids: np.ndarray | None = None,
    config: GraphDynamicDiscrepancyConfigV1 | None = None,
) -> GraphDynamicDiscrepancyBeliefV1:
    """Filter a causal residual prefix into one graph-modal endpoint belief."""

    if config is None:
        cfg = GraphDynamicDiscrepancyConfigV1()
    elif isinstance(config, GraphDynamicDiscrepancyConfigV1):
        cfg = config
    else:
        raise TypeError("config must be a GraphDynamicDiscrepancyConfigV1")
    frame_dt = _real(frame_dt_s, name="frame_dt_s")
    _require(frame_dt > 0.0, "frame_dt_s must be positive")
    raw_residual = np.asarray(residual_m)
    _require(
        raw_residual.dtype.kind in {"i", "u", "f"},
        "residual_m must be real numeric",
    )
    residual = np.asarray(raw_residual, dtype=np.float64)
    raw_availability = np.asarray(valid)
    _require(
        raw_availability.dtype.kind == "b",
        "valid must be a Boolean array",
    )
    availability = np.asarray(raw_availability, dtype=bool)
    basis = _validate_graph_basis(graph_basis)
    _require(
        residual.ndim == 3 and residual.shape[2] == 3,
        "residual_m must have shape (frame, node, 3)",
    )
    frame_count, node_count, _ = residual.shape
    _require(frame_count >= 1, "residual_m must contain at least one frame")
    _require(
        availability.shape == (frame_count, node_count),
        "valid shape changed",
    )
    _require(
        basis.shape[0] == node_count,
        "graph_basis does not cover every residual node",
    )
    finite = np.all(np.isfinite(residual), axis=2)
    usable = availability & finite
    if prior_reliability is None:
        reliability = np.ones((frame_count, node_count), dtype=np.float64)
    else:
        raw_reliability = np.asarray(prior_reliability)
        _require(
            raw_reliability.dtype.kind in {"i", "u", "f"},
            "prior_reliability must be real numeric",
        )
        reliability = np.asarray(raw_reliability, dtype=np.float64).copy()
        _require(
            reliability.shape == (frame_count, node_count),
            "prior_reliability shape changed",
        )
        _require(
            np.all(np.isfinite(reliability))
            and np.all((reliability >= 0.0) & (reliability <= 1.0)),
            "prior_reliability must lie in [0, 1]",
        )
    reliability[~usable] = 0.0
    covariance = _covariance_array(
        observation_covariance_m2,
        frame_count=frame_count,
        node_count=node_count,
        default_std_m=cfg.observation_std_m,
    )
    group_labels = _frame_group_labels(
        correlation_group_ids,
        frame_count=frame_count,
        node_count=node_count,
    )
    rank = basis.shape[1]
    position_dimension = 3 * rank
    state_dimension = 6 * rank
    state_mean = np.zeros(state_dimension, dtype=np.float64)
    state_covariance = np.zeros(
        (state_dimension, state_dimension),
        dtype=np.float64,
    )
    state_covariance[:position_dimension, :position_dimension] = (
        np.eye(position_dimension) * cfg.initial_position_std_m**2
    )
    state_covariance[position_dimension:, position_dimension:] = (
        np.eye(position_dimension) * cfg.initial_velocity_std_mps**2
    )
    transition, process_noise, _ = _transition_and_noise(
        rank,
        frame_dt_s=frame_dt,
        velocity_retention=cfg.velocity_retention,
        process_position_std_m=cfg.process_position_std_m,
        process_acceleration_std_mps2=cfg.process_acceleration_std_mps2,
    )
    accepted = np.zeros(frame_count, dtype=bool)
    reasons: list[str] = []
    frame_diagnostics: list[dict[str, Any]] = []
    for frame in range(frame_count):
        if frame:
            state_mean = transition @ state_mean
            state_covariance = _symmetric(
                transition @ state_covariance @ transition.T + process_noise
            )
        (
            state_mean,
            state_covariance,
            accepted[frame],
            reason,
            diagnostics,
        ) = _frame_update(
            state_mean,
            state_covariance,
            residual[frame],
            usable[frame],
            basis,
            covariance[frame],
            reliability[frame],
            group_labels[frame],
            cfg,
        )
        reasons.append(reason)
        frame_diagnostics.append(
            {"frame_index": frame, "reason": reason, **diagnostics}
        )
    state = _state_array(state_mean, rank)
    summary = {
        "schema": GRAPH_DYNAMIC_DISCREPANCY_SCHEMA,
        "schema_version": GRAPH_DYNAMIC_DISCREPANCY_VERSION,
        "scientific_boundary": GRAPH_DYNAMIC_DISCREPANCY_BOUNDARY,
        "frame_count": frame_count,
        "node_count": node_count,
        "graph_rank": rank,
        "accepted_update_count": int(np.sum(accepted)),
        "rejected_update_count": int(np.sum(~accepted & np.any(usable, axis=1))),
        "unsupported_frame_count": int(np.sum(~np.any(usable, axis=1))),
        "observation_likelihood": "grouped Student-t IRLS",
        "prior_reliability_uses_innovation": False,
        "frame_diagnostics": frame_diagnostics,
    }
    return GraphDynamicDiscrepancyBeliefV1(
        graph_basis=basis,
        state_mean=state,
        state_covariance=state_covariance,
        frame_dt_s=frame_dt,
        velocity_retention=cfg.velocity_retention,
        process_position_std_m=cfg.process_position_std_m,
        process_acceleration_std_mps2=cfg.process_acceleration_std_mps2,
        last_frame_index=frame_count - 1,
        update_accepted=accepted,
        update_reasons=tuple(reasons),
        diagnostics=summary,
    )
