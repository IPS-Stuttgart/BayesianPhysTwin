"""Joint node and horizon forecasts for graph discrepancy beliefs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ._graph_dynamic_discrepancy_common import (
    _DEFAULT_MAXIMUM_COVARIANCE_BYTES,
    _integer,
    _require,
    _state_vector,
    _symmetric,
    _transition_and_noise,
)
from ._graph_dynamic_discrepancy_contract import (
    GraphDynamicDiscrepancyBeliefV1,
    GraphDynamicDiscrepancyForecastV1,
)


def forecast_graph_dynamic_discrepancy(
    self: GraphDynamicDiscrepancyBeliefV1,
    horizon_steps: Sequence[int] | np.ndarray,
    *,
    node_indices: Sequence[int] | np.ndarray | None = None,
    modal_acceleration_mps2: np.ndarray | None = None,
    maximum_covariance_bytes: int = _DEFAULT_MAXIMUM_COVARIANCE_BYTES,
) -> GraphDynamicDiscrepancyForecastV1:
    """Forecast a registered node/horizon query with full joint covariance."""

    raw_horizons = np.asarray(horizon_steps)
    _require(
        raw_horizons.ndim == 1 and len(raw_horizons) > 0, "horizons are empty"
    )
    _require(
        np.issubdtype(raw_horizons.dtype, np.integer)
        and raw_horizons.dtype.kind != "b",
        "horizon_steps must contain integers",
    )
    horizons = np.asarray(raw_horizons, dtype=np.int64)
    _require(np.all(horizons >= 1), "horizon_steps must be positive")
    _require(
        np.all(np.diff(horizons) > 0),
        "horizon_steps must be strictly increasing",
    )
    nodes: np.ndarray
    if node_indices is None:
        nodes = np.arange(self.node_count, dtype=np.int64)
    else:
        raw_nodes = np.asarray(node_indices)
        _require(
            raw_nodes.ndim == 1 and len(raw_nodes) > 0, "node_indices is empty"
        )
        _require(
            np.issubdtype(raw_nodes.dtype, np.integer) and raw_nodes.dtype.kind != "b",
            "node_indices must contain integers",
        )
        nodes = np.asarray(raw_nodes, dtype=np.int64)
        _require(
            np.all((nodes >= 0) & (nodes < self.node_count)),
            "node index lies outside the graph",
        )
        _require(
            len(np.unique(nodes)) == len(nodes),
            "node_indices must be unique",
        )
    budget = _integer(
        maximum_covariance_bytes,
        name="maximum_covariance_bytes",
        minimum=1,
    )
    query_dimension = 3 * len(horizons) * len(nodes)
    required_bytes = query_dimension * query_dimension * 8
    if required_bytes > budget:
        raise MemoryError(
            "joint covariance requires "
            f"{required_bytes} bytes but the budget is {budget}"
        )
    transition, process_noise, control = _transition_and_noise(
        self.rank,
        frame_dt_s=self.frame_dt_s,
        velocity_retention=self.velocity_retention,
        process_position_std_m=self.process_position_std_m,
        process_acceleration_std_mps2=self.process_acceleration_std_mps2,
    )
    maximum_horizon = int(horizons[-1])
    acceleration: np.ndarray
    if modal_acceleration_mps2 is None:
        acceleration = np.zeros(
            (maximum_horizon, self.rank, 3),
            dtype=np.float64,
        )
    else:
        raw_acceleration = np.asarray(modal_acceleration_mps2)
        _require(
            raw_acceleration.dtype.kind in {"i", "u", "f"},
            "modal_acceleration_mps2 must be real numeric",
        )
        supplied = np.asarray(raw_acceleration, dtype=np.float64)
        if supplied.shape == (self.rank, 3):
            acceleration = np.repeat(
                supplied[None],
                maximum_horizon,
                axis=0,
            )
        elif supplied.shape == (maximum_horizon, self.rank, 3):
            acceleration = supplied.copy()
        else:
            raise ValueError(
                "modal_acceleration_mps2 must have shape (rank, 3) or "
                "(maximum_horizon, rank, 3)"
            )
        _require(
            np.all(np.isfinite(acceleration)),
            "modal_acceleration_mps2 must be finite",
        )
    state_mean = _state_vector(self.state_mean)
    state_covariance = np.asarray(self.state_covariance, dtype=np.float64)
    means: dict[int, np.ndarray] = {}
    covariances: dict[int, np.ndarray] = {}
    requested = set(map(int, horizons))
    for step in range(1, maximum_horizon + 1):
        state_mean = transition @ state_mean + control @ acceleration[step - 1].reshape(
            -1
        )
        state_covariance = _symmetric(
            transition @ state_covariance @ transition.T + process_noise
        )
        if step in requested:
            means[step] = state_mean.copy()
            covariances[step] = state_covariance.copy()
    powers = [np.eye(len(state_mean), dtype=np.float64)]
    for _ in range(maximum_horizon):
        powers.append(transition @ powers[-1])
    selected_basis = self.graph_basis[nodes]
    position_design = np.kron(selected_basis, np.eye(3, dtype=np.float64))
    query = np.concatenate(
        (
            position_design,
            np.zeros_like(position_design),
        ),
        axis=1,
    )
    mean: np.ndarray = np.empty(
        (len(horizons), len(nodes), 3), dtype=np.float64
    )
    covariance: np.ndarray = np.empty(
        (query_dimension, query_dimension),
        dtype=np.float64,
    )
    node_dimension = 3 * len(nodes)
    for first_position, first_horizon in enumerate(horizons):
        first_step = int(first_horizon)
        mean[first_position] = (query @ means[first_step]).reshape(len(nodes), 3)
        first_slice = slice(
            first_position * node_dimension,
            (first_position + 1) * node_dimension,
        )
        for second_position in range(first_position, len(horizons)):
            second_step = int(horizons[second_position])
            second_slice = slice(
                second_position * node_dimension,
                (second_position + 1) * node_dimension,
            )
            cross_state = covariances[first_step] @ powers[second_step - first_step].T
            block = query @ cross_state @ query.T
            covariance[first_slice, second_slice] = block
            covariance[second_slice, first_slice] = block.T
    covariance = _symmetric(covariance)
    return GraphDynamicDiscrepancyForecastV1(
        horizon_steps=horizons,
        node_indices=nodes,
        mean_m=mean,
        joint_covariance_m2=covariance,
    )


__all__ = ["forecast_graph_dynamic_discrepancy"]
