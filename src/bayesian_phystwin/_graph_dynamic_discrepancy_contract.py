"""Immutable contracts for graph-modal discrepancy dynamics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._graph_dynamic_discrepancy_common import (
    _DEFAULT_MAXIMUM_COVARIANCE_BYTES,
    GRAPH_DYNAMIC_DISCREPANCY_BOUNDARY,
    GRAPH_DYNAMIC_DISCREPANCY_SCHEMA,
    GRAPH_DYNAMIC_DISCREPANCY_VERSION,
    _integer,
    _json_mapping,
    _readonly,
    _real,
    _require,
    _validate_graph_basis,
)


@dataclass(frozen=True, slots=True)
class GraphDynamicDiscrepancyForecastV1:
    """Joint node/horizon Gaussian query from one discrepancy belief."""

    horizon_steps: np.ndarray
    node_indices: np.ndarray
    mean_m: np.ndarray
    joint_covariance_m2: np.ndarray

    def __post_init__(self) -> None:
        horizons = _readonly(
            self.horizon_steps,
            name="horizon_steps",
            dtype=np.int64,
        )
        nodes = _readonly(
            self.node_indices,
            name="node_indices",
            dtype=np.int64,
        )
        mean = _readonly(self.mean_m, name="mean_m")
        covariance = _readonly(
            self.joint_covariance_m2,
            name="joint_covariance_m2",
        )
        _require(
            horizons.ndim == 1 and len(horizons) > 0, "horizon_steps is empty"
        )
        _require(nodes.ndim == 1 and len(nodes) > 0, "node_indices is empty")
        _require(np.all(horizons >= 1), "horizon_steps must be positive")
        _require(
            np.all(np.diff(horizons) > 0),
            "horizon_steps must be strictly increasing",
        )
        _require(np.all(nodes >= 0), "node_indices must be nonnegative")
        _require(
            len(np.unique(nodes)) == len(nodes),
            "node_indices must be unique",
        )
        _require(
            mean.shape == (len(horizons), len(nodes), 3),
            "mean_m shape changed",
        )
        dimension = 3 * len(horizons) * len(nodes)
        _require(
            covariance.shape == (dimension, dimension),
            "joint_covariance_m2 shape changed",
        )
        _require(
            np.allclose(covariance, covariance.T, atol=1e-10, rtol=1e-10),
            "joint_covariance_m2 must be symmetric",
        )
        minimum = float(np.min(np.linalg.eigvalsh(covariance), initial=0.0))
        _require(
            minimum >= -1e-8,
            "joint_covariance_m2 must be positive semidefinite",
        )
        object.__setattr__(self, "horizon_steps", horizons)
        object.__setattr__(self, "node_indices", nodes)
        object.__setattr__(self, "mean_m", mean)
        object.__setattr__(self, "joint_covariance_m2", covariance)

    @property
    def marginal_covariance_m2(self) -> np.ndarray:
        """Return read-only 3-D marginal covariance for each node and horizon."""

        horizon_count, node_count, _ = self.mean_m.shape
        result = np.empty(
            (horizon_count, node_count, 3, 3),
            dtype=np.float64,
        )
        block_size = 3 * node_count
        for horizon in range(horizon_count):
            for node in range(node_count):
                start = horizon * block_size + 3 * node
                result[horizon, node] = self.joint_covariance_m2[
                    start : start + 3,
                    start : start + 3,
                ]
        result.setflags(write=False)
        return result


@dataclass(frozen=True, slots=True)
class GraphDynamicDiscrepancyBeliefV1:
    """Graph-modal discrepancy endpoint and recursive uncertainty."""

    graph_basis: np.ndarray
    state_mean: np.ndarray
    state_covariance: np.ndarray
    frame_dt_s: float
    velocity_retention: float
    process_position_std_m: float
    process_acceleration_std_mps2: float
    last_frame_index: int
    update_accepted: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    update_reasons: tuple[str, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        basis = _validate_graph_basis(self.graph_basis)
        state = _readonly(self.state_mean, name="state_mean")
        covariance = _readonly(self.state_covariance, name="state_covariance")
        raw_accepted = np.asarray(self.update_accepted)
        _require(
            raw_accepted.ndim == 1 and raw_accepted.dtype.kind == "b",
            "update_accepted must be a Boolean vector",
        )
        accepted = _readonly(
            raw_accepted,
            name="update_accepted",
            dtype=bool,
            finite=False,
        )
        rank = basis.shape[1]
        dimension = 6 * rank
        _require(
            state.shape == (2, rank, 3),
            "state_mean must have shape (2, graph_rank, 3)",
        )
        _require(
            covariance.shape == (dimension, dimension),
            "state_covariance shape changed",
        )
        _require(
            np.allclose(covariance, covariance.T, atol=1e-10, rtol=1e-10),
            "state_covariance must be symmetric",
        )
        minimum = float(np.min(np.linalg.eigvalsh(covariance), initial=0.0))
        _require(
            minimum >= -1e-8,
            "state_covariance must be positive semidefinite",
        )
        frame_dt = _real(self.frame_dt_s, name="frame_dt_s")
        _require(frame_dt > 0.0, "frame_dt_s must be positive")
        retention = _real(self.velocity_retention, name="velocity_retention")
        _require(
            0.0 <= retention <= 1.0,
            "velocity_retention must lie in [0, 1]",
        )
        process_position = _real(
            self.process_position_std_m,
            name="process_position_std_m",
        )
        process_acceleration = _real(
            self.process_acceleration_std_mps2,
            name="process_acceleration_std_mps2",
        )
        _require(
            process_position >= 0.0 and process_acceleration >= 0.0,
            "process scales must be nonnegative",
        )
        last_frame = _integer(
            self.last_frame_index,
            name="last_frame_index",
        )
        _require(accepted.ndim == 1, "update_accepted must be a vector")
        _require(
            all(isinstance(reason, str) and reason for reason in self.update_reasons),
            "update_reasons must contain nonempty strings",
        )
        reasons = tuple(self.update_reasons)
        _require(
            len(reasons) == len(accepted),
            "update_reasons must match update_accepted",
        )
        object.__setattr__(self, "graph_basis", basis)
        object.__setattr__(self, "state_mean", state)
        object.__setattr__(self, "state_covariance", covariance)
        object.__setattr__(self, "frame_dt_s", frame_dt)
        object.__setattr__(self, "velocity_retention", retention)
        object.__setattr__(self, "process_position_std_m", process_position)
        object.__setattr__(
            self,
            "process_acceleration_std_mps2",
            process_acceleration,
        )
        object.__setattr__(self, "last_frame_index", last_frame)
        object.__setattr__(self, "update_accepted", accepted)
        object.__setattr__(self, "update_reasons", reasons)
        object.__setattr__(
            self,
            "diagnostics",
            _json_mapping(self.diagnostics, name="diagnostics"),
        )

    @property
    def node_count(self) -> int:
        return self.graph_basis.shape[0]

    @property
    def rank(self) -> int:
        return self.graph_basis.shape[1]

    @property
    def accepted_update_count(self) -> int:
        return int(np.sum(self.update_accepted))

    @property
    def position_coefficients_m(self) -> np.ndarray:
        return self.state_mean[0]

    @property
    def velocity_coefficients_mps(self) -> np.ndarray:
        return self.state_mean[1]

    @property
    def position_field_m(self) -> np.ndarray:
        result = self.graph_basis @ self.position_coefficients_m
        result.setflags(write=False)
        return result

    @property
    def velocity_field_mps(self) -> np.ndarray:
        result = self.graph_basis @ self.velocity_coefficients_mps
        result.setflags(write=False)
        return result

    @classmethod
    def from_last_residual(
        cls,
        residual_m: np.ndarray,
        *,
        frame_dt_s: float = 1.0,
        last_frame_index: int = 0,
    ) -> GraphDynamicDiscrepancyBeliefV1:
        """Embed held-last-residual prediction as an exact deterministic case."""

        residual = _readonly(residual_m, name="residual_m")
        _require(
            residual.ndim == 2 and residual.shape[1] == 3,
            "residual_m must have shape (node, 3)",
        )
        node_count = len(residual)
        basis = np.eye(node_count, dtype=np.float64)
        state: np.ndarray = np.zeros((2, node_count, 3), dtype=np.float64)
        state[0] = residual
        covariance = np.zeros((6 * node_count, 6 * node_count))
        return cls(
            graph_basis=basis,
            state_mean=state,
            state_covariance=covariance,
            frame_dt_s=frame_dt_s,
            velocity_retention=0.0,
            process_position_std_m=0.0,
            process_acceleration_std_mps2=0.0,
            last_frame_index=last_frame_index,
            diagnostics={
                "schema": GRAPH_DYNAMIC_DISCREPANCY_SCHEMA,
                "schema_version": GRAPH_DYNAMIC_DISCREPANCY_VERSION,
                "nested_special_case": "last-residual",
                "scientific_boundary": GRAPH_DYNAMIC_DISCREPANCY_BOUNDARY,
            },
        )

    @classmethod
    def from_independent_endpoint_posterior(
        cls,
        mean_m: np.ndarray,
        variance_m2: np.ndarray,
        *,
        process_std_m: float = 0.0,
        frame_dt_s: float = 1.0,
        last_frame_index: int = 0,
    ) -> GraphDynamicDiscrepancyBeliefV1:
        """Embed the current independent random-walk endpoint posterior exactly."""

        mean = _readonly(mean_m, name="mean_m")
        _require(
            mean.ndim == 2 and mean.shape[1] == 3,
            "mean_m must have shape (node, 3)",
        )
        node_count = len(mean)
        raw_variance_input = np.asarray(variance_m2)
        _require(
            raw_variance_input.dtype.kind in {"i", "u", "f"},
            "variance_m2 must be real numeric",
        )
        raw_variance = np.asarray(raw_variance_input, dtype=np.float64)
        if raw_variance.shape == (node_count,):
            variance = np.repeat(raw_variance[:, None], 3, axis=1)
        elif raw_variance.shape == (node_count, 3):
            variance = raw_variance.copy()
        else:
            raise ValueError("variance_m2 must have shape (node,) or (node, 3)")
        _require(
            np.all(np.isfinite(variance)) and np.all(variance >= 0.0),
            "variance_m2 must be finite and nonnegative",
        )
        process_std = _real(process_std_m, name="process_std_m")
        _require(process_std >= 0.0, "process_std_m must be nonnegative")
        basis = np.eye(node_count, dtype=np.float64)
        state: np.ndarray = np.zeros((2, node_count, 3), dtype=np.float64)
        state[0] = mean
        covariance = np.zeros((6 * node_count, 6 * node_count))
        covariance[: 3 * node_count, : 3 * node_count] = np.diag(variance.reshape(-1))
        return cls(
            graph_basis=basis,
            state_mean=state,
            state_covariance=covariance,
            frame_dt_s=frame_dt_s,
            velocity_retention=0.0,
            process_position_std_m=process_std,
            process_acceleration_std_mps2=0.0,
            last_frame_index=last_frame_index,
            diagnostics={
                "schema": GRAPH_DYNAMIC_DISCREPANCY_SCHEMA,
                "schema_version": GRAPH_DYNAMIC_DISCREPANCY_VERSION,
                "nested_special_case": "independent-random-walk-endpoint",
                "scientific_boundary": GRAPH_DYNAMIC_DISCREPANCY_BOUNDARY,
            },
        )

    def forecast(
        self,
        horizon_steps: Sequence[int] | np.ndarray,
        *,
        node_indices: Sequence[int] | np.ndarray | None = None,
        modal_acceleration_mps2: np.ndarray | None = None,
        maximum_covariance_bytes: int = _DEFAULT_MAXIMUM_COVARIANCE_BYTES,
    ) -> GraphDynamicDiscrepancyForecastV1:
        """Forecast a registered node/horizon query with full joint covariance."""

        from ._graph_dynamic_discrepancy_forecast import (
            forecast_graph_dynamic_discrepancy,
        )

        return forecast_graph_dynamic_discrepancy(
            self,
            horizon_steps,
            node_indices=node_indices,
            modal_acceleration_mps2=modal_acceleration_mps2,
            maximum_covariance_bytes=maximum_covariance_bytes,
        )


__all__ = [
    "GraphDynamicDiscrepancyBeliefV1",
    "GraphDynamicDiscrepancyForecastV1",
]
