"""Adapt portable observation beliefs to gauge-aware state updates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gauge_aware_belief import GaugeAwareObservationBatch
from .observation_belief import ObservationBeliefV1


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _readonly(
    values: np.ndarray,
    *,
    dtype: np.dtype | type | None = None,
) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def global_translation_bias_jacobian(
    observation_count: int,
) -> np.ndarray:
    """Return a shared three-axis translation nuisance design."""

    _require(observation_count >= 1, "observation_count must be positive")
    return np.repeat(
        np.eye(3, dtype=np.float64)[None],
        observation_count,
        axis=0,
    )


def _helmert_view_contrasts(view_count: int) -> np.ndarray:
    """Return orthonormal zero-sum contrasts between views."""

    _require(view_count >= 1, "view_count must be positive")
    contrasts = np.zeros((view_count, max(view_count - 1, 0)))
    for column in range(view_count - 1):
        denominator = np.sqrt((column + 1) * (column + 2))
        contrasts[: column + 1, column] = 1.0 / denominator
        contrasts[column + 1, column] = -(column + 1) / denominator
    return contrasts


def centered_view_translation_bias_jacobian(
    view_indices: np.ndarray,
    *,
    view_count: int,
) -> np.ndarray:
    """Return per-view translation contrasts without duplicating the mean."""

    views = np.asarray(view_indices, dtype=np.int64)
    _require(views.ndim == 1 and len(views), "view_indices must be nonempty")
    _require(
        np.all((views >= 0) & (views < view_count)),
        "view index exceeds view_count",
    )
    contrasts = _helmert_view_contrasts(view_count)
    result = np.zeros((len(views), 3, 3 * contrasts.shape[1]))
    identity = np.eye(3, dtype=np.float64)
    for row, view in enumerate(views):
        for contrast in range(contrasts.shape[1]):
            start = 3 * contrast
            result[row, :, start : start + 3] = (
                contrasts[view, contrast] * identity
            )
    return result


def _expanded_gauge_design(
    belief: ObservationBeliefV1,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Expand shared factor groups into independent nuisance columns."""

    rank = belief.factor_rank
    if rank == 0:
        return (
            np.zeros((belief.observation_count, 3, 0)),
            np.zeros(0, dtype=np.int64),
            (),
        )
    factor_groups = np.unique(belief.factor_group_ids)
    parameter_count = len(factor_groups) * rank
    design = np.zeros((belief.observation_count, 3, parameter_count))
    parameter_groups = np.repeat(factor_groups, rank)
    names: list[str] = []
    for group_position, group_id in enumerate(factor_groups):
        selected = belief.factor_group_ids == group_id
        start = group_position * rank
        design[selected, :, start : start + rank] = (
            belief.low_rank_factor_m[selected]
        )
        names.extend(
            f"factor-group-{int(group_id)}:{factor_name}"
            for factor_name in belief.factor_names
        )
    return design, parameter_groups, tuple(names)


def _row_group_values(
    belief: ObservationBeliefV1,
    values: np.ndarray,
) -> np.ndarray:
    positions = {
        int(group_id): position
        for position, group_id in enumerate(belief.group_ids)
    }
    return np.asarray(
        [
            values[positions[int(group_id)]]
            for group_id in belief.correlation_group_ids
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class ObservationBeliefGaugeAdapterResult:
    """Gauge-aware batch plus provenance not used as state reliability."""

    batch: GaugeAwareObservationBatch
    observation_artifact_id: str
    gauge_parameter_names: tuple[str, ...]
    gauge_parameter_group_ids: np.ndarray
    association_probability: np.ndarray

    def __post_init__(self) -> None:
        _require(
            self.observation_artifact_id == self.batch.metadata.get(
                "observation_artifact_id"
            ),
            "observation artifact provenance changed",
        )
        groups = _readonly(self.gauge_parameter_group_ids, dtype=np.int64)
        association = _readonly(
            self.association_probability,
            dtype=np.float64,
        )
        _require(
            groups.shape == (len(self.gauge_parameter_names),),
            "gauge parameter group IDs changed shape",
        )
        _require(
            association.shape == (len(self.batch.innovation_m),),
            "association probability changed shape",
        )
        _require(
            np.all(np.isfinite(association))
            and np.all((association >= 0.0) & (association <= 1.0)),
            "association probability must lie in [0, 1]",
        )
        object.__setattr__(self, "gauge_parameter_group_ids", groups)
        object.__setattr__(self, "association_probability", association)

    def summary(self) -> dict[str, object]:
        return {
            "observation_artifact_id": self.observation_artifact_id,
            "observation_count": len(self.batch.innovation_m),
            "state_mode_count": self.batch.state_jacobian.shape[2],
            "gauge_parameter_count": len(self.gauge_parameter_names),
            "shared_bias_parameter_count": (
                self.batch.shared_bias_jacobian.shape[2]
            ),
            "view_bias_parameter_count": (
                self.batch.view_bias_jacobian.shape[2]
            ),
            "association_used_as_prior_reliability": False,
            "low_rank_covariance_treatment": (
                "explicit standard-normal nuisance parameters"
            ),
        }


def build_gauge_aware_batch_from_observation_belief(
    belief: ObservationBeliefV1,
    *,
    physical_prediction_xyz_m: np.ndarray,
    state_jacobian: np.ndarray,
    query_state_jacobian: np.ndarray,
    physical_response_scale_m: float,
    shared_bias_jacobian: np.ndarray | None = None,
    view_bias_jacobian: np.ndarray | None = None,
    state_prior_covariance_m2: np.ndarray | None = None,
    anchor_innovation_m: np.ndarray | None = None,
    anchor_covariance_m2: np.ndarray | None = None,
    anchor_state_jacobian: np.ndarray | None = None,
) -> ObservationBeliefGaugeAdapterResult:
    """Build one residual-independent, covariance-safe gauge-aware batch.

    The observation innovation is formed once here. Local covariance remains
    conditional covariance. Shared low-rank covariance factors become explicit
    standard-normal nuisance parameters and are therefore not added to the
    local covariance again.

    Association probability is retained only as a diagnostic. Row reliability,
    group nominal probability, and composite-likelihood weight remain distinct
    residual-independent inputs to the downstream solver.
    """

    predicted = np.asarray(physical_prediction_xyz_m, dtype=np.float64)
    _require(
        predicted.shape == belief.mean_xyz_m.shape,
        "physical prediction must match the observation belief",
    )
    _require(
        np.all(np.isfinite(predicted)),
        "physical prediction contains non-finite values",
    )
    state = np.asarray(state_jacobian, dtype=np.float64)
    query = np.asarray(query_state_jacobian, dtype=np.float64)
    _require(
        state.ndim == 3
        and state.shape[:2] == (belief.observation_count, 3)
        and state.shape[2] >= 1,
        "state_jacobian must have shape (N, 3, S) with S >= 1",
    )
    _require(
        query.ndim == 3
        and query.shape[1:] == (3, state.shape[2])
        and len(query),
        "query_state_jacobian must have shape (Q, 3, S)",
    )
    _require(
        np.all(np.isfinite(state)) and np.all(np.isfinite(query)),
        "state or query Jacobian contains non-finite values",
    )

    gauge, gauge_groups, gauge_names = _expanded_gauge_design(belief)
    shared = (
        global_translation_bias_jacobian(belief.observation_count)
        if shared_bias_jacobian is None
        else np.asarray(shared_bias_jacobian, dtype=np.float64)
    )
    view = (
        centered_view_translation_bias_jacobian(
            belief.view_indices,
            view_count=len(belief.view_names),
        )
        if view_bias_jacobian is None
        else np.asarray(view_bias_jacobian, dtype=np.float64)
    )
    row_nominal_probability = _row_group_values(
        belief,
        belief.group_prior_nominal_probability,
    )
    row_composite_weight = _row_group_values(
        belief,
        belief.group_composite_weight,
    )
    correlation_groups = tuple(
        f"{belief.stream_id}:correlation-group-{int(group_id)}"
        for group_id in belief.correlation_group_ids
    )
    metadata = {
        "observation_artifact_id": belief.artifact_id,
        "observation_source_repository": belief.source_repository,
        "observation_source_revision": belief.source_revision,
        "association_used_as_prior_reliability": False,
        "innovation_formed_once": True,
        "low_rank_covariance_double_counted": False,
    }
    batch = GaugeAwareObservationBatch(
        innovation_m=belief.mean_xyz_m - predicted,
        observation_covariance_m2=belief.local_covariance_m2,
        state_jacobian=state,
        gauge_jacobian=gauge,
        shared_bias_jacobian=shared,
        view_bias_jacobian=view,
        query_state_jacobian=query,
        gauge_prior_covariance=np.eye(gauge.shape[2]),
        correlation_group_ids=correlation_groups,
        prior_reliability=belief.prior_reliability,
        prior_nominal_probability=row_nominal_probability,
        composite_weight=row_composite_weight,
        physical_response_scale_m=physical_response_scale_m,
        state_prior_covariance_m2=state_prior_covariance_m2,
        anchor_innovation_m=anchor_innovation_m,
        anchor_covariance_m2=anchor_covariance_m2,
        anchor_state_jacobian=anchor_state_jacobian,
        metadata=metadata,
    )
    return ObservationBeliefGaugeAdapterResult(
        batch=batch,
        observation_artifact_id=belief.artifact_id,
        gauge_parameter_names=gauge_names,
        gauge_parameter_group_ids=gauge_groups,
        association_probability=belief.association_probability,
    )


__all__ = [
    "ObservationBeliefGaugeAdapterResult",
    "build_gauge_aware_batch_from_observation_belief",
    "centered_view_translation_bias_jacobian",
    "global_translation_bias_jacobian",
]
