"""Content-addressed pre-outcome graph tournament contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    literal_lower_hex,
    plain_json,
)
from ._graph_dynamic_discrepancy_contract import GraphDynamicDiscrepancyForecastV1
from ._graph_dynamic_tournament_common import (
    GRAPH_DYNAMIC_TOURNAMENT_BOUNDARY,
    GRAPH_DYNAMIC_TOURNAMENT_BUNDLE_SCHEMA,
    GRAPH_DYNAMIC_TOURNAMENT_BUNDLE_VERSION,
    GRAPH_DYNAMIC_TOURNAMENT_FAMILY,
    GRAPH_DYNAMIC_TOURNAMENT_PREDICTION_SCHEMA,
    GRAPH_DYNAMIC_TOURNAMENT_PREDICTION_VERSION,
    GRAPH_DYNAMIC_TOURNAMENT_SCORING_POLICY_SCHEMA,
    GRAPH_DYNAMIC_TOURNAMENT_SCORING_POLICY_VERSION,
    FloatArray,
    IntArray,
    array_record,
    canonical_string,
    finite_real,
    float_array,
    genuine_boolean,
    genuine_integer,
    identifier,
    integer_vector,
    validated_covariance,
)
from ._portable_contracts import content_id
from .discrepancy_candidate_tournament import CandidateSpec


@dataclass(frozen=True, slots=True)
class GraphDynamicTournamentScoringPolicyV1:
    """Frozen point, proper-score, and optional interval semantics."""

    covariance_floor_m2: float = 1e-12
    nominal_interval_coverage: float | None = 0.9
    marginal_standard_score: float | None = 1.6448536269514722
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        floor = finite_real(
            self.covariance_floor_m2,
            name="covariance_floor_m2",
            minimum=0.0,
        )
        if floor <= 0.0:
            raise ValueError("covariance_floor_m2 must be positive")
        coverage = self.nominal_interval_coverage
        standard = self.marginal_standard_score
        if coverage is None and standard is None:
            normalized_coverage = None
            normalized_standard = None
        elif coverage is not None and standard is not None:
            normalized_coverage = finite_real(
                coverage,
                name="nominal_interval_coverage",
                minimum=0.0,
                maximum=1.0,
            )
            if normalized_coverage in {0.0, 1.0}:
                raise ValueError(
                    "nominal_interval_coverage must lie strictly inside (0, 1)"
                )
            normalized_standard = finite_real(
                standard,
                name="marginal_standard_score",
                minimum=0.0,
            )
            if normalized_standard <= 0.0:
                raise ValueError("marginal_standard_score must be positive")
        else:
            raise ValueError(
                "nominal_interval_coverage and marginal_standard_score "
                "must both be present or absent"
            )
        object.__setattr__(self, "covariance_floor_m2", floor)
        object.__setattr__(
            self,
            "nominal_interval_coverage",
            normalized_coverage,
        )
        object.__setattr__(
            self,
            "marginal_standard_score",
            normalized_standard,
        )
        expected_id = content_id(self.descriptor())
        if self.artifact_id is not None:
            supplied = literal_lower_hex(
                self.artifact_id,
                name="artifact_id",
                lengths={64},
            )
            if supplied != expected_id:
                raise ValueError("artifact_id does not match scoring policy content")
        object.__setattr__(self, "artifact_id", expected_id)

    @property
    def point_loss_id(self) -> str:
        return "endpoint-coordinate-rmse-v1"

    @property
    def proper_score_id(self) -> str:
        return "joint-gaussian-nll-per-coordinate-v1"

    @property
    def interval_semantics_id(self) -> str:
        if self.nominal_interval_coverage is None:
            return "interval-disabled-v1"
        percentage = int(round(100.0 * self.nominal_interval_coverage))
        return f"coordinatewise-marginal-{percentage}-v1"

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": GRAPH_DYNAMIC_TOURNAMENT_SCORING_POLICY_SCHEMA,
            "schema_version": GRAPH_DYNAMIC_TOURNAMENT_SCORING_POLICY_VERSION,
            "point_loss_id": self.point_loss_id,
            "proper_score_id": self.proper_score_id,
            "interval_semantics_id": self.interval_semantics_id,
            "covariance_floor_m2": self.covariance_floor_m2,
            "nominal_interval_coverage": self.nominal_interval_coverage,
            "marginal_standard_score": self.marginal_standard_score,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}


@dataclass(frozen=True, slots=True)
class GraphDynamicTournamentPredictionV1:
    """One graph forecast sealed before its scored target is available."""

    candidate_id: str
    unit_id: str
    group_id: str
    horizon: str
    source_revision: str
    configuration_sha256: str
    prediction_barrier_sha256: str
    source_horizon_steps: IntArray
    node_indices: IntArray
    source_mean_m: FloatArray
    source_joint_covariance_m2: FloatArray
    selected_horizon_index: int
    physical_fallback_mean_m: FloatArray
    physical_fallback_covariance_m2: FloatArray
    graph_rank: int
    parameter_count: int
    runtime_milliseconds: float
    accepted: bool
    reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        candidate_id = identifier(self.candidate_id, name="candidate_id")
        unit_id = canonical_string(self.unit_id, name="unit_id")
        group_id = canonical_string(self.group_id, name="group_id")
        horizon = canonical_string(self.horizon, name="horizon")
        source_revision = literal_lower_hex(
            self.source_revision,
            name="source_revision",
            lengths={40},
        )
        configuration = literal_lower_hex(
            self.configuration_sha256,
            name="configuration_sha256",
            lengths={64},
        )
        barrier = literal_lower_hex(
            self.prediction_barrier_sha256,
            name="prediction_barrier_sha256",
            lengths={64},
        )
        horizon_steps = integer_vector(
            self.source_horizon_steps,
            name="source_horizon_steps",
        )
        if not len(horizon_steps) or np.any(horizon_steps < 1):
            raise ValueError("source_horizon_steps must be positive")
        if np.any(np.diff(horizon_steps) <= 0):
            raise ValueError("source_horizon_steps must be strictly increasing")
        nodes = integer_vector(self.node_indices, name="node_indices")
        if not len(nodes) or np.any(nodes < 0):
            raise ValueError("node_indices must be nonempty and nonnegative")
        if len(np.unique(nodes)) != len(nodes):
            raise ValueError("node_indices must be unique")
        mean = float_array(self.source_mean_m, name="source_mean_m")
        expected_mean_shape = (len(horizon_steps), len(nodes), 3)
        if mean.shape != expected_mean_shape:
            raise ValueError("source_mean_m shape changed")
        complete_dimension = 3 * len(horizon_steps) * len(nodes)
        covariance = validated_covariance(
            self.source_joint_covariance_m2,
            name="source_joint_covariance_m2",
            dimension=complete_dimension,
        )
        selected = genuine_integer(
            self.selected_horizon_index,
            name="selected_horizon_index",
        )
        if selected >= len(horizon_steps):
            raise ValueError("selected_horizon_index lies outside the forecast")
        query_dimension = 3 * len(nodes)
        fallback_mean = float_array(
            self.physical_fallback_mean_m,
            name="physical_fallback_mean_m",
        )
        if fallback_mean.shape != (len(nodes), 3):
            raise ValueError("physical_fallback_mean_m shape changed")
        fallback_covariance = validated_covariance(
            self.physical_fallback_covariance_m2,
            name="physical_fallback_covariance_m2",
            dimension=query_dimension,
        )
        rank = genuine_integer(self.graph_rank, name="graph_rank", minimum=1)
        parameter_count = genuine_integer(
            self.parameter_count,
            name="parameter_count",
        )
        runtime = finite_real(
            self.runtime_milliseconds,
            name="runtime_milliseconds",
            minimum=0.0,
        )
        accepted = genuine_boolean(self.accepted, name="accepted")
        reason = canonical_string(self.reason, name="reason")
        if accepted and reason != "prediction-admissible":
            raise ValueError(
                "an accepted prediction reason must be prediction-admissible"
            )
        if not accepted and reason == "prediction-admissible":
            raise ValueError("a rejected prediction must retain a rejection reason")
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="graph tournament prediction metadata",
        )

        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "unit_id", unit_id)
        object.__setattr__(self, "group_id", group_id)
        object.__setattr__(self, "horizon", horizon)
        object.__setattr__(self, "source_revision", source_revision)
        object.__setattr__(self, "configuration_sha256", configuration)
        object.__setattr__(self, "prediction_barrier_sha256", barrier)
        object.__setattr__(self, "source_horizon_steps", horizon_steps)
        object.__setattr__(self, "node_indices", nodes)
        object.__setattr__(self, "source_mean_m", mean)
        object.__setattr__(self, "source_joint_covariance_m2", covariance)
        object.__setattr__(self, "selected_horizon_index", selected)
        object.__setattr__(self, "physical_fallback_mean_m", fallback_mean)
        object.__setattr__(
            self,
            "physical_fallback_covariance_m2",
            fallback_covariance,
        )
        object.__setattr__(self, "graph_rank", rank)
        object.__setattr__(self, "parameter_count", parameter_count)
        object.__setattr__(self, "runtime_milliseconds", runtime)
        object.__setattr__(self, "accepted", accepted)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "metadata", metadata)

        expected_id = content_id(self.descriptor())
        if self.artifact_id is not None:
            supplied = literal_lower_hex(
                self.artifact_id,
                name="artifact_id",
                lengths={64},
            )
            if supplied != expected_id:
                raise ValueError("artifact_id does not match prediction content")
        object.__setattr__(self, "artifact_id", expected_id)

    @property
    def horizon_step(self) -> int:
        return int(self.source_horizon_steps[self.selected_horizon_index])

    @property
    def mean_m(self) -> FloatArray:
        return cast(FloatArray, self.source_mean_m[self.selected_horizon_index])

    @property
    def covariance_m2(self) -> FloatArray:
        block = 3 * len(self.node_indices)
        start = block * self.selected_horizon_index
        return cast(
            FloatArray,
            self.source_joint_covariance_m2[
                start : start + block,
                start : start + block,
            ],
        )

    @property
    def state_dimension(self) -> int:
        return 6 * self.graph_rank

    @property
    def covariance_bytes(self) -> int:
        return int(self.source_joint_covariance_m2.nbytes)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": GRAPH_DYNAMIC_TOURNAMENT_PREDICTION_SCHEMA,
            "schema_version": GRAPH_DYNAMIC_TOURNAMENT_PREDICTION_VERSION,
            "candidate_id": self.candidate_id,
            "family": GRAPH_DYNAMIC_TOURNAMENT_FAMILY,
            "unit_id": self.unit_id,
            "group_id": self.group_id,
            "horizon": self.horizon,
            "source_revision": self.source_revision,
            "configuration_sha256": self.configuration_sha256,
            "prediction_barrier_sha256": self.prediction_barrier_sha256,
            "source_horizon_steps": array_record(self.source_horizon_steps),
            "node_indices": array_record(self.node_indices),
            "source_mean_m": array_record(self.source_mean_m),
            "source_joint_covariance_m2": array_record(self.source_joint_covariance_m2),
            "selected_horizon_index": self.selected_horizon_index,
            "horizon_step": self.horizon_step,
            "physical_fallback_mean_m": array_record(self.physical_fallback_mean_m),
            "physical_fallback_covariance_m2": array_record(
                self.physical_fallback_covariance_m2
            ),
            "graph_rank": self.graph_rank,
            "state_dimension": self.state_dimension,
            "parameter_count": self.parameter_count,
            "runtime_milliseconds": self.runtime_milliseconds,
            "covariance_bytes": self.covariance_bytes,
            "accepted": self.accepted,
            "reason": self.reason,
            "prediction_sealed_before_scoring": True,
            "scientific_boundary": GRAPH_DYNAMIC_TOURNAMENT_BOUNDARY,
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}


def seal_graph_dynamic_tournament_prediction(
    forecast: GraphDynamicDiscrepancyForecastV1,
    *,
    selected_horizon_index: int,
    candidate_id: str,
    unit_id: str,
    group_id: str,
    horizon: str,
    source_revision: str,
    configuration_sha256: str,
    prediction_barrier_sha256: str,
    physical_fallback_mean_m: object,
    physical_fallback_covariance_m2: object,
    graph_rank: int,
    parameter_count: int,
    runtime_milliseconds: float,
    accepted: bool,
    reason: str,
    metadata: Mapping[str, Any] | None = None,
) -> GraphDynamicTournamentPredictionV1:
    """Seal a graph forecast without accepting any scored target."""

    if not isinstance(forecast, GraphDynamicDiscrepancyForecastV1):
        raise TypeError("forecast must be a GraphDynamicDiscrepancyForecastV1")
    return GraphDynamicTournamentPredictionV1(
        candidate_id=candidate_id,
        unit_id=unit_id,
        group_id=group_id,
        horizon=horizon,
        source_revision=source_revision,
        configuration_sha256=configuration_sha256,
        prediction_barrier_sha256=prediction_barrier_sha256,
        source_horizon_steps=forecast.horizon_steps,
        node_indices=forecast.node_indices,
        source_mean_m=forecast.mean_m,
        source_joint_covariance_m2=forecast.joint_covariance_m2,
        selected_horizon_index=selected_horizon_index,
        physical_fallback_mean_m=physical_fallback_mean_m,
        physical_fallback_covariance_m2=physical_fallback_covariance_m2,
        graph_rank=graph_rank,
        parameter_count=parameter_count,
        runtime_milliseconds=runtime_milliseconds,
        accepted=accepted,
        reason=reason,
        metadata={} if metadata is None else metadata,
    )


@dataclass(frozen=True, slots=True)
class GraphDynamicTournamentPredictionBundleV1:
    """Complete graph candidate roster sealed before source outcomes are scored."""

    predictions: Sequence[GraphDynamicTournamentPredictionV1]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None
    _candidate: CandidateSpec = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.predictions, (str, bytes)) or not isinstance(
            self.predictions,
            Sequence,
        ):
            raise ValueError("predictions must be a sequence")
        predictions = tuple(self.predictions)
        if not predictions or any(
            not isinstance(item, GraphDynamicTournamentPredictionV1)
            for item in predictions
        ):
            raise ValueError(
                "predictions must contain GraphDynamicTournamentPredictionV1 values"
            )
        unit_ids = [item.unit_id for item in predictions]
        if len(set(unit_ids)) != len(unit_ids):
            raise ValueError("prediction unit_id values must be unique")
        shared = {
            (
                item.candidate_id,
                item.source_revision,
                item.configuration_sha256,
                item.prediction_barrier_sha256,
                item.parameter_count,
            )
            for item in predictions
        }
        if len(shared) != 1:
            raise ValueError(
                "bundle predictions changed candidate or registered configuration"
            )
        ordered = tuple(
            sorted(
                predictions,
                key=lambda item: (item.group_id, item.unit_id, item.horizon),
            )
        )
        first = ordered[0]
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="graph tournament bundle metadata",
        )
        object.__setattr__(self, "predictions", ordered)
        object.__setattr__(self, "metadata", metadata)

        expected_id = content_id(self.descriptor())
        if self.artifact_id is not None:
            supplied = literal_lower_hex(
                self.artifact_id,
                name="artifact_id",
                lengths={64},
            )
            if supplied != expected_id:
                raise ValueError("artifact_id does not match prediction bundle content")
        object.__setattr__(self, "artifact_id", expected_id)
        candidate = CandidateSpec(
            candidate_id=first.candidate_id,
            family=GRAPH_DYNAMIC_TOURNAMENT_FAMILY,
            state_dimension=max(item.state_dimension for item in ordered),
            parameter_count=first.parameter_count,
            runtime_milliseconds=float(
                sum(item.runtime_milliseconds for item in ordered)
            ),
            covariance_bytes=sum(item.covariance_bytes for item in ordered),
            source_revision=first.source_revision,
            configuration_sha256=first.configuration_sha256,
            prediction_artifact_sha256=expected_id,
        )
        object.__setattr__(self, "_candidate", candidate)

    @property
    def candidate_id(self) -> str:
        return self.predictions[0].candidate_id

    @property
    def source_revision(self) -> str:
        return self.predictions[0].source_revision

    @property
    def configuration_sha256(self) -> str:
        return self.predictions[0].configuration_sha256

    @property
    def prediction_barrier_sha256(self) -> str:
        return self.predictions[0].prediction_barrier_sha256

    @property
    def candidate(self) -> CandidateSpec:
        return self._candidate

    @property
    def physical_fallback_artifact_sha256(self) -> str:
        return content_id(
            {
                "schema": "bayesian_phystwin.graph_dynamic_physical_fallback_bundle",
                "schema_version": 1,
                "units": [
                    {
                        "unit_id": item.unit_id,
                        "group_id": item.group_id,
                        "horizon": item.horizon,
                        "mean_m": array_record(item.physical_fallback_mean_m),
                        "covariance_m2": array_record(
                            item.physical_fallback_covariance_m2
                        ),
                    }
                    for item in self.predictions
                ],
            }
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": GRAPH_DYNAMIC_TOURNAMENT_BUNDLE_SCHEMA,
            "schema_version": GRAPH_DYNAMIC_TOURNAMENT_BUNDLE_VERSION,
            "candidate_id": self.predictions[0].candidate_id,
            "family": GRAPH_DYNAMIC_TOURNAMENT_FAMILY,
            "source_revision": self.predictions[0].source_revision,
            "configuration_sha256": self.predictions[0].configuration_sha256,
            "prediction_barrier_sha256": self.predictions[0].prediction_barrier_sha256,
            "unit_count": len(self.predictions),
            "prediction_artifact_ids": [item.artifact_id for item in self.predictions],
            "physical_fallback_artifact_sha256": (
                self.physical_fallback_artifact_sha256
            ),
            "maximum_state_dimension": max(
                item.state_dimension for item in self.predictions
            ),
            "parameter_count": self.predictions[0].parameter_count,
            "total_runtime_milliseconds": float(
                sum(item.runtime_milliseconds for item in self.predictions)
            ),
            "total_covariance_bytes": sum(
                item.covariance_bytes for item in self.predictions
            ),
            "prediction_sealed_before_scoring": True,
            "scientific_boundary": GRAPH_DYNAMIC_TOURNAMENT_BOUNDARY,
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {
            **self.descriptor(),
            "predictions": [item.to_record() for item in self.predictions],
            "artifact_id": self.artifact_id,
        }


def build_graph_dynamic_tournament_prediction_bundle(
    predictions: Sequence[GraphDynamicTournamentPredictionV1],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> GraphDynamicTournamentPredictionBundleV1:
    """Bind the complete graph candidate roster into one prediction artifact."""

    return GraphDynamicTournamentPredictionBundleV1(
        predictions=predictions,
        metadata={} if metadata is None else metadata,
    )


__all__ = [
    "GraphDynamicTournamentPredictionBundleV1",
    "GraphDynamicTournamentPredictionV1",
    "GraphDynamicTournamentScoringPolicyV1",
    "build_graph_dynamic_tournament_prediction_bundle",
    "seal_graph_dynamic_tournament_prediction",
]
