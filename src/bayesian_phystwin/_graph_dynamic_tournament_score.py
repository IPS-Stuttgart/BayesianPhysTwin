"""Post-outcome scoring for sealed graph-modal tournament predictions."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

import numpy as np

from ._canonical_contracts import literal_lower_hex
from ._graph_dynamic_tournament_common import (
    GRAPH_DYNAMIC_TOURNAMENT_BOUNDARY,
    GRAPH_DYNAMIC_TOURNAMENT_SCORE_SCHEMA,
    GRAPH_DYNAMIC_TOURNAMENT_SCORE_VERSION,
    FloatArray,
    array_record,
    float_array,
    identifier,
)
from ._graph_dynamic_tournament_contract import (
    GraphDynamicTournamentPredictionBundleV1,
    GraphDynamicTournamentPredictionV1,
    GraphDynamicTournamentScoringPolicyV1,
)
from ._portable_contracts import content_id
from .discrepancy_candidate_tournament import CandidateSpec, TournamentRecord


def candidate_spec_dict(value: CandidateSpec) -> dict[str, object]:
    """Return the exact candidate JSON shape consumed by the tournament."""

    return {
        "candidate_id": value.candidate_id,
        "family": value.family,
        "state_dimension": value.state_dimension,
        "parameter_count": value.parameter_count,
        "runtime_milliseconds": value.runtime_milliseconds,
        "covariance_bytes": value.covariance_bytes,
        "source_revision": value.source_revision,
        "configuration_sha256": value.configuration_sha256,
        "prediction_artifact_sha256": value.prediction_artifact_sha256,
    }


def tournament_record_dict(value: TournamentRecord) -> dict[str, object]:
    """Return the exact record JSON shape consumed by the tournament."""

    return {
        "candidate_id": value.candidate_id,
        "unit_id": value.unit_id,
        "group_id": value.group_id,
        "horizon": value.horizon,
        "accepted": value.accepted,
        "point_loss": value.point_loss,
        "fallback_point_loss": value.fallback_point_loss,
        "deployed_point_loss": value.deployed_point_loss,
        "proper_score": value.proper_score,
        "fallback_proper_score": value.fallback_proper_score,
        "deployed_proper_score": value.deployed_proper_score,
        "interval_covered": value.interval_covered,
        "interval_width": value.interval_width,
    }


def _regularized_score(
    mean_m: FloatArray,
    covariance_m2: FloatArray,
    target_m: FloatArray,
    policy: GraphDynamicTournamentScoringPolicyV1,
) -> tuple[float, float, bool | None, float | None]:
    error = np.asarray(target_m - mean_m, dtype=np.float64).reshape(-1)
    covariance = np.asarray(covariance_m2, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    clipped = np.maximum(eigenvalues, policy.covariance_floor_m2)
    projected = eigenvectors.T @ error
    mahalanobis = float(np.sum(np.square(projected) / clipped))
    log_determinant = float(np.sum(np.log(clipped)))
    dimension = len(error)
    point_loss = float(np.sqrt(np.mean(np.square(error))))
    proper_score = (
        0.5
        * (dimension * math.log(2.0 * math.pi) + log_determinant + mahalanobis)
        / dimension
    )
    if not np.isfinite(point_loss) or not np.isfinite(proper_score):
        raise ValueError("tournament score is not finite")
    if policy.nominal_interval_coverage is None:
        return point_loss, proper_score, None, None
    standard = cast(float, policy.marginal_standard_score)
    diagonal = np.sum(np.square(eigenvectors) * clipped[None, :], axis=1)
    half_width = standard * np.sqrt(diagonal)
    covered = bool(np.all(np.abs(error) <= half_width))
    width = float(np.mean(2.0 * half_width))
    if not np.isfinite(width):
        raise ValueError("tournament interval width is not finite")
    return point_loss, proper_score, covered, width


def _matched_records(
    prediction: GraphDynamicTournamentPredictionV1,
    target: FloatArray,
    policy: GraphDynamicTournamentScoringPolicyV1,
    *,
    physical_fallback_candidate_id: str,
) -> tuple[TournamentRecord, TournamentRecord]:
    (
        candidate_point,
        candidate_proper,
        candidate_covered,
        candidate_width,
    ) = _regularized_score(
        prediction.mean_m,
        prediction.covariance_m2,
        target,
        policy,
    )
    (
        fallback_point,
        fallback_proper,
        fallback_covered,
        fallback_width,
    ) = _regularized_score(
        prediction.physical_fallback_mean_m,
        prediction.physical_fallback_covariance_m2,
        target,
        policy,
    )
    accepted = prediction.accepted
    candidate_record = TournamentRecord(
        candidate_id=prediction.candidate_id,
        unit_id=prediction.unit_id,
        group_id=prediction.group_id,
        horizon=prediction.horizon,
        accepted=accepted,
        point_loss=candidate_point,
        fallback_point_loss=fallback_point,
        deployed_point_loss=candidate_point if accepted else fallback_point,
        proper_score=candidate_proper,
        fallback_proper_score=fallback_proper,
        deployed_proper_score=candidate_proper if accepted else fallback_proper,
        interval_covered=candidate_covered if accepted else fallback_covered,
        interval_width=candidate_width if accepted else fallback_width,
    )
    fallback_record = TournamentRecord(
        candidate_id=physical_fallback_candidate_id,
        unit_id=prediction.unit_id,
        group_id=prediction.group_id,
        horizon=prediction.horizon,
        accepted=False,
        point_loss=fallback_point,
        fallback_point_loss=fallback_point,
        deployed_point_loss=fallback_point,
        proper_score=fallback_proper,
        fallback_proper_score=fallback_proper,
        deployed_proper_score=fallback_proper,
        interval_covered=fallback_covered,
        interval_width=fallback_width,
    )
    return candidate_record, fallback_record


@dataclass(frozen=True, slots=True)
class GraphDynamicTournamentScoredBundleV1:
    """Immutable post-outcome records for one complete graph candidate roster."""

    prediction_bundle: GraphDynamicTournamentPredictionBundleV1
    targets_m: Sequence[FloatArray]
    scoring_policy: GraphDynamicTournamentScoringPolicyV1
    physical_fallback_candidate_id: str = "physical_fallback"
    artifact_id: str | None = None
    candidate: CandidateSpec = field(init=False)
    candidate_records: tuple[TournamentRecord, ...] = field(
        init=False,
        repr=False,
    )
    physical_fallback_records: tuple[TournamentRecord, ...] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.prediction_bundle,
            GraphDynamicTournamentPredictionBundleV1,
        ):
            raise TypeError(
                "prediction_bundle must be a GraphDynamicTournamentPredictionBundleV1"
            )
        if not isinstance(
            self.scoring_policy,
            GraphDynamicTournamentScoringPolicyV1,
        ):
            raise TypeError(
                "scoring_policy must be a GraphDynamicTournamentScoringPolicyV1"
            )
        if isinstance(self.targets_m, (str, bytes)) or not isinstance(
            self.targets_m,
            Sequence,
        ):
            raise ValueError("targets_m must be a sequence")
        raw_targets = tuple(self.targets_m)
        predictions = self.prediction_bundle.predictions
        if len(raw_targets) != len(predictions):
            raise ValueError("targets_m must match the complete prediction roster")
        targets: list[FloatArray] = []
        candidate_records: list[TournamentRecord] = []
        fallback_records: list[TournamentRecord] = []
        fallback_candidate = identifier(
            self.physical_fallback_candidate_id,
            name="physical_fallback_candidate_id",
        )
        for index, (prediction, raw_target) in enumerate(
            zip(predictions, raw_targets, strict=True)
        ):
            target = float_array(raw_target, name=f"targets_m[{index}]")
            if target.shape != prediction.mean_m.shape:
                raise ValueError(
                    f"targets_m[{index}] shape must match its selected forecast"
                )
            candidate_record, fallback_record = _matched_records(
                prediction,
                target,
                self.scoring_policy,
                physical_fallback_candidate_id=fallback_candidate,
            )
            targets.append(target)
            candidate_records.append(candidate_record)
            fallback_records.append(fallback_record)

        object.__setattr__(self, "targets_m", tuple(targets))
        object.__setattr__(
            self,
            "physical_fallback_candidate_id",
            fallback_candidate,
        )
        object.__setattr__(
            self,
            "candidate",
            self.prediction_bundle.candidate,
        )
        object.__setattr__(
            self,
            "candidate_records",
            tuple(candidate_records),
        )
        object.__setattr__(
            self,
            "physical_fallback_records",
            tuple(fallback_records),
        )
        expected_id = content_id(self.descriptor())
        if self.artifact_id is not None:
            supplied = literal_lower_hex(
                self.artifact_id,
                name="artifact_id",
                lengths={64},
            )
            if supplied != expected_id:
                raise ValueError("artifact_id does not match scored bundle content")
        object.__setattr__(self, "artifact_id", expected_id)

    @property
    def tournament_records(self) -> tuple[TournamentRecord, ...]:
        """Return physical fallback first, then graph candidate records."""

        return self.physical_fallback_records + self.candidate_records

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": GRAPH_DYNAMIC_TOURNAMENT_SCORE_SCHEMA,
            "schema_version": GRAPH_DYNAMIC_TOURNAMENT_SCORE_VERSION,
            "prediction_bundle_id": self.prediction_bundle.artifact_id,
            "scoring_policy_id": self.scoring_policy.artifact_id,
            "targets_m": [array_record(target) for target in self.targets_m],
            "candidate": candidate_spec_dict(self.candidate),
            "candidate_records": [
                tournament_record_dict(record) for record in self.candidate_records
            ],
            "physical_fallback_records": [
                tournament_record_dict(record)
                for record in self.physical_fallback_records
            ],
            "scientific_boundary": GRAPH_DYNAMIC_TOURNAMENT_BOUNDARY,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}


def score_graph_dynamic_tournament_prediction_bundle(
    prediction_bundle: GraphDynamicTournamentPredictionBundleV1,
    targets_m: Sequence[object],
    *,
    scoring_policy: GraphDynamicTournamentScoringPolicyV1 | None = None,
    physical_fallback_candidate_id: str = "physical_fallback",
) -> GraphDynamicTournamentScoredBundleV1:
    """Score a complete already-sealed roster after outcomes are available."""

    return GraphDynamicTournamentScoredBundleV1(
        prediction_bundle=prediction_bundle,
        targets_m=cast(Sequence[FloatArray], targets_m),
        scoring_policy=(
            GraphDynamicTournamentScoringPolicyV1()
            if scoring_policy is None
            else scoring_policy
        ),
        physical_fallback_candidate_id=physical_fallback_candidate_id,
    )


__all__ = [
    "GraphDynamicTournamentScoredBundleV1",
    "candidate_spec_dict",
    "score_graph_dynamic_tournament_prediction_bundle",
    "tournament_record_dict",
]
