"""Session-specific placebo separation for held-out action transport."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from bayesian_phystwin_experiments.cross_action_placebo_v1 import (
    CROSS_ACTION_PLACEBO_CLAIM_BOUNDARY,
    CROSS_ACTION_PLACEBO_SCORE_ORIENTATION,
    CrossActionPlaceboScoreRowV1,
    PlaceboArm,
    PlaceboDecision,
    SealedCrossActionPlaceboPredictionV1,
    _boolean,
    _construction_ids,
    _content_id,
    _digest,
    _finite,
    _frozen_mapping,
    _integer,
    _label,
    _percentile_interval,
    _placebos,
    _plain_json,
    _wilson_interval,
)
from bayesian_phystwin_experiments.cross_action_transport_contracts_v1 import (
    _optional_labels,
)
from bayesian_phystwin_experiments.cross_action_transport_v2 import (
    SessionActionSetV2,
)

CROSS_ACTION_PLACEBO_V2_SCHEMA: Final = (
    "bayesian_phystwin.cross_action_placebo_session_specific"
)
CROSS_ACTION_PLACEBO_V2_VERSION: Final = 2


@dataclass(frozen=True, slots=True)
class CrossActionPlaceboProtocolV2:
    """Target-closed placebo protocol with a per-session action subset."""

    parent_transport_protocol_id: str
    target_roster_id: str
    session_action_sets: tuple[SessionActionSetV2, ...]
    physical_arm_label: str
    placebo_arms: tuple[PlaceboArm, ...]
    arm_construction_ids: Mapping[str, str]
    minimum_sessions: int
    bootstrap_replicates: int
    bootstrap_seed: int
    confidence_level: float
    minimum_placebo_contrast: float
    score_orientation: str = CROSS_ACTION_PLACEBO_SCORE_ORIENTATION
    method_frozen_before_target: bool = True
    roster_frozen_before_target: bool = True
    predictions_sealed_before_target: bool = True
    target_outcomes_used_for_selection: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parent_transport_protocol_id",
            _digest(
                self.parent_transport_protocol_id,
                name="parent_transport_protocol_id",
            ),
        )
        object.__setattr__(
            self,
            "target_roster_id",
            _digest(self.target_roster_id, name="target_roster_id"),
        )
        session_sets = tuple(self.session_action_sets)
        if len(session_sets) < 2 or any(
            not isinstance(value, SessionActionSetV2) for value in session_sets
        ):
            raise TypeError(
                "session_action_sets must contain at least two "
                "SessionActionSetV2 values"
            )
        session_sets = tuple(
            sorted(session_sets, key=lambda value: value.object_session_id)
        )
        session_ids = tuple(value.object_session_id for value in session_sets)
        if len(session_ids) != len(set(session_ids)):
            raise ValueError("session_action_sets must not repeat a session")
        object.__setattr__(self, "session_action_sets", session_sets)

        physical = _label(self.physical_arm_label, name="physical_arm_label")
        placebos = _placebos(self.placebo_arms)
        if physical in {arm.value for arm in placebos}:
            raise ValueError("physical_arm_label must be distinct from placebo arms")
        arm_labels = (physical, *(arm.value for arm in placebos))
        constructions = _construction_ids(
            self.arm_construction_ids,
            arm_labels=arm_labels,
        )
        object.__setattr__(self, "physical_arm_label", physical)
        object.__setattr__(self, "placebo_arms", placebos)
        object.__setattr__(self, "arm_construction_ids", constructions)

        minimum_sessions = _integer(
            self.minimum_sessions,
            name="minimum_sessions",
            minimum=2,
        )
        if minimum_sessions > len(session_sets):
            raise ValueError("minimum_sessions exceeds the frozen target roster")
        object.__setattr__(self, "minimum_sessions", minimum_sessions)
        object.__setattr__(
            self,
            "bootstrap_replicates",
            _integer(
                self.bootstrap_replicates,
                name="bootstrap_replicates",
                minimum=100,
            ),
        )
        object.__setattr__(
            self,
            "bootstrap_seed",
            _integer(self.bootstrap_seed, name="bootstrap_seed", minimum=0),
        )
        confidence = _finite(
            self.confidence_level,
            name="confidence_level",
            minimum=0.0,
            maximum=1.0,
        )
        if confidence in {0.0, 1.0}:
            raise ValueError("confidence_level must be strictly between zero and one")
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(
            self,
            "minimum_placebo_contrast",
            _finite(
                self.minimum_placebo_contrast,
                name="minimum_placebo_contrast",
                minimum=0.0,
            ),
        )
        orientation = _label(self.score_orientation, name="score_orientation")
        if orientation != CROSS_ACTION_PLACEBO_SCORE_ORIENTATION:
            raise ValueError("score_orientation must be 'lower_is_better'")
        object.__setattr__(self, "score_orientation", orientation)
        for name in (
            "method_frozen_before_target",
            "roster_frozen_before_target",
            "predictions_sealed_before_target",
            "target_outcomes_used_for_selection",
        ):
            object.__setattr__(self, name, _boolean(getattr(self, name), name=name))
        if (
            not (
                self.method_frozen_before_target
                and self.roster_frozen_before_target
                and self.predictions_sealed_before_target
            )
            or self.target_outcomes_used_for_selection
        ):
            raise ValueError(
                "placebo protocol must be frozen, sealed, and target-selection free"
            )
        object.__setattr__(
            self,
            "metadata",
            _frozen_mapping(self.metadata, name="protocol metadata"),
        )

    @property
    def target_session_ids(self) -> tuple[str, ...]:
        return tuple(value.object_session_id for value in self.session_action_sets)

    @property
    def action_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    action
                    for session in self.session_action_sets
                    for action in session.action_ids
                }
            )
        )

    @property
    def arm_labels(self) -> tuple[str, ...]:
        return (self.physical_arm_label, *(arm.value for arm in self.placebo_arms))

    def off_diagonal_action_pairs_for_session(
        self,
        object_session_id: str,
    ) -> tuple[tuple[str, str], ...]:
        session_id = _label(object_session_id, name="object_session_id")
        for session in self.session_action_sets:
            if session.object_session_id == session_id:
                return session.off_diagonal_action_pairs
        raise KeyError(f"unregistered object session: {session_id}")

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": CROSS_ACTION_PLACEBO_V2_SCHEMA,
            "schema_version": CROSS_ACTION_PLACEBO_V2_VERSION,
            "artifact_kind": "CrossActionPlaceboProtocolV2",
            "parent_transport_protocol_id": self.parent_transport_protocol_id,
            "target_roster_id": self.target_roster_id,
            "action_ids": list(self.action_ids),
            "target_session_ids": list(self.target_session_ids),
            "session_action_sets": [
                value.descriptor() for value in self.session_action_sets
            ],
            "physical_arm_label": self.physical_arm_label,
            "placebo_arms": [arm.value for arm in self.placebo_arms],
            "arm_construction_ids": dict(self.arm_construction_ids),
            "minimum_sessions": self.minimum_sessions,
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_seed": self.bootstrap_seed,
            "confidence_level": self.confidence_level,
            "minimum_placebo_contrast": self.minimum_placebo_contrast,
            "score_orientation": self.score_orientation,
            "method_frozen_before_target": self.method_frozen_before_target,
            "roster_frozen_before_target": self.roster_frozen_before_target,
            "predictions_sealed_before_target": self.predictions_sealed_before_target,
            "target_outcomes_used_for_selection": (
                self.target_outcomes_used_for_selection
            ),
            "metadata": _plain_json(self.metadata),
            "claim_boundary": CROSS_ACTION_PLACEBO_CLAIM_BOUNDARY,
        }

    @property
    def protocol_id(self) -> str:
        return _content_id(self.descriptor())


@dataclass(frozen=True, slots=True)
class CrossActionPlaceboResultV2:
    """Session-clustered placebo contrasts for session-specific action sets."""

    protocol: CrossActionPlaceboProtocolV2
    score_rows: Sequence[CrossActionPlaceboScoreRowV1]
    target_accounting_id: str
    excluded_session_ids: tuple[str, ...] = ()

    independent_session_count: int = field(init=False)
    session_mean_scores: np.ndarray = field(init=False, repr=False)
    session_placebo_contrasts: np.ndarray = field(init=False, repr=False)
    mean_placebo_contrasts: np.ndarray = field(init=False)
    placebo_contrast_intervals: np.ndarray = field(init=False)
    placebo_win_fractions: np.ndarray = field(init=False)
    placebo_win_fraction_intervals: np.ndarray = field(init=False)
    selected_physical_prediction_count: int = field(init=False)
    decision: PlaceboDecision = field(init=False)
    result_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.protocol, CrossActionPlaceboProtocolV2):
            raise TypeError("protocol must be CrossActionPlaceboProtocolV2")
        target_accounting_id = _digest(
            self.target_accounting_id,
            name="target_accounting_id",
        )
        excluded_sessions = _optional_labels(
            self.excluded_session_ids,
            name="excluded_session_ids",
        )
        rows = tuple(self.score_rows)
        if not rows or any(
            not isinstance(row, CrossActionPlaceboScoreRowV1) for row in rows
        ):
            raise TypeError(
                "score_rows must contain CrossActionPlaceboScoreRowV1 values"
            )
        protocol = self.protocol
        scored_sessions = tuple(
            sorted({row.prediction.object_session_id for row in rows})
        )
        if set(scored_sessions) & set(excluded_sessions):
            raise ValueError("scored and excluded target sessions must be disjoint")
        if set(scored_sessions) | set(excluded_sessions) != set(
            protocol.target_session_ids
        ):
            raise ValueError("target session accounting must cover the frozen roster")

        expected_keys = {
            (session, source, target, arm)
            for session in scored_sessions
            for source, target in protocol.off_diagonal_action_pairs_for_session(
                session
            )
            for arm in protocol.arm_labels
        }
        by_key: dict[
            tuple[str, str, str, str], CrossActionPlaceboScoreRowV1
        ] = {}
        attestation_ids: set[str] = set()
        scorer_ids: set[str] = set()
        batch_ids: set[str] = set()
        commit_ids: set[str] = set()
        for row in rows:
            prediction = row.prediction
            if prediction.protocol_id != protocol.protocol_id:
                raise ValueError("prediction belongs to another placebo protocol")
            key = (
                prediction.object_session_id,
                prediction.source_action_id,
                prediction.target_action_id,
                prediction.arm_label,
            )
            if key in by_key:
                raise ValueError("duplicate placebo score row")
            expected_construction = protocol.arm_construction_ids.get(
                prediction.arm_label
            )
            if prediction.construction_id != expected_construction:
                raise ValueError("prediction construction does not match protocol")
            by_key[key] = row
            attestation_ids.add(row.target_access_attestation_id)
            scorer_ids.add(row.scorer_id)
            batch_ids.add(prediction.prediction_batch_id)
            commit_ids.add(prediction.commit_id)
        if set(by_key) != expected_keys:
            missing = expected_keys - set(by_key)
            extra = set(by_key) - expected_keys
            raise ValueError(
                "score table does not match the frozen session-specific matrix: "
                f"missing={len(missing)}, extra={len(extra)}"
            )
        if len(attestation_ids) != 1:
            raise ValueError("all rows must bind one target-access attestation")
        if len(scorer_ids) != 1:
            raise ValueError("all rows must bind one frozen scorer")
        if len(batch_ids) != 1:
            raise ValueError("all predictions must belong to one sealed batch")
        if len(commit_ids) != 1:
            raise ValueError("all predictions must bind one exact source revision")

        session_count = len(scored_sessions)
        arm_count = len(protocol.arm_labels)
        session_scores = np.empty((session_count, arm_count), dtype=np.float64)
        selected_count = 0
        for session_index, session in enumerate(scored_sessions):
            pairs = protocol.off_diagonal_action_pairs_for_session(session)
            for source, target in pairs:
                pair_rows = [
                    by_key[(session, source, target, arm)]
                    for arm in protocol.arm_labels
                ]
                if len({row.target_outcome_id for row in pair_rows}) != 1:
                    raise ValueError(
                        "all arms for an action pair must score the same outcome"
                    )
                predictions = [row.prediction for row in pair_rows]
                if (
                    len(
                        {
                            prediction.parent_transport_prediction_id
                            for prediction in predictions
                        }
                    )
                    != 1
                ):
                    raise ValueError(
                        "all arms must bind the same parent transport prediction"
                    )
                dispositions = {
                    (prediction.candidate_selected, prediction.exact_fallback)
                    for prediction in predictions
                }
                if len(dispositions) != 1:
                    raise ValueError("all arms must share one parent disposition")
                selected, fallback = next(iter(dispositions))
                if selected:
                    selected_count += 1
                if fallback:
                    if (
                        len(
                            {
                                prediction.prediction_artifact_id
                                for prediction in predictions
                            }
                        )
                        != 1
                    ):
                        raise ValueError(
                            "exact fallback must select one identical "
                            "prediction artifact"
                        )
                    if len({row.proper_score for row in pair_rows}) != 1:
                        raise ValueError(
                            "exact fallback must produce one identical proper score"
                        )
            for arm_index, arm in enumerate(protocol.arm_labels):
                arm_rows = [
                    by_key[(session, source, target, arm)]
                    for source, target in pairs
                ]
                session_scores[session_index, arm_index] = float(
                    np.mean([row.proper_score for row in arm_rows])
                )

        contrasts = session_scores[:, 1:] - session_scores[:, [0]]
        mean_contrasts = np.mean(contrasts, axis=0)
        intervals = np.empty((len(protocol.placebo_arms), 2), dtype=np.float64)
        win_fractions = np.mean(contrasts <= 0.0, axis=0)
        win_intervals = np.empty_like(intervals)
        for index in range(len(protocol.placebo_arms)):
            intervals[index] = _percentile_interval(
                contrasts[:, index],
                replicates=protocol.bootstrap_replicates,
                seed=protocol.bootstrap_seed + index,
                confidence_level=protocol.confidence_level,
            )
            wins = int(np.sum(contrasts[:, index] <= 0.0))
            win_intervals[index] = _wilson_interval(
                wins,
                session_count,
                protocol.confidence_level,
            )

        if session_count < protocol.minimum_sessions:
            decision = PlaceboDecision.INSUFFICIENT_SESSIONS
        elif selected_count == 0:
            decision = PlaceboDecision.NOT_SUPPORTED
        elif np.all(intervals[:, 0] > protocol.minimum_placebo_contrast):
            decision = PlaceboDecision.SUPPORTED
        else:
            decision = PlaceboDecision.NOT_SUPPORTED

        for array in (
            session_scores,
            contrasts,
            mean_contrasts,
            intervals,
            win_fractions,
            win_intervals,
        ):
            array.setflags(write=False)
        object.__setattr__(
            self,
            "score_rows",
            tuple(sorted(rows, key=lambda row: row.score_row_id)),
        )
        object.__setattr__(self, "target_accounting_id", target_accounting_id)
        object.__setattr__(self, "excluded_session_ids", excluded_sessions)
        object.__setattr__(self, "independent_session_count", session_count)
        object.__setattr__(self, "session_mean_scores", session_scores)
        object.__setattr__(self, "session_placebo_contrasts", contrasts)
        object.__setattr__(self, "mean_placebo_contrasts", mean_contrasts)
        object.__setattr__(self, "placebo_contrast_intervals", intervals)
        object.__setattr__(self, "placebo_win_fractions", win_fractions)
        object.__setattr__(self, "placebo_win_fraction_intervals", win_intervals)
        object.__setattr__(self, "selected_physical_prediction_count", selected_count)
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "result_id", _content_id(self.descriptor()))

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": CROSS_ACTION_PLACEBO_V2_SCHEMA,
            "schema_version": CROSS_ACTION_PLACEBO_V2_VERSION,
            "artifact_kind": "CrossActionPlaceboResultV2",
            "protocol_id": self.protocol.protocol_id,
            "target_accounting_id": self.target_accounting_id,
            "excluded_session_ids": list(self.excluded_session_ids),
            "score_row_ids": [row.score_row_id for row in self.score_rows],
            "placebo_arms": [arm.value for arm in self.protocol.placebo_arms],
            "independent_session_count": self.independent_session_count,
            "selected_physical_prediction_count": (
                self.selected_physical_prediction_count
            ),
            "mean_placebo_contrasts": self.mean_placebo_contrasts.tolist(),
            "placebo_contrast_intervals": self.placebo_contrast_intervals.tolist(),
            "placebo_win_fractions": self.placebo_win_fractions.tolist(),
            "placebo_win_fraction_intervals": (
                self.placebo_win_fraction_intervals.tolist()
            ),
            "decision": self.decision.value,
            "claim_boundary": CROSS_ACTION_PLACEBO_CLAIM_BOUNDARY,
        }

    def to_record(self) -> dict[str, object]:
        return {
            **self.descriptor(),
            "result_id": self.result_id,
            "session_ids": [
                session
                for session in self.protocol.target_session_ids
                if session not in self.excluded_session_ids
            ],
            "arm_labels": list(self.protocol.arm_labels),
            "session_mean_scores": self.session_mean_scores.tolist(),
            "session_placebo_contrasts": self.session_placebo_contrasts.tolist(),
        }


__all__ = [
    "CROSS_ACTION_PLACEBO_V2_SCHEMA",
    "CROSS_ACTION_PLACEBO_V2_VERSION",
    "CrossActionPlaceboProtocolV2",
    "CrossActionPlaceboResultV2",
    "CrossActionPlaceboScoreRowV1",
    "PlaceboArm",
    "PlaceboDecision",
    "SealedCrossActionPlaceboPredictionV1",
]
