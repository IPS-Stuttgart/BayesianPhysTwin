"""Placebo-separation certificate for prospective cross-action transport.

This target-closed supplement tests whether one frozen physical-transport
candidate beats deliberately broken transport controls on the same held-out
outcomes. Complete physical object/sessions, not action pairs or frames, are the
independent statistical units.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from statistics import NormalDist
from types import MappingProxyType
from typing import Any, Final

import numpy as np

CROSS_ACTION_PLACEBO_SCHEMA: Final = "bayesian_phystwin.cross_action_placebo"
CROSS_ACTION_PLACEBO_VERSION: Final = 1
CROSS_ACTION_PLACEBO_CLAIM_BOUNDARY: Final = (
    "A positive result establishes bounded separation from the exact registered "
    "wrong-action, wrong-object, phase-shifted, and/or identity-permuted controls "
    "on the frozen object/session and action roster. It does not establish a "
    "unique physical cause, arbitrary-action transfer, arbitrary-object transfer, "
    "deployment safety, calibrated real-data uncertainty, Prob4D competence, "
    "Causal4D intervention benefit, or state of the art."
)


class PlaceboArm(str, Enum):
    """Controls that preserve nuisance statistics but break physical transport."""

    WRONG_ACTION = "wrong_action"
    WRONG_OBJECT = "wrong_object"
    PHASE_SHIFTED = "phase_shifted"
    IDENTITY_PERMUTED = "identity_permuted"


class PlaceboDecision(str, Enum):
    """Conjunctive decision over all registered placebo contrasts."""

    SUPPORTED = "physical_transport_placebo_separation_supported"
    NOT_SUPPORTED = "physical_transport_placebo_separation_not_supported"
    INSUFFICIENT_SESSIONS = "insufficient_independent_sessions"


def _plain_json(value: object) -> object:
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not np.isfinite(value):
            raise ValueError("JSON floats must be finite")
        return value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, nested in value.items():
            if type(key) is not str:
                raise ValueError("JSON mapping keys must be strings")
            result[key] = _plain_json(nested)
        return result
    if isinstance(value, (tuple, list)):
        return [_plain_json(nested) for nested in value]
    if isinstance(value, np.generic):
        return _plain_json(value.item())
    raise ValueError(f"value of type {type(value).__name__} is not strict JSON")


def _frozen_mapping(value: Mapping[str, Any], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    plain = _plain_json(value)
    assert isinstance(plain, dict)
    return MappingProxyType(plain)


def _content_id(value: object) -> str:
    payload = json.dumps(
        _plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _digest(value: object, *, name: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex string")
    if value != value.lower() or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex string")
    return value


def _commit(value: object, *, name: str) -> str:
    if type(value) is not str or len(value) not in {40, 64}:
        raise ValueError(f"{name} must be a lowercase Git or SHA-256 hex string")
    if value != value.lower() or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase Git or SHA-256 hex string")
    return value


def _label(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty literal string")
    return value


def _labels(values: Sequence[str], *, name: str, minimum: int = 1) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of strings")
    result = tuple(_label(value, name=name) for value in values)
    if len(result) < minimum:
        raise ValueError(f"{name} must contain at least {minimum} values")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(sorted(result))


def _finite(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real scalar")
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(raw.item())
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def _integer(value: object, *, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


def _boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a literal boolean")
    return value


def _placebos(values: Sequence[PlaceboArm]) -> tuple[PlaceboArm, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("placebo_arms must be a sequence")
    result = tuple(values)
    if not result or any(not isinstance(value, PlaceboArm) for value in result):
        raise TypeError("placebo_arms must contain PlaceboArm values")
    if len(result) != len(set(result)):
        raise ValueError("placebo_arms must not contain duplicates")
    return tuple(sorted(result, key=lambda value: value.value))


def _construction_ids(
    values: Mapping[str, str],
    *,
    arm_labels: Sequence[str],
) -> Mapping[str, str]:
    if not isinstance(values, Mapping):
        raise TypeError("arm_construction_ids must be a mapping")
    expected = set(arm_labels)
    supplied = set(values)
    if supplied != expected:
        raise ValueError(
            "arm_construction_ids must contain exactly every registered arm"
        )
    canonical = {
        arm: _digest(values[arm], name=f"arm_construction_ids[{arm!r}]")
        for arm in sorted(expected)
    }
    return MappingProxyType(canonical)


def _percentile_interval(
    values: np.ndarray,
    *,
    replicates: int,
    seed: int,
    confidence_level: float,
) -> tuple[float, float]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or len(vector) < 2 or not np.all(np.isfinite(vector)):
        raise ValueError(
            "bootstrap values must be a finite vector with at least two groups"
        )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(vector), size=(replicates, len(vector)))
    means = np.mean(vector[indices], axis=1)
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lower), float(upper)


def _wilson_interval(
    successes: int,
    total: int,
    confidence_level: float,
) -> tuple[float, float]:
    if total < 1 or not 0 <= successes <= total:
        raise ValueError("Wilson counts are invalid")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("Wilson confidence_level must lie in (0, 1)")
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    proportion = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (proportion + z2 / (2.0 * total)) / denominator
    radius = z * np.sqrt(
        proportion * (1.0 - proportion) / total + z2 / (4.0 * total * total)
    ) / denominator
    return max(0.0, float(center - radius)), min(1.0, float(center + radius))


@dataclass(frozen=True, slots=True)
class CrossActionPlaceboProtocolV1:
    """Target-closed placebo protocol bound to one transport protocol."""

    parent_transport_protocol_id: str
    target_roster_id: str
    action_ids: Sequence[str]
    target_session_ids: Sequence[str]
    physical_arm_label: str
    placebo_arms: Sequence[PlaceboArm]
    arm_construction_ids: Mapping[str, str]
    minimum_sessions: int
    bootstrap_replicates: int
    bootstrap_seed: int
    confidence_level: float
    minimum_placebo_contrast: float
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
        actions = _labels(self.action_ids, name="action_ids", minimum=2)
        sessions = _labels(
            self.target_session_ids,
            name="target_session_ids",
            minimum=2,
        )
        physical = _label(self.physical_arm_label, name="physical_arm_label")
        placebos = _placebos(self.placebo_arms)
        if physical in {arm.value for arm in placebos}:
            raise ValueError("physical_arm_label must be distinct from placebo arms")
        arm_labels = (physical, *(arm.value for arm in placebos))
        constructions = _construction_ids(
            self.arm_construction_ids,
            arm_labels=arm_labels,
        )
        object.__setattr__(self, "action_ids", actions)
        object.__setattr__(self, "target_session_ids", sessions)
        object.__setattr__(self, "physical_arm_label", physical)
        object.__setattr__(self, "placebo_arms", placebos)
        object.__setattr__(self, "arm_construction_ids", constructions)
        object.__setattr__(
            self,
            "minimum_sessions",
            _integer(self.minimum_sessions, name="minimum_sessions", minimum=2),
        )
        if self.minimum_sessions > len(sessions):
            raise ValueError("minimum_sessions exceeds the frozen target roster")
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
        for name in (
            "method_frozen_before_target",
            "roster_frozen_before_target",
            "predictions_sealed_before_target",
            "target_outcomes_used_for_selection",
        ):
            object.__setattr__(self, name, _boolean(getattr(self, name), name=name))
        if not (
            self.method_frozen_before_target
            and self.roster_frozen_before_target
            and self.predictions_sealed_before_target
        ) or self.target_outcomes_used_for_selection:
            raise ValueError(
                "placebo protocol must be frozen, sealed, and target-selection free"
            )
        object.__setattr__(
            self,
            "metadata",
            _frozen_mapping(self.metadata, name="protocol metadata"),
        )

    @property
    def off_diagonal_action_pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (source, target)
            for source in self.action_ids
            for target in self.action_ids
            if source != target
        )

    @property
    def arm_labels(self) -> tuple[str, ...]:
        return (self.physical_arm_label, *(arm.value for arm in self.placebo_arms))

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": CROSS_ACTION_PLACEBO_SCHEMA,
            "schema_version": CROSS_ACTION_PLACEBO_VERSION,
            "artifact_kind": "CrossActionPlaceboProtocolV1",
            "parent_transport_protocol_id": self.parent_transport_protocol_id,
            "target_roster_id": self.target_roster_id,
            "action_ids": list(self.action_ids),
            "target_session_ids": list(self.target_session_ids),
            "off_diagonal_action_pairs": [
                list(pair) for pair in self.off_diagonal_action_pairs
            ],
            "physical_arm_label": self.physical_arm_label,
            "placebo_arms": [arm.value for arm in self.placebo_arms],
            "arm_construction_ids": dict(self.arm_construction_ids),
            "minimum_sessions": self.minimum_sessions,
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_seed": self.bootstrap_seed,
            "confidence_level": self.confidence_level,
            "minimum_placebo_contrast": self.minimum_placebo_contrast,
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
class SealedCrossActionPlaceboPredictionV1:
    """One target-blind physical or placebo prediction."""

    protocol_id: str
    object_session_id: str
    source_action_id: str
    target_action_id: str
    arm_label: str
    parent_transport_prediction_id: str
    construction_id: str
    prediction_artifact_id: str
    prediction_batch_id: str
    commit_id: str
    candidate_selected: bool
    exact_fallback: bool
    prediction_sealed_before_target: bool
    target_outcomes_used: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "protocol_id",
            "parent_transport_prediction_id",
            "construction_id",
            "prediction_artifact_id",
            "prediction_batch_id",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        object.__setattr__(self, "commit_id", _commit(self.commit_id, name="commit_id"))
        for name in (
            "object_session_id",
            "source_action_id",
            "target_action_id",
            "arm_label",
        ):
            object.__setattr__(self, name, _label(getattr(self, name), name=name))
        if self.source_action_id == self.target_action_id:
            raise ValueError(
                "placebo certificate accepts off-diagonal action pairs only"
            )
        for name in (
            "candidate_selected",
            "exact_fallback",
            "prediction_sealed_before_target",
            "target_outcomes_used",
        ):
            object.__setattr__(self, name, _boolean(getattr(self, name), name=name))
        if not self.prediction_sealed_before_target or self.target_outcomes_used:
            raise ValueError("predictions must be sealed before target access")
        if self.candidate_selected == self.exact_fallback:
            raise ValueError(
                "prediction must bind exactly one of candidate selection or fallback"
            )
        object.__setattr__(
            self,
            "metadata",
            _frozen_mapping(self.metadata, name="prediction metadata"),
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": CROSS_ACTION_PLACEBO_SCHEMA,
            "schema_version": CROSS_ACTION_PLACEBO_VERSION,
            "artifact_kind": "SealedCrossActionPlaceboPredictionV1",
            "protocol_id": self.protocol_id,
            "object_session_id": self.object_session_id,
            "source_action_id": self.source_action_id,
            "target_action_id": self.target_action_id,
            "arm_label": self.arm_label,
            "parent_transport_prediction_id": self.parent_transport_prediction_id,
            "construction_id": self.construction_id,
            "prediction_artifact_id": self.prediction_artifact_id,
            "prediction_batch_id": self.prediction_batch_id,
            "commit_id": self.commit_id,
            "candidate_selected": self.candidate_selected,
            "exact_fallback": self.exact_fallback,
            "prediction_sealed_before_target": self.prediction_sealed_before_target,
            "target_outcomes_used": self.target_outcomes_used,
            "metadata": _plain_json(self.metadata),
        }

    @property
    def prediction_id(self) -> str:
        return _content_id(self.descriptor())


@dataclass(frozen=True, slots=True)
class CrossActionPlaceboScoreRowV1:
    """One post-access score bound to one sealed prediction."""

    prediction: SealedCrossActionPlaceboPredictionV1
    target_outcome_id: str
    target_access_attestation_id: str
    scorer_id: str
    proper_score: float
    target_side_selection_used: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.prediction, SealedCrossActionPlaceboPredictionV1):
            raise TypeError(
                "prediction must be a SealedCrossActionPlaceboPredictionV1"
            )
        for name in (
            "target_outcome_id",
            "target_access_attestation_id",
            "scorer_id",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "proper_score",
            _finite(self.proper_score, name="proper_score"),
        )
        selected = _boolean(
            self.target_side_selection_used,
            name="target_side_selection_used",
        )
        if selected:
            raise ValueError("target-side model or threshold selection is forbidden")
        object.__setattr__(self, "target_side_selection_used", selected)
        object.__setattr__(
            self,
            "metadata",
            _frozen_mapping(self.metadata, name="score-row metadata"),
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": CROSS_ACTION_PLACEBO_SCHEMA,
            "schema_version": CROSS_ACTION_PLACEBO_VERSION,
            "artifact_kind": "CrossActionPlaceboScoreRowV1",
            "prediction_id": self.prediction.prediction_id,
            "target_outcome_id": self.target_outcome_id,
            "target_access_attestation_id": self.target_access_attestation_id,
            "scorer_id": self.scorer_id,
            "proper_score": self.proper_score,
            "target_side_selection_used": self.target_side_selection_used,
            "metadata": _plain_json(self.metadata),
        }

    @property
    def score_row_id(self) -> str:
        return _content_id(self.descriptor())


@dataclass(frozen=True, slots=True)
class CrossActionPlaceboResultV1:
    """Session-clustered placebo contrasts for one frozen transport candidate."""

    protocol: CrossActionPlaceboProtocolV1
    score_rows: Sequence[CrossActionPlaceboScoreRowV1]
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
        if not isinstance(self.protocol, CrossActionPlaceboProtocolV1):
            raise TypeError("protocol must be CrossActionPlaceboProtocolV1")
        rows = tuple(self.score_rows)
        if not rows or any(
            not isinstance(row, CrossActionPlaceboScoreRowV1) for row in rows
        ):
            raise TypeError(
                "score_rows must contain CrossActionPlaceboScoreRowV1 values"
            )
        protocol = self.protocol
        expected_keys = {
            (session, source, target, arm)
            for session in protocol.target_session_ids
            for source, target in protocol.off_diagonal_action_pairs
            for arm in protocol.arm_labels
        }
        by_key: dict[tuple[str, str, str, str], CrossActionPlaceboScoreRowV1] = {}
        attestation_ids: set[str] = set()
        scorer_ids: set[str] = set()
        batch_ids: set[str] = set()
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
        if set(by_key) != expected_keys:
            missing = expected_keys - set(by_key)
            extra = set(by_key) - expected_keys
            raise ValueError(
                f"score table does not match the frozen complete matrix: "
                f"missing={len(missing)}, extra={len(extra)}"
            )
        if len(attestation_ids) != 1:
            raise ValueError("all rows must bind one target-access attestation")
        if len(scorer_ids) != 1:
            raise ValueError("all rows must bind one frozen scorer")
        if len(batch_ids) != 1:
            raise ValueError("all predictions must belong to one sealed batch")

        session_count = len(protocol.target_session_ids)
        arm_count = len(protocol.arm_labels)
        session_scores = np.empty((session_count, arm_count), dtype=np.float64)
        selected_count = 0
        for session_index, session in enumerate(protocol.target_session_ids):
            for source, target in protocol.off_diagonal_action_pairs:
                pair_rows = [
                    by_key[(session, source, target, arm)]
                    for arm in protocol.arm_labels
                ]
                if len({row.target_outcome_id for row in pair_rows}) != 1:
                    raise ValueError(
                        "all arms for an action pair must score the same outcome"
                    )
                predictions = [row.prediction for row in pair_rows]
                parent_prediction_ids = {
                    prediction.parent_transport_prediction_id
                    for prediction in predictions
                }
                if len(parent_prediction_ids) != 1:
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
                if fallback and len(
                    {prediction.prediction_artifact_id for prediction in predictions}
                ) != 1:
                    raise ValueError(
                        "exact fallback must select one identical prediction artifact"
                    )
            for arm_index, arm in enumerate(protocol.arm_labels):
                arm_rows = [
                    by_key[(session, source, target, arm)]
                    for source, target in protocol.off_diagonal_action_pairs
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
            "schema": CROSS_ACTION_PLACEBO_SCHEMA,
            "schema_version": CROSS_ACTION_PLACEBO_VERSION,
            "artifact_kind": "CrossActionPlaceboResultV1",
            "protocol_id": self.protocol.protocol_id,
            "score_row_ids": [row.score_row_id for row in self.score_rows],
            "placebo_arms": [arm.value for arm in self.protocol.placebo_arms],
            "independent_session_count": len(self.protocol.target_session_ids),
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
            "session_ids": list(self.protocol.target_session_ids),
            "arm_labels": list(self.protocol.arm_labels),
            "session_mean_scores": self.session_mean_scores.tolist(),
            "session_placebo_contrasts": self.session_placebo_contrasts.tolist(),
        }


__all__ = [
    "CROSS_ACTION_PLACEBO_CLAIM_BOUNDARY",
    "CROSS_ACTION_PLACEBO_SCHEMA",
    "CROSS_ACTION_PLACEBO_VERSION",
    "CrossActionPlaceboProtocolV1",
    "CrossActionPlaceboResultV1",
    "CrossActionPlaceboScoreRowV1",
    "PlaceboArm",
    "PlaceboDecision",
    "SealedCrossActionPlaceboPredictionV1",
]
