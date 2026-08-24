"""Prospective held-out action transport evidence for BayesianPhysTwin.

The contracts keep target-blind prediction publication separate from target
scoring.  Action pairs are nested observations; complete physical object
sessions are the independent statistical units.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final, cast

import numpy as np

from bayesian_phystwin._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    genuine_integer,
    literal_lower_hex,
    plain_json,
)
from bayesian_phystwin._portable_contracts import content_id

CROSS_ACTION_TRANSPORT_SCHEMA: Final = "bayesian_phystwin.cross_action_transport"
CROSS_ACTION_TRANSPORT_VERSION: Final = 1
CROSS_ACTION_TRANSPORT_SEMANTICS: Final = (
    "target-blind-cross-action-prediction-and-session-bootstrap-v1"
)
CROSS_ACTION_TRANSPORT_CLAIM_BOUNDARY: Final = (
    "A positive result establishes bounded held-out action transport only for "
    "the exact frozen object/session roster, action matrix, physical query, "
    "score, candidates, guards, software stack, and numerical environment. It "
    "does not establish a unique physical cause, general unseen-object transfer, "
    "calibrated raw uncertainty, deployment safety, Prob4D provider competence, "
    "Causal4D intervention benefit, or deformable-object state of the art."
)


class TransportArm(str, Enum):
    """Registered comparison arms."""

    PHYSICAL_FALLBACK = "physical_fallback"
    LAST_RESIDUAL = "last_residual"
    DISCREPANCY_ONLY = "discrepancy_only"
    STATE_ONLY = "state_only"
    STATE_PARAMETER = "state_parameter"
    GUARDED_PHYSICAL = "guarded_physical"


class PredictionDisposition(str, Enum):
    """Target-blind complete-belief routing decision."""

    BASELINE_REFERENCE = "baseline_reference"
    CANDIDATE_SELECTED = "candidate_selected"
    EXACT_FALLBACK = "exact_fallback"


class TransportDecision(str, Enum):
    """Registered action-transport decision."""

    SUPPORTED = "physical_transport_supported"
    NOT_SUPPORTED = "physical_transport_not_supported"
    INSUFFICIENT_SESSIONS = "insufficient_off_diagonal_sessions"


def _digest(value: object, *, name: str) -> str:
    return cast(str, literal_lower_hex(value, name=name, lengths={64}))


def _commit(value: object, *, name: str) -> str:
    return cast(str, literal_lower_hex(value, name=name, lengths={40, 64}))


def _finite(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real number")
    result = float(raw.item())
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def _probability(value: object, *, name: str) -> float:
    result = _finite(value, name=name, minimum=0.0, maximum=1.0)
    if result in {0.0, 1.0}:
        raise ValueError(f"{name} must be strictly between zero and one")
    return result


def _labels(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of strings")
    result = tuple(values)
    if not result or any(type(value) is not str or not value for value in result):
        raise ValueError(f"{name} must contain nonempty literal strings")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(sorted(result))


def _optional_labels(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of strings")
    result = tuple(values)
    if any(type(value) is not str or not value for value in result):
        raise ValueError(f"{name} must contain nonempty literal strings")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(sorted(result))


def _arms(values: Sequence[TransportArm]) -> tuple[TransportArm, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("registered_arms must be a sequence")
    result = tuple(values)
    if not result or any(not isinstance(value, TransportArm) for value in result):
        raise TypeError("registered_arms must contain TransportArm values")
    if len(result) != len(set(result)):
        raise ValueError("registered_arms must not contain duplicates")
    return tuple(sorted(result, key=lambda value: value.value))


@dataclass(frozen=True, slots=True)
class CrossActionProtocolV1:
    """Target-closed protocol for one crossed-action study."""

    development_roster_id: str
    calibration_roster_id: str
    target_roster_id: str
    query_id: str
    query_jacobian_id: str
    score_definition_id: str
    grouping_rule_id: str
    interval_method_id: str
    target_access_policy_id: str
    model_stack_id: str
    numerical_environment_id: str
    technical_failure_policy_id: str
    action_ids: tuple[str, ...]
    target_session_ids: tuple[str, ...]
    registered_arms: tuple[TransportArm, ...]
    physical_transport_arm: TransportArm
    discrepancy_reference_arm: TransportArm
    matched_comparator_arm: TransportArm
    minimum_sessions: int
    bootstrap_replicates: int
    bootstrap_seed: int
    confidence_level: float
    minimum_off_diagonal_gain: float
    minimum_discrepancy_contrast: float
    minimum_comparator_contrast: float
    maximum_harmful_session_fraction: float
    harmful_gain_margin: float = 0.0
    lower_is_better: bool = True
    method_frozen_before_target: bool = True
    roster_frozen_before_target: bool = True
    target_outcomes_used: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "development_roster_id",
            "calibration_roster_id",
            "target_roster_id",
            "query_id",
            "query_jacobian_id",
            "score_definition_id",
            "grouping_rule_id",
            "interval_method_id",
            "target_access_policy_id",
            "model_stack_id",
            "numerical_environment_id",
            "technical_failure_policy_id",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        actions = _labels(self.action_ids, name="action_ids")
        if len(actions) < 2:
            raise ValueError("cross-action transport requires at least two actions")
        object.__setattr__(self, "action_ids", actions)
        target_sessions = _labels(
            self.target_session_ids,
            name="target_session_ids",
        )
        object.__setattr__(self, "target_session_ids", target_sessions)
        arms = _arms(self.registered_arms)
        object.__setattr__(self, "registered_arms", arms)
        for name in (
            "physical_transport_arm",
            "discrepancy_reference_arm",
            "matched_comparator_arm",
        ):
            arm = getattr(self, name)
            if not isinstance(arm, TransportArm) or arm not in arms:
                raise ValueError(f"{name} must be one registered arm")
        if TransportArm.PHYSICAL_FALLBACK not in arms:
            raise ValueError("the physical fallback arm is mandatory")
        if self.physical_transport_arm in {
            TransportArm.PHYSICAL_FALLBACK,
            self.discrepancy_reference_arm,
            self.matched_comparator_arm,
        }:
            raise ValueError("physical_transport_arm must be a distinct candidate")
        if self.discrepancy_reference_arm is TransportArm.PHYSICAL_FALLBACK:
            raise ValueError("discrepancy_reference_arm cannot be the fallback")
        if self.matched_comparator_arm is TransportArm.PHYSICAL_FALLBACK:
            raise ValueError("matched_comparator_arm cannot be the fallback")
        if self.discrepancy_reference_arm is self.matched_comparator_arm:
            raise ValueError("discrepancy and comparator arms must be distinct")
        object.__setattr__(
            self,
            "minimum_sessions",
            genuine_integer(self.minimum_sessions, name="minimum_sessions", minimum=2),
        )
        object.__setattr__(
            self,
            "bootstrap_replicates",
            genuine_integer(
                self.bootstrap_replicates,
                name="bootstrap_replicates",
                minimum=100,
            ),
        )
        object.__setattr__(
            self,
            "bootstrap_seed",
            genuine_integer(self.bootstrap_seed, name="bootstrap_seed", minimum=0),
        )
        object.__setattr__(
            self,
            "confidence_level",
            _probability(self.confidence_level, name="confidence_level"),
        )
        for name in (
            "minimum_off_diagonal_gain",
            "minimum_discrepancy_contrast",
            "minimum_comparator_contrast",
            "harmful_gain_margin",
        ):
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), name=name, minimum=0.0),
            )
        object.__setattr__(
            self,
            "maximum_harmful_session_fraction",
            _finite(
                self.maximum_harmful_session_fraction,
                name="maximum_harmful_session_fraction",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        lower_is_better = genuine_boolean(
            self.lower_is_better,
            name="lower_is_better",
        )
        if not lower_is_better:
            raise ValueError("v1 requires a lower-is-better proper score")
        object.__setattr__(self, "lower_is_better", lower_is_better)
        frozen = genuine_boolean(
            self.method_frozen_before_target,
            name="method_frozen_before_target",
        )
        roster_frozen = genuine_boolean(
            self.roster_frozen_before_target,
            name="roster_frozen_before_target",
        )
        target_used = genuine_boolean(
            self.target_outcomes_used,
            name="target_outcomes_used",
        )
        if not frozen or not roster_frozen or target_used:
            raise ValueError("protocol must be frozen and target-outcome free")
        object.__setattr__(self, "method_frozen_before_target", frozen)
        object.__setattr__(self, "roster_frozen_before_target", roster_frozen)
        object.__setattr__(self, "target_outcomes_used", target_used)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="protocol metadata"),
        )

    @property
    def action_pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (source, target)
            for source in self.action_ids
            for target in self.action_ids
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": CROSS_ACTION_TRANSPORT_SCHEMA,
            "schema_version": CROSS_ACTION_TRANSPORT_VERSION,
            "artifact_kind": "CrossActionProtocolV1",
            "semantics": CROSS_ACTION_TRANSPORT_SEMANTICS,
            "development_roster_id": self.development_roster_id,
            "calibration_roster_id": self.calibration_roster_id,
            "target_roster_id": self.target_roster_id,
            "query_id": self.query_id,
            "query_jacobian_id": self.query_jacobian_id,
            "score_definition_id": self.score_definition_id,
            "grouping_rule_id": self.grouping_rule_id,
            "interval_method_id": self.interval_method_id,
            "target_access_policy_id": self.target_access_policy_id,
            "model_stack_id": self.model_stack_id,
            "numerical_environment_id": self.numerical_environment_id,
            "technical_failure_policy_id": self.technical_failure_policy_id,
            "action_ids": list(self.action_ids),
            "target_session_ids": list(self.target_session_ids),
            "action_pairs": [list(pair) for pair in self.action_pairs],
            "registered_arms": [arm.value for arm in self.registered_arms],
            "physical_transport_arm": self.physical_transport_arm.value,
            "discrepancy_reference_arm": self.discrepancy_reference_arm.value,
            "matched_comparator_arm": self.matched_comparator_arm.value,
            "minimum_sessions": self.minimum_sessions,
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_seed": self.bootstrap_seed,
            "confidence_level": self.confidence_level,
            "minimum_off_diagonal_gain": self.minimum_off_diagonal_gain,
            "minimum_discrepancy_contrast": self.minimum_discrepancy_contrast,
            "minimum_comparator_contrast": self.minimum_comparator_contrast,
            "maximum_harmful_session_fraction": (
                self.maximum_harmful_session_fraction
            ),
            "harmful_gain_margin": self.harmful_gain_margin,
            "lower_is_better": self.lower_is_better,
            "method_frozen_before_target": self.method_frozen_before_target,
            "roster_frozen_before_target": self.roster_frozen_before_target,
            "target_outcomes_used": self.target_outcomes_used,
            "metadata": plain_json(self.metadata),
            "claim_boundary": CROSS_ACTION_TRANSPORT_CLAIM_BOUNDARY,
        }

    @property
    def protocol_id(self) -> str:
        return cast(str, content_id(self.descriptor()))


@dataclass(frozen=True, slots=True)
class SealedTransportPredictionV1:
    """One target-blind complete-belief prediction."""

    protocol_id: str
    object_session_id: str
    source_action_id: str
    target_action_id: str
    arm: TransportArm
    baseline_belief_id: str
    candidate_belief_id: str | None
    selected_belief_id: str
    disposition: PredictionDisposition
    prediction_artifact_id: str
    source_evidence_id: str
    prediction_batch_id: str
    commit_id: str
    prediction_sealed_before_target: bool
    target_outcomes_used: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "protocol_id",
            "baseline_belief_id",
            "selected_belief_id",
            "prediction_artifact_id",
            "source_evidence_id",
            "prediction_batch_id",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        if self.candidate_belief_id is not None:
            object.__setattr__(
                self,
                "candidate_belief_id",
                _digest(self.candidate_belief_id, name="candidate_belief_id"),
            )
        object.__setattr__(self, "commit_id", _commit(self.commit_id, name="commit_id"))
        for name in ("object_session_id", "source_action_id", "target_action_id"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be a nonempty literal string")
        if not isinstance(self.arm, TransportArm):
            raise TypeError("arm must be a TransportArm")
        if not isinstance(self.disposition, PredictionDisposition):
            raise TypeError("disposition must be a PredictionDisposition")
        sealed = genuine_boolean(
            self.prediction_sealed_before_target,
            name="prediction_sealed_before_target",
        )
        target_used = genuine_boolean(
            self.target_outcomes_used,
            name="target_outcomes_used",
        )
        if not sealed or target_used:
            raise ValueError("predictions must be sealed before target access")
        if self.disposition is PredictionDisposition.BASELINE_REFERENCE:
            if self.arm is not TransportArm.PHYSICAL_FALLBACK:
                raise ValueError("baseline_reference is reserved for physical fallback")
            if self.candidate_belief_id is not None:
                raise ValueError("baseline_reference cannot bind a candidate belief")
            if self.selected_belief_id != self.baseline_belief_id:
                raise ValueError("baseline_reference must select the baseline")
        elif self.disposition is PredictionDisposition.EXACT_FALLBACK:
            if self.candidate_belief_id is None:
                raise ValueError("exact_fallback must retain the rejected candidate ID")
            if self.selected_belief_id != self.baseline_belief_id:
                raise ValueError("exact_fallback must select the exact baseline belief")
        else:
            if self.candidate_belief_id is None:
                raise ValueError("candidate_selected requires a candidate belief")
            if self.selected_belief_id != self.candidate_belief_id:
                raise ValueError("candidate_selected must select the candidate belief")
        object.__setattr__(self, "prediction_sealed_before_target", sealed)
        object.__setattr__(self, "target_outcomes_used", target_used)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="prediction metadata"),
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": CROSS_ACTION_TRANSPORT_SCHEMA,
            "schema_version": CROSS_ACTION_TRANSPORT_VERSION,
            "artifact_kind": "SealedTransportPredictionV1",
            "protocol_id": self.protocol_id,
            "object_session_id": self.object_session_id,
            "source_action_id": self.source_action_id,
            "target_action_id": self.target_action_id,
            "arm": self.arm.value,
            "baseline_belief_id": self.baseline_belief_id,
            "candidate_belief_id": self.candidate_belief_id,
            "selected_belief_id": self.selected_belief_id,
            "disposition": self.disposition.value,
            "prediction_artifact_id": self.prediction_artifact_id,
            "source_evidence_id": self.source_evidence_id,
            "prediction_batch_id": self.prediction_batch_id,
            "commit_id": self.commit_id,
            "prediction_sealed_before_target": self.prediction_sealed_before_target,
            "target_outcomes_used": self.target_outcomes_used,
            "metadata": plain_json(self.metadata),
        }

    @property
    def prediction_id(self) -> str:
        return cast(str, content_id(self.descriptor()))


@dataclass(frozen=True, slots=True)
class TransportScoreRowV1:
    """One post-access score bound to a sealed prediction."""

    prediction: SealedTransportPredictionV1
    target_outcome_id: str
    target_access_attestation_id: str
    scorer_id: str
    proper_score: float
    target_side_selection_used: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.prediction, SealedTransportPredictionV1):
            raise TypeError("prediction must be a SealedTransportPredictionV1")
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
        selected = genuine_boolean(
            self.target_side_selection_used,
            name="target_side_selection_used",
        )
        if selected:
            raise ValueError("target-side model or threshold selection is forbidden")
        object.__setattr__(self, "target_side_selection_used", selected)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="score-row metadata"),
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": CROSS_ACTION_TRANSPORT_SCHEMA,
            "schema_version": CROSS_ACTION_TRANSPORT_VERSION,
            "artifact_kind": "TransportScoreRowV1",
            "prediction_id": self.prediction.prediction_id,
            "target_outcome_id": self.target_outcome_id,
            "target_access_attestation_id": self.target_access_attestation_id,
            "scorer_id": self.scorer_id,
            "proper_score": self.proper_score,
            "target_side_selection_used": self.target_side_selection_used,
            "metadata": plain_json(self.metadata),
        }

    @property
    def score_row_id(self) -> str:
        return cast(str, content_id(self.descriptor()))

__all__ = [
    "CROSS_ACTION_TRANSPORT_CLAIM_BOUNDARY",
    "CROSS_ACTION_TRANSPORT_SCHEMA",
    "CROSS_ACTION_TRANSPORT_SEMANTICS",
    "CROSS_ACTION_TRANSPORT_VERSION",
    "CrossActionProtocolV1",
    "PredictionDisposition",
    "SealedTransportPredictionV1",
    "TransportArm",
    "TransportDecision",
    "TransportScoreRowV1",
]
