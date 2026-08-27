"""Chronological sparse-pair cross-action transport evidence.

Version 2 preserves the Cartesian v1 protocol and adds a separate contract for
causal acquisitions that expose exactly one source->target action pair per
physical session. Complete physical sessions remain the independent units.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from math import comb
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
from bayesian_phystwin_experiments.cross_action_transport_contracts_v1 import (
    PredictionDisposition,
    TransportArm,
)

CROSS_ACTION_TRANSPORT_V2_SCHEMA: Final = "bayesian_phystwin.cross_action_transport"
CROSS_ACTION_TRANSPORT_V2_VERSION: Final = 2
CROSS_ACTION_TRANSPORT_V2_SEMANTICS: Final = (
    "target-blind-chronological-sparse-pair-session-inference-v2"
)
CAUSAL4D_SLOTH_MULTI_ACTION_V1_DESIGN_SHA256: Final = (
    "6d61f2bea96af0ba04faaf3476990b58cd87e0a9c826420c254a012dec647968"
)
CROSS_ACTION_TRANSPORT_V2_CLAIM_BOUNDARY: Final = (
    "A positive result establishes bounded chronological held-out action "
    "transport for the exact frozen physical-session roster, source->target "
    "information order, query, score, candidates, guards, software stack, and "
    "environment. It does not establish reverse-direction reuse, arbitrary-action "
    "or unseen-object generalization, a unique physical cause, deployment safety, "
    "Prob4D provider competence, Causal4D intervention benefit, or state of the art."
)


class SparseTransportDecision(str, Enum):
    """Registered decision for sparse chronological transport."""

    SUPPORTED = "physical_transport_supported"
    NOT_SUPPORTED = "physical_transport_not_supported"
    INSUFFICIENT_SESSIONS = "insufficient_independent_sessions"
    INSUFFICIENT_ACCEPTED_UPDATES = "insufficient_accepted_physical_updates"


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


def _literal(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty literal string")
    return value


def _optional_literal(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _literal(value, name=name)


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
class ChronologicalSessionPairV2:
    """One preregistered source->target pair within one physical grasp session."""

    object_session_id: str
    source_execution_id: str
    target_execution_id: str
    source_action_id: str
    target_action_id: str
    contact_id: str | None = None
    stratum_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "object_session_id",
            "source_execution_id",
            "target_execution_id",
            "source_action_id",
            "target_action_id",
        ):
            object.__setattr__(self, name, _literal(getattr(self, name), name=name))
        if self.source_execution_id == self.target_execution_id:
            raise ValueError("source and target executions must be distinct")
        if self.source_action_id == self.target_action_id:
            raise ValueError("v2 requires a genuinely cross-action pair")
        object.__setattr__(
            self, "contact_id", _optional_literal(self.contact_id, name="contact_id")
        )
        object.__setattr__(
            self, "stratum_id", _optional_literal(self.stratum_id, name="stratum_id")
        )
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="session-pair metadata"),
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "object_session_id": self.object_session_id,
            "source_execution_id": self.source_execution_id,
            "target_execution_id": self.target_execution_id,
            "source_action_id": self.source_action_id,
            "target_action_id": self.target_action_id,
            "contact_id": self.contact_id,
            "stratum_id": self.stratum_id,
            "metadata": plain_json(self.metadata),
        }

    @property
    def information_order_id(self) -> str:
        return cast(str, content_id({"chronological_pair_v2": self.descriptor()}))


@dataclass(frozen=True, slots=True)
class CrossActionProtocolV2:
    """Target-closed sparse-pair protocol for chronological action transport."""

    causal4d_design_id: str
    development_roster_id: str
    calibration_roster_id: str
    target_roster_id: str
    query_id: str
    query_jacobian_id: str
    score_definition_id: str
    grouping_rule_id: str
    interval_method_id: str
    harm_interval_method_id: str
    target_access_policy_id: str
    technical_failure_policy_id: str
    model_stack_id: str
    numerical_environment_id: str
    candidate_family_id: str
    support_policy_id: str
    identifiability_policy_id: str
    multi_action_identifiability_policy_id: str
    estimability_policy_id: str
    guard_policy_id: str
    session_pairs: tuple[ChronologicalSessionPairV2, ...]
    registered_arms: tuple[TransportArm, ...]
    physical_transport_arm: TransportArm
    discrepancy_reference_arm: TransportArm
    matched_comparator_arm: TransportArm
    minimum_sessions: int
    minimum_accepted_physical_sessions: int
    bootstrap_replicates: int
    bootstrap_seed: int
    confidence_level: float
    minimum_transport_gain: float
    minimum_discrepancy_contrast: float
    minimum_comparator_contrast: float
    maximum_harmful_accepted_fraction: float
    harmful_gain_margin: float = 0.0
    lower_is_better: bool = True
    method_frozen_before_target: bool = True
    roster_frozen_before_target: bool = True
    target_outcomes_used: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "causal4d_design_id",
            "development_roster_id",
            "calibration_roster_id",
            "target_roster_id",
            "query_id",
            "query_jacobian_id",
            "score_definition_id",
            "grouping_rule_id",
            "interval_method_id",
            "harm_interval_method_id",
            "target_access_policy_id",
            "technical_failure_policy_id",
            "model_stack_id",
            "numerical_environment_id",
            "candidate_family_id",
            "support_policy_id",
            "identifiability_policy_id",
            "multi_action_identifiability_policy_id",
            "estimability_policy_id",
            "guard_policy_id",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))

        pairs = tuple(self.session_pairs)
        if len(pairs) < 2 or any(
            not isinstance(pair, ChronologicalSessionPairV2) for pair in pairs
        ):
            raise TypeError(
                "session_pairs must contain at least two "
                "ChronologicalSessionPairV2 values"
            )
        pairs = tuple(sorted(pairs, key=lambda pair: pair.object_session_id))
        sessions = [pair.object_session_id for pair in pairs]
        if len(sessions) != len(set(sessions)):
            raise ValueError("each physical session must appear exactly once")
        executions = [
            execution
            for pair in pairs
            for execution in (pair.source_execution_id, pair.target_execution_id)
        ]
        if len(executions) != len(set(executions)):
            raise ValueError(
                "each execution must appear in exactly one chronological pair"
            )
        object.__setattr__(self, "session_pairs", pairs)

        arms = _arms(self.registered_arms)
        object.__setattr__(self, "registered_arms", arms)
        if TransportArm.PHYSICAL_FALLBACK not in arms:
            raise ValueError("the physical fallback arm is mandatory")
        for name in (
            "physical_transport_arm",
            "discrepancy_reference_arm",
            "matched_comparator_arm",
        ):
            arm = getattr(self, name)
            if not isinstance(arm, TransportArm) or arm not in arms:
                raise ValueError(f"{name} must be one registered arm")
        if self.physical_transport_arm in {
            TransportArm.PHYSICAL_FALLBACK,
            self.discrepancy_reference_arm,
            self.matched_comparator_arm,
        }:
            raise ValueError("physical_transport_arm must be a distinct candidate")
        if TransportArm.PHYSICAL_FALLBACK in {
            self.discrepancy_reference_arm,
            self.matched_comparator_arm,
        }:
            raise ValueError("reference arms cannot be the physical fallback")
        if self.discrepancy_reference_arm is self.matched_comparator_arm:
            raise ValueError("discrepancy and comparator arms must be distinct")

        minimum_sessions = genuine_integer(
            self.minimum_sessions, name="minimum_sessions", minimum=2
        )
        if minimum_sessions > len(pairs):
            raise ValueError("minimum_sessions cannot exceed the frozen session roster")
        object.__setattr__(self, "minimum_sessions", minimum_sessions)
        minimum_accepted = genuine_integer(
            self.minimum_accepted_physical_sessions,
            name="minimum_accepted_physical_sessions",
            minimum=1,
        )
        if minimum_accepted > len(pairs):
            raise ValueError(
                "minimum_accepted_physical_sessions cannot exceed the frozen roster"
            )
        object.__setattr__(self, "minimum_accepted_physical_sessions", minimum_accepted)
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
            "minimum_transport_gain",
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
            "maximum_harmful_accepted_fraction",
            _finite(
                self.maximum_harmful_accepted_fraction,
                name="maximum_harmful_accepted_fraction",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        if (
            _clopper_pearson_upper(
                0,
                len(pairs),
                confidence=self.confidence_level,
            )
            > self.maximum_harmful_accepted_fraction
        ):
            raise ValueError(
                "harm cap is impossible for the frozen session roster "
                "at the registered confidence level"
            )
        lower_is_better = genuine_boolean(self.lower_is_better, name="lower_is_better")
        if not lower_is_better:
            raise ValueError("v2 requires a lower-is-better registered score")
        object.__setattr__(self, "lower_is_better", lower_is_better)
        frozen = genuine_boolean(
            self.method_frozen_before_target, name="method_frozen_before_target"
        )
        roster_frozen = genuine_boolean(
            self.roster_frozen_before_target, name="roster_frozen_before_target"
        )
        target_used = genuine_boolean(
            self.target_outcomes_used, name="target_outcomes_used"
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
    def target_session_ids(self) -> tuple[str, ...]:
        return tuple(pair.object_session_id for pair in self.session_pairs)

    @property
    def pair_by_session(self) -> dict[str, ChronologicalSessionPairV2]:
        return {pair.object_session_id: pair for pair in self.session_pairs}

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": CROSS_ACTION_TRANSPORT_V2_SCHEMA,
            "schema_version": CROSS_ACTION_TRANSPORT_V2_VERSION,
            "artifact_kind": "CrossActionProtocolV2",
            "semantics": CROSS_ACTION_TRANSPORT_V2_SEMANTICS,
            "causal4d_design_id": self.causal4d_design_id,
            "development_roster_id": self.development_roster_id,
            "calibration_roster_id": self.calibration_roster_id,
            "target_roster_id": self.target_roster_id,
            "query_id": self.query_id,
            "query_jacobian_id": self.query_jacobian_id,
            "score_definition_id": self.score_definition_id,
            "grouping_rule_id": self.grouping_rule_id,
            "interval_method_id": self.interval_method_id,
            "harm_interval_method_id": self.harm_interval_method_id,
            "target_access_policy_id": self.target_access_policy_id,
            "technical_failure_policy_id": self.technical_failure_policy_id,
            "model_stack_id": self.model_stack_id,
            "numerical_environment_id": self.numerical_environment_id,
            "candidate_family_id": self.candidate_family_id,
            "support_policy_id": self.support_policy_id,
            "identifiability_policy_id": self.identifiability_policy_id,
            "multi_action_identifiability_policy_id": (
                self.multi_action_identifiability_policy_id
            ),
            "estimability_policy_id": self.estimability_policy_id,
            "guard_policy_id": self.guard_policy_id,
            "session_pairs": [
                {
                    **pair.descriptor(),
                    "information_order_id": pair.information_order_id,
                }
                for pair in self.session_pairs
            ],
            "registered_arms": [arm.value for arm in self.registered_arms],
            "physical_transport_arm": self.physical_transport_arm.value,
            "discrepancy_reference_arm": self.discrepancy_reference_arm.value,
            "matched_comparator_arm": self.matched_comparator_arm.value,
            "minimum_sessions": self.minimum_sessions,
            "minimum_accepted_physical_sessions": (
                self.minimum_accepted_physical_sessions
            ),
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_seed": self.bootstrap_seed,
            "confidence_level": self.confidence_level,
            "minimum_transport_gain": self.minimum_transport_gain,
            "minimum_discrepancy_contrast": self.minimum_discrepancy_contrast,
            "minimum_comparator_contrast": self.minimum_comparator_contrast,
            "maximum_harmful_accepted_fraction": (
                self.maximum_harmful_accepted_fraction
            ),
            "harmful_gain_margin": self.harmful_gain_margin,
            "lower_is_better": self.lower_is_better,
            "method_frozen_before_target": self.method_frozen_before_target,
            "roster_frozen_before_target": self.roster_frozen_before_target,
            "target_outcomes_used": self.target_outcomes_used,
            "metadata": plain_json(self.metadata),
            "claim_boundary": CROSS_ACTION_TRANSPORT_V2_CLAIM_BOUNDARY,
        }

    @property
    def protocol_id(self) -> str:
        return cast(str, content_id(self.descriptor()))


@dataclass(frozen=True, slots=True)
class SealedTransportPredictionV2:
    """One target-blind belief prediction for one registered chronological pair."""

    protocol_id: str
    information_order_id: str
    object_session_id: str
    source_execution_id: str
    target_execution_id: str
    source_action_id: str
    target_action_id: str
    arm: TransportArm
    baseline_belief_id: str
    candidate_belief_id: str | None
    selected_belief_id: str
    disposition: PredictionDisposition
    prediction_artifact_id: str
    source_evidence_id: str
    admission_evidence_id: str
    prediction_batch_id: str
    commit_id: str
    prediction_sealed_before_target: bool
    target_outcomes_used: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "protocol_id",
            "information_order_id",
            "baseline_belief_id",
            "selected_belief_id",
            "prediction_artifact_id",
            "source_evidence_id",
            "admission_evidence_id",
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
        for name in (
            "object_session_id",
            "source_execution_id",
            "target_execution_id",
            "source_action_id",
            "target_action_id",
        ):
            object.__setattr__(self, name, _literal(getattr(self, name), name=name))
        if self.source_execution_id == self.target_execution_id:
            raise ValueError("source and target executions must be distinct")
        if self.source_action_id == self.target_action_id:
            raise ValueError("v2 requires a genuinely cross-action prediction")
        if not isinstance(self.arm, TransportArm):
            raise TypeError("arm must be a TransportArm")
        if not isinstance(self.disposition, PredictionDisposition):
            raise TypeError("disposition must be a PredictionDisposition")
        sealed = genuine_boolean(
            self.prediction_sealed_before_target,
            name="prediction_sealed_before_target",
        )
        target_used = genuine_boolean(
            self.target_outcomes_used, name="target_outcomes_used"
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
            "schema": CROSS_ACTION_TRANSPORT_V2_SCHEMA,
            "schema_version": CROSS_ACTION_TRANSPORT_V2_VERSION,
            "artifact_kind": "SealedTransportPredictionV2",
            "protocol_id": self.protocol_id,
            "information_order_id": self.information_order_id,
            "object_session_id": self.object_session_id,
            "source_execution_id": self.source_execution_id,
            "target_execution_id": self.target_execution_id,
            "source_action_id": self.source_action_id,
            "target_action_id": self.target_action_id,
            "arm": self.arm.value,
            "baseline_belief_id": self.baseline_belief_id,
            "candidate_belief_id": self.candidate_belief_id,
            "selected_belief_id": self.selected_belief_id,
            "disposition": self.disposition.value,
            "prediction_artifact_id": self.prediction_artifact_id,
            "source_evidence_id": self.source_evidence_id,
            "admission_evidence_id": self.admission_evidence_id,
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
class TransportScoreRowV2:
    """One post-access score bound to one sealed chronological prediction."""

    prediction: SealedTransportPredictionV2
    target_outcome_id: str
    target_access_attestation_id: str
    scorer_id: str
    proper_score: float
    target_side_selection_used: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.prediction, SealedTransportPredictionV2):
            raise TypeError("prediction must be a SealedTransportPredictionV2")
        for name in (
            "target_outcome_id",
            "target_access_attestation_id",
            "scorer_id",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        object.__setattr__(
            self, "proper_score", _finite(self.proper_score, name="proper_score")
        )
        selected = genuine_boolean(
            self.target_side_selection_used, name="target_side_selection_used"
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
            "schema": CROSS_ACTION_TRANSPORT_V2_SCHEMA,
            "schema_version": CROSS_ACTION_TRANSPORT_V2_VERSION,
            "artifact_kind": "TransportScoreRowV2",
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


@dataclass(frozen=True, slots=True)
class ArmTransportSummaryV2:
    """Session-level summary with accepted-update harm accounting."""

    arm: TransportArm
    mean_gain: float
    gain_interval: tuple[float, float]
    win_sessions: int
    scored_sessions: int
    selected_sessions: int
    fallback_sessions: int
    harmful_accepted_sessions: int
    harmful_accepted_fraction: float | None
    harmful_accepted_fraction_upper: float

    def descriptor(self) -> dict[str, object]:
        return {
            "arm": self.arm.value,
            "mean_gain": self.mean_gain,
            "gain_interval": list(self.gain_interval),
            "win_sessions": self.win_sessions,
            "scored_sessions": self.scored_sessions,
            "selected_sessions": self.selected_sessions,
            "fallback_sessions": self.fallback_sessions,
            "harmful_accepted_sessions": self.harmful_accepted_sessions,
            "harmful_accepted_fraction": self.harmful_accepted_fraction,
            "harmful_accepted_fraction_upper": self.harmful_accepted_fraction_upper,
        }


def _interval(
    values: np.ndarray,
    *,
    replicates: int,
    seed: int,
    confidence: float,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(replicates, values.size))
    means = np.mean(values[indices], axis=1)
    alpha = 0.5 * (1.0 - confidence)
    lower, upper = np.quantile(means, [alpha, 1.0 - alpha], method="linear")
    return float(lower), float(upper)


def _binomial_cdf(events: int, total: int, probability: float) -> float:
    return float(
        sum(
            comb(total, index)
            * probability**index
            * (1.0 - probability) ** (total - index)
            for index in range(events + 1)
        )
    )


def _clopper_pearson_upper(
    events: int,
    total: int,
    *,
    confidence: float,
) -> float:
    """One-sided exact upper confidence bound for a Bernoulli event rate."""

    if total <= 0:
        return 1.0
    if events < 0 or events > total:
        raise ValueError("harm counts must satisfy 0 <= events <= total")
    if events == total:
        return 1.0
    alpha = 1.0 - confidence
    low = events / total
    high = 1.0
    for _ in range(100):
        middle = 0.5 * (low + high)
        if _binomial_cdf(events, total, middle) > alpha:
            low = middle
        else:
            high = middle
    return high


def _seed(base: int, stream: int) -> int:
    return int(np.random.SeedSequence([base, stream]).generate_state(1)[0])


@dataclass(frozen=True, slots=True)
class CrossActionTransportResultV2:
    """Evaluate a complete sparse chronological score table."""

    protocol: CrossActionProtocolV2
    score_rows: tuple[TransportScoreRowV2, ...]
    target_accounting_id: str
    excluded_session_ids: tuple[str, ...] = ()
    technical_failure_session_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    independent_session_count: int = field(init=False)
    arm_summaries: tuple[ArmTransportSummaryV2, ...] = field(init=False)
    discrepancy_contrast: float | None = field(init=False)
    discrepancy_contrast_interval: tuple[float, float] | None = field(init=False)
    comparator_contrast: float | None = field(init=False)
    comparator_contrast_interval: tuple[float, float] | None = field(init=False)
    decision: SparseTransportDecision = field(init=False)
    result_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.protocol, CrossActionProtocolV2):
            raise TypeError("protocol must be a CrossActionProtocolV2")
        target_accounting_id = _digest(
            self.target_accounting_id, name="target_accounting_id"
        )
        excluded = _optional_labels(
            self.excluded_session_ids, name="excluded_session_ids"
        )
        technical = _optional_labels(
            self.technical_failure_session_ids,
            name="technical_failure_session_ids",
        )
        if set(excluded) & set(technical):
            raise ValueError("excluded and technical-failure sessions must be disjoint")

        rows = tuple(self.score_rows)
        if any(not isinstance(row, TransportScoreRowV2) for row in rows):
            raise TypeError("score_rows must contain TransportScoreRowV2 values")
        rows = tuple(
            sorted(
                rows,
                key=lambda row: (
                    row.prediction.object_session_id,
                    row.prediction.arm.value,
                ),
            )
        )
        if len({row.score_row_id for row in rows}) != len(rows):
            raise ValueError("score rows must be unique")
        if any(row.prediction.protocol_id != self.protocol.protocol_id for row in rows):
            raise ValueError("every prediction must bind the exact protocol")
        if rows:
            if len({row.prediction.prediction_batch_id for row in rows}) != 1:
                raise ValueError("one sealed prediction batch is required")
            if len({row.target_access_attestation_id for row in rows}) != 1:
                raise ValueError("one target-access attestation is required")
            if len({row.scorer_id for row in rows}) != 1:
                raise ValueError("one frozen scorer is required")
            if len({row.prediction.commit_id for row in rows}) != 1:
                raise ValueError("one exact BayesianPhysTwin revision is required")

        by_session: dict[str, dict[TransportArm, TransportScoreRowV2]] = {}
        pair_by_session = self.protocol.pair_by_session
        for row in rows:
            prediction = row.prediction
            pair = pair_by_session.get(prediction.object_session_id)
            if pair is None:
                raise ValueError("score row uses an unregistered physical session")
            expected_identity = (
                pair.information_order_id,
                pair.source_execution_id,
                pair.target_execution_id,
                pair.source_action_id,
                pair.target_action_id,
            )
            observed_identity = (
                prediction.information_order_id,
                prediction.source_execution_id,
                prediction.target_execution_id,
                prediction.source_action_id,
                prediction.target_action_id,
            )
            if observed_identity != expected_identity:
                raise ValueError(
                    "prediction must preserve the exact registered "
                    "source->target chronology"
                )
            if prediction.arm not in self.protocol.registered_arms:
                raise ValueError("prediction uses an unregistered arm")
            arm_rows = by_session.setdefault(prediction.object_session_id, {})
            if prediction.arm in arm_rows:
                raise ValueError("duplicate arm within one physical session")
            arm_rows[prediction.arm] = row

        scored_sessions = tuple(sorted(by_session))
        accounted = set(scored_sessions) | set(excluded) | set(technical)
        if (
            set(scored_sessions) & set(excluded)
            or set(scored_sessions) & set(technical)
            or accounted != set(self.protocol.target_session_ids)
        ):
            raise ValueError(
                "scored, excluded, and technical-failure accounting must cover "
                "the frozen sparse session roster exactly once"
            )

        expected_arms = set(self.protocol.registered_arms)
        for session in scored_sessions:
            arm_rows = by_session[session]
            if set(arm_rows) != expected_arms:
                raise ValueError(
                    "every scored physical session must contain every registered arm"
                )
            baseline_row = arm_rows[TransportArm.PHYSICAL_FALLBACK]
            if (
                baseline_row.prediction.disposition
                is not PredictionDisposition.BASELINE_REFERENCE
            ):
                raise ValueError("physical fallback must be the baseline reference")
            if len({row.target_outcome_id for row in arm_rows.values()}) != 1:
                raise ValueError("all arms in one session must score the same target")
            if (
                len({row.prediction.baseline_belief_id for row in arm_rows.values()})
                != 1
            ):
                raise ValueError("all arms in one session must share one baseline")
            for arm, row in arm_rows.items():
                if (
                    arm is not TransportArm.PHYSICAL_FALLBACK
                    and row.prediction.disposition
                    is PredictionDisposition.EXACT_FALLBACK
                    and row.proper_score != baseline_row.proper_score
                ):
                    raise ValueError(
                        "exact-fallback predictions must score identically to "
                        "physical fallback"
                    )

        summaries: list[ArmTransportSummaryV2] = []
        session_gain: dict[TransportArm, np.ndarray] = {}
        if scored_sessions:
            for stream, arm in enumerate(
                sorted(
                    (
                        candidate
                        for candidate in self.protocol.registered_arms
                        if candidate is not TransportArm.PHYSICAL_FALLBACK
                    ),
                    key=lambda value: value.value,
                )
            ):
                values = np.asarray(
                    [
                        by_session[session][TransportArm.PHYSICAL_FALLBACK].proper_score
                        - by_session[session][arm].proper_score
                        for session in scored_sessions
                    ],
                    dtype=np.float64,
                )
                session_gain[arm] = values
                dispositions = [
                    by_session[session][arm].prediction.disposition
                    for session in scored_sessions
                ]
                selected_mask = np.asarray(
                    [
                        disposition is PredictionDisposition.CANDIDATE_SELECTED
                        for disposition in dispositions
                    ],
                    dtype=bool,
                )
                fallback_count = sum(
                    disposition is PredictionDisposition.EXACT_FALLBACK
                    for disposition in dispositions
                )
                harmful_selected = int(
                    np.count_nonzero(
                        selected_mask & (values < -self.protocol.harmful_gain_margin)
                    )
                )
                selected_count = int(np.count_nonzero(selected_mask))
                harmful_fraction = (
                    harmful_selected / selected_count if selected_count else None
                )
                summaries.append(
                    ArmTransportSummaryV2(
                        arm=arm,
                        mean_gain=float(np.mean(values)),
                        gain_interval=_interval(
                            values,
                            replicates=self.protocol.bootstrap_replicates,
                            seed=_seed(self.protocol.bootstrap_seed, stream),
                            confidence=self.protocol.confidence_level,
                        ),
                        win_sessions=int(np.count_nonzero(values > 0.0)),
                        scored_sessions=len(scored_sessions),
                        selected_sessions=selected_count,
                        fallback_sessions=fallback_count,
                        harmful_accepted_sessions=harmful_selected,
                        harmful_accepted_fraction=harmful_fraction,
                        harmful_accepted_fraction_upper=_clopper_pearson_upper(
                            harmful_selected,
                            selected_count,
                            confidence=self.protocol.confidence_level,
                        ),
                    )
                )

        discrepancy_contrast: float | None = None
        discrepancy_interval: tuple[float, float] | None = None
        comparator_contrast: float | None = None
        comparator_interval: tuple[float, float] | None = None
        if scored_sessions:
            physical = session_gain[self.protocol.physical_transport_arm]
            discrepancy = session_gain[self.protocol.discrepancy_reference_arm]
            comparator = session_gain[self.protocol.matched_comparator_arm]
            discrepancy_values = physical - discrepancy
            comparator_values = physical - comparator
            discrepancy_contrast = float(np.mean(discrepancy_values))
            discrepancy_interval = _interval(
                discrepancy_values,
                replicates=self.protocol.bootstrap_replicates,
                seed=_seed(self.protocol.bootstrap_seed, 2001),
                confidence=self.protocol.confidence_level,
            )
            comparator_contrast = float(np.mean(comparator_values))
            comparator_interval = _interval(
                comparator_values,
                replicates=self.protocol.bootstrap_replicates,
                seed=_seed(self.protocol.bootstrap_seed, 2002),
                confidence=self.protocol.confidence_level,
            )

        if len(scored_sessions) < self.protocol.minimum_sessions:
            decision = SparseTransportDecision.INSUFFICIENT_SESSIONS
        else:
            physical_summary = next(
                summary
                for summary in summaries
                if summary.arm is self.protocol.physical_transport_arm
            )
            if (
                physical_summary.selected_sessions
                < self.protocol.minimum_accepted_physical_sessions
            ):
                decision = SparseTransportDecision.INSUFFICIENT_ACCEPTED_UPDATES
            elif (
                physical_summary.gain_interval[0] > self.protocol.minimum_transport_gain
                and discrepancy_interval is not None
                and discrepancy_interval[0] > self.protocol.minimum_discrepancy_contrast
                and comparator_interval is not None
                and comparator_interval[0] > self.protocol.minimum_comparator_contrast
                and physical_summary.harmful_accepted_fraction_upper
                <= self.protocol.maximum_harmful_accepted_fraction
            ):
                decision = SparseTransportDecision.SUPPORTED
            else:
                decision = SparseTransportDecision.NOT_SUPPORTED

        metadata = frozen_finite_json_mapping(self.metadata, name="result metadata")
        object.__setattr__(self, "score_rows", rows)
        object.__setattr__(self, "target_accounting_id", target_accounting_id)
        object.__setattr__(self, "excluded_session_ids", excluded)
        object.__setattr__(self, "technical_failure_session_ids", technical)
        object.__setattr__(self, "independent_session_count", len(scored_sessions))
        object.__setattr__(self, "arm_summaries", tuple(summaries))
        object.__setattr__(self, "discrepancy_contrast", discrepancy_contrast)
        object.__setattr__(self, "discrepancy_contrast_interval", discrepancy_interval)
        object.__setattr__(self, "comparator_contrast", comparator_contrast)
        object.__setattr__(self, "comparator_contrast_interval", comparator_interval)
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "result_id", cast(str, content_id(self.descriptor())))

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": CROSS_ACTION_TRANSPORT_V2_SCHEMA,
            "schema_version": CROSS_ACTION_TRANSPORT_V2_VERSION,
            "artifact_kind": "CrossActionTransportResultV2",
            "protocol_id": self.protocol.protocol_id,
            "target_accounting_id": self.target_accounting_id,
            "excluded_session_ids": list(self.excluded_session_ids),
            "technical_failure_session_ids": list(self.technical_failure_session_ids),
            "score_row_ids": [row.score_row_id for row in self.score_rows],
            "independent_session_count": self.independent_session_count,
            "arm_summaries": [summary.descriptor() for summary in self.arm_summaries],
            "discrepancy_contrast": self.discrepancy_contrast,
            "discrepancy_contrast_interval": (
                None
                if self.discrepancy_contrast_interval is None
                else list(self.discrepancy_contrast_interval)
            ),
            "comparator_contrast": self.comparator_contrast,
            "comparator_contrast_interval": (
                None
                if self.comparator_contrast_interval is None
                else list(self.comparator_contrast_interval)
            ),
            "decision": self.decision.value,
            "metadata": plain_json(self.metadata),
            "claim_boundary": CROSS_ACTION_TRANSPORT_V2_CLAIM_BOUNDARY,
        }

    @property
    def supports_physical_transport(self) -> bool:
        return self.decision is SparseTransportDecision.SUPPORTED

    def summary(self) -> dict[str, object]:
        return {
            "result_id": self.result_id,
            "protocol_id": self.protocol.protocol_id,
            "decision": self.decision.value,
            "supports_physical_transport": self.supports_physical_transport,
            "independent_session_count": self.independent_session_count,
            "excluded_session_count": len(self.excluded_session_ids),
            "technical_failure_session_count": len(self.technical_failure_session_ids),
            "discrepancy_contrast": self.discrepancy_contrast,
            "discrepancy_contrast_interval": (
                None
                if self.discrepancy_contrast_interval is None
                else list(self.discrepancy_contrast_interval)
            ),
            "comparator_contrast": self.comparator_contrast,
            "comparator_contrast_interval": (
                None
                if self.comparator_contrast_interval is None
                else list(self.comparator_contrast_interval)
            ),
            "arm_summaries": [summary.descriptor() for summary in self.arm_summaries],
            "claim_boundary": CROSS_ACTION_TRANSPORT_V2_CLAIM_BOUNDARY,
        }


__all__ = [
    "ArmTransportSummaryV2",
    "CAUSAL4D_SLOTH_MULTI_ACTION_V1_DESIGN_SHA256",
    "CROSS_ACTION_TRANSPORT_V2_CLAIM_BOUNDARY",
    "CROSS_ACTION_TRANSPORT_V2_SCHEMA",
    "CROSS_ACTION_TRANSPORT_V2_SEMANTICS",
    "CROSS_ACTION_TRANSPORT_V2_VERSION",
    "ChronologicalSessionPairV2",
    "CrossActionProtocolV2",
    "CrossActionTransportResultV2",
    "PredictionDisposition",
    "SealedTransportPredictionV2",
    "SparseTransportDecision",
    "TransportArm",
    "TransportScoreRowV2",
]
