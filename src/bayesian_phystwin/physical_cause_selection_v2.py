"""Evidence-bound selection among complete physical-cause beliefs.

Version 2 keeps the V1 complete-belief routing semantics but replaces an
unverified caller-supplied regret scalar with a typed, content-addressed source
evidence set. Every candidate bound is tied to the same domain, physical query,
query Jacobian, grouping rule, source roster, proper score, score table, and
simultaneous interval procedure before a target-facing decision is allowed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar, cast

import numpy as np

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    genuine_integer,
    literal_lower_hex,
    plain_json,
)
from ._portable_contracts import content_id
from .complete_belief_selection import ArtifactBelief
from .physical_cause_selection_v1 import (
    PhysicalCause,
    PhysicalCauseAmbiguityFallback,
    PhysicalCauseCandidateV1,
    PhysicalCauseDecisionPolicyV1,
    PhysicalCauseDecisionV1,
    select_physical_cause,
)

PHYSICAL_CAUSE_DECISION_V2_SCHEMA = "bayesian_phystwin.physical_cause_decision"
PHYSICAL_CAUSE_DECISION_V2_VERSION = 2
PHYSICAL_CAUSE_DECISION_V2_CLAIM_BOUNDARY = (
    "Source-evidence-bound routing among complete beliefs only. The decision "
    "verifies one simultaneous candidate family against exact domain, query, "
    "Jacobian, grouping, roster, score, and interval identities, but it does "
    "not prove a unique data-generating cause, unseen-object transfer, "
    "calibrated uncertainty, deployment safety, or downstream Causal4D benefit."
)


def _sha256(value: object, *, name: str) -> str:
    return cast(str, literal_lower_hex(value, name=name, lengths={64}))


def _finite_real(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite real number") from error
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


def _confidence_level(value: object) -> float:
    result = _finite_real(value, name="confidence_level", minimum=0.0, maximum=1.0)
    if result in {0.0, 1.0}:
        raise ValueError("confidence_level must be strictly between zero and one")
    return result


def _canonical_causes(
    values: Sequence[PhysicalCause],
    *,
    name: str,
) -> tuple[PhysicalCause, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of PhysicalCause values")
    result = tuple(values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    if any(not isinstance(value, PhysicalCause) for value in result):
        raise TypeError(f"{name} must contain PhysicalCause values")
    if PhysicalCause.BASELINE in result:
        raise ValueError(f"{name} cannot include the baseline cause")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicate causes")
    return tuple(sorted(result, key=lambda value: value.value))


@dataclass(frozen=True, slots=True)
class PhysicalCauseCandidateEvidenceV2:
    """One candidate's simultaneous source-regret certificate."""

    candidate_id: str
    cause: PhysicalCause
    belief_id: str
    construction_id: str
    candidate_score_id: str
    upper_regret: float
    inference_admissible: bool
    evaluated_group_count: int
    simultaneous_bound: bool
    candidate_frozen_before_scores: bool
    target_outcomes_used: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "candidate_id",
            "belief_id",
            "construction_id",
            "candidate_score_id",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name=name))
        if not isinstance(self.cause, PhysicalCause):
            raise TypeError("cause must be a PhysicalCause")
        if self.cause is PhysicalCause.BASELINE:
            raise ValueError("baseline cannot have candidate source evidence")
        object.__setattr__(
            self,
            "upper_regret",
            _finite_real(self.upper_regret, name="upper_regret"),
        )
        object.__setattr__(
            self,
            "inference_admissible",
            genuine_boolean(self.inference_admissible, name="inference_admissible"),
        )
        object.__setattr__(
            self,
            "evaluated_group_count",
            genuine_integer(
                self.evaluated_group_count,
                name="evaluated_group_count",
                minimum=1,
            ),
        )
        simultaneous = genuine_boolean(
            self.simultaneous_bound,
            name="simultaneous_bound",
        )
        frozen = genuine_boolean(
            self.candidate_frozen_before_scores,
            name="candidate_frozen_before_scores",
        )
        target_used = genuine_boolean(
            self.target_outcomes_used,
            name="target_outcomes_used",
        )
        if not simultaneous:
            raise ValueError("candidate evidence must use a simultaneous bound")
        if not frozen:
            raise ValueError("candidate must be frozen before source scores")
        if target_used:
            raise ValueError("candidate evidence cannot use target outcomes")
        object.__setattr__(self, "simultaneous_bound", simultaneous)
        object.__setattr__(self, "candidate_frozen_before_scores", frozen)
        object.__setattr__(self, "target_outcomes_used", target_used)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="candidate evidence metadata",
            ),
        )

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": "bayesian_phystwin.physical_cause_candidate_evidence",
            "schema_version": PHYSICAL_CAUSE_DECISION_V2_VERSION,
            "candidate_id": self.candidate_id,
            "cause": self.cause.value,
            "belief_id": self.belief_id,
            "construction_id": self.construction_id,
            "candidate_score_id": self.candidate_score_id,
            "upper_regret": self.upper_regret,
            "inference_admissible": self.inference_admissible,
            "evaluated_group_count": self.evaluated_group_count,
            "simultaneous_bound": self.simultaneous_bound,
            "candidate_frozen_before_scores": self.candidate_frozen_before_scores,
            "target_outcomes_used": self.target_outcomes_used,
            "metadata": plain_json(self.metadata),
        }

    @property
    def evidence_id(self) -> str:
        return cast(str, content_id(self.descriptor()))


@dataclass(frozen=True, slots=True)
class PhysicalCauseEvidenceSetV2:
    """One jointly scored candidate family under a common source protocol."""

    common_domain_id: str
    registered_query_id: str
    query_jacobian_id: str
    grouping_rule_id: str
    source_roster_id: str
    score_definition_id: str
    source_score_table_id: str
    interval_method_id: str
    simultaneous_interval_id: str
    confidence_level: float
    source_group_count: int
    registered_candidate_causes: tuple[PhysicalCause, ...]
    candidate_evidence: tuple[PhysicalCauseCandidateEvidenceV2, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "common_domain_id",
            "registered_query_id",
            "query_jacobian_id",
            "grouping_rule_id",
            "source_roster_id",
            "score_definition_id",
            "source_score_table_id",
            "interval_method_id",
            "simultaneous_interval_id",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "confidence_level",
            _confidence_level(self.confidence_level),
        )
        source_group_count = genuine_integer(
            self.source_group_count,
            name="source_group_count",
            minimum=1,
        )
        object.__setattr__(self, "source_group_count", source_group_count)
        registered_causes = _canonical_causes(
            self.registered_candidate_causes,
            name="registered_candidate_causes",
        )
        object.__setattr__(self, "registered_candidate_causes", registered_causes)

        if isinstance(self.candidate_evidence, (str, bytes)):
            raise TypeError(
                "candidate_evidence must be a sequence of "
                "PhysicalCauseCandidateEvidenceV2 values"
            )
        evidence = tuple(self.candidate_evidence)
        if any(
            not isinstance(value, PhysicalCauseCandidateEvidenceV2)
            for value in evidence
        ):
            raise TypeError(
                "candidate_evidence must contain "
                "PhysicalCauseCandidateEvidenceV2 values"
            )
        canonical = tuple(sorted(evidence, key=lambda value: value.cause.value))
        evidence_causes = tuple(value.cause for value in canonical)
        if evidence_causes != registered_causes:
            raise ValueError(
                "candidate evidence causes must match the registered candidate family"
            )
        if any(
            value.evaluated_group_count != source_group_count for value in canonical
        ):
            raise ValueError(
                "every candidate must be evaluated on the complete source group roster"
            )
        for name, values in (
            ("candidate IDs", [value.candidate_id for value in canonical]),
            ("candidate belief IDs", [value.belief_id for value in canonical]),
            (
                "candidate construction IDs",
                [value.construction_id for value in canonical],
            ),
            ("candidate score IDs", [value.candidate_score_id for value in canonical]),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must be unique")
        object.__setattr__(self, "candidate_evidence", canonical)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="evidence set metadata"),
        )

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": "bayesian_phystwin.physical_cause_evidence_set",
            "schema_version": PHYSICAL_CAUSE_DECISION_V2_VERSION,
            "common_domain_id": self.common_domain_id,
            "registered_query_id": self.registered_query_id,
            "query_jacobian_id": self.query_jacobian_id,
            "grouping_rule_id": self.grouping_rule_id,
            "source_roster_id": self.source_roster_id,
            "score_definition_id": self.score_definition_id,
            "source_score_table_id": self.source_score_table_id,
            "interval_method_id": self.interval_method_id,
            "simultaneous_interval_id": self.simultaneous_interval_id,
            "confidence_level": self.confidence_level,
            "source_group_count": self.source_group_count,
            "registered_candidate_causes": [
                value.value for value in self.registered_candidate_causes
            ],
            "candidate_evidence": [
                value.descriptor() for value in self.candidate_evidence
            ],
            "candidate_evidence_ids": [
                value.evidence_id for value in self.candidate_evidence
            ],
            "metadata": plain_json(self.metadata),
        }

    @property
    def evidence_set_id(self) -> str:
        return cast(str, content_id(self.descriptor()))


@dataclass(frozen=True, slots=True)
class PhysicalCauseDecisionPolicyV2:
    """Target-facing policy bound to one exact simultaneous source evidence set."""

    baseline_belief_id: str
    common_domain_id: str
    registered_query_id: str
    query_jacobian_id: str
    grouping_rule_id: str
    source_roster_id: str
    score_definition_id: str
    source_score_table_id: str
    interval_method_id: str
    simultaneous_interval_id: str
    source_evidence_set_id: str
    confidence_level: float
    minimum_source_groups: int
    registered_candidate_causes: tuple[PhysicalCause, ...]
    minimum_improvement: float = 0.0
    tie_tolerance: float = 0.0
    ambiguity_fallback: PhysicalCauseAmbiguityFallback = (
        PhysicalCauseAmbiguityFallback.BASELINE
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "baseline_belief_id",
            "common_domain_id",
            "registered_query_id",
            "query_jacobian_id",
            "grouping_rule_id",
            "source_roster_id",
            "score_definition_id",
            "source_score_table_id",
            "interval_method_id",
            "simultaneous_interval_id",
            "source_evidence_set_id",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "confidence_level",
            _confidence_level(self.confidence_level),
        )
        object.__setattr__(
            self,
            "minimum_source_groups",
            genuine_integer(
                self.minimum_source_groups,
                name="minimum_source_groups",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "registered_candidate_causes",
            _canonical_causes(
                self.registered_candidate_causes,
                name="registered_candidate_causes",
            ),
        )
        object.__setattr__(
            self,
            "minimum_improvement",
            _finite_real(
                self.minimum_improvement,
                name="minimum_improvement",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "tie_tolerance",
            _finite_real(self.tie_tolerance, name="tie_tolerance", minimum=0.0),
        )
        if not isinstance(self.ambiguity_fallback, PhysicalCauseAmbiguityFallback):
            raise TypeError(
                "ambiguity_fallback must be a PhysicalCauseAmbiguityFallback"
            )
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="policy metadata"),
        )

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": "bayesian_phystwin.physical_cause_policy",
            "schema_version": PHYSICAL_CAUSE_DECISION_V2_VERSION,
            "baseline_belief_id": self.baseline_belief_id,
            "common_domain_id": self.common_domain_id,
            "registered_query_id": self.registered_query_id,
            "query_jacobian_id": self.query_jacobian_id,
            "grouping_rule_id": self.grouping_rule_id,
            "source_roster_id": self.source_roster_id,
            "score_definition_id": self.score_definition_id,
            "source_score_table_id": self.source_score_table_id,
            "interval_method_id": self.interval_method_id,
            "simultaneous_interval_id": self.simultaneous_interval_id,
            "source_evidence_set_id": self.source_evidence_set_id,
            "confidence_level": self.confidence_level,
            "minimum_source_groups": self.minimum_source_groups,
            "registered_candidate_causes": [
                value.value for value in self.registered_candidate_causes
            ],
            "minimum_improvement": self.minimum_improvement,
            "tie_tolerance": self.tie_tolerance,
            "ambiguity_fallback": self.ambiguity_fallback.value,
            "metadata": plain_json(self.metadata),
        }

    @property
    def policy_id(self) -> str:
        return cast(str, content_id(self.descriptor()))


def _validate_policy_evidence_binding(
    policy: PhysicalCauseDecisionPolicyV2,
    evidence_set: PhysicalCauseEvidenceSetV2,
) -> None:
    if policy.source_evidence_set_id != evidence_set.evidence_set_id:
        raise ValueError("policy does not bind the supplied source evidence set")
    for name in (
        "common_domain_id",
        "registered_query_id",
        "query_jacobian_id",
        "grouping_rule_id",
        "source_roster_id",
        "score_definition_id",
        "source_score_table_id",
        "interval_method_id",
        "simultaneous_interval_id",
        "confidence_level",
        "registered_candidate_causes",
    ):
        if getattr(policy, name) != getattr(evidence_set, name):
            raise ValueError(f"policy and source evidence disagree on {name}")
    if evidence_set.source_group_count < policy.minimum_source_groups:
        raise ValueError("source evidence has fewer groups than the frozen minimum")


def _validate_candidate_evidence_binding(
    candidates: Sequence[PhysicalCauseCandidateV1],
    evidence_set: PhysicalCauseEvidenceSetV2,
) -> None:
    canonical = tuple(sorted(candidates, key=lambda value: value.cause.value))
    if (
        tuple(value.cause for value in canonical)
        != evidence_set.registered_candidate_causes
    ):
        raise ValueError(
            "supplied candidates do not match the registered candidate family"
        )
    by_cause = {value.cause: value for value in evidence_set.candidate_evidence}
    for candidate in canonical:
        evidence = by_cause[candidate.cause]
        expected = (
            candidate.candidate_id,
            candidate.cause,
            candidate.belief_id,
            candidate.construction_id,
            candidate.upper_regret,
            candidate.inference_admissible,
        )
        supplied = (
            evidence.candidate_id,
            evidence.cause,
            evidence.belief_id,
            evidence.construction_id,
            evidence.upper_regret,
            evidence.inference_admissible,
        )
        if supplied != expected:
            raise ValueError(
                f"source evidence does not bind the {candidate.cause.value} candidate"
            )


def _v1_policy(
    policy: PhysicalCauseDecisionPolicyV2,
    evidence_set: PhysicalCauseEvidenceSetV2,
) -> PhysicalCauseDecisionPolicyV1:
    return PhysicalCauseDecisionPolicyV1(
        baseline_belief_id=policy.baseline_belief_id,
        common_domain_id=policy.common_domain_id,
        registered_query_id=policy.registered_query_id,
        source_evidence_id=evidence_set.evidence_set_id,
        minimum_improvement=policy.minimum_improvement,
        tie_tolerance=policy.tie_tolerance,
        ambiguity_fallback=policy.ambiguity_fallback,
        metadata={"v2_policy_id": policy.policy_id},
    )


@dataclass(frozen=True, slots=True)
class PhysicalCauseDecisionV2:
    """Self-validating V2 decision plus its exact V1 routing projection."""

    policy: PhysicalCauseDecisionPolicyV2
    source_evidence: PhysicalCauseEvidenceSetV2
    routed_decision: PhysicalCauseDecisionV1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.policy, PhysicalCauseDecisionPolicyV2):
            raise TypeError("policy must be a PhysicalCauseDecisionPolicyV2")
        if not isinstance(self.source_evidence, PhysicalCauseEvidenceSetV2):
            raise TypeError("source_evidence must be a PhysicalCauseEvidenceSetV2")
        if not isinstance(self.routed_decision, PhysicalCauseDecisionV1):
            raise TypeError("routed_decision must be a PhysicalCauseDecisionV1")
        _validate_policy_evidence_binding(self.policy, self.source_evidence)
        _validate_candidate_evidence_binding(
            self.routed_decision.candidates,
            self.source_evidence,
        )
        expected_policy = _v1_policy(self.policy, self.source_evidence)
        if self.routed_decision.policy.descriptor() != expected_policy.descriptor():
            raise ValueError("routed V1 decision does not match the V2 policy")
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="decision metadata"),
        )

    @property
    def selected_cause(self) -> PhysicalCause:
        return self.routed_decision.selected_cause

    @property
    def selected_candidate_id(self) -> str | None:
        return self.routed_decision.selected_candidate_id

    @property
    def selected_belief_id(self) -> str:
        return self.routed_decision.selected_belief_id

    @property
    def exact_baseline_fallback(self) -> bool:
        return self.routed_decision.exact_baseline_fallback

    @property
    def ambiguity_detected(self) -> bool:
        return self.routed_decision.ambiguity_detected

    @property
    def reason(self) -> str:
        return self.routed_decision.reason

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": PHYSICAL_CAUSE_DECISION_V2_SCHEMA,
            "schema_version": PHYSICAL_CAUSE_DECISION_V2_VERSION,
            "policy": self.policy.descriptor(),
            "policy_id": self.policy.policy_id,
            "source_evidence": self.source_evidence.descriptor(),
            "source_evidence_set_id": self.source_evidence.evidence_set_id,
            "routed_decision": self.routed_decision.descriptor(),
            "routed_decision_id": self.routed_decision.decision_id,
            "selected_cause": self.selected_cause.value,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_belief_id": self.selected_belief_id,
            "exact_baseline_fallback": self.exact_baseline_fallback,
            "ambiguity_detected": self.ambiguity_detected,
            "reason": self.reason,
            "metadata": plain_json(self.metadata),
            "claim_boundary": PHYSICAL_CAUSE_DECISION_V2_CLAIM_BOUNDARY,
        }

    @property
    def decision_id(self) -> str:
        return cast(str, content_id(self.descriptor()))

    def to_record(self) -> dict[str, Any]:
        return {**self.descriptor(), "decision_id": self.decision_id}


BeliefT = TypeVar("BeliefT", bound=ArtifactBelief)


def select_physical_cause_v2(
    baseline: BeliefT,
    candidates: Sequence[tuple[BeliefT, PhysicalCauseCandidateV1]],
    policy: PhysicalCauseDecisionPolicyV2,
    source_evidence: PhysicalCauseEvidenceSetV2,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[BeliefT, PhysicalCauseDecisionV2]:
    """Validate source evidence, then reuse the exact V1 routing semantics."""

    if not isinstance(policy, PhysicalCauseDecisionPolicyV2):
        raise TypeError("policy must be a PhysicalCauseDecisionPolicyV2")
    if not isinstance(source_evidence, PhysicalCauseEvidenceSetV2):
        raise TypeError("source_evidence must be a PhysicalCauseEvidenceSetV2")
    if baseline.artifact_id != policy.baseline_belief_id:
        raise ValueError("policy does not bind the baseline belief")
    if isinstance(candidates, (str, bytes)):
        raise TypeError("candidates must be belief/candidate pairs")
    pairs = tuple(candidates)
    specs: list[PhysicalCauseCandidateV1] = []
    for index, item in enumerate(pairs):
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(f"candidates[{index}] must be a belief/candidate pair")
        belief, candidate = item
        if not isinstance(candidate, PhysicalCauseCandidateV1):
            raise TypeError(
                f"candidates[{index}] must contain a PhysicalCauseCandidateV1"
            )
        if belief.artifact_id != candidate.belief_id:
            raise ValueError(f"candidates[{index}] belief does not match candidate")
        specs.append(candidate)

    _validate_policy_evidence_binding(policy, source_evidence)
    _validate_candidate_evidence_binding(specs, source_evidence)
    selected, routed = select_physical_cause(
        baseline,
        pairs,
        _v1_policy(policy, source_evidence),
        metadata={"v2_source_evidence_set_id": source_evidence.evidence_set_id},
    )
    decision = PhysicalCauseDecisionV2(
        policy=policy,
        source_evidence=source_evidence,
        routed_decision=routed,
        metadata={} if metadata is None else metadata,
    )
    if selected.artifact_id != decision.selected_belief_id:
        raise AssertionError("selected belief identity differs from the V2 decision")
    return selected, decision


__all__ = [
    "PHYSICAL_CAUSE_DECISION_V2_CLAIM_BOUNDARY",
    "PHYSICAL_CAUSE_DECISION_V2_SCHEMA",
    "PHYSICAL_CAUSE_DECISION_V2_VERSION",
    "PhysicalCauseCandidateEvidenceV2",
    "PhysicalCauseDecisionPolicyV2",
    "PhysicalCauseDecisionV2",
    "PhysicalCauseEvidenceSetV2",
    "select_physical_cause_v2",
]
