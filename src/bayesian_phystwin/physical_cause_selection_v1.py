"""Source-calibrated selection among complete physical-cause beliefs.

This module makes the state-versus-discrepancy decision explicit without
pretending that predictive evidence uniquely identifies a latent physical cause.
Each candidate is a complete content-addressed belief with source-only regret
evidence. Ambiguous physical attribution resolves to a registered readout-
discrepancy belief or the exact caller-owned physical baseline.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

import numpy as np

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    literal_lower_hex,
    plain_json,
)
from ._portable_contracts import content_id
from .complete_belief_selection import ArtifactBelief

PHYSICAL_CAUSE_DECISION_SCHEMA = "bayesian_phystwin.physical_cause_decision"
PHYSICAL_CAUSE_DECISION_VERSION = 1
PHYSICAL_CAUSE_DECISION_CLAIM_BOUNDARY = (
    "Source-calibrated routing among complete beliefs only. The decision binds "
    "candidate construction, admissibility, physical-support evidence, and an "
    "upper regret bound, but it does not prove a unique physical cause, provider "
    "competence, uncertainty calibration, unseen-object transfer, deployment "
    "safety, or downstream Causal4D benefit."
)


class PhysicalCause(str, Enum):
    """Mutually exclusive interpretations offered to one routing decision."""

    BASELINE = "baseline"
    OBSERVATION_BIAS = "observation_bias"
    READOUT_DISCREPANCY = "readout_discrepancy"
    PHYSICAL_PARAMETER = "physical_parameter"
    PHYSICAL_STATE = "physical_state"


class PhysicalCauseAmbiguityFallback(str, Enum):
    """Permitted outcomes when more than one cause is statistically tied."""

    BASELINE = "baseline"
    READOUT_DISCREPANCY = "readout_discrepancy"


def _finite_real(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
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
    return result


def _sha256(value: object, *, name: str) -> str:
    return literal_lower_hex(value, name=name, lengths={64})


def _optional_sha256(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _sha256(value, name=name)


def _nonempty_literal(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty literal string")
    return value


@dataclass(frozen=True, slots=True)
class PhysicalCauseCandidateV1:
    """One complete-belief interpretation with source-only regret evidence.

    ``upper_regret`` is a source-calibrated upper bound on
    ``candidate proper score - baseline proper score``. Lower is better. A
    candidate can advance only when the bound is strictly below the negative
    policy improvement margin, so a tie always retains the baseline.
    """

    cause: PhysicalCause
    belief_id: str
    construction_id: str
    upper_regret: float
    inference_admissible: bool
    reason: str
    physical_response_id: str | None = None
    identifiability_report_id: str | None = None
    parameter_sensitivity_id: str | None = None
    bias_design_id: str | None = None
    discrepancy_model_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.cause, PhysicalCause):
            raise TypeError("cause must be a PhysicalCause")
        if self.cause is PhysicalCause.BASELINE:
            raise ValueError(
                "baseline is owned by the decision policy, not a candidate"
            )

        object.__setattr__(self, "belief_id", _sha256(self.belief_id, name="belief_id"))
        object.__setattr__(
            self,
            "construction_id",
            _sha256(self.construction_id, name="construction_id"),
        )
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
            "reason",
            _nonempty_literal(self.reason, name="reason"),
        )

        for name in (
            "physical_response_id",
            "identifiability_report_id",
            "parameter_sensitivity_id",
            "bias_design_id",
            "discrepancy_model_id",
        ):
            object.__setattr__(
                self,
                name,
                _optional_sha256(getattr(self, name), name=name),
            )

        self._validate_cause_evidence()
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="candidate metadata"),
        )

    def _validate_cause_evidence(self) -> None:
        physical = self.physical_response_id
        identifiable = self.identifiability_report_id
        parameter = self.parameter_sensitivity_id
        bias = self.bias_design_id
        discrepancy = self.discrepancy_model_id

        if self.cause is PhysicalCause.OBSERVATION_BIAS:
            if bias is None:
                raise ValueError("observation-bias candidates require bias_design_id")
            if any(
                value is not None
                for value in (physical, identifiable, parameter, discrepancy)
            ):
                raise ValueError(
                    "observation-bias candidates cannot claim physical or "
                    "discrepancy evidence"
                )
        elif self.cause is PhysicalCause.READOUT_DISCREPANCY:
            if discrepancy is None:
                raise ValueError(
                    "readout-discrepancy candidates require discrepancy_model_id"
                )
            if any(value is not None for value in (physical, identifiable, parameter)):
                raise ValueError(
                    "readout-discrepancy candidates cannot claim state or "
                    "parameter evidence"
                )
        elif self.cause is PhysicalCause.PHYSICAL_PARAMETER:
            if parameter is None or identifiable is None:
                raise ValueError(
                    "physical-parameter candidates require parameter sensitivity "
                    "and identifiability evidence"
                )
            if physical is not None or discrepancy is not None:
                raise ValueError(
                    "physical-parameter candidates cannot claim state-response or "
                    "discrepancy evidence"
                )
        elif self.cause is PhysicalCause.PHYSICAL_STATE:
            if physical is None or identifiable is None:
                raise ValueError(
                    "physical-state candidates require physical response and "
                    "identifiability evidence"
                )
            if parameter is not None or discrepancy is not None:
                raise ValueError(
                    "physical-state candidates cannot claim parameter or "
                    "discrepancy evidence"
                )
        else:  # pragma: no cover - all enum members are handled above.
            raise AssertionError("unsupported physical cause")

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": "bayesian_phystwin.physical_cause_candidate",
            "schema_version": PHYSICAL_CAUSE_DECISION_VERSION,
            "cause": self.cause.value,
            "belief_id": self.belief_id,
            "construction_id": self.construction_id,
            "upper_regret": self.upper_regret,
            "inference_admissible": self.inference_admissible,
            "reason": self.reason,
            "physical_response_id": self.physical_response_id,
            "identifiability_report_id": self.identifiability_report_id,
            "parameter_sensitivity_id": self.parameter_sensitivity_id,
            "bias_design_id": self.bias_design_id,
            "discrepancy_model_id": self.discrepancy_model_id,
            "metadata": plain_json(self.metadata),
        }

    @property
    def candidate_id(self) -> str:
        return content_id(self.descriptor())


@dataclass(frozen=True, slots=True)
class PhysicalCauseDecisionPolicyV1:
    """Frozen source-only policy for selecting or abstaining among causes."""

    baseline_belief_id: str
    common_domain_id: str
    registered_query_id: str
    source_evidence_id: str
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
            "source_evidence_id",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name=name))
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
            "schema_version": PHYSICAL_CAUSE_DECISION_VERSION,
            "baseline_belief_id": self.baseline_belief_id,
            "common_domain_id": self.common_domain_id,
            "registered_query_id": self.registered_query_id,
            "source_evidence_id": self.source_evidence_id,
            "minimum_improvement": self.minimum_improvement,
            "tie_tolerance": self.tie_tolerance,
            "ambiguity_fallback": self.ambiguity_fallback.value,
            "metadata": plain_json(self.metadata),
        }

    @property
    def policy_id(self) -> str:
        return content_id(self.descriptor())


@dataclass(frozen=True, slots=True)
class _ResolvedPhysicalCause:
    selected_cause: PhysicalCause
    selected_candidate_id: str | None
    selected_belief_id: str
    exact_baseline_fallback: bool
    ambiguity_detected: bool
    reason: str


def _canonical_candidates(
    candidates: Sequence[PhysicalCauseCandidateV1],
) -> tuple[PhysicalCauseCandidateV1, ...]:
    if isinstance(candidates, (str, bytes)):
        raise TypeError("candidates must be a sequence of PhysicalCauseCandidateV1")
    result = tuple(candidates)
    if any(not isinstance(candidate, PhysicalCauseCandidateV1) for candidate in result):
        raise TypeError("candidates must contain PhysicalCauseCandidateV1 values")
    causes = [candidate.cause for candidate in result]
    if len(causes) != len(set(causes)):
        raise ValueError("at most one candidate per physical cause is permitted")
    candidate_ids = [candidate.candidate_id for candidate in result]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate IDs must be unique")
    belief_ids = [candidate.belief_id for candidate in result]
    if len(belief_ids) != len(set(belief_ids)):
        raise ValueError("candidate belief IDs must be unique")
    return tuple(sorted(result, key=lambda candidate: candidate.cause.value))


def _resolve_physical_cause(
    policy: PhysicalCauseDecisionPolicyV1,
    candidates: tuple[PhysicalCauseCandidateV1, ...],
) -> _ResolvedPhysicalCause:
    eligible = tuple(
        candidate
        for candidate in candidates
        if candidate.inference_admissible
        and candidate.upper_regret < -policy.minimum_improvement
    )
    if not eligible:
        return _ResolvedPhysicalCause(
            selected_cause=PhysicalCause.BASELINE,
            selected_candidate_id=None,
            selected_belief_id=policy.baseline_belief_id,
            exact_baseline_fallback=True,
            ambiguity_detected=False,
            reason="no-source-supported-candidate",
        )

    best_upper_regret = min(candidate.upper_regret for candidate in eligible)
    near_best = tuple(
        candidate
        for candidate in eligible
        if candidate.upper_regret <= best_upper_regret + policy.tie_tolerance
    )
    if len(near_best) == 1:
        selected = near_best[0]
        return _ResolvedPhysicalCause(
            selected_cause=selected.cause,
            selected_candidate_id=selected.candidate_id,
            selected_belief_id=selected.belief_id,
            exact_baseline_fallback=False,
            ambiguity_detected=False,
            reason="source-supported-unique-cause",
        )

    if (
        policy.ambiguity_fallback
        is PhysicalCauseAmbiguityFallback.READOUT_DISCREPANCY
    ):
        discrepancy = next(
            (
                candidate
                for candidate in near_best
                if candidate.cause is PhysicalCause.READOUT_DISCREPANCY
            ),
            None,
        )
        if discrepancy is not None:
            return _ResolvedPhysicalCause(
                selected_cause=discrepancy.cause,
                selected_candidate_id=discrepancy.candidate_id,
                selected_belief_id=discrepancy.belief_id,
                exact_baseline_fallback=False,
                ambiguity_detected=True,
                reason="ambiguous-select-readout-discrepancy",
            )

    return _ResolvedPhysicalCause(
        selected_cause=PhysicalCause.BASELINE,
        selected_candidate_id=None,
        selected_belief_id=policy.baseline_belief_id,
        exact_baseline_fallback=True,
        ambiguity_detected=True,
        reason="ambiguous-exact-baseline-fallback",
    )


@dataclass(frozen=True, slots=True)
class PhysicalCauseDecisionV1:
    """Self-validating routing record for one complete physical-cause belief."""

    policy: PhysicalCauseDecisionPolicyV1
    candidates: tuple[PhysicalCauseCandidateV1, ...]
    selected_cause: PhysicalCause
    selected_candidate_id: str | None
    selected_belief_id: str
    exact_baseline_fallback: bool
    ambiguity_detected: bool
    reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.policy, PhysicalCauseDecisionPolicyV1):
            raise TypeError("policy must be a PhysicalCauseDecisionPolicyV1")
        candidates = _canonical_candidates(self.candidates)
        object.__setattr__(self, "candidates", candidates)
        if any(
            candidate.belief_id == self.policy.baseline_belief_id
            for candidate in candidates
        ):
            raise ValueError("candidate beliefs must be distinct from the baseline")

        if not isinstance(self.selected_cause, PhysicalCause):
            raise TypeError("selected_cause must be a PhysicalCause")
        selected_candidate_id = _optional_sha256(
            self.selected_candidate_id,
            name="selected_candidate_id",
        )
        selected_belief_id = _sha256(
            self.selected_belief_id,
            name="selected_belief_id",
        )
        exact_fallback = genuine_boolean(
            self.exact_baseline_fallback,
            name="exact_baseline_fallback",
        )
        ambiguity = genuine_boolean(
            self.ambiguity_detected,
            name="ambiguity_detected",
        )
        reason = _nonempty_literal(self.reason, name="reason")

        resolved = _resolve_physical_cause(self.policy, candidates)
        supplied = (
            self.selected_cause,
            selected_candidate_id,
            selected_belief_id,
            exact_fallback,
            ambiguity,
            reason,
        )
        expected = (
            resolved.selected_cause,
            resolved.selected_candidate_id,
            resolved.selected_belief_id,
            resolved.exact_baseline_fallback,
            resolved.ambiguity_detected,
            resolved.reason,
        )
        if supplied != expected:
            raise ValueError("physical-cause decision contradicts the frozen policy")

        object.__setattr__(self, "selected_candidate_id", selected_candidate_id)
        object.__setattr__(self, "selected_belief_id", selected_belief_id)
        object.__setattr__(self, "exact_baseline_fallback", exact_fallback)
        object.__setattr__(self, "ambiguity_detected", ambiguity)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="decision metadata"),
        )

    @property
    def policy_id(self) -> str:
        return self.policy.policy_id

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": PHYSICAL_CAUSE_DECISION_SCHEMA,
            "schema_version": PHYSICAL_CAUSE_DECISION_VERSION,
            "policy": self.policy.descriptor(),
            "policy_id": self.policy_id,
            "candidates": [candidate.descriptor() for candidate in self.candidates],
            "candidate_ids": [candidate.candidate_id for candidate in self.candidates],
            "selected_cause": self.selected_cause.value,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_belief_id": self.selected_belief_id,
            "exact_baseline_fallback": self.exact_baseline_fallback,
            "ambiguity_detected": self.ambiguity_detected,
            "reason": self.reason,
            "metadata": plain_json(self.metadata),
            "claim_boundary": PHYSICAL_CAUSE_DECISION_CLAIM_BOUNDARY,
        }

    @property
    def decision_id(self) -> str:
        return content_id(self.descriptor())

    def to_record(self) -> dict[str, Any]:
        return {**self.descriptor(), "decision_id": self.decision_id}


BeliefT = TypeVar("BeliefT", bound=ArtifactBelief)


def _belief_artifact_id(value: object, *, name: str) -> str:
    try:
        artifact_id = value.artifact_id  # type: ignore[attr-defined]
    except AttributeError as error:
        raise TypeError(f"{name} must expose artifact_id") from error
    return _sha256(artifact_id, name=f"{name}.artifact_id")


def select_physical_cause(
    baseline: BeliefT,
    candidates: Sequence[tuple[BeliefT, PhysicalCauseCandidateV1]],
    policy: PhysicalCauseDecisionPolicyV1,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[BeliefT, PhysicalCauseDecisionV1]:
    """Select one exact candidate belief or reuse the exact baseline object."""

    if not isinstance(policy, PhysicalCauseDecisionPolicyV1):
        raise TypeError("policy must be a PhysicalCauseDecisionPolicyV1")
    baseline_id = _belief_artifact_id(baseline, name="baseline")
    if baseline_id != policy.baseline_belief_id:
        raise ValueError("policy does not bind the baseline belief")
    if isinstance(candidates, (str, bytes)):
        raise TypeError("candidates must be belief/candidate pairs")

    candidate_pairs = tuple(candidates)
    specs: list[PhysicalCauseCandidateV1] = []
    beliefs_by_candidate_id: dict[str, BeliefT] = {}
    for index, item in enumerate(candidate_pairs):
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(f"candidates[{index}] must be a belief/candidate pair")
        belief, candidate = item
        if not isinstance(candidate, PhysicalCauseCandidateV1):
            raise TypeError(
                f"candidates[{index}] must contain a PhysicalCauseCandidateV1"
            )
        belief_id = _belief_artifact_id(belief, name=f"candidates[{index}].belief")
        if belief_id != candidate.belief_id:
            raise ValueError(f"candidates[{index}] belief does not match candidate")
        if belief_id == baseline_id:
            raise ValueError("candidate beliefs must be distinct from the baseline")
        specs.append(candidate)
        beliefs_by_candidate_id[candidate.candidate_id] = belief

    canonical = _canonical_candidates(specs)
    if len(beliefs_by_candidate_id) != len(canonical):
        raise ValueError("candidate IDs must be unique")
    resolved = _resolve_physical_cause(policy, canonical)
    decision = PhysicalCauseDecisionV1(
        policy=policy,
        candidates=canonical,
        selected_cause=resolved.selected_cause,
        selected_candidate_id=resolved.selected_candidate_id,
        selected_belief_id=resolved.selected_belief_id,
        exact_baseline_fallback=resolved.exact_baseline_fallback,
        ambiguity_detected=resolved.ambiguity_detected,
        reason=resolved.reason,
        metadata={} if metadata is None else metadata,
    )

    if resolved.exact_baseline_fallback:
        selected = baseline
        if selected is not baseline:  # pragma: no cover - identity invariant.
            raise AssertionError("fallback did not reuse the exact baseline object")
    else:
        candidate_id = resolved.selected_candidate_id
        if candidate_id is None:  # pragma: no cover - validated by the resolver.
            raise AssertionError("selected candidate is missing its identity")
        selected = beliefs_by_candidate_id[candidate_id]
    if selected.artifact_id != decision.selected_belief_id:
        raise AssertionError("selected belief identity differs from the decision")
    return selected, decision


__all__ = [
    "PHYSICAL_CAUSE_DECISION_CLAIM_BOUNDARY",
    "PHYSICAL_CAUSE_DECISION_SCHEMA",
    "PHYSICAL_CAUSE_DECISION_VERSION",
    "PhysicalCause",
    "PhysicalCauseAmbiguityFallback",
    "PhysicalCauseCandidateV1",
    "PhysicalCauseDecisionPolicyV1",
    "PhysicalCauseDecisionV1",
    "select_physical_cause",
]
