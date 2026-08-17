"""Query-specific admission for candidate physical-twin belief updates.

The certificate composes provider competence, query calibration, identifiability,
source-frozen regret, and information-gain evidence. Rejection selects the exact
baseline belief rather than a reconstructed approximation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Final

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    plain_json,
)
from ._portable_contracts import content_id, sha256_digest

QUERY_UPDATE_ADMISSION_SCHEMA: Final = "bayesian_phystwin.query_update_admission"
QUERY_UPDATE_ADMISSION_VERSION: Final = 1
QUERY_UPDATE_ADMISSION_CLAIM_BOUNDARY: Final = (
    "Software composition and exact-fallback evidence only. Authorization does "
    "not by itself establish provider competence, fresh-object physical benefit, "
    "calibrated deployment uncertainty, Causal4D intervention benefit, safety, "
    "or state of the art."
)
_AUTHORIZED_REASON: Final = "query-update-authorized"


def _canonical_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    return value


def _finite_real(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def _canonical_reasons(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("reasons must be a sequence of canonical strings")
    result = tuple(
        _canonical_text(value, name=f"reasons[{index}]")
        for index, value in enumerate(values)
    )
    if not result:
        raise ValueError("reasons must not be empty")
    if len(set(result)) != len(result):
        raise ValueError("reasons must not contain duplicates")
    return result


@dataclass(frozen=True, slots=True)
class QueryUpdateAdmissionPolicyV1:
    """Frozen thresholds for one query-specific update decision."""

    minimum_identifiable_fraction: float = 0.5
    maximum_regret_upper_bound: float = 0.0
    minimum_expected_information_gain: float = 0.0
    require_provider_competence: bool = True
    require_query_calibration: bool = True
    numerical_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "minimum_identifiable_fraction",
            _finite_real(
                self.minimum_identifiable_fraction,
                name="minimum_identifiable_fraction",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "maximum_regret_upper_bound",
            _finite_real(
                self.maximum_regret_upper_bound,
                name="maximum_regret_upper_bound",
            ),
        )
        object.__setattr__(
            self,
            "minimum_expected_information_gain",
            _finite_real(
                self.minimum_expected_information_gain,
                name="minimum_expected_information_gain",
                minimum=0.0,
            ),
        )
        for name in (
            "require_provider_competence",
            "require_query_calibration",
        ):
            object.__setattr__(
                self,
                name,
                genuine_boolean(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "numerical_tolerance",
            _finite_real(
                self.numerical_tolerance,
                name="numerical_tolerance",
                minimum=0.0,
            ),
        )

    @property
    def policy_id(self) -> str:
        return content_id(self.descriptor())

    def descriptor(self) -> dict[str, object]:
        return {
            "minimum_identifiable_fraction": self.minimum_identifiable_fraction,
            "maximum_regret_upper_bound": self.maximum_regret_upper_bound,
            "minimum_expected_information_gain": (
                self.minimum_expected_information_gain
            ),
            "require_provider_competence": self.require_provider_competence,
            "require_query_calibration": self.require_query_calibration,
            "numerical_tolerance": self.numerical_tolerance,
        }


@dataclass(frozen=True, slots=True)
class QueryUpdateEvidenceV1:
    """Source-frozen evidence supplied to the query admission policy."""

    physical_query_id: str
    baseline_belief_id: str
    candidate_belief_id: str
    fallback_belief_id: str
    provider_decision_id: str
    query_calibration_id: str
    identifiability_diagnostic_id: str
    regret_evidence_id: str
    information_gain_evidence_id: str
    provider_competence_passed: bool
    query_calibration_passed: bool
    identifiable_fraction: float
    regret_upper_bound: float
    expected_information_gain: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "physical_query_id",
            "baseline_belief_id",
            "candidate_belief_id",
            "fallback_belief_id",
            "provider_decision_id",
            "query_calibration_id",
            "identifiability_diagnostic_id",
            "regret_evidence_id",
            "information_gain_evidence_id",
        ):
            object.__setattr__(
                self,
                name,
                sha256_digest(getattr(self, name), name=name),
            )
        if self.candidate_belief_id == self.baseline_belief_id:
            raise ValueError("candidate_belief_id must differ from baseline_belief_id")
        if self.fallback_belief_id != self.baseline_belief_id:
            raise ValueError("fallback_belief_id must equal baseline_belief_id")
        for name in (
            "provider_competence_passed",
            "query_calibration_passed",
        ):
            object.__setattr__(
                self,
                name,
                genuine_boolean(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "identifiable_fraction",
            _finite_real(
                self.identifiable_fraction,
                name="identifiable_fraction",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "regret_upper_bound",
            _finite_real(
                self.regret_upper_bound,
                name="regret_upper_bound",
            ),
        )
        object.__setattr__(
            self,
            "expected_information_gain",
            _finite_real(
                self.expected_information_gain,
                name="expected_information_gain",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="query update evidence metadata",
            ),
        )
        expected = content_id(self.descriptor())
        if self.artifact_id is not None:
            supplied = sha256_digest(self.artifact_id, name="artifact_id")
            if supplied != expected:
                raise ValueError("artifact_id does not match query update evidence")
        object.__setattr__(self, "artifact_id", expected)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": "bayesian_phystwin.query_update_evidence",
            "schema_version": 1,
            "physical_query_id": self.physical_query_id,
            "baseline_belief_id": self.baseline_belief_id,
            "candidate_belief_id": self.candidate_belief_id,
            "fallback_belief_id": self.fallback_belief_id,
            "provider_decision_id": self.provider_decision_id,
            "query_calibration_id": self.query_calibration_id,
            "identifiability_diagnostic_id": self.identifiability_diagnostic_id,
            "regret_evidence_id": self.regret_evidence_id,
            "information_gain_evidence_id": self.information_gain_evidence_id,
            "provider_competence_passed": self.provider_competence_passed,
            "query_calibration_passed": self.query_calibration_passed,
            "identifiable_fraction": self.identifiable_fraction,
            "regret_upper_bound": self.regret_upper_bound,
            "expected_information_gain": self.expected_information_gain,
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}


def _decision_reasons(
    policy: QueryUpdateAdmissionPolicyV1,
    evidence: QueryUpdateEvidenceV1,
) -> tuple[str, ...]:
    reasons: list[str] = []
    tolerance = policy.numerical_tolerance
    if policy.require_provider_competence and not evidence.provider_competence_passed:
        reasons.append("provider-competence-not-passed")
    if policy.require_query_calibration and not evidence.query_calibration_passed:
        reasons.append("query-calibration-not-passed")
    if (
        evidence.identifiable_fraction + tolerance
        < policy.minimum_identifiable_fraction
    ):
        reasons.append("identifiable-query-fraction-below-threshold")
    if evidence.regret_upper_bound > policy.maximum_regret_upper_bound + tolerance:
        reasons.append("query-regret-upper-bound-exceeds-threshold")
    if (
        evidence.expected_information_gain + tolerance
        < policy.minimum_expected_information_gain
    ):
        reasons.append("query-information-gain-below-threshold")
    return tuple(reasons or [_AUTHORIZED_REASON])


@dataclass(frozen=True, slots=True)
class QueryUpdateAdmissionCertificateV1:
    """Content-addressed accept-or-exact-fallback query decision."""

    policy: QueryUpdateAdmissionPolicyV1
    evidence: QueryUpdateEvidenceV1
    authorized: bool
    selected_belief_id: str
    exact_fallback: bool
    reasons: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.policy, QueryUpdateAdmissionPolicyV1):
            raise TypeError("policy must be QueryUpdateAdmissionPolicyV1")
        if not isinstance(self.evidence, QueryUpdateEvidenceV1):
            raise TypeError("evidence must be QueryUpdateEvidenceV1")
        authorized = genuine_boolean(self.authorized, name="authorized")
        fallback = genuine_boolean(self.exact_fallback, name="exact_fallback")
        reasons = _canonical_reasons(self.reasons)
        expected_reasons = _decision_reasons(self.policy, self.evidence)
        expected_authorized = expected_reasons == (_AUTHORIZED_REASON,)
        if reasons != expected_reasons:
            raise ValueError("reasons do not match the frozen query admission policy")
        if authorized != expected_authorized:
            raise ValueError("authorized does not match the frozen query evidence")
        if fallback == authorized:
            raise ValueError(
                "exact_fallback must be the logical opposite of authorized"
            )
        selected = sha256_digest(self.selected_belief_id, name="selected_belief_id")
        expected_selected = (
            self.evidence.candidate_belief_id
            if authorized
            else self.evidence.baseline_belief_id
        )
        if selected != expected_selected:
            raise ValueError("selected_belief_id contradicts the query decision")
        object.__setattr__(self, "authorized", authorized)
        object.__setattr__(self, "selected_belief_id", selected)
        object.__setattr__(self, "exact_fallback", fallback)
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="query update admission metadata",
            ),
        )
        expected = content_id(self.descriptor())
        if self.artifact_id is not None:
            supplied = sha256_digest(self.artifact_id, name="artifact_id")
            if supplied != expected:
                raise ValueError("artifact_id does not match query admission")
        object.__setattr__(self, "artifact_id", expected)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": QUERY_UPDATE_ADMISSION_SCHEMA,
            "schema_version": QUERY_UPDATE_ADMISSION_VERSION,
            "policy_id": self.policy.policy_id,
            "evidence_id": self.evidence.artifact_id,
            "physical_query_id": self.evidence.physical_query_id,
            "baseline_belief_id": self.evidence.baseline_belief_id,
            "candidate_belief_id": self.evidence.candidate_belief_id,
            "fallback_belief_id": self.evidence.fallback_belief_id,
            "authorized": self.authorized,
            "selected_belief_id": self.selected_belief_id,
            "exact_fallback": self.exact_fallback,
            "reasons": list(self.reasons),
            "claim_boundary": QUERY_UPDATE_ADMISSION_CLAIM_BOUNDARY,
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}


def evaluate_query_update_admission(
    evidence: QueryUpdateEvidenceV1,
    *,
    policy: QueryUpdateAdmissionPolicyV1 | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> QueryUpdateAdmissionCertificateV1:
    """Evaluate all gates and return the candidate or exact baseline identity."""

    selected_policy = policy or QueryUpdateAdmissionPolicyV1()
    if not isinstance(evidence, QueryUpdateEvidenceV1):
        raise TypeError("evidence must be QueryUpdateEvidenceV1")
    if not isinstance(selected_policy, QueryUpdateAdmissionPolicyV1):
        raise TypeError("policy must be QueryUpdateAdmissionPolicyV1")
    reasons = _decision_reasons(selected_policy, evidence)
    authorized = reasons == (_AUTHORIZED_REASON,)
    selected = (
        evidence.candidate_belief_id if authorized else evidence.baseline_belief_id
    )
    return QueryUpdateAdmissionCertificateV1(
        policy=selected_policy,
        evidence=evidence,
        authorized=authorized,
        selected_belief_id=selected,
        exact_fallback=not authorized,
        reasons=reasons,
        metadata={} if metadata is None else metadata,
    )


__all__ = [
    "QUERY_UPDATE_ADMISSION_CLAIM_BOUNDARY",
    "QUERY_UPDATE_ADMISSION_SCHEMA",
    "QUERY_UPDATE_ADMISSION_VERSION",
    "QueryUpdateAdmissionCertificateV1",
    "QueryUpdateAdmissionPolicyV1",
    "QueryUpdateEvidenceV1",
    "evaluate_query_update_admission",
]
