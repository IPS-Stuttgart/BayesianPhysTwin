"""Query-specific admission after provider and covariance competence gates.

The certificate composes existing frozen artifacts. It never turns software
compatibility into scientific evidence and fails closed to the physical belief.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any, Final, cast

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    genuine_integer,
    plain_json,
)
from ._portable_contracts import (
    content_id,
    load_strict_json_object,
    require_exact_fields,
    sha256_digest,
    write_atomic_json,
)
from .evidence_decision_v1 import EvidenceDecisionV1
from .physical_query_v1 import PhysicalQueryDecisionMarginsV1, PhysicalQueryV1
from .query_covariance_decision_v1 import QueryCovarianceTreatmentDecisionV1

QUERY_ADMISSION_SCHEMA: Final = "bayesian_phystwin.query_admission_certificate"
QUERY_ADMISSION_VERSION: Final = 1
QUERY_ADMISSION_CLAIM_BOUNDARY: Final = (
    "Software admission evidence only. A passing certificate records that one "
    "already-frozen provider decision, covariance decision, and query-specific "
    "source evaluation satisfy the declared policy. It does not establish "
    "fresh-object benefit, deployment calibration, Causal4D intervention "
    "benefit, deployment safety, or state of the art."
)
_PROVIDER_STATUSES: Final = frozenset({"pass", "fail", "degraded", "inconclusive"})
_POLICY_FIELDS: Final = frozenset(
    {
        "provider_decision_key",
        "require_provider_claim_authorized",
        "minimum_provider_evidence_level",
        "minimum_group_count",
        "minimum_identifiable_subspace_overlap",
        "minimum_expected_information_gain",
        "maximum_harmful_group_fraction",
        "numerical_tolerance",
    }
)
_EVIDENCE_FIELDS: Final = frozenset(
    {
        "candidate_belief_id",
        "candidate_query_mean_id",
        "candidate_query_covariance_id",
        "baseline_query_mean_id",
        "baseline_query_covariance_id",
        "evaluation_artifact_id",
        "score_metric",
        "width_unit",
        "statistical_unit",
        "independent_group_count",
        "mean_score_regret",
        "score_regret_upper_bound",
        "maximum_score_increase",
        "worst_group_score_regret",
        "harmful_group_fraction",
        "accepted_coverage",
        "mean_full_width",
        "identifiable_subspace_overlap",
        "shared_covariance_relevance",
        "expected_information_gain",
        "policy_frozen_before_evaluation_outcomes",
        "evaluation_outcomes_used_for_candidate_selection",
        "evaluation_groups_independent",
        "metadata",
    }
)
_CERTIFICATE_FIELDS: Final = frozenset(
    {
        "physical_query_id",
        "provider_decision_id",
        "covariance_decision_id",
        "decision_margins",
        "primary_proper_score",
        "physical_unit",
        "statistical_unit",
        "policy",
        "evidence",
        "baseline_belief_id",
        "selected_belief_id",
        "exact_fallback_id",
        "provider_status",
        "provider_claim_authorized",
        "provider_evidence_level",
        "covariance_treatment_authorized",
        "provider_competence_passed",
        "query_nonharm_passed",
        "query_information_passed",
        "admitted",
        "exact_fallback",
        "reasons",
        "metadata",
    }
)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{name} must use literal string keys")
    return cast(Mapping[str, Any], value)


def _text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be canonical nonempty text")
    return value


def _finite(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_exclusive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None:
        invalid = result <= minimum if minimum_exclusive else result < minimum
        if invalid:
            relation = ">" if minimum_exclusive else ">="
            raise ValueError(f"{name} must be {relation} {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


def _fraction(value: object, *, name: str) -> float:
    return _finite(value, name=name, minimum=0.0, maximum=1.0)


@dataclass(frozen=True, slots=True)
class QueryAdmissionPolicyV1:
    """Thresholds additional to those already frozen by ``PhysicalQueryV1``."""

    provider_decision_key: str = "source-provider-gate"
    require_provider_claim_authorized: bool = True
    minimum_provider_evidence_level: int = 3
    minimum_group_count: int = 6
    minimum_identifiable_subspace_overlap: float = 0.5
    minimum_expected_information_gain: float = 0.0
    maximum_harmful_group_fraction: float = 0.0
    numerical_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "provider_decision_key",
            _text(self.provider_decision_key, name="provider_decision_key"),
        )
        object.__setattr__(
            self,
            "require_provider_claim_authorized",
            genuine_boolean(
                self.require_provider_claim_authorized,
                name="require_provider_claim_authorized",
            ),
        )
        level = genuine_integer(
            self.minimum_provider_evidence_level,
            name="minimum_provider_evidence_level",
            minimum=1,
        )
        if level > 3:
            raise ValueError("minimum_provider_evidence_level must be at most 3")
        object.__setattr__(self, "minimum_provider_evidence_level", level)
        object.__setattr__(
            self,
            "minimum_group_count",
            genuine_integer(
                self.minimum_group_count,
                name="minimum_group_count",
                minimum=2,
            ),
        )
        object.__setattr__(
            self,
            "minimum_identifiable_subspace_overlap",
            _fraction(
                self.minimum_identifiable_subspace_overlap,
                name="minimum_identifiable_subspace_overlap",
            ),
        )
        object.__setattr__(
            self,
            "minimum_expected_information_gain",
            _finite(
                self.minimum_expected_information_gain,
                name="minimum_expected_information_gain",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "maximum_harmful_group_fraction",
            _fraction(
                self.maximum_harmful_group_fraction,
                name="maximum_harmful_group_fraction",
            ),
        )
        object.__setattr__(
            self,
            "numerical_tolerance",
            _finite(
                self.numerical_tolerance,
                name="numerical_tolerance",
                minimum=0.0,
                minimum_exclusive=True,
            ),
        )

    @property
    def policy_id(self) -> str:
        return content_id(self.descriptor())

    def descriptor(self) -> dict[str, object]:
        return {
            "provider_decision_key": self.provider_decision_key,
            "require_provider_claim_authorized": (
                self.require_provider_claim_authorized
            ),
            "minimum_provider_evidence_level": (
                self.minimum_provider_evidence_level
            ),
            "minimum_group_count": self.minimum_group_count,
            "minimum_identifiable_subspace_overlap": (
                self.minimum_identifiable_subspace_overlap
            ),
            "minimum_expected_information_gain": (
                self.minimum_expected_information_gain
            ),
            "maximum_harmful_group_fraction": (
                self.maximum_harmful_group_fraction
            ),
            "numerical_tolerance": self.numerical_tolerance,
        }

    @classmethod
    def from_mapping(cls, value: object) -> QueryAdmissionPolicyV1:
        source = dict(_mapping(value, name="query admission policy"))
        require_exact_fields(
            source,
            expected=_POLICY_FIELDS,
            name="query admission policy",
        )
        return cls(
            provider_decision_key=source["provider_decision_key"],
            require_provider_claim_authorized=source[
                "require_provider_claim_authorized"
            ],
            minimum_provider_evidence_level=source[
                "minimum_provider_evidence_level"
            ],
            minimum_group_count=source["minimum_group_count"],
            minimum_identifiable_subspace_overlap=source[
                "minimum_identifiable_subspace_overlap"
            ],
            minimum_expected_information_gain=source[
                "minimum_expected_information_gain"
            ],
            maximum_harmful_group_fraction=source[
                "maximum_harmful_group_fraction"
            ],
            numerical_tolerance=source["numerical_tolerance"],
        )


@dataclass(frozen=True, slots=True)
class QueryAdmissionEvidenceV1:
    """Source-evaluated query metrics with immutable mean/covariance identities."""

    candidate_belief_id: str
    candidate_query_mean_id: str
    candidate_query_covariance_id: str
    baseline_query_mean_id: str
    baseline_query_covariance_id: str
    evaluation_artifact_id: str
    score_metric: str
    width_unit: str
    statistical_unit: str
    independent_group_count: int
    mean_score_regret: float
    score_regret_upper_bound: float
    maximum_score_increase: float
    worst_group_score_regret: float
    harmful_group_fraction: float
    accepted_coverage: float
    mean_full_width: float
    identifiable_subspace_overlap: float
    shared_covariance_relevance: float
    expected_information_gain: float
    policy_frozen_before_evaluation_outcomes: bool
    evaluation_outcomes_used_for_candidate_selection: bool
    evaluation_groups_independent: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "candidate_belief_id",
            "candidate_query_mean_id",
            "candidate_query_covariance_id",
            "baseline_query_mean_id",
            "baseline_query_covariance_id",
            "evaluation_artifact_id",
        ):
            object.__setattr__(
                self,
                name,
                sha256_digest(getattr(self, name), name=name),
            )
        for name in ("score_metric", "width_unit", "statistical_unit"):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "independent_group_count",
            genuine_integer(
                self.independent_group_count,
                name="independent_group_count",
                minimum=1,
            ),
        )
        mean_regret = _finite(self.mean_score_regret, name="mean_score_regret")
        upper_regret = _finite(
            self.score_regret_upper_bound,
            name="score_regret_upper_bound",
        )
        maximum_increase = _finite(
            self.maximum_score_increase,
            name="maximum_score_increase",
        )
        worst_regret = _finite(
            self.worst_group_score_regret,
            name="worst_group_score_regret",
        )
        for name, value in (
            ("score_regret_upper_bound", upper_regret),
            ("maximum_score_increase", maximum_increase),
            ("worst_group_score_regret", worst_regret),
        ):
            if value < mean_regret:
                raise ValueError(f"{name} must cover mean_score_regret")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "mean_score_regret", mean_regret)
        object.__setattr__(
            self,
            "harmful_group_fraction",
            _fraction(self.harmful_group_fraction, name="harmful_group_fraction"),
        )
        object.__setattr__(
            self,
            "accepted_coverage",
            _fraction(self.accepted_coverage, name="accepted_coverage"),
        )
        object.__setattr__(
            self,
            "mean_full_width",
            _finite(self.mean_full_width, name="mean_full_width", minimum=0.0),
        )
        object.__setattr__(
            self,
            "identifiable_subspace_overlap",
            _fraction(
                self.identifiable_subspace_overlap,
                name="identifiable_subspace_overlap",
            ),
        )
        object.__setattr__(
            self,
            "shared_covariance_relevance",
            _fraction(
                self.shared_covariance_relevance,
                name="shared_covariance_relevance",
            ),
        )
        object.__setattr__(
            self,
            "expected_information_gain",
            _finite(
                self.expected_information_gain,
                name="expected_information_gain",
                minimum=0.0,
            ),
        )
        for name in (
            "policy_frozen_before_evaluation_outcomes",
            "evaluation_outcomes_used_for_candidate_selection",
            "evaluation_groups_independent",
        ):
            object.__setattr__(
                self,
                name,
                genuine_boolean(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="query admission evidence metadata",
            ),
        )

    @property
    def evidence_id(self) -> str:
        return content_id(self.descriptor())

    def descriptor(self) -> dict[str, object]:
        return {
            "candidate_belief_id": self.candidate_belief_id,
            "candidate_query_mean_id": self.candidate_query_mean_id,
            "candidate_query_covariance_id": self.candidate_query_covariance_id,
            "baseline_query_mean_id": self.baseline_query_mean_id,
            "baseline_query_covariance_id": self.baseline_query_covariance_id,
            "evaluation_artifact_id": self.evaluation_artifact_id,
            "score_metric": self.score_metric,
            "width_unit": self.width_unit,
            "statistical_unit": self.statistical_unit,
            "independent_group_count": self.independent_group_count,
            "mean_score_regret": self.mean_score_regret,
            "score_regret_upper_bound": self.score_regret_upper_bound,
            "maximum_score_increase": self.maximum_score_increase,
            "worst_group_score_regret": self.worst_group_score_regret,
            "harmful_group_fraction": self.harmful_group_fraction,
            "accepted_coverage": self.accepted_coverage,
            "mean_full_width": self.mean_full_width,
            "identifiable_subspace_overlap": self.identifiable_subspace_overlap,
            "shared_covariance_relevance": self.shared_covariance_relevance,
            "expected_information_gain": self.expected_information_gain,
            "policy_frozen_before_evaluation_outcomes": (
                self.policy_frozen_before_evaluation_outcomes
            ),
            "evaluation_outcomes_used_for_candidate_selection": (
                self.evaluation_outcomes_used_for_candidate_selection
            ),
            "evaluation_groups_independent": self.evaluation_groups_independent,
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {"evidence_id": self.evidence_id, **self.descriptor()}

    @classmethod
    def from_mapping(cls, value: object) -> QueryAdmissionEvidenceV1:
        source = dict(_mapping(value, name="query admission evidence"))
        require_exact_fields(
            source,
            expected=_EVIDENCE_FIELDS | {"evidence_id"},
            name="query admission evidence",
        )
        expected_id = sha256_digest(source.pop("evidence_id"), name="evidence_id")
        result = cls(
            candidate_belief_id=source["candidate_belief_id"],
            candidate_query_mean_id=source["candidate_query_mean_id"],
            candidate_query_covariance_id=source[
                "candidate_query_covariance_id"
            ],
            baseline_query_mean_id=source["baseline_query_mean_id"],
            baseline_query_covariance_id=source[
                "baseline_query_covariance_id"
            ],
            evaluation_artifact_id=source["evaluation_artifact_id"],
            score_metric=source["score_metric"],
            width_unit=source["width_unit"],
            statistical_unit=source["statistical_unit"],
            independent_group_count=source["independent_group_count"],
            mean_score_regret=source["mean_score_regret"],
            score_regret_upper_bound=source["score_regret_upper_bound"],
            maximum_score_increase=source["maximum_score_increase"],
            worst_group_score_regret=source["worst_group_score_regret"],
            harmful_group_fraction=source["harmful_group_fraction"],
            accepted_coverage=source["accepted_coverage"],
            mean_full_width=source["mean_full_width"],
            identifiable_subspace_overlap=source[
                "identifiable_subspace_overlap"
            ],
            shared_covariance_relevance=source["shared_covariance_relevance"],
            expected_information_gain=source["expected_information_gain"],
            policy_frozen_before_evaluation_outcomes=source[
                "policy_frozen_before_evaluation_outcomes"
            ],
            evaluation_outcomes_used_for_candidate_selection=source[
                "evaluation_outcomes_used_for_candidate_selection"
            ],
            evaluation_groups_independent=source[
                "evaluation_groups_independent"
            ],
            metadata=_mapping(
                source["metadata"],
                name="query admission evidence metadata",
            ),
        )
        if result.evidence_id != expected_id:
            raise ValueError("query admission evidence identity changed")
        return result


def _decision_reasons(
    *,
    policy: QueryAdmissionPolicyV1,
    evidence: QueryAdmissionEvidenceV1,
    provider_status: str,
    provider_claim_authorized: bool,
    provider_evidence_level: int,
    covariance_treatment_authorized: bool,
    decision_margins: PhysicalQueryDecisionMarginsV1,
) -> tuple[str, ...]:
    tolerance = policy.numerical_tolerance
    reasons: list[str] = []
    if provider_status != "pass":
        reasons.append("provider-decision-not-pass")
    if policy.require_provider_claim_authorized and not provider_claim_authorized:
        reasons.append("provider-claim-not-authorized")
    if provider_evidence_level < policy.minimum_provider_evidence_level:
        reasons.append("provider-evidence-level-below-threshold")
    if not covariance_treatment_authorized:
        reasons.append("query-covariance-treatment-not-authorized")
    if not evidence.policy_frozen_before_evaluation_outcomes:
        reasons.append("query-policy-not-frozen-before-outcomes")
    if evidence.evaluation_outcomes_used_for_candidate_selection:
        reasons.append("evaluation-outcomes-used-for-candidate-selection")
    if not evidence.evaluation_groups_independent:
        reasons.append("evaluation-groups-not-independent")
    if evidence.independent_group_count < policy.minimum_group_count:
        reasons.append("insufficient-independent-groups")
    if (
        evidence.score_regret_upper_bound
        > decision_margins.practical_equivalence_score + tolerance
    ):
        reasons.append("query-score-regret-upper-bound-exceeds-margin")
    if (
        evidence.maximum_score_increase
        > decision_margins.maximum_harmful_score_increase + tolerance
    ):
        reasons.append("maximum-score-increase-exceeds-margin")
    if (
        evidence.worst_group_score_regret
        > decision_margins.maximum_worst_group_score_regret + tolerance
    ):
        reasons.append("worst-group-score-regret-exceeds-margin")
    if (
        evidence.accepted_coverage + tolerance
        < decision_margins.minimum_accepted_coverage
    ):
        reasons.append("accepted-coverage-below-margin")
    if evidence.mean_full_width > decision_margins.maximum_mean_width + tolerance:
        reasons.append("mean-width-exceeds-margin")
    if (
        evidence.shared_covariance_relevance + tolerance
        < decision_margins.minimum_shared_covariance_relevance
    ):
        reasons.append("shared-covariance-relevance-below-margin")
    if (
        evidence.harmful_group_fraction
        > policy.maximum_harmful_group_fraction + tolerance
    ):
        reasons.append("harmful-group-fraction-exceeds-policy")
    if (
        evidence.identifiable_subspace_overlap + tolerance
        < policy.minimum_identifiable_subspace_overlap
    ):
        reasons.append("identifiable-subspace-overlap-below-threshold")
    if (
        evidence.expected_information_gain + tolerance
        < policy.minimum_expected_information_gain
    ):
        reasons.append("expected-information-gain-below-threshold")
    return tuple(sorted(reasons))


_PROVIDER_REASONS: Final = frozenset(
    {
        "provider-decision-not-pass",
        "provider-claim-not-authorized",
        "provider-evidence-level-below-threshold",
    }
)
_NONHARM_REASONS: Final = frozenset(
    {
        "query-policy-not-frozen-before-outcomes",
        "evaluation-outcomes-used-for-candidate-selection",
        "evaluation-groups-not-independent",
        "insufficient-independent-groups",
        "query-score-regret-upper-bound-exceeds-margin",
        "maximum-score-increase-exceeds-margin",
        "worst-group-score-regret-exceeds-margin",
        "harmful-group-fraction-exceeds-policy",
        "accepted-coverage-below-margin",
        "mean-width-exceeds-margin",
    }
)
_INFORMATION_REASONS: Final = frozenset(
    {
        "identifiable-subspace-overlap-below-threshold",
        "shared-covariance-relevance-below-margin",
        "expected-information-gain-below-threshold",
    }
)


@dataclass(frozen=True, slots=True)
class QueryAdmissionCertificateV1:
    """Content-addressed accept/fallback decision for one physical query."""

    physical_query_id: str
    provider_decision_id: str
    covariance_decision_id: str
    decision_margins: PhysicalQueryDecisionMarginsV1
    primary_proper_score: str
    physical_unit: str
    statistical_unit: str
    policy: QueryAdmissionPolicyV1
    evidence: QueryAdmissionEvidenceV1
    baseline_belief_id: str
    selected_belief_id: str
    exact_fallback_id: str
    provider_status: str
    provider_claim_authorized: bool
    provider_evidence_level: int
    covariance_treatment_authorized: bool
    provider_competence_passed: bool
    query_nonharm_passed: bool
    query_information_passed: bool
    admitted: bool
    exact_fallback: bool
    reasons: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "physical_query_id",
            "provider_decision_id",
            "covariance_decision_id",
            "baseline_belief_id",
            "selected_belief_id",
            "exact_fallback_id",
        ):
            object.__setattr__(
                self,
                name,
                sha256_digest(getattr(self, name), name=name),
            )
        if not isinstance(self.decision_margins, PhysicalQueryDecisionMarginsV1):
            raise TypeError(
                "decision_margins must be PhysicalQueryDecisionMarginsV1"
            )
        for name in ("primary_proper_score", "physical_unit", "statistical_unit"):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        if self.decision_margins.width_unit != self.physical_unit:
            raise ValueError("decision margin width unit differs from physical_unit")
        if not isinstance(self.policy, QueryAdmissionPolicyV1):
            raise TypeError("policy must be QueryAdmissionPolicyV1")
        if not isinstance(self.evidence, QueryAdmissionEvidenceV1):
            raise TypeError("evidence must be QueryAdmissionEvidenceV1")
        if self.evidence.score_metric != self.primary_proper_score:
            raise ValueError("query evidence proper score differs from certificate")
        if self.evidence.width_unit != self.physical_unit:
            raise ValueError("query evidence width unit differs from certificate")
        if self.evidence.statistical_unit != self.statistical_unit:
            raise ValueError("query evidence statistical unit differs from certificate")

        provider_status = _text(self.provider_status, name="provider_status")
        if provider_status not in _PROVIDER_STATUSES:
            raise ValueError("provider_status is not registered")
        provider_authorized = genuine_boolean(
            self.provider_claim_authorized,
            name="provider_claim_authorized",
        )
        provider_level = genuine_integer(
            self.provider_evidence_level,
            name="provider_evidence_level",
            minimum=1,
        )
        if provider_level > 3:
            raise ValueError("provider_evidence_level must be at most 3")
        covariance_authorized = genuine_boolean(
            self.covariance_treatment_authorized,
            name="covariance_treatment_authorized",
        )
        expected_reasons = _decision_reasons(
            policy=self.policy,
            evidence=self.evidence,
            provider_status=provider_status,
            provider_claim_authorized=provider_authorized,
            provider_evidence_level=provider_level,
            covariance_treatment_authorized=covariance_authorized,
            decision_margins=self.decision_margins,
        )
        supplied_reasons = tuple(sorted(self.reasons))
        if supplied_reasons == ("query-admission-authorized",):
            supplied_reasons = ()
        if len(supplied_reasons) != len(set(supplied_reasons)):
            raise ValueError("query admission reasons must not contain duplicates")
        if any(type(item) is not str or not item for item in supplied_reasons):
            raise ValueError("query admission reasons must contain text")
        if supplied_reasons != expected_reasons:
            raise ValueError("query admission reasons contradict certificate inputs")

        provider_passed = not bool(_PROVIDER_REASONS & set(supplied_reasons))
        nonharm_passed = not bool(_NONHARM_REASONS & set(supplied_reasons))
        information_passed = not bool(_INFORMATION_REASONS & set(supplied_reasons))
        admitted = not supplied_reasons
        exact_fallback = not admitted
        selected = (
            self.evidence.candidate_belief_id
            if admitted
            else self.baseline_belief_id
        )
        expected_values = {
            "provider_competence_passed": provider_passed,
            "query_nonharm_passed": nonharm_passed,
            "query_information_passed": information_passed,
            "admitted": admitted,
            "exact_fallback": exact_fallback,
        }
        for name, expected in expected_values.items():
            supplied = genuine_boolean(getattr(self, name), name=name)
            if supplied != expected:
                raise ValueError(f"{name} contradicts query admission reasons")
            object.__setattr__(self, name, supplied)
        if self.selected_belief_id != selected:
            raise ValueError("selected_belief_id contradicts admission decision")

        object.__setattr__(self, "provider_status", provider_status)
        object.__setattr__(
            self,
            "provider_claim_authorized",
            provider_authorized,
        )
        object.__setattr__(self, "provider_evidence_level", provider_level)
        object.__setattr__(
            self,
            "covariance_treatment_authorized",
            covariance_authorized,
        )
        object.__setattr__(
            self,
            "reasons",
            ("query-admission-authorized",) if admitted else supplied_reasons,
        )
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="query admission certificate metadata",
            ),
        )

    @property
    def artifact_id(self) -> str:
        return content_id(self.descriptor())

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": QUERY_ADMISSION_SCHEMA,
            "schema_version": QUERY_ADMISSION_VERSION,
            "physical_query_id": self.physical_query_id,
            "provider_decision_id": self.provider_decision_id,
            "covariance_decision_id": self.covariance_decision_id,
            "decision_margins": self.decision_margins.descriptor(),
            "primary_proper_score": self.primary_proper_score,
            "physical_unit": self.physical_unit,
            "statistical_unit": self.statistical_unit,
            "policy": {
                **self.policy.descriptor(),
                "policy_id": self.policy.policy_id,
            },
            "evidence": self.evidence.to_record(),
            "baseline_belief_id": self.baseline_belief_id,
            "selected_belief_id": self.selected_belief_id,
            "exact_fallback_id": self.exact_fallback_id,
            "provider_status": self.provider_status,
            "provider_claim_authorized": self.provider_claim_authorized,
            "provider_evidence_level": self.provider_evidence_level,
            "covariance_treatment_authorized": (
                self.covariance_treatment_authorized
            ),
            "provider_competence_passed": self.provider_competence_passed,
            "query_nonharm_passed": self.query_nonharm_passed,
            "query_information_passed": self.query_information_passed,
            "admitted": self.admitted,
            "exact_fallback": self.exact_fallback,
            "reasons": list(self.reasons),
            "claim_boundary": QUERY_ADMISSION_CLAIM_BOUNDARY,
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {"artifact_id": self.artifact_id, **self.descriptor()}

    @classmethod
    def from_mapping(cls, value: object) -> QueryAdmissionCertificateV1:
        source = dict(_mapping(value, name="query admission certificate"))
        extras = {"artifact_id", "schema", "schema_version", "claim_boundary"}
        require_exact_fields(
            source,
            expected=_CERTIFICATE_FIELDS | extras,
            name="query admission certificate",
        )
        expected_id = sha256_digest(source.pop("artifact_id"), name="artifact_id")
        if source.pop("schema") != QUERY_ADMISSION_SCHEMA:
            raise ValueError("query admission schema changed")
        version = genuine_integer(
            source.pop("schema_version"),
            name="schema_version",
            minimum=1,
        )
        if version != QUERY_ADMISSION_VERSION:
            raise ValueError("query admission schema version changed")
        if source.pop("claim_boundary") != QUERY_ADMISSION_CLAIM_BOUNDARY:
            raise ValueError("query admission claim boundary changed")
        decision_margins = PhysicalQueryDecisionMarginsV1.from_mapping(
            source["decision_margins"]
        )
        policy_record = dict(_mapping(source["policy"], name="policy"))
        policy_id = sha256_digest(policy_record.pop("policy_id"), name="policy_id")
        policy = QueryAdmissionPolicyV1.from_mapping(policy_record)
        if policy.policy_id != policy_id:
            raise ValueError("query admission policy identity changed")
        evidence = QueryAdmissionEvidenceV1.from_mapping(source["evidence"])
        reasons = source["reasons"]
        if isinstance(reasons, (str, bytes)) or not isinstance(reasons, list):
            raise ValueError("query admission reasons must be a JSON array")
        result = cls(
            physical_query_id=source["physical_query_id"],
            provider_decision_id=source["provider_decision_id"],
            covariance_decision_id=source["covariance_decision_id"],
            decision_margins=decision_margins,
            primary_proper_score=source["primary_proper_score"],
            physical_unit=source["physical_unit"],
            statistical_unit=source["statistical_unit"],
            policy=policy,
            evidence=evidence,
            baseline_belief_id=source["baseline_belief_id"],
            selected_belief_id=source["selected_belief_id"],
            exact_fallback_id=source["exact_fallback_id"],
            provider_status=source["provider_status"],
            provider_claim_authorized=source["provider_claim_authorized"],
            provider_evidence_level=source["provider_evidence_level"],
            covariance_treatment_authorized=source[
                "covariance_treatment_authorized"
            ],
            provider_competence_passed=source[
                "provider_competence_passed"
            ],
            query_nonharm_passed=source["query_nonharm_passed"],
            query_information_passed=source["query_information_passed"],
            admitted=source["admitted"],
            exact_fallback=source["exact_fallback"],
            reasons=tuple(reasons),
            metadata=_mapping(
                source["metadata"],
                name="query admission certificate metadata",
            ),
        )
        if result.artifact_id != expected_id:
            raise ValueError("query admission certificate identity changed")
        return result


def compose_query_admission(
    query: PhysicalQueryV1,
    provider_decision: EvidenceDecisionV1,
    covariance_decision: QueryCovarianceTreatmentDecisionV1,
    evidence: QueryAdmissionEvidenceV1,
    *,
    policy: QueryAdmissionPolicyV1 | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> QueryAdmissionCertificateV1:
    """Compose one fail-closed query-specific admission certificate."""

    if not isinstance(query, PhysicalQueryV1):
        raise TypeError("query must be PhysicalQueryV1")
    if not isinstance(provider_decision, EvidenceDecisionV1):
        raise TypeError("provider_decision must be EvidenceDecisionV1")
    if not isinstance(covariance_decision, QueryCovarianceTreatmentDecisionV1):
        raise TypeError(
            "covariance_decision must be QueryCovarianceTreatmentDecisionV1"
        )
    if not isinstance(evidence, QueryAdmissionEvidenceV1):
        raise TypeError("evidence must be QueryAdmissionEvidenceV1")
    selected_policy = QueryAdmissionPolicyV1() if policy is None else policy
    if not isinstance(selected_policy, QueryAdmissionPolicyV1):
        raise TypeError("policy must be QueryAdmissionPolicyV1")

    frozen_provider_id = query.evidence_decision_ids.get(
        selected_policy.provider_decision_key
    )
    if frozen_provider_id is None:
        raise ValueError("physical query does not bind the provider decision key")
    if frozen_provider_id != provider_decision.decision_id:
        raise ValueError("provider decision differs from the physical query freeze")
    if covariance_decision.physical_query_id != query.query_id:
        raise ValueError("covariance decision belongs to another physical query")
    if covariance_decision.exact_fallback_id != query.exact_fallback_id:
        raise ValueError("covariance decision exact fallback differs from the query")
    if evidence.score_metric != query.primary_proper_score:
        raise ValueError("query evidence uses a different proper score")
    if evidence.width_unit != query.physical_unit:
        raise ValueError("query evidence width unit differs from the physical query")
    if evidence.statistical_unit != query.bootstrap.independent_group_definition:
        raise ValueError("query evidence uses a different statistical unit")

    reasons = _decision_reasons(
        policy=selected_policy,
        evidence=evidence,
        provider_status=provider_decision.status,
        provider_claim_authorized=provider_decision.claim_authorized,
        provider_evidence_level=provider_decision.evidence_level,
        covariance_treatment_authorized=covariance_decision.authorized,
        decision_margins=query.decision_margins,
    )
    admitted = not reasons
    provider_passed = not bool(_PROVIDER_REASONS & set(reasons))
    nonharm_passed = not bool(_NONHARM_REASONS & set(reasons))
    information_passed = not bool(_INFORMATION_REASONS & set(reasons))
    return QueryAdmissionCertificateV1(
        physical_query_id=cast(str, query.query_id),
        provider_decision_id=provider_decision.decision_id,
        covariance_decision_id=cast(str, covariance_decision.artifact_id),
        decision_margins=query.decision_margins,
        primary_proper_score=query.primary_proper_score,
        physical_unit=query.physical_unit,
        statistical_unit=query.bootstrap.independent_group_definition,
        policy=selected_policy,
        evidence=evidence,
        baseline_belief_id=query.baseline_physical_belief_id,
        selected_belief_id=(
            evidence.candidate_belief_id
            if admitted
            else query.baseline_physical_belief_id
        ),
        exact_fallback_id=query.exact_fallback_id,
        provider_status=provider_decision.status,
        provider_claim_authorized=provider_decision.claim_authorized,
        provider_evidence_level=provider_decision.evidence_level,
        covariance_treatment_authorized=covariance_decision.authorized,
        provider_competence_passed=provider_passed,
        query_nonharm_passed=nonharm_passed,
        query_information_passed=information_passed,
        admitted=admitted,
        exact_fallback=not admitted,
        reasons=("query-admission-authorized",) if admitted else reasons,
        metadata={} if metadata is None else metadata,
    )


def write_query_admission_certificate(
    certificate: QueryAdmissionCertificateV1,
    path: str | Path,
) -> None:
    if not isinstance(certificate, QueryAdmissionCertificateV1):
        raise TypeError("certificate must be QueryAdmissionCertificateV1")
    write_atomic_json(certificate.to_record(), path, overwrite=False)


def load_query_admission_certificate(
    path: str | Path,
) -> QueryAdmissionCertificateV1:
    return QueryAdmissionCertificateV1.from_mapping(
        load_strict_json_object(path, label="query admission certificate")
    )


__all__ = [
    "QUERY_ADMISSION_CLAIM_BOUNDARY",
    "QUERY_ADMISSION_SCHEMA",
    "QUERY_ADMISSION_VERSION",
    "QueryAdmissionCertificateV1",
    "QueryAdmissionEvidenceV1",
    "QueryAdmissionPolicyV1",
    "compose_query_admission",
    "load_query_admission_certificate",
    "write_query_admission_certificate",
]
