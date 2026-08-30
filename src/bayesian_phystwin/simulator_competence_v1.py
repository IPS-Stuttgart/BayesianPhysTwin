"""Query-conditional competence certificates for deformable simulators.

Numerical validity, source competence, and selective deployment risk are
different claims.  This contract composes them without permitting a backend,
runtime, query, horizon, score model, threshold, or fallback substitution.
Rejected queries return the caller's exact fallback object.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any, Final, cast

from ._canonical_contracts import (
    canonical_string_tuple,
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
from .guard_harm_risk import (
    RISK_SCORE_SEMANTICS,
    GuardHarmRiskCertificateV1,
)
from .material_backend_evidence_v1 import (
    MaterialBackendEvidenceStatusV1,
    require_material_backend_evidence_stage,
)
from .material_backend_qualification_v1 import MaterialBackendQualificationV1
from .material_backend_v1 import resolve_material_backend_profile

QUERY_CONTEXT_SCHEMA: Final = "bayesian-phystwin.simulator-query-context"
SIMULATOR_COMPETENCE_POLICY_SCHEMA: Final = (
    "bayesian-phystwin.simulator-competence-policy"
)
QUERY_CONDITIONAL_CERTIFICATE_SCHEMA: Final = (
    "bayesian-phystwin.query-conditional-competence-certificate"
)
QUERY_CONDITIONAL_DECISION_SCHEMA: Final = (
    "bayesian-phystwin.query-conditional-competence-decision"
)
SCHEMA_VERSION: Final = 1
QUERY_CONDITIONAL_COMPETENCE_CLAIM_BOUNDARY: Final = (
    "A passing certificate bounds baseline-relative harm among accepted queries "
    "for one exact simulator runtime, pre-outcome score policy, query domain, "
    "loss, and finite-group certification population. It does not establish "
    "universal simulator validity, zero risk, safety outside the registered "
    "domain, official benchmark superiority, or state of the art."
)
_AUTHORIZED_REASON: Final = "simulator-query-authorized"


def _canonical_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    return value


def _finite_real(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if strictly_positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _sorted_unique_strings(
    values: Sequence[str],
    *,
    name: str,
) -> tuple[str, ...]:
    result = canonical_string_tuple(values, name=name, allow_empty=False)
    if any(value.strip() != value for value in result):
        raise ValueError(f"{name} must contain canonical strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique strings")
    return tuple(sorted(result))


def _sorted_unique_digests(
    values: Sequence[str],
    *,
    name: str,
) -> tuple[str, ...]:
    result = canonical_string_tuple(values, name=name, allow_empty=False)
    digests = tuple(
        sha256_digest(value, name=f"{name}[{index}]")
        for index, value in enumerate(result)
    )
    if len(set(digests)) != len(digests):
        raise ValueError(f"{name} must contain unique digests")
    return tuple(sorted(digests))


@dataclass(frozen=True, slots=True)
class SimulatorQueryContextV1:
    """One outcome-unopened object/action/horizon/query context."""

    object_context_id: str
    object_domain_id: str
    action_context_id: str
    action_domain_id: str
    horizon_seconds: float
    horizon_step_count: int
    query_functional_id: str
    loss_metric: str
    preoutcome_features_id: str
    outcome_observed: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "object_context_id",
            "object_domain_id",
            "action_context_id",
            "action_domain_id",
            "query_functional_id",
            "preoutcome_features_id",
        ):
            object.__setattr__(
                self,
                name,
                sha256_digest(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "horizon_seconds",
            _finite_real(
                self.horizon_seconds,
                name="horizon_seconds",
                strictly_positive=True,
            ),
        )
        object.__setattr__(
            self,
            "horizon_step_count",
            genuine_integer(
                self.horizon_step_count,
                name="horizon_step_count",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "loss_metric",
            _canonical_text(self.loss_metric, name="loss_metric"),
        )
        observed = genuine_boolean(self.outcome_observed, name="outcome_observed")
        if observed:
            raise ValueError("a simulator query context must be outcome-unopened")
        object.__setattr__(self, "outcome_observed", False)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="simulator query metadata",
            ),
        )
        expected = content_id(self.descriptor())
        if self.artifact_id is not None:
            supplied = sha256_digest(self.artifact_id, name="artifact_id")
            if supplied != expected:
                raise ValueError("artifact_id does not match simulator query context")
        object.__setattr__(self, "artifact_id", expected)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": QUERY_CONTEXT_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "object_context_id": self.object_context_id,
            "object_domain_id": self.object_domain_id,
            "action_context_id": self.action_context_id,
            "action_domain_id": self.action_domain_id,
            "horizon_seconds": self.horizon_seconds,
            "horizon_step_count": self.horizon_step_count,
            "query_functional_id": self.query_functional_id,
            "loss_metric": self.loss_metric,
            "preoutcome_features_id": self.preoutcome_features_id,
            "outcome_observed": self.outcome_observed,
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}

    @classmethod
    def from_mapping(cls, value: object) -> SimulatorQueryContextV1:
        if not isinstance(value, Mapping):
            raise ValueError("simulator query context must be a mapping")
        expected = frozenset(
            {
                "schema",
                "schema_version",
                "object_context_id",
                "object_domain_id",
                "action_context_id",
                "action_domain_id",
                "horizon_seconds",
                "horizon_step_count",
                "query_functional_id",
                "loss_metric",
                "preoutcome_features_id",
                "outcome_observed",
                "metadata",
                "artifact_id",
            }
        )
        require_exact_fields(value, expected=expected, name="simulator query context")
        if value["schema"] != QUERY_CONTEXT_SCHEMA:
            raise ValueError("simulator query context schema changed")
        if value["schema_version"] != SCHEMA_VERSION:
            raise ValueError("simulator query context version changed")
        return cls(
            object_context_id=cast(str, value["object_context_id"]),
            object_domain_id=cast(str, value["object_domain_id"]),
            action_context_id=cast(str, value["action_context_id"]),
            action_domain_id=cast(str, value["action_domain_id"]),
            horizon_seconds=cast(float, value["horizon_seconds"]),
            horizon_step_count=cast(int, value["horizon_step_count"]),
            query_functional_id=cast(str, value["query_functional_id"]),
            loss_metric=cast(str, value["loss_metric"]),
            preoutcome_features_id=cast(str, value["preoutcome_features_id"]),
            outcome_observed=cast(bool, value["outcome_observed"]),
            metadata=cast(Mapping[str, Any], value["metadata"]),
            artifact_id=cast(str, value["artifact_id"]),
        )


@dataclass(frozen=True, slots=True)
class SimulatorCompetencePolicyV1:
    """Frozen scope and score policy for one simulator runtime."""

    canonical_profile_id: str
    producer_profile_id: str
    runtime_id: str
    backend_evidence_status_id: str
    method_artifact_id: str
    method_selection_partition_id: str
    method_selection_group_ids: Sequence[str]
    object_domain_id: str
    action_domain_id: str
    allowed_query_functional_ids: Sequence[str]
    minimum_horizon_seconds: float
    maximum_horizon_seconds: float
    maximum_horizon_step_count: int
    loss_metric: str
    statistical_unit: str
    risk_feature_schema_id: str
    risk_model_id: str
    threshold_source_artifact_id: str
    risk_threshold: float
    fallback_policy_id: str
    exact_fallback_required: bool = True
    source_protocol_frozen: bool = True
    risk_policy_frozen_before_certification_outcomes: bool = True
    certification_outcomes_used_for_method_selection: bool = False
    target_outcomes_used: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    policy_id: str | None = None

    def __post_init__(self) -> None:
        resolved = resolve_material_backend_profile(self.producer_profile_id)
        if resolved.profile_id != self.canonical_profile_id:
            raise ValueError(
                "producer_profile_id does not belong to canonical_profile_id"
            )
        object.__setattr__(self, "canonical_profile_id", resolved.profile_id)
        object.__setattr__(
            self,
            "producer_profile_id",
            resolved.producer_profile_id,
        )
        for name in (
            "runtime_id",
            "backend_evidence_status_id",
            "method_artifact_id",
            "method_selection_partition_id",
            "object_domain_id",
            "action_domain_id",
            "risk_feature_schema_id",
            "risk_model_id",
            "threshold_source_artifact_id",
            "fallback_policy_id",
        ):
            object.__setattr__(
                self,
                name,
                sha256_digest(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "method_selection_group_ids",
            _sorted_unique_strings(
                self.method_selection_group_ids,
                name="method_selection_group_ids",
            ),
        )
        object.__setattr__(
            self,
            "allowed_query_functional_ids",
            _sorted_unique_digests(
                self.allowed_query_functional_ids,
                name="allowed_query_functional_ids",
            ),
        )
        minimum_horizon = _finite_real(
            self.minimum_horizon_seconds,
            name="minimum_horizon_seconds",
            strictly_positive=True,
        )
        maximum_horizon = _finite_real(
            self.maximum_horizon_seconds,
            name="maximum_horizon_seconds",
            strictly_positive=True,
        )
        if maximum_horizon < minimum_horizon:
            raise ValueError("maximum_horizon_seconds is below the minimum")
        object.__setattr__(self, "minimum_horizon_seconds", minimum_horizon)
        object.__setattr__(self, "maximum_horizon_seconds", maximum_horizon)
        object.__setattr__(
            self,
            "maximum_horizon_step_count",
            genuine_integer(
                self.maximum_horizon_step_count,
                name="maximum_horizon_step_count",
                minimum=1,
            ),
        )
        for name in ("loss_metric", "statistical_unit"):
            object.__setattr__(
                self,
                name,
                _canonical_text(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "risk_threshold",
            _finite_real(self.risk_threshold, name="risk_threshold"),
        )
        required_true = (
            "exact_fallback_required",
            "source_protocol_frozen",
            "risk_policy_frozen_before_certification_outcomes",
        )
        for name in required_true:
            value = genuine_boolean(getattr(self, name), name=name)
            if not value:
                raise ValueError(f"{name} must be true")
            object.__setattr__(self, name, True)
        required_false = (
            "certification_outcomes_used_for_method_selection",
            "target_outcomes_used",
        )
        for name in required_false:
            value = genuine_boolean(getattr(self, name), name=name)
            if value:
                raise ValueError(f"{name} must be false")
            object.__setattr__(self, name, False)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="simulator competence policy metadata",
            ),
        )
        expected = content_id(self.descriptor())
        if self.policy_id is not None:
            supplied = sha256_digest(self.policy_id, name="policy_id")
            if supplied != expected:
                raise ValueError("policy_id does not match simulator competence policy")
        object.__setattr__(self, "policy_id", expected)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": SIMULATOR_COMPETENCE_POLICY_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "claim_boundary": QUERY_CONDITIONAL_COMPETENCE_CLAIM_BOUNDARY,
            "canonical_profile_id": self.canonical_profile_id,
            "producer_profile_id": self.producer_profile_id,
            "runtime_id": self.runtime_id,
            "backend_evidence_status_id": self.backend_evidence_status_id,
            "method_artifact_id": self.method_artifact_id,
            "method_selection_partition_id": self.method_selection_partition_id,
            "method_selection_group_ids": list(self.method_selection_group_ids),
            "object_domain_id": self.object_domain_id,
            "action_domain_id": self.action_domain_id,
            "allowed_query_functional_ids": list(self.allowed_query_functional_ids),
            "minimum_horizon_seconds": self.minimum_horizon_seconds,
            "maximum_horizon_seconds": self.maximum_horizon_seconds,
            "maximum_horizon_step_count": self.maximum_horizon_step_count,
            "loss_metric": self.loss_metric,
            "statistical_unit": self.statistical_unit,
            "risk_feature_schema_id": self.risk_feature_schema_id,
            "risk_model_id": self.risk_model_id,
            "risk_score_semantics": RISK_SCORE_SEMANTICS,
            "threshold_source_artifact_id": self.threshold_source_artifact_id,
            "risk_threshold": self.risk_threshold,
            "fallback_policy_id": self.fallback_policy_id,
            "exact_fallback_required": self.exact_fallback_required,
            "source_protocol_frozen": self.source_protocol_frozen,
            "risk_policy_frozen_before_certification_outcomes": (
                self.risk_policy_frozen_before_certification_outcomes
            ),
            "certification_outcomes_used_for_method_selection": (
                self.certification_outcomes_used_for_method_selection
            ),
            "target_outcomes_used": self.target_outcomes_used,
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "policy_id": self.policy_id}

    @classmethod
    def from_mapping(cls, value: object) -> SimulatorCompetencePolicyV1:
        if not isinstance(value, Mapping):
            raise ValueError("simulator competence policy must be a mapping")
        expected = frozenset(
            {
                "schema",
                "schema_version",
                "claim_boundary",
                "canonical_profile_id",
                "producer_profile_id",
                "runtime_id",
                "backend_evidence_status_id",
                "method_artifact_id",
                "method_selection_partition_id",
                "method_selection_group_ids",
                "object_domain_id",
                "action_domain_id",
                "allowed_query_functional_ids",
                "minimum_horizon_seconds",
                "maximum_horizon_seconds",
                "maximum_horizon_step_count",
                "loss_metric",
                "statistical_unit",
                "risk_feature_schema_id",
                "risk_model_id",
                "risk_score_semantics",
                "threshold_source_artifact_id",
                "risk_threshold",
                "fallback_policy_id",
                "exact_fallback_required",
                "source_protocol_frozen",
                "risk_policy_frozen_before_certification_outcomes",
                "certification_outcomes_used_for_method_selection",
                "target_outcomes_used",
                "metadata",
                "policy_id",
            }
        )
        require_exact_fields(
            value,
            expected=expected,
            name="simulator competence policy",
        )
        if value["schema"] != SIMULATOR_COMPETENCE_POLICY_SCHEMA:
            raise ValueError("simulator competence policy schema changed")
        if value["schema_version"] != SCHEMA_VERSION:
            raise ValueError("simulator competence policy version changed")
        if value["claim_boundary"] != QUERY_CONDITIONAL_COMPETENCE_CLAIM_BOUNDARY:
            raise ValueError("simulator competence claim boundary changed")
        if value["risk_score_semantics"] != RISK_SCORE_SEMANTICS:
            raise ValueError("simulator competence risk-score semantics changed")
        return cls(
            canonical_profile_id=cast(str, value["canonical_profile_id"]),
            producer_profile_id=cast(str, value["producer_profile_id"]),
            runtime_id=cast(str, value["runtime_id"]),
            backend_evidence_status_id=cast(
                str,
                value["backend_evidence_status_id"],
            ),
            method_artifact_id=cast(str, value["method_artifact_id"]),
            method_selection_partition_id=cast(
                str,
                value["method_selection_partition_id"],
            ),
            method_selection_group_ids=cast(
                Sequence[str],
                value["method_selection_group_ids"],
            ),
            object_domain_id=cast(str, value["object_domain_id"]),
            action_domain_id=cast(str, value["action_domain_id"]),
            allowed_query_functional_ids=cast(
                Sequence[str],
                value["allowed_query_functional_ids"],
            ),
            minimum_horizon_seconds=cast(float, value["minimum_horizon_seconds"]),
            maximum_horizon_seconds=cast(float, value["maximum_horizon_seconds"]),
            maximum_horizon_step_count=cast(
                int,
                value["maximum_horizon_step_count"],
            ),
            loss_metric=cast(str, value["loss_metric"]),
            statistical_unit=cast(str, value["statistical_unit"]),
            risk_feature_schema_id=cast(str, value["risk_feature_schema_id"]),
            risk_model_id=cast(str, value["risk_model_id"]),
            threshold_source_artifact_id=cast(
                str,
                value["threshold_source_artifact_id"],
            ),
            risk_threshold=cast(float, value["risk_threshold"]),
            fallback_policy_id=cast(str, value["fallback_policy_id"]),
            exact_fallback_required=cast(bool, value["exact_fallback_required"]),
            source_protocol_frozen=cast(bool, value["source_protocol_frozen"]),
            risk_policy_frozen_before_certification_outcomes=cast(
                bool,
                value["risk_policy_frozen_before_certification_outcomes"],
            ),
            certification_outcomes_used_for_method_selection=cast(
                bool,
                value["certification_outcomes_used_for_method_selection"],
            ),
            target_outcomes_used=cast(bool, value["target_outcomes_used"]),
            metadata=cast(Mapping[str, Any], value["metadata"]),
            policy_id=cast(str, value["policy_id"]),
        )


@dataclass(frozen=True, slots=True)
class QueryConditionalCompetenceCertificateV1:
    """Source-frozen finite-group competence certificate."""

    policy: SimulatorCompetencePolicyV1
    backend_evidence_status: MaterialBackendEvidenceStatusV1
    harm_risk_certificate: GuardHarmRiskCertificateV1
    certificate_frozen_before_target_outcomes: bool
    target_outcomes_used: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.policy, SimulatorCompetencePolicyV1):
            raise TypeError("policy must be SimulatorCompetencePolicyV1")
        if not isinstance(
            self.backend_evidence_status,
            MaterialBackendEvidenceStatusV1,
        ):
            raise TypeError(
                "backend_evidence_status must be MaterialBackendEvidenceStatusV1"
            )
        if not isinstance(self.harm_risk_certificate, GuardHarmRiskCertificateV1):
            raise TypeError("harm_risk_certificate must be GuardHarmRiskCertificateV1")
        status = self.backend_evidence_status
        risk = self.harm_risk_certificate
        policy = self.policy
        if status.stage != "source-competent":
            raise ValueError(
                "a pre-target competence certificate requires exact source-competent "
                "backend evidence"
            )
        if status.artifact_id != policy.backend_evidence_status_id:
            raise ValueError("backend evidence status does not match policy")
        for name in ("canonical_profile_id", "producer_profile_id", "runtime_id"):
            if getattr(status, name) != getattr(policy, name):
                raise ValueError(f"backend evidence {name} does not match policy")
        if not status.exact_fallback_verified:
            raise ValueError("backend evidence must verify exact fallback")
        source_groups = set(status.source_group_ids)
        method_groups = set(policy.method_selection_group_ids)
        if not source_groups.issubset(method_groups):
            raise ValueError(
                "backend source groups must be declared method-selection groups"
            )
        if not risk.certified:
            raise ValueError("harm-risk certificate is not certified")
        if risk.guard_policy_id != policy.policy_id:
            raise ValueError("harm-risk guard policy does not match policy")
        if risk.threshold_source_artifact_id != policy.threshold_source_artifact_id:
            raise ValueError("harm-risk threshold source does not match policy")
        if risk.threshold != policy.risk_threshold:
            raise ValueError("harm-risk threshold does not match policy")
        if risk.metric != policy.loss_metric:
            raise ValueError("harm-risk metric does not match policy")
        if risk.statistical_unit != policy.statistical_unit:
            raise ValueError("harm-risk statistical unit does not match policy")
        if not set(risk.threshold_selection_group_ids).issubset(method_groups):
            raise ValueError(
                "threshold-selection groups must be method-selection groups"
            )
        overlap = sorted(method_groups & set(risk.group_ids))
        if overlap:
            raise ValueError(
                f"method-selection and certification groups overlap: {overlap}"
            )
        frozen = genuine_boolean(
            self.certificate_frozen_before_target_outcomes,
            name="certificate_frozen_before_target_outcomes",
        )
        if not frozen:
            raise ValueError("certificate must be frozen before target outcomes")
        target_used = genuine_boolean(
            self.target_outcomes_used,
            name="target_outcomes_used",
        )
        if target_used:
            raise ValueError("target outcomes cannot enter a competence certificate")
        object.__setattr__(self, "certificate_frozen_before_target_outcomes", True)
        object.__setattr__(self, "target_outcomes_used", False)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="query-conditional competence certificate metadata",
            ),
        )
        expected = content_id(self.descriptor())
        if self.artifact_id is not None:
            supplied = sha256_digest(self.artifact_id, name="artifact_id")
            if supplied != expected:
                raise ValueError("artifact_id does not match competence certificate")
        object.__setattr__(self, "artifact_id", expected)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": QUERY_CONDITIONAL_CERTIFICATE_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "claim_boundary": QUERY_CONDITIONAL_COMPETENCE_CLAIM_BOUNDARY,
            "policy_id": self.policy.policy_id,
            "backend_evidence_status_id": self.backend_evidence_status.artifact_id,
            "harm_risk_certificate_id": self.harm_risk_certificate.artifact_id,
            "certificate_frozen_before_target_outcomes": (
                self.certificate_frozen_before_target_outcomes
            ),
            "target_outcomes_used": self.target_outcomes_used,
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {
            **self.descriptor(),
            "policy": self.policy.to_record(),
            "backend_evidence_status": self.backend_evidence_status.to_payload(),
            "harm_risk_certificate": self.harm_risk_certificate.to_record(),
            "artifact_id": self.artifact_id,
        }

    @classmethod
    def from_mapping(cls, value: object) -> QueryConditionalCompetenceCertificateV1:
        if not isinstance(value, Mapping):
            raise ValueError("competence certificate must be a mapping")
        expected = frozenset(
            {
                "schema",
                "schema_version",
                "claim_boundary",
                "policy_id",
                "backend_evidence_status_id",
                "harm_risk_certificate_id",
                "certificate_frozen_before_target_outcomes",
                "target_outcomes_used",
                "metadata",
                "policy",
                "backend_evidence_status",
                "harm_risk_certificate",
                "artifact_id",
            }
        )
        require_exact_fields(value, expected=expected, name="competence certificate")
        if value["schema"] != QUERY_CONDITIONAL_CERTIFICATE_SCHEMA:
            raise ValueError("competence certificate schema changed")
        if value["schema_version"] != SCHEMA_VERSION:
            raise ValueError("competence certificate version changed")
        if value["claim_boundary"] != QUERY_CONDITIONAL_COMPETENCE_CLAIM_BOUNDARY:
            raise ValueError("competence certificate claim boundary changed")
        policy = SimulatorCompetencePolicyV1.from_mapping(value["policy"])
        status_value = value["backend_evidence_status"]
        if not isinstance(status_value, Mapping):
            raise ValueError("backend_evidence_status must be a mapping")
        status = MaterialBackendEvidenceStatusV1.from_payload(status_value)
        risk = GuardHarmRiskCertificateV1.from_mapping(value["harm_risk_certificate"])
        if value["policy_id"] != policy.policy_id:
            raise ValueError("embedded policy identity changed")
        if value["backend_evidence_status_id"] != status.artifact_id:
            raise ValueError("embedded backend evidence identity changed")
        if value["harm_risk_certificate_id"] != risk.artifact_id:
            raise ValueError("embedded harm-risk identity changed")
        return cls(
            policy=policy,
            backend_evidence_status=status,
            harm_risk_certificate=risk,
            certificate_frozen_before_target_outcomes=cast(
                bool,
                value["certificate_frozen_before_target_outcomes"],
            ),
            target_outcomes_used=cast(bool, value["target_outcomes_used"]),
            metadata=cast(Mapping[str, Any], value["metadata"]),
            artifact_id=cast(str, value["artifact_id"]),
        )


def build_query_conditional_competence_certificate_v1(
    *,
    policy: SimulatorCompetencePolicyV1,
    backend_evidence_status: MaterialBackendEvidenceStatusV1,
    qualification: MaterialBackendQualificationV1,
    source_decision: EvidenceDecisionV1,
    harm_risk_certificate: GuardHarmRiskCertificateV1,
    certificate_frozen_before_target_outcomes: bool,
    target_outcomes_used: bool,
    metadata: Mapping[str, Any] | None = None,
) -> QueryConditionalCompetenceCertificateV1:
    """Replay source evidence and build one pre-target certificate."""

    if backend_evidence_status.stage != "source-competent":
        raise ValueError("backend evidence status must stop at source competence")
    require_material_backend_evidence_stage(
        backend_evidence_status,
        "source-competent",
        qualification=qualification,
        source_decision=source_decision,
    )
    metadata_bindings = {
        "query_competence_method_id": policy.method_artifact_id,
        "method_selection_partition_id": policy.method_selection_partition_id,
    }
    for name, expected in metadata_bindings.items():
        if source_decision.metadata.get(name) != expected:
            raise ValueError(f"source decision metadata {name} does not match")
    return QueryConditionalCompetenceCertificateV1(
        policy=policy,
        backend_evidence_status=backend_evidence_status,
        harm_risk_certificate=harm_risk_certificate,
        certificate_frozen_before_target_outcomes=(
            certificate_frozen_before_target_outcomes
        ),
        target_outcomes_used=target_outcomes_used,
        metadata={} if metadata is None else metadata,
    )


def _runtime_decision_reasons(
    *,
    certificate: QueryConditionalCompetenceCertificateV1,
    query: SimulatorQueryContextV1,
    risk_score: object,
    canonical_profile_id: str,
    producer_profile_id: str,
    runtime_id: str,
    risk_feature_schema_id: str,
    risk_model_id: str,
    fallback_policy_id: str,
) -> tuple[tuple[str, ...], float | None]:
    policy = certificate.policy
    reasons: list[str] = []
    if canonical_profile_id != policy.canonical_profile_id:
        reasons.append("canonical-backend-profile-mismatch")
    if producer_profile_id != policy.producer_profile_id:
        reasons.append("producer-backend-profile-mismatch")
    if runtime_id != policy.runtime_id:
        reasons.append("backend-runtime-mismatch")
    if risk_feature_schema_id != policy.risk_feature_schema_id:
        reasons.append("risk-feature-schema-mismatch")
    if risk_model_id != policy.risk_model_id:
        reasons.append("risk-model-mismatch")
    if fallback_policy_id != policy.fallback_policy_id:
        reasons.append("fallback-policy-mismatch")
    if query.object_domain_id != policy.object_domain_id:
        reasons.append("object-domain-out-of-scope")
    if query.action_domain_id != policy.action_domain_id:
        reasons.append("action-domain-out-of-scope")
    if query.query_functional_id not in policy.allowed_query_functional_ids:
        reasons.append("query-functional-out-of-scope")
    if query.loss_metric != policy.loss_metric:
        reasons.append("query-loss-metric-mismatch")
    if (
        not (
            policy.minimum_horizon_seconds
            <= query.horizon_seconds
            <= policy.maximum_horizon_seconds
        )
        or query.horizon_step_count > policy.maximum_horizon_step_count
    ):
        reasons.append("query-horizon-out-of-scope")
    normalized_score: float | None
    try:
        normalized_score = _finite_real(risk_score, name="risk_score")
    except ValueError:
        normalized_score = None
        reasons.append("risk-score-invalid")
    if normalized_score is not None and normalized_score > policy.risk_threshold:
        reasons.append("risk-score-exceeds-threshold")
    return tuple(reasons or [_AUTHORIZED_REASON]), normalized_score


@dataclass(frozen=True, slots=True)
class QueryConditionalCompetenceDecisionV1:
    """Content-addressed candidate-or-exact-fallback routing decision."""

    certificate_id: str
    query_id: str
    canonical_profile_id: str
    producer_profile_id: str
    runtime_id: str
    risk_feature_schema_id: str
    risk_model_id: str
    fallback_policy_id: str
    candidate_prediction_id: str
    fallback_prediction_id: str
    risk_score: float | None
    authorized: bool
    selected_prediction_id: str
    exact_fallback: bool
    reasons: Sequence[str]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "certificate_id",
            "query_id",
            "runtime_id",
            "risk_feature_schema_id",
            "risk_model_id",
            "fallback_policy_id",
            "candidate_prediction_id",
            "fallback_prediction_id",
            "selected_prediction_id",
        ):
            object.__setattr__(
                self,
                name,
                sha256_digest(getattr(self, name), name=name),
            )
        if self.candidate_prediction_id == self.fallback_prediction_id:
            raise ValueError("candidate and fallback predictions must differ")
        for name in ("canonical_profile_id", "producer_profile_id"):
            object.__setattr__(
                self,
                name,
                _canonical_text(getattr(self, name), name=name),
            )
        score = self.risk_score
        if score is not None:
            score = _finite_real(score, name="risk_score")
        object.__setattr__(self, "risk_score", score)
        authorized = genuine_boolean(self.authorized, name="authorized")
        fallback = genuine_boolean(self.exact_fallback, name="exact_fallback")
        if fallback == authorized:
            raise ValueError("exact_fallback must be the opposite of authorized")
        reasons = _sorted_unique_strings(self.reasons, name="reasons")
        expected_selected = (
            self.candidate_prediction_id if authorized else self.fallback_prediction_id
        )
        if self.selected_prediction_id != expected_selected:
            raise ValueError("selected prediction contradicts competence decision")
        if authorized != (reasons == (_AUTHORIZED_REASON,)):
            raise ValueError("authorized does not match competence reasons")
        object.__setattr__(self, "authorized", authorized)
        object.__setattr__(self, "exact_fallback", fallback)
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="query competence decision metadata",
            ),
        )
        expected = content_id(self.descriptor())
        if self.artifact_id is not None:
            supplied = sha256_digest(self.artifact_id, name="artifact_id")
            if supplied != expected:
                raise ValueError("artifact_id does not match competence decision")
        object.__setattr__(self, "artifact_id", expected)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": QUERY_CONDITIONAL_DECISION_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "claim_boundary": QUERY_CONDITIONAL_COMPETENCE_CLAIM_BOUNDARY,
            "certificate_id": self.certificate_id,
            "query_id": self.query_id,
            "canonical_profile_id": self.canonical_profile_id,
            "producer_profile_id": self.producer_profile_id,
            "runtime_id": self.runtime_id,
            "risk_feature_schema_id": self.risk_feature_schema_id,
            "risk_model_id": self.risk_model_id,
            "fallback_policy_id": self.fallback_policy_id,
            "candidate_prediction_id": self.candidate_prediction_id,
            "fallback_prediction_id": self.fallback_prediction_id,
            "risk_score": self.risk_score,
            "authorized": self.authorized,
            "selected_prediction_id": self.selected_prediction_id,
            "exact_fallback": self.exact_fallback,
            "reasons": list(self.reasons),
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}


def route_query_conditional_prediction_v1(
    *,
    certificate: QueryConditionalCompetenceCertificateV1,
    query: SimulatorQueryContextV1,
    risk_score: object,
    canonical_profile_id: str,
    producer_profile_id: str,
    runtime_id: str,
    risk_feature_schema_id: str,
    risk_model_id: str,
    fallback_policy_id: str,
    candidate_prediction_id: str,
    fallback_prediction_id: str,
    candidate_prediction: object,
    fallback_prediction: object,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[QueryConditionalCompetenceDecisionV1, object]:
    """Route one query and preserve exact fallback object identity on rejection."""

    if not isinstance(certificate, QueryConditionalCompetenceCertificateV1):
        raise TypeError("certificate must be QueryConditionalCompetenceCertificateV1")
    if not isinstance(query, SimulatorQueryContextV1):
        raise TypeError("query must be SimulatorQueryContextV1")
    normalized_ids = {
        "runtime_id": sha256_digest(runtime_id, name="runtime_id"),
        "risk_feature_schema_id": sha256_digest(
            risk_feature_schema_id,
            name="risk_feature_schema_id",
        ),
        "risk_model_id": sha256_digest(risk_model_id, name="risk_model_id"),
        "fallback_policy_id": sha256_digest(
            fallback_policy_id,
            name="fallback_policy_id",
        ),
        "candidate_prediction_id": sha256_digest(
            candidate_prediction_id,
            name="candidate_prediction_id",
        ),
        "fallback_prediction_id": sha256_digest(
            fallback_prediction_id,
            name="fallback_prediction_id",
        ),
    }
    reasons, normalized_score = _runtime_decision_reasons(
        certificate=certificate,
        query=query,
        risk_score=risk_score,
        canonical_profile_id=canonical_profile_id,
        producer_profile_id=producer_profile_id,
        runtime_id=normalized_ids["runtime_id"],
        risk_feature_schema_id=normalized_ids["risk_feature_schema_id"],
        risk_model_id=normalized_ids["risk_model_id"],
        fallback_policy_id=normalized_ids["fallback_policy_id"],
    )
    authorized = reasons == (_AUTHORIZED_REASON,)
    selected_id = (
        normalized_ids["candidate_prediction_id"]
        if authorized
        else normalized_ids["fallback_prediction_id"]
    )
    decision = QueryConditionalCompetenceDecisionV1(
        certificate_id=cast(str, certificate.artifact_id),
        query_id=cast(str, query.artifact_id),
        canonical_profile_id=canonical_profile_id,
        producer_profile_id=producer_profile_id,
        runtime_id=normalized_ids["runtime_id"],
        risk_feature_schema_id=normalized_ids["risk_feature_schema_id"],
        risk_model_id=normalized_ids["risk_model_id"],
        fallback_policy_id=normalized_ids["fallback_policy_id"],
        candidate_prediction_id=normalized_ids["candidate_prediction_id"],
        fallback_prediction_id=normalized_ids["fallback_prediction_id"],
        risk_score=normalized_score,
        authorized=authorized,
        selected_prediction_id=selected_id,
        exact_fallback=not authorized,
        reasons=reasons,
        metadata={} if metadata is None else metadata,
    )
    selected_prediction = candidate_prediction if authorized else fallback_prediction
    return decision, selected_prediction


def save_query_conditional_competence_certificate_v1(
    certificate: QueryConditionalCompetenceCertificateV1,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(certificate, QueryConditionalCompetenceCertificateV1):
        raise TypeError("certificate must be QueryConditionalCompetenceCertificateV1")
    write_atomic_json(certificate.to_record(), path, overwrite=overwrite)


def load_query_conditional_competence_certificate_v1(
    path: str | Path,
) -> QueryConditionalCompetenceCertificateV1:
    value = load_strict_json_object(path, label="competence certificate")
    return QueryConditionalCompetenceCertificateV1.from_mapping(value)


__all__ = [
    "QUERY_CONDITIONAL_COMPETENCE_CLAIM_BOUNDARY",
    "QUERY_CONDITIONAL_CERTIFICATE_SCHEMA",
    "QUERY_CONDITIONAL_DECISION_SCHEMA",
    "QUERY_CONTEXT_SCHEMA",
    "SCHEMA_VERSION",
    "SIMULATOR_COMPETENCE_POLICY_SCHEMA",
    "QueryConditionalCompetenceCertificateV1",
    "QueryConditionalCompetenceDecisionV1",
    "SimulatorCompetencePolicyV1",
    "SimulatorQueryContextV1",
    "build_query_conditional_competence_certificate_v1",
    "load_query_conditional_competence_certificate_v1",
    "route_query_conditional_prediction_v1",
    "save_query_conditional_competence_certificate_v1",
]
