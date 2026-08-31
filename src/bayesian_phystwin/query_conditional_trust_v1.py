"""Compose simulator competence with exact query-decision sufficiency.

A simulator is useful for a plan only when two distinct statements hold:

1. its prediction is source-qualified and selectively competent for the exact
   object, action, horizon, and physical query; and
2. the action selected from that prediction has bounded worst-case regret over
   every prior-supported latent belief consistent with the registered query
   posterior.

This module requires both certificates and otherwise returns the caller-owned
fallback plan object exactly. It does not turn either certificate into a safety
guarantee or authorize a different action, query, loss, runtime, or domain.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Final, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from ._canonical_contracts import (
    canonical_string_tuple,
    frozen_finite_json_mapping,
    genuine_boolean,
    immutable_array,
    plain_json,
)
from ._portable_contracts import content_id, sha256_digest
from .query_decision_certificate_v1 import (
    QUERY_DECISION_CERTIFICATE_CLAIM_BOUNDARY,
    QUERY_DECISION_CERTIFICATE_SEMANTICS,
    QUERY_DECISION_CERTIFICATE_VERSION,
    QueryDecisionCertificateV1,
    query_decision_certificate,
)
from .simulator_competence_v1 import (
    QueryConditionalCompetenceCertificateV1,
    QueryConditionalCompetenceDecisionV1,
    SimulatorQueryContextV1,
    route_query_conditional_prediction_v1,
)

REGISTERED_QUERY_DECISION_SCHEMA: Final = (
    "bayesian-phystwin.registered-query-decision-certificate"
)
QUERY_CONDITIONAL_TRUST_DECISION_SCHEMA: Final = (
    "bayesian-phystwin.query-conditional-trust-decision"
)
SCHEMA_VERSION: Final = 1
DECISION_ADMISSION_RULE: Final = (
    "candidate-worst-case-regret-at-most-registered-tolerance-v1"
)
QUERY_CONDITIONAL_TRUST_CLAIM_BOUNDARY: Final = (
    "A plan is admitted only when the registered simulator-competence router "
    "accepts the exact object/action/horizon/query context and the same action "
    "has bounded worst-case regret over every prior-supported latent belief "
    "consistent with the registered query posterior. Rejection returns the "
    "caller-owned fallback plan exactly. This does not establish universal "
    "simulator validity, provider correctness, loss correctness, deployment "
    "safety, or transfer outside the registered domain."
)
_AUTHORIZED_REASON: Final = "query-conditional-plan-authorized"

FloatArray: TypeAlias = npt.NDArray[np.float64]


def _canonical_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    return value


def _finite_nonnegative_or_none(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative real number or null")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative real number or null")
    return result


def _ordered_unique_digests(
    values: Sequence[str],
    *,
    name: str,
) -> tuple[str, ...]:
    strings = canonical_string_tuple(values, name=name, allow_empty=False)
    digests = tuple(
        sha256_digest(value, name=f"{name}[{index}]")
        for index, value in enumerate(strings)
    )
    if len(set(digests)) != len(digests):
        raise ValueError(f"{name} must contain unique digests")
    return digests


def _canonical_reasons(values: Sequence[str]) -> tuple[str, ...]:
    reasons = canonical_string_tuple(values, name="reasons", allow_empty=False)
    if len(set(reasons)) != len(reasons):
        raise ValueError("reasons must contain unique strings")
    if any(value.strip() != value for value in reasons):
        raise ValueError("reasons must contain canonical strings")
    return tuple(sorted(reasons))


def _loss_matrix(
    value: object,
    *,
    hypothesis_count: int,
) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError("loss_by_hypothesis_action must contain real numbers")
    losses = cast(FloatArray, immutable_array(raw, dtype=np.float64))
    if losses.ndim != 2 or losses.shape[0] != hypothesis_count:
        raise ValueError(
            "loss_by_hypothesis_action must have shape "
            "(hypothesis_count, action_count)"
        )
    if losses.shape[1] < 2:
        raise ValueError("at least two registered actions are required")
    if not np.all(np.isfinite(losses)):
        raise ValueError("loss_by_hypothesis_action must be finite")
    return losses


def _decision_certificate_descriptor(
    certificate: QueryDecisionCertificateV1,
    loss_by_hypothesis_action: FloatArray,
) -> dict[str, object]:
    return {
        "schema": "bayesian-phystwin.registered-finite-decision-problem",
        "schema_version": QUERY_DECISION_CERTIFICATE_VERSION,
        "semantics": QUERY_DECISION_CERTIFICATE_SEMANTICS,
        "claim_boundary": QUERY_DECISION_CERTIFICATE_CLAIM_BOUNDARY,
        "prior_weights": certificate.prior_weights.tolist(),
        "prior_support_mask": certificate.prior_support_mask.tolist(),
        "quotient_weights": certificate.quotient_weights.tolist(),
        "class_index": certificate.class_index.tolist(),
        "loss_by_hypothesis_action": loss_by_hypothesis_action.tolist(),
        "class_pairwise_max_loss_gap": (
            certificate.class_pairwise_max_loss_gap.tolist()
        ),
        "pairwise_worst_case_loss_gap": (
            certificate.pairwise_worst_case_loss_gap.tolist()
        ),
        "worst_case_regret": certificate.worst_case_regret.tolist(),
        "minimax_action_index": certificate.minimax_action_index,
        "minimax_worst_case_regret": certificate.minimax_worst_case_regret,
        "regret_tolerance": certificate.regret_tolerance,
        "tolerance_admissible_action_mask": (
            certificate.tolerance_admissible_action_mask.tolist()
        ),
        "robustly_optimal_action_mask": (
            certificate.robustly_optimal_action_mask.tolist()
        ),
    }


@dataclass(frozen=True, slots=True)
class RegisteredQueryDecisionCertificateV1:
    """One outcome-unopened finite decision problem and its exact solution."""

    certificate: QueryDecisionCertificateV1
    loss_by_hypothesis_action: Any
    action_ids: Sequence[str]
    action_domain_id: str
    query_functional_id: str
    loss_metric: str
    hypothesis_set_artifact_id: str
    quotient_registration_id: str
    quotient_posterior_artifact_id: str
    loss_model_artifact_id: str
    certificate_frozen_before_target_outcomes: bool = True
    target_outcomes_used: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.certificate, QueryDecisionCertificateV1):
            raise TypeError("certificate must be QueryDecisionCertificateV1")
        losses = _loss_matrix(
            self.loss_by_hypothesis_action,
            hypothesis_count=self.certificate.hypothesis_count,
        )
        expected = query_decision_certificate(
            self.certificate.prior_weights,
            self.certificate.quotient_weights,
            self.certificate.class_index,
            losses,
            regret_tolerance=self.certificate.regret_tolerance,
        )
        supplied_id = content_id(
            _decision_certificate_descriptor(self.certificate, losses)
        )
        expected_id = content_id(_decision_certificate_descriptor(expected, losses))
        if supplied_id != expected_id:
            raise ValueError("query decision certificate does not match recomputation")
        object.__setattr__(self, "certificate", expected)
        object.__setattr__(self, "loss_by_hypothesis_action", losses)
        action_ids = _ordered_unique_digests(self.action_ids, name="action_ids")
        if len(action_ids) != expected.action_count:
            raise ValueError("action_ids must match the decision action count")
        object.__setattr__(self, "action_ids", action_ids)
        for name in (
            "action_domain_id",
            "query_functional_id",
            "hypothesis_set_artifact_id",
            "quotient_registration_id",
            "quotient_posterior_artifact_id",
            "loss_model_artifact_id",
        ):
            object.__setattr__(
                self,
                name,
                sha256_digest(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "loss_metric",
            _canonical_text(self.loss_metric, name="loss_metric"),
        )
        frozen = genuine_boolean(
            self.certificate_frozen_before_target_outcomes,
            name="certificate_frozen_before_target_outcomes",
        )
        if not frozen:
            raise ValueError("decision certificate must be frozen before outcomes")
        target_used = genuine_boolean(
            self.target_outcomes_used,
            name="target_outcomes_used",
        )
        if target_used:
            raise ValueError("target outcomes cannot enter a decision certificate")
        object.__setattr__(self, "certificate_frozen_before_target_outcomes", True)
        object.__setattr__(self, "target_outcomes_used", False)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="registered query decision metadata",
            ),
        )
        expected = content_id(self.descriptor())
        if self.artifact_id is not None:
            supplied = sha256_digest(self.artifact_id, name="artifact_id")
            if supplied != expected:
                raise ValueError("artifact_id does not match registered decision")
        object.__setattr__(self, "artifact_id", expected)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": REGISTERED_QUERY_DECISION_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "claim_boundary": QUERY_CONDITIONAL_TRUST_CLAIM_BOUNDARY,
            "decision_admission_rule": DECISION_ADMISSION_RULE,
            "decision_certificate_id": content_id(
                _decision_certificate_descriptor(
                    self.certificate,
                    self.loss_by_hypothesis_action,
                )
            ),
            "decision_certificate": _decision_certificate_descriptor(
                self.certificate,
                self.loss_by_hypothesis_action,
            ),
            "action_ids": list(self.action_ids),
            "action_domain_id": self.action_domain_id,
            "query_functional_id": self.query_functional_id,
            "loss_metric": self.loss_metric,
            "hypothesis_set_artifact_id": self.hypothesis_set_artifact_id,
            "quotient_registration_id": self.quotient_registration_id,
            "quotient_posterior_artifact_id": self.quotient_posterior_artifact_id,
            "loss_model_artifact_id": self.loss_model_artifact_id,
            "certificate_frozen_before_target_outcomes": (
                self.certificate_frozen_before_target_outcomes
            ),
            "target_outcomes_used": self.target_outcomes_used,
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}


@dataclass(frozen=True, slots=True)
class QueryConditionalTrustDecisionV1:
    """Content-addressed conjunction of simulator and decision certificates."""

    competence_decision_id: str
    registered_decision_id: str
    query_id: str
    candidate_action_id: str
    fallback_action_id: str
    candidate_plan_id: str
    fallback_plan_id: str
    selected_action_id: str
    selected_plan_id: str
    candidate_worst_case_regret: float | None
    regret_tolerance: float
    simulator_authorized: bool
    decision_authorized: bool
    authorized: bool
    exact_fallback: bool
    reasons: Sequence[str]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "competence_decision_id",
            "registered_decision_id",
            "query_id",
            "candidate_action_id",
            "fallback_action_id",
            "candidate_plan_id",
            "fallback_plan_id",
            "selected_action_id",
            "selected_plan_id",
        ):
            object.__setattr__(
                self,
                name,
                sha256_digest(getattr(self, name), name=name),
            )
        if self.candidate_action_id == self.fallback_action_id:
            raise ValueError("candidate and fallback actions must differ")
        if self.candidate_plan_id == self.fallback_plan_id:
            raise ValueError("candidate and fallback plans must differ")
        object.__setattr__(
            self,
            "candidate_worst_case_regret",
            _finite_nonnegative_or_none(
                self.candidate_worst_case_regret,
                name="candidate_worst_case_regret",
            ),
        )
        tolerance = _finite_nonnegative_or_none(
            self.regret_tolerance,
            name="regret_tolerance",
        )
        assert tolerance is not None
        object.__setattr__(self, "regret_tolerance", tolerance)
        simulator_authorized = genuine_boolean(
            self.simulator_authorized,
            name="simulator_authorized",
        )
        decision_authorized = genuine_boolean(
            self.decision_authorized,
            name="decision_authorized",
        )
        authorized = genuine_boolean(self.authorized, name="authorized")
        fallback = genuine_boolean(self.exact_fallback, name="exact_fallback")
        if authorized != (simulator_authorized and decision_authorized):
            raise ValueError("authorization must require both certificates")
        if fallback == authorized:
            raise ValueError("exact_fallback must be the opposite of authorized")
        expected_action = (
            self.candidate_action_id if authorized else self.fallback_action_id
        )
        expected_plan = self.candidate_plan_id if authorized else self.fallback_plan_id
        if self.selected_action_id != expected_action:
            raise ValueError("selected action contradicts trust decision")
        if self.selected_plan_id != expected_plan:
            raise ValueError("selected plan contradicts trust decision")
        reasons = _canonical_reasons(self.reasons)
        if authorized != (reasons == (_AUTHORIZED_REASON,)):
            raise ValueError("authorized does not match trust decision reasons")
        object.__setattr__(self, "simulator_authorized", simulator_authorized)
        object.__setattr__(self, "decision_authorized", decision_authorized)
        object.__setattr__(self, "authorized", authorized)
        object.__setattr__(self, "exact_fallback", fallback)
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="query conditional trust decision metadata",
            ),
        )
        expected = content_id(self.descriptor())
        if self.artifact_id is not None:
            supplied = sha256_digest(self.artifact_id, name="artifact_id")
            if supplied != expected:
                raise ValueError("artifact_id does not match trust decision")
        object.__setattr__(self, "artifact_id", expected)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": QUERY_CONDITIONAL_TRUST_DECISION_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "claim_boundary": QUERY_CONDITIONAL_TRUST_CLAIM_BOUNDARY,
            "decision_admission_rule": DECISION_ADMISSION_RULE,
            "competence_decision_id": self.competence_decision_id,
            "registered_decision_id": self.registered_decision_id,
            "query_id": self.query_id,
            "candidate_action_id": self.candidate_action_id,
            "fallback_action_id": self.fallback_action_id,
            "candidate_plan_id": self.candidate_plan_id,
            "fallback_plan_id": self.fallback_plan_id,
            "selected_action_id": self.selected_action_id,
            "selected_plan_id": self.selected_plan_id,
            "candidate_worst_case_regret": self.candidate_worst_case_regret,
            "regret_tolerance": self.regret_tolerance,
            "simulator_authorized": self.simulator_authorized,
            "decision_authorized": self.decision_authorized,
            "authorized": self.authorized,
            "exact_fallback": self.exact_fallback,
            "reasons": list(self.reasons),
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}


def route_query_conditional_plan_v1(
    *,
    competence_certificate: QueryConditionalCompetenceCertificateV1,
    query: SimulatorQueryContextV1,
    risk_score: object,
    canonical_profile_id: str,
    producer_profile_id: str,
    runtime_id: str,
    risk_feature_schema_id: str,
    risk_model_id: str,
    fallback_policy_id: str,
    registered_decision: RegisteredQueryDecisionCertificateV1,
    candidate_action_id: str,
    fallback_action_id: str,
    candidate_plan_id: str,
    fallback_plan_id: str,
    candidate_plan: object,
    fallback_plan: object,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[
    QueryConditionalTrustDecisionV1,
    QueryConditionalCompetenceDecisionV1,
    object,
]:
    """Admit one plan only when simulator and decision certificates agree."""

    if not isinstance(registered_decision, RegisteredQueryDecisionCertificateV1):
        raise TypeError(
            "registered_decision must be RegisteredQueryDecisionCertificateV1"
        )
    if registered_decision.artifact_id != content_id(
        registered_decision.descriptor()
    ):
        raise ValueError("registered decision artifact identity changed")
    if candidate_plan is fallback_plan:
        raise ValueError("candidate and fallback plans must be distinct objects")
    competence_decision, _ = route_query_conditional_prediction_v1(
        certificate=competence_certificate,
        query=query,
        risk_score=risk_score,
        canonical_profile_id=canonical_profile_id,
        producer_profile_id=producer_profile_id,
        runtime_id=runtime_id,
        risk_feature_schema_id=risk_feature_schema_id,
        risk_model_id=risk_model_id,
        fallback_policy_id=fallback_policy_id,
        candidate_prediction_id=candidate_plan_id,
        fallback_prediction_id=fallback_plan_id,
        candidate_prediction=candidate_plan,
        fallback_prediction=fallback_plan,
    )
    candidate_action = sha256_digest(
        candidate_action_id,
        name="candidate_action_id",
    )
    fallback_action = sha256_digest(
        fallback_action_id,
        name="fallback_action_id",
    )
    candidate_plan_digest = sha256_digest(candidate_plan_id, name="candidate_plan_id")
    fallback_plan_digest = sha256_digest(fallback_plan_id, name="fallback_plan_id")
    reasons: list[str] = []
    if not competence_decision.authorized:
        reasons.extend(
            f"simulator-{reason}" for reason in competence_decision.reasons
        )

    if registered_decision.action_domain_id != query.action_domain_id:
        reasons.append("decision-action-domain-mismatch")
    if registered_decision.query_functional_id != query.query_functional_id:
        reasons.append("decision-query-functional-mismatch")
    if registered_decision.loss_metric != query.loss_metric:
        reasons.append("decision-loss-metric-mismatch")
    if (
        registered_decision.quotient_posterior_artifact_id
        != query.preoutcome_features_id
    ):
        reasons.append("decision-preoutcome-posterior-mismatch")
    if query.action_context_id != candidate_action:
        reasons.append("query-candidate-action-mismatch")

    candidate_index: int | None = None
    try:
        candidate_index = tuple(registered_decision.action_ids).index(candidate_action)
    except ValueError:
        reasons.append("candidate-action-outside-registered-set")
    if fallback_action not in registered_decision.action_ids:
        reasons.append("fallback-action-outside-registered-set")

    candidate_regret: float | None = None
    decision_authorized = False
    if candidate_index is not None:
        exact = query_decision_certificate(
            registered_decision.certificate.prior_weights,
            registered_decision.certificate.quotient_weights,
            registered_decision.certificate.class_index,
            registered_decision.loss_by_hypothesis_action,
            regret_tolerance=registered_decision.certificate.regret_tolerance,
        )
        candidate_regret = float(exact.worst_case_regret[candidate_index])
        decision_authorized = bool(
            exact.tolerance_admissible_action_mask[candidate_index]
        )
        if not decision_authorized:
            reasons.append("candidate-action-regret-exceeds-tolerance")
    if any(reason.startswith("decision-") for reason in reasons) or any(
        reason in {
            "query-candidate-action-mismatch",
            "candidate-action-outside-registered-set",
            "fallback-action-outside-registered-set",
        }
        for reason in reasons
    ):
        decision_authorized = False

    authorized = competence_decision.authorized and decision_authorized
    final_reasons = (_AUTHORIZED_REASON,) if authorized else tuple(sorted(set(reasons)))
    if not final_reasons:
        final_reasons = ("decision-certificate-not-authorized",)
    selected_action = candidate_action if authorized else fallback_action
    selected_plan_id = candidate_plan_digest if authorized else fallback_plan_digest
    decision = QueryConditionalTrustDecisionV1(
        competence_decision_id=cast(str, competence_decision.artifact_id),
        registered_decision_id=cast(str, registered_decision.artifact_id),
        query_id=cast(str, query.artifact_id),
        candidate_action_id=candidate_action,
        fallback_action_id=fallback_action,
        candidate_plan_id=candidate_plan_digest,
        fallback_plan_id=fallback_plan_digest,
        selected_action_id=selected_action,
        selected_plan_id=selected_plan_id,
        candidate_worst_case_regret=candidate_regret,
        regret_tolerance=registered_decision.certificate.regret_tolerance,
        simulator_authorized=competence_decision.authorized,
        decision_authorized=decision_authorized,
        authorized=authorized,
        exact_fallback=not authorized,
        reasons=final_reasons,
        metadata={} if metadata is None else metadata,
    )
    selected_plan = candidate_plan if authorized else fallback_plan
    return decision, competence_decision, selected_plan


__all__ = [
    "DECISION_ADMISSION_RULE",
    "QUERY_CONDITIONAL_TRUST_CLAIM_BOUNDARY",
    "QUERY_CONDITIONAL_TRUST_DECISION_SCHEMA",
    "REGISTERED_QUERY_DECISION_SCHEMA",
    "SCHEMA_VERSION",
    "QueryConditionalTrustDecisionV1",
    "RegisteredQueryDecisionCertificateV1",
    "route_query_conditional_plan_v1",
]
