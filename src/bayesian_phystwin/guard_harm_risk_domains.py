"""Simultaneous finite-group harm certification across calibration domains.

A pooled harmful-update bound can pass while a minority physical regime remains
unsafe. This module composes the calibration-frozen domain guard with one exact
finite-group harm certificate per deployable domain. Bonferroni simultaneous
coverage is applied across the complete supported-domain roster, and any failed
or missing domain forces exact complete-belief fallback everywhere.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType
from typing import Any, TypeVar

import numpy as np

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    genuine_integer,
    plain_json,
)
from ._portable_contracts import content_id, sha256_digest
from .calibration_domain_guard import CalibrationDomainGuardCertificateV1
from .complete_belief_selection import (
    ArtifactBelief,
    CompleteBeliefGuardDecisionV1,
    CompleteBeliefSelectionV1,
    select_complete_belief,
)
from .guard_harm_risk import (
    RISK_SCORE_SEMANTICS,
    GuardHarmRiskCertificateV1,
    certify_guard_harm_risk,
)

DOMAIN_GUARD_HARM_RISK_SCHEMA = "bayesian_phystwin.domain_guard_harm_risk_certificate"
DOMAIN_GUARD_HARM_RISK_VERSION = 1
DOMAIN_HARM_RISK_POLICY_SCHEMA = "bayesian_phystwin.domain_harm_risk_policy"
DOMAIN_HARM_RISK_POLICY_VERSION = 1
MULTIPLICITY_METHOD = "bonferroni-supported-domains-v1"

BeliefT = TypeVar("BeliefT", bound=ArtifactBelief)


def _canonical_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    return value


def _canonical_strings(value: object, *, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of canonical strings")
    result = tuple(
        _canonical_string(item, name=f"{name}[{index}]")
        for index, item in enumerate(tuple(value))
    )
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _finite_real(value: object, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _open_probability(value: object, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if not 0.0 < result < 1.0:
        raise ValueError(f"{name} must lie strictly inside (0, 1)")
    return result


def _float_vector(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a one-dimensional numeric array")
    result = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return np.array(result, dtype=np.float64, copy=True, order="C")


def _boolean_vector(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind != "b":
        raise ValueError(f"{name} must be a one-dimensional boolean array")
    return np.array(raw, dtype=np.bool_, copy=True, order="C")


def _required_per_domain_confidence(
    family_confidence_level: float,
    supported_domain_count: int,
) -> float:
    family = _open_probability(
        family_confidence_level,
        name="family_confidence_level",
    )
    count = genuine_integer(
        supported_domain_count,
        name="supported_domain_count",
        minimum=1,
    )
    return 1.0 - (1.0 - family) / count


def domain_harm_risk_policy_id(
    *,
    domain_guard_certificate_id: str,
    domain_decision_id: str,
    domain_id: str,
    threshold_source_artifact_id: str,
    statistical_unit: str,
    metric: str,
    threshold: float,
    harm_margin: float,
    target_harm_probability: float,
    family_confidence_level: float,
    supported_domain_count: int,
) -> str:
    """Return the frozen policy identity for one supported domain."""

    guard_id = sha256_digest(
        domain_guard_certificate_id,
        name="domain_guard_certificate_id",
    )
    decision_id = sha256_digest(domain_decision_id, name="domain_decision_id")
    source_id = sha256_digest(
        threshold_source_artifact_id,
        name="threshold_source_artifact_id",
    )
    domain = _canonical_string(domain_id, name="domain_id")
    unit = _canonical_string(statistical_unit, name="statistical_unit")
    metric_name = _canonical_string(metric, name="metric")
    threshold_value = _finite_real(threshold, name="threshold")
    margin = _finite_real(harm_margin, name="harm_margin", minimum=0.0)
    target = _open_probability(
        target_harm_probability,
        name="target_harm_probability",
    )
    family = _open_probability(
        family_confidence_level,
        name="family_confidence_level",
    )
    count = genuine_integer(
        supported_domain_count,
        name="supported_domain_count",
        minimum=1,
    )
    per_domain = _required_per_domain_confidence(family, count)
    return content_id(
        {
            "schema": DOMAIN_HARM_RISK_POLICY_SCHEMA,
            "schema_version": DOMAIN_HARM_RISK_POLICY_VERSION,
            "domain_guard_certificate_id": guard_id,
            "domain_decision_id": decision_id,
            "domain_id": domain,
            "threshold_source_artifact_id": source_id,
            "statistical_unit": unit,
            "metric": metric_name,
            "risk_score_semantics": RISK_SCORE_SEMANTICS,
            "threshold": threshold_value,
            "harm_margin": margin,
            "target_harm_probability": target,
            "family_confidence_level": family,
            "supported_domain_count": count,
            "multiplicity_method": MULTIPLICITY_METHOD,
            "per_domain_confidence_level": per_domain,
        }
    )


@dataclass(frozen=True, slots=True)
class DomainGuardHarmRiskCertificateV1:
    """Bind one domain guard to simultaneous per-domain harm certificates."""

    domain_guard_certificate: CalibrationDomainGuardCertificateV1
    domain_certificates: Mapping[str, GuardHarmRiskCertificateV1]
    family_confidence_level: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        guard = self.domain_guard_certificate
        if not isinstance(guard, CalibrationDomainGuardCertificateV1):
            raise TypeError(
                "domain_guard_certificate must be a CalibrationDomainGuardCertificateV1"
            )
        if guard.artifact_id is None:
            raise ValueError("domain guard certificate must have an artifact identity")
        family = _open_probability(
            self.family_confidence_level,
            name="family_confidence_level",
        )
        if not isinstance(self.domain_certificates, Mapping):
            raise TypeError("domain_certificates must be a mapping")
        certificates: dict[str, GuardHarmRiskCertificateV1] = {}
        for raw_domain, certificate in self.domain_certificates.items():
            domain = _canonical_string(raw_domain, name="domain certificate key")
            if domain in certificates:
                raise ValueError("domain_certificates must not contain duplicates")
            if not isinstance(certificate, GuardHarmRiskCertificateV1):
                raise TypeError(
                    "domain_certificates values must be "
                    "GuardHarmRiskCertificateV1 records"
                )
            certificates[domain] = certificate

        supported = guard.supported_domains
        if not supported:
            raise ValueError(
                "domain guard authorizes no domain requiring certification"
            )
        if set(certificates) != set(supported):
            missing = sorted(set(supported) - set(certificates))
            extra = sorted(set(certificates) - set(supported))
            raise ValueError(
                "domain certificate roster must equal supported domains; "
                f"missing={missing}, extra={extra}"
            )
        certificates = dict(sorted(certificates.items()))
        required_confidence = _required_per_domain_confidence(
            family,
            len(supported),
        )

        calibration_groups = {
            group_id for decision in guard.decisions for group_id in decision.group_ids
        }
        certification_groups: set[str] = set()
        threshold_selection_groups: set[str] = set()
        reference: GuardHarmRiskCertificateV1 | None = None
        for domain, certificate in certificates.items():
            decision = guard.decision_for_domain(domain)
            if decision is None or not decision.calibration_supported:
                raise ValueError(
                    f"domain {domain!r} is not supported by the domain guard"
                )
            if certificate.statistical_unit != guard.statistical_unit:
                raise ValueError("domain harm certificate statistical unit changed")
            if certificate.metric != guard.metric:
                raise ValueError("domain harm certificate metric changed")
            if certificate.certification_partition_id == guard.calibration_partition_id:
                raise ValueError(
                    "calibration and certification partitions must be distinct"
                )
            if not math.isclose(
                certificate.confidence_level,
                required_confidence,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError(
                    "domain harm certificate confidence does not satisfy "
                    "Bonferroni simultaneous coverage"
                )
            expected_policy_id = domain_harm_risk_policy_id(
                domain_guard_certificate_id=str(guard.artifact_id),
                domain_decision_id=str(decision.artifact_id),
                domain_id=domain,
                threshold_source_artifact_id=(certificate.threshold_source_artifact_id),
                statistical_unit=certificate.statistical_unit,
                metric=certificate.metric,
                threshold=certificate.threshold,
                harm_margin=certificate.harm_margin,
                target_harm_probability=certificate.target_harm_probability,
                family_confidence_level=family,
                supported_domain_count=len(supported),
            )
            if certificate.guard_policy_id != expected_policy_id:
                raise ValueError(
                    f"domain {domain!r} harm certificate is not bound to the "
                    "domain guard policy"
                )
            if reference is None:
                reference = certificate
            else:
                shared_fields = (
                    "threshold_source_artifact_id",
                    "certification_partition_id",
                    "threshold_selection_group_ids",
                    "statistical_unit",
                    "metric",
                    "threshold",
                    "harm_margin",
                    "target_harm_probability",
                    "minimum_accepted_group_count",
                )
                for name in shared_fields:
                    if getattr(certificate, name) != getattr(reference, name):
                        raise ValueError(
                            f"domain harm certificates disagree on shared {name}"
                        )
            overlap = certification_groups & set(certificate.group_ids)
            if overlap:
                raise ValueError(
                    "certification groups must belong to exactly one domain: "
                    f"{sorted(overlap)}"
                )
            certification_groups.update(certificate.group_ids)
            threshold_selection_groups.update(certificate.threshold_selection_group_ids)

        calibration_overlap = calibration_groups & certification_groups
        if calibration_overlap:
            raise ValueError(
                "calibration and certification groups must be disjoint: "
                f"{sorted(calibration_overlap)}"
            )
        calibration_threshold_overlap = calibration_groups & threshold_selection_groups
        if calibration_threshold_overlap:
            raise ValueError(
                "calibration and threshold-selection groups must be globally "
                f"disjoint: {sorted(calibration_threshold_overlap)}"
            )

        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="domain guard harm-risk metadata",
        )
        object.__setattr__(self, "family_confidence_level", family)
        object.__setattr__(
            self,
            "domain_certificates",
            MappingProxyType(certificates),
        )
        object.__setattr__(self, "metadata", metadata)
        expected_id = content_id(self.descriptor())
        supplied_id = self.artifact_id
        if supplied_id is not None:
            supplied_id = sha256_digest(supplied_id, name="artifact_id")
            if supplied_id != expected_id:
                raise ValueError("artifact_id does not match domain harm certificate")
        object.__setattr__(self, "artifact_id", expected_id)

    @property
    def supported_domains(self) -> tuple[str, ...]:
        return self.domain_guard_certificate.supported_domains

    @property
    def per_domain_confidence_level(self) -> float:
        return _required_per_domain_confidence(
            self.family_confidence_level,
            len(self.supported_domains),
        )

    @property
    def certified_domains(self) -> tuple[str, ...]:
        return tuple(
            domain
            for domain, certificate in self.domain_certificates.items()
            if certificate.certified
        )

    @property
    def failed_domains(self) -> tuple[str, ...]:
        return tuple(
            domain
            for domain, certificate in self.domain_certificates.items()
            if not certificate.certified
        )

    @property
    def all_supported_domains_certified(self) -> bool:
        return not self.failed_domains

    @property
    def deployment_admissible(self) -> bool:
        return (
            self.domain_guard_certificate.deployment_admissible
            and self.all_supported_domains_certified
        )

    def certificate_for_domain(
        self,
        domain_id: str,
    ) -> GuardHarmRiskCertificateV1 | None:
        domain = _canonical_string(domain_id, name="domain_id")
        return self.domain_certificates.get(domain)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": DOMAIN_GUARD_HARM_RISK_SCHEMA,
            "schema_version": DOMAIN_GUARD_HARM_RISK_VERSION,
            "multiplicity_method": MULTIPLICITY_METHOD,
            "family_confidence_level": self.family_confidence_level,
            "per_domain_confidence_level": self.per_domain_confidence_level,
            "supported_domains": list(self.supported_domains),
            "certified_domains": list(self.certified_domains),
            "failed_domains": list(self.failed_domains),
            "all_supported_domains_certified": (self.all_supported_domains_certified),
            "deployment_admissible": self.deployment_admissible,
            "domain_guard_certificate": (self.domain_guard_certificate.to_record()),
            "domain_certificates": [
                {
                    "domain_id": domain,
                    "certificate": certificate.to_record(),
                }
                for domain, certificate in self.domain_certificates.items()
            ],
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}


def certify_domain_guard_harm_risk(
    *,
    domain_guard_certificate: CalibrationDomainGuardCertificateV1,
    threshold_source_artifact_id: str,
    certification_partition_id: str,
    threshold_selection_group_ids: Sequence[str],
    group_ids: Sequence[str],
    domain_ids: Sequence[str],
    risk_scores: object,
    candidate_losses: object,
    fallback_losses: object,
    fallback_identity_verified: object,
    threshold: float,
    harm_margin: float,
    target_harm_probability: float,
    family_confidence_level: float,
    minimum_accepted_group_count: int,
    threshold_frozen_before_certification_outcomes: bool,
    certification_outcomes_used_for_threshold_selection: bool,
    certification_groups_independent: bool,
    metadata: Mapping[str, Any] | None = None,
) -> DomainGuardHarmRiskCertificateV1:
    """Certify every domain authorized by one calibration-frozen guard."""

    if not isinstance(domain_guard_certificate, CalibrationDomainGuardCertificateV1):
        raise TypeError(
            "domain_guard_certificate must be a CalibrationDomainGuardCertificateV1"
        )
    if domain_guard_certificate.artifact_id is None:
        raise ValueError("domain guard certificate must have an artifact identity")
    supported = domain_guard_certificate.supported_domains
    if not supported:
        raise ValueError("domain guard authorizes no domain requiring certification")
    groups = _canonical_strings(group_ids, name="group_ids")
    domains = _canonical_strings(domain_ids, name="domain_ids")
    if len(set(groups)) != len(groups):
        raise ValueError("group_ids must not contain duplicates")
    scores = _float_vector(risk_scores, name="risk_scores")
    candidate = _float_vector(candidate_losses, name="candidate_losses")
    fallback = _float_vector(fallback_losses, name="fallback_losses")
    fallback_verified = _boolean_vector(
        fallback_identity_verified,
        name="fallback_identity_verified",
    )
    if not (
        len(groups)
        == len(domains)
        == len(scores)
        == len(candidate)
        == len(fallback)
        == len(fallback_verified)
    ):
        raise ValueError("certification identifiers and arrays must have equal lengths")
    if set(domains) != set(supported):
        missing = sorted(set(supported) - set(domains))
        extra = sorted(set(domains) - set(supported))
        raise ValueError(
            "certification domains must equal supported domains; "
            f"missing={missing}, extra={extra}"
        )
    family = _open_probability(
        family_confidence_level,
        name="family_confidence_level",
    )
    per_domain_confidence = _required_per_domain_confidence(
        family,
        len(supported),
    )
    threshold_value = _finite_real(threshold, name="threshold")
    margin = _finite_real(harm_margin, name="harm_margin", minimum=0.0)
    target = _open_probability(
        target_harm_probability,
        name="target_harm_probability",
    )
    minimum_accepted = genuine_integer(
        minimum_accepted_group_count,
        name="minimum_accepted_group_count",
        minimum=1,
    )
    threshold_frozen = genuine_boolean(
        threshold_frozen_before_certification_outcomes,
        name="threshold_frozen_before_certification_outcomes",
    )
    outcomes_used = genuine_boolean(
        certification_outcomes_used_for_threshold_selection,
        name="certification_outcomes_used_for_threshold_selection",
    )
    groups_independent = genuine_boolean(
        certification_groups_independent,
        name="certification_groups_independent",
    )

    certificates: dict[str, GuardHarmRiskCertificateV1] = {}
    for domain in supported:
        decision = domain_guard_certificate.decision_for_domain(domain)
        if decision is None or decision.artifact_id is None:
            raise ValueError(f"supported domain {domain!r} lacks a decision identity")
        indices = np.asarray(
            [index for index, value in enumerate(domains) if value == domain],
            dtype=np.int64,
        )
        policy_id = domain_harm_risk_policy_id(
            domain_guard_certificate_id=str(domain_guard_certificate.artifact_id),
            domain_decision_id=str(decision.artifact_id),
            domain_id=domain,
            threshold_source_artifact_id=threshold_source_artifact_id,
            statistical_unit=domain_guard_certificate.statistical_unit,
            metric=domain_guard_certificate.metric,
            threshold=threshold_value,
            harm_margin=margin,
            target_harm_probability=target,
            family_confidence_level=family,
            supported_domain_count=len(supported),
        )
        certificates[domain] = certify_guard_harm_risk(
            guard_policy_id=policy_id,
            threshold_source_artifact_id=threshold_source_artifact_id,
            certification_partition_id=certification_partition_id,
            statistical_unit=domain_guard_certificate.statistical_unit,
            metric=domain_guard_certificate.metric,
            threshold_selection_group_ids=threshold_selection_group_ids,
            group_ids=tuple(groups[int(index)] for index in indices),
            risk_scores=scores[indices],
            candidate_losses=candidate[indices],
            fallback_losses=fallback[indices],
            fallback_identity_verified=fallback_verified[indices],
            threshold=threshold_value,
            harm_margin=margin,
            target_harm_probability=target,
            confidence_level=per_domain_confidence,
            minimum_accepted_group_count=minimum_accepted,
            threshold_frozen_before_certification_outcomes=threshold_frozen,
            certification_outcomes_used_for_threshold_selection=outcomes_used,
            certification_groups_independent=groups_independent,
            metadata={
                "domain_id": domain,
                "domain_guard_certificate_id": str(
                    domain_guard_certificate.artifact_id
                ),
                "domain_decision_id": str(decision.artifact_id),
            },
        )
    return DomainGuardHarmRiskCertificateV1(
        domain_guard_certificate=domain_guard_certificate,
        domain_certificates=certificates,
        family_confidence_level=family,
        metadata={} if metadata is None else metadata,
    )


def select_domain_guard_harm_risk_belief(
    baseline: BeliefT,
    candidate: BeliefT,
    certificate: DomainGuardHarmRiskCertificateV1,
    *,
    domain_id: str,
    common_domain_id: str,
    inference_admissible: bool,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[BeliefT, CompleteBeliefSelectionV1]:
    """Select candidate only after simultaneous domain harm certification."""

    if not isinstance(certificate, DomainGuardHarmRiskCertificateV1):
        raise TypeError("certificate must be a DomainGuardHarmRiskCertificateV1")
    domain = _canonical_string(domain_id, name="domain_id")
    common = sha256_digest(common_domain_id, name="common_domain_id")
    inference_ok = genuine_boolean(
        inference_admissible,
        name="inference_admissible",
    )
    guard = certificate.domain_guard_certificate
    decision = guard.decision_for_domain(domain)
    domain_certificate = certificate.certificate_for_domain(domain)
    if not inference_ok:
        reason = "inference-rejected"
    elif decision is None:
        reason = "unknown-calibration-domain"
    elif not guard.deployment_admissible:
        reason = "calibration-information-boundary-rejected"
    elif not decision.calibration_supported:
        reason = "calibration-domain-rejected"
    elif domain_certificate is None:
        reason = "domain-harm-risk-certificate-missing"
    elif not domain_certificate.certified:
        reason = "domain-harm-risk-rejected"
    elif not certificate.all_supported_domains_certified:
        reason = "cross-domain-harm-risk-rejected"
    else:
        reason = "domain-harm-risk-authorized"
    accepted = reason == "domain-harm-risk-authorized"
    caller_metadata = frozen_finite_json_mapping(
        metadata,
        name="domain harm-risk selection metadata",
    )
    routing_metadata = {
        "guard": DOMAIN_GUARD_HARM_RISK_SCHEMA,
        "domain_id": domain,
        "domain_guard_certificate_id": str(guard.artifact_id),
        "domain_decision_id": None if decision is None else decision.artifact_id,
        "domain_harm_certificate_id": (
            None if domain_certificate is None else domain_certificate.artifact_id
        ),
        "domain_harm_certified": bool(
            domain_certificate is not None and domain_certificate.certified
        ),
        "domain_harm_upper_bound": (
            None
            if domain_certificate is None
            else domain_certificate.one_sided_upper_bound
        ),
        "family_confidence_level": certificate.family_confidence_level,
        "per_domain_confidence_level": certificate.per_domain_confidence_level,
        "failed_domains": list(certificate.failed_domains),
        "all_supported_domains_certified": (
            certificate.all_supported_domains_certified
        ),
        "certificate_deployment_admissible": certificate.deployment_admissible,
        "routing_reason": reason,
        "caller": plain_json(caller_metadata),
    }
    guard_decision = CompleteBeliefGuardDecisionV1(
        baseline_belief_id=baseline.artifact_id,
        candidate_belief_id=candidate.artifact_id,
        common_domain_id=common,
        certificate_id=str(certificate.artifact_id),
        inference_admissible=inference_ok,
        regret_guard_accepted=accepted,
        reason=reason,
        metadata=routing_metadata,
    )
    return select_complete_belief(
        baseline,
        candidate,
        guard_decision,
        metadata=routing_metadata,
    )


__all__ = [
    "DOMAIN_GUARD_HARM_RISK_SCHEMA",
    "DOMAIN_GUARD_HARM_RISK_VERSION",
    "DOMAIN_HARM_RISK_POLICY_SCHEMA",
    "DOMAIN_HARM_RISK_POLICY_VERSION",
    "MULTIPLICITY_METHOD",
    "DomainGuardHarmRiskCertificateV1",
    "certify_domain_guard_harm_risk",
    "domain_harm_risk_policy_id",
    "select_domain_guard_harm_risk_belief",
]
