"""Claim-bearing admission for domain covariance calibration.

The version-1 fitter remains unchanged. This additive layer binds physical
semantics, applies stronger finite-group criteria, and replaces the public
application Boolean with a content-addressed evidence decision.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    genuine_integer,
    plain_json,
)
from ._portable_contracts import content_id, sha256_digest
from .calibration_domain_guard import CalibrationDomainGuardCertificateV1
from .domain_covariance_calibration import (
    DomainCovarianceCalibrationApplicationV1,
    DomainCovarianceCalibrationCertificateV1,
    DomainCovarianceCalibrationConfigV1,
    DomainCovarianceCalibrationDecisionV1,
    apply_domain_covariance_calibration,
    fit_domain_covariance_calibration,
)
from .evidence_decision_v1 import EvidenceDecisionV1

CALIBRATION_V2_SCHEMA = "bayesian_phystwin.domain_covariance_calibration_v2"
CALIBRATION_V2_VERSION = 2
APPLICATION_V2_SCHEMA = "bayesian_phystwin.domain_covariance_calibration_application_v2"
APPLICATION_V2_VERSION = 2


def _text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    return value


def _number(
    value: object,
    *,
    name: str,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite number")
    result = float(raw.item())
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def _numeric_array_digest(value: np.ndarray) -> str:
    """Digest canonical float64 numerical content and shape."""

    array = np.ascontiguousarray(value, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(b"\0float64\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _exact_array_digest(value: np.ndarray) -> str:
    """Digest the exact returned array shape, dtype, and bytes."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _covariance_dimension(value: np.ndarray) -> int:
    if not isinstance(value, np.ndarray):
        raise TypeError("raw_covariance must be a numpy.ndarray")
    if value.dtype.kind not in "iuf":
        raise ValueError("raw_covariance must contain real numeric values")
    if value.ndim not in {2, 3}:
        raise ValueError("raw_covariance must have shape (d, d) or (m, d, d)")
    if value.shape[-2] < 1 or value.shape[-2] != value.shape[-1]:
        raise ValueError("raw_covariance matrices must be nonempty and square")
    if not np.all(np.isfinite(value)):
        raise ValueError("raw_covariance must contain only finite values")
    return int(value.shape[-1])


@dataclass(frozen=True, slots=True)
class CovarianceSemanticsV2:
    """Physical and query semantics for one covariance family."""

    covariance_dimension: int
    coordinate_frame: str
    physical_unit: str
    query_type: str
    horizon_semantics: str

    def __post_init__(self) -> None:
        dimension = genuine_integer(
            self.covariance_dimension,
            name="covariance_dimension",
            minimum=1,
        )
        object.__setattr__(self, "covariance_dimension", dimension)
        for name in (
            "coordinate_frame",
            "physical_unit",
            "query_type",
            "horizon_semantics",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))

    @property
    def semantics_id(self) -> str:
        return content_id(self.descriptor())

    def descriptor(self) -> dict[str, object]:
        return {
            "covariance_dimension": self.covariance_dimension,
            "coordinate_frame": self.coordinate_frame,
            "physical_unit": self.physical_unit,
            "query_type": self.query_type,
            "horizon_semantics": self.horizon_semantics,
        }


@dataclass(frozen=True, slots=True)
class DomainCovarianceCalibrationPolicyV2:
    """Frozen finite-group and evidence-admission policy."""

    allow_covariance_shrinkage: bool = False
    minimum_group_count: int = 6
    minimum_group_win_fraction: float = 0.75
    minimum_mean_loo_nll_improvement: float = 0.01
    maximum_single_group_loo_nll_regression: float = 0.0
    require_claim_authorized_decision: bool = False
    minimum_evidence_level: int = 2
    numerical_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        for name in (
            "allow_covariance_shrinkage",
            "require_claim_authorized_decision",
        ):
            value = genuine_boolean(getattr(self, name), name=name)
            object.__setattr__(self, name, value)
        count = genuine_integer(
            self.minimum_group_count,
            name="minimum_group_count",
            minimum=2,
        )
        level = genuine_integer(
            self.minimum_evidence_level,
            name="minimum_evidence_level",
            minimum=1,
        )
        if level > 3:
            raise ValueError("minimum_evidence_level must be at most 3")
        object.__setattr__(self, "minimum_group_count", count)
        object.__setattr__(self, "minimum_evidence_level", level)
        fraction = _number(
            self.minimum_group_win_fraction,
            name="minimum_group_win_fraction",
            maximum=1.0,
        )
        object.__setattr__(self, "minimum_group_win_fraction", fraction)
        for name in (
            "minimum_mean_loo_nll_improvement",
            "maximum_single_group_loo_nll_regression",
            "numerical_tolerance",
        ):
            object.__setattr__(
                self,
                name,
                _number(getattr(self, name), name=name),
            )

    def descriptor(self) -> dict[str, object]:
        return {
            "allow_covariance_shrinkage": self.allow_covariance_shrinkage,
            "minimum_group_count": self.minimum_group_count,
            "minimum_group_win_fraction": self.minimum_group_win_fraction,
            "minimum_mean_loo_nll_improvement": (self.minimum_mean_loo_nll_improvement),
            "maximum_single_group_loo_nll_regression": (
                self.maximum_single_group_loo_nll_regression
            ),
            "require_claim_authorized_decision": (
                self.require_claim_authorized_decision
            ),
            "minimum_evidence_level": self.minimum_evidence_level,
            "numerical_tolerance": self.numerical_tolerance,
        }


def _win_fraction(
    decision: DomainCovarianceCalibrationDecisionV1,
    tolerance: float,
) -> float:
    held = cast(
        Sequence[tuple[str, float, float, float, float]],
        decision.leave_one_group_out,
    )
    return sum(row[3] - row[4] > tolerance for row in held) / len(held)


def _reasons(
    decision: DomainCovarianceCalibrationDecisionV1,
    source: DomainCovarianceCalibrationCertificateV1,
    policy: DomainCovarianceCalibrationPolicyV2,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not decision.calibration_supported:
        reasons.append("source-calibration-rejected")
    if len(decision.group_ids) < policy.minimum_group_count:
        reasons.append("insufficient-independent-groups")
    if (
        decision.mean_loo_nll_improvement + policy.numerical_tolerance
        < policy.minimum_mean_loo_nll_improvement
    ):
        reasons.append("mean-loo-nll-improvement-below-practical-margin")
    if (
        decision.worst_loo_nll_regression
        > policy.maximum_single_group_loo_nll_regression + policy.numerical_tolerance
    ):
        reasons.append("single-group-loo-nll-regression-exceeds-policy")
    if (
        _win_fraction(decision, policy.numerical_tolerance) + policy.numerical_tolerance
        < policy.minimum_group_win_fraction
    ):
        reasons.append("group-win-fraction-below-threshold")
    if not policy.allow_covariance_shrinkage and any(
        scale < 1.0 for scale in source.config.covariance_scales
    ):
        reasons.append("covariance-shrinkage-grid-disallowed")
    return tuple(reasons or ["authorization-criteria-passed"])


@dataclass(frozen=True, slots=True)
class DomainCovarianceCalibrationCertificateV2:
    """A version-1 fit bound to semantics and claim-bearing policy."""

    source_certificate: DomainCovarianceCalibrationCertificateV1
    semantics: CovarianceSemanticsV2
    policy: DomainCovarianceCalibrationPolicyV2
    admission_claim_id: str
    admission_protocol_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(
            self.source_certificate,
            DomainCovarianceCalibrationCertificateV1,
        ):
            raise TypeError("source_certificate must be a version-1 certificate")
        if not isinstance(self.semantics, CovarianceSemanticsV2):
            raise TypeError("semantics must be CovarianceSemanticsV2")
        if not isinstance(self.policy, DomainCovarianceCalibrationPolicyV2):
            raise TypeError("policy must be DomainCovarianceCalibrationPolicyV2")
        object.__setattr__(
            self,
            "admission_claim_id",
            _text(self.admission_claim_id, name="admission_claim_id"),
        )
        object.__setattr__(
            self,
            "admission_protocol_id",
            _text(self.admission_protocol_id, name="admission_protocol_id"),
        )
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="domain covariance v2 metadata",
        )
        object.__setattr__(self, "metadata", metadata)
        expected = content_id(self.descriptor())
        if self.artifact_id is not None:
            supplied = sha256_digest(self.artifact_id, name="artifact_id")
            if supplied != expected:
                raise ValueError("artifact_id does not match v2 certificate")
        object.__setattr__(self, "artifact_id", expected)

    def decision_for_domain(
        self,
        domain_id: str,
    ) -> DomainCovarianceCalibrationDecisionV1 | None:
        return self.source_certificate.decision_for_domain(domain_id)

    def reasons_for_domain(self, domain_id: str) -> tuple[str, ...]:
        decision = self.decision_for_domain(domain_id)
        if decision is None:
            return ("unknown-calibration-domain",)
        return _reasons(decision, self.source_certificate, self.policy)

    def domain_supported(self, domain_id: str) -> bool:
        return self.reasons_for_domain(domain_id) == ("authorization-criteria-passed",)

    @property
    def supported_domains(self) -> tuple[str, ...]:
        if not self.source_certificate.deployment_admissible:
            return ()
        return tuple(
            decision.domain_id
            for decision in self.source_certificate.decisions
            if self.domain_supported(decision.domain_id)
        )

    def descriptor(self) -> dict[str, object]:
        decisions = [
            {
                "domain_id": decision.domain_id,
                "source_decision_id": decision.artifact_id,
                "group_count": len(decision.group_ids),
                "group_win_fraction": _win_fraction(
                    decision,
                    self.policy.numerical_tolerance,
                ),
                "reasons": list(self.reasons_for_domain(decision.domain_id)),
            }
            for decision in self.source_certificate.decisions
        ]
        return {
            "schema": CALIBRATION_V2_SCHEMA,
            "schema_version": CALIBRATION_V2_VERSION,
            "source_certificate_id": self.source_certificate.artifact_id,
            "semantics": {
                **self.semantics.descriptor(),
                "semantics_id": self.semantics.semantics_id,
            },
            "policy": self.policy.descriptor(),
            "admission_claim_id": self.admission_claim_id,
            "admission_protocol_id": self.admission_protocol_id,
            "decisions": decisions,
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}


@dataclass(frozen=True, slots=True)
class DomainCovarianceCalibrationApplicationV2:
    """Content-addressed outcome of one version-2 application."""

    certificate_id: str
    certificate_semantics_id: str
    application_semantics_id: str
    domain_id: str
    evidence_decision_id: str
    evidence_admissible: bool
    applied: bool
    reason: str
    source_application_id: str | None
    raw_numeric_sha256: str
    output_numeric_sha256: str
    raw_array_sha256: str
    output_array_sha256: str
    exact_fallback: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "certificate_id",
            "certificate_semantics_id",
            "application_semantics_id",
            "evidence_decision_id",
            "raw_numeric_sha256",
            "output_numeric_sha256",
            "raw_array_sha256",
            "output_array_sha256",
        ):
            digest = sha256_digest(getattr(self, name), name=name)
            object.__setattr__(self, name, digest)
        object.__setattr__(self, "domain_id", _text(self.domain_id, name="domain_id"))
        if self.source_application_id is not None:
            source_id = sha256_digest(
                self.source_application_id,
                name="source_application_id",
            )
            object.__setattr__(self, "source_application_id", source_id)
        for name in ("evidence_admissible", "applied", "exact_fallback"):
            value = genuine_boolean(getattr(self, name), name=name)
            object.__setattr__(self, name, value)
        if self.applied == self.exact_fallback:
            raise ValueError("applied and exact_fallback must be logical opposites")
        object.__setattr__(self, "reason", _text(self.reason, name="reason"))
        if self.applied and not self.evidence_admissible:
            raise ValueError("applied calibration requires admissible evidence")
        if self.applied and self.source_application_id is None:
            raise ValueError("applied calibration requires a source application")
        if self.applied and self.reason != "calibration-domain-authorized":
            raise ValueError("applied calibration has an invalid reason")
        if self.exact_fallback and (
            self.raw_numeric_sha256 != self.output_numeric_sha256
            or self.raw_array_sha256 != self.output_array_sha256
        ):
            raise ValueError("exact fallback must preserve covariance identity")
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="domain covariance v2 application metadata",
        )
        object.__setattr__(self, "metadata", metadata)
        expected = content_id(self.descriptor())
        if self.artifact_id is not None:
            supplied = sha256_digest(self.artifact_id, name="artifact_id")
            if supplied != expected:
                raise ValueError("artifact_id does not match v2 application")
        object.__setattr__(self, "artifact_id", expected)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": APPLICATION_V2_SCHEMA,
            "schema_version": APPLICATION_V2_VERSION,
            "certificate_id": self.certificate_id,
            "certificate_semantics_id": self.certificate_semantics_id,
            "application_semantics_id": self.application_semantics_id,
            "domain_id": self.domain_id,
            "evidence_decision_id": self.evidence_decision_id,
            "evidence_admissible": self.evidence_admissible,
            "applied": self.applied,
            "reason": self.reason,
            "source_application_id": self.source_application_id,
            "raw_numeric_sha256": self.raw_numeric_sha256,
            "output_numeric_sha256": self.output_numeric_sha256,
            "raw_array_sha256": self.raw_array_sha256,
            "output_array_sha256": self.output_array_sha256,
            "exact_fallback": self.exact_fallback,
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}


def _default_source_config(
    policy: DomainCovarianceCalibrationPolicyV2,
) -> DomainCovarianceCalibrationConfigV1:
    return DomainCovarianceCalibrationConfigV1(
        covariance_scales=(1.0, 2.0, 4.0, 8.0, 16.0),
        isotropic_variances=(0.0, 1e-8, 1e-6, 1e-4),
        minimum_group_count=policy.minimum_group_count,
        minimum_mean_loo_nll_improvement=(policy.minimum_mean_loo_nll_improvement),
        maximum_single_group_loo_nll_regression=(
            policy.maximum_single_group_loo_nll_regression
        ),
        numerical_tolerance=policy.numerical_tolerance,
    )


def fit_domain_covariance_calibration_v2(
    *,
    predictor_id: str,
    calibration_partition_id: str,
    statistical_unit: str,
    residual_semantics: str,
    covariance_semantics: str,
    coordinate_frame: str,
    physical_unit: str,
    query_type: str,
    horizon_semantics: str,
    admission_claim_id: str,
    admission_protocol_id: str,
    event_ids: Sequence[str],
    group_ids: Sequence[str],
    domain_ids: Sequence[str],
    residuals: object,
    covariances: object,
    domain_guard: CalibrationDomainGuardCertificateV1,
    predictor_frozen_before_calibration_outcomes: bool,
    transform_grid_frozen_before_calibration_outcomes: bool,
    application_outcomes_used_for_calibration_selection: bool,
    calibration_groups_independent: bool,
    source_config: DomainCovarianceCalibrationConfigV1 | None = None,
    policy: DomainCovarianceCalibrationPolicyV2 | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DomainCovarianceCalibrationCertificateV2:
    """Fit version 1, then bind the claim-bearing version-2 policy."""

    settings = DomainCovarianceCalibrationPolicyV2() if policy is None else policy
    if not isinstance(settings, DomainCovarianceCalibrationPolicyV2):
        raise TypeError("policy must be DomainCovarianceCalibrationPolicyV2")
    array = np.asarray(covariances)
    if array.ndim != 3:
        raise ValueError("calibration covariances must have shape (events, d, d)")
    if array.shape[1] < 1 or array.shape[1] != array.shape[2]:
        raise ValueError("calibration covariance matrices must be nonempty and square")
    semantics = CovarianceSemanticsV2(
        covariance_dimension=int(array.shape[1]),
        coordinate_frame=coordinate_frame,
        physical_unit=physical_unit,
        query_type=query_type,
        horizon_semantics=horizon_semantics,
    )
    config = (
        _default_source_config(settings) if source_config is None else source_config
    )
    source = fit_domain_covariance_calibration(
        predictor_id=predictor_id,
        calibration_partition_id=calibration_partition_id,
        statistical_unit=statistical_unit,
        residual_semantics=residual_semantics,
        covariance_semantics=covariance_semantics,
        event_ids=event_ids,
        group_ids=group_ids,
        domain_ids=domain_ids,
        residuals=residuals,
        covariances=covariances,
        domain_guard=domain_guard,
        predictor_frozen_before_calibration_outcomes=(
            predictor_frozen_before_calibration_outcomes
        ),
        transform_grid_frozen_before_calibration_outcomes=(
            transform_grid_frozen_before_calibration_outcomes
        ),
        application_outcomes_used_for_calibration_selection=(
            application_outcomes_used_for_calibration_selection
        ),
        calibration_groups_independent=calibration_groups_independent,
        config=config,
        metadata=metadata,
    )
    return DomainCovarianceCalibrationCertificateV2(
        source_certificate=source,
        semantics=semantics,
        policy=settings,
        admission_claim_id=admission_claim_id,
        admission_protocol_id=admission_protocol_id,
        metadata={} if metadata is None else metadata,
    )


def _evidence_admissible(
    decision: EvidenceDecisionV1,
    certificate: DomainCovarianceCalibrationCertificateV2,
) -> bool:
    if decision.status != "pass" or decision.run_classification != "confirmatory":
        return False
    if decision.claim_id != certificate.admission_claim_id:
        return False
    if decision.protocol_id != certificate.admission_protocol_id:
        return False
    if decision.evidence_level < certificate.policy.minimum_evidence_level:
        return False
    return not (
        certificate.policy.require_claim_authorized_decision
        and not decision.claim_authorized
    )


def apply_domain_covariance_calibration_v2(
    raw_covariance: np.ndarray,
    certificate: DomainCovarianceCalibrationCertificateV2,
    *,
    domain_id: str,
    application_semantics: CovarianceSemanticsV2,
    evidence_decision: EvidenceDecisionV1,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, DomainCovarianceCalibrationApplicationV2]:
    """Apply only with matching semantics, policy, and evidence admission."""

    if not isinstance(certificate, DomainCovarianceCalibrationCertificateV2):
        raise TypeError("certificate must be a version-2 certificate")
    if not isinstance(application_semantics, CovarianceSemanticsV2):
        raise TypeError("application_semantics must be CovarianceSemanticsV2")
    if not isinstance(evidence_decision, EvidenceDecisionV1):
        raise TypeError("evidence_decision must be EvidenceDecisionV1")
    domain = _text(domain_id, name="domain_id")
    dimension = _covariance_dimension(raw_covariance)
    validation_output, _ = apply_domain_covariance_calibration(
        raw_covariance,
        certificate.source_certificate,
        domain_id=domain,
        inference_admissible=False,
    )
    assert validation_output is raw_covariance
    raw_numeric_digest = _numeric_array_digest(raw_covariance)
    raw_array_digest = _exact_array_digest(raw_covariance)
    evidence_ok = _evidence_admissible(evidence_decision, certificate)
    semantics_ok = bool(
        application_semantics.semantics_id == certificate.semantics.semantics_id
        and dimension == application_semantics.covariance_dimension
    )
    source_record: DomainCovarianceCalibrationApplicationV1 | None = None
    if not semantics_ok:
        output, reason = raw_covariance, "covariance-semantics-mismatch"
    elif not evidence_ok:
        output, reason = raw_covariance, "evidence-decision-rejected"
    elif certificate.decision_for_domain(domain) is None:
        output, reason = raw_covariance, "unknown-calibration-domain"
    elif not certificate.source_certificate.deployment_admissible:
        output = raw_covariance
        reason = "calibration-information-boundary-rejected"
    elif not certificate.domain_supported(domain):
        output, reason = raw_covariance, "calibration-policy-rejected"
    else:
        output, source_record = apply_domain_covariance_calibration(
            raw_covariance,
            certificate.source_certificate,
            domain_id=domain,
            inference_admissible=True,
            metadata={
                "v2_certificate_id": certificate.artifact_id,
                "evidence_decision_id": evidence_decision.decision_id,
            },
        )
        reason = source_record.reason
    output_numeric_digest = _numeric_array_digest(output)
    output_array_digest = _exact_array_digest(output)
    applied = bool(source_record is not None and source_record.applied)
    record = DomainCovarianceCalibrationApplicationV2(
        certificate_id=str(certificate.artifact_id),
        certificate_semantics_id=certificate.semantics.semantics_id,
        application_semantics_id=application_semantics.semantics_id,
        domain_id=domain,
        evidence_decision_id=evidence_decision.decision_id,
        evidence_admissible=evidence_ok,
        applied=applied,
        reason=reason,
        source_application_id=(
            None if source_record is None else source_record.artifact_id
        ),
        raw_numeric_sha256=raw_numeric_digest,
        output_numeric_sha256=output_numeric_digest,
        raw_array_sha256=raw_array_digest,
        output_array_sha256=output_array_digest,
        exact_fallback=not applied,
        metadata={} if metadata is None else metadata,
    )
    return output, record


__all__ = [
    "APPLICATION_V2_SCHEMA",
    "APPLICATION_V2_VERSION",
    "CALIBRATION_V2_SCHEMA",
    "CALIBRATION_V2_VERSION",
    "CovarianceSemanticsV2",
    "DomainCovarianceCalibrationApplicationV2",
    "DomainCovarianceCalibrationCertificateV2",
    "DomainCovarianceCalibrationPolicyV2",
    "apply_domain_covariance_calibration_v2",
    "fit_domain_covariance_calibration_v2",
]
