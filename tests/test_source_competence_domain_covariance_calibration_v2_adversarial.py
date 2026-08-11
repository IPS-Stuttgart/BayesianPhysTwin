from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin.calibration_domain_guard import (
    fit_calibration_domain_guard,
)
from bayesian_phystwin.domain_covariance_calibration import (
    DomainCovarianceCalibrationConfigV1,
)
from bayesian_phystwin.domain_covariance_calibration_v2 import (
    CovarianceSemanticsV2,
    DomainCovarianceCalibrationApplicationV2,
    DomainCovarianceCalibrationCertificateV2,
    DomainCovarianceCalibrationPolicyV2,
    apply_domain_covariance_calibration_v2,
    fit_domain_covariance_calibration_v2,
)
from bayesian_phystwin.evidence_decision_v1 import (
    DecisionMetricV1,
    EvidenceDecisionV1,
)
from bayesian_phystwin.repository_provenance import RepositoryState

PARTITION_ID = "1" * 64
PREDICTOR_ID = "2" * 64
CLAIM_ID = "claim/domain-covariance-v2-adversarial"
PROTOCOL_ID = "protocol/domain-covariance-v2-adversarial"


def _guard():
    group_ids = tuple(f"group-{index}" for index in range(6))
    return fit_calibration_domain_guard(
        calibration_partition_id=PARTITION_ID,
        statistical_unit="independent-session",
        metric="endpoint-error-m",
        group_ids=group_ids,
        domain_ids=("dynamic",) * 6,
        candidate_losses=np.full(6, 0.9),
        fallback_losses=np.ones(6),
        guard_frozen_before_application_outcomes=True,
        application_outcomes_used_for_guard_selection=False,
        calibration_groups_independent=True,
    )


def _fit_kwargs() -> dict[str, object]:
    return {
        "predictor_id": PREDICTOR_ID,
        "calibration_partition_id": PARTITION_ID,
        "statistical_unit": "independent-session",
        "residual_semantics": "endpoint-position-error-m",
        "covariance_semantics": "endpoint-position-covariance-m2",
        "coordinate_frame": "phystwin-world",
        "physical_unit": "m2",
        "query_type": "endpoint-position",
        "horizon_semantics": "fixed-endpoint-frame",
        "admission_claim_id": CLAIM_ID,
        "admission_protocol_id": PROTOCOL_ID,
        "event_ids": tuple(f"event-{index}" for index in range(6)),
        "group_ids": tuple(f"group-{index}" for index in range(6)),
        "domain_ids": ("dynamic",) * 6,
        "residuals": np.full((6, 1), 2.0),
        "covariances": np.ones((6, 1, 1)),
        "domain_guard": _guard(),
        "predictor_frozen_before_calibration_outcomes": True,
        "transform_grid_frozen_before_calibration_outcomes": True,
        "application_outcomes_used_for_calibration_selection": False,
        "calibration_groups_independent": True,
        "source_config": DomainCovarianceCalibrationConfigV1(
            covariance_scales=(1.0, 4.0),
            isotropic_variances=(0.0,),
            minimum_group_count=6,
        ),
        "policy": DomainCovarianceCalibrationPolicyV2(
            minimum_group_count=6,
            minimum_group_win_fraction=0.0,
            minimum_mean_loo_nll_improvement=0.0,
            maximum_single_group_loo_nll_regression=100.0,
        ),
    }


def _certificate() -> DomainCovarianceCalibrationCertificateV2:
    return fit_domain_covariance_calibration_v2(**_fit_kwargs())  # type: ignore[arg-type]


def _decision(
    *,
    status: str = "pass",
    run_classification: str = "confirmatory",
    claim_id: str = CLAIM_ID,
    protocol_id: str = PROTOCOL_ID,
    evidence_level: int = 3,
    claim_authorized: bool = True,
) -> EvidenceDecisionV1:
    return EvidenceDecisionV1(
        claim_id=claim_id,
        protocol_id=protocol_id,
        status=status,  # type: ignore[arg-type]
        run_classification=run_classification,  # type: ignore[arg-type]
        claim_authorized=claim_authorized,
        evidence_level=evidence_level,
        metric=DecisionMetricV1(
            name="held-group-log-score",
            comparison="calibrated-versus-raw",
            rule="registered-policy-pass",
            observed_value=1.0,
            threshold_value=0.0,
            unit="nats-per-group",
        ),
        run_manifest_id="3" * 64,
        evidence_fingerprint="4" * 64,
        evidence_summary_sha256="5" * 64,
        repositories=(
            RepositoryState(
                repository="IPS-Stuttgart/BayesianPhysTwin",
                revision="6" * 40,
                dirty=False,
                role="primary",
            ),
        ),
        created_utc="2026-08-11T00:00:00+00:00",
    )


def _apply(
    certificate: DomainCovarianceCalibrationCertificateV2,
    *,
    raw: np.ndarray | None = None,
    decision: EvidenceDecisionV1 | None = None,
):
    return apply_domain_covariance_calibration_v2(
        np.asarray([[1.0]]) if raw is None else raw,
        certificate,
        domain_id="dynamic",
        application_semantics=certificate.semantics,
        evidence_decision=_decision() if decision is None else decision,
        metadata={"test": "adversarial"},
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("coordinate_frame", ""),
        ("coordinate_frame", " world"),
        ("physical_unit", 1),
        ("query_type", "position "),
        ("horizon_semantics", None),
    ],
)
def test_semantics_reject_noncanonical_strings(field: str, value: object) -> None:
    values: dict[str, object] = {
        "covariance_dimension": 1,
        "coordinate_frame": "world",
        "physical_unit": "m2",
        "query_type": "position",
        "horizon_semantics": "fixed",
    }
    values[field] = value
    with pytest.raises(ValueError, match="canonical string"):
        CovarianceSemanticsV2(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("minimum_group_win_fraction", True),
        ("minimum_group_win_fraction", [0.5]),
        ("minimum_group_win_fraction", "0.5"),
        ("minimum_group_win_fraction", 1.1),
        ("minimum_mean_loo_nll_improvement", -0.1),
        ("maximum_single_group_loo_nll_regression", float("nan")),
        ("numerical_tolerance", float("inf")),
    ],
)
def test_policy_rejects_ambiguous_numbers(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        DomainCovarianceCalibrationPolicyV2(**{field: value})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "raw",
    [
        np.asarray([[1.0 + 0.0j]]),
        np.asarray([1.0]),
        np.ones((2, 3)),
        np.empty((0, 0)),
        np.asarray([[float("nan")]]),
    ],
)
def test_apply_rejects_malformed_covariance_arrays(raw: np.ndarray) -> None:
    with pytest.raises(ValueError):
        _apply(_certificate(), raw=raw)


def test_certificate_rejects_wrong_component_types_and_identifiers() -> None:
    certificate = _certificate()
    with pytest.raises(TypeError, match="source_certificate"):
        replace(certificate, source_certificate=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="semantics"):
        replace(certificate, semantics=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="policy"):
        replace(certificate, policy=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="admission_claim_id"):
        replace(certificate, admission_claim_id="")
    with pytest.raises(ValueError, match="admission_protocol_id"):
        replace(certificate, admission_protocol_id=" protocol")


def test_policy_can_reject_group_count_and_worst_group_harm() -> None:
    certificate = _certificate()
    strict_count = replace(
        certificate,
        policy=replace(certificate.policy, minimum_group_count=7),
        artifact_id=None,
    )
    strict_harm = replace(
        certificate,
        policy=replace(
            certificate.policy,
            maximum_single_group_loo_nll_regression=0.0,
        ),
        artifact_id=None,
    )

    assert "insufficient-independent-groups" in strict_count.reasons_for_domain(
        "dynamic"
    )
    assert "single-group-loo-nll-regression-exceeds-policy" in (
        strict_harm.reasons_for_domain("dynamic")
    )
    assert strict_count.to_record()["artifact_id"] == strict_count.artifact_id


def test_application_record_rejects_inconsistent_states() -> None:
    certificate = _certificate()
    _, applied = _apply(certificate)
    rejected_output, rejected = _apply(
        certificate,
        decision=_decision(status="fail"),
    )
    assert rejected_output.shape == (1, 1)

    with pytest.raises(ValueError, match="logical opposites"):
        replace(applied, exact_fallback=True, artifact_id=None)
    with pytest.raises(ValueError, match="admissible evidence"):
        replace(applied, evidence_admissible=False, artifact_id=None)
    with pytest.raises(ValueError, match="source application"):
        replace(applied, source_application_id=None, artifact_id=None)
    with pytest.raises(ValueError, match="invalid reason"):
        replace(applied, reason="other", artifact_id=None)
    with pytest.raises(ValueError, match="preserve covariance identity"):
        replace(
            rejected,
            output_array_sha256="7" * 64,
            artifact_id=None,
        )
    with pytest.raises(ValueError, match="logical opposites"):
        DomainCovarianceCalibrationApplicationV2(
            certificate_id="a" * 64,
            certificate_semantics_id="b" * 64,
            application_semantics_id="b" * 64,
            domain_id="dynamic",
            evidence_decision_id="c" * 64,
            evidence_admissible=False,
            applied=False,
            reason="fallback",
            source_application_id=None,
            raw_numeric_sha256="d" * 64,
            output_numeric_sha256="d" * 64,
            raw_array_sha256="e" * 64,
            output_array_sha256="e" * 64,
            exact_fallback=False,
        )
    assert applied.to_record()["artifact_id"] == applied.artifact_id


def test_fit_rejects_wrong_policy_and_covariance_shapes() -> None:
    wrong_policy = _fit_kwargs()
    wrong_policy["policy"] = True
    with pytest.raises(TypeError, match="policy"):
        fit_domain_covariance_calibration_v2(**wrong_policy)  # type: ignore[arg-type]

    for covariances in (
        np.ones((6, 1)),
        np.ones((6, 0, 0)),
        np.ones((6, 1, 2)),
    ):
        values = _fit_kwargs()
        values["covariances"] = covariances
        with pytest.raises(ValueError, match="covariance"):
            fit_domain_covariance_calibration_v2(**values)  # type: ignore[arg-type]


def test_claim_authorization_requirement_accepts_authorized_decision() -> None:
    values = _fit_kwargs()
    values["policy"] = DomainCovarianceCalibrationPolicyV2(
        minimum_group_count=6,
        minimum_group_win_fraction=0.0,
        minimum_mean_loo_nll_improvement=0.0,
        maximum_single_group_loo_nll_regression=100.0,
        require_claim_authorized_decision=True,
    )
    certificate = fit_domain_covariance_calibration_v2(  # type: ignore[arg-type]
        **values
    )

    output, record = _apply(
        certificate,
        decision=_decision(claim_authorized=True),
    )

    assert output is not None
    assert record.evidence_admissible


def test_apply_rejects_wrong_certificate_and_domain_identifier() -> None:
    certificate = _certificate()
    with pytest.raises(TypeError, match="version-2 certificate"):
        apply_domain_covariance_calibration_v2(  # type: ignore[arg-type]
            np.asarray([[1.0]]),
            True,
            domain_id="dynamic",
            application_semantics=certificate.semantics,
            evidence_decision=_decision(),
        )
    with pytest.raises(ValueError, match="domain_id"):
        apply_domain_covariance_calibration_v2(
            np.asarray([[1.0]]),
            certificate,
            domain_id=" dynamic",
            application_semantics=certificate.semantics,
            evidence_decision=_decision(),
        )
