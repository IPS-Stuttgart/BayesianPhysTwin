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
    DomainCovarianceCalibrationPolicyV2,
    apply_domain_covariance_calibration_v2,
    fit_domain_covariance_calibration_v2,
)
from bayesian_phystwin.evidence_decision_v1 import (
    DecisionMetricV1,
    EvidenceDecisionV1,
)
from bayesian_phystwin.repository_provenance import RepositoryState

PARTITION_ID = "a" * 64
PREDICTOR_ID = "b" * 64
CLAIM_ID = "claim/domain-covariance-calibration-v2"
PROTOCOL_ID = "protocol/domain-covariance-calibration-v2"


def _guard():
    groups = tuple(f"dynamic-{index}" for index in range(6)) + tuple(
        f"static-{index}" for index in range(6)
    )
    domains = ("dynamic",) * 6 + ("quasi-static",) * 6
    fallback = np.ones(12, dtype=np.float64)
    candidate = np.asarray([0.9] * 6 + [1.1] * 6, dtype=np.float64)
    return fit_calibration_domain_guard(
        calibration_partition_id=PARTITION_ID,
        statistical_unit="independent-physical-session",
        metric="point-loss-m",
        group_ids=groups,
        domain_ids=domains,
        candidate_losses=candidate,
        fallback_losses=fallback,
        guard_frozen_before_application_outcomes=True,
        application_outcomes_used_for_guard_selection=False,
        calibration_groups_independent=True,
    )


def _fit(
    *,
    dynamic_residuals: tuple[float, ...] = (2.0,) * 6,
    source_config: DomainCovarianceCalibrationConfigV1 | None = None,
    policy: DomainCovarianceCalibrationPolicyV2 | None = None,
    predictor_frozen: bool = True,
):
    event_ids: list[str] = []
    group_ids: list[str] = []
    domain_ids: list[str] = []
    residuals: list[list[float]] = []
    covariances: list[list[list[float]]] = []
    for domain, prefix, values in (
        ("dynamic", "dynamic", dynamic_residuals),
        ("quasi-static", "static", (1.0,) * 6),
    ):
        for index, value in enumerate(values):
            event_ids.append(f"{prefix}-{index}-event")
            group_ids.append(f"{prefix}-{index}")
            domain_ids.append(domain)
            residuals.append([value])
            covariances.append([[1.0]])
    return fit_domain_covariance_calibration_v2(
        predictor_id=PREDICTOR_ID,
        calibration_partition_id=PARTITION_ID,
        statistical_unit="independent-physical-session",
        residual_semantics="prediction-error-m",
        covariance_semantics="raw-predictive-covariance-m2",
        coordinate_frame="phystwin-world",
        physical_unit="m2",
        query_type="endpoint-position",
        horizon_semantics="fixed-endpoint-frame",
        admission_claim_id=CLAIM_ID,
        admission_protocol_id=PROTOCOL_ID,
        event_ids=event_ids,
        group_ids=group_ids,
        domain_ids=domain_ids,
        residuals=np.asarray(residuals, dtype=np.float64),
        covariances=np.asarray(covariances, dtype=np.float64),
        domain_guard=_guard(),
        predictor_frozen_before_calibration_outcomes=predictor_frozen,
        transform_grid_frozen_before_calibration_outcomes=True,
        application_outcomes_used_for_calibration_selection=False,
        calibration_groups_independent=True,
        source_config=source_config,
        policy=policy,
    )


def _evidence_decision(
    *,
    status: str = "pass",
    claim_id: str = CLAIM_ID,
    protocol_id: str = PROTOCOL_ID,
    claim_authorized: bool = False,
    evidence_level: int = 3,
    run_classification: str = "confirmatory",
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
            comparison="candidate-versus-raw",
            rule="registered-thresholds-pass",
            observed_value=1.0,
            threshold_value=0.0,
            unit="nats-per-group",
        ),
        run_manifest_id="c" * 64,
        evidence_fingerprint="d" * 64,
        evidence_summary_sha256="e" * 64,
        repositories=(
            RepositoryState(
                repository="IPS-Stuttgart/BayesianPhysTwin",
                revision="f" * 40,
                dirty=False,
                role="primary",
            ),
        ),
        created_utc="2026-08-11T00:00:00+00:00",
    )


def _permissive_policy(**overrides: object) -> DomainCovarianceCalibrationPolicyV2:
    values: dict[str, object] = {
        "minimum_group_count": 6,
        "minimum_group_win_fraction": 0.0,
        "minimum_mean_loo_nll_improvement": 0.0,
        "maximum_single_group_loo_nll_regression": 100.0,
        "require_claim_authorized_decision": False,
    }
    values.update(overrides)
    return DomainCovarianceCalibrationPolicyV2(**values)  # type: ignore[arg-type]


def test_default_policy_is_conservative_and_applies_bound_decision() -> None:
    certificate = _fit()
    raw = np.asarray([[1.0]])

    output, record = apply_domain_covariance_calibration_v2(
        raw,
        certificate,
        domain_id="dynamic",
        evidence_decision=_evidence_decision(),
    )

    assert min(certificate.source_certificate.config.covariance_scales) == 1.0
    assert certificate.supported_domains == ("dynamic",)
    np.testing.assert_allclose(output, [[4.0]])
    assert output is not raw
    assert record.applied
    assert not record.exact_fallback
    assert record.evidence_admissible
    assert record.source_application_id is not None
    assert record.reason == "calibration-domain-authorized"


def test_covariance_semantics_mismatch_returns_exact_input_object() -> None:
    certificate = _fit()
    raw = np.eye(2)

    output, record = apply_domain_covariance_calibration_v2(
        raw,
        certificate,
        domain_id="dynamic",
        evidence_decision=_evidence_decision(),
    )

    assert output is raw
    assert record.exact_fallback
    assert record.reason == "covariance-semantics-mismatch"
    assert record.raw_covariance_sha256 == record.output_covariance_sha256


@pytest.mark.parametrize(
    "decision",
    [
        _evidence_decision(status="fail"),
        _evidence_decision(run_classification="diagnostic"),
        _evidence_decision(claim_id="other-claim"),
        _evidence_decision(protocol_id="other-protocol"),
        _evidence_decision(evidence_level=1),
    ],
)
def test_unmatched_evidence_decision_forces_exact_fallback(
    decision: EvidenceDecisionV1,
) -> None:
    raw = np.asarray([[1.0]])

    output, record = apply_domain_covariance_calibration_v2(
        raw,
        _fit(),
        domain_id="dynamic",
        evidence_decision=decision,
    )

    assert output is raw
    assert not record.evidence_admissible
    assert record.reason == "evidence-decision-rejected"


def test_policy_can_require_claim_authorization() -> None:
    certificate = _fit(
        policy=_permissive_policy(require_claim_authorized_decision=True)
    )
    raw = np.asarray([[1.0]])

    output, record = apply_domain_covariance_calibration_v2(
        raw,
        certificate,
        domain_id="dynamic",
        evidence_decision=_evidence_decision(claim_authorized=False),
    )

    assert output is raw
    assert not record.evidence_admissible
    assert record.reason == "evidence-decision-rejected"


def test_shrinkage_grid_requires_explicit_policy_opt_in() -> None:
    source_config = DomainCovarianceCalibrationConfigV1(
        covariance_scales=(0.5, 1.0, 4.0),
        isotropic_variances=(0.0,),
        minimum_group_count=6,
    )
    rejected = _fit(
        source_config=source_config,
        policy=_permissive_policy(),
    )
    accepted = _fit(
        source_config=source_config,
        policy=_permissive_policy(allow_covariance_shrinkage=True),
    )

    assert "covariance-shrinkage-grid-disallowed" in rejected.reasons_for_domain(
        "dynamic"
    )
    assert not rejected.domain_supported("dynamic")
    assert accepted.domain_supported("dynamic")


def test_group_win_fraction_is_a_separate_authorization_gate() -> None:
    source_config = DomainCovarianceCalibrationConfigV1(
        covariance_scales=(1.0, 4.0),
        isotropic_variances=(0.0,),
        minimum_group_count=6,
        maximum_single_group_loo_nll_regression=100.0,
    )
    certificate = _fit(
        dynamic_residuals=(2.0, 2.0, 2.0, 2.0, 2.0, 0.1),
        source_config=source_config,
        policy=_permissive_policy(minimum_group_win_fraction=1.0),
    )

    assert certificate.decision_for_domain("dynamic") is not None
    assert "group-win-fraction-below-threshold" in certificate.reasons_for_domain(
        "dynamic"
    )
    assert not certificate.domain_supported("dynamic")


def test_unknown_domain_and_nonprospective_fit_both_fail_closed() -> None:
    raw = np.asarray([[1.0]])
    evidence = _evidence_decision()

    unknown_output, unknown_record = apply_domain_covariance_calibration_v2(
        raw,
        _fit(),
        domain_id="unseen",
        evidence_decision=evidence,
    )
    boundary_output, boundary_record = apply_domain_covariance_calibration_v2(
        raw,
        _fit(predictor_frozen=False),
        domain_id="dynamic",
        evidence_decision=evidence,
    )

    assert unknown_output is raw
    assert unknown_record.reason == "unknown-calibration-domain"
    assert boundary_output is raw
    assert boundary_record.reason == "calibration-information-boundary-rejected"


def test_identity_transform_is_retained_as_exact_fallback() -> None:
    certificate = _fit(
        dynamic_residuals=(1.0,) * 6,
        source_config=DomainCovarianceCalibrationConfigV1(
            covariance_scales=(1.0, 4.0),
            isotropic_variances=(0.0,),
            minimum_group_count=6,
        ),
        policy=_permissive_policy(),
    )
    raw = np.asarray([[1.0]])

    output, record = apply_domain_covariance_calibration_v2(
        raw,
        certificate,
        domain_id="dynamic",
        evidence_decision=_evidence_decision(),
    )

    assert output is raw
    assert record.exact_fallback
    assert record.reason == "calibration-identity-transform-retained"


@pytest.mark.parametrize(
    "factory",
    [
        lambda: CovarianceSemanticsV2(0, "world", "m2", "position", "fixed"),
        lambda: DomainCovarianceCalibrationPolicyV2(
            minimum_group_win_fraction=1.1
        ),
        lambda: DomainCovarianceCalibrationPolicyV2(minimum_evidence_level=4),
        lambda: DomainCovarianceCalibrationPolicyV2(
            allow_covariance_shrinkage="false"  # type: ignore[arg-type]
        ),
    ],
)
def test_v2_contracts_reject_ambiguous_configuration(factory) -> None:
    with pytest.raises(ValueError):
        factory()


def test_content_identity_rejects_certificate_and_application_tampering() -> None:
    certificate = _fit()
    _, application = apply_domain_covariance_calibration_v2(
        np.asarray([[1.0]]),
        certificate,
        domain_id="dynamic",
        evidence_decision=_evidence_decision(),
    )

    with pytest.raises(ValueError, match="artifact_id"):
        replace(certificate, artifact_id="0" * 64)
    with pytest.raises(ValueError, match="artifact_id"):
        replace(application, artifact_id="0" * 64)


def test_apply_rejects_malformed_types_before_admission() -> None:
    certificate = _fit()
    decision = _evidence_decision()

    with pytest.raises(TypeError, match="numpy.ndarray"):
        apply_domain_covariance_calibration_v2(  # type: ignore[arg-type]
            [[1.0]],
            certificate,
            domain_id="dynamic",
            evidence_decision=decision,
        )
    with pytest.raises(TypeError, match="EvidenceDecisionV1"):
        apply_domain_covariance_calibration_v2(
            np.asarray([[1.0]]),
            certificate,
            domain_id="dynamic",
            evidence_decision=True,  # type: ignore[arg-type]
        )
