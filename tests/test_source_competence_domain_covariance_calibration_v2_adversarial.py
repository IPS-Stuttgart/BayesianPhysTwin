from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin.calibration_domain_guard import fit_calibration_domain_guard
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
    groups = tuple(f"group-{index}" for index in range(6))
    return fit_calibration_domain_guard(
        calibration_partition_id=PARTITION_ID,
        statistical_unit="independent-session",
        metric="endpoint-error-m",
        group_ids=groups,
        domain_ids=("dynamic",) * 6,
        candidate_losses=np.full(6, 0.9),
        fallback_losses=np.ones(6),
        guard_frozen_before_application_outcomes=True,
        application_outcomes_used_for_guard_selection=False,
        calibration_groups_independent=True,
    )


def _fit(
    *,
    residuals: np.ndarray | None = None,
    policy: DomainCovarianceCalibrationPolicyV2 | None = None,
    source_config: DomainCovarianceCalibrationConfigV1 | None = None,
) -> DomainCovarianceCalibrationCertificateV2:
    return fit_domain_covariance_calibration_v2(
        predictor_id=PREDICTOR_ID,
        calibration_partition_id=PARTITION_ID,
        statistical_unit="independent-session",
        residual_semantics="endpoint-position-error-m",
        covariance_semantics="endpoint-position-covariance-m2",
        coordinate_frame="phystwin-world",
        physical_unit="m2",
        query_type="endpoint-position",
        horizon_semantics="fixed-endpoint-frame",
        admission_claim_id=CLAIM_ID,
        admission_protocol_id=PROTOCOL_ID,
        event_ids=tuple(f"event-{index}" for index in range(6)),
        group_ids=tuple(f"group-{index}" for index in range(6)),
        domain_ids=("dynamic",) * 6,
        residuals=np.full((6, 1), 2.0) if residuals is None else residuals,
        covariances=np.ones((6, 1, 1)),
        domain_guard=_guard(),
        predictor_frozen_before_calibration_outcomes=True,
        transform_grid_frozen_before_calibration_outcomes=True,
        application_outcomes_used_for_calibration_selection=False,
        calibration_groups_independent=True,
        source_config=(
            DomainCovarianceCalibrationConfigV1(
                covariance_scales=(1.0, 4.0),
                isotropic_variances=(0.0,),
                minimum_group_count=6,
            )
            if source_config is None
            else source_config
        ),
        policy=(
            DomainCovarianceCalibrationPolicyV2(
                minimum_group_count=6,
                minimum_group_win_fraction=0.0,
                minimum_mean_loo_nll_improvement=0.0,
                maximum_single_group_loo_nll_regression=100.0,
            )
            if policy is None
            else policy
        ),
    )


def _decision(
    *,
    status: str = "pass",
    claim_authorized: bool | None = None,
) -> EvidenceDecisionV1:
    authorized = status == "pass" if claim_authorized is None else claim_authorized
    return EvidenceDecisionV1(
        claim_id=CLAIM_ID,
        protocol_id=PROTOCOL_ID,
        status=status,  # type: ignore[arg-type]
        run_classification="confirmatory",
        claim_authorized=authorized,
        evidence_level=3,
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
        _apply(_fit(), raw=raw)


def test_certificate_contracts_and_content_identity_fail_closed() -> None:
    certificate = _fit()
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
    with pytest.raises(ValueError, match="artifact_id"):
        replace(certificate, artifact_id="0" * 64)
    assert certificate.to_record()["artifact_id"] == certificate.artifact_id


def test_policy_rejects_group_count_and_actual_worst_group_harm() -> None:
    certificate = _fit()
    strict_count = replace(
        certificate,
        policy=replace(certificate.policy, minimum_group_count=7),
        artifact_id=None,
    )
    harm_certificate = _fit(
        residuals=np.asarray(
            [[2.0], [2.0], [2.0], [2.0], [2.0], [0.1]],
            dtype=np.float64,
        )
    )
    strict_harm = replace(
        harm_certificate,
        policy=replace(
            harm_certificate.policy,
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


def test_application_record_rejects_inconsistent_states() -> None:
    certificate = _fit()
    _, applied = _apply(certificate)
    _, rejected = _apply(certificate, decision=_decision(status="fail"))

    with pytest.raises(ValueError, match="logical opposites"):
        replace(applied, exact_fallback=True, artifact_id=None)
    with pytest.raises(ValueError, match="admissible evidence"):
        replace(applied, evidence_admissible=False, artifact_id=None)
    with pytest.raises(ValueError, match="source application"):
        replace(applied, source_application_id=None, artifact_id=None)
    with pytest.raises(ValueError, match="invalid reason"):
        replace(applied, reason="other", artifact_id=None)
    with pytest.raises(ValueError, match="preserve covariance identity"):
        replace(rejected, output_array_sha256="7" * 64, artifact_id=None)
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
    with pytest.raises(ValueError, match="artifact_id"):
        replace(applied, artifact_id="0" * 64)
    assert applied.to_record()["artifact_id"] == applied.artifact_id


def test_fit_and_apply_reject_ambiguous_inputs() -> None:
    common = {
        "predictor_id": PREDICTOR_ID,
        "calibration_partition_id": PARTITION_ID,
        "statistical_unit": "independent-session",
        "residual_semantics": "error-m",
        "covariance_semantics": "covariance-m2",
        "coordinate_frame": "world",
        "physical_unit": "m2",
        "query_type": "position",
        "horizon_semantics": "fixed",
        "admission_claim_id": CLAIM_ID,
        "admission_protocol_id": PROTOCOL_ID,
        "event_ids": tuple(f"event-{index}" for index in range(6)),
        "group_ids": tuple(f"group-{index}" for index in range(6)),
        "domain_ids": ("dynamic",) * 6,
        "residuals": np.ones((6, 1)),
        "domain_guard": _guard(),
        "predictor_frozen_before_calibration_outcomes": True,
        "transform_grid_frozen_before_calibration_outcomes": True,
        "application_outcomes_used_for_calibration_selection": False,
        "calibration_groups_independent": True,
    }
    for covariances in (
        np.ones((6, 1)),
        np.ones((6, 0, 0)),
        np.ones((6, 1, 2)),
    ):
        with pytest.raises(ValueError, match="covariance"):
            fit_domain_covariance_calibration_v2(
                **common,
                covariances=covariances,
            )  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="policy"):
        fit_domain_covariance_calibration_v2(
            **common,
            covariances=np.ones((6, 1, 1)),
            policy=True,
        )  # type: ignore[arg-type]

    certificate = _fit()
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


def test_claim_authorization_policy_accepts_authorized_decision() -> None:
    certificate = _fit(
        policy=DomainCovarianceCalibrationPolicyV2(
            minimum_group_count=6,
            minimum_group_win_fraction=0.0,
            minimum_mean_loo_nll_improvement=0.0,
            maximum_single_group_loo_nll_regression=100.0,
            require_claim_authorized_decision=True,
        )
    )

    _, record = _apply(certificate, decision=_decision(claim_authorized=True))

    assert record.evidence_admissible
