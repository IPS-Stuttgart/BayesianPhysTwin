from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest

from bayesian_phystwin.covariance_only_hybrid import (
    compose_covariance_only_hybrid,
)
from bayesian_phystwin.domain_covariance_calibration_v2 import (
    DomainCovarianceCalibrationApplicationV2,
)
from bayesian_phystwin.evidence_decision_v1 import (
    DecisionMetricV1,
    EvidenceDecisionV1,
)
from bayesian_phystwin.guard_harm_risk_artifacts import (
    certify_guard_harm_risk_from_artifacts,
)
from bayesian_phystwin.query_covariance_treatment import (
    EVALUATION_BASELINE_BELIEF_ID_METADATA_KEY,
    EVALUATION_CALIBRATION_APPLICATION_ID_METADATA_KEY,
    EVALUATION_CALIBRATION_PARTITION_ID_METADATA_KEY,
    EVALUATION_CANDIDATE_BELIEF_ID_METADATA_KEY,
    EVALUATION_CANDIDATE_COVARIANCE_ID_METADATA_KEY,
    EVALUATION_COMMON_DOMAIN_ID_METADATA_KEY,
    EVALUATION_HARM_RISK_CERTIFICATE_ID_METADATA_KEY,
    EVALUATION_HYBRID_RECORD_ID_METADATA_KEY,
    EVALUATION_MEAN_FULL_WIDTH_RATIO_METADATA_KEY,
    EVALUATION_MEAN_IDENTITY_VERIFIED_METADATA_KEY,
    EVALUATION_QUERY_ID_METADATA_KEY,
    EVALUATION_QUERY_RELEVANCE_CERTIFICATE_ID_METADATA_KEY,
    EVALUATION_SIMULTANEOUS_COVERAGE_METADATA_KEY,
    EVALUATION_STATISTICAL_UNIT_METADATA_KEY,
    CovarianceOnlyTreatmentPolicyV1,
    QueryCovarianceRelevancePolicyV1,
    certify_query_covariance_relevance,
    decide_covariance_only_treatment,
    select_covariance_only_belief,
)
from bayesian_phystwin.repository_provenance import RepositoryState

QUERY_ID = "1" * 64
PARTITION_ID = "2" * 64
JACOBIAN_ID = "3" * 64
BASELINE_ID = "4" * 64
CANDIDATE_ID = "5" * 64
COMMON_DOMAIN_ID = "6" * 64
EVALUATION_CLAIM_ID = "covariance-only-query-treatment-v1"
EVALUATION_PROTOCOL_ID = "covariance-only-query-treatment-protocol-v1"
PROPER_SCORE_NAME = "query-gaussian-nll-upper-confidence-bound"
PROPER_SCORE_COMPARISON = "candidate-minus-last-residual"
PROPER_SCORE_RULE = "upper-confidence-bound-at-most"
STATISTICAL_UNIT = "physical-object-session"


@dataclass(frozen=True)
class _Belief:
    artifact_id: str


def _hybrid():
    mean = np.zeros((2, 2), dtype=np.float64)
    covariance = np.repeat(np.eye(2, dtype=np.float64)[None], 2, axis=0)
    return compose_covariance_only_hybrid(
        mean,
        covariance,
        reference_predictor_id="last_residual",
        covariance_donor_id="independent_endpoint_v1",
        covariance_scale=np.asarray([8.0, 16.0]),
    )


def _relevance(
    hybrid,
    *,
    covariance_artifact_id: str | None = None,
    material: bool = True,
    target_outcomes_used: bool = False,
):
    if material:
        factor = np.asarray([[1.0], [0.0]], dtype=np.float64)
        jacobian = np.eye(2, dtype=np.float64)
        maximum_null = 1.0
    else:
        factor = np.asarray([[0.0], [1.0]], dtype=np.float64)
        jacobian = np.asarray([[1.0, 0.0]], dtype=np.float64)
        maximum_null = 0.5
    policy = QueryCovarianceRelevancePolicyV1(
        minimum_shared_trace_fraction=0.2,
        minimum_maximum_generalized_eigenvalue=0.5,
        minimum_effective_rank=1,
        maximum_null_mode_fraction=maximum_null,
    )
    return certify_query_covariance_relevance(
        query_id=QUERY_ID,
        covariance_artifact_id=(
            hybrid.record.artifact_id
            if covariance_artifact_id is None
            else covariance_artifact_id
        ),
        jacobian_artifact_id=JACOBIAN_ID,
        calibration_partition_id=PARTITION_ID,
        statistical_unit=STATISTICAL_UNIT,
        local_covariance=0.1 * np.eye(2, dtype=np.float64),
        shared_factor=factor,
        query_jacobian=jacobian,
        query_noise_covariance=(
            0.01 * np.eye(jacobian.shape[0], dtype=np.float64)
        ),
        policy=policy,
        frozen_before_target_outcomes=True,
        target_outcomes_used_for_selection=target_outcomes_used,
        calibration_groups_independent=True,
    )


def _calibration_application(hybrid, *, matching_input: bool = True):
    return DomainCovarianceCalibrationApplicationV2(
        certificate_id="7" * 64,
        certificate_semantics_id="8" * 64,
        application_semantics_id="8" * 64,
        domain_id="dynamic",
        evidence_decision_id="9" * 64,
        evidence_admissible=True,
        applied=True,
        reason="calibration-domain-authorized",
        source_application_id="a" * 64,
        raw_numeric_sha256="b" * 64,
        output_numeric_sha256="c" * 64,
        raw_array_sha256=(
            hybrid.record.output_covariance_sha256
            if matching_input
            else "d" * 64
        ),
        output_array_sha256="e" * 64,
        exact_fallback=False,
    )


def _harm_certificate():
    count = 60
    groups = tuple(f"certification-{index:03d}" for index in range(count))
    selected = tuple(f"{index + 1:064x}" for index in range(count))
    fallback = tuple(f"{index + count + 1:064x}" for index in range(count))
    return certify_guard_harm_risk_from_artifacts(
        guard_policy_id="f" * 64,
        threshold_source_artifact_id="0" * 64,
        certification_partition_id="1" * 64,
        statistical_unit=STATISTICAL_UNIT,
        metric="gaussian-nll",
        threshold_selection_group_ids=("selection-a", "selection-b"),
        group_ids=groups,
        risk_scores=np.zeros(count, dtype=np.float64),
        candidate_losses=np.ones(count, dtype=np.float64),
        fallback_losses=np.ones(count, dtype=np.float64),
        selected_artifact_ids=selected,
        fallback_artifact_ids=fallback,
        threshold=0.0,
        harm_margin=0.0,
        target_harm_probability=0.1,
        confidence_level=0.95,
        minimum_accepted_group_count=20,
        threshold_frozen_before_certification_outcomes=True,
        certification_outcomes_used_for_threshold_selection=False,
        certification_groups_independent=True,
    )


def _policy(*, mode: str = "explicit-joint"):
    return CovarianceOnlyTreatmentPolicyV1(
        covariance_mode=mode,
        evaluation_claim_id=EVALUATION_CLAIM_ID,
        evaluation_protocol_id=EVALUATION_PROTOCOL_ID,
        proper_score_metric_name=PROPER_SCORE_NAME,
        proper_score_comparison=PROPER_SCORE_COMPARISON,
        proper_score_rule=PROPER_SCORE_RULE,
        maximum_proper_score_value=0.0,
        minimum_simultaneous_coverage=0.9,
        maximum_mean_full_width_ratio=4.0,
    )


def _evaluation_decision(
    *,
    hybrid,
    calibration,
    harm,
    relevance,
    coverage: float = 0.91,
    width_ratio: float = 3.1,
    proper_score_value: float = -0.2,
    binding_override: dict[str, object] | None = None,
):
    metadata: dict[str, object] = {
        EVALUATION_BASELINE_BELIEF_ID_METADATA_KEY: BASELINE_ID,
        EVALUATION_CANDIDATE_BELIEF_ID_METADATA_KEY: CANDIDATE_ID,
        EVALUATION_COMMON_DOMAIN_ID_METADATA_KEY: COMMON_DOMAIN_ID,
        EVALUATION_HYBRID_RECORD_ID_METADATA_KEY: hybrid.record.artifact_id,
        EVALUATION_CANDIDATE_COVARIANCE_ID_METADATA_KEY: (
            calibration.output_array_sha256
        ),
        EVALUATION_CALIBRATION_APPLICATION_ID_METADATA_KEY: calibration.artifact_id,
        EVALUATION_HARM_RISK_CERTIFICATE_ID_METADATA_KEY: harm.artifact_id,
        EVALUATION_QUERY_RELEVANCE_CERTIFICATE_ID_METADATA_KEY: (
            relevance.artifact_id
        ),
        EVALUATION_QUERY_ID_METADATA_KEY: QUERY_ID,
        EVALUATION_CALIBRATION_PARTITION_ID_METADATA_KEY: PARTITION_ID,
        EVALUATION_STATISTICAL_UNIT_METADATA_KEY: STATISTICAL_UNIT,
        EVALUATION_MEAN_IDENTITY_VERIFIED_METADATA_KEY: True,
        EVALUATION_SIMULTANEOUS_COVERAGE_METADATA_KEY: coverage,
        EVALUATION_MEAN_FULL_WIDTH_RATIO_METADATA_KEY: width_ratio,
    }
    if binding_override is not None:
        metadata.update(binding_override)
    return EvidenceDecisionV1(
        claim_id=EVALUATION_CLAIM_ID,
        protocol_id=EVALUATION_PROTOCOL_ID,
        status="pass",
        run_classification="confirmatory",
        claim_authorized=False,
        evidence_level=2,
        metric=DecisionMetricV1(
            name=PROPER_SCORE_NAME,
            comparison=PROPER_SCORE_COMPARISON,
            rule=PROPER_SCORE_RULE,
            observed_value=proper_score_value,
            threshold_value=0.0,
            unit="nats",
        ),
        run_manifest_id="b" * 64,
        evidence_fingerprint="c" * 64,
        evidence_summary_sha256="d" * 64,
        repositories=(
            RepositoryState(
                repository="IPS-Stuttgart/BayesianPhysTwin",
                revision="e" * 40,
                dirty=False,
                role="primary",
            ),
        ),
        metadata=metadata,
        created_utc="2026-08-11T00:00:00+00:00",
    )


def _decision(
    *,
    mode: str = "explicit-joint",
    matching_input: bool = True,
    coverage: float = 0.91,
    width_ratio: float = 3.1,
    proper_score_value: float = -0.2,
    binding_override: dict[str, object] | None = None,
):
    hybrid = _hybrid()
    calibration = _calibration_application(
        hybrid,
        matching_input=matching_input,
    )
    relevance = _relevance(
        hybrid,
        covariance_artifact_id=calibration.output_array_sha256,
        material=True,
    )
    harm = _harm_certificate()
    evaluation = _evaluation_decision(
        hybrid=hybrid,
        calibration=calibration,
        harm=harm,
        relevance=relevance,
        coverage=coverage,
        width_ratio=width_ratio,
        proper_score_value=proper_score_value,
        binding_override=binding_override,
    )
    decision = decide_covariance_only_treatment(
        baseline_belief_id=BASELINE_ID,
        candidate_belief_id=CANDIDATE_ID,
        common_domain_id=COMMON_DOMAIN_ID,
        candidate_covariance_artifact_id=calibration.output_array_sha256,
        query_id=QUERY_ID,
        calibration_partition_id=PARTITION_ID,
        statistical_unit=STATISTICAL_UNIT,
        hybrid_record=hybrid.record,
        calibration_application=calibration,
        harm_risk_certificate=harm,
        query_relevance=relevance,
        evaluation_decision=evaluation,
        candidate_inference_admissible=True,
        treatment_frozen_before_target_outcomes=True,
        target_outcomes_used_for_treatment_selection=False,
        independent_statistical_units=True,
        policy=_policy(mode=mode),
    )
    return hybrid, relevance, evaluation, decision


def test_query_projection_detects_material_shared_covariance() -> None:
    hybrid = _hybrid()
    certificate = _relevance(hybrid, material=True)

    assert certificate.deployment_admissible
    assert certificate.shared_covariance_material
    assert certificate.effective_query_rank == 1
    assert certificate.shared_trace_fraction > 0.8
    assert certificate.maximum_generalized_eigenvalue > 1.0
    assert certificate.reasons == ("shared-query-covariance-material",)


def test_materiality_is_independent_of_evidence_admissibility() -> None:
    hybrid = _hybrid()
    certificate = _relevance(
        hybrid,
        material=True,
        target_outcomes_used=True,
    )

    assert certificate.shared_covariance_material
    assert not certificate.deployment_admissible
    assert "target-outcomes-used-for-query-rule" in certificate.reasons


def test_query_projection_detects_query_null_shared_mode() -> None:
    hybrid = _hybrid()
    certificate = _relevance(hybrid, material=False)

    assert certificate.deployment_admissible
    assert not certificate.shared_covariance_material
    assert certificate.effective_query_rank == 0
    assert certificate.null_mode_fraction == 1.0
    assert "effective-query-rank-below-threshold" in certificate.reasons
    assert "query-null-mode-fraction-exceeds-limit" in certificate.reasons


def test_query_relevance_rejects_complex_covariance() -> None:
    hybrid = _hybrid()
    with pytest.raises(ValueError, match="real numeric"):
        certify_query_covariance_relevance(
            query_id=QUERY_ID,
            covariance_artifact_id=hybrid.record.artifact_id,
            jacobian_artifact_id=JACOBIAN_ID,
            calibration_partition_id=PARTITION_ID,
            statistical_unit=STATISTICAL_UNIT,
            local_covariance=np.eye(2, dtype=np.complex128),
            shared_factor=np.ones((2, 1), dtype=np.float64),
            query_jacobian=np.eye(2, dtype=np.float64),
            query_noise_covariance=None,
            policy=QueryCovarianceRelevancePolicyV1(
                minimum_shared_trace_fraction=0.1,
                minimum_maximum_generalized_eigenvalue=0.1,
            ),
            frozen_before_target_outcomes=True,
            target_outcomes_used_for_selection=False,
            calibration_groups_independent=True,
        )


def test_covariance_only_treatment_selects_complete_candidate() -> None:
    _, _, evaluation, decision = _decision()
    baseline = _Belief(BASELINE_ID)
    candidate = _Belief(CANDIDATE_ID)

    selected, record = select_covariance_only_belief(
        baseline,
        candidate,
        decision,
    )

    assert decision.treatment_authorized
    assert decision.evaluation_decision_id == evaluation.decision_id
    assert decision.proper_score_supported
    assert decision.reasons == ("covariance-treatment-authorized",)
    assert selected is candidate
    assert record.selected_candidate
    assert record.selected_belief_id == CANDIDATE_ID


def test_width_rejection_returns_exact_baseline_object() -> None:
    _, _, _, decision = _decision(width_ratio=4.01)
    baseline = _Belief(BASELINE_ID)
    candidate = _Belief(CANDIDATE_ID)

    selected, record = select_covariance_only_belief(
        baseline,
        candidate,
        decision,
    )

    assert not decision.treatment_authorized
    assert "mean-full-width-ratio-exceeds-limit" in decision.reasons
    assert selected is baseline
    assert not record.selected_candidate
    assert record.selected_belief_id == BASELINE_ID


def test_proper_score_rejection_returns_exact_baseline_object() -> None:
    _, _, _, decision = _decision(proper_score_value=0.01)
    baseline = _Belief(BASELINE_ID)
    candidate = _Belief(CANDIDATE_ID)

    selected, _ = select_covariance_only_belief(
        baseline,
        candidate,
        decision,
    )

    assert not decision.treatment_authorized
    assert "proper-score-gate-rejected" in decision.reasons
    assert selected is baseline


def test_marginal_mode_rejects_material_shared_covariance() -> None:
    _, _, _, decision = _decision(mode="marginal")

    assert not decision.treatment_authorized
    assert "query-covariance-mode-mismatch" in decision.reasons


def test_calibration_input_identity_mismatch_rejects() -> None:
    _, _, _, decision = _decision(matching_input=False)

    assert not decision.treatment_authorized
    assert "hybrid-calibration-input-identity-mismatch" in decision.reasons


def test_evaluation_binding_mismatch_rejects() -> None:
    _, _, _, decision = _decision(
        binding_override={
            EVALUATION_CANDIDATE_BELIEF_ID_METADATA_KEY: "f" * 64,
        }
    )

    assert not decision.treatment_authorized
    assert "evaluation-evidence-decision-rejected" in decision.reasons
    assert "proper-score-gate-rejected" in decision.reasons


def test_target_informed_treatment_rejects() -> None:
    hybrid = _hybrid()
    calibration = _calibration_application(hybrid)
    relevance = _relevance(
        hybrid,
        covariance_artifact_id=calibration.output_array_sha256,
        material=True,
    )
    harm = _harm_certificate()
    evaluation = _evaluation_decision(
        hybrid=hybrid,
        calibration=calibration,
        harm=harm,
        relevance=relevance,
    )
    decision = decide_covariance_only_treatment(
        baseline_belief_id=BASELINE_ID,
        candidate_belief_id=CANDIDATE_ID,
        common_domain_id=COMMON_DOMAIN_ID,
        candidate_covariance_artifact_id=calibration.output_array_sha256,
        query_id=QUERY_ID,
        calibration_partition_id=PARTITION_ID,
        statistical_unit=STATISTICAL_UNIT,
        hybrid_record=hybrid.record,
        calibration_application=calibration,
        harm_risk_certificate=harm,
        query_relevance=relevance,
        evaluation_decision=evaluation,
        candidate_inference_admissible=True,
        treatment_frozen_before_target_outcomes=True,
        target_outcomes_used_for_treatment_selection=True,
        independent_statistical_units=True,
        policy=_policy(),
    )

    assert not decision.treatment_authorized
    assert "target-outcomes-used-for-treatment-selection" in decision.reasons


def test_content_addressed_decisions_reject_boolean_tampering() -> None:
    _, relevance, _, decision = _decision()

    with pytest.raises(ValueError, match="shared_covariance_material"):
        replace(relevance, shared_covariance_material=False)
    with pytest.raises(ValueError, match="treatment_authorized"):
        replace(decision, treatment_authorized=False)
