"""Evidence-bound covariance-only treatment admission and exact fallback."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, TypeVar

from .._canonical_contracts import genuine_boolean
from ..complete_belief_selection import (
    ArtifactBelief,
    CompleteBeliefGuardDecisionV1,
    CompleteBeliefSelectionV1,
    select_complete_belief,
)
from ..covariance_only_hybrid import CovarianceOnlyHybridRecordV1
from ..domain_covariance_calibration_v2 import (
    DomainCovarianceCalibrationApplicationV2,
)
from ..evidence_decision_v1 import EvidenceDecisionV1
from ..guard_harm_risk_artifacts import GuardHarmRiskArtifactCertificateV1
from ._admission_decision import CovarianceOnlyTreatmentDecisionV1
from ._admission_policy import (
    CovarianceOnlyTreatmentPolicyV1,
    treatment_reasons,
)
from ._common import (
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
    canonical_string,
    finite_float,
    required_artifact_id,
)
from ._relevance_types import QueryCovarianceRelevanceCertificateV1

BeliefT = TypeVar("BeliefT", bound=ArtifactBelief)


def _evaluation_metrics(
    decision: EvidenceDecisionV1,
    *,
    policy: CovarianceOnlyTreatmentPolicyV1,
    expected_metadata: Mapping[str, object],
) -> tuple[bool, bool, float, float, float | None]:
    bindings_match = all(
        decision.metadata.get(key) == value
        for key, value in expected_metadata.items()
    )
    evidence_admissible = bool(
        decision.status == "pass"
        and decision.run_classification == "confirmatory"
        and decision.claim_id == policy.evaluation_claim_id
        and decision.protocol_id == policy.evaluation_protocol_id
        and decision.evidence_level >= policy.minimum_evidence_level
        and (
            not policy.require_claim_authorized_decision
            or decision.claim_authorized
        )
        and bindings_match
    )
    metric = decision.metric
    threshold = metric.threshold_value
    metric_identity_matches = bool(
        metric.name == policy.proper_score_metric_name
        and metric.comparison == policy.proper_score_comparison
        and metric.rule == policy.proper_score_rule
        and threshold is not None
        and math.isclose(
            float(threshold),
            policy.maximum_proper_score_value,
            abs_tol=policy.numerical_tolerance,
            rel_tol=0.0,
        )
    )
    proper_score_supported = bool(
        evidence_admissible
        and metric_identity_matches
        and metric.observed_value
        <= policy.maximum_proper_score_value + policy.numerical_tolerance
    )
    coverage = finite_float(
        decision.metadata.get(EVALUATION_SIMULTANEOUS_COVERAGE_METADATA_KEY),
        name=EVALUATION_SIMULTANEOUS_COVERAGE_METADATA_KEY,
        minimum=0.0,
        maximum=1.0,
    )
    width_ratio = finite_float(
        decision.metadata.get(EVALUATION_MEAN_FULL_WIDTH_RATIO_METADATA_KEY),
        name=EVALUATION_MEAN_FULL_WIDTH_RATIO_METADATA_KEY,
        minimum=0.0,
    )
    if width_ratio <= 0.0:
        raise ValueError("evaluation mean_full_width_ratio must be positive")
    return (
        evidence_admissible,
        proper_score_supported,
        coverage,
        width_ratio,
        None if threshold is None else float(threshold),
    )


def decide_covariance_only_treatment(
    *,
    baseline_belief_id: str,
    candidate_belief_id: str,
    common_domain_id: str,
    candidate_covariance_artifact_id: str,
    query_id: str,
    calibration_partition_id: str,
    statistical_unit: str,
    hybrid_record: CovarianceOnlyHybridRecordV1,
    calibration_application: DomainCovarianceCalibrationApplicationV2,
    harm_risk_certificate: GuardHarmRiskArtifactCertificateV1,
    query_relevance: QueryCovarianceRelevanceCertificateV1,
    evaluation_decision: EvidenceDecisionV1,
    candidate_inference_admissible: bool,
    treatment_frozen_before_target_outcomes: bool,
    target_outcomes_used_for_treatment_selection: bool,
    independent_statistical_units: bool,
    policy: CovarianceOnlyTreatmentPolicyV1,
    metadata: Mapping[str, Any] | None = None,
) -> CovarianceOnlyTreatmentDecisionV1:
    """Compose exact-mean, calibration, harm, query, and evidence gates."""

    if not isinstance(hybrid_record, CovarianceOnlyHybridRecordV1):
        raise TypeError("hybrid_record must be CovarianceOnlyHybridRecordV1")
    if not isinstance(
        calibration_application,
        DomainCovarianceCalibrationApplicationV2,
    ):
        raise TypeError(
            "calibration_application must be DomainCovarianceCalibrationApplicationV2"
        )
    if not isinstance(
        harm_risk_certificate,
        GuardHarmRiskArtifactCertificateV1,
    ):
        raise TypeError(
            "harm_risk_certificate must be GuardHarmRiskArtifactCertificateV1"
        )
    if not isinstance(query_relevance, QueryCovarianceRelevanceCertificateV1):
        raise TypeError(
            "query_relevance must be QueryCovarianceRelevanceCertificateV1"
        )
    if not isinstance(evaluation_decision, EvidenceDecisionV1):
        raise TypeError("evaluation_decision must be EvidenceDecisionV1")
    if not isinstance(policy, CovarianceOnlyTreatmentPolicyV1):
        raise TypeError("policy must be CovarianceOnlyTreatmentPolicyV1")

    baseline_id = required_artifact_id(
        baseline_belief_id,
        name="baseline_belief_id",
    )
    candidate_id = required_artifact_id(
        candidate_belief_id,
        name="candidate_belief_id",
    )
    domain_id = required_artifact_id(common_domain_id, name="common_domain_id")
    covariance_id = required_artifact_id(
        candidate_covariance_artifact_id,
        name="candidate_covariance_artifact_id",
    )
    query_digest = required_artifact_id(query_id, name="query_id")
    partition_id = required_artifact_id(
        calibration_partition_id,
        name="calibration_partition_id",
    )
    unit = canonical_string(statistical_unit, name="statistical_unit")
    hybrid_id = required_artifact_id(
        hybrid_record.artifact_id,
        name="hybrid_record.artifact_id",
    )
    calibration_application_id = required_artifact_id(
        calibration_application.artifact_id,
        name="calibration_application.artifact_id",
    )
    harm_id = required_artifact_id(
        harm_risk_certificate.artifact_id,
        name="harm_risk_certificate.artifact_id",
    )
    relevance_id = required_artifact_id(
        query_relevance.artifact_id,
        name="query_relevance.artifact_id",
    )
    if query_relevance.query_id != query_digest:
        raise ValueError("query relevance certificate binds a different query")
    if query_relevance.covariance_artifact_id != covariance_id:
        raise ValueError(
            "query relevance certificate binds a different covariance candidate"
        )
    if query_relevance.calibration_partition_id != partition_id:
        raise ValueError(
            "query relevance certificate binds a different calibration partition"
        )
    if query_relevance.statistical_unit != unit:
        raise ValueError("query relevance statistical unit does not match treatment")

    calibration_input_verified = bool(
        hybrid_record.output_covariance_sha256
        == calibration_application.raw_array_sha256
    )
    calibration_applied = bool(
        calibration_application.applied
        and not calibration_application.exact_fallback
        and calibration_application.evidence_admissible
    )
    query_mode_supported = bool(
        (
            policy.covariance_mode == "explicit-joint"
            and query_relevance.shared_covariance_material
        )
        or (
            policy.covariance_mode == "marginal"
            and not query_relevance.shared_covariance_material
        )
    )
    expected_metadata = {
        EVALUATION_BASELINE_BELIEF_ID_METADATA_KEY: baseline_id,
        EVALUATION_CANDIDATE_BELIEF_ID_METADATA_KEY: candidate_id,
        EVALUATION_COMMON_DOMAIN_ID_METADATA_KEY: domain_id,
        EVALUATION_HYBRID_RECORD_ID_METADATA_KEY: hybrid_id,
        EVALUATION_CANDIDATE_COVARIANCE_ID_METADATA_KEY: covariance_id,
        EVALUATION_CALIBRATION_APPLICATION_ID_METADATA_KEY: (
            calibration_application_id
        ),
        EVALUATION_HARM_RISK_CERTIFICATE_ID_METADATA_KEY: harm_id,
        EVALUATION_QUERY_RELEVANCE_CERTIFICATE_ID_METADATA_KEY: relevance_id,
        EVALUATION_QUERY_ID_METADATA_KEY: query_digest,
        EVALUATION_CALIBRATION_PARTITION_ID_METADATA_KEY: partition_id,
        EVALUATION_STATISTICAL_UNIT_METADATA_KEY: unit,
        EVALUATION_MEAN_IDENTITY_VERIFIED_METADATA_KEY: True,
    }
    (
        evaluation_admissible,
        proper_score_supported,
        coverage,
        width_ratio,
        score_threshold,
    ) = _evaluation_metrics(
        evaluation_decision,
        policy=policy,
        expected_metadata=expected_metadata,
    )
    inference_ok = genuine_boolean(
        candidate_inference_admissible,
        name="candidate_inference_admissible",
    )
    frozen = genuine_boolean(
        treatment_frozen_before_target_outcomes,
        name="treatment_frozen_before_target_outcomes",
    )
    target_used = genuine_boolean(
        target_outcomes_used_for_treatment_selection,
        name="target_outcomes_used_for_treatment_selection",
    )
    independent = genuine_boolean(
        independent_statistical_units,
        name="independent_statistical_units",
    )
    reasons = treatment_reasons(
        candidate_inference_admissible=inference_ok,
        calibration_input_identity_verified=calibration_input_verified,
        calibration_applied=calibration_applied,
        harm_risk_certified=harm_risk_certificate.certified,
        query_evidence_admissible=query_relevance.deployment_admissible,
        query_mode_supported=query_mode_supported,
        evaluation_evidence_admissible=evaluation_admissible,
        proper_score_supported=proper_score_supported,
        simultaneous_coverage=coverage,
        mean_full_width_ratio=width_ratio,
        treatment_frozen_before_target_outcomes=frozen,
        target_outcomes_used_for_treatment_selection=target_used,
        independent_statistical_units=independent,
        policy=policy,
    )
    return CovarianceOnlyTreatmentDecisionV1(
        baseline_belief_id=baseline_id,
        candidate_belief_id=candidate_id,
        common_domain_id=domain_id,
        query_id=query_digest,
        calibration_partition_id=partition_id,
        statistical_unit=unit,
        hybrid_record_id=hybrid_id,
        candidate_covariance_artifact_id=covariance_id,
        calibration_application_id=calibration_application_id,
        calibration_certificate_id=calibration_application.certificate_id,
        harm_risk_certificate_id=harm_id,
        query_relevance_certificate_id=relevance_id,
        evaluation_decision_id=evaluation_decision.decision_id,
        candidate_inference_admissible=inference_ok,
        calibration_input_identity_verified=calibration_input_verified,
        calibration_applied=calibration_applied,
        harm_risk_certified=harm_risk_certificate.certified,
        query_evidence_admissible=query_relevance.deployment_admissible,
        query_mode_supported=query_mode_supported,
        evaluation_evidence_admissible=evaluation_admissible,
        proper_score_supported=proper_score_supported,
        proper_score_observed_value=evaluation_decision.metric.observed_value,
        proper_score_threshold_value=score_threshold,
        simultaneous_coverage=coverage,
        mean_full_width_ratio=width_ratio,
        treatment_frozen_before_target_outcomes=frozen,
        target_outcomes_used_for_treatment_selection=target_used,
        independent_statistical_units=independent,
        policy=policy,
        treatment_authorized=(reasons == ("covariance-treatment-authorized",)),
        reasons=reasons,
        metadata={} if metadata is None else metadata,
    )


def select_covariance_only_belief(
    baseline: BeliefT,
    candidate: BeliefT,
    decision: CovarianceOnlyTreatmentDecisionV1,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[BeliefT, CompleteBeliefSelectionV1]:
    """Select the complete candidate or return the exact baseline object."""

    if not isinstance(decision, CovarianceOnlyTreatmentDecisionV1):
        raise TypeError("decision must be CovarianceOnlyTreatmentDecisionV1")
    guard = CompleteBeliefGuardDecisionV1(
        baseline_belief_id=decision.baseline_belief_id,
        candidate_belief_id=decision.candidate_belief_id,
        common_domain_id=decision.common_domain_id,
        certificate_id=required_artifact_id(
            decision.artifact_id,
            name="decision.artifact_id",
        ),
        inference_admissible=decision.candidate_inference_admissible,
        regret_guard_accepted=decision.treatment_authorized,
        reason=(
            "covariance-treatment-authorized"
            if decision.treatment_authorized
            else ",".join(decision.reasons)
        ),
        metadata={
            "covariance_treatment_decision_id": decision.artifact_id,
            "query_id": decision.query_id,
        },
    )
    return select_complete_belief(
        baseline,
        candidate,
        guard,
        metadata={
            "covariance_treatment_decision_id": decision.artifact_id,
            **({} if metadata is None else dict(metadata)),
        },
    )
