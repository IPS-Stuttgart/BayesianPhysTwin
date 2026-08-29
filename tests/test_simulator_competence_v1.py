from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bayesian_phystwin.evidence_decision_v1 import (
    DecisionMetricV1,
    EvidenceDecisionV1,
)
from bayesian_phystwin.guard_harm_risk import (
    GuardHarmRiskCertificateV1,
    certify_guard_harm_risk,
)
from bayesian_phystwin.material_backend_evidence_v1 import (
    MaterialBackendEvidenceStatusV1,
    build_material_backend_evidence_status_v1,
)
from bayesian_phystwin.material_backend_qualification_v1 import (
    MaterialBackendQualificationV1,
)
from bayesian_phystwin.repository_provenance import RepositoryState
from bayesian_phystwin.simulator_competence_v1 import (
    QueryConditionalCompetenceCertificateV1,
    SimulatorCompetencePolicyV1,
    SimulatorQueryContextV1,
    build_query_conditional_competence_certificate_v1,
    load_query_conditional_competence_certificate_v1,
    route_query_conditional_prediction_v1,
    save_query_conditional_competence_certificate_v1,
)

A = "a" * 64
B = "b" * 64
C = "c" * 64
D = "d" * 64
E = "e" * 64
F = "f" * 64
ONE = "1" * 64
TWO = "2" * 64
THREE = "3" * 64
FOUR = "4" * 64
FIVE = "5" * 64
SIX = "6" * 64
SEVEN = "7" * 64
EIGHT = "8" * 64
NINE = "9" * 64
REVISION = "1" * 40


def _qualification() -> MaterialBackendQualificationV1:
    return MaterialBackendQualificationV1(
        canonical_profile_id="jax-fem-quasistatic-v1",
        producer_profile_id="jax-fem-quasistatic-v1",
        transport="lagrangian-export-v1",
        runtime_id=A,
        qualification_protocol_id=B,
        source_evidence_id=C,
        source_group_ids=("source-c", "source-a", "source-b"),
        incumbent_runtime_id=D,
        units_coordinate_entity_order_valid=True,
        deterministic_replay_valid=True,
        maximum_zero_action_drift_m=0.0001,
        allowed_zero_action_drift_m=0.001,
        maximum_rigid_equivariance_error_m=0.0001,
        allowed_rigid_equivariance_error_m=0.001,
        time_step_refinement_relative_error=0.01,
        allowed_time_step_refinement_relative_error=0.05,
        topology_identity_preserved=True,
        physical_sanity_violations=0,
        gradient_claimed=True,
        maximum_jacobian_relative_error=0.01,
        allowed_jacobian_relative_error=0.05,
        source_query_parity_rmse_m=0.001,
        allowed_source_query_parity_rmse_m=0.005,
        exact_fallback_verified=True,
        protocol_frozen_before_source_outcomes=True,
        target_outcomes_used=False,
    )


def _source_decision(
    qualification: MaterialBackendQualificationV1,
    *,
    method_id: str = E,
    partition_id: str = F,
) -> EvidenceDecisionV1:
    qualification_id = qualification.artifact_id
    assert qualification_id is not None
    return EvidenceDecisionV1(
        claim_id="query-conditional-source-competence",
        protocol_id="query-conditional-source-protocol-v1",
        status="pass",
        run_classification="controlled",
        claim_authorized=False,
        evidence_level=2,
        metric=DecisionMetricV1(
            name="selective-endpoint-rmse",
            comparison="candidate-minus-fallback",
            rule="source-nonregression-and-risk-ordering",
            observed_value=-0.01,
            threshold_value=0.0,
            unit="m",
        ),
        run_manifest_id=ONE,
        evidence_fingerprint=TWO,
        evidence_summary_sha256=THREE,
        repositories=(
            RepositoryState(
                repository="IPS-Stuttgart/BayesianPhysTwin",
                revision=REVISION,
                dirty=False,
                role="primary",
            ),
        ),
        metadata={
            "evidence_role": "source-competence",
            "canonical_profile_id": "jax-fem-quasistatic-v1",
            "producer_profile_id": "jax-fem-quasistatic-v1",
            "runtime_id": A,
            "qualification_artifact_id": qualification_id,
            "query_competence_method_id": method_id,
            "method_selection_partition_id": partition_id,
        },
        created_utc="2026-08-30T00:00:00+00:00",
    )


def _status(
    qualification: MaterialBackendQualificationV1,
    source_decision: EvidenceDecisionV1,
) -> MaterialBackendEvidenceStatusV1:
    return build_material_backend_evidence_status_v1(
        canonical_profile_id="jax-fem-quasistatic-v1",
        producer_profile_id="jax-fem-quasistatic-v1",
        adapter_evidence_id=FOUR,
        runtime_id=A,
        native_replay_evidence_id=FIVE,
        qualification=qualification,
        source_decision=source_decision,
    )


def _policy(
    status: MaterialBackendEvidenceStatusV1, **changes: Any
) -> SimulatorCompetencePolicyV1:
    status_id = status.artifact_id
    assert status_id is not None
    values: dict[str, Any] = {
        "canonical_profile_id": "jax-fem-quasistatic-v1",
        "producer_profile_id": "jax-fem-quasistatic-v1",
        "runtime_id": A,
        "backend_evidence_status_id": status_id,
        "method_artifact_id": E,
        "method_selection_partition_id": F,
        "method_selection_group_ids": (
            "source-a",
            "source-b",
            "source-c",
            "threshold-a",
        ),
        "object_domain_id": SIX,
        "action_domain_id": SEVEN,
        "allowed_query_functional_ids": (EIGHT,),
        "minimum_horizon_seconds": 0.1,
        "maximum_horizon_seconds": 2.0,
        "maximum_horizon_step_count": 200,
        "loss_metric": "endpoint-rmse-m",
        "statistical_unit": "independent-physical-object-v1",
        "risk_feature_schema_id": NINE,
        "risk_model_id": B,
        "threshold_source_artifact_id": C,
        "risk_threshold": 0.25,
        "fallback_policy_id": D,
    }
    values.update(changes)
    return SimulatorCompetencePolicyV1(**values)


def _risk_certificate(
    policy: SimulatorCompetencePolicyV1,
    *,
    count: int = 14,
    minimum_accepted_group_count: int = 14,
    group_ids: tuple[str, ...] | None = None,
) -> GuardHarmRiskCertificateV1:
    groups = group_ids or tuple(f"certification-{index:02d}" for index in range(count))
    return certify_guard_harm_risk(
        guard_policy_id=policy.policy_id,
        threshold_source_artifact_id=policy.threshold_source_artifact_id,
        certification_partition_id=ONE,
        statistical_unit=policy.statistical_unit,
        metric=policy.loss_metric,
        threshold_selection_group_ids=("threshold-a",),
        group_ids=groups,
        risk_scores=np.linspace(0.01, 0.20, count),
        candidate_losses=np.ones(count),
        fallback_losses=np.ones(count),
        fallback_identity_verified=np.ones(count, dtype=bool),
        threshold=policy.risk_threshold,
        harm_margin=0.0,
        target_harm_probability=0.20,
        confidence_level=0.95,
        minimum_accepted_group_count=minimum_accepted_group_count,
        threshold_frozen_before_certification_outcomes=True,
        certification_outcomes_used_for_threshold_selection=False,
        certification_groups_independent=True,
    )


def _bundle() -> tuple[
    MaterialBackendQualificationV1,
    EvidenceDecisionV1,
    MaterialBackendEvidenceStatusV1,
    SimulatorCompetencePolicyV1,
    GuardHarmRiskCertificateV1,
    QueryConditionalCompetenceCertificateV1,
]:
    qualification = _qualification()
    source = _source_decision(qualification)
    status = _status(qualification, source)
    policy = _policy(status)
    risk = _risk_certificate(policy)
    certificate = build_query_conditional_competence_certificate_v1(
        policy=policy,
        backend_evidence_status=status,
        qualification=qualification,
        source_decision=source,
        harm_risk_certificate=risk,
        certificate_frozen_before_target_outcomes=True,
        target_outcomes_used=False,
    )
    return qualification, source, status, policy, risk, certificate


def _query(**changes: Any) -> SimulatorQueryContextV1:
    values: dict[str, Any] = {
        "object_context_id": ONE,
        "object_domain_id": SIX,
        "action_context_id": TWO,
        "action_domain_id": SEVEN,
        "horizon_seconds": 1.0,
        "horizon_step_count": 100,
        "query_functional_id": EIGHT,
        "loss_metric": "endpoint-rmse-m",
        "preoutcome_features_id": THREE,
        "outcome_observed": False,
    }
    values.update(changes)
    return SimulatorQueryContextV1(**values)


def _route(
    certificate: QueryConditionalCompetenceCertificateV1,
    *,
    query: SimulatorQueryContextV1 | None = None,
    risk_score: object = 0.20,
    **changes: Any,
) -> tuple[Any, object, object, object]:
    candidate = object()
    fallback = object()
    values: dict[str, Any] = {
        "certificate": certificate,
        "query": query or _query(),
        "risk_score": risk_score,
        "canonical_profile_id": "jax-fem-quasistatic-v1",
        "producer_profile_id": "jax-fem-quasistatic-v1",
        "runtime_id": A,
        "risk_feature_schema_id": NINE,
        "risk_model_id": B,
        "fallback_policy_id": D,
        "candidate_prediction_id": FOUR,
        "fallback_prediction_id": FIVE,
        "candidate_prediction": candidate,
        "fallback_prediction": fallback,
    }
    values.update(changes)
    decision, selected = route_query_conditional_prediction_v1(**values)
    return decision, selected, candidate, fallback


def test_certified_in_scope_low_risk_query_selects_candidate() -> None:
    *_, certificate = _bundle()

    decision, selected, candidate, fallback = _route(certificate)

    assert decision.authorized
    assert not decision.exact_fallback
    assert decision.reasons == ("simulator-query-authorized",)
    assert selected is candidate
    assert selected is not fallback


def test_high_risk_query_returns_exact_fallback_object() -> None:
    *_, certificate = _bundle()

    decision, selected, candidate, fallback = _route(
        certificate,
        risk_score=0.2500001,
    )

    assert not decision.authorized
    assert decision.exact_fallback
    assert "risk-score-exceeds-threshold" in decision.reasons
    assert selected is fallback
    assert selected is not candidate


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        (
            {"canonical_profile_id": "wrong-profile"},
            "canonical-backend-profile-mismatch",
        ),
        (
            {"producer_profile_id": "wrong-producer"},
            "producer-backend-profile-mismatch",
        ),
        ({"runtime_id": E}, "backend-runtime-mismatch"),
        ({"risk_feature_schema_id": E}, "risk-feature-schema-mismatch"),
        ({"risk_model_id": E}, "risk-model-mismatch"),
        ({"fallback_policy_id": E}, "fallback-policy-mismatch"),
    ],
)
def test_runtime_binding_substitution_forces_fallback(
    changes: dict[str, object],
    reason: str,
) -> None:
    *_, certificate = _bundle()

    decision, selected, _, fallback = _route(certificate, **changes)

    assert not decision.authorized
    assert reason in decision.reasons
    assert selected is fallback


@pytest.mark.parametrize(
    ("query_changes", "reason"),
    [
        ({"object_domain_id": A}, "object-domain-out-of-scope"),
        ({"action_domain_id": A}, "action-domain-out-of-scope"),
        ({"query_functional_id": A}, "query-functional-out-of-scope"),
        ({"loss_metric": "another-loss"}, "query-loss-metric-mismatch"),
        ({"horizon_seconds": 2.1}, "query-horizon-out-of-scope"),
        ({"horizon_step_count": 201}, "query-horizon-out-of-scope"),
    ],
)
def test_query_scope_substitution_forces_fallback(
    query_changes: dict[str, object],
    reason: str,
) -> None:
    *_, certificate = _bundle()

    decision, selected, _, fallback = _route(
        certificate,
        query=_query(**query_changes),
    )

    assert not decision.authorized
    assert reason in decision.reasons
    assert selected is fallback


@pytest.mark.parametrize("risk_score", [float("nan"), float("inf"), True, "0.1"])
def test_invalid_risk_score_forces_fallback(risk_score: object) -> None:
    *_, certificate = _bundle()

    decision, selected, _, fallback = _route(
        certificate,
        risk_score=risk_score,
    )

    assert decision.risk_score is None
    assert decision.reasons == ("risk-score-invalid",)
    assert selected is fallback


def test_query_context_rejects_any_opened_outcome() -> None:
    with pytest.raises(ValueError, match="outcome-unopened"):
        _query(outcome_observed=True)


def test_underpowered_or_overlapping_certificate_fails_closed() -> None:
    qualification = _qualification()
    source = _source_decision(qualification)
    status = _status(qualification, source)
    policy = _policy(status)
    underpowered = _risk_certificate(
        policy,
        count=5,
        minimum_accepted_group_count=14,
    )
    assert not underpowered.certified
    with pytest.raises(ValueError, match="not certified"):
        QueryConditionalCompetenceCertificateV1(
            policy=policy,
            backend_evidence_status=status,
            harm_risk_certificate=underpowered,
            certificate_frozen_before_target_outcomes=True,
            target_outcomes_used=False,
        )

    overlapping_policy = _policy(
        status,
        method_selection_group_ids=(
            "source-a",
            "source-b",
            "source-c",
            "threshold-a",
            "certification-00",
        ),
    )
    overlapping_risk = _risk_certificate(overlapping_policy)
    with pytest.raises(ValueError, match="groups overlap"):
        QueryConditionalCompetenceCertificateV1(
            policy=overlapping_policy,
            backend_evidence_status=status,
            harm_risk_certificate=overlapping_risk,
            certificate_frozen_before_target_outcomes=True,
            target_outcomes_used=False,
        )


def test_pre_target_certificate_rejects_stage_gap_and_target_use() -> None:
    qualification = _qualification()
    source = _source_decision(qualification)
    status = _status(qualification, source)
    policy = _policy(status)
    risk = _risk_certificate(policy)
    qualified_only = build_material_backend_evidence_status_v1(
        canonical_profile_id="jax-fem-quasistatic-v1",
        producer_profile_id="jax-fem-quasistatic-v1",
        adapter_evidence_id=FOUR,
        runtime_id=A,
        native_replay_evidence_id=FIVE,
        qualification=qualification,
    )
    with pytest.raises(ValueError, match="source competence"):
        build_query_conditional_competence_certificate_v1(
            policy=policy,
            backend_evidence_status=qualified_only,
            qualification=qualification,
            source_decision=source,
            harm_risk_certificate=risk,
            certificate_frozen_before_target_outcomes=True,
            target_outcomes_used=False,
        )
    with pytest.raises(ValueError, match="target outcomes"):
        QueryConditionalCompetenceCertificateV1(
            policy=policy,
            backend_evidence_status=status,
            harm_risk_certificate=risk,
            certificate_frozen_before_target_outcomes=True,
            target_outcomes_used=True,
        )


def test_source_decision_must_bind_method_and_partition() -> None:
    qualification = _qualification()
    wrong_source = _source_decision(qualification, method_id=FIVE)
    status = _status(qualification, wrong_source)
    policy = _policy(status)
    risk = _risk_certificate(policy)

    with pytest.raises(ValueError, match="query_competence_method_id"):
        build_query_conditional_competence_certificate_v1(
            policy=policy,
            backend_evidence_status=status,
            qualification=qualification,
            source_decision=wrong_source,
            harm_risk_certificate=risk,
            certificate_frozen_before_target_outcomes=True,
            target_outcomes_used=False,
        )


def test_certificate_roundtrip_preserves_all_nested_identities(tmp_path: Path) -> None:
    *_, certificate = _bundle()
    path = tmp_path / "competence-certificate.json"

    save_query_conditional_competence_certificate_v1(certificate, path)
    restored = load_query_conditional_competence_certificate_v1(path)

    assert restored.artifact_id == certificate.artifact_id
    assert restored.policy.policy_id == certificate.policy.policy_id
    assert (
        restored.backend_evidence_status.artifact_id
        == certificate.backend_evidence_status.artifact_id
    )
    assert (
        restored.harm_risk_certificate.artifact_id
        == certificate.harm_risk_certificate.artifact_id
    )
    with pytest.raises(FileExistsError):
        save_query_conditional_competence_certificate_v1(certificate, path)


def test_prediction_id_alias_and_decision_tampering_are_rejected() -> None:
    *_, certificate = _bundle()
    with pytest.raises(ValueError, match="must differ"):
        _route(certificate, fallback_prediction_id=FOUR)

    decision, *_ = _route(certificate)
    with pytest.raises(ValueError, match="selected prediction contradicts"):
        replace(decision, selected_prediction_id=decision.fallback_prediction_id)
    with pytest.raises(ValueError, match="artifact_id does not match"):
        replace(decision, artifact_id=A)
