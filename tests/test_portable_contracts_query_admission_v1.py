from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from bayesian_phystwin.evidence_decision_v1 import (
    DecisionMetricV1,
    EvidenceDecisionV1,
)
from bayesian_phystwin.physical_query_v1 import (
    COMPLETE_EXPLICIT_JOINT_GAUGE_COVARIANCE,
    MARGINAL_GAUGE_COVARIANCE,
    PhysicalQueryBootstrapV1,
    PhysicalQueryDecisionMarginsV1,
    PhysicalQueryV1,
)
from bayesian_phystwin.probabilistic_scoring import GAUSSIAN_NLL_PER_DIMENSION
from bayesian_phystwin.query_admission_v1 import (
    QueryAdmissionEvidenceV1,
    QueryAdmissionPolicyV1,
    compose_query_admission,
    load_query_admission_certificate,
    write_query_admission_certificate,
)
from bayesian_phystwin.query_covariance_decision_v1 import (
    TRACE_RELEVANCE_DIAGNOSTIC,
    TRACE_RELEVANCE_SELECTION_RULE,
    QueryCovarianceTreatmentDecisionV1,
)
from bayesian_phystwin.repository_provenance import RepositoryState


def _sha256(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _repositories() -> tuple[RepositoryState, ...]:
    return (
        RepositoryState(
            repository="IPS-Stuttgart/BayesianPhysTwin",
            revision="a" * 40,
            dirty=False,
            role="primary",
        ),
        RepositoryState(
            repository="IPS-Stuttgart/Prob4D",
            revision="b" * 40,
            dirty=False,
            role="observation",
        ),
        RepositoryState(
            repository="IPS-Stuttgart/Causal4D",
            revision="c" * 40,
            dirty=False,
            role="downstream",
        ),
    )


def _provider(**changes: object) -> EvidenceDecisionV1:
    values: dict[str, object] = {
        "claim_id": "bpt.provider.competence",
        "protocol_id": "fresh-provider-v1",
        "status": "pass",
        "run_classification": "confirmatory",
        "claim_authorized": True,
        "evidence_level": 3,
        "metric": DecisionMetricV1(
            name="physical_query_regret",
            comparison="candidate_vs_physical_fallback",
            rule="upper_bound_le_zero",
            observed_value=-0.02,
            threshold_value=0.0,
            unit="nll-per-dimension",
        ),
        "run_manifest_id": _sha256("run-manifest"),
        "evidence_fingerprint": _sha256("evidence-fingerprint"),
        "evidence_summary_sha256": _sha256("evidence-summary"),
        "repositories": _repositories(),
        "limitations": ("query-specific source evidence only",),
        "metadata": {"target_opened": False},
        "created_utc": "2026-08-14T08:00:00+00:00",
    }
    values.update(changes)
    return EvidenceDecisionV1(**values)  # type: ignore[arg-type]


def _query(provider_decision_id: str) -> PhysicalQueryV1:
    return PhysicalQueryV1(
        query_name="held-out-endpoint-displacement",
        dimension=2,
        component_order=("early-displacement", "late-displacement"),
        physical_unit="m",
        coordinate_frame="registered-world-frame",
        horizon_values=(0.08, 0.20),
        horizon_unit="s",
        jacobian_provider_id=_sha256("jacobian-provider"),
        baseline_physical_belief_id=_sha256("physical-belief"),
        exact_fallback_id=_sha256("fallback-bytes"),
        covariance_treatments=(
            MARGINAL_GAUGE_COVARIANCE,
            COMPLETE_EXPLICIT_JOINT_GAUGE_COVARIANCE,
        ),
        principal_covariance_treatment=(
            COMPLETE_EXPLICIT_JOINT_GAUGE_COVARIANCE
        ),
        primary_proper_score=GAUSSIAN_NLL_PER_DIMENSION,
        decision_margins=PhysicalQueryDecisionMarginsV1(
            practical_equivalence_score=0.02,
            maximum_harmful_score_increase=0.03,
            minimum_accepted_coverage=0.85,
            maximum_mean_width=0.20,
            maximum_worst_group_score_regret=0.04,
            minimum_shared_covariance_relevance=0.10,
            width_unit="m",
        ),
        shared_covariance_diagnostic=TRACE_RELEVANCE_DIAGNOSTIC,
        computational_selection_rule=TRACE_RELEVANCE_SELECTION_RULE,
        bootstrap=PhysicalQueryBootstrapV1(
            independent_group_definition="complete physical object/session",
            method="paired-stratified-group-bootstrap",
            resamples=10_000,
            seed=1731,
            confidence_level=0.95,
            stratification_keys=("object-stratum",),
        ),
        package_artifact_ids={
            "bayesian-phystwin": _sha256("bpt-wheel"),
            "prob4d": _sha256("prob4d-wheel"),
            "causal4d": _sha256("causal4d-wheel"),
        },
        provider_manifest_id=_sha256("provider-manifest"),
        evidence_decision_ids={
            "source-provider-gate": provider_decision_id,
        },
        repositories=_repositories(),
        metadata={"target_opened": False},
    )


def _covariance_decision(query: PhysicalQueryV1) -> QueryCovarianceTreatmentDecisionV1:
    return QueryCovarianceTreatmentDecisionV1(
        physical_query_id=query.query_id,
        source_observation_artifact_id=_sha256("observation"),
        covariance_value_certificate_id=_sha256("covariance-certificate"),
        selected_covariance_treatment=(
            COMPLETE_EXPLICIT_JOINT_GAUGE_COVARIANCE
        ),
        exact_fallback_id=query.exact_fallback_id,
        shared_covariance_relevance=0.2,
        relevance_diagnostic=TRACE_RELEVANCE_DIAGNOSTIC,
        selection_rule=TRACE_RELEVANCE_SELECTION_RULE,
        covariance_value_certified=True,
        covariance_treatment_matches_principal=True,
        authorized=True,
        reasons=("covariance-treatment-authorized",),
    )


def _evidence(query: PhysicalQueryV1, **changes: object) -> QueryAdmissionEvidenceV1:
    values: dict[str, object] = {
        "candidate_belief_id": _sha256("candidate-belief"),
        "candidate_query_mean_id": _sha256("candidate-query-mean"),
        "candidate_query_covariance_id": _sha256("candidate-query-covariance"),
        "baseline_query_mean_id": _sha256("baseline-query-mean"),
        "baseline_query_covariance_id": _sha256("baseline-query-covariance"),
        "evaluation_artifact_id": _sha256("query-evaluation"),
        "score_metric": query.primary_proper_score,
        "width_unit": query.physical_unit,
        "statistical_unit": query.bootstrap.independent_group_definition,
        "independent_group_count": 12,
        "mean_score_regret": -0.02,
        "score_regret_upper_bound": 0.0,
        "maximum_score_increase": 0.01,
        "worst_group_score_regret": 0.02,
        "harmful_group_fraction": 0.0,
        "accepted_coverage": 0.90,
        "mean_full_width": 0.12,
        "identifiable_subspace_overlap": 0.80,
        "shared_covariance_relevance": 0.20,
        "expected_information_gain": 0.30,
        "policy_frozen_before_evaluation_outcomes": True,
        "evaluation_outcomes_used_for_candidate_selection": False,
        "evaluation_groups_independent": True,
        "metadata": {"target_opened": False},
    }
    values.update(changes)
    return QueryAdmissionEvidenceV1(**values)  # type: ignore[arg-type]


def _policy() -> QueryAdmissionPolicyV1:
    return QueryAdmissionPolicyV1(
        minimum_group_count=10,
        minimum_identifiable_subspace_overlap=0.5,
        minimum_expected_information_gain=0.1,
        maximum_harmful_group_fraction=0.0,
    )


def test_query_admission_authorizes_complete_conjunctive_pass(
    tmp_path: Path,
) -> None:
    provider = _provider()
    query = _query(provider.decision_id)
    evidence = _evidence(query)

    certificate = compose_query_admission(
        query,
        provider,
        _covariance_decision(query),
        evidence,
        policy=_policy(),
    )

    assert certificate.admitted
    assert not certificate.exact_fallback
    assert certificate.provider_competence_passed
    assert certificate.query_nonharm_passed
    assert certificate.query_information_passed
    assert certificate.selected_belief_id == evidence.candidate_belief_id
    assert certificate.reasons == ("query-admission-authorized",)

    path = tmp_path / "query-admission.json"
    write_query_admission_certificate(certificate, path)
    assert load_query_admission_certificate(path) == certificate


def test_provider_failure_keeps_exact_physical_fallback() -> None:
    provider = _provider(
        status="fail",
        claim_authorized=False,
        run_classification="confirmatory",
    )
    query = _query(provider.decision_id)

    certificate = compose_query_admission(
        query,
        provider,
        _covariance_decision(query),
        _evidence(query),
        policy=_policy(),
    )

    assert not certificate.admitted
    assert certificate.exact_fallback
    assert not certificate.provider_competence_passed
    assert certificate.query_nonharm_passed
    assert certificate.query_information_passed
    assert certificate.selected_belief_id == query.baseline_physical_belief_id
    assert "provider-decision-not-pass" in certificate.reasons
    assert "provider-claim-not-authorized" in certificate.reasons


def test_query_regret_and_information_failures_are_separate() -> None:
    provider = _provider()
    query = _query(provider.decision_id)
    evidence = _evidence(
        query,
        score_regret_upper_bound=0.03,
        maximum_score_increase=0.04,
        worst_group_score_regret=0.05,
        identifiable_subspace_overlap=0.2,
        shared_covariance_relevance=0.05,
        expected_information_gain=0.01,
    )

    certificate = compose_query_admission(
        query,
        provider,
        _covariance_decision(query),
        evidence,
        policy=_policy(),
    )

    assert certificate.provider_competence_passed
    assert not certificate.query_nonharm_passed
    assert not certificate.query_information_passed
    assert certificate.exact_fallback
    assert certificate.selected_belief_id == query.baseline_physical_belief_id
    assert "query-score-regret-upper-bound-exceeds-margin" in certificate.reasons
    assert "maximum-score-increase-exceeds-margin" in certificate.reasons
    assert "identifiable-subspace-overlap-below-threshold" in certificate.reasons
    assert "shared-covariance-relevance-below-margin" in certificate.reasons


def test_information_order_and_grouping_fail_closed() -> None:
    provider = _provider()
    query = _query(provider.decision_id)
    evidence = _evidence(
        query,
        independent_group_count=3,
        policy_frozen_before_evaluation_outcomes=False,
        evaluation_outcomes_used_for_candidate_selection=True,
        evaluation_groups_independent=False,
    )

    certificate = compose_query_admission(
        query,
        provider,
        _covariance_decision(query),
        evidence,
        policy=_policy(),
    )

    assert certificate.exact_fallback
    assert "insufficient-independent-groups" in certificate.reasons
    assert "query-policy-not-frozen-before-outcomes" in certificate.reasons
    assert (
        "evaluation-outcomes-used-for-candidate-selection"
        in certificate.reasons
    )
    assert "evaluation-groups-not-independent" in certificate.reasons


def test_artifact_binding_mismatches_are_rejected() -> None:
    provider = _provider()
    query = _query(provider.decision_id)
    evidence = _evidence(query)

    other_provider = _provider(evidence_summary_sha256=_sha256("other-summary"))
    with pytest.raises(ValueError, match="provider decision differs"):
        compose_query_admission(
            query,
            other_provider,
            _covariance_decision(query),
            evidence,
            policy=_policy(),
        )

    with pytest.raises(ValueError, match="proper score"):
        compose_query_admission(
            query,
            provider,
            _covariance_decision(query),
            replace(evidence, score_metric="energy-score"),
            policy=_policy(),
        )

    mismatched_covariance = replace(
        _covariance_decision(query),
        physical_query_id=_sha256("other-query"),
        artifact_id=None,
    )
    with pytest.raises(ValueError, match="another physical query"):
        compose_query_admission(
            query,
            provider,
            mismatched_covariance,
            evidence,
            policy=_policy(),
        )


def test_redundant_decision_fields_cannot_be_tampered() -> None:
    provider = _provider()
    query = _query(provider.decision_id)
    certificate = compose_query_admission(
        query,
        provider,
        _covariance_decision(query),
        _evidence(query),
        policy=_policy(),
    )

    with pytest.raises(ValueError, match="admitted contradicts"):
        replace(certificate, admitted=False)
    with pytest.raises(ValueError, match="reasons contradict"):
        replace(
            certificate,
            reasons=("provider-decision-not-pass",),
        )
