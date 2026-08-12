from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pytest

from bayesian_phystwin.complete_belief_selection import (
    CompleteBeliefGuardDecisionV1,
)
from bayesian_phystwin.inference.components_v1 import (
    BeliefComponentAdmissionDecisionV1,
    BeliefComponentAdmissionPolicyV1,
    compose_belief_component_admission,
    load_belief_component_admission_decision,
    load_belief_component_admission_policy,
    route_belief_component_admission,
    write_belief_component_admission_decision,
    write_belief_component_admission_policy,
)
from bayesian_phystwin.physical_query_v1 import (
    COMPLETE_EXPLICIT_JOINT_GAUGE_COVARIANCE,
)
from bayesian_phystwin.query_covariance_decision_v1 import (
    QueryCovarianceTreatmentDecisionV1,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _Belief:
    artifact_id: str


def _policy(
    *,
    allow_covariance_only: bool = True,
    allow_mean_only: bool = False,
) -> BeliefComponentAdmissionPolicyV1:
    return BeliefComponentAdmissionPolicyV1(
        common_domain_id=_digest("domain"),
        exact_fallback_arm_id=_digest("fallback-arm"),
        deterministic_reference_arm_id=_digest("reference-arm"),
        mean_candidate_arm_id=_digest("mean-arm"),
        covariance_candidate_arm_id=_digest("covariance-arm"),
        full_belief_arm_id=_digest("full-arm"),
        exact_fallback_policy_id=_digest("fallback-policy"),
        reference_covariance_policy_id=_digest("reference-policy"),
        candidate_covariance_policy_id=_digest("candidate-policy"),
        allow_covariance_only=allow_covariance_only,
        allow_mean_only=allow_mean_only,
        metadata={"owner": "source-only"},
    )


def _mean_guard(
    policy: BeliefComponentAdmissionPolicyV1,
    *,
    inference_admissible: bool,
    regret_guard_accepted: bool,
) -> CompleteBeliefGuardDecisionV1:
    return CompleteBeliefGuardDecisionV1(
        baseline_belief_id=policy.deterministic_reference_arm_id,
        candidate_belief_id=policy.mean_candidate_arm_id,
        common_domain_id=policy.common_domain_id,
        certificate_id=_digest("mean-certificate"),
        inference_admissible=inference_admissible,
        regret_guard_accepted=regret_guard_accepted,
        reason="registered point-mean guard",
    )


def _covariance_decision(
    policy: BeliefComponentAdmissionPolicyV1,
    *,
    authorized: bool,
) -> QueryCovarianceTreatmentDecisionV1:
    return QueryCovarianceTreatmentDecisionV1(
        physical_query_id=_digest("query"),
        source_observation_artifact_id=_digest("observation"),
        projection_summary_id=_digest("projection"),
        value_certificate_id=_digest("covariance-certificate"),
        candidate_policy_id=policy.candidate_covariance_policy_id,
        reference_policy_id=policy.reference_covariance_policy_id,
        exact_fallback_id=policy.exact_fallback_policy_id,
        shared_covariance_relevance=0.5,
        relevance_threshold=0.1,
        selected_covariance_treatment=(COMPLETE_EXPLICIT_JOINT_GAUGE_COVARIANCE),
        principal_covariance_treatment=(COMPLETE_EXPLICIT_JOINT_GAUGE_COVARIANCE),
        principal_treatment_matches=True,
        value_certificate_certified=authorized,
        authorized=authorized,
        reasons=(
            ("covariance-treatment-authorized",)
            if authorized
            else ("covariance-value-certificate-rejected",)
        ),
    )


def _beliefs(
    policy: BeliefComponentAdmissionPolicyV1,
) -> dict[str, _Belief]:
    return {
        "exact_fallback_belief": _Belief(policy.exact_fallback_arm_id),
        "deterministic_reference_belief": _Belief(
            policy.deterministic_reference_arm_id
        ),
        "mean_candidate_belief": _Belief(policy.mean_candidate_arm_id),
        "covariance_candidate_belief": _Belief(policy.covariance_candidate_arm_id),
        "full_belief": _Belief(policy.full_belief_arm_id),
    }


@pytest.mark.parametrize(
    (
        "allow_covariance_only",
        "allow_mean_only",
        "mean_admissible",
        "mean_guarded",
        "covariance_authorized",
        "covariance_admissible",
        "common_admissible",
        "reference_supported",
        "expected_mode",
    ),
    [
        (True, False, True, True, True, True, True, True, "full-belief"),
        (True, False, False, False, True, True, True, True, "covariance-only"),
        (
            True,
            False,
            True,
            True,
            False,
            True,
            True,
            True,
            "deterministic-reference",
        ),
        (True, True, True, True, False, True, True, True, "mean-only"),
        (
            True,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            "deterministic-reference",
        ),
        (
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            "deterministic-reference",
        ),
        (True, False, True, True, True, True, False, True, "exact-fallback"),
        (True, False, True, True, True, True, True, False, "exact-fallback"),
        (
            True,
            False,
            False,
            False,
            True,
            False,
            True,
            True,
            "deterministic-reference",
        ),
    ],
)
def test_component_admission_matrix_and_exact_object_routing(
    allow_covariance_only: bool,
    allow_mean_only: bool,
    mean_admissible: bool,
    mean_guarded: bool,
    covariance_authorized: bool,
    covariance_admissible: bool,
    common_admissible: bool,
    reference_supported: bool,
    expected_mode: str,
) -> None:
    policy = _policy(
        allow_covariance_only=allow_covariance_only,
        allow_mean_only=allow_mean_only,
    )
    decision = compose_belief_component_admission(
        policy,
        _mean_guard(
            policy,
            inference_admissible=mean_admissible,
            regret_guard_accepted=mean_guarded,
        ),
        _covariance_decision(policy, authorized=covariance_authorized),
        covariance_candidate_admissible=covariance_admissible,
        common_prerequisites_admissible=common_admissible,
        reference_supported=reference_supported,
        metadata={"case": expected_mode},
    )

    assert decision.selected_mode == expected_mode
    result = route_belief_component_admission(
        decision,
        **_beliefs(policy),
        metadata={"router": "exact-object"},
    )
    expected = {
        "exact-fallback": result.exact_fallback_belief,
        "deterministic-reference": result.deterministic_reference_belief,
        "mean-only": result.mean_candidate_belief,
        "covariance-only": result.covariance_candidate_belief,
        "full-belief": result.full_belief,
    }[expected_mode]
    assert result.selected_belief is expected
    assert result.artifact_id == result.artifact_id
    assert result.exact_fallback is (expected_mode == "exact-fallback")


def test_policy_and_decision_roundtrip_and_atomic_publication(tmp_path) -> None:
    policy = _policy(allow_mean_only=True)
    decision = compose_belief_component_admission(
        policy,
        _mean_guard(
            policy,
            inference_admissible=True,
            regret_guard_accepted=True,
        ),
        _covariance_decision(policy, authorized=True),
        covariance_candidate_admissible=True,
        common_prerequisites_admissible=True,
        reference_supported=True,
    )

    restored_policy = BeliefComponentAdmissionPolicyV1.from_mapping(policy.to_record())
    restored_decision = BeliefComponentAdmissionDecisionV1.from_mapping(
        decision.to_record()
    )
    assert restored_policy.policy_id == policy.policy_id
    assert restored_decision.artifact_id == decision.artifact_id

    policy_path = tmp_path / "policy.json"
    decision_path = tmp_path / "decision.json"
    write_belief_component_admission_policy(policy, policy_path)
    write_belief_component_admission_decision(decision, decision_path)
    assert load_belief_component_admission_policy(policy_path) == policy
    assert load_belief_component_admission_decision(decision_path) == decision
    with pytest.raises(FileExistsError):
        write_belief_component_admission_policy(policy, policy_path)
    with pytest.raises(FileExistsError):
        write_belief_component_admission_decision(decision, decision_path)
