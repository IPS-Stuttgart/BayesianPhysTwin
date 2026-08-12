from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, replace
from typing import Any, cast

import pytest

from bayesian_phystwin.complete_belief_selection import (
    CompleteBeliefGuardDecisionV1,
)
from bayesian_phystwin.inference.components_v1 import (
    BELIEF_COMPONENT_ADMISSION_CLAIM_BOUNDARY,
    BELIEF_COMPONENT_ADMISSION_DECISION_SCHEMA,
    BELIEF_COMPONENT_ADMISSION_POLICY_SCHEMA,
    BeliefComponentAdmissionDecisionV1,
    BeliefComponentAdmissionPolicyV1,
    BeliefComponentAdmissionResultV1,
    compose_belief_component_admission,
    load_belief_component_admission_decision,
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


def _policy() -> BeliefComponentAdmissionPolicyV1:
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
    )


def _mean_guard(
    policy: BeliefComponentAdmissionPolicyV1,
) -> CompleteBeliefGuardDecisionV1:
    return CompleteBeliefGuardDecisionV1(
        baseline_belief_id=policy.deterministic_reference_arm_id,
        candidate_belief_id=policy.mean_candidate_arm_id,
        common_domain_id=policy.common_domain_id,
        certificate_id=_digest("mean-certificate"),
        inference_admissible=True,
        regret_guard_accepted=True,
        reason="registered point-mean guard",
    )


def _covariance_decision(
    policy: BeliefComponentAdmissionPolicyV1,
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
        selected_covariance_treatment=(
            COMPLETE_EXPLICIT_JOINT_GAUGE_COVARIANCE
        ),
        principal_covariance_treatment=(
            COMPLETE_EXPLICIT_JOINT_GAUGE_COVARIANCE
        ),
        principal_treatment_matches=True,
        value_certificate_certified=True,
        authorized=True,
        reasons=("covariance-treatment-authorized",),
    )


def _decision() -> BeliefComponentAdmissionDecisionV1:
    policy = _policy()
    return compose_belief_component_admission(
        policy,
        _mean_guard(policy),
        _covariance_decision(policy),
        covariance_candidate_admissible=True,
        common_prerequisites_admissible=True,
        reference_supported=True,
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
        "covariance_candidate_belief": _Belief(
            policy.covariance_candidate_arm_id
        ),
        "full_belief": _Belief(policy.full_belief_arm_id),
    }


def test_policy_rejects_aliasing_and_coerced_permissions() -> None:
    policy = _policy()
    with pytest.raises(ValueError, match="arm identities must be distinct"):
        replace(policy, mean_candidate_arm_id=policy.full_belief_arm_id)
    with pytest.raises(ValueError, match="covariance policies must differ"):
        replace(
            policy,
            candidate_covariance_policy_id=(
                policy.reference_covariance_policy_id
            ),
        )
    with pytest.raises(ValueError, match="allow_covariance_only must be boolean"):
        replace(policy, allow_covariance_only=cast(Any, 1))
    with pytest.raises(ValueError, match="allow_mean_only must be boolean"):
        replace(policy, allow_mean_only=cast(Any, "false"))


def test_policy_mapping_rejects_schema_version_boundary_and_identity() -> None:
    policy = _policy()
    base = policy.to_record()
    mutations = (
        ("schema", "other", "policy schema changed"),
        ("schema_version", True, "must be an integer >= 1"),
        ("schema_version", 2, "policy version changed"),
        ("claim_boundary", "weaker", "claim boundary changed"),
        ("policy_id", _digest("forged"), "policy_id does not match"),
    )
    for field, value, message in mutations:
        payload = copy.deepcopy(base)
        payload[field] = value
        with pytest.raises(ValueError, match=message):
            BeliefComponentAdmissionPolicyV1.from_mapping(payload)

    with pytest.raises(ValueError, match="must be a JSON object"):
        BeliefComponentAdmissionPolicyV1.from_mapping([])
    nonliteral = cast(dict[Any, Any], copy.deepcopy(base))
    nonliteral[1] = "bad"
    with pytest.raises(ValueError, match="literal string keys"):
        BeliefComponentAdmissionPolicyV1.from_mapping(nonliteral)


def test_decision_constructor_rejects_invalid_gates_and_derived_fields() -> None:
    decision = _decision()
    with pytest.raises(TypeError, match="policy must be"):
        replace(decision, policy=cast(Any, object()))
    with pytest.raises(
        ValueError,
        match="mean_regret_guard_accepted requires mean inference",
    ):
        replace(
            decision,
            mean_inference_admissible=False,
            mean_regret_guard_accepted=True,
        )
    with pytest.raises(ValueError, match="unsupported component-admission mode"):
        replace(decision, selected_mode=cast(Any, "other"))
    with pytest.raises(ValueError, match="selected component arm contradicts"):
        replace(decision, selected_arm_id=_digest("other-arm"))
    with pytest.raises(ValueError, match="reasons do not match"):
        replace(decision, reasons=("wrong",))
    with pytest.raises(ValueError, match="reasons must not contain duplicates"):
        replace(decision, reasons=("same", "same"))
    with pytest.raises(ValueError, match="nonempty strings"):
        replace(decision, reasons=cast(Any, (1,)))
    with pytest.raises(ValueError, match="artifact_id does not match"):
        replace(decision, artifact_id=_digest("forged"))


def test_decision_mapping_rejects_schema_and_derived_summary_tampering() -> None:
    decision = _decision()
    base = decision.to_record()
    mutations = (
        ("schema", "other", "decision schema changed"),
        ("schema_version", True, "must be an integer >= 1"),
        ("schema_version", 2, "decision version changed"),
        ("claim_boundary", "weaker", "claim boundary changed"),
        ("artifact_id", _digest("forged"), "artifact_id does not match"),
        ("mean_authorized", False, "mean_authorized contradicts"),
        ("covariance_authorized", False, "covariance_authorized contradicts"),
        ("exact_fallback", True, "exact_fallback contradicts"),
    )
    for field, value, message in mutations:
        payload = copy.deepcopy(base)
        payload[field] = value
        with pytest.raises(ValueError, match=message):
            BeliefComponentAdmissionDecisionV1.from_mapping(payload)

    payload = copy.deepcopy(base)
    payload["reasons"] = "not-an-array"
    with pytest.raises(ValueError, match="reasons must be a JSON array"):
        BeliefComponentAdmissionDecisionV1.from_mapping(payload)


def test_composer_rejects_wrong_types_and_cross_artifact_mismatches() -> None:
    policy = _policy()
    mean = _mean_guard(policy)
    covariance = _covariance_decision(policy)
    kwargs = {
        "covariance_candidate_admissible": True,
        "common_prerequisites_admissible": True,
        "reference_supported": True,
    }
    with pytest.raises(TypeError, match="policy must be"):
        compose_belief_component_admission(
            cast(Any, object()), mean, covariance, **kwargs
        )
    with pytest.raises(TypeError, match="mean_guard_decision must be"):
        compose_belief_component_admission(
            policy, cast(Any, object()), covariance, **kwargs
        )
    with pytest.raises(TypeError, match="covariance_decision must be"):
        compose_belief_component_admission(
            policy, mean, cast(Any, object()), **kwargs
        )

    mean_mutations = (
        ("common_domain_id", _digest("other-domain"), "different common domain"),
        (
            "baseline_belief_id",
            _digest("other-reference"),
            "deterministic reference",
        ),
        ("candidate_belief_id", _digest("other-mean"), "mean candidate"),
    )
    for field, value, message in mean_mutations:
        with pytest.raises(ValueError, match=message):
            compose_belief_component_admission(
                policy, replace(mean, **{field: value}), covariance, **kwargs
            )

    covariance_mutations = (
        ("exact_fallback_id", _digest("other-fallback"), "exact fallback"),
        ("reference_policy_id", _digest("other-reference"), "reference policy"),
        ("candidate_policy_id", _digest("other-candidate"), "candidate policy"),
    )
    for field, value, message in covariance_mutations:
        with pytest.raises(ValueError, match=message):
            compose_belief_component_admission(
                policy,
                mean,
                replace(covariance, **{field: value}, artifact_id=None),
                **kwargs,
            )

    for field in kwargs:
        malformed = dict(kwargs)
        malformed[field] = 1
        with pytest.raises(ValueError, match=f"{field} must be boolean"):
            compose_belief_component_admission(
                policy,
                mean,
                covariance,
                **cast(dict[str, Any], malformed),
            )


def test_router_rejects_wrong_arm_identity_and_reconstructed_selection() -> None:
    decision = _decision()
    beliefs = _beliefs(decision.policy)
    with pytest.raises(TypeError, match="decision must be"):
        route_belief_component_admission(
            cast(Any, object()),
            **beliefs,
        )

    bad_beliefs = dict(beliefs)
    bad_beliefs["full_belief"] = _Belief(_digest("wrong-full"))
    with pytest.raises(ValueError, match="full-belief belief does not match"):
        route_belief_component_admission(decision, **bad_beliefs)

    with pytest.raises(TypeError, match="must expose artifact_id"):
        route_belief_component_admission(
            decision,
            **{
                **beliefs,
                "mean_candidate_belief": cast(Any, object()),
            },
        )

    with pytest.raises(ValueError, match="exact registered belief object"):
        BeliefComponentAdmissionResultV1(
            decision=decision,
            **beliefs,
            selected_belief=_Belief(decision.policy.full_belief_arm_id),
        )


def test_writer_type_guards_and_strict_loader(tmp_path) -> None:
    with pytest.raises(TypeError, match="policy must be"):
        write_belief_component_admission_policy(
            cast(Any, object()),
            tmp_path / "policy.json",
        )
    with pytest.raises(TypeError, match="decision must be"):
        write_belief_component_admission_decision(
            cast(Any, object()),
            tmp_path / "decision.json",
        )

    path = tmp_path / "duplicate.json"
    path.write_text('{"schema": 1, "schema": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_belief_component_admission_decision(path)


def test_public_constants_remain_exact() -> None:
    assert BELIEF_COMPONENT_ADMISSION_POLICY_SCHEMA.endswith("_policy")
    assert BELIEF_COMPONENT_ADMISSION_DECISION_SCHEMA.endswith("_decision")
    assert "deployment safety" in BELIEF_COMPONENT_ADMISSION_CLAIM_BOUNDARY
