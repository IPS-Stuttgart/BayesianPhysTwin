from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any, cast

import pytest

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.complete_belief_selection import (
    CompleteBeliefGuardDecisionV1,
)
from bayesian_phystwin.inference.component_beliefs_v1 import (
    COMPONENTIZED_BELIEF_CLAIM_BOUNDARY,
    ComponentizedBeliefAdmissionResultV1,
    ComponentizedBeliefArmSetV1,
    bind_componentized_belief_arms,
    route_componentized_belief_admission,
)
from bayesian_phystwin.inference.components_v1 import (
    BeliefComponentAdmissionPolicyV1,
    compose_belief_component_admission,
    route_belief_component_admission,
)
from bayesian_phystwin.physical_query_v1 import (
    COMPLETE_EXPLICIT_JOINT_GAUGE_COVARIANCE,
)
from bayesian_phystwin.query_covariance_decision_v1 import (
    QueryCovarianceTreatmentDecisionV1,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


@dataclass(frozen=True)
class _Belief:
    artifact_id: str
    common_domain_id: str
    mean_component_id: str
    covariance_component_id: str


@dataclass
class _MutableBelief:
    artifact_id: str
    common_domain_id: str
    mean_component_id: str
    covariance_component_id: str


def _beliefs(*, mutable: bool = False) -> dict[str, Any]:
    kind = _MutableBelief if mutable else _Belief
    domain = _digest("domain")
    reference_mean = _digest("reference-mean")
    candidate_mean = _digest("candidate-mean")
    reference_covariance = _digest("reference-covariance")
    candidate_covariance = _digest("candidate-covariance")
    return {
        "exact_fallback_belief": kind(
            _digest("fallback-arm"),
            domain,
            _digest("physical-mean"),
            _digest("physical-covariance"),
        ),
        "deterministic_reference_belief": kind(
            _digest("reference-arm"),
            domain,
            reference_mean,
            reference_covariance,
        ),
        "mean_candidate_belief": kind(
            _digest("mean-arm"),
            domain,
            candidate_mean,
            reference_covariance,
        ),
        "covariance_candidate_belief": kind(
            _digest("covariance-arm"),
            domain,
            reference_mean,
            candidate_covariance,
        ),
        "full_belief": kind(
            _digest("full-arm"),
            domain,
            candidate_mean,
            candidate_covariance,
        ),
    }


def _policy(
    beliefs: dict[str, Any],
    *,
    metadata: dict[str, object] | None = None,
) -> BeliefComponentAdmissionPolicyV1:
    return BeliefComponentAdmissionPolicyV1(
        common_domain_id=beliefs["deterministic_reference_belief"].common_domain_id,
        exact_fallback_arm_id=beliefs["exact_fallback_belief"].artifact_id,
        deterministic_reference_arm_id=beliefs[
            "deterministic_reference_belief"
        ].artifact_id,
        mean_candidate_arm_id=beliefs["mean_candidate_belief"].artifact_id,
        covariance_candidate_arm_id=beliefs["covariance_candidate_belief"].artifact_id,
        full_belief_arm_id=beliefs["full_belief"].artifact_id,
        exact_fallback_policy_id=_digest("fallback-policy"),
        reference_covariance_policy_id=_digest("reference-policy"),
        candidate_covariance_policy_id=_digest("candidate-policy"),
        metadata={} if metadata is None else metadata,
    )


def _bind(
    beliefs: dict[str, Any] | None = None,
    *,
    policy: BeliefComponentAdmissionPolicyV1 | None = None,
    metadata: Any = None,
) -> ComponentizedBeliefArmSetV1[Any]:
    values = _beliefs() if beliefs is None else beliefs
    return bind_componentized_belief_arms(
        _policy(values) if policy is None else policy,
        exact_fallback_belief=values["exact_fallback_belief"],
        deterministic_reference_belief=values["deterministic_reference_belief"],
        mean_candidate_belief=values["mean_candidate_belief"],
        covariance_candidate_belief=values["covariance_candidate_belief"],
        full_belief=values["full_belief"],
        metadata={"owner": "source-only"} if metadata is None else metadata,
    )


def _guard(
    policy: BeliefComponentAdmissionPolicyV1,
    accepted: bool,
) -> CompleteBeliefGuardDecisionV1:
    return CompleteBeliefGuardDecisionV1(
        baseline_belief_id=policy.deterministic_reference_arm_id,
        candidate_belief_id=policy.mean_candidate_arm_id,
        common_domain_id=policy.common_domain_id,
        certificate_id=_digest("mean-certificate"),
        inference_admissible=accepted,
        regret_guard_accepted=accepted,
        reason="registered point-mean guard",
    )


def _covariance(
    policy: BeliefComponentAdmissionPolicyV1,
    authorized: bool,
) -> QueryCovarianceTreatmentDecisionV1:
    treatment = COMPLETE_EXPLICIT_JOINT_GAUGE_COVARIANCE
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
        selected_covariance_treatment=treatment,
        principal_covariance_treatment=treatment,
        principal_treatment_matches=True,
        value_certificate_certified=authorized,
        authorized=authorized,
        reasons=(
            ("covariance-treatment-authorized",)
            if authorized
            else ("covariance-value-certificate-rejected",)
        ),
    )


def _decision(
    policy: BeliefComponentAdmissionPolicyV1,
    *,
    mean: bool = True,
    covariance: bool = True,
    common: bool = True,
) -> Any:
    return compose_belief_component_admission(
        policy,
        _guard(policy, mean),
        _covariance(policy, covariance),
        covariance_candidate_admissible=True,
        common_prerequisites_admissible=common,
        reference_supported=True,
    )


def _raw_route(arm_set: ComponentizedBeliefArmSetV1[Any], decision: Any) -> Any:
    return route_belief_component_admission(
        decision,
        exact_fallback_belief=arm_set.exact_fallback_belief,
        deterministic_reference_belief=arm_set.deterministic_reference_belief,
        mean_candidate_belief=arm_set.mean_candidate_belief,
        covariance_candidate_belief=arm_set.covariance_candidate_belief,
        full_belief=arm_set.full_belief,
    )


def test_arm_set_is_content_addressed_and_immutable() -> None:
    arm_set = _bind()

    assert arm_set.artifact_id == content_id(arm_set.descriptor())
    assert arm_set.to_record()["artifact_id"] == arm_set.artifact_id
    assert arm_set.descriptor()["claim_boundary"] == (
        COMPONENTIZED_BELIEF_CLAIM_BOUNDARY
    )
    assert arm_set.descriptor()["semantic_grid"] == {
        "mean_only_retains_reference_covariance": True,
        "covariance_only_retains_reference_mean": True,
        "full_reuses_candidate_mean_and_covariance": True,
    }
    assert arm_set.common_domain_id == arm_set.policy.common_domain_id
    with pytest.raises(TypeError, match="immutable"):
        arm_set.metadata["owner"] = "tampered"  # type: ignore[index]


@pytest.mark.parametrize(
    ("role", "changes", "match"),
    [
        (
            "mean_candidate_belief",
            {"mean_component_id": _digest("reference-mean")},
            "mean candidate must differ",
        ),
        (
            "covariance_candidate_belief",
            {"covariance_component_id": _digest("reference-covariance")},
            "covariance candidate must differ",
        ),
        (
            "mean_candidate_belief",
            {"covariance_component_id": _digest("candidate-covariance")},
            "mean-only arm must retain",
        ),
        (
            "covariance_candidate_belief",
            {"mean_component_id": _digest("candidate-mean")},
            "covariance-only arm must retain",
        ),
        (
            "full_belief",
            {"mean_component_id": _digest("reference-mean")},
            "reuse the candidate mean",
        ),
        (
            "full_belief",
            {"covariance_component_id": _digest("reference-covariance")},
            "reuse the candidate covariance",
        ),
        (
            "full_belief",
            {"common_domain_id": _digest("other-domain")},
            "different common domain",
        ),
    ],
)
def test_semantically_mislabeled_arms_are_rejected(
    role: str,
    changes: dict[str, str],
    match: str,
) -> None:
    beliefs = _beliefs()
    beliefs[role] = replace(beliefs[role], **changes)
    with pytest.raises(ValueError, match=match):
        _bind(beliefs)


def test_policy_interface_digest_and_metadata_mismatches_are_rejected() -> None:
    beliefs = _beliefs()
    policy = _policy(beliefs)
    wrong_policy = replace(
        policy,
        exact_fallback_arm_id=_digest("other-arm"),
        policy_id=None,
    )
    with pytest.raises(ValueError, match="does not match the policy arm"):
        _bind(beliefs, policy=wrong_policy)

    invalid = dict(beliefs)
    invalid["full_belief"] = object()
    with pytest.raises(TypeError, match="must expose artifact_id"):
        _bind(invalid, policy=policy)

    invalid = dict(beliefs)
    invalid["full_belief"] = replace(beliefs["full_belief"], artifact_id="bad")
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        _bind(invalid, policy=policy)
    with pytest.raises(TypeError, match="policy must be"):
        _bind(beliefs, policy=cast(Any, object()))
    with pytest.raises(ValueError, match="metadata must be a mapping"):
        _bind(beliefs, metadata=0)


def test_post_binding_mutation_is_detected() -> None:
    beliefs = _beliefs(mutable=True)
    arm_set = _bind(beliefs)
    beliefs["mean_candidate_belief"].mean_component_id = _digest("mutated")

    with pytest.raises(RuntimeError, match="changed after arm-set binding"):
        arm_set.revalidate_current_bindings()


@pytest.mark.parametrize(
    ("mean", "covariance", "common", "mode", "attribute"),
    [
        (True, True, True, "full-belief", "full_belief"),
        (False, True, True, "covariance-only", "covariance_candidate_belief"),
        (
            True,
            False,
            True,
            "deterministic-reference",
            "deterministic_reference_belief",
        ),
        (True, True, False, "exact-fallback", "exact_fallback_belief"),
    ],
)
def test_semantic_router_returns_the_exact_registered_object(
    mean: bool,
    covariance: bool,
    common: bool,
    mode: str,
    attribute: str,
) -> None:
    arm_set = _bind()
    result = route_componentized_belief_admission(
        _decision(arm_set.policy, mean=mean, covariance=covariance, common=common),
        arm_set,
        metadata={"case": mode},
    )

    assert result.selected_mode == mode
    assert result.selected_belief is getattr(arm_set, attribute)
    assert result.routing.metadata["componentized_belief_arm_set_id"] == (
        arm_set.artifact_id
    )
    assert result.exact_fallback is (mode == "exact-fallback")
    assert result.artifact_id == content_id(result.descriptor())


def test_result_records_selected_component_identities_and_immutable_metadata() -> None:
    arm_set = _bind()
    result = route_componentized_belief_admission(
        _decision(arm_set.policy, mean=False),
        arm_set,
    )
    record = result.to_record()

    assert record["selected_mean_component_id"] == (
        arm_set.deterministic_reference_belief.mean_component_id
    )
    assert record["selected_covariance_component_id"] == (
        arm_set.covariance_candidate_belief.covariance_component_id
    )
    assert result.metadata == {}
    with pytest.raises(TypeError, match="immutable"):
        result.metadata["case"] = "tampered"  # type: ignore[index]


def test_router_and_result_reject_wrong_types_policy_and_objects() -> None:
    arm_set = _bind()
    decision = _decision(arm_set.policy, mean=False, covariance=False)

    with pytest.raises(TypeError, match="decision must be"):
        route_componentized_belief_admission(cast(Any, object()), arm_set)
    with pytest.raises(TypeError, match="arm_set must be"):
        route_componentized_belief_admission(decision, cast(Any, object()))

    routing = _raw_route(arm_set, _decision(arm_set.policy, mean=False))
    with pytest.raises(TypeError, match="arm_set must be"):
        ComponentizedBeliefAdmissionResultV1(cast(Any, object()), routing)
    with pytest.raises(TypeError, match="routing must be"):
        ComponentizedBeliefAdmissionResultV1(arm_set, cast(Any, object()))

    forged = replace(
        routing,
        full_belief=replace(
            arm_set.full_belief,
            artifact_id=arm_set.full_belief.artifact_id,
        ),
    )
    with pytest.raises(ValueError, match="reuse every exact object"):
        ComponentizedBeliefAdmissionResultV1(arm_set, forged)

    other_policy = _policy(_beliefs(), metadata={"other": True})
    other_decision = _decision(other_policy, mean=False)
    with pytest.raises(ValueError, match="different component policy"):
        route_componentized_belief_admission(other_decision, arm_set)
    with pytest.raises(ValueError, match="different component policy"):
        ComponentizedBeliefAdmissionResultV1(
            arm_set,
            _raw_route(arm_set, other_decision),
        )
