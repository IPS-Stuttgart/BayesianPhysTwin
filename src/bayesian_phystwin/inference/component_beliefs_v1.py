"""Semantic binding for component-wise complete-belief routing.

``components_v1`` deliberately routes arbitrary caller-owned beliefs by artifact
identity.  This companion module adds the stronger, optional contract needed to
prove that the registered arm labels match their contents: a covariance-only arm
must retain the deterministic-reference mean, a mean-only arm must retain the
reference covariance, and the full arm must combine both candidate components.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar, cast, runtime_checkable

from .._canonical_contracts import frozen_finite_json_mapping, plain_json
from .._portable_contracts import content_id, sha256_digest
from .components_v1 import (
    BeliefComponentAdmissionDecisionV1,
    BeliefComponentAdmissionPolicyV1,
    BeliefComponentAdmissionResultV1,
    ComponentAdmissionModeV1,
    route_belief_component_admission,
)

COMPONENTIZED_BELIEF_ARM_SET_SCHEMA = "bayesian_phystwin.componentized_belief_arm_set"
COMPONENTIZED_BELIEF_ADMISSION_RESULT_SCHEMA = (
    "bayesian_phystwin.componentized_belief_admission_result"
)
COMPONENTIZED_BELIEF_VERSION = 1
COMPONENTIZED_BELIEF_CLAIM_BOUNDARY = (
    "Component identity, semantic arm composition, and exact-object routing "
    "only. A valid binding does not establish provider competence, calibrated "
    "uncertainty, unseen-object transfer, deployment safety, Causal4D benefit, "
    "or state of the art."
)


@runtime_checkable
class ComponentizedBelief(Protocol):
    """Complete belief exposing immutable identities for its routed components."""

    @property
    def artifact_id(self) -> str: ...

    @property
    def common_domain_id(self) -> str: ...

    @property
    def mean_component_id(self) -> str: ...

    @property
    def covariance_component_id(self) -> str: ...


BeliefT = TypeVar("BeliefT", bound=ComponentizedBelief)

_ARM_ATTRIBUTE_BY_ROLE = {
    "exact-fallback": "exact_fallback_belief",
    "deterministic-reference": "deterministic_reference_belief",
    "mean-only": "mean_candidate_belief",
    "covariance-only": "covariance_candidate_belief",
    "full-belief": "full_belief",
}
_POLICY_ID_BY_ROLE = {
    "exact-fallback": "exact_fallback_arm_id",
    "deterministic-reference": "deterministic_reference_arm_id",
    "mean-only": "mean_candidate_arm_id",
    "covariance-only": "covariance_candidate_arm_id",
    "full-belief": "full_belief_arm_id",
}


def _belief_record(value: object, *, role: str) -> dict[str, str]:
    if not isinstance(value, ComponentizedBelief):
        raise TypeError(
            f"{role} belief must expose artifact_id, common_domain_id, "
            "mean_component_id, and covariance_component_id"
        )
    return {
        "belief_id": sha256_digest(
            value.artifact_id,
            name=f"{role} belief artifact_id",
        ),
        "common_domain_id": sha256_digest(
            value.common_domain_id,
            name=f"{role} belief common_domain_id",
        ),
        "mean_component_id": sha256_digest(
            value.mean_component_id,
            name=f"{role} belief mean_component_id",
        ),
        "covariance_component_id": sha256_digest(
            value.covariance_component_id,
            name=f"{role} belief covariance_component_id",
        ),
    }


def _arm_records(
    arm_set: ComponentizedBeliefArmSetV1[Any],
) -> dict[str, dict[str, str]]:
    return {
        role: _belief_record(
            getattr(arm_set, attribute),
            role=role,
        )
        for role, attribute in _ARM_ATTRIBUTE_BY_ROLE.items()
    }


def _validate_arm_records(
    policy: BeliefComponentAdmissionPolicyV1,
    records: Mapping[str, Mapping[str, str]],
) -> None:
    for role, policy_attribute in _POLICY_ID_BY_ROLE.items():
        expected_id = sha256_digest(
            getattr(policy, policy_attribute),
            name=f"policy {policy_attribute}",
        )
        if records[role]["belief_id"] != expected_id:
            raise ValueError(f"{role} belief does not match the policy arm")
        if records[role]["common_domain_id"] != policy.common_domain_id:
            raise ValueError(f"{role} belief binds a different common domain")

    reference = records["deterministic-reference"]
    mean_only = records["mean-only"]
    covariance_only = records["covariance-only"]
    full = records["full-belief"]

    if mean_only["mean_component_id"] == reference["mean_component_id"]:
        raise ValueError(
            "mean candidate must differ from the deterministic-reference mean"
        )
    if (
        covariance_only["covariance_component_id"]
        == reference["covariance_component_id"]
    ):
        raise ValueError(
            "covariance candidate must differ from the reference covariance"
        )
    if mean_only["covariance_component_id"] != reference["covariance_component_id"]:
        raise ValueError("mean-only arm must retain the reference covariance")
    if covariance_only["mean_component_id"] != reference["mean_component_id"]:
        raise ValueError(
            "covariance-only arm must retain the deterministic-reference mean"
        )
    if full["mean_component_id"] != mean_only["mean_component_id"]:
        raise ValueError("full-belief arm must reuse the candidate mean")
    if full["covariance_component_id"] != covariance_only["covariance_component_id"]:
        raise ValueError("full-belief arm must reuse the candidate covariance")


@dataclass(frozen=True, slots=True)
class ComponentizedBeliefArmSetV1(Generic[BeliefT]):
    """Content-addressed semantic binding for the five registered belief arms."""

    policy: BeliefComponentAdmissionPolicyV1
    exact_fallback_belief: BeliefT
    deterministic_reference_belief: BeliefT
    mean_candidate_belief: BeliefT
    covariance_candidate_belief: BeliefT
    full_belief: BeliefT
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str = field(init=False)
    _bound_arm_records: Mapping[str, Any] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.policy, BeliefComponentAdmissionPolicyV1):
            raise TypeError("policy must be a BeliefComponentAdmissionPolicyV1")
        records = _arm_records(cast(ComponentizedBeliefArmSetV1[Any], self))
        _validate_arm_records(self.policy, records)
        frozen_records = frozen_finite_json_mapping(
            records,
            name="componentized belief arm records",
        )
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="componentized belief arm-set metadata",
        )
        object.__setattr__(self, "_bound_arm_records", frozen_records)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "artifact_id", content_id(self.descriptor()))

    @property
    def common_domain_id(self) -> str:
        return self.policy.common_domain_id

    def revalidate_current_bindings(self) -> None:
        """Fail if a caller-owned belief changed after the arm set was bound."""

        current = _arm_records(cast(ComponentizedBeliefArmSetV1[Any], self))
        if current != plain_json(self._bound_arm_records):
            raise RuntimeError(
                "componentized belief identities changed after arm-set binding"
            )
        _validate_arm_records(self.policy, current)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": COMPONENTIZED_BELIEF_ARM_SET_SCHEMA,
            "schema_version": COMPONENTIZED_BELIEF_VERSION,
            "policy_id": sha256_digest(self.policy.policy_id, name="policy_id"),
            "common_domain_id": self.common_domain_id,
            "arms": plain_json(self._bound_arm_records),
            "semantic_grid": {
                "mean_only_retains_reference_covariance": True,
                "covariance_only_retains_reference_mean": True,
                "full_reuses_candidate_mean_and_covariance": True,
            },
            "claim_boundary": COMPONENTIZED_BELIEF_CLAIM_BOUNDARY,
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {"artifact_id": self.artifact_id, **self.descriptor()}


@dataclass(frozen=True, slots=True)
class ComponentizedBeliefAdmissionResultV1(Generic[BeliefT]):
    """Exact routed belief bound to a semantically validated arm set."""

    arm_set: ComponentizedBeliefArmSetV1[BeliefT]
    routing: BeliefComponentAdmissionResultV1[BeliefT]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.arm_set, ComponentizedBeliefArmSetV1):
            raise TypeError("arm_set must be a ComponentizedBeliefArmSetV1")
        if not isinstance(self.routing, BeliefComponentAdmissionResultV1):
            raise TypeError("routing must be a BeliefComponentAdmissionResultV1")
        self.arm_set.revalidate_current_bindings()
        if self.routing.decision.policy.policy_id != self.arm_set.policy.policy_id:
            raise ValueError("routing decision binds a different component policy")
        for attribute in _ARM_ATTRIBUTE_BY_ROLE.values():
            if getattr(self.routing, attribute) is not getattr(self.arm_set, attribute):
                raise ValueError(
                    "routing must reuse every exact object from the validated arm set"
                )
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="componentized belief routing metadata",
        )
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "artifact_id", content_id(self.descriptor()))

    @property
    def selected_belief(self) -> BeliefT:
        return self.routing.selected_belief

    @property
    def selected_mode(self) -> ComponentAdmissionModeV1:
        return self.routing.decision.selected_mode

    @property
    def selected_mean_component_id(self) -> str:
        return sha256_digest(
            self.selected_belief.mean_component_id,
            name="selected belief mean_component_id",
        )

    @property
    def selected_covariance_component_id(self) -> str:
        return sha256_digest(
            self.selected_belief.covariance_component_id,
            name="selected belief covariance_component_id",
        )

    @property
    def exact_fallback(self) -> bool:
        return self.routing.exact_fallback

    def descriptor(self) -> dict[str, object]:
        self.arm_set.revalidate_current_bindings()
        return {
            "schema": COMPONENTIZED_BELIEF_ADMISSION_RESULT_SCHEMA,
            "schema_version": COMPONENTIZED_BELIEF_VERSION,
            "arm_set_id": self.arm_set.artifact_id,
            "component_admission_result_id": self.routing.artifact_id,
            "decision_id": self.routing.decision.artifact_id,
            "policy_id": self.arm_set.policy.policy_id,
            "selected_mode": self.selected_mode,
            "selected_belief_id": self.selected_belief.artifact_id,
            "selected_mean_component_id": self.selected_mean_component_id,
            "selected_covariance_component_id": (self.selected_covariance_component_id),
            "exact_fallback": self.exact_fallback,
            "claim_boundary": COMPONENTIZED_BELIEF_CLAIM_BOUNDARY,
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {"artifact_id": self.artifact_id, **self.descriptor()}


def bind_componentized_belief_arms(
    policy: BeliefComponentAdmissionPolicyV1,
    *,
    exact_fallback_belief: BeliefT,
    deterministic_reference_belief: BeliefT,
    mean_candidate_belief: BeliefT,
    covariance_candidate_belief: BeliefT,
    full_belief: BeliefT,
    metadata: Mapping[str, Any] | None = None,
) -> ComponentizedBeliefArmSetV1[BeliefT]:
    """Bind five caller-owned beliefs and prove the registered component grid."""

    return ComponentizedBeliefArmSetV1(
        policy=policy,
        exact_fallback_belief=exact_fallback_belief,
        deterministic_reference_belief=deterministic_reference_belief,
        mean_candidate_belief=mean_candidate_belief,
        covariance_candidate_belief=covariance_candidate_belief,
        full_belief=full_belief,
        metadata={} if metadata is None else metadata,
    )


def route_componentized_belief_admission(
    decision: BeliefComponentAdmissionDecisionV1,
    arm_set: ComponentizedBeliefArmSetV1[BeliefT],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> ComponentizedBeliefAdmissionResultV1[BeliefT]:
    """Route through ``components_v1`` after revalidating component semantics."""

    if not isinstance(decision, BeliefComponentAdmissionDecisionV1):
        raise TypeError("decision must be a BeliefComponentAdmissionDecisionV1")
    if not isinstance(arm_set, ComponentizedBeliefArmSetV1):
        raise TypeError("arm_set must be a ComponentizedBeliefArmSetV1")
    arm_set.revalidate_current_bindings()
    if decision.policy.policy_id != arm_set.policy.policy_id:
        raise ValueError("decision binds a different component policy")
    caller_metadata = frozen_finite_json_mapping(
        {} if metadata is None else metadata,
        name="componentized belief routing metadata",
    )
    routing = route_belief_component_admission(
        decision,
        exact_fallback_belief=arm_set.exact_fallback_belief,
        deterministic_reference_belief=arm_set.deterministic_reference_belief,
        mean_candidate_belief=arm_set.mean_candidate_belief,
        covariance_candidate_belief=arm_set.covariance_candidate_belief,
        full_belief=arm_set.full_belief,
        metadata={
            "componentized_belief_arm_set_id": arm_set.artifact_id,
            "caller": plain_json(caller_metadata),
        },
    )
    return ComponentizedBeliefAdmissionResultV1(
        arm_set=arm_set,
        routing=routing,
        metadata=caller_metadata,
    )


__all__ = [
    "COMPONENTIZED_BELIEF_ADMISSION_RESULT_SCHEMA",
    "COMPONENTIZED_BELIEF_ARM_SET_SCHEMA",
    "COMPONENTIZED_BELIEF_CLAIM_BOUNDARY",
    "COMPONENTIZED_BELIEF_VERSION",
    "ComponentizedBelief",
    "ComponentizedBeliefAdmissionResultV1",
    "ComponentizedBeliefArmSetV1",
    "bind_componentized_belief_arms",
    "route_componentized_belief_admission",
]
