"""Prospective protocol for cross-action broken-mechanism controls."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from bayesian_phystwin._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    genuine_integer,
    plain_json,
)
from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin_experiments._cross_action_physicality_common_v1 import (
    CROSS_ACTION_PHYSICALITY_CLAIM_BOUNDARY,
    CROSS_ACTION_PHYSICALITY_SCHEMA,
    CROSS_ACTION_PHYSICALITY_SEMANTICS,
    CROSS_ACTION_PHYSICALITY_VERSION,
    FAMILYWISE_METHOD,
    FAMILYWISE_METHOD_ID,
    REQUIRED_PLACEBO_POLICIES,
    BrokenMechanismPolicy,
    _commit,
    _digest,
    _finite,
    _probability,
)
from bayesian_phystwin_experiments.cross_action_transport_contracts_v1 import (
    TransportArm,
)
from bayesian_phystwin_experiments.cross_action_transport_v2 import (
    CrossActionProtocolV2,
)


@dataclass(frozen=True, slots=True)
class CrossActionPhysicalityProtocolV1:
    """Target-closed protocol for four broken-mechanism controls."""

    parent_protocol: CrossActionProtocolV2
    wrong_source_action_policy_id: str
    wrong_object_session_policy_id: str
    phase_shifted_source_policy_id: str
    identity_permuted_policy_id: str
    prediction_batch_id: str
    commit_id: str
    scorer_id: str
    minimum_sessions: int
    bootstrap_replicates: int
    bootstrap_seed: int
    confidence_level: float
    minimum_placebo_separation_margin: float
    lower_is_better: bool = True
    method_frozen_before_target: bool = True
    roster_frozen_before_target: bool = True
    target_outcomes_used: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.parent_protocol, CrossActionProtocolV2):
            raise TypeError("parent_protocol must be a CrossActionProtocolV2")
        if (
            self.parent_protocol.physical_transport_arm
            is not TransportArm.GUARDED_PHYSICAL
        ):
            raise ValueError(
                "physicality v1 requires guarded_physical as the parent arm"
            )
        if not self.parent_protocol.lower_is_better:
            raise ValueError("physicality v1 requires a lower-is-better parent score")

        policy_id_names = (
            "wrong_source_action_policy_id",
            "wrong_object_session_policy_id",
            "phase_shifted_source_policy_id",
            "identity_permuted_policy_id",
        )
        for name in (*policy_id_names, "prediction_batch_id", "scorer_id"):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        if len({getattr(self, name) for name in policy_id_names}) != len(
            policy_id_names
        ):
            raise ValueError("each placebo policy must have a distinct implementation")
        object.__setattr__(self, "commit_id", _commit(self.commit_id, name="commit_id"))

        source_action_counts: dict[str, int] = {}
        for pair in self.parent_protocol.session_pairs:
            source_action_counts[pair.source_action_id] = (
                source_action_counts.get(pair.source_action_id, 0) + 1
            )
        if len(source_action_counts) < 2:
            raise ValueError(
                "wrong_source_action requires at least two source action profiles"
            )
        if any(count < 2 for count in source_action_counts.values()):
            raise ValueError(
                "wrong_object_session requires two sessions per source action profile"
            )

        minimum_sessions = genuine_integer(
            self.minimum_sessions, name="minimum_sessions", minimum=2
        )
        if minimum_sessions > len(self.parent_protocol.session_pairs):
            raise ValueError("minimum_sessions cannot exceed the parent session roster")
        object.__setattr__(self, "minimum_sessions", minimum_sessions)
        object.__setattr__(
            self,
            "bootstrap_replicates",
            genuine_integer(
                self.bootstrap_replicates,
                name="bootstrap_replicates",
                minimum=100,
            ),
        )
        object.__setattr__(
            self,
            "bootstrap_seed",
            genuine_integer(self.bootstrap_seed, name="bootstrap_seed", minimum=0),
        )
        object.__setattr__(
            self,
            "confidence_level",
            _probability(self.confidence_level, name="confidence_level"),
        )
        object.__setattr__(
            self,
            "minimum_placebo_separation_margin",
            _finite(
                self.minimum_placebo_separation_margin,
                name="minimum_placebo_separation_margin",
                minimum=0.0,
            ),
        )

        lower_is_better = genuine_boolean(self.lower_is_better, name="lower_is_better")
        frozen = genuine_boolean(
            self.method_frozen_before_target, name="method_frozen_before_target"
        )
        roster_frozen = genuine_boolean(
            self.roster_frozen_before_target, name="roster_frozen_before_target"
        )
        target_used = genuine_boolean(
            self.target_outcomes_used, name="target_outcomes_used"
        )
        if not lower_is_better:
            raise ValueError("physicality v1 requires a lower-is-better score")
        if not frozen or not roster_frozen or target_used:
            raise ValueError(
                "physicality protocol must be frozen and target-outcome free"
            )
        object.__setattr__(self, "lower_is_better", lower_is_better)
        object.__setattr__(self, "method_frozen_before_target", frozen)
        object.__setattr__(self, "roster_frozen_before_target", roster_frozen)
        object.__setattr__(self, "target_outcomes_used", target_used)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="physicality metadata"),
        )

    def policy_implementation_id(self, policy: BrokenMechanismPolicy) -> str:
        if not isinstance(policy, BrokenMechanismPolicy):
            raise TypeError("policy must be a BrokenMechanismPolicy")
        return {
            BrokenMechanismPolicy.WRONG_SOURCE_ACTION: (
                self.wrong_source_action_policy_id
            ),
            BrokenMechanismPolicy.WRONG_OBJECT_SESSION: (
                self.wrong_object_session_policy_id
            ),
            BrokenMechanismPolicy.PHASE_SHIFTED_SOURCE: (
                self.phase_shifted_source_policy_id
            ),
            BrokenMechanismPolicy.IDENTITY_PERMUTED: (
                self.identity_permuted_policy_id
            ),
        }[policy]

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": CROSS_ACTION_PHYSICALITY_SCHEMA,
            "schema_version": CROSS_ACTION_PHYSICALITY_VERSION,
            "artifact_kind": "CrossActionPhysicalityProtocolV1",
            "semantics": CROSS_ACTION_PHYSICALITY_SEMANTICS,
            "parent_protocol_id": self.parent_protocol.protocol_id,
            "required_placebo_policies": [
                policy.value for policy in REQUIRED_PLACEBO_POLICIES
            ],
            "policy_implementation_ids": {
                policy.value: self.policy_implementation_id(policy)
                for policy in REQUIRED_PLACEBO_POLICIES
            },
            "prediction_batch_id": self.prediction_batch_id,
            "commit_id": self.commit_id,
            "scorer_id": self.scorer_id,
            "minimum_sessions": self.minimum_sessions,
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_seed": self.bootstrap_seed,
            "confidence_level": self.confidence_level,
            "familywise_method": FAMILYWISE_METHOD,
            "familywise_method_id": FAMILYWISE_METHOD_ID,
            "minimum_placebo_separation_margin": (
                self.minimum_placebo_separation_margin
            ),
            "lower_is_better": self.lower_is_better,
            "method_frozen_before_target": self.method_frozen_before_target,
            "roster_frozen_before_target": self.roster_frozen_before_target,
            "target_outcomes_used": self.target_outcomes_used,
            "metadata": plain_json(self.metadata),
            "claim_boundary": CROSS_ACTION_PHYSICALITY_CLAIM_BOUNDARY,
        }

    @property
    def protocol_id(self) -> str:
        return cast(str, content_id(self.descriptor()))
