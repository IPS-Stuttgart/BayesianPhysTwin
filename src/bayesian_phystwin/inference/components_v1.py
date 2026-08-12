"""Separate point-mean and covariance admission for guarded beliefs.

The module composes already-frozen decisions. It does not estimate either
component, choose thresholds, or authorize a scientific claim. Routing always
reuses one exact caller-owned belief object.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar, cast

from .._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    genuine_integer,
    plain_json,
)
from .._portable_contracts import (
    content_id,
    load_strict_json_object,
    require_exact_fields,
    sha256_digest,
    write_atomic_json,
)
from ..complete_belief_selection import (
    ArtifactBelief,
    CompleteBeliefGuardDecisionV1,
)
from ..query_covariance_decision_v1 import (
    QueryCovarianceTreatmentDecisionV1,
)

BELIEF_COMPONENT_ADMISSION_POLICY_SCHEMA = (
    "bayesian_phystwin.belief_component_admission_policy"
)
BELIEF_COMPONENT_ADMISSION_DECISION_SCHEMA = (
    "bayesian_phystwin.belief_component_admission_decision"
)
BELIEF_COMPONENT_ADMISSION_RESULT_SCHEMA = (
    "bayesian_phystwin.belief_component_admission_result"
)
BELIEF_COMPONENT_ADMISSION_VERSION = 1
BELIEF_COMPONENT_ADMISSION_CLAIM_BOUNDARY = (
    "Software evidence composition and exact-object routing only. A selected "
    "arm does not establish provider competence, uncertainty calibration, "
    "unseen-object transfer, deployment safety, Causal4D benefit, or state of "
    "the art."
)

ComponentAdmissionModeV1 = Literal[
    "exact-fallback",
    "deterministic-reference",
    "covariance-only",
    "mean-only",
    "full-belief",
]
BeliefT = TypeVar("BeliefT", bound=ArtifactBelief)

_MODES = frozenset(
    {
        "exact-fallback",
        "deterministic-reference",
        "covariance-only",
        "mean-only",
        "full-belief",
    }
)
_POLICY_FIELDS = frozenset(
    {
        "policy_id",
        "schema",
        "schema_version",
        "common_domain_id",
        "exact_fallback_arm_id",
        "deterministic_reference_arm_id",
        "mean_candidate_arm_id",
        "covariance_candidate_arm_id",
        "full_belief_arm_id",
        "exact_fallback_policy_id",
        "reference_covariance_policy_id",
        "candidate_covariance_policy_id",
        "allow_covariance_only",
        "allow_mean_only",
        "claim_boundary",
        "metadata",
    }
)
_DECISION_FIELDS = frozenset(
    {
        "artifact_id",
        "schema",
        "schema_version",
        "policy",
        "mean_guard_decision_id",
        "covariance_decision_id",
        "mean_inference_admissible",
        "mean_regret_guard_accepted",
        "covariance_treatment_authorized",
        "covariance_candidate_admissible",
        "common_prerequisites_admissible",
        "reference_supported",
        "mean_authorized",
        "covariance_authorized",
        "selected_mode",
        "selected_arm_id",
        "exact_fallback",
        "reasons",
        "claim_boundary",
        "metadata",
    }
)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{name} must use literal string keys")
    return cast(Mapping[str, Any], value)


def _version(value: object, *, name: str) -> int:
    result = genuine_integer(value, name=name, minimum=1)
    if result != BELIEF_COMPONENT_ADMISSION_VERSION:
        raise ValueError(f"{name} changed")
    return result


def _mode(value: object) -> ComponentAdmissionModeV1:
    if type(value) is not str or value not in _MODES:
        raise ValueError("unsupported component-admission mode")
    return cast(ComponentAdmissionModeV1, value)


def _reasons(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("reasons must be a JSON array")
    raw = tuple(value)
    if any(type(reason) is not str or not reason for reason in raw):
        raise ValueError("reasons must contain nonempty strings")
    result = tuple(sorted(cast(tuple[str, ...], raw)))
    if len(result) != len(set(result)):
        raise ValueError("reasons must not contain duplicates")
    return result


def _belief_id(value: object, *, name: str) -> str:
    try:
        artifact_id = cast(ArtifactBelief, value).artifact_id
    except AttributeError as error:
        raise TypeError(f"{name} must expose artifact_id") from error
    return sha256_digest(artifact_id, name=f"{name}.artifact_id")


@dataclass(frozen=True, slots=True)
class BeliefComponentAdmissionPolicyV1:
    """Frozen arm identities and permissions for component-wise admission."""

    common_domain_id: str
    exact_fallback_arm_id: str
    deterministic_reference_arm_id: str
    mean_candidate_arm_id: str
    covariance_candidate_arm_id: str
    full_belief_arm_id: str
    exact_fallback_policy_id: str
    reference_covariance_policy_id: str
    candidate_covariance_policy_id: str
    allow_covariance_only: bool = True
    allow_mean_only: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    policy_id: str | None = None

    def __post_init__(self) -> None:
        digest_fields = (
            "common_domain_id",
            "exact_fallback_arm_id",
            "deterministic_reference_arm_id",
            "mean_candidate_arm_id",
            "covariance_candidate_arm_id",
            "full_belief_arm_id",
            "exact_fallback_policy_id",
            "reference_covariance_policy_id",
            "candidate_covariance_policy_id",
        )
        for name in digest_fields:
            object.__setattr__(
                self,
                name,
                sha256_digest(getattr(self, name), name=name),
            )
        arm_ids = (
            self.exact_fallback_arm_id,
            self.deterministic_reference_arm_id,
            self.mean_candidate_arm_id,
            self.covariance_candidate_arm_id,
            self.full_belief_arm_id,
        )
        if len(set(arm_ids)) != len(arm_ids):
            raise ValueError("component-admission arm identities must be distinct")
        if self.reference_covariance_policy_id == self.candidate_covariance_policy_id:
            raise ValueError("reference and candidate covariance policies must differ")
        object.__setattr__(
            self,
            "allow_covariance_only",
            genuine_boolean(
                self.allow_covariance_only,
                name="allow_covariance_only",
            ),
        )
        object.__setattr__(
            self,
            "allow_mean_only",
            genuine_boolean(self.allow_mean_only, name="allow_mean_only"),
        )
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="component-admission policy metadata",
            ),
        )
        expected = content_id(self.descriptor())
        if self.policy_id is not None:
            supplied = sha256_digest(self.policy_id, name="policy_id")
            if supplied != expected:
                raise ValueError("policy_id does not match policy content")
        object.__setattr__(self, "policy_id", expected)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": BELIEF_COMPONENT_ADMISSION_POLICY_SCHEMA,
            "schema_version": BELIEF_COMPONENT_ADMISSION_VERSION,
            "common_domain_id": self.common_domain_id,
            "exact_fallback_arm_id": self.exact_fallback_arm_id,
            "deterministic_reference_arm_id": self.deterministic_reference_arm_id,
            "mean_candidate_arm_id": self.mean_candidate_arm_id,
            "covariance_candidate_arm_id": self.covariance_candidate_arm_id,
            "full_belief_arm_id": self.full_belief_arm_id,
            "exact_fallback_policy_id": self.exact_fallback_policy_id,
            "reference_covariance_policy_id": self.reference_covariance_policy_id,
            "candidate_covariance_policy_id": self.candidate_covariance_policy_id,
            "allow_covariance_only": self.allow_covariance_only,
            "allow_mean_only": self.allow_mean_only,
            "claim_boundary": BELIEF_COMPONENT_ADMISSION_CLAIM_BOUNDARY,
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {"policy_id": self.policy_id, **self.descriptor()}

    @classmethod
    def from_mapping(cls, value: object) -> BeliefComponentAdmissionPolicyV1:
        source = _mapping(value, name="component-admission policy")
        require_exact_fields(
            source,
            expected=_POLICY_FIELDS,
            name="component-admission policy",
        )
        if source["schema"] != BELIEF_COMPONENT_ADMISSION_POLICY_SCHEMA:
            raise ValueError("component-admission policy schema changed")
        _version(
            source["schema_version"],
            name="component-admission policy version",
        )
        if source["claim_boundary"] != BELIEF_COMPONENT_ADMISSION_CLAIM_BOUNDARY:
            raise ValueError("component-admission policy claim boundary changed")
        return cls(
            common_domain_id=cast(str, source["common_domain_id"]),
            exact_fallback_arm_id=cast(str, source["exact_fallback_arm_id"]),
            deterministic_reference_arm_id=cast(
                str,
                source["deterministic_reference_arm_id"],
            ),
            mean_candidate_arm_id=cast(str, source["mean_candidate_arm_id"]),
            covariance_candidate_arm_id=cast(
                str,
                source["covariance_candidate_arm_id"],
            ),
            full_belief_arm_id=cast(str, source["full_belief_arm_id"]),
            exact_fallback_policy_id=cast(
                str,
                source["exact_fallback_policy_id"],
            ),
            reference_covariance_policy_id=cast(
                str,
                source["reference_covariance_policy_id"],
            ),
            candidate_covariance_policy_id=cast(
                str,
                source["candidate_covariance_policy_id"],
            ),
            allow_covariance_only=cast(bool, source["allow_covariance_only"]),
            allow_mean_only=cast(bool, source["allow_mean_only"]),
            metadata=_mapping(source["metadata"], name="metadata"),
            policy_id=cast(str, source["policy_id"]),
        )


def _expected_outcome(
    policy: BeliefComponentAdmissionPolicyV1,
    *,
    mean_inference_admissible: bool,
    mean_regret_guard_accepted: bool,
    covariance_treatment_authorized: bool,
    covariance_candidate_admissible: bool,
    common_prerequisites_admissible: bool,
    reference_supported: bool,
) -> tuple[ComponentAdmissionModeV1, str, tuple[str, ...]]:
    if not common_prerequisites_admissible:
        return (
            "exact-fallback",
            policy.exact_fallback_arm_id,
            ("common-prerequisites-rejected", "selected-exact-fallback"),
        )
    if not reference_supported:
        return (
            "exact-fallback",
            policy.exact_fallback_arm_id,
            ("deterministic-reference-unsupported", "selected-exact-fallback"),
        )
    mean = mean_inference_admissible and mean_regret_guard_accepted
    covariance = (
        covariance_treatment_authorized and covariance_candidate_admissible
    )
    reasons = [
        "mean-update-authorized" if mean else "mean-update-rejected",
        (
            "covariance-update-authorized"
            if covariance
            else "covariance-update-rejected"
        ),
    ]
    if mean and covariance:
        mode: ComponentAdmissionModeV1 = "full-belief"
        arm_id = policy.full_belief_arm_id
        reasons.append("selected-full-belief")
    elif covariance and policy.allow_covariance_only:
        mode = "covariance-only"
        arm_id = policy.covariance_candidate_arm_id
        reasons.append("selected-covariance-only")
    elif mean and policy.allow_mean_only:
        mode = "mean-only"
        arm_id = policy.mean_candidate_arm_id
        reasons.append("selected-mean-only")
    else:
        mode = "deterministic-reference"
        arm_id = policy.deterministic_reference_arm_id
        if covariance and not policy.allow_covariance_only:
            reasons.append("covariance-only-disallowed")
        if mean and not policy.allow_mean_only:
            reasons.append("mean-only-disallowed")
        reasons.append("selected-deterministic-reference")
    return mode, arm_id, tuple(sorted(reasons))


@dataclass(frozen=True, slots=True)
class BeliefComponentAdmissionDecisionV1:
    """Content-addressed component-wise admission decision."""

    policy: BeliefComponentAdmissionPolicyV1
    mean_guard_decision_id: str
    covariance_decision_id: str
    mean_inference_admissible: bool
    mean_regret_guard_accepted: bool
    covariance_treatment_authorized: bool
    covariance_candidate_admissible: bool
    common_prerequisites_admissible: bool
    reference_supported: bool
    selected_mode: ComponentAdmissionModeV1
    selected_arm_id: str
    reasons: Sequence[str]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.policy, BeliefComponentAdmissionPolicyV1):
            raise TypeError("policy must be a BeliefComponentAdmissionPolicyV1")
        for name in ("mean_guard_decision_id", "covariance_decision_id"):
            object.__setattr__(
                self,
                name,
                sha256_digest(getattr(self, name), name=name),
            )
        for name in (
            "mean_inference_admissible",
            "mean_regret_guard_accepted",
            "covariance_treatment_authorized",
            "covariance_candidate_admissible",
            "common_prerequisites_admissible",
            "reference_supported",
        ):
            object.__setattr__(
                self,
                name,
                genuine_boolean(getattr(self, name), name=name),
            )
        if self.mean_regret_guard_accepted and not self.mean_inference_admissible:
            raise ValueError(
                "mean_regret_guard_accepted requires mean inference admission"
            )
        mode = _mode(self.selected_mode)
        arm_id = sha256_digest(self.selected_arm_id, name="selected_arm_id")
        reasons = _reasons(self.reasons)
        expected = _expected_outcome(
            self.policy,
            mean_inference_admissible=self.mean_inference_admissible,
            mean_regret_guard_accepted=self.mean_regret_guard_accepted,
            covariance_treatment_authorized=self.covariance_treatment_authorized,
            covariance_candidate_admissible=self.covariance_candidate_admissible,
            common_prerequisites_admissible=(
                self.common_prerequisites_admissible
            ),
            reference_supported=self.reference_supported,
        )
        if (mode, arm_id) != expected[:2]:
            raise ValueError("selected component arm contradicts admission gates")
        if reasons != expected[2]:
            raise ValueError("reasons do not match component-admission gates")
        object.__setattr__(self, "selected_mode", mode)
        object.__setattr__(self, "selected_arm_id", arm_id)
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="component-admission decision metadata",
            ),
        )
        expected_id = content_id(self.descriptor())
        if self.artifact_id is not None:
            supplied = sha256_digest(self.artifact_id, name="artifact_id")
            if supplied != expected_id:
                raise ValueError("artifact_id does not match decision content")
        object.__setattr__(self, "artifact_id", expected_id)

    @property
    def mean_authorized(self) -> bool:
        return self.mean_inference_admissible and self.mean_regret_guard_accepted

    @property
    def covariance_authorized(self) -> bool:
        return (
            self.covariance_treatment_authorized
            and self.covariance_candidate_admissible
        )

    @property
    def exact_fallback(self) -> bool:
        return self.selected_mode == "exact-fallback"

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": BELIEF_COMPONENT_ADMISSION_DECISION_SCHEMA,
            "schema_version": BELIEF_COMPONENT_ADMISSION_VERSION,
            "policy": self.policy.to_record(),
            "mean_guard_decision_id": self.mean_guard_decision_id,
            "covariance_decision_id": self.covariance_decision_id,
            "mean_inference_admissible": self.mean_inference_admissible,
            "mean_regret_guard_accepted": self.mean_regret_guard_accepted,
            "covariance_treatment_authorized": (
                self.covariance_treatment_authorized
            ),
            "covariance_candidate_admissible": (
                self.covariance_candidate_admissible
            ),
            "common_prerequisites_admissible": (
                self.common_prerequisites_admissible
            ),
            "reference_supported": self.reference_supported,
            "mean_authorized": self.mean_authorized,
            "covariance_authorized": self.covariance_authorized,
            "selected_mode": self.selected_mode,
            "selected_arm_id": self.selected_arm_id,
            "exact_fallback": self.exact_fallback,
            "reasons": list(self.reasons),
            "claim_boundary": BELIEF_COMPONENT_ADMISSION_CLAIM_BOUNDARY,
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {"artifact_id": self.artifact_id, **self.descriptor()}

    @classmethod
    def from_mapping(cls, value: object) -> BeliefComponentAdmissionDecisionV1:
        source = _mapping(value, name="component-admission decision")
        require_exact_fields(
            source,
            expected=_DECISION_FIELDS,
            name="component-admission decision",
        )
        if source["schema"] != BELIEF_COMPONENT_ADMISSION_DECISION_SCHEMA:
            raise ValueError("component-admission decision schema changed")
        _version(
            source["schema_version"],
            name="component-admission decision version",
        )
        if source["claim_boundary"] != BELIEF_COMPONENT_ADMISSION_CLAIM_BOUNDARY:
            raise ValueError("component-admission decision claim boundary changed")
        result = cls(
            policy=BeliefComponentAdmissionPolicyV1.from_mapping(source["policy"]),
            mean_guard_decision_id=cast(
                str,
                source["mean_guard_decision_id"],
            ),
            covariance_decision_id=cast(str, source["covariance_decision_id"]),
            mean_inference_admissible=cast(
                bool,
                source["mean_inference_admissible"],
            ),
            mean_regret_guard_accepted=cast(
                bool,
                source["mean_regret_guard_accepted"],
            ),
            covariance_treatment_authorized=cast(
                bool,
                source["covariance_treatment_authorized"],
            ),
            covariance_candidate_admissible=cast(
                bool,
                source["covariance_candidate_admissible"],
            ),
            common_prerequisites_admissible=cast(
                bool,
                source["common_prerequisites_admissible"],
            ),
            reference_supported=cast(bool, source["reference_supported"]),
            selected_mode=cast(
                ComponentAdmissionModeV1,
                source["selected_mode"],
            ),
            selected_arm_id=cast(str, source["selected_arm_id"]),
            reasons=_reasons(source["reasons"]),
            metadata=_mapping(source["metadata"], name="metadata"),
            artifact_id=cast(str, source["artifact_id"]),
        )
        for field_name, expected in (
            ("mean_authorized", result.mean_authorized),
            ("covariance_authorized", result.covariance_authorized),
            ("exact_fallback", result.exact_fallback),
        ):
            supplied = genuine_boolean(source[field_name], name=field_name)
            if supplied != expected:
                raise ValueError(f"{field_name} contradicts decision gates")
        return result


def compose_belief_component_admission(
    policy: BeliefComponentAdmissionPolicyV1,
    mean_guard_decision: CompleteBeliefGuardDecisionV1,
    covariance_decision: QueryCovarianceTreatmentDecisionV1,
    *,
    covariance_candidate_admissible: bool,
    common_prerequisites_admissible: bool,
    reference_supported: bool,
    metadata: Mapping[str, Any] | None = None,
) -> BeliefComponentAdmissionDecisionV1:
    """Compose independent point and covariance decisions without cross-rescue."""

    if not isinstance(policy, BeliefComponentAdmissionPolicyV1):
        raise TypeError("policy must be a BeliefComponentAdmissionPolicyV1")
    if not isinstance(mean_guard_decision, CompleteBeliefGuardDecisionV1):
        raise TypeError(
            "mean_guard_decision must be a CompleteBeliefGuardDecisionV1"
        )
    if not isinstance(
        covariance_decision,
        QueryCovarianceTreatmentDecisionV1,
    ):
        raise TypeError(
            "covariance_decision must be a QueryCovarianceTreatmentDecisionV1"
        )
    if mean_guard_decision.common_domain_id != policy.common_domain_id:
        raise ValueError("mean guard binds a different common domain")
    if (
        mean_guard_decision.baseline_belief_id
        != policy.deterministic_reference_arm_id
    ):
        raise ValueError("mean guard does not bind the deterministic reference")
    if mean_guard_decision.candidate_belief_id != policy.mean_candidate_arm_id:
        raise ValueError("mean guard does not bind the registered mean candidate")
    if covariance_decision.exact_fallback_id != policy.exact_fallback_policy_id:
        raise ValueError("covariance decision binds a different exact fallback")
    if (
        covariance_decision.reference_policy_id
        != policy.reference_covariance_policy_id
    ):
        raise ValueError("covariance decision binds a different reference policy")
    if (
        covariance_decision.candidate_policy_id
        != policy.candidate_covariance_policy_id
    ):
        raise ValueError("covariance decision binds a different candidate policy")
    covariance_admissible = genuine_boolean(
        covariance_candidate_admissible,
        name="covariance_candidate_admissible",
    )
    common_admissible = genuine_boolean(
        common_prerequisites_admissible,
        name="common_prerequisites_admissible",
    )
    supported = genuine_boolean(reference_supported, name="reference_supported")
    outcome = _expected_outcome(
        policy,
        mean_inference_admissible=mean_guard_decision.inference_admissible,
        mean_regret_guard_accepted=mean_guard_decision.regret_guard_accepted,
        covariance_treatment_authorized=covariance_decision.authorized,
        covariance_candidate_admissible=covariance_admissible,
        common_prerequisites_admissible=common_admissible,
        reference_supported=supported,
    )
    return BeliefComponentAdmissionDecisionV1(
        policy=policy,
        mean_guard_decision_id=mean_guard_decision.decision_id,
        covariance_decision_id=cast(str, covariance_decision.artifact_id),
        mean_inference_admissible=mean_guard_decision.inference_admissible,
        mean_regret_guard_accepted=mean_guard_decision.regret_guard_accepted,
        covariance_treatment_authorized=covariance_decision.authorized,
        covariance_candidate_admissible=covariance_admissible,
        common_prerequisites_admissible=common_admissible,
        reference_supported=supported,
        selected_mode=outcome[0],
        selected_arm_id=outcome[1],
        reasons=outcome[2],
        metadata={} if metadata is None else metadata,
    )


@dataclass(frozen=True, slots=True)
class BeliefComponentAdmissionResultV1(Generic[BeliefT]):
    """Exact caller-owned belief selected by a component-admission decision."""

    decision: BeliefComponentAdmissionDecisionV1
    exact_fallback_belief: BeliefT
    deterministic_reference_belief: BeliefT
    mean_candidate_belief: BeliefT
    covariance_candidate_belief: BeliefT
    full_belief: BeliefT
    selected_belief: BeliefT
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.decision, BeliefComponentAdmissionDecisionV1):
            raise TypeError(
                "decision must be a BeliefComponentAdmissionDecisionV1"
            )
        policy = self.decision.policy
        arms = {
            "exact-fallback": (
                self.exact_fallback_belief,
                policy.exact_fallback_arm_id,
            ),
            "deterministic-reference": (
                self.deterministic_reference_belief,
                policy.deterministic_reference_arm_id,
            ),
            "mean-only": (
                self.mean_candidate_belief,
                policy.mean_candidate_arm_id,
            ),
            "covariance-only": (
                self.covariance_candidate_belief,
                policy.covariance_candidate_arm_id,
            ),
            "full-belief": (self.full_belief, policy.full_belief_arm_id),
        }
        for role, (belief, expected_id) in arms.items():
            if _belief_id(belief, name=f"{role}_belief") != expected_id:
                raise ValueError(f"{role} belief does not match the policy arm")
        expected_selected = arms[self.decision.selected_mode][0]
        if self.selected_belief is not expected_selected:
            raise ValueError(
                "selected routing must reuse the exact registered belief object"
            )
        if _belief_id(
            self.selected_belief,
            name="selected_belief",
        ) != self.decision.selected_arm_id:
            raise ValueError("selected belief ID differs from the decision")
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="component-admission result metadata",
            ),
        )

    @property
    def artifact_id(self) -> str:
        return content_id(self.to_record())

    @property
    def exact_fallback(self) -> bool:
        return self.decision.exact_fallback

    def to_record(self) -> dict[str, object]:
        return {
            "schema": BELIEF_COMPONENT_ADMISSION_RESULT_SCHEMA,
            "schema_version": BELIEF_COMPONENT_ADMISSION_VERSION,
            "decision_id": self.decision.artifact_id,
            "policy_id": self.decision.policy.policy_id,
            "selected_mode": self.decision.selected_mode,
            "selected_belief_id": self.selected_belief.artifact_id,
            "exact_fallback": self.exact_fallback,
            "metadata": plain_json(self.metadata),
        }


def route_belief_component_admission(
    decision: BeliefComponentAdmissionDecisionV1,
    *,
    exact_fallback_belief: BeliefT,
    deterministic_reference_belief: BeliefT,
    mean_candidate_belief: BeliefT,
    covariance_candidate_belief: BeliefT,
    full_belief: BeliefT,
    metadata: Mapping[str, Any] | None = None,
) -> BeliefComponentAdmissionResultV1[BeliefT]:
    """Return one exact arm object selected by the frozen component decision."""

    if not isinstance(decision, BeliefComponentAdmissionDecisionV1):
        raise TypeError("decision must be a BeliefComponentAdmissionDecisionV1")
    selected = {
        "exact-fallback": exact_fallback_belief,
        "deterministic-reference": deterministic_reference_belief,
        "mean-only": mean_candidate_belief,
        "covariance-only": covariance_candidate_belief,
        "full-belief": full_belief,
    }[decision.selected_mode]
    return BeliefComponentAdmissionResultV1(
        decision=decision,
        exact_fallback_belief=exact_fallback_belief,
        deterministic_reference_belief=deterministic_reference_belief,
        mean_candidate_belief=mean_candidate_belief,
        covariance_candidate_belief=covariance_candidate_belief,
        full_belief=full_belief,
        selected_belief=selected,
        metadata={} if metadata is None else metadata,
    )


def load_belief_component_admission_policy(
    path: str | Path,
) -> BeliefComponentAdmissionPolicyV1:
    return BeliefComponentAdmissionPolicyV1.from_mapping(
        load_strict_json_object(path, label="component-admission policy")
    )


def write_belief_component_admission_policy(
    policy: BeliefComponentAdmissionPolicyV1,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(policy, BeliefComponentAdmissionPolicyV1):
        raise TypeError("policy must be a BeliefComponentAdmissionPolicyV1")
    write_atomic_json(policy.to_record(), path, overwrite=overwrite)


def load_belief_component_admission_decision(
    path: str | Path,
) -> BeliefComponentAdmissionDecisionV1:
    return BeliefComponentAdmissionDecisionV1.from_mapping(
        load_strict_json_object(path, label="component-admission decision")
    )


def write_belief_component_admission_decision(
    decision: BeliefComponentAdmissionDecisionV1,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(decision, BeliefComponentAdmissionDecisionV1):
        raise TypeError("decision must be a BeliefComponentAdmissionDecisionV1")
    write_atomic_json(decision.to_record(), path, overwrite=overwrite)


__all__ = [
    "BELIEF_COMPONENT_ADMISSION_CLAIM_BOUNDARY",
    "BELIEF_COMPONENT_ADMISSION_DECISION_SCHEMA",
    "BELIEF_COMPONENT_ADMISSION_POLICY_SCHEMA",
    "BELIEF_COMPONENT_ADMISSION_RESULT_SCHEMA",
    "BELIEF_COMPONENT_ADMISSION_VERSION",
    "BeliefComponentAdmissionDecisionV1",
    "BeliefComponentAdmissionPolicyV1",
    "BeliefComponentAdmissionResultV1",
    "ComponentAdmissionModeV1",
    "compose_belief_component_admission",
    "load_belief_component_admission_decision",
    "load_belief_component_admission_policy",
    "route_belief_component_admission",
    "write_belief_component_admission_decision",
    "write_belief_component_admission_policy",
]
