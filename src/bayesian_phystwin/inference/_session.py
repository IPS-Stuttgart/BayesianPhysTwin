"""Provider-neutral orchestration for guarded complete-belief updates."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar, cast, runtime_checkable

from .._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    plain_json,
)
from .._validation import lowercase_sha256
from ..complete_belief_selection import (
    ArtifactBelief,
    CompleteBeliefGuardDecisionV1,
)
from ._guarded import (
    GuardedCandidateInference,
    GuardedUpdateResultV1,
    finalize_guarded_update,
)

ObservationT = TypeVar("ObservationT")
ObservationT_contra = TypeVar("ObservationT_contra", contravariant=True)
BeliefT = TypeVar("BeliefT", bound=ArtifactBelief)
BeliefT_contra = TypeVar(
    "BeliefT_contra",
    bound=ArtifactBelief,
    contravariant=True,
)


def _artifact_id(value: object, *, name: str) -> str:
    try:
        artifact_id = cast(ArtifactBelief, value).artifact_id
    except AttributeError as error:
        raise TypeError(f"{name} must expose artifact_id") from error
    return lowercase_sha256(artifact_id, name=f"{name}.artifact_id")


def _content_id(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            plain_json(payload),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class CandidateProposalV1(Generic[BeliefT]):
    """Bind one undeployed inference result to one complete candidate belief."""

    inference: GuardedCandidateInference
    candidate_belief: BeliefT
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.inference, GuardedCandidateInference):
            raise TypeError(
                "inference must expose candidate_id and inference_admissible"
            )
        lowercase_sha256(
            self.inference.candidate_id,
            name="inference.candidate_id",
        )
        genuine_boolean(
            self.inference.inference_admissible,
            name="inference.inference_admissible",
        )
        _artifact_id(self.candidate_belief, name="candidate_belief")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="candidate proposal metadata",
            ),
        )

    @property
    def proposal_id(self) -> str:
        return _content_id(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "schema": "bayesian_phystwin.candidate_proposal",
            "schema_version": 1,
            "inference_candidate_id": self.inference.candidate_id,
            "inference_admissible": self.inference.inference_admissible,
            "candidate_belief_id": self.candidate_belief.artifact_id,
            "metadata": plain_json(self.metadata),
        }


@runtime_checkable
class SessionCandidateFactory(Protocol[ObservationT_contra, BeliefT]):
    """Construct one typed candidate without choosing deployment policy."""

    def __call__(
        self,
        prior: BeliefT,
        observation: ObservationT_contra,
        *,
        context: Mapping[str, Any],
    ) -> CandidateProposalV1[BeliefT]: ...


@runtime_checkable
class SessionGuardPolicy(Protocol[BeliefT_contra]):
    """Choose a guard decision without constructing or copying a belief."""

    def __call__(
        self,
        inference: GuardedCandidateInference,
        baseline_belief: BeliefT_contra,
        candidate_belief: BeliefT_contra,
        *,
        context: Mapping[str, Any],
    ) -> CompleteBeliefGuardDecisionV1: ...


@dataclass(frozen=True, slots=True)
class InferenceSession(Generic[ObservationT, BeliefT]):
    """Compose provider-neutral candidate construction and guarded routing."""

    session_id: str
    candidate_factory: SessionCandidateFactory[ObservationT, BeliefT]
    guard_policy: SessionGuardPolicy[BeliefT]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "session_id",
            lowercase_sha256(self.session_id, name="session_id"),
        )
        if not isinstance(self.candidate_factory, SessionCandidateFactory):
            raise TypeError("candidate_factory must be callable")
        if not isinstance(self.guard_policy, SessionGuardPolicy):
            raise TypeError("guard_policy must be callable")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="inference session metadata",
            ),
        )

    def assimilate(
        self,
        prior: BeliefT,
        observation: ObservationT,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> GuardedUpdateResultV1[BeliefT]:
        """Infer one candidate, obtain one guard decision, and route exactly."""

        _artifact_id(prior, name="prior")
        if context is None:
            validated_context: Mapping[str, Any] = {}
        elif isinstance(context, Mapping):
            validated_context = frozen_finite_json_mapping(
                context,
                name="inference session context",
            )
        else:
            raise TypeError("context must be a mapping or None")

        proposal = self.candidate_factory(
            prior,
            observation,
            context=validated_context,
        )
        if not isinstance(proposal, CandidateProposalV1):
            raise TypeError("candidate_factory must return CandidateProposalV1")

        decision = self.guard_policy(
            proposal.inference,
            prior,
            proposal.candidate_belief,
            context=validated_context,
        )
        if not isinstance(decision, CompleteBeliefGuardDecisionV1):
            raise TypeError("guard_policy must return CompleteBeliefGuardDecisionV1")

        return finalize_guarded_update(
            proposal.inference,
            prior,
            proposal.candidate_belief,
            decision,
            metadata={
                "session_id": self.session_id,
                "proposal_id": proposal.proposal_id,
                "session": plain_json(self.metadata),
                "proposal": plain_json(proposal.metadata),
                "context": plain_json(validated_context),
            },
        )


__all__ = [
    "CandidateProposalV1",
    "InferenceSession",
    "SessionCandidateFactory",
    "SessionGuardPolicy",
]
