"""Stable orchestration for strict inference and complete-belief selection."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Generic, Protocol, TypeVar, cast, runtime_checkable

import numpy as np

from .._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    plain_json,
)
from .._validation import lowercase_sha256, optional_instance
from ..complete_belief_selection import (
    ArtifactBelief,
    CompleteBeliefGuardDecisionV1,
    CompleteBeliefSelectionV1,
    select_complete_belief,
)
from ..observation_belief import ObservationBeliefV1
from ..physical_linearization import PhysicalLinearizationV1
from ..posterior_covariance_semantics import PosteriorCovarianceSemanticsV1
from ..prior_aware_gauge_belief import PriorAwareGaugeConfigV1
from ..prospective_prob4d_update import (
    ClaimBearingProb4DCandidateV1,
    infer_claim_bearing_prob4d_candidate_from_artifacts,
)
from ._anchor_dependence import AnchorDependenceV1

BeliefT = TypeVar("BeliefT", bound=ArtifactBelief)

_LEGACY_ANCHOR_DEPENDENCE_KEYS = frozenset(
    {
        "anchor_correlation_group_ids",
        "anchor_prior_reliability",
        "anchor_prior_nominal_probability",
        "anchor_composite_weight",
        "anchor_bias_jacobian",
        "anchor_bias_prior_covariance",
    }
)


@runtime_checkable
class GuardedCandidateInference(Protocol):
    """Minimum candidate-inference identity consumed by the deployment router."""

    @property
    def candidate_id(self) -> str: ...

    @property
    def inference_admissible(self) -> bool: ...


def _content_id(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            plain_json(payload),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _belief_id(value: object, *, name: str) -> str:
    try:
        artifact_id = cast(ArtifactBelief, value).artifact_id
    except AttributeError as error:
        raise TypeError(f"{name} must expose artifact_id") from error
    return lowercase_sha256(artifact_id, name=f"{name}.artifact_id")


@dataclass(frozen=True, slots=True)
class GuardedUpdateResultV1(Generic[BeliefT]):
    """Complete accepted-update or exact-fallback routing result."""

    inference_candidate_id: str
    inference_admissible: bool
    baseline_belief: BeliefT
    candidate_belief: BeliefT
    selected_belief: BeliefT
    guard_decision: CompleteBeliefGuardDecisionV1
    selection: CompleteBeliefSelectionV1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        inference_candidate_id = lowercase_sha256(
            self.inference_candidate_id,
            name="inference_candidate_id",
        )
        inference_admissible = genuine_boolean(
            self.inference_admissible,
            name="inference_admissible",
        )
        baseline_id = _belief_id(self.baseline_belief, name="baseline_belief")
        candidate_id = _belief_id(self.candidate_belief, name="candidate_belief")
        selected_id = _belief_id(self.selected_belief, name="selected_belief")
        if not isinstance(self.guard_decision, CompleteBeliefGuardDecisionV1):
            raise TypeError("guard_decision must be a CompleteBeliefGuardDecisionV1")
        if not isinstance(self.selection, CompleteBeliefSelectionV1):
            raise TypeError("selection must be a CompleteBeliefSelectionV1")
        if self.guard_decision.inference_admissible != inference_admissible:
            raise ValueError(
                "guard decision disagrees with candidate inference admissibility"
            )
        if self.guard_decision.baseline_belief_id != baseline_id:
            raise ValueError("guard decision does not bind the baseline belief")
        if self.guard_decision.candidate_belief_id != candidate_id:
            raise ValueError("guard decision does not bind the candidate belief")
        if self.selection.baseline_belief_id != baseline_id:
            raise ValueError("selection does not bind the baseline belief")
        if self.selection.candidate_belief_id != candidate_id:
            raise ValueError("selection does not bind the candidate belief")
        if self.selection.guard_decision_id != self.guard_decision.decision_id:
            raise ValueError("selection does not bind the guard decision")
        if self.selection.selected_belief_id != selected_id:
            raise ValueError("selection does not bind the selected belief")
        if self.selection.selected_candidate:
            if self.selected_belief is not self.candidate_belief:
                raise ValueError(
                    "accepted routing must reuse the exact candidate belief object"
                )
        elif self.selected_belief is not self.baseline_belief:
            raise ValueError(
                "rejected routing must reuse the exact baseline belief object"
            )
        object.__setattr__(
            self,
            "inference_candidate_id",
            inference_candidate_id,
        )
        object.__setattr__(
            self,
            "inference_admissible",
            inference_admissible,
        )
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="guarded update metadata",
            ),
        )

    @property
    def selected_candidate(self) -> bool:
        return self.selection.selected_candidate

    @property
    def exact_fallback(self) -> bool:
        return (
            not self.selected_candidate and self.selected_belief is self.baseline_belief
        )

    @property
    def artifact_id(self) -> str:
        return _content_id(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "schema": "bayesian_phystwin.guarded_update",
            "schema_version": 1,
            "inference_candidate_id": self.inference_candidate_id,
            "inference_admissible": self.inference_admissible,
            "baseline_belief_id": self.baseline_belief.artifact_id,
            "candidate_belief_id": self.candidate_belief.artifact_id,
            "guard_decision_id": self.guard_decision.decision_id,
            "selection_id": self.selection.selection_id,
            "selected_belief_id": self.selected_belief.artifact_id,
            "selected_candidate": self.selected_candidate,
            "exact_fallback": self.exact_fallback,
            "metadata": plain_json(self.metadata),
        }


def finalize_guarded_update(
    inference: GuardedCandidateInference,
    baseline_belief: BeliefT,
    candidate_belief: BeliefT,
    guard_decision: CompleteBeliefGuardDecisionV1,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> GuardedUpdateResultV1[BeliefT]:
    """Route one complete candidate or return the exact baseline object."""

    if not isinstance(inference, GuardedCandidateInference):
        raise TypeError("inference must expose candidate_id and inference_admissible")
    inference_candidate_id = lowercase_sha256(
        inference.candidate_id,
        name="inference.candidate_id",
    )
    inference_admissible = genuine_boolean(
        inference.inference_admissible,
        name="inference.inference_admissible",
    )
    _belief_id(baseline_belief, name="baseline_belief")
    _belief_id(candidate_belief, name="candidate_belief")
    if not isinstance(guard_decision, CompleteBeliefGuardDecisionV1):
        raise TypeError("guard_decision must be a CompleteBeliefGuardDecisionV1")
    if guard_decision.inference_admissible != inference_admissible:
        raise ValueError(
            "guard decision disagrees with candidate inference admissibility"
        )
    caller_metadata: Mapping[str, Any]
    if metadata is None:
        caller_metadata = {}
    elif isinstance(metadata, Mapping):
        caller_metadata = metadata
    else:
        raise TypeError("metadata must be a mapping or None")
    selected, selection = select_complete_belief(
        baseline_belief,
        candidate_belief,
        guard_decision,
        metadata={
            "inference_candidate_id": inference_candidate_id,
            "caller": plain_json(
                frozen_finite_json_mapping(
                    caller_metadata,
                    name="guarded update metadata",
                )
            ),
        },
    )
    return GuardedUpdateResultV1(
        inference_candidate_id=inference_candidate_id,
        inference_admissible=inference_admissible,
        baseline_belief=baseline_belief,
        candidate_belief=candidate_belief,
        selected_belief=selected,
        guard_decision=guard_decision,
        selection=selection,
        metadata=caller_metadata,
    )


def _typed_anchor_dependence_kwargs(
    anchor_dependence: AnchorDependenceV1 | None,
    legacy_anchor_dependence: Mapping[str, Any],
    *,
    anchor_innovation_m: np.ndarray | None,
) -> tuple[dict[str, object], AnchorDependenceV1 | None]:
    validated = optional_instance(
        anchor_dependence,
        AnchorDependenceV1,
        name="anchor_dependence",
    )
    unknown_legacy_keys = sorted(
        set(legacy_anchor_dependence) - _LEGACY_ANCHOR_DEPENDENCE_KEYS
    )
    if unknown_legacy_keys:
        raise TypeError(
            "unknown legacy anchor dependence keywords: "
            + ", ".join(unknown_legacy_keys)
        )
    if validated is None:
        return dict(legacy_anchor_dependence), None
    if legacy_anchor_dependence:
        raise ValueError(
            "anchor_dependence cannot be combined with legacy anchor keywords"
        )
    if anchor_innovation_m is None:
        raise ValueError("anchor_dependence requires anchor_innovation_m")
    innovation = np.asarray(anchor_innovation_m)
    if innovation.ndim != 2 or innovation.shape[1] != 3:
        raise ValueError("anchor_innovation_m must have shape (A, 3)")
    validated.require_anchor_count(len(innovation))
    return validated.inference_kwargs(), validated


def _bind_anchor_dependence_identity(
    candidate: ClaimBearingProb4DCandidateV1,
    anchor_dependence: AnchorDependenceV1,
) -> ClaimBearingProb4DCandidateV1:
    artifact_id = lowercase_sha256(
        anchor_dependence.artifact_id,
        name="anchor_dependence.artifact_id",
    )
    lineage = dict(candidate.result.input_lineage)
    existing = lineage.get("anchor_dependence_artifact_id")
    if existing is not None and existing != artifact_id:
        raise ValueError(
            "candidate lineage contradicts anchor_dependence artifact identity"
        )
    lineage["anchor_dependence_artifact_id"] = artifact_id
    bound_result = replace(candidate.result, input_lineage=lineage)
    bound_update = replace(candidate.update_v1, result=bound_result)
    return replace(candidate, update_v1=bound_update)


def infer_prob4d_candidate(
    observation_belief: ObservationBeliefV1,
    linearization: PhysicalLinearizationV1,
    *,
    physical_prediction_xyz_m: np.ndarray,
    shared_bias_jacobian: np.ndarray | None = None,
    view_bias_jacobian: np.ndarray | None = None,
    state_prior_covariance_m2: np.ndarray | None = None,
    anchor_innovation_m: np.ndarray | None = None,
    anchor_covariance_m2: np.ndarray | None = None,
    anchor_state_jacobian: np.ndarray | None = None,
    config: PriorAwareGaugeConfigV1 | None = None,
    covariance_semantics: PosteriorCovarianceSemanticsV1 | None = None,
    anchor_dependence: AnchorDependenceV1 | None = None,
    **legacy_anchor_dependence: Any,
) -> ClaimBearingProb4DCandidateV1:
    """Run strict Prob4D admission and return a typed, undeployed candidate.

    ``anchor_dependence`` is the preferred stable contract. The historical
    individual anchor keywords remain accepted for compatibility, but callers
    may not mix both forms in one update.
    """

    if not isinstance(observation_belief, ObservationBeliefV1):
        raise TypeError("observation_belief must be an ObservationBeliefV1")
    if not isinstance(linearization, PhysicalLinearizationV1):
        raise TypeError("linearization must be a PhysicalLinearizationV1")
    validated_config = optional_instance(
        config,
        PriorAwareGaugeConfigV1,
        name="config",
    )
    validated_covariance_semantics = optional_instance(
        covariance_semantics,
        PosteriorCovarianceSemanticsV1,
        name="covariance_semantics",
    )
    dependence_kwargs, validated_anchor_dependence = _typed_anchor_dependence_kwargs(
        anchor_dependence,
        legacy_anchor_dependence,
        anchor_innovation_m=anchor_innovation_m,
    )
    candidate = infer_claim_bearing_prob4d_candidate_from_artifacts(
        observation_belief,
        linearization,
        physical_prediction_xyz_m=physical_prediction_xyz_m,
        shared_bias_jacobian=shared_bias_jacobian,
        view_bias_jacobian=view_bias_jacobian,
        state_prior_covariance_m2=state_prior_covariance_m2,
        anchor_innovation_m=anchor_innovation_m,
        anchor_covariance_m2=anchor_covariance_m2,
        anchor_state_jacobian=anchor_state_jacobian,
        config=validated_config,
        covariance_semantics=validated_covariance_semantics,
        **dependence_kwargs,
    )
    if validated_anchor_dependence is None:
        return candidate
    return _bind_anchor_dependence_identity(
        candidate,
        validated_anchor_dependence,
    )


__all__ = [
    "GuardedCandidateInference",
    "GuardedUpdateResultV1",
    "finalize_guarded_update",
    "infer_prob4d_candidate",
]
