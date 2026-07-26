"""Fail-closed orchestration for prior-aware physical belief updates.

The numerical update, nonlinear closure check, and deployment guard are kept
separate.  Observation support may decide whether a candidate is eligible, but
it is never folded back into the prior perception reliability or evaluated as
a second likelihood term.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, TypeVar

import numpy as np

from ._gauge_aware_contracts import GaugeAwareBeliefResult, GaugeAwareSelection
from .complete_belief_selection import (
    ArtifactBelief,
    CompleteBeliefGuardDecisionV1,
    CompleteBeliefSelectionV1,
    select_complete_belief,
)
from .gauge_aware_belief import (
    decode_gauge_aware_query,
    select_gauge_aware_candidate,
)
from .observation_belief import ObservationBeliefV1, array_sha256
from .physical_linearization import (
    NonlinearClosureV1,
    PhysicalLinearizationV1,
    build_gauge_aware_batch_from_artifacts,
    evaluate_nonlinear_closure,
)
from .prior_aware_gauge_belief import (
    PriorAwareGaugeConfigV1,
    update_prior_aware_gauge_belief,
)


class RegretDecision(Protocol):
    """Minimum source-certificate decision consumed by the pipeline."""

    candidate_accepted: bool
    selected_value: np.ndarray
    reason: str


BeliefT = TypeVar("BeliefT", bound=ArtifactBelief)


def _validate_sha256(value: str, *, name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _validated_metadata(values: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(
            json.dumps(dict(values), sort_keys=True, allow_nan=False)
        )
    except (TypeError, ValueError) as error:
        raise ValueError("metadata must contain finite JSON values") from error


def _content_id(values: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(values),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ProspectiveSupportDecisionV1:
    """Target-free support required before a state update may be deployed.

    Structural support covers correspondence or independent-view/modality
    evidence.  Physical support covers action-conditioned response evidence.
    Both are routing evidence only: they must not be reused as prior
    reliability or as another likelihood evaluation.
    """

    structural_evidence_id: str
    physical_evidence_id: str
    structural_support_accepted: bool
    physical_support_accepted: bool
    structural_support_kind: str
    physical_support_kind: str
    future_target_read: bool = False
    used_as_prior_reliability: bool = False
    candidate_innovation_reused: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_sha256(
            self.structural_evidence_id,
            name="structural_evidence_id",
        )
        _validate_sha256(
            self.physical_evidence_id,
            name="physical_evidence_id",
        )
        if not self.structural_support_kind or not self.physical_support_kind:
            raise ValueError("support kinds must be nonempty")
        for name, value in (
            ("structural_support_accepted", self.structural_support_accepted),
            ("physical_support_accepted", self.physical_support_accepted),
            ("future_target_read", self.future_target_read),
            ("used_as_prior_reliability", self.used_as_prior_reliability),
            ("candidate_innovation_reused", self.candidate_innovation_reused),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")
        if self.future_target_read:
            raise ValueError("prospective support must not read a future target")
        if self.used_as_prior_reliability:
            raise ValueError(
                "support routing must remain separate from prior reliability"
            )
        if self.candidate_innovation_reused:
            raise ValueError(
                "support routing must not process the candidate innovation again"
            )
        object.__setattr__(self, "metadata", _validated_metadata(self.metadata))

    @property
    def accepted(self) -> bool:
        return (
            self.structural_support_accepted
            and self.physical_support_accepted
        )

    @property
    def decision_id(self) -> str:
        return _content_id(
            {
                "schema": "bayesian_phystwin.prospective_support",
                "schema_version": 1,
                "structural_evidence_id": self.structural_evidence_id,
                "physical_evidence_id": self.physical_evidence_id,
                "structural_support_accepted": (
                    self.structural_support_accepted
                ),
                "physical_support_accepted": self.physical_support_accepted,
                "structural_support_kind": self.structural_support_kind,
                "physical_support_kind": self.physical_support_kind,
                "future_target_read": self.future_target_read,
                "used_as_prior_reliability": self.used_as_prior_reliability,
                "candidate_innovation_reused": self.candidate_innovation_reused,
                "metadata": dict(self.metadata),
            }
        )


@dataclass(frozen=True)
class GuardedBeliefPipelineConfigV1:
    """Frozen local-linearization closure tolerances."""

    closure_absolute_tolerance_m: float = 0.005
    closure_relative_tolerance: float = 0.25
    closure_denominator_floor_m: float = 1e-12

    def __post_init__(self) -> None:
        values = (
            self.closure_absolute_tolerance_m,
            self.closure_relative_tolerance,
            self.closure_denominator_floor_m,
        )
        if any(not np.isfinite(value) for value in values):
            raise ValueError("closure tolerances must be finite")
        if (
            self.closure_absolute_tolerance_m < 0.0
            or self.closure_relative_tolerance < 0.0
            or self.closure_denominator_floor_m <= 0.0
        ):
            raise ValueError(
                "closure tolerances must be nonnegative and the floor positive"
            )


@dataclass(frozen=True)
class GuardedBeliefPipelineOutcomeV1:
    """Audit record for one complete-belief routing decision."""

    observation_artifact_id: str
    physical_linearization_id: str
    support_decision: ProspectiveSupportDecisionV1
    numerical_result: GaugeAwareBeliefResult
    nonlinear_closure: NonlinearClosureV1
    query_selection: GaugeAwareSelection
    complete_decision: CompleteBeliefGuardDecisionV1
    complete_selection: CompleteBeliefSelectionV1
    linearized_query_m: np.ndarray
    nonlinear_query_sha256: str
    innovation_processed_once: bool = True
    support_used_as_prior_reliability: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (
            ("observation_artifact_id", self.observation_artifact_id),
            ("physical_linearization_id", self.physical_linearization_id),
            ("nonlinear_query_sha256", self.nonlinear_query_sha256),
        ):
            _validate_sha256(value, name=name)
        if (
            self.nonlinear_closure.linearization_artifact_id
            != self.physical_linearization_id
        ):
            raise ValueError("closure is not bound to the physical linearization")
        if (
            self.complete_selection.guard_decision_id
            != self.complete_decision.decision_id
        ):
            raise ValueError("selection is not bound to the complete decision")
        if (
            self.complete_decision.metadata.get("nonlinear_query_sha256")
            != self.nonlinear_query_sha256
        ):
            raise ValueError("complete decision does not bind the nonlinear query")
        if not self.innovation_processed_once:
            raise ValueError("the physical innovation must be processed once")
        if self.support_used_as_prior_reliability:
            raise ValueError(
                "support evidence must not alter prior perception reliability"
            )
        expected_candidate = self.complete_selection.selected_candidate
        if self.query_selection.candidate_accepted != expected_candidate:
            raise ValueError("query and complete-belief routing disagree")
        object.__setattr__(
            self,
            "linearized_query_m",
            _readonly(self.linearized_query_m),
        )
        object.__setattr__(self, "metadata", _validated_metadata(self.metadata))

    @property
    def selected_candidate(self) -> bool:
        return self.complete_selection.selected_candidate


def run_prior_aware_guarded_belief_update(
    observation_belief: ObservationBeliefV1,
    linearization: PhysicalLinearizationV1,
    *,
    baseline_belief: BeliefT,
    candidate_belief: BeliefT,
    physical_prediction_xyz_m: np.ndarray,
    baseline_query_m: np.ndarray,
    nonlinear_candidate_query_m: np.ndarray,
    support_decision: ProspectiveSupportDecisionV1,
    regret_decision: RegretDecision,
    source_certificate_id: str,
    common_domain_id: str,
    inference_config: PriorAwareGaugeConfigV1 | None = None,
    pipeline_config: GuardedBeliefPipelineConfigV1 | None = None,
    shared_bias_jacobian: np.ndarray | None = None,
    view_bias_jacobian: np.ndarray | None = None,
    state_prior_covariance_m2: np.ndarray | None = None,
    anchor_innovation_m: np.ndarray | None = None,
    anchor_covariance_m2: np.ndarray | None = None,
    anchor_state_jacobian: np.ndarray | None = None,
    anchor_dependence: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[BeliefT, GuardedBeliefPipelineOutcomeV1]:
    """Infer, close, guard, and route one complete Bayesian twin belief.

    The caller supplies the nonlinear candidate replay and a source-fitted
    regret decision.  This function verifies the observation/linearization
    lineage, forms the innovation once, and requires every gate before routing
    the complete candidate belief.
    """

    cfg = pipeline_config or GuardedBeliefPipelineConfigV1()
    _validate_sha256(source_certificate_id, name="source_certificate_id")
    _validate_sha256(common_domain_id, name="common_domain_id")
    _validate_sha256(baseline_belief.artifact_id, name="baseline artifact_id")
    _validate_sha256(candidate_belief.artifact_id, name="candidate artifact_id")
    if baseline_belief.artifact_id != linearization.baseline_belief_id:
        raise ValueError("linearization does not bind the baseline belief")
    if candidate_belief.artifact_id == baseline_belief.artifact_id:
        raise ValueError("candidate belief must differ from the baseline")

    adapted = build_gauge_aware_batch_from_artifacts(
        observation_belief,
        linearization,
        physical_prediction_xyz_m=physical_prediction_xyz_m,
        shared_bias_jacobian=shared_bias_jacobian,
        view_bias_jacobian=view_bias_jacobian,
        state_prior_covariance_m2=state_prior_covariance_m2,
        anchor_innovation_m=anchor_innovation_m,
        anchor_covariance_m2=anchor_covariance_m2,
        anchor_state_jacobian=anchor_state_jacobian,
        **dict(anchor_dependence or {}),
    )
    numerical_result = update_prior_aware_gauge_belief(
        adapted.batch,
        config=inference_config,
    )

    baseline_query = np.asarray(baseline_query_m)
    nonlinear_query = np.asarray(nonlinear_candidate_query_m)
    expected_query_shape = linearization.query_state_jacobian.shape[:2]
    if baseline_query.shape != expected_query_shape:
        raise ValueError(
            f"baseline query must have shape {expected_query_shape}"
        )
    if nonlinear_query.shape != baseline_query.shape:
        raise ValueError("nonlinear candidate query shape changed")
    if (
        not np.all(np.isfinite(baseline_query))
        or not np.all(np.isfinite(nonlinear_query))
    ):
        raise ValueError("query arrays must be finite")

    linearized_query = baseline_query + decode_gauge_aware_query(
        numerical_result,
        linearization.query_state_jacobian,
    )
    closure = evaluate_nonlinear_closure(
        linearization.artifact_id,
        baseline_query_m=baseline_query,
        linearized_query_m=linearized_query,
        nonlinear_query_m=nonlinear_query,
        absolute_tolerance_m=cfg.closure_absolute_tolerance_m,
        relative_tolerance=cfg.closure_relative_tolerance,
        denominator_floor_m=cfg.closure_denominator_floor_m,
        metadata={
            "observation_artifact_id": observation_belief.artifact_id,
            "baseline_belief_id": baseline_belief.artifact_id,
            "candidate_belief_id": candidate_belief.artifact_id,
        },
    )

    gated_admissible = (
        numerical_result.inference_admissible
        and support_decision.accepted
        and closure.candidate_valid
    )
    if not numerical_result.inference_admissible:
        gated_reason = numerical_result.reason
    elif not support_decision.accepted:
        gated_reason = "prospective-support-rejected"
    elif not closure.candidate_valid:
        gated_reason = "nonlinear-closure-rejected"
    else:
        gated_reason = "all-inference-gates-accepted"
    gated_result = replace(
        numerical_result,
        inference_admissible=gated_admissible,
        reason=gated_reason,
    )
    query_selection = select_gauge_aware_candidate(
        baseline_query,
        nonlinear_query,
        gated_result,
        regret_decision=regret_decision,
    )
    baseline_query_sha256 = array_sha256(baseline_query)
    linearized_query_sha256 = array_sha256(linearized_query)
    nonlinear_query_sha256 = array_sha256(nonlinear_query)
    regret_selected_value_sha256 = array_sha256(regret_decision.selected_value)
    regret_decision_id = _content_id(
        {
            "schema": "bayesian_phystwin.regret_decision_binding",
            "schema_version": 1,
            "source_certificate_id": source_certificate_id,
            "candidate_accepted": bool(regret_decision.candidate_accepted),
            "selected_value_sha256": regret_selected_value_sha256,
            "reason": str(regret_decision.reason),
        }
    )
    complete_decision = CompleteBeliefGuardDecisionV1(
        baseline_belief_id=baseline_belief.artifact_id,
        candidate_belief_id=candidate_belief.artifact_id,
        common_domain_id=common_domain_id,
        certificate_id=source_certificate_id,
        inference_admissible=gated_admissible,
        regret_guard_accepted=query_selection.candidate_accepted,
        reason=query_selection.reason,
        metadata={
            "observation_artifact_id": observation_belief.artifact_id,
            "physical_linearization_id": linearization.artifact_id,
            "nonlinear_closure_id": closure.closure_id,
            "support_decision_id": support_decision.decision_id,
            "baseline_query_sha256": baseline_query_sha256,
            "linearized_query_sha256": linearized_query_sha256,
            "nonlinear_query_sha256": nonlinear_query_sha256,
            "regret_decision_id": regret_decision_id,
            "regret_selected_value_sha256": regret_selected_value_sha256,
            "numerical_inference_admissible": (
                numerical_result.inference_admissible
            ),
            "prospective_support_accepted": support_decision.accepted,
            "nonlinear_closure_accepted": closure.candidate_valid,
            "source_regret_accepted_before_other_gates": bool(
                regret_decision.candidate_accepted
            ),
            "innovation_processed_once": True,
            "support_used_as_prior_reliability": False,
        },
    )
    selected, complete_selection = select_complete_belief(
        baseline_belief,
        candidate_belief,
        complete_decision,
        metadata=metadata,
    )
    if complete_selection.selected_candidate != query_selection.candidate_accepted:
        raise AssertionError("query and complete-belief routing diverged")
    outcome = GuardedBeliefPipelineOutcomeV1(
        observation_artifact_id=observation_belief.artifact_id,
        physical_linearization_id=linearization.artifact_id,
        support_decision=support_decision,
        numerical_result=numerical_result,
        nonlinear_closure=closure,
        query_selection=query_selection,
        complete_decision=complete_decision,
        complete_selection=complete_selection,
        linearized_query_m=linearized_query,
        nonlinear_query_sha256=nonlinear_query_sha256,
        metadata=metadata or {},
    )
    return selected, outcome


__all__ = [
    "GuardedBeliefPipelineConfigV1",
    "GuardedBeliefPipelineOutcomeV1",
    "ProspectiveSupportDecisionV1",
    "RegretDecision",
    "run_prior_aware_guarded_belief_update",
]
