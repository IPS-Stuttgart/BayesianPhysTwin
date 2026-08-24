"""Target-closed action design for query-identifying physical probes.

The module ranks an externally supplied finite action roster by the expected
contraction of one registered physical-query covariance after explicit nuisance
marginalization.  The caller supplies the action roster and safety/risk inputs;
this module does not generate or execute physical actions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    literal_lower_hex,
    plain_json,
)
from ._portable_contracts import content_id
from ._query_identifying_action_contracts import (
    QUERY_IDENTIFYING_ACTION_DESIGN_CLAIM_BOUNDARY,
    QUERY_IDENTIFYING_ACTION_DESIGN_SCHEMA,
    QUERY_IDENTIFYING_ACTION_DESIGN_SEMANTICS,
    QUERY_IDENTIFYING_ACTION_DESIGN_VERSION,
    QUERY_IDENTIFYING_ACTION_EVALUATION_SCHEMA,
    QUERY_IDENTIFYING_ACTION_EVALUATION_VERSION,
    QUERY_IDENTIFYING_ACTION_SCHEMA,
    QUERY_IDENTIFYING_ACTION_VERSION,
    QueryIdentifyingActionCandidateV1,
    QueryIdentifyingActionEvaluationV1,
    QueryIdentifyingActionStatus,
    QueryIdentifyingDesignStatus,
    _array_record,
    _finite_nonnegative,
    _finite_positive,
    _immutable_float64,
    _information_state_record,
    _normalized_covariance,
    _positive_definite,
    _query_covariance,
    _real_matrix,
)
from .nuisance_aware_information import NuisanceAwareInformationState


@dataclass(frozen=True, slots=True)
class QueryIdentifyingActionDesignV1:
    """Rank a frozen action roster by query-specific posterior contraction."""

    prior_belief_id: str
    query_id: str
    query_scale_id: str
    protocol_id: str
    prior_state_precision: np.ndarray = field(repr=False)
    prior_nuisance_precision: np.ndarray = field(repr=False)
    prior_state_nuisance_precision: np.ndarray = field(repr=False)
    query_jacobian: np.ndarray = field(repr=False)
    query_scale: np.ndarray = field(repr=False)
    candidates: Sequence[QueryIdentifyingActionCandidateV1] = field(repr=False)
    cost_weight: float = 0.0
    risk_weight: float = 0.0
    maximum_risk: float = 1.0
    minimum_objective_improvement: float = 0.0
    numerical_tolerance: float = 1e-12
    selection_tolerance: float = 1e-12
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    baseline_query_covariance: np.ndarray = field(init=False, repr=False)
    baseline_normalized_query_covariance: np.ndarray = field(init=False, repr=False)
    baseline_normalized_query_trace: float = field(init=False)
    evaluations: tuple[QueryIdentifyingActionEvaluationV1, ...] = field(
        init=False,
        repr=False,
    )
    status: QueryIdentifyingDesignStatus = field(init=False)
    selected_action_id: str | None = field(init=False)
    selected_candidate_id: str | None = field(init=False)
    selected_evaluation_id: str | None = field(init=False)

    def __post_init__(self) -> None:
        for name in ("prior_belief_id", "query_id", "query_scale_id", "protocol_id"):
            object.__setattr__(
                self,
                name,
                cast(
                    str,
                    literal_lower_hex(
                        getattr(self, name),
                        name=name,
                        lengths={64},
                    ),
                ),
            )
        tolerance = _finite_positive(
            self.numerical_tolerance,
            name="numerical_tolerance",
        )
        selection_tolerance = _finite_nonnegative(
            self.selection_tolerance,
            name="selection_tolerance",
        )
        cost_weight = _finite_nonnegative(self.cost_weight, name="cost_weight")
        risk_weight = _finite_nonnegative(self.risk_weight, name="risk_weight")
        maximum_risk = _finite_nonnegative(self.maximum_risk, name="maximum_risk")
        minimum_improvement = _finite_nonnegative(
            self.minimum_objective_improvement,
            name="minimum_objective_improvement",
        )

        state = NuisanceAwareInformationState(
            state_precision=self.prior_state_precision,
            nuisance_precision=self.prior_nuisance_precision,
            state_nuisance_precision=self.prior_state_nuisance_precision,
        )
        query = _real_matrix(self.query_jacobian, name="query_jacobian")
        if query.shape[0] == 0 or query.shape[1] != state.state_dimension:
            raise ValueError(
                "query_jacobian must have one column per physical coefficient "
                "and at least one query row"
            )
        query_scale = _positive_definite(self.query_scale, name="query_scale")
        if query_scale.shape != (query.shape[0], query.shape[0]):
            raise ValueError(
                "query_scale must have one row and column per query coordinate"
            )

        candidates = tuple(self.candidates)
        if not candidates:
            raise ValueError("candidates must contain at least one action")
        if any(
            not isinstance(item, QueryIdentifyingActionCandidateV1)
            for item in candidates
        ):
            raise TypeError(
                "every candidate must be QueryIdentifyingActionCandidateV1"
            )
        action_ids = [item.action_id for item in candidates]
        if len(set(action_ids)) != len(action_ids):
            raise ValueError("candidate action_id values must be unique")
        for candidate in candidates:
            if candidate.state_dimension != state.state_dimension:
                raise ValueError(
                    "candidate state_jacobian dimension differs from prior"
                )
            if candidate.nuisance_dimension != state.nuisance_dimension:
                raise ValueError(
                    "candidate nuisance_jacobian dimension differs from prior"
                )
        candidates = tuple(sorted(candidates, key=lambda item: item.action_id))

        scale_cholesky = np.linalg.cholesky(query_scale)
        baseline_query = _query_covariance(state, query)
        baseline_normalized = _normalized_covariance(
            baseline_query,
            scale_cholesky,
        )
        baseline_trace = float(np.trace(baseline_normalized))
        if baseline_trace < -tolerance:
            raise ValueError("baseline normalized query covariance is not positive")
        baseline_trace = max(0.0, baseline_trace)
        trivial_query = baseline_trace <= tolerance

        evaluations = tuple(
            self._evaluate_candidate(
                candidate,
                prior=state,
                query=query,
                scale_cholesky=scale_cholesky,
                baseline_trace=baseline_trace,
                cost_weight=cost_weight,
                risk_weight=risk_weight,
                maximum_risk=maximum_risk,
                minimum_improvement=minimum_improvement,
                tolerance=tolerance,
                selection_tolerance=selection_tolerance,
                trivial_query=trivial_query,
            )
            for candidate in candidates
        )
        eligible = [
            evaluation
            for evaluation in evaluations
            if evaluation.status is QueryIdentifyingActionStatus.ELIGIBLE
        ]
        selected: QueryIdentifyingActionEvaluationV1 | None = None
        if trivial_query:
            status = QueryIdentifyingDesignStatus.TRIVIAL_QUERY
        elif eligible:
            best_objective = min(item.dimensionless_objective for item in eligible)
            tied = [
                item
                for item in eligible
                if item.dimensionless_objective
                <= best_objective + selection_tolerance
            ]
            selected = min(tied, key=lambda item: item.candidate.action_id)
            status = QueryIdentifyingDesignStatus.ACTION_SELECTED
        elif any(
            evaluation.status is QueryIdentifyingActionStatus.INSUFFICIENT_GAIN
            for evaluation in evaluations
        ):
            status = QueryIdentifyingDesignStatus.INSUFFICIENT_GAIN
        else:
            status = QueryIdentifyingDesignStatus.NO_ELIGIBLE_ACTION

        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="query-identifying action design metadata",
        )
        for name, value in (
            ("prior_state_precision", state.state_precision),
            ("prior_nuisance_precision", state.nuisance_precision),
            (
                "prior_state_nuisance_precision",
                state.state_nuisance_precision,
            ),
            ("query_jacobian", query),
            ("query_scale", query_scale),
            ("baseline_query_covariance", baseline_query),
            (
                "baseline_normalized_query_covariance",
                baseline_normalized,
            ),
        ):
            object.__setattr__(self, name, _immutable_float64(value))
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "cost_weight", cost_weight)
        object.__setattr__(self, "risk_weight", risk_weight)
        object.__setattr__(self, "maximum_risk", maximum_risk)
        object.__setattr__(self, "minimum_objective_improvement", minimum_improvement)
        object.__setattr__(self, "numerical_tolerance", tolerance)
        object.__setattr__(self, "selection_tolerance", selection_tolerance)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "baseline_normalized_query_trace", baseline_trace)
        object.__setattr__(self, "evaluations", evaluations)
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "selected_action_id",
            None if selected is None else selected.candidate.action_id,
        )
        object.__setattr__(
            self,
            "selected_candidate_id",
            None if selected is None else selected.candidate.artifact_id,
        )
        object.__setattr__(
            self,
            "selected_evaluation_id",
            None if selected is None else selected.artifact_id,
        )

        expected_id = cast(str, content_id(self.descriptor()))
        supplied_id = self.artifact_id
        if supplied_id is not None:
            supplied_id = cast(
                str,
                literal_lower_hex(
                    supplied_id,
                    name="artifact_id",
                    lengths={64},
                ),
            )
            if supplied_id != expected_id:
                raise ValueError(
                    "query-identifying design artifact_id does not match content"
                )
        object.__setattr__(self, "artifact_id", expected_id)

    @staticmethod
    def _evaluate_candidate(
        candidate: QueryIdentifyingActionCandidateV1,
        *,
        prior: NuisanceAwareInformationState,
        query: np.ndarray,
        scale_cholesky: np.ndarray,
        baseline_trace: float,
        cost_weight: float,
        risk_weight: float,
        maximum_risk: float,
        minimum_improvement: float,
        tolerance: float,
        selection_tolerance: float,
        trivial_query: bool,
    ) -> QueryIdentifyingActionEvaluationV1:
        updated = prior.observation_information_gain(
            candidate.state_jacobian,
            candidate.nuisance_jacobian,
            candidate.observation_covariance,
            reliability=candidate.reliability_vector,
        )
        ideal_state = prior.add_observation(
            candidate.state_jacobian,
            np.zeros_like(candidate.nuisance_jacobian),
            candidate.observation_covariance,
            reliability=candidate.reliability_vector,
        )
        posterior_query = _query_covariance(updated.updated_state, query)
        ideal_query = _query_covariance(ideal_state, query)
        normalized = _normalized_covariance(posterior_query, scale_cholesky)
        normalized_ideal = _normalized_covariance(ideal_query, scale_cholesky)
        posterior_trace = float(np.trace(normalized))
        ideal_trace = float(np.trace(normalized_ideal))
        if posterior_trace < -tolerance or ideal_trace < -tolerance:
            raise ValueError("posterior normalized query covariance is not positive")
        posterior_trace = max(0.0, posterior_trace)
        ideal_trace = max(0.0, ideal_trace)
        reduction = baseline_trace - posterior_trace
        ideal_reduction = baseline_trace - ideal_trace
        if reduction < -tolerance or ideal_reduction < -tolerance:
            raise ValueError("prospective conditioning increased query covariance")
        reduction = max(0.0, reduction)
        ideal_reduction = max(0.0, ideal_reduction)
        nuisance_effect = ideal_reduction - reduction
        if not np.isfinite(nuisance_effect):
            raise ValueError("nuisance trace effect is not finite")
        maximum_eigenvalue = float(np.linalg.eigvalsh(normalized)[-1])
        maximum_eigenvalue = max(0.0, maximum_eigenvalue)
        objective = (
            posterior_trace
            + cost_weight * candidate.dimensionless_cost
            + risk_weight * candidate.dimensionless_risk
        )
        improvement = baseline_trace - objective

        if trivial_query:
            status = QueryIdentifyingActionStatus.TRIVIAL_QUERY
        elif not candidate.safety_admissible:
            status = QueryIdentifyingActionStatus.SAFETY_REJECTED
        elif candidate.dimensionless_risk > maximum_risk + selection_tolerance:
            status = QueryIdentifyingActionStatus.RISK_REJECTED
        elif improvement + selection_tolerance < minimum_improvement:
            status = QueryIdentifyingActionStatus.INSUFFICIENT_GAIN
        else:
            status = QueryIdentifyingActionStatus.ELIGIBLE

        return QueryIdentifyingActionEvaluationV1(
            candidate=candidate,
            status=status,
            posterior_query_covariance=posterior_query,
            ideal_posterior_query_covariance=ideal_query,
            normalized_query_covariance=normalized,
            normalized_ideal_query_covariance=normalized_ideal,
            baseline_normalized_query_trace=baseline_trace,
            posterior_normalized_query_trace=posterior_trace,
            ideal_normalized_query_trace=ideal_trace,
            query_trace_reduction=reduction,
            ideal_query_trace_reduction=ideal_reduction,
            nuisance_trace_effect=nuisance_effect,
            normalized_query_maximum_eigenvalue=maximum_eigenvalue,
            marginal_state_information_gain_nats=updated.mutual_information_nats,
            dimensionless_objective=objective,
            objective_improvement=improvement,
        )

    @property
    def action_selected(self) -> bool:
        return self.status is QueryIdentifyingDesignStatus.ACTION_SELECTED

    @property
    def no_action_recommended(self) -> bool:
        return not self.action_selected

    @property
    def selected_evaluation(self) -> QueryIdentifyingActionEvaluationV1 | None:
        selected_id = self.selected_evaluation_id
        if selected_id is None:
            return None
        return next(
            item for item in self.evaluations if item.artifact_id == selected_id
        )

    def descriptor(self) -> dict[str, object]:
        prior_state = NuisanceAwareInformationState(
            state_precision=self.prior_state_precision,
            nuisance_precision=self.prior_nuisance_precision,
            state_nuisance_precision=self.prior_state_nuisance_precision,
        )
        return {
            "schema": QUERY_IDENTIFYING_ACTION_DESIGN_SCHEMA,
            "schema_version": QUERY_IDENTIFYING_ACTION_DESIGN_VERSION,
            "semantics": QUERY_IDENTIFYING_ACTION_DESIGN_SEMANTICS,
            "prior_belief_id": self.prior_belief_id,
            "query_id": self.query_id,
            "query_scale_id": self.query_scale_id,
            "protocol_id": self.protocol_id,
            "prior_information_state": _information_state_record(prior_state),
            "query_jacobian": _array_record(self.query_jacobian),
            "query_scale": _array_record(self.query_scale),
            "baseline_query_covariance": _array_record(
                self.baseline_query_covariance
            ),
            "baseline_normalized_query_covariance": _array_record(
                self.baseline_normalized_query_covariance
            ),
            "baseline_normalized_query_trace": (
                self.baseline_normalized_query_trace
            ),
            "cost_weight": self.cost_weight,
            "risk_weight": self.risk_weight,
            "maximum_risk": self.maximum_risk,
            "minimum_objective_improvement": (
                self.minimum_objective_improvement
            ),
            "numerical_tolerance": self.numerical_tolerance,
            "selection_tolerance": self.selection_tolerance,
            "candidate_ids": [item.artifact_id for item in self.candidates],
            "evaluations": [item.to_record() for item in self.evaluations],
            "status": self.status.value,
            "selected_action_id": self.selected_action_id,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_evaluation_id": self.selected_evaluation_id,
            "metadata": plain_json(self.metadata),
            "claim_boundary": QUERY_IDENTIFYING_ACTION_DESIGN_CLAIM_BOUNDARY,
        }

    def summary(self) -> dict[str, object]:
        return {
            "schema": QUERY_IDENTIFYING_ACTION_DESIGN_SCHEMA,
            "schema_version": QUERY_IDENTIFYING_ACTION_DESIGN_VERSION,
            "artifact_id": self.artifact_id,
            "status": self.status.value,
            "action_selected": self.action_selected,
            "no_action_recommended": self.no_action_recommended,
            "selected_action_id": self.selected_action_id,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_evaluation_id": self.selected_evaluation_id,
            "baseline_normalized_query_trace": (
                self.baseline_normalized_query_trace
            ),
            "evaluations": [item.summary() for item in self.evaluations],
            "claim_boundary": QUERY_IDENTIFYING_ACTION_DESIGN_CLAIM_BOUNDARY,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}


__all__ = [
    "QUERY_IDENTIFYING_ACTION_DESIGN_CLAIM_BOUNDARY",
    "QUERY_IDENTIFYING_ACTION_DESIGN_SCHEMA",
    "QUERY_IDENTIFYING_ACTION_DESIGN_SEMANTICS",
    "QUERY_IDENTIFYING_ACTION_DESIGN_VERSION",
    "QUERY_IDENTIFYING_ACTION_EVALUATION_SCHEMA",
    "QUERY_IDENTIFYING_ACTION_EVALUATION_VERSION",
    "QUERY_IDENTIFYING_ACTION_SCHEMA",
    "QUERY_IDENTIFYING_ACTION_VERSION",
    "QueryIdentifyingActionCandidateV1",
    "QueryIdentifyingActionEvaluationV1",
    "QueryIdentifyingActionStatus",
    "QueryIdentifyingActionDesignV1",
    "QueryIdentifyingDesignStatus",
]
