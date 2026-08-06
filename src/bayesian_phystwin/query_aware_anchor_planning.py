"""Query-aware active-anchor planning under explicit nuisance uncertainty.

The existing nuisance-aware information state scores observations by their
marginal information about the complete physical state. Physical experiments,
however, are usually judged through a lower-dimensional query such as an
endpoint, contact location, or future tracked point. This module selects
candidate anchors by the expected reduction of that declared query covariance
after nuisance marginalization.

The planner is deliberately conservative. Candidate costs are applied before
tie-breaking, and candidates carrying the same non-``None`` dependence-group
identifier are mutually exclusive. This prevents several near-duplicate
measurements from being treated as independent anchors.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np

from .nuisance_aware_information import NuisanceAwareInformationState

_NUMERICAL_TOLERANCE = 1e-12


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a nonnegative integer")
    result = int(value)
    _require(result >= 0, f"{name} must be a nonnegative integer")
    return result


def _finite_matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    _require(result.ndim == 2, f"{name} must be a matrix")
    _require(np.all(np.isfinite(result)), f"{name} contains non-finite values")
    return result


def _marginal_state_covariance(
    state: NuisanceAwareInformationState,
) -> np.ndarray:
    precision = state.marginal_state_precision()
    try:
        cholesky = np.linalg.cholesky(precision)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "marginal state precision must be positive definite"
        ) from error
    identity = np.eye(state.state_dimension, dtype=np.float64)
    lower_solution = np.linalg.solve(cholesky, identity)
    covariance = np.linalg.solve(cholesky.T, lower_solution)
    covariance = 0.5 * (covariance + covariance.T)
    _require(
        np.all(np.isfinite(covariance)),
        "marginal state covariance contains non-finite values",
    )
    return covariance


def query_covariance(
    state: NuisanceAwareInformationState,
    query_jacobian: np.ndarray,
) -> np.ndarray:
    """Return covariance of a declared linear physical query.

    ``query_jacobian`` maps physical-state coefficients to query coordinates.
    Nuisance coefficients are marginalized before the query covariance is
    formed. The returned array is a detached read-only value.
    """

    query = _finite_matrix(query_jacobian, name="query_jacobian")
    _require(query.shape[0] >= 1, "query_jacobian must contain at least one row")
    _require(
        query.shape[1] == state.state_dimension,
        "query_jacobian state dimension changed",
    )
    _require(
        np.linalg.norm(query) > 0.0,
        "query_jacobian must contain a nonzero query direction",
    )
    covariance = query @ _marginal_state_covariance(state) @ query.T
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    scale = max(1.0, float(np.linalg.norm(covariance, ord=2)))
    _require(
        float(np.min(eigenvalues)) >= -_NUMERICAL_TOLERANCE * scale,
        "query covariance is not positive semidefinite",
    )
    covariance.setflags(write=False)
    return covariance


def query_variance_trace(
    state: NuisanceAwareInformationState,
    query_jacobian: np.ndarray,
) -> float:
    """Return total variance of the declared linear physical query."""

    return float(np.trace(query_covariance(state, query_jacobian)))


@dataclass(frozen=True)
class QueryAwareAnchorSelection:
    """Deterministic greedy anchor plan for one declared physical query."""

    selected_indices: np.ndarray
    query_trace_reductions: np.ndarray
    score_per_cost: np.ndarray
    selected_costs: np.ndarray
    initial_query_variance_trace: float
    final_query_variance_trace: float
    final_state: NuisanceAwareInformationState

    def __post_init__(self) -> None:
        indices = np.asarray(self.selected_indices, dtype=np.int64).copy()
        reductions = np.asarray(
            self.query_trace_reductions,
            dtype=np.float64,
        ).copy()
        scores = np.asarray(self.score_per_cost, dtype=np.float64).copy()
        costs = np.asarray(self.selected_costs, dtype=np.float64).copy()
        _require(indices.ndim == 1, "selected_indices must be a vector")
        _require(
            reductions.shape == indices.shape
            and scores.shape == indices.shape
            and costs.shape == indices.shape,
            "selection diagnostic counts differ",
        )
        _require(np.all(indices >= 0), "selected_indices must be nonnegative")
        _require(
            len(np.unique(indices)) == len(indices),
            "selected_indices must be unique",
        )
        _require(
            np.all(np.isfinite(reductions)) and np.all(reductions >= 0.0),
            "query trace reductions must be finite and nonnegative",
        )
        _require(
            np.all(np.isfinite(scores)) and np.all(scores >= 0.0),
            "score_per_cost must be finite and nonnegative",
        )
        _require(
            np.all(np.isfinite(costs)) and np.all(costs > 0.0),
            "selected costs must be finite and positive",
        )
        _require(
            np.isfinite(self.initial_query_variance_trace)
            and self.initial_query_variance_trace >= 0.0,
            "initial query variance trace must be finite and nonnegative",
        )
        _require(
            np.isfinite(self.final_query_variance_trace)
            and self.final_query_variance_trace >= 0.0,
            "final query variance trace must be finite and nonnegative",
        )
        scale = max(1.0, self.initial_query_variance_trace)
        _require(
            self.final_query_variance_trace
            <= self.initial_query_variance_trace + _NUMERICAL_TOLERANCE * scale,
            "query variance increased after anchor selection",
        )
        for name, value in (
            ("selected_indices", indices),
            ("query_trace_reductions", reductions),
            ("score_per_cost", scores),
            ("selected_costs", costs),
        ):
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def total_cost(self) -> float:
        """Return total declared acquisition cost of selected anchors."""

        return float(np.sum(self.selected_costs))

    @property
    def total_query_trace_reduction(self) -> float:
        """Return total expected reduction in query covariance trace."""

        return self.initial_query_variance_trace - self.final_query_variance_trace


def greedy_query_aware_selection(
    prior: NuisanceAwareInformationState,
    query_jacobian: np.ndarray,
    state_jacobians: Sequence[np.ndarray],
    nuisance_jacobians: Sequence[np.ndarray | None],
    observation_covariances: Sequence[np.ndarray],
    *,
    reliabilities: Sequence[float | np.ndarray] | None = None,
    costs: Sequence[float] | None = None,
    dependence_groups: Sequence[Hashable | None] | None = None,
    count: int,
    minimum_trace_reduction: float = 0.0,
) -> QueryAwareAnchorSelection:
    """Greedily choose anchors by query-variance reduction per unit cost.

    Candidate observations are evaluated through
    :meth:`NuisanceAwareInformationState.add_observation`, so shared gauge,
    spatial-bias, timing, or provider nuisance variables are marginalized rather
    than silently folded into independent local noise. Ties are broken by the
    lowest original candidate index.

    A non-``None`` dependence-group identifier is exclusive: after one member
    is selected, all remaining members of that group are removed. Use this for
    duplicate depth samples, repeated views sharing one acquisition, or other
    candidates that must not be counted as independent anchors.
    """

    query = _finite_matrix(query_jacobian, name="query_jacobian")
    _require(
        query.shape[1] == prior.state_dimension,
        "query_jacobian state dimension changed",
    )
    candidate_count = len(state_jacobians)
    _require(
        len(nuisance_jacobians) == candidate_count
        and len(observation_covariances) == candidate_count,
        "candidate input counts differ",
    )
    selection_count = _nonnegative_integer(count, name="count")
    _require(
        np.isfinite(minimum_trace_reduction)
        and minimum_trace_reduction >= 0.0,
        "minimum_trace_reduction must be finite and nonnegative",
    )

    if reliabilities is None:
        reliability_values: Sequence[float | np.ndarray] = [1.0] * candidate_count
    else:
        _require(
            len(reliabilities) == candidate_count,
            "reliability candidate count differs",
        )
        reliability_values = reliabilities

    if costs is None:
        cost_values = np.ones(candidate_count, dtype=np.float64)
    else:
        cost_values = np.asarray(costs, dtype=np.float64)
        _require(
            cost_values.shape == (candidate_count,),
            "cost candidate count differs",
        )
        _require(
            np.all(np.isfinite(cost_values)) and np.all(cost_values > 0.0),
            "candidate costs must be finite and positive",
        )

    if dependence_groups is None:
        group_values: tuple[Hashable | None, ...] = (None,) * candidate_count
    else:
        _require(
            len(dependence_groups) == candidate_count,
            "dependence-group candidate count differs",
        )
        group_values = tuple(dependence_groups)
        for group in group_values:
            if group is not None:
                try:
                    hash(group)
                except TypeError as error:
                    raise ValueError(
                        "dependence-group identifiers must be hashable"
                    ) from error

    state = prior
    current_trace = query_variance_trace(state, query)
    initial_trace = current_trace
    remaining = set(range(candidate_count))
    selected: list[int] = []
    reductions: list[float] = []
    scores: list[float] = []
    selected_costs: list[float] = []

    while remaining and len(selected) < selection_count:
        evaluations: dict[int, tuple[NuisanceAwareInformationState, float, float]] = {}
        for candidate in sorted(remaining):
            updated = state.add_observation(
                state_jacobians[candidate],
                nuisance_jacobians[candidate],
                observation_covariances[candidate],
                reliability=reliability_values[candidate],
            )
            updated_trace = query_variance_trace(updated, query)
            reduction = current_trace - updated_trace
            scale = max(1.0, current_trace)
            if reduction < -_NUMERICAL_TOLERANCE * scale:
                raise RuntimeError(
                    "query variance increased after a candidate observation"
                )
            reduction = max(reduction, 0.0)
            score = reduction / float(cost_values[candidate])
            evaluations[candidate] = (updated, reduction, score)

        best_score = max(evaluation[2] for evaluation in evaluations.values())
        tied = [
            candidate
            for candidate, evaluation in evaluations.items()
            if np.isclose(
                evaluation[2],
                best_score,
                rtol=0.0,
                atol=_NUMERICAL_TOLERANCE,
            )
        ]
        chosen = min(tied)
        chosen_state, chosen_reduction, chosen_score = evaluations[chosen]
        if chosen_reduction <= minimum_trace_reduction:
            break

        selected.append(chosen)
        reductions.append(chosen_reduction)
        scores.append(chosen_score)
        selected_costs.append(float(cost_values[chosen]))
        state = chosen_state
        current_trace = query_variance_trace(state, query)

        chosen_group = group_values[chosen]
        if chosen_group is None:
            remaining.remove(chosen)
        else:
            remaining = {
                candidate
                for candidate in remaining
                if group_values[candidate] != chosen_group
            }

    return QueryAwareAnchorSelection(
        selected_indices=np.asarray(selected, dtype=np.int64),
        query_trace_reductions=np.asarray(reductions, dtype=np.float64),
        score_per_cost=np.asarray(scores, dtype=np.float64),
        selected_costs=np.asarray(selected_costs, dtype=np.float64),
        initial_query_variance_trace=initial_trace,
        final_query_variance_trace=current_trace,
        final_state=state,
    )
