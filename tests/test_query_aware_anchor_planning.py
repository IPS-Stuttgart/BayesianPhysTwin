import numpy as np
import pytest

from bayesian_phystwin.nuisance_aware_information import (
    NuisanceAwareInformationState,
)
from bayesian_phystwin.query_aware_anchor_planning import (
    greedy_query_aware_selection,
    query_variance_trace,
)


def _prior(*, state_dimension: int = 2, nuisance_precision: float = 1.0):
    return NuisanceAwareInformationState.from_independent_priors(
        np.eye(state_dimension),
        np.asarray([[nuisance_precision]], dtype=np.float64),
    )


def test_planner_prefers_query_relevant_anchor() -> None:
    selection = greedy_query_aware_selection(
        _prior(),
        np.asarray([[1.0, 0.0]]),
        [np.asarray([[0.0, 10.0]]), np.asarray([[2.0, 0.0]])],
        [np.zeros((1, 1)), np.zeros((1, 1))],
        [np.eye(1), np.eye(1)],
        count=1,
    )

    assert selection.selected_indices.tolist() == [1]
    assert selection.initial_query_variance_trace == pytest.approx(1.0)
    assert selection.final_query_variance_trace == pytest.approx(0.2)


def test_planner_marginalizes_weakly_constrained_timing_or_bias() -> None:
    selection = greedy_query_aware_selection(
        _prior(state_dimension=1, nuisance_precision=1e-8),
        np.asarray([[1.0]]),
        [np.asarray([[100.0]]), np.asarray([[2.0]])],
        [np.asarray([[100.0]]), np.asarray([[0.0]])],
        [np.eye(1), np.eye(1)],
        count=1,
    )

    assert selection.selected_indices.tolist() == [1]
    assert selection.final_query_variance_trace == pytest.approx(0.2)


def test_cost_normalization_changes_selection_order() -> None:
    selection = greedy_query_aware_selection(
        NuisanceAwareInformationState.from_independent_priors(np.eye(1)),
        np.asarray([[1.0]]),
        [np.asarray([[3.0]]), np.asarray([[1.0]])],
        [None, None],
        [np.eye(1), np.eye(1)],
        costs=[20.0, 1.0],
        count=1,
    )

    assert selection.selected_indices.tolist() == [1]
    assert selection.total_cost == pytest.approx(1.0)


def test_dependence_groups_are_mutually_exclusive() -> None:
    selection = greedy_query_aware_selection(
        NuisanceAwareInformationState.from_independent_priors(np.eye(1)),
        np.asarray([[1.0]]),
        [np.asarray([[4.0]]), np.asarray([[3.0]]), np.asarray([[1.0]])],
        [None, None, None],
        [np.eye(1), np.eye(1), np.eye(1)],
        dependence_groups=["same-depth-capture", "same-depth-capture", None],
        count=2,
    )

    assert selection.selected_indices.tolist() == [0, 2]
    assert 1 not in selection.selected_indices
    assert selection.final_query_variance_trace < selection.initial_query_variance_trace


def test_minimum_reduction_can_leave_plan_empty() -> None:
    prior = NuisanceAwareInformationState.from_independent_priors(np.eye(1))
    selection = greedy_query_aware_selection(
        prior,
        np.asarray([[1.0]]),
        [np.asarray([[1.0]])],
        [None],
        [np.eye(1)],
        count=1,
        minimum_trace_reduction=0.75,
    )

    assert selection.selected_indices.size == 0
    assert selection.final_state is prior
    assert query_variance_trace(selection.final_state, np.asarray([[1.0]])) == 1.0


def test_invalid_candidate_cost_is_rejected() -> None:
    with pytest.raises(ValueError, match="cost"):
        greedy_query_aware_selection(
            NuisanceAwareInformationState.from_independent_priors(np.eye(1)),
            np.asarray([[1.0]]),
            [np.asarray([[1.0]])],
            [None],
            [np.eye(1)],
            costs=[0.0],
            count=1,
        )
