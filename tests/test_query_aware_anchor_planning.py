from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin.nuisance_aware_information import (
    NuisanceAwareInformationState,
)
from bayesian_phystwin.query_aware_anchor_planning import (
    QueryAwareAnchorSelection,
    greedy_query_aware_selection,
    query_covariance,
    query_variance_trace,
)


def _prior(
    *,
    state_dimension: int = 2,
    nuisance_precision: float = 1.0,
) -> NuisanceAwareInformationState:
    return NuisanceAwareInformationState.from_independent_priors(
        np.eye(state_dimension),
        np.asarray([[nuisance_precision]], dtype=np.float64),
    )


def _valid_selection() -> QueryAwareAnchorSelection:
    return QueryAwareAnchorSelection(
        selected_indices=np.asarray([0], dtype=np.int64),
        query_trace_reductions=np.asarray([0.5], dtype=np.float64),
        score_per_cost=np.asarray([0.25], dtype=np.float64),
        selected_costs=np.asarray([2.0], dtype=np.float64),
        initial_query_variance_trace=1.0,
        final_query_variance_trace=0.5,
        final_state=_prior(state_dimension=1),
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


def test_query_and_selection_arrays_are_irreversibly_immutable() -> None:
    covariance = query_covariance(_prior(state_dimension=1), np.asarray([[1.0]]))
    selection = _valid_selection()

    for array in (
        covariance,
        selection.selected_indices,
        selection.query_trace_reductions,
        selection.score_per_cost,
        selection.selected_costs,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_selection_contract_checks_derived_diagnostics() -> None:
    selection = _valid_selection()
    assert selection.total_cost == pytest.approx(2.0)
    assert selection.total_query_trace_reduction == pytest.approx(0.5)

    with pytest.raises(ValueError, match="score, cost"):
        replace(selection, score_per_cost=np.asarray([0.5]))
    with pytest.raises(ValueError, match="trace change"):
        replace(selection, final_query_variance_trace=0.4)


def test_selection_contract_rejects_invalid_state_and_indices() -> None:
    selection = _valid_selection()
    with pytest.raises(TypeError, match="final_state"):
        replace(selection, final_state=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="integer vector"):
        replace(selection, selected_indices=np.asarray([0.5]))
    with pytest.raises(ValueError, match="unique"):
        replace(
            selection,
            selected_indices=np.asarray([0, 0]),
            query_trace_reductions=np.asarray([0.25, 0.25]),
            score_per_cost=np.asarray([0.125, 0.125]),
            selected_costs=np.asarray([2.0, 2.0]),
        )


def test_planner_ties_break_by_original_index() -> None:
    selection = greedy_query_aware_selection(
        NuisanceAwareInformationState.from_independent_priors(np.eye(1)),
        np.asarray([[1.0]]),
        [np.asarray([[1.0]]), np.asarray([[1.0]])],
        [None, None],
        [np.eye(1), np.eye(1)],
        count=1,
    )
    assert selection.selected_indices.tolist() == [0]


def test_zero_count_and_empty_candidate_set_are_valid() -> None:
    prior = NuisanceAwareInformationState.from_independent_priors(np.eye(1))
    zero = greedy_query_aware_selection(
        prior,
        np.asarray([[1.0]]),
        [np.asarray([[1.0]])],
        [None],
        [np.eye(1)],
        count=0,
    )
    empty = greedy_query_aware_selection(
        prior,
        np.asarray([[1.0]]),
        [],
        [],
        [],
        count=2,
    )
    assert zero.selected_indices.size == 0
    assert empty.selected_indices.size == 0
    assert zero.final_state is prior
    assert empty.final_state is prior


def test_planner_rejects_wrong_prior_and_unhashable_groups() -> None:
    with pytest.raises(TypeError, match="prior"):
        greedy_query_aware_selection(
            object(),  # type: ignore[arg-type]
            np.asarray([[1.0]]),
            [],
            [],
            [],
            count=0,
        )
    with pytest.raises(ValueError, match="hashable"):
        greedy_query_aware_selection(
            NuisanceAwareInformationState.from_independent_priors(np.eye(1)),
            np.asarray([[1.0]]),
            [np.asarray([[1.0]])],
            [None],
            [np.eye(1)],
            dependence_groups=[[]],  # type: ignore[list-item]
            count=1,
        )


@pytest.mark.parametrize(
    ("query", "message"),
    [
        (np.asarray([1.0]), "matrix"),
        (np.asarray([[0.0]]), "nonzero"),
        (np.asarray([[1.0, 0.0]]), "state dimension"),
    ],
)
def test_query_covariance_rejects_invalid_query(
    query: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        query_covariance(_prior(state_dimension=1), query)


def test_planner_rejects_boolean_cost_count_and_wrong_candidate_counts() -> None:
    prior = NuisanceAwareInformationState.from_independent_priors(np.eye(1))
    with pytest.raises(ValueError, match="numeric vector"):
        greedy_query_aware_selection(
            prior,
            np.asarray([[1.0]]),
            [np.asarray([[1.0]])],
            [None],
            [np.eye(1)],
            costs=[True],
            count=1,
        )
    with pytest.raises(ValueError, match="nonnegative integer"):
        greedy_query_aware_selection(
            prior,
            np.asarray([[1.0]]),
            [],
            [],
            [],
            count=True,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="candidate input counts"):
        greedy_query_aware_selection(
            prior,
            np.asarray([[1.0]]),
            [np.asarray([[1.0]])],
            [],
            [np.eye(1)],
            count=1,
        )
