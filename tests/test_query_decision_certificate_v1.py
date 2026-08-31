from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from bayesian_phystwin.query_decision_certificate_v1 import (
    query_decision_certificate,
)


def _enumerated_worst_case_regret(
    prior: np.ndarray,
    quotient: np.ndarray,
    classes: np.ndarray,
    losses: np.ndarray,
) -> np.ndarray:
    members = [
        np.flatnonzero((classes == class_id) & (prior > 0.0))
        for class_id in range(quotient.size)
    ]
    positive_classes = np.flatnonzero(quotient > 0.0)
    selected_rosters = [members[class_id] for class_id in positive_classes]
    regrets: list[np.ndarray] = []
    for selected in product(*selected_rosters):
        expected_loss = np.zeros(losses.shape[1], dtype=np.float64)
        for roster_index, hypothesis_id in enumerate(selected):
            class_id = int(positive_classes[roster_index])
            expected_loss += quotient[class_id] * losses[hypothesis_id]
        regrets.append(expected_loss - np.min(expected_loss))
    return np.max(np.stack(regrets, axis=0), axis=0)


def test_exact_worst_case_regret_matches_vertex_enumeration() -> None:
    prior = np.array([0.1, 0.2, 0.3, 0.4])
    quotient = np.array([0.6, 0.4])
    classes = np.array([0, 0, 1, 1])
    losses = np.array(
        [
            [0.0, 2.0, 3.0],
            [1.0, 0.5, 2.0],
            [0.2, 1.0, 0.0],
            [2.0, 0.1, 0.4],
        ]
    )

    result = query_decision_certificate(prior, quotient, classes, losses)
    expected = _enumerated_worst_case_regret(
        prior,
        quotient,
        classes,
        losses,
    )

    np.testing.assert_allclose(result.worst_case_regret, expected, atol=1e-12)
    assert result.minimax_action_index == int(np.argmin(expected))


def test_latent_state_can_be_ambiguous_while_decision_is_robust() -> None:
    prior = np.array([0.2, 0.3, 0.1, 0.4])
    quotient = np.array([0.45, 0.55])
    classes = np.array([0, 0, 1, 1])
    losses = np.array(
        [
            [0.0, 2.0],
            [0.1, 1.5],
            [0.2, 2.5],
            [0.0, 1.8],
        ]
    )

    result = query_decision_certificate(prior, quotient, classes, losses)

    assert result.minimax_action_index == 0
    assert result.minimax_worst_case_regret == pytest.approx(0.0)
    np.testing.assert_array_equal(
        result.robustly_optimal_action_mask,
        np.array([True, False]),
    )
    assert result.uniquely_robustly_optimal
    assert result.uniquely_tolerance_identified


def test_certificate_depends_on_prior_support_not_positive_prior_magnitudes() -> None:
    prior_a = np.array([0.1, 0.2, 0.3, 0.4])
    prior_b = np.array([0.2, 0.1, 0.6, 0.1])
    quotient = np.array([0.6, 0.4])
    classes = np.array([0, 0, 1, 1])
    losses = np.array(
        [
            [0.0, 2.0, 3.0],
            [1.0, 0.5, 2.0],
            [0.2, 1.0, 0.0],
            [2.0, 0.1, 0.4],
        ]
    )

    result_a = query_decision_certificate(prior_a, quotient, classes, losses)
    result_b = query_decision_certificate(prior_b, quotient, classes, losses)

    np.testing.assert_allclose(
        result_a.pairwise_worst_case_loss_gap,
        result_b.pairwise_worst_case_loss_gap,
    )
    np.testing.assert_allclose(
        result_a.worst_case_regret,
        result_b.worst_case_regret,
    )


def test_zero_prior_hypothesis_is_excluded_from_exact_supremum() -> None:
    prior = np.array([1.0, 0.0])
    quotient = np.array([1.0])
    classes = np.array([0, 0])
    losses = np.array(
        [
            [0.0, 1.0],
            [1000.0, -1000.0],
        ]
    )

    result = query_decision_certificate(prior, quotient, classes, losses)

    assert result.minimax_action_index == 0
    assert result.minimax_worst_case_regret == pytest.approx(0.0)
    np.testing.assert_array_equal(
        result.prior_support_mask,
        np.array([True, False]),
    )


def test_regret_tolerance_can_admit_multiple_actions_without_false_uniqueness() -> None:
    prior = np.array([0.5, 0.5])
    quotient = np.array([1.0])
    classes = np.array([0, 0])
    losses = np.array(
        [
            [0.0, 0.2],
            [0.2, 0.0],
        ]
    )

    result = query_decision_certificate(
        prior,
        quotient,
        classes,
        losses,
        regret_tolerance=0.2,
    )

    np.testing.assert_allclose(result.worst_case_regret, np.array([0.2, 0.2]))
    np.testing.assert_array_equal(
        result.tolerance_admissible_action_mask,
        np.array([True, True]),
    )
    assert result.has_tolerance_admissible_action
    assert not result.uniquely_tolerance_identified
    assert not result.has_robustly_optimal_action


def test_zero_mass_supported_class_does_not_change_certificate() -> None:
    prior = np.array([0.3, 0.3, 0.2, 0.2])
    quotient = np.array([1.0, 0.0])
    classes = np.array([0, 0, 1, 1])
    losses = np.array(
        [
            [0.0, 1.0],
            [0.1, 0.9],
            [1000.0, -1000.0],
            [-1000.0, 1000.0],
        ]
    )

    result = query_decision_certificate(prior, quotient, classes, losses)

    assert result.minimax_action_index == 0
    assert result.minimax_worst_case_regret == pytest.approx(0.0)


def test_returned_arrays_are_immutable() -> None:
    result = query_decision_certificate(
        np.array([0.5, 0.5]),
        np.array([1.0]),
        np.array([0, 0]),
        np.array([[0.0, 1.0], [0.5, 2.0]]),
    )
    with pytest.raises(ValueError):
        result.worst_case_regret[0] = 1.0
    with pytest.raises(ValueError):
        result.pairwise_worst_case_loss_gap[0, 1] = 1.0
    with pytest.raises(ValueError):
        result.prior_support_mask[0] = False


@pytest.mark.parametrize(
    ("prior", "quotient", "classes", "losses", "message"),
    [
        (
            np.array([0.5, 0.5]),
            np.array([0.5, 0.4]),
            np.array([0, 1]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            "sum to one",
        ),
        (
            np.array([0.5, 0.5]),
            np.array([0.5, 0.5]),
            np.array([0, 2]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            "contiguous",
        ),
        (
            np.array([0.5, 0.5]),
            np.array([1.0]),
            np.array([0, 0]),
            np.array([[0.0], [1.0]]),
            "at least two",
        ),
        (
            np.array([1.0, 0.0]),
            np.array([0.5, 0.5]),
            np.array([0, 1]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            "zero prior support",
        ),
    ],
)
def test_invalid_inputs_fail_closed(
    prior: np.ndarray,
    quotient: np.ndarray,
    classes: np.ndarray,
    losses: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        query_decision_certificate(prior, quotient, classes, losses)


def test_negative_tolerance_is_rejected() -> None:
    with pytest.raises(ValueError, match="regret_tolerance"):
        query_decision_certificate(
            np.array([0.5, 0.5]),
            np.array([1.0]),
            np.array([0, 0]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            regret_tolerance=-1.0,
        )
