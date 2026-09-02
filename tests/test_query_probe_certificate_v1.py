from __future__ import annotations

import itertools

import numpy as np
import pytest

from bayesian_phystwin.query_decision_certificate_v1 import (
    query_decision_certificate,
)
from bayesian_phystwin.query_probe_certificate_v1 import (
    act_probe_fallback_certificate,
    query_probe_certificate,
)


def _ambiguous_two_worlds():
    prior = np.asarray([0.5, 0.5])
    quotient = np.asarray([1.0])
    classes = np.asarray([0, 0])
    losses = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    return prior, quotient, classes, losses


def test_one_outcome_probe_reduces_exactly_to_direct_actions() -> None:
    prior = np.asarray([0.2, 0.3, 0.5])
    quotient = np.asarray([0.5, 0.5])
    classes = np.asarray([0, 0, 1])
    losses = np.asarray([[0.0, 2.0], [1.0, 0.5], [3.0, 0.0]])
    direct = query_decision_certificate(prior, quotient, classes, losses)
    probe = query_probe_certificate(
        prior,
        quotient,
        classes,
        losses,
        np.ones((3, 1)),
        probe_cost=0.3,
    )

    assert probe.contingent_action_indices.tolist() == [[0], [1]]
    np.testing.assert_allclose(
        probe.expected_action_loss_by_hypothesis_policy,
        losses,
    )
    np.testing.assert_allclose(
        probe.meta_loss_by_hypothesis_policy,
        losses + 0.3,
    )
    np.testing.assert_allclose(
        probe.policy_decision_certificate.worst_case_regret,
        direct.worst_case_regret,
    )


def test_perfect_probe_resolves_opposing_actions_in_common_union() -> None:
    prior, quotient, classes, losses = _ambiguous_two_worlds()
    result = act_probe_fallback_certificate(
        prior,
        quotient,
        classes,
        losses,
        [np.asarray([[1.0, 0.0], [0.0, 1.0]])],
        probe_costs=[0.05],
        fallback_action_index=0,
        regret_tolerance=0.05,
    )

    assert result.direct_decision_certificate.minimax_worst_case_regret == pytest.approx(
        1.0
    )
    assert result.route == "probe"
    assert result.selected_probe_index == 0
    assert result.selected_contingent_action_indices.tolist() == [0, 1]
    assert result.selected_worst_case_regret == pytest.approx(0.05)


def test_probe_cost_is_compared_against_direct_actions_not_added_to_internal_regret() -> None:
    prior, quotient, classes, losses = _ambiguous_two_worlds()
    result = act_probe_fallback_certificate(
        prior,
        quotient,
        classes,
        losses,
        [np.asarray([[1.0, 0.0], [0.0, 1.0]])],
        probe_costs=[0.6],
        fallback_action_index=0,
        regret_tolerance=0.5,
    )

    probe = result.probe_certificates[0]
    assert probe.minimax_within_probe_worst_case_regret == pytest.approx(0.0)
    assert result.meta_decision_certificate.minimax_worst_case_regret == pytest.approx(
        0.6
    )
    assert result.route == "fallback"
    assert not result.certified


def test_common_union_can_probe_even_when_direct_only_certificate_passes() -> None:
    prior = np.asarray([0.2, 0.4, 0.4])
    quotient = np.asarray([0.5, 0.5])
    classes = np.asarray([0, 1, 1])
    losses = np.asarray(
        [
            [0.0, 4.0],
            [0.0, 0.0],
            [2.0, 0.0],
        ]
    )
    direct = query_decision_certificate(prior, quotient, classes, losses)
    assert direct.has_robustly_optimal_action
    assert direct.minimax_action_index == 0

    perfect = np.eye(3)
    result = act_probe_fallback_certificate(
        prior,
        quotient,
        classes,
        losses,
        [perfect],
        fallback_action_index=0,
        regret_tolerance=0.0,
    )

    assert result.route == "probe"
    assert result.selected_contingent_action_indices.tolist() == [0, 0, 1]
    assert result.selected_worst_case_regret == pytest.approx(0.0)
    assert result.meta_decision_certificate.worst_case_regret[0] == pytest.approx(1.0)


def test_union_formula_matches_every_extreme_complete_lift() -> None:
    prior = np.asarray([0.25, 0.25, 0.25, 0.25])
    quotient = np.asarray([0.35, 0.65])
    classes = np.asarray([0, 0, 1, 1])
    losses = np.asarray(
        [
            [0.0, 1.2, 0.7],
            [1.1, 0.1, 0.5],
            [0.8, 0.4, 0.0],
            [0.2, 0.9, 0.6],
        ]
    )
    likelihoods = [
        np.asarray(
            [
                [0.85, 0.15],
                [0.25, 0.75],
                [0.55, 0.45],
                [0.10, 0.90],
            ]
        ),
        np.asarray(
            [
                [0.60, 0.40],
                [0.60, 0.40],
                [0.20, 0.80],
                [0.20, 0.80],
            ]
        ),
    ]
    result = act_probe_fallback_certificate(
        prior,
        quotient,
        classes,
        losses,
        likelihoods,
        probe_costs=[0.03, 0.08],
        fallback_action_index=0,
        regret_tolerance=10.0,
    )

    extreme_lifts = []
    for left, right in itertools.product((0, 1), (2, 3)):
        complete = np.zeros(4)
        complete[left] = quotient[0]
        complete[right] = quotient[1]
        extreme_lifts.append(complete)

    exhaustive = []
    for meta_index in range(result.meta_action_count):
        regrets = []
        for complete in extreme_lifts:
            risk = complete @ result.meta_loss_by_hypothesis
            regrets.append(float(risk[meta_index] - np.min(risk)))
        exhaustive.append(max(regrets))
    np.testing.assert_allclose(
        result.meta_decision_certificate.worst_case_regret,
        exhaustive,
        atol=1e-12,
        rtol=0.0,
    )


def test_uninformative_probe_can_only_add_exogenous_randomization() -> None:
    prior, quotient, classes, losses = _ambiguous_two_worlds()
    result = act_probe_fallback_certificate(
        prior,
        quotient,
        classes,
        losses,
        [np.full((2, 2), 0.5)],
        fallback_action_index=0,
        regret_tolerance=0.5,
    )

    assert result.route == "probe"
    assert result.selected_contingent_action_indices.tolist() == [0, 1]
    assert result.selected_worst_case_regret == pytest.approx(0.5)


def test_without_affordable_certificate_router_returns_registered_fallback() -> None:
    prior, quotient, classes, losses = _ambiguous_two_worlds()
    result = act_probe_fallback_certificate(
        prior,
        quotient,
        classes,
        losses,
        [],
        fallback_action_index=1,
        regret_tolerance=0.2,
    )
    assert result.route == "fallback"
    assert result.fallback_action_index == 1
    assert result.selected_meta_action_index is None

    permissive = act_probe_fallback_certificate(
        prior,
        quotient,
        classes,
        losses,
        [],
        fallback_action_index=1,
        regret_tolerance=1.0,
    )
    assert permissive.route == "act"
    assert permissive.selected_direct_action_index == 0


def test_random_small_unions_match_all_extreme_lifts() -> None:
    rng = np.random.default_rng(84531)
    for _ in range(20):
        losses = rng.normal(size=(4, 2))
        likelihood = rng.dirichlet(np.ones(2), size=4)
        prior = np.full(4, 0.25)
        quotient = np.asarray([0.35, 0.65])
        classes = np.asarray([0, 0, 1, 1])
        result = act_probe_fallback_certificate(
            prior,
            quotient,
            classes,
            losses,
            [likelihood],
            probe_costs=[0.1],
            fallback_action_index=0,
            regret_tolerance=10.0,
        )
        extreme_lifts = []
        for left, right in itertools.product((0, 1), (2, 3)):
            complete = np.zeros(4)
            complete[left] = quotient[0]
            complete[right] = quotient[1]
            extreme_lifts.append(complete)
        exhaustive = []
        for meta_index in range(result.meta_action_count):
            regrets = []
            for complete in extreme_lifts:
                risk = complete @ result.meta_loss_by_hypothesis
                regrets.append(float(risk[meta_index] - np.min(risk)))
            exhaustive.append(max(regrets))
        np.testing.assert_allclose(
            result.meta_decision_certificate.worst_case_regret,
            exhaustive,
            atol=1e-12,
            rtol=0.0,
        )


def test_validation_fails_closed() -> None:
    prior, quotient, classes, losses = _ambiguous_two_worlds()

    with pytest.raises(ValueError, match="row must sum to one"):
        query_probe_certificate(
            prior,
            quotient,
            classes,
            losses,
            np.asarray([[0.4, 0.4], [0.5, 0.5]]),
        )
    with pytest.raises(ValueError, match="exceeds cap"):
        query_probe_certificate(
            prior,
            quotient,
            classes,
            losses,
            np.full((2, 20), 0.05),
            max_policy_count=10,
        )
    with pytest.raises(ValueError, match="one entry per probe"):
        act_probe_fallback_certificate(
            prior,
            quotient,
            classes,
            losses,
            [np.full((2, 2), 0.5)],
            probe_costs=[],
            fallback_action_index=0,
        )
    with pytest.raises(ValueError, match="outside the direct action set"):
        act_probe_fallback_certificate(
            prior,
            quotient,
            classes,
            losses,
            [],
            fallback_action_index=3,
        )
    with pytest.raises(ValueError, match="meta-action count"):
        act_probe_fallback_certificate(
            prior,
            quotient,
            classes,
            losses,
            [np.full((2, 2), 0.5)],
            fallback_action_index=0,
            max_meta_action_count=3,
        )
