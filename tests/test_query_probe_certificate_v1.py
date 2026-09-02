from __future__ import annotations

import itertools

import numpy as np
import pytest

from bayesian_phystwin.query_decision_certificate_v1 import (
    query_decision_certificate,
)
from bayesian_phystwin.query_probe_certificate_v1 import (
    act_probe_fallback_decision,
    query_probe_certificate,
)


def _ambiguous_two_worlds(*, tolerance: float = 0.0):
    prior = np.asarray([0.5, 0.5])
    quotient = np.asarray([1.0])
    classes = np.asarray([0, 0])
    losses = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    direct = query_decision_certificate(
        prior,
        quotient,
        classes,
        losses,
        regret_tolerance=tolerance,
    )
    return prior, quotient, classes, losses, direct


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
    )

    assert probe.contingent_action_indices.tolist() == [[0], [1]]
    np.testing.assert_allclose(
        probe.policy_decision_certificate.worst_case_regret,
        direct.worst_case_regret,
    )
    assert probe.minimax_contingent_action_indices.tolist() == [
        direct.minimax_action_index
    ]
    assert probe.minimax_worst_case_regret == pytest.approx(
        direct.minimax_worst_case_regret
    )


def test_perfect_probe_resolves_opposing_actions_without_completing_state() -> None:
    prior, quotient, classes, losses, direct = _ambiguous_two_worlds()
    probe = query_probe_certificate(
        prior,
        quotient,
        classes,
        losses,
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        probe_cost=0.05,
    )

    assert direct.minimax_worst_case_regret == pytest.approx(1.0)
    assert not direct.has_tolerance_admissible_action
    assert probe.minimax_contingent_action_indices.tolist() == [0, 1]
    assert probe.minimax_worst_case_regret == pytest.approx(0.0)
    assert probe.total_regret_plus_cost == pytest.approx(0.05)


def test_formula_matches_exhaustive_extreme_complete_lifts() -> None:
    prior = np.asarray([0.25, 0.25, 0.5])
    quotient = np.asarray([0.6, 0.4])
    classes = np.asarray([0, 0, 1])
    losses = np.asarray(
        [
            [0.0, 1.2, 0.7],
            [1.1, 0.1, 0.5],
            [0.8, 0.4, 0.0],
        ]
    )
    likelihood = np.asarray(
        [
            [0.85, 0.15],
            [0.25, 0.75],
            [0.55, 0.45],
        ]
    )
    certificate = query_probe_certificate(
        prior,
        quotient,
        classes,
        losses,
        likelihood,
    )

    policy_loss = certificate.policy_loss_by_hypothesis
    extreme_lifts = (
        np.asarray([0.6, 0.0, 0.4]),
        np.asarray([0.0, 0.6, 0.4]),
    )
    exhaustive = []
    for policy_index in range(certificate.policy_count):
        regrets = []
        for complete in extreme_lifts:
            risk = complete @ policy_loss
            regrets.append(float(risk[policy_index] - np.min(risk)))
        exhaustive.append(max(regrets))

    np.testing.assert_allclose(
        certificate.policy_decision_certificate.worst_case_regret,
        exhaustive,
        atol=1e-12,
        rtol=0.0,
    )


def test_probe_likelihood_can_supply_randomization_without_information() -> None:
    prior, quotient, classes, losses, _ = _ambiguous_two_worlds()
    probe = query_probe_certificate(
        prior,
        quotient,
        classes,
        losses,
        np.full((2, 2), 0.5),
    )

    # The observation carries no state information, but policy [0, 1] uses its
    # exogenous randomness as a fair randomized action. This is why empirical
    # studies need an uninformative/scrambled-probe control.
    assert probe.minimax_contingent_action_indices.tolist() == [0, 1]
    assert probe.minimax_worst_case_regret == pytest.approx(0.5)


def test_act_probe_fallback_routes_in_priority_order() -> None:
    prior, quotient, classes, losses, direct = _ambiguous_two_worlds()
    perfect = query_probe_certificate(
        prior,
        quotient,
        classes,
        losses,
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        probe_cost=0.05,
    )

    probe_route = act_probe_fallback_decision(
        direct,
        [perfect],
        fallback_action_index=0,
        maximum_probe_total_value=0.1,
    )
    assert probe_route.route == "probe"
    assert probe_route.probe_index == 0
    assert probe_route.contingent_action_indices.tolist() == [0, 1]

    fallback_route = act_probe_fallback_decision(
        direct,
        [perfect],
        fallback_action_index=0,
        maximum_probe_total_value=0.01,
    )
    assert fallback_route.route == "fallback"
    assert fallback_route.fallback_action_index == 0

    _, _, _, _, direct_with_tolerance = _ambiguous_two_worlds(tolerance=1.0)
    act_route = act_probe_fallback_decision(
        direct_with_tolerance,
        [perfect],
        fallback_action_index=0,
        maximum_probe_total_value=0.1,
    )
    assert act_route.route == "act"
    assert act_route.direct_action_index == direct_with_tolerance.minimax_action_index


def test_validation_fails_closed() -> None:
    prior, quotient, classes, losses, direct = _ambiguous_two_worlds()

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
    with pytest.raises(ValueError, match="outside the direct action set"):
        act_probe_fallback_decision(
            direct,
            [],
            fallback_action_index=3,
            maximum_probe_total_value=0.0,
        )


def test_random_small_cases_match_all_extreme_lifts() -> None:
    rng = np.random.default_rng(84531)
    for _ in range(20):
        losses = rng.normal(size=(4, 2))
        likelihood = rng.dirichlet(np.ones(2), size=4)
        prior = np.full(4, 0.25)
        quotient = np.asarray([0.35, 0.65])
        classes = np.asarray([0, 0, 1, 1])
        certificate = query_probe_certificate(
            prior,
            quotient,
            classes,
            losses,
            likelihood,
        )
        policy_loss = certificate.policy_loss_by_hypothesis
        extreme_lifts = []
        for left, right in itertools.product((0, 1), (2, 3)):
            complete = np.zeros(4)
            complete[left] = quotient[0]
            complete[right] = quotient[1]
            extreme_lifts.append(complete)
        exhaustive = []
        for policy_index in range(certificate.policy_count):
            regrets = []
            for complete in extreme_lifts:
                risk = complete @ policy_loss
                regrets.append(float(risk[policy_index] - np.min(risk)))
            exhaustive.append(max(regrets))
        np.testing.assert_allclose(
            certificate.policy_decision_certificate.worst_case_regret,
            exhaustive,
            atol=1e-12,
            rtol=0.0,
        )
