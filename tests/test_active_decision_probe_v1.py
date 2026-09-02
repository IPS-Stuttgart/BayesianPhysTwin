import itertools

import numpy as np
import pytest

from bayesian_phystwin.active_decision_probe_v1 import (
    active_decision_probe_certificate,
    decision_probe_candidate,
    select_minimum_cost_decision_probe,
)


def passive_minimax(prior, quotient, classes, losses):
    support = prior > 0
    action_count = losses.shape[1]
    regrets = []
    for action in range(action_count):
        pairwise = []
        for benchmark in range(action_count):
            value = 0.0
            for class_id, mass in enumerate(quotient):
                members = np.flatnonzero((classes == class_id) & support)
                value += mass * np.max(
                    losses[members, action] - losses[members, benchmark]
                )
            pairwise.append(value)
        regrets.append(max(pairwise))
    return min(regrets), int(np.argmin(regrets))


def test_no_probe_matches_passive_certificate():
    prior = np.array([0.2, 0.3, 0.1, 0.4])
    quotient = np.array([0.45, 0.55])
    classes = np.array([0, 0, 1, 1])
    losses = np.array(
        [
            [0.0, 1.0, 0.5],
            [0.3, 0.2, 0.6],
            [1.0, 0.0, 0.4],
            [0.7, 0.5, 0.0],
        ]
    )
    active = active_decision_probe_certificate(
        prior,
        quotient,
        classes,
        np.ones((4, 1)),
        losses,
    )
    expected_regret, expected_action = passive_minimax(prior, quotient, classes, losses)
    assert active.minimax_worst_case_regret == pytest.approx(expected_regret)
    assert active.minimax_terminal_policy.tolist() == [expected_action]


def test_perfect_decision_probe_removes_regret_without_identifying_state():
    prior = np.full(4, 0.25)
    quotient = np.array([1.0])
    classes = np.zeros(4, dtype=int)
    losses = np.array(
        [
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )
    no_probe = active_decision_probe_certificate(
        prior, quotient, classes, np.ones((4, 1)), losses
    )
    decision_probe = active_decision_probe_certificate(
        prior,
        quotient,
        classes,
        np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        ),
        losses,
    )
    assert no_probe.minimax_worst_case_regret == pytest.approx(1.0)
    assert decision_probe.minimax_worst_case_regret == pytest.approx(0.0)
    assert decision_probe.minimax_terminal_policy.tolist() == [0, 1]


def test_uninformative_probe_has_no_value():
    prior = np.full(4, 0.25)
    quotient = np.array([1.0])
    classes = np.zeros(4, dtype=int)
    losses = np.array(
        [
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )
    no_probe = active_decision_probe_certificate(
        prior, quotient, classes, np.ones((4, 1)), losses
    )
    uninformative = active_decision_probe_certificate(
        prior,
        quotient,
        classes,
        np.tile(np.array([1.0, 0.0]), (4, 1)),
        losses,
    )
    assert uninformative.minimax_worst_case_regret == pytest.approx(
        no_probe.minimax_worst_case_regret
    )


def test_closed_form_matches_complete_belief_vertices():
    prior = np.full(4, 0.25)
    quotient = np.array([0.4, 0.6])
    classes = np.array([0, 0, 1, 1])
    likelihood = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.7, 0.3],
            [0.1, 0.9],
        ]
    )
    losses = np.array(
        [
            [[0.0, 1.0], [0.2, 0.7]],
            [[0.5, 0.1], [0.8, 0.0]],
            [[0.3, 0.6], [0.0, 0.9]],
            [[0.9, 0.0], [0.4, 0.2]],
        ]
    )
    certificate = active_decision_probe_certificate(
        prior, quotient, classes, likelihood, losses
    )
    policies = list(itertools.product(range(2), repeat=2))
    vertices = []
    for member0 in (0, 1):
        for member1 in (2, 3):
            q = np.zeros(4)
            q[member0] = quotient[0]
            q[member1] = quotient[1]
            vertices.append(q)

    brute_regret = []
    for policy in policies:
        worst = 0.0
        for q in vertices:
            candidate_loss = sum(
                q[i]
                * sum(
                    likelihood[i, outcome] * losses[i, outcome, policy[outcome]]
                    for outcome in range(2)
                )
                for i in range(4)
            )
            best = min(
                sum(
                    q[i]
                    * sum(
                        likelihood[i, outcome] * losses[i, outcome, benchmark[outcome]]
                        for outcome in range(2)
                    )
                    for i in range(4)
                )
                for benchmark in policies
            )
            worst = max(worst, candidate_loss - best)
        brute_regret.append(worst)
    assert certificate.policy_worst_case_regret == pytest.approx(brute_regret)
    assert certificate.minimax_worst_case_regret == pytest.approx(min(brute_regret))


def test_selector_prefers_cheapest_certifying_probe():
    prior = np.full(4, 0.25)
    quotient = np.array([1.0])
    classes = np.zeros(4, dtype=int)
    losses = np.array(
        [
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )
    no_probe = decision_probe_candidate("no_probe", 0.0, np.ones((4, 1)), losses)
    decision_probe = decision_probe_candidate(
        "decision_probe",
        1.0,
        np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        ),
        losses,
    )
    state_probe = decision_probe_candidate("state_probe", 4.0, np.eye(4), losses)
    selection = select_minimum_cost_decision_probe(
        prior,
        quotient,
        classes,
        (no_probe, decision_probe, state_probe),
    )
    assert selection.selected_probe_name == "decision_probe"
    assert selection.fallback_required is False


@pytest.mark.parametrize(
    ("likelihood", "message"),
    [
        (np.array([[1.0], [1.0]]), "shape"),
        (np.array([[0.2, 0.2]]), "sum to one"),
    ],
)
def test_invalid_likelihood_fails_closed(likelihood, message):
    with pytest.raises(ValueError, match=message):
        active_decision_probe_certificate(
            [1.0],
            [1.0],
            [0],
            likelihood,
            [[0.0, 1.0]],
        )
