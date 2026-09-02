import itertools

import numpy as np
import pytest

from bayesian_phystwin.active_decision_probe_v1 import (
    active_decision_probe_certificate,
    decision_probe_candidate,
)
from bayesian_phystwin.outcome_certified_decision_probe_v1 import (
    outcome_certified_decision_probe,
    select_minimum_cost_outcome_certified_probe,
)


def passive_pairwise_gap(prior, quotient, classes, losses):
    support = prior > 0.0
    action_count = losses.shape[1]
    result = np.zeros((action_count, action_count))
    for action in range(action_count):
        for benchmark in range(action_count):
            for class_id, mass in enumerate(quotient):
                members = np.flatnonzero(
                    (classes == class_id) & support
                )
                result[action, benchmark] += mass * np.max(
                    losses[members, action]
                    - losses[members, benchmark]
                )
    return result


def test_no_probe_matches_passive_pairwise_certificate():
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

    certificate = outcome_certified_decision_probe(
        prior,
        quotient,
        classes,
        np.ones((4, 1)),
        losses,
    )
    expected = passive_pairwise_gap(
        prior,
        quotient,
        classes,
        losses,
    )

    np.testing.assert_allclose(
        certificate.outcome_pairwise_worst_case_gap[0],
        expected,
        atol=2e-11,
    )
    expected_regret = np.maximum(np.max(expected, axis=1), 0.0)
    np.testing.assert_allclose(
        certificate.outcome_action_worst_case_regret[0],
        expected_regret,
        atol=2e-11,
    )
    assert certificate.outcome_minimax_action_index[0] == int(
        np.argmin(expected_regret)
    )


def test_perfect_decision_probe_certifies_each_outcome_without_state_identity():
    prior = np.full(4, 0.25)
    quotient = np.array([1.0])
    classes = np.zeros(4, dtype=int)
    likelihood = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    losses = np.array(
        [
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )

    certificate = outcome_certified_decision_probe(
        prior,
        quotient,
        classes,
        likelihood,
        losses,
    )

    assert certificate.outcome_minimax_action_index.tolist() == [0, 1]
    assert certificate.outcome_minimax_worst_case_regret.tolist() == [0.0, 0.0]
    assert certificate.all_reachable_outcomes_certified
    assert np.count_nonzero(likelihood[:, 0]) == 2
    assert np.count_nonzero(likelihood[:, 1]) == 2


def test_expected_regret_can_hide_a_rare_uncertified_outcome():
    prior = np.full(2, 0.5)
    quotient = np.array([1.0])
    classes = np.zeros(2, dtype=int)
    likelihood = np.array(
        [
            [0.99, 0.01],
            [0.0, 1.0],
        ]
    )
    losses = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    )

    ex_ante = active_decision_probe_certificate(
        prior,
        quotient,
        classes,
        likelihood,
        losses,
        regret_tolerance=0.05,
    )
    post_outcome = outcome_certified_decision_probe(
        prior,
        quotient,
        classes,
        likelihood,
        losses,
        regret_tolerance=0.05,
    )

    assert ex_ante.minimax_worst_case_regret == pytest.approx(0.01)
    assert ex_ante.has_tolerance_admissible_policy
    assert post_outcome.outcome_minimax_worst_case_regret[0] == pytest.approx(0.0)
    assert post_outcome.outcome_minimax_worst_case_regret[1] == pytest.approx(
        1.0,
        abs=2e-11,
    )
    assert post_outcome.outcome_tolerance_certified_mask.tolist() == [True, False]
    assert not post_outcome.all_reachable_outcomes_certified


def test_fractional_support_matches_complete_belief_vertices():
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
    certificate = outcome_certified_decision_probe(
        prior,
        quotient,
        classes,
        likelihood,
        losses,
    )

    vertices = []
    for member0 in (0, 1):
        for member1 in (2, 3):
            q = np.zeros(4)
            q[member0] = quotient[0]
            q[member1] = quotient[1]
            vertices.append(q)
    brute = np.zeros((2, 2, 2))
    for outcome in range(2):
        for action in range(2):
            for benchmark in range(2):
                values = []
                for q in vertices:
                    denominator = float(q @ likelihood[:, outcome])
                    if denominator <= 0.0:
                        continue
                    difference = (
                        losses[:, outcome, action]
                        - losses[:, outcome, benchmark]
                    )
                    values.append(
                        float(q @ (likelihood[:, outcome] * difference))
                        / denominator
                    )
                brute[outcome, action, benchmark] = max(values)

    np.testing.assert_allclose(
        certificate.outcome_pairwise_worst_case_gap,
        brute,
        atol=2e-11,
    )


def test_selector_chooses_cheapest_probe_certified_for_every_outcome():
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
    probes = (
        decision_probe_candidate(
            "no_probe",
            0.0,
            np.ones((4, 1)),
            losses,
        ),
        decision_probe_candidate(
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
        ),
        decision_probe_candidate(
            "state_probe",
            4.0,
            np.eye(4),
            losses,
        ),
    )

    selection = select_minimum_cost_outcome_certified_probe(
        prior,
        quotient,
        classes,
        probes,
    )

    assert selection.selected_probe_name == "decision_probe"
    assert selection.admissible_probe_mask.tolist() == [False, True, True]


def test_unreachable_outcome_is_ignored_by_probe_selection():
    prior = np.full(2, 0.5)
    quotient = np.array([1.0])
    classes = np.zeros(2, dtype=int)
    likelihood = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )
    losses = np.array(
        [
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )

    certificate = outcome_certified_decision_probe(
        prior,
        quotient,
        classes,
        likelihood,
        losses,
    )

    assert certificate.reachable_outcome_mask.tolist() == [True, False]
    assert certificate.outcome_minimax_action_index.tolist() == [0, -1]
    assert certificate.all_reachable_outcomes_certified


@pytest.mark.parametrize(
    "root_tolerance",
    [0.0, -1.0, float("nan")],
)
def test_invalid_root_tolerance_fails_closed(root_tolerance):
    with pytest.raises(ValueError, match="root_tolerance"):
        outcome_certified_decision_probe(
            [1.0],
            [1.0],
            [0],
            [[1.0]],
            [[0.0, 1.0]],
            root_tolerance=root_tolerance,
        )
