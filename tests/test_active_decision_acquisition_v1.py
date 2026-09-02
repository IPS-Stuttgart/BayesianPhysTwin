from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from bayesian_phystwin.active_decision_acquisition_v1 import (
    ACTIVE_DECISION_ACQUISITION_CLAIM_BOUNDARY,
    DeterministicDecisionProbeV1,
    conditioned_query_decision_certificate,
    evaluate_deterministic_probe,
    minimum_cost_global_decision_identifying_probe_set,
    synthesize_minimax_active_decision_policy,
)


def _enumerate_conditioned_pairwise_gap(
    prior: np.ndarray,
    quotient: np.ndarray,
    classes: np.ndarray,
    losses: np.ndarray,
    consistent: np.ndarray,
    radii: np.ndarray | None = None,
) -> np.ndarray:
    """Enumerate extreme feasible conditioned beliefs for small tests."""

    if radii is None:
        radii = np.zeros_like(losses)
    class_options: list[list[int | None]] = []
    for class_id, mass in enumerate(quotient):
        supported = np.flatnonzero((classes == class_id) & (prior > 0.0))
        retained = np.flatnonzero(
            (classes == class_id) & (prior > 0.0) & consistent
        )
        if mass == 0.0 or retained.size == 0:
            class_options.append([None])
        elif retained.size == supported.size:
            class_options.append([int(index) for index in retained])
        else:
            class_options.append([None, *(int(index) for index in retained)])

    action_count = losses.shape[1]
    pairwise = np.full((action_count, action_count), -np.inf)
    np.fill_diagonal(pairwise, 0.0)
    feasible = 0
    for selected in product(*class_options):
        event_mass = sum(
            quotient[class_id]
            for class_id, hypothesis in enumerate(selected)
            if hypothesis is not None
        )
        if event_mass <= 1e-12:
            continue
        feasible += 1
        posterior = np.zeros(prior.size, dtype=np.float64)
        for class_id, hypothesis in enumerate(selected):
            if hypothesis is not None:
                posterior[hypothesis] = quotient[class_id] / event_mass
        for action in range(action_count):
            for benchmark in range(action_count):
                if action == benchmark:
                    continue
                values = (
                    losses[:, action]
                    - losses[:, benchmark]
                    + radii[:, action]
                    + radii[:, benchmark]
                )
                pairwise[action, benchmark] = max(
                    pairwise[action, benchmark],
                    float(posterior @ values),
                )
    if feasible == 0:
        raise ValueError("test event is impossible")
    np.fill_diagonal(pairwise, 0.0)
    return pairwise


def test_conditioned_certificate_matches_exhaustive_extreme_beliefs() -> None:
    rng = np.random.default_rng(20260902)
    for _ in range(512):
        hypothesis_count = int(rng.integers(2, 8))
        action_count = int(rng.integers(2, 5))
        class_count = int(rng.integers(1, min(hypothesis_count, 4) + 1))
        classes = np.arange(hypothesis_count, dtype=np.int64) % class_count
        rng.shuffle(classes)
        prior = rng.random(hypothesis_count)
        prior[rng.random(hypothesis_count) < 0.2] = 0.0
        for class_id in range(class_count):
            members = np.flatnonzero(classes == class_id)
            if not np.any(prior[members] > 0.0):
                prior[members[0]] = 1.0
        prior /= np.sum(prior)
        quotient = rng.random(class_count)
        quotient[rng.random(class_count) < 0.2] = 0.0
        if np.sum(quotient) == 0.0:
            quotient[0] = 1.0
        quotient /= np.sum(quotient)
        losses = rng.normal(size=(hypothesis_count, action_count))
        radii = 0.05 * rng.random(size=losses.shape)
        consistent = rng.random(hypothesis_count) < 0.55
        feasible = (prior > 0.0) & (quotient[classes] > 0.0)
        if not np.any(consistent & feasible):
            consistent[np.flatnonzero(feasible)[0]] = True

        result = conditioned_query_decision_certificate(
            prior,
            quotient,
            classes,
            losses,
            consistent_hypothesis_mask=consistent,
            loss_radius_by_hypothesis_action=radii,
        )
        expected = _enumerate_conditioned_pairwise_gap(
            prior,
            quotient,
            classes,
            losses,
            consistent,
            radii,
        )

        np.testing.assert_allclose(
            result.pairwise_worst_case_loss_gap,
            expected,
            rtol=0.0,
            atol=1e-11,
        )
        np.testing.assert_allclose(
            result.worst_case_regret,
            np.maximum(np.max(expected, axis=1), 0.0),
            rtol=0.0,
            atol=1e-11,
        )


def test_full_history_reduces_to_existing_quotient_certificate_formula() -> None:
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

    result = conditioned_query_decision_certificate(
        prior,
        quotient,
        classes,
        losses,
    )
    expected = _enumerate_conditioned_pairwise_gap(
        prior,
        quotient,
        classes,
        losses,
        np.ones(prior.size, dtype=np.bool_),
    )

    np.testing.assert_allclose(result.pairwise_worst_case_loss_gap, expected)
    np.testing.assert_allclose(
        result.class_mass_lower_before_normalization,
        quotient,
    )
    np.testing.assert_allclose(
        result.class_mass_upper_before_normalization,
        quotient,
    )
    assert not result.loss_uncertainty_used
    assert result.maximum_loss_radius == 0.0


def test_probe_outcomes_are_certified_without_inventing_within_class_odds() -> None:
    prior = np.full(4, 0.25)
    quotient = np.array([0.5, 0.5])
    classes = np.array([0, 0, 1, 1])
    losses = np.array([[0.0, 2.0], [2.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    probe = DeterministicDecisionProbeV1(
        "decision-bit",
        np.array([0, 1, 0, 0]),
        1.0,
    )

    evaluation = evaluate_deterministic_probe(
        prior,
        quotient,
        classes,
        losses,
        probe,
    )

    assert evaluation.all_outcomes_tolerance_certified
    assert evaluation.worst_outcome_minimax_regret == pytest.approx(0.0)
    assert {row.outcome_index for row in evaluation.outcome_certificates} == {0, 1}
    assert all(
        row.certificate.has_tolerance_admissible_action
        for row in evaluation.outcome_certificates
    )


def test_exact_policy_prefers_cheaper_decision_probe_to_full_state_probe() -> None:
    prior = np.full(4, 0.25)
    quotient = np.array([0.5, 0.5])
    classes = np.array([0, 0, 1, 1])
    losses = np.array([[0.0, 2.0], [2.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    decision_probe = DeterministicDecisionProbeV1(
        "decision-bit",
        np.array([0, 1, 0, 0]),
        1.0,
    )
    full_state_probe = DeterministicDecisionProbeV1(
        "full-state",
        np.arange(4),
        3.0,
    )

    policy = synthesize_minimax_active_decision_policy(
        prior,
        quotient,
        classes,
        losses,
        (full_state_probe, decision_probe),
    )
    root = next(node for node in policy.nodes if node.state_id == policy.root_state_id)

    assert policy.feasible
    assert policy.root_worst_case_cost == pytest.approx(1.0)
    assert root.selected_probe_id == "decision-bit"
    assert not root.certified
    assert all(
        next(node for node in policy.nodes if node.state_id == child_id).certified
        for _, child_id in root.outcome_children
    )
    assert (
        policy.summary()["claim_boundary"]
        == ACTIVE_DECISION_ACQUISITION_CLAIM_BOUNDARY
    )


def test_adaptive_policy_uses_second_probe_only_on_ambiguous_branch() -> None:
    prior = np.full(6, 1.0 / 6.0)
    quotient = np.array([1.0])
    classes = np.zeros(6, dtype=np.int64)
    losses = np.array(
        [
            [0.0, 3.0, 3.0],
            [0.0, 3.0, 3.0],
            [3.0, 0.0, 3.0],
            [3.0, 0.0, 3.0],
            [3.0, 3.0, 0.0],
            [3.0, 3.0, 0.0],
        ]
    )
    first = DeterministicDecisionProbeV1(
        "is-action-zero",
        np.array([0, 0, 1, 1, 1, 1]),
        1.0,
    )
    second = DeterministicDecisionProbeV1(
        "one-versus-two",
        np.array([0, 0, 0, 0, 1, 1]),
        1.0,
    )
    full = DeterministicDecisionProbeV1("full-state", np.arange(6), 3.0)

    policy = synthesize_minimax_active_decision_policy(
        prior,
        quotient,
        classes,
        losses,
        (full, second, first),
    )
    root = next(node for node in policy.nodes if node.state_id == policy.root_state_id)

    assert policy.root_worst_case_cost == pytest.approx(2.0)
    assert root.selected_probe_id == "is-action-zero"
    children = {
        outcome: next(node for node in policy.nodes if node.state_id == child_id)
        for outcome, child_id in root.outcome_children
    }
    assert children[0].certified
    assert children[1].selected_probe_id == "one-versus-two"


def test_policy_fails_closed_when_available_probes_cannot_resolve_decision() -> None:
    prior = np.array([0.5, 0.5])
    quotient = np.array([1.0])
    classes = np.array([0, 0])
    losses = np.array([[0.0, 1.0], [1.0, 0.0]])
    uninformative = DeterministicDecisionProbeV1(
        "uninformative",
        np.array([0, 0]),
        1.0,
    )

    policy = synthesize_minimax_active_decision_policy(
        prior,
        quotient,
        classes,
        losses,
        (uninformative,),
    )

    assert not policy.feasible
    assert math_is_infinite(policy.root_worst_case_cost)
    root = next(node for node in policy.nodes if node.state_id == policy.root_state_id)
    assert root.selected_probe_id is None
    assert not root.certified


def math_is_infinite(value: float) -> bool:
    return bool(np.isinf(value))


def test_global_probe_set_is_exact_weighted_set_cover() -> None:
    prior = np.full(4, 0.25)
    classes = np.zeros(4, dtype=np.int64)
    losses = np.array([[0.0, 2.0], [2.0, 0.0], [0.0, 2.0], [2.0, 0.0]])
    cheap_a = DeterministicDecisionProbeV1("a", np.array([0, 1, 0, 1]), 1.0)
    redundant = DeterministicDecisionProbeV1("b", np.array([0, 1, 0, 1]), 2.0)
    full = DeterministicDecisionProbeV1("full", np.arange(4), 4.0)

    result = minimum_cost_global_decision_identifying_probe_set(
        prior,
        classes,
        losses,
        (full, redundant, cheap_a),
    )

    assert result.feasible
    assert result.exact
    assert result.selected_probe_ids == ("a",)
    assert result.total_cost == pytest.approx(1.0)
    assert result.conflict_pair_count == 4


def test_global_probe_set_reports_unresolvable_decision_conflict() -> None:
    result = minimum_cost_global_decision_identifying_probe_set(
        [0.5, 0.5],
        [0, 0],
        [[0.0, 1.0], [1.0, 0.0]],
        (
            DeterministicDecisionProbeV1(
                "same-outcome",
                np.array([0, 0]),
                1.0,
            ),
        ),
    )

    assert not result.feasible
    assert np.isinf(result.total_cost)
    assert result.conflict_pair_count == 1


def test_loss_uncertainty_can_block_an_optimistic_nominal_certificate() -> None:
    prior = np.array([0.5, 0.5])
    quotient = np.array([1.0])
    classes = np.array([0, 0])
    losses = np.array([[0.0, 0.2], [0.0, 0.2]])

    nominal = conditioned_query_decision_certificate(
        prior,
        quotient,
        classes,
        losses,
        regret_tolerance=0.0,
    )
    robust = conditioned_query_decision_certificate(
        prior,
        quotient,
        classes,
        losses,
        loss_radius_by_hypothesis_action=np.full_like(losses, 0.15),
        regret_tolerance=0.0,
    )

    assert nominal.has_tolerance_admissible_action
    assert nominal.minimax_action_index == 0
    assert not robust.has_tolerance_admissible_action
    assert robust.loss_uncertainty_used
    assert robust.maximum_loss_radius == pytest.approx(0.15)
    np.testing.assert_allclose(robust.worst_case_regret, np.array([0.1, 0.5]))


def test_probe_and_certificate_arrays_are_immutable_and_ids_are_stable() -> None:
    outcomes = np.array([0, 1, 0])
    probe_a = DeterministicDecisionProbeV1("view-2", outcomes, 2.5)
    probe_b = DeterministicDecisionProbeV1("view-2", outcomes.copy(), 2.5)
    result = conditioned_query_decision_certificate(
        [0.2, 0.3, 0.5],
        [0.4, 0.6],
        [0, 0, 1],
        [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
    )

    assert probe_a.probe_content_id == probe_b.probe_content_id
    with pytest.raises(ValueError, match="read-only"):
        probe_a.outcome_index[0] = 1
    with pytest.raises(ValueError, match="read-only"):
        result.worst_case_regret[0] = 0.0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"prior_weights": [0.4, 0.4]}, "sum to one"),
        ({"class_index": [0, 2]}, "contiguous"),
        ({"loss_by_hypothesis_action": [[0.0], [1.0]]}, "action_count"),
        (
            {
                "loss_radius_by_hypothesis_action": [
                    [-0.1, 0.0],
                    [0.0, 0.0],
                ]
            },
            "nonnegative",
        ),
        ({"consistent_hypothesis_mask": [False, False]}, "impossible"),
    ],
)
def test_conditioned_certificate_rejects_invalid_contracts(
    kwargs: dict[str, object],
    match: str,
) -> None:
    values: dict[str, object] = {
        "prior_weights": [0.5, 0.5],
        "quotient_weights": [1.0],
        "class_index": [0, 0],
        "loss_by_hypothesis_action": [[0.0, 1.0], [1.0, 0.0]],
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=match):
        conditioned_query_decision_certificate(**values)
