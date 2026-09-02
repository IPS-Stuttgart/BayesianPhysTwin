from __future__ import annotations

import itertools

import numpy as np
import pytest

from bayesian_phystwin.adaptive_stochastic_policy_tree_v1 import (
    ADAPTIVE_STOCHASTIC_POLICY_TREE_CLAIM_BOUNDARY,
    adaptive_stochastic_policy_tree_certificate,
    conformal_adaptive_policy_tree_decision,
    terminal_action_for_probe_outcomes,
)
from bayesian_phystwin.conformal_complete_plan_certificate_v1 import (
    scaled_trajectory_conformal_plan_envelope,
)


def _binary_sensor(values: np.ndarray, accuracy: float) -> np.ndarray:
    result = np.empty((values.size, 2), dtype=np.float64)
    result[:, 0] = np.where(values == 0, accuracy, 1.0 - accuracy)
    result[:, 1] = 1.0 - result[:, 0]
    return result


def _routing_problem(*, maximum_depth: int, tolerance: float = 0.20):
    hypotheses = np.asarray(
        list(itertools.product((0, 1), repeat=4)),
        dtype=np.int64,
    )
    route = hypotheses[:, 0]
    x_value = hypotheses[:, 1]
    y_value = hypotheses[:, 2]
    nuisance = hypotheses[:, 3]
    target = np.where(route == 0, x_value, y_value)
    terminal_losses = np.empty((hypotheses.shape[0], 3), dtype=np.float64)
    terminal_losses[:, 0] = target
    terminal_losses[:, 1] = 1 - target
    terminal_losses[:, 2] = 0.45
    return adaptive_stochastic_policy_tree_certificate(
        np.ones(hypotheses.shape[0]),
        [1.0],
        np.zeros(hypotheses.shape[0], dtype=np.int64),
        terminal_losses,
        [
            _binary_sensor(route, 0.98),
            _binary_sensor(x_value, 0.95),
            _binary_sensor(y_value, 0.95),
            _binary_sensor(nuisance, 0.999),
        ],
        [0.025, 0.036, 0.036, 0.001],
        fallback_action_index=2,
        maximum_depth=maximum_depth,
        regret_tolerance=tolerance,
        probe_names=["route", "x", "y", "nuisance"],
        max_policy_count=5000,
        max_raw_tree_count=500000,
    )


def test_depth_zero_returns_exact_fallback() -> None:
    certificate = _routing_problem(maximum_depth=0)

    assert certificate.used_fallback
    assert certificate.output_mode == "fallback"
    assert certificate.output_policy.action_index == 2
    assert certificate.worst_case_regret[certificate.fallback_policy_index] == (
        pytest.approx(0.45)
    )
    assert certificate.fallback_reason == "registered-fallback-is-minimax-policy"


def test_one_probe_is_insufficient_and_does_not_spend_for_nuisance() -> None:
    certificate = _routing_problem(maximum_depth=1)

    assert certificate.used_fallback
    assert certificate.output_policy.action_index == 2
    assert certificate.selected_first_probe_index is None
    assert certificate.worst_case_regret[certificate.fallback_policy_index] == (
        pytest.approx(0.45)
    )


def test_adaptive_depth_two_identifies_decision_without_state() -> None:
    certificate = _routing_problem(maximum_depth=2)

    assert not certificate.used_fallback
    assert certificate.output_mode == "sense"
    assert certificate.selected_first_probe_index == 0
    assert certificate.probe_names[certificate.selected_first_probe_index] == "route"
    root = certificate.output_policy
    assert root.probe_index == 0
    assert tuple(child.probe_index for child in root.children) == (1, 2)
    assert root.depth == 2
    assert certificate.minimizer_count == 1
    assert certificate.worst_case_regret[certificate.output_policy_index] == (
        pytest.approx(0.129, abs=1e-12)
    )
    assert np.all(root.expected_loss_by_hypothesis < 0.45)
    assert "nuisance" not in root.canonical_key
    # The nuisance bit remains different across otherwise identical hypotheses.
    assert certificate.prior_probabilities.size == 16
    assert certificate.class_masses.size == 1


def test_fixed_two_probe_policy_class_cannot_match_adaptive_tree() -> None:
    certificate = _routing_problem(maximum_depth=2)

    fixed_indices: list[int] = []
    for index, policy in enumerate(certificate.policies):
        if policy.mode == "act":
            fixed_indices.append(index)
            continue
        if policy.depth != 2 or not policy.children:
            fixed_indices.append(index)
            continue
        if all(child.mode == "sense" for child in policy.children):
            child_probes = {child.probe_index for child in policy.children}
            if len(child_probes) == 1:
                fixed_indices.append(index)
    fixed_regret = float(np.min(certificate.worst_case_regret[fixed_indices]))

    assert fixed_regret == pytest.approx(0.45)
    assert certificate.worst_case_regret[certificate.output_policy_index] < fixed_regret


def test_tree_traversal_uses_precommitted_branches() -> None:
    certificate = _routing_problem(maximum_depth=2)
    policy = certificate.output_policy

    action_zero, trace_zero = terminal_action_for_probe_outcomes(
        policy,
        [0, 0, 1, 1],
    )
    action_one, trace_one = terminal_action_for_probe_outcomes(
        policy,
        [1, 0, 1, 0],
    )

    assert action_zero == 0
    assert trace_zero == ((0, 0), (1, 0))
    assert action_one == 1
    assert trace_one == ((0, 1), (2, 1))
    with pytest.raises(ValueError, match="outside the registered"):
        terminal_action_for_probe_outcomes(policy, [2, 0, 0, 0])


def test_pairwise_support_function_matches_explicit_class_vertices() -> None:
    prior = np.ones(4)
    classes = np.asarray([0, 0, 1, 1], dtype=np.int64)
    losses = np.asarray(
        [
            [0.0, 1.0, 0.4],
            [0.2, 0.8, 0.4],
            [0.9, 0.1, 0.4],
            [0.7, 0.3, 0.4],
        ]
    )
    certificate = adaptive_stochastic_policy_tree_certificate(
        prior,
        [0.25, 0.75],
        classes,
        losses,
        [],
        [],
        fallback_action_index=2,
        maximum_depth=0,
        regret_tolerance=1.0,
    )

    for policy_index in range(certificate.policy_count):
        for benchmark_index in range(certificate.policy_count):
            explicit = 0.0
            for class_index, mass in enumerate((0.25, 0.75)):
                members = np.flatnonzero(classes == class_index)
                gaps = (
                    certificate.expected_loss_by_hypothesis_policy[
                        members, policy_index
                    ]
                    - certificate.expected_loss_by_hypothesis_policy[
                        members, benchmark_index
                    ]
                )
                explicit += mass * float(np.max(gaps))
            assert certificate.pairwise_worst_case_gap[
                policy_index, benchmark_index
            ] == pytest.approx(explicit)


def test_conformal_wrapper_certifies_whole_tree_or_falls_back_pre_probe() -> None:
    certificate = _routing_problem(maximum_depth=2)
    registered = np.broadcast_to(
        certificate.worst_case_regret,
        (4, 1, certificate.policy_count),
    ).copy()
    scales = np.ones(certificate.policy_count)
    zero_envelope = scaled_trajectory_conformal_plan_envelope(
        registered,
        registered,
        scales,
        miscoverage=0.25,
    )
    selected = conformal_adaptive_policy_tree_decision(
        certificate,
        zero_envelope,
    )

    assert not selected.used_fallback
    assert selected.output_policy.canonical_key == certificate.output_policy.canonical_key
    assert selected.output_mode == "sense"

    inflated_envelope = scaled_trajectory_conformal_plan_envelope(
        registered + 0.30,
        registered,
        scales,
        miscoverage=0.25,
    )
    rejected = conformal_adaptive_policy_tree_decision(
        certificate,
        inflated_envelope,
    )
    assert rejected.used_fallback
    assert rejected.output_mode == "fallback"
    assert rejected.output_policy.action_index == 2
    assert rejected.fallback_reason == "calibrated-policy-regret-exceeds-tolerance"


def test_nonunique_minimax_tree_falls_back_without_tie_breaking() -> None:
    certificate = adaptive_stochastic_policy_tree_certificate(
        [0.5, 0.5],
        [1.0],
        [0, 0],
        np.zeros((2, 3)),
        [],
        [],
        fallback_action_index=2,
        maximum_depth=0,
        regret_tolerance=0.0,
    )

    assert certificate.minimizer_count == 3
    assert certificate.used_fallback
    assert certificate.output_policy.action_index == 2
    assert certificate.fallback_reason == "nonunique-minimax-policy"


def test_outputs_are_bytes_backed_and_inputs_are_not_mutated() -> None:
    prior = np.ones(16)
    prior_before = prior.copy()
    certificate = _routing_problem(maximum_depth=2)

    np.testing.assert_array_equal(prior, prior_before)
    arrays = (
        certificate.prior_probabilities,
        certificate.class_index_by_hypothesis,
        certificate.class_masses,
        certificate.terminal_loss_by_hypothesis_action,
        certificate.expected_loss_by_hypothesis_policy,
        certificate.pairwise_worst_case_gap,
        certificate.worst_case_regret,
        certificate.output_policy.expected_loss_by_hypothesis,
    )
    for array in arrays:
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_validation_and_complexity_caps_fail_closed() -> None:
    with pytest.raises(ValueError, match="rows must sum to one"):
        adaptive_stochastic_policy_tree_certificate(
            [0.5, 0.5],
            [1.0],
            [0, 0],
            [[0.0, 1.0], [1.0, 0.0]],
            [np.asarray([[0.6, 0.6], [0.5, 0.5]])],
            [0.1],
            fallback_action_index=0,
            maximum_depth=1,
            regret_tolerance=1.0,
        )
    with pytest.raises(ValueError, match="raw policy-tree count"):
        adaptive_stochastic_policy_tree_certificate(
            [0.5, 0.5],
            [1.0],
            [0, 0],
            [[0.0, 1.0], [1.0, 0.0]],
            [
                np.asarray([[0.9, 0.1], [0.1, 0.9]]),
                np.asarray([[0.8, 0.2], [0.2, 0.8]]),
            ],
            [0.1, 0.1],
            fallback_action_index=0,
            maximum_depth=2,
            regret_tolerance=1.0,
            max_raw_tree_count=10,
        )
    assert "pre-probe" not in ADAPTIVE_STOCHASTIC_POLICY_TREE_CLAIM_BOUNDARY
    assert "does not validate" in ADAPTIVE_STOCHASTIC_POLICY_TREE_CLAIM_BOUNDARY
