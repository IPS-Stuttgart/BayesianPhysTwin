"""Synthetic contracts for outcome-blind active probe selection."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import numpy as np
import pytest

from experiments.tracking_cloth_deformation_v1.active_probe import (
    entropy,
    model_spread,
    normalize_weights,
    pairwise_trajectory_mse,
    parameter_information_utility,
    pseudo_posteriors,
    select_action,
    simulate_policy,
    task_variance_reduction_utility,
    update_weights,
    weights_from_records,
)


class AccessLog(Mapping[str, np.ndarray]):
    def __init__(self, values: dict[str, np.ndarray]):
        self.values = values
        self.accessed: list[str] = []

    def __getitem__(self, key: str) -> np.ndarray:
        self.accessed.append(key)
        return self.values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)


def distances(scale_a: float, scale_b: float = 0.0) -> np.ndarray:
    # Three models; model 0 and 1 separate in the first mode, 1 and 2 in the second.
    vectors = np.array([[0.0, 0.0], [scale_a, 0.0], [scale_a, scale_b]])
    delta = vectors[:, None] - vectors[None]
    return np.sum(delta * delta, axis=-1)


def test_weights_and_updates_are_normalized_and_immutable() -> None:
    weights = normalize_weights([1.0, 2.0, 0.0])
    assert weights.sum() == pytest.approx(1.0)
    assert np.all(weights > 0.0)
    with pytest.raises(ValueError):
        weights[0] = 0.0
    updated = update_weights(weights, [0.0, 2.0, 4.0], 1.0)
    assert updated[0] > weights[0]
    fitted = weights_from_records([[0.0, 1.0, 2.0], [0.0, 1.0, 3.0]], 1.0)
    assert fitted[0] > fitted[1] > fitted[2]


def test_pairwise_trajectory_metric_matches_point_squared_error() -> None:
    bank = np.zeros((3, 2, 2, 3))
    bank[1, :, :, 0] = 2.0
    bank[2, :, :, 1] = 3.0
    valid = np.array([[False, True], [True, False]])
    matrix = pairwise_trajectory_mse(bank, valid)
    assert matrix[0, 1] == pytest.approx(4.0)
    assert matrix[0, 2] == pytest.approx(9.0)
    assert matrix[1, 2] == pytest.approx(13.0)
    np.testing.assert_array_equal(np.diag(matrix), 0.0)


def test_information_and_task_objectives_can_choose_different_probes() -> None:
    weights = np.array([0.45, 0.45, 0.10])
    # Probe a separates the high-mass pair; probe b isolates the low-mass model.
    probe_a = distances(3.0, 0.0)
    probe_b = distances(0.0, 8.0)
    # The held-out task only distinguishes model 2 from the first two.
    target = distances(0.0, 20.0)
    parameter = {
        action: parameter_information_utility(weights, matrix, 1.0)
        for action, matrix in {"a": probe_a, "b": probe_b}.items()
    }
    task = {
        action: task_variance_reduction_utility(weights, matrix, target, 1.0)
        for action, matrix in {"a": probe_a, "b": probe_b}.items()
    }
    # Verify the objectives are genuinely different rather than hard-code an order.
    assert parameter["a"] > parameter["b"]
    assert task["b"] > task["a"]
    assert (
        select_action(
            policy="parameter_information",
            weights=weights,
            remaining_actions=("a", "b"),
            probe_distances={"a": probe_a, "b": probe_b},
            target_distance=target,
            temperature=1.0,
            fixed_order=("a", "b"),
        )[0]
        == "a"
    )
    assert (
        select_action(
            policy="task_directed",
            weights=weights,
            remaining_actions=("a", "b"),
            probe_distances={"a": probe_a, "b": probe_b},
            target_distance=target,
            temperature=1.0,
            fixed_order=("a", "b"),
        )[0]
        == "b"
    )


def test_task_utility_is_expected_target_spread_contraction() -> None:
    weights = normalize_weights([0.4, 0.4, 0.2])
    probe = distances(0.0, 5.0)
    target = distances(0.0, 10.0)
    posteriors = pseudo_posteriors(weights, probe, 0.5)
    expected = sum(
        weights[index] * model_spread(posteriors[index], target) for index in range(3)
    )
    manual = 1.0 - expected / model_spread(weights, target)
    assert task_variance_reduction_utility(
        weights, probe, target, 0.5
    ) == pytest.approx(manual)
    assert entropy(weights) > 0.0


def test_policy_consumes_only_selected_outcomes() -> None:
    probe = {
        "a": distances(3.0),
        "b": distances(0.0, 8.0),
        "c": distances(1.0, 1.0),
    }
    losses = AccessLog(
        {
            "a": np.array([0.0, 1.0, 2.0]),
            "b": np.array([2.0, 1.0, 0.0]),
            "c": np.array([1.0, 0.0, 1.0]),
        }
    )
    states = simulate_policy(
        policy="task_directed",
        initial_weights=[0.45, 0.45, 0.10],
        probe_distances=probe,
        target_distance=distances(0.0, 20.0),
        observed_losses=losses,
        temperature=1.0,
        fixed_order=("a", "b", "c"),
        budgets=(0, 1, 2),
    )
    assert losses.accessed == list(states[2].selected_actions)
    assert len(losses.accessed) == 2
    assert states[0].selected_actions == ()
    assert len(states[1].selected_actions) == 1
    assert len(states[2].selected_actions) == 2


def test_all_probe_posterior_is_order_invariant() -> None:
    probe = {
        "a": distances(3.0),
        "b": distances(0.0, 8.0),
        "c": distances(1.0, 1.0),
    }
    values = {
        "a": np.array([0.0, 1.0, 2.0]),
        "b": np.array([2.0, 1.0, 0.0]),
        "c": np.array([1.0, 0.0, 1.0]),
    }
    finals = []
    for policy in ("fixed_order", "parameter_information", "task_directed"):
        states = simulate_policy(
            policy=policy,
            initial_weights=[0.4, 0.4, 0.2],
            probe_distances=probe,
            target_distance=distances(0.0, 20.0),
            observed_losses=values,
            temperature=1.0,
            fixed_order=("c", "a", "b"),
            budgets=(0, 1, 2, 3),
        )
        finals.append(states[3].weights)
    np.testing.assert_allclose(finals[0], finals[1], atol=1e-15)
    np.testing.assert_allclose(finals[0], finals[2], atol=1e-15)


def test_deterministic_tie_break_and_fixed_order() -> None:
    zeros = np.zeros((3, 3))
    distances_by_action = {"z": zeros, "a": zeros}
    selected, _ = select_action(
        policy="task_directed",
        weights=[1, 1, 1],
        remaining_actions=("z", "a"),
        probe_distances=distances_by_action,
        target_distance=zeros,
        temperature=1.0,
        fixed_order=("z", "a"),
    )
    assert selected == "a"
    selected, utilities = select_action(
        policy="fixed_order",
        weights=[1, 1, 1],
        remaining_actions=("a", "z"),
        probe_distances=distances_by_action,
        target_distance=zeros,
        temperature=1.0,
        fixed_order=("z", "a"),
    )
    assert selected == "z"
    assert utilities == {"a": None, "z": None}


@pytest.mark.parametrize(
    "call",
    [
        lambda: normalize_weights([0.0, 0.0]),
        lambda: normalize_weights([1.0]),
        lambda: update_weights([1, 1], [0, -1], 1.0),
        lambda: pseudo_posteriors([1, 1], [[0, 1], [2, 0]], 1.0),
        lambda: pairwise_trajectory_mse(np.zeros((2, 1, 1, 2)), [[True]]),
    ],
)
def test_invalid_inputs_fail_closed(call) -> None:
    with pytest.raises(ValueError):
        call()
