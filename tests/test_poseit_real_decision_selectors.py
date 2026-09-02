from __future__ import annotations

import inspect

import numpy as np
import pytest

from bayesian_phystwin_experiments.poseit_real_decision_selectors import (
    POSE_COUNT,
    PoseItGaussianState,
    condition_on_pose_features,
    decision_value_of_probe,
    expected_best_utility,
    pose_feature_indices,
    pose_stability_index,
    select_next_probe,
    stability_probabilities,
    system_identification_value_of_probe,
    trace_policy,
    trace_probe_order,
)


def _identity_state(
    *,
    feature_dimension: int = 1,
    available_poses: tuple[int, ...] = tuple(range(1, POSE_COUNT + 1)),
) -> PoseItGaussianState:
    coordinate_count = POSE_COUNT * (feature_dimension + 1)
    return PoseItGaussianState(
        mean=np.zeros(coordinate_count),
        covariance=np.eye(coordinate_count),
        feature_dimension=feature_dimension,
        available_poses=available_poses,
    )


def _divergent_selector_state() -> PoseItGaussianState:
    feature_dimension = 1
    coordinate_count = POSE_COUNT * (feature_dimension + 1)
    loadings = np.zeros((coordinate_count, 2), dtype=np.float64)
    loadings[pose_feature_indices(2, feature_dimension)[0], 0] = 1.0
    loadings[pose_stability_index(16, feature_dimension), 0] = 1.0
    loadings[pose_feature_indices(3, feature_dimension)[0], 1] = 1.0
    for pose in range(4, 11):
        loadings[pose_stability_index(pose, feature_dimension), 1] = 1.0
    covariance = loadings @ loadings.T + 0.2 * np.eye(coordinate_count)
    mean = np.zeros(coordinate_count, dtype=np.float64)
    for pose in range(2, POSE_COUNT + 1):
        mean[pose_stability_index(pose, feature_dimension)] = -3.0
    mean[pose_stability_index(16, feature_dimension)] = 0.0
    return PoseItGaussianState(mean, covariance, feature_dimension)


def test_conditioning_observes_pre_shake_features_but_not_stability() -> None:
    state = _identity_state(feature_dimension=2)
    latent_indices = tuple(
        pose_stability_index(pose, state.feature_dimension)
        for pose in state.action_poses
    )

    posterior = condition_on_pose_features(state, 1, (0.25, -0.5))

    feature_indices = pose_feature_indices(1, state.feature_dimension)
    np.testing.assert_allclose(posterior.mean[list(feature_indices)], (0.25, -0.5))
    np.testing.assert_allclose(posterior.covariance[list(feature_indices), :], 0.0)
    np.testing.assert_allclose(posterior.covariance[:, list(feature_indices)], 0.0)
    np.testing.assert_array_equal(
        posterior.mean[list(latent_indices)], state.mean[list(latent_indices)]
    )
    assert posterior.observed_poses == (1,)


def test_selector_interfaces_have_no_outcome_or_label_input() -> None:
    for function in (
        condition_on_pose_features,
        decision_value_of_probe,
        system_identification_value_of_probe,
        trace_policy,
    ):
        names = set(inspect.signature(function).parameters)
        assert "outcome" not in names
        assert "label" not in names
        assert "stable" not in names


def test_decision_and_system_identification_selectors_diverge() -> None:
    state = condition_on_pose_features(_divergent_selector_state(), 1, (0.0,))
    remaining = state.action_poses

    decision = select_next_probe(
        state,
        remaining,
        selector="decision_directed",
    )
    identification = select_next_probe(
        state,
        remaining,
        selector="system_identification",
    )

    assert decision == 2
    assert identification == 3
    assert decision_value_of_probe(state, 2) > decision_value_of_probe(state, 3)
    assert system_identification_value_of_probe(
        state, 3
    ) > system_identification_value_of_probe(state, 2)


def test_expected_utility_includes_zero_utility_abstention() -> None:
    state = _identity_state()
    mean = state.mean.copy()
    for pose in state.action_poses:
        mean[pose_stability_index(pose, state.feature_dimension)] = -10.0
    pessimistic = PoseItGaussianState(
        mean,
        state.covariance,
        state.feature_dimension,
    )

    assert expected_best_utility(pessimistic) == pytest.approx(0.0)
    assert np.all(stability_probabilities(pessimistic) < 0.5)


def test_registered_lowest_pose_tie_break_is_deterministic() -> None:
    state = condition_on_pose_features(_identity_state(), 1, (0.0,))

    selected = select_next_probe(
        state,
        (7, 2, 5),
        selector="fixed",
    )

    assert selected == 2


def test_trace_uses_only_anchor_and_selected_pre_shake_features() -> None:
    prior = _divergent_selector_state()
    features = {pose: np.asarray([0.01 * pose]) for pose in prior.available_poses}

    trace = trace_policy(prior, features, selector="decision_directed")

    assert len(trace.states) == 4
    assert len(trace.selected_poses) == 3
    assert trace.states[0].observed_poses == (1,)
    for budget, state in enumerate(trace.states):
        assert state.observed_poses == (1, *trace.selected_poses[:budget])
        observed_coordinates = {
            coordinate
            for pose in state.observed_poses
            for coordinate in pose_feature_indices(pose, state.feature_dimension)
        }
        latent_coordinates = {
            pose_stability_index(pose, state.feature_dimension)
            for pose in state.available_poses
        }
        assert observed_coordinates.isdisjoint(latent_coordinates)


def test_structurally_unavailable_actions_are_excluded_without_dropping_family() -> (
    None
):
    prior = _identity_state(available_poses=(1, 3, 7))
    features = {1: np.asarray([0.0]), 3: np.asarray([0.1]), 7: np.asarray([0.2])}

    trace = trace_policy(prior, features, selector="fixed")

    assert trace.selected_poses == (3, 7)
    assert len(trace.states) == 4
    assert trace.states[-1].observed_poses == (1, 3, 7)


def test_anchor_only_family_carries_forward_abstention_at_every_budget() -> None:
    prior = _identity_state(available_poses=(1,))

    trace = trace_policy(prior, {1: np.asarray([0.0])}, selector="fixed")

    assert trace.selected_poses == ()
    assert len(trace.states) == 4
    assert all(expected_best_utility(state) == 0.0 for state in trace.states)
    assert all(state.observed_poses == (1,) for state in trace.states)


def test_fixed_control_rejects_unavailable_pose_before_feature_access() -> None:
    prior = _identity_state(available_poses=(1, 3, 7))
    features = {1: np.asarray([0.0]), 3: np.asarray([0.1]), 7: np.asarray([0.2])}

    with pytest.raises(ValueError, match="unavailable"):
        trace_probe_order(prior, features, (2,), selector="fixed")


def test_decision_value_is_reproducible_under_registered_common_draws() -> None:
    state = condition_on_pose_features(_divergent_selector_state(), 1, (0.0,))

    first = decision_value_of_probe(state, 2)
    second = decision_value_of_probe(state, 2)

    assert first == second
    assert first > 0.0
