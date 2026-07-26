import numpy as np
import pytest

from bayesian_phystwin.deform360_action_anchored_state import (
    ActionAnchoredStateConfig,
    align_chain_orientation,
    chain_laplacian,
    estimate_action_anchored_chain_state,
)


def _chain(node_count: int = 7) -> np.ndarray:
    return np.column_stack(
        (
            np.linspace(0.0, 0.6, node_count),
            np.zeros(node_count),
            np.zeros(node_count),
        )
    )


def test_chain_laplacian_has_constant_nullspace() -> None:
    laplacian = chain_laplacian(6)
    np.testing.assert_allclose(laplacian, laplacian.T)
    np.testing.assert_allclose(laplacian @ np.ones(6), 0.0)
    assert np.all(np.linalg.eigvalsh(laplacian) >= -1e-12)


def test_orientation_alignment_reverses_material_order() -> None:
    current = _chain()
    previous = current[::-1] - np.array([0.01, 0.0, 0.0])
    aligned, diagnostics = align_chain_orientation(previous, current)
    assert diagnostics["reversed"] is True
    np.testing.assert_allclose(aligned, current - np.array([0.01, 0.0, 0.0]))


def test_action_anchors_remove_shared_observation_velocity_bias() -> None:
    previous = _chain()
    true_velocity = np.tile(np.array([0.03, -0.01, 0.02]), (len(previous), 1))
    observation_bias = np.array([0.12, -0.06, 0.03])
    dt = 0.1
    current = previous + dt * (true_velocity + observation_bias)
    anchor_indices = np.array([0, len(previous) - 1])
    previous_controllers = previous[anchor_indices]
    current_controllers = previous_controllers + dt * true_velocity[anchor_indices]

    estimate = estimate_action_anchored_chain_state(
        previous,
        current,
        previous_controllers,
        current_controllers,
        anchor_indices,
        dt_seconds=dt,
    )

    np.testing.assert_allclose(
        estimate.shared_observation_bias_m_s,
        observation_bias,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        estimate.bias_corrected_action_velocity_m_s,
        true_velocity,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        estimate.bias_corrected_action_velocity_m_s[anchor_indices],
        estimate.anchor_velocity_m_s,
        atol=1e-12,
    )
    assert estimate.accepted


def test_harmonic_action_velocity_interpolates_endpoint_motion() -> None:
    previous = _chain(5)
    current = previous.copy()
    anchors = np.array([0, 4])
    previous_controllers = previous[anchors]
    endpoint_velocity = np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]])
    current_controllers = previous_controllers + 0.1 * endpoint_velocity
    estimate = estimate_action_anchored_chain_state(
        previous,
        current,
        previous_controllers,
        current_controllers,
        anchors,
        dt_seconds=0.1,
    )
    np.testing.assert_allclose(
        estimate.action_harmonic_velocity_m_s[:, 0],
        np.linspace(0.0, 0.4, 5),
        atol=1e-8,
    )


def test_target_free_speed_gate_rejects_without_mutating_candidates() -> None:
    previous = _chain()
    current = previous.copy()
    anchors = np.array([0, len(previous) - 1])
    previous_controllers = previous[anchors]
    current_controllers = previous_controllers + np.array([0.5, 0.0, 0.0])
    estimate = estimate_action_anchored_chain_state(
        previous,
        current,
        previous_controllers,
        current_controllers,
        anchors,
        dt_seconds=0.1,
        config=ActionAnchoredStateConfig(maximum_initial_speed_m_s=1.0),
    )
    assert not estimate.accepted
    assert (
        np.max(np.linalg.norm(estimate.bias_corrected_action_velocity_m_s, axis=1))
        > 1.0
    )


def test_ambiguous_orientation_fails_closed() -> None:
    current = np.zeros((5, 3))
    with pytest.raises(ValueError, match="orientation is ambiguous"):
        estimate_action_anchored_chain_state(
            current,
            current,
            np.zeros((2, 3)),
            np.zeros((2, 3)),
            np.array([0, 4]),
            dt_seconds=0.1,
        )
