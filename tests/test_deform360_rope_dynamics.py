from __future__ import annotations

import numpy as np

from causal4d_public.deform360_rope_dynamics import (
    RopeDynamicsObservation,
    SharedRopeDynamicsParameters,
    fit_shared_rope_dynamics,
    rollout_rope_dynamics,
)


def _rest_rope(node_count: int = 11) -> tuple[np.ndarray, np.ndarray]:
    positions = np.column_stack(
        (np.linspace(-0.2, 0.2, node_count), np.zeros(node_count), np.zeros(node_count))
    )
    rest = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return positions, rest


def _synthetic_observation(
    episode_id: str,
    *,
    phase: float,
    both_contacts: bool,
    parameters: SharedRopeDynamicsParameters,
) -> RopeDynamicsObservation:
    initial, rest = _rest_rope()
    frame_count = 180
    dt = 0.02
    time = np.arange(frame_count) * dt
    controller_count = 2 if both_contacts else 1
    controllers = np.empty((frame_count, controller_count, 3), dtype=np.float64)
    controllers[:, 0] = initial[0]
    controllers[:, 0, 1] += 0.045 * np.sin(1.7 * time + phase)
    controllers[:, 0, 2] += 0.025 * (1.0 - np.cos(1.2 * time + phase))
    contact_nodes = (0,)
    if both_contacts:
        controllers[:, 1] = initial[-1]
        controllers[:, 1, 1] += 0.035 * np.sin(1.1 * time + 0.4 + phase)
        controllers[:, 1, 2] += 0.03 * np.sin(0.8 * time + phase)
        contact_nodes = (0, len(initial) - 1)
    active = np.ones((frame_count, controller_count), dtype=bool)
    trajectory = rollout_rope_dynamics(
        initial,
        np.zeros_like(initial),
        controllers,
        active,
        contact_nodes,
        np.zeros((controller_count, 3)),
        rest,
        parameters,
        dt_seconds=dt,
        gravity_m_s2=np.zeros(3),
        substeps=6,
    )
    return RopeDynamicsObservation(
        episode_id=episode_id,
        positions_m=trajectory,
        controller_positions_m=controllers,
        contact_active=active,
        contact_node_indices=contact_nodes,
        contact_offsets_m=np.zeros((controller_count, 3)),
        dt_seconds=dt,
    )


def test_shared_fit_recovers_synthetic_source_dynamics() -> None:
    truth = SharedRopeDynamicsParameters(60.0, 2.2, 18.0, 0.8, 95.0, 3.5, 0.7)
    _, rest = _rest_rope()
    observations = (
        _synthetic_observation(
            "source-a", phase=0.0, both_contacts=False, parameters=truth
        ),
        _synthetic_observation(
            "source-b", phase=0.7, both_contacts=True, parameters=truth
        ),
        _synthetic_observation(
            "source-c", phase=1.3, both_contacts=True, parameters=truth
        ),
    )

    fitted = fit_shared_rope_dynamics(
        observations,
        rest,
        gravity_m_s2=np.zeros(3),
        ridge_strength=1e-6,
    )

    np.testing.assert_allclose(
        fitted.parameters.as_array(),
        truth.as_array(),
        rtol=0.30,
        atol=0.4,
    )
    assert fitted.residual_acceleration_rmse_m_s2 < 0.4
    assert fitted.design_row_count > 10_000


def test_inactive_second_contact_changes_held_out_rollout() -> None:
    initial, rest = _rest_rope()
    parameters = SharedRopeDynamicsParameters(50.0, 2.0, 16.0, 0.7, 90.0, 3.0, 0.8)
    frame_count = 80
    controllers = np.repeat(initial[[0, -1]][None], frame_count, axis=0)
    controllers[:, 1, 2] += np.linspace(0.0, 0.12, frame_count)
    both = np.ones((frame_count, 2), dtype=bool)
    one = both.copy()
    one[:, 1] = False

    both_rollout = rollout_rope_dynamics(
        initial,
        np.zeros_like(initial),
        controllers,
        both,
        (0, len(initial) - 1),
        np.zeros((2, 3)),
        rest,
        parameters,
        dt_seconds=0.02,
        gravity_m_s2=np.zeros(3),
    )
    one_rollout = rollout_rope_dynamics(
        initial,
        np.zeros_like(initial),
        controllers,
        one,
        (0, len(initial) - 1),
        np.zeros((2, 3)),
        rest,
        parameters,
        dt_seconds=0.02,
        gravity_m_s2=np.zeros(3),
    )

    assert np.linalg.norm(both_rollout[-1, -1] - one_rollout[-1, -1]) > 0.04
    assert both_rollout[-1, -1, 2] > one_rollout[-1, -1, 2]


def test_constraint_projection_preserves_rope_edge_lengths() -> None:
    initial, rest = _rest_rope()
    parameters = SharedRopeDynamicsParameters(0.0, 0.0, 0.0, 0.0, 180.0, 4.0, 0.0)
    frame_count = 60
    controllers = np.repeat(initial[[0]][None], frame_count, axis=0)
    controllers[:, 0, 2] += np.linspace(0.0, 0.18, frame_count)
    active = np.ones((frame_count, 1), dtype=bool)

    unconstrained = rollout_rope_dynamics(
        initial,
        np.zeros_like(initial),
        controllers,
        active,
        (0,),
        np.zeros((1, 3)),
        rest,
        parameters,
        dt_seconds=0.02,
        gravity_m_s2=np.zeros(3),
        substeps=6,
    )
    constrained = rollout_rope_dynamics(
        initial,
        np.zeros_like(initial),
        controllers,
        active,
        (0,),
        np.zeros((1, 3)),
        rest,
        parameters,
        dt_seconds=0.02,
        gravity_m_s2=np.zeros(3),
        substeps=6,
        constraint_iterations=24,
    )

    unconstrained_error = np.max(
        np.abs(np.linalg.norm(np.diff(unconstrained, axis=1), axis=2) - rest)
    )
    constrained_error = np.max(
        np.abs(np.linalg.norm(np.diff(constrained, axis=1), axis=2) - rest)
    )
    assert unconstrained_error > 0.01
    assert constrained_error < 2e-4


def test_kinematic_contact_pins_active_grasp_node() -> None:
    initial, rest = _rest_rope()
    parameters = SharedRopeDynamicsParameters(0.0, 0.0, 8.0, 0.2, 0.0, 0.0, 0.1)
    frame_count = 40
    controllers = np.repeat(initial[[0]][None], frame_count, axis=0)
    controllers[:, 0, 1] += np.linspace(0.0, 0.12, frame_count)
    contact_offset = np.asarray([[0.0, 0.0, 0.01]])
    active = np.ones((frame_count, 1), dtype=bool)

    trajectory = rollout_rope_dynamics(
        initial + contact_offset,
        np.zeros_like(initial),
        controllers,
        active,
        (0,),
        contact_offset,
        rest,
        parameters,
        dt_seconds=0.02,
        gravity_m_s2=np.zeros(3),
        substeps=4,
        constraint_iterations=24,
        kinematic_contacts=True,
    )

    np.testing.assert_allclose(
        trajectory[1:, 0], controllers[1:, 0] + contact_offset[0], atol=1e-12
    )
    edge_error = np.abs(np.linalg.norm(np.diff(trajectory, axis=1), axis=2) - rest)
    assert np.max(edge_error[1:]) < 3e-4
