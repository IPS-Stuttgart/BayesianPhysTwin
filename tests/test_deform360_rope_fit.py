from __future__ import annotations

import numpy as np

from causal4d_public.deform360_rope_dynamics import (
    RopeDynamicsObservation,
    SharedRopeDynamicsParameters,
    rollout_rope_dynamics,
)
from causal4d_public.deform360_rope_fit import (
    RopeForwardFitConfig,
    fit_forward_rope_dynamics,
)


def _velocity_coupled_observation(
    episode_id: str, scale: float
) -> RopeDynamicsObservation:
    node_count = 9
    initial = np.column_stack(
        (
            np.linspace(-0.16, 0.16, node_count),
            np.zeros(node_count),
            np.zeros(node_count),
        )
    )
    rest = np.linalg.norm(np.diff(initial, axis=0), axis=1)
    frame_count = 36
    dt = 0.02
    controllers = np.repeat(initial[[0]][None], frame_count, axis=0)
    controllers[6:, 0, 2] += np.linspace(0.0, scale, frame_count - 6)
    active = np.ones((frame_count, 1), dtype=bool)
    truth = SharedRopeDynamicsParameters(0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0)
    positions = rollout_rope_dynamics(
        initial,
        np.zeros_like(initial),
        controllers,
        active,
        (0,),
        np.zeros((1, 3)),
        rest,
        truth,
        dt_seconds=dt,
        gravity_m_s2=np.zeros(3),
        substeps=2,
        constraint_iterations=8,
    )
    return RopeDynamicsObservation(
        episode_id=episode_id,
        positions_m=positions,
        controller_positions_m=controllers,
        contact_active=active,
        contact_node_indices=(0,),
        contact_offsets_m=np.zeros((1, 3)),
        dt_seconds=dt,
    )


def test_forward_fit_selects_transferable_velocity_coupling() -> None:
    config = RopeForwardFitConfig(
        bending_acceleration_grid=(0.0,),
        contact_acceleration_grid=(0.0,),
        contact_damping_grid=(0.0, 5.0),
        drag_grid=(0.0,),
        substeps=2,
        constraint_iterations=8,
        minimum_pooled_chamfer_improvement_fraction=0.01,
        minimum_loo_better_episode_fraction=1.0,
    )
    result = fit_forward_rope_dynamics(
        (
            _velocity_coupled_observation("source-a", 0.08),
            _velocity_coupled_observation("source-b", 0.12),
        ),
        config=config,
    )

    assert result["selected_parameters"]["contact_damping_per_s"] == 5.0
    assert result["source_competence_gate"]["passed"] is True
    assert all(
        row["chamfer_better_than_persistence"]
        for row in result["leave_one_episode_out"]
    )
