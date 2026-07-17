from __future__ import annotations

import numpy as np
import pytest

from causal4d_public.deform360_bayesian_residual import (
    TemporalBayesianResidualModelConfig,
)
from causal4d_public.deform360_bayesian_residual_data import (
    Deform360ResidualSourceEpisode,
)
from causal4d_public.deform360_temporal_residual_experiment import (
    TemporalResidualTrainingConfig,
    rollout_temporal_residual_model,
    train_temporal_residual_model,
)


torch = pytest.importorskip("torch")


def _episode() -> Deform360ResidualSourceEpisode:
    frame_count = 12
    initial = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.1, 0.1, 0.0],
            [0.0, 0.1, 0.0],
        ],
        dtype=np.float32,
    )
    time = np.linspace(0.0, 1.0, frame_count, dtype=np.float32)
    positions = np.repeat(initial[None], frame_count, axis=0)
    positions[:, :, 1] += 0.02 * time[:, None]
    positions[:, :, 2] += 0.01 * np.square(time[:, None])
    physics = np.repeat(initial[None], frame_count, axis=0)
    physics[:, :, 1] += 0.012 * time[:, None]
    velocities = np.zeros_like(positions)
    velocities[1:] = (positions[1:] - positions[:-1]) * 30.0
    controllers = np.zeros((frame_count, 1, 3), dtype=np.float32)
    controllers[:, 0, 0] = 0.05
    controllers[:, 0, 1] = 0.02 * time
    controller_velocity = np.zeros_like(controllers)
    controller_velocity[1:] = (controllers[1:] - controllers[:-1]) * 30.0
    return Deform360ResidualSourceEpisode(
        object_id="synthetic",
        episode_id=0,
        positions_m=positions,
        observed_velocities_mps=velocities,
        physics_positions_m=physics,
        physics_prior_kind="trusted_sealed_graph_action_support",
        prior_reliability=np.ones((frame_count, len(initial)), dtype=np.float32),
        controller_positions_m=controllers,
        controller_velocities_mps=controller_velocity,
        closure_probability=np.ones((frame_count, 1), dtype=np.float32),
        controller_group_ids=np.zeros(1, dtype=np.int64),
        controller_geometry="gripper_surface",
        edge_index=np.array(
            [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]],
            dtype=np.int64,
        ),
        cluster_ids=np.arange(len(initial), dtype=np.int64),
        frame_interval_s=1.0 / 30.0,
        physics_response_scale=0.3,
    )


def test_temporal_training_and_exact_physics_fallback() -> None:
    episode = _episode()
    model, summary = train_temporal_residual_model(
        [episode],
        model_config=TemporalBayesianResidualModelConfig(
            hidden_dim=16,
            message_steps=1,
            temporal_hidden_dim=16,
        ),
        training_config=TemporalResidualTrainingConfig(
            steps=2,
            context_steps=2,
            rollout_steps=2,
            seed=3,
        ),
        device="cpu",
    )

    rollout = rollout_temporal_residual_model(
        model,
        episode,
        physics_velocity_retention=0.85,
        device="cpu",
        utility_threshold=1.1,
    )

    assert summary["context_steps"] == 2
    assert np.all(np.isfinite(rollout["position_variance_m2"]))
    assert not np.any(rollout["accepted"])
    assert rollout["residual_m"].tobytes() == rollout["physics_m"].tobytes()
