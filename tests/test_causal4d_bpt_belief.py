import numpy as np

from causal4d.bpt_belief import (
    BPTBeliefExportConfig,
    build_twin_belief_from_replays,
    lift_isotropic_discrepancy_variance,
)
from causal4d.contracts import build_causal_context


def _inputs():
    frame_count = 8
    train_end = 5
    observed = np.zeros((frame_count, 3, 3), dtype=float)
    observed[:, :, 0] = np.arange(frame_count)[:, None] * 0.01
    actions = np.zeros((frame_count, 1, 3), dtype=float)
    context = build_causal_context(
        protocol_id="belief_test",
        case_id="synthetic",
        observations=observed,
        observed_actions=actions,
        counterfactual_actions=actions,
        intervention_frame=train_end,
    )
    replay = np.zeros((2, frame_count, 5, 3), dtype=float)
    replay[0, :, :3] = observed
    replay[1, :, :3] = observed
    replay[1, :, :, 0] -= np.arange(frame_count)[:, None] * 0.001
    replay[:, :, 3, 0] = replay[:, :, 0, 0]
    replay[:, :, 4, 0] = replay[:, :, 2, 0]
    velocity = np.zeros_like(replay)
    velocity[:, 1:] = np.diff(replay, axis=1) / 0.03
    valid = np.ones((frame_count, 3), dtype=bool)
    return context, replay, velocity, observed, valid


def _belief(context, replay, velocity, observed, valid):
    return build_twin_belief_from_replays(
        context=context,
        replay_positions_m=replay,
        replay_velocities_mps=velocity,
        observed_positions_m=observed,
        observed_valid=valid,
        theta=np.asarray([[0.0, 0.0], [0.2, -0.1]]),
        theta_names=("object", "controller"),
        weights=np.asarray([0.7, 0.3]),
        config=BPTBeliefExportConfig(interpolation_neighbors=2),
    )


def test_full_belief_uses_particle_specific_endpoint_state() -> None:
    context, replay, velocity, observed, valid = _inputs()
    belief = _belief(context, replay, velocity, observed, valid)
    assert np.array_equal(belief.endpoint_position_m, replay[:, 4])
    assert np.array_equal(belief.endpoint_velocity_mps, velocity[:, 4])
    assert not np.array_equal(
        belief.endpoint_position_m[0], belief.endpoint_position_m[1]
    )
    assert belief.metadata["future_frames_read_by_estimator"] == 0
    assert belief.metadata["maximum_pairwise_endpoint_rmse_m"] > 0.0


def test_belief_estimation_cannot_see_changed_future_frames() -> None:
    context, replay, velocity, observed, valid = _inputs()
    first = _belief(context, replay, velocity, observed, valid)
    changed_replay = replay.copy()
    changed_velocity = velocity.copy()
    changed_observed = observed.copy()
    changed_valid = valid.copy()
    changed_replay[:, 5:] += 1000.0
    changed_velocity[:, 5:] -= 1000.0
    changed_observed[5:] = -1000.0
    changed_valid[5:] = False
    second = _belief(
        context,
        changed_replay,
        changed_velocity,
        changed_observed,
        changed_valid,
    )
    assert first.artifact_id == second.artifact_id
    assert np.array_equal(first.endpoint_position_m, second.endpoint_position_m)
    assert np.array_equal(first.discrepancy_mean_m, second.discrepancy_mean_m)
    assert np.array_equal(
        first.discrepancy_variance_m2,
        second.discrepancy_variance_m2,
    )


def test_discrepancy_is_separate_from_the_replayed_state() -> None:
    context, replay, velocity, observed, valid = _inputs()
    belief = _belief(context, replay, velocity, observed, valid)
    assert np.array_equal(belief.endpoint_position_m, replay[:, 4])
    assert np.linalg.norm(belief.discrepancy_mean_m[1]) > 0.0
    assert "not injected" in belief.metadata["discrepancy_role"]


def test_variance_lift_uses_squared_interpolation_weights() -> None:
    tracked = np.asarray([1.0, 4.0])
    indices = np.asarray([[0, 1]])
    weights = np.asarray([[0.25, 0.75]])
    lifted = lift_isotropic_discrepancy_variance(
        tracked,
        state_count=3,
        indices=indices,
        weights=weights,
    )
    expected_extra = 0.25**2 * 1.0 + 0.75**2 * 4.0
    assert np.allclose(lifted[:2, 0], tracked)
    assert np.allclose(lifted[2], expected_extra)
