import numpy as np

from causal4d.rollout_bank import JointRolloutBank, SparseTrajectoryEvidence


def _bank() -> JointRolloutBank:
    trajectories = np.zeros((2, 1, 7, 2, 3), dtype=float)
    time = np.arange(7, dtype=float)
    trajectories[0, 0, :, :, 0] = -0.01 * time[:, None]
    trajectories[1, 0, :, :, 0] = 0.01 * time[:, None]
    trajectories[1, 0, :, :, 2] = 0.002 * time[:, None]
    return JointRolloutBank(
        hypothesis_ids=("left", "right"),
        hypothesis_metadata=({"action": {"proposal_id": "left"}}, {"action": {"proposal_id": "right"}}),
        hypothesis_prior_weights=np.asarray([0.5, 0.5]),
        parameter_particles=np.asarray([[0.0, 0.0]]),
        parameter_weights=np.asarray([1.0]),
        trajectories=trajectories,
        variance_floor_m2=1e-6,
    )


def test_prefix_update_cannot_see_changed_future() -> None:
    bank = _bank()
    observations = bank.trajectories[1, 0].copy()
    changed = observations.copy()
    changed[3:] += 100.0
    first = bank.update_from_observations(
        observations,
        prefix_frame_count=3,
        scale_m=0.005,
        likelihood_power=4.0,
    )
    second = bank.update_from_observations(
        changed,
        prefix_frame_count=3,
        scale_m=0.005,
        likelihood_power=4.0,
    )
    assert np.array_equal(first, second)
    assert first[1, 0] > first[0, 0]


def test_sparse_displacement_evidence_ranks_matching_physical_rollout() -> None:
    bank = _bank()
    nodes = np.asarray([0, 1])
    target = bank.trajectories[1, 0, 1:][:, nodes]
    evidence = SparseTrajectoryEvidence(
        positions_m=target,
        node_indices=nodes,
        rollout_frame_indices=np.arange(1, 7, dtype=float),
        scale_m=0.01,
        likelihood_weight=8.0,
        compare_displacements=True,
        anchor_positions_m=bank.trajectories[1, 0, 0, nodes],
    )
    weights = bank.update_from_sparse_evidence(evidence)
    prediction = bank.predictive_distribution(weights, method="sparse_evidence")
    assert weights[1, 0] > 0.99
    assert prediction.mean.shape == (7, 2, 3)
    assert prediction.interval_lower is not None
    assert np.all(prediction.interval_lower <= prediction.interval_upper)


def test_observation_update_scores_discrepancy_as_readout_not_state() -> None:
    trajectories = np.zeros((1, 2, 5, 1, 3), dtype=float)
    bank = JointRolloutBank(
        hypothesis_ids=("nominal",),
        hypothesis_metadata=({"contact": {}},),
        hypothesis_prior_weights=np.asarray([1.0]),
        parameter_particles=np.asarray([[0.0], [1.0]]),
        parameter_weights=np.asarray([0.5, 0.5]),
        trajectories=trajectories,
    )
    observations = np.zeros((5, 1, 3), dtype=float)
    observations[1:, 0, 0] = 0.01
    discrepancy = np.zeros((2, 1, 3), dtype=float)
    discrepancy[1, 0, 0] = 0.01
    unchanged = bank.trajectories.copy()
    posterior = bank.update_from_observations(
        observations,
        prefix_frame_count=4,
        scale_m=0.001,
        likelihood_power=20.0,
        particle_discrepancy_m=discrepancy,
        particle_discrepancy_variance_m2=np.zeros_like(discrepancy),
    )
    assert posterior[0, 1] > 0.99
    assert np.array_equal(bank.trajectories, unchanged)
