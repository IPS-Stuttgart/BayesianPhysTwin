import numpy as np

from bayesian_phystwin import (
    RandomWalkBiasConfig,
    filter_random_walk_bias,
    robust_random_walk_log_evidence_batch,
)


def test_random_walk_bias_tracks_gradual_drift_without_following_gross_outlier() -> None:
    rng = np.random.default_rng(2)
    drift = np.concatenate([np.zeros(10), np.linspace(0.0, 0.06, 20)])
    residual = drift + rng.normal(scale=0.003, size=drift.size)
    residual[15] += 0.20
    prior = np.full(drift.size, 0.95)
    prior[15] = 0.05

    result = filter_random_walk_bias(
        prior,
        residual,
        0.003**2,
        sequence_ids=["track"] * drift.size,
        time_values=np.arange(drift.size),
        config=RandomWalkBiasConfig(
            process_variance=2.5e-5,
            initial_variance=1e-6,
        ),
        bias_probability=np.concatenate([np.zeros(10), np.ones(20)]),
    )

    assert abs(result.bias_mean[-1] - drift[-1]) < 0.01
    assert result.inlier_probability[15] < 0.1
    assert abs(result.bias_mean[15] - result.bias_mean[14]) < 0.03
    assert np.isfinite(result.total_log_evidence)


def test_batched_bias_evidence_prefers_correct_shared_physics_hypothesis() -> None:
    time = np.arange(25)
    drift = np.linspace(0.0, 0.04, time.size)
    correct = drift
    wrong = drift + 0.08
    residual = np.vstack([correct, wrong])

    evidence = robust_random_walk_log_evidence_batch(
        np.full(time.size, 0.9),
        residual,
        0.005**2,
        sequence_ids=["track"] * time.size,
        time_values=time,
    )

    assert evidence.shape == (2,)
    assert evidence[0] > evidence[1]


def test_multiple_sequences_have_independent_bias_states() -> None:
    result = filter_random_walk_bias(
        np.full(6, 0.9),
        np.array([0.0, 0.01, 0.02, 0.0, -0.01, -0.02]),
        0.002**2,
        sequence_ids=["a", "a", "a", "b", "b", "b"],
        time_values=[0, 1, 2, 0, 1, 2],
    )

    assert result.bias_mean[2] > 0.0
    assert result.bias_mean[5] < 0.0
    assert set(result.sequence_log_evidence) == {"a", "b"}
