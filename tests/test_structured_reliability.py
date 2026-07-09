import numpy as np
import pytest

from bayesian_phystwin import (
    MarkovReliabilityConfig,
    markov_log_evidence_batch,
    smooth_markov_reliability,
)


def test_markov_smoothing_uses_future_evidence_for_persistent_corruption() -> None:
    prior = np.full(8, 0.75)
    log_inlier = np.array([0.0, 0.0, -0.2, -0.3, -0.5, -1.0, -8.0, -10.0])
    log_outlier = np.array([-5.0, -5.0, -0.1, -0.1, -0.1, 0.0, 0.0, 0.0])
    iid = np.exp(np.log(prior) + log_inlier)
    iid /= iid + np.exp(np.log1p(-prior) + log_outlier)

    result = smooth_markov_reliability(
        prior,
        log_inlier,
        log_outlier,
        sequence_ids=["track-1"] * 8,
        time_values=np.arange(8),
        config=MarkovReliabilityConfig(
            inlier_persistence=0.98,
            outlier_persistence=0.98,
        ),
    )

    assert result.posterior_inlier_probability[5] < iid[5]
    assert result.posterior_inlier_probability[-1] < 1e-4
    assert result.sequence_count == 1
    assert np.isfinite(result.total_log_evidence)


def test_sequences_are_smoothed_independently_and_returned_in_input_order() -> None:
    result = smooth_markov_reliability(
        prior_reliability=np.full(4, 0.8),
        log_inlier_density=np.array([0.0, -10.0, 0.0, 0.0]),
        log_outlier_density=np.array([-5.0, 0.0, -5.0, -5.0]),
        sequence_ids=["a", "a", "b", "b"],
        time_values=[1, 0, 1, 0],
    )

    assert result.posterior_inlier_probability.shape == (4,)
    assert set(result.sequence_log_evidence) == {"a", "b"}
    assert result.posterior_inlier_probability[1] < result.posterior_inlier_probability[0]
    assert result.posterior_inlier_probability[2] > 0.9


def test_invalid_persistence_raises() -> None:
    with pytest.raises(ValueError):
        smooth_markov_reliability(
            np.array([0.5]),
            np.array([0.0]),
            np.array([0.0]),
            ["a"],
            [0],
            config=MarkovReliabilityConfig(inlier_persistence=1.0),
        )


def test_batched_evidence_matches_scalar_sequence_evidence() -> None:
    prior = np.array([0.8, 0.7, 0.6, 0.9])
    log_inlier = np.array(
        [
            [0.0, -0.1, -2.0, -3.0],
            [-1.0, -0.5, -0.2, -0.1],
        ]
    )
    log_outlier = np.array(
        [
            [-2.0, -1.0, -0.1, 0.0],
            [-0.2, -0.2, -1.0, -2.0],
        ]
    )
    ids = ["a", "a", "b", "b"]
    times = [0, 1, 0, 1]

    batched = markov_log_evidence_batch(prior, log_inlier, log_outlier, ids, times)
    scalar = np.array(
        [
            smooth_markov_reliability(
                prior,
                log_inlier[index],
                log_outlier[index],
                ids,
                times,
            ).total_log_evidence
            for index in range(2)
        ]
    )

    assert np.allclose(batched, scalar)
