import numpy as np
import pytest

from bayesian_phystwin import (
    PseudoMeasurementBatch,
    RobustLikelihoodConfig,
    robust_mixture_likelihood,
)


def test_large_residual_has_low_posterior_inlier_probability() -> None:
    batch = PseudoMeasurementBatch(
        observed=[[0.1], [8.0]],
        predicted=[[0.0], [0.0]],
        variance=1.0,
    )

    result = robust_mixture_likelihood(
        batch,
        prior_reliability=np.array([0.9, 0.9]),
        config=RobustLikelihoodConfig(outlier_variance_multiplier=100.0),
    )

    assert result.posterior_inlier_probability[0] > 0.95
    assert result.posterior_inlier_probability[1] < 1e-8
    assert np.all(np.isfinite(result.negative_log_likelihood))


def test_prior_reliability_changes_ambiguous_responsibility() -> None:
    batch = PseudoMeasurementBatch(
        observed=[[1.0], [1.0]],
        predicted=[[0.0], [0.0]],
        variance=1.0,
    )

    result = robust_mixture_likelihood(
        batch,
        prior_reliability=np.array([0.9, 0.1]),
    )

    assert result.posterior_inlier_probability[0] > result.posterior_inlier_probability[1]


def test_invalid_outlier_variance_multiplier_raises() -> None:
    batch = PseudoMeasurementBatch(observed=[[0.0]], predicted=[[0.0]])

    with pytest.raises(ValueError):
        robust_mixture_likelihood(
            batch,
            config=RobustLikelihoodConfig(outlier_variance_multiplier=1.0),
        )


def test_model_discrepancy_is_separate_additive_variance() -> None:
    batch = PseudoMeasurementBatch(
        observed=[[0.2]],
        predicted=[[0.0]],
        variance=0.01,
    )

    without_discrepancy = robust_mixture_likelihood(
        batch,
        prior_reliability=np.array([0.9]),
    )
    with_discrepancy = robust_mixture_likelihood(
        batch,
        prior_reliability=np.array([0.9]),
        config=RobustLikelihoodConfig(model_discrepancy_variance=0.03),
    )

    assert (
        with_discrepancy.posterior_inlier_probability[0]
        > without_discrepancy.posterior_inlier_probability[0]
    )
