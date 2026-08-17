import numpy as np
import pytest

from bayesian_phystwin import (
    PseudoMeasurementBatch,
    RobustLikelihoodConfig,
    RobustLikelihoodResult,
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

    assert (
        result.posterior_inlier_probability[0] > result.posterior_inlier_probability[1]
    )


def test_invalid_outlier_variance_multiplier_raises() -> None:
    with pytest.raises(ValueError):
        RobustLikelihoodConfig(outlier_variance_multiplier=1.0)


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


@pytest.mark.parametrize("prior", [[-0.01], [1.01], [np.nan], [np.inf]])
def test_invalid_prior_reliability_fails_closed(prior: list[float]) -> None:
    batch = PseudoMeasurementBatch(observed=[[0.0]], predicted=[[0.0]])

    with pytest.raises(ValueError, match="prior_reliability"):
        robust_mixture_likelihood(batch, prior_reliability=np.asarray(prior))


def test_exact_zero_and_one_prior_reliability_are_supported() -> None:
    batch = PseudoMeasurementBatch(
        observed=[[0.0], [0.0]],
        predicted=[[0.0], [0.0]],
    )

    result = robust_mixture_likelihood(
        batch,
        prior_reliability=np.array([0.0, 1.0]),
    )

    assert np.all(np.isfinite(result.negative_log_likelihood))
    assert np.all(0.0 <= result.posterior_inlier_probability)
    assert np.all(result.posterior_inlier_probability <= 1.0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("outlier_variance_multiplier", np.inf),
        ("outlier_variance_multiplier", True),
        ("model_discrepancy_variance", np.nan),
        ("probability_floor", np.nan),
    ],
)
def test_nonfinite_or_boolean_likelihood_config_fails_closed(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=field):
        RobustLikelihoodConfig(**{field: value})


def test_string_encoded_prior_reliability_fails_closed() -> None:
    batch = PseudoMeasurementBatch(observed=[[0.0]], predicted=[[0.0]])

    with pytest.raises(ValueError, match="real numeric"):
        robust_mixture_likelihood(
            batch,
            prior_reliability=np.array(["0.5"]),
        )


def test_likelihood_result_defensively_owns_read_only_arrays() -> None:
    nll = np.array([1.0])
    posterior = np.array([0.5])
    log_inlier = np.array([-1.0])
    log_outlier = np.array([-2.0])

    result = RobustLikelihoodResult(
        negative_log_likelihood=nll,
        posterior_inlier_probability=posterior,
        log_inlier_density=log_inlier,
        log_outlier_density=log_outlier,
    )
    nll[0] = 9.0
    posterior[0] = 0.9

    assert result.negative_log_likelihood[0] == pytest.approx(1.0)
    assert result.posterior_inlier_probability[0] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="read-only"):
        result.posterior_inlier_probability[0] = 0.1


def test_unrepresentable_density_fails_closed() -> None:
    batch = PseudoMeasurementBatch(
        observed=[[1e308]],
        predicted=[[0.0]],
        variance=1.0,
    )

    with pytest.raises(ValueError, match="log density"):
        robust_mixture_likelihood(
            batch,
            prior_reliability=np.array([0.5]),
        )


def test_default_reliability_path_is_supported() -> None:
    batch = PseudoMeasurementBatch(
        observed=[[0.0]],
        predicted=[[0.0]],
        confidence=[0.8],
    )

    result = robust_mixture_likelihood(batch)

    assert result.posterior_inlier_probability[0] > 0.8


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("outlier_variance_multiplier", "2.0", "finite real"),
        ("model_discrepancy_variance", -0.1, "nonnegative"),
        ("probability_floor", 0.0, "probability_floor"),
        ("probability_floor", 0.5, "probability_floor"),
    ],
)
def test_out_of_domain_likelihood_config_fails_closed(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RobustLikelihoodConfig(**{field: value})


def test_wrong_likelihood_config_type_fails_closed() -> None:
    batch = PseudoMeasurementBatch(observed=[[0.0]], predicted=[[0.0]])

    with pytest.raises(TypeError, match="RobustLikelihoodConfig"):
        robust_mixture_likelihood(batch, config=object())  # type: ignore[arg-type]


def test_prior_shape_mismatch_fails_closed() -> None:
    batch = PseudoMeasurementBatch(observed=[[0.0]], predicted=[[0.0]])

    with pytest.raises(ValueError, match="shape"):
        robust_mixture_likelihood(
            batch,
            prior_reliability=np.array([0.5, 0.5]),
        )


def test_unrepresentable_residual_fails_closed() -> None:
    batch = PseudoMeasurementBatch(
        observed=[[1e308]],
        predicted=[[-1e308]],
    )

    with pytest.raises(ValueError, match="residual"):
        robust_mixture_likelihood(
            batch,
            prior_reliability=np.array([0.5]),
        )


def test_unrepresentable_inlier_variance_fails_closed() -> None:
    batch = PseudoMeasurementBatch(
        observed=[[0.0]],
        predicted=[[0.0]],
        variance=1e308,
    )

    with pytest.raises(ValueError, match="inlier variance"):
        robust_mixture_likelihood(
            batch,
            prior_reliability=np.array([0.5]),
            config=RobustLikelihoodConfig(model_discrepancy_variance=1e308),
        )


def test_unrepresentable_outlier_variance_fails_closed() -> None:
    batch = PseudoMeasurementBatch(
        observed=[[0.0]],
        predicted=[[0.0]],
        variance=1e308,
    )

    with pytest.raises(ValueError, match="outlier variance"):
        robust_mixture_likelihood(
            batch,
            prior_reliability=np.array([0.5]),
            config=RobustLikelihoodConfig(outlier_variance_multiplier=2.0),
        )


def test_mean_negative_log_likelihood_is_reported() -> None:
    result = RobustLikelihoodResult(
        negative_log_likelihood=np.array([1.0, 3.0]),
        posterior_inlier_probability=np.array([0.5, 0.5]),
        log_inlier_density=np.array([-1.0, -1.0]),
        log_outlier_density=np.array([-2.0, -2.0]),
    )

    assert result.mean_negative_log_likelihood == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("nll", "posterior", "log_inlier", "log_outlier", "message"),
    [
        (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            "nonempty vectors",
        ),
        (
            np.array([[1.0]]),
            np.array([[0.5]]),
            np.array([[-1.0]]),
            np.array([[-2.0]]),
            "nonempty vectors",
        ),
        (
            np.array([1.0]),
            np.array([0.5, 0.5]),
            np.array([-1.0]),
            np.array([-2.0]),
            "equal shape",
        ),
        (
            np.array([np.nan]),
            np.array([0.5]),
            np.array([-1.0]),
            np.array([-2.0]),
            "finite",
        ),
        (
            np.array([1.0]),
            np.array([1.1]),
            np.array([-1.0]),
            np.array([-2.0]),
            "lie in",
        ),
    ],
)
def test_likelihood_result_rejects_malformed_arrays(
    nll: np.ndarray,
    posterior: np.ndarray,
    log_inlier: np.ndarray,
    log_outlier: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RobustLikelihoodResult(
            negative_log_likelihood=nll,
            posterior_inlier_probability=posterior,
            log_inlier_density=log_inlier,
            log_outlier_density=log_outlier,
        )


def test_likelihood_result_rejects_string_arrays() -> None:
    with pytest.raises(ValueError, match="real numeric"):
        RobustLikelihoodResult(
            negative_log_likelihood=np.array(["1.0"]),
            posterior_inlier_probability=np.array([0.5]),
            log_inlier_density=np.array([-1.0]),
            log_outlier_density=np.array([-2.0]),
        )
