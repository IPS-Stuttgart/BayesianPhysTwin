import numpy as np

from causal4d.parameter_support import (
    reduce_parameter_support,
    weighted_parameter_moments,
)


def _posterior() -> tuple[np.ndarray, np.ndarray]:
    particles = np.asarray(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ]
    )
    weights = np.asarray([0.10, 0.20, 0.15, 0.05, 0.50])
    return particles, weights


def test_top_mass_reports_discarded_probability() -> None:
    particles, weights = _posterior()
    reduction = reduce_parameter_support(
        particles,
        weights,
        maximum_count=2,
        method="top_mass",
    )
    np.testing.assert_array_equal(reduction.indices, [4, 1])
    assert np.isclose(reduction.directly_retained_probability_mass, 0.70)
    assert np.isclose(reduction.represented_probability_mass, 0.70)
    assert np.isclose(np.sum(reduction.weights), 1.0)


def test_weighted_coreset_is_deterministic_and_represents_full_mass() -> None:
    particles, weights = _posterior()
    first = reduce_parameter_support(
        particles,
        weights,
        maximum_count=3,
        method="weighted_coreset",
    )
    second = reduce_parameter_support(
        particles,
        weights,
        maximum_count=3,
        method="weighted_coreset",
    )
    np.testing.assert_array_equal(first.indices, second.indices)
    np.testing.assert_array_equal(first.weights, second.weights)
    assert first.represented_probability_mass == 1.0
    assert first.directly_retained_probability_mass < 1.0
    assert first.covariance_error_frobenius < 0.30


def test_full_support_preserves_parameter_moments_for_both_methods() -> None:
    particles, weights = _posterior()
    expected_mean, expected_covariance = weighted_parameter_moments(
        particles,
        weights,
    )
    for method in ("top_mass", "weighted_coreset"):
        reduction = reduce_parameter_support(
            particles,
            weights,
            maximum_count=len(weights),
            method=method,
        )
        mean, covariance = weighted_parameter_moments(
            particles[reduction.indices],
            reduction.weights,
        )
        np.testing.assert_allclose(mean, expected_mean, atol=1e-15)
        np.testing.assert_allclose(covariance, expected_covariance, atol=1e-15)
        assert reduction.represented_probability_mass == 1.0
