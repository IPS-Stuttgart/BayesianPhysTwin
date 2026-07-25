import numpy as np

from bayesian_phystwin.phystwin_bias_aware_ray import (
    decide_prefix_admission,
    remove_affine_ray_nuisance,
)
from bayesian_phystwin.phystwin_cotracker3_cues import (
    CoTracker3RayDiscrepancyPosterior,
)


def _ray(mean: np.ndarray) -> CoTracker3RayDiscrepancyPosterior:
    count = len(mean)
    return CoTracker3RayDiscrepancyPosterior(
        mean_m=np.asarray(mean, dtype=float),
        variance_m2=np.full(count, 1e-6),
        observed=np.ones(count, dtype=bool),
        update_count=np.full(count, 3, dtype=np.int64),
        final_inlier_probability=np.full(count, 0.8),
        camera_support=np.full(count, 3, dtype=np.int16),
    )


def test_affine_ray_field_is_treated_as_nuisance() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    coefficients = np.array(
        [
            [0.1, 0.0, 0.0],
            [0.0, -0.2, 0.0],
            [0.0, 0.0, 0.3],
            [0.01, -0.02, 0.03],
        ]
    )
    mean = np.column_stack((positions, np.ones(len(positions)))) @ coefficients

    endpoint = remove_affine_ray_nuisance(
        _ray(mean),
        positions,
        unobserved_variance_m2=0.05**2,
    )

    np.testing.assert_allclose(endpoint.posterior.mean, 0.0, atol=1e-12)
    assert np.all(endpoint.posterior.variance > 1e-6)
    assert endpoint.diagnostics.affine_explained_energy_fraction > 0.999999


def test_non_affine_ray_component_is_retained() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ]
    )
    mean = np.zeros_like(positions)
    mean[-1, 2] = 0.02

    endpoint = remove_affine_ray_nuisance(
        _ray(mean),
        positions,
        unobserved_variance_m2=0.05**2,
    )

    assert np.linalg.norm(endpoint.posterior.mean) > 0.0
    assert endpoint.diagnostics.retained_magnitude_median_m >= 0.0


def test_missing_ray_support_returns_zero_update_for_exact_fallback() -> None:
    ray = _ray(np.ones((4, 3)))
    ray = CoTracker3RayDiscrepancyPosterior(
        mean_m=ray.mean_m,
        variance_m2=ray.variance_m2,
        observed=np.zeros(4, dtype=bool),
        update_count=np.zeros(4, dtype=np.int64),
        final_inlier_probability=np.zeros(4),
        camera_support=np.zeros(4, dtype=np.int16),
    )

    endpoint = remove_affine_ray_nuisance(
        ray,
        np.zeros((4, 3)),
        unobserved_variance_m2=0.05**2,
    )

    np.testing.assert_array_equal(endpoint.posterior.mean, 0.0)
    assert endpoint.diagnostics.observed_count == 0


def test_prefix_admission_requires_every_guard() -> None:
    accepted = decide_prefix_admission(
        baseline_all_m=0.02,
        candidate_all_m=0.019,
        baseline_early_m=0.018,
        candidate_early_m=0.0178,
        baseline_late_m=0.022,
        candidate_late_m=0.0202,
        observed_fraction=0.1,
        median_inlier_probability=0.7,
        minimum_observed_fraction=0.02,
        minimum_inlier_probability=0.2,
        minimum_absolute_improvement_m=0.0001,
        minimum_relative_improvement=0.01,
    )
    rejected = decide_prefix_admission(
        baseline_all_m=0.02,
        candidate_all_m=0.019,
        baseline_early_m=0.018,
        candidate_early_m=0.0181,
        baseline_late_m=0.022,
        candidate_late_m=0.0199,
        observed_fraction=0.1,
        median_inlier_probability=0.7,
        minimum_observed_fraction=0.02,
        minimum_inlier_probability=0.2,
        minimum_absolute_improvement_m=0.0001,
        minimum_relative_improvement=0.01,
    )

    assert accepted.accepted
    assert not rejected.accepted
    assert not rejected.gates["early_prefix_nonregression"]
