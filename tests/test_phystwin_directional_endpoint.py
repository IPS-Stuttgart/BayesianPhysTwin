import numpy as np

from bayesian_phystwin.phystwin_bayesian_anchor import (
    robust_random_walk_endpoint,
)
from bayesian_phystwin.phystwin_directional_endpoint import (
    robust_directional_endpoint,
)


def _tangent_projectors(point_count: int) -> np.ndarray:
    projector = np.diag([1.0, 1.0, 0.0])
    return np.repeat(projector[None], point_count, axis=0)


def _run(
    source: np.ndarray,
    source_valid: np.ndarray,
    multiview: np.ndarray,
    multiview_valid: np.ndarray,
    priority: np.ndarray,
):
    return robust_directional_endpoint(
        source,
        source_valid,
        multiview,
        multiview_valid,
        _tangent_projectors(source.shape[1]),
        priority,
        end_frame=len(source),
        process_variance=1e-6,
        observation_variance=1e-4,
        initial_variance=1e-3,
        inlier_prior=0.95,
        outlier_variance_multiplier=100.0,
    )


def test_nonpriority_source_path_matches_existing_robust_filter() -> None:
    source = np.array(
        [
            [[0.01, -0.02, 0.03]],
            [[0.02, -0.01, 0.04]],
            [[0.03, 0.00, 0.05]],
        ]
    )
    valid = np.ones(source.shape[:2], dtype=bool)
    multiview = np.zeros_like(source)
    result = _run(
        source,
        valid,
        multiview,
        np.zeros_like(valid),
        np.array([False]),
    )
    reference = robust_random_walk_endpoint(
        source,
        valid,
        end_frame=len(source),
        process_variance=1e-6,
        observation_variance=1e-4,
        initial_variance=1e-3,
        inlier_prior=0.95,
        outlier_variance_multiplier=100.0,
    )

    np.testing.assert_allclose(result.mean, reference.mean, atol=1e-12)
    np.testing.assert_allclose(result.variance, reference.variance, atol=1e-12)
    np.testing.assert_array_equal(result.update_count, reference.update_count)


def test_priority_routes_source_normal_and_multiview_tangent() -> None:
    source = np.array([[[100.0, 100.0, 0.02]]])
    multiview = np.array([[[0.03, -0.01, 100.0]]])
    valid = np.ones((1, 1), dtype=bool)

    result = _run(
        source,
        valid,
        multiview,
        valid,
        np.array([True]),
    )

    assert result.mean[0, 0] > 0.0
    assert result.mean[0, 1] < 0.0
    assert result.mean[0, 2] > 0.0
    assert abs(result.mean[0, 0]) < 0.03
    assert abs(result.mean[0, 2]) < 0.02
    np.testing.assert_array_equal(result.source_update_count, [1])
    np.testing.assert_array_equal(result.tangent_update_count, [1])


def test_source_tangent_residual_cannot_change_priority_posterior() -> None:
    source_a = np.array([[[0.0, 0.0, 0.02]]])
    source_b = np.array([[[50.0, -70.0, 0.02]]])
    multiview = np.array([[[0.03, -0.01, 0.0]]])
    valid = np.ones((1, 1), dtype=bool)

    result_a = _run(
        source_a,
        valid,
        multiview,
        valid,
        np.array([True]),
    )
    result_b = _run(
        source_b,
        valid,
        multiview,
        valid,
        np.array([True]),
    )

    np.testing.assert_allclose(result_a.mean, result_b.mean, atol=1e-12)
    np.testing.assert_allclose(
        result_a.covariance,
        result_b.covariance,
        atol=1e-12,
    )


def test_tangent_updates_continue_when_source_depth_is_missing() -> None:
    source = np.array(
        [
            [[0.0, 0.0, 0.02]],
            [[0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
        ]
    )
    source_valid = np.array([[True], [False], [False]])
    multiview = np.array(
        [
            [[0.01, 0.0, 0.0]],
            [[0.02, 0.0, 0.0]],
            [[0.03, 0.0, 0.0]],
        ]
    )
    multiview_valid = np.ones((3, 1), dtype=bool)

    result = _run(
        source,
        source_valid,
        multiview,
        multiview_valid,
        np.array([True]),
    )

    assert result.mean[0, 0] > 0.015
    assert result.mean[0, 2] > 0.0
    np.testing.assert_array_equal(result.source_update_count, [1])
    np.testing.assert_array_equal(result.tangent_update_count, [3])


def test_gross_tangent_outlier_is_robustly_downweighted() -> None:
    source = np.zeros((1, 1, 3))
    source_valid = np.zeros((1, 1), dtype=bool)
    valid = np.ones((1, 1), dtype=bool)
    inlier = _run(
        source,
        source_valid,
        np.array([[[0.01, 0.0, 0.0]]]),
        valid,
        np.array([True]),
    )
    outlier = _run(
        source,
        source_valid,
        np.array([[[10.0, 0.0, 0.0]]]),
        valid,
        np.array([True]),
    )

    assert outlier.final_inlier_probability[0] < inlier.final_inlier_probability[0]
    assert abs(outlier.mean[0, 0]) < 1.0
    assert outlier.variance[0] >= np.max(
        np.linalg.eigvalsh(outlier.covariance[0])
    )
