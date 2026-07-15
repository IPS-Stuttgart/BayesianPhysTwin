from __future__ import annotations

import numpy as np

from causal4d_public.deform360_dense_fusion import (
    DenseVelocityFusionConfig,
    fuse_correlated_velocity_observations,
    graph_complete_velocity,
)


def _fuse(values: np.ndarray):
    valid = np.ones(values.shape[:2], dtype=bool)
    reliability = np.full(values.shape[:2], 0.8, dtype=float)
    return fuse_correlated_velocity_observations(
        values,
        valid,
        reliability,
        DenseVelocityFusionConfig(),
    )


def test_duplicate_correlated_view_does_not_reduce_covariance() -> None:
    value = np.array([[[0.2, -0.1, 0.05]]], dtype=float)
    single = _fuse(value)
    duplicated = _fuse(np.repeat(value, 8, axis=1))

    np.testing.assert_allclose(duplicated.mean_mps, single.mean_mps)
    np.testing.assert_allclose(duplicated.covariance_m2ps2, single.covariance_m2ps2)
    assert duplicated.effective_sample_size[0] <= 2.0
    assert duplicated.contributor_count[0] == 8


def test_unknown_correlation_is_not_more_confident_than_independence() -> None:
    values = np.zeros((1, 2, 3), dtype=float)
    fused = _fuse(values)
    single_variance = (
        np.square(DenseVelocityFusionConfig().base_standard_deviation_mps) / 0.8
    )
    naive_independent = 0.5 * single_variance * np.eye(3)

    difference = fused.covariance_m2ps2[0] - naive_independent
    assert np.min(np.linalg.eigvalsh(difference)) >= -1e-7


def test_robust_likelihood_rejects_gross_view_outlier_once() -> None:
    values = np.array(
        [[[0.1, 0.0, 0.0], [0.11, 0.0, 0.0], [8.0, 0.0, 0.0]]],
        dtype=float,
    )
    fused = _fuse(values)

    assert abs(float(fused.mean_mps[0, 0]) - 0.105) < 0.04
    assert fused.consistency_weight[0, 2] < 0.01
    np.testing.assert_allclose(fused.prior_reliability[0], 0.8)


def test_graph_completion_fills_missing_nodes_from_metric_variance() -> None:
    points = np.stack([np.arange(6) * 0.01, np.zeros(6), np.zeros(6)], axis=1)
    values = np.zeros((6, 1, 3), dtype=float)
    values[0, 0, 1] = 0.25
    valid = np.zeros((6, 1), dtype=bool)
    valid[0, 0] = True
    reliability = valid.astype(float)
    config = DenseVelocityFusionConfig(
        graph_neighbors=2,
        graph_radius_m=0.021,
        graph_prior_strength=10.0,
    )
    fused = fuse_correlated_velocity_observations(values, valid, reliability, config)
    posterior = graph_complete_velocity(points, fused, config)

    assert posterior.directly_observed.tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    np.testing.assert_allclose(posterior.mean_mps[:, 1], 0.25, atol=2e-3)
    assert posterior.springs.shape[1] == 2
    assert max(posterior.solve_relative_residuals) < 1e-5
