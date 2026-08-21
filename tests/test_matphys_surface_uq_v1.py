from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.matphys_surface_uq_v1 import (
    backproject_masked_depth,
    deterministic_camera_partition,
    deterministic_subsample_indices,
    evaluate_gaussian_events,
    fit_isotropic_variance,
    fit_matphys_scale,
    isotropic_total_covariance,
    matphys_total_covariance,
    nearest_surface_events,
)


def test_camera_partition_is_disjoint_and_canonical() -> None:
    provider, scoring = deterministic_camera_partition(
        ["cam3", "cam1", "cam2", "cam0"],
        scoring_camera_ids=["cam3", "cam1"],
    )

    assert provider == ("cam0", "cam2")
    assert scoring == ("cam1", "cam3")
    assert not set(provider) & set(scoring)


def test_camera_partition_rejects_overlap_contract_failures() -> None:
    with pytest.raises(ValueError, match="repeats"):
        deterministic_camera_partition(
            ["cam0", "cam1", "cam1", "cam2"],
            scoring_camera_ids=["cam2", "cam1"],
        )
    with pytest.raises(ValueError, match="unavailable"):
        deterministic_camera_partition(
            ["cam0", "cam1", "cam2", "cam3"],
            scoring_camera_ids=["cam2", "missing"],
        )


def test_subsample_is_key_bound_and_reproducible() -> None:
    first = deterministic_subsample_indices(100, 9, key="case/camera/frame")
    second = deterministic_subsample_indices(100, 9, key="case/camera/frame")
    other = deterministic_subsample_indices(100, 9, key="case/camera/other")

    np.testing.assert_array_equal(first, second)
    assert len(np.unique(first)) == 9
    assert not np.array_equal(first, other)


def test_backprojection_uses_metric_depth_and_camera_to_world() -> None:
    depth = np.array([[1.0, 2.0], [0.0, 1.0]], dtype=np.float32)
    mask = np.array([[True, True], [True, False]])
    intrinsics = np.array(
        [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]
    )
    transform = np.eye(4)
    transform[:3, 3] = [1.0, 2.0, 3.0]

    points = backproject_masked_depth(
        depth,
        mask,
        intrinsics,
        transform,
        maximum_points=10,
        subsample_key="fixture",
    )

    np.testing.assert_allclose(points, [[1.0, 2.0, 4.0], [2.0, 2.0, 5.0]])


def test_nearest_surface_events_reject_far_nodes() -> None:
    mean = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ]
    )
    covariance = np.broadcast_to(np.eye(3) * 1e-4, (2, 2, 3, 3)).copy()
    clouds = [
        np.array([[0.01, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        np.array([[0.02, 0.0, 0.0]]),
    ]

    events = nearest_surface_events(
        mean,
        covariance,
        clouds,
        maximum_distance_m=0.05,
    )

    assert events.attempted_event_count == 4
    assert events.accepted_event_count == 2
    np.testing.assert_array_equal(events.node_index, [0, 0])
    np.testing.assert_allclose(events.residual_m[:, 0], [0.01, 0.02])


def test_matphys_scale_improves_anisotropic_likelihood() -> None:
    generator = np.random.default_rng(260822)
    base_covariance = np.diag([9e-6, 1e-6, 4e-6])
    covariance = np.broadcast_to(base_covariance, (4000, 3, 3)).copy()
    residual = generator.multivariate_normal(
        np.zeros(3),
        5.0 * base_covariance + np.eye(3) * 1e-6,
        size=4000,
    )

    scale = fit_matphys_scale(
        residual,
        covariance,
        observation_floor_m=0.001,
    )
    matphys = matphys_total_covariance(
        covariance,
        scale=scale,
        observation_floor_m=0.001,
    )
    isotropic_variance = fit_isotropic_variance(
        residual,
        observation_floor_m=0.001,
    )
    isotropic = isotropic_total_covariance(
        len(residual), variance_m2=isotropic_variance
    )
    matphys_metrics = evaluate_gaussian_events(residual, matphys)
    isotropic_metrics = evaluate_gaussian_events(residual, isotropic)

    assert 4.5 < scale < 5.5
    assert matphys_metrics["mean_nll"] < isotropic_metrics["mean_nll"]
    assert 0.87 < matphys_metrics["coverage_90"] < 0.93


def test_covariance_floor_keeps_zero_spread_positive_definite() -> None:
    residual = np.zeros((3, 3), dtype=np.float64)
    covariance = np.zeros((3, 3, 3), dtype=np.float64)

    total = matphys_total_covariance(
        covariance,
        scale=0.0,
        observation_floor_m=0.005,
    )
    metrics = evaluate_gaussian_events(residual, total)

    assert metrics["coverage_90"] == 1.0
    assert np.all(np.linalg.eigvalsh(total) > 0.0)
