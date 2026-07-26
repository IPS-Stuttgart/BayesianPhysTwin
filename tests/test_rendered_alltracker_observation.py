from __future__ import annotations

import numpy as np

from bayesian_phystwin.rendered_alltracker_observation import (
    RenderedAllTrackerConfig,
    build_rendered_alltracker_observation,
    project_world_points,
)


def _cameras() -> tuple[np.ndarray, np.ndarray]:
    intrinsics = np.repeat(np.eye(3)[None], 3, axis=0)
    intrinsics[:, 0, 0] = 200.0
    intrinsics[:, 1, 1] = 200.0
    intrinsics[:, 0, 2] = 100.0
    intrinsics[:, 1, 2] = 60.0
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[0, 0, 3] = -0.2
    poses[1, 0, 3] = 0.2
    poses[2, 1, 3] = 0.2
    return intrinsics, poses


def _observations(
    point: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    intrinsics, poses = _cameras()
    target = np.asarray([0.03, -0.02, 1.1] if point is None else point)
    pixels = np.stack(
        [
            project_world_points(target[None], intrinsics[camera], poses[camera])[0][0]
            for camera in range(3)
        ]
    )
    tracks = pixels[None, :, None, :]
    quality = np.full((1, 3, 1), 0.9)
    cycle = np.full((1, 3, 1), 0.5)
    support = np.ones((1, 3, 1), dtype=bool)
    return tracks, quality, cycle, support, intrinsics, poses


def test_exact_multiview_geometry_recovers_point() -> None:
    truth = np.array([0.03, -0.02, 1.1])
    tracks, quality, cycle, support, intrinsics, poses = _observations(truth)
    result = build_rendered_alltracker_observation(
        tracks,
        quality,
        cycle,
        support,
        support,
        intrinsics,
        poses,
    )
    assert result.valid[0, 0]
    assert np.allclose(result.points_world_m[0, 0], truth, atol=1e-8)
    assert result.effective_camera_count[0, 0] == 3
    assert not result.two_view_fallback[0, 0]


def test_cycle_failure_is_rejected_before_triangulation() -> None:
    tracks, quality, cycle, support, intrinsics, poses = _observations()
    cycle[:] = 20.0
    result = build_rendered_alltracker_observation(
        tracks,
        quality,
        cycle,
        support,
        support,
        intrinsics,
        poses,
    )
    assert not result.valid[0, 0]
    assert np.isnan(result.points_world_m[0, 0]).all()
    assert result.raw_camera_count[0, 0] == 0


def test_duplicate_camera_does_not_create_redundant_support() -> None:
    tracks, quality, cycle, support, intrinsics, poses = _observations()
    poses[2] = poses[1]
    tracks[:, 2] = tracks[:, 1]
    result = build_rendered_alltracker_observation(
        tracks,
        quality,
        cycle,
        support,
        support,
        intrinsics,
        poses,
    )
    assert result.valid[0, 0]
    assert result.raw_camera_count[0, 0] == 3
    assert result.effective_camera_count[0, 0] == 2
    assert result.two_view_fallback[0, 0]


def test_two_view_unknown_correlation_adds_covariance_floor() -> None:
    tracks, quality, cycle, support, intrinsics, poses = _observations()
    support[:, 2] = False
    tiny_floor = build_rendered_alltracker_observation(
        tracks,
        quality,
        cycle,
        support,
        support,
        intrinsics,
        poses,
        config=RenderedAllTrackerConfig(two_view_extra_std_m=1e-6),
    )
    conservative = build_rendered_alltracker_observation(
        tracks,
        quality,
        cycle,
        support,
        support,
        intrinsics,
        poses,
    )
    assert tiny_floor.valid[0, 0] and conservative.valid[0, 0]
    assert np.trace(conservative.covariance_m2[0, 0]) > np.trace(
        tiny_floor.covariance_m2[0, 0]
    )
    assert conservative.two_view_fallback[0, 0]
    assert conservative.prior_reliability[0, 0] < 0.9


def test_support_is_not_derived_from_physical_residual() -> None:
    tracks, quality, cycle, support, intrinsics, poses = _observations()
    first = build_rendered_alltracker_observation(
        tracks,
        quality,
        cycle,
        support,
        support,
        intrinsics,
        poses,
    )
    shifted_truth = np.array([0.30, -0.20, 1.1])
    shifted_tracks, *_ = _observations(shifted_truth)
    second = build_rendered_alltracker_observation(
        shifted_tracks,
        quality,
        cycle,
        support,
        support,
        intrinsics,
        poses,
    )
    assert first.valid[0, 0] and second.valid[0, 0]
    assert first.prior_reliability[0, 0] == second.prior_reliability[0, 0]


def test_projection_marks_points_behind_camera() -> None:
    pixels, depth = project_world_points(
        np.array([[0.0, 0.0, -1.0]]),
        np.eye(3),
        np.eye(4),
    )
    assert depth[0] < 0.0
    assert np.isnan(pixels[0]).all()
