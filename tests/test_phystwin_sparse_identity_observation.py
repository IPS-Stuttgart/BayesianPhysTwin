from pathlib import Path

import numpy as np

from bayesian_phystwin.phystwin_cotracker3_cues import project_world_points
from bayesian_phystwin.phystwin_sparse_identity_observation import (
    SparseIdentityObservationConfig,
    load_cotracker3_sparse_identity_observations,
    sparse_identity_endpoint,
)


def _archive_arrays(
    *,
    frame_count: int = 3,
    third_view_quality: float = 1.0,
) -> dict[str, np.ndarray]:
    intrinsics = np.repeat(np.eye(3)[None], 3, axis=0)
    intrinsics[:, 0, 0] = 200.0
    intrinsics[:, 1, 1] = 200.0
    intrinsics[:, 0, 2] = 50.0
    intrinsics[:, 1, 2] = 50.0
    camera_to_world = np.repeat(np.eye(4)[None], 3, axis=0)
    camera_to_world[1, 0, 3] = 1.0
    camera_to_world[2, 1, 3] = 1.0
    points = np.zeros((frame_count, 1, 3), dtype=float)
    points[:, 0, 0] = np.linspace(0.25, 0.35, frame_count)
    points[:, 0, 1] = -0.1
    points[:, 0, 2] = 5.0
    tracks = np.empty((3, frame_count, 1, 2), dtype=float)
    for camera in range(3):
        for frame in range(frame_count):
            tracks[camera, frame], _ = project_world_points(
                points[frame],
                intrinsics[camera],
                camera_to_world[camera],
            )
    quality = np.ones((3, frame_count, 1), dtype=float)
    quality[2] = third_view_quality
    return {
        "multiview_tracks_xy_prefix": tracks,
        "multiview_quality_probability_prefix": quality,
        "multiview_view_valid_prefix": np.ones(
            (3, frame_count, 1),
            dtype=bool,
        ),
        "multiview_intrinsics": intrinsics,
        "multiview_camera_to_world": camera_to_world,
        "forward_backward_error_px": np.zeros((frame_count + 2, 1)),
        "forward_backward_valid": np.ones((frame_count + 2, 1), dtype=bool),
        "boundary_distance": np.full((frame_count + 2, 1), 20.0),
        "cue_available": np.concatenate(
            [
                np.ones((frame_count, 1), dtype=bool),
                np.zeros((2, 1), dtype=bool),
            ]
        ),
    }


def _write_archive(path: Path, **kwargs: object) -> dict[str, np.ndarray]:
    arrays = _archive_arrays(**kwargs)
    np.savez_compressed(path, **arrays)
    return arrays


def _config() -> SparseIdentityObservationConfig:
    return SparseIdentityObservationConfig(
        minimum_view_quality=0.5,
        maximum_cycle_error_px=1.0,
        maximum_reprojection_error_px=1.0,
        pixel_noise_std=0.5,
        shared_bias_std_m=0.001,
        two_view_extra_std_m=0.01,
        minimum_ray_angle_degrees=0.1,
    )


def test_two_view_fallback_is_valid_but_more_uncertain_than_redundant_views(
    tmp_path: Path,
) -> None:
    redundant_path = tmp_path / "redundant.npz"
    two_view_path = tmp_path / "two-view.npz"
    _write_archive(redundant_path)
    _write_archive(two_view_path, third_view_quality=0.1)
    initial = np.array([[10.0, 20.0, 30.0]])

    redundant = load_cotracker3_sparse_identity_observations(
        redundant_path,
        initial,
        train_end_frame=3,
        config=_config(),
    )
    two_view = load_cotracker3_sparse_identity_observations(
        two_view_path,
        initial,
        train_end_frame=3,
        config=_config(),
    )

    assert np.all(redundant.valid)
    assert np.all(two_view.valid)
    np.testing.assert_array_equal(redundant.effective_camera_count, 3)
    np.testing.assert_array_equal(two_view.effective_camera_count, 2)
    assert not np.any(redundant.two_view_fallback)
    assert np.all(two_view.two_view_fallback)
    assert np.all(two_view.observation_variance_m2 > redundant.observation_variance_m2)
    assert np.all(two_view.prior_reliability < redundant.prior_reliability)
    np.testing.assert_allclose(
        redundant.points_world_m[:, 0],
        [[10.0, 20.0, 30.0], [10.05, 20.0, 30.0], [10.1, 20.0, 30.0]],
        atol=1e-7,
    )


def test_duplicate_camera_is_collapsed_before_support_and_covariance(
    tmp_path: Path,
) -> None:
    original_path = tmp_path / "original.npz"
    duplicate_path = tmp_path / "duplicate.npz"
    arrays = _write_archive(original_path, third_view_quality=0.1)
    camera_fields = {
        "multiview_tracks_xy_prefix",
        "multiview_quality_probability_prefix",
        "multiview_view_valid_prefix",
        "multiview_intrinsics",
        "multiview_camera_to_world",
    }
    duplicated = {
        name: (
            np.concatenate([value, value[:1]], axis=0)
            if name in camera_fields
            else value
        )
        for name, value in arrays.items()
    }
    np.savez_compressed(duplicate_path, **duplicated)
    initial = np.array([[0.25, -0.1, 5.0]])

    original = load_cotracker3_sparse_identity_observations(
        original_path,
        initial,
        train_end_frame=3,
        config=_config(),
    )
    duplicate = load_cotracker3_sparse_identity_observations(
        duplicate_path,
        initial,
        train_end_frame=3,
        config=_config(),
    )

    np.testing.assert_array_equal(original.effective_camera_count, 2)
    np.testing.assert_array_equal(duplicate.effective_camera_count, 2)
    np.testing.assert_array_equal(original.raw_camera_count, 2)
    np.testing.assert_array_equal(duplicate.raw_camera_count, 3)
    np.testing.assert_allclose(
        duplicate.observation_covariance_m2,
        original.observation_covariance_m2,
        rtol=0.0,
        atol=0.0,
    )
    naive_independent = original.observation_covariance_m2 / 2.0
    covariance_difference = duplicate.observation_covariance_m2 - naive_independent
    assert np.all(np.linalg.eigvalsh(covariance_difference) >= -1e-12)
    np.testing.assert_allclose(
        duplicate.prior_reliability,
        original.prior_reliability,
        rtol=0.0,
        atol=0.0,
    )


def test_duplicate_views_alone_do_not_satisfy_two_view_support(
    tmp_path: Path,
) -> None:
    path = tmp_path / "duplicates-only.npz"
    arrays = _archive_arrays()
    camera_fields = {
        "multiview_tracks_xy_prefix",
        "multiview_quality_probability_prefix",
        "multiview_view_valid_prefix",
        "multiview_intrinsics",
        "multiview_camera_to_world",
    }
    duplicated = {
        name: (
            np.concatenate([value[:1], value[:1]], axis=0)
            if name in camera_fields
            else value
        )
        for name, value in arrays.items()
    }
    np.savez_compressed(path, **duplicated)

    observations = load_cotracker3_sparse_identity_observations(
        path,
        np.array([[0.25, -0.1, 5.0]]),
        train_end_frame=3,
        config=_config(),
    )

    np.testing.assert_array_equal(observations.raw_camera_count, 2)
    np.testing.assert_array_equal(observations.effective_camera_count, 1)
    assert not np.any(observations.valid)
    assert np.all(np.isnan(observations.points_world_m))


def test_future_cue_mutation_cannot_change_prefix_observation(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.npz"
    second_path = tmp_path / "second.npz"
    arrays = _write_archive(first_path, frame_count=4)
    changed = {name: value.copy() for name, value in arrays.items()}
    changed["forward_backward_error_px"][2:] = 1e6
    changed["forward_backward_valid"][2:] = False
    changed["boundary_distance"][2:] = -1e6
    changed["cue_available"][2:] = False
    changed["multiview_tracks_xy_prefix"][:, 2:] += 1e6
    changed["multiview_quality_probability_prefix"][:, 2:] = 0.0
    changed["multiview_view_valid_prefix"][:, 2:] = False
    np.savez_compressed(second_path, **changed)
    initial = np.array([[0.25, -0.1, 5.0]])

    first = load_cotracker3_sparse_identity_observations(
        first_path,
        initial,
        train_end_frame=2,
        config=_config(),
    )
    second = load_cotracker3_sparse_identity_observations(
        second_path,
        initial,
        train_end_frame=2,
        config=_config(),
    )

    for name in (
        "points_world_m",
        "observation_covariance_m2",
        "observation_variance_m2",
        "prior_reliability",
        "valid",
        "raw_camera_count",
        "effective_camera_count",
    ):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))


def test_prior_reliability_is_independent_of_phystwin_state_residual(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cues.npz"
    _write_archive(path)
    observations = load_cotracker3_sparse_identity_observations(
        path,
        np.array([[0.25, -0.1, 5.0]]),
        train_end_frame=3,
        config=_config(),
    )
    near = observations.points_world_m.copy()
    far = near.copy()
    far[..., 0] += 0.5
    reliability_before = observations.prior_reliability.copy()

    near_endpoint = sparse_identity_endpoint(
        observations,
        near,
        end_frame=3,
        process_variance=1e-6,
        initial_variance=1e-4,
        inlier_prior=0.95,
        outlier_variance_multiplier=100.0,
    )
    far_endpoint = sparse_identity_endpoint(
        observations,
        far,
        end_frame=3,
        process_variance=1e-6,
        initial_variance=1e-4,
        inlier_prior=0.95,
        outlier_variance_multiplier=100.0,
    )

    np.testing.assert_array_equal(
        observations.prior_reliability,
        reliability_before,
    )
    assert (
        far_endpoint.final_inlier_probability[0]
        < near_endpoint.final_inlier_probability[0]
    )


def test_robust_endpoint_rejects_one_gross_sparse_identity_outlier(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cues.npz"
    arrays = _write_archive(path, frame_count=4)
    corrupted = {name: value.copy() for name, value in arrays.items()}
    wrong_point = np.array([[0.9, -0.1, 1.0]])
    for camera in range(3):
        xy, _ = project_world_points(
            wrong_point,
            corrupted["multiview_intrinsics"][camera],
            corrupted["multiview_camera_to_world"][camera],
        )
        corrupted["multiview_tracks_xy_prefix"][camera, 3] = xy
    np.savez_compressed(path, **corrupted)
    observations = load_cotracker3_sparse_identity_observations(
        path,
        np.array([[0.25, -0.1, 5.0]]),
        train_end_frame=4,
        config=_config(),
    )
    baseline = observations.points_world_m.copy()
    baseline[3] = baseline[2]

    endpoint = sparse_identity_endpoint(
        observations,
        baseline,
        end_frame=4,
        process_variance=1e-8,
        initial_variance=1e-4,
        inlier_prior=0.95,
        outlier_variance_multiplier=100.0,
    )

    assert endpoint.final_inlier_probability[0] < 0.5
    assert endpoint.update_count[0] == 4
