import numpy as np
from pathlib import Path

from bayesian_phystwin.phystwin_cotracker3_cues import (
    infer_cotracker3_ray_discrepancy,
    load_cotracker3_multiview_depth_observations,
    load_cotracker3_multiview_observations,
    pack_multiview_triangulation,
    project_world_points,
    triangulate_multiview_tracks,
)


def test_project_world_points_uses_camera_to_world_pose() -> None:
    intrinsic = np.array(
        [[100.0, 0.0, 20.0], [0.0, 100.0, 10.0], [0.0, 0.0, 1.0]]
    )
    camera_to_world = np.eye(4)
    camera_to_world[0, 3] = 1.0

    pixels, depth = project_world_points(
        np.array([[1.0, 0.0, 5.0], [2.0, 1.0, 5.0]]),
        intrinsic,
        camera_to_world,
    )

    np.testing.assert_allclose(pixels, [[20.0, 10.0], [40.0, 30.0]])
    np.testing.assert_allclose(depth, [5.0, 5.0])


def test_triangulate_multiview_tracks_recovers_point_and_zero_error() -> None:
    intrinsics = np.repeat(np.eye(3)[None], 2, axis=0)
    intrinsics[:, 0, 0] = 100.0
    intrinsics[:, 1, 1] = 100.0
    intrinsics[:, 0, 2] = 50.0
    intrinsics[:, 1, 2] = 50.0
    camera_to_world = np.repeat(np.eye(4)[None], 2, axis=0)
    camera_to_world[1, 0, 3] = 1.0
    point = np.array([[0.25, -0.10, 5.0]])
    tracks = np.empty((2, 1, 1, 2), dtype=float)
    for camera in range(2):
        tracks[camera, 0], _ = project_world_points(
            point,
            intrinsics[camera],
            camera_to_world[camera],
        )

    reconstructed, error, count = triangulate_multiview_tracks(
        tracks,
        np.ones((2, 1, 1), dtype=bool),
        np.ones((2, 1, 1), dtype=float),
        intrinsics,
        camera_to_world,
    )

    np.testing.assert_allclose(reconstructed[0, 0], point[0], atol=1e-7)
    np.testing.assert_allclose(error, 0.0, atol=1e-7)
    np.testing.assert_array_equal(count, [[2]])


def test_triangulate_multiview_tracks_ignores_invalid_nan_view() -> None:
    intrinsics = np.repeat(np.eye(3)[None], 3, axis=0)
    intrinsics[:, 0, 0] = 100.0
    intrinsics[:, 1, 1] = 100.0
    camera_to_world = np.repeat(np.eye(4)[None], 3, axis=0)
    camera_to_world[1, 0, 3] = 1.0
    camera_to_world[2, 1, 3] = 1.0
    point = np.array([[0.25, -0.10, 5.0]])
    tracks = np.full((3, 1, 1, 2), np.nan, dtype=float)
    for camera in range(2):
        tracks[camera, 0], _ = project_world_points(
            point,
            intrinsics[camera],
            camera_to_world[camera],
        )
    valid = np.array([[[True]], [[True]], [[False]]])

    reconstructed, error, count = triangulate_multiview_tracks(
        tracks,
        valid,
        valid.astype(float),
        intrinsics,
        camera_to_world,
    )

    np.testing.assert_allclose(reconstructed[0, 0], point[0], atol=1e-7)
    np.testing.assert_allclose(error, 0.0, atol=1e-7)
    np.testing.assert_array_equal(count, [[2]])


def test_pack_multiview_triangulation_neutralizes_future_rows() -> None:
    points = np.array(
        [
            [[0.0, 1.0, 2.0], [np.nan, np.nan, np.nan]],
            [[0.1, 1.1, 2.1], [3.0, 4.0, 5.0]],
        ]
    )
    reprojection = np.array([[0.5, np.nan], [0.75, 0.25]])
    camera_count = np.array([[3, 1], [2, 3]])

    packed = pack_multiview_triangulation(
        points,
        reprojection,
        camera_count,
        frame_count=4,
    )

    assert packed["multiview_points_world_m"].dtype == np.float32
    np.testing.assert_array_equal(
        packed["multiview_point_valid"],
        [
            [True, False],
            [True, True],
            [False, False],
            [False, False],
        ],
    )
    np.testing.assert_allclose(
        packed["multiview_points_world_m"][:2, 0],
        points[:, 0],
    )
    assert np.all(np.isnan(packed["multiview_points_world_m"][2:]))
    assert np.all(np.isnan(packed["multiview_points_world_m"][0, 1]))


def test_pack_multiview_triangulation_requires_two_views() -> None:
    packed = pack_multiview_triangulation(
        np.array([[[1.0, 2.0, 3.0]]]),
        np.array([[0.1]]),
        np.array([[1]]),
        frame_count=1,
    )

    assert not packed["multiview_point_valid"][0, 0]
    assert np.all(np.isnan(packed["multiview_points_world_m"][0, 0]))


def _write_multiview_archive(
    path: Path,
    *,
    third_view_quality: float = 1.0,
) -> None:
    intrinsics = np.repeat(np.eye(3)[None], 3, axis=0)
    intrinsics[:, 0, 0] = 100.0
    intrinsics[:, 1, 1] = 100.0
    intrinsics[:, 0, 2] = 50.0
    intrinsics[:, 1, 2] = 50.0
    camera_to_world = np.repeat(np.eye(4)[None], 3, axis=0)
    camera_to_world[1, 0, 3] = 1.0
    camera_to_world[2, 1, 3] = 1.0
    points = np.array(
        [
            [[0.25, -0.10, 5.0]],
            [[0.35, -0.10, 5.0]],
        ]
    )
    tracks = np.empty((3, 2, 1, 2), dtype=float)
    for camera in range(3):
        for frame in range(2):
            tracks[camera, frame], _ = project_world_points(
                points[frame],
                intrinsics[camera],
                camera_to_world[camera],
            )
    quality = np.ones((3, 2, 1), dtype=float)
    quality[2] = third_view_quality
    np.savez_compressed(
        path,
        multiview_tracks_xy_prefix=tracks,
        multiview_quality_probability_prefix=quality,
        multiview_view_valid_prefix=np.ones((3, 2, 1), dtype=bool),
        multiview_intrinsics=intrinsics,
        multiview_camera_to_world=camera_to_world,
        forward_backward_error_px=np.zeros((4, 1), dtype=float),
        forward_backward_valid=np.ones((4, 1), dtype=bool),
        boundary_distance=np.ones((4, 1), dtype=float),
        cue_available=np.array([[True], [True], [False], [False]]),
    )


def test_load_cotracker3_multiview_observations_anchors_displacement(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "cues.npz"
    _write_multiview_archive(archive)

    observations = load_cotracker3_multiview_observations(
        archive,
        np.array([[10.0, 20.0, 30.0]]),
        train_end_frame=2,
        minimum_view_quality=0.5,
        maximum_reprojection_error_px=1.0,
        maximum_cycle_error_px=1.0,
        minimum_camera_count=3,
    )

    np.testing.assert_array_equal(observations.valid, [[True], [True]])
    np.testing.assert_array_equal(observations.camera_count, [[3], [3]])
    np.testing.assert_allclose(
        observations.points_world_m[:, 0],
        [[10.0, 20.0, 30.0], [10.1, 20.0, 30.0]],
        atol=1e-7,
    )
    np.testing.assert_allclose(observations.reprojection_error_px, 0.0, atol=1e-7)
    np.testing.assert_allclose(observations.minimum_view_quality, 1.0)


def test_load_cotracker3_multiview_observations_requires_redundancy(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "cues.npz"
    _write_multiview_archive(archive, third_view_quality=0.1)

    observations = load_cotracker3_multiview_observations(
        archive,
        np.array([[0.25, -0.10, 5.0]]),
        train_end_frame=2,
        minimum_view_quality=0.5,
        maximum_reprojection_error_px=1.0,
        maximum_cycle_error_px=1.0,
        minimum_camera_count=3,
    )

    np.testing.assert_array_equal(observations.camera_count, [[2], [2]])
    assert not np.any(observations.valid)
    assert np.all(np.isnan(observations.points_world_m))


def _write_synthetic_depth(
    archive: Path,
    raw_case_dir: Path,
    *,
    corrupt_last_view: bool = False,
) -> None:
    with np.load(archive) as cues:
        tracks = np.asarray(cues["multiview_tracks_xy_prefix"])
    for camera in range(tracks.shape[0]):
        depth_dir = raw_case_dir / "depth" / str(camera)
        depth_dir.mkdir(parents=True)
        for frame in range(tracks.shape[1]):
            depth = np.zeros((100, 100), dtype=np.float32)
            xy = np.rint(tracks[camera, frame, 0]).astype(int)
            value_mm = 5000.0
            if corrupt_last_view and camera == 2 and frame == 1:
                value_mm = 4000.0
            depth[xy[1], xy[0]] = value_mm
            np.save(depth_dir / f"{frame}.npy", depth)


def test_load_cotracker3_multiview_depth_observations_fuses_displacements(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "cues.npz"
    raw_case_dir = tmp_path / "raw"
    _write_multiview_archive(archive)
    _write_synthetic_depth(archive, raw_case_dir)

    observations = load_cotracker3_multiview_depth_observations(
        archive,
        raw_case_dir,
        np.array([[10.0, 20.0, 30.0]]),
        train_end_frame=2,
        minimum_view_quality=0.5,
        maximum_view_disagreement_m=0.01,
        maximum_cycle_error_px=1.0,
        minimum_camera_count=3,
    )

    np.testing.assert_array_equal(observations.valid, [[True], [True]])
    np.testing.assert_allclose(
        observations.points_world_m[:, 0],
        [[10.0, 20.0, 30.0], [10.1, 20.0, 30.0]],
        atol=1e-7,
    )
    np.testing.assert_allclose(observations.view_disagreement_m, 0.0, atol=1e-7)


def test_load_cotracker3_multiview_depth_observations_rejects_disagreement(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "cues.npz"
    raw_case_dir = tmp_path / "raw"
    _write_multiview_archive(archive)
    _write_synthetic_depth(archive, raw_case_dir, corrupt_last_view=True)

    observations = load_cotracker3_multiview_depth_observations(
        archive,
        raw_case_dir,
        np.array([[0.25, -0.10, 5.0]]),
        train_end_frame=2,
        minimum_view_quality=0.5,
        maximum_view_disagreement_m=0.01,
        maximum_cycle_error_px=1.0,
        minimum_camera_count=3,
    )

    assert observations.valid[0, 0]
    assert not observations.valid[1, 0]
    assert np.all(np.isnan(observations.points_world_m[1, 0]))


def test_infer_cotracker3_ray_discrepancy_recovers_image_supported_motion(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "cues.npz"
    _write_multiview_archive(archive)
    baseline = np.repeat(
        np.array([[[0.25, -0.10, 5.0]]]),
        2,
        axis=0,
    )

    posterior = infer_cotracker3_ray_discrepancy(
        archive,
        baseline,
        end_frame=2,
        window_frames=1,
        minimum_view_quality=0.5,
        maximum_cycle_error_px=1.0,
        minimum_camera_count=3,
        pixel_noise_std=0.1,
        prior_std_m=1.0,
    )

    assert posterior.observed[0]
    assert posterior.camera_support[0] == 3
    np.testing.assert_allclose(
        posterior.mean_m[0],
        [0.1, 0.0, 0.0],
        atol=2e-4,
    )
    assert posterior.variance_m2[0] < 1.0


def test_infer_cotracker3_ray_discrepancy_duplicate_views_do_not_add_confidence(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "cues.npz"
    duplicated_archive = tmp_path / "duplicated.npz"
    _write_multiview_archive(archive)
    with np.load(archive) as cues:
        arrays = {name: np.asarray(cues[name]) for name in cues.files}
    for name in (
        "multiview_tracks_xy_prefix",
        "multiview_quality_probability_prefix",
        "multiview_view_valid_prefix",
        "multiview_intrinsics",
        "multiview_camera_to_world",
    ):
        arrays[name] = np.concatenate([arrays[name], arrays[name]], axis=0)
    np.savez_compressed(duplicated_archive, **arrays)
    baseline = np.repeat(
        np.array([[[0.25, -0.10, 5.0]]]),
        2,
        axis=0,
    )
    arguments = {
        "end_frame": 2,
        "window_frames": 1,
        "minimum_view_quality": 0.5,
        "maximum_cycle_error_px": 1.0,
        "pixel_noise_std": 0.1,
        "prior_std_m": 1.0,
    }

    original = infer_cotracker3_ray_discrepancy(
        archive,
        baseline,
        minimum_camera_count=3,
        **arguments,
    )
    duplicated = infer_cotracker3_ray_discrepancy(
        duplicated_archive,
        baseline,
        minimum_camera_count=6,
        **arguments,
    )

    np.testing.assert_allclose(duplicated.mean_m, original.mean_m, atol=1e-12)
    np.testing.assert_allclose(
        duplicated.variance_m2,
        original.variance_m2,
        rtol=1e-12,
        atol=1e-12,
    )
