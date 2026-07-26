from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.cotracker3_prefix import CoTracker3PrefixRuntime
from bayesian_phystwin.deform360_cotracker_bias_source import (
    PenguinCoTrackerSourceConfig,
    build_causal_cotracker_measurement,
    conservative_triangulation_variance_m2,
    penguin_episode_directory,
)


def test_correlated_view_duplication_cannot_reduce_metric_variance() -> None:
    fused = np.asarray([0.0, 0.0, 1.0])
    base = conservative_triangulation_variance_m2(
        inlier_view_count=2,
        selected_camera_count=5,
        leave_one_view_points_m=np.empty((0, 3)),
        fused_point_m=fused,
        variance_floor_m2=0.005**2,
        two_view_variance_multiplier=4.0,
    )
    duplicated = conservative_triangulation_variance_m2(
        inlier_view_count=4,
        selected_camera_count=10,
        leave_one_view_points_m=np.repeat(fused[None], 4, axis=0),
        fused_point_m=fused,
        variance_floor_m2=0.005**2,
        two_view_variance_multiplier=4.0,
    )

    assert duplicated >= base
    assert base >= 4.0 * 0.005**2


def test_leave_one_view_disagreement_only_inflates_variance() -> None:
    fused = np.asarray([0.0, 0.0, 1.0])
    result = conservative_triangulation_variance_m2(
        inlier_view_count=4,
        selected_camera_count=5,
        leave_one_view_points_m=np.asarray(
            [[0.0, 0.0, 1.0], [0.02, 0.0, 1.0]]
        ),
        fused_point_m=fused,
        variance_floor_m2=0.005**2,
        two_view_variance_multiplier=4.0,
    )

    assert result == pytest.approx(0.02**2)


def test_penguin_source_episode_mapping_is_explicit(tmp_path: Path) -> None:
    assert penguin_episode_directory(tmp_path, 1) == (
        tmp_path.resolve() / "171-penguin" / "episode_0000"
    )
    assert penguin_episode_directory(tmp_path, 9) == (
        tmp_path.resolve() / "171-penguin-ep0009" / "episode_0000"
    )
    with pytest.raises(ValueError, match="outside the source panel"):
        penguin_episode_directory(tmp_path, 2)


def test_prefix_resize_preserves_shape_when_already_small() -> None:
    rgb = np.zeros((3, 64, 96, 3), dtype=np.uint8)

    resized, original_shape = CoTracker3PrefixRuntime._resize_prefix(rgb, 128)

    assert resized is rgb
    assert original_shape == (64, 96)


class _StaticPrefixTracker:
    def track_prefix(
        self,
        video_path: str | Path,
        query_pixels_xy: np.ndarray,
        update_frame: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        assert Path(video_path).name == "undistorted.mp4"
        return (
            np.asarray(query_pixels_xy, dtype=np.float32),
            np.ones(len(query_pixels_xy), dtype=bool),
            {
                "maximum_video_frame_read": update_frame,
                "decoded_rgb_prefix_sha256": f"frame-{update_frame}",
            },
        )


def _write_camera_assets(
    root: Path,
    camera: str,
    point_depth_mm: int,
) -> None:
    h5py = pytest.importorskip("h5py")
    camera_dir = root / camera
    camera_dir.mkdir(parents=True)
    (camera_dir / "undistorted.mp4").write_bytes(b"fake")
    with h5py.File(camera_dir / "mask_refined.h5", "w") as stream:
        stream.create_dataset("data", data=np.ones((1, 64, 64), dtype=np.uint8))
    with h5py.File(camera_dir / "rendered_depth.h5", "w") as stream:
        stream.create_dataset(
            "data",
            data=np.full((1, 64, 64), point_depth_mm, dtype=np.uint16),
        )


def test_causal_measurement_uses_only_prefix_tracker_and_frame_zero_assets(
    tmp_path: Path,
) -> None:
    camera_names = [f"camera-{index}" for index in range(5)]
    intrinsics = {}
    extrinsics = {}
    for index, camera in enumerate(camera_names):
        _write_camera_assets(tmp_path, camera, 1000)
        intrinsics[camera] = np.asarray(
            [[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]]
        )
        transform = np.eye(4)
        transform[0, 3] = 0.04 * (index - 2)
        extrinsics[camera] = transform
    np.save(tmp_path / "undistorted_intrinsics.npy", intrinsics)
    np.save(tmp_path / "extrinsics.npy", extrinsics)
    x, y = np.meshgrid(np.linspace(-0.08, 0.08, 5), np.linspace(-0.08, 0.08, 4))
    frame_zero = np.column_stack((x.ravel(), y.ravel(), np.ones(x.size)))
    config = PenguinCoTrackerSourceConfig(
        update_frames=(1, 2, 3),
        center_count=8,
        selected_camera_count=5,
    )

    arrays, report = build_causal_cotracker_measurement(
        tmp_path,
        frame_zero,
        (5, len(frame_zero), 3),
        _StaticPrefixTracker(),
        config=config,
    )

    assert arrays["measurement_m"].shape == (5, len(frame_zero), 3)
    assert arrays["center_ids"].shape == (8,)
    assert np.all(arrays["prior_reliability"] > 0.0)
    assert np.all(
        arrays["observation_variance_m2"]
        >= config.observation_variance_floor_m2
    )
    assert report["information_boundary"]["pcd_clean_read"] is False
    assert report["information_boundary"]["future_rgb_read"] is False
    assert [
        update["maximum_video_frame_read"] for update in report["updates"]
    ] == [1, 2, 3]
