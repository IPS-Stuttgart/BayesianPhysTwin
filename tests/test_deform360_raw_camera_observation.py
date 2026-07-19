from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

import bayesian_phystwin.deform360_raw_camera_observation as raw_camera

from bayesian_phystwin.deform360_raw_camera_observation import (
    ALLTRACKER_RUNTIME_SOURCE_SHA256,
    AllTrackerPrefixRuntime,
    RawCameraObservationConfig,
    _causal_selected_camera_inputs,
    build_raw_camera_measurement_case,
    project_world_points,
    select_frame_zero_observation_plan,
    triangulate_observation_ransac,
)


def _camera_to_world(x: float, y: float = 0.0) -> np.ndarray:
    result = np.eye(4)
    result[:3, 3] = (x, y, 0.0)
    return result


def test_projection_uses_camera_to_world_calibration() -> None:
    intrinsics = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    points = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 2.0]])

    pixels, depth = project_world_points(
        points,
        intrinsics,
        _camera_to_world(1.0),
    )

    np.testing.assert_allclose(depth, [5.0, 2.0])
    np.testing.assert_allclose(pixels, [[50.0, 40.0], [0.0, 90.0]])


def test_frame_zero_plan_is_deterministic_and_multiview_supported() -> None:
    point_count = 24
    angle = np.linspace(0.0, 2.0 * np.pi, point_count, endpoint=False)
    points = np.stack(
        (0.2 * np.cos(angle), 0.15 * np.sin(angle), np.full(point_count, 2.0)),
        axis=1,
    )
    cameras = tuple(f"camera-{index}" for index in range(8))
    extrinsics = {
        camera: _camera_to_world(
            0.4 * np.cos(2.0 * np.pi * index / len(cameras)),
            0.4 * np.sin(2.0 * np.pi * index / len(cameras)),
        )
        for index, camera in enumerate(cameras)
    }
    support = np.ones((point_count, len(cameras)), dtype=bool)
    projected = {camera: np.zeros((point_count, 2)) for camera in cameras}
    config = RawCameraObservationConfig(selected_camera_count=4)

    first = select_frame_zero_observation_plan(
        points,
        cameras,
        support,
        projected,
        extrinsics,
        config=config,
    )
    second = select_frame_zero_observation_plan(
        points,
        cameras,
        support,
        projected,
        extrinsics,
        config=config,
    )

    np.testing.assert_array_equal(first["center_ids"], second["center_ids"])
    assert first["selected_cameras"] == second["selected_cameras"]
    assert len(first["center_ids"]) == 16
    assert len(first["selected_cameras"]) == 4
    assert all(
        np.sum(support[point_id]) >= config.minimum_initial_view_count
        for point_id in first["center_ids"]
    )


def test_dlt_ransac_rejects_one_bad_view() -> None:
    intrinsics = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    extrinsics = {
        "left": _camera_to_world(-0.5),
        "right": _camera_to_world(0.5),
        "bad": _camera_to_world(0.0, 0.5),
    }
    point = np.array([0.05, -0.02, 3.0])
    projection_matrices = {
        camera: intrinsics @ np.linalg.inv(c2w)[:3]
        for camera, c2w in extrinsics.items()
    }
    observations = {
        camera: project_world_points(point[None], intrinsics, c2w)[0][0]
        for camera, c2w in extrinsics.items()
    }
    observations["bad"] = observations["bad"] + np.array([120.0, -80.0])

    estimate, diagnostic = triangulate_observation_ransac(
        observations,
        projection_matrices,
        {camera: c2w[:3, 3] for camera, c2w in extrinsics.items()},
        point,
        config=RawCameraObservationConfig(),
    )

    assert estimate is not None
    np.testing.assert_allclose(estimate, point, atol=1e-6)
    assert diagnostic["accepted"] is True
    assert diagnostic["inlier_view_count"] == 2
    assert "bad" not in diagnostic["inlier_cameras"]


def test_triangulation_rejects_nearly_antiparallel_rays() -> None:
    intrinsics = np.eye(3)
    extrinsics = {
        "left": _camera_to_world(-1.0),
        "right": _camera_to_world(1.0),
    }
    point = np.array([0.0, 0.0, 0.01])
    projection_matrices = {
        camera: intrinsics @ np.linalg.inv(c2w)[:3]
        for camera, c2w in extrinsics.items()
    }
    observations = {
        camera: project_world_points(point[None], intrinsics, c2w)[0][0]
        for camera, c2w in extrinsics.items()
    }

    estimate, diagnostic = triangulate_observation_ransac(
        observations,
        projection_matrices,
        {camera: c2w[:3, 3] for camera, c2w in extrinsics.items()},
        point,
        config=RawCameraObservationConfig(minimum_ray_angle_degrees=2.0),
    )

    assert estimate is None
    assert diagnostic["decision"] == "ray_angle_failure"
    assert diagnostic["maximum_ray_angle_degrees"] < 2.0


def test_measurement_builder_has_no_target_or_outcome_argument() -> None:
    parameters = inspect.signature(build_raw_camera_measurement_case).parameters

    assert "target" not in parameters
    assert "outcome" not in parameters
    assert set(parameters) == {
        "panel_case_dir",
        "processed_episode_dir",
        "output_dir",
        "runtime",
        "config",
    }


def test_camera_manifest_hashes_only_causal_materialized_slices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        raw_camera,
        "_read_h5_frame_zero",
        lambda _path: np.arange(12, dtype=np.uint16).reshape(3, 4),
    )
    updates = [
        {
            "frame": frame,
            "tracker": [
                {
                    "camera": "camera-0",
                    "maximum_video_frame_read": frame,
                    "decoded_rgb_prefix_sha256": str(frame) * 64,
                }
            ],
        }
        for frame in (19, 38, 57)
    ]

    result = _causal_selected_camera_inputs(
        tmp_path,
        ["camera-0"],
        updates,
    )["camera-0"]

    assert result["video"]["decoded_prefix_sha256_by_update"] == {
        "19": "19" * 64,
        "38": "38" * 64,
        "57": "57" * 64,
    }
    assert result["video"]["whole_file_hashed_or_read"] is False
    assert result["frame_zero_mask"]["only_index_read"] == 0
    assert result["frame_zero_depth"]["only_index_read"] == 0
    assert "sha256" not in result["video"]
    assert "sha256" not in result["frame_zero_mask"]
    assert "sha256" not in result["frame_zero_depth"]


def test_alltracker_runtime_fails_closed_on_source_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "alltracker"
    (source / "nets").mkdir(parents=True)
    (source / "nets" / "alltracker.py").write_text("# fixture\n", encoding="utf-8")
    checkpoint = tmp_path / "alltracker.pth"
    checkpoint.write_bytes(b"fixture")
    monkeypatch.setattr(raw_camera, "_source_tree_sha256", lambda _root: "0" * 64)

    with pytest.raises(ValueError, match="source differs"):
        AllTrackerPrefixRuntime(
            source,
            checkpoint,
            device="cpu",
            config=RawCameraObservationConfig(),
        )

    monkeypatch.setattr(
        raw_camera,
        "_source_tree_sha256",
        lambda _root: ALLTRACKER_RUNTIME_SOURCE_SHA256,
    )
    monkeypatch.setattr(raw_camera, "_sha256", lambda _path: "0" * 64)
    with pytest.raises(ValueError, match="checkpoint differs"):
        AllTrackerPrefixRuntime(
            source,
            checkpoint,
            device="cpu",
            config=RawCameraObservationConfig(),
        )
