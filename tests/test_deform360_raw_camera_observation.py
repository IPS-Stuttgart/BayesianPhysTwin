from __future__ import annotations

import inspect
import itertools
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
    select_nested_frame_zero_observation_plans,
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


def test_nested_frame_zero_plans_are_deterministic_and_strictly_nested() -> None:
    point_count = 24
    angle = np.linspace(0.0, 2.0 * np.pi, point_count, endpoint=False)
    points = np.stack(
        (0.2 * np.cos(angle), 0.15 * np.sin(angle), np.full(point_count, 2.0)),
        axis=1,
    )
    cameras = tuple(f"camera-{index:02d}" for index in range(10))
    extrinsics = {
        camera: _camera_to_world(
            0.4 * np.cos(2.0 * np.pi * index / len(cameras)),
            0.4 * np.sin(2.0 * np.pi * index / len(cameras)),
        )
        for index, camera in enumerate(cameras)
    }
    support = np.ones((point_count, len(cameras)), dtype=bool)
    # Break some support ties without making any center ineligible.
    support[::3, 1] = False
    support[1::4, 7] = False
    projected = {
        camera: np.column_stack(
            (
                np.arange(point_count, dtype=float) + index,
                np.full(point_count, index, dtype=float),
            )
        )
        for index, camera in enumerate(cameras)
    }
    config = RawCameraObservationConfig(selected_camera_count=8)

    first = select_nested_frame_zero_observation_plans(
        points,
        cameras,
        support,
        projected,
        extrinsics,
        config=config,
    )
    second = select_nested_frame_zero_observation_plans(
        points,
        cameras,
        support,
        projected,
        extrinsics,
        config=config,
    )

    plan4 = first["prefix_plans"][4]
    plan8 = first["prefix_plans"][8]
    assert first["camera_activation_order"] == (
        "camera-00",
        "camera-04",
        "camera-05",
        "camera-09",
        "camera-02",
        "camera-03",
        "camera-06",
        "camera-08",
    )
    assert plan4["selection_score"][:3] == (16, 16, 64)
    assert plan4["selection_score"][3] == pytest.approx(22.53679749174791)
    assert plan8["selection_score"][:3] == (16, 16, 128)
    assert plan8["selection_score"][3] == pytest.approx(22.53679749174791)
    legacy4 = select_frame_zero_observation_plan(
        points,
        cameras,
        support,
        projected,
        extrinsics,
        config=RawCameraObservationConfig(selected_camera_count=4),
    )
    assert plan8["selected_cameras"][:4] == plan4["selected_cameras"]
    assert plan4["selected_cameras"] == legacy4["selected_cameras"]
    np.testing.assert_array_equal(
        plan4["selected_camera_indices"],
        legacy4["selected_camera_indices"],
    )
    assert plan4["selection_score"] == legacy4["selection_score"]
    np.testing.assert_array_equal(plan4["center_ids"], plan8["center_ids"])
    assert first["camera_activation_order"][:4] == plan4["selected_cameras"]
    assert first["camera_activation_order"][:8] == plan8["selected_cameras"]
    assert first["camera_activation_order"] == second["camera_activation_order"]
    assert first["activation_stages"] == second["activation_stages"]
    np.testing.assert_array_equal(
        first["camera_activation_indices"],
        second["camera_activation_indices"],
    )
    selected_four = tuple(plan4["selected_camera_indices"].tolist())
    remaining = tuple(
        index for index in range(len(cameras)) if index not in set(selected_four)
    )
    origins = np.stack([extrinsics[camera][:3, 3] for camera in cameras])
    additions = max(
        itertools.combinations(remaining, 4),
        key=lambda candidate: raw_camera._camera_subset_score(
            points,
            plan4["center_ids"],
            support,
            selected_four + candidate,
            origins,
            minimum_initial_view_count=config.minimum_initial_view_count,
        ),
    )
    assert tuple(plan8["selected_camera_indices"]) == selected_four + additions


def test_nested_frame_zero_planner_is_deterministic_and_mutates_no_input() -> None:
    point_count = 20
    points = np.column_stack(
        (
            np.linspace(-0.2, 0.2, point_count),
            np.linspace(-0.1, 0.1, point_count),
            np.full(point_count, 2.0),
        )
    )
    cameras = tuple(f"camera-{index:02d}" for index in range(8))
    extrinsics = {
        camera: _camera_to_world(
            0.4 * np.cos(2.0 * np.pi * index / len(cameras)),
            0.4 * np.sin(2.0 * np.pi * index / len(cameras)),
        )
        for index, camera in enumerate(cameras)
    }
    support = np.ones((point_count, len(cameras)), dtype=bool)
    projected = {
        camera: np.full((point_count, 2), index, dtype=float)
        for index, camera in enumerate(cameras)
    }
    points_before = points.copy()
    support_before = support.copy()
    projected_before = {camera: pixels.copy() for camera, pixels in projected.items()}
    extrinsics_before = {
        camera: transform.copy() for camera, transform in extrinsics.items()
    }
    config = RawCameraObservationConfig(selected_camera_count=8)

    forward = select_nested_frame_zero_observation_plans(
        points,
        cameras,
        support,
        projected,
        extrinsics,
        config=config,
    )
    repeated = select_nested_frame_zero_observation_plans(
        points,
        cameras,
        support,
        projected,
        extrinsics,
        config=config,
    )

    assert forward["camera_activation_order"] == repeated["camera_activation_order"]
    assert forward["activation_stages"] == repeated["activation_stages"]
    assert not forward["candidate_ids"].flags.writeable
    assert not forward["center_ids"].flags.writeable
    assert (
        forward["prefix_plans"][4]["candidate_ids"]
        is forward["prefix_plans"][8]["candidate_ids"]
    )
    assert (
        forward["prefix_plans"][4]["center_ids"]
        is forward["prefix_plans"][8]["center_ids"]
    )
    with pytest.raises(ValueError, match="read-only"):
        forward["prefix_plans"][4]["candidate_ids"][0] = -1
    with pytest.raises(ValueError, match="read-only"):
        forward["prefix_plans"][4]["center_ids"][0] = -1
    np.testing.assert_array_equal(points, points_before)
    np.testing.assert_array_equal(support, support_before)
    for camera in cameras:
        np.testing.assert_array_equal(projected[camera], projected_before[camera])
        np.testing.assert_array_equal(extrinsics[camera], extrinsics_before[camera])


def test_nested_eight_camera_prefix_preserves_legacy_plan_contract() -> None:
    point_count = 24
    angle = np.linspace(0.0, 2.0 * np.pi, point_count, endpoint=False)
    points = np.stack(
        (0.2 * np.cos(angle), 0.15 * np.sin(angle), np.full(point_count, 2.0)),
        axis=1,
    )
    cameras = tuple(f"camera-{index:02d}" for index in range(8))
    extrinsics = {
        camera: _camera_to_world(
            0.4 * np.cos(2.0 * np.pi * index / len(cameras)),
            0.4 * np.sin(2.0 * np.pi * index / len(cameras)),
        )
        for index, camera in enumerate(cameras)
    }
    support = np.ones((point_count, len(cameras)), dtype=bool)
    projected = {
        camera: np.full((point_count, 2), index, dtype=float)
        for index, camera in enumerate(cameras)
    }
    config = RawCameraObservationConfig(selected_camera_count=8)

    legacy = select_frame_zero_observation_plan(
        points,
        cameras,
        support,
        projected,
        extrinsics,
        config=config,
    )
    nested = select_nested_frame_zero_observation_plans(
        points,
        cameras,
        support,
        projected,
        extrinsics,
        config=config,
    )["prefix_plans"][8]

    assert set(nested) == set(legacy)
    assert set(nested["selected_cameras"]) == set(legacy["selected_cameras"])
    np.testing.assert_array_equal(nested["candidate_ids"], legacy["candidate_ids"])
    np.testing.assert_array_equal(nested["center_ids"], legacy["center_ids"])
    assert nested["selection_score"] == legacy["selection_score"]
    assert nested["camera_names"] == legacy["camera_names"]
    for camera in cameras:
        np.testing.assert_array_equal(
            nested["query_ids"][camera],
            legacy["query_ids"][camera],
        )
        np.testing.assert_array_equal(
            nested["query_pixels"][camera],
            legacy["query_pixels"][camera],
        )


def test_nested_frame_zero_planner_rejects_noninteger_prefix_counts() -> None:
    points = np.column_stack(
        (
            np.linspace(-0.2, 0.2, 16),
            np.linspace(-0.1, 0.1, 16),
            np.full(16, 2.0),
        )
    )
    cameras = tuple(f"camera-{index:02d}" for index in range(8))
    support = np.ones((16, 8), dtype=bool)
    projected = {camera: np.zeros((16, 2)) for camera in cameras}
    extrinsics = {
        camera: _camera_to_world(
            0.4 * np.cos(2.0 * np.pi * index / len(cameras)),
            0.4 * np.sin(2.0 * np.pi * index / len(cameras)),
        )
        for index, camera in enumerate(cameras)
    }

    with pytest.raises(ValueError, match="exact integers"):
        select_nested_frame_zero_observation_plans(
            points,
            cameras,
            support,
            projected,
            extrinsics,
            config=RawCameraObservationConfig(selected_camera_count=8),
            prefix_camera_counts=(4.9, 8),
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
