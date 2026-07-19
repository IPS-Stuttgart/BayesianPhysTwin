from __future__ import annotations

import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from bayesian_phystwin.deform360_held_physical_prior import (
    FRAME_COUNT,
    OFFICIAL_PHYSTWIN_REVISION,
    OFFICIAL_REAL_CONFIG_SHA256,
    WARP_DYNAMICS,
    build_physical_prediction_archive,
    build_prediction_only_artifacts,
    select_action_window,
    sha256_file,
    validate_physical_prediction_manifest,
)
from bayesian_phystwin.deform360_held_protocol import (
    create_held_protocol_lock,
    held_artifact_sha256,
)


CASE_NAME = "083-blanket-cloth-ep0000"


def _bound_file(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _make_robot(path: Path, frame_count: int = FRAME_COUNT) -> None:
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], frame_count, axis=0)
    poses[:, 0, 3] = np.linspace(0.0, 0.02, frame_count)
    actions = np.zeros((frame_count, 5, 3), dtype=np.float64)
    actions[:, 0] = poses[:, :3, 3]
    actions[:, 1:4] = poses[:, :3, :3]
    openings = np.linspace(0.10, 0.05, frame_count)
    actions[:, 4, 0] = openings
    np.savez_compressed(
        path,
        format_version=np.asarray(1, dtype=np.uint16),
        actions=actions,
        T_worlds=poses,
        openings=openings,
        bimanual=np.asarray(False, dtype=np.bool_),
    )


def _make_frame_zero_bundle(
    path: Path, *, encoded_frames: tuple[int, ...] = (0,)
) -> None:
    camera_count = 2
    point_count = 128
    points = np.column_stack(
        (
            np.linspace(0.0, 0.1, point_count),
            np.zeros(point_count),
            np.ones(point_count) * 0.2,
        )
    ).astype(np.float32)
    colors = np.tile(np.array([[0.2, 0.4, 0.6]], dtype=np.float32), (point_count, 1))
    rgb = np.zeros((camera_count, 2, 3, 3), dtype=np.uint8)
    mask = np.ones(rgb.shape[:3], dtype=bool)
    depth = np.ones(rgb.shape[:3], dtype=np.float32)
    intrinsics = np.repeat(np.eye(3, dtype=np.float64)[None], camera_count, axis=0)
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], camera_count, axis=0)
    projections = np.repeat(
        np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)[None],
        camera_count,
        axis=0,
    )
    np.savez_compressed(
        path,
        frame_indices=np.asarray(encoded_frames, dtype=np.int64),
        camera_names=np.asarray(["cam0", "cam1"]),
        rgb_frame0=rgb,
        mask_frame0=mask,
        depth_frame0_m=depth,
        depth_valid_frame0=mask,
        intrinsics=intrinsics,
        camera_to_world=poses,
        projection_world_to_pixel=projections,
        object_points_world_m=points,
        object_colors_rgb=colors,
        object_color_support_count=np.ones(point_count, dtype=np.uint8) * 2,
        visual_hull_points_world_m=points,
    )


def _make_locked_frame_zero(
    tmp_path: Path, *, encoded_frames: tuple[int, ...] = (0,)
) -> tuple[Path, Path]:
    lock_path = tmp_path / "held_lock.json"
    create_held_protocol_lock(lock_path, immutable_bindings={"test": "0" * 64})
    bundle_path = tmp_path / "frame_zero.npz"
    _make_frame_zero_bundle(bundle_path, encoded_frames=encoded_frames)
    robot_path = tmp_path / "robot.npz"
    _make_robot(robot_path, frame_count=100)
    selected_robot_path = tmp_path / "known_action_76.npz"
    _make_robot(selected_robot_path)
    robot_metadata_path = tmp_path / "robot.meta.json"
    robot_metadata_path.write_text("{}\n", encoding="utf-8")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "Deform360HeldFrameZeroBundle",
        "protocol_id": "deform360-held-online-belief-v1",
        "case_name": CASE_NAME,
        "object_id": "083-blanket-cloth",
        "episode_id": 0,
        "role": "calibration",
        "frame_indices": [0],
        "lock_sha256": sha256_file(lock_path),
        "bundle": _bound_file(bundle_path),
        "action_inputs": {
            "robot_trajectory": _bound_file(robot_path),
            "robot_metadata": _bound_file(robot_metadata_path),
        },
        "action_alignment": {
            "selected_raw_frame_range_half_open": [8, 89],
            "prediction_raw_frame_range_half_open": [8, 84],
            "selected_action_bundle": _bound_file(selected_robot_path),
        },
        "information_boundary": {
            "maximum_object_rgb_frame_read": 0,
            "object_observation_frames_used": [0],
            "known_future_robot_action_read": True,
            "future_object_rgb_read": False,
            "future_object_geometry_read": False,
            "future_depth_or_mask_read": False,
            "future_tactile_read": False,
            "outcome_created": False,
            "outcome_read": False,
            "whole_future_container_hashed_or_read": False,
        },
    }
    manifest["artifact_sha256"] = held_artifact_sha256(manifest)
    manifest_path = tmp_path / "frame_zero_manifest.json"
    _write_json(manifest_path, manifest)
    return lock_path, manifest_path


def test_prediction_input_repeats_only_frame_zero(tmp_path: Path) -> None:
    lock_path, manifest_path = _make_locked_frame_zero(tmp_path)
    data_path = tmp_path / "prediction.pkl"
    summary_path = tmp_path / "prediction.json"
    summary = build_prediction_only_artifacts(
        manifest_path,
        lock_path,
        data_path,
        summary_path,
        case_name=CASE_NAME,
        role="calibration",
    )
    with data_path.open("rb") as stream:
        data = pickle.load(stream)
    assert data["object_points"].shape == (FRAME_COUNT, 128, 3)
    assert np.array_equal(
        data["object_points"],
        np.repeat(data["object_points"][:1], FRAME_COUNT, axis=0),
    )
    assert data["controller_points"].shape == (FRAME_COUNT, 768, 3)
    assert summary["information_boundary"]["future_object_geometry_read"] is False
    assert summary["point_count"] == 128


def test_prediction_input_rejects_multiframe_bundle(tmp_path: Path) -> None:
    lock_path, manifest_path = _make_locked_frame_zero(tmp_path, encoded_frames=(0, 1))
    with pytest.raises(ValueError, match="nonzero frame"):
        build_prediction_only_artifacts(
            manifest_path,
            lock_path,
            tmp_path / "prediction.pkl",
            tmp_path / "prediction.json",
            case_name=CASE_NAME,
            role="calibration",
        )


def test_action_window_uses_earliest_closed_path_maximum() -> None:
    frame_count = 100
    actions = np.zeros((frame_count, 5, 3), dtype=np.float64)
    openings = np.ones(frame_count, dtype=np.float64) * 0.05
    # Identical action path in two overlapping candidates; the first must win.
    actions[:, 0, 0] = np.arange(frame_count, dtype=np.float64)
    result = select_action_window(actions, openings)
    assert result["prediction_frame_range_half_open"] == [8, 84]
    assert result["staging_frame_range_half_open"] == [8, 89]


def _warp_result(
    path: Path,
    *,
    simulator_path: Path,
    graph_path: Path,
    semantic_sha256: str,
    scale: float,
    trajectory: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_path = path.with_name("official_phystwin_trajectory.npz")
    np.savez_compressed(trajectory_path, vertices=trajectory)
    value = {
        "passed": True,
        "official_phystwin_revision": OFFICIAL_PHYSTWIN_REVISION,
        "data_sha256": sha256_file(simulator_path),
        "config_sha256": OFFICIAL_REAL_CONFIG_SHA256,
        "config_overrides": {
            "controller_max_neighbours": 1,
            "controller_radius": 0.03,
            "dashpot_damping": 100.0,
            "drag_damping": 10.0,
            "init_spring_Y": 10_000.0,
        },
        "support_dynamics": {"mode": "official-ground"},
        "canonical_reusable_graph": {
            "file_sha256": sha256_file(graph_path),
            "reusable_graph_sha256": semantic_sha256,
            "controller_patch_size_per_anchor": 16,
        },
        "realized_actuation": {"controller_displacement_scale": scale},
        "trajectory_sha256": sha256_file(trajectory_path),
    }
    _write_json(path, value)


def test_physical_archive_matches_frozen_driven_minus_zero_formula(
    tmp_path: Path,
) -> None:
    lock_path, frame_zero_manifest = _make_locked_frame_zero(tmp_path)
    prediction_path = tmp_path / "prediction.pkl"
    prediction_summary = tmp_path / "prediction_summary.json"
    build_prediction_only_artifacts(
        frame_zero_manifest,
        lock_path,
        prediction_path,
        prediction_summary,
        case_name=CASE_NAME,
        role="calibration",
    )
    simulator_path = tmp_path / "simulator.pkl"
    simulator_path.write_bytes(b"simulator")
    semantic_sha256 = "a" * 64
    graph_path = tmp_path / "graph.npz"
    vertex_count = 128
    vertices = np.column_stack(
        (
            np.arange(vertex_count) * 0.001,
            np.zeros(vertex_count),
            np.zeros(vertex_count),
        )
    ).astype(np.float32)
    springs = np.column_stack(
        (np.arange(vertex_count - 1), np.arange(1, vertex_count))
    ).astype(np.int32)
    np.savez_compressed(
        graph_path,
        vertices=vertices,
        springs=springs,
        rest_lengths=np.ones(vertex_count - 1, dtype=np.float32) * 0.001,
        contact_anchor_indices=np.asarray([0], dtype=np.int64),
        observed_node_count=np.asarray(vertex_count),
        reusable_graph_sha256=np.asarray(semantic_sha256),
    )
    state_path = tmp_path / "state.npz"
    np.savez_compressed(
        state_path,
        readout_weights=np.eye(vertex_count, dtype=np.float32),
        canonical_graph_sha256=np.asarray(semantic_sha256),
    )
    twin_path = tmp_path / "twin.json"
    twin = {
        "passed": True,
        "object_id": "083-blanket-cloth",
        "episode_id": 0,
        "input_sha256": {"episode_final_data": sha256_file(prediction_path)},
        "output_sha256": {"simulator_final_data": sha256_file(simulator_path)},
        "information_boundary": {
            "target_access": False,
            "post_initial_object_observation_used": False,
        },
    }
    _write_json(twin_path, twin)
    zero = np.repeat(vertices[None], FRAME_COUNT, axis=0)
    driven = zero.copy()
    driven[:, :, 1] += np.arange(FRAME_COUNT, dtype=np.float32)[:, None] * 0.001
    driven_result = tmp_path / "driven" / "official_phystwin_smoke.json"
    zero_result = tmp_path / "zero" / "official_phystwin_smoke.json"
    _warp_result(
        driven_result,
        simulator_path=simulator_path,
        graph_path=graph_path,
        semantic_sha256=semantic_sha256,
        scale=1.0,
        trajectory=driven,
    )
    _warp_result(
        zero_result,
        simulator_path=simulator_path,
        graph_path=graph_path,
        semantic_sha256=semantic_sha256,
        scale=0.0,
        trajectory=zero,
    )
    archive_path = tmp_path / "physical_prediction.npz"
    manifest_path = tmp_path / "physical_prediction.json"
    result = build_physical_prediction_archive(
        prediction_path,
        simulator_path,
        graph_path,
        state_path,
        twin_path,
        driven_result,
        zero_result,
        archive_path,
        manifest_path,
        frame_zero_manifest_path=frame_zero_manifest,
        lock_path=lock_path,
        case_name=CASE_NAME,
        role="calibration",
        runtime_provenance={},
        stage_runtime_seconds={"test": 0.0},
    )
    assert result["frozen_predictor"]["warp_dynamics"] == WARP_DYNAMICS
    with np.load(archive_path, allow_pickle=False) as stored:
        assert stored["prediction_m"].shape == (FRAME_COUNT, vertex_count, 3)
        assert np.allclose(stored["prediction_m"][0], stored["frame_zero_points_m"])
        assert np.all(stored["action_support"] <= 1.0)
        assert np.all(stored["action_support"] > 0.0)

    tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered["information_boundary"]["future_object_geometry_read"] = True
    tampered["artifact_sha256"] = held_artifact_sha256(tampered)
    _write_json(manifest_path, tampered)
    with pytest.raises(ValueError, match="information boundary"):
        validate_physical_prediction_manifest(manifest_path, verify_archive=True)
