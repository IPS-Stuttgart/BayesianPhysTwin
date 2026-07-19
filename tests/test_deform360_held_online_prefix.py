from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import sys

import numpy as np
import pytest

import bayesian_phystwin.deform360_held_online_prefix as held_prefix
from bayesian_phystwin.deform360_held_online_prefix import (
    _decode_action_aligned_prefix,
    predict_support_gated_selected_backbone_rbf,
    reject_forbidden_prefix_input,
)
from bayesian_phystwin.deform360_held_protocol import (
    ONLINE_ARTIFACT_ROLES,
    create_held_protocol_lock,
    create_physical_prior_seal,
    create_prefix_stage_authorization,
    held_artifact_sha256,
)
from bayesian_phystwin.deform360_raw_camera_gated_evaluation import (
    RBF_ARM_PREFIX,
    SELECTED_BACKBONE_ARM,
    evaluate_covariance_gated_arrays,
)
from bayesian_phystwin.deform360_raw_camera_observation import (
    ALLTRACKER_CHECKPOINT_SHA256,
    ALLTRACKER_RUNTIME_SOURCE_SHA256,
    RawCameraObservationConfig,
)


CASE_NAME = "083-blanket-cloth-ep0000"


def _record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _synthetic_inputs() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    frame_count = 76
    point_count = 6
    centers = np.asarray([0, 1, 2, 3], dtype=np.int64)
    frame_zero = np.column_stack(
        (
            np.linspace(0.0, 0.10, point_count),
            np.linspace(-0.02, 0.03, point_count),
            np.linspace(0.20, 0.25, point_count),
        )
    ).astype(np.float32)
    physical = np.repeat(frame_zero[None], frame_count, axis=0)
    persistence = np.repeat(frame_zero[None], frame_count, axis=0)
    physical[:, :, 0] += np.linspace(0.0, 0.075, frame_count)[:, None]
    persistence[0] = physical[0]
    measurement = np.full_like(physical, np.nan)
    measurement_validity = np.zeros((frame_count, point_count), dtype=bool)
    # Select physical, then persistence, then physical, while applying a
    # nonzero RBF correction on every sufficiently supported update.
    expected = (physical, persistence, physical)
    for update, backbone in zip((19, 38, 57), expected):
        measurement[update, centers] = backbone[update, centers]
        measurement[update, centers, 1] += 0.004
        measurement_validity[update, centers] = True
    return physical, persistence, measurement, measurement_validity, centers


def test_target_free_predictor_is_array_equivalent_to_open_ungated_arm() -> None:
    physical, persistence, measurement, measurement_validity, centers = (
        _synthetic_inputs()
    )
    prediction, selected_raw, diagnostic = predict_support_gated_selected_backbone_rbf(
        physical,
        persistence,
        measurement,
        measurement_validity,
        center_ids=centers,
    )
    covariance = np.full(physical.shape[:2] + (3, 3), np.nan)
    covariance_validity = np.zeros(physical.shape[:2], dtype=bool)
    report, trajectories = evaluate_covariance_gated_arrays(
        physical,
        persistence,
        physical.copy(),
        np.ones(physical.shape[:2], dtype=bool),
        np.ones(physical.shape[:2], dtype=bool),
        measurement,
        measurement_validity,
        covariance,
        covariance_validity,
        center_ids=centers,
        scored_frames=(20, 39, 58),
        gate_thresholds={"ungated": -np.inf},
    )

    np.testing.assert_array_equal(prediction, trajectories[f"{RBF_ARM_PREFIX}_ungated"])
    np.testing.assert_array_equal(selected_raw, trajectories[SELECTED_BACKBONE_ARM])
    assert [record["selected_backbone"] for record in diagnostic["updates"]] == [
        record["selected_backbone"] for record in report["updates"]
    ]


def test_insufficient_support_is_bit_exact_persistence() -> None:
    physical, persistence, measurement, measurement_validity, centers = (
        _synthetic_inputs()
    )
    measurement_validity[:] = False
    measurement[:] = np.nan

    prediction, selected_raw, diagnostic = predict_support_gated_selected_backbone_rbf(
        physical,
        persistence,
        measurement,
        measurement_validity,
        center_ids=centers,
    )

    post_update_frames = np.asarray(
        [*range(20, 38), *range(39, 57), *range(58, 76)], dtype=np.int64
    )
    np.testing.assert_array_equal(
        prediction[post_update_frames], persistence[post_update_frames]
    )
    np.testing.assert_array_equal(
        selected_raw[post_update_frames], persistence[post_update_frames]
    )
    assert all(
        record["selector_decision"] == "insufficient_support_persistence"
        and record["rbf_correction_applied"] is False
        for record in diagnostic["updates"]
    )


def test_action_aligned_prefix_seeks_to_raw_start_and_never_reads_raw_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "undistorted.mp4"
    video.write_bytes(b"fixture")

    class Capture:
        def __init__(self) -> None:
            self.position = 0
            self.read_positions: list[int] = []
            self.set_calls: list[tuple[int, int]] = []

        def set(self, key: int, value: int) -> bool:
            self.set_calls.append((key, value))
            self.position = value
            return True

        def read(self) -> tuple[bool, np.ndarray]:
            position = self.position
            self.read_positions.append(position)
            self.position += 1
            bgr = np.asarray([[[position, position + 1, position + 2]]], dtype=np.uint8)
            return True, bgr

        def release(self) -> None:
            return None

    captures: list[Capture] = []

    def make_capture(_: str) -> Capture:
        capture = Capture()
        captures.append(capture)
        return capture

    fake_cv2 = SimpleNamespace(
        CAP_PROP_POS_FRAMES=7,
        COLOR_BGR2RGB=9,
        VideoCapture=make_capture,
        cvtColor=lambda value, _: value[..., ::-1],
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    expected_frame_zero = np.asarray([[[64, 63, 62]]], dtype=np.uint8)

    frames, audit = _decode_action_aligned_prefix(
        video,
        source_frame_start=62,
        logical_update_frame=2,
        expected_frame_zero=expected_frame_zero,
    )

    assert captures[0].set_calls == [(7, 62)]
    assert captures[0].read_positions == [62, 63, 64]
    assert frames[:, 0, 0, 2].tolist() == [62, 63, 64]
    assert audit["logical_prefix_frame_range_half_open"] == [0, 3]
    assert audit["source_prefix_frame_range_half_open"] == [62, 65]
    assert audit["maximum_source_rgb_frame_read"] == 64

    with pytest.raises(ValueError, match="logical frame zero differs"):
        _decode_action_aligned_prefix(
            video,
            source_frame_start=0,
            logical_update_frame=2,
            expected_frame_zero=expected_frame_zero,
        )


@pytest.mark.parametrize(
    "name, message",
    [
        ("mask_refined.h5", "HDF5"),
        ("rendered_depth.hdf5", "HDF5"),
        ("target_data.npz", "future-derived"),
        ("outcome.json", "outcome"),
        ("ground_truth.mp4", "future-derived"),
    ],
)
def test_forbidden_prefix_inputs_fail_closed(
    tmp_path: Path, name: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        reject_forbidden_prefix_input(tmp_path / name, purpose="test input")


def _make_held_chain(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    lock = tmp_path / "held-lock.json"
    create_held_protocol_lock(lock, immutable_bindings={"test": "a" * 64})
    bundle = tmp_path / "frame_zero_bundle.npz"
    camera_names = np.asarray(["cam0", "cam1"])
    point_count = 4
    points = np.column_stack(
        (
            np.linspace(0.0, 0.03, point_count),
            np.linspace(0.0, 0.03, point_count),
            np.ones(point_count),
        )
    ).astype(np.float32)
    camera_to_world = np.repeat(np.eye(4)[None], 2, axis=0)
    intrinsics = np.repeat(np.eye(3)[None], 2, axis=0)
    projection = np.repeat(
        np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)[None],
        2,
        axis=0,
    )
    np.savez_compressed(
        bundle,
        frame_indices=np.asarray([0], dtype=np.int64),
        camera_names=camera_names,
        rgb_frame0=np.zeros((2, 2, 2, 3), dtype=np.uint8),
        mask_frame0=np.ones((2, 2, 2), dtype=bool),
        depth_frame0_m=np.ones((2, 2, 2), dtype=np.float32),
        depth_valid_frame0=np.ones((2, 2, 2), dtype=bool),
        intrinsics=intrinsics,
        camera_to_world=camera_to_world,
        projection_world_to_pixel=projection,
        object_points_world_m=points,
        object_colors_rgb=np.ones((point_count, 3), dtype=np.float32) * 0.5,
    )
    robot = tmp_path / "robot.npz"
    robot.write_bytes(b"known robot trajectory")
    robot_metadata = tmp_path / "robot.meta.json"
    robot_metadata.write_text("{}\n", encoding="utf-8")
    selected_action = tmp_path / "known_action_76.npz"
    selected_action.write_bytes(b"selected known action")
    frame_manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "Deform360HeldFrameZeroBundle",
        "protocol_id": "deform360-held-online-belief-v1",
        "case_name": CASE_NAME,
        "object_id": "083-blanket-cloth",
        "episode_id": 0,
        "role": "calibration",
        "frame_indices": [0],
        "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "bundle": _record(bundle),
        "action_inputs": {
            "robot_trajectory": _record(robot),
            "robot_metadata": _record(robot_metadata),
        },
        "action_alignment": {
            "selected_raw_frame_range_half_open": [62, 143],
            "prediction_raw_frame_range_half_open": [62, 138],
            "selected_action_bundle": _record(selected_action),
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
    frame_manifest["artifact_sha256"] = held_artifact_sha256(frame_manifest)
    frame_manifest_path = tmp_path / "frame_zero_manifest.json"
    _write_json(frame_manifest_path, frame_manifest)

    physical = np.repeat(points[None], 76, axis=0)
    physical[:, :, 0] += np.linspace(0.0, 0.02, 76)[:, None]
    persistence = np.repeat(points[None], 76, axis=0)
    prediction_archive = tmp_path / "physical_prediction.npz"
    np.savez_compressed(
        prediction_archive,
        prediction_m=physical,
        persistence_m=persistence,
        frame_zero_points_m=points,
    )
    physical_files: dict[str, Path] = {
        "physical_prediction_archive": prediction_archive,
    }
    for role in (
        "prediction_only_input",
        "prediction_only_summary",
        "physical_prediction_manifest",
    ):
        path = tmp_path / f"{role}.bin"
        path.write_bytes(role.encode("utf-8"))
        physical_files[role] = path
    physical_seal = tmp_path / "physical_prior_seal.json"
    create_physical_prior_seal(
        physical_seal,
        lock,
        frame_manifest_path,
        physical_files,
        case_name=CASE_NAME,
        role="calibration",
    )
    authorization = tmp_path / "prefix_authorization.json"
    create_prefix_stage_authorization(authorization, lock, physical_seal)
    episode = tmp_path / "083-blanket-cloth" / "episode_0000"
    for camera in camera_names:
        camera_dir = episode / str(camera)
        camera_dir.mkdir(parents=True)
        (camera_dir / "undistorted.mp4").write_bytes(b"bounded RGB fixture")
    return lock, frame_manifest_path, physical_seal, authorization, episode


def test_runner_emits_exact_seven_roles_and_frozen_prediction_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock, frame_manifest, physical_seal, authorization, episode = _make_held_chain(
        tmp_path
    )
    config = RawCameraObservationConfig(
        center_count=3,
        selected_camera_count=2,
    )
    runtime = SimpleNamespace(
        config=config,
        source_sha256=ALLTRACKER_RUNTIME_SOURCE_SHA256,
        checkpoint_sha256=ALLTRACKER_CHECKPOINT_SHA256,
        device_name="test",
    )
    monkeypatch.setattr(
        held_prefix,
        "select_frame_zero_observation_plan",
        lambda *_args, **_kwargs: {
            "center_ids": np.asarray([0, 1, 2], dtype=np.int64),
            "candidate_ids": np.asarray([0, 1, 2, 3], dtype=np.int64),
            "selected_cameras": ("cam0", "cam1"),
            "selection_score": (4, 4, 0.0),
            "query_ids": {
                "cam0": np.asarray([0, 1, 2], dtype=np.int64),
                "cam1": np.asarray([0, 1, 2], dtype=np.int64),
            },
            "query_pixels": {
                "cam0": np.zeros((3, 2), dtype=float),
                "cam1": np.zeros((3, 2), dtype=float),
            },
        },
    )

    def fake_decode(
        _path: Path,
        *,
        source_frame_start: int,
        logical_update_frame: int,
        expected_frame_zero: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, object]]:
        frames = np.repeat(
            np.asarray(expected_frame_zero)[None], logical_update_frame + 1, axis=0
        )
        return frames, {
            "logical_prefix_frame_range_half_open": [0, logical_update_frame + 1],
            "source_prefix_frame_range_half_open": [
                source_frame_start,
                source_frame_start + logical_update_frame + 1,
            ],
            "maximum_logical_rgb_frame_read": logical_update_frame,
            "maximum_source_rgb_frame_read": source_frame_start + logical_update_frame,
            "decoded_frame_count": logical_update_frame + 1,
            "decoded_rgb_prefix_sha256": f"{logical_update_frame:064x}",
            "logical_frame_zero_matches_bundle": True,
            "whole_video_hashed_or_read": False,
        }

    monkeypatch.setattr(held_prefix, "_decode_action_aligned_prefix", fake_decode)
    monkeypatch.setattr(
        held_prefix,
        "_infer_tracks_from_rgb",
        lambda _runtime, _rgb, query, *, logical_update_frame, reverse: (
            np.asarray(query, dtype=np.float32),
            np.ones(len(query), dtype=bool),
            {
                "direction": "reverse" if reverse else "forward",
                "logical_update_frame": logical_update_frame,
            },
        ),
    )

    def fake_triangulate(
        observations: dict[str, np.ndarray],
        _projection: dict[str, np.ndarray],
        _origins: dict[str, np.ndarray],
        initial: np.ndarray,
        *,
        config: RawCameraObservationConfig,
    ) -> tuple[np.ndarray, dict[str, object]]:
        del config
        cameras = sorted(observations)
        return np.asarray(initial, dtype=np.float32), {
            "available_view_count": len(cameras),
            "accepted": True,
            "decision": "accepted",
            "inlier_view_count": len(cameras),
            "inlier_cameras": cameras,
            "median_reprojection_error_px": 0.5,
            "maximum_ray_angle_degrees": 45.0,
            "displacement_from_initial_m": 0.0,
        }

    monkeypatch.setattr(held_prefix, "triangulate_observation_ransac", fake_triangulate)
    monkeypatch.setattr(
        held_prefix,
        "jacobian_measurement_covariance",
        lambda *_args, **_kwargs: (
            np.eye(3) * 1.0e-4,
            {"decision": "accepted"},
        ),
    )
    monkeypatch.setattr(
        held_prefix,
        "leave_one_camera_out_covariance",
        lambda *_args, **_kwargs: (np.eye(3) * 1.0e-5, np.zeros((2, 3))),
    )
    monkeypatch.setattr(
        held_prefix,
        "inflate_covariance_from_cycle",
        lambda *_args, **_kwargs: (
            np.eye(3) * 2.0e-4,
            {
                "cycle_error_median_px": 0.0,
                "cycle_error_maximum_px": 0.0,
                "cycle_pixel_sigma": 0.5,
                "jacobian_covariance_scale": 1.0,
            },
        ),
    )
    output = tmp_path / "online"

    result = held_prefix.run_held_online_prefix_case(
        lock,
        frame_manifest,
        physical_seal,
        authorization,
        episode,
        output,
        runtime,
        case_name=CASE_NAME,
        role="calibration",
        observation_config=config,
    )

    seal = result["online_prediction_seal"]
    assert set(seal["online_artifacts"]) == set(ONLINE_ARTIFACT_ROLES)
    assert seal["information_boundary"]["outcome_read"] is False
    assert (
        result["measurement_manifest"]["tracker"]["query_routing"][
            "legacy_open27_routing_reused"
        ]
        is False
    )
    with np.load(output / "online_prediction.npz", allow_pickle=False) as stored:
        required = {
            "center_ids",
            "primary_prediction_m",
            "selected_raw_backbone_m",
            "frame_zero_points_m",
            "prediction_diagnostic_json_utf8",
        }
        assert required.issubset(stored.files)
        assert stored["center_ids"].shape == (3,)
        assert stored["primary_prediction_m"].shape == (76, 4, 3)
        assert stored["selected_raw_backbone_m"].shape == (76, 4, 3)
        assert stored["frame_zero_points_m"].shape == (4, 3)
        diagnostic = json.loads(
            stored["prediction_diagnostic_json_utf8"].tobytes().decode("utf-8")
        )
        assert len(diagnostic["updates"]) == 3
    with pytest.raises(FileExistsError):
        held_prefix.run_held_online_prefix_case(
            lock,
            frame_manifest,
            physical_seal,
            authorization,
            episode,
            output,
            runtime,
            case_name=CASE_NAME,
            role="calibration",
            observation_config=config,
        )
