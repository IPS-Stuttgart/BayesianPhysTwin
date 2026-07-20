from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import sys

import numpy as np
import pytest

from deform360_held_test_helpers import (
    dummy_immutable_bindings,
    write_robot_kinematics_fixture,
    write_robot_metadata_fixture,
)

import bayesian_phystwin.deform360_held_online_prefix as held_prefix
import bayesian_phystwin.deform360_held_physical_prior as held_physical
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
from bayesian_phystwin.deform360_raw_camera_uncertainty import (
    RawCameraUncertaintyConfig,
)
from bayesian_phystwin.phystwin_online_belief import RecursiveRbfBeliefConfig


CASE_NAME = "083-blanket-cloth-ep0000"
TEST_CAMERAS = (
    "brics-odroid-001_cam0",
    "cam1",
    "cam2",
    "cam3",
    "cam4",
    "cam5",
    "cam6",
    "cam7",
)


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


def _array_records(arrays: dict[str, np.ndarray]) -> dict[str, object]:
    return {
        name: {
            "shape": list(value.shape),
            "dtype": value.dtype.str,
            "sha256": held_prefix._sha256_array(value),
        }
        for name, value in sorted(arrays.items())
    }


def _selected_view_diagnostic(camera: str) -> dict[str, object]:
    return {
        "camera": camera,
        "automatic_candidate_count": 1,
        "eligible_candidate_count": 1,
        "rejected_candidate_count": 0,
        "rejection_counts": {
            "mask_threshold": 0,
            "reference_appearance_threshold": 0,
            "total": 0,
        },
        "maximum_reference_appearance_similarity": 1.0,
        "view_selected": True,
        "abstained": False,
        "abstention_reason": None,
        "selected": {"candidate_index": 0},
    }


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


def test_predictor_rejects_altered_rbf_length_scale() -> None:
    physical, persistence, measurement, measurement_validity, centers = (
        _synthetic_inputs()
    )
    altered = RecursiveRbfBeliefConfig(
        local_blend=1.0,
        length_scale_fraction=0.2,
    )

    with pytest.raises(ValueError, match="held RBF configuration changed"):
        predict_support_gated_selected_backbone_rbf(
            physical,
            persistence,
            measurement,
            measurement_validity,
            center_ids=centers,
            rbf_config=altered,
        )


@pytest.mark.parametrize(
    "altered",
    [
        RawCameraObservationConfig(selected_camera_count=7),
        RawCameraObservationConfig(alltracker_max_side=640),
    ],
)
def test_runner_rejects_altered_observation_configuration(altered: object) -> None:
    runtime = SimpleNamespace(
        config=altered,
        source_sha256=ALLTRACKER_RUNTIME_SOURCE_SHA256,
        checkpoint_sha256=ALLTRACKER_CHECKPOINT_SHA256,
    )
    with pytest.raises(ValueError, match="held observation configuration changed"):
        held_prefix.run_held_online_prefix_case(
            "missing-lock",
            "missing-frame-zero",
            "missing-physical-seal",
            "missing-prefix-authorization",
            "missing-episode",
            "missing-output",
            runtime,
            case_name=CASE_NAME,
            role="calibration",
            observation_config=altered,
        )


def test_runner_rejects_altered_uncertainty_floor() -> None:
    config = RawCameraObservationConfig()
    runtime = SimpleNamespace(
        config=config,
        source_sha256=ALLTRACKER_RUNTIME_SOURCE_SHA256,
        checkpoint_sha256=ALLTRACKER_CHECKPOINT_SHA256,
    )
    with pytest.raises(ValueError, match="held uncertainty configuration changed"):
        held_prefix.run_held_online_prefix_case(
            "missing-lock",
            "missing-frame-zero",
            "missing-physical-seal",
            "missing-prefix-authorization",
            "missing-episode",
            "missing-output",
            runtime,
            case_name=CASE_NAME,
            role="calibration",
            observation_config=config,
            uncertainty_config=RawCameraUncertaintyConfig(pixel_noise_floor_px=0.75),
        )


def test_state_action_alignment_uses_incoming_not_future_transition() -> None:
    frame_zero = held_prefix.logical_state_action_alignment(62, 0)
    update = held_prefix.logical_state_action_alignment(62, 19)

    assert frame_zero["source_state_frame"] == 62
    assert frame_zero["incoming_selected_action_transition"] is None
    assert update["source_state_frame"] == 81
    assert update["selected_action_state_index"] == 19
    assert update["incoming_selected_action_transition"] == [18, 19]
    assert update["incoming_source_action_transition"] == [80, 81]
    assert update["selected_robot_kinematics_state_index"] == 19
    assert update["incoming_selected_robot_kinematics_transition"] == [18, 19]
    assert update["commanded_control_or_delta_action_used"] is False
    assert update["selected_action_fields_are_compatibility_aliases"] is True


def test_nested_prefix_hashes_reject_earlier_frame_drift() -> None:
    old = tuple(f"{index:064x}" for index in range(20))
    current = [*old, *(f"{index:064x}" for index in range(20, 39))]
    assert held_prefix._require_nested_prefix_hashes(
        old, current, camera="cam", update=38
    ) == tuple(current)
    current[3] = "f" * 64
    with pytest.raises(ValueError, match="nested causal prefixes differ"):
        held_prefix._require_nested_prefix_hashes(old, current, camera="cam", update=38)


def test_physical_archive_validation_rejects_nonpersistent_control() -> None:
    points = np.zeros((16, 3), dtype=np.float32)
    repeated = np.repeat(points[None], 76, axis=0)
    arrays = {
        "prediction_m": repeated.copy(),
        "persistence_m": repeated.copy(),
        "driven_readout_m": repeated.copy(),
        "zero_action_readout_m": repeated.copy(),
        "action_support": np.ones(16, dtype=np.float32),
        "frame_zero_points_m": points.copy(),
    }
    held_prefix._validate_physical_archive_arrays(arrays, points)
    arrays["persistence_m"][1, 0, 0] = 0.01
    with pytest.raises(ValueError, match="persistence contract changed"):
        held_prefix._validate_physical_archive_arrays(arrays, points)


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
    lock_payload = create_held_protocol_lock(
        lock, immutable_bindings=dummy_immutable_bindings()
    )
    bundle = tmp_path / "frame_zero_bundle.npz"
    camera_count = len(TEST_CAMERAS)
    camera_names = np.asarray(TEST_CAMERAS)
    point_count = 128
    points = np.column_stack(
        (
            np.linspace(0.0, 0.03, point_count),
            np.linspace(0.0, 0.03, point_count),
            np.ones(point_count),
        )
    ).astype(np.float32)
    camera_to_world = np.repeat(np.eye(4)[None], camera_count, axis=0)
    intrinsics = np.repeat(np.eye(3)[None], camera_count, axis=0)
    projection = np.repeat(
        np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)[None],
        camera_count,
        axis=0,
    )
    bundle_arrays = {
        "frame_indices": np.asarray([0], dtype=np.int64),
        "camera_names": camera_names,
        "rgb_frame0": np.zeros((camera_count, 2, 2, 3), dtype=np.uint8),
        "mask_frame0": np.ones((camera_count, 2, 2), dtype=bool),
        "depth_frame0_m": np.ones((camera_count, 2, 2), dtype=np.float32),
        "depth_valid_frame0": np.ones((camera_count, 2, 2), dtype=bool),
        "intrinsics": intrinsics,
        "camera_to_world": camera_to_world,
        "projection_world_to_pixel": projection,
        "object_points_world_m": points,
        "object_colors_rgb": np.ones((point_count, 3), dtype=np.uint8) * 128,
        "object_color_support_count": np.ones(point_count, dtype=np.uint8) * 8,
        "visual_hull_points_world_m": points.copy(),
    }
    np.savez_compressed(bundle, **bundle_arrays)
    source_frame_count = 150
    episode = tmp_path / "aligned" / "083-blanket-cloth" / "episode_0000"
    robot_dir = episode / "robot"
    robot, _selected_action, action_alignment = write_robot_kinematics_fixture(
        robot_dir,
        source_frame_count=source_frame_count,
        selected_start=62,
    )
    robot_metadata = write_robot_metadata_fixture(
        robot_dir / "robot.meta.json",
        source_frame_count=source_frame_count,
        cameras=TEST_CAMERAS,
    )
    intrinsics_path = episode / "undistorted_intrinsics.npy"
    extrinsics_path = episode / "extrinsics.npy"
    np.save(
        intrinsics_path,
        {camera: np.eye(3, dtype=np.float64) for camera in TEST_CAMERAS},
    )
    np.save(
        extrinsics_path,
        {camera: np.eye(4, dtype=np.float64) for camera in TEST_CAMERAS},
    )
    for camera in camera_names:
        camera_dir = episode / str(camera)
        camera_dir.mkdir(parents=True)
        (camera_dir / "undistorted.mp4").write_bytes(b"bounded RGB fixture")
    view_diagnostics = [_selected_view_diagnostic(camera) for camera in TEST_CAMERAS]
    frame_manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "Deform360HeldFrameZeroBundle",
        "protocol_id": "deform360-held-online-belief-v6",
        "case_name": CASE_NAME,
        "object_id": "083-blanket-cloth",
        "episode_id": 0,
        "role": "calibration",
        "frame_indices": [0],
        "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "lock_artifact_sha256": lock_payload["artifact_sha256"],
        "bundle": _record(bundle),
        "arrays": _array_records(bundle_arrays),
        "action_inputs": {
            "robot_trajectory": _record(robot),
            "robot_metadata": _record(robot_metadata),
        },
        "calibration_inputs": {
            "intrinsics": _record(intrinsics_path),
            "extrinsics": _record(extrinsics_path),
        },
        "action_alignment": action_alignment,
        "camera_policy": {
            "policy_id": held_prefix.FRAME_ZERO_CAMERA_SELECTION_POLICY_ID,
            "rule": held_prefix.FRAME_ZERO_CAMERA_SELECTION_RULE,
            "reference_camera": TEST_CAMERAS[0],
            "minimum_selected_camera_count": 8,
            "candidate_cameras": list(TEST_CAMERAS),
            "candidate_camera_count": len(TEST_CAMERAS),
            "selected_cameras": list(TEST_CAMERAS),
            "selected_camera_count": len(TEST_CAMERAS),
            "abstained_cameras": [],
            "abstained_camera_count": 0,
        },
        "camera_frame_zero_access": [
            {
                "camera": camera,
                "path": str((episode / camera / "undistorted.mp4").resolve()),
                "source_aligned_frame_index": 62,
                "decoded_frame_count": 1,
                "maximum_rgb_frame_read": 0,
                "action_window_frame_index": 0,
                "decoded_rgb_sha256": "d" * 64,
                "whole_file_hashed_or_read": False,
            }
            for camera in TEST_CAMERAS
        ],
        "sam2": {
            **held_prefix.HELD_FRAME_ZERO_SAM2,
            "view_diagnostics": view_diagnostics,
            "view_diagnostics_sha256": (
                held_prefix.frame_zero_view_diagnostics_sha256(view_diagnostics)
            ),
        },
        "config": held_prefix.HELD_FRAME_ZERO_CONFIG,
        "geometry_qa": {"geometry_qa_passed": True},
        "information_boundary": {
            "maximum_object_rgb_frame_read": 0,
            "object_observation_frames_used": [0],
            "known_aligned_realized_robot_kinematics_read": True,
            "known_robot_trajectory_semantics": action_alignment[
                "trajectory_semantics"
            ],
            "robot_delta_command_read": False,
            "commanded_control_read": False,
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
    physical_arrays = {
        "prediction_m": physical,
        "persistence_m": persistence,
        "driven_readout_m": physical.copy(),
        "zero_action_readout_m": persistence.copy(),
        "action_support": np.ones(point_count, dtype=np.float32),
        "frame_zero_points_m": points,
    }
    prediction_archive = tmp_path / "physical_prediction.npz"
    np.savez_compressed(prediction_archive, **physical_arrays)
    prediction_input = tmp_path / "prediction_only_input.bin"
    prediction_input.write_bytes(b"prediction-only input")
    physical_input_paths = {"prediction_only_input": prediction_input}
    for input_role in (
        "simulator_final_data",
        "episode_graph",
        "state_artifact",
        "twin_summary",
        "driven_result",
        "zero_action_result",
        "driven_trajectory",
        "zero_action_trajectory",
    ):
        path = tmp_path / f"{input_role}.bin"
        path.write_bytes(input_role.encode("utf-8"))
        physical_input_paths[input_role] = path
    prediction_summary = tmp_path / "prediction_only_summary.json"
    prediction_summary.write_text("{}\n", encoding="utf-8")
    physical_manifest_path = tmp_path / "physical_prediction_manifest.json"
    _controllers, robot_kinematics_window = held_physical.load_controller_trajectory(
        action_alignment["selected_robot_kinematics_bundle"]["path"],
        source_robot_path=robot,
        expected_selected_raw_frame_range=[62, 143],
        expected_prediction_raw_frame_range=[62, 138],
    )
    physical_manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": held_physical.ARTIFACT_KIND,
        "protocol_id": "deform360-held-online-belief-v6",
        "case_name": CASE_NAME,
        "object_id": "083-blanket-cloth",
        "episode_id": 0,
        "role": "calibration",
        "physical_mode": "warp_twin",
        "physical_admitted": True,
        "fallback_diagnostics": None,
        "frozen_predictor": {
            "official_phystwin_revision": held_physical.OFFICIAL_PHYSTWIN_REVISION,
            "official_real_config_sha256": held_physical.OFFICIAL_REAL_CONFIG_SHA256,
            "length_scale_m": held_physical.LENGTH_SCALE_M,
            "action_response": held_physical.ACTION_RESPONSE,
            "autonomous_drift_response": held_physical.AUTONOMOUS_DRIFT_RESPONSE,
            "frame_count": 76,
            "observed_graph_node_count": 128,
            "total_graph_node_count": 128,
            "point_count": point_count,
            "warp_dynamics": held_physical.WARP_DYNAMICS,
        },
        "physical_prediction_archive": {
            **_record(prediction_archive),
            "array_sha256": {
                name: held_prefix._sha256_array(value)
                for name, value in physical_arrays.items()
            },
        },
        "input_files": {
            role: _record(path) for role, path in physical_input_paths.items()
        },
        "held_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "frame_zero_manifest_sha256": hashlib.sha256(
            frame_manifest_path.read_bytes()
        ).hexdigest(),
        "frame_zero_manifest_artifact_sha256": frame_manifest["artifact_sha256"],
        "robot_kinematics_window": robot_kinematics_window,
        "information_boundary": {
            "object_observation_frames_used": [0],
            "known_future_aligned_robot_kinematics_read": True,
            "known_future_robot_action_read": True,
            "future_object_rgb_read": False,
            "future_object_geometry_read": False,
            "future_object_track_read": False,
            "future_object_visibility_read": False,
            "future_tactile_read": False,
            "external_target_scoring_in_warp": False,
            "outcome_created": False,
            "outcome_read": False,
            "physical_prediction_hashed_before_outcome": True,
        },
        "passed": True,
    }
    physical_manifest["artifact_sha256"] = held_artifact_sha256(physical_manifest)
    _write_json(physical_manifest_path, physical_manifest)
    physical_files: dict[str, Path] = {
        "prediction_only_input": prediction_input,
        "prediction_only_summary": prediction_summary,
        "physical_prediction_archive": prediction_archive,
        "physical_prediction_manifest": physical_manifest_path,
    }
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
    return lock, frame_manifest_path, physical_seal, authorization, episode


def test_frame_zero_consumer_rejects_altered_builder_configuration(
    tmp_path: Path,
) -> None:
    lock, manifest_path, _, _, _ = _make_held_chain(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config"] = deepcopy(manifest["config"])
    manifest["config"]["rng_seed"] = 1
    manifest["artifact_sha256"] = held_artifact_sha256(manifest)
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="frame-zero configuration changed"):
        held_prefix._load_frame_zero_arrays(
            manifest_path,
            lock,
            case_name=CASE_NAME,
            role="calibration",
        )


def _npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {name: np.asarray(stored[name]).copy() for name in stored.files}


def test_online_prefix_independently_rejects_robot_schema_parity_and_scalar_drift(
    tmp_path: Path,
) -> None:
    _lock, manifest_path, _physical, _authorization, _episode = _make_held_chain(
        tmp_path
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_path = Path(manifest["action_inputs"]["robot_trajectory"]["path"])
    baseline = _npz_arrays(raw_path)
    cases: tuple[tuple[str, dict[str, np.ndarray], str], ...] = (
        (
            "extra",
            {**baseline, "unexpected": np.asarray(1, dtype=np.int64)},
            "field set changed",
        ),
        (
            "dtype",
            {**baseline, "actions": baseline["actions"].astype(np.float32)},
            "actions must have dtype float64",
        ),
        (
            "scalar",
            {**baseline, "bimanual": np.asarray([False], dtype=np.bool_)},
            "bimanual must be a bool scalar",
        ),
    )
    parity = {name: value.copy() for name, value in baseline.items()}
    parity["actions"][0, 0, 0] += 0.1
    cases = (*cases, ("parity", parity, "row 0 does not match"))
    for _name, arrays, message in cases:
        np.savez_compressed(raw_path, **arrays)
        manifest["action_inputs"]["robot_trajectory"] = _record(raw_path)
        with pytest.raises(ValueError, match=message):
            held_prefix._validate_selected_action_bundle(
                manifest, source_frame_start=62
            )
    np.savez_compressed(raw_path, **baseline)


def test_online_prefix_rejects_valid_but_wrong_selected_robot_slice(
    tmp_path: Path,
) -> None:
    _lock, manifest_path, _physical, _authorization, _episode = _make_held_chain(
        tmp_path
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected_path = Path(
        manifest["action_alignment"]["selected_robot_kinematics_bundle"]["path"]
    )
    arrays = _npz_arrays(selected_path)
    arrays["T_worlds"][:, 1, 3] += 0.05
    arrays["actions"][:, 0, 1] += 0.05
    np.savez_compressed(selected_path, **arrays)
    selected_record = _record(selected_path)
    selected_state = held_prefix.load_robot_kinematics_archive(selected_path)
    manifest["action_alignment"]["selected_robot_kinematics_bundle"] = selected_record
    manifest["action_alignment"]["selected_action_bundle"] = selected_record
    manifest["action_alignment"]["selected_action_arrays"] = (
        held_prefix.robot_kinematics_array_records(selected_state)
    )
    with pytest.raises(ValueError, match="exact source slice"):
        held_prefix._validate_selected_action_bundle(
            manifest, source_frame_start=62
        )


def test_aligned_metadata_validates_all_candidates_for_selected_subset(
    tmp_path: Path,
) -> None:
    _lock, manifest_path, _physical, _authorization, _episode = _make_held_chain(
        tmp_path
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected_subset = TEST_CAMERAS[1:5]
    manifest["camera_policy"]["selected_cameras"] = list(selected_subset)
    audit = held_prefix._validate_aligned_robot_video_metadata(
        manifest,
        selected_subset,
        source_frame_start=62,
        source_frame_stop=143,
    )
    assert audit["declared_candidate_camera_order"] == list(TEST_CAMERAS)
    assert audit["selected_bundle_camera_order"] == list(selected_subset)
    assert (
        audit["frame_zero_source_index_verified_for_every_candidate_camera"] is True
    )

    manifest["camera_frame_zero_access"][-1]["source_aligned_frame_index"] = 63
    with pytest.raises(ValueError, match="source index"):
        held_prefix._validate_aligned_robot_video_metadata(
            manifest,
            selected_subset,
            source_frame_start=62,
            source_frame_stop=143,
        )


def test_reference_abstention_requires_exact_preregistered_strategy_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        held_prefix,
        "validate_semantic_gate_audit",
        lambda _audit: {
            "true_label": "spider toy",
            "selected_cameras": ["camera-00"],
        },
    )
    assert held_prefix._reference_camera_may_be_absent({}) is False
    assert (
        held_prefix._reference_camera_may_be_absent(
            {"geometry_fallback": {"selected_strategy": "legacy"}}
        )
        is False
    )
    strategy = held_prefix.REFERENCE_OPTIONAL_COMMON_EXACT8_STRATEGY
    proposal = {
        "camera": "camera-00",
        "candidate_index": 2,
        "mask_sha256": "a" * 64,
    }
    assignment = {
        "strategy": held_prefix.REFERENCE_OPTIONAL_COMMON_ASSIGNMENT_STRATEGY,
        "policy_id": held_prefix.REFERENCE_OPTIONAL_COMMON_ASSIGNMENT_POLICY_ID,
        "selected_proposals": [proposal],
    }
    semantic_gate = {
        "selected_exact8": [
            {
                "camera": "camera-00",
                "candidate_index": 2,
                "selected_mask_sha256": "a" * 64,
            }
        ]
    }
    safeguard = {
        "contract_sha256": held_prefix.FRAME_ZERO_SEMANTIC_GATE_CONTRACT_SHA256,
        "assignment": assignment,
        "official_urdf": {},
        "semantic_selected_proposals": [proposal],
        "semantic_gate": semantic_gate,
        "robot_subtraction": {},
    }
    safeguard["artifact_sha256"] = held_prefix.held_artifact_sha256(safeguard)
    valid = {
        "object_id": "170-spider",
        "geometry_fallback": {
            "policy_id": held_prefix.REFERENCE_OPTIONAL_COMMON_ASSIGNMENT_POLICY_ID,
            "ordered_strategies": list(
                held_prefix.FRAME_ZERO_SEMANTIC_GATE_CONTRACT["application_order"]
            ),
            "selected_strategy": strategy,
            "attempts": [
                {"strategy": "legacy", "status": "failed"},
                {"strategy": strategy, "status": "passed"},
            ],
            "common_assignment": assignment,
            "reference_optional_safeguard": safeguard,
        }
    }
    assert held_prefix._reference_camera_may_be_absent(valid) is True

    for tampered in ("attempt", "assignment", "policy"):
        value = deepcopy(valid)
        if tampered == "attempt":
            value["geometry_fallback"]["attempts"][-1]["strategy"] = "other"
        elif tampered == "assignment":
            value["geometry_fallback"]["common_assignment"]["strategy"] = "other"
        else:
            value["geometry_fallback"]["common_assignment"]["policy_id"] = "other"
        with pytest.raises(ValueError, match="strategy audit changed"):
            held_prefix._reference_camera_may_be_absent(value)

    wrong_object = deepcopy(valid)
    wrong_object["object_id"] = "083-blanket-cloth"
    with pytest.raises(ValueError, match="held object/assignment"):
        held_prefix._reference_camera_may_be_absent(wrong_object)

    wrong_assignment_mask = deepcopy(valid)
    wrong_assignment_mask["geometry_fallback"]["common_assignment"][
        "selected_proposals"
    ][0]["mask_sha256"] = "b" * 64
    wrong_safeguard = wrong_assignment_mask["geometry_fallback"][
        "reference_optional_safeguard"
    ]
    wrong_safeguard["artifact_sha256"] = held_prefix.held_artifact_sha256(
        wrong_safeguard
    )
    with pytest.raises(ValueError, match="held object/assignment"):
        held_prefix._reference_camera_may_be_absent(wrong_assignment_mask)


def test_runner_emits_exact_seven_roles_and_frozen_prediction_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock, frame_manifest, physical_seal, authorization, episode = _make_held_chain(
        tmp_path
    )
    config = RawCameraObservationConfig()
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
            "center_ids": np.arange(16, dtype=np.int64),
            "candidate_ids": np.arange(16, dtype=np.int64),
            "selected_cameras": TEST_CAMERAS,
            "selection_score": (16, 16, 128, 45.0),
            "query_ids": {
                camera: np.arange(16, dtype=np.int64) for camera in TEST_CAMERAS
            },
            "query_pixels": {
                camera: np.zeros((16, 2), dtype=float) for camera in TEST_CAMERAS
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
            "decoded_rgb_frame_sha256": [
                held_prefix._sha256_array(np.asarray(expected_frame_zero))
            ]
            * (logical_update_frame + 1),
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
        lambda *_args, **_kwargs: (np.eye(3) * 1.0e-5, np.zeros((8, 3))),
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
    alignment = result["measurement_manifest"]["action_alignment"]
    assert alignment["commanded_control_or_delta_action_used"] is False
    assert (
        alignment["robot_kinematics_semantics"]
        == held_prefix.ROBOT_KINEMATICS_WINDOW_CONTRACT["trajectory_semantics"]
    )
    inputs = result["measurement_manifest"]["inputs"]
    assert (
        inputs["selected_robot_kinematics_bundle"]
        == inputs["selected_action_bundle"]
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
        assert stored["center_ids"].shape == (16,)
        assert stored["primary_prediction_m"].shape == (76, 128, 3)
        assert stored["selected_raw_backbone_m"].shape == (76, 128, 3)
        assert stored["frame_zero_points_m"].shape == (128, 3)
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
