from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import bayesian_phystwin.deform360_adaptive_covariance_confirmation_measurement as measurement
from bayesian_phystwin.deform360_adaptive_covariance_confirmation_lock import (
    write_confirmation_cohort_lock,
)
from bayesian_phystwin.deform360_raw_camera_observation import (
    RawCameraObservationConfig,
    project_world_points,
)


H1 = "a" * 40
H2 = "b" * 40


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _external_array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    descriptor = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(
        descriptor + b"\0" + array.view(np.uint8).tobytes()
    ).hexdigest()


def _result_sha256(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("result_sha256", None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _locked_identity(lock: dict[str, Any], case_id: str) -> dict[str, Any]:
    rows = [
        {
            "case": case_id,
            "object_id": record["object_id"],
            "episode_id": episode["episode_id"],
            "episode_key": f"{record['object_id']}/{episode['episode_id']}",
            "stratum": stratum,
            "role": "calibration",
        }
        for stratum, records in lock["cohort"].items()
        for record in records
        for episode in record["episodes"]
        if episode["case_id"] == case_id
    ]
    assert len(rows) == 1
    return rows[0]


def _write_external_physical_chain(
    root: Path,
    *,
    lock: dict[str, Any],
    case_id: str,
    frame_zero: np.ndarray,
    physical_mode: str = "warp_twin",
) -> tuple[Path, Path, Path, np.ndarray]:
    root.mkdir()
    physical = np.repeat(frame_zero[None], 76, axis=0)
    arrays = {
        "action_support": np.zeros(len(frame_zero), dtype=np.float32),
        "driven_readout_m": physical.copy(),
        "frame_zero_points_m": frame_zero.copy(),
        "persistence_m": physical.copy(),
        "prediction_m": physical.copy(),
        "zero_action_readout_m": physical.copy(),
    }
    physical_path = root / "physical_prediction.npz"
    np.savez_compressed(physical_path, **arrays)
    array_hashes = {
        role: _external_array_sha256(arrays[role])
        for role in measurement.EXTERNAL_PHYSICAL_ARRAY_ROLES
    }
    identity = _locked_identity(lock, case_id)
    manifest_path = root / measurement.EXTERNAL_PHYSICAL_MANIFEST_FILENAME
    # The frozen outer sealer copies this manifest without rewriting the work
    # archive path.  The file and array identities, not this stale path, bind
    # the copied archive.
    if physical_mode == "warp_twin":
        input_roles = measurement._WARP_EXTERNAL_PHYSICAL_INPUT_ROLES
        fallback_diagnostics = None
    else:
        assert physical_mode == "persistence_fallback"
        input_roles = measurement._TWIN_EXTERNAL_PHYSICAL_INPUT_ROLES
        fallback_diagnostics = {
            "reason": "automatic_twin_source_admission_failed",
            "automatic_twin_exit_code": 3,
            "automatic_twin_result_sha256": "4" * 64,
            "automatic_twin_state_metrics": {
                "passed": False,
                "observed_target_fraction": 0.5,
                "symmetric_chamfer_m": 0.01,
            },
            "warp_attempted": False,
        }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": measurement.EXTERNAL_PHYSICAL_ARTIFACT_KIND,
        "protocol_id": measurement.PROTOCOL_ID,
        "protocol_config_sha256": lock["artifact_sha256"],
        **identity,
        "physical_mode": physical_mode,
        "physical_admitted": physical_mode == "warp_twin",
        "fallback_diagnostics": fallback_diagnostics,
        "frozen_predictor": {"frame_count": 76},
        "physical_prediction_archive": {
            "path": "/stale/work/physical_prediction.npz",
            "file_sha256": _file_sha256(physical_path),
            "array_sha256": array_hashes,
        },
        "input_files": {
            role: {
                "path": f"/sealed/input/{role}",
                "sha256": "3" * 64,
            }
            for role in input_roles
        },
        "runtime_provenance": {},
        "information_boundary": dict(measurement._EXTERNAL_PHYSICAL_BOUNDARY),
        "passed": True,
    }
    manifest["result_sha256"] = _result_sha256(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    seal_path = root / measurement.EXTERNAL_BACKBONE_SEAL_FILENAME
    seal: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": measurement.EXTERNAL_BACKBONE_ARTIFACT_KIND,
        "protocol_id": measurement.PROTOCOL_ID,
        "protocol_config_sha256": lock["artifact_sha256"],
        **identity,
        "frame_count": 76,
        "material_point_count": len(frame_zero),
        "material_identity_sha256": array_hashes["frame_zero_points_m"],
        "prediction_archive": {
            "path": str(physical_path),
            "file_sha256": _file_sha256(physical_path),
            "array_sha256": array_hashes,
        },
        "physical_manifest": {
            "path": str(manifest_path),
            "file_sha256": _file_sha256(manifest_path),
        },
        "information_boundary": dict(measurement._EXTERNAL_BACKBONE_BOUNDARY),
    }
    seal["result_sha256"] = _result_sha256(seal)
    seal_path.write_text(
        json.dumps(seal, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return physical_path, manifest_path, seal_path, physical


def _camera_to_world(x: float, y: float) -> np.ndarray:
    result = np.eye(4)
    result[:3, 3] = (x, y, 0.0)
    return result


class _FakeRuntime:
    def __init__(self, config: RawCameraObservationConfig) -> None:
        self.config = config
        self.source_sha256 = measurement.ALLTRACKER_RUNTIME_SOURCE_SHA256
        self.checkpoint_sha256 = measurement.ALLTRACKER_CHECKPOINT_SHA256
        self.device_name = "synthetic-cpu"
        self.calls: list[tuple[int, str]] = []

    def track_prefix(
        self,
        video_path: Path,
        query_pixels: np.ndarray,
        update_frame: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        self.calls.append((update_frame, video_path.parent.name))
        return (
            np.asarray(query_pixels, dtype=np.float32).copy(),
            np.ones(len(query_pixels), dtype=bool),
            {
                "maximum_video_frame_read": update_frame,
                "decoded_rgb_prefix_sha256": f"{update_frame:064x}",
            },
        )


def test_nested_measurement_builder_is_target_free_and_materializes_shadow_order(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    lock_path = tmp_path / "h2-lock.json"
    lock = write_confirmation_cohort_lock(lock_path, H1)
    case_id = lock["selected_case_ids"][0]
    point_count = 24
    angle = np.linspace(0.0, 2.0 * np.pi, point_count, endpoint=False)
    frame_zero = np.stack(
        (
            0.2 * np.cos(angle),
            0.15 * np.sin(angle),
            np.full(point_count, 2.0),
        ),
        axis=1,
    ).astype(np.float32)
    physical_path, physical_manifest, physical_seal, physical = (
        _write_external_physical_chain(
            tmp_path / "backbone",
            lock=lock,
            case_id=case_id,
            frame_zero=frame_zero,
        )
    )
    processed = tmp_path / "prefix"
    processed.mkdir()
    np.save(processed / "undistorted_intrinsics.npy", np.asarray([1]))
    np.save(processed / "extrinsics.npy", np.asarray([1]))
    cameras = tuple(f"camera-{index:02d}" for index in range(8))
    intrinsic = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    intrinsics = {camera: intrinsic for camera in cameras}
    extrinsics = {
        camera: _camera_to_world(
            0.4 * np.cos(2.0 * np.pi * index / len(cameras)),
            0.4 * np.sin(2.0 * np.pi * index / len(cameras)),
        )
        for index, camera in enumerate(cameras)
    }
    projected = {
        camera: project_world_points(
            frame_zero,
            intrinsics[camera],
            extrinsics[camera],
        )[0]
        for camera in cameras
    }
    support = np.ones((point_count, len(cameras)), dtype=bool)
    monkeypatch.setattr(
        measurement,
        "_load_calibration",
        lambda _processed: (intrinsics, extrinsics),
    )
    monkeypatch.setattr(
        measurement,
        "frame_zero_camera_support",
        lambda *_args, **_kwargs: (cameras, support, projected),
    )
    monkeypatch.setattr(
        measurement,
        "_causal_selected_camera_inputs",
        lambda *_args, **_kwargs: {"synthetic": True},
    )
    monkeypatch.setattr(
        measurement,
        "_validated_source_stage_lineage",
        lambda **_kwargs: {
            "manifest_record": {
                "prediction_prefix_manifest": {
                    "path": "/prefix.json",
                    "file_sha256": "1" * 64,
                    "result_sha256": "2" * 64,
                },
                "frame_zero_manifest": {
                    "path": "/frame-zero.json",
                    "file_sha256": "3" * 64,
                    "result_sha256": "4" * 64,
                },
                "source_preparation_manifest_file_sha256": "5" * 64,
            },
            "camera_records": {
                camera: {
                    "camera": camera,
                    "prefix_video_sha256": "6" * 64,
                    "frame_zero_video_sha256": "7" * 64,
                    "frame_zero_mask_sha256": "8" * 64,
                }
                for camera in cameras
            },
            "planning_cameras": cameras,
            "depth_file_sha256_by_camera": {camera: "9" * 64 for camera in cameras},
            "snapshots": (),
        },
    )
    monkeypatch.setattr(
        measurement,
        "_validate_planning_camera_source_bindings",
        lambda *_args, **_kwargs: {
            "frame_zero_records": {},
            "snapshots": (),
        },
    )
    config = RawCameraObservationConfig(selected_camera_count=8)
    runtime = _FakeRuntime(config)

    result = measurement.build_confirmation_nested_measurements(
        lock_path,
        H2,
        case_id,
        physical_path,
        processed,
        tmp_path / "measurement-build",
        runtime,
        physical_manifest=physical_manifest,
        physical_prediction_seal=physical_seal,
        source_custody_seal=tmp_path / "source-custody.json",
        expected_h1=H1,
        observation_config=config,
    )

    assert result["case_identity"]["case_id"] == case_id
    assert result["camera_accounting"] == {
        "adaptive_charge_is_causal_offline_policy_demand": True,
        "all_eight_streams_eventually_tracked_for_fixed8_shadow": True,
        "realized_acquisition_or_wall_clock_saving_claimed": False,
        "frame_zero_all_camera_planning_excluded": True,
    }
    assert len(result["updates"]) == 3
    assert all(
        update["four_view_decision_materialized_before_shadow_extra_four"] is True
        for update in result["updates"]
    )
    activation = result["plan"]["camera_activation_order"]
    assert len(activation) == 8 and len(set(activation)) == 8
    assert result["plan"]["selected_cameras_by_budget"] == {
        "4": activation[:4],
        "8": activation,
    }
    assert runtime.calls == [
        (frame, camera) for frame in (19, 38, 57) for camera in activation
    ]
    for update in result["updates"]:
        tracker = update["tracker"]
        assert [record["execution_index_within_update"] for record in tracker] == list(
            range(8)
        )
        assert all(
            record["execution_role"] == "adaptive_first_four"
            and record["four_view_decision_already_materialized"] is False
            for record in tracker[:4]
        )
        assert all(
            record["execution_role"]
            in {
                "fixed_eight_shadow_after_four_decision",
                "adaptive_eight_escalation",
            }
            and record["four_view_decision_already_materialized"] is True
            for record in tracker[4:]
        )
    assert (
        result["inputs"]["physical_backbone"]["external_backbone_seal_result_sha256"]
        == json.loads(physical_seal.read_text(encoding="utf-8"))["result_sha256"]
    )
    for budget in (4, 8):
        root = tmp_path / "measurement-build" / f"budget-{budget}"
        with np.load(
            root / measurement.MEASUREMENT_ARCHIVE_FILENAME,
            allow_pickle=False,
        ) as stored:
            assert set(stored.files) == set(measurement.MEASUREMENT_ARRAY_ROLES)
            assert stored["selected_cameras"].tolist() == activation[:budget]
            assert stored["measurement_m"].shape == physical.shape
            assert stored["measurement_m"].dtype == np.dtype(np.float32)
            assert stored["center_ids"].shape == (16,)
            centers = np.asarray(stored["center_ids"], dtype=np.int64)
            np.testing.assert_allclose(
                stored["measurement_m"][19, centers],
                frame_zero[centers],
                rtol=0.0,
                atol=1.0e-6,
            )
            center_id = int(centers[0])
            triangulated = np.asarray(
                stored["measurement_m"][19, center_id],
                dtype=float,
            )
        with np.load(
            root / measurement.UNCERTAINTY_ARCHIVE_FILENAME,
            allow_pickle=False,
        ) as stored:
            assert set(stored.files) == set(measurement.UNCERTAINTY_ARRAY_ROLES)
            assert stored["measurement_covariance_m2"].shape == (
                76,
                point_count,
                3,
                3,
            )
            assert stored["measurement_covariance_m2"].dtype == np.dtype(np.float32)
            center_record = next(
                record
                for record in result["updates"][0]["centers"][str(budget)]
                if record["center_id"] == center_id
            )
            inlier_cameras = tuple(center_record["inlier_cameras"])
            projection_matrices = {
                camera: measurement._projection_matrix(
                    intrinsics[camera],
                    extrinsics[camera],
                )
                for camera in inlier_cameras
            }
            observations = {
                camera: projected[camera][center_id] for camera in inlier_cameras
            }
            geometric, diagnostic = measurement.jacobian_measurement_covariance(
                triangulated,
                [projection_matrices[camera] for camera in sorted(inlier_cameras)],
                center_record["pixel_sigma"],
                maximum_condition_number=(
                    measurement.RawCameraUncertaintyConfig().maximum_information_condition_number
                ),
            )
            assert geometric is not None and diagnostic["decision"] == "accepted"
            empirical, _ = measurement.leave_one_camera_out_covariance(
                observations,
                projection_matrices,
            )
            expected_covariance = geometric + empirical
            expected_covariance = 0.5 * (expected_covariance + expected_covariance.T)
            np.testing.assert_allclose(
                stored["measurement_covariance_m2"][19, center_id],
                expected_covariance,
                rtol=2.0e-6,
                atol=1.0e-10,
            )
        output_record = result["outputs"][str(budget)]
        assert set(output_record["measurement_archive"]["arrays"]) == set(
            measurement.MEASUREMENT_ARRAY_ROLES
        )
        assert set(output_record["uncertainty_archive"]["arrays"]) == set(
            measurement.UNCERTAINTY_ARRAY_ROLES
        )


def test_nested_measurement_builder_has_no_evaluation_argument() -> None:
    parameters = inspect.signature(
        measurement.build_confirmation_nested_measurements
    ).parameters

    assert all(
        token not in name
        for name in parameters
        for token in ("target", "outcome", "metric", "score", "ground_truth")
    )


def test_source_stage_lineage_rejects_cross_case_and_mutated_camera_inputs(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    h5py = pytest.importorskip("h5py")

    def write_frame_zero(path: Path, value: np.ndarray) -> None:
        with h5py.File(path, "w") as handle:
            handle.create_dataset("data", data=np.asarray(value)[None])

    lock_path = tmp_path / "h2-lock.json"
    lock = write_confirmation_cohort_lock(lock_path, H1)
    case_id = lock["selected_case_ids"][0]
    staged = tmp_path / case_id
    processed = staged / "prefix" / "episode_0000"
    cameras = ("camera-00", "camera-01")
    mask_zero_by_camera = {
        cameras[0]: np.asarray([[1, 0], [0, 1]], dtype=np.uint8),
        cameras[1]: np.asarray([[0, 1], [1, 0]], dtype=np.uint8),
    }
    depth_zero_by_camera = {
        cameras[0]: np.asarray([[1.0, 1.1], [1.2, 1.3]], dtype=np.float32),
        cameras[1]: np.asarray([[2.0, 2.1], [2.2, 2.3]], dtype=np.float32),
    }
    videos: dict[str, Path] = {}
    masks: dict[str, Path] = {}
    depths: dict[str, Path] = {}
    for index, camera in enumerate(cameras):
        camera_root = processed / camera
        camera_root.mkdir(parents=True)
        videos[camera] = camera_root / "undistorted.mp4"
        videos[camera].write_bytes(f"sealed-prefix-video-{index}".encode())
        masks[camera] = camera_root / "mask_refined.h5"
        depths[camera] = camera_root / "rendered_depth.h5"
        write_frame_zero(masks[camera], mask_zero_by_camera[camera])
        write_frame_zero(depths[camera], depth_zero_by_camera[camera])
    original_mask_file_by_camera = {
        camera: masks[camera].read_bytes() for camera in cameras
    }
    identity = _locked_identity(lock, case_id)
    prefix: dict[str, Any] = {
        "artifact_kind": "Deform360BiasAwarePredictionPrefix",
        "protocol_id": measurement.PROTOCOL_ID,
        "protocol_config_sha256": lock["artifact_sha256"],
        **identity,
        "inputs_sha256": {
            "protocol": _file_sha256(lock_path),
            "source_preparation_manifest": "9" * 64,
        },
        "camera_count": len(cameras),
        "camera_records": [
            {
                "camera": camera,
                "prefix_video_sha256": _file_sha256(videos[camera]),
                "frame_zero_video_sha256": "7" * 64,
                "frame_zero_mask_sha256": _file_sha256(masks[camera]),
            }
            for camera in cameras
        ],
    }
    prefix["result_sha256"] = _result_sha256(prefix)
    prefix_path = staged / "prediction_prefix_manifest.json"
    prefix_path.write_text(
        json.dumps(prefix, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    frame: dict[str, Any] = {
        "artifact_kind": "Deform360BiasAwareFrameZeroReconstruction",
        "protocol_id": measurement.PROTOCOL_ID,
        "protocol_config_sha256": lock["artifact_sha256"],
        **identity,
        "inputs_sha256": {
            "prediction_prefix_manifest": _file_sha256(prefix_path),
        },
        "cameras": list(cameras),
        "camera_count": len(cameras),
        "outputs_sha256": {
            "depth_by_camera": {
                camera: _file_sha256(depths[camera]) for camera in cameras
            }
        },
    }
    frame["result_sha256"] = _result_sha256(frame)
    frame_path = staged / "frame_zero_reconstruction_manifest.json"
    frame_path.write_text(
        json.dumps(frame, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    physical_manifest = tmp_path / measurement.EXTERNAL_PHYSICAL_MANIFEST_FILENAME
    physical_manifest.write_text(
        json.dumps(
            {
                "input_files": {
                    "prediction_prefix_manifest": {
                        "path": str(prefix_path),
                        "sha256": _file_sha256(prefix_path),
                    },
                    "frame_zero_manifest": {
                        "path": str(frame_path),
                        "sha256": _file_sha256(frame_path),
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    source_custody = tmp_path / "source-custody.json"
    source_custody.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        measurement,
        "validate_confirmation_source_custody_envelope",
        lambda *_args, **_kwargs: {
            "artifact_sha256": "8" * 64,
            "camera_panel": list(cameras),
            "manifests": {
                "prediction_prefix": {
                    "file_sha256": _file_sha256(prefix_path),
                    "result_sha256": prefix["result_sha256"],
                },
                "frame_zero": {
                    "file_sha256": _file_sha256(frame_path),
                    "result_sha256": frame["result_sha256"],
                },
                "source_preparation": {"file_sha256": "9" * 64},
            },
        },
    )
    lineage = measurement._validated_source_stage_lineage(
        lock=lock,
        lock_snapshot=measurement._snapshot_regular_file(
            lock_path,
            label="H2 lock",
        ),
        identity=measurement._case_identity(lock, case_id),
        h2_commit=H2,
        physical_manifest_snapshot=measurement._snapshot_regular_file(
            physical_manifest,
            label="physical manifest",
        ),
        processed_episode_dir=processed,
        source_custody_seal=source_custody,
    )
    selected_camera = cameras[0]
    selected = {
        selected_camera: {
            "video": {"path": str(videos[selected_camera])},
            "frame_zero_mask": {
                "path": str(masks[selected_camera]),
                "frame_zero_array_sha256": measurement._array_sha256(
                    mask_zero_by_camera[selected_camera]
                ),
            },
            "frame_zero_depth": {
                "path": str(depths[selected_camera]),
                "frame_zero_array_sha256": measurement._array_sha256(
                    depth_zero_by_camera[selected_camera]
                ),
            },
        }
    }
    measurement._validate_planning_camera_source_bindings(
        processed,
        selected,
        lineage["camera_records"],
        lineage["planning_cameras"],
        lineage["depth_file_sha256_by_camera"],
    )
    with pytest.raises(ValueError, match="calibration camera panel differs"):
        measurement._validate_calibration_camera_panel(
            {cameras[0]: np.eye(3)},
            {camera: np.eye(4) for camera in cameras},
            lineage["planning_cameras"],
        )

    with pytest.raises(ValueError, match="outside the exact physical staged case"):
        measurement._validated_source_stage_lineage(
            lock=lock,
            lock_snapshot=measurement._snapshot_regular_file(
                lock_path,
                label="H2 lock",
            ),
            identity=measurement._case_identity(lock, case_id),
            h2_commit=H2,
            physical_manifest_snapshot=measurement._snapshot_regular_file(
                physical_manifest,
                label="physical manifest",
            ),
            processed_episode_dir=tmp_path / "another-case" / "prefix" / "episode_0000",
            source_custody_seal=source_custody,
        )
    videos[selected_camera].write_bytes(b"swapped-video")
    with pytest.raises(ValueError, match="video differs"):
        measurement._validate_planning_camera_source_bindings(
            processed,
            selected,
            lineage["camera_records"],
            lineage["planning_cameras"],
            lineage["depth_file_sha256_by_camera"],
        )
    videos[selected_camera].write_bytes(b"sealed-prefix-video-0")
    selected[selected_camera]["frame_zero_mask"]["frame_zero_array_sha256"] = "5" * 64
    with pytest.raises(ValueError, match="camera files differ"):
        measurement._validate_planning_camera_source_bindings(
            processed,
            selected,
            lineage["camera_records"],
            lineage["planning_cameras"],
            lineage["depth_file_sha256_by_camera"],
        )
    selected[selected_camera]["frame_zero_mask"]["frame_zero_array_sha256"] = (
        measurement._array_sha256(mask_zero_by_camera[selected_camera])
    )

    unselected_camera = cameras[1]
    write_frame_zero(
        masks[unselected_camera],
        np.zeros_like(mask_zero_by_camera[unselected_camera]),
    )
    with pytest.raises(ValueError, match="planning mask/depth differs"):
        measurement._validate_planning_camera_source_bindings(
            processed,
            selected,
            lineage["camera_records"],
            lineage["planning_cameras"],
            lineage["depth_file_sha256_by_camera"],
        )
    masks[unselected_camera].write_bytes(
        original_mask_file_by_camera[unselected_camera]
    )
    write_frame_zero(
        depths[unselected_camera],
        depth_zero_by_camera[unselected_camera] + np.float32(1.0),
    )
    with pytest.raises(ValueError, match="planning mask/depth differs"):
        measurement._validate_planning_camera_source_bindings(
            processed,
            selected,
            lineage["camera_records"],
            lineage["planning_cameras"],
            lineage["depth_file_sha256_by_camera"],
        )


def test_nested_measurement_rejects_same_shape_backbone_from_another_case(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "h2-lock.json"
    lock = write_confirmation_cohort_lock(lock_path, H1)
    source_case, requested_case = lock["selected_case_ids"][:2]
    frame_zero = np.column_stack(
        (
            np.linspace(-0.1, 0.1, 24),
            np.zeros(24),
            np.full(24, 2.0),
        )
    ).astype(np.float32)
    physical_path, physical_manifest, physical_seal, _ = _write_external_physical_chain(
        tmp_path / "backbone",
        lock=lock,
        case_id=source_case,
        frame_zero=frame_zero,
    )
    output = tmp_path / "measurement-build"

    with pytest.raises(ValueError, match="not bound to this H2 case"):
        measurement.build_confirmation_nested_measurements(
            lock_path,
            H2,
            requested_case,
            physical_path,
            tmp_path / "prefix",
            output,
            _FakeRuntime(RawCameraObservationConfig(selected_camera_count=8)),
            physical_manifest=physical_manifest,
            physical_prediction_seal=physical_seal,
            source_custody_seal=tmp_path / "source-custody.json",
            expected_h1=H1,
        )

    assert not output.exists()


def test_nested_measurement_accepts_bound_external_persistence_disposition(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "h2-lock.json"
    lock = write_confirmation_cohort_lock(lock_path, H1)
    case_id = lock["selected_case_ids"][0]
    frame_zero = np.column_stack(
        (
            np.linspace(-0.1, 0.1, 24),
            np.zeros(24),
            np.full(24, 2.0),
        )
    ).astype(np.float32)
    physical_path, physical_manifest, physical_seal, _ = _write_external_physical_chain(
        tmp_path / "backbone",
        lock=lock,
        case_id=case_id,
        frame_zero=frame_zero,
        physical_mode="persistence_fallback",
    )
    archive_snapshot = measurement._snapshot_regular_file(
        physical_path,
        label="physical archive",
    )
    arrays = measurement._physical_arrays(archive_snapshot)

    provenance = measurement._validate_external_physical_provenance(
        lock=lock,
        identity=measurement._case_identity(lock, case_id),
        archive_snapshot=archive_snapshot,
        manifest_snapshot=measurement._snapshot_regular_file(
            physical_manifest,
            label="external physical manifest",
        ),
        seal_snapshot=measurement._snapshot_regular_file(
            physical_seal,
            label="external backbone seal",
        ),
        arrays=arrays,
    )

    assert provenance["physical_archive_file_sha256"] == _file_sha256(physical_path)


def test_nested_measurement_rejects_causality_claim_from_runtime(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A tracker cannot merely claim a longer prefix and still be consumed."""

    lock_path = tmp_path / "h2-lock.json"
    lock = write_confirmation_cohort_lock(lock_path, H1)
    case_id = lock["selected_case_ids"][0]
    count = 24
    frame_zero = np.column_stack(
        (
            np.linspace(-0.1, 0.1, count),
            np.linspace(-0.05, 0.05, count),
            np.full(count, 2.0),
        )
    ).astype(np.float32)
    physical_path, physical_manifest, physical_seal, _ = _write_external_physical_chain(
        tmp_path / "backbone",
        lock=lock,
        case_id=case_id,
        frame_zero=frame_zero,
    )
    processed = tmp_path / "prefix"
    processed.mkdir()
    np.save(processed / "undistorted_intrinsics.npy", np.asarray([1]))
    np.save(processed / "extrinsics.npy", np.asarray([1]))
    cameras = tuple(f"camera-{index:02d}" for index in range(8))
    intrinsic = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    intrinsics = {camera: intrinsic for camera in cameras}
    extrinsics = {
        camera: _camera_to_world(
            0.4 * np.cos(2.0 * np.pi * index / len(cameras)),
            0.4 * np.sin(2.0 * np.pi * index / len(cameras)),
        )
        for index, camera in enumerate(cameras)
    }
    projected = {
        camera: project_world_points(
            frame_zero,
            intrinsics[camera],
            extrinsics[camera],
        )[0]
        for camera in cameras
    }
    monkeypatch.setattr(
        measurement,
        "_load_calibration",
        lambda _processed: (intrinsics, extrinsics),
    )
    monkeypatch.setattr(
        measurement,
        "frame_zero_camera_support",
        lambda *_args, **_kwargs: (
            cameras,
            np.ones((count, len(cameras)), dtype=bool),
            projected,
        ),
    )
    monkeypatch.setattr(
        measurement,
        "_validated_source_stage_lineage",
        lambda **_kwargs: {
            "manifest_record": {},
            "camera_records": {
                camera: {
                    "camera": camera,
                    "prefix_video_sha256": "6" * 64,
                    "frame_zero_video_sha256": "7" * 64,
                    "frame_zero_mask_sha256": "8" * 64,
                }
                for camera in cameras
            },
            "planning_cameras": cameras,
            "depth_file_sha256_by_camera": {camera: "9" * 64 for camera in cameras},
            "snapshots": (),
        },
    )

    class _OverreadRuntime(_FakeRuntime):
        def track_prefix(
            self,
            video_path: Path,
            query_pixels: np.ndarray,
            update_frame: int,
        ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
            tracks, visible, record = super().track_prefix(
                video_path,
                query_pixels,
                update_frame,
            )
            record["maximum_video_frame_read"] = update_frame + 1
            return tracks, visible, record

    output = tmp_path / "measurement-build"
    with pytest.raises(ValueError, match="prefix provenance"):
        measurement.build_confirmation_nested_measurements(
            lock_path,
            H2,
            case_id,
            physical_path,
            processed,
            output,
            _OverreadRuntime(RawCameraObservationConfig(selected_camera_count=8)),
            physical_manifest=physical_manifest,
            physical_prediction_seal=physical_seal,
            source_custody_seal=tmp_path / "source-custody.json",
            expected_h1=H1,
        )

    assert not output.exists()
