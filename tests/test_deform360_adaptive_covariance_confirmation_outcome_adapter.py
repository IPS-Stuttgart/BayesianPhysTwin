from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pytest

import bayesian_phystwin.deform360_adaptive_covariance_confirmation_external_runtime as runtime
import bayesian_phystwin.deform360_adaptive_covariance_confirmation_failure as failure
import bayesian_phystwin.deform360_adaptive_covariance_confirmation_lock as lock
import bayesian_phystwin.deform360_adaptive_covariance_confirmation_measurement as measurement
import bayesian_phystwin.deform360_adaptive_covariance_confirmation_outcome_adapter as adapter
import bayesian_phystwin.deform360_adaptive_covariance_confirmation_scoring as scoring
import bayesian_phystwin.deform360_adaptive_covariance_confirmation_seal as seal
import bayesian_phystwin.deform360_raw_camera_observation as raw_observation
from bayesian_phystwin.deform360_adaptive_covariance_rbf import (
    ADAPTIVE_COVARIANCE_PROTOCOL_ID,
)


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_fixture_artifact(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return _sha256(path)


def _synthetic_decoded_payload_sha256(payload: bytes, *, frame_count: int) -> str:
    digest = hashlib.sha256()
    digest.update(b"synthetic-decoded-rgb24-prefix-v1\0")
    digest.update(str(frame_count).encode("ascii"))
    digest.update(b"\0")
    digest.update(payload)
    return digest.hexdigest()


def _synthetic_decoded_prefix_sha256(path: Path, *, frame_count: int) -> str:
    """Stand in only for ffmpeg while preserving content-sensitive binding."""

    return _synthetic_decoded_payload_sha256(
        path.read_bytes(),
        frame_count=frame_count,
    )


@pytest.fixture(scope="module", autouse=True)
def _synthetic_source_custody_validator() -> Any:
    """Keep outcome tests focused; the custody module has full replay tests."""

    original_envelope = adapter.validate_confirmation_source_custody_envelope
    original_replay = adapter.validate_confirmation_source_custody_seal
    original_decode = adapter._decoded_raw_rgb_prefix_sha256

    def validate(seal_path: str | Path, *_args: object, **_kwargs: object) -> Any:
        return json.loads(Path(seal_path).read_bytes())

    adapter.validate_confirmation_source_custody_envelope = validate
    adapter.validate_confirmation_source_custody_seal = validate
    adapter._decoded_raw_rgb_prefix_sha256 = _synthetic_decoded_prefix_sha256
    try:
        yield
    finally:
        adapter.validate_confirmation_source_custody_envelope = original_envelope
        adapter.validate_confirmation_source_custody_seal = original_replay
        adapter._decoded_raw_rgb_prefix_sha256 = original_decode


def _identity(lock_payload: dict[str, Any], case_id: str) -> dict[str, Any]:
    matches = []
    for stratum, object_records in lock_payload["cohort"].items():
        for object_record in object_records:
            for episode in object_record["episodes"]:
                if episode["case_id"] == case_id:
                    matches.append(
                        {
                            "case_id": case_id,
                            "stratum": stratum,
                            "object_id": object_record["object_id"],
                            "episode_id": int(episode["episode_id"]),
                        }
                    )
    assert len(matches) == 1
    return matches[0]


def _prediction_arrays(
    *,
    persistence_only: bool = False,
) -> dict[str, np.ndarray]:
    frame = np.arange(76, dtype=np.float32)[:, None, None]
    point = np.arange(20, dtype=np.float32)[None, :, None]
    coordinate = np.arange(3, dtype=np.float32)[None, None, :]
    trajectory = (
        frame * np.float32(0.001)
        + point * np.float32(0.01)
        + coordinate * np.float32(0.1)
    )
    if persistence_only:
        trajectory = np.repeat(trajectory[:1], 76, axis=0)
    return {role: trajectory.copy() for role in seal.ARRAY_ROLES}


def _larger_permuted_official_target(
    sealed_target: np.ndarray,
) -> np.ndarray:
    point_count = sealed_target.shape[1]
    permutation = np.roll(
        np.arange(point_count, dtype=np.int64),
        3,
    )
    matched = sealed_target[:, permutation].copy()
    matched[0, :, 0] += np.float32(0.003)
    extras = np.empty(
        (sealed_target.shape[0], 4, 3),
        dtype=np.float32,
    )
    extras[:, :, 0] = np.float32(1.0) + np.arange(
        4,
        dtype=np.float32,
    )[None, :] * np.float32(0.02)
    extras[:, :, 1] = np.arange(
        sealed_target.shape[0],
        dtype=np.float32,
    )[:, None] * np.float32(0.0001)
    extras[:, :, 2] = np.float32(0.25)
    return np.concatenate((matched, extras), axis=1)


def _cameras() -> dict[int, list[str]]:
    cameras = [f"camera-{index:02d}" for index in range(8)]
    return {4: cameras[:4], 8: cameras}


def _routing(*, retained_failure: bool = False) -> dict[str, Any]:
    cameras = _cameras()[8]
    unreliable = {
        "valid_covariance_center_count": 0,
        "valid_covariance_center_ids": [],
        "normalized_covariance_dispersion": None,
        "reliable": False,
    }
    return {
        "protocol_id": ADAPTIVE_COVARIANCE_PROTOCOL_ID,
        "fallback": {
            "trajectory": ("persistence" if retained_failure else "physical_prior"),
            "rbf_state_update": False,
            "bit_exact": True,
        },
        "updates": [
            {
                "frame": frame,
                "stop_frame_exclusive": stop,
                "route": "physical_prior_fallback",
                "selected_camera_budget": None,
                "tracked_camera_count": 8,
                "tracked_cameras": cameras,
                "rbf_correction_applied": False,
                "state_updated": False,
                "selected_backbone": (
                    "persistence" if retained_failure else "physical_prior"
                ),
                **(
                    {
                        "camera_streams_charged_as_attempted": True,
                        "dynamic_observation_available": False,
                        "tracker_inference_executed": False,
                    }
                    if retained_failure
                    else {}
                ),
                "budget_diagnostics": {
                    "4": dict(unreliable),
                    "8": dict(unreliable),
                },
            }
            for frame, stop in zip((19, 38, 57), (38, 57, 76), strict=True)
        ],
    }


def _disposition(
    nested_measurement_dir: Path,
    *,
    retained_failure: bool = False,
) -> dict[str, Any]:
    manifest_path = nested_measurement_dir / measurement.MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_bytes())
    value = {
        "status": "prediction_complete",
        "case_retained": True,
        "disposition_based_on_target_or_outcome": False,
        "center_ids": list(range(16)),
        "causal_input_hashes": {
            "nested_measurement_manifest": {
                "file_sha256": _sha256(manifest_path),
                "artifact_sha256": manifest["artifact_sha256"],
            },
        },
        "notes": "target-free outcome-adapter fixture",
    }
    if retained_failure:
        value.update(
            {
                "status": "retained_technical_failure",
                "failure_code": "prediction_runtime_failure",
                "fallback_label": "persistence_only",
            }
        )
    return value


def _measurement_arrays(
    *,
    retained_failure: bool = False,
) -> dict[str, np.ndarray]:
    return {
        "measurement_m": np.zeros((76, 20, 3), dtype=np.float32),
        "measurement_validity": np.full(
            (76, 20),
            not retained_failure,
            dtype=bool,
        ),
        "measurement_covariance_m2": np.zeros(
            (76, 20, 3, 3),
            dtype=np.float32,
        ),
        "measurement_covariance_valid": np.full(
            (76, 20),
            not retained_failure,
            dtype=bool,
        ),
    }


def _build_identity_persistence_staged_case(
    root: Path,
    *,
    lock_payload: dict[str, Any],
    lock_path: Path,
    h1: str,
    h2: str,
    case_id: str,
) -> tuple[Path, Path]:
    staged = root / case_id
    staged.mkdir(parents=True)
    identity = _identity(lock_payload, case_id)
    external = {
        "case": case_id,
        "object_id": identity["object_id"],
        "episode_id": identity["episode_id"],
        "episode_key": f"{identity['object_id']}/{identity['episode_id']}",
        "stratum": identity["stratum"],
        "role": "calibration",
    }
    known_action = staged / "known-action" / "robot.npz"
    known_action.parent.mkdir()
    np.savez_compressed(known_action, actions=np.zeros((81, 1), dtype=np.float32))
    prefix: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360BiasAwarePredictionPrefix",
        "protocol_id": lock.PROTOCOL_ID,
        "protocol_config_sha256": lock_payload["artifact_sha256"],
        **external,
        "inputs_sha256": {
            "protocol": _sha256(lock_path),
            "source_preparation_manifest": "9" * 64,
        },
        "staged_robot_sha256": {"known_action": _sha256(known_action)},
        "information_boundary": {
            "source_object_frames_after_prefix_read": False,
            "future_dense_reconstruction_read": False,
            "future_particle_tracks_read": False,
            "target_metric_read": False,
        },
    }
    prefix["result_sha256"] = adapter._result_sha256(prefix)
    prefix_path = staged / failure.PREDICTION_PREFIX_MANIFEST_FILENAME
    _write_json(prefix_path, prefix)

    points = _prediction_arrays(persistence_only=True)["physical_prior_m"][0]
    colors = np.zeros_like(points)
    geometry_path = staged / failure.FRAME_ZERO_ARCHIVE_FILENAME
    np.savez_compressed(geometry_path, points_m=points, colors=colors)
    splat_path = staged / failure.FRAME_ZERO_SPLAT_RELATIVE_PATH
    splat_path.parent.mkdir(parents=True)
    splat_path.write_bytes(b"sealed-splat-fixture")
    fallback_diagnostics = {
        "source": "strict-multiview-visual-hull-surface",
        "reason": "fixture physical fallback",
    }
    material_hash = adapter._external_array_sha256(points)
    marker = {
        "schema_version": 1,
        "artifact_kind": measurement.IDENTITY_PERSISTENCE_ADAPTER_KIND,
        "policy": measurement.IDENTITY_PERSISTENCE_POLICY,
        "implementation_commit_h1": h1,
        "cohort_lock_commit_h2": h2,
        "cohort_lock_artifact_sha256": lock_payload["artifact_sha256"],
        "adapter_source_sha256": _sha256(Path(failure.__file__)),
        "deform360_revision": runtime.DEFORM360_EXECUTION_COMMIT,
        "pcd_stage_source_sha256": failure.PCD_STAGE_SOURCE_SHA256,
        "frame_zero_splat_file_sha256": _sha256(splat_path),
        "seed_parameters": {
            "crop_half_extent_m": 0.5,
            "seed_count": 10000,
            "rng_seed": 0,
        },
        "previous_material": {
            "source": "strict-multiview-visual-hull-surface",
            "point_count": 20,
            "array_sha256": "2" * 64,
            "file_sha256": "3" * 64,
        },
        "adapted_material": {
            "source": measurement.IDENTITY_PERSISTENCE_POLICY,
            "point_count": 20,
            "array_sha256": material_hash,
            "file_sha256": _sha256(geometry_path),
        },
        "preserved_fallback_diagnostics_sha256": hashlib.sha256(
            adapter._canonical_bytes(fallback_diagnostics)
        ).hexdigest(),
        "physical_twin_admitted": False,
    }
    frame_zero: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360BiasAwareFrameZeroReconstruction",
        "protocol_id": lock.PROTOCOL_ID,
        "protocol_config_sha256": lock_payload["artifact_sha256"],
        **external,
        "material_point_source": measurement.IDENTITY_PERSISTENCE_POLICY,
        "physical_policy": "persistence_only",
        "material_point_count": 20,
        "material_identity_sha256": material_hash,
        "fallback_diagnostics": fallback_diagnostics,
        measurement.IDENTITY_PERSISTENCE_ADAPTER_KEY: marker,
        "inputs_sha256": {
            "prediction_prefix_manifest": _sha256(prefix_path),
        },
        "outputs_sha256": {
            "frame_zero_points": _sha256(geometry_path),
            "frame_zero_splat": _sha256(splat_path),
        },
        "information_boundary": {
            "object_observation_frames_used": [0],
            "future_object_rgb_read": False,
            "future_dense_reconstruction_read": False,
            "future_particle_tracks_read": False,
            "target_metric_read": False,
        },
    }
    frame_zero["result_sha256"] = adapter._result_sha256(frame_zero)
    _write_json(
        staged / failure.FRAME_ZERO_MANIFEST_FILENAME,
        frame_zero,
    )
    processed = staged / failure.PROCESSED_PREFIX_RELATIVE_PATH
    processed.mkdir(parents=True)
    np.save(processed / "undistorted_intrinsics.npy", np.eye(3))
    np.save(processed / "extrinsics.npy", np.eye(4))
    return staged, processed


def _build_nested_measurement(
    case_root: Path,
    *,
    identity: dict[str, Any],
    binding: dict[str, str],
    retained_failure: bool = False,
    staged_case: Path | None = None,
    processed_prefix: Path | None = None,
) -> None:
    case_root.mkdir(parents=True)
    arrays = _measurement_arrays(retained_failure=retained_failure)
    outputs = {
        str(budget): measurement._write_budget_archives(
            case_root / f"budget-{budget}",
            arrays=arrays,
            centers=np.arange(16, dtype=np.int64),
            selected_cameras=_cameras()[budget],
        )
        for budget in (4, 8)
    }
    dummy_sha = "1" * 64
    if retained_failure:
        assert staged_case is not None and processed_prefix is not None
        prefix_path = staged_case / failure.PREDICTION_PREFIX_MANIFEST_FILENAME
        frame_zero_path = staged_case / failure.FRAME_ZERO_MANIFEST_FILENAME
        prefix = json.loads(prefix_path.read_bytes())
        frame_zero = json.loads(frame_zero_path.read_bytes())
        retained_source = {
            "failure_code": "prediction_runtime_failure",
            "prediction_prefix_manifest": {
                "path": str(prefix_path),
                "file_sha256": _sha256(prefix_path),
                "result_sha256": prefix["result_sha256"],
            },
            "frame_zero_manifest": {
                "path": str(frame_zero_path),
                "file_sha256": _sha256(frame_zero_path),
                "result_sha256": frame_zero["result_sha256"],
            },
            "processed_prefix_episode": {
                "path": str(processed_prefix),
                "intrinsics_file_sha256": _sha256(
                    processed_prefix / "undistorted_intrinsics.npy"
                ),
                "extrinsics_file_sha256": _sha256(processed_prefix / "extrinsics.npy"),
            },
            "dynamic_point_observations_available": False,
        }
        inputs = {
            "physical_backbone": {
                "external_backbone_seal_file_sha256": dummy_sha,
                "external_backbone_seal_result_sha256": dummy_sha,
                "external_physical_manifest_file_sha256": dummy_sha,
                "external_physical_manifest_result_sha256": dummy_sha,
                "physical_archive_file_sha256": dummy_sha,
                "physical_archive_array_sha256": {
                    role: dummy_sha
                    for role in measurement.EXTERNAL_PHYSICAL_ARRAY_ROLES
                },
            },
            "physical_archive": {
                "sha256": dummy_sha,
                "frame_zero_array_sha256": frame_zero["material_identity_sha256"],
            },
            "intrinsics_sha256": retained_source["processed_prefix_episode"][
                "intrinsics_file_sha256"
            ],
            "extrinsics_sha256": retained_source["processed_prefix_episode"][
                "extrinsics_file_sha256"
            ],
            "selected_camera_prefixes_and_frame_zero": {},
            "source_stage_lineage": {
                "prediction_prefix_manifest": dict(
                    retained_source["prediction_prefix_manifest"]
                ),
                "frame_zero_manifest": dict(retained_source["frame_zero_manifest"]),
                "source_preparation_manifest_file_sha256": prefix["inputs_sha256"][
                    "source_preparation_manifest"
                ],
            },
            "retained_failure_source": retained_source,
        }
        tracker = {
            "name": "AllTracker",
            "molmomotion_revision": measurement.ALLTRACKER_MOLMOMOTION_REVISION,
            "source_tree": measurement.ALLTRACKER_SOURCE_TREE,
            "runtime_source_sha256": (raw_observation.ALLTRACKER_RUNTIME_SOURCE_SHA256),
            "checkpoint_sha256": (raw_observation.ALLTRACKER_CHECKPOINT_SHA256),
            "device": "not-executed",
            "execution_status": measurement.RETAINED_MEASUREMENT_FAILURE_STATUS,
            "failure_code": "prediction_runtime_failure",
            "inference_executed": False,
        }
        updates = []
        for frame in (19, 38, 57):
            reliability = {
                "frame": frame,
                "valid_covariance_center_count": 0,
                "valid_covariance_center_ids": [],
                "covariance_quantile": 0.9,
                "radial_standard_deviation_quantile_m": None,
                "frame_zero_bbox_diagonal_m": 1.0,
                "normalized_covariance_dispersion": None,
                "probabilistic_calibration_claimed": False,
                "reliable": False,
            }
            tracker_records = [
                {
                    "prefix_frame_range_half_open": [0, frame + 1],
                    "maximum_video_frame_read": frame,
                    "decoded_frame_count": frame + 1,
                    "decoded_rgb_prefix_sha256": dummy_sha,
                    "original_image_shape": [2, 2],
                    "camera": camera,
                    "query_ids": list(range(16)),
                    "execution_role": (
                        "adaptive_first_four"
                        if index < 4
                        else "adaptive_eight_escalation"
                    ),
                    "execution_index_within_update": index,
                    "four_view_decision_already_materialized": index >= 4,
                    "camera_stream_attempted": True,
                    "tracker_inference_executed": False,
                    "dynamic_observation_available": False,
                    "failure_code": "prediction_runtime_failure",
                }
                for index, camera in enumerate(_cameras()[8])
            ]
            center_records = {
                str(budget): [
                    {
                        "center_id": center_id,
                        "measurement_available": False,
                        "covariance_valid": False,
                        "decision": (
                            "retained_technical_failure_measurement_unavailable"
                        ),
                        "failure_code": "prediction_runtime_failure",
                    }
                    for center_id in range(16)
                ]
                for budget in (4, 8)
            }
            updates.append(
                {
                    "frame": frame,
                    "four_view_decision_materialized_before_shadow_extra_four": True,
                    "four_view_reliable_before_shadow": False,
                    "offline_shadow_extra_four_tracked": False,
                    "adaptive_route": "physical_prior_fallback",
                    "adaptive_charged_camera_streams": 8,
                    "budget_reliability": {
                        "4": dict(reliability),
                        "8": dict(reliability),
                    },
                    "tracker": tracker_records,
                    "centers": center_records,
                }
            )
        camera_accounting = dict(measurement.RETAINED_FAILURE_CAMERA_ACCOUNTING)
    else:
        staged = case_root.parent.parent / "standard-staged" / identity["case_id"]
        inputs = {
            "physical_backbone": {
                "external_backbone_seal_file_sha256": dummy_sha,
                "external_backbone_seal_result_sha256": dummy_sha,
                "external_physical_manifest_file_sha256": dummy_sha,
                "external_physical_manifest_result_sha256": dummy_sha,
                "physical_archive_file_sha256": dummy_sha,
                "physical_archive_array_sha256": {
                    role: dummy_sha
                    for role in measurement.EXTERNAL_PHYSICAL_ARRAY_ROLES
                },
            },
            "physical_archive": {
                "sha256": dummy_sha,
                "frame_zero_array_sha256": dummy_sha,
            },
            "intrinsics_sha256": dummy_sha,
            "extrinsics_sha256": dummy_sha,
            "selected_camera_prefixes_and_frame_zero": {},
            "source_stage_lineage": {
                "prediction_prefix_manifest": {
                    "path": str(staged / failure.PREDICTION_PREFIX_MANIFEST_FILENAME),
                    "file_sha256": "2" * 64,
                    "result_sha256": "3" * 64,
                },
                "frame_zero_manifest": {
                    "path": str(staged / failure.FRAME_ZERO_MANIFEST_FILENAME),
                    "file_sha256": "4" * 64,
                    "result_sha256": "5" * 64,
                },
                "source_preparation_manifest_file_sha256": "6" * 64,
            },
        }
        tracker = {"name": "fixture"}
        updates = []
        camera_accounting = {
            "adaptive_charge_is_causal_offline_policy_demand": True,
            "all_eight_streams_eventually_tracked_for_fixed8_shadow": True,
            "realized_acquisition_or_wall_clock_saving_claimed": False,
            "frame_zero_all_camera_planning_excluded": True,
        }
    lineage = inputs["source_stage_lineage"]
    custody_path = (
        case_root.parent.parent
        / "synthetic-source-custody"
        / f"{identity['case_id']}.json"
    )
    custody_path.parent.mkdir(parents=True, exist_ok=True)
    source_episode_dir = (
        case_root.parent.parent
        / "synthetic-source"
        / identity["object_id"]
        / f"episode_{identity['episode_id']:04d}"
    )
    staged_case_dir = Path(lineage["prediction_prefix_manifest"]["path"]).parent
    source_robot_payload = f"{identity['case_id']}:authorized-future:robot\n".encode()
    source_robot_sha256 = _write_fixture_artifact(
        source_episode_dir / "robot" / "robot.npz",
        source_robot_payload,
    )
    raw_rgb_by_camera: dict[str, str] = {}
    for camera in _cameras()[8]:
        staged_video_payload = (
            f"{identity['case_id']}:{camera}:authorized-future:video\n".encode()
        )
        _write_fixture_artifact(
            staged_case_dir / "prefix" / "episode_0000" / camera / "undistorted.mp4",
            staged_video_payload,
        )
        raw_rgb_by_camera[camera] = _synthetic_decoded_payload_sha256(
            staged_video_payload,
            frame_count=58,
        )
    if retained_failure:
        staged_splat_path = staged_case_dir / failure.FRAME_ZERO_SPLAT_RELATIVE_PATH
        staged_splat_sha256 = _sha256(staged_splat_path)
    else:
        staged_splat_sha256 = _write_fixture_artifact(
            staged_case_dir / failure.FRAME_ZERO_SPLAT_RELATIVE_PATH,
            f"{identity['case_id']}:authorized-future:frame-zero-splat\n".encode(),
        )
    custody = {
        "artifact_sha256": "7" * 64,
        "camera_panel": _cameras()[8],
        "manifests": {
            "source_preparation": {
                "file_sha256": lineage["source_preparation_manifest_file_sha256"],
            },
            "prediction_prefix": {
                "file_sha256": lineage["prediction_prefix_manifest"]["file_sha256"],
                "result_sha256": lineage["prediction_prefix_manifest"]["result_sha256"],
            },
            "frame_zero": {
                "file_sha256": lineage["frame_zero_manifest"]["file_sha256"],
                "result_sha256": lineage["frame_zero_manifest"]["result_sha256"],
            },
        },
        "path_binding": {
            "source_episode_dir": str(source_episode_dir),
            "staged_case_dir": str(staged_case_dir),
        },
        "raw_rgb24_prefix": {
            "by_camera": raw_rgb_by_camera,
        },
        "camera_custody": {
            camera: {
                "decoded_rgb24_prefix_sha256": raw_rgb_by_camera[camera],
                "source_prefix_frame_range_half_open": [0, 58],
            }
            for camera in _cameras()[8]
        },
        "frame_zero_custody": {
            "splat_file_sha256": staged_splat_sha256,
        },
        "inventories": {
            "aligned_source_episode": {
                "records": [
                    {
                        "path": "robot/robot.npz",
                        "type": "file",
                        "size_bytes": len(source_robot_payload),
                        "sha256": source_robot_sha256,
                    }
                ],
            },
        },
    }
    _write_json(custody_path, custody)
    lineage["source_custody_seal"] = {
        "path": str(custody_path),
        "file_sha256": _sha256(custody_path),
        "artifact_sha256": custody["artifact_sha256"],
    }
    payload: dict[str, Any] = {
        "schema_version": measurement.SCHEMA_VERSION,
        "artifact_kind": measurement.ARTIFACT_KIND,
        "protocol_id": lock.PROTOCOL_ID,
        "case_identity": identity,
        "lock_binding": binding,
        "config": {"fixture": "target-free"},
        "plan": {
            "candidate_ids": list(range(20)),
            "center_ids": list(range(16)),
            "camera_activation_order": _cameras()[8],
            "selected_cameras_by_budget": {
                "4": _cameras()[4],
                "8": _cameras()[8],
            },
            "selection_score": {"4": [0.0], "8": [0.0]},
        },
        "inputs": inputs,
        "tracker": tracker,
        "updates": updates,
        "outputs": outputs,
        "camera_accounting": camera_accounting,
        "information_boundary": {
            "target_path_argument_accepted": False,
            "outcome_path_argument_accepted": False,
            "target_metric_or_outcome_score_computed": False,
            "future_geometry_read": False,
            "video_prefix_rule": "update u reads exactly frames [0,u]",
            "maximum_video_frame_read_by_update": [19, 38, 57],
            "four_view_decision_precedes_shadow_extra_four": True,
        },
    }
    payload["artifact_sha256"] = measurement._canonical_sha256(payload)
    _write_json(
        case_root / measurement.MANIFEST_FILENAME,
        payload,
    )


@pytest.fixture(scope="module")
def complete_confirmation(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("confirmation-outcome-adapter")
    repository = root / "adapter"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "fixture@example.invalid")
    _git(repository, "config", "user.name", "Fixture")
    (repository / "README.md").write_text("implementation H1\n", encoding="utf-8")
    _git(repository, "add", "README.md")
    _git(repository, "commit", "-q", "-m", "implementation")
    h1 = _git(repository, "rev-parse", "HEAD")

    lock_path = repository / runtime.COHORT_LOCK_REPOSITORY_PATH
    lock.write_confirmation_cohort_lock(lock_path, h1)
    _git(repository, "add", runtime.COHORT_LOCK_REPOSITORY_PATH)
    _git(repository, "commit", "-q", "-m", "freeze cohort")
    h2 = _git(repository, "rev-parse", "HEAD")
    lock_payload = lock.load_confirmation_cohort_lock(
        lock_path,
        expected_implementation_commit_h1=h1,
    )
    lock_file_sha256 = _sha256(lock_path)
    lock_binding = {
        "implementation_commit_h1": h1,
        "cohort_lock_commit_h2": h2,
        "cohort_lock_artifact_sha256": lock_payload["artifact_sha256"],
        "cohort_lock_file_sha256": lock_file_sha256,
    }

    case_dirs: dict[str, Path] = {}
    measurement_dirs: dict[str, Path] = {}
    retained_case_id = lock_payload["selected_case_ids"][-1]
    retained_staged, retained_processed = _build_identity_persistence_staged_case(
        root / "staged",
        lock_payload=lock_payload,
        lock_path=lock_path,
        h1=h1,
        h2=h2,
        case_id=retained_case_id,
    )
    for case_id in lock_payload["selected_case_ids"]:
        retained = case_id == retained_case_id
        nested_dir = root / "nested-measurements" / case_id
        _build_nested_measurement(
            nested_dir,
            identity=_identity(lock_payload, case_id),
            binding=lock_binding,
            retained_failure=retained,
            staged_case=retained_staged if retained else None,
            processed_prefix=retained_processed if retained else None,
        )
        measurement_dirs[case_id] = nested_dir
        case_dir = root / "case-seals" / case_id
        seal.seal_confirmation_case(
            lock_path,
            h2,
            case_id,
            case_dir,
            _prediction_arrays(persistence_only=retained),
            _cameras(),
            _routing(retained_failure=retained),
            _disposition(
                nested_dir,
                retained_failure=retained,
            ),
            expected_h1=h1,
        )
        case_dirs[case_id] = case_dir

    barrier_path = root / "barriers" / "prediction-barrier.json"
    seal.create_confirmation_prediction_barrier(
        barrier_path,
        lock_path,
        h2,
        case_dirs,
        expected_h1=h1,
    )
    compatibility_root = root / "compatibility"
    adapter.build_confirmation_outcome_compatibility(
        repository,
        lock_path,
        h2,
        barrier_path,
        case_dirs,
        measurement_dirs,
        compatibility_root,
        expected_h1=h1,
    )
    assert not _git(repository, "status", "--porcelain", "--untracked-files=all")
    return {
        "root": root,
        "repository": repository,
        "h1": h1,
        "h2": h2,
        "lock_path": lock_path,
        "lock": lock_payload,
        "case_dirs": case_dirs,
        "measurement_dirs": measurement_dirs,
        "barrier_path": barrier_path,
        "compatibility_root": compatibility_root,
        "retained_case_id": retained_case_id,
        "retained_staged": retained_staged,
    }


def _validate_compatibility(fixture: dict[str, Any]):
    return adapter.validate_confirmation_outcome_compatibility(
        fixture["repository"],
        fixture["lock_path"],
        fixture["h2"],
        fixture["barrier_path"],
        fixture["case_dirs"],
        fixture["measurement_dirs"],
        fixture["compatibility_root"],
        expected_h1=fixture["h1"],
    )


def test_target_free_compatibility_replays_all_34_cases_and_authorizes_exact_root(
    complete_confirmation: dict[str, Any],
) -> None:
    parameters = inspect.signature(
        adapter.build_confirmation_outcome_compatibility
    ).parameters
    assert all(
        token not in name
        for name in parameters
        for token in ("target", "outcome", "metric", "score")
    )
    validated = _validate_compatibility(complete_confirmation)
    manifest = validated.manifest
    assert manifest["case_count"] == 34
    assert (
        manifest["exact_case_ids"] == complete_confirmation["lock"]["selected_case_ids"]
    )
    assert (
        manifest["lock_binding"]["implementation_commit_h1"]
        == (complete_confirmation["h1"])
    )
    assert (
        manifest["lock_binding"]["cohort_lock_commit_h2"]
        == (complete_confirmation["h2"])
    )
    assert manifest["information_boundary"]["target_array_read"] is False

    first = manifest["cases"][0]
    case_id = first["case"]
    copied = (
        validated.measurement_root / case_id / measurement.MEASUREMENT_ARCHIVE_FILENAME
    )
    source = (
        complete_confirmation["measurement_dirs"][case_id]
        / "budget-8"
        / measurement.MEASUREMENT_ARCHIVE_FILENAME
    )
    assert copied.read_bytes() == source.read_bytes()
    assert first["selected_cameras"] == _cameras()[8]

    authorizer = adapter.make_confirmation_outcome_authorizer(
        complete_confirmation["repository"],
        complete_confirmation["lock_path"],
        complete_confirmation["h2"],
        complete_confirmation["barrier_path"],
        complete_confirmation["case_dirs"],
        complete_confirmation["measurement_dirs"],
        complete_confirmation["compatibility_root"],
        expected_h1=complete_confirmation["h1"],
    )
    record, compatibility_seal = authorizer(
        manifest,
        protocol_path=complete_confirmation["lock_path"],
        role="calibration",
        artifact_root=validated.prediction_root,
        object_id=first["object_id"],
        episode_id=first["episode_id"],
    )
    assert record["case"] == case_id
    assert record["role"] == "calibration"
    assert compatibility_seal["source_case_seal"] == first["case_seal"]
    with pytest.raises(ValueError, match="another H2 authorization binding"):
        authorizer(
            manifest,
            protocol_path=complete_confirmation["lock_path"],
            role="calibration",
            artifact_root=validated.measurement_root,
            object_id=first["object_id"],
            episode_id=first["episode_id"],
        )


def test_compatibility_rejects_regenerated_same_case_source_lineage(
    complete_confirmation: dict[str, Any],
) -> None:
    case_id = complete_confirmation["lock"]["selected_case_ids"][0]
    manifest_path = (
        complete_confirmation["measurement_dirs"][case_id]
        / measurement.MANIFEST_FILENAME
    )
    original = manifest_path.read_bytes()
    payload = json.loads(original)
    payload["inputs"]["source_stage_lineage"][
        "source_preparation_manifest_file_sha256"
    ] = "a" * 64
    payload["artifact_sha256"] = measurement._canonical_sha256(
        {key: value for key, value in payload.items() if key != "artifact_sha256"}
    )
    _write_json(manifest_path, payload)
    try:
        with pytest.raises(ValueError, match="sealed prediction input"):
            _validate_compatibility(complete_confirmation)
    finally:
        manifest_path.write_bytes(original)
    _validate_compatibility(complete_confirmation)


def _write_synthetic_authorized_target(
    fixture: dict[str, Any],
    case_id: str,
    *,
    larger_permuted_official_target: bool = False,
) -> tuple[Path, Path]:
    compatibility = _validate_compatibility(fixture)
    case = next(
        record
        for record in compatibility.manifest["cases"]
        if record["case"] == case_id
    )
    external_record = {
        key: case[key]
        for key in (
            "case",
            "object_id",
            "episode_id",
            "episode_key",
            "stratum",
            "role",
        )
    }
    dummy_sha = "1" * 64
    retained_source = case["nested_measurement"].get("retained_failure_source")
    source_stage_lineage = case["nested_measurement"]["source_stage_lineage"]
    identity_persistence = (
        retained_source.get("identity_persistence_adapter")
        if isinstance(retained_source, dict)
        else None
    )
    future_root = fixture["root"] / "authorized-futures" / case_id
    future_root.mkdir(parents=True, exist_ok=False)
    future_episode = future_root / "episode_0000"
    robot_payload = f"{case_id}:authorized-future:robot\n".encode()
    robot_sha256 = _write_fixture_artifact(
        future_episode / "robot" / "robot.npz",
        robot_payload,
    )
    if identity_persistence is None:
        frame_zero_splat_payload = (
            f"{case_id}:authorized-future:frame-zero-splat\n".encode()
        )
    else:
        sealed_splat_path = (
            Path(source_stage_lineage["frame_zero_manifest"]["path"]).parent
            / failure.FRAME_ZERO_SPLAT_RELATIVE_PATH
        )
        frame_zero_splat_payload = sealed_splat_path.read_bytes()
    frame_zero_splat_sha256 = _write_fixture_artifact(
        future_episode / "splatfacto" / "splat_0.ply",
        frame_zero_splat_payload,
    )
    intrinsics_sha256 = _write_fixture_artifact(
        future_episode / "undistorted_intrinsics.npy",
        f"{case_id}:authorized-future:intrinsics\n".encode(),
    )
    extrinsics_sha256 = _write_fixture_artifact(
        future_episode / "extrinsics.npy",
        f"{case_id}:authorized-future:extrinsics\n".encode(),
    )
    camera_records = []
    for camera in case["selected_cameras"]:
        camera_root = future_episode / camera
        video_path = camera_root / "undistorted.mp4"
        video_sha256 = _write_fixture_artifact(
            video_path,
            f"{case_id}:{camera}:authorized-future:video\n".encode(),
        )
        timestamps_sha256 = _write_fixture_artifact(
            camera_root / "aligned_timestamps.txt",
            b"".join(f"{index / 30.0:.9f}\n".encode() for index in range(81)),
        )
        masks_sha256 = _write_fixture_artifact(
            camera_root / "mask_refined.h5",
            f"{case_id}:{camera}:authorized-future:masks\n".encode(),
        )
        camera_records.append(
            {
                "camera": camera,
                "video_sha256": video_sha256,
                "decoded_sealed_prefix_sha256": (
                    _synthetic_decoded_prefix_sha256(
                        video_path,
                        frame_count=58,
                    )
                ),
                "timestamps_sha256": timestamps_sha256,
                "masks_sha256": masks_sha256,
                "sam2_diagnostics": {},
            }
        )
    compatibility_manifest_sha = _sha256(compatibility.manifest_path)
    lock_sha = _sha256(fixture["lock_path"])
    future: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": adapter.EXTERNAL_AUTHORIZED_FUTURE_KIND,
        "protocol_id": lock.PROTOCOL_ID,
        "protocol_config_sha256": fixture["lock"]["artifact_sha256"],
        **external_record,
        "code_revision": runtime.EXTERNAL_EXECUTION_COMMIT,
        "raw_frame_range_half_open": [0, 81],
        "frame_count": 81,
        "selected_cameras": case["selected_cameras"],
        "camera_records": camera_records,
        "inputs_sha256": {
            "protocol": lock_sha,
            "prediction_cohort_seal": compatibility_manifest_sha,
            "prediction_seal": case["compatibility_prediction"]["seal_file_sha256"],
            "prediction_archive": case["compatibility_prediction"][
                "archive_file_sha256"
            ],
            "prediction_prefix_manifest": source_stage_lineage[
                "prediction_prefix_manifest"
            ]["file_sha256"],
            "source_preparation_manifest": source_stage_lineage[
                "source_preparation_manifest_file_sha256"
            ],
            "frame_zero_reconstruction_manifest": source_stage_lineage[
                "frame_zero_manifest"
            ]["file_sha256"],
            "source_robot": robot_sha256,
            "measurement_archive": case["compatibility_measurement"]["file_sha256"],
            "generic_selector_source": (
                "79b161fa66489f75b5b078c7ae409387feed74c51a38b86e89800d0aa578b1df"
            ),
            "sam2_checkpoint": (
                "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38"
            ),
            "calibration_gate": None,
        },
        "outputs_sha256": {
            "robot": robot_sha256,
            "frame_zero_splat": frame_zero_splat_sha256,
            "intrinsics": intrinsics_sha256,
            "extrinsics": extrinsics_sha256,
        },
        "authorization": {
            "prediction_cohort_result_sha256": compatibility.manifest["result_sha256"],
            "prediction_result_sha256": case["compatibility_prediction"][
                "seal_result_sha256"
            ],
            "calibration_gate_result_sha256": None,
            "prediction_cohort_verified_before_future_read": True,
            "target_access_gate_verified": False,
        },
        "information_boundary": {
            "future_rgb_read_after_cohort_authorization": True,
            "future_masks_created_after_cohort_authorization": True,
            "future_dense_reconstruction_created": False,
            "future_particle_tracks_created": False,
            "target_metric_computed": False,
            "future_tactile_read": False,
        },
    }
    future["result_sha256"] = adapter._result_sha256(future)
    future_manifest = future_root / adapter.EXTERNAL_AUTHORIZED_FUTURE_MANIFEST_FILENAME
    _write_json(future_manifest, future)

    prediction_path = (
        compatibility.prediction_root
        / case_id
        / adapter.EXTERNAL_PREDICTION_ARCHIVE_FILENAME
    )
    with np.load(prediction_path, allow_pickle=False) as stored:
        sealed_target = np.asarray(
            stored["selected_raw_backbone"],
            dtype=np.float32,
        )
    target = (
        _larger_permuted_official_target(sealed_target)
        if larger_permuted_official_target
        else sealed_target
    )
    visibility = np.ones(target.shape[:2], dtype=bool)
    validity = np.ones(target.shape[:2], dtype=bool)
    outcome_root = fixture["root"] / "authorized-outcomes" / case_id
    outcome_root.mkdir(parents=True, exist_ok=False)
    archive_path = outcome_root / adapter.EXTERNAL_TARGET_ARCHIVE_FILENAME
    np.savez_compressed(
        archive_path,
        target_m=target,
        target_visibility=visibility,
        target_validity=validity,
    )
    reconstruction_metadata_sha256 = _write_fixture_artifact(
        future_episode / "splatfacto" / "splatfacto.meta.json",
        f"{case_id}:authorized-outcome:reconstruction-metadata\n".encode(),
    )
    point_cloud_metadata_sha256 = _write_fixture_artifact(
        future_episode / "pcd_clean" / "pcd_clean.meta.json",
        f"{case_id}:authorized-outcome:point-cloud-metadata\n".encode(),
    )
    gripper_metadata_sha256: dict[str, str] = {}
    depth_metadata_sha256: dict[str, str] = {}
    tracking_metadata_sha256: dict[str, str] = {}
    for camera in case["selected_cameras"]:
        camera_root = future_episode / camera
        gripper_metadata_sha256[camera] = _write_fixture_artifact(
            camera_root / "rendered_urdf.meta.json",
            f"{case_id}:{camera}:authorized-outcome:gripper-metadata\n".encode(),
        )
        depth_metadata_sha256[camera] = _write_fixture_artifact(
            camera_root / "rendered_depth.meta.json",
            f"{case_id}:{camera}:authorized-outcome:depth-metadata\n".encode(),
        )
        tracking_metadata_sha256[camera] = _write_fixture_artifact(
            camera_root / "tracking" / "tracking.meta.json",
            f"{case_id}:{camera}:authorized-outcome:tracking-metadata\n".encode(),
        )
    outcome: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": adapter.EXTERNAL_AUTHORIZED_OUTCOME_KIND,
        "protocol_id": lock.PROTOCOL_ID,
        "protocol_config_sha256": fixture["lock"]["artifact_sha256"],
        **external_record,
        "code_revision": runtime.EXTERNAL_EXECUTION_COMMIT,
        "deform360_revision": runtime.DEFORM360_EXECUTION_COMMIT,
        "cameras": case["selected_cameras"],
        "raw_frame_count": 81,
        "target_frame_count": 76,
        "material_point_count": int(target.shape[1]),
        "material_identity_sha256": adapter._external_array_sha256(target[0]),
        "reconstruction": {
            "minimum_visual_hull_points": 512,
            "voxel_resolution": 120,
            "cube_half_extent_m": 0.5,
            "first_frame_iterations": 500,
            "warm_start_iterations": 250,
            "sealed_frame_zero_splat_reused": True,
        },
        "inputs_sha256": {
            "protocol": lock_sha,
            "prediction_cohort_seal": compatibility_manifest_sha,
            "prediction_seal": case["compatibility_prediction"]["seal_file_sha256"],
            "prediction_archive": case["compatibility_prediction"][
                "archive_file_sha256"
            ],
            "authorized_future_manifest": _sha256(future_manifest),
            "tracking_checkpoint": dummy_sha,
            "cotracker_predictor": dummy_sha,
            "calibration_gate": None,
            "reconstruct_stage": dummy_sha,
            "urdf_render": dummy_sha,
            "depth_stage": dummy_sha,
            "tracking_stage": dummy_sha,
            "pcd_stage": dummy_sha,
        },
        "stage_metadata_sha256": {
            "reconstruction": reconstruction_metadata_sha256,
            "gripper_masks": gripper_metadata_sha256,
            "depth": depth_metadata_sha256,
            "tracking": tracking_metadata_sha256,
            "point_cloud": point_cloud_metadata_sha256,
        },
        "output": {
            "target_archive": str(archive_path),
            "target_archive_sha256": _sha256(archive_path),
            "target_array_sha256": adapter._external_array_sha256(target),
            "frame_zero_bit_exact_to_sealed_baseline": (
                target.shape == sealed_target.shape
                and np.array_equal(target[0], sealed_target[0])
            ),
        },
        "authorization": {
            "prediction_cohort_result_sha256": compatibility.manifest["result_sha256"],
            "prediction_result_sha256": case["compatibility_prediction"][
                "seal_result_sha256"
            ],
            "calibration_gate_result_sha256": None,
        },
        "information_boundary": {
            "prediction_cohort_verified_before_target_construction": True,
            "future_tactile_read": False,
            "prediction_metric_computed": False,
        },
    }
    outcome["result_sha256"] = adapter._result_sha256(outcome)
    _write_json(
        outcome_root / adapter.EXTERNAL_AUTHORIZED_OUTCOME_MANIFEST_FILENAME,
        outcome,
    )
    return future_root, outcome_root


def test_native_target_loader_returns_read_only_official_arrays_and_evidence(
    complete_confirmation: dict[str, Any],
) -> None:
    case_id = complete_confirmation["lock"]["selected_case_ids"][0]
    future_root, outcome_root = _write_synthetic_authorized_target(
        complete_confirmation,
        case_id,
    )
    validated = adapter.validate_confirmation_native_official_target(
        complete_confirmation["repository"],
        complete_confirmation["lock_path"],
        complete_confirmation["h2"],
        complete_confirmation["barrier_path"],
        complete_confirmation["case_dirs"],
        complete_confirmation["measurement_dirs"],
        complete_confirmation["compatibility_root"],
        case_id,
        future_root,
        outcome_root,
        expected_h1=complete_confirmation["h1"],
    )
    assert validated.target_m.shape == (76, 20, 3)
    assert validated.target_visibility.dtype == np.dtype(bool)
    assert validated.target_validity.dtype == np.dtype(bool)
    assert not validated.target_m.flags.writeable
    assert not validated.target_visibility.flags.writeable
    assert not validated.target_validity.flags.writeable
    assert (
        validated.evidence["lock_binding"]["implementation_commit_h1"]
        == (complete_confirmation["h1"])
    )
    assert (
        validated.evidence["lock_binding"]["cohort_lock_commit_h2"]
        == (complete_confirmation["h2"])
    )
    assert set(validated.evidence["target_archive"]["arrays"]) == {
        "target_m",
        "target_visibility",
        "target_validity",
    }

    arrays = adapter.load_confirmation_native_official_target(
        complete_confirmation["repository"],
        complete_confirmation["lock_path"],
        complete_confirmation["h2"],
        complete_confirmation["barrier_path"],
        complete_confirmation["case_dirs"],
        complete_confirmation["measurement_dirs"],
        complete_confirmation["compatibility_root"],
        case_id,
        future_root,
        outcome_root,
        expected_h1=complete_confirmation["h1"],
    )
    assert set(arrays) == {
        "target_m",
        "target_visibility",
        "target_validity",
    }
    assert all(not value.flags.writeable for value in arrays.values())


def test_production_validator_and_scoring_loader_transport_larger_native_target(
    complete_confirmation: dict[str, Any],
) -> None:
    case_id = complete_confirmation["lock"]["selected_case_ids"][2]
    future_root, outcome_root = _write_synthetic_authorized_target(
        complete_confirmation,
        case_id,
        larger_permuted_official_target=True,
    )
    all_cases = complete_confirmation["lock"]["selected_case_ids"]
    future_dirs = {
        selected: (
            future_root
            if selected == case_id
            else complete_confirmation["root"]
            / "unopened-authorized-futures"
            / selected
        )
        for selected in all_cases
    }
    outcome_dirs = {
        selected: (
            outcome_root
            if selected == case_id
            else complete_confirmation["root"]
            / "unopened-authorized-outcomes"
            / selected
        )
        for selected in all_cases
    }
    observed: list[adapter.ConfirmationNativeOfficialTarget] = []

    def production_validator(
        *args: Any,
        **kwargs: Any,
    ) -> adapter.ConfirmationNativeOfficialTarget:
        native = adapter.validate_confirmation_native_official_target(
            *args,
            **kwargs,
        )
        observed.append(native)
        return native

    callback = scoring.build_confirmation_case_target_loader(
        complete_confirmation["repository"],
        complete_confirmation["lock_path"],
        complete_confirmation["h2"],
        complete_confirmation["barrier_path"],
        complete_confirmation["case_dirs"],
        complete_confirmation["measurement_dirs"],
        complete_confirmation["compatibility_root"],
        future_dirs,
        outcome_dirs,
        expected_h1=complete_confirmation["h1"],
        native_target_validator=production_validator,
    )
    barrier = json.loads(complete_confirmation["barrier_path"].read_bytes())
    barrier_case = next(
        row for row in barrier["ordered_case_seals"] if row["case_id"] == case_id
    )
    result = callback(
        case_id,
        complete_confirmation["case_dirs"][case_id],
        barrier_case,
    )

    assert len(observed) == 1
    assert observed[0].target_m.shape == (76, 24, 3)
    assert observed[0].evidence["identity_persistence_adapter"] is None
    compatibility = _validate_compatibility(complete_confirmation)
    compatibility_case = next(
        row for row in compatibility.manifest["cases"] if row["case"] == case_id
    )
    assert (
        observed[0].evidence["nested_measurement"]["source_stage_lineage"]
        == compatibility_case["nested_measurement"]["source_stage_lineage"]
    )
    assert set(result["metrics"]) == {"adaptive", "fixed8", "fixed4"}
    outcome_path = outcome_root / adapter.EXTERNAL_AUTHORIZED_OUTCOME_MANIFEST_FILENAME
    original_outcome = outcome_path.read_bytes()
    outcome = json.loads(original_outcome)
    assert outcome["output"]["frame_zero_bit_exact_to_sealed_baseline"] is False
    outcome["output"]["frame_zero_bit_exact_to_sealed_baseline"] = True
    outcome["result_sha256"] = adapter._result_sha256(outcome)
    _write_json(outcome_path, outcome)
    try:
        with pytest.raises(
            ValueError,
            match="frame-zero identity declaration changed",
        ):
            adapter.validate_confirmation_native_official_target(
                complete_confirmation["repository"],
                complete_confirmation["lock_path"],
                complete_confirmation["h2"],
                complete_confirmation["barrier_path"],
                complete_confirmation["case_dirs"],
                complete_confirmation["measurement_dirs"],
                complete_confirmation["compatibility_root"],
                case_id,
                future_root,
                outcome_root,
                expected_h1=complete_confirmation["h1"],
            )
    finally:
        outcome_path.write_bytes(original_outcome)


def test_native_validator_rejects_regenerated_same_case_future_stage_sources(
    complete_confirmation: dict[str, Any],
) -> None:
    case_id = complete_confirmation["lock"]["selected_case_ids"][3]
    future_root, outcome_root = _write_synthetic_authorized_target(
        complete_confirmation,
        case_id,
    )
    future_path = future_root / adapter.EXTERNAL_AUTHORIZED_FUTURE_MANIFEST_FILENAME
    original = future_path.read_bytes()
    replacements = {
        "prediction_prefix_manifest": "a" * 64,
        "frame_zero_reconstruction_manifest": "b" * 64,
        "source_preparation_manifest": "c" * 64,
    }
    for input_role, replacement in replacements.items():
        future = json.loads(original)
        future["inputs_sha256"][input_role] = replacement
        future["result_sha256"] = adapter._result_sha256(future)
        _write_json(future_path, future)
        try:
            with pytest.raises(
                ValueError,
                match="sealed source-stage lineage",
            ):
                adapter.validate_confirmation_native_official_target(
                    complete_confirmation["repository"],
                    complete_confirmation["lock_path"],
                    complete_confirmation["h2"],
                    complete_confirmation["barrier_path"],
                    complete_confirmation["case_dirs"],
                    complete_confirmation["measurement_dirs"],
                    complete_confirmation["compatibility_root"],
                    case_id,
                    future_root,
                    outcome_root,
                    expected_h1=complete_confirmation["h1"],
                )
        finally:
            future_path.write_bytes(original)


def test_native_validator_replays_actual_future_and_stage_files(
    complete_confirmation: dict[str, Any],
) -> None:
    case_id = complete_confirmation["lock"]["selected_case_ids"][4]
    future_root, outcome_root = _write_synthetic_authorized_target(
        complete_confirmation,
        case_id,
    )
    compatibility = _validate_compatibility(complete_confirmation)
    case = next(
        record
        for record in compatibility.manifest["cases"]
        if record["case"] == case_id
    )
    first_camera = case["selected_cameras"][0]
    video_path = future_root / "episode_0000" / first_camera / "undistorted.mp4"
    original_video = video_path.read_bytes()
    video_path.write_bytes(original_video + b"post-manifest-video-change")
    try:
        with pytest.raises(ValueError, match="video file changed"):
            adapter.validate_confirmation_native_official_target(
                complete_confirmation["repository"],
                complete_confirmation["lock_path"],
                complete_confirmation["h2"],
                complete_confirmation["barrier_path"],
                complete_confirmation["case_dirs"],
                complete_confirmation["measurement_dirs"],
                complete_confirmation["compatibility_root"],
                case_id,
                future_root,
                outcome_root,
                expected_h1=complete_confirmation["h1"],
            )
    finally:
        video_path.write_bytes(original_video)

    metadata_path = future_root / "episode_0000" / "splatfacto" / "splatfacto.meta.json"
    original_metadata = metadata_path.read_bytes()
    metadata_path.write_bytes(original_metadata + b"post-manifest-metadata-change")
    try:
        with pytest.raises(
            ValueError,
            match="reconstruction metadata changed",
        ):
            adapter.validate_confirmation_native_official_target(
                complete_confirmation["repository"],
                complete_confirmation["lock_path"],
                complete_confirmation["h2"],
                complete_confirmation["barrier_path"],
                complete_confirmation["case_dirs"],
                complete_confirmation["measurement_dirs"],
                complete_confirmation["compatibility_root"],
                case_id,
                future_root,
                outcome_root,
                expected_h1=complete_confirmation["h1"],
            )
    finally:
        metadata_path.write_bytes(original_metadata)


def test_native_validator_rejects_rehashed_future_prefix_outside_source_custody(
    complete_confirmation: dict[str, Any],
) -> None:
    case_id = complete_confirmation["lock"]["selected_case_ids"][5]
    future_root, outcome_root = _write_synthetic_authorized_target(
        complete_confirmation,
        case_id,
    )
    future_manifest_path = (
        future_root / adapter.EXTERNAL_AUTHORIZED_FUTURE_MANIFEST_FILENAME
    )
    outcome_manifest_path = (
        outcome_root / adapter.EXTERNAL_AUTHORIZED_OUTCOME_MANIFEST_FILENAME
    )
    original_future_manifest = future_manifest_path.read_bytes()
    original_outcome_manifest = outcome_manifest_path.read_bytes()
    future = json.loads(original_future_manifest)
    first_camera = future["selected_cameras"][0]
    video_path = future_root / "episode_0000" / first_camera / "undistorted.mp4"
    original_video = video_path.read_bytes()
    video_path.write_bytes(original_video + b"reencoded-after-custody")
    camera_record = next(
        record
        for record in future["camera_records"]
        if record["camera"] == first_camera
    )
    camera_record["video_sha256"] = _sha256(video_path)
    camera_record["decoded_sealed_prefix_sha256"] = _synthetic_decoded_prefix_sha256(
        video_path, frame_count=58
    )
    future["result_sha256"] = adapter._result_sha256(future)
    _write_json(future_manifest_path, future)
    outcome = json.loads(original_outcome_manifest)
    outcome["inputs_sha256"]["authorized_future_manifest"] = _sha256(
        future_manifest_path
    )
    outcome["result_sha256"] = adapter._result_sha256(outcome)
    _write_json(outcome_manifest_path, outcome)
    try:
        with pytest.raises(ValueError, match="source[- ]custody|decoded prefix"):
            adapter.validate_confirmation_native_official_target(
                complete_confirmation["repository"],
                complete_confirmation["lock_path"],
                complete_confirmation["h2"],
                complete_confirmation["barrier_path"],
                complete_confirmation["case_dirs"],
                complete_confirmation["measurement_dirs"],
                complete_confirmation["compatibility_root"],
                case_id,
                future_root,
                outcome_root,
                expected_h1=complete_confirmation["h1"],
            )
    finally:
        video_path.write_bytes(original_video)
        future_manifest_path.write_bytes(original_future_manifest)
        outcome_manifest_path.write_bytes(original_outcome_manifest)


def test_native_target_loader_rejects_rehashed_camera_and_visibility_changes(
    complete_confirmation: dict[str, Any],
) -> None:
    case_id = complete_confirmation["lock"]["selected_case_ids"][1]
    future_root, outcome_root = _write_synthetic_authorized_target(
        complete_confirmation,
        case_id,
    )
    future_path = future_root / adapter.EXTERNAL_AUTHORIZED_FUTURE_MANIFEST_FILENAME
    original_future = future_path.read_bytes()
    changed_future = json.loads(original_future)
    changed_future["selected_cameras"][0:2] = reversed(
        changed_future["selected_cameras"][0:2]
    )
    changed_future["camera_records"][0:2] = reversed(
        changed_future["camera_records"][0:2]
    )
    changed_future["result_sha256"] = adapter._result_sha256(changed_future)
    _write_json(future_path, changed_future)
    try:
        with pytest.raises(ValueError, match="authorized future differs"):
            adapter.load_confirmation_native_official_target(
                complete_confirmation["repository"],
                complete_confirmation["lock_path"],
                complete_confirmation["h2"],
                complete_confirmation["barrier_path"],
                complete_confirmation["case_dirs"],
                complete_confirmation["measurement_dirs"],
                complete_confirmation["compatibility_root"],
                case_id,
                future_root,
                outcome_root,
                expected_h1=complete_confirmation["h1"],
            )
    finally:
        future_path.write_bytes(original_future)

    archive_path = outcome_root / adapter.EXTERNAL_TARGET_ARCHIVE_FILENAME
    outcome_path = outcome_root / adapter.EXTERNAL_AUTHORIZED_OUTCOME_MANIFEST_FILENAME
    outcome = json.loads(outcome_path.read_bytes())
    with np.load(archive_path, allow_pickle=False) as stored:
        target = np.asarray(stored["target_m"])
        visibility = np.asarray(stored["target_visibility"]).copy()
        validity = np.asarray(stored["target_validity"])
    visibility[0, 0] = False
    np.savez_compressed(
        archive_path,
        target_m=target,
        target_visibility=visibility,
        target_validity=validity,
    )
    outcome["output"]["target_archive_sha256"] = _sha256(archive_path)
    outcome["result_sha256"] = adapter._result_sha256(outcome)
    _write_json(outcome_path, outcome)
    with pytest.raises(ValueError, match="target array checksum changed"):
        adapter.load_confirmation_native_official_target(
            complete_confirmation["repository"],
            complete_confirmation["lock_path"],
            complete_confirmation["h2"],
            complete_confirmation["barrier_path"],
            complete_confirmation["case_dirs"],
            complete_confirmation["measurement_dirs"],
            complete_confirmation["compatibility_root"],
            case_id,
            future_root,
            outcome_root,
            expected_h1=complete_confirmation["h1"],
        )


def test_retained_tracker_free_carrier_and_identity_adapter_remain_bound_to_future(
    complete_confirmation: dict[str, Any],
) -> None:
    case_id = complete_confirmation["retained_case_id"]
    compatibility = _validate_compatibility(complete_confirmation)
    case = next(
        record
        for record in compatibility.manifest["cases"]
        if record["case"] == case_id
    )
    nested = case["nested_measurement"]
    assert nested["measurement_execution"] == (
        "retained-technical-failure-tracker-not-executed"
    )
    source = nested["retained_failure_source"]
    assert source["failure_code"] == "prediction_runtime_failure"
    assert (
        source["identity_persistence_adapter"]["adapted_material"]["point_count"] == 20
    )
    assert case["selected_cameras"] == _cameras()[8]

    future_root, outcome_root = _write_synthetic_authorized_target(
        complete_confirmation,
        case_id,
    )
    native = adapter.validate_confirmation_native_official_target(
        complete_confirmation["repository"],
        complete_confirmation["lock_path"],
        complete_confirmation["h2"],
        complete_confirmation["barrier_path"],
        complete_confirmation["case_dirs"],
        complete_confirmation["measurement_dirs"],
        complete_confirmation["compatibility_root"],
        case_id,
        future_root,
        outcome_root,
        expected_h1=complete_confirmation["h1"],
    )
    assert (
        native.evidence["identity_persistence_adapter"]
        == source["identity_persistence_adapter"]
    )
    assert (
        adapter._external_array_sha256(native.target_m[0])
        == source["identity_persistence_adapter"]["adapted_material"]["array_sha256"]
    )

    future_path = future_root / adapter.EXTERNAL_AUTHORIZED_FUTURE_MANIFEST_FILENAME
    future = json.loads(future_path.read_bytes())
    future["inputs_sha256"]["frame_zero_reconstruction_manifest"] = "4" * 64
    future["result_sha256"] = adapter._result_sha256(future)
    _write_json(future_path, future)
    with pytest.raises(ValueError, match="sealed source-stage lineage"):
        adapter.load_confirmation_native_official_target(
            complete_confirmation["repository"],
            complete_confirmation["lock_path"],
            complete_confirmation["h2"],
            complete_confirmation["barrier_path"],
            complete_confirmation["case_dirs"],
            complete_confirmation["measurement_dirs"],
            complete_confirmation["compatibility_root"],
            case_id,
            future_root,
            outcome_root,
            expected_h1=complete_confirmation["h1"],
        )


def test_retained_tracker_free_manifest_rejects_postbarrier_claimed_inference(
    complete_confirmation: dict[str, Any],
) -> None:
    case_id = complete_confirmation["retained_case_id"]
    manifest_path = (
        complete_confirmation["measurement_dirs"][case_id]
        / measurement.MANIFEST_FILENAME
    )
    original = manifest_path.read_bytes()
    payload = json.loads(original)
    payload["tracker"]["inference_executed"] = True
    payload["artifact_sha256"] = measurement._canonical_sha256(
        {key: value for key, value in payload.items() if key != "artifact_sha256"}
    )
    _write_json(manifest_path, payload)
    try:
        with pytest.raises(
            ValueError,
            match="sealed prediction input",
        ):
            _validate_compatibility(complete_confirmation)
    finally:
        manifest_path.write_bytes(original)
    _validate_compatibility(complete_confirmation)


def test_reserved_cli_abbreviations_cannot_override_bound_stage_paths() -> None:
    with pytest.raises(ValueError, match="reserved option abbreviation"):
        adapter._reject_reserved_option_abbreviations(
            ["--prot=/tmp/other"],
            {"--protocol", "--prediction-root"},
        )
    adapter._reject_reserved_option_abbreviations(
        ["--tracking-checkpoint", "/tmp/checkpoint"],
        {"--protocol", "--prediction-root"},
    )


def test_authorized_future_runner_injects_and_replays_custody_bound_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id = "synthetic-object__episode_0007"
    object_id = "synthetic-object"
    episode_id = 7
    h1 = "1" * 40
    h2 = "2" * 40
    execution = tmp_path / "execution"
    deform360 = tmp_path / "deform360"
    compatibility_root = tmp_path / "compatibility"
    prediction_root = compatibility_root / "predictions"
    measurement_root = compatibility_root / "measurements"
    source_aligned_root = tmp_path / "aligned"
    source_episode = source_aligned_root / object_id / f"episode_{episode_id:04d}"
    staged_case = tmp_path / "staged" / case_id
    for path in (
        execution,
        deform360,
        compatibility_root,
        prediction_root,
        measurement_root,
        source_episode,
        staged_case,
    ):
        path.mkdir(parents=True, exist_ok=True)
    lock_path = tmp_path / "cohort-lock.json"
    lock_path.write_text("{}\n", encoding="utf-8")
    compatibility_manifest_path = compatibility_root / "manifest.json"
    compatibility_manifest_path.write_text("{}\n", encoding="utf-8")
    custody_path = tmp_path / "source-custody.json"
    custody = {
        "artifact_sha256": "3" * 64,
        "path_binding": {
            "source_episode_dir": str(source_episode),
            "staged_case_dir": str(staged_case),
        },
    }
    _write_json(custody_path, custody)
    compatibility = adapter.ConfirmationOutcomeCompatibility(
        root=compatibility_root,
        manifest_path=compatibility_manifest_path,
        prediction_root=prediction_root,
        measurement_root=measurement_root,
        manifest={
            "cases": [
                {
                    "case": case_id,
                    "object_id": object_id,
                    "episode_id": episode_id,
                    "nested_measurement": {
                        "source_stage_lineage": {
                            "source_custody_seal": {
                                "path": str(custody_path),
                                "file_sha256": _sha256(custody_path),
                                "artifact_sha256": custody["artifact_sha256"],
                            },
                        },
                    },
                },
            ],
        },
    )
    events: list[tuple[str, Any]] = []

    def replay_custody(*arguments: Any, **keywords: Any) -> dict[str, Any]:
        assert arguments[0] == custody_path
        assert arguments[1] == lock_path
        assert arguments[2] == h2
        assert arguments[3] == case_id
        assert arguments[4] == source_episode
        assert arguments[5] == staged_case
        assert keywords == {"expected_h1": h1}
        events.append(("replay", len(events)))
        return custody

    class RuntimeActivation:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_arguments: object) -> bool:
            return False

    class StageModule:
        @staticmethod
        def main() -> int:
            events.append(("main", list(adapter.sys.argv)))
            return 17

    monkeypatch.setattr(
        adapter,
        "validate_confirmation_outcome_compatibility",
        lambda *_arguments, **_keywords: compatibility,
    )
    monkeypatch.setattr(
        adapter,
        "validate_confirmation_outcome_execution_repository",
        lambda *_arguments, **_keywords: None,
    )
    monkeypatch.setattr(
        adapter,
        "validate_deform360_execution_repository",
        lambda *_arguments, **_keywords: None,
    )
    monkeypatch.setattr(
        adapter,
        "validate_confirmation_source_custody_envelope",
        lambda *_arguments, **_keywords: custody,
    )
    monkeypatch.setattr(
        adapter,
        "validate_confirmation_source_custody_seal",
        replay_custody,
    )
    monkeypatch.setattr(
        adapter,
        "activate_confirmation_external_runtime",
        lambda *_arguments, **_keywords: RuntimeActivation(),
    )
    monkeypatch.setattr(
        adapter,
        "_load_external_stage",
        lambda *_arguments, **_keywords: StageModule,
    )
    monkeypatch.setattr(
        adapter,
        "validate_external_module_provenance",
        lambda *_arguments, **_keywords: None,
    )
    monkeypatch.setattr(
        adapter,
        "patch_confirmation_outcome_stage_module",
        lambda *_arguments, **_keywords: None,
    )

    stage_arguments = [
        "--object-id",
        object_id,
        "--episode-id",
        str(episode_id),
        "--output",
        str(tmp_path / "future"),
    ]
    status = adapter.run_confirmation_outcome_stage(
        "authorized-future",
        stage_arguments,
        adapter_repository=tmp_path,
        execution_repository=execution,
        deform360_repository=deform360,
        lock_path=lock_path,
        h2_commit=h2,
        barrier_path=tmp_path / "barrier.json",
        case_seal_dirs={},
        nested_measurement_dirs={},
        compatibility_root=compatibility_root,
        expected_h1=h1,
    )
    assert status == 17
    assert [event[0] for event in events] == ["replay", "main", "replay"]
    invoked_arguments = events[1][1]
    assert isinstance(invoked_arguments, list)
    assert invoked_arguments.count("--staged-case-dir") == 1
    assert invoked_arguments[invoked_arguments.index("--staged-case-dir") + 1] == str(
        staged_case
    )
    assert invoked_arguments.count("--source-aligned-root") == 1
    assert invoked_arguments[
        invoked_arguments.index("--source-aligned-root") + 1
    ] == str(source_aligned_root)

    mismatched_staged_case = tmp_path / "staged" / "other-case"
    mismatched_staged_case.mkdir()
    with pytest.raises(ValueError, match="authorization-bound path"):
        adapter.run_confirmation_outcome_stage(
            "authorized-future",
            [
                *stage_arguments,
                "--staged-case-dir",
                str(mismatched_staged_case),
            ],
            adapter_repository=tmp_path,
            execution_repository=execution,
            deform360_repository=deform360,
            lock_path=lock_path,
            h2_commit=h2,
            barrier_path=tmp_path / "barrier.json",
            case_seal_dirs={},
            nested_measurement_dirs={},
            compatibility_root=compatibility_root,
            expected_h1=h1,
        )
