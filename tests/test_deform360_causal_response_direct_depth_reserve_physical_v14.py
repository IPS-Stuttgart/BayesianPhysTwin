from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from bayesian_phystwin.deform360_causal_response_direct_depth_assets import (
    canonical_sha256,
)
from bayesian_phystwin.deform360_causal_response_direct_depth_cohort import (
    validate_v14_staging_queue,
)
from bayesian_phystwin.deform360_causal_response_direct_depth_preflight import (
    deform360_v14_case_hash,
)
from bayesian_phystwin.deform360_causal_response_direct_depth_reserve_physical_v14 import (
    RESERVE_GEOMETRY_APPLICATION_V2_ID,
    RESERVE_GEOMETRY_APPLICATION_V2_KIND,
    RESERVE_PHYSICAL_PRELOCK_CONTRACT,
    RESERVE_PHYSICAL_PRELOCK_ID,
    RESERVE_PHYSICAL_PRELOCK_KIND,
    RESERVE_PHYSICAL_RUNTIME_CONTRACT,
    RESERVE_PHYSICAL_RUNTIME_ID,
    RESERVE_PHYSICAL_RUNTIME_KIND,
    load_v14_reserve_physical_prelock,
    load_v14_reserve_physical_runtime,
    reserve_geometry_ledger_sha256,
    v14_reserve_physical_case_record,
    validate_v14_reserve_geometry_bundle_v2,
    validate_v14_reserve_physical_action,
)
from bayesian_phystwin.deform360_causal_response_direct_depth_reserve_v14 import (
    RESERVE_GEOMETRY_RANKS,
)
from bayesian_phystwin.deform360_causal_response_prefix_geometry import (
    GEOMETRY_CONTRACT,
    GEOMETRY_MANIFEST_KIND,
    GEOMETRY_PROTOCOL_ID,
    GEOMETRY_RESULT_KIND,
)
from bayesian_phystwin.deform360_causal_response_preflight import (
    deform360_object_hash,
)
from bayesian_phystwin.deform360_object_exclusion import file_sha256

ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = (
    ROOT / "configs/sota/deform360_causal_response_direct_depth_v14_staging_queue.json"
)
RESERVE_BATCH_PATH = (
    ROOT / "configs/sota/deform360_causal_response_direct_depth_v14_reserve_batch_v1.json"
)
RESERVE_GEOMETRY_PATH = (
    ROOT
    / "configs/sota/deform360_causal_response_direct_depth_v14_reserve_geometry_v1.json"
)
RESERVE_GEOMETRY_RUNTIME_V2_PATH = (
    ROOT
    / "configs/sota/"
    "deform360_causal_response_direct_depth_v14_reserve_geometry_runtime_v2.json"
)
AUTOMATIC_TWIN = (
    ROOT
    / "scripts/remote/"
    "build_deform360_causal_response_direct_depth_v14_automatic_twin.py"
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _config_sha256(payload: dict, namespace: bytes) -> str:
    canonical = dict(payload)
    canonical.pop("config_sha256", None)
    return hashlib.sha256(
        namespace
        + json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _reserve_prelock(path: Path) -> dict:
    queue = validate_v14_staging_queue(QUEUE_PATH)
    geometry_cases = []
    for rank in RESERVE_GEOMETRY_RANKS:
        candidate = queue["candidates"][rank - 1]
        geometry_cases.append(
            {
                "queue_rank": rank,
                "object_hash": deform360_object_hash(candidate["object_id"]),
                "case_hash": deform360_v14_case_hash(
                    candidate["object_id"],
                    candidate["episode_id"],
                ),
                "metadata_sha256": candidate["metadata_sha256"],
                "physical_node_count": 128 + rank,
                "successful_camera_count": 12,
                "runtime_contract_version": "reserve-v2",
                "geometry_manifest_artifact_sha256": _digest(
                    f"manifest-artifact-{rank}"
                ),
                "geometry_manifest_file_sha256": _digest(f"manifest-file-{rank}"),
                "geometry_result_artifact_sha256": _digest(
                    f"result-artifact-{rank}"
                ),
                "geometry_result_file_sha256": _digest(f"result-file-{rank}"),
                "runtime_application_artifact_sha256": _digest(
                    f"application-artifact-{rank}"
                ),
                "runtime_application_file_sha256": _digest(
                    f"application-file-{rank}"
                ),
            }
        )
    payload = {
        "schema_version": 1,
        "artifact_kind": RESERVE_PHYSICAL_PRELOCK_KIND,
        "contract": RESERVE_PHYSICAL_PRELOCK_CONTRACT,
        "protocol_id": RESERVE_PHYSICAL_PRELOCK_ID,
        "method_protocol_id": "deform360-causal-response-direct-depth-v14-source",
        "status": "locked_after_reserve_geometry_before_physical_execution",
        "config_sha256": "0" * 64,
        "implementation": {
            "parent_commit": "a" * 40,
            "file_sha256": {
                "artifact_module": _digest("artifact-module"),
                "automatic_twin": _digest("automatic-twin"),
                "parent_physical_runner": _digest("parent-runner"),
                "reserve_physical_module": _digest("reserve-module"),
                "reserve_physical_runner": _digest("reserve-runner"),
            },
        },
        "parent_artifacts": {
            "geometry_protocol_file_sha256": _digest("geometry-protocol"),
            "runtime_v1_file_sha256": _digest("runtime-v1"),
            "runtime_v2_file_sha256": _digest("runtime-v2"),
            "staging_queue_file_sha256": file_sha256(QUEUE_PATH),
            "staging_queue_sha256": queue["queue_sha256"],
            "validation_v1_file_sha256": _digest("validation-v1"),
            "validation_v2_file_sha256": _digest("validation-v2"),
            **{
                key: {
                    "config_sha256": _digest(f"{key}-config"),
                    "file_sha256": _digest(f"{key}-file"),
                }
                for key in (
                    "parent_physical_prelock",
                    "reserve_batch",
                    "reserve_geometry",
                    "reserve_geometry_runtime_v2",
                )
            },
        },
        "numerical_contract": {
            "canonical_node_count": 384,
            "graph_basis_rank": 8,
            "prediction_frame_count": 76,
            "automatic_twin_source": "frame_zero_geometry_only",
            "future_robot_action_known": True,
            "automatic_twin_inadmissible_fallback": "bit_exact_persistence",
        },
        "geometry_cases": geometry_cases,
        "geometry_ledger_sha256": reserve_geometry_ledger_sha256(geometry_cases),
        "information_boundary": {
            "object_observation_frames_used": [0],
            "known_robot_action_frames_used": list(range(76)),
            "future_object_observation_read": False,
            "prefix_tactile_read": False,
            "identity_or_metric_outcome_read": False,
            "source_lock_required_before_execution": False,
            "source_lock_construction_uses_output_hashes_only": True,
            "plaintext_identity_retained_in_sealed_output": False,
            "held_v8_artifact_or_process_access": False,
        },
    }
    payload["config_sha256"] = _config_sha256(
        payload,
        (
            b"deform360-causal-response-direct-depth-reserve-physical-"
            b"prelock-v14-v1\0"
        ),
    )
    _write_json(path, payload)
    return payload


def _reserve_runtime(
    path: Path,
    *,
    prelock_path: Path,
    stage_path: Path,
    action_path: Path,
) -> dict:
    prelock = json.loads(prelock_path.read_text())
    stage = json.loads(stage_path.read_text())
    action_cases = []
    for rank in RESERVE_GEOMETRY_RANKS:
        record = next(
            row for row in prelock["geometry_cases"] if row["queue_rank"] == rank
        )
        action_cases.append(
            {
                "queue_rank": rank,
                "object_hash": record["object_hash"],
                "case_hash": record["case_hash"],
                "window_stage_artifact_sha256": (
                    stage["artifact_sha256"]
                    if rank == 15
                    else _digest(f"stage-artifact-{rank}")
                ),
                "window_stage_file_sha256": (
                    file_sha256(stage_path)
                    if rank == 15
                    else _digest(f"stage-file-{rank}")
                ),
                "known_action_file_sha256": (
                    file_sha256(action_path)
                    if rank == 15
                    else _digest(f"action-{rank}")
                ),
                "staged_frame_count": 81,
            }
        )
    payload = {
        "schema_version": 1,
        "artifact_kind": RESERVE_PHYSICAL_RUNTIME_KIND,
        "contract": RESERVE_PHYSICAL_RUNTIME_CONTRACT,
        "protocol_id": RESERVE_PHYSICAL_RUNTIME_ID,
        "status": "locked_before_reserve_physical_execution",
        "config_sha256": "0" * 64,
        "parent_physical_prelock": {
            "config_sha256": prelock["config_sha256"],
            "file_sha256": file_sha256(prelock_path),
        },
        "action_contract": {
            "known_action_source": "exact_action_only_staged_window_robot",
            "accepted_staged_frame_counts": [76, 81],
            "physical_frame_count": 76,
            "object_observation_source": "frame_zero_only",
        },
        "action_cases": action_cases,
        "implementation": {
            "parent_commit": "b" * 40,
            "file_sha256": {
                "physical_runner": _digest("physical-runner"),
                "reserve_runner": _digest("reserve-runner"),
                "runtime_module": _digest("runtime-module"),
            },
        },
        "information_boundary": {
            "known_future_robot_action_read": True,
            "future_object_observation_read": False,
            "future_tactile_read": False,
            "future_identity_or_metric_read": False,
            "target_object_or_outcome_read": False,
            "held_v8_artifact_or_process_access": False,
        },
    }
    payload["config_sha256"] = _config_sha256(
        payload,
        (
            b"deform360-causal-response-direct-depth-reserve-physical-"
            b"runtime-v14-v1\0"
        ),
    )
    _write_json(path, payload)
    return payload


def test_reserve_prelock_and_action_are_hash_bound(tmp_path: Path) -> None:
    prelock_path = tmp_path / "prelock.json"
    prelock = _reserve_prelock(prelock_path)
    record = v14_reserve_physical_case_record(
        prelock,
        QUEUE_PATH,
        queue_rank=15,
    )
    assert load_v14_reserve_physical_prelock(prelock_path) == prelock
    assert record["queue_rank"] == 15
    assert record["runtime_contract_version"] == "reserve-v2"

    stage_path = tmp_path / "stage.json"
    _write_json(
        stage_path,
        {
            "artifact_sha256": _digest("stage-artifact"),
            "status": "staged",
            "queue_rank": 15,
            "object_hash": record["object_hash"],
            "case_hash": record["case_hash"],
        },
    )
    action_path = tmp_path / "robot.npz"
    action_path.write_bytes(b"known 81-frame action")
    runtime_path = tmp_path / "runtime.json"
    runtime = _reserve_runtime(
        runtime_path,
        prelock_path=prelock_path,
        stage_path=stage_path,
        action_path=action_path,
    )
    loaded = load_v14_reserve_physical_runtime(
        runtime_path,
        parent_prelock_path=prelock_path,
    )
    action_record = validate_v14_reserve_physical_action(
        loaded,
        queue_rank=15,
        object_hash=record["object_hash"],
        case_hash=record["case_hash"],
        window_stage_result_path=stage_path,
        known_action_path=action_path,
        staged_frame_count=81,
    )
    assert action_record["known_action_file_sha256"] == file_sha256(action_path)

    action_path.write_bytes(b"mutated action")
    with pytest.raises(ValueError, match="action differs"):
        validate_v14_reserve_physical_action(
            runtime,
            queue_rank=15,
            object_hash=record["object_hash"],
            case_hash=record["case_hash"],
            window_stage_result_path=stage_path,
            known_action_path=action_path,
            staged_frame_count=81,
        )


def test_automatic_twin_dispatches_reserve_without_changing_baseline(
    tmp_path: Path,
) -> None:
    spec = importlib.util.spec_from_file_location("_v14_auto_twin_test", AUTOMATIC_TWIN)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    reserve_path = tmp_path / "reserve.json"
    _reserve_prelock(reserve_path)
    reserve, builder = module._load_physical_protocol(reserve_path)
    assert reserve["artifact_kind"] == RESERVE_PHYSICAL_PRELOCK_KIND
    assert builder.__name__ == "v14_reserve_physical_case_record"

    baseline_path = (
        ROOT
        / "configs/sota/"
        "deform360_causal_response_direct_depth_v14_physical_prelock.json"
    )
    baseline, baseline_builder = module._load_physical_protocol(baseline_path)
    assert baseline["artifact_kind"] != RESERVE_PHYSICAL_PRELOCK_KIND
    assert baseline_builder.__name__ == "v14_physical_case_record"


def test_reserve_geometry_runtime_v2_bundle_is_fully_validated(
    tmp_path: Path,
) -> None:
    geometry_protocol = json.loads(RESERVE_GEOMETRY_PATH.read_text())
    runtime_v2 = json.loads(RESERVE_GEOMETRY_RUNTIME_V2_PATH.read_text())
    episode = tmp_path / "episode"
    fixed = {
        "intrinsics": episode / "undistorted_intrinsics.npy",
        "extrinsics": episode / "extrinsics.npy",
        "robot": episode / "robot" / "robot.npz",
        "frame_zero_splat": episode / "splatfacto" / "splat_0.ply",
        "frame_zero_points": episode / "start_obj_pcd.ply",
    }
    for role, path in fixed.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{role}-bytes".encode())
    cameras = [f"camera_{index}" for index in range(8)]
    depth = {}
    for camera in cameras:
        path = episode / camera / "rendered_depth.h5"
        path.parent.mkdir(parents=True)
        path.write_bytes(f"{camera}-depth".encode())
        depth[camera] = file_sha256(path)

    manifest = {
        "schema_version": 1,
        "artifact_kind": GEOMETRY_MANIFEST_KIND,
        "contract": GEOMETRY_CONTRACT,
        "protocol_id": GEOMETRY_PROTOCOL_ID,
        "geometry_protocol_config_sha256": geometry_protocol["config_sha256"],
        "status": "ready_for_physical_preflight",
        "queue_rank": 15,
        "object_hash": _digest("object-15"),
        "case_hash": _digest("case-15"),
        "physical_node_count": 256,
        "cameras": cameras,
        "camera_records": [
            {
                "camera": camera,
                "rgb_frame_count": 58,
                "mask_frame_count": 58,
                "depth_frame_count": 58,
                "gripper_mask_frame_count": 58,
            }
            for camera in cameras
        ],
        "outputs_sha256": {
            **{role: file_sha256(path) for role, path in fixed.items()},
            "depth_by_camera": depth,
        },
        "runtime": {
            key: geometry_protocol["runtime"][key]
            for key in (
                "gsplat_extension_sha256",
                "python_version",
                "torch_version",
            )
        },
    }
    manifest["artifact_sha256"] = canonical_sha256(
        manifest,
        namespace=(
            b"deform360-causal-response-direct-depth-prefix-geometry-v14\0"
        ),
        digest_key="artifact_sha256",
    )
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)

    result = {
        "schema_version": 1,
        "artifact_kind": GEOMETRY_RESULT_KIND,
        "contract": GEOMETRY_CONTRACT,
        "protocol_id": GEOMETRY_PROTOCOL_ID,
        "geometry_protocol_config_sha256": geometry_protocol["config_sha256"],
        "status": "ready_for_source_lock",
        "queue_rank": 15,
        "object_hash": manifest["object_hash"],
        "case_hash": manifest["case_hash"],
        "geometry_manifest_artifact_sha256": manifest["artifact_sha256"],
        "geometry_manifest_file_sha256": file_sha256(manifest_path),
    }
    result["artifact_sha256"] = canonical_sha256(
        result,
        namespace=(
            b"deform360-causal-response-direct-depth-prefix-geometry-result-v14\0"
        ),
        digest_key="artifact_sha256",
    )
    result_path = tmp_path / "result.json"
    _write_json(result_path, result)

    application = {
        "schema_version": 1,
        "artifact_kind": RESERVE_GEOMETRY_APPLICATION_V2_KIND,
        "protocol_id": RESERVE_GEOMETRY_APPLICATION_V2_ID,
        "status": "reserve_geometry_runtime_v2_applied",
        "runtime_v2_config_sha256": runtime_v2["config_sha256"],
        "runtime_v2_file_sha256": file_sha256(RESERVE_GEOMETRY_RUNTIME_V2_PATH),
        "reserve_geometry_config_sha256": geometry_protocol["config_sha256"],
        "reserve_geometry_file_sha256": file_sha256(RESERVE_GEOMETRY_PATH),
        "geometry_result_artifact_sha256": result["artifact_sha256"],
        "geometry_result_file_sha256": file_sha256(result_path),
        "queue_rank": 15,
        "object_hash": manifest["object_hash"],
        "case_hash": manifest["case_hash"],
        "information_boundary": {
            "maximum_object_observation_frame": 57,
            "future_object_observation_read": False,
            "future_identity_or_metric_read": False,
            "source_outcome_read": False,
            "target_object_or_outcome_read": False,
            "held_v8_artifact_or_process_access": False,
        },
    }
    application["artifact_sha256"] = canonical_sha256(
        application,
        namespace=(
            b"deform360-causal-response-direct-depth-reserve-geometry-"
            b"application-v14-v2\0"
        ),
        digest_key="artifact_sha256",
    )
    application_path = tmp_path / "application.json"
    _write_json(application_path, application)

    loaded = validate_v14_reserve_geometry_bundle_v2(
        manifest_path=manifest_path,
        result_path=result_path,
        application_path=application_path,
        geometry_protocol_path=RESERVE_GEOMETRY_PATH,
        reserve_batch_path=RESERVE_BATCH_PATH,
        runtime_v2_path=RESERVE_GEOMETRY_RUNTIME_V2_PATH,
        geometry_episode=episode,
    )
    assert loaded == (manifest, result, application)

    (episode / cameras[0] / "rendered_depth.h5").write_bytes(b"changed")
    with pytest.raises(ValueError, match="camera outputs changed"):
        validate_v14_reserve_geometry_bundle_v2(
            manifest_path=manifest_path,
            result_path=result_path,
            application_path=application_path,
            geometry_protocol_path=RESERVE_GEOMETRY_PATH,
            reserve_batch_path=RESERVE_BATCH_PATH,
            runtime_v2_path=RESERVE_GEOMETRY_RUNTIME_V2_PATH,
            geometry_episode=episode,
        )
