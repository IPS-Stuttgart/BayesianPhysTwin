from __future__ import annotations

import json
from pathlib import Path

import pytest

from bayesian_phystwin.deform360_causal_response_direct_depth_assets import (
    canonical_sha256,
)
from bayesian_phystwin.deform360_causal_response_prefix_geometry import (
    GEOMETRY_CONTRACT,
    GEOMETRY_MANIFEST_KIND,
    GEOMETRY_PROTOCOL_ID,
    GEOMETRY_RESULT_KIND,
    load_v14_prefix_geometry_protocol,
)
from bayesian_phystwin.deform360_causal_response_prefix_geometry_validation import (
    RUNTIME_APPLICATION_KIND,
    RUNTIME_PROTOCOL_ID,
    load_v14_prefix_geometry_validation,
    validate_v14_prefix_geometry_bundle,
)
from bayesian_phystwin.deform360_object_exclusion import file_sha256

ROOT = Path(__file__).resolve().parents[1]
GEOMETRY = ROOT / "configs" / "sota" / (
    "deform360_causal_response_direct_depth_v14_prefix_geometry.json"
)
RUNTIME = ROOT / "configs" / "sota" / (
    "deform360_causal_response_direct_depth_v14_prefix_geometry_runtime.json"
)
VALIDATION = ROOT / "configs" / "sota" / (
    "deform360_causal_response_direct_depth_v14_prefix_geometry_validation.json"
)
VALIDATION_MODULE = ROOT / "src" / "bayesian_phystwin" / (
    "deform360_causal_response_prefix_geometry_validation.py"
)


def _write_json(
    path: Path,
    payload: dict[str, object],
    *,
    namespace: bytes,
) -> Path:
    payload["artifact_sha256"] = canonical_sha256(
        payload,
        namespace=namespace,
        digest_key="artifact_sha256",
    )
    path.write_text(
        json.dumps(payload, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return path


def _write_bundle(
    tmp_path: Path,
    *,
    plaintext: str | None = None,
) -> tuple[Path, Path, Path, Path]:
    geometry = load_v14_prefix_geometry_protocol(GEOMETRY)
    runtime = json.loads(RUNTIME.read_text(encoding="utf-8"))
    episode = tmp_path / "geometry" / "episode_0000"
    camera = "camera-a"
    fixed = {
        "intrinsics": episode / "undistorted_intrinsics.npy",
        "extrinsics": episode / "extrinsics.npy",
        "robot": episode / "robot" / "robot.npz",
        "frame_zero_splat": episode / "splatfacto" / "splat_0.ply",
        "frame_zero_points": episode / "start_obj_pcd.ply",
    }
    for name, path in fixed.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode("ascii"))
    depth = episode / camera / "rendered_depth.h5"
    depth.parent.mkdir(parents=True)
    depth.write_bytes(b"depth")
    object_hash = "a" * 64
    case_hash = "b" * 64
    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": GEOMETRY_MANIFEST_KIND,
        "contract": GEOMETRY_CONTRACT,
        "protocol_id": GEOMETRY_PROTOCOL_ID,
        "geometry_protocol_config_sha256": geometry["config_sha256"],
        "status": "ready_for_physical_preflight",
        "queue_rank": 3,
        "object_hash": object_hash,
        "case_hash": case_hash,
        "physical_node_count": 128,
        "cameras": [camera],
        "camera_records": [
            {
                "camera": camera,
                "rgb_frame_count": 58,
                "mask_frame_count": 58,
                "depth_frame_count": 58,
                "gripper_mask_frame_count": 58,
            }
        ],
        "calibration_valid": True,
        "runtime": {
            "runtime_amendment_config_sha256": runtime["config_sha256"],
            "runtime_amendment_file_sha256": file_sha256(RUNTIME),
            "gsplat_extension_sha256": runtime["runtime_amendment"][
                "rebuilt_extension_sha256"
            ],
        },
        "outputs_sha256": {
            **{name: file_sha256(path) for name, path in fixed.items()},
            "depth_by_camera": {camera: file_sha256(depth)},
        },
    }
    if plaintext is not None:
        manifest["forbidden_note"] = plaintext
    manifest_path = _write_json(
        tmp_path / "manifest.json",
        manifest,
        namespace=(
            b"deform360-causal-response-direct-depth-prefix-geometry-v14\0"
        ),
    )
    result = {
        "schema_version": 1,
        "artifact_kind": GEOMETRY_RESULT_KIND,
        "status": "ready_for_source_lock",
        "queue_rank": 3,
        "object_hash": object_hash,
        "case_hash": case_hash,
        "physical_node_count": 128,
        "geometry_manifest_artifact_sha256": manifest["artifact_sha256"],
        "geometry_manifest_file_sha256": file_sha256(manifest_path),
    }
    result_path = _write_json(
        tmp_path / "rank-003.json",
        result,
        namespace=(
            b"deform360-causal-response-direct-depth-prefix-geometry-"
            b"result-v14\0"
        ),
    )
    application = {
        "schema_version": 1,
        "artifact_kind": RUNTIME_APPLICATION_KIND,
        "protocol_id": RUNTIME_PROTOCOL_ID,
        "status": "runtime_amendment_applied",
        "runtime_amendment_config_sha256": runtime["config_sha256"],
        "runtime_amendment_file_sha256": file_sha256(RUNTIME),
        "geometry_result_artifact_sha256": result["artifact_sha256"],
        "geometry_result_file_sha256": file_sha256(result_path),
    }
    application_path = _write_json(
        tmp_path / "rank-003.runtime.json",
        application,
        namespace=(
            b"deform360-causal-response-direct-depth-prefix-geometry-"
            b"runtime-application-v14\0"
        ),
    )
    return manifest_path, result_path, application_path, episode


def test_v14_validation_amendment_binds_parents_and_module() -> None:
    validation = load_v14_prefix_geometry_validation(VALIDATION)

    assert validation["parent_geometry_protocol"]["file_sha256"] == file_sha256(
        GEOMETRY
    )
    assert validation["parent_runtime_amendment"]["file_sha256"] == file_sha256(
        RUNTIME
    )
    assert validation["implementation_file_sha256"]["validation_module"] == (
        file_sha256(VALIDATION_MODULE)
    )
    assert validation["validation_policy"]["applies_to_queue_ranks"] == list(
        range(3, 15)
    )


def test_v14_bundle_validator_uses_official_splat_filename(
    tmp_path: Path,
) -> None:
    manifest, result, application, episode = _write_bundle(tmp_path)
    geometry = load_v14_prefix_geometry_protocol(GEOMETRY)
    runtime = json.loads(RUNTIME.read_text(encoding="utf-8"))
    validation = load_v14_prefix_geometry_validation(VALIDATION)

    loaded, _, _ = validate_v14_prefix_geometry_bundle(
        manifest_path=manifest,
        result_path=result,
        runtime_application_path=application,
        geometry_protocol=geometry,
        runtime_amendment=runtime,
        validation_amendment=validation,
        geometry_episode=episode,
        forbidden_plaintext="secret-object-id",
    )

    assert loaded["physical_node_count"] == 128
    assert (episode / "splatfacto" / "splat_0.ply").is_file()
    assert not (episode / "splatfacto" / "splat_000000.ply").exists()


def test_v14_bundle_validator_rejects_plaintext_identity(
    tmp_path: Path,
) -> None:
    manifest, result, application, episode = _write_bundle(
        tmp_path,
        plaintext="secret-object-id",
    )
    geometry = load_v14_prefix_geometry_protocol(GEOMETRY)
    runtime = json.loads(RUNTIME.read_text(encoding="utf-8"))
    validation = load_v14_prefix_geometry_validation(VALIDATION)

    with pytest.raises(ValueError, match="leaked plaintext"):
        validate_v14_prefix_geometry_bundle(
            manifest_path=manifest,
            result_path=result,
            runtime_application_path=application,
            geometry_protocol=geometry,
            runtime_amendment=runtime,
            validation_amendment=validation,
            geometry_episode=episode,
            forbidden_plaintext="secret-object-id",
        )
