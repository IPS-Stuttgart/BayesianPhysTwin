import json
import zipfile
from pathlib import Path

import pytest

import bayesian_phystwin.pokeflex_fresh12_staging as staging
from bayesian_phystwin.pokeflex_conservative_shrinkage_target import (
    ACTION_ROBUST_FRESH6_PUBLIC_TARGET_TAKE_IDS,
    ACTION_ROBUST_FRESH6_PUBLIC_ZIP_SHA256,
    FRESH12_PUBLIC_TARGET_TAKE_IDS,
    FRESH12_PUBLIC_ZIP_SHA256,
    TARGET_PROTOCOL_ACTION_ROBUST_FRESH6_V3,
    TARGET_PROTOCOL_FRESH12_PUBLIC_V1,
    file_sha256,
)
from bayesian_phystwin.pokeflex_fresh12_staging import (
    STAGE_MANIFEST_KIND,
    stage_manifest_sha256,
    validate_pokeflex_fresh12_stage_manifest,
    validate_staged_file,
)


def _protocol() -> dict[str, object]:
    return {
        "protocol_id": TARGET_PROTOCOL_FRESH12_PUBLIC_V1,
        "protocol_sha256": "1" * 64,
    }


def _manifest(take_id: str) -> dict[str, object]:
    rows = sorted(
        [
            ("robot_data.json", "robot", None),
            ("kinect/0/camera_parameters.json", "calibration", "0"),
            ("kinect/1/camera_parameters.json", "calibration", "1"),
            ("kinect/0/depth/00001.png", "depth", "0"),
            ("kinect/1/depth/00001.png", "depth", "1"),
            ("meshes/mesh-f00001.obj", "mesh", None),
        ]
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": STAGE_MANIFEST_KIND,
        "protocol_sha256": "1" * 64,
        "take_id": take_id,
        "archive_name": f"{take_id}.zip",
        "archive_sha256": FRESH12_PUBLIC_ZIP_SHA256[take_id],
        "archive_byte_count": 100,
        "target_mesh_geometry_decoded": False,
        "outcome_metric_computed": False,
        "files": [
            {
                "path": path,
                "kind": kind,
                "camera": camera,
                "byte_count": 1,
                "sha256": "a" * 64,
            }
            for path, kind, camera in rows
        ],
    }
    payload["stage_manifest_sha256"] = stage_manifest_sha256(payload)
    return payload


def test_stage_manifest_binds_archive_and_authorized_inventory(tmp_path: Path) -> None:
    take_id = FRESH12_PUBLIC_TARGET_TAKE_IDS[0]
    manifest = _manifest(take_id)
    path = tmp_path / "source_stage_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = validate_pokeflex_fresh12_stage_manifest(
        path,
        _protocol(),
        expected_take_id=take_id,
    )

    assert loaded["take_id"] == take_id
    assert len(loaded["files_by_path"]) == 6


def test_stage_manifest_supports_action_robust_fresh6(tmp_path: Path) -> None:
    take_id = ACTION_ROBUST_FRESH6_PUBLIC_TARGET_TAKE_IDS[0]
    manifest = _manifest(FRESH12_PUBLIC_TARGET_TAKE_IDS[0])
    manifest["take_id"] = take_id
    manifest["archive_name"] = f"{take_id}.zip"
    manifest["archive_sha256"] = ACTION_ROBUST_FRESH6_PUBLIC_ZIP_SHA256[take_id]
    manifest["stage_manifest_sha256"] = stage_manifest_sha256(manifest)
    path = tmp_path / "source_stage_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    protocol = {
        "protocol_id": TARGET_PROTOCOL_ACTION_ROBUST_FRESH6_V3,
        "protocol_sha256": "1" * 64,
    }

    loaded = validate_pokeflex_fresh12_stage_manifest(
        path,
        protocol,
        expected_take_id=take_id,
    )

    assert loaded["take_id"] == take_id


def test_stage_manifest_rejects_resigned_unauthorized_member(tmp_path: Path) -> None:
    take_id = FRESH12_PUBLIC_TARGET_TAKE_IDS[0]
    manifest = _manifest(take_id)
    manifest["files"][0]["path"] = "realsense/0/depth/00001.png"
    manifest["stage_manifest_sha256"] = stage_manifest_sha256(manifest)
    path = tmp_path / "source_stage_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="unauthorized member"):
        validate_pokeflex_fresh12_stage_manifest(path, _protocol())


def test_staged_file_verification_rejects_changed_bytes(tmp_path: Path) -> None:
    root = tmp_path / "take"
    path = root / "robot_data.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"original")
    files = {
        "robot_data.json": {
            "path": "robot_data.json",
            "byte_count": len(b"original"),
            "sha256": staging._bytes_sha256(b"original"),
        }
    }

    validate_staged_file(path, root, files)
    path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="size changed|bytes changed"):
        validate_staged_file(path, root, files)


def test_opaque_stager_extracts_only_authorized_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    take_id = FRESH12_PUBLIC_TARGET_TAKE_IDS[0]
    archive = tmp_path / f"{take_id}.zip"
    members = {
        f"{take_id}/robot_data.json": b"[]",
        f"{take_id}/kinect/0/camera_parameters.json": b"{}",
        f"{take_id}/kinect/1/camera_parameters.json": b"{}",
        f"{take_id}/kinect/0/depth/00001.png": b"zero",
        f"{take_id}/kinect/1/depth/00001.png": b"one",
        f"{take_id}/meshes/mesh-f00001.obj": b"v 0 0 0\n",
        f"{take_id}/realsense/0/depth/00001.png": b"forbidden",
    }
    with zipfile.ZipFile(archive, "w") as payload:
        for name, content in members.items():
            payload.writestr(name, content)
    digest = file_sha256(archive)
    monkeypatch.setitem(staging.FRESH12_PUBLIC_ZIP_SHA256, take_id, digest)
    monkeypatch.setattr(
        staging,
        "load_pokeflex_shrinkage_target_protocol",
        lambda _: _protocol(),
    )

    result = staging.stage_pokeflex_fresh12_archive(
        archive,
        tmp_path / "staged",
        tmp_path / "protocol.json",
    )

    destination = tmp_path / "staged" / take_id
    assert (destination / "robot_data.json").is_file()
    assert (destination / "meshes" / "mesh-f00001.obj").is_file()
    assert not (destination / "realsense").exists()
    assert result["target_mesh_geometry_decoded"] is False
    assert result["outcome_metric_computed"] is False
