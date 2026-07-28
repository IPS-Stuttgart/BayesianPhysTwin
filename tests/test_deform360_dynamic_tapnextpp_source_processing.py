from __future__ import annotations

import json
from pathlib import Path

import pytest

from bayesian_phystwin.deform360_dynamic_tapnextpp_source_processing import (
    load_dynamic_source_processing_protocol,
    load_dynamic_source_processing_runtime_amendment,
    validate_dynamic_source_mask_artifact,
)
from bayesian_phystwin.deform360_dynamic_tapnextpp_source_window import (
    FROZEN_CAMERA_PANEL,
    MASK_ARTIFACT_KIND,
    canonical_sha256,
    file_sha256,
    load_dynamic_source_mask_protocol,
)


ROOT = Path(__file__).resolve().parents[1]
PROCESSING_PROTOCOL = (
    ROOT / "configs" / "sota" / "deform360_dynamic_tapnextpp_source_processing_v1.json"
)
MASK_PROTOCOL = (
    ROOT / "configs" / "sota" / "deform360_dynamic_tapnextpp_source_masks_v1.json"
)
RUNTIME_AMENDMENT = (
    ROOT
    / "configs"
    / "sota"
    / "deform360_dynamic_tapnextpp_source_processing_runtime_amendment_v1.json"
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _mask_fixture(tmp_path: Path, *, revision: str) -> tuple[Path, Path, dict]:
    mask_protocol = load_dynamic_source_mask_protocol(MASK_PROTOCOL)
    case = {
        "object_id": "025-bag-small-cloth",
        "episode_id": 0,
        "queue_rank": 1,
        "category": "sheet",
        "catalog_oid": "a" * 40,
        "metadata_sha256": "b" * 64,
        "bimanual": "no",
    }
    episode = tmp_path / case["object_id"] / "episode_0000"
    episode.mkdir(parents=True)
    records = []
    for camera in FROZEN_CAMERA_PANEL:
        camera_dir = episode / camera
        camera_dir.mkdir()
        mask = camera_dir / "mask_refined.h5"
        mask.write_bytes(f"mask-{camera}".encode())
        records.append(
            {
                "camera": camera,
                "status": "success",
                "mask_sha256": file_sha256(mask),
                "frame_count": 81,
            }
        )
    manifest = {
        "schema_version": 1,
        "artifact_kind": MASK_ARTIFACT_KIND,
        "protocol_id": mask_protocol["protocol_id"],
        "protocol_config_sha256": mask_protocol["config_sha256"],
        **case,
        "status": "ready_for_source_processing",
        "code_revision": revision,
        "successful_camera_count": len(records),
        "camera_records": records,
    }
    manifest["result_sha256"] = canonical_sha256(
        manifest,
        digest_key="result_sha256",
    )
    path = episode / "dynamic_tapnextpp_source_masks.json"
    _write_json(path, manifest)
    return path, episode, case


def test_dynamic_processing_protocol_is_locked() -> None:
    protocol = load_dynamic_source_processing_protocol(PROCESSING_PROTOCOL)

    assert protocol["camera_policy"]["minimum_camera_count"] == 8
    assert protocol["source_admission"]["minimum_camera_count"] == 8
    assert protocol["failure_accounting"]["minimum_final_admissions"] == 20
    assert (
        protocol["parent_mask_protocol"]["implementation_commit"]
        == "2f731eef2c637977bf5cf4010760a8014bbe4161"
    )


def test_dynamic_processing_protocol_rejects_recomputed_change(
    tmp_path: Path,
) -> None:
    changed = json.loads(PROCESSING_PROTOCOL.read_text(encoding="utf-8"))
    changed["source_admission"]["minimum_camera_count"] = 3
    changed["config_sha256"] = canonical_sha256(
        changed,
        digest_key="config_sha256",
    )
    path = tmp_path / "changed.json"
    _write_json(path, changed)

    with pytest.raises(ValueError, match="source-admission contract"):
        load_dynamic_source_processing_protocol(path)


def test_dynamic_processing_runtime_amendment_is_locked() -> None:
    amendment = load_dynamic_source_processing_runtime_amendment(
        RUNTIME_AMENDMENT,
        parent_protocol_path=PROCESSING_PROTOCOL,
    )

    assert amendment["trigger"]["technical_failure_count"] == 8
    assert amendment["trigger"]["successful_reconstruction_count"] == 0
    assert amendment["application_policy"]["unattempted_queue_entry_count"] == 26
    assert amendment["application_policy"]["retry_failed_entries"] is False
    assert (
        amendment["runtime_contract"]["required_backend_probe"]
        == "CameraModelType.PINHOLE"
    )


def test_dynamic_processing_runtime_amendment_rejects_retry(
    tmp_path: Path,
) -> None:
    changed = json.loads(RUNTIME_AMENDMENT.read_text(encoding="utf-8"))
    changed["application_policy"]["retry_failed_entries"] = True
    changed["config_sha256"] = canonical_sha256(
        changed,
        digest_key="config_sha256",
    )
    path = tmp_path / "changed-runtime.json"
    _write_json(path, changed)

    with pytest.raises(ValueError, match="application policy"):
        load_dynamic_source_processing_runtime_amendment(
            path,
            parent_protocol_path=PROCESSING_PROTOCOL,
        )


def test_dynamic_mask_artifact_binds_mask_execution_commit(
    tmp_path: Path,
) -> None:
    revision = "c" * 40
    path, episode, case = _mask_fixture(tmp_path, revision=revision)
    mask_protocol = load_dynamic_source_mask_protocol(MASK_PROTOCOL)

    manifest, cameras = validate_dynamic_source_mask_artifact(
        path,
        mask_protocol=mask_protocol,
        case=case,
        mask_episode_dir=episode,
        expected_code_revision=revision,
    )

    assert manifest["successful_camera_count"] == len(FROZEN_CAMERA_PANEL)
    assert cameras == tuple(sorted(FROZEN_CAMERA_PANEL))

    with pytest.raises(ValueError, match="artifact changed"):
        validate_dynamic_source_mask_artifact(
            path,
            mask_protocol=mask_protocol,
            case=case,
            mask_episode_dir=episode,
            expected_code_revision="d" * 40,
        )
