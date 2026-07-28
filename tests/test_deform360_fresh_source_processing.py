from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from bayesian_phystwin.deform360_fresh_source_processing import (
    load_fresh_source_processing_protocol,
    validate_fresh_source_mask_artifact,
)
from bayesian_phystwin.deform360_fresh_source_window import (
    FROZEN_CAMERA_PANEL,
    canonical_sha256,
    file_sha256,
    load_fresh_source_mask_protocol,
)

ROOT = Path(__file__).resolve().parents[1]
PROCESSING_PROTOCOL = (
    ROOT / "configs" / "sota" / "deform360_fresh_source_processing_v1.json"
)
MASK_PROTOCOL = ROOT / "configs" / "sota" / "deform360_fresh_source_masks_v1.json"
MASK_CAMPAIGN = (
    ROOT
    / "results"
    / "sota"
    / "deform360_fresh_source_lock_v1"
    / "fresh_source_mask_campaign_v1.json"
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _mask_fixture(tmp_path: Path, *, successful: int = 8) -> tuple[Path, Path]:
    episode = tmp_path / "masks" / "006-fur" / "episode_0000"
    episode.mkdir(parents=True)
    records = []
    for index, camera in enumerate(FROZEN_CAMERA_PANEL):
        row = {"camera": camera}
        if index < successful:
            camera_dir = episode / camera
            camera_dir.mkdir()
            mask = camera_dir / "mask_refined.h5"
            mask.write_bytes(f"mask-{camera}".encode())
            row.update(
                {
                    "status": "success",
                    "mask_sha256": file_sha256(mask),
                    "frame_count": 81,
                }
            )
        else:
            row.update(
                {
                    "status": "technical_failure",
                    "error": "fixture failure",
                }
            )
        records.append(row)
    manifest = {
        "schema_version": 1,
        "artifact_kind": "Deform360FreshSourceMasks",
        "protocol_id": "deform360-fresh-object-pairwise-belief-masks-v1",
        "protocol_config_sha256": (
            "1c530153e693149e3defc54d20c153dbaff2aa26009006f7c7a805ca7db0f67c"
        ),
        "object_id": "006-fur",
        "episode_id": 0,
        "queue_rank": 1,
        "category": "filament",
        "catalog_oid": "fixture",
        "status": (
            "ready_for_source_processing" if successful >= 8 else "technical_failure"
        ),
        "code_revision": "c2c0346606eb09a8171a28c22305eed0239427ad",
        "successful_camera_count": successful,
        "camera_records": records,
    }
    manifest["result_sha256"] = canonical_sha256(
        manifest, digest_key="result_sha256"
    )
    path = episode / "fresh_source_masks.json"
    _write_json(path, manifest)
    return path, episode


def test_processing_protocol_is_locked() -> None:
    protocol = load_fresh_source_processing_protocol(PROCESSING_PROTOCOL)
    assert protocol["camera_policy"]["minimum_camera_count"] == 8
    assert protocol["failure_accounting"]["minimum_final_admissions"] == 12


def test_processing_protocol_rejects_recomputed_change(tmp_path: Path) -> None:
    protocol = json.loads(PROCESSING_PROTOCOL.read_text(encoding="utf-8"))
    protocol["camera_policy"]["minimum_camera_count"] = 7
    protocol["config_sha256"] = canonical_sha256(
        protocol, digest_key="config_sha256"
    )
    path = tmp_path / "changed.json"
    _write_json(path, protocol)
    with pytest.raises(ValueError, match="camera policy"):
        load_fresh_source_processing_protocol(path)


def test_mask_campaign_has_complete_fail_closed_accounting() -> None:
    campaign = json.loads(MASK_CAMPAIGN.read_text(encoding="utf-8"))
    assert campaign["result_sha256"] == canonical_sha256(
        campaign, digest_key="result_sha256"
    )
    assert campaign["case_count"] == 18
    assert campaign["ready_for_source_processing_count"] == 15
    assert campaign["technical_failure_count"] == 3
    assert len(campaign["cases"]) == 18
    assert sorted(row["queue_rank"] for row in campaign["cases"]) == list(
        range(1, 19)
    )
    assert sum(
        row["status"] == "ready_for_source_processing"
        for row in campaign["cases"]
    ) == 15
    assert sum(row["status"] == "technical_failure" for row in campaign["cases"]) == 3


def test_processing_mask_artifact_returns_sorted_successes(tmp_path: Path) -> None:
    path, episode = _mask_fixture(tmp_path, successful=9)
    mask_protocol = load_fresh_source_mask_protocol(MASK_PROTOCOL)
    manifest, cameras = validate_fresh_source_mask_artifact(
        path,
        mask_protocol=mask_protocol,
        case={
            "object_id": "006-fur",
            "episode_id": 0,
            "queue_rank": 1,
            "category": "filament",
            "catalog_oid": "fixture",
        },
        mask_episode_dir=episode,
    )
    assert manifest["successful_camera_count"] == 9
    assert cameras == tuple(sorted(FROZEN_CAMERA_PANEL[:9]))


def test_processing_mask_artifact_rejects_failed_case(tmp_path: Path) -> None:
    path, episode = _mask_fixture(tmp_path, successful=7)
    mask_protocol = load_fresh_source_mask_protocol(MASK_PROTOCOL)
    with pytest.raises(ValueError, match="not processing-ready"):
        validate_fresh_source_mask_artifact(
            path,
            mask_protocol=mask_protocol,
            case={
                "object_id": "006-fur",
                "episode_id": 0,
                "queue_rank": 1,
                "category": "filament",
                "catalog_oid": "fixture",
            },
            mask_episode_dir=episode,
        )


def test_processing_mask_artifact_rejects_changed_mask(tmp_path: Path) -> None:
    path, episode = _mask_fixture(tmp_path, successful=8)
    mask_protocol = load_fresh_source_mask_protocol(MASK_PROTOCOL)
    first = episode / FROZEN_CAMERA_PANEL[0] / "mask_refined.h5"
    first.write_bytes(b"changed")
    with pytest.raises(ValueError, match="frozen source mask changed"):
        validate_fresh_source_mask_artifact(
            path,
            mask_protocol=mask_protocol,
            case={
                "object_id": "006-fur",
                "episode_id": 0,
                "queue_rank": 1,
                "category": "filament",
                "catalog_oid": "fixture",
            },
            mask_episode_dir=episode,
        )


def test_processing_mask_artifact_rejects_changed_manifest(tmp_path: Path) -> None:
    path, episode = _mask_fixture(tmp_path, successful=8)
    payload = json.loads(path.read_text(encoding="utf-8"))
    changed = copy.deepcopy(payload)
    changed["successful_camera_count"] = 9
    _write_json(path, changed)
    mask_protocol = load_fresh_source_mask_protocol(MASK_PROTOCOL)
    with pytest.raises(ValueError, match="artifact changed"):
        validate_fresh_source_mask_artifact(
            path,
            mask_protocol=mask_protocol,
            case={
                "object_id": "006-fur",
                "episode_id": 0,
                "queue_rank": 1,
                "category": "filament",
                "catalog_oid": "fixture",
            },
            mask_episode_dir=episode,
        )
