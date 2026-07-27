from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.deform360_dynamic_tapnextpp_source_window import (
    FROZEN_CAMERA_PANEL,
    PROTOCOL_ID,
    RAW_FRAME_COUNT,
    STAGE_KIND,
    canonical_sha256,
    load_dynamic_source_mask_protocol,
    load_dynamic_source_window_protocol,
    select_fresh_source_window,
    validate_dynamic_source_preparation,
    validate_dynamic_source_window_stage,
    validate_dynamic_window_sources,
)
from bayesian_phystwin.deform360_fresh_source_window import (
    select_fresh_source_window as select_legacy_window,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT / "configs" / "sota" / "deform360_dynamic_tapnextpp_source_window_v1.json"
)
QUEUE = ROOT / "configs" / "sota" / "deform360_dynamic_tapnextpp_staging_queue_v1.json"
DOWNLOAD = (
    ROOT
    / "results"
    / "sota"
    / "deform360_dynamic_tapnextpp_provider_v1"
    / "source_download_manifest_v1.json"
)
MASK_PROTOCOL = (
    ROOT / "configs" / "sota" / "deform360_dynamic_tapnextpp_source_masks_v1.json"
)


def test_dynamic_source_protocol_binds_download_inventory() -> None:
    protocol, queue, download = validate_dynamic_window_sources(
        PROTOCOL,
        QUEUE,
        DOWNLOAD,
    )

    assert protocol["protocol_id"] == PROTOCOL_ID
    assert len(queue["candidates"]) == 36
    assert download["object_count"] == 36
    assert sum(row["file_count"] for row in download["objects"]) == 2820
    assert sum(row["total_bytes"] for row in download["objects"]) == 3_192_349_000


def test_dynamic_window_selector_preserves_frozen_behavior() -> None:
    actions = np.zeros((180, 5, 3), dtype=np.float64)
    actions[:, 1:4, :] = np.eye(3)[None]
    actions[40:90, 0, 0] = np.arange(50, dtype=np.float64)
    actions[90:, 0, 0] = 49.0
    openings = np.zeros(180, dtype=np.float64)

    assert select_fresh_source_window(
        actions,
        openings,
    ) == select_legacy_window(actions, openings)


def test_dynamic_window_protocol_rejects_recomputed_tampering(
    tmp_path: Path,
) -> None:
    changed = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    changed["source_bindings"]["download"]["object_count"] = 35
    changed["config_sha256"] = canonical_sha256(
        changed,
        digest_key="config_sha256",
    )
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ValueError, match="download binding changed"):
        load_dynamic_source_window_protocol(path)


def test_dynamic_mask_protocol_binds_window_execution_commit() -> None:
    window = load_dynamic_source_window_protocol(PROTOCOL)
    mask = load_dynamic_source_mask_protocol(MASK_PROTOCOL)

    assert mask["parent_window_protocol"]["config_sha256"] == window["config_sha256"]
    assert (
        mask["parent_window_protocol"]["implementation_commit"]
        == "a55739635514177871a3f64728e979a69e1ab5df"
    )
    assert mask["mask_contract"]["minimum_successful_cameras"] == 8
    assert mask["generic_selector"]["manual_prompting"] is False


def test_dynamic_window_stage_binds_execution_commit(tmp_path: Path) -> None:
    protocol = load_dynamic_source_window_protocol(PROTOCOL)
    case = {
        "queue_rank": 1,
        "object_id": "025-bag-small-cloth",
        "catalog_oid": "a" * 40,
        "episode_id": 0,
        "category": "sheet",
        "metadata_sha256": "b" * 64,
        "bimanual": "no",
    }
    revision = "c" * 40
    stage: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": STAGE_KIND,
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": protocol["config_sha256"],
        **case,
        "code_revision": revision,
        "staged_frame_count": RAW_FRAME_COUNT,
        "camera_count": len(FROZEN_CAMERA_PANEL),
        "camera_records": [
            {
                "camera": camera,
                "decoded_frame_count": RAW_FRAME_COUNT,
            }
            for camera in FROZEN_CAMERA_PANEL
        ],
        "information_boundary": {
            "known_future_action_read": True,
            "object_rgb_materialized_after_selection_seal_built": True,
            "object_geometry_read": False,
            "object_tracks_read": False,
            "object_response_used_for_window_selection": False,
            "tactile_read": False,
            "target_metric_read": False,
        },
    }
    stage["result_sha256"] = canonical_sha256(
        stage,
        digest_key="result_sha256",
    )
    path = tmp_path / "stage.json"
    path.write_text(json.dumps(stage), encoding="utf-8")

    assert (
        validate_dynamic_source_window_stage(
            path,
            window_protocol=protocol,
            case=case,
            expected_code_revision=revision,
        )["result_sha256"]
        == stage["result_sha256"]
    )

    with pytest.raises(ValueError, match="stage changed"):
        validate_dynamic_source_window_stage(
            path,
            window_protocol=protocol,
            case=case,
            expected_code_revision="d" * 40,
        )


def test_preparation_accepts_legacy_boolean_bimanual_field() -> None:
    protocol = load_dynamic_source_window_protocol(PROTOCOL)
    case = {
        "queue_rank": 1,
        "object_id": "025-bag-small-cloth",
        "catalog_oid": "a" * 40,
        "episode_id": 0,
        "category": "sheet",
        "metadata_sha256": "b" * 64,
        "bimanual": "no",
    }
    revision = "c" * 40
    manifest = {
        "schema_version": 1,
        "artifact_kind": "Deform360DynamicTapNextppSourcePreparation",
        "protocol_config_sha256": protocol["config_sha256"],
        **case,
        "bimanual": False,
        "code_revision": revision,
    }
    manifest["result_sha256"] = canonical_sha256(
        manifest,
        digest_key="result_sha256",
    )

    validate_dynamic_source_preparation(
        manifest,
        window_protocol=protocol,
        case=case,
        expected_code_revision=revision,
    )

    manifest["bimanual"] = True
    manifest["result_sha256"] = canonical_sha256(
        manifest,
        digest_key="result_sha256",
    )
    with pytest.raises(ValueError, match="bimanual preparation"):
        validate_dynamic_source_preparation(
            manifest,
            window_protocol=protocol,
            case=case,
            expected_code_revision=revision,
        )
