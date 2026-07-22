from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from bayesian_phystwin import deform360_bias_aware_prospective_artifacts as artifacts
from bayesian_phystwin.deform360_bias_aware_prospective_protocol import (
    PROTOCOL_ID as V1_PROTOCOL_ID,
)
from bayesian_phystwin.deform360_bias_aware_prospective_v2_protocol import (
    PROTOCOL_ID as V2_PROTOCOL_ID,
)
from bayesian_phystwin.deform360_bias_aware_prospective_v2_runtime import (
    EXECUTION_LOCK_ARTIFACT_KIND,
    activate_v2_prediction_runtime,
    load_bias_aware_prospective_v2_execution_protocol,
    patch_v2_stage_module,
    prospective_v2_case_record,
    prospective_v2_case_records,
    validate_v2_execution_lock,
    validate_v2_fresh_download_manifest,
)
from bayesian_phystwin.deform360_bias_aware_prospective_artifacts import (
    canonical_sha256,
    file_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "configs/sota/deform360_bias_aware_guarded_belief_prospective_v2.json"
)


def test_v2_execution_protocol_normalizes_the_combined_cohort() -> None:
    protocol = load_bias_aware_prospective_v2_execution_protocol(PROTOCOL)
    assert protocol["config"]["protocol_id"] == V2_PROTOCOL_ID
    assert len(prospective_v2_case_records(PROTOCOL, role="calibration")) == 12
    assert len(prospective_v2_case_records(PROTOCOL, role="target")) == 24
    assert prospective_v2_case_record(
        PROTOCOL, object_id="078-fishing-line", episode_id=4
    ) == {
        "case": "078-fishing-line-ep0004",
        "object_id": "078-fishing-line",
        "episode_id": 4,
        "episode_key": "078-fishing-line/4",
        "stratum": "filament",
        "role": "calibration",
    }


def test_v2_runtime_is_process_local_and_restores_v1() -> None:
    assert artifacts.PROTOCOL_ID == V1_PROTOCOL_ID
    with activate_v2_prediction_runtime():
        assert artifacts.PROTOCOL_ID == V2_PROTOCOL_ID
        assert artifacts.prospective_case_record(
            PROTOCOL, object_id="161-tube", episode_id=4
        )["case"] == "161-tube-ep0004"
    assert artifacts.PROTOCOL_ID == V1_PROTOCOL_ID


def test_prepare_stage_patch_uses_fresh_v2_authorization(tmp_path: Path) -> None:
    module = SimpleNamespace(
        PROTOCOL_ID=V1_PROTOCOL_ID,
        load_bias_aware_prospective_protocol=None,
        prospective_case_record=None,
        bias_aware_prospective_download_plan=None,
        _validate_download_manifest=None,
    )
    patch_v2_stage_module(
        module,
        stage="prepare-source",
        repository=ROOT,
        execution_lock=tmp_path / "lock.json",
    )
    assert module.PROTOCOL_ID == V2_PROTOCOL_ID
    plan = module.bias_aware_prospective_download_plan(PROTOCOL)
    assert plan.object_ids == ("078-fishing-line", "161-tube", "088-snake")


def test_v2_download_manifest_validator_rejects_open_target_flag(
    tmp_path: Path,
) -> None:
    metadata = tmp_path / "metadata.json"
    metadata.write_text("{}\n", encoding="utf-8")
    protocol = load_bias_aware_prospective_v2_execution_protocol(PROTOCOL)
    payload = {
        "artifact_kind": "Deform360BiasAwareProspectiveV2FreshDownload",
        "protocol_id": V2_PROTOCOL_ID,
        "protocol_config_sha256": protocol["config_sha256"],
        "revision": protocol["config"]["dataset"]["revision"],
        "objects": [
            {
                "object_id": "078-fishing-line",
                "selected_episode_ids": [4],
                "metadata_sha256": file_sha256(metadata),
            }
        ],
        "information_boundary": {
            "fresh_calibration_objects_only": True,
            "fresh_calibration_future_opened": False,
            "reserved_target_downloaded": False,
            "reserved_target_media_read": False,
            "target_metrics_opened": False,
        },
    }
    payload["manifest_sha256"] = canonical_sha256(
        payload, digest_key="manifest_sha256"
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    validate_v2_fresh_download_manifest(
        manifest,
        protocol_config_sha256=protocol["config_sha256"],
        object_id="078-fishing-line",
        episode_id=4,
        metadata_path=metadata,
    )
    payload["information_boundary"]["reserved_target_media_read"] = True
    payload["manifest_sha256"] = canonical_sha256(
        payload, digest_key="manifest_sha256"
    )
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    try:
        validate_v2_fresh_download_manifest(
            manifest,
            protocol_config_sha256=protocol["config_sha256"],
            object_id="078-fishing-line",
            episode_id=4,
            metadata_path=metadata,
        )
    except ValueError as error:
        assert "information boundary" in str(error)
    else:
        raise AssertionError("open target flag was accepted")


def test_execution_lock_binds_all_declared_files(tmp_path: Path) -> None:
    source = tmp_path / "bound.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    payload = {
        "artifact_kind": EXECUTION_LOCK_ARTIFACT_KIND,
        "protocol_id": V2_PROTOCOL_ID,
        "adapter_lock_commit": "not-checked-in-unit-test",
        "files_sha256": {"bound.py": file_sha256(source)},
        "information_boundary": {
            "outcome_loader_installed": False,
            "calibration_future_access_authorized": False,
            "target_access_authorized": False,
        },
    }
    payload["config_sha256"] = canonical_sha256(payload, digest_key="config_sha256")
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps(payload), encoding="utf-8")
    validate_v2_execution_lock(
        lock, repository=tmp_path, require_clean_repository=False
    )
    source.write_text("VALUE = 2\n", encoding="utf-8")
    try:
        validate_v2_execution_lock(
            lock, repository=tmp_path, require_clean_repository=False
        )
    except ValueError as error:
        assert "execution file changed" in str(error)
    else:
        raise AssertionError("changed execution file was accepted")
