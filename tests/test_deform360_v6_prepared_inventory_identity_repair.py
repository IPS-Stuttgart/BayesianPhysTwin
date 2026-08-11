from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACTIVE_RUNNER = ROOT / "scripts/ci/run_deform360_v6_source_prediction_evidence.sh"
SCIENCE_RUNNER = ROOT / (
    "scripts/ci/archive/run_deform360_v6_source_prediction_evidence_v2.sh"
)
SELECTOR_WRAPPER = ROOT / (
    "scripts/ci/archive/"
    "run_deform360_v6_source_prediction_evidence_selector_repair_v3.sh"
)
INVENTORY_WRAPPER = ROOT / (
    "scripts/ci/archive/"
    "run_deform360_v6_source_prediction_evidence_inventory_repair_v4.sh"
)
REPAIR = ROOT / (
    "protocols/amendments/"
    "deform360_official_hub_fresh_object_session_v6_"
    "prepared_inventory_identity_repair.json"
)
SOURCE_LOCK = ROOT / (
    "protocols/locks/deform360_official_hub_joint_sparse_source_execution_v5.json"
)

REPAIR_ID = "bd9b1b9e37529c7a7e555ff8ec7e62521bece77fe8554b33047b1d33a2de7fa4"
INVENTORY_REVISION = "e190c94014e6024e324d860618662526af6ea682"
INVENTORY_ID = "6994aa621b38dc8fb21cd38e43363bde3ea12dd644532addeecfc07a30f84e7b"
INVENTORY_FILE_SHA256 = (
    "4da96c4f636d195f7aea5d971fbd83bd3b0f35b1c66a77af68007bbd08a69007"
)


def _git_blob_sha(path: Path) -> str:
    payload = path.read_bytes()
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def test_prepared_inventory_repair_is_content_addressed_and_target_closed() -> None:
    payload = json.loads(REPAIR.read_text(encoding="utf-8"))
    declared = payload.pop("repair_id")
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    assert declared == REPAIR_ID == hashlib.sha256(canonical).hexdigest()
    assert payload["correction"] == {
        "authoritative_artifact_id": 9026043628,
        "authoritative_artifact_sha256": (
            "d0041af0ba0cfe6e5c5bd4008c47adb3ed4cf0cf0f6754eff67a238e746c7a86"
        ),
        "authoritative_workflow_run_id": 31272512658,
        "corrected_revision": INVENTORY_REVISION,
        "expected_file_sha256": INVENTORY_FILE_SHA256,
        "expected_inventory_id": INVENTORY_ID,
        "field": "prepared_source_inventory.implementation_revision",
        "previous_runtime_value": "current-protected-main-source-revision",
    }
    assert payload["failed_execution_evidence"] == {
        "artifact_id": 9090402942,
        "artifact_sha256": (
            "edbfe17c5c85c4e9f7375026fd15593132ee748456a60a5c97b2ea87776a3304"
        ),
        "execution_receipt_id": (
            "a62ead70994b330d3eecf8d35afe6baa1275c428f4c90c4e18157aac55f3733f"
        ),
        "observed_file_sha256": (
            "7714c5d4b0aaed32358f21d133a84ab038d2f98eb1c48639d43df736ae801acf"
        ),
        "observed_inventory_id": (
            "3972737207fd684e5e31cd507c4b3a0e9e2d0ed1b9d5cd427774f019b2b704cc"
        ),
        "physical_manifest_count": 0,
        "source_prediction_seal_count": 0,
        "terminal_stage": "physical-source:026-sock-cloth-ep0007",
        "workflow_run_id": 31462653379,
    }
    assert payload["repair_scope"]["runtime_identity_replay_only"] is True
    assert all(
        value is False
        for key, value in payload["repair_scope"].items()
        if key != "runtime_identity_replay_only"
    )
    assert not any(payload["information_boundary"].values())
    assert payload["execution_authorization"] == {
        "event": "push-to-protected-main-after-reviewed-merge",
        "fresh_target_payload_access_authorized": False,
        "fresh_target_selection_authorized": False,
        "runner_name": "workstation2",
        "source_prediction_batch_required_before_suffix_access": True,
    }


def test_repair_replays_the_inventory_identity_bound_by_the_source_lock() -> None:
    lock = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))
    bound = lock["physical_baseline"]["prepared_source_inventory"]

    assert bound == {
        "file_sha256": INVENTORY_FILE_SHA256,
        "inventory_id": INVENTORY_ID,
    }
    wrapper = INVENTORY_WRAPPER.read_text(encoding="utf-8")
    science = SCIENCE_RUNNER.read_text(encoding="utf-8")
    assert science.count('--implementation-revision "${BPT_SOURCE_SHA}"') == 1
    assert f'PREPARED_INVENTORY_REVISION="{INVENTORY_REVISION}"' in wrapper
    assert f'PREPARED_INVENTORY_ID="{INVENTORY_ID}"' in wrapper
    assert f'PREPARED_INVENTORY_FILE_SHA256="{INVENTORY_FILE_SHA256}"' in wrapper
    assert "patched.count(inventory_old) != 1" in wrapper
    assert "patched.count(inventory_new) != 1" in wrapper
    assert "replayed prepared-source inventory file identity changed" in wrapper
    assert "replayed prepared-source inventory content identity changed" in wrapper


def test_active_runner_binds_exact_repair_wrapper_bytes() -> None:
    active = ACTIVE_RUNNER.read_text(encoding="utf-8")
    selector = SELECTOR_WRAPPER.read_text(encoding="utf-8")
    inventory = INVENTORY_WRAPPER.read_text(encoding="utf-8")

    assert (
        'SELECTOR_WRAPPER="scripts/ci/archive/'
        'run_deform360_v6_source_prediction_evidence_inventory_repair_v4.sh"' in active
    )
    assert f'SELECTOR_WRAPPER_BLOB_SHA="{_git_blob_sha(INVENTORY_WRAPPER)}"' in active
    assert (
        f'SELECTOR_WRAPPER_V3_BLOB_SHA="{_git_blob_sha(SELECTOR_WRAPPER)}"' in inventory
    )
    assert "runtime_prepared_source_inventory_identity_repair_id" in inventory
    assert "runtime_prepared_source_inventory" in inventory
    assert 'development_suffix_opened": False' in inventory
    assert 'v6_target_payloads_opened": False' in inventory
    assert 'fresh_target_selection_authorized": False' in inventory
    assert (
        'ARCHIVED_RUNNER_BLOB_SHA="42dd4f3e0d05f18b9ff0a0bdcf90fbd282f0f6f1"'
        in selector
    )


def test_repair_shells_are_syntactically_valid() -> None:
    for path in (ACTIVE_RUNNER, SELECTOR_WRAPPER, INVENTORY_WRAPPER):
        subprocess.run(["bash", "-n", str(path)], cwd=ROOT, check=True)
