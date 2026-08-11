from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPAIR = ROOT / (
    "protocols/amendments/"
    "deform360_official_hub_fresh_object_session_v6_"
    "sam2_checkpoint_identity_repair.json"
)
WRAPPER = ROOT / "scripts/ci/run_deform360_v6_source_prediction_evidence.sh"
ARCHIVED_RUNNER = ROOT / (
    "scripts/ci/archive/run_deform360_v6_source_prediction_evidence_v1.sh"
)

REPAIR_ID = "28cee70eaa0e8561a320f87d4e51d6c2aad365927814dc94864e299fc145be99"
ARCHIVED_RUNNER_BLOB_SHA = "9680176e74e933485e1812bf79b626250925ed1a"
PREVIOUS_SHA256 = "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38"
CORRECTED_SHA256 = "7442e4e9b732a508f80e141e7c2913437a3610ee0c77381a66658c3a445df87b"


def _content_id(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _git_blob_sha(path: Path) -> str:
    content = path.read_bytes()
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content).hexdigest()


def test_runtime_repair_is_content_addressed_and_target_closed() -> None:
    payload = json.loads(REPAIR.read_text(encoding="utf-8"))
    declared = payload.pop("repair_id")

    assert declared == REPAIR_ID == _content_id(payload)
    assert payload["schema"] == (
        "bayesian-phystwin.deform360-v6-source-runtime-identity-repair"
    )
    assert payload["schema_version"] == 1
    assert payload["superseded_execution_amendment_id"] == (
        "f8ed525480a6a96265af3cd58e62a96bf1ed748294d0af02aa6386763b993b7f"
    )

    correction = payload["correction"]
    assert correction["field"] == "runtime_sources.sam2_checkpoint_sha256"
    assert correction["checkpoint_model"] == "sam2_hiera_large"
    assert correction["checkpoint_filename"] == "sam2_hiera_large.pt"
    assert correction["previous_sha256"] == PREVIOUS_SHA256
    assert correction["previous_identity"] == "sam2.1_hiera_small.pt"
    assert correction["corrected_sha256"] == CORRECTED_SHA256
    assert correction["corrected_byte_count"] == 897_952_466
    assert correction["repository"] == "facebookresearch/sam2"
    assert correction["repository_revision"] == (
        "2b90b9f5ceec907a1c18123530e92e794ad901a4"
    )

    failed = payload["failed_execution_evidence"]
    assert failed["workflow_run_id"] == 31_456_530_482
    assert failed["artifact_id"] == 9_088_273_849
    assert failed["execution_receipt_id"] == (
        "3159b09724a0e9082bbf0020c38f0c5ec25c8ce3cc08d92f1eb9fa3418c9316d"
    )
    assert failed["prepared_source_object_count"] == 10
    assert failed["prepared_source_stream_count"] == 324
    assert failed["physical_manifest_count"] == 0
    assert failed["source_prediction_seal_count"] == 0

    scope = payload["repair_scope"]
    assert scope["runtime_byte_identity_only"] is True
    assert all(
        scope[field] is False
        for field in (
            "model_family_changed",
            "model_size_changed",
            "repository_revision_changed",
            "source_cohort_changed",
            "camera_panel_changed",
            "candidate_roster_changed",
            "loss_or_gate_changed",
            "replacement_allowed",
            "claim_authorized",
        )
    )
    assert payload["information_boundary"]
    assert not any(payload["information_boundary"].values())
    authorization = payload["execution_authorization"]
    assert authorization["source_prediction_batch_required_before_suffix_access"]
    assert not authorization["fresh_target_selection_authorized"]
    assert not authorization["fresh_target_payload_access_authorized"]


def test_repair_wrapper_preserves_the_reviewed_runner_and_binds_receipts() -> None:
    wrapper = WRAPPER.read_text(encoding="utf-8")
    archived = ARCHIVED_RUNNER.read_text(encoding="utf-8")

    assert _git_blob_sha(ARCHIVED_RUNNER) == ARCHIVED_RUNNER_BLOB_SHA
    assert archived.count(f'SAM2_SHA256="{PREVIOUS_SHA256}"') == 1
    assert CORRECTED_SHA256 not in archived

    assert f'RUNTIME_REPAIR_ID="{REPAIR_ID}"' in wrapper
    assert f'ARCHIVED_RUNNER_BLOB_SHA="{ARCHIVED_RUNNER_BLOB_SHA}"' in wrapper
    assert f'PREVIOUS_SAM2_SHA256="{PREVIOUS_SHA256}"' in wrapper
    assert f'CORRECTED_SAM2_SHA256="{CORRECTED_SHA256}"' in wrapper
    assert "text.count(old) != 1" in wrapper
    assert '"runtime_identity_repair_id"' in wrapper
    assert '"runtime_checkpoint_identity"' in wrapper
    assert "run_deform360_fresh_object_session_source_v6.py" not in wrapper
