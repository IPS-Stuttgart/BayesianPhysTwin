from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPAIR = ROOT / (
    "protocols/amendments/"
    "deform360_official_hub_fresh_object_session_v6_"
    "generic_selector_identity_repair.json"
)
WRAPPER = ROOT / "scripts/ci/run_deform360_v6_source_prediction_evidence.sh"
ARCHIVED_RUNNER = ROOT / (
    "scripts/ci/archive/"
    "run_deform360_v6_source_prediction_evidence_v2.sh"
)

REPAIR_ID = "d7e516ced90469589c3e4c3c12672a503fe8bbdb3a6f3316d852c266fd0f3d90"
ARCHIVED_RUNNER_BLOB_SHA = "42dd4f3e0d05f18b9ff0a0bdcf90fbd282f0f6f1"
PREVIOUS_SHA256 = "79b161fa66489f75b5b078c7ae409387feed74c51a38b86e89800d0aa578b1df"
CORRECTED_SHA256 = "c10391578c73dde47fbce160312559a7e638007e9053ec89373fe575cc64d7e5"


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


def test_selector_repair_is_content_addressed_and_target_closed() -> None:
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
    assert correction == {
        "corrected_byte_count": 17_310,
        "corrected_sha256": CORRECTED_SHA256,
        "field": "runtime_sources.generic_selector_source_sha256",
        "path": "src/causal4d_public/deform360_object_sam2.py",
        "previous_sha256": PREVIOUS_SHA256,
        "repository": "IPS-Stuttgart/Causal4D",
        "repository_revision": (
            "50e3682a5dbf976b20cc9115b6e7a975d0144ea5"
        ),
        "selector_semantics": "deform360-object-sam2-generic-selector",
    }

    failed = payload["failed_execution_evidence"]
    assert failed["workflow_run_id"] == 31_458_096_956
    assert failed["artifact_id"] == 9_088_797_337
    assert failed["artifact_digest"] == (
        "sha256:"
        "4438365b1664020f5398dfc1b6bdcd749b7499c8c1168ef62ca0fa49cb95d63a"
    )
    assert failed["execution_receipt_id"] == (
        "cfcfeab74ee9cc88002e398afa2655ccc1a56752787fe6b44a961061fb7cd040"
    )
    assert failed["prepared_source_object_count"] == 10
    assert failed["prepared_source_stream_count"] == 324
    assert failed["physical_manifest_count"] == 0
    assert failed["source_prediction_seal_count"] == 0

    probe = payload["diagnostic_probe"]
    assert probe["workflow_run_id"] == 31_458_663_573
    assert probe["complete_history_searched"] is True
    assert probe["historical_match_found"] is False
    assert probe["observed_sha256"] == CORRECTED_SHA256
    assert probe["observed_byte_count"] == 17_310

    scope = payload["repair_scope"]
    assert scope["runtime_byte_identity_only"] is True
    assert all(
        scope[field] is False
        for field in (
            "model_family_changed",
            "model_size_changed",
            "repository_revision_changed",
            "selector_semantics_changed",
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
    assert authorization["runner_name"] == "workstation2"
    assert authorization["source_prediction_batch_required_before_suffix_access"]
    assert not authorization["fresh_target_selection_authorized"]
    assert not authorization["fresh_target_payload_access_authorized"]


def test_wrapper_patches_only_the_stale_selector_identity() -> None:
    wrapper = WRAPPER.read_text(encoding="utf-8")
    archived = ARCHIVED_RUNNER.read_text(encoding="utf-8")

    assert _git_blob_sha(ARCHIVED_RUNNER) == ARCHIVED_RUNNER_BLOB_SHA
    assert archived.count(f'SELECTOR_SHA256="{PREVIOUS_SHA256}"') == 1
    assert CORRECTED_SHA256 not in archived

    assert f'REPAIR_ID="{REPAIR_ID}"' in wrapper
    assert f'ARCHIVED_RUNNER_BLOB_SHA="{ARCHIVED_RUNNER_BLOB_SHA}"' in wrapper
    assert f'PREVIOUS_SELECTOR_SHA256="{PREVIOUS_SHA256}"' in wrapper
    assert f'CORRECTED_SELECTOR_SHA256="{CORRECTED_SHA256}"' in wrapper
    assert "text.count(old) != 1" in wrapper
    assert "patched.count(new) != 1" in wrapper
    assert '"runtime_identity_repair_id"' in wrapper
    assert '"runtime_selector_identity"' in wrapper
    assert "historical_match_revision" not in wrapper
    assert "run_deform360_fresh_object_session_source_v6.py" not in wrapper
