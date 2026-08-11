from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.deform360_bias_aware_prospective_physical import (
    UPSTREAM_FILE_SHA256,
)

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / (
    "protocols/amendments/"
    "deform360_official_hub_fresh_object_session_v6_"
    "frozen_upstream_materialization.json"
)
ACTIVE_RUNNER = ROOT / "scripts/ci/run_deform360_v6_source_prediction_evidence.sh"
SELECTOR_RUNNER = ROOT / (
    "scripts/ci/archive/"
    "run_deform360_v6_source_prediction_evidence_selector_repair_v1.sh"
)
MATERIALIZATION_ID = "2056084bd44845446f78600ca42edd8fb23b4003431c87d53ff8d73a5dc275c0"
FROZEN_REVISION = "9f69d5d6c5d81d6d6e8f123c18ddba73dc4afa65"
SELECTOR_RUNNER_BLOB_SHA = "5958db6362917e6bc355b194abdac4736e39a5a4"
LOCATOR_REPORT_ID = "75c1be85233e1835dfef5a1227a28e8938995335ead701fe8d3dfd8b5960a087"


def _git_blob_sha(path: Path) -> str:
    content = path.read_bytes()
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content).hexdigest()


def test_materialization_amendment_is_content_addressed_and_target_closed() -> None:
    payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    declared = payload.pop("amendment_id")

    assert declared == MATERIALIZATION_ID == content_id(payload)
    assert payload["schema"] == (
        "bayesian-phystwin.deform360-v6-frozen-upstream-materialization"
    )
    assert payload["schema_version"] == 1

    parent = payload["parent_execution"]
    assert parent["execution_amendment_id"] == (
        "f8ed525480a6a96265af3cd58e62a96bf1ed748294d0af02aa6386763b993b7f"
    )
    assert parent["selector_runtime_repair_id"] == (
        "d7e516ced90469589c3e4c3c12672a503fe8bbdb3a6f3316d852c266fd0f3d90"
    )
    assert parent["protected_workflow_run_id"] == 31_460_025_917
    assert parent["protected_artifact_id"] == 9_089_464_088
    assert parent["protected_execution_receipt_id"] == (
        "6232968ec10c630b62e6933783b278bc4aba2362bb29c1eec6d2be47a001c0e0"
    )
    assert parent["terminal_status"] == "source-inputs-incomplete"
    assert parent["terminal_stage"] == "locate-frozen-physical-upstream"
    assert parent["physical_manifest_count"] == 0
    assert parent["source_prediction_seal_count"] == 0

    locator = payload["history_locator"]
    assert locator["report_id"] == LOCATOR_REPORT_ID
    assert locator["workflow_run_id"] == 31_461_017_011
    assert locator["artifact_id"] == 9_089_783_219
    assert locator["complete_history_searched"] is True
    assert locator["candidate_commit_count"] == 12
    assert locator["anchor_match_count"] == 3
    assert locator["exact_match_count"] == 1

    source = payload["frozen_source"]
    assert source["repository"] == "IPS-Stuttgart/BayesianPhysTwin"
    assert source["revision"] == FROZEN_REVISION
    assert source["refs_pointing_at"] == []
    assert source["containing_tags"] == []
    assert source["required_file_sha256"] == dict(sorted(UPSTREAM_FILE_SHA256.items()))

    assert payload["materialization"] == {
        "all_required_files_must_match_sha256": True,
        "archived_runner_blob_may_be_modified": False,
        "main_checkout_files_may_be_modified": False,
        "method": "detached-temporary-git-worktree",
        "runtime_discovery_root_extension_only": True,
        "source_revision_must_match_exactly": True,
        "symlinked_required_files_allowed": False,
        "temporary_worktree_removed_after_execution": True,
    }
    assert payload["repair_scope"]["historical_source_materialization_only"]
    assert not any(
        value
        for key, value in payload["repair_scope"].items()
        if key != "historical_source_materialization_only"
    )
    assert not any(payload["information_boundary"].values())
    authorization = payload["execution_authorization"]
    assert authorization["runner_name"] == "workstation2"
    assert authorization["source_prediction_batch_required_before_suffix_access"]
    assert not authorization["fresh_target_selection_authorized"]
    assert not authorization["fresh_target_payload_access_authorized"]


def test_active_runner_wraps_exact_selector_runner_without_mutating_it() -> None:
    active = ACTIVE_RUNNER.read_text(encoding="utf-8")
    selector = SELECTOR_RUNNER.read_text(encoding="utf-8")

    assert _git_blob_sha(SELECTOR_RUNNER) == SELECTOR_RUNNER_BLOB_SHA
    assert (
        'SELECTOR_RUNNER_BLOB_SHA="5958db6362917e6bc355b194abdac4736e39a5a4"'
        in active
    )
    assert f'MATERIALIZATION_ID="{MATERIALIZATION_ID}"' in active
    assert f'FROZEN_UPSTREAM_REVISION="{FROZEN_REVISION}"' in active
    assert f'LOCATOR_REPORT_ID="{LOCATOR_REPORT_ID}"' in active
    assert 'git worktree add --detach "${FROZEN_UPSTREAM_ROOT}"' in active
    assert 'git worktree remove --force "${FROZEN_UPSTREAM_ROOT}"' in active
    assert 'export RUNNER_WORKSPACE="${WORKTREE_PARENT}"' in active
    assert 'bash "${SELECTOR_RUNNER}"' in active
    assert "runtime_frozen_upstream_materialization" in active
    assert "required_file_sha256" in active
    assert "git checkout" not in active
    assert "git reset" not in active
    assert "git clean" not in active

    assert "runtime_identity_repair_id" in selector
    assert "runtime_selector_identity" in selector
    assert "historical_match_revision" not in selector
    assert "run_deform360_fresh_object_session_source_v6.py" not in selector


def test_materialization_receipt_preserves_exact_hash_roster() -> None:
    text = ACTIVE_RUNNER.read_text(encoding="utf-8")

    assert "UPSTREAM_FILE_SHA256" in text
    assert "if required != dict(sorted(UPSTREAM_FILE_SHA256.items())):" in text
    assert "frozen upstream file is missing or symlinked" in text
    assert "frozen upstream file escapes worktree" in text
    assert "frozen upstream file identity changed" in text
    assert '"main_checkout_modified": False' in text
    assert '"dataset_opened": False' in text
    assert '"source_residual_opened": False' in text
    assert '"development_suffix_opened": False' in text
    assert '"target_payload_opened": False' in text
    assert '"target_outcome_used": False' in text
