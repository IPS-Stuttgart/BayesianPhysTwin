from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bayesian_phystwin._portable_contracts import content_id

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / (
    "protocols/amendments/"
    "deform360_official_hub_fresh_object_session_v6_"
    "frozen_prepared_inventory_reuse.json"
)
SHIM = ROOT / "scripts/ci/frozen_prepared_inventory_python_shim.sh"
WORKFLOW = ROOT / ".github/workflows/deform360-v6-frozen-inventory-resume.yml"
AMENDMENT_ID = "1d4087e22d7c7cd3fcec09c6f392427c90c0eaa5adbb8d12e35b89e215dd5ed9"
FILE_SHA256 = "4da96c4f636d195f7aea5d971fbd83bd3b0f35b1c66a77af68007bbd08a69007"
INVENTORY_ID = "6994aa621b38dc8fb21cd38e43363bde3ea12dd644532addeecfc07a30f84e7b"


def _blob_sha(path: Path) -> str:
    content = path.read_bytes()
    return hashlib.sha1(
        f"blob {len(content)}\0".encode("ascii") + content
    ).hexdigest()


def test_reuse_amendment_is_content_addressed_and_target_closed() -> None:
    payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    declared = payload.pop("amendment_id")

    assert declared == AMENDMENT_ID == content_id(payload)
    assert payload["schema"] == (
        "bayesian-phystwin.deform360-v6-frozen-prepared-inventory-reuse"
    )
    assert payload["schema_version"] == 1
    assert payload["claim_authorized"] is False
    assert not any(payload["information_boundary"].values())

    frozen = payload["frozen_inventory"]
    assert frozen["workflow_run_id"] == 31_272_512_658
    assert frozen["artifact_id"] == 9_026_043_628
    assert frozen["file_sha256"] == FILE_SHA256
    assert frozen["inventory_id"] == INVENTORY_ID
    assert frozen["object_count"] == 10

    parent = payload["parent_execution"]
    assert parent["workflow_run_id"] == 31_462_653_379
    assert parent["artifact_id"] == 9_090_402_942
    assert parent["physical_manifest_count"] == 0
    assert parent["source_prediction_seal_count"] == 0
    assert parent["terminal_stage"] == "physical-source:026-sock-cloth-ep0007"

    comparison = payload["semantic_comparison"]
    assert comparison["differing_paths"] == [
        "implementation_revision",
        "inventory_id",
    ]
    assert comparison["all_object_records_identical"]
    assert comparison["all_source_artifact_hashes_identical"]
    assert comparison["all_selection_and_provider_locks_identical"]
    assert comparison["all_information_boundary_fields_identical"]
    assert not comparison["cohort_or_payload_change_detected"]


def test_shim_intercepts_only_the_inventory_builder() -> None:
    text = SHIM.read_text(encoding="utf-8")

    assert f'EXPECTED_FILE_SHA256="{FILE_SHA256}"' in text
    assert f'EXPECTED_INVENTORY_ID="{INVENTORY_ID}"' in text
    assert f'EXPECTED_AMENDMENT_ID="{AMENDMENT_ID}"' in text
    assert (
        'INVENTORY_COMMAND="scripts/science/'
        'inventory_deform360_calibration_prepared_source.py"' in text
    )
    assert 'exec "${REAL_BPT_PYTHON}" "$@"' in text
    assert 'if [[ "${1:-}" != "${INVENTORY_COMMAND}" ]]' in text
    assert 'test ! -e "${destination}"' in text
    assert 'cp --reflink=auto --preserve=mode,timestamps' in text
    assert "all_object_records_identical" in text
    assert "cohort_or_payload_change_detected" in text
    assert "--implementation-revision" not in text


def test_resume_workflow_downloads_exact_artifact_and_preserves_runners() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "31272512658" in text
    assert "deform360-calibration-retained-source-admission-31272512658-1" in text
    assert "frozen-prepared-inventory/prepared-source-inventory.json" in text
    assert "frozen_prepared_inventory_python_shim.sh" in text
    assert "REAL_BPT_PYTHON" in text
    assert "FROZEN_PREPARED_INVENTORY" in text
    assert "run_deform360_v6_source_prediction_evidence.sh" in text
    assert "workstation2" in text
    assert "workflow_dispatch:" not in text
    assert "contents: write" not in text
    assert "git push" not in text
    assert _blob_sha(SHIM)
