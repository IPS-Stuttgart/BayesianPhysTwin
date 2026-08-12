from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bayesian_phystwin._portable_contracts import content_id

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / (
    "protocols/amendments/"
    "deform360_official_hub_fresh_object_session_v6_"
    "case_stdin_marker_bootstrap_binding.json"
)
WORKFLOW = ROOT / (
    ".github/workflows/deform360-v6-source-prediction-evidence-dual-runtime.yml"
)


def test_case_stdin_marker_bootstrap_binding_is_content_addressed() -> None:
    payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    declared = payload.pop("repair_id")

    assert declared == content_id(payload)
    assert not any(payload["information_boundary"].values())
    assert payload["failed_execution_evidence"] == {
        "artifact_digest": (
            "sha256:400aa520c3f2084154c31fdfff1159e884cbc660f74e51ae113725b313d60631"
        ),
        "artifact_id": 9129246543,
        "error_message": "BPT_CASE_STDIN_ISOLATION_MARKER is required",
        "execution_receipt_id": (
            "bb525151235354c83d0a791dcfef849d03ddabe09e5bdf3bfb5181972c35ddfd"
        ),
        "exit_code": 1,
        "physical_manifest_count": 0,
        "runtime_case_subprocess_stdin_isolation_activated": False,
        "source_prediction_seal_count": 0,
        "source_revision": "2350d1bde3b1a7b20a3602e7fd08f01aecab882b",
        "terminal_stage": "build-isolated-primary-and-frame-zero-runtimes",
        "workflow_run_attempt": 1,
        "workflow_run_id": 31565059126,
    }
    assert payload["repair_scope"]["case_stdin_marker_bootstrap_binding_completed"]
    assert all(
        value is False
        for key, value in payload["repair_scope"].items()
        if key != "case_stdin_marker_bootstrap_binding_completed"
    )


def test_workflow_binds_marker_before_first_dispatcher_probe() -> None:
    payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    workflow = WORKFLOW.read_text(encoding="utf-8")
    amendment_digest = hashlib.sha256(AMENDMENT.read_bytes()).hexdigest()

    assert f"CASE_STDIN_MARKER_BOOTSTRAP_REPAIR_ID: {payload['repair_id']}" in workflow
    assert f"CASE_STDIN_MARKER_BOOTSTRAP_REPAIR_SHA256: {amendment_digest}" in workflow
    assert str(AMENDMENT.relative_to(ROOT)) in workflow
    probe = (
        'BPT_OFFICIAL_PHYSTWIN_RUNTIME_MARKER="${official_runtime_marker}" \\\n'
        '            BPT_CASE_STDIN_ISOLATION_MARKER="${case_stdin_isolation_marker}" \\\n'
        "            \"${dispatcher}\" - <<'PY'"
    )
    assert probe in workflow
    assert workflow.count("\"${dispatcher}\" - <<'PY'") == 1
    assert workflow.count('test ! -e "${case_stdin_isolation_marker}"') == 2
    assert '"runtime_case_stdin_marker_bootstrap_binding"' in workflow
