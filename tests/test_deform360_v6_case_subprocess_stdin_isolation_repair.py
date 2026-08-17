from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bayesian_phystwin._portable_contracts import content_id

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / (
    "protocols/amendments/"
    "deform360_official_hub_fresh_object_session_v6_"
    "case_subprocess_stdin_isolation.json"
)
WORKFLOW = ROOT / (
    ".github/workflows/deform360-v6-source-prediction-evidence-dual-runtime.yml"
)
DISPATCHER = ROOT / "scripts/ci/dispatch_deform360_v6_source_python.sh"


def test_case_subprocess_stdin_isolation_repair_is_content_addressed() -> None:
    payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    declared = payload.pop("repair_id")

    assert declared == content_id(payload)
    assert not any(payload["information_boundary"].values())
    assert payload["failed_execution_evidence"] == {
        "artifact_digest": (
            "sha256:4830a19fd2521e293bbe67e4904bf247627f5410f3baf80e87da793cb51c94c9"
        ),
        "artifact_id": 9127039980,
        "diagnosis": (
            "A per-case child inherited the object-roster stream on standard input. "
            "After the first complete physical manifest, the next shell read received "
            "only a tail fragment and the materializer received an empty --object-id "
            "value."
        ),
        "execution_receipt_id": (
            "308a46ef7fbe1c05098d5df40c7f387400e3367f9b1d2b26a78efc4ff0336dc0"
        ),
        "exit_code": 2,
        "physical_manifest_count": 1,
        "source_prediction_seal_count": 0,
        "source_revision": "f8229074e01142e90357fd863beec0556229d9b4",
        "status": "source-technical-failure-retained",
        "terminal_stage_kind": "physical-source-materialization",
        "workflow_run_attempt": 1,
        "workflow_run_id": 31558648421,
    }
    scope = payload["repair_scope"]
    assert scope["case_subprocess_stdin_isolation_completed"]
    assert all(
        value is False
        for key, value in scope.items()
        if key != "case_subprocess_stdin_isolation_completed"
    )


def test_dispatcher_and_workflow_bind_the_exact_isolation_repair() -> None:
    payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    workflow = WORKFLOW.read_text(encoding="utf-8")
    dispatcher = DISPATCHER.read_text(encoding="utf-8")
    amendment_digest = hashlib.sha256(AMENDMENT.read_bytes()).hexdigest()
    dispatcher_digest = hashlib.sha256(DISPATCHER.read_bytes()).hexdigest()

    assert payload["repair_id"] in dispatcher
    assert 'exec "${BPT_FRAME_ZERO_PYTHON}" "${arguments[@]}" </dev/null' in dispatcher
    assert 'exec "${BPT_PRIMARY_PYTHON}" "$@" </dev/null' in dispatcher
    assert 'exec "${BPT_PRIMARY_PYTHON}" "$@"\n' in dispatcher
    assert f"CASE_STDIN_ISOLATION_REPAIR_ID: {payload['repair_id']}" in workflow
    assert f"CASE_STDIN_ISOLATION_REPAIR_SHA256: {amendment_digest}" in workflow
    assert f"DISPATCHER_SHA256: {dispatcher_digest}" in workflow
    assert "runtime_case_subprocess_stdin_isolation" in workflow
    assert "BPT_CASE_STDIN_ISOLATION_MARKER" in workflow
