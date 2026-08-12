from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from bayesian_phystwin._portable_contracts import content_id

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / (
    "protocols/amendments/"
    "deform360_official_hub_fresh_object_session_v6_"
    "case_stdin_marker_bootstrap_binding.json"
)
SOURCE_PLAN_AMENDMENT = ROOT / (
    "protocols/amendments/"
    "deform360_official_hub_fresh_object_session_v6_"
    "source_plan_environment.json"
)
SOURCE_PLAN_LAUNCHER = (
    ROOT / "scripts/ci/run_deform360_v6_source_prediction_evidence.sh"
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
        "            BPT_CASE_STDIN_ISOLATION_MARKER="
        '"${case_stdin_isolation_marker}" \\\n'
        "            \"${dispatcher}\" - <<'PY'"
    )
    assert probe in workflow
    assert workflow.count("\"${dispatcher}\" - <<'PY'") == 1
    assert workflow.count('test ! -e "${case_stdin_isolation_marker}"') == 2
    assert '"runtime_case_stdin_marker_bootstrap_binding"' in workflow


def test_source_plan_environment_repair_is_content_addressed() -> None:
    payload = json.loads(SOURCE_PLAN_AMENDMENT.read_text(encoding="utf-8"))
    declared = payload.pop("repair_id")

    assert declared == content_id(payload)
    assert declared == (
        "65096d1d4e8903eeacef0fc50816e47752a61e0d1bb4b6601f291bfcffb9ac4e"
    )
    assert not any(payload["information_boundary"].values())
    assert payload["failed_execution_evidence"] == {
        "artifact_digest": (
            "sha256:147342a12d05d93378eb652520974a5a001c6e105d083ae6fe707778ca1d165a"
        ),
        "artifact_id": 9135953420,
        "error_messages": [
            "KeyError: 'RUN_ROOT'",
            "KeyError: 'CUDA_HOST_COMPILER_PROBE_PASSED'",
        ],
        "execution_receipt_id": (
            "79bd32e1af16b3529aeb190494c892cfdb927d526a0f1ef0202aafc99c9188cb"
        ),
        "exit_code": 1,
        "physical_manifest_count": 10,
        "source_prediction_seal_count": 0,
        "source_revision": "4b8523f758ba3f1eed67674adc7747949f6a9f77",
        "terminal_stage": "materialize-source-plan",
        "workflow_run_attempt": 1,
        "workflow_run_id": 31581551099,
    }
    assert payload["repair_scope"]["source_plan_environment_binding_completed"]
    assert payload["repair_scope"]["terminal_receipt_compatibility_completed"]
    assert all(
        value is False
        for key, value in payload["repair_scope"].items()
        if key
        not in {
            "source_plan_environment_binding_completed",
            "terminal_receipt_compatibility_completed",
        }
    )


def test_source_plan_launcher_binds_exact_predecessor_and_run_root() -> None:
    payload = json.loads(SOURCE_PLAN_AMENDMENT.read_text(encoding="utf-8"))
    source = SOURCE_PLAN_LAUNCHER.read_text(encoding="utf-8")
    amendment_digest = hashlib.sha256(SOURCE_PLAN_AMENDMENT.read_bytes()).hexdigest()

    assert 'readonly BASE_REVISION="812da43f993b4fc5e1f6a96bcc308756b131fc4c"' in source
    assert (
        'readonly BASE_LAUNCHER_BLOB_SHA="'
        'b2b2307a2f89f3983cce349e1220033bf7f8f50c"' in source
    )
    assert f'SOURCE_PLAN_ENVIRONMENT_REPAIR_ID=\\\n"{payload["repair_id"]}"' in source
    assert f'SOURCE_PLAN_ENVIRONMENT_REPAIR_SHA256=\\\n"{amendment_digest}"' in source
    assert 'git show "${BASE_REVISION}:${LAUNCHER_PATH}"' in source
    assert 'git hash-object "${base_launcher}"' in source
    assert (
        'expected_run_root="${RESULTS_ROOT}/bayesian-phystwin/"\\\n'
        '"deform360-v6-source-prediction/${AMENDMENT_ID}/'
        '${BPT_SOURCE_SHA}"' in source
    )
    assert "RUN_ROOT differs from the deterministic source-plan path" in source
    assert "export RUN_ROOT" in source
    assert '"run_root_environment_bound": True' in source
    assert '"source_plan_algorithm_changed": False' in source


def test_source_plan_launcher_removes_only_synthetic_legacy_fields() -> None:
    source = SOURCE_PLAN_LAUNCHER.read_text(encoding="utf-8")

    assert 'export_default CUDA_HOST_COMPILER_PROBE_PASSED "false"' in source
    assert 'export_default NINJA_PYTORCH_PROBE_PASSED "false"' in source
    assert 'receipt.pop("runtime_cuda_host_compiler_repair", None)' in source
    assert 'receipt.pop("runtime_ninja_build_tool_repair", None)' in source
    assert '"legacy_receipt_defaults_removed": list(legacy_defaults)' in source
    assert '"failed_execution_receipt_id": (' in source
    assert "fresh_target_payload_access_authorized" not in source
    assert "fresh_target_selection_authorized" not in source


def test_source_plan_environment_launcher_has_valid_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(SOURCE_PLAN_LAUNCHER)], check=True)
