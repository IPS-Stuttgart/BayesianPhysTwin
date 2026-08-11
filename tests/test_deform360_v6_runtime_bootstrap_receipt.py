from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/deform360-v6-source-prediction-evidence.yml"


def _steps() -> list[dict[str, Any]]:
    payload = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return list(payload["jobs"]["evidence"]["steps"])


def _step(name: str) -> dict[str, Any]:
    matches = [step for step in _steps() if step.get("name") == name]
    assert len(matches) == 1
    return matches[0]


def _fallback_python() -> str:
    run = str(_step("Ensure bounded execution receipt exists")["run"])
    marker = "\"${BPT_PYTHON:-python}\" - <<'PY'\n"
    assert run.count(marker) == 1
    remainder = run.split(marker, 1)[1]
    body, delimiter = remainder.rsplit("\nPY", 1)
    assert not delimiter.strip()
    return body + "\n"


def _run_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    runtime_exit_code: str,
    execute_exit_code: str,
) -> dict[str, Any]:
    runtime_log = tmp_path / "runtime.log"
    runtime_log.write_text(
        "pre-commit 4.6.2 requires virtualenv, which is not installed.\n",
        encoding="utf-8",
    )
    output = tmp_path / "execution-receipt.json"
    environment = {
        "BPT_SOURCE_SHA": "a004edbf5389714d033488ddc9fd54e131ec5b98",
        "GITHUB_RUN_ID": "31497776180",
        "GITHUB_RUN_ATTEMPT": "1",
        "RUNNER_NAME": "workstation2",
        "RUNTIME_EXIT_CODE": runtime_exit_code,
        "EXECUTE_EXIT_CODE": execute_exit_code,
        "RUNTIME_LOG": str(runtime_log),
        "RECEIPT_PATH": str(output),
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    exec(compile(_fallback_python(), str(WORKFLOW), "exec"), {})
    return json.loads(output.read_text(encoding="utf-8"))


def test_gpu_runtime_dependency_is_exact_and_gates_science() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    runtime = _step("Build isolated GPU source runtime")
    execute = _step("Generate real prefix-only source prediction evidence")

    assert runtime["id"] == "runtime"
    assert workflow.count('"virtualenv==21.7.4"') == 1
    assert 'version("virtualenv") != "21.7.4"' in str(runtime["run"])
    assert execute["if"] == "steps.runtime.outputs.exit_code == '0'"


def test_bootstrap_failure_receipt_is_bounded_and_target_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    receipt = _run_fallback(
        monkeypatch,
        tmp_path,
        runtime_exit_code="1",
        execute_exit_code="",
    )

    assert receipt["status"] == "source-technical-failure-retained"
    assert receipt["terminal_stage"] == "build-isolated-gpu-source-runtime"
    assert receipt["exit_code"] == 1
    assert receipt["physical_manifest_count"] == 0
    assert receipt["source_prediction_seal_count"] == 0
    assert "requires virtualenv" in receipt["error"]
    assert receipt["claim_authorized"] is False
    assert receipt["fresh_target_selection_authorized"] is False
    assert receipt["fresh_target_payload_access_authorized"] is False
    assert not any(receipt["information_boundary"].values())
    assert len(receipt["receipt_id"]) == 64


def test_missing_success_receipt_is_retained_as_technical_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    receipt = _run_fallback(
        monkeypatch,
        tmp_path,
        runtime_exit_code="0",
        execute_exit_code="0",
    )

    assert receipt["status"] == "source-technical-failure-retained"
    assert (
        receipt["terminal_stage"]
        == "generate-real-prefix-only-source-prediction-evidence"
    )
    assert receipt["exit_code"] == 1
    assert receipt["error"] == "source execution completed without a bounded receipt"
