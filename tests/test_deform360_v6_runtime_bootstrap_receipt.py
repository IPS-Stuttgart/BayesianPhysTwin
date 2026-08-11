from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/deform360-v6-source-prediction-evidence.yml"
PROCESSING_ENV = {
    "DEFORM360_PHYSICAL_REVISION": "0fe36f0b7a7a917ba62b5f8cee707299a9a4a317",
    "DEFORM360_PROCESSING_PYPROJECT_SHA256": (
        "0ccfe6a386c184613191ccdaa8f2912bc3c148a7dda9c9f126959301d250232e"
    ),
    "DEFORM360_PROCESSING_REPAIR_ID": (
        "15a02312f15c7acaeb756f432f4d1b5015697ca17d8966726bcd82e33c6b795f"
    ),
    "DEFORM360_PROCESSING_REPAIR_PATH": (
        "protocols/amendments/"
        "deform360_official_hub_fresh_object_session_v6_processing_runtime.json"
    ),
    "DEFORM360_PROCESSING_REPAIR_SHA256": (
        "b60a18821b0e260519ffda2289b20cb247b1a36c91eac9a528d953111e7b520c"
    ),
    "GSPLAT_VERSION": "1.4.0",
    "NERFSTUDIO_VERSION": "1.1.5",
    "EXPECTED_PIP_CHECK_CONFLICT": (
        "pyrecest 2.4.3 has requirement numpy<2.5,>=2.0, but you have numpy 1.26.4."
    ),
    "NUMPY_VERSION": "1.26.4",
    "NUSCENES_DEVKIT_VERSION": "1.2.0",
    "PYRECEST_VERSION": "2.4.3",
    "RUNTIME_DEPENDENCY_SCOPE_REPAIR_ID": (
        "1b5f822991ed674554f4052f8112255b33d911bbb0f4797840ba1879e452f460"
    ),
    "RUNTIME_DEPENDENCY_SCOPE_REPAIR_PATH": (
        "protocols/amendments/"
        "deform360_official_hub_fresh_object_session_v6_runtime_dependency_scope.json"
    ),
    "RUNTIME_DEPENDENCY_SCOPE_REPAIR_SHA256": (
        "86d0a49bdf93adf25f214b69bdd52e774f1028493dc9b2228dbf1bef14518a31"
    ),
}


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


def _run_fallback_shell(
    tmp_path: Path,
    *,
    evidence_root: Path,
) -> subprocess.CompletedProcess[str]:
    runtime_log = tmp_path / "runtime.log"
    runtime_log.write_text("runtime bootstrap failed\n", encoding="utf-8")
    environment = os.environ.copy()
    environment.update(
        {
            **PROCESSING_ENV,
            "BPT_PYTHON": sys.executable,
            "BPT_SOURCE_SHA": "a004edbf5389714d033488ddc9fd54e131ec5b98",
            "EVIDENCE_ROOT": str(evidence_root),
            "EXECUTE_EXIT_CODE": "",
            "GITHUB_RUN_ATTEMPT": "1",
            "GITHUB_RUN_ID": "31497776180",
            "RUNNER_NAME": "workstation2",
            "RUNTIME_EXIT_CODE": "1",
            "RUNTIME_LOG": str(runtime_log),
        }
    )
    return subprocess.run(
        ["bash", "-c", str(_step("Ensure bounded execution receipt exists")["run"])],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )


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
        **PROCESSING_ENV,
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
    assert '-e "./_deform360_physical[processing]"' in str(runtime["run"])
    assert 'version("nerfstudio") != "1.1.5"' in str(runtime["run"])
    assert 'version("gsplat") != "1.4.0"' in str(runtime["run"])
    assert 'version("numpy") != "1.26.4"' in str(runtime["run"])
    assert 'version("pyrecest") != "2.4.3"' in str(runtime["run"])
    assert 'version("nuscenes-devkit") != "1.2.0"' in str(runtime["run"])
    assert "from nerfstudio.configs import method_configs" in str(runtime["run"])
    assert "from nerfstudio.scripts import exporter" in str(runtime["run"])
    assert "DEFORM360_PROCESSING_PYPROJECT_SHA256" in str(runtime["run"])
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
    assert receipt["runtime_deform360_processing_dependencies"]["activated"] is False
    assert (
        receipt["runtime_deform360_processing_dependencies"]["install_target"]
        == "_deform360_physical[processing]"
    )
    assert receipt["runtime_dependency_scope_repair"]["activated"] is False
    assert receipt["runtime_dependency_scope_repair"]["pyrecest_runtime_used"] is False
    assert (
        receipt["runtime_dependency_scope_repair"]["other_dependency_conflicts_allowed"]
        is False
    )
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


def test_fallback_rejects_dangling_receipt_symlink(
    tmp_path: Path,
) -> None:
    evidence_root = tmp_path / "evidence"
    compact = evidence_root / "deform360-v6-source-prediction-evidence"
    compact.mkdir(parents=True)
    escaped = tmp_path / "escaped-receipt.json"
    receipt = compact / "execution-receipt.json"
    receipt.symlink_to(escaped)

    result = _run_fallback_shell(tmp_path, evidence_root=evidence_root)

    assert result.returncode != 0
    assert "refusing symlinked execution receipt" in result.stderr
    assert receipt.is_symlink()
    assert not escaped.exists()


def test_existing_receipt_fast_path_still_validates_log_directory(
    tmp_path: Path,
) -> None:
    evidence_root = tmp_path / "evidence"
    compact = evidence_root / "deform360-v6-source-prediction-evidence"
    compact.mkdir(parents=True)
    receipt = compact / "execution-receipt.json"
    receipt.write_text('{"status":"sealed"}\n', encoding="utf-8")

    result = _run_fallback_shell(tmp_path, evidence_root=evidence_root)

    assert result.returncode == 0
    assert receipt.read_text(encoding="utf-8") == '{"status":"sealed"}\n'
    assert (compact / "logs").is_dir()


@pytest.mark.parametrize("symlink_name", ["compact", "logs"])
def test_fallback_rejects_symlinked_evidence_directories(
    tmp_path: Path,
    symlink_name: str,
) -> None:
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    compact = evidence_root / "deform360-v6-source-prediction-evidence"
    escaped = tmp_path / f"escaped-{symlink_name}"
    escaped.mkdir()
    if symlink_name == "compact":
        compact.symlink_to(escaped, target_is_directory=True)
    else:
        compact.mkdir()
        (compact / "logs").symlink_to(escaped, target_is_directory=True)

    result = _run_fallback_shell(tmp_path, evidence_root=evidence_root)

    assert result.returncode != 0
    assert not any(escaped.iterdir())
