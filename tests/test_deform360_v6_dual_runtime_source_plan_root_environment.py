from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (
    ROOT / ".github/workflows/deform360-v6-source-prediction-evidence-dual-runtime.yml"
)
ARCHIVED_RUNNER = (
    ROOT / "scripts/ci/archive/run_deform360_v6_source_prediction_evidence_v2.sh"
)
ARCHIVED_RUNNER_BLOB_SHA = "42dd4f3e0d05f18b9ff0a0bdcf90fbd282f0f6f1"


def _workflow_steps() -> list[dict[str, Any]]:
    payload = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return list(payload["jobs"]["evidence"]["steps"])


def _git_blob_sha(path: Path) -> str:
    content = path.read_bytes()
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content, usedforsecurity=False).hexdigest()


def test_dual_runtime_source_execution_exports_archived_inline_roots() -> None:
    matches = [
        step
        for step in _workflow_steps()
        if step.get("name") == "Generate real prefix-only source prediction evidence"
    ]
    assert len(matches) == 1
    environment = matches[0]["env"]
    run_root = (
        "${{ env.RESULTS_ROOT }}/bayesian-phystwin/"
        "deform360-v6-source-prediction/${{ env.AMENDMENT_ID }}/"
        "${{ github.sha }}"
    )

    assert environment["RUN_ROOT"] == run_root
    assert environment["PREDICTION_ROOT"] == f"{run_root}/prediction-panel"
    assert _git_blob_sha(ARCHIVED_RUNNER) == ARCHIVED_RUNNER_BLOB_SHA
