from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/deform360-v6-source-prediction-evidence.yml"
ARCHIVED_RUNNER = (
    ROOT / "scripts/ci/archive/run_deform360_v6_source_prediction_evidence_v2.sh"
)
ARCHIVED_RUNNER_BLOB_SHA = "42dd4f3e0d05f18b9ff0a0bdcf90fbd282f0f6f1"


def _workflow_steps() -> list[dict[str, Any]]:
    payload = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return list(payload["jobs"]["evidence"]["steps"])


def _workflow_step(name: str) -> dict[str, Any]:
    matches = [step for step in _workflow_steps() if step.get("name") == name]
    assert len(matches) == 1
    return matches[0]


def _inline_python_after(stage: str) -> str:
    text = ARCHIVED_RUNNER.read_text(encoding="utf-8")
    stage_marker = f'set_stage "{stage}"\n'
    assert text.count(stage_marker) == 1
    tail = text.split(stage_marker, 1)[1]
    heredoc_marker = "\"${BPT_PYTHON}\" - <<'PY'\n"
    assert heredoc_marker in tail
    body, remainder = tail.split(heredoc_marker, 1)[1].split("\nPY\n", 1)
    assert remainder
    return body + "\n"


def _git_blob_sha(path: Path) -> str:
    content = path.read_bytes()
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content, usedforsecurity=False).hexdigest()


def test_source_execution_exports_exact_archived_inline_roots() -> None:
    step = _workflow_step("Generate real prefix-only source prediction evidence")
    environment = step["env"]
    run_root = (
        "${{ env.RESULTS_ROOT }}/bayesian-phystwin/"
        "deform360-v6-source-prediction/${{ env.AMENDMENT_ID }}/"
        "${{ github.sha }}"
    )

    assert environment["RUN_ROOT"] == run_root
    assert environment["PREDICTION_ROOT"] == f"{run_root}/prediction-panel"
    assert _git_blob_sha(ARCHIVED_RUNNER) == ARCHIVED_RUNNER_BLOB_SHA


def test_archived_source_plan_and_evidence_blocks_use_bound_roots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    plan = {
        "schema": "fixture.deform360-v6-source-prediction-plan",
        "record_count": 100,
    }
    (run_root / "source-plan-inputs.json").write_text(
        json.dumps({"source_prediction_plan": plan}),
        encoding="utf-8",
    )
    monkeypatch.setenv("RUN_ROOT", str(run_root))

    source_plan_block = _inline_python_after("materialize-source-plan")
    assert 'os.environ["RUN_ROOT"]' in source_plan_block
    exec(compile(source_plan_block, str(ARCHIVED_RUNNER), "exec"), {})
    assert (
        json.loads((run_root / "source-plan.json").read_text(encoding="utf-8"))
        == plan
    )

    prediction_root = run_root / "prediction-panel"
    seals = prediction_root / "source-seals"
    seals.mkdir(parents=True)
    (prediction_root / "source-prediction-receipt.json").write_text(
        json.dumps(
            {
                "prediction_record_count": 100,
                "information_boundary": {
                    "development_suffix_opened": False,
                    "target_outcomes_used": False,
                },
            }
        ),
        encoding="utf-8",
    )
    (prediction_root / "source-prediction-batch.json").write_text(
        json.dumps({"record_count": 100, "fold_count": 10}),
        encoding="utf-8",
    )
    for index in range(100):
        (seals / f"seal-{index:03d}.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("PREDICTION_ROOT", str(prediction_root))

    verification_block = _inline_python_after("verify-prediction-evidence")
    assert 'os.environ["PREDICTION_ROOT"]' in verification_block
    exec(compile(verification_block, str(ARCHIVED_RUNNER), "exec"), {})
