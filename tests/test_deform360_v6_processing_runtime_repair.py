from __future__ import annotations

import json
from pathlib import Path

from bayesian_phystwin._portable_contracts import content_id

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / (
    "protocols/amendments/"
    "deform360_official_hub_fresh_object_session_v6_processing_runtime.json"
)
WORKFLOW = ROOT / ".github/workflows/deform360-v6-source-prediction-evidence.yml"
RUNNER = ROOT / "scripts/ci/run_deform360_v6_source_prediction_evidence.sh"


def test_processing_runtime_repair_is_content_addressed_and_target_closed() -> None:
    payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    declared = payload.pop("repair_id")

    assert declared == content_id(payload)
    assert payload["correction"] == {
        "deform360_pyproject_file_sha256": (
            "0ccfe6a386c184613191ccdaa8f2912bc3c148a7dda9c9f126959301d250232e"
        ),
        "deform360_repository": "lhy0807/deform360",
        "deform360_repository_revision": ("0fe36f0b7a7a917ba62b5f8cee707299a9a4a317"),
        "gsplat_version": "1.4.0",
        "install_target": "_deform360_physical[processing]",
        "nerfstudio_version": "1.1.5",
        "required_imports": [
            "nerfstudio.configs.method_configs",
            "nerfstudio.scripts.exporter",
            "gsplat",
        ],
        "requirements_processing_git_file_used": False,
    }
    assert not any(payload["information_boundary"].values())
    assert payload["repair_scope"]["deform360_processing_dependencies_completed"]
    assert all(
        value is False
        for key, value in payload["repair_scope"].items()
        if key != "deform360_processing_dependencies_completed"
    )


def test_processing_runtime_identity_is_bound_into_workflow_and_receipt() -> None:
    payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    workflow = WORKFLOW.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")

    assert f"DEFORM360_PROCESSING_REPAIR_ID: {payload['repair_id']}" in workflow
    assert str(AMENDMENT.relative_to(ROOT)) in workflow
    assert f'PROCESSING_RUNTIME_REPAIR_ID="{payload["repair_id"]}"' in runner
    assert str(AMENDMENT.relative_to(ROOT)) in runner
    assert '"runtime_deform360_processing_dependencies"' in runner
