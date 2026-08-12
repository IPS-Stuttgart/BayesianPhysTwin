from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bayesian_phystwin._portable_contracts import content_id

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / (
    "protocols/amendments/"
    "deform360_official_hub_fresh_object_session_v6_primary_cma_runtime.json"
)
WORKFLOW = ROOT / (
    ".github/workflows/deform360-v6-source-prediction-evidence-dual-runtime.yml"
)


def test_primary_cma_runtime_repair_is_content_addressed_and_closed() -> None:
    payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    declared = payload.pop("repair_id")

    assert declared == content_id(payload)
    assert not any(payload["information_boundary"].values())
    assert payload["failed_execution_evidence"]["physical_manifest_count"] == 0
    assert payload["failed_execution_evidence"]["source_prediction_seal_count"] == 0
    assert payload["failed_execution_evidence"]["error_type"] == ("ModuleNotFoundError")
    assert payload["repair_scope"]["primary_cma_import_dependency_completed"]
    assert all(
        value is False
        for key, value in payload["repair_scope"].items()
        if key != "primary_cma_import_dependency_completed"
    )

    correction = payload["correction"]
    assert correction["distribution"] == "cma"
    assert correction["version"] == "4.4.4"
    assert correction["wheel_filename"] == "cma-4.4.4-py3-none-any.whl"
    assert correction["wheel_sha256"] == (
        "edb6d02eb2aac2d54650f16a8f0c70711ff17445957de7c9de92ff7fd4b7ef38"
    )
    assert correction["import_only_dependency"]
    assert correction["cma_optimizer_selected"] is False


def test_workflow_hash_pins_probes_and_receipts_primary_cma_runtime() -> None:
    payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    workflow = WORKFLOW.read_text(encoding="utf-8")
    amendment_digest = hashlib.sha256(AMENDMENT.read_bytes()).hexdigest()
    correction = payload["correction"]

    assert f"PRIMARY_CMA_RUNTIME_REPAIR_ID: {payload['repair_id']}" in workflow
    assert f"PRIMARY_CMA_RUNTIME_REPAIR_SHA256: {amendment_digest}" in workflow
    assert f'PRIMARY_CMA_VERSION: "{correction["version"]}"' in workflow
    assert f"PRIMARY_CMA_WHEEL_SHA256: {correction['wheel_sha256']}" in workflow
    assert str(AMENDMENT.relative_to(ROOT)) in workflow
    assert (
        '"cma==${PRIMARY_CMA_VERSION} --hash=sha256:${PRIMARY_CMA_WHEEL_SHA256}"'
    ) in workflow
    assert "--no-deps" in workflow
    assert "--only-binary=:all:" in workflow
    assert "--require-hashes" in workflow
    assert "import cma" in workflow
    assert 'version("cma") != os.environ["PRIMARY_CMA_VERSION"]' in workflow
    assert "callable(cma.CMAEvolutionStrategy)" in workflow
    assert "runtime_primary_cma_import_dependency" in workflow
    assert '"cma_optimizer_selected": False' in workflow
