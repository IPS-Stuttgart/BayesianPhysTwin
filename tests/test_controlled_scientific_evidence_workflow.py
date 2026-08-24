from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_orchestrator() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "science"
        / "run_controlled_evidence_workflow_v1.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_controlled_evidence_workflow_v1",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_controlled_study_roster_and_claim_boundaries() -> None:
    module = _load_orchestrator()
    assert module.STUDIES == (
        "simulation-based-calibration",
        "synthetic-benchmark-sbc",
        "recursive-corruption",
    )
    assert module.STUDY_CHOICES == (*module.STUDIES, "all-controlled")
    assert set(module.CLAIM_BOUNDARIES) == set(module.STUDIES)
    for boundary in module.CLAIM_BOUNDARIES.values():
        assert "state of the art" in boundary


def test_byte_replay_comparison_detects_drift(tmp_path: Path) -> None:
    module = _load_orchestrator()
    primary = tmp_path / "primary"
    replay = tmp_path / "replay"
    primary.mkdir()
    replay.mkdir()
    (primary / "result.json").write_text('{"value": 1}\n', encoding="utf-8")
    (replay / "result.json").write_text('{"value": 1}\n', encoding="utf-8")
    matched = module._compare_files(primary, replay, ("result.json",))
    assert matched["byte_identical"] is True
    (replay / "result.json").write_text('{"value": 2}\n', encoding="utf-8")
    drifted = module._compare_files(primary, replay, ("result.json",))
    assert drifted["byte_identical"] is False


def test_manifest_is_content_addressed_and_target_closed(tmp_path: Path) -> None:
    module = _load_orchestrator()
    (tmp_path / "environment.txt").write_text("python=3.12\n", encoding="utf-8")
    summary = {
        "schema_version": 1,
        "artifact_kind": "ControlledScientificEvidenceBundleSummary",
        "selected_studies": ["recursive-corruption"],
        "verify_replay": True,
        "all_passed": False,
        "outcomes": {},
        "target_outcomes_used": False,
        "deform360_confirmation_opened": False,
        "causal4d_physical_outcome_used": False,
    }
    manifest = module._write_manifest(
        tmp_path,
        summary=summary,
        repository="IPS-Stuttgart/BayesianPhysTwin",
        commit="a" * 40,
        workflow_run_id=123,
        workflow_run_attempt=1,
    )
    manifest_id = manifest["manifest_id"]
    logical = dict(manifest)
    del logical["manifest_id"]
    assert manifest_id == module._canonical_id(logical)
    assert manifest["target_outcomes_used"] is False
    assert manifest["deform360_confirmation_opened"] is False
    assert manifest["causal4d_physical_outcome_used"] is False
    paths = {entry["path"] for entry in manifest["files"]}
    assert paths == {"bundle-summary.json", "environment.txt"}


def test_workflow_is_permanent_parameterized_and_read_only() -> None:
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github" / "workflows" / "paper-evidence.yml").read_text(
        encoding="utf-8"
    )
    assert workflow.startswith(
        "# workflow-lifecycle: permanent\n"
        "# workflow-owner: IPS-Stuttgart maintainers\n"
    )
    assert "evidence_study:" in workflow
    assert "all-controlled" in workflow
    assert "verify_replay:" in workflow
    assert "run_controlled_evidence_workflow_v1.py" in workflow
    assert "permissions:\n  contents: read" in workflow
    assert "pull_request_target" not in workflow
    assert "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1" in workflow
    assert "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97" in workflow
    assert "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a" in workflow
