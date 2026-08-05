from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "deform360-calibration-acquisition.yml"
SCRIPT = ROOT / "scripts" / "science" / "run_deform360_calibration_acquisition.py"
RUNTIME_HELPERS = tuple(sorted((ROOT / "scripts" / "science").glob("_deform360_calibration_acquisition_runtime_*.py")))


def test_workflow_opens_calibration_only_by_explicit_main_dispatch() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "pull_request:" in workflow
    assert "workflow_dispatch:" in workflow
    assert "open_calibration_payloads" in workflow
    assert "refs/heads/main" in workflow
    assert "--open-calibration-payloads" in workflow
    assert "runs-on: [self-hosted, Linux, X64, nvidia-smi]" in workflow
    assert "contents: read" in workflow
    assert "contents: write" not in workflow
    assert "persist-credentials: false" in workflow
    assert "d8522a4403b766aeb387510c04e89032a56fdf35" in workflow
    assert "confirmation_root" not in workflow
    assert "target_root" not in workflow
    assert "raw/**" not in workflow
    assert "processed/**" not in workflow
    assert "actions/upload-artifact@" in workflow
    assert "validate_calibration_acquisition_result" in workflow
    assert "build_calibration_evidence_ledger" in workflow
    assert 'content identity changed: {filename}' in workflow
    assert "${GITHUB_WORKSPACE}/src" in workflow
    assert '"huggingface_hub==1.26.0"' in workflow
    assert '"numpy==1.26.4"' in workflow
    assert "pip freeze --all" in workflow
    assert 'PYTHONDONTWRITEBYTECODE: "1"' in workflow
    assert "git -C _deform360 status" in workflow
    assert "--untracked-files=all" in workflow


def test_runtime_source_exposes_no_confirmation_or_target_arguments() -> None:
    source = "\n".join(path.read_text(encoding="utf-8") for path in (SCRIPT, *RUNTIME_HELPERS))
    assert "--open-calibration-payloads" in source
    assert "--confirmation" not in source
    assert "--target" not in source
    assert "--outcome" not in source
    assert "payload-allowlist.json" in source
    assert "expected_metadata_sha256" in source
    assert "_LOCAL_PROCESSING_EPISODE_INDEX = 0" in source
    assert "source_episode_id" in source
    assert "technical_failure_retained_without_replacement" in source
    assert "download_manifest_id" in source
    assert "failure_log_sha256" in source
    assert "_require_bayesian_phystwin_import" in source
    assert '"inputs/stage0-selection.json"' in source
    assert '"inputs/visual-provider-lock.json"' in source
    assert '"implementation/deform360_calibration_acquisition.py"' in source
    assert "--untracked-files=all" in source
