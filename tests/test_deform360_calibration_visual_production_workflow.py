from __future__ import annotations

from pathlib import Path

WORKFLOW = Path(".github/workflows/deform360-calibration-visual-production.yml")


def test_visual_production_workflow_is_main_only_and_resumable() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "workflow_dispatch:" in text
    assert "github.event_name == 'workflow_dispatch'" in text
    assert "github.ref == 'refs/heads/main'" in text
    assert "runs-on: [self-hosted, Linux, X64, nvidia-smi]" in text
    assert "cancel-in-progress: false" in text
    assert "--resume" in text
    assert "--attempt-id" in text
    assert ".production.lock" not in text


def test_visual_production_workflow_pins_every_external_source() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "25d90ef7f78ba4307f4555cb636d666004e1bf66" in text
    assert "9cb4e9679f5f34e249945544052464ef46324bc2" in text
    assert (
        "uses: ./.github/workflows/deform360-calibration-prepared-inventory.yml" in text
    )
    assert "deform360-calibration-retained-source-admission-" in text
    assert "persist-credentials: false" in text
    assert "actions/checkout@v7" in text
    assert "actions/upload-artifact@v7" in text


def test_visual_production_consumes_authoritative_custody_boundary() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "Authoritative retained-source admission" in text
    assert "needs.retained-source.outputs.admission_id" in text
    assert "needs.retained-source.outputs.artifact_digest" in text
    assert "execute_deform360_calibration_visual_production.py" in text
    assert "calibration-visual-execution-admission.json" in text
    assert "prediction-seal.json" in text
    assert "visual-production-result.json" in text


def test_visual_production_artifact_excludes_large_predictions_and_targets() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    upload = text[text.index("Upload compact calibration-only production evidence") :]

    assert "*.npz" not in upload
    assert "predictions.json" not in upload
    assert "confirmation-processed" not in text
    assert "reserved_evaluation_frames_opened=false" in text
    assert "confirmation_payloads_opened=false" in text
    assert "target_outcomes_used=false" in text
    assert "replacement_allowed=true" not in text


def test_hugging_face_token_is_not_workflow_wide() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    global_env = text[text.index("env:") : text.index("jobs:")]

    assert "HF_TOKEN" not in global_env
    assert text.count("HF_TOKEN: ${{ secrets.HF_TOKEN }}") == 2
