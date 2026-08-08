from __future__ import annotations

from pathlib import Path

WORKFLOW = Path(".github/workflows/deform360-calibration-visual-production.yml")
EXECUTOR = Path("scripts/science/execute_deform360_calibration_visual_production.py")


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


def test_visual_production_pins_the_audited_admission_and_external_sources() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "25d90ef7f78ba4307f4555cb636d666004e1bf66" in text
    assert "9cb4e9679f5f34e249945544052464ef46324bc2" in text
    assert 'ADMISSION_RUN_ID: "31272985733"' in text
    assert 'ADMISSION_ARTIFACT_ID: "9026183221"' in text
    assert (
        "ADMISSION_ARTIFACT_NAME: "
        "deform360-calibration-retained-source-admission-31272985733-1"
    ) in text
    assert (
        "ADMISSION_ARTIFACT_DIGEST: "
        "sha256:d13a3aed7b63effab637215feee15c61d9cb69330dbe8f666a6e37b00b35b836"
    ) in text
    assert (
        "ADMISSION_ID: "
        "4dd68e209b4c1a206a209786f57b0a4a96bd102a79a0f8f60d436fabd5d584ba"
    ) in text
    assert "uses: actions/github-script@v8" in text
    assert "listWorkflowRunArtifacts" in text
    assert "artifact.digest !== expectedDigest" in text
    assert "artifact.workflow_run.head_sha !== expectedHead" in text
    assert "uses: actions/download-artifact@v8" in text
    assert "artifact-ids: ${{ needs.admission-lock.outputs.artifact_id }}" in text
    assert "digest-mismatch: error" in text
    assert "persist-credentials: false" in text
    assert "uses: ./.github/workflows/deform360-calibration-prepared-inventory.yml" not in text
    assert "needs.retained-source" not in text


def test_visual_production_revalidates_the_complete_audited_artifact() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "sha256sum --check --strict SHA256SUMS" in text
    assert "ADMISSION_FILE_SHA256:" in text
    assert "INVENTORY_FILE_SHA256:" in text
    assert "PLAN_FILE_SHA256:" in text
    assert "RECEIPT_FILE_SHA256:" in text
    assert "audited receipt changed" in text
    assert "admission inventory-file identity changed" in text
    assert "admission plan-file identity changed" in text
    assert "confirmation boundary opened" in text
    assert "target boundary opened" in text
    assert "execute_deform360_calibration_visual_production.py" in text
    assert "calibration-visual-execution-admission.json" in text
    assert "prediction-seal.json" in text
    assert "visual-production-result.json" in text


def test_resolved_roots_are_available_before_first_same_step_use() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    block = text[
        text.index("      - name: Resolve and separate protected roots") :
        text.index("      - name: Download exact audited retained-source admission")
    ]

    assert "mapfile -t resolved_roots" in block
    assert 'PROCESSED_ROOT="${resolved_roots[0]}"' in block
    assert 'OUTPUT_ROOT="${resolved_roots[1]}"' in block
    assert 'HF_CACHE_DIR="${resolved_roots[2]}"' in block
    assert 'PRODUCTION_RUN_ROOT="${OUTPUT_ROOT}/${ADMISSION_ID}"' in block
    assert 'export PROCESSED_ROOT OUTPUT_ROOT HF_CACHE_DIR PRODUCTION_RUN_ROOT' in block
    assert block.index('OUTPUT_ROOT="${resolved_roots[1]}"') < block.index(
        'mkdir -p -- "${OUTPUT_ROOT}" "${HF_CACHE_DIR}"'
    )
    assert 'Path(os.environ["GITHUB_ENV"])' not in block


def test_source_bytes_are_revalidated_inside_the_locked_job_loop() -> None:
    text = EXECUTOR.read_text(encoding="utf-8")

    lock = text.index('with lock_path.open("a+b") as lock_stream:')
    loop = text.index("        for job in jobs:", lock)
    video = text.index("            source_video = _verify_source(", loop)
    timestamps = text.index('job["source_timestamps"]', video)
    existing = text.index("            existing = _existing_receipt(", timestamps)
    command = text.index("            command = build_deform360_calibration_visual_command(", existing)
    process = text.index("                produced = _run(command)", command)

    assert lock < loop < video < timestamps < existing < command < process
    assert "sources: dict[str, Path]" not in text
    assert "source_video_path=source_video" in text


def test_visual_production_artifact_excludes_large_predictions_and_targets() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    upload = text[text.index("Upload compact production evidence") :]

    assert "*.npz" not in upload
    assert "predictions.json" not in upload
    assert "confirmation-processed" not in text
    assert "reserved_evaluation_frames_opened=false" in text
    assert "confirmation_payloads_opened=false" in text
    assert "target_outcomes_used=false" in text
    assert "replacement_allowed=true" not in text


def test_technical_failures_are_uploaded_then_fail_the_workflow() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    upload = text.index("Upload compact production evidence")
    failure_gate = text.index("Fail workflow when technical failures were retained")
    assert upload < failure_gate
    assert "steps.production.outputs.terminal_code == '3'" in text
    assert "exit 1" in text[failure_gate:]


def test_transient_environment_and_evidence_are_removed() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    cleanup = text[text.index("Remove transient environment and evidence copies") :]

    assert "PRODUCTION_ENV_ROOT" in cleanup
    assert "EVIDENCE_ROOT" in cleanup
    assert "ADMISSION_ROOT" in cleanup
    assert cleanup.count("rm -rf") == 3


def test_hugging_face_token_is_not_workflow_wide() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    global_env = text[text.index("env:") : text.index("jobs:")]

    assert "HF_TOKEN" not in global_env
    assert text.count("HF_TOKEN: ${{ secrets.HF_TOKEN }}") == 2
