from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bayesian_phystwin._portable_contracts import content_id

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/science/audit_deform360_prob4d_support_stop.py"


def _load_auditor():  # type: ignore[no-untyped-def]
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("support_stop_auditor", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


auditor = _load_auditor()

SOURCE_REVISION = "d" * 40
AUDITOR_REVISION = "a" * 40
PRODUCTION_ID = "1" * 64
ADMISSION_ID = "2" * 64
PROB4D_REVISION = "3" * 40
MOTIONCRAFTER_REVISION = "4" * 40
RUN_ID = 31297018948
ARTIFACT_ID = 9033414269


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_checksums(root: Path) -> None:
    members = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = [
        f"{_sha256(path)}  {path.relative_to(root).as_posix()}\n"
        for path in members
    ]
    (root / "SHA256SUMS").write_text("".join(lines), encoding="ascii")


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    support_root = source / "metric-support"
    support_root.mkdir(parents=True)

    jobs = [
        {"object_id": "sheet-1", "camera_id": "cam-1", "status": "supported"},
        {"object_id": "sheet-1", "camera_id": "cam-2", "status": "supported"},
        {"object_id": "volume-1", "camera_id": "cam-1", "status": "supported"},
        {"object_id": "volume-1", "camera_id": "cam-2", "status": "support-negative"},
    ]
    result = {
        "implementation_revision": SOURCE_REVISION,
        "production_result_id": PRODUCTION_ID,
        "admission_id": ADMISSION_ID,
        "object_count": 2,
        "admitted_stream_count": 4,
        "supported_stream_count": 3,
        "support_negative_stream_count": 1,
        "technical_failure_stream_count": 0,
        "supported_object_count": 2,
        "plan_emitted": False,
        "plan_file": None,
        "status": "support-negatives-retained",
        "jobs": jobs,
    }
    result["result_id"] = content_id(result)
    _write_json(support_root / "metric-batch-result.json", result)

    empty_digest = hashlib.sha256(b"").hexdigest()
    _write_json(
        support_root / "support-receipt.json",
        {
            "schema": "bayesian-phystwin.deform360-prob4d-source-support-receipt",
            "schema_version": 1,
            "implementation_revision": SOURCE_REVISION,
            "metric_batch_step_outcome": "success",
            "metric_batch_result_id": result["result_id"],
            "metric_batch_status": result["status"],
            "metric_batch_stderr_sha256": empty_digest,
            "new_measurements_required": False,
            "human_approval_required": False,
            "confirmation_payloads_opened": False,
            "target_outcomes_used": False,
        },
    )
    _write_json(
        source / "pipeline-receipt.json",
        {
            "schema": "bayesian-phystwin.deform360-prob4d-source-pipeline-receipt",
            "schema_version": 1,
            "implementation_revision": SOURCE_REVISION,
            "visual_production_result_id": PRODUCTION_ID,
            "prob4d_revision": PROB4D_REVISION,
            "motioncrafter_revision": MOTIONCRAFTER_REVISION,
            "stage_outcomes": {
                "metric_batch": "success",
                "support_gate": "failure",
                "samples": "skipped",
                "calibration": "skipped",
                "source_gate": "skipped",
            },
            "stderr_sha256": {
                "metric-batch": empty_digest,
                "samples": empty_digest,
                "calibration": empty_digest,
                "source-gate": empty_digest,
            },
            "source_gate_result_id": None,
            "source_gate_passed": None,
            "confirmation_access_authorized": None,
            "information_boundary": {
                "public_released_measurements_used": True,
                "new_measurements_required": False,
                "human_approval_required": False,
                "confirmation_payloads_opened": False,
                "target_outcomes_used": False,
                "future_frames_used": False,
                "replacement_allowed": False,
            },
        },
    )
    _write_checksums(source)

    repository = tmp_path / "repository"
    validator = (
        repository / "scripts/science/materialize_deform360_prob4d_metric_batch.py"
    )
    validator.parent.mkdir(parents=True)
    validator.write_text(
        "import json\nfrom pathlib import Path\n"
        "def validate_deform360_prob4d_metric_batch(root):\n"
        "    result = Path(root) / 'metric-batch-result.json'\n"
        "    return json.loads(result.read_text())\n",
        encoding="utf-8",
    )
    return source, repository


def _audit(tmp_path: Path) -> dict[str, object]:
    source, repository = _fixture(tmp_path)
    return auditor.audit_support_stop(
        source_root=source,
        output_directory=tmp_path / "audit",
        repository_root=repository,
        source_run_id=RUN_ID,
        source_run_attempt=1,
        source_run_conclusion="failure",
        source_head_sha=SOURCE_REVISION,
        source_artifact_id=ARTIFACT_ID,
        source_artifact_name=f"deform360-prob4d-source-gate-{RUN_ID}-1",
        auditor_revision=AUDITOR_REVISION,
        expected_production_result_id=PRODUCTION_ID,
        expected_admission_id=ADMISSION_ID,
        expected_prob4d_revision=PROB4D_REVISION,
        expected_motioncrafter_revision=MOTIONCRAFTER_REVISION,
        expected_object_count=2,
        expected_admitted_stream_count=4,
    )


def test_support_stop_is_a_valid_terminal_negative(tmp_path: Path) -> None:
    result = _audit(tmp_path)

    assert result["audit_status"] == "validated-support-negative"
    assert result["supported_stream_count"] == 3
    assert result["support_negative_stream_count"] == 1
    assert result["technical_failure_stream_count"] == 0
    assert result["confirmation_access_authorized"] is False
    assert (tmp_path / "audit/SHA256SUMS").is_file()


def test_support_stop_rejects_fabricated_confirmation_decision(tmp_path: Path) -> None:
    source, repository = _fixture(tmp_path)
    pipeline_path = source / "pipeline-receipt.json"
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    pipeline["confirmation_access_authorized"] = False
    _write_json(pipeline_path, pipeline)
    _write_checksums(source)

    with pytest.raises(ValueError, match="audit is invalid"):
        auditor.audit_support_stop(
            source_root=source,
            output_directory=tmp_path / "audit",
            repository_root=repository,
            source_run_id=RUN_ID,
            source_run_attempt=1,
            source_run_conclusion="failure",
            source_head_sha=SOURCE_REVISION,
            source_artifact_id=ARTIFACT_ID,
            source_artifact_name=f"deform360-prob4d-source-gate-{RUN_ID}-1",
            auditor_revision=AUDITOR_REVISION,
            expected_production_result_id=PRODUCTION_ID,
            expected_admission_id=ADMISSION_ID,
            expected_prob4d_revision=PROB4D_REVISION,
            expected_motioncrafter_revision=MOTIONCRAFTER_REVISION,
            expected_object_count=2,
            expected_admitted_stream_count=4,
        )


def test_support_stop_rejects_relabelled_support_receipt(tmp_path: Path) -> None:
    source, repository = _fixture(tmp_path)
    support_path = source / "metric-support/support-receipt.json"
    support = json.loads(support_path.read_text(encoding="utf-8"))
    support["metric_batch_result_id"] = "f" * 64
    _write_json(support_path, support)
    _write_checksums(source)

    with pytest.raises(ValueError, match="audit is invalid"):
        auditor.audit_support_stop(
            source_root=source,
            output_directory=tmp_path / "audit",
            repository_root=repository,
            source_run_id=RUN_ID,
            source_run_attempt=1,
            source_run_conclusion="failure",
            source_head_sha=SOURCE_REVISION,
            source_artifact_id=ARTIFACT_ID,
            source_artifact_name=f"deform360-prob4d-source-gate-{RUN_ID}-1",
            auditor_revision=AUDITOR_REVISION,
            expected_production_result_id=PRODUCTION_ID,
            expected_admission_id=ADMISSION_ID,
            expected_prob4d_revision=PROB4D_REVISION,
            expected_motioncrafter_revision=MOTIONCRAFTER_REVISION,
            expected_object_count=2,
            expected_admitted_stream_count=4,
        )
