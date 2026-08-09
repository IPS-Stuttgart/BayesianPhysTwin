from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

SCRIPT = Path("scripts/ci/audit_deform360_prob4d_support_negative.py")
WORKFLOW = Path(
    ".github/workflows/revalidate-deform360-prob4d-support-negative-once.yml"
)
SPEC = importlib.util.spec_from_file_location(
    "_audit_deform360_prob4d_support_negative",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
assert isinstance(MODULE, ModuleType)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _label_digest(label: str) -> str:
    return _sha256(label.encode("utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_source_checksums(root: Path) -> None:
    lines = []
    for relative in MODULE.EXPECTED_SOURCE_FILES:
        path = root / relative
        lines.append(f"{_sha256(path.read_bytes())}  {relative}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="ascii")


def _configure_small_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MODULE, "SOURCE_WORKFLOW_RUN_ID", 12345)
    monkeypatch.setattr(MODULE, "SOURCE_WORKFLOW_RUN_ATTEMPT", 1)
    monkeypatch.setattr(MODULE, "SOURCE_WORKFLOW_CONCLUSION", "failure")
    monkeypatch.setattr(MODULE, "SOURCE_HEAD_SHA", "a" * 40)
    monkeypatch.setattr(MODULE, "SOURCE_ARTIFACT_ID", 67890)
    monkeypatch.setattr(MODULE, "SOURCE_ARTIFACT_NAME", "fixture-artifact")
    monkeypatch.setattr(MODULE, "SOURCE_ARTIFACT_DIGEST", "sha256:" + "b" * 64)
    monkeypatch.setattr(MODULE, "PRODUCTION_RESULT_ID", "c" * 64)
    monkeypatch.setattr(MODULE, "ADMISSION_ID", "d" * 64)
    monkeypatch.setattr(MODULE, "PROB4D_REVISION", "e" * 40)
    monkeypatch.setattr(MODULE, "MOTIONCRAFTER_REVISION", "f" * 40)
    monkeypatch.setattr(MODULE, "OBJECT_COUNT", 2)
    monkeypatch.setattr(MODULE, "ADMITTED_STREAM_COUNT", 6)
    monkeypatch.setattr(MODULE, "SUPPORTED_STREAM_COUNT", 4)
    monkeypatch.setattr(MODULE, "SUPPORT_NEGATIVE_STREAM_COUNT", 2)
    monkeypatch.setattr(MODULE, "TECHNICAL_FAILURE_STREAM_COUNT", 0)
    monkeypatch.setattr(MODULE, "SUPPORTED_OBJECT_COUNT", 2)
    monkeypatch.setattr(
        MODULE,
        "SUPPORT_NEGATIVE_OBJECT_COUNTS_BY_STRATUM",
        {"sheet": 1, "volumetric": 1},
    )
    monkeypatch.setattr(MODULE, "SUPPORT_NEGATIVE_CAMERA_COUNT", 2)


def _metric_job(
    *,
    object_id: str,
    episode_id: int,
    stratum: str,
    camera_id: str,
    supported: bool,
) -> dict[str, object]:
    job_id = _label_digest(f"{object_id}:{camera_id}")
    return {
        "job_id": job_id,
        "object_id": object_id,
        "episode_id": episode_id,
        "stratum": stratum,
        "camera_id": camera_id,
        "output_relative_directory": (
            f"objects/{object_id}/episode_{episode_id:04d}/views/{camera_id}"
        ),
        "status": "supported" if supported else "support-negative",
        "metric_artifact_id": _label_digest(f"metric:{job_id}") if supported else None,
        "projected_point_count": 7 if supported else 0,
        "failure_reason": None if supported else MODULE.SUPPORT_NEGATIVE_REASON,
        "failure_detail_sha256": None,
    }


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, object], dict[str, object]]:
    _configure_small_contract(monkeypatch)
    root = tmp_path / "source"
    jobs = [
        _metric_job(
            object_id="object-a",
            episode_id=1,
            stratum="sheet",
            camera_id="camera-0",
            supported=True,
        ),
        _metric_job(
            object_id="object-a",
            episode_id=1,
            stratum="sheet",
            camera_id="camera-1",
            supported=False,
        ),
        _metric_job(
            object_id="object-a",
            episode_id=1,
            stratum="sheet",
            camera_id="camera-2",
            supported=True,
        ),
        _metric_job(
            object_id="object-b",
            episode_id=2,
            stratum="volumetric",
            camera_id="camera-0",
            supported=True,
        ),
        _metric_job(
            object_id="object-b",
            episode_id=2,
            stratum="volumetric",
            camera_id="camera-1",
            supported=True,
        ),
        _metric_job(
            object_id="object-b",
            episode_id=2,
            stratum="volumetric",
            camera_id="camera-2",
            supported=False,
        ),
    ]
    jobs.sort(key=lambda row: (row["object_id"], row["camera_id"], row["job_id"]))
    metric: dict[str, object] = {
        "schema": "bayesian-phystwin.deform360-prob4d-metric-batch",
        "schema_version": 1,
        "semantics": "all-sealed-calibration-streams-released-robot-gauge-v1",
        "implementation_revision": MODULE.SOURCE_HEAD_SHA,
        "production_result_id": MODULE.PRODUCTION_RESULT_ID,
        "admission_id": MODULE.ADMISSION_ID,
        "object_count": MODULE.OBJECT_COUNT,
        "admitted_stream_count": MODULE.ADMITTED_STREAM_COUNT,
        "supported_stream_count": MODULE.SUPPORTED_STREAM_COUNT,
        "support_negative_stream_count": MODULE.SUPPORT_NEGATIVE_STREAM_COUNT,
        "technical_failure_stream_count": MODULE.TECHNICAL_FAILURE_STREAM_COUNT,
        "supported_object_count": MODULE.SUPPORTED_OBJECT_COUNT,
        "plan_emitted": False,
        "plan_file": None,
        "status": "support-negatives-retained",
        "jobs": jobs,
        "source_artifacts": {
            "metric-prior-policy.json": "1" * 64,
            "prepared-source-inventory.json": "2" * 64,
            "selection.json": "3" * 64,
            "visual-production-result.json": "4" * 64,
            "visual-provider-spec.json": "5" * 64,
        },
        "information_boundary": dict(MODULE.METRIC_BOUNDARY),
        "claim_boundary": MODULE.CLAIM_BOUNDARY,
    }
    metric["result_id"] = MODULE._content_id(metric)
    monkeypatch.setattr(MODULE, "METRIC_BATCH_RESULT_ID", metric["result_id"])
    support: dict[str, object] = {
        "schema": "bayesian-phystwin.deform360-prob4d-source-support-receipt",
        "schema_version": 1,
        "implementation_revision": MODULE.SOURCE_HEAD_SHA,
        "metric_batch_step_outcome": "success",
        "metric_batch_result_id": metric["result_id"],
        "metric_batch_status": "support-negatives-retained",
        "metric_batch_stderr_sha256": MODULE.EMPTY_SHA256,
        "new_measurements_required": False,
        "human_approval_required": False,
        "confirmation_payloads_opened": False,
        "target_outcomes_used": False,
    }
    pipeline: dict[str, object] = {
        "schema": "bayesian-phystwin.deform360-prob4d-source-pipeline-receipt",
        "schema_version": 1,
        "implementation_revision": MODULE.SOURCE_HEAD_SHA,
        "visual_production_result_id": MODULE.PRODUCTION_RESULT_ID,
        "prob4d_revision": MODULE.PROB4D_REVISION,
        "motioncrafter_revision": MODULE.MOTIONCRAFTER_REVISION,
        "stage_outcomes": {
            "metric_batch": "success",
            "support_gate": "failure",
            "samples": "skipped",
            "calibration": "skipped",
            "source_gate": "skipped",
        },
        "stderr_sha256": {
            "metric-batch": MODULE.EMPTY_SHA256,
            "samples": MODULE.EMPTY_SHA256,
            "calibration": MODULE.EMPTY_SHA256,
            "source-gate": MODULE.EMPTY_SHA256,
        },
        "source_gate_result_id": None,
        "source_gate_passed": None,
        "confirmation_access_authorized": None,
        "information_boundary": dict(MODULE.PIPELINE_BOUNDARY),
    }
    _write_json(root / "metric-support/metric-batch-result.json", metric)
    _write_json(root / "metric-support/support-receipt.json", support)
    _write_json(root / "pipeline-receipt.json", pipeline)
    _write_source_checksums(root)
    return root, metric, pipeline


def _verify_output_checksums(root: Path) -> None:
    lines = (root / "SHA256SUMS").read_text(encoding="ascii").splitlines()
    for line in lines:
        digest, relative = line.split("  ", maxsplit=1)
        assert _sha256((root / relative).read_bytes()) == digest


def test_audit_accepts_the_early_terminal_support_negative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _metric, _pipeline = _fixture(tmp_path, monkeypatch)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (tmp_path / "unrelated-link").symlink_to(unrelated, target_is_directory=True)

    receipt = MODULE.audit_support_negative(
        source,
        tmp_path / "audit",
        auditor_revision="9" * 40,
    )

    assert receipt["audit_status"] == "validated-negative"
    assert receipt["terminal_stage"] == "support-gate"
    assert receipt["support_gate_passed"] is False
    assert receipt["confirmation_access_authorized"] is False
    assert receipt["supported_stream_count"] == 4
    assert receipt["support_negative_stream_count"] == 2
    identity = dict(receipt)
    declared_id = identity.pop("audit_id")
    assert MODULE._content_id(identity) == declared_id
    _verify_output_checksums(tmp_path / "audit")


def test_audit_rejects_a_rehashed_changed_support_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, metric, _pipeline = _fixture(tmp_path, monkeypatch)
    jobs = metric["jobs"]
    assert isinstance(jobs, list)
    negative = next(row for row in jobs if row["status"] == "support-negative")
    negative["failure_reason"] = "changed-support-reason"
    identity = dict(metric)
    identity.pop("result_id")
    metric["result_id"] = MODULE._content_id(identity)
    _write_json(source / "metric-support/metric-batch-result.json", metric)
    _write_source_checksums(source)

    with pytest.raises(ValueError, match="metric result identity changed"):
        MODULE.audit_support_negative(
            source,
            tmp_path / "audit",
            auditor_revision="9" * 40,
        )


def test_audit_rejects_a_decision_in_a_skipped_downstream_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _metric, pipeline = _fixture(tmp_path, monkeypatch)
    pipeline["source_gate_result_id"] = "8" * 64
    pipeline["source_gate_passed"] = False
    pipeline["confirmation_access_authorized"] = False
    _write_json(source / "pipeline-receipt.json", pipeline)
    _write_source_checksums(source)

    with pytest.raises(ValueError, match="skipped source gate contains a decision"):
        MODULE.audit_support_negative(
            source,
            tmp_path / "audit",
            auditor_revision="9" * 40,
        )


def test_support_negative_auditor_workflow_is_target_closed_and_exactly_bound() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    document = yaml.load(text, Loader=yaml.BaseLoader)

    assert isinstance(document, dict)
    assert "pull_request:" in text
    assert "push:" in text
    assert "workflow_dispatch:" not in text
    assert "runs-on: ubuntu-latest" in text
    assert "runs-on: self-hosted" not in text
    assert 'SOURCE_RUN_ID: "31297018948"' in text
    assert 'SOURCE_ARTIFACT_ID: "9033414269"' in text
    assert "7247a2a260509c4c226e7ca437aff09d090abf6d2ca08f471a2143ea7d4bf7de" in text
    assert "audit_deform360_prob4d_support_negative.py" in text
    assert "validated-negative" in text
    assert "support-gate" in text
    assert "supported_stream_count == 313" in text
    assert "support_negative_stream_count == 11" in text
    assert "confirmation_access_authorized == false" in text
    assert "confirmation payloads opened: \\`false\\`" in text
    assert "target outcomes used: \\`false\\`" in text
    assert "/mnt/lexar4tb" not in text
