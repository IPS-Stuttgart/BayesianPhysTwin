from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from bayesian_phystwin.deform360_provider_failure_census_v1 import (
    validate_deform360_provider_failure_census_payload,
)
from bayesian_phystwin.deform360_source_support_failure_evidence_v1 import (
    DEFORM360_SOURCE_SUPPORT_AGGREGATION_POLICY_ID,
    DEFORM360_SOURCE_SUPPORT_GLOBAL_REJECTION_REASON,
    DEFORM360_SOURCE_SUPPORT_NEGATIVE_REASON,
    Deform360SourceSupportEvidenceLockV1,
    build_deform360_source_support_failure_evidence_v1,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "scripts/science/materialize_deform360_source_support_failure_evidence_v1.py"
)
SPEC = importlib.util.spec_from_file_location(
    "deform360_source_support_failure_evidence_materializer",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
materializer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = materializer
SPEC.loader.exec_module(materializer)
materialize = materializer.materialize_deform360_source_support_failure_evidence_v1

WORKFLOW = Path(
    ".github/workflows/launch-deform360-provider-failure-census-v1-once.yml"
)
MANIFEST = Path("MANIFEST.in")

REVISION = "1" * 40
PRODUCTION_RESULT_ID = "2" * 64
ADMISSION_ID = "3" * 64
ARTIFACT_SHA = "4" * 64
SUPPORT_RECEIPT_SHA = "5" * 64
PIPELINE_RECEIPT_SHA = "6" * 64
SOURCE_ARTIFACT_ID = 1234
SOURCE_RUN_ID = 5678
SOURCE_ARTIFACT_NAME = "synthetic-source-support-artifact"
SOURCE_ARTIFACTS = {
    "metric-prior-policy.json": "7" * 64,
    "prepared-source-inventory.json": "8" * 64,
    "selection.json": "9" * 64,
    "visual-production-result.json": "a" * 64,
    "visual-provider-spec.json": "b" * 64,
}
BOUNDARY = {
    "calibration_robot_state_access_attempted": True,
    "calibration_robot_state_opened": True,
    "calibration_camera_calibration_opened": True,
    "calibration_camera_images_opened": False,
    "calibration_tactile_payloads_opened": False,
    "rendered_depth_opened": False,
    "confirmation_payloads_opened": False,
    "target_outcomes_used": False,
    "future_frames_used": False,
    "replacement_allowed": False,
    "human_approval_required": False,
}
OBJECTS = (
    ("026-sock-cloth", 7, "sheet", 36, ("brics-odroid-002_cam0",)),
    ("031-cotton-cloth", 0, "sheet", 32, ()),
    ("036-napkin-cloth", 9, "sheet", 32, ("brics-odroid-025_cam0",)),
    (
        "058-roll-napkin",
        1,
        "volumetric",
        32,
        ("brics-odroid-002_cam0", "brics-odroid-007_cam1"),
    ),
    (
        "152-slime",
        8,
        "volumetric",
        32,
        (
            "brics-odroid-002_cam0",
            "brics-odroid-012_cam1",
            "brics-odroid-016_cam0",
        ),
    ),
    (
        "153-cake",
        5,
        "volumetric",
        32,
        ("brics-odroid-002_cam0", "brics-odroid-016_cam0"),
    ),
    (
        "167-glove-gray-cloth",
        0,
        "sheet",
        32,
        ("brics-odroid-002_cam0", "brics-odroid-007_cam1"),
    ),
    ("186-monster", 6, "volumetric", 32, ()),
    ("193-frog", 7, "volumetric", 32, ()),
    ("198-kneepad-cloth", 2, "sheet", 32, ()),
)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _metric_batch() -> dict[str, Any]:
    jobs: list[dict[str, object]] = []
    for object_id, episode_id, stratum, admitted, negative_cameras in OBJECTS:
        cameras = list(negative_cameras)
        index = 0
        while len(cameras) < admitted:
            candidate = f"synthetic-camera-{index:03d}"
            index += 1
            if candidate not in cameras:
                cameras.append(candidate)
        for camera_id in cameras:
            negative = camera_id in negative_cameras
            job_id = _digest(f"{object_id}:{camera_id}")
            jobs.append(
                {
                    "job_id": job_id,
                    "object_id": object_id,
                    "episode_id": episode_id,
                    "stratum": stratum,
                    "camera_id": camera_id,
                    "output_relative_directory": (
                        f"objects/{object_id}/episode_{episode_id:04d}/views/"
                        f"{camera_id}"
                    ),
                    "status": "support-negative" if negative else "supported",
                    "metric_artifact_id": (
                        None if negative else _digest(f"metric:{job_id}")
                    ),
                    "projected_point_count": 0 if negative else 10,
                    "failure_reason": (
                        DEFORM360_SOURCE_SUPPORT_NEGATIVE_REASON if negative else None
                    ),
                    "failure_detail_sha256": None,
                }
            )
    identity: dict[str, object] = {
        "schema": "bayesian-phystwin.deform360-prob4d-metric-batch",
        "schema_version": 1,
        "semantics": "all-sealed-calibration-streams-released-robot-gauge-v1",
        "implementation_revision": REVISION,
        "production_result_id": PRODUCTION_RESULT_ID,
        "admission_id": ADMISSION_ID,
        "object_count": 10,
        "admitted_stream_count": 324,
        "supported_stream_count": 313,
        "support_negative_stream_count": 11,
        "technical_failure_stream_count": 0,
        "supported_object_count": 10,
        "plan_emitted": False,
        "plan_file": None,
        "status": "support-negatives-retained",
        "jobs": jobs,
        "source_artifacts": dict(SOURCE_ARTIFACTS),
        "information_boundary": dict(BOUNDARY),
        "claim_boundary": "synthetic source-only metric batch",
    }
    return {
        **identity,
        "result_id": hashlib.sha256(
            json.dumps(
                identity,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
    }


def _refresh_result_id(batch: dict[str, Any]) -> None:
    identity = {key: value for key, value in batch.items() if key != "result_id"}
    batch["result_id"] = hashlib.sha256(
        json.dumps(
            identity,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _lock(
    batch: dict[str, Any],
    *,
    relative_path: str = "source/metric-batch-result.json",
    payload_bytes: bytes | None = None,
) -> Deform360SourceSupportEvidenceLockV1:
    raw = _canonical_bytes(batch) if payload_bytes is None else payload_bytes
    return Deform360SourceSupportEvidenceLockV1(
        source_workflow_run_id=SOURCE_RUN_ID,
        source_workflow_run_attempt=1,
        source_revision=REVISION,
        source_artifact_id=SOURCE_ARTIFACT_ID,
        source_artifact_name=SOURCE_ARTIFACT_NAME,
        source_artifact_sha256=ARTIFACT_SHA,
        metric_batch_relative_path=relative_path,
        metric_batch_sha256=hashlib.sha256(raw).hexdigest(),
        metric_batch_bytes=len(raw),
        metric_batch_result_id=cast(str, batch["result_id"]),
        production_result_id=PRODUCTION_RESULT_ID,
        admission_id=ADMISSION_ID,
        support_receipt_sha256=SUPPORT_RECEIPT_SHA,
        pipeline_receipt_sha256=PIPELINE_RECEIPT_SHA,
    )


def _report(batch: dict[str, Any]) -> tuple[dict[str, object], dict[str, object]]:
    payload = build_deform360_source_support_failure_evidence_v1(
        batch,
        lock=_lock(batch),
    )
    return payload, validate_deform360_provider_failure_census_payload(payload)


def test_equal_object_evidence_preserves_the_terminal_negative() -> None:
    payload, report = _report(_metric_batch())

    assert payload["provider_id"] == PRODUCTION_RESULT_ID
    assert [record["case_id"] for record in payload["records"]] == [
        row[0] for row in OBJECTS
    ]
    assert report["record_count"] == 10
    assert report["accepted_count"] == 0
    assert report["classified_rejection_count"] == 6
    assert report["unresolved_rejection_count"] == 4
    assert report["primary_category_counts"]["unsupported-provider-geometry"] == 6
    assert report["primary_category_counts"]["unresolved-rejection"] == 4
    metadata = payload["metadata"]
    assert metadata["aggregation_policy_id"] == (
        DEFORM360_SOURCE_SUPPORT_AGGREGATION_POLICY_ID
    )
    assert metadata["confirmation_payloads_opened"] is False
    assert metadata["adaptive_confirmation_payloads_opened"] is False
    assert metadata["target_outcomes_used"] is False
    assert metadata["future_frames_used"] is False
    assert metadata["replacement_allowed"] is False


def test_complete_support_objects_remain_unresolved_not_accepted() -> None:
    payload, _report_value = _report(_metric_batch())
    records = {record["case_id"]: record for record in payload["records"]}

    for object_id in (
        "031-cotton-cloth",
        "186-monster",
        "193-frog",
        "198-kneepad-cloth",
    ):
        record = records[object_id]
        assert record["accepted"] is False
        assert record["result_reason"] == (
            DEFORM360_SOURCE_SUPPORT_GLOBAL_REJECTION_REASON
        )
        assert record["signals"]["provider_support_complete"] is True
        assert record["metrics"]["support_negative_stream_count"] == 0


def test_negative_objects_bind_exact_cameras_and_jobs() -> None:
    payload, _report_value = _report(_metric_batch())
    records = {record["case_id"]: record for record in payload["records"]}

    for object_id, _episode, _stratum, _admitted, cameras in OBJECTS:
        record = records[object_id]
        assert record["metrics"]["support_negative_camera_ids"] == list(cameras)
        assert len(record["metrics"]["support_negative_job_ids"]) == len(cameras)
        assert record["signals"]["provider_support_complete"] is (not cameras)
        if cameras:
            assert record["result_reason"] == DEFORM360_SOURCE_SUPPORT_NEGATIVE_REASON


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("schema", "schema changed"),
        ("revision", "implementation revision changed"),
        ("boundary", "information boundary changed"),
        ("plan", "metric plan must not exist"),
        ("drop-job", "must retain all 324 jobs"),
        ("technical", "technical or unknown job status"),
        ("camera", "support-negative cameras changed"),
        ("duplicate", "repeats an object/camera stream"),
        ("count", "accounting changed"),
        ("result-id", "result ID changed"),
    ],
)
def test_metric_batch_mutations_fail_closed(mutation: str, match: str) -> None:
    batch = _metric_batch()
    if mutation == "schema":
        batch["schema"] = "wrong"
    elif mutation == "revision":
        batch["implementation_revision"] = "f" * 40
    elif mutation == "boundary":
        batch["information_boundary"]["confirmation_payloads_opened"] = True
    elif mutation == "plan":
        batch["plan_emitted"] = True
    elif mutation == "drop-job":
        cast(list[dict[str, object]], batch["jobs"]).pop()
    elif mutation == "technical":
        cast(list[dict[str, object]], batch["jobs"])[0]["status"] = "technical-failure"
    elif mutation == "camera":
        jobs = cast(list[dict[str, object]], batch["jobs"])
        negative = next(job for job in jobs if job["status"] == "support-negative")
        negative["camera_id"] = "wrong-camera"
        negative["output_relative_directory"] = (
            f"objects/{negative['object_id']}/episode_{negative['episode_id']:04d}/"
            "views/wrong-camera"
        )
    elif mutation == "duplicate":
        jobs = cast(list[dict[str, object]], batch["jobs"])
        jobs[1]["camera_id"] = jobs[0]["camera_id"]
        jobs[1]["output_relative_directory"] = jobs[0]["output_relative_directory"]
    elif mutation == "count":
        batch["supported_stream_count"] = 312
    elif mutation == "result-id":
        batch["result_id"] = "f" * 64
    if mutation != "result-id":
        _refresh_result_id(batch)
    with pytest.raises(ValueError, match=match):
        build_deform360_source_support_failure_evidence_v1(
            batch,
            lock=_lock(batch),
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("source_workflow_run_id", True, "genuine integer"),
        ("source_revision", "bad", "full Git revision"),
        ("source_artifact_sha256", "bad", "SHA-256"),
        ("metric_batch_relative_path", "../bad.json", "canonical relative"),
        ("metric_batch_bytes", 0, "at least 1"),
    ],
)
def test_source_lock_is_literal_and_content_addressed(
    field: str,
    value: object,
    match: str,
) -> None:
    batch = _metric_batch()
    lock = _lock(batch)
    with pytest.raises(ValueError, match=match):
        replace(lock, **{field: value})


def test_materializer_is_atomic_and_reuses_only_identical_content(
    tmp_path: Path,
) -> None:
    batch = _metric_batch()
    raw = _canonical_bytes(batch)
    relative = "source/metric-batch-result.json"
    source = tmp_path / relative
    source.parent.mkdir(parents=True)
    source.write_bytes(raw)
    lock = _lock(batch, relative_path=relative, payload_bytes=raw)

    first = materialize(tmp_path, lock=lock)
    second = materialize(tmp_path, lock=lock)

    assert first["reused_existing"] is False
    assert second["reused_existing"] is True
    assert first["evidence_sha256"] == second["evidence_sha256"]
    target = tmp_path / cast(str, first["materialization_directory_relative"])
    assert sorted(path.name for path in target.iterdir()) == [
        "SHA256SUMS",
        "materialization-receipt.json",
        "provider-failure-evidence.json",
    ]
    evidence = target / "provider-failure-evidence.json"
    assert hashlib.sha256(evidence.read_bytes()).hexdigest() == first["evidence_sha256"]
    receipt = json.loads((target / "materialization-receipt.json").read_text())
    assert receipt["classified_rejection_count"] == 6
    assert receipt["unresolved_rejection_count"] == 4


def test_materializer_rejects_source_drift_and_published_tampering(
    tmp_path: Path,
) -> None:
    batch = _metric_batch()
    raw = _canonical_bytes(batch)
    relative = "source/metric-batch-result.json"
    source = tmp_path / relative
    source.parent.mkdir(parents=True)
    source.write_bytes(raw)
    lock = _lock(batch, relative_path=relative, payload_bytes=raw)

    summary = materialize(tmp_path, lock=lock)
    target = tmp_path / cast(str, summary["materialization_directory_relative"])
    evidence = target / "provider-failure-evidence.json"
    evidence.write_bytes(evidence.read_bytes() + b" ")
    with pytest.raises(ValueError, match="published evidence bytes changed"):
        materialize(tmp_path, lock=lock)

    restored = build_deform360_source_support_failure_evidence_v1(
        batch,
        lock=lock,
    )
    evidence.write_bytes(_canonical_bytes(restored))
    source.write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="SHA-256 changed"):
        materialize(tmp_path, lock=lock)


def test_one_shot_workflow_is_source_only_and_binds_the_frozen_result() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    document = yaml.load(text, Loader=yaml.BaseLoader)

    assert isinstance(document, dict)
    assert "pull_request:" in text
    assert "push:" in text
    assert "branches: [main]" in text
    assert "runs-on: self-hosted" in text
    assert "AUTHORIZED_RUNNER_NAME: workstation2" in text
    assert "31297018948" in text
    assert "9033414269" in text
    assert "679550aff53d3b615f63c66ee78318258893867511dd6c33100d1cf10c0f5be6" in text
    assert "materialize_deform360_source_support_failure_evidence_v1.py" in text
    assert "validate_deform360_provider_failure_census_payload" in text
    assert "bpt diagnostic run diagnose-provider-failures" in text
    assert ".classified_rejection_count == 6" in text
    assert ".unresolved_rejection_count == 4" in text
    assert '.primary_category_counts["unsupported-provider-geometry"] == 6' in text
    assert "confirmation_payloads_opened=false" in text
    assert "adaptive_confirmation_payloads_opened=false" in text
    assert "target_outcomes_used=false" in text
    assert "future_frames_used=false" in text
    assert "replacement_allowed=false" in text
    assert "actions/upload-artifact@v7" in text
    assert "MANIFEST.in" in text
    manifest = MANIFEST.read_text(encoding="utf-8")
    assert "include docs/deform360_source_support_failure_evidence_v1.md" in manifest
    assert (
        "include scripts/science/"
        "materialize_deform360_source_support_failure_evidence_v1.py" in manifest
    )
    assert "issues: write" in text
    assert "secrets." not in text
    assert "HF_TOKEN" not in text
    assert 'find "${DEFORM360_OFFICIAL_RAW_ROOT}"' not in text
    assert 'find "${DEFORM360_ADAPTIVE_CONFIRMATION_RAW_ROOT}"' not in text
