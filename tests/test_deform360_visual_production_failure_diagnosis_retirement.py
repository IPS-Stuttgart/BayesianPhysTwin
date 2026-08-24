"""Contracts for the retired Deform360 production-failure diagnosis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bayesian_phystwin._portable_contracts import content_id

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "archive/github-actions/retired-one-shot-v1"
RECORD_PATH = (
    ROOT
    / "results/diagnostics/"
    "deform360_visual_production_failure_diagnosis_retirement_v1/"
    "retirement.json"
)
DIAGNOSIS_PATH = (
    ROOT / ".github/workflows/deform360-visual-production-failure-diagnosis.yml"
)
REPORTER_PATH = ROOT / ".github/workflows/deform360-failure-diagnosis-reporter.yml"
ARCHIVED_DIAGNOSIS = ARCHIVE / "deform360-visual-production-failure-diagnosis.yml"
ARCHIVED_REPORTER = ARCHIVE / "deform360-failure-diagnosis-reporter.yml"


def _record() -> dict[str, object]:
    return json.loads(RECORD_PATH.read_text(encoding="utf-8"))


def _git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def test_terminal_diagnosis_and_reporter_are_exactly_archived() -> None:
    assert not DIAGNOSIS_PATH.exists()
    assert not REPORTER_PATH.exists()

    assert ARCHIVED_DIAGNOSIS.stat().st_size == 17_987
    assert _git_blob_sha1(ARCHIVED_DIAGNOSIS) == (
        "b0f303c408dc66a5175e81cee38470fdbd02bd9b"
    )
    assert ARCHIVED_REPORTER.stat().st_size == 9_033
    assert _git_blob_sha1(ARCHIVED_REPORTER) == (
        "7d9b9d44eefe85fd825592bbe7490655935a44f6"
    )


def test_retirement_record_is_content_addressed_and_fail_closed() -> None:
    record = _record()
    declared = record.pop("record_id")
    assert declared == content_id(record)
    assert declared == (
        "d6d7476284dd79e17aa13716e162a56d649a113ca7c1ba18354d213be3208d4e"
    )

    diagnosis = record["diagnosis"]
    assert isinstance(diagnosis, dict)
    assert diagnosis["contract_run_id"] == 31277968816
    assert diagnosis["contract_run_conclusion"] == "success"
    assert diagnosis["workflow_run_id"] == 31278099099
    assert diagnosis["workflow_run_attempt"] == 1
    assert diagnosis["workflow_conclusion"] == "failure"
    assert diagnosis["failure_stage"] == (
        "verify-all-receipts-and-retain-one-common-stderr-representative"
    )
    assert diagnosis["failure_type"] == "ValueError"
    assert diagnosis["failure_message"] == (
        "production result production_result_id changed: None"
    )
    assert diagnosis["artifact_produced"] is False
    assert diagnosis["artifact_id"] is None
    assert diagnosis["confirmation_payloads_opened"] is False
    assert diagnosis["adaptive_confirmation_payloads_opened"] is False
    assert diagnosis["official_raw_payload_opened"] is False
    assert diagnosis["target_outcomes_used"] is False

    reporter = record["reporter"]
    assert isinstance(reporter, dict)
    assert reporter["final_workflow_run_id"] == 31278996852
    assert reporter["final_workflow_conclusion"] == "failure"
    assert reporter["failure_stage"] == "resolve-the-exact-trusted-main-diagnosis"
    assert reporter["diagnosis_artifact_downloaded"] is False
    assert reporter["issue_comment_published"] is False
    runs = reporter["workflow_runs"]
    assert isinstance(runs, list)
    assert [run["run_id"] for run in runs] == [
        31278360084,
        31278541397,
        31278773734,
        31278996852,
    ]
    assert {run["conclusion"] for run in runs} == {"failure"}

    retirement = record["retirement"]
    assert retirement == {
        "confirmation_authorized": False,
        "diagnosis_workflow_deleted": True,
        "replacement_allowed": False,
        "reporter_workflow_deleted": True,
        "rerun_authorized": False,
        "scientific_result_changed": False,
        "target_access_authorized": False,
    }


def test_later_source_support_remains_the_authoritative_terminal_result() -> None:
    supersession = _record()["supersession"]
    assert isinstance(supersession, dict)
    assert supersession["corrected_visual_production_run_id"] == 31279398563
    assert supersession["registered_source_support_run_id"] == 31297018948
    assert supersession["registered_source_support_artifact_id"] == 9033414269
    assert supersession["registered_source_support_artifact_sha256"] == (
        "7247a2a260509c4c226e7ca437aff09d090abf6d2ca08f471a2143ea7d4bf7de"
    )
    assert supersession["admitted_stream_count"] == 324
    assert supersession["supported_stream_count"] == 313
    assert supersession["retained_support_negative_count"] == 11
    assert supersession["terminal_status"] == "support-negatives-retained"
    assert supersession["confirmation_access_authorized"] is False
