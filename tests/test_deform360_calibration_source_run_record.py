"""Contracts for non-sensitive Deform360 calibration-source run records."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from bayesian_phystwin.deform360_calibration_source_run_record import (
    DEFORM360_CALIBRATION_DOWNLOAD_SCHEMA,
    DEFORM360_CALIBRATION_SOURCE_PLAN_SCHEMA,
    DEFORM360_CALIBRATION_SOURCE_PROTOCOL_ID,
    DEFORM360_CALIBRATION_SOURCE_RESULT_SCHEMA,
    DEFORM360_CALIBRATION_SOURCE_RUN_SCHEMA,
    DEFORM360_DATASET_REVISION,
    _canonical_sha256,
    build_deform360_calibration_source_run_record,
    main,
    save_deform360_calibration_source_run_record,
)

SOURCE_REVISION = "1" * 40
PROCESSING_REVISION = "2" * 40


def _object_rows(
    *,
    supported_sheet: int,
    supported_volumetric: int,
    supported_status: str,
    unsupported_status: str,
) -> list[dict[str, Any]]:
    rows = []
    for stratum, supported in (
        ("sheet", supported_sheet),
        ("volumetric", supported_volumetric),
    ):
        for index in range(5):
            rows.append(
                {
                    "object_id": f"{stratum}-{index}",
                    "episode_id": index,
                    "stratum": stratum,
                    "status": (
                        supported_status if index < supported else unsupported_status
                    ),
                }
            )
    return rows


def _gate(supported_sheet: int, supported_volumetric: int) -> dict[str, Any]:
    supported = supported_sheet + supported_volumetric
    return {
        "supported_object_count": supported,
        "supported_by_stratum": {
            "sheet": supported_sheet,
            "volumetric": supported_volumetric,
        },
        "minimum_supported_objects": 8,
        "minimum_supported_per_stratum": 4,
        "support_passed": (
            supported >= 8
            and supported_sheet >= 4
            and supported_volumetric >= 4
        ),
    }


def _write_digest_json(
    path: Path,
    payload: dict[str, Any],
    *,
    digest_key: str,
) -> str:
    payload[digest_key] = _canonical_sha256(payload, digest_key=digest_key)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return str(payload[digest_key])


def _write_plan(
    path: Path,
    *,
    planned_sheet: int = 5,
    planned_volumetric: int = 5,
) -> tuple[str, list[str]]:
    rows = _object_rows(
        supported_sheet=planned_sheet,
        supported_volumetric=planned_volumetric,
        supported_status="planned",
        unsupported_status="unsupported_without_replacement",
    )
    payload: dict[str, Any] = {
        "schema": DEFORM360_CALIBRATION_SOURCE_PLAN_SCHEMA,
        "schema_version": 1,
        "protocol_id": DEFORM360_CALIBRATION_SOURCE_PROTOCOL_ID,
        "dataset_revision": DEFORM360_DATASET_REVISION,
        "processing_revision": PROCESSING_REVISION,
        "objects": rows,
        "gate": _gate(planned_sheet, planned_volumetric),
        "information_boundary": {
            "repository_names_opened": True,
            "calibration_payloads_opened": False,
            "confirmation_payloads_opened": False,
            "target_outcomes_used": False,
            "replacement_allowed": False,
        },
    }
    digest = _write_digest_json(path, payload, digest_key="plan_sha256")
    object_ids = sorted(
        row["object_id"] for row in rows if row["status"] == "planned"
    )
    return digest, object_ids


def _write_download(
    path: Path,
    *,
    plan_sha256: str,
    object_ids: list[str],
) -> str:
    payload: dict[str, Any] = {
        "schema": DEFORM360_CALIBRATION_DOWNLOAD_SCHEMA,
        "schema_version": 1,
        "protocol_id": DEFORM360_CALIBRATION_SOURCE_PROTOCOL_ID,
        "plan_sha256": plan_sha256,
        "dataset_revision": DEFORM360_DATASET_REVISION,
        "data_root": "/not-published",
        "files": [],
        "object_ids": object_ids,
        "information_boundary": {
            "calibration_payloads_opened": True,
            "confirmation_payloads_opened": False,
            "target_outcomes_used": False,
            "replacement_allowed": False,
        },
    }
    return _write_digest_json(path, payload, digest_key="download_sha256")


def _write_result(
    path: Path,
    *,
    supported_sheet: int = 5,
    supported_volumetric: int = 5,
) -> None:
    plan_path = path.with_name("plan.json")
    download_path = path.with_name("download.json")
    plan_sha256, object_ids = _write_plan(plan_path)
    download_sha256 = _write_download(
        download_path,
        plan_sha256=plan_sha256,
        object_ids=object_ids,
    )
    payload: dict[str, Any] = {
        "schema": DEFORM360_CALIBRATION_SOURCE_RESULT_SCHEMA,
        "schema_version": 1,
        "protocol_id": DEFORM360_CALIBRATION_SOURCE_PROTOCOL_ID,
        "plan_sha256": plan_sha256,
        "download_sha256": download_sha256,
        "dataset_revision": DEFORM360_DATASET_REVISION,
        "processing_revision": PROCESSING_REVISION,
        "objects": _object_rows(
            supported_sheet=supported_sheet,
            supported_volumetric=supported_volumetric,
            supported_status="source_prepared",
            unsupported_status="technical_failure_without_replacement",
        ),
        "gate": _gate(supported_sheet, supported_volumetric),
        "next_stage": "frozen calibration candidates",
        "information_boundary": {
            "calibration_camera_payloads_opened": True,
            "calibration_tactile_payloads_opened": True,
            "calibration_robot_state_derived": True,
            "calibration_target_metrics_computed": False,
            "confirmation_payloads_opened": False,
            "target_outcomes_used": False,
            "replacement_allowed": False,
        },
    }
    _write_digest_json(path, payload, digest_key="result_sha256")


def _rewrite_digest(
    path: Path,
    payload: dict[str, Any],
    *,
    digest_key: str = "result_sha256",
) -> None:
    _write_digest_json(path, payload, digest_key=digest_key)


def _record(result: Path, **overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "source_revision": SOURCE_REVISION,
        "processing_revision": PROCESSING_REVISION,
        "workflow_run_id": 123,
        "workflow_run_attempt": 2,
        "workload_exit_code": 0,
        "confirmation_boundary_exit_code": 0,
        "plan_json": result.with_name("plan.json"),
        "download_json": result.with_name("download.json"),
        "result_json": result,
    }
    values.update(overrides)
    return build_deform360_calibration_source_run_record(**values)


def test_success_record_is_content_addressed_and_non_sensitive(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    _write_result(result)

    record = _record(result)

    assert record["schema"] == DEFORM360_CALIBRATION_SOURCE_RUN_SCHEMA
    assert record["status"] == "succeeded"
    assert record["exit_code"] == 0
    assert record["confirmation_boundary_verified"] is True
    assert record["confirmation_payloads_opened"] is False
    assert record["plan_available"] is True
    assert record["plan_valid"] is True
    assert record["download_available"] is True
    assert record["download_valid"] is True
    assert record["result_available"] is True
    assert record["result_valid"] is True
    assert record["support_gate"]["supported_object_count"] == 10
    assert record["record_sha256"] == _canonical_sha256(record)
    serialized = json.dumps(record, sort_keys=True)
    assert str(tmp_path) not in serialized
    assert "object_id" not in serialized
    assert '"objects":' not in serialized


def test_support_gate_failure_is_distinct_from_infrastructure_failure(
    tmp_path: Path,
) -> None:
    result = tmp_path / "result.json"
    _write_result(result, supported_sheet=4, supported_volumetric=3)

    record = _record(result, workload_exit_code=3)

    assert record["status"] == "failed"
    assert record["exit_code"] == 3
    assert record["failure_stage"] == "calibration-source-support-gate"
    assert record["result_valid"] is True
    assert record["support_gate"]["support_passed"] is False


def test_names_only_admission_gate_is_distinct_from_processing_gate(
    tmp_path: Path,
) -> None:
    plan = tmp_path / "plan.json"
    _write_plan(
        plan,
        planned_sheet=4,
        planned_volumetric=3,
    )

    record = _record(
        tmp_path / "missing-result.json",
        workload_exit_code=3,
        plan_json=plan,
        download_json=tmp_path / "missing-download.json",
    )

    assert record["status"] == "failed"
    assert record["exit_code"] == 3
    assert record["failure_stage"] == "calibration-source-admission-gate"
    assert record["plan_valid"] is True
    assert record["plan_support_gate"]["support_passed"] is False
    assert record["download_available"] is False
    assert record["result_available"] is False


def test_failed_workload_retains_failure_without_requiring_a_result(
    tmp_path: Path,
) -> None:
    record = _record(
        tmp_path / "missing.json",
        workload_exit_code=5,
    )

    assert record["status"] == "failed"
    assert record["exit_code"] == 5
    assert record["failure_stage"] == "calibration-source-workload"
    assert record["result_available"] is False
    assert record["result_error"] == "missing"
    assert record["confirmation_payloads_opened"] is False


def test_boundary_failure_overrides_workload_and_does_not_claim_absence(
    tmp_path: Path,
) -> None:
    record = _record(
        tmp_path / "missing.json",
        workload_exit_code=3,
        confirmation_boundary_exit_code=9,
    )

    assert record["exit_code"] == 9
    assert record["failure_stage"] == "confirmation-boundary"
    assert record["confirmation_boundary_verified"] is False
    assert record["confirmation_payloads_opened"] is None


def test_success_without_artifacts_fails_at_the_first_missing_contract(
    tmp_path: Path,
) -> None:
    record = _record(tmp_path / "missing.json")

    assert record["status"] == "failed"
    assert record["exit_code"] == 4
    assert record["failure_stage"] == "plan-contract"


def test_missing_chain_stage_is_classified_precisely(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    plan = tmp_path / "plan.json"
    download = tmp_path / "download.json"
    plan_sha256, object_ids = _write_plan(plan)

    without_download = _record(result)
    assert without_download["failure_stage"] == "download-contract"

    _write_download(
        download,
        plan_sha256=plan_sha256,
        object_ids=object_ids,
    )
    without_result = _record(result)
    assert without_result["failure_stage"] == "result-contract"


@pytest.mark.parametrize(
    "payload",
    (
        '{"schema":"first","schema":"second"}',
        '{"value":NaN}',
    ),
)
def test_noncanonical_json_is_rejected_as_a_result_contract_failure(
    tmp_path: Path,
    payload: str,
) -> None:
    result = tmp_path / "result.json"
    _write_result(result)
    result.write_text(payload, encoding="utf-8")

    record = _record(result, workload_exit_code=1)

    assert record["exit_code"] == 4
    assert record["failure_stage"] == "result-contract"
    assert record["result_available"] is True
    assert record["result_valid"] is False
    assert record["result_error"] == "invalid-json"


def test_tampered_result_is_not_masked_by_a_workload_failure(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    _write_result(result)
    payload = json.loads(result.read_text(encoding="utf-8"))
    payload["information_boundary"]["target_outcomes_used"] = True
    _rewrite_digest(result, payload)

    record = _record(result, workload_exit_code=1)

    assert record["exit_code"] == 4
    assert record["failure_stage"] == "result-contract"
    assert record["result_available"] is True
    assert record["result_valid"] is False
    assert record["result_error"] == "invalid-contract"
    assert record["result_sha256"] is None
    assert record["support_gate"] is None
    assert "target_outcomes_used" not in json.dumps(record, sort_keys=True)


def test_gate_thresholds_and_object_rows_are_frozen(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    _write_result(result)
    payload = json.loads(result.read_text(encoding="utf-8"))
    payload["gate"]["minimum_supported_objects"] = 0
    _rewrite_digest(result, payload)
    assert _record(result)["result_valid"] is False

    _write_result(result)
    payload = json.loads(result.read_text(encoding="utf-8"))
    payload["objects"][0]["status"] = "technical_failure_without_replacement"
    _rewrite_digest(result, payload)
    assert _record(result)["result_valid"] is False


def test_plan_download_and_result_digests_form_one_chain(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    plan = tmp_path / "plan.json"
    download = tmp_path / "download.json"
    _write_result(result)

    plan_payload = json.loads(plan.read_text(encoding="utf-8"))
    plan_payload["audit_note"] = "redigested plan"
    _rewrite_digest(plan, plan_payload, digest_key="plan_sha256")
    plan_mismatch = _record(result, workload_exit_code=1)
    assert plan_mismatch["plan_valid"] is True
    assert plan_mismatch["download_valid"] is False
    assert plan_mismatch["failure_stage"] == "download-contract"

    _write_result(result)
    download_payload = json.loads(download.read_text(encoding="utf-8"))
    download_payload["audit_note"] = "redigested download"
    _rewrite_digest(
        download,
        download_payload,
        digest_key="download_sha256",
    )
    download_mismatch = _record(result, workload_exit_code=1)
    assert download_mismatch["download_valid"] is True
    assert download_mismatch["result_valid"] is False
    assert download_mismatch["failure_stage"] == "result-contract"


def test_result_cohort_is_bound_to_the_plan(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    _write_result(result)
    payload = json.loads(result.read_text(encoding="utf-8"))
    payload["objects"][0]["object_id"] = "different-object"
    _rewrite_digest(result, payload)

    record = _record(result, workload_exit_code=1)

    assert record["result_valid"] is False
    assert record["failure_stage"] == "result-contract"


def test_valid_result_and_workload_status_must_agree(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    _write_result(result)

    record = _record(result, workload_exit_code=1)

    assert record["exit_code"] == 4
    assert record["failure_stage"] == "result-contract"


def test_atomic_publication_does_not_replace_an_existing_record(
    tmp_path: Path,
) -> None:
    result = tmp_path / "result.json"
    output = tmp_path / "execution-manifest.json"
    _write_result(result)
    record = _record(result)

    save_deform360_calibration_source_run_record(record, output)
    first = output.read_bytes()
    with pytest.raises(FileExistsError):
        save_deform360_calibration_source_run_record(record, output)
    assert output.read_bytes() == first


def test_concurrent_publication_has_exactly_one_winner(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    output = tmp_path / "execution-manifest.json"
    _write_result(result)
    record = _record(result)

    def publish() -> bool:
        try:
            save_deform360_calibration_source_run_record(record, output)
        except FileExistsError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: publish(), range(2)))

    assert sorted(outcomes) == [False, True]
    assert json.loads(output.read_text(encoding="utf-8")) == record


def test_cli_writes_failure_record_and_returns_the_effective_status(
    tmp_path: Path,
) -> None:
    output = tmp_path / "execution-manifest.json"

    status = main(
        [
            "--output",
            str(output),
            "--source-revision",
            SOURCE_REVISION,
            "--processing-revision",
            PROCESSING_REVISION,
            "--workflow-run-id",
            "123",
            "--workflow-run-attempt",
            "1",
            "--workload-exit-code",
            "5",
            "--confirmation-boundary-exit-code",
            "0",
            "--plan-json",
            str(tmp_path / "missing-plan.json"),
            "--download-json",
            str(tmp_path / "missing-download.json"),
            "--result-json",
            str(tmp_path / "missing-result.json"),
        ]
    )

    assert status == 5
    assert json.loads(output.read_text(encoding="utf-8"))["exit_code"] == 5
