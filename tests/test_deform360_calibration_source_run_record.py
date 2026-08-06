"""Contracts for non-sensitive Deform360 calibration-source run records."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from bayesian_phystwin.deform360_calibration_source_run_record import (
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


def _write_result(
    path: Path,
    *,
    supported_sheet: int = 5,
    supported_volumetric: int = 5,
) -> None:
    objects = []
    for stratum, supported in (
        ("sheet", supported_sheet),
        ("volumetric", supported_volumetric),
    ):
        for index in range(5):
            objects.append(
                {
                    "object_id": f"{stratum}-{index}",
                    "episode_id": index,
                    "stratum": stratum,
                    "status": (
                        "source_prepared"
                        if index < supported
                        else "technical_failure_without_replacement"
                    ),
                }
            )
    supported = supported_sheet + supported_volumetric
    payload: dict[str, Any] = {
        "schema": DEFORM360_CALIBRATION_SOURCE_RESULT_SCHEMA,
        "schema_version": 1,
        "protocol_id": DEFORM360_CALIBRATION_SOURCE_PROTOCOL_ID,
        "plan_sha256": "3" * 64,
        "download_sha256": "4" * 64,
        "dataset_revision": DEFORM360_DATASET_REVISION,
        "processing_revision": PROCESSING_REVISION,
        "objects": objects,
        "gate": {
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
        },
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
    payload["result_sha256"] = _canonical_sha256(
        payload,
        digest_key="result_sha256",
    )
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _rewrite_digest(path: Path, payload: dict[str, Any]) -> None:
    payload["result_sha256"] = _canonical_sha256(
        payload,
        digest_key="result_sha256",
    )
    path.write_text(json.dumps(payload), encoding="utf-8")


def _record(result: Path, **overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "source_revision": SOURCE_REVISION,
        "processing_revision": PROCESSING_REVISION,
        "workflow_run_id": 123,
        "workflow_run_attempt": 2,
        "workload_exit_code": 0,
        "confirmation_boundary_exit_code": 0,
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


def test_success_without_a_valid_result_fails_the_record_contract(
    tmp_path: Path,
) -> None:
    record = _record(tmp_path / "missing.json")

    assert record["status"] == "failed"
    assert record["exit_code"] == 4
    assert record["failure_stage"] == "result-contract"


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
            "--result-json",
            str(tmp_path / "missing.json"),
        ]
    )

    assert status == 5
    assert json.loads(output.read_text(encoding="utf-8"))["exit_code"] == 5
