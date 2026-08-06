"""Contracts for non-sensitive Deform360 calibration-source run records."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

import pytest

import bayesian_phystwin._deform360_calibration_source_run_record_impl as impl
from bayesian_phystwin.deform360_calibration_source_run_record import (
    DEFORM360_CALIBRATION_SOURCE_RUN_SCHEMA,
    _canonical_sha256,
    build_deform360_calibration_source_run_record,
    main,
    save_deform360_calibration_source_run_record,
)

SOURCE_REVISION = "1" * 40
PROCESSING_REVISION = "2" * 40


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


def _valid_source_locks() -> dict[str, Any]:
    keys = (
        "source_protocol_file_sha256",
        "source_protocol_sha256",
        "stage0_protocol_file_sha256",
        "stage0_protocol_sha256",
        "selection_lock_file_sha256",
        "selection_artifact_sha256",
        "content_selection_sha256",
        "visual_provider_lock_file_sha256",
        "visual_provider_lock_id",
    )
    result: dict[str, Any] = {
        "source_locks_available": True,
        "source_locks_valid": True,
        "source_locks_error": None,
    }
    result.update({key: f"{index:064x}" for index, key in enumerate(keys, 1)})
    return result


def _invalid_source_locks(
    *,
    available: bool = True,
    error: str = "invalid-contract",
) -> dict[str, Any]:
    result = _valid_source_locks()
    result.update(
        {
            "source_locks_available": available,
            "source_locks_valid": False,
            "source_locks_error": error,
        }
    )
    for key in tuple(result):
        if key.endswith("sha256") or key == "visual_provider_lock_id":
            result[key] = None
    return result


def _valid_plan(*, supported_sheet: int = 5, supported_volumetric: int = 5):
    return {
        "plan_available": True,
        "plan_valid": True,
        "plan_error": None,
        "plan_file_sha256": "a" * 64,
        "plan_sha256": "b" * 64,
        "plan_support_gate": _gate(supported_sheet, supported_volumetric),
    }


def _valid_download() -> dict[str, Any]:
    return {
        "download_available": True,
        "download_valid": True,
        "download_error": None,
        "download_file_sha256": "c" * 64,
        "download_sha256": "d" * 64,
    }


def _valid_result(*, supported_sheet: int = 5, supported_volumetric: int = 5):
    return {
        "result_available": True,
        "result_valid": True,
        "result_error": None,
        "result_file_sha256": "e" * 64,
        "result_sha256": "f" * 64,
        "support_gate": _gate(supported_sheet, supported_volumetric),
    }


def _missing_download() -> dict[str, Any]:
    return {
        "download_available": False,
        "download_valid": False,
        "download_error": "missing",
        "download_file_sha256": None,
        "download_sha256": None,
    }


def _missing_result() -> dict[str, Any]:
    return {
        "result_available": False,
        "result_valid": False,
        "result_error": "missing",
        "result_file_sha256": None,
        "result_sha256": None,
        "support_gate": None,
    }


@contextmanager
def _summary_contracts(
    *,
    source_locks: dict[str, Any] | None = None,
    plan: dict[str, Any] | None = None,
    download: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
) -> Iterator[None]:
    source_locks = source_locks or _valid_source_locks()
    plan = plan or _valid_plan()
    download = download or _valid_download()
    result = result or _valid_result()
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                impl,
                "source_lock_summary",
                return_value=(source_locks, {}, frozenset()),
            )
        )
        stack.enter_context(
            patch.object(
                impl,
                "plan_summary",
                return_value=(plan, frozenset(), frozenset()),
            )
        )
        stack.enter_context(
            patch.object(impl, "download_summary", return_value=download)
        )
        stack.enter_context(
            patch.object(impl, "result_summary", return_value=result)
        )
        yield


def _write_result(path: Path, **_ignored: Any) -> None:
    path.write_text("{}\n", encoding="utf-8")


def _record(result: Path, **overrides: Any) -> dict[str, Any]:
    source_locks = overrides.pop("source_locks", None)
    plan = overrides.pop("plan", None)
    download = overrides.pop("download", None)
    result_summary = overrides.pop("result_summary", None)
    values: dict[str, Any] = {
        "source_revision": SOURCE_REVISION,
        "processing_revision": PROCESSING_REVISION,
        "workflow_run_id": 123,
        "workflow_run_attempt": 2,
        "workload_exit_code": 0,
        "confirmation_boundary_exit_code": 0,
        "source_protocol_json": result.with_name("source-protocol.json"),
        "stage0_protocol_json": result.with_name("stage0-protocol.json"),
        "selection_lock": result.with_name("selection-lock.json"),
        "visual_provider_lock": result.with_name("visual-provider-lock.json"),
        "plan_json": result.with_name("plan.json"),
        "download_json": result.with_name("download.json"),
        "result_json": result,
    }
    values.update(overrides)
    with _summary_contracts(
        source_locks=source_locks,
        plan=plan,
        download=download,
        result=result_summary,
    ):
        return build_deform360_calibration_source_run_record(**values)


def test_success_record_is_content_addressed_and_non_sensitive(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    _write_result(result)

    record = _record(result)

    assert record["schema"] == DEFORM360_CALIBRATION_SOURCE_RUN_SCHEMA
    assert record["status"] == "succeeded"
    assert record["exit_code"] == 0
    assert record["source_locks_valid"] is True
    assert record["confirmation_boundary_verified"] is True
    assert record["confirmation_payloads_opened"] is False
    assert record["plan_valid"] is True
    assert record["download_valid"] is True
    assert record["result_valid"] is True
    assert record["support_gate"]["supported_object_count"] == 10
    assert record["record_sha256"] == _canonical_sha256(record)
    serialized = json.dumps(record, sort_keys=True)
    assert str(tmp_path) not in serialized
    assert "object_id" not in serialized
    assert '"objects":' not in serialized


def test_source_lock_failure_precedes_every_artifact_stage(tmp_path: Path) -> None:
    result = tmp_path / "result.json"

    record = _record(result, source_locks=_invalid_source_locks())

    assert record["status"] == "failed"
    assert record["exit_code"] == 4
    assert record["failure_stage"] == "source-lock-contract"
    assert record["source_locks_valid"] is False
    assert record["source_protocol_sha256"] is None


def test_support_gate_failure_is_distinct_from_infrastructure_failure(
    tmp_path: Path,
) -> None:
    result = tmp_path / "result.json"

    record = _record(
        result,
        workload_exit_code=3,
        result_summary=_valid_result(supported_sheet=4, supported_volumetric=3),
    )

    assert record["status"] == "failed"
    assert record["exit_code"] == 3
    assert record["failure_stage"] == "calibration-source-support-gate"
    assert record["support_gate"]["support_passed"] is False


def test_names_only_admission_gate_is_distinct_from_processing_gate(
    tmp_path: Path,
) -> None:
    result = tmp_path / "result.json"

    record = _record(
        result,
        workload_exit_code=3,
        plan=_valid_plan(supported_sheet=4, supported_volumetric=3),
        download=_missing_download(),
        result_summary=_missing_result(),
    )

    assert record["exit_code"] == 3
    assert record["failure_stage"] == "calibration-source-admission-gate"
    assert record["plan_support_gate"]["support_passed"] is False


def test_failed_workload_retains_failure_without_requiring_a_result(
    tmp_path: Path,
) -> None:
    record = _record(
        tmp_path / "missing.json",
        workload_exit_code=5,
        result_summary=_missing_result(),
    )

    assert record["exit_code"] == 5
    assert record["failure_stage"] == "calibration-source-workload"
    assert record["result_error"] == "missing"


def test_boundary_failure_overrides_every_other_status(tmp_path: Path) -> None:
    record = _record(
        tmp_path / "missing.json",
        workload_exit_code=3,
        confirmation_boundary_exit_code=9,
        source_locks=_invalid_source_locks(),
    )

    assert record["exit_code"] == 9
    assert record["failure_stage"] == "confirmation-boundary"
    assert record["confirmation_boundary_verified"] is False
    assert record["confirmation_payloads_opened"] is None


def test_valid_result_and_workload_status_must_agree(tmp_path: Path) -> None:
    record = _record(tmp_path / "result.json", workload_exit_code=1)

    assert record["exit_code"] == 4
    assert record["failure_stage"] == "result-contract"


def test_atomic_publication_does_not_replace_an_existing_record(
    tmp_path: Path,
) -> None:
    output = tmp_path / "execution-manifest.json"
    record = _record(tmp_path / "result.json")

    save_deform360_calibration_source_run_record(record, output)
    first = output.read_bytes()
    with pytest.raises(FileExistsError):
        save_deform360_calibration_source_run_record(record, output)
    assert output.read_bytes() == first


def test_concurrent_publication_has_exactly_one_winner(tmp_path: Path) -> None:
    output = tmp_path / "execution-manifest.json"
    record = _record(tmp_path / "result.json")

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
    args = [
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
        "--source-protocol-json",
        str(tmp_path / "source-protocol.json"),
        "--stage0-protocol-json",
        str(tmp_path / "stage0-protocol.json"),
        "--selection-lock",
        str(tmp_path / "selection-lock.json"),
        "--visual-provider-lock",
        str(tmp_path / "visual-provider-lock.json"),
        "--plan-json",
        str(tmp_path / "plan.json"),
        "--download-json",
        str(tmp_path / "download.json"),
        "--result-json",
        str(tmp_path / "result.json"),
    ]

    with _summary_contracts(result=_missing_result()):
        status = main(args)

    assert status == 5
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["exit_code"] == 5
    assert saved["source_locks_valid"] is True
