"""Adversarial validation of Deform360 calibration terminal records."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from test_deform360_calibration_source_run_record import _build_chain, _record

import bayesian_phystwin.deform360_calibration_source_run_record as public_record
from bayesian_phystwin._deform360_calibration_run_common import (
    RECORD_WRITE_EXIT_CODE,
)
from bayesian_phystwin.deform360_calibration_source_run_record import (
    _canonical_sha256,
    load_deform360_calibration_source_run_record,
    save_deform360_calibration_source_run_record,
    validate_deform360_calibration_source_run_record,
)


def _redigested(record: dict[str, Any]) -> dict[str, Any]:
    record["record_sha256"] = _canonical_sha256(record)
    return record


def _success_record(tmp_path: Path) -> dict[str, Any]:
    return _record(_build_chain(tmp_path))


def test_strict_validator_accepts_the_builder_output(tmp_path: Path) -> None:
    record = _success_record(tmp_path)

    validated = validate_deform360_calibration_source_run_record(record)

    assert validated == record
    assert validated is not record


def test_redigested_extra_field_cannot_enter_the_public_record(
    tmp_path: Path,
) -> None:
    record = deepcopy(_success_record(tmp_path))
    record["local_sensitive_path"] = "/private/calibration/object-identity"
    _redigested(record)
    output = tmp_path / "execution-manifest.json"

    with pytest.raises(ValueError, match="fields changed"):
        validate_deform360_calibration_source_run_record(record)
    with pytest.raises(ValueError, match="fields changed"):
        save_deform360_calibration_source_run_record(record, output)
    assert not output.exists()


def test_cli_rejects_a_redigested_record_that_fails_strict_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = deepcopy(_success_record(tmp_path))
    record["unexpected_private_field"] = "/private/object-id"
    _redigested(record)
    monkeypatch.setattr(
        public_record,
        "build_deform360_calibration_source_run_record",
        lambda **_kwargs: record,
    )
    output = tmp_path / "execution-manifest.json"

    status = public_record.main(
        [
            "--output",
            str(output),
            "--source-revision",
            "1" * 40,
            "--processing-revision",
            "2" * 40,
            "--workflow-run-id",
            "123",
            "--workflow-run-attempt",
            "1",
            "--workload-exit-code",
            "0",
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
    )

    assert status == RECORD_WRITE_EXIT_CODE
    assert not output.exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("status", "failed", "status is inconsistent"),
        ("exit_code", 9, "exit code is inconsistent"),
        (
            "failure_stage",
            "calibration-source-workload",
            "failure stage is inconsistent",
        ),
        (
            "confirmation_payloads_opened",
            True,
            "confirmation-payload statement is inconsistent",
        ),
        ("schema_version", True, "schema version changed"),
        ("workflow_run_id", True, "workflow_run_id"),
    ),
)
def test_redigesting_does_not_authorize_inconsistent_derived_fields(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    record = deepcopy(_success_record(tmp_path))
    record[field] = value
    _redigested(record)

    with pytest.raises(ValueError, match=message):
        validate_deform360_calibration_source_run_record(record)


@pytest.mark.parametrize(
    "mutation",
    (
        {
            "source_locks_valid": True,
            "source_locks_error": "invalid-contract",
        },
        {
            "source_locks_available": False,
            "source_locks_valid": True,
        },
        {
            "plan_valid": True,
            "plan_error": "invalid-contract",
        },
        {
            "plan_available": False,
            "plan_valid": True,
        },
        {
            "result_valid": False,
            "result_error": "invalid-contract",
        },
        {
            "download_valid": False,
            "download_error": "missing",
        },
    ),
)
def test_summary_shapes_are_recomputed_not_trusted(
    tmp_path: Path,
    mutation: dict[str, object],
) -> None:
    record = deepcopy(_success_record(tmp_path))
    record.update(mutation)
    _redigested(record)

    with pytest.raises(ValueError):
        validate_deform360_calibration_source_run_record(record)


def test_redigested_source_lock_digest_drift_is_rejected(tmp_path: Path) -> None:
    record = deepcopy(_success_record(tmp_path))
    record["selection_artifact_sha256"] = None
    _redigested(record)

    with pytest.raises(ValueError, match="selection_artifact_sha256"):
        validate_deform360_calibration_source_run_record(record)


def test_redigested_gate_threshold_drift_is_rejected(tmp_path: Path) -> None:
    record = deepcopy(_success_record(tmp_path))
    record["support_gate"]["minimum_supported_objects"] = 7
    _redigested(record)

    with pytest.raises(ValueError, match="minimum supported object count changed"):
        validate_deform360_calibration_source_run_record(record)


def test_strict_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    record = _success_record(tmp_path)
    path = tmp_path / "execution-manifest.json"
    encoded = json.dumps(record, sort_keys=True)
    path.write_text(
        '{"schema":"forged",' + encoded[1:],
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        load_deform360_calibration_source_run_record(path)


def test_strict_loader_round_trips_a_valid_record(tmp_path: Path) -> None:
    record = _success_record(tmp_path)
    path = tmp_path / "execution-manifest.json"
    path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    assert load_deform360_calibration_source_run_record(path) == record
