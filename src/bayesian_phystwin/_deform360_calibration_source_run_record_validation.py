"""Strict validation for Deform360 calibration terminal records."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ._deform360_calibration_run_common import (
    DEFORM360_CALIBRATION_SOURCE_RUN_CLAIM_BOUNDARY,
    DEFORM360_CALIBRATION_SOURCE_RUN_SCHEMA,
    DEFORM360_CALIBRATION_SOURCE_RUN_SEMANTICS,
    DEFORM360_CALIBRATION_SOURCE_RUN_VERSION,
    DEFORM360_DATASET_REVISION,
    EXPECTED_OBJECT_COUNT,
    EXPECTED_OBJECTS_PER_STRATUM,
    EXPECTED_STRATA,
    canonical_sha256,
    exit_code,
    integer_field,
    load_json_object,
    positive_integer,
    revision,
    sha256,
    validated_support_gate,
)
from ._deform360_calibration_source_run_record_impl import _execution_outcome

_SOURCE_LOCK_DIGEST_KEYS = (
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
_RECORD_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
        "record_sha256",
        "status",
        "exit_code",
        "workload_exit_code",
        "failure_stage",
        "confirmation_boundary_exit_code",
        "confirmation_boundary_verified",
        "confirmation_payloads_opened",
        "source_revision",
        "dataset_revision",
        "processing_revision",
        "workflow_run_id",
        "workflow_run_attempt",
        "source_locks_available",
        "source_locks_valid",
        "source_locks_error",
        *_SOURCE_LOCK_DIGEST_KEYS,
        "plan_available",
        "plan_valid",
        "plan_error",
        "plan_file_sha256",
        "plan_sha256",
        "plan_support_gate",
        "download_available",
        "download_valid",
        "download_error",
        "download_file_sha256",
        "download_sha256",
        "result_available",
        "result_valid",
        "result_error",
        "result_file_sha256",
        "result_sha256",
        "support_gate",
        "claim_boundary",
    }
)
_ARTIFACT_ERRORS = frozenset(
    {"missing", "unreadable", "invalid-json", "invalid-contract"}
)


def _validated_record_gate(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    by_stratum = value.get("supported_by_stratum")
    if not isinstance(by_stratum, Mapping) or set(by_stratum) != set(EXPECTED_STRATA):
        raise ValueError(f"{name}.supported_by_stratum changed")
    sheet = integer_field(
        by_stratum.get("sheet"),
        name=f"{name}.supported_by_stratum.sheet",
        maximum=EXPECTED_OBJECTS_PER_STRATUM,
    )
    volumetric = integer_field(
        by_stratum.get("volumetric"),
        name=f"{name}.supported_by_stratum.volumetric",
        maximum=EXPECTED_OBJECTS_PER_STRATUM,
    )
    supported = integer_field(
        value.get("supported_object_count"),
        name=f"{name}.supported_object_count",
        maximum=EXPECTED_OBJECT_COUNT,
    )
    return validated_support_gate(
        {"gate": value},
        artifact=name,
        object_supported=supported,
        object_supported_by_stratum={
            "sheet": sheet,
            "volumetric": volumetric,
        },
    )


def _validate_source_lock_summary(record: Mapping[str, Any]) -> None:
    available = record["source_locks_available"]
    valid = record["source_locks_valid"]
    error = record["source_locks_error"]
    if type(available) is not bool or type(valid) is not bool:
        raise ValueError("source-lock availability flags must be booleans")
    if error is not None and (type(error) is not str or error not in _ARTIFACT_ERRORS):
        raise ValueError("source_locks_error changed")
    digests = [record[key] for key in _SOURCE_LOCK_DIGEST_KEYS]
    if valid:
        if not available or error is not None:
            raise ValueError("valid source-lock summary is inconsistent")
        for key, digest in zip(_SOURCE_LOCK_DIGEST_KEYS, digests, strict=True):
            sha256(digest, name=key)
        return
    if any(digest is not None for digest in digests):
        raise ValueError("invalid source-lock summary retained derived evidence")
    if not available:
        if error != "missing":
            raise ValueError("missing source-lock summary is inconsistent")
        return
    if error not in {"unreadable", "invalid-json", "invalid-contract"}:
        raise ValueError("available invalid source-lock summary is inconsistent")


def _validate_artifact_summary(
    record: Mapping[str, Any],
    *,
    prefix: str,
    gate_key: str | None,
) -> None:
    available = record[f"{prefix}_available"]
    valid = record[f"{prefix}_valid"]
    error = record[f"{prefix}_error"]
    file_digest = record[f"{prefix}_file_sha256"]
    artifact_digest = record[f"{prefix}_sha256"]
    gate = record[gate_key] if gate_key is not None else None

    if type(available) is not bool or type(valid) is not bool:
        raise ValueError(f"{prefix} availability flags must be booleans")
    if error is not None and (type(error) is not str or error not in _ARTIFACT_ERRORS):
        raise ValueError(f"{prefix}_error changed")

    if valid:
        if not available or error is not None:
            raise ValueError(f"valid {prefix} summary is inconsistent")
        sha256(file_digest, name=f"{prefix}_file_sha256")
        sha256(artifact_digest, name=f"{prefix}_sha256")
        if gate_key is not None:
            _validated_record_gate(gate, name=gate_key)
        return

    if artifact_digest is not None or gate is not None:
        raise ValueError(f"invalid {prefix} summary retained derived evidence")
    if not available:
        if error != "missing" or file_digest is not None:
            raise ValueError(f"missing {prefix} summary is inconsistent")
        return
    if error not in {"unreadable", "invalid-json", "invalid-contract"}:
        raise ValueError(f"available invalid {prefix} summary is inconsistent")
    if error == "unreadable":
        if file_digest is not None:
            raise ValueError(f"unreadable {prefix} unexpectedly has a digest")
    else:
        sha256(file_digest, name=f"{prefix}_file_sha256")


def validate_deform360_calibration_source_run_record(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate all fields and recompute every derived terminal decision."""

    if not isinstance(record, Mapping):
        raise ValueError("execution record must be an object")
    if set(record) != _RECORD_KEYS:
        raise ValueError("execution record fields changed")
    if record.get("schema") != DEFORM360_CALIBRATION_SOURCE_RUN_SCHEMA:
        raise ValueError("execution record schema changed")
    if (
        type(record.get("schema_version")) is not int
        or record["schema_version"] != DEFORM360_CALIBRATION_SOURCE_RUN_VERSION
    ):
        raise ValueError("execution record schema version changed")
    if record.get("semantics") != DEFORM360_CALIBRATION_SOURCE_RUN_SEMANTICS:
        raise ValueError("execution record semantics changed")
    if record.get("dataset_revision") != DEFORM360_DATASET_REVISION:
        raise ValueError("execution record dataset revision changed")
    if record.get("claim_boundary") != (
        DEFORM360_CALIBRATION_SOURCE_RUN_CLAIM_BOUNDARY
    ):
        raise ValueError("execution record claim boundary changed")

    revision(record["source_revision"], name="source_revision")
    revision(record["processing_revision"], name="processing_revision")
    positive_integer(record["workflow_run_id"], name="workflow_run_id")
    positive_integer(
        record["workflow_run_attempt"],
        name="workflow_run_attempt",
    )
    workload_exit_code = exit_code(
        record["workload_exit_code"],
        name="workload_exit_code",
    )
    boundary_exit_code = exit_code(
        record["confirmation_boundary_exit_code"],
        name="confirmation_boundary_exit_code",
    )

    verified = record.get("confirmation_boundary_verified")
    if type(verified) is not bool or verified is not (boundary_exit_code == 0):
        raise ValueError("confirmation-boundary decision is inconsistent")
    expected_opened = False if verified else None
    if record.get("confirmation_payloads_opened") is not expected_opened:
        raise ValueError("confirmation-payload statement is inconsistent")

    _validate_source_lock_summary(record)
    _validate_artifact_summary(record, prefix="plan", gate_key="plan_support_gate")
    _validate_artifact_summary(record, prefix="download", gate_key=None)
    _validate_artifact_summary(record, prefix="result", gate_key="support_gate")

    expected_exit_code, expected_failure_stage = _execution_outcome(
        workload_exit_code=workload_exit_code,
        confirmation_boundary_exit_code=boundary_exit_code,
        source_locks=record,
        plan=record,
        download=record,
        result=record,
    )
    observed_exit_code = exit_code(record["exit_code"], name="exit_code")
    if observed_exit_code != expected_exit_code:
        raise ValueError("execution record exit code is inconsistent")
    if record.get("failure_stage") != expected_failure_stage:
        raise ValueError("execution record failure stage is inconsistent")
    expected_status = "succeeded" if expected_exit_code == 0 else "failed"
    if record.get("status") != expected_status:
        raise ValueError("execution record status is inconsistent")

    record_digest = sha256(record.get("record_sha256"), name="record_sha256")
    if record_digest != canonical_sha256(record):
        raise ValueError("record_sha256 does not match record content")
    return dict(record)


def load_deform360_calibration_source_run_record(path: Path) -> dict[str, Any]:
    """Load strict JSON and validate one terminal record from disk."""

    record, _ = load_json_object(path)
    return validate_deform360_calibration_source_run_record(record)
