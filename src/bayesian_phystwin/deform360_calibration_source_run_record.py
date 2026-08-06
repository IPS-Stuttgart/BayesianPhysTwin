"""Non-sensitive completion records for direct Deform360 calibration runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

DEFORM360_CALIBRATION_SOURCE_RUN_SCHEMA: Final = (
    "bayesian-phystwin.deform360-calibration-source-run"
)
DEFORM360_CALIBRATION_SOURCE_RUN_VERSION: Final = 1
DEFORM360_CALIBRATION_SOURCE_RUN_SEMANTICS: Final = (
    "non-sensitive-direct-calibration-source-completion-v1"
)
DEFORM360_CALIBRATION_SOURCE_RESULT_SCHEMA: Final = (
    "bayesian-phystwin/deform360-calibration-source-result-v1"
)
DEFORM360_CALIBRATION_SOURCE_PROTOCOL_ID: Final = (
    "deform360-official-hub-calibration-source-v1"
)
DEFORM360_DATASET_REVISION: Final = "f804696d7a133908c7497ffdab43819d879b5cbc"
DEFORM360_CALIBRATION_SOURCE_RUN_CLAIM_BOUNDARY: Final = (
    "Execution and information-boundary evidence only. This record does not "
    "establish observation-provider competence, physical-query benefit, "
    "calibration, independent-object transfer, deployment safety, or state of "
    "the art."
)
_RESULT_CONTRACT_EXIT_CODE: Final = 4
_SUPPORT_GATE_EXIT_CODE: Final = 3
_RECORD_WRITE_EXIT_CODE: Final = 70
_EXPECTED_OBJECT_COUNT: Final = 10
_EXPECTED_OBJECTS_PER_STRATUM: Final = 5
_MINIMUM_SUPPORTED_OBJECTS: Final = 8
_MINIMUM_SUPPORTED_PER_STRATUM: Final = 4
_EXPECTED_STRATA: Final = ("sheet", "volumetric")
_ALLOWED_OBJECT_STATUSES: Final = frozenset(
    {
        "source_prepared",
        "technical_failure_without_replacement",
        "unsupported_without_replacement",
    }
)
_EXPECTED_INFORMATION_BOUNDARY: Final = {
    "calibration_camera_payloads_opened": True,
    "calibration_tactile_payloads_opened": True,
    "calibration_robot_state_derived": True,
    "calibration_target_metrics_computed": False,
    "confirmation_payloads_opened": False,
    "target_outcomes_used": False,
    "replacement_allowed": False,
}
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_strict_pairs,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(
    value: Mapping[str, Any],
    *,
    digest_key: str = "record_sha256",
) -> str:
    payload = dict(value)
    payload.pop(digest_key, None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _exit_code(value: int, *, name: str) -> int:
    if type(value) is not int or not 0 <= value <= 255:
        raise ValueError(f"{name} must be an integer in [0, 255]")
    return value


def _positive_integer(value: int, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _revision(value: str, *, name: str) -> str:
    if type(value) is not str or _REVISION_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 40-character revision")
    return value


def _sha256(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _integer_field(value: object, *, name: str, maximum: int | None = None) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} exceeds {maximum}")
    return value


def _invalid_result(
    *,
    available: bool,
    error: str,
    file_sha256: str | None,
) -> dict[str, Any]:
    return {
        "result_available": available,
        "result_valid": False,
        "result_error": error,
        "result_file_sha256": file_sha256,
        "result_sha256": None,
        "support_gate": None,
    }


def _object_support_counts(value: Mapping[str, Any]) -> tuple[int, dict[str, int]]:
    objects = value.get("objects")
    if not isinstance(objects, list) or len(objects) != _EXPECTED_OBJECT_COUNT:
        raise ValueError("result must contain exactly ten object rows")
    seen: set[str] = set()
    cohort_by_stratum = dict.fromkeys(_EXPECTED_STRATA, 0)
    supported_by_stratum = dict.fromkeys(_EXPECTED_STRATA, 0)
    for row in objects:
        if not isinstance(row, Mapping):
            raise ValueError("result object row is malformed")
        object_id = row.get("object_id")
        if type(object_id) is not str or not object_id or object_id in seen:
            raise ValueError("result object identities are missing or duplicated")
        seen.add(object_id)
        episode_id = row.get("episode_id")
        if type(episode_id) is not int or episode_id < 0:
            raise ValueError("result episode identity is invalid")
        stratum = row.get("stratum")
        if stratum not in _EXPECTED_STRATA:
            raise ValueError("result object stratum changed")
        status = row.get("status")
        if status not in _ALLOWED_OBJECT_STATUSES:
            raise ValueError("result object status changed")
        cohort_by_stratum[stratum] += 1
        if status == "source_prepared":
            supported_by_stratum[stratum] += 1
    if any(
        count != _EXPECTED_OBJECTS_PER_STRATUM
        for count in cohort_by_stratum.values()
    ):
        raise ValueError("result cohort strata changed")
    return sum(supported_by_stratum.values()), supported_by_stratum


def _validated_support_gate(
    value: Mapping[str, Any],
    *,
    object_supported: int,
    object_supported_by_stratum: Mapping[str, int],
) -> dict[str, Any]:
    gate = value.get("gate")
    if not isinstance(gate, Mapping):
        raise ValueError("result gate is missing")
    expected_keys = {
        "supported_object_count",
        "supported_by_stratum",
        "minimum_supported_objects",
        "minimum_supported_per_stratum",
        "support_passed",
    }
    if set(gate) != expected_keys:
        raise ValueError("result gate fields changed")
    supported = _integer_field(
        gate.get("supported_object_count"),
        name="supported_object_count",
        maximum=_EXPECTED_OBJECT_COUNT,
    )
    by_stratum = gate.get("supported_by_stratum")
    if not isinstance(by_stratum, Mapping) or set(by_stratum) != set(_EXPECTED_STRATA):
        raise ValueError("supported_by_stratum changed")
    sheet = _integer_field(
        by_stratum.get("sheet"),
        name="supported_by_stratum.sheet",
        maximum=_EXPECTED_OBJECTS_PER_STRATUM,
    )
    volumetric = _integer_field(
        by_stratum.get("volumetric"),
        name="supported_by_stratum.volumetric",
        maximum=_EXPECTED_OBJECTS_PER_STRATUM,
    )
    observed_by_stratum = {"sheet": sheet, "volumetric": volumetric}
    if supported != sheet + volumetric:
        raise ValueError("supported object counts disagree")
    if supported != object_supported or observed_by_stratum != dict(
        object_supported_by_stratum
    ):
        raise ValueError("support gate disagrees with object rows")
    minimum = _integer_field(
        gate.get("minimum_supported_objects"),
        name="minimum_supported_objects",
        maximum=_EXPECTED_OBJECT_COUNT,
    )
    minimum_per_stratum = _integer_field(
        gate.get("minimum_supported_per_stratum"),
        name="minimum_supported_per_stratum",
        maximum=_EXPECTED_OBJECTS_PER_STRATUM,
    )
    if minimum != _MINIMUM_SUPPORTED_OBJECTS:
        raise ValueError("minimum supported object count changed")
    if minimum_per_stratum != _MINIMUM_SUPPORTED_PER_STRATUM:
        raise ValueError("minimum supported per stratum changed")
    support_passed = gate.get("support_passed")
    if type(support_passed) is not bool:
        raise ValueError("support_passed must be a boolean")
    expected_pass = (
        supported >= _MINIMUM_SUPPORTED_OBJECTS
        and min(sheet, volumetric) >= _MINIMUM_SUPPORTED_PER_STRATUM
    )
    if support_passed is not expected_pass:
        raise ValueError("support gate decision disagrees with its counts")
    return {
        "supported_object_count": supported,
        "supported_by_stratum": observed_by_stratum,
        "minimum_supported_objects": minimum,
        "minimum_supported_per_stratum": minimum_per_stratum,
        "support_passed": support_passed,
    }


def _result_summary(
    path: Path,
    *,
    processing_revision: str,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        return _invalid_result(available=False, error="missing", file_sha256=None)
    result_file_sha256 = _file_sha256(path)
    try:
        value = _load_json_object(path)
    except (OSError, ValueError):
        return _invalid_result(
            available=True,
            error="invalid-json",
            file_sha256=result_file_sha256,
        )
    try:
        if value.get("schema") != DEFORM360_CALIBRATION_SOURCE_RESULT_SCHEMA:
            raise ValueError("result schema changed")
        if value.get("schema_version") != 1:
            raise ValueError("result schema version changed")
        if value.get("protocol_id") != DEFORM360_CALIBRATION_SOURCE_PROTOCOL_ID:
            raise ValueError("result protocol changed")
        if value.get("dataset_revision") != DEFORM360_DATASET_REVISION:
            raise ValueError("result dataset revision changed")
        if value.get("processing_revision") != processing_revision:
            raise ValueError("result processing revision changed")
        _sha256(value.get("plan_sha256"), name="plan_sha256")
        _sha256(value.get("download_sha256"), name="download_sha256")
        result_sha256 = _sha256(
            value.get("result_sha256"),
            name="result_sha256",
        )
        if result_sha256 != _canonical_sha256(
            value,
            digest_key="result_sha256",
        ):
            raise ValueError("result digest does not match its content")
        information_boundary = value.get("information_boundary")
        if not isinstance(information_boundary, Mapping) or dict(
            information_boundary
        ) != dict(_EXPECTED_INFORMATION_BOUNDARY):
            raise ValueError("result information boundary changed")
        object_supported, object_supported_by_stratum = _object_support_counts(value)
        support_gate = _validated_support_gate(
            value,
            object_supported=object_supported,
            object_supported_by_stratum=object_supported_by_stratum,
        )
    except ValueError:
        return _invalid_result(
            available=True,
            error="invalid-contract",
            file_sha256=result_file_sha256,
        )
    return {
        "result_available": True,
        "result_valid": True,
        "result_error": None,
        "result_file_sha256": result_file_sha256,
        "result_sha256": result_sha256,
        "support_gate": support_gate,
    }


def _execution_outcome(
    *,
    workload_exit_code: int,
    confirmation_boundary_exit_code: int,
    result: Mapping[str, Any],
) -> tuple[int, str | None]:
    if confirmation_boundary_exit_code != 0:
        return confirmation_boundary_exit_code, "confirmation-boundary"
    if result["result_available"] and not result["result_valid"]:
        return _RESULT_CONTRACT_EXIT_CODE, "result-contract"
    if result["result_valid"]:
        support_gate = result["support_gate"]
        if not isinstance(support_gate, Mapping):
            return _RESULT_CONTRACT_EXIT_CODE, "result-contract"
        support_passed = support_gate.get("support_passed")
        expected_workload = 0 if support_passed is True else _SUPPORT_GATE_EXIT_CODE
        if workload_exit_code != expected_workload:
            return _RESULT_CONTRACT_EXIT_CODE, "result-contract"
        if support_passed is False:
            return _SUPPORT_GATE_EXIT_CODE, "calibration-source-support-gate"
        return 0, None
    if workload_exit_code != 0:
        return workload_exit_code, "calibration-source-workload"
    return _RESULT_CONTRACT_EXIT_CODE, "result-contract"


def build_deform360_calibration_source_run_record(
    *,
    source_revision: str,
    processing_revision: str,
    workflow_run_id: int,
    workflow_run_attempt: int,
    workload_exit_code: int,
    confirmation_boundary_exit_code: int,
    result_json: Path,
) -> dict[str, Any]:
    """Build one portable record without paths, object IDs, or target outcomes."""

    source_revision = _revision(source_revision, name="source_revision")
    processing_revision = _revision(
        processing_revision,
        name="processing_revision",
    )
    workflow_run_id = _positive_integer(workflow_run_id, name="workflow_run_id")
    workflow_run_attempt = _positive_integer(
        workflow_run_attempt,
        name="workflow_run_attempt",
    )
    workload_exit_code = _exit_code(
        workload_exit_code,
        name="workload_exit_code",
    )
    confirmation_boundary_exit_code = _exit_code(
        confirmation_boundary_exit_code,
        name="confirmation_boundary_exit_code",
    )
    result = _result_summary(
        result_json,
        processing_revision=processing_revision,
    )
    confirmation_boundary_verified = confirmation_boundary_exit_code == 0
    effective_exit_code, failure_stage = _execution_outcome(
        workload_exit_code=workload_exit_code,
        confirmation_boundary_exit_code=confirmation_boundary_exit_code,
        result=result,
    )
    record: dict[str, Any] = {
        "schema": DEFORM360_CALIBRATION_SOURCE_RUN_SCHEMA,
        "schema_version": DEFORM360_CALIBRATION_SOURCE_RUN_VERSION,
        "semantics": DEFORM360_CALIBRATION_SOURCE_RUN_SEMANTICS,
        "record_sha256": None,
        "status": "succeeded" if effective_exit_code == 0 else "failed",
        "exit_code": effective_exit_code,
        "workload_exit_code": workload_exit_code,
        "failure_stage": failure_stage,
        "confirmation_boundary_exit_code": confirmation_boundary_exit_code,
        "confirmation_boundary_verified": confirmation_boundary_verified,
        "confirmation_payloads_opened": (
            False if confirmation_boundary_verified else None
        ),
        "source_revision": source_revision,
        "dataset_revision": DEFORM360_DATASET_REVISION,
        "processing_revision": processing_revision,
        "workflow_run_id": workflow_run_id,
        "workflow_run_attempt": workflow_run_attempt,
        **result,
        "claim_boundary": DEFORM360_CALIBRATION_SOURCE_RUN_CLAIM_BOUNDARY,
    }
    record["record_sha256"] = _canonical_sha256(record)
    return record


def save_deform360_calibration_source_run_record(
    record: Mapping[str, Any],
    path: Path,
) -> None:
    """Publish a run record atomically without replacing an earlier record."""

    expected = record.get("record_sha256")
    if type(expected) is not str or _SHA256_RE.fullmatch(expected) is None:
        raise ValueError("record_sha256 is invalid")
    if expected != _canonical_sha256(record):
        raise ValueError("record_sha256 does not match record content")
    path = path.absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.tmp-",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--processing-revision", required=True)
    parser.add_argument("--workflow-run-id", type=int, required=True)
    parser.add_argument("--workflow-run-attempt", type=int, required=True)
    parser.add_argument("--workload-exit-code", type=int, required=True)
    parser.add_argument(
        "--confirmation-boundary-exit-code",
        type=int,
        required=True,
    )
    parser.add_argument("--result-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        record = build_deform360_calibration_source_run_record(
            source_revision=args.source_revision,
            processing_revision=args.processing_revision,
            workflow_run_id=args.workflow_run_id,
            workflow_run_attempt=args.workflow_run_attempt,
            workload_exit_code=args.workload_exit_code,
            confirmation_boundary_exit_code=args.confirmation_boundary_exit_code,
            result_json=args.result_json,
        )
        save_deform360_calibration_source_run_record(record, args.output)
    except (OSError, TypeError, ValueError) as error:
        print(f"cannot publish calibration-source run record: {error}", file=sys.stderr)
        return _RECORD_WRITE_EXIT_CODE
    print(json.dumps(record, indent=2, sort_keys=True, allow_nan=False))
    return int(record["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
