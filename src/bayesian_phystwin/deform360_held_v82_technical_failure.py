"""Integrity closure for the consumed held-v8.2 technical failure."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any


PROTOCOL_ID = "deform360-held-online-belief-v8.2"
EXECUTION_ATTEMPT = 1
REPORT_KIND = "Deform360HeldV82TechnicalFailureReport"
COMPLETION_KIND = "Deform360HeldV82TechnicalFailureIntegrityCompletion"
POINTER_KIND = "Deform360HeldV82TechnicalFailurePointer"
STATUS = "technical-failure-before-first-target-or-query-artifact"
RESULT_STATUS = "NO_CALIBRATION_RESULT"
REPORT_NAME = "execution-technical-failure-attempt1.json"
_SHA256_LENGTH = 64


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def artifact_sha256(value: Mapping[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("artifact_sha256", None)
    return hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()


def signed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["artifact_sha256"] = artifact_sha256(result)
    return result


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _stable_state(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def stable_file(
    path: str | Path,
    *,
    collect: bool = False,
    required_mode: int | None = None,
) -> dict[str, Any]:
    source = _absolute(path)
    before = os.lstat(source)
    _require(
        stat.S_ISREG(before.st_mode)
        and not stat.S_ISLNK(before.st_mode)
        and before.st_nlink == 1
        and source.resolve(strict=True) == source,
        f"regular single-link file required: {source}",
    )
    if required_mode is not None:
        _require(
            stat.S_IMODE(before.st_mode) == required_mode,
            f"file mode changed: {source}",
        )
    descriptor = os.open(
        source,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    digest = hashlib.sha256()
    payload = bytearray() if collect else None
    try:
        opened = os.fstat(descriptor)
        _require(
            _stable_state(opened) == _stable_state(before),
            f"file changed while opening: {source}",
        )
        while block := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(block)
            if payload is not None:
                payload.extend(block)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = os.lstat(source)
    _require(
        _stable_state(before)
        == _stable_state(opened_after)
        == _stable_state(after),
        f"file changed while hashing: {source}",
    )
    return {
        "path": os.fspath(source),
        "sha256": digest.hexdigest(),
        "size_bytes": before.st_size,
        "mode_octal": f"{stat.S_IMODE(before.st_mode):04o}",
        **({"payload": bytes(payload)} if payload is not None else {}),
    }


def load_signed(path: str | Path, *, role: str) -> tuple[dict[str, Any], dict[str, Any]]:
    observed = stable_file(path, collect=True, required_mode=0o400)
    payload = observed.pop("payload")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{role} is not UTF-8 JSON") from error
    _require(isinstance(value, dict), f"{role} is not a JSON object")
    _require(
        value.get("artifact_sha256") == artifact_sha256(value),
        f"{role} signature changed",
    )
    return value, observed


def archive_inventory(
    root: str | Path,
    *,
    excluded_code_directory: str,
    exclude_report: bool = False,
) -> dict[str, Any]:
    source = _absolute(root)
    root_state = os.lstat(source)
    _require(
        stat.S_ISDIR(root_state.st_mode)
        and not stat.S_ISLNK(root_state.st_mode)
        and source.resolve(strict=True) == source,
        "archive root is absent, linked, or non-canonical",
    )
    rows: list[dict[str, Any]] = []
    for current, directories, files in os.walk(
        source, topdown=True, followlinks=False
    ):
        parent = Path(current)
        relative_parent = parent.relative_to(source)
        directories[:] = sorted(
            name
            for name in directories
            if not (
                relative_parent == Path(".")
                and name == excluded_code_directory
            )
        )
        for name in directories:
            path = parent / name
            observed = os.lstat(path)
            _require(
                stat.S_ISDIR(observed.st_mode) and not stat.S_ISLNK(observed.st_mode),
                f"directory is linked or special: {path}",
            )
            rows.append(
                {
                    "path": path.relative_to(source).as_posix(),
                    "type": "directory",
                    "mode_octal": f"{stat.S_IMODE(observed.st_mode):04o}",
                }
            )
        for name in sorted(files):
            path = parent / name
            relative = path.relative_to(source)
            if exclude_report and relative == Path(REPORT_NAME):
                continue
            observed = stable_file(path)
            rows.append(
                {
                    "path": relative.as_posix(),
                    "type": "file",
                    "sha256": observed["sha256"],
                    "size_bytes": observed["size_bytes"],
                    "mode_octal": observed["mode_octal"],
                }
            )
    rows.sort(key=lambda row: str(row["path"]))
    _require(
        len(rows) == len({str(row["path"]) for row in rows}),
        "archive inventory has duplicate paths",
    )
    return {
        "entry_count": len(rows),
        "directory_count": sum(row["type"] == "directory" for row in rows),
        "regular_file_count": sum(row["type"] == "file" for row in rows),
        "regular_file_bytes": sum(
            int(row.get("size_bytes", 0)) for row in rows if row["type"] == "file"
        ),
        "inventory_sha256": hashlib.sha256(
            _canonical_bytes({"rows": rows})
        ).hexdigest(),
        "excluded_deployed_code_directory": excluded_code_directory,
        "excluded_report": REPORT_NAME if exclude_report else None,
    }


def _record_matches(
    declared: object,
    observed: Mapping[str, Any],
    *,
    role: str,
    artifact: str | None = None,
) -> None:
    _require(isinstance(declared, Mapping), f"{role} binding is absent")
    for key in ("path", "sha256", "size_bytes"):
        _require(declared.get(key) == observed.get(key), f"{role} binding changed")
    if artifact is not None:
        _require(
            declared.get("artifact_sha256") == artifact,
            f"{role} artifact binding changed",
        )


def validate_v82_technical_failure_lineage(
    *,
    archive_path: str | Path,
    report_path: str | Path,
    pointer_path: str | Path,
    completion_path: str | Path,
    verify_content_inventory: bool = False,
) -> dict[str, Any]:
    archive = _absolute(archive_path)
    report, report_record = load_signed(report_path, role="v8.2 failure report")
    completion, completion_record = load_signed(
        completion_path,
        role="v8.2 failure completion",
    )
    pointer, pointer_record = load_signed(pointer_path, role="v8.2 failure pointer")

    _require(
        report.get("artifact_kind") == REPORT_KIND
        and report.get("protocol_id") == PROTOCOL_ID
        and report.get("execution_attempt") == EXECUTION_ATTEMPT
        and report.get("status") == STATUS
        and report.get("result_status") == RESULT_STATUS
        and report.get("immutable_archive_path") == os.fspath(archive),
        "v8.2 failure report identity changed",
    )
    information = report.get("information_boundary")
    _require(
        information
        == {
            "first_complete_cohort_barrier_crossed": True,
            "official_target_artifact_created_or_read": False,
            "official_query_artifact_created_or_read": False,
            "queried_prediction_created_or_read": False,
            "score_created_or_read": False,
            "gate_decision_created_or_read": False,
            "confirmation_created_or_read": False,
            "failure_evidence_is_source_side_runtime_only": True,
            "forensic_operator_deserialized_protected_outcome_payload": False,
        },
        "v8.2 failure information boundary changed",
    )
    code = report.get("deployed_code")
    _require(
        isinstance(code, Mapping)
        and isinstance(code.get("path"), str)
        and code.get("path")
        and isinstance(code.get("git_head"), str),
        "v8.2 deployed-code binding is absent",
    )

    _require(
        completion.get("artifact_kind") == COMPLETION_KIND
        and completion.get("protocol_id") == PROTOCOL_ID
        and completion.get("execution_attempt") == EXECUTION_ATTEMPT
        and completion.get("status") == STATUS
        and completion.get("result_status") == RESULT_STATUS
        and completion.get("immutable_archive_path") == os.fspath(archive),
        "v8.2 failure completion identity changed",
    )
    _record_matches(
        completion.get("report"),
        report_record,
        role="v8.2 failure report",
        artifact=report["artifact_sha256"],
    )
    inventory = completion.get("archive_inventory")
    _require(
        isinstance(inventory, Mapping)
        and _valid_sha256(inventory.get("inventory_sha256"))
        and inventory.get("excluded_deployed_code_directory") == code["path"]
        and inventory.get("excluded_report") is None,
        "v8.2 archive inventory binding changed",
    )
    root_state = os.lstat(archive)
    _require(
        stat.S_ISDIR(root_state.st_mode)
        and not stat.S_ISLNK(root_state.st_mode)
        and stat.S_IMODE(root_state.st_mode) == 0o500,
        "v8.2 archive root is not sealed",
    )
    _require(
        _absolute(report_path) == archive / REPORT_NAME,
        "v8.2 report is outside its archive",
    )
    if verify_content_inventory:
        observed_inventory = archive_inventory(
            archive,
            excluded_code_directory=str(code["path"]),
        )
        _require(
            dict(inventory) == observed_inventory,
            "v8.2 archive content inventory changed",
        )

    _require(
        pointer.get("artifact_kind") == POINTER_KIND
        and pointer.get("protocol_id") == PROTOCOL_ID
        and pointer.get("execution_attempt") == EXECUTION_ATTEMPT
        and pointer.get("status") == STATUS
        and pointer.get("result_status") == RESULT_STATUS
        and pointer.get("immutable_archive_path") == os.fspath(archive),
        "v8.2 failure pointer identity changed",
    )
    _record_matches(
        pointer.get("report"),
        report_record,
        role="v8.2 failure pointer report",
        artifact=report["artifact_sha256"],
    )
    _record_matches(
        pointer.get("completion"),
        completion_record,
        role="v8.2 failure pointer completion",
        artifact=completion["artifact_sha256"],
    )
    _require(
        pointer.get("archive_inventory_sha256") == inventory["inventory_sha256"],
        "v8.2 failure pointer archive binding changed",
    )
    return {
        "v82_technical_failure_report": {
            **report_record,
            "artifact_sha256": report["artifact_sha256"],
        },
        "v82_technical_failure_pointer": {
            **pointer_record,
            "artifact_sha256": pointer["artifact_sha256"],
        },
        "v82_technical_failure_integrity_completion": {
            **completion_record,
            "artifact_sha256": completion["artifact_sha256"],
        },
        "v82_technical_failure_archive_integrity": {
            "path": os.fspath(archive),
            **dict(inventory),
        },
        "v82_calibration_result": RESULT_STATUS,
    }


__all__ = [
    "COMPLETION_KIND",
    "EXECUTION_ATTEMPT",
    "POINTER_KIND",
    "PROTOCOL_ID",
    "REPORT_KIND",
    "REPORT_NAME",
    "RESULT_STATUS",
    "STATUS",
    "archive_inventory",
    "artifact_sha256",
    "load_signed",
    "signed",
    "stable_file",
    "validate_v82_technical_failure_lineage",
]
