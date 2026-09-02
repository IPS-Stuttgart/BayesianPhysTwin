#!/usr/bin/env python3
"""Inventory the remote PoseIt ZIP only after its full-byte receipt is sealed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bayesian_phystwin._portable_contracts import (
    content_id,
    load_strict_json_object,
)
from bayesian_phystwin_experiments.poseit_real_decision_protocol import (
    load_poseit_preaccess_mapping_constraints,
    load_poseit_range_transport_lock,
    load_poseit_real_decision_method_lock,
    load_poseit_real_decision_protocol,
    poseit_mapping_constraints_config_sha256,
    poseit_mapping_constraints_file_sha256,
    poseit_method_lock_config_sha256,
    poseit_method_lock_file_sha256,
    poseit_protocol_config_sha256,
    poseit_protocol_file_sha256,
    poseit_range_transport_lock_config_sha256,
    poseit_range_transport_lock_file_sha256,
)
from bayesian_phystwin_experiments.poseit_remote_archive import (
    RangeOpener,
    RemoteArchiveExpectation,
)
from bayesian_phystwin_experiments.poseit_remote_zip import (
    RemoteZipRangeEvent,
    fetch_remote_central_directory,
    parse_remote_central_directory,
    read_remote_zip_layout,
    remote_zip_structure_summary,
)

SCHEMA = "bayesian-phystwin.poseit-remote-archive-structure-lock"
PRIVATE_MANIFEST_SCHEMA = "bayesian-phystwin.poseit-remote-private-member-manifest"
SCHEMA_VERSION = 1
EXECUTION_LOCK_ID = "poseit-real-decision-remote-structure-transport-lock-v1"
EXECUTION_LOCK_STATUS = (
    "frozen-after-full-range-hash-and-before-central-directory-access"
)
ACQUISITION_RECEIPT_SCHEMA = "bayesian-phystwin.poseit-range-hash-acquisition-receipt"

_ACQUISITION_RECEIPT_FIELDS = frozenset(
    {
        "archive_bytes_retained",
        "archive_bytes_streamed_opaquely",
        "archive_member_names_opened",
        "archive_sha256",
        "archive_size_bytes",
        "chunk_count",
        "chunk_size_bytes",
        "confirmation_opened",
        "held_v8_accessed",
        "mapping_constraints_config_sha256",
        "mapping_constraints_file_sha256",
        "max_attempts_per_range",
        "max_workers",
        "member_payload_bytes_opened",
        "method_lock_config_sha256",
        "method_lock_file_sha256",
        "object_roles_assigned",
        "phase_labels_opened",
        "protocol_config_sha256",
        "protocol_file_sha256",
        "range_transport_core_file_sha256",
        "receipt_id",
        "schema",
        "schema_version",
        "sensor_payloads_opened",
        "shake_outcomes_opened",
        "source",
        "source_file_id",
        "source_file_name",
        "source_last_modified",
        "source_url",
        "transport_attempts",
        "transport_lock_config_sha256",
        "transport_lock_file_sha256",
        "zip_central_directory_parsed",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: object, *, message: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), message)
    assert isinstance(value, Mapping)
    return value


def _exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    name: str,
) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    _require(not missing and not extra, f"{name} fields changed: {missing=}, {extra=}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _expectation(transport: Mapping[str, Any]) -> RemoteArchiveExpectation:
    source = _mapping(
        transport.get("source_identity"), message="range source identity is missing"
    )
    execution = _mapping(
        transport.get("execution"), message="range execution lock is missing"
    )
    return RemoteArchiveExpectation(
        source_url=str(source["source_url"]),
        file_name=str(source["content_disposition_file_name"]),
        size_bytes=int(source["archive_size_bytes"]),
        last_modified=str(source["last_modified"]),
        chunk_size_bytes=int(execution["chunk_size_bytes"]),
        max_workers=int(execution["max_workers"]),
        max_attempts_per_range=int(execution["max_attempts_per_range"]),
        timeout_seconds=float(execution["timeout_seconds_per_range"]),
    )


def _validate_acquisition_receipt(
    receipt: Mapping[str, Any],
    *,
    expectation: RemoteArchiveExpectation,
    source_file_id: str,
    protocol_file_sha256: str,
    protocol_config_sha256: str,
    mapping_file_sha256: str,
    mapping_config_sha256: str,
    method_file_sha256: str,
    method_config_sha256: str,
    transport_file_sha256: str,
    transport_config_sha256: str,
    range_core_file_sha256: str,
) -> dict[str, Any]:
    _exact_fields(
        receipt,
        _ACQUISITION_RECEIPT_FIELDS,
        name="acquisition receipt",
    )
    _require(
        receipt.get("schema") == ACQUISITION_RECEIPT_SCHEMA, "receipt schema changed"
    )
    _require(receipt.get("schema_version") == 1, "receipt schema version changed")
    _require(
        receipt.get("source") == "official-public-google-drive-file-exact-range-hash",
        "receipt source changed",
    )
    _require(receipt.get("source_file_id") == source_file_id, "source file ID changed")
    _require(
        receipt.get("source_file_name") == expectation.file_name, "file name changed"
    )
    _require(receipt.get("source_url") == expectation.source_url, "source URL changed")
    _require(
        receipt.get("source_last_modified") == expectation.last_modified,
        "source last-modified changed",
    )
    _require(
        receipt.get("archive_size_bytes") == expectation.size_bytes,
        "receipt archive size changed",
    )
    _require(_is_sha256(receipt.get("archive_sha256")), "archive SHA-256 is malformed")
    _require(
        receipt.get("chunk_size_bytes") == expectation.chunk_size_bytes,
        "receipt chunk size changed",
    )
    expected_chunks = (
        expectation.size_bytes + expectation.chunk_size_bytes - 1
    ) // expectation.chunk_size_bytes
    _require(
        receipt.get("chunk_count") == expected_chunks, "receipt chunk count changed"
    )
    _require(
        receipt.get("max_workers") == expectation.max_workers, "worker count changed"
    )
    _require(
        receipt.get("max_attempts_per_range") == expectation.max_attempts_per_range,
        "range attempt bound changed",
    )
    attempts = receipt.get("transport_attempts")
    _require(type(attempts) is int, "transport attempt count is malformed")
    assert isinstance(attempts, int)
    _require(
        expected_chunks
        <= attempts
        <= expected_chunks * expectation.max_attempts_per_range,
        "transport attempt count is outside its registered bound",
    )
    expected_digests = {
        "protocol_file_sha256": protocol_file_sha256,
        "protocol_config_sha256": protocol_config_sha256,
        "mapping_constraints_file_sha256": mapping_file_sha256,
        "mapping_constraints_config_sha256": mapping_config_sha256,
        "method_lock_file_sha256": method_file_sha256,
        "method_lock_config_sha256": method_config_sha256,
        "transport_lock_file_sha256": transport_file_sha256,
        "transport_lock_config_sha256": transport_config_sha256,
        "range_transport_core_file_sha256": range_core_file_sha256,
    }
    for key, expected in expected_digests.items():
        _require(receipt.get(key) == expected, f"receipt parent changed: {key}")
    _require(
        receipt.get("archive_bytes_streamed_opaquely") is True, "hash was not opaque"
    )
    for key in (
        "archive_bytes_retained",
        "zip_central_directory_parsed",
        "archive_member_names_opened",
        "member_payload_bytes_opened",
        "phase_labels_opened",
        "sensor_payloads_opened",
        "shake_outcomes_opened",
        "object_roles_assigned",
        "confirmation_opened",
        "held_v8_accessed",
    ):
        _require(receipt.get(key) is False, f"receipt boundary changed: {key}")
    identity = dict(receipt)
    receipt_id = identity.pop("receipt_id")
    _require(receipt_id == content_id(identity), "acquisition receipt ID changed")
    return dict(receipt)


def _validate_execution_lock(
    lock: Mapping[str, Any],
    *,
    expectation: RemoteArchiveExpectation,
    output_root: Path,
    acquisition_receipt: Mapping[str, Any],
    acquisition_receipt_file_sha256: str,
    parent_digests: Mapping[str, Mapping[str, str]],
    range_core_file_sha256: str,
    remote_zip_core_file_sha256: str,
    current_host: str,
) -> dict[str, Any]:
    _exact_fields(
        lock,
        frozenset(
            {
                "boundaries",
                "contract",
                "execution",
                "lock_id",
                "outputs",
                "parent_artifacts",
                "schema_version",
                "source_identity",
                "status",
            }
        ),
        name="structure execution lock",
    )
    _require(lock.get("contract") == EXECUTION_LOCK_ID, "structure lock ID changed")
    _require(lock.get("schema_version") == 1, "structure lock schema changed")
    _require(
        lock.get("status") == EXECUTION_LOCK_STATUS, "structure lock status changed"
    )
    source = _mapping(
        lock.get("source_identity"), message="structure source is missing"
    )
    _exact_fields(
        source,
        frozenset({"archive_size_bytes", "file_name", "last_modified", "source_url"}),
        name="structure source identity",
    )
    _require(
        source.get("source_url") == expectation.source_url, "structure URL changed"
    )
    _require(
        source.get("archive_size_bytes") == expectation.size_bytes,
        "structure size changed",
    )
    _require(
        source.get("file_name") == expectation.file_name, "structure file name changed"
    )
    _require(
        source.get("last_modified") == expectation.last_modified,
        "structure last-modified changed",
    )

    parents = _mapping(
        lock.get("parent_artifacts"), message="structure parents are missing"
    )
    _exact_fields(
        parents,
        frozenset(
            {
                "acquisition_receipt",
                "mapping_constraints",
                "method_lock",
                "protocol",
                "range_transport_core",
                "range_transport_lock",
                "remote_zip_core",
            }
        ),
        name="structure parents",
    )
    for name in (
        "protocol",
        "mapping_constraints",
        "method_lock",
        "range_transport_lock",
    ):
        parent = _mapping(
            parents.get(name), message=f"structure parent is missing: {name}"
        )
        _exact_fields(
            parent,
            frozenset({"config_sha256", "file_sha256"}),
            name=f"structure parent {name}",
        )
        expected = parent_digests[name]
        _require(
            parent.get("file_sha256") == expected["file_sha256"], f"{name} file changed"
        )
        _require(
            parent.get("config_sha256") == expected["config_sha256"],
            f"{name} config changed",
        )
    acquisition_parent = _mapping(
        parents.get("acquisition_receipt"),
        message="acquisition receipt parent is missing",
    )
    _exact_fields(
        acquisition_parent,
        frozenset({"archive_sha256", "file_sha256", "receipt_id"}),
        name="acquisition receipt parent",
    )
    _require(
        acquisition_parent.get("file_sha256") == acquisition_receipt_file_sha256,
        "acquisition receipt file changed",
    )
    _require(
        acquisition_parent.get("receipt_id") == acquisition_receipt.get("receipt_id"),
        "acquisition receipt identity changed",
    )
    _require(
        acquisition_parent.get("archive_sha256")
        == acquisition_receipt.get("archive_sha256"),
        "acquisition archive identity changed",
    )
    range_core = _mapping(
        parents.get("range_transport_core"),
        message="range core parent is missing",
    )
    _exact_fields(
        range_core,
        frozenset({"file_sha256"}),
        name="range core parent",
    )
    _require(
        range_core.get("file_sha256") == range_core_file_sha256, "range core changed"
    )
    remote_zip_core = _mapping(
        parents.get("remote_zip_core"),
        message="remote ZIP core parent is missing",
    )
    _exact_fields(
        remote_zip_core,
        frozenset({"file_sha256", "freeze_commit"}),
        name="remote ZIP core parent",
    )
    _require(
        remote_zip_core.get("file_sha256") == remote_zip_core_file_sha256,
        "remote ZIP core changed",
    )
    _require(
        _is_commit(remote_zip_core.get("freeze_commit")), "ZIP core commit is malformed"
    )

    execution = _mapping(
        lock.get("execution"), message="structure execution is missing"
    )
    _exact_fields(
        execution,
        frozenset(
            {
                "central_directory_chunk_size_bytes",
                "concurrent_process_limit",
                "max_attempts_per_range",
                "maximum_central_directory_size_bytes",
                "run_host",
                "timeout_seconds_per_range",
            }
        ),
        name="structure execution",
    )
    _require(
        type(execution.get("central_directory_chunk_size_bytes")) is int
        and int(execution["central_directory_chunk_size_bytes"]) > 0,
        "central-directory chunk size is invalid",
    )
    _require(
        type(execution.get("maximum_central_directory_size_bytes")) is int
        and int(execution["maximum_central_directory_size_bytes"]) > 0,
        "central-directory size bound is invalid",
    )
    _require(execution.get("concurrent_process_limit") == 1, "process limit changed")
    _require(execution.get("run_host") == current_host, "structure run host changed")
    _require(
        execution.get("max_attempts_per_range") == expectation.max_attempts_per_range,
        "structure range attempt bound changed",
    )
    _require(
        execution.get("timeout_seconds_per_range") == expectation.timeout_seconds,
        "structure range timeout changed",
    )

    outputs = _mapping(lock.get("outputs"), message="structure outputs are missing")
    _exact_fields(
        outputs,
        frozenset(
            {
                "attempt_ledger_path",
                "output_root",
                "private_manifest_file_name",
                "public_lock_file_name",
            }
        ),
        name="structure outputs",
    )
    _require(
        outputs.get("output_root") == str(output_root), "structure output root changed"
    )
    for key in ("public_lock_file_name", "private_manifest_file_name"):
        value = outputs.get(key)
        _require(
            isinstance(value, str) and Path(value).name == value and bool(value),
            f"structure output name is unsafe: {key}",
        )
    _require(
        outputs.get("public_lock_file_name")
        != outputs.get("private_manifest_file_name"),
        "structure output names collide",
    )
    attempt_ledger = outputs.get("attempt_ledger_path")
    _require(
        isinstance(attempt_ledger, str) and Path(attempt_ledger).is_absolute(),
        "structure attempt-ledger path is invalid",
    )
    attempt_ledger_path = Path(str(attempt_ledger))
    _require(
        str(attempt_ledger_path.resolve()) == str(attempt_ledger_path),
        "structure attempt-ledger path is not canonical",
    )
    _require(
        output_root not in attempt_ledger_path.parents,
        "structure attempt ledger must be outside the output root",
    )
    boundaries = _mapping(
        lock.get("boundaries"), message="structure boundaries are missing"
    )
    _exact_fields(
        boundaries,
        frozenset(
            {
                "confirmation_opened",
                "held_v8_accessed",
                "member_local_headers_opened",
                "member_payloads_decompressed",
                "object_roles_assigned",
                "phase_labels_opened",
                "sensor_payloads_decoded",
                "shake_outcomes_opened",
            }
        ),
        name="structure boundaries",
    )
    for key in (
        "confirmation_opened",
        "held_v8_accessed",
        "member_local_headers_opened",
        "member_payloads_decompressed",
        "object_roles_assigned",
        "phase_labels_opened",
        "sensor_payloads_decoded",
        "shake_outcomes_opened",
    ):
        _require(boundaries.get(key) is False, f"structure boundary changed: {key}")
    identity = dict(lock)
    lock_id = identity.pop("lock_id")
    _require(lock_id == content_id(identity), "structure execution lock ID changed")
    return dict(lock)


def _range_audit(events: tuple[RemoteZipRangeEvent, ...]) -> dict[str, Any]:
    _require(bool(events), "structure parser made no exact-range requests")
    indices = [event.index for event in events]
    _require(
        len(set(indices)) == len(indices), "structure range indices are duplicated"
    )
    digest = hashlib.sha256(b"poseit-remote-zip-range-audit-v1\0")
    for event in events:
        _require(event.attempts > 0, "structure range attempt count is invalid")
        digest.update(
            f"{event.index}\0{event.start}\0{event.end}\0{event.attempts}\n".encode()
        )
    return {
        "logical_range_count": len(events),
        "range_audit_sha256": digest.hexdigest(),
        "total_range_bytes": sum(event.end - event.start + 1 for event in events),
        "transport_attempts": sum(event.attempts for event in events),
    }


def _build_artifacts(
    expectation: RemoteArchiveExpectation,
    acquisition_receipt: Mapping[str, Any],
    execution_lock: Mapping[str, Any],
    *,
    execution_lock_file_sha256: str,
    attempt_ledger_file_sha256: str,
    attempt_id: str,
    opener: RangeOpener | None = None,
) -> tuple[dict[str, Any], bytes]:
    execution = _mapping(
        execution_lock.get("execution"), message="structure execution is missing"
    )
    events: list[RemoteZipRangeEvent] = []
    layout = read_remote_zip_layout(
        expectation,
        opener=opener,
        audit=events.append,
    )
    central_directory = fetch_remote_central_directory(
        expectation,
        layout,
        opener=opener,
        audit=events.append,
        chunk_size_bytes=int(execution["central_directory_chunk_size_bytes"]),
        maximum_size_bytes=int(execution["maximum_central_directory_size_bytes"]),
    )
    members = parse_remote_central_directory(
        central_directory,
        expected_entries=layout.entry_count,
    )
    structure = remote_zip_structure_summary(layout, central_directory, members)
    audit = _range_audit(tuple(events))
    records = sorted(
        (member.as_record() for member in members),
        key=lambda item: str(item["name"]),
    )
    private_identity = {
        "schema": PRIVATE_MANIFEST_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "archive_sha256": acquisition_receipt["archive_sha256"],
        "archive_size_bytes": expectation.size_bytes,
        "acquisition_receipt_id": acquisition_receipt["receipt_id"],
        "structure_execution_lock_id": execution_lock["lock_id"],
        "structure_attempt_ledger_file_sha256": attempt_ledger_file_sha256,
        "structure_attempt_id": attempt_id,
        "members": records,
        "member_local_headers_opened": False,
        "member_payloads_decompressed": False,
        "phase_labels_opened": False,
        "sensor_payloads_decoded": False,
        "shake_outcomes_opened": False,
        "confirmation_opened": False,
        "held_v8_accessed": False,
    }
    private_manifest = {
        **private_identity,
        "manifest_id": content_id(private_identity),
    }
    private_manifest_bytes = _canonical_bytes(private_manifest)
    public_identity = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "archive_file_name": expectation.file_name,
        "archive_sha256": acquisition_receipt["archive_sha256"],
        "archive_size_bytes": expectation.size_bytes,
        "acquisition_receipt_id": acquisition_receipt["receipt_id"],
        "structure_execution_lock_file_sha256": execution_lock_file_sha256,
        "structure_execution_lock_id": execution_lock["lock_id"],
        "structure_attempt_ledger_file_sha256": attempt_ledger_file_sha256,
        "structure_attempt_id": attempt_id,
        "parent_artifacts": execution_lock["parent_artifacts"],
        "private_member_manifest_sha256": _sha256_bytes(private_manifest_bytes),
        "structure": structure,
        "range_audit": audit,
        "archive_bytes_retained": False,
        "zip_central_directory_parsed": True,
        "archive_member_names_opened": True,
        "member_local_headers_opened": False,
        "member_payloads_decompressed": False,
        "member_payload_integrity_verified": False,
        "phase_labels_opened": False,
        "sensor_payloads_decoded": False,
        "shake_outcomes_opened": False,
        "object_roles_assigned": False,
        "confirmation_opened": False,
        "held_v8_accessed": False,
    }
    return {
        **public_identity,
        "lock_id": content_id(public_identity),
    }, private_manifest_bytes


def _write_file(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _consume_attempt_ledger(
    path: Path,
    *,
    execution_lock_file_sha256: str,
    execution_lock_id: str,
    acquisition_receipt_file_sha256: str,
    acquisition_receipt_id: str,
    output_root: Path,
) -> tuple[str, str]:
    identity = {
        "schema": "bayesian-phystwin.poseit-remote-structure-attempt-ledger",
        "schema_version": 1,
        "status": "consumed-before-first-remote-structure-range",
        "structure_execution_lock_file_sha256": execution_lock_file_sha256,
        "structure_execution_lock_id": execution_lock_id,
        "acquisition_receipt_file_sha256": acquisition_receipt_file_sha256,
        "acquisition_receipt_id": acquisition_receipt_id,
        "output_root": str(output_root),
        "member_local_headers_opened": False,
        "member_payloads_decompressed": False,
        "phase_labels_opened": False,
        "sensor_payloads_decoded": False,
        "shake_outcomes_opened": False,
        "confirmation_opened": False,
        "held_v8_accessed": False,
    }
    payload = {**identity, "attempt_id": content_id(identity)}
    payload_bytes = _canonical_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not path.exists(), f"structure attempt ledger already exists: {path}")
    _write_file(path, payload_bytes)
    _fsync_directory(path.parent)
    return str(payload["attempt_id"]), _sha256_bytes(payload_bytes)


def _write_bundle_once(
    output_root: Path,
    *,
    public_file_name: str,
    public_payload: bytes,
    private_file_name: str,
    private_payload: bytes,
) -> None:
    _require(not output_root.exists(), f"output root already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = output_root.with_name(f".{output_root.name}.partial")
    _require(not stage.exists(), f"staging root already exists: {stage}")
    stage.mkdir(mode=0o700)
    committed = False
    try:
        _write_file(stage / private_file_name, private_payload)
        _write_file(stage / public_file_name, public_payload)
        _fsync_directory(stage)
        os.rename(stage, output_root)
        committed = True
        _fsync_directory(output_root.parent)
    finally:
        if not committed and stage.exists():
            shutil.rmtree(stage)


def _run(
    *,
    acquisition_receipt_path: Path,
    execution_lock_path: Path,
    expected_execution_lock_sha256: str,
    protocol_path: Path,
    mapping_constraints_path: Path,
    method_lock_path: Path,
    range_transport_lock_path: Path,
    range_transport_core_path: Path,
    remote_zip_core_path: Path,
    output_root: Path,
    opener: RangeOpener | None = None,
    current_host: str | None = None,
) -> dict[str, Any]:
    _require(
        _is_sha256(expected_execution_lock_sha256),
        "execution lock SHA-256 is malformed",
    )
    actual_execution_lock_sha256 = _sha256(execution_lock_path)
    _require(
        actual_execution_lock_sha256 == expected_execution_lock_sha256,
        "structure execution lock file changed",
    )
    protocol = load_poseit_real_decision_protocol(protocol_path)
    mapping = load_poseit_preaccess_mapping_constraints(
        mapping_constraints_path,
        parent_protocol_path=protocol_path,
    )
    method = load_poseit_real_decision_method_lock(
        method_lock_path,
        parent_protocol_path=protocol_path,
        mapping_constraints_path=mapping_constraints_path,
    )
    transport = load_poseit_range_transport_lock(
        range_transport_lock_path,
        parent_protocol_path=protocol_path,
        mapping_constraints_path=mapping_constraints_path,
        method_lock_path=method_lock_path,
        range_transport_core_path=range_transport_core_path,
    )
    expectation = _expectation(transport)
    protocol_digests = {
        "file_sha256": poseit_protocol_file_sha256(protocol_path),
        "config_sha256": poseit_protocol_config_sha256(protocol),
    }
    mapping_digests = {
        "file_sha256": poseit_mapping_constraints_file_sha256(mapping_constraints_path),
        "config_sha256": poseit_mapping_constraints_config_sha256(mapping),
    }
    method_digests = {
        "file_sha256": poseit_method_lock_file_sha256(method_lock_path),
        "config_sha256": poseit_method_lock_config_sha256(method),
    }
    transport_digests = {
        "file_sha256": poseit_range_transport_lock_file_sha256(
            range_transport_lock_path
        ),
        "config_sha256": poseit_range_transport_lock_config_sha256(transport),
    }
    range_core_file_sha256 = _sha256(range_transport_core_path)
    remote_zip_core_file_sha256 = _sha256(remote_zip_core_path)
    receipt_file_sha256 = _sha256(acquisition_receipt_path)
    receipt = _validate_acquisition_receipt(
        load_strict_json_object(
            acquisition_receipt_path,
            label="PoseIt range-hash acquisition receipt",
        ),
        expectation=expectation,
        source_file_id=str(transport["source_identity"]["google_drive_file_id"]),
        protocol_file_sha256=protocol_digests["file_sha256"],
        protocol_config_sha256=protocol_digests["config_sha256"],
        mapping_file_sha256=mapping_digests["file_sha256"],
        mapping_config_sha256=mapping_digests["config_sha256"],
        method_file_sha256=method_digests["file_sha256"],
        method_config_sha256=method_digests["config_sha256"],
        transport_file_sha256=transport_digests["file_sha256"],
        transport_config_sha256=transport_digests["config_sha256"],
        range_core_file_sha256=range_core_file_sha256,
    )
    execution_lock = _validate_execution_lock(
        load_strict_json_object(
            execution_lock_path,
            label="PoseIt remote structure execution lock",
        ),
        expectation=expectation,
        output_root=output_root,
        acquisition_receipt=receipt,
        acquisition_receipt_file_sha256=receipt_file_sha256,
        parent_digests={
            "protocol": protocol_digests,
            "mapping_constraints": mapping_digests,
            "method_lock": method_digests,
            "range_transport_lock": transport_digests,
        },
        range_core_file_sha256=range_core_file_sha256,
        remote_zip_core_file_sha256=remote_zip_core_file_sha256,
        current_host=current_host or socket.gethostname(),
    )
    outputs = _mapping(
        execution_lock.get("outputs"), message="structure outputs are missing"
    )
    attempt_id, attempt_ledger_file_sha256 = _consume_attempt_ledger(
        Path(str(outputs["attempt_ledger_path"])),
        execution_lock_file_sha256=actual_execution_lock_sha256,
        execution_lock_id=str(execution_lock["lock_id"]),
        acquisition_receipt_file_sha256=receipt_file_sha256,
        acquisition_receipt_id=str(receipt["receipt_id"]),
        output_root=output_root,
    )
    public_lock, private_manifest_bytes = _build_artifacts(
        expectation,
        receipt,
        execution_lock,
        execution_lock_file_sha256=actual_execution_lock_sha256,
        attempt_ledger_file_sha256=attempt_ledger_file_sha256,
        attempt_id=attempt_id,
        opener=opener,
    )
    public_name = str(outputs["public_lock_file_name"])
    private_name = str(outputs["private_manifest_file_name"])
    public_bytes = _canonical_bytes(public_lock)
    _write_bundle_once(
        output_root,
        public_file_name=public_name,
        public_payload=public_bytes,
        private_file_name=private_name,
        private_payload=private_manifest_bytes,
    )
    return {
        "archive_sha256": public_lock["archive_sha256"],
        "lock_id": public_lock["lock_id"],
        "member_count": (
            int(public_lock["structure"]["regular_member_count"])
            + int(public_lock["structure"]["directory_member_count"])
        ),
        "private_manifest_sha256": public_lock["private_member_manifest_sha256"],
        "public_lock_sha256": _sha256_bytes(public_bytes),
        "range_audit": public_lock["range_audit"],
        "structure_attempt_id": attempt_id,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition-receipt", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--expected-execution-lock-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--mapping-constraints", type=Path, required=True)
    parser.add_argument("--method-lock", type=Path, required=True)
    parser.add_argument("--range-transport-lock", type=Path, required=True)
    parser.add_argument("--range-transport-core", type=Path, required=True)
    parser.add_argument("--remote-zip-core", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    result = _run(
        acquisition_receipt_path=arguments.acquisition_receipt.resolve(strict=True),
        execution_lock_path=arguments.execution_lock.resolve(strict=True),
        expected_execution_lock_sha256=arguments.expected_execution_lock_sha256,
        protocol_path=arguments.protocol.resolve(strict=True),
        mapping_constraints_path=arguments.mapping_constraints.resolve(strict=True),
        method_lock_path=arguments.method_lock.resolve(strict=True),
        range_transport_lock_path=arguments.range_transport_lock.resolve(strict=True),
        range_transport_core_path=arguments.range_transport_core.resolve(strict=True),
        remote_zip_core_path=arguments.remote_zip_core.resolve(strict=True),
        output_root=arguments.output_root.resolve(),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
