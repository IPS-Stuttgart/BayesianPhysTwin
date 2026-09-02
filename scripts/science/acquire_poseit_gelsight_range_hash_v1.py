#!/usr/bin/env python3
"""Hash the official PoseIt archive by exact ranges without retaining it."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bayesian_phystwin._portable_contracts import content_id
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
    RemoteHashProgress,
    RemoteHashResult,
    hash_remote_archive,
)

SCHEMA = "bayesian-phystwin.poseit-range-hash-acquisition-receipt"
SCHEMA_VERSION = 1
PROGRESS_SCHEMA = "bayesian-phystwin.poseit-range-hash-progress"

_Hasher = Callable[..., RemoteHashResult]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _write_atomic(path: Path, payload: object) -> None:
    stage = path.with_name(f".{path.name}.partial")
    _require(not stage.exists(), f"staging path already exists: {stage}")
    try:
        with stage.open("xb") as stream:
            stream.write(_canonical_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(stage, path)
    finally:
        stage.unlink(missing_ok=True)


def _write_once(path: Path, payload: object) -> None:
    stage = path.with_name(f".{path.name}.partial")
    _require(not path.exists(), f"output already exists: {path}")
    _require(not stage.exists(), f"staging path already exists: {stage}")
    linked = False
    try:
        with stage.open("xb") as stream:
            stream.write(_canonical_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(stage, path)
        linked = True
    except BaseException:
        if linked and stage.samefile(path):
            path.unlink()
        raise
    finally:
        stage.unlink(missing_ok=True)


def _expectation(lock: dict[str, Any]) -> RemoteArchiveExpectation:
    source = lock["source_identity"]
    execution = lock["execution"]
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


def _acquire_range_hash(
    receipt: Path,
    progress_path: Path,
    protocol_path: Path,
    mapping_constraints_path: Path,
    method_lock_path: Path,
    transport_lock_path: Path,
    range_transport_core_path: Path,
    *,
    expected_protocol_sha256: str,
    expected_mapping_constraints_sha256: str,
    expected_method_lock_sha256: str,
    expected_transport_lock_sha256: str,
    opener: RangeOpener | None = None,
    hasher: _Hasher | None = None,
) -> dict[str, Any]:
    _require(receipt != progress_path, "receipt and progress paths must differ")
    _require(not receipt.exists(), f"receipt already exists: {receipt}")
    _require(len(expected_protocol_sha256) == 64, "protocol SHA-256 is malformed")
    _require(
        len(expected_mapping_constraints_sha256) == 64,
        "mapping-constraint SHA-256 is malformed",
    )
    _require(len(expected_method_lock_sha256) == 64, "method-lock SHA-256 is malformed")
    _require(
        len(expected_transport_lock_sha256) == 64,
        "transport-lock SHA-256 is malformed",
    )

    protocol_file_sha256 = poseit_protocol_file_sha256(protocol_path)
    _require(
        protocol_file_sha256 == expected_protocol_sha256,
        "protocol file SHA-256 changed",
    )
    protocol = load_poseit_real_decision_protocol(protocol_path)
    mapping_file_sha256 = poseit_mapping_constraints_file_sha256(
        mapping_constraints_path
    )
    _require(
        mapping_file_sha256 == expected_mapping_constraints_sha256,
        "mapping-constraint file SHA-256 changed",
    )
    mapping = load_poseit_preaccess_mapping_constraints(
        mapping_constraints_path,
        parent_protocol_path=protocol_path,
    )
    method_file_sha256 = poseit_method_lock_file_sha256(method_lock_path)
    _require(
        method_file_sha256 == expected_method_lock_sha256,
        "method-lock file SHA-256 changed",
    )
    method = load_poseit_real_decision_method_lock(
        method_lock_path,
        parent_protocol_path=protocol_path,
        mapping_constraints_path=mapping_constraints_path,
    )
    transport_file_sha256 = poseit_range_transport_lock_file_sha256(transport_lock_path)
    _require(
        transport_file_sha256 == expected_transport_lock_sha256,
        "transport-lock file SHA-256 changed",
    )
    transport = load_poseit_range_transport_lock(
        transport_lock_path,
        parent_protocol_path=protocol_path,
        mapping_constraints_path=mapping_constraints_path,
        method_lock_path=method_lock_path,
        range_transport_core_path=range_transport_core_path,
    )
    expectation = _expectation(transport)
    expected_chunks = (
        expectation.size_bytes + expectation.chunk_size_bytes - 1
    ) // expectation.chunk_size_bytes

    receipt.parent.mkdir(parents=True, exist_ok=True)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_identity = {
        "schema": PROGRESS_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "transport_lock_file_sha256": transport_file_sha256,
        "transport_lock_config_sha256": (
            poseit_range_transport_lock_config_sha256(transport)
        ),
        "archive_size_bytes": expectation.size_bytes,
        "chunk_size_bytes": expectation.chunk_size_bytes,
        "total_chunks": expected_chunks,
        "completed_chunks": 0,
        "bytes_hashed": 0,
        "transport_attempts": 0,
        "authoritative_result": False,
        "archive_member_names_opened": False,
        "member_payload_bytes_opened": False,
        "phase_labels_opened": False,
        "shake_outcomes_opened": False,
        "confirmation_opened": False,
        "held_v8_accessed": False,
    }
    _write_atomic(progress_path, progress_identity)

    def update_progress(item: RemoteHashProgress) -> None:
        _write_atomic(
            progress_path,
            {
                **progress_identity,
                "completed_chunks": item.completed_chunks,
                "bytes_hashed": item.bytes_hashed,
                "transport_attempts": item.transport_attempts,
            },
        )

    if hasher is None:
        if opener is None:
            result = hash_remote_archive(expectation, progress=update_progress)
        else:
            result = hash_remote_archive(
                expectation,
                opener=opener,
                progress=update_progress,
            )
    else:
        kwargs: dict[str, object] = {"progress": update_progress}
        if opener is not None:
            kwargs["opener"] = opener
        result = hasher(expectation, **kwargs)
    _require(
        result.archive_size_bytes == expectation.size_bytes,
        "range hash returned the wrong archive size",
    )
    _require(result.chunk_count == expected_chunks, "range hash chunk count changed")
    _require(len(result.archive_sha256) == 64, "range hash SHA-256 is malformed")

    identity = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "source": "official-public-google-drive-file-exact-range-hash",
        "source_file_id": transport["source_identity"]["google_drive_file_id"],
        "source_file_name": expectation.file_name,
        "source_url": expectation.source_url,
        "source_last_modified": expectation.last_modified,
        "archive_size_bytes": result.archive_size_bytes,
        "archive_sha256": result.archive_sha256,
        "chunk_size_bytes": result.chunk_size_bytes,
        "chunk_count": result.chunk_count,
        "max_workers": expectation.max_workers,
        "max_attempts_per_range": expectation.max_attempts_per_range,
        "transport_attempts": result.transport_attempts,
        "protocol_file_sha256": protocol_file_sha256,
        "protocol_config_sha256": poseit_protocol_config_sha256(protocol),
        "mapping_constraints_file_sha256": mapping_file_sha256,
        "mapping_constraints_config_sha256": (
            poseit_mapping_constraints_config_sha256(mapping)
        ),
        "method_lock_file_sha256": method_file_sha256,
        "method_lock_config_sha256": poseit_method_lock_config_sha256(method),
        "transport_lock_file_sha256": transport_file_sha256,
        "transport_lock_config_sha256": (
            poseit_range_transport_lock_config_sha256(transport)
        ),
        "range_transport_core_file_sha256": poseit_protocol_file_sha256(
            range_transport_core_path
        ),
        "archive_bytes_streamed_opaquely": True,
        "archive_bytes_retained": False,
        "zip_central_directory_parsed": False,
        "archive_member_names_opened": False,
        "member_payload_bytes_opened": False,
        "phase_labels_opened": False,
        "sensor_payloads_opened": False,
        "shake_outcomes_opened": False,
        "object_roles_assigned": False,
        "confirmation_opened": False,
        "held_v8_accessed": False,
    }
    receipt_payload = {**identity, "receipt_id": content_id(identity)}
    _write_once(receipt, receipt_payload)
    _write_atomic(
        progress_path,
        {
            **progress_identity,
            "completed_chunks": result.chunk_count,
            "bytes_hashed": result.archive_size_bytes,
            "transport_attempts": result.transport_attempts,
            "authoritative_result": True,
            "receipt_id": receipt_payload["receipt_id"],
        },
    )
    return receipt_payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--mapping-constraints", type=Path, required=True)
    parser.add_argument("--expected-mapping-constraints-sha256", required=True)
    parser.add_argument("--method-lock", type=Path, required=True)
    parser.add_argument("--expected-method-lock-sha256", required=True)
    parser.add_argument("--transport-lock", type=Path, required=True)
    parser.add_argument("--expected-transport-lock-sha256", required=True)
    parser.add_argument("--range-transport-core", type=Path, required=True)
    arguments = parser.parse_args()
    result = _acquire_range_hash(
        arguments.receipt.resolve(),
        arguments.progress.resolve(),
        arguments.protocol.resolve(strict=True),
        arguments.mapping_constraints.resolve(strict=True),
        arguments.method_lock.resolve(strict=True),
        arguments.transport_lock.resolve(strict=True),
        arguments.range_transport_core.resolve(strict=True),
        expected_protocol_sha256=arguments.expected_protocol_sha256,
        expected_mapping_constraints_sha256=(
            arguments.expected_mapping_constraints_sha256
        ),
        expected_method_lock_sha256=arguments.expected_method_lock_sha256,
        expected_transport_lock_sha256=arguments.expected_transport_lock_sha256,
    )
    print(
        json.dumps(
            {
                "archive_sha256": result["archive_sha256"],
                "archive_size_bytes": result["archive_size_bytes"],
                "archive_bytes_retained": False,
                "receipt": str(arguments.receipt.resolve()),
                "receipt_id": result["receipt_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
