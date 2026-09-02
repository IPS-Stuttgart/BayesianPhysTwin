from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import urllib.request
import zipfile
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType

import pytest

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin_experiments.poseit_remote_archive import (
    RemoteArchiveExpectation,
)
from bayesian_phystwin_experiments.poseit_remote_zip import RemoteZipRangeEvent

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "scripts" / "science" / "build_poseit_remote_archive_structure_lock_v1.py"
)


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "poseit_remote_structure_lock", SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Response:
    def __init__(self, data: bytes, start: int, end: int) -> None:
        payload = data[start : end + 1]
        self._payload = payload
        self._position = 0
        self.status = 206
        self.headers: Mapping[str, str] = {
            "Accept-Ranges": "bytes",
            "Content-Disposition": 'attachment; filename="gelsight.zip"',
            "Content-Length": str(len(payload)),
            "Content-Range": f"bytes {start}-{end}/{len(data)}",
            "Content-Type": "application/octet-stream",
            "Last-Modified": "Sat, 20 Aug 2022 02:26:04 GMT",
        }

    def read(self, amount: int = -1) -> bytes:
        if amount < 0:
            amount = len(self._payload) - self._position
        block = self._payload[self._position : self._position + amount]
        self._position += len(block)
        return block

    def geturl(self) -> str:
        return "https://drive.usercontent.google.com/download"

    def close(self) -> None:
        pass


def _archive() -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("object-a/pose-01/metadata.json", "PRIVATE LABEL")
        bundle.writestr("object-a/pose-01/sensor.npy", b"PRIVATE PAYLOAD")
        bundle.writestr("object-b/pose-16/sensor.npy", b"SECOND PAYLOAD")
    return stream.getvalue()


def _expectation(data: bytes) -> RemoteArchiveExpectation:
    return RemoteArchiveExpectation(
        source_url="https://drive.usercontent.google.com/download?id=frozen",
        file_name="gelsight.zip",
        size_bytes=len(data),
        last_modified="Sat, 20 Aug 2022 02:26:04 GMT",
        chunk_size_bytes=32,
        max_workers=1,
        max_attempts_per_range=3,
        timeout_seconds=5.0,
    )


def _opener(data: bytes, requests: list[tuple[int, int]]):
    def open_range(request: urllib.request.Request, timeout: float) -> _Response:
        assert timeout == 5.0
        header = request.get_header("Range")
        assert header is not None
        start_text, end_text = header.removeprefix("bytes=").split("-", maxsplit=1)
        start, end = int(start_text), int(end_text)
        requests.append((start, end))
        return _Response(data, start, end)

    return open_range


def _parent_digests() -> dict[str, dict[str, str]]:
    return {
        name: {
            "file_sha256": character * 64,
            "config_sha256": character.upper() * 64,
        }
        for name, character in (
            ("protocol", "a"),
            ("mapping_constraints", "b"),
            ("method_lock", "c"),
            ("range_transport_lock", "d"),
        )
    }


def _execution_lock(
    expectation: RemoteArchiveExpectation,
    output_root: Path,
    *,
    receipt_id: str = "e" * 64,
    archive_sha256: str = "f" * 64,
    receipt_file_sha256: str = "1" * 64,
) -> dict[str, object]:
    parents: dict[str, object] = {
        **_parent_digests(),
        "acquisition_receipt": {
            "file_sha256": receipt_file_sha256,
            "receipt_id": receipt_id,
            "archive_sha256": archive_sha256,
        },
        "range_transport_core": {"file_sha256": "2" * 64},
        "remote_zip_core": {
            "file_sha256": "3" * 64,
            "freeze_commit": "4" * 40,
        },
    }
    identity = {
        "contract": "poseit-real-decision-remote-structure-transport-lock-v1",
        "schema_version": 1,
        "status": "frozen-after-full-range-hash-and-before-central-directory-access",
        "source_identity": {
            "source_url": expectation.source_url,
            "file_name": expectation.file_name,
            "archive_size_bytes": expectation.size_bytes,
            "last_modified": expectation.last_modified,
        },
        "parent_artifacts": parents,
        "execution": {
            "central_directory_chunk_size_bytes": 31,
            "maximum_central_directory_size_bytes": 1024 * 1024,
            "concurrent_process_limit": 1,
            "run_host": "test-host",
            "max_attempts_per_range": expectation.max_attempts_per_range,
            "timeout_seconds_per_range": expectation.timeout_seconds,
        },
        "outputs": {
            "output_root": str(output_root),
            "attempt_ledger_path": str(
                output_root.parent / "structure-attempt-v1.json"
            ),
            "public_lock_file_name": "archive-structure-lock-v1.json",
            "private_manifest_file_name": "private-member-manifest-v1.json",
        },
        "boundaries": {
            "confirmation_opened": False,
            "held_v8_accessed": False,
            "member_local_headers_opened": False,
            "member_payloads_decompressed": False,
            "object_roles_assigned": False,
            "phase_labels_opened": False,
            "sensor_payloads_decoded": False,
            "shake_outcomes_opened": False,
        },
    }
    return {**identity, "lock_id": content_id(identity)}


def _valid_receipt(
    module: ModuleType,
    expectation: RemoteArchiveExpectation,
) -> tuple[dict[str, object], dict[str, str]]:
    digests = {
        "protocol_file_sha256": "a" * 64,
        "protocol_config_sha256": "A" * 64,
        "mapping_constraints_file_sha256": "b" * 64,
        "mapping_constraints_config_sha256": "B" * 64,
        "method_lock_file_sha256": "c" * 64,
        "method_lock_config_sha256": "C" * 64,
        "transport_lock_file_sha256": "d" * 64,
        "transport_lock_config_sha256": "D" * 64,
        "range_transport_core_file_sha256": "2" * 64,
    }
    chunks = (
        expectation.size_bytes + expectation.chunk_size_bytes - 1
    ) // expectation.chunk_size_bytes
    identity: dict[str, object] = {
        "schema": module.ACQUISITION_RECEIPT_SCHEMA,
        "schema_version": 1,
        "source": "official-public-google-drive-file-exact-range-hash",
        "source_file_id": "official-file",
        "source_file_name": expectation.file_name,
        "source_url": expectation.source_url,
        "source_last_modified": expectation.last_modified,
        "archive_size_bytes": expectation.size_bytes,
        "archive_sha256": "f" * 64,
        "chunk_size_bytes": expectation.chunk_size_bytes,
        "chunk_count": chunks,
        "max_workers": expectation.max_workers,
        "max_attempts_per_range": expectation.max_attempts_per_range,
        "transport_attempts": chunks,
        **digests,
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
    return {**identity, "receipt_id": content_id(identity)}, digests


def test_remote_structure_builds_content_bound_private_and_public_artifacts(
    tmp_path: Path,
) -> None:
    module = _module()
    data = _archive()
    expectation = _expectation(data)
    receipt = {"archive_sha256": "f" * 64, "receipt_id": "e" * 64}
    execution_lock = _execution_lock(expectation, tmp_path / "structure")
    requests: list[tuple[int, int]] = []

    public, private_bytes = module._build_artifacts(
        expectation,
        receipt,
        execution_lock,
        execution_lock_file_sha256="5" * 64,
        attempt_ledger_file_sha256="6" * 64,
        attempt_id="7" * 64,
        opener=_opener(data, requests),
    )

    private = json.loads(private_bytes)
    assert public["lock_id"] == content_id(
        {key: value for key, value in public.items() if key != "lock_id"}
    )
    assert private["manifest_id"] == content_id(
        {key: value for key, value in private.items() if key != "manifest_id"}
    )
    assert (
        public["private_member_manifest_sha256"]
        == hashlib.sha256(private_bytes).hexdigest()
    )
    assert [record["name"] for record in private["members"]] == [
        "object-a/pose-01/metadata.json",
        "object-a/pose-01/sensor.npy",
        "object-b/pose-16/sensor.npy",
    ]
    assert "object-a" not in json.dumps(public)
    assert public["zip_central_directory_parsed"] is True
    assert public["member_payloads_decompressed"] is False
    assert public["structure_attempt_id"] == "7" * 64
    assert public["shake_outcomes_opened"] is False
    assert public["confirmation_opened"] is False
    central_offset = int(public["structure"]["central_directory_offset"])
    assert requests
    assert all(start >= central_offset for start, _ in requests)
    assert public["range_audit"]["logical_range_count"] == len(requests)
    assert public["range_audit"]["transport_attempts"] == len(requests)


def test_acquisition_receipt_validation_is_exact_and_fail_closed() -> None:
    module = _module()
    expectation = _expectation(_archive())
    receipt, digests = _valid_receipt(module, expectation)

    validated = module._validate_acquisition_receipt(
        receipt,
        expectation=expectation,
        source_file_id="official-file",
        protocol_file_sha256=digests["protocol_file_sha256"],
        protocol_config_sha256=digests["protocol_config_sha256"],
        mapping_file_sha256=digests["mapping_constraints_file_sha256"],
        mapping_config_sha256=digests["mapping_constraints_config_sha256"],
        method_file_sha256=digests["method_lock_file_sha256"],
        method_config_sha256=digests["method_lock_config_sha256"],
        transport_file_sha256=digests["transport_lock_file_sha256"],
        transport_config_sha256=digests["transport_lock_config_sha256"],
        range_core_file_sha256=digests["range_transport_core_file_sha256"],
    )
    assert validated["receipt_id"] == receipt["receipt_id"]

    drifted = dict(receipt)
    drifted["zip_central_directory_parsed"] = True
    with pytest.raises(ValueError, match="boundary changed"):
        module._validate_acquisition_receipt(
            drifted,
            expectation=expectation,
            source_file_id="official-file",
            protocol_file_sha256=digests["protocol_file_sha256"],
            protocol_config_sha256=digests["protocol_config_sha256"],
            mapping_file_sha256=digests["mapping_constraints_file_sha256"],
            mapping_config_sha256=digests["mapping_constraints_config_sha256"],
            method_file_sha256=digests["method_lock_file_sha256"],
            method_config_sha256=digests["method_lock_config_sha256"],
            transport_file_sha256=digests["transport_lock_file_sha256"],
            transport_config_sha256=digests["transport_lock_config_sha256"],
            range_core_file_sha256=digests["range_transport_core_file_sha256"],
        )


def test_execution_lock_binds_receipt_core_host_and_output(tmp_path: Path) -> None:
    module = _module()
    expectation = _expectation(_archive())
    output_root = (tmp_path / "structure").resolve()
    lock = _execution_lock(expectation, output_root)
    receipt = {"receipt_id": "e" * 64, "archive_sha256": "f" * 64}

    validated = module._validate_execution_lock(
        lock,
        expectation=expectation,
        output_root=output_root,
        acquisition_receipt=receipt,
        acquisition_receipt_file_sha256="1" * 64,
        parent_digests=_parent_digests(),
        range_core_file_sha256="2" * 64,
        remote_zip_core_file_sha256="3" * 64,
        current_host="test-host",
    )
    assert validated["lock_id"] == lock["lock_id"]

    with pytest.raises(ValueError, match="output root changed"):
        module._validate_execution_lock(
            lock,
            expectation=expectation,
            output_root=(tmp_path / "different").resolve(),
            acquisition_receipt=receipt,
            acquisition_receipt_file_sha256="1" * 64,
            parent_digests=_parent_digests(),
            range_core_file_sha256="2" * 64,
            remote_zip_core_file_sha256="3" * 64,
            current_host="test-host",
        )

    invalid_outputs = dict(lock["outputs"])
    invalid_outputs["attempt_ledger_path"] = str(output_root / "attempt.json")
    invalid_identity = {key: value for key, value in lock.items() if key != "lock_id"}
    invalid_identity["outputs"] = invalid_outputs
    invalid_lock = {**invalid_identity, "lock_id": content_id(invalid_identity)}
    with pytest.raises(ValueError, match="outside the output root"):
        module._validate_execution_lock(
            invalid_lock,
            expectation=expectation,
            output_root=output_root,
            acquisition_receipt=receipt,
            acquisition_receipt_file_sha256="1" * 64,
            parent_digests=_parent_digests(),
            range_core_file_sha256="2" * 64,
            remote_zip_core_file_sha256="3" * 64,
            current_host="test-host",
        )


def test_structure_bundle_is_atomic_and_write_once(tmp_path: Path) -> None:
    module = _module()
    output_root = tmp_path / "structure"

    module._write_bundle_once(
        output_root,
        public_file_name="public.json",
        public_payload=b"public\n",
        private_file_name="private.json",
        private_payload=b"private\n",
    )

    assert (output_root / "public.json").read_bytes() == b"public\n"
    assert (output_root / "private.json").read_bytes() == b"private\n"
    assert not (tmp_path / ".structure.partial").exists()
    with pytest.raises(ValueError, match="already exists"):
        module._write_bundle_once(
            output_root,
            public_file_name="public.json",
            public_payload=b"replacement\n",
            private_file_name="private.json",
            private_payload=b"replacement\n",
        )


def test_structure_attempt_ledger_is_consumed_before_remote_reads(
    tmp_path: Path,
) -> None:
    module = _module()
    ledger = tmp_path / "structure-attempt-v1.json"

    attempt_id, file_sha256 = module._consume_attempt_ledger(
        ledger,
        execution_lock_file_sha256="1" * 64,
        execution_lock_id="2" * 64,
        acquisition_receipt_file_sha256="3" * 64,
        acquisition_receipt_id="4" * 64,
        output_root=tmp_path / "structure",
    )

    payload = json.loads(ledger.read_bytes())
    identity = {key: value for key, value in payload.items() if key != "attempt_id"}
    assert attempt_id == content_id(identity)
    assert payload["status"] == "consumed-before-first-remote-structure-range"
    assert file_sha256 == hashlib.sha256(ledger.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="already exists"):
        module._consume_attempt_ledger(
            ledger,
            execution_lock_file_sha256="1" * 64,
            execution_lock_id="2" * 64,
            acquisition_receipt_file_sha256="3" * 64,
            acquisition_receipt_id="4" * 64,
            output_root=tmp_path / "structure",
        )


def test_range_audit_rejects_duplicate_logical_indices() -> None:
    module = _module()
    events = (
        RemoteZipRangeEvent(index=0, start=1, end=2, attempts=1),
        RemoteZipRangeEvent(index=0, start=3, end=4, attempts=1),
    )

    with pytest.raises(ValueError, match="duplicated"):
        module._range_audit(events)
