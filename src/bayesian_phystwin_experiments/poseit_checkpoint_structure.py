"""Explicit, one-shot structure access through a completed checkpoint acquisition."""

from __future__ import annotations

import hashlib
import io
import re
import socket
import sys
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from bayesian_phystwin._portable_contracts import exact_revision, sha256_digest
from bayesian_phystwin_experiments import poseit_checkpoint_acquisition as acquisition
from bayesian_phystwin_experiments.poseit_hash_checkpoint import RHashCheckpointEngine
from bayesian_phystwin_experiments.poseit_remote_archive import (
    RangeOpener,
    RangeResponse,
    RemoteArchiveExpectation,
    fetch_exact_range,
)
from bayesian_phystwin_experiments.poseit_remote_zip import (
    fetch_remote_central_directory,
    parse_remote_central_directory,
    read_remote_zip_layout,
    remote_zip_structure_summary,
)

ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION_FILES = frozenset(
    {
        "scripts/science/acquire_poseit_checkpointed_range_hash_v1.py",
        "scripts/science/build_poseit_checkpoint_structure_v1.py",
        "src/bayesian_phystwin/__init__.py",
        "src/bayesian_phystwin/_canonical_contracts.py",
        "src/bayesian_phystwin/_portable_contracts.py",
        "src/bayesian_phystwin_experiments/__init__.py",
        "src/bayesian_phystwin_experiments/poseit_checkpoint_acquisition.py",
        "src/bayesian_phystwin_experiments/poseit_checkpoint_structure.py",
        "src/bayesian_phystwin_experiments/poseit_hash_checkpoint.py",
        "src/bayesian_phystwin_experiments/poseit_remote_archive.py",
        "src/bayesian_phystwin_experiments/poseit_remote_zip.py",
    }
)
BOUNDARIES = {
    "member_local_headers_parsed": False,
    "member_payloads_decompressed": False,
    "member_payload_integrity_verified": False,
    "phase_labels_opened": False,
    "sensor_payloads_decoded": False,
    "shake_outcomes_opened": False,
    "object_roles_assigned": False,
    "confirmation_opened": False,
    "held_v8_accessed": False,
    "scientific_result": False,
}
_RANGE = re.compile(r"bytes=(0|[1-9][0-9]*)-(0|[1-9][0-9]*)")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _file_sha256(path: Path) -> str:
    _require(
        path.is_absolute() and path.resolve() == path and path.is_file(),
        "missing, noncanonical, or linked administrative file",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class _Chunk:
    start: int
    end: int
    sha256: str
    record_id: str


class _MemoryResponse(io.BytesIO):
    """Parser adapter, not a second HTTP observation or transport attempt."""

    def __init__(
        self, data: bytes, expectation: RemoteArchiveExpectation, start: int, end: int
    ) -> None:
        super().__init__(data)
        self.status = 206
        self.headers: Mapping[str, str] = {
            "Accept-Ranges": "bytes",
            "Content-Disposition": f'attachment; filename="{expectation.file_name}"',
            "Content-Length": str(len(data)),
            "Content-Range": f"bytes {start}-{end}/{expectation.size_bytes}",
            "Content-Type": "application/octet-stream",
            "Last-Modified": expectation.last_modified,
        }
        self.url = expectation.source_url

    def geturl(self) -> str:
        return self.url

    def read(self, amount: int | None = -1) -> bytes:
        return super().read(amount)


class _AuthenticatedRanges:
    def __init__(
        self,
        expectation: RemoteArchiveExpectation,
        chunks: tuple[_Chunk, ...],
        *,
        opener: RangeOpener,
        maximum_cached_chunks: int,
    ) -> None:
        self.expectation = expectation
        self.chunks = chunks
        self.opener = opener
        self.maximum_cached_chunks = maximum_cached_chunks
        self.cache: dict[int, bytes] = {}
        self.network: list[dict[str, Any]] = []
        self.logical: list[dict[str, Any]] = []
        self.failed = False
        self.closed = False

    def _open(self, request: urllib.request.Request, timeout: float) -> RangeResponse:
        observation: dict[str, Any] = {
            "range": request.get_header("Range"),
            "http_status": None,
            "body_identity_accepted": False,
        }
        self.network.append(observation)
        try:
            response = self.opener(request, timeout)
        except urllib.error.HTTPError as error:
            # The frozen range core must see HTTP rejection, not retry it as a socket.
            response = cast(RangeResponse, error)
        observation["http_status"] = response.status
        return response

    def __call__(
        self, request: urllib.request.Request, timeout: float
    ) -> _MemoryResponse:
        _require(not self.failed and not self.closed, "structure reader is terminal")
        try:
            _require(
                request.get_method() == "GET"
                and request.full_url == self.expectation.source_url
                and request.data is None
                and timeout == self.expectation.timeout_seconds,
                "parser request changed the registered source or timeout",
            )
            match = _RANGE.fullmatch(request.get_header("Range") or "")
            _require(match is not None, "parser request has an invalid exact range")
            assert match is not None
            start, end = map(int, match.groups())
            _require(
                0 <= start <= end < self.expectation.size_bytes,
                "parser range is outside the archive",
            )
            size = self.expectation.chunk_size_bytes
            indices = tuple(range(start // size, end // size + 1))
            needed = set(indices) - set(self.cache)
            _require(
                len(self.cache) + len(needed) <= self.maximum_cached_chunks,
                "structure opaque-byte memory bound exceeded",
            )
            pieces: list[bytes] = []
            for index in indices:
                chunk = self.chunks[index]
                if index not in self.cache:
                    block = fetch_exact_range(
                        self.expectation,
                        index=index,
                        start=chunk.start,
                        end=chunk.end,
                        opener=self._open,
                    )
                    _require(
                        hashlib.sha256(block.data).hexdigest() == chunk.sha256,
                        "remote bytes differ from the completed acquisition",
                    )
                    self.cache[index] = block.data
                    self.network[-1]["body_identity_accepted"] = True
                pieces.append(
                    self.cache[index][
                        max(start, chunk.start) - chunk.start : min(end, chunk.end)
                        - chunk.start
                        + 1
                    ]
                )
            payload = b"".join(pieces)
            self.logical.append(
                {
                    "start": start,
                    "end": end,
                    "slice_sha256": hashlib.sha256(payload).hexdigest(),
                    "checkpoint_record_ids": [
                        self.chunks[i].record_id for i in indices
                    ],
                }
            )
            return _MemoryResponse(payload, self.expectation, start, end)
        except Exception as error:
            self.failed = True
            self.cache.clear()
            # Do not allow the parser's outer range loop to multiply transport retries.
            raise ValueError(
                "authenticated structure range failed; no retry"
            ) from error

    def audit(self) -> dict[str, Any]:
        return {
            "network_attempt_count": len(self.network),
            "network_attempts": list(self.network),
            "parser_ranges": list(self.logical),
            "parser_adapter_responses_are_http_observations": False,
            "opaque_adjacent_bytes_may_be_buffered": True,
            "raw_bytes_persisted": False,
        }

    def close(self) -> None:
        self.cache.clear()
        self.closed = True


def _verify_acquisition(
    spec: acquisition.AcquisitionSpec,
    engine: RHashCheckpointEngine,
    expected_receipt_sha256: str,
) -> tuple[dict[str, Any], tuple[_Chunk, ...]]:
    """Caller holds the acquisition lock for verification and subsequent parsing."""
    path = spec.root / "acquisition-receipt.json"
    _require(
        _file_sha256(path) == expected_receipt_sha256, "acquisition receipt changed"
    )
    receipt = acquisition._read(path, "checkpointed-range-acquisition-receipt")
    _require(
        receipt == acquisition._completed_receipt(spec, engine),
        "acquisition receipt disagrees with its complete custody chain",
    )
    previous: str | None = None
    chunks: list[_Chunk] = []
    for index in range(spec.chunk_count):
        record = acquisition._read(
            spec.root / "chunks" / f"{index:06d}.json", "checkpoint-range"
        )
        _require(record["previous_id"] == previous, "verified checkpoint chain changed")
        chunks.append(
            _Chunk(
                record["start"],
                record["end"],
                record["range_sha256"],
                record["record_id"],
            )
        )
        previous = record["record_id"]
    _require(previous == receipt["chunk_tip"], "verified checkpoint tip changed")
    _require(
        _file_sha256(path) == expected_receipt_sha256, "receipt changed during audit"
    )
    return receipt, tuple(chunks)


def _load_authorization(
    path: Path, expected_sha256: str, spec: acquisition.AcquisitionSpec
) -> dict[str, Any]:
    sha256_digest(expected_sha256, name="structure authorization digest")
    _require(_file_sha256(path) == expected_sha256, "structure authorization changed")
    record = acquisition._read(path, "checkpoint-structure-authorization")
    acquisition._fields(
        record,
        "scope",
        "spec_id",
        "amendment_sha256",
        "acquisition_receipt_file_sha256",
        "acquisition_receipt_record_id",
        "archive_sha256",
        "implementation_revision",
        "implementation_files",
        "hostname",
        "output_root",
        "attempt_ledger_path",
        "not_before_utc",
        "attempt_limit",
        "central_directory_chunk_size_bytes",
        "maximum_central_directory_size_bytes",
        "boundaries",
    )
    _require(record["scope"] == "central-directory-only", "structure scope changed")
    _require(
        record["spec_id"] == spec.spec_id
        and record["amendment_sha256"] == spec.amendment_sha256,
        "structure acquisition specification changed",
    )
    for name in (
        "acquisition_receipt_file_sha256",
        "acquisition_receipt_record_id",
        "archive_sha256",
    ):
        sha256_digest(record[name], name=name)
    exact_revision(record["implementation_revision"], name="structure implementation")
    bindings = record["implementation_files"]
    _require(
        isinstance(bindings, dict) and set(bindings) == IMPLEMENTATION_FILES,
        "structure implementation roster changed",
    )
    for relative, expected in bindings.items():
        sha256_digest(expected, name=relative)
        bound_path = ROOT / relative
        _require(
            _file_sha256(bound_path) == expected, f"structure code changed: {relative}"
        )
        if not relative.startswith("src/"):
            continue
        name = relative[4:-3].replace("/", ".").removesuffix(".__init__")
        module = sys.modules.get(name)
        _require(
            module is not None and Path(str(module.__file__)).resolve() == bound_path,
            f"structure import did not use the bound tree: {name}",
        )
    _require(record["hostname"] == socket.gethostname(), "structure host changed")
    _require(
        record["output_root"] == str(spec.root.parent / "checkpoint-structure-v1")
        and record["attempt_ledger_path"]
        == str(spec.root.parent / "checkpoint-structure-attempt-v1.json"),
        "structure output or write-once ledger path changed",
    )
    for name in ("output_root", "attempt_ledger_path"):
        target = Path(record[name])
        _require(target.resolve() == target, "structure path is linked")
    _require(
        type(record["attempt_limit"]) is int and record["attempt_limit"] == 1,
        "structure attempt limit changed",
    )
    for name in (
        "central_directory_chunk_size_bytes",
        "maximum_central_directory_size_bytes",
    ):
        _require(
            type(record[name]) is int and record[name] > 0, "invalid structure bound"
        )
    _require(
        record["central_directory_chunk_size_bytes"]
        <= record["maximum_central_directory_size_bytes"],
        "central-directory request exceeds its bound",
    )
    _require(
        isinstance(record["boundaries"], dict)
        and set(record["boundaries"]) == set(BOUNDARIES)
        and all(value is False for value in record["boundaries"].values()),
        "structure scientific boundary changed",
    )
    _require(
        acquisition._utc(record["not_before_utc"])
        >= acquisition._utc(spec.first_request_not_before_utc),
        "structure provider cooldown was shortened",
    )
    return record


def run_checkpointed_structure(
    spec: acquisition.AcquisitionSpec,
    engine: RHashCheckpointEngine,
    authorization_path: Path,
    *,
    expected_authorization_sha256: str,
    opener: RangeOpener,
    clock: acquisition.Clock = acquisition._now,
) -> dict[str, Any]:
    """Run one separately authorized inventory; no default network opener or retry."""
    authorization = _load_authorization(
        authorization_path, expected_authorization_sha256, spec
    )
    _require(
        clock() >= acquisition._utc(authorization["not_before_utc"]),
        "structure provider cooldown has not elapsed",
    )
    output = Path(authorization["output_root"])
    ledger = Path(authorization["attempt_ledger_path"])
    with acquisition._locked(spec):
        _require(
            not output.exists() and not ledger.exists(), "structure attempt consumed"
        )
        receipt, chunks = _verify_acquisition(
            spec, engine, authorization["acquisition_receipt_file_sha256"]
        )
        _require(
            receipt["record_id"] == authorization["acquisition_receipt_record_id"]
            and receipt["archive_sha256"] == authorization["archive_sha256"],
            "structure authorization refers to a different acquisition",
        )
        _require(
            clock()
            >= acquisition._utc(
                acquisition._attempt_history(spec)[-1]["finished_at_utc"]
            ),
            "structure clock predates acquisition completion",
        )
        attempt = acquisition._seal(
            "checkpoint-structure-attempt",
            authorization_file_sha256=expected_authorization_sha256,
            authorization_record_id=authorization["record_id"],
            acquisition_receipt_record_id=receipt["record_id"],
            started_at_utc=clock().isoformat(),
            boundaries=dict(BOUNDARIES),
        )
        acquisition._write(ledger, attempt)
        maximum = authorization["maximum_central_directory_size_bytes"]
        size = spec.expectation.chunk_size_bytes
        reader = _AuthenticatedRanges(
            spec.expectation,
            chunks,
            opener=opener,
            maximum_cached_chunks=(maximum + size - 1) // size + 4,
        )
        names_opened = False
        try:
            output.mkdir(mode=0o700)
            layout = read_remote_zip_layout(spec.expectation, opener=reader)
            central = fetch_remote_central_directory(
                spec.expectation,
                layout,
                opener=reader,
                chunk_size_bytes=authorization["central_directory_chunk_size_bytes"],
                maximum_size_bytes=maximum,
            )
            names_opened = True
            members = parse_remote_central_directory(
                central, expected_entries=layout.entry_count
            )
            common = {
                "archive_sha256": receipt["archive_sha256"],
                "archive_size_bytes": spec.expectation.size_bytes,
                "acquisition_receipt_record_id": receipt["record_id"],
                "acquisition_receipt_file_sha256": authorization[
                    "acquisition_receipt_file_sha256"
                ],
                "structure_authorization_record_id": authorization["record_id"],
                "structure_authorization_file_sha256": expected_authorization_sha256,
                "attempt_record_id": attempt["record_id"],
                "boundaries": dict(BOUNDARIES),
            }
            private = acquisition._seal(
                "checkpoint-private-member-manifest",
                **common,
                members=sorted(
                    (member.as_record() for member in members),
                    key=lambda m: str(m["name"]),
                ),
            )
            private_path = output / "private-member-manifest.json"
            acquisition._write(private_path, private)
            result = acquisition._seal(
                "checkpoint-archive-structure-lock",
                **common,
                private_member_manifest_file_sha256=_file_sha256(private_path),
                structure=remote_zip_structure_summary(layout, central, members),
                range_audit=reader.audit(),
                zip_central_directory_parsed=True,
                archive_member_names_opened=True,
                archive_bytes_retained=False,
                source_and_confirmation_authorized=False,
            )
            acquisition._write(output / "structure-lock.json", result)
            terminal = acquisition._seal(
                "checkpoint-structure-terminal",
                attempt_record_id=attempt["record_id"],
                status="complete",
                result_record_id=result["record_id"],
                ended_at_utc=clock().isoformat(),
                archive_member_names_opened=True,
                boundaries=dict(BOUNDARIES),
            )
            acquisition._write(output / "terminal.json", terminal)
            return result
        except BaseException as error:
            if output.is_dir():
                acquisition._write(
                    output / "terminal.json",
                    acquisition._seal(
                        "checkpoint-structure-terminal",
                        attempt_record_id=attempt["record_id"],
                        status="failed-preserved",
                        failure_type=type(error).__name__,
                        ended_at_utc=clock().isoformat(),
                        archive_member_names_opened=names_opened,
                        range_audit=reader.audit(),
                        boundaries=dict(BOUNDARIES),
                    ),
                )
            raise
        finally:
            reader.close()


def verify_checkpointed_structure(
    spec: acquisition.AcquisitionSpec,
    engine: RHashCheckpointEngine,
    authorization_path: Path,
    *,
    expected_authorization_sha256: str,
    expected_result_sha256: str,
) -> dict[str, Any]:
    """Verify saved publication custody offline, not unavailable raw ZIP payloads."""
    sha256_digest(expected_result_sha256, name="structure result digest")
    authorization = _load_authorization(
        authorization_path, expected_authorization_sha256, spec
    )
    output = Path(authorization["output_root"])
    with acquisition._locked(spec):
        receipt, _ = _verify_acquisition(
            spec, engine, authorization["acquisition_receipt_file_sha256"]
        )
        _require(
            receipt["record_id"] == authorization["acquisition_receipt_record_id"]
            and receipt["archive_sha256"] == authorization["archive_sha256"],
            "structure authorization refers to a different acquisition",
        )
        _require(
            output.is_dir()
            and {p.name for p in output.iterdir()}
            == {"structure-lock.json", "private-member-manifest.json", "terminal.json"}
            and all(p.is_file() and p.resolve() == p for p in output.iterdir()),
            "structure publication is incomplete or contains unregistered files",
        )
        _require(
            _file_sha256(output / "structure-lock.json") == expected_result_sha256,
            "published structure result changed",
        )
        attempt = acquisition._read(
            Path(authorization["attempt_ledger_path"]), "checkpoint-structure-attempt"
        )
        acquisition._fields(
            attempt,
            "authorization_file_sha256",
            "authorization_record_id",
            "acquisition_receipt_record_id",
            "started_at_utc",
            "boundaries",
        )
        _require(
            attempt["authorization_file_sha256"] == expected_authorization_sha256
            and attempt["authorization_record_id"] == authorization["record_id"]
            and attempt["acquisition_receipt_record_id"] == receipt["record_id"],
            "structure attempt parents changed",
        )
        result = acquisition._read(
            output / "structure-lock.json", "checkpoint-archive-structure-lock"
        )
        private = acquisition._read(
            output / "private-member-manifest.json",
            "checkpoint-private-member-manifest",
        )
        common = {
            "archive_sha256": receipt["archive_sha256"],
            "archive_size_bytes": spec.expectation.size_bytes,
            "acquisition_receipt_record_id": receipt["record_id"],
            "acquisition_receipt_file_sha256": authorization[
                "acquisition_receipt_file_sha256"
            ],
            "structure_authorization_record_id": authorization["record_id"],
            "structure_authorization_file_sha256": expected_authorization_sha256,
            "attempt_record_id": attempt["record_id"],
            "boundaries": dict(BOUNDARIES),
        }
        acquisition._fields(
            result,
            *common,
            "private_member_manifest_file_sha256",
            "structure",
            "range_audit",
            "zip_central_directory_parsed",
            "archive_member_names_opened",
            "archive_bytes_retained",
            "source_and_confirmation_authorized",
        )
        acquisition._fields(private, *common, "members")
        for record in (result, private):
            _require(
                all(record[name] == value for name, value in common.items()),
                "structure publication parents changed",
            )
        for name, expected in {
            "zip_central_directory_parsed": True,
            "archive_member_names_opened": True,
            "archive_bytes_retained": False,
            "source_and_confirmation_authorized": False,
        }.items():
            _require(
                result[name] is expected, f"structure result boundary changed: {name}"
            )
        _require(
            _file_sha256(output / "private-member-manifest.json")
            == result["private_member_manifest_file_sha256"],
            "private manifest binding changed",
        )
        terminal = acquisition._read(
            output / "terminal.json", "checkpoint-structure-terminal"
        )
        acquisition._fields(
            terminal,
            "attempt_record_id",
            "status",
            "result_record_id",
            "ended_at_utc",
            "archive_member_names_opened",
            "boundaries",
        )
        _require(
            terminal["status"] == "complete"
            and terminal["attempt_record_id"] == attempt["record_id"]
            and terminal["result_record_id"] == result["record_id"]
            and terminal["archive_member_names_opened"] is True,
            "structure publication has no matching complete terminal",
        )
        for record in (attempt, result, private, terminal):
            _require(
                isinstance(record["boundaries"], dict)
                and set(record["boundaries"]) == set(BOUNDARIES)
                and all(value is False for value in record["boundaries"].values()),
                "published scientific boundary changed",
            )
        _require(
            acquisition._utc(terminal["ended_at_utc"])
            >= acquisition._utc(attempt["started_at_utc"])
            >= max(
                acquisition._utc(authorization["not_before_utc"]),
                acquisition._utc(
                    acquisition._attempt_history(spec)[-1]["finished_at_utc"]
                ),
            ),
            "structure publication timestamp ordering changed",
        )
        return result
