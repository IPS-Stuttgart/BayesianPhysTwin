"""Durable opaque range hashing with explicit attempts and no scientific access."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast
from urllib.parse import urlsplit

from bayesian_phystwin._portable_contracts import (
    content_id,
    load_strict_json_object,
    require_exact_fields,
    sha256_digest,
    write_atomic_json,
)

from .poseit_hash_checkpoint import (
    BLOCK_SIZE,
    CheckpointedSha256,
    RHashCheckpointEngine,
)
from .poseit_remote_archive import (
    RangeOpener,
    RangeResponse,
    RemoteArchiveExpectation,
    RemoteRangeBlock,
    fetch_exact_range,
)

Clock = Callable[[], datetime]
_BOUNDARIES = {
    "archive_bytes_retained": False,
    "archive_member_names_opened": False,
    "member_payload_bytes_opened": False,
    "phase_labels_opened": False,
    "sensor_payloads_opened": False,
    "shake_outcomes_opened": False,
    "object_roles_assigned": False,
    "confirmation_opened": False,
    "held_v8_accessed": False,
    "structure_access_authorized": False,
    "scientific_result": False,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _boundaries(value: object) -> None:
    _require(
        isinstance(value, Mapping)
        and set(value) == set(_BOUNDARIES)
        and all(item is False for item in value.values()),
        "scientific access boundary changed",
    )


def _utc(value: str) -> datetime:
    _require(isinstance(value, str), "timestamp must be a UTC string")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    _require(result.utcoffset() == timedelta(0), "timestamp must be UTC")
    return result


def _seal(kind: str, **fields: Any) -> dict[str, Any]:
    identity = {
        "schema": f"bayesian-phystwin.poseit-{kind}",
        "schema_version": 1,
        **fields,
    }
    return {**identity, "record_id": content_id(identity)}


def _read(path: Path, kind: str) -> dict[str, Any]:
    _require(
        path.is_file() and not path.is_symlink(),
        f"missing or linked custody file: {path.name}",
    )
    payload = dict(load_strict_json_object(path, label=kind))
    identity = dict(payload)
    record_id = identity.pop("record_id", None)
    _require(record_id == content_id(identity), f"{kind} content binding changed")
    _require(
        payload.get("schema") == f"bayesian-phystwin.poseit-{kind}",
        f"{kind} schema changed",
    )
    _require(
        type(payload.get("schema_version")) is int and payload["schema_version"] == 1,
        "schema version changed",
    )
    return payload


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    write_atomic_json(payload, path, overwrite=False)


def _fields(record: Mapping[str, Any], *names: str) -> None:
    require_exact_fields(
        record,
        name=str(record.get("schema", "custody record")),
        expected=frozenset({"schema", "schema_version", "record_id", *names}),
    )


@dataclass(frozen=True)
class AcquisitionSpec:
    root: Path
    lock_path: Path
    expectation: RemoteArchiveExpectation
    amendment_sha256: str
    library_sha256: str
    parent_sha256: Mapping[str, str]
    first_request_not_before_utc: str
    resume_delay_seconds: int = 86400

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "parent_sha256", MappingProxyType(dict(self.parent_sha256))
        )
        for path in (self.root, self.lock_path):
            _require(
                path.is_absolute() and path.resolve() == path,
                "custody paths must be absolute and unlinked",
            )
        _require(
            self.lock_path != self.root and self.root not in self.lock_path.parents,
            "lock must be outside the output root",
        )
        _require(
            self.expectation.chunk_size_bytes % BLOCK_SIZE == 0,
            "range size must be SHA-block aligned",
        )
        _require(
            type(self.resume_delay_seconds) is int
            and self.resume_delay_seconds >= 86400,
            "resume cooldown is too short",
        )
        _utc(self.first_request_not_before_utc)
        for name, value in {
            "amendment": self.amendment_sha256,
            "library": self.library_sha256,
            **self.parent_sha256,
        }.items():
            sha256_digest(value, name=name)

    @property
    def chunk_count(self) -> int:
        return (
            self.expectation.size_bytes + self.expectation.chunk_size_bytes - 1
        ) // self.expectation.chunk_size_bytes

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "lock_path": str(self.lock_path),
            "source": asdict(self.expectation),
            "amendment_sha256": self.amendment_sha256,
            "library_sha256": self.library_sha256,
            "parent_sha256": dict(self.parent_sha256),
            "first_request_not_before_utc": self.first_request_not_before_utc,
            "resume_delay_seconds": self.resume_delay_seconds,
        }

    @property
    def spec_id(self) -> str:
        return content_id(self.identity)


@contextmanager
def _locked(spec: AcquisitionSpec) -> Iterator[None]:
    _require(spec.lock_path.resolve() == spec.lock_path, "lock path became linked")
    descriptor = os.open(spec.lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        _require(
            stat.S_ISREG(os.fstat(descriptor).st_mode), "lock is not a regular file"
        )
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _require(spec.root.resolve() == spec.root, "output root became linked")
        if spec.root.exists():
            _require(spec.root.is_dir(), "output root is not a directory")
            _require(
                all(not item.is_symlink() for item in spec.root.iterdir()),
                "output root contains a linked custody entry",
            )
            _require(
                {item.name for item in spec.root.iterdir()}
                <= {"attempts", "chunks", "acquisition-receipt.json"},
                "unregistered file in output root",
            )
        yield
    finally:
        os.close(descriptor)


def _numbered(directory: Path, *, suffix: str) -> list[Path]:
    _require(not directory.is_symlink(), "custody directory is linked")
    if not directory.exists():
        return []
    _require(
        directory.is_dir() and not directory.is_symlink(), "custody directory is linked"
    )
    entries = sorted(directory.iterdir())
    for index, entry in enumerate(entries):
        _require(
            entry.name == f"{index:06d}{suffix}" and not entry.is_symlink(),
            "custody sequence is missing, reordered, linked, or has an orphan",
        )
    return entries


def _attempt_history(spec: AcquisitionSpec) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    previous_id: str | None = None
    previous_tip: str | None = None
    previous_count = 0
    for index, directory in enumerate(_numbered(spec.root / "attempts", suffix="")):
        start = _read(directory / "start.json", "checkpoint-attempt-start")
        terminal = _read(directory / "terminal.json", "checkpoint-attempt-terminal")
        _fields(
            start,
            "spec_id",
            "attempt",
            "authorization_id",
            "authorization_file_sha256",
            "previous_terminal_id",
            "start_count",
            "start_tip",
            "started_at_utc",
            "pid",
        )
        _fields(
            terminal,
            "spec_id",
            "attempt",
            "start_id",
            "end_count",
            "end_tip",
            "status",
            "finished_at_utc",
            "resume_not_before_utc",
            "error",
            "invalid_response_ids",
            "uncommitted_transport_attempt_count",
            "boundaries",
        )
        for record, fields in (
            (start, ("attempt", "start_count", "pid")),
            (terminal, ("attempt", "end_count")),
        ):
            _require(
                all(type(record[field]) is int for field in fields),
                "attempt counters must be integers",
            )
        _require(start["pid"] > 0, "process identity is missing")
        _require(
            terminal["uncommitted_transport_attempt_count"] is None,
            "uncommitted transport attempts cannot be reconstructed",
        )
        authorization = _load_authorization(
            directory / "authorization.json", start["authorization_file_sha256"], spec
        )
        _require(
            start["spec_id"]
            == terminal["spec_id"]
            == authorization["spec_id"]
            == spec.spec_id,
            "attempt specification changed",
        )
        _require(
            start["attempt"]
            == terminal["attempt"]
            == authorization["attempt"]
            == index,
            "attempt number changed",
        )
        _require(
            start["authorization_id"] == authorization["record_id"],
            "attempt authorization changed",
        )
        _require(
            start["previous_terminal_id"]
            == authorization["previous_terminal_id"]
            == previous_id,
            "attempt history changed",
        )
        _require(
            start["start_tip"] == authorization["start_tip"] == previous_tip,
            "attempt start prefix changed",
        )
        _require(
            start["start_count"] == authorization["start_count"] == previous_count,
            "attempt start count changed",
        )
        _require(
            terminal["start_id"] == start["record_id"], "attempt start binding changed"
        )
        _require(
            terminal["status"] in {"failed-preserved", "complete"},
            "attempt is not terminal",
        )
        _require(
            type(terminal["end_count"]) is int
            and previous_count <= terminal["end_count"] <= spec.chunk_count,
            "attempt end count changed",
        )
        _boundaries(terminal["boundaries"])
        started = _utc(start["started_at_utc"])
        finished = _utc(terminal["finished_at_utc"])
        _require(
            started
            >= max(
                _utc(authorization["not_before_utc"]),
                _utc(spec.first_request_not_before_utc),
            ),
            "attempt started before its cooldown",
        )
        _require(finished >= started, "terminal timestamp precedes start")
        if terminal["status"] == "failed-preserved":
            _require(
                _utc(terminal["resume_not_before_utc"])
                >= finished + timedelta(seconds=spec.resume_delay_seconds),
                "terminal cooldown was shortened",
            )
            _require(
                isinstance(terminal["error"], Mapping), "failure identity is missing"
            )
            _require(
                set(terminal["error"]) == {"type", "message"}
                and all(isinstance(value, str) for value in terminal["error"].values()),
                "failure identity fields changed",
            )
        else:
            _require(
                terminal["end_count"] == spec.chunk_count and terminal["error"] is None,
                "terminal completion is incomplete",
            )
            _require(
                terminal["resume_not_before_utc"] is None,
                "complete attempt cannot authorize resumption",
            )
            _require(
                not terminal["invalid_response_ids"],
                "completion has a rejected response",
            )
        _require(
            isinstance(terminal["invalid_response_ids"], list)
            and all(
                isinstance(value, str) for value in terminal["invalid_response_ids"]
            )
            and len(set(terminal["invalid_response_ids"]))
            == len(terminal["invalid_response_ids"]),
            "failure response list changed",
        )
        response_names = [
            f"response-{i:03d}.json"
            for i in range(len(terminal["invalid_response_ids"]))
        ]
        _require(
            {item.name for item in directory.iterdir()}
            == {"start.json", "terminal.json", "authorization.json", *response_names},
            "attempt evidence is missing or has an orphan",
        )
        for name, expected_id in zip(
            response_names, terminal["invalid_response_ids"], strict=True
        ):
            response = _read(directory / name, "checkpoint-invalid-response")
            _fields(
                response,
                "request_range",
                "http_status",
                "final_host",
                "content_type",
                "content_length",
                "content_range",
                "observed_at_utc",
                "response_body_read",
            )
            _require(
                response["record_id"] == expected_id
                and response["response_body_read"] is False,
                "failure response observation changed",
            )
            _require(
                type(response["http_status"]) is int and response["http_status"] != 206,
                "failure response status changed",
            )
            _require(
                started <= _utc(response["observed_at_utc"]) <= finished,
                "failure response timestamp is outside the attempt",
            )
        if history:
            _require(
                history[-1]["status"] == "failed-preserved",
                "attempt follows a completed acquisition",
            )
            _require(
                min(started, _utc(authorization["not_before_utc"]))
                >= _utc(history[-1]["resume_not_before_utc"]),
                "resumption preceded failure cooldown",
            )
        history.append(terminal)
        previous_id, previous_tip, previous_count = (
            terminal["record_id"],
            terminal["end_tip"],
            terminal["end_count"],
        )
    return history


def _prefix(
    spec: AcquisitionSpec,
    engine: RHashCheckpointEngine,
    history: list[dict[str, Any]],
    *,
    expected_count: int,
    expected_tip: str | None,
) -> list[dict[str, Any]]:
    paths = _numbered(spec.root / "chunks", suffix=".json")
    _require(len(paths) == expected_count, "checkpoint rollback or unadvertised suffix")
    records: list[dict[str, Any]] = []
    previous_id: str | None = None
    for index, path in enumerate(paths):
        record = _read(path, "checkpoint-range")
        require_exact_fields(
            record,
            name="checkpoint range",
            expected=frozenset(
                {
                    "schema",
                    "schema_version",
                    "record_id",
                    "spec_id",
                    "index",
                    "start",
                    "end",
                    "attempt",
                    "previous_id",
                    "range_sha256",
                    "range_attempts",
                    "checkpoint",
                    "final",
                    "archive_sha256",
                    "boundaries",
                }
            ),
        )
        start = index * spec.expectation.chunk_size_bytes
        end = (
            min(start + spec.expectation.chunk_size_bytes, spec.expectation.size_bytes)
            - 1
        )
        _require(
            all(type(record[name]) is int for name in ("index", "start", "end")),
            "checkpoint range counters must be integers",
        )
        _require(
            record["spec_id"] == spec.spec_id and record["index"] == index,
            "checkpoint specification or index changed",
        )
        _require(record["previous_id"] == previous_id, "checkpoint chain changed")
        _require(
            record["start"] == start and record["end"] == end,
            "checkpoint range changed",
        )
        attempt = record["attempt"]
        _require(
            type(attempt) is int and 0 <= attempt < len(history),
            "checkpoint attempt is missing",
        )
        lower = history[attempt - 1]["end_count"] if attempt else 0
        _require(
            lower <= index < history[attempt]["end_count"],
            "checkpoint has the wrong attempt owner",
        )
        sha256_digest(record["range_sha256"], name="range digest")
        _require(
            type(record["range_attempts"]) is int
            and 1
            <= record["range_attempts"]
            <= spec.expectation.max_attempts_per_range,
            "range attempt count changed",
        )
        final = index == spec.chunk_count - 1
        _require(record["final"] is final, "checkpoint final flag changed")
        _boundaries(record["boundaries"])
        checkpoint = record["checkpoint"]
        with engine.restore(
            checkpoint, expected_checkpoint_id=checkpoint["checkpoint_id"]
        ) as state:
            _require(
                state.bytes_hashed == (end + 1) - (end + 1) % BLOCK_SIZE,
                "checkpoint byte counter changed",
            )
            if final:
                sha256_digest(record["archive_sha256"], name="archive digest")
                if (end + 1) % BLOCK_SIZE == 0:
                    _require(
                        record["archive_sha256"] == state.hexdigest(),
                        "final aligned digest changed",
                    )
            else:
                _require(
                    record["archive_sha256"] is None,
                    "partial checkpoint claims a full hash",
                )
        records.append(record)
        previous_id = record["record_id"]
    _require(previous_id == expected_tip, "authorized checkpoint tip changed")
    for terminal in history:
        end_count = terminal["end_count"]
        tip = records[end_count - 1]["record_id"] if end_count else None
        _require(terminal["end_tip"] == tip, "terminal checkpoint binding changed")
    return records


def _load_authorization(
    path: Path, expected_sha256: str, spec: AcquisitionSpec
) -> dict[str, Any]:
    sha256_digest(expected_sha256, name="authorization digest")
    _require(
        path.is_file() and path.resolve() == path.absolute() and not path.is_symlink(),
        "authorization path is linked or missing",
    )
    _require(
        hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256,
        "authorization file changed",
    )
    authorization = _read(path, "checkpoint-attempt-authorization")
    _require(
        path.read_bytes()
        == (
            json.dumps(authorization, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode(),
        "authorization must use canonical pretty JSON",
    )
    require_exact_fields(
        authorization,
        name="attempt authorization",
        expected=frozenset(
            {
                "schema",
                "schema_version",
                "record_id",
                "spec_id",
                "attempt",
                "previous_terminal_id",
                "start_tip",
                "start_count",
                "not_before_utc",
                "attempt_limit",
                "boundaries",
            }
        ),
    )
    _require(
        authorization["spec_id"] == spec.spec_id, "authorization specification changed"
    )
    _require(
        type(authorization["attempt"]) is int and authorization["attempt"] >= 0,
        "invalid attempt number",
    )
    _require(
        type(authorization["start_count"]) is int
        and 0 <= authorization["start_count"] <= spec.chunk_count,
        "invalid authorized prefix",
    )
    _require(
        type(authorization["attempt_limit"]) is int
        and authorization["attempt_limit"] == 1,
        "attempt limit changed",
    )
    _boundaries(authorization["boundaries"])
    _utc(authorization["not_before_utc"])
    return authorization


def _default_open(request: urllib.request.Request, timeout: float) -> RangeResponse:
    return cast(RangeResponse, urllib.request.urlopen(request, timeout=timeout))


def _completed_receipt(
    spec: AcquisitionSpec,
    engine: RHashCheckpointEngine,
) -> dict[str, Any]:
    _require(
        engine.library_sha256 == spec.library_sha256, "native library binding changed"
    )
    history = _attempt_history(spec)
    _require(
        bool(history) and history[-1]["status"] == "complete",
        "acquisition has no complete terminal record",
    )
    _require(history[-1]["end_count"] == spec.chunk_count, "full archive is not hashed")
    records = _prefix(
        spec,
        engine,
        history,
        expected_count=spec.chunk_count,
        expected_tip=history[-1]["end_tip"],
    )
    return _seal(
        "checkpointed-range-acquisition-receipt",
        spec=spec.identity,
        spec_id=spec.spec_id,
        archive_sha256=records[-1]["archive_sha256"],
        archive_size_bytes=spec.expectation.size_bytes,
        chunk_count=len(records),
        chunk_tip=records[-1]["record_id"],
        terminal_tip=history[-1]["record_id"],
        attempt_count=len(history),
        transport_attempts_for_committed_chunks=sum(
            item["range_attempts"] for item in records
        ),
        uncommitted_transport_attempt_count=None,
        archive_bytes_streamed_opaquely=True,
        boundaries=dict(_BOUNDARIES),
    )


def publish_completed_receipt(
    spec: AcquisitionSpec, engine: RHashCheckpointEngine
) -> dict[str, Any]:
    """Finish a crashed publication from terminal custody only; never contact a source."""
    with _locked(spec):
        receipt = _completed_receipt(spec, engine)
        _write(spec.root / "acquisition-receipt.json", receipt)
        return receipt


def verify_completed_receipt(
    spec: AcquisitionSpec, engine: RHashCheckpointEngine
) -> dict[str, Any]:
    with _locked(spec):
        receipt = _read(
            spec.root / "acquisition-receipt.json",
            "checkpointed-range-acquisition-receipt",
        )
        _require(
            receipt == _completed_receipt(spec, engine),
            "completion receipt disagrees with its custody chain",
        )
        return receipt


def run_checkpointed_attempt(
    spec: AcquisitionSpec,
    engine: RHashCheckpointEngine,
    authorization_path: Path,
    *,
    expected_authorization_sha256: str,
    opener: RangeOpener = _default_open,
    clock: Clock = _now,
) -> dict[str, Any]:
    """Run one authorized attempt; preserve any failure and never auto-restart."""
    authorization = _load_authorization(
        authorization_path, expected_authorization_sha256, spec
    )
    _require(
        engine.library_sha256 == spec.library_sha256, "native library binding changed"
    )
    not_before = max(
        _utc(authorization["not_before_utc"]), _utc(spec.first_request_not_before_utc)
    )
    _require(clock() >= not_before, "provider cooldown has not elapsed")
    with _locked(spec):
        _require(
            not (spec.root / "acquisition-receipt.json").exists(),
            "completion receipt already exists",
        )
        history = _attempt_history(spec)
        attempt = len(history)
        _require(
            authorization["attempt"] == attempt, "attempt already consumed or not next"
        )
        previous = history[-1] if history else None
        _require(
            previous is None or previous["status"] == "failed-preserved",
            "previous acquisition is complete",
        )
        _require(
            authorization["previous_terminal_id"]
            == (previous["record_id"] if previous else None),
            "previous terminal binding changed",
        )
        start_count = previous["end_count"] if previous else 0
        start_tip = previous["end_tip"] if previous else None
        _require(
            authorization["start_count"] == start_count
            and authorization["start_tip"] == start_tip,
            "authorized prefix changed",
        )
        if previous:
            resume_at = _utc(previous["resume_not_before_utc"])
            _require(
                _utc(authorization["not_before_utc"]) >= resume_at
                and clock() >= resume_at,
                "failure cooldown has not elapsed",
            )
        prefix = _prefix(
            spec, engine, history, expected_count=start_count, expected_tip=start_tip
        )
        directory = spec.root / "attempts" / f"{attempt:06d}"
        directory.parent.mkdir(parents=True, exist_ok=True)
        directory.mkdir(mode=0o700)
        _write(directory / "authorization.json", authorization)
        start_record = _seal(
            "checkpoint-attempt-start",
            spec_id=spec.spec_id,
            attempt=attempt,
            authorization_id=authorization["record_id"],
            authorization_file_sha256=expected_authorization_sha256,
            previous_terminal_id=authorization["previous_terminal_id"],
            start_count=start_count,
            start_tip=start_tip,
            started_at_utc=clock().isoformat(),
            pid=os.getpid(),
        )
        _write(directory / "start.json", start_record)
        state: CheckpointedSha256 | None = None
        stop = threading.Event()
        response_lock = threading.Lock()
        invalid_responses: list[dict[str, Any]] = []

        def audited_open(
            request: urllib.request.Request, timeout: float
        ) -> RangeResponse:
            _require(
                not stop.is_set(), "attempt stopped; no additional request admitted"
            )
            try:
                response = opener(request, timeout)
            except urllib.error.HTTPError as error:
                # HTTP rejection is an identity failure, not a socket retry.
                response = cast(RangeResponse, error)
            if response.status != 206:
                stop.set()
                with response_lock:
                    observation = _seal(
                        "checkpoint-invalid-response",
                        request_range=request.get_header("Range"),
                        http_status=response.status,
                        final_host=urlsplit(response.geturl()).hostname,
                        content_type=response.headers.get("Content-Type"),
                        content_length=response.headers.get("Content-Length"),
                        content_range=response.headers.get("Content-Range"),
                        observed_at_utc=clock().isoformat(),
                        response_body_read=False,
                    )
                    try:
                        _write(
                            directory / f"response-{len(invalid_responses):03d}.json",
                            observation,
                        )
                    except BaseException:
                        response.close()
                        raise
                    invalid_responses.append(observation)
            return response

        completed = start_count
        tip = start_tip
        futures: dict[int, Future[RemoteRangeBlock]] = {}
        pool: ThreadPoolExecutor | None = None

        def fetch(index: int, start: int, end: int) -> RemoteRangeBlock:
            try:
                return fetch_exact_range(
                    spec.expectation,
                    index=index,
                    start=start,
                    end=end,
                    opener=audited_open,
                )
            except BaseException:
                stop.set()
                raise

        def submit(index: int) -> None:
            start = index * spec.expectation.chunk_size_bytes
            end = (
                min(
                    start + spec.expectation.chunk_size_bytes,
                    spec.expectation.size_bytes,
                )
                - 1
            )
            assert pool is not None
            futures[index] = pool.submit(fetch, index, start, end)

        try:
            state = (
                engine.restore(
                    prefix[-1]["checkpoint"],
                    expected_checkpoint_id=prefix[-1]["checkpoint"]["checkpoint_id"],
                )
                if prefix
                else engine.new()
            )
            pool = ThreadPoolExecutor(max_workers=spec.expectation.max_workers)
            next_index = start_count
            while next_index < min(
                start_count + spec.expectation.max_workers, spec.chunk_count
            ):
                submit(next_index)
                next_index += 1
            for index in range(start_count, spec.chunk_count):
                block = futures.pop(index).result()
                _require(block.index == index, "range order changed")
                aligned = len(block.data) - len(block.data) % BLOCK_SIZE
                state.update_blocks(block.data[:aligned])
                final = index == spec.chunk_count - 1
                record = _seal(
                    "checkpoint-range",
                    spec_id=spec.spec_id,
                    index=index,
                    start=block.start,
                    end=block.end,
                    attempt=attempt,
                    previous_id=tip,
                    range_sha256=hashlib.sha256(block.data).hexdigest(),
                    range_attempts=block.attempts,
                    checkpoint=state.checkpoint(),
                    final=final,
                    archive_sha256=state.hexdigest(block.data[aligned:])
                    if final
                    else None,
                    boundaries=dict(_BOUNDARIES),
                )
                _write(spec.root / "chunks" / f"{index:06d}.json", record)
                tip, completed = record["record_id"], index + 1
                if next_index < spec.chunk_count:
                    submit(next_index)
                    next_index += 1
        except BaseException as error:
            stop.set()
            for future in futures.values():
                future.cancel()
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
            finished = clock()
            _write(
                directory / "terminal.json",
                _seal(
                    "checkpoint-attempt-terminal",
                    spec_id=spec.spec_id,
                    attempt=attempt,
                    start_id=start_record["record_id"],
                    end_count=completed,
                    end_tip=tip,
                    status="failed-preserved",
                    finished_at_utc=finished.isoformat(),
                    resume_not_before_utc=(
                        finished + timedelta(seconds=spec.resume_delay_seconds)
                    ).isoformat(),
                    error={"type": type(error).__name__, "message": str(error)},
                    invalid_response_ids=[
                        item["record_id"] for item in invalid_responses
                    ],
                    uncommitted_transport_attempt_count=None,
                    boundaries=dict(_BOUNDARIES),
                ),
            )
            raise
        finally:
            stop.set()
            for future in futures.values():
                future.cancel()
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
            if state is not None:
                state.close()
        _write(
            directory / "terminal.json",
            _seal(
                "checkpoint-attempt-terminal",
                spec_id=spec.spec_id,
                attempt=attempt,
                start_id=start_record["record_id"],
                end_count=completed,
                end_tip=tip,
                status="complete",
                finished_at_utc=clock().isoformat(),
                resume_not_before_utc=None,
                error=None,
                invalid_response_ids=[item["record_id"] for item in invalid_responses],
                uncommitted_transport_attempt_count=None,
                boundaries=dict(_BOUNDARIES),
            ),
        )
        receipt = _completed_receipt(spec, engine)
        _write(spec.root / "acquisition-receipt.json", receipt)
        return receipt
