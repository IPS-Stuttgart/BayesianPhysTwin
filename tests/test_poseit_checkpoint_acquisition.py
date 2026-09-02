from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
import threading
import urllib.error
import urllib.request
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from email.message import Message
from pathlib import Path
from typing import Any

import pytest

from bayesian_phystwin._portable_contracts import content_id, write_atomic_json
from bayesian_phystwin_experiments import poseit_checkpoint_acquisition as acquisition
from bayesian_phystwin_experiments.poseit_hash_checkpoint import RHashCheckpointEngine
from bayesian_phystwin_experiments.poseit_remote_archive import RemoteArchiveExpectation

NOW = datetime(2026, 9, 4, tzinfo=timezone.utc)
LATER = NOW + timedelta(days=1)


@pytest.fixture(scope="module")
def engine() -> RHashCheckpointEngine:
    configured = os.environ.get("POSEIT_TEST_RHASH_LIBRARY")
    if not configured:
        pytest.skip("requires an explicitly built local RHash v1.4.6 test library")
    library = Path(configured)
    return RHashCheckpointEngine(
        library,
        expected_library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
    )


def _spec(
    root: Path, engine: RHashCheckpointEngine, size: int = 321, workers: int = 1
) -> acquisition.AcquisitionSpec:
    return acquisition.AcquisitionSpec(
        root=root / "custody",
        lock_path=root / "custody.lock",
        expectation=RemoteArchiveExpectation(
            source_url="https://drive.usercontent.google.com/download?id=synthetic-only",
            file_name="synthetic.zip",
            size_bytes=size,
            last_modified="Thu, 01 Jan 1970 00:00:00 GMT",
            chunk_size_bytes=64,
            max_workers=workers,
            max_attempts_per_range=3,
            timeout_seconds=1,
        ),
        amendment_sha256="a" * 64,
        library_sha256=engine.library_sha256,
        parent_sha256={"synthetic_parent": "b" * 64},
        first_request_not_before_utc=NOW.isoformat(),
    )


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _authorization(
    spec: acquisition.AcquisitionSpec,
    *,
    previous: dict[str, Any] | None = None,
    not_before: datetime = NOW,
    **changes: Any,
) -> tuple[Path, str]:
    identity = {
        "spec_id": spec.spec_id,
        "attempt": previous["attempt"] + 1 if previous else 0,
        "previous_terminal_id": previous["record_id"] if previous else None,
        "start_tip": previous["end_tip"] if previous else None,
        "start_count": previous["end_count"] if previous else 0,
        "not_before_utc": not_before.isoformat(),
        "attempt_limit": 1,
        "boundaries": dict(acquisition._BOUNDARIES),
        **changes,
    }
    record = acquisition._seal("checkpoint-attempt-authorization", **identity)
    path = spec.root.parent / f"authorization-{record['record_id']}.json"
    write_atomic_json(record, path, overwrite=False)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


class Response(io.BytesIO):
    def __init__(
        self,
        data: bytes,
        spec: acquisition.AcquisitionSpec,
        start: int,
        end: int,
        *,
        status: int = 206,
        header_changes: dict[str, str] | None = None,
    ) -> None:
        super().__init__(data)
        self.status = status
        self.headers = {
            "Content-Type": "application/zip",
            "Content-Disposition": f'attachment; filename="{spec.expectation.file_name}"',
            "Accept-Ranges": "bytes",
            "Last-Modified": spec.expectation.last_modified,
            "Content-Range": f"bytes {start}-{end}/{spec.expectation.size_bytes}",
            "Content-Length": str(len(data)),
            **(header_changes or {}),
        }
        self.read_calls = 0

    def geturl(self) -> str:
        return "https://drive.usercontent.google.com/synthetic-only"

    def read(self, size: int = -1) -> bytes:
        self.read_calls += 1
        return super().read(size)


class Source:
    def __init__(
        self,
        spec: acquisition.AcquisitionSpec,
        *,
        fail_index: int | None = None,
        header_changes: dict[str, str] | None = None,
        out_of_order: bool = False,
        socket_failures: int = 0,
    ) -> None:
        self.spec = spec
        self.data = bytes(
            (index * 17 + 3) % 256 for index in range(spec.expectation.size_bytes)
        )
        self.fail_index = fail_index
        self.header_changes = header_changes
        self.out_of_order = out_of_order
        self.socket_failures = socket_failures
        self.calls: list[tuple[int, int]] = []
        self.responses: list[Response] = []
        self.second_ready = threading.Event()
        self.lock = threading.Lock()

    def __call__(self, request: urllib.request.Request, timeout: float) -> Response:
        assert timeout == self.spec.expectation.timeout_seconds
        header = request.get_header("Range")
        assert header is not None and header.startswith("bytes=")
        start, end = map(int, header[6:].split("-"))
        index = start // self.spec.expectation.chunk_size_bytes
        with self.lock:
            self.calls.append((start, end))
            if self.socket_failures:
                self.socket_failures -= 1
                raise OSError("synthetic socket failure")
        if self.out_of_order:
            if index == 0:
                assert self.second_ready.wait(5)
            elif index == 1:
                self.second_ready.set()
        response = Response(
            self.data[start : end + 1],
            self.spec,
            start,
            end,
            status=200 if index == self.fail_index else 206,
            header_changes=self.header_changes,
        )
        with self.lock:
            self.responses.append(response)
        return response


def _run(
    spec: acquisition.AcquisitionSpec,
    engine: RHashCheckpointEngine,
    source: Source,
    *,
    auth: tuple[Path, str] | None = None,
    now: datetime = NOW,
) -> dict[str, Any]:
    path, digest = auth or _authorization(spec)
    return acquisition.run_checkpointed_attempt(
        spec,
        engine,
        path,
        expected_authorization_sha256=digest,
        opener=source,
        clock=lambda: now,
    )


def _failed_prefix(
    root: Path,
    engine: RHashCheckpointEngine,
    *,
    fail_index: int = 3,
) -> tuple[acquisition.AcquisitionSpec, dict[str, Any], tuple[Path, str]]:
    spec = _spec(root, engine)
    auth = _authorization(spec)
    source = Source(spec, fail_index=fail_index)
    with pytest.raises(ValueError, match="HTTP 206"):
        _run(spec, engine, source, auth=auth)
    terminal = _json(spec.root / "attempts/000000/terminal.json")
    assert terminal["status"] == "failed-preserved"
    assert terminal["end_count"] == fail_index
    assert terminal["uncommitted_transport_attempt_count"] is None
    assert not (spec.root / "acquisition-receipt.json").exists()
    failed = [response for response in source.responses if response.status != 206]
    assert len(failed) == 1 and failed[0].read_calls == 0 and failed[0].closed
    return spec, terminal, auth


@pytest.mark.parametrize("size", [1, 63, 64, 65, 130, 320, 4099])
def test_exact_streaming_hash_and_write_once_receipt(
    tmp_path: Path,
    engine: RHashCheckpointEngine,
    size: int,
) -> None:
    spec = _spec(tmp_path, engine, size)
    source = Source(spec)
    auth = _authorization(spec)
    receipt = _run(spec, engine, source, auth=auth)
    assert receipt["archive_sha256"] == hashlib.sha256(source.data).hexdigest()
    assert receipt["chunk_count"] == spec.chunk_count
    assert receipt["transport_attempts_for_committed_chunks"] == spec.chunk_count
    assert all(value is False for value in receipt["boundaries"].values())
    assert acquisition.verify_completed_receipt(spec, engine) == receipt
    assert all(
        path.suffix == ".json" for path in spec.root.rglob("*") if path.is_file()
    )
    after = Source(spec)
    with pytest.raises(ValueError, match="already exists"):
        _run(spec, engine, after, auth=auth)
    with pytest.raises(FileExistsError):
        acquisition.publish_completed_receipt(spec, engine)
    assert after.calls == []


def test_parallel_out_of_order_transport_preserves_ordered_hash(
    tmp_path: Path,
    engine: RHashCheckpointEngine,
) -> None:
    spec = _spec(tmp_path, engine, 4099, workers=8)
    source = Source(spec, out_of_order=True)
    result = _run(spec, engine, source)
    assert source.second_ready.is_set()
    assert result["archive_sha256"] == hashlib.sha256(source.data).hexdigest()
    assert sorted(source.calls) == [
        (start, min(start + 63, 4098)) for start in range(0, 4099, 64)
    ]


@pytest.mark.parametrize("fail_index", [0, 3, 5])
def test_fresh_authorized_attempt_resumes_only_after_committed_prefix(
    tmp_path: Path,
    engine: RHashCheckpointEngine,
    fail_index: int,
) -> None:
    spec, terminal, _ = _failed_prefix(tmp_path, engine, fail_index=fail_index)
    before = {path: path.read_bytes() for path in spec.root.rglob("*.json")}
    auth = _authorization(spec, previous=terminal, not_before=LATER)
    source = Source(spec)
    receipt = _run(spec, engine, source, auth=auth, now=LATER)
    assert source.calls[0][0] == fail_index * 64
    assert all(start >= fail_index * 64 for start, _ in source.calls)
    assert receipt["archive_sha256"] == hashlib.sha256(source.data).hexdigest()
    assert receipt["attempt_count"] == 2
    assert receipt["transport_attempts_for_committed_chunks"] == spec.chunk_count
    assert all(path.read_bytes() == data for path, data in before.items())
    assert acquisition.verify_completed_receipt(spec, engine) == receipt


def test_repeated_transport_failures_require_distinct_sealed_authorizations(
    tmp_path: Path,
    engine: RHashCheckpointEngine,
) -> None:
    spec, first, _ = _failed_prefix(tmp_path, engine)
    second_auth = _authorization(spec, previous=first, not_before=LATER)
    with pytest.raises(ValueError, match="HTTP 206"):
        _run(spec, engine, Source(spec, fail_index=4), auth=second_auth, now=LATER)
    second = _json(spec.root / "attempts/000001/terminal.json")
    assert second["end_count"] == 4
    latest = LATER + timedelta(days=1)
    final = Source(spec)
    receipt = _run(
        spec,
        engine,
        final,
        auth=_authorization(spec, previous=second, not_before=latest),
        now=latest,
    )
    assert receipt["attempt_count"] == 3
    assert final.calls == [(256, 319), (320, 320)]
    assert receipt["archive_sha256"] == hashlib.sha256(final.data).hexdigest()


@pytest.mark.parametrize(
    "case",
    [
        "cooldown",
        "old_authorization",
        "replay",
        "wrong_tip",
        "wrong_terminal",
        "wrong_count",
    ],
)
def test_invalid_resume_never_contacts_provider(
    tmp_path: Path,
    engine: RHashCheckpointEngine,
    case: str,
) -> None:
    spec, terminal, original = _failed_prefix(tmp_path, engine)
    changes: dict[str, Any] = {}
    time = LATER
    not_before = LATER
    if case == "cooldown":
        time = LATER - timedelta(seconds=1)
    elif case == "old_authorization":
        not_before = NOW + timedelta(seconds=1)
    elif case == "wrong_tip":
        changes["start_tip"] = "f" * 64
    elif case == "wrong_terminal":
        changes["previous_terminal_id"] = "f" * 64
    elif case == "wrong_count":
        changes["start_count"] = 2
    auth = (
        original
        if case == "replay"
        else _authorization(spec, previous=terminal, not_before=not_before, **changes)
    )
    source = Source(spec)
    with pytest.raises(ValueError):
        _run(spec, engine, source, auth=auth, now=time)
    assert source.calls == []
    assert not (spec.root / "attempts/000001").exists()


@pytest.mark.parametrize(
    "case",
    [
        "alter_chunk",
        "repin_chunk",
        "remove_chunk",
        "extra_chunk",
        "missing_terminal",
        "unknown_start",
        "unknown_terminal",
        "linked_chunks",
        "orphan",
        "library",
        "spec",
    ],
)
def test_changed_custody_fails_before_request(
    tmp_path: Path,
    engine: RHashCheckpointEngine,
    case: str,
) -> None:
    spec, terminal, _ = _failed_prefix(tmp_path, engine)
    auth = _authorization(spec, previous=terminal, not_before=LATER)
    path = spec.root / "chunks/000001.json"
    if case in {"alter_chunk", "repin_chunk"}:
        record = _json(path)
        record["range_sha256"] = "f" * 64
        if case == "repin_chunk":
            identity = {
                key: value for key, value in record.items() if key != "record_id"
            }
            record["record_id"] = content_id(identity)
        path.write_text(json.dumps(record))
    elif case == "remove_chunk":
        path.unlink()
    elif case == "extra_chunk":
        (spec.root / "chunks/000003.json").write_bytes(path.read_bytes())
    elif case == "missing_terminal":
        (spec.root / "attempts/000000/terminal.json").unlink()
    elif case in {"unknown_start", "unknown_terminal"}:
        path = spec.root / "attempts/000000" / (case[8:] + ".json")
        record = _json(path)
        record["unknown"] = True
        identity = {key: value for key, value in record.items() if key != "record_id"}
        record["record_id"] = content_id(identity)
        path.write_text(json.dumps(record))
    elif case == "linked_chunks":
        destination = tmp_path / "relocated-chunks"
        (spec.root / "chunks").rename(destination)
        (spec.root / "chunks").symlink_to(destination, target_is_directory=True)
    elif case == "orphan":
        (spec.root / "orphan.json").write_text("{}")
    elif case == "library":
        spec = replace(spec, library_sha256="f" * 64)
    elif case == "spec":
        spec = replace(spec, parent_sha256={"another_parent": "f" * 64})
    source = Source(spec)
    with pytest.raises(ValueError):
        _run(spec, engine, source, auth=auth, now=LATER)
    assert source.calls == []


@pytest.mark.parametrize(
    "changes",
    [
        {"attempt": True},
        {"attempt_limit": True},
        {"attempt_limit": 2},
        {"start_count": False},
        {"unknown": True},
        {"boundaries": {**acquisition._BOUNDARIES, "scientific_result": 0}},
        {"boundaries": {**acquisition._BOUNDARIES, "confirmation_opened": True}},
    ],
)
def test_invalid_authorization_fields_never_start(
    tmp_path: Path,
    engine: RHashCheckpointEngine,
    changes: dict[str, Any],
) -> None:
    spec = _spec(tmp_path, engine)
    auth = _authorization(spec, **changes)
    source = Source(spec)
    with pytest.raises(ValueError):
        _run(spec, engine, source, auth=auth)
    assert source.calls == []
    assert not spec.root.exists()


@pytest.mark.parametrize("case", ["symlink", "changed_hash", "noncanonical"])
def test_authorization_file_binding_is_required(
    tmp_path: Path,
    engine: RHashCheckpointEngine,
    case: str,
) -> None:
    spec = _spec(tmp_path, engine)
    path, digest = _authorization(spec)
    if case == "symlink":
        link = tmp_path / "linked-authorization.json"
        link.symlink_to(path)
        path = link
    elif case == "changed_hash":
        digest = "f" * 64
    else:
        path.write_text(json.dumps(_json(path)))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    source = Source(spec)
    with pytest.raises(ValueError):
        _run(spec, engine, source, auth=(path, digest))
    assert source.calls == []


def test_shared_lock_rejects_concurrent_attempt_before_consumption(
    tmp_path: Path,
    engine: RHashCheckpointEngine,
) -> None:
    spec = _spec(tmp_path, engine)
    source = Source(spec)
    with spec.lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            _run(spec, engine, source)
    assert source.calls == []
    assert not spec.root.exists()


@pytest.mark.parametrize(
    "header,value",
    [
        ("Last-Modified", "changed"),
        ("Content-Range", "bytes 0-63/1000"),
        ("Content-Length", "63"),
        ("Content-Type", "text/html"),
        ("Content-Disposition", 'attachment; filename="another.zip"'),
    ],
)
def test_identity_drift_preserves_failure_without_reading_body(
    tmp_path: Path,
    engine: RHashCheckpointEngine,
    header: str,
    value: str,
) -> None:
    spec = _spec(tmp_path, engine)
    source = Source(spec, header_changes={header: value})
    with pytest.raises(ValueError):
        _run(spec, engine, source)
    assert len(source.calls) == 1
    assert all(
        response.read_calls == 0 and response.closed for response in source.responses
    )
    terminal = _json(spec.root / "attempts/000000/terminal.json")
    assert terminal["end_count"] == 0
    assert terminal["status"] == "failed-preserved"


def test_http_error_is_terminal_not_an_ordinary_transport_retry(
    tmp_path: Path,
    engine: RHashCheckpointEngine,
) -> None:
    spec = _spec(tmp_path, engine)
    body = Response(b"synthetic HTML error; must not be read", spec, 0, 63)
    calls = 0

    def opener(request: urllib.request.Request, timeout: float) -> Any:
        nonlocal calls
        calls += 1
        raise urllib.error.HTTPError(
            request.full_url, 403, "Forbidden", Message(), body
        )

    path, digest = _authorization(spec)
    with pytest.raises(ValueError, match="HTTP 206"):
        acquisition.run_checkpointed_attempt(
            spec,
            engine,
            path,
            expected_authorization_sha256=digest,
            opener=opener,
            clock=lambda: NOW,
        )
    assert calls == 1 and body.closed and body.read_calls == 0
    response = _json(spec.root / "attempts/000000/response-000.json")
    assert response["http_status"] == 403 and response["response_body_read"] is False


@pytest.mark.parametrize("failures", [1, 2, 3])
def test_frozen_socket_retry_limit_and_counting(
    tmp_path: Path,
    engine: RHashCheckpointEngine,
    failures: int,
) -> None:
    spec = _spec(tmp_path, engine, 65)
    source = Source(spec, socket_failures=failures)
    if failures == 3:
        with pytest.raises(OSError, match="synthetic socket failure"):
            _run(spec, engine, source)
        assert len(source.calls) == 3
    else:
        receipt = _run(spec, engine, source)
        assert receipt["transport_attempts_for_committed_chunks"] == failures + 2
        assert receipt["archive_sha256"] == hashlib.sha256(source.data).hexdigest()


def test_missing_receipt_after_terminal_completion_can_be_published_offline(
    tmp_path: Path,
    engine: RHashCheckpointEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(tmp_path, engine)
    source = Source(spec)
    original = acquisition._write

    def crash(path: Path, value: Any) -> None:
        if path.name == "acquisition-receipt.json":
            raise OSError("synthetic receipt publication failure")
        original(path, value)

    monkeypatch.setattr(acquisition, "_write", crash)
    with pytest.raises(OSError, match="publication failure"):
        _run(spec, engine, source)
    monkeypatch.setattr(acquisition, "_write", original)
    assert _json(spec.root / "attempts/000000/terminal.json")["status"] == "complete"
    receipt = acquisition.publish_completed_receipt(spec, engine)
    assert receipt["archive_sha256"] == hashlib.sha256(source.data).hexdigest()
    assert acquisition.verify_completed_receipt(spec, engine) == receipt


def test_native_initialization_failure_preserves_consumed_attempt(
    tmp_path: Path,
    engine: RHashCheckpointEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(tmp_path, engine)
    source = Source(spec)

    def fail() -> Any:
        raise RuntimeError("synthetic native initialization failure")

    monkeypatch.setattr(engine, "new", fail)
    with pytest.raises(RuntimeError, match="native initialization failure"):
        _run(spec, engine, source)
    terminal = _json(spec.root / "attempts/000000/terminal.json")
    assert terminal["status"] == "failed-preserved" and terminal["end_count"] == 0
    assert source.calls == []
