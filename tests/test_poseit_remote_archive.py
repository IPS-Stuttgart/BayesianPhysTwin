from __future__ import annotations

import hashlib
import threading
import urllib.error
import urllib.request
from collections.abc import Mapping

import pytest

from bayesian_phystwin_experiments.poseit_remote_archive import (
    RemoteArchiveExpectation,
    RemoteHashProgress,
    fetch_exact_range,
    hash_remote_archive,
)


class _Response:
    def __init__(
        self,
        payload: bytes,
        *,
        start: int,
        total_size: int,
        status: int = 206,
        final_url: str = "https://drive.usercontent.google.com/download",
        file_name: str = "gelsight.zip",
        last_modified: str = "Sat, 20 Aug 2022 02:26:04 GMT",
        content_range: str | None = None,
    ) -> None:
        self._payload = payload
        self._position = 0
        self.status = status
        end = start + len(payload) - 1
        self.headers: Mapping[str, str] = {
            "Accept-Ranges": "bytes",
            "Content-Disposition": f'attachment; filename="{file_name}"',
            "Content-Length": str(len(payload)),
            "Content-Range": content_range or f"bytes {start}-{end}/{total_size}",
            "Content-Type": "application/octet-stream",
            "Last-Modified": last_modified,
        }
        self._final_url = final_url
        self.closed = False

    def read(self, amount: int = -1) -> bytes:
        if amount < 0:
            amount = len(self._payload) - self._position
        block = self._payload[self._position : self._position + amount]
        self._position += len(block)
        return block

    def geturl(self) -> str:
        return self._final_url

    def close(self) -> None:
        self.closed = True


def _expectation(
    size: int, *, workers: int = 3, attempts: int = 1
) -> RemoteArchiveExpectation:
    return RemoteArchiveExpectation(
        source_url="https://drive.usercontent.google.com/download?id=frozen",
        file_name="gelsight.zip",
        size_bytes=size,
        last_modified="Sat, 20 Aug 2022 02:26:04 GMT",
        chunk_size_bytes=7,
        max_workers=workers,
        max_attempts_per_range=attempts,
        timeout_seconds=11.0,
    )


def _range(request: urllib.request.Request) -> tuple[int, int]:
    value = request.get_header("Range")
    assert value is not None
    prefix, bounds = value.split("=", maxsplit=1)
    assert prefix == "bytes"
    start, end = bounds.split("-", maxsplit=1)
    return int(start), int(end)


def test_remote_hash_is_exact_and_ordered_with_bounded_parallel_ranges() -> None:
    payload = bytes(range(101))
    expectation = _expectation(len(payload), workers=4)
    requested: list[tuple[int, int]] = []
    request_lock = threading.Lock()
    progress: list[RemoteHashProgress] = []

    def opener(request: urllib.request.Request, timeout: float) -> _Response:
        assert timeout == 11.0
        start, end = _range(request)
        with request_lock:
            requested.append((start, end))
        return _Response(
            payload[start : end + 1],
            start=start,
            total_size=len(payload),
        )

    result = hash_remote_archive(expectation, opener=opener, progress=progress.append)

    assert result.archive_sha256 == hashlib.sha256(payload).hexdigest()
    assert result.archive_size_bytes == len(payload)
    assert result.chunk_count == 15
    assert result.transport_attempts == 15
    assert sorted(requested) == [
        (start, min(start + 7, len(payload)) - 1) for start in range(0, len(payload), 7)
    ]
    assert [item.completed_chunks for item in progress] == list(range(1, 16))
    assert [item.bytes_hashed for item in progress] == [
        min(index * 7, len(payload)) for index in range(1, 16)
    ]


def test_range_transport_retries_only_retryable_failures() -> None:
    payload = b"range-payload"
    expectation = _expectation(len(payload), attempts=2)
    calls = 0

    def opener(request: urllib.request.Request, timeout: float) -> _Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise urllib.error.URLError("temporary")
        start, end = _range(request)
        return _Response(
            payload[start : end + 1],
            start=start,
            total_size=len(payload),
        )

    block = fetch_exact_range(
        expectation,
        index=0,
        start=0,
        end=len(payload) - 1,
        opener=opener,
    )

    assert block.data == payload
    assert block.attempts == 2
    assert calls == 2


@pytest.mark.parametrize(
    ("response_kwargs", "message"),
    (
        ({"status": 200}, "HTTP 206"),
        ({"final_url": "https://example.invalid/archive"}, "unregistered host"),
        ({"file_name": "other.zip"}, "name changed"),
        ({"last_modified": "different"}, "last-modified"),
        ({"content_range": "bytes 1-3/4"}, "different byte range"),
        ({"content_range": "bytes 0-3/5"}, "size identity"),
    ),
)
def test_range_transport_rejects_identity_drift_without_retry(
    response_kwargs: dict[str, object], message: str
) -> None:
    payload = b"four"
    expectation = _expectation(len(payload), attempts=3)
    calls = 0

    def opener(request: urllib.request.Request, timeout: float) -> _Response:
        nonlocal calls
        calls += 1
        return _Response(
            payload,
            start=0,
            total_size=len(payload),
            **response_kwargs,  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match=message):
        fetch_exact_range(
            expectation,
            index=0,
            start=0,
            end=len(payload) - 1,
            opener=opener,
        )

    assert calls == 1


def test_range_transport_rejects_truncated_response() -> None:
    payload = b"complete"
    expectation = _expectation(len(payload))

    def opener(request: urllib.request.Request, timeout: float) -> _Response:
        response = _Response(payload, start=0, total_size=len(payload))
        response._payload = payload[:-1]
        return response

    with pytest.raises(OSError, match="ended early"):
        fetch_exact_range(
            expectation,
            index=0,
            start=0,
            end=len(payload) - 1,
            opener=opener,
        )


def test_remote_expectation_rejects_unsafe_or_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="file name"):
        RemoteArchiveExpectation(
            source_url="https://drive.google.com/download",
            file_name="../gelsight.zip",
            size_bytes=1,
            last_modified="known",
            chunk_size_bytes=1,
            max_workers=1,
            max_attempts_per_range=1,
            timeout_seconds=1.0,
        )
