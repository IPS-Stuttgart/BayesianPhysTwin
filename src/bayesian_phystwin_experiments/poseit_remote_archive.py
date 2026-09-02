"""Exact range transport for the public PoseIt archive."""

from __future__ import annotations

import hashlib
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from email.message import Message
from pathlib import Path
from typing import Protocol, cast
from urllib.parse import urlsplit


class RangeResponse(Protocol):
    headers: Mapping[str, str]
    status: int

    def close(self) -> None: ...

    def geturl(self) -> str: ...

    def read(self, amount: int = -1) -> bytes: ...


RangeOpener = Callable[[urllib.request.Request, float], RangeResponse]
ProgressCallback = Callable[["RemoteHashProgress"], None]

_CONTENT_RANGE = re.compile(r"bytes ([0-9]+)-([0-9]+)/([0-9]+)")
_RETRYABLE = (OSError, TimeoutError, urllib.error.URLError)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _header_filename(disposition: str) -> str | None:
    if not disposition:
        return None
    message = Message()
    message["Content-Disposition"] = disposition
    return message.get_filename()


def _allowed_final_host(url: str) -> str:
    parsed = urlsplit(url)
    _require(parsed.scheme == "https", "PoseIt range response left HTTPS")
    host = (parsed.hostname or "").casefold()
    _require(
        host in {"drive.google.com", "drive.usercontent.google.com"}
        or host.endswith(".googleusercontent.com"),
        "PoseIt range response reached an unregistered host",
    )
    return host


def _default_open(request: urllib.request.Request, timeout: float) -> RangeResponse:
    return cast(RangeResponse, urllib.request.urlopen(request, timeout=timeout))


@dataclass(frozen=True)
class RemoteArchiveExpectation:
    source_url: str
    file_name: str
    size_bytes: int
    last_modified: str
    chunk_size_bytes: int
    max_workers: int
    max_attempts_per_range: int
    timeout_seconds: float
    user_agent: str = "BayesianPhysTwin-PoseIt-Range-Acquisition/1"

    def __post_init__(self) -> None:
        parsed = urlsplit(self.source_url)
        _require(parsed.scheme == "https", "PoseIt source URL must use HTTPS")
        _require(bool(parsed.hostname), "PoseIt source URL has no host")
        _require(Path(self.file_name).name == self.file_name, "file name is unsafe")
        _require(self.size_bytes > 0, "archive size must be positive")
        _require(self.chunk_size_bytes > 0, "chunk size must be positive")
        _require(self.max_workers > 0, "worker count must be positive")
        _require(
            self.max_attempts_per_range > 0,
            "range attempt count must be positive",
        )
        _require(self.timeout_seconds > 0.0, "range timeout must be positive")
        _require(bool(self.last_modified), "last-modified identity is empty")
        _require(bool(self.user_agent), "user agent is empty")


@dataclass(frozen=True)
class RemoteRangeBlock:
    index: int
    start: int
    end: int
    data: bytes
    attempts: int


@dataclass(frozen=True)
class RemoteHashProgress:
    completed_chunks: int
    total_chunks: int
    bytes_hashed: int
    archive_size_bytes: int
    transport_attempts: int


@dataclass(frozen=True)
class RemoteHashResult:
    archive_sha256: str
    archive_size_bytes: int
    chunk_size_bytes: int
    chunk_count: int
    transport_attempts: int


def _read_exact(response: RangeResponse, expected_size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = expected_size
    while remaining:
        block = response.read(min(1024 * 1024, remaining))
        if not block:
            raise OSError("PoseIt range response ended early")
        _require(len(block) <= remaining, "PoseIt range response exceeded its range")
        chunks.append(block)
        remaining -= len(block)
    _require(response.read(1) == b"", "PoseIt range response had trailing bytes")
    return b"".join(chunks)


def _validate_response(
    response: RangeResponse,
    expectation: RemoteArchiveExpectation,
    *,
    start: int,
    end: int,
) -> None:
    expected_size = end - start + 1
    _require(response.status == 206, "PoseIt range request did not return HTTP 206")
    _allowed_final_host(response.geturl())
    disposition = response.headers.get("Content-Disposition", "")
    _require(
        "attachment" in disposition.casefold(),
        "PoseIt range response is not an attachment",
    )
    _require(
        _header_filename(disposition) == expectation.file_name,
        "PoseIt range archive name changed",
    )
    content_type = response.headers.get("Content-Type", "")
    _require(
        "text/html" not in content_type.casefold(), "PoseIt range response is HTML"
    )
    _require(
        response.headers.get("Accept-Ranges", "").casefold() == "bytes",
        "PoseIt source no longer advertises byte ranges",
    )
    _require(
        response.headers.get("Last-Modified", "") == expectation.last_modified,
        "PoseIt source last-modified identity changed",
    )
    match = _CONTENT_RANGE.fullmatch(response.headers.get("Content-Range", ""))
    _require(match is not None, "PoseIt content-range header is malformed")
    assert match is not None
    actual_start, actual_end, total_size = (int(value) for value in match.groups())
    _require(
        (actual_start, actual_end) == (start, end),
        "PoseIt response returned a different byte range",
    )
    _require(
        total_size == expectation.size_bytes,
        "PoseIt source size identity changed",
    )
    _require(
        response.headers.get("Content-Length", "") == str(expected_size),
        "PoseIt range content length changed",
    )


def fetch_exact_range(
    expectation: RemoteArchiveExpectation,
    *,
    index: int,
    start: int,
    end: int,
    opener: RangeOpener = _default_open,
) -> RemoteRangeBlock:
    """Fetch one exact range with only frozen transport retries."""

    _require(index >= 0, "range index must be nonnegative")
    _require(0 <= start <= end, "range bounds are invalid")
    _require(end < expectation.size_bytes, "range exceeds the archive")
    request = urllib.request.Request(
        expectation.source_url,
        headers={
            "Range": f"bytes={start}-{end}",
            "User-Agent": expectation.user_agent,
        },
        method="GET",
    )
    last_error: BaseException | None = None
    for attempt in range(1, expectation.max_attempts_per_range + 1):
        response: RangeResponse | None = None
        try:
            response = opener(request, expectation.timeout_seconds)
            _validate_response(response, expectation, start=start, end=end)
            data = _read_exact(response, end - start + 1)
            return RemoteRangeBlock(
                index=index,
                start=start,
                end=end,
                data=data,
                attempts=attempt,
            )
        except _RETRYABLE as error:
            last_error = error
            if attempt == expectation.max_attempts_per_range:
                raise
        finally:
            if response is not None:
                response.close()
    assert last_error is not None
    raise last_error


def hash_remote_archive(
    expectation: RemoteArchiveExpectation,
    *,
    opener: RangeOpener = _default_open,
    progress: ProgressCallback | None = None,
) -> RemoteHashResult:
    """Hash every remote archive byte in order without retaining the archive."""

    chunk_count = (
        expectation.size_bytes + expectation.chunk_size_bytes - 1
    ) // expectation.chunk_size_bytes

    def submit(executor: ThreadPoolExecutor, index: int) -> Future[RemoteRangeBlock]:
        start = index * expectation.chunk_size_bytes
        end = min(start + expectation.chunk_size_bytes, expectation.size_bytes) - 1
        return executor.submit(
            fetch_exact_range,
            expectation,
            index=index,
            start=start,
            end=end,
            opener=opener,
        )

    digest = hashlib.sha256()
    bytes_hashed = 0
    transport_attempts = 0
    futures: dict[int, Future[RemoteRangeBlock]] = {}
    executor = ThreadPoolExecutor(
        max_workers=expectation.max_workers,
        thread_name_prefix="poseit-range",
    )
    try:
        next_index = 0
        while next_index < min(expectation.max_workers, chunk_count):
            futures[next_index] = submit(executor, next_index)
            next_index += 1
        for index in range(chunk_count):
            block = futures.pop(index).result()
            _require(block.index == index, "PoseIt range order changed")
            digest.update(block.data)
            bytes_hashed += len(block.data)
            transport_attempts += block.attempts
            if next_index < chunk_count:
                futures[next_index] = submit(executor, next_index)
                next_index += 1
            if progress is not None:
                progress(
                    RemoteHashProgress(
                        completed_chunks=index + 1,
                        total_chunks=chunk_count,
                        bytes_hashed=bytes_hashed,
                        archive_size_bytes=expectation.size_bytes,
                        transport_attempts=transport_attempts,
                    )
                )
    finally:
        for future in futures.values():
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)

    _require(
        bytes_hashed == expectation.size_bytes, "PoseIt archive hash is incomplete"
    )
    return RemoteHashResult(
        archive_sha256=digest.hexdigest(),
        archive_size_bytes=bytes_hashed,
        chunk_size_bytes=expectation.chunk_size_bytes,
        chunk_count=chunk_count,
        transport_attempts=transport_attempts,
    )


__all__ = [
    "ProgressCallback",
    "RangeOpener",
    "RangeResponse",
    "RemoteArchiveExpectation",
    "RemoteHashProgress",
    "RemoteHashResult",
    "RemoteRangeBlock",
    "fetch_exact_range",
    "hash_remote_archive",
]
