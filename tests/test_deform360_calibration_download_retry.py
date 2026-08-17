"""Transient-only retry contracts for frozen Deform360 file acquisition."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from scripts.science.deform360_calibration_source import download as acquisition

_PAYLOAD = b"frozen calibration payload\n"


class _Response:
    def __init__(
        self,
        status_code: int,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers = {} if headers is None else headers


class _HttpError(Exception):
    def __init__(
        self,
        status_code: int,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(f"HTTP status code {status_code}")
        self.response = _Response(status_code, headers=headers)


def _record(*, payload: bytes = _PAYLOAD) -> dict[str, Any]:
    return {
        "path": "raw/calibration-object/camera/frame.bin",
        "size": len(payload),
        "blob_id": "a" * 40,
        "lfs_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write(root: Path, payload: bytes = _PAYLOAD) -> Path:
    path = root / _record(payload=payload)["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_verified_existing_file_resumes_without_network_access(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write(root)
    calls = 0

    def hub_download(**_kwargs: Any) -> str:
        nonlocal calls
        calls += 1
        raise AssertionError("verified local bytes must bypass the Hub")

    result = acquisition.download_one(
        record=_record(),
        root=root,
        hub_download=hub_download,
    )

    assert calls == 0
    assert result["downloaded_size"] == len(_PAYLOAD)
    assert result["downloaded_sha256"] == hashlib.sha256(_PAYLOAD).hexdigest()


def test_xet_429_text_retries_then_resumes_from_completed_file(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    calls = 0
    delays: list[float] = []

    def hub_download(**_kwargs: Any) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionError(
                "CAS service error: HTTP status client error (429 Too Many Requests)"
            )
        return str(_write(root))

    result = acquisition.download_one(
        record=_record(),
        root=root,
        hub_download=hub_download,
        max_attempts=2,
        initial_backoff_seconds=1.0,
        max_backoff_seconds=10.0,
        sleeper=delays.append,
    )

    assert calls == 2
    assert delays == [1.0]
    assert result["downloaded_sha256"] == hashlib.sha256(_PAYLOAD).hexdigest()


def test_retry_after_is_honored_for_rate_limits(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    calls = 0
    delays: list[float] = []

    def hub_download(**_kwargs: Any) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _HttpError(429, headers={"Retry-After": "7"})
        return str(_write(root))

    acquisition.download_one(
        record=_record(),
        root=root,
        hub_download=hub_download,
        max_attempts=2,
        initial_backoff_seconds=1.0,
        max_backoff_seconds=10.0,
        sleeper=delays.append,
    )

    assert calls == 2
    assert delays == [7.0]


def test_connection_failures_use_bounded_exponential_backoff(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    calls = 0
    delays: list[float] = []

    def hub_download(**_kwargs: Any) -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ConnectionError("temporary transport failure")
        return str(_write(root))

    acquisition.download_one(
        record=_record(),
        root=root,
        hub_download=hub_download,
        max_attempts=3,
        initial_backoff_seconds=1.0,
        max_backoff_seconds=2.0,
        sleeper=delays.append,
    )

    assert calls == 3
    assert delays == [1.0, 2.0]


@pytest.mark.parametrize("status_code", (401, 403, 404))
def test_authentication_and_revision_errors_fail_immediately(
    tmp_path: Path,
    status_code: int,
) -> None:
    calls = 0
    delays: list[float] = []

    def hub_download(**_kwargs: Any) -> str:
        nonlocal calls
        calls += 1
        raise _HttpError(status_code)

    with pytest.raises(_HttpError):
        acquisition.download_one(
            record=_record(),
            root=tmp_path.resolve(),
            hub_download=hub_download,
            max_attempts=6,
            initial_backoff_seconds=1.0,
            max_backoff_seconds=10.0,
            sleeper=delays.append,
        )

    assert calls == 1
    assert delays == []


def test_digest_mismatch_is_not_retried(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    calls = 0
    delays: list[float] = []

    def hub_download(**_kwargs: Any) -> str:
        nonlocal calls
        calls += 1
        return str(_write(root, b"wrong bytes\n"))

    with pytest.raises(ValueError, match="size changed|digest changed"):
        acquisition.download_one(
            record=_record(),
            root=root,
            hub_download=hub_download,
            max_attempts=6,
            initial_backoff_seconds=1.0,
            max_backoff_seconds=10.0,
            sleeper=delays.append,
        )

    assert calls == 1
    assert delays == []


def test_transient_failure_exhaustion_preserves_last_error(tmp_path: Path) -> None:
    calls = 0
    delays: list[float] = []

    def hub_download(**_kwargs: Any) -> str:
        nonlocal calls
        calls += 1
        raise _HttpError(503)

    with pytest.raises(_HttpError, match="503"):
        acquisition.download_one(
            record=_record(),
            root=tmp_path.resolve(),
            hub_download=hub_download,
            max_attempts=3,
            initial_backoff_seconds=1.0,
            max_backoff_seconds=10.0,
            sleeper=delays.append,
        )

    assert calls == 3
    assert delays == [1.0, 2.0]


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        ({"max_attempts": 0}, "max_attempts"),
        ({"initial_backoff_seconds": -1.0}, "initial backoff"),
        (
            {"initial_backoff_seconds": 2.0, "max_backoff_seconds": 1.0},
            "maximum backoff",
        ),
    ),
)
def test_invalid_retry_configuration_fails_before_network_access(
    tmp_path: Path,
    arguments: dict[str, Any],
    message: str,
) -> None:
    calls = 0

    def hub_download(**_kwargs: Any) -> str:
        nonlocal calls
        calls += 1
        return str(_write(tmp_path.resolve()))

    with pytest.raises(ValueError, match=message):
        acquisition.download_one(
            record=_record(),
            root=tmp_path.resolve(),
            hub_download=hub_download,
            **arguments,
        )

    assert calls == 0
