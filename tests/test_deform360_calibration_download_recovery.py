from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.science.deform360_calibration_source.cli import (
    _prepare_public_hub_environment,
)
from scripts.science.deform360_calibration_source.contracts import DATASET_REVISION
from scripts.science.deform360_calibration_source.download import (
    DownloadRetryExhaustedError,
    _is_transient_download_error,
    _retry_delay,
    download_one,
)


def _record(payload: bytes) -> dict[str, object]:
    return {
        "path": "raw/calibration-object/metadata.json",
        "size": len(payload),
        "blob_id": "b" * 40,
        "lfs_sha256": hashlib.sha256(payload).hexdigest(),
    }


def test_verified_local_file_is_reused_without_network(tmp_path: Path) -> None:
    payload = b"already verified calibration bytes"
    record = _record(payload)
    destination = tmp_path / str(record["path"])
    destination.parent.mkdir(parents=True)
    destination.write_bytes(payload)

    def forbidden_download(**kwargs: object) -> str:
        del kwargs
        raise AssertionError("verified local bytes must not be downloaded again")

    result = download_one(
        record=record,
        root=tmp_path,
        hub_download=forbidden_download,
    )

    assert result["download_source"] == "verified_local"
    assert result["download_attempt_count"] == 0
    assert result["downloaded_sha256"] == record["lfs_sha256"]


def test_mismatched_local_file_is_replaced_by_exact_hub_bytes(tmp_path: Path) -> None:
    payload = b"locked calibration bytes"
    record = _record(payload)
    destination = tmp_path / str(record["path"])
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"partial")
    calls = 0

    def download(**kwargs: object) -> str:
        nonlocal calls
        calls += 1
        assert kwargs["revision"] == DATASET_REVISION
        destination.write_bytes(payload)
        return str(destination)

    result = download_one(
        record=record,
        root=tmp_path,
        hub_download=download,
    )

    assert calls == 1
    assert result["download_source"] == "hub_download"
    assert result["download_attempt_count"] == 1
    assert destination.read_bytes() == payload


def test_http_429_uses_bounded_exponential_retry(tmp_path: Path) -> None:
    payload = b"eventually available calibration bytes"
    record = _record(payload)
    destination = tmp_path / str(record["path"])
    calls = 0
    sleeps: list[float] = []

    class RateLimitedError(RuntimeError):
        def __init__(self) -> None:
            super().__init__("HTTP status client error (429 Too Many Requests)")
            self.response = SimpleNamespace(status_code=429)

    def download(**kwargs: object) -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise RateLimitedError
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        return str(destination)

    result = download_one(
        record=record,
        root=tmp_path,
        hub_download=download,
        max_attempts=4,
        initial_backoff_seconds=2.0,
        maximum_backoff_seconds=3.0,
        sleep=sleeps.append,
    )

    assert calls == 3
    assert sleeps == [2.0, 3.0]
    assert result["download_attempt_count"] == 3
    assert result["download_source"] == "hub_download"


def test_complete_bytes_written_before_transient_error_are_recovered(
    tmp_path: Path,
) -> None:
    payload = b"complete bytes before transport exception"
    record = _record(payload)
    destination = tmp_path / str(record["path"])
    sleeps: list[float] = []

    def download(**kwargs: object) -> str:
        del kwargs
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        raise ConnectionError("connection closed after complete response")

    result = download_one(
        record=record,
        root=tmp_path,
        hub_download=download,
        max_attempts=3,
        sleep=sleeps.append,
    )

    assert sleeps == []
    assert result["download_source"] == "verified_local_after_transient_error"
    assert result["download_attempt_count"] == 1


def test_nontransient_error_is_not_retried(tmp_path: Path) -> None:
    calls = 0
    sleeps: list[float] = []

    def download(**kwargs: object) -> str:
        nonlocal calls
        del kwargs
        calls += 1
        raise ValueError("invalid immutable revision")

    with pytest.raises(ValueError, match="immutable revision"):
        download_one(
            record=_record(b"bytes"),
            root=tmp_path,
            hub_download=download,
            max_attempts=5,
            sleep=sleeps.append,
        )

    assert calls == 1
    assert sleeps == []


def test_transient_retry_budget_fails_closed_without_path_disclosure(
    tmp_path: Path,
) -> None:
    record = _record(b"bytes")
    sleeps: list[float] = []

    def download(**kwargs: object) -> str:
        del kwargs
        raise RuntimeError("Xet CAS HTTP status client error (429 Too Many Requests)")

    with pytest.raises(DownloadRetryExhaustedError) as captured:
        download_one(
            record=record,
            root=tmp_path,
            hub_download=download,
            max_attempts=3,
            initial_backoff_seconds=1.0,
            maximum_backoff_seconds=5.0,
            sleep=sleeps.append,
        )

    assert sleeps == [1.0, 2.0]
    assert str(record["path"]) not in str(captured.value)
    assert "retry budget exhausted" in str(captured.value)


def test_path_escape_is_rejected_before_network(tmp_path: Path) -> None:
    record = _record(b"bytes")
    record["path"] = "../confirmation/secret.bin"

    with pytest.raises(ValueError, match="unsafe"):
        download_one(
            record=record,
            root=tmp_path,
            hub_download=lambda **kwargs: str(tmp_path / "unexpected"),
        )


def test_transient_error_classifier_handles_status_and_timeout() -> None:
    error = RuntimeError("outer")
    error.__cause__ = RuntimeError("HTTP 503 Service Unavailable")

    assert _is_transient_download_error(error)
    assert _is_transient_download_error(TimeoutError("timed out"))
    assert not _is_transient_download_error(ValueError("bad revision"))


def test_retry_delay_is_bounded() -> None:
    assert _retry_delay(
        failed_attempt=1,
        initial_backoff_seconds=2.0,
        maximum_backoff_seconds=5.0,
    ) == 2.0
    assert _retry_delay(
        failed_attempt=4,
        initial_backoff_seconds=2.0,
        maximum_backoff_seconds=5.0,
    ) == 5.0


def test_cli_freezes_credential_free_non_xet_hub_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "HF_HUB_DISABLE_XET",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN",
        "HF_HUB_DISABLE_PROGRESS_BARS",
        "HF_HUB_DISABLE_TELEMETRY",
    ):
        monkeypatch.setenv(name, "0")

    _prepare_public_hub_environment()

    assert os.environ["HF_HUB_DISABLE_XET"] == "1"
    assert os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
