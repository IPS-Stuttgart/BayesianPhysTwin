#!/usr/bin/env python3
"""Acquire the frozen public PoseIt GelSight archive without inspecting it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from collections.abc import Callable, Mapping
from email.message import Message
from pathlib import Path
from typing import Any, BinaryIO, Protocol, cast
from urllib.parse import urlencode, urlsplit

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin_experiments.poseit_real_decision_protocol import (
    POSEIT_GELSIGHT_FILE_ID,
    load_poseit_real_decision_protocol,
    poseit_protocol_config_sha256,
    poseit_protocol_file_sha256,
)

SCHEMA = "bayesian-phystwin.poseit-archive-acquisition-receipt"
SCHEMA_VERSION = 1
ARCHIVE_FILE_NAME = "gelsight.zip"
SOURCE_URL = "https://drive.usercontent.google.com/download?" + urlencode(
    {
        "id": POSEIT_GELSIGHT_FILE_ID,
        "export": "download",
        "confirm": "t",
    }
)
_HTML_PREFIX_LIMIT = 16 * 1024
_BLOCK_SIZE = 1024 * 1024


class _Response(Protocol):
    headers: Mapping[str, str]
    status: int

    def close(self) -> None: ...

    def geturl(self) -> str: ...

    def read(self, amount: int = -1) -> bytes: ...


_Opener = Callable[[urllib.request.Request, float], _Response]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _header_filename(disposition: str) -> str | None:
    if not disposition:
        return None
    message = Message()
    message["Content-Disposition"] = disposition
    return message.get_filename()


def _allowed_final_host(url: str) -> str:
    parsed = urlsplit(url)
    _require(parsed.scheme == "https", "PoseIt download left HTTPS")
    host = (parsed.hostname or "").casefold()
    _require(
        host in {"drive.google.com", "drive.usercontent.google.com"}
        or host.endswith(".googleusercontent.com"),
        "PoseIt download reached an unregistered host",
    )
    return host


def _default_open(request: urllib.request.Request, timeout: float) -> _Response:
    return cast(_Response, urllib.request.urlopen(request, timeout=timeout))


def _response_metadata(response: _Response) -> tuple[str, str, str]:
    disposition = response.headers.get("Content-Disposition", "")
    content_type = response.headers.get("Content-Type", "")
    filename = _header_filename(disposition)
    _require(response.status == 200, "PoseIt download did not return HTTP 200")
    if "attachment" not in disposition.casefold() or filename is None:
        if "text/html" in content_type.casefold():
            prefix = response.read(_HTML_PREFIX_LIMIT).decode("utf-8", errors="ignore")
            if (
                "too many users have viewed or downloaded this file recently"
                in prefix.casefold()
            ):
                raise RuntimeError("PoseIt archive is still Google Drive quota blocked")
        raise RuntimeError("PoseIt locator did not return an archive attachment")
    _require(Path(filename).name == ARCHIVE_FILE_NAME, "PoseIt archive name changed")
    _require("text/html" not in content_type.casefold(), "archive response is HTML")
    final_host = _allowed_final_host(response.geturl())
    return disposition, content_type, final_host


def _stream_response(response: _Response, stream: BinaryIO) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    while True:
        block = response.read(_BLOCK_SIZE)
        if not block:
            break
        stream.write(block)
        digest.update(block)
        size += len(block)
    _require(size > 0, "PoseIt archive response was empty")
    stream.flush()
    os.fsync(stream.fileno())
    return size, digest.hexdigest()


def _exclusive_link(staged: Path, destination: Path) -> None:
    os.link(staged, destination)


def _same_file(left: Path, right: Path) -> bool:
    try:
        return left.samefile(right)
    except FileNotFoundError:
        return False


def _acquire(
    archive: Path,
    receipt: Path,
    protocol_path: Path,
    *,
    expected_protocol_sha256: str,
    opener: _Opener = _default_open,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    _require(archive.name == ARCHIVE_FILE_NAME, "archive file name changed")
    _require(archive != receipt, "archive and receipt paths must differ")
    _require(not archive.exists(), "archive destination already exists")
    _require(not receipt.exists(), "acquisition receipt already exists")
    _require(timeout_seconds > 0.0, "download timeout must be positive")
    _require(len(expected_protocol_sha256) == 64, "protocol SHA-256 is malformed")
    protocol_file_sha256 = poseit_protocol_file_sha256(protocol_path)
    _require(
        protocol_file_sha256 == expected_protocol_sha256,
        "protocol file SHA-256 changed",
    )
    protocol = load_poseit_real_decision_protocol(protocol_path)

    archive.parent.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    archive_stage = archive.with_name(f".{archive.name}.acquire-v1.partial")
    receipt_stage = receipt.with_name(f".{receipt.name}.acquire-v1.partial")
    _require(not archive_stage.exists(), "archive staging path already exists")
    _require(not receipt_stage.exists(), "receipt staging path already exists")

    request = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "BayesianPhysTwin-PoseIt-Acquisition/1"},
        method="GET",
    )
    response: _Response | None = None
    archive_linked = False
    receipt_linked = False
    try:
        response = opener(request, timeout_seconds)
        disposition, content_type, final_host = _response_metadata(response)
        with archive_stage.open("xb") as stream:
            archive_size, archive_sha256 = _stream_response(response, stream)
        identity = {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "source": "official-public-google-drive-file",
            "source_file_id": POSEIT_GELSIGHT_FILE_ID,
            "source_file_name": ARCHIVE_FILE_NAME,
            "source_final_host": final_host,
            "response_content_disposition": disposition,
            "response_content_type": content_type,
            "archive_size_bytes": archive_size,
            "archive_sha256": archive_sha256,
            "protocol_file_sha256": protocol_file_sha256,
            "protocol_config_sha256": poseit_protocol_config_sha256(protocol),
            "archive_bytes_streamed_opaquely": True,
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
        result = {**identity, "receipt_id": content_id(identity)}
        with receipt_stage.open("xb") as stream:
            stream.write(_canonical_bytes(result))
            stream.flush()
            os.fsync(stream.fileno())
        _exclusive_link(archive_stage, archive)
        archive_linked = True
        _exclusive_link(receipt_stage, receipt)
        receipt_linked = True
        return result
    except BaseException:
        if receipt_linked and _same_file(receipt_stage, receipt):
            receipt.unlink()
        if archive_linked and _same_file(archive_stage, archive):
            archive.unlink()
        raise
    finally:
        if response is not None:
            response.close()
        archive_stage.unlink(missing_ok=True)
        receipt_stage.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    arguments = parser.parse_args()
    result = _acquire(
        arguments.archive.resolve(),
        arguments.receipt.resolve(),
        arguments.protocol.resolve(strict=True),
        expected_protocol_sha256=arguments.expected_protocol_sha256,
        timeout_seconds=arguments.timeout_seconds,
    )
    print(
        json.dumps(
            {
                "archive": str(arguments.archive.resolve()),
                "archive_sha256": result["archive_sha256"],
                "archive_size_bytes": result["archive_size_bytes"],
                "member_payload_bytes_opened": False,
                "receipt": str(arguments.receipt.resolve()),
                "receipt_id": result["receipt_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
