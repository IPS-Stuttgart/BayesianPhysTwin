"""Strict input and atomic publication for provider-failure reports."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

from ._canonical_contracts import plain_json

DEFAULT_MAXIMUM_INPUT_BYTES: Final = 64 * 1024 * 1024


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    """Return the SHA-256 of one finite, canonical JSON mapping."""

    encoded = json.dumps(
        plain_json(payload),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"input JSON contains non-finite constant {value!r}")


def _strict_json_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"input JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _ordinary_file_signature(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def load_provider_failure_input(
    path: str | Path,
    *,
    maximum_input_bytes: int = DEFAULT_MAXIMUM_INPUT_BYTES,
) -> tuple[Mapping[str, object], dict[str, object]]:
    """Read one unchanged ordinary UTF-8 JSON object with strict semantics."""

    if isinstance(maximum_input_bytes, bool) or not isinstance(
        maximum_input_bytes, int
    ):
        raise TypeError("maximum_input_bytes must be a genuine integer")
    if maximum_input_bytes < 1:
        raise ValueError("maximum_input_bytes must be positive")

    input_path = Path(path).resolve(strict=True)
    before = input_path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("provider-failure input must be an ordinary file")
    if before.st_size > maximum_input_bytes:
        raise ValueError("provider-failure input exceeds its byte budget")
    payload_bytes = input_path.read_bytes()
    after = input_path.stat()
    if _ordinary_file_signature(before) != _ordinary_file_signature(after):
        raise ValueError("provider-failure input changed while it was read")
    if len(payload_bytes) != before.st_size:
        raise ValueError("provider-failure input size changed while it was read")
    try:
        text = payload_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("provider-failure input must be UTF-8 JSON") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError("provider-failure input is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError("provider-failure input must contain a JSON object")
    artifact = {
        "path": str(input_path),
        "sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "bytes": len(payload_bytes),
    }
    return cast(Mapping[str, object], value), artifact


def _sync_parent_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            pass
    finally:
        os.close(descriptor)


def publish_provider_failure_report(
    path: str | Path,
    report: Mapping[str, Any],
    *,
    input_artifact: Mapping[str, Any],
    overwrite: bool = False,
) -> dict[str, Any]:
    """Publish one verified report atomically with a host-local status digest."""

    if type(overwrite) is not bool:
        raise TypeError("overwrite must be a bool")
    if "input_artifact" in report or "status_sha256" in report:
        raise ValueError("report already contains publication-owned fields")
    emitted = cast(dict[str, Any], plain_json(report))
    emitted["input_artifact"] = plain_json(input_artifact)
    emitted["status_sha256"] = canonical_json_sha256(emitted)
    serialized = (
        json.dumps(
            emitted,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )

    output_path = Path(path).resolve(strict=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, output_path)
        else:
            os.link(temporary, output_path)
            temporary.unlink()
        _sync_parent_directory(output_path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return emitted


__all__ = [
    "DEFAULT_MAXIMUM_INPUT_BYTES",
    "canonical_json_sha256",
    "load_provider_failure_input",
    "publish_provider_failure_report",
]
