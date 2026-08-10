"""Strict bounded JSON input and atomic report publication."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

DEFAULT_MAXIMUM_INPUT_BYTES: Final = 64 * 1024 * 1024


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"input JSON contains non-finite constant {value!r}")


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"input JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _signature(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def load_json_object(
    path: str | Path,
    *,
    maximum_input_bytes: int = DEFAULT_MAXIMUM_INPUT_BYTES,
) -> tuple[Mapping[str, object], dict[str, object]]:
    """Read one unchanged ordinary UTF-8 JSON object."""

    if type(maximum_input_bytes) is not int:
        raise TypeError("maximum_input_bytes must be a genuine integer")
    if maximum_input_bytes < 1:
        raise ValueError("maximum_input_bytes must be positive")
    supplied = Path(path)
    if supplied.is_symlink():
        raise ValueError("input JSON must not be a symbolic link")
    input_path = supplied.resolve(strict=True)
    before = input_path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("input JSON must be an ordinary file")
    if before.st_size > maximum_input_bytes:
        raise ValueError("input JSON exceeds its byte budget")
    payload_bytes = input_path.read_bytes()
    after = input_path.stat()
    if _signature(before) != _signature(after):
        raise ValueError("input JSON changed while it was read")
    if len(payload_bytes) != before.st_size:
        raise ValueError("input JSON size changed while it was read")
    try:
        text = payload_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("input must be UTF-8 JSON") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError("input is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError("input JSON must contain an object")
    return cast(Mapping[str, object], value), {
        "path": str(input_path),
        "sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "bytes": len(payload_bytes),
    }


def _sync_directory(path: Path) -> None:
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


def publish_json_report(
    path: str | Path,
    report: Mapping[str, Any],
    *,
    input_artifact: Mapping[str, Any],
    overwrite: bool = False,
) -> dict[str, Any]:
    """Atomically publish one content-addressed JSON report."""

    if type(overwrite) is not bool:
        raise TypeError("overwrite must be a bool")
    if "input_artifact" in report or "status_sha256" in report:
        raise ValueError("report already contains publication-owned fields")
    emitted = dict(report)
    emitted["input_artifact"] = dict(input_artifact)
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
    if os.path.lexists(output_path) and not overwrite:
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
        _sync_directory(output_path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return emitted


__all__ = [
    "DEFAULT_MAXIMUM_INPUT_BYTES",
    "canonical_json_sha256",
    "load_json_object",
    "publish_json_report",
]
