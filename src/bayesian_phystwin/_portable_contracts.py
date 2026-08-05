"""Strict JSON and content-addressing helpers for portable public contracts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ._canonical_contracts import (
    canonical_string_tuple,
    frozen_finite_json_mapping,
    plain_json,
)


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Serialize one finite mapping into canonical UTF-8 JSON bytes."""

    return json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def content_id(value: Mapping[str, Any]) -> str:
    """Return a SHA-256 content identity for one canonical JSON mapping."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not permitted: {value}")


def load_strict_json_object(path: str | Path, *, label: str) -> Mapping[str, Any]:
    """Load one UTF-8 JSON object while rejecting duplicate keys and NaN."""

    source = Path(path)
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object_pairs,
            parse_constant=_reject_nonfinite_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label} {source}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} root must be a JSON object")
    return payload


def require_exact_fields(
    value: Mapping[str, Any],
    *,
    expected: frozenset[str],
    name: str,
) -> None:
    """Reject missing and unknown public-contract fields."""

    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise ValueError(f"{name} fields changed: missing={missing}, extra={extra}")


def nonempty_string(value: object, *, name: str) -> str:
    """Require a plain, nonempty string without coercion."""

    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def sha256_digest(value: object, *, name: str) -> str:
    """Require a lowercase SHA-256 digest."""

    digest = nonempty_string(value, name=name)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def exact_revision(value: object, *, name: str) -> str:
    """Require an exact lowercase 40- or 64-character source revision."""

    revision = nonempty_string(value, name=name)
    if len(revision) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError(f"{name} must be an exact lowercase revision")
    return revision


def repository_name(value: object, *, name: str) -> str:
    """Require a canonical ``owner/name`` repository identifier."""

    repository = nonempty_string(value, name=name)
    parts = repository.split("/")
    if len(parts) != 2 or any(not part or part.strip() != part for part in parts):
        raise ValueError(f"{name} must be a canonical owner/name repository")
    return repository


def canonical_sorted_strings(
    values: Sequence[str],
    *,
    name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    """Copy a sequence into a sorted, unique string tuple."""

    items = canonical_string_tuple(values, name=name, allow_empty=allow_empty)
    if len(set(items)) != len(items):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(sorted(items))


def source_artifact_mapping(
    values: Mapping[str, str],
    *,
    name: str,
    allow_empty: bool = False,
) -> Mapping[str, Any]:
    """Validate and freeze a path-to-SHA-256 mapping."""

    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping")
    if not allow_empty and not values:
        raise ValueError(f"{name} must not be empty")
    normalized: dict[str, str] = {}
    for path, digest in values.items():
        if type(path) is not str or not path or path.strip() != path:
            raise ValueError(f"{name} keys must be non-empty canonical paths")
        normalized[path] = sha256_digest(digest, name=f"{name} entry {path}")
    return frozen_finite_json_mapping(normalized, name=name)


def write_atomic_json(
    value: Mapping[str, Any],
    path: str | Path,
    *,
    overwrite: bool,
) -> None:
    """Write canonical pretty JSON through fsync and atomic replacement."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    data = (
        json.dumps(
            plain_json(value),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "canonical_json_bytes",
    "canonical_sorted_strings",
    "content_id",
    "exact_revision",
    "load_strict_json_object",
    "nonempty_string",
    "repository_name",
    "require_exact_fields",
    "sha256_digest",
    "source_artifact_mapping",
    "write_atomic_json",
]
