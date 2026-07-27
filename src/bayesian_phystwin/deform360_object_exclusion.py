"""Hash-only Deform360 object-exclusion manifests.

The merger deliberately operates on already-hashed physical-object identities.
It cannot recover or emit source object IDs, and it binds every input manifest
by both its canonical content digest and its byte-level file digest.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
EXCLUSION_KIND = "Deform360FreshObjectExclusionManifest"
HASH_NAMESPACE = "deform360-fresh-object-exclusion-v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_IDENTITY_KEYS = frozenset(
    {
        "case_id",
        "case_ids",
        "episode_id",
        "episode_ids",
        "object_id",
        "object_ids",
        "outcome",
        "outcomes",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha256(path: str | Path) -> str:
    """Hash a file without loading it fully into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(
    artifact: Mapping[str, Any],
    *,
    digest_key: str = "exclusion_sha256",
) -> str:
    """Return the canonical JSON digest with ``digest_key`` omitted."""

    payload = dict(artifact)
    payload.pop(digest_key, None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _contains_forbidden_identity_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key) in _FORBIDDEN_IDENTITY_KEYS
            or _contains_forbidden_identity_key(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_identity_key(child) for child in value)
    return False


def validate_object_exclusion_manifest(
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize one hash-only exclusion manifest."""

    try:
        normalized = json.loads(
            json.dumps(dict(artifact), sort_keys=True, allow_nan=False)
        )
    except (TypeError, ValueError) as error:
        raise ValueError("exclusion manifest must contain finite JSON") from error
    _require(
        normalized.get("schema_version") == SCHEMA_VERSION,
        "unsupported exclusion schema",
    )
    _require(
        normalized.get("artifact_kind") == EXCLUSION_KIND,
        "unsupported exclusion artifact kind",
    )
    _require(
        normalized.get("hash_namespace") == HASH_NAMESPACE,
        "exclusion hash namespace changed",
    )
    owner = normalized.get("owner")
    _require(isinstance(owner, str) and bool(owner), "exclusion owner is empty")
    hashes = normalized.get("object_hashes")
    _require(
        isinstance(hashes, list)
        and bool(hashes)
        and hashes == sorted(set(hashes))
        and all(isinstance(value, str) and _HEX64.fullmatch(value) for value in hashes),
        "object hashes must be a nonempty sorted unique SHA-256 list",
    )
    sources = normalized.get("source_artifact_sha256s")
    _require(
        isinstance(sources, list)
        and bool(sources)
        and sources == sorted(set(sources))
        and all(
            isinstance(value, str) and _HEX64.fullmatch(value) for value in sources
        ),
        "source artifact hashes must be a nonempty sorted unique SHA-256 list",
    )
    boundary = normalized.get("information_boundary")
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("target_artifact_read") is False
        and boundary.get("object_ids_emitted") is False,
        "exclusion manifest crossed its information boundary",
    )
    _require(
        not _contains_forbidden_identity_key(normalized),
        "exclusion manifest contains plaintext identity or outcome fields",
    )
    digest = normalized.get("exclusion_sha256")
    _require(
        isinstance(digest, str)
        and bool(_HEX64.fullmatch(digest))
        and digest == canonical_sha256(normalized),
        "exclusion manifest checksum changed",
    )
    return normalized


def load_object_exclusion_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate one hash-only exclusion manifest."""

    source = Path(path)
    try:
        artifact = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read exclusion manifest: {source}") from error
    _require(isinstance(artifact, Mapping), "exclusion manifest is not an object")
    return validate_object_exclusion_manifest(artifact)


def merge_object_exclusion_manifests(
    paths: Sequence[str | Path],
    *,
    owner: str,
) -> dict[str, Any]:
    """Merge validated manifests without reading plaintext object identities."""

    _require(isinstance(owner, str) and bool(owner), "merged owner is empty")
    sources = tuple(Path(path).resolve() for path in paths)
    _require(bool(sources), "no exclusion manifests were provided")
    _require(len(sources) == len(set(sources)), "duplicate exclusion path")

    members: list[dict[str, Any]] = []
    object_hashes: set[str] = set()
    input_hash_count = 0
    source_file_hashes: list[str] = []
    for source in sources:
        artifact = load_object_exclusion_manifest(source)
        file_digest = file_sha256(source)
        member_hashes = artifact["object_hashes"]
        object_hashes.update(member_hashes)
        input_hash_count += len(member_hashes)
        source_file_hashes.append(file_digest)
        members.append(
            {
                "owner": artifact["owner"],
                "exclusion_sha256": artifact["exclusion_sha256"],
                "file_sha256": file_digest,
                "object_hash_count": len(member_hashes),
            }
        )
    members.sort(
        key=lambda item: (
            item["owner"],
            item["exclusion_sha256"],
            item["file_sha256"],
        )
    )
    merged: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": EXCLUSION_KIND,
        "hash_namespace": HASH_NAMESPACE,
        "owner": owner,
        "object_hashes": sorted(object_hashes),
        "source_artifact_sha256s": sorted(source_file_hashes),
        "composition": {
            "rule": "set-union-of-validated-hash-only-manifests",
            "member_count": len(members),
            "input_hash_count": input_hash_count,
            "unique_object_hash_count": len(object_hashes),
            "members": members,
        },
        "information_boundary": {
            "target_artifact_read": False,
            "object_ids_emitted": False,
            "input_manifests_hash_only": True,
        },
    }
    merged["exclusion_sha256"] = canonical_sha256(merged)
    return validate_object_exclusion_manifest(merged)


def write_object_exclusion_manifest(
    path: str | Path,
    artifact: Mapping[str, Any],
) -> None:
    """Validate and atomically create one exclusion manifest."""

    normalized = validate_object_exclusion_manifest(artifact)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _require(not target.exists(), f"refusing to replace exclusion manifest: {target}")
    temporary = target.with_name(f".{target.name}.tmp")
    _require(not temporary.exists(), f"temporary output already exists: {temporary}")
    temporary.write_text(
        json.dumps(normalized, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


__all__ = [
    "EXCLUSION_KIND",
    "HASH_NAMESPACE",
    "SCHEMA_VERSION",
    "canonical_sha256",
    "file_sha256",
    "load_object_exclusion_manifest",
    "merge_object_exclusion_manifests",
    "validate_object_exclusion_manifest",
    "write_object_exclusion_manifest",
]
