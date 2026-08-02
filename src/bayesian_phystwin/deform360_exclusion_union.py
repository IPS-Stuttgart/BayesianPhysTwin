"""Hash-only Deform360 exclusion unions for independent prospective cohorts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .deform360_tactile_features import canonical_artifact_sha256, file_sha256

EXCLUSION_KIND = "Deform360FreshObjectExclusionManifest"
HASH_NAMESPACE = "deform360-fresh-object-exclusion-v1"
_HASH_PREFIX = b"deform360-fresh-object-exclusion-v1\0"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_OBJECT_ID = re.compile(r"^[0-9]{3}-[a-z0-9][a-z0-9-]*$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def object_exclusion_hash(object_id: str) -> str:
    """Hash an object identity in the shared Deform360 exclusion namespace."""

    _require(_OBJECT_ID.fullmatch(object_id) is not None, "invalid Deform360 object ID")
    return hashlib.sha256(_HASH_PREFIX + object_id.encode("utf-8")).hexdigest()


def _canonical_exclusion_sha256(payload: Mapping[str, Any]) -> str:
    stripped = dict(payload)
    stripped.pop("exclusion_sha256", None)
    encoded = json.dumps(stripped, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_exclusion_manifest(payload: Mapping[str, Any]) -> None:
    """Validate a hash-only manifest without recovering protected identities."""

    _require(payload.get("artifact_kind") == EXCLUSION_KIND, "wrong exclusion kind")
    _require(payload.get("hash_namespace") == HASH_NAMESPACE, "wrong hash namespace")
    _require(
        payload.get("exclusion_sha256") == _canonical_exclusion_sha256(payload),
        "exclusion checksum changed",
    )
    hashes = payload.get("object_hashes")
    _require(
        isinstance(hashes, list)
        and bool(hashes)
        and hashes == sorted(set(hashes))
        and all(isinstance(value, str) and _HEX64.fullmatch(value) for value in hashes),
        "exclusion hashes are malformed",
    )
    boundary = payload.get("information_boundary", {})
    _require(
        boundary.get("object_ids_emitted") is False,
        "exclusion manifest emitted object identities",
    )


def _validate_source_artifact(payload: Mapping[str, Any]) -> None:
    if "artifact_sha256" in payload:
        _require(
            payload["artifact_sha256"] == canonical_artifact_sha256(payload),
            "source artifact checksum changed",
        )
        return
    if "result_sha256" in payload:
        stripped = dict(payload)
        stripped.pop("result_sha256", None)
        encoded = json.dumps(stripped, sort_keys=True, separators=(",", ":")).encode()
        _require(
            payload["result_sha256"] == hashlib.sha256(encoded).hexdigest(),
            "source result checksum changed",
        )
        return
    raise ValueError("source artifact has no canonical checksum")


def source_object_ids(payload: Mapping[str, Any]) -> set[str]:
    """Extract only the documented opened-object fields from known source artifacts."""

    kind = payload.get("artifact_kind")
    if kind == "Deform360TactileRegretGuardSourceDiagnostic":
        rows = payload.get("cross_fitted", {}).get("combined", {}).get("cases", [])
        object_ids = {str(row["object"]) for row in rows}
    elif kind in {
        "Deform360SelectiveVirtualSensingPredictionCohortSeal",
        "Deform360BiasAwareProspectiveV2CalibrationCohortSeal",
    }:
        object_ids = {str(row["object_id"]) for row in payload.get("cases", [])}
    else:
        raise ValueError(f"unsupported opened-source artifact: {kind}")
    _require(
        bool(object_ids)
        and all(_OBJECT_ID.fullmatch(value) is not None for value in object_ids),
        "opened-source object identities are malformed",
    )
    return object_ids


def build_exclusion_union(
    exclusion_manifests: Sequence[tuple[str | Path, Mapping[str, Any]]],
    opened_source_artifacts: Sequence[tuple[str | Path, Mapping[str, Any]]],
    *,
    additional_opened_object_ids: Sequence[str] = (),
    additional_source_artifacts: Sequence[str | Path] = (),
    owner: str,
) -> dict[str, Any]:
    """Union independent hashes with prior opened-source identities."""

    _require(bool(owner), "exclusion owner is empty")
    _require(bool(exclusion_manifests), "independent exclusion manifests are required")
    external_hashes: set[str] = set()
    input_manifests = []
    source_digests = []
    for path_value, payload in exclusion_manifests:
        validate_exclusion_manifest(payload)
        path = Path(path_value)
        external_hashes.update(str(value) for value in payload["object_hashes"])
        digest = file_sha256(path)
        source_digests.append(digest)
        input_manifests.append(
            {
                "owner": str(payload["owner"]),
                "object_hash_count": len(payload["object_hashes"]),
                "exclusion_sha256": str(payload["exclusion_sha256"]),
                "file_sha256": digest,
            }
        )

    opened_ids = set(str(value) for value in additional_opened_object_ids)
    source_inputs = []
    for path_value, payload in opened_source_artifacts:
        _validate_source_artifact(payload)
        object_ids = source_object_ids(payload)
        opened_ids.update(object_ids)
        path = Path(path_value)
        digest = file_sha256(path)
        source_digests.append(digest)
        source_inputs.append(
            {
                "artifact_kind": str(payload["artifact_kind"]),
                "opened_object_count": len(object_ids),
                "file_sha256": digest,
            }
        )
    _require(
        all(_OBJECT_ID.fullmatch(value) is not None for value in opened_ids),
        "additional opened object identity is malformed",
    )
    for path_value in additional_source_artifacts:
        source_digests.append(file_sha256(path_value))

    local_hashes = {object_exclusion_hash(value) for value in opened_ids}
    union = sorted(external_hashes | local_hashes)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": EXCLUSION_KIND,
        "hash_namespace": HASH_NAMESPACE,
        "owner": owner,
        "object_hashes": union,
        "source_artifact_sha256s": sorted(set(source_digests)),
        "input_manifests": sorted(
            input_manifests,
            key=lambda row: str(row["exclusion_sha256"]),
        ),
        "opened_source_inputs": sorted(
            source_inputs,
            key=lambda row: str(row["file_sha256"]),
        ),
        "accounting": {
            "independent_hash_count": len(external_hashes),
            "opened_source_object_count": len(opened_ids),
            "new_opened_source_hash_count": len(local_hashes - external_hashes),
            "union_hash_count": len(union),
        },
        "information_boundary": {
            "fresh_target_artifact_read": False,
            "previously_opened_source_artifacts_read": True,
            "held_runtime_tree_accessed": False,
            "object_ids_emitted": False,
        },
    }
    payload["exclusion_sha256"] = _canonical_exclusion_sha256(payload)
    validate_exclusion_manifest(payload)
    return payload


__all__ = [
    "EXCLUSION_KIND",
    "HASH_NAMESPACE",
    "build_exclusion_union",
    "object_exclusion_hash",
    "source_object_ids",
    "validate_exclusion_manifest",
]
