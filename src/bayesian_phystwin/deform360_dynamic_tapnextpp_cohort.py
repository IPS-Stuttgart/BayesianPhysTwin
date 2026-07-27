"""Metadata-only cohort artifacts for the dynamic TAPNext++ study."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .deform360_object_exclusion import (
    file_sha256,
    load_object_exclusion_manifest,
)
from .tapnextpp_dynamic_multiview import PROTOCOL_ID

METADATA_PREFLIGHT_KIND = "Deform360DynamicTAPNextPPMetadataPreflight"
STAGING_QUEUE_KIND = "Deform360FreshSourceStagingQueue"
DATASET_REPOSITORY = "brownu/deform360"
DATASET_REVISION = "7fea8e20231a47641d1d2bc8791920ec4e62ec5e"
MORPHOLOGY_STRATA = ("sheet", "compact", "complex")
_OBJECT_ID = re.compile(r"^[0-9]{3}-[a-z0-9][a-z0-9-]*$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON artifact: {source}") from error
    _require(isinstance(payload, dict), "artifact root must be an object")
    return payload


def _canonical_sha256(
    payload: Mapping[str, Any],
    *,
    digest_key: str,
) -> str:
    canonical = dict(payload)
    canonical.pop(digest_key, None)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _require(not target.exists(), f"refusing to replace artifact: {target}")
    temporary = target.with_name(f".{target.name}.tmp")
    _require(not temporary.exists(), f"temporary artifact exists: {temporary}")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def _valid_digest(value: Any, *, length: int = 64) -> bool:
    pattern = _HEX40 if length == 40 else _HEX64
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def morphology_stratum(object_id: str) -> str:
    """Return the frozen public-name morphology stratum."""

    _require(_OBJECT_ID.fullmatch(object_id) is not None, "object ID is malformed")
    if "cloth" in object_id:
        return "sheet"
    prefix = int(object_id[:3])
    return "compact" if prefix < 138 else "complex"


def _object_order(object_id: str) -> tuple[int, str]:
    return int(object_id[:3]), object_id


def _validate_catalog(
    catalog_path: str | Path,
    *,
    expected_file_sha256: str,
    expected_catalog_sha256: str,
) -> list[dict[str, str]]:
    path = Path(catalog_path)
    _require(file_sha256(path) == expected_file_sha256, "public catalog file changed")
    artifact = _read_json(path)
    _require(
        artifact.get("catalog_sha256") == expected_catalog_sha256,
        "public catalog content binding changed",
    )
    rows = artifact.get("objects")
    _require(isinstance(rows, list) and bool(rows), "public catalog is empty")
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        _require(isinstance(row, Mapping), "public catalog row is malformed")
        object_id = row.get("object_id")
        oid = row.get("oid")
        _require(
            isinstance(object_id, str)
            and _OBJECT_ID.fullmatch(object_id) is not None
            and object_id not in seen,
            "public object identity is malformed or repeated",
        )
        _require(_valid_digest(oid, length=40), "public catalog OID is malformed")
        seen.add(object_id)
        normalized.append({"object_id": object_id, "catalog_oid": str(oid)})
    return normalized


def build_metadata_preflight(
    output_path: str | Path,
    *,
    protocol_path: str | Path,
    catalog_path: str | Path,
    exclusion_path: str | Path,
    metadata_root: str | Path,
) -> dict[str, Any]:
    """Validate pinned public metadata without opening episode media."""

    protocol_source = Path(protocol_path)
    protocol = _read_json(protocol_source)
    _require(protocol.get("protocol_id") == PROTOCOL_ID, "protocol ID changed")
    source = protocol.get("data_source")
    _require(isinstance(source, Mapping), "protocol data source is missing")
    _require(
        source.get("repository") == DATASET_REPOSITORY
        and source.get("dataset_revision") == DATASET_REVISION,
        "dataset binding changed",
    )
    catalog = _validate_catalog(
        catalog_path,
        expected_file_sha256=str(source.get("public_catalog_file_sha256")),
        expected_catalog_sha256=str(source.get("public_catalog_sha256")),
    )
    exclusion = load_object_exclusion_manifest(exclusion_path)
    boundary = protocol.get("fresh_object_boundary")
    _require(isinstance(boundary, Mapping), "fresh-object boundary is missing")
    _require(
        exclusion["exclusion_sha256"] == boundary.get("exclusion_sha256")
        and file_sha256(exclusion_path) == boundary.get("exclusion_file_sha256"),
        "fresh-object exclusion binding changed",
    )

    root = Path(metadata_root)
    rows: list[dict[str, Any]] = []
    excluded = set(exclusion["object_hashes"])
    for catalog_row in catalog:
        object_id = catalog_row["object_id"]
        object_hash = hashlib.sha256(
            b"deform360-fresh-object-exclusion-v1\0"
            + object_id.encode("utf-8")
        ).hexdigest()
        if object_hash in excluded:
            continue
        metadata_path = root / f"{object_id}.json"
        reasons: list[str] = []
        try:
            metadata = _read_json(metadata_path)
        except ValueError:
            metadata = {}
            reasons.append("metadata-unreadable")
        descriptive_label = metadata.get("object")
        if not isinstance(descriptive_label, str) or not descriptive_label:
            reasons.append("metadata-object-label")
        sequences = metadata.get("sequences")
        if not isinstance(sequences, Mapping) or not sequences:
            sequences = {}
            reasons.append("metadata-sequences")
        episode_zero = sequences.get("0")
        if not isinstance(episode_zero, Mapping):
            episode_zero = {}
            reasons.append("episode-zero-missing")
        if any(
            not isinstance(record, Mapping)
            or record.get("bimanual") not in {"yes", "no"}
            for record in sequences.values()
        ):
            reasons.append("invalid-bimanual-enum")
        metadata_digest = (
            file_sha256(metadata_path) if metadata_path.is_file() else None
        )
        rows.append(
            {
                **catalog_row,
                "metadata_sha256": metadata_digest,
                "metadata_object_label": descriptive_label,
                "sequence_count": len(sequences),
                "episode_zero_bimanual": episode_zero.get("bimanual"),
                "accepted": not reasons,
                "rejection_reasons": sorted(set(reasons)),
            }
        )
    rows.sort(key=lambda row: _object_order(str(row["object_id"])))
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": METADATA_PREFLIGHT_KIND,
        "protocol_id": PROTOCOL_ID,
        "dataset": {
            "repository": DATASET_REPOSITORY,
            "revision": DATASET_REVISION,
        },
        "bindings": {
            "protocol_file_sha256": file_sha256(protocol_source),
            "catalog_file_sha256": file_sha256(catalog_path),
            "exclusion_file_sha256": file_sha256(exclusion_path),
            "exclusion_sha256": exclusion["exclusion_sha256"],
        },
        "counts": {
            "nonexcluded": len(rows),
            "accepted": sum(bool(row["accepted"]) for row in rows),
            "rejected": sum(not bool(row["accepted"]) for row in rows),
        },
        "objects": rows,
        "information_boundary": {
            "public_metadata_only": True,
            "episode_media_read": False,
            "future_geometry_read": False,
            "target_metric_read": False,
        },
    }
    payload["preflight_sha256"] = _canonical_sha256(
        payload,
        digest_key="preflight_sha256",
    )
    normalized = validate_metadata_preflight(payload)
    _write_json_atomic(output_path, normalized)
    return normalized


def validate_metadata_preflight(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one metadata-only preflight artifact."""

    normalized = json.loads(json.dumps(dict(payload), allow_nan=False))
    _require(normalized.get("schema_version") == 1, "preflight schema changed")
    _require(
        normalized.get("artifact_kind") == METADATA_PREFLIGHT_KIND,
        "preflight artifact kind changed",
    )
    _require(normalized.get("protocol_id") == PROTOCOL_ID, "protocol ID changed")
    rows = normalized.get("objects")
    _require(isinstance(rows, list) and bool(rows), "preflight objects are empty")
    identities = [row.get("object_id") for row in rows if isinstance(row, Mapping)]
    _require(
        len(identities) == len(rows)
        and len(set(identities)) == len(rows)
        and identities == sorted(identities, key=_object_order),
        "preflight object ordering or identity changed",
    )
    counts = normalized.get("counts")
    _require(isinstance(counts, Mapping), "preflight counts are missing")
    _require(counts.get("nonexcluded") == len(rows), "preflight count changed")
    _require(
        counts.get("accepted") == sum(bool(row.get("accepted")) for row in rows),
        "accepted preflight count changed",
    )
    _require(
        counts.get("rejected") == sum(not bool(row.get("accepted")) for row in rows),
        "rejected preflight count changed",
    )
    boundary = normalized.get("information_boundary")
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("public_metadata_only") is True
        and boundary.get("episode_media_read") is False
        and boundary.get("future_geometry_read") is False
        and boundary.get("target_metric_read") is False,
        "preflight crossed its information boundary",
    )
    _require(
        normalized.get("preflight_sha256")
        == _canonical_sha256(normalized, digest_key="preflight_sha256"),
        "preflight checksum changed",
    )
    return normalized


def load_metadata_preflight(path: str | Path) -> dict[str, Any]:
    """Load and validate one metadata-only preflight artifact."""

    return validate_metadata_preflight(_read_json(path))


def selected_staging_candidates(
    preflight: Mapping[str, Any],
    *,
    count_per_stratum: int = 12,
) -> list[dict[str, Any]]:
    """Apply the frozen deterministic morphology rule."""

    artifact = validate_metadata_preflight(preflight)
    _require(count_per_stratum >= 1, "stratum count must be positive")
    buckets: dict[str, list[dict[str, Any]]] = {
        name: [] for name in MORPHOLOGY_STRATA
    }
    for row in artifact["objects"]:
        if not row["accepted"]:
            continue
        buckets[morphology_stratum(row["object_id"])].append(row)
    for rows in buckets.values():
        rows.sort(key=lambda row: _object_order(row["object_id"]))
        _require(
            len(rows) >= count_per_stratum,
            "metadata preflight has too few objects in a morphology stratum",
        )

    candidates: list[dict[str, Any]] = []
    for index in range(count_per_stratum):
        for stratum in MORPHOLOGY_STRATA:
            row = buckets[stratum][index]
            candidates.append(
                {
                    "queue_rank": len(candidates) + 1,
                    "object_id": row["object_id"],
                    "catalog_oid": row["catalog_oid"],
                    "episode_id": 0,
                    "category": stratum,
                    "metadata_sha256": row["metadata_sha256"],
                    "bimanual": row["episode_zero_bimanual"],
                }
            )
    return candidates


def build_staging_queue(
    output_path: str | Path,
    *,
    protocol_path: str | Path,
    preflight_path: str | Path,
    implementation_commit: str,
) -> dict[str, Any]:
    """Seal the 36-object source-admission queue before episode payloads."""

    _require(
        _valid_digest(implementation_commit, length=40),
        "implementation commit is malformed",
    )
    protocol_source = Path(protocol_path)
    protocol = _read_json(protocol_source)
    _require(protocol.get("protocol_id") == PROTOCOL_ID, "protocol ID changed")
    expected_count = int(protocol["data_source"]["staged_candidate_count"])
    preflight_source = Path(preflight_path)
    preflight = load_metadata_preflight(preflight_source)
    candidates = selected_staging_candidates(preflight)
    _require(len(candidates) == expected_count, "staged candidate count changed")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": STAGING_QUEUE_KIND,
        "protocol_id": PROTOCOL_ID,
        "status": "source_only_locked_before_payload",
        "frozen_method": {
            "commit": implementation_commit,
            "arm": "physics_guided_dynamic_tapnextpp_bias_aware_guarded_update",
        },
        "source_lock": {
            "implementation_commit": implementation_commit,
            "config_sha256": file_sha256(protocol_source),
        },
        "input_bindings": {
            "metadata_preflight": {
                "artifact": str(preflight_source),
                "preflight_sha256": preflight["preflight_sha256"],
                "file_sha256": file_sha256(preflight_source),
            },
            "dataset_revision": DATASET_REVISION,
            "exclusion_sha256": preflight["bindings"]["exclusion_sha256"],
        },
        "queue_contract": {
            "staged_candidate_count": expected_count,
            "required_admitted_count": 20,
            "source_object_count": 8,
            "sealed_target_object_count": 12,
            "candidate_episode": 0,
            "selection": (
                "first 12 accepted objects per frozen morphology stratum, "
                "interleaved sheet/compact/complex"
            ),
            "implicit_replacement": False,
            "insufficient_admissions": (
                "stop and version a reserve queue before reading more payload"
            ),
        },
        "candidates": candidates,
        "stratum_counts": {
            stratum: sum(row["category"] == stratum for row in candidates)
            for stratum in MORPHOLOGY_STRATA
        },
        "information_boundary": {
            "public_metadata_read": True,
            "episode_media_read_before_queue_lock": False,
            "processed_geometry_read_before_queue_lock": False,
            "future_object_positions_deserialized": False,
            "outcome_or_metric_read": False,
            "held_v8_target_query_score_barrier_or_outcome_access": False,
        },
    }
    payload["queue_sha256"] = _canonical_sha256(
        payload,
        digest_key="queue_sha256",
    )
    normalized = validate_staging_queue(payload, preflight=preflight)
    _write_json_atomic(output_path, normalized)
    return normalized


def validate_staging_queue(
    payload: Mapping[str, Any],
    *,
    preflight: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a locked staging queue and optionally reproduce selection."""

    normalized = json.loads(json.dumps(dict(payload), allow_nan=False))
    _require(normalized.get("schema_version") == 1, "queue schema changed")
    _require(
        normalized.get("artifact_kind") == STAGING_QUEUE_KIND,
        "queue artifact kind changed",
    )
    _require(normalized.get("protocol_id") == PROTOCOL_ID, "protocol ID changed")
    _require(
        normalized.get("status") == "source_only_locked_before_payload",
        "queue is not locked before payload",
    )
    candidates = normalized.get("candidates")
    _require(isinstance(candidates, list) and len(candidates) == 36, "queue size changed")
    _require(
        [row.get("queue_rank") for row in candidates] == list(range(1, 37)),
        "queue ranks changed",
    )
    identities = [row.get("object_id") for row in candidates]
    _require(len(set(identities)) == len(identities), "queue repeats an object")
    _require(
        normalized.get("stratum_counts")
        == {stratum: 12 for stratum in MORPHOLOGY_STRATA},
        "queue stratum counts changed",
    )
    boundary = normalized.get("information_boundary")
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("episode_media_read_before_queue_lock") is False
        and boundary.get("processed_geometry_read_before_queue_lock") is False
        and boundary.get("future_object_positions_deserialized") is False
        and boundary.get("outcome_or_metric_read") is False
        and boundary.get("held_v8_target_query_score_barrier_or_outcome_access")
        is False,
        "queue crossed its information boundary",
    )
    _require(
        normalized.get("queue_sha256")
        == _canonical_sha256(normalized, digest_key="queue_sha256"),
        "queue checksum changed",
    )
    if preflight is not None:
        _require(
            candidates == selected_staging_candidates(preflight),
            "queue differs from frozen metadata-only selection",
        )
    return normalized


def load_staging_queue(
    path: str | Path,
    *,
    preflight_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load and validate one staging queue."""

    preflight = (
        None if preflight_path is None else load_metadata_preflight(preflight_path)
    )
    return validate_staging_queue(_read_json(path), preflight=preflight)


__all__ = [
    "DATASET_REPOSITORY",
    "DATASET_REVISION",
    "METADATA_PREFLIGHT_KIND",
    "MORPHOLOGY_STRATA",
    "STAGING_QUEUE_KIND",
    "build_metadata_preflight",
    "build_staging_queue",
    "load_metadata_preflight",
    "load_staging_queue",
    "morphology_stratum",
    "selected_staging_candidates",
    "validate_metadata_preflight",
    "validate_staging_queue",
]
