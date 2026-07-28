"""Metadata-only cohort artifacts for the dynamic TAPNext++ study."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .deform360_dynamic_tapnextpp_source_processing import (
    load_dynamic_source_processing_protocol,
    load_dynamic_source_processing_runtime_amendment,
)
from .deform360_fresh_source_lock import (
    ADMISSION_KIND,
    FreshSourceAdmissionConfig,
    validate_fresh_source_admission,
)
from .deform360_object_exclusion import (
    file_sha256,
    load_object_exclusion_manifest,
)
from .tapnextpp_dynamic_multiview import PROTOCOL_ID

METADATA_PREFLIGHT_KIND = "Deform360DynamicTAPNextPPMetadataPreflight"
STAGING_QUEUE_KIND = "Deform360FreshSourceStagingQueue"
TERMINAL_DISPOSITION_KIND = "Deform360DynamicTAPNextPPTerminalDisposition"
COHORT_LOCK_KIND = "Deform360DynamicTAPNextPPCohortLock"
DATASET_REPOSITORY = "brownu/deform360"
DATASET_REVISION = "7fea8e20231a47641d1d2bc8791920ec4e62ec5e"
MORPHOLOGY_STRATA = ("sheet", "compact", "complex")
SOURCE_OBJECT_COUNT = 8
TARGET_OBJECT_COUNT = 12
SELECTED_STRATUM_COUNTS = {"sheet": 7, "compact": 7, "complex": 6}
SOURCE_STRATUM_COUNTS = {"sheet": 3, "compact": 3, "complex": 2}
TARGET_STRATUM_COUNTS = {"sheet": 4, "compact": 4, "complex": 4}
_OBJECT_ID = re.compile(r"^[0-9]{3}-[a-z0-9][a-z0-9-]*$")
_REASON_CODE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_TERMINAL_REASON_CODES = {
    "mask-technical-failure",
    "missing-required-camera",
    "runtime-backend-unavailable",
    "source-admission-rejected",
    "source-processing-failure",
    "window-stage-failure",
}


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


def _dynamic_admission_config(
    processing_protocol: Mapping[str, Any],
) -> FreshSourceAdmissionConfig:
    values = processing_protocol.get("source_admission")
    _require(isinstance(values, Mapping), "source-admission config is missing")
    _require(
        values.get("future_geometry_deserialized_for_admission") is False,
        "source admission may deserialize future geometry",
    )
    return FreshSourceAdmissionConfig(
        minimum_camera_count=int(values["minimum_camera_count"]),
        minimum_point_count=int(values["minimum_point_count"]),
        maximum_point_count=int(values["maximum_point_count"]),
        required_frame_count=int(values["required_frame_count"]),
        update_frames=tuple(int(value) for value in values["update_frames"]),
        minimum_test_frame_count=int(values["minimum_test_frame_count"]),
    )


def _queue_identity(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row["object_id"]), int(row["episode_id"])


def _object_hash(object_id: str) -> str:
    return hashlib.sha256(
        b"deform360-fresh-object-exclusion-v1\0"
        + object_id.encode("utf-8")
    ).hexdigest()


def _case_hash(object_id: str, episode_id: int) -> str:
    return hashlib.sha256(
        b"deform360-dynamic-tapnextpp-case-v1\0"
        + f"{object_id}\0{episode_id}".encode()
    ).hexdigest()


def build_terminal_disposition(
    output_path: str | Path,
    *,
    queue_path: str | Path,
    queue_rank: int,
    stage: str,
    reason_code: str,
    evidence_path: str | Path,
    producer_commit: str,
) -> dict[str, Any]:
    """Seal one source-only technical failure without substituting a case."""

    queue_source = Path(queue_path)
    queue = load_staging_queue(queue_source)
    _require(1 <= queue_rank <= len(queue["candidates"]), "queue rank is invalid")
    _require(
        reason_code in _TERMINAL_REASON_CODES
        and _REASON_CODE.fullmatch(reason_code) is not None,
        "terminal reason code is not preregistered",
    )
    _require(stage in {"mask", "source_processing", "window_stage"}, "stage is invalid")
    _require(_valid_digest(producer_commit, length=40), "producer commit is malformed")
    evidence = Path(evidence_path)
    _require(evidence.is_file(), "terminal evidence file is missing")
    row = queue["candidates"][queue_rank - 1]
    evidence_result_sha256: str | None = None
    if evidence.suffix == ".json":
        payload = _read_json(evidence)
        _require(
            payload.get("object_id") == row["object_id"]
            and payload.get("episode_id") == row["episode_id"]
            and payload.get("category") == row["category"]
            and payload.get("queue_rank") == row["queue_rank"],
            "terminal JSON evidence belongs to another queue entry",
        )
        _require(
            payload.get("status") in {"source_rejected", "technical_failure"},
            "terminal JSON evidence is not a terminal disposition",
        )
        _require(
            payload.get("result_sha256")
            == _canonical_sha256(payload, digest_key="result_sha256"),
            "terminal JSON evidence checksum changed",
        )
        boundary = payload.get("information_boundary")
        _require(
            isinstance(boundary, Mapping)
            and boundary.get("target_metric_read") is False
            and boundary.get(
                "held_v8_target_query_score_barrier_or_outcome_access"
            )
            is False,
            "terminal JSON evidence crossed the target boundary",
        )
        evidence_result_sha256 = str(payload["result_sha256"])
    else:
        _require(
            stage == "window_stage",
            "non-JSON evidence is allowed only for a window-stage failure",
        )
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": TERMINAL_DISPOSITION_KIND,
        "protocol_id": PROTOCOL_ID,
        "status": "technical_failure",
        "queue_rank": row["queue_rank"],
        "object_id": row["object_id"],
        "episode_id": row["episode_id"],
        "category": row["category"],
        "object_hash": _object_hash(str(row["object_id"])),
        "case_hash": _case_hash(str(row["object_id"]), int(row["episode_id"])),
        "stage": stage,
        "reason_code": reason_code,
        "producer_commit": producer_commit,
        "evidence": {
            "basename": evidence.name,
            "file_sha256": file_sha256(evidence),
            "result_sha256": evidence_result_sha256,
        },
        "bindings": {
            "queue_sha256": queue["queue_sha256"],
            "queue_file_sha256": file_sha256(queue_source),
        },
        "information_boundary": {
            "source_evidence_only": True,
            "future_object_positions_deserialized": False,
            "provider_outcome_or_metric_read": False,
            "held_v8_target_query_score_barrier_or_outcome_access": False,
            "implicit_replacement": False,
        },
    }
    artifact["disposition_sha256"] = _canonical_sha256(
        artifact,
        digest_key="disposition_sha256",
    )
    normalized = validate_terminal_disposition(artifact, queue=queue)
    _write_json_atomic(output_path, normalized)
    return normalized


def validate_terminal_disposition(
    artifact: Mapping[str, Any],
    *,
    queue: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one terminal source-only queue disposition."""

    normalized = json.loads(json.dumps(dict(artifact), allow_nan=False))
    _require(normalized.get("schema_version") == 1, "disposition schema changed")
    _require(
        normalized.get("artifact_kind") == TERMINAL_DISPOSITION_KIND,
        "disposition artifact kind changed",
    )
    _require(normalized.get("protocol_id") == PROTOCOL_ID, "protocol ID changed")
    _require(
        normalized.get("status") == "technical_failure",
        "disposition is not terminal",
    )
    queue_rank = normalized.get("queue_rank")
    object_id = normalized.get("object_id")
    episode_id = normalized.get("episode_id")
    category = normalized.get("category")
    _require(
        isinstance(queue_rank, int)
        and not isinstance(queue_rank, bool)
        and queue_rank >= 1
        and isinstance(object_id, str)
        and _OBJECT_ID.fullmatch(object_id) is not None
        and isinstance(episode_id, int)
        and not isinstance(episode_id, bool)
        and episode_id >= 0
        and category in MORPHOLOGY_STRATA,
        "disposition identity is malformed",
    )
    _require(
        normalized.get("object_hash") == _object_hash(object_id)
        and normalized.get("case_hash") == _case_hash(object_id, episode_id),
        "disposition identity hashes changed",
    )
    _require(
        normalized.get("stage") in {"mask", "source_processing", "window_stage"}
        and normalized.get("reason_code") in _TERMINAL_REASON_CODES,
        "disposition reason is not preregistered",
    )
    _require(
        _valid_digest(normalized.get("producer_commit"), length=40),
        "disposition producer commit is malformed",
    )
    evidence = normalized.get("evidence")
    _require(
        isinstance(evidence, Mapping)
        and isinstance(evidence.get("basename"), str)
        and bool(evidence["basename"])
        and _valid_digest(evidence.get("file_sha256"))
        and (
            evidence.get("result_sha256") is None
            or _valid_digest(evidence.get("result_sha256"))
        ),
        "disposition evidence binding is malformed",
    )
    boundary = normalized.get("information_boundary")
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("source_evidence_only") is True
        and boundary.get("future_object_positions_deserialized") is False
        and boundary.get("provider_outcome_or_metric_read") is False
        and boundary.get(
            "held_v8_target_query_score_barrier_or_outcome_access"
        )
        is False
        and boundary.get("implicit_replacement") is False,
        "disposition crossed its information boundary",
    )
    _require(
        normalized.get("disposition_sha256")
        == _canonical_sha256(normalized, digest_key="disposition_sha256"),
        "disposition checksum changed",
    )
    bindings = normalized.get("bindings")
    _require(
        isinstance(bindings, Mapping)
        and _valid_digest(bindings.get("queue_sha256"))
        and _valid_digest(bindings.get("queue_file_sha256")),
        "disposition queue binding is malformed",
    )
    if queue is not None:
        validated_queue = validate_staging_queue(queue)
        _require(
            queue_rank <= len(validated_queue["candidates"]),
            "disposition queue rank is outside the queue",
        )
        row = validated_queue["candidates"][queue_rank - 1]
        _require(
            _queue_identity(row) == (object_id, episode_id)
            and row["category"] == category
            and bindings["queue_sha256"] == validated_queue["queue_sha256"],
            "disposition differs from the locked queue",
        )
    return normalized


def load_terminal_disposition(
    path: str | Path,
    *,
    queue: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and validate one terminal source-only disposition."""

    return validate_terminal_disposition(_read_json(path), queue=queue)


def _selected_admission_rows(
    ledger: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {
        stratum: [] for stratum in MORPHOLOGY_STRATA
    }
    for row in ledger:
        if row["status"] == "admitted":
            buckets[str(row["category"])].append(dict(row))
    for stratum, required in SELECTED_STRATUM_COUNTS.items():
        buckets[stratum].sort(key=lambda row: int(row["queue_rank"]))
        _require(
            len(buckets[stratum]) >= required,
            f"insufficient admitted {stratum} objects",
        )
        buckets[stratum] = buckets[stratum][:required]
    selected: list[dict[str, Any]] = []
    for index in range(max(SELECTED_STRATUM_COUNTS.values())):
        for stratum in MORPHOLOGY_STRATA:
            if index < SELECTED_STRATUM_COUNTS[stratum]:
                selected.append(buckets[stratum][index])
    _require(
        len(selected) == SOURCE_OBJECT_COUNT + TARGET_OBJECT_COUNT,
        "selected cohort size changed",
    )
    return selected


def _cohort_case(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "queue_rank": row["queue_rank"],
        "case": row["case"],
        "object_id": row["object_id"],
        "episode_id": row["episode_id"],
        "category": row["category"],
        "object_hash": row["object_hash"],
        "case_hash": row["case_hash"],
        "admission_sha256": row["artifact_sha256"],
    }


def build_dynamic_provider_cohort_lock(
    output_path: str | Path,
    *,
    protocol_path: str | Path,
    source_evaluation_protocol_path: str | Path,
    queue_path: str | Path,
    processing_protocol_path: str | Path,
    runtime_amendment_path: str | Path,
    admission_paths: Sequence[str | Path],
    terminal_disposition_paths: Sequence[str | Path],
    provider_commit: str,
    source_processing_commit: str,
    cohort_lock_builder_commit: str,
) -> dict[str, Any]:
    """Lock source and target together from a complete source-only ledger."""

    _require(_valid_digest(provider_commit, length=40), "provider commit is malformed")
    _require(
        _valid_digest(source_processing_commit, length=40),
        "source-processing commit is malformed",
    )
    _require(
        _valid_digest(cohort_lock_builder_commit, length=40),
        "cohort-lock builder commit is malformed",
    )
    protocol_source = Path(protocol_path)
    protocol = _read_json(protocol_source)
    _require(protocol.get("protocol_id") == PROTOCOL_ID, "protocol ID changed")
    source_evaluation_source = Path(source_evaluation_protocol_path)
    source_evaluation = _read_json(source_evaluation_source)
    _require(
        source_evaluation.get("artifact_kind")
        == "Deform360DynamicTAPNextPPSourceEvaluationProtocol"
        and source_evaluation.get("protocol_id") == PROTOCOL_ID
        and source_evaluation.get("status")
        == "locked-before-cohort-and-provider-outcomes",
        "source evaluation protocol is incompatible",
    )
    queue_source = Path(queue_path)
    queue = load_staging_queue(queue_source)
    processing_source = Path(processing_protocol_path)
    processing = load_dynamic_source_processing_protocol(processing_source)
    runtime_source = Path(runtime_amendment_path)
    runtime = load_dynamic_source_processing_runtime_amendment(
        runtime_source,
        parent_protocol_path=processing_source,
    )
    admission_config = _dynamic_admission_config(processing)
    by_identity = {
        _queue_identity(row): row for row in queue["candidates"]
    }
    ledger: dict[tuple[str, int], dict[str, Any]] = {}
    for source_path in admission_paths:
        source = Path(source_path)
        admission = _read_json(source)
        validate_fresh_source_admission(
            admission,
            expected_config=admission_config,
        )
        identity = (str(admission["object_id"]), int(admission["episode_id"]))
        _require(identity in by_identity, "admission is outside the locked queue")
        row = by_identity[identity]
        _require(
            admission["category"] == row["category"],
            "admission category differs from the queue",
        )
        _require(identity not in ledger, "queue entry has multiple dispositions")
        ledger[identity] = {
            "queue_rank": row["queue_rank"],
            "case": admission["case"],
            "object_id": admission["object_id"],
            "episode_id": admission["episode_id"],
            "category": admission["category"],
            "object_hash": _object_hash(str(admission["object_id"])),
            "case_hash": _case_hash(
                str(admission["object_id"]),
                int(admission["episode_id"]),
            ),
            "status": "admitted" if admission["accepted"] else "source_rejected",
            "artifact_kind": ADMISSION_KIND,
            "artifact_sha256": admission["admission_sha256"],
            "artifact_file_sha256": file_sha256(source),
        }
    for source_path in terminal_disposition_paths:
        source = Path(source_path)
        disposition = load_terminal_disposition(source, queue=queue)
        identity = (
            str(disposition["object_id"]),
            int(disposition["episode_id"]),
        )
        _require(identity not in ledger, "queue entry has multiple dispositions")
        ledger[identity] = {
            "queue_rank": disposition["queue_rank"],
            "case": f"{identity[0]}-ep{identity[1]:04d}",
            "object_id": identity[0],
            "episode_id": identity[1],
            "category": disposition["category"],
            "object_hash": disposition["object_hash"],
            "case_hash": disposition["case_hash"],
            "status": "technical_failure",
            "artifact_kind": TERMINAL_DISPOSITION_KIND,
            "artifact_sha256": disposition["disposition_sha256"],
            "artifact_file_sha256": file_sha256(source),
        }
    _require(
        set(ledger) == set(by_identity),
        "source disposition ledger is incomplete",
    )
    ordered_ledger = sorted(ledger.values(), key=lambda row: int(row["queue_rank"]))
    _require(
        [row["queue_rank"] for row in ordered_ledger] == list(range(1, 37)),
        "source disposition ledger ordering changed",
    )
    selected = _selected_admission_rows(ordered_ledger)
    source_cases = [_cohort_case(row) for row in selected[:SOURCE_OBJECT_COUNT]]
    target_cases = [_cohort_case(row) for row in selected[SOURCE_OBJECT_COUNT:]]
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": COHORT_LOCK_KIND,
        "protocol_id": PROTOCOL_ID,
        "status": "locked_before_provider_outcomes",
        "source_cases": source_cases,
        "sealed_target_cases": target_cases,
        "disposition_ledger": ordered_ledger,
        "counts": {
            "queued": len(ordered_ledger),
            "admitted": sum(row["status"] == "admitted" for row in ordered_ledger),
            "source_rejected": sum(
                row["status"] == "source_rejected" for row in ordered_ledger
            ),
            "technical_failure": sum(
                row["status"] == "technical_failure" for row in ordered_ledger
            ),
            "selected_source": len(source_cases),
            "selected_target": len(target_cases),
        },
        "stratum_counts": {
            "selected": SELECTED_STRATUM_COUNTS,
            "source": SOURCE_STRATUM_COUNTS,
            "target": TARGET_STRATUM_COUNTS,
        },
        "selection_rule": (
            "take the first admitted 7 sheet, 7 compact, and 6 complex queue "
            "entries; interleave sheet/compact/complex; assign the first 8 to "
            "source and the remaining 12 to the sealed target"
        ),
        "bindings": {
            "provider_protocol_file_sha256": file_sha256(protocol_source),
            "source_evaluation_protocol_file_sha256": file_sha256(
                source_evaluation_source
            ),
            "staging_queue_sha256": queue["queue_sha256"],
            "staging_queue_file_sha256": file_sha256(queue_source),
            "source_processing_protocol_sha256": processing["config_sha256"],
            "source_processing_protocol_file_sha256": file_sha256(
                processing_source
            ),
            "runtime_amendment_sha256": runtime["config_sha256"],
            "runtime_amendment_file_sha256": file_sha256(runtime_source),
            "provider_commit": provider_commit,
            "source_processing_commit": source_processing_commit,
            "cohort_lock_builder_commit": cohort_lock_builder_commit,
        },
        "information_boundary": {
            "source_admission_and_terminal_evidence_only": True,
            "future_object_positions_deserialized": False,
            "provider_outcome_or_metric_read": False,
            "target_outcome_or_metric_read": False,
            "held_v8_target_query_score_barrier_or_outcome_access": False,
            "cohort_membership_uses_outcomes": False,
            "implicit_replacement": False,
        },
    }
    artifact["cohort_lock_sha256"] = _canonical_sha256(
        artifact,
        digest_key="cohort_lock_sha256",
    )
    normalized = validate_dynamic_provider_cohort_lock(artifact)
    _write_json_atomic(output_path, normalized)
    return normalized


def validate_dynamic_provider_cohort_lock(
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a complete source/target lock without reading outcomes."""

    normalized = json.loads(json.dumps(dict(artifact), allow_nan=False))
    _require(normalized.get("schema_version") == 1, "cohort schema changed")
    _require(
        normalized.get("artifact_kind") == COHORT_LOCK_KIND,
        "cohort artifact kind changed",
    )
    _require(normalized.get("protocol_id") == PROTOCOL_ID, "protocol ID changed")
    _require(
        normalized.get("status") == "locked_before_provider_outcomes",
        "cohort was not locked before provider outcomes",
    )
    ledger = normalized.get("disposition_ledger")
    _require(
        isinstance(ledger, list)
        and len(ledger) == 36
        and [row.get("queue_rank") for row in ledger] == list(range(1, 37)),
        "cohort disposition ledger is incomplete",
    )
    identities = [
        (row.get("object_id"), row.get("episode_id")) for row in ledger
    ]
    _require(len(set(identities)) == 36, "cohort ledger repeats a queue entry")
    for row in ledger:
        _require(
            isinstance(row, Mapping)
            and row.get("category") in MORPHOLOGY_STRATA
            and row.get("status")
            in {"admitted", "source_rejected", "technical_failure"}
            and _valid_digest(row.get("object_hash"))
            and _valid_digest(row.get("case_hash"))
            and _valid_digest(row.get("artifact_sha256"))
            and _valid_digest(row.get("artifact_file_sha256")),
            "cohort disposition row is malformed",
        )
    selected = _selected_admission_rows(ledger)
    expected_source = [_cohort_case(row) for row in selected[:SOURCE_OBJECT_COUNT]]
    expected_target = [_cohort_case(row) for row in selected[SOURCE_OBJECT_COUNT:]]
    _require(
        normalized.get("source_cases") == expected_source
        and normalized.get("sealed_target_cases") == expected_target,
        "cohort partitions differ from the frozen selection",
    )
    _require(
        normalized.get("stratum_counts")
        == {
            "selected": SELECTED_STRATUM_COUNTS,
            "source": SOURCE_STRATUM_COUNTS,
            "target": TARGET_STRATUM_COUNTS,
        },
        "cohort stratum counts changed",
    )
    counts = normalized.get("counts")
    _require(
        isinstance(counts, Mapping)
        and counts.get("queued") == 36
        and counts.get("admitted")
        == sum(row["status"] == "admitted" for row in ledger)
        and counts.get("source_rejected")
        == sum(row["status"] == "source_rejected" for row in ledger)
        and counts.get("technical_failure")
        == sum(row["status"] == "technical_failure" for row in ledger)
        and counts.get("selected_source") == SOURCE_OBJECT_COUNT
        and counts.get("selected_target") == TARGET_OBJECT_COUNT,
        "cohort counts changed",
    )
    bindings = normalized.get("bindings")
    _require(
        isinstance(bindings, Mapping)
        and all(
            _valid_digest(bindings.get(key), length=40)
            for key in (
                "cohort_lock_builder_commit",
                "provider_commit",
                "source_processing_commit",
            )
        )
        and all(
            _valid_digest(bindings.get(key))
            for key in (
                "provider_protocol_file_sha256",
                "source_evaluation_protocol_file_sha256",
                "runtime_amendment_file_sha256",
                "runtime_amendment_sha256",
                "source_processing_protocol_file_sha256",
                "source_processing_protocol_sha256",
                "staging_queue_file_sha256",
                "staging_queue_sha256",
            )
        ),
        "cohort provenance bindings are malformed",
    )
    boundary = normalized.get("information_boundary")
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("source_admission_and_terminal_evidence_only") is True
        and boundary.get("future_object_positions_deserialized") is False
        and boundary.get("provider_outcome_or_metric_read") is False
        and boundary.get("target_outcome_or_metric_read") is False
        and boundary.get(
            "held_v8_target_query_score_barrier_or_outcome_access"
        )
        is False
        and boundary.get("cohort_membership_uses_outcomes") is False
        and boundary.get("implicit_replacement") is False,
        "cohort lock crossed its information boundary",
    )
    _require(
        normalized.get("cohort_lock_sha256")
        == _canonical_sha256(normalized, digest_key="cohort_lock_sha256"),
        "cohort lock checksum changed",
    )
    return normalized


def load_dynamic_provider_cohort_lock(path: str | Path) -> dict[str, Any]:
    """Load and validate one dynamic provider cohort lock."""

    return validate_dynamic_provider_cohort_lock(_read_json(path))


def dynamic_provider_case_record(
    cohort: Mapping[str, Any],
    *,
    object_id: str,
    episode_id: int,
    partition: str,
) -> dict[str, Any]:
    """Return one explicitly authorized source or sealed-target case."""

    validated = validate_dynamic_provider_cohort_lock(cohort)
    _require(partition in {"source", "target"}, "cohort partition is invalid")
    key = "source_cases" if partition == "source" else "sealed_target_cases"
    matches = [
        row
        for row in validated[key]
        if row["object_id"] == object_id and row["episode_id"] == episode_id
    ]
    _require(len(matches) == 1, "case is absent from the requested cohort partition")
    return dict(matches[0])


__all__ = [
    "COHORT_LOCK_KIND",
    "DATASET_REPOSITORY",
    "DATASET_REVISION",
    "METADATA_PREFLIGHT_KIND",
    "MORPHOLOGY_STRATA",
    "SELECTED_STRATUM_COUNTS",
    "SOURCE_OBJECT_COUNT",
    "SOURCE_STRATUM_COUNTS",
    "STAGING_QUEUE_KIND",
    "TARGET_OBJECT_COUNT",
    "TARGET_STRATUM_COUNTS",
    "TERMINAL_DISPOSITION_KIND",
    "build_dynamic_provider_cohort_lock",
    "build_metadata_preflight",
    "build_staging_queue",
    "build_terminal_disposition",
    "dynamic_provider_case_record",
    "load_dynamic_provider_cohort_lock",
    "load_metadata_preflight",
    "load_staging_queue",
    "load_terminal_disposition",
    "morphology_stratum",
    "selected_staging_candidates",
    "validate_dynamic_provider_cohort_lock",
    "validate_metadata_preflight",
    "validate_staging_queue",
    "validate_terminal_disposition",
]
