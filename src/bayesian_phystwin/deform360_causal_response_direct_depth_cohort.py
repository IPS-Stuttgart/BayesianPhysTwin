"""Outcome-blind fresh-source queue for V14 causal direct depth."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .deform360_dynamic_tapnextpp_cohort import load_metadata_preflight
from .deform360_object_exclusion import (
    file_sha256,
    load_object_exclusion_manifest,
)

PROTOCOL_ID = "deform360-causal-response-direct-depth-v14-source"
QUEUE_KIND = "Deform360FreshSourceStagingQueue"
QUEUE_CONTRACT = "deform360-causal-response-direct-depth-staging-v14"
DATASET_REPOSITORY = "brownu/deform360"
DATASET_REVISION = "7fea8e20231a47641d1d2bc8791920ec4e62ec5e"
EXCLUSION_OWNER = "bayesian-phystwin-v14-fresh-source-owner"
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


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("queue_sha256", None)
    return hashlib.sha256(
        b"deform360-causal-response-direct-depth-staging-v14\0"
        + json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _object_hash(object_id: str) -> str:
    return hashlib.sha256(
        b"deform360-fresh-object-exclusion-v1\0"
        + object_id.encode("utf-8")
    ).hexdigest()


def _object_order(object_id: str) -> tuple[int, str]:
    _require(_OBJECT_ID.fullmatch(object_id) is not None, "object ID is malformed")
    return int(object_id[:3]), object_id


def morphology_stratum(object_id: str) -> str:
    """Return the frozen public-name morphology stratum."""

    prefix, _ = _object_order(object_id)
    if "cloth" in object_id:
        return "sheet"
    return "compact" if prefix < 138 else "complex"


def _validate_catalog(path: str | Path) -> list[dict[str, str]]:
    payload = _read_json(path)
    _require(
        payload.get("artifact_kind") == "Deform360PublicObjectCatalogSnapshot",
        "public catalog kind changed",
    )
    rows = payload.get("objects")
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
            "public catalog object identity is malformed or repeated",
        )
        _require(
            isinstance(oid, str) and _HEX40.fullmatch(oid) is not None,
            "public catalog OID is malformed",
        )
        seen.add(object_id)
        normalized.append({"object_id": object_id, "catalog_oid": oid})
    _require(
        isinstance(payload.get("catalog_sha256"), str)
        and _HEX64.fullmatch(str(payload["catalog_sha256"])) is not None,
        "public catalog content digest is malformed",
    )
    return normalized


def _interleaved_candidates(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    buckets = {stratum: [] for stratum in MORPHOLOGY_STRATA}
    for row in rows:
        buckets[morphology_stratum(str(row["object_id"]))].append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda row: _object_order(str(row["object_id"])))

    ordered: list[dict[str, Any]] = []
    index = 0
    while any(index < len(buckets[stratum]) for stratum in MORPHOLOGY_STRATA):
        for stratum in MORPHOLOGY_STRATA:
            if index >= len(buckets[stratum]):
                continue
            row = buckets[stratum][index]
            ordered.append(
                {
                    "queue_rank": len(ordered) + 1,
                    "object_id": row["object_id"],
                    "episode_id": 0,
                    "category": stratum,
                    "catalog_oid": row["catalog_oid"],
                    "metadata_sha256": row["metadata_sha256"],
                    "bimanual": row["episode_zero_bimanual"],
                }
            )
        index += 1
    return ordered


def build_v14_staging_queue(
    output_path: str | Path,
    *,
    protocol_path: str | Path,
    catalog_path: str | Path,
    metadata_preflight_path: str | Path,
    exclusion_path: str | Path,
) -> dict[str, Any]:
    """Lock all fresh metadata-admissible candidates before episode payload access."""

    protocol_source = Path(protocol_path)
    protocol = _read_json(protocol_source)
    _require(protocol.get("protocol_id") == PROTOCOL_ID, "protocol ID changed")
    _require(
        protocol.get("status")
        == "implementation-locked-before-fresh-source-selection",
        "V14 protocol is not locked before source selection",
    )
    implementation_commit = protocol.get("implementation_commit")
    _require(
        isinstance(implementation_commit, str)
        and _HEX40.fullmatch(implementation_commit) is not None,
        "V14 implementation commit is malformed",
    )
    exclusion_source = Path(exclusion_path)
    exclusion = load_object_exclusion_manifest(exclusion_source)
    freshness = protocol.get("freshness_boundary")
    _require(isinstance(freshness, Mapping), "V14 freshness boundary is missing")
    _require(
        exclusion["owner"] == EXCLUSION_OWNER
        and exclusion["exclusion_sha256"]
        == freshness.get("manifest_exclusion_sha256")
        and file_sha256(exclusion_source)
        == freshness.get("manifest_file_sha256"),
        "V14 exclusion binding changed",
    )

    catalog_source = Path(catalog_path)
    catalog = _validate_catalog(catalog_source)
    metadata_source = Path(metadata_preflight_path)
    metadata = load_metadata_preflight(metadata_source)
    _require(
        metadata["bindings"]["catalog_file_sha256"] == file_sha256(catalog_source),
        "metadata cache binds another public catalog",
    )
    metadata_by_id = {
        str(row["object_id"]): row for row in metadata["objects"]
    }
    excluded = set(exclusion["object_hashes"])
    fresh_catalog = [
        row for row in catalog if _object_hash(row["object_id"]) not in excluded
    ]
    _require(
        fresh_catalog
        and all(row["object_id"] in metadata_by_id for row in fresh_catalog),
        "metadata cache does not cover every fresh catalog object",
    )

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for catalog_row in fresh_catalog:
        metadata_row = metadata_by_id[catalog_row["object_id"]]
        merged = {**catalog_row, **metadata_row}
        if metadata_row["accepted"]:
            accepted.append(merged)
        else:
            rejected.append(
                {
                    "object_hash": _object_hash(catalog_row["object_id"]),
                    "rejection_reasons": metadata_row["rejection_reasons"],
                }
            )
    candidates = _interleaved_candidates(accepted)
    _require(len(candidates) >= 12, "V14 has fewer than twelve fresh candidates")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": QUEUE_KIND,
        "contract": QUEUE_CONTRACT,
        "protocol_id": PROTOCOL_ID,
        "status": "source_only_locked_before_payload",
        "dataset": {
            "repository": DATASET_REPOSITORY,
            "revision": DATASET_REVISION,
        },
        "source_lock": {
            "implementation_commit": implementation_commit,
            "method_config_sha256": protocol["config_sha256"],
            "protocol_file_sha256": file_sha256(protocol_source),
        },
        "input_bindings": {
            "public_catalog": {
                "artifact": str(catalog_source),
                "catalog_sha256": _read_json(catalog_source)["catalog_sha256"],
                "file_sha256": file_sha256(catalog_source),
            },
            "metadata_cache": {
                "artifact": str(metadata_source),
                "preflight_sha256": metadata["preflight_sha256"],
                "file_sha256": file_sha256(metadata_source),
            },
            "fresh_object_exclusion": {
                "artifact": str(exclusion_source),
                "owner": exclusion["owner"],
                "exclusion_sha256": exclusion["exclusion_sha256"],
                "file_sha256": file_sha256(exclusion_source),
                "excluded_object_count": len(exclusion["object_hashes"]),
            },
        },
        "queue_contract": {
            "candidate_episode": 0,
            "candidate_count": len(candidates),
            "required_admitted_source_count": 12,
            "selection": (
                "round-robin public-name morphology order; within each stratum "
                "ascending public object ID"
            ),
            "final_source_panel": (
                "first 12 candidates with accepted V14 outcome-blind source "
                "preflights, in immutable queue order"
            ),
            "preflight_rejection_is_not_a_selected_case": True,
            "prediction_or_outcome_triggers_replacement": False,
            "replacement_after_source_lock": False,
            "insufficient_preflight_admissions": (
                "close the source campaign; do not alter queue or admission rules"
            ),
            "fold_assignment_after_admission": (
                "accepted source index modulo 3, producing four objects per fold"
            ),
        },
        "metadata_dispositions": {
            "fresh_catalog_count": len(fresh_catalog),
            "accepted_candidate_count": len(candidates),
            "rejected_count": len(rejected),
            "rejected_hash_only": sorted(
                rejected, key=lambda row: str(row["object_hash"])
            ),
        },
        "stratum_counts": {
            stratum: sum(
                candidate["category"] == stratum for candidate in candidates
            )
            for stratum in MORPHOLOGY_STRATA
        },
        "candidates": candidates,
        "information_boundary": {
            "public_catalog_and_metadata_read": True,
            "episode_media_read_before_queue_lock": False,
            "processed_geometry_read_before_queue_lock": False,
            "prefix_object_response_read_before_queue_lock": False,
            "future_object_positions_deserialized": False,
            "outcome_or_metric_read": False,
            "target_object_selected_or_read": False,
            "held_v8_target_query_score_barrier_or_outcome_access": False,
            "held_v8_plaintext_identity_read": False,
        },
    }
    payload["queue_sha256"] = _canonical_sha256(payload)
    normalized = validate_v14_staging_queue(payload)
    output = Path(output_path)
    _require(not output.exists(), f"refusing to replace V14 queue: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(normalized, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return normalized


def validate_v14_staging_queue(
    payload_or_path: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    """Validate V14 queue order, freshness, and information boundary."""

    payload = (
        _read_json(payload_or_path)
        if isinstance(payload_or_path, (str, Path))
        else json.loads(json.dumps(dict(payload_or_path), allow_nan=False))
    )
    _require(
        payload.get("schema_version") == 1
        and payload.get("artifact_kind") == QUEUE_KIND
        and payload.get("contract") == QUEUE_CONTRACT
        and payload.get("protocol_id") == PROTOCOL_ID
        and payload.get("status") == "source_only_locked_before_payload",
        "V14 staging queue kind, contract, or status changed",
    )
    candidates = payload.get("candidates")
    _require(
        isinstance(candidates, list) and len(candidates) >= 12,
        "V14 staging queue is incomplete",
    )
    _require(
        [row.get("queue_rank") for row in candidates]
        == list(range(1, len(candidates) + 1)),
        "V14 queue ranks changed",
    )
    identities = [row.get("object_id") for row in candidates]
    _require(
        len(set(identities)) == len(candidates)
        and all(
            isinstance(identity, str)
            and _OBJECT_ID.fullmatch(identity) is not None
            for identity in identities
        ),
        "V14 queue object identities are malformed or repeated",
    )
    _require(
        all(
            row.get("episode_id") == 0
            and row.get("category") == morphology_stratum(str(row["object_id"]))
            and isinstance(row.get("metadata_sha256"), str)
            and _HEX64.fullmatch(str(row["metadata_sha256"])) is not None
            and row.get("bimanual") in {"yes", "no"}
            for row in candidates
        ),
        "V14 candidate metadata contract changed",
    )
    contract = payload.get("queue_contract")
    _require(
        isinstance(contract, Mapping)
        and contract.get("candidate_count") == len(candidates)
        and contract.get("required_admitted_source_count") == 12
        and contract.get("preflight_rejection_is_not_a_selected_case") is True
        and contract.get("prediction_or_outcome_triggers_replacement") is False
        and contract.get("replacement_after_source_lock") is False,
        "V14 source-selection contract changed",
    )
    boundary = payload.get("information_boundary")
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("episode_media_read_before_queue_lock") is False
        and boundary.get("processed_geometry_read_before_queue_lock") is False
        and boundary.get("prefix_object_response_read_before_queue_lock") is False
        and boundary.get("future_object_positions_deserialized") is False
        and boundary.get("outcome_or_metric_read") is False
        and boundary.get("target_object_selected_or_read") is False
        and boundary.get("held_v8_target_query_score_barrier_or_outcome_access")
        is False
        and boundary.get("held_v8_plaintext_identity_read") is False,
        "V14 staging queue crossed its information boundary",
    )
    _require(
        payload.get("queue_sha256") == _canonical_sha256(payload),
        "V14 staging queue checksum changed",
    )
    return payload


__all__ = [
    "DATASET_REPOSITORY",
    "DATASET_REVISION",
    "EXCLUSION_OWNER",
    "MORPHOLOGY_STRATA",
    "PROTOCOL_ID",
    "QUEUE_CONTRACT",
    "QUEUE_KIND",
    "build_v14_staging_queue",
    "morphology_stratum",
    "validate_v14_staging_queue",
]
