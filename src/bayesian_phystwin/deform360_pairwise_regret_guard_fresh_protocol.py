"""Prospective fresh-data lock for the pairwise regret guard.

The open Deform360 development studies collectively consume nearly the entire
public object catalog.  This module performs the remaining source-only work in
a fail-closed order:

1. validate and union independently produced hash-only exclusion manifests;
2. compare that union with a public directory-only catalog;
3. select the complete set of metadata-valid episodes from the sole untouched
   physical object; and
4. bind the already-qualified runtime implementation and source certificate.

No camera frame, processed geometry, target trajectory, or outcome is accepted
by this module.  The resulting lock therefore authorizes only a single-object,
multi-action technical replication.  It cannot support a broad fresh-object or
state-of-the-art claim.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .deform360_pairwise_regret_guard_source import (
    pairwise_regret_certificate_from_dict,
)

EXCLUSION_KIND = "Deform360FreshObjectExclusionManifest"
UNION_KIND = "Deform360PairwiseRegretGuardFreshExclusionUnion"
LOCK_KIND = "Deform360PairwiseRegretGuardFreshTechnicalLock"
CATALOG_KIND = "Deform360PublicObjectCatalogSnapshot"
HASH_NAMESPACE = "deform360-fresh-object-exclusion-v1"
HASH_PREFIX = HASH_NAMESPACE.encode("ascii") + b"\0"
DATASET_REPOSITORY = "brownu/deform360"
DATASET_REVISION = "7fea8e20231a47641d1d2bc8791920ec4e62ec5e"

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_OBJECT_ID = re.compile(r"^[0-9]{3}-[a-z0-9][a-z0-9-]*$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any], *, digest_key: str) -> str:
    canonical = dict(payload)
    canonical.pop(digest_key, None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _seal(payload: Mapping[str, Any], *, digest_key: str) -> dict[str, Any]:
    result = json.loads(json.dumps(payload, allow_nan=False))
    result[digest_key] = _canonical_sha256(result, digest_key=digest_key)
    return result


def _load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON artifact: {source}") from exc
    _require(isinstance(payload, dict), f"JSON artifact is not an object: {source}")
    return payload


def object_exclusion_hash(object_id: str) -> str:
    _require(
        isinstance(object_id, str) and _OBJECT_ID.fullmatch(object_id) is not None,
        "object ID is malformed",
    )
    return hashlib.sha256(HASH_PREFIX + object_id.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FreshTechnicalLockConfig:
    """Frozen bindings for the only remaining public Deform360 object."""

    expected_exclusion_manifest_sha256s: tuple[str, ...]
    expected_public_object_count: int = 190
    expected_excluded_public_object_count: int = 189
    expected_remaining_public_object_count: int = 1
    minimum_valid_episode_count: int = 3
    runtime_method_commit: str = "fc2ff43f2e100ce910bf8085a7e9e0144d87bae8"
    source_protocol_file_sha256: str = (
        "e00cdb8ea92b7f501054dcbd44a011019ec9b75001825c0bc523af9831fc7e9b"
    )
    source_qualification_file_sha256: str = (
        "9fbc744deada239b1279827e762bfbabb32abe8b9d3a6ff68bd984ad22e710a3"
    )

    def __post_init__(self) -> None:
        _require(
            bool(self.expected_exclusion_manifest_sha256s)
            and tuple(sorted(set(self.expected_exclusion_manifest_sha256s)))
            == self.expected_exclusion_manifest_sha256s
            and all(
                _HEX64.fullmatch(value) is not None
                for value in self.expected_exclusion_manifest_sha256s
            ),
            "expected exclusion digests are malformed",
        )
        _require(
            self.expected_public_object_count >= 1,
            "expected public object count is invalid",
        )
        _require(
            0 <= self.expected_excluded_public_object_count
            <= self.expected_public_object_count,
            "expected excluded count is invalid",
        )
        _require(
            self.expected_remaining_public_object_count
            == self.expected_public_object_count
            - self.expected_excluded_public_object_count,
            "expected public counts are inconsistent",
        )
        _require(
            self.minimum_valid_episode_count >= 1,
            "minimum valid episode count is invalid",
        )
        _require(
            _HEX40.fullmatch(self.runtime_method_commit) is not None,
            "runtime method commit is malformed",
        )
        _require(
            _HEX64.fullmatch(self.source_protocol_file_sha256) is not None
            and _HEX64.fullmatch(self.source_qualification_file_sha256) is not None,
            "source binding digest is malformed",
        )


DEFAULT_EXCLUSION_MANIFEST_SHA256S = tuple(
    sorted(
        (
            "03c7b42fbdd0a8d9427a1ca8ee7fa8564c7691aa8e63402efefc7d7db16777a6",
            "18054955f5d8effb69eebc58aca2b3783e4e1fd0aa604f87bc2611f1f19a967c",
            "181796725382bcbe377b824dfac90243c6d3b0c9f9754fbeeb87cb6343d486ff",
            "36effc81b8c0c1dba07892c20d32cbb4ca250cc6fde7b3aed678e5cd57f6a40b",
            "5454f86c6b434c1f10b7762c3ed00e887b6184f03b0e61ce9b02651d2fed0e66",
            "562640ce93bcb6c230dce8c684888e2895cb31e6a6f06b8e52858c263e667635",
            "6d4ebc063d004bb679a365420c8076e3dbf1ca09812834dd5190ca6d0a2eafd7",
            "9ee5883ef49e6242220e0143c2e84c3b68521ed56a2a2c55a510be5dc0066bef",
            "b27847a5788ed17b8acd47de33b1a2af48c2afa284a88c7940faca47ec4cf136",
            "c8b79a1f6b76853229a5877428252ab69fcaf5b655f901e00b60cc1325795730",
            "cf472da17400ce2191d0af9b0b25788fd27b5e5c9976293e2814d2604d8da684",
            "edc7c9184f59965de4fe182e31a2e0dbc81e2c15037ecbe9cef76a9204f0f394",
            "f8437ecb01d327ef08403de9bcb0f8155b85e8308db2cd7067397f8d0538f02b",
            "fc7e24379f65a6162e8908716f214f92fbffae34407ddd752599368c768413bb",
            "fe62c0f3284c078ecc44e9c1fce28fecbd17223a29bf2a95762efc5e85b0fcd2",
        )
    )
)


def default_fresh_technical_lock_config() -> FreshTechnicalLockConfig:
    return FreshTechnicalLockConfig(
        expected_exclusion_manifest_sha256s=DEFAULT_EXCLUSION_MANIFEST_SHA256S
    )


def validate_exclusion_manifest(payload: Mapping[str, Any]) -> None:
    _require(payload.get("schema_version") == 1, "wrong exclusion schema")
    _require(payload.get("artifact_kind") == EXCLUSION_KIND, "wrong exclusion kind")
    _require(payload.get("hash_namespace") == HASH_NAMESPACE, "hash namespace changed")
    _require(
        payload.get("exclusion_sha256")
        == _canonical_sha256(payload, digest_key="exclusion_sha256"),
        "exclusion checksum changed",
    )
    hashes = payload.get("object_hashes")
    _require(
        isinstance(hashes, list)
        and hashes == sorted(set(hashes))
        and all(isinstance(value, str) and _HEX64.fullmatch(value) for value in hashes),
        "exclusion hashes are malformed",
    )
    boundary = payload.get("information_boundary")
    _require(isinstance(boundary, Mapping), "exclusion boundary is missing")
    _require(
        boundary.get("target_artifact_read") is False,
        "exclusion manifest crossed its target boundary",
    )
    _require(
        boundary.get("object_ids_emitted") is False,
        "exclusion manifest emitted protected object IDs",
    )


def build_exclusion_union(
    manifest_paths: Sequence[str | Path],
    *,
    config: FreshTechnicalLockConfig | None = None,
) -> dict[str, Any]:
    """Build the exact union of the preregistered independent manifests."""

    cfg = config or default_fresh_technical_lock_config()
    _require(bool(manifest_paths), "no exclusion manifests were supplied")
    by_digest: dict[str, tuple[dict[str, Any], str, str]] = {}
    for raw_path in manifest_paths:
        path = Path(raw_path).resolve()
        payload = _load_json(path)
        validate_exclusion_manifest(payload)
        digest = str(payload["exclusion_sha256"])
        record = (payload, file_sha256(path), path.name)
        previous = by_digest.setdefault(digest, record)
        _require(
            previous[0] == payload,
            "duplicate exclusion digest has different canonical content",
        )
    observed = tuple(sorted(by_digest))
    _require(
        observed == cfg.expected_exclusion_manifest_sha256s,
        "exclusion manifest set differs from the frozen source inventory",
    )
    union = sorted(
        {
            value
            for payload, _, _ in by_digest.values()
            for value in payload["object_hashes"]
        }
    )
    artifact = {
        "schema_version": 1,
        "artifact_kind": UNION_KIND,
        "hash_namespace": HASH_NAMESPACE,
        "owner": "bayesian-phystwin-pairwise-regret-guard-fresh-v1",
        "source_manifest_count": len(by_digest),
        "object_hash_count": len(union),
        "object_hashes": union,
        "source_manifests": [
            {
                "exclusion_sha256": digest,
                "file_sha256": by_digest[digest][1],
                "file_basename": by_digest[digest][2],
                "owner": by_digest[digest][0].get("owner"),
                "object_hash_count": len(by_digest[digest][0]["object_hashes"]),
            }
            for digest in observed
        ],
        "information_boundary": {
            "input_manifests_hash_only": True,
            "target_artifact_read": False,
            "object_ids_emitted": False,
            "held_v8_runtime_tree_accessed": False,
        },
    }
    return _seal(artifact, digest_key="union_sha256")


def validate_exclusion_union(
    payload: Mapping[str, Any],
    *,
    config: FreshTechnicalLockConfig | None = None,
) -> None:
    cfg = config or default_fresh_technical_lock_config()
    _require(payload.get("schema_version") == 1, "wrong union schema")
    _require(payload.get("artifact_kind") == UNION_KIND, "wrong union kind")
    _require(payload.get("hash_namespace") == HASH_NAMESPACE, "union namespace changed")
    _require(
        payload.get("union_sha256")
        == _canonical_sha256(payload, digest_key="union_sha256"),
        "union checksum changed",
    )
    hashes = payload.get("object_hashes")
    _require(
        isinstance(hashes, list)
        and hashes == sorted(set(hashes))
        and payload.get("object_hash_count") == len(hashes)
        and all(isinstance(value, str) and _HEX64.fullmatch(value) for value in hashes),
        "union object hashes are malformed",
    )
    sources = payload.get("source_manifests")
    _require(
        isinstance(sources, list)
        and payload.get("source_manifest_count") == len(sources)
        and tuple(row.get("exclusion_sha256") for row in sources)
        == cfg.expected_exclusion_manifest_sha256s,
        "union source inventory changed",
    )
    boundary = payload.get("information_boundary", {})
    _require(
        boundary.get("input_manifests_hash_only") is True
        and boundary.get("target_artifact_read") is False
        and boundary.get("object_ids_emitted") is False
        and boundary.get("held_v8_runtime_tree_accessed") is False,
        "union information boundary changed",
    )


def _validate_public_catalog(payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    _require(payload.get("schema_version") == 1, "wrong public catalog schema")
    _require(payload.get("artifact_kind") == CATALOG_KIND, "wrong public catalog kind")
    _require(
        payload.get("catalog_sha256")
        == _canonical_sha256(payload, digest_key="catalog_sha256"),
        "public catalog checksum changed",
    )
    objects = payload.get("objects")
    _require(isinstance(objects, list), "public catalog objects are missing")
    rows: list[dict[str, Any]] = []
    for row in objects:
        _require(isinstance(row, Mapping), "public catalog row is malformed")
        object_id = row.get("object_id")
        oid = row.get("oid")
        _require(
            isinstance(object_id, str) and _OBJECT_ID.fullmatch(object_id) is not None,
            "public object ID is malformed",
        )
        _require(
            isinstance(oid, str) and _HEX40.fullmatch(oid) is not None,
            "public object OID is malformed",
        )
        rows.append({"object_id": object_id, "oid": oid})
    _require(
        len({row["object_id"] for row in rows}) == len(rows),
        "public catalog repeats an object ID",
    )
    _require(
        payload.get("public_object_count") == len(rows),
        "public catalog count changed",
    )
    boundary = payload.get("information_boundary", {})
    _require(
        boundary.get("media_or_episode_payload_read") is False
        and boundary.get("public_directory_metadata_only") is True
        and boundary.get("target_metric_read") is False,
        "public catalog crossed the source-only boundary",
    )
    return tuple(rows)


def _validate_source_bindings(
    source_protocol_path: str | Path,
    source_qualification_path: str | Path,
    config: FreshTechnicalLockConfig,
) -> dict[str, Any]:
    _require(
        file_sha256(source_protocol_path) == config.source_protocol_file_sha256,
        "source protocol file changed",
    )
    protocol = _load_json(source_protocol_path)
    _require(
        protocol.get("artifact_kind") == "Deform360PairwiseRegretGuardSourceProtocol",
        "wrong source protocol kind",
    )
    _require(
        protocol.get("decision", {}).get("source_gate_passed") is True
        and protocol.get("decision", {}).get("fresh_accuracy_evaluation_allowed")
        is True,
        "source protocol did not authorize a fresh accuracy evaluation",
    )
    protocol_boundary = protocol.get("information_boundary", {})
    _require(
        protocol_boundary.get("runtime_candidate_accepts_target") is False
        and protocol_boundary.get("runtime_candidate_accepts_outcome") is False,
        "source protocol runtime boundary changed",
    )
    _require(
        file_sha256(source_qualification_path)
        == config.source_qualification_file_sha256,
        "source qualification file changed",
    )
    qualification = _load_json(source_qualification_path)
    _require(
        qualification.get("artifact_kind")
        == "Deform360PairwiseRegretGuardSourceQualification",
        "wrong source qualification kind",
    )
    _require(
        qualification.get("source_gate_passed") is True
        and qualification.get("fresh_accuracy_evaluation_allowed") is True
        and qualification.get("calibrated_safety_claim_allowed") is False,
        "source qualification decision changed",
    )
    qualification_boundary = qualification.get("information_boundary", {})
    _require(
        qualification_boundary.get("runtime_candidate_accepts_target") is False
        and qualification_boundary.get("runtime_candidate_accepts_outcome") is False
        and qualification_boundary.get("fresh_outcomes_may_not_select_or_refit_this_lock")
        is True,
        "source qualification boundary changed",
    )
    certificate_payload = qualification.get("deployment_artifact", {}).get(
        "candidate_certificate"
    )
    _require(
        isinstance(certificate_payload, Mapping),
        "source deployment certificate is missing",
    )
    certificate = pairwise_regret_certificate_from_dict(certificate_payload)
    return {
        "source_protocol_file_sha256": config.source_protocol_file_sha256,
        "source_qualification_file_sha256": (
            config.source_qualification_file_sha256
        ),
        "deployment_certificate_feature_count": len(certificate.feature_center),
        "calibrated_safety_claim_allowed": False,
    }


def _metadata_episodes(
    payload: Mapping[str, Any], *, object_id: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _require(payload.get("object") == object_id, "metadata object identity changed")
    sequences = payload.get("sequences")
    _require(isinstance(sequences, Mapping) and bool(sequences), "metadata sequences missing")
    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for key, value in sorted(sequences.items(), key=lambda item: int(item[0])):
        _require(key.isdigit(), "metadata episode key is malformed")
        episode_id = int(key)
        _require(isinstance(value, Mapping), "metadata episode row is malformed")
        reasons: list[str] = []
        action = value.get("action")
        bimanual = value.get("bimanual")
        nonprehensile = value.get("nonprehensile")
        if not isinstance(action, str) or not action:
            reasons.append("action must be a nonempty string")
        if bimanual not in {"yes", "no"}:
            reasons.append("bimanual must be exactly 'yes' or 'no'")
        if nonprehensile not in {"yes", "no"}:
            reasons.append("nonprehensile must be exactly 'yes' or 'no'")
        row = {
            "episode_id": episode_id,
            "action": action,
            "bimanual": bimanual,
            "nonprehensile": nonprehensile,
        }
        if reasons:
            rejected.append({**row, "rejection_reasons": reasons})
        else:
            valid.append(row)
    return valid, rejected


def build_fresh_technical_lock(
    exclusion_union_path: str | Path,
    public_catalog_path: str | Path,
    metadata_path: str | Path,
    source_protocol_path: str | Path,
    source_qualification_path: str | Path,
    *,
    config: FreshTechnicalLockConfig | None = None,
) -> dict[str, Any]:
    """Lock all valid episodes from the sole untouched public object."""

    cfg = config or default_fresh_technical_lock_config()
    union = _load_json(exclusion_union_path)
    validate_exclusion_union(union, config=cfg)
    catalog = _load_json(public_catalog_path)
    objects = _validate_public_catalog(catalog)
    _require(
        len(objects) == cfg.expected_public_object_count,
        "public object count differs from the frozen catalog",
    )
    excluded = set(union["object_hashes"])
    excluded_rows = [
        row for row in objects if object_exclusion_hash(row["object_id"]) in excluded
    ]
    remaining = [
        row for row in objects if object_exclusion_hash(row["object_id"]) not in excluded
    ]
    _require(
        len(excluded_rows) == cfg.expected_excluded_public_object_count,
        "excluded public-object count changed",
    )
    _require(
        len(remaining) == cfg.expected_remaining_public_object_count == 1,
        "the public catalog no longer has exactly one untouched object",
    )
    selected = remaining[0]
    metadata = _load_json(metadata_path)
    valid_episodes, rejected_episodes = _metadata_episodes(
        metadata, object_id=selected["object_id"]
    )
    _require(
        len(valid_episodes) >= cfg.minimum_valid_episode_count,
        "too few metadata-valid episodes remain",
    )
    _require(
        any(row["bimanual"] == "yes" for row in valid_episodes)
        and any(row["bimanual"] == "no" for row in valid_episodes),
        "valid episodes do not contain both uni- and bimanual actions",
    )
    source_binding = _validate_source_bindings(
        source_protocol_path, source_qualification_path, cfg
    )
    artifact = {
        "schema_version": 1,
        "artifact_kind": LOCK_KIND,
        "protocol_id": "deform360-pairwise-regret-guard-fresh-technical-v1",
        "status": "source_only_locked_before_episode_media_or_processed_geometry",
        "dataset": {
            "repository": DATASET_REPOSITORY,
            "revision": DATASET_REVISION,
            "public_catalog_sha256": catalog["catalog_sha256"],
            "public_catalog_file_sha256": file_sha256(public_catalog_path),
            "public_object_count": len(objects),
            "excluded_public_object_count": len(excluded_rows),
            "remaining_public_object_count": len(remaining),
        },
        "exclusion_union": {
            "union_sha256": union["union_sha256"],
            "union_file_sha256": file_sha256(exclusion_union_path),
            "source_manifest_count": union["source_manifest_count"],
            "object_hash_count": union["object_hash_count"],
        },
        "method": {
            "runtime_method_commit": cfg.runtime_method_commit,
            **source_binding,
            "refit_on_fresh_data_allowed": False,
            "fresh_outcomes_may_change_admission": False,
        },
        "selected_physical_object": {
            "object_id": selected["object_id"],
            "catalog_oid": selected["oid"],
            "metadata_file_sha256": file_sha256(metadata_path),
            "valid_episode_count": len(valid_episodes),
            "rejected_episode_count": len(rejected_episodes),
            "valid_episodes": valid_episodes,
            "rejected_episodes": rejected_episodes,
        },
        "execution_contract": {
            "ordinary_predictions_required_before_outcome_open": len(valid_episodes),
            "failed_predictions_retained_without_replacement": True,
            "malformed_metadata_episodes_unsealable": True,
            "all_prediction_seals_must_precede_any_outcome_open": True,
            "primary_comparison": (
                "selected physical/persistence backbone versus the frozen guarded "
                "pairwise update"
            ),
            "primary_metrics": [
                "post-update hidden-identity RMSE",
                "post-update hidden symmetric Chamfer",
            ],
            "aggregation": (
                "equal update intervals within episode and equal valid episodes "
                "within the single physical object"
            ),
        },
        "information_boundary": {
            "catalog_directory_metadata_only": True,
            "episode_metadata_read": True,
            "episode_media_read": False,
            "processed_geometry_read": False,
            "future_object_positions_deserialized": False,
            "outcome_or_metric_read": False,
            "held_v8_runtime_or_target_artifact_access": False,
        },
        "claim_boundary": (
            "This is a no-refit, single-object multi-action technical replication. "
            "It can test whether the frozen update transfers to the sole untouched "
            "public Deform360 object, but it is not multi-object confirmation, a "
            "calibrated safety result, official Deform360 parity, or a state-of-the-"
            "art claim."
        ),
    }
    return _seal(artifact, digest_key="lock_sha256")


def validate_fresh_technical_lock(
    payload: Mapping[str, Any],
    *,
    config: FreshTechnicalLockConfig | None = None,
) -> None:
    cfg = config or default_fresh_technical_lock_config()
    _require(payload.get("schema_version") == 1, "wrong technical-lock schema")
    _require(payload.get("artifact_kind") == LOCK_KIND, "wrong technical-lock kind")
    _require(
        payload.get("lock_sha256")
        == _canonical_sha256(payload, digest_key="lock_sha256"),
        "technical-lock checksum changed",
    )
    dataset = payload.get("dataset", {})
    _require(
        dataset.get("repository") == DATASET_REPOSITORY
        and dataset.get("revision") == DATASET_REVISION
        and dataset.get("public_object_count") == cfg.expected_public_object_count
        and dataset.get("excluded_public_object_count")
        == cfg.expected_excluded_public_object_count
        and dataset.get("remaining_public_object_count")
        == cfg.expected_remaining_public_object_count,
        "technical-lock dataset binding changed",
    )
    method = payload.get("method", {})
    _require(
        method.get("runtime_method_commit") == cfg.runtime_method_commit
        and method.get("source_protocol_file_sha256")
        == cfg.source_protocol_file_sha256
        and method.get("source_qualification_file_sha256")
        == cfg.source_qualification_file_sha256
        and method.get("calibrated_safety_claim_allowed") is False
        and method.get("refit_on_fresh_data_allowed") is False
        and method.get("fresh_outcomes_may_change_admission") is False,
        "technical-lock method binding changed",
    )
    selected = payload.get("selected_physical_object", {})
    valid = selected.get("valid_episodes")
    rejected = selected.get("rejected_episodes")
    _require(
        isinstance(valid, list)
        and len(valid) >= cfg.minimum_valid_episode_count
        and selected.get("valid_episode_count") == len(valid)
        and isinstance(rejected, list)
        and selected.get("rejected_episode_count") == len(rejected),
        "technical-lock episode accounting changed",
    )
    execution = payload.get("execution_contract", {})
    _require(
        execution.get("ordinary_predictions_required_before_outcome_open")
        == len(valid)
        and execution.get("failed_predictions_retained_without_replacement") is True
        and execution.get("malformed_metadata_episodes_unsealable") is True
        and execution.get("all_prediction_seals_must_precede_any_outcome_open")
        is True,
        "technical-lock execution contract changed",
    )
    boundary = payload.get("information_boundary", {})
    _require(
        boundary.get("episode_media_read") is False
        and boundary.get("processed_geometry_read") is False
        and boundary.get("future_object_positions_deserialized") is False
        and boundary.get("outcome_or_metric_read") is False
        and boundary.get("held_v8_runtime_or_target_artifact_access") is False,
        "technical lock crossed its source-only boundary",
    )


def write_json_artifact(payload: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "DEFAULT_EXCLUSION_MANIFEST_SHA256S",
    "FreshTechnicalLockConfig",
    "build_exclusion_union",
    "build_fresh_technical_lock",
    "default_fresh_technical_lock_config",
    "file_sha256",
    "object_exclusion_hash",
    "validate_exclusion_manifest",
    "validate_exclusion_union",
    "validate_fresh_technical_lock",
    "write_json_artifact",
]
