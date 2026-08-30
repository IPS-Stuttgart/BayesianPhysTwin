#!/usr/bin/env python3
"""Build a target-blind Deform360 query-validation readiness record."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "bayesian-phystwin/deform360-query-validation-readiness-v1"
INVENTORY_SCHEMA = "bayesian-phystwin/deform360-metadata-inventory-v1"
EXCLUSION_KIND = "deform360-covariance-only-target-exclusion-v1"
HASH_NAMESPACE = "deform360-fresh-object-exclusion-v1"
HASH_PREFIX = b"deform360-fresh-object-exclusion-v1\0"
OBJECT_RE = re.compile(r"^\d{3}-.+")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load JSON object: {path}") from error
    _require(isinstance(value, dict), f"expected a JSON object: {path}")
    return value


def _object_hash(object_id: str) -> str:
    return hashlib.sha256(HASH_PREFIX + object_id.encode("utf-8")).hexdigest()


def _rank(seed: str, *parts: object) -> str:
    payload = seed.encode("utf-8")
    for part in parts:
        payload += b"\0" + str(part).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _object_stratum(object_id: str) -> str:
    return "sheet" if object_id.endswith("-cloth") else "volumetric"


def _regular_readable_file(path: Path) -> bool:
    try:
        mode = path.lstat().st_mode
    except OSError:
        return False
    return stat.S_ISREG(mode) and not path.is_symlink() and os.access(path, os.R_OK)


def _action_family(action: str, families: Mapping[str, Sequence[str]]) -> str | None:
    token = action.strip().lower().split(maxsplit=1)[0] if action.strip() else ""
    matches = [
        family
        for family, prefixes in families.items()
        if token in {str(item).lower() for item in prefixes}
    ]
    return matches[0] if len(matches) == 1 else None


def _episode_records(
    metadata: Mapping[str, Any],
    *,
    object_id: str,
    families: Mapping[str, Sequence[str]],
    rank_seed: str,
) -> tuple[dict[str, Any], ...]:
    sequences = metadata.get("sequences")
    if not isinstance(sequences, Mapping):
        return ()
    records: list[dict[str, Any]] = []
    for raw_id, raw_record in sequences.items():
        if not isinstance(raw_id, str) or not raw_id.isdigit():
            continue
        if not isinstance(raw_record, Mapping):
            continue
        action = raw_record.get("action")
        if not isinstance(action, str) or not action.strip():
            continue
        family = _action_family(action, families)
        if family is None:
            continue
        episode_id = int(raw_id)
        records.append(
            {
                "episode_id": episode_id,
                "action": action.strip(),
                "action_family": family,
                "bimanual": raw_record.get("bimanual"),
                "nonprehensile": raw_record.get("nonprehensile"),
                "rank": _rank(rank_seed, object_id, episode_id),
            }
        )
    return tuple(sorted(records, key=lambda row: (row["rank"], row["episode_id"])))


def _select_pair(
    episodes: Sequence[Mapping[str, Any]],
    *,
    object_id: str,
    rank_seed: str,
) -> dict[str, Any] | None:
    pairs: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []
    for source in episodes:
        for target in episodes:
            if source["episode_id"] == target["episode_id"]:
                continue
            if source["action_family"] == target["action_family"]:
                continue
            rank = _rank(
                rank_seed,
                object_id,
                source["episode_id"],
                target["episode_id"],
                "source-target-pair",
            )
            pairs.append((rank, source, target))
    if not pairs:
        return None
    rank, source, target = min(
        pairs,
        key=lambda item: (
            item[0],
            item[1]["episode_id"],
            item[2]["episode_id"],
        ),
    )
    return {
        "pair_rank": rank,
        "source": dict(source),
        "target": dict(target),
    }


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    _require(protocol.get("schema") == SCHEMA, "protocol schema changed")
    _require(protocol.get("schema_version") == 1, "protocol version changed")
    _require(
        protocol.get("status") == "locked-before-development-metadata-access",
        "protocol is not locked before metadata access",
    )
    dataset = protocol.get("dataset")
    _require(isinstance(dataset, Mapping), "dataset contract is missing")
    root = dataset.get("root")
    _require(
        isinstance(root, str) and Path(root).is_absolute(),
        "dataset root must be absolute",
    )
    runner = protocol.get("runner")
    _require(isinstance(runner, Mapping), "runner contract is missing")
    _require(runner.get("label") == "gpuserver4090", "runner label changed")
    selection = protocol.get("development_metadata_selection")
    _require(isinstance(selection, Mapping), "selection contract is missing")
    _require(selection.get("objects_per_stratum") == 4, "cohort size changed")
    boundary = protocol.get("information_boundary")
    _require(isinstance(boundary, Mapping), "information boundary is missing")
    for key in (
        "camera_media_decoded",
        "robot_or_tactile_arrays_opened",
        "geometry_or_track_annotations_opened",
        "target_future_opened",
        "score_bearing_outcomes_opened",
    ):
        _require(boundary.get(key) is False, f"information boundary changed: {key}")


def build_readiness(
    *,
    data_root: Path,
    protocol_path: Path,
    inventory_path: Path,
    historical_exclusion_path: Path,
) -> dict[str, Any]:
    protocol = _load_json(protocol_path)
    _validate_protocol(protocol)
    _require(
        data_root == Path(protocol["dataset"]["root"]).resolve(),
        "runtime dataset root differs from the locked protocol",
    )
    inventory = _load_json(inventory_path)
    _require(inventory.get("schema") == INVENTORY_SCHEMA, "inventory schema changed")
    preflight = protocol["preflight_binding"]
    inventory_matches_preflight = (
        inventory.get("content_inventory_sha256")
        == preflight["content_inventory_sha256"]
    )

    exclusion = _load_json(historical_exclusion_path)
    historical = protocol["historical_exclusion_binding"]
    _require(
        _file_sha256(historical_exclusion_path) == historical["file_sha256"],
        "historical exclusion file digest changed",
    )
    _require(
        exclusion.get("artifact_kind") == EXCLUSION_KIND,
        "historical exclusion kind changed",
    )
    _require(
        exclusion.get("hash_namespace") == HASH_NAMESPACE,
        "historical exclusion namespace changed",
    )
    canonical = dict(exclusion)
    canonical.pop("exclusion_sha256", None)
    _require(
        _sha256_json(canonical) == historical["canonical_sha256"],
        "historical exclusion canonical digest changed",
    )
    hashes = exclusion.get("object_hashes")
    _require(
        isinstance(hashes, list)
        and len(hashes) == len(set(hashes))
        and all(isinstance(item, str) and SHA256_RE.fullmatch(item) for item in hashes),
        "historical exclusion hashes are malformed",
    )
    excluded_hashes = set(hashes)

    object_rows = inventory.get("objects")
    _require(isinstance(object_rows, list), "inventory object roster is missing")
    raw_objects: list[dict[str, Any]] = []
    for row in object_rows:
        _require(isinstance(row, Mapping), "inventory object record is malformed")
        object_id = row.get("object_id")
        _require(
            isinstance(object_id, str) and OBJECT_RE.fullmatch(object_id) is not None,
            "inventory object identity is malformed",
        )
        object_hash = _object_hash(object_id)
        raw_objects.append(
            {
                "object_id": object_id,
                "object_hash": object_hash,
                "classification": row.get("classification"),
                "stratum": _object_stratum(object_id),
                "historically_excluded": object_hash in excluded_hashes,
                "numeric_path_counts": row.get("numeric_path_counts", {}),
            }
        )

    selection = protocol["development_metadata_selection"]
    rank_seed = str(selection["rank_seed"])
    objects_per_stratum = int(selection["objects_per_stratum"])
    selected: list[dict[str, Any]] = []
    for stratum in ("sheet", "volumetric"):
        candidates = sorted(
            (
                row
                for row in raw_objects
                if inventory_matches_preflight
                and row["classification"] == "candidate_name_only"
                and row["stratum"] == stratum
            ),
            key=lambda row: (
                _rank(rank_seed, stratum, row["object_id"]),
                row["object_id"],
            ),
        )
        for row in candidates[:objects_per_stratum]:
            selected.append(
                {
                    **row,
                    "selection_rank": _rank(rank_seed, stratum, row["object_id"]),
                }
            )

    families = selection["action_families"]
    metadata_ready: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    metadata_opened_count = 0
    raw_prefix = data_root / "raw-repository" / "raw"
    for row in selected:
        object_id = row["object_id"]
        metadata_path = raw_prefix / object_id / "metadata.json"
        relative = metadata_path.relative_to(data_root).as_posix()
        if not _regular_readable_file(metadata_path):
            unsupported.append(
                {
                    **row,
                    "metadata_path": relative,
                    "reason": "metadata-json-missing-unreadable-or-not-regular",
                }
            )
            continue
        raw = metadata_path.read_bytes()
        metadata_opened_count += 1
        try:
            metadata = json.loads(raw)
        except json.JSONDecodeError:
            unsupported.append(
                {
                    **row,
                    "metadata_path": relative,
                    "metadata_sha256": hashlib.sha256(raw).hexdigest(),
                    "reason": "metadata-json-invalid",
                }
            )
            continue
        if not isinstance(metadata, Mapping):
            unsupported.append(
                {
                    **row,
                    "metadata_path": relative,
                    "metadata_sha256": hashlib.sha256(raw).hexdigest(),
                    "reason": "metadata-root-not-object",
                }
            )
            continue
        episodes = _episode_records(
            metadata,
            object_id=object_id,
            families=families,
            rank_seed=rank_seed,
        )
        pair = _select_pair(episodes, object_id=object_id, rank_seed=rank_seed)
        if pair is None:
            unsupported.append(
                {
                    **row,
                    "metadata_path": relative,
                    "metadata_sha256": hashlib.sha256(raw).hexdigest(),
                    "recognized_episode_count": len(episodes),
                    "reason": "no-distinct-action-family-source-target-pair",
                }
            )
            continue
        metadata_ready.append(
            {
                **row,
                "metadata_path": relative,
                "metadata_sha256": hashlib.sha256(raw).hexdigest(),
                "recognized_episode_count": len(episodes),
                "source_target_pair": pair,
            }
        )

    counts: dict[str, int] = {}
    for row in raw_objects:
        key = f"{row['classification']}:{row['stratum']}"
        counts[key] = counts.get(key, 0) + 1
    provisional = [row for row in raw_objects if not row["historically_excluded"]]
    current_union_complete = False
    complete_design = len(metadata_ready) == 2 * objects_per_stratum
    if not inventory_matches_preflight:
        status = "names-only-inventory-changed-relock-required"
    elif complete_design:
        status = "development-metadata-design-ready"
    else:
        status = "development-metadata-design-incomplete"
    return {
        "artifact_kind": "Deform360QueryValidationReadinessV1",
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "dataset": {
            "root": str(data_root),
            "names_only_content_inventory_sha256": inventory[
                "content_inventory_sha256"
            ],
            "matches_retained_preflight": inventory_matches_preflight,
            "retained_preflight_content_inventory_sha256": preflight[
                "content_inventory_sha256"
            ],
            "recognized_object_count": len(raw_objects),
            "classification_stratum_counts": dict(sorted(counts.items())),
        },
        "historical_exclusion": {
            "source_commit": historical["commit"],
            "object_hash_count": len(excluded_hashes),
            "canonical_sha256": historical["canonical_sha256"],
            "current_complete_cross_project_union_verified": current_union_complete,
            "provisionally_unexcluded_object_count": len(provisional),
            "provisionally_unexcluded_by_stratum": {
                stratum: sum(row["stratum"] == stratum for row in provisional)
                for stratum in ("sheet", "volumetric")
            },
            "interpretation": (
                "Objects absent from the historical union are only provisionally "
                "unexcluded; no post-cutoff complete cross-project delta is bound."
            ),
        },
        "development_metadata_design": {
            "purpose": "adapter-and-source-target-contract-development-only",
            "rank_seed": rank_seed,
            "requested_objects_per_stratum": objects_per_stratum,
            "selected_name_only_count": len(selected),
            "metadata_ready_count": len(metadata_ready),
            "metadata_json_opened_count": metadata_opened_count,
            "selected": metadata_ready,
            "unsupported": unsupported,
            "target_array_or_media_opened": False,
            "score_bearing_outcome_opened": False,
        },
        "decision": {
            "status": status,
            "development_metadata_design_ready": complete_design,
            "fresh_confirmation_authorized": False,
            "target_payload_access_authorized": False,
            "model_scoring_authorized": False,
            "next_action": (
                "freeze one released-processed-annotation or official-processing "
                "adapter and source-only prediction contract before opening any "
                "selected target future"
            ),
        },
        "information_boundary": {
            "dataset_names_and_sizes_opened": True,
            "selected_metadata_json_opened": metadata_opened_count > 0,
            "camera_media_decoded": False,
            "robot_or_tactile_arrays_opened": False,
            "geometry_or_track_annotations_opened": False,
            "target_future_opened": False,
            "score_bearing_outcomes_opened": False,
        },
        "claim_boundary": (
            "This target-blind readiness audit may establish only a deterministic "
            "development metadata roster and source-target episode design. It is "
            "not a fresh cohort lock, payload-integrity certificate, model "
            "evaluation, real-data accuracy result, calibration result, physical "
            "transport result, or authorization to open target futures."
        ),
    }


def _report(result: Mapping[str, Any]) -> str:
    dataset = result["dataset"]
    exclusion = result["historical_exclusion"]
    design = result["development_metadata_design"]
    decision = result["decision"]
    lines = [
        "# Deform360 query-validation readiness",
        "",
        f"- Decision: `{decision['status']}`",
        f"- Recognized objects: `{dataset['recognized_object_count']}`",
        (
            "- Historical exclusion hashes: "
            f"`{exclusion['object_hash_count']}`"
        ),
        (
            "- Provisionally unexcluded by the historical union: "
            f"`{exclusion['provisionally_unexcluded_object_count']}`"
        ),
        f"- Selected metadata-ready objects: `{design['metadata_ready_count']}`",
        "- Fresh confirmation authorized: `false`",
        "- Target payload access authorized: `false`",
        "",
        "## Development metadata pairs",
        "",
        "| Object | Stratum | Source | Target | Historical exclusion |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in design["selected"]:
        pair = row["source_target_pair"]
        source = pair["source"]
        target = pair["target"]
        lines.append(
            "| {object_id} | {stratum} | {source_id}: {source_action} | "
            "{target_id}: {target_action} | {excluded} |".format(
                object_id=row["object_id"],
                stratum=row["stratum"],
                source_id=source["episode_id"],
                source_action=source["action"].replace("|", "/"),
                target_id=target["episode_id"],
                target_action=target["action"].replace("|", "/"),
                excluded=str(row["historically_excluded"]).lower(),
            )
        )
    if design["unsupported"]:
        lines.extend(["", "## Unsupported selections", ""])
        for row in design["unsupported"]:
            lines.append(f"- `{row['object_id']}`: `{row['reason']}`")
    lines.extend(["", result["claim_boundary"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--historical-exclusion", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    result = build_readiness(
        data_root=args.data_root.resolve(),
        protocol_path=args.protocol.resolve(),
        inventory_path=args.inventory.resolve(),
        historical_exclusion_path=args.historical_exclusion.resolve(),
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "readiness.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(_report(result), encoding="utf-8")
    print(json.dumps(result["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
