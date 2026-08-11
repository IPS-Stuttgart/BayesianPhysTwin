#!/usr/bin/env python3
"""Freeze a metadata-only, factorial Deform360 covariance target roster."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

PROTOCOL_SCHEMA = "bayesian-phystwin/deform360-covariance-only-target-protocol-v1"
SELECTION_SCHEMA = "bayesian-phystwin/deform360-covariance-only-target-selection-v1"
EXCLUSION_SCHEMA = "deform360-covariance-only-target-exclusion-v1"
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


def _require_sha256(value: object, *, name: str) -> str:
    result = str(value)
    _require(SHA256_RE.fullmatch(result) is not None, f"{name} is not SHA-256")
    return result


def _object_hash(object_id: str) -> str:
    return hashlib.sha256(HASH_PREFIX + object_id.encode("utf-8")).hexdigest()


def _rank(seed: str, *parts: object) -> str:
    payload = seed.encode("utf-8")
    for part in parts:
        payload += b"\0" + str(part).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _object_stratum(object_id: str) -> str:
    return "sheet" if object_id.endswith("-cloth") else "volumetric"


def _protocol_digest(protocol: Mapping[str, Any]) -> str:
    content = dict(protocol)
    content.pop("protocol_sha256", None)
    return _sha256_json(content)


def load_protocol(path: Path, *, repository: Path) -> tuple[dict[str, Any], set[str]]:
    """Load the pre-metadata protocol and its hash-only exclusion set."""

    protocol = _load_json(path.resolve())
    _require(protocol.get("schema") == PROTOCOL_SCHEMA, "protocol schema changed")
    _require(protocol.get("schema_version") == 1, "protocol version changed")
    status = protocol.get("status")
    _require(
        status
        in {
            "locked-before-target-metadata-access",
            "schema-amended-before-target-roster-and-payload-access",
        },
        "protocol is not locked before target payload access",
    )
    expected = _require_sha256(
        protocol.get("protocol_sha256"), name="protocol_sha256"
    )
    _require(expected == _protocol_digest(protocol), "protocol digest changed")
    boundary = protocol.get("information_boundary")
    _require(isinstance(boundary, Mapping), "information boundary is missing")
    _require(boundary.get("camera_media_decoded") is False, "camera boundary changed")
    _require(
        boundary.get("robot_or_tactile_arrays_opened") is False,
        "sensor-array boundary changed",
    )
    _require(
        boundary.get("geometry_or_track_annotations_opened") is False,
        "annotation boundary changed",
    )
    _require(
        boundary.get("target_outcomes_opened") is False,
        "target outcome boundary changed",
    )
    selection = protocol.get("selection")
    _require(isinstance(selection, Mapping), "selection contract is missing")
    _require(selection.get("roster_size") == 24, "target roster size changed")
    _require(
        selection.get("candidate_objects_per_stratum") == 16,
        "candidate-panel size changed",
    )
    nonprehensile_policy = selection.get(
        "nonprehensile_selection_policy",
        "strict-yes-no-or-terminate",
    )
    _require(
        nonprehensile_policy
        in {
            "strict-yes-no-or-terminate",
            "record-only-never-used-for-selection",
        },
        "nonprehensile selection policy changed",
    )
    if status == "locked-before-target-metadata-access":
        _require(
            selection.get("metadata_invalid_candidate_policy")
            == "terminate before target payload; do not replace",
            "metadata failure policy changed",
        )
        _require(
            nonprehensile_policy == "strict-yes-no-or-terminate",
            "v1 metadata policy changed",
        )
    else:
        amendment = protocol.get("amendment")
        _require(isinstance(amendment, Mapping), "schema amendment is missing")
        _require(
            amendment.get("candidate_panel_reused_without_replacement") is True,
            "schema amendment replaced the candidate panel",
        )
        _require(
            amendment.get("target_roster_created_before_amendment") is False,
            "schema amendment followed target-roster creation",
        )
        _require(
            amendment.get("target_payload_opened_before_amendment") is False,
            "schema amendment followed target payload access",
        )
        _require(
            amendment.get("only_selection_change")
            == "nonprehensile is record-only and cannot affect eligibility or assignment",
            "schema amendment changed more than the nonselective field",
        )
        _require(
            nonprehensile_policy == "record-only-never-used-for-selection",
            "schema amendment policy changed",
        )
    exclusion_record = protocol.get("exclusion")
    _require(isinstance(exclusion_record, Mapping), "exclusion record is missing")
    relative = exclusion_record.get("artifact_path")
    _require(isinstance(relative, str) and relative, "exclusion path is missing")
    exclusion_path = (repository.resolve() / relative).resolve()
    _require(
        exclusion_path.is_relative_to(repository.resolve()),
        "exclusion path escaped repository",
    )
    _require(
        _file_sha256(exclusion_path)
        == _require_sha256(
            exclusion_record.get("file_sha256"), name="exclusion file SHA-256"
        ),
        "exclusion file digest changed",
    )
    exclusion = _load_json(exclusion_path)
    _require(
        exclusion.get("artifact_kind") == EXCLUSION_SCHEMA,
        "exclusion artifact kind changed",
    )
    _require(
        exclusion.get("hash_namespace") == HASH_NAMESPACE,
        "exclusion hash namespace changed",
    )
    hashes = exclusion.get("object_hashes")
    _require(
        isinstance(hashes, list)
        and len(hashes) == len(set(hashes))
        and all(isinstance(item, str) and SHA256_RE.fullmatch(item) for item in hashes),
        "exclusion object hashes are malformed",
    )
    canonical = dict(exclusion)
    canonical.pop("exclusion_sha256", None)
    _require(
        _sha256_json(canonical)
        == _require_sha256(
            exclusion_record.get("canonical_sha256"),
            name="exclusion canonical SHA-256",
        )
        == exclusion.get("exclusion_sha256"),
        "exclusion canonical digest changed",
    )
    _require(
        len(hashes) == exclusion_record.get("object_hash_count"),
        "exclusion object count changed",
    )
    return protocol, set(hashes)


def select_candidate_panel(
    available_objects: Sequence[str],
    *,
    excluded_hashes: set[str],
    seed: str,
    count_per_stratum: int,
) -> tuple[dict[str, Any], ...]:
    """Select the fixed object panel from names and hash exclusions only."""

    _require(
        len(available_objects) == len(set(available_objects)),
        "available object names contain duplicates",
    )
    result: list[dict[str, Any]] = []
    for stratum in ("sheet", "volumetric"):
        eligible = sorted(
            (
                object_id
                for object_id in available_objects
                if OBJECT_RE.fullmatch(object_id)
                and _object_stratum(object_id) == stratum
                and _object_hash(object_id) not in excluded_hashes
            ),
            key=lambda object_id: (_rank(seed, stratum, object_id), object_id),
        )
        _require(
            len(eligible) >= count_per_stratum,
            f"{stratum} has only {len(eligible)} fresh objects",
        )
        result.extend(
            {
                "object_id": object_id,
                "object_hash": _object_hash(object_id),
                "object_rank": _rank(seed, stratum, object_id),
                "stratum": stratum,
                "metadata_path": f"raw/{object_id}/metadata.json",
            }
            for object_id in eligible[:count_per_stratum]
        )
    return tuple(sorted(result, key=lambda row: (row["stratum"], row["object_rank"])))


def _action_family(action: str, families: Mapping[str, Sequence[str]]) -> str:
    token = action.strip().lower().split(maxsplit=1)[0] if action.strip() else ""
    matches = [
        family for family, prefixes in families.items() if token in set(prefixes)
    ]
    _require(len(matches) == 1, f"action has no unique registered family: {action!r}")
    return matches[0]


def _episode_options(
    metadata: Mapping[str, Any],
    *,
    object_id: str,
    stratum: str,
    seed: str,
    families: Mapping[str, Sequence[str]],
    nonprehensile_policy: str,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    sequences = metadata.get("sequences")
    _require(isinstance(sequences, Mapping) and sequences, f"{object_id} has no sequences")
    by_cell: dict[tuple[str, str, str], dict[str, Any]] = {}
    for raw_id, raw in sequences.items():
        _require(
            isinstance(raw_id, str) and raw_id.isdigit(),
            f"{object_id} has a noninteger episode ID",
        )
        _require(isinstance(raw, Mapping), f"{object_id} episode {raw_id} is malformed")
        episode_id = int(raw_id)
        action = raw.get("action")
        bimanual = raw.get("bimanual")
        nonprehensile = raw.get("nonprehensile")
        _require(
            isinstance(action, str) and action.strip(),
            f"{object_id} episode {episode_id} action is malformed",
        )
        _require(
            bimanual in {"yes", "no"},
            f"{object_id} episode {episode_id} bimanual is malformed",
        )
        nonprehensile_valid = nonprehensile in {"yes", "no"}
        if nonprehensile_policy == "strict-yes-no-or-terminate":
            _require(
                nonprehensile_valid,
                f"{object_id} episode {episode_id} nonprehensile is malformed",
            )
        else:
            _require(
                nonprehensile_policy == "record-only-never-used-for-selection",
                "unknown nonprehensile selection policy",
            )
        family = _action_family(action, families)
        cell = (stratum, str(bimanual), family)
        record = {
            "episode_id": episode_id,
            "episode_rank": _rank(seed, object_id, episode_id),
            "action": action,
            "action_family": family,
            "bimanual": bimanual,
            "nonprehensile": nonprehensile if nonprehensile_valid else None,
            "nonprehensile_metadata_valid": nonprehensile_valid,
        }
        previous = by_cell.get(cell)
        if previous is None or (
            record["episode_rank"], record["episode_id"]
        ) < (previous["episode_rank"], previous["episode_id"]):
            by_cell[cell] = record
    return by_cell


def _factorial_cells(protocol: Mapping[str, Any]) -> tuple[tuple[str, str, str], ...]:
    selection = protocol["selection"]
    exact = selection["exact_factorial_cells"]
    families = tuple(sorted(selection["action_families"]))
    return tuple(
        (stratum, bimanual, family)
        for stratum in exact["object_stratum"]
        for bimanual in exact["bimanual"]
        for family in families
    )


def _match_factorial_roster(
    panel: Sequence[Mapping[str, Any]],
    *,
    options: Mapping[str, Mapping[tuple[str, str, str], Mapping[str, Any]]],
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    cells = _factorial_cells(protocol)
    per_cell = int(protocol["selection"]["exact_factorial_cells"]["sessions_per_cell"])
    eligible: dict[tuple[str, str, str], list[str]] = {}
    panel_by_id = {str(row["object_id"]): row for row in panel}
    for cell in cells:
        eligible[cell] = sorted(
            (object_id for object_id, rows in options.items() if cell in rows),
            key=lambda object_id: (
                options[object_id][cell]["episode_rank"],
                panel_by_id[object_id]["object_rank"],
                object_id,
            ),
        )
        _require(
            len(eligible[cell]) >= per_cell,
            f"factorial cell {cell!r} has insufficient objects",
        )
    slots = [(cell, index) for cell in cells for index in range(per_cell)]
    slots.sort(key=lambda slot: (len(eligible[slot[0]]), slot[0], slot[1]))
    object_to_slot: dict[str, tuple[tuple[str, str, str], int]] = {}
    slot_to_object: dict[tuple[tuple[str, str, str], int], str] = {}

    def assign(
        slot: tuple[tuple[str, str, str], int],
        seen: set[str],
    ) -> bool:
        cell = slot[0]
        for object_id in eligible[cell]:
            if object_id in seen:
                continue
            seen.add(object_id)
            previous = object_to_slot.get(object_id)
            if previous is None or assign(previous, seen):
                object_to_slot[object_id] = slot
                slot_to_object[slot] = object_id
                return True
        return False

    for slot in slots:
        _require(assign(slot, set()), "fixed candidate panel has no factorial match")
    _require(len(slot_to_object) == len(slots), "factorial match is incomplete")
    roster: list[dict[str, Any]] = []
    for slot in sorted(slot_to_object):
        cell = slot[0]
        object_id = slot_to_object[slot]
        base = panel_by_id[object_id]
        episode = options[object_id][cell]
        roster.append(
            {
                "object_id": object_id,
                "object_hash": base["object_hash"],
                "stratum": base["stratum"],
                "metadata_path": base["metadata_path"],
                **dict(episode),
                "factorial_cell": {
                    "stratum": cell[0],
                    "bimanual": cell[1],
                    "action_family": cell[2],
                    "replicate": slot[1],
                },
            }
        )
    _require(
        len(roster) == protocol["selection"]["roster_size"],
        "factorial roster size changed",
    )
    _require(
        len({row["object_id"] for row in roster}) == len(roster),
        "factorial roster repeats a physical object",
    )
    return roster


def build_selection(
    snapshot: Mapping[str, Any],
    *,
    repository: Path,
    protocol_path: Path,
    implementation_revision: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the exact target roster from a names/metadata-only snapshot."""

    protocol, excluded_hashes = load_protocol(protocol_path, repository=repository)
    available = snapshot.get("raw_objects")
    metadata = snapshot.get("metadata_by_object")
    metadata_sha = snapshot.get("metadata_sha256_by_object")
    opened = snapshot.get("opened_paths")
    resolved = snapshot.get("resolved_revision")
    _require(
        isinstance(available, list)
        and all(isinstance(item, str) for item in available),
        "snapshot raw_objects is malformed",
    )
    _require(isinstance(metadata, Mapping), "snapshot metadata is missing")
    _require(isinstance(metadata_sha, Mapping), "snapshot metadata hashes are missing")
    _require(isinstance(opened, list), "snapshot opened paths are missing")
    _require(
        resolved == protocol["dataset"]["revision"],
        "resolved dataset revision changed",
    )
    selection = protocol["selection"]
    panel = select_candidate_panel(
        available,
        excluded_hashes=excluded_hashes,
        seed=str(selection["seed"]),
        count_per_stratum=int(selection["candidate_objects_per_stratum"]),
    )
    panel_ids = {row["object_id"] for row in panel}
    expected_paths = {row["metadata_path"] for row in panel}
    _require(set(opened) == expected_paths, "metadata access exceeded fixed panel")
    _require(set(metadata) == panel_ids, "metadata object set changed")
    _require(set(metadata_sha) == panel_ids, "metadata hash object set changed")
    families = selection["action_families"]
    nonprehensile_policy = selection.get(
        "nonprehensile_selection_policy",
        "strict-yes-no-or-terminate",
    )
    options: dict[str, dict[tuple[str, str, str], dict[str, Any]]] = {}
    bound_panel: list[dict[str, Any]] = []
    for row in panel:
        object_id = row["object_id"]
        digest = _require_sha256(
            metadata_sha[object_id], name=f"{object_id} metadata SHA-256"
        )
        object_options = _episode_options(
            metadata[object_id],
            object_id=object_id,
            stratum=row["stratum"],
            seed=str(selection["seed"]),
            families=families,
            nonprehensile_policy=str(nonprehensile_policy),
        )
        options[object_id] = object_options
        bound_panel.append(
            {
                **dict(row),
                "metadata_sha256": digest,
                "valid_episode_count": len(metadata[object_id]["sequences"]),
                "eligible_factorial_cells": [
                    {
                        "stratum": cell[0],
                        "bimanual": cell[1],
                        "action_family": cell[2],
                    }
                    for cell in sorted(object_options)
                ],
            }
        )
    roster = _match_factorial_roster(
        bound_panel,
        options=options,
        protocol=protocol,
    )
    metadata_by_id = {row["object_id"]: row for row in bound_panel}
    for row in roster:
        row["metadata_sha256"] = metadata_by_id[row["object_id"]]["metadata_sha256"]
    content: dict[str, Any] = {
        "schema": SELECTION_SCHEMA,
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol["protocol_sha256"],
        "dataset": protocol["dataset"],
        "implementation_revision": implementation_revision,
        "exclusion": protocol["exclusion"],
        "candidate_panel": bound_panel,
        "target_roster": roster,
        "failure_accounting_denominator": len(roster),
        "replacement_allowed": False,
        "information_boundary": {
            "object_directory_names_opened": True,
            "metadata_json_opened": True,
            "opened_metadata_paths": sorted(opened),
            "camera_media_decoded": False,
            "robot_or_tactile_arrays_opened": False,
            "geometry_or_track_annotations_opened": False,
            "target_outcomes_opened": False,
        },
        "next_gate": (
            "commit this exact roster before repository support inspection, "
            "target payload download, decoding, processing, or scoring"
        ),
    }
    content["roster_sha256"] = _sha256_json(roster)
    content["selection_sha256"] = _sha256_json(content)
    touched: dict[str, Any] = {
        "artifact_kind": "deform360-covariance-only-target-metadata-scope-exclusion-v1",
        "schema_version": 1,
        "hash_namespace": HASH_NAMESPACE,
        "object_hashes": sorted(row["object_hash"] for row in bound_panel),
        "object_hash_count": len(bound_panel),
        "selection_sha256": content["selection_sha256"],
        "scope": "all objects in the fixed metadata-only candidate panel",
        "information_boundary": {
            "object_ids_emitted": False,
            "camera_media_decoded": False,
            "target_outcomes_opened": False,
        },
    }
    touched["exclusion_sha256"] = _sha256_json(touched)
    return content, touched


def _download_snapshot(
    protocol: Mapping[str, Any],
    *,
    excluded_hashes: set[str],
    metadata_cache: Path,
) -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as error:
        raise RuntimeError("live selection requires huggingface_hub") from error
    dataset = protocol["dataset"]
    api = HfApi()
    info = api.repo_info(
        repo_id=dataset["repository"],
        repo_type="dataset",
        revision=dataset["revision"],
        files_metadata=False,
    )
    _require(info.sha == dataset["revision"], "official dataset revision changed")
    entries = api.list_repo_tree(
        repo_id=dataset["repository"],
        repo_type="dataset",
        revision=dataset["revision"],
        path_in_repo=dataset["raw_prefix"],
        recursive=False,
        expand=False,
    )
    available = sorted(
        {
            PurePosixPath(path).name
            for entry in entries
            if isinstance((path := getattr(entry, "path", None)), str)
            and OBJECT_RE.fullmatch(PurePosixPath(path).name)
        }
    )
    selection = protocol["selection"]
    panel = select_candidate_panel(
        available,
        excluded_hashes=excluded_hashes,
        seed=str(selection["seed"]),
        count_per_stratum=int(selection["candidate_objects_per_stratum"]),
    )
    metadata_by_object: dict[str, Any] = {}
    metadata_sha256_by_object: dict[str, str] = {}
    opened_paths: list[str] = []
    metadata_cache.mkdir(parents=True, exist_ok=True)
    for row in panel:
        relative = row["metadata_path"]
        downloaded = Path(
            hf_hub_download(
                repo_id=dataset["repository"],
                repo_type="dataset",
                revision=dataset["revision"],
                filename=relative,
                local_dir=str(metadata_cache),
            )
        )
        raw = downloaded.read_bytes()
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid metadata JSON: {relative}") from error
        _require(isinstance(value, dict), f"metadata is not an object: {relative}")
        metadata_by_object[row["object_id"]] = value
        metadata_sha256_by_object[row["object_id"]] = hashlib.sha256(raw).hexdigest()
        opened_paths.append(relative)
    return {
        "resolved_revision": info.sha,
        "raw_objects": available,
        "metadata_by_object": metadata_by_object,
        "metadata_sha256_by_object": metadata_sha256_by_object,
        "opened_paths": opened_paths,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_revision(repository: Path) -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--touched-exclusion-output", type=Path, required=True)
    parser.add_argument("--metadata-cache", type=Path)
    parser.add_argument("--snapshot-json", type=Path)
    parser.add_argument("--implementation-revision")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repository = args.repository.resolve()
    protocol, excluded_hashes = load_protocol(args.protocol, repository=repository)
    if args.snapshot_json is not None:
        snapshot = _load_json(args.snapshot_json.resolve())
    else:
        _require(args.metadata_cache is not None, "live selection needs metadata-cache")
        snapshot = _download_snapshot(
            protocol,
            excluded_hashes=excluded_hashes,
            metadata_cache=args.metadata_cache.resolve(),
        )
    selection, touched = build_selection(
        snapshot,
        repository=repository,
        protocol_path=args.protocol,
        implementation_revision=(
            args.implementation_revision or _git_revision(repository)
        ),
    )
    _write_json(args.output.resolve(), selection)
    _write_json(args.touched_exclusion_output.resolve(), touched)
    print(
        json.dumps(
            {
                "candidate_panel_count": len(selection["candidate_panel"]),
                "target_roster_count": len(selection["target_roster"]),
                "roster_sha256": selection["roster_sha256"],
                "selection_sha256": selection["selection_sha256"],
                "camera_media_decoded": False,
                "target_outcomes_opened": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
