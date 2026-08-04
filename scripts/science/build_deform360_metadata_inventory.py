#!/usr/bin/env python3
"""Build a target-blind, names-only inventory of a mounted Deform360 cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

INVENTORY_SCHEMA = "bayesian-phystwin/deform360-metadata-inventory-v1"
PROTOCOL_SCHEMA = "bayesian-phystwin/deform360-metadata-preflight-protocol"
NUMERIC_SUFFIXES = (".h5", ".npy", ".npz", ".ply")
_EPISODE_ALIAS = re.compile(
    r"^(?P<object>\d{3}-.+?)(?:-ep|_episode_)(?P<episode>\d{1,4})(?:\D.*)?$",
    re.IGNORECASE,
)
_GENERIC_EPISODE = re.compile(r"(?:episode[_-]?|ep)(\d{1,4})", re.IGNORECASE)


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


def _result_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("inventory_sha256", None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected a JSON object: {path}")
    return value


def load_preflight_protocol(path: Path) -> dict[str, Any]:
    """Load and validate the locked metadata-only protocol."""

    protocol = _load_json(path.resolve())
    _require(
        protocol.get("schema") == PROTOCOL_SCHEMA,
        "unexpected Deform360 metadata-preflight protocol schema",
    )
    _require(protocol.get("schema_version") == 1, "unsupported protocol version")
    _require(
        protocol.get("status") == "locked-before-new-dataset-payload-access",
        "metadata preflight must remain locked before payload access",
    )
    suffixes = tuple(protocol.get("numeric_suffixes", ()))
    _require(
        suffixes == NUMERIC_SUFFIXES,
        "metadata preflight numeric suffix contract changed",
    )
    boundary = protocol.get("information_boundary")
    _require(isinstance(boundary, dict), "information_boundary must be an object")
    _require(
        boundary.get("dataset_payload_opened") is False
        and boundary.get("names_and_directory_structure_only") is True
        and boundary.get("reserved_target_outcomes_opened") is False,
        "metadata preflight information boundary changed",
    )
    prior = protocol.get("prior_protocols")
    _require(isinstance(prior, dict), "prior_protocols must be an object")
    _require(set(prior) == {"v1", "v2"}, "expected v1 and v2 prior protocols")
    for name, record in prior.items():
        _require(isinstance(record, dict), f"prior protocol {name} must be an object")
        _require(
            isinstance(record.get("path"), str) and record["path"],
            f"prior protocol {name} path is missing",
        )
        expected = record.get("expected_config_sha256")
        _require(
            isinstance(expected, str)
            and len(expected) == 64
            and all(character in "0123456789abcdef" for character in expected),
            f"prior protocol {name} checksum is invalid",
        )
    return protocol


def _cohort_objects(value: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()
    for records in value.values():
        _require(isinstance(records, list), "cohort strata must contain lists")
        for record in records:
            _require(isinstance(record, dict), "cohort records must be objects")
            object_id = record.get("object_id")
            _require(
                isinstance(object_id, str) and object_id,
                "cohort object_id must be a nonempty string",
            )
            result.add(object_id)
    return result


def _load_prior_context(
    repository: Path,
    protocol: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    loaded: dict[str, dict[str, Any]] = {}
    source_records: dict[str, dict[str, str]] = {}
    for name, record in protocol["prior_protocols"].items():
        source_path = repository / str(record["path"])
        payload = _load_json(source_path)
        expected = str(record["expected_config_sha256"])
        _require(
            payload.get("config_sha256") == expected,
            f"prior protocol {name} config checksum changed",
        )
        config = payload.get("config")
        _require(isinstance(config, dict), f"prior protocol {name} lacks config")
        loaded[name] = config
        source_records[name] = {
            "path": str(record["path"]),
            "config_sha256": expected,
        }

    v1 = loaded["v1"]
    v2 = loaded["v2"]
    open_or_reserved = set(map(str, v1.get("open_or_reserved_objects", ())))
    candidate_names: set[str] = set()
    candidate_pools = v1.get("candidate_pools")
    _require(isinstance(candidate_pools, dict), "v1 candidate_pools are missing")
    for values in candidate_pools.values():
        _require(isinstance(values, list), "candidate pool must be a list")
        candidate_names.update(map(str, values))

    prior_calibration = _cohort_objects(v1.get("calibration_cohort", {}))
    prior_calibration.update(_cohort_objects(v2.get("calibration_cohort", {})))
    reserved_target = _cohort_objects(v1.get("target_cohort", {}))
    reserved_target.update(_cohort_objects(v2.get("target_cohort", {})))

    known = open_or_reserved | candidate_names | prior_calibration | reserved_target
    _require(known, "prior protocols define no Deform360 objects")
    overlap = prior_calibration & reserved_target
    _require(not overlap, f"calibration and target objects overlap: {sorted(overlap)}")

    classifications: dict[str, str] = {}
    for object_id in sorted(known):
        if object_id in reserved_target:
            classification = "reserved_target"
        elif object_id in prior_calibration:
            classification = "prior_calibration"
        elif object_id in open_or_reserved:
            classification = "prior_open_or_reserved"
        else:
            classification = "candidate_name_only"
        classifications[object_id] = classification
    return classifications, source_records


def _identify(
    parts: Sequence[str],
    known_objects: set[str],
) -> tuple[str | None, int | None]:
    object_id: str | None = None
    episode: int | None = None
    for part in parts:
        if part in known_objects:
            object_id = part
            break
        match = _EPISODE_ALIAS.fullmatch(part)
        if match is not None and match.group("object") in known_objects:
            object_id = match.group("object")
            episode = int(match.group("episode"))
            break
    if object_id is None:
        return None, None
    if episode is None:
        for part in parts:
            match = _GENERIC_EPISODE.search(part)
            if match is not None:
                episode = int(match.group(1))
                break
    return object_id, episode


def _contract_hints(filename: str) -> tuple[str, ...]:
    lowered = filename.lower()
    hints: list[str] = []
    if lowered == "sampled_hulls.npz":
        hints.append("packed_visual_hulls")
    if "control_point" in lowered:
        hints.append("control_points")
    if "particle" in lowered:
        hints.append("particles")
    if "track" in lowered or lowered in {"vel.h5", "visibility.h5"}:
        hints.append("tracking")
    if "trajectory" in lowered or "positions" in lowered:
        hints.append("trajectory")
    if lowered == "robot.npz":
        hints.append("robot_state")
    if lowered == "frame_zero_points.npz":
        hints.append("frame_zero_points")
    if lowered == "rendered_depth.h5":
        hints.append("rendered_depth")
    if lowered == "mask_refined.h5":
        hints.append("object_mask")
    return tuple(sorted(set(hints)))


def build_metadata_inventory(
    data_root: Path,
    *,
    repository: Path,
    protocol_path: Path,
    revision: str | None = None,
) -> dict[str, Any]:
    """Inventory path names without opening or hashing dataset payloads."""

    root = data_root.expanduser().resolve()
    repo = repository.expanduser().resolve()
    protocol_file = protocol_path.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Deform360 data root is missing: {root}")
    if not repo.is_dir():
        raise FileNotFoundError(f"repository root is missing: {repo}")
    protocol = load_preflight_protocol(protocol_file)
    classifications, source_records = _load_prior_context(repo, protocol)
    known_objects = set(classifications)
    sample_limit = int(protocol["sample_path_limit_per_object"])
    _require(sample_limit >= 0, "sample_path_limit_per_object must be nonnegative")

    extension_counts: Counter[str] = Counter()
    top_level_roots: list[dict[str, str]] = []
    object_sample_paths: dict[str, list[str]] = defaultdict(list)
    object_numeric_paths: dict[str, list[str]] = defaultdict(list)
    object_numeric_counts: dict[str, Counter[str]] = defaultdict(Counter)
    object_contract_hints: dict[str, Counter[str]] = defaultdict(Counter)
    object_top_level_roots: dict[str, set[str]] = defaultdict(set)
    object_episodes: dict[str, set[int]] = defaultdict(set)
    total_files = 0
    total_directories = 0

    for path in sorted(root.iterdir(), key=lambda value: value.name):
        top_level_roots.append(
            {
                "name": path.name,
                "kind": "directory" if path.is_dir() else "file",
            }
        )

    ignored_directories = {".git", "__pycache__", "node_modules"}
    for directory, names, files in os.walk(root, followlinks=False):
        names[:] = sorted(name for name in names if name not in ignored_directories)
        total_directories += 1
        for name in sorted(files):
            total_files += 1
            suffix = Path(name).suffix.lower() or "<none>"
            extension_counts[suffix] += 1
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            parts = tuple(relative.split("/"))
            object_id, episode = _identify(parts, known_objects)
            if object_id is None:
                continue
            if len(object_sample_paths[object_id]) < sample_limit:
                object_sample_paths[object_id].append(relative)
            object_top_level_roots[object_id].add(parts[0])
            if episode is not None:
                object_episodes[object_id].add(episode)
            if suffix not in NUMERIC_SUFFIXES:
                continue
            object_numeric_paths[object_id].append(relative)
            object_numeric_counts[object_id][suffix] += 1
            for hint in _contract_hints(name):
                object_contract_hints[object_id][hint] += 1

    objects: list[dict[str, Any]] = []
    for object_id in sorted(known_objects):
        numeric_paths = sorted(set(object_numeric_paths[object_id]))
        sample_paths = sorted(set(object_sample_paths[object_id]))
        if not numeric_paths and not sample_paths:
            continue
        objects.append(
            {
                "object_id": object_id,
                "classification": classifications[object_id],
                "episode_ids_from_names": sorted(object_episodes[object_id]),
                "top_level_roots": sorted(object_top_level_roots[object_id]),
                "sample_paths": sample_paths,
                "numeric_paths": numeric_paths,
                "numeric_path_counts": dict(
                    sorted(object_numeric_counts[object_id].items())
                ),
                "contract_hint_counts": dict(
                    sorted(object_contract_hints[object_id].items())
                ),
            }
        )

    classification_counts = Counter(record["classification"] for record in objects)
    reserved_present = sorted(
        record["object_id"]
        for record in objects
        if record["classification"] == "reserved_target"
    )
    result: dict[str, Any] = {
        "schema": INVENTORY_SCHEMA,
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "repository_revision": revision,
        "dataset_root": str(root),
        "source_protocols": source_records,
        "information_boundary": {
            "dataset_payload_opened": False,
            "file_contents_hashed": False,
            "names_and_directory_structure_only": True,
            "reserved_target_outcomes_opened": False,
        },
        "known_object_vocabulary_count": len(known_objects),
        "recognized_object_count": len(objects),
        "classification_counts": dict(sorted(classification_counts.items())),
        "reserved_target_objects_present_by_name": reserved_present,
        "total_files_named": total_files,
        "total_directories_named": total_directories,
        "top_level_roots": top_level_roots,
        "extension_counts": dict(sorted(extension_counts.items())),
        "objects": objects,
    }
    result["inventory_sha256"] = _result_sha256(result)
    return result


def write_inventory(path: Path, inventory: Mapping[str, Any]) -> None:
    """Write a canonical, newline-terminated inventory artifact."""

    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(inventory, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    inventory = build_metadata_inventory(
        args.data_root,
        repository=args.repository,
        protocol_path=args.protocol,
        revision=os.environ.get("GITHUB_SHA"),
    )
    write_inventory(args.output, inventory)
    summary = {
        "inventory_sha256": inventory["inventory_sha256"],
        "recognized_object_count": inventory["recognized_object_count"],
        "classification_counts": inventory["classification_counts"],
        "reserved_target_objects_present_by_name": (
            inventory["reserved_target_objects_present_by_name"]
        ),
        "objects_with_numeric_paths": sum(
            bool(record["numeric_paths"]) for record in inventory["objects"]
        ),
        "contract_hint_counts": dict(
            sorted(
                sum(
                    (
                        Counter(record["contract_hint_counts"])
                        for record in inventory["objects"]
                    ),
                    Counter(),
                ).items()
            )
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
