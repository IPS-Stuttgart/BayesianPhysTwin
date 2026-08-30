#!/usr/bin/env python3
"""Bind Deform360 timestamp clusters to ordered action metadata target-blindly.

For incomplete objects, every monotone missing-sequence alignment permitted by
protocol is retained.  A pair is called cross-action only when its semantic
action signatures differ under every retained alignment.  The program reads
small JSON metadata only; it never decodes media or loads numerical arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

PROTOCOL_SCHEMA: Final = "bayesian-phystwin/deform360-rope-action-mapping"
RESULT_SCHEMA: Final = "bayesian-phystwin/deform360-rope-action-mapping-result"
ACTION_HINTS: Final = (
    "action",
    "description",
    "instruction",
    "interaction",
    "label",
    "manipulation",
    "motion",
    "name",
    "primitive",
    "task",
    "type",
)
EXCLUDED_HINTS: Final = (
    "camera",
    "capture",
    "date",
    "episode_id",
    "episode_idx",
    "index",
    "material",
    "object",
    "serial",
    "time",
    "timestamp",
)


class MappingError(ValueError):
    """Raised when a frozen target-blind mapping contract is invalid."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MappingError(message)


def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise MappingError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=pairs_hook
        )
    except MappingError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MappingError(f"cannot read JSON: {path}") from error
    require(type(value) is dict, f"JSON root must be an object: {path}")
    return value


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_id(value: Mapping[str, object], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = read_json(path.resolve())
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "unexpected protocol schema")
    require(protocol.get("schema_version") == 3, "unsupported protocol version")
    require(
        protocol.get("status")
        == "frozen-before-action-metadata-resolution-after-name-only-roster",
        "protocol status changed",
    )
    objects = protocol.get("object_ids")
    require(
        type(objects) is list
        and len(objects) == 4
        and objects == sorted(set(objects))
        and all(type(item) is str for item in objects),
        "object roster must contain four sorted unique strings",
    )
    root = protocol.get("root")
    require(type(root) is str and root.startswith("/"), "root must be absolute")
    mapping = protocol.get("mapping")
    require(type(mapping) is dict, "mapping policy missing")
    require(mapping.get("chronological_monotone_alignment") is True, "bad mapping")
    maximum_missing = mapping.get("maximum_missing_sequence_records")
    require(
        type(maximum_missing) is int and 0 <= maximum_missing <= 2,
        "invalid missing-sequence budget",
    )
    expected_boundary = {
        "directory_and_filename_inventory_only": True,
        "small_metadata_json_allowed": True,
        "media_payload_decoded": False,
        "numeric_arrays_loaded": False,
        "large_payloads_hashed": False,
        "target_future_opened": False,
        "score_bearing_outcomes_used": False,
    }
    require(
        protocol.get("information_boundary") == expected_boundary,
        "information boundary changed",
    )
    return protocol


def find_object_records(value: object) -> list[dict[str, Any]]:
    if type(value) is dict:
        for key in ("objects", "object_records"):
            candidate = value.get(key)
            if (
                type(candidate) is list
                and candidate
                and all(type(item) is dict and "object_id" in item for item in candidate)
            ):
                return list(candidate)
        for child in value.values():
            found = find_object_records(child)
            if found:
                return found
    elif type(value) is list:
        for child in value:
            found = find_object_records(child)
            if found:
                return found
    return []


def scalar_leaves(
    value: object,
    path: tuple[str, ...] = (),
    depth: int = 0,
) -> list[tuple[tuple[str, ...], object]]:
    if depth > 7:
        return []
    if type(value) is dict:
        found: list[tuple[tuple[str, ...], object]] = []
        for key, child in value.items():
            if type(key) is str:
                found.extend(scalar_leaves(child, (*path, key), depth + 1))
        return found
    if type(value) is list:
        found = []
        for index, child in enumerate(value[:64]):
            found.extend(scalar_leaves(child, (*path, str(index)), depth + 1))
        return found
    if type(value) in {str, int, float, bool}:
        return [(path, value)]
    return []


def semantic_values(record: Mapping[str, object]) -> list[dict[str, str]]:
    rows: dict[tuple[str, str], dict[str, str]] = {}
    for path, raw in scalar_leaves(record):
        if not path:
            continue
        full_path = ".".join(path)
        lowered = full_path.casefold()
        leaf = path[-1].casefold()
        if not any(hint in lowered for hint in ACTION_HINTS):
            continue
        if any(hint in leaf for hint in EXCLUDED_HINTS):
            continue
        if type(raw) is str:
            value = " ".join(raw.split())
        elif type(raw) in {int, float} and type(raw) is not bool:
            value = str(raw)
        else:
            continue
        if not value or len(value) > 200:
            continue
        rows[(full_path, value.casefold())] = {"path": full_path, "value": value}
    return [rows[key] for key in sorted(rows)]


def signature(record: Mapping[str, object]) -> tuple[str | None, list[dict[str, str]]]:
    values = semantic_values(record)
    if not values:
        return None, values
    normalized = [row["value"].casefold() for row in values]
    return hashlib.sha256(canonical_bytes(normalized)).hexdigest(), values


def bimanual(record: Mapping[str, object]) -> bool | None:
    for path, value in scalar_leaves(record):
        if path and path[-1].casefold() == "bimanual":
            if type(value) is bool:
                return value
            if type(value) is int and value in {0, 1}:
                return bool(value)
    return None


def sequence_records(metadata: Mapping[str, object]) -> tuple[str, list[dict[str, object]]]:
    for key in ("sequences", "episodes", "interactions", "trials"):
        candidate = metadata.get(key)
        if (
            type(candidate) is list
            and 2 <= len(candidate) <= 20
            and all(type(item) is dict for item in candidate)
        ):
            return key, list(candidate)
    raise MappingError("no ordered sequence-record list found")


def field_int(record: Mapping[str, object], names: Sequence[str], default: int) -> int:
    for name in names:
        value = record.get(name)
        if type(value) is int and value >= 0:
            return value
    return default


def normalized_episodes(record: Mapping[str, Any]) -> list[dict[str, int]]:
    raw = record.get("episodes")
    require(type(raw) is list, f"episodes missing for {record.get('object_id')}")
    episodes: list[dict[str, int]] = []
    for fallback, item in enumerate(raw):
        require(type(item) is dict, "episode record must be an object")
        episodes.append(
            {
                "episode_index": field_int(
                    item, ("episode_index", "cluster_index", "index"), fallback
                ),
                "camera_pairs": field_int(
                    item,
                    (
                        "camera_pairs",
                        "camera_pair_count",
                        "camera_stream_count",
                        "camera_count",
                    ),
                    0,
                ),
                "tactile_pairs": field_int(
                    item,
                    (
                        "tactile_pairs",
                        "tactile_pair_count",
                        "tactile_stream_count",
                        "tactile_count",
                    ),
                    0,
                ),
            }
        )
    episodes.sort(key=lambda item: item["episode_index"])
    return episodes


def alignments(
    episode_count: int,
    sequence_count: int,
    maximum_missing: int,
) -> list[tuple[int, ...]]:
    require(episode_count >= 2, "at least two episodes are required")
    require(sequence_count >= episode_count, "fewer sequence records than episodes")
    require(
        sequence_count - episode_count <= maximum_missing,
        "missing-sequence budget exceeded",
    )
    return list(itertools.combinations(range(sequence_count), episode_count))


def choose_pair(
    episodes: Sequence[Mapping[str, int]],
    mappings: Sequence[Sequence[int]],
    signatures: Sequence[str | None],
) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    for source in range(len(episodes) - 1):
        for target in range(source + 1, len(episodes)):
            comparisons: list[bool] = []
            source_indices: set[int] = set()
            target_indices: set[int] = set()
            for mapping in mappings:
                source_index = int(mapping[source])
                target_index = int(mapping[target])
                source_indices.add(source_index)
                target_indices.add(target_index)
                source_signature = signatures[source_index]
                target_signature = signatures[target_index]
                comparisons.append(
                    source_signature is not None
                    and target_signature is not None
                    and source_signature != target_signature
                )
            source_episode = episodes[source]
            target_episode = episodes[target]
            candidates.append(
                {
                    "source_position": source,
                    "target_position": target,
                    "source_episode_index": source_episode["episode_index"],
                    "target_episode_index": target_episode["episode_index"],
                    "source_sequence_indices": sorted(source_indices),
                    "target_sequence_indices": sorted(target_indices),
                    "robustly_different_action": bool(comparisons)
                    and all(comparisons),
                    "minimum_camera_pairs": min(
                        source_episode["camera_pairs"], target_episode["camera_pairs"]
                    ),
                    "minimum_tactile_pairs": min(
                        source_episode["tactile_pairs"],
                        target_episode["tactile_pairs"],
                    ),
                    "temporal_separation": target - source,
                }
            )
    require(candidates, "no chronological source-target candidate")
    candidates.sort(
        key=lambda row: (
            -int(bool(row["robustly_different_action"])),
            -int(row["minimum_camera_pairs"]),
            -int(row["minimum_tactile_pairs"]),
            -int(row["temporal_separation"]),
            int(row["source_position"]),
            int(row["target_position"]),
        )
    )
    selected = dict(candidates[0])
    selected["candidate_pair_count"] = len(candidates)
    return selected


def resolve_object(
    protocol: Mapping[str, Any],
    roster_record: Mapping[str, Any],
) -> dict[str, object]:
    object_id = roster_record.get("object_id")
    require(type(object_id) is str, "object ID missing")
    episodes = normalized_episodes(roster_record)
    metadata_path = Path(protocol["root"]) / object_id / "metadata.json"
    require(metadata_path.is_file(), f"metadata missing: {object_id}")
    require(not metadata_path.is_symlink(), f"metadata is a symlink: {object_id}")
    maximum_bytes = int(protocol["maximum_metadata_json_bytes"])
    require(
        0 < metadata_path.stat().st_size <= maximum_bytes,
        f"metadata outside size policy: {object_id}",
    )
    raw = metadata_path.read_bytes()
    metadata = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs_hook)
    require(type(metadata) is dict, f"metadata root is not an object: {object_id}")
    container, sequences = sequence_records(metadata)
    summaries: list[dict[str, object]] = []
    signatures: list[str | None] = []
    for index, record in enumerate(sequences):
        action_signature, values = signature(record)
        signatures.append(action_signature)
        summaries.append(
            {
                "sequence_index": index,
                "record_sha256": hashlib.sha256(canonical_bytes(record)).hexdigest(),
                "action_signature": action_signature,
                "semantic_action_values": values,
                "bimanual": bimanual(record),
                "top_level_keys": sorted(record),
            }
        )
    mapping_candidates = alignments(
        len(episodes),
        len(sequences),
        int(protocol["mapping"]["maximum_missing_sequence_records"]),
    )
    pair = choose_pair(episodes, mapping_candidates, signatures)
    source_bimanual: set[bool] = set()
    target_bimanual: set[bool] = set()
    for mapping in mapping_candidates:
        source_value = summaries[int(mapping[int(pair["source_position"])])]["bimanual"]
        target_value = summaries[int(mapping[int(pair["target_position"])])]["bimanual"]
        if type(source_value) is bool:
            source_bimanual.add(source_value)
        if type(target_value) is bool:
            target_bimanual.add(target_value)
    pair["source_bimanual_values"] = sorted(source_bimanual)
    pair["target_bimanual_values"] = sorted(target_bimanual)
    pair["bimanual_resolved"] = (
        len(source_bimanual) == 1 and len(target_bimanual) == 1
    )
    return {
        "object_id": object_id,
        "metadata_path": str(metadata_path),
        "metadata_sha256": hashlib.sha256(raw).hexdigest(),
        "metadata_top_level_shape": {
            key: type(value).__name__ for key, value in sorted(metadata.items())
        },
        "sequence_container": container,
        "episode_count": len(episodes),
        "sequence_record_count": len(sequences),
        "missing_sequence_record_count": len(sequences) - len(episodes),
        "alignment_candidate_count": len(mapping_candidates),
        "exact_ordinal_mapping": len(mapping_candidates) == 1,
        "semantically_labelled_sequence_count": sum(
            value is not None for value in signatures
        ),
        "all_sequences_semantically_labelled": all(
            value is not None for value in signatures
        ),
        "episodes": episodes,
        "sequence_summaries": summaries,
        "selected_pair": pair,
    }


def build_result(
    protocol: Mapping[str, Any],
    roster: Mapping[str, Any],
    revision: str,
) -> dict[str, object]:
    records = find_object_records(roster)
    require(records, "no object records found in roster")
    by_id = {record.get("object_id"): record for record in records}
    require(
        set(protocol["object_ids"]).issubset(by_id),
        "registered object missing from roster",
    )
    objects = [
        resolve_object(protocol, by_id[object_id])
        for object_id in protocol["object_ids"]
    ]
    robust = sum(
        bool(record["selected_pair"]["robustly_different_action"])
        for record in objects
    )
    exact = sum(bool(record["exact_ordinal_mapping"]) for record in objects)
    semantic = sum(
        bool(record["all_sequences_semantically_labelled"]) for record in objects
    )
    bimanual_count = sum(
        bool(record["selected_pair"]["bimanual_resolved"]) for record in objects
    )
    if robust == 4:
        decision = "four-object-robust-cross-action-roster-ready"
    elif robust >= 2:
        decision = "partial-robust-cross-action-roster-ready"
    elif robust == 1:
        decision = "single-object-robust-cross-action-roster-ready"
    else:
        decision = "cross-episode-roster-ready-action-separation-unresolved"
    result: dict[str, object] = {
        "schema": RESULT_SCHEMA,
        "schema_version": 3,
        "protocol_id": protocol["protocol_id"],
        "repository_revision": revision,
        "source_roster_run_id": protocol["source_roster"]["run_id"],
        "source_roster_result_id": protocol["source_roster"]["result_id"],
        "source_roster_observed_result_id": roster.get("result_id"),
        "objects": objects,
        "summary": {
            "registered_object_count": 4,
            "selected_pair_count": len(objects),
            "exact_ordinal_mapping_object_count": exact,
            "ambiguous_monotone_mapping_object_count": 4 - exact,
            "all_sequences_semantically_labelled_object_count": semantic,
            "robust_cross_action_selected_object_count": robust,
            "bimanual_resolved_selected_object_count": bimanual_count,
            "decision": decision,
        },
        "information_boundary": protocol["information_boundary"],
        "claim_boundary": protocol["claim_boundary"],
    }
    result["result_id"] = content_id(result, "result_id")
    return result


def render_report(result: Mapping[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# Deform360 action-mapping result v3",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Result ID: `{result['result_id']}`",
        f"- Exact ordinal mappings: `{summary['exact_ordinal_mapping_object_count']}/4`",
        f"- Robust cross-action pairs: `{summary['robust_cross_action_selected_object_count']}/4`",
        f"- Bimanual state resolved: `{summary['bimanual_resolved_selected_object_count']}/4`",
        "",
        "| Object | Episodes | Sequences | Alignments | Source | Target | Robust cross-action |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for record in result["objects"]:
        pair = record["selected_pair"]
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | `{}` |".format(
                record["object_id"],
                record["episode_count"],
                record["sequence_record_count"],
                record["alignment_candidate_count"],
                pair["source_episode_index"],
                pair["target_episode_index"],
                str(pair["robustly_different_action"]).lower(),
            )
        )
    lines.extend(
        [
            "",
            "No media, numerical array, target future, or score-bearing outcome was opened.",
        ]
    )
    return "\n".join(lines) + "\n"


def self_test() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "raw"
        object_ids = ["001-rope", "002-rope-silk", "003-cable", "081-stripe-rope"]
        roster_objects: list[dict[str, object]] = []
        labels = [
            "pull",
            "twist",
            "lift",
            "drag",
            "bend",
            "shake",
            "stretch",
            "push",
            "wrap",
            "fold",
        ]
        for position, object_id in enumerate(object_ids):
            object_dir = root / object_id
            object_dir.mkdir(parents=True)
            sequences = [
                {
                    "manipulation_type": label,
                    "bimanual": bool(index % 2),
                    "sequence_id": index,
                }
                for index, label in enumerate(labels)
            ]
            (object_dir / "metadata.json").write_text(
                json.dumps({"sequences": sequences}), encoding="utf-8"
            )
            episode_count = 10 if position < 2 else 9
            roster_objects.append(
                {
                    "object_id": object_id,
                    "episodes": [
                        {
                            "episode_index": index,
                            "camera_pairs": 37,
                            "tactile_pairs": 4,
                        }
                        for index in range(episode_count)
                    ],
                }
            )
        protocol = {
            "schema": PROTOCOL_SCHEMA,
            "schema_version": 3,
            "protocol_id": "fixture",
            "status": "frozen-before-action-metadata-resolution-after-name-only-roster",
            "root": str(root),
            "object_ids": object_ids,
            "maximum_metadata_json_bytes": 1048576,
            "mapping": {
                "chronological_monotone_alignment": True,
                "maximum_missing_sequence_records": 2,
            },
            "source_roster": {"run_id": 1, "result_id": "fixture"},
            "information_boundary": {
                "directory_and_filename_inventory_only": True,
                "small_metadata_json_allowed": True,
                "media_payload_decoded": False,
                "numeric_arrays_loaded": False,
                "large_payloads_hashed": False,
                "target_future_opened": False,
                "score_bearing_outcomes_used": False,
            },
            "claim_boundary": "fixture",
        }
        protocol_path = Path(temporary) / "protocol.json"
        write_json(protocol_path, protocol)
        result = build_result(
            load_protocol(protocol_path),
            {"result_id": "fixture", "objects": roster_objects},
            "1" * 40,
        )
        require(
            result["summary"]["exact_ordinal_mapping_object_count"] == 2,
            "fixture exact-mapping count changed",
        )
        require(
            result["summary"]["robust_cross_action_selected_object_count"] == 4,
            "fixture robust cross-action count changed",
        )
    print("self-test passed")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--roster-result", type=Path)
    parser.add_argument("--repository-revision")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    if arguments.self_test:
        self_test()
        return 0
    require(arguments.protocol is not None, "--protocol is required")
    require(arguments.roster_result is not None, "--roster-result is required")
    require(arguments.repository_revision is not None, "--repository-revision is required")
    require(arguments.output is not None, "--output is required")
    protocol = load_protocol(arguments.protocol)
    roster = read_json(arguments.roster_result.resolve())
    require(
        roster.get("result_id") == protocol["source_roster"]["result_id"],
        "source roster result ID changed",
    )
    result = build_result(protocol, roster, arguments.repository_revision)
    arguments.output.mkdir(parents=True, exist_ok=False)
    write_json(arguments.output / "action_mapping.json", result)
    selected: dict[str, object] = {
        "schema": "bayesian-phystwin/deform360-rope-selected-pairs-v3",
        "schema_version": 3,
        "source_result_id": result["result_id"],
        "pairs": [
            {"object_id": record["object_id"], **record["selected_pair"]}
            for record in result["objects"]
        ],
    }
    selected["manifest_id"] = content_id(selected, "manifest_id")
    write_json(arguments.output / "selected_pairs_v3.json", selected)
    (arguments.output / "report.md").write_text(
        render_report(result), encoding="utf-8", newline="\n"
    )
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MappingError as error:
        print(f"Deform360 action mapping failed: {error}", file=os.sys.stderr)
        raise SystemExit(2) from error
