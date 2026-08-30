#!/usr/bin/env python3
"""Prepare a timestamp-clustered Deform360 rope/cable roster without opening payloads."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from itertools import permutations
from pathlib import Path
from typing import Any, Final

SCHEMA: Final = "bayesian-phystwin/deform360-same-object-rope-pilot-result-v2"
PROTOCOL_SCHEMA: Final = "bayesian-phystwin/deform360-same-object-rope-pilot-v2"
CAMERA_RE: Final = re.compile(r"^brics-odroid-\d+_cam\d+$")
TACTILE_RE: Final = re.compile(r"^brics-odroid_tactile[^/]+$")
CAPTURE_TIMESTAMP_RE: Final = re.compile(r"_(\d{10,})$")
ACTION_KEY_FRAGMENTS: Final = ("action", "primitive", "task", "manipulation")
METADATA_BASENAMES: Final = {
    "action.json",
    "actions.json",
    "episode.json",
    "episodes.json",
    "info.json",
    "manifest.json",
    "metadata.json",
}


class PilotError(ValueError):
    """Raised when the registered target-blind pilot contract is invalid."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PilotError(message)


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_id(value: Mapping[str, Any], field: str = "result_id") -> str:
    payload = dict(value)
    payload.pop(field, None)
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, child in pairs:
        if key in value:
            raise PilotError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs_hook,
        )
    except PilotError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PilotError(f"cannot read JSON: {path}") from error
    require(type(value) is dict, f"JSON root must be an object: {path}")
    return value


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = read_json(path.resolve())
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "unexpected protocol schema")
    require(protocol.get("schema_version") == 2, "unsupported protocol version")
    require(
        protocol.get("status")
        == "frozen-before-payload-decoding-after-name-only-coverage-amendment",
        "protocol is not frozen at the amended target-blind boundary",
    )
    object_ids = protocol.get("object_ids")
    require(
        type(object_ids) is list
        and len(object_ids) == 4
        and object_ids == sorted(set(object_ids))
        and all(type(item) is str for item in object_ids),
        "object_ids must contain four sorted unique strings",
    )
    root = protocol.get("root")
    require(type(root) is str and root.startswith("/"), "root must be absolute")
    boundary = protocol.get("information_boundary")
    require(type(boundary) is dict, "information boundary is missing")
    expected_false = (
        "media_payload_decoded",
        "numeric_arrays_loaded",
        "large_payloads_hashed",
        "target_future_opened",
        "score_bearing_outcomes_used",
    )
    for key in expected_false:
        require(boundary.get(key) is False, f"information boundary opened: {key}")
    eligibility = protocol.get("eligibility")
    require(type(eligibility) is dict, "eligibility contract is missing")
    integer_fields = (
        "maximum_interstream_capture_gap",
        "minimum_camera_pairs_per_episode",
        "preferred_camera_pairs_per_episode",
        "minimum_tactile_pairs_per_episode",
        "minimum_eligible_episodes_per_object",
        "minimum_objects_for_pilot",
    )
    for key in integer_fields:
        require(
            type(eligibility.get(key)) is int and eligibility[key] > 0,
            f"bad eligibility field: {key}",
        )
    require(
        eligibility["minimum_camera_pairs_per_episode"]
        <= eligibility["preferred_camera_pairs_per_episode"],
        "minimum camera support exceeds preferred support",
    )
    require(
        eligibility["minimum_objects_for_pilot"] == len(object_ids),
        "the pilot must require all registered objects",
    )
    amendment = protocol.get("amendment")
    require(type(amendment) is dict, "coverage amendment is missing")
    require(
        amendment.get("based_on_run_id") == 33325025964,
        "unexpected coverage-amendment source",
    )
    require(
        amendment.get("target_outcomes_used") is False,
        "coverage amendment used target outcomes",
    )
    return protocol


def safe_iterdir(path: Path) -> tuple[tuple[Path, ...], str | None]:
    try:
        return tuple(sorted(path.iterdir(), key=lambda item: item.name)), None
    except OSError as error:
        return (), f"{type(error).__name__}: {error}"


def capture_timestamp(stem: str) -> int | None:
    match = CAPTURE_TIMESTAMP_RE.search(stem)
    return int(match.group(1)) if match is not None else None


def paired_capture_records(
    directory: Path,
    payload_suffix: str,
    *,
    modality: str,
    exclude_median: bool = False,
) -> tuple[list[dict[str, object]], list[str]]:
    entries, error = safe_iterdir(directory)
    if error is not None:
        return [], [error]
    payloads: dict[str, Path] = {}
    timestamps: set[str] = set()
    for path in entries:
        try:
            regular = path.is_file() and not path.is_symlink()
        except OSError:
            regular = False
        if not regular:
            continue
        suffix = path.suffix.lower()
        if suffix == payload_suffix:
            if exclude_median and path.stem.lower().startswith("median_"):
                continue
            payloads[path.stem] = path
        elif suffix == ".txt":
            timestamps.add(path.stem)
    records: list[dict[str, object]] = []
    errors: list[str] = []
    for stem in sorted(payloads.keys() & timestamps):
        value = capture_timestamp(stem)
        if value is None:
            errors.append(f"unparseable capture timestamp: {directory / stem}")
            continue
        records.append(
            {
                "modality": modality,
                "stream": directory.name,
                "stem": stem,
                "capture_timestamp": value,
            }
        )
    return records, errors


def cluster_capture_records(
    records: Sequence[Mapping[str, object]],
    maximum_gap: int,
) -> list[dict[str, object]]:
    ordered = sorted(
        records,
        key=lambda row: (
            int(row["capture_timestamp"]),
            str(row["modality"]),
            str(row["stream"]),
            str(row["stem"]),
        ),
    )
    groups: list[list[Mapping[str, object]]] = []
    current: list[Mapping[str, object]] = []
    previous: int | None = None
    for record in ordered:
        timestamp = int(record["capture_timestamp"])
        if previous is not None and timestamp - previous > maximum_gap:
            groups.append(current)
            current = []
        current.append(record)
        previous = timestamp
    if current:
        groups.append(current)

    clustered: list[dict[str, object]] = []
    for index, group in enumerate(groups):
        timestamps = [int(row["capture_timestamp"]) for row in group]
        camera_streams = sorted(
            {
                str(row["stream"])
                for row in group
                if row["modality"] == "camera"
            }
        )
        tactile_streams = sorted(
            {
                str(row["stream"])
                for row in group
                if row["modality"] == "tactile"
            }
        )
        stream_keys = [
            (str(row["modality"]), str(row["stream"]))
            for row in group
        ]
        duplicate_stream_records = len(stream_keys) - len(set(stream_keys))
        clustered.append(
            {
                "episode_key": f"episode_{index:04d}",
                "episode_index": index,
                "capture_timestamp_min": min(timestamps),
                "capture_timestamp_max": max(timestamps),
                "capture_timestamp_span": max(timestamps) - min(timestamps),
                "camera_pairs": len(camera_streams),
                "tactile_pairs": len(tactile_streams),
                "camera_streams": camera_streams,
                "tactile_streams": tactile_streams,
                "duplicate_stream_records": duplicate_stream_records,
                "capture_records": [dict(row) for row in group],
            }
        )
    return clustered


def scalar_text(value: object) -> str | None:
    if type(value) is str:
        stripped = " ".join(value.split())
        return stripped if stripped else None
    if type(value) in {int, float}:
        return str(value)
    return None


def first_action_text(
    value: object,
    *,
    action_context: bool = False,
) -> str | None:
    if action_context:
        text = scalar_text(value)
        if text is not None:
            return text
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                continue
            direct = any(
                fragment in key.casefold() for fragment in ACTION_KEY_FRAGMENTS
            )
            text = first_action_text(
                child,
                action_context=action_context or direct,
            )
            if text is not None:
                return text
    elif type(value) is list:
        for child in value:
            text = first_action_text(
                child,
                action_context=action_context,
            )
            if text is not None:
                return text
    return None


def action_sequence_candidates(
    value: object,
    episode_count: int,
    *,
    path: tuple[str, ...] = (),
) -> list[tuple[str, list[str]]]:
    candidates: list[tuple[str, list[str]]] = []
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                continue
            child_path = (*path, key)
            action_key = any(
                fragment in key.casefold() for fragment in ACTION_KEY_FRAGMENTS
            )
            if action_key and type(child) is list and len(child) == episode_count:
                sequence = [first_action_text(item) for item in child]
                if all(text is not None for text in sequence):
                    candidates.append(
                        (".".join(child_path), [str(text) for text in sequence])
                    )
            if action_key and type(child) is dict and len(child) == episode_count:
                numeric: list[tuple[int, object]] = []
                for raw_key, item in child.items():
                    if type(raw_key) is str and raw_key.isdigit():
                        numeric.append((int(raw_key), item))
                if len(numeric) == episode_count:
                    numeric.sort()
                    sequence = [first_action_text(item) for _, item in numeric]
                    if all(text is not None for text in sequence):
                        candidates.append(
                            (".".join(child_path), [str(text) for text in sequence])
                        )
            candidates.extend(
                action_sequence_candidates(
                    child,
                    episode_count,
                    path=child_path,
                )
            )
    elif type(value) is list:
        if len(value) == episode_count:
            sequence = [first_action_text(item) for item in value]
            if all(text is not None for text in sequence):
                candidates.append(
                    (".".join(path) or "<root>", [str(text) for text in sequence])
                )
        for index, child in enumerate(value):
            candidates.extend(
                action_sequence_candidates(
                    child,
                    episode_count,
                    path=(*path, str(index)),
                )
            )
    return candidates


def all_action_labels(value: object) -> set[str]:
    labels: set[str] = set()
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                continue
            if any(fragment in key.casefold() for fragment in ACTION_KEY_FRAGMENTS):
                text = scalar_text(child)
                if text is not None and len(text) <= 160:
                    labels.add(text)
                elif type(child) is list:
                    for item in child:
                        text = first_action_text(item)
                        if text is not None and len(text) <= 160:
                            labels.add(text)
            labels.update(all_action_labels(child))
    elif type(value) is list:
        for child in value:
            labels.update(all_action_labels(child))
    return labels


def scan_metadata(
    object_dir: Path,
    maximum_bytes: int,
    episode_count: int,
) -> tuple[list[dict[str, object]], list[str] | None, set[str], list[str]]:
    records: list[dict[str, object]] = []
    candidates: list[tuple[str, list[str]]] = []
    labels: set[str] = set()
    errors: list[str] = []
    base_depth = len(object_dir.parts)

    def onerror(error: OSError) -> None:
        errors.append(f"{type(error).__name__}: {error}")

    try:
        for directory, names, files in os.walk(
            object_dir,
            followlinks=False,
            onerror=onerror,
        ):
            current = Path(directory)
            depth = len(current.parts) - base_depth
            names[:] = sorted(
                name for name in names if name not in {".git", "__pycache__"}
            )
            if depth >= 5:
                names[:] = []
            for name in sorted(files):
                if name.casefold() not in METADATA_BASENAMES:
                    continue
                path = current / name
                try:
                    if path.is_symlink() or not path.is_file():
                        continue
                    size = path.stat().st_size
                    if size <= 0 or size > maximum_bytes:
                        records.append(
                            {
                                "path": str(path),
                                "parsed": False,
                                "reason": "size-outside-policy",
                            }
                        )
                        continue
                    raw = path.read_bytes()
                    value = json.loads(
                        raw.decode("utf-8"),
                        object_pairs_hook=pairs_hook,
                    )
                    file_candidates = action_sequence_candidates(
                        value,
                        episode_count,
                    )
                    candidates.extend(
                        (f"{path}:{candidate_path}", sequence)
                        for candidate_path, sequence in file_candidates
                    )
                    file_labels = all_action_labels(value)
                    labels.update(file_labels)
                    records.append(
                        {
                            "path": str(path),
                            "parsed": True,
                            "sha256": hashlib.sha256(raw).hexdigest(),
                            "action_sequence_candidate_count": len(file_candidates),
                            "action_label_count": len(file_labels),
                        }
                    )
                except (
                    OSError,
                    UnicodeError,
                    json.JSONDecodeError,
                    PilotError,
                ) as error:
                    records.append(
                        {
                            "path": str(path),
                            "parsed": False,
                            "reason": type(error).__name__,
                        }
                    )
    except OSError as error:
        errors.append(f"{type(error).__name__}: {error}")

    candidates.sort(key=lambda item: (item[0], item[1]))
    sequence = candidates[0][1] if candidates else None
    return records, sequence, labels, sorted(set(errors))


def build_candidates(
    object_id: str,
    episodes: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    eligible = [episode for episode in episodes if episode["visual_eligible"] is True]
    candidates: list[dict[str, object]] = []
    for source, target in permutations(eligible, 2):
        source_actions = set(source["action_labels"])
        target_actions = set(target["action_labels"])
        different_action = bool(
            source_actions
            and target_actions
            and source_actions.isdisjoint(target_actions)
        )
        candidate: dict[str, object] = {
            "object_id": object_id,
            "source_episode_key": source["episode_key"],
            "target_episode_key": target["episode_key"],
            "source_episode_index": source["episode_index"],
            "target_episode_index": target["episode_index"],
            "source_action_labels": sorted(source_actions),
            "target_action_labels": sorted(target_actions),
            "different_action_labels": different_action,
            "minimum_camera_pairs": min(
                int(source["camera_pairs"]),
                int(target["camera_pairs"]),
            ),
            "minimum_tactile_pairs": min(
                int(source["tactile_pairs"]),
                int(target["tactile_pairs"]),
            ),
            "both_visuotactile_eligible": bool(
                source["visuotactile_eligible"]
                and target["visuotactile_eligible"]
            ),
            "target_future_opened": False,
        }
        candidate["pair_id"] = content_id(candidate, "pair_id")
        candidates.append(candidate)
    candidates.sort(
        key=lambda item: (
            -int(bool(item["different_action_labels"])),
            -int(bool(item["both_visuotactile_eligible"])),
            -int(item["minimum_camera_pairs"]),
            -int(item["minimum_tactile_pairs"]),
            int(item["source_episode_index"]),
            int(item["target_episode_index"]),
        )
    )
    return candidates


def scan_object(
    root: Path,
    object_id: str,
    eligibility: Mapping[str, Any],
    maximum_metadata_bytes: int,
) -> dict[str, object]:
    object_dir = root / object_id
    entries, access_error = safe_iterdir(object_dir)
    result: dict[str, object] = {
        "object_id": object_id,
        "object_dir": str(object_dir),
        "present": object_dir.exists(),
        "access_error": access_error,
        "capture_errors": [],
        "metadata": [],
        "metadata_errors": [],
        "global_action_labels": [],
        "action_sequence_resolved": False,
        "episodes": [],
        "candidate_pairs": [],
        "selected_pair": None,
    }
    if access_error is not None or not object_dir.is_dir():
        return result

    camera_dirs = [
        path
        for path in entries
        if path.is_dir() and not path.is_symlink() and CAMERA_RE.fullmatch(path.name)
    ]
    tactile_dirs = [
        path
        for path in entries
        if path.is_dir() and not path.is_symlink() and TACTILE_RE.fullmatch(path.name)
    ]
    capture_records: list[dict[str, object]] = []
    capture_errors: list[str] = []
    for directory in camera_dirs:
        records, errors = paired_capture_records(
            directory,
            ".mp4",
            modality="camera",
        )
        capture_records.extend(records)
        capture_errors.extend(errors)
    for directory in tactile_dirs:
        records, errors = paired_capture_records(
            directory,
            ".npy",
            modality="tactile",
            exclude_median=True,
        )
        capture_records.extend(records)
        capture_errors.extend(errors)

    maximum_gap = int(eligibility["maximum_interstream_capture_gap"])
    episodes = cluster_capture_records(capture_records, maximum_gap)
    metadata, action_sequence, labels, metadata_errors = scan_metadata(
        object_dir,
        maximum_metadata_bytes,
        len(episodes),
    )
    minimum_camera = int(eligibility["minimum_camera_pairs_per_episode"])
    minimum_tactile = int(eligibility["minimum_tactile_pairs_per_episode"])
    for episode in episodes:
        index = int(episode["episode_index"])
        action_labels = (
            [action_sequence[index]]
            if action_sequence is not None and index < len(action_sequence)
            else []
        )
        episode["object_id"] = object_id
        episode["action_labels"] = action_labels
        episode["visual_eligible"] = (
            int(episode["camera_pairs"]) >= minimum_camera
            and int(episode["duplicate_stream_records"]) == 0
        )
        episode["visuotactile_eligible"] = bool(
            episode["visual_eligible"]
            and int(episode["tactile_pairs"]) >= minimum_tactile
        )

    candidates = build_candidates(object_id, episodes)
    result.update(
        {
            "capture_errors": sorted(set(capture_errors)),
            "metadata": metadata,
            "metadata_errors": metadata_errors,
            "global_action_labels": sorted(labels),
            "action_sequence_resolved": action_sequence is not None,
            "camera_directory_count": len(camera_dirs),
            "tactile_directory_count": len(tactile_dirs),
            "paired_capture_record_count": len(capture_records),
            "episode_count": len(episodes),
            "visual_eligible_episode_count": sum(
                episode["visual_eligible"] is True for episode in episodes
            ),
            "visuotactile_eligible_episode_count": sum(
                episode["visuotactile_eligible"] is True for episode in episodes
            ),
            "episodes": episodes,
            "candidate_pairs": candidates,
            "selected_pair": candidates[0] if candidates else None,
        }
    )
    return result


def flatten_rows(
    objects: Sequence[Mapping[str, object]],
    field: str,
) -> list[dict[str, object]]:
    return [
        dict(row)
        for object_record in objects
        for row in object_record.get(field, [])
        if type(row) is dict
    ]


def csv_value(value: object) -> object:
    if type(value) in {list, dict}:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fields})


def build_result(
    protocol: Mapping[str, Any],
    revision: str,
) -> dict[str, object]:
    require(
        len(revision) == 40
        and all(character in "0123456789abcdef" for character in revision),
        "revision must be a full lowercase commit SHA",
    )
    root = Path(protocol["root"])
    require(root.is_dir(), f"dataset root unavailable: {root}")
    resolved_root = root.resolve(strict=True)
    eligibility = protocol["eligibility"]
    maximum_metadata_bytes = int(
        protocol["information_boundary"]["maximum_metadata_json_bytes"]
    )
    objects = [
        scan_object(
            resolved_root,
            object_id,
            eligibility,
            maximum_metadata_bytes,
        )
        for object_id in protocol["object_ids"]
    ]
    minimum_episodes = int(eligibility["minimum_eligible_episodes_per_object"])
    accessible = sum(
        record["access_error"] is None and record["present"] is True
        for record in objects
    )
    visual_pairable = sum(
        int(record.get("visual_eligible_episode_count", 0)) >= minimum_episodes
        for record in objects
    )
    visuotactile_pairable = sum(
        int(record.get("visuotactile_eligible_episode_count", 0))
        >= minimum_episodes
        for record in objects
    )
    selected_pairs = [
        record["selected_pair"]
        for record in objects
        if type(record.get("selected_pair")) is dict
    ]
    different_action = sum(
        pair["different_action_labels"] is True for pair in selected_pairs
    )
    minimum_objects = int(eligibility["minimum_objects_for_pilot"])
    if visuotactile_pairable >= minimum_objects:
        decision = "four-object-visuotactile-roster-ready"
    elif visual_pairable >= minimum_objects:
        decision = "four-object-visual-roster-ready"
    else:
        decision = "insufficient-four-object-visual-roster"
    result: dict[str, object] = {
        "schema": SCHEMA,
        "schema_version": 2,
        "protocol_id": protocol["protocol_id"],
        "repository_revision": revision,
        "dataset_root": str(resolved_root),
        "amendment": protocol["amendment"],
        "information_boundary": protocol["information_boundary"],
        "objects": objects,
        "selected_pairs": selected_pairs,
        "summary": {
            "registered_object_count": len(objects),
            "accessible_object_count": accessible,
            "episode_count": sum(int(record.get("episode_count", 0)) for record in objects),
            "candidate_pair_count": sum(
                len(record.get("candidate_pairs", [])) for record in objects
            ),
            "selected_pair_count": len(selected_pairs),
            "visual_pairable_object_count": visual_pairable,
            "visuotactile_pairable_object_count": visuotactile_pairable,
            "different_action_selected_pair_count": different_action,
            "action_sequence_resolved_object_count": sum(
                record.get("action_sequence_resolved") is True for record in objects
            ),
            "decision": decision,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    result["result_id"] = content_id(result)
    return result


def write_outputs(output: Path, result: Mapping[str, object]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    selected = result["selected_pairs"]
    (output / "selected_pairs.json").write_text(
        json.dumps(selected, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    objects = result["objects"]
    require(type(objects) is list, "result objects field is malformed")
    write_csv(output / "episodes.csv", flatten_rows(objects, "episodes"))
    write_csv(
        output / "candidate_pairs.csv",
        flatten_rows(objects, "candidate_pairs"),
    )
    summary = result["summary"]
    lines = [
        "## Target-blind Deform360 rope/cable roster v2",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Result ID: `{result['result_id']}`",
        f"- Accessible objects: `{summary['accessible_object_count']}`",
        f"- Clustered physical episodes: `{summary['episode_count']}`",
        f"- Visual-pairable objects: `{summary['visual_pairable_object_count']}`",
        (
            "- Visuotactile-pairable objects: "
            f"`{summary['visuotactile_pairable_object_count']}`"
        ),
        (
            "- Action sequences resolved: "
            f"`{summary['action_sequence_resolved_object_count']}`"
        ),
        "- Media payloads decoded: `false`",
        "- Numerical arrays loaded: `false`",
        "- Target futures opened: `false`",
        "",
        "| Object | Episodes | Visual | Visuotactile | Cameras | Tactile |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for record in objects:
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} |".format(
                record["object_id"],
                record.get("episode_count", 0),
                record.get("visual_eligible_episode_count", 0),
                record.get("visuotactile_eligible_episode_count", 0),
                record.get("camera_directory_count", 0),
                record.get("tactile_directory_count", 0),
            )
        )
    lines.extend(["", str(result["claim_boundary"])])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fixture_protocol(root: Path) -> dict[str, object]:
    return {
        "schema": PROTOCOL_SCHEMA,
        "schema_version": 2,
        "protocol_id": "fixture-v2",
        "status": (
            "frozen-before-payload-decoding-after-name-only-coverage-amendment"
        ),
        "runner_id": "fixture",
        "root": str(root),
        "object_ids": [
            "001-rope",
            "002-rope-silk",
            "003-cable",
            "081-stripe-rope",
        ],
        "information_boundary": {
            "directory_and_filename_inventory_only": True,
            "small_metadata_json_allowed": True,
            "maximum_metadata_json_bytes": 1048576,
            "media_payload_decoded": False,
            "numeric_arrays_loaded": False,
            "large_payloads_hashed": False,
            "target_future_opened": False,
            "score_bearing_outcomes_used": False,
        },
        "eligibility": {
            "maximum_interstream_capture_gap": 250000,
            "minimum_camera_pairs_per_episode": 12,
            "preferred_camera_pairs_per_episode": 37,
            "minimum_tactile_pairs_per_episode": 2,
            "minimum_eligible_episodes_per_object": 2,
            "minimum_objects_for_pilot": 4,
        },
        "amendment": {
            "based_on_run_id": 33325025964,
            "target_outcomes_used": False,
        },
        "claim_boundary": "fixture",
    }


def self_test() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "raw"
        protocol = fixture_protocol(root)
        for object_offset, object_id in enumerate(protocol["object_ids"]):
            object_dir = root / object_id
            object_dir.mkdir(parents=True)
            metadata = {
                "episodes": [
                    {"action": "pull left"},
                    {"action": "lift center"},
                    {"action": "push right"},
                ]
            }
            (object_dir / "metadata.json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            base = 1766000000000000 + object_offset * 1000000000
            for camera_index in range(12):
                stream = object_dir / f"brics-odroid-{camera_index + 1:03d}_cam0"
                stream.mkdir()
                for episode_index in range(3):
                    timestamp = (
                        base
                        + episode_index * 12000000
                        + camera_index * 1000
                    )
                    stem = f"{stream.name}_{timestamp}"
                    (stream / f"{stem}.mp4").write_bytes(b"fixture-video")
                    (stream / f"{stem}.txt").write_text("0\n", encoding="utf-8")
            for tactile_index in range(4):
                stream = object_dir / f"brics-odroid_tactile_{tactile_index}"
                stream.mkdir()
                for episode_index in range(3):
                    timestamp = (
                        base
                        + episode_index * 12000000
                        + 20000
                        + tactile_index * 1000
                    )
                    stem = f"{stream.name}_{timestamp}"
                    (stream / f"{stem}.npy").write_bytes(b"fixture-array")
                    (stream / f"{stem}.txt").write_text("0\n", encoding="utf-8")
        protocol_path = Path(temporary) / "protocol.json"
        protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
        loaded = load_protocol(protocol_path)
        result = build_result(loaded, "1" * 40)
        summary = result["summary"]
        require(
            summary["decision"] == "four-object-visuotactile-roster-ready",
            "fixture decision changed",
        )
        require(summary["episode_count"] == 12, "fixture cluster count changed")
        require(
            summary["visual_pairable_object_count"] == 4,
            "fixture visual object count changed",
        )
        require(
            summary["visuotactile_pairable_object_count"] == 4,
            "fixture tactile object count changed",
        )
        require(
            summary["different_action_selected_pair_count"] == 4,
            "fixture action pairing changed",
        )
        first_object = result["objects"][0]
        first_episode = first_object["episodes"][0]
        require(first_episode["camera_pairs"] == 12, "fixture camera count changed")
        require(first_episode["tactile_pairs"] == 4, "fixture tactile count changed")
        require(
            first_episode["duplicate_stream_records"] == 0,
            "fixture duplicate detection changed",
        )
    print("self-test passed")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    if arguments.self_test:
        self_test()
        return 0
    require(arguments.protocol is not None, "--protocol is required")
    require(arguments.revision is not None, "--revision is required")
    require(arguments.output is not None, "--output is required")
    protocol = load_protocol(arguments.protocol)
    result = build_result(protocol, arguments.revision)
    write_outputs(arguments.output, result)
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PilotError as error:
        print(f"Deform360 rope pilot v2 failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
