#!/usr/bin/env python3
"""Audit mounted Deform360 fragments without decoding dataset payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

SCHEMA: Final = "bayesian-phystwin/deform360-existing-data-eligibility"
PROTOCOL_SCHEMA: Final = (
    "bayesian-phystwin/deform360-existing-data-eligibility-protocol"
)
OBJECT_RE: Final = re.compile(r"^\d{3}-.+$")
CAMERA_RE: Final = re.compile(r"^brics-odroid-\d+_cam\d+$")
TACTILE_RE: Final = re.compile(r"^brics-odroid_tactile[^/]+$")
EPISODE_RE: Final = re.compile(r"^episode_(\d+)$", re.IGNORECASE)
ROBOT_HINTS: Final = ("action", "command", "gripper", "robot", "tcp_pose")
TARGET_HINTS: Final = (
    "control_point",
    "frame_zero_points",
    "particles",
    "pcd_clean",
    "positions",
    "track",
    "trajectory",
)


class AuditError(ValueError):
    """Raised when an audit contract fails closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise AuditError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream, object_pairs_hook=pairs_hook)
    except AuditError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AuditError(f"cannot read JSON: {path}") from error
    require(type(value) is dict, f"JSON root must be an object: {path}")
    return value


def canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def content_id(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("result_id", None)
    payload.pop("repository_revision", None)
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = read_json(path.resolve())
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "unexpected protocol schema")
    require(protocol.get("schema_version") == 1, "unsupported protocol version")
    require(
        protocol.get("status") == "frozen-before-fragment-payload-audit",
        "protocol is not frozen before audit",
    )
    expected_boundary = {
        "directory_and_filename_inventory_only": True,
        "small_metadata_json_allowed": True,
        "media_payload_decoded": False,
        "numeric_arrays_loaded": False,
        "large_payloads_hashed": False,
        "target_outcomes_used": False,
    }
    require(
        protocol.get("information_boundary") == expected_boundary,
        "information boundary changed",
    )
    runners = protocol.get("runners")
    require(
        type(runners) is dict and set(runners) == {"gpuserver4090", "gpuserver6000"},
        "runner roster changed",
    )
    object_ids: list[str] = []
    for runner in runners.values():
        roots = runner.get("roots") if type(runner) is dict else None
        require(type(roots) is list and roots, "runner roots must be nonempty")
        for record in roots:
            require(
                type(record) is dict
                and set(record) == {"root", "kind", "expected_object_ids"},
                "root record fields changed",
            )
            require(
                type(record["root"]) is str and record["root"].startswith("/"),
                "root must be absolute",
            )
            require(
                record["kind"] in {"raw", "processed", "mixed"},
                "bad root kind",
            )
            ids = record["expected_object_ids"]
            require(
                type(ids) is list
                and ids == sorted(set(ids))
                and all(
                    type(value) is str and OBJECT_RE.fullmatch(value) for value in ids
                ),
                "expected object IDs must be sorted, unique, and canonical",
            )
            object_ids.extend(ids)
    require(
        len(object_ids) == len(set(object_ids)),
        "object occurs in multiple roots",
    )
    return protocol


def walk(root: Path, max_depth: int) -> Iterable[tuple[Path, list[str], list[str]]]:
    resolved = root.resolve(strict=True)
    base_depth = len(resolved.parts)
    for directory, names, files in os.walk(resolved, followlinks=False):
        current = Path(directory)
        names[:] = sorted(name for name in names if name not in {".git", "__pycache__"})
        if len(current.parts) - base_depth >= max_depth:
            names[:] = []
        yield current, names, sorted(files)


def object_directories(root: Path, object_id: str) -> tuple[Path, ...]:
    candidates: set[Path] = set()
    for candidate in (
        root / object_id,
        root / "raw" / object_id,
        root / "aligned" / object_id,
        root / "processed" / object_id,
    ):
        if candidate.is_dir() and not candidate.is_symlink():
            candidates.add(candidate.resolve(strict=True))
    if candidates:
        return tuple(sorted(candidates))
    for directory, names, _ in walk(root, 5):
        for name in names:
            if name == object_id or name.startswith(
                (f"{object_id}-ep", f"{object_id}_")
            ):
                candidate = directory / name
                if candidate.is_dir() and not candidate.is_symlink():
                    candidates.add(candidate.resolve(strict=True))
    return tuple(sorted(candidates))


def paired_stems(directory: Path, suffix: str, exclude_median: bool) -> set[str]:
    payload: set[str] = set()
    timestamps: set[str] = set()
    for path in directory.iterdir():
        if not path.is_file() or path.is_symlink():
            continue
        if path.suffix.lower() == suffix:
            if not (exclude_median and path.stem.lower().startswith("median_")):
                payload.add(path.stem)
        elif path.suffix.lower() == ".txt":
            timestamps.add(path.stem)
    return payload & timestamps


def raw_episodes(object_dir: Path) -> list[dict[str, Any]]:
    camera_streams = {
        path.name: paired_stems(path, ".mp4", False)
        for path in object_dir.iterdir()
        if path.is_dir() and CAMERA_RE.fullmatch(path.name)
    }
    tactile_streams = {
        path.name: paired_stems(path, ".npy", True)
        for path in object_dir.iterdir()
        if path.is_dir() and TACTILE_RE.fullmatch(path.name)
    }
    stems = sorted(set().union(*camera_streams.values(), *tactile_streams.values()))
    return [
        {
            "episode_key": stem,
            "camera_pairs": sum(stem in values for values in camera_streams.values()),
            "tactile_pairs": sum(stem in values for values in tactile_streams.values()),
            "object_dir": str(object_dir),
        }
        for stem in stems
    ]


def episode_directories(object_dir: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            directory
            for directory, _, _ in walk(object_dir, 4)
            if EPISODE_RE.fullmatch(directory.name)
        )
    )


def processed_episodes(object_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode_dir in episode_directories(object_dir):
        videos = 0
        robot = False
        target = False
        for directory, names, files in walk(episode_dir, 6):
            relatives = [
                (directory / name).relative_to(episode_dir).as_posix()
                for name in (*names, *files)
            ]
            videos += sum(
                Path(name).suffix.lower() in {".avi", ".mov", ".mp4"} for name in files
            )
            robot = robot or any(
                any(token in relative.lower() for token in ROBOT_HINTS)
                for relative in relatives
            )
            target = target or any(
                any(token in relative.lower() for token in TARGET_HINTS)
                for relative in relatives
            )
        match = EPISODE_RE.fullmatch(episode_dir.name)
        rows.append(
            {
                "episode_index": int(match.group(1)) if match else None,
                "camera_videos": videos,
                "has_robot_carrier": robot,
                "has_target_carrier": target,
                "episode_dir": str(episode_dir),
            }
        )
    return rows


def action_values(value: object, fragments: tuple[str, ...]) -> set[str]:
    found: set[str] = set()
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is str and any(part in key.lower() for part in fragments):
                if type(child) in {str, int, float} and type(child) is not bool:
                    found.add(str(child))
                elif type(child) is list:
                    found.update(
                        str(item)
                        for item in child
                        if type(item) in {str, int, float} and type(item) is not bool
                    )
            found.update(action_values(child, fragments))
    elif type(value) is list:
        for child in value:
            found.update(action_values(child, fragments))
    return found


def metadata_actions(
    object_dir: Path,
    metadata: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], set[str]]:
    allowed = set(metadata["allowed_basenames"])
    fragments = tuple(metadata["action_key_fragments"])
    maximum = int(metadata["maximum_json_bytes"])
    records: list[dict[str, Any]] = []
    actions: set[str] = set()
    for directory, _, files in walk(object_dir, 5):
        for name in files:
            path = directory / name
            if (
                name.lower() not in allowed
                or path.is_symlink()
                or not path.is_file()
                or path.stat().st_size > maximum
            ):
                continue
            try:
                raw = path.read_bytes()
                value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs_hook)
            except (OSError, UnicodeError, json.JSONDecodeError, AuditError):
                records.append({"path": str(path), "parsed": False})
                continue
            values = action_values(value, fragments)
            actions.update(values)
            records.append(
                {
                    "path": str(path),
                    "parsed": True,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "action_values": sorted(values),
                }
            )
    return records, actions


def best_rows(rows: Iterable[Mapping[str, Any]], identity: str) -> list[dict[str, Any]]:
    best: dict[object, dict[str, Any]] = {}
    for source in rows:
        row = dict(source)
        key = row.get(identity)
        score = sum(
            int(row.get(field, 0))
            for field in ("camera_pairs", "tactile_pairs", "camera_videos")
        )
        score += 100 * int(bool(row.get("has_target_carrier")))
        current = best.get(key)
        current_score = -1
        if current is not None:
            current_score = sum(
                int(current.get(field, 0))
                for field in ("camera_pairs", "tactile_pairs", "camera_videos")
            )
            current_score += 100 * int(bool(current.get("has_target_carrier")))
        if score > current_score:
            best[key] = row
    return [best[key] for key in sorted(best, key=str)]


def inspect_object(
    object_id: str,
    roots: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    processed_rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    actions: set[str],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    raw = best_rows(raw_rows, "episode_key")
    processed = best_rows(processed_rows, "episode_index")
    minimum_camera = int(thresholds["minimum_raw_camera_pairs_per_episode"])
    minimum_tactile = int(thresholds["minimum_raw_tactile_pairs_per_episode"])
    minimum_video = int(thresholds["minimum_processed_camera_videos_per_episode"])
    required = int(thresholds["minimum_episodes_per_transport_object"])
    raw_usable = [row for row in raw if row["camera_pairs"] >= minimum_camera]
    raw_visuotactile = [
        row for row in raw_usable if row["tactile_pairs"] >= minimum_tactile
    ]
    processed_rgb = [row for row in processed if row["camera_videos"] >= minimum_video]
    processed_ready = [
        row
        for row in processed_rgb
        if row["has_robot_carrier"] and row["has_target_carrier"]
    ]
    action_list = sorted(actions)
    action_diversity = len(action_list) >= 2
    if len(processed_ready) >= required and action_diversity:
        classification = "processed_transport_ready"
    elif len(raw_visuotactile) >= required:
        classification = "raw_visuotactile_transport_candidate"
    elif len(raw_usable) >= required:
        classification = "raw_visual_transport_candidate"
    elif len(processed_rgb) >= required:
        classification = "processed_rgb_transport_candidate"
    elif max(len(raw_usable), len(processed_rgb), len(processed)) >= 1:
        classification = "single_episode_calibration_or_control"
    else:
        classification = "incomplete"
    core = classification in {
        "processed_transport_ready",
        "raw_visuotactile_transport_candidate",
        "raw_visual_transport_candidate",
        "processed_rgb_transport_candidate",
    }
    return {
        "object_id": object_id,
        "classification": classification,
        "core_multi_episode_candidate": core,
        "raw_episode_count": len(raw),
        "raw_usable_episode_count": len(raw_usable),
        "raw_visuotactile_episode_count": len(raw_visuotactile),
        "processed_episode_count": len(processed),
        "processed_rgb_episode_count": len(processed_rgb),
        "processed_transport_ready_episode_count": len(processed_ready),
        "action_diversity_verified": action_diversity,
        "action_values": action_list,
        "roots": roots,
        "metadata_records": sorted(records, key=lambda row: row["path"]),
        "raw_episodes": raw,
        "processed_episodes": processed,
    }


def audit(
    protocol: Mapping[str, Any],
    runner_id: str,
    revision: str | None,
) -> dict[str, Any]:
    require(runner_id in protocol["runners"], f"unregistered runner: {runner_id}")
    thresholds = protocol["thresholds"]
    metadata = protocol["metadata"]
    objects: list[dict[str, Any]] = []
    missing_roots: list[str] = []
    missing_objects: list[dict[str, str]] = []

    for root_record in protocol["runners"][runner_id]["roots"]:
        root = Path(root_record["root"])
        if not root.is_dir():
            missing_roots.append(str(root))
            continue
        root = root.resolve(strict=True)
        for object_id in root_record["expected_object_ids"]:
            directories = object_directories(root, object_id)
            if not directories and len(root_record["expected_object_ids"]) == 1:
                directories = (root,)
            if not directories:
                missing_objects.append({"root": str(root), "object_id": object_id})
                continue
            roots: list[dict[str, Any]] = []
            raw_rows: list[dict[str, Any]] = []
            processed_rows: list[dict[str, Any]] = []
            records: list[dict[str, Any]] = []
            actions: set[str] = set()
            for directory in directories:
                roots.append(
                    {
                        "root": str(root),
                        "declared_kind": root_record["kind"],
                        "object_directory": str(directory),
                    }
                )
                item_records, item_actions = metadata_actions(directory, metadata)
                records.extend(item_records)
                actions.update(item_actions)
                child_names = {
                    path.name for path in directory.iterdir() if path.is_dir()
                }
                if root_record["kind"] in {"raw", "mixed"} and any(
                    CAMERA_RE.fullmatch(name) or TACTILE_RE.fullmatch(name)
                    for name in child_names
                ):
                    raw_rows.extend(raw_episodes(directory))
                if root_record["kind"] in {"processed", "mixed"}:
                    processed_rows.extend(processed_episodes(directory))
            objects.append(
                inspect_object(
                    object_id,
                    roots,
                    raw_rows,
                    processed_rows,
                    records,
                    actions,
                    thresholds,
                )
            )

    objects.sort(key=lambda row: row["object_id"])
    counts = Counter(row["classification"] for row in objects)
    core = [row for row in objects if row["core_multi_episode_candidate"]]
    ready = [
        row for row in objects if row["classification"] == "processed_transport_ready"
    ]
    bounded = int(thresholds["minimum_multi_episode_objects_for_bounded_study"])
    pilot = int(thresholds["minimum_multi_episode_objects_for_pilot"])
    if len(ready) >= bounded:
        decision = "sufficient_for_bounded_existing_data_study"
    elif len(core) >= bounded:
        decision = "conditionally_sufficient_pending_processing_and_action_audit"
    elif len(core) >= pilot:
        decision = "sufficient_for_pilot_only"
    else:
        decision = "insufficient_for_cross_episode_transport"

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "repository_revision": revision,
        "runner_id": runner_id,
        "official_processing_revision": protocol["official_processing_revision"],
        "decision": decision,
        "thresholds": thresholds,
        "information_boundary": dict(protocol["information_boundary"]),
        "summary": {
            "recognized_object_count": len(objects),
            "core_multi_episode_candidate_count": len(core),
            "processed_transport_ready_object_count": len(ready),
            "action_diversity_verified_candidate_count": sum(
                row["action_diversity_verified"] for row in core
            ),
            "classification_counts": dict(sorted(counts.items())),
            "missing_root_count": len(missing_roots),
            "missing_expected_object_count": len(missing_objects),
        },
        "missing_roots": sorted(missing_roots),
        "missing_expected_objects": sorted(
            missing_objects,
            key=lambda row: (row["root"], row["object_id"]),
        ),
        "objects": objects,
    }
    result["result_id"] = content_id(result)
    return result


def write_result(path: Path, value: Mapping[str, Any]) -> None:
    path.resolve().parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument(
        "--runner-id",
        choices=("gpuserver4090", "gpuserver6000"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    try:
        protocol = load_protocol(arguments.protocol)
        result = audit(
            protocol,
            arguments.runner_id,
            os.environ.get("GITHUB_SHA"),
        )
        write_result(arguments.output, result)
    except (AuditError, FileNotFoundError, OSError) as error:
        print(f"Deform360 eligibility audit failed: {error}")
        return 2
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "result_id": result["result_id"],
                **result["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
