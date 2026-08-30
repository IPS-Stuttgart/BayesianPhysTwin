#!/usr/bin/env python3
"""Prepare a deterministic Deform360 rope/cable pilot without opening payloads."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from itertools import permutations
from pathlib import Path
from typing import Any, Final

SCHEMA: Final = "bayesian-phystwin/deform360-same-object-rope-pilot-result"
PROTOCOL_SCHEMA: Final = "bayesian-phystwin/deform360-same-object-rope-pilot"
CAMERA_RE: Final = re.compile(r"^brics-odroid-\d+_cam\d+$")
TACTILE_RE: Final = re.compile(r"^brics-odroid_tactile[^/]+$")
EPISODE_RE: Final = re.compile(
    r"(?:^|[_-])(?:episode|ep)[_-]?0*(\d+)(?:[_-]|$)", re.IGNORECASE
)
TRAILING_INDEX_RE: Final = re.compile(r"(?:^|[_-])0*(\d{1,4})$")
METADATA_BASENAMES: Final = {
    "action.json",
    "actions.json",
    "episode.json",
    "episodes.json",
    "info.json",
    "manifest.json",
    "metadata.json",
}
ACTION_KEY_FRAGMENTS: Final = ("action", "primitive", "task", "manipulation")
EPISODE_KEY_FRAGMENTS: Final = ("episode", "episode_id", "episode_idx")


class PilotError(ValueError):
    """Raised when the registered target-blind pilot contract is invalid."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PilotError(message)


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def content_id(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("result_id", None)
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
            path.read_text(encoding="utf-8"), object_pairs_hook=pairs_hook
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
    require(protocol.get("schema_version") == 1, "unsupported protocol version")
    require(
        protocol.get("status") == "frozen-before-payload-decoding",
        "protocol is not frozen before payload decoding",
    )
    object_ids = protocol.get("object_ids")
    require(
        type(object_ids) is list
        and len(object_ids) == 4
        and object_ids == sorted(set(object_ids))
        and all(type(item) is str for item in object_ids),
        "object_ids must contain four sorted unique strings",
    )
    boundary = protocol.get("information_boundary")
    require(type(boundary) is dict, "information boundary is missing")
    require(boundary.get("media_payload_decoded") is False, "media boundary opened")
    require(boundary.get("numeric_arrays_loaded") is False, "array boundary opened")
    require(boundary.get("target_future_opened") is False, "target boundary opened")
    require(
        boundary.get("score_bearing_outcomes_used") is False,
        "score-bearing outcome boundary opened",
    )
    root = protocol.get("root")
    require(type(root) is str and root.startswith("/"), "root must be absolute")
    eligibility = protocol.get("eligibility")
    require(type(eligibility) is dict, "eligibility contract is missing")
    for key in (
        "minimum_camera_pairs_per_episode",
        "preferred_camera_pairs_per_episode",
        "minimum_tactile_pairs_per_episode",
        "minimum_eligible_episodes_per_object",
        "minimum_objects_for_pilot",
    ):
        require(type(eligibility.get(key)) is int, f"bad eligibility field: {key}")
    return protocol


def safe_iterdir(path: Path) -> tuple[tuple[Path, ...], str | None]:
    try:
        return tuple(sorted(path.iterdir(), key=lambda item: item.name)), None
    except OSError as error:
        return (), f"{type(error).__name__}: {error}"


def paired_stems(directory: Path, suffix: str, *, exclude_median: bool = False) -> set[str]:
    entries, error = safe_iterdir(directory)
    if error is not None:
        return set()
    payloads: set[str] = set()
    timestamps: set[str] = set()
    for path in entries:
        try:
            regular = path.is_file() and not path.is_symlink()
        except OSError:
            regular = False
        if not regular:
            continue
        lowered_suffix = path.suffix.lower()
        if lowered_suffix == suffix:
            if not (exclude_median and path.stem.lower().startswith("median_")):
                payloads.add(path.stem)
        elif lowered_suffix == ".txt":
            timestamps.add(path.stem)
    return payloads & timestamps


def episode_index(stem: str) -> int | None:
    match = EPISODE_RE.search(stem)
    if match is not None:
        return int(match.group(1))
    match = TRAILING_INDEX_RE.search(stem)
    if match is not None:
        return int(match.group(1))
    if stem.isdigit():
        return int(stem)
    return None


def scalar_strings(value: object) -> set[str]:
    if type(value) is str:
        stripped = value.strip()
        return {stripped} if stripped else set()
    if type(value) in {int, float} and type(value) is not bool:
        return {str(value)}
    if type(value) is list:
        return set().union(*(scalar_strings(child) for child in value)) if value else set()
    return set()


def metadata_actions(value: object) -> tuple[dict[int, set[str]], set[str]]:
    indexed: dict[int, set[str]] = defaultdict(set)
    global_actions: set[str] = set()

    def visit(child: object, inherited_index: int | None = None) -> None:
        if type(child) is list:
            for position, item in enumerate(child):
                visit(item, inherited_index if inherited_index is not None else position)
            return
        if type(child) is not dict:
            return

        local_index = inherited_index
        for key, item in child.items():
            if type(key) is not str:
                continue
            lowered = key.lower()
            if any(fragment == lowered for fragment in EPISODE_KEY_FRAGMENTS):
                candidates = scalar_strings(item)
                for candidate in candidates:
                    match = re.search(r"\d+", candidate)
                    if match is not None:
                        local_index = int(match.group(0))
                        break

        for key, item in child.items():
            if type(key) is not str:
                continue
            lowered = key.lower()
            if any(fragment in lowered for fragment in ACTION_KEY_FRAGMENTS):
                values = scalar_strings(item)
                values = {value for value in values if len(value) <= 160}
                if local_index is None:
                    global_actions.update(values)
                else:
                    indexed[local_index].update(values)
            visit(item, local_index)

    visit(value)
    return dict(indexed), global_actions


def scan_metadata(
    object_dir: Path, maximum_bytes: int
) -> tuple[list[dict[str, Any]], dict[int, set[str]], set[str], list[str]]:
    records: list[dict[str, Any]] = []
    indexed: dict[int, set[str]] = defaultdict(set)
    global_actions: set[str] = set()
    errors: list[str] = []
    base_depth = len(object_dir.parts)

    def onerror(error: OSError) -> None:
        errors.append(f"{type(error).__name__}: {error}")

    try:
        walker: Iterable[tuple[str, list[str], list[str]]] = os.walk(
            object_dir, followlinks=False, onerror=onerror
        )
        for directory, names, files in walker:
            current = Path(directory)
            depth = len(current.parts) - base_depth
            names[:] = sorted(name for name in names if name not in {".git", "__pycache__"})
            if depth >= 5:
                names[:] = []
            for name in sorted(files):
                if name.lower() not in METADATA_BASENAMES:
                    continue
                path = current / name
                try:
                    if path.is_symlink() or not path.is_file():
                        continue
                    size = path.stat().st_size
                    if size > maximum_bytes:
                        records.append(
                            {"path": str(path), "parsed": False, "reason": "too-large"}
                        )
                        continue
                    raw = path.read_bytes()
                    value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs_hook)
                    per_index, global_values = metadata_actions(value)
                    for index, values in per_index.items():
                        indexed[index].update(values)
                    global_actions.update(global_values)
                    records.append(
                        {
                            "path": str(path),
                            "parsed": True,
                            "sha256": hashlib.sha256(raw).hexdigest(),
                            "indexed_action_count": sum(len(values) for values in per_index.values()),
                            "global_actions": sorted(global_values),
                        }
                    )
                except (OSError, UnicodeError, json.JSONDecodeError, PilotError) as error:
                    records.append(
                        {
                            "path": str(path),
                            "parsed": False,
                            "reason": type(error).__name__,
                        }
                    )
    except OSError as error:
        errors.append(f"{type(error).__name__}: {error}")
    return records, dict(indexed), global_actions, sorted(set(errors))


def scan_object(
    root: Path, object_id: str, eligibility: Mapping[str, Any], maximum_bytes: int
) -> dict[str, Any]:
    object_dir = root / object_id
    entries, access_error = safe_iterdir(object_dir)
    result: dict[str, Any] = {
        "object_id": object_id,
        "object_dir": str(object_dir),
        "present": object_dir.exists(),
        "access_error": access_error,
        "metadata": [],
        "metadata_errors": [],
        "global_actions": [],
        "episodes": [],
        "candidate_pairs": [],
        "selected_pair": None,
    }
    if access_error is not None or not object_dir.is_dir():
        return result

    camera_dirs = [
        path for path in entries if path.is_dir() and CAMERA_RE.fullmatch(path.name)
    ]
    tactile_dirs = [
        path for path in entries if path.is_dir() and TACTILE_RE.fullmatch(path.name)
    ]
    camera = {path.name: paired_stems(path, ".mp4") for path in camera_dirs}
    tactile = {
        path.name: paired_stems(path, ".npy", exclude_median=True)
        for path in tactile_dirs
    }
    stems = sorted(set().union(*camera.values(), *tactile.values()))
    records, indexed_actions, global_actions, metadata_errors = scan_metadata(
        object_dir, maximum_bytes
    )
    result["metadata"] = records
    result["metadata_errors"] = metadata_errors
    result["global_actions"] = sorted(global_actions)
    result["camera_directory_count"] = len(camera_dirs)
    result["tactile_directory_count"] = len(tactile_dirs)

    minimum_camera = int(eligibility["minimum_camera_pairs_per_episode"])
    minimum_tactile = int(eligibility["minimum_tactile_pairs_per_episode"])
    episodes: list[dict[str, Any]] = []
    for stem in stems:
        index = episode_index(stem)
        actions = set(indexed_actions.get(index, set())) if index is not None else set()
        if not actions and len(global_actions) == 1:
            actions = set(global_actions)
        camera_pairs = sum(stem in values for values in camera.values())
        tactile_pairs = sum(stem in values for values in tactile.values())
        episodes.append(
            {
                "object_id": object_id,
                "episode_key": stem,
                "episode_index": index,
                "camera_pairs": camera_pairs,
                "tactile_pairs": tactile_pairs,
                "action_labels": sorted(actions),
                "visual_eligible": camera_pairs >= minimum_camera,
                "visuotactile_eligible": camera_pairs >= minimum_camera
                and tactile_pairs >= minimum_tactile,
            }
        )
    result["episodes"] = episodes

    eligible = [episode for episode in episodes if episode["visual_eligible"]]
    candidates: list[dict[str, Any]] = []
    for source, target in permutations(eligible, 2):
        source_actions = set(source["action_labels"])
        target_actions = set(target["action_labels"])
        different_action = bool(
            source_actions and target_actions and source_actions.isdisjoint(target_actions)
        )
        candidate = {
            "object_id": object_id,
            "source_episode_key": source["episode_key"],
            "target_episode_key": target["episode_key"],
            "source_episode_index": source["episode_index"],
            "target_episode_index": target["episode_index"],
            "source_action_labels": sorted(source_actions),
            "target_action_labels": sorted(target_actions),
            "different_action_labels": different_action,
            "minimum_camera_pairs": min(source["camera_pairs"], target["camera_pairs"]),
            "minimum_tactile_pairs": min(source["tactile_pairs"], target["tactile_pairs"]),
            "both_visuotactile_eligible": bool(
                source["visuotactile_eligible"] and target["visuotactile_eligible"]
            ),
            "target_future_opened": False,
        }
        candidate["pair_id"] = content_id(candidate)
        candidates.append(candidate)
    candidates.sort(
        key=lambda item: (
            -int(item["different_action_labels"]),
            -int(item["both_visuotactile_eligible"]),
            -int(item["minimum_camera_pairs"]),
            -int(item["minimum_tactile_pairs"]),
            str(item["source_episode_key"]),
            str(item["target_episode_key"]),
        )
    )
    result["candidate_pairs"] = candidates
    result["selected_pair"] = candidates[0] if candidates else None
    return result


def flatten_episode_rows(objects: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for item in objects for row in item.get("episodes", [])]


def flatten_pair_rows(objects: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for item in objects for row in item.get("candidate_pairs", [])]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )


def decision(objects: Sequence[Mapping[str, Any]], eligibility: Mapping[str, Any]) -> str:
    minimum_episodes = int(eligibility["minimum_eligible_episodes_per_object"])
    minimum_objects = int(eligibility["minimum_objects_for_pilot"])
    accessible = [item for item in objects if item.get("access_error") is None]
    visual = [
        item
        for item in accessible
        if sum(bool(row["visual_eligible"]) for row in item["episodes"])
        >= minimum_episodes
    ]
    visuotactile = [
        item
        for item in accessible
        if sum(bool(row["visuotactile_eligible"]) for row in item["episodes"])
        >= minimum_episodes
    ]
    cross_action = [
        item
        for item in visuotactile
        if item.get("selected_pair") is not None
        and bool(item["selected_pair"]["different_action_labels"])
    ]
    if len(accessible) < minimum_objects:
        return "blocked-by-object-access"
    if len(visual) < minimum_objects:
        return "insufficient-four-object-visual-roster"
    if len(visuotactile) < minimum_objects:
        return "four-object-visual-roster-ready-tactile-incomplete"
    if len(cross_action) < minimum_objects:
        return "four-object-visuotactile-roster-ready-action-labels-unresolved"
    return "four-object-cross-action-source-seal-ready"


def render_report(result: Mapping[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# Deform360 same-object rope/cable pilot v1",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Registered objects: **{summary['registered_object_count']}**",
        f"- Accessible objects: **{summary['accessible_object_count']}**",
        f"- Visual-pairable objects: **{summary['visual_pairable_object_count']}**",
        f"- Visuotactile-pairable objects: **{summary['visuotactile_pairable_object_count']}**",
        f"- Selected different-action pairs: **{summary['different_action_selected_pair_count']}**",
        "- Media decoded: **no**",
        "- Numeric arrays loaded: **no**",
        "- Target futures opened: **no**",
        "",
        "| Object | Episodes | Visual eligible | Visuotactile eligible | Selected source | Selected target | Different action |",
        "|---|---:|---:|---:|---|---|---:|",
    ]
    for item in result["objects"]:
        selected = item.get("selected_pair") or {}
        lines.append(
            "| `{}` | {} | {} | {} | `{}` | `{}` | {} |".format(
                item["object_id"],
                len(item.get("episodes", [])),
                sum(bool(row["visual_eligible"]) for row in item.get("episodes", [])),
                sum(
                    bool(row["visuotactile_eligible"])
                    for row in item.get("episodes", [])
                ),
                selected.get("source_episode_key", ""),
                selected.get("target_episode_key", ""),
                "yes" if selected.get("different_action_labels") else "no",
            )
        )
    lines.extend(
        [
            "",
            "The selected target episode identifies only a future scoring unit. This run did not open that future. The next stage must freeze the geometry/action adapter, physical hypothesis bank, source-only fit, comparators, and prediction seal before target scoring.",
            "",
            f"Result ID: `{result['result_id']}`",
            "",
        ]
    )
    return "\n".join(lines)


def run(protocol: Mapping[str, Any], revision: str | None = None) -> dict[str, Any]:
    root = Path(protocol["root"])
    maximum_bytes = int(protocol["information_boundary"]["maximum_metadata_json_bytes"])
    root_error: str | None = None
    try:
        root_present = root.is_dir()
        if root_present:
            _, root_error = safe_iterdir(root)
    except OSError as error:
        root_present = False
        root_error = f"{type(error).__name__}: {error}"
    objects = [
        scan_object(root, object_id, protocol["eligibility"], maximum_bytes)
        for object_id in protocol["object_ids"]
    ] if root_present and root_error is None else [
        {
            "object_id": object_id,
            "object_dir": str(root / object_id),
            "present": False,
            "access_error": root_error or "registered root is unavailable",
            "metadata": [],
            "metadata_errors": [],
            "global_actions": [],
            "episodes": [],
            "candidate_pairs": [],
            "selected_pair": None,
        }
        for object_id in protocol["object_ids"]
    ]

    minimum_episodes = int(protocol["eligibility"]["minimum_eligible_episodes_per_object"])
    accessible = [item for item in objects if item.get("access_error") is None]
    visual = [
        item
        for item in accessible
        if sum(bool(row["visual_eligible"]) for row in item["episodes"])
        >= minimum_episodes
    ]
    visuotactile = [
        item
        for item in accessible
        if sum(bool(row["visuotactile_eligible"]) for row in item["episodes"])
        >= minimum_episodes
    ]
    selected = [item["selected_pair"] for item in objects if item.get("selected_pair")]
    summary = {
        "decision": decision(objects, protocol["eligibility"]),
        "registered_object_count": len(objects),
        "accessible_object_count": len(accessible),
        "visual_pairable_object_count": len(visual),
        "visuotactile_pairable_object_count": len(visuotactile),
        "selected_pair_count": len(selected),
        "different_action_selected_pair_count": sum(
            bool(pair["different_action_labels"]) for pair in selected
        ),
        "episode_count": sum(len(item.get("episodes", [])) for item in objects),
        "candidate_pair_count": sum(
            len(item.get("candidate_pairs", [])) for item in objects
        ),
        "root_present": root_present,
        "root_access_error": root_error,
    }
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "repository_revision": revision,
        "runner_id": protocol["runner_id"],
        "root": str(root),
        "information_boundary": protocol["information_boundary"],
        "objects": objects,
        "selected_pairs": selected,
        "summary": summary,
        "claim_boundary": protocol["claim_boundary"],
    }
    result["result_id"] = content_id(result)
    return result


def publish(result: Mapping[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "selected_pairs.json").write_text(
        json.dumps(result["selected_pairs"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(output / "episodes.csv", flatten_episode_rows(result["objects"]))
    write_csv(output / "candidate_pairs.csv", flatten_pair_rows(result["objects"]))
    (output / "report.md").write_text(render_report(result), encoding="utf-8")


def build_fixture(root: Path, object_ids: Sequence[str]) -> dict[str, Any]:
    data_root = root / "raw"
    for object_id in object_ids:
        object_dir = data_root / object_id
        for camera in range(32):
            stream = object_dir / f"brics-odroid-{camera // 4}_cam{camera % 4}"
            stream.mkdir(parents=True, exist_ok=True)
            for episode in range(2):
                stem = f"capture_episode_{episode:04d}"
                (stream / f"{stem}.mp4").write_bytes(b"")
                (stream / f"{stem}.txt").write_text("0\n", encoding="utf-8")
        for tactile_index in range(2):
            stream = object_dir / f"brics-odroid_tactile{tactile_index}"
            stream.mkdir(parents=True, exist_ok=True)
            for episode in range(2):
                stem = f"capture_episode_{episode:04d}"
                (stream / f"{stem}.npy").write_bytes(b"")
                (stream / f"{stem}.txt").write_text("0\n", encoding="utf-8")
        (object_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "episodes": [
                        {"episode_id": 0, "action": "pull"},
                        {"episode_id": 1, "action": "twist"},
                    ]
                }
            ),
            encoding="utf-8",
        )
    return {
        "schema": PROTOCOL_SCHEMA,
        "schema_version": 1,
        "protocol_id": "fixture",
        "status": "frozen-before-payload-decoding",
        "runner_id": "fixture",
        "root": str(data_root),
        "object_ids": sorted(object_ids),
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
            "minimum_camera_pairs_per_episode": 32,
            "preferred_camera_pairs_per_episode": 37,
            "minimum_tactile_pairs_per_episode": 2,
            "minimum_eligible_episodes_per_object": 2,
            "minimum_objects_for_pilot": 4,
            "require_distinct_source_and_target_episode": True,
            "prefer_different_action_labels": True,
        },
        "selection": {},
        "next_stage_contract": {},
        "claim_boundary": "fixture",
    }


def self_test() -> None:
    object_ids = ["001-rope", "002-rope-silk", "003-cable", "081-stripe-rope"]
    with tempfile.TemporaryDirectory() as temporary:
        protocol = build_fixture(Path(temporary), object_ids)
        result = run(protocol, "fixture")
        require(
            result["summary"]["decision"]
            == "four-object-cross-action-source-seal-ready",
            "positive fixture did not reach the registered source-seal gate",
        )
        require(len(result["selected_pairs"]) == 4, "positive fixture lost objects")
        require(
            all(pair["different_action_labels"] for pair in result["selected_pairs"]),
            "positive fixture lost action separation",
        )
        broken = dict(protocol)
        broken["root"] = str(Path(temporary) / "missing")
        negative = run(broken, "fixture")
        require(
            negative["summary"]["decision"] == "blocked-by-object-access",
            "missing-root fixture did not fail closed",
        )
    print("self-test passed")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--protocol", default="protocols/deform360_same_object_rope_pilot_v1.json"
    )
    value.add_argument("--output")
    value.add_argument("--revision")
    value.add_argument("--self-test", action="store_true")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    if arguments.self_test:
        self_test()
        return 0
    require(arguments.output is not None, "--output is required")
    protocol = load_protocol(Path(arguments.protocol))
    result = run(protocol, arguments.revision)
    publish(result, Path(arguments.output))
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PilotError as error:
        print(f"Deform360 rope pilot preparation failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
