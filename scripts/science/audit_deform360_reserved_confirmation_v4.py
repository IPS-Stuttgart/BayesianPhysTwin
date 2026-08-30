#!/usr/bin/env python3
"""Audit reserved Deform360 objects without opening numeric payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

TACTILE_RE = re.compile(r"tactile", re.IGNORECASE)
SCHEMA = "bayesian-phystwin/deform360-reserved-confirmation-readiness-v4"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def episode_records(metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = metadata.get("sequences", metadata.get("episodes", metadata.get("takes")))
    if isinstance(raw, Mapping):
        items = sorted(
            raw.items(),
            key=lambda item: (
                0 if str(item[0]).isdigit() else 1,
                int(item[0]) if str(item[0]).isdigit() else str(item[0]),
            ),
        )
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        items = list(enumerate(raw))
    else:
        return []
    rows = []
    for raw_id, record in items:
        if not isinstance(record, Mapping):
            continue
        action = record.get("action")
        rows.append(
            {
                "episode_id": int(raw_id) if str(raw_id).isdigit() else len(rows),
                "action": action.strip()
                if isinstance(action, str) and action.strip()
                else None,
                "bimanual": record.get("bimanual"),
            }
        )
    return rows


def episode_directory(parent: Path, episode_id: int) -> Path | None:
    for name in (
        f"episode_{episode_id}",
        f"episode_{episode_id:04d}",
        f"episode-{episode_id}",
    ):
        candidate = parent / name
        if candidate.is_dir():
            return candidate
    return None


def file_identity(path: Path) -> dict[str, Any]:
    return {"name": path.name, "size_bytes": int(path.stat().st_size)}


def inspect_object(root: Path, object_id: str, minimum_episodes: int) -> dict[str, Any]:
    raw = root / "raw-repository" / "raw" / object_id
    processed = root / "processed-repository" / "processed" / object_id
    metadata_path = raw / "metadata.json"
    if not metadata_path.is_file():
        return {"object_id": object_id, "eligible": False, "reason": "metadata-missing"}
    metadata_bytes = metadata_path.read_bytes()
    metadata = json.loads(metadata_bytes)
    if not isinstance(metadata, dict):
        return {
            "object_id": object_id,
            "eligible": False,
            "reason": "metadata-not-object",
        }
    episodes = episode_records(metadata)
    tactile_groups: list[dict[str, Any]] = []
    if raw.is_dir():
        for directory in sorted(
            (path for path in raw.iterdir() if path.is_dir()), key=lambda p: p.name
        ):
            if not TACTILE_RE.search(directory.name):
                continue
            files = sorted(
                (
                    path
                    for path in directory.glob("*.npy")
                    if not path.name.lower().startswith("median_")
                    and path.stat().st_size > 0
                ),
                key=lambda path: path.name,
            )
            if files:
                tactile_groups.append(
                    {
                        "directory": directory.name,
                        "recording_count": len(files),
                        "recordings": [file_identity(path) for path in files],
                        "matches_metadata_episode_count": len(files) == len(episodes),
                    }
                )
    complete_episode_ids: list[int] = []
    robot_files: list[dict[str, Any]] = []
    for episode in episodes:
        episode_id = int(episode["episode_id"])
        directory = episode_directory(processed, episode_id)
        if directory is None:
            continue
        robot = next(
            (
                path
                for path in (
                    directory / "robot" / "robot.npz",
                    directory / "robot" / "robot.npy",
                )
                if path.is_file() and path.stat().st_size > 0
            ),
            None,
        )
        tactile_complete = bool(tactile_groups) and all(
            group["matches_metadata_episode_count"] for group in tactile_groups
        )
        if robot is not None:
            robot_files.append(
                {
                    "episode_id": episode_id,
                    "relative_path": robot.relative_to(root).as_posix(),
                    **file_identity(robot),
                }
            )
        if robot is not None and tactile_complete:
            complete_episode_ids.append(episode_id)
    action_count = sum(row["action"] is not None for row in episodes)
    eligible = bool(
        len(complete_episode_ids) >= minimum_episodes
        and action_count == len(episodes)
        and len(tactile_groups) >= 2
    )
    return {
        "object_id": object_id,
        "eligible": eligible,
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "episode_count": len(episodes),
        "action_count": action_count,
        "episodes": episodes,
        "tactile_group_count": len(tactile_groups),
        "tactile_groups": tactile_groups,
        "robot_file_count": len(robot_files),
        "robot_files": robot_files,
        "complete_episode_ids": complete_episode_ids,
    }


def run(root: Path, protocol_path: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    protocol = read_json(protocol_path)
    if root != Path(str(protocol["dataset_root"])):
        raise ValueError("dataset root changed")
    reserved = list(map(str, protocol["reserved_object_ids"]))
    development = set(map(str, protocol["development_object_ids"]))
    if development & set(reserved):
        raise ValueError("development and reserved rosters overlap")
    minimum = int(protocol["selection"]["minimum_episodes"])
    rows = [inspect_object(root, object_id, minimum) for object_id in reserved]
    eligible = [row["object_id"] for row in rows if row["eligible"]]
    result = {
        "schema": SCHEMA,
        "schema_version": 4,
        "dataset_root": str(root),
        "protocol_path": str(protocol_path),
        "summary": {
            "reserved_object_count": len(reserved),
            "metadata_present_count": sum("episode_count" in row for row in rows),
            "eligible_object_count": len(eligible),
            "eligible_object_ids": eligible,
        },
        "objects": rows,
        "information_boundary": {
            "reserved_metadata_json_opened": True,
            "reserved_file_names_and_sizes_opened": True,
            "reserved_robot_numeric_payloads_opened": False,
            "reserved_tactile_numeric_payloads_opened": False,
            "reserved_target_scores_computed": False,
            "camera_pixels_opened": False,
            "geometry_or_point_cloud_opened": False,
        },
        "confirmation_authorized": False,
        "paper_claim_authorized": False,
    }
    canonical = json.dumps(
        result, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    result["result_sha256"] = hashlib.sha256(canonical).hexdigest()
    return result


def report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# Deform360 reserved confirmation readiness v4",
        "",
        f"- Reserved objects: **{summary['reserved_object_count']}**",
        f"- Metadata present: **{summary['metadata_present_count']}**",
        f"- Payload-ready under the frozen carrier contract: **{summary['eligible_object_count']}**",
        f"- Eligible roster: `{', '.join(summary['eligible_object_ids'])}`",
        "",
        "| Object | Episodes/actions | Robot carriers | Tactile groups | Eligible |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in result["objects"]:
        lines.append(
            f"| `{row['object_id']}` | {row.get('episode_count', 0)}/{row.get('action_count', 0)} | "
            f"{row.get('robot_file_count', 0)} | {row.get('tactile_group_count', 0)} | "
            f"{str(bool(row.get('eligible'))).lower()} |"
        )
    lines += [
        "",
        "Only released metadata plus file names and sizes were opened. Reserved numeric",
        "robot/tactile payloads, camera pixels, geometry, and scores remain unopened.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.data_root, args.protocol)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.output_report.write_text(report(result))
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
