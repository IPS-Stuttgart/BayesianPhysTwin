#!/usr/bin/env python3
"""Discover Deform360 objects not numerically opened by the v3/v4 study.

The audit reads only released metadata JSON plus directory entries and file sizes.
It never loads robot, tactile, camera, geometry, or point-cloud numeric payloads.
Every carrier-complete object outside the explicitly bound opened roster is
selected deterministically; no outcome-dependent filtering is permitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

TACTILE_RE = re.compile(r"tactile", re.IGNORECASE)
SCHEMA = "bayesian-phystwin/deform360-untouched-readiness-v5"
PROTOCOL_SCHEMA = "bayesian-phystwin/deform360-untouched-readiness-protocol-v5"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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

    rows: list[dict[str, Any]] = []
    for raw_id, record in items:
        if not isinstance(record, Mapping):
            continue
        action = record.get("action")
        rows.append(
            {
                "episode_id": int(raw_id) if str(raw_id).isdigit() else len(rows),
                "action": (
                    action.strip()
                    if isinstance(action, str) and action.strip()
                    else None
                ),
                "bimanual": record.get("bimanual"),
                "nonprehensile": record.get("nonprehensile"),
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


def file_identity(path: Path, root: Path) -> dict[str, Any]:
    return {
        "name": path.name,
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": int(path.stat().st_size),
    }


def inspect_object(root: Path, object_id: str, minimum_episodes: int) -> dict[str, Any]:
    raw = root / "raw-repository" / "raw" / object_id
    processed = root / "processed-repository" / "processed" / object_id
    metadata_path = raw / "metadata.json"
    if not metadata_path.is_file():
        return {
            "object_id": object_id,
            "eligible": False,
            "reason": "metadata-missing",
        }

    metadata_bytes = metadata_path.read_bytes()
    metadata = json.loads(metadata_bytes)
    if not isinstance(metadata, dict):
        return {
            "object_id": object_id,
            "eligible": False,
            "reason": "metadata-not-object",
        }
    episodes = episode_records(metadata)
    episode_ids = [int(row["episode_id"]) for row in episodes]
    contiguous_ids = episode_ids == list(range(len(episodes)))

    tactile_groups: list[dict[str, Any]] = []
    if raw.is_dir():
        directories = (path for path in raw.iterdir() if path.is_dir())
        for directory in sorted(directories, key=lambda path: path.name):
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
                        "directory": directory.relative_to(root).as_posix(),
                        "recording_count": len(files),
                        "recordings": [file_identity(path, root) for path in files],
                        "matches_metadata_episode_count": len(files) == len(episodes),
                    }
                )

    tactile_complete = bool(tactile_groups) and all(
        group["matches_metadata_episode_count"] for group in tactile_groups
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
        if robot is not None:
            robot_files.append(
                {
                    "episode_id": episode_id,
                    **file_identity(robot, root),
                }
            )
            if tactile_complete and contiguous_ids:
                complete_episode_ids.append(episode_id)

    action_count = sum(row["action"] is not None for row in episodes)
    eligible = bool(
        len(complete_episode_ids) >= minimum_episodes
        and action_count == len(episodes)
        and len(tactile_groups) >= 2
        and tactile_complete
        and contiguous_ids
    )
    reason = None
    if not eligible:
        if not episodes:
            reason = "episode-metadata-unavailable"
        elif not contiguous_ids:
            reason = "episode-ids-not-contiguous"
        elif action_count != len(episodes):
            reason = "action-metadata-incomplete"
        elif len(tactile_groups) < 2:
            reason = "fewer-than-two-tactile-groups"
        elif not tactile_complete:
            reason = "tactile-recording-count-mismatch"
        elif len(complete_episode_ids) < minimum_episodes:
            reason = "insufficient-robot-complete-episodes"
        else:
            reason = "carrier-contract-not-met"

    action_by_id = {int(row["episode_id"]): row["action"] for row in episodes}
    target_episode_id = max(complete_episode_ids) if eligible else None
    return {
        "object_id": object_id,
        "eligible": eligible,
        "reason": reason,
        "metadata_relative_path": metadata_path.relative_to(root).as_posix(),
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "episode_count": len(episodes),
        "action_count": action_count,
        "episode_ids_contiguous_from_zero": contiguous_ids,
        "episodes": episodes,
        "tactile_group_count": len(tactile_groups),
        "tactile_groups": tactile_groups,
        "robot_file_count": len(robot_files),
        "robot_files": robot_files,
        "complete_episode_ids": complete_episode_ids,
        "source_episode_count": max(len(complete_episode_ids) - 1, 0),
        "target_episode_id": target_episode_id,
        "target_action": action_by_id.get(target_episode_id),
    }


def validate_protocol(protocol: dict[str, Any], root: Path) -> None:
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unexpected untouched-readiness protocol schema")
    if protocol.get("status") != "frozen-before-untouched-numeric-payload-access":
        raise ValueError("untouched-readiness protocol is not frozen")
    if Path(str(protocol["dataset_root"])) != root:
        raise ValueError("dataset root changed")
    opened = list(map(str, protocol["numeric_payload_opened_object_ids"]))
    if len(opened) != len(set(opened)):
        raise ValueError("opened object roster contains duplicates")
    if protocol["selection"].get("include_every_eligible_object") is not True:
        raise ValueError("outcome-blind all-eligible selection is required")
    if protocol.get("paper_claim_authorized") is not False:
        raise ValueError("readiness protocol self-authorized a paper claim")


def run(root: Path, protocol_path: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    protocol = read_json(protocol_path)
    validate_protocol(protocol, root)

    raw_root = root / "raw-repository" / "raw"
    if not raw_root.is_dir():
        raise ValueError(f"raw repository unavailable: {raw_root}")
    opened = set(map(str, protocol["numeric_payload_opened_object_ids"]))
    all_object_ids = sorted(
        path.name
        for path in raw_root.iterdir()
        if path.is_dir() and (path / "metadata.json").is_file()
    )
    candidate_ids = [
        object_id for object_id in all_object_ids if object_id not in opened
    ]
    minimum = int(protocol["selection"]["minimum_complete_episodes_per_object"])
    rows = [inspect_object(root, object_id, minimum) for object_id in candidate_ids]
    eligible = [row["object_id"] for row in rows if row["eligible"]]
    selection_manifest = [
        {
            "object_id": row["object_id"],
            "metadata_sha256": row["metadata_sha256"],
            "complete_episode_ids": row["complete_episode_ids"],
            "target_episode_id": row["target_episode_id"],
            "target_action": row["target_action"],
            "robot_files": row["robot_files"],
            "tactile_groups": [
                {
                    "directory": group["directory"],
                    "recording_count": group["recording_count"],
                    "recordings": group["recordings"],
                }
                for group in row["tactile_groups"]
            ],
        }
        for row in rows
        if row["eligible"]
    ]
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 5,
        "status": "complete",
        "dataset_root": str(root),
        "protocol_path": str(protocol_path),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "summary": {
            "metadata_object_count": len(all_object_ids),
            "numeric_opened_exclusion_count": len(opened),
            "candidate_object_count": len(candidate_ids),
            "eligible_object_count": len(eligible),
            "eligible_object_ids": eligible,
        },
        "selection_manifest": selection_manifest,
        "selection_manifest_sha256": canonical_digest(selection_manifest),
        "objects": rows,
        "information_boundary": {
            "metadata_json_opened": True,
            "directory_names_and_file_sizes_opened": True,
            "robot_numeric_payloads_opened": False,
            "tactile_numeric_payloads_opened": False,
            "target_outcomes_scored": False,
            "camera_pixels_opened": False,
            "geometry_or_point_cloud_opened": False,
        },
        "confirmation_authorized": False,
        "paper_claim_authorized": False,
    }
    result["result_sha256"] = canonical_digest(result)
    return result


def report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# Deform360 untouched-object readiness v5",
        "",
        f"- Metadata-bearing objects: **{summary['metadata_object_count']}**",
        "- Previously numerically opened exclusions: "
        f"**{summary['numeric_opened_exclusion_count']}**",
        f"- Untouched candidates audited: **{summary['candidate_object_count']}**",
        f"- Carrier-complete untouched objects: **{summary['eligible_object_count']}**",
        "- Deterministic eligible roster: "
        f"`{', '.join(summary['eligible_object_ids'])}`",
        "",
        "| Object | Episodes/actions | Robot carriers | Tactile groups | "
        "Target | Eligible | Reason |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in result["objects"]:
        lines.append(
            f"| `{row['object_id']}` | "
            f"{row.get('episode_count', 0)}/{row.get('action_count', 0)} | "
            f"{row.get('robot_file_count', 0)} | {row.get('tactile_group_count', 0)} | "
            f"{row.get('target_episode_id', '')} | "
            f"{str(bool(row.get('eligible'))).lower()} | "
            f"{row.get('reason') or ''} |"
        )
    lines.extend(
        [
            "",
            "Selection includes every carrier-complete object outside the bound v3/v4",
            "numeric-opened roster. Only metadata, names, and file sizes were read;",
            "numeric robot/tactile payloads and all target outcomes remain unopened.",
            "",
        ]
    )
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
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.output_report.write_text(report(result), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
