#!/usr/bin/env python3
"""Audit Deform360 carriers for an action-conditioned real-data experiment.

The audit reads directory/file metadata and released ``metadata.json`` files only.
It never decodes camera, tactile, robot, geometry, or trajectory payloads.  Its
purpose is to determine which development objects can support a time-resolved,
action-aware follow-up on the gpuserver4090 mount.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "bayesian-phystwin/deform360-icra-carrier-audit-v2"
OBJECT_RE = re.compile(r"^\d{3}-.+")
TACTILE_RE = re.compile(r"tactile", re.IGNORECASE)
CAMERA_RE = re.compile(r"cam\d+", re.IGNORECASE)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _regular_readable(path: Path) -> bool:
    try:
        mode = path.lstat().st_mode
    except OSError:
        return False
    return stat.S_ISREG(mode) and not path.is_symlink() and os.access(path, os.R_OK)


def _episode_records(metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = metadata.get("sequences", metadata.get("episodes", metadata.get("takes")))
    records: list[dict[str, Any]] = []
    if isinstance(raw, Mapping):
        items = sorted(
            raw.items(),
            key=lambda item: (
                0 if isinstance(item[0], str) and item[0].isdigit() else 1,
                int(item[0])
                if isinstance(item[0], str) and item[0].isdigit()
                else str(item[0]),
            ),
        )
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        items = list(enumerate(raw))
    else:
        return records
    for raw_id, value in items:
        if not isinstance(value, Mapping):
            continue
        episode_id = int(raw_id) if str(raw_id).isdigit() else len(records)
        action = value.get("action")
        if not isinstance(action, str) or not action.strip():
            for key in (
                "action_name",
                "manipulation",
                "primitive",
                "description",
                "instruction",
            ):
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    action = candidate
                    break
        records.append(
            {
                "episode_id": episode_id,
                "action": action.strip()
                if isinstance(action, str) and action.strip()
                else None,
                "bimanual": value.get("bimanual"),
                "nonprehensile": value.get("nonprehensile"),
                "metadata_keys": sorted(map(str, value.keys())),
            }
        )
    return records


def _files(parent: Path, suffix: str) -> list[Path]:
    try:
        return sorted(
            (
                path
                for path in parent.iterdir()
                if path.is_file() and path.suffix.lower() == suffix
            ),
            key=lambda path: path.name,
        )
    except OSError:
        return []


def _nonmedian_tactile(parent: Path) -> list[Path]:
    return [
        path
        for path in _files(parent, ".npy")
        if not path.name.lower().startswith("median_")
    ]


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _object_row(root: Path, object_id: str) -> dict[str, Any]:
    object_dir = root / "raw-repository" / "raw" / object_id
    metadata_path = object_dir / "metadata.json"
    if not object_dir.is_dir():
        return {
            "object_id": object_id,
            "present": False,
            "reason": "raw-object-directory-missing",
        }
    metadata: dict[str, Any] = {}
    metadata_error: str | None = None
    if (
        _regular_readable(metadata_path)
        and metadata_path.stat().st_size <= 2 * 1024 * 1024
    ):
        try:
            metadata = _load_json(metadata_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            metadata_error = f"{type(error).__name__}: {error}"
    else:
        metadata_error = "metadata-json-missing-unreadable-or-oversize"
    episodes = _episode_records(metadata)
    episode_count = len(episodes)

    tactile_groups: list[dict[str, Any]] = []
    camera_groups: list[dict[str, Any]] = []
    try:
        children = sorted(
            (path for path in object_dir.iterdir() if path.is_dir()),
            key=lambda p: p.name,
        )
    except OSError:
        children = []
    for child in children:
        tactile_files = (
            _nonmedian_tactile(child) if TACTILE_RE.search(child.name) else []
        )
        if tactile_files:
            tactile_groups.append(
                {
                    "directory": _safe_relative(child, root),
                    "recording_count": len(tactile_files),
                    "timestamp_sidecar_count": len(_files(child, ".txt")),
                    "median_count": len(
                        [
                            path
                            for path in _files(child, ".npy")
                            if path.name.lower().startswith("median_")
                        ]
                    ),
                    "recording_stems": [path.stem for path in tactile_files],
                    "recording_sizes_bytes": [
                        int(path.stat().st_size) for path in tactile_files
                    ],
                    "episode_order_mapping_ready": episode_count > 0
                    and len(tactile_files) == episode_count,
                }
            )
        if CAMERA_RE.search(child.name):
            videos = _files(child, ".mp4")
            stamps = _files(child, ".txt")
            if videos or stamps:
                camera_groups.append(
                    {
                        "directory": _safe_relative(child, root),
                        "video_count": len(videos),
                        "timestamp_count": len(stamps),
                        "paired_episode_count": len(
                            set(path.stem for path in videos)
                            & set(path.stem for path in stamps)
                        ),
                        "episode_order_mapping_ready": episode_count > 0
                        and len(videos) == episode_count
                        and len(stamps) == episode_count,
                    }
                )

    calibration_dir = object_dir / "calibration_refined"
    calibration = {
        "directory_present": calibration_dir.is_dir(),
        "intrinsics_present": _regular_readable(calibration_dir / "intrinsics.npy"),
        "extrinsics_present": _regular_readable(calibration_dir / "extrinsics.npy"),
        "distortion_present": _regular_readable(calibration_dir / "dist.npy"),
    }

    processed_candidates: list[str] = []
    robot_candidates: list[str] = []
    pcd_clean_directories: list[dict[str, Any]] = []
    for processed_parent in (
        root / "processed-repository" / "processed" / object_id,
        root / "processed-repository" / object_id,
        root / "processed" / object_id,
    ):
        if not processed_parent.is_dir():
            continue
        for path in processed_parent.rglob("*"):
            lower = path.name.lower()
            if path.is_file() and lower in {
                "robot.npz",
                "robot.npy",
                "actions.npz",
                "controls.npz",
            }:
                robot_candidates.append(_safe_relative(path, root))
            if path.is_dir() and path.name == "pcd_clean":
                count = len(_files(path, ".npz"))
                pcd_clean_directories.append(
                    {"directory": _safe_relative(path, root), "frame_count": count}
                )
            if path.is_file() and path.suffix.lower() in {".npz", ".h5", ".ply"}:
                processed_candidates.append(_safe_relative(path, root))

    camera_ready_count = sum(
        group["episode_order_mapping_ready"] for group in camera_groups
    )
    tactile_ready_count = sum(
        group["episode_order_mapping_ready"] for group in tactile_groups
    )
    action_count = sum(record["action"] is not None for record in episodes)
    official_processing_candidate = bool(
        episode_count >= 3
        and action_count == episode_count
        and all(calibration.values())
        and camera_ready_count >= 8
        and tactile_ready_count >= 2
    )
    one_dimensional = any(
        token in object_id.lower() for token in ("rope", "cable", "line", "band")
    )
    return {
        "object_id": object_id,
        "present": True,
        "one_dimensional_priority": one_dimensional,
        "metadata": {
            "path": _safe_relative(metadata_path, root),
            "sha256": _sha256_bytes(metadata_path.read_bytes())
            if metadata and _regular_readable(metadata_path)
            else None,
            "error": metadata_error,
            "top_level_keys": sorted(map(str, metadata.keys())),
            "episode_count": episode_count,
            "action_count": action_count,
            "episodes": episodes,
        },
        "calibration": calibration,
        "camera_group_count": len(camera_groups),
        "camera_groups_ready_for_order_mapping": camera_ready_count,
        "camera_groups": camera_groups,
        "tactile_group_count": len(tactile_groups),
        "tactile_groups_ready_for_order_mapping": tactile_ready_count,
        "tactile_groups": tactile_groups,
        "processed_numeric_candidate_count": len(processed_candidates),
        "processed_numeric_candidate_examples": processed_candidates[:20],
        "robot_state_candidates": sorted(set(robot_candidates)),
        "pcd_clean_directories": pcd_clean_directories,
        "official_robot_tactile_processing_candidate": official_processing_candidate,
    }


def audit(data_root: Path, protocol_path: Path) -> dict[str, Any]:
    root = data_root.resolve(strict=True)
    protocol = _load_json(protocol_path)
    expected = Path(str(protocol["dataset"]["root"]))
    if root != expected:
        raise ValueError(f"dataset root changed: {root} != {expected}")
    development = [str(value) for value in protocol["development_object_ids"]]
    reserved = {str(value) for value in protocol["forbidden_reserved_object_ids"]}
    if set(development) & reserved:
        raise ValueError("development and reserved rosters overlap")
    objects = [_object_row(root, object_id) for object_id in development]
    candidates = [
        row
        for row in objects
        if row.get("official_robot_tactile_processing_candidate") is True
    ]
    candidates.sort(
        key=lambda row: (
            not bool(row["one_dimensional_priority"]),
            -int(row["camera_groups_ready_for_order_mapping"]),
            -int(row["tactile_groups_ready_for_order_mapping"]),
            row["object_id"],
        )
    )
    action_counts = Counter(
        episode["action"]
        for row in objects
        for episode in row.get("metadata", {}).get("episodes", [])
        if episode.get("action")
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 2,
        "dataset_root": str(root),
        "protocol_path": str(protocol_path),
        "information_boundary": {
            "metadata_json_opened": True,
            "directory_names_and_file_sizes_opened": True,
            "camera_media_decoded": False,
            "tactile_numeric_payloads_opened": False,
            "robot_numeric_payloads_opened": False,
            "geometry_or_trajectory_payloads_opened": False,
            "target_outcomes_scored": False,
            "reserved_object_payloads_opened": False,
        },
        "summary": {
            "development_object_count": len(development),
            "present_object_count": sum(bool(row.get("present")) for row in objects),
            "objects_with_complete_action_metadata": sum(
                row.get("metadata", {}).get("episode_count", 0) >= 3
                and row.get("metadata", {}).get("episode_count")
                == row.get("metadata", {}).get("action_count")
                for row in objects
            ),
            "official_robot_tactile_processing_candidate_count": len(candidates),
            "objects_with_existing_robot_state": sum(
                bool(row.get("robot_state_candidates")) for row in objects
            ),
            "objects_with_existing_pcd_clean": sum(
                bool(row.get("pcd_clean_directories")) for row in objects
            ),
            "recommended_object_ids": [row["object_id"] for row in candidates[:8]],
            "action_vocabulary": dict(sorted(action_counts.items())),
        },
        "objects": objects,
    }
    result["result_sha256"] = _sha256_bytes(_canonical_bytes(result))
    return result


def report(result: Mapping[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# Deform360 ICRA carrier audit v2",
        "",
        f"- Development objects present: **{summary['present_object_count']}/{summary['development_object_count']}**",
        f"- Objects with complete action metadata: **{summary['objects_with_complete_action_metadata']}**",
        f"- Robot+tactile processing candidates: **{summary['official_robot_tactile_processing_candidate_count']}**",
        f"- Objects with an existing robot-state carrier: **{summary['objects_with_existing_robot_state']}**",
        f"- Objects with an existing `pcd_clean` carrier: **{summary['objects_with_existing_pcd_clean']}**",
        f"- Recommended first objects: `{', '.join(summary['recommended_object_ids'])}`",
        "",
        "## Candidate objects",
        "",
        "| Object | Episodes/actions | Ready cameras | Ready tactile groups | Existing robot | Existing pcd_clean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    candidate_ids = set(summary["recommended_object_ids"])
    for row in result["objects"]:
        if (
            row.get("official_robot_tactile_processing_candidate")
            or row["object_id"] in candidate_ids
        ):
            metadata = row.get("metadata", {})
            lines.append(
                f"| `{row['object_id']}` | {metadata.get('episode_count', 0)}/{metadata.get('action_count', 0)} | "
                f"{row.get('camera_groups_ready_for_order_mapping', 0)} | "
                f"{row.get('tactile_groups_ready_for_order_mapping', 0)} | "
                f"{int(bool(row.get('robot_state_candidates')))} | "
                f"{int(bool(row.get('pcd_clean_directories')))} |"
            )
    lines.extend(
        [
            "",
            "The audit opened metadata and file-system metadata only. It did not decode any",
            "camera, tactile, robot, geometry, trajectory, or reserved-object payload.",
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
    result = audit(args.data_root, args.protocol)
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
