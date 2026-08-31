#!/usr/bin/env python3
"""Compatibility wrapper for selective Deform360 TCN carrier materialization.

The pinned public snapshots contain two transport irregularities relative to the
frozen local evaluation mount:

* a small number of metadata episodes have no processed robot carrier; and
* one tactile sensor directory for ``052-rubber-duck`` contains one extra data
  file, even though the other synchronized sensor directories contain exactly
  one file per metadata episode.

The frozen evaluator admits only robot-complete episodes.  For a tactile directory
with extra files, this wrapper selects the monotone subset whose timestamps best
align to the same episode timestamps in the complete peer sensor directories.  It
does not download camera, video, point-cloud, or splat payloads.
"""

from __future__ import annotations

import importlib.util
import math
import statistics
import sys
import urllib.error
from pathlib import Path
from typing import Any, Sequence

# Explicitly retained for the workflow's closed payload/revision contract.
RAW_REVISION = "5ea8c5d3fc7b4a7b4f9f921f2ceb1de24610f6a4"
PROCESSED_REVISION = "e92deaf7e437e7e51ad464706ae647f522a279d9"
FORBIDDEN_PAYLOAD_TOKENS = ("camera", "pcd", "splat")

HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "materialize_deform360_tcn_carriers_v6_base.py"
SPEC = importlib.util.spec_from_file_location(
    "deform360_tcn_carrier_materializer_v6_base",
    BASE_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import base materializer: {BASE_PATH}")
base = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = base
SPEC.loader.exec_module(base)

if base.RAW_REVISION != RAW_REVISION:
    raise RuntimeError("raw release revision differs from wrapper contract")
if base.PROCESSED_REVISION != PROCESSED_REVISION:
    raise RuntimeError("processed release revision differs from wrapper contract")
if not set(FORBIDDEN_PAYLOAD_TOKENS).issubset(set(base.FORBIDDEN_TOKENS)):
    raise RuntimeError("forbidden payload boundary differs from wrapper contract")

_original_discover_object_plans = base.discover_object_plans


def _processed_episode_directories(object_id: str) -> dict[str, str]:
    object_root = f"processed/{object_id}"
    entries = base.list_tree(
        base.PROCESSED_REPOSITORY,
        base.PROCESSED_REVISION,
        object_root,
    )
    return {
        Path(str(item.get("path", ""))).name: str(item.get("path"))
        for item in entries
        if item.get("type") == "directory"
    }


def _robot_plan(
    root: Path,
    object_id: str,
    episode_id: int,
    directories: dict[str, str],
) -> Any | None:
    episode_path = None
    for name in (
        f"episode_{episode_id}",
        f"episode_{episode_id:04d}",
        f"episode-{episode_id}",
    ):
        if name in directories:
            episode_path = directories[name]
            break
    if episode_path is None:
        return None

    robot_directory = f"{episode_path}/robot"
    try:
        entries = base.list_tree(
            base.PROCESSED_REPOSITORY,
            base.PROCESSED_REVISION,
            robot_directory,
        )
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise

    files = sorted(
        (
            item
            for item in entries
            if item.get("type") == "file"
            and Path(str(item.get("path", ""))).name.lower()
            in {"robot.npy", "robot.npz"}
        ),
        key=lambda item: (
            0
            if Path(str(item.get("path", ""))).name.lower() == "robot.npy"
            else 1,
            str(item.get("path", "")),
        ),
    )
    if not files:
        return None

    selected = files[0]
    source = str(selected["path"])
    filename = Path(source).name
    destination = (
        root
        / "processed-repository"
        / "processed"
        / object_id
        / f"episode_{episode_id}"
        / "robot"
        / filename
    )
    return base.DownloadPlan(
        repository=base.PROCESSED_REPOSITORY,
        revision=base.PROCESSED_REVISION,
        source_candidates=(source,),
        destination=destination,
        kind="robot",
        expected_size=int(selected.get("size") or 0) or None,
    )


def _directory_entries(directory: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    children = base.list_tree(base.RAW_REPOSITORY, base.RAW_REVISION, directory)
    selected = sorted(
        (
            item
            for item in children
            if item.get("type") == "file"
            and str(item.get("path", "")).lower().endswith(".npy")
        ),
        key=lambda item: str(item.get("path", "")),
    )
    data = [
        item
        for item in selected
        if not Path(str(item.get("path", ""))).name.lower().startswith("median_")
    ]
    medians = [item for item in selected if item not in data]
    return data, medians


def _timestamp(item: dict[str, Any]) -> int:
    stem = Path(str(item["path"])).stem
    try:
        return int(stem.rsplit("_", 1)[-1])
    except ValueError as error:
        raise base.MaterializationError(
            f"cannot parse tactile timestamp: {item['path']}"
        ) from error


def _monotone_timestamp_subset(
    candidates: Sequence[dict[str, Any]],
    references: Sequence[float],
) -> list[dict[str, Any]]:
    """Select one ordered candidate per reference with minimum absolute offset."""

    candidate_rows = sorted(candidates, key=_timestamp)
    m = len(candidate_rows)
    n = len(references)
    if m < n:
        raise base.MaterializationError(
            f"cannot align {m} tactile files to {n} episode timestamps"
        )

    cost = [[math.inf] * (n + 1) for _ in range(m + 1)]
    take = [[False] * (n + 1) for _ in range(m + 1)]
    cost[0][0] = 0.0
    for i in range(1, m + 1):
        cost[i][0] = 0.0
        for j in range(1, min(i, n) + 1):
            skipped = cost[i - 1][j]
            selected = cost[i - 1][j - 1] + abs(
                float(_timestamp(candidate_rows[i - 1])) - float(references[j - 1])
            )
            if selected < skipped:
                cost[i][j] = selected
                take[i][j] = True
            else:
                cost[i][j] = skipped

    if not math.isfinite(cost[m][n]):
        raise base.MaterializationError("tactile timestamp alignment is infeasible")
    indices: list[int] = []
    i, j = m, n
    while j:
        if i <= 0:
            raise base.MaterializationError("tactile alignment backtracking failed")
        if take[i][j]:
            indices.append(i - 1)
            i -= 1
            j -= 1
        else:
            i -= 1
    indices.reverse()
    return [candidate_rows[index] for index in indices]


def _restore_extra_tactile_directories(
    root: Path,
    inventory: dict[str, Any],
) -> tuple[list[Any], dict[str, Any]]:
    episode_count = len(inventory["episode_ids"])
    tactile_directories = tuple(inventory["tactile_directories"])
    accepted_names = set(inventory["tactile_episode_counts"])

    peers: list[list[int]] = []
    directory_payloads: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for directory in tactile_directories:
        data, medians = _directory_entries(directory)
        directory_payloads[directory] = (data, medians)
        if Path(directory).name in accepted_names and len(data) == episode_count:
            peers.append([_timestamp(item) for item in sorted(data, key=_timestamp)])

    if not peers:
        return [], inventory
    references = [
        float(statistics.median(peer[index] for peer in peers))
        for index in range(episode_count)
    ]

    plans: list[Any] = []
    restored: dict[str, Any] = {}
    for directory, (data, medians) in directory_payloads.items():
        name = Path(directory).name
        if name in accepted_names or len(data) <= episode_count:
            continue
        selected_data = _monotone_timestamp_subset(data, references)
        selected_names = {Path(str(item["path"])).name for item in selected_data}
        discarded = [
            str(item["path"])
            for item in sorted(data, key=_timestamp)
            if Path(str(item["path"])).name not in selected_names
        ]

        # Remove an unselected stale cache file if a prior transport attempt
        # materialized the entire remote directory.
        local_directory = root / "raw-repository" / directory
        if local_directory.is_dir():
            for path in local_directory.glob("*.npy"):
                if path.name.lower().startswith("median_"):
                    continue
                if path.name not in selected_names:
                    path.unlink()

        for item in [*selected_data, *medians]:
            source = str(item["path"])
            plans.append(
                base.DownloadPlan(
                    repository=base.RAW_REPOSITORY,
                    revision=base.RAW_REVISION,
                    source_candidates=(source,),
                    destination=root / "raw-repository" / source,
                    kind="tactile",
                    expected_size=int(item.get("size") or 0) or None,
                )
            )
        inventory["tactile_episode_counts"][name] = len(selected_data)
        restored[name] = {
            "remote_nonmedian_count": len(data),
            "selected_nonmedian_count": len(selected_data),
            "selected_files": sorted(selected_names),
            "discarded_files": discarded,
            "selection_rule": (
                "minimum-cost monotone timestamp alignment to complete peer "
                "sensor directories"
            ),
        }

    if restored:
        inventory["restored_extra_tactile_directories"] = restored
    return plans, inventory


def discover_object_plans(
    root: Path,
    object_id: str,
) -> tuple[Any, list[Any], dict[str, Any]]:
    metadata_record, plans, inventory = _original_discover_object_plans(
        root,
        object_id,
    )
    tactile_plans = [plan for plan in plans if plan.kind != "robot"]
    restored_plans, inventory = _restore_extra_tactile_directories(root, inventory)
    tactile_plans.extend(restored_plans)

    directories = _processed_episode_directories(object_id)
    robot_plans: list[Any] = []
    available: list[int] = []
    missing: list[int] = []
    for raw_episode_id in inventory["episode_ids"]:
        episode_id = int(raw_episode_id)
        plan = _robot_plan(root, object_id, episode_id, directories)
        if plan is None:
            missing.append(episode_id)
            continue
        robot_plans.append(plan)
        available.append(episode_id)

    # Match the frozen evaluator's carrier-completeness admission boundary.
    if len(robot_plans) < 4:
        raise base.MaterializationError(
            f"too few robot-complete episodes for {object_id}: "
            f"available={available}, missing={missing}"
        )

    inventory = dict(inventory)
    inventory.update(
        {
            "available_robot_episode_ids": available,
            "missing_robot_episode_ids": missing,
            "robot_complete_episode_count": len(available),
            "planned_tactile_files": len(tactile_plans),
            "planned_robot_files": len(robot_plans),
            "robot_episode_rule": (
                "processed robot carrier must exist; identical to the frozen "
                "local evaluator's carrier-completeness rule"
            ),
        }
    )
    return metadata_record, tactile_plans + robot_plans, inventory


base.discover_object_plans = discover_object_plans

if __name__ == "__main__":
    base.main()
