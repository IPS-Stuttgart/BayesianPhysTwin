#!/usr/bin/env python3
"""Compatibility wrapper for selective Deform360 TCN carrier materialization.

The pinned processed release is incomplete for a small number of metadata
episodes. The original evaluator defines an episode as usable only when both
its tactile carriers and processed robot carrier exist. This wrapper preserves
that rule during remote materialization instead of treating every metadata
entry as a mandatory robot download.
"""

from __future__ import annotations

import importlib.util
import sys
import urllib.error
from pathlib import Path
from typing import Any

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


def discover_object_plans(
    root: Path,
    object_id: str,
) -> tuple[Any, list[Any], dict[str, Any]]:
    metadata_record, plans, inventory = _original_discover_object_plans(
        root,
        object_id,
    )
    tactile_plans = [plan for plan in plans if plan.kind != "robot"]
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
