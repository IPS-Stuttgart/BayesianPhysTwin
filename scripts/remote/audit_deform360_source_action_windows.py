#!/usr/bin/env python3
"""Audit full source action support against the locked Deform360 slice."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from causal4d_public.deform360_action_audit import summarize_robot_action
from causal4d_public.deform360_dense_reusable_panel import (
    authorize_dense_panel_episode,
    load_dense_reusable_panel_config,
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--aligned-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _aggregate(records: list[dict[str, Any]]) -> dict[str, float | int]:
    return {
        "episode_count": len(records),
        "mean_locked_to_best_displacement_ratio": float(
            np.mean(
                [row["action"]["locked_to_best_displacement_ratio"] for row in records]
            )
        ),
        "mean_locked_to_best_path_ratio": float(
            np.mean([row["action"]["locked_to_best_path_ratio"] for row in records])
        ),
        "mean_locked_to_best_contact_conditioned_path_ratio": float(
            np.mean(
                [
                    row["action"]["locked_to_best_contact_conditioned_path_ratio"]
                    for row in records
                ]
            )
        ),
        "mean_locked_controller_displacement_m": float(
            np.mean(
                [
                    row["action"]["locked_window"][
                        "mean_displacement_from_window_start_m"
                    ]
                    for row in records
                ]
            )
        ),
        "mean_best_controller_displacement_m": float(
            np.mean(
                [
                    row["action"]["best_equal_length_displacement_window"][
                        "mean_displacement_from_window_start_m"
                    ]
                    for row in records
                ]
            )
        ),
        "mean_best_contact_conditioned_path_m": float(
            np.mean(
                [
                    row["action"]["best_contact_conditioned_path_window"][
                        "mean_closed_weighted_path_length_m"
                    ]
                    for row in records
                ]
            )
        ),
    }


def main() -> int:
    args = _parse_args()
    protocol = load_dense_reusable_panel_config(args.config)
    config = protocol["config"]
    selection = config["frame_protocol"]["window_selection"]
    locked_start, locked_stop = config["frame_protocol"][
        "superseded_fixed_raw_aligned_range_half_open"
    ]
    records: list[dict[str, Any]] = []
    for object_row in config["cohort"]:
        object_id = str(object_row["object_id"])
        for episode_id in object_row["source_episode_ids"]:
            authorize_dense_panel_episode(
                protocol,
                object_id=object_id,
                episode_id=int(episode_id),
                phase="source",
                source_admission_passed=False,
            )
            robot_path = (
                args.aligned_root
                / object_id
                / f"episode_{int(episode_id):04d}"
                / "robot"
                / "robot.npz"
            )
            if not robot_path.is_file():
                raise FileNotFoundError(f"missing source robot artifact {robot_path}")
            with np.load(robot_path, allow_pickle=False) as robot:
                action = summarize_robot_action(
                    robot["actions"],
                    robot["openings"],
                    locked_start=int(locked_start),
                    locked_stop=int(locked_stop),
                    candidate_start_frame=int(selection["candidate_starts"]["first"]),
                    candidate_stride_frames=int(
                        selection["candidate_starts"]["stride"]
                    ),
                )
                declared_bimanual = bool(robot["bimanual"])
            if (action["gripper_count"] == 2) != declared_bimanual:
                raise ValueError(f"bimanual metadata disagrees in {robot_path}")
            records.append(
                {
                    "object_id": object_id,
                    "stratum": str(object_row["stratum"]),
                    "episode_id": int(episode_id),
                    "robot_path": str(robot_path.resolve()),
                    "robot_sha256": _sha256_file(robot_path),
                    "action": action,
                }
            )

    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_object[record["object_id"]].append(record)
        by_stratum[record["stratum"]].append(record)
    payload = {
        "schema_version": 1,
        "artifact_kind": "Deform360SourceActionWindowAudit",
        "protocol_id": config["protocol_id"],
        "config_sha256": protocol["config_sha256"],
        "evidence_scope": "source-actions-only",
        "superseded_fixed_raw_aligned_range_half_open": [locked_start, locked_stop],
        "prospective_window_rule": config["frame_protocol"]["window_selection"],
        "records": records,
        "aggregate": _aggregate(records),
        "aggregate_by_object": {
            key: _aggregate(value) for key, value in sorted(by_object.items())
        },
        "aggregate_by_stratum": {
            key: _aggregate(value) for key, value in sorted(by_stratum.items())
        },
        "target_paths_constructed": False,
        "target_initial_frame_read": False,
        "target_future_read": False,
    }
    payload["result_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
