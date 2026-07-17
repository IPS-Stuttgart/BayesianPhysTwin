#!/usr/bin/env python3
"""Audit source/calibration frame coverage without opening panel targets."""

from __future__ import annotations

import argparse
import hashlib
import json
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--replication-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _path_length(actions: np.ndarray, start: int, stop: int) -> float:
    selected = actions[start:stop]
    if len(selected) < 2:
        return 0.0
    step_length = np.linalg.norm(np.diff(selected, axis=0), axis=-1)
    return float(np.mean(step_length, axis=1).sum())


def main() -> int:
    args = _parse_args()
    config_path = (
        args.repo / "configs/causal4d_public/deform360_dense_reusable_panel_v1.json"
    )
    protocol = load_dense_reusable_panel_config(config_path)
    config = protocol["config"]
    selection = config["frame_protocol"]["window_selection"]
    old_start, old_stop = config["frame_protocol"][
        "superseded_fixed_raw_aligned_range_half_open"
    ]
    records = []

    for row in config["cohort"]:
        object_id = str(row["object_id"])
        for phase, episode_ids in (
            ("source", row["source_episode_ids"]),
            ("calibration", row["calibration_episode_ids"]),
        ):
            for episode_id in episode_ids:
                authorize_dense_panel_episode(
                    protocol,
                    object_id=object_id,
                    episode_id=int(episode_id),
                    phase=phase,
                    source_admission_passed=phase == "calibration",
                )
                episode_dir = (
                    args.replication_root
                    / "aligned"
                    / object_id
                    / f"episode_{int(episode_id):04d}"
                )
                alignment_path = episode_dir / "alignment.json"
                robot_path = episode_dir / "robot" / "robot.npz"
                alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
                with np.load(robot_path) as robot:
                    raw_actions = np.asarray(robot["actions"], dtype=np.float64)
                    openings = np.asarray(robot["openings"], dtype=np.float64)
                    bimanual = bool(robot["bimanual"])
                frame_count = int(alignment["frame_count"])
                if raw_actions.ndim not in (3, 4) or raw_actions.shape[-1] != 3:
                    raise ValueError(
                        f"invalid robot actions for {object_id}/{episode_id}"
                    )
                controller_group_count = 1
                actions = raw_actions
                if raw_actions.ndim == 4:
                    controller_group_count = int(raw_actions.shape[1])
                    actions = raw_actions.reshape(len(raw_actions), -1, 3)
                if len(actions) != frame_count or frame_count < old_stop:
                    raise ValueError(
                        f"invalid frame coverage for {object_id}/{episode_id}"
                    )
                if not np.isfinite(actions).all():
                    raise ValueError(
                        f"non-finite robot actions for {object_id}/{episode_id}"
                    )
                full_path = _path_length(actions, 0, frame_count)
                action_summary = summarize_robot_action(
                    raw_actions,
                    openings,
                    locked_start=int(old_start),
                    locked_stop=int(old_stop),
                    candidate_start_frame=int(selection["candidate_starts"]["first"]),
                    candidate_stride_frames=int(
                        selection["candidate_starts"]["stride"]
                    ),
                )
                selected_range = action_summary["best_contact_conditioned_path_window"][
                    "frame_range_half_open"
                ]
                selected_path = _path_length(
                    actions, int(selected_range[0]), int(selected_range[1])
                )
                records.append(
                    {
                        "object_id": object_id,
                        "stratum": row["stratum"],
                        "phase": phase,
                        "episode_id": int(episode_id),
                        "frame_count": frame_count,
                        "bimanual": bimanual,
                        "controller_group_count": controller_group_count,
                        "controller_point_count": int(actions.shape[1]),
                        "full_mean_point_path_m": full_path,
                        "selected_raw_frame_range_half_open": selected_range,
                        "selected_mean_point_path_m": selected_path,
                        "selected_fraction_of_full_path": (
                            selected_path / full_path if full_path > 0.0 else 0.0
                        ),
                        "alignment_timeline_sha256": alignment["timeline_sha256"],
                    }
                )

    payload = {
        "schema_version": 1,
        "artifact_kind": "Deform360DenseReusablePanelSourceFrameAudit",
        "protocol_id": config["protocol_id"],
        "config_sha256": protocol["config_sha256"],
        "window_selection": config["frame_protocol"]["window_selection"],
        "superseded_fixed_raw_aligned_range_half_open": [old_start, old_stop],
        "record_count": len(records),
        "records": records,
        "target_prefix_read": False,
        "target_future_read": False,
        "passed": len(records) == 45,
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
