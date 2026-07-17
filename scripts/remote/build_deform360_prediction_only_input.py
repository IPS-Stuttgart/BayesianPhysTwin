#!/usr/bin/env python3
"""Build a PhysTwin input from frame-zero geometry and the known robot action."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from causal4d_public.deform360_independent_source import (
    authorize_independent_source_episode,
    load_independent_source_lock,
    sha256_array,
    sha256_file,
    validate_prediction_only_bundle,
)
from deform360.processing.control_points_stage import _frame_controller_points
from deform360.processing.pcd_stage import (
    CROP_HALF_EXTENT_M,
    SEED_POINT_COUNT,
    seed_points_from_splat,
)
from deform360.robot import load_robot_state


def _result_sha256(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, default=76)
    parser.add_argument("--rng-seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    lock = load_independent_source_lock(args.lock)
    authorization = authorize_independent_source_episode(
        lock, args.object_id, args.episode_id
    )
    if args.frame_count != 76:
        raise ValueError("the independent-source predictor requires 76 frames")
    episode_dir = args.episode_dir.resolve()
    manifest_path = episode_dir / "dense_source_smoke.manifest.json"
    alignment_path = episode_dir / "action_aligned_source_staging.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    if (
        manifest.get("source_only") is not True
        or manifest.get("target_episode_accessed") is not False
        or manifest.get("calibration_episode_accessed") is not False
        or manifest.get("object_id") != args.object_id
        or int(manifest.get("episode_index", -1)) != args.episode_id
    ):
        raise ValueError("staged episode is outside the source-only boundary")
    if (
        alignment.get("object_id") != args.object_id
        or int(alignment.get("episode_id", -1)) != args.episode_id
        or alignment.get("source_only") is not True
        or alignment.get("target_action_read") is not False
        or alignment.get("target_observation_read") is not False
        or alignment.get("target_future_read") is not False
    ):
        raise ValueError("staged episode does not use the locked action alignment")
    frame_range = alignment.get("selected_raw_frame_range_half_open")
    if (
        not isinstance(frame_range, list)
        or len(frame_range) != 2
        or int(frame_range[1]) - int(frame_range[0]) != 81
    ):
        raise ValueError("action-aligned source window is not 81 frames")

    splat_path = episode_dir / "splatfacto" / "splat_0.ply"
    robot_path = episode_dir / "robot" / "robot.npz"
    points, colors = seed_points_from_splat(
        splat_path,
        crop_half_extent_m=CROP_HALF_EXTENT_M,
        seed_count=SEED_POINT_COUNT,
        rng_seed=args.rng_seed,
    )
    minimum = int(lock["frozen_predictor"]["minimum_observed_graph_node_count"])
    if len(points) < minimum:
        raise ValueError(
            f"frame-zero reconstruction has {len(points)} points, below {minimum}"
        )
    state = load_robot_state(robot_path)
    if state.num_frames < args.frame_count:
        raise ValueError("robot trajectory is shorter than the prediction horizon")
    controllers = np.stack(
        [_frame_controller_points(state, frame) for frame in range(args.frame_count)]
    ).astype(np.float32)
    object_points = np.repeat(points[None], args.frame_count, axis=0).astype(np.float32)
    object_colors = np.repeat(colors[None], args.frame_count, axis=0).astype(np.float32)
    observed = np.ones((args.frame_count, len(points)), dtype=bool)
    payload = {
        "object_points": object_points,
        "object_colors": object_colors,
        "object_visibilities": observed,
        "object_motions_valid": observed.copy(),
        "controller_points": controllers,
        "surface_points": np.empty((0, 3), dtype=np.float32),
        "interior_points": np.empty((0, 3), dtype=np.float32),
        "prediction_only_input": {
            "schema_version": 1,
            **authorization,
            "object_observation_frames_used": [0],
            "known_future_robot_trajectory_used": True,
            "future_object_observations_present": False,
            "future_tactile_used": False,
            "component_policy": (
                "retain all points from the object-only strict-hull frame-zero splat"
            ),
            "frame_zero_splat_sha256": sha256_file(splat_path),
            "robot_trajectory_sha256": sha256_file(robot_path),
            "action_alignment_result_sha256": alignment["result_sha256"],
        },
    }
    validation = validate_prediction_only_bundle(
        payload, object_id=args.object_id, episode_id=args.episode_id
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    summary = {
        "schema_version": 1,
        "artifact_kind": "Deform360PredictionOnlyInput",
        **authorization,
        "lock_sha256": sha256_file(args.lock),
        "frame_zero": {
            "point_count": len(points),
            "points_sha256": sha256_array(points),
            "colors_sha256": sha256_array(colors),
            "splat_sha256": sha256_file(splat_path),
            "component_policy": "retain_all_object_only_strict_hull_points",
        },
        "robot": {
            "input_frame_count": state.num_frames,
            "prediction_frame_count": args.frame_count,
            "controller_point_count": controllers.shape[1],
            "robot_sha256": sha256_file(robot_path),
            "controller_trajectory_sha256": sha256_array(controllers),
        },
        "validation": validation,
        "input_sha256": {
            "source_manifest": sha256_file(manifest_path),
            "action_alignment": sha256_file(alignment_path),
            "frame_zero_splat": sha256_file(splat_path),
            "robot": sha256_file(robot_path),
        },
        "output_sha256": sha256_file(args.output),
        "information_boundary": {
            "object_observation_frames_used": [0],
            "future_robot_action_available": True,
            "post_initial_object_observation_used": False,
            "future_tactile_used": False,
            "target_access": False,
        },
        "passed": True,
    }
    summary["result_sha256"] = _result_sha256(summary)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
