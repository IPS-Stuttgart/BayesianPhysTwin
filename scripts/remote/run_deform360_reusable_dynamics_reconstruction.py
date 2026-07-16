#!/usr/bin/env python3
"""Run the frozen dense reconstruction for one reusable-twin calibration case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from causal4d_public.deform360_dense_source import sha256_file
from causal4d_public.deform360_reusable_dynamics import (
    load_reusable_dynamics_config,
    load_reusable_dynamics_pipeline_config,
    reusable_dynamics_result_sha256,
)
from deform360.processing import reconstruct_stage


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--aligned-dir", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    parent_path = (
        args.repo / "configs/causal4d_public/deform360_reusable_dynamics_081_v1.json"
    )
    pipeline_path = (
        args.repo
        / "configs/causal4d_public/deform360_reusable_dynamics_pipeline_081_v1.json"
    )
    parent = load_reusable_dynamics_config(parent_path)
    pipeline = load_reusable_dynamics_pipeline_config(
        pipeline_path, parent=parent
    )
    frozen = pipeline["config"]
    episode_dir = args.aligned_dir / f"episode_{args.episode:04d}"
    staging_path = episode_dir / "reusable_dynamics_staging.manifest.json"
    staging = json.loads(staging_path.read_text(encoding="utf-8"))
    if staging.get("artifact_kind") != "Deform360ReusableDynamicsCalibrationStaging":
        raise ValueError("unexpected dynamics staging artifact")
    if staging.get("config_sha256") != parent["config_sha256"]:
        raise ValueError("dynamics staging uses another parent protocol")
    if staging.get("result_sha256") != reusable_dynamics_result_sha256(staging):
        raise ValueError("dynamics staging checksum mismatch")
    if staging.get("frame_count") != frozen["staging"]["frame_count"]:
        raise ValueError("dynamics staging frame count changed")
    boundary = staging.get("information_boundary", {})
    if boundary.get("method_or_hyperparameter_changes_allowed") is not False:
        raise ValueError("staging permits calibration-driven changes")
    if boundary.get("target_media_read") is not False:
        raise ValueError("staging read target media")
    cameras = list(staging["accepted_cameras"])

    reconstruction = frozen["reconstruction"]
    original = reconstruct_stage.visual_hull_points

    def strict_visual_hull_points(*call_args: object, **call_kwargs: object):
        call_kwargs["min_points"] = reconstruction["minimum_visual_hull_points"]
        return original(*call_args, **call_kwargs)

    reconstruct_stage.visual_hull_points = strict_visual_hull_points
    try:
        outputs = reconstruct_stage.process_reconstruction_episode(
            args.aligned_dir,
            args.episode,
            cameras=cameras,
            first_frame_iterations=reconstruction["first_frame_iterations"],
            warm_start_iterations=reconstruction["warm_start_iterations"],
            cube_half_extent_m=reconstruction["cube_half_extent_m"],
            voxel_resolution=reconstruction["voxel_resolution"],
            overwrite=True,
        )
    finally:
        reconstruct_stage.visual_hull_points = original
    if sorted(outputs) != list(range(frozen["staging"]["frame_count"])):
        raise ValueError("dense reconstruction returned an incomplete frame set")

    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360ReusableDynamicsReconstruction",
        "pipeline_config_sha256": pipeline["config_sha256"],
        "parent_config_sha256": parent["config_sha256"],
        "object_id": staging["object_id"],
        "episode_id": staging["episode_id"],
        "staged_episode_id": args.episode,
        "accepted_cameras": cameras,
        "settings": reconstruction,
        "input_sha256": {
            "parent_protocol": sha256_file(parent_path),
            "pipeline_protocol": sha256_file(pipeline_path),
            "staging_manifest": sha256_file(staging_path),
        },
        "output_sha256": {
            str(frame): sha256_file(path) for frame, path in sorted(outputs.items())
        },
        "information_boundary": {
            "calibration_reconstruction_completed": True,
            "prediction_metrics_computed": False,
            "method_or_hyperparameter_changes_allowed": False,
            "target_media_read": False,
        },
        "claim_boundary": "observation reconstruction only; no dynamics claim",
    }
    result["result_sha256"] = reusable_dynamics_result_sha256(result)
    output_path = episode_dir / "reusable_dynamics_reconstruction.json"
    if output_path.exists():
        raise FileExistsError(f"reconstruction artifact already exists: {output_path}")
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": True,
                "episode_id": staging["episode_id"],
                "frame_count": len(outputs),
                "camera_count": len(cameras),
                "result_sha256": result["result_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
