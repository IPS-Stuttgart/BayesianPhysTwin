#!/usr/bin/env python3
"""Run the frozen depth, tracking, PCD, and control-point stages."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from causal4d_public.deform360_dense_source import sha256_file
from causal4d_public.deform360_reusable_dynamics import (
    load_reusable_dynamics_config,
    load_reusable_dynamics_pipeline_config,
    reusable_dynamics_result_sha256,
)
from deform360.processing import (
    control_points_stage,
    depth_stage,
    pcd_stage,
    tracking_stage,
)


def _git_revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _hash_tree(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--deform360-repo", type=Path, required=True)
    parser.add_argument("--cotracker-repo", type=Path, required=True)
    parser.add_argument("--aligned-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
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
    if _git_revision(args.deform360_repo) != frozen["deform360_processing_revision"]:
        raise ValueError("Deform360 processing revision changed")
    if sha256_file(args.checkpoint) != frozen["tracking"]["checkpoint_sha256"]:
        raise ValueError("CoTracker checkpoint changed")

    episode_dir = args.aligned_dir / f"episode_{args.episode:04d}"
    reconstruction_path = episode_dir / "reusable_dynamics_reconstruction.json"
    reconstruction = json.loads(reconstruction_path.read_text(encoding="utf-8"))
    if reconstruction.get("artifact_kind") != "Deform360ReusableDynamicsReconstruction":
        raise ValueError("unexpected reconstruction artifact")
    if reconstruction.get("pipeline_config_sha256") != pipeline["config_sha256"]:
        raise ValueError("reconstruction uses another observation pipeline")
    if reconstruction.get("result_sha256") != reusable_dynamics_result_sha256(
        reconstruction
    ):
        raise ValueError("reconstruction artifact checksum mismatch")
    cameras = list(reconstruction["accepted_cameras"])

    depth_outputs = depth_stage.process_depth_episode(
        args.aligned_dir,
        args.episode,
        cameras=cameras,
        overwrite=True,
        preview=frozen["depth"]["preview_video"],
    )
    tracking_outputs = tracking_stage.process_tracking_episode(
        args.aligned_dir,
        args.episode,
        cameras=cameras,
        checkpoint=args.checkpoint,
        overwrite=True,
    )
    pcd_dir = pcd_stage.process_pcd_episode(
        args.aligned_dir,
        args.episode,
        cameras=cameras,
        overwrite=True,
        rng_seed=frozen["point_cloud"]["rng_seed"],
    )
    control_outputs = control_points_stage.process_control_points_episode(
        args.aligned_dir,
        args.episode,
        cameras=cameras,
        overwrite=True,
    )
    pcd_files = sorted(pcd_dir.glob("*.npz"))
    if len(pcd_files) != frozen["point_cloud"]["expected_frame_count"]:
        raise ValueError("point-cloud stage returned another frame count")

    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360ReusableDynamicsObservations",
        "pipeline_config_sha256": pipeline["config_sha256"],
        "parent_config_sha256": parent["config_sha256"],
        "object_id": reconstruction["object_id"],
        "episode_id": reconstruction["episode_id"],
        "staged_episode_id": args.episode,
        "accepted_cameras": cameras,
        "point_cloud_frame_count": len(pcd_files),
        "input_sha256": {
            "parent_protocol": sha256_file(parent_path),
            "pipeline_protocol": sha256_file(pipeline_path),
            "reconstruction": sha256_file(reconstruction_path),
            "tracking_checkpoint": sha256_file(args.checkpoint),
        },
        "implementation_revision": {
            "deform360_processing": _git_revision(args.deform360_repo),
            "cotracker": _git_revision(args.cotracker_repo),
        },
        "output_sha256": {
            "depth": {
                camera: sha256_file(path)
                for camera, path in sorted(depth_outputs.items())
            },
            "tracking": {
                camera: _hash_tree(path)
                for camera, path in sorted(tracking_outputs.items())
            },
            "point_cloud": {
                path.name: sha256_file(path) for path in pcd_files
            },
            "control_points": {
                name: sha256_file(path)
                for name, path in sorted(control_outputs.items())
            },
        },
        "information_boundary": {
            "calibration_observation_pipeline_completed": True,
            "prediction_metrics_computed": False,
            "method_or_hyperparameter_changes_allowed": False,
            "target_media_read": False,
        },
        "claim_boundary": "observation artifacts only; no dynamics claim",
    }
    result["result_sha256"] = reusable_dynamics_result_sha256(result)
    output_path = episode_dir / "reusable_dynamics_observations.json"
    if output_path.exists():
        raise FileExistsError(f"observation artifact already exists: {output_path}")
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": True,
                "episode_id": reconstruction["episode_id"],
                "camera_count": len(cameras),
                "point_cloud_frame_count": len(pcd_files),
                "result_sha256": result["result_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
