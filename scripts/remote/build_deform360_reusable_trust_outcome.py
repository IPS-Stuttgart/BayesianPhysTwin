#!/usr/bin/env python3
"""Build a fresh-panel outcome only after its prospective access seal exists."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from causal4d_public.deform360_dense_source import sha256_file
from causal4d_public.deform360_independent_source import load_independent_source_lock
from causal4d_public.deform360_reusable_physics import (
    validate_reusable_physics_fit_grid_seal,
)
from causal4d_public.deform360_reusable_trust_protocol import (
    authorize_reusable_trust_held_outcome,
    load_reusable_trust_protocol,
    validate_reusable_trust_prediction_cohort_seal,
)
from deform360.processing import depth_stage, pcd_stage, tracking_stage
from deform360.processing.control_points_stage import _frame_controller_points
from deform360.processing.episode import episode_cameras
from deform360.robot import load_robot_state


def _result_sha256(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    parser.add_argument("--parent-lock", type=Path, required=True)
    parser.add_argument("--physics-addendum", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--observation-consensus-lock", type=Path, required=True)
    parser.add_argument("--operation", choices=("fit", "held-outcome"), required=True)
    parser.add_argument("--future-access-seal", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--aligned-dir", type=Path, required=True)
    parser.add_argument("--staged-episode", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-validated-stages", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    protocol = load_reusable_trust_protocol(
        args.parent_lock, args.physics_addendum, args.execution_lock
    )
    access_seal = json.loads(
        args.future_access_seal.read_text(encoding="utf-8")
    )
    if args.operation == "fit":
        access = validate_reusable_physics_fit_grid_seal(
            access_seal, protocol=protocol, verify_responses=True
        )
        if (
            access["object_id"] != args.object_id
            or access["episode_id"] != args.episode_id
        ):
            raise ValueError("fit-grid seal belongs to another episode")
        first_response = access_seal["responses"]["0"]
        response_payload = json.loads(
            Path(first_response["response_json_path"]).read_text(encoding="utf-8")
        )
        initial_archive = Path(response_payload["prediction_archive"]["path"])
        role = "object-level-fit"
    else:
        validate_reusable_trust_prediction_cohort_seal(
            access_seal, protocol=protocol, verify_predictions=True
        )
        authorize_reusable_trust_held_outcome(
            protocol,
            access_seal,
            object_id=args.object_id,
            episode_id=args.episode_id,
        )
        prediction_record = access_seal["predictions"][
            f"{args.object_id}/{args.episode_id}"
        ]
        prediction_payload = json.loads(
            Path(prediction_record["prediction_json_path"]).read_text(
                encoding="utf-8"
            )
        )
        initial_archive = Path(prediction_payload["output"]["path"])
        role = "held-out-evaluation"

    episode_dir = args.aligned_dir / f"episode_{args.staged_episode:04d}"
    manifest_path = episode_dir / "dense_source_smoke.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("source_only") is not True
        or manifest.get("target_episode_accessed") is not False
        or manifest.get("calibration_episode_accessed") is not False
        or manifest.get("object_id") != args.object_id
        or int(manifest.get("episode_index", -1)) != args.episode_id
    ):
        raise ValueError("staged episode differs from the authorized fresh episode")

    reconstruction_path = episode_dir / "strict_hull_reconstruction_full.meta.json"
    reconstruction = json.loads(reconstruction_path.read_text(encoding="utf-8"))
    if (
        reconstruction.get("frame_zero_only") is not False
        or reconstruction.get("future_access_seal", {}).get("result_sha256")
        != access_seal["result_sha256"]
        or reconstruction.get("information_boundary", {}).get(
            "prediction_seal_verified_before_future_reconstruction"
        )
        is not True
        or sorted(int(frame) for frame in reconstruction.get("outputs", {}))
        != list(range(81))
    ):
        raise ValueError("full reconstruction was not opened after this access seal")

    parent_processing = json.loads(
        (
            args.repo
            / "configs/causal4d_public/deform360_reusable_dynamics_pipeline_081_v1.json"
        ).read_text(encoding="utf-8")
    )["config"]
    if _git_revision(args.deform360_repo) != parent_processing[
        "deform360_processing_revision"
    ]:
        raise ValueError("Deform360 processing revision changed")
    if sha256_file(args.checkpoint) != parent_processing["tracking"][
        "checkpoint_sha256"
    ]:
        raise ValueError("CoTracker checkpoint changed")
    consensus_lock = load_independent_source_lock(args.observation_consensus_lock)
    if consensus_lock["protocol_id"] != protocol["parent"]["parent_protocol_id"]:
        raise ValueError("observation consensus is not the fresh protocol parent")

    cameras = list(episode_cameras(episode_dir))
    overwrite = not args.reuse_validated_stages
    depth_outputs = depth_stage.process_depth_episode(
        args.aligned_dir,
        args.staged_episode,
        cameras=cameras,
        overwrite=overwrite,
        preview=False,
    )
    tracking_outputs = tracking_stage.process_tracking_episode(
        args.aligned_dir,
        args.staged_episode,
        cameras=cameras,
        checkpoint=args.checkpoint,
        overwrite=overwrite,
    )

    consensus = consensus_lock["frozen_predictor"]["observation_consensus"]
    threshold = float(consensus["maximum_speed_m_per_s"])
    minimum_inliers = int(consensus["minimum_camera_inlier_count"])
    original_threshold = pcd_stage.FUSE_RANSAC_THRESHOLD
    original_minimum = pcd_stage.FUSE_RANSAC_MIN_INLIERS
    pcd_stage.FUSE_RANSAC_THRESHOLD = threshold
    pcd_stage.FUSE_RANSAC_MIN_INLIERS = minimum_inliers
    try:
        pcd_dir = pcd_stage.process_pcd_episode(
            args.aligned_dir,
            args.staged_episode,
            cameras=cameras,
            overwrite=overwrite,
            rng_seed=0,
        )
    finally:
        pcd_stage.FUSE_RANSAC_THRESHOLD = original_threshold
        pcd_stage.FUSE_RANSAC_MIN_INLIERS = original_minimum

    pcd_files = sorted(pcd_dir.glob("*.npz"))
    if len(pcd_files) != 76:
        raise ValueError("fresh outcome must contain 76 point frames")
    object_points = []
    object_colors = []
    for path in pcd_files:
        with np.load(path, allow_pickle=False) as stored:
            object_points.append(np.asarray(stored["pts"], dtype=np.float32))
            object_colors.append(np.asarray(stored["colors"], dtype=np.float32))
    points = np.stack(object_points)
    colors = np.stack(object_colors)
    if points.shape[0] != 76 or colors.shape != points.shape:
        raise ValueError("fresh outcome point identities are inconsistent")

    with np.load(initial_archive, allow_pickle=False) as stored:
        sealed_initial = np.asarray(stored["frame_zero_points_m"], dtype=np.float32)
    if not np.array_equal(points[0], sealed_initial):
        raise ValueError("outcome frame-zero identities differ from the sealed input")

    robot = load_robot_state(episode_dir / "robot" / "robot.npz")
    controllers = np.stack(
        [_frame_controller_points(robot, frame) for frame in range(76)]
    ).astype(np.float32)
    valid = np.ones(points.shape[:2], dtype=bool)
    target = {
        "object_points": points,
        "object_colors": colors,
        "object_visibilities": valid,
        "object_motions_valid": valid.copy(),
        "controller_points": controllers,
        "surface_points": np.empty((0, 3), dtype=np.float32),
        "interior_points": np.empty((0, 3), dtype=np.float32),
    }
    args.output_data.parent.mkdir(parents=True, exist_ok=True)
    with args.output_data.open("wb") as stream:
        pickle.dump(target, stream, protocol=pickle.HIGHEST_PROTOCOL)

    payload = {
        "schema_version": 1,
        "artifact_kind": "Deform360ReusableTwinFreshOutcome",
        "protocol_id": protocol["parent"]["protocol_id"],
        "physics_addendum_id": protocol["addendum"]["protocol_id"],
        "role": role,
        "object_id": args.object_id,
        "episode_id": args.episode_id,
        "episode_key": f"{args.object_id}/{args.episode_id}",
        "future_access_seal_result_sha256": access_seal["result_sha256"],
        "camera_count": len(cameras),
        "point_count": int(points.shape[1]),
        "frame_count": int(points.shape[0]),
        "fusion": {
            "ransac_threshold_m_per_s": threshold,
            "minimum_camera_inlier_count": minimum_inliers,
            "rng_seed": 0,
        },
        "input_sha256": {
            "parent_lock": sha256_file(args.parent_lock),
            "physics_addendum": sha256_file(args.physics_addendum),
            "observation_consensus_lock": sha256_file(
                args.observation_consensus_lock
            ),
            "future_access_seal": sha256_file(args.future_access_seal),
            "source_manifest": sha256_file(manifest_path),
            "full_reconstruction": sha256_file(reconstruction_path),
            "tracking_checkpoint": sha256_file(args.checkpoint),
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
            "point_cloud": {path.name: sha256_file(path) for path in pcd_files},
            "target_data": sha256_file(args.output_data),
        },
        "implementation_revision": {
            "deform360_processing": _git_revision(args.deform360_repo)
        },
        "information_boundary": {
            "future_access_seal_verified_before_future_open": True,
            "fit_outcome_opened": args.operation == "fit",
            "held_outcome_opened": args.operation == "held-outcome",
            "future_tactile_read": False,
            "prediction_metric_computed": False,
        },
        "claim_boundary": (
            "fresh-panel outcome construction after the prospective access seal; "
            "no metric or state-of-the-art claim"
        ),
    }
    payload["result_sha256"] = _result_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
