#!/usr/bin/env python3
"""Reconstruct and seal frame zero only for one fresh Deform360 case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from bayesian_phystwin.deform360_bias_aware_prospective_artifacts import (
    array_sha256,
    canonical_sha256,
    file_sha256,
    prospective_case_records,
)
from bayesian_phystwin.deform360_bias_aware_prospective_protocol import (
    PROTOCOL_ID,
    load_bias_aware_prospective_protocol,
)
from deform360.processing import depth_stage, pcd_stage, reconstruct_stage, urdf_render


DEFORM360_REVISION = "0fe36f0b7a7a917ba62b5f8cee707299a9a4a317"
RECONSTRUCT_SOURCE_SHA256 = (
    "53a1e8b73e56a1c68a0c4344b279c2817ed4b3ed93e8f5ea792def26d5099c7c"
)
DEPTH_SOURCE_SHA256 = (
    "34befb732107b805f1e1924699f1e26fc2ca5d3041561b920d8c23d8e85feef0"
)
PCD_SOURCE_SHA256 = (
    "87553e1ea3dac5a90e46114c76aaf65901b43a064025626ae6871523065c864d"
)
URDF_RENDER_SOURCE_SHA256 = (
    "c4d6a10e980ed4952f974d2e8a991c6fb819a3e6fdc6c121d3ce6925c94c2467"
)
MINIMUM_VISUAL_HULL_POINTS = 512
VOXEL_RESOLUTION = 120
FIRST_FRAME_ITERATIONS = 500
WARM_START_ITERATIONS = 250


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _git_revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--deform360-repo", type=Path, required=True)
    parser.add_argument("--staged-case-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    protocol = load_bias_aware_prospective_protocol(args.protocol)
    staged = args.staged_case_dir.resolve()
    matches = [row for row in prospective_case_records(args.protocol) if row["case"] == staged.name]
    _require(len(matches) == 1, "case is outside the prospective lock")
    record = matches[0]
    stage_manifest_path = staged / "prediction_prefix_manifest.json"
    stage_manifest = json.loads(stage_manifest_path.read_text(encoding="utf-8"))
    _require(
        stage_manifest.get("artifact_kind") == "Deform360BiasAwarePredictionPrefix"
        and stage_manifest.get("protocol_id") == PROTOCOL_ID
        and stage_manifest.get("protocol_config_sha256") == protocol["config_sha256"]
        and stage_manifest.get("result_sha256")
        == canonical_sha256(stage_manifest, digest_key="result_sha256"),
        "prediction-prefix manifest is incompatible",
    )
    _require(
        all(stage_manifest.get(key) == value for key, value in record.items()),
        "prediction-prefix case identity changed",
    )
    boundary = stage_manifest.get("information_boundary", {})
    _require(
        boundary.get("source_object_frames_after_prefix_read") is False
        and boundary.get("future_dense_reconstruction_read") is False
        and boundary.get("future_particle_tracks_read") is False
        and boundary.get("target_metric_read") is False,
        "prediction-prefix boundary changed",
    )
    deform360_repo = args.deform360_repo.resolve()
    _require(_git_revision(deform360_repo) == DEFORM360_REVISION, "Deform360 changed")
    sources = {
        "reconstruct_stage": (
            deform360_repo / "deform360" / "processing" / "reconstruct_stage.py",
            RECONSTRUCT_SOURCE_SHA256,
        ),
        "depth_stage": (
            deform360_repo / "deform360" / "processing" / "depth_stage.py",
            DEPTH_SOURCE_SHA256,
        ),
        "pcd_stage": (
            deform360_repo / "deform360" / "processing" / "pcd_stage.py",
            PCD_SOURCE_SHA256,
        ),
        "urdf_render": (
            deform360_repo / "deform360" / "processing" / "urdf_render.py",
            URDF_RENDER_SOURCE_SHA256,
        ),
    }
    for name, (path, expected) in sources.items():
        _require(file_sha256(path) == expected, f"Deform360 {name} changed")

    frame_zero_root = staged / "frame-zero"
    frame_zero_episode = frame_zero_root / "episode_0000"
    prefix_episode = staged / "prefix" / "episode_0000"
    cameras = sorted(
        path.name
        for path in frame_zero_episode.iterdir()
        if path.is_dir() and (path / "mask_refined.h5").is_file()
    )
    _require(len(cameras) >= 8, "too few frame-zero cameras")
    original_visual_hull = reconstruct_stage.visual_hull_points

    def strict_visual_hull(*call_args: object, **call_kwargs: object):
        call_kwargs["min_points"] = MINIMUM_VISUAL_HULL_POINTS
        return original_visual_hull(*call_args, **call_kwargs)

    reconstruct_stage.visual_hull_points = strict_visual_hull
    try:
        outputs = reconstruct_stage.process_reconstruction_episode(
            frame_zero_root,
            0,
            cameras=cameras,
            first_frame_iterations=FIRST_FRAME_ITERATIONS,
            warm_start_iterations=WARM_START_ITERATIONS,
            voxel_resolution=VOXEL_RESOLUTION,
            overwrite=False,
            keep_scratch=False,
        )
    finally:
        reconstruct_stage.visual_hull_points = original_visual_hull
    _require(set(outputs) == {0}, "frame-zero reconstruction opened another frame")
    gripper_masks = urdf_render.process_gripper_masks_episode(
        frame_zero_root, 0, cameras=cameras, overwrite=False
    )
    _require(set(gripper_masks) == set(cameras), "gripper-mask panel is incomplete")
    depth_outputs = depth_stage.process_depth_episode(
        frame_zero_root, 0, cameras=cameras, overwrite=False, preview=False
    )
    _require(set(depth_outputs) == set(cameras), "depth panel is incomplete")
    for camera in cameras:
        source_camera = frame_zero_episode / camera
        prefix_camera = prefix_episode / camera
        for name in ("rendered_depth.h5", "rendered_depth.meta.json"):
            shutil.copy2(source_camera / name, prefix_camera / name)

    splat_path = Path(outputs[0]).resolve()
    points, colors = pcd_stage.seed_points_from_splat(
        splat_path,
        crop_half_extent_m=pcd_stage.CROP_HALF_EXTENT_M,
        seed_count=pcd_stage.SEED_POINT_COUNT,
        rng_seed=0,
    )
    _require(len(points) >= 128 and np.all(np.isfinite(points)), "point cloud failed")
    points_path = staged / "frame_zero_points.npz"
    np.savez_compressed(points_path, points_m=points, colors=colors)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "Deform360BiasAwareFrameZeroReconstruction",
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": protocol["config_sha256"],
        **record,
        "deform360_revision": DEFORM360_REVISION,
        "cameras": cameras,
        "camera_count": len(cameras),
        "minimum_visual_hull_points": MINIMUM_VISUAL_HULL_POINTS,
        "voxel_resolution": VOXEL_RESOLUTION,
        "first_frame_iterations": FIRST_FRAME_ITERATIONS,
        "warm_start_iterations": WARM_START_ITERATIONS,
        "material_point_count": len(points),
        "material_identity_sha256": array_sha256(points),
        "inputs_sha256": {
            "prediction_prefix_manifest": file_sha256(stage_manifest_path),
            **{name: file_sha256(path) for name, (path, _) in sources.items()},
        },
        "outputs_sha256": {
            "frame_zero_splat": file_sha256(splat_path),
            "frame_zero_points": file_sha256(points_path),
            "depth_by_camera": {
                camera: file_sha256(prefix_episode / camera / "rendered_depth.h5")
                for camera in cameras
            },
            "gripper_mask_by_camera": {
                camera: file_sha256(frame_zero_episode / camera / "rendered_urdf.h5")
                for camera in cameras
            },
        },
        "information_boundary": {
            "object_observation_frames_used": [0],
            "future_object_rgb_read": False,
            "future_dense_reconstruction_read": False,
            "future_particle_tracks_read": False,
            "target_metric_read": False,
        },
    }
    manifest["result_sha256"] = canonical_sha256(
        manifest, digest_key="result_sha256"
    )
    manifest_path = staged / "frame_zero_reconstruction_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
