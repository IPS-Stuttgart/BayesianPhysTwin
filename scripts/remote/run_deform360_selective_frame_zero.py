#!/usr/bin/env python3
"""Reconstruct and seal only frame zero of one prospective Deform360 case."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from bayesian_phystwin.deform360_selective_virtual_sensing_artifacts import (
    build_selective_backbone_seal,
    selective_case_records,
)
from bayesian_phystwin.deform360_selective_virtual_sensing_protocol import (
    PROTOCOL_ID,
    load_selective_virtual_sensing_protocol,
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _canonical_sha256(payload: dict[str, object]) -> str:
    unsigned = dict(payload)
    unsigned.pop("result_sha256", None)
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
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


def _case(protocol: Path, case_name: str) -> dict[str, object]:
    matches = [
        record for record in selective_case_records(protocol) if record["case"] == case_name
    ]
    _require(len(matches) == 1, "case is outside the locked prospective panel")
    return matches[0]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--deform360-repo", type=Path, required=True)
    parser.add_argument("--staged-case-dir", type=Path, required=True)
    parser.add_argument("--backbone-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    protocol = load_selective_virtual_sensing_protocol(args.protocol)
    staged_case = args.staged_case_dir.resolve()
    record = _case(args.protocol, staged_case.name)
    stage_manifest_path = staged_case / "prediction_prefix_manifest.json"
    stage_manifest = json.loads(stage_manifest_path.read_text(encoding="utf-8"))
    _require(
        stage_manifest.get("artifact_kind") == "Deform360SelectivePredictionPrefix"
        and stage_manifest.get("protocol_id") == PROTOCOL_ID
        and stage_manifest.get("protocol_config_sha256") == protocol["config_sha256"]
        and stage_manifest.get("result_sha256") == _canonical_sha256(stage_manifest),
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
    _require(_git_revision(deform360_repo) == DEFORM360_REVISION, "Deform360 revision changed")
    source_checks = {
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
    for name, (path, expected) in source_checks.items():
        _require(_sha256(path) == expected, f"Deform360 {name} source changed")

    frame_zero_root = staged_case / "frame-zero"
    frame_zero_episode = frame_zero_root / "episode_0000"
    prefix_episode = staged_case / "prefix" / "episode_0000"
    _require(frame_zero_episode.is_dir(), "frame-zero episode is missing")
    _require(prefix_episode.is_dir(), "prediction prefix episode is missing")
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
        frame_zero_root,
        0,
        cameras=cameras,
        overwrite=False,
    )
    _require(
        set(gripper_masks) == set(cameras),
        "frame-zero gripper-mask panel is incomplete",
    )
    depth_outputs = depth_stage.process_depth_episode(
        frame_zero_root,
        0,
        cameras=cameras,
        overwrite=False,
        preview=False,
    )
    _require(set(depth_outputs) == set(cameras), "frame-zero depth panel is incomplete")
    for camera in cameras:
        source_camera = frame_zero_episode / camera
        prefix_camera = prefix_episode / camera
        shutil.copy2(
            source_camera / "rendered_depth.h5",
            prefix_camera / "rendered_depth.h5",
        )
        shutil.copy2(
            source_camera / "rendered_depth.meta.json",
            prefix_camera / "rendered_depth.meta.json",
        )

    splat_path = Path(outputs[0]).resolve()
    points, colors = pcd_stage.seed_points_from_splat(
        splat_path,
        crop_half_extent_m=pcd_stage.CROP_HALF_EXTENT_M,
        seed_count=pcd_stage.SEED_POINT_COUNT,
        rng_seed=0,
    )
    _require(len(points) > 16 and np.all(np.isfinite(points)), "frame-zero point cloud failed")
    points_path = staged_case / "frame_zero_points.npz"
    np.savez_compressed(points_path, points_m=points, colors=colors)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "Deform360SelectiveFrameZeroReconstruction",
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
        "material_identity_sha256": _array_sha256(points),
        "inputs_sha256": {
            "prediction_prefix_manifest": _sha256(stage_manifest_path),
            **{name: _sha256(path) for name, (path, _) in source_checks.items()},
        },
        "outputs_sha256": {
            "frame_zero_splat": _sha256(splat_path),
            "frame_zero_points": _sha256(points_path),
            "depth_by_camera": {
                camera: _sha256(prefix_episode / camera / "rendered_depth.h5")
                for camera in cameras
            },
            "gripper_mask_by_camera": {
                camera: _sha256(frame_zero_episode / camera / "rendered_urdf.h5")
                for camera in cameras
            },
        },
        "information_boundary": {
            "object_observation_frames_used": [0],
            "future_dense_reconstruction_read": False,
            "future_particle_tracks_read": False,
            "target_metric_read": False,
        },
    }
    manifest["result_sha256"] = _canonical_sha256(manifest)
    reconstruction_manifest = staged_case / "frame_zero_reconstruction_manifest.json"
    reconstruction_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    backbone_dir = args.backbone_root.resolve() / str(record["case"])
    backbone = build_selective_backbone_seal(
        args.protocol,
        backbone_dir,
        object_id=str(record["object_id"]),
        episode_id=int(record["episode_id"]),
        frame_zero_points_m=points,
        frame_zero_reconstruction_manifest=reconstruction_manifest,
        prediction_stage_manifest=stage_manifest_path,
    )
    print(
        json.dumps(
            {
                "passed": True,
                "case": record["case"],
                "camera_count": len(cameras),
                "material_point_count": len(points),
                "frame_zero_result_sha256": manifest["result_sha256"],
                "backbone_result_sha256": backbone["result_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
