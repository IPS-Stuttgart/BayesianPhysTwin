#!/usr/bin/env python3
"""Construct one authorized Deform360 target without computing a metric."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping

import numpy as np

from bayesian_phystwin.deform360_online_belief_evaluation import _sha256
from bayesian_phystwin.deform360_selective_virtual_sensing_artifacts import (
    VIRTUAL_SENSING_ARCHIVE_FILENAME,
    authorize_selective_target_case,
)
from bayesian_phystwin.deform360_selective_virtual_sensing_protocol import (
    PROTOCOL_ID,
    load_selective_virtual_sensing_protocol,
)
from deform360.processing import (
    depth_stage,
    pcd_stage,
    reconstruct_stage,
    tracking_stage,
    urdf_render,
)


DEFORM360_REVISION = "0fe36f0b7a7a917ba62b5f8cee707299a9a4a317"
SOURCE_SHA256 = {
    "reconstruct_stage": "53a1e8b73e56a1c68a0c4344b279c2817ed4b3ed93e8f5ea792def26d5099c7c",
    "urdf_render": "c4d6a10e980ed4952f974d2e8a991c6fb819a3e6fdc6c121d3ce6925c94c2467",
    "depth_stage": "34befb732107b805f1e1924699f1e26fc2ca5d3041561b920d8c23d8e85feef0",
    "tracking_stage": "04533cd9cd900ae2f5bd139568ed1a2442661f14ceda009dd7bb85e4fbd83ec2",
    "pcd_stage": "87553e1ea3dac5a90e46114c76aaf65901b43a064025626ae6871523065c864d",
}
TRACKING_CHECKPOINT_SHA256 = (
    "2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834"
)
COTRACKER_REVISION = "82e02e8029753ad4ef13cf06be7f4fc5facdda4d"
COTRACKER_TREE = "f0296ad047b50c1530063b67e575908257478cab"
COTRACKER_PREDICTOR_SHA256 = (
    "783536d6c77790fa6c8e005f2df1d6bd4f0d8955c2c67d464bfb4e64d366375f"
)
MINIMUM_VISUAL_HULL_POINTS = 512
VOXEL_RESOLUTION = 120
CUBE_HALF_EXTENT_M = 0.5
FIRST_FRAME_ITERATIONS = 500
WARM_START_ITERATIONS = 250
RAW_FRAME_COUNT = 81
TARGET_FRAME_COUNT = 76
TARGET_ARCHIVE_FILENAME = "target_trajectory.npz"
OUTCOME_MANIFEST_FILENAME = "authorized_outcome_manifest.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("result_sha256", None)
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _git_revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_repository(repository: Path) -> str:
    revision = _git_revision(repository)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    _require(not status.strip(), f"repository has uncommitted files: {repository}")
    return revision


def _stage_metadata_sha256(episode: Path, cameras: list[str]) -> dict[str, Any]:
    return {
        "reconstruction": _sha256(episode / "splatfacto" / "splatfacto.meta.json"),
        "gripper_masks": {
            camera: _sha256(episode / camera / "rendered_urdf.meta.json")
            for camera in cameras
        },
        "depth": {
            camera: _sha256(episode / camera / "rendered_depth.meta.json")
            for camera in cameras
        },
        "tracking": {
            camera: _sha256(episode / camera / "tracking" / "tracking.meta.json")
            for camera in cameras
        },
        "point_cloud": _sha256(episode / "pcd_clean" / "pcd_clean.meta.json"),
    }


def _validate_runtime_constants() -> None:
    expected = {
        "pcd_seed_point_count": (pcd_stage.SEED_POINT_COUNT, 10000),
        "pcd_crop_half_extent_m": (pcd_stage.CROP_HALF_EXTENT_M, 0.5),
        "pcd_radius_neighbors": (pcd_stage.RADIUS_NEIGHBORS, 30),
        "pcd_radius_m": (pcd_stage.RADIUS_M, 0.02),
        "pcd_stat_neighbors": (pcd_stage.STAT_NEIGHBORS, 30),
        "pcd_stat_std_ratio": (pcd_stage.STAT_STD_RATIO, 3.5),
        "pcd_ransac_threshold": (pcd_stage.FUSE_RANSAC_THRESHOLD, 0.01),
        "pcd_ransac_min_inliers": (pcd_stage.FUSE_RANSAC_MIN_INLIERS, 4),
        "pcd_tail_frames": (pcd_stage.TAIL_FRAMES_SKIPPED, 5),
        "pcd_frame_rate": (pcd_stage.FRAME_RATE_HZ, 30.0),
        "tracking_pivot_skip": (tracking_stage.PIVOT_SKIP, 5),
        "tracking_sequence_length": (tracking_stage.SEQ_LEN, 15),
        "tracking_gap": (tracking_stage.GAP, 5),
        "tracking_grid_size": (tracking_stage.GRID_SIZE, 40),
        "tracking_resize_factor": (tracking_stage.RESIZE_FACTOR, 4),
    }
    changed = [name for name, (actual, frozen) in expected.items() if actual != frozen]
    _require(not changed, f"Deform360 runtime constants changed: {changed}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cohort-seal", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--failure-root", type=Path, required=True)
    parser.add_argument("--authorized-case-dir", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--deform360-repo", type=Path, required=True)
    parser.add_argument("--tracking-checkpoint", type=Path, required=True)
    parser.add_argument("--cotracker-repository", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    code_revision = _require_clean_repository(args.repo.resolve())
    protocol_path = args.protocol.resolve()
    protocol = load_selective_virtual_sensing_protocol(protocol_path)
    cohort_path = args.cohort_seal.resolve()
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    record, prediction_seal = authorize_selective_target_case(
        cohort,
        protocol_path=protocol_path,
        prediction_root=args.prediction_root,
        failure_root=args.failure_root,
        object_id=args.object_id,
        episode_id=args.episode_id,
    )

    authorized_case = args.authorized_case_dir.resolve()
    _require(authorized_case.name == record["case"], "authorized case name changed")
    reveal_manifest_path = authorized_case / "authorized_future_manifest.json"
    reveal = json.loads(reveal_manifest_path.read_text(encoding="utf-8"))
    _require(
        reveal.get("artifact_kind") == "Deform360SelectiveAuthorizedFuture"
        and reveal.get("protocol_id") == PROTOCOL_ID
        and reveal.get("protocol_config_sha256") == protocol["config_sha256"]
        and reveal.get("result_sha256") == _canonical_sha256(reveal)
        and all(reveal.get(key) == value for key, value in record.items()),
        "authorized-future manifest changed",
    )
    _require(
        reveal.get("authorization", {}).get("prediction_cohort_result_sha256")
        == cohort["result_sha256"]
        and reveal.get("authorization", {}).get("prediction_result_sha256")
        == prediction_seal["result_sha256"]
        and reveal.get("authorization", {}).get(
            "eligible_cohort_verified_before_future_read"
        )
        is True,
        "future was not opened under this cohort authorization",
    )
    cameras = [str(camera) for camera in reveal["selected_cameras"]]
    _require(
        len(cameras) == 8 and len(set(cameras)) == 8, "target camera panel changed"
    )

    deform360_repo = args.deform360_repo.resolve()
    _require(
        _git_revision(deform360_repo) == DEFORM360_REVISION,
        "Deform360 revision changed",
    )
    source_paths = {
        "reconstruct_stage": deform360_repo
        / "deform360"
        / "processing"
        / "reconstruct_stage.py",
        "urdf_render": deform360_repo / "deform360" / "processing" / "urdf_render.py",
        "depth_stage": deform360_repo / "deform360" / "processing" / "depth_stage.py",
        "tracking_stage": deform360_repo
        / "deform360"
        / "processing"
        / "tracking_stage.py",
        "pcd_stage": deform360_repo / "deform360" / "processing" / "pcd_stage.py",
    }
    for name, path in source_paths.items():
        _require(_sha256(path) == SOURCE_SHA256[name], f"official {name} changed")
    checkpoint = args.tracking_checkpoint.resolve()
    _require(
        _sha256(checkpoint) == TRACKING_CHECKPOINT_SHA256,
        "CoTracker checkpoint changed",
    )
    cotracker_repository = args.cotracker_repository.resolve()
    _require(
        _git_revision(cotracker_repository) == COTRACKER_REVISION
        and subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=cotracker_repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        == COTRACKER_TREE
        and _sha256(cotracker_repository / "cotracker" / "predictor.py")
        == COTRACKER_PREDICTOR_SHA256,
        "CoTracker source checkout changed",
    )
    sys.path.insert(0, str(cotracker_repository))
    _validate_runtime_constants()

    episode = authorized_case / "episode_0000"
    sealed_splat_sha = reveal["outputs_sha256"]["frame_zero_splat"]
    splat_zero = episode / "splatfacto" / "splat_0.ply"
    _require(_sha256(splat_zero) == sealed_splat_sha, "sealed frame-zero splat changed")
    original_visual_hull = reconstruct_stage.visual_hull_points

    def strict_visual_hull(*call_args: object, **call_kwargs: object):
        call_kwargs["min_points"] = MINIMUM_VISUAL_HULL_POINTS
        return original_visual_hull(*call_args, **call_kwargs)

    reconstruct_stage.visual_hull_points = strict_visual_hull
    try:
        splats = reconstruct_stage.process_reconstruction_episode(
            authorized_case,
            0,
            cameras=cameras,
            first_frame_iterations=FIRST_FRAME_ITERATIONS,
            warm_start_iterations=WARM_START_ITERATIONS,
            cube_half_extent_m=CUBE_HALF_EXTENT_M,
            voxel_resolution=VOXEL_RESOLUTION,
            overwrite=False,
            keep_scratch=False,
        )
    finally:
        reconstruct_stage.visual_hull_points = original_visual_hull
    _require(
        set(splats) == set(range(RAW_FRAME_COUNT)),
        "authorized reconstruction is incomplete",
    )
    _require(
        _sha256(Path(splats[0])) == sealed_splat_sha,
        "full reconstruction changed frame-zero splat",
    )

    gripper_masks = urdf_render.process_gripper_masks_episode(
        authorized_case, 0, cameras=cameras, overwrite=False
    )
    depths = depth_stage.process_depth_episode(
        authorized_case, 0, cameras=cameras, overwrite=False, preview=False
    )
    tracks = tracking_stage.process_tracking_episode(
        authorized_case,
        0,
        cameras=cameras,
        checkpoint=checkpoint,
        overwrite=False,
    )
    _require(
        set(gripper_masks) == set(depths) == set(tracks) == set(cameras),
        "authorized target stages have different camera panels",
    )
    pcd_dir = pcd_stage.process_pcd_episode(
        authorized_case,
        0,
        cameras=cameras,
        overwrite=False,
        rng_seed=0,
    )
    pcd_files = [pcd_dir / f"{frame:06d}.npz" for frame in range(TARGET_FRAME_COUNT)]
    _require(
        all(path.is_file() for path in pcd_files), "target point cloud is incomplete"
    )
    points = []
    for path in pcd_files:
        with np.load(path, allow_pickle=False) as stored:
            points.append(np.asarray(stored["pts"], dtype=np.float32))
    target = np.stack(points)
    _require(
        target.ndim == 3
        and target.shape[0] == TARGET_FRAME_COUNT
        and target.shape[2] == 3
        and np.all(np.isfinite(target)),
        "authorized target trajectory is invalid",
    )

    prediction_dir = args.prediction_root.resolve() / str(record["case"])
    prediction_archive = prediction_dir / VIRTUAL_SENSING_ARCHIVE_FILENAME
    with np.load(prediction_archive, allow_pickle=False) as stored:
        sealed_persistence = np.asarray(stored["persistence_m"], dtype=np.float32)
    _require(
        sealed_persistence.shape == target.shape,
        "target material point cardinality differs from the prediction",
    )
    _require(
        np.array_equal(target[0], sealed_persistence[0]),
        "target frame-zero identities differ from sealed persistence",
    )
    visibility = np.ones(target.shape[:2], dtype=bool)
    validity = np.ones(target.shape[:2], dtype=bool)

    destination = args.output_root.resolve() / str(record["case"])
    _require(
        not destination.exists(), f"authorized outcome already exists: {destination}"
    )
    scratch = destination.with_name(f".{destination.name}.incomplete-{os.getpid()}")
    _require(not scratch.exists(), f"authorized-outcome scratch exists: {scratch}")
    scratch.mkdir(parents=True)
    archive_path = scratch / TARGET_ARCHIVE_FILENAME
    try:
        np.savez_compressed(
            archive_path,
            target_m=target,
            target_visibility=visibility,
            target_validity=validity,
        )
        payload: dict[str, Any] = {
            "schema_version": 1,
            "artifact_kind": "Deform360SelectiveAuthorizedOutcome",
            "protocol_id": PROTOCOL_ID,
            "protocol_config_sha256": protocol["config_sha256"],
            **record,
            "code_revision": code_revision,
            "deform360_revision": DEFORM360_REVISION,
            "camera_count": len(cameras),
            "cameras": cameras,
            "raw_frame_count": RAW_FRAME_COUNT,
            "target_frame_count": TARGET_FRAME_COUNT,
            "material_point_count": int(target.shape[1]),
            "material_identity_sha256": _array_sha256(target[0]),
            "reconstruction": {
                "minimum_visual_hull_points": MINIMUM_VISUAL_HULL_POINTS,
                "voxel_resolution": VOXEL_RESOLUTION,
                "cube_half_extent_m": CUBE_HALF_EXTENT_M,
                "first_frame_iterations": FIRST_FRAME_ITERATIONS,
                "warm_start_iterations": WARM_START_ITERATIONS,
                "sealed_frame_zero_splat_reused": True,
            },
            "inputs_sha256": {
                "protocol": _sha256(protocol_path),
                "prediction_cohort_seal": _sha256(cohort_path),
                "prediction_seal": _sha256(
                    prediction_dir / "virtual_sensing_prediction_seal.json"
                ),
                "prediction_archive": _sha256(prediction_archive),
                "authorized_future_manifest": _sha256(reveal_manifest_path),
            "tracking_checkpoint": _sha256(checkpoint),
            "cotracker_predictor": _sha256(
                cotracker_repository / "cotracker" / "predictor.py"
            ),
                **{name: _sha256(path) for name, path in source_paths.items()},
            },
            "stage_metadata_sha256": _stage_metadata_sha256(episode, cameras),
            "output": {
                "target_archive": str(destination / TARGET_ARCHIVE_FILENAME),
                "target_archive_sha256": _sha256(archive_path),
                "target_array_sha256": _array_sha256(target),
                "frame_zero_bit_exact_to_sealed_persistence": True,
            },
            "authorization": {
                "prediction_cohort_result_sha256": cohort["result_sha256"],
                "prediction_result_sha256": prediction_seal["result_sha256"],
            },
            "information_boundary": {
                "eligible_prediction_cohort_verified_before_target_construction": True,
                "future_tactile_read": False,
                "prediction_metric_computed": False,
            },
        }
        payload["result_sha256"] = _canonical_sha256(payload)
        manifest_path = scratch / OUTCOME_MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(scratch, destination)
    except Exception:
        shutil.rmtree(scratch, ignore_errors=True)
        raise
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
