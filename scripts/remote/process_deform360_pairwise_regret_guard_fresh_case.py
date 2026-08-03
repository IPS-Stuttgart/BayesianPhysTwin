#!/usr/bin/env python3
"""Run the frozen official source pipeline for one fresh technical case."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from bayesian_phystwin.deform360_pairwise_regret_guard_fresh_processing import (
    COTRACKER_CHECKPOINT_SHA256,
    COTRACKER_PREDICTOR_SHA256,
    COTRACKER_REVISION,
    COTRACKER_TREE,
    DEFORM360_REVISION,
    DEFORM360_SOURCE_SHA256,
    MASK_KIND,
    PREDICTION_FRAME_COUNT,
    PROCESSING_KIND,
    RAW_FRAME_COUNT,
    WINDOW_STAGE_KIND,
    build_fresh_source_admission,
    fresh_processing_case,
    seal_case_artifact,
    validate_case_artifact,
    validate_fresh_processing_sources,
    validate_fresh_source_admission,
    write_json_artifact,
)
from bayesian_phystwin.deform360_pairwise_regret_guard_fresh_protocol import (
    file_sha256,
)

MASK_FILENAME = "fresh_pairwise_masks.json"
PROCESSING_FILENAME = "fresh_pairwise_processing.json"
ADMISSION_FILENAME = "fresh_pairwise_admission.json"
STAGE_FILENAME = "fresh_pairwise_window_stage.json"


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


def _git_tree(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_repository(repository: Path, expected: str) -> str:
    revision = _git_revision(repository)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    _require(not status.strip(), f"repository has uncommitted files: {repository}")
    _require(revision == expected, "processing implementation revision changed")
    return revision


def _source_paths(repository: Path) -> dict[str, Path]:
    processing = repository / "deform360" / "processing"
    return {
        "reconstruct_stage": processing / "reconstruct_stage.py",
        "urdf_render": processing / "urdf_render.py",
        "depth_stage": processing / "depth_stage.py",
        "tracking_stage": processing / "tracking_stage.py",
        "pcd_stage": processing / "pcd_stage.py",
        "control_points_stage": processing / "control_points_stage.py",
    }


def _validate_runtime_constants(*, pcd_stage: Any, tracking_stage: Any) -> None:
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


def _validate_mask_artifact(
    manifest: dict[str, Any],
    *,
    protocol: dict[str, Any],
    case: dict[str, Any],
    mask_episode: Path,
) -> tuple[str, ...]:
    validate_case_artifact(
        manifest,
        artifact_kind=MASK_KIND,
        protocol=protocol,
        case=case,
    )
    _require(
        manifest.get("status") == "ready_for_processing",
        "mask artifact is not processing-ready",
    )
    rows = manifest.get("camera_records")
    expected = tuple(protocol["dataset"]["camera_panel"])
    _require(
        isinstance(rows, list) and tuple(row.get("camera") for row in rows) == expected,
        "mask camera panel changed",
    )
    successful = tuple(
        sorted(str(row["camera"]) for row in rows if row.get("status") == "success")
    )
    _require(
        len(successful) >= int(protocol["mask"]["minimum_successful_cameras"]),
        "mask support is below the frozen minimum",
    )
    by_camera = {str(row["camera"]): row for row in rows}
    for camera in successful:
        path = mask_episode / camera / "mask_refined.h5"
        _require(
            path.is_file()
            and file_sha256(path) == by_camera[camera]["mask_sha256"]
            and by_camera[camera]["frame_count"] == RAW_FRAME_COUNT,
            f"frozen source mask changed: {camera}",
        )
    return successful


def _stage_metadata_sha256(episode: Path, cameras: tuple[str, ...]) -> dict[str, Any]:
    return {
        "reconstruction": file_sha256(episode / "splatfacto" / "splatfacto.meta.json"),
        "gripper_masks": {
            camera: file_sha256(episode / camera / "rendered_urdf.meta.json")
            for camera in cameras
        },
        "depth": {
            camera: file_sha256(episode / camera / "rendered_depth.meta.json")
            for camera in cameras
        },
        "tracking": {
            camera: file_sha256(episode / camera / "tracking" / "tracking.meta.json")
            for camera in cameras
        },
        "point_cloud": file_sha256(episode / "pcd_clean" / "pcd_clean.meta.json"),
        "control_points": file_sha256(episode / "control_points.meta.json"),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--technical-lock", type=Path, required=True)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--download-manifest", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--admission-root", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--deform360-repo", type=Path, required=True)
    parser.add_argument("--tracking-checkpoint", type=Path, required=True)
    parser.add_argument("--cotracker-repository", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    protocol, lock, _, download = validate_fresh_processing_sources(
        args.protocol.resolve(),
        args.technical_lock.resolve(),
        args.source_plan.resolve(),
        args.download_manifest.resolve(),
    )
    case = fresh_processing_case(lock, args.object_id, args.episode_id)
    code_revision = _require_clean_repository(
        args.repo.resolve(), str(protocol["implementation_commit"])
    )
    stage_episode = (
        args.stage_root.resolve() / args.object_id / f"episode_{args.episode_id:04d}"
    )
    stage_path = stage_episode / STAGE_FILENAME
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    validate_case_artifact(
        stage,
        artifact_kind=WINDOW_STAGE_KIND,
        protocol=protocol,
        case=case,
    )
    mask_episode = (
        args.mask_root.resolve() / args.object_id / f"episode_{args.episode_id:04d}"
    )
    mask_path = mask_episode / MASK_FILENAME
    mask = json.loads(mask_path.read_text(encoding="utf-8"))
    cameras = _validate_mask_artifact(
        mask, protocol=protocol, case=case, mask_episode=mask_episode
    )
    metadata = args.metadata_root.resolve() / args.object_id / "metadata.json"
    metadata_record = next(
        row
        for row in download["files"]
        if row["path"] == f"raw/{args.object_id}/metadata.json"
    )
    _require(
        metadata.is_file() and file_sha256(metadata) == metadata_record["sha256"],
        "source metadata changed",
    )
    deform360_repository = args.deform360_repo.resolve()
    _require(
        _git_revision(deform360_repository) == DEFORM360_REVISION,
        "Deform360 revision changed",
    )
    source_paths = _source_paths(deform360_repository)
    _require(
        all(
            path.is_file() and file_sha256(path) == DEFORM360_SOURCE_SHA256[name]
            for name, path in source_paths.items()
        ),
        "Deform360 processing source changed",
    )
    cotracker_repository = args.cotracker_repository.resolve()
    checkpoint = args.tracking_checkpoint.resolve()
    _require(
        _git_revision(cotracker_repository) == COTRACKER_REVISION
        and _git_tree(cotracker_repository) == COTRACKER_TREE
        and file_sha256(cotracker_repository / "cotracker" / "predictor.py")
        == COTRACKER_PREDICTOR_SHA256
        and file_sha256(checkpoint) == COTRACKER_CHECKPOINT_SHA256,
        "CoTracker dependency changed",
    )
    sys.path.insert(0, str(cotracker_repository))
    from deform360.processing import (  # noqa: PLC0415
        control_points_stage,
        depth_stage,
        pcd_stage,
        reconstruct_stage,
        tracking_stage,
        urdf_render,
    )

    _validate_runtime_constants(pcd_stage=pcd_stage, tracking_stage=tracking_stage)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "preflight_passed",
                    **case,
                    "code_revision": code_revision,
                    "protocol_sha256": protocol["protocol_sha256"],
                    "window_stage_result_sha256": stage["result_sha256"],
                    "mask_result_sha256": mask["result_sha256"],
                    "camera_count": len(cameras),
                    "cameras": list(cameras),
                    "target_metric_read": False,
                    "held_v8_access": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    destination = (
        args.output_root.resolve() / args.object_id / f"episode_{args.episode_id:04d}"
    )
    admission_path = (
        args.admission_root.resolve()
        / f"{args.object_id}-ep{args.episode_id:04d}.admission.json"
    )
    _require(not destination.exists(), "source processing already exists")
    _require(not admission_path.exists(), "source admission already exists")
    scratch = args.output_root.resolve() / f".scratch-{case['case']}-{os.getpid()}"
    _require(not scratch.exists(), "source processing scratch already exists")
    scratch.mkdir(parents=True)
    episode = scratch / f"episode_{args.episode_id:04d}"
    shutil.copytree(stage_episode, episode)
    shutil.copy2(metadata, scratch / "metadata.json")
    for camera in cameras:
        shutil.copy2(
            mask_episode / camera / "mask_refined.h5",
            episode / camera / "mask_refined.h5",
        )
    shutil.copy2(mask_path, episode / MASK_FILENAME)
    inputs_sha256 = {
        "protocol": file_sha256(args.protocol),
        "window_stage": file_sha256(stage_path),
        "mask_manifest": file_sha256(mask_path),
        "metadata": file_sha256(metadata),
        "tracking_checkpoint": file_sha256(checkpoint),
        **{name: file_sha256(path) for name, path in source_paths.items()},
    }
    try:
        reconstruction = protocol["processing"]
        original_visual_hull = reconstruct_stage.visual_hull_points

        def strict_visual_hull(*call_args: object, **call_kwargs: object) -> Any:
            call_kwargs["min_points"] = int(
                reconstruction["minimum_visual_hull_points"]
            )
            return original_visual_hull(*call_args, **call_kwargs)

        reconstruct_stage.visual_hull_points = strict_visual_hull
        try:
            splats = reconstruct_stage.process_reconstruction_episode(
                scratch,
                args.episode_id,
                cameras=cameras,
                first_frame_iterations=int(reconstruction["first_frame_iterations"]),
                warm_start_iterations=int(reconstruction["warm_start_iterations"]),
                cube_half_extent_m=float(reconstruction["cube_half_extent_m"]),
                voxel_resolution=int(reconstruction["voxel_resolution"]),
                overwrite=True,
                keep_scratch=False,
            )
        finally:
            reconstruct_stage.visual_hull_points = original_visual_hull
        _require(
            set(splats) == set(range(RAW_FRAME_COUNT)), "incomplete reconstruction"
        )
        gripper_masks = urdf_render.process_gripper_masks_episode(
            scratch, args.episode_id, cameras=cameras, overwrite=True
        )
        depths = depth_stage.process_depth_episode(
            scratch,
            args.episode_id,
            cameras=cameras,
            overwrite=True,
            preview=False,
        )
        tracks = tracking_stage.process_tracking_episode(
            scratch,
            args.episode_id,
            cameras=cameras,
            checkpoint=checkpoint,
            overwrite=True,
        )
        _require(
            set(gripper_masks) == set(depths) == set(tracks) == set(cameras),
            "source stages used different camera panels",
        )
        pcd_dir = pcd_stage.process_pcd_episode(
            scratch, args.episode_id, cameras=cameras, overwrite=True, rng_seed=0
        )
        _require(
            all(
                (pcd_dir / f"{frame:06d}.npz").is_file()
                for frame in range(PREDICTION_FRAME_COUNT)
            ),
            "point-cloud trajectory is incomplete",
        )
        outputs = control_points_stage.process_control_points_episode(
            scratch, args.episode_id, cameras=cameras, overwrite=True
        )
        admission = build_fresh_source_admission(
            episode, metadata, protocol=protocol, case=case
        )
        validate_fresh_source_admission(admission, protocol=protocol, case=case)
        write_json_artifact(admission, episode / ADMISSION_FILENAME)
        status = "admitted" if admission["accepted"] else "source_rejected"
        artifact = seal_case_artifact(
            PROCESSING_KIND,
            protocol=protocol,
            case=case,
            payload={
                "status": status,
                "code_revision": code_revision,
                "cameras": list(cameras),
                "camera_count": len(cameras),
                "raw_frame_count": RAW_FRAME_COUNT,
                "output_frame_count": PREDICTION_FRAME_COUNT,
                "mask_result_sha256": mask["result_sha256"],
                "inputs_sha256": inputs_sha256,
                "stage_metadata_sha256": _stage_metadata_sha256(episode, cameras),
                "outputs_sha256": {
                    name: file_sha256(path) for name, path in outputs.items()
                },
                "admission_sha256": admission["result_sha256"],
                "admission_accepted": admission["accepted"],
                "admission_rejection_reasons": admission["rejection_reasons"],
                "information_boundary": {
                    "source_rgb_and_masks_read": True,
                    "future_geometry_deserialized_for_admission": False,
                    "target_metric_read": False,
                    "technical_failure_causes_no_implicit_replacement": True,
                    "held_v8_runtime_or_target_artifact_access": False,
                },
            },
        )
        write_json_artifact(artifact, episode / PROCESSING_FILENAME)
        destination.parent.mkdir(parents=True, exist_ok=True)
        episode.rename(destination)
        shutil.rmtree(scratch, ignore_errors=True)
        admission_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(destination / ADMISSION_FILENAME, admission_path)
        print(json.dumps(artifact, indent=2, sort_keys=True))
        return 0 if admission["accepted"] else 3
    except Exception as exc:
        failure = seal_case_artifact(
            PROCESSING_KIND,
            protocol=protocol,
            case=case,
            payload={
                "status": "technical_failure",
                "code_revision": code_revision,
                "cameras": list(cameras),
                "camera_count": len(cameras),
                "inputs_sha256": inputs_sha256,
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "information_boundary": {
                    "source_rgb_and_masks_read": True,
                    "future_geometry_deserialized_for_admission": False,
                    "target_metric_read": False,
                    "technical_failure_causes_no_implicit_replacement": True,
                    "held_v8_runtime_or_target_artifact_access": False,
                },
            },
        )
        write_json_artifact(failure, episode / PROCESSING_FILENAME)
        destination.parent.mkdir(parents=True, exist_ok=True)
        episode.rename(destination)
        shutil.rmtree(scratch, ignore_errors=True)
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
