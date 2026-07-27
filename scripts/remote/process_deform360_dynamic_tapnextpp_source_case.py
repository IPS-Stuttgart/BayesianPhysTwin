#!/usr/bin/env python3
"""Run official source processing for one dynamic TAPNext++ case."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bayesian_phystwin.deform360_fresh_source_lock import (
    FreshSourceAdmissionConfig,
    build_fresh_source_admission,
    write_fresh_source_artifact,
)
from bayesian_phystwin.deform360_dynamic_tapnextpp_source_processing import (
    COTRACKER_CHECKPOINT_SHA256,
    COTRACKER_PREDICTOR_SHA256,
    COTRACKER_REVISION,
    COTRACKER_TREE,
    DEFORM360_REVISION,
    DEFORM360_SOURCE_SHA256,
    PROCESSING_ARTIFACT_KIND,
    load_dynamic_source_processing_protocol,
    validate_dynamic_source_mask_artifact,
)
from bayesian_phystwin.deform360_dynamic_tapnextpp_source_window import (
    PREDICTION_FRAME_COUNT,
    RAW_FRAME_COUNT,
    canonical_sha256,
    dynamic_source_case,
    file_sha256,
    load_dynamic_source_mask_protocol,
    validate_dynamic_source_window_stage,
    validate_dynamic_window_sources,
)

MASK_MANIFEST_FILENAME = "dynamic_tapnextpp_source_masks.json"
PROCESSING_MANIFEST_FILENAME = "dynamic_tapnextpp_source_processing.json"
ADMISSION_FILENAME = "dynamic_tapnextpp_source_admission.json"
WINDOW_STAGE_FILENAME = "dynamic_tapnextpp_source_window_stage.json"


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--processing-protocol", type=Path, required=True)
    parser.add_argument("--window-protocol", type=Path, required=True)
    parser.add_argument("--mask-protocol", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
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


def _source_paths(deform360_repository: Path) -> dict[str, Path]:
    processing = deform360_repository / "deform360" / "processing"
    return {
        "reconstruct_stage": processing / "reconstruct_stage.py",
        "urdf_render": processing / "urdf_render.py",
        "depth_stage": processing / "depth_stage.py",
        "tracking_stage": processing / "tracking_stage.py",
        "pcd_stage": processing / "pcd_stage.py",
        "control_points_stage": processing / "control_points_stage.py",
    }


def _validate_runtime_constants(
    *,
    pcd_stage: Any,
    tracking_stage: Any,
) -> None:
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


def _download_record(
    download: Mapping[str, Any],
    *,
    object_id: str,
    episode_id: int,
) -> Mapping[str, Any]:
    rows = [
        row
        for row in download.get("objects", ())
        if isinstance(row, Mapping)
        and row.get("object_id") == object_id
        and row.get("episode_id") == episode_id
    ]
    _require(len(rows) == 1, "case is absent from the frozen download manifest")
    return rows[0]


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


def _failure_manifest(
    *,
    protocol: Mapping[str, Any],
    case: Mapping[str, Any],
    code_revision: str,
    cameras: tuple[str, ...],
    inputs: Mapping[str, Any],
    error: BaseException,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": PROCESSING_ARTIFACT_KIND,
        "protocol_id": protocol["protocol_id"],
        "protocol_config_sha256": protocol["config_sha256"],
        **case,
        "status": "technical_failure",
        "code_revision": code_revision,
        "cameras": list(cameras),
        "camera_count": len(cameras),
        "inputs_sha256": dict(inputs),
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
        "information_boundary": {
            "source_rgb_and_masks_read": True,
            "future_geometry_deserialized_for_admission": False,
            "target_metric_read": False,
            "held_v8_target_query_score_barrier_or_outcome_access": False,
            "technical_failure_causes_no_implicit_replacement": True,
        },
    }
    payload["result_sha256"] = canonical_sha256(payload, digest_key="result_sha256")
    return payload


def main() -> int:
    args = _parse_args()
    repository = args.repo.resolve()
    code_revision = _require_clean_repository(repository)
    processing_protocol_path = args.processing_protocol.resolve()
    processing_protocol = load_dynamic_source_processing_protocol(
        processing_protocol_path
    )
    window_protocol, queue, download = validate_dynamic_window_sources(
        args.window_protocol.resolve(),
        args.queue.resolve(),
        args.download_manifest.resolve(),
    )
    mask_protocol_path = args.mask_protocol.resolve()
    mask_protocol = load_dynamic_source_mask_protocol(mask_protocol_path)
    parent = processing_protocol["parent_mask_protocol"]
    _require(
        mask_protocol["config_sha256"] == parent["config_sha256"]
        and file_sha256(mask_protocol_path) == parent["file_sha256"],
        "processing protocol binds another mask protocol",
    )
    case = dynamic_source_case(queue, args.object_id, args.episode_id)
    download_record = _download_record(
        download,
        object_id=args.object_id,
        episode_id=args.episode_id,
    )

    stage_object = args.stage_root.resolve() / args.object_id
    stage_episode = stage_object / f"episode_{args.episode_id:04d}"
    stage_path = stage_episode / WINDOW_STAGE_FILENAME
    stage = validate_dynamic_source_window_stage(
        stage_path,
        window_protocol=window_protocol,
        case=case,
        expected_code_revision=mask_protocol["parent_window_protocol"][
            "implementation_commit"
        ],
    )
    mask_episode = (
        args.mask_root.resolve() / args.object_id / f"episode_{args.episode_id:04d}"
    )
    mask_manifest_path = mask_episode / MASK_MANIFEST_FILENAME
    mask_manifest, cameras = validate_dynamic_source_mask_artifact(
        mask_manifest_path,
        mask_protocol=mask_protocol,
        case=case,
        mask_episode_dir=mask_episode,
        expected_code_revision=processing_protocol["parent_mask_protocol"][
            "implementation_commit"
        ],
    )
    _require(
        len(cameras)
        >= int(processing_protocol["camera_policy"]["minimum_camera_count"]),
        "processing camera panel is below the frozen minimum",
    )

    metadata = args.metadata_root.resolve() / args.object_id / "metadata.json"
    _require(
        metadata.is_file()
        and file_sha256(metadata) == download_record["metadata_sha256"],
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
    from deform360.processing import (
        control_points_stage,
        depth_stage,
        pcd_stage,
        reconstruct_stage,
        tracking_stage,
        urdf_render,
    )

    _validate_runtime_constants(
        pcd_stage=pcd_stage,
        tracking_stage=tracking_stage,
    )
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "preflight_passed",
                    **case,
                    "code_revision": code_revision,
                    "processing_protocol_config_sha256": processing_protocol[
                        "config_sha256"
                    ],
                    "window_stage_result_sha256": stage["result_sha256"],
                    "mask_result_sha256": mask_manifest["result_sha256"],
                    "camera_count": len(cameras),
                    "cameras": list(cameras),
                    "metadata_sha256": file_sha256(metadata),
                    "target_metric_read": False,
                    "held_v8_access": False,
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0

    destination = args.output_root.resolve() / args.object_id
    admission_path = (
        args.admission_root.resolve()
        / f"{args.object_id}-ep{args.episode_id:04d}.admission.json"
    )
    _require(not destination.exists(), f"source processing exists: {destination}")
    _require(not admission_path.exists(), f"source admission exists: {admission_path}")
    scratch = destination.with_name(f".{destination.name}.incomplete-{os.getpid()}")
    _require(not scratch.exists(), "source-processing scratch exists")
    shutil.copytree(stage_object, scratch)
    shutil.copy2(metadata, scratch / "metadata.json")
    episode = scratch / f"episode_{args.episode_id:04d}"
    for camera in cameras:
        shutil.copy2(
            mask_episode / camera / "mask_refined.h5",
            episode / camera / "mask_refined.h5",
        )
    shutil.copy2(mask_manifest_path, episode / MASK_MANIFEST_FILENAME)

    inputs_sha256 = {
        "processing_protocol": file_sha256(processing_protocol_path),
        "window_stage": file_sha256(stage_path),
        "mask_protocol": file_sha256(mask_protocol_path),
        "mask_manifest": file_sha256(mask_manifest_path),
        "metadata": file_sha256(metadata),
        "tracking_checkpoint": file_sha256(checkpoint),
        **{name: file_sha256(path) for name, path in source_paths.items()},
    }
    reconstruction = processing_protocol["reconstruction"]
    try:
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
            set(splats) == set(range(RAW_FRAME_COUNT)),
            "source reconstruction is incomplete",
        )
        gripper_masks = urdf_render.process_gripper_masks_episode(
            scratch,
            args.episode_id,
            cameras=cameras,
            overwrite=True,
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
            scratch,
            args.episode_id,
            cameras=cameras,
            overwrite=True,
            rng_seed=0,
        )
        _require(
            all(
                (pcd_dir / f"{frame:06d}.npz").is_file()
                for frame in range(PREDICTION_FRAME_COUNT)
            ),
            "source point-cloud trajectory is incomplete",
        )
        outputs = control_points_stage.process_control_points_episode(
            scratch,
            args.episode_id,
            cameras=cameras,
            overwrite=True,
        )
        admission = build_fresh_source_admission(
            episode,
            metadata,
            object_id=args.object_id,
            episode_id=args.episode_id,
            category=str(case["category"]),
            config=FreshSourceAdmissionConfig(
                minimum_camera_count=int(
                    processing_protocol["source_admission"]["minimum_camera_count"]
                ),
                minimum_point_count=int(
                    processing_protocol["source_admission"]["minimum_point_count"]
                ),
                maximum_point_count=int(
                    processing_protocol["source_admission"]["maximum_point_count"]
                ),
                required_frame_count=int(
                    processing_protocol["source_admission"]["required_frame_count"]
                ),
                update_frames=tuple(
                    int(value)
                    for value in processing_protocol["source_admission"][
                        "update_frames"
                    ]
                ),
                minimum_test_frame_count=int(
                    processing_protocol["source_admission"]["minimum_test_frame_count"]
                ),
            ),
        )
        write_fresh_source_artifact(admission, episode / ADMISSION_FILENAME)
        status = "admitted" if admission["accepted"] else "source_rejected"
        payload: dict[str, Any] = {
            "schema_version": 1,
            "artifact_kind": PROCESSING_ARTIFACT_KIND,
            "protocol_id": processing_protocol["protocol_id"],
            "protocol_config_sha256": processing_protocol["config_sha256"],
            **case,
            "status": status,
            "code_revision": code_revision,
            "cameras": list(cameras),
            "camera_count": len(cameras),
            "raw_frame_count": RAW_FRAME_COUNT,
            "output_frame_count": PREDICTION_FRAME_COUNT,
            "mask_result_sha256": mask_manifest["result_sha256"],
            "inputs_sha256": inputs_sha256,
            "stage_metadata_sha256": _stage_metadata_sha256(episode, cameras),
            "outputs_sha256": {
                name: file_sha256(path) for name, path in outputs.items()
            },
            "admission_sha256": admission["admission_sha256"],
            "admission_accepted": admission["accepted"],
            "admission_rejection_reasons": admission["rejection_reasons"],
            "information_boundary": {
                "source_rgb_and_masks_read": True,
                "future_geometry_deserialized_for_admission": False,
                "target_metric_read": False,
                "held_v8_target_query_score_barrier_or_outcome_access": False,
                "sealed_window_and_mask_artifacts_mutated": False,
                "technical_failure_causes_no_implicit_replacement": True,
            },
        }
        payload["result_sha256"] = canonical_sha256(payload, digest_key="result_sha256")
        (episode / PROCESSING_MANIFEST_FILENAME).write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        scratch.rename(destination)
        admission_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            destination / f"episode_{args.episode_id:04d}" / ADMISSION_FILENAME,
            admission_path,
        )
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0 if admission["accepted"] else 3
    except Exception as exc:
        if not scratch.exists():
            raise
        failure = _failure_manifest(
            protocol=processing_protocol,
            case=case,
            code_revision=code_revision,
            cameras=cameras,
            inputs=inputs_sha256,
            error=exc,
        )
        (episode / PROCESSING_MANIFEST_FILENAME).write_text(
            json.dumps(failure, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        scratch.rename(destination)
        print(json.dumps(failure, indent=2, sort_keys=True, allow_nan=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
