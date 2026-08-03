#!/usr/bin/env python3
"""Create frozen generic-SAM2 masks for one fresh source window."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from bayesian_phystwin.deform360_pairwise_regret_guard_fresh_processing import (
    MASK_KIND,
    RAW_FRAME_COUNT,
    SAM2_BASE_SOURCE_SHA256,
    SAM2_CHECKPOINT_SHA256,
    SAM2_COMMIT,
    SAM2_OBJECT_SOURCE_SHA256,
    WINDOW_STAGE_KIND,
    fresh_processing_case,
    seal_case_artifact,
    validate_case_artifact,
    validate_fresh_processing_sources,
    write_json_artifact,
)
from bayesian_phystwin.deform360_pairwise_regret_guard_fresh_protocol import (
    file_sha256,
)

MASK_FILENAME = "fresh_pairwise_masks.json"
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


def _write_masks(path: Path, masks: list[np.ndarray]) -> None:
    values = np.asarray(masks, dtype=np.uint8)
    _require(
        values.ndim == 3
        and values.shape[0] == RAW_FRAME_COUNT
        and np.all(np.count_nonzero(values, axis=(1, 2)) > 0),
        "SAM2 returned invalid or empty masks",
    )
    with h5py.File(path, "w") as stream:
        stream.create_dataset(
            "data",
            data=values,
            dtype=np.uint8,
            compression="gzip",
            compression_opts=4,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--technical-lock", type=Path, required=True)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--download-manifest", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--selector-source-root", type=Path, required=True)
    parser.add_argument("--sam2-repository", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    protocol, lock, _, _ = validate_fresh_processing_sources(
        args.protocol.resolve(),
        args.technical_lock.resolve(),
        args.source_plan.resolve(),
        args.download_manifest.resolve(),
    )
    case = fresh_processing_case(lock, args.object_id, args.episode_id)
    code_revision = _require_clean_repository(
        args.repo.resolve(), str(protocol["implementation_commit"])
    )
    source_episode = (
        args.stage_root.resolve() / args.object_id / f"episode_{args.episode_id:04d}"
    )
    stage_path = source_episode / STAGE_FILENAME
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    validate_case_artifact(
        stage,
        artifact_kind=WINDOW_STAGE_KIND,
        protocol=protocol,
        case=case,
    )
    _require(stage.get("status") == "staged", "source window is not staged")
    selector_root = args.selector_source_root.resolve()
    selectors = {
        "base": selector_root / "causal4d_public" / "deform360_sam2.py",
        "object": selector_root / "causal4d_public" / "deform360_object_sam2.py",
    }
    _require(
        file_sha256(selectors["base"]) == SAM2_BASE_SOURCE_SHA256
        and file_sha256(selectors["object"]) == SAM2_OBJECT_SOURCE_SHA256,
        "generic SAM2 selector source changed",
    )
    sam2_repository = args.sam2_repository.resolve()
    checkpoint = args.checkpoint.resolve()
    _require(_git_revision(sam2_repository) == SAM2_COMMIT, "SAM2 commit changed")
    _require(
        file_sha256(checkpoint) == SAM2_CHECKPOINT_SHA256,
        "SAM2 checkpoint changed",
    )
    sys.path.insert(0, str(selector_root))
    from causal4d_public.deform360_object_sam2 import (  # noqa: PLC0415
        DeformableObjectSam2VideoPredictor,
    )

    destination = (
        args.output_root.resolve() / args.object_id / f"episode_{args.episode_id:04d}"
    )
    _require(not destination.exists(), "fresh source masks already exist")
    scratch = destination.with_name(f".{destination.name}.incomplete-{os.getpid()}")
    _require(not scratch.exists(), "fresh source mask scratch already exists")
    scratch.mkdir(parents=True)
    predictor = DeformableObjectSam2VideoPredictor(
        sam2_repository, checkpoint, device=args.device
    )
    records: list[dict[str, Any]] = []
    try:
        for row in stage["camera_records"]:
            camera = str(row["camera"])
            video = source_episode / camera / "undistorted.mp4"
            _require(
                video.is_file() and file_sha256(video) == row["video_sha256"],
                f"staged video changed: {camera}",
            )
            output_camera = scratch / camera
            output_camera.mkdir()
            try:
                initial_mask, initialization = predictor.select_initial_mask(video)
                propagated = list(
                    predictor.segment_from_initial_mask(
                        video,
                        initial_mask,
                        initialization={
                            "policy": "frozen_generic_exact_frame_zero",
                            "source_frame_index": 0,
                            "future_object_observations_used": True,
                            "selection": initialization,
                        },
                    )
                )
                _require(
                    [index for index, _ in propagated] == list(range(RAW_FRAME_COUNT)),
                    f"SAM2 returned incomplete frames: {camera}",
                )
                masks = [np.asarray(mask, dtype=bool) for _, mask in propagated]
                mask_path = output_camera / "mask_refined.h5"
                _write_masks(mask_path, masks)
                areas = np.asarray(
                    [np.count_nonzero(mask) for mask in masks], dtype=np.int64
                )
                records.append(
                    {
                        "camera": camera,
                        "status": "success",
                        "input_video_sha256": row["video_sha256"],
                        "mask_sha256": file_sha256(mask_path),
                        "frame_count": len(masks),
                        "area_pixels_min": int(np.min(areas)),
                        "area_pixels_median": float(np.median(areas)),
                        "area_pixels_max": int(np.max(areas)),
                        "initialization": initialization,
                    }
                )
            except BaseException as exc:
                shutil.rmtree(output_camera, ignore_errors=True)
                records.append(
                    {
                        "camera": camera,
                        "status": "technical_failure",
                        "input_video_sha256": row["video_sha256"],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    finally:
        predictor.close()
    successful = sum(row["status"] == "success" for row in records)
    minimum = int(protocol["mask"]["minimum_successful_cameras"])
    status = "ready_for_processing" if successful >= minimum else "technical_failure"
    artifact = seal_case_artifact(
        MASK_KIND,
        protocol=protocol,
        case=case,
        payload={
            "status": status,
            "code_revision": code_revision,
            "window_stage_result_sha256": stage["result_sha256"],
            "window_stage_file_sha256": file_sha256(stage_path),
            "input_camera_count": len(records),
            "successful_camera_count": successful,
            "minimum_successful_cameras": minimum,
            "camera_records": records,
            "dependencies": {
                "selector_object_source_sha256": file_sha256(selectors["object"]),
                "selector_base_source_sha256": file_sha256(selectors["base"]),
                "sam2_commit": _git_revision(sam2_repository),
                "sam2_checkpoint_sha256": file_sha256(checkpoint),
            },
            "information_boundary": {
                "source_rgb_read": True,
                "all_81_frames_used_to_create_observation_assets": True,
                "manual_prompting_or_mask_selection": False,
                "object_geometry_read": False,
                "particle_tracks_read": False,
                "target_metric_read": False,
                "technical_failures_cause_no_implicit_replacement": True,
                "held_v8_runtime_or_target_artifact_access": False,
            },
        },
    )
    write_json_artifact(artifact, scratch / MASK_FILENAME)
    destination.parent.mkdir(parents=True, exist_ok=True)
    scratch.rename(destination)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if status == "ready_for_processing" else 2


if __name__ == "__main__":
    raise SystemExit(main())
