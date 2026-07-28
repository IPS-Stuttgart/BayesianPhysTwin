#!/usr/bin/env python3
"""Create generic-SAM2 masks for a sealed dynamic TAPNext++ source window."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import h5py
import numpy as np

from bayesian_phystwin.deform360_dynamic_tapnextpp_source_window import (
    FROZEN_BASE_SAM2_SOURCE_SHA256,
    FROZEN_CAMERA_PANEL,
    FROZEN_OBJECT_SAM2_SOURCE_SHA256,
    FROZEN_SAM2_CHECKPOINT_SHA256,
    FROZEN_SAM2_COMMIT,
    MASK_ARTIFACT_KIND,
    RAW_FRAME_COUNT,
    canonical_sha256,
    dynamic_source_case,
    file_sha256,
    load_dynamic_source_mask_protocol,
    validate_dynamic_source_window_stage,
    validate_dynamic_window_sources,
)


MASK_MANIFEST_FILENAME = "dynamic_tapnextpp_source_masks.json"
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
    parser.add_argument("--window-protocol", type=Path, required=True)
    parser.add_argument("--mask-protocol", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
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
    code_revision = _require_clean_repository(args.repo.resolve())
    window_protocol, queue, _ = validate_dynamic_window_sources(
        args.window_protocol.resolve(),
        args.queue.resolve(),
        args.download_manifest.resolve(),
    )
    mask_protocol = load_dynamic_source_mask_protocol(args.mask_protocol.resolve())
    _require(
        mask_protocol["parent_window_protocol"]["config_sha256"]
        == window_protocol["config_sha256"]
        and file_sha256(args.window_protocol.resolve())
        == mask_protocol["parent_window_protocol"]["file_sha256"],
        "mask protocol binds another window protocol",
    )
    case = dynamic_source_case(queue, args.object_id, args.episode_id)
    source_episode = (
        args.stage_root.resolve() / args.object_id / f"episode_{args.episode_id:04d}"
    )
    stage_path = source_episode / WINDOW_STAGE_FILENAME
    stage = validate_dynamic_source_window_stage(
        stage_path,
        window_protocol=window_protocol,
        case=case,
        expected_code_revision=mask_protocol["parent_window_protocol"][
            "implementation_commit"
        ],
    )

    selector_root = args.selector_source_root.resolve()
    selector_sources = {
        "base": selector_root / "causal4d_public" / "deform360_sam2.py",
        "object": (selector_root / "causal4d_public" / "deform360_object_sam2.py"),
    }
    _require(
        file_sha256(selector_sources["base"]) == FROZEN_BASE_SAM2_SOURCE_SHA256
        and file_sha256(selector_sources["object"]) == FROZEN_OBJECT_SAM2_SOURCE_SHA256,
        "generic SAM2 selector source changed",
    )
    sam2_repository = args.sam2_repository.resolve()
    checkpoint = args.checkpoint.resolve()
    _require(
        _git_revision(sam2_repository) == FROZEN_SAM2_COMMIT,
        "SAM2 repository changed",
    )
    _require(
        file_sha256(checkpoint) == FROZEN_SAM2_CHECKPOINT_SHA256,
        "SAM2 checkpoint changed",
    )
    sys.path.insert(0, str(selector_root))
    from causal4d_public.deform360_object_sam2 import (  # noqa: PLC0415
        DeformableObjectSam2VideoPredictor,
    )

    destination = (
        args.output_root.resolve() / args.object_id / f"episode_{args.episode_id:04d}"
    )
    _require(not destination.exists(), f"fresh source masks exist: {destination}")
    scratch = destination.with_name(f".{destination.name}.incomplete-{os.getpid()}")
    _require(not scratch.exists(), "fresh source mask scratch exists")
    scratch.mkdir(parents=True)
    predictor = DeformableObjectSam2VideoPredictor(
        sam2_repository,
        checkpoint,
        device=args.device,
    )
    camera_records: list[dict[str, Any]] = []
    try:
        for row in stage["camera_records"]:
            camera = str(row["camera"])
            _require(camera in FROZEN_CAMERA_PANEL, "stage camera panel changed")
            video = source_episode / camera / "undistorted.mp4"
            _require(
                video.is_file() and file_sha256(video) == row["video_sha256"],
                f"staged source video changed: {camera}",
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
                    [np.count_nonzero(mask) for mask in masks],
                    dtype=np.int64,
                )
                camera_records.append(
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
                camera_records.append(
                    {
                        "camera": camera,
                        "status": "technical_failure",
                        "input_video_sha256": row["video_sha256"],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    finally:
        predictor.close()

    successful = sum(row["status"] == "success" for row in camera_records)
    minimum = int(mask_protocol["mask_contract"]["minimum_successful_cameras"])
    status = (
        "ready_for_source_processing" if successful >= minimum else "technical_failure"
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": MASK_ARTIFACT_KIND,
        "protocol_id": mask_protocol["protocol_id"],
        "protocol_config_sha256": mask_protocol["config_sha256"],
        "parent_window_protocol_config_sha256": window_protocol["config_sha256"],
        **case,
        "status": status,
        "code_revision": code_revision,
        "window_stage_result_sha256": stage["result_sha256"],
        "window_stage_file_sha256": file_sha256(stage_path),
        "input_camera_count": len(camera_records),
        "successful_camera_count": successful,
        "minimum_successful_cameras": minimum,
        "camera_records": camera_records,
        "dependencies": {
            "selector_object_source_sha256": file_sha256(selector_sources["object"]),
            "selector_base_source_sha256": file_sha256(selector_sources["base"]),
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
            "prediction_access_requires_a_separate_prefix_artifact": True,
            "technical_failures_cause_no_implicit_replacement": True,
        },
    }
    manifest["result_sha256"] = canonical_sha256(manifest, digest_key="result_sha256")
    (scratch / MASK_MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    scratch.rename(destination)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if status == "ready_for_source_processing" else 2


if __name__ == "__main__":
    raise SystemExit(main())
