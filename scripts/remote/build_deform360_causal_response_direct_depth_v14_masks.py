#!/usr/bin/env python3
"""Build prefix-only generic-SAM2 masks for one staged V14 source case."""

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

from bayesian_phystwin.deform360_causal_response_direct_depth_assets import (
    ASSET_PROTOCOL_ID,
    MASK_ARTIFACT_KIND,
    MASK_CONTRACT,
    PREFIX_FRAME_COUNT,
    canonical_sha256,
    load_v14_asset_protocol,
    validate_v14_staged_window,
)
from bayesian_phystwin.deform360_causal_response_direct_depth_cohort import (
    validate_v14_staging_queue,
)
from bayesian_phystwin.deform360_exact_video_cadence import (
    decoded_frame_count,
    decoded_prefix_sha256,
    trim_video_exact_30hz,
)
from bayesian_phystwin.deform360_object_exclusion import file_sha256


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON artifact: {path}") from error
    _require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
    return payload


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


def _resolve_required_executable(path: Path, *, name: str) -> Path:
    expanded = path.expanduser()
    _require(expanded.is_absolute(), f"{name} path must be absolute")
    resolved = expanded.resolve()
    _require(
        resolved.is_file() and os.access(resolved, os.X_OK),
        f"{name} executable is unavailable",
    )
    try:
        subprocess.run(
            [str(resolved), "-version"],
            check=True,
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(f"{name} executable cannot run") from error
    return resolved


def _write_masks(path: Path, masks: list[np.ndarray]) -> None:
    values = np.asarray(masks, dtype=np.uint8)
    _require(
        values.ndim == 3
        and values.shape[0] == PREFIX_FRAME_COUNT
        and np.all(np.count_nonzero(values, axis=(1, 2)) > 0),
        "SAM2 returned invalid or empty V14 prefix masks",
    )
    with h5py.File(path, "w") as stream:
        stream.create_dataset(
            "data",
            data=values,
            dtype=np.uint8,
            compression="gzip",
            compression_opts=4,
        )


def _write_result(path: Path, payload: dict[str, Any]) -> None:
    _require(not path.exists(), f"refusing to replace V14 mask result: {path}")
    payload["artifact_sha256"] = canonical_sha256(
        payload,
        namespace=b"deform360-causal-response-direct-depth-prefix-masks-v14\0",
        digest_key="artifact_sha256",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    _require(not temporary.exists(), f"temporary V14 mask result exists: {temporary}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--method-protocol", type=Path, required=True)
    parser.add_argument("--asset-protocol", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--stage-result", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--candidate-rank", type=int, required=True)
    parser.add_argument("--selector-source-root", type=Path, required=True)
    parser.add_argument("--sam2-repository", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--device", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    ffmpeg = _resolve_required_executable(args.ffmpeg, name="ffmpeg")
    repository = args.repo.resolve()
    code_revision = _require_clean_repository(repository)
    method_path = args.method_protocol.resolve()
    method = _read_json(method_path)
    asset_path = args.asset_protocol.resolve()
    asset = load_v14_asset_protocol(asset_path)
    _require(
        method.get("protocol_id") == asset["parent_method"]["protocol_id"]
        and method.get("config_sha256")
        == asset["parent_method"]["config_sha256"]
        and file_sha256(method_path) == asset["parent_method"]["file_sha256"],
        "V14 asset protocol binds another method",
    )
    queue_path = args.queue.resolve()
    queue = validate_v14_staging_queue(queue_path)
    _require(
        queue["queue_sha256"] == asset["staging"]["queue_sha256"]
        and file_sha256(queue_path)
        == asset["staging"]["queue_file_sha256"],
        "V14 asset protocol binds another staging queue",
    )
    rank = args.candidate_rank
    _require(
        1 <= rank <= len(queue["candidates"]),
        "V14 mask rank is outside the queue",
    )
    candidate = queue["candidates"][rank - 1]
    object_id = str(candidate["object_id"])
    episode_id = int(candidate["episode_id"])
    stage_episode = (
        args.stage_root.resolve()
        / object_id
        / f"episode_{episode_id:04d}"
    )
    stage, _ = validate_v14_staged_window(
        args.stage_result.resolve(),
        protocol=method,
        asset_protocol=asset,
        queue=queue,
        queue_rank=rank,
        stage_episode=stage_episode,
    )

    selector_root = args.selector_source_root.resolve()
    selector_sources = {
        "base": selector_root / "causal4d_public" / "deform360_sam2.py",
        "object": selector_root
        / "causal4d_public"
        / "deform360_object_sam2.py",
    }
    mask_contract = asset["mask"]
    _require(
        file_sha256(selector_sources["base"])
        == mask_contract["selector_base_source_sha256"]
        and file_sha256(selector_sources["object"])
        == mask_contract["selector_object_source_sha256"],
        "V14 generic SAM2 selector source changed",
    )
    sam2_repository = args.sam2_repository.resolve()
    checkpoint = args.checkpoint.resolve()
    _require(
        _git_revision(sam2_repository) == mask_contract["sam2_commit"]
        and file_sha256(checkpoint)
        == mask_contract["sam2_checkpoint_sha256"],
        "V14 SAM2 dependency changed",
    )
    sys.path.insert(0, str(selector_root))
    from causal4d_public.deform360_object_sam2 import (  # noqa: PLC0415
        DeformableObjectSam2VideoPredictor,
    )

    output_episode = (
        args.output_root.resolve()
        / object_id
        / f"episode_{episode_id:04d}"
    )
    result_path = args.result.resolve()
    _require(
        not output_episode.exists() and not result_path.exists(),
        "V14 prefix mask output or result already exists",
    )
    scratch = output_episode.with_name(
        f".{output_episode.name}.incomplete-{os.getpid()}"
    )
    _require(not scratch.exists(), f"V14 prefix mask scratch exists: {scratch}")
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
            source_video = stage_episode / camera / "undistorted.mp4"
            output_camera = scratch / camera
            output_camera.mkdir()
            prefix_video = output_camera / "prefix.mp4"
            try:
                source_prefix_sha256 = decoded_prefix_sha256(
                    ffmpeg,
                    source_video,
                    PREFIX_FRAME_COUNT,
                )
                trim_video_exact_30hz(
                    ffmpeg,
                    source_video,
                    prefix_video,
                    0,
                    PREFIX_FRAME_COUNT,
                )
                initial_mask, initialization = predictor.select_initial_mask(
                    prefix_video
                )
                propagated = list(
                    predictor.segment_from_initial_mask(
                        prefix_video,
                        initial_mask,
                        initialization={
                            "policy": "v14_generic_exact_prefix_frame_zero",
                            "source_frame_index": 0,
                            "maximum_source_frame_index": (
                                PREFIX_FRAME_COUNT - 1
                            ),
                            "future_object_observations_used": False,
                            "selection": initialization,
                        },
                    )
                )
                _require(
                    [index for index, _ in propagated]
                    == list(range(PREFIX_FRAME_COUNT)),
                    f"SAM2 returned incomplete V14 prefix frames: {camera}",
                )
                masks = [
                    np.asarray(mask, dtype=bool) for _, mask in propagated
                ]
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
                        "source_video_sha256": row["video_sha256"],
                        "source_decoded_prefix_sha256": source_prefix_sha256,
                        "prefix_video_sha256": file_sha256(prefix_video),
                        "prefix_decoded_frame_count": decoded_frame_count(
                            prefix_video
                        ),
                        "mask_sha256": file_sha256(mask_path),
                        "frame_count": len(masks),
                        "area_pixels_min": int(np.min(areas)),
                        "area_pixels_median": float(np.median(areas)),
                        "area_pixels_max": int(np.max(areas)),
                        "initialization": initialization,
                    }
                )
            except Exception as error:
                shutil.rmtree(output_camera, ignore_errors=True)
                camera_records.append(
                    {
                        "camera": camera,
                        "status": "technical_failure",
                        "source_video_sha256": row["video_sha256"],
                        "failure": {
                            "type": type(error).__name__,
                            "reason": "prefix-mask-generation-failed",
                        },
                    }
                )
    finally:
        predictor.close()
    successful = sum(row["status"] == "success" for row in camera_records)
    minimum = int(mask_contract["minimum_successful_camera_count"])
    status = (
        "ready_for_prefix_geometry"
        if successful >= minimum
        else "technical_preflight_failure"
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": MASK_ARTIFACT_KIND,
        "contract": MASK_CONTRACT,
        "protocol_id": ASSET_PROTOCOL_ID,
        "asset_protocol_config_sha256": asset["config_sha256"],
        "asset_protocol_file_sha256": file_sha256(asset_path),
        "method_protocol_config_sha256": method["config_sha256"],
        "queue_sha256": queue["queue_sha256"],
        "queue_rank": rank,
        "object_hash": stage["object_hash"],
        "case_hash": stage["case_hash"],
        "category": candidate["category"],
        "status": status,
        "code_revision": code_revision,
        "window_stage_artifact_sha256": stage["artifact_sha256"],
        "window_stage_file_sha256": file_sha256(
            args.stage_result.resolve()
        ),
        "input_camera_count": len(camera_records),
        "successful_camera_count": successful,
        "minimum_successful_camera_count": minimum,
        "prefix_frame_count": PREFIX_FRAME_COUNT,
        "maximum_object_observation_frame": PREFIX_FRAME_COUNT - 1,
        "camera_records": camera_records,
        "dependencies": {
            "ffmpeg_path": str(ffmpeg),
            "ffmpeg_sha256": file_sha256(ffmpeg),
            "selector_object_source_sha256": file_sha256(
                selector_sources["object"]
            ),
            "selector_base_source_sha256": file_sha256(
                selector_sources["base"]
            ),
            "sam2_commit": _git_revision(sam2_repository),
            "sam2_checkpoint_sha256": file_sha256(checkpoint),
        },
        "information_boundary": {
            "source_object_frames_read": [0, PREFIX_FRAME_COUNT - 1],
            "source_object_frames_after_prefix_read": False,
            "manual_prompting_or_mask_selection": False,
            "object_geometry_read": False,
            "particle_identity_or_metric_read": False,
            "target_object_or_outcome_read": False,
            "held_v8_artifact_or_process_access": False,
            "technical_preflight_failure_is_not_a_prediction": True,
        },
    }
    output_episode.parent.mkdir(parents=True, exist_ok=True)
    scratch.rename(output_episode)
    _write_result(result_path, payload)
    print(status)
    print(payload["artifact_sha256"])
    return 0 if status == "ready_for_prefix_geometry" else 2


if __name__ == "__main__":
    raise SystemExit(main())
