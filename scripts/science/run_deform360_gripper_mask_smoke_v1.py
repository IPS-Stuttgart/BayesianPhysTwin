#!/usr/bin/env python3
"""Render one source-camera UMI gripper mask and retain compact evidence only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import traceback
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-episode-root", required=True, type=Path)
    parser.add_argument("--source-object", required=True)
    parser.add_argument("--source-episode", required=True, type=int)
    parser.add_argument("--smoke-root", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--workflow-run-id", required=True, type=int)
    parser.add_argument("--workflow-run-attempt", required=True, type=int)
    parser.add_argument("--official-deform360-revision", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    import h5py
    import numpy as np

    from deform360.processing.urdf_render import process_gripper_masks_episode

    source_episode = args.source_episode_root.resolve(strict=True)
    smoke_root = args.smoke_root.resolve()
    evidence = args.evidence_root.resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    if smoke_root.exists():
        raise FileExistsError(f"smoke root already exists: {smoke_root}")

    result: dict[str, Any] = {
        "schema": "bayesian-phystwin/deform360-source-gripper-mask-smoke-v1",
        "repository": args.repository,
        "revision": args.revision,
        "workflow_run_id": args.workflow_run_id,
        "workflow_run_attempt": args.workflow_run_attempt,
        "runner_name": os.environ.get("RUNNER_NAME"),
        "required_runner_label": "gpuserver4090",
        "official_deform360_revision": args.official_deform360_revision,
        "source_object": args.source_object,
        "source_episode": args.source_episode,
        "mask": {},
        "errors": [],
        "information_boundary": {
            "source_video_header_opened": False,
            "source_robot_payload_opened": False,
            "persistent_gripper_mask_written": False,
            "requested_processed_tree_modified": False,
            "target_directory_contents_listed": False,
            "target_numeric_payload_opened": False,
            "target_scoring_performed": False,
            "paper_claim_authorized": False,
        },
    }

    try:
        cameras = sorted(
            path.name
            for path in source_episode.iterdir()
            if path.is_dir()
            and (path / "undistorted.mp4").is_file()
            and (path / "aligned_timestamps.txt").is_file()
        )
        if not cameras:
            raise FileNotFoundError("source episode has no aligned camera")
        camera = cameras[0]
        aligned_object = smoke_root / args.source_object
        smoke_episode = aligned_object / f"episode_{args.source_episode:04d}"
        smoke_camera = smoke_episode / camera
        smoke_camera.mkdir(parents=True, exist_ok=False)

        for name in ("undistorted_intrinsics.npy", "extrinsics.npy", "alignment.json"):
            source = source_episode / name
            if source.exists():
                (smoke_episode / name).symlink_to(source)
        robot_dir = smoke_episode / "robot"
        robot_dir.mkdir()
        (robot_dir / "robot.npz").symlink_to(source_episode / "robot" / "robot.npz")
        for name in ("undistorted.mp4", "aligned_timestamps.txt"):
            (smoke_camera / name).symlink_to(source_episode / camera / name)

        outputs = process_gripper_masks_episode(
            aligned_object,
            args.source_episode,
            cameras=[camera],
            overwrite=True,
        )
        result["information_boundary"]["source_video_header_opened"] = True
        result["information_boundary"]["source_robot_payload_opened"] = True
        mask_path = outputs[camera]
        meta_path = mask_path.with_name("rendered_urdf.meta.json")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        with h5py.File(mask_path, "r") as store:
            data = store["data"]
            frame_nonzero = np.asarray(
                [np.count_nonzero(data[index]) for index in range(data.shape[0])],
                dtype=np.int64,
            )
            shape = [int(value) for value in data.shape]
            dtype = str(data.dtype)
        frames_nonempty = int(np.count_nonzero(frame_nonzero))
        if frames_nonempty < 1:
            raise RuntimeError("rendered gripper mask is empty in every frame")
        result["mask"] = {
            "status": "success",
            "camera_selection": "lexicographically-first-aligned-camera",
            "camera": camera,
            "aligned_camera_count": len(cameras),
            "shape": shape,
            "dtype": dtype,
            "frames_nonempty": frames_nonempty,
            "minimum_nonzero_pixels": int(frame_nonzero.min()),
            "median_nonzero_pixels": float(np.median(frame_nonzero)),
            "maximum_nonzero_pixels": int(frame_nonzero.max()),
            "mask_size_bytes": mask_path.stat().st_size,
            "mask_sha256": _sha256(mask_path),
            "metadata_sha256": _sha256(meta_path),
            "stage_inputs": metadata.get("inputs"),
            "stage_parameters": metadata.get("parameters"),
        }
        result["decision"] = "source-gripper-mask-runtime-qualified"
        result["next_action"] = (
            "Reuse the qualified renderer for every source camera after object-mask "
            "qualification, before depth and point-cloud construction."
        )
    except Exception as error:
        result["mask"]["status"] = "failure"
        result["errors"].append(
            {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(limit=24),
            }
        )
        result["decision"] = "source-gripper-mask-runtime-not-qualified"
        result["next_action"] = (
            "Repair only the recorded EGL, URDF, calibration, or renderer dependency "
            "failure before multi-camera execution."
        )
    finally:
        shutil.rmtree(smoke_root, ignore_errors=True)

    result["claim_boundary"] = (
        "Source-only one-camera UMI silhouette runtime qualification. No object mask, "
        "depth, geometry, physical belief, target result, calibration, safety, or paper claim."
    )
    canonical = json.dumps(
        result, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    result["result_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    (evidence / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (evidence / "report.md").write_text(
        "# Deform360 source gripper-mask smoke v1\n\n"
        f"Decision: `{result['decision']}`\n\n"
        f"Camera: `{result['mask'].get('camera', 'unresolved')}`\n\n"
        f"Nonempty frames: `{result['mask'].get('frames_nonempty', 0)}`\n\n"
        f"Next action: {result['next_action']}\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
