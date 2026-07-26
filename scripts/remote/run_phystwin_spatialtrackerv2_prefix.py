#!/usr/bin/env python3
"""Run frozen SpatialTrackerV2 RGB-D inference on PhysTwin frame-zero queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bayesian_phystwin.phystwin_spatialtrackerv2_competence import (
    camera_tracks_to_world,
    project_world_queries_to_pixels,
)


def _load_input(path: Path) -> dict[str, np.ndarray]:
    required = {
        "video",
        "depths",
        "intrinsics",
        "extrinsics",
        "query_point",
    }
    with np.load(path) as archive:
        missing = required.difference(archive.files)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"SpatialTrackerV2 input lacks fields: {names}")
        arrays = {name: np.asarray(archive[name]) for name in required}
    video = arrays["video"]
    depths = arrays["depths"]
    intrinsics = arrays["intrinsics"]
    extrinsics = arrays["extrinsics"]
    queries = arrays["query_point"]
    if video.ndim != 4 or video.shape[-1] != 3 or video.dtype != np.uint8:
        raise ValueError("video must be uint8 with shape (T, H, W, 3)")
    if depths.shape != video.shape[:3]:
        raise ValueError("depths must have shape (T, H, W)")
    if intrinsics.shape != (len(video), 3, 3):
        raise ValueError("intrinsics must have shape (T, 3, 3)")
    if extrinsics.shape != (len(video), 4, 4):
        raise ValueError("extrinsics must have shape (T, 4, 4)")
    if queries.ndim != 2 or queries.shape[1] != 4:
        raise ValueError("query_point must have shape (N, 4)")
    if not np.all(queries[:, 0] == 0.0):
        raise ValueError("only frame-zero queries are authorized")
    return arrays


def run(args: argparse.Namespace) -> int:
    """Execute the official model with a fixed calibrated gauge."""

    import torch
    from models.SpaTrackV2.models.predictor import Predictor

    arrays = _load_input(Path(args.input))
    video_uint8 = arrays["video"]
    depths = np.asarray(arrays["depths"], dtype=np.float32)
    intrinsics = np.asarray(arrays["intrinsics"], dtype=np.float32)
    world_to_camera = np.asarray(arrays["extrinsics"], dtype=np.float64)
    query_points = np.asarray(arrays["query_point"], dtype=np.float64)
    query_pixels, query_depth = project_world_queries_to_pixels(
        query_points[:, 1:],
        intrinsics[0],
        world_to_camera[0],
    )
    height, width = video_uint8.shape[1:3]
    if np.any(
        (query_pixels[:, 1] < 0.0)
        | (query_pixels[:, 1] > width - 1)
        | (query_pixels[:, 2] < 0.0)
        | (query_pixels[:, 2] > height - 1)
    ):
        raise ValueError("a frame-zero query projects outside the RGB-D image")
    video = np.transpose(video_uint8, (0, 3, 1, 2)).astype(np.float32)
    camera_to_world = np.linalg.inv(world_to_camera)

    model = Predictor.from_pretrained(args.checkpoint_dir)
    model.spatrack.track_num = int(args.track_num)
    model.eval()
    model.to(args.device)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        (
            camera_to_initial_camera,
            _,
            _,
            _,
            track3d_camera,
            _,
            visibility_probability,
            confidence,
            _,
        ) = model.forward(
            video,
            depth=depths,
            intrs=intrinsics,
            extrs=camera_to_world.astype(np.float32),
            queries=query_pixels,
            fps=1,
            full_point=False,
            iters_track=int(args.iters_track),
            query_no_BA=True,
            fixed_cam=False,
            stage=1,
            unc_metric=None,
            support_frame=len(video) - 1,
            replace_ratio=float(args.replace_ratio),
        )

    track3d_camera_array = track3d_camera[:, :, :3].detach().cpu().numpy()
    camera_to_initial = camera_to_initial_camera.detach().cpu().numpy()
    coords_initial_camera = (
        np.einsum(
            "tij,tnj->tni",
            camera_to_initial[:, :3, :3],
            track3d_camera_array,
        )
        + camera_to_initial[:, :3, 3][:, None, :]
    )
    coords_world = camera_tracks_to_world(
        coords_initial_camera,
        camera_to_world[0],
    )
    visibility = np.squeeze(
        visibility_probability.detach().cpu().numpy(),
        axis=-1,
    )
    confidence_array = np.squeeze(
        confidence.detach().cpu().numpy(),
        axis=-1,
    )
    valid = (
        np.all(np.isfinite(coords_world), axis=2)
        & np.isfinite(visibility)
        & (visibility >= float(args.visibility_threshold))
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        coords_world_m=coords_world.astype(np.float32),
        valid=valid,
        visibility_probability=visibility.astype(np.float32),
        confidence=confidence_array.astype(np.float32),
        query_points=query_points.astype(np.float32),
        query_pixels_xyt=query_pixels.astype(np.float32),
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "frame_count": len(coords_world),
                "query_count": coords_world.shape[1],
                "visibility_fraction": float(np.mean(valid)),
                "minimum_query_depth_m": float(np.min(query_depth)),
                "maximum_query_depth_m": float(np.max(query_depth)),
                "future_target_loaded": False,
                "held_v8_access": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--track-num", type=int, default=756)
    parser.add_argument("--iters-track", type=int, default=4)
    parser.add_argument("--replace-ratio", type=float, default=0.2)
    parser.add_argument("--visibility-threshold", type=float, default=0.5)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
