#!/usr/bin/env python3
"""Generate a sealed render-to-real AllTracker prefix prediction.

The runner consumes only an already-rendered PhysTwin prefix, frame-zero
association-oracle query positions, allowed object masks, and fixed camera
calibration. It never reads the later manual identity trajectory or any frame
after the released training endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.phystwin_alltracker_cues import (
    PhysTwinAllTrackerCueConfig,
    PhysTwinAllTrackerRunner,
)
from bayesian_phystwin.rendered_alltracker_observation import (
    RenderedAllTrackerConfig,
    build_rendered_alltracker_observation,
    project_world_points,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_camera_panel(
    path: Path,
    *,
    names: list[str],
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    by_name = {str(row["img_name"]): row for row in rows}
    missing = sorted(set(names) - set(by_name))
    _require(not missing, f"camera JSON is missing {missing}")
    intrinsics = []
    camera_to_world = []
    for name in names:
        row = by_name[name]
        source_width = int(row["width"])
        source_height = int(row["height"])
        _require(
            source_width >= width and source_height >= height,
            "render resolution exceeds the calibrated image",
        )
        scale_x = width / source_width
        scale_y = height / source_height
        intrinsics.append(
            np.asarray(
                [
                    [float(row["fx"]) * scale_x, 0.0, 0.5 * width],
                    [0.0, float(row["fy"]) * scale_y, 0.5 * height],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
        )
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = np.asarray(row["rotation"], dtype=np.float64)
        pose[:3, 3] = np.asarray(row["position"], dtype=np.float64)
        camera_to_world.append(pose)
    return np.stack(intrinsics), np.stack(camera_to_world)


def _nearest_nodes(
    frame_zero_nodes_m: np.ndarray,
    query_positions_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    distances, indices = cKDTree(frame_zero_nodes_m).query(
        query_positions_m,
        k=1,
    )
    return indices.astype(np.int64), distances.astype(np.float64)


def _sample_mask(mask: np.ndarray, pixels_xy: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    finite = np.all(np.isfinite(pixels_xy), axis=1)
    x = np.zeros(len(pixels_xy), dtype=np.int64)
    y = np.zeros(len(pixels_xy), dtype=np.int64)
    x[finite] = np.rint(pixels_xy[finite, 0]).astype(np.int64)
    y[finite] = np.rint(pixels_xy[finite, 1]).astype(np.int64)
    inside = (
        finite
        & (x >= 0)
        & (x < width)
        & (y >= 0)
        & (y < height)
    )
    sampled = np.zeros(len(pixels_xy), dtype=bool)
    sampled[inside] = mask[y[inside], x[inside]]
    return sampled


def _read_masks(
    pattern: str,
    *,
    frames: np.ndarray,
    camera_count: int,
    width: int,
    height: int,
) -> tuple[np.ndarray, dict[str, str]]:
    import cv2

    masks = np.empty((len(frames), camera_count, height, width), dtype=bool)
    hashes: dict[str, str] = {}
    for frame_position, frame in enumerate(frames):
        for camera in range(camera_count):
            path = Path(pattern.format(camera=camera, frame=int(frame)))
            _require(path.is_file(), f"missing prefix mask {path}")
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            _require(image is not None, f"cannot read prefix mask {path}")
            resized = cv2.resize(
                image,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            masks[frame_position, camera] = resized > 0
            hashes[str(path)] = _sha256(path)
    return masks, hashes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--photometric-carrier", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--query-input", type=Path, required=True)
    parser.add_argument("--camera-json", type=Path, required=True)
    parser.add_argument("--mask-pattern", required=True)
    parser.add_argument("--alltracker-source", type=Path, required=True)
    parser.add_argument("--alltracker-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--camera-names", nargs="+", default=["cam0", "cam1", "cam2"])
    parser.add_argument("--expected-frames", type=int, nargs="+", required=True)
    parser.add_argument("--maximum-query-association-m", type=float, default=0.005)
    parser.add_argument("--minimum-render-alpha", type=float, default=0.1)
    parser.add_argument("--minimum-quality", type=float, default=0.5)
    parser.add_argument("--maximum-cycle-error-px", type=float, default=5.0)
    parser.add_argument("--maximum-reprojection-error-px", type=float, default=3.0)
    parser.add_argument("--minimum-camera-count", type=int, default=2)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--inference-iterations", type=int, default=4)
    parser.add_argument("--window-length", type=int, default=16)
    args = parser.parse_args()

    _require(not args.output.exists(), "output already exists")
    for path in (
        args.photometric_carrier,
        args.trajectory,
        args.query_input,
        args.camera_json,
        args.alltracker_checkpoint,
    ):
        _require(path.is_file(), f"missing input {path}")
    _require(args.alltracker_source.is_dir(), "AllTracker source is missing")

    with np.load(args.photometric_carrier) as carrier:
        required = {
            "frames",
            "observed_rgb",
            "baseline_rgb",
            "baseline_alpha",
        }
        missing = required.difference(carrier.files)
        _require(not missing, f"photometric carrier lacks {sorted(missing)}")
        frames = np.asarray(carrier["frames"], dtype=np.int64)
        observed = np.asarray(carrier["observed_rgb"], dtype=np.float32)
        baseline = np.asarray(carrier["baseline_rgb"], dtype=np.float32)
        alpha = np.asarray(carrier["baseline_alpha"], dtype=np.float32)
    expected = np.asarray(args.expected_frames, dtype=np.int64)
    _require(np.array_equal(frames, expected), "carrier frame inventory changed")
    _require(
        observed.shape == baseline.shape
        and observed.ndim == 5
        and observed.shape[-1] == 3,
        "rendered RGB shape changed",
    )
    _require(
        alpha.shape == observed.shape[:-1],
        "render alpha shape changed",
    )
    _require(
        observed.shape[1] == len(args.camera_names),
        "camera inventory changed",
    )
    _require(
        np.all(np.isfinite(observed))
        and np.all(np.isfinite(baseline))
        and np.all(np.isfinite(alpha)),
        "render carrier contains non-finite values",
    )

    with args.trajectory.open("rb") as handle:
        trajectory = np.asarray(pickle.load(handle), dtype=np.float64)
    _require(
        trajectory.ndim == 3 and trajectory.shape[2] == 3,
        "trajectory must have shape (frame, node, 3)",
    )
    _require(int(np.max(frames)) < len(trajectory), "frame exceeds trajectory")
    with np.load(args.query_input) as query_archive:
        _require(
            "query_point" in query_archive.files,
            "query input lacks query_point",
        )
        query_point = np.asarray(query_archive["query_point"], dtype=np.float64)
    _require(
        query_point.ndim == 2
        and query_point.shape[1] == 4
        and np.all(query_point[:, 0] == 0.0)
        and np.all(np.isfinite(query_point)),
        "queries must be finite frame-zero world points",
    )
    query_positions = query_point[:, 1:]
    node_indices, association_distance = _nearest_nodes(
        trajectory[0],
        query_positions,
    )
    _require(
        float(np.max(association_distance)) <= args.maximum_query_association_m,
        "frame-zero query association exceeds the frozen tolerance",
    )
    physical_query_positions = trajectory[frames][:, node_indices]

    height, width = observed.shape[2:4]
    intrinsics, camera_to_world = _load_camera_panel(
        args.camera_json,
        names=args.camera_names,
        width=width,
        height=height,
    )
    masks, mask_hashes = _read_masks(
        args.mask_pattern,
        frames=frames,
        camera_count=len(args.camera_names),
        width=width,
        height=height,
    )

    frame_count = len(frames)
    camera_count = len(args.camera_names)
    identity_count = len(query_positions)
    source_pixels = np.full(
        (frame_count, camera_count, identity_count, 2),
        np.nan,
        dtype=np.float32,
    )
    target_pixels = np.full_like(source_pixels, np.nan)
    recovered_source_pixels = np.full_like(source_pixels, np.nan)
    quality = np.zeros(
        (frame_count, camera_count, identity_count),
        dtype=np.float32,
    )
    cycle_error = np.full_like(quality, np.inf)
    source_supported = np.zeros_like(quality, dtype=bool)
    target_supported = np.zeros_like(quality, dtype=bool)

    runner_config = PhysTwinAllTrackerCueConfig(
        train_end_frame=2,
        max_side=args.max_side,
        inference_iterations=args.inference_iterations,
        window_length=args.window_length,
        visibility_threshold=args.minimum_quality,
    )
    runner = PhysTwinAllTrackerRunner(
        args.alltracker_source,
        args.alltracker_checkpoint,
        device=args.device,
        config=runner_config,
    )
    try:
        for frame_position in range(frame_count):
            for camera in range(camera_count):
                pixels, depth = project_world_points(
                    physical_query_positions[frame_position],
                    intrinsics[camera],
                    camera_to_world[camera],
                )
                source_pixels[frame_position, camera] = pixels
                source_supported[frame_position, camera] = (
                    (depth > 0.0)
                    & _sample_mask(
                        alpha[frame_position, camera]
                        >= args.minimum_render_alpha,
                        pixels,
                    )
                )
                tracker_queries = np.where(
                    np.all(np.isfinite(pixels), axis=1, keepdims=True),
                    pixels,
                    0.0,
                )
                pair = np.clip(
                    np.rint(
                        255.0
                        * np.stack(
                            (
                                baseline[frame_position, camera],
                                observed[frame_position, camera],
                            )
                        )
                    ),
                    0.0,
                    255.0,
                ).astype(np.uint8)
                forward = runner.track(pair, tracker_queries)
                predicted = forward.tracks_xy[1]
                target_pixels[frame_position, camera] = predicted
                reverse = runner.track(pair[::-1], predicted)
                recovered = reverse.tracks_xy[1]
                recovered_source_pixels[frame_position, camera] = recovered
                quality[frame_position, camera] = np.minimum(
                    forward.quality_probability[1],
                    reverse.quality_probability[1],
                )
                cycle = np.linalg.norm(
                    recovered - tracker_queries,
                    axis=1,
                )
                cycle[~source_supported[frame_position, camera]] = np.inf
                cycle_error[frame_position, camera] = cycle
                target_supported[frame_position, camera] = _sample_mask(
                    masks[frame_position, camera],
                    predicted,
                )
    finally:
        runner.close()

    observation = build_rendered_alltracker_observation(
        target_pixels,
        quality,
        cycle_error,
        source_supported,
        target_supported,
        intrinsics,
        camera_to_world,
        config=RenderedAllTrackerConfig(
            minimum_quality=args.minimum_quality,
            maximum_cycle_error_px=args.maximum_cycle_error_px,
            maximum_reprojection_error_px=args.maximum_reprojection_error_px,
            minimum_camera_count=args.minimum_camera_count,
        ),
    )
    association_offset = (
        query_positions - trajectory[0, node_indices]
    )
    anchored_points = (
        observation.points_world_m + association_offset[None]
    )

    args.output.mkdir(parents=True)
    prediction_path = args.output / "prediction.npz"
    np.savez_compressed(
        prediction_path,
        frames=frames,
        query_point=query_point.astype(np.float32),
        node_indices=node_indices,
        association_distance_m=association_distance,
        physical_query_positions_m=physical_query_positions.astype(np.float32),
        source_pixels_xy=source_pixels,
        target_pixels_xy=target_pixels,
        recovered_source_pixels_xy=recovered_source_pixels,
        quality_probability=quality,
        cycle_error_px=cycle_error,
        source_supported=source_supported,
        target_supported=target_supported,
        intrinsics=intrinsics,
        camera_to_world=camera_to_world,
        points_world_m=observation.points_world_m,
        covariance_m2=observation.covariance_m2,
        valid=observation.valid,
        raw_camera_count=observation.raw_camera_count,
        effective_camera_count=observation.effective_camera_count,
        reprojection_error_px=observation.reprojection_error_px,
        prior_reliability=observation.prior_reliability,
        two_view_fallback=observation.two_view_fallback,
        association_offset_m=association_offset,
        anchored_points_world_m=anchored_points,
    )
    report = {
        "schema_version": 1,
        "artifact_kind": "PhysTwinRenderedAllTrackerPrefixPrediction",
        "status": "prediction_complete_unscored",
        "frames": frames.tolist(),
        "query_count": identity_count,
        "camera_names": args.camera_names,
        "valid_count": int(np.sum(observation.valid)),
        "valid_fraction": float(np.mean(observation.valid)),
        "two_view_fallback_count": int(np.sum(observation.two_view_fallback)),
        "maximum_association_distance_m": float(
            np.max(association_distance)
        ),
        "alltracker": {
            "source_sha256": runner.source_sha256,
            "checkpoint_sha256": runner.checkpoint_sha256,
            "max_side": args.max_side,
            "inference_iterations": args.inference_iterations,
            "window_length": args.window_length,
        },
        "inputs": {
            "photometric_carrier_sha256": _sha256(args.photometric_carrier),
            "trajectory_sha256": _sha256(args.trajectory),
            "query_input_sha256": _sha256(args.query_input),
            "camera_json_sha256": _sha256(args.camera_json),
            "mask_sha256": mask_hashes,
        },
        "prediction_sha256": _sha256(prediction_path),
        "prediction_points_sha256": _array_sha256(
            anchored_points
        ),
        "prediction_valid_sha256": _array_sha256(observation.valid),
        "later_manual_identity_trajectory_read": False,
        "future_frame_after_120_read": False,
        "cotracker_comparator_read": False,
        "held_v8_read": False,
        "claim_boundary": (
            "Opened-source association-oracle competence prediction only. "
            "This is unscored until the prediction seal is written."
        ),
    }
    report_path = args.output / "prediction_report.json"
    _write_json(report_path, report)
    seal_path = args.output / "PREDICTION_SEAL"
    seal_path.write_text(_sha256(report_path) + "\n", encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
