#!/usr/bin/env python3
"""Gate a source-only Deform360 splat against its multiview hull seed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial import cKDTree

from causal4d_public.deform360_dense_source import sha256_file
from deform360.annotations import H5Array
from deform360.processing.episode import episode_cameras, load_episode_calibration
from deform360.processing.pcd_stage import (
    CROP_HALF_EXTENT_M,
    SEED_POINT_COUNT,
    seed_points_from_splat,
)
from deform360.processing.reconstruct_stage import visual_hull_points


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-dir", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--minimum-hull-points", type=int, default=512)
    parser.add_argument("--voxel-resolution", type=int, default=120)
    parser.add_argument("--maximum-chamfer-m", type=float, default=0.05)
    parser.add_argument("--maximum-center-error-m", type=float, default=0.05)
    parser.add_argument("--maximum-minor-span-m", type=float, default=0.12)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _pca_summary(points: np.ndarray) -> dict[str, Any]:
    center = np.median(points, axis=0)
    centered = points - center
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ axes.T
    quantiles = np.percentile(projected, [1.0, 50.0, 99.0], axis=0)
    spans = np.sort(quantiles[2] - quantiles[0])[::-1]
    return {
        "median_world_m": center.tolist(),
        "pca_q01_to_q99_spans_m_descending": spans.tolist(),
        "robust_pca_volume_m3": float(np.prod(spans)),
    }


def _symmetric_chamfer_m(first: np.ndarray, second: np.ndarray) -> float:
    first_to_second = cKDTree(second).query(first, workers=-1)[0]
    second_to_first = cKDTree(first).query(second, workers=-1)[0]
    return float(0.5 * (first_to_second.mean() + second_to_first.mean()))


def _load_frame_inputs(
    episode_dir: Path, frame: int
) -> tuple[
    list[str],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    cameras = list(episode_cameras(episode_dir))
    intrinsics, extrinsics = load_episode_calibration(episode_dir)
    masks: dict[str, np.ndarray] = {}
    images: dict[str, np.ndarray] = {}
    for camera in cameras:
        camera_dir = episode_dir / camera
        with H5Array(camera_dir / "mask_refined.h5") as store:
            masks[camera] = np.asarray(store[frame], dtype=np.uint8)
        capture = cv2.VideoCapture(str(camera_dir / "undistorted.mp4"))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, bgr = capture.read()
        finally:
            capture.release()
        if not ok:
            raise RuntimeError(f"cannot read frame {frame} for {camera}")
        images[camera] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cameras, masks, images, intrinsics, extrinsics


def main() -> int:
    args = _parse_args()
    episode_dir = args.aligned_dir / f"episode_{args.episode:04d}"
    source_manifest = episode_dir / "dense_source_smoke.manifest.json"
    source_boundary = json.loads(source_manifest.read_text(encoding="utf-8"))
    if not source_boundary.get("source_only"):
        raise ValueError("strict-hull audit accepts only source-only data")
    splat_path = episode_dir / "splatfacto" / f"splat_{args.frame}.ply"
    cameras, masks, images, intrinsics, extrinsics = _load_frame_inputs(
        episode_dir, args.frame
    )
    hull, _ = visual_hull_points(
        masks,
        images,
        {camera: intrinsics[camera] for camera in cameras},
        {camera: extrinsics[camera] for camera in cameras},
        voxel_resolution=args.voxel_resolution,
        min_points=args.minimum_hull_points,
    )
    reconstruction, _ = seed_points_from_splat(
        splat_path,
        crop_half_extent_m=CROP_HALF_EXTENT_M,
        seed_count=SEED_POINT_COUNT,
        rng_seed=0,
    )
    hull_summary = _pca_summary(hull)
    reconstruction_summary = _pca_summary(reconstruction)
    hull_spans = np.asarray(
        hull_summary["pca_q01_to_q99_spans_m_descending"], dtype=float
    )
    reconstruction_spans = np.asarray(
        reconstruction_summary["pca_q01_to_q99_spans_m_descending"], dtype=float
    )
    center_error = float(
        np.linalg.norm(
            np.asarray(reconstruction_summary["median_world_m"])
            - np.asarray(hull_summary["median_world_m"])
        )
    )
    chamfer = _symmetric_chamfer_m(reconstruction, hull)
    major_span_ratio = float(reconstruction_spans[0] / hull_spans[0])
    volume_ratio = float(
        reconstruction_summary["robust_pca_volume_m3"]
        / max(hull_summary["robust_pca_volume_m3"], np.finfo(float).tiny)
    )
    gates = {
        "chamfer": bool(chamfer <= args.maximum_chamfer_m),
        "center": bool(center_error <= args.maximum_center_error_m),
        "major_span": bool(0.70 <= major_span_ratio <= 1.40),
        "minor_span": bool(
            reconstruction_spans[1] <= args.maximum_minor_span_m
        ),
        "not_phantom_volume": bool(volume_ratio <= 8.0),
    }
    payload: dict[str, Any] = {
        "schema": "bayesian-phystwin/deform360-strict-hull-audit/v1",
        "source_only": True,
        "episode": args.episode,
        "frame": args.frame,
        "camera_count": len(cameras),
        "minimum_hull_points": args.minimum_hull_points,
        "voxel_resolution": args.voxel_resolution,
        "source_manifest_sha256": sha256_file(source_manifest),
        "splat_sha256": sha256_file(splat_path),
        "hull_point_count": len(hull),
        "reconstruction_point_count": len(reconstruction),
        "hull": hull_summary,
        "reconstruction": reconstruction_summary,
        "symmetric_chamfer_m": chamfer,
        "median_center_error_m": center_error,
        "major_span_ratio": major_span_ratio,
        "robust_pca_volume_ratio": volume_ratio,
        "thresholds": {
            "maximum_chamfer_m": args.maximum_chamfer_m,
            "maximum_center_error_m": args.maximum_center_error_m,
            "maximum_minor_span_m": args.maximum_minor_span_m,
            "major_span_ratio": [0.70, 1.40],
            "maximum_robust_pca_volume_ratio": 8.0,
        },
        "acceptance_gates": gates,
        "passed": all(gates.values()),
        "claim_boundary": (
            "exploratory source-only reconstruction QA; no calibration or target "
            "episode was read"
        ),
    }
    payload["result_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    output_path = args.output or episode_dir / "strict_hull_audit.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
