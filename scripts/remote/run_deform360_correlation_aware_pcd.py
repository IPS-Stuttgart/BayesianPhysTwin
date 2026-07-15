#!/usr/bin/env python3
"""Build and score a source-only correlation-aware Deform360 point track."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from causal4d_public.deform360_dense_fusion import (
    DenseVelocityFusionConfig,
    fuse_correlated_velocity_observations,
    graph_complete_velocity,
    knn_springs,
)
from causal4d_public.deform360_dense_source import sha256_file
from deform360.processing.episode import (
    camera_frame_count,
    episode_cameras,
    load_episode_calibration,
)
from deform360.processing.pcd_stage import (
    CROP_HALF_EXTENT_M,
    FRAME_RATE_HZ,
    SEED_POINT_COUNT,
    TAIL_FRAMES_SKIPPED,
    _sample_camera_velocity,
    invert_transform,
    list_episode_splats,
    seed_points_from_splat,
)
from deform360.robot import load_robot_state


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _result_sha256(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def _symmetric_chamfer_m(first: np.ndarray, second: np.ndarray) -> float:
    from scipy.spatial import cKDTree

    first_to_second = cKDTree(second).query(first, workers=-1)[0]
    second_to_first = cKDTree(first).query(second, workers=-1)[0]
    return float(0.5 * (first_to_second.mean() + second_to_first.mean()))


def _maximum_actuator_speed_mps(episode_dir: Path) -> float:
    state = load_robot_state(episode_dir / "robot" / "robot.npz")
    positions = np.asarray(state.T_worlds)[..., :3, 3]
    speed = FRAME_RATE_HZ * np.linalg.norm(np.diff(positions, axis=0), axis=-1)
    return float(np.max(speed))


def _load_view_measurements(
    episode_dir: Path,
    cameras: list[str],
    intrinsics: dict[str, np.ndarray],
    extrinsics: dict[str, np.ndarray],
    points: np.ndarray,
    frame: int,
) -> tuple[np.ndarray, np.ndarray]:
    velocities = []
    validities = []
    for camera in cameras:
        tracking = episode_dir / camera / "tracking"
        with h5py.File(tracking / "vel.h5", "r") as stream:
            velocity_frame = np.asarray(stream["data"][frame])
        with h5py.File(tracking / "visibility.h5", "r") as stream:
            visibility_frame = np.asarray(stream["data"][frame])
        velocity, valid = _sample_camera_velocity(
            points,
            intrinsics[camera],
            invert_transform(extrinsics[camera]),
            velocity_frame,
            visibility_frame,
        )
        velocities.append(velocity)
        validities.append(valid)
    return np.stack(velocities, axis=1), np.stack(validities, axis=1)


def _source_reconstruction_targets(
    splat_paths: list[Path], frame_count: int, rng_seed: int
) -> list[np.ndarray]:
    return [
        seed_points_from_splat(
            splat_paths[frame],
            crop_half_extent_m=CROP_HALF_EXTENT_M,
            seed_count=SEED_POINT_COUNT,
            rng_seed=rng_seed,
        )[0]
        for frame in range(frame_count)
    ]


def _run_candidate(
    episode_dir: Path,
    cameras: list[str],
    intrinsics: dict[str, np.ndarray],
    extrinsics: dict[str, np.ndarray],
    initial_points: np.ndarray,
    colors: np.ndarray,
    targets: list[np.ndarray],
    default_points: list[np.ndarray],
    config: DenseVelocityFusionConfig,
    speed_limit_mps: float,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, np.ndarray]]]:
    points = initial_points.copy()
    springs = knn_springs(
        points,
        neighbors=config.graph_neighbors,
        radius_m=config.graph_radius_m,
    )
    frames: list[dict[str, np.ndarray]] = []
    metrics = []
    for frame, target in enumerate(targets):
        per_view, valid = _load_view_measurements(
            episode_dir,
            cameras,
            intrinsics,
            extrinsics,
            points,
            frame,
        )
        speed = np.linalg.norm(per_view, axis=2)
        prior_reliability = np.where(
            valid,
            1.0 / (1.0 + np.power(speed / speed_limit_mps, 4.0)),
            0.0,
        )
        observations = fuse_correlated_velocity_observations(
            per_view,
            valid,
            prior_reliability,
            config,
        )
        posterior = graph_complete_velocity(
            points,
            observations,
            config,
            springs=springs,
        )
        metric = {
            "frame": frame,
            "correlation_aware_chamfer_m": _symmetric_chamfer_m(points, target),
            "persistence_chamfer_m": _symmetric_chamfer_m(initial_points, target),
            "released_ransac_chamfer_m": _symmetric_chamfer_m(
                default_points[frame], target
            ),
            "direct_support_fraction": float(np.mean(posterior.directly_observed)),
            "mean_contributor_count": float(np.mean(posterior.contributor_count)),
            "mean_effective_sample_size": float(
                np.mean(posterior.effective_sample_size)
            ),
            "mean_velocity_mps": float(
                np.mean(np.linalg.norm(posterior.mean_mps, axis=1))
            ),
            "p99_velocity_mps": float(
                np.quantile(np.linalg.norm(posterior.mean_mps, axis=1), 0.99)
            ),
            "spatial_outlier_fraction": float(
                np.mean(posterior.spatial_robust_weight < 0.5)
            ),
        }
        metrics.append(metric)
        frames.append(
            {
                "pts": points.copy(),
                "colors": colors,
                "vels": posterior.mean_mps,
                "velocity_variance_m2ps2": posterior.observation_variance_m2ps2,
                "directly_observed": posterior.directly_observed,
                "contributor_count": posterior.contributor_count,
                "effective_sample_size": posterior.effective_sample_size,
                "prior_reliability": posterior.prior_reliability,
                "posterior_reliability": posterior.posterior_reliability,
                "spatial_robust_weight": posterior.spatial_robust_weight,
            }
        )
        points = (points + posterior.mean_mps / FRAME_RATE_HZ).astype(np.float32)

    candidate_dir = output_dir / (
        "lambda_" + str(config.graph_prior_strength).replace(".", "p")
    )
    candidate_dir.mkdir(parents=True, exist_ok=True)
    for frame, values in enumerate(frames):
        np.savez_compressed(candidate_dir / f"{frame:06d}.npz", **values)
    future = metrics[1:]
    summary = {
        "config": asdict(config),
        "speed_limit_mps": speed_limit_mps,
        "frames": metrics,
        "mean_future_correlation_aware_chamfer_m": float(
            np.mean([row["correlation_aware_chamfer_m"] for row in future])
        ),
        "mean_future_persistence_chamfer_m": float(
            np.mean([row["persistence_chamfer_m"] for row in future])
        ),
        "mean_future_released_ransac_chamfer_m": float(
            np.mean([row["released_ransac_chamfer_m"] for row in future])
        ),
        "mean_direct_support_fraction": float(
            np.mean([row["direct_support_fraction"] for row in metrics])
        ),
        "output_dir": str(candidate_dir.resolve()),
    }
    return summary, frames


def _write_final_data(
    baseline_path: Path,
    output_path: Path,
    frames: list[dict[str, np.ndarray]],
) -> None:
    with baseline_path.open("rb") as stream:
        payload = pickle.load(stream)
    if len(payload["object_points"]) != len(frames):
        raise ValueError(
            "baseline final_data frame count differs from correlation-aware frames"
        )
    payload["object_points"] = np.stack([frame["pts"] for frame in frames])
    payload["object_colors"] = np.stack([frame["colors"] for frame in frames])
    with output_path.open("wb") as stream:
        pickle.dump(payload, stream, protocol=4)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--graph-prior-strength", type=float, action="append")
    parser.add_argument("--speed-limit-multiplier", type=float, default=2.5)
    parser.add_argument("--minimum-speed-limit-mps", type=float, default=0.5)
    parser.add_argument("--rng-seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = args.episode_dir / "dense_source_smoke.manifest.json"
    source_boundary = json.loads(manifest.read_text(encoding="utf-8"))
    if not source_boundary.get("source_only"):
        raise ValueError("dense fusion accepts only a source-only staged episode")
    baseline_final_data = args.episode_dir / "final_data.pkl"
    if not baseline_final_data.exists():
        raise FileNotFoundError("run the released control-points stage first")

    cameras = episode_cameras(args.episode_dir)
    intrinsics, extrinsics = load_episode_calibration(args.episode_dir)
    num_frames = camera_frame_count(args.episode_dir, cameras[0])
    splat_paths = list_episode_splats(args.episode_dir, num_frames)
    frame_count = num_frames - TAIL_FRAMES_SKIPPED
    initial_points, colors = seed_points_from_splat(
        splat_paths[0],
        crop_half_extent_m=CROP_HALF_EXTENT_M,
        seed_count=SEED_POINT_COUNT,
        rng_seed=args.rng_seed,
    )
    targets = _source_reconstruction_targets(splat_paths, frame_count, args.rng_seed)
    default_paths = sorted((args.episode_dir / "pcd_clean").glob("*.npz"))
    if len(default_paths) != frame_count:
        raise ValueError("released pcd frame count is inconsistent")
    default_points = [np.load(path)["pts"] for path in default_paths]
    maximum_actuator_speed = _maximum_actuator_speed_mps(args.episode_dir)
    speed_limit = max(
        args.minimum_speed_limit_mps,
        args.speed_limit_multiplier * maximum_actuator_speed,
    )
    strengths = args.graph_prior_strength or [0.1, 1.0, 10.0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    candidate_frames: list[list[dict[str, np.ndarray]]] = []
    for strength in strengths:
        config = DenseVelocityFusionConfig(graph_prior_strength=strength)
        summary, frames = _run_candidate(
            args.episode_dir,
            cameras,
            intrinsics,
            extrinsics,
            initial_points,
            colors,
            targets,
            default_points,
            config,
            speed_limit,
            args.output_dir,
        )
        candidates.append(summary)
        candidate_frames.append(frames)
    best_index = int(
        np.argmin(
            [row["mean_future_correlation_aware_chamfer_m"] for row in candidates]
        )
    )
    best = candidates[best_index]
    best_frames = candidate_frames[best_index]
    final_data_path = args.output_dir / "final_data_correlation_aware.pkl"
    _write_final_data(baseline_final_data, final_data_path, best_frames)
    best_dir = Path(best["output_dir"])
    selected_dir = args.output_dir / "selected_pcd"
    if selected_dir.exists():
        shutil.rmtree(selected_dir)
    shutil.copytree(best_dir, selected_dir)

    payload = {
        "schema": "bayesian-phystwin/deform360-correlation-aware-pcd/v1",
        "source_only": True,
        "source_boundary_sha256": sha256_file(manifest),
        "episode_dir": str(args.episode_dir.resolve()),
        "camera_count": len(cameras),
        "frame_count": frame_count,
        "maximum_actuator_speed_mps": maximum_actuator_speed,
        "speed_limit_mps": speed_limit,
        "candidates": candidates,
        "selected_candidate_index": best_index,
        "selected_config": best["config"],
        "selected_improvement_vs_persistence_percent": float(
            100.0
            * (
                best["mean_future_correlation_aware_chamfer_m"]
                / best["mean_future_persistence_chamfer_m"]
                - 1.0
            )
        ),
        "selected_improvement_vs_released_ransac_percent": float(
            100.0
            * (
                best["mean_future_correlation_aware_chamfer_m"]
                / best["mean_future_released_ransac_chamfer_m"]
                - 1.0
            )
        ),
        "final_data_sha256": sha256_file(final_data_path),
    }
    payload["result_sha256"] = _result_sha256(payload)
    output_path = args.output_dir / "correlation_aware_pcd.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
