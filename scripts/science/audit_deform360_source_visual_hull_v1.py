#!/usr/bin/env python3
"""Audit source-only Deform360 geometry by held-out-camera visual-hull reprojection.

The registered source episode is split by camera calibration alone: spatially
dispersed held-out cameras are never used during voxel carving. A visual hull
is fitted independently at preregistered tactile-contact frames from the
remaining cameras and projected into the held-out views. Correct calibration
is compared with a deterministic local-camera yaw perturbation control.

This is a source geometry competence diagnostic. It does not open a target,
train Splatfacto, identify physical parameters, or authorize a paper claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _load_camera_dict(path: Path) -> dict[str, np.ndarray]:
    """Load a trusted first-party scalar NumPy camera dictionary."""
    loaded = np.load(path, allow_pickle=True)
    try:
        value = loaded.item()
    finally:
        if hasattr(loaded, "close"):
            loaded.close()
    if not isinstance(value, dict):
        raise ValueError(f"expected a camera dictionary in {path}")
    result: dict[str, np.ndarray] = {}
    for camera, matrix in value.items():
        if not isinstance(camera, str) or not camera:
            raise ValueError(f"invalid camera key in {path}")
        result[camera] = np.asarray(matrix, dtype=np.float64)
    return result


def _validate_intrinsics(
    intrinsics: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for camera, value in intrinsics.items():
        matrix = np.asarray(value, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError(f"invalid finite (3,3) intrinsics for {camera}")
        if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
            raise ValueError(f"non-positive focal length for {camera}")
        if not np.allclose(
            matrix[2], (0.0, 0.0, 1.0), rtol=0.0, atol=1e-8
        ):
            raise ValueError(f"invalid homogeneous intrinsics row for {camera}")
        result[camera] = matrix
    return result


def _validate_extrinsics(
    extrinsics: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for camera, value in extrinsics.items():
        matrix = np.asarray(value, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError(f"invalid finite (4,4) extrinsics for {camera}")
        if not np.allclose(
            matrix[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=1e-8
        ):
            raise ValueError(f"invalid homogeneous extrinsics row for {camera}")
        rotation = matrix[:3, :3]
        if not np.allclose(
            rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-6
        ):
            raise ValueError(f"non-orthonormal camera rotation for {camera}")
        if not np.isclose(
            np.linalg.det(rotation), 1.0, rtol=0.0, atol=1e-6
        ):
            raise ValueError(f"camera rotation determinant is not +1 for {camera}")
        result[camera] = matrix
    return result


def _camera_centers(
    extrinsics: Mapping[str, np.ndarray],
    cameras: Sequence[str],
) -> np.ndarray:
    return np.stack(
        [
            np.asarray(extrinsics[camera], dtype=np.float64)[:3, 3]
            for camera in cameras
        ]
    )


def _farthest_point_holdout(
    extrinsics: Mapping[str, np.ndarray],
    cameras: Sequence[str],
    *,
    holdout_count: int,
) -> tuple[list[str], list[str], list[str]]:
    """Select spatially dispersed held-out cameras without using masks."""
    names = sorted(dict.fromkeys(cameras))
    if holdout_count <= 0 or holdout_count >= len(names):
        raise ValueError("holdout_count must be between 1 and camera_count - 1")
    centers = _camera_centers(extrinsics, names)
    rig_center = np.mean(centers, axis=0)
    center_distances = np.linalg.norm(centers - rig_center, axis=1)
    first = max(
        range(len(names)),
        key=lambda index: (float(center_distances[index]), -index),
    )
    selected = [first]
    while len(selected) < holdout_count:
        selected_centers = centers[np.asarray(selected, dtype=np.int64)]
        pairwise = np.linalg.norm(
            centers[:, None, :] - selected_centers[None, :, :], axis=2
        )
        minimum_distance = np.min(pairwise, axis=1)
        minimum_distance[np.asarray(selected, dtype=np.int64)] = -np.inf
        next_index = max(
            range(len(names)),
            key=lambda index: (float(minimum_distance[index]), -index),
        )
        selected.append(next_index)
    holdout_order = [names[index] for index in selected]
    holdout_set = set(holdout_order)
    training = [name for name in names if name not in holdout_set]
    return training, sorted(holdout_set), holdout_order


def _perturb_camera_yaw(
    camera_to_world: np.ndarray,
    *,
    yaw_degrees: float,
) -> np.ndarray:
    """Apply a deterministic local-camera yaw error without using mask labels."""
    angle = np.deg2rad(float(yaw_degrees))
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    local_yaw = np.array(
        [
            [cosine, 0.0, sine],
            [0.0, 1.0, 0.0],
            [-sine, 0.0, cosine],
        ],
        dtype=np.float64,
    )
    perturbed = np.asarray(camera_to_world, dtype=np.float64).copy()
    perturbed[:3, :3] = perturbed[:3, :3] @ local_yaw
    return perturbed


def _project_points(
    points_world: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project row-vector world points with Deform360 c2w calibration."""
    points = np.asarray(points_world, dtype=np.float64)
    K = np.asarray(intrinsics, dtype=np.float64)
    c2w = np.asarray(camera_to_world, dtype=np.float64)
    height, width = image_shape
    rotation = c2w[:3, :3]
    translation = c2w[:3, 3]
    camera_points = (points - translation) @ rotation
    depth = camera_points[:, 2]
    in_front = depth > 1e-8
    u = np.zeros(len(points), dtype=np.float64)
    v = np.zeros(len(points), dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        u[in_front] = (
            camera_points[in_front, 0] / depth[in_front] * K[0, 0]
            + K[0, 2]
        )
        v[in_front] = (
            camera_points[in_front, 1] / depth[in_front] * K[1, 1]
            + K[1, 2]
        )
    u_pixel = np.floor(u).astype(np.int64)
    v_pixel = np.floor(v).astype(np.int64)
    valid = (
        in_front
        & np.isfinite(u)
        & np.isfinite(v)
        & (u_pixel >= 0)
        & (u_pixel < width)
        & (v_pixel >= 0)
        & (v_pixel < height)
    )
    return u_pixel, v_pixel, depth, valid


def _binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return mask.copy()
    padded = np.pad(
        mask.astype(np.uint8), ((radius, radius), (radius, radius))
    )
    integral = np.pad(padded, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    width = 2 * radius + 1
    window_sum = (
        integral[width:, width:]
        - integral[:-width, width:]
        - integral[width:, :-width]
        + integral[:-width, :-width]
    )
    return window_sum > 0


def _binary_erode(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return mask.copy()
    padded = np.pad(
        mask.astype(np.uint8),
        ((radius, radius), (radius, radius)),
        constant_values=0,
    )
    integral = np.pad(padded, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    width = 2 * radius + 1
    window_sum = (
        integral[width:, width:]
        - integral[:-width, width:]
        - integral[width:, :-width]
        + integral[:-width, :-width]
    )
    return window_sum == width * width


def _render_points(
    points_world: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    image_shape: tuple[int, int],
    *,
    voxel_size_m: float,
    maximum_radius_px: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    u, v, depth, valid = _project_points(
        points_world, intrinsics, camera_to_world, image_shape
    )
    base = np.zeros(image_shape, dtype=bool)
    if np.any(valid):
        base[v[valid], u[valid]] = True
        median_depth = float(np.median(depth[valid]))
        footprint = (
            0.5
            * np.sqrt(3.0)
            * float(voxel_size_m)
            * max(float(intrinsics[0, 0]), float(intrinsics[1, 1]))
            / max(median_depth, 1e-8)
        )
        radius = int(
            np.clip(np.ceil(footprint), 1, max(int(maximum_radius_px), 1))
        )
    else:
        median_depth = float("nan")
        radius = 1
    rendered = _binary_dilate(base, radius)
    return rendered, {
        "projected_point_count": int(np.count_nonzero(valid)),
        "unique_projected_pixel_count_before_splat": int(
            np.count_nonzero(base)
        ),
        "median_projected_depth_m": (
            median_depth if np.isfinite(median_depth) else None
        ),
        "splat_radius_px": radius,
    }


def _boundary(mask: np.ndarray) -> np.ndarray:
    value = np.asarray(mask, dtype=bool)
    return value & ~_binary_erode(value, 1)


def _boundary_f1(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    tolerance_px: int,
) -> dict[str, float]:
    truth_boundary = _boundary(truth)
    prediction_boundary = _boundary(prediction)
    truth_count = int(np.count_nonzero(truth_boundary))
    prediction_count = int(np.count_nonzero(prediction_boundary))
    if truth_count == 0 or prediction_count == 0:
        return {
            "boundary_precision": 0.0,
            "boundary_recall": 0.0,
            "boundary_f1": 0.0,
        }
    truth_neighborhood = _binary_dilate(truth_boundary, tolerance_px)
    prediction_neighborhood = _binary_dilate(
        prediction_boundary, tolerance_px
    )
    precision = float(
        np.count_nonzero(prediction_boundary & truth_neighborhood)
        / prediction_count
    )
    recall = float(
        np.count_nonzero(truth_boundary & prediction_neighborhood) / truth_count
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    return {
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": f1,
    }


def _silhouette_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    boundary_tolerance_px: int,
) -> dict[str, float]:
    truth_mask = np.asarray(truth, dtype=bool)
    prediction_mask = np.asarray(prediction, dtype=bool)
    if truth_mask.shape != prediction_mask.shape:
        raise ValueError("truth and prediction shapes differ")
    truth_area = int(np.count_nonzero(truth_mask))
    prediction_area = int(np.count_nonzero(prediction_mask))
    intersection = int(np.count_nonzero(truth_mask & prediction_mask))
    union = int(np.count_nonzero(truth_mask | prediction_mask))
    precision = intersection / prediction_area if prediction_area else 0.0
    recall = intersection / truth_area if truth_area else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    metrics = {
        "iou": float(intersection / union) if union else 1.0,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "truth_area_px": float(truth_area),
        "prediction_area_px": float(prediction_area),
        "prediction_to_truth_area_ratio": float(
            prediction_area / max(truth_area, 1)
        ),
    }
    metrics.update(
        _boundary_f1(
            truth_mask,
            prediction_mask,
            tolerance_px=boundary_tolerance_px,
        )
    )
    return metrics


def _visual_hull_points(
    masks: Mapping[str, np.ndarray],
    intrinsics: Mapping[str, np.ndarray],
    extrinsics: Mapping[str, np.ndarray],
    *,
    grid_center_world: np.ndarray,
    cube_half_extent_m: float,
    voxel_resolution: int,
    minimum_hull_points: int,
    minimum_consensus_fraction: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Voxel-carve a visual hull using training masks only."""
    cameras = sorted(masks)
    if len(cameras) < 2:
        raise ValueError("at least two non-empty training masks are required")
    axis = np.linspace(
        -cube_half_extent_m,
        cube_half_extent_m,
        voxel_resolution,
        dtype=np.float64,
    )
    grid = (
        np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
        .reshape(-1, 3)
        + np.asarray(grid_center_world, dtype=np.float64)
    )
    hit_count = np.zeros(len(grid), dtype=np.uint16)
    for camera in cameras:
        mask = np.asarray(masks[camera], dtype=bool)
        u, v, _depth, valid = _project_points(
            grid,
            intrinsics[camera],
            extrinsics[camera],
            mask.shape,
        )
        hit = np.zeros(len(grid), dtype=bool)
        valid_indices = np.flatnonzero(valid)
        hit[valid_indices] = mask[v[valid_indices], u[valid_indices]]
        hit_count += hit.astype(np.uint16)

    minimum_required = max(
        1, int(np.ceil(minimum_consensus_fraction * len(cameras)))
    )
    required = len(cameras)
    hull_count = int(np.count_nonzero(hit_count >= required))
    while required > minimum_required and hull_count < minimum_hull_points:
        required -= 1
        hull_count = int(np.count_nonzero(hit_count >= required))
    if hull_count == 0:
        raise ValueError("visual hull is empty at the minimum consensus")
    hull = grid[hit_count >= required]
    voxel_size = 2.0 * float(cube_half_extent_m) / max(
        voxel_resolution - 1, 1
    )
    minimum = np.min(hull, axis=0)
    maximum = np.max(hull, axis=0)
    return hull.astype(np.float32), {
        "grid_center_world_m": np.asarray(
            grid_center_world, dtype=float
        ).tolist(),
        "cube_half_extent_m": float(cube_half_extent_m),
        "voxel_resolution": int(voxel_resolution),
        "voxel_size_m": float(voxel_size),
        "grid_point_count": int(len(grid)),
        "training_camera_count": int(len(cameras)),
        "required_consensus_views": int(required),
        "required_consensus_fraction": float(required / len(cameras)),
        "hull_point_count": int(len(hull)),
        "hull_min_world_m": minimum.astype(float).tolist(),
        "hull_max_world_m": maximum.astype(float).tolist(),
        "hull_bbox_diagonal_m": float(np.linalg.norm(maximum - minimum)),
    }


def _select_contact_frames(
    total_positive_signal: np.ndarray,
    quantiles: Iterable[float],
) -> list[int]:
    signal = np.asarray(total_positive_signal, dtype=np.float64)
    positive = np.flatnonzero(signal > 0.0)
    quantile_values = [float(value) for value in quantiles]
    if len(positive) < len(quantile_values):
        raise ValueError("too few positive tactile frames for frozen quantiles")
    selected = [
        int(positive[int(np.rint(value * (len(positive) - 1)))])
        for value in quantile_values
    ]
    if len(set(selected)) != len(selected):
        supplemental = np.linspace(
            0, len(positive) - 1, len(quantile_values), dtype=np.int64
        )
        selected = [int(positive[index]) for index in supplemental]
    if len(set(selected)) != len(selected):
        raise ValueError("contact frame quantiles are not unique")
    return selected


def _mask_frame(path: Path, frame_index: int) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise ModuleNotFoundError(
            "visual-hull auditing requires the optional h5py dependency"
        ) from exc
    with h5py.File(path, "r") as handle:
        if "data" not in handle:
            raise KeyError(f"{path} has no HDF5 dataset named 'data'")
        data = handle["data"]
        if data.ndim != 3:
            raise ValueError(f"{path} has shape {data.shape}, expected (T,H,W)")
        if frame_index < 0 or frame_index >= data.shape[0]:
            raise IndexError(f"frame {frame_index} is outside {path}")
        frame = np.asarray(data[frame_index])
        shape = [int(value) for value in data.shape]
        dtype = str(data.dtype)
    if frame.dtype not in (np.dtype(np.bool_), np.dtype(np.uint8)):
        raise TypeError(f"{path} mask dtype is {frame.dtype}, expected bool/uint8")
    if frame.dtype == np.dtype(np.uint8) and np.any(
        (frame != 0) & (frame != 1)
    ):
        raise ValueError(f"{path} uint8 mask is not binary")
    mask = frame.astype(bool, copy=False)
    return mask, {
        "carrier_shape": shape,
        "carrier_dtype": dtype,
        "selected_frame_mask_sha256": _array_sha256(mask),
        "selected_frame_area_px": int(np.count_nonzero(mask)),
    }


def _tactile_total(
    root: Path,
    sensor_names: Sequence[str],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    total = None
    records = []
    frame_count = None
    for sensor in sensor_names:
        path = root / sensor / "synced_tactile.npy"
        values = np.load(path, allow_pickle=False, mmap_mode="r")
        if values.ndim != 3 or values.shape[1:] != (16, 32):
            raise ValueError(f"unexpected tactile shape for {sensor}: {values.shape}")
        if frame_count is None:
            frame_count = int(values.shape[0])
        elif int(values.shape[0]) != frame_count:
            raise ValueError("tactile frame counts disagree")
        positive = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
        signal = positive.sum(axis=(1, 2))
        total = signal if total is None else total + signal
        records.append(
            {
                "sensor": sensor,
                "path": str(path),
                "shape": [int(value) for value in values.shape],
                "dtype": str(values.dtype),
                "positive_frame_count": int(np.count_nonzero(signal > 0.0)),
            }
        )
    if total is None:
        raise ValueError("no tactile sensors configured")
    return total, records


def _aggregate(
    records: Sequence[Mapping[str, Any]], key: str
) -> dict[str, float]:
    values = np.asarray(
        [float(record[key]) for record in records], dtype=np.float64
    )
    if len(values) == 0:
        raise ValueError(f"no values available for {key}")
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p25": float(np.percentile(values, 25.0)),
        "p75": float(np.percentile(values, 75.0)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def _camera_block_bootstrap(
    records: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    for record in records:
        grouped.setdefault(str(record["camera"]), []).append(
            float(record[metric])
        )
    cameras = sorted(grouped)
    if len(cameras) < 2:
        raise ValueError("camera-block bootstrap requires at least two cameras")
    camera_means = np.asarray(
        [np.mean(grouped[camera]) for camera in cameras], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sample = rng.integers(0, len(cameras), size=len(cameras))
        draws[index] = float(np.mean(camera_means[sample]))
    return {
        "metric": metric,
        "block": "held-out-camera",
        "camera_count": len(cameras),
        "replicates": int(replicates),
        "seed": int(seed),
        "estimate": float(np.mean(camera_means)),
        "ci95": [
            float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)),
        ],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-episode-root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--workflow-run-id", required=True, type=int)
    parser.add_argument("--workflow-run-attempt", required=True, type=int)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.source_episode_root.resolve(strict=True)
    protocol_path = args.protocol.resolve(strict=True)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    observed = root.parent.name, int(root.name.split("_")[-1])
    expected = protocol["source_object"], int(protocol["source_episode"])
    if observed != expected:
        raise ValueError(f"source episode {observed} differs from {expected}")

    intrinsics_path = root / "undistorted_intrinsics.npy"
    extrinsics_path = root / "extrinsics.npy"
    intrinsics = _validate_intrinsics(_load_camera_dict(intrinsics_path))
    extrinsics = _validate_extrinsics(_load_camera_dict(extrinsics_path))
    candidate_cameras = sorted(
        camera
        for camera in set(intrinsics) & set(extrinsics)
        if (root / camera / "mask_refined.h5").is_file()
    )
    split = protocol["camera_split"]
    holdout_count = int(
        np.clip(
            np.rint(
                len(candidate_cameras) * float(split["holdout_fraction"])
            ),
            int(split["minimum_holdout_cameras"]),
            int(split["maximum_holdout_cameras"]),
        )
    )
    training_cameras, heldout_cameras, holdout_order = (
        _farthest_point_holdout(
            extrinsics,
            candidate_cameras,
            holdout_count=holdout_count,
        )
    )
    yaw_perturbation_degrees = float(
        protocol["controls"]["camera_local_yaw_perturbation_degrees"]
    )
    rig_center = np.mean(
        _camera_centers(extrinsics, candidate_cameras), axis=0
    )

    tactile, tactile_records = _tactile_total(
        root, protocol["frame_selection"]["tactile_sensors"]
    )
    frames = _select_contact_frames(
        tactile, protocol["frame_selection"]["positive_signal_quantiles"]
    )

    visual = protocol["visual_hull"]
    projection = protocol["projection"]
    per_frame = []
    training_records: list[dict[str, Any]] = []
    heldout_records: list[dict[str, Any]] = []
    control_records: list[dict[str, Any]] = []
    selected_mask_records = []
    minimum_active_training = int(
        protocol["qualification_gates"][
            "minimum_active_training_cameras_per_frame"
        ]
    )
    minimum_active_heldout = int(
        protocol["qualification_gates"][
            "minimum_active_heldout_cameras_per_frame"
        ]
    )

    for frame in frames:
        masks: dict[str, np.ndarray] = {}
        frame_input_records = []
        for camera in candidate_cameras:
            mask, record = _mask_frame(
                root / camera / "mask_refined.h5", frame
            )
            masks[camera] = mask
            record.update({"camera": camera, "frame": frame})
            frame_input_records.append(record)
        active_training = [
            camera for camera in training_cameras if np.any(masks[camera])
        ]
        active_heldout = [
            camera for camera in heldout_cameras if np.any(masks[camera])
        ]
        if len(active_training) < minimum_active_training:
            raise ValueError(
                f"frame {frame} has only {len(active_training)} active "
                "training cameras"
            )
        if len(active_heldout) < minimum_active_heldout:
            raise ValueError(
                f"frame {frame} has only {len(active_heldout)} active "
                "held-out cameras"
            )

        hull, hull_record = _visual_hull_points(
            {camera: masks[camera] for camera in active_training},
            intrinsics,
            extrinsics,
            grid_center_world=rig_center,
            cube_half_extent_m=float(visual["cube_half_extent_m"]),
            voxel_resolution=int(visual["voxel_resolution"]),
            minimum_hull_points=int(visual["minimum_hull_points"]),
            minimum_consensus_fraction=float(
                visual["minimum_consensus_fraction"]
            ),
        )
        frame_training_records = []
        for camera in active_training:
            prediction, render_record = _render_points(
                hull,
                intrinsics[camera],
                extrinsics[camera],
                masks[camera].shape,
                voxel_size_m=float(hull_record["voxel_size_m"]),
                maximum_radius_px=int(
                    projection["maximum_splat_radius_px"]
                ),
            )
            metrics = _silhouette_metrics(
                masks[camera],
                prediction,
                boundary_tolerance_px=int(
                    projection["boundary_tolerance_px"]
                ),
            )
            record = {
                "frame": frame,
                "camera": camera,
                **metrics,
                **render_record,
            }
            training_records.append(record)
            frame_training_records.append(record)

        frame_heldout_records = []
        frame_control_records = []
        for camera in active_heldout:
            prediction, render_record = _render_points(
                hull,
                intrinsics[camera],
                extrinsics[camera],
                masks[camera].shape,
                voxel_size_m=float(hull_record["voxel_size_m"]),
                maximum_radius_px=int(
                    projection["maximum_splat_radius_px"]
                ),
            )
            metrics = _silhouette_metrics(
                masks[camera],
                prediction,
                boundary_tolerance_px=int(
                    projection["boundary_tolerance_px"]
                ),
            )
            record = {
                "frame": frame,
                "camera": camera,
                **metrics,
                **render_record,
            }
            heldout_records.append(record)
            frame_heldout_records.append(record)

            perturbed_extrinsics = _perturb_camera_yaw(
                extrinsics[camera],
                yaw_degrees=yaw_perturbation_degrees,
            )
            perturbed_prediction, perturbed_render = _render_points(
                hull,
                intrinsics[camera],
                perturbed_extrinsics,
                masks[camera].shape,
                voxel_size_m=float(hull_record["voxel_size_m"]),
                maximum_radius_px=int(
                    projection["maximum_splat_radius_px"]
                ),
            )
            perturbed_metrics = _silhouette_metrics(
                masks[camera],
                perturbed_prediction,
                boundary_tolerance_px=int(
                    projection["boundary_tolerance_px"]
                ),
            )
            control_record = {
                "frame": frame,
                "camera": camera,
                "camera_local_yaw_perturbation_degrees": (
                    yaw_perturbation_degrees
                ),
                **perturbed_metrics,
                **perturbed_render,
            }
            control_records.append(control_record)
            frame_control_records.append(control_record)

        per_frame.append(
            {
                "frame": frame,
                "active_training_cameras": active_training,
                "active_heldout_cameras": active_heldout,
                "inactive_training_cameras": sorted(
                    set(training_cameras) - set(active_training)
                ),
                "inactive_heldout_cameras": sorted(
                    set(heldout_cameras) - set(active_heldout)
                ),
                "visual_hull": hull_record,
                "training_iou": _aggregate(
                    frame_training_records, "iou"
                ),
                "heldout_iou": _aggregate(frame_heldout_records, "iou"),
                "perturbed_extrinsic_iou": _aggregate(
                    frame_control_records, "iou"
                ),
            }
        )
        selected_mask_records.extend(frame_input_records)

    training_iou = _aggregate(training_records, "iou")
    heldout_iou = _aggregate(heldout_records, "iou")
    heldout_recall = _aggregate(heldout_records, "recall")
    heldout_precision = _aggregate(heldout_records, "precision")
    heldout_boundary = _aggregate(heldout_records, "boundary_f1")
    heldout_area_ratio = _aggregate(
        heldout_records, "prediction_to_truth_area_ratio"
    )
    control_iou = _aggregate(control_records, "iou")
    calibration_gain = float(
        heldout_iou["median"] - control_iou["median"]
    )
    bootstrap = _camera_block_bootstrap(
        heldout_records,
        metric="iou",
        replicates=int(protocol["bootstrap"]["replicates"]),
        seed=int(protocol["bootstrap"]["seed"]),
    )

    gates = protocol["qualification_gates"]
    minimum_consensus = min(
        float(frame["visual_hull"]["required_consensus_fraction"])
        for frame in per_frame
    )
    checks = {
        "minimum_candidate_camera_count": len(candidate_cameras)
        >= int(gates["minimum_candidate_camera_count"]),
        "minimum_training_camera_count": len(training_cameras)
        >= int(gates["minimum_training_camera_count"]),
        "minimum_heldout_camera_count": len(heldout_cameras)
        >= int(gates["minimum_heldout_camera_count"]),
        "minimum_evaluated_frame_count": len(frames)
        >= int(gates["minimum_evaluated_frame_count"]),
        "minimum_heldout_frame_camera_pairs": len(heldout_records)
        >= int(gates["minimum_heldout_frame_camera_pairs"]),
        "training_reprojection_iou": training_iou["median"]
        >= float(gates["minimum_median_training_iou"]),
        "heldout_reprojection_iou": heldout_iou["median"]
        >= float(gates["minimum_median_heldout_iou"]),
        "heldout_iou_lower_quartile": heldout_iou["p25"]
        >= float(gates["minimum_p25_heldout_iou"]),
        "heldout_boundary_f1": heldout_boundary["median"]
        >= float(gates["minimum_median_heldout_boundary_f1"]),
        "heldout_recall": heldout_recall["median"]
        >= float(gates["minimum_median_heldout_recall"]),
        "correct_calibration_beats_perturbation": calibration_gain
        >= float(gates["minimum_correct_vs_perturbed_iou_gain"]),
        "camera_block_bootstrap_lower_bound": bootstrap["ci95"][0]
        >= float(
            gates["minimum_camera_block_bootstrap_iou_lower_bound"]
        ),
        "prediction_area_ratio_not_too_small": heldout_area_ratio["median"]
        >= float(
            gates["minimum_median_prediction_to_truth_area_ratio"]
        ),
        "prediction_area_ratio_not_too_large": heldout_area_ratio["median"]
        <= float(
            gates["maximum_median_prediction_to_truth_area_ratio"]
        ),
        "minimum_consensus_fraction": minimum_consensus
        >= float(gates["minimum_required_consensus_fraction"]),
    }
    qualified = all(checks.values())

    result: dict[str, Any] = {
        "schema": "bayesian-phystwin/deform360-source-visual-hull-v1",
        "repository": args.repository,
        "revision": args.revision,
        "workflow_run_id": args.workflow_run_id,
        "workflow_run_attempt": args.workflow_run_attempt,
        "runner_name": os.environ.get("RUNNER_NAME"),
        "required_runner_label": "gpuserver4090",
        "source_object": protocol["source_object"],
        "source_episode": int(protocol["source_episode"]),
        "source_episode_root": str(root),
        "official_deform360_reference_revision": protocol[
            "official_deform360_reference_revision"
        ],
        "protocol_sha256": _sha256(protocol_path),
        "implementation_sha256": _sha256(Path(__file__).resolve()),
        "calibration": {
            "intrinsics_file_sha256": _sha256(intrinsics_path),
            "extrinsics_file_sha256": _sha256(extrinsics_path),
            "rig_center_world_m": rig_center.astype(float).tolist(),
        },
        "camera_split": {
            "method": split["method"],
            "candidate_cameras": candidate_cameras,
            "training_cameras": training_cameras,
            "heldout_cameras": heldout_cameras,
            "heldout_selection_order": holdout_order,
            "camera_local_yaw_perturbation_degrees": (
                yaw_perturbation_degrees
            ),
        },
        "frame_selection": {
            "method": protocol["frame_selection"]["method"],
            "selected_frames": frames,
            "positive_tactile_frame_count": int(
                np.count_nonzero(tactile > 0.0)
            ),
            "tactile_signal_sha256": _array_sha256(tactile),
            "tactile_records": tactile_records,
        },
        "selected_mask_inputs": selected_mask_records,
        "per_frame": per_frame,
        "aggregate": {
            "training_iou": training_iou,
            "heldout_iou": heldout_iou,
            "heldout_precision": heldout_precision,
            "heldout_recall": heldout_recall,
            "heldout_boundary_f1": heldout_boundary,
            "heldout_prediction_to_truth_area_ratio": heldout_area_ratio,
            "perturbed_extrinsic_iou": control_iou,
            "correct_vs_perturbed_extrinsic_median_iou_gain": (
                calibration_gain
            ),
            "heldout_camera_block_bootstrap": bootstrap,
        },
        "qualification_checks": checks,
        "source_visual_hull_qualified": qualified,
        "decision": (
            "source-visual-hull-qualified"
            if qualified
            else "source-visual-hull-not-qualified"
        ),
        "information_boundary": {
            "source_calibration_opened": True,
            "source_object_masks_opened": True,
            "source_camera_pixels_opened": False,
            "source_splatfacto_training_performed": False,
            "persistent_dataset_write_performed": False,
            "target_directory_contents_listed": False,
            "target_numeric_payload_opened": False,
            "target_scoring_performed": False,
            "fresh_confirmation_authorized": False,
            "paper_claim_authorized": False,
        },
        "statistical_scope": protocol["statistical_scope"],
        "claim_boundary": protocol["claim_boundary"],
    }
    canonical = json.dumps(
        result, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    result["result_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    _write_json(output / "result.json", result)
    (output / "report.md").write_text(
        "# Deform360 source visual-hull gate v1\n\n"
        f"Decision: `{result['decision']}`\n\n"
        f"Frames: `{frames}`\n\n"
        f"Cameras: `{len(training_cameras)}` training / "
        f"`{len(heldout_cameras)}` held out\n\n"
        f"Median training IoU: `{training_iou['median']:.6f}`\n\n"
        f"Median held-out IoU: `{heldout_iou['median']:.6f}`\n\n"
        f"Held-out IoU p25: `{heldout_iou['p25']:.6f}`\n\n"
        f"Median held-out boundary F1: "
        f"`{heldout_boundary['median']:.6f}`\n\n"
        f"Correct-minus-perturbed-extrinsic median IoU: "
        f"`{calibration_gain:.6f}`\n\n"
        f"Held-out camera-block mean-IoU 95% interval: "
        f"`[{bootstrap['ci95'][0]:.6f}, {bootstrap['ci95'][1]:.6f}]`\n\n"
        "This is a single-source-episode multiview geometry diagnostic; "
        "it is not an independent-object result or paper claim.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
