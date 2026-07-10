"""Reproduce the official PhysTwin 3D evaluation metrics."""

from __future__ import annotations

import hashlib
import json
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np


def _load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nearest_distances(
    reference: np.ndarray,
    query: np.ndarray,
    *,
    p: int,
    chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest distances and indices, preferring SciPy when available."""

    reference_array = np.asarray(reference, dtype=float)
    query_array = np.asarray(query, dtype=float)
    if reference_array.ndim != 2 or reference_array.shape[1] != 3:
        raise ValueError("reference must have shape (N, 3)")
    if query_array.ndim != 2 or query_array.shape[1] != 3:
        raise ValueError("query must have shape (M, 3)")
    if len(reference_array) == 0:
        raise ValueError("reference must contain at least one point")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            from scipy.spatial import cKDTree
    except (ImportError, OSError, ValueError, Warning):
        cKDTree = None

    if cKDTree is not None:
        distances, indices = cKDTree(reference_array).query(query_array, p=p)
        return np.asarray(distances), np.asarray(indices, dtype=np.int64)

    distances = np.empty(len(query_array), dtype=float)
    indices = np.empty(len(query_array), dtype=np.int64)
    for start in range(0, len(query_array), chunk_size):
        stop = min(start + chunk_size, len(query_array))
        delta = np.abs(
            query_array[start:stop, None, :] - reference_array[None, :, :]
        )
        pairwise = (
            np.sum(delta, axis=2)
            if p == 1
            else np.sqrt(np.sum(delta * delta, axis=2))
        )
        local_indices = np.argmin(pairwise, axis=1)
        indices[start:stop] = local_indices
        distances[start:stop] = pairwise[
            np.arange(stop - start), local_indices
        ]
    return distances, indices


def _validate_arrays(
    vertices: np.ndarray,
    object_points: np.ndarray,
    object_visibilities: np.ndarray,
    gt_track_3d: np.ndarray,
    *,
    test_frame: int,
) -> None:
    if vertices.ndim != 3 or vertices.shape[2] != 3:
        raise ValueError("vertices must have shape (T, N, 3)")
    if object_points.ndim != 3 or object_points.shape[2] != 3:
        raise ValueError("object_points must have shape (T, M, 3)")
    if object_visibilities.shape != object_points.shape[:2]:
        raise ValueError("object_visibilities must have shape (T, M)")
    if gt_track_3d.ndim != 3 or gt_track_3d.shape[2] != 3:
        raise ValueError("gt_track_3d must have shape (T, K, 3)")
    if min(vertices.shape[0], object_points.shape[0], gt_track_3d.shape[0]) < test_frame:
        raise ValueError("all inputs must cover the complete evaluation split")


def evaluate_official_phystwin_interval(
    vertices: np.ndarray,
    object_points: np.ndarray,
    object_visibilities: np.ndarray,
    gt_track_3d: np.ndarray,
    *,
    num_surface_points: int,
    start_frame: int,
    end_frame: int,
) -> dict[str, object]:
    """Evaluate one half-open interval with the released 3D metric contract."""

    vertices_array = np.asarray(vertices, dtype=float)
    points_array = np.asarray(object_points, dtype=float)
    visibility_array = np.asarray(object_visibilities, dtype=bool)
    tracks_array = np.asarray(gt_track_3d, dtype=float)
    _validate_arrays(
        vertices_array,
        points_array,
        visibility_array,
        tracks_array,
        test_frame=end_frame,
    )
    if not 0 <= start_frame < end_frame:
        raise ValueError("expected 0 <= start_frame < end_frame")
    if not 1 <= num_surface_points <= vertices_array.shape[1]:
        raise ValueError("num_surface_points exceeds the predicted state")

    initial_track_mask = np.isfinite(tracks_array[0]).all(axis=1)
    _, track_indices = _nearest_distances(
        vertices_array[0],
        tracks_array[0, initial_track_mask],
        p=2,
    )
    chamfer_by_frame: list[float] = []
    track_by_frame: list[float] = []
    for frame_index in range(start_frame, end_frame):
        observed = points_array[frame_index, visibility_array[frame_index]]
        predicted_surface = vertices_array[frame_index, :num_surface_points]
        chamfer_distances, _ = _nearest_distances(
            predicted_surface,
            observed,
            p=1,
        )
        chamfer_by_frame.append(float(np.mean(chamfer_distances)))

        current_tracks = tracks_array[frame_index, initial_track_mask]
        current_mask = np.isfinite(current_tracks).all(axis=1)
        if np.any(current_mask):
            predicted_tracks = vertices_array[frame_index, track_indices][current_mask]
            residual = predicted_tracks - current_tracks[current_mask]
            track_by_frame.append(float(np.mean(np.linalg.norm(residual, axis=1))))
        else:
            track_by_frame.append(0.0)
    return {
        "frame_start": start_frame,
        "frame_end_exclusive": end_frame,
        "frame_count": end_frame - start_frame,
        "chamfer_distance_m": float(np.mean(chamfer_by_frame)),
        "track_error_m": float(np.mean(track_by_frame)),
    }


def evaluate_official_phystwin_arrays(
    vertices: np.ndarray,
    object_points: np.ndarray,
    object_visibilities: np.ndarray,
    gt_track_3d: np.ndarray,
    *,
    num_surface_points: int,
    train_frame: int,
    test_frame: int,
) -> dict[str, object]:
    """Evaluate arrays using the released PhysTwin CD and track definitions."""

    vertices_array = np.asarray(vertices, dtype=float)
    points_array = np.asarray(object_points, dtype=float)
    visibility_array = np.asarray(object_visibilities, dtype=bool)
    tracks_array = np.asarray(gt_track_3d, dtype=float)
    _validate_arrays(
        vertices_array,
        points_array,
        visibility_array,
        tracks_array,
        test_frame=test_frame,
    )
    if not 1 < train_frame < test_frame:
        raise ValueError("expected 1 < train_frame < test_frame")
    if not 1 <= num_surface_points <= vertices_array.shape[1]:
        raise ValueError("num_surface_points exceeds the predicted state")

    return {
        "train": evaluate_official_phystwin_interval(
            vertices_array,
            points_array,
            visibility_array,
            tracks_array,
            num_surface_points=num_surface_points,
            start_frame=1,
            end_frame=train_frame,
        ),
        "test": evaluate_official_phystwin_interval(
            vertices_array,
            points_array,
            visibility_array,
            tracks_array,
            num_surface_points=num_surface_points,
            start_frame=train_frame,
            end_frame=test_frame,
        ),
    }


def evaluate_official_phystwin_files(
    trajectory_path: str | Path,
    final_data_path: str | Path,
    gt_track_path: str | Path,
    split_path: str | Path,
) -> dict[str, object]:
    """Load official artifacts and return metrics with input provenance."""

    vertices = np.asarray(_load_pickle(trajectory_path))
    final_data = _load_pickle(final_data_path)
    gt_track_3d = np.asarray(_load_pickle(gt_track_path))
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    num_surface_points = (
        int(np.asarray(final_data["object_points"]).shape[1])
        + int(np.asarray(final_data["surface_points"]).shape[0])
    )
    evaluation = evaluate_official_phystwin_arrays(
        vertices,
        final_data["object_points"],
        final_data["object_visibilities"],
        gt_track_3d,
        num_surface_points=num_surface_points,
        train_frame=int(split["train"][1]),
        test_frame=int(split["test"][1]),
    )
    return {
        "schema_version": 1,
        "metric_contract": {
            "chamfer_distance_m": (
                "Per-frame mean one-way L1 nearest-neighbor distance from visible "
                "observed object points to predicted surface points, then frame mean."
            ),
            "track_error_m": (
                "Frame-0 nearest-vertex correspondence for finite manual tracks; "
                "per-frame mean Euclidean error, then frame mean."
            ),
            "compatibility": (
                "Matches the released PhysTwin evaluate_chamfer.py and "
                "evaluate_track.py definitions."
            ),
        },
        "inputs": {
            name: {
                "path": str(Path(path).resolve()),
                "sha256": _sha256(path),
            }
            for name, path in {
                "trajectory": trajectory_path,
                "final_data": final_data_path,
                "gt_track_3d": gt_track_path,
                "split": split_path,
            }.items()
        },
        "split": split,
        "evaluation": evaluation,
    }


def write_official_evaluation(summary: dict[str, object], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
