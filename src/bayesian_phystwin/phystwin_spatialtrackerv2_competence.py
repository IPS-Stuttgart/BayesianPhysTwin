"""SpatialTrackerV2 adapters for a prefix-only PhysTwin competence control."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SpatialTrackerV2Prediction:
    """Canonical calibrated world-space tracker output."""

    coords_world_m: np.ndarray
    valid: np.ndarray
    visibility_probability: np.ndarray
    confidence: np.ndarray
    query_points: np.ndarray
    query_pixels_xyt: np.ndarray


def project_world_queries_to_pixels(
    query_positions_world_m: np.ndarray,
    intrinsics: np.ndarray,
    world_to_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project frame-zero world queries into one calibrated RGB-D camera."""

    queries = np.asarray(query_positions_world_m, dtype=float)
    camera_matrix = np.asarray(intrinsics, dtype=float)
    extrinsic = np.asarray(world_to_camera, dtype=float)
    if queries.ndim != 2 or queries.shape[1] != 3 or len(queries) == 0:
        raise ValueError("query_positions_world_m must have nonempty shape (N, 3)")
    if camera_matrix.shape != (3, 3):
        raise ValueError("intrinsics must have shape (3, 3)")
    if extrinsic.shape != (4, 4):
        raise ValueError("world_to_camera must have shape (4, 4)")
    if not (
        np.all(np.isfinite(queries))
        and np.all(np.isfinite(camera_matrix))
        and np.all(np.isfinite(extrinsic))
    ):
        raise ValueError("queries and calibration must be finite")

    homogeneous = np.concatenate(
        (queries, np.ones((len(queries), 1))),
        axis=1,
    )
    camera = (extrinsic @ homogeneous.T).T[:, :3]
    depth = camera[:, 2]
    if np.any(depth <= 0.0):
        raise ValueError("frame-zero queries must lie in front of the camera")
    projected = (camera_matrix @ camera.T).T
    pixels = projected[:, :2] / projected[:, 2:3]
    xyt = np.concatenate(
        (np.zeros((len(queries), 1)), pixels),
        axis=1,
    )
    return xyt.astype(np.float32), depth


def camera_tracks_to_world(
    camera_tracks_m: np.ndarray,
    camera_to_world: np.ndarray,
) -> np.ndarray:
    """Map initial-camera-gauge tracks into the approved PhysTwin world frame."""

    tracks = np.asarray(camera_tracks_m, dtype=float)
    transform = np.asarray(camera_to_world, dtype=float)
    if tracks.ndim != 3 or tracks.shape[2] != 3:
        raise ValueError("camera_tracks_m must have shape (T, N, 3)")
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("camera_to_world must be a finite 4x4 transform")
    homogeneous = np.concatenate(
        (tracks, np.ones((*tracks.shape[:2], 1))),
        axis=2,
    )
    world = np.einsum("ij,tnj->tni", transform, homogeneous)
    scale = world[:, :, 3:4]
    if np.any(np.abs(scale) < 1e-12):
        raise ValueError("camera_to_world produced an invalid homogeneous scale")
    return world[:, :, :3] / scale


def _validate_prediction_arrays(
    coords_world_m: np.ndarray,
    valid: np.ndarray,
    visibility_probability: np.ndarray,
    confidence: np.ndarray,
    query_points: np.ndarray,
    query_pixels_xyt: np.ndarray,
) -> SpatialTrackerV2Prediction:
    coords = np.asarray(coords_world_m)
    visibility = np.asarray(valid)
    probability = np.asarray(visibility_probability)
    confidence_array = np.asarray(confidence)
    queries = np.asarray(query_points)
    pixels = np.asarray(query_pixels_xyt)
    if coords.ndim != 3 or coords.shape[2] != 3 or coords.shape[0] == 0:
        raise ValueError("coords_world_m must have nonempty shape (T, N, 3)")
    if visibility.dtype != np.bool_ or visibility.shape != coords.shape[:2]:
        raise ValueError("valid must be boolean with shape (T, N)")
    if probability.shape != coords.shape[:2]:
        raise ValueError("visibility_probability must have shape (T, N)")
    if confidence_array.shape != coords.shape[:2]:
        raise ValueError("confidence must have shape (T, N)")
    if queries.shape != (coords.shape[1], 4):
        raise ValueError("query_points must have shape (N, 4)")
    if pixels.shape != (coords.shape[1], 3):
        raise ValueError("query_pixels_xyt must have shape (N, 3)")
    if not (
        np.all(np.isfinite(probability))
        and np.all(np.isfinite(confidence_array))
        and np.all(np.isfinite(queries))
        and np.all(np.isfinite(pixels))
    ):
        raise ValueError("prediction probabilities and queries must be finite")
    if np.any((probability < 0.0) | (probability > 1.2 + 1e-6)):
        raise ValueError("visibility_probability is outside the model range")
    finite = np.all(np.isfinite(coords), axis=2)
    return SpatialTrackerV2Prediction(
        coords_world_m=np.asarray(coords, dtype=np.float64),
        valid=np.asarray(visibility & finite, dtype=bool),
        visibility_probability=np.asarray(probability, dtype=np.float64),
        confidence=np.asarray(confidence_array, dtype=np.float64),
        query_points=np.asarray(queries, dtype=np.float64),
        query_pixels_xyt=np.asarray(pixels, dtype=np.float64),
    )


def load_spatialtrackerv2_prediction(
    path: str | Path,
) -> SpatialTrackerV2Prediction:
    """Load the minimal sealed fields from the frozen external runner."""

    required = {
        "coords_world_m",
        "valid",
        "visibility_probability",
        "confidence",
        "query_points",
        "query_pixels_xyt",
    }
    with np.load(path) as archive:
        missing = required.difference(archive.files)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"SpatialTrackerV2 result lacks required fields: {names}")
        arrays = {name: np.asarray(archive[name]) for name in required}
    return _validate_prediction_arrays(**arrays)


def validate_spatialtrackerv2_prediction_contract(
    prediction: SpatialTrackerV2Prediction,
    expected_query_points: np.ndarray,
    expected_query_pixels_xyt: np.ndarray,
    *,
    expected_frame_count: int,
    world_tolerance_m: float = 1e-7,
    pixel_tolerance: float = 1e-4,
) -> None:
    """Bind output to the locked prefix queries before opening score labels."""

    world = np.asarray(expected_query_points, dtype=float)
    pixels = np.asarray(expected_query_pixels_xyt, dtype=float)
    if prediction.coords_world_m.shape[0] != expected_frame_count:
        raise ValueError("prediction frame count differs from the locked prefix")
    if prediction.query_points.shape != world.shape or not np.allclose(
        prediction.query_points,
        world,
        rtol=0.0,
        atol=world_tolerance_m,
    ):
        raise ValueError("world queries differ from the locked input")
    if prediction.query_pixels_xyt.shape != pixels.shape or not np.allclose(
        prediction.query_pixels_xyt,
        pixels,
        rtol=0.0,
        atol=pixel_tolerance,
    ):
        raise ValueError("pixel queries differ from the locked projection")
    if not (
        np.all(prediction.query_points[:, 0] == 0.0)
        and np.all(prediction.query_pixels_xyt[:, 0] == 0.0)
    ):
        raise ValueError("competence-v1 permits frame-zero queries only")


def save_canonical_spatialtrackerv2_prediction(
    path: str | Path,
    prediction: SpatialTrackerV2Prediction,
) -> None:
    """Save a compact prediction carrier without copied RGB-D tensors."""

    np.savez_compressed(
        path,
        coords_world_m=prediction.coords_world_m,
        valid=prediction.valid,
        visibility_probability=prediction.visibility_probability,
        confidence=prediction.confidence,
        query_points=prediction.query_points,
        query_pixels_xyt=prediction.query_pixels_xyt,
    )


def load_canonical_spatialtrackerv2_prediction(
    path: str | Path,
) -> SpatialTrackerV2Prediction:
    """Load a compact sealed SpatialTrackerV2 carrier."""

    return load_spatialtrackerv2_prediction(path)


__all__ = [
    "SpatialTrackerV2Prediction",
    "camera_tracks_to_world",
    "load_canonical_spatialtrackerv2_prediction",
    "load_spatialtrackerv2_prediction",
    "project_world_queries_to_pixels",
    "save_canonical_spatialtrackerv2_prediction",
    "validate_spatialtrackerv2_prediction_contract",
]
