"""Regenerate continuous CoTracker3 and multiview cues from raw PhysTwin video."""

from __future__ import annotations

import hashlib
import json
import pickle
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .phystwin_raw_cues import (
    PhysTwinRawCueConfig,
    load_phystwin_raw_track_map,
)


@dataclass(frozen=True)
class CoTracker3CueConfig:
    """Frozen extraction settings for leakage-free training-video cues."""

    train_end_frame: int
    iterations: int = 6
    window_length: int = 16
    minimum_cycle_quality: float = 0.1
    minimum_multiview_quality: float = 0.1
    multiview_initial_depth_tolerance_m: float = 0.02
    initial_match_tolerance_m: float = 1e-6


@dataclass(frozen=True)
class CoTracker3Prediction:
    """Continuous low-level CoTracker3 outputs in input-image coordinates."""

    tracks_xy: np.ndarray
    visibility_probability: np.ndarray
    confidence_probability: np.ndarray


@dataclass(frozen=True)
class CoTracker3MultiviewObservation:
    """Causal, residual-independent multiview 3D observations."""

    points_world_m: np.ndarray
    valid: np.ndarray
    camera_count: np.ndarray
    reprojection_error_px: np.ndarray
    minimum_view_quality: np.ndarray


@dataclass(frozen=True)
class CoTracker3MultiviewDepthObservation:
    """Gauge-anchored multiview RGB-D observations of material queries."""

    points_world_m: np.ndarray
    valid: np.ndarray
    camera_count: np.ndarray
    view_disagreement_m: np.ndarray
    minimum_view_quality: np.ndarray


@dataclass(frozen=True)
class CoTracker3RayDiscrepancyPosterior:
    """Persistent 3D correction inferred directly from multiview image rays."""

    mean_m: np.ndarray
    variance_m2: np.ndarray
    observed: np.ndarray
    update_count: np.ndarray
    final_inlier_probability: np.ndarray
    camera_support: np.ndarray


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _distribution(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return {"count": 0, "minimum": None, "median": None, "mean": None,
                "p95": None, "maximum": None}
    return {
        "count": int(len(finite)),
        "minimum": float(np.min(finite)),
        "median": float(np.median(finite)),
        "mean": float(np.mean(finite)),
        "p95": float(np.quantile(finite, 0.95)),
        "maximum": float(np.max(finite)),
    }


def _git_revision(path: str | Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(Path(path)), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


class CoTracker3OnlineRunner:
    """Thin adapter that retains probabilities discarded by the public predictor."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        cotracker_root: str | Path,
        device: str = "cuda",
        window_length: int = 16,
        iterations: int = 6,
    ) -> None:
        root = str(Path(cotracker_root).resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            import torch
            from cotracker.predictor import CoTrackerOnlinePredictor
        except ImportError as error:
            raise RuntimeError(
                "CoTracker3 extraction requires torch and an official co-tracker checkout"
            ) from error
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if window_length < 4 or window_length % 2:
            raise ValueError("window_length must be an even integer of at least four")
        self._torch = torch
        self._iterations = iterations
        self._device = torch.device(device)
        self._predictor = CoTrackerOnlinePredictor(
            checkpoint=str(checkpoint_path),
            window_len=window_length,
        ).to(self._device).eval()

    def track(self, video: np.ndarray, queries_xy: np.ndarray) -> CoTracker3Prediction:
        """Track frame-zero queries through one video prefix."""

        torch = self._torch
        frames = np.asarray(video)
        queries = np.asarray(queries_xy, dtype=np.float32)
        if frames.ndim != 4 or frames.shape[3] != 3:
            raise ValueError("video must have shape (T, H, W, 3)")
        if frames.dtype != np.uint8:
            raise ValueError("video must contain uint8 RGB frames")
        if queries.ndim != 2 or queries.shape[1] != 2:
            raise ValueError("queries_xy must have shape (N, 2)")
        if len(frames) <= self._predictor.step:
            raise ValueError("video must be longer than one CoTracker step")
        if len(queries) == 0:
            shape = (len(frames), 0)
            return CoTracker3Prediction(
                tracks_xy=np.empty((*shape, 2), dtype=np.float32),
                visibility_probability=np.empty(shape, dtype=np.float32),
                confidence_probability=np.empty(shape, dtype=np.float32),
            )

        frame_count, height, width, _ = frames.shape
        query_tensor = torch.from_numpy(
            np.column_stack(
                [np.zeros(len(queries), dtype=np.float32), queries]
            )
        )[None].to(self._device)
        query_tensor[:, :, 1:] *= query_tensor.new_tensor(
            [
                (self._predictor.interp_shape[1] - 1) / (width - 1),
                (self._predictor.interp_shape[0] - 1) / (height - 1),
            ]
        )
        self._predictor.model.init_video_online_processing()
        tracks = visibility = confidence = None
        with torch.no_grad():
            for start in range(
                0,
                frame_count - self._predictor.step,
                self._predictor.step,
            ):
                chunk = torch.from_numpy(
                    np.ascontiguousarray(
                        frames[start : start + 2 * self._predictor.step]
                    )
                ).permute(0, 3, 1, 2)[None].float().to(self._device)
                batch, length, channels, source_height, source_width = chunk.shape
                chunk = torch.nn.functional.interpolate(
                    chunk.reshape(
                        batch * length,
                        channels,
                        source_height,
                        source_width,
                    ),
                    tuple(self._predictor.interp_shape),
                    mode="bilinear",
                    align_corners=True,
                ).reshape(
                    batch,
                    length,
                    channels,
                    *self._predictor.interp_shape,
                )
                tracks, visibility, confidence, _ = self._predictor.model(
                    video=chunk,
                    queries=query_tensor,
                    iters=self._iterations,
                    is_online=True,
                )
        assert tracks is not None and visibility is not None and confidence is not None
        tracks = tracks * tracks.new_tensor(
            [
                (width - 1) / (self._predictor.interp_shape[1] - 1),
                (height - 1) / (self._predictor.interp_shape[0] - 1),
            ]
        )
        return CoTracker3Prediction(
            tracks_xy=tracks[0].cpu().numpy().astype(np.float32),
            visibility_probability=visibility[0].cpu().numpy().astype(np.float32),
            confidence_probability=confidence[0].cpu().numpy().astype(np.float32),
        )


def project_world_points(
    points_world: np.ndarray,
    intrinsic: np.ndarray,
    camera_to_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points, returning x/y pixels and signed camera depth."""

    points = np.asarray(points_world, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_world must have shape (N, 3)")
    world_to_camera = np.linalg.inv(np.asarray(camera_to_world, dtype=float))
    homogeneous = np.column_stack([points, np.ones(len(points))])
    camera = homogeneous @ world_to_camera.T
    projected = camera[:, :3] @ np.asarray(intrinsic, dtype=float).T
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels = projected[:, :2] / projected[:, 2:3]
    return pixels, camera[:, 2]


def triangulate_multiview_tracks(
    tracks_xy: np.ndarray,
    valid: np.ndarray,
    weights: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Triangulate tracks and return weighted RMS reprojection error in pixels."""

    tracks = np.asarray(tracks_xy, dtype=float)
    validity = np.asarray(valid, dtype=bool)
    reliability = np.asarray(weights, dtype=float)
    if tracks.ndim != 4 or tracks.shape[3] != 2:
        raise ValueError("tracks_xy must have shape (C, T, N, 2)")
    if validity.shape != tracks.shape[:3] or reliability.shape != validity.shape:
        raise ValueError("valid and weights must match tracks' first three axes")
    camera_count, frame_count, track_count, _ = tracks.shape
    intrinsics_array = np.asarray(intrinsics, dtype=float)
    extrinsics = np.asarray(camera_to_world, dtype=float)
    if intrinsics_array.shape != (camera_count, 3, 3):
        raise ValueError("intrinsics must have shape (C, 3, 3)")
    if extrinsics.shape != (camera_count, 4, 4):
        raise ValueError("camera_to_world must have shape (C, 4, 4)")
    if np.any(reliability < 0.0) or not np.all(np.isfinite(reliability)):
        raise ValueError("weights must be finite and nonnegative")

    points = np.full((frame_count, track_count, 3), np.nan, dtype=float)
    error = np.full((frame_count, track_count), np.nan, dtype=float)
    camera_counts = np.sum(validity, axis=0).astype(np.int16)
    inverse_intrinsics = np.linalg.inv(intrinsics_array)
    centers = extrinsics[:, :3, 3]
    rotations = extrinsics[:, :3, :3]
    identity = np.eye(3)
    for frame in range(frame_count):
        matrix = np.zeros((track_count, 3, 3), dtype=float)
        right_hand_side = np.zeros((track_count, 3), dtype=float)
        effective_weight = np.where(validity[:, frame], reliability[:, frame], 0.0)
        for camera in range(camera_count):
            camera_tracks = np.where(
                validity[camera, frame, :, None],
                tracks[camera, frame],
                0.0,
            )
            pixel_homogeneous = np.column_stack(
                [camera_tracks, np.ones(track_count)]
            )
            rays_camera = pixel_homogeneous @ inverse_intrinsics[camera].T
            rays_world = rays_camera @ rotations[camera].T
            ray_norm = np.linalg.norm(rays_world, axis=1)
            rays_world /= np.maximum(ray_norm[:, None], 1e-12)
            projectors = identity - np.einsum(
                "ni,nj->nij", rays_world, rays_world
            )
            weighted = effective_weight[camera, :, None, None] * projectors
            matrix += weighted
            right_hand_side += np.einsum(
                "nij,j->ni", weighted, centers[camera]
            )
        selected = (camera_counts[frame] >= 2) & (
            np.sum(effective_weight, axis=0) > 0.0
        )
        if not np.any(selected):
            continue
        stabilized = matrix[selected] + 1e-10 * identity
        solved = np.linalg.solve(
            stabilized,
            right_hand_side[selected, :, None],
        )[:, :, 0]
        points[frame, selected] = solved
        squared_error = np.zeros(np.sum(selected), dtype=float)
        weight_sum = np.zeros(np.sum(selected), dtype=float)
        for camera in range(camera_count):
            projected, depth = project_world_points(
                solved,
                intrinsics_array[camera],
                extrinsics[camera],
            )
            camera_valid = validity[camera, frame, selected] & (depth > 0.0)
            camera_weight = effective_weight[camera, selected] * camera_valid
            delta_sq = np.sum(
                np.square(projected - tracks[camera, frame, selected]),
                axis=1,
            )
            squared_error += camera_weight * np.where(camera_valid, delta_sq, 0.0)
            weight_sum += camera_weight
        usable = weight_sum > 0.0
        frame_error = np.full(np.sum(selected), np.nan, dtype=float)
        frame_error[usable] = np.sqrt(squared_error[usable] / weight_sum[usable])
        error[frame, selected] = frame_error
    return points, error, camera_counts


def pack_multiview_triangulation(
    points_world_m: np.ndarray,
    reprojection_error_px: np.ndarray,
    camera_count: np.ndarray,
    *,
    frame_count: int,
) -> dict[str, np.ndarray]:
    """Pack prefix triangulations into leakage-safe full-sequence cue arrays."""

    points = np.asarray(points_world_m, dtype=float)
    reprojection = np.asarray(reprojection_error_px, dtype=float)
    counts = np.asarray(camera_count)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("points_world_m must have shape (T_prefix, N, 3)")
    if reprojection.shape != points.shape[:2]:
        raise ValueError("reprojection_error_px must match points' first two axes")
    if counts.shape != points.shape[:2]:
        raise ValueError("camera_count must match points' first two axes")
    prefix_frame_count, track_count, _ = points.shape
    if frame_count < prefix_frame_count:
        raise ValueError("frame_count cannot be shorter than the prefix")
    if np.any(counts < 0):
        raise ValueError("camera_count must be nonnegative")

    valid = (
        np.all(np.isfinite(points), axis=2)
        & np.isfinite(reprojection)
        & (counts >= 2)
    )
    full_points = np.full(
        (frame_count, track_count, 3),
        np.nan,
        dtype=np.float32,
    )
    full_valid = np.zeros((frame_count, track_count), dtype=bool)
    full_points[:prefix_frame_count] = np.where(
        valid[:, :, None],
        points,
        np.nan,
    ).astype(np.float32)
    full_valid[:prefix_frame_count] = valid
    return {
        "multiview_points_world_m": full_points,
        "multiview_point_valid": full_valid,
    }


def load_cotracker3_multiview_observations(
    cues_path: str | Path,
    initial_world_points_m: np.ndarray,
    *,
    train_end_frame: int,
    minimum_view_quality: float,
    maximum_reprojection_error_px: float,
    maximum_cycle_error_px: float,
    minimum_camera_count: int = 3,
) -> CoTracker3MultiviewObservation:
    """Load and re-triangulate a leakage-free CoTracker3 observation prefix."""

    if train_end_frame <= 0:
        raise ValueError("train_end_frame must be positive")
    if not 0.0 <= minimum_view_quality <= 1.0:
        raise ValueError("minimum_view_quality must lie in [0, 1]")
    if maximum_reprojection_error_px <= 0.0:
        raise ValueError("maximum_reprojection_error_px must be positive")
    if maximum_cycle_error_px <= 0.0:
        raise ValueError("maximum_cycle_error_px must be positive")
    if minimum_camera_count < 2:
        raise ValueError("minimum_camera_count must be at least two")

    required = {
        "multiview_tracks_xy_prefix",
        "multiview_quality_probability_prefix",
        "multiview_view_valid_prefix",
        "multiview_intrinsics",
        "multiview_camera_to_world",
        "forward_backward_error_px",
        "forward_backward_valid",
        "boundary_distance",
        "cue_available",
    }
    with np.load(cues_path) as archive:
        missing = required.difference(archive.files)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(
                f"CoTracker3 archive lacks multiview schema-v2 fields: {names}"
            )
        tracks = np.asarray(
            archive["multiview_tracks_xy_prefix"][:, :train_end_frame],
            dtype=float,
        )
        quality = np.asarray(
            archive["multiview_quality_probability_prefix"][
                :, :train_end_frame
            ],
            dtype=float,
        )
        view_valid = np.asarray(
            archive["multiview_view_valid_prefix"][:, :train_end_frame],
            dtype=bool,
        )
        intrinsics = np.asarray(archive["multiview_intrinsics"], dtype=float)
        camera_to_world = np.asarray(
            archive["multiview_camera_to_world"],
            dtype=float,
        )
        cycle_error = np.asarray(
            archive["forward_backward_error_px"][:train_end_frame],
            dtype=float,
        )
        cycle_valid = np.asarray(
            archive["forward_backward_valid"][:train_end_frame],
            dtype=bool,
        )
        boundary = np.asarray(
            archive["boundary_distance"][:train_end_frame],
            dtype=float,
        )
        available = np.asarray(
            archive["cue_available"][:train_end_frame],
            dtype=bool,
        )

    if tracks.shape[:3] != quality.shape or tracks.shape[3] != 2:
        raise ValueError("multiview track and quality shapes are inconsistent")
    if view_valid.shape != quality.shape:
        raise ValueError("multiview validity does not match quality")
    if tracks.shape[1] != train_end_frame:
        raise ValueError("archive is shorter than train_end_frame")
    initial = np.asarray(initial_world_points_m, dtype=float)
    if initial.shape != (tracks.shape[2], 3):
        raise ValueError("initial_world_points_m must have shape (N, 3)")
    if cycle_error.shape != tracks.shape[1:3]:
        raise ValueError("cycle error does not match multiview tracks")
    if cycle_valid.shape != cycle_error.shape:
        raise ValueError("cycle validity does not match cycle error")
    if boundary.shape != cycle_error.shape or available.shape != cycle_error.shape:
        raise ValueError("source reliability cues do not match multiview tracks")
    if minimum_camera_count > tracks.shape[0]:
        raise ValueError("minimum_camera_count exceeds the archived camera count")

    selected_views = (
        view_valid
        & np.isfinite(tracks).all(axis=3)
        & np.isfinite(quality)
        & (quality >= minimum_view_quality)
    )
    points, reprojection, camera_count = triangulate_multiview_tracks(
        tracks,
        selected_views,
        np.where(selected_views, quality, 0.0),
        intrinsics,
        camera_to_world,
    )
    initial_valid = (
        np.all(np.isfinite(points[0]), axis=1)
        & (camera_count[0] >= minimum_camera_count)
    )
    anchored = np.full_like(points, np.nan)
    anchored[:, initial_valid] = (
        initial[initial_valid][None]
        + points[:, initial_valid]
        - points[0, initial_valid][None]
    )
    valid = (
        np.all(np.isfinite(anchored), axis=2)
        & np.isfinite(reprojection)
        & (camera_count >= minimum_camera_count)
        & (reprojection <= maximum_reprojection_error_px)
        & cycle_valid
        & (cycle_error <= maximum_cycle_error_px)
        & (boundary > 0.0)
        & available
    )
    minimum_quality = np.min(
        np.where(selected_views, quality, np.inf),
        axis=0,
    )
    minimum_quality[~np.isfinite(minimum_quality)] = 0.0
    anchored[~valid] = np.nan
    return CoTracker3MultiviewObservation(
        points_world_m=anchored,
        valid=valid,
        camera_count=camera_count,
        reprojection_error_px=reprojection,
        minimum_view_quality=minimum_quality,
    )


def load_cotracker3_multiview_depth_observations(
    cues_path: str | Path,
    raw_case_dir: str | Path,
    initial_world_points_m: np.ndarray,
    *,
    train_end_frame: int,
    minimum_view_quality: float,
    maximum_view_disagreement_m: float,
    maximum_cycle_error_px: float,
    minimum_camera_count: int = 3,
) -> CoTracker3MultiviewDepthObservation:
    """Fuse gauge-anchored per-view depth lifts without using state residuals."""

    if train_end_frame <= 0:
        raise ValueError("train_end_frame must be positive")
    if not 0.0 <= minimum_view_quality <= 1.0:
        raise ValueError("minimum_view_quality must lie in [0, 1]")
    if maximum_view_disagreement_m <= 0.0:
        raise ValueError("maximum_view_disagreement_m must be positive")
    if maximum_cycle_error_px <= 0.0:
        raise ValueError("maximum_cycle_error_px must be positive")
    if minimum_camera_count < 2:
        raise ValueError("minimum_camera_count must be at least two")

    required = {
        "multiview_tracks_xy_prefix",
        "multiview_quality_probability_prefix",
        "multiview_view_valid_prefix",
        "multiview_intrinsics",
        "multiview_camera_to_world",
        "forward_backward_error_px",
        "forward_backward_valid",
        "boundary_distance",
        "cue_available",
    }
    with np.load(cues_path) as archive:
        missing = required.difference(archive.files)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(
                f"CoTracker3 archive lacks multiview schema-v2 fields: {names}"
            )
        tracks = np.asarray(
            archive["multiview_tracks_xy_prefix"][:, :train_end_frame],
            dtype=float,
        )
        quality = np.asarray(
            archive["multiview_quality_probability_prefix"][
                :, :train_end_frame
            ],
            dtype=float,
        )
        archived_view_valid = np.asarray(
            archive["multiview_view_valid_prefix"][:, :train_end_frame],
            dtype=bool,
        )
        intrinsics = np.asarray(archive["multiview_intrinsics"], dtype=float)
        camera_to_world = np.asarray(
            archive["multiview_camera_to_world"],
            dtype=float,
        )
        cycle_error = np.asarray(
            archive["forward_backward_error_px"][:train_end_frame],
            dtype=float,
        )
        cycle_valid = np.asarray(
            archive["forward_backward_valid"][:train_end_frame],
            dtype=bool,
        )
        boundary = np.asarray(
            archive["boundary_distance"][:train_end_frame],
            dtype=float,
        )
        available = np.asarray(
            archive["cue_available"][:train_end_frame],
            dtype=bool,
        )

    if tracks.shape[:3] != quality.shape or tracks.shape[3] != 2:
        raise ValueError("multiview track and quality shapes are inconsistent")
    if archived_view_valid.shape != quality.shape:
        raise ValueError("multiview validity does not match quality")
    if tracks.shape[1] != train_end_frame:
        raise ValueError("archive is shorter than train_end_frame")
    camera_count, _, track_count, _ = tracks.shape
    initial = np.asarray(initial_world_points_m, dtype=float)
    if initial.shape != (track_count, 3):
        raise ValueError("initial_world_points_m must have shape (N, 3)")
    if intrinsics.shape != (camera_count, 3, 3):
        raise ValueError("archived intrinsics do not match the camera count")
    if camera_to_world.shape != (camera_count, 4, 4):
        raise ValueError("archived camera poses do not match the camera count")
    if minimum_camera_count > camera_count:
        raise ValueError("minimum_camera_count exceeds the archived camera count")
    if cycle_error.shape != tracks.shape[1:3]:
        raise ValueError("cycle error does not match multiview tracks")
    if cycle_valid.shape != cycle_error.shape:
        raise ValueError("cycle validity does not match cycle error")
    if boundary.shape != cycle_error.shape or available.shape != cycle_error.shape:
        raise ValueError("source reliability cues do not match multiview tracks")

    world_by_view = np.full(
        (camera_count, train_end_frame, track_count, 3),
        np.nan,
        dtype=float,
    )
    depth_valid = np.zeros(tracks.shape[:3], dtype=bool)
    raw_path = Path(raw_case_dir)
    for camera in range(camera_count):
        inverse_intrinsic = np.linalg.inv(intrinsics[camera])
        for frame in range(train_end_frame):
            depth_path = raw_path / "depth" / str(camera) / f"{frame}.npy"
            if not depth_path.is_file():
                raise FileNotFoundError(f"missing raw depth frame: {depth_path}")
            depth = np.asarray(np.load(depth_path), dtype=float)
            xy = tracks[camera, frame]
            finite = np.all(np.isfinite(xy), axis=1)
            pixels = np.rint(np.where(finite[:, None], xy, 0.0)).astype(np.int64)
            inside = (
                finite
                & (pixels[:, 0] >= 0)
                & (pixels[:, 0] < depth.shape[1])
                & (pixels[:, 1] >= 0)
                & (pixels[:, 1] < depth.shape[0])
            )
            selected = np.flatnonzero(inside)
            if len(selected) == 0:
                continue
            z_m = (
                depth[pixels[selected, 1], pixels[selected, 0]].astype(float)
                / 1000.0
            )
            positive = np.isfinite(z_m) & (z_m > 0.0)
            selected = selected[positive]
            z_m = z_m[positive]
            if len(selected) == 0:
                continue
            homogeneous_pixels = np.column_stack(
                [xy[selected], np.ones(len(selected))]
            )
            camera_points = (
                homogeneous_pixels @ inverse_intrinsic.T
            ) * z_m[:, None]
            homogeneous_camera = np.column_stack(
                [camera_points, np.ones(len(camera_points))]
            )
            world_points = homogeneous_camera @ camera_to_world[camera].T
            world_by_view[camera, frame, selected] = world_points[:, :3]
            depth_valid[camera, frame, selected] = True

    selected_views = (
        archived_view_valid
        & depth_valid
        & np.isfinite(quality)
        & (quality >= minimum_view_quality)
    )
    initial_view_valid = selected_views[:, 0]
    selected_views &= initial_view_valid[:, None]
    anchored_by_view = np.full_like(world_by_view, np.nan)
    for camera in range(camera_count):
        selected = np.flatnonzero(initial_view_valid[camera])
        camera_world = world_by_view[camera]
        anchored_by_view[camera][:, selected] = (
            initial[selected][None]
            + camera_world[:, selected]
            - camera_world[0, selected][None]
        )

    masked = np.ma.array(
        anchored_by_view,
        mask=np.broadcast_to(~selected_views[:, :, :, None], anchored_by_view.shape),
    )
    fused = np.ma.median(masked, axis=0).filled(np.nan)
    counts = np.sum(selected_views, axis=0).astype(np.int16)
    squared_distance = np.sum(
        np.square(anchored_by_view - fused[None]),
        axis=3,
    )
    weight = np.where(selected_views, quality, 0.0)
    weight_sum = np.sum(weight, axis=0)
    disagreement = np.full(counts.shape, np.nan, dtype=float)
    usable = weight_sum > 0.0
    disagreement[usable] = np.sqrt(
        np.sum(
            weight * np.where(selected_views, squared_distance, 0.0),
            axis=0,
        )[usable]
        / weight_sum[usable]
    )
    minimum_quality = np.min(
        np.where(selected_views, quality, np.inf),
        axis=0,
    )
    minimum_quality[~np.isfinite(minimum_quality)] = 0.0
    valid = (
        np.all(np.isfinite(fused), axis=2)
        & (counts >= minimum_camera_count)
        & np.isfinite(disagreement)
        & (disagreement <= maximum_view_disagreement_m)
        & cycle_valid
        & (cycle_error <= maximum_cycle_error_px)
        & (boundary > 0.0)
        & available
    )
    fused[~valid] = np.nan
    return CoTracker3MultiviewDepthObservation(
        points_world_m=fused,
        valid=valid,
        camera_count=counts,
        view_disagreement_m=disagreement,
        minimum_view_quality=minimum_quality,
    )


def infer_cotracker3_ray_discrepancy(
    cues_path: str | Path,
    baseline_points_world_m: np.ndarray,
    *,
    end_frame: int,
    window_frames: int,
    minimum_view_quality: float,
    maximum_cycle_error_px: float,
    minimum_camera_count: int = 2,
    pixel_noise_std: float = 2.0,
    prior_std_m: float = 0.05,
    degrees_of_freedom: float = 4.0,
    robust_iterations: int = 3,
) -> CoTracker3RayDiscrepancyPosterior:
    """Infer one robust persistent correction from causal multiview ray evidence."""

    if end_frame <= 0:
        raise ValueError("end_frame must be positive")
    if window_frames <= 0:
        raise ValueError("window_frames must be positive")
    if not 0.0 <= minimum_view_quality <= 1.0:
        raise ValueError("minimum_view_quality must lie in [0, 1]")
    if maximum_cycle_error_px <= 0.0:
        raise ValueError("maximum_cycle_error_px must be positive")
    if minimum_camera_count < 2:
        raise ValueError("minimum_camera_count must be at least two")
    if pixel_noise_std <= 0.0 or prior_std_m <= 0.0:
        raise ValueError("ray likelihood scales must be positive")
    if degrees_of_freedom <= 0.0:
        raise ValueError("degrees_of_freedom must be positive")
    if robust_iterations < 1:
        raise ValueError("robust_iterations must be positive")

    required = {
        "multiview_tracks_xy_prefix",
        "multiview_quality_probability_prefix",
        "multiview_view_valid_prefix",
        "multiview_intrinsics",
        "multiview_camera_to_world",
        "forward_backward_error_px",
        "forward_backward_valid",
        "boundary_distance",
        "cue_available",
    }
    with np.load(cues_path) as archive:
        missing = required.difference(archive.files)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(
                f"CoTracker3 archive lacks multiview schema-v2 fields: {names}"
            )
        tracks_all = np.asarray(
            archive["multiview_tracks_xy_prefix"],
            dtype=float,
        )
        quality_all = np.asarray(
            archive["multiview_quality_probability_prefix"],
            dtype=float,
        )
        view_valid_all = np.asarray(
            archive["multiview_view_valid_prefix"],
            dtype=bool,
        )
        intrinsics = np.asarray(archive["multiview_intrinsics"], dtype=float)
        camera_to_world = np.asarray(
            archive["multiview_camera_to_world"],
            dtype=float,
        )
        cycle_error_all = np.asarray(
            archive["forward_backward_error_px"],
            dtype=float,
        )
        cycle_valid_all = np.asarray(
            archive["forward_backward_valid"],
            dtype=bool,
        )
        boundary_all = np.asarray(
            archive["boundary_distance"],
            dtype=float,
        )
        available_all = np.asarray(
            archive["cue_available"],
            dtype=bool,
        )

    baseline = np.asarray(baseline_points_world_m, dtype=float)
    if baseline.ndim != 3 or baseline.shape[2] != 3:
        raise ValueError("baseline_points_world_m must have shape (T, N, 3)")
    camera_count, archived_frames, track_count, coordinate_count = tracks_all.shape
    if coordinate_count != 2:
        raise ValueError("multiview tracks must contain x/y coordinates")
    if end_frame > archived_frames or end_frame > len(baseline):
        raise ValueError("end_frame exceeds the causal cue or baseline prefix")
    if baseline.shape[1] != track_count:
        raise ValueError("baseline point count does not match CoTracker queries")
    if quality_all.shape != tracks_all.shape[:3]:
        raise ValueError("multiview quality does not match tracks")
    if view_valid_all.shape != quality_all.shape:
        raise ValueError("multiview validity does not match quality")
    if intrinsics.shape != (camera_count, 3, 3):
        raise ValueError("archived intrinsics do not match the camera count")
    if camera_to_world.shape != (camera_count, 4, 4):
        raise ValueError("archived camera poses do not match the camera count")
    if minimum_camera_count > camera_count:
        raise ValueError("minimum_camera_count exceeds the archived camera count")
    if cycle_error_all.shape[1] != track_count:
        raise ValueError("cycle error does not match CoTracker queries")

    start_frame = max(0, end_frame - window_frames)
    tracks = tracks_all[:, start_frame:end_frame]
    quality = quality_all[:, start_frame:end_frame]
    view_valid = view_valid_all[:, start_frame:end_frame].copy()
    cycle_error = cycle_error_all[start_frame:end_frame]
    cycle_valid = cycle_valid_all[start_frame:end_frame]
    boundary = boundary_all[start_frame:end_frame]
    available = available_all[start_frame:end_frame]
    source_valid = (
        cycle_valid
        & np.isfinite(cycle_error)
        & (cycle_error <= maximum_cycle_error_px)
        & (boundary > 0.0)
        & available
    )
    view_valid &= np.isfinite(tracks).all(axis=3)
    view_valid &= np.isfinite(quality) & (quality >= minimum_view_quality)
    view_valid &= source_valid[None]
    redundant = np.sum(view_valid, axis=0) >= minimum_camera_count
    view_valid &= redundant[None]

    frame_count = end_frame - start_frame
    inverse_intrinsics = np.linalg.inv(intrinsics)
    centers = camera_to_world[:, :3, 3]
    rotations = camera_to_world[:, :3, :3]
    identity = np.eye(3)
    ray_direction = np.zeros((camera_count, frame_count, track_count, 3))
    projector = np.zeros((camera_count, frame_count, track_count, 3, 3))
    metric_sigma = np.full(
        (camera_count, frame_count, track_count),
        np.inf,
        dtype=float,
    )
    for camera in range(camera_count):
        pixel_homogeneous = np.concatenate(
            [tracks[camera], np.ones((*tracks[camera].shape[:2], 1))],
            axis=2,
        )
        rays_camera = pixel_homogeneous @ inverse_intrinsics[camera].T
        rays_world = rays_camera @ rotations[camera].T
        rays_world /= np.maximum(
            np.linalg.norm(rays_world, axis=2, keepdims=True),
            1e-12,
        )
        ray_direction[camera] = rays_world
        projector[camera] = identity - np.einsum(
            "tni,tnj->tnij",
            rays_world,
            rays_world,
        )
        distance = np.linalg.norm(
            baseline[start_frame:end_frame] - centers[camera],
            axis=2,
        )
        focal = float(
            np.sqrt(intrinsics[camera, 0, 0] * intrinsics[camera, 1, 1])
        )
        metric_sigma[camera] = pixel_noise_std * distance / focal

    robust_weight = np.ones_like(quality)
    mean = np.zeros((track_count, 3), dtype=float)
    precision = np.repeat(
        (identity / prior_std_m**2)[None],
        track_count,
        axis=0,
    )
    update_count = np.sum(view_valid, axis=(0, 1)).astype(np.int64)
    camera_support = np.sum(np.any(view_valid, axis=1), axis=0).astype(np.int16)
    for iteration in range(robust_iterations):
        precision = np.repeat(
            (identity / prior_std_m**2)[None],
            track_count,
            axis=0,
        )
        right_hand_side = np.zeros((track_count, 3), dtype=float)
        raw_observation_precision = np.where(
            view_valid,
            quality
            * robust_weight
            / np.maximum(np.square(metric_sigma), 1e-12),
            0.0,
        )
        temporal_sum = np.sum(raw_observation_precision, axis=1)
        temporal_square_sum = np.sum(
            np.square(raw_observation_precision),
            axis=1,
        )
        temporal_effective_count = np.divide(
            np.square(temporal_sum),
            temporal_square_sum,
            out=np.ones_like(temporal_sum),
            where=temporal_square_sum > 0.0,
        )
        observation_precision = raw_observation_precision / np.maximum(
            temporal_effective_count[:, None],
            1.0,
        )
        # Unknown cross-view correlation is handled conservatively: distinct
        # rays improve geometry, but their nominal information is averaged.
        observation_precision /= np.maximum(
            camera_support[None, None],
            1,
        )
        for camera in range(camera_count):
            for local_frame, frame in enumerate(
                range(start_frame, end_frame)
            ):
                selected = view_valid[camera, local_frame]
                if not np.any(selected):
                    continue
                current_precision = observation_precision[camera, local_frame]
                weighted_projector = (
                    current_precision[:, None, None]
                    * projector[camera, local_frame]
                )
                precision[selected] += weighted_projector[selected]
                offset = centers[camera] - baseline[frame]
                right_hand_side[selected] += (
                    current_precision[selected, None]
                    * np.einsum(
                        "nij,nj->ni",
                        projector[camera, local_frame, selected],
                        offset[selected],
                    )
                )
        mean = np.linalg.solve(
            precision,
            right_hand_side[:, :, None],
        )[:, :, 0]
        if iteration + 1 == robust_iterations:
            break
        for camera in range(camera_count):
            for local_frame, frame in enumerate(
                range(start_frame, end_frame)
            ):
                selected = view_valid[camera, local_frame]
                if not np.any(selected):
                    continue
                projected, depth = project_world_points(
                    baseline[frame] + mean,
                    intrinsics[camera],
                    camera_to_world[camera],
                )
                radial_px = np.linalg.norm(
                    projected - tracks[camera, local_frame],
                    axis=1,
                )
                standardized_sq = np.square(radial_px / pixel_noise_std)
                student_weight = (
                    degrees_of_freedom + 2.0
                ) / (degrees_of_freedom + standardized_sq)
                robust_weight[camera, local_frame] = np.where(
                    selected & (depth > 0.0),
                    np.clip(student_weight, 0.02, 1.0),
                    0.0,
                )

    covariance = np.linalg.inv(precision)
    variance = np.mean(np.diagonal(covariance, axis1=1, axis2=2), axis=1)
    observed = (update_count > 0) & (camera_support >= minimum_camera_count)
    mean[~observed] = 0.0
    variance[~observed] = prior_std_m**2
    final_inlier = np.zeros(track_count, dtype=float)
    robust_mass = np.sum(np.where(view_valid, robust_weight, 0.0), axis=(0, 1))
    final_inlier[observed] = robust_mass[observed] / update_count[observed]
    return CoTracker3RayDiscrepancyPosterior(
        mean_m=mean,
        variance_m2=variance,
        observed=observed,
        update_count=update_count,
        final_inlier_probability=final_inlier,
        camera_support=camera_support,
    )


def _load_video_prefix(
    raw_case_dir: Path,
    camera: int,
    end_frame: int,
) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("CoTracker3 extraction requires Pillow") from error
    image_dir = raw_case_dir / "color" / str(camera)
    frames = []
    for frame in range(end_frame):
        path = image_dir / f"{frame}.png"
        if not path.is_file():
            raise FileNotFoundError(f"missing raw RGB frame: {path}")
        with Image.open(path) as image:
            frames.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
    return np.stack(frames)


def _pixels_inside_mask(tracks_xy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    tracks = np.asarray(tracks_xy, dtype=float)
    finite = np.all(np.isfinite(tracks), axis=1)
    pixels = np.rint(np.where(finite[:, None], tracks, 0.0)).astype(np.int64)
    height, width = mask.shape
    inside = (
        finite
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    selected = np.flatnonzero(inside)
    inside[selected] &= mask[pixels[selected, 1], pixels[selected, 0]]
    return inside


def _initial_multiview_eligibility(
    world_points: np.ndarray,
    camera_points: np.ndarray,
    object_mask: np.ndarray,
    intrinsic: np.ndarray,
    camera_to_world: np.ndarray,
    *,
    depth_tolerance_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pixels, depth = project_world_points(world_points, intrinsic, camera_to_world)
    rounded = np.rint(pixels).astype(np.int64)
    height, width = object_mask.shape
    in_bounds = (
        np.all(np.isfinite(pixels), axis=1)
        & (depth > 0.0)
        & (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    surface_distance = np.full(len(world_points), np.inf, dtype=float)
    selected = np.flatnonzero(in_bounds)
    if len(selected):
        raw_surface = camera_points[rounded[selected, 1], rounded[selected, 0]]
        surface_distance[selected] = np.linalg.norm(
            raw_surface - world_points[selected], axis=1
        )
    eligible = in_bounds & (surface_distance <= depth_tolerance_m)
    mask_selected = np.flatnonzero(eligible)
    eligible[mask_selected] &= object_mask[
        rounded[mask_selected, 1], rounded[mask_selected, 0]
    ]
    return pixels, eligible, surface_distance


def build_phystwin_cotracker3_cues(
    final_data_path: str | Path,
    raw_case_dir: str | Path,
    checkpoint_path: str | Path,
    cotracker_root: str | Path,
    output_npz_path: str | Path,
    *,
    config: CoTracker3CueConfig,
    base_cues_path: str | Path | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    """Regenerate continuous source, cycle, boundary, and multiview cues."""

    if config.train_end_frame <= config.window_length // 2:
        raise ValueError("train_end_frame is too short for online CoTracker3")
    if not 0.0 <= config.minimum_cycle_quality <= 1.0:
        raise ValueError("minimum_cycle_quality must lie in [0, 1]")
    if not 0.0 <= config.minimum_multiview_quality <= 1.0:
        raise ValueError("minimum_multiview_quality must lie in [0, 1]")
    if config.multiview_initial_depth_tolerance_m <= 0.0:
        raise ValueError("multiview_initial_depth_tolerance_m must be positive")
    raw_path = Path(raw_case_dir)
    checkpoint = Path(checkpoint_path)
    tracker_root = Path(cotracker_root)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not (tracker_root / ".git").exists():
        raise FileNotFoundError("cotracker_root must be an official git checkout")
    mapping = load_phystwin_raw_track_map(
        final_data_path,
        raw_path,
        config=PhysTwinRawCueConfig(
            initial_match_tolerance_m=config.initial_match_tolerance_m
        ),
    )
    frame_count, track_count = mapping.final_visible.shape
    if config.train_end_frame >= frame_count:
        raise ValueError("train_end_frame must leave at least one held-out frame")
    metadata = json.loads((raw_path / "metadata.json").read_text(encoding="utf-8"))
    intrinsics = np.asarray(metadata["intrinsics"], dtype=float)
    with (raw_path / "calibrate.pkl").open("rb") as handle:
        camera_to_world = np.asarray(pickle.load(handle), dtype=float)
    camera_count = len(mapping.track_paths)
    if intrinsics.shape != (camera_count, 3, 3):
        raise ValueError("metadata intrinsics do not match the raw cameras")
    if camera_to_world.shape != (camera_count, 4, 4):
        raise ValueError("calibrate.pkl does not contain one 4x4 pose per camera")
    with (raw_path / "mask" / "processed_masks.pkl").open("rb") as handle:
        processed_masks = pickle.load(handle)
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as error:
        raise RuntimeError("rich cue extraction requires scipy") from error

    cues: dict[str, np.ndarray] = {}
    if base_cues_path is not None:
        with np.load(base_cues_path) as archive:
            cues.update({name: np.asarray(archive[name]) for name in archive.files})
    cue_available = np.zeros((frame_count, track_count), dtype=bool)
    cue_available[: config.train_end_frame] = True
    confidence = np.ones((frame_count, track_count), dtype=np.float32)
    visibility = np.ones_like(confidence)
    source_tracks = np.full((frame_count, track_count, 2), np.nan, dtype=np.float32)
    cycle_error = np.zeros((frame_count, track_count), dtype=np.float32)
    cycle_valid = np.zeros((frame_count, track_count), dtype=bool)
    boundary = np.full((frame_count, track_count), 1e6, dtype=np.float32)
    multiview_tracks = np.full(
        (camera_count, config.train_end_frame, track_count, 2),
        np.nan,
        dtype=np.float32,
    )
    multiview_quality = np.zeros(
        (camera_count, config.train_end_frame, track_count), dtype=np.float32
    )
    multiview_valid = np.zeros_like(multiview_quality, dtype=bool)
    initial_eligible = np.zeros((camera_count, track_count), dtype=bool)
    initial_surface_distance = np.full(
        (camera_count, track_count), np.inf, dtype=np.float32
    )
    parity: dict[str, dict[str, float | int | None]] = {}
    runner = CoTracker3OnlineRunner(
        checkpoint,
        cotracker_root=tracker_root,
        device=device,
        window_length=config.window_length,
        iterations=config.iterations,
    )
    for camera in range(camera_count):
        video = _load_video_prefix(raw_path, camera, config.train_end_frame)
        archived_tracks = mapping.tracks_by_camera[camera]
        archived_queries_xy = archived_tracks[0, :, ::-1].astype(np.float32)
        forward = runner.track(video, archived_queries_xy)
        reverse = runner.track(
            np.ascontiguousarray(video[::-1]),
            forward.tracks_xy[-1],
        )
        reverse_tracks = reverse.tracks_xy[::-1]
        reverse_quality = (
            reverse.visibility_probability * reverse.confidence_probability
        )[::-1]
        forward_quality = (
            forward.visibility_probability * forward.confidence_probability
        )
        archive_delta = np.linalg.norm(
            forward.tracks_xy
            - archived_tracks[: config.train_end_frame, :, ::-1],
            axis=2,
        )
        parity[str(camera)] = _distribution(archive_delta)

        selected = np.flatnonzero(mapping.source_camera == camera)
        raw_ids = mapping.source_track[selected]
        selected_forward_tracks = forward.tracks_xy[:, raw_ids]
        selected_forward_quality = forward_quality[:, raw_ids]
        source_tracks[: config.train_end_frame, selected] = selected_forward_tracks
        confidence[: config.train_end_frame, selected] = (
            forward.confidence_probability[:, raw_ids]
        )
        visibility[: config.train_end_frame, selected] = (
            forward.visibility_probability[:, raw_ids]
        )
        selected_cycle_error = np.linalg.norm(
            selected_forward_tracks - reverse_tracks[:, raw_ids], axis=2
        )
        selected_cycle_valid = (
            selected_forward_quality >= config.minimum_cycle_quality
        ) & (reverse_quality[:, raw_ids] >= config.minimum_cycle_quality)
        cycle_error[: config.train_end_frame, selected] = selected_cycle_error
        cycle_valid[: config.train_end_frame, selected] = selected_cycle_valid
        multiview_tracks[camera][:, selected] = selected_forward_tracks
        multiview_quality[camera][:, selected] = selected_forward_quality

        for frame in range(config.train_end_frame):
            object_mask = np.asarray(
                processed_masks[frame][camera]["object"], dtype=bool
            )
            distance = distance_transform_edt(object_mask) / max(object_mask.shape)
            frame_tracks = selected_forward_tracks[frame]
            pixels = np.rint(frame_tracks).astype(np.int64)
            inside = _pixels_inside_mask(frame_tracks, object_mask)
            values = np.zeros(len(selected), dtype=np.float32)
            indexes = np.flatnonzero(inside)
            values[indexes] = distance[
                pixels[indexes, 1], pixels[indexes, 0]
            ]
            boundary[frame, selected] = values

        projected, eligible, surface_distance = _initial_multiview_eligibility(
            mapping.source_world_points,
            mapping.camera_points[camera],
            np.asarray(processed_masks[0][camera]["object"], dtype=bool),
            intrinsics[camera],
            camera_to_world[camera],
            depth_tolerance_m=config.multiview_initial_depth_tolerance_m,
        )
        initial_eligible[camera] = eligible
        initial_surface_distance[camera] = surface_distance.astype(np.float32)
        cross_view = np.flatnonzero(eligible & (mapping.source_camera != camera))
        if len(cross_view):
            cross_prediction = runner.track(video, projected[cross_view].astype(np.float32))
            multiview_tracks[camera][:, cross_view] = cross_prediction.tracks_xy
            multiview_quality[camera][:, cross_view] = (
                cross_prediction.visibility_probability
                * cross_prediction.confidence_probability
            )
        for frame in range(config.train_end_frame):
            object_mask = np.asarray(
                processed_masks[frame][camera]["object"], dtype=bool
            )
            inside = _pixels_inside_mask(multiview_tracks[camera, frame], object_mask)
            multiview_valid[camera, frame] = (
                initial_eligible[camera]
                & inside
                & (
                    multiview_quality[camera, frame]
                    >= config.minimum_multiview_quality
                )
            )

    triangulated_points, reprojection_error, multiview_camera_count = (
        triangulate_multiview_tracks(
        multiview_tracks,
        multiview_valid,
        multiview_quality,
        intrinsics,
        camera_to_world,
        )
    )
    reprojection_full = np.zeros((frame_count, track_count), dtype=np.float32)
    reprojection_valid_full = np.zeros((frame_count, track_count), dtype=bool)
    camera_count_full = np.zeros((frame_count, track_count), dtype=np.int16)
    triangulated = np.isfinite(reprojection_error) & (multiview_camera_count >= 2)
    packed_triangulation = pack_multiview_triangulation(
        triangulated_points,
        reprojection_error,
        multiview_camera_count,
        frame_count=frame_count,
    )
    reprojection_prefix = reprojection_full[: config.train_end_frame]
    reprojection_prefix[triangulated] = reprojection_error[triangulated]
    reprojection_valid_full[: config.train_end_frame] = triangulated
    camera_count_full[: config.train_end_frame] = multiview_camera_count
    network_quality = confidence * visibility
    cues.update(
        {
            "confidence": confidence,
            "visibility_probability": visibility,
            "cotracker_quality_probability": network_quality,
            "forward_backward_error_px": cycle_error,
            "forward_backward_valid": cycle_valid,
            "multiview_reprojection_error_px": reprojection_full,
            "multiview_valid": reprojection_valid_full,
            "multiview_camera_count": camera_count_full,
            **packed_triangulation,
            "boundary_distance": boundary,
            "cue_available": cue_available,
            "source_camera": mapping.source_camera,
            "source_track": mapping.source_track,
            "initial_match_distance_m": mapping.initial_match_distance_m.astype(
                np.float32
            ),
            "source_tracks_xy": source_tracks,
            "multiview_initial_eligible": initial_eligible,
            "multiview_initial_surface_distance_m": initial_surface_distance,
            "multiview_tracks_xy_prefix": multiview_tracks,
            "multiview_quality_probability_prefix": multiview_quality,
            "multiview_view_valid_prefix": multiview_valid,
            "multiview_intrinsics": intrinsics.astype(np.float64),
            "multiview_camera_to_world": camera_to_world.astype(np.float64),
        }
    )
    output = Path(output_npz_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **cues)

    fit_visible = mapping.final_visible[: config.train_end_frame]
    cycle_selection = cycle_valid[: config.train_end_frame] & fit_visible
    multiview_selection = triangulated & fit_visible
    tracker_revision = _git_revision(tracker_root)
    summary: dict[str, Any] = {
        "schema_version": 2,
        "config": asdict(config),
        "contract": {
            "source_queries": "all 5000 archived frame-zero queries rerun jointly per camera",
            "source_probability": "separate low-level CoTracker3 visibility and <=12-pixel confidence probabilities",
            "forward_backward": "cycle disagreement over the training-video prefix only",
            "multiview": "weighted ray triangulation and RMS calibrated reprojection error",
            "multiview_coordinates": (
                "world-frame points in metres plus prefix-only cross-view tracks "
                "and weights; covariance is not calibrated"
            ),
            "causality": "no frame at or after train_end_frame is decoded or tracked",
            "held_out_storage": "future cue entries are neutral and cue_available is false",
        },
        "software": {
            "cotracker_root": str(tracker_root.resolve()),
            "cotracker_revision": tracker_revision,
            "checkpoint": {
                "path": str(checkpoint.resolve()),
                "sha256": _sha256(checkpoint),
            },
        },
        "inputs": {
            "final_data": str(Path(final_data_path).resolve()),
            "raw_case_dir": str(raw_path.resolve()),
            "base_cues": None if base_cues_path is None else str(Path(base_cues_path).resolve()),
        },
        "frame_count": frame_count,
        "training_frame_count": config.train_end_frame,
        "track_count": track_count,
        "camera_count": camera_count,
        "archive_track_parity_error_px": parity,
        "training_visible_cues": {
            "visibility_probability": _distribution(
                visibility[: config.train_end_frame][fit_visible]
            ),
            "confidence_probability": _distribution(
                confidence[: config.train_end_frame][fit_visible]
            ),
            "quality_probability": _distribution(
                network_quality[: config.train_end_frame][fit_visible]
            ),
            "forward_backward_valid_fraction": float(
                np.mean(cycle_valid[: config.train_end_frame][fit_visible])
            ),
            "forward_backward_error_px": _distribution(
                cycle_error[: config.train_end_frame][cycle_selection]
            ),
            "multiview_valid_fraction": float(np.mean(multiview_selection)),
            "multiview_reprojection_error_px": _distribution(
                reprojection_error[multiview_selection]
            ),
            "boundary_distance": _distribution(
                boundary[: config.train_end_frame][fit_visible]
            ),
        },
        "multiview_initial_eligible_fraction_by_camera": {
            str(camera): float(np.mean(initial_eligible[camera]))
            for camera in range(camera_count)
        },
        "output_npz": str(output.resolve()),
    }
    return summary


def write_cotracker3_cue_summary(
    summary: dict[str, Any], path: str | Path
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
