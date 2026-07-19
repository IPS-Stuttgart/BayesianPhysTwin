"""Causal raw-camera measurements for the open Deform360 belief protocol.

Measurement construction is deliberately separated from outcome evaluation.
The builder accepts a sealed frame-zero physical prediction, calibrated RGB
videos, and frame-zero visibility assets.  It has no target or outcome input.
For update frame ``u``, AllTracker receives exactly video frames ``[0, u]``.
Tracked pixels are robustly triangulated, and only the resulting sparse 3-D
measurements are passed to the recursive belief evaluator.

The released Deform360 target trajectory is itself a reconstructed proxy.  It
is opened only by :func:`evaluate_raw_camera_measurement_cohort`, after the
measurement archive and its manifest already exist.  This protocol is an
outcome-open development experiment, not official Deform360 Table-4 parity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import pickle
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .deform360_online_belief_evaluation import (
    ARMS,
    EXPECTED_SOURCE_EPISODES,
    PRIMARY_METRICS,
    UPDATE_FRAMES,
    _physical_object_cluster_bootstrap,
    _resolve_prediction_archive,
    _sha256,
    _validate_deform360_outcome_manifest,
    evaluate_deform360_online_belief_arrays,
)
from .phystwin_online_belief import deterministic_farthest_point_ids


PROTOCOL_ID = "deform360-raw-camera-alltracker-v1-development"
MEASUREMENT_FILENAME = "measurement.npz"
MANIFEST_FILENAME = "measurement_manifest.json"
ALLTRACKER_MOLMOMOTION_REVISION = "61f5b21b694ad8f854ec7ecd2400005acc73f685"
ALLTRACKER_SOURCE_TREE = "eab012351a01051fa31a348e5b1f21c0e1ed970d"
ALLTRACKER_RUNTIME_SOURCE_SHA256 = (
    "388b4b893ee5d206c5b396be5321fee05339d4d39efa74dc00fdea1a3f447817"
)
ALLTRACKER_CHECKPOINT_SHA256 = (
    "ffd9ebcfb6d206d594b646999a150540f92c049cf9b2bf940facf7123f62aa1d"
)


@dataclass(frozen=True)
class RawCameraObservationConfig:
    """Outcome-free choices for sparse multiview observations."""

    center_count: int = 16
    selected_camera_count: int = 8
    minimum_initial_view_count: int = 2
    minimum_triangulation_view_count: int = 2
    minimum_ray_angle_degrees: float = 2.0
    frame_zero_depth_tolerance_m: float = 0.015
    reprojection_inlier_threshold_px: float = 3.0
    maximum_reprojection_median_px: float = 3.0
    maximum_displacement_from_initial_m: float = 0.5
    alltracker_max_side: int = 512
    alltracker_inference_iterations: int = 4
    alltracker_window_length: int = 16
    visibility_threshold: float = 0.5
    update_frames: tuple[int, ...] = UPDATE_FRAMES

    def __post_init__(self) -> None:
        if self.center_count < 1:
            raise ValueError("center_count must be positive")
        if self.selected_camera_count < 2:
            raise ValueError("at least two cameras are required")
        if self.minimum_initial_view_count < 2:
            raise ValueError("initial visibility requires at least two views")
        if self.minimum_triangulation_view_count < 2:
            raise ValueError("triangulation requires at least two views")
        if not 0.0 < self.minimum_ray_angle_degrees < 180.0:
            raise ValueError("minimum ray angle must lie in (0, 180)")
        if self.frame_zero_depth_tolerance_m <= 0.0:
            raise ValueError("frame-zero depth tolerance must be positive")
        if self.reprojection_inlier_threshold_px <= 0.0:
            raise ValueError("reprojection threshold must be positive")
        if self.maximum_reprojection_median_px <= 0.0:
            raise ValueError("maximum reprojection median must be positive")
        if self.maximum_displacement_from_initial_m <= 0.0:
            raise ValueError("maximum displacement must be positive")
        if self.alltracker_max_side < 64:
            raise ValueError("AllTracker maximum side is implausibly small")
        if self.alltracker_inference_iterations < 1:
            raise ValueError("AllTracker iterations must be positive")
        if self.alltracker_window_length < 2:
            raise ValueError("AllTracker window length must exceed one")
        if not 0.0 < self.visibility_threshold < 1.0:
            raise ValueError("visibility threshold must lie in (0, 1)")
        if tuple(sorted(set(self.update_frames))) != self.update_frames:
            raise ValueError("update_frames must be strictly increasing")


def expected_open_case_names() -> tuple[str, ...]:
    """Return only the explicitly outcome-open 27-case panel."""

    return tuple(
        f"{object_id}-ep{episode_id:04d}"
        for object_id, episode_ids in EXPECTED_SOURCE_EPISODES.items()
        for episode_id in episode_ids
    )


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    """Hash one materialized array, including its dtype and shape."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _source_tree_sha256(root: Path) -> str:
    """Hash every vendored source asset with deterministic path framing."""

    digest = hashlib.sha256()
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _load_calibration(
    processed_episode_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    intrinsic_path = processed_episode_dir / "undistorted_intrinsics.npy"
    extrinsic_path = processed_episode_dir / "extrinsics.npy"
    intrinsics = np.load(intrinsic_path, allow_pickle=True).item()
    extrinsics = np.load(extrinsic_path, allow_pickle=True).item()
    if not isinstance(intrinsics, dict) or not isinstance(extrinsics, dict):
        raise ValueError("Deform360 calibration archives must contain dictionaries")
    return intrinsics, extrinsics


def project_world_points(
    points_m: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points to ``(x, y)`` pixels and camera-z depth."""

    points = np.asarray(points_m, dtype=float)
    k = np.asarray(intrinsics, dtype=float)
    c2w = np.asarray(camera_to_world, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_m must have shape (N, 3)")
    if k.shape != (3, 3) or c2w.shape != (4, 4):
        raise ValueError("invalid camera calibration shape")
    world_to_camera = np.linalg.inv(c2w)
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=float)))
    camera = (world_to_camera @ homogeneous.T).T[:, :3]
    depth = camera[:, 2]
    pixels = np.full((len(points), 2), np.nan, dtype=float)
    front = depth > 1e-9
    pixels[front, 0] = k[0, 0] * camera[front, 0] / depth[front] + k[0, 2]
    pixels[front, 1] = k[1, 1] * camera[front, 1] / depth[front] + k[1, 2]
    return pixels, depth


def _read_h5_frame_zero(path: Path) -> np.ndarray:
    """Read exactly the first HDF5 frame, never a future slice."""

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - server-only dependency
        raise RuntimeError("h5py is required for Deform360 frame-zero assets") from exc
    with h5py.File(path, "r") as handle:
        if "data" not in handle or handle["data"].ndim != 3:
            raise ValueError(f"unsupported Deform360 HDF5 payload: {path}")
        return np.asarray(handle["data"][0])


def frame_zero_camera_support(
    frame_zero_points_m: np.ndarray,
    processed_episode_dir: str | Path,
    intrinsics: Mapping[str, Any],
    extrinsics: Mapping[str, Any],
    *,
    depth_tolerance_m: float,
) -> tuple[tuple[str, ...], np.ndarray, dict[str, np.ndarray]]:
    """Visibility support using only masks/depth at frame zero.

    A point must project in front of the camera, land inside its frame-zero
    SAM2 mask, and agree with frame-zero rendered depth.  The HDF5 files also
    contain later frames, but this routine indexes only element zero.
    """

    root = Path(processed_episode_dir)
    cameras = tuple(
        sorted(
            camera
            for camera in set(intrinsics) & set(extrinsics)
            if (root / camera / "undistorted.mp4").is_file()
        )
    )
    if len(cameras) < 2:
        raise ValueError("fewer than two calibrated Deform360 cameras")
    points = np.asarray(frame_zero_points_m, dtype=float)
    support = np.zeros((len(points), len(cameras)), dtype=bool)
    projected: dict[str, np.ndarray] = {}
    for camera_index, camera in enumerate(cameras):
        pixels, depth = project_world_points(
            points,
            np.asarray(intrinsics[camera]),
            np.asarray(extrinsics[camera]),
        )
        projected[camera] = pixels
        mask = _read_h5_frame_zero(root / camera / "mask_refined.h5").astype(bool)
        encoded_depth = _read_h5_frame_zero(root / camera / "rendered_depth.h5")
        depth_map_m = np.asarray(encoded_depth, dtype=float) / 1000.0
        if mask.shape != depth_map_m.shape:
            raise ValueError(f"mask/depth shape differs for {camera}")
        height, width = mask.shape
        rounded = np.rint(pixels).astype(np.int64)
        inside = (
            (depth > 0.0)
            & np.all(np.isfinite(pixels), axis=1)
            & (rounded[:, 0] >= 0)
            & (rounded[:, 0] < width)
            & (rounded[:, 1] >= 0)
            & (rounded[:, 1] < height)
        )
        ids = np.flatnonzero(inside)
        if len(ids):
            rows = rounded[ids, 1]
            columns = rounded[ids, 0]
            sampled_depth = depth_map_m[rows, columns]
            support[ids, camera_index] = (
                mask[rows, columns]
                & (sampled_depth > 0.0)
                & (np.abs(sampled_depth - depth[ids]) <= depth_tolerance_m)
            )
    return cameras, support, projected


def _maximum_ray_angle_degrees(
    point_m: np.ndarray,
    camera_indices: Sequence[int],
    camera_origins_m: np.ndarray,
) -> float:
    maximum = 0.0
    for first, second in itertools.combinations(camera_indices, 2):
        first_ray = point_m - camera_origins_m[first]
        second_ray = point_m - camera_origins_m[second]
        denominator = np.linalg.norm(first_ray) * np.linalg.norm(second_ray)
        if denominator <= 1e-12:
            continue
        cosine = float(np.dot(first_ray, second_ray) / denominator)
        angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
        # Both nearly parallel (0 degrees) and nearly antiparallel (180 degrees)
        # rays are ill-conditioned for triangulation.  Report the conventional
        # acute triangulation angle, whose optimum is 90 degrees.
        maximum = max(
            maximum,
            min(angle, 180.0 - angle),
        )
    return maximum


def select_frame_zero_observation_plan(
    frame_zero_points_m: np.ndarray,
    cameras: Sequence[str],
    support: np.ndarray,
    projected_pixels: Mapping[str, np.ndarray],
    extrinsics: Mapping[str, Any],
    *,
    config: RawCameraObservationConfig,
) -> dict[str, Any]:
    """Select geometry-spanning centers and a deterministic camera subset."""

    points = np.asarray(frame_zero_points_m, dtype=float)
    camera_names = tuple(cameras)
    supported = np.asarray(support, dtype=bool)
    if supported.shape != (len(points), len(camera_names)):
        raise ValueError("support shape differs from points/cameras")
    if len(camera_names) < config.selected_camera_count:
        raise ValueError("fewer cameras than the fixed selected-camera count")
    origins = np.stack(
        [np.asarray(extrinsics[camera], dtype=float)[:3, 3] for camera in camera_names]
    )
    candidate_ids: list[int] = []
    for point_id in range(len(points)):
        views = np.flatnonzero(supported[point_id])
        if len(views) < config.minimum_initial_view_count:
            continue
        if (
            _maximum_ray_angle_degrees(points[point_id], views, origins)
            < config.minimum_ray_angle_degrees
        ):
            continue
        candidate_ids.append(point_id)
    candidates = np.asarray(candidate_ids, dtype=np.int64)
    if len(candidates) < config.center_count:
        raise ValueError("too few multiview-visible frame-zero candidates")
    centers = deterministic_farthest_point_ids(
        points,
        candidates,
        config.center_count,
    )

    best_subset: tuple[int, ...] | None = None
    best_score: tuple[int, int, int, float] | None = None
    for subset in itertools.combinations(
        range(len(camera_names)), config.selected_camera_count
    ):
        counts = np.sum(supported[centers][:, subset], axis=1)
        angles = [
            _maximum_ray_angle_degrees(
                points[point_id],
                [index for index in subset if supported[point_id, index]],
                origins,
            )
            for point_id in centers
            if counts[np.flatnonzero(centers == point_id)[0]] >= 2
        ]
        score = (
            int(np.sum(counts >= config.minimum_initial_view_count)),
            int(np.sum(counts >= 3)),
            int(np.sum(counts)),
            0.0 if not angles else float(np.median(angles)),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_subset = subset
    if best_subset is None or best_score is None:
        raise AssertionError("camera selection produced no subset")
    selected_cameras = tuple(camera_names[index] for index in best_subset)
    query_ids = {
        camera: centers[supported[centers, camera_names.index(camera)]].astype(np.int64)
        for camera in selected_cameras
    }
    query_pixels = {
        camera: np.asarray(projected_pixels[camera], dtype=float)[query_ids[camera]]
        for camera in selected_cameras
    }
    return {
        "candidate_ids": candidates,
        "center_ids": centers,
        "selected_cameras": selected_cameras,
        "selected_camera_indices": np.asarray(best_subset, dtype=np.int64),
        "selection_score": best_score,
        "query_ids": query_ids,
        "query_pixels": query_pixels,
        "support": supported,
        "camera_names": camera_names,
    }


def _projection_matrix(
    intrinsics: np.ndarray, camera_to_world: np.ndarray
) -> np.ndarray:
    return (
        np.asarray(intrinsics, dtype=float)
        @ np.linalg.inv(np.asarray(camera_to_world, dtype=float))[:3]
    )


def _linear_triangulation(
    observations: Sequence[tuple[str, np.ndarray]],
    projection_matrices: Mapping[str, np.ndarray],
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for camera, pixel in observations:
        matrix = projection_matrices[camera]
        x, y = np.asarray(pixel, dtype=float)
        rows.extend((x * matrix[2] - matrix[0], y * matrix[2] - matrix[1]))
    _, _, right = np.linalg.svd(np.stack(rows))
    homogeneous = right[-1]
    if abs(float(homogeneous[3])) <= 1e-12:
        raise ValueError("triangulation produced a point at infinity")
    result = homogeneous[:3] / homogeneous[3]
    if not np.all(np.isfinite(result)):
        raise ValueError("triangulation produced a non-finite point")
    return result


def _reproject(
    point_m: np.ndarray, projection_matrix: np.ndarray
) -> tuple[np.ndarray, float]:
    homogeneous = projection_matrix @ np.append(np.asarray(point_m, dtype=float), 1.0)
    depth = float(homogeneous[2])
    if abs(depth) <= 1e-12:
        return np.full(2, np.nan), depth
    return homogeneous[:2] / depth, depth


def triangulate_observation_ransac(
    observations: Mapping[str, np.ndarray],
    projection_matrices: Mapping[str, np.ndarray],
    camera_origins_m: Mapping[str, np.ndarray],
    initial_point_m: np.ndarray,
    *,
    config: RawCameraObservationConfig,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Robust DLT triangulation with deterministic two-view hypotheses."""

    ordered = tuple(
        (camera, np.asarray(observations[camera], dtype=float))
        for camera in sorted(observations)
        if camera in projection_matrices
        and np.asarray(observations[camera]).shape == (2,)
        and np.all(np.isfinite(observations[camera]))
    )
    diagnostic: dict[str, Any] = {
        "available_view_count": len(ordered),
        "accepted": False,
        "decision": "insufficient_views",
    }
    if len(ordered) < config.minimum_triangulation_view_count:
        return None, diagnostic
    best: tuple[tuple[int, float], np.ndarray] | None = None
    for pair in itertools.combinations(ordered, 2):
        try:
            hypothesis = _linear_triangulation(pair, projection_matrices)
        except (ValueError, np.linalg.LinAlgError):
            continue
        errors = []
        for camera, pixel in ordered:
            projected, depth = _reproject(hypothesis, projection_matrices[camera])
            errors.append(
                np.inf if depth <= 0.0 else float(np.linalg.norm(projected - pixel))
            )
        inliers = np.asarray(errors) <= config.reprojection_inlier_threshold_px
        inlier_errors = np.asarray(errors)[inliers]
        key = (
            int(np.sum(inliers)),
            -float(np.median(inlier_errors)) if len(inlier_errors) else -np.inf,
        )
        if best is None or key > best[0]:
            best = (key, inliers)
    if best is None or best[0][0] < config.minimum_triangulation_view_count:
        diagnostic["decision"] = "ransac_support_failure"
        return None, diagnostic
    inlier_observations = tuple(
        observation for observation, keep in zip(ordered, best[1]) if keep
    )
    try:
        point = _linear_triangulation(inlier_observations, projection_matrices)
    except (ValueError, np.linalg.LinAlgError):
        diagnostic["decision"] = "refit_failure"
        return None, diagnostic
    reprojection_errors: list[float] = []
    for camera, pixel in inlier_observations:
        projected, depth = _reproject(point, projection_matrices[camera])
        if depth <= 0.0:
            diagnostic["decision"] = "negative_depth"
            return None, diagnostic
        reprojection_errors.append(float(np.linalg.norm(projected - pixel)))
    median_reprojection = float(np.median(reprojection_errors))
    camera_names = [camera for camera, _ in inlier_observations]
    origins = np.stack([camera_origins_m[camera] for camera in camera_names])
    ray_angle = _maximum_ray_angle_degrees(
        point,
        list(range(len(origins))),
        origins,
    )
    displacement = float(
        np.linalg.norm(point - np.asarray(initial_point_m, dtype=float))
    )
    diagnostic.update(
        {
            "inlier_view_count": len(inlier_observations),
            "inlier_cameras": camera_names,
            "median_reprojection_error_px": median_reprojection,
            "maximum_ray_angle_degrees": ray_angle,
            "displacement_from_initial_m": displacement,
        }
    )
    if median_reprojection > config.maximum_reprojection_median_px:
        diagnostic["decision"] = "reprojection_failure"
        return None, diagnostic
    if ray_angle < config.minimum_ray_angle_degrees:
        diagnostic["decision"] = "ray_angle_failure"
        return None, diagnostic
    if displacement > config.maximum_displacement_from_initial_m:
        diagnostic["decision"] = "displacement_failure"
        return None, diagnostic
    diagnostic["accepted"] = True
    diagnostic["decision"] = "accepted"
    return point.astype(np.float32), diagnostic


class AllTrackerPrefixRuntime:
    """One-model runtime for exact causal AllTracker video prefixes."""

    def __init__(
        self,
        source_root: str | Path,
        checkpoint: str | Path,
        *,
        device: str,
        config: RawCameraObservationConfig,
    ) -> None:
        self.source_root = Path(source_root).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        self.device_name = str(device)
        self.config = config
        if not (self.source_root / "nets" / "alltracker.py").is_file():
            raise FileNotFoundError(self.source_root / "nets" / "alltracker.py")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        self.source_sha256 = _source_tree_sha256(self.source_root)
        if self.source_sha256 != ALLTRACKER_RUNTIME_SOURCE_SHA256:
            raise ValueError("AllTracker runtime source differs from the frozen tree")
        self.checkpoint_sha256 = _sha256(self.checkpoint)
        if self.checkpoint_sha256 != ALLTRACKER_CHECKPOINT_SHA256:
            raise ValueError("AllTracker checkpoint differs from the frozen checksum")
        sys.path.insert(0, str(self.source_root))
        try:
            import torch
            from nets.alltracker import Net
        except ImportError as exc:  # pragma: no cover - GPU integration only
            raise RuntimeError(
                "AllTracker runtime dependencies are unavailable"
            ) from exc
        self._torch = torch
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = False
        self._device = torch.device(self.device_name)
        model = Net(config.alltracker_window_length)
        try:
            payload = torch.load(
                self.checkpoint,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:  # pragma: no cover - older torch
            payload = torch.load(self.checkpoint, map_location="cpu")
        state = (
            payload["model"]
            if isinstance(payload, dict) and "model" in payload
            else payload
        )
        model.load_state_dict(state, strict=True)
        self._model = model.to(self._device).eval()
        for parameter in self._model.parameters():
            parameter.requires_grad = False

    def close(self) -> None:
        self._model = None
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def track_prefix(
        self,
        video_path: str | Path,
        query_pixels_xy: np.ndarray,
        update_frame: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Track frame-zero pixels using exactly frames ``[0, update]``."""

        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - GPU integration only
            raise RuntimeError("OpenCV is required for AllTracker video input") from exc
        queries = np.asarray(query_pixels_xy, dtype=float)
        if (
            queries.ndim != 2
            or queries.shape[1] != 2
            or not np.all(np.isfinite(queries))
        ):
            raise ValueError("query pixels must have finite shape (N, 2)")
        capture = cv2.VideoCapture(str(video_path))
        frames: list[np.ndarray] = []
        prefix_digest = hashlib.sha256()
        try:
            for frame_index in range(update_frame + 1):
                okay, bgr = capture.read()
                if not okay:
                    raise ValueError(
                        f"cannot read causal frame {frame_index} from {video_path}"
                    )
                rgb_frame = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                frames.append(rgb_frame)
                prefix_digest.update(str(rgb_frame.dtype).encode("ascii"))
                prefix_digest.update(
                    np.asarray(rgb_frame.shape, dtype=np.int64).tobytes()
                )
                prefix_digest.update(rgb_frame.tobytes())
        finally:
            capture.release()
        rgb = np.stack(frames)
        original_height, original_width = rgb.shape[1:3]
        scale = min(
            1.0,
            self.config.alltracker_max_side / max(original_height, original_width),
        )
        height = max(8, int(original_height * scale) // 8 * 8)
        width = max(8, int(original_width * scale) // 8 * 8)
        if (height, width) != (original_height, original_width):
            rgb = np.stack(
                [
                    cv2.resize(
                        frame,
                        (width, height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    for frame in rgb
                ]
            )
        torch = self._torch
        video = (
            torch.from_numpy(np.ascontiguousarray(rgb))
            .permute(0, 3, 1, 2)[None]
            .float()
            .to(self._device)
        )
        start = time.perf_counter()
        with torch.no_grad():
            flows, visibility_confidence, _, _ = self._model(
                video,
                iters=self.config.alltracker_inference_iterations,
                sw=None,
                is_training=False,
            )
        if flows.ndim == 4:
            flows = flows[:, None]
            flows = torch.cat((torch.zeros_like(flows[:, :1]), flows), dim=1)
        if visibility_confidence.ndim == 4:
            visibility_confidence = visibility_confidence[:, None]
        if visibility_confidence.shape[1] == flows.shape[1] - 1:
            visibility_confidence = torch.cat(
                (torch.ones_like(visibility_confidence[:, :1]), visibility_confidence),
                dim=1,
            )
        if flows.shape[1] != update_frame + 1:
            raise ValueError(
                "AllTracker output does not match the exact causal prefix length"
            )
        y_grid, x_grid = torch.meshgrid(
            torch.arange(height, device=self._device),
            torch.arange(width, device=self._device),
            indexing="ij",
        )
        grid = torch.stack((x_grid, y_grid), dim=0)[None, None].float()
        trajectories = flows + grid
        x_query = np.clip(
            np.rint(queries[:, 0] * width / original_width).astype(np.int64),
            0,
            width - 1,
        )
        y_query = np.clip(
            np.rint(queries[:, 1] * height / original_height).astype(np.int64),
            0,
            height - 1,
        )
        current = trajectories[0, update_frame, :, y_query, x_query].T
        current[:, 0] *= original_width / width
        current[:, 1] *= original_height / height
        confidence = visibility_confidence[0, update_frame, 0, y_query, x_query]
        visible = confidence > self.config.visibility_threshold
        runtime = time.perf_counter() - start
        tracks = current.detach().cpu().numpy().astype(np.float32)
        mask = visible.detach().cpu().numpy().astype(bool)
        confidence_np = confidence.detach().cpu().numpy().astype(np.float32)
        del video, flows, visibility_confidence, trajectories, current, confidence
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return (
            tracks,
            mask,
            {
                "prefix_frame_range_half_open": [0, update_frame + 1],
                "maximum_video_frame_read": update_frame,
                "decoded_frame_count": update_frame + 1,
                "decoded_rgb_prefix_sha256": prefix_digest.hexdigest(),
                "original_image_shape": [original_height, original_width],
                "inference_image_shape": [height, width],
                "query_count": len(queries),
                "visible_query_count": int(np.sum(mask)),
                "visibility_confidence_minimum": (
                    None if not len(confidence_np) else float(np.min(confidence_np))
                ),
                "visibility_confidence_median": (
                    None if not len(confidence_np) else float(np.median(confidence_np))
                ),
                "visibility_confidence_maximum": (
                    None if not len(confidence_np) else float(np.max(confidence_np))
                ),
                "runtime_seconds": runtime,
            },
        )

    def track_reversed_prefix(
        self,
        video_path: str | Path,
        endpoint_query_pixels_xy: np.ndarray,
        update_frame: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Track update-frame pixels back through exactly reversed frames ``[0,u]``."""

        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - GPU integration only
            raise RuntimeError("OpenCV is required for AllTracker video input") from exc
        queries = np.asarray(endpoint_query_pixels_xy, dtype=float)
        if (
            queries.ndim != 2
            or queries.shape[1] != 2
            or not np.all(np.isfinite(queries))
        ):
            raise ValueError("endpoint query pixels must have finite shape (N, 2)")
        capture = cv2.VideoCapture(str(video_path))
        frames: list[np.ndarray] = []
        prefix_digest = hashlib.sha256()
        try:
            for frame_index in range(update_frame + 1):
                okay, bgr = capture.read()
                if not okay:
                    raise ValueError(
                        f"cannot read causal frame {frame_index} from {video_path}"
                    )
                rgb_frame = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                frames.append(rgb_frame)
                prefix_digest.update(str(rgb_frame.dtype).encode("ascii"))
                prefix_digest.update(
                    np.asarray(rgb_frame.shape, dtype=np.int64).tobytes()
                )
                prefix_digest.update(rgb_frame.tobytes())
        finally:
            capture.release()
        rgb = np.stack(frames[::-1])
        original_height, original_width = rgb.shape[1:3]
        scale = min(
            1.0,
            self.config.alltracker_max_side / max(original_height, original_width),
        )
        height = max(8, int(original_height * scale) // 8 * 8)
        width = max(8, int(original_width * scale) // 8 * 8)
        if (height, width) != (original_height, original_width):
            rgb = np.stack(
                [
                    cv2.resize(
                        frame,
                        (width, height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    for frame in rgb
                ]
            )
        torch = self._torch
        video = (
            torch.from_numpy(np.ascontiguousarray(rgb))
            .permute(0, 3, 1, 2)[None]
            .float()
            .to(self._device)
        )
        start = time.perf_counter()
        with torch.no_grad():
            flows, visibility_confidence, _, _ = self._model(
                video,
                iters=self.config.alltracker_inference_iterations,
                sw=None,
                is_training=False,
            )
        if flows.ndim == 4:
            flows = flows[:, None]
            flows = torch.cat((torch.zeros_like(flows[:, :1]), flows), dim=1)
        if visibility_confidence.ndim == 4:
            visibility_confidence = visibility_confidence[:, None]
        if visibility_confidence.shape[1] == flows.shape[1] - 1:
            visibility_confidence = torch.cat(
                (torch.ones_like(visibility_confidence[:, :1]), visibility_confidence),
                dim=1,
            )
        if flows.shape[1] != update_frame + 1:
            raise ValueError(
                "AllTracker reverse output does not match the exact causal prefix length"
            )
        y_grid, x_grid = torch.meshgrid(
            torch.arange(height, device=self._device),
            torch.arange(width, device=self._device),
            indexing="ij",
        )
        grid = torch.stack((x_grid, y_grid), dim=0)[None, None].float()
        trajectories = flows + grid
        x_query = np.clip(
            np.rint(queries[:, 0] * width / original_width).astype(np.int64),
            0,
            width - 1,
        )
        y_query = np.clip(
            np.rint(queries[:, 1] * height / original_height).astype(np.int64),
            0,
            height - 1,
        )
        recovered = trajectories[0, update_frame, :, y_query, x_query].T
        recovered[:, 0] *= original_width / width
        recovered[:, 1] *= original_height / height
        confidence = visibility_confidence[0, update_frame, 0, y_query, x_query]
        visible = confidence > self.config.visibility_threshold
        runtime = time.perf_counter() - start
        tracks = recovered.detach().cpu().numpy().astype(np.float32)
        mask = visible.detach().cpu().numpy().astype(bool)
        confidence_np = confidence.detach().cpu().numpy().astype(np.float32)
        del video, flows, visibility_confidence, trajectories, recovered, confidence
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return (
            tracks,
            mask,
            {
                "direction": "reverse_exact_prefix",
                "source_prefix_frame_range_half_open": [0, update_frame + 1],
                "maximum_source_video_frame_read": update_frame,
                "decoded_frame_count": update_frame + 1,
                "decoded_rgb_prefix_sha256": prefix_digest.hexdigest(),
                "model_frame_order": [update_frame, 0],
                "original_image_shape": [original_height, original_width],
                "inference_image_shape": [height, width],
                "query_count": len(queries),
                "visible_query_count": int(np.sum(mask)),
                "visibility_confidence_minimum": (
                    None if not len(confidence_np) else float(np.min(confidence_np))
                ),
                "visibility_confidence_median": (
                    None if not len(confidence_np) else float(np.median(confidence_np))
                ),
                "visibility_confidence_maximum": (
                    None if not len(confidence_np) else float(np.max(confidence_np))
                ),
                "runtime_seconds": runtime,
            },
        )


def _causal_selected_camera_inputs(
    processed_episode_dir: Path,
    selected_cameras: Sequence[str],
    update_diagnostics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Describe camera inputs without reading bytes beyond causal slices."""

    prefix_hashes: dict[str, dict[str, str]] = {
        camera: {} for camera in selected_cameras
    }
    for update in update_diagnostics:
        frame = int(update["frame"])
        for tracker in update["tracker"]:
            camera = str(tracker["camera"])
            if camera not in prefix_hashes:
                raise ValueError("tracker diagnostic names an unselected camera")
            if tracker.get("maximum_video_frame_read") != frame:
                raise ValueError("tracker diagnostic crossed its causal prefix")
            prefix_hashes[camera][str(frame)] = str(
                tracker["decoded_rgb_prefix_sha256"]
            )
    result: dict[str, Any] = {}
    for camera in selected_cameras:
        camera_dir = processed_episode_dir / camera
        mask_zero = _read_h5_frame_zero(camera_dir / "mask_refined.h5")
        depth_zero = _read_h5_frame_zero(camera_dir / "rendered_depth.h5")
        result[camera] = {
            "video": {
                "path": str(camera_dir / "undistorted.mp4"),
                "decoded_prefix_sha256_by_update": prefix_hashes[camera],
                "whole_file_hashed_or_read": False,
            },
            "frame_zero_mask": {
                "path": str(camera_dir / "mask_refined.h5"),
                "frame_zero_array_sha256": _array_sha256(mask_zero),
                "only_index_read": 0,
                "whole_file_hashed_or_read": False,
            },
            "frame_zero_depth": {
                "path": str(camera_dir / "rendered_depth.h5"),
                "frame_zero_array_sha256": _array_sha256(depth_zero),
                "only_index_read": 0,
                "whole_file_hashed_or_read": False,
            },
        }
    return result


def _validate_prediction_seal(seal: Mapping[str, Any]) -> None:
    if seal.get("artifact_kind") != "Deform360IndependentSourcePredictionSeal":
        raise ValueError("unsupported Deform360 prediction seal")
    boundary = seal.get("information_boundary", {})
    if not (
        boundary.get("object_observation_frames_used") == [0]
        and boundary.get("future_object_track_read") is False
        and boundary.get("prediction_hashed_before_future_outcome_scoring") is True
    ):
        raise ValueError("physical prediction crossed the frame-zero boundary")


def build_raw_camera_measurement_case(
    panel_case_dir: str | Path,
    processed_episode_dir: str | Path,
    output_dir: str | Path,
    runtime: AllTrackerPrefixRuntime,
    *,
    config: RawCameraObservationConfig | None = None,
) -> dict[str, Any]:
    """Build one measurement archive without opening an outcome or target."""

    cfg = config or runtime.config
    case_dir = Path(panel_case_dir).resolve()
    processed = Path(processed_episode_dir).resolve()
    output = Path(output_dir).resolve()
    if case_dir.name not in expected_open_case_names():
        raise ValueError("case is outside the explicit outcome-open panel")
    seal_path = case_dir / "prediction_seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    _validate_prediction_seal(seal)
    archive_path = _resolve_prediction_archive(case_dir, seal)
    with np.load(archive_path, allow_pickle=False) as stored:
        prior = np.asarray(stored["prediction_m"]).copy()
        persistence = np.asarray(stored["persistence_m"]).copy()
        frame_zero = np.asarray(stored["frame_zero_points_m"]).copy()
    if prior.shape != persistence.shape or prior.ndim != 3 or prior.shape[2] != 3:
        raise ValueError("sealed trajectories have invalid shape")
    if frame_zero.shape != prior.shape[1:]:
        raise ValueError("sealed frame-zero point shape differs from trajectory")
    if cfg.update_frames[-1] >= len(prior):
        raise ValueError("sealed trajectory does not reach every update")
    intrinsics, extrinsics = _load_calibration(processed)
    cameras, support, projected = frame_zero_camera_support(
        frame_zero,
        processed,
        intrinsics,
        extrinsics,
        depth_tolerance_m=cfg.frame_zero_depth_tolerance_m,
    )
    plan = select_frame_zero_observation_plan(
        frame_zero,
        cameras,
        support,
        projected,
        extrinsics,
        config=cfg,
    )
    centers = np.asarray(plan["center_ids"], dtype=np.int64)
    candidates = np.asarray(plan["candidate_ids"], dtype=np.int64)
    selected_cameras = tuple(plan["selected_cameras"])
    projection_matrices = {
        camera: _projection_matrix(intrinsics[camera], extrinsics[camera])
        for camera in selected_cameras
    }
    camera_origins = {
        camera: np.asarray(extrinsics[camera], dtype=float)[:3, 3]
        for camera in selected_cameras
    }
    measurement = np.full(prior.shape, np.nan, dtype=np.float32)
    measurement_visibility = np.zeros(prior.shape[:2], dtype=bool)
    measurement_validity = np.zeros(prior.shape[:2], dtype=bool)
    measurement[0, candidates] = frame_zero[candidates]
    measurement_visibility[0, candidates] = True
    measurement_validity[0, candidates] = True
    inlier_count = np.zeros((len(cfg.update_frames), len(centers)), dtype=np.int16)
    reprojection_median = np.full(
        (len(cfg.update_frames), len(centers)), np.nan, dtype=np.float32
    )
    ray_angle = np.full_like(reprojection_median, np.nan)
    tracker_visibility = np.zeros_like(inlier_count)
    update_diagnostics: list[dict[str, Any]] = []

    for update_index, update_frame in enumerate(cfg.update_frames):
        tracks_by_camera: dict[str, dict[int, np.ndarray]] = {}
        tracker_records: list[dict[str, Any]] = []
        for camera in selected_cameras:
            query_ids = np.asarray(plan["query_ids"][camera], dtype=np.int64)
            query_pixels = np.asarray(plan["query_pixels"][camera], dtype=float)
            tracks, visible, tracker_record = runtime.track_prefix(
                processed / camera / "undistorted.mp4",
                query_pixels,
                update_frame,
            )
            tracks_by_camera[camera] = {
                int(point_id): tracks[index]
                for index, point_id in enumerate(query_ids)
                if visible[index]
            }
            tracker_record.update(
                {
                    "camera": camera,
                    "query_ids": query_ids.tolist(),
                }
            )
            tracker_records.append(tracker_record)
        center_records: list[dict[str, Any]] = []
        for center_index, center_id in enumerate(centers):
            observations = {
                camera: tracks_by_camera[camera][int(center_id)]
                for camera in selected_cameras
                if int(center_id) in tracks_by_camera[camera]
            }
            tracker_visibility[update_index, center_index] = len(observations)
            point, diagnostic = triangulate_observation_ransac(
                observations,
                projection_matrices,
                camera_origins,
                frame_zero[center_id],
                config=cfg,
            )
            diagnostic["center_id"] = int(center_id)
            center_records.append(diagnostic)
            if point is None:
                continue
            measurement[update_frame, center_id] = point
            measurement_visibility[update_frame, center_id] = True
            measurement_validity[update_frame, center_id] = True
            inlier_count[update_index, center_index] = int(
                diagnostic["inlier_view_count"]
            )
            reprojection_median[update_index, center_index] = float(
                diagnostic["median_reprojection_error_px"]
            )
            ray_angle[update_index, center_index] = float(
                diagnostic["maximum_ray_angle_degrees"]
            )
        update_diagnostics.append(
            {
                "frame": update_frame,
                "prefix_frame_range_half_open": [0, update_frame + 1],
                "maximum_video_frame_read": update_frame,
                "tracker": tracker_records,
                "centers": center_records,
                "accepted_center_count": int(
                    np.sum(measurement_validity[update_frame, centers])
                ),
            }
        )

    output.mkdir(parents=True, exist_ok=False)
    archive_output = output / MEASUREMENT_FILENAME
    np.savez_compressed(
        archive_output,
        measurement_m=measurement,
        measurement_visibility=measurement_visibility,
        measurement_validity=measurement_validity,
        candidate_ids=candidates,
        center_ids=centers,
        selected_cameras=np.asarray(selected_cameras),
        update_frames=np.asarray(cfg.update_frames, dtype=np.int64),
        tracker_visible_view_count=tracker_visibility,
        triangulation_inlier_view_count=inlier_count,
        triangulation_median_reprojection_px=reprojection_median,
        triangulation_maximum_ray_angle_degrees=ray_angle,
    )
    input_paths = {
        "prediction_seal": seal_path,
        "prediction_archive": archive_path,
        "intrinsics": processed / "undistorted_intrinsics.npy",
        "extrinsics": processed / "extrinsics.npy",
    }
    selected_camera_inputs = _causal_selected_camera_inputs(
        processed,
        selected_cameras,
        update_diagnostics,
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360CausalRawCameraMeasurement",
        "protocol_id": PROTOCOL_ID,
        "case": case_dir.name,
        "object_id": str(seal["object_id"]),
        "episode_id": int(seal["episode_id"]),
        "episode_key": str(seal["episode_key"]),
        "config": asdict(cfg),
        "plan": {
            "candidate_count": len(candidates),
            "candidate_ids": candidates.tolist(),
            "center_ids": centers.tolist(),
            "selected_cameras": list(selected_cameras),
            "selection_score": list(plan["selection_score"]),
            "selection_inputs": (
                "sealed frame-zero points, calibration, and HDF5 index zero only"
            ),
        },
        "tracker": {
            "name": "AllTracker",
            "molmomotion_revision": ALLTRACKER_MOLMOMOTION_REVISION,
            "source_tree": ALLTRACKER_SOURCE_TREE,
            "runtime_source_sha256": runtime.source_sha256,
            "source_root": str(runtime.source_root),
            "checkpoint": str(runtime.checkpoint),
            "checkpoint_sha256": runtime.checkpoint_sha256,
            "device": runtime.device_name,
        },
        "inputs": {
            key: {"path": str(path), "sha256": _sha256(path)}
            for key, path in input_paths.items()
        },
        "selected_camera_inputs": selected_camera_inputs,
        "updates": update_diagnostics,
        "output": {
            "measurement_archive": str(archive_output),
            "measurement_archive_sha256": _sha256(archive_output),
            "accepted_measurement_count_by_update": [
                int(np.sum(measurement_validity[frame, centers]))
                for frame in cfg.update_frames
            ],
        },
        "information_boundary": {
            "target_data_read": False,
            "outcome_manifest_read": False,
            "future_reconstruction_after_frame_zero_read": False,
            "video_prefix_rule": "update u reads exactly frames [0, u]",
            "maximum_video_frame_read_by_update": list(cfg.update_frames),
            "frame_zero_hdf5_indices_read": [0],
        },
        "claim_boundary": (
            "outcome-unaware raw-RGB measurement construction on an already-open "
            "development panel; evaluation occurs in a separate process"
        ),
    }
    manifest["result_sha256"] = _canonical_sha256(manifest)
    manifest_path = output / MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_raw_camera_measurement_cohort(
    panel_root: str | Path,
    processed_root: str | Path,
    output_root: str | Path,
    runtime: AllTrackerPrefixRuntime,
    *,
    config: RawCameraObservationConfig | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, Any]:
    """Build one deterministic shard of the explicit open-27 panel."""

    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid shard index/count")
    panel = Path(panel_root).resolve()
    processed = Path(processed_root).resolve()
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases = expected_open_case_names()
    selected = [
        case for index, case in enumerate(cases) if index % shard_count == shard_index
    ]
    built: list[dict[str, Any]] = []
    for case in selected:
        case_output = output / case
        if case_output.exists():
            manifest_path = case_output / MANIFEST_FILENAME
            if not manifest_path.is_file():
                raise ValueError(f"incomplete existing output: {case_output}")
            built.append(json.loads(manifest_path.read_text(encoding="utf-8")))
            continue
        built.append(
            build_raw_camera_measurement_case(
                panel / case,
                processed / case / "episode_0000",
                case_output,
                runtime,
                config=config,
            )
        )
    summary = {
        "protocol_id": PROTOCOL_ID,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "case_count": len(built),
        "cases": [record["case"] for record in built],
        "measurement_manifest_sha256": {
            record["case"]: _sha256(output / record["case"] / MANIFEST_FILENAME)
            for record in built
        },
    }
    (output / f"build-shard-{shard_index:02d}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _load_measurement_artifact(
    case_dir: Path,
    measurement_dir: Path,
    seal: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest_path = measurement_dir / MANIFEST_FILENAME
    archive_path = measurement_dir / MEASUREMENT_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    claimed_result_sha256 = manifest.get("result_sha256")
    unsigned_manifest = dict(manifest)
    unsigned_manifest.pop("result_sha256", None)
    if claimed_result_sha256 != _canonical_sha256(unsigned_manifest):
        raise ValueError("measurement manifest content checksum changed")
    if manifest.get("artifact_kind") != "Deform360CausalRawCameraMeasurement":
        raise ValueError("unsupported raw-camera measurement artifact")
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("raw-camera protocol ID changed")
    for key in ("object_id", "episode_id", "episode_key"):
        if manifest.get(key) != seal.get(key):
            raise ValueError(f"measurement {key} differs from prediction seal")
    if manifest.get("information_boundary", {}).get("target_data_read") is not False:
        raise ValueError("measurement manifest crossed the target boundary")
    if manifest.get("output", {}).get("measurement_archive_sha256") != _sha256(
        archive_path
    ):
        raise ValueError("measurement archive checksum changed")
    if manifest.get("inputs", {}).get("prediction_seal", {}).get("sha256") != _sha256(
        case_dir / "prediction_seal.json"
    ):
        raise ValueError("measurement was built from a different prediction seal")
    with np.load(archive_path, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]).copy() for name in stored.files}
    return manifest, arrays


def _load_open_case_for_evaluation(
    case_dir: Path,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seal_path = case_dir / "prediction_seal.json"
    target_path = case_dir / "target_data.pkl"
    outcome_path = case_dir / "outcome.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    _validate_prediction_seal(seal)
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    if (
        outcome.get("information_boundary", {}).get(
            "source_future_opened_for_outcome_construction"
        )
        is not True
    ):
        raise ValueError("source outcome is not open")
    _validate_deform360_outcome_manifest(seal_path, target_path, seal, outcome)
    archive_path = _resolve_prediction_archive(case_dir, seal)
    with np.load(archive_path, allow_pickle=False) as stored:
        prior = np.asarray(stored["prediction_m"]).copy()
        persistence = np.asarray(stored["persistence_m"]).copy()
        frame_zero = np.asarray(stored["frame_zero_points_m"]).copy()
    with target_path.open("rb") as handle:
        target_data = pickle.load(handle)
    target = np.asarray(target_data["object_points"])
    visibility = np.asarray(target_data["object_visibilities"], dtype=bool)
    validity = np.asarray(target_data["object_motions_valid"], dtype=bool)
    if not np.array_equal(target[0].astype(np.float32), frame_zero.astype(np.float32)):
        raise ValueError("target frame zero differs from sealed prediction")
    return seal, prior, persistence, target, visibility, validity


def evaluate_raw_camera_measurement_case(
    panel_case_dir: str | Path,
    measurement_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate one already-hashed measurement against its open outcome."""

    case_dir = Path(panel_case_dir).resolve()
    if case_dir.name not in expected_open_case_names():
        raise ValueError("case is outside the explicit outcome-open panel")
    seal = json.loads((case_dir / "prediction_seal.json").read_text(encoding="utf-8"))
    _validate_prediction_seal(seal)
    manifest, measurement_arrays = _load_measurement_artifact(
        case_dir,
        Path(measurement_dir).resolve(),
        seal,
    )
    # The target/outcome files are opened only after the measurement artifact has
    # been loaded and both of its independent checksums have been verified.
    open_seal, prior, persistence, target, visibility, validity = (
        _load_open_case_for_evaluation(case_dir)
    )
    if open_seal != seal:
        raise ValueError("prediction seal changed while opening the outcome")
    measurement = np.asarray(measurement_arrays["measurement_m"], dtype=float)
    measurement_visibility = np.asarray(
        measurement_arrays["measurement_visibility"], dtype=bool
    )
    measurement_validity = np.asarray(
        measurement_arrays["measurement_validity"], dtype=bool
    )
    raw_report, raw_arrays = evaluate_deform360_online_belief_arrays(
        prior,
        persistence,
        target,
        visibility,
        validity,
        measurement_m=measurement,
        measurement_visibility=measurement_visibility,
        measurement_validity=measurement_validity,
    )
    centers = np.asarray(raw_report["center_ids"], dtype=np.int64)
    oracle_measurement = np.full(target.shape, np.nan, dtype=float)
    oracle_visibility = np.zeros(target.shape[:2], dtype=bool)
    oracle_validity = np.zeros(target.shape[:2], dtype=bool)
    frame_zero_ids = np.flatnonzero(measurement_visibility[0] & measurement_validity[0])
    oracle_measurement[0, frame_zero_ids] = target[0, frame_zero_ids]
    oracle_visibility[0, frame_zero_ids] = True
    oracle_validity[0, frame_zero_ids] = True
    for frame in UPDATE_FRAMES:
        supported = (
            measurement_visibility[frame]
            & measurement_validity[frame]
            & visibility[frame]
            & validity[frame]
            & np.all(np.isfinite(target[frame]), axis=1)
        )
        oracle_measurement[frame, supported] = target[frame, supported]
        oracle_visibility[frame, supported] = True
        oracle_validity[frame, supported] = True
    oracle_report, _ = evaluate_deform360_online_belief_arrays(
        prior,
        persistence,
        target,
        visibility,
        validity,
        measurement_m=oracle_measurement,
        measurement_visibility=oracle_visibility,
        measurement_validity=oracle_validity,
    )
    if raw_report["center_ids"] != oracle_report["center_ids"]:
        raise AssertionError("raw and same-support oracle centers differ")
    observation_error_by_update: list[dict[str, Any]] = []
    for frame in UPDATE_FRAMES:
        supported = (
            measurement_visibility[frame, centers]
            & measurement_validity[frame, centers]
            & visibility[frame, centers]
            & validity[frame, centers]
            & np.all(np.isfinite(measurement[frame, centers]), axis=1)
            & np.all(np.isfinite(target[frame, centers]), axis=1)
        )
        errors = np.linalg.norm(
            measurement[frame, centers[supported]] - target[frame, centers[supported]],
            axis=1,
        )
        observation_error_by_update.append(
            {
                "frame": frame,
                "count": len(errors),
                "mean_m": None if not len(errors) else float(np.mean(errors)),
                "median_m": None if not len(errors) else float(np.median(errors)),
                "p90_m": None if not len(errors) else float(np.quantile(errors, 0.9)),
                "maximum_m": None if not len(errors) else float(np.max(errors)),
            }
        )
    report = {
        "protocol_id": PROTOCOL_ID,
        "case": case_dir.name,
        "object_id": str(seal["object_id"]),
        "episode_id": int(seal["episode_id"]),
        "measurement_manifest_sha256": _sha256(
            Path(measurement_dir) / MANIFEST_FILENAME
        ),
        "measurement_archive_sha256": _sha256(
            Path(measurement_dir) / MEASUREMENT_FILENAME
        ),
        "measurement_result_sha256": manifest["result_sha256"],
        "raw_measurement": raw_report,
        "same_support_target_oracle": oracle_report,
        "observation_error_by_update": observation_error_by_update,
        "information_boundary": {
            "measurement_hashed_before_target_open_in_this_evaluator": True,
            "measurement_builder_target_read": False,
            "target_role": "scoring only and explicitly labeled same-support oracle",
        },
    }
    return report, raw_arrays


def evaluate_raw_camera_measurement_cohort(
    panel_root: str | Path,
    measurement_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Evaluate the complete hashed raw-camera open-27 panel."""

    panel = Path(panel_root).resolve()
    measurements = Path(measurement_root).resolve()
    output = Path(output_dir).resolve()
    missing = [
        case
        for case in expected_open_case_names()
        if not (measurements / case / MANIFEST_FILENAME).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing raw-camera measurements: {missing}")
    output.mkdir(parents=True, exist_ok=False)
    reports: list[dict[str, Any]] = []
    groups: dict[str, str] = {}
    artifacts: list[dict[str, str]] = []
    for case in expected_open_case_names():
        report, arrays = evaluate_raw_camera_measurement_case(
            panel / case,
            measurements / case,
        )
        groups[case] = str(report["object_id"])
        report_path = output / f"{case}.json"
        arrays_path = output / f"{case}.npz"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        np.savez_compressed(arrays_path, **arrays)
        artifacts.append(
            {
                "case": case,
                "report_sha256": _sha256(report_path),
                "arrays_sha256": _sha256(arrays_path),
            }
        )
        reports.append(report)
    aggregate = {
        stream: {
            arm: {
                metric: float(
                    np.mean(
                        [report[stream]["scores"][arm][metric] for report in reports]
                    )
                )
                for metric in PRIMARY_METRICS
            }
            for arm in ARMS
        }
        for stream in ("raw_measurement", "same_support_target_oracle")
    }
    comparisons: dict[str, Any] = {}
    for stream in ("raw_measurement", "same_support_target_oracle"):
        for arm in (
            "recursive_rbf_risk_limited",
            "recursive_rbf_causal_continuation",
            "recursive_rbf_correspondence_safe",
            "risk_limited_frozen_current_state",
        ):
            for metric in PRIMARY_METRICS:
                differences = {
                    str(report["case"]): float(
                        report[stream]["scores"][arm][metric]
                        - report[stream]["scores"]["physical_prior"][metric]
                    )
                    for report in reports
                }
                result = _physical_object_cluster_bootstrap(differences, groups)
                baseline = aggregate[stream]["physical_prior"][metric]
                result["relative_change"] = (
                    None
                    if baseline == 0.0
                    else aggregate[stream][arm][metric] / baseline - 1.0
                )
                result["episode_wins"] = int(
                    np.sum(np.asarray(list(differences.values())) < 0.0)
                )
                comparisons[f"{stream}:{arm}:vs_physical:{metric}"] = result
    observation_errors = [
        update
        for report in reports
        for update in report["observation_error_by_update"]
        if update["mean_m"] is not None
    ]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "episode_count": len(reports),
        "physical_object_count": len(set(groups.values())),
        "aggregate": aggregate,
        "comparisons": comparisons,
        "observation_error": {
            "case_update_count": len(observation_errors),
            "mean_of_case_update_means_m": float(
                np.mean([value["mean_m"] for value in observation_errors])
            ),
            "mean_of_case_update_medians_m": float(
                np.mean([value["median_m"] for value in observation_errors])
            ),
            "maximum_case_update_error_m": float(
                np.max([value["maximum_m"] for value in observation_errors])
            ),
        },
        "measurement_support": {
            "accepted_center_count_by_update": [
                int(
                    report["raw_measurement"]["updates"][update_index][
                        "available_center_count"
                    ]
                )
                for report in reports
                for update_index in range(len(UPDATE_FRAMES))
            ],
            "accepted_belief_update_count": int(
                sum(
                    bool(update["accepted"])
                    for report in reports
                    for update in report["raw_measurement"]["updates"]
                )
            ),
        },
        "artifacts": artifacts,
        "claim_boundary": (
            "outcome-open development transfer using causal raw RGB prefixes and "
            "AllTracker multiview triangulation; target is reconstructed proxy; "
            "not an official Deform360 or open-loop SOTA result"
        ),
    }
    summary["result_sha256"] = _canonical_sha256(summary)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = [
    "ALLTRACKER_MOLMOMOTION_REVISION",
    "ALLTRACKER_SOURCE_TREE",
    "AllTrackerPrefixRuntime",
    "MANIFEST_FILENAME",
    "MEASUREMENT_FILENAME",
    "PROTOCOL_ID",
    "RawCameraObservationConfig",
    "build_raw_camera_measurement_case",
    "build_raw_camera_measurement_cohort",
    "evaluate_raw_camera_measurement_case",
    "evaluate_raw_camera_measurement_cohort",
    "expected_open_case_names",
    "frame_zero_camera_support",
    "project_world_points",
    "select_frame_zero_observation_plan",
    "triangulate_observation_ransac",
]
