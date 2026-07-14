"""Sparse multiview geometry artifacts for the locked Deform360 replication."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .deform360_object_sam2 import DeformableObjectSam2VideoPredictor
from .deform360_visual_hull import carve_candidate_points, regular_grid_in_bounds


REPLICATION_MASK_SCHEMA_VERSION = 1
REPLICATION_HULL_SCHEMA_VERSION = 2


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    descriptor = _canonical_bytes(
        {"dtype": array.dtype.str, "shape": list(array.shape)}
    )
    return hashlib.sha256(
        descriptor + b"\0" + array.view(np.uint8).tobytes()
    ).hexdigest()


@dataclass(frozen=True)
class ReplicationGeometryConfig:
    """Source-locked geometry choices shared by every object stratum."""

    prefix_frame_count: int = 6
    score_frame_stride: int = 6
    minimum_camera_count: int = 8
    initial_cube_half_extent_m: float = 0.5
    initial_voxel_resolution: int = 120
    initial_minimum_hull_points: int = 64
    initial_maximum_mask_dilation_pixels: int = 5
    local_voxel_size_filament_m: float = 0.004
    local_voxel_size_sheet_m: float = 0.008
    local_voxel_size_volumetric_m: float = 0.010
    local_initial_margin_m: float = 0.05
    local_expansion_factor: float = 1.6
    local_maximum_expansion_attempts: int = 3
    local_maximum_grid_point_count: int = 500_000
    consensus_fraction_of_peak: float = 0.55
    minimum_consensus_votes: int = 8
    minimum_hull_point_count: int = 64
    minimum_available_future_frame_fraction: float = 0.90
    maximum_archived_hull_points: int = 2048

    def __post_init__(self) -> None:
        _require(self.prefix_frame_count >= 2, "prefix must contain two frames")
        _require(self.score_frame_stride >= 1, "score stride must be positive")
        _require(self.minimum_camera_count >= 2, "geometry needs two cameras")
        _require(self.initial_voxel_resolution >= 16, "initial grid is too coarse")
        _require(self.initial_minimum_hull_points >= 16, "initial hull is too small")
        _require(
            self.initial_maximum_mask_dilation_pixels >= 0,
            "initial mask dilation must be nonnegative",
        )
        for value in (
            self.local_voxel_size_filament_m,
            self.local_voxel_size_sheet_m,
            self.local_voxel_size_volumetric_m,
            self.local_initial_margin_m,
        ):
            _require(value > 0.0, "geometry length scales must be positive")
        _require(self.local_expansion_factor > 1.0, "expansion factor must exceed one")
        _require(
            self.local_maximum_expansion_attempts >= 1,
            "geometry needs one local attempt",
        )
        _require(
            self.maximum_archived_hull_points >= 32,
            "too few archived hull points",
        )
        _require(
            0.0 < self.minimum_available_future_frame_fraction <= 1.0,
            "invalid available-frame fraction",
        )

    def voxel_size_for_stratum(self, stratum: str) -> float:
        values = {
            "filament": self.local_voxel_size_filament_m,
            "sheet": self.local_voxel_size_sheet_m,
            "volumetric": self.local_voxel_size_volumetric_m,
        }
        _require(stratum in values, f"unsupported geometry stratum: {stratum}")
        return values[stratum]


def replication_geometry_frame_indices(
    frame_count: int,
    prefix_start_frame: int,
    config: ReplicationGeometryConfig,
    *,
    prefix_only: bool = False,
) -> tuple[int, ...]:
    """Select the prefix endpoint and the fixed future scoring grid."""

    _require(frame_count >= 2, "episode has too few frames")
    _require(prefix_start_frame >= 0, "prefix start must be nonnegative")
    prefix_endpoint = prefix_start_frame + config.prefix_frame_count - 1
    _require(prefix_endpoint < frame_count, "prefix endpoint exceeds the episode")
    if prefix_only:
        return (prefix_endpoint,)
    future = list(range(prefix_endpoint + 1, frame_count, config.score_frame_stride))
    _require(future, "prefix leaves no future score frame")
    if future[-1] != frame_count - 1:
        future.append(frame_count - 1)
    return tuple([prefix_endpoint, *future])


def pack_multiview_masks(masks: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Pack ``(C,T,H,W)`` boolean masks along image columns."""

    values = np.asarray(masks, dtype=bool)
    _require(values.ndim == 4, "multiview masks must have shape (C,T,H,W)")
    _require(all(size > 0 for size in values.shape), "mask array is empty")
    return np.packbits(values, axis=3), (values.shape[2], values.shape[3])


def unpack_multiview_masks(
    packed: np.ndarray, image_shape: Sequence[int]
) -> np.ndarray:
    """Invert :func:`pack_multiview_masks` without pickle-backed arrays."""

    values = np.asarray(packed, dtype=np.uint8)
    shape = tuple(map(int, image_shape))
    _require(values.ndim == 4 and len(shape) == 2, "packed mask shape is invalid")
    height, width = shape
    _require(values.shape[2] == height, "packed mask height differs")
    return np.unpackbits(values, axis=3, count=width).astype(bool)


def _extract_selected_frames(
    video_path: Path, frame_indices: Sequence[int], output_dir: Path
) -> list[Path]:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - host integration
        raise RuntimeError("OpenCV is required for replication geometry") from error
    requested = tuple(map(int, frame_indices))
    _require(requested and list(requested) == sorted(set(requested)), "invalid frames")
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    _require(capture.isOpened(), f"cannot open aligned video: {video_path}")
    written = []
    try:
        next_output = 0
        requested_set = set(requested)
        frame = 0
        while next_output < len(requested):
            ok, image = capture.read()
            _require(ok, f"cannot decode frame {requested[next_output]} from {video_path}")
            if frame in requested_set:
                path = output_dir / f"{next_output:06d}.jpg"
                _require(cv2.imwrite(str(path), image), f"cannot write sampled frame: {path}")
                written.append(path)
                next_output += 1
            frame += 1
    finally:
        capture.release()
    return written


def build_replication_mask_archive(
    episode_dir: str | Path,
    cameras: Sequence[str],
    frame_indices: Sequence[int],
    predictor: DeformableObjectSam2VideoPredictor,
    reference_rgb: np.ndarray,
    reference_mask: np.ndarray,
    output_archive_path: str | Path,
    *,
    reference_camera: str,
    initial_masks_by_camera: Mapping[str, np.ndarray] | None = None,
    fallback_initial_masks_by_camera: Mapping[str, np.ndarray] | None = None,
    scratch_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run pinned SAM2 on exact sampled frames and write a packed mask archive."""

    directory = Path(episode_dir).resolve()
    selected_cameras = tuple(map(str, cameras))
    _require(len(selected_cameras) == len(set(selected_cameras)), "camera repeated")
    _require(len(selected_cameras) >= 2, "two cameras are required")
    indices = tuple(map(int, frame_indices))
    _require(indices and list(indices) == sorted(set(indices)), "frame indices invalid")
    output = Path(output_archive_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    scratch = (
        Path(scratch_dir).resolve()
        if scratch_dir is not None
        else output.parent / f".{output.stem}-frames"
    )
    scratch.mkdir(parents=True, exist_ok=True)
    camera_masks = []
    camera_records = []
    try:
        for camera in selected_cameras:
            frame_dir = scratch / camera
            if frame_dir.exists():
                shutil.rmtree(frame_dir)
            paths = _extract_selected_frames(
                directory / camera / "undistorted.mp4", indices, frame_dir
            )
            _require(len(paths) == len(indices), "sampled frame count differs")
            if initial_masks_by_camera is None:
                try:
                    initial, selection = predictor.select_initial_mask_with_reference(
                        frame_dir,
                        reference_rgb,
                        reference_mask,
                        reference_camera=reference_camera,
                    )
                    initialization = {
                        "policy": "source-reference-appearance",
                        "selection": selection,
                    }
                except ValueError as error:
                    if (
                        fallback_initial_masks_by_camera is None
                        or camera not in fallback_initial_masks_by_camera
                        or "no reference-consistent mask" not in str(error)
                    ):
                        raise
                    initial = np.asarray(
                        fallback_initial_masks_by_camera[camera], dtype=bool
                    )
                    initialization = {
                        "policy": "sealed-source-reference-camera-mask-fallback",
                        "fallback_reason": str(error),
                        "initial_mask_sha256": _sha256_array(initial),
                    }
            else:
                _require(camera in initial_masks_by_camera, f"initial mask missing: {camera}")
                initial = np.asarray(initial_masks_by_camera[camera], dtype=bool)
                initialization = {
                    "policy": "sealed-prefix-initial-mask",
                    "initial_mask_sha256": _sha256_array(initial),
                }
            before = len(predictor.diagnostics)
            propagated = list(
                predictor.segment_from_initial_mask(
                    frame_dir,
                    initial,
                    initialization=initialization,
                )
            )
            _require(
                [index for index, _ in propagated] == list(range(len(indices))),
                f"SAM2 frame ordering changed for {camera}",
            )
            masks = np.stack([mask for _, mask in propagated]).astype(bool)
            _require(len(predictor.diagnostics) == before + 1, "SAM2 diagnostic missing")
            camera_masks.append(masks)
            camera_records.append(
                {
                    "camera": camera,
                    "video_sha256": _sha256_file(
                        directory / camera / "undistorted.mp4"
                    ),
                    "initial_mask_sha256": _sha256_array(initial),
                    "mask_sequence_sha256": _sha256_array(masks),
                    "sam2": predictor.diagnostics[-1],
                }
            )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    masks = np.stack(camera_masks)
    _require(len({values.shape for values in camera_masks}) == 1, "mask shapes differ")
    packed, image_shape = pack_multiview_masks(masks)
    np.savez_compressed(
        output,
        frame_indices=np.asarray(indices, dtype=np.int32),
        cameras=np.asarray(selected_cameras),
        packed_masks=packed,
        image_shape=np.asarray(image_shape, dtype=np.int32),
    )
    payload = {
        "schema_version": REPLICATION_MASK_SCHEMA_VERSION,
        "artifact_kind": "Deform360ReplicationSampledSam2Masks",
        "episode_dir": str(directory),
        "frame_indices": list(indices),
        "cameras": list(selected_cameras),
        "camera_records": camera_records,
        "archive": {
            "path": str(output),
            "sha256": _sha256_file(output),
            "bytes": output.stat().st_size,
            "packed_masks_sha256": _sha256_array(packed),
            "image_shape": list(image_shape),
        },
        "information_boundary": {
            "maximum_raw_frame_index_read": max(indices),
            "only_declared_raw_frame_indices_archived": True,
        },
    }
    payload["result_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return payload


def load_replication_mask_archive(payload: Mapping[str, Any]) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Validate and load one packed sampled-mask artifact."""

    _require(payload.get("schema_version") == REPLICATION_MASK_SCHEMA_VERSION, "mask schema changed")
    expected = dict(payload)
    observed = expected.pop("result_sha256", None)
    _require(observed == hashlib.sha256(_canonical_bytes(expected)).hexdigest(), "mask checksum mismatch")
    archive = Path(payload["archive"]["path"])
    _require(archive.is_file() and _sha256_file(archive) == payload["archive"]["sha256"], "mask archive changed")
    with np.load(archive, allow_pickle=False) as stored:
        indices = np.asarray(stored["frame_indices"], dtype=np.int32)
        cameras = tuple(map(str, stored["cameras"].tolist()))
        packed = np.asarray(stored["packed_masks"], dtype=np.uint8)
        image_shape = np.asarray(stored["image_shape"], dtype=np.int32)
    _require(_sha256_array(packed) == payload["archive"]["packed_masks_sha256"], "packed masks changed")
    return cameras, indices, unpack_multiview_masks(packed, image_shape)


def _dilate_masks(
    masks: Mapping[str, np.ndarray], radius_pixels: int
) -> Mapping[str, np.ndarray]:
    if radius_pixels == 0:
        return masks
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - host integration
        raise RuntimeError("OpenCV is required for mask dilation") from error
    kernel = np.ones(
        (2 * radius_pixels + 1, 2 * radius_pixels + 1), dtype=np.uint8
    )
    return {
        camera: cv2.dilate(mask.astype(np.uint8), kernel) > 0
        for camera, mask in masks.items()
    }


def _local_hull(
    prior_hull: np.ndarray,
    masks: Mapping[str, np.ndarray],
    intrinsics: Mapping[str, np.ndarray],
    extrinsics: Mapping[str, np.ndarray],
    *,
    voxel_size_m: float,
    config: ReplicationGeometryConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    quantiles = np.quantile(prior_hull, [0.01, 0.99], axis=0)
    margin = config.local_initial_margin_m
    attempts = []
    hull = np.empty((0, 3), dtype=np.float64)
    for attempt in range(config.local_maximum_expansion_attempts):
        grid, grid_diagnostic = regular_grid_in_bounds(
            quantiles[0] - margin,
            quantiles[1] + margin,
            requested_voxel_size_m=voxel_size_m,
            maximum_point_count=config.local_maximum_grid_point_count,
        )
        dilation_attempts = []
        for radius in range(config.initial_maximum_mask_dilation_pixels + 1):
            hull, carve = carve_candidate_points(
                grid,
                _dilate_masks(masks, radius),
                intrinsics,
                extrinsics,
                consensus_fraction_of_peak=config.consensus_fraction_of_peak,
                minimum_consensus_votes=config.minimum_consensus_votes,
            )
            dilation_attempts.append(
                {"mask_dilation_radius_pixels": radius, "carving": carve}
            )
            if len(hull) >= config.minimum_hull_point_count:
                break
        attempts.append(
            {
                "attempt": attempt,
                "margin_m": margin,
                "grid": grid_diagnostic,
                "dilation_attempts": dilation_attempts,
            }
        )
        if len(hull) >= config.minimum_hull_point_count:
            break
        margin *= config.local_expansion_factor
    _require(
        len(hull) >= config.minimum_hull_point_count,
        f"local hull is too small (final_count={len(hull)})",
    )
    return hull, {"attempts": attempts, "final_hull_point_count": len(hull)}


def _initial_hull(
    masks: Mapping[str, np.ndarray],
    intrinsics: Mapping[str, np.ndarray],
    extrinsics: Mapping[str, np.ndarray],
    config: ReplicationGeometryConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    cameras = tuple(sorted(masks))
    centers = np.stack(
        [np.asarray(extrinsics[camera], dtype=np.float64)[:3, 3] for camera in cameras]
    )
    center = np.mean(centers, axis=0)
    axis = np.linspace(
        -config.initial_cube_half_extent_m,
        config.initial_cube_half_extent_m,
        config.initial_voxel_resolution,
    )
    grid = (
        np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
        .reshape(-1, 3)
        + center
    )
    attempts = []
    hull = np.empty((0, 3), dtype=np.float64)
    for radius in range(config.initial_maximum_mask_dilation_pixels + 1):
        hull, carving = carve_candidate_points(
            grid,
            _dilate_masks(masks, radius),
            intrinsics,
            extrinsics,
            consensus_fraction_of_peak=config.consensus_fraction_of_peak,
            minimum_consensus_votes=config.minimum_consensus_votes,
        )
        attempts.append({"mask_dilation_radius_pixels": radius, "carving": carving})
        if len(hull) >= config.initial_minimum_hull_points:
            break
    _require(
        len(hull) >= config.initial_minimum_hull_points,
        f"strict-consensus initial hull is too small (final_count={len(hull)})",
    )
    return hull, {
        "grid_center_world_m": center.tolist(),
        "grid_point_count": len(grid),
        "voxel_resolution": config.initial_voxel_resolution,
        "selected_mask_dilation_radius_pixels": attempts[-1][
            "mask_dilation_radius_pixels"
        ],
        "attempts": attempts,
        "carving": attempts[-1]["carving"],
    }


def _deterministic_subsample(points: np.ndarray, maximum_count: int) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if len(values) <= maximum_count:
        return values
    indices = np.linspace(0, len(values) - 1, maximum_count, dtype=np.int64)
    return values[indices]


def _geometry_quality(
    archived: Sequence[np.ndarray], config: ReplicationGeometryConfig
) -> dict[str, Any]:
    available = np.asarray([len(points) > 0 for points in archived], dtype=bool)
    _require(len(available) >= 1 and available[0], "prefix endpoint hull is unavailable")
    future_frame_count = max(0, len(available) - 1)
    available_future_fraction = (
        float(np.mean(available[1:])) if future_frame_count else 1.0
    )
    _require(
        available_future_fraction >= config.minimum_available_future_frame_fraction,
        "too many future hull observations are unavailable",
    )
    return {
        "available_frame_count": int(np.count_nonzero(available)),
        "total_frame_count": len(available),
        "available_future_frame_fraction": available_future_fraction,
    }


def build_replication_hull_archive(
    episode_dir: str | Path,
    mask_artifact: Mapping[str, Any],
    stratum: str,
    output_archive_path: str | Path,
    *,
    config: ReplicationGeometryConfig | None = None,
) -> dict[str, Any]:
    """Carve one initial and several local multiview hulls from sampled masks."""

    cfg = config or ReplicationGeometryConfig()
    directory = Path(episode_dir).resolve()
    cameras, frame_indices, all_masks = load_replication_mask_archive(mask_artifact)
    _require(len(cameras) >= cfg.minimum_camera_count, "too few locked cameras")
    try:
        from deform360.processing.episode import load_episode_calibration
    except ImportError as error:  # pragma: no cover - host integration
        raise RuntimeError("the pinned Deform360 runtime is required") from error
    intrinsics, extrinsics = load_episode_calibration(directory)
    selected_intrinsics = {camera: intrinsics[camera] for camera in cameras}
    selected_extrinsics = {camera: extrinsics[camera] for camera in cameras}
    archived = []
    diagnostics = []
    previous: np.ndarray | None = None
    for output_index, raw_frame in enumerate(frame_indices):
        masks = {
            camera: all_masks[camera_index, output_index]
            for camera_index, camera in enumerate(cameras)
            if np.any(all_masks[camera_index, output_index])
        }
        _require(len(masks) >= cfg.minimum_camera_count, "too many empty SAM2 masks")
        if previous is None:
            hull, initial = _initial_hull(
                masks,
                {camera: selected_intrinsics[camera] for camera in masks},
                {camera: selected_extrinsics[camera] for camera in masks},
                cfg,
            )
            diagnostic = {
                "method": "strict-consensus-global-hull",
                "hull_point_count": len(hull),
                **initial,
            }
        else:
            try:
                hull, local = _local_hull(
                    previous,
                    masks,
                    {camera: selected_intrinsics[camera] for camera in masks},
                    {camera: selected_extrinsics[camera] for camera in masks},
                    voxel_size_m=cfg.voxel_size_for_stratum(stratum),
                    config=cfg,
                )
                diagnostic = {"method": "prior-bounded-local-hull", **local}
            except ValueError as error:
                if "local hull is too small" not in str(error):
                    raise
                try:
                    hull, initial = _initial_hull(
                        masks,
                        {camera: selected_intrinsics[camera] for camera in masks},
                        {camera: selected_extrinsics[camera] for camera in masks},
                        cfg,
                    )
                except ValueError as fallback_error:
                    selected = np.empty((0, 3), dtype=np.float64)
                    archived.append(selected)
                    diagnostics.append(
                        {
                            "raw_frame_index": int(raw_frame),
                            "nonempty_camera_count": len(masks),
                            "archived_hull_point_count": 0,
                            "available": False,
                            "method": "unavailable-strict-consensus-hull",
                            "failure_reason": (
                                f"{error}; {fallback_error}"
                            ),
                        }
                    )
                    continue
                diagnostic = {
                    "method": "strict-consensus-global-hull-after-local-support-failure",
                    "fallback_reason": str(error),
                    "hull_point_count": len(hull),
                    **initial,
                }
        previous = np.asarray(hull, dtype=np.float64)
        selected = _deterministic_subsample(previous, cfg.maximum_archived_hull_points)
        archived.append(selected)
        diagnostics.append(
            {
                "raw_frame_index": int(raw_frame),
                "nonempty_camera_count": len(masks),
                "archived_hull_point_count": len(selected),
                "available": True,
                **diagnostic,
            }
        )
    geometry_quality = _geometry_quality(archived, cfg)
    offsets = np.zeros(len(archived) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(points) for points in archived])
    concatenated = np.concatenate(archived, axis=0)
    output = Path(output_archive_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        frame_indices=frame_indices,
        point_offsets=offsets,
        points_world_m=concatenated,
    )
    payload = {
        "schema_version": REPLICATION_HULL_SCHEMA_VERSION,
        "artifact_kind": "Deform360ReplicationSampledVisualHulls",
        "episode_dir": str(directory),
        "stratum": stratum,
        "config": asdict(cfg),
        "mask_result_sha256": mask_artifact["result_sha256"],
        "frame_indices": frame_indices.astype(int).tolist(),
        "frame_diagnostics": diagnostics,
        "geometry_quality": geometry_quality,
        "archive": {
            "path": str(output),
            "sha256": _sha256_file(output),
            "bytes": output.stat().st_size,
            "points_sha256": _sha256_array(concatenated),
            "offsets_sha256": _sha256_array(offsets),
        },
        "information_boundary": dict(mask_artifact["information_boundary"]),
    }
    payload["result_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return payload


def load_replication_hull_archive(
    payload: Mapping[str, Any],
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Load ragged hulls from a pickle-free concatenated archive."""

    _require(payload.get("schema_version") == REPLICATION_HULL_SCHEMA_VERSION, "hull schema changed")
    expected = dict(payload)
    observed = expected.pop("result_sha256", None)
    _require(observed == hashlib.sha256(_canonical_bytes(expected)).hexdigest(), "hull checksum mismatch")
    path = Path(payload["archive"]["path"])
    _require(path.is_file() and _sha256_file(path) == payload["archive"]["sha256"], "hull archive changed")
    with np.load(path, allow_pickle=False) as stored:
        frames = np.asarray(stored["frame_indices"], dtype=np.int32)
        offsets = np.asarray(stored["point_offsets"], dtype=np.int64)
        points = np.asarray(stored["points_world_m"], dtype=np.float64)
    _require(_sha256_array(points) == payload["archive"]["points_sha256"], "hull points changed")
    _require(_sha256_array(offsets) == payload["archive"]["offsets_sha256"], "hull offsets changed")
    hulls = tuple(points[offsets[index] : offsets[index + 1]] for index in range(len(frames)))
    return frames, hulls


__all__ = [
    "ReplicationGeometryConfig",
    "build_replication_hull_archive",
    "build_replication_mask_archive",
    "load_replication_hull_archive",
    "load_replication_mask_archive",
    "pack_multiview_masks",
    "replication_geometry_frame_indices",
    "unpack_multiview_masks",
]
