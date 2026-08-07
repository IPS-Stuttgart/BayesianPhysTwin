"""Recover independent per-camera reliability cues for processed PhysTwin tracks."""

from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PhysTwinRawCueConfig:
    """Mapping and normalization settings for raw camera cues."""

    initial_match_tolerance_m: float = 1e-6
    boundary_normalization: str = "maximum_image_dimension"


@dataclass(frozen=True)
class PhysTwinRawTrackMap:
    """Exact correspondence between processed object tracks and raw queries."""

    final_points: np.ndarray
    final_visible: np.ndarray
    camera_points: np.ndarray
    track_paths: tuple[Path, ...]
    tracks_by_camera: tuple[np.ndarray, ...]
    visibility_by_camera: tuple[np.ndarray, ...]
    source_camera: np.ndarray
    source_track: np.ndarray
    initial_match_distance_m: np.ndarray
    source_world_points: np.ndarray


def _load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
    }


def load_phystwin_raw_track_map(
    final_data_path: str | Path,
    raw_case_dir: str | Path,
    *,
    config: PhysTwinRawCueConfig | None = None,
    final_data_payload: Mapping[str, Any] | None = None,
) -> PhysTwinRawTrackMap:
    """Recover the release preprocessing's exact raw-query correspondence.

    ``final_data_payload`` lets digest-bound callers pass the exact mapping they
    already verified and deserialized. When omitted, the historical path-based
    behavior is retained for ordinary development callers.
    """

    cfg = config or PhysTwinRawCueConfig()
    if cfg.initial_match_tolerance_m <= 0.0:
        raise ValueError("initial_match_tolerance_m must be positive")
    try:
        from scipy.spatial import cKDTree
    except ImportError as error:
        raise RuntimeError("raw track mapping requires scipy") from error

    final_data = (
        _load_pickle(final_data_path)
        if final_data_payload is None
        else final_data_payload
    )
    if not isinstance(final_data, Mapping):
        raise TypeError("final_data_payload must contain a mapping")
    final_points = np.asarray(final_data["object_points"], dtype=float)
    final_visible = np.asarray(final_data["object_visibilities"], dtype=bool)
    if final_points.ndim != 3 or final_points.shape[2] != 3:
        raise ValueError("object_points must have shape (T, N, 3)")
    if final_visible.shape != final_points.shape[:2]:
        raise ValueError("object_visibilities must match object_points")
    frame_count, final_track_count, _ = final_points.shape

    raw_path = Path(raw_case_dir)
    track_paths = tuple(sorted((raw_path / "cotracker").glob("*.npz")))
    if not track_paths:
        raise FileNotFoundError("raw case contains no cotracker NPZ files")
    pcd_path = raw_path / "pcd" / "0.npz"
    if not pcd_path.is_file():
        raise FileNotFoundError("raw case requires pcd/0.npz")
    with np.load(pcd_path) as archive:
        camera_points = np.asarray(archive["points"], dtype=float)
    if camera_points.ndim != 4 or camera_points.shape[-1] != 3:
        raise ValueError("pcd/0.npz points must have shape (C, H, W, 3)")
    if len(track_paths) != len(camera_points):
        raise ValueError("camera count differs between tracks and pcd/0.npz")

    tracks_by_camera: list[np.ndarray] = []
    visibility_by_camera: list[np.ndarray] = []
    raw_initial_points: list[np.ndarray] = []
    clamped_initial_points: list[np.ndarray] = []
    camera_ids: list[np.ndarray] = []
    raw_track_ids: list[np.ndarray] = []
    for camera, path in enumerate(track_paths):
        with np.load(path) as archive:
            tracks = np.asarray(archive["tracks"], dtype=float)
            visibility = np.asarray(archive["visibility"], dtype=bool)
        if tracks.ndim != 3 or tracks.shape[0] != frame_count or tracks.shape[2] != 2:
            raise ValueError(f"{path.name} tracks have an incompatible shape")
        if visibility.shape != tracks.shape[:2]:
            raise ValueError(f"{path.name} visibility does not match tracks")
        pixels = np.rint(tracks[0]).astype(int)
        height, width = camera_points.shape[1:3]
        if np.any(pixels[:, 0] < 0) or np.any(pixels[:, 0] >= height):
            raise ValueError(f"{path.name} contains out-of-bounds initial rows")
        if np.any(pixels[:, 1] < 0) or np.any(pixels[:, 1] >= width):
            raise ValueError(f"{path.name} contains out-of-bounds initial columns")
        initial = camera_points[camera, pixels[:, 0], pixels[:, 1]].copy()
        clamped = initial.copy()
        # Match the released preprocessing's ground-plane clamp for lookup.
        clamped[clamped[:, 2] > 0.0, 2] = 0.0
        tracks_by_camera.append(tracks)
        visibility_by_camera.append(visibility)
        raw_initial_points.append(initial)
        clamped_initial_points.append(clamped)
        camera_ids.append(np.full(len(initial), camera, dtype=np.int16))
        raw_track_ids.append(np.arange(len(initial), dtype=np.int32))

    concatenated_initial = np.concatenate(clamped_initial_points, axis=0)
    concatenated_world = np.concatenate(raw_initial_points, axis=0)
    concatenated_camera = np.concatenate(camera_ids)
    concatenated_track = np.concatenate(raw_track_ids)
    match_distance, match_index = cKDTree(concatenated_initial).query(final_points[0])
    if float(np.max(match_distance)) > cfg.initial_match_tolerance_m:
        raise ValueError(
            "final tracks do not map to raw queries within tolerance; "
            f"maximum distance is {float(np.max(match_distance)):.6g} m"
        )
    if len(np.unique(match_index)) != final_track_count:
        raise ValueError("raw-query mapping is not one-to-one")
    return PhysTwinRawTrackMap(
        final_points=final_points,
        final_visible=final_visible,
        camera_points=camera_points,
        track_paths=track_paths,
        tracks_by_camera=tuple(tracks_by_camera),
        visibility_by_camera=tuple(visibility_by_camera),
        source_camera=concatenated_camera[match_index],
        source_track=concatenated_track[match_index],
        initial_match_distance_m=np.asarray(match_distance, dtype=float),
        source_world_points=concatenated_world[match_index],
    )


def build_phystwin_raw_camera_cues(
    final_data_path: str | Path,
    raw_case_dir: str | Path,
    output_npz_path: str | Path,
    *,
    config: PhysTwinRawCueConfig | None = None,
    base_cues_path: str | Path | None = None,
) -> dict[str, object]:
    """Map final tracks to raw queries and recover mask-boundary reliability.

    The released CoTracker archives retain binary visibility but not predicted
    confidence. Boundary distance is therefore the principal new independent
    cue; any existing motion cue can be merged through ``base_cues_path``.
    """

    cfg = config or PhysTwinRawCueConfig()
    if cfg.initial_match_tolerance_m <= 0.0:
        raise ValueError("initial_match_tolerance_m must be positive")
    if cfg.boundary_normalization != "maximum_image_dimension":
        raise ValueError("boundary_normalization must be 'maximum_image_dimension'")
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as error:
        raise RuntimeError("raw camera cues require scipy") from error
    mapping = load_phystwin_raw_track_map(
        final_data_path,
        raw_case_dir,
        config=cfg,
    )
    final_points = mapping.final_points
    final_visible = mapping.final_visible
    track_paths = mapping.track_paths
    tracks_by_camera = mapping.tracks_by_camera
    visibility_by_camera = mapping.visibility_by_camera
    source_camera = mapping.source_camera
    source_track = mapping.source_track
    match_distance = mapping.initial_match_distance_m
    frame_count, final_track_count, _ = final_points.shape

    raw_path = Path(raw_case_dir)
    mask_path = raw_path / "mask" / "processed_masks.pkl"
    if not mask_path.is_file():
        raise FileNotFoundError("raw case requires mask/processed_masks.pkl")

    raw_visibility = np.zeros((frame_count, final_track_count), dtype=bool)
    boundary_distance = np.zeros((frame_count, final_track_count), dtype=np.float32)
    processed_masks = _load_pickle(mask_path)
    for camera in range(len(track_paths)):
        selected = np.flatnonzero(source_camera == camera)
        selected_tracks = source_track[selected]
        tracks = tracks_by_camera[camera]
        visibility = visibility_by_camera[camera]
        raw_visibility[:, selected] = visibility[:, selected_tracks]
        for frame in range(frame_count):
            object_mask = np.asarray(
                processed_masks[frame][camera]["object"],
                dtype=bool,
            )
            distance = distance_transform_edt(object_mask) / max(object_mask.shape)
            pixels = np.rint(tracks[frame, selected_tracks]).astype(int)
            in_bounds = (
                (pixels[:, 0] >= 0)
                & (pixels[:, 0] < object_mask.shape[0])
                & (pixels[:, 1] >= 0)
                & (pixels[:, 1] < object_mask.shape[1])
            )
            values: np.ndarray = np.zeros(len(selected), dtype=float)
            values[in_bounds] = distance[
                pixels[in_bounds, 0],
                pixels[in_bounds, 1],
            ]
            boundary_distance[frame, selected] = values

    cues: dict[str, Any] = {}
    if base_cues_path is not None:
        with np.load(base_cues_path) as archive:
            cues.update({name: np.asarray(archive[name]) for name in archive.files})
    cues.update(
        {
            "confidence": raw_visibility.astype(np.float32),
            "occluded": np.logical_not(raw_visibility),
            "boundary_distance": boundary_distance,
            "source_camera": source_camera,
            "source_track": source_track,
            "initial_match_distance_m": match_distance.astype(np.float32),
            "raw_visibility": raw_visibility,
        }
    )
    output = Path(output_npz_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **cues)
    visible_boundary = boundary_distance[final_visible]
    summary: dict[str, object] = {
        "schema_version": 1,
        "config": asdict(cfg),
        "contract": {
            "mapping": "exact initial 3D point match to a unique raw camera query",
            "confidence": "released binary CoTracker visibility; raw continuous confidence was not saved",
            "boundary_distance": "inside-object Euclidean mask distance divided by maximum image dimension",
            "causality": "all cues at frame t use only raw perception outputs at frame t",
        },
        "inputs": {
            "final_data": {
                "path": str(Path(final_data_path).resolve()),
                "sha256": _sha256(final_data_path),
            },
            "raw_case_dir": str(raw_path.resolve()),
            "base_cues": (
                None
                if base_cues_path is None
                else {
                    "path": str(Path(base_cues_path).resolve()),
                    "sha256": _sha256(base_cues_path),
                }
            ),
        },
        "frame_count": frame_count,
        "track_count": final_track_count,
        "camera_track_counts": {
            str(camera): int(np.sum(source_camera == camera))
            for camera in range(len(track_paths))
        },
        "mapping": {
            "maximum_distance_m": float(np.max(match_distance)),
            "unique_raw_queries": final_track_count,
        },
        "raw_visibility_rate": float(np.mean(raw_visibility)),
        "final_visibility_rate": float(np.mean(final_visible)),
        "visible_boundary_distance": _distribution(visible_boundary),
        "output_npz": str(output.resolve()),
    }
    return summary


def write_raw_cue_summary(summary: dict[str, object], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
