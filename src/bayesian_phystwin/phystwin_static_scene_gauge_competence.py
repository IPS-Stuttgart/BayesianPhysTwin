"""Prefix-only material-identity scoring for static-scene gauge artifacts."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .phystwin_cotracker3_cues import _sha256
from .phystwin_static_scene_gauge import (
    load_static_scene_corrected_source_tracks,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class StaticSceneGaugeCompetenceConfig:
    """Frozen prefix scorer settings."""

    train_end_frame: int
    minimum_quality: float = 0.1
    maximum_cycle_error_px: float = 12.0
    late_frame_count: int = 20

    def __post_init__(self) -> None:
        _require(
            self.train_end_frame >= 2,
            "train_end_frame must be at least two",
        )
        _require(
            0.0 <= self.minimum_quality <= 1.0,
            "minimum quality must lie in [0, 1]",
        )
        _require(
            self.maximum_cycle_error_px > 0.0,
            "maximum cycle error must be positive",
        )
        _require(self.late_frame_count >= 1, "late frame count must be positive")


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _nearest_initial_identities(
    object_points_m: np.ndarray,
    manual_points_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    object_points = np.asarray(object_points_m, dtype=np.float64)
    manual_points = np.asarray(manual_points_m, dtype=np.float64)
    _require(
        object_points.ndim == 2 and object_points.shape[1] == 3,
        "initial object points must have shape (N, 3)",
    )
    _require(
        manual_points.ndim == 2 and manual_points.shape[1] == 3,
        "initial manual points must have shape (M, 3)",
    )
    _require(
        np.all(np.isfinite(object_points))
        and np.all(np.isfinite(manual_points)),
        "initial identity coordinates must be finite",
    )
    squared = np.sum(
        (manual_points[:, None] - object_points[None]) ** 2,
        axis=2,
    )
    index = np.argmin(squared, axis=1)
    return np.sqrt(squared[np.arange(len(index)), index]), index


def _lift_source_depth_tracks(
    tracks_xy: np.ndarray,
    *,
    source_camera: np.ndarray,
    raw_case_dir: Path,
    initial_world_points_m: np.ndarray,
    quality: np.ndarray,
    cycle_error_px: np.ndarray,
    cycle_valid: np.ndarray,
    boundary_distance: np.ndarray,
    cue_available: np.ndarray,
    config: StaticSceneGaugeCompetenceConfig,
) -> tuple[np.ndarray, np.ndarray]:
    tracks = np.asarray(tracks_xy, dtype=np.float64)
    camera_id = np.asarray(source_camera, dtype=np.int64)
    initial = np.asarray(initial_world_points_m, dtype=np.float64)
    end = config.train_end_frame
    _require(
        tracks.ndim == 3 and tracks.shape[2] == 2 and len(tracks) >= end,
        "source tracks have an incompatible shape",
    )
    _require(
        len(camera_id) == tracks.shape[1] == len(initial),
        "source camera or initial geometry count changed",
    )
    metadata = json.loads(
        (raw_case_dir / "metadata.json").read_text(encoding="utf-8")
    )
    intrinsics = np.asarray(metadata["intrinsics"], dtype=np.float64)
    camera_to_world = np.asarray(
        _load_pickle(raw_case_dir / "calibrate.pkl"),
        dtype=np.float64,
    )
    _require(
        intrinsics.ndim == 3 and intrinsics.shape[1:] == (3, 3),
        "camera intrinsics must have shape (C, 3, 3)",
    )
    _require(
        camera_to_world.shape == (len(intrinsics), 4, 4),
        "camera transforms must have shape (C, 4, 4)",
    )

    world = np.full((end, len(initial), 3), np.nan, dtype=np.float64)
    depth_valid = np.zeros((end, len(initial)), dtype=bool)
    for camera in range(len(intrinsics)):
        selected = np.flatnonzero(camera_id == camera)
        if len(selected) == 0:
            continue
        inverse_intrinsic = np.linalg.inv(intrinsics[camera])
        for frame in range(end):
            depth_path = (
                raw_case_dir / "depth" / str(camera) / f"{frame}.npy"
            )
            depth = np.asarray(np.load(depth_path), dtype=np.float64)
            xy = tracks[frame, selected]
            pixels = np.rint(xy).astype(np.int64)
            inside = (
                np.all(np.isfinite(xy), axis=1)
                & (pixels[:, 0] >= 0)
                & (pixels[:, 0] < depth.shape[1])
                & (pixels[:, 1] >= 0)
                & (pixels[:, 1] < depth.shape[0])
            )
            local = np.flatnonzero(inside)
            if len(local) == 0:
                continue
            z = depth[pixels[local, 1], pixels[local, 0]] / 1000.0
            positive = np.isfinite(z) & (z > 0.0)
            local = local[positive]
            z = z[positive]
            if len(local) == 0:
                continue
            homogeneous_pixels = np.column_stack(
                (xy[local], np.ones(len(local)))
            )
            camera_points = (
                homogeneous_pixels @ inverse_intrinsic.T
            ) * z[:, None]
            homogeneous_camera = np.column_stack(
                (camera_points, np.ones(len(camera_points)))
            )
            world_points = (
                homogeneous_camera @ camera_to_world[camera].T
            )
            target = selected[local]
            world[frame, target] = world_points[:, :3]
            depth_valid[frame, target] = True

    initial_valid = depth_valid[0] & np.all(np.isfinite(world[0]), axis=1)
    anchored = np.full_like(world, np.nan)
    anchored[:, initial_valid] = (
        initial[initial_valid][None]
        + world[:, initial_valid]
        - world[0, initial_valid][None]
    )
    valid = (
        depth_valid
        & initial_valid[None]
        & np.asarray(cue_available[:end], dtype=bool)
        & (np.asarray(quality[:end], dtype=float) >= config.minimum_quality)
        & np.asarray(cycle_valid[:end], dtype=bool)
        & (
            np.asarray(cycle_error_px[:end], dtype=float)
            <= config.maximum_cycle_error_px
        )
        & (np.asarray(boundary_distance[:end], dtype=float) > 0.0)
        & np.all(np.isfinite(anchored), axis=2)
    )
    anchored[~valid] = np.nan
    return anchored, valid


def _metrics(
    points_m: np.ndarray,
    manual_tracks_m: np.ndarray,
    valid: np.ndarray,
    *,
    late_frame_count: int,
) -> dict[str, float | int]:
    error = np.linalg.norm(points_m - manual_tracks_m, axis=2)
    selected = error[valid]
    _require(len(selected) > 0, "manual prefix has no common support")
    late_start = max(0, len(error) - late_frame_count)
    late = error[late_start:][valid[late_start:]]
    _require(len(late) > 0, "manual late prefix has no common support")
    return {
        "point_frame_count": int(len(selected)),
        "mean_error_mm": float(np.mean(selected) * 1000.0),
        "rmse_mm": float(np.sqrt(np.mean(selected**2)) * 1000.0),
        "late_mean_error_mm": float(np.mean(late) * 1000.0),
    }


def evaluate_phystwin_static_scene_gauge_prefix(
    cues_path: str | Path,
    gauge_path: str | Path,
    raw_case_dir: str | Path,
    final_data_path: str | Path,
    manual_tracks_path: str | Path,
    *,
    case: str,
    config: StaticSceneGaugeCompetenceConfig,
) -> dict[str, Any]:
    """Score raw and corrected automatic tracks on the allowed prefix only."""

    cues_file = Path(cues_path)
    gauge_file = Path(gauge_path)
    raw_path = Path(raw_case_dir)
    final_file = Path(final_data_path)
    manual_file = Path(manual_tracks_path)
    corrected_tracks, gauge_variance, gauge_supported, gauge_summary = (
        load_static_scene_corrected_source_tracks(cues_file, gauge_file)
    )
    with np.load(cues_file) as archive:
        raw_tracks = np.asarray(archive["source_tracks_xy"])
        source_camera = np.asarray(archive["source_camera"], dtype=np.int64)
        quality_key = (
            "source_quality_probability"
            if "source_quality_probability" in archive.files
            else "cotracker_quality_probability"
        )
        quality = np.asarray(archive[quality_key], dtype=np.float64)
        cycle_error = np.asarray(
            archive["forward_backward_error_px"],
            dtype=np.float64,
        )
        cycle_valid = np.asarray(
            archive["forward_backward_valid"],
            dtype=bool,
        )
        boundary = np.asarray(
            archive["boundary_distance"],
            dtype=np.float64,
        )
        available = np.asarray(archive["cue_available"], dtype=bool)
    final_data = _load_pickle(final_file)
    initial = np.asarray(final_data["object_points"], dtype=np.float64)[0]
    manual = np.asarray(_load_pickle(manual_file), dtype=np.float64)[
        : config.train_end_frame
    ]
    _require(
        manual.ndim == 3
        and manual.shape[2] == 3
        and len(manual) == config.train_end_frame,
        "manual tracks must have shape (T, M, 3)",
    )
    initial_distance, identity = _nearest_initial_identities(
        initial,
        manual[0],
    )

    lift_arguments = {
        "source_camera": source_camera,
        "raw_case_dir": raw_path,
        "initial_world_points_m": initial,
        "quality": quality,
        "cycle_error_px": cycle_error,
        "cycle_valid": cycle_valid,
        "boundary_distance": boundary,
        "cue_available": available,
        "config": config,
    }
    raw_points, raw_valid = _lift_source_depth_tracks(
        raw_tracks,
        **lift_arguments,
    )
    corrected_points, corrected_valid = _lift_source_depth_tracks(
        corrected_tracks,
        **lift_arguments,
    )
    manual_valid = np.all(np.isfinite(manual), axis=2)
    common = (
        manual_valid
        & raw_valid[:, identity]
        & corrected_valid[:, identity]
    )
    raw_selected = raw_points[:, identity]
    corrected_selected = corrected_points[:, identity]
    raw_metrics = _metrics(
        raw_selected,
        manual,
        common,
        late_frame_count=config.late_frame_count,
    )
    corrected_metrics = _metrics(
        corrected_selected,
        manual,
        common,
        late_frame_count=config.late_frame_count,
    )
    changes = {
        name: float(
            1.0
            - corrected_metrics[name]
            / max(float(raw_metrics[name]), 1e-12)
        )
        for name in ("mean_error_mm", "rmse_mm", "late_mean_error_mm")
    }
    return {
        "schema_version": 1,
        "artifact_kind": "PhysTwinStaticSceneGaugePrefixCompetenceV1",
        "case": case,
        "config": asdict(config),
        "raw": raw_metrics,
        "static_scene_gauge": corrected_metrics,
        "relative_improvement": changes,
        "support": {
            "manual_identity_count": int(manual.shape[1]),
            "common_point_frame_count": int(np.sum(common)),
            "common_point_frame_fraction": float(np.mean(common)),
            "gauge_supported_dense_fraction": float(
                np.mean(gauge_supported)
            ),
            "gauge_variance_px2_median": (
                float(np.median(gauge_variance[gauge_supported]))
                if np.any(gauge_supported)
                else None
            ),
            "initial_identity_match_mm": (
                initial_distance * 1000.0
            ).tolist(),
        },
        "inputs": {
            "cues_sha256": _sha256(cues_file),
            "gauge_sha256": _sha256(gauge_file),
            "final_data_sha256": _sha256(final_file),
            "manual_tracks_sha256": _sha256(manual_file),
        },
        "gauge_camera_summaries": gauge_summary["camera_summaries"],
        "information_boundary": {
            "prediction_rgb_frame_range_half_open": [
                0,
                config.train_end_frame,
            ],
            "manual_tracks_role": "prefix-only score after gauge construction",
            "future_rgb_read": False,
            "future_manual_track_read": False,
            "future_simulator_outcome_read": False,
            "claim": "opened-source observation-feeder competence only",
        },
    }


__all__ = [
    "StaticSceneGaugeCompetenceConfig",
    "evaluate_phystwin_static_scene_gauge_prefix",
]
