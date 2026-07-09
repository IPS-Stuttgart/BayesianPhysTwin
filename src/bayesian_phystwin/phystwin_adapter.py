"""Adapters for exporting tracked-point residuals from official PhysTwin artifacts."""

from __future__ import annotations

import csv
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


EXPORT_COLUMNS = (
    "frame",
    "track_id",
    "vertex_id",
    "source",
    "observed_x",
    "observed_y",
    "observed_z",
    "predicted_x",
    "predicted_y",
    "predicted_z",
    "variance",
    "confidence",
    "occluded",
    "boundary_distance",
    "flow_inconsistency",
    "track_valid",
    "visible",
)


@dataclass(frozen=True)
class PhysTwinExportConfig:
    """Configuration for the official PhysTwin tracked-point artifact layout."""

    variance: float = 1e-4
    start_frame: int = 1
    end_frame: int | None = None
    include_invalid: bool = False
    correspondence: str = "direct"
    source: str = "phystwin_track"
    nearest_chunk_size: int = 1024


def _to_numpy(value: Any, *, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if array.dtype == object:
        raise ValueError(f"{name} must be a dense numeric array")
    return array


def _load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _nearest_vertex_indices(
    observed_first_frame: np.ndarray,
    trajectory_first_frame: np.ndarray,
    *,
    chunk_size: int,
) -> np.ndarray:
    if chunk_size <= 0:
        raise ValueError("nearest_chunk_size must be positive")
    indexes = np.empty(observed_first_frame.shape[0], dtype=int)
    for start in range(0, observed_first_frame.shape[0], chunk_size):
        stop = min(start + chunk_size, observed_first_frame.shape[0])
        delta = (
            observed_first_frame[start:stop, None, :]
            - trajectory_first_frame[None, :, :]
        )
        indexes[start:stop] = np.argmin(np.sum(np.square(delta), axis=2), axis=1)
    return indexes


def _load_cues(
    path: str | Path | None,
    *,
    frame_count: int,
    track_count: int,
) -> dict[str, np.ndarray]:
    if path is None:
        return {}
    allowed = {"confidence", "occluded", "boundary_distance", "flow_inconsistency"}
    with np.load(path) as archive:
        cues = {name: np.asarray(archive[name]) for name in allowed if name in archive}
    for name, values in cues.items():
        if values.shape not in {(frame_count, track_count), (frame_count - 1, track_count)}:
            raise ValueError(
                f"cue {name} must have shape ({frame_count}, {track_count}) or "
                f"({frame_count - 1}, {track_count}), got {values.shape}"
            )
        if not np.all(np.isfinite(values.astype(float))):
            raise ValueError(f"cue {name} must contain finite values")
    return cues


def _cue_at(
    cues: dict[str, np.ndarray],
    name: str,
    frame: int,
    track: int,
    *,
    frame_count: int,
    default: float | bool | None,
) -> float | bool | None:
    if name not in cues:
        return default
    values = cues[name]
    cue_frame = frame if values.shape[0] == frame_count else frame - 1
    return values[cue_frame, track].item()


def export_phystwin_residuals(
    final_data_path: str | Path,
    trajectory_path: str | Path,
    output_csv_path: str | Path,
    *,
    cues_path: str | Path | None = None,
    config: PhysTwinExportConfig | None = None,
) -> dict[str, Any]:
    """Export the exact tracked-point residual pairs used by PhysTwin.

    ``final_data.pkl`` and ``inference.pkl`` are Python pickle files. Only load
    artifacts from a trusted PhysTwin run.
    """

    cfg = config or PhysTwinExportConfig()
    if cfg.variance <= 0.0:
        raise ValueError("variance must be positive")
    if cfg.correspondence not in {"direct", "nearest"}:
        raise ValueError("correspondence must be 'direct' or 'nearest'")

    final_data = _load_pickle(final_data_path)
    if not isinstance(final_data, dict):
        raise ValueError("final_data pickle must contain a dictionary")
    required = {"object_points", "object_visibilities", "object_motions_valid"}
    missing = required - set(final_data)
    if missing:
        raise ValueError(f"final_data is missing keys: {', '.join(sorted(missing))}")

    observed = _to_numpy(final_data["object_points"], name="object_points").astype(float)
    visible = _to_numpy(
        final_data["object_visibilities"],
        name="object_visibilities",
    ).astype(bool)
    motion_valid = _to_numpy(
        final_data["object_motions_valid"],
        name="object_motions_valid",
    ).astype(bool)
    trajectory = _to_numpy(_load_pickle(trajectory_path), name="trajectory").astype(float)

    if observed.ndim != 3 or observed.shape[2] != 3:
        raise ValueError(f"object_points must have shape (T, N, 3), got {observed.shape}")
    frame_count, track_count, _ = observed.shape
    if visible.shape != (frame_count, track_count):
        raise ValueError("object_visibilities must match object_points first two axes")
    if motion_valid.shape not in {
        (frame_count, track_count),
        (frame_count - 1, track_count),
    }:
        raise ValueError(
            "object_motions_valid must have shape (T, N) or (T-1, N), got "
            f"{motion_valid.shape}"
        )
    if trajectory.ndim != 3 or trajectory.shape[2] != 3:
        raise ValueError(f"trajectory must have shape (T, M, 3), got {trajectory.shape}")
    if trajectory.shape[0] < frame_count:
        raise ValueError("trajectory has fewer frames than object_points")
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(trajectory)):
        raise ValueError("object_points and trajectory must contain finite values")

    start_frame = cfg.start_frame
    end_frame = frame_count if cfg.end_frame is None else cfg.end_frame
    if not 1 <= start_frame < end_frame <= frame_count:
        raise ValueError(
            f"frame range must satisfy 1 <= start < end <= {frame_count}, "
            f"got {start_frame}:{end_frame}"
        )

    if cfg.correspondence == "direct":
        if trajectory.shape[1] < track_count:
            raise ValueError("direct correspondence requires at least N simulator vertices")
        vertex_indices = np.arange(track_count, dtype=int)
    else:
        vertex_indices = _nearest_vertex_indices(
            observed[0],
            trajectory[0],
            chunk_size=cfg.nearest_chunk_size,
        )
    initial_residual = observed[0] - trajectory[0, vertex_indices]

    cues = _load_cues(
        cues_path,
        frame_count=frame_count,
        track_count=track_count,
    )
    rows: list[dict[str, str]] = []
    skipped_invalid = 0
    for frame in range(start_frame, end_frame):
        valid_frame = frame - 1
        for track in range(track_count):
            track_valid = bool(motion_valid[valid_frame, track])
            if not cfg.include_invalid and not track_valid:
                skipped_invalid += 1
                continue
            is_visible = bool(visible[frame, track])
            confidence = _cue_at(
                cues,
                "confidence",
                frame,
                track,
                frame_count=frame_count,
                default=1.0 if is_visible else 0.0,
            )
            occluded = _cue_at(
                cues,
                "occluded",
                frame,
                track,
                frame_count=frame_count,
                default=not is_visible,
            )
            boundary = _cue_at(
                cues,
                "boundary_distance",
                frame,
                track,
                frame_count=frame_count,
                default=None,
            )
            flow = _cue_at(
                cues,
                "flow_inconsistency",
                frame,
                track,
                frame_count=frame_count,
                default=None,
            )
            vertex = int(vertex_indices[track])
            row = {
                "frame": str(frame),
                "track_id": str(track),
                "vertex_id": str(vertex),
                "source": cfg.source,
                "variance": f"{cfg.variance:.12g}",
                "confidence": f"{float(confidence):.12g}",
                "occluded": "true" if bool(occluded) else "false",
                "boundary_distance": "" if boundary is None else f"{float(boundary):.12g}",
                "flow_inconsistency": "" if flow is None else f"{float(flow):.12g}",
                "track_valid": "true" if track_valid else "false",
                "visible": "true" if is_visible else "false",
            }
            for axis, axis_name in enumerate(("x", "y", "z")):
                row[f"observed_{axis_name}"] = f"{observed[frame, track, axis]:.12g}"
                row[f"predicted_{axis_name}"] = (
                    f"{trajectory[frame, vertex, axis]:.12g}"
                )
            rows.append(row)

    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EXPORT_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    return {
        "schema_version": 1,
        "final_data_path": str(Path(final_data_path).resolve()),
        "trajectory_path": str(Path(trajectory_path).resolve()),
        "cues_path": None if cues_path is None else str(Path(cues_path).resolve()),
        "output_csv_path": str(output_path.resolve()),
        "config": asdict(cfg),
        "frame_count": frame_count,
        "track_count": track_count,
        "simulator_vertex_count": int(trajectory.shape[1]),
        "exported_measurement_count": len(rows),
        "skipped_invalid_count": skipped_invalid,
        "initial_alignment_rmse": float(
            np.sqrt(np.mean(np.sum(np.square(initial_residual), axis=1)))
        ),
        "initial_alignment_max_norm": float(
            np.max(np.linalg.norm(initial_residual, axis=1))
        ),
        "cue_fields": sorted(cues),
    }


def write_export_summary(summary: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
