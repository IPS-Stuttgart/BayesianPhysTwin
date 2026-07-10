"""Adapters for exporting tracked-point residuals from official PhysTwin artifacts."""

from __future__ import annotations

import csv
import itertools
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import binary_calibration_metrics


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


@dataclass(frozen=True)
class PhysTwinMotionCueConfig:
    """Configuration for continuous local track-motion consistency cues."""

    neighbor_count: int = 16
    minimum_valid_neighbors: int = 4
    neighbor_radius: float | None = 0.01
    neighbor_reference: str = "current"
    insufficient_neighbor_value: float = 0.10
    nearest_chunk_size: int = 1024
    evaluation_flow_scale: float | None = None


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


def _nearest_track_neighbors(
    first_frame: np.ndarray,
    *,
    neighbor_count: int,
    chunk_size: int,
    candidate_valid: np.ndarray | None = None,
) -> np.ndarray:
    track_count = first_frame.shape[0]
    if not 1 <= neighbor_count < track_count:
        raise ValueError("neighbor_count must be in [1, track_count)")
    if chunk_size <= 0:
        raise ValueError("nearest_chunk_size must be positive")
    if candidate_valid is not None:
        candidate_valid = np.asarray(candidate_valid, dtype=bool)
        if candidate_valid.shape != (track_count,):
            raise ValueError(f"candidate_valid must have shape ({track_count},)")
        if int(np.sum(candidate_valid)) <= neighbor_count:
            raise ValueError("not enough valid tracks to build the requested neighborhood")
    neighbors = np.empty((track_count, neighbor_count), dtype=int)
    for start in range(0, track_count, chunk_size):
        stop = min(start + chunk_size, track_count)
        delta = first_frame[start:stop, None, :] - first_frame[None, :, :]
        distance_sq = np.sum(np.square(delta), axis=2)
        if candidate_valid is not None:
            distance_sq[:, ~candidate_valid] = np.inf
        local_rows = np.arange(stop - start)
        distance_sq[local_rows, np.arange(start, stop)] = np.inf
        partition = np.argpartition(distance_sq, neighbor_count - 1, axis=1)
        neighbors[start:stop] = partition[:, :neighbor_count]
    return neighbors


def _radius_track_neighbors(
    points: np.ndarray,
    *,
    radius: float,
    neighbor_count: int,
    candidate_valid: np.ndarray,
) -> np.ndarray:
    """Find local neighbors with a cell list instead of an all-pairs matrix."""

    track_count = len(points)
    valid = np.asarray(candidate_valid, dtype=bool)
    if valid.shape != (track_count,):
        raise ValueError(f"candidate_valid must have shape ({track_count},)")
    cell_coordinates = np.floor(points / radius).astype(np.int64)
    cells: dict[tuple[int, int, int], list[int]] = {}
    for index in np.flatnonzero(valid):
        key = tuple(int(value) for value in cell_coordinates[index])
        cells.setdefault(key, []).append(int(index))

    offsets = tuple(itertools.product((-1, 0, 1), repeat=3))
    neighbors = np.full((track_count, neighbor_count), -1, dtype=int)
    radius_sq = radius * radius
    for track in range(track_count):
        center = cell_coordinates[track]
        candidates: list[int] = []
        for offset in offsets:
            key = (
                int(center[0] + offset[0]),
                int(center[1] + offset[1]),
                int(center[2] + offset[2]),
            )
            candidates.extend(cells.get(key, ()))
        if not candidates:
            continue
        candidate_indices = np.asarray(candidates, dtype=int)
        candidate_indices = candidate_indices[candidate_indices != track]
        if len(candidate_indices) == 0:
            continue
        delta = points[candidate_indices] - points[track]
        distance_sq = np.einsum("ij,ij->i", delta, delta)
        inside = distance_sq <= radius_sq
        candidate_indices = candidate_indices[inside]
        distance_sq = distance_sq[inside]
        if len(candidate_indices) == 0:
            continue
        order = np.lexsort((candidate_indices, distance_sq))[:neighbor_count]
        selected = candidate_indices[order]
        neighbors[track, : len(selected)] = selected
    return neighbors


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def build_phystwin_motion_cues(
    final_data_path: str | Path,
    output_npz_path: str | Path,
    *,
    config: PhysTwinMotionCueConfig | None = None,
) -> dict[str, Any]:
    """Build continuous neighbor-motion cues without simulator residuals.

    This retains the magnitude discarded by PhysTwin's binary
    ``object_motions_valid`` filtering. The cue is the distance between each
    visible track's motion and the component-wise median motion of its visible
    first-frame neighbors.
    """

    cfg = config or PhysTwinMotionCueConfig()
    if cfg.minimum_valid_neighbors < 1:
        raise ValueError("minimum_valid_neighbors must be positive")
    if cfg.minimum_valid_neighbors > cfg.neighbor_count:
        raise ValueError("minimum_valid_neighbors cannot exceed neighbor_count")
    if cfg.neighbor_radius is not None and cfg.neighbor_radius <= 0.0:
        raise ValueError("neighbor_radius must be positive when provided")
    if cfg.neighbor_reference not in {"first", "current"}:
        raise ValueError("neighbor_reference must be 'first' or 'current'")
    if cfg.insufficient_neighbor_value < 0.0:
        raise ValueError("insufficient_neighbor_value must be nonnegative")
    if cfg.evaluation_flow_scale is not None and cfg.evaluation_flow_scale <= 0.0:
        raise ValueError("evaluation_flow_scale must be positive when provided")

    final_data = _load_pickle(final_data_path)
    if not isinstance(final_data, dict):
        raise ValueError("final_data pickle must contain a dictionary")
    required = {"object_points", "object_visibilities"}
    missing = required - set(final_data)
    if missing:
        raise ValueError(f"final_data is missing keys: {', '.join(sorted(missing))}")
    points = _to_numpy(final_data["object_points"], name="object_points").astype(float)
    visible = _to_numpy(
        final_data["object_visibilities"],
        name="object_visibilities",
    ).astype(bool)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError(f"object_points must have shape (T, N, 3), got {points.shape}")
    frame_count, track_count, _ = points.shape
    if visible.shape != (frame_count, track_count):
        raise ValueError("object_visibilities must match object_points first two axes")
    if not np.all(np.isfinite(points)):
        raise ValueError("object_points must contain finite values")

    fixed_neighbors = None
    if cfg.neighbor_reference == "first":
        fixed_neighbors = _nearest_track_neighbors(
            points[0],
            neighbor_count=cfg.neighbor_count,
            chunk_size=cfg.nearest_chunk_size,
        )
    flow_inconsistency = np.full(
        (frame_count - 1, track_count),
        cfg.insufficient_neighbor_value,
        dtype=float,
    )
    valid_neighbor_count = np.zeros((frame_count - 1, track_count), dtype=np.int16)
    motion_visible = np.logical_and(visible[:-1], visible[1:])

    for frame in range(frame_count - 1):
        motion = points[frame + 1] - points[frame]
        reference_points = points[0] if fixed_neighbors is not None else points[frame]
        if fixed_neighbors is not None:
            neighbors = fixed_neighbors
        elif cfg.neighbor_radius is not None:
            neighbors = _radius_track_neighbors(
                reference_points,
                radius=cfg.neighbor_radius,
                neighbor_count=cfg.neighbor_count,
                candidate_valid=motion_visible[frame],
            )
        else:
            neighbors = _nearest_track_neighbors(
                reference_points,
                neighbor_count=cfg.neighbor_count,
                chunk_size=cfg.nearest_chunk_size,
                candidate_valid=motion_visible[frame],
            )
        for track in range(track_count):
            if not motion_visible[frame, track]:
                continue
            candidate_neighbors = neighbors[track]
            candidate_neighbors = candidate_neighbors[candidate_neighbors >= 0]
            valid_neighbors = candidate_neighbors[motion_visible[frame, candidate_neighbors]]
            if cfg.neighbor_radius is not None:
                distances = np.linalg.norm(
                    reference_points[valid_neighbors] - reference_points[track],
                    axis=1,
                )
                valid_neighbors = valid_neighbors[distances <= cfg.neighbor_radius]
            valid_neighbor_count[frame, track] = len(valid_neighbors)
            if len(valid_neighbors) < cfg.minimum_valid_neighbors:
                flow_inconsistency[frame, track] = cfg.insufficient_neighbor_value
                continue
            expected_motion = np.median(motion[valid_neighbors], axis=0)
            flow_inconsistency[frame, track] = np.linalg.norm(
                motion[track] - expected_motion
            )

    output_path = Path(output_npz_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        confidence=visible.astype(float),
        occluded=np.logical_not(visible),
        flow_inconsistency=flow_inconsistency,
        valid_neighbor_count=valid_neighbor_count,
        motion_observed=motion_visible,
    )
    valid_values = flow_inconsistency[motion_visible]
    if valid_values.size == 0:
        raise ValueError("final_data contains no visible inter-frame motions")
    summary = {
        "schema_version": 1,
        "final_data_path": str(Path(final_data_path).resolve()),
        "output_npz_path": str(output_path.resolve()),
        "config": asdict(cfg),
        "frame_count": frame_count,
        "track_count": track_count,
        "visible_motion_count": int(np.sum(motion_visible)),
        "undefined_motion_count": int(np.sum(np.logical_not(motion_visible))),
        "insufficient_neighbor_count": int(
            np.sum(motion_visible & (valid_neighbor_count < cfg.minimum_valid_neighbors))
        ),
        "flow_inconsistency": _distribution(valid_values),
    }
    if cfg.evaluation_flow_scale is not None:
        if "object_motions_valid" not in final_data:
            raise ValueError(
                "evaluation_flow_scale requires object_motions_valid in final_data"
            )
        motion_valid = _to_numpy(
            final_data["object_motions_valid"],
            name="object_motions_valid",
        ).astype(bool)
        if motion_valid.shape not in {
            (frame_count, track_count),
            (frame_count - 1, track_count),
        }:
            raise ValueError(
                "object_motions_valid must have shape (T, N) or (T-1, N)"
            )
        labels = motion_valid[: frame_count - 1][motion_visible]
        values = flow_inconsistency[motion_visible]
        prior = np.clip(
            np.exp(-values / cfg.evaluation_flow_scale),
            1e-3,
            1.0 - 1e-3,
        )
        group_summary: dict[str, dict[str, float | int]] = {}
        for name, selected in (
            ("hard_valid", labels),
            ("hard_invalid", np.logical_not(labels)),
        ):
            selected_values = values[selected]
            group_summary[name] = (
                {"count": 0}
                if len(selected_values) == 0
                else {
                    "count": int(len(selected_values)),
                    **_distribution(selected_values),
                    "mean_prior": float(np.mean(prior[selected])),
                }
            )
        summary["hard_gate_comparison"] = {
            "warning": "PhysTwin's hard gate is a heuristic label, not corruption ground truth.",
            "flow_scale": cfg.evaluation_flow_scale,
            "calibration": binary_calibration_metrics(prior, labels).as_dict(),
            "groups": group_summary,
        }
    return summary


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
