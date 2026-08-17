"""Partial factor selection for Deform360 joint-sparse geometric v4."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from .deform360_joint_sparse_geometric_common_v4 import (
    _require,
    _sha256_file,
)
from .deform360_joint_sparse_geometric_npz_v4 import (
    _load_camera_center,
    _load_metric_sparse_frames,
    _load_prediction_support_windows,
    _MetricFrame,
)


def _voxel(point: np.ndarray, size: float) -> tuple[int, int, int]:
    values = np.floor(np.asarray(point, dtype=np.float64) / size).astype(np.int64)
    return cast(tuple[int, int, int], tuple(map(int, values)))


def _cluster_id(voxel: tuple[int, int, int]) -> str:
    return hashlib.sha256(
        f"world-voxel-v1:{voxel[0]}:{voxel[1]}:{voxel[2]}".encode()
    ).hexdigest()


def _deterministic_select(
    values: Sequence[tuple[tuple[int, str, int, int], np.ndarray]],
    *,
    maximum: int,
    seed: str,
) -> list[tuple[tuple[int, str, int, int], np.ndarray]]:
    if len(values) <= maximum:
        return list(values)
    ranked = sorted(
        values,
        key=lambda item: hashlib.sha256(
            f"{seed}:{item[0][0]}:{item[0][1]}:{item[0][2]}:{item[0][3]}".encode()
        ).digest(),
    )
    return sorted(ranked[:maximum], key=lambda item: item[0])


@dataclass(frozen=True, slots=True)
class _Candidate:
    job_id: str
    camera_id: str
    window_id: str
    frame: int
    row: int
    column: int
    point_world_m: np.ndarray
    camera_center_world_m: np.ndarray
    spatial_cluster_id: str
    correlation_group_id: str
    support_digest: str


def _collect_stream_candidates(
    *,
    job_id: str,
    camera_id: str,
    causal_range: tuple[int, int],
    prediction_manifest_path: Path,
    metric_prefix_path: Path,
    metric_calibration_path: Path,
    object_id: str,
    episode_id: int,
    policy: Mapping[str, Any],
) -> tuple[list[_Candidate], int, dict[str, str], dict[str, Any]]:
    metric_frames, image_shape = _load_metric_sparse_frames(
        metric_prefix_path, causal_range=causal_range
    )
    camera_center, calibration_id = _load_camera_center(
        metric_calibration_path,
        object_id=object_id,
        episode_id=episode_id,
        camera_id=camera_id,
    )
    windows, run_spec_sha = _load_prediction_support_windows(
        prediction_manifest_path,
        causal_range=causal_range,
        image_shape=image_shape,
        expected_motioncrafter_revision=cast(str, policy["motioncrafter_revision"]),
    )
    voxel_size = cast(float, policy["world_voxel_size_m"])
    maximum = cast(int, policy["maximum_factors_per_camera_window"])
    candidates: list[_Candidate] = []
    dropped = 0
    support_artifacts: dict[str, str] = {}
    window_counts: dict[str, int] = {}
    for window in windows:
        unique: dict[tuple[int, str], tuple[tuple[int, str, int, int], np.ndarray]] = {}
        for local_index, frame_value in enumerate(window.frame_indices):
            frame = int(frame_value)
            metric = metric_frames.get(frame)
            _require(metric is not None, "prediction window leaves metric prefix")
            metric = cast(_MetricFrame, metric)
            supported = window.valid_mask[local_index, metric.rows, metric.columns]
            for row, column, point in zip(
                metric.rows[supported],
                metric.columns[supported],
                metric.points_world_m[supported],
                strict=True,
            ):
                voxel = _voxel(point, voxel_size)
                cluster = _cluster_id(voxel)
                key = (frame, cluster)
                rank_key = (frame, cluster, int(row), int(column))
                previous = unique.get(key)
                if previous is None or rank_key < previous[0]:
                    unique[key] = (rank_key, np.asarray(point, dtype=np.float64))
        ordered = sorted(unique.values(), key=lambda item: item[0])
        selected = _deterministic_select(
            ordered, maximum=maximum, seed=f"{job_id}:{window.window_id}"
        )
        dropped += len(ordered) - len(selected)
        window_counts[window.window_id] = len(selected)
        support_key = hashlib.sha256(window.window_id.encode("utf-8")).hexdigest()
        support_artifacts[f"support/{job_id}/{support_key}.mask"] = (
            window.support_digest
        )
        for (frame, cluster, row, column), point in selected:
            group = hashlib.sha256(
                f"frame-world-voxel-v1:{object_id}:{episode_id}:{frame}:{cluster}".encode()
            ).hexdigest()
            candidates.append(
                _Candidate(
                    job_id=job_id,
                    camera_id=camera_id,
                    window_id=window.window_id,
                    frame=frame,
                    row=row,
                    column=column,
                    point_world_m=point,
                    camera_center_world_m=camera_center,
                    spatial_cluster_id=cluster,
                    correlation_group_id=group,
                    support_digest=window.support_digest,
                )
            )
    source_artifacts = {
        f"prediction/{job_id}.json": _sha256_file(prediction_manifest_path),
        f"metric/{job_id}.npz": _sha256_file(metric_prefix_path),
        f"metric-calibration/{job_id}.json": _sha256_file(metric_calibration_path),
        **support_artifacts,
    }
    metadata = {
        "camera_id": camera_id,
        "calibration_id": calibration_id,
        "run_spec_sha256": run_spec_sha,
        "window_factor_counts": dict(sorted(window_counts.items())),
        "dropped_by_camera_window_cap": dropped,
    }
    return candidates, dropped, source_artifacts, metadata
