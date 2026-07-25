"""Exact accelerator for the frozen Deform360 camera-subset objective."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np

from .deform360_raw_camera_observation import (
    RawCameraObservationConfig,
    _maximum_ray_angle_degrees,
)
from .phystwin_online_belief import deterministic_farthest_point_ids


_NATIVE_SOURCE = (
    Path(__file__).resolve().parent / "native" / "deform360_exact_camera_subset.cpp"
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compile_exact_camera_subset_solver(
    cache_dir: str | Path | None = None,
) -> tuple[Path, dict[str, str]]:
    """Compile the checksum-addressed helper without modifying the source tree."""

    source_sha256 = _file_sha256(_NATIVE_SOURCE)
    root = Path(cache_dir or Path.home() / ".cache" / "bayesian_phystwin").resolve()
    root.mkdir(parents=True, exist_ok=True)
    executable = root / f"deform360-exact-camera-subset-{source_sha256[:16]}"
    if not executable.is_file():
        temporary = executable.with_name(f".{executable.name}.{os.getpid()}.tmp")
        completed = subprocess.run(
            [
                "g++",
                "-O3",
                "-DNDEBUG",
                "-std=c++17",
                str(_NATIVE_SOURCE),
                "-o",
                str(temporary),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        del completed
        temporary.replace(executable)
    return executable, {
        "native_source_sha256": source_sha256,
        "native_executable_sha256": _file_sha256(executable),
    }


def _candidate_centers(
    points: np.ndarray,
    support: np.ndarray,
    camera_origins: np.ndarray,
    *,
    config: RawCameraObservationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    candidate_ids: list[int] = []
    for point_id in range(len(points)):
        views = np.flatnonzero(support[point_id])
        if len(views) < config.minimum_initial_view_count:
            continue
        if (
            _maximum_ray_angle_degrees(points[point_id], views, camera_origins)
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
    return candidates, centers


def _solver_payload(
    points: np.ndarray,
    centers: np.ndarray,
    support: np.ndarray,
    camera_origins: np.ndarray,
    selected_count: int,
) -> str:
    center_support = np.asarray(support[centers], dtype=bool)
    center_count, camera_count = center_support.shape
    if center_count > 63:
        raise ValueError("native selector supports at most 63 centers")
    masks: list[int] = []
    for camera in range(camera_count):
        mask = 0
        for center in range(center_count):
            if center_support[center, camera]:
                mask |= 1 << center
        masks.append(mask)
    support_counts = np.sum(center_support, axis=0).astype(np.int64)
    pair_angles = np.zeros((center_count, camera_count, camera_count), dtype=float)
    for center_index, point_id in enumerate(centers):
        views = np.flatnonzero(center_support[center_index])
        for left_position, first in enumerate(views):
            for second in views[left_position + 1 :]:
                angle = _maximum_ray_angle_degrees(
                    points[point_id],
                    (int(first), int(second)),
                    camera_origins,
                )
                pair_angles[center_index, first, second] = angle
                pair_angles[center_index, second, first] = angle
    lines = [f"{camera_count} {center_count} {selected_count}"]
    lines.append(" ".join(str(value) for value in masks))
    lines.append(" ".join(str(int(value)) for value in support_counts))
    lines.append(
        " ".join(format(float(value), ".17g") for value in pair_angles.ravel())
    )
    return "\n".join(lines) + "\n"


def _selection_score(
    points: np.ndarray,
    centers: np.ndarray,
    support: np.ndarray,
    camera_origins: np.ndarray,
    subset: Sequence[int],
    *,
    minimum_initial_view_count: int,
) -> tuple[int, int, int, float]:
    counts = np.sum(support[centers][:, subset], axis=1)
    angles = [
        _maximum_ray_angle_degrees(
            points[point_id],
            [index for index in subset if support[point_id, index]],
            camera_origins,
        )
        for center_index, point_id in enumerate(centers)
        if counts[center_index] >= 2
    ]
    return (
        int(np.sum(counts >= minimum_initial_view_count)),
        int(np.sum(counts >= 3)),
        int(np.sum(counts)),
        0.0 if not angles else float(np.median(angles)),
    )


def select_frame_zero_observation_plan_exact_fast(
    frame_zero_points_m: np.ndarray,
    cameras: Sequence[str],
    support: np.ndarray,
    projected_pixels: Mapping[str, np.ndarray],
    extrinsics: Mapping[str, Any],
    *,
    config: RawCameraObservationConfig,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Return the exact lexicographic optimum of the frozen selector."""

    if config.minimum_initial_view_count != 2:
        raise ValueError("exact accelerator is locked to two-view initialization")
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
    candidates, centers = _candidate_centers(
        points,
        supported,
        origins,
        config=config,
    )
    executable, _ = compile_exact_camera_subset_solver(cache_dir)
    completed = subprocess.run(
        [str(executable)],
        input=_solver_payload(
            points,
            centers,
            supported,
            origins,
            config.selected_camera_count,
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    lines = completed.stdout.strip().splitlines()
    if len(lines) != 2:
        raise RuntimeError("exact camera solver returned malformed output")
    subset = tuple(int(value) for value in lines[0].split())
    if (
        len(subset) != config.selected_camera_count
        or tuple(sorted(set(subset))) != subset
        or subset[-1] >= len(camera_names)
    ):
        raise RuntimeError("exact camera solver returned an invalid subset")
    native_score = lines[1].split()
    score = _selection_score(
        points,
        centers,
        supported,
        origins,
        subset,
        minimum_initial_view_count=config.minimum_initial_view_count,
    )
    if (
        len(native_score) != 4
        or tuple(int(value) for value in native_score[:3]) != score[:3]
        or not np.isclose(float(native_score[3]), score[3], rtol=0.0, atol=1e-12)
    ):
        raise RuntimeError("native and Python camera scores disagree")
    selected_cameras = tuple(camera_names[index] for index in subset)
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
        "selected_camera_indices": np.asarray(subset, dtype=np.int64),
        "selection_score": score,
        "query_ids": query_ids,
        "query_pixels": query_pixels,
        "support": supported,
        "camera_names": camera_names,
    }


__all__ = [
    "compile_exact_camera_subset_solver",
    "select_frame_zero_observation_plan_exact_fast",
]
