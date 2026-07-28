"""Target-free camera and material-query planning in projected response space."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any

import numpy as np


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ProjectedObservabilityConfig:
    """Frozen target-free support requirements for projected response."""

    center_count: int = 16
    minimum_camera_count: int = 3
    maximum_camera_count: int = 8
    minimum_spatial_group_count: int = 3
    minimum_points_per_camera: int = 4
    minimum_projected_response_rms_m: float = 0.0005

    def __post_init__(self) -> None:
        _require(self.center_count >= 1, "center_count must be positive")
        _require(
            self.minimum_camera_count >= 2,
            "minimum_camera_count must be at least two",
        )
        _require(
            self.maximum_camera_count >= self.minimum_camera_count,
            "camera-count interval is invalid",
        )
        _require(
            1
            <= self.minimum_spatial_group_count
            <= self.minimum_camera_count,
            "minimum_spatial_group_count is invalid",
        )
        _require(
            self.minimum_points_per_camera >= 2,
            "minimum_points_per_camera must be at least two",
        )
        _require(
            self.center_count >= self.minimum_points_per_camera,
            "center_count is too small for per-camera support",
        )
        _require(
            np.isfinite(self.minimum_projected_response_rms_m)
            and self.minimum_projected_response_rms_m > 0.0,
            "minimum projected response must be finite and positive",
        )


@dataclass(frozen=True)
class ProjectedObservabilityPlan:
    """Immutable plan built without an RGB track or future outcome."""

    camera_names: tuple[str, ...]
    spatial_group_ids: tuple[str, ...]
    selected_camera_indices: np.ndarray
    center_ids: np.ndarray
    projected_response_rms_m: np.ndarray
    eligible_camera_point: np.ndarray
    config: ProjectedObservabilityConfig
    artifact_id: str

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.camera_names)
        groups = tuple(str(group) for group in self.spatial_group_ids)
        selected = np.asarray(
            self.selected_camera_indices,
            dtype=np.int64,
        ).copy()
        centers = np.asarray(self.center_ids, dtype=np.int64).copy()
        response = np.asarray(
            self.projected_response_rms_m,
            dtype=np.float64,
        ).copy()
        eligible = np.asarray(self.eligible_camera_point, dtype=bool).copy()
        _require(len(names) == len(set(names)), "camera names must be unique")
        _require(len(groups) == len(names), "spatial group count changed")
        _require(
            response.ndim == 2
            and response.shape[0] == len(names)
            and eligible.shape == response.shape,
            "projected observability array shape changed",
        )
        _require(
            np.all(np.isfinite(response)) and np.all(response >= 0.0),
            "projected response RMS must be finite and nonnegative",
        )
        _require(
            selected.ndim == 1
            and len(selected) >= self.config.minimum_camera_count
            and len(selected) <= self.config.maximum_camera_count
            and len(np.unique(selected)) == len(selected)
            and np.all((selected >= 0) & (selected < len(names))),
            "selected camera indices are invalid",
        )
        _require(
            centers.ndim == 1
            and 1 <= len(centers) <= self.config.center_count
            and len(np.unique(centers)) == len(centers)
            and np.all((centers >= 0) & (centers < response.shape[1])),
            "center IDs are invalid",
        )
        _require(
            len({groups[index] for index in selected})
            >= self.config.minimum_spatial_group_count,
            "selected cameras do not span enough spatial groups",
        )
        _require(
            all(
                int(np.sum(eligible[index, centers]))
                >= self.config.minimum_points_per_camera
                for index in selected
            ),
            "selected camera lost projected material support",
        )
        _require(
            str(self.artifact_id).startswith("sha256:")
            and len(str(self.artifact_id)) == len("sha256:") + 64,
            "artifact_id must be a SHA-256 identifier",
        )
        for name, value in (
            ("selected_camera_indices", selected),
            ("center_ids", centers),
            ("projected_response_rms_m", response),
            ("eligible_camera_point", eligible),
        ):
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "camera_names", names)
        object.__setattr__(self, "spatial_group_ids", groups)

    @property
    def selected_camera_names(self) -> tuple[str, ...]:
        """Selected camera names in deterministic canonical order."""

        return tuple(
            self.camera_names[index] for index in self.selected_camera_indices
        )

    def query_ids(self, camera_name: str) -> np.ndarray:
        """Return projected-observable selected identities for one camera."""

        try:
            index = self.camera_names.index(str(camera_name))
        except ValueError as exc:
            raise KeyError(camera_name) from exc
        _require(
            index in set(int(value) for value in self.selected_camera_indices),
            "camera is not selected",
        )
        result = self.center_ids[
            self.eligible_camera_point[index, self.center_ids]
        ].copy()
        result.setflags(write=False)
        return result

    def to_dict(self) -> dict[str, Any]:
        """Return a compact JSON-compatible planning record."""

        selected = tuple(int(value) for value in self.selected_camera_indices)
        return {
            "schema_version": 1,
            "artifact_id": self.artifact_id,
            "config": asdict(self.config),
            "candidate_camera_names": list(self.camera_names),
            "candidate_spatial_group_ids": list(self.spatial_group_ids),
            "selected_camera_names": [
                self.camera_names[index] for index in selected
            ],
            "selected_spatial_group_ids": [
                self.spatial_group_ids[index] for index in selected
            ],
            "center_ids": self.center_ids.tolist(),
            "query_ids_by_camera": {
                self.camera_names[index]: self.query_ids(
                    self.camera_names[index]
                ).tolist()
                for index in selected
            },
            "eligible_counts_by_camera": {
                self.camera_names[index]: int(
                    np.sum(
                        self.eligible_camera_point[
                            index,
                            self.center_ids,
                        ]
                    )
                )
                for index in selected
            },
            "projected_response_rms_m_by_camera": {
                self.camera_names[index]: {
                    "minimum": float(
                        np.min(
                            self.projected_response_rms_m[
                                index,
                                self.query_ids(self.camera_names[index]),
                            ]
                        )
                    ),
                    "median": float(
                        np.median(
                            self.projected_response_rms_m[
                                index,
                                self.query_ids(self.camera_names[index]),
                            ]
                        )
                    ),
                    "maximum": float(
                        np.max(
                            self.projected_response_rms_m[
                                index,
                                self.query_ids(self.camera_names[index]),
                            ]
                        )
                    ),
                }
                for index in selected
            },
        }


def _projected_response_rms_m(
    physical_pixels_px: np.ndarray,
    initial_depth_m: np.ndarray,
    focal_lengths_px: np.ndarray,
    frame_zero_support: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pixels = np.asarray(physical_pixels_px, dtype=np.float64)
    depth = np.asarray(initial_depth_m, dtype=np.float64)
    focal = np.asarray(focal_lengths_px, dtype=np.float64)
    support = np.asarray(frame_zero_support, dtype=bool)
    camera_count, frame_count, point_count, _ = pixels.shape
    del frame_count, point_count
    finite = np.all(np.isfinite(pixels), axis=3)
    complete = support.T & np.all(finite, axis=1)
    scale_m_per_px = depth[:, :, None] / focal[:, None, :]
    raw_response = (pixels - pixels[:, :1]) * scale_m_per_px[:, None]
    centered = np.zeros_like(raw_response)
    for camera in range(camera_count):
        rows = complete[camera]
        if not np.any(rows):
            continue
        for frame in range(1, pixels.shape[1]):
            translation = np.median(
                raw_response[camera, frame, rows],
                axis=0,
            )
            centered[camera, frame] = (
                raw_response[camera, frame] - translation
            )
    rms = np.sqrt(
        np.mean(
            np.sum(np.square(centered[:, 1:]), axis=3),
            axis=1,
        )
    )
    rms[~complete] = 0.0
    return rms, complete


def _geometry_novelty(
    positions_m: np.ndarray,
    point_id: int,
    selected: list[int],
) -> float:
    if selected:
        return float(
            np.min(
                np.linalg.norm(
                    positions_m[selected] - positions_m[point_id],
                    axis=1,
                )
            )
        )
    centroid = np.mean(positions_m, axis=0)
    return float(np.linalg.norm(positions_m[point_id] - centroid))


def _select_centers(
    positions_m: np.ndarray,
    eligible: np.ndarray,
    response_rms_m: np.ndarray,
    config: ProjectedObservabilityConfig,
) -> np.ndarray | None:
    candidate_ids = np.flatnonzero(np.any(eligible, axis=0))
    if len(candidate_ids) < config.minimum_points_per_camera:
        return None
    selected: list[int] = []
    selected_set: set[int] = set()
    deficits = np.full(
        eligible.shape[0],
        config.minimum_points_per_camera,
        dtype=np.int64,
    )
    while np.any(deficits > 0) and len(selected) < config.center_count:
        best_id: int | None = None
        best_score: tuple[int, int, float, float, int] | None = None
        for raw_point_id in candidate_ids:
            point_id = int(raw_point_id)
            if point_id in selected_set:
                continue
            supported = eligible[:, point_id]
            deficit_coverage = int(np.sum(supported & (deficits > 0)))
            shared_count = int(np.sum(supported))
            if deficit_coverage == 0:
                continue
            response_score = float(
                np.median(response_rms_m[supported, point_id])
            )
            score = (
                deficit_coverage,
                shared_count,
                response_score,
                _geometry_novelty(positions_m, point_id, selected),
                -point_id,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_id = point_id
        if best_id is None:
            return None
        selected.append(best_id)
        selected_set.add(best_id)
        deficits = np.maximum(deficits - eligible[:, best_id], 0)
    if np.any(deficits > 0):
        return None
    while (
        len(selected) < min(config.center_count, len(candidate_ids))
        and len(selected_set) < len(candidate_ids)
    ):
        best_id = None
        fill_best_score: tuple[int, float, float, float, int] | None = None
        for raw_point_id in candidate_ids:
            point_id = int(raw_point_id)
            if point_id in selected_set:
                continue
            supported = eligible[:, point_id]
            response_score = float(
                np.median(response_rms_m[supported, point_id])
            )
            fill_score = (
                int(np.sum(supported)),
                response_score,
                _geometry_novelty(positions_m, point_id, selected),
                0.0,
                -point_id,
            )
            if fill_best_score is None or fill_score > fill_best_score:
                fill_best_score = fill_score
                best_id = point_id
        if best_id is None:
            break
        selected.append(best_id)
        selected_set.add(best_id)
    return np.asarray(selected, dtype=np.int64)


def plan_projected_observability(
    positions_m: np.ndarray,
    camera_names: tuple[str, ...],
    spatial_group_ids: tuple[str, ...],
    physical_pixels_px: np.ndarray,
    initial_depth_m: np.ndarray,
    focal_lengths_px: np.ndarray,
    frame_zero_support: np.ndarray,
    *,
    config: ProjectedObservabilityConfig | None = None,
) -> ProjectedObservabilityPlan:
    """Select physically observable per-camera queries before reading RGB."""

    cfg = config or ProjectedObservabilityConfig()
    positions = np.asarray(positions_m, dtype=np.float64)
    names = tuple(str(name) for name in camera_names)
    groups = tuple(str(group) for group in spatial_group_ids)
    pixels = np.asarray(physical_pixels_px, dtype=np.float64)
    depth = np.asarray(initial_depth_m, dtype=np.float64)
    focal = np.asarray(focal_lengths_px, dtype=np.float64)
    support = np.asarray(frame_zero_support, dtype=bool)
    _require(
        positions.ndim == 2
        and positions.shape[1] == 3
        and np.all(np.isfinite(positions)),
        "positions_m must have finite shape (N, 3)",
    )
    _require(len(names) == len(set(names)), "camera names must be unique")
    _require(len(groups) == len(names), "spatial group count changed")
    _require(
        pixels.ndim == 4
        and pixels.shape[0] == len(names)
        and pixels.shape[1] >= 3
        and pixels.shape[2:] == (len(positions), 2),
        "physical pixels must have shape (C, T, N, 2)",
    )
    _require(
        depth.shape == (len(names), len(positions))
        and np.all(np.isfinite(depth))
        and np.all(depth > 0.0),
        "initial depth is invalid",
    )
    _require(
        focal.shape == (len(names), 2)
        and np.all(np.isfinite(focal))
        and np.all(focal > 0.0),
        "focal length is invalid",
    )
    _require(
        support.shape == (len(positions), len(names)),
        "frame-zero support shape changed",
    )
    order = np.argsort(np.asarray(names), kind="stable")
    names = tuple(names[int(index)] for index in order)
    groups = tuple(groups[int(index)] for index in order)
    pixels = pixels[order]
    depth = depth[order]
    focal = focal[order]
    support = support[:, order]
    response_rms, complete = _projected_response_rms_m(
        pixels,
        depth,
        focal,
        support,
    )
    eligible = complete & (
        response_rms >= cfg.minimum_projected_response_rms_m
    )
    camera_candidates = np.flatnonzero(
        np.sum(eligible, axis=1) >= cfg.minimum_points_per_camera
    )
    _require(
        len(camera_candidates) >= cfg.minimum_camera_count,
        "too few cameras contain projected physical response",
    )
    best_indices: np.ndarray | None = None
    best_centers: np.ndarray | None = None
    best_score: tuple[int, int, int, float, float] | None = None
    maximum = min(cfg.maximum_camera_count, len(camera_candidates))
    for camera_count in range(maximum, cfg.minimum_camera_count - 1, -1):
        found_at_count = False
        for subset_raw in combinations(camera_candidates.tolist(), camera_count):
            subset = np.asarray(subset_raw, dtype=np.int64)
            if (
                len({groups[int(index)] for index in subset})
                < cfg.minimum_spatial_group_count
            ):
                continue
            subset_eligible = eligible[subset]
            subset_response = response_rms[subset]
            centers = _select_centers(
                positions,
                subset_eligible,
                subset_response,
                cfg,
            )
            if centers is None:
                continue
            counts = np.sum(subset_eligible[:, centers], axis=1)
            per_node_support = np.sum(subset_eligible[:, centers], axis=0)
            supported_response = subset_response[:, centers][
                subset_eligible[:, centers]
            ]
            score = (
                int(np.min(counts)),
                int(np.sum(np.maximum(per_node_support - 1, 0))),
                int(np.sum(counts)),
                float(np.min(supported_response)),
                float(np.median(supported_response)),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_indices = subset
                best_centers = centers
            found_at_count = True
        if found_at_count:
            break
    if best_indices is None or best_centers is None:
        raise ValueError("projected observability multicover is infeasible")
    payload = {
        "config": asdict(cfg),
        "camera_names": list(names),
        "spatial_group_ids": list(groups),
        "selected_camera_indices": best_indices.tolist(),
        "center_ids": best_centers.tolist(),
        "positions_sha256": _array_sha256(positions),
        "projected_response_rms_sha256": _array_sha256(response_rms),
        "eligible_camera_point_sha256": _array_sha256(eligible),
    }
    artifact_id = "sha256:" + hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return ProjectedObservabilityPlan(
        camera_names=names,
        spatial_group_ids=groups,
        selected_camera_indices=best_indices,
        center_ids=best_centers,
        projected_response_rms_m=response_rms,
        eligible_camera_point=eligible,
        config=cfg,
        artifact_id=artifact_id,
    )


__all__ = [
    "ProjectedObservabilityConfig",
    "ProjectedObservabilityPlan",
    "plan_projected_observability",
]
