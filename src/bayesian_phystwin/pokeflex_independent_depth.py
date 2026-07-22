"""Independent metric-depth anchors for PokeFlex state updates.

The published PokeFlex Kinect checkpoint and the existing Bayesian registration
path both consume the fixed Kinect pair.  The eye-in-hand RealSense pair is a
separate sensor family.  This module keeps that family explicit, preserves its
metric covariance, and combines its two views conservatively when their
cross-correlation is unknown.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


POKEFLEX_INDEPENDENT_DEPTH_SCHEMA_VERSION = 1


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _points(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    _require(result.ndim == 2 and result.shape[1] == 3, f"{name} must be Nx3")
    _require(len(result) > 0, f"{name} is empty")
    _require(np.all(np.isfinite(result)), f"{name} contains non-finite values")
    return result


@dataclass(frozen=True)
class PokeFlexIndependentDepthAnchor:
    """A causal, outcome-independent metric point anchor."""

    take_id: str
    frame_id: int
    causal_cutoff_frame: int
    points_m: np.ndarray
    variance_m2: np.ndarray
    sensor_index: np.ndarray
    sensor_names: tuple[str, ...]
    calibration_sha256: tuple[str, ...]
    source_kind: str = "realsense_d405_depth"
    outcome_independent: bool = True
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        _require(bool(self.take_id), "anchor take id is empty")
        _require(self.frame_id >= 1, "anchor frame id must be positive")
        _require(
            self.frame_id <= self.causal_cutoff_frame,
            "anchor frame exceeds its causal cutoff",
        )
        _require(
            self.source_kind == "realsense_d405_depth",
            "anchor source family changed",
        )
        _require(self.outcome_independent, "outcome-derived anchors are forbidden")
        points = _points(self.points_m, "anchor points").copy()
        variance = np.asarray(self.variance_m2, dtype=np.float64).copy()
        sensors = np.asarray(self.sensor_index, dtype=np.int64).copy()
        _require(variance.shape == (len(points),), "anchor variance shape changed")
        _require(sensors.shape == (len(points),), "anchor sensor index shape changed")
        _require(
            np.all(np.isfinite(variance)) and np.all(variance > 0.0),
            "anchor variance must be positive metric variance",
        )
        _require(bool(self.sensor_names), "anchor sensor inventory is empty")
        _require(
            len(self.sensor_names) == len(self.calibration_sha256),
            "anchor calibration inventory changed",
        )
        _require(
            np.all((sensors >= 0) & (sensors < len(self.sensor_names))),
            "anchor sensor index is invalid",
        )
        _require(
            set(map(int, np.unique(sensors))) == set(range(len(self.sensor_names))),
            "every declared anchor sensor must contribute points",
        )
        for name in self.sensor_names:
            _require(bool(name), "anchor sensor name is empty")
        for digest in self.calibration_sha256:
            _require(
                len(digest) == 64 and all(char in "0123456789abcdef" for char in digest),
                "anchor calibration checksum is invalid",
            )
        points.setflags(write=False)
        variance.setflags(write=False)
        sensors.setflags(write=False)
        object.__setattr__(self, "points_m", points)
        object.__setattr__(self, "variance_m2", variance)
        object.__setattr__(self, "sensor_index", sensors)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def metadata_dict(self) -> dict[str, object]:
        """Return JSON-safe metadata without duplicating point arrays."""

        return {
            "schema_version": POKEFLEX_INDEPENDENT_DEPTH_SCHEMA_VERSION,
            "artifact_kind": "PokeFlexIndependentDepthAnchor",
            "take_id": self.take_id,
            "frame_id": self.frame_id,
            "causal_cutoff_frame": self.causal_cutoff_frame,
            "source_kind": self.source_kind,
            "outcome_independent": self.outcome_independent,
            "sensor_names": list(self.sensor_names),
            "calibration_sha256": list(self.calibration_sha256),
            "point_count": len(self.points_m),
            "variance_unit": "m^2",
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class PokeFlexDepthCalibration:
    """Translation-only source calibration against the allowed template mesh."""

    translation_m: np.ndarray
    inlier_count: int
    median_residual_m: float
    p90_residual_m: float

    def __post_init__(self) -> None:
        translation = np.asarray(self.translation_m, dtype=np.float64).copy()
        _require(translation.shape == (3,), "calibration translation must be 3D")
        _require(np.all(np.isfinite(translation)), "calibration is non-finite")
        _require(self.inlier_count >= 1, "calibration has no inliers")
        _require(
            np.isfinite(self.median_residual_m) and self.median_residual_m >= 0.0,
            "calibration median residual is invalid",
        )
        _require(
            np.isfinite(self.p90_residual_m) and self.p90_residual_m >= 0.0,
            "calibration p90 residual is invalid",
        )
        translation.setflags(write=False)
        object.__setattr__(self, "translation_m", translation)


@dataclass(frozen=True)
class PokeFlexAnchorGuardResult:
    """Baseline-relative decision under unknown cross-sensor correlation."""

    accepted: bool
    reason: str
    selected_vertices_m: np.ndarray
    baseline_scores_mm: np.ndarray
    candidate_scores_mm: np.ndarray
    per_sensor_regret_mm: np.ndarray
    covariance_intersection_upper_regret_mm: float

    def __post_init__(self) -> None:
        selected = _points(self.selected_vertices_m, "selected vertices").copy()
        baseline = np.asarray(self.baseline_scores_mm, dtype=np.float64).copy()
        candidate = np.asarray(self.candidate_scores_mm, dtype=np.float64).copy()
        regret = np.asarray(self.per_sensor_regret_mm, dtype=np.float64).copy()
        _require(
            baseline.ndim == 1 and baseline.shape == candidate.shape == regret.shape,
            "anchor guard score shape changed",
        )
        _require(len(baseline) >= 1, "anchor guard has no sensor scores")
        _require(
            np.all(np.isfinite(baseline))
            and np.all(np.isfinite(candidate))
            and np.all(np.isfinite(regret)),
            "anchor guard scores are non-finite",
        )
        _require(
            np.isfinite(self.covariance_intersection_upper_regret_mm),
            "anchor guard aggregate is non-finite",
        )
        for value in (selected, baseline, candidate, regret):
            value.setflags(write=False)
        object.__setattr__(self, "selected_vertices_m", selected)
        object.__setattr__(self, "baseline_scores_mm", baseline)
        object.__setattr__(self, "candidate_scores_mm", candidate)
        object.__setattr__(self, "per_sensor_regret_mm", regret)


def realsense_depth_to_world_points(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    world_to_camera: np.ndarray,
    *,
    depth_scale: float = 10000.0,
    minimum_depth_m: float = 0.05,
    maximum_depth_m: float = 0.60,
    invalid_depth_value: int = 65535,
    stride: int = 1,
) -> np.ndarray:
    """Back-project one D405 frame using the released PokeFlex convention."""

    image = np.asarray(depth)
    camera = np.asarray(intrinsics, dtype=np.float64)
    extrinsic = np.asarray(world_to_camera, dtype=np.float64)
    _require(image.ndim == 2, "depth image must be two-dimensional")
    _require(camera.shape == (3, 3), "depth intrinsics must be 3x3")
    _require(extrinsic.shape == (4, 4), "depth extrinsics must be 4x4")
    _require(depth_scale > 0.0, "depth scale must be positive")
    _require(0.0 < minimum_depth_m < maximum_depth_m, "depth range is invalid")
    _require(stride >= 1, "depth stride must be positive")

    rows, columns = np.indices(image.shape)
    values = image.astype(np.float64) / depth_scale
    valid = (
        (image != invalid_depth_value)
        & (values >= minimum_depth_m)
        & (values <= maximum_depth_m)
    )
    if stride > 1:
        valid &= (rows % stride == 0) & (columns % stride == 0)
    z = values[valid]
    _require(len(z) > 0, "RealSense frame has no valid metric depth")
    x = (columns[valid] - camera[0, 2]) * z / camera[0, 0]
    y = (rows[valid] - camera[1, 2]) * z / camera[1, 1]
    homogeneous = np.column_stack((x, y, z, np.ones_like(z)))
    camera_to_world = np.linalg.inv(extrinsic)
    return (camera_to_world @ homogeneous.T).T[:, :3]


def crop_points_to_geometry(
    points_m: np.ndarray,
    geometry_vertices_m: np.ndarray,
    *,
    padding_m: float = 0.05,
) -> np.ndarray:
    """Crop a raw anchor cloud without consulting a future outcome."""

    points = _points(points_m, "anchor cloud")
    geometry = _points(geometry_vertices_m, "allowed geometry")
    _require(padding_m > 0.0, "anchor crop padding must be positive")
    lower = geometry.min(axis=0) - padding_m
    upper = geometry.max(axis=0) + padding_m
    selected = points[np.all((points >= lower) & (points <= upper), axis=1)]
    _require(len(selected) > 0, "anchor crop is empty")
    return selected


def select_points_near_geometry(
    points_m: np.ndarray,
    geometry_vertices_m: np.ndarray,
    *,
    maximum_distance_m: float,
) -> np.ndarray:
    """Keep source points supported by allowed static geometry.

    This is an association/support operation, not a reliability score: it uses
    neither the candidate state innovation nor a future outcome.
    """

    from scipy.spatial import cKDTree

    points = _points(points_m, "anchor cloud")
    geometry = _points(geometry_vertices_m, "allowed geometry")
    _require(maximum_distance_m > 0.0, "geometry support radius must be positive")
    distance = cKDTree(geometry).query(points, k=1)[0]
    selected = points[distance <= maximum_distance_m]
    _require(len(selected) > 0, "geometry-supported anchor cloud is empty")
    return selected


def calibrate_depth_translation(
    points_m: np.ndarray,
    template_vertices_m: np.ndarray,
    *,
    maximum_association_m: float = 0.04,
    trim_quantile: float = 0.75,
    iterations: int = 6,
    minimum_inliers: int = 32,
) -> PokeFlexDepthCalibration:
    """Estimate a robust source-only world translation from template geometry."""

    from scipy.spatial import cKDTree

    points = _points(points_m, "calibration points").copy()
    template = _points(template_vertices_m, "template vertices")
    _require(maximum_association_m > 0.0, "association radius must be positive")
    _require(0.5 <= trim_quantile < 1.0, "trim quantile is invalid")
    _require(iterations >= 1, "calibration iterations must be positive")
    _require(minimum_inliers >= 3, "calibration support is too small")
    points = crop_points_to_geometry(
        points,
        template,
        padding_m=max(0.05, maximum_association_m),
    )
    tree = cKDTree(template)
    translation = np.zeros(3, dtype=np.float64)
    for _ in range(iterations):
        shifted = points + translation
        distance, index = tree.query(shifted, k=1)
        cutoff = min(maximum_association_m, float(np.quantile(distance, trim_quantile)))
        inlier = distance <= cutoff
        _require(int(np.sum(inlier)) >= minimum_inliers, "calibration support collapsed")
        residual = template[index[inlier]] - shifted[inlier]
        step = np.median(residual, axis=0)
        translation += step
        if float(np.linalg.norm(step)) <= 1e-7:
            break
    shifted = points + translation
    distance = tree.query(shifted, k=1)[0]
    cutoff = min(maximum_association_m, float(np.quantile(distance, trim_quantile)))
    inlier_distance = distance[distance <= cutoff]
    _require(len(inlier_distance) >= minimum_inliers, "calibration support collapsed")
    return PokeFlexDepthCalibration(
        translation_m=translation,
        inlier_count=len(inlier_distance),
        median_residual_m=float(np.median(inlier_distance)),
        p90_residual_m=float(np.quantile(inlier_distance, 0.9)),
    )


def _cluster_sensor_points(
    points_m: np.ndarray,
    variance_m2: float,
    *,
    voxel_size_m: float,
    maximum_cluster_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    points = _points(points_m, "sensor points")
    _require(variance_m2 > 0.0, "sensor variance must be positive")
    _require(voxel_size_m > 0.0, "anchor voxel size must be positive")
    _require(maximum_cluster_count >= 1, "anchor cluster cap must be positive")
    keys = np.floor(points / voxel_size_m).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    ordered_keys = keys[order]
    ordered_points = points[order]
    starts = np.r_[
        0,
        1 + np.flatnonzero(np.any(np.diff(ordered_keys, axis=0), axis=1)),
    ]
    counts = np.diff(np.r_[starts, len(ordered_points)])
    centroids = np.add.reduceat(ordered_points, starts, axis=0) / counts[:, None]
    if len(centroids) > maximum_cluster_count:
        selected = np.linspace(
            0, len(centroids) - 1, maximum_cluster_count, dtype=np.int64
        )
        centroids = centroids[selected]
    variance = np.full(len(centroids), variance_m2, dtype=np.float64)
    return centroids, variance


def build_independent_depth_anchor(
    *,
    take_id: str,
    frame_id: int,
    causal_cutoff_frame: int,
    sensor_points_m: Sequence[np.ndarray],
    sensor_names: Sequence[str],
    calibration_sha256: Sequence[str],
    sensor_variance_m2: Sequence[float] | float,
    voxel_size_m: float = 0.004,
    maximum_clusters_per_sensor: int = 256,
    metadata: Mapping[str, object] | None = None,
) -> PokeFlexIndependentDepthAnchor:
    """Build a correlation-limited anchor with fixed information per sensor."""

    _require(len(sensor_points_m) >= 1, "anchor has no sensor point clouds")
    _require(
        len(sensor_points_m) == len(sensor_names) == len(calibration_sha256),
        "anchor sensor inventory changed",
    )
    if np.isscalar(sensor_variance_m2):
        variances = [float(sensor_variance_m2)] * len(sensor_points_m)
    else:
        variances = list(map(float, sensor_variance_m2))
    _require(len(variances) == len(sensor_points_m), "anchor variance count changed")
    clustered_points = []
    clustered_variance = []
    clustered_sensor = []
    for index, (points, variance) in enumerate(zip(sensor_points_m, variances)):
        centroids, per_point_variance = _cluster_sensor_points(
            points,
            variance,
            voxel_size_m=voxel_size_m,
            maximum_cluster_count=maximum_clusters_per_sensor,
        )
        clustered_points.append(centroids)
        clustered_variance.append(per_point_variance)
        clustered_sensor.append(np.full(len(centroids), index, dtype=np.int64))
    return PokeFlexIndependentDepthAnchor(
        take_id=take_id,
        frame_id=frame_id,
        causal_cutoff_frame=causal_cutoff_frame,
        points_m=np.vstack(clustered_points),
        variance_m2=np.concatenate(clustered_variance),
        sensor_index=np.concatenate(clustered_sensor),
        sensor_names=tuple(sensor_names),
        calibration_sha256=tuple(calibration_sha256),
        metadata=metadata,
    )


def save_independent_depth_anchor(
    anchor: PokeFlexIndependentDepthAnchor, path: str | Path
) -> Path:
    """Write one typed NPZ artifact; metadata declares the causal boundary."""

    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        points_m=anchor.points_m,
        variance_m2=anchor.variance_m2,
        sensor_index=anchor.sensor_index,
        metadata_json=np.asarray(
            json.dumps(anchor.metadata_dict(), sort_keys=True, allow_nan=False)
        ),
    )
    return destination


def load_independent_depth_anchor(
    path: str | Path,
) -> PokeFlexIndependentDepthAnchor:
    """Load and validate one typed independent-depth artifact."""

    source = Path(path).resolve()
    with np.load(source, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        _require(
            metadata.get("schema_version")
            == POKEFLEX_INDEPENDENT_DEPTH_SCHEMA_VERSION,
            "unsupported independent-depth schema",
        )
        _require(
            metadata.get("artifact_kind") == "PokeFlexIndependentDepthAnchor",
            "unexpected independent-depth artifact kind",
        )
        anchor = PokeFlexIndependentDepthAnchor(
            take_id=str(metadata["take_id"]),
            frame_id=int(metadata["frame_id"]),
            causal_cutoff_frame=int(metadata["causal_cutoff_frame"]),
            points_m=payload["points_m"],
            variance_m2=payload["variance_m2"],
            sensor_index=payload["sensor_index"],
            sensor_names=tuple(map(str, metadata["sensor_names"])),
            calibration_sha256=tuple(map(str, metadata["calibration_sha256"])),
            source_kind=str(metadata["source_kind"]),
            outcome_independent=bool(metadata["outcome_independent"]),
            metadata=metadata.get("metadata", {}),
        )
    _require(metadata.get("point_count") == len(anchor.points_m), "point count changed")
    _require(metadata.get("variance_unit") == "m^2", "variance unit changed")
    return anchor


def anchor_fit_scores_mm(
    vertices_m: np.ndarray,
    anchor: PokeFlexIndependentDepthAnchor,
    *,
    trim_quantile: float = 0.9,
) -> np.ndarray:
    """Return equal-weight robust scores for each declared sensor group."""

    from scipy.spatial import cKDTree

    vertices = _points(vertices_m, "candidate vertices")
    _require(0.5 <= trim_quantile < 1.0, "anchor trim quantile is invalid")
    tree = cKDTree(vertices)
    scores = []
    for sensor in range(len(anchor.sensor_names)):
        points = anchor.points_m[anchor.sensor_index == sensor]
        variance = anchor.variance_m2[anchor.sensor_index == sensor]
        distance = np.asarray(tree.query(points, k=1)[0], dtype=np.float64)
        cutoff = float(np.quantile(distance, trim_quantile))
        keep = distance <= cutoff
        weights = 1.0 / variance[keep]
        weights /= np.sum(weights)
        scores.append(float(1000.0 * np.sum(weights * distance[keep])))
    return np.asarray(scores, dtype=np.float64)


def apply_independent_depth_guard(
    baseline_vertices_m: np.ndarray,
    candidate_vertices_m: np.ndarray,
    anchor: PokeFlexIndependentDepthAnchor,
    *,
    minimum_improvement_mm: float = 0.0,
    trim_quantile: float = 0.9,
) -> PokeFlexAnchorGuardResult:
    """Select a candidate only if every correlated-sensor extreme supports it."""

    baseline = _points(baseline_vertices_m, "baseline vertices")
    candidate = _points(candidate_vertices_m, "candidate vertices")
    _require(baseline.shape == candidate.shape, "candidate topology changed")
    _require(minimum_improvement_mm >= 0.0, "minimum improvement is negative")
    baseline_scores = anchor_fit_scores_mm(
        baseline, anchor, trim_quantile=trim_quantile
    )
    candidate_scores = anchor_fit_scores_mm(
        candidate, anchor, trim_quantile=trim_quantile
    )
    regret = candidate_scores - baseline_scores
    upper = float(np.max(regret))
    accepted = upper < -minimum_improvement_mm
    selected = candidate.copy() if accepted else baseline.copy()
    if not accepted:
        _require(np.array_equal(selected, baseline), "anchor fallback changed baseline")
    return PokeFlexAnchorGuardResult(
        accepted=accepted,
        reason="independent-depth-supported" if accepted else "exact-baseline-fallback",
        selected_vertices_m=selected,
        baseline_scores_mm=baseline_scores,
        candidate_scores_mm=candidate_scores,
        per_sensor_regret_mm=regret,
        covariance_intersection_upper_regret_mm=upper,
    )
