"""Correlation-aware metric object carriers for the Deform360 source prefix.

The carrier is an intermediate observation artifact, not a physical twin.  It
turns masked MotionCrafter point maps into a small set of metric 3-D points
while preserving camera-level correlation and the unresolved tactile-to-gripper
assignment.  No physical-state innovation is used to select masks, matches, or
prior reliability.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._portable_contracts import content_id, load_strict_json_object
from .deform360_tactile_metric_gauge import (
    SimilarityTransform,
    apply_similarity,
    covariance_intersection_equal_weight,
)

DEFORM360_METRIC_OBJECT_CARRIER_LOCK_SCHEMA = (
    "bayesian-phystwin.deform360-metric-object-carrier-lock"
)
DEFORM360_METRIC_OBJECT_CARRIER_LOCK_VERSION = 1

METRIC_OBJECT_CARRIER_POLICY = {
    "source_frame_start": 108,
    "source_frame_stop_exclusive": 150,
    "initial_mask_source_frame": 108,
    "carrier_source_frame": 149,
    "source_shape": [720, 1280],
    "target_shape": [320, 640],
    "block_size_px": 8,
    "minimum_mask_pixels_per_block": 16,
    "minimum_valid_fraction_per_block": 0.5,
    "minimum_deform_fraction_for_full_reliability": 0.5,
    "cross_view_maximum_distance_m": 0.03,
    "maximum_pairwise_percentile_90_m": 0.025,
    "required_view_count": 3,
    "required_assignment_count": 2,
    "carrier_node_count": 128,
    "local_covariance_floor_m": 0.005,
    "matching": "mutual-nearest-neighbor-under-each-assignment",
    "node_selection": "deterministic-reference-image-farthest-point",
    "cross_view_fusion": "equal-weight-covariance-intersection-plus-spread",
    "dense_correlation": "one-row-per-fixed-image-block-no-pixel-precision-sum",
    "assignment_policy": "retain-direct-and-swapped-at-equal-prior-mass",
}

METRIC_OBJECT_CARRIER_INFORMATION_BOUNDARY = {
    "calibration_camera_prefix_allowed": True,
    "calibration_provider_values_allowed_after_lock": True,
    "calibration_scores_opened": False,
    "confirmation_payloads_opened": False,
    "future_camera_frames_used": False,
    "future_tactile_values_used": False,
    "held_v8_accessed": False,
    "physical_state_residual_used_for_reliability": False,
    "target_outcomes_used": False,
}


@dataclass(frozen=True, slots=True)
class BlockPointCandidates:
    """One conservative point-map summary per fixed image block."""

    block_yx: np.ndarray
    pixel_xy: np.ndarray
    points_world_m: np.ndarray
    covariance_m2: np.ndarray
    prior_reliability: np.ndarray
    deform_fraction: np.ndarray


@dataclass(frozen=True, slots=True)
class MetricObjectCarrier:
    """Assignment-mixture metric carrier with shared camera uncertainty."""

    points_world_m: np.ndarray
    covariance_m2: np.ndarray
    assignment_mixture_covariance_m2: np.ndarray
    marginal_covariance_m2: np.ndarray
    prior_reliability: np.ndarray
    reference_pixel_xy: np.ndarray
    contributor_indices: np.ndarray
    pairwise_distance_m: np.ndarray


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _sha256(value: object, *, name: str) -> str:
    _require(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"invalid {name}",
    )
    return str(value)


def _revision(value: object, *, name: str) -> str:
    _require(
        type(value) is str
        and len(value) in {40, 64}
        and all(character in "0123456789abcdef" for character in value),
        f"invalid {name}",
    )
    return str(value)


def validate_metric_object_carrier_lock(value: Mapping[str, Any]) -> str:
    """Validate the frozen pre-mask source-only carrier lock."""

    artifact_id = _sha256(value.get("artifact_id"), name="artifact_id")
    descriptor = dict(value)
    descriptor.pop("artifact_id")
    _require(content_id(descriptor) == artifact_id, "carrier lock identity changed")
    _require(
        value.get("schema") == DEFORM360_METRIC_OBJECT_CARRIER_LOCK_SCHEMA
        and value.get("schema_version")
        == DEFORM360_METRIC_OBJECT_CARRIER_LOCK_VERSION,
        "unsupported carrier lock",
    )
    _require(
        value.get("status") == "locked-source-only-pre-mask",
        "carrier lock has the wrong status",
    )
    _require(
        value.get("policy") == METRIC_OBJECT_CARRIER_POLICY,
        "carrier policy changed",
    )
    _require(
        value.get("information_boundary")
        == METRIC_OBJECT_CARRIER_INFORMATION_BOUNDARY,
        "carrier information boundary changed",
    )
    implementation = value.get("implementation")
    source = value.get("source_case")
    parents = value.get("parents")
    cameras = value.get("cameras")
    sam2 = value.get("sam2")
    for item, name in (
        (implementation, "implementation"),
        (source, "source_case"),
        (parents, "parents"),
        (sam2, "sam2"),
    ):
        _require(isinstance(item, Mapping), f"missing {name}")
    assert isinstance(implementation, Mapping)
    assert isinstance(source, Mapping)
    assert isinstance(parents, Mapping)
    assert isinstance(sam2, Mapping)
    _revision(implementation.get("revision"), name="implementation revision")
    for name in ("runner_source_sha256", "module_source_sha256"):
        _sha256(implementation.get(name), name=name)
    _require(source.get("object_id") == "026-sock-cloth", "source object changed")
    _require(source.get("processing_episode_index") == 0, "source episode changed")
    _require(source.get("causal_frame_stop") == 150, "source cutoff changed")
    _require(
        isinstance(cameras, list)
        and len(cameras) == METRIC_OBJECT_CARRIER_POLICY["required_view_count"]
        and len(set(cameras)) == len(cameras),
        "carrier camera panel changed",
    )
    _require(
        value.get("reference_camera") == cameras[0],
        "carrier reference camera changed",
    )
    for parent_name, parent in parents.items():
        _require(isinstance(parent, Mapping), f"invalid parent {parent_name}")
        assert isinstance(parent, Mapping)
        _sha256(parent.get("sha256"), name=f"{parent_name} sha256")
        if "artifact_id" in parent:
            _sha256(parent.get("artifact_id"), name=f"{parent_name} artifact_id")
    _revision(sam2.get("repository_revision"), name="SAM2 revision")
    for name in ("checkpoint_sha256", "selector_source_sha256"):
        _sha256(sam2.get(name), name=name)
    for name in ("repository_path", "checkpoint_path", "selector_source_path"):
        _require(type(sam2.get(name)) is str, f"missing SAM2 {name}")
    providers = value.get("providers")
    _require(isinstance(providers, list) and len(providers) == len(cameras), "providers changed")
    assert isinstance(cameras, list)
    for camera, provider in zip(cameras, providers, strict=True):
        _require(isinstance(provider, Mapping), "invalid provider record")
        assert isinstance(provider, Mapping)
        _require(provider.get("camera") == camera, "provider order changed")
        for name in ("video_path", "prediction_manifest_path", "window_path"):
            _require(type(provider.get(name)) is str, f"missing provider {name}")
        for name in ("video_sha256", "prediction_manifest_sha256", "window_sha256"):
            _sha256(provider.get(name), name=name)
        _require(provider.get("window_source_frames") == [125, 150], "window changed")
    return artifact_id


def load_metric_object_carrier_lock(path: str | Path) -> Mapping[str, Any]:
    value = load_strict_json_object(path, label="metric object-carrier lock")
    validate_metric_object_carrier_lock(value)
    return value


def cover_resize_mask_nearest(
    mask: np.ndarray,
    *,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Map a source mask through MotionCrafter's cover-resize center crop."""

    source = np.asarray(mask, dtype=bool)
    _require(source.ndim == 2, "source mask must be two-dimensional")
    source_height, source_width = source.shape
    target_height, target_width = target_shape
    _require(min(source_height, source_width, target_height, target_width) > 0, "invalid shape")
    scale = max(target_height / source_height, target_width / source_width)
    resized_height = int(round(source_height * scale))
    resized_width = int(round(source_width * scale))
    crop_y = (resized_height - target_height) // 2
    crop_x = (resized_width - target_width) // 2
    resized_y = np.arange(target_height, dtype=np.float64) + crop_y
    resized_x = np.arange(target_width, dtype=np.float64) + crop_x
    source_y = np.floor((resized_y + 0.5) * source_height / resized_height).astype(int)
    source_x = np.floor((resized_x + 0.5) * source_width / resized_width).astype(int)
    source_y = np.clip(source_y, 0, source_height - 1)
    source_x = np.clip(source_x, 0, source_width - 1)
    return source[np.ix_(source_y, source_x)]


def _robust_covariance(points: np.ndarray, *, floor_m: float) -> np.ndarray:
    center = np.median(points, axis=0)
    mad = 1.4826 * np.median(np.abs(points - center), axis=0)
    variance = np.maximum(mad**2, floor_m**2)
    return np.diag(variance)


def reduce_masked_point_map(
    point_map: np.ndarray,
    valid_mask: np.ndarray,
    object_mask: np.ndarray,
    deform_mask: np.ndarray,
    *,
    transform: SimilarityTransform,
    gauge_covariance_m2: np.ndarray,
    block_size_px: int,
    minimum_mask_pixels: int,
    minimum_valid_fraction: float,
    full_reliability_deform_fraction: float,
    covariance_floor_m: float,
) -> BlockPointCandidates:
    """Summarize dense correlated pixels without precision accumulation."""

    points = np.asarray(point_map, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    selected = np.asarray(object_mask, dtype=bool)
    deform = np.asarray(deform_mask, dtype=bool)
    _require(points.ndim == 3 and points.shape[2] == 3, "invalid point map")
    _require(valid.shape == selected.shape == deform.shape == points.shape[:2], "mask shape changed")
    covariance = np.asarray(gauge_covariance_m2, dtype=np.float64)
    _require(covariance.shape == (3, 3), "gauge covariance changed shape")
    _require(block_size_px >= 1 and minimum_mask_pixels >= 1, "invalid block policy")
    height, width = valid.shape
    block_yx: list[tuple[int, int]] = []
    pixel_xy: list[tuple[float, float]] = []
    world_points: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    reliabilities: list[float] = []
    deform_fractions: list[float] = []
    for row_start in range(0, height, block_size_px):
        row_stop = min(row_start + block_size_px, height)
        for column_start in range(0, width, block_size_px):
            column_stop = min(column_start + block_size_px, width)
            block_object = selected[row_start:row_stop, column_start:column_stop]
            object_count = int(np.count_nonzero(block_object))
            if object_count < minimum_mask_pixels:
                continue
            block_valid = valid[row_start:row_stop, column_start:column_stop]
            support = block_object & block_valid
            support_count = int(np.count_nonzero(support))
            valid_fraction = support_count / object_count
            if support_count < 3 or valid_fraction < minimum_valid_fraction:
                continue
            local_points = points[row_start:row_stop, column_start:column_stop][support]
            finite = np.all(np.isfinite(local_points), axis=1)
            local_points = local_points[finite]
            if len(local_points) < 3:
                continue
            local_world = apply_similarity(transform, local_points)
            center = np.median(local_world, axis=0)
            rows, columns = np.nonzero(support)
            deform_fraction = float(
                np.mean(deform[row_start:row_stop, column_start:column_stop][support])
            )
            deform_reliability = min(
                1.0,
                deform_fraction / max(full_reliability_deform_fraction, 1e-12),
            )
            mask_fraction = object_count / block_object.size
            reliability = min(valid_fraction, mask_fraction, 0.5 + 0.5 * deform_reliability)
            block_yx.append((row_start // block_size_px, column_start // block_size_px))
            pixel_xy.append(
                (
                    float(column_start + np.median(columns)),
                    float(row_start + np.median(rows)),
                )
            )
            world_points.append(center)
            covariances.append(
                covariance
                + _robust_covariance(local_world, floor_m=covariance_floor_m)
            )
            reliabilities.append(float(np.clip(reliability, 0.0, 1.0)))
            deform_fractions.append(deform_fraction)
    _require(world_points, "masked point map has no admissible blocks")
    return BlockPointCandidates(
        block_yx=np.asarray(block_yx, dtype=np.int64),
        pixel_xy=np.asarray(pixel_xy, dtype=np.float64),
        points_world_m=np.asarray(world_points, dtype=np.float64),
        covariance_m2=np.asarray(covariances, dtype=np.float64),
        prior_reliability=np.asarray(reliabilities, dtype=np.float64),
        deform_fraction=np.asarray(deform_fractions, dtype=np.float64),
    )


def _nearest(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _require(len(source) > 0 and len(target) > 0, "empty matching point set")
    indices = np.empty(len(source), dtype=np.int64)
    distances = np.empty(len(source), dtype=np.float64)
    chunk = 256
    for start in range(0, len(source), chunk):
        stop = min(start + chunk, len(source))
        squared = np.sum(
            (source[start:stop, None, :] - target[None, :, :]) ** 2,
            axis=2,
        )
        local = np.argmin(squared, axis=1)
        indices[start:stop] = local
        distances[start:stop] = np.sqrt(squared[np.arange(stop - start), local])
    return indices, distances


def mutual_nearest_mapping(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    maximum_distance_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return reference-to-candidate mutual nearest neighbors or ``-1``."""

    forward, distances = _nearest(reference, candidate)
    reverse, _ = _nearest(candidate, reference)
    rows = np.arange(len(reference), dtype=np.int64)
    accepted = (reverse[forward] == rows) & (distances <= maximum_distance_m)
    mapping = np.full(len(reference), -1, dtype=np.int64)
    mapping[accepted] = forward[accepted]
    rejected_distances = distances.copy()
    rejected_distances[~accepted] = np.inf
    return mapping, rejected_distances


def deterministic_farthest_point_indices(
    coordinates: np.ndarray,
    count: int,
) -> np.ndarray:
    """Select spatially spread rows with deterministic tie handling."""

    values = np.asarray(coordinates, dtype=np.float64)
    _require(values.ndim == 2 and values.shape[1] >= 1, "invalid FPS coordinates")
    _require(1 <= count <= len(values), "invalid FPS count")
    order = np.lexsort(tuple(values[:, axis] for axis in reversed(range(values.shape[1]))))
    first = int(order[0])
    selected = [first]
    minimum_squared = np.sum((values - values[first]) ** 2, axis=1)
    minimum_squared[first] = -np.inf
    while len(selected) < count:
        maximum = float(np.max(minimum_squared))
        candidates = np.flatnonzero(np.isclose(minimum_squared, maximum, rtol=0.0, atol=1e-15))
        next_index = int(candidates[0])
        selected.append(next_index)
        squared = np.sum((values - values[next_index]) ** 2, axis=1)
        minimum_squared = np.minimum(minimum_squared, squared)
        minimum_squared[np.asarray(selected, dtype=int)] = -np.inf
    return np.asarray(selected, dtype=np.int64)


def fuse_unknown_correlated_points(
    points_world_m: np.ndarray,
    covariance_m2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse views without assuming independence and retain disagreement spread."""

    points = np.asarray(points_world_m, dtype=np.float64)
    covariance = np.asarray(covariance_m2, dtype=np.float64)
    _require(points.ndim == 2 and points.shape[1] == 3, "invalid point rows")
    _require(covariance.shape == (len(points), 3, 3), "invalid covariance rows")
    mean = np.mean(points, axis=0)
    centered = points - mean
    spread = centered.T @ centered / len(points)
    intersection = covariance_intersection_equal_weight(covariance)
    return mean, intersection + spread


def build_metric_object_carrier(
    candidates_by_assignment: Sequence[Mapping[str, BlockPointCandidates]],
    *,
    camera_order: Sequence[str],
    reference_camera: str,
    maximum_distance_m: float,
    node_count: int,
) -> MetricObjectCarrier:
    """Build a node-aligned carrier that passes under every assignment."""

    cameras = tuple(camera_order)
    _require(len(cameras) >= 2 and reference_camera in cameras, "invalid camera panel")
    _require(len(candidates_by_assignment) >= 1, "assignment mixture is empty")
    for assignment in candidates_by_assignment:
        _require(set(assignment) == set(cameras), "assignment camera panel changed")
    other_cameras = tuple(camera for camera in cameras if camera != reference_camera)
    mappings: list[dict[str, np.ndarray]] = []
    distances: list[dict[str, np.ndarray]] = []
    eligible: np.ndarray | None = None
    for assignment in candidates_by_assignment:
        reference = assignment[reference_camera]
        assignment_mapping: dict[str, np.ndarray] = {}
        assignment_distance: dict[str, np.ndarray] = {}
        current = np.ones(len(reference.points_world_m), dtype=bool)
        for camera in other_cameras:
            mapping, distance = mutual_nearest_mapping(
                reference.points_world_m,
                assignment[camera].points_world_m,
                maximum_distance_m=maximum_distance_m,
            )
            assignment_mapping[camera] = mapping
            assignment_distance[camera] = distance
            current &= mapping >= 0
        mappings.append(assignment_mapping)
        distances.append(assignment_distance)
        eligible = current if eligible is None else eligible & current
    assert eligible is not None
    eligible_indices = np.flatnonzero(eligible)
    _require(len(eligible_indices) >= node_count, "insufficient three-view carrier support")
    reference_pixels = candidates_by_assignment[0][reference_camera].pixel_xy
    local_selection = deterministic_farthest_point_indices(
        reference_pixels[eligible_indices], node_count
    )
    selected = eligible_indices[local_selection]

    assignment_points: list[np.ndarray] = []
    assignment_covariance: list[np.ndarray] = []
    assignment_reliability: list[np.ndarray] = []
    contributor_indices = np.empty(
        (len(candidates_by_assignment), node_count, len(cameras)), dtype=np.int64
    )
    pairwise_distance = np.zeros(
        (len(candidates_by_assignment), node_count, len(other_cameras)),
        dtype=np.float64,
    )
    for assignment_index, assignment in enumerate(candidates_by_assignment):
        points_for_assignment = []
        covariance_for_assignment = []
        reliability_for_assignment = []
        for output_index, reference_row in enumerate(selected):
            rows = []
            node_points = []
            node_covariance = []
            node_reliability = []
            for camera_index, camera in enumerate(cameras):
                if camera == reference_camera:
                    row = int(reference_row)
                else:
                    row = int(mappings[assignment_index][camera][reference_row])
                rows.append(row)
                candidate = assignment[camera]
                node_points.append(candidate.points_world_m[row])
                node_covariance.append(candidate.covariance_m2[row])
                node_reliability.append(candidate.prior_reliability[row])
                contributor_indices[assignment_index, output_index, camera_index] = row
            fused_point, fused_covariance = fuse_unknown_correlated_points(
                np.asarray(node_points), np.asarray(node_covariance)
            )
            geometric_factor = float(
                np.exp(
                    -np.mean(
                        [
                            distances[assignment_index][camera][reference_row]
                            for camera in other_cameras
                        ]
                    )
                    / maximum_distance_m
                )
            )
            points_for_assignment.append(fused_point)
            covariance_for_assignment.append(fused_covariance)
            reliability_for_assignment.append(
                min(float(value) for value in node_reliability) * geometric_factor
            )
            pairwise_distance[assignment_index, output_index] = [
                distances[assignment_index][camera][reference_row]
                for camera in other_cameras
            ]
        assignment_points.append(np.asarray(points_for_assignment))
        assignment_covariance.append(np.asarray(covariance_for_assignment))
        assignment_reliability.append(np.asarray(reliability_for_assignment))
    point_array = np.asarray(assignment_points)
    covariance_array = np.asarray(assignment_covariance)
    reliability_array = np.asarray(assignment_reliability)
    mixture_center = np.mean(point_array, axis=0)
    centered = point_array - mixture_center[None, :, :]
    mixture_covariance = np.einsum("ani,anj->nij", centered, centered) / len(point_array)
    marginal = covariance_array + mixture_covariance[None, :, :, :]
    return MetricObjectCarrier(
        points_world_m=point_array,
        covariance_m2=covariance_array,
        assignment_mixture_covariance_m2=mixture_covariance,
        marginal_covariance_m2=marginal,
        prior_reliability=reliability_array,
        reference_pixel_xy=reference_pixels[selected],
        contributor_indices=contributor_indices,
        pairwise_distance_m=pairwise_distance,
    )


__all__ = [
    "BlockPointCandidates",
    "DEFORM360_METRIC_OBJECT_CARRIER_LOCK_SCHEMA",
    "METRIC_OBJECT_CARRIER_INFORMATION_BOUNDARY",
    "METRIC_OBJECT_CARRIER_POLICY",
    "MetricObjectCarrier",
    "build_metric_object_carrier",
    "cover_resize_mask_nearest",
    "deterministic_farthest_point_indices",
    "fuse_unknown_correlated_points",
    "load_metric_object_carrier_lock",
    "mutual_nearest_mapping",
    "reduce_masked_point_map",
    "validate_metric_object_carrier_lock",
]
