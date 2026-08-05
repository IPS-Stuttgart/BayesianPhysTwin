"""Source-only tactile metric-gauge feasibility helpers.

Known gripper poses place active tactile taxels in the Deform360 world frame.
Those points can constrain the otherwise free metric gauge of a decoded visual
point map, but only when the contact is visible from several geometrically
distinct cameras.  This module keeps the released tactile-group assignment as
an explicit mixture and validates gauge fits by holding out complete frames so
that neighboring taxels cannot manufacture support through duplication.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ContactCameraCandidate:
    camera: str
    minimum_assignment_coverage: float
    minimum_margin_px: float
    view_direction: np.ndarray


@dataclass(frozen=True, slots=True)
class SimilarityTransform:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray


@dataclass(frozen=True, slots=True)
class HeldFrameGaugeQuality:
    admitted: bool
    reason_codes: tuple[str, ...]
    residual_vectors_m: np.ndarray
    covariance_m2: np.ndarray
    median_error_m: float
    percentile_90_error_m: float
    maximum_error_m: float
    held_frame_count: int


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def cover_resize_source_to_target(
    source_xy: np.ndarray,
    *,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Apply MotionCrafter's cover-resize and center-crop pixel mapping."""

    coordinates = np.asarray(source_xy, dtype=np.float64)
    _require(coordinates.shape[-1] == 2, "pixel coordinates must end in (x, y)")
    source_height, source_width = source_shape
    target_height, target_width = target_shape
    _require(min(source_height, source_width, target_height, target_width) > 0, "invalid shape")
    scale = max(target_height / source_height, target_width / source_width)
    resized_height = int(round(source_height * scale))
    resized_width = int(round(source_width * scale))
    crop_row = (resized_height - target_height) // 2
    crop_column = (resized_width - target_width) // 2
    result = np.empty_like(coordinates)
    result[..., 0] = (
        (coordinates[..., 0] + 0.5) * resized_width / source_width
        - 0.5
        - crop_column
    )
    result[..., 1] = (
        (coordinates[..., 1] + 0.5) * resized_height / source_height
        - 0.5
        - crop_row
    )
    return result


def contact_camera_candidates(
    world_points_hypotheses_m: np.ndarray,
    *,
    intrinsics_by_camera: Mapping[str, np.ndarray],
    world_from_camera_by_camera: Mapping[str, np.ndarray],
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> tuple[ContactCameraCandidate, ...]:
    """Measure assignment-robust contact visibility without image outcomes."""

    points = np.asarray(world_points_hypotheses_m, dtype=np.float64)
    _require(
        points.ndim == 3 and points.shape[1:] == (2, 3) and len(points) > 0,
        "world points must have shape (N, 2, 3)",
    )
    _require(np.all(np.isfinite(points)), "world points contain non-finite values")
    _require(
        set(intrinsics_by_camera) == set(world_from_camera_by_camera),
        "camera calibration sets differ",
    )
    target_height, target_width = target_shape
    contact_center = np.mean(points, axis=(0, 1))
    candidates: list[ContactCameraCandidate] = []
    for camera in sorted(intrinsics_by_camera):
        intrinsics = np.asarray(intrinsics_by_camera[camera], dtype=np.float64)
        world_from_camera = np.asarray(
            world_from_camera_by_camera[camera], dtype=np.float64
        )
        _require(intrinsics.shape == (3, 3), f"invalid intrinsics for {camera}")
        _require(world_from_camera.shape == (4, 4), f"invalid extrinsics for {camera}")
        _require(
            np.all(np.isfinite(intrinsics)) and np.all(np.isfinite(world_from_camera)),
            f"non-finite calibration for {camera}",
        )
        camera_from_world = np.linalg.inv(world_from_camera)
        coverages: list[float] = []
        visible_margins: list[np.ndarray] = []
        for hypothesis in range(points.shape[1]):
            homogeneous = np.concatenate(
                (points[:, hypothesis], np.ones((len(points), 1))), axis=1
            )
            camera_points = (camera_from_world @ homogeneous.T).T[:, :3]
            depth = camera_points[:, 2]
            source_xy = np.empty((len(points), 2), dtype=np.float64)
            source_xy[:, 0] = (
                intrinsics[0, 0] * camera_points[:, 0] / depth + intrinsics[0, 2]
            )
            source_xy[:, 1] = (
                intrinsics[1, 1] * camera_points[:, 1] / depth + intrinsics[1, 2]
            )
            target_xy = cover_resize_source_to_target(
                source_xy,
                source_shape=source_shape,
                target_shape=target_shape,
            )
            margins = np.min(
                np.column_stack(
                    (
                        target_xy[:, 0],
                        target_width - 1 - target_xy[:, 0],
                        target_xy[:, 1],
                        target_height - 1 - target_xy[:, 1],
                    )
                ),
                axis=1,
            )
            visible = (depth > 0.0) & np.isfinite(margins) & (margins >= 0.0)
            coverages.append(float(np.mean(visible)))
            if np.any(visible):
                visible_margins.append(margins[visible])
        minimum_margin = (
            float(min(np.min(values) for values in visible_margins))
            if len(visible_margins) == points.shape[1]
            else float("-inf")
        )
        direction = contact_center - world_from_camera[:3, 3]
        norm = float(np.linalg.norm(direction))
        _require(norm > 0.0, f"camera {camera} is at the contact center")
        direction = direction / norm
        direction.setflags(write=False)
        candidates.append(
            ContactCameraCandidate(
                camera=camera,
                minimum_assignment_coverage=float(min(coverages)),
                minimum_margin_px=minimum_margin,
                view_direction=direction,
            )
        )
    return tuple(candidates)


def select_contact_camera_panel(
    candidates: Sequence[ContactCameraCandidate],
    *,
    panel_size: int,
    minimum_coverage: float,
    minimum_margin_px: float,
    minimum_angular_separation_deg: float,
) -> tuple[ContactCameraCandidate, ...]:
    """Select a deterministic visibility-first, angularly diverse panel."""

    _require(type(panel_size) is int and panel_size > 0, "invalid panel size")
    _require(0.0 <= minimum_coverage <= 1.0, "invalid minimum coverage")
    _require(minimum_margin_px >= 0.0, "invalid minimum margin")
    _require(
        0.0 <= minimum_angular_separation_deg <= 180.0,
        "invalid angular separation",
    )
    eligible = [
        candidate
        for candidate in candidates
        if candidate.minimum_assignment_coverage >= minimum_coverage
        and candidate.minimum_margin_px >= minimum_margin_px
    ]
    if not eligible:
        return ()
    selected = [
        sorted(eligible, key=lambda item: (-item.minimum_margin_px, item.camera))[0]
    ]
    remaining = [item for item in eligible if item.camera != selected[0].camera]
    while remaining and len(selected) < panel_size:
        scored: list[tuple[float, ContactCameraCandidate]] = []
        for candidate in remaining:
            angles = [
                float(
                    np.degrees(
                        np.arccos(
                            np.clip(
                                np.dot(candidate.view_direction, chosen.view_direction),
                                -1.0,
                                1.0,
                            )
                        )
                    )
                )
                for chosen in selected
            ]
            scored.append((min(angles), candidate))
        angle, choice = sorted(
            scored,
            key=lambda item: (-item[0], -item[1].minimum_margin_px, item[1].camera),
        )[0]
        if angle < minimum_angular_separation_deg:
            break
        selected.append(choice)
        remaining = [item for item in remaining if item.camera != choice.camera]
    return tuple(selected)


def _weighted_similarity(
    source_points: np.ndarray,
    target_points: np.ndarray,
    weights: np.ndarray,
) -> SimilarityTransform:
    total = float(np.sum(weights))
    _require(total > 0.0, "similarity weights have zero mass")
    normalized = weights / total
    source_mean = np.sum(normalized[:, None] * source_points, axis=0)
    target_mean = np.sum(normalized[:, None] * target_points, axis=0)
    source_centered = source_points - source_mean
    target_centered = target_points - target_mean
    source_singular = np.linalg.svd(source_centered, compute_uv=False)
    _require(
        len(source_singular) >= 2
        and source_singular[0] > 0.0
        and source_singular[1] / source_singular[0] >= 1e-4,
        "similarity source geometry is degenerate",
    )
    covariance = (source_centered * normalized[:, None]).T @ target_centered
    left, singular, right_transpose = np.linalg.svd(covariance)
    orientation = np.eye(3)
    if np.linalg.det(right_transpose.T @ left.T) < 0.0:
        orientation[-1, -1] = -1.0
    rotation = right_transpose.T @ orientation @ left.T
    variance = float(np.sum(normalized * np.sum(source_centered**2, axis=1)))
    _require(variance > 0.0, "similarity source variance is zero")
    scale = float(np.sum(singular * np.diag(orientation)) / variance)
    _require(np.isfinite(scale) and scale > 0.0, "similarity scale is invalid")
    translation = target_mean - scale * (rotation @ source_mean)
    rotation.setflags(write=False)
    translation.setflags(write=False)
    return SimilarityTransform(scale, rotation, translation)


def apply_similarity(transform: SimilarityTransform, points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    _require(values.ndim == 2 and values.shape[1] == 3, "points must have shape (N, 3)")
    return (
        transform.scale * (transform.rotation @ values.T)
    ).T + transform.translation


def fit_robust_similarity(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    huber_delta_m: float,
    iterations: int = 8,
) -> SimilarityTransform:
    """Fit one Huber-reweighted Sim(3) from source to metric target points."""

    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    _require(
        source.shape == target.shape and source.ndim == 2 and source.shape[1] == 3,
        "similarity points must share shape (N, 3)",
    )
    _require(len(source) >= 3, "similarity fit needs at least three points")
    _require(
        np.all(np.isfinite(source)) and np.all(np.isfinite(target)),
        "similarity points contain non-finite values",
    )
    _require(huber_delta_m > 0.0, "Huber delta must be positive")
    _require(type(iterations) is int and iterations > 0, "iterations must be positive")
    weights = np.ones(len(source), dtype=np.float64)
    transform = _weighted_similarity(source, target, weights)
    for _ in range(iterations):
        residual = np.linalg.norm(apply_similarity(transform, source) - target, axis=1)
        weights = np.minimum(1.0, huber_delta_m / np.maximum(residual, 1e-12))
        transform = _weighted_similarity(source, target, weights)
    return transform


def held_frame_gauge_quality(
    source_points: np.ndarray,
    target_points_m: np.ndarray,
    frame_ids: np.ndarray,
    *,
    huber_delta_m: float,
    covariance_floor_m: float,
    maximum_median_error_m: float,
    maximum_percentile_90_error_m: float,
) -> HeldFrameGaugeQuality:
    """Validate a metric gauge while holding out each complete source frame."""

    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points_m, dtype=np.float64)
    frames = np.asarray(frame_ids, dtype=np.int64)
    _require(
        source.shape == target.shape
        and source.ndim == 2
        and source.shape[1] == 3
        and frames.shape == (len(source),),
        "held-frame inputs have incompatible shapes",
    )
    unique_frames = np.unique(frames)
    _require(len(unique_frames) >= 3, "held-frame validation needs at least three frames")
    residuals: list[np.ndarray] = []
    reasons: list[str] = []
    for frame in unique_frames:
        train = frames != frame
        test = ~train
        try:
            transform = fit_robust_similarity(
                source[train],
                target[train],
                huber_delta_m=huber_delta_m,
            )
        except ValueError:
            reasons.append("degenerate-held-frame-fit")
            continue
        residuals.append(apply_similarity(transform, source[test]) - target[test])
    if residuals:
        residual_vectors = np.concatenate(residuals, axis=0)
        errors = np.linalg.norm(residual_vectors, axis=1)
        median_error = float(np.median(errors))
        percentile_90 = float(np.percentile(errors, 90.0))
        maximum_error = float(np.max(errors))
        centered = residual_vectors - np.mean(residual_vectors, axis=0)
        empirical = (
            centered.T @ centered / max(1, len(centered) - 1)
            if len(centered) > 1
            else np.zeros((3, 3), dtype=np.float64)
        )
    else:
        residual_vectors = np.empty((0, 3), dtype=np.float64)
        median_error = percentile_90 = maximum_error = float("inf")
        empirical = np.zeros((3, 3), dtype=np.float64)
    if median_error > maximum_median_error_m:
        reasons.append("median-held-frame-error-too-large")
    if percentile_90 > maximum_percentile_90_error_m:
        reasons.append("tail-held-frame-error-too-large")
    covariance = empirical + covariance_floor_m**2 * np.eye(3)
    residual_vectors.setflags(write=False)
    covariance.setflags(write=False)
    return HeldFrameGaugeQuality(
        admitted=not reasons,
        reason_codes=tuple(sorted(set(reasons))),
        residual_vectors_m=residual_vectors,
        covariance_m2=covariance,
        median_error_m=median_error,
        percentile_90_error_m=percentile_90,
        maximum_error_m=maximum_error,
        held_frame_count=len(unique_frames),
    )


def covariance_intersection_equal_weight(covariances_m2: np.ndarray) -> np.ndarray:
    """Fuse unknown-correlated covariances without independence inflation."""

    covariances = np.asarray(covariances_m2, dtype=np.float64)
    _require(
        covariances.ndim == 3
        and covariances.shape[1:] == (3, 3)
        and len(covariances) > 0,
        "covariances must have shape (V, 3, 3)",
    )
    information = np.zeros((3, 3), dtype=np.float64)
    weight = 1.0 / len(covariances)
    for covariance in covariances:
        _require(
            np.allclose(covariance, covariance.T, atol=1e-12, rtol=0.0)
            and np.all(np.linalg.eigvalsh(covariance) > 0.0),
            "covariance must be positive definite",
        )
        information += weight * np.linalg.inv(covariance)
    result = np.linalg.inv(information)
    return 0.5 * (result + result.T)


__all__ = [
    "ContactCameraCandidate",
    "HeldFrameGaugeQuality",
    "SimilarityTransform",
    "apply_similarity",
    "contact_camera_candidates",
    "covariance_intersection_equal_weight",
    "cover_resize_source_to_target",
    "fit_robust_similarity",
    "held_frame_gauge_quality",
    "select_contact_camera_panel",
]
