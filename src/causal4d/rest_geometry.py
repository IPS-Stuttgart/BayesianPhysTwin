"""Leakage-safe frame and rest-geometry correction for deformable twins."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bayesian_phystwin.phystwin_graph_discrepancy import (
    graph_smoothed_discrepancy_posterior,
)


FRAME_MODES = ("none", "translation", "se3")


@dataclass(frozen=True)
class RigidFrameCorrection:
    """Proper rigid transform using the row-vector convention ``x @ R + t``."""

    linear: np.ndarray
    translation: np.ndarray
    mode: str
    rotation_angle_rad: float
    fitted_point_count: int


@dataclass(frozen=True)
class GraphRestGeometryCorrection:
    """Frame/nonrigid decomposition and physically injected rest state."""

    frame: RigidFrameCorrection
    nonrigid_field: np.ndarray
    endpoint_correction: np.ndarray
    corrected_reference_vertices: np.ndarray
    corrected_rest_lengths: np.ndarray
    unclipped_rest_length_ratio: np.ndarray
    rest_length_ratio: np.ndarray
    observed_nonrigid_residual: np.ndarray
    graph_observation_weight: np.ndarray
    graph_reference_variance: float


def _points(values: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _proper_rotation(matrix: np.ndarray) -> np.ndarray:
    left, _, right = np.linalg.svd(np.asarray(matrix, dtype=float))
    rotation = left @ right
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right
    return rotation


def _column_axis_angle(rotation: np.ndarray) -> tuple[np.ndarray, float]:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0]), 0.0
    if np.pi - angle < 1e-6:
        eigenvalues, eigenvectors = np.linalg.eig(rotation)
        axis = np.real(eigenvectors[:, int(np.argmin(np.abs(eigenvalues - 1.0)))])
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-12:
            raise ValueError("could not recover the fitted rotation axis")
        return axis / norm, angle
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    ) / (2.0 * np.sin(angle))
    return axis / np.linalg.norm(axis), angle


def _column_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = np.asarray(axis, dtype=float)
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def rotation_angle(linear: np.ndarray) -> float:
    """Return the angle of a proper row-vector rotation matrix."""

    matrix = np.asarray(linear, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("linear must be a finite 3x3 matrix")
    return _column_axis_angle(matrix.T)[1]


def fit_weighted_frame_correction(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    mode: str = "se3",
    maximum_rotation_rad: float = np.deg2rad(5.0),
    maximum_translation_m: float = 0.02,
) -> RigidFrameCorrection:
    """Fit a bounded reliability-weighted frame correction.

    ``source`` and ``target`` must contain material correspondences. Zero-weight
    points are ignored, which lets robust endpoint support define the fit.
    """

    if mode not in FRAME_MODES:
        raise ValueError(f"mode must be one of {FRAME_MODES}")
    source_points = _points(source, name="source")
    target_points = _points(target, name="target")
    if target_points.shape != source_points.shape:
        raise ValueError("source and target must have matching shapes")
    reliability = np.asarray(weights, dtype=float)
    if reliability.shape != (len(source_points),):
        raise ValueError("weights must match the point count")
    if np.any(reliability < 0.0) or not np.all(np.isfinite(reliability)):
        raise ValueError("weights must be finite and nonnegative")
    if maximum_rotation_rad < 0.0 or maximum_translation_m < 0.0:
        raise ValueError("frame bounds must be nonnegative")
    active = reliability > 0.0
    minimum_points = 3 if mode == "se3" else 1
    if int(np.sum(active)) < minimum_points:
        raise ValueError(f"{mode} frame fit needs at least {minimum_points} points")
    if mode == "none":
        return RigidFrameCorrection(
            linear=np.eye(3),
            translation=np.zeros(3),
            mode=mode,
            rotation_angle_rad=0.0,
            fitted_point_count=int(np.sum(active)),
        )

    source_active = source_points[active]
    target_active = target_points[active]
    active_weights = reliability[active]
    active_weights /= np.sum(active_weights)
    source_centroid = np.sum(active_weights[:, None] * source_active, axis=0)
    target_centroid = np.sum(active_weights[:, None] * target_active, axis=0)
    linear = np.eye(3)
    if mode == "se3":
        centered_source = source_active - source_centroid
        centered_target = target_active - target_centroid
        covariance = (active_weights[:, None] * centered_source).T @ centered_target
        linear = _proper_rotation(covariance)
        axis, angle = _column_axis_angle(linear.T)
        if angle > maximum_rotation_rad:
            linear = _column_rotation(axis, maximum_rotation_rad).T
    translation = target_centroid - source_centroid @ linear
    translation_norm = float(np.linalg.norm(translation))
    if translation_norm > maximum_translation_m:
        translation *= maximum_translation_m / translation_norm
    angle = rotation_angle(linear)
    return RigidFrameCorrection(
        linear=linear,
        translation=translation,
        mode=mode,
        rotation_angle_rad=angle,
        fitted_point_count=int(np.sum(active)),
    )


def scaled_frame_correction(
    correction: RigidFrameCorrection,
    scale: float,
) -> RigidFrameCorrection:
    """Scale a fitted correction from identity without leaving SE(3)."""

    if not 0.0 <= scale <= 1.0 or not np.isfinite(scale):
        raise ValueError("scale must lie in [0, 1]")
    axis, angle = _column_axis_angle(np.asarray(correction.linear, dtype=float).T)
    linear = _column_rotation(axis, scale * angle).T
    return RigidFrameCorrection(
        linear=linear,
        translation=scale * np.asarray(correction.translation, dtype=float),
        mode=correction.mode,
        rotation_angle_rad=scale * angle,
        fitted_point_count=correction.fitted_point_count,
    )


def apply_frame_correction(
    points: np.ndarray,
    correction: RigidFrameCorrection,
) -> np.ndarray:
    """Apply a frame correction to any array whose final dimension is three."""

    values = np.asarray(points, dtype=float)
    if values.ndim < 2 or values.shape[-1] != 3 or not np.all(np.isfinite(values)):
        raise ValueError("points must be finite with final dimension three")
    return values @ correction.linear + correction.translation


def rotate_vectors(
    vectors: np.ndarray,
    correction: RigidFrameCorrection,
) -> np.ndarray:
    """Rotate vectors without applying frame translation."""

    values = np.asarray(vectors, dtype=float)
    if values.ndim < 1 or values.shape[-1] != 3 or not np.all(np.isfinite(values)):
        raise ValueError("vectors must be finite with final dimension three")
    return values @ correction.linear


def clip_vector_norm(values: np.ndarray, maximum_norm: float) -> np.ndarray:
    """Clip row-wise vector norms while preserving directions."""

    vectors = _points(values, name="values")
    if maximum_norm <= 0.0 or not np.isfinite(maximum_norm):
        raise ValueError("maximum_norm must be positive and finite")
    norms = np.linalg.norm(vectors, axis=1)
    scale = np.minimum(1.0, maximum_norm / np.maximum(norms, 1e-15))
    return vectors * scale[:, None]


def corrected_spring_rest_lengths(
    reference_vertices: np.ndarray,
    springs: np.ndarray,
    released_rest_lengths: np.ndarray,
    *,
    num_object_springs: int,
    nonrigid_field: np.ndarray,
    correction_scale: float,
    maximum_log_ratio: float = np.log(1.15),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild object rest lengths while preserving actuation spring lengths."""

    vertices = _points(reference_vertices, name="reference_vertices")
    edges = np.asarray(springs, dtype=np.int64)
    released = np.asarray(released_rest_lengths, dtype=float)
    field = _points(nonrigid_field, name="nonrigid_field")
    if edges.ndim != 2 or edges.shape[1] != 2 or len(edges) != len(released):
        raise ValueError("springs and released_rest_lengths must agree")
    if not 0 < num_object_springs <= len(edges):
        raise ValueError("num_object_springs must lie in (0, S]")
    if field.shape != vertices.shape:
        raise ValueError("nonrigid_field must match reference_vertices")
    if np.any(edges < 0) or np.any(edges >= len(vertices)):
        raise ValueError("spring endpoint exceeds reference_vertices")
    if np.any(released <= 0.0) or not np.all(np.isfinite(released)):
        raise ValueError("released rest lengths must be positive and finite")
    if not 0.0 <= correction_scale <= 1.0 or not np.isfinite(correction_scale):
        raise ValueError("correction_scale must lie in [0, 1]")
    if maximum_log_ratio <= 0.0 or not np.isfinite(maximum_log_ratio):
        raise ValueError("maximum_log_ratio must be positive and finite")

    corrected_vertices = vertices + correction_scale * field
    object_edges = edges[:num_object_springs]
    geometric = np.linalg.norm(
        corrected_vertices[object_edges[:, 0]]
        - corrected_vertices[object_edges[:, 1]],
        axis=1,
    )
    released_object = released[:num_object_springs]
    unclipped_ratio = geometric / released_object
    ratio = np.exp(
        np.clip(np.log(np.maximum(unclipped_ratio, 1e-12)), -maximum_log_ratio, maximum_log_ratio)
    )
    corrected = released.copy()
    corrected[:num_object_springs] = released_object * ratio
    return corrected_vertices, corrected, unclipped_ratio, ratio


def reattach_controller_rest_lengths(
    corrected_object_vertices: np.ndarray,
    controller_reference_vertices: np.ndarray,
    springs: np.ndarray,
    rest_lengths: np.ndarray,
    *,
    num_object_springs: int,
    maximum_log_ratio: float = np.log(1.15),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recompute attachment rest lengths after changing object rest geometry."""

    object_vertices = _points(
        corrected_object_vertices, name="corrected_object_vertices"
    )
    controllers = _points(
        controller_reference_vertices, name="controller_reference_vertices"
    )
    edges = np.asarray(springs, dtype=np.int64)
    released = np.asarray(rest_lengths, dtype=float)
    if edges.ndim != 2 or edges.shape[1] != 2 or len(edges) != len(released):
        raise ValueError("springs and rest_lengths must agree")
    if not 0 < num_object_springs <= len(edges):
        raise ValueError("num_object_springs must lie in (0, S]")
    if maximum_log_ratio <= 0.0 or not np.isfinite(maximum_log_ratio):
        raise ValueError("maximum_log_ratio must be positive and finite")
    object_count = len(object_vertices)
    all_vertices = np.concatenate((object_vertices, controllers), axis=0)
    if np.any(edges < 0) or np.any(edges >= len(all_vertices)):
        raise ValueError("spring endpoint exceeds the corrected graph")
    tail = edges[num_object_springs:]
    if not len(tail):
        return released.copy(), np.empty(0), np.empty(0)
    if np.any(tail[:, 0] < object_count) or np.any(tail[:, 1] >= object_count):
        raise ValueError("controller springs must be ordered controller-to-object")
    geometric = np.linalg.norm(
        all_vertices[tail[:, 0]] - all_vertices[tail[:, 1]],
        axis=1,
    )
    released_tail = released[num_object_springs:]
    raw_ratio = geometric / released_tail
    ratio = np.exp(
        np.clip(np.log(np.maximum(raw_ratio, 1e-12)), -maximum_log_ratio, maximum_log_ratio)
    )
    corrected = released.copy()
    corrected[num_object_springs:] = released_tail * ratio
    return corrected, raw_ratio, ratio


def infer_graph_rest_geometry_correction(
    endpoint_state: np.ndarray,
    reference_vertices: np.ndarray,
    springs: np.ndarray,
    released_rest_lengths: np.ndarray,
    *,
    num_object_springs: int,
    endpoint_mean: np.ndarray,
    endpoint_variance: np.ndarray,
    observed: np.ndarray,
    laplacian,
    graph_prior_strength: float,
    frame_mode: str = "se3",
    frame_scale: float = 1.0,
    rest_geometry_scale: float = 1.0,
    maximum_frame_rotation_rad: float = np.deg2rad(5.0),
    maximum_frame_translation_m: float = 0.02,
    maximum_nonrigid_norm_m: float = 0.01,
    maximum_rest_log_ratio: float = np.log(1.15),
) -> GraphRestGeometryCorrection:
    """Infer a bounded frame plus graph-regularized rest-geometry update."""

    state = _points(endpoint_state, name="endpoint_state")
    reference = _points(reference_vertices, name="reference_vertices")
    means = _points(endpoint_mean, name="endpoint_mean")
    variance = np.asarray(endpoint_variance, dtype=float)
    support = np.asarray(observed, dtype=bool)
    if state.shape != reference.shape:
        raise ValueError("endpoint_state and reference_vertices must match")
    if len(means) > len(state):
        raise ValueError("endpoint observations exceed the object state")
    if variance.shape != (len(means),) or support.shape != (len(means),):
        raise ValueError("endpoint variance and support must match endpoint_mean")
    if not np.any(support):
        raise ValueError("at least one endpoint observation is required")
    if np.any(variance[support] <= 0.0) or not np.all(np.isfinite(variance[support])):
        raise ValueError("supported endpoint variances must be positive and finite")

    reference_variance = float(np.median(variance[support]))
    frame_weights = np.zeros(len(means), dtype=float)
    frame_weights[support] = reference_variance / variance[support]
    frame = fit_weighted_frame_correction(
        state[: len(means)],
        state[: len(means)] + means,
        frame_weights,
        mode=frame_mode,
        maximum_rotation_rad=maximum_frame_rotation_rad,
        maximum_translation_m=maximum_frame_translation_m,
    )
    frame = scaled_frame_correction(frame, frame_scale)
    frame_displacement = apply_frame_correction(state, frame) - state
    observed_nonrigid = means - frame_displacement[: len(means)]
    graph_posterior = graph_smoothed_discrepancy_posterior(
        observed_nonrigid,
        variance,
        support,
        laplacian,
        prior_strength=graph_prior_strength,
    )
    nonrigid = clip_vector_norm(
        graph_posterior.mean,
        maximum_nonrigid_norm_m,
    )
    corrected_frame_reference = apply_frame_correction(reference, frame)
    corrected_vertices, corrected_lengths, unclipped_ratio, ratio = (
        corrected_spring_rest_lengths(
            corrected_frame_reference,
            np.asarray(springs)[:num_object_springs],
            np.asarray(released_rest_lengths)[:num_object_springs],
            num_object_springs=num_object_springs,
            nonrigid_field=nonrigid,
            correction_scale=rest_geometry_scale,
            maximum_log_ratio=maximum_rest_log_ratio,
        )
    )
    # Restore the full controller-spring tail when one was supplied. The graph
    # posterior and reference state intentionally cover object vertices only.
    released_all = np.asarray(released_rest_lengths, dtype=float)
    if len(released_all) > num_object_springs:
        full_lengths = released_all.copy()
        full_lengths[:num_object_springs] = corrected_lengths
        corrected_lengths = full_lengths
    endpoint_correction = frame_displacement + rest_geometry_scale * nonrigid
    return GraphRestGeometryCorrection(
        frame=frame,
        nonrigid_field=nonrigid,
        endpoint_correction=endpoint_correction,
        corrected_reference_vertices=corrected_vertices,
        corrected_rest_lengths=corrected_lengths,
        unclipped_rest_length_ratio=unclipped_ratio,
        rest_length_ratio=ratio,
        observed_nonrigid_residual=observed_nonrigid,
        graph_observation_weight=graph_posterior.observation_weight,
        graph_reference_variance=graph_posterior.reference_variance,
    )
