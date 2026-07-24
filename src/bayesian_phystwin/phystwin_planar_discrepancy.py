"""Low-capacity discrepancy fields in a fixed canonical planar frame.

The planar frame is a deterministic PCA proxy derived from the initial object
geometry.  It is useful for diagnosing thin objects such as cloth, where
decoding a field from the current 3D positions can mix spatially adjacent but
materially distant regions after folding.  It is not a recovered UV map.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _validate_points(name: str, value: np.ndarray) -> np.ndarray:
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must be finite")
    return points


def _canonical_sign(vector: np.ndarray) -> np.ndarray:
    result = np.asarray(vector, dtype=float).copy()
    pivot = int(np.argmax(np.abs(result)))
    if result[pivot] < 0.0:
        result *= -1.0
    return result


def _feature_matrix(coordinates: np.ndarray, degree: int) -> np.ndarray:
    uv = np.asarray(coordinates, dtype=float)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("coordinates must have shape (N, 2)")
    if degree not in (0, 1, 2):
        raise ValueError("degree must be zero, one, or two")
    one = np.ones(len(uv), dtype=float)
    if degree == 0:
        return one[:, None]
    u = uv[:, 0]
    v = uv[:, 1]
    if degree == 1:
        return np.column_stack((one, u, v))
    return np.column_stack((one, u, v, np.square(u), u * v, np.square(v)))


@dataclass(frozen=True)
class CanonicalPlanarDiscrepancy:
    """A robust polynomial discrepancy field in fixed planar coordinates."""

    center_m: np.ndarray
    basis: np.ndarray
    coordinate_center: np.ndarray
    coordinate_scale: np.ndarray
    coefficients_m: np.ndarray
    degree: int
    fit_count: int
    robust_scale_m: float

    def __post_init__(self) -> None:
        center = np.asarray(self.center_m, dtype=float).copy()
        basis = np.asarray(self.basis, dtype=float).copy()
        coordinate_center = np.asarray(self.coordinate_center, dtype=float).copy()
        coordinate_scale = np.asarray(self.coordinate_scale, dtype=float).copy()
        coefficients = np.asarray(self.coefficients_m, dtype=float).copy()
        feature_count = (1, 3, 6)[self.degree] if self.degree in (0, 1, 2) else -1
        if center.shape != (3,):
            raise ValueError("center_m must have shape (3,)")
        if basis.shape != (3, 2):
            raise ValueError("basis must have shape (3, 2)")
        if coordinate_center.shape != (2,) or coordinate_scale.shape != (2,):
            raise ValueError("coordinate moments must have shape (2,)")
        if coefficients.shape != (feature_count, 3):
            raise ValueError("coefficients_m does not match degree")
        if not all(
            np.all(np.isfinite(value))
            for value in (
                center,
                basis,
                coordinate_center,
                coordinate_scale,
                coefficients,
            )
        ):
            raise ValueError("planar discrepancy parameters must be finite")
        if np.any(coordinate_scale <= 0.0):
            raise ValueError("coordinate_scale must be positive")
        if self.fit_count < feature_count:
            raise ValueError("fit_count is smaller than the feature count")
        if not np.isfinite(self.robust_scale_m) or self.robust_scale_m <= 0.0:
            raise ValueError("robust_scale_m must be positive")
        for name, value in (
            ("center_m", center),
            ("basis", basis),
            ("coordinate_center", coordinate_center),
            ("coordinate_scale", coordinate_scale),
            ("coefficients_m", coefficients),
        ):
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    def coordinates(self, points_m: np.ndarray) -> np.ndarray:
        """Map 3D points into the fixed standardized planar frame."""

        points = _validate_points("points_m", points_m)
        projected = (points - self.center_m) @ self.basis
        return (projected - self.coordinate_center) / self.coordinate_scale

    def predict(self, points_m: np.ndarray) -> np.ndarray:
        """Decode the fitted 3D discrepancy at arbitrary initial points."""

        return _feature_matrix(self.coordinates(points_m), self.degree) @ (
            self.coefficients_m
        )


def fit_canonical_planar_discrepancy(
    geometry_points_m: np.ndarray,
    observation_indices: np.ndarray,
    observation_mean_m: np.ndarray,
    observation_variance_m2: np.ndarray,
    observed: np.ndarray,
    *,
    degree: int,
    ridge_strength: float = 1e-3,
    huber_multiplier: float = 1.5,
    maximum_iterations: int = 20,
) -> CanonicalPlanarDiscrepancy:
    """Fit a robust prefix discrepancy without reading target outcomes.

    ``geometry_points_m`` and ``observation_indices`` define a fixed coordinate
    attachment.  Observation variance supplies metric precision; residuals are
    used once inside Huber IRLS and are not converted into a prior reliability.
    """

    geometry = _validate_points("geometry_points_m", geometry_points_m)
    indices = np.asarray(observation_indices, dtype=np.int64)
    mean = np.asarray(observation_mean_m, dtype=float)
    variance = np.asarray(observation_variance_m2, dtype=float)
    mask = np.asarray(observed, dtype=bool).copy()
    if indices.ndim != 1:
        raise ValueError("observation_indices must be a vector")
    if mean.shape != (len(indices), 3):
        raise ValueError("observation_mean_m must have shape (M, 3)")
    if variance.shape not in ((len(indices),), (len(indices), 3)):
        raise ValueError("observation_variance_m2 must have shape (M,) or (M, 3)")
    if mask.shape != (len(indices),):
        raise ValueError("observed must have shape (M,)")
    if np.any(indices < 0) or np.any(indices >= len(geometry)):
        raise ValueError("observation index exceeds geometry")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("observation_indices must be unique")
    if degree not in (0, 1, 2):
        raise ValueError("degree must be zero, one, or two")
    if not np.isfinite(ridge_strength) or ridge_strength < 0.0:
        raise ValueError("ridge_strength must be nonnegative")
    if not np.isfinite(huber_multiplier) or huber_multiplier <= 0.0:
        raise ValueError("huber_multiplier must be positive")
    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive")

    scalar_variance = (
        variance if variance.ndim == 1 else np.mean(variance, axis=1)
    )
    mask &= np.all(np.isfinite(mean), axis=1)
    mask &= np.isfinite(scalar_variance) & (scalar_variance > 0.0)
    feature_count = (1, 3, 6)[degree]
    if int(np.sum(mask)) < feature_count:
        raise ValueError("too few valid observations for requested degree")

    center = np.mean(geometry, axis=0)
    centered = geometry - center
    covariance = centered.T @ centered / max(len(centered), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues, kind="mergesort")[::-1]
    basis = np.column_stack(
        (
            _canonical_sign(eigenvectors[:, order[0]]),
            _canonical_sign(eigenvectors[:, order[1]]),
        )
    )
    # Re-orthogonalize after deterministic sign selection.
    basis[:, 1] -= basis[:, 0] * float(basis[:, 0] @ basis[:, 1])
    basis[:, 1] /= max(float(np.linalg.norm(basis[:, 1])), 1e-15)

    projected = centered @ basis
    coordinate_center = np.median(projected, axis=0)
    lower = np.quantile(projected, 0.05, axis=0)
    upper = np.quantile(projected, 0.95, axis=0)
    coordinate_scale = np.maximum(upper - lower, 1e-6)
    coordinates = (projected - coordinate_center) / coordinate_scale
    design = _feature_matrix(coordinates[indices[mask]], degree)
    target = mean[mask]

    precision = 1.0 / scalar_variance[mask]
    precision /= max(float(np.median(precision)), 1e-15)
    precision = np.clip(precision, 1e-2, 1e2)
    robust_weight = np.ones(len(target), dtype=float)
    penalty = np.eye(feature_count, dtype=float)
    penalty[0, 0] = 0.0
    coefficients = np.zeros((feature_count, 3), dtype=float)
    robust_scale = 1e-6
    for _ in range(maximum_iterations):
        total_weight = precision * robust_weight
        root_weight = np.sqrt(total_weight)
        weighted_design = design * root_weight[:, None]
        weighted_target = target * root_weight[:, None]
        normal = weighted_design.T @ weighted_design + ridge_strength * penalty
        rhs = weighted_design.T @ weighted_target
        updated = np.linalg.solve(normal, rhs)
        radial = np.linalg.norm(target - design @ updated, axis=1)
        median = float(np.median(radial))
        mad = float(np.median(np.abs(radial - median)))
        robust_scale = max(1e-6, 1.4826 * mad)
        cutoff = huber_multiplier * robust_scale
        updated_weight = np.minimum(1.0, cutoff / np.maximum(radial, 1e-15))
        if np.allclose(updated, coefficients, rtol=1e-8, atol=1e-10):
            coefficients = updated
            robust_weight = updated_weight
            break
        coefficients = updated
        robust_weight = updated_weight

    return CanonicalPlanarDiscrepancy(
        center_m=center,
        basis=basis,
        coordinate_center=coordinate_center,
        coordinate_scale=coordinate_scale,
        coefficients_m=coefficients,
        degree=degree,
        fit_count=int(np.sum(mask)),
        robust_scale_m=robust_scale,
    )
