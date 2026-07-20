"""Deterministic non-rigid coherent point-drift registration.

The implementation follows the Gaussian-kernel non-rigid CPD EM updates of
Myronenko and Song (2010).  It is intentionally small and NumPy-only so the
online-belief evaluation can include a classical set-registration control
without introducing a second perception or geometry dependency.

``fit_nonrigid_cpd`` treats both inputs as unordered point sets.  A fitted
transformation can subsequently be queried at arbitrary points, which makes
it possible to register a physical rollout from sparse observations at the
current update frame.  Each fit is independent; this module stores no state
across update times.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NonrigidCpdConfig:
    """Fixed non-rigid CPD hyperparameters in normalized object coordinates."""

    beta: float = 2.0
    regularization: float = 3.0
    outlier_weight: float = 0.1
    maximum_iterations: int = 100
    relative_tolerance: float = 1e-5
    minimum_variance: float = 1e-8
    minimum_scale_m: float = 1e-6

    def __post_init__(self) -> None:
        if not np.isfinite(self.beta) or self.beta <= 0.0:
            raise ValueError("beta must be positive")
        if not np.isfinite(self.regularization) or self.regularization <= 0.0:
            raise ValueError("regularization must be positive")
        if not np.isfinite(self.outlier_weight) or not 0.0 <= self.outlier_weight < 1.0:
            raise ValueError("outlier_weight must lie in [0, 1)")
        if self.maximum_iterations < 1:
            raise ValueError("maximum_iterations must be positive")
        if not np.isfinite(self.relative_tolerance) or self.relative_tolerance <= 0.0:
            raise ValueError("relative_tolerance must be positive")
        if not np.isfinite(self.minimum_variance) or self.minimum_variance <= 0.0:
            raise ValueError("minimum_variance must be positive")
        if not np.isfinite(self.minimum_scale_m) or self.minimum_scale_m <= 0.0:
            raise ValueError("minimum_scale_m must be positive")


@dataclass(frozen=True)
class NonrigidCpdTransform:
    """A Gaussian-kernel CPD deformation defined at source control points."""

    source_points_normalized: np.ndarray
    weights_normalized: np.ndarray
    origin_m: np.ndarray
    scale_m: float
    beta: float
    iterations: int
    converged: bool
    variance_normalized2: float
    effective_correspondence_count: float

    def __post_init__(self) -> None:
        source = np.asarray(self.source_points_normalized, dtype=float).copy()
        weights = np.asarray(self.weights_normalized, dtype=float).copy()
        origin = np.asarray(self.origin_m, dtype=float).copy()
        if source.ndim != 2 or source.shape[1] != 3 or len(source) == 0:
            raise ValueError("source_points_normalized must have shape (M, 3)")
        if weights.shape != source.shape:
            raise ValueError("weights_normalized must match the source shape")
        if origin.shape != (3,):
            raise ValueError("origin_m must have shape (3,)")
        if not (
            np.all(np.isfinite(source))
            and np.all(np.isfinite(weights))
            and np.all(np.isfinite(origin))
        ):
            raise ValueError("CPD transform arrays must be finite")
        positive = (
            self.scale_m,
            self.beta,
            self.variance_normalized2,
            self.effective_correspondence_count,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("CPD transform scales and counts must be positive")
        if self.iterations < 0:
            raise ValueError("iterations must be nonnegative")
        source.setflags(write=False)
        weights.setflags(write=False)
        origin.setflags(write=False)
        object.__setattr__(self, "source_points_normalized", source)
        object.__setattr__(self, "weights_normalized", weights)
        object.__setattr__(self, "origin_m", origin)

    def transform(self, query_points_m: np.ndarray) -> np.ndarray:
        """Apply the fitted spatial deformation to arbitrary 3D query points."""

        query = np.asarray(query_points_m, dtype=float)
        if query.ndim != 2 or query.shape[1] != 3:
            raise ValueError("query_points_m must have shape (Q, 3)")
        if not np.all(np.isfinite(query)):
            raise ValueError("query_points_m must be finite")
        normalized = (query - self.origin_m) / self.scale_m
        kernel = _gaussian_kernel(
            normalized,
            self.source_points_normalized,
            beta=self.beta,
        )
        transformed = normalized + kernel @ self.weights_normalized
        return transformed * self.scale_m + self.origin_m


def _gaussian_kernel(
    first: np.ndarray,
    second: np.ndarray,
    *,
    beta: float,
) -> np.ndarray:
    squared = np.sum(
        np.square(first[:, None, :] - second[None, :, :]),
        axis=2,
    )
    return np.exp(-squared / (2.0 * beta**2))


def _initial_variance(source: np.ndarray, target: np.ndarray) -> float:
    source_count, dimension = source.shape
    target_count = len(target)
    pairwise_squared = np.sum(
        np.square(source[:, None, :] - target[None, :, :]),
        axis=2,
    )
    return float(np.sum(pairwise_squared) / (dimension * source_count * target_count))


def fit_nonrigid_cpd(
    source_points_m: np.ndarray,
    target_points_m: np.ndarray,
    *,
    config: NonrigidCpdConfig | None = None,
) -> NonrigidCpdTransform:
    """Fit an unordered non-rigid CPD transformation from source to target.

    Coordinates are normalized by the source RMS radius before applying the
    declared CPD defaults.  This only removes unit/object-scale dependence; it
    does not center the target separately and therefore cannot erase the
    translation that registration is intended to estimate.
    """

    source_m = np.asarray(source_points_m, dtype=float)
    target_m = np.asarray(target_points_m, dtype=float)
    if source_m.ndim != 2 or source_m.shape[1] != 3 or len(source_m) < 3:
        raise ValueError("source_points_m must have shape (M, 3), M >= 3")
    if target_m.ndim != 2 or target_m.shape[1] != 3 or len(target_m) < 3:
        raise ValueError("target_points_m must have shape (N, 3), N >= 3")
    if not np.all(np.isfinite(source_m)) or not np.all(np.isfinite(target_m)):
        raise ValueError("source and target points must be finite")
    cfg = config or NonrigidCpdConfig()

    origin_m = np.mean(source_m, axis=0)
    centered = source_m - origin_m
    scale_m = max(
        float(np.sqrt(np.mean(np.sum(np.square(centered), axis=1)))),
        cfg.minimum_scale_m,
    )
    source = centered / scale_m
    target = (target_m - origin_m) / scale_m
    source_count, dimension = source.shape
    target_count = len(target)
    kernel = _gaussian_kernel(source, source, beta=cfg.beta)
    weights = np.zeros_like(source)
    transformed = source.copy()
    variance = max(_initial_variance(source, target), cfg.minimum_variance)
    effective_count = float(min(source_count, target_count))
    converged = False
    iterations = 0

    for iteration in range(1, cfg.maximum_iterations + 1):
        squared_distance = np.sum(
            np.square(transformed[:, None, :] - target[None, :, :]),
            axis=2,
        )
        probability = np.exp(-squared_distance / (2.0 * variance))
        if cfg.outlier_weight == 0.0:
            outlier = 0.0
        else:
            outlier = (
                (2.0 * np.pi * variance) ** (dimension / 2.0)
                * cfg.outlier_weight
                / (1.0 - cfg.outlier_weight)
                * source_count
                / target_count
            )
        denominator = np.sum(probability, axis=0) + outlier
        probability /= np.maximum(denominator[None, :], np.finfo(float).tiny)

        probability_source = np.sum(probability, axis=1)
        effective_count = float(np.sum(probability_source))
        if effective_count <= np.finfo(float).eps:
            raise RuntimeError("CPD lost all effective correspondences")
        probability_target = np.sum(probability, axis=0)
        weighted_target = probability @ target
        system = probability_source[:, None] * kernel
        system += cfg.regularization * variance * np.eye(source_count)
        right = weighted_target - probability_source[:, None] * source
        try:
            weights = np.linalg.solve(system, right)
        except np.linalg.LinAlgError:
            weights = np.linalg.lstsq(system, right, rcond=None)[0]
        transformed = source + kernel @ weights

        target_energy = float(
            np.sum(probability_target * np.sum(np.square(target), axis=1))
        )
        cross_energy = float(np.sum(weighted_target * transformed))
        source_energy = float(
            np.sum(probability_source * np.sum(np.square(transformed), axis=1))
        )
        updated_variance = max(
            (target_energy - 2.0 * cross_energy + source_energy)
            / (effective_count * dimension),
            cfg.minimum_variance,
        )
        relative_change = abs(updated_variance - variance) / max(
            variance,
            cfg.minimum_variance,
        )
        variance = updated_variance
        iterations = iteration
        if relative_change <= cfg.relative_tolerance:
            converged = True
            break

    return NonrigidCpdTransform(
        source_points_normalized=source,
        weights_normalized=weights,
        origin_m=origin_m,
        scale_m=scale_m,
        beta=cfg.beta,
        iterations=iterations,
        converged=converged,
        variance_normalized2=variance,
        effective_correspondence_count=effective_count,
    )
