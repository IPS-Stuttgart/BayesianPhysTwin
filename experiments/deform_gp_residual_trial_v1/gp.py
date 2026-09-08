"""Nodewise nonlinear residual regression with a finite-rank Matérn GP.

Exact Gaussian inference for the specified Nyström kernel approximation, not
variational GP inference and not a claim of calibrated physical uncertainty.
The unpenalized intercept and linear feature block match the existing DEFORM
ridge mean. All training targets are used; only the kernel basis is reduced.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.spatial.distance import cdist


@dataclass(frozen=True)
class GPConfig:
    ridge: float = 1.0
    amplitude: float = 3.0
    length_multiplier: float = 1.0
    max_anchors: int = 128
    seed: int = 20260907
    kernel_jitter: float = 1e-9

    def validate(self) -> None:
        for key in ('ridge', 'length_multiplier', 'kernel_jitter'):
            value = getattr(self, key)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if not math.isfinite(self.amplitude) or self.amplitude < 0:
            raise ValueError('amplitude must be finite and nonnegative')
        if isinstance(self.max_anchors, bool) or not isinstance(self.max_anchors, int) or self.max_anchors < 1:
            raise ValueError('max_anchors must be a positive integer')
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError('seed must be a nonnegative integer')


def matrix(value: Any, label: str) -> np.ndarray:
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 2 or not all(a.shape) or not np.isfinite(a).all():
        raise ValueError(f'{label} must be a nonempty, finite matrix')
    return a


def matern32(x: np.ndarray, z: np.ndarray, length: float) -> np.ndarray:
    if not math.isfinite(length) or length <= 0:
        raise ValueError('length must be positive')
    distance = math.sqrt(3.0) * cdist(x, z) / length
    return (1.0 + distance) * np.exp(-distance)


def balanced_anchor_indices(groups: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    """Round-robin sampling across trajectory clusters; never reads targets."""
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    queues = [list(rng.permutation(np.flatnonzero(groups == g))) for g in unique]
    chosen: list[int] = []
    while len(chosen) < min(maximum, len(groups)):
        for queue in queues:
            if queue:
                chosen.append(int(queue.pop()))
                if len(chosen) == min(maximum, len(groups)):
                    break
    return np.asarray(sorted(chosen), dtype=np.int64)


@dataclass
class ResidualGP:
    config: GPConfig
    location: np.ndarray
    scale: np.ndarray
    anchors: np.ndarray
    anchor_cholesky: np.ndarray
    weights: np.ndarray
    precision_inverse: np.ndarray
    residual_variance: np.ndarray
    cluster_covariance: np.ndarray
    training_rows: int
    trajectory_clusters: int
    anchor_indices: np.ndarray

    def design(self, features: np.ndarray) -> np.ndarray:
        x = matrix(features, 'features')
        if x.shape[1] != len(self.location):
            raise ValueError('feature dimension differs from fit')
        standardized = (x - self.location) / self.scale
        parts = [np.ones((len(x), 1)), standardized]
        if self.config.amplitude:
            cross = matern32(standardized, self.anchors,
                             self.config.length_multiplier * math.sqrt(x.shape[1]))
            # K_XZ L_Z^{-T}; its Gram matrix is K_XZ (K_ZZ+jI)^{-1} K_ZX.
            nonlinear = solve_triangular(self.anchor_cholesky, cross.T, lower=True,
                                        check_finite=False).T
            parts.append(self.config.amplitude * nonlinear)
        return np.concatenate(parts, axis=1)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.design(features) @ self.weights

    def marginal_variance(self, features: np.ndarray, *, robust: bool = False,
                          observation_noise: bool = True) -> np.ndarray:
        design = self.design(features)
        if robust:
            variance = np.einsum('np,cpq,nq->nc', design, self.cluster_covariance, design)
        else:
            leverage = np.einsum('np,pq,nq->n', design, self.precision_inverse, design)
            variance = leverage[:, None] * self.residual_variance[None]
        if observation_noise:
            variance = variance + self.residual_variance
        return np.maximum(variance, 0.0)

    def joint_covariance(self, features: np.ndarray, *, robust: bool = False,
                         observation_noise: bool = False) -> np.ndarray:
        """Return [coordinate, query, query]; no cross-coordinate covariance.

        Intended for small registered query sets, not a full trajectory tensor.
        The model-based posterior assumes conditionally independent noise. The
        sandwich option preserves trajectory clustering but is not a GP posterior.
        """
        design = self.design(features)
        if len(design) > 2048:
            raise ValueError('joint queries capped at 2048 to avoid dense allocation')
        if robust:
            cov = np.einsum('np,cpq,mq->cnm', design, self.cluster_covariance, design)
        else:
            cov = self.residual_variance[:, None, None] * (
                design @ self.precision_inverse @ design.T)[None]
        cov = (cov + cov.transpose(0, 2, 1)) / 2
        if observation_noise:
            cov += self.residual_variance[:, None, None] * np.eye(len(design))[None]
        return cov

    def arrays(self) -> dict[str, np.ndarray]:
        return {key: np.asarray(getattr(self, key)) for key in (
            'location', 'scale', 'anchors', 'anchor_cholesky', 'weights',
            'precision_inverse', 'residual_variance', 'cluster_covariance',
            'anchor_indices')}


def fit_gp(features: np.ndarray, residual: np.ndarray, groups: np.ndarray,
           config: GPConfig = GPConfig()) -> ResidualGP:
    config.validate()
    x, y = matrix(features, 'features'), matrix(residual, 'residual')
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) != len(x) or len(y) != len(x):
        raise ValueError('features, residuals and trajectory groups must align')
    if groups.dtype.kind not in 'iuUS':
        raise ValueError('trajectory groups must be integer or string identifiers')
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError('at least two trajectory clusters are required')
    location, scale = np.mean(x, axis=0), np.std(x, axis=0)
    scale = np.where(scale > 1e-10, scale, 1.0)
    standardized = (x - location) / scale
    indices = (balanced_anchor_indices(groups, config.max_anchors, config.seed)
               if config.amplitude else np.empty(0, dtype=np.int64))
    anchors = standardized[indices]
    if config.amplitude:
        kzz = matern32(anchors, anchors, config.length_multiplier * math.sqrt(x.shape[1]))
        chol = np.linalg.cholesky(kzz + config.kernel_jitter * np.eye(len(anchors)))
    else:
        chol = np.empty((0, 0), dtype=np.float64)
    model = ResidualGP(config, location, scale, anchors, chol, np.empty((0, 0)),
                       np.empty((0, 0)), np.empty(0), np.empty((0, 0, 0)),
                       len(x), len(unique), indices)
    design = model.design(x)
    penalty = np.eye(design.shape[1]) * config.ridge
    penalty[0, 0] = 0.0  # Exactly the existing DEFORM intercept convention.
    normal = design.T @ design + penalty
    factor = cho_factor(normal, lower=True, check_finite=False)
    inverse = cho_solve(factor, np.eye(len(normal)), check_finite=False)
    weights = cho_solve(factor, design.T @ y, check_finite=False)
    error = y - design @ weights
    variance = np.mean(error * error, axis=0)
    scores = np.stack([design[groups == g].T @ error[groups == g] for g in unique])
    meat = np.einsum('gpc,gqc->cpq', scores, scores) * len(unique) / (len(unique) - 1)
    robust_covariance = np.einsum('ip,cpq,qj->cij', inverse, meat, inverse)
    model.weights = weights
    model.precision_inverse = (inverse + inverse.T) / 2
    model.residual_variance = variance
    model.cluster_covariance = (robust_covariance + robust_covariance.transpose(0, 2, 1)) / 2
    return model
