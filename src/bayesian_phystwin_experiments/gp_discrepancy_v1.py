"""Experimental hierarchical GP discrepancy; development protocol owned by #945.

No stable inference path is changed. Outputs are predictive readout residuals,
not inferred physical parameters or reachable physical-state corrections.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _finite(value: object, ndim: int, name: str) -> Array:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != ndim or 0 in result.shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a nonempty finite {ndim}-D array")
    return result


def _identities(value: object, count: int) -> NDArray[np.str_]:
    result = np.asarray(value)
    if result.shape != (count,) or result.dtype.kind != "U":
        raise ValueError("recording IDs must be a one-dimensional Unicode array")
    if any(not item.strip() for item in result.tolist()):
        raise ValueError("recording IDs must be nonempty")
    return result


def _rows(x: object, time: object, ids: object, maximum: int) -> tuple[Array, Array, NDArray[np.str_]]:
    features = _finite(x, 2, "features")
    times = _finite(time, 1, "times")
    names = _identities(ids, len(features))
    if times.shape != (len(features),) or len(features) > maximum:
        raise ValueError("unaligned rows or declared dense-GP row limit exceeded")
    if len(set(zip(names.tolist(), times.tolist(), strict=True))) != len(features):
        raise ValueError("duplicate recording/time observations are forbidden")
    return features, times, names


def matern32(x: Array, y: Array, lengthscale: float) -> Array:
    """Unit-variance Matérn-3/2 kernel; nonnegative distance by construction."""
    if not np.isfinite(lengthscale) or lengthscale <= 0:
        raise ValueError("lengthscale must be positive and finite")
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("kernel feature dimensions differ")
    # Row-wise distances avoid an N x M x feature_count temporary and avoid
    # cancellation/clipping in ||x||^2 + ||y||^2 - 2*x@y.T.
    squared = np.empty((len(x), len(y)), dtype=np.float64)
    for i, row in enumerate(x):
        squared[i] = np.sum(np.square((y - row) / lengthscale), axis=1)
    distance = np.sqrt(3.0 * squared)
    return (1.0 + distance) * np.exp(-distance)


@dataclass(frozen=True)
class GPConfig:
    lengthscale: float = 1.0
    time_lengthscale: float = 0.3
    shared_variance: float = 1.0
    session_variance: float = 0.3
    noise_variance: float = 0.05
    output_scale_floor: float = 1e-6
    max_rows: int = 768

    def __post_init__(self) -> None:
        positive = (self.lengthscale, self.time_lengthscale, self.shared_variance,
                    self.noise_variance, self.output_scale_floor)
        if any(not np.isfinite(v) or v <= 0 for v in positive):
            raise ValueError("scales, shared variance, and observation noise must be positive")
        if not np.isfinite(self.session_variance) or self.session_variance < 0:
            raise ValueError("session variance must be nonnegative")
        if isinstance(self.max_rows, bool) or not isinstance(self.max_rows, int) or self.max_rows < 1:
            raise ValueError("max_rows must be a positive integer")


@dataclass(frozen=True)
class GPPrediction:
    mean: Array  # time x mode
    covariance: Array  # mode x time x time; includes declared observation noise

    def joint_covariance(self, max_dimension: int = 2048) -> Array:
        """Time-major/mode-minor covariance; modes are conditionally independent."""
        n, modes = self.mean.shape
        if n * modes > max_dimension:
            raise ValueError("joint covariance exceeds declared dimension bound")
        joint = np.zeros((n * modes, n * modes), dtype=np.float64)
        for mode in range(modes):
            joint[mode::modes, mode::modes] = self.covariance[mode]
        return joint

    def sample(self, seed: int, count: int) -> Array:
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("sample count must be a positive integer")
        rng = np.random.default_rng(seed)
        n, modes = self.mean.shape
        draws = np.empty((count, n, modes), dtype=np.float64)
        for mode in range(modes):
            root = np.linalg.cholesky(self.covariance[mode])
            draws[:, :, mode] = self.mean[:, mode] + rng.standard_normal((count, n)) @ root.T
        return draws


@dataclass(frozen=True)
class RecordingDiscrepancyGP:
    config: GPConfig
    features: Array
    times: Array
    ids: NDArray[np.str_]
    feature_location: Array
    feature_scale: Array
    output_location: Array
    output_scale: Array
    cholesky: Array
    alpha: Array

    @staticmethod
    def fit(x: object, time: object, ids: object, residual: object,
            config: GPConfig = GPConfig()) -> RecordingDiscrepancyGP:
        features, times, names = _rows(x, time, ids, config.max_rows)
        response = _finite(residual, 2, "residual")
        if len(response) != len(features):
            raise ValueError("response rows differ from feature rows")
        location = features.mean(axis=0)
        scale = features.std(axis=0)
        scale = np.where(scale == 0.0, 1.0, scale)
        # RMS-normalized feature distance keeps lengthscales interpretable when
        # flattening the existing DLO causal features across free vertices.
        scale = scale * np.sqrt(features.shape[1])
        standardized = (features - location) / scale
        output_location = response.mean(axis=0)
        output_scale = np.maximum(response.std(axis=0), config.output_scale_floor)
        kernel = config.shared_variance * matern32(standardized, standardized, config.lengthscale)
        kernel += config.session_variance * matern32(times[:, None], times[:, None], config.time_lengthscale) * (names[:, None] == names[None, :])
        kernel += config.noise_variance * np.eye(len(features))
        # No hidden jitter, eigenvalue clipping, pseudo-inverse or retry.
        cholesky = np.linalg.cholesky(kernel)
        normalized = (response - output_location) / output_scale
        alpha = np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, normalized))
        return RecordingDiscrepancyGP(config, standardized.copy(), times.copy(), names.copy(),
                                     location, scale, output_location, output_scale,
                                     cholesky, alpha)

    def predict(self, x: object, time: object, ids: object) -> GPPrediction:
        features, times, names = _rows(x, time, ids, self.config.max_rows)
        if features.shape[1] != self.features.shape[1]:
            raise ValueError("prediction feature dimension differs from fitted model")
        # A repeated training row would count the very same observation twice.
        fitted_keys = set(zip(self.ids.tolist(), self.times.tolist(), strict=True))
        query_keys = set(zip(names.tolist(), times.tolist(), strict=True))
        if fitted_keys & query_keys:
            raise ValueError("query overlaps a conditioned recording/time observation")
        standardized = (features - self.feature_location) / self.feature_scale
        config = self.config
        cross = config.shared_variance * matern32(standardized, self.features, config.lengthscale)
        cross += config.session_variance * matern32(times[:, None], self.times[:, None], config.time_lengthscale) * (names[:, None] == self.ids[None, :])
        prior = config.shared_variance * matern32(standardized, standardized, config.lengthscale)
        prior += config.session_variance * matern32(times[:, None], times[:, None], config.time_lengthscale) * (names[:, None] == names[None, :])
        prior += config.noise_variance * np.eye(len(times))
        solved = np.linalg.solve(self.cholesky, cross.T)
        conditional = prior - solved.T @ solved
        conditional = (conditional + conditional.T) / 2.0
        np.linalg.cholesky(conditional)  # Explicit fail-closed covariance check.
        mean = self.output_location + (cross @ self.alpha) * self.output_scale
        covariance = self.output_scale[:, None, None] ** 2 * conditional[None, :, :]
        return GPPrediction(mean, covariance)


def chain_modes(node_count: int, rank: int, clamped_each_end: int = 2) -> Array:
    """Dirichlet modes on a chain; clamped rows are exactly zero."""
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (node_count, rank, clamped_each_end)):
        raise ValueError("chain sizes must be integers")
    free_count = node_count - 2 * clamped_each_end
    if clamped_each_end < 1 or not 1 <= rank <= free_count:
        raise ValueError("invalid chain dimensions or rank")
    restricted = np.diag(np.full(free_count, 2.0))
    restricted += np.diag(np.full(free_count - 1, -1.0), 1)
    restricted += np.diag(np.full(free_count - 1, -1.0), -1)
    _, vectors = np.linalg.eigh(restricted)
    basis = np.zeros((node_count, rank), dtype=np.float64)
    vectors = vectors[:, :rank]
    for column in range(rank):
        pivot = int(np.argmax(np.abs(vectors[:, column])))
        if vectors[pivot, column] < 0:
            vectors[:, column] *= -1
    basis[clamped_each_end:-clamped_each_end] = vectors
    return basis


def frame_block_diagonal(covariance: object, frame_width: int) -> Array:
    """Remove only cross-time dependence, preserving every per-frame block."""
    full = _finite(covariance, 2, "covariance")
    if full.shape[0] != full.shape[1] or frame_width < 1 or len(full) % frame_width:
        raise ValueError("covariance does not contain complete equal-sized frames")
    if not np.allclose(full, full.T, atol=1e-12, rtol=1e-10):
        raise ValueError("covariance must be symmetric")
    np.linalg.cholesky(full)
    result = np.zeros_like(full)
    for start in range(0, len(full), frame_width):
        block = slice(start, start + frame_width)
        result[block, block] = full[block, block]
    return result


def empirical_lowrank(errors: object, rank: int, diagonal_fraction: float,
                      variance_floor: float) -> Array:
    """Shrunk residual second moment about a fixed mean, fitted on calibration recordings."""
    residuals = _finite(errors, 2, "calibration errors")
    if not 1 <= rank <= min(residuals.shape):
        raise ValueError("rank exceeds available recordings or dimensions")
    if not np.isfinite(diagonal_fraction) or not 0 < diagonal_fraction <= 1:
        raise ValueError("diagonal_fraction must be in (0, 1]")
    if not np.isfinite(variance_floor) or variance_floor <= 0:
        raise ValueError("variance_floor must be positive")
    _, singular, vectors = np.linalg.svd(residuals / np.sqrt(len(residuals)), full_matrices=False)
    loading = vectors[:rank].T * singular[:rank]
    diagonal = diagonal_fraction * np.mean(residuals ** 2, axis=0) + variance_floor
    return (1 - diagonal_fraction) * (loading @ loading.T) + np.diag(diagonal)


def gaussian_metrics(error: object, covariance: object) -> dict[str, float]:
    residual = _finite(error, 1, "error")
    matrix = _finite(covariance, 2, "covariance")
    if matrix.shape != (len(residual), len(residual)) or not np.allclose(matrix, matrix.T, atol=1e-12, rtol=1e-10):
        raise ValueError("covariance dimensions or symmetry invalid")
    root = np.linalg.cholesky(matrix)
    whitened = np.linalg.solve(root, residual)
    nees = float(whitened @ whitened)
    logdet = float(2 * np.log(np.diag(root)).sum())
    std = np.sqrt(np.diag(matrix))
    return {
        "nll_per_dimension": float(0.5 * (nees + logdet + len(residual) * np.log(2 * np.pi)) / len(residual)),
        "normalized_nees": nees / len(residual),
        "marginal_90_coverage": float(np.mean(np.abs(residual) <= 1.6448536269514722 * std)),
        "mean_full_90_width": float(np.mean(2 * 1.6448536269514722 * std)),
    }
