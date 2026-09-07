"""Source-development GP residuals; not a promoted physical-belief backend.

Exact conditioning on a deterministic, group-balanced subset of observations.
Outputs are independent conditional on the shared kernel hyperparameters.
Noise is explicitly specified in output-normalized units. No jitter is added.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def finite_matrix(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite matrix")
    return array


def matern32(left: np.ndarray, right: np.ndarray, length_scale: float) -> np.ndarray:
    """Unit-amplitude Matern-3/2; divide distance by sqrt(feature count)."""
    left = finite_matrix(left, "left features")
    right = finite_matrix(right, "right features")
    if left.shape[1] != right.shape[1]:
        raise ValueError("feature counts differ")
    if not np.isfinite(length_scale) or length_scale <= 0:
        raise ValueError("length_scale must be finite and positive")
    result = np.empty((len(left), len(right)), dtype=np.float64)
    denominator = length_scale * np.sqrt(left.shape[1])
    for start in range(0, len(left), 64):
        difference = left[start : start + 64, None, :] - right[None, :, :]
        radius = np.sqrt(3.0) * np.linalg.norm(difference, axis=2) / denominator
        result[start : start + 64] = (1.0 + radius) * np.exp(-radius)
    return result


def grouped_support(groups: np.ndarray, per_group: int) -> np.ndarray:
    """Retain evenly spaced rows within every complete recording.

    Input order must be chronological within each group. Selection never uses
    residual values. Duplicate rows are not additional independent recordings.
    """
    groups = np.asarray(groups)
    if groups.ndim != 1 or not groups.size or groups.dtype.kind not in "Ui":
        raise ValueError("groups must be a nonempty Unicode or integer vector")
    if isinstance(per_group, bool) or not isinstance(per_group, int) or per_group < 1:
        raise ValueError("per_group must be a positive integer")
    selected = []
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        positions = np.linspace(0, len(indices) - 1, min(per_group, len(indices)))
        selected.extend(indices[np.rint(positions).astype(np.int64)])
    return np.sort(np.asarray(selected, dtype=np.int64))


def require_disjoint_groups(*partitions: list[str]) -> None:
    """Reject duplicate recording identities within or across partitions."""
    seen: set[str] = set()
    for names in partitions:
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("recording identities must be nonempty strings")
        if len(set(names)) != len(names) or seen.intersection(names):
            raise ValueError("recording groups overlap")
        seen.update(names)


@dataclass(frozen=True)
class GaussianResidual:
    """Zero-prior-mean GP with fit-only feature and output scales."""

    location: np.ndarray
    scale: np.ndarray
    output_scale: np.ndarray
    support: np.ndarray
    cholesky: np.ndarray
    alpha: np.ndarray
    length_scale: float
    noise_variance: float

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        residuals: np.ndarray,
        groups: np.ndarray,
        *,
        length_scale: float,
        noise_variance: float,
        per_group: int = 8,
    ) -> GaussianResidual:
        features = finite_matrix(features, "fit features")
        residuals = finite_matrix(residuals, "fit residuals")
        groups = np.asarray(groups)
        if len(features) != len(residuals) or groups.shape != (len(features),):
            raise ValueError("fit rows and recording groups do not align")
        if not np.isfinite(noise_variance) or noise_variance <= 0:
            raise ValueError("noise_variance must be finite and positive")
        selected = grouped_support(groups, per_group)
        location = features.mean(axis=0)
        scale = features.std(axis=0)
        # Versioned normalization convention, not covariance regularization.
        scale = np.where(scale > 1e-10, scale, 1.0)
        output_scale = np.sqrt(np.mean(np.square(residuals), axis=0))
        output_scale = np.where(output_scale > 1e-10, output_scale, 1.0)
        support = (features[selected] - location) / scale
        covariance = matern32(support, support, length_scale)
        covariance += noise_variance * np.eye(len(support))
        cholesky = np.linalg.cholesky(covariance)
        response = residuals[selected] / output_scale
        alpha = np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, response))
        return cls(
            location,
            scale,
            output_scale,
            support,
            cholesky,
            alpha,
            float(length_scale),
            float(noise_variance),
        )

    def predict(
        self,
        features: np.ndarray,
        *,
        full_covariance: bool = False,
        include_noise: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return mean and per-output latent covariance or marginal variance.

        include_noise adds the declared likelihood variance, not uncertainty in
        the physical backbone. Full covariance shape is (outputs, rows, rows).
        """
        features = finite_matrix(features, "query features")
        if features.shape[1] != len(self.location):
            raise ValueError("query feature count differs from fit")
        query = (features - self.location) / self.scale
        cross = matern32(query, self.support, self.length_scale)
        mean = (cross @ self.alpha) * self.output_scale
        solved = np.linalg.solve(self.cholesky, cross.T)
        noise = self.noise_variance if include_noise else 0.0
        if full_covariance:
            covariance = matern32(query, query, self.length_scale) - solved.T @ solved
            covariance += noise * np.eye(len(query))
            if not np.isfinite(covariance).all() or np.any(np.diag(covariance) < 0):
                raise ArithmeticError(
                    "invalid posterior covariance; no clipping applied"
                )
            return mean, np.square(self.output_scale)[:, None, None] * covariance
        variance = 1.0 + noise - np.sum(np.square(solved), axis=0)
        if not np.isfinite(variance).all() or np.any(variance < 0):
            raise ArithmeticError("invalid posterior variance; no clipping applied")
        return mean, variance[:, None] * np.square(self.output_scale)

    def predict_mean(self, features: np.ndarray) -> np.ndarray:
        """Avoid unnecessary variance solves during mean-only model selection."""
        features = finite_matrix(features, "query features")
        if features.shape[1] != len(self.location):
            raise ValueError("query feature count differs from fit")
        query = (features - self.location) / self.scale
        return (
            matern32(query, self.support, self.length_scale) @ self.alpha
        ) * self.output_scale


def apply_canonical_correction(
    baseline: np.ndarray,
    correction: np.ndarray,
    frames: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Correct free-node readouts; zero strength is exact array identity."""
    if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("correction strength must be in [0, 1]")
    baseline = np.asarray(baseline)
    correction = np.asarray(correction, dtype=np.float64)
    frames = np.asarray(frames, dtype=np.float64)
    if (
        baseline.ndim != 4
        or baseline.shape[-1] != 3
        or baseline.shape[2] < 5
        or correction.shape != (*baseline.shape[:2], baseline.shape[2] - 4, 3)
        or frames.shape != (baseline.shape[0], 3, 3)
        or not all(np.isfinite(value).all() for value in (baseline, correction, frames))
    ):
        raise ValueError("correction inputs are invalid")
    if not np.allclose(np.swapaxes(frames, 1, 2) @ frames, np.eye(3), atol=1e-10):
        raise ValueError("frames must be orthonormal")
    if strength == 0.0:
        return baseline
    candidate = baseline.copy()
    candidate[:, :, 2:-2] += strength * np.einsum("ntvj,nij->ntvi", correction, frames)
    return candidate
