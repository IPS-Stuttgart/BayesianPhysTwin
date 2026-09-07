"""Conditional Gaussian-mixture regression for a development-only DP screen.

Fit-only PCA; joint input/output mixture; test responsibilities use inputs only.
The DP is a truncated variational approximation, not an exact infinite model.
Mixture assignments apply to complete six-frame, all-free-node residual blocks.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp, ndtr
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Spec:
    family: str
    components: int = 1
    concentration: float = 1.0

    @property
    def name(self) -> str:
        return f"{self.family}-k{self.components}-a{self.concentration:g}"


def finite_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not x.size or not np.isfinite(x).all():
        raise ValueError("expected a nonempty finite matrix")
    return x


def median_of_mixture(w: np.ndarray, m: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Coordinatewise median. Shapes: (N,K), (N,K,D), (N,K,D)."""
    if w.shape != m.shape[:2] or v.shape != m.shape or np.any(v <= 0):
        raise ValueError("invalid mixture dimensions or variance")
    s = np.sqrt(v)
    low = np.min(m - 10 * s, axis=1)
    high = np.max(m + 10 * s, axis=1)
    for _ in range(45):
        mid = (low + high) / 2
        cdf = np.sum(w[:, :, None] * ndtr((mid[:, None, :] - m) / s), axis=1)
        low = np.where(cdf < 0.5, mid, low)
        high = np.where(cdf >= 0.5, mid, high)
    return (low + high) / 2


class ConditionalResidual:
    def __init__(self, spec: Spec, seed: int = 20260907):
        self.spec, self.seed = spec, seed

    def fit(self, x: np.ndarray, y: np.ndarray) -> ConditionalResidual:
        x, y = finite_matrix(x), finite_matrix(y)
        if len(x) != len(y) or len(x) < 16 or y.shape[1] % 3:
            raise ValueError("invalid regression sample or xyz output shape")
        self.scaler = StandardScaler().fit(x)
        self.xpca = PCA(
            n_components=min(6, x.shape[1], len(x) - 1), whiten=True, svd_solver="full"
        )
        self.ypca = PCA(
            n_components=min(8, y.shape[1], len(y) - 1), whiten=True, svd_solver="full"
        )
        a = self.xpca.fit_transform(self.scaler.transform(x))
        b = self.ypca.fit_transform(y)
        if (
            np.min(self.xpca.explained_variance_) <= 1e-12
            or np.min(self.ypca.explained_variance_) <= 1e-16
        ):
            raise ValueError("degenerate fit-only PCA basis")
        self.dx = a.shape[1]
        self.basis = (
            self.ypca.components_ * np.sqrt(self.ypca.explained_variance_)[:, None]
        )
        reconstruction_error = (y - self.ypca.inverse_transform(b)).reshape(
            len(y), -1, 3
        )
        self.discarded = np.einsum(
            "npi,npj->pij", reconstruction_error, reconstruction_error
        ) / len(y)
        self.discarded += np.eye(3)[None] * 1e-6
        if self.spec.family == "ridge":
            design = np.column_stack([np.ones(len(a)), a])
            penalty = np.eye(design.shape[1])
            penalty[0, 0] = 0
            self.beta = np.linalg.solve(design.T @ design + penalty, design.T @ b)
            residual = b - design @ self.beta
            self.ridge_cov = residual.T @ residual / len(a) + np.eye(b.shape[1]) * 0.05
            self.metadata = {
                "converged": True,
                "active_weight_gt_001": 1,
                "warnings": [],
            }
            return self
        args = dict(
            n_components=self.spec.components,
            covariance_type="full",
            reg_covar=0.05,
            n_init=2,
            max_iter=300,
            tol=1e-4,
            random_state=self.seed,
        )
        if self.spec.family == "gmm":
            self.mixture = GaussianMixture(**args)
        elif self.spec.family in ("dp", "finite-bayes"):
            prior_type = (
                "dirichlet_process"
                if self.spec.family == "dp"
                else "dirichlet_distribution"
            )
            alpha = (
                self.spec.concentration
                if self.spec.family == "dp"
                else self.spec.concentration / self.spec.components
            )
            self.mixture = BayesianGaussianMixture(
                **args,
                weight_concentration_prior_type=prior_type,
                weight_concentration_prior=alpha,
            )
        else:
            raise ValueError("unknown mixture family")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            self.mixture.fit(np.column_stack([a, b]))
        self.metadata = {
            "converged": bool(self.mixture.converged_),
            "active_weight_gt_001": int(np.sum(self.mixture.weights_ > 0.01)),
            "weights": self.mixture.weights_.tolist(),
            "iterations": int(self.mixture.n_iter_),
            "warnings": [str(v.message) for v in caught],
        }
        return self

    def latent_components(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = self.xpca.transform(self.scaler.transform(finite_matrix(x)))
        if self.spec.family == "ridge":
            m = np.column_stack([np.ones(len(a)), a]) @ self.beta
            return np.ones((len(a), 1)), m[:, None, :], self.ridge_cov[None]
        logs, means, covariances = [], [], []
        for weight, mean, cov in zip(
            self.mixture.weights_,
            self.mixture.means_,
            self.mixture.covariances_,
            strict=True,
        ):
            xx = cov[: self.dx, : self.dx]
            xy = cov[: self.dx, self.dx :]
            delta = a - mean[: self.dx]
            solve = np.linalg.solve(xx, xy)
            logs.append(
                np.log(max(weight, 1e-300))
                - 0.5
                * (
                    self.dx * np.log(2 * np.pi)
                    + np.linalg.slogdet(xx)[1]
                    + np.sum(delta * np.linalg.solve(xx, delta.T).T, axis=1)
                )
            )
            means.append(mean[self.dx :] + delta @ solve)
            conditional_cov = cov[self.dx :, self.dx :] - xy.T @ solve
            covariances.append((conditional_cov + conditional_cov.T) / 2)
        log_weights = np.stack(logs, axis=1)
        weights = np.exp(log_weights - logsumexp(log_weights, axis=1, keepdims=True))
        return weights, np.stack(means, axis=1), np.stack(covariances)

    def predict(self, x: np.ndarray, frames: np.ndarray) -> dict[str, np.ndarray]:
        """No outcome argument. Rotate distributions before reading world-axis L1 medians."""
        weights, latent_mean, latent_cov = self.latent_components(x)
        frames = np.asarray(frames, dtype=float)
        if frames.shape != (len(x), 3, 3) or not np.isfinite(frames).all():
            raise ValueError("one finite action frame is required per block")
        canonical = (latent_mean @ self.basis + self.ypca.mean_).reshape(
            len(x), -1, self.basis.shape[1] // 3, 3
        )
        means = np.einsum("nkpc,nwc->nkpw", canonical, frames).reshape(
            len(x), weights.shape[1], -1
        )
        basis = self.basis.reshape(self.basis.shape[0], -1, 3)
        world_basis = np.einsum("qpc,nwc->nqpw", basis, frames).reshape(
            len(x), self.basis.shape[0], -1
        )
        variances = np.einsum("nqd,kqr,nrd->nkd", world_basis, latent_cov, world_basis)
        discarded = np.einsum(
            "nwc,pcd,nwd->npw", frames, self.discarded, frames
        ).reshape(len(x), -1)
        variances = np.maximum(variances + discarded[:, None, :], 1e-12)
        return {
            "mean": np.sum(weights[:, :, None] * means, axis=1),
            "median": median_of_mixture(weights, means, variances),
            "hard_mean": means[np.arange(len(x)), weights.argmax(axis=1)],
            "weights": weights,
        }
