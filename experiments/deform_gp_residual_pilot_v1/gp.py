"""Fixed-hyperparameter sparse variational Gaussian regression for a pilot.

The three output coordinates are independent conditional GPs, not a learned
cross-coordinate model. All observations enter the Gaussian sufficient
statistics. Inducing inputs are selected from training inputs without targets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.spatial.distance import cdist


def matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not x.size or not np.isfinite(x).all():
        raise ValueError("expected a nonempty finite matrix")
    return x


def kernel(
    x: np.ndarray, z: np.ndarray, length: float, arc_length: float | None
) -> np.ndarray:
    """Matern-5/2 on standardized dynamics, optionally times material RBF.

    For shared models, the last column is the unstandardized material coordinate.
    Dynamics distance is divided by sqrt(feature dimension).
    """
    if length <= 0 or (arc_length is not None and arc_length <= 0):
        raise ValueError("kernel lengths must be positive")
    dx, dz = (x[:, :-1], z[:, :-1]) if arc_length is not None else (x, z)
    radius = np.sqrt(cdist(dx, dz, metric="sqeuclidean") / dx.shape[1]) / length
    q = np.sqrt(5.0) * radius
    k = (1 + q + q * q / 3) * np.exp(-q)
    if arc_length is not None:
        k *= np.exp(-0.5 * ((x[:, -1, None] - z[None, :, -1]) / arc_length) ** 2)
    return k


@dataclass
class SparseGP:
    """Analytic optimum q(u) for Gaussian-likelihood variational GP regression.

    Hyperparameters are selected on validation, not estimated on test labels.
    A fit-only constant mean and output RMS normalization are explicit.
    """

    inducing: np.ndarray
    inducing_cholesky: np.ndarray
    gram: np.ndarray
    cross: np.ndarray
    location: np.ndarray
    scale: np.ndarray
    length: float
    arc_length: float | None
    observation_count: int

    @classmethod
    def prepare(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        *,
        length: float,
        arc_length: float | None = None,
        inducing_count: int = 128,
        seed: int = 20260907,
    ) -> SparseGP:
        x, y = matrix(x), matrix(y)
        if len(x) != len(y) or inducing_count < 1:
            raise ValueError("invalid observation or inducing count")
        # Stratified positions in trajectory-major input order avoid using only
        # a few recordings; the fixed RNG only jitters positions inside strata.
        m = min(inducing_count, len(x))
        rng = np.random.default_rng(seed)
        edges = np.linspace(0, len(x), m + 1, dtype=int)
        indices = np.array([rng.integers(edges[i], edges[i + 1]) for i in range(m)])
        z = x[indices].copy()
        kmm = kernel(z, z, length, arc_length)
        kmm.flat[:: m + 1] += 1e-8
        lower = np.linalg.cholesky(kmm)
        location = y.mean(axis=0)
        scale = np.maximum(y.std(axis=0), 1e-6)
        yy = (y - location) / scale
        gram = np.zeros((m, m))
        cross = np.zeros((m, y.shape[1]))
        for start in range(0, len(x), 2048):
            stop = min(start + 2048, len(x))
            phi = solve_triangular(
                lower,
                kernel(x[start:stop], z, length, arc_length).T,
                lower=True,
                check_finite=False,
            ).T
            gram += phi.T @ phi
            cross += phi.T @ yy[start:stop]
        return cls(z, lower, gram, cross, location, scale, length, arc_length, len(x))

    def features(self, x: np.ndarray) -> np.ndarray:
        return solve_triangular(
            self.inducing_cholesky,
            kernel(matrix(x), self.inducing, self.length, self.arc_length).T,
            lower=True,
            check_finite=False,
        ).T

    def coefficients(self, noise: float) -> tuple[np.ndarray, tuple[np.ndarray, bool]]:
        if not np.isfinite(noise) or noise <= 0:
            raise ValueError("noise variance must be positive")
        factor = cho_factor(
            self.gram + noise * np.eye(len(self.inducing)),
            lower=True,
            check_finite=False,
        )
        return cho_solve(factor, self.cross, check_finite=False), factor

    def predict(
        self, x: np.ndarray, *, noise: float, return_cov: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None]:
        x = matrix(x)
        weights, factor = self.coefficients(noise)
        pieces = []
        for start in range(0, len(x), 2048):
            pieces.append(self.features(x[start : start + 2048]) @ weights)
        mean = np.concatenate(pieces) * self.scale + self.location
        if not return_cov:
            return mean, None
        if len(x) > 2048:
            raise ValueError("joint covariance queries are capped at 2048 points")
        phi = self.features(x)
        cov = kernel(x, x, self.length, self.arc_length) - phi @ phi.T
        cov += noise * phi @ cho_solve(factor, phi.T, check_finite=False)
        cov = (cov + cov.T) / 2
        # Observation noise is included; covariance axes are point,point,output.
        cov.flat[:: len(x) + 1] += noise
        return mean, cov[:, :, None] * self.scale[None, None, :] ** 2


def self_test() -> dict[str, bool]:
    """Numerical controls, not empirical evidence."""
    x = np.linspace(-1, 1, 17)[:, None]
    y = np.column_stack((np.sin(3 * x[:, 0]), np.cos(2 * x[:, 0])))
    gp = SparseGP.prepare(x, y, length=0.8, inducing_count=len(x))
    test = np.linspace(-0.9, 0.9, 11)[:, None]
    mean, cov = gp.predict(test, noise=0.1, return_cov=True)
    kd = kernel(x, x, 0.8, None)
    kt = kernel(test, x, 0.8, None)
    expected = kt @ np.linalg.solve(
        kd + 0.1 * np.eye(len(x)), (y - gp.location) / gp.scale
    )
    expected = expected * gp.scale + gp.location
    np.testing.assert_allclose(mean, expected, atol=1e-6)
    expected_cov = (
        kernel(test, test, 0.8, None)
        - kt @ np.linalg.solve(kd + 0.1 * np.eye(len(x)), kt.T)
        + 0.1 * np.eye(len(test))
    )
    np.testing.assert_allclose(cov, expected_cov[:, :, None] * gp.scale**2, atol=1e-6)
    for c in range(y.shape[1]):
        assert np.linalg.eigvalsh(cov[:, :, c]).min() > 0
    z = SparseGP.prepare(x, np.zeros_like(y), length=1, inducing_count=8)
    assert np.array_equal(z.predict(test, noise=0.1)[0], np.zeros((len(test), 2)))
    rng = np.random.default_rng(1)
    shared = rng.normal(size=(20, 5))
    assert np.linalg.eigvalsh(kernel(shared, shared, 1, 0.4)).min() > -1e-10
    try:
        gp.predict(test, noise=0)
        raise AssertionError("zero noise was accepted")
    except ValueError:
        pass
    return {
        "exact_gp_mean_parity": True,
        "exact_gp_covariance_parity": True,
        "predictive_covariance_psd": True,
        "zero_target_control": True,
        "material_product_kernel_psd": True,
        "invalid_noise_rejected": True,
    }


if __name__ == "__main__":
    print(self_test())
