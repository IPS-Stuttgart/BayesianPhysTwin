"""Development-only DP-gated ridge experts; no latent-physics or calibration claim.

The gate is a variational joint Gaussian mixture over source-fitted embeddings
of causal features and training residuals. Query weights integrate out residuals.
One shared gate is used for all material nodes at a given forecast time.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture


def finite(values: np.ndarray, ndim: int) -> np.ndarray:
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != ndim or a.size == 0 or not np.isfinite(a).all():
        raise ValueError(f"Expected nonempty finite {ndim}-D array")
    return a


def marginal_weights(
    x: np.ndarray, means: np.ndarray, covs: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """p(component|features): marginalize, never impute, unknown residuals."""
    x = finite(x, 2)
    d = x.shape[1]
    scores = []
    for mu, cov, weight in zip(means, covs, weights, strict=True):
        c = cov[:d, :d]
        chol = np.linalg.cholesky(c)
        whitened = np.linalg.solve(chol, (x - mu[:d]).T).T
        scores.append(
            np.log(max(float(weight), 1e-300))
            - np.log(np.diag(chol)).sum()
            - 0.5 * np.sum(whitened**2, axis=1)
        )
    logp = np.stack(scores, axis=1)
    result = np.exp(logp - logsumexp(logp, axis=1, keepdims=True))
    if not np.isfinite(result).all():
        raise FloatingPointError("Non-finite gate")
    return result


@dataclass(frozen=True)
class Spec:
    kind: str
    components: int = 1
    alpha: float = 1.0
    seed: int = 0

    @property
    def name(self) -> str:
        return f"{self.kind}-k{self.components}-a{self.alpha:g}-seed{self.seed}"


class ResidualExperts:
    """Plug-in variational gate plus weighted local ridge conditional means.

    This is a two-stage prototype, not a fully Bayesian DP regression posterior.
    Temporal dependence, correlated measurement errors, and state correction are
    not modeled here. Multiple nodes are one gating sample, not independent votes.
    """

    def __init__(
        self,
        spec: Spec,
        ridge: float = 1.0,
        feature_rank: int = 6,
        response_rank: int = 4,
        gate_stride: int = 10,
    ):
        if (
            spec.kind not in {"ridge", "rff", "finite", "dp", "finite_bayes"}
            or spec.components < 1
            or spec.alpha <= 0
            or ridge <= 0
            or gate_stride < 1
        ):
            raise ValueError("Invalid expert specification")
        self.spec, self.ridge = spec, float(ridge)
        self.feature_rank, self.response_rank = feature_rank, response_rank
        self.gate_stride = gate_stride

    @staticmethod
    def _gate_input(x: np.ndarray) -> np.ndarray:
        return np.concatenate((x.mean(axis=2), x.std(axis=2)), axis=-1)

    def fit(self, features: np.ndarray, residual: np.ndarray) -> ResidualExperts:
        x, y = finite(features, 4), finite(residual, 4)
        if x.shape[:3] != y.shape[:3] or y.shape[-1] != 3:
            raise ValueError("Feature and residual axes differ")
        n, h, nodes, dim = x.shape
        self.shape = (nodes, dim)
        self.location = x.mean(axis=(0, 1))
        scale = x.std(axis=(0, 1))
        self.scale = np.where(scale > 1e-10, scale, 1.0)
        standardized = (x - self.location) / self.scale
        self.warnings = []
        rng = np.random.default_rng(self.spec.seed)
        self.rff_w = rng.normal(size=(dim, 64)) / np.sqrt(dim)
        self.rff_b = rng.uniform(0, 2 * np.pi, size=64)
        if self.spec.kind in {"ridge", "rff"}:
            responsibility = np.ones((n * h, 1))
            self.gate = None
            self.active_components = 1
        else:
            gx = self._gate_input(standardized).reshape(n * h, -1)
            gy = y.reshape(n * h, -1)
            sample = np.arange(n * h).reshape(n, h)[:, :: self.gate_stride].ravel()
            self.gx_center = gx[sample].mean(axis=0)
            self.gx_scale = np.maximum(gx[sample].std(axis=0), 1e-8)
            self.gy_center = gy[sample].mean(axis=0)
            self.gy_scale = np.maximum(gy[sample].std(axis=0), 1e-6)
            self.x_pca = PCA(
                n_components=min(self.feature_rank, gx.shape[1], len(sample) - 1),
                whiten=True,
                svd_solver="full",
            )
            self.y_pca = PCA(
                n_components=min(self.response_rank, gy.shape[1], len(sample) - 1),
                whiten=True,
                svd_solver="full",
            )
            ex = self.x_pca.fit_transform((gx[sample] - self.gx_center) / self.gx_scale)
            ey = self.y_pca.fit_transform((gy[sample] - self.gy_center) / self.gy_scale)
            train = np.concatenate((ex, ey), axis=1)
            common = dict(
                n_components=self.spec.components,
                covariance_type="full",
                reg_covar=0.03,
                max_iter=300,
                tol=1e-4,
                n_init=2,
                random_state=self.spec.seed,
            )
            if self.spec.kind == "finite":
                self.gate = GaussianMixture(**common)
            else:
                prior = (
                    "dirichlet_process"
                    if self.spec.kind == "dp"
                    else "dirichlet_distribution"
                )
                concentration = (
                    self.spec.alpha
                    if self.spec.kind == "dp"
                    else self.spec.alpha / self.spec.components
                )
                self.gate = BayesianGaussianMixture(
                    **common,
                    weight_concentration_prior_type=prior,
                    weight_concentration_prior=concentration,
                )
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always", ConvergenceWarning)
                self.gate.fit(train)
            self.warnings = [str(w.message) for w in recorded]
            ex = self.x_pca.transform((gx - self.gx_center) / self.gx_scale)
            ey = self.y_pca.transform((gy - self.gy_center) / self.gy_scale)
            responsibility = self.gate.predict_proba(np.concatenate((ex, ey), axis=1))
            self.active_components = int(np.sum(self.gate.weights_ > 0.01))
        k = responsibility.shape[1]
        self.component_mass = responsibility.sum(axis=0)
        design_dim = dim + 1 + (64 if self.spec.kind == "rff" else 0)
        self.coef = np.zeros((k, nodes, design_dim, 3))
        penalty = np.eye(design_dim) * self.ridge
        penalty[0, 0] = 0
        for node in range(nodes):
            design = self._design(standardized[:, :, node].reshape(n * h, dim))
            response = y[:, :, node].reshape(n * h, 3)
            for component in range(k):
                w = responsibility[:, component]
                if w.sum() < 1e-10:
                    continue
                gram = design.T @ (design * w[:, None]) + penalty
                self.coef[component, node] = np.linalg.solve(
                    gram, design.T @ (response * w[:, None])
                )
        return self

    def _design(self, x: np.ndarray) -> np.ndarray:
        values = [np.ones((*x.shape[:-1], 1)), x]
        if self.spec.kind == "rff":
            values.append(np.sqrt(2 / 64) * np.cos(x @ self.rff_w + self.rff_b))
        return np.concatenate(values, axis=-1)

    def weights(self, features: np.ndarray) -> np.ndarray:
        x = finite(features, 4)
        n, h, nodes, dim = x.shape
        if (nodes, dim) != self.shape:
            raise ValueError("Query feature axes differ")
        if self.gate is None:
            return np.ones((n, h, 1))
        gx = self._gate_input((x - self.location) / self.scale).reshape(n * h, -1)
        ex = self.x_pca.transform((gx - self.gx_center) / self.gx_scale)
        return marginal_weights(
            ex, self.gate.means_, self.gate.covariances_, self.gate.weights_
        ).reshape(n, h, -1)

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = finite(features, 4)
        w = self.weights(x)
        x = (x - self.location) / self.scale
        design = self._design(x)
        prediction = np.zeros((*x.shape[:3], 3))
        for component in range(w.shape[-1]):
            mean = np.einsum("ntvd,vdc->ntvc", design, self.coef[component])
            prediction += w[..., component, None, None] * mean
        if not np.isfinite(prediction).all():
            raise FloatingPointError("Non-finite residual prediction")
        return prediction


def self_test() -> dict:
    """Controlled mechanism checks; not evidence of real-data improvement."""
    rng = np.random.default_rng(149)
    x = rng.normal(size=(40, 40, 2, 3))
    x[..., 0] += np.where(np.arange(40)[:, None, None] % 2 == 0, -3, 3)
    sign = np.where(x[..., :1] < 0, -1, 1)
    y = np.broadcast_to(0.008 * sign, (*x.shape[:3], 3)).copy()
    y += 0.0005 * rng.normal(size=y.shape)
    train, test = np.arange(28), np.arange(28, 40)
    result = {}
    for spec in (Spec("ridge"), Spec("finite", 2), Spec("dp", 8)):
        model = ResidualExperts(
            spec, feature_rank=6, response_rank=2, gate_stride=2
        ).fit(x[train], y[train])
        pred = model.predict(x[test])
        assert np.allclose(model.weights(x[test]).sum(-1), 1)
        assert np.array_equal(pred, model.predict(x[test]))
        result[spec.kind] = {
            "mae_m": float(np.mean(np.abs(pred - y[test]))),
            "active_components": model.active_components,
        }
    assert result["dp"]["mae_m"] < result["ridge"]["mae_m"]
    z = np.zeros((2, 1))
    mu = np.array([[0.0, -100.0], [0.0, 100.0]])
    cov = np.broadcast_to(np.eye(2), (2, 2, 2)).copy()
    assert np.allclose(marginal_weights(z, mu, cov, np.array([0.3, 0.7])), [0.3, 0.7])
    try:
        finite(np.array([np.nan]), 1)
    except ValueError:
        pass
    else:
        raise AssertionError("Non-finite input accepted")
    return {
        "status": "passed",
        "synthetic_only": True,
        "metrics": result,
        "checks": [
            "finite_input",
            "deterministic_prediction",
            "normalized_gate",
            "unknown_response_marginalization",
            "observable_regime_positive_control",
        ],
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_test(), indent=2))
