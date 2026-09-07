"""Input-conditioned, partially pooled residual experts for an exploratory pilot.

The DP gate is sklearn's truncated variational DP Gaussian mixture, not an HDP
or a claim to recover physical modes. One gate is shared by all material nodes.
Only training causal features enter the gate; target residuals fit the experts.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class GateSpec:
    family: str
    components: int
    concentration: float = 1.0

    @property
    def key(self) -> str:
        return f"{self.family}-k{self.components}-a{self.concentration:g}"


def finite(values: np.ndarray, ndim: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != ndim or not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite {ndim}-D array")
    return array


class ResidualExperts:
    """Shared kinematic gate and node-specific ridge experts shrunk to one model."""

    def __init__(self, spec: GateSpec, ridge: float, seed: int = 7):
        if spec.family not in {"single", "finite", "finite_bayes", "dp"}:
            raise ValueError("Unknown gate family")
        if spec.components < 1 or ridge <= 0 or not np.isfinite(ridge):
            raise ValueError("Invalid component count or ridge")
        if spec.concentration <= 0 or not np.isfinite(spec.concentration):
            raise ValueError("Concentration must be positive")
        if spec.family == "single" and spec.components != 1:
            raise ValueError("Single expert must have K=1")
        self.spec, self.ridge, self.seed = spec, float(ridge), int(seed)
        self.fitted = False

    def _gate_features(self, features: np.ndarray, fit: bool) -> np.ndarray:
        # All output nodes share the same regime, avoiding independent vertex labels.
        raw = features.mean(axis=2)
        n, t, d = raw.shape
        if fit:
            # Equal number of separated kinematic summaries per trajectory.
            ticks = np.unique(np.linspace(0, t - 1, min(16, t), dtype=int))
            subsample = raw[:, ticks].reshape(-1, d)
            self.gate_scaler = StandardScaler().fit(subsample)
            standardized = self.gate_scaler.transform(subsample)
            rank = min(8, d, len(subsample) - 1)
            self.gate_pca = PCA(n_components=rank, svd_solver="full").fit(standardized)
            self.gate_ticks = ticks
        reduced = self.gate_pca.transform(
            self.gate_scaler.transform(raw.reshape(-1, d))
        )
        return reduced.reshape(n, t, -1)

    def fit(self, features: np.ndarray, residual: np.ndarray) -> ResidualExperts:
        x = finite(features, 4, "features")
        y = finite(residual, 4, "residual")
        if x.shape[:3] != y.shape[:3] or y.shape[-1] != 3:
            raise ValueError("Feature/residual shapes differ")
        self.nodes, self.dim = x.shape[2:]
        self.location = x.mean(axis=(0, 1))
        self.scale = x.std(axis=(0, 1))
        self.scale = np.where(self.scale > 1e-10, self.scale, 1.0)
        self.gate = None
        self.convergence_warnings = []
        k = self.spec.components
        if k == 1:
            responsibilities = np.ones((*x.shape[:2], 1))
            self.weights = np.ones(1)
            self.converged = True
        else:
            z = self._gate_features(x, fit=True)
            train_z = z[:, self.gate_ticks].reshape(-1, z.shape[-1])
            common = dict(
                n_components=k,
                covariance_type="full",
                reg_covar=1e-4,
                max_iter=500,
                n_init=3,
                tol=1e-4,
                random_state=self.seed,
            )
            if self.spec.family == "finite":
                self.gate = GaussianMixture(**common)
            else:
                prior_type = (
                    "dirichlet_process"
                    if self.spec.family == "dp"
                    else "dirichlet_distribution"
                )
                concentration = (
                    self.spec.concentration
                    if self.spec.family == "dp"
                    else self.spec.concentration / k
                )
                self.gate = BayesianGaussianMixture(
                    **common,
                    weight_concentration_prior_type=prior_type,
                    weight_concentration_prior=concentration,
                )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                self.gate.fit(train_z)
                self.convergence_warnings = [str(w.message) for w in caught]
            self.weights = self.gate.weights_.copy()
            self.converged = bool(self.gate.converged_)
            responsibilities = self.gate.predict_proba(z.reshape(-1, z.shape[-1]))
            responsibilities = responsibilities.reshape(*x.shape[:2], k)
        self.coefficients = np.zeros((k, self.nodes, self.dim + 1, 3))
        self.global_coefficients = np.zeros((self.nodes, self.dim + 1, 3))
        penalty = self.ridge * np.eye(self.dim + 1)
        global_penalty = penalty.copy()
        global_penalty[0, 0] = 0.0
        for node in range(self.nodes):
            design = self._design(x, node)
            response = y[:, :, node].reshape(-1, 3)
            global_coef = np.linalg.solve(
                design.T @ design + global_penalty, design.T @ response
            )
            self.global_coefficients[node] = global_coef
            remaining = response - design @ global_coef
            for component in range(k):
                if k == 1:
                    coefficient = global_coef
                else:
                    weights = responsibilities[:, :, component].reshape(-1)
                    weighted = design * weights[:, None]
                    delta = np.linalg.solve(
                        design.T @ weighted + penalty, weighted.T @ remaining
                    )
                    coefficient = global_coef + delta
                self.coefficients[component, node] = coefficient
        self.fitted = True
        return self

    def _design(self, x: np.ndarray, node: int) -> np.ndarray:
        standardized = (x[:, :, node] - self.location[node]) / self.scale[node]
        flat = standardized.reshape(-1, self.dim)
        return np.column_stack((np.ones(len(flat)), flat))

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Model has not been fitted")
        x = finite(features, 4, "features")
        if x.shape[2:] != (self.nodes, self.dim):
            raise ValueError("Query feature dimensions changed")
        if self.gate is None:
            return np.ones((*x.shape[:2], 1))
        z = self._gate_features(x, fit=False)
        return self.gate.predict_proba(z.reshape(-1, z.shape[-1])).reshape(
            *x.shape[:2], self.spec.components
        )

    def predict(self, features: np.ndarray, *, hard: bool = False) -> np.ndarray:
        x = finite(features, 4, "features")
        probabilities = self.probabilities(x)
        if hard:
            probabilities = np.eye(self.spec.components)[probabilities.argmax(axis=-1)]
        result = np.zeros((*x.shape[:3], 3))
        for node in range(self.nodes):
            design = self._design(x, node)
            component_mean = np.einsum(
                "nd,kdc->nkc", design, self.coefficients[:, node], optimize=True
            )
            mean = np.einsum(
                "nk,nkc->nc",
                probabilities.reshape(-1, self.spec.components),
                component_mean,
                optimize=True,
            )
            result[:, :, node] = mean.reshape(*x.shape[:2], 3)
        if not np.isfinite(result).all():
            raise RuntimeError("Nonfinite prediction")
        return result

    def diagnostics(self) -> dict:
        return {
            "family": self.spec.family,
            "max_components": self.spec.components,
            "concentration": self.spec.concentration,
            "seed": self.seed,
            "ridge": self.ridge,
            "weights": self.weights.tolist(),
            "active_components_above_001": int(np.sum(self.weights > 0.01)),
            "converged": self.converged,
            "convergence_warnings": self.convergence_warnings,
        }
