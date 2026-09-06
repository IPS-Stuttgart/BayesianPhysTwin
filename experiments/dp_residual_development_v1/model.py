"""Conditional variational Gaussian-mixture regression diagnostic.

This is a finite-truncated DP pilot, NOT a sticky HDP state-space estimator.
Mixture parameters are variational point summaries; the predictive Gaussian
mixture retains regime ambiguity but does not integrate all parameter uncertainty.
A single component covers an entire residual field, not one independent node.
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp, ndtr
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture
from sklearn.exceptions import ConvergenceWarning


@dataclass(frozen=True)
class Specification:
    family: str = 'dp'
    components: int = 8
    concentration: float = 1.0
    context_rank: int = 6
    seed: int = 42
    regularization: float = 0.01
    max_iter: int = 300

    @property
    def name(self) -> str:
        return f'{self.family}-k{self.components}-a{self.concentration:g}-s{self.seed}'


def finite(values: np.ndarray, ndim: int) -> np.ndarray:
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != ndim or a.size == 0 or not np.isfinite(a).all():
        raise ValueError('Input must be a nonempty finite array with the required rank')
    return a


class ConditionalMixture:
    """Joint p(context, residual field), conditioned on outcome-free context.

    Context whitening and response scales are fitted on fitting data only.
    The declared positive covariance regularization is part of this diagnostic,
    not an implicit repair of a frozen scientific path.
    """

    def __init__(self, specification: Specification):
        self.specification = specification

    def fit(self, context: np.ndarray, residual: np.ndarray) -> 'ConditionalMixture':
        x, y = finite(context, 2), finite(residual, 2)
        if len(x) != len(y):
            raise ValueError('Context and residual sample counts differ')
        s = self.specification
        if s.family not in ('single', 'finite', 'dp') or s.components < 1:
            raise ValueError('Invalid family or component cap')
        self.x_location = x.mean(0)
        scale = x.std(0)
        self.x_scale = np.where(scale > 1e-10, scale, 1.0)
        rank = min(s.context_rank, x.shape[1], len(x) - 1)
        self.pca = PCA(n_components=rank, svd_solver='full', whiten=False)
        projected = self.pca.fit_transform((x - self.x_location) / self.x_scale)
        # Avoid whitening rank-deficient directions; zero modes remain zero.
        self.pc_scale = np.where(projected.std(0) > 1e-10, projected.std(0), 1.0)
        projected /= self.pc_scale
        self.y_location = y.mean(0)
        self.y_scale = np.maximum(y.std(0), 0.001)  # one mm, explicit scale floor
        yn = (y - self.y_location) / self.y_scale
        joint = np.concatenate([projected, yn], axis=1)
        self.dx, self.dy = projected.shape[1], y.shape[1]
        prior = 'dirichlet_process' if s.family == 'dp' else 'dirichlet_distribution'
        count = 1 if s.family == 'single' else s.components
        # Match total finite-Dirichlet mass to DP concentration.
        concentration = s.concentration if s.family == 'dp' else s.concentration / count
        self.mixture = BayesianGaussianMixture(
            n_components=count, covariance_type='full',
            weight_concentration_prior_type=prior,
            weight_concentration_prior=concentration,
            reg_covar=s.regularization, mean_precision_prior=0.1,
            max_iter=s.max_iter, tol=1e-4, n_init=1,
            init_params='kmeans', random_state=s.seed,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            self.mixture.fit(joint)
        self.fit_warnings = [str(w.message) for w in caught]
        self.context_cholesky, self.regressions, self.conditionals = [], [], []
        for covariance in self.mixture.covariances_:
            xx = covariance[:self.dx, :self.dx]
            xy = covariance[:self.dx, self.dx:]
            ch = cho_factor(xx, lower=True, check_finite=True)
            beta = cho_solve(ch, xy)
            conditional = covariance[self.dx:, self.dx:] - xy.T @ beta
            conditional = (conditional + conditional.T) / 2
            # No clipping or hidden jitter: invalid covariance fails.
            np.linalg.cholesky(conditional)
            self.context_cholesky.append(ch)
            self.regressions.append(beta)
            self.conditionals.append(conditional * self.y_scale[:, None] * self.y_scale[None, :])
        self.conditionals = np.asarray(self.conditionals)
        self.residual_cholesky = [cho_factor(c, lower=True) for c in self.conditionals]
        return self

    def components(self, context: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = finite(context, 2)
        projected = self.pca.transform((x - self.x_location) / self.x_scale) / self.pc_scale
        logs, means = [], []
        for k, (ch, beta) in enumerate(zip(self.context_cholesky, self.regressions)):
            diff = projected - self.mixture.means_[k, :self.dx]
            quadratic = np.einsum('ni,ni->n', diff, cho_solve(ch, diff.T).T)
            logdet = 2 * np.log(np.diag(ch[0])).sum()
            logs.append(np.log(self.mixture.weights_[k]) - .5 * (self.dx * np.log(2*np.pi) + logdet + quadratic))
            normalized = self.mixture.means_[k, self.dx:] + diff @ beta
            means.append(self.y_location + normalized * self.y_scale)
        logw = np.stack(logs, axis=1)
        weights = np.exp(logw - logsumexp(logw, axis=1, keepdims=True))
        return weights, np.stack(means, axis=1)

    def predict(self, context: np.ndarray, *, hard: bool = False) -> np.ndarray:
        weights, means = self.components(context)
        if hard:
            return means[np.arange(len(weights)), weights.argmax(1)]
        return np.einsum('nk,nkd->nd', weights, means)

    def logpdf(self, context: np.ndarray, residual: np.ndarray, *, hard: bool = False) -> np.ndarray:
        weights, means = self.components(context)
        y = finite(residual, 2)
        log_terms = []
        for k, ch in enumerate(self.residual_cholesky):
            diff = y - means[:, k]
            quadratic = np.einsum('ni,ni->n', diff, cho_solve(ch, diff.T).T)
            logdet = 2 * np.log(np.diag(ch[0])).sum()
            log_terms.append(-.5 * (self.dy * np.log(2*np.pi) + logdet + quadratic))
        logs = np.stack(log_terms, axis=1)
        if hard:
            return logs[np.arange(len(y)), weights.argmax(1)]
        with np.errstate(divide='ignore'):
            return logsumexp(np.log(weights) + logs, axis=1)

    def intervals(self, context: np.ndarray, mass: float = .9) -> tuple[np.ndarray, np.ndarray]:
        if not 0 < mass < 1:
            raise ValueError('Mass must be between zero and one')
        weights, means = self.components(context)
        sd = np.sqrt(np.diagonal(self.conditionals, axis1=1, axis2=2))
        # Mixture quantiles, not moment-matched Gaussian intervals.
        lo0 = np.min(means - 12 * sd[None], axis=1)
        hi0 = np.max(means + 12 * sd[None], axis=1)
        result = []
        for probability in ((1-mass)/2, (1+mass)/2):
            lo, hi = lo0.copy(), hi0.copy()
            for _ in range(45):
                mid = (lo + hi) / 2
                cdf = np.sum(weights[:, :, None] * ndtr((mid[:, None] - means) / sd[None]), axis=1)
                below = cdf < probability
                lo = np.where(below, mid, lo)
                hi = np.where(below, hi, mid)
            result.append((lo + hi) / 2)
        return result[0], result[1]

    def metadata(self) -> dict:
        return {
            'specification': self.specification.__dict__,
            'converged': bool(self.mixture.converged_),
            'iterations': int(self.mixture.n_iter_),
            'weights': self.mixture.weights_.tolist(),
            'active_weight_gt_001': int(np.sum(self.mixture.weights_ > .01)),
            'tail_component_weight': float(self.mixture.weights_[-1]),
            'warnings': self.fit_warnings,
            'predictive_approximation': 'conditional-Gaussian-variational-parameter-summary',
            'temporal_likelihood': False,
            'correlated_field_dimension': self.dy,
        }
