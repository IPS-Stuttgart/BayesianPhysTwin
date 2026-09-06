"""Development-only residual expert checks; optional scikit-learn dependency."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from experiments.dp_residual_development_v1.model import (
    ResidualExperts,
    Spec,
    finite,
    marginal_weights,
    self_test,
)


def test_marginal_gate_matches_independent_gaussian_reference():
    rng = np.random.default_rng(143)
    x = rng.normal(size=(9, 2))
    means = rng.normal(size=(3, 4))
    a = rng.normal(size=(3, 4, 4))
    cov = a @ a.transpose(0, 2, 1) + np.eye(4)[None]
    weights = np.array([.2, .3, .5])
    scores = np.stack([
        np.log(w) + multivariate_normal.logpdf(x, mean=m[:2], cov=c[:2, :2])
        for m, c, w in zip(means, cov, weights, strict=True)
    ], axis=1)
    expected = np.exp(scores-logsumexp(scores, axis=1, keepdims=True))
    actual = marginal_weights(x, means, cov, weights)
    np.testing.assert_allclose(actual, expected, atol=1e-13)
    means[:, 2:] += 1000
    np.testing.assert_array_equal(actual, marginal_weights(x, means, cov, weights))


def test_ridge_mean_matches_direct_normal_equations():
    rng = np.random.default_rng(147)
    x = rng.normal(size=(8, 9, 2, 4))
    y = rng.normal(size=(8, 9, 2, 3))
    model = ResidualExperts(Spec('ridge')).fit(x, y)
    pred = model.predict(x)
    for node in range(2):
        a = x[:, :, node].reshape(-1, 4)
        a = (a-a.mean(0))/a.std(0)
        design = np.column_stack((np.ones(len(a)), a))
        penalty = np.diag([0., 1., 1., 1., 1.])
        coef = np.linalg.solve(design.T@design+penalty,
                               design.T@y[:, :, node].reshape(-1, 3))
        np.testing.assert_allclose(pred[:, :, node].reshape(-1, 3), design@coef,
                                   atol=1e-13)


@pytest.mark.parametrize('kind', ['ridge', 'rff', 'finite', 'finite_bayes', 'dp'])
def test_query_batch_order_invariance(kind):
    rng = np.random.default_rng(66)
    x = rng.normal(size=(8, 12, 2, 3))
    y = rng.normal(size=(8, 12, 2, 3))*.001
    model = ResidualExperts(Spec(kind, 2), feature_rank=3,
                             response_rank=2, gate_stride=2).fit(x[:6], y[:6])
    predictions = model.predict(x[6:])
    np.testing.assert_allclose(model.predict(x[6:][::-1])[::-1], predictions, atol=1e-13)
    np.testing.assert_allclose(np.concatenate([model.predict(x[6:7]), model.predict(x[7:8])]),
                               predictions, atol=1e-13)
    np.testing.assert_allclose(model.weights(x[6:]).sum(-1), 1., atol=1e-13)


@pytest.mark.parametrize('a', [np.array([np.nan]), np.array([np.inf]), np.array([])])
def test_nonfinite_and_empty_rejected(a):
    with pytest.raises(ValueError):
        finite(a, 1)


def test_controlled_observable_regime():
    result = self_test()
    assert result['status'] == 'passed'
    assert result['synthetic_only'] is True
