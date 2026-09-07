"""Numerical and information-boundary tests; synthetic fixtures are not evidence."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import run as m


def test_gaussian_conditioning():
    model = dict(
        nx=1,
        weights=np.ones(1),
        means=np.array([[2.0, 3.0]]),
        covariances=np.array([[[4.0, 2.0], [2.0, 5.0]]]),
    )
    y, w = m.conditional(model, np.array([[0.0], [2.0], [4.0]]))
    np.testing.assert_allclose(y[:, 0], [2.0, 3.0, 4.0])
    np.testing.assert_allclose(w, 1.0)


def test_mixture_marginal_gate_and_permutation():
    model = dict(
        nx=1,
        weights=np.array([0.4, 0.6]),
        means=np.array([[-3.0, -5.0], [3.0, 5.0]]),
        covariances=np.tile(np.eye(2), (2, 1, 1)),
    )
    x = np.array([[-4.0], [0.0], [4.0]])
    y, weights = m.conditional(model, x)
    np.testing.assert_allclose(weights.sum(1), 1.0)
    assert y[0, 0] < -4.9 and y[-1, 0] > 4.9
    permuted = {
        **model,
        **{k: model[k][::-1] for k in ("weights", "means", "covariances")},
    }
    np.testing.assert_allclose(m.conditional(permuted, x)[0], y)


def test_pca_roundtrip_and_reject_invalid():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(30, 4))
    transform = m.compress_fit(a, 4)
    np.testing.assert_allclose(
        m.decode(transform, m.project(transform, a)), a, atol=1e-12
    )
    with pytest.raises(ValueError):
        m.compress_fit(np.array([[np.nan], [0.0]]), 1)


def test_constant_representation():
    a = np.ones((20, 4))
    transform = m.compress_fit(a, 6)
    np.testing.assert_array_equal(m.decode(transform, m.project(transform, a)), a)


def test_exact_fallback_does_not_evaluate_model():
    base = np.array([[np.pi, -0.0, 1.0]])
    selected = {"shrinkage": 0.0, "model": None}
    pred = m.predict_family(selected, None, None, base)
    assert pred.tobytes() == base.tobytes()
    assert pred is not base


def test_future_internal_exclusion():
    rng = np.random.default_rng(4)
    original = rng.normal(size=(500, 12, 3))
    for origin in m.CONFIG["origins"]:
        changed = original.copy()
        changed[origin + 1 :, 2:10] += 1e8
        for horizon in m.CONFIG["horizons"]:
            for actual, expected in zip(
                m.ref.inputs(changed, origin, horizon),
                m.ref.inputs(original, origin, horizon),
                strict=True,
            ):
                np.testing.assert_array_equal(actual, expected)


def test_bootstrap_uses_complete_trajectories():
    result = m.bootstrap_delta(np.zeros(24), np.full(24, 0.001))
    np.testing.assert_allclose(result["ci95_mm"], [-1.0, -1.0])
    assert result["wins"] == 24
    with pytest.raises(ValueError):
        m.bootstrap_delta(np.zeros(360), np.ones(360))


def test_dp_positive_control():
    rng = np.random.default_rng(5)
    z = rng.integers(0, 2, 240) * 2 - 1
    x = (2 * z + rng.normal(0, 0.35, len(z)))[:, None]
    y = (3 * z + rng.normal(0, 0.15, len(z)))[:, None]
    model = m.fit_mixture(x, y, "dp", 4, 1.0)
    pred, weights = m.conditional(model, np.array([[-2.0], [2.0]]))
    assert pred[0, 0] < -2.5 and pred[1, 0] > 2.5
    assert model["prior"] == "dirichlet_process"
    np.testing.assert_allclose(weights.sum(1), 1.0)


def test_finite_uses_same_expert_family():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(70, 2))
    y = x[:, :1] + rng.normal(0, 0.2, size=(70, 1))
    model = m.fit_mixture(x, y, "finite", 2, 1.0)
    assert model["prior"] == "dirichlet_distribution"
    pred, weights = m.conditional(model, x[:5])
    assert pred.shape == (5, 1) and weights.shape == (5, 2)


def test_target_free_prediction_interface():
    import inspect

    assert list(inspect.signature(m.conditional).parameters) == ["model", "x"]
    assert list(inspect.signature(m.predict_family).parameters) == [
        "selected",
        "transform",
        "x",
        "baseline",
    ]
