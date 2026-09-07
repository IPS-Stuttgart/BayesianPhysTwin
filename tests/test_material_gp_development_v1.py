"""Algebra and interface tests; these are not empirical forecasting evidence."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/experiments/material_gp_development_v1.py"
)
SPEC = importlib.util.spec_from_file_location("gp_development", PATH)
gp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gp)


@pytest.mark.parametrize("material_length", [None, 0.4])
def test_kernel_positive_semidefinite(material_length):
    x = np.random.default_rng(3).normal(size=(30, 7))
    k = gp.kernel(x, x, 1.5, material_length)
    np.testing.assert_allclose(k, k.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(k), gp.kernel_diag(x), atol=1e-7)
    assert np.linalg.eigvalsh(k).min() >= -1e-9


def test_full_inducing_matches_exact_gp():
    rng = np.random.default_rng(7)
    x, y, q = (
        rng.normal(size=(22, 5)),
        rng.normal(size=(22, 3)),
        rng.normal(size=(5, 5)),
    )
    model = gp.SparseGP(1.5, 0.2, inducing=22).fit(x, y)
    mean, var = model.predict(q, True)
    k = gp.kernel(x, x, 1.5) + 0.2 * np.eye(len(x))
    kqx = gp.kernel(q, x, 1.5)
    expected = kqx @ np.linalg.solve(k, y)
    expected_var = (
        gp.kernel_diag(q) - np.einsum("ij,ji->i", kqx, np.linalg.solve(k, kqx.T)) + 0.2
    )
    np.testing.assert_allclose(mean, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(var, expected_var, rtol=1e-5, atol=1e-6)


def test_sparse_is_finite_positive_and_deterministic():
    rng = np.random.default_rng(9)
    x, y = rng.normal(size=(60, 4)), rng.normal(size=(60, 3))
    first = gp.SparseGP(inducing=12).fit(x, y).predict(x, True)
    second = gp.SparseGP(inducing=12).fit(x, y).predict(x, True)
    assert np.isfinite(first[0]).all()
    assert np.min(first[1]) > 0
    for a, b in zip(first, second, strict=False):
        np.testing.assert_array_equal(a, b)


def test_invalid_training_rejected():
    with pytest.raises(ValueError):
        gp.SparseGP(noise=0)
    with pytest.raises(ValueError):
        gp.SparseGP().fit(np.zeros((3, 2)), np.zeros((2, 3)))
    with pytest.raises(ValueError):
        gp.SparseGP().fit(np.full((3, 2), np.nan), np.zeros((3, 3)))


def test_material_distance_reduces_correlation():
    x = np.zeros((1, 4))
    z = x.copy()
    z[:, -1] = 1
    assert gp.kernel(x, z, 1.5, 0.4)[0, 0] < gp.kernel(x, x, 1.5, 0.4)[0, 0]
    np.testing.assert_allclose(gp.kernel(x, z, 1.5), gp.kernel(x, x, 1.5))


def test_candidate_exact_boundary_and_zero_shrinkage():
    baseline = np.random.default_rng(1).normal(size=(2, 5, 12, 3))
    correction = np.ones((2, 5, 8, 3))
    np.testing.assert_array_equal(gp.candidate(baseline, correction, 0), baseline)
    altered = gp.candidate(baseline, correction, 0.25)
    np.testing.assert_array_equal(
        altered[:, :, [0, 1, -2, -1]], baseline[:, :, [0, 1, -2, -1]]
    )
    np.testing.assert_allclose(altered[:, :, 2:-2] - baseline[:, :, 2:-2], 0.25)


@pytest.mark.parametrize("shared", [False, True])
def test_residual_prediction_does_not_refit_normalizer(shared, monkeypatch):
    monkeypatch.setattr(gp, "TOTAL_INDUCING", 16)
    rng = np.random.default_rng(2)
    x = rng.normal(size=(3, 6, 4, 5))
    y = rng.normal(size=(3, 6, 4, 3))
    model = gp.ResidualGP(shared, 1.5, 0.25).fit(x, y)
    location = model.location.copy()
    mean, var = model.predict(x + 3, variance=True)
    assert mean.shape == var.shape == y.shape
    assert np.min(var) > 0
    np.testing.assert_array_equal(location, model.location)
