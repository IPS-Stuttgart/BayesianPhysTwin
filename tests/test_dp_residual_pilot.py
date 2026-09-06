"""CPU-only regression checks for the exploratory trajectory-summary DP pilot.

These checks do not open any recordings, train a physical backend, or establish
an empirical DP advantage. The optional mixture dependencies are installed by
the dedicated cached-pilot workflow rather than added to the runtime package.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def pilot():
    pytest.importorskip("scipy")
    pytest.importorskip("sklearn")
    path = Path(__file__).resolve().parents[1] / "scripts/remote/run_dp_residual_dlo1_pilot.py"
    spec = importlib.util.spec_from_file_location("_dp_pilot_regression", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def training():
    rng = np.random.default_rng(7123)
    features = rng.normal(size=(20, 12, 2, 5))
    residual = rng.normal(scale=0.003, size=(20, 12, 2, 3))
    return features, residual


def test_weighted_ridge_matches_independent_normal_equations(pilot, training):
    features, residual = training
    design = pilot.Design.fit(features)
    stats = pilot.sufficient_statistics(design, features, residual)
    weights = np.linspace(0.2, 1.0, len(features))[:, None]
    ridge = 0.37
    fits, support = pilot.fit_experts(stats, weights, ridge)
    assert not support[0]["global_expert_fallback"]
    for node in range(features.shape[2]):
        x = design.node(features, node).reshape(-1, features.shape[-1] + 1)
        y = residual[:, :, node].reshape(-1, 3)
        w = np.repeat(weights[:, 0], features.shape[1])
        penalty = np.eye(x.shape[1]) * ridge
        penalty[0, 0] = 0.0
        expected = np.linalg.solve(x.T @ (w[:, None] * x) + penalty,
                                   x.T @ (w[:, None] * y))
        np.testing.assert_allclose(fits[0][0][node], expected, rtol=1e-10, atol=1e-12)
        eigenvalues = np.linalg.eigvalsh(fits[0][1][node])
        assert np.min(eigenvalues) >= -1e-12


def test_empty_component_uses_global_expert(pilot, training):
    features, residual = training
    design = pilot.Design.fit(features)
    stats = pilot.sufficient_statistics(design, features, residual)
    weights = np.column_stack((np.ones(len(features)), np.zeros(len(features))))
    fits, support = pilot.fit_experts(stats, weights, 1.0)
    assert support[1]["global_expert_fallback"]
    assert support[1]["expected_trajectories"] == 0.0
    for global_array, fallback_array in zip(fits[0], fits[1], strict=True):
        np.testing.assert_array_equal(global_array, fallback_array)


@pytest.mark.parametrize("family", ["dp", "finite"])
def test_prediction_gate_marginalizes_residual_coordinates(pilot, training, family):
    features, residual = training
    gate = pilot.fit_gate(features, residual, family, 3, 1.0)
    expected_prior = "dirichlet_process" if family == "dp" else "dirichlet_distribution"
    assert gate.mixture.weight_concentration_prior_type == expected_prior
    weights = gate.weights(features[:3])
    assert weights.shape == (3, 3)
    assert np.isfinite(weights).all() and np.all(weights >= 0.0)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-12)
    altered = copy.deepcopy(gate)
    d = altered.context.pca.n_components_
    # Changing only the residual part must not affect the marginal context gate.
    altered.mixture.means_[:, d:] += 1e6
    altered.residual.location[:] = np.nan
    altered.train_weights[:] = np.nan
    np.testing.assert_array_equal(altered.weights(features[:3]), weights)


@pytest.mark.parametrize("shrinkage", [0.0, 0.5, 1.0])
def test_mixture_moments_include_between_expert_uncertainty(pilot, shrinkage):
    baseline = np.zeros((2, 3, 6, 3))
    corrections = np.empty((2, 2, 3, 2, 3))
    corrections[0] = 0.003
    corrections[1] = -0.002
    variances = np.full_like(corrections, 3e-6)
    weights = np.array([[0.25, 0.75], [0.6, 0.4]])
    target = baseline.copy()
    _, mean, variance = pilot.distribution_metrics(
        baseline, target, corrections, variances, weights, shrinkage)
    w = weights.T[:, :, None, None, None]
    component_mean = shrinkage * corrections
    component_variance = variances + ((1.0 - shrinkage) * corrections) ** 2 + pilot.FLOOR
    expected_mean = np.sum(w * component_mean, axis=0)
    within = np.sum(w * component_variance, axis=0)
    between = np.sum(w * (component_mean - expected_mean) ** 2, axis=0)
    np.testing.assert_allclose(mean, expected_mean, atol=1e-15)
    np.testing.assert_allclose(variance, within + between, atol=1e-15)
    if shrinkage > 0.0:
        assert np.all(variance > within)
    prediction = pilot.point_prediction(baseline, corrections, weights, shrinkage)
    np.testing.assert_allclose(prediction[:, :, 2:-2], mean, atol=1e-15)
    np.testing.assert_array_equal(prediction[:, :, [0, 1, -2, -1]],
                                  baseline[:, :, [0, 1, -2, -1]])
    if shrinkage == 0.0:
        np.testing.assert_array_equal(prediction, baseline)


def test_scoring_cannot_modify_predictions(pilot):
    baseline = np.zeros((1, 2, 6, 3))
    correction = np.full((1, 1, 2, 2, 3), 0.002)
    variance = np.full_like(correction, 1e-6)
    weights = np.ones((1, 1))
    before = pilot.point_prediction(baseline, correction, weights, 0.5)
    pilot.distribution_metrics(baseline, np.ones_like(baseline), correction,
                               variance, weights, 0.5)
    after = pilot.point_prediction(baseline, correction, weights, 0.5)
    np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(baseline, np.zeros_like(baseline))


def test_model_serialization_is_pickle_free(pilot, training, tmp_path):
    features, residual = training
    design = pilot.Design.fit(features)
    stats = pilot.sufficient_statistics(design, features, residual)
    fits, support = pilot.fit_experts(stats, np.ones((len(features), 1)), 1.0)
    path = tmp_path / "model.npz"
    pilot.save_fitted(path, design, fits, None, support)
    with np.load(path, allow_pickle=False) as archive:
        for key in archive.files:
            assert archive[key].dtype.kind != "O"
        np.testing.assert_array_equal(archive["beta_0"], fits[0][0])
