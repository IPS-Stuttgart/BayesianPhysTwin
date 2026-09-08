"""Mechanism and leakage tests; synthetic fixtures are not physical evidence."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).with_name("run.py")
SPEC = importlib.util.spec_from_file_location("passive_noise", PATH)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def sample_model():
    x = np.random.default_rng(1).normal(size=(160, 72)) * 0.01
    return m.fit(x)


def test_source_target_masks_are_disjoint_and_complete():
    for b, expected in ((2, 28), (4, 70)):
        source = set(map(tuple, m.masks(b, True)))
        target = set(map(tuple, m.masks(b, False)))
        assert not source & target
        assert len(source | target) == expected
        assert len(source) == 9 - b


def test_supplied_noise_preserves_shared_mode():
    actual, supplied = m.noise("shared10", 2)
    assert np.array_equal(actual, supplied)
    assert actual[0, 3] == pytest.approx(0.01**2)
    assert actual[0, 0] == pytest.approx(0.01**2 + 0.002**2)
    assert np.linalg.eigvalsh(actual).min() > 0


def test_miscalibration_does_not_change_actual_observations():
    actual, _ = m.noise("shared10", 2)
    a, r = m.noise("shared10_Rquarter", 2)
    b, s = m.noise("shared10_Rfour", 2)
    np.testing.assert_array_equal(a, actual)
    np.testing.assert_array_equal(b, actual)
    np.testing.assert_allclose(r, 0.25 * actual)
    np.testing.assert_allclose(s, 4 * actual)


def test_independent_noise_augmented_ridge_equivalence():
    model = sample_model()
    nodes = np.array([1, 4])
    _, r = m.noise("shared10", 2)
    a, v = m.gain(model, nodes, "adaptive_empirical", 0.01, r)
    b, w = m.gain(model, nodes, "direct_noise_aware_ridge", 0.01, r)
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(v, w, rtol=1e-12, atol=1e-12)


def test_noise_adaptation_changes_gain_and_large_noise_returns_prior():
    model = sample_model()
    nodes = np.array([0, 7])
    a, v = m.gain(model, nodes, "adaptive_empirical", 0.01, np.zeros((6, 6)))
    b, w = m.gain(model, nodes, "adaptive_empirical", 0.01, np.eye(6) * 1e6)
    assert np.linalg.norm(b) < 1e-7 * np.linalg.norm(a)
    assert np.all(w >= v - 1e-14)


def test_fixed_gain_is_constant_for_noise_inputs():
    model = sample_model()
    nodes = np.array([1, 6])
    for arm in ("fixed_ridge", "pooled_noise_ridge"):
        a, _ = m.gain(model, nodes, arm, 0.1, np.zeros((6, 6)))
        b, _ = m.gain(model, nodes, arm, 0.1, np.eye(6))
        np.testing.assert_array_equal(a, b)


def test_exact_expected_noise_risk_against_monte_carlo():
    rng = np.random.default_rng(83)
    g = rng.normal(size=(4, 6))
    r, _ = m.noise("shared10", 2)
    bias = rng.normal(size=4) * 0.01
    exact = m.expected_error2(bias, np.zeros(4), g, r)
    eps = rng.multivariate_normal(np.zeros(6), r, size=150000)
    simulation = np.mean((bias + eps @ g.T) ** 2, axis=0)
    np.testing.assert_allclose(exact, simulation, rtol=0.02)


def test_no_hidden_or_future_outcome_argument():
    rng = np.random.default_rng(10)
    model = sample_model()
    nodes = np.array([2, 7])
    g, _ = m.gain(model, nodes, "adaptive_empirical", 0.1, np.zeros((6, 6)))
    truth = rng.normal(size=(5, 72))
    before = m.predict(model, g, truth[:, m.coords(nodes)].copy(), nodes)
    truth[:, m.hidden(nodes)] = 1e10
    truth[:, 24:] = -1e10
    after = m.predict(model, g, truth[:, m.coords(nodes)].copy(), nodes)
    np.testing.assert_array_equal(before, after)


def test_rod_covariance_same_marginals_psd():
    source = np.random.default_rng(11).normal(size=(8, 18, 72))
    empirical = m.fit(source)
    rod = m.fit(source, 2)
    np.testing.assert_allclose(
        np.diag(empirical["cov"]), np.diag(rod["cov"]), atol=1e-12
    )
    assert np.linalg.eigvalsh(rod["cov"]).min() > 0


def test_invalid_observations_and_covariance_rejected():
    model = sample_model()
    nodes = np.array([0, 1])
    for r in (np.eye(6) * -1, np.full((6, 6), np.nan), np.eye(3)):
        with pytest.raises(ValueError):
            m.gain(model, nodes, "adaptive_empirical", 0.1, r)
    with pytest.raises(ValueError):
        m.coords(np.array([1.2, 3.4]))
    with pytest.raises(ValueError):
        m.coords(np.array([2, 2]))
    with pytest.raises(ValueError):
        m.fit(np.full((8, 18, 72), np.nan))


def test_calibration_uses_only_source_and_complete_folds():
    rng = np.random.default_rng(12)
    source = rng.normal(size=(4, 3, 72)) * 0.01
    original = source.copy()
    result = m.source_cv(source, 2, "adaptive_empirical", {"reg": 0.1}, True)
    assert len(result["static_variance"]) == 72
    assert len(result["posterior_scale_by_horizon"]) == 3
    assert np.all(np.array(result["static_variance"]) > 0)
    np.testing.assert_array_equal(source, original)
