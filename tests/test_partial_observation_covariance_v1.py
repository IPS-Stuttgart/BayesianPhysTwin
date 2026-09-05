"""Focused mechanism/information-boundary tests; synthetic tests are not evidence."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / 'experiments/partial_observation_covariance_v1/run.py'
SPEC = importlib.util.spec_from_file_location('conditioning_experiment', PATH)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def test_gaussian_gain_and_zero_cross_covariance():
    cov = np.eye(120)
    cov[3, 0] = cov[0, 3] = .5
    model = {'cov': cov, 'noise_scale': 1.0}
    gain, variance = m.gains(model, np.array([0]), {'family': 'empirical', 'reg': 1.0})
    assert gain[3, 0] == pytest.approx(.25)
    assert variance[3] == pytest.approx(.875)
    model['cov'] = np.eye(120)
    gain, _ = m.gains(model, np.array([0]), {'family': 'empirical', 'reg': 1.0})
    assert np.array_equal(gain[3:], np.zeros_like(gain[3:]))


def test_ridge_and_gaussian_are_same_conditional_mean():
    rng = np.random.default_rng(71)
    samples = rng.normal(size=(100, 120))
    model = m.covariance_model(samples, 'empirical')
    a, _ = m.gains(model, np.array([2, 3]), {'family': 'empirical', 'reg': .01})
    b, _ = m.gains(model, np.array([2, 3]), {'family': 'ridge', 'reg': .01})
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)


def test_structured_covariance_preserves_coordinate_marginals_and_psd():
    rng = np.random.default_rng(42)
    samples = rng.normal(size=(80, 120)) * np.linspace(.1, 3, 120)
    empirical = m.covariance_model(samples, 'empirical')['cov']
    structured = m.covariance_model(samples, 'rod_modes', rank=2)['cov']
    np.testing.assert_allclose(np.diag(empirical), np.diag(structured), rtol=1e-12)
    assert np.linalg.eigvalsh(structured).min() > 0
    signs = np.tile(np.repeat([1, -1, 1, -1, -1, 1, -1, 1], 3), 5)
    scrambled = structured * np.outer(signs, signs)
    np.testing.assert_array_equal(np.diag(scrambled), np.diag(structured))
    assert np.linalg.eigvalsh(scrambled).min() > 0


def test_hidden_and_future_outcomes_cannot_change_predictions():
    rng = np.random.default_rng(7)
    model = m.covariance_model(rng.normal(size=(60, 120)), 'rod_modes')
    param = {'family': 'rod_modes', 'reg': .1}
    base = rng.normal(size=(5, 8, 3))
    nodes = np.array([2, 3])
    outcome = rng.normal(size=(5, 8, 3))
    before, _ = m.predict_readout(base, outcome[0, nodes], nodes, model, param)
    outcome[1:] = 1e20
    hidden = np.setdiff1d(np.arange(8), nodes)
    outcome[0, hidden] = -1e20
    after, _ = m.predict_readout(base, outcome[0, nodes], nodes, model, param)
    np.testing.assert_array_equal(before, after)
    changed, _ = m.predict_readout(base, outcome[0, nodes] + 1, nodes, model, param)
    assert not np.array_equal(before, changed)


def test_masks_exclude_all_observed_nodes_from_all_scoring_horizons():
    for b in m.BUDGETS:
        assert len(m.masks(b)) == 9 - b
        for nodes in m.masks(b):
            for h in range(len(m.HORIZONS)):
                hidden = m.hidden_coordinates(nodes, h) - h * 24
                assert len(hidden) == (8 - b) * 3
                assert not set(hidden) & set(m.coordinates(nodes))


def test_nonfinite_source_fails_closed():
    bad = np.zeros((2, 120))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        m.covariance_model(bad, 'empirical')


def test_model_does_not_depend_on_source_trajectory_order():
    rng = np.random.default_rng(77)
    source = rng.normal(size=(8, 18, 120))
    a = m.fit(source, {'family': 'rod_modes', 'rank': 2})
    b = m.fit(source[::-1], {'family': 'rod_modes', 'rank': 2})
    np.testing.assert_allclose(a['mean'], b['mean'], atol=1e-14)
    np.testing.assert_allclose(a['cov'], b['cov'], atol=1e-14)
