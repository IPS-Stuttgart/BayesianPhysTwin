"""Numerical and information-order checks; synthetic fixtures are not evidence."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('heldout_uncertainty', Path(__file__).with_name('run.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_mathematical_invariants():
    m.self_test()


def test_empirical_crps_matches_pairwise_definition():
    rng = np.random.default_rng(82)
    samples = rng.normal(size=(9, 7))
    target = rng.normal(size=9)
    direct = np.abs(samples - target[:, None]).mean(axis=1) - 0.5 * np.abs(samples[:, :, None] - samples[:, None, :]).mean(axis=(1, 2))
    np.testing.assert_allclose(m.crps({'kind': 'empirical', 'samples': samples}, target), direct, atol=1e-14)


def test_empirical_strict_event_boundary():
    d = {'kind': 'empirical', 'samples': np.array([[-1., 0., 1.]])}
    # Values exactly at +/- threshold are NOT exceedances.
    p = 1-m.cdf(d, np.array([[1.]])) + m.cdf(d, np.array([[-1.]]), left=True)
    np.testing.assert_array_equal(p, [[0.]])


def test_source_test_outcome_mutation_cannot_change_source_model():
    rng = np.random.default_rng(817)
    q, meta = m.query_bank()
    qcal = q[[v['family'] == 'centroid' for v in meta]]
    f, c = rng.normal(size=(39, 96)), rng.normal(size=(9, 96))
    a = m.model(f, c)
    sa, cva = m.calibrate(f, c, qcal)
    # Neither routine receives source-test arrays or target query observations.
    arbitrary_test_outcomes = rng.normal(size=(8, 96)) * 1e9
    assert np.isfinite(arbitrary_test_outcomes).all()
    b = m.model(f, c)
    sb, cvb = m.calibrate(f, c, qcal)
    np.testing.assert_array_equal(a['covariance'], b['covariance'])
    assert sa == sb and cva == cvb


def test_positive_controls_do_not_change_means():
    rng = np.random.default_rng(184)
    f, c = rng.normal(size=(39, 96)), rng.normal(size=(9, 96))
    model = m.model(f, c)
    q, _ = m.query_bank()
    for method in m.METHODS:
        for scale in (0.25, 1., 4.):
            d = m.distribution(model, method, q, scale)
            np.testing.assert_allclose(d['samples'].mean(axis=1), 0, atol=1e-12)


def test_joint_model_preserves_empirical_complete_trajectory_dependence():
    rng = np.random.default_rng(811)
    common = rng.normal(size=(9, 1))
    e = np.repeat(common, 96, axis=1)
    f = np.zeros((39, 96))
    model = m.model(f, e)
    q = np.zeros((2, 96)); q[0, :2] = (0.5, 0.5); q[1, :2] = (1., -1.)
    joint = m.distribution(model, 'joint_student', q, 1)
    diagonal = m.distribution(model, 'diagonal_student', q, 1)
    assert joint['sd'][0] > diagonal['sd'][0]
    assert joint['sd'][1] < diagonal['sd'][1]
    empirical = m.distribution(model, 'trajectory_bootstrap', q, 1)
    np.testing.assert_allclose(empirical['samples'][1], 0, atol=1e-14)
