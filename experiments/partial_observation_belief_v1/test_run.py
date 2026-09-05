"""Synthetic software checks, not empirical evidence."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('partial_test', Path(__file__).with_name('run.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_masks_fixed_and_budgets():
    assert len(m.masks()) == 9
    for name, ids in m.masks().items():
        assert len(ids) == int(name.rsplit('_', 1)[1])
        assert len(set(ids)) == len(ids)
        assert np.all((0 <= ids) & (ids < 8))


def test_covariance_and_marginal_parity():
    e = np.random.default_rng(1).normal(size=(150, 72))
    c = m.covariance(e, rank=4)
    assert np.linalg.eigvalsh(c).min() > 0
    np.testing.assert_allclose(np.diag(c), np.diag(e.T@e/len(e)), atol=1e-9)
    signs = np.random.default_rng(2).choice([-1, 1], 72)
    np.testing.assert_array_equal(np.diag(c), np.diag(c*signs[:, None]*signs[None]))


def test_map_equivalence_and_variance_reduction():
    e = np.random.default_rng(4).normal(size=(100, 72))*.01
    c = m.covariance(e, rank=8)
    k, v = m.conditional(c, [1, 4], .002)
    np.testing.assert_allclose(k, m.map_gain(c, [1, 4], .002), atol=1e-8)
    assert np.all(v <= np.diag(c)+1e-12)


def test_shared_information_positive_control():
    c = np.ones((72, 72))*.0001+np.eye(72)*1e-6
    update = m.gain(c, [0], .001)@np.ones(3)*.01
    assert .009 < update[9] < .011
    diagonal_update = m.gain(np.diag(np.diag(c)), [0], .001)@np.ones(3)*.01
    assert diagonal_update[9] == 0


def test_future_hidden_noninterference():
    p = {'gap': 8, 'offsets': [0, 4, 16]}
    a = np.random.default_rng(1).normal(size=(500, 12, 3))*.1
    feature, _, times = m.observation(a, 1, p)
    dim = len(feature)
    model = (np.zeros(dim), np.ones(dim), np.zeros(72), np.zeros((dim, 72)))
    names = m.masks()
    matrices, variances, controls = {}, {}, {}
    for name, ids in names.items():
        matrices[name] = {arm: m.gain(np.eye(72), ids, .002)
                          for arm in ('prior', 'joint_lowrank')}
        variances[name] = {arm: np.ones(72) for arm in matrices[name]}
        controls[name] = ((np.zeros(dim+3*len(ids)), np.ones(dim+3*len(ids)),
                           np.zeros(72), np.zeros((dim+3*len(ids), 72))), .5)
    models = model, np.zeros(72), matrices, variances, controls
    for name, ids in names.items():
        b = a.copy()
        hidden = np.setdiff1d(np.arange(2, 10), ids+2)
        b[times[0], hidden] += 100
        for time in times[1:]:
            b[time, 2:10] -= 100
        first = m.case_predictions(a, 1, name, models, p)[0]
        second = m.case_predictions(b, 1, name, models, p)[0]
        for arm in first:
            np.testing.assert_array_equal(first[arm], second[arm])


def test_ridge_shapes_and_constant_features():
    rng = np.random.default_rng(9)
    x = np.c_[rng.normal(size=(50, 4)), np.ones(50)]
    y = x[:, :2]
    result = m.predict(m.fit_ridge(x, y, .1), x)
    assert result.shape == (50, 2)
    assert np.isfinite(result).all()
    assert np.mean((result-y)**2) < .001


def test_diagonal_cannot_complete_hidden_region():
    ids = [0, 4]
    k = m.gain(np.eye(72), ids, .01)
    hidden = m.coords(np.setdiff1d(np.arange(8), ids))
    np.testing.assert_array_equal(k[hidden], 0)
    np.testing.assert_array_equal(k[24:], 0)


def test_loader_rejects_wrong_shape(tmp_path):
    import pickle
    import pytest
    path = tmp_path/'bad.pkl'
    path.write_bytes(pickle.dumps(np.zeros((5, 3, 12))))
    with pytest.raises(ValueError):
        m.read_trajectory(path)
