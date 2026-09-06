"""Numerical and information-boundary tests, not empirical evidence."""

import copy
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiments.dp_residual_source_screen_v1.run import (
    ConditionalMixture,
    FullFeatureExperts,
    Reduction,
    load_source,
    partitions,
    score,
    validate_xy,
)


class IdentityReduction:
    def x(self, x):
        return x

    def decode(self, y):
        return y


def test_conditional_gaussian_analytic():
    estimator = SimpleNamespace(
        weights_=np.array([1.0]),
        means_=np.zeros((1, 2)),
        covariances_=np.array([[[1.0, 0.6], [0.6, 2.0]]]),
    )
    model = ConditionalMixture(IdentityReduction(), estimator, {})
    x = np.array([[-2.0], [0.0], [3.0]])
    np.testing.assert_allclose(model.predict(x), 0.6 * x)
    np.testing.assert_allclose(model.feature_weights(x), 1.0)


def test_units_are_percent_endpoint_span_not_millimetres():
    y = np.full((2, 3), 0.01)
    result = score(y, np.zeros_like(y), np.array(["a", "b"]))
    assert result["nrmse_percent_span"] == pytest.approx(1.0)
    assert result["nl1_percent_span"] == pytest.approx(1.0)
    assert "rmse_mm" not in result


def test_score_weights_trajectories_equally():
    y = np.array([[0.01], [0.01], [0.03]])
    result = score(y, np.zeros_like(y), np.array(["a", "a", "b"]))
    assert result["nrmse_percent_span"] == pytest.approx(2.0)


def test_all_folds_use_disjoint_complete_trajectories():
    names = np.repeat([f"{i}.pkl" for i in range(56)], 19)
    all_tests = []
    for _, fit, val, test in partitions(names, {}, 5):
        assert len(val) == 9
        assert not set(fit) & set(val)
        assert not set(fit) & set(test)
        assert not set(val) & set(test)
        assert set(fit) | set(val) | set(test) == set(names)
        all_tests.extend(test)
    assert len(all_tests) == len(set(all_tests)) == 56


def test_source_identity_rejects_other_bytes(tmp_path: Path):
    (tmp_path / "source_model.npz").write_bytes(b"wrong source")
    with pytest.raises(ValueError, match="SHA-256"):
        load_source(tmp_path)


def test_bad_arrays_rejected():
    with pytest.raises(ValueError):
        validate_xy(np.zeros((3, 2)), np.zeros((2, 4)))
    with pytest.raises(ValueError):
        validate_xy(np.array([[np.nan]]), np.zeros((1, 1)))


def test_held_outcomes_not_in_prediction_api():
    for cls in (ConditionalMixture, FullFeatureExperts):
        assert list(inspect.signature(cls.predict).parameters) == ["self", "x"]


def test_experts_ignore_other_query_rows_and_component_order():
    rng = np.random.default_rng(313)
    x = rng.normal(size=(120, 5))
    y = x @ rng.normal(size=(5, 9)) + 0.1 * rng.normal(size=(120, 9))
    with threadpool_limits(limits=1):
        red = Reduction.fit(x, y, rank=3)
        mixture = ConditionalMixture.fit(x, y, red, "dp", 2, 1.0)
        scaler = StandardScaler().fit(x)
        global_model = Ridge(alpha=1.0).fit(scaler.transform(x), y)
        experts = FullFeatureExperts.fit(
            mixture, x, y, scaler, global_model, 10.0
        )
        q = rng.normal(size=(6, 5))
        before = experts.predict(q)
        np.testing.assert_allclose(before[:1], experts.predict(q[:1]), atol=1e-12)
        np.testing.assert_allclose(mixture.feature_weights(q).sum(axis=1), 1.0)
        changed = copy.deepcopy(experts)
        e = changed.gate.estimator
        e.weights_ = e.weights_[::-1].copy()
        e.means_ = e.means_[::-1].copy()
        e.covariances_ = e.covariances_[::-1].copy()
        changed.deltas = changed.deltas[::-1]
        np.testing.assert_allclose(before, changed.predict(q), atol=1e-12)
        assert before.shape == (6, 9)
        assert np.isfinite(before).all()
