"""Numerical, custody and exact-fallback tests; no recorded outcomes here."""

import inspect

import numpy as np
import pytest

from .conditioning import (
    condition_block,
    fit_models,
    forecast_pair,
    prefix_indices,
    world_forecast,
)
from .controls import (
    apply_probability_calibration,
    context_features,
    fit_probability_calibration,
    regret_predictions,
)
from .replication import authorize_source, stratified_bootstrap
from .risk import TemporalBlocks, deploy, forecast_risk, ranked_acceptance
from .study import METHODS, folds, source_gate


def fixture():
    rng = np.random.default_rng(23)
    features = rng.normal(size=(4, 498, 2, 3))
    truth = rng.normal(size=(4, 498, 2, 3)) * 0.01
    models = fit_models(features, truth, np.arange(4))
    parameters = [{"a": 0.5, "b": 0.5, "normalizer": 1, "q_m2": [1e-4] * 3}] * 2
    return features, truth, models, parameters


def test_future_blind_interface_and_suffix_mutation():
    f, target, models, parameters = fixture()
    obs = prefix_indices(50)
    original = forecast_pair(models, parameters, f[0], target[0, obs], 50, 25)
    target[0, 50:] = 1e6
    again = forecast_pair(models, parameters, f[0], target[0, obs], 50, 25)
    assert np.array_equal(original[0], again[0])
    assert np.array_equal(original[1], again[1])
    assert np.array_equal(original[3].blocks, again[3].blocks)
    assert "truth" not in inspect.signature(forecast_risk).parameters
    assert "target" not in inspect.signature(forecast_risk).parameters


@pytest.mark.parametrize("endpoint", [50, 150, 250])
def test_exact_prefix(endpoint):
    obs = prefix_indices(endpoint)
    assert len(obs) == 10
    assert obs[0] == endpoint - 46
    assert obs[-1] == endpoint - 1
    assert np.array_equal(np.diff(obs), np.full(9, 5))


def test_schur_complement_full_temporal_covariance():
    k = np.array([[2.0, 0.3, 0.5], [0.3, 1.0, 0.4], [0.5, 0.4, 1.5]])
    delta, covariance = condition_block(k, np.array([0.2]), 1)
    np.testing.assert_allclose(delta, k[1:, 0] * 0.1)
    np.testing.assert_allclose(covariance, k[1:, 1:] - np.outer(k[1:, 0], k[0, 1:]) / 2)
    assert covariance[0, 1] != 0


def test_analytic_regret_against_monte_carlo():
    rng = np.random.default_rng(31)
    b = np.array([0.0, 0.0]).reshape(2, 1, 1)
    c = np.array([0.002, -0.003]).reshape(2, 1, 1)
    mu = np.array([0.001, -0.001]).reshape(2, 1, 1)
    k = np.array([[4e-6, 1e-6], [1e-6, 9e-6]])
    result = forecast_risk(b, c, mu, TemporalBlocks(k[None, None]))
    y = rng.multivariate_normal(mu.ravel(), k, size=200000)
    regret = 1e6 * np.mean((c.ravel() - y) ** 2 - (b.ravel() - y) ** 2, axis=1)
    assert result.valid
    assert abs(regret.mean() - result.regret_mean_mm2) < 0.06
    assert abs(regret.var() / result.regret_variance_mm4 - 1) < 0.02
    assert abs(np.mean(regret > 1) - result.harm_probability) < 0.004


def test_covariance_changes_tail_not_mean_and_matched_parity():
    b, c = np.zeros((2, 1, 1)), np.full((2, 1, 1), 0.001)
    small = TemporalBlocks(np.eye(2)[None, None] * 1e-7)
    large = TemporalBlocks(np.eye(2)[None, None] * 1e-4)
    a, z = forecast_risk(b, c, c, small), forecast_risk(b, c, c, large)
    assert a.regret_mean_mm2 == z.regret_mean_mm2 == -1
    assert a.harm_probability < z.harm_probability
    assert (
        forecast_risk(b.copy(), c.copy(), c.copy(), TemporalBlocks(large.blocks.copy()))
        == z
    )


@pytest.mark.parametrize("bad", [np.nan, -1.0])
def test_invalid_covariance_fails_closed(bad):
    b = np.zeros((2, 1, 1))
    c = np.ones_like(b)
    operator = TemporalBlocks(np.eye(2)[None, None] * bad)
    result = forecast_risk(b, c, c, operator)
    assert not result.valid and result.failure
    chosen = ranked_acceptance([0.0], ["a"], [result.valid], 1)
    assert not chosen[0]
    assert deploy(b, c, bool(chosen[0])).tobytes() == b.tobytes()


def test_asymmetric_covariance_rejected():
    operator = TemporalBlocks(np.array([[1.0, 0.9], [0.1, 1.0]])[None, None])
    assert not forecast_risk(
        np.zeros((2, 1, 1)), np.ones((2, 1, 1)), np.ones((2, 1, 1)), operator
    ).valid


def test_cross_time_covariance_is_not_marginal_only():
    b, c = np.zeros((3, 1, 1)), np.full((3, 1, 1), 0.001)
    diagonal = np.eye(3) * 1e-5
    correlated = diagonal + (np.ones((3, 3)) - np.eye(3)) * 8e-6
    a = forecast_risk(b, c, c, TemporalBlocks(diagonal[None, None]))
    z = forecast_risk(b, c, c, TemporalBlocks(correlated[None, None]))
    assert z.regret_variance_mm4 > 2 * a.regret_variance_mm4


def test_ties_invalid_rows_and_byte_fallback():
    choice = ranked_acceptance(
        [0.2, 0.2, 0.2, 0.0], ["b", "a", "c", "invalid"], [1, 1, 1, 0]
    )
    assert choice.tolist() == [True, True, False, False]
    b = np.array([-0.0, 0.0], dtype=np.float64)
    c = np.ones_like(b)
    assert deploy(b, c, False).tobytes() == b.tobytes()
    assert deploy(b, c, True).tobytes() == c.tobytes()


def test_clamped_motion_parity():
    b = np.arange(3 * 12 * 3, dtype=float).reshape(3, 12, 3)
    result = world_forecast(b, np.ones((3, 8, 3)), np.eye(3))
    assert result[:, [0, 1, -2, -1]].tobytes() == b[:, [0, 1, -2, -1]].tobytes()


def test_controls_features_do_not_accept_outcomes():
    signature = inspect.signature(context_features)
    assert (
        "truth" not in signature.parameters
        and "future_target" not in signature.parameters
    )
    z = context_features(
        np.zeros((25, 8, 3)),
        np.ones((25, 8, 3)),
        np.zeros((10, 8, 3)),
        np.zeros((25, 4, 3)),
        50,
        25,
    )
    assert np.isfinite(z).all()


def test_native_clamp_parity_is_relative_to_baseline_not_action():
    baseline = np.zeros((25, 12, 3))
    action = np.ones((25, 4, 3)) * 0.01
    candidate = world_forecast(baseline, np.ones((25, 8, 3)), np.eye(3))
    assert not np.array_equal(baseline[:, [0, 1, -2, -1]], action)
    assert np.array_equal(candidate[:, [0, 1, -2, -1]], baseline[:, [0, 1, -2, -1]])


def test_recording_fold_partition():
    outer = folds(np.arange(40), 5, 0)
    assert all(len(a) == 8 for a in outer)
    assert len(set(np.concatenate(outer))) == 40
    for held in outer:
        train = np.setdiff1d(np.arange(40), held)
        assert not set(train) & set(held)
        inner = folds(train, 4, 100)
        assert all(len(a) == 8 for a in inner)


def test_probability_calibration_source_only_monotone():
    raw = np.linspace(0, 1, 24)
    fit = fit_probability_calibration(raw, raw > 0.6)
    calibrated = apply_probability_calibration(raw, fit)
    assert np.all(np.diff(calibrated) > 0)
    assert np.all((calibrated > 0) & (calibrated < 1))


def test_empirical_regret_models_accept_no_query_labels():
    rows = [
        {
            "recording": str(i),
            "features": np.array([i / 16, i % 2]),
            "regret_mm2": float(i - 8),
        }
        for i in range(16)
    ]
    query = [{"features": np.array([0.8, 1])}]
    for kind in ("ridge_risk", "kernel_risk"):
        p = regret_predictions(rows, query, kind, 1)
        assert p.shape == (1,) and 0 <= p[0] <= 1


def test_source_gate_stops_without_harmful_contexts():
    rows = [
        {
            "horizon": 100,
            "key": str(i),
            "valid": True,
            "regret_mm2": -1.0,
            "mse_b_mm2": 10.0,
            "mse_c_mm2": 9.0,
            "raw": dict.fromkeys(METHODS, 0.1),
            "probability": dict.fromkeys(METHODS, 0.1),
            "score": dict.fromkeys(METHODS, 0.1),
        }
        for i in range(20)
    ]
    gate = source_gate(rows)
    assert not gate["passed"] and not gate["replication_authorized"]
    assert gate["harm_reduction_fraction"] is None


def test_failed_source_cannot_read_replication_data(tmp_path):
    import json

    path = tmp_path / "source.json"
    path.write_text(
        json.dumps(
            {
                "contract": "deform-posterior-regret-source-result-v1",
                "source_gate": {"passed": False},
            }
        )
    )
    with pytest.raises(ValueError, match="authorize"):
        authorize_source(path)


def test_cluster_bootstrap_retains_recordings_and_pairs():
    objects = {}
    for obj in ("DLO4", "DLO5"):
        rows = []
        for i in range(12):
            for endpoint in (50, 150, 250):
                rows.append(
                    {
                        "recording": str(i),
                        "endpoint": endpoint,
                        "regret_mm2": 2.0,
                        "mse_b_mm2": 10.0,
                        "mse_c_mm2": 12.0,
                        "accepted50": {"posterior": False, "constant_diagonal": True},
                    }
                )
        objects[obj] = rows
    result = stratified_bootstrap(objects, "constant_diagonal")
    assert result["repetitions"] == 10000
    assert result["harm_difference"] == -3
    assert result["harm_difference_upper975"] == -3
    assert result["rmse_ratio_upper975"] < 1


def test_gp_matches_direct_penalized_feature_regression():
    from .gp import GPConfig, fit_gp

    rng = np.random.default_rng(42)
    x = rng.normal(size=(60, 3))
    y = rng.normal(size=(60, 3))
    model = fit_gp(x, y, np.repeat(np.arange(6), 10), GPConfig(max_anchors=16))
    d = model.design(x)
    penalty = np.eye(d.shape[1])
    penalty[0, 0] = 0
    expected = np.linalg.solve(d.T @ d + penalty, d.T @ y)
    np.testing.assert_allclose(model.weights, expected, rtol=1e-9, atol=1e-10)


def test_same_candidate_arrays_across_selectors():
    f, target, models, parameters = fixture()
    b, c, _, full, raw, _ = forecast_pair(
        models, parameters, f[0], target[0, prefix_indices(50)], 50, 25
    )
    b_bytes, c_bytes = b.tobytes(), c.tobytes()
    for operator in (full, raw, TemporalBlocks(full.blocks.copy())):
        assert forecast_risk(b, c, c, operator).valid
        assert b.tobytes() == b_bytes and c.tobytes() == c_bytes
