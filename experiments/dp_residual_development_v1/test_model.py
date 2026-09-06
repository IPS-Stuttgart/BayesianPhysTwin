import numpy as np
from experiments.dp_residual_development_v1.model import GateSpec, ResidualExperts


def fixture(seed=4):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(24, 20, 2, 4))
    labels = rng.choice([-1, 1], size=(24, 1, 1))
    x[:, :, :, 0] = 3 * labels + .2 * x[:, :, :, 0]
    x[:, :, 1, 0] = x[:, :, 0, 0]
    y = np.zeros((24, 20, 2, 3))
    y[..., 0] = np.sign(x[..., 0]) * x[..., 1]
    y[..., 1] = np.sign(x[..., 0]) * x[..., 2]
    return x, y


def test_single_reproduces_direct_ridge():
    x, y = fixture()
    model = ResidualExperts(GateSpec('single', 1), 1).fit(x, y)
    pred = model.predict(x)
    for node in range(2):
        design = model._design(x, node)
        pen = np.eye(design.shape[1])
        pen[0, 0] = 0
        expected = design @ np.linalg.solve(design.T @ design + pen,
                                            design.T @ y[:, :, node].reshape(-1, 3))
        np.testing.assert_allclose(pred[:, :, node].reshape(-1, 3), expected, atol=1e-10)


def test_positive_control_and_probability_normalization():
    x, y = fixture()
    single = ResidualExperts(GateSpec('single', 1), 1).fit(x[:18], y[:18])
    finite = ResidualExperts(GateSpec('finite', 2), 1).fit(x[:18], y[:18])
    np.testing.assert_allclose(finite.probabilities(x[18:]).sum(-1), 1, atol=1e-12)
    err_single = np.mean((single.predict(x[18:]) - y[18:])**2)
    err_finite = np.mean((finite.predict(x[18:]) - y[18:])**2)
    assert err_finite < .1 * err_single


def test_dp_and_finite_bayes_are_distinct_priors():
    x, y = fixture()
    dp = ResidualExperts(GateSpec('dp', 4, .1), 1).fit(x[:18], y[:18])
    fb = ResidualExperts(GateSpec('finite_bayes', 4, .1), 1).fit(x[:18], y[:18])
    assert dp.gate.weight_concentration_prior_type == 'dirichlet_process'
    assert fb.gate.weight_concentration_prior_type == 'dirichlet_distribution'
    assert np.isfinite(dp.predict(x[18:])).all()


def test_input_validation():
    import pytest
    x, y = fixture()
    with pytest.raises(RuntimeError):
        ResidualExperts(GateSpec('single', 1), 1).predict(x)
    with pytest.raises(ValueError):
        ResidualExperts(GateSpec('dp', 0), 1)
    with pytest.raises(ValueError):
        ResidualExperts(GateSpec('single', 1), 1).fit(x, y[:-1])
    x[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        ResidualExperts(GateSpec('single', 1), 1).fit(x, y)


def test_predict_does_not_mutate_or_refit():
    x, y = fixture()
    model = ResidualExperts(GateSpec('finite', 2), 1).fit(x[:18], y[:18])
    query = x[18:].copy()
    p = model.predict(query)
    query_copy = query.copy()
    params = model.coefficients.copy()
    model.predict(query, hard=True)
    np.testing.assert_array_equal(query, query_copy)
    np.testing.assert_array_equal(params, model.coefficients)
    np.testing.assert_array_equal(p, model.predict(query))


def test_free_future_truth_cannot_change_causal_features():
    from bayesian_phystwin_experiments.deform_dlo_local_residual import (
        build_deform_local_residual_features, deform_causal_inputs)
    full = np.zeros((3, 12, 13, 3))
    full[..., 0] = np.linspace(0, 1, 13)
    full[..., 1] = np.arange(12)[None, :, None] * .001
    baseline = full[:, 2:].copy()
    initial, action = deform_causal_inputs(full)
    original, frame = build_deform_local_residual_features(initial, action, baseline)
    full[:, 2:, 2:-2] += 1000
    changed_initial, changed_action = deform_causal_inputs(full)
    changed, changed_frame = build_deform_local_residual_features(changed_initial, changed_action, baseline)
    np.testing.assert_array_equal(original, changed)
    np.testing.assert_array_equal(frame, changed_frame)


def test_zero_strength_and_clamped_nodes_remain_exact():
    from experiments.dp_residual_development_v1.run import correction_prediction
    baseline = np.zeros((2, 5, 13, 3))
    local = np.ones((2, 5, 9, 3))
    frames = np.broadcast_to(np.eye(3), (2, 3, 3))
    np.testing.assert_array_equal(correction_prediction(baseline, local, frames, 0), baseline)
    prediction = correction_prediction(baseline, local, frames, .5)
    np.testing.assert_array_equal(prediction[:, :, [0, 1, -2, -1]], baseline[:, :, [0, 1, -2, -1]])
    np.testing.assert_allclose(prediction[:, :, 2:-2], .5)
