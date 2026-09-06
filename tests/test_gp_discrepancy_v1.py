"""Implementation tests, not evidence of real-object improvement."""

import numpy as np
import pytest

from bayesian_phystwin_experiments.gp_discrepancy_v1 import (
    GPConfig,
    RecordingDiscrepancyGP,
    chain_modes,
    empirical_lowrank,
    frame_block_diagonal,
    gaussian_metrics,
    matern32,
)


def fixture(config=GPConfig()):
    time = np.tile(np.linspace(0, 1, 6), 4)
    ids = np.repeat(["fit-a", "fit-b", "fit-c", "fit-d"], 6)
    x = np.column_stack((time, np.repeat([-1.0, -0.3, 0.4, 1.1], 6)))
    y = np.column_stack((np.sin(2 * x[:, 1]) * (1 + time), np.cos(3 * time)))
    return RecordingDiscrepancyGP.fit(x, time, ids, y, config), x, time, ids, y


def query():
    time = np.array([0.2, 0.5, 0.8])
    return np.column_stack((time, np.full(3, 0.1))), time, np.repeat("new", 3)


def test_matern_analytic_and_psd():
    x = np.array([[0.0], [1.0], [2.0]])
    kernel = matern32(x, x, 1.0)
    assert kernel[0, 0] == 1.0
    assert kernel[0, 1] == pytest.approx((1 + np.sqrt(3)) * np.exp(-np.sqrt(3)))
    assert np.linalg.eigvalsh(kernel).min() > 0


@pytest.mark.parametrize("kwargs", [
    {"lengthscale": 0}, {"time_lengthscale": -1}, {"shared_variance": 0},
    {"session_variance": -1}, {"noise_variance": 0}, {"output_scale_floor": 0},
    {"noise_variance": float("nan")}, {"max_rows": True}, {"max_rows": 0},
])
def test_config_fails_closed(kwargs):
    with pytest.raises(ValueError):
        GPConfig(**kwargs)


def test_kernel_ridge_equivalent_mean():
    model, x, time, ids, y = fixture(GPConfig(session_variance=0))
    qx, qt, qi = query()
    prediction = model.predict(qx, qt, qi)
    train_x = (x - model.feature_location) / model.feature_scale
    query_x = (qx - model.feature_location) / model.feature_scale
    kernel = matern32(train_x, train_x, model.config.lengthscale)
    kernel += np.eye(len(x)) * model.config.noise_variance
    cross = matern32(query_x, train_x, model.config.lengthscale)
    expected = model.output_location + cross @ np.linalg.solve(kernel, y - model.output_location)
    np.testing.assert_allclose(prediction.mean, expected, rtol=1e-11, atol=1e-12)


def test_training_row_permutation_invariance():
    model, x, time, ids, y = fixture()
    order = np.random.default_rng(4).permutation(len(x))
    shuffled = RecordingDiscrepancyGP.fit(x[order], time[order], ids[order], y[order])
    a, b = model.predict(*query()), shuffled.predict(*query())
    np.testing.assert_allclose(a.mean, b.mean, atol=1e-12)
    np.testing.assert_allclose(a.covariance, b.covariance, atol=1e-12)


def test_recording_renaming_invariance():
    model, x, time, ids, y = fixture()
    renamed = RecordingDiscrepancyGP.fit(x, time, np.char.add("prefix-", ids), y)
    np.testing.assert_allclose(model.predict(*query()).covariance, renamed.predict(*query()).covariance)


def test_new_recording_effect_does_not_vanish():
    model, *_ = fixture()
    qx, qt, qi = query()
    prediction = model.predict(qx, qt, qi)
    random_effect = model.config.session_variance * matern32(qt[:, None], qt[:, None], model.config.time_lengthscale)
    random_effect += np.eye(len(qt)) * model.config.noise_variance
    for mode, covariance in enumerate(prediction.covariance):
        remaining = covariance / model.output_scale[mode] ** 2 - random_effect
        assert np.linalg.eigvalsh(remaining).min() >= -1e-11


def test_independent_recordings_keep_only_shared_cross_covariance():
    model, *_ = fixture()
    qx, qt, qi = query()
    together = model.predict(qx, qt, qi)
    separate = model.predict(qx, qt, np.array(["new-a", "new-b", "new-c"]))
    np.testing.assert_array_equal(together.mean, separate.mean)
    difference = together.covariance - separate.covariance
    expected = model.config.session_variance * matern32(qt[:, None], qt[:, None], model.config.time_lengthscale)
    np.fill_diagonal(expected, 0.0)
    np.testing.assert_allclose(difference, model.output_scale[:, None, None] ** 2 * expected, atol=1e-12)


def test_same_recording_prefix_conditions_its_random_effect():
    model, *_ = fixture(GPConfig(session_variance=2.0))
    qx, qt, qi = query()
    # .2 and .8 were observed; .5 was not.
    fresh = model.predict(qx[1:2], qt[1:2], qi[1:2])
    seen = model.predict(qx[1:2], qt[1:2], np.array(["fit-b"]))
    assert np.all(seen.covariance[:, 0, 0] < fresh.covariance[:, 0, 0])


def test_duplicate_fit_observations_rejected():
    _, x, time, ids, y = fixture()
    with pytest.raises(ValueError, match="duplicate"):
        RecordingDiscrepancyGP.fit(np.r_[x, x[:1]], np.r_[time, time[:1]], np.r_[ids, ids[:1]], np.r_[y, y[:1]])


def test_same_observation_cannot_be_queried_as_new_evidence():
    model, x, time, ids, _ = fixture()
    with pytest.raises(ValueError, match="overlaps"):
        model.predict(x[:1], time[:1], ids[:1])


@pytest.mark.parametrize("bad", [np.array([1, 2, 3]), np.array(["", "a", "b"]), np.array([b"a", b"b", b"c"])])
def test_identity_validation(bad):
    model, *_ = fixture()
    x, time, _ = query()
    with pytest.raises(ValueError):
        model.predict(x, time, bad)


def test_dense_limit_and_nonfinite_failures():
    model, x, time, ids, y = fixture()
    with pytest.raises(ValueError, match="limit"):
        RecordingDiscrepancyGP.fit(x, time, ids, y, GPConfig(max_rows=5))
    qx, qt, qi = query()
    qx[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        model.predict(qx, qt, qi)


def test_clamps_and_orthonormal_modes():
    basis = chain_modes(13, 4)
    np.testing.assert_array_equal(basis[[0, 1, 11, 12]], 0)
    np.testing.assert_allclose(basis.T @ basis, np.eye(4), atol=1e-14)
    corrected = np.arange(39.0).reshape(13, 3) + basis @ np.ones((4, 3))
    np.testing.assert_array_equal(corrected[[0, 1, 11, 12]], np.arange(39.0).reshape(13, 3)[[0, 1, 11, 12]])


def test_block_control_preserves_every_frame_covariance():
    model, *_ = fixture()
    prediction = model.predict(*query())
    full = prediction.joint_covariance()
    blocks = frame_block_diagonal(full, 2)
    for start in range(0, 6, 2):
        np.testing.assert_array_equal(blocks[start:start + 2, start:start + 2], full[start:start + 2, start:start + 2])
    assert np.count_nonzero(full - blocks) > 0
    assert np.linalg.eigvalsh(blocks).min() > 0


def test_joint_samples_preserve_covariance():
    model, *_ = fixture()
    prediction = model.predict(*query())
    draws = prediction.sample(12, 30000).reshape(30000, -1)
    np.testing.assert_allclose(draws.mean(axis=0), prediction.mean.ravel(), atol=0.012)
    np.testing.assert_allclose(np.cov(draws, rowvar=False), prediction.joint_covariance(), atol=0.012)


def test_lowrank_is_positive_and_uses_recordings_as_rows():
    errors = np.random.default_rng(3).normal(size=(7, 12))
    covariance = empirical_lowrank(errors, rank=3, diagonal_fraction=0.2, variance_floor=1e-6)
    assert covariance.shape == (12, 12)
    assert np.linalg.eigvalsh(covariance).min() > 0
    with pytest.raises(ValueError, match="rank"):
        empirical_lowrank(errors, rank=8, diagonal_fraction=0.2, variance_floor=1e-6)


def test_gaussian_metrics_analytic():
    values = gaussian_metrics(np.ones(3), np.eye(3))
    assert values["normalized_nees"] == pytest.approx(1)
    assert values["nll_per_dimension"] == pytest.approx(0.5 * (1 + np.log(2 * np.pi)))
    assert values["marginal_90_coverage"] == 1
    with pytest.raises(ValueError):
        gaussian_metrics(np.ones(3), -np.eye(3))
