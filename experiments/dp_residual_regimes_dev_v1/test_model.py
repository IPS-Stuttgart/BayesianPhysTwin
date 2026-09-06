"""Small deterministic algebra and information-boundary checks."""
import numpy as np
import pytest

from .model import ConditionalResidual, Spec, median_of_mixture


def test_single_gaussian_median():
    rng = np.random.default_rng(1)
    m = rng.normal(size=(4, 1, 6))
    np.testing.assert_allclose(median_of_mixture(np.ones((4, 1)), m, np.ones_like(m)), m[:, 0], atol=1e-10)


@pytest.mark.parametrize("family", ["ridge", "gmm", "dp", "finite-bayes"])
def test_input_only_predictions(family):
    rng = np.random.default_rng(12)
    x = rng.normal(size=(60, 8))
    y = x @ rng.normal(size=(8, 12)) * 0.01 + rng.normal(size=(60, 12)) * 0.002
    model = ConditionalResidual(Spec(family, 2)).fit(x, y)
    frames = np.repeat(np.eye(3)[None], 5, axis=0)
    test_x = rng.normal(size=(5, 8))
    heldout_y = rng.normal(size=(5, 12))
    before = model.predict(test_x, frames)
    heldout_y[:] = 1e10
    after = model.predict(test_x, frames)
    np.testing.assert_array_equal(before["median"], after["median"])
    np.testing.assert_allclose(before["weights"].sum(axis=1), 1)
    assert before["median"].shape == (5, 12)


def test_conditional_mean_formula():
    rng = np.random.default_rng(23)
    x = rng.normal(size=(100, 8))
    y = x @ rng.normal(size=(8, 12)) * 0.01 + rng.normal(size=(100, 12)) * 0.002
    model = ConditionalResidual(Spec("gmm", 1)).fit(x, y)
    w, mean, cov = model.latent_components(x[:3])
    a = model.xpca.transform(model.scaler.transform(x[:3]))
    m, s = model.mixture.means_[0], model.mixture.covariances_[0]
    expected = m[model.dx:] + (a-m[:model.dx]) @ np.linalg.solve(s[:model.dx,:model.dx], s[:model.dx,model.dx:])
    np.testing.assert_allclose(mean[:,0], expected)
    np.testing.assert_allclose(w, 1)
    assert np.linalg.eigvalsh(cov[0]).min() > 0


def test_rotation_and_nonfinite_inputs():
    rng = np.random.default_rng(18)
    x = rng.normal(size=(100, 8))
    y = rng.normal(size=(100, 12))*0.01
    model = ConditionalResidual(Spec("gmm", 1)).fit(x, y)
    frame = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    identity = np.repeat(np.eye(3)[None], 3, axis=0)
    a = model.predict(x[:3], identity)["mean"].reshape(3, -1, 3)
    b = model.predict(x[:3], np.repeat(frame[None], 3, axis=0))["mean"].reshape(3, -1, 3)
    np.testing.assert_allclose(b, a @ frame.T, atol=1e-12)
    with pytest.raises(ValueError):
        model.predict(np.full((3, 8), np.nan), identity)
