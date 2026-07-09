import numpy as np

from bayesian_phystwin import ParameterEnsemble


def test_parameter_update_moves_mean_toward_low_residual_particle() -> None:
    particles = np.array([[0.0], [1.0], [2.0]])
    ensemble = ParameterEnsemble.from_prior_samples(particles)

    ensemble.update_from_residuals(np.array([4.0, 0.0, 4.0]), variance=1.0)

    assert abs(ensemble.mean()[0] - 1.0) < 0.25
    assert ensemble.effective_sample_size < 3.0


def test_systematic_resample_restores_uniform_weights() -> None:
    particles = np.linspace(0.0, 1.0, 8)[:, None]
    ensemble = ParameterEnsemble.from_prior_samples(particles)
    ensemble.update_from_residuals(np.linspace(0.0, 4.0, 8), variance=0.5)

    ensemble.systematic_resample(np.random.default_rng(3))

    assert ensemble.particles.shape == particles.shape
    assert np.allclose(ensemble.weights, np.full(8, 1.0 / 8.0))

