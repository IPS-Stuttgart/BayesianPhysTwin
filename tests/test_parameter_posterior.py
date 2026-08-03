import numpy as np
import pytest

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


def test_constructor_defensively_copies_and_normalizes() -> None:
    particles = np.array([[0.0, 1.0], [2.0, 3.0]])
    log_weights = np.array([0.0, -2.0])
    ensemble = ParameterEnsemble(particles, log_weights)
    particles[:] = 99.0
    log_weights[:] = 99.0

    assert np.array_equal(ensemble.particles, [[0.0, 1.0], [2.0, 3.0]])
    assert np.isclose(np.sum(ensemble.weights), 1.0)
    assert ensemble.particle_count == 2
    assert ensemble.dimension == 2


@pytest.mark.parametrize(
    "particles",
    [
        np.empty((0, 2)),
        np.empty((2, 0)),
        np.array([[np.nan]]),
        np.array([1.0, 2.0]),
    ],
)
def test_prior_samples_reject_invalid_particles(particles: np.ndarray) -> None:
    with pytest.raises(ValueError):
        ParameterEnsemble.from_prior_samples(particles)


def test_constructor_rejects_invalid_log_weights() -> None:
    with pytest.raises(ValueError, match="one value per particle"):
        ParameterEnsemble(np.ones((2, 1)), np.zeros(3))
    with pytest.raises(ValueError, match="negative infinity"):
        ParameterEnsemble(np.ones((2, 1)), np.array([0.0, np.nan]))
    with pytest.raises(ValueError, match="at least one"):
        ParameterEnsemble(np.ones((2, 1)), np.array([-np.inf, -np.inf]))


@pytest.mark.parametrize(
    ("residual", "variance", "reliability", "message"),
    [
        (np.array([-1.0, 0.0]), 1.0, None, "nonnegative"),
        (np.array([0.0, np.nan]), 1.0, None, "finite"),
        (np.zeros(2), 0.0, None, "positive"),
        (np.zeros(2), np.inf, None, "positive"),
        (np.zeros(2), 1.0, np.array([0.5, np.nan]), "finite"),
        (np.zeros(2), 1.0, np.array([0.5, 1.1]), r"\[0, 1\]"),
    ],
)
def test_update_rejects_invalid_likelihood_inputs(
    residual: np.ndarray,
    variance: float,
    reliability: np.ndarray | None,
    message: str,
) -> None:
    ensemble = ParameterEnsemble.from_prior_samples(np.array([[0.0], [1.0]]))
    with pytest.raises((ValueError, FloatingPointError), match=message):
        ensemble.update_from_residuals(
            residual,
            variance=variance,
            reliability=reliability,
        )


def test_public_state_corruption_fails_closed() -> None:
    ensemble = ParameterEnsemble.from_prior_samples(np.array([[0.0], [1.0]]))
    ensemble.particles[0, 0] = np.nan
    with pytest.raises(ValueError, match="particles"):
        _ = ensemble.weights


def test_resampling_rejects_invalid_jitter() -> None:
    ensemble = ParameterEnsemble.from_prior_samples(
        np.array([[0.0, 1.0], [2.0, 3.0]])
    )
    with pytest.raises(ValueError, match="jitter_std"):
        ensemble.systematic_resample(jitter_std=np.ones(3))
    with pytest.raises(ValueError, match="nonnegative"):
        ensemble.systematic_resample(jitter_std=-1.0)
    with pytest.raises(ValueError, match="finite"):
        ensemble.systematic_resample(jitter_std=np.nan)


def test_covariance_is_symmetric_and_read_only() -> None:
    ensemble = ParameterEnsemble.from_prior_samples(
        np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]])
    )
    covariance = ensemble.covariance()

    assert np.allclose(covariance, covariance.T)
    assert not covariance.flags.writeable
    assert not ensemble.mean().flags.writeable
    assert not ensemble.weights.flags.writeable


def test_update_rejects_shape_mismatches_and_overflow() -> None:
    ensemble = ParameterEnsemble.from_prior_samples(np.array([[0.0], [1.0]]))
    with pytest.raises(ValueError, match="residual_sums"):
        ensemble.update_from_residuals(np.zeros(3), variance=1.0)
    with pytest.raises(ValueError, match="reliability"):
        ensemble.update_from_residuals(
            np.zeros(2),
            variance=1.0,
            reliability=np.ones(3),
        )
    with pytest.raises(FloatingPointError, match="overflowed"):
        ensemble.update_from_residuals(
            np.full(2, np.finfo(np.float64).max),
            variance=np.finfo(np.float64).tiny,
        )


def test_positive_jitter_path_is_exercised() -> None:
    ensemble = ParameterEnsemble.from_prior_samples(
        np.array([[0.0, 0.0], [1.0, 1.0]])
    )
    ensemble.systematic_resample(
        np.random.default_rng(7),
        jitter_std=np.array([0.1, 0.2]),
    )
    assert np.all(np.isfinite(ensemble.particles))


def test_resampling_rejects_nonfinite_generator_output() -> None:
    class NonfiniteGenerator:
        def random(self) -> float:
            return 0.5

        def normal(self, **kwargs) -> np.ndarray:
            return np.full(kwargs["size"], np.inf)

    ensemble = ParameterEnsemble.from_prior_samples(np.array([[0.0], [1.0]]))
    with pytest.raises(FloatingPointError, match="non-finite"):
        ensemble.systematic_resample(  # type: ignore[arg-type]
            NonfiniteGenerator(),
            jitter_std=0.1,
        )


def test_mutated_invalid_shapes_fail_closed() -> None:
    ensemble = ParameterEnsemble.from_prior_samples(np.array([[0.0], [1.0]]))
    ensemble.particles = np.ones(2)
    with pytest.raises(ValueError, match="shape"):
        _ = ensemble.particle_count

    ensemble.particles = np.empty((0, 1))
    ensemble.log_weights = np.empty(0)
    with pytest.raises(ValueError, match="at least one"):
        _ = ensemble.dimension


def test_invalid_exponential_totals_fail_closed(monkeypatch) -> None:
    import bayesian_phystwin.parameter_posterior as posterior_module

    ensemble = ParameterEnsemble.from_prior_samples(np.array([[0.0], [1.0]]))
    monkeypatch.setattr(
        posterior_module.np,
        "exp",
        lambda value: np.zeros_like(value, dtype=np.float64),
    )
    with pytest.raises(FloatingPointError, match="invalid particle weights"):
        _ = ensemble.weights
    with pytest.raises(FloatingPointError, match="normalization"):
        ensemble._renormalize_log_weights()
