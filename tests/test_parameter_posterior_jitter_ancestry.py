import numpy as np

from bayesian_phystwin import ParameterEnsemble


class _RecordingGenerator:
    def __init__(self) -> None:
        self.scale: np.ndarray | None = None

    def random(self) -> float:
        return 0.5

    def normal(self, *, loc: float, scale: np.ndarray, size: tuple[int, ...]) -> np.ndarray:
        assert loc == 0.0
        self.scale = np.asarray(scale, dtype=np.float64).copy()
        return np.zeros(size, dtype=np.float64)


def test_particle_specific_jitter_follows_resampled_ancestor() -> None:
    ensemble = ParameterEnsemble(
        particles=np.asarray([[0.0], [1.0], [2.0]]),
        log_weights=np.log(np.asarray([0.8, 0.1, 0.1])),
    )
    generator = _RecordingGenerator()
    source_jitter = np.asarray([[0.1], [0.2], [0.3]])

    ensemble.systematic_resample(  # type: ignore[arg-type]
        generator,
        jitter_std=source_jitter,
    )

    np.testing.assert_array_equal(ensemble.particles, np.asarray([[0.0], [0.0], [1.0]]))
    assert generator.scale is not None
    np.testing.assert_array_equal(generator.scale, np.asarray([[0.1], [0.1], [0.2]]))
