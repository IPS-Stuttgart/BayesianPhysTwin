import numpy as np

from bayesian_phystwin.phystwin_pgrd_adapter import MetricNormalizer
from bayesian_phystwin.phystwin_pgrd_calibrated import (
    collect_teacher_forced_pgrd_pairs,
    fit_calibrated_velocity_readout,
    rollout_calibrated_pgrd_correction,
)


class _ConstantXPredictor:
    def reset(self) -> None:
        return None

    def predict(
        self,
        x: np.ndarray,
        v: np.ndarray,
        x_history: np.ndarray,
        v_history: np.ndarray,
        x_sim: np.ndarray,
        v_sim: np.ndarray,
    ) -> np.ndarray:
        del v, x_history, v_history, x_sim, v_sim
        result = np.zeros_like(x)
        result[:, 0] = 1.0
        return result


def _trajectory() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    initial = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.02, 0.00, 0.00],
            [0.00, 0.02, 0.00],
            [0.02, 0.02, 0.00],
        ]
    )
    baseline = np.repeat(initial[None], 10, axis=0)
    observed = baseline.copy()
    observed[:, :, 0] += 0.001 * np.arange(10)[:, None]
    valid = np.ones((10, 4), dtype=bool)
    return baseline, observed, valid


def test_readout_recovers_linear_map_and_clips_gain() -> None:
    features = np.eye(3)
    targets = 2.0 * features

    readout = fit_calibrated_velocity_readout(
        features, targets, ridge=1e-12, maximum_gain=1.5
    )

    np.testing.assert_allclose(np.linalg.svd(readout, compute_uv=False), 1.5)


def test_teacher_pairs_use_prefix_residual_increments() -> None:
    baseline, observed, valid = _trajectory()
    normalizer = MetricNormalizer.fit(baseline[0], 0.5)

    features, targets = collect_teacher_forced_pgrd_pairs(
        baseline,
        observed[:7],
        valid[:7],
        np.arange(4),
        _ConstantXPredictor(),
        normalizer,
        history_length=2,
        simulation_dt=0.1,
        model_frame_stride=1,
    )

    assert features.shape == targets.shape
    assert len(features) == 16
    np.testing.assert_allclose(features[:, 0], 0.1)
    np.testing.assert_allclose(targets[:, 0], 0.025)


def test_calibrated_rollout_recovers_constant_residual_velocity() -> None:
    baseline, observed, valid = _trajectory()
    normalizer = MetricNormalizer.fit(baseline[0], 0.5)
    features, targets = collect_teacher_forced_pgrd_pairs(
        baseline,
        observed[:7],
        valid[:7],
        np.arange(4),
        _ConstantXPredictor(),
        normalizer,
        history_length=2,
        simulation_dt=0.1,
        model_frame_stride=1,
    )
    readout = fit_calibrated_velocity_readout(
        features, targets, ridge=1e-12, maximum_gain=5.0
    )

    correction = rollout_calibrated_pgrd_correction(
        baseline,
        observed[:7],
        valid[:7],
        np.arange(4),
        _ConstantXPredictor(),
        normalizer,
        readout,
        start_frame=7,
        end_frame=10,
        history_length=2,
        simulation_dt=0.1,
        model_frame_stride=1,
        maximum_residual_m=0.1,
    )

    np.testing.assert_allclose(
        correction[:, :, 0],
        np.repeat(np.array([0.007, 0.008, 0.009])[:, None], 4, axis=1),
        atol=1e-10,
    )
    np.testing.assert_allclose(correction[:, :, 1:], 0.0, atol=1e-12)
