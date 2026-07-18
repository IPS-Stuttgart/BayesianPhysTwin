from __future__ import annotations

import numpy as np

from bayesian_phystwin.phystwin_pgrd_adapter import MetricNormalizer
from bayesian_phystwin.phystwin_pgrd_native import (
    NativePGRDTrainingConfig,
    aggregate_native_pgrd_gate,
    build_teacher_forced_pgrd_sequence,
    load_feature_sequence,
    save_feature_sequence,
)


class _DifferenceFeatures:
    def spatial_features(
        self,
        x: np.ndarray,
        v: np.ndarray,
        x_history: np.ndarray,
        v_history: np.ndarray,
        x_sim: np.ndarray,
        v_sim: np.ndarray,
    ) -> np.ndarray:
        assert v.shape == x.shape == x_sim.shape == v_sim.shape
        assert x_history.shape == v_history.shape == (len(x), 2, 3)
        return x_sim - x


def _metric_result(cd_ratio: float, track_ratio: float) -> dict[str, object]:
    return {
        "metric_ratios_relative_to_persistence": {
            "chamfer_distance_m": cd_ratio,
            "track_error_m": track_ratio,
        }
    }


def test_teacher_forced_sequence_is_cadence_aligned_and_metric(tmp_path) -> None:
    config = NativePGRDTrainingConfig(number_of_points=4, epochs=1)
    frame_count = 16
    point_count = 6
    baseline = np.zeros((frame_count, point_count, 3), dtype=float)
    baseline[:, :, 0] = np.arange(point_count)[None] * 0.02
    baseline[:, :, 2] = np.arange(frame_count)[:, None] * 0.001
    correction = np.zeros_like(baseline)
    correction[:, :, 1] = np.arange(frame_count)[:, None] * 0.0002
    observed = baseline + correction
    valid = np.ones((frame_count, point_count), dtype=bool)
    sample_indices = np.array([0, 2, 4, 5])
    normalizer = MetricNormalizer.fit(baseline[0, sample_indices], 0.5)

    sequence = build_teacher_forced_pgrd_sequence(
        "synthetic",
        baseline,
        observed,
        valid,
        sample_indices,
        _DifferenceFeatures(),
        normalizer,
        end_frame=frame_count,
        config=config,
    )

    np.testing.assert_array_equal(sequence.target_frames, [9, 12, 15])
    assert sequence.spatial_features.shape == (3, 4, 3)
    assert sequence.target_residual_velocity.shape == (3, 4, 3)
    assert sequence.valid.all()
    expected = normalizer.velocities_to_model(correction[15, sample_indices]) / 0.1
    np.testing.assert_allclose(sequence.target_residual_velocity[-1], expected)

    path = tmp_path / "features.npz"
    save_feature_sequence(sequence, path)
    restored = load_feature_sequence(path)
    assert restored.case == sequence.case
    np.testing.assert_array_equal(restored.target_frames, sequence.target_frames)
    np.testing.assert_array_equal(restored.sample_indices, sequence.sample_indices)
    np.testing.assert_allclose(restored.spatial_features, sequence.spatial_features)
    np.testing.assert_allclose(
        restored.target_residual_velocity, sequence.target_residual_velocity
    )
    np.testing.assert_array_equal(restored.valid, sequence.valid)


def test_native_gate_requires_transfer_in_both_metrics() -> None:
    config = NativePGRDTrainingConfig(epochs=1)
    passed = aggregate_native_pgrd_gate(
        {
            "a": _metric_result(0.97, 0.98),
            "b": _metric_result(0.98, 0.97),
            "c": _metric_result(0.99, 0.99),
        },
        config=config,
    )
    assert passed["passed"] is True
    assert passed["exploratory_19_case_future_authorized"] is True

    failed = aggregate_native_pgrd_gate(
        {
            "a": _metric_result(0.96, 1.01),
            "b": _metric_result(0.97, 1.01),
            "c": _metric_result(0.98, 1.01),
        },
        config=config,
    )
    assert failed["passed"] is False
    assert failed["exploratory_19_case_future_authorized"] is False


def test_native_gate_rejects_one_large_case_regression() -> None:
    result = aggregate_native_pgrd_gate(
        {
            "a": _metric_result(0.90, 0.90),
            "b": _metric_result(0.90, 0.90),
            "c": _metric_result(1.03, 0.90),
        },
        config=NativePGRDTrainingConfig(epochs=1),
    )
    assert result["balanced_improvement"] > 0.01
    assert result["passed"] is False
