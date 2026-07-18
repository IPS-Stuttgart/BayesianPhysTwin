from __future__ import annotations

import numpy as np

from bayesian_phystwin.phystwin_pgrd_adapter import MetricNormalizer
from bayesian_phystwin.phystwin_pgrd_unrolled import (
    UnrolledPGRDTrainingConfig,
    available_window_starts,
    build_unrolled_pgrd_sequence,
    load_unrolled_sequence,
    save_unrolled_sequence,
)


def _sequence() -> tuple[object, UnrolledPGRDTrainingConfig]:
    config = UnrolledPGRDTrainingConfig(
        number_of_points=4,
        rollout_steps=3,
        windows_per_case_per_epoch=1,
        epochs=1,
    )
    frame_count = 25
    point_count = 6
    baseline = np.zeros((frame_count, point_count, 3), dtype=float)
    baseline[:, :, 0] = np.arange(point_count)[None] * 0.02
    baseline[:, :, 2] = np.arange(frame_count)[:, None] * 0.001
    residual = np.zeros_like(baseline)
    residual[:, :, 1] = np.arange(frame_count)[:, None] * 0.0001
    observed = baseline + residual
    valid = np.ones((frame_count, point_count), dtype=bool)
    valid[7, 2] = False
    observed[7, 2] = 100.0
    indices = np.array([0, 2, 4, 5])
    normalizer = MetricNormalizer.fit(baseline[0, indices], 0.5)
    sequence = build_unrolled_pgrd_sequence(
        "synthetic",
        baseline,
        observed,
        valid,
        indices,
        normalizer,
        end_frame=frame_count,
        config=config,
    )
    return sequence, config


def test_unrolled_sequence_is_causal_and_round_trips(tmp_path) -> None:
    sequence, _ = _sequence()
    np.testing.assert_array_equal(sequence.target_frames, [9, 12, 15, 18, 21, 24])
    assert sequence.baseline_m.shape == (25, 4, 3)
    assert sequence.observed_m.shape == (25, 4, 3)
    assert sequence.valid.shape == (25, 4)
    # The invalid outlier is interpolated inside the permitted source prefix.
    np.testing.assert_allclose(
        sequence.observed_m[7, 1] - sequence.baseline_m[7, 1],
        [0.0, 0.0007, 0.0],
    )

    path = tmp_path / "sequence.npz"
    save_unrolled_sequence(sequence, path)
    restored = load_unrolled_sequence(path)
    assert restored.case == sequence.case
    np.testing.assert_array_equal(restored.target_frames, sequence.target_frames)
    np.testing.assert_array_equal(restored.sample_indices, sequence.sample_indices)
    np.testing.assert_allclose(restored.baseline_m, sequence.baseline_m)
    np.testing.assert_allclose(restored.observed_m, sequence.observed_m)
    np.testing.assert_array_equal(restored.valid, sequence.valid)
    np.testing.assert_allclose(restored.center_m, sequence.center_m)
    np.testing.assert_allclose(
        restored.rotation_model_from_metric,
        sequence.rotation_model_from_metric,
    )
    assert restored.scale_per_m == sequence.scale_per_m


def test_unrolled_windows_require_complete_recursive_targets() -> None:
    sequence, config = _sequence()
    np.testing.assert_array_equal(
        available_window_starts(sequence, config.rollout_steps), [0, 1, 2, 3]
    )


def test_unrolled_evaluation_contract_matches_training_geometry() -> None:
    _, config = _sequence()
    evaluation = config.evaluation_config()
    assert evaluation.number_of_points == config.number_of_points
    assert evaluation.normalized_extent == config.normalized_extent
    assert evaluation.yaw_degrees == config.yaw_degrees
    assert evaluation.model_frame_stride == config.model_frame_stride
    assert evaluation.maximum_residual_m == config.maximum_residual_m
    assert evaluation.minimum_balanced_improvement == 0.01
