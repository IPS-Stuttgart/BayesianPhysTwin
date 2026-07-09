import numpy as np
import pytest

from bayesian_phystwin import binary_calibration_metrics


def test_binary_calibration_metrics_for_ranked_predictions() -> None:
    metrics = binary_calibration_metrics(
        np.array([0.9, 0.8, 0.2, 0.1]),
        np.array([True, True, False, False]),
        n_bins=5,
    )

    assert metrics.count == 4
    assert metrics.positive_rate == pytest.approx(0.5)
    assert metrics.brier_score == pytest.approx(0.025)
    assert metrics.roc_auc == pytest.approx(1.0)
    assert 0.0 <= metrics.expected_calibration_error <= 1.0


def test_auc_is_undefined_for_one_class() -> None:
    metrics = binary_calibration_metrics(
        np.array([0.8, 0.9]),
        np.array([True, True]),
    )

    assert metrics.roc_auc is None
