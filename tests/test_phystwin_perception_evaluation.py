import numpy as np

from bayesian_phystwin.phystwin_perception_evaluation import (
    ReliabilityTransform,
    compose_perception_reliability,
    reliability_error_metrics,
)


def test_compose_perception_reliability_uses_valid_rich_cues_only() -> None:
    cues = {
        "confidence": np.array([1.0, 0.8, 0.5]),
        "visibility_probability": np.array([1.0, 0.5, 1.0]),
        "forward_backward_error_px": np.array([2.0, 100.0, 4.0]),
        "forward_backward_valid": np.array([True, False, True]),
        "multiview_reprojection_error_px": np.array([1.0, 2.0, 100.0]),
        "multiview_valid": np.array([True, True, False]),
    }

    reliability = compose_perception_reliability(
        cues,
        ReliabilityTransform(
            "rich",
            forward_backward_scale_px=2.0,
            multiview_scale_px=1.0,
        ),
    )

    np.testing.assert_allclose(
        reliability,
        [np.exp(-2.0), 0.4 * np.exp(-2.0), 0.5 * np.exp(-2.0)],
    )


def test_reliability_error_metrics_rewards_correct_ranking() -> None:
    metrics = reliability_error_metrics(
        np.array([0.1, 0.2, 0.8, 0.9]),
        np.array([0.020, 0.010, 0.002, 0.001]),
    )

    assert metrics["spearman_reliability_vs_error"] == -1.0
    assert metrics["lowest_reliability_quartile_error_m"] == 0.020
    assert metrics["highest_reliability_quartile_error_m"] == 0.001
    assert metrics["highest_reliability_half_error_ratio"] < 0.2
    assert metrics["unreliability_auroc"]["error_at_least_0.005_m"] == 1.0
