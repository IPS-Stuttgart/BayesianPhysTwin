import numpy as np

from bayesian_phystwin.phystwin_comparison import (
    official_metrics_by_frame,
    paired_block_bootstrap,
)


def test_frame_metrics_match_simple_translation_error() -> None:
    vertices = np.zeros((3, 2, 3))
    observed = np.zeros((3, 2, 3))
    observed[1:, :, 0] = 0.01
    visible = np.ones((3, 2), dtype=bool)
    tracks = observed[:, :1]

    metrics = official_metrics_by_frame(
        vertices,
        observed,
        visible,
        tracks,
        num_surface_points=2,
        start_frame=1,
        end_frame=3,
    )

    np.testing.assert_allclose(metrics["chamfer_distance_m"], 0.01)
    np.testing.assert_allclose(metrics["track_error_m"], 0.01)


def test_paired_bootstrap_detects_uniform_improvement() -> None:
    baseline = {
        "chamfer_distance_m": np.full(12, 2.0),
        "track_error_m": np.full(12, 4.0),
    }
    candidate = {
        "chamfer_distance_m": np.full(12, 1.0),
        "track_error_m": np.full(12, 2.0),
    }

    result = paired_block_bootstrap(
        {"case": (baseline, candidate)},
        samples=100,
        block_length=3,
        seed=4,
    )

    assert result["per_case"]["case"]["chamfer_distance_m"][
        "observed_percent_change"
    ] == -50.0
    assert result["macro"]["track_error_m"][
        "case_and_frame_bootstrap_percent_change"
    ]["probability_improved"] == 1.0
