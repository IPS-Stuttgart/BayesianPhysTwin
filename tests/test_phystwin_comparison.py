import numpy as np
import pytest

from bayesian_phystwin.phystwin_comparison import (
    official_metrics_by_frame,
    paired_block_bootstrap,
    phystwin_physical_object_cluster,
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


def test_cluster_bootstrap_weights_physical_objects_equally() -> None:
    baseline = {
        "chamfer_distance_m": np.ones(8),
        "track_error_m": np.ones(8),
    }
    cases = {
        "object_a_first": (
            baseline,
            {key: 0.5 * value for key, value in baseline.items()},
        ),
        "object_a_second": (
            baseline,
            {key: 0.5 * value for key, value in baseline.items()},
        ),
        "object_b": (
            baseline,
            {key: 1.2 * value for key, value in baseline.items()},
        ),
    }

    result = paired_block_bootstrap(
        cases,
        samples=200,
        block_length=2,
        seed=4,
        clusters={
            "object_a_first": "object_a",
            "object_a_second": "object_a",
            "object_b": "object_b",
        },
    )

    assert result["macro"]["chamfer_distance_m"][
        "observed_macro_percent_change"
    ] == pytest.approx(-26.6666666667)
    assert result["cluster_macro"]["metrics"]["chamfer_distance_m"][
        "observed_equal_cluster_percent_change"
    ] == pytest.approx(-15.0)


def test_released_case_names_map_to_conservative_object_clusters() -> None:
    assert phystwin_physical_object_cluster("double_lift_cloth_1") == "cloth_1"
    assert phystwin_physical_object_cluster("single_lift_cloth") == "cloth"
    assert phystwin_physical_object_cluster("single_push_rope_1") == "rope_1"
    assert phystwin_physical_object_cluster("rope_double_hand") == "rope"
