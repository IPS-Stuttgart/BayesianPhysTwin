import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


def _load_runner():
    path = (
        Path(__file__).parents[1]
        / "scripts"
        / "remote"
        / "run_matphys_sparse_prefix_headroom.py"
    )
    name = "run_matphys_sparse_prefix_headroom_test"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_recursive_rbf_refit_falls_back_exactly_when_availability_drops() -> None:
    runner = _load_runner()
    baseline = np.zeros((6, 1, 3), dtype=float)
    observation_points = np.ones_like(baseline)
    observation_valid = np.array(
        [[True], [False], [False], [False], [False], [False]]
    )
    structure = np.zeros((1, 3), dtype=float)

    validation, center_ids, _, = runner._recursive_rbf_correction(
        observation_points,
        observation_valid,
        baseline,
        structure,
        fit_end_frame=2,
        query_start_frame=2,
        query_end_frame=4,
        original_count=1,
        center_count=1,
        minimum_availability_fraction=0.5,
    )
    assert validation.shape == (2, 1, 3)
    np.testing.assert_array_equal(center_ids, [0])

    correction, refit_centers, availability, reason = (
        runner._recursive_rbf_refit_or_fallback(
            observation_points,
            observation_valid,
            baseline,
            structure,
            fit_end_frame=4,
            query_start_frame=4,
            query_end_frame=6,
            original_count=1,
            center_count=1,
            minimum_availability_fraction=0.5,
        )
    )

    np.testing.assert_array_equal(correction, np.zeros((2, 1, 3)))
    assert refit_centers.size == 0
    assert availability.size == 0
    assert reason == "only 0 RBF centers meet the availability gate"


def test_recursive_rbf_refit_does_not_hide_unrelated_value_errors() -> None:
    runner = _load_runner()
    baseline = np.zeros((2, 1, 3), dtype=float)

    with np.testing.assert_raises_regex(
        ValueError,
        "RBF minimum availability must lie",
    ):
        runner._recursive_rbf_refit_or_fallback(
            baseline,
            np.ones((2, 1), dtype=bool),
            baseline,
            np.zeros((1, 3), dtype=float),
            fit_end_frame=1,
            query_start_frame=1,
            query_end_frame=2,
            original_count=1,
            center_count=1,
            minimum_availability_fraction=2.0,
        )


def test_selection_score_falls_back_to_supported_prefix_metric() -> None:
    runner = _load_runner()

    score = runner._selection_score(
        {
            "chamfer_distance_m": 0.008,
            "track_error_m": 0.0,
        },
        {
            "chamfer_distance_m": 0.010,
            "track_error_m": 0.0,
        },
        chamfer_weight=0.5,
    )

    assert score == pytest.approx(0.8)


def test_selection_score_rejects_interval_without_metric_support() -> None:
    runner = _load_runner()

    with pytest.raises(
        ValueError,
        match="selection baseline has no positive metric support",
    ):
        runner._selection_score(
            {
                "chamfer_distance_m": 0.0,
                "track_error_m": 0.0,
            },
            {
                "chamfer_distance_m": 0.0,
                "track_error_m": 0.0,
            },
        )
