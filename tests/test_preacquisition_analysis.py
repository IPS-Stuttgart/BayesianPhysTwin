from __future__ import annotations

import numpy as np
import pytest

from causal4d.preacquisition_analysis import (
    cluster_paired_bootstrap,
    cluster_robust_linear_regression,
    conformal_rank_plan,
    persistence_shrinkage_gate,
)


def test_cluster_paired_bootstrap_weights_sessions_equally() -> None:
    result = cluster_paired_bootstrap(
        candidate=[1.0, 1.0, 8.0],
        baseline=[2.0, 2.0, 10.0],
        cluster_ids=["a", "a", "b"],
        bootstrap_replicates=200,
        seed=7,
    )
    assert result["mean_difference"] == pytest.approx(-1.5)
    assert result["session_count"] == 2
    assert result["replication_unit"] == "session"


def test_persistence_shrinkage_requires_cluster_interval_below_one() -> None:
    result = persistence_shrinkage_gate(
        nominal_correction_rms_m=[1.0, 1.1, 1.0, 1.2],
        mechanism_correction_rms_m=[0.5, 0.55, 0.4, 0.5],
        cluster_ids=["a", "a", "b", "b"],
        bootstrap_replicates=200,
        seed=3,
    )
    assert result["passed"] is True
    assert result["ratio_interval_95"][1] < 1.0


def test_cluster_robust_regression_recovers_linear_coefficients() -> None:
    features = np.arange(8, dtype=float).reshape(-1, 1)
    response = 2.0 + 3.0 * features[:, 0]
    result = cluster_robust_linear_regression(
        response,
        features,
        ["a", "a", "b", "b", "c", "c", "d", "d"],
        feature_names=["speed"],
    )
    assert result["parameters"]["intercept"]["coefficient"] == pytest.approx(2.0)
    assert result["parameters"]["speed"]["coefficient"] == pytest.approx(3.0)


def test_conformal_rank_exposes_nine_session_minimum_at_90_percent() -> None:
    too_small = conformal_rank_plan(2, coverage=0.90)
    finite = conformal_rank_plan(9, coverage=0.90)
    assert too_small["finite_without_infinite_sentinel"] is False
    assert too_small["minimum_calibration_units_for_finite_interval"] == 9
    assert finite["finite_without_infinite_sentinel"] is True
    assert finite["order_statistic_rank_one_based"] == 9
