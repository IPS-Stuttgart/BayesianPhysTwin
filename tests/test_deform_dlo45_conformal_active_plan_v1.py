from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "deform_dlo45_conformal_active_plan_v1"
    / "evaluate.py"
)
SPEC = importlib.util.spec_from_file_location("conformal_active_plan", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def row(*, realized: float, bound: float, nonfallback: bool = True) -> dict[str, object]:
    return {
        "dlo": "DLO4",
        "trajectory": "a.pkl",
        "current_frame": 1,
        "nonfallback": nonfallback,
        "certificate_worst_case_regret": bound,
        "normalized_realized_regret": realized,
        "physical_task_mse": 1.0,
        "fallback_task_mse": 4.0,
        "harmful_vs_fallback": False,
        "sensor_count": 2,
        "effective_hypothesis_count": 3.0,
    }


def test_score_is_trajectory_maximum_not_window_mean() -> None:
    rows = [row(realized=0.1, bound=0.05), row(realized=0.9, bound=0.1)]
    assert MODULE.trajectory_max_excess(rows) == 0.8


def test_fallback_rows_do_not_enter_base_policy_excess_score() -> None:
    rows = [
        row(realized=100.0, bound=0.0, nonfallback=False),
        row(realized=0.2, bound=0.1),
    ]
    assert MODULE.trajectory_max_excess(rows) == 0.1


def test_unavailable_split_conformal_rank_returns_infinite_radius() -> None:
    result = MODULE.split_conformal_quantile([0.1, 0.2], miscoverage=0.1)
    assert result["radius"] == "infinite"
    assert MODULE.numeric_radius(result) == math.inf


def test_order_statistic_and_finite_sample_lower_bound() -> None:
    result = MODULE.split_conformal_quantile(
        [0.1, 0.2, 0.3, 0.4], miscoverage=0.25
    )
    assert result["finite_sample_rank"] == 4
    assert result["radius"] == 0.4
    assert result["finite_sample_coverage_lower_bound"] == 0.8


def test_inflated_certificate_retains_action_or_exactly_falls_back() -> None:
    grouped = {
        ("DLO4", "a.pkl"): [
            row(realized=0.15, bound=0.02),
            row(realized=0.30, bound=0.08),
        ]
    }
    result = MODULE.evaluate_operating_point(
        grouped,
        radius=0.20,
        tolerance=0.25,
        seed=1,
        replicates=50,
    )
    assert result["nonfallback_count"] == 1
    assert result["harmful_nonfallback_count"] == 0
    assert result["regret_budget_exceed_count"] == 0
    expected_rmse = 1000.0 * np.sqrt((1.0 + 4.0) / 2.0)
    assert result["pooled_rmse_mm"] == expected_rmse


def test_state_can_remain_ambiguous_when_action_is_retained() -> None:
    grouped = {("DLO4", "a.pkl"): [row(realized=0.05, bound=0.01)]}
    result = MODULE.evaluate_operating_point(
        grouped,
        radius=0.1,
        tolerance=0.25,
        seed=1,
        replicates=20,
    )
    assert result["state_ambiguous_nonfallback_fraction"] == 1.0
