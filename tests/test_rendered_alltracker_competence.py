from __future__ import annotations

import numpy as np

from bayesian_phystwin.rendered_alltracker_competence import (
    covariance_diagnostics,
    evaluate_competence_gates,
    shared_support_metrics,
    trajectory_metrics,
)


def _panel() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = np.zeros((3, 2, 3), dtype=float)
    target[:, :, 0] = np.array([[0.0, 0.1], [0.01, 0.11], [0.02, 0.12]])
    prediction = target + np.array([0.001, 0.0, 0.0])
    valid = np.ones((3, 2), dtype=bool)
    return prediction, valid, target


def test_trajectory_metrics_use_exact_finite_support() -> None:
    prediction, valid, target = _panel()
    target[1, 1] = np.nan
    metrics = trajectory_metrics(prediction, valid, target)
    assert metrics["supported_count"] == 5
    assert metrics["target_count"] == 5
    assert np.isclose(metrics["position_rmse_m"], 0.001)


def test_shared_support_prevents_support_advantage() -> None:
    candidate, valid, target = _panel()
    comparator = target + 0.004
    comparator_valid = valid.copy()
    comparator_valid[0, 0] = False
    shared = shared_support_metrics(
        candidate,
        valid,
        comparator,
        comparator_valid,
        target,
    )
    assert shared["shared_count"] == 5
    assert shared["candidate_rmse_m"] < shared["comparator_rmse_m"]


def test_covariance_diagnostics_report_nees_and_coverage() -> None:
    prediction, valid, target = _panel()
    covariance = np.broadcast_to(1e-6 * np.eye(3), (3, 2, 3, 3)).copy()
    diagnostics = covariance_diagnostics(
        prediction,
        covariance,
        valid,
        target,
    )
    assert diagnostics["count"] == 6
    assert np.isclose(diagnostics["mean_nees"], 1.0)
    assert diagnostics["coverage_90"] == 1.0


def test_competence_gate_is_a_conjunction() -> None:
    candidate = {"support_fraction": 0.8, "position_rmse_m": 0.004}
    final_frame = {"position_rmse_m": 0.006}
    physical = {"candidate_relative_improvement_fraction": 0.2}
    cotracker = {"candidate_relative_improvement_fraction": 0.25}
    passed = evaluate_competence_gates(
        candidate,
        final_frame,
        physical,
        cotracker,
        minimum_support_fraction=0.5,
        maximum_position_rmse_m=0.005,
        maximum_final_frame_rmse_m=0.008,
        minimum_physical_improvement_fraction=0.1,
        minimum_cotracker_improvement_fraction=0.2,
    )
    assert passed["competence_gate_passed"]
    candidate["position_rmse_m"] = 0.006
    failed = evaluate_competence_gates(
        candidate,
        final_frame,
        physical,
        cotracker,
        minimum_support_fraction=0.5,
        maximum_position_rmse_m=0.005,
        maximum_final_frame_rmse_m=0.008,
        minimum_physical_improvement_fraction=0.1,
        minimum_cotracker_improvement_fraction=0.2,
    )
    assert not failed["competence_gate_passed"]
