from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.selective_risk import (
    bootstrap_guard_evaluation,
    evaluate_guard,
    evaluate_guard_by_stratum,
    evaluate_matched_guards,
    evaluate_prediction_intervals,
    evaluate_prediction_intervals_by_horizon,
    selective_risk_curve,
)


def test_evaluate_guard_reports_tail_risk_and_exact_fallback() -> None:
    evaluation = evaluate_guard(
        baseline_loss=np.ones(4),
        candidate_loss=np.asarray((0.5, 2.0, 0.8, 3.0)),
        accepted=np.asarray((True, True, False, False)),
    )

    assert evaluation.observation_count == 4
    assert evaluation.accepted_count == 2
    assert evaluation.coverage == pytest.approx(0.5)
    assert evaluation.fallback_rate == pytest.approx(0.5)
    assert evaluation.baseline_mean_loss == pytest.approx(1.0)
    assert evaluation.candidate_mean_loss == pytest.approx(1.575)
    assert evaluation.selected_mean_loss == pytest.approx(1.125)
    assert evaluation.selected_mean_excess_loss == pytest.approx(0.125)
    assert evaluation.accepted_mean_excess_loss == pytest.approx(0.25)
    assert evaluation.accepted_high_quantile_excess_loss == pytest.approx(0.925)
    assert evaluation.harmful_accepted_rate == pytest.approx(0.5)
    assert evaluation.worst_accepted_excess_loss == pytest.approx(1.0)


def test_no_acceptance_returns_the_baseline_exactly() -> None:
    baseline = np.asarray((1.0, 2.0, 3.0))
    evaluation = evaluate_guard(
        baseline,
        np.asarray((0.5, 4.0, 2.5)),
        np.zeros(3, dtype=bool),
    )

    assert evaluation.selected_mean_loss == pytest.approx(np.mean(baseline))
    assert evaluation.selected_mean_excess_loss == pytest.approx(0.0)
    assert evaluation.coverage == 0.0
    assert evaluation.fallback_rate == 1.0
    assert evaluation.accepted_mean_excess_loss is None
    assert evaluation.accepted_high_quantile_excess_loss is None
    assert evaluation.harmful_accepted_rate is None
    assert evaluation.worst_accepted_excess_loss is None


def test_selective_curve_includes_zero_coverage_and_preserves_ties() -> None:
    baseline = np.ones(4)
    candidate = np.asarray((0.5, 1.2, 0.8, 2.0))
    scores = np.asarray((0.9, 0.9, 0.5, 0.1))

    higher = selective_risk_curve(baseline, candidate, scores)
    assert [point.threshold for point in higher] == [None, 0.9, 0.5, 0.1]
    assert [point.evaluation.accepted_count for point in higher] == [0, 2, 3, 4]
    assert [point.evaluation.coverage for point in higher] == pytest.approx(
        (0.0, 0.5, 0.75, 1.0)
    )
    assert higher[0].evaluation.selected_mean_excess_loss == 0.0

    lower = selective_risk_curve(
        baseline,
        candidate,
        scores,
        higher_is_safer=False,
    )
    assert [point.threshold for point in lower] == [None, 0.1, 0.5, 0.9]
    assert [point.evaluation.accepted_count for point in lower] == [0, 1, 2, 4]


def test_curve_can_omit_zero_coverage_endpoint() -> None:
    points = selective_risk_curve(
        np.ones(2),
        np.asarray((0.5, 1.5)),
        np.asarray((0.9, 0.1)),
        include_zero_coverage=False,
    )

    assert [point.threshold for point in points] == [0.9, 0.1]


def test_harmful_tolerance_and_quantile_are_predeclared_controls() -> None:
    evaluation = evaluate_guard(
        baseline_loss=np.ones(3),
        candidate_loss=np.asarray((1.0, 1.01, 1.2)),
        accepted=np.ones(3, dtype=bool),
        harmful_tolerance=0.05,
        high_quantile=0.5,
    )

    assert evaluation.harmful_accepted_rate == pytest.approx(1.0 / 3.0)
    assert evaluation.accepted_high_quantile_excess_loss == pytest.approx(0.01)
    assert evaluation.worst_accepted_excess_loss == pytest.approx(0.2)


def test_guard_strata_support_reliability_and_identifiable_rank_analysis() -> None:
    strata = evaluate_guard_by_stratum(
        baseline_loss=np.ones(6),
        candidate_loss=np.asarray((0.5, 1.4, 0.8, 0.9, 1.6, 0.7)),
        accepted=np.asarray((True, True, True, False, True, False)),
        stratum_ids=np.asarray(("high", "high", "medium", "medium", "low", "low")),
    )

    assert [item.stratum for item in strata] == ["high", "medium", "low"]
    assert [item.evaluation.coverage for item in strata] == pytest.approx(
        (1.0, 0.5, 0.5)
    )
    assert strata[0].evaluation.harmful_accepted_rate == pytest.approx(0.5)
    assert strata[2].evaluation.selected_mean_excess_loss == pytest.approx(0.3)


def test_matched_guards_share_the_same_fallback_and_loss_units() -> None:
    comparison = evaluate_matched_guards(
        baseline_loss=np.ones(4),
        candidate_losses={
            "bayesian": np.asarray((0.5, 2.0, 0.8, 3.0)),
            "last_residual": np.asarray((0.7, 1.1, 0.7, 1.4)),
        },
        accepted_by_method={
            "bayesian": np.asarray((True, False, True, False)),
            "last_residual": np.asarray((True, True, True, False)),
        },
        reference_method="bayesian",
    )

    methods = {item.method: item for item in comparison.methods}
    assert comparison.reference_method == "bayesian"
    assert methods["bayesian"].evaluation.selected_mean_loss == pytest.approx(0.825)
    assert methods["bayesian"].selected_mean_loss_difference_from_reference == 0.0
    assert methods["last_residual"].evaluation.selected_mean_loss == pytest.approx(
        0.875
    )
    assert methods[
        "last_residual"
    ].selected_mean_loss_difference_from_reference == pytest.approx(0.05)


def test_prediction_interval_metrics_report_calibration_and_sharpness() -> None:
    evaluation = evaluate_prediction_intervals(
        target=np.asarray((0.0, 1.0, 2.0, 3.0)),
        lower=np.asarray((-0.5, 0.5, 2.1, 2.0)),
        upper=np.asarray((0.5, 1.5, 2.6, 2.9)),
        nominal_coverage=0.5,
    )

    assert evaluation.observation_count == 4
    assert evaluation.empirical_coverage == pytest.approx(0.5)
    assert evaluation.coverage_error == pytest.approx(0.0)
    assert evaluation.mean_interval_width == pytest.approx(0.85)
    assert evaluation.median_interval_width == pytest.approx(0.95)
    assert evaluation.below_interval_rate == pytest.approx(0.25)
    assert evaluation.above_interval_rate == pytest.approx(0.25)


def test_prediction_interval_metrics_are_reported_by_horizon() -> None:
    results = evaluate_prediction_intervals_by_horizon(
        target=np.asarray((0.0, 1.0, 2.0, 3.0)),
        lower=np.asarray((-0.5, 0.5, 1.5, 3.1)),
        upper=np.asarray((0.5, 1.5, 2.5, 3.5)),
        horizon=np.asarray((1.0, 1.0, 2.0, 2.0)),
        nominal_coverage=0.75,
    )

    assert [item.horizon for item in results] == [1.0, 2.0]
    assert results[0].evaluation.empirical_coverage == 1.0
    assert results[1].evaluation.empirical_coverage == 0.5
    assert results[1].evaluation.mean_interval_width == pytest.approx(0.7)


def test_cluster_bootstrap_is_reproducible_and_keeps_group_units() -> None:
    baseline = np.ones(6)
    candidate = np.asarray((0.5, 0.8, 1.5, 1.2, 0.7, 1.1))
    accepted = np.asarray((True, True, True, False, False, True))
    groups = np.asarray(("a", "a", "b", "b", "c", "c"))

    first = bootstrap_guard_evaluation(
        baseline,
        candidate,
        accepted,
        groups,
        bootstrap_repeats=200,
        seed=17,
    )
    second = bootstrap_guard_evaluation(
        baseline,
        candidate,
        accepted,
        groups,
        bootstrap_repeats=200,
        seed=17,
    )

    assert first == second
    assert first.group_count == 3
    assert first.point_estimate.accepted_count == 4
    assert first.coverage.finite_replicates == 200
    assert first.selected_mean_excess_loss.lower is not None
    assert first.selected_mean_excess_loss.upper is not None
    assert first.accepted_mean_excess_loss.estimate == pytest.approx(-0.025)
    assert first.accepted_high_quantile_excess_loss.lower is not None


def test_bootstrap_retains_replicates_with_no_accepted_rows() -> None:
    summary = bootstrap_guard_evaluation(
        baseline_loss=np.ones(4),
        candidate_loss=np.asarray((0.5, 0.8, 1.2, 1.3)),
        accepted=np.asarray((True, True, False, False)),
        group_ids=np.asarray(("accepted", "accepted", "fallback", "fallback")),
        bootstrap_repeats=400,
        seed=3,
    )

    assert summary.coverage.finite_replicates == 400
    assert summary.selected_mean_excess_loss.finite_replicates == 400
    assert 0 < summary.accepted_mean_excess_loss.finite_replicates < 400
    assert (
        summary.accepted_high_quantile_excess_loss.finite_replicates
        == summary.accepted_mean_excess_loss.finite_replicates
    )
    assert summary.accepted_mean_excess_loss.lower is not None


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            (np.ones(2), np.ones(3), np.ones(2, dtype=bool)),
            "candidate_loss",
        ),
        (
            (np.ones(2), np.ones(2), np.asarray((1, 0))),
            "Boolean vector",
        ),
        (
            (np.ones(2), np.asarray((1.0, np.nan)), np.ones(2, dtype=bool)),
            "finite",
        ),
    ],
)
def test_guard_rejects_invalid_vectors(arguments, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_guard(*arguments)


def test_evaluation_rejects_invalid_control_arguments() -> None:
    baseline = np.ones(2)
    candidate = np.ones(2)
    accepted = np.ones(2, dtype=bool)

    with pytest.raises(ValueError, match="higher_is_safer"):
        selective_risk_curve(
            baseline,
            candidate,
            np.ones(2),
            higher_is_safer=1,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="include_zero_coverage"):
        selective_risk_curve(
            baseline,
            candidate,
            np.ones(2),
            include_zero_coverage=1,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="harmful_tolerance"):
        evaluate_guard(baseline, candidate, accepted, harmful_tolerance=-1.0)
    with pytest.raises(ValueError, match="high_quantile"):
        evaluate_guard(baseline, candidate, accepted, high_quantile=0.0)
    with pytest.raises(ValueError, match="same keys"):
        evaluate_matched_guards(
            baseline,
            {"bayesian": candidate},
            {"other": accepted},
            reference_method="bayesian",
        )
    with pytest.raises(ValueError, match="reference_method"):
        evaluate_matched_guards(
            baseline,
            {"bayesian": candidate},
            {"bayesian": accepted},
            reference_method="missing",
        )


def test_interval_and_bootstrap_validation_is_fail_closed() -> None:
    baseline = np.ones(2)
    candidate = np.ones(2)
    accepted = np.ones(2, dtype=bool)

    with pytest.raises(ValueError, match="lower must not exceed upper"):
        evaluate_prediction_intervals(
            np.asarray((0.0, 1.0)),
            np.asarray((0.5, 0.0)),
            np.asarray((0.4, 2.0)),
        )
    with pytest.raises(ValueError, match="nominal_coverage"):
        evaluate_prediction_intervals(
            baseline,
            baseline,
            candidate,
            nominal_coverage=1.0,
        )
    with pytest.raises(ValueError, match="horizon must be nonnegative"):
        evaluate_prediction_intervals_by_horizon(
            baseline,
            baseline,
            candidate,
            np.asarray((0.0, -1.0)),
        )
    with pytest.raises(ValueError, match="bootstrap_repeats"):
        bootstrap_guard_evaluation(
            baseline,
            candidate,
            accepted,
            np.asarray((0, 1)),
            bootstrap_repeats=0,
        )
    with pytest.raises(ValueError, match="confidence_level"):
        bootstrap_guard_evaluation(
            baseline,
            candidate,
            accepted,
            np.asarray((0, 1)),
            confidence_level=1.0,
        )
    with pytest.raises(ValueError, match="group_ids"):
        bootstrap_guard_evaluation(
            baseline,
            candidate,
            accepted,
            np.asarray((0,)),
        )
