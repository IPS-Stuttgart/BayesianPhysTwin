from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.selective_risk import (
    bootstrap_guard_evaluation,
    evaluate_guard,
    selective_risk_curve,
)


def test_evaluate_guard_reports_exact_fallback_and_harmful_updates() -> None:
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
    assert evaluation.harmful_accepted_rate is None
    assert evaluation.worst_accepted_excess_loss is None


def test_selective_curve_preserves_ties_and_supports_both_score_directions() -> None:
    baseline = np.ones(4)
    candidate = np.asarray((0.5, 1.2, 0.8, 2.0))
    scores = np.asarray((0.9, 0.9, 0.5, 0.1))

    higher = selective_risk_curve(baseline, candidate, scores)
    assert [point.threshold for point in higher] == [0.9, 0.5, 0.1]
    assert [point.evaluation.accepted_count for point in higher] == [2, 3, 4]
    assert [point.evaluation.coverage for point in higher] == pytest.approx(
        (0.5, 0.75, 1.0)
    )

    lower = selective_risk_curve(
        baseline,
        candidate,
        scores,
        higher_is_safer=False,
    )
    assert [point.threshold for point in lower] == [0.1, 0.5, 0.9]
    assert [point.evaluation.accepted_count for point in lower] == [1, 2, 4]


def test_harmful_tolerance_is_applied_only_to_positive_excess() -> None:
    evaluation = evaluate_guard(
        baseline_loss=np.ones(3),
        candidate_loss=np.asarray((1.0, 1.01, 1.2)),
        accepted=np.ones(3, dtype=bool),
        harmful_tolerance=0.05,
    )

    assert evaluation.harmful_accepted_rate == pytest.approx(1.0 / 3.0)
    assert evaluation.worst_accepted_excess_loss == pytest.approx(0.2)


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
    assert first.accepted_mean_excess_loss.estimate == pytest.approx(0.225)


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


def test_curve_and_bootstrap_reject_invalid_control_arguments() -> None:
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
    with pytest.raises(ValueError, match="harmful_tolerance"):
        evaluate_guard(baseline, candidate, accepted, harmful_tolerance=-1.0)
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
