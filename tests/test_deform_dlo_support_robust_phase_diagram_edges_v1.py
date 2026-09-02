from __future__ import annotations

import math

import numpy as np
import pytest

import bayesian_phystwin.support_robust_phase_diagram_v1 as phase


def _direct_problem(*, fallback_best: bool) -> tuple[object, ...]:
    prior = np.asarray([0.5, 0.5], dtype=np.float64)
    quotient = np.asarray([1.0], dtype=np.float64)
    classes = np.asarray([0, 0], dtype=np.int64)
    if fallback_best:
        losses = np.asarray([[2.0, 1.0, 0.0], [2.0, 1.0, 0.0]])
    else:
        losses = np.asarray([[0.0, 2.0, 1.0], [0.0, 2.0, 1.0]])
    outcomes = np.asarray([[0, 0]], dtype=np.int64)
    costs = np.asarray([10.0], dtype=np.float64)
    return prior, quotient, classes, losses, outcomes, costs


def test_scalar_and_vector_contracts_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="finite nonnegative"):
        phase._finite_nonnegative_real(True, name="value")
    with pytest.raises(ValueError, match="finite nonnegative"):
        phase._finite_nonnegative_real(float("inf"), name="value")
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        phase._probability(1.01, name="epsilon")
    with pytest.raises(ValueError, match="positive integer"):
        phase._positive_integer(False, name="cap")
    with pytest.raises(ValueError, match="positive integer"):
        phase._positive_integer(0, name="cap")
    with pytest.raises(ValueError, match="real numeric"):
        phase._nonnegative_vector(["bad"], size=1, name="vector")
    with pytest.raises(ValueError, match="exactly 1"):
        phase._nonnegative_vector([0.0, 1.0], size=1, name="vector")
    with pytest.raises(ValueError, match="finite nonnegative"):
        phase._nonnegative_vector([np.nan], size=1, name="vector")
    with pytest.raises(ValueError, match="finite nonnegative"):
        phase._nonnegative_vector([-1.0], size=1, name="vector")


def test_summary_methods_cover_complete_phase_record() -> None:
    prior, quotient, classes, losses, outcomes, costs = _direct_problem(
        fallback_best=False
    )
    diagram = phase.support_robust_phase_diagram(
        prior,
        quotient,
        classes,
        losses,
        outcomes,
        costs,
        fallback_action_index=2,
        regret_tolerance=0.25,
        maximum_support_miss_probability=0.5,
        probe_names=["expensive_probe"],
        unknown_terminal_loss_lower_by_action=[0.0, 2.0, 1.0],
        unknown_terminal_loss_upper_by_action=[0.0, 2.0, 1.0],
    )

    summary = diagram.summary()
    assert summary["version"] == phase.SUPPORT_ROBUST_PHASE_DIAGRAM_VERSION
    assert summary["semantics"] == phase.SUPPORT_ROBUST_PHASE_DIAGRAM_SEMANTICS
    assert summary["plan_count"] == diagram.plan_count
    assert summary["point_decisions"]
    assert len(summary["interval_decisions"]) == len(diagram.interval_decisions)
    assert all(item["open_interval"] for item in summary["interval_decisions"])


def test_admissible_direct_action_and_direct_fallback_are_distinguished() -> None:
    prior, quotient, classes, losses, outcomes, costs = _direct_problem(
        fallback_best=False
    )
    act = phase.support_robust_phase_diagram(
        prior,
        quotient,
        classes,
        losses,
        outcomes,
        costs,
        fallback_action_index=2,
        regret_tolerance=0.0,
        maximum_support_miss_probability=0.1,
        probe_names=["expensive_probe"],
        unknown_terminal_loss_lower_by_action=[0.0, 2.0, 1.0],
        unknown_terminal_loss_upper_by_action=[0.0, 2.0, 1.0],
    ).decision_at(0.0)
    assert act.has_admissible_plan
    assert act.output_mode == "act"
    assert not act.used_fallback
    assert act.selected_probe_index is None

    prior, quotient, classes, losses, outcomes, costs = _direct_problem(
        fallback_best=True
    )
    fallback = phase.support_robust_phase_diagram(
        prior,
        quotient,
        classes,
        losses,
        outcomes,
        costs,
        fallback_action_index=2,
        regret_tolerance=0.0,
        maximum_support_miss_probability=0.1,
        probe_names=["expensive_probe"],
        unknown_terminal_loss_lower_by_action=[2.0, 1.0, 0.0],
        unknown_terminal_loss_upper_by_action=[2.0, 1.0, 0.0],
    ).decision_at(0.0)
    assert fallback.has_admissible_plan
    assert fallback.output_mode == "fallback"
    assert fallback.used_fallback


def test_partial_loss_boxes_fail_closed() -> None:
    prior, quotient, classes, losses, outcomes, costs = _direct_problem(
        fallback_best=False
    )
    with pytest.raises(ValueError, match="both required"):
        phase.support_robust_phase_diagram(
            prior,
            quotient,
            classes,
            losses,
            outcomes,
            costs,
            fallback_action_index=2,
            unknown_plan_loss_lower=np.zeros(6),
        )
    with pytest.raises(ValueError, match="both required"):
        phase.support_robust_phase_diagram(
            prior,
            quotient,
            classes,
            losses,
            outcomes,
            costs,
            fallback_action_index=2,
            unknown_terminal_loss_lower_by_action=[0.0, 0.0, 0.0],
        )
    with pytest.raises(ValueError, match="maximum_breakpoint_count"):
        phase.support_robust_phase_diagram(
            prior,
            quotient,
            classes,
            losses,
            outcomes,
            costs,
            fallback_action_index=2,
            maximum_breakpoint_count=0,
            unknown_terminal_loss_lower_by_action=[0.0, 2.0, 1.0],
            unknown_terminal_loss_upper_by_action=[0.0, 2.0, 1.0],
        )


def test_upper_envelope_handles_duplicate_slopes_and_hull_pops() -> None:
    duplicate = phase._upper_envelope_segments(
        np.asarray([1.0, 2.0, 2.0]),
        np.asarray([0.0, 0.0, 0.0]),
        0.5,
    )
    assert len(duplicate) == 1
    assert duplicate[0].benchmark_plan_index == 1
    assert duplicate[0].intercept == 2.0

    popped = phase._upper_envelope_segments(
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.0, 1.0, 2.0]),
        0.25,
    )
    assert len(popped) == 1
    assert popped[0].benchmark_plan_index == 0
    assert popped[0].left == 0.0
    assert popped[0].right == 0.25


def test_breakpoint_deduplication_clips_and_preserves_endpoints() -> None:
    result = phase._deduplicate_breakpoints(
        [-1.0, 1e-13, 0.2, 0.2 + 5e-13, 0.7],
        maximum_epsilon=0.5,
    )
    np.testing.assert_allclose(result, [0.0, 0.2, 0.5], atol=1e-12, rtol=0.0)
    assert not result.flags.writeable


def test_breakpoint_caps_fail_closed() -> None:
    segment = phase._EnvelopeSegment(0.25, 1.0, 0.0, 1.0, 0)
    with pytest.raises(ValueError, match="phase-breakpoint count"):
        phase._phase_breakpoints(
            ((segment,),),
            tolerance=0.4,
            maximum_epsilon=1.0,
            maximum_breakpoint_count=2,
        )

    envelopes = tuple(
        (phase._EnvelopeSegment(0.0, 1.0, float(index), float(index + 1), 0),)
        for index in range(5)
    )
    with pytest.raises(ValueError, match="candidate phase-breakpoint"):
        phase._phase_breakpoints(
            envelopes,
            tolerance=0.5,
            maximum_epsilon=1.0,
            maximum_breakpoint_count=1,
        )


def test_maximum_admissible_epsilon_covers_all_boundary_modes() -> None:
    above_at_zero = (
        phase._EnvelopeSegment(0.0, 1.0, 0.3, 0.0, 0),
    )
    assert math.isnan(
        phase._maximum_admissible_epsilon(
            above_at_zero,
            tolerance=0.25,
            maximum_epsilon=1.0,
        )
    )

    admissible_everywhere = (
        phase._EnvelopeSegment(0.0, 0.5, 0.1, 0.1, 0),
    )
    assert phase._maximum_admissible_epsilon(
        admissible_everywhere,
        tolerance=0.2,
        maximum_epsilon=0.5,
    ) == pytest.approx(0.5)

    crossing = (
        phase._EnvelopeSegment(0.0, 1.0, 0.1, 0.5, 0),
    )
    assert phase._maximum_admissible_epsilon(
        crossing,
        tolerance=0.3,
        maximum_epsilon=1.0,
    ) == pytest.approx(0.4)

    discontinuous_defensive_partition = (
        phase._EnvelopeSegment(0.0, 0.4, 0.1, 0.0, 0),
        phase._EnvelopeSegment(0.4, 1.0, 0.4, 0.0, 1),
    )
    assert phase._maximum_admissible_epsilon(
        discontinuous_defensive_partition,
        tolerance=0.25,
        maximum_epsilon=1.0,
    ) == pytest.approx(0.4)
