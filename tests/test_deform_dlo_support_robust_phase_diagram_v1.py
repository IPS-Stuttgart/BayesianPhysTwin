from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.act_sense_fallback_certificate_v1 import (
    act_sense_fallback_certificate,
)
from bayesian_phystwin.support_robust_phase_diagram_v1 import (
    SUPPORT_ROBUST_PHASE_DIAGRAM_CLAIM_BOUNDARY,
    support_robust_phase_diagram,
)


def _problem() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
]:
    side = np.asarray([0, 0, 1, 1], dtype=np.int64)
    prior = np.full(4, 0.25, dtype=np.float64)
    quotient = np.asarray([1.0], dtype=np.float64)
    classes = np.zeros(4, dtype=np.int64)
    losses = np.empty((4, 3), dtype=np.float64)
    losses[:, 0] = np.where(side == 0, 0.0, 4.0)
    losses[:, 1] = np.where(side == 1, 0.0, 4.0)
    losses[:, 2] = 1.5
    outcomes = np.vstack((side, side))
    costs = np.asarray([0.0, 0.1], dtype=np.float64)
    names = ["quick_tug", "camera"]
    return prior, quotient, classes, losses, outcomes, costs, names


def _plan_box() -> tuple[
    tuple[object, ...],
    np.ndarray,
    np.ndarray,
    int,
    int,
]:
    prior, quotient, classes, losses, outcomes, costs, names = _problem()
    parent = act_sense_fallback_certificate(
        prior,
        quotient,
        classes,
        losses,
        outcomes,
        costs,
        fallback_action_index=2,
        regret_tolerance=0.25,
        probe_names=names,
    )
    quick = next(
        index
        for index, plan in enumerate(parent.plans)
        if plan.probe_name == "quick_tug"
        and tuple(plan.terminal_action_by_outcome.tolist()) == (0, 1)
    )
    camera = next(
        index
        for index, plan in enumerate(parent.plans)
        if plan.probe_name == "camera"
        and tuple(plan.terminal_action_by_outcome.tolist()) == (0, 1)
    )
    lower = np.zeros(parent.plan_count, dtype=np.float64)
    upper = np.full(parent.plan_count, 4.0, dtype=np.float64)
    upper[quick] = 3.0
    upper[camera] = 1.0
    return (
        (prior, quotient, classes, losses, outcomes, costs, names),
        lower,
        upper,
        quick,
        camera,
    )


def _diagram():
    problem, lower, upper, _, _ = _plan_box()
    prior, quotient, classes, losses, outcomes, costs, names = problem
    return support_robust_phase_diagram(
        prior,
        quotient,
        classes,
        losses,
        outcomes,
        costs,
        fallback_action_index=2,
        regret_tolerance=0.25,
        maximum_support_miss_probability=0.25,
        probe_names=names,
        unknown_plan_loss_lower=lower,
        unknown_plan_loss_upper=upper,
    )


def test_exact_phase_diagram_switches_quick_camera_fallback() -> None:
    _, _, _, quick, camera = _plan_box()
    diagram = _diagram()

    zero = diagram.decision_at(0.0)
    moderate = diagram.decision_at(0.1)
    large = diagram.decision_at(0.2)
    assert zero.output_mode == "sense"
    assert zero.output_plan_index == quick
    assert zero.selected_probe_name == "quick_tug"
    assert zero.minimax_worst_case_regret == pytest.approx(0.0)
    assert moderate.output_mode == "sense"
    assert moderate.output_plan_index == camera
    assert moderate.selected_probe_name == "camera"
    assert moderate.minimax_worst_case_regret == pytest.approx(0.19)
    assert large.output_mode == "fallback"
    assert large.output_plan_index == diagram.base_certificate.fallback_plan_index
    assert not large.has_admissible_plan


def test_breakpoints_include_exact_plan_and_fallback_transitions() -> None:
    diagram = _diagram()
    expected_probe_switch = 0.1 / 2.1
    expected_fallback_switch = (0.25 - 0.1) / 0.9

    assert np.min(np.abs(diagram.breakpoints - expected_probe_switch)) < 1e-10
    assert np.min(np.abs(diagram.breakpoints - expected_fallback_switch)) < 1e-10

    probe_point = diagram.decision_at(expected_probe_switch)
    fallback_point = diagram.decision_at(expected_fallback_switch)
    assert probe_point.selected_probe_name == "quick_tug"
    assert fallback_point.selected_probe_name == "camera"
    assert fallback_point.has_admissible_plan
    assert diagram.decision_at(expected_fallback_switch + 1e-7).output_mode == (
        "fallback"
    )


def test_every_open_phase_cell_matches_direct_formula() -> None:
    diagram = _diagram()
    for interval in diagram.interval_decisions:
        for fraction in (0.1, 0.37, 0.63, 0.9):
            epsilon = interval.support_miss_left + fraction * (
                interval.support_miss_right - interval.support_miss_left
            )
            decision = diagram.decision_at(epsilon)
            assert decision.minimax_plan_index == (interval.decision.minimax_plan_index)
            assert decision.output_plan_index == interval.decision.output_plan_index
            assert decision.output_mode == interval.decision.output_mode

            pairwise = (
                diagram.represented_pairwise_worst_case_loss_gap
                + epsilon * diagram.support_miss_pairwise_slope
            )
            np.fill_diagonal(pairwise, 0.0)
            regret = np.maximum(np.max(pairwise, axis=1), 0.0)
            assert decision.minimax_worst_case_regret == pytest.approx(
                float(np.min(regret))
            )


def test_plan_specific_maximum_support_miss_is_tight() -> None:
    _, _, _, quick, camera = _plan_box()
    diagram = _diagram()
    maximum = diagram.plan_maximum_admissible_support_miss

    assert maximum[quick] == pytest.approx(0.25 / 3.0)
    assert maximum[camera] == pytest.approx((0.25 - 0.1) / 0.9)
    assert diagram.maximum_any_plan_admissible_support_miss == pytest.approx(
        maximum[camera]
    )
    assert diagram.decision_at(maximum[camera]).has_admissible_plan
    assert not diagram.decision_at(maximum[camera] + 1e-8).has_admissible_plan


def test_terminal_action_box_is_lifted_to_complete_contingent_plans() -> None:
    prior, quotient, classes, losses, outcomes, costs, names = _problem()
    diagram = support_robust_phase_diagram(
        prior,
        quotient,
        classes,
        losses,
        outcomes[:1],
        costs[:1],
        fallback_action_index=2,
        regret_tolerance=1.0,
        maximum_support_miss_probability=0.2,
        probe_names=names[:1],
        unknown_terminal_loss_lower_by_action=[0.0, 0.5, 1.0],
        unknown_terminal_loss_upper_by_action=[2.0, 3.0, 1.5],
    )

    assert np.array_equal(diagram.unknown_plan_loss_lower[:3], [0.0, 0.5, 1.0])
    assert np.array_equal(diagram.unknown_plan_loss_upper[:3], [2.0, 3.0, 1.5])
    correct_plan = next(
        index
        for index, plan in enumerate(diagram.base_certificate.plans)
        if plan.probe_name == "quick_tug"
        and tuple(plan.terminal_action_by_outcome.tolist()) == (0, 1)
    )
    assert diagram.unknown_plan_loss_lower[correct_plan] == pytest.approx(0.0)
    assert diagram.unknown_plan_loss_upper[correct_plan] == pytest.approx(3.0)


def test_zero_miss_decision_matches_parent_certificate() -> None:
    problem, lower, upper, _, _ = _plan_box()
    prior, quotient, classes, losses, outcomes, costs, names = problem
    parent = act_sense_fallback_certificate(
        prior,
        quotient,
        classes,
        losses,
        outcomes,
        costs,
        fallback_action_index=2,
        regret_tolerance=0.25,
        probe_names=names,
    )
    diagram = support_robust_phase_diagram(
        prior,
        quotient,
        classes,
        losses,
        outcomes,
        costs,
        fallback_action_index=2,
        regret_tolerance=0.25,
        maximum_support_miss_probability=0.0,
        probe_names=names,
        unknown_plan_loss_lower=lower,
        unknown_plan_loss_upper=upper,
    )

    decision = diagram.point_decisions[0]
    assert diagram.breakpoints.tolist() == [0.0]
    assert diagram.interval_decisions == ()
    assert decision.minimax_plan_index == parent.minimax_plan_index
    assert decision.output_plan_index == parent.output_plan_index
    assert decision.output_mode == parent.output_mode


def test_inputs_outputs_and_claim_boundary_are_fail_closed() -> None:
    problem, lower, upper, _, _ = _plan_box()
    prior, quotient, classes, losses, outcomes, costs, names = problem
    with pytest.raises(ValueError, match="exactly one"):
        support_robust_phase_diagram(
            prior,
            quotient,
            classes,
            losses,
            outcomes,
            costs,
            fallback_action_index=2,
            probe_names=names,
        )
    with pytest.raises(ValueError, match="lower bounds"):
        support_robust_phase_diagram(
            prior,
            quotient,
            classes,
            losses,
            outcomes,
            costs,
            fallback_action_index=2,
            probe_names=names,
            unknown_plan_loss_lower=upper,
            unknown_plan_loss_upper=lower,
        )
    with pytest.raises(ValueError, match="maximum_phase_plan_count"):
        support_robust_phase_diagram(
            prior,
            quotient,
            classes,
            losses,
            outcomes,
            costs,
            fallback_action_index=2,
            probe_names=names,
            maximum_phase_plan_count=10,
            unknown_plan_loss_lower=lower,
            unknown_plan_loss_upper=upper,
        )

    diagram = _diagram()
    assert not diagram.breakpoints.flags.writeable
    assert not diagram.support_miss_pairwise_slope.flags.writeable
    assert not diagram.plan_maximum_admissible_support_miss.flags.writeable
    with pytest.raises(ValueError, match="exceeds"):
        diagram.decision_at(0.3)
    assert "does not estimate" in SUPPORT_ROBUST_PHASE_DIAGRAM_CLAIM_BOUNDARY
