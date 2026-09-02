from __future__ import annotations

import itertools

import numpy as np
import pytest

from bayesian_phystwin.act_sense_fallback_certificate_v1 import (
    act_sense_fallback_certificate,
)
from bayesian_phystwin.support_robust_act_sense_fallback_certificate_v1 import (
    SUPPORT_ROBUST_ACT_SENSE_FALLBACK_CLAIM_BOUNDARY,
    support_robust_act_sense_fallback_certificate,
)


def _ambiguous_tether_problem(
    *, resolved_left: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    side = np.repeat(np.array([0, 1], dtype=np.int64), 2)
    losses = np.empty((4, 3), dtype=np.float64)
    losses[:, 0] = np.where(side == 0, 0.0, 4.0)
    losses[:, 1] = np.where(side == 1, 0.0, 4.0)
    losses[:, 2] = 1.5
    probes = np.vstack((side, side))
    if resolved_left:
        keep = side == 0
        losses = losses[keep]
        probes = probes[:, keep]
    hypothesis_count = losses.shape[0]
    return (
        np.full(hypothesis_count, 1.0 / hypothesis_count),
        np.array([1.0]),
        np.zeros(hypothesis_count, dtype=np.int64),
        losses,
        probes,
    )


def _certificate(*, epsilon: float, resolved_left: bool = False):
    return support_robust_act_sense_fallback_certificate(
        *_ambiguous_tether_problem(resolved_left=resolved_left),
        [0.0, 0.16],
        fallback_action_index=2,
        support_miss_probability_upper=epsilon,
        unknown_terminal_loss_lower_by_action=[0.0, 0.0, 1.5],
        unknown_terminal_loss_upper_by_action=[0.8, 0.8, 1.5],
        unknown_probe_loss_lower=[0.0, 0.16],
        unknown_probe_loss_upper=[2.0, 0.16],
        regret_tolerance=0.25,
        probe_names=["quick_tug", "camera"],
    )


def test_zero_support_miss_reproduces_exact_contingent_plan_certificate() -> None:
    problem = _ambiguous_tether_problem()
    base = act_sense_fallback_certificate(
        *problem,
        [0.0, 0.16],
        fallback_action_index=2,
        regret_tolerance=0.25,
        probe_names=["quick_tug", "camera"],
    )
    robust = _certificate(epsilon=0.0)

    np.testing.assert_allclose(
        robust.support_robust_pairwise_worst_case_loss_gap,
        base.plan_certificate.pairwise_worst_case_loss_gap,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        robust.support_robust_worst_case_regret,
        base.plan_certificate.worst_case_regret,
        rtol=0.0,
        atol=0.0,
    )
    base_plan = base.plans[base.output_plan_index]
    assert robust.output_plan.mode == base_plan.mode
    assert robust.output_plan.probe_index == base_plan.probe_index
    assert robust.output_plan.direct_action_index == base_plan.direct_action_index
    np.testing.assert_array_equal(
        robust.output_plan.terminal_action_by_outcome,
        base_plan.terminal_action_by_outcome,
    )
    assert robust.output_mode == base.output_mode == "sense"
    assert robust.output_plan.probe_name == "quick_tug"


def test_support_miss_drives_act_tug_camera_fallback_phase_diagram() -> None:
    act = _certificate(epsilon=0.1, resolved_left=True)
    tug = _certificate(epsilon=0.0)
    camera = _certificate(epsilon=0.1)
    fallback = _certificate(epsilon=0.2)

    assert act.output_mode == "act"
    assert act.terminal_action() == 0
    assert act.minimax_worst_case_regret == pytest.approx(0.08)

    assert tug.output_mode == "sense"
    assert tug.output_plan.probe_name == "quick_tug"
    assert tug.terminal_action(0) == 0
    assert tug.terminal_action(1) == 1

    assert camera.output_mode == "sense"
    assert camera.output_plan.probe_name == "camera"
    assert camera.minimax_worst_case_regret == pytest.approx(0.24)

    assert fallback.output_mode == "fallback"
    assert fallback.used_fallback
    assert fallback.output_plan_index == fallback.fallback_plan_index == 2
    assert fallback.terminal_action() == 2
    assert fallback.minimax_worst_case_regret == pytest.approx(0.32)


def test_reported_support_miss_budget_is_tight_for_selected_camera_plan() -> None:
    at_boundary = _certificate(epsilon=0.1125)
    beyond_boundary = _certificate(epsilon=0.1125001)

    assert at_boundary.output_mode == "sense"
    assert at_boundary.output_plan.probe_name == "camera"
    assert at_boundary.minimax_worst_case_regret == pytest.approx(0.25)
    assert at_boundary.selected_support_miss_budget == pytest.approx(0.1125)
    assert beyond_boundary.output_mode == "fallback"


def test_pairwise_formula_matches_exhaustive_vertices() -> None:
    prior = np.array([0.5, 0.5])
    quotient = np.array([1.0])
    classes = np.array([0, 0], dtype=np.int64)
    losses = np.array([[0.0, 1.0], [1.0, 0.0]])
    probe_outcomes = np.array([[0, 1]], dtype=np.int64)
    epsilon = 0.3
    result = support_robust_act_sense_fallback_certificate(
        prior,
        quotient,
        classes,
        losses,
        probe_outcomes,
        [0.2],
        fallback_action_index=0,
        support_miss_probability_upper=epsilon,
        unknown_terminal_loss_lower_by_action=[0.1, 0.0],
        unknown_terminal_loss_upper_by_action=[1.4, 1.2],
        unknown_probe_loss_lower=[0.1],
        unknown_probe_loss_upper=[0.5],
        regret_tolerance=1.0,
    )

    plan_lower = result.unknown_plan_loss_lower
    plan_upper = result.unknown_plan_loss_upper
    exhaustive = np.full((result.plan_count, result.plan_count), -np.inf)
    for represented_hypothesis in range(2):
        represented_losses = result.represented_certificate.plan_loss_by_hypothesis[
            represented_hypothesis
        ]
        for rho in (0.0, epsilon):
            for bits in itertools.product((0, 1), repeat=result.plan_count):
                unknown_losses = np.where(bits, plan_upper, plan_lower)
                mixture = (1.0 - rho) * represented_losses + rho * unknown_losses
                exhaustive = np.maximum(
                    exhaustive,
                    mixture[:, None] - mixture[None, :],
                )
    np.fill_diagonal(exhaustive, 0.0)
    np.testing.assert_allclose(
        result.support_robust_pairwise_worst_case_loss_gap,
        exhaustive,
        rtol=0.0,
        atol=1e-12,
    )


def test_invalid_boxes_and_arrays_fail_closed_and_outputs_are_immutable() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        support_robust_act_sense_fallback_certificate(
            *_ambiguous_tether_problem(),
            [0.0, 0.16],
            fallback_action_index=2,
            support_miss_probability_upper=0.1,
            unknown_terminal_loss_lower_by_action=[1.0, 0.0, 1.5],
            unknown_terminal_loss_upper_by_action=[0.8, 0.8, 1.5],
            unknown_probe_loss_lower=[0.0, 0.16],
            unknown_probe_loss_upper=[2.0, 0.16],
            regret_tolerance=0.25,
        )
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        _certificate(epsilon=1.01)

    result = _certificate(epsilon=0.1)
    for array in (
        result.unknown_plan_loss_lower,
        result.unknown_plan_loss_upper,
        result.unknown_pairwise_max_loss_gap,
        result.support_robust_worst_case_regret,
        result.maximum_admissible_support_miss_probability,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = 0.0
    assert "does not estimate or validate" in (
        SUPPORT_ROBUST_ACT_SENSE_FALLBACK_CLAIM_BOUNDARY
    )
