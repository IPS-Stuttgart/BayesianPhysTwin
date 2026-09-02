from __future__ import annotations

import math

import numpy as np
import pytest

from bayesian_phystwin.act_sense_fallback_certificate_v1 import (
    act_sense_fallback_certificate,
)
from bayesian_phystwin.conformal_complete_plan_certificate_v1 import (
    CONFORMAL_COMPLETE_PLAN_CERTIFICATE_CLAIM_BOUNDARY,
    complete_plan_regret_tensor,
    conformal_act_sense_fallback_decision,
    scaled_trajectory_conformal_plan_envelope,
    support_robust_plan_width_scales,
)
from bayesian_phystwin.support_robust_act_sense_fallback_certificate_v1 import (
    support_robust_act_sense_fallback_certificate,
)


def _base_certificate():
    return act_sense_fallback_certificate(
        [0.5, 0.5],
        [1.0],
        [0, 0],
        [[0.0, 2.0, 1.0], [2.0, 0.0, 1.0]],
        [[0, 1]],
        [0.25],
        fallback_action_index=2,
        regret_tolerance=1.0,
        probe_names=["diagnostic"],
    )


def _support_certificate(*, epsilon: float = 0.05):
    side = np.repeat(np.array([0, 1], dtype=np.int64), 2)
    losses = np.empty((4, 3), dtype=np.float64)
    losses[:, 0] = np.where(side == 0, 0.0, 4.0)
    losses[:, 1] = np.where(side == 1, 0.0, 4.0)
    losses[:, 2] = 1.5
    return support_robust_act_sense_fallback_certificate(
        np.full(4, 0.25),
        [1.0],
        np.zeros(4, dtype=np.int64),
        losses,
        np.vstack((side, side)),
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


def _plan_index(certificate, *, probe_name: str, mapping: tuple[int, ...]) -> int:
    for index, plan in enumerate(certificate.plans):
        if (
            plan.probe_name == probe_name
            and tuple(plan.terminal_action_by_outcome.tolist()) == mapping
        ):
            return index
    raise AssertionError("requested contingent plan was not enumerated")


def _envelope_with_score(certificate, score: float):
    scales = support_robust_plan_width_scales(
        certificate,
        minimum_scale=0.1,
    )
    registered = np.broadcast_to(
        certificate.support_robust_worst_case_regret,
        (4, 1, certificate.plan_count),
    ).copy()
    realized = registered + score * scales[None, None, :]
    return scaled_trajectory_conformal_plan_envelope(
        realized,
        registered,
        scales,
        miscoverage=0.25,
    )


def test_complete_plan_losses_apply_frozen_outcome_map() -> None:
    certificate = _base_certificate()
    terminal_losses = np.asarray(
        [[[0.0, 2.0, 1.0]], [[2.0, 0.0, 1.0]]],
        dtype=np.float64,
    )
    outcomes = np.asarray([[[0]], [[1]]], dtype=np.int64)
    result = complete_plan_regret_tensor(
        certificate,
        terminal_losses,
        outcomes,
        [0.25],
    )

    adaptive = _plan_index(
        certificate,
        probe_name="diagnostic",
        mapping=(0, 1),
    )
    ignores_outcome = _plan_index(
        certificate,
        probe_name="diagnostic",
        mapping=(0, 0),
    )
    assert result.plan_loss_by_trajectory_decision_plan[:, 0, adaptive].tolist() == (
        pytest.approx([0.25, 0.25])
    )
    assert result.realized_regret_by_trajectory_decision_plan[
        :, 0, adaptive
    ].tolist() == pytest.approx([0.25, 0.25])
    assert result.plan_loss_by_trajectory_decision_plan[
        :, 0, ignores_outcome
    ].tolist() == pytest.approx([0.25, 2.25])
    assert result.best_plan_loss_by_trajectory_decision[:, 0].tolist() == (
        pytest.approx([0.0, 0.0])
    )
    assert result.trajectory_count == 2
    assert result.decision_count == 1
    assert result.plan_count == 12
    with pytest.raises(ValueError, match="unregistered outcome"):
        complete_plan_regret_tensor(
            certificate,
            terminal_losses,
            np.asarray([[[0]], [[2]]]),
            [0.25],
        )


def test_scaled_trajectory_envelope_uses_simultaneous_order_statistic() -> None:
    realized = np.zeros((4, 2, 3), dtype=np.float64)
    for trajectory, score in enumerate((0.1, 0.2, 0.3, 0.4)):
        realized[trajectory, 0, 0] = score
    envelope = scaled_trajectory_conformal_plan_envelope(
        realized,
        np.zeros_like(realized),
        [1.0, 2.0, 4.0],
        miscoverage=0.2,
    )

    assert envelope.order_statistic_rank == 4
    assert envelope.score_quantile == pytest.approx(0.4)
    assert envelope.inflation_by_plan.tolist() == pytest.approx([0.4, 0.8, 1.6])
    assert envelope.finite_sample_coverage_lower_bound == pytest.approx(0.8)
    assert envelope.trajectory_scores.tolist() == pytest.approx(
        [0.1, 0.2, 0.3, 0.4]
    )
    assert not envelope.inflation_by_plan.flags.writeable
    assert not envelope.trajectory_scores.flags.writeable


def test_small_calibration_sample_fails_closed_with_infinite_envelope() -> None:
    envelope = scaled_trajectory_conformal_plan_envelope(
        np.zeros((2, 1, 3)),
        np.zeros((2, 1, 3)),
        [1.0, 1.0, 1.0],
        miscoverage=0.1,
    )
    assert envelope.order_statistic_rank == 3
    assert math.isinf(envelope.score_quantile)
    assert np.isinf(envelope.inflation_by_plan).all()
    assert envelope.finite_sample_coverage_lower_bound == pytest.approx(1.0)


def test_scaled_calibration_switches_fragile_probe_then_falls_back() -> None:
    certificate = _support_certificate(epsilon=0.05)

    quick = conformal_act_sense_fallback_decision(
        certificate,
        _envelope_with_score(certificate, 0.0),
    )
    assert quick.output_mode == "sense"
    assert quick.output_plan.probe_name == "quick_tug"
    assert quick.terminal_action(0) == 0
    assert quick.terminal_action(1) == 1

    camera = conformal_act_sense_fallback_decision(
        certificate,
        _envelope_with_score(certificate, 0.05),
    )
    assert camera.output_mode == "sense"
    assert camera.output_plan.probe_name == "camera"
    assert camera.calibrated_regret_upper_by_plan[camera.output_plan_index] == (
        pytest.approx(0.24)
    )

    fallback = conformal_act_sense_fallback_decision(
        certificate,
        _envelope_with_score(certificate, 0.10),
    )
    assert fallback.output_mode == "fallback"
    assert fallback.terminal_action() == 2
    assert fallback.fallback_reason == "calibrated-regret-exceeds-tolerance"


def test_nonunique_calibrated_minimizer_returns_exact_fallback() -> None:
    certificate = act_sense_fallback_certificate(
        [0.5, 0.5],
        [1.0],
        [0, 0],
        np.zeros((2, 3)),
        np.empty((0, 2), dtype=np.int64),
        [],
        fallback_action_index=2,
        regret_tolerance=0.0,
    )
    envelope = scaled_trajectory_conformal_plan_envelope(
        np.zeros((4, 1, 3)),
        np.zeros((4, 1, 3)),
        [1.0, 1.0, 1.0],
        miscoverage=0.25,
    )
    decision = conformal_act_sense_fallback_decision(certificate, envelope)

    assert decision.minimizer_count == 3
    assert decision.output_mode == "fallback"
    assert decision.output_plan_index == certificate.fallback_plan_index
    assert decision.terminal_action() == 2
    assert decision.fallback_reason == "nonunique-calibrated-minimax-plan"


def test_validation_and_claim_boundary_are_fail_closed() -> None:
    certificate = _support_certificate()
    scales = support_robust_plan_width_scales(certificate, minimum_scale=0.1)
    assert np.all(scales > 0.0)
    assert not scales.flags.writeable

    with pytest.raises(ValueError, match="strictly positive"):
        support_robust_plan_width_scales(certificate, minimum_scale=0.0)
    with pytest.raises(ValueError, match="equal shape"):
        scaled_trajectory_conformal_plan_envelope(
            np.zeros((4, 1, certificate.plan_count)),
            np.zeros((4, 2, certificate.plan_count)),
            scales,
            miscoverage=0.25,
        )
    with pytest.raises(ValueError, match="booleans"):
        scaled_trajectory_conformal_plan_envelope(
            np.zeros((4, 1, certificate.plan_count)),
            np.zeros((4, 1, certificate.plan_count)),
            scales,
            miscoverage=0.25,
            candidate_plan_mask=np.ones(certificate.plan_count, dtype=np.int64),
        )
    with pytest.raises(ValueError, match=r"in \(0, 1\)"):
        scaled_trajectory_conformal_plan_envelope(
            np.zeros((4, 1, certificate.plan_count)),
            np.zeros((4, 1, certificate.plan_count)),
            scales,
            miscoverage=0.0,
        )
    assert "trajectory" in CONFORMAL_COMPLETE_PLAN_CERTIFICATE_CLAIM_BOUNDARY
    assert "does not validate" in (
        CONFORMAL_COMPLETE_PLAN_CERTIFICATE_CLAIM_BOUNDARY
    )
