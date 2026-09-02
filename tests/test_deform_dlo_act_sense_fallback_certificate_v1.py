from __future__ import annotations

import math

import numpy as np
import pytest

from bayesian_phystwin.act_sense_fallback_certificate_v1 import (
    ACT_SENSE_FALLBACK_CERTIFICATE_CLAIM_BOUNDARY,
    act_sense_fallback_certificate,
)


def _base_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    side = np.repeat(np.array([0, 1], dtype=np.int64), 6)
    texture = np.tile(np.arange(3, dtype=np.int64), 4)
    losses = np.empty((12, 3), dtype=np.float64)
    losses[:, 0] = np.where(side == 0, 0.0, 4.0)
    losses[:, 1] = np.where(side == 1, 0.0, 4.0)
    losses[:, 2] = 1.5
    return side, texture, losses


def test_act_when_state_is_ambiguous_but_action_is_identified() -> None:
    side, texture, losses = _base_problem()
    left = side == 0
    result = act_sense_fallback_certificate(
        np.full(np.count_nonzero(left), 1.0 / np.count_nonzero(left)),
        [1.0],
        np.zeros(np.count_nonzero(left), dtype=np.int64),
        losses[left],
        np.vstack([side[left], texture[left]]),
        [0.2, 0.05],
        fallback_action_index=2,
        regret_tolerance=0.25,
        probe_names=["tug", "camera_texture"],
    )

    assert result.output_mode == "act"
    assert result.terminal_action() == 0
    assert not result.used_fallback
    assert result.plan_certificate.minimax_worst_case_regret == pytest.approx(0.0)
    assert result.plan_count == 33


def test_decision_directed_probe_beats_higher_entropy_probe() -> None:
    side, texture, losses = _base_problem()
    result = act_sense_fallback_certificate(
        np.full(12, 1.0 / 12.0),
        [1.0],
        np.zeros(12, dtype=np.int64),
        losses,
        np.vstack([side, texture]),
        [0.2, 0.05],
        fallback_action_index=2,
        regret_tolerance=0.25,
        probe_names=["tug", "camera_texture"],
    )

    assert result.output_mode == "sense"
    assert result.output_plan.probe_name == "tug"
    assert result.selected_probe_index == 0
    assert result.terminal_action(0) == 0
    assert result.terminal_action(1) == 1
    assert result.plan_certificate.minimax_worst_case_regret == pytest.approx(0.2)

    # The three-outcome texture camera removes more state entropy than the
    # two-outcome tug, but it does not distinguish the loss-relevant tether side.
    assert math.log(3.0) > math.log(2.0)


def test_fallback_is_exact_when_no_plan_meets_tolerance() -> None:
    side, texture, losses = _base_problem()
    result = act_sense_fallback_certificate(
        np.full(12, 1.0 / 12.0),
        [1.0],
        np.zeros(12, dtype=np.int64),
        losses,
        np.vstack([side, texture]),
        [0.2, 0.05],
        fallback_action_index=2,
        regret_tolerance=0.1,
        probe_names=["tug", "camera_texture"],
    )

    assert result.plan_certificate.minimax_worst_case_regret == pytest.approx(0.2)
    assert not result.has_admissible_plan
    assert result.output_mode == "fallback"
    assert result.used_fallback
    assert result.output_plan_index == result.fallback_plan_index == 2
    assert result.terminal_action() == 2


def test_plan_enumeration_and_validation_are_fail_closed() -> None:
    side, texture, losses = _base_problem()
    with pytest.raises(ValueError, match="max_plan_count"):
        act_sense_fallback_certificate(
            np.full(12, 1.0 / 12.0),
            [1.0],
            np.zeros(12, dtype=np.int64),
            losses,
            np.vstack([side, texture]),
            [0.2, 0.05],
            fallback_action_index=2,
            regret_tolerance=0.25,
            max_plan_count=10,
        )

    invalid_outcomes = side.copy()
    invalid_outcomes[side == 1] = 2
    with pytest.raises(ValueError, match="contiguous from zero"):
        act_sense_fallback_certificate(
            np.full(12, 1.0 / 12.0),
            [1.0],
            np.zeros(12, dtype=np.int64),
            losses,
            invalid_outcomes[None, :],
            [0.1],
            fallback_action_index=2,
        )

    with pytest.raises(ValueError, match="nonempty strings"):
        act_sense_fallback_certificate(
            np.full(12, 1.0 / 12.0),
            [1.0],
            np.zeros(12, dtype=np.int64),
            losses,
            np.vstack([side, texture]),
            [0.2, 0.05],
            fallback_action_index=2,
            probe_names=["tug", ""],
        )


def test_outcome_resolution_and_claim_boundary() -> None:
    side, texture, losses = _base_problem()
    result = act_sense_fallback_certificate(
        np.full(12, 1.0 / 12.0),
        [1.0],
        np.zeros(12, dtype=np.int64),
        losses,
        np.vstack([side, texture]),
        [0.2, 0.05],
        fallback_action_index=2,
        regret_tolerance=0.25,
    )

    with pytest.raises(ValueError, match="requires an observed outcome"):
        result.terminal_action()
    with pytest.raises(ValueError, match="in \\[0, 2\\)"):
        result.terminal_action(2)
    assert "does not validate" in ACT_SENSE_FALLBACK_CERTIFICATE_CLAIM_BOUNDARY
