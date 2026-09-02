from __future__ import annotations

import math

import numpy as np
import pytest

from bayesian_phystwin.conformal_regret_envelope_v1 import (
    support_robust_decision,
    trajectory_conformal_regret_envelope,
)


def test_exact_split_conformal_rank_has_exchangeable_coverage() -> None:
    scores = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5])
    covered = []
    for test_index in range(len(scores)):
        calibration = np.delete(scores, test_index)
        realized = calibration[:, None, None]
        registered = np.zeros_like(realized)
        envelope = trajectory_conformal_regret_envelope(
            realized,
            registered,
            miscoverage=0.2,
        )
        covered.append(scores[test_index] <= envelope.radius)
    assert np.mean(covered) == pytest.approx(0.8)


def test_trajectory_score_is_simultaneous_over_decisions_and_actions() -> None:
    realized = np.asarray(
        [
            [[0.1, 0.2, 0.3], [0.4, 0.1, 0.2]],
            [[0.2, 0.3, 0.4], [0.1, 0.9, 0.3]],
            [[0.4, 0.2, 0.1], [0.2, 0.3, 0.5]],
            [[0.3, 0.2, 0.2], [0.1, 0.4, 0.2]],
        ]
    )
    registered = np.full_like(realized, 0.1)
    envelope = trajectory_conformal_regret_envelope(
        realized,
        registered,
        miscoverage=0.2,
        candidate_action_mask=np.asarray([False, True, True]),
    )
    np.testing.assert_allclose(
        envelope.trajectory_nonconformity_scores,
        [0.2, 0.8, 0.4, 0.3],
    )
    assert envelope.finite_sample_rank == 4
    assert envelope.radius == pytest.approx(0.8)


def test_insufficient_calibration_returns_infinite_radius() -> None:
    realized = np.zeros((8, 2, 3))
    envelope = trajectory_conformal_regret_envelope(
        realized,
        realized,
        miscoverage=0.1,
    )
    assert envelope.finite_sample_rank == 9
    assert math.isinf(envelope.radius)
    assert not envelope.has_finite_radius


def test_envelope_never_tightens_registered_bound() -> None:
    realized = np.zeros((4, 1, 2))
    registered = np.ones_like(realized)
    envelope = trajectory_conformal_regret_envelope(
        realized,
        registered,
        miscoverage=0.2,
    )
    assert envelope.radius == 0.0


def test_support_robust_decision_selects_unique_admissible_nonfallback() -> None:
    result = support_robust_decision(
        [0.4, 0.02, 0.2],
        conformal_radius=0.03,
        regret_tolerance=0.06,
        fallback_action_index=0,
    )
    assert result.selected_action_index == 1
    assert not result.used_fallback
    np.testing.assert_allclose(result.inflated_regret_upper_bound, [0.43, 0.05, 0.23])
    np.testing.assert_array_equal(
        result.tolerance_admissible_action_mask, [False, True, False]
    )


def test_support_robust_decision_falls_back_on_tied_minimum() -> None:
    result = support_robust_decision(
        [0.5, 0.02, 0.02],
        conformal_radius=0.01,
        regret_tolerance=0.05,
    )
    assert result.selected_action_index == 0
    assert result.used_fallback
    np.testing.assert_array_equal(
        result.tolerance_admissible_action_mask, [False, True, True]
    )


def test_support_robust_decision_falls_back_for_infinite_radius() -> None:
    result = support_robust_decision(
        [0.1, 0.01, 0.02],
        conformal_radius=float("inf"),
        regret_tolerance=1.0,
    )
    assert result.selected_action_index == 0
    assert result.used_fallback


@pytest.mark.parametrize(
    ("realized", "registered"),
    [
        (np.zeros((2, 3)), np.zeros((2, 3))),
        (np.zeros((2, 3, 2)), np.zeros((2, 3, 3))),
        (np.full((2, 3, 2), np.nan), np.zeros((2, 3, 2))),
        (-np.ones((2, 3, 2)), np.zeros((2, 3, 2))),
    ],
)
def test_invalid_calibration_inputs_fail_closed(
    realized: np.ndarray, registered: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        trajectory_conformal_regret_envelope(
            realized,
            registered,
            miscoverage=0.2,
        )


@pytest.mark.parametrize("miscoverage", [0.0, 1.0, -0.1, float("nan"), True])
def test_invalid_miscoverage_is_rejected(miscoverage: object) -> None:
    with pytest.raises(ValueError):
        trajectory_conformal_regret_envelope(
            np.zeros((4, 1, 2)),
            np.zeros((4, 1, 2)),
            miscoverage=miscoverage,  # type: ignore[arg-type]
        )


def test_support_robust_decision_keeps_fallback_when_it_minimizes() -> None:
    result = support_robust_decision(
        [0.01, 0.02, 0.03],
        conformal_radius=0.01,
        regret_tolerance=0.10,
    )
    assert result.selected_action_index == 0
    assert result.used_fallback


def test_support_robust_decision_requires_fallback_in_action_mask() -> None:
    with pytest.raises(ValueError, match="include fallback_action_index"):
        support_robust_decision(
            [0.2, 0.01, 0.1],
            conformal_radius=0.0,
            regret_tolerance=0.05,
            candidate_action_mask=[False, True, True],
        )
