from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.support_robust_decision_certificate_v1 import (
    SUPPORT_ROBUST_DECISION_CERTIFICATE_SEMANTICS,
    split_conformal_trajectory_envelope,
    support_robust_action_decision,
    trajectory_policy_regret_excess,
)


def test_split_conformal_uses_finite_sample_correct_order_statistic() -> None:
    envelope = split_conformal_trajectory_envelope(
        [0.3, 0.1, 0.4, 0.2],
        miscoverage=0.4,
    )

    assert envelope.order_statistic == 3
    assert envelope.radius == pytest.approx(0.3)
    assert envelope.finite
    assert envelope.nominal_coverage == pytest.approx(0.6)
    assert envelope.summary()["calibration_unit"] == "complete_trajectory"
    assert envelope.summary()["semantics"] == SUPPORT_ROBUST_DECISION_CERTIFICATE_SEMANTICS
    assert not envelope.calibration_scores.flags.writeable


def test_too_few_calibration_trajectories_produces_infinite_radius() -> None:
    envelope = split_conformal_trajectory_envelope(
        np.linspace(0.0, 0.7, 8),
        miscoverage=0.1,
    )

    assert envelope.order_statistic == 9
    assert np.isinf(envelope.radius)
    assert not envelope.finite


def test_exact_exchangeable_rank_enumeration_meets_nominal_coverage() -> None:
    # Four calibration scores and one exchangeable test score. At alpha=.4,
    # k=3 and exactly 3/5 possible test ranks are covered.
    scores = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5])
    covered = []
    for test_index in range(scores.size):
        calibration = np.delete(scores, test_index)
        envelope = split_conformal_trajectory_envelope(calibration, miscoverage=0.4)
        covered.append(scores[test_index] <= envelope.radius)

    assert np.mean(covered) == pytest.approx(0.6)


def test_trajectory_score_uses_only_nonfallback_base_decisions() -> None:
    losses = np.asarray(
        [
            [0.0, 2.0, 1.0],
            [3.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
        ]
    )
    bounds = np.asarray(
        [
            [0.0, 0.5, 0.2],
            [0.0, 0.1, 0.1],
            [0.0, 0.4, 0.2],
        ]
    )
    # Decision 0 selects action 1 and has excess 2-.5=1.5. Decision 1 is
    # fallback and is ignored even though fallback has realized regret 3.
    # Decision 2 selects action 2 and has excess 1-.2=.8.
    result = trajectory_policy_regret_excess(
        losses,
        bounds,
        [1, 0, 2],
        fallback_action_index=0,
    )

    np.testing.assert_allclose(result.realized_regret, [2.0, 3.0, 1.0])
    np.testing.assert_allclose(result.finite_support_regret_bound, [0.5, 0.0, 0.2])
    np.testing.assert_allclose(result.regret_excess, [1.5, 0.0, 0.8])
    np.testing.assert_array_equal(result.nonfallback_mask, [True, False, True])
    assert result.nonfallback_count == 2
    assert result.score == pytest.approx(1.5)
    assert not result.regret_excess.flags.writeable


def test_all_fallback_trajectory_has_zero_nonconformity() -> None:
    result = trajectory_policy_regret_excess(
        [[2.0, 0.0], [1.0, 0.0]],
        [[0.0, 0.1], [0.0, 0.1]],
        [0, 0],
        fallback_action_index=0,
    )

    assert result.score == 0.0
    assert result.nonfallback_count == 0


def test_support_robust_wrapper_executes_only_inside_inflated_bound() -> None:
    accepted = support_robust_action_decision(
        base_selected_action_index=2,
        fallback_action_index=0,
        action_count=3,
        finite_support_regret_bound=0.04,
        conformal_radius=0.10,
        operational_regret_tolerance=0.15,
    )
    rejected = support_robust_action_decision(
        base_selected_action_index=2,
        fallback_action_index=0,
        action_count=3,
        finite_support_regret_bound=0.04,
        conformal_radius=0.12,
        operational_regret_tolerance=0.15,
    )

    assert accepted.execute_base_nonfallback
    assert accepted.returned_action_index == 2
    assert accepted.support_robust_regret_bound == pytest.approx(0.14)
    assert not rejected.execute_base_nonfallback
    assert rejected.returned_action_index == 0
    assert rejected.reason == "support_robust_regret_exceeds_tolerance"


def test_infinite_radius_and_base_fallback_are_fail_closed() -> None:
    unavailable = support_robust_action_decision(
        base_selected_action_index=1,
        fallback_action_index=0,
        action_count=2,
        finite_support_regret_bound=0.0,
        conformal_radius=float("inf"),
        operational_regret_tolerance=1.0,
    )
    already_fallback = support_robust_action_decision(
        base_selected_action_index=0,
        fallback_action_index=0,
        action_count=2,
        finite_support_regret_bound=0.0,
        conformal_radius=0.0,
        operational_regret_tolerance=1.0,
    )

    assert unavailable.returned_action_index == 0
    assert unavailable.reason == "conformal_radius_unavailable"
    assert already_fallback.returned_action_index == 0
    assert already_fallback.reason == "base_policy_fallback"


@pytest.mark.parametrize(
    ("scores", "alpha"),
    [
        ([], 0.2),
        ([0.1, -0.1], 0.2),
        ([0.1, np.nan], 0.2),
        ([0.1], 0.0),
        ([0.1], 1.0),
    ],
)
def test_bad_conformal_inputs_are_rejected(scores: list[float], alpha: float) -> None:
    with pytest.raises(ValueError):
        split_conformal_trajectory_envelope(scores, miscoverage=alpha)


def test_bad_trajectory_shapes_and_action_indices_are_rejected() -> None:
    with pytest.raises(ValueError):
        trajectory_policy_regret_excess(
            [[0.0, 1.0]],
            [[0.0]],
            [1],
            fallback_action_index=0,
        )
    with pytest.raises(ValueError):
        trajectory_policy_regret_excess(
            [[0.0, 1.0]],
            [[0.0, 0.1]],
            [2],
            fallback_action_index=0,
        )
    with pytest.raises(ValueError):
        trajectory_policy_regret_excess(
            [[0.0, 1.0]],
            [[0.0, -0.1]],
            [1],
            fallback_action_index=0,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"action_count": 1},
        {"base_selected_action_index": 3},
        {"fallback_action_index": -1},
        {"finite_support_regret_bound": -0.1},
        {"conformal_radius": float("nan")},
        {"operational_regret_tolerance": -0.1},
    ],
)
def test_bad_operational_inputs_are_rejected(kwargs: dict[str, object]) -> None:
    valid: dict[str, object] = {
        "base_selected_action_index": 1,
        "fallback_action_index": 0,
        "action_count": 3,
        "finite_support_regret_bound": 0.1,
        "conformal_radius": 0.1,
        "operational_regret_tolerance": 0.3,
    }
    valid.update(kwargs)
    with pytest.raises(ValueError):
        support_robust_action_decision(**valid)  # type: ignore[arg-type]
