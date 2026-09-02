from __future__ import annotations

import math

import numpy as np
import pytest

from bayesian_phystwin.conformal_regret_envelope_v1 import (
    CONFORMAL_REGRET_ENVELOPE_CLAIM_BOUNDARY,
    CONFORMAL_REGRET_ENVELOPE_SEMANTICS,
    CONFORMAL_REGRET_ENVELOPE_VERSION,
    support_robust_decision,
    trajectory_conformal_regret_envelope,
)


def _calibration() -> tuple[np.ndarray, np.ndarray]:
    realized = np.asarray(
        [
            [[0.1, 0.2, 0.3], [0.4, 0.1, 0.2]],
            [[0.2, 0.3, 0.4], [0.1, 0.9, 0.3]],
            [[0.4, 0.2, 0.1], [0.2, 0.3, 0.5]],
            [[0.3, 0.2, 0.2], [0.1, 0.4, 0.2]],
        ],
        dtype=np.float64,
    )
    return realized, np.full_like(realized, 0.1)


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
    realized, registered = _calibration()
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
    np.testing.assert_array_equal(envelope.candidate_action_mask, [False, True, True])
    assert envelope.finite_sample_rank == 4
    assert envelope.radius == pytest.approx(0.8)
    assert envelope.nominal_coverage == pytest.approx(0.8)
    assert envelope.has_finite_radius
    assert envelope.calibration_trajectory_count == 4
    assert envelope.decision_count_per_trajectory == 2
    assert envelope.action_count == 3
    summary = envelope.summary()
    assert summary["version"] == CONFORMAL_REGRET_ENVELOPE_VERSION
    assert summary["semantics"] == CONFORMAL_REGRET_ENVELOPE_SEMANTICS
    assert summary["claim_boundary"] == CONFORMAL_REGRET_ENVELOPE_CLAIM_BOUNDARY
    assert summary["has_finite_radius"] is True
    with pytest.raises(ValueError):
        envelope.trajectory_nonconformity_scores[0] = 0.0


def test_default_candidate_mask_selects_every_action() -> None:
    realized, registered = _calibration()
    envelope = trajectory_conformal_regret_envelope(
        realized,
        registered,
        miscoverage=0.25,
    )
    np.testing.assert_array_equal(envelope.candidate_action_mask, [True, True, True])


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
    assert envelope.summary()["has_finite_radius"] is False


def test_envelope_never_tightens_registered_bound() -> None:
    realized = np.zeros((4, 1, 2))
    registered = np.ones_like(realized)
    envelope = trajectory_conformal_regret_envelope(
        realized,
        registered,
        miscoverage=0.2,
    )
    assert envelope.radius == 0.0


@pytest.mark.parametrize(
    ("realized", "registered", "message"),
    [
        (
            np.asarray([[['bad']]], dtype=object),
            np.zeros((1, 1, 1)),
            "real numeric",
        ),
        (np.zeros((2, 3)), np.zeros((2, 3)), "shape"),
        (np.zeros((0, 3, 2)), np.zeros((0, 3, 2)), "shape"),
        (
            np.full((2, 3, 2), np.nan),
            np.zeros((2, 3, 2)),
            "finite",
        ),
        (-np.ones((2, 3, 2)), np.zeros((2, 3, 2)), "nonnegative"),
        (
            np.zeros((2, 3, 2)),
            np.full((2, 3, 2), np.inf),
            "finite",
        ),
        (
            np.zeros((2, 3, 2)),
            -np.ones((2, 3, 2)),
            "nonnegative",
        ),
    ],
)
def test_invalid_trajectory_tensors_fail_closed(
    realized: np.ndarray, registered: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        trajectory_conformal_regret_envelope(
            realized,
            registered,
            miscoverage=0.2,
        )


def test_mismatched_trajectory_tensor_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="equal shape"):
        trajectory_conformal_regret_envelope(
            np.zeros((2, 3, 2)),
            np.zeros((2, 3, 3)),
            miscoverage=0.2,
        )


@pytest.mark.parametrize("miscoverage", [0.0, 1.0, -0.1, float("nan"), True, "0.2"])
def test_invalid_miscoverage_is_rejected(miscoverage: object) -> None:
    with pytest.raises(ValueError, match="strictly between"):
        trajectory_conformal_regret_envelope(
            np.zeros((4, 1, 2)),
            np.zeros((4, 1, 2)),
            miscoverage=miscoverage,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "mask",
    [
        [1, 0, 1],
        [True, False],
        [False, False, False],
        np.asarray([[True, False, True]]),
    ],
)
def test_invalid_calibration_action_masks_are_rejected(mask: object) -> None:
    realized, registered = _calibration()
    with pytest.raises(ValueError):
        trajectory_conformal_regret_envelope(
            realized,
            registered,
            miscoverage=0.2,
            candidate_action_mask=mask,
        )


def test_support_robust_decision_selects_unique_admissible_nonfallback() -> None:
    result = support_robust_decision(
        [0.4, 0.02, 0.2],
        conformal_radius=0.03,
        regret_tolerance=0.06,
        fallback_action_index=0,
    )
    assert result.selected_action_index == 1
    assert not result.used_fallback
    np.testing.assert_allclose(result.registered_worst_case_regret, [0.4, 0.02, 0.2])
    np.testing.assert_allclose(result.inflated_regret_upper_bound, [0.43, 0.05, 0.23])
    np.testing.assert_array_equal(result.candidate_action_mask, [True, True, True])
    np.testing.assert_array_equal(
        result.tolerance_admissible_action_mask, [False, True, False]
    )
    summary = result.summary()
    assert summary["selected_action_index"] == 1
    assert summary["fallback_action_index"] == 0
    assert summary["used_fallback"] is False
    assert summary["admissible_action_count"] == 1
    assert summary["version"] == CONFORMAL_REGRET_ENVELOPE_VERSION
    assert summary["semantics"] == CONFORMAL_REGRET_ENVELOPE_SEMANTICS
    assert summary["claim_boundary"] == CONFORMAL_REGRET_ENVELOPE_CLAIM_BOUNDARY
    with pytest.raises(ValueError):
        result.inflated_regret_upper_bound[0] = 0.0


def test_support_robust_decision_honors_candidate_mask() -> None:
    result = support_robust_decision(
        [0.4, 0.01, 0.02],
        conformal_radius=0.01,
        regret_tolerance=0.04,
        candidate_action_mask=[True, False, True],
    )
    assert result.selected_action_index == 2
    assert not result.used_fallback
    np.testing.assert_array_equal(result.candidate_action_mask, [True, False, True])


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


def test_support_robust_decision_falls_back_when_best_nonfallback_exceeds_budget() -> None:
    result = support_robust_decision(
        [0.5, 0.02, 0.2],
        conformal_radius=0.04,
        regret_tolerance=0.05,
    )
    assert result.selected_action_index == 0
    assert result.used_fallback
    assert not np.any(result.tolerance_admissible_action_mask)


def test_support_robust_decision_falls_back_for_infinite_radius() -> None:
    result = support_robust_decision(
        [0.1, 0.01, 0.02],
        conformal_radius=float("inf"),
        regret_tolerance=1.0,
    )
    assert result.selected_action_index == 0
    assert result.used_fallback
    assert not np.any(result.tolerance_admissible_action_mask)


def test_support_robust_decision_keeps_fallback_when_it_minimizes() -> None:
    result = support_robust_decision(
        [0.01, 0.02, 0.03],
        conformal_radius=0.01,
        regret_tolerance=0.10,
    )
    assert result.selected_action_index == 0
    assert result.used_fallback


@pytest.mark.parametrize(
    ("regrets", "message"),
    [
        (["bad", "values"], "real numeric"),
        ([[0.1, 0.2]], "at least two actions"),
        ([0.1], "at least two actions"),
        ([0.1, float("nan")], "finite and nonnegative"),
        ([0.1, float("inf")], "finite and nonnegative"),
        ([0.1, -0.2], "finite and nonnegative"),
    ],
)
def test_invalid_registered_regret_vectors_are_rejected(
    regrets: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        support_robust_decision(
            regrets,
            conformal_radius=0.0,
            regret_tolerance=0.1,
        )


@pytest.mark.parametrize("fallback", [True, 1.5, "0", -1, 3])
def test_invalid_fallback_indices_are_rejected(fallback: object) -> None:
    with pytest.raises(ValueError, match="fallback_action_index"):
        support_robust_decision(
            [0.2, 0.01, 0.1],
            conformal_radius=0.0,
            regret_tolerance=0.1,
            fallback_action_index=fallback,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("radius", [True, "0.1", float("nan"), -0.1, float("-inf")])
def test_invalid_conformal_radii_are_rejected(radius: object) -> None:
    with pytest.raises(ValueError, match="conformal_radius"):
        support_robust_decision(
            [0.2, 0.01, 0.1],
            conformal_radius=radius,  # type: ignore[arg-type]
            regret_tolerance=0.1,
        )


@pytest.mark.parametrize("tolerance", [True, "0.1", float("nan"), float("inf"), -0.1])
def test_invalid_regret_tolerances_are_rejected(tolerance: object) -> None:
    with pytest.raises(ValueError, match="regret_tolerance"):
        support_robust_decision(
            [0.2, 0.01, 0.1],
            conformal_radius=0.0,
            regret_tolerance=tolerance,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "mask",
    [
        [1, 0, 1],
        [True, False],
        [False, False, False],
        np.asarray([[True, False, True]]),
    ],
)
def test_invalid_decision_action_masks_are_rejected(mask: object) -> None:
    with pytest.raises(ValueError):
        support_robust_decision(
            [0.2, 0.01, 0.1],
            conformal_radius=0.0,
            regret_tolerance=0.1,
            candidate_action_mask=mask,
        )


def test_support_robust_decision_requires_fallback_in_action_mask() -> None:
    with pytest.raises(ValueError, match="include fallback_action_index"):
        support_robust_decision(
            [0.2, 0.01, 0.1],
            conformal_radius=0.0,
            regret_tolerance=0.05,
            candidate_action_mask=[False, True, True],
        )
