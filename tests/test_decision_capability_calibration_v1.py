from __future__ import annotations

import itertools

import numpy as np
import pytest

from bayesian_phystwin.decision_capability_atlas_v1 import (
    AffineCapabilityHalfspacesV1,
)
from bayesian_phystwin.decision_capability_calibration_v1 import (
    DECISION_CAPABILITY_CALIBRATION_CLAIM_BOUNDARY,
    affine_box_pairwise_undercoverage_score,
    finite_group_atlas_calibration,
    maximize_affine_lower_envelope_on_box,
    statistically_corrected_halfspaces,
)
from bayesian_phystwin.decision_capability_task_uncertainty_v1 import (
    box_robust_center_halfspaces,
)


def _region() -> AffineCapabilityHalfspacesV1:
    # Action zero versus action one. The model pairwise envelope is
    # max(x + y - 1, -x + y - 1), so the zero-regret region is |x| + y <= 1.
    return AffineCapabilityHalfspacesV1(
        action_index=0,
        regret_tolerance=0.0,
        active_class_index=np.array([0]),
        normal=np.array([[1.0, 1.0], [-1.0, 1.0]]),
        offset=np.array([1.0, 1.0]),
        benchmark_action_index=np.array([1, 1]),
        witness_hypothesis_index=np.array([[0], [1]]),
    )


def test_calibration_quantile_and_coverage() -> None:
    result = finite_group_atlas_calibration(
        [-0.2, 0.0, 0.1, 0.4, 0.2, 0.3, 0.05, 0.15, 0.25],
        alpha=0.1,
    )
    assert result.quantile_rank == 9
    assert result.quantile_value == pytest.approx(0.4)
    assert result.correction == pytest.approx(0.4)
    assert result.guaranteed_marginal_coverage == pytest.approx(0.9)
    assert result.summary()["calibration_group_count"] == 9
    assert (
        result.summary()["claim_boundary"]
        == DECISION_CAPABILITY_CALIBRATION_CLAIM_BOUNDARY
    )


def test_calibration_refuses_unattainable_alpha() -> None:
    with pytest.raises(ValueError, match="too small"):
        finite_group_atlas_calibration([0.0] * 8, alpha=0.1)


def test_correction_shifts_region_and_is_immutable() -> None:
    corrected = statistically_corrected_halfspaces(_region(), 0.2)
    np.testing.assert_allclose(corrected.offset, [0.8, 0.8])
    assert corrected.benchmark_action_index.tolist() == [1, 1]
    with pytest.raises(ValueError, match="read-only"):
        corrected.offset[0] = 0.0


def test_lower_envelope_1d_exact() -> None:
    # max over [-2, 2] of min(x, -x) is zero at x = 0.
    result = maximize_affine_lower_envelope_on_box(
        [0.0, 0.0],
        [[1.0], [-1.0]],
        [[-2.0, 2.0]],
    )
    assert result.maximum_value == pytest.approx(0.0, abs=1e-10)
    np.testing.assert_allclose(result.task_parameter, [0.0], atol=1e-10)
    assert set(result.active_affine_index.tolist()) == {0, 1}


def test_lower_envelope_2d_matches_known_optimum() -> None:
    # min(1-x-y, 1+x-y, y) peaks at (0, 0.5) with value 0.5.
    result = maximize_affine_lower_envelope_on_box(
        [1.0, 1.0, 0.0],
        [[-1.0, -1.0], [1.0, -1.0], [0.0, 1.0]],
        [[-1.0, 1.0], [0.0, 1.0]],
    )
    assert result.maximum_value == pytest.approx(0.5, abs=1e-10)
    np.testing.assert_allclose(result.task_parameter, [0.0, 0.5], atol=1e-10)


def test_pairwise_undercoverage_score_known() -> None:
    # The realized gap y - 0.7 exceeds |x| + y - 1 by at most 0.3.
    result = affine_box_pairwise_undercoverage_score(
        _region(),
        [0.0, -0.7],
        [[0.0, 0.0], [0.0, 1.0]],
        [[-1.0, 1.0], [0.0, 1.0]],
    )
    assert result.nonnegative_score == pytest.approx(0.3, abs=1e-10)
    assert result.critical_benchmark_action_index == 1
    assert result.critical_task_parameter[0] == pytest.approx(0.0, abs=1e-10)


def test_pairwise_overcoverage_clips_to_zero() -> None:
    result = affine_box_pairwise_undercoverage_score(
        _region(),
        [0.0, -2.0],
        [[0.0, 0.0], [0.0, 0.0]],
        [[-1.0, 1.0], [0.0, 1.0]],
    )
    assert result.nonnegative_score == 0.0
    assert result.raw_score_by_benchmark[0] < 0.0


def test_statistical_correction_covers_realized_gap_on_box() -> None:
    correction = affine_box_pairwise_undercoverage_score(
        _region(),
        [0.0, -0.7],
        [[0.0, 0.0], [0.0, 1.0]],
        [[-1.0, 1.0], [0.0, 1.0]],
    ).nonnegative_score
    corrected = statistically_corrected_halfspaces(_region(), correction)
    for x, y in itertools.product(
        np.linspace(-1.0, 1.0, 41),
        np.linspace(0.0, 1.0, 41),
    ):
        model = np.max(-corrected.offset + corrected.normal @ np.array([x, y]))
        realized = -0.7 + y
        assert realized <= model + 1e-9


def test_statistical_and_objective_corrections_commute() -> None:
    base = _region()
    width = np.array([0.05, 0.1])
    first = statistically_corrected_halfspaces(
        box_robust_center_halfspaces(base, width),
        0.2,
    )
    second = box_robust_center_halfspaces(
        statistically_corrected_halfspaces(base, 0.2),
        width,
    )
    np.testing.assert_allclose(first.offset, second.offset)


def test_active_set_cap_fails_closed() -> None:
    with pytest.raises(ValueError, match="exceeding"):
        maximize_affine_lower_envelope_on_box(
            [0.0, 0.0, 0.0],
            [[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]],
            [[-1.0, 1.0], [-1.0, 1.0]],
            maximum_active_sets=1,
        )


def test_invalid_contracts_fail_closed() -> None:
    with pytest.raises(ValueError, match="strictly"):
        finite_group_atlas_calibration([0.0], alpha=0.0)
    with pytest.raises(ValueError, match="nonnegative"):
        statistically_corrected_halfspaces(_region(), -0.1)
    with pytest.raises(ValueError, match="inconsistent rows"):
        maximize_affine_lower_envelope_on_box(
            [0.0],
            [[1.0], [2.0]],
            [[0.0, 1.0]],
        )
    with pytest.raises(ValueError, match="lower < upper"):
        maximize_affine_lower_envelope_on_box(
            [0.0],
            [[1.0]],
            [[1.0, 1.0]],
        )
    with pytest.raises(ValueError, match="shape"):
        affine_box_pairwise_undercoverage_score(
            _region(),
            [0.0, 0.0],
            [[0.0], [0.0]],
            [[-1.0, 1.0], [0.0, 1.0]],
        )
