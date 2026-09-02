from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from bayesian_phystwin.decision_capability_atlas_v1 import (
    AffineCapabilityHalfspacesV1,
    affine_capability_halfspaces,
    affine_decision_capability_atlas,
)
from bayesian_phystwin.decision_capability_task_uncertainty_v1 import (
    box_robust_center_halfspaces,
)
from bayesian_phystwin.statistically_valid_capability_atlas_v1 import (
    STATISTICALLY_VALID_CAPABILITY_ATLAS_CLAIM_BOUNDARY,
    calibrated_box_robust_center_halfspaces,
    calibrated_capability_halfspaces,
    calibrated_task_atlas,
    calibration_unit_maximum,
    continuous_pairwise_undercoverage_score,
    split_conformal_capability_correction,
)


def _controlled_family() -> tuple[np.ndarray, ...]:
    displacement = np.array(
        [
            [-1.1, -0.1, 0.7],
            [-0.7, 0.1, 1.1],
            [-1.0, 0.0, 0.6],
            [-0.6, 0.0, 1.0],
        ]
    )
    risk = np.array(
        [
            [0.4, 0.05, 0.8],
            [0.8, 0.05, 0.4],
            [0.5, 0.02, 0.9],
            [0.9, 0.02, 0.5],
        ]
    )
    return (
        np.full(4, 0.25),
        np.array([0.5, 0.5]),
        np.array([0, 0, 1, 1]),
        np.square(displacement),
        np.stack((-2.0 * displacement, risk), axis=2),
    )


def _regions() -> list[AffineCapabilityHalfspacesV1]:
    prior, quotient, classes, intercept, coefficient = _controlled_family()
    return [
        affine_capability_halfspaces(
            prior,
            quotient,
            classes,
            intercept,
            coefficient,
            action_index=action,
        )
        for action in range(3)
    ]


def _absolute_value_region() -> AffineCapabilityHalfspacesV1:
    return AffineCapabilityHalfspacesV1(
        action_index=0,
        regret_tolerance=0.0,
        active_class_index=np.array([0], dtype=np.int64),
        normal=np.array([[1.0], [-1.0]]),
        offset=np.array([0.0, 0.0]),
        benchmark_action_index=np.array([1, 1], dtype=np.int64),
        witness_hypothesis_index=np.array([[0], [1]], dtype=np.int64),
    )


def test_split_conformal_uses_finite_sample_order_statistic() -> None:
    result = split_conformal_capability_correction(
        [0.7, -0.2, 0.1, 0.5, 0.4, 0.9, 0.0, 0.3, 0.2],
        miscoverage=0.2,
    )
    assert result.order_statistic_rank == 8
    assert result.raw_quantile == pytest.approx(0.7)
    assert result.nonnegative_correction == pytest.approx(0.7)
    assert result.summary()["claim_boundary"] == (
        STATISTICALLY_VALID_CAPABILITY_ATLAS_CLAIM_BOUNDARY
    )


def test_negative_quantile_cannot_enlarge_model_atlas() -> None:
    result = split_conformal_capability_correction(
        [-0.8, -0.6, -0.4, -0.2],
        miscoverage=0.4,
    )
    assert result.raw_quantile < 0.0
    assert result.nonnegative_correction == 0.0


def test_too_small_calibration_sample_fails_closed() -> None:
    with pytest.raises(ValueError, match="too small"):
        split_conformal_capability_correction([0.1, 0.2, 0.3], miscoverage=0.05)


def test_calibration_shifts_every_halfspace_and_contracts_atlas() -> None:
    regions = _regions()
    tasks = np.asarray(
        [(x, r) for x in np.linspace(-1.5, 1.5, 61) for r in np.linspace(0, 4, 51)]
    )
    nominal = calibrated_task_atlas(regions, tasks, nonnegative_correction=0.0)
    calibrated = calibrated_task_atlas(regions, tasks, nonnegative_correction=0.15)
    assert np.all(~calibrated.action_capability_mask | nominal.action_capability_mask)
    assert np.count_nonzero(calibrated.capability_mask) < np.count_nonzero(
        nominal.capability_mask
    )
    shifted = calibrated_capability_halfspaces(regions[0], 0.15)
    np.testing.assert_allclose(shifted.normal, regions[0].normal)
    np.testing.assert_allclose(shifted.offset, regions[0].offset - 0.15)
    np.testing.assert_array_equal(
        shifted.witness_hypothesis_index,
        regions[0].witness_hypothesis_index,
    )


def test_data_and_box_objective_corrections_commute() -> None:
    region = _regions()[1]
    width = np.array([0.1, 0.2])
    first = calibrated_box_robust_center_halfspaces(region, 0.12, width)
    second = calibrated_capability_halfspaces(
        box_robust_center_halfspaces(region, width),
        0.12,
    )
    np.testing.assert_allclose(first.normal, second.normal)
    np.testing.assert_allclose(first.offset, second.offset)


def test_continuous_score_finds_interior_kink_not_box_vertex() -> None:
    result = continuous_pairwise_undercoverage_score(
        _absolute_value_region(),
        benchmark_action_index=1,
        realized_gap_intercept=0.5,
        realized_gap_coefficient=[0.0],
        task_bounds=[[-1.0, 1.0]],
    )
    assert result.score == pytest.approx(0.5)
    np.testing.assert_allclose(result.task_parameter, [0.0], atol=1e-12)
    assert result.realized_pairwise_gap == pytest.approx(0.5)
    assert result.model_pairwise_gap == pytest.approx(0.0)


def test_continuous_score_matches_dense_independent_grid() -> None:
    result = continuous_pairwise_undercoverage_score(
        _absolute_value_region(),
        benchmark_action_index=1,
        realized_gap_intercept=0.1,
        realized_gap_coefficient=[0.35],
        task_bounds=[[-1.0, 1.0]],
    )
    grid = np.linspace(-1.0, 1.0, 100_001)
    expected = float(np.max(0.1 + 0.35 * grid - np.maximum(grid, -grid)))
    assert result.score == pytest.approx(expected, abs=2e-5)


def test_unit_score_aggregates_all_routed_cases_before_calibration() -> None:
    witnesses = [
        continuous_pairwise_undercoverage_score(
            _absolute_value_region(),
            benchmark_action_index=1,
            realized_gap_intercept=value,
            realized_gap_coefficient=[0.0],
            task_bounds=[[-1.0, 1.0]],
        )
        for value in (0.2, 0.7)
    ]
    unit = calibration_unit_maximum(witnesses)
    assert unit.score == pytest.approx(0.7)
    assert unit.selected_case_index == 1
    assert unit.case_count == 2


def test_continuous_score_complexity_cap_fails_before_enumeration() -> None:
    with pytest.raises(ValueError, match="exceeding"):
        continuous_pairwise_undercoverage_score(
            _absolute_value_region(),
            benchmark_action_index=1,
            realized_gap_intercept=0.0,
            realized_gap_coefficient=[0.0],
            task_bounds=[[-1.0, 1.0]],
            maximum_candidate_vertices=1,
        )


def test_calibrated_task_mask_matches_direct_shifted_membership() -> None:
    regions = _regions()
    tasks = np.array([[-1.2, 0.2], [0.0, 2.0], [1.2, 0.2], [-0.45, 0.0]])
    result = calibrated_task_atlas(regions, tasks, nonnegative_correction=0.05)
    expected = np.column_stack(
        [
            calibrated_capability_halfspaces(region, 0.05).contains(tasks)
            for region in regions
        ]
    )
    np.testing.assert_array_equal(result.action_capability_mask, expected)
    assert result.summary()["task_count"] == 4


def test_zero_correction_matches_original_pointwise_atlas() -> None:
    prior, quotient, classes, intercept, coefficient = _controlled_family()
    tasks = np.asarray(list(product([-1.0, 0.0, 1.0], [0.0, 1.0, 3.0])))
    pointwise = affine_decision_capability_atlas(
        prior,
        quotient,
        classes,
        intercept,
        coefficient,
        tasks,
    )
    calibrated = calibrated_task_atlas(
        _regions(),
        tasks,
        nonnegative_correction=0.0,
    )
    np.testing.assert_array_equal(
        calibrated.action_capability_mask,
        pointwise.robustly_optimal_action_mask,
    )


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (
            lambda: split_conformal_capability_correction([0.1], miscoverage=1.0),
            "strictly",
        ),
        (
            lambda: calibrated_capability_halfspaces(_absolute_value_region(), -0.1),
            "nonnegative",
        ),
        (
            lambda: continuous_pairwise_undercoverage_score(
                _absolute_value_region(),
                benchmark_action_index=2,
                realized_gap_intercept=0.0,
                realized_gap_coefficient=[0.0],
                task_bounds=[[-1.0, 1.0]],
            ),
            "absent",
        ),
        (
            lambda: continuous_pairwise_undercoverage_score(
                _absolute_value_region(),
                benchmark_action_index=1,
                realized_gap_intercept=0.0,
                realized_gap_coefficient=[0.0, 1.0],
                task_bounds=[[-1.0, 1.0]],
            ),
            "wrong dimension",
        ),
    ],
)
def test_invalid_contracts_fail_closed(call: object, match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        call()  # type: ignore[operator]


def test_outputs_are_immutable() -> None:
    correction = split_conformal_capability_correction(
        [0.1, 0.2, 0.3, 0.4],
        miscoverage=0.4,
    )
    with pytest.raises(ValueError, match="read-only"):
        correction.sorted_scores[0] = 0.0
    witness = continuous_pairwise_undercoverage_score(
        _absolute_value_region(),
        benchmark_action_index=1,
        realized_gap_intercept=0.5,
        realized_gap_coefficient=[0.0],
        task_bounds=[[-1.0, 1.0]],
    )
    with pytest.raises(ValueError, match="read-only"):
        witness.task_parameter[0] = 1.0
