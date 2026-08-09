import math

import numpy as np
import pytest

from bayesian_phystwin.calibration import (
    finite_group_conformal_rank,
    maximum_finite_group_coverage,
    minimum_groups_for_finite_conformal,
    plan_finite_group_calibration,
)
from bayesian_phystwin.grouped_conformal import (
    GroupedConformalResult,
    finite_group_conformal_quantile,
    group_max_nonconformity_scores,
    grouped_conformal_upper_bounds,
)


def test_group_scores_weight_each_independent_unit_once() -> None:
    scores = group_max_nonconformity_scores(
        (np.asarray([1.0]), np.full(100, 2.0)),
        (np.asarray([1.0]), np.ones(100)),
        score="scaled",
    )

    np.testing.assert_allclose(scores, [1.0, 2.0])
    assert not scores.flags.writeable


def test_nine_groups_are_required_for_a_finite_ninety_percent_bound() -> None:
    impossible, impossible_rank = finite_group_conformal_quantile(
        np.arange(8.0),
        0.9,
    )
    finite, finite_rank = finite_group_conformal_quantile(np.arange(9.0), 0.9)

    assert math.isinf(impossible)
    assert impossible_rank == 9
    assert finite == 8.0
    assert finite_rank == 9


def test_group_quantile_does_not_promote_exact_float_rank_boundary() -> None:
    quantile, rank = finite_group_conformal_quantile(np.arange(24.0), 0.28)

    assert rank == 7
    assert quantile == 6.0


@pytest.mark.parametrize("count", [5, 10, 12])
def test_advertised_maximum_finite_coverage_round_trips(count: int) -> None:
    coverage = maximum_finite_group_coverage(count)

    assert finite_group_conformal_rank(count, coverage) == count
    assert minimum_groups_for_finite_conformal(coverage) == count
    design = plan_finite_group_calibration(
        count,
        coverage,
        predictor_frozen_before_scores=True,
        calibration_outcomes_used_for_selection=False,
    )
    assert design.finite_sample_rank == count

    coverage_above = float(np.nextafter(coverage, 1.0))
    assert finite_group_conformal_rank(count, coverage_above) == count + 1


def test_scaled_bounds_cover_all_registered_future_endpoints_together() -> None:
    result = grouped_conformal_upper_bounds(
        (
            np.asarray([1.0, 2.0]),
            np.asarray([1.5, 3.0]),
            np.asarray([2.0, 4.0]),
        ),
        (np.ones(2), np.ones(2), np.ones(2)),
        np.asarray([2.0, 4.0]),
        coverage=0.5,
        score="scaled",
    )

    np.testing.assert_allclose(result.calibration_group_scores, [2.0, 3.0, 4.0])
    np.testing.assert_allclose(result.upper_bound, [6.0, 12.0])
    assert result.quantile == 3.0
    assert result.finite_sample_rank == 2
    assert result.calibration_group_count == 3
    assert result.nominal_coverage == 0.5
    assert result.score == "scaled"
    assert not result.upper_bound.flags.writeable
    assert not result.calibration_group_scores.flags.writeable


def test_additive_bounds_are_clipped_to_nonnegative_losses() -> None:
    result = grouped_conformal_upper_bounds(
        (np.asarray([0.0]), np.asarray([0.0])),
        (np.asarray([1.0]), np.asarray([2.0])),
        np.asarray([0.25, 0.5]),
        coverage=0.5,
        score="additive",
    )

    assert result.quantile == -1.0
    np.testing.assert_array_equal(result.upper_bound, [0.0, 0.0])


def test_impossible_group_rank_returns_infinite_bounds() -> None:
    result = grouped_conformal_upper_bounds(
        tuple(np.asarray([float(index)]) for index in range(8)),
        tuple(np.ones(1) for _ in range(8)),
        np.asarray([1.0, 2.0]),
        coverage=0.9,
        score="scaled",
    )

    assert math.isinf(result.quantile)
    assert np.all(np.isinf(result.upper_bound))


@pytest.mark.parametrize(
    ("targets", "predictions", "score", "message"),
    [
        ((), (), "scaled", "at least one calibration group"),
        ((np.ones(1),), (), "scaled", "same number of groups"),
        ((np.ones(2),), (np.ones(1),), "scaled", "equal shape"),
        ((np.empty(0),), (np.empty(0),), "scaled", "cannot be empty"),
        ((np.asarray([-1.0]),), (np.ones(1),), "scaled", "nonnegative"),
        ((np.asarray([np.inf]),), (np.ones(1),), "scaled", "finite"),
        (
            (np.ones(1),),
            (np.asarray([np.inf]),),
            "scaled",
            "prediction must be finite",
        ),
        ((np.ones(1),), (np.zeros(1),), "scaled", "must be positive"),
    ],
)
def test_group_score_validation_fails_closed(
    targets: tuple[np.ndarray, ...],
    predictions: tuple[np.ndarray, ...],
    score: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        group_max_nonconformity_scores(
            targets,
            predictions,
            score=score,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("coverage", [0.0, 1.0, float("nan")])
def test_group_quantile_rejects_invalid_coverage(coverage: float) -> None:
    with pytest.raises(ValueError, match="coverage"):
        finite_group_conformal_quantile(np.ones(9), coverage)


def test_group_quantile_rejects_empty_or_nonfinite_scores() -> None:
    with pytest.raises(ValueError, match="at least one"):
        finite_group_conformal_quantile(np.empty(0), 0.5)
    with pytest.raises(ValueError, match="finite"):
        finite_group_conformal_quantile(np.asarray([np.nan]), 0.5)


@pytest.mark.parametrize(
    ("future", "score", "message"),
    [
        (np.empty(0), "additive", "cannot be empty"),
        (np.asarray([np.inf]), "additive", "must be finite"),
        (np.asarray([0.0]), "scaled", "must be positive"),
    ],
)
def test_future_prediction_validation_fails_closed(
    future: np.ndarray,
    score: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        grouped_conformal_upper_bounds(
            (np.ones(1),),
            (np.ones(1),),
            future,
            coverage=0.5,
            score=score,  # type: ignore[arg-type]
        )


def test_invalid_score_is_rejected() -> None:
    with pytest.raises(ValueError, match="score"):
        group_max_nonconformity_scores(
            (np.ones(1),),
            (np.ones(1),),
            score="unknown",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"upper_bound": np.empty(0)},
        {"upper_bound": np.asarray([np.nan])},
        {"upper_bound": np.asarray([-1.0])},
        {"calibration_group_count": 0},
        {"calibration_group_count": 2},
        {"calibration_group_scores": np.asarray([np.inf])},
        {"finite_sample_rank": 0},
        {"nominal_coverage": 1.0},
        {"score": "unknown"},
        {"quantile": float("nan")},
        {"quantile": -math.inf},
        {"finite_sample_rank": 2, "quantile": 1.0},
        {"quantile": math.inf},
    ],
)
def test_result_contract_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {
        "upper_bound": np.ones(1),
        "calibration_group_scores": np.ones(1),
        "quantile": 1.0,
        "finite_sample_rank": 1,
        "calibration_group_count": 1,
        "nominal_coverage": 0.5,
        "score": "scaled",
    }
    values.update(kwargs)
    with pytest.raises(ValueError):
        GroupedConformalResult(**values)  # type: ignore[arg-type]


def test_result_contract_rejects_forged_feasible_rank() -> None:
    with pytest.raises(ValueError, match="finite_sample_rank"):
        GroupedConformalResult(
            upper_bound=np.ones(1),
            calibration_group_scores=np.asarray([1.0, 2.0, 3.0]),
            quantile=1.0,
            finite_sample_rank=1,
            calibration_group_count=3,
            nominal_coverage=0.5,
            score="scaled",
        )


def test_result_contract_rejects_forged_finite_quantile() -> None:
    with pytest.raises(ValueError, match="quantile"):
        GroupedConformalResult(
            upper_bound=np.ones(1),
            calibration_group_scores=np.asarray([1.0, 2.0, 3.0]),
            quantile=1.0,
            finite_sample_rank=2,
            calibration_group_count=3,
            nominal_coverage=0.5,
            score="scaled",
        )
