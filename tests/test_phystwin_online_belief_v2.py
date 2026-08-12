from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.phystwin_online_belief import (
    finite_sample_absolute_residual_quantile_m as finite_sample_v1,
    update_recursive_rbf_belief as update_v1,
)
from bayesian_phystwin.phystwin_online_belief_v2 import (
    RecursiveRbfBeliefConfig,
    RecursiveRbfBeliefSnapshot,
    finite_sample_absolute_residual_quantile_m,
    initialize_recursive_rbf_belief,
    update_recursive_rbf_belief,
)


def _problem() -> tuple[
    np.ndarray,
    np.ndarray,
    RecursiveRbfBeliefConfig,
]:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    center_ids = np.arange(3, dtype=np.int64)
    return points, center_ids, RecursiveRbfBeliefConfig()


def _prior() -> tuple[
    np.ndarray,
    RecursiveRbfBeliefConfig,
    RecursiveRbfBeliefSnapshot,
]:
    points, center_ids, config = _problem()
    prior = initialize_recursive_rbf_belief(
        center_ids,
        points,
        points,
        config=config,
    )
    return points, config, prior


@pytest.mark.parametrize(
    "available",
    [
        np.asarray([1, 0, 1], dtype=np.int64),
        np.asarray([1.0, 0.0, 1.0], dtype=float),
        np.asarray([1.0, np.nan, 0.0], dtype=float),
        np.asarray(["yes", "no", "yes"], dtype=object),
    ],
)
def test_quantile_rejects_nonboolean_availability(available: np.ndarray) -> None:
    residual = np.asarray(
        [[0.001, 0.0, 0.0], [0.002, 0.0, 0.0], [0.003, 0.0, 0.0]]
    )

    with pytest.raises(ValueError, match="available must contain only booleans"):
        finite_sample_absolute_residual_quantile_m(residual, available, 0.9)


@pytest.mark.parametrize(
    "available",
    [
        np.asarray([1, 0, 1], dtype=np.int64),
        np.asarray([1.0, 2.0, 0.0], dtype=float),
        np.asarray([True, False, None], dtype=object),
    ],
)
def test_update_rejects_nonboolean_availability(available: np.ndarray) -> None:
    points, config, prior = _prior()

    with pytest.raises(ValueError, match="available must contain only booleans"):
        update_recursive_rbf_belief(
            prior,
            1,
            points,
            np.zeros((3, 3), dtype=float),
            available,
            config=config,
        )


def test_quantile_does_not_mutate_caller_mask_when_residual_is_nonfinite() -> None:
    residual = np.asarray(
        [[0.001, 0.0, 0.0], [np.nan, np.nan, np.nan], [0.003, 0.0, 0.0]]
    )
    available = np.ones(3, dtype=bool)
    expected = available.copy()

    result = finite_sample_absolute_residual_quantile_m(residual, available, 0.5)

    assert result == pytest.approx(0.0)
    np.testing.assert_array_equal(available, expected)


def test_update_does_not_mutate_caller_mask_when_residual_is_nonfinite() -> None:
    points, config, prior = _prior()
    residual = np.zeros((3, 3), dtype=float)
    residual[1] = np.nan
    available = np.ones(3, dtype=bool)
    expected = available.copy()

    posterior, reliability = update_recursive_rbf_belief(
        prior,
        1,
        points,
        residual,
        available,
        config=config,
    )

    np.testing.assert_array_equal(available, expected)
    np.testing.assert_array_equal(posterior.update_count, np.asarray([1, 0, 1]))
    assert reliability[1] == 0.0


def test_read_only_boolean_mask_is_supported_without_aliasing() -> None:
    points, config, prior = _prior()
    available = np.ones(3, dtype=bool)
    available.setflags(write=False)

    posterior, reliability = update_recursive_rbf_belief(
        prior,
        1,
        points,
        np.zeros((3, 3), dtype=float),
        available,
        config=config,
    )

    np.testing.assert_array_equal(posterior.update_count, np.ones(3, dtype=np.int64))
    assert np.all(reliability > 0.0)
    assert not available.flags.writeable


def test_v2_quantile_is_exactly_v1_for_valid_boolean_inputs() -> None:
    residual = np.asarray(
        [[0.001, -0.002, 0.003], [0.004, -0.005, 0.006], [0.007, 0.0, 0.0]]
    )
    available = np.asarray([True, False, True])

    registered = finite_sample_v1(residual, available.copy(), 0.9)
    prospective = finite_sample_absolute_residual_quantile_m(
        residual,
        available,
        0.9,
    )

    assert prospective == registered


def test_v2_update_is_exactly_v1_for_valid_boolean_inputs() -> None:
    points, config, prior = _prior()
    residual = np.asarray(
        [[0.01, 0.0, 0.0], [0.02, -0.01, 0.0], [0.03, 0.0, 0.01]]
    )
    available = np.asarray([True, False, True])

    registered, registered_reliability = update_v1(
        prior,
        3,
        points,
        residual,
        available.copy(),
        config=config,
    )
    prospective, prospective_reliability = update_recursive_rbf_belief(
        prior,
        3,
        points,
        residual,
        available,
        config=config,
    )

    np.testing.assert_array_equal(prospective.center_ids, registered.center_ids)
    np.testing.assert_array_equal(
        prospective.center_positions_m,
        registered.center_positions_m,
    )
    np.testing.assert_array_equal(prospective.global_mean_m, registered.global_mean_m)
    np.testing.assert_array_equal(
        prospective.global_variance_m2,
        registered.global_variance_m2,
    )
    np.testing.assert_array_equal(prospective.local_mean_m, registered.local_mean_m)
    np.testing.assert_array_equal(
        prospective.local_variance_m2,
        registered.local_variance_m2,
    )
    np.testing.assert_array_equal(prospective.update_count, registered.update_count)
    assert prospective.last_update_frame == registered.last_update_frame
    assert prospective.object_scale_m == registered.object_scale_m
    np.testing.assert_array_equal(prospective_reliability, registered_reliability)
