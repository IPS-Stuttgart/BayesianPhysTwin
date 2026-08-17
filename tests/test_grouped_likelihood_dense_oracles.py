from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin._prior_aware_gauge_math import (
    PriorAwareGaugeConfigV1,
    _student_t_mixture_statistics,
)
from bayesian_phystwin.grouped_likelihood import _covariance_statistics


def _random_spd_blocks(generator: np.random.Generator, count: int) -> np.ndarray:
    factors = generator.normal(size=(count, 3, 3))
    covariance = np.einsum("nij,nkj->nik", factors, factors)
    covariance += 0.25 * np.eye(3)[None, :, :]
    return covariance


def _dense_covariance(
    local_covariance: np.ndarray,
    low_rank_factor: np.ndarray,
    factor_group_ids: np.ndarray,
    *,
    diagonal_variance: float,
) -> np.ndarray:
    count = len(local_covariance)
    dense = np.zeros((3 * count, 3 * count), dtype=np.float64)
    for row, block in enumerate(local_covariance):
        selected = slice(3 * row, 3 * row + 3)
        dense[selected, selected] = block + diagonal_variance * np.eye(3)
    rank = low_rank_factor.shape[2]
    if rank:
        for group_id in np.unique(factor_group_ids):
            rows = np.flatnonzero(factor_group_ids == group_id)
            coordinates = np.concatenate(
                [np.arange(3 * row, 3 * row + 3, dtype=np.int64) for row in rows]
            )
            factor = low_rank_factor[rows].reshape(3 * len(rows), rank)
            dense[np.ix_(coordinates, coordinates)] += factor @ factor.T
    return dense


@pytest.mark.parametrize("seed", range(12))
def test_woodbury_statistics_match_a_dense_covariance_oracle(seed: int) -> None:
    generator = np.random.default_rng(seed)
    count = int(generator.integers(1, 9))
    rank = int(generator.integers(0, 5))
    local = _random_spd_blocks(generator, count)
    low_rank = generator.normal(scale=0.15, size=(count, 3, rank))
    groups = generator.integers(0, max(1, min(count, 4)), size=count, dtype=np.int64)
    residual = generator.normal(size=(count, 3))
    model_discrepancy = 0.03
    jitter = 1.0e-10

    actual_logdet, actual_mahalanobis = _covariance_statistics(
        residual,
        local,
        low_rank,
        groups,
        model_discrepancy_variance_m2=model_discrepancy,
        covariance_jitter_m2=jitter,
    )
    dense = _dense_covariance(
        local,
        low_rank,
        groups,
        diagonal_variance=model_discrepancy + jitter,
    )
    sign, expected_logdet = np.linalg.slogdet(dense)
    assert sign == 1.0
    expected_mahalanobis = float(
        residual.reshape(-1) @ np.linalg.solve(dense, residual.reshape(-1))
    )

    assert actual_logdet == pytest.approx(expected_logdet, rel=2.0e-11, abs=2.0e-11)
    assert actual_mahalanobis == pytest.approx(
        expected_mahalanobis,
        rel=2.0e-11,
        abs=2.0e-11,
    )


def test_woodbury_statistics_are_invariant_to_row_permutation() -> None:
    generator = np.random.default_rng(20260809)
    count = 8
    local = _random_spd_blocks(generator, count)
    low_rank = generator.normal(scale=0.1, size=(count, 3, 3))
    groups = np.asarray([0, 1, 0, 2, 1, 2, 0, 1], dtype=np.int64)
    residual = generator.normal(size=(count, 3))
    permutation = generator.permutation(count)

    reference = _covariance_statistics(
        residual,
        local,
        low_rank,
        groups,
        model_discrepancy_variance_m2=0.02,
        covariance_jitter_m2=1.0e-12,
    )
    permuted = _covariance_statistics(
        residual[permutation],
        local[permutation],
        low_rank[permutation],
        groups[permutation],
        model_discrepancy_variance_m2=0.02,
        covariance_jitter_m2=1.0e-12,
    )

    assert permuted[0] == pytest.approx(reference[0], rel=1.0e-12, abs=1.0e-12)
    assert permuted[1] == pytest.approx(reference[1], rel=1.0e-12, abs=1.0e-12)


def test_nearly_collinear_low_rank_factors_match_the_dense_oracle() -> None:
    generator = np.random.default_rng(71)
    count = 6
    local = _random_spd_blocks(generator, count)
    first = generator.normal(scale=0.2, size=(count, 3, 1))
    low_rank = np.concatenate(
        (first, first * (1.0 + 1.0e-10), first * (1.0 - 1.0e-10)),
        axis=2,
    )
    groups = np.zeros(count, dtype=np.int64)
    residual = generator.normal(size=(count, 3))

    actual = _covariance_statistics(
        residual,
        local,
        low_rank,
        groups,
        model_discrepancy_variance_m2=0.0,
        covariance_jitter_m2=1.0e-12,
    )
    dense = _dense_covariance(
        local,
        low_rank,
        groups,
        diagonal_variance=1.0e-12,
    )
    sign, expected_logdet = np.linalg.slogdet(dense)
    assert sign == 1.0
    expected_mahalanobis = float(
        residual.reshape(-1) @ np.linalg.solve(dense, residual.reshape(-1))
    )

    assert actual[0] == pytest.approx(expected_logdet, rel=1.0e-10, abs=1.0e-10)
    assert actual[1] == pytest.approx(
        expected_mahalanobis,
        rel=1.0e-10,
        abs=1.0e-10,
    )


@pytest.mark.parametrize(
    ("squared_mahalanobis", "dimension", "prior_nominal"),
    [
        (0.2, 3, 0.05),
        (1.5, 12, 0.5),
        (9.0, 21, 0.95),
        (40.0, 48, 0.999),
    ],
)
def test_student_t_mixture_score_and_curvature_match_finite_differences(
    squared_mahalanobis: float,
    dimension: int,
    prior_nominal: float,
) -> None:
    config = PriorAwareGaugeConfigV1(
        degrees_of_freedom=6.5,
        outlier_covariance_multiplier=30.0,
        probability_floor=1.0e-8,
        minimum_robust_precision=0.0,
    )
    statistics = _student_t_mixture_statistics(
        squared_mahalanobis,
        dimension,
        prior_nominal,
        config,
    )
    step = 2.0e-4 * max(1.0, squared_mahalanobis)

    def log_density(value: float) -> float:
        return _student_t_mixture_statistics(
            value,
            dimension,
            prior_nominal,
            config,
        ).log_mixture_density

    left = log_density(squared_mahalanobis - step)
    center = log_density(squared_mahalanobis)
    right = log_density(squared_mahalanobis + step)
    numerical_score = (right - left) / (2.0 * step)
    numerical_curvature = (right - 2.0 * center + left) / step**2

    assert numerical_score == pytest.approx(
        -0.5 * statistics.expected_precision,
        rel=2.0e-6,
        abs=2.0e-8,
    )
    assert numerical_curvature == pytest.approx(
        -0.5 * statistics.expected_precision_derivative,
        rel=2.0e-4,
        abs=2.0e-7,
    )
