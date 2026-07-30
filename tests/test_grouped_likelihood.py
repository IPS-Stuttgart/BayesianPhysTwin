import numpy as np
import pytest
from test_grouped_conformal import (
    test_additive_bounds_are_clipped_to_nonnegative_losses,
    test_future_prediction_validation_fails_closed,
    test_group_quantile_rejects_empty_or_nonfinite_scores,
    test_group_quantile_rejects_invalid_coverage,
    test_group_score_validation_fails_closed,
    test_group_scores_weight_each_independent_unit_once,
    test_impossible_group_rank_returns_infinite_bounds,
    test_invalid_score_is_rejected,
    test_nine_groups_are_required_for_a_finite_ninety_percent_bound,
    test_result_contract_rejects_invalid_values,
    test_scaled_bounds_cover_all_registered_future_endpoints_together,
)

from bayesian_phystwin.grouped_likelihood import (
    GroupedStudentTLikelihoodConfig,
    GroupedStudentTLikelihoodResult,
    _covariance_statistics,
    grouped_student_t_mixture_likelihood,
)
from bayesian_phystwin.observation_belief import ObservationBeliefV1

_GROUPED_CONFORMAL_STABLE_TESTS = (
    test_additive_bounds_are_clipped_to_nonnegative_losses,
    test_future_prediction_validation_fails_closed,
    test_group_quantile_rejects_empty_or_nonfinite_scores,
    test_group_quantile_rejects_invalid_coverage,
    test_group_score_validation_fails_closed,
    test_group_scores_weight_each_independent_unit_once,
    test_impossible_group_rank_returns_infinite_bounds,
    test_invalid_score_is_rejected,
    test_nine_groups_are_required_for_a_finite_ninety_percent_bound,
    test_result_contract_rejects_invalid_values,
    test_scaled_bounds_cover_all_registered_future_endpoints_together,
)


def _belief() -> ObservationBeliefV1:
    local = np.repeat(np.eye(3)[None], 4, axis=0) * 1e-4
    factors = np.zeros((4, 3, 2))
    factors[:2, 0, 0] = 0.002
    factors[2:, 1, 1] = 0.003
    return ObservationBeliefV1(
        case_id="case-1",
        stream_id="prob4d:points",
        causal_frame_stop=12,
        view_names=("camera0",),
        window_names=("window0", "window1"),
        factor_names=("gauge_latent_0", "gauge_latent_1"),
        source_repository="FlorianPfaff/Prob4D",
        source_revision="a" * 40,
        source_artifact_sha256="b" * 64,
        declared_frame_ids=np.asarray([8, 9]),
        mean_xyz_m=np.asarray(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.1, 0.0, 1.0],
                [1.1, 0.0, 1.0],
            ]
        ),
        frame_ids=np.asarray([8, 8, 9, 9]),
        entity_ids=np.asarray([0, 1, 0, 1]),
        view_indices=np.zeros(4, dtype=int),
        window_indices=np.asarray([0, 0, 1, 1]),
        correlation_group_ids=np.asarray([0, 0, 1, 1]),
        factor_group_ids=np.asarray([0, 0, 1, 1]),
        prior_reliability=np.asarray([0.9, 0.8, 0.7, 0.6]),
        association_probability=np.ones(4),
        local_covariance_m2=local,
        low_rank_factor_m=factors,
        group_ids=np.asarray([0, 1]),
        group_prior_nominal_probability=np.asarray([0.85, 0.65]),
        group_composite_weight=np.asarray([0.5, 0.5]),
        metadata={"causal_source": "prefix only"},
    )


def test_prior_reliability_is_not_recomputed_from_residual() -> None:
    belief = _belief()
    clean = grouped_student_t_mixture_likelihood(belief, belief.mean_xyz_m)
    shifted = grouped_student_t_mixture_likelihood(belief, belief.mean_xyz_m + 0.2)

    np.testing.assert_array_equal(
        clean.prior_nominal_probability,
        belief.group_prior_nominal_probability,
    )
    np.testing.assert_array_equal(
        shifted.prior_nominal_probability,
        belief.group_prior_nominal_probability,
    )
    assert np.all(
        shifted.posterior_nominal_probability < clean.posterior_nominal_probability
    )
    assert 0.0 < clean.mean_posterior_nominal_probability <= 1.0


def test_low_rank_factor_increases_coherent_uncertainty() -> None:
    belief = _belief()
    shifted = belief.mean_xyz_m.copy()
    shifted[:2, 0] += 0.01

    with_factor = grouped_student_t_mixture_likelihood(belief, shifted)
    without_factor = grouped_student_t_mixture_likelihood(
        type(belief)(
            **{
                **belief.__dict__,
                "low_rank_factor_m": np.zeros_like(belief.low_rank_factor_m),
            }
        ),
        shifted,
    )

    assert (
        with_factor.covariance_mahalanobis_squared[0]
        < without_factor.covariance_mahalanobis_squared[0]
    )


def test_composite_weight_scales_group_log_evidence() -> None:
    belief = _belief()
    result = grouped_student_t_mixture_likelihood(
        belief,
        belief.mean_xyz_m,
        config=GroupedStudentTLikelihoodConfig(
            degrees_of_freedom=6.0,
        ),
    )
    np.testing.assert_allclose(
        result.weighted_negative_log_likelihood,
        result.composite_weight * result.negative_log_likelihood,
    )
    assert result.total_negative_log_likelihood == np.sum(
        result.weighted_negative_log_likelihood
    )


def test_blockwise_woodbury_matches_dense_covariance() -> None:
    original = _belief()
    belief = ObservationBeliefV1(
        **{
            **original.__dict__,
            "correlation_group_ids": np.zeros(4, dtype=np.int64),
            "group_ids": np.asarray([0], dtype=np.int64),
            "group_prior_nominal_probability": np.asarray([0.8]),
            "group_composite_weight": np.asarray([1.0]),
        }
    )
    predicted = belief.mean_xyz_m.copy()
    predicted[:, 0] += np.asarray([0.010, -0.004, 0.006, -0.002])
    config = GroupedStudentTLikelihoodConfig(
        degrees_of_freedom=7.0,
        model_discrepancy_variance_m2=2e-6,
        covariance_jitter_m2=1e-10,
    )
    result = grouped_student_t_mixture_likelihood(
        belief,
        predicted,
        config=config,
    )

    count = belief.observation_count
    dimension = 3 * count
    covariance = np.zeros((dimension, dimension), dtype=np.float64)
    diagonal_addition = (
        config.model_discrepancy_variance_m2 + config.covariance_jitter_m2
    ) * np.eye(3)
    for row, local in enumerate(belief.local_covariance_m2):
        row_slice = slice(3 * row, 3 * row + 3)
        covariance[row_slice, row_slice] = local + diagonal_addition

    rank = belief.factor_rank
    for factor_group in np.unique(belief.factor_group_ids):
        selected = belief.factor_group_ids == factor_group
        factor_matrix = np.zeros((dimension, rank), dtype=np.float64)
        factor_blocks = factor_matrix.reshape(count, 3, rank)
        factor_blocks[selected] = belief.low_rank_factor_m[selected]
        covariance += factor_matrix @ factor_matrix.T

    residual = (belief.mean_xyz_m - predicted).reshape(dimension)
    sign, expected_log_determinant = np.linalg.slogdet(covariance)
    assert sign > 0.0
    expected_mahalanobis = float(residual @ np.linalg.solve(covariance, residual))

    np.testing.assert_allclose(
        result.covariance_log_determinant_m2,
        [expected_log_determinant],
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.covariance_mahalanobis_squared,
        [expected_mahalanobis],
        rtol=1e-10,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("settings", "message"),
    (
        ({"degrees_of_freedom": 2.0}, "degrees_of_freedom"),
        ({"outlier_covariance_multiplier": 1.0}, "outlier_covariance_multiplier"),
        ({"model_discrepancy_variance_m2": -1.0}, "model_discrepancy"),
        ({"probability_floor": 0.0}, "probability_floor"),
        ({"covariance_jitter_m2": 0.0}, "covariance_jitter"),
    ),
)
def test_grouped_likelihood_config_rejects_invalid_values(
    settings: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        GroupedStudentTLikelihoodConfig(**settings)


def test_grouped_likelihood_rejects_invalid_prediction() -> None:
    belief = _belief()
    with pytest.raises(ValueError, match="match the observation mean"):
        grouped_student_t_mixture_likelihood(belief, np.zeros((3, 3)))

    nonfinite = belief.mean_xyz_m.copy()
    nonfinite[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        grouped_student_t_mixture_likelihood(belief, nonfinite)


def test_covariance_statistics_rejects_non_spd_blocks() -> None:
    with pytest.raises(ValueError, match="local covariance"):
        _covariance_statistics(
            np.zeros((1, 3)),
            -np.eye(3)[None],
            np.zeros((1, 3, 0)),
            np.zeros(1, dtype=np.int64),
            model_discrepancy_variance_m2=0.0,
            covariance_jitter_m2=1e-12,
        )


def test_covariance_statistics_reports_defensive_gram_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_cholesky = np.linalg.cholesky
    call_count = 0

    def fail_second_cholesky(matrix: np.ndarray) -> np.ndarray:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise np.linalg.LinAlgError("forced Gram failure")
        return real_cholesky(matrix)

    monkeypatch.setattr(np.linalg, "cholesky", fail_second_cholesky)
    with pytest.raises(ValueError, match="low-rank covariance update"):
        _covariance_statistics(
            np.zeros((1, 3)),
            np.eye(3)[None],
            np.ones((1, 3, 1)),
            np.zeros(1, dtype=np.int64),
            model_discrepancy_variance_m2=0.0,
            covariance_jitter_m2=1e-12,
        )


def test_grouped_likelihood_result_rejects_invalid_vectors() -> None:
    base = {
        "group_ids": np.asarray([0]),
        "dimensions": np.asarray([3]),
        "negative_log_likelihood": np.asarray([1.0]),
        "weighted_negative_log_likelihood": np.asarray([1.0]),
        "posterior_nominal_probability": np.asarray([0.8]),
        "prior_nominal_probability": np.asarray([0.9]),
        "composite_weight": np.asarray([1.0]),
        "log_nominal_density": np.asarray([-1.0]),
        "log_outlier_density": np.asarray([-2.0]),
        "mean_association_probability": np.asarray([1.0]),
        "covariance_log_determinant_m2": np.asarray([-3.0]),
        "covariance_mahalanobis_squared": np.asarray([0.5]),
    }
    with pytest.raises(ValueError, match="dimensions"):
        GroupedStudentTLikelihoodResult(
            **{
                **base,
                "dimensions": np.asarray([0]),
            }
        )
    with pytest.raises(ValueError, match="finite group vector"):
        GroupedStudentTLikelihoodResult(
            **{
                **base,
                "negative_log_likelihood": np.asarray([np.nan]),
            }
        )
