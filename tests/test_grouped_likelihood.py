import numpy as np

from bayesian_phystwin.grouped_likelihood import (
    GroupedStudentTLikelihoodConfig,
    grouped_student_t_mixture_likelihood,
)
from bayesian_phystwin.observation_belief import ObservationBeliefV1


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
