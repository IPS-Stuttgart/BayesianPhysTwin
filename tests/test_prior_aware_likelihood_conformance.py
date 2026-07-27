import math

import numpy as np
import pytest

from bayesian_phystwin._gauge_aware_contracts import GaugeAwareObservationBatch
from bayesian_phystwin._prior_aware_gauge_math import (
    PriorAwareGaugeConfigV1,
    _solve_spd_posterior,
    _student_t_mixture_statistics,
)
from bayesian_phystwin.prior_aware_gauge_belief import (
    update_prior_aware_gauge_belief,
)


def _direct_component(
    squared_mahalanobis: float,
    dimension: int,
    covariance_multiplier: float,
    config: PriorAwareGaugeConfigV1,
) -> tuple[float, float]:
    scale_multiplier = (
        (config.degrees_of_freedom - 2.0)
        / config.degrees_of_freedom
        * covariance_multiplier
    )
    log_density = (
        math.lgamma(0.5 * (config.degrees_of_freedom + dimension))
        - math.lgamma(0.5 * config.degrees_of_freedom)
        - 0.5
        * (
            dimension * math.log(config.degrees_of_freedom * math.pi)
            + dimension * math.log(scale_multiplier)
        )
        - 0.5
        * (config.degrees_of_freedom + dimension)
        * math.log1p(
            squared_mahalanobis
            / (config.degrees_of_freedom * scale_multiplier)
        )
    )
    precision = (
        config.degrees_of_freedom + dimension
    ) / (
        config.degrees_of_freedom * scale_multiplier
        + squared_mahalanobis
    )
    return log_density, precision


def _direct_statistics(
    squared_mahalanobis: float,
    dimension: int,
    prior_nominal_probability: float,
    config: PriorAwareGaugeConfigV1,
) -> tuple[float, float, float]:
    rho = float(
        np.clip(
            prior_nominal_probability,
            config.probability_floor,
            1.0 - config.probability_floor,
        )
    )
    log_nominal, nominal_precision = _direct_component(
        squared_mahalanobis,
        dimension,
        1.0,
        config,
    )
    log_outlier, outlier_precision = _direct_component(
        squared_mahalanobis,
        dimension,
        config.outlier_covariance_multiplier,
        config,
    )
    weighted_nominal = math.log(rho) + log_nominal
    weighted_outlier = math.log1p(-rho) + log_outlier
    log_mixture = float(np.logaddexp(weighted_nominal, weighted_outlier))
    responsibility = math.exp(weighted_nominal - log_mixture)
    precision = (
        responsibility * nominal_precision
        + (1.0 - responsibility) * outlier_precision
    )
    return log_mixture, responsibility, precision


@pytest.mark.parametrize(
    ("dimension", "residual"),
    [
        (1, np.asarray([0.7])),
        (3, np.asarray([0.7, -0.2, 0.4])),
    ],
)
def test_mixture_statistics_match_direct_density_and_score(
    dimension: int,
    residual: np.ndarray,
) -> None:
    config = PriorAwareGaugeConfigV1(minimum_robust_precision=0.0)
    squared_mahalanobis = float(residual @ residual)

    actual = _student_t_mixture_statistics(
        squared_mahalanobis,
        dimension,
        0.73,
        config,
    )
    expected = _direct_statistics(
        squared_mahalanobis,
        dimension,
        0.73,
        config,
    )

    assert actual.log_mixture_density == pytest.approx(expected[0], rel=1e-13)
    assert actual.posterior_nominal_probability == pytest.approx(
        expected[1], rel=1e-13
    )
    assert actual.expected_precision == pytest.approx(expected[2], rel=1e-13)
    assert not actual.precision_floor_active


@pytest.mark.parametrize(
    "residual",
    [
        np.asarray([0.6]),
        np.asarray([0.6, -0.3, 0.2]),
    ],
)
def test_mixture_precision_is_negative_log_density_gradient(
    residual: np.ndarray,
) -> None:
    config = PriorAwareGaugeConfigV1(minimum_robust_precision=0.0)
    dimension = len(residual)
    statistics = _student_t_mixture_statistics(
        float(residual @ residual),
        dimension,
        0.61,
        config,
    )
    numerical_gradient = np.empty(dimension)
    step = 1e-6
    for index in range(dimension):
        offset = np.zeros(dimension)
        offset[index] = step
        plus = _student_t_mixture_statistics(
            float((residual + offset) @ (residual + offset)),
            dimension,
            0.61,
            config,
        )
        minus = _student_t_mixture_statistics(
            float((residual - offset) @ (residual - offset)),
            dimension,
            0.61,
            config,
        )
        numerical_gradient[index] = -(
            plus.log_mixture_density - minus.log_mixture_density
        ) / (2.0 * step)

    np.testing.assert_allclose(
        numerical_gradient,
        statistics.expected_precision * residual,
        rtol=2e-7,
        atol=2e-9,
    )


def _empty_design(count: int) -> np.ndarray:
    return np.zeros((count, 3, 0), dtype=np.float64)


def _batch(
    innovation_x: np.ndarray,
    *,
    reliability: np.ndarray | None = None,
    composite_weight: float = 1.0,
) -> GaugeAwareObservationBatch:
    count = len(innovation_x)
    row_reliability = (
        np.ones(count)
        if reliability is None
        else np.asarray(reliability, dtype=np.float64)
    )
    innovation = np.zeros((count, 3), dtype=np.float64)
    innovation[:, 0] = innovation_x
    state = np.zeros((count, 3, 1), dtype=np.float64)
    state[:, 0, 0] = 1.0
    return GaugeAwareObservationBatch(
        innovation_m=innovation,
        observation_covariance_m2=np.tile(
            np.eye(3, dtype=np.float64) * 0.01,
            (count, 1, 1),
        ),
        state_jacobian=state,
        gauge_jacobian=_empty_design(count),
        shared_bias_jacobian=_empty_design(count),
        view_bias_jacobian=_empty_design(count),
        query_state_jacobian=state.copy(),
        gauge_prior_covariance=np.zeros((0, 0), dtype=np.float64),
        correlation_group_ids=("group-0",) * count,
        prior_reliability=row_reliability,
        prior_nominal_probability=np.full(count, 0.7),
        composite_weight=np.full(count, composite_weight),
        physical_response_scale_m=1.0,
        state_prior_covariance_m2=np.asarray([[0.04]]),
    )


def _config() -> PriorAwareGaugeConfigV1:
    return PriorAwareGaugeConfigV1(
        effective_samples_per_correlation_group=64.0,
        degrees_of_freedom=5.0,
        outlier_covariance_multiplier=25.0,
        minimum_robust_precision=0.0,
        maximum_iterations=50,
        convergence_tolerance=1e-12,
        minimum_conditional_information_fraction=0.0,
        minimum_identifiable_fraction=0.01,
        minimum_query_sensitivity_fraction=0.0,
        maximum_state_update_m=2.0,
        maximum_update_to_physical_response_ratio=2.0,
    )


def test_solver_map_responsibility_and_working_curvature_match_dense_grid() -> None:
    batch = _batch(np.asarray([0.1]))
    config = _config()
    result = update_prior_aware_gauge_belief(batch, config=config)

    assert result.inference_admissible
    grid = np.linspace(-0.25, 0.25, 20_001)
    squared_mahalanobis = np.square(0.1 - grid) / 0.01
    scale = (config.degrees_of_freedom - 2.0) / config.degrees_of_freedom

    def log_component(multiplier: float) -> np.ndarray:
        component_scale = scale * multiplier
        return (
            math.lgamma(0.5 * (config.degrees_of_freedom + 3))
            - math.lgamma(0.5 * config.degrees_of_freedom)
            - 1.5
            * math.log(config.degrees_of_freedom * math.pi)
            - 1.5 * math.log(component_scale)
            - 0.5
            * (config.degrees_of_freedom + 3)
            * np.log1p(
                squared_mahalanobis
                / (config.degrees_of_freedom * component_scale)
            )
        )

    log_mixture = np.logaddexp(
        math.log(0.7) + log_component(1.0),
        math.log(0.3)
        + log_component(config.outlier_covariance_multiplier),
    )
    objective = 0.5 * np.square(grid) / 0.04 - log_mixture
    dense_map = float(grid[int(np.argmin(objective))])

    state = float(result.state_coefficients[0])
    assert state == pytest.approx(dense_map, abs=5e-5)
    final_statistics = _student_t_mixture_statistics(
        (0.1 - state) ** 2 / 0.01,
        3,
        0.7,
        config,
    )
    posterior = result.diagnostics[
        "observation_group_posterior_nominal_probability"
    ]
    assert posterior == pytest.approx(
        [final_statistics.posterior_nominal_probability],
        rel=2e-9,
        abs=2e-11,
    )
    assert result.robust_weights[0] == pytest.approx(
        final_statistics.expected_precision,
        rel=2e-9,
    )

    expected_working_variance = 1.0 / (
        1.0 / 0.04 + final_statistics.expected_precision / 0.01
    )
    assert result.posterior_covariance[0, 0] == pytest.approx(
        expected_working_variance,
        rel=2e-9,
    )

    curvature_step = 1e-5

    def posterior_objective(value: float) -> float:
        statistics = _student_t_mixture_statistics(
            (0.1 - value) ** 2 / 0.01,
            3,
            0.7,
            config,
        )
        return 0.5 * value**2 / 0.04 - statistics.log_mixture_density

    exact_state_curvature = (
        posterior_objective(state + curvature_step)
        - 2.0 * posterior_objective(state)
        + posterior_objective(state - curvature_step)
    ) / curvature_step**2
    state_mapping = float(result.identifiable_state_transform[0, 0])
    assert result.diagnostics[
        "exact_reduced_mixture_hessian_minimum_eigenvalue"
    ] == pytest.approx(
        exact_state_curvature * state_mapping**2,
        rel=3e-6,
    )
    assert result.diagnostics["robust_likelihood_objective"] == (
        "exact-group-mixture-gradient"
    )
    assert result.diagnostics["posterior_covariance_kind"] == (
        "working-gauss-newton-irls-not-exact-mixture-hessian"
    )


def test_one_group_receives_one_shared_mixture_responsibility() -> None:
    batch = _batch(np.asarray([0.02, 0.25]))
    config = _config()
    result = update_prior_aware_gauge_belief(batch, config=config)

    assert result.inference_admissible
    state = float(result.state_coefficients[0])
    squared_mahalanobis = float(
        np.sum(np.square(batch.innovation_m[:, 0] - state) / 0.01)
    )
    statistics = _student_t_mixture_statistics(
        squared_mahalanobis,
        6,
        0.7,
        config,
    )
    np.testing.assert_allclose(
        result.robust_weights,
        statistics.expected_precision,
        rtol=2e-8,
        atol=2e-10,
    )
    assert result.diagnostics[
        "observation_group_posterior_nominal_probability"
    ] == pytest.approx(
        [statistics.posterior_nominal_probability],
        rel=2e-8,
        abs=2e-10,
    )


def test_zero_reliability_row_is_inert_for_group_mixture() -> None:
    config = _config()
    reference = update_prior_aware_gauge_belief(
        _batch(np.asarray([0.02])),
        config=config,
    )
    augmented = update_prior_aware_gauge_belief(
        _batch(
            np.asarray([0.02, 100.0]),
            reliability=np.asarray([1.0, 0.0]),
        ),
        config=config,
    )

    assert reference.inference_admissible
    assert augmented.inference_admissible
    np.testing.assert_allclose(
        augmented.state_coefficients,
        reference.state_coefficients,
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        augmented.posterior_covariance,
        reference.posterior_covariance,
        rtol=0.0,
        atol=1e-14,
    )
    assert augmented.diagnostics[
        "observation_group_posterior_nominal_probability"
    ] == pytest.approx(
        reference.diagnostics[
            "observation_group_posterior_nominal_probability"
        ],
        abs=1e-14,
    )


def test_row_reliability_scales_group_mahalanobis_distance() -> None:
    reliability = np.asarray([1.0, 0.25])
    batch = _batch(
        np.asarray([0.02, 0.25]),
        reliability=reliability,
    )
    config = _config()
    result = update_prior_aware_gauge_belief(batch, config=config)

    assert result.inference_admissible
    state = float(result.state_coefficients[0])
    squared_mahalanobis = float(
        np.sum(
            reliability
            * np.square(batch.innovation_m[:, 0] - state)
            / 0.01
        )
    )
    statistics = _student_t_mixture_statistics(
        squared_mahalanobis,
        6,
        0.7,
        config,
    )
    np.testing.assert_allclose(
        result.robust_weights,
        statistics.expected_precision,
        rtol=2e-8,
        atol=2e-10,
    )
    assert result.diagnostics["observation_group_power"] == pytest.approx([1.0])


def test_exact_mixture_is_default_and_precision_floor_is_explicit() -> None:
    exact = PriorAwareGaugeConfigV1()
    assert exact.minimum_robust_precision == 0.0
    exact_statistics = _student_t_mixture_statistics(
        1e6,
        3,
        0.8,
        exact,
    )
    assert exact_statistics.expected_precision < 0.01
    assert not exact_statistics.precision_floor_active

    floored = PriorAwareGaugeConfigV1(minimum_robust_precision=0.01)
    floored_statistics = _student_t_mixture_statistics(
        1e6,
        3,
        0.8,
        floored,
    )
    assert floored_statistics.expected_precision == pytest.approx(0.01)
    assert floored_statistics.precision_floor_active


def test_spd_posterior_solve_returns_symmetric_positive_covariance() -> None:
    normal = np.asarray([[4.0, 1.0], [1.0, 3.0]])
    right = np.asarray([1.0, 2.0])

    solution, covariance = _solve_spd_posterior(normal, right)

    np.testing.assert_allclose(solution, np.linalg.solve(normal, right))
    np.testing.assert_allclose(covariance, np.linalg.solve(normal, np.eye(2)))
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-15)
    assert np.min(np.linalg.eigvalsh(covariance)) > 0.0
