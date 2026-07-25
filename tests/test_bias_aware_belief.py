import numpy as np
import pytest

from bayesian_phystwin.bias_aware_belief import (
    BiasAwareStateUpdateConfig,
    apply_group_regret_bound,
    apply_regret_guard,
    build_physical_response_basis,
    decode_bias_aware_state,
    fit_source_regret_certificate,
    fit_source_group_regret_bound,
    restrict_state_basis_to_identifiable_subspace,
    update_bias_aware_state,
)
from bayesian_phystwin.bias_aware_belief_benchmark import (
    BiasAwareBeliefBenchmarkConfig,
    run_bias_aware_belief_benchmark,
)


def _constant_camera_problem(
    *, view_count: int = 3, point_count: int = 8
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    innovation = np.zeros((view_count, point_count, 3), dtype=np.float64)
    innovation[..., 0] = 0.03
    available = np.ones((view_count, point_count), dtype=bool)
    state_basis = np.ones((point_count, 1), dtype=np.float64)
    shared_bias_basis = np.ones((point_count, 1), dtype=np.float64)
    return innovation, available, state_basis, shared_bias_basis


def _centered_mode(point_count: int) -> np.ndarray:
    return np.linspace(-1.0, 1.0, point_count)[:, None]


def test_physical_response_basis_respects_action_support() -> None:
    response = np.zeros((3, 10, 3), dtype=np.float64)
    response[:, :5, 0] = np.linspace(0.0, 0.02, 3)[:, None]
    response[:, :5, 1] = np.linspace(0.0, 0.01, 5)[None]
    support = np.zeros(10)
    support[:5] = 1.0

    physical_basis = build_physical_response_basis(
        response,
        action_support=support,
        rank=2,
    )

    assert physical_basis.basis.shape == (10, 2)
    np.testing.assert_array_equal(physical_basis.basis[5:], 0.0)
    assert physical_basis.supported_point_count == 5
    assert physical_basis.maximum_response_m > 0.02
    assert 0.99 <= physical_basis.explained_energy_fraction <= 1.0


def test_physical_response_basis_rejects_actionless_window() -> None:
    with pytest.raises(ValueError, match="below the declared support threshold"):
        build_physical_response_basis(np.zeros((3, 8, 3)))


def test_identifiable_subspace_removes_only_confounded_reachable_mode() -> None:
    point_count = 12
    global_mode = np.ones(point_count)
    local_mode = np.linspace(-1.0, 1.0, point_count)
    query_basis = np.column_stack((global_mode, local_mode))
    bias_design = global_mode[:, None]

    result = restrict_state_basis_to_identifiable_subspace(
        query_basis,
        query_basis,
        bias_design,
    )

    assert result.query_basis.shape == (point_count, 1)
    np.testing.assert_allclose(
        np.abs(result.query_basis[:, 0]), np.abs(local_mode), atol=1e-12
    )
    assert result.identifiable_fractions[0] == pytest.approx(1.0)


def test_identifiable_subspace_rejects_fully_confounded_response() -> None:
    state = np.ones((8, 1))
    with pytest.raises(ValueError, match="fully confounded"):
        restrict_state_basis_to_identifiable_subspace(state, state, state)


def test_unanchored_common_mode_returns_exact_zero_update() -> None:
    innovation, available, state_basis, bias_basis = _constant_camera_problem()

    result = update_bias_aware_state(
        innovation,
        available,
        state_basis,
        bias_basis,
    )

    assert not result.accepted
    assert result.reason == "unanchored-common-mode-ambiguity"
    assert result.diagnostics["state_bias_subspace_cosine"] == pytest.approx(1.0)
    update = decode_bias_aware_state(result, state_basis)
    assert update.tobytes() == np.zeros_like(update).tobytes()


def test_independent_anchor_separates_state_from_shared_camera_bias() -> None:
    innovation, available, state_basis, bias_basis = _constant_camera_problem()
    config = BiasAwareStateUpdateConfig(
        observation_std_m=0.001,
        anchor_std_m=0.0001,
        state_prior_std_m=0.1,
        shared_bias_prior_std_m=0.1,
        camera_bias_prior_std_m=0.1,
    )

    result = update_bias_aware_state(
        innovation,
        available,
        state_basis,
        bias_basis,
        anchor_innovation_m=np.asarray([[0.01, 0.0, 0.0]]),
        anchor_state_basis=np.ones((1, 1)),
        config=config,
    )

    assert result.accepted
    assert result.state_coefficients_m[0, 0] == pytest.approx(0.01, abs=2e-4)
    reconstructed = (
        result.state_coefficients_m[0]
        + result.shared_bias_coefficients_m[0]
        + result.camera_biases_m
    )
    np.testing.assert_allclose(reconstructed[:, 0], 0.03, atol=3e-4)
    assert "maximum_state_bias_posterior_correlation" in result.diagnostics


def test_action_local_state_is_recovered_alongside_common_camera_bias() -> None:
    point_count = 10
    state_basis = _centered_mode(point_count)
    shared_bias_basis = np.ones((point_count, 1))
    innovation = np.zeros((4, point_count, 3), dtype=np.float64)
    innovation[..., 0] = 0.01 * state_basis[:, 0] + 0.02

    result = update_bias_aware_state(
        innovation,
        np.ones((4, point_count), dtype=bool),
        state_basis,
        shared_bias_basis,
        config=BiasAwareStateUpdateConfig(
            observation_std_m=0.001,
            state_prior_std_m=0.1,
            shared_bias_prior_std_m=0.1,
            camera_bias_prior_std_m=0.1,
        ),
    )

    assert result.accepted
    assert result.state_coefficients_m[0, 0] == pytest.approx(0.01, abs=2e-4)
    assert result.diagnostics["state_bias_subspace_cosine"] < 1e-10


def test_independent_anchor_can_update_without_camera_support() -> None:
    point_count = 6
    result = update_bias_aware_state(
        np.full((2, point_count, 3), np.nan),
        np.zeros((2, point_count), dtype=bool),
        np.ones((point_count, 1)),
        np.ones((point_count, 1)),
        anchor_innovation_m=np.asarray([[0.012, 0.0, 0.0]]),
        anchor_state_basis=np.ones((1, 1)),
        config=BiasAwareStateUpdateConfig(anchor_std_m=0.0001),
    )

    assert result.accepted
    assert result.state_coefficients_m[0, 0] == pytest.approx(0.012, abs=1e-4)
    np.testing.assert_array_equal(result.prior_reliability, 0.0)


def test_duplicate_correlated_views_do_not_increase_state_confidence() -> None:
    point_count = 24
    state_basis = _centered_mode(point_count)
    one_view = np.zeros((1, point_count, 3), dtype=np.float64)
    one_view[0, :, 0] = 0.01 * state_basis[:, 0]
    many_views = np.repeat(one_view, 8, axis=0)
    config = BiasAwareStateUpdateConfig(effective_samples_per_view=16.0)

    single = update_bias_aware_state(
        one_view,
        np.ones((1, point_count), dtype=bool),
        state_basis,
        np.zeros((point_count, 0)),
        config=config,
    )
    duplicated = update_bias_aware_state(
        many_views,
        np.ones((8, point_count), dtype=bool),
        state_basis,
        np.zeros((point_count, 0)),
        config=config,
    )

    assert single.accepted and duplicated.accepted
    assert duplicated.posterior_covariance_m2[0, 0] == pytest.approx(
        single.posterior_covariance_m2[0, 0], rel=1e-10
    )
    prior_precision = 1.0 / config.state_prior_std_m**2
    one_view_information = (
        config.effective_samples_per_view
        * np.mean(np.square(state_basis[:, 0]))
        / config.observation_std_m**2
    )
    naive_independent_variance = 1.0 / (
        prior_precision + 8.0 * one_view_information
    )
    assert duplicated.posterior_covariance_m2[0, 0] > naive_independent_variance


def test_duplicate_correlated_pixels_hit_effective_sample_cap() -> None:
    base_basis = _centered_mode(16)
    base = np.zeros((1, 16, 3), dtype=np.float64)
    base[0, :, 0] = 0.01 * base_basis[:, 0]
    duplicated = np.tile(base, (1, 8, 1))
    duplicated_basis = np.tile(base_basis, (8, 1))
    config = BiasAwareStateUpdateConfig(effective_samples_per_view=16.0)

    base_result = update_bias_aware_state(
        base,
        np.ones((1, 16), dtype=bool),
        base_basis,
        np.zeros((16, 0)),
        config=config,
    )
    duplicated_result = update_bias_aware_state(
        duplicated,
        np.ones((1, 128), dtype=bool),
        duplicated_basis,
        np.zeros((128, 0)),
        config=config,
    )

    assert duplicated_result.posterior_covariance_m2[0, 0] == pytest.approx(
        base_result.posterior_covariance_m2[0, 0], rel=1e-10
    )


def test_innovation_changes_robust_weight_not_prior_reliability() -> None:
    point_count = 9
    state_basis = _centered_mode(point_count)
    nominal = np.zeros((1, point_count, 3), dtype=np.float64)
    nominal[0, :, 0] = 0.01 * state_basis[:, 0]
    outlier = nominal.copy()
    outlier[0, -1, 0] += 0.20
    reliability = np.linspace(0.4, 1.0, point_count)[None]

    clean = update_bias_aware_state(
        nominal,
        np.ones((1, point_count), dtype=bool),
        state_basis,
        np.zeros((point_count, 0)),
        prior_reliability=reliability,
    )
    contaminated = update_bias_aware_state(
        outlier,
        np.ones((1, point_count), dtype=bool),
        state_basis,
        np.zeros((point_count, 0)),
        prior_reliability=reliability,
    )

    np.testing.assert_array_equal(clean.prior_reliability, reliability)
    np.testing.assert_array_equal(contaminated.prior_reliability, reliability)
    assert contaminated.robust_weights[0, -1] < clean.robust_weights[0, -1]
    assert contaminated.state_coefficients_m[0, 0] == pytest.approx(0.01, abs=0.005)


def _source_regret_problem(
    regret_offset: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feature_values = np.tile(np.asarray([0.0, 0.5, 1.0]), 4)
    features = feature_values[:, None]
    groups = [f"object-{index}" for index in range(4) for _ in range(3)]
    regret = regret_offset - 0.002 * feature_values
    return features, regret, groups


def test_regret_guard_accepts_supported_source_calibrated_benefit() -> None:
    features, regret, groups = _source_regret_problem(-0.02)
    certificate = fit_source_regret_certificate(
        features,
        regret,
        groups,
        ridge_penalty=0.1,
    )

    decision = apply_regret_guard(
        np.asarray([1.0, 2.0]),
        np.asarray([0.5, 1.5]),
        np.asarray([0.5]),
        certificate,
    )

    assert decision.candidate_accepted
    np.testing.assert_array_equal(decision.selected_value, [0.5, 1.5])
    assert decision.upper_regret < 0.0


def test_regret_guard_returns_baseline_bit_exact_when_uncertified() -> None:
    features, regret, groups = _source_regret_problem(0.02)
    certificate = fit_source_regret_certificate(features, regret, groups)
    baseline = np.asarray([0.0, -0.0, 1.25], dtype=np.float32)

    decision = apply_regret_guard(
        baseline,
        np.asarray([9.0, 9.0, 9.0], dtype=np.float32),
        np.asarray([0.5]),
        certificate,
    )

    assert not decision.candidate_accepted
    assert decision.reason == "exact-baseline-fallback"
    assert decision.selected_value.dtype == baseline.dtype
    assert decision.selected_value.tobytes() == baseline.tobytes()


def test_regret_guard_rejects_extrapolative_candidate() -> None:
    features, regret, groups = _source_regret_problem(-0.02)
    certificate = fit_source_regret_certificate(features, regret, groups)
    baseline = np.asarray([1.0])

    decision = apply_regret_guard(
        baseline,
        np.asarray([0.0]),
        np.asarray([3.0]),
        certificate,
    )

    assert not decision.candidate_accepted
    assert decision.reason == "outside-source-support-exact-baseline-fallback"
    assert np.isinf(decision.upper_regret)
    assert decision.selected_value.tobytes() == baseline.tobytes()


def test_group_cross_fit_expands_bound_for_one_bad_source_object() -> None:
    features, regret, groups = _source_regret_problem(-0.02)
    regret = regret.copy()
    regret[np.asarray(groups) == "object-3"] = 0.08

    certificate = fit_source_regret_certificate(features, regret, groups)

    assert certificate.upper_residual_quantile > 0.05
    assert certificate.upper_regret(np.asarray([0.5])) > 0.0
    assert certificate.finite_sample_coverage == pytest.approx(0.8)


def test_source_group_regret_bound_reports_achievable_resolution() -> None:
    regret = np.asarray([-0.003, -0.002, -0.004, -0.001])
    groups = ["a", "b", "c", "d"]

    bound = fit_source_group_regret_bound(regret, groups)

    assert bound.finite_sample_rank == 4
    assert bound.finite_sample_coverage == pytest.approx(0.8)
    assert bound.upper_regret_m == pytest.approx(-0.001)
    assert bound.candidate_certified
    decision = apply_group_regret_bound(
        np.asarray([1.0]), np.asarray([0.0]), bound
    )
    assert decision.candidate_accepted
    np.testing.assert_array_equal(decision.selected_value, [0.0])


def test_source_group_regret_bound_falls_back_on_harmful_group() -> None:
    bound = fit_source_group_regret_bound(
        np.asarray([-0.003, -0.002, 0.001, -0.004]),
        ["a", "b", "c", "d"],
    )
    baseline = np.asarray([0.0, -0.0], dtype=np.float32)

    decision = apply_group_regret_bound(
        baseline,
        np.asarray([1.0, 1.0], dtype=np.float32),
        bound,
    )

    assert not decision.candidate_accepted
    assert decision.selected_value.tobytes() == baseline.tobytes()


def test_synthetic_mechanism_controls_pass() -> None:
    result = run_bias_aware_belief_benchmark(
        BiasAwareBeliefBenchmarkConfig(
            seed=7,
            trial_count=24,
            target_sample_count=96,
        )
    )

    assert result["all_gates_pass"], result
