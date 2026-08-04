from dataclasses import dataclass

import numpy as np
import pytest

from bayesian_phystwin.gauge_aware_belief import (
    GaugeAwareBeliefConfig,
    GaugeAwareObservationBatch,
    decode_gauge_aware_query,
    select_gauge_aware_candidate,
    update_gauge_aware_belief,
)


def _design(mode: np.ndarray) -> np.ndarray:
    result = np.zeros((len(mode), 3, 1), dtype=np.float64)
    result[:, 0, 0] = mode
    return result


def _empty_design(count: int) -> np.ndarray:
    return np.zeros((count, 3, 0), dtype=np.float64)


def _batch(
    mode: np.ndarray,
    innovation_x: np.ndarray,
    *,
    gauge_mode: np.ndarray | None = None,
    groups: tuple[str, ...] | None = None,
    physical_response_scale_m: float = 0.05,
    anchor_state: np.ndarray | None = None,
    anchor_innovation_x: np.ndarray | None = None,
    anchor_groups: tuple[str, ...] | None = None,
    anchor_bias_mode: np.ndarray | None = None,
) -> GaugeAwareObservationBatch:
    count = len(mode)
    gauge = _empty_design(count) if gauge_mode is None else _design(gauge_mode)
    gauge_count = gauge.shape[2]
    innovation = np.zeros((count, 3), dtype=np.float64)
    innovation[:, 0] = innovation_x
    kwargs = {}
    if anchor_state is not None:
        anchor_innovation = np.zeros((len(anchor_state), 3), dtype=np.float64)
        anchor_innovation[:, 0] = anchor_innovation_x
        kwargs = {
            "anchor_innovation_m": anchor_innovation,
            "anchor_covariance_m2": np.tile(
                np.eye(3) * 1e-8, (len(anchor_state), 1, 1)
            ),
            "anchor_state_jacobian": _design(anchor_state),
            "anchor_correlation_group_ids": anchor_groups,
        }
        if anchor_bias_mode is not None:
            kwargs["anchor_bias_jacobian"] = _design(anchor_bias_mode)
            kwargs["anchor_bias_prior_covariance"] = np.asarray([[0.01]])
    return GaugeAwareObservationBatch(
        innovation_m=innovation,
        observation_covariance_m2=np.tile(np.eye(3) * 1e-6, (count, 1, 1)),
        state_jacobian=_design(mode),
        gauge_jacobian=gauge,
        shared_bias_jacobian=_empty_design(count),
        view_bias_jacobian=_empty_design(count),
        query_state_jacobian=_design(mode),
        gauge_prior_covariance=np.eye(gauge_count) * 0.01,
        correlation_group_ids=groups or tuple("window-0" for _ in range(count)),
        prior_reliability=np.ones(count),
        physical_response_scale_m=physical_response_scale_m,
        state_prior_covariance_m2=np.asarray([[0.01]]),
        metadata={"observation_artifact_id": "a" * 64},
        **kwargs,
    )


@dataclass(frozen=True)
class _GuardDecision:
    selected_value: np.ndarray
    candidate_accepted: bool
    reason: str = "test"


def test_gauge_aware_update_separates_translation_gauge_from_local_state() -> None:
    mode = np.linspace(-1.0, 1.0, 12)
    batch = _batch(
        mode,
        0.01 * mode + 0.03,
        gauge_mode=np.ones_like(mode),
    )

    result = update_gauge_aware_belief(
        batch,
        config=GaugeAwareBeliefConfig(
            effective_samples_per_correlation_group=12.0,
        ),
    )

    assert result.inference_admissible
    assert result.accepted
    assert result.reason == "inference-admissible"
    assert result.state_coefficients[0] == pytest.approx(0.01, abs=2e-4)
    assert result.gauge_delta[0] == pytest.approx(0.03, abs=2e-4)
    assert result.identifiable_fractions[0] == pytest.approx(1.0)
    assert result.input_lineage["observation_artifact_id"] == "a" * 64
    decoded = decode_gauge_aware_query(result, batch.query_state_jacobian)
    np.testing.assert_allclose(decoded[:, 0], 0.01 * mode, atol=2e-4)


def test_fully_gauge_confounded_query_retains_prior_and_falls_back_bit_exact() -> None:
    mode = np.ones(10)
    batch = _batch(mode, np.full(10, 0.03), gauge_mode=mode)

    result = update_gauge_aware_belief(batch)
    baseline = np.asarray([0.0, -0.0, 1.5], dtype=np.float32)
    selection = select_gauge_aware_candidate(
        baseline,
        np.asarray([9.0, 9.0, 9.0], dtype=np.float32),
        result,
    )

    assert not result.inference_admissible
    assert result.reason == "no-identifiable-query-state"
    assert result.posterior_covariance[0, 0] == pytest.approx(0.01)
    assert result.posterior_covariance[1, 1] == pytest.approx(0.01)
    assert not selection.candidate_accepted
    assert not selection.regret_guard_present
    assert selection.selected_value.dtype == baseline.dtype
    assert selection.selected_value.tobytes() == baseline.tobytes()


def test_independent_anchor_breaks_state_gauge_ambiguity() -> None:
    mode = np.ones(10)
    batch = _batch(
        mode,
        np.full(10, 0.03),
        gauge_mode=mode,
        anchor_state=np.ones(2),
        anchor_innovation_x=np.full(2, 0.01),
    )

    result = update_gauge_aware_belief(
        batch,
        config=GaugeAwareBeliefConfig(
            effective_samples_per_correlation_group=10.0,
        ),
    )

    assert result.inference_admissible
    assert result.state_coefficients[0] == pytest.approx(0.01, abs=3e-4)
    assert result.gauge_delta[0] == pytest.approx(0.02, abs=3e-4)
    assert result.diagnostics["active_anchor_count"] == 2


def test_shared_anchor_bias_does_not_falsely_break_state_gauge_ambiguity() -> None:
    mode = np.ones(10)
    batch = _batch(
        mode,
        np.full(10, 0.03),
        gauge_mode=mode,
        anchor_state=np.ones(4),
        anchor_innovation_x=np.full(4, 0.01),
        anchor_bias_mode=np.ones(4),
    )

    result = update_gauge_aware_belief(batch)

    assert not result.inference_admissible
    assert result.reason == "no-identifiable-query-state"
    assert result.diagnostics["anchor_bias_parameter_count"] == 1


def test_duplicate_rows_in_one_correlation_group_do_not_add_confidence() -> None:
    mode = np.linspace(-1.0, 1.0, 8)
    base = _batch(mode, 0.01 * mode)
    duplicated = _batch(
        np.tile(mode, 2),
        np.tile(0.01 * mode, 2),
        groups=tuple("window-0" for _ in range(16)),
    )
    config = GaugeAwareBeliefConfig(
        effective_samples_per_correlation_group=8.0,
    )

    base_result = update_gauge_aware_belief(base, config=config)
    duplicate_result = update_gauge_aware_belief(duplicated, config=config)

    assert base_result.inference_admissible
    assert duplicate_result.inference_admissible
    assert duplicate_result.posterior_covariance[0, 0] == pytest.approx(
        base_result.posterior_covariance[0, 0], rel=1e-10
    )
    assert duplicate_result.diagnostics[
        "effective_observation_information_mass"
    ] == pytest.approx(8.0)


def test_duplicate_anchor_rows_in_one_group_do_not_add_confidence() -> None:
    mode = np.ones(8)
    base = _batch(
        mode,
        np.full(8, 0.02),
        gauge_mode=mode,
        anchor_state=np.ones(2),
        anchor_innovation_x=np.full(2, 0.01),
        anchor_groups=("depth-frame", "depth-frame"),
    )
    duplicated = _batch(
        mode,
        np.full(8, 0.02),
        gauge_mode=mode,
        anchor_state=np.ones(4),
        anchor_innovation_x=np.full(4, 0.01),
        anchor_groups=("depth-frame",) * 4,
    )
    config = GaugeAwareBeliefConfig(
        effective_samples_per_anchor_correlation_group=2.0,
    )

    base_result = update_gauge_aware_belief(base, config=config)
    duplicate_result = update_gauge_aware_belief(duplicated, config=config)

    assert base_result.inference_admissible
    assert duplicate_result.inference_admissible
    assert duplicate_result.posterior_covariance[0, 0] == pytest.approx(
        base_result.posterior_covariance[0, 0], rel=1e-10
    )
    assert duplicate_result.diagnostics[
        "effective_anchor_information_mass"
    ] == pytest.approx(2.0)


def test_student_t_update_downweights_one_factor_outlier() -> None:
    mode = np.linspace(-1.0, 1.0, 15)
    innovation = 0.01 * mode
    innovation[-1] += 0.20
    batch = _batch(mode, innovation)

    result = update_gauge_aware_belief(
        batch,
        config=GaugeAwareBeliefConfig(
            effective_samples_per_correlation_group=15.0,
        ),
    )

    assert result.inference_admissible
    assert result.robust_weights[-1] < 0.05
    assert result.state_coefficients[0] == pytest.approx(0.01, abs=0.003)


def test_gauge_aware_final_system_rechecks_condition_number(monkeypatch) -> None:
    conditions = iter((1.0, 1e6))
    monkeypatch.setattr(np.linalg, "cond", lambda _: next(conditions))
    mode = np.linspace(-1.0, 1.0, 9)

    result = update_gauge_aware_belief(
        _batch(mode, 0.01 * mode),
        config=GaugeAwareBeliefConfig(
            maximum_iterations=1,
            maximum_condition_number=10.0,
        ),
    )

    assert not result.inference_admissible
    assert result.reason == "ill-conditioned-posterior"
    assert result.diagnostics["condition_number"] == pytest.approx(1e6)
    np.testing.assert_array_equal(result.state_coefficients, 0.0)


def test_gauge_aware_final_cholesky_failure_falls_back(monkeypatch) -> None:
    original_cholesky = np.linalg.cholesky
    call_count = 0

    def fail_final_cholesky(matrix: np.ndarray) -> np.ndarray:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise np.linalg.LinAlgError("synthetic final-system failure")
        return original_cholesky(matrix)

    monkeypatch.setattr(np.linalg, "cholesky", fail_final_cholesky)
    mode = np.linspace(-1.0, 1.0, 9)

    result = update_gauge_aware_belief(
        _batch(mode, 0.01 * mode),
        config=GaugeAwareBeliefConfig(maximum_iterations=1),
    )

    assert call_count == 2
    assert not result.inference_admissible
    assert result.reason == "singular-posterior"
    np.testing.assert_array_equal(result.state_coefficients, 0.0)


def test_gauge_aware_spd_paths_do_not_use_numpy_inverse(monkeypatch) -> None:
    def reject_inverse(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("np.linalg.inv must not be used for SPD systems")

    monkeypatch.setattr(np.linalg, "inv", reject_inverse)
    mode = np.linspace(-1.0, 1.0, 9)

    result = update_gauge_aware_belief(
        _batch(mode, 0.01 * mode),
        config=GaugeAwareBeliefConfig(maximum_iterations=1),
    )

    assert result.inference_admissible
    assert result.diagnostics["posterior_solver"] == "cholesky"
    assert result.diagnostics["final_system_uses_returned_robust_weights"] is True
    np.testing.assert_allclose(
        result.posterior_covariance,
        result.posterior_covariance.T,
        atol=0.0,
        rtol=0.0,
    )


def test_update_is_rejected_when_query_correction_exceeds_physical_response() -> None:
    mode = np.linspace(-1.0, 1.0, 12)
    batch = _batch(
        mode,
        0.08 * mode,
        physical_response_scale_m=0.01,
    )

    result = update_gauge_aware_belief(
        batch,
        config=GaugeAwareBeliefConfig(
            effective_samples_per_correlation_group=12.0,
            maximum_update_to_physical_response_ratio=2.0,
        ),
    )

    assert not result.inference_admissible
    assert result.reason == "implausible-state-update"
    assert result.diagnostics["maximum_query_state_update_m"] > 0.02
    assert result.posterior_covariance[0, 0] == pytest.approx(0.01)


def _two_mode_batch(
    state: np.ndarray,
    query: np.ndarray,
    innovation: np.ndarray,
    prior: np.ndarray,
) -> GaugeAwareObservationBatch:
    count = len(state)
    return GaugeAwareObservationBatch(
        innovation_m=innovation,
        observation_covariance_m2=np.tile(np.eye(3) * 1e-6, (count, 1, 1)),
        state_jacobian=state,
        gauge_jacobian=_empty_design(count),
        shared_bias_jacobian=_empty_design(count),
        view_bias_jacobian=_empty_design(count),
        query_state_jacobian=query,
        gauge_prior_covariance=np.zeros((0, 0)),
        correlation_group_ids=tuple("window-0" for _ in range(count)),
        prior_reliability=np.ones(count),
        physical_response_scale_m=0.05,
        state_prior_covariance_m2=prior,
    )


def test_query_irrelevant_state_mode_retains_prior_variance() -> None:
    count = 14
    first = np.linspace(-1.0, 1.0, count)
    second = np.cos(np.linspace(0.0, 2.0 * np.pi, count, endpoint=False))
    state = np.zeros((count, 3, 2), dtype=np.float64)
    state[:, 0, 0] = first
    state[:, 1, 1] = second
    query = np.zeros((count, 3, 2), dtype=np.float64)
    query[:, 0, 0] = first
    innovation = np.zeros((count, 3), dtype=np.float64)
    innovation[:, 0] = 0.012 * first
    batch = _two_mode_batch(state, query, innovation, np.eye(2) * 0.01)

    result = update_gauge_aware_belief(
        batch,
        config=GaugeAwareBeliefConfig(
            effective_samples_per_correlation_group=float(count),
        ),
    )

    assert result.inference_admissible
    assert result.identifiable_state_transform.shape == (2, 1)
    assert result.state_coefficients[0] == pytest.approx(0.012, abs=2e-4)
    assert result.state_coefficients[1] == pytest.approx(0.0, abs=1e-12)
    assert result.posterior_covariance[1, 1] == pytest.approx(0.01, rel=1e-10)
    assert result.diagnostics["unsupported_state_prior_preserved"]


def test_state_reparameterization_preserves_query_mean_and_covariance() -> None:
    count = 18
    first = np.linspace(-1.0, 1.0, count)
    second = np.sin(np.linspace(0.0, 2.0 * np.pi, count, endpoint=False))
    state = np.zeros((count, 3, 2), dtype=np.float64)
    state[:, 0, 0] = first
    state[:, 1, 1] = second
    query = state.copy()
    innovation = np.zeros((count, 3), dtype=np.float64)
    innovation[:, 0] = 0.008 * first
    innovation[:, 1] = -0.004 * second
    prior = np.asarray([[0.012, 0.002], [0.002, 0.008]])
    original = update_gauge_aware_belief(
        _two_mode_batch(state, query, innovation, prior),
        config=GaugeAwareBeliefConfig(
            effective_samples_per_correlation_group=float(count),
        ),
    )

    transform = np.asarray([[2.0, 0.3], [-0.4, 0.7]])
    inverse = np.linalg.inv(transform)
    transformed_state = np.einsum("mcs,sk->mck", state, transform)
    transformed_query = np.einsum("qcs,sk->qck", query, transform)
    transformed_prior = inverse @ prior @ inverse.T
    reparameterized = update_gauge_aware_belief(
        _two_mode_batch(
            transformed_state,
            transformed_query,
            innovation,
            transformed_prior,
        ),
        config=GaugeAwareBeliefConfig(
            effective_samples_per_correlation_group=float(count),
        ),
    )

    original_mean = decode_gauge_aware_query(original, query)
    transformed_mean = decode_gauge_aware_query(
        reparameterized, transformed_query
    )
    np.testing.assert_allclose(transformed_mean, original_mean, atol=2e-8)

    original_state_cov = original.posterior_covariance[:2, :2]
    transformed_state_cov = reparameterized.posterior_covariance[:2, :2]
    transformed_cov_in_original_coordinates = (
        transform @ transformed_state_cov @ transform.T
    )
    np.testing.assert_allclose(
        transformed_cov_in_original_coordinates,
        original_state_cov,
        atol=2e-8,
        rtol=2e-8,
    )


def test_candidate_selection_requires_regret_guard() -> None:
    mode = np.linspace(-1.0, 1.0, 12)
    result = update_gauge_aware_belief(_batch(mode, 0.01 * mode))
    baseline = np.asarray([0.0, -0.0, 1.5], dtype=np.float32)
    candidate = np.asarray([0.2, 0.1, 1.4], dtype=np.float32)

    unguarded = select_gauge_aware_candidate(baseline, candidate, result)
    rejected = select_gauge_aware_candidate(
        baseline,
        candidate,
        result,
        regret_decision=_GuardDecision(
            selected_value=baseline.copy(),
            candidate_accepted=False,
        ),
    )
    accepted = select_gauge_aware_candidate(
        baseline,
        candidate,
        result,
        regret_decision=_GuardDecision(
            selected_value=candidate.copy(),
            candidate_accepted=True,
        ),
    )

    assert result.inference_admissible
    assert not unguarded.candidate_accepted
    assert unguarded.reason == "missing-regret-guard-exact-baseline-fallback"
    assert unguarded.selected_value.tobytes() == baseline.tobytes()
    assert not rejected.candidate_accepted
    assert rejected.regret_guard_present
    assert rejected.selected_value.tobytes() == baseline.tobytes()
    assert accepted.candidate_accepted
    assert accepted.inference_admissible
    assert accepted.regret_guard_accepted
    assert accepted.selected_value.tobytes() == candidate.tobytes()


def test_regret_decision_must_be_bound_to_exact_candidate() -> None:
    mode = np.linspace(-1.0, 1.0, 12)
    result = update_gauge_aware_belief(_batch(mode, 0.01 * mode))
    baseline = np.asarray([0.0, 1.0], dtype=np.float32)
    candidate = np.asarray([2.0, 3.0], dtype=np.float32)

    with pytest.raises(ValueError, match="not bound"):
        select_gauge_aware_candidate(
            baseline,
            candidate,
            result,
            regret_decision=_GuardDecision(
                selected_value=np.asarray([2.0, 4.0], dtype=np.float32),
                candidate_accepted=True,
            ),
        )
