from dataclasses import dataclass

import numpy as np
import pytest

from bayesian_phystwin._gauge_aware_contracts import GaugeAwareObservationBatch
from bayesian_phystwin.complete_belief_selection import (
    CompleteBeliefGuardDecisionV1,
    select_complete_belief,
)
from bayesian_phystwin.physical_linearization import (
    PhysicalLinearizationV1,
    evaluate_nonlinear_closure,
    validate_observation_linearization_alignment,
)
from bayesian_phystwin.prior_aware_gauge_belief import (
    PriorAwareGaugeConfigV1,
    update_prior_aware_gauge_belief,
)
from bayesian_phystwin.propagated_state_belief import (
    PropagatedStateBeliefConfig,
    infer_propagated_state_belief,
)

A = "a" * 64
B = "b" * 64
C = "c" * 64


def _empty(count: int) -> np.ndarray:
    return np.zeros((count, 3, 0), dtype=float)


def _confounded_batch(
    gauge_variance: float, *, outlier: bool = False
) -> GaugeAwareObservationBatch:
    count = 12
    state = np.zeros((count, 3, 1))
    state[:, 0, 0] = 1.0
    innovation = np.zeros((count, 3))
    innovation[:, 0] = 0.01
    if outlier:
        innovation[-4:, 0] = 0.25
    groups = tuple("g0" if index < 8 else "g1" for index in range(count))
    return GaugeAwareObservationBatch(
        innovation_m=innovation,
        observation_covariance_m2=np.tile(np.eye(3) * 1e-6, (count, 1, 1)),
        state_jacobian=state,
        gauge_jacobian=state.copy(),
        shared_bias_jacobian=_empty(count),
        view_bias_jacobian=_empty(count),
        query_state_jacobian=state.copy(),
        gauge_prior_covariance=np.asarray([[gauge_variance]]),
        correlation_group_ids=groups,
        prior_reliability=np.ones(count),
        prior_nominal_probability=np.asarray([0.95] * 8 + [0.80] * 4),
        composite_weight=np.ones(count),
        physical_response_scale_m=0.05,
        state_prior_covariance_m2=np.asarray([[0.01]]),
        metadata={"observation_artifact_id": C},
    )


def test_tight_gauge_prior_approaches_known_gauge_solution() -> None:
    result = update_prior_aware_gauge_belief(
        _confounded_batch(1e-10),
        config=PriorAwareGaugeConfigV1(
            effective_samples_per_correlation_group=12,
            minimum_identifiable_fraction=0.01,
        ),
    )
    assert result.inference_admissible
    assert result.state_coefficients[0] == pytest.approx(0.01, abs=6e-4)
    assert result.diagnostics["identifiability_mode"] == "prior-aware-schur-v1"


def test_diffuse_gauge_prior_falls_back_or_suppresses_state() -> None:
    result = update_prior_aware_gauge_belief(
        _confounded_batch(1e6),
        config=PriorAwareGaugeConfigV1(
            effective_samples_per_correlation_group=12,
            minimum_identifiable_fraction=0.10,
        ),
    )
    assert not result.inference_admissible or abs(result.state_coefficients[0]) < 0.005


def test_group_mixture_downweights_corrupted_group() -> None:
    result = update_prior_aware_gauge_belief(
        _confounded_batch(1e-10, outlier=True),
        config=PriorAwareGaugeConfigV1(
            effective_samples_per_correlation_group=12,
            minimum_identifiable_fraction=0.01,
        ),
    )
    posterior = result.diagnostics["observation_group_posterior_nominal_probability"]
    assert posterior[1] < posterior[0]
    assert result.diagnostics["prior_nominal_probability_used_inside_mixture"]


@dataclass
class Observation:
    artifact_id: str
    frame_ids: np.ndarray
    entity_ids: np.ndarray
    view_indices: np.ndarray
    window_indices: np.ndarray


def _linearization() -> PhysicalLinearizationV1:
    state = np.zeros((3, 3, 1))
    state[:, 0, 0] = [-1.0, 0.0, 1.0]
    return PhysicalLinearizationV1(
        observation_artifact_id=A,
        baseline_belief_id=B,
        action_prefix_id=C,
        simulator_revision="sim-1",
        frame_ids=np.asarray([1, 1, 2]),
        entity_ids=np.asarray([0, 1, 0]),
        view_indices=np.asarray([0, 0, 0]),
        window_indices=np.asarray([0, 0, 1]),
        state_jacobian=state,
        query_state_jacobian=state.copy(),
        physical_response_m=np.asarray([[0.01, 0.0, 0.0]] * 3),
    )


def test_linearization_rejects_row_permutation() -> None:
    linearization = _linearization()
    permutation = np.asarray([1, 0, 2])
    observation = Observation(
        artifact_id=A,
        frame_ids=linearization.frame_ids[permutation],
        entity_ids=linearization.entity_ids[permutation],
        view_indices=linearization.view_indices[permutation],
        window_indices=linearization.window_indices[permutation],
    )
    with pytest.raises(ValueError, match="differ"):
        validate_observation_linearization_alignment(observation, linearization)


def test_response_scale_is_bound_to_linearization() -> None:
    assert _linearization().physical_response_scale_m == pytest.approx(0.01)


def test_nonlinear_closure_fails_large_remainder() -> None:
    linearization = _linearization()
    baseline = np.zeros((3, 3))
    linear = np.ones((3, 3)) * 0.01
    nonlinear = linear.copy()
    nonlinear[0, 0] += 0.1
    closure = evaluate_nonlinear_closure(
        linearization.artifact_id,
        baseline_query_m=baseline,
        linearized_query_m=linear,
        nonlinear_query_m=nonlinear,
        absolute_tolerance_m=0.02,
        relative_tolerance=0.2,
    )
    assert not closure.candidate_valid


@dataclass
class Belief:
    artifact_id: str
    payload: np.ndarray


def test_complete_belief_fallback_reuses_exact_baseline_object() -> None:
    baseline = Belief("d" * 64, np.asarray([0.0, -0.0], dtype=np.float32))
    candidate = Belief("e" * 64, np.asarray([1.0, 1.0], dtype=np.float32))
    decision = CompleteBeliefGuardDecisionV1(
        baseline_belief_id=baseline.artifact_id,
        candidate_belief_id=candidate.artifact_id,
        common_domain_id="f" * 64,
        certificate_id="9" * 64,
        inference_admissible=True,
        regret_guard_accepted=False,
        reason="source certificate rejected",
    )
    selected, manifest = select_complete_belief(baseline, candidate, decision)
    assert selected is baseline
    assert manifest.selected_belief_id == baseline.artifact_id
    assert not manifest.selected_candidate


def _one_step_propagated_state_problem() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    innovation = np.zeros((1, 3, 3), dtype=np.float64)
    innovation[0, :, 0] = np.asarray([1.0, 1.0, 10.0])
    available = np.ones((1, 3), dtype=bool)
    response = np.zeros((1, 3, 3, 1), dtype=np.float64)
    response[0, :, 0, 0] = 1.0
    bias_basis = np.zeros((3, 0), dtype=np.float64)
    return innovation, available, response, bias_basis


def test_propagated_state_final_system_uses_returned_robust_weights() -> None:
    innovation, available, response, bias_basis = _one_step_propagated_state_problem()
    config = PropagatedStateBeliefConfig(
        observation_std_m=1.0,
        state_weight_prior_std=10.0,
        effective_samples_per_frame=3.0,
        effective_frame_count=1.0,
        maximum_iterations=1,
        reject_unidentifiable_state=False,
    )

    result = infer_propagated_state_belief(
        innovation,
        available,
        response,
        bias_basis,
        observation_variance_m2=np.ones(available.shape),
        config=config,
    )

    assert result.accepted
    robust = result.robust_weights[0]
    expected_precision = 1.0 / config.state_weight_prior_std**2 + np.sum(robust)
    expected_right = float(robust @ innovation[0, :, 0])
    assert result.state_weights[0] == pytest.approx(
        expected_right / expected_precision,
        rel=1e-12,
        abs=1e-12,
    )
    assert result.posterior_covariance[0, 0] == pytest.approx(
        1.0 / expected_precision,
        rel=1e-12,
        abs=1e-12,
    )
    assert result.diagnostics["final_system_uses_returned_robust_weights"] is True
    assert result.diagnostics["posterior_solver"] == "cholesky"


def test_propagated_state_spd_paths_do_not_use_numpy_inverse(monkeypatch) -> None:
    innovation, available, response, bias_basis = _one_step_propagated_state_problem()

    def fail_inverse(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("np.linalg.inv must not be used for SPD systems")

    monkeypatch.setattr(np.linalg, "inv", fail_inverse)
    result = infer_propagated_state_belief(
        innovation,
        available,
        response,
        bias_basis,
        observation_variance_m2=np.ones(available.shape),
        state_prior_covariance=np.asarray([[4.0]]),
        config=PropagatedStateBeliefConfig(
            maximum_iterations=2,
            reject_unidentifiable_state=False,
        ),
    )

    assert result.accepted
    np.testing.assert_allclose(
        result.posterior_covariance,
        result.posterior_covariance.T,
        atol=0.0,
        rtol=0.0,
    )


def test_propagated_state_rejects_non_positive_definite_prior() -> None:
    innovation, available, response, bias_basis = _one_step_propagated_state_problem()

    with pytest.raises(ValueError, match="positive definite"):
        infer_propagated_state_belief(
            innovation,
            available,
            response,
            bias_basis,
            observation_variance_m2=np.ones(available.shape),
            state_prior_covariance=np.asarray([[0.0]]),
            config=PropagatedStateBeliefConfig(
                maximum_iterations=1,
                reject_unidentifiable_state=False,
            ),
        )


def test_propagated_state_ill_conditioned_posterior_falls_back() -> None:
    innovation, available, response, bias_basis = _one_step_propagated_state_problem()

    result = infer_propagated_state_belief(
        innovation,
        available,
        response,
        bias_basis,
        observation_variance_m2=np.ones(available.shape),
        config=PropagatedStateBeliefConfig(
            maximum_iterations=1,
            maximum_condition_number=0.5,
            reject_unidentifiable_state=False,
        ),
    )

    assert not result.accepted
    assert result.reason == "ill-conditioned-posterior"


def test_propagated_state_final_cholesky_failure_falls_back(monkeypatch) -> None:
    innovation, available, response, bias_basis = _one_step_propagated_state_problem()
    original_cholesky = np.linalg.cholesky
    call_count = 0

    def fail_second_cholesky(matrix: np.ndarray) -> np.ndarray:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise np.linalg.LinAlgError("forced final-system failure")
        return original_cholesky(matrix)

    monkeypatch.setattr(np.linalg, "cholesky", fail_second_cholesky)
    result = infer_propagated_state_belief(
        innovation,
        available,
        response,
        bias_basis,
        observation_variance_m2=np.ones(available.shape),
        config=PropagatedStateBeliefConfig(
            maximum_iterations=1,
            reject_unidentifiable_state=False,
        ),
    )

    assert call_count == 2
    assert not result.accepted
    assert result.reason == "singular-posterior"
