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
    posterior = result.diagnostics[
        "observation_group_posterior_nominal_probability"
    ]
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
    selected, manifest = select_complete_belief(
        baseline, candidate, decision
    )
    assert selected is baseline
    assert manifest.selected_belief_id == baseline.artifact_id
    assert not manifest.selected_candidate
