from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from bayesian_phystwin.causal4d_provider_v1 import (
    GuardDecisionV1,
    PhysicalBeliefV1,
    ProviderManifestV1,
    load_physical_belief,
    save_physical_belief,
    select_physical_belief,
)
from bayesian_phystwin.physical_linearization import (
    PhysicalLinearizationV1,
    evaluate_nonlinear_closure,
    validate_observation_linearization_alignment,
)
from bayesian_phystwin.prior_aware_gauge_belief import (
    PriorAwareGaugeConfigV1,
    select_guarded_gauge_candidate,
    update_prior_aware_gauge_belief,
)

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64
DIGEST_D = "d" * 64


def _belief(
    manifest_id: str,
    offset: float,
    *,
    metadata=None,
) -> PhysicalBeliefV1:
    position = np.zeros((2, 3, 3), dtype=np.float64) + offset
    return PhysicalBeliefV1(
        provider_manifest_id=manifest_id,
        endpoint_frame=4,
        particle_ids=("p0", "p1"),
        theta_names=("spring",),
        endpoint_position_m=position,
        endpoint_velocity_mps=np.zeros_like(position),
        theta=np.asarray([[0.0], [1.0]]),
        discrepancy_mean_m=np.zeros_like(position),
        discrepancy_variance_m2=np.ones_like(position) * 1e-4,
        weights=np.asarray([0.4, 0.6]),
        metadata=metadata or {},
    )


def _guard(
    *, candidate_valid: bool = True, accepted: bool = True
) -> GuardDecisionV1:
    return GuardDecisionV1(
        candidate_valid=candidate_valid,
        guard_accepted=accepted,
        reason="locked source certificate",
        certificate_id=DIGEST_A,
        development_partition_sha256=DIGEST_B,
        observation_artifact_id=DIGEST_C,
        linearization_artifact_id=DIGEST_D,
        primary_losses={"track_rmse_m": 0.01},
    )


def test_complete_belief_fallback_reuses_exact_baseline(tmp_path) -> None:
    manifest = ProviderManifestV1(provider_revision="abc123")
    baseline = _belief(manifest.manifest_id, 0.0)
    candidate = _belief(
        manifest.manifest_id,
        1.0,
        metadata={
            "observation_artifact_id": DIGEST_C,
            "linearization_artifact_id": DIGEST_D,
        },
    )
    selected, selection = select_physical_belief(
        baseline,
        candidate,
        _guard(candidate_valid=True, accepted=False),
    )
    assert selected is baseline
    assert selection.selected_belief_id == baseline.artifact_id
    assert not selection.selected_candidate

    path = tmp_path / "belief.npz"
    save_physical_belief(path, candidate)
    restored = load_physical_belief(path)
    assert restored.artifact_id == candidate.artifact_id
    np.testing.assert_array_equal(
        restored.endpoint_position_m,
        candidate.endpoint_position_m,
    )


def test_guard_cannot_accept_invalid_candidate() -> None:
    with pytest.raises(ValueError, match="requires candidate_valid"):
        _guard(candidate_valid=False, accepted=True)


@dataclass
class FakeObservation:
    artifact_id: str
    frame_ids: np.ndarray
    entity_ids: np.ndarray
    view_indices: np.ndarray
    window_indices: np.ndarray


def _linearization() -> PhysicalLinearizationV1:
    frame = np.asarray([1, 1, 2])
    entity = np.asarray([0, 1, 0])
    view = np.asarray([0, 0, 0])
    window = np.asarray([0, 0, 1])
    state = np.zeros((3, 3, 1))
    state[:, 0, 0] = [-1.0, 0.0, 1.0]
    query = state.copy()
    return PhysicalLinearizationV1(
        observation_artifact_id=DIGEST_C,
        baseline_belief_id=DIGEST_A,
        action_prefix_id=DIGEST_B,
        simulator_revision="sim-123",
        frame_ids=frame,
        entity_ids=entity,
        view_indices=view,
        window_indices=window,
        state_jacobian=state,
        query_state_jacobian=query,
        physical_response_m=np.asarray([[0.01, 0.0, 0.0]] * 3),
    )


def test_linearization_rejects_row_permutation() -> None:
    linearization = _linearization()
    observation = FakeObservation(
        artifact_id=DIGEST_C,
        frame_ids=linearization.frame_ids[[1, 0, 2]],
        entity_ids=linearization.entity_ids[[1, 0, 2]],
        view_indices=linearization.view_indices[[1, 0, 2]],
        window_indices=linearization.window_indices[[1, 0, 2]],
    )
    with pytest.raises(ValueError, match="differ"):
        validate_observation_linearization_alignment(
            observation,
            linearization,
        )


def test_nonlinear_closure_reports_failure() -> None:
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
class Batch:
    innovation_m: np.ndarray
    observation_covariance_m2: np.ndarray
    state_jacobian: np.ndarray
    gauge_jacobian: np.ndarray
    shared_bias_jacobian: np.ndarray
    view_bias_jacobian: np.ndarray
    query_state_jacobian: np.ndarray
    gauge_prior_covariance: np.ndarray
    correlation_group_ids: tuple[str, ...]
    prior_reliability: np.ndarray
    prior_nominal_probability: np.ndarray
    composite_weight: np.ndarray
    physical_response_scale_m: float
    state_prior_covariance_m2: np.ndarray | None = None
    anchor_innovation_m: np.ndarray | None = None
    anchor_covariance_m2: np.ndarray | None = None
    anchor_state_jacobian: np.ndarray | None = None


def _confounded_batch(gauge_variance: float) -> Batch:
    count = 12
    mode = np.ones(count)
    innovation = np.zeros((count, 3))
    innovation[:, 0] = 0.01
    state = np.zeros((count, 3, 1))
    state[:, 0, 0] = mode
    gauge = state.copy()
    empty = np.zeros((count, 3, 0))
    return Batch(
        innovation_m=innovation,
        observation_covariance_m2=np.tile(
            np.eye(3) * 1e-6,
            (count, 1, 1),
        ),
        state_jacobian=state,
        gauge_jacobian=gauge,
        shared_bias_jacobian=empty,
        view_bias_jacobian=empty,
        query_state_jacobian=state,
        gauge_prior_covariance=np.asarray([[gauge_variance]]),
        correlation_group_ids=tuple("g0" for _ in range(count)),
        prior_reliability=np.ones(count),
        prior_nominal_probability=np.full(count, 0.95),
        composite_weight=np.ones(count),
        physical_response_scale_m=0.05,
        state_prior_covariance_m2=np.asarray([[0.01]]),
    )


def test_prior_aware_mode_uses_tight_gauge_prior_continuously() -> None:
    tight = update_prior_aware_gauge_belief(
        _confounded_batch(1e-10),
        config=PriorAwareGaugeConfigV1(
            effective_samples_per_correlation_group=12,
            minimum_identifiable_fraction=0.01,
        ),
    )
    diffuse = update_prior_aware_gauge_belief(
        _confounded_batch(1e6),
        config=PriorAwareGaugeConfigV1(
            effective_samples_per_correlation_group=12,
            minimum_identifiable_fraction=0.10,
        ),
    )
    assert tight.candidate_valid
    assert tight.state_coefficients[0] == pytest.approx(0.01, abs=5e-4)
    assert (
        not diffuse.candidate_valid
        or abs(diffuse.state_coefficients[0]) < 0.005
    )


def test_guarded_array_selection_requires_guard() -> None:
    result = update_prior_aware_gauge_belief(
        _confounded_batch(1e-10),
        config=PriorAwareGaugeConfigV1(
            effective_samples_per_correlation_group=12,
            minimum_identifiable_fraction=0.01,
        ),
    )
    baseline = np.asarray([0.0, -0.0, 2.0], dtype=np.float32)
    selection = select_guarded_gauge_candidate(
        baseline,
        np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        result,
        _guard(
            candidate_valid=result.candidate_valid,
            accepted=False,
        ),
    )
    assert not selection.selected_candidate
    assert selection.selected_value.dtype == baseline.dtype
    assert selection.selected_value.tobytes() == baseline.tobytes()
