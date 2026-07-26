from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from bayesian_phystwin.bias_aware_belief import GuardedUpdateDecision
from bayesian_phystwin.guarded_belief_pipeline import (
    GuardedBeliefPipelineConfigV1,
    ProspectiveSupportDecisionV1,
    run_prior_aware_guarded_belief_update,
)
from bayesian_phystwin.observation_belief import ObservationBeliefV1
from bayesian_phystwin.physical_linearization import PhysicalLinearizationV1
from bayesian_phystwin.prior_aware_gauge_belief import PriorAwareGaugeConfigV1


@dataclass
class Belief:
    artifact_id: str
    payload: np.ndarray


def _observation_and_linearization(
    *,
    global_translation_mode: bool = False,
) -> tuple[ObservationBeliefV1, PhysicalLinearizationV1, np.ndarray]:
    count = 12
    state = np.zeros((count, 3, 1))
    if global_translation_mode:
        state[:, 0, 0] = 1.0
    else:
        state[:, 0, 0] = np.linspace(-1.0, 1.0, count)
    prediction = np.zeros((count, 3))
    mean = prediction + state[:, :, 0] * 0.01
    observation = ObservationBeliefV1(
        case_id="synthetic-source-control",
        stream_id="synthetic:causal-prefix",
        causal_frame_stop=2,
        view_names=("view-0",),
        window_names=("window-0",),
        factor_names=(),
        source_repository="synthetic",
        source_revision="source-v1",
        source_artifact_sha256="a" * 64,
        declared_frame_ids=np.asarray([1]),
        mean_xyz_m=mean,
        frame_ids=np.ones(count, dtype=np.int64),
        entity_ids=np.arange(count, dtype=np.int64),
        view_indices=np.zeros(count, dtype=np.int64),
        window_indices=np.zeros(count, dtype=np.int64),
        correlation_group_ids=np.arange(count, dtype=np.int64),
        factor_group_ids=np.zeros(count, dtype=np.int64),
        prior_reliability=np.ones(count),
        association_probability=np.ones(count),
        local_covariance_m2=np.tile(np.eye(3) * 1e-6, (count, 1, 1)),
        low_rank_factor_m=np.zeros((count, 3, 0)),
        group_ids=np.arange(count, dtype=np.int64),
        group_prior_nominal_probability=np.full(count, 0.99),
        group_composite_weight=np.ones(count),
    )
    query = state[[0, count // 2, count - 1]].copy()
    linearization = PhysicalLinearizationV1(
        observation_artifact_id=observation.artifact_id,
        baseline_belief_id="b" * 64,
        action_prefix_id="c" * 64,
        simulator_revision="synthetic-simulator-v1",
        frame_ids=observation.frame_ids,
        entity_ids=observation.entity_ids,
        view_indices=observation.view_indices,
        window_indices=observation.window_indices,
        state_jacobian=state,
        query_state_jacobian=query,
        physical_response_m=query[:, :, 0] * 0.02,
    )
    return observation, linearization, prediction


def _support(
    *,
    structural: bool = True,
    physical: bool = True,
) -> ProspectiveSupportDecisionV1:
    return ProspectiveSupportDecisionV1(
        structural_evidence_id="d" * 64,
        physical_evidence_id="e" * 64,
        structural_support_accepted=structural,
        physical_support_accepted=physical,
        structural_support_kind="pairwise-consensus-with-redundancy",
        physical_support_kind="action-conditioned-prefix-response",
    )


def _run(
    *,
    support: ProspectiveSupportDecisionV1 | None = None,
    nonlinear_offset_m: float = 0.0,
    regret_accepted: bool = True,
    regret_selected_value: np.ndarray | None = None,
    global_translation_mode: bool = False,
    inference_config: PriorAwareGaugeConfigV1 | None = None,
):
    observation, linearization, prediction = _observation_and_linearization(
        global_translation_mode=global_translation_mode
    )
    baseline = Belief("b" * 64, np.asarray([0.0], dtype=np.float32))
    candidate = Belief("f" * 64, np.asarray([1.0], dtype=np.float32))
    baseline_query = np.zeros(linearization.query_state_jacobian.shape[:2])
    expected_coefficient = 0.01
    nonlinear_query = baseline_query + np.einsum(
        "qcs,s->qc",
        linearization.query_state_jacobian,
        np.asarray([expected_coefficient]),
    )
    nonlinear_query = nonlinear_query.copy()
    nonlinear_query[0, 0] += nonlinear_offset_m
    selected_query = (
        np.asarray(regret_selected_value)
        if regret_selected_value is not None
        else nonlinear_query
        if regret_accepted
        else baseline_query
    )
    regret = GuardedUpdateDecision(
        selected_value=selected_query,
        candidate_accepted=regret_accepted,
        predicted_regret=-0.001 if regret_accepted else 0.001,
        upper_regret=-0.0005 if regret_accepted else 0.002,
        reason="synthetic-source-certificate",
    )
    selected, outcome = run_prior_aware_guarded_belief_update(
        observation,
        linearization,
        baseline_belief=baseline,
        candidate_belief=candidate,
        physical_prediction_xyz_m=prediction,
        baseline_query_m=baseline_query,
        nonlinear_candidate_query_m=nonlinear_query,
        support_decision=support or _support(),
        regret_decision=regret,
        source_certificate_id="1" * 64,
        common_domain_id="2" * 64,
        inference_config=inference_config
        or PriorAwareGaugeConfigV1(
            minimum_identifiable_fraction=0.05,
            minimum_conditional_information_fraction=1e-4,
        ),
        pipeline_config=GuardedBeliefPipelineConfigV1(
            closure_absolute_tolerance_m=0.002,
            closure_relative_tolerance=0.25,
        ),
        state_prior_covariance_m2=np.asarray([[0.02**2]]),
    )
    return baseline, candidate, selected, outcome


def test_action_supported_identifiable_update_selects_complete_candidate() -> None:
    baseline, candidate, selected, outcome = _run()

    assert outcome.numerical_result.inference_admissible
    assert outcome.support_decision.accepted
    assert outcome.nonlinear_closure.candidate_valid
    assert outcome.selected_candidate
    assert selected is candidate
    assert selected is not baseline
    assert outcome.complete_selection.selected_belief_id == candidate.artifact_id
    assert (
        outcome.complete_decision.metadata["nonlinear_query_sha256"]
        == outcome.nonlinear_query_sha256
    )


@pytest.mark.parametrize(
    ("support", "regret_accepted"),
    [
        (_support(structural=False), True),
        (_support(physical=False), True),
        (_support(), False),
    ],
)
def test_rejected_gate_returns_same_baseline_object(
    support: ProspectiveSupportDecisionV1,
    regret_accepted: bool,
) -> None:
    baseline, _, selected, outcome = _run(
        support=support,
        regret_accepted=regret_accepted,
    )

    assert selected is baseline
    assert not outcome.selected_candidate
    assert outcome.complete_selection.selected_belief_id == baseline.artifact_id


def test_nonlinear_closure_rejects_large_local_model_remainder() -> None:
    baseline, _, selected, outcome = _run(nonlinear_offset_m=0.05)

    assert outcome.numerical_result.inference_admissible
    assert not outcome.nonlinear_closure.candidate_valid
    assert selected is baseline
    assert not outcome.selected_candidate


def test_diffuse_common_bias_prior_blocks_camera_translation_as_state() -> None:
    baseline, _, selected, outcome = _run(
        global_translation_mode=True,
        inference_config=PriorAwareGaugeConfigV1(
            shared_bias_prior_std_m=1e6,
            minimum_identifiable_fraction=0.50,
            minimum_conditional_information_fraction=0.50,
        ),
    )

    assert not outcome.numerical_result.inference_admissible
    assert selected is baseline
    assert not outcome.selected_candidate


def test_support_cannot_be_declared_as_prior_reliability() -> None:
    with pytest.raises(ValueError, match="separate from prior reliability"):
        ProspectiveSupportDecisionV1(
            structural_evidence_id="d" * 64,
            physical_evidence_id="e" * 64,
            structural_support_accepted=True,
            physical_support_accepted=True,
            structural_support_kind="pairwise",
            physical_support_kind="physical",
            used_as_prior_reliability=True,
        )


def test_support_cannot_read_future_target() -> None:
    with pytest.raises(ValueError, match="must not read a future target"):
        ProspectiveSupportDecisionV1(
            structural_evidence_id="d" * 64,
            physical_evidence_id="e" * 64,
            structural_support_accepted=True,
            physical_support_accepted=True,
            structural_support_kind="pairwise",
            physical_support_kind="physical",
            future_target_read=True,
        )


def test_support_cannot_reuse_candidate_innovation() -> None:
    with pytest.raises(ValueError, match="candidate innovation again"):
        ProspectiveSupportDecisionV1(
            structural_evidence_id="d" * 64,
            physical_evidence_id="e" * 64,
            structural_support_accepted=True,
            physical_support_accepted=True,
            structural_support_kind="pairwise",
            physical_support_kind="physical",
            candidate_innovation_reused=True,
        )


def test_regret_decision_must_bind_exact_candidate_query() -> None:
    with pytest.raises(
        ValueError,
        match="regret decision is not bound",
    ):
        _run(
            regret_accepted=True,
            regret_selected_value=np.full((3, 3), 0.123),
        )
