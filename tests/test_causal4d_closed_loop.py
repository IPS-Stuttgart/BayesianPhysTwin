from dataclasses import replace

import numpy as np

from causal4d.closed_loop import (
    CandidatePlan,
    PlanningConstraints,
    RecedingHorizonPlanner,
    condition_plan_on_recursive_belief,
    graph_discrepancy_adjusted_plan_moments,
    run_receding_horizon,
)
from causal4d.contracts import PhysicalPosterior, build_causal_context
from causal4d.graph_temporal_discrepancy import GraphTemporalDiscrepancyModel
from causal4d.physical_validation import physical_posterior_moments
from causal4d.semantic_posterior import SparseSemanticEvidence, build_task_posterior


def _physical(
    action_id: str,
    direction: float,
    *,
    initial_x: float = 0.0,
    node_count: int = 1,
) -> PhysicalPosterior:
    observations = np.zeros((8, 1, 3), dtype=float)
    actions = np.zeros((8, 1, 3), dtype=float)
    context = build_causal_context(
        protocol_id="closed_loop_unit",
        case_id="synthetic",
        observations=observations,
        observed_actions=actions,
        counterfactual_actions=actions,
        intervention_frame=3,
        counterfactual_action_id=action_id,
    )
    trajectories = np.zeros((2, 5, node_count, 3), dtype=float)
    trajectories[0, :, :, 0] = initial_x + direction * np.arange(5)[:, None] * 0.01
    trajectories[1, :, :, 0] = initial_x + direction * np.arange(5)[:, None] * 0.02
    return PhysicalPosterior(
        context=context,
        component_ids=(f"{action_id}:p0", f"{action_id}:p1"),
        state_trajectories_m=trajectories,
        readout_trajectories_m=trajectories,
        readout_variance_m2=np.full((2, node_count, 3), 1e-6),
        weights=np.asarray([0.5, 0.5]),
        phi=np.asarray([[0.9], [1.1]]),
        kappa_cf=np.asarray([[0.0], [1.0]]),
        hypothesis_indices=np.asarray([0, 1]),
        twin_particle_indices=np.asarray([0, 1]),
        phi_names=("gain",),
        kappa_names=("contact",),
        source_twin_belief_id="1" * 64,
        source_factual_intervention_id="2" * 64,
        source_query_id="3" * 64,
    )


def _plan(
    action_id: str,
    direction: float,
    *,
    unsafe=False,
    initial_x: float = 0.0,
    node_count: int = 1,
) -> CandidatePlan:
    physical = _physical(
        action_id,
        direction,
        initial_x=initial_x,
        node_count=node_count,
    )
    target = np.zeros((4, 1, 3), dtype=float)
    target[:, 0, 0] = initial_x + np.arange(1, 5) * 0.02
    evidence = SparseSemanticEvidence(
        positions_m=target,
        node_indices=np.asarray([0]),
        physical_frame_indices=np.arange(1, 5, dtype=float),
        scale_m=0.003,
        compare_displacements=True,
        anchor_positions_m=np.asarray([[initial_x, 0.0, 0.0]]),
        source="language:right",
    )
    task = build_task_posterior(physical, evidence, beta=20.0)
    controls = np.zeros((4, 1, 3), dtype=float)
    controls[:, 0, 0] = np.arange(1, 5) * (0.5 if unsafe else 0.01) * direction
    return CandidatePlan(
        action_id=action_id,
        controller_points_m=controls,
        control_anchor_m=np.zeros((1, 3)),
        physical=physical,
        task=task,
    )


def _graph_model(
    *,
    node_count: int = 2,
    transition: float = 0.8,
    innovation_variance: float = 4e-4,
) -> GraphTemporalDiscrepancyModel:
    basis = np.ones((node_count, 1), dtype=float) / np.sqrt(node_count)
    return GraphTemporalDiscrepancyModel(
        basis=basis,
        eigenvalues=np.asarray([0.0]),
        transition=np.asarray([[transition]]),
        innovation_covariance=np.asarray([[innovation_variance]]),
        projection_variance_m2=np.full(3, 1e-8),
        selected_rank=1,
        candidate_validation_rmse_m=((1, 0.0),),
        spectral_radius_before_clipping=transition,
        spectral_radius=transition,
        fit_frame_count=10,
        projection_ridge=1e-5,
        dynamics_ridge=1e-4,
    )


def test_closed_loop_replans_updates_latents_and_respects_constraints() -> None:
    planner = RecedingHorizonPlanner(
        PlanningConstraints(
            maximum_control_step_m=0.05,
            maximum_state_displacement_m=0.20,
            maximum_predictive_std_m=0.10,
            effort_weight=0.01,
            risk_weight=0.0,
        ),
        observation_scale_m=0.001,
        observation_likelihood_power=80.0,
    )
    provider_calls = []

    def plans(frame, belief):
        provider_calls.append((frame, belief))
        if frame == 0:
            assert belief is None
            return (
                _plan("gentle_right", 1.0),
                _plan("left", -1.0),
                _plan("unsafe_right", 1.0, unsafe=True),
            )
        assert belief is not None
        assert belief.twin_particle_marginal[1] > 0.95
        endpoint = float(
            np.einsum(
                "k,k->",
                belief.component_weights,
                belief.component_state_position_m[:, 0, 0],
            )
        )
        assert endpoint > 0.03
        return (
            _plan("finish_right", 1.0, initial_x=endpoint),
            _plan("finish_left", -1.0, initial_x=endpoint),
        )

    def observe(plan, frame_count):
        # The world follows particle 1, providing new physical evidence after
        # every short executed segment.
        values = plan.physical.readout_trajectories_m[1, 1 : frame_count + 1]
        return values, np.ones(values.shape[:2], dtype=bool)

    result = run_receding_horizon(
        planner,
        plans,
        observe,
        initial_frame=0,
        stop_frame=4,
        replan_interval_frames=2,
        task_success=lambda steps, belief: (
            steps[-1].selected_action_id == "finish_right"
            and belief.twin_particle_marginal[1] > 0.95
        ),
    )
    assert len(provider_calls) == 2
    assert [step.selected_action_id for step in result.steps] == [
        "gentle_right",
        "finish_right",
    ]
    unsafe = next(
        assessment
        for assessment in result.steps[0].assessments
        if assessment.action_id == "unsafe_right"
    )
    assert not unsafe.feasible
    assert unsafe.violations == ("control_step_limit",)
    assert result.language_task_success
    assert result.all_selected_plans_feasible
    assert result.final_belief.twin_particle_marginal[1] > 0.95
    assert np.all(result.final_belief.component_state_velocity_mps[:, 0, 0] > 0.0)
    assert result.steps[0].stop_frame == result.steps[1].start_frame


def test_closed_loop_observation_update_starts_from_physics_not_task_weights() -> None:
    plan = _plan("right", 1.0)
    assert plan.task is not None and plan.task.task_weights[1] > 0.99
    planner = RecedingHorizonPlanner(
        PlanningConstraints(0.1, 0.2, 0.1),
        observation_scale_m=0.001,
        observation_likelihood_power=80.0,
    )
    observations = plan.physical.readout_trajectories_m[0, 1:3]
    belief = planner.assimilate(
        plan,
        observations,
        observation_frame_stop=2,
    )
    assert belief.twin_particle_marginal[0] > 0.95
    assert belief.twin_particle_marginal[1] < 0.05
    assert belief.graph_discrepancy_coefficient_mean is None
    assert belief.graph_discrepancy_coefficient_covariance is None


def test_graph_discrepancy_partial_mask_contracts_only_observed_coordinate() -> None:
    plan = _plan("right", 1.0, node_count=2)
    model = _graph_model(transition=0.8, innovation_variance=4e-4)
    planner = RecedingHorizonPlanner(
        PlanningConstraints(0.1, 0.2, 0.1),
        observation_scale_m=0.001,
        graph_discrepancy_model=model,
        graph_discrepancy_dynamics="learned",
    )
    observations = plan.physical.readout_trajectories_m[0, 1:3].astype(float)
    observations[0, 0, 0] += 0.02
    mask = np.zeros_like(observations, dtype=bool)
    mask[0, 0, 0] = True

    belief = planner.assimilate(
        plan,
        observations,
        observation_frame_stop=2,
        mask=mask,
    )

    assert belief.graph_discrepancy_coefficient_mean is not None
    assert belief.graph_discrepancy_coefficient_covariance is not None
    covariance = belief.graph_discrepancy_coefficient_covariance
    propagation_only = 4e-4 * (1.0 + 0.8**2)
    np.testing.assert_allclose(covariance[:, 1:, 0, 0], propagation_only)
    assert np.all(covariance[:, 0, 0, 0] < propagation_only)
    assert belief.graph_discrepancy_coefficient_mean[0, 0, 0] > 0.0
    np.testing.assert_array_equal(
        belief.component_state_position_m,
        plan.physical.state_trajectories_m[:, 2],
    )


def test_graph_discrepancy_defaults_to_persistence_with_learned_opt_in() -> None:
    plan = _plan("right", 1.0, node_count=2)
    coefficient_mean = np.zeros((2, 1, 3), dtype=float)
    coefficient_mean[:, 0, 0] = 0.02
    plan = replace(
        plan,
        graph_discrepancy_coefficient_mean=coefficient_mean,
        graph_discrepancy_coefficient_covariance=np.zeros((2, 3, 1, 1)),
    )
    model = _graph_model(transition=0.5)
    observations = plan.physical.readout_trajectories_m[0, 1:2].astype(float)
    mask = np.zeros_like(observations, dtype=bool)
    mask[0, 0, 1] = True
    settings = {
        "constraints": PlanningConstraints(0.1, 0.2, 0.1),
        "observation_scale_m": 0.001,
        "graph_discrepancy_model": model,
    }

    persistent = RecedingHorizonPlanner(**settings).assimilate(
        plan,
        observations,
        observation_frame_stop=1,
        mask=mask,
    )
    learned = RecedingHorizonPlanner(
        **settings,
        graph_discrepancy_dynamics="learned",
    ).assimilate(
        plan,
        observations,
        observation_frame_stop=1,
        mask=mask,
    )

    np.testing.assert_allclose(
        persistent.graph_discrepancy_coefficient_mean[:, 0, 0],
        0.02,
    )
    np.testing.assert_allclose(
        learned.graph_discrepancy_coefficient_mean[:, 0, 0],
        0.01,
    )
    np.testing.assert_allclose(
        persistent.graph_discrepancy_coefficient_covariance[:, 0],
        np.broadcast_to(model.innovation_covariance, (2, 1, 1)),
    )
    np.testing.assert_allclose(
        learned.graph_discrepancy_coefficient_covariance[:, 0],
        np.broadcast_to(model.innovation_covariance, (2, 1, 1)),
    )


def test_graph_discrepancy_changes_readout_moments_not_physical_state() -> None:
    plan = _plan("right", 1.0, node_count=2)
    physical_artifact_id = plan.physical.artifact_id
    physical_state = plan.physical.state_trajectories_m.copy()
    coefficient_mean = np.zeros((2, 1, 3), dtype=float)
    coefficient_mean[:, 0, 0] = 0.01
    coefficient_covariance = np.full((2, 3, 1, 1), 1e-4)
    adjusted_plan = replace(
        plan,
        graph_discrepancy_coefficient_mean=coefficient_mean,
        graph_discrepancy_coefficient_covariance=coefficient_covariance,
    )
    model = _graph_model()
    baseline_mean, baseline_variance = physical_posterior_moments(plan.physical)
    adjusted_mean, adjusted_variance = graph_discrepancy_adjusted_plan_moments(
        adjusted_plan,
        model,
    )
    _, variance_with_projection = graph_discrepancy_adjusted_plan_moments(
        adjusted_plan,
        model,
        projection_variance_mode="add",
    )
    planner = RecedingHorizonPlanner(
        PlanningConstraints(0.1, 0.2, 0.1),
        graph_discrepancy_model=model,
    )

    baseline_assessment = planner.assess(plan)
    adjusted_assessment = planner.assess(adjusted_plan)

    assert np.max(np.abs(adjusted_mean - baseline_mean)) > 0.0
    assert np.all(adjusted_variance > baseline_variance)
    assert adjusted_assessment.predictive_risk > baseline_assessment.predictive_risk
    np.testing.assert_allclose(
        variance_with_projection - adjusted_variance,
        np.broadcast_to(model.projection_variance_m2, adjusted_variance.shape),
    )
    assert adjusted_plan.physical.artifact_id == physical_artifact_id
    np.testing.assert_array_equal(
        adjusted_plan.physical.state_trajectories_m,
        physical_state,
    )


def test_graph_discrepancy_preserves_physical_component_weight_update() -> None:
    plan = _plan("right", 1.0, node_count=2)
    planner = RecedingHorizonPlanner(
        PlanningConstraints(0.1, 0.2, 0.1),
        observation_scale_m=0.001,
        observation_likelihood_power=80.0,
        graph_discrepancy_model=_graph_model(innovation_variance=1e-7),
    )
    observations = plan.physical.readout_trajectories_m[1, 1:3].astype(float)
    observations[:, :, 1] += 0.002

    belief = planner.assimilate(
        plan,
        observations,
        observation_frame_stop=2,
    )

    assert belief.twin_particle_marginal[1] > 0.95
    assert belief.twin_particle_marginal[0] < 0.05
    assert belief.graph_discrepancy_coefficient_mean is not None
    assert belief.graph_discrepancy_coefficient_mean[1, 0, 1] > 0.0


def test_graph_discrepancy_is_robust_to_one_bad_track() -> None:
    plan = _plan("right", 1.0, node_count=2)
    planner = RecedingHorizonPlanner(
        PlanningConstraints(0.1, 0.2, 0.1),
        observation_scale_m=0.001,
        graph_discrepancy_model=_graph_model(innovation_variance=4e-4),
    )
    observations = plan.physical.readout_trajectories_m[0, 1:2].astype(float)
    observations[0, 0, 0] += 0.005
    observations[0, 1, 0] += 1.0
    mask = np.zeros_like(observations, dtype=bool)
    mask[:, :, 0] = True

    belief = planner.assimilate(
        plan,
        observations,
        observation_frame_stop=1,
        mask=mask,
    )

    assert belief.graph_discrepancy_coefficient_mean is not None
    assert abs(belief.graph_discrepancy_coefficient_mean[0, 0, 0]) < 0.05


def test_graph_discrepancy_transports_without_entering_physical_artifact() -> None:
    first_plan = _plan("right", 1.0, node_count=2)
    planner = RecedingHorizonPlanner(
        PlanningConstraints(0.1, 0.2, 0.1),
        observation_scale_m=0.001,
        graph_discrepancy_model=_graph_model(),
    )
    observations = first_plan.physical.readout_trajectories_m[1, 1:3].astype(float)
    observations[:, :, 2] += 0.003
    belief = planner.assimilate(
        first_plan,
        observations,
        observation_frame_stop=2,
    )
    next_plan = _plan("next", 1.0, initial_x=0.04, node_count=2)

    conditioned = condition_plan_on_recursive_belief(next_plan, belief)
    without_graph = condition_plan_on_recursive_belief(
        next_plan,
        replace(
            belief,
            graph_discrepancy_coefficient_mean=None,
            graph_discrepancy_coefficient_covariance=None,
        ),
    )

    np.testing.assert_allclose(
        conditioned.graph_discrepancy_coefficient_mean,
        belief.graph_discrepancy_coefficient_mean,
    )
    np.testing.assert_allclose(
        conditioned.graph_discrepancy_coefficient_covariance,
        belief.graph_discrepancy_coefficient_covariance,
    )
    assert conditioned.physical.artifact_id == without_graph.physical.artifact_id
    assert conditioned.graph_discrepancy_coefficient_mean.flags.writeable is False
    assert conditioned.graph_discrepancy_coefficient_covariance.flags.writeable is False
