import numpy as np

from causal4d.closed_loop import (
    CandidatePlan,
    PlanningConstraints,
    RecedingHorizonPlanner,
    run_receding_horizon,
)
from causal4d.contracts import PhysicalPosterior, build_causal_context
from causal4d.semantic_posterior import SparseSemanticEvidence, build_task_posterior


def _physical(
    action_id: str,
    direction: float,
    *,
    initial_x: float = 0.0,
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
    trajectories = np.zeros((2, 5, 1, 3), dtype=float)
    trajectories[0, :, 0, 0] = initial_x + direction * np.arange(5) * 0.01
    trajectories[1, :, 0, 0] = initial_x + direction * np.arange(5) * 0.02
    return PhysicalPosterior(
        context=context,
        component_ids=(f"{action_id}:p0", f"{action_id}:p1"),
        state_trajectories_m=trajectories,
        readout_trajectories_m=trajectories,
        readout_variance_m2=np.full((2, 1, 3), 1e-6),
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
) -> CandidatePlan:
    physical = _physical(action_id, direction, initial_x=initial_x)
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
    controls[:, 0, 0] = (
        np.arange(1, 5) * (0.5 if unsafe else 0.01) * direction
    )
    return CandidatePlan(
        action_id=action_id,
        controller_points_m=controls,
        control_anchor_m=np.zeros((1, 3)),
        physical=physical,
        task=task,
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
