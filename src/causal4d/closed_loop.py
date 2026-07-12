"""Constrained receding-horizon planning over Causal4D posteriors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from causal4d.contracts import PhysicalPosterior, TaskPosterior
from causal4d.physical_validation import physical_posterior_moments


@dataclass(frozen=True)
class PlanningConstraints:
    """Actuation and physical-risk limits enforced before plan selection."""

    maximum_control_step_m: float
    maximum_state_displacement_m: float
    maximum_predictive_std_m: float
    effort_weight: float = 1.0
    risk_weight: float = 1.0

    def __post_init__(self) -> None:
        values = (
            self.maximum_control_step_m,
            self.maximum_state_displacement_m,
            self.maximum_predictive_std_m,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("planning constraint limits must be finite and positive")
        if self.effort_weight < 0.0 or self.risk_weight < 0.0:
            raise ValueError("planning cost weights must be nonnegative")


@dataclass(frozen=True)
class CandidatePlan:
    """One physically simulated action and optional semantic reweighting."""

    action_id: str
    controller_points_m: np.ndarray
    control_anchor_m: np.ndarray
    physical: PhysicalPosterior
    task: TaskPosterior | None = None
    frame_dt_s: float = 1.0

    def __post_init__(self) -> None:
        controls = np.asarray(self.controller_points_m, dtype=float)
        anchor = np.asarray(self.control_anchor_m, dtype=float)
        if not self.action_id:
            raise ValueError("candidate action_id must be nonempty")
        if controls.ndim != 3 or controls.shape[2] != 3 or not np.all(
            np.isfinite(controls)
        ):
            raise ValueError("controller_points_m must have finite shape (H, C, 3)")
        if anchor.shape != controls.shape[1:] or not np.all(np.isfinite(anchor)):
            raise ValueError("control_anchor_m must have shape (C, 3)")
        if len(controls) != self.physical.state_trajectories_m.shape[1] - 1:
            raise ValueError("control horizon must match physical rollout horizon")
        if not np.isfinite(self.frame_dt_s) or self.frame_dt_s <= 0.0:
            raise ValueError("frame_dt_s must be finite and positive")
        if self.task is not None:
            if self.task.physical_posterior_id != self.physical.artifact_id:
                raise ValueError("candidate task does not reference candidate physics")
            if self.task.component_ids != self.physical.component_ids:
                raise ValueError("candidate task and physical support differ")
        object.__setattr__(self, "controller_points_m", controls)
        object.__setattr__(self, "control_anchor_m", anchor)


@dataclass(frozen=True)
class PlanAssessment:
    action_id: str
    feasible: bool
    score: float
    semantic_log_evidence: float
    control_effort: float
    predictive_risk: float
    violations: tuple[str, ...]
    applied_beta: float


@dataclass(frozen=True)
class RecursivePhysicalBelief:
    """Observation-updated support passed into the next replanning call."""

    source_physical_posterior_id: str
    observation_frame_stop: int
    component_weights: np.ndarray
    twin_particle_indices: np.ndarray
    phi_support: np.ndarray
    kappa_support: np.ndarray
    component_state_position_m: np.ndarray
    component_state_velocity_mps: np.ndarray
    twin_particle_marginal: np.ndarray
    phi_mean: np.ndarray
    kappa_mean: np.ndarray
    effective_components: float

    def __post_init__(self) -> None:
        component = np.asarray(self.component_weights, dtype=float).copy()
        particle_indices = np.asarray(
            self.twin_particle_indices,
            dtype=np.int64,
        ).copy()
        phi_support = np.asarray(self.phi_support, dtype=float).copy()
        kappa_support = np.asarray(self.kappa_support, dtype=float).copy()
        state_position = np.asarray(
            self.component_state_position_m,
            dtype=float,
        ).copy()
        state_velocity = np.asarray(
            self.component_state_velocity_mps,
            dtype=float,
        ).copy()
        particles = np.asarray(self.twin_particle_marginal, dtype=float).copy()
        phi = np.asarray(self.phi_mean, dtype=float).copy()
        kappa = np.asarray(self.kappa_mean, dtype=float).copy()
        if component.ndim != 1 or not np.isclose(np.sum(component), 1.0):
            raise ValueError("recursive component weights must sum to one")
        if particles.ndim != 1 or not np.isclose(np.sum(particles), 1.0):
            raise ValueError("recursive particle marginal must sum to one")
        if np.any(component < 0.0) or np.any(particles < 0.0):
            raise ValueError("recursive weights must be nonnegative")
        if particle_indices.shape != component.shape:
            raise ValueError("recursive particle indices must match component support")
        if phi_support.shape[0] != len(component) or kappa_support.shape[0] != len(
            component
        ):
            raise ValueError("recursive latent support must match component weights")
        if (
            state_position.ndim != 3
            or state_position.shape[0] != len(component)
            or state_position.shape[2] != 3
            or state_velocity.shape != state_position.shape
        ):
            raise ValueError("recursive endpoint state must have shape (K, N, 3)")
        for values in (
            component,
            particle_indices,
            phi_support,
            kappa_support,
            state_position,
            state_velocity,
            particles,
            phi,
            kappa,
        ):
            values.setflags(write=False)
        object.__setattr__(self, "component_weights", component)
        object.__setattr__(self, "twin_particle_indices", particle_indices)
        object.__setattr__(self, "phi_support", phi_support)
        object.__setattr__(self, "kappa_support", kappa_support)
        object.__setattr__(self, "component_state_position_m", state_position)
        object.__setattr__(self, "component_state_velocity_mps", state_velocity)
        object.__setattr__(self, "twin_particle_marginal", particles)
        object.__setattr__(self, "phi_mean", phi)
        object.__setattr__(self, "kappa_mean", kappa)


@dataclass(frozen=True)
class ClosedLoopStep:
    start_frame: int
    stop_frame: int
    selected_action_id: str
    assessments: tuple[PlanAssessment, ...]
    posterior_effective_components: float
    twin_particle_marginal: tuple[float, ...]
    phi_mean: tuple[float, ...]
    kappa_mean: tuple[float, ...]


@dataclass(frozen=True)
class ClosedLoopResult:
    steps: tuple[ClosedLoopStep, ...]
    final_belief: RecursivePhysicalBelief
    language_task_success: bool
    all_selected_plans_feasible: bool


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + float(np.log(np.sum(np.exp(values - maximum))))


class RecedingHorizonPlanner:
    """Select, execute, observe, update, and replan without semantic state updates."""

    def __init__(
        self,
        constraints: PlanningConstraints,
        *,
        observation_scale_m: float = 0.006,
        observation_likelihood_power: float = 8.0,
        degrees_of_freedom: float = 4.0,
    ) -> None:
        if observation_scale_m <= 0.0 or observation_likelihood_power <= 0.0:
            raise ValueError("closed-loop observation likelihood settings are invalid")
        if degrees_of_freedom <= 0.0:
            raise ValueError("degrees_of_freedom must be positive")
        self.constraints = constraints
        self.observation_scale_m = float(observation_scale_m)
        self.observation_likelihood_power = float(observation_likelihood_power)
        self.degrees_of_freedom = float(degrees_of_freedom)

    def assess(self, plan: CandidatePlan) -> PlanAssessment:
        controls = np.concatenate(
            (plan.control_anchor_m[None], plan.controller_points_m),
            axis=0,
        )
        control_steps = np.linalg.norm(np.diff(controls, axis=0), axis=2)
        maximum_control_step = float(np.max(control_steps, initial=0.0))
        mean, variance = physical_posterior_moments(plan.physical)
        displacement = np.linalg.norm(mean - mean[0][None], axis=2)
        maximum_displacement = float(np.max(displacement, initial=0.0))
        maximum_std = float(np.max(np.sqrt(variance), initial=0.0))
        violations = []
        if maximum_control_step > self.constraints.maximum_control_step_m:
            violations.append("control_step_limit")
        if maximum_displacement > self.constraints.maximum_state_displacement_m:
            violations.append("state_displacement_limit")
        if maximum_std > self.constraints.maximum_predictive_std_m:
            violations.append("predictive_risk_limit")
        effort = float(np.sum(np.square(np.diff(controls, axis=0))))
        risk = float(np.mean(np.sqrt(variance)))
        semantic_log_evidence = 0.0
        applied_beta = 0.0
        if plan.task is not None:
            applied_beta = float(plan.task.beta)
            log_terms = np.log(np.maximum(plan.physical.weights, 1e-300)) + (
                applied_beta * plan.task.semantic_log_scores
            )
            semantic_log_evidence = _logsumexp(log_terms)
        score = (
            semantic_log_evidence
            - self.constraints.effort_weight * effort
            - self.constraints.risk_weight * risk
        )
        if violations:
            score = -np.inf
        return PlanAssessment(
            action_id=plan.action_id,
            feasible=not violations,
            score=float(score),
            semantic_log_evidence=semantic_log_evidence,
            control_effort=effort,
            predictive_risk=risk,
            violations=tuple(violations),
            applied_beta=applied_beta,
        )

    def select(
        self,
        plans: Sequence[CandidatePlan],
    ) -> tuple[CandidatePlan, tuple[PlanAssessment, ...]]:
        candidates = tuple(plans)
        if not candidates:
            raise ValueError("closed-loop planning requires candidate plans")
        if len({plan.action_id for plan in candidates}) != len(candidates):
            raise ValueError("candidate action ids must be unique")
        assessments = tuple(self.assess(plan) for plan in candidates)
        feasible = [
            (assessment.score, plan.action_id, plan)
            for plan, assessment in zip(candidates, assessments, strict=True)
            if assessment.feasible
        ]
        if not feasible:
            raise RuntimeError("no candidate plan satisfies the physical constraints")
        _, _, selected = max(feasible, key=lambda value: (value[0], value[1]))
        return selected, assessments

    def assimilate(
        self,
        plan: CandidatePlan,
        observations_m: np.ndarray,
        *,
        observation_frame_stop: int,
        mask: np.ndarray | None = None,
    ) -> RecursivePhysicalBelief:
        """Update theta/phi/kappa from physical evidence, never task weights."""

        observations = np.asarray(observations_m, dtype=float)
        if observations.ndim != 3 or observations.shape[2] != 3:
            raise ValueError("closed-loop observations must have shape (R, N, 3)")
        observed_count = len(observations)
        expected_shape = (
            observed_count,
            plan.physical.readout_trajectories_m.shape[2],
            3,
        )
        if observations.shape != expected_shape or not 1 <= observed_count < plan.physical.readout_trajectories_m.shape[1]:
            raise ValueError("closed-loop observations do not fit the plan horizon")
        valid = np.isfinite(observations)
        if mask is not None:
            supplied = np.asarray(mask, dtype=bool)
            if supplied.shape == observations.shape[:2]:
                supplied = np.repeat(supplied[:, :, None], 3, axis=2)
            if supplied.shape != observations.shape:
                raise ValueError("closed-loop observation mask has an invalid shape")
            valid &= supplied
        if not np.any(valid):
            raise ValueError("closed-loop update has no valid observations")
        predicted = plan.physical.readout_trajectories_m[
            :, 1 : observed_count + 1
        ].astype(float)
        variance = plan.physical.readout_variance_m2[:, None].astype(float)
        scale = np.sqrt(self.observation_scale_m**2 + variance)
        standardized = (predicted - observations[None]) / scale
        terms = -0.5 * (self.degrees_of_freedom + 1.0) * np.log1p(
            np.square(standardized) / self.degrees_of_freedom
        )
        scores = np.sum(np.where(valid[None], terms, 0.0), axis=(1, 2, 3))
        scores /= int(np.sum(valid))
        # The update starts from the physical posterior. Language never enters
        # state, theta, phi, or kappa inference.
        log_weights = np.log(np.maximum(plan.physical.weights, 1e-300)) + (
            self.observation_likelihood_power * scores
        )
        maximum = float(np.max(log_weights))
        weights = np.exp(log_weights - maximum)
        weights /= np.sum(weights)
        particle_count = int(np.max(plan.physical.twin_particle_indices)) + 1
        particle_marginal = np.bincount(
            plan.physical.twin_particle_indices,
            weights=weights,
            minlength=particle_count,
        )
        phi_mean = np.einsum("k,kd->d", weights, plan.physical.phi)
        kappa_mean = np.einsum("k,kd->d", weights, plan.physical.kappa_cf)
        return RecursivePhysicalBelief(
            source_physical_posterior_id=plan.physical.artifact_id,
            observation_frame_stop=observation_frame_stop,
            component_weights=weights,
            twin_particle_indices=plan.physical.twin_particle_indices,
            phi_support=plan.physical.phi,
            kappa_support=plan.physical.kappa_cf,
            component_state_position_m=plan.physical.state_trajectories_m[
                :, observed_count
            ],
            component_state_velocity_mps=(
                plan.physical.state_trajectories_m[:, observed_count]
                - plan.physical.state_trajectories_m[:, observed_count - 1]
            )
            / plan.frame_dt_s,
            twin_particle_marginal=particle_marginal,
            phi_mean=phi_mean,
            kappa_mean=kappa_mean,
            effective_components=1.0 / float(np.sum(np.square(weights))),
        )


def condition_plan_on_recursive_belief(
    plan: CandidatePlan,
    belief: RecursivePhysicalBelief,
) -> CandidatePlan:
    """Transfer the updated ``(theta, phi, kappa)`` joint to a new action plan."""

    latent_mass: dict[tuple[int, tuple[float, ...], tuple[float, ...]], float] = {}
    for index, weight in enumerate(belief.component_weights):
        key = (
            int(belief.twin_particle_indices[index]),
            tuple(map(float, belief.phi_support[index])),
            tuple(map(float, belief.kappa_support[index])),
        )
        latent_mass[key] = latent_mass.get(key, 0.0) + float(weight)
    weights = np.empty_like(plan.physical.weights)
    for index in range(len(weights)):
        key = (
            int(plan.physical.twin_particle_indices[index]),
            tuple(map(float, plan.physical.phi[index])),
            tuple(map(float, plan.physical.kappa_cf[index])),
        )
        weights[index] = latent_mass.get(key, 0.0)
    retained_mass = float(np.sum(weights))
    if retained_mass <= 0.0:
        raise ValueError("new plan has no support for the recursive latent belief")
    weights /= retained_mass
    physical = PhysicalPosterior(
        context=plan.physical.context,
        component_ids=plan.physical.component_ids,
        state_trajectories_m=plan.physical.state_trajectories_m,
        readout_trajectories_m=plan.physical.readout_trajectories_m,
        readout_variance_m2=plan.physical.readout_variance_m2,
        weights=weights,
        phi=plan.physical.phi,
        kappa_cf=plan.physical.kappa_cf,
        hypothesis_indices=plan.physical.hypothesis_indices,
        twin_particle_indices=plan.physical.twin_particle_indices,
        phi_names=plan.physical.phi_names,
        kappa_names=plan.physical.kappa_names,
        source_twin_belief_id=plan.physical.source_twin_belief_id,
        source_factual_intervention_id=plan.physical.source_factual_intervention_id,
        source_query_id=plan.physical.source_query_id,
        metadata={
            **plan.physical.metadata,
            "recursive_parent_physical_posterior_id": (
                belief.source_physical_posterior_id
            ),
            "recursive_observation_frame_stop": belief.observation_frame_stop,
            "recursive_retained_latent_mass": retained_mass,
        },
    )
    task = None
    if plan.task is not None:
        if plan.task.beta == 0.0:
            task_weights = physical.weights.copy()
        else:
            log_weights = np.log(np.maximum(physical.weights, 1e-300)) + (
                plan.task.beta * plan.task.semantic_log_scores
            )
            maximum = float(np.max(log_weights))
            task_weights = np.exp(log_weights - maximum)
            task_weights /= np.sum(task_weights)
        task = TaskPosterior(
            context=physical.context,
            physical_posterior_id=physical.artifact_id,
            component_ids=physical.component_ids,
            physical_weights=physical.weights,
            task_weights=task_weights,
            semantic_log_scores=plan.task.semantic_log_scores,
            beta=plan.task.beta,
            query_node_indices=plan.task.query_node_indices,
            semantic_source=plan.task.semantic_source,
            metadata={
                **plan.task.metadata,
                "recursive_physical_prior": True,
            },
        )
    return CandidatePlan(
        action_id=plan.action_id,
        controller_points_m=plan.controller_points_m,
        control_anchor_m=plan.control_anchor_m,
        physical=physical,
        task=task,
        frame_dt_s=plan.frame_dt_s,
    )


PlanProvider = Callable[
    [int, RecursivePhysicalBelief | None],
    Sequence[CandidatePlan],
]
ObservationProvider = Callable[
    [CandidatePlan, int],
    tuple[np.ndarray, np.ndarray | None],
]
TaskSuccess = Callable[[tuple[ClosedLoopStep, ...], RecursivePhysicalBelief], bool]


def run_receding_horizon(
    planner: RecedingHorizonPlanner,
    plan_provider: PlanProvider,
    observation_provider: ObservationProvider,
    *,
    initial_frame: int,
    stop_frame: int,
    replan_interval_frames: int,
    task_success: TaskSuccess,
) -> ClosedLoopResult:
    """Execute short segments and request fresh plans after every update."""

    if not 0 <= initial_frame < stop_frame or replan_interval_frames < 1:
        raise ValueError("closed-loop frame interval is invalid")
    frame = initial_frame
    belief = None
    steps = []
    while frame < stop_frame:
        plans = tuple(plan_provider(frame, belief))
        if belief is not None:
            plans = tuple(
                condition_plan_on_recursive_belief(plan, belief) for plan in plans
            )
        selected, assessments = planner.select(plans)
        execute_count = min(
            replan_interval_frames,
            stop_frame - frame,
            len(selected.controller_points_m),
        )
        if execute_count < 1:
            raise RuntimeError("selected plan has no executable control segment")
        observations, mask = observation_provider(selected, execute_count)
        next_frame = frame + execute_count
        belief = planner.assimilate(
            selected,
            observations,
            observation_frame_stop=next_frame,
            mask=mask,
        )
        steps.append(
            ClosedLoopStep(
                start_frame=frame,
                stop_frame=next_frame,
                selected_action_id=selected.action_id,
                assessments=assessments,
                posterior_effective_components=belief.effective_components,
                twin_particle_marginal=tuple(map(float, belief.twin_particle_marginal)),
                phi_mean=tuple(map(float, belief.phi_mean)),
                kappa_mean=tuple(map(float, belief.kappa_mean)),
            )
        )
        frame = next_frame
    if belief is None:
        raise RuntimeError("closed-loop planner executed no steps")
    step_tuple = tuple(steps)
    return ClosedLoopResult(
        steps=step_tuple,
        final_belief=belief,
        language_task_success=bool(task_success(step_tuple, belief)),
        all_selected_plans_feasible=all(
            next(
                assessment.feasible
                for assessment in step.assessments
                if assessment.action_id == step.selected_action_id
            )
            for step in step_tuple
        ),
    )
