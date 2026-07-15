"""Constrained receding-horizon planning over Causal4D posteriors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import numpy as np

from causal4d.contracts import PhysicalPosterior, TaskPosterior
from causal4d.graph_temporal_discrepancy import GraphTemporalDiscrepancyModel
from causal4d.physical_validation import physical_posterior_moments


def _validated_graph_discrepancy_state(
    coefficient_mean: np.ndarray | None,
    coefficient_covariance: np.ndarray | None,
    *,
    component_count: int,
    owner: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Validate a component-wise Gaussian state over graph coefficients."""

    if (coefficient_mean is None) != (coefficient_covariance is None):
        raise ValueError(
            f"{owner} graph discrepancy mean and covariance must be supplied together"
        )
    if coefficient_mean is None:
        return None, None
    mean = np.asarray(coefficient_mean, dtype=float).copy()
    covariance = np.asarray(coefficient_covariance, dtype=float).copy()
    if mean.ndim != 3 or mean.shape[0] != component_count or mean.shape[2] != 3:
        raise ValueError(f"{owner} graph discrepancy mean must have shape (K, rank, 3)")
    rank = mean.shape[1]
    if rank < 1 or covariance.shape != (component_count, 3, rank, rank):
        raise ValueError(
            f"{owner} graph discrepancy covariance must have shape (K, 3, rank, rank)"
        )
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
        raise ValueError(f"{owner} graph discrepancy state must be finite")
    if not np.allclose(
        covariance,
        covariance.swapaxes(-1, -2),
        atol=1e-10,
        rtol=1e-10,
    ):
        raise ValueError(f"{owner} graph discrepancy covariance must be symmetric")
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(covariance), initial=0.0))
    if minimum_eigenvalue < -1e-10:
        raise ValueError(
            f"{owner} graph discrepancy covariance must be positive semidefinite"
        )
    mean.setflags(write=False)
    covariance.setflags(write=False)
    return mean, covariance


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
    graph_discrepancy_coefficient_mean: np.ndarray | None = None
    graph_discrepancy_coefficient_covariance: np.ndarray | None = None

    def __post_init__(self) -> None:
        controls = np.asarray(self.controller_points_m, dtype=float)
        anchor = np.asarray(self.control_anchor_m, dtype=float)
        if not self.action_id:
            raise ValueError("candidate action_id must be nonempty")
        if (
            controls.ndim != 3
            or controls.shape[2] != 3
            or not np.all(np.isfinite(controls))
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
        discrepancy_mean, discrepancy_covariance = _validated_graph_discrepancy_state(
            self.graph_discrepancy_coefficient_mean,
            self.graph_discrepancy_coefficient_covariance,
            component_count=len(self.physical.weights),
            owner="candidate",
        )
        object.__setattr__(self, "controller_points_m", controls)
        object.__setattr__(self, "control_anchor_m", anchor)
        object.__setattr__(
            self,
            "graph_discrepancy_coefficient_mean",
            discrepancy_mean,
        )
        object.__setattr__(
            self,
            "graph_discrepancy_coefficient_covariance",
            discrepancy_covariance,
        )


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
    graph_discrepancy_coefficient_mean: np.ndarray | None = None
    graph_discrepancy_coefficient_covariance: np.ndarray | None = None

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
        discrepancy_mean, discrepancy_covariance = _validated_graph_discrepancy_state(
            self.graph_discrepancy_coefficient_mean,
            self.graph_discrepancy_coefficient_covariance,
            component_count=len(component),
            owner="recursive",
        )
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
        object.__setattr__(
            self,
            "graph_discrepancy_coefficient_mean",
            discrepancy_mean,
        )
        object.__setattr__(
            self,
            "graph_discrepancy_coefficient_covariance",
            discrepancy_covariance,
        )


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


def _positive_semidefinite(matrix: np.ndarray) -> np.ndarray:
    """Remove only floating-point negative covariance eigenvalues."""

    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    return (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T


def graph_discrepancy_adjusted_plan_moments(
    plan: CandidatePlan,
    model: GraphTemporalDiscrepancyModel,
    *,
    dynamics: Literal["persistence", "learned"] = "persistence",
    projection_variance_mode: Literal[
        "included_in_readout", "add"
    ] = "included_in_readout",
) -> tuple[np.ndarray, np.ndarray]:
    """Moment-match a plan after forecasting its separate graph-field belief.

    ``included_in_readout`` avoids counting the model's projection residual twice
    when ``PhysicalPosterior.readout_variance_m2`` was calibrated from the same
    residual. Select ``add`` only when the physical variance excludes it.
    """

    if dynamics not in {"persistence", "learned"}:
        raise ValueError("graph discrepancy dynamics must be persistence or learned")
    if projection_variance_mode not in {"included_in_readout", "add"}:
        raise ValueError("graph projection variance mode must be included or add")
    source_mean = plan.graph_discrepancy_coefficient_mean
    source_covariance = plan.graph_discrepancy_coefficient_covariance
    if source_mean is None or source_covariance is None:
        raise ValueError("adjusted moments require a graph discrepancy belief")
    component_count, frame_count, node_count, _ = (
        plan.physical.readout_trajectories_m.shape
    )
    rank = model.selected_rank
    if source_mean.shape != (component_count, rank, 3):
        raise ValueError("candidate graph discrepancy rank does not match the model")
    if model.basis.shape[0] < node_count:
        raise ValueError("graph discrepancy basis does not cover plan readout nodes")
    basis = model.basis[:node_count]
    transition = model.transition if dynamics == "learned" else np.eye(rank)
    coefficient_mean = source_mean.copy()
    coefficient_covariance = source_covariance.copy()
    components = plan.physical.readout_trajectories_m.astype(float).copy()
    conditional_variance = np.broadcast_to(
        plan.physical.readout_variance_m2[:, None].astype(float),
        components.shape,
    ).copy()
    projection_variance = (
        model.projection_variance_m2
        if projection_variance_mode == "add"
        else np.zeros(3, dtype=float)
    )
    for frame in range(frame_count):
        if frame > 0:
            coefficient_mean = np.einsum(
                "ij,kjc->kic",
                transition,
                coefficient_mean,
            )
            for component in range(component_count):
                for coordinate in range(3):
                    coefficient_covariance[component, coordinate] = (
                        _positive_semidefinite(
                            transition
                            @ coefficient_covariance[component, coordinate]
                            @ transition.T
                            + model.innovation_covariance
                        )
                    )
        components[:, frame] += np.einsum(
            "nr,krc->knc",
            basis,
            coefficient_mean,
        )
        for component in range(component_count):
            for coordinate in range(3):
                conditional_variance[component, frame, :, coordinate] += (
                    np.einsum(
                        "ni,ij,nj->n",
                        basis,
                        coefficient_covariance[component, coordinate],
                        basis,
                    )
                    + projection_variance[coordinate]
                )
    mean = np.einsum("k,ktnc->tnc", plan.physical.weights, components)
    centered = components - mean[None]
    epistemic = np.einsum(
        "k,ktnc->tnc",
        plan.physical.weights,
        np.square(centered),
    )
    conditional = np.einsum(
        "k,ktnc->tnc",
        plan.physical.weights,
        conditional_variance,
    )
    return mean, np.maximum(epistemic + conditional, np.finfo(float).tiny)


class RecedingHorizonPlanner:
    """Select, execute, observe, update, and replan without semantic state updates."""

    def __init__(
        self,
        constraints: PlanningConstraints,
        *,
        observation_scale_m: float = 0.006,
        observation_likelihood_power: float = 8.0,
        degrees_of_freedom: float = 4.0,
        graph_discrepancy_model: GraphTemporalDiscrepancyModel | None = None,
        graph_discrepancy_dynamics: Literal["persistence", "learned"] = "persistence",
        graph_projection_variance_mode: Literal[
            "included_in_readout", "add"
        ] = "included_in_readout",
    ) -> None:
        if observation_scale_m <= 0.0 or observation_likelihood_power <= 0.0:
            raise ValueError("closed-loop observation likelihood settings are invalid")
        if degrees_of_freedom <= 0.0:
            raise ValueError("degrees_of_freedom must be positive")
        if graph_discrepancy_dynamics not in {"persistence", "learned"}:
            raise ValueError(
                "graph_discrepancy_dynamics must be 'persistence' or 'learned'"
            )
        if graph_projection_variance_mode not in {"included_in_readout", "add"}:
            raise ValueError(
                "graph_projection_variance_mode must be 'included_in_readout' or 'add'"
            )
        self.constraints = constraints
        self.observation_scale_m = float(observation_scale_m)
        self.observation_likelihood_power = float(observation_likelihood_power)
        self.degrees_of_freedom = float(degrees_of_freedom)
        self.graph_discrepancy_model = graph_discrepancy_model
        self.graph_discrepancy_dynamics = graph_discrepancy_dynamics
        self.graph_projection_variance_mode = graph_projection_variance_mode

    def assess(self, plan: CandidatePlan) -> PlanAssessment:
        controls = np.concatenate(
            (plan.control_anchor_m[None], plan.controller_points_m),
            axis=0,
        )
        control_steps = np.linalg.norm(np.diff(controls, axis=0), axis=2)
        maximum_control_step = float(np.max(control_steps, initial=0.0))
        if (
            self.graph_discrepancy_model is None
            and plan.graph_discrepancy_coefficient_mean is not None
        ):
            raise ValueError(
                "candidate graph discrepancy state requires a planner model"
            )
        if plan.graph_discrepancy_coefficient_mean is not None:
            mean, variance = graph_discrepancy_adjusted_plan_moments(
                plan,
                self.graph_discrepancy_model,
                dynamics=self.graph_discrepancy_dynamics,
                projection_variance_mode=self.graph_projection_variance_mode,
            )
        else:
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

    def _assimilate_graph_discrepancy(
        self,
        plan: CandidatePlan,
        observations: np.ndarray,
        valid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filter a separate low-rank readout discrepancy for every component."""

        model = self.graph_discrepancy_model
        if model is None:
            raise RuntimeError("graph discrepancy update requires a model")
        component_count = len(plan.physical.weights)
        node_count = observations.shape[1]
        rank = model.selected_rank
        if model.basis.shape[0] < node_count:
            raise ValueError(
                "graph discrepancy basis does not cover the physical readout nodes"
            )
        if plan.graph_discrepancy_coefficient_mean is None:
            coefficient_mean = np.zeros((component_count, rank, 3), dtype=float)
            coefficient_covariance = np.zeros(
                (component_count, 3, rank, rank),
                dtype=float,
            )
        else:
            coefficient_mean = np.asarray(
                plan.graph_discrepancy_coefficient_mean,
                dtype=float,
            ).copy()
            coefficient_covariance = np.asarray(
                plan.graph_discrepancy_coefficient_covariance,
                dtype=float,
            ).copy()
            if coefficient_mean.shape != (component_count, rank, 3):
                raise ValueError(
                    "candidate graph discrepancy rank does not match the planner model"
                )

        basis = model.basis[:node_count]
        transition = (
            model.transition
            if self.graph_discrepancy_dynamics == "learned"
            else np.eye(rank)
        )
        innovation_covariance = model.innovation_covariance
        scores = np.zeros(component_count, dtype=float)
        valid_count = int(np.sum(valid))
        identity = np.eye(rank)
        for frame in range(len(observations)):
            coefficient_mean = np.einsum(
                "ij,kjc->kic",
                transition,
                coefficient_mean,
            )
            for component in range(component_count):
                for coordinate in range(3):
                    prior_covariance = coefficient_covariance[
                        component,
                        coordinate,
                    ]
                    coefficient_covariance[component, coordinate] = (
                        _positive_semidefinite(
                            transition @ prior_covariance @ transition.T
                            + innovation_covariance
                        )
                    )

            for component in range(component_count):
                physical_prediction = plan.physical.readout_trajectories_m[
                    component,
                    frame + 1,
                ].astype(float)
                for coordinate in range(3):
                    selected = np.flatnonzero(valid[frame, :, coordinate])
                    if len(selected) == 0:
                        continue
                    mean = coefficient_mean[component, :, coordinate]
                    covariance = coefficient_covariance[component, coordinate]
                    design = basis[selected]
                    noise_variance = (
                        self.observation_scale_m**2
                        + plan.physical.readout_variance_m2[
                            component,
                            selected,
                            coordinate,
                        ].astype(float)
                        + (
                            model.projection_variance_m2[coordinate]
                            if self.graph_projection_variance_mode == "add"
                            else 0.0
                        )
                    )
                    noise_variance = np.maximum(noise_variance, 1e-15)
                    residual = (
                        observations[frame, selected, coordinate]
                        - physical_prediction[selected, coordinate]
                        - design @ mean
                    )
                    predictive_variance = noise_variance + np.einsum(
                        "ni,ij,nj->n",
                        design,
                        covariance,
                        design,
                    )
                    scores[component] += float(
                        np.sum(
                            -0.5 * np.log(predictive_variance)
                            - (
                                0.5
                                * (self.degrees_of_freedom + 1.0)
                                * np.log1p(
                                    np.square(residual)
                                    / (self.degrees_of_freedom * predictive_variance)
                                )
                            )
                        )
                    )

                    # A sequential Student-t Kalman update bounds the influence
                    # of individual bad tracks while retaining partial-node and
                    # coordinate masks. The field corrects readout likelihoods;
                    # it is deliberately never injected into physical state.
                    for node, measurement_variance in zip(
                        selected,
                        noise_variance,
                        strict=True,
                    ):
                        design_row = basis[node]
                        innovation = float(
                            observations[frame, node, coordinate]
                            - physical_prediction[node, coordinate]
                            - design_row @ mean
                        )
                        latent_variance = float(design_row @ covariance @ design_row)
                        predictive_scalar_variance = (
                            latent_variance + measurement_variance
                        )
                        standardized_square = innovation**2 / predictive_scalar_variance
                        robust_weight = min(
                            1.0,
                            (self.degrees_of_freedom + 1.0)
                            / (self.degrees_of_freedom + standardized_square),
                        )
                        effective_variance = measurement_variance / max(
                            robust_weight,
                            1e-6,
                        )
                        denominator = latent_variance + effective_variance
                        gain = covariance @ design_row / denominator
                        mean = mean + gain * innovation
                        update = identity - np.outer(gain, design_row)
                        covariance = (
                            update @ covariance @ update.T
                            + np.outer(gain, gain) * effective_variance
                        )
                    coefficient_mean[component, :, coordinate] = mean
                    coefficient_covariance[component, coordinate] = (
                        _positive_semidefinite(covariance)
                    )
        return coefficient_mean, coefficient_covariance, scores / valid_count

    def assimilate(
        self,
        plan: CandidatePlan,
        observations_m: np.ndarray,
        *,
        observation_frame_stop: int,
        mask: np.ndarray | None = None,
    ) -> RecursivePhysicalBelief:
        """Update physical support and optional readout discrepancy from evidence."""

        observations = np.asarray(observations_m, dtype=float)
        if observations.ndim != 3 or observations.shape[2] != 3:
            raise ValueError("closed-loop observations must have shape (R, N, 3)")
        observed_count = len(observations)
        expected_shape = (
            observed_count,
            plan.physical.readout_trajectories_m.shape[2],
            3,
        )
        if (
            observations.shape != expected_shape
            or not 1 <= observed_count < plan.physical.readout_trajectories_m.shape[1]
        ):
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
        discrepancy_mean = None
        discrepancy_covariance = None
        if self.graph_discrepancy_model is None:
            if plan.graph_discrepancy_coefficient_mean is not None:
                raise ValueError(
                    "candidate graph discrepancy state requires a planner model"
                )
            predicted = plan.physical.readout_trajectories_m[
                :, 1 : observed_count + 1
            ].astype(float)
            variance = plan.physical.readout_variance_m2[:, None].astype(float)
            scale = np.sqrt(self.observation_scale_m**2 + variance)
            standardized = (predicted - observations[None]) / scale
            terms = (
                -0.5
                * (self.degrees_of_freedom + 1.0)
                * np.log1p(np.square(standardized) / self.degrees_of_freedom)
            )
            scores = np.sum(
                np.where(valid[None], terms, 0.0),
                axis=(1, 2, 3),
            )
            scores /= int(np.sum(valid))
        else:
            discrepancy_mean, discrepancy_covariance, scores = (
                self._assimilate_graph_discrepancy(
                    plan,
                    observations,
                    valid,
                )
            )
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
            graph_discrepancy_coefficient_mean=discrepancy_mean,
            graph_discrepancy_coefficient_covariance=discrepancy_covariance,
        )


def _latent_support_key(
    particle_index: int,
    phi: np.ndarray,
    kappa: np.ndarray,
) -> tuple[int, tuple[float, ...], tuple[float, ...]]:
    return (
        int(particle_index),
        tuple(map(float, phi)),
        tuple(map(float, kappa)),
    )


def _transport_graph_discrepancy(
    plan: CandidatePlan,
    belief: RecursivePhysicalBelief,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Moment-match coefficient state onto a replanned latent support."""

    source_mean = belief.graph_discrepancy_coefficient_mean
    source_covariance = belief.graph_discrepancy_coefficient_covariance
    if source_mean is None:
        return (
            plan.graph_discrepancy_coefficient_mean,
            plan.graph_discrepancy_coefficient_covariance,
        )
    rank = source_mean.shape[1]
    target_mean = np.zeros((len(plan.physical.weights), rank, 3), dtype=float)
    target_covariance = np.zeros(
        (len(plan.physical.weights), 3, rank, rank),
        dtype=float,
    )
    members: dict[
        tuple[int, tuple[float, ...], tuple[float, ...]],
        list[int],
    ] = {}
    for index in range(len(belief.component_weights)):
        key = _latent_support_key(
            belief.twin_particle_indices[index],
            belief.phi_support[index],
            belief.kappa_support[index],
        )
        members.setdefault(key, []).append(index)

    moments: dict[
        tuple[int, tuple[float, ...], tuple[float, ...]],
        tuple[np.ndarray, np.ndarray],
    ] = {}
    for key, indices in members.items():
        member_weights = belief.component_weights[indices]
        mass = float(np.sum(member_weights))
        if mass <= 0.0:
            continue
        member_weights = member_weights / mass
        mean = np.einsum(
            "k,krc->rc",
            member_weights,
            source_mean[indices],
        )
        covariance = np.zeros((3, rank, rank), dtype=float)
        for local_index, source_index in enumerate(indices):
            difference = source_mean[source_index] - mean
            for coordinate in range(3):
                covariance[coordinate] += member_weights[local_index] * (
                    source_covariance[source_index, coordinate]
                    + np.outer(
                        difference[:, coordinate],
                        difference[:, coordinate],
                    )
                )
        moments[key] = mean, covariance

    for index in range(len(plan.physical.weights)):
        key = _latent_support_key(
            plan.physical.twin_particle_indices[index],
            plan.physical.phi[index],
            plan.physical.kappa_cf[index],
        )
        if key in moments:
            target_mean[index], target_covariance[index] = moments[key]
    return target_mean, target_covariance


def condition_plan_on_recursive_belief(
    plan: CandidatePlan,
    belief: RecursivePhysicalBelief,
) -> CandidatePlan:
    """Transfer physical and graph-discrepancy beliefs to a new action plan."""

    latent_mass: dict[tuple[int, tuple[float, ...], tuple[float, ...]], float] = {}
    for index, weight in enumerate(belief.component_weights):
        key = _latent_support_key(
            belief.twin_particle_indices[index],
            belief.phi_support[index],
            belief.kappa_support[index],
        )
        latent_mass[key] = latent_mass.get(key, 0.0) + float(weight)
    weights = np.empty_like(plan.physical.weights)
    for index in range(len(weights)):
        key = _latent_support_key(
            plan.physical.twin_particle_indices[index],
            plan.physical.phi[index],
            plan.physical.kappa_cf[index],
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
    discrepancy_mean, discrepancy_covariance = _transport_graph_discrepancy(
        plan,
        belief,
    )
    return CandidatePlan(
        action_id=plan.action_id,
        controller_points_m=plan.controller_points_m,
        control_anchor_m=plan.control_anchor_m,
        physical=physical,
        task=task,
        frame_dt_s=plan.frame_dt_s,
        graph_discrepancy_coefficient_mean=discrepancy_mean,
        graph_discrepancy_coefficient_covariance=discrepancy_covariance,
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
