"""Graph-general Bayesian inference over realized contact interventions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from statistics import NormalDist
from typing import Any, Sequence

import numpy as np

from causal4d.baselines import ParameterPosterior, PredictiveDistribution
from causal4d.benchmark import ObjectProtocol
from causal4d.simulator import (
    Action,
    GraphObject,
    WorldCondition,
    graph_adjacency,
    resolved_contact_nodes,
    simulate,
    simulate_particles,
)


@dataclass(frozen=True)
class LatentContactConfig:
    """Locked hypothesis, observation, calibration, and success-gate choices."""

    observation_fraction: float = 0.20
    observation_noise_std_m: float = 0.0015
    likelihood_scales_m: tuple[float, ...] = (0.0015, 0.0025, 0.004, 0.006)
    dynamic_likelihood_weights: tuple[float, ...] = (0.0, 1.0, 4.0)
    likelihood_powers: tuple[float, ...] = (0.20, 0.50, 1.00)
    posterior_temperatures: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)
    parameter_particle_count: int = 12
    gain_values: tuple[float, ...] = (0.70, 0.85, 1.00)
    delay_values: tuple[int, ...] = (0, 1, 2)
    slip_values: tuple[float, ...] = (0.0, 0.20)
    rotation_values_deg: tuple[float, ...] = (0.0, 8.0)
    node_prior_smoothing: float = 1.0
    categorical_prior_smoothing: float = 0.35
    gain_prior_bandwidth: float = 0.08
    slip_prior_bandwidth: float = 0.08
    rotation_prior_bandwidth_deg: float = 4.0
    confidence_level: float = 0.90
    variance_scale_min: float = 0.001
    variance_scale_max: float = 25.0
    gate_gap_closure: float = 0.50
    gate_matched_degradation: float = 0.10
    gate_coverage_tolerance: float = 0.05
    gate_node_accuracy: float = 0.80
    gate_node_credible_coverage: float = 0.80
    gate_node_calibration_error: float = 0.15
    gate_gain_mae: float = 0.15
    gate_gain_coverage: float = 0.80
    gate_delay_mae_steps: float = 0.50
    gate_delay_map_accuracy: float = 0.80
    gate_delay_coverage: float = 0.80
    gate_minimum_topology_gap_closure: float = 0.0

    def __post_init__(self) -> None:
        if not 0.10 <= self.observation_fraction <= 0.20:
            raise ValueError("observation_fraction must be in [0.10, 0.20]")
        if self.observation_noise_std_m <= 0.0:
            raise ValueError("observation_noise_std_m must be positive")
        if not self.likelihood_scales_m or min(self.likelihood_scales_m) <= 0.0:
            raise ValueError("likelihood scales must be positive")
        if (
            not self.dynamic_likelihood_weights
            or min(self.dynamic_likelihood_weights) < 0.0
        ):
            raise ValueError("dynamic likelihood weights must be non-negative")
        if (
            not self.likelihood_powers
            or min(self.likelihood_powers) <= 0.0
            or max(self.likelihood_powers) > 1.0
        ):
            raise ValueError("likelihood_powers must be in (0, 1]")
        if not self.posterior_temperatures or min(self.posterior_temperatures) < 1.0:
            raise ValueError("posterior_temperatures must be at least one")
        if self.parameter_particle_count < 1:
            raise ValueError("parameter_particle_count must be positive")
        if not self.gain_values or min(self.gain_values) <= 0.0:
            raise ValueError("gain_values must be positive")
        if not self.delay_values or min(self.delay_values) < 0:
            raise ValueError("delay_values must be non-negative")
        if (
            not self.slip_values
            or min(self.slip_values) < 0.0
            or max(self.slip_values) >= 1.0
        ):
            raise ValueError("slip_values must be in [0, 1)")
        if self.node_prior_smoothing <= 0.0 or self.categorical_prior_smoothing <= 0.0:
            raise ValueError("prior smoothing values must be positive")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0, 1)")
        if (
            self.variance_scale_min <= 0.0
            or self.variance_scale_max < self.variance_scale_min
        ):
            raise ValueError("invalid variance scale bounds")

    @property
    def rotation_values_radians(self) -> tuple[float, ...]:
        return tuple(float(np.deg2rad(value)) for value in self.rotation_values_deg)

    def prefix_frame_count(self, frame_count: int) -> int:
        return max(
            3,
            min(frame_count - 1, int(np.ceil(self.observation_fraction * frame_count))),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContactState:
    """One graph contact assignment and its continuous/discrete realization."""

    contact_nodes: tuple[int, ...]
    gain_multiplier: float
    delay_steps: int
    slip_fraction: float
    rotation_radians: float

    def action(self, command: Action) -> Action:
        if len(self.contact_nodes) != len(command.contact_nodes):
            raise ValueError("contact assignment and command cardinalities differ")
        return Action(
            action_id=command.action_id,
            split=command.split,
            contact_nodes=self.contact_nodes,
            commanded_forces=command.commanded_forces,
        )

    def condition(self, *, name: str = "contact_hypothesis") -> WorldCondition:
        return WorldCondition(
            name=name,
            contact_gain_multiplier=self.gain_multiplier,
            contact_delay_steps=self.delay_steps,
            contact_spread=self.slip_fraction,
            control_rotation_radians=self.rotation_radians,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "contact_nodes": list(self.contact_nodes),
            "gain_multiplier": float(self.gain_multiplier),
            "delay_steps": int(self.delay_steps),
            "slip_fraction": float(self.slip_fraction),
            "rotation_degrees": float(np.rad2deg(self.rotation_radians)),
        }


def true_contact_state(
    graph_object: GraphObject,
    action: Action,
    condition: WorldCondition,
) -> ContactState:
    """Extract evaluator-only contact ground truth from a simulated world."""

    return ContactState(
        contact_nodes=resolved_contact_nodes(graph_object, action, condition),
        gain_multiplier=condition.contact_gain_multiplier,
        delay_steps=condition.contact_delay_steps,
        slip_fraction=condition.contact_spread,
        rotation_radians=condition.control_rotation_radians,
    )


@dataclass(frozen=True)
class ContactPrior:
    """Object-agnostic factorized prior learned from source-topology labels."""

    shift_probability: float
    gain_probabilities: tuple[float, ...]
    delay_probabilities: tuple[float, ...]
    slip_probabilities: tuple[float, ...]
    rotation_probabilities: tuple[float, ...]
    source_objects: tuple[str, ...]
    source_condition_count: int
    source_action_split: str

    def __post_init__(self) -> None:
        if not 0.0 < self.shift_probability < 1.0:
            raise ValueError("shift_probability must be in (0, 1)")
        for probabilities in (
            self.gain_probabilities,
            self.delay_probabilities,
            self.slip_probabilities,
            self.rotation_probabilities,
        ):
            values = np.asarray(probabilities, dtype=float)
            if np.any(values <= 0.0) or not np.isclose(np.sum(values), 1.0):
                raise ValueError("prior probabilities must be positive and sum to one")
        if not self.source_objects or self.source_condition_count < 1:
            raise ValueError("contact prior requires labelled source conditions")

    def as_dict(self, config: LatentContactConfig) -> dict[str, Any]:
        return {
            "shift_probability": self.shift_probability,
            "gain": dict(
                zip(map(str, config.gain_values), self.gain_probabilities, strict=True)
            ),
            "delay": dict(
                zip(
                    map(str, config.delay_values), self.delay_probabilities, strict=True
                )
            ),
            "slip": dict(
                zip(map(str, config.slip_values), self.slip_probabilities, strict=True)
            ),
            "rotation_deg": dict(
                zip(
                    map(str, config.rotation_values_deg),
                    self.rotation_probabilities,
                    strict=True,
                )
            ),
            "source_objects": list(self.source_objects),
            "source_condition_count": self.source_condition_count,
            "source_action_split": self.source_action_split,
        }


def _kernel_probabilities(
    candidates: Sequence[float],
    labels: Sequence[float],
    *,
    bandwidth: float,
    smoothing: float,
) -> tuple[float, ...]:
    candidate_values = np.asarray(candidates, dtype=float)
    scores = np.full(candidate_values.shape, smoothing, dtype=float)
    for label in labels:
        scores += np.exp(-0.5 * np.square((candidate_values - label) / bandwidth))
    scores /= np.sum(scores)
    return tuple(map(float, scores))


def _categorical_probabilities(
    candidates: Sequence[int],
    labels: Sequence[int],
    *,
    smoothing: float,
) -> tuple[float, ...]:
    scores = np.full(len(candidates), smoothing, dtype=float)
    index_by_value = {value: index for index, value in enumerate(candidates)}
    for label in labels:
        if label in index_by_value:
            scores[index_by_value[label]] += 1.0
        else:
            nearest = int(np.argmin(np.abs(np.asarray(candidates) - label)))
            scores[nearest] += 1.0
    scores /= np.sum(scores)
    return tuple(map(float, scores))


def fit_contact_prior(
    source_protocols: Sequence[ObjectProtocol],
    config: LatentContactConfig,
    *,
    action_split: str = "train",
) -> ContactPrior:
    """Fit an action-conditioned prior from labelled source topologies only."""

    protocols = tuple(source_protocols)
    if not protocols:
        raise ValueError("at least one source topology is required")
    if action_split == "train":
        conditions = tuple(
            condition
            for protocol in protocols
            for condition in protocol.training_conditions
        )
    elif action_split == "validation":
        conditions = tuple(protocol.validation_condition for protocol in protocols)
    elif action_split == "test":
        conditions = tuple(
            condition
            for protocol in protocols
            for condition in protocol.test_conditions
        )
    else:
        raise ValueError("action_split must be train, validation, or test")
    shift_count = sum(int(condition.shift_contact_nodes) for condition in conditions)
    smoothing = config.node_prior_smoothing
    shift_probability = (shift_count + smoothing) / (len(conditions) + 2.0 * smoothing)
    return ContactPrior(
        shift_probability=float(shift_probability),
        gain_probabilities=_kernel_probabilities(
            config.gain_values,
            [condition.contact_gain_multiplier for condition in conditions],
            bandwidth=config.gain_prior_bandwidth,
            smoothing=config.categorical_prior_smoothing,
        ),
        delay_probabilities=_categorical_probabilities(
            config.delay_values,
            [condition.contact_delay_steps for condition in conditions],
            smoothing=config.categorical_prior_smoothing,
        ),
        slip_probabilities=_kernel_probabilities(
            config.slip_values,
            [condition.contact_spread for condition in conditions],
            bandwidth=config.slip_prior_bandwidth,
            smoothing=config.categorical_prior_smoothing,
        ),
        rotation_probabilities=_kernel_probabilities(
            config.rotation_values_deg,
            [
                float(np.rad2deg(condition.control_rotation_radians))
                for condition in conditions
            ],
            bandwidth=config.rotation_prior_bandwidth_deg,
            smoothing=config.categorical_prior_smoothing,
        ),
        source_objects=tuple(protocol.graph_object.name for protocol in protocols),
        source_condition_count=len(conditions),
        source_action_split=action_split,
    )


@dataclass(frozen=True)
class GraphContactHypothesisModel:
    """Generate graph-relative contact hypotheses on unseen topologies."""

    prior: ContactPrior
    config: LatentContactConfig

    def node_assignments(
        self,
        graph_object: GraphObject,
        action: Action,
    ) -> tuple[tuple[int, ...], ...]:
        adjacency = graph_adjacency(graph_object)
        options = [
            (
                node,
                *tuple(neighbour for neighbour in adjacency[node] if neighbour != node),
            )
            for node in action.contact_nodes
        ]
        assignments: list[tuple[int, ...]] = []
        for assignment in product(*options):
            candidate = tuple(map(int, assignment))
            if len(set(candidate)) == len(candidate) and candidate not in assignments:
                assignments.append(candidate)
        return tuple(assignments)

    def hypotheses(
        self,
        graph_object: GraphObject,
        action: Action,
    ) -> tuple[tuple[ContactState, ...], np.ndarray]:
        assignments = self.node_assignments(graph_object, action)
        nominal = action.contact_nodes
        shifted_count = max(len(assignments) - 1, 1)
        node_probabilities = {
            assignment: (
                1.0 - self.prior.shift_probability
                if assignment == nominal
                else self.prior.shift_probability / shifted_count
            )
            for assignment in assignments
        }
        states: list[ContactState] = []
        weights: list[float] = []
        for (
            node_assignment,
            gain_index,
            delay_index,
            slip_index,
            rotation_index,
        ) in product(
            assignments,
            range(len(self.config.gain_values)),
            range(len(self.config.delay_values)),
            range(len(self.config.slip_values)),
            range(len(self.config.rotation_values_radians)),
        ):
            states.append(
                ContactState(
                    contact_nodes=node_assignment,
                    gain_multiplier=self.config.gain_values[gain_index],
                    delay_steps=self.config.delay_values[delay_index],
                    slip_fraction=self.config.slip_values[slip_index],
                    rotation_radians=self.config.rotation_values_radians[
                        rotation_index
                    ],
                )
            )
            weights.append(
                node_probabilities[node_assignment]
                * self.prior.gain_probabilities[gain_index]
                * self.prior.delay_probabilities[delay_index]
                * self.prior.slip_probabilities[slip_index]
                * self.prior.rotation_probabilities[rotation_index]
            )
        normalized = np.asarray(weights, dtype=float)
        normalized /= np.sum(normalized)
        return tuple(states), normalized


def select_parameter_support(
    posterior: ParameterPosterior,
    maximum_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Retain the highest-mass physical particles and renormalize their weights."""

    count = min(maximum_count, posterior.particles.shape[0])
    order = np.argsort(posterior.weights, kind="mergesort")[::-1][:count]
    weights = posterior.weights[order].copy()
    weights /= np.sum(weights)
    return posterior.particles[order], weights


def _weighted_component_quantile(
    components: np.ndarray,
    weights: np.ndarray,
    probability: float,
) -> np.ndarray:
    """Compute a weighted empirical quantile for every flattened coordinate."""

    order = np.argsort(components, axis=0, kind="mergesort")
    sorted_components = np.take_along_axis(components, order, axis=0)
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights, axis=0)
    indices = np.argmax(cumulative >= probability, axis=0)
    return sorted_components[indices, np.arange(components.shape[1])]


@dataclass(frozen=True)
class ContactRolloutBank:
    """Joint finite posterior support over contact states and physical parameters."""

    graph_object: GraphObject
    action: Action
    contact_states: tuple[ContactState, ...]
    contact_prior_weights: np.ndarray
    parameter_particles: np.ndarray
    parameter_weights: np.ndarray
    trajectories: np.ndarray
    variance_floor_m2: float
    confidence_level: float

    def __post_init__(self) -> None:
        contact_weights = np.asarray(self.contact_prior_weights, dtype=float)
        parameter_weights = np.asarray(self.parameter_weights, dtype=float)
        trajectories = np.asarray(self.trajectories, dtype=float)
        expected = (
            len(self.contact_states),
            self.parameter_particles.shape[0],
            self.action.frame_count,
            self.graph_object.node_count,
            2,
        )
        if trajectories.shape != expected:
            raise ValueError(f"trajectory bank must have shape {expected}")
        if contact_weights.shape != (len(self.contact_states),):
            raise ValueError("contact prior weights have invalid shape")
        if parameter_weights.shape != (self.parameter_particles.shape[0],):
            raise ValueError("parameter weights have invalid shape")
        if not np.isclose(np.sum(contact_weights), 1.0) or not np.isclose(
            np.sum(parameter_weights), 1.0
        ):
            raise ValueError("rollout-bank priors must sum to one")
        if self.variance_floor_m2 <= 0.0:
            raise ValueError("variance_floor_m2 must be positive")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0, 1)")

    @property
    def prior_joint_weights(self) -> np.ndarray:
        return self.contact_prior_weights[:, None] * self.parameter_weights[None, :]

    def update_weights(
        self,
        observations: np.ndarray,
        *,
        prefix_frame_count: int,
        likelihood_scale_m: float,
        likelihood_power: float,
        dynamic_likelihood_weight: float = 0.0,
        observed_nodes: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Update the joint contact/physics posterior using only an early prefix."""

        observations = np.asarray(observations, dtype=float)
        expected_shape = (
            self.action.frame_count,
            self.graph_object.node_count,
            2,
        )
        if observations.shape != expected_shape:
            raise ValueError(f"observations must have shape {expected_shape}")
        if not 2 <= prefix_frame_count < self.action.frame_count:
            raise ValueError("prefix_frame_count must leave at least one future frame")
        nodes = np.asarray(
            tuple(observed_nodes)
            if observed_nodes is not None
            else tuple(range(self.graph_object.node_count)),
            dtype=int,
        )
        predicted = self.trajectories[:, :, 1:prefix_frame_count, :, :]
        predicted = predicted[:, :, :, nodes, :]
        observed = observations[1:prefix_frame_count, nodes, :]
        squared_error = np.sum(
            np.square(predicted - observed[None, None, ...]),
            axis=(2, 3, 4),
        )
        if dynamic_likelihood_weight:
            predicted_velocity = np.diff(predicted, axis=2)
            observed_velocity = np.diff(observed, axis=0)
            velocity_error = np.sum(
                np.square(predicted_velocity - observed_velocity[None, None, ...]),
                axis=(2, 3, 4),
            )
            predicted_acceleration = np.diff(predicted_velocity, axis=2)
            observed_acceleration = np.diff(observed_velocity, axis=0)
            acceleration_error = np.sum(
                np.square(
                    predicted_acceleration - observed_acceleration[None, None, ...]
                ),
                axis=(2, 3, 4),
            )
            squared_error += dynamic_likelihood_weight * (
                0.5 * velocity_error + acceleration_error / 6.0
            )
        log_weights = np.log(np.maximum(self.prior_joint_weights, 1e-300))
        log_weights -= 0.5 * likelihood_power * squared_error / likelihood_scale_m**2
        log_weights -= float(np.max(log_weights))
        weights = np.exp(log_weights)
        weights /= np.sum(weights)
        return weights

    def predictive_distribution(
        self,
        joint_weights: np.ndarray | None = None,
        *,
        method: str,
        variance_multiplier: float = 1.0,
        include_intervals: bool = True,
    ) -> PredictiveDistribution:
        weights = (
            self.prior_joint_weights
            if joint_weights is None
            else np.asarray(joint_weights, dtype=float)
        )
        if weights.shape != self.prior_joint_weights.shape or not np.isclose(
            np.sum(weights), 1.0
        ):
            raise ValueError("joint_weights must match the bank and sum to one")
        mean = np.sum(weights[:, :, None, None, None] * self.trajectories, axis=(0, 1))
        variance = np.sum(
            weights[:, :, None, None, None]
            * np.square(self.trajectories - mean[None, None, ...]),
            axis=(0, 1),
        )
        variance = variance_multiplier * (variance + self.variance_floor_m2)
        if not include_intervals:
            return PredictiveDistribution(method=method, mean=mean, variance=variance)
        component_trajectories = mean[None, None, ...] + np.sqrt(
            variance_multiplier
        ) * (self.trajectories - mean[None, None, ...])
        flat_components = component_trajectories.reshape(-1, int(np.prod(mean.shape)))
        flat_weights = weights.reshape(-1)
        tail = 0.5 * (1.0 - self.confidence_level)
        lower = _weighted_component_quantile(
            flat_components, flat_weights, tail
        ).reshape(mean.shape)
        upper = _weighted_component_quantile(
            flat_components, flat_weights, 1.0 - tail
        ).reshape(mean.shape)
        normal_quantile = NormalDist().inv_cdf(0.5 * (1.0 + self.confidence_level))
        floor_margin = normal_quantile * np.sqrt(
            variance_multiplier * self.variance_floor_m2
        )
        return PredictiveDistribution(
            method=method,
            mean=mean,
            variance=variance,
            interval_lower=lower - floor_margin,
            interval_upper=upper + floor_margin,
        )

    def contact_marginal(self, joint_weights: np.ndarray) -> np.ndarray:
        weights = np.asarray(joint_weights, dtype=float)
        if weights.shape != self.prior_joint_weights.shape:
            raise ValueError("joint_weights have invalid shape")
        marginal = np.sum(weights, axis=1)
        marginal /= np.sum(marginal)
        return marginal

    def parameter_marginal(self, joint_weights: np.ndarray) -> np.ndarray:
        weights = np.asarray(joint_weights, dtype=float)
        if weights.shape != self.prior_joint_weights.shape:
            raise ValueError("joint_weights have invalid shape")
        marginal = np.sum(weights, axis=0)
        marginal /= np.sum(marginal)
        return marginal


def build_rollout_bank(
    graph_object: GraphObject,
    action: Action,
    posterior: ParameterPosterior,
    model: GraphContactHypothesisModel,
    *,
    simulator_config,
    parameter_particle_count: int,
    variance_floor_m2: float,
    confidence_level: float,
) -> ContactRolloutBank:
    """Simulate every graph contact hypothesis under selected physical particles."""

    states, contact_weights = model.hypotheses(graph_object, action)
    particles, parameter_weights = select_parameter_support(
        posterior, parameter_particle_count
    )
    trajectories = np.stack(
        [
            simulate_particles(
                graph_object,
                state.action(action),
                particles,
                state.condition(),
                simulator_config,
            )
            for state in states
        ],
        axis=0,
    )
    return ContactRolloutBank(
        graph_object=graph_object,
        action=action,
        contact_states=states,
        contact_prior_weights=contact_weights,
        parameter_particles=particles,
        parameter_weights=parameter_weights,
        trajectories=trajectories,
        variance_floor_m2=variance_floor_m2,
        confidence_level=confidence_level,
    )


def posterior_predictive_for_state(
    graph_object: GraphObject,
    action: Action,
    state: ContactState,
    posterior: ParameterPosterior,
    *,
    simulator_config,
    variance_floor_m2: float,
    method: str,
    observations: np.ndarray | None = None,
    prefix_frame_count: int | None = None,
    likelihood_scale_m: float | None = None,
    likelihood_power: float | None = None,
    dynamic_likelihood_weight: float = 0.0,
    posterior_temperature: float = 1.0,
) -> PredictiveDistribution:
    """Return a fixed-contact predictive, optionally updating theta from a prefix."""

    trajectories = simulate_particles(
        graph_object,
        state.action(action),
        posterior.particles,
        state.condition(name=method),
        simulator_config,
    )
    particle_weights = posterior.weights.copy()
    if observations is not None:
        if (
            prefix_frame_count is None
            or likelihood_scale_m is None
            or likelihood_power is None
        ):
            raise ValueError(
                "online fixed-contact prediction requires likelihood settings"
            )
        observations = np.asarray(observations, dtype=float)
        if observations.shape != trajectories.shape[1:]:
            raise ValueError(
                "observations and fixed-contact trajectories differ in shape"
            )
        predicted = trajectories[:, 1:prefix_frame_count]
        observed = observations[1:prefix_frame_count]
        squared_error = np.sum(
            np.square(predicted - observed[None, ...]), axis=(1, 2, 3)
        )
        if dynamic_likelihood_weight:
            predicted_velocity = np.diff(predicted, axis=1)
            observed_velocity = np.diff(observed, axis=0)
            velocity_error = np.sum(
                np.square(predicted_velocity - observed_velocity[None, ...]),
                axis=(1, 2, 3),
            )
            predicted_acceleration = np.diff(predicted_velocity, axis=1)
            observed_acceleration = np.diff(observed_velocity, axis=0)
            acceleration_error = np.sum(
                np.square(predicted_acceleration - observed_acceleration[None, ...]),
                axis=(1, 2, 3),
            )
            squared_error += dynamic_likelihood_weight * (
                0.5 * velocity_error + acceleration_error / 6.0
            )
        log_weights = np.log(np.maximum(particle_weights, 1e-300))
        log_weights -= 0.5 * likelihood_power * squared_error / likelihood_scale_m**2
        log_weights *= posterior_temperature
        log_weights -= float(np.max(log_weights))
        particle_weights = np.exp(log_weights)
        particle_weights /= np.sum(particle_weights)
    elif any(
        value is not None
        for value in (prefix_frame_count, likelihood_scale_m, likelihood_power)
    ):
        raise ValueError("observations are required for an online theta update")

    weights = particle_weights[:, None, None, None]
    mean = np.sum(weights * trajectories, axis=0)
    variance = np.sum(weights * np.square(trajectories - mean[None, ...]), axis=0)
    variance += variance_floor_m2
    return PredictiveDistribution(method=method, mean=mean, variance=variance)


def true_parameter_predictive_for_state(
    graph_object: GraphObject,
    action: Action,
    state: ContactState,
    *,
    simulator_config,
    variance_floor_m2: float,
    method: str = "oracle_contact_theta",
) -> PredictiveDistribution:
    """Return the strict simulation ceiling with evaluator-only z and theta."""

    trajectory = simulate(
        graph_object,
        state.action(action),
        graph_object.true_parameters,
        state.condition(name=method),
        simulator_config,
    )
    variance = np.full_like(trajectory, variance_floor_m2)
    return PredictiveDistribution(method=method, mean=trajectory, variance=variance)
