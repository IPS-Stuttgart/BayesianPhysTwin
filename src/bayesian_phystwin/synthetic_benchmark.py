"""Controlled fixed-graph benchmark for state and parameter uncertainty."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .calibration import binary_calibration_metrics
from .drift_bias import RandomWalkBiasConfig, robust_random_walk_log_evidence_batch
from .pseudo_measurements import (
    PseudoMeasurementBatch,
    ReliabilityConfig,
    score_reliability,
)
from .structured_reliability import (
    MarkovReliabilityConfig,
    markov_log_evidence_batch,
    smooth_markov_reliability,
)


METHODS = (
    "unweighted_gaussian",
    "huber",
    "cue_weighted",
    "iid_mixture",
    "markov_mixture",
    "structured_bias_filter",
    "oracle_covariates",
    "oracle_inliers",
)
PARAMETER_NAMES = ("stiffness", "damping", "control_scale")


@dataclass(frozen=True)
class SyntheticBenchmarkConfig:
    """Configuration for a one-dimensional fixed spring-graph benchmark."""

    node_count: int = 5
    step_count: int = 90
    train_step_count: int = 60
    time_step: float = 0.025
    observation_std: float = 0.006
    true_stiffness: float = 6.0
    true_damping: float = 0.45
    true_control_scale: float = 1.0
    stiffness_min: float = 4.0
    stiffness_max: float = 8.0
    stiffness_count: int = 17
    damping_min: float = 0.20
    damping_max: float = 0.70
    damping_count: int = 11
    control_scale_min: float = 0.80
    control_scale_max: float = 1.20
    control_scale_count: int = 9
    outlier_variance_multiplier: float = 100.0
    huber_delta: float = 1.5
    state_variance_floor: float = 1e-8
    bias_process_variance: float = 1.0e-5
    bias_initial_variance: float = 1.0e-7
    bias_cue_persistence: float = 0.85
    bias_cue_threshold: float = 0.20
    bias_minimum_run_length: int = 5
    bias_activation_offset: float = 0.05
    bias_activation_scale: float = 0.45


@dataclass(frozen=True)
class SyntheticObservations:
    observed: np.ndarray
    confidence: np.ndarray
    occluded: np.ndarray
    boundary_distance: np.ndarray
    flow_inconsistency: np.ndarray
    inlier_target: np.ndarray
    corruption_type: np.ndarray


def _validate_config(config: SyntheticBenchmarkConfig) -> None:
    if config.node_count < 3:
        raise ValueError("node_count must be at least 3")
    if not 3 <= config.train_step_count < config.step_count:
        raise ValueError("train_step_count must be at least 3 and below step_count")
    if config.time_step <= 0.0 or config.observation_std <= 0.0:
        raise ValueError("time_step and observation_std must be positive")
    for name in ("stiffness_count", "damping_count", "control_scale_count"):
        if getattr(config, name) < 2:
            raise ValueError(f"{name} must be at least 2")
    for lower_name, upper_name in (
        ("stiffness_min", "stiffness_max"),
        ("damping_min", "damping_max"),
        ("control_scale_min", "control_scale_max"),
    ):
        if getattr(config, lower_name) >= getattr(config, upper_name):
            raise ValueError(f"{lower_name} must be below {upper_name}")
    if config.outlier_variance_multiplier <= 1.0:
        raise ValueError("outlier_variance_multiplier must be greater than 1")
    if config.huber_delta <= 0.0 or config.state_variance_floor <= 0.0:
        raise ValueError("huber_delta and state_variance_floor must be positive")
    if config.bias_process_variance < 0.0 or config.bias_initial_variance < 0.0:
        raise ValueError("bias variances must be nonnegative")
    if not 0.0 <= config.bias_cue_persistence < 1.0:
        raise ValueError("bias_cue_persistence must be in [0, 1)")
    if not 0.0 <= config.bias_cue_threshold <= 1.0:
        raise ValueError("bias_cue_threshold must be in [0, 1]")
    if config.bias_minimum_run_length < 1:
        raise ValueError("bias_minimum_run_length must be positive")
    if config.bias_activation_scale <= 0.0:
        raise ValueError("bias_activation_scale must be positive")


def spring_graph_laplacian(node_count: int) -> np.ndarray:
    """Return a path-graph Laplacian with the first free node grounded."""

    if node_count < 1:
        raise ValueError("node_count must be positive")
    laplacian = np.zeros((node_count, node_count), dtype=float)
    laplacian[0, 0] += 1.0
    for left in range(node_count - 1):
        right = left + 1
        laplacian[left, left] += 1.0
        laplacian[right, right] += 1.0
        laplacian[left, right] -= 1.0
        laplacian[right, left] -= 1.0
    return laplacian


def make_action(config: SyntheticBenchmarkConfig, mode: str) -> np.ndarray:
    """Create known quasi-static or dynamically informative control inputs."""

    time = np.arange(config.step_count, dtype=float) * config.time_step
    if mode == "dynamic":
        action = (
            0.75 * np.sin(2.0 * np.pi * 0.75 * time)
            + 0.45 * np.sin(2.0 * np.pi * 1.65 * time + 0.4)
            + 0.20 * np.sign(np.sin(2.0 * np.pi * 0.32 * time))
        )
    elif mode == "quasi_static":
        phase = np.linspace(0.0, 1.0, config.step_count)
        action = np.piecewise(
            phase,
            [phase < 0.35, (phase >= 0.35) & (phase < 0.70), phase >= 0.70],
            [
                lambda value: 1.5 * value,
                0.525,
                lambda value: 0.525 - 1.2 * (value - 0.70),
            ],
        )
    else:
        raise ValueError(f"unknown action mode {mode!r}")
    return np.asarray(action, dtype=float)


def simulate_parameter_particles(
    particles: np.ndarray,
    action: np.ndarray,
    config: SyntheticBenchmarkConfig,
) -> np.ndarray:
    """Simulate all parameter particles through the fixed spring graph."""

    particles = np.asarray(particles, dtype=float)
    action = np.asarray(action, dtype=float)
    if particles.ndim != 2 or particles.shape[1] != 3:
        raise ValueError("particles must have shape (p, 3)")
    if action.shape != (config.step_count,):
        raise ValueError(f"action must have shape ({config.step_count},)")

    particle_count = particles.shape[0]
    state = np.zeros((particle_count, config.node_count), dtype=float)
    velocity = np.zeros_like(state)
    trajectory = np.empty(
        (particle_count, config.step_count, config.node_count),
        dtype=float,
    )
    laplacian = spring_graph_laplacian(config.node_count)
    actuator = np.zeros(config.node_count, dtype=float)
    actuator[-1] = 1.0
    if config.node_count >= 4:
        actuator[config.node_count // 2] = -0.35

    stiffness = particles[:, 0, None]
    damping = particles[:, 1, None]
    control_scale = particles[:, 2, None]
    for step in range(config.step_count):
        trajectory[:, step] = state
        spring_force = -(state @ laplacian.T) * stiffness
        damping_force = -(velocity @ laplacian.T) * damping - 0.03 * velocity
        control_force = control_scale * action[step] * actuator[None, :]
        acceleration = spring_force + damping_force + control_force
        velocity = velocity + config.time_step * acceleration
        state = state + config.time_step * velocity
    return trajectory


def parameter_grid(config: SyntheticBenchmarkConfig) -> np.ndarray:
    axes = (
        np.linspace(config.stiffness_min, config.stiffness_max, config.stiffness_count),
        np.linspace(config.damping_min, config.damping_max, config.damping_count),
        np.linspace(
            config.control_scale_min,
            config.control_scale_max,
            config.control_scale_count,
        ),
    )
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([axis.reshape(-1) for axis in mesh])


def _window(start_fraction: float, end_fraction: float, length: int) -> slice:
    start = min(length - 1, max(0, int(round(start_fraction * length))))
    end = min(length, max(start + 1, int(round(end_fraction * length))))
    return slice(start, end)


def generate_observations(
    true_trajectory: np.ndarray,
    *,
    condition: str,
    seed: int,
    config: SyntheticBenchmarkConfig,
) -> SyntheticObservations:
    """Inject clean, independent, or temporally correlated corruption."""

    rng = np.random.default_rng(seed)
    shape = true_trajectory.shape
    observed = true_trajectory + rng.normal(scale=config.observation_std, size=shape)
    confidence = np.clip(rng.normal(loc=0.94, scale=0.025, size=shape), 0.80, 0.99)
    occluded = np.zeros(shape, dtype=bool)
    boundary_distance = np.full(shape, 0.08, dtype=float)
    flow_inconsistency = np.full(shape, 0.003, dtype=float)
    inlier_target = np.ones(shape, dtype=bool)
    corruption_type = np.full(shape, "clean", dtype="<U16")

    def corrupt(
        selection: tuple[slice, int] | np.ndarray,
        name: str,
    ) -> None:
        inlier_target[selection] = False
        corruption_type[selection] = name

    if condition == "clean":
        pass
    elif condition == "iid":
        selected = rng.random(size=shape) < 0.12
        observed[selected] += rng.normal(scale=7.0 * config.observation_std, size=np.sum(selected))
        confidence[selected] = np.minimum(confidence[selected], 0.55)
        flow_inconsistency[selected] = 0.10
        inlier_target[selected] = False
        corruption_type[selected] = "iid_outlier"
    elif condition == "correlated":
        train = config.train_step_count

        occlusion_window = _window(0.22, 0.43, train)
        occlusion_node = min(1, config.node_count - 1)
        count = occlusion_window.stop - occlusion_window.start
        observed[occlusion_window, occlusion_node] += (
            0.035 + rng.normal(scale=0.025, size=count)
        )
        confidence[occlusion_window, occlusion_node] = 0.12
        occluded[occlusion_window, occlusion_node] = True
        boundary_distance[occlusion_window, occlusion_node] = 0.002
        flow_inconsistency[occlusion_window, occlusion_node] = 0.16
        corrupt((occlusion_window, occlusion_node), "occlusion")

        drift_window = _window(0.35, 1.0, train)
        drift_count = drift_window.stop - drift_window.start
        drift_fraction = np.linspace(0.0, 1.0, drift_count)
        drift_bias = 0.065 * np.square(drift_fraction)
        drift_nodes = sorted(
            {
                min(2, config.node_count - 1),
                min(3, config.node_count - 1),
            }
        )
        for node_offset, drift_node in enumerate(drift_nodes):
            bias_scale = 1.0 - 0.2 * node_offset
            observed[drift_window, drift_node] += bias_scale * drift_bias
            confidence[drift_window, drift_node] = 0.90 - 0.08 * drift_fraction
            flow_inconsistency[drift_window, drift_node] = 0.03
            corrupt((drift_window, drift_node), "drift")

        boundary_window = _window(0.08, 0.24, train)
        boundary_node = config.node_count - 1
        count = boundary_window.stop - boundary_window.start
        observed[boundary_window, boundary_node] += rng.normal(
            scale=3.0 * config.observation_std,
            size=count,
        )
        confidence[boundary_window, boundary_node] = 0.68
        boundary_distance[boundary_window, boundary_node] = 0.0015
        corrupt((boundary_window, boundary_node), "boundary")

        flow_window = _window(0.66, 0.84, train)
        flow_node = 0
        count = flow_window.stop - flow_window.start
        observed[flow_window, flow_node] += 0.025 * np.sin(np.linspace(0.0, np.pi, count))
        confidence[flow_window, flow_node] = 0.62
        flow_inconsistency[flow_window, flow_node] = 0.22
        corrupt((flow_window, flow_node), "flow")
    else:
        raise ValueError(f"unknown corruption condition {condition!r}")

    return SyntheticObservations(
        observed=observed,
        confidence=confidence,
        occluded=occluded,
        boundary_distance=boundary_distance,
        flow_inconsistency=flow_inconsistency,
        inlier_target=inlier_target,
        corruption_type=corruption_type,
    )


def _log_densities(
    residual: np.ndarray,
    variance: float,
    outlier_variance_multiplier: float,
) -> tuple[np.ndarray, np.ndarray]:
    inlier = -0.5 * (np.log(2.0 * np.pi * variance) + np.square(residual) / variance)
    outlier_variance = variance * outlier_variance_multiplier
    outlier = -0.5 * (
        np.log(2.0 * np.pi * outlier_variance)
        + np.square(residual) / outlier_variance
    )
    return inlier, outlier


def _temporally_smoothed_bias_probability(
    raw_probability: np.ndarray,
    *,
    step_count: int,
    node_count: int,
    persistence: float = 0.85,
    cue_threshold: float = 0.2,
    minimum_run_length: int = 5,
    activation_offset: float = 0.05,
    activation_scale: float = 0.45,
) -> np.ndarray:
    """Suppress isolated drift cues while retaining persistent track evidence."""

    raw = np.asarray(raw_probability, dtype=float).reshape(step_count, node_count)
    forward = np.zeros_like(raw)
    backward = np.zeros_like(raw)
    for node in range(node_count):
        state = 0.0
        for step in range(step_count):
            state = persistence * state + (1.0 - persistence) * raw[step, node]
            forward[step, node] = state
        state = 0.0
        for step in range(step_count - 1, -1, -1):
            state = persistence * state + (1.0 - persistence) * raw[step, node]
            backward[step, node] = state
    smoothed = 0.5 * (forward + backward)
    probability = np.clip(
        (smoothed - activation_offset) / activation_scale,
        0.0,
        1.0,
    )
    for node in range(node_count):
        longest_run = 0
        current_run = 0
        for active in raw[:, node] >= cue_threshold:
            current_run = current_run + 1 if active else 0
            longest_run = max(longest_run, current_run)
        if longest_run < minimum_run_length:
            probability[:, node] = 0.0
    return probability.reshape(-1)


def _normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    shifted = log_weights - np.max(log_weights)
    weights = np.exp(shifted)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise FloatingPointError("parameter posterior has invalid weights")
    return weights / total


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    return float(np.interp(probability, cumulative, sorted_values))


def _weighted_crps(values: np.ndarray, weights: np.ndarray, target: float) -> float:
    order = np.argsort(values, kind="mergesort")
    values = values[order]
    weights = weights[order]
    first_term = float(np.sum(weights * np.abs(values - target)))
    cumulative_weight = 0.0
    cumulative_value = 0.0
    pairwise_half = 0.0
    for value, weight in zip(values, weights):
        pairwise_half += weight * (value * cumulative_weight - cumulative_value)
        cumulative_weight += float(weight)
        cumulative_value += float(weight * value)
    return first_term - pairwise_half


def _posterior_metrics(
    particles: np.ndarray,
    trajectories: np.ndarray,
    weights: np.ndarray,
    true_parameters: np.ndarray,
    true_trajectory: np.ndarray,
    config: SyntheticBenchmarkConfig,
) -> dict[str, Any]:
    mean = np.average(particles, axis=0, weights=weights)
    variance = np.average(np.square(particles - mean), axis=0, weights=weights)
    parameter_metrics: dict[str, Any] = {}
    for index, name in enumerate(PARAMETER_NAMES):
        lower = _weighted_quantile(particles[:, index], weights, 0.05)
        upper = _weighted_quantile(particles[:, index], weights, 0.95)
        truth = float(true_parameters[index])
        parameter_metrics[name] = {
            "mean": float(mean[index]),
            "std": float(np.sqrt(variance[index])),
            "absolute_error": abs(float(mean[index]) - truth),
            "credible_interval_90": [lower, upper],
            "covered_90": bool(lower <= truth <= upper),
            "crps": _weighted_crps(particles[:, index], weights, truth),
        }

    state_mean = np.average(trajectories, axis=0, weights=weights)
    state_variance = np.average(
        np.square(trajectories - state_mean[None, :, :]),
        axis=0,
        weights=weights,
    )
    state_error = state_mean - true_trajectory
    state_std = np.sqrt(np.maximum(state_variance, config.state_variance_floor))
    train = slice(1, config.train_step_count)
    future = slice(config.train_step_count, config.step_count)
    z90 = 1.6448536269514722

    return {
        "posterior_effective_sample_size": float(1.0 / np.sum(np.square(weights))),
        "parameters": parameter_metrics,
        "state": {
            "train_rmse": float(np.sqrt(np.mean(np.square(state_error[train])))),
            "future_rmse": float(np.sqrt(np.mean(np.square(state_error[future])))),
            "nees": float(
                np.mean(
                    np.square(state_error[1:])
                    / np.maximum(state_variance[1:], config.state_variance_floor)
                )
            ),
            "gaussian_coverage_90": float(
                np.mean(np.abs(state_error[1:]) <= z90 * state_std[1:])
            ),
        },
    }


def _reliability_metrics(
    observations: SyntheticObservations,
    truth: np.ndarray,
    config: SyntheticBenchmarkConfig,
    reliability_config: ReliabilityConfig,
    markov_config: MarkovReliabilityConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_count = config.train_step_count
    observed = observations.observed[:train_count].reshape(-1)
    predicted = truth[:train_count].reshape(-1)
    target = observations.inlier_target[:train_count].reshape(-1)
    batch = PseudoMeasurementBatch(
        observed=observed[:, None],
        predicted=predicted[:, None],
        variance=config.observation_std**2,
        confidence=observations.confidence[:train_count].reshape(-1),
        occluded=observations.occluded[:train_count].reshape(-1),
        boundary_distance=observations.boundary_distance[:train_count].reshape(-1),
        flow_inconsistency=observations.flow_inconsistency[:train_count].reshape(-1),
    )
    prior = score_reliability(batch, reliability_config).weights
    log_inlier, log_outlier = _log_densities(
        observed - predicted,
        config.observation_std**2,
        config.outlier_variance_multiplier,
    )
    log_iid_inlier = np.log(prior) + log_inlier
    log_iid_outlier = np.log1p(-prior) + log_outlier
    iid = np.exp(log_iid_inlier - np.logaddexp(log_iid_inlier, log_iid_outlier))
    sequence_ids = np.tile(np.arange(config.node_count), train_count)
    time_values = np.repeat(np.arange(train_count), config.node_count)
    markov = smooth_markov_reliability(
        prior,
        log_inlier,
        log_outlier,
        sequence_ids,
        time_values,
        config=markov_config,
    ).posterior_inlier_probability

    return prior, {
        "inlier_rate": float(np.mean(target)),
        "prior": binary_calibration_metrics(prior, target).as_dict(),
        "iid_posterior": binary_calibration_metrics(iid, target).as_dict(),
        "markov_posterior": binary_calibration_metrics(markov, target).as_dict(),
    }


def run_synthetic_case(
    *,
    seed: int,
    condition: str,
    action_mode: str,
    config: SyntheticBenchmarkConfig,
    particles: np.ndarray | None = None,
    particle_trajectories: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run all inference baselines for one synthetic replicate."""

    _validate_config(config)
    particles = parameter_grid(config) if particles is None else np.asarray(particles, dtype=float)
    action = make_action(config, action_mode)
    if particle_trajectories is None:
        particle_trajectories = simulate_parameter_particles(particles, action, config)
    true_parameters = np.array(
        [config.true_stiffness, config.true_damping, config.true_control_scale],
        dtype=float,
    )
    true_trajectory = simulate_parameter_particles(
        true_parameters[None, :],
        action,
        config,
    )[0]
    observations = generate_observations(
        true_trajectory,
        condition=condition,
        seed=seed,
        config=config,
    )

    reliability_config = ReliabilityConfig(
        boundary_scale=0.02,
        flow_scale=0.08,
        occlusion_weight=0.03,
    )
    markov_config = MarkovReliabilityConfig(
        inlier_persistence=0.985,
        outlier_persistence=0.94,
    )
    prior, reliability_metrics = _reliability_metrics(
        observations,
        true_trajectory,
        config,
        reliability_config,
        markov_config,
    )

    train = config.train_step_count
    observed_flat = observations.observed[:train].reshape(-1)
    predicted_flat = particle_trajectories[:, :train].reshape(particles.shape[0], -1)
    residual = observed_flat[None, :] - predicted_flat
    variance = config.observation_std**2
    log_inlier, log_outlier = _log_densities(
        residual,
        variance,
        config.outlier_variance_multiplier,
    )
    target = observations.inlier_target[:train].reshape(-1)
    flow = observations.flow_inconsistency[:train].reshape(-1)
    occluded = observations.occluded[:train].reshape(-1)
    boundary = observations.boundary_distance[:train].reshape(-1)
    raw_bias_probability = np.clip((flow - 0.005) / 0.08, 0.0, 1.0)
    raw_bias_probability *= (~occluded).astype(float)
    raw_bias_probability *= np.clip(boundary / 0.01, 0.0, 1.0)
    bias_probability = _temporally_smoothed_bias_probability(
        raw_bias_probability,
        step_count=train,
        node_count=config.node_count,
        persistence=config.bias_cue_persistence,
        cue_threshold=config.bias_cue_threshold,
        minimum_run_length=config.bias_minimum_run_length,
        activation_offset=config.bias_activation_offset,
        activation_scale=config.bias_activation_scale,
    )
    sequence_ids = np.tile(np.arange(config.node_count), train)
    time_values = np.repeat(np.arange(train), config.node_count)

    standardized = residual / config.observation_std
    absolute = np.abs(standardized)
    huber = np.where(
        absolute <= config.huber_delta,
        0.5 * np.square(standardized),
        config.huber_delta * (absolute - 0.5 * config.huber_delta),
    )
    iid_log_likelihood = np.sum(
        np.logaddexp(
            np.log(prior)[None, :] + log_inlier,
            np.log1p(-prior)[None, :] + log_outlier,
        ),
        axis=1,
    )
    oracle_prior = np.where(target, 1.0 - 1e-6, 1e-6)
    method_log_likelihoods = {
        "unweighted_gaussian": np.sum(log_inlier, axis=1),
        "huber": -np.sum(huber, axis=1),
        "cue_weighted": -0.5 * np.sum(prior[None, :] * np.square(standardized), axis=1),
        "iid_mixture": iid_log_likelihood,
        "markov_mixture": markov_log_evidence_batch(
            prior,
            log_inlier,
            log_outlier,
            sequence_ids,
            time_values,
            config=markov_config,
        ),
        "structured_bias_filter": robust_random_walk_log_evidence_batch(
            prior,
            residual,
            variance,
            sequence_ids,
            time_values,
            config=RandomWalkBiasConfig(
                process_variance=config.bias_process_variance,
                initial_variance=config.bias_initial_variance,
                outlier_variance_multiplier=config.outlier_variance_multiplier,
            ),
            bias_probability=bias_probability,
        ),
        "oracle_covariates": np.sum(
            np.logaddexp(
                np.log(oracle_prior)[None, :] + log_inlier,
                np.log1p(-oracle_prior)[None, :] + log_outlier,
            ),
            axis=1,
        ),
        "oracle_inliers": np.sum(log_inlier[:, target], axis=1),
    }

    methods: dict[str, Any] = {}
    for method in METHODS:
        weights = _normalize_log_weights(method_log_likelihoods[method])
        methods[method] = _posterior_metrics(
            particles,
            particle_trajectories,
            weights,
            true_parameters,
            true_trajectory,
            config,
        )

    unique_corruption, counts = np.unique(
        observations.corruption_type[:train],
        return_counts=True,
    )
    return {
        "seed": seed,
        "condition": condition,
        "action_mode": action_mode,
        "corruption_counts": {
            str(name): int(count) for name, count in zip(unique_corruption, counts)
        },
        "reliability": reliability_metrics,
        "methods": methods,
    }


def _aggregate_runs(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for run in runs:
        for method, metrics in run["methods"].items():
            groups.setdefault((run["condition"], run["action_mode"], method), []).append(metrics)

    aggregate: list[dict[str, Any]] = []
    for (condition, action_mode, method), metrics_list in groups.items():
        row: dict[str, Any] = {
            "condition": condition,
            "action_mode": action_mode,
            "method": method,
            "replicate_count": len(metrics_list),
            "state_train_rmse_mean": float(
                np.mean([metrics["state"]["train_rmse"] for metrics in metrics_list])
            ),
            "state_future_rmse_mean": float(
                np.mean([metrics["state"]["future_rmse"] for metrics in metrics_list])
            ),
            "state_nees_mean": float(
                np.mean([metrics["state"]["nees"] for metrics in metrics_list])
            ),
            "state_coverage_90_mean": float(
                np.mean(
                    [metrics["state"]["gaussian_coverage_90"] for metrics in metrics_list]
                )
            ),
        }
        for parameter in PARAMETER_NAMES:
            row[f"{parameter}_mae"] = float(
                np.mean(
                    [
                        metrics["parameters"][parameter]["absolute_error"]
                        for metrics in metrics_list
                    ]
                )
            )
            row[f"{parameter}_coverage_90"] = float(
                np.mean(
                    [
                        metrics["parameters"][parameter]["covered_90"]
                        for metrics in metrics_list
                    ]
                )
            )
            row[f"{parameter}_crps_mean"] = float(
                np.mean(
                    [metrics["parameters"][parameter]["crps"] for metrics in metrics_list]
                )
            )
        aggregate.append(row)
    return aggregate


def _aggregate_reliability(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for run in runs:
        for estimator in ("prior", "iid_posterior", "markov_posterior"):
            groups.setdefault(
                (run["condition"], run["action_mode"], estimator),
                [],
            ).append(run["reliability"][estimator])

    aggregate: list[dict[str, Any]] = []
    for (condition, action_mode, estimator), metrics_list in groups.items():
        auc_values = [
            metrics["roc_auc"]
            for metrics in metrics_list
            if metrics["roc_auc"] is not None
        ]
        aggregate.append(
            {
                "condition": condition,
                "action_mode": action_mode,
                "estimator": estimator,
                "replicate_count": len(metrics_list),
                "brier_score_mean": float(
                    np.mean([metrics["brier_score"] for metrics in metrics_list])
                ),
                "log_loss_mean": float(
                    np.mean([metrics["log_loss"] for metrics in metrics_list])
                ),
                "expected_calibration_error_mean": float(
                    np.mean(
                        [metrics["expected_calibration_error"] for metrics in metrics_list]
                    )
                ),
                "roc_auc_mean": None if not auc_values else float(np.mean(auc_values)),
            }
        )
    return aggregate


def run_synthetic_benchmark(
    *,
    seeds: Sequence[int],
    conditions: Sequence[str],
    action_modes: Sequence[str],
    config: SyntheticBenchmarkConfig | None = None,
) -> dict[str, Any]:
    """Run a reproducible benchmark grid and return serializable results."""

    cfg = config or SyntheticBenchmarkConfig()
    _validate_config(cfg)
    if not seeds:
        raise ValueError("at least one seed is required")
    particles = parameter_grid(cfg)
    trajectories_by_action = {
        action_mode: simulate_parameter_particles(
            particles,
            make_action(cfg, action_mode),
            cfg,
        )
        for action_mode in action_modes
    }
    runs = [
        run_synthetic_case(
            seed=int(seed),
            condition=condition,
            action_mode=action_mode,
            config=cfg,
            particles=particles,
            particle_trajectories=trajectories_by_action[action_mode],
        )
        for action_mode in action_modes
        for condition in conditions
        for seed in seeds
    ]
    return {
        "schema_version": 2,
        "config": asdict(cfg),
        "parameter_grid_size": int(particles.shape[0]),
        "seeds": [int(seed) for seed in seeds],
        "conditions": list(conditions),
        "action_modes": list(action_modes),
        "runs": runs,
        "aggregate": _aggregate_runs(runs),
        "reliability_aggregate": _aggregate_reliability(runs),
    }


def write_benchmark_json(result: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_benchmark_csv(result: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = result["aggregate"]
    if not rows:
        raise ValueError("benchmark result has no aggregate rows")
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_reliability_csv(result: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = result["reliability_aggregate"]
    if not rows:
        raise ValueError("benchmark result has no reliability aggregate rows")
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
