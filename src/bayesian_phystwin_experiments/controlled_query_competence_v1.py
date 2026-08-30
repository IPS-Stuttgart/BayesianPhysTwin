"""Controlled query-conditional simulator competence experiment.

This source-only instrument uses a deliberately misspecified spring-graph world.
A causal screen is observed first.  A query-aware selector then chooses one of
four contact-model simulators, and a separately trained risk model decides
whether that candidate may replace the exact nominal fallback for one future
action, horizon, and query.  Future truth is simulated only after every
pre-outcome feature and candidate identity has been fixed.

The independent statistical unit is one seeded episode with one preassigned
query.  The experiment is controlled mechanism evidence, not physical evidence
and not a backend or state-of-the-art claim.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.guard_harm_risk import one_sided_binomial_upper_bound

SCHEMA: Final = "bayesian-phystwin.controlled-query-competence"
SCHEMA_VERSION: Final = 1
SOURCE_BUNDLE_SHA256: Final = (
    "6e1dc500f0f982827005d216d36c30e46051d61a8a80d100042c00d0ed5aa738"
)
CLAIM_BOUNDARY: Final = (
    "Controlled in-population spring-graph mechanism evidence only. The "
    "independent unit is one seeded episode with one preassigned query. A "
    "passing result does not establish unseen-topology transfer, physical "
    "provider competence, universal simulator validity, deployment safety, "
    "official benchmark superiority, or state of the art."
)

FRAME_COUNT: Final = 56
SCREEN_FRAMES: Final = (2, 4, 6, 8, 10)
OBSERVATION_NOISE: Final = 0.0015
TRUE_HYPOTHESIS_PRIOR: Final = np.asarray((0.35, 0.25, 0.25, 0.15))
HORIZONS: Final = (14, 28, 55)
QUERY_NAMES: Final = (
    "global_endpoint",
    "sensor_path",
    "centroid_endpoint",
    "peak_edge_strain",
)
ACTION_NAMES: Final = (
    "passive",
    "left_lift",
    "right_drag",
    "centre_pulse",
    "dual_stretch",
    "reverse_sweep",
    "diagonal_hook",
)
TOPOLOGY_NAMES: Final = ("rope", "cloth", "soft_block")
HYPOTHESIS_NAMES: Final = (
    "nominal",
    "shifted",
    "compliant_slip",
    "shifted_slip",
)

HARM_MARGIN: Final = 0.025
TARGET_HARM_PROBABILITY: Final = 0.10
CONFIDENCE_LEVEL: Final = 0.95
MINIMUM_ACCEPTED_COUNT: Final = 100
MINIMUM_ACCEPTED_PER_QUERY: Final = 20
MINIMUM_USEFUL_COVERAGE: Final = 0.25
THRESHOLD_GRID: Final = (0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.33, 0.50)
LOGISTIC_L2_PENALTY: Final = 1.0
LOGISTIC_MAX_ITERATIONS: Final = 80
LOGISTIC_TOLERANCE: Final = 1e-9

CALIBRATION_COUNT_PER_TOPOLOGY: Final = 96
RISK_TRAIN_COUNT_PER_TOPOLOGY: Final = 192
THRESHOLD_SELECTION_COUNT_PER_TOPOLOGY: Final = 384
SOURCE_GATE_COUNT_PER_TOPOLOGY: Final = 384
CONFIRMATION_COUNT_PER_TOPOLOGY: Final = 512

CALIBRATION_SEED_BASE: Final = 202608100
RISK_TRAIN_SEED_BASE: Final = 202608400
THRESHOLD_SELECTION_SEED_BASE: Final = 202608700
SOURCE_GATE_SEED_BASE: Final = 202608950
CONFIRMATION_SEED_BASE: Final = 202609300
QUERY_SEED_OFFSET: Final = 100_000
BOOTSTRAP_REPLICATES: Final = 2_000
SOURCE_BOOTSTRAP_SEED: Final = 20261001
CONFIRMATION_BOOTSTRAP_SEED: Final = 20261002


@dataclass(frozen=True, slots=True)
class GraphObjectV1:
    name: str
    rest_positions: np.ndarray
    edges: tuple[tuple[int, int], ...]
    mass: float
    support: float
    nominal_parameters: np.ndarray
    sensor_nodes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ActionV1:
    name: str
    nodes: tuple[int, ...]
    forces: np.ndarray


@dataclass(frozen=True, slots=True)
class ContactHypothesisV1:
    name: str
    shifted_attachment: bool
    gain: float
    delay_steps: int
    neighbour_spread: float
    rotation_radians: float


@dataclass(frozen=True, slots=True)
class ScreenCalibrationV1:
    prior: np.ndarray
    residual_variance: np.ndarray

    def to_record(self) -> dict[str, object]:
        return {
            "prior": self.prior.tolist(),
            "residual_variance": self.residual_variance.tolist(),
        }


@dataclass(frozen=True, slots=True)
class QueryOutcomeV1:
    group_id: str
    partition: str
    topology: str
    action: str
    horizon_step_count: int
    query_name: str
    true_hypothesis_index: int
    candidate_model_index: int
    feature_vector: np.ndarray
    model_losses: np.ndarray
    candidate_loss: float
    fallback_loss: float
    harmful_candidate: bool

    @property
    def candidate_available(self) -> bool:
        return self.candidate_model_index != 0

    @property
    def regret(self) -> float:
        return self.candidate_loss - self.fallback_loss


def _grid_edges(rows: int, columns: int) -> tuple[tuple[int, int], ...]:
    edges: list[tuple[int, int]] = []
    for row in range(rows):
        for column in range(columns):
            node = row * columns + column
            if column + 1 < columns:
                edges.append((node, node + 1))
            if row + 1 < rows:
                edges.append((node, node + columns))
            if row + 1 < rows and column + 1 < columns:
                edges.extend(((node, node + columns + 1), (node + 1, node + columns)))
    return tuple(edges)


def build_objects_v1() -> tuple[GraphObjectV1, ...]:
    rope_positions = np.column_stack((np.linspace(-0.30, 0.30, 7), np.zeros(7)))
    cloth_positions = np.asarray(
        [
            (column * 0.12 - 0.12, row * 0.12 - 0.12)
            for row in range(3)
            for column in range(3)
        ],
        dtype=np.float64,
    )
    block_positions = np.asarray(
        [
            (column * 0.11 - 0.165, row * 0.13 - 0.065)
            for row in range(2)
            for column in range(4)
        ],
        dtype=np.float64,
    )
    return (
        GraphObjectV1(
            "rope",
            rope_positions,
            tuple((index, index + 1) for index in range(6)),
            0.82,
            0.50,
            np.asarray((8.5, 0.62, 0.92)),
            (0, 3, 6),
        ),
        GraphObjectV1(
            "cloth",
            cloth_positions,
            _grid_edges(3, 3),
            1.08,
            0.42,
            np.asarray((6.4, 0.84, 0.78)),
            (0, 4, 8),
        ),
        GraphObjectV1(
            "soft_block",
            block_positions,
            _grid_edges(2, 4),
            1.34,
            0.68,
            np.asarray((10.6, 1.08, 1.06)),
            (0, 3, 6),
        ),
    )


def build_hypotheses_v1() -> tuple[ContactHypothesisV1, ...]:
    return (
        ContactHypothesisV1("nominal", False, 1.00, 0, 0.00, 0.0),
        ContactHypothesisV1("shifted", True, 0.72, 2, 0.00, math.radians(8.0)),
        ContactHypothesisV1("compliant_slip", False, 0.78, 1, 0.20, math.radians(8.0)),
        ContactHypothesisV1("shifted_slip", True, 0.88, 1, 0.20, math.radians(-8.0)),
    )


def _envelope(name: str) -> np.ndarray:
    phase = np.linspace(0.0, 1.0, FRAME_COUNT - 1)
    if name == "smooth":
        return np.sin(np.pi * phase) ** 2
    if name == "hold":
        return np.clip(phase / 0.2, 0.0, 1.0) * np.clip((1.0 - phase) / 0.2, 0.0, 1.0)
    if name == "double":
        return np.sin(2.0 * np.pi * phase) ** 2
    if name == "impulse":
        return np.exp(-0.5 * ((phase - 0.22) / 0.09) ** 2)
    if name == "passive":
        return np.zeros(FRAME_COUNT - 1)
    raise ValueError(f"unknown force envelope: {name}")


def _force_schedule(
    vectors: tuple[tuple[float, float], ...],
    envelope: str,
    *,
    rotation: float = 0.0,
) -> np.ndarray:
    result = _envelope(envelope)[:, None, None] * np.asarray(vectors)[None, :, :]
    if rotation != 0.0:
        phase = np.linspace(0.0, 1.0, FRAME_COUNT - 1)
        angles = rotation * (phase - 0.5)
        cosine = np.cos(angles)
        sine = np.sin(angles)
        x_values = result[..., 0].copy()
        y_values = result[..., 1].copy()
        result[..., 0] = cosine[:, None] * x_values - sine[:, None] * y_values
        result[..., 1] = sine[:, None] * x_values + cosine[:, None] * y_values
    return result


def build_actions_v1(graph_object: GraphObjectV1) -> tuple[ActionV1, ...]:
    positions = graph_object.rest_positions
    center = positions.mean(axis=0)
    left = int(np.argmin(positions[:, 0]))
    right = int(np.argmax(positions[:, 0]))
    middle = int(np.argmin(np.linalg.norm(positions - center, axis=1)))
    upper = int(np.argmax(positions[:, 1] + 0.05 * positions[:, 0]))
    return (
        ActionV1("passive", (middle,), _force_schedule(((0.0, 0.0),), "passive")),
        ActionV1("left_lift", (left,), _force_schedule(((0.08, 0.48),), "smooth")),
        ActionV1("right_drag", (right,), _force_schedule(((0.43, 0.10),), "hold")),
        ActionV1(
            "centre_pulse", (middle,), _force_schedule(((-0.08, -0.36),), "double")
        ),
        ActionV1(
            "dual_stretch",
            (left, right),
            _force_schedule(((-0.31, 0.12), (0.31, 0.12)), "smooth"),
        ),
        ActionV1(
            "reverse_sweep",
            (upper,),
            _force_schedule(((-0.34, 0.27),), "hold", rotation=0.8),
        ),
        ActionV1(
            "diagonal_hook",
            (right,),
            _force_schedule(((-0.30, 0.52),), "impulse", rotation=-0.5),
        ),
    )


def _laplacian(graph_object: GraphObjectV1) -> np.ndarray:
    count = len(graph_object.rest_positions)
    result = np.zeros((count, count))
    for first, second in graph_object.edges:
        result[first, first] += 1.0
        result[second, second] += 1.0
        result[first, second] -= 1.0
        result[second, first] -= 1.0
    return result


def _adjacency(graph_object: GraphObjectV1) -> tuple[tuple[int, ...], ...]:
    result: list[list[int]] = [[] for _ in graph_object.rest_positions]
    for first, second in graph_object.edges:
        result[first].append(second)
        result[second].append(first)
    return tuple(tuple(sorted(values)) for values in result)


def _characteristic_length(graph_object: GraphObjectV1) -> float:
    return float(
        np.median(
            [
                np.linalg.norm(
                    graph_object.rest_positions[first]
                    - graph_object.rest_positions[second]
                )
                for first, second in graph_object.edges
            ]
        )
    )


def _resolved_nodes(
    graph_object: GraphObjectV1,
    nodes: tuple[int, ...],
    hypothesis: ContactHypothesisV1,
) -> tuple[int, ...]:
    if not hypothesis.shifted_attachment:
        return nodes
    adjacency = _adjacency(graph_object)
    occupied = set(nodes)
    resolved: list[int] = []
    for node in nodes:
        candidates = sorted(
            adjacency[node],
            key=lambda candidate: (
                candidate in occupied,
                np.linalg.norm(
                    graph_object.rest_positions[candidate]
                    - graph_object.rest_positions[node]
                ),
                candidate,
            ),
        )
        resolved.append(candidates[0] if candidates else node)
    return tuple(resolved)


def simulate_trajectory_v1(
    graph_object: GraphObjectV1,
    action: ActionV1,
    parameters: np.ndarray,
    hypothesis: ContactHypothesisV1,
    *,
    nonlinearity: float,
    time_step: float = 0.03,
    drag: float = 0.18,
) -> np.ndarray:
    """Integrate one deterministic spring-graph trajectory."""

    laplacian = _laplacian(graph_object)
    adjacency = _adjacency(graph_object)
    displacement = np.zeros_like(graph_object.rest_positions)
    velocity = np.zeros_like(displacement)
    trajectory = np.empty((FRAME_COUNT, len(displacement), 2))
    trajectory[0] = graph_object.rest_positions
    nodes = _resolved_nodes(graph_object, action.nodes, hypothesis)
    edge_array = np.asarray(graph_object.edges, dtype=np.int64)
    first = edge_array[:, 0]
    second = edge_array[:, 1]
    characteristic = _characteristic_length(graph_object)
    nonlinear_coefficient = nonlinearity * parameters[0] / characteristic**2
    angle = hypothesis.rotation_radians
    rotation = np.asarray(
        ((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle)))
    )

    for frame in range(1, FRAME_COUNT):
        external = np.zeros_like(displacement)
        action_index = frame - 1 - hypothesis.delay_steps
        if action_index >= 0:
            scheduled = parameters[2] * hypothesis.gain * action.forces[action_index]
            scheduled = scheduled @ rotation.T
            for local_index, node in enumerate(nodes):
                force = scheduled[local_index]
                neighbours = adjacency[node]
                if hypothesis.neighbour_spread > 0.0 and neighbours:
                    external[node] += (1.0 - hypothesis.neighbour_spread) * force
                    share = hypothesis.neighbour_spread / len(neighbours)
                    external[np.asarray(neighbours)] += share * force
                else:
                    external[node] += force
        total_force = (
            external
            - parameters[0] * (laplacian @ displacement)
            - parameters[1] * (laplacian @ velocity)
            - drag * velocity
            - graph_object.support * displacement
        )
        if nonlinearity > 0.0:
            relative = displacement[second] - displacement[first]
            edge_force = (
                nonlinear_coefficient
                * np.sum(relative * relative, axis=1)[:, None]
                * relative
            )
            node_force = np.zeros_like(displacement)
            np.add.at(node_force, first, edge_force)
            np.add.at(node_force, second, -edge_force)
            total_force += node_force
        velocity += time_step * total_force / graph_object.mass
        displacement += time_step * velocity
        trajectory[frame] = graph_object.rest_positions + displacement
    if not np.all(np.isfinite(trajectory)):
        raise ValueError("spring-graph trajectory is non-finite")
    return trajectory


def _screen_action(graph_object: GraphObjectV1) -> ActionV1:
    base = build_actions_v1(graph_object)[3]
    return ActionV1("screen", base.nodes, 0.35 * base.forces)


def _screen_feature(graph_object: GraphObjectV1, trajectory: np.ndarray) -> np.ndarray:
    sensors = np.asarray(graph_object.sensor_nodes)
    values = (
        trajectory[np.asarray(SCREEN_FRAMES)][:, sensors]
        - graph_object.rest_positions[sensors][None, :, :]
    )
    return values.reshape(-1)


def _peak_edge_strain(
    graph_object: GraphObjectV1,
    trajectory: np.ndarray,
    horizon: int,
) -> float:
    rest_lengths = np.asarray(
        [
            np.linalg.norm(
                graph_object.rest_positions[first] - graph_object.rest_positions[second]
            )
            for first, second in graph_object.edges
        ]
    )
    maximum = 0.0
    for state in trajectory[: horizon + 1]:
        lengths = np.asarray(
            [
                np.linalg.norm(state[first] - state[second])
                for first, second in graph_object.edges
            ]
        )
        maximum = max(
            maximum, float(np.max(np.abs(lengths - rest_lengths) / rest_lengths))
        )
    return maximum


def query_value_v1(
    graph_object: GraphObjectV1,
    trajectory: np.ndarray,
    horizon: int,
    query_name: str,
) -> np.ndarray:
    """Return one dimensionless query value."""

    if horizon not in HORIZONS:
        raise ValueError("query horizon is not registered")
    characteristic = _characteristic_length(graph_object)
    if query_name == "global_endpoint":
        return (
            (trajectory[horizon] - graph_object.rest_positions) / characteristic
        ).reshape(-1)
    if query_name == "sensor_path":
        frames = np.unique(
            np.clip(
                np.asarray((horizon // 4, horizon // 2, 3 * horizon // 4, horizon)),
                1,
                horizon,
            )
        )
        sensors = np.asarray(graph_object.sensor_nodes)
        return (
            (
                trajectory[frames][:, sensors]
                - graph_object.rest_positions[sensors][None]
            )
            / characteristic
        ).reshape(-1)
    if query_name == "centroid_endpoint":
        return (
            np.mean(trajectory[horizon] - graph_object.rest_positions, axis=0)
            / characteristic
        )
    if query_name == "peak_edge_strain":
        return np.asarray((_peak_edge_strain(graph_object, trajectory, horizon),))
    raise ValueError("query functional is not registered")


def _query_loss(first: np.ndarray, second: np.ndarray) -> float:
    difference = np.asarray(first) - np.asarray(second)
    return float(np.sqrt(np.mean(difference * difference)))


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + math.log(float(np.sum(np.exp(values - maximum))))


def screen_posterior_v1(
    observed: np.ndarray,
    model_features: np.ndarray,
    calibration: ScreenCalibrationV1,
) -> np.ndarray:
    variance = calibration.residual_variance
    log_likelihood = -0.5 * (
        np.sum((observed[None] - model_features) ** 2 / variance[None], axis=1)
        + np.sum(np.log(2.0 * np.pi * variance))
    )
    log_weights = np.log(np.clip(calibration.prior, 1e-300, None)) + log_likelihood
    log_weights -= _logsumexp(log_weights)
    return np.exp(log_weights)


def _entropy(weights: np.ndarray) -> float:
    clipped = np.clip(weights, 1e-300, 1.0)
    return float(-np.sum(clipped * np.log(clipped)))


def _model_trajectories(
    graph_object: GraphObjectV1,
    hypotheses: tuple[ContactHypothesisV1, ...],
) -> tuple[tuple[np.ndarray, ...], ...]:
    return tuple(
        tuple(
            simulate_trajectory_v1(
                graph_object,
                action,
                graph_object.nominal_parameters,
                hypothesis,
                nonlinearity=0.0,
            )
            for hypothesis in hypotheses
        )
        for action in build_actions_v1(graph_object)
    )


def fit_screen_calibration_v1(
    *,
    count_per_topology: int = CALIBRATION_COUNT_PER_TOPOLOGY,
    seed_base: int = CALIBRATION_SEED_BASE,
) -> ScreenCalibrationV1:
    objects = build_objects_v1()
    hypotheses = build_hypotheses_v1()
    counts = np.ones(len(hypotheses))
    residuals: list[np.ndarray] = []
    for topology_index, graph_object in enumerate(objects):
        model_features = np.stack(
            [
                _screen_feature(
                    graph_object,
                    simulate_trajectory_v1(
                        graph_object,
                        _screen_action(graph_object),
                        graph_object.nominal_parameters,
                        hypothesis,
                        nonlinearity=0.0,
                    ),
                )
                for hypothesis in hypotheses
            ]
        )
        generator = np.random.default_rng(seed_base + topology_index)
        for _ in range(count_per_topology):
            true_index = int(generator.choice(len(hypotheses), p=TRUE_HYPOTHESIS_PRIOR))
            parameters = graph_object.nominal_parameters * np.exp(
                generator.normal(0.0, (0.08, 0.10, 0.06))
            )
            nonlinearity = max(0.0, float(generator.normal(0.18, 0.025)))
            truth = simulate_trajectory_v1(
                graph_object,
                _screen_action(graph_object),
                parameters,
                hypotheses[true_index],
                nonlinearity=nonlinearity,
            )
            observed = _screen_feature(graph_object, truth) + generator.normal(
                0.0, OBSERVATION_NOISE, size=model_features.shape[1]
            )
            counts[true_index] += 1.0
            residuals.append(observed - model_features[true_index])
    variance = np.maximum(
        np.var(np.stack(residuals), axis=0, ddof=1),
        OBSERVATION_NOISE**2,
    )
    return ScreenCalibrationV1(prior=counts / counts.sum(), residual_variance=variance)


def feature_names_v1() -> tuple[str, ...]:
    names = [f"posterior_{name}" for name in HYPOTHESIS_NAMES]
    names.extend(
        (
            "predicted_query_regret",
            "predicted_harm_probability",
            "normalized_posterior_entropy",
            "one_minus_maximum_posterior",
            "candidate_fallback_disagreement",
            "normalized_horizon",
            "candidate_expected_squared_loss",
            "fallback_expected_squared_loss",
        )
    )
    names.extend(f"action_{name}" for name in ACTION_NAMES)
    names.extend(f"query_{name}" for name in QUERY_NAMES)
    names.extend(f"candidate_{name}" for name in HYPOTHESIS_NAMES)
    names.extend(f"topology_{name}" for name in TOPOLOGY_NAMES)
    names.extend(
        (
            "harm_probability_times_disagreement",
            "uncertainty_times_disagreement",
        )
    )
    return tuple(names)


def preoutcome_route_v1(
    *,
    topology_index: int,
    action_index: int,
    horizon: int,
    query_index: int,
    posterior: np.ndarray,
    query_outputs: tuple[np.ndarray, ...],
) -> tuple[int, np.ndarray]:
    """Fix the candidate and risk features without accepting future truth."""

    if posterior.shape != (len(HYPOTHESIS_NAMES),):
        raise ValueError("posterior shape changed")
    pairwise = np.asarray(
        [
            [_query_loss(first, second) for second in query_outputs]
            for first in query_outputs
        ]
    )
    expected_squared = np.sum(posterior[None, :] * pairwise**2, axis=1)
    candidate_index = int(np.argmin(expected_squared))
    fallback_index = 0
    candidate_fallback = float(pairwise[candidate_index, fallback_index])
    predicted_harm = float(
        posterior @ (pairwise[candidate_index] > pairwise[fallback_index] + HARM_MARGIN)
    )
    predicted_regret = float(
        posterior @ (pairwise[candidate_index] - pairwise[fallback_index])
    )
    uncertainty = 1.0 - float(np.max(posterior))
    feature = np.concatenate(
        (
            posterior,
            np.asarray(
                (
                    predicted_regret,
                    predicted_harm,
                    _entropy(posterior) / math.log(len(posterior)),
                    uncertainty,
                    candidate_fallback,
                    horizon / max(HORIZONS),
                    expected_squared[candidate_index],
                    expected_squared[fallback_index],
                )
            ),
            np.eye(len(ACTION_NAMES))[action_index],
            np.eye(len(QUERY_NAMES))[query_index],
            np.eye(len(HYPOTHESIS_NAMES))[candidate_index],
            np.eye(len(TOPOLOGY_NAMES))[topology_index],
            np.asarray(
                (
                    predicted_harm * candidate_fallback,
                    uncertainty * candidate_fallback,
                )
            ),
        )
    )
    if feature.shape != (len(feature_names_v1()),) or not np.all(np.isfinite(feature)):
        raise ValueError("pre-outcome risk feature contract changed")
    return candidate_index, feature


def generate_partition_v1(
    *,
    partition: str,
    count_per_topology: int,
    seed_base: int,
    calibration: ScreenCalibrationV1,
) -> tuple[QueryOutcomeV1, ...]:
    """Generate one independent partition with one preassigned query per episode."""

    if not partition or partition.strip() != partition:
        raise ValueError("partition must be a canonical string")
    objects = build_objects_v1()
    hypotheses = build_hypotheses_v1()
    outcomes: list[QueryOutcomeV1] = []
    for topology_index, graph_object in enumerate(objects):
        actions = build_actions_v1(graph_object)
        model_trajectories = _model_trajectories(graph_object, hypotheses)
        screen_models = np.stack(
            [
                _screen_feature(
                    graph_object,
                    simulate_trajectory_v1(
                        graph_object,
                        _screen_action(graph_object),
                        graph_object.nominal_parameters,
                        hypothesis,
                        nonlinearity=0.0,
                    ),
                )
                for hypothesis in hypotheses
            ]
        )
        truth_generator = np.random.default_rng(seed_base + topology_index)
        query_generator = np.random.default_rng(
            seed_base + QUERY_SEED_OFFSET + topology_index
        )
        for episode_index in range(count_per_topology):
            true_index = int(
                truth_generator.choice(len(hypotheses), p=TRUE_HYPOTHESIS_PRIOR)
            )
            parameters = graph_object.nominal_parameters * np.exp(
                truth_generator.normal(0.0, (0.08, 0.10, 0.06))
            )
            nonlinearity = max(
                0.0,
                float(truth_generator.normal(0.18, 0.025)),
            )
            screen_truth = simulate_trajectory_v1(
                graph_object,
                _screen_action(graph_object),
                parameters,
                hypotheses[true_index],
                nonlinearity=nonlinearity,
            )
            screen_observed = _screen_feature(
                graph_object, screen_truth
            ) + truth_generator.normal(
                0.0, OBSERVATION_NOISE, size=screen_models.shape[1]
            )
            posterior = screen_posterior_v1(
                screen_observed,
                screen_models,
                calibration,
            )

            action_index = int(query_generator.integers(len(actions)))
            horizon = HORIZONS[int(query_generator.integers(len(HORIZONS)))]
            query_index = int(query_generator.integers(len(QUERY_NAMES)))
            query_name = QUERY_NAMES[query_index]
            model_outputs = tuple(
                query_value_v1(
                    graph_object,
                    trajectory,
                    horizon,
                    query_name,
                )
                for trajectory in model_trajectories[action_index]
            )
            candidate_index, feature = preoutcome_route_v1(
                topology_index=topology_index,
                action_index=action_index,
                horizon=horizon,
                query_index=query_index,
                posterior=posterior,
                query_outputs=model_outputs,
            )

            # Future truth is opened only after the route and feature are fixed.
            future_truth = simulate_trajectory_v1(
                graph_object,
                actions[action_index],
                parameters,
                hypotheses[true_index],
                nonlinearity=nonlinearity,
            )
            truth_output = query_value_v1(
                graph_object,
                future_truth,
                horizon,
                query_name,
            )
            model_losses = np.asarray(
                [_query_loss(output, truth_output) for output in model_outputs]
            )
            candidate_loss = float(model_losses[candidate_index])
            fallback_loss = float(model_losses[0])
            group_id = f"{partition}:{graph_object.name}:{episode_index:06d}"
            outcomes.append(
                QueryOutcomeV1(
                    group_id=group_id,
                    partition=partition,
                    topology=graph_object.name,
                    action=actions[action_index].name,
                    horizon_step_count=horizon,
                    query_name=query_name,
                    true_hypothesis_index=true_index,
                    candidate_model_index=candidate_index,
                    feature_vector=feature,
                    model_losses=model_losses,
                    candidate_loss=candidate_loss,
                    fallback_loss=fallback_loss,
                    harmful_candidate=(candidate_loss > fallback_loss + HARM_MARGIN),
                )
            )
    group_ids = [outcome.group_id for outcome in outcomes]
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("partition group IDs are not unique")
    return tuple(outcomes)


def _array_sha256(value: np.ndarray) -> str:
    canonical = np.ascontiguousarray(value, dtype=np.dtype("<f8"))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


@dataclass(frozen=True, slots=True)
class FrozenLogisticRiskModelV1:
    model_name: str
    selected_feature_names: tuple[str, ...]
    feature_center: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    l2_penalty: float
    iteration_count: int
    converged: bool
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        if not self.model_name or self.model_name.strip() != self.model_name:
            raise ValueError("risk model name is invalid")
        if len(set(self.selected_feature_names)) != len(self.selected_feature_names):
            raise ValueError("risk model feature names are not unique")
        known = set(feature_names_v1())
        if (
            not self.selected_feature_names
            or not set(self.selected_feature_names) <= known
        ):
            raise ValueError("risk model feature set changed")
        count = len(self.selected_feature_names)
        center = np.asarray(self.feature_center, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        if center.shape != (count,) or scale.shape != (count,):
            raise ValueError("risk model normalization shape changed")
        if coefficients.shape != (count + 1,):
            raise ValueError("risk model coefficient shape changed")
        if not (
            np.all(np.isfinite(center))
            and np.all(np.isfinite(scale))
            and np.all(np.isfinite(coefficients))
            and np.all(scale > 0.0)
        ):
            raise ValueError("risk model contains invalid values")
        immutable_center = center.copy()
        immutable_scale = scale.copy()
        immutable_coefficients = coefficients.copy()
        immutable_center.setflags(write=False)
        immutable_scale.setflags(write=False)
        immutable_coefficients.setflags(write=False)
        object.__setattr__(self, "feature_center", immutable_center)
        object.__setattr__(self, "feature_scale", immutable_scale)
        object.__setattr__(self, "coefficients", immutable_coefficients)
        expected = content_id(self.descriptor())
        if self.artifact_id is not None and self.artifact_id != expected:
            raise ValueError("risk model artifact identity changed")
        object.__setattr__(self, "artifact_id", expected)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": "bayesian-phystwin.controlled-query-risk-model",
            "schema_version": 1,
            "model_name": self.model_name,
            "selected_feature_names": list(self.selected_feature_names),
            "feature_center": self.feature_center.tolist(),
            "feature_center_sha256": _array_sha256(self.feature_center),
            "feature_scale": self.feature_scale.tolist(),
            "feature_scale_sha256": _array_sha256(self.feature_scale),
            "coefficients": self.coefficients.tolist(),
            "coefficients_sha256": _array_sha256(self.coefficients),
            "l2_penalty": self.l2_penalty,
            "iteration_count": self.iteration_count,
            "converged": self.converged,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> FrozenLogisticRiskModelV1:
        expected = {
            "schema",
            "schema_version",
            "model_name",
            "selected_feature_names",
            "feature_center",
            "feature_center_sha256",
            "feature_scale",
            "feature_scale_sha256",
            "coefficients",
            "coefficients_sha256",
            "l2_penalty",
            "iteration_count",
            "converged",
            "artifact_id",
        }
        if set(record) != expected:
            raise ValueError("risk model record fields changed")
        if record["schema"] != "bayesian-phystwin.controlled-query-risk-model":
            raise ValueError("risk model record schema changed")
        if record["schema_version"] != 1:
            raise ValueError("risk model record version changed")
        if type(record["converged"]) is not bool:
            raise ValueError("risk model convergence flag changed")
        model = cls(
            model_name=str(record["model_name"]),
            selected_feature_names=tuple(record["selected_feature_names"]),
            feature_center=np.asarray(record["feature_center"], dtype=np.float64),
            feature_scale=np.asarray(record["feature_scale"], dtype=np.float64),
            coefficients=np.asarray(record["coefficients"], dtype=np.float64),
            l2_penalty=float(record["l2_penalty"]),
            iteration_count=int(record["iteration_count"]),
            converged=bool(record["converged"]),
            artifact_id=str(record["artifact_id"]),
        )
        hash_fields = (
            ("feature_center_sha256", model.feature_center),
            ("feature_scale_sha256", model.feature_scale),
            ("coefficients_sha256", model.coefficients),
        )
        for name, value in hash_fields:
            if record[name] != _array_sha256(value):
                raise ValueError(f"risk model {name} changed")
        return model

    def score(self, feature_vector: np.ndarray) -> float:
        all_names = feature_names_v1()
        index = {name: position for position, name in enumerate(all_names)}
        selected = np.asarray(
            [feature_vector[index[name]] for name in self.selected_feature_names]
        )
        standardized = (selected - self.feature_center) / self.feature_scale
        linear = float(self.coefficients[0] + standardized @ self.coefficients[1:])
        return float(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, linear)))))


def fit_logistic_risk_model_v1(
    outcomes: tuple[QueryOutcomeV1, ...],
    *,
    model_name: str,
    selected_feature_names: tuple[str, ...],
    l2_penalty: float = LOGISTIC_L2_PENALTY,
) -> FrozenLogisticRiskModelV1:
    """Fit a deterministic L2 logistic harm model on source outcomes only."""

    eligible = [outcome for outcome in outcomes if outcome.candidate_available]
    if len(eligible) < MINIMUM_ACCEPTED_COUNT:
        raise ValueError("too few source candidate outcomes for risk fitting")
    all_names = feature_names_v1()
    positions = [all_names.index(name) for name in selected_feature_names]
    features = np.stack([outcome.feature_vector[positions] for outcome in eligible])
    labels = np.asarray(
        [outcome.harmful_candidate for outcome in eligible], dtype=np.float64
    )
    if len(np.unique(labels)) != 2:
        raise ValueError("source risk labels must contain both classes")
    center = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale[scale < 1e-8] = 1.0
    design = np.column_stack((np.ones(len(features)), (features - center) / scale))
    coefficients = np.zeros(design.shape[1])
    converged = False
    iteration_count = 0
    penalty = l2_penalty * np.diag(
        np.concatenate((np.zeros(1), np.ones(design.shape[1] - 1)))
    )
    for iteration in range(LOGISTIC_MAX_ITERATIONS):
        linear = np.clip(design @ coefficients, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-linear))
        weights = probability * (1.0 - probability)
        gradient = design.T @ (probability - labels) + penalty @ coefficients
        hessian = design.T @ (weights[:, None] * design) + penalty
        try:
            step = np.linalg.solve(
                hessian + np.eye(hessian.shape[0]) * 1e-9,
                gradient,
            )
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        coefficients -= step
        iteration_count = iteration + 1
        if float(np.max(np.abs(step))) < LOGISTIC_TOLERANCE:
            converged = True
            break
    if not converged:
        raise ValueError("source logistic risk fit did not converge")
    return FrozenLogisticRiskModelV1(
        model_name=model_name,
        selected_feature_names=selected_feature_names,
        feature_center=center,
        feature_scale=scale,
        coefficients=coefficients,
        l2_penalty=l2_penalty,
        iteration_count=iteration_count,
        converged=converged,
    )


def risk_model_feature_sets_v1() -> dict[str, tuple[str, ...]]:
    names = feature_names_v1()
    return {
        "full_query_conditional": names,
        "context_agnostic": tuple(
            name
            for name in names
            if name.startswith("posterior_")
            or name
            in {
                "normalized_posterior_entropy",
                "one_minus_maximum_posterior",
            }
        ),
        "uncertainty_only": (
            "normalized_posterior_entropy",
            "one_minus_maximum_posterior",
        ),
        "model_disagreement_only": (
            "predicted_query_regret",
            "predicted_harm_probability",
            "candidate_fallback_disagreement",
            "candidate_expected_squared_loss",
            "fallback_expected_squared_loss",
        ),
    }


def _bootstrap_mean_interval(
    values: np.ndarray,
    *,
    seed: int,
) -> dict[str, float]:
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("bootstrap values are invalid")
    generator = np.random.default_rng(seed)
    estimates = np.empty(BOOTSTRAP_REPLICATES)
    for start in range(0, BOOTSTRAP_REPLICATES, 200):
        batch = min(200, BOOTSTRAP_REPLICATES - start)
        indices = generator.integers(0, len(values), size=(batch, len(values)))
        estimates[start : start + batch] = np.mean(values[indices], axis=1)
    lower, upper = np.quantile(estimates, (0.025, 0.975))
    return {
        "mean": float(np.mean(values)),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
    }


def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    positive = scores[labels]
    negative = scores[~labels]
    if len(positive) == 0 or len(negative) == 0:
        return None
    comparisons = positive[:, None] - negative[None, :]
    return float(np.mean(comparisons > 0.0) + 0.5 * np.mean(comparisons == 0.0))


def _stratum_summary(
    outcomes: tuple[QueryOutcomeV1, ...],
    accepted: np.ndarray,
) -> dict[str, list[dict[str, object]]]:
    fields = {
        "topology": lambda item: item.topology,
        "action": lambda item: item.action,
        "horizon": lambda item: str(item.horizon_step_count),
        "query": lambda item: item.query_name,
    }
    result: dict[str, list[dict[str, object]]] = {}
    for field, accessor in fields.items():
        rows: list[dict[str, object]] = []
        for value in sorted({accessor(outcome) for outcome in outcomes}):
            mask = np.asarray([accessor(outcome) == value for outcome in outcomes])
            selected = mask & accepted
            harmful = np.asarray([outcome.harmful_candidate for outcome in outcomes])
            regrets = np.asarray([outcome.regret for outcome in outcomes])
            rows.append(
                {
                    "value": value,
                    "query_count": int(np.sum(mask)),
                    "accepted_count": int(np.sum(selected)),
                    "coverage": float(np.mean(accepted[mask])),
                    "harmful_accepted_count": int(np.sum(selected & harmful)),
                    "mean_selected_regret": float(
                        np.mean(np.where(accepted[mask], regrets[mask], 0.0))
                    ),
                }
            )
        result[field] = rows
    return result


def evaluate_selective_policy_v1(
    outcomes: tuple[QueryOutcomeV1, ...],
    scores: np.ndarray,
    threshold: float,
    *,
    bootstrap_seed: int,
) -> dict[str, object]:
    if scores.shape != (len(outcomes),) or not np.all(np.isfinite(scores)):
        raise ValueError("risk score vector changed")
    candidate_available = np.asarray(
        [outcome.candidate_available for outcome in outcomes]
    )
    accepted = candidate_available & (scores <= threshold)
    harmful = np.asarray([outcome.harmful_candidate for outcome in outcomes])
    regrets = np.asarray([outcome.regret for outcome in outcomes])
    selected_regrets = np.where(accepted, regrets, 0.0)
    accepted_count = int(np.sum(accepted))
    harmful_count = int(np.sum(accepted & harmful))
    per_query = {
        query: int(
            np.sum(
                accepted
                & np.asarray([outcome.query_name == query for outcome in outcomes])
            )
        )
        for query in QUERY_NAMES
    }
    gate_checks = {
        "minimum_accepted_count": accepted_count >= MINIMUM_ACCEPTED_COUNT,
        "minimum_useful_coverage": float(np.mean(accepted)) >= MINIMUM_USEFUL_COVERAGE,
        "minimum_accepted_per_query": min(per_query.values())
        >= MINIMUM_ACCEPTED_PER_QUERY,
        "harm_upper_bound": False,
        "selected_loss_improves_fallback": False,
        "exact_fallback_identity": True,
    }
    upper = one_sided_binomial_upper_bound(
        harmful_count,
        accepted_count,
        CONFIDENCE_LEVEL,
    )
    interval = _bootstrap_mean_interval(selected_regrets, seed=bootstrap_seed)
    gate_checks["harm_upper_bound"] = upper <= TARGET_HARM_PROBABILITY
    gate_checks["selected_loss_improves_fallback"] = interval["ci95_upper"] < 0.0
    available_scores = scores[candidate_available]
    available_labels = harmful[candidate_available]
    return {
        "query_count": len(outcomes),
        "candidate_available_count": int(np.sum(candidate_available)),
        "accepted_count": accepted_count,
        "coverage": float(np.mean(accepted)),
        "harmful_accepted_count": harmful_count,
        "harmful_accepted_rate": (
            None if accepted_count == 0 else harmful_count / accepted_count
        ),
        "exact_one_sided_95_harm_upper_bound": upper,
        "target_harm_probability": TARGET_HARM_PROBABILITY,
        "harm_margin": HARM_MARGIN,
        "selected_regret": interval,
        "accepted_mean_regret": (
            None if accepted_count == 0 else float(np.mean(regrets[accepted]))
        ),
        "accepted_per_query": per_query,
        "risk_brier_score": float(
            np.mean((available_scores - available_labels.astype(float)) ** 2)
        ),
        "risk_auc": _binary_auc(available_scores, available_labels),
        "gate_checks": gate_checks,
        "gate_passed": all(gate_checks.values()),
        "strata": _stratum_summary(outcomes, accepted),
    }


def select_threshold_v1(
    outcomes: tuple[QueryOutcomeV1, ...],
    scores: np.ndarray,
) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    selected: dict[str, object] | None = None
    for threshold in THRESHOLD_GRID:
        evaluation = evaluate_selective_policy_v1(
            outcomes,
            scores,
            threshold,
            bootstrap_seed=SOURCE_BOOTSTRAP_SEED,
        )
        checks = dict(evaluation["gate_checks"])
        # Threshold selection does not use its own bootstrap interval as a gate.
        checks["selected_loss_improves_fallback"] = (
            float(evaluation["selected_regret"]["mean"]) < 0.0
        )
        eligible = all(checks.values())
        candidate = {
            "threshold": threshold,
            "accepted_count": evaluation["accepted_count"],
            "coverage": evaluation["coverage"],
            "harmful_accepted_count": evaluation["harmful_accepted_count"],
            "harm_upper_bound": evaluation["exact_one_sided_95_harm_upper_bound"],
            "mean_selected_regret": evaluation["selected_regret"]["mean"],
            "accepted_per_query": evaluation["accepted_per_query"],
            "eligible": eligible,
        }
        candidates.append(candidate)
        if eligible:
            selected = candidate
    return {
        "threshold_grid": list(THRESHOLD_GRID),
        "candidate_summaries": candidates,
        "selected_threshold": None if selected is None else selected["threshold"],
        "selection_passed": selected is not None,
    }


def _model_scores(
    model: FrozenLogisticRiskModelV1,
    outcomes: tuple[QueryOutcomeV1, ...],
) -> np.ndarray:
    return np.asarray([model.score(outcome.feature_vector) for outcome in outcomes])


def _fixed_model_summary(
    outcomes: tuple[QueryOutcomeV1, ...],
    model_index: int,
) -> dict[str, object]:
    regrets = np.asarray(
        [
            outcome.model_losses[model_index] - outcome.fallback_loss
            for outcome in outcomes
        ]
    )
    harmful = regrets > HARM_MARGIN
    return {
        "model_index": model_index,
        "model_name": HYPOTHESIS_NAMES[model_index],
        "mean_regret": float(np.mean(regrets)),
        "harmful_count": int(np.sum(harmful)),
        "harmful_rate": float(np.mean(harmful)),
    }


def experiment_protocol_v1() -> dict[str, object]:
    protocol = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "source_bundle_sha256": SOURCE_BUNDLE_SHA256,
        "claim_boundary": CLAIM_BOUNDARY,
        "statistical_unit": "one independently seeded episode with one preassigned query",
        "information_boundary": (
            "screen observation, query context, model forecasts, candidate identity, "
            "and risk features are fixed before future truth is simulated"
        ),
        "topologies": list(TOPOLOGY_NAMES),
        "actions": list(ACTION_NAMES),
        "horizons": list(HORIZONS),
        "queries": list(QUERY_NAMES),
        "hypotheses": list(HYPOTHESIS_NAMES),
        "fallback_model_index": 0,
        "candidate_selection": "minimum posterior expected squared query disagreement",
        "loss": "dimensionless root-mean-square query error",
        "harm_margin": HARM_MARGIN,
        "risk_models": {
            name: list(features)
            for name, features in risk_model_feature_sets_v1().items()
        },
        "risk_fit": {
            "family": "deterministic L2 logistic regression",
            "l2_penalty": LOGISTIC_L2_PENALTY,
            "maximum_iterations": LOGISTIC_MAX_ITERATIONS,
            "tolerance": LOGISTIC_TOLERANCE,
        },
        "threshold_grid": list(THRESHOLD_GRID),
        "target_harm_probability": TARGET_HARM_PROBABILITY,
        "confidence_level": CONFIDENCE_LEVEL,
        "minimum_accepted_count": MINIMUM_ACCEPTED_COUNT,
        "minimum_accepted_per_query": MINIMUM_ACCEPTED_PER_QUERY,
        "minimum_useful_coverage": MINIMUM_USEFUL_COVERAGE,
        "partitions": {
            "screen_calibration": {
                "seed_base": CALIBRATION_SEED_BASE,
                "count_per_topology": CALIBRATION_COUNT_PER_TOPOLOGY,
            },
            "risk_training": {
                "seed_base": RISK_TRAIN_SEED_BASE,
                "count_per_topology": RISK_TRAIN_COUNT_PER_TOPOLOGY,
            },
            "threshold_selection": {
                "seed_base": THRESHOLD_SELECTION_SEED_BASE,
                "count_per_topology": THRESHOLD_SELECTION_COUNT_PER_TOPOLOGY,
            },
            "source_gate": {
                "seed_base": SOURCE_GATE_SEED_BASE,
                "count_per_topology": SOURCE_GATE_COUNT_PER_TOPOLOGY,
            },
            "confirmation": {
                "seed_base": CONFIRMATION_SEED_BASE,
                "count_per_topology": CONFIRMATION_COUNT_PER_TOPOLOGY,
                "opened_by_source_stage": False,
            },
        },
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "source_bootstrap_seed": SOURCE_BOOTSTRAP_SEED,
        "confirmation_bootstrap_seed": CONFIRMATION_BOOTSTRAP_SEED,
        "source_gate_required_before_confirmation": True,
        "confirmation_attempt_limit": 1,
        "no_reselection_or_retry": True,
        "unseen_topology_transfer_claimed": False,
        "prob4d_used": False,
        "protected_artifacts_used": False,
    }
    return {**protocol, "protocol_id": content_id(protocol)}


def run_source_stage_v1() -> dict[str, object]:
    """Fit and independently source-check every frozen risk-model arm."""

    calibration = fit_screen_calibration_v1()
    training = generate_partition_v1(
        partition="risk-training",
        count_per_topology=RISK_TRAIN_COUNT_PER_TOPOLOGY,
        seed_base=RISK_TRAIN_SEED_BASE,
        calibration=calibration,
    )
    threshold_selection = generate_partition_v1(
        partition="threshold-selection",
        count_per_topology=THRESHOLD_SELECTION_COUNT_PER_TOPOLOGY,
        seed_base=THRESHOLD_SELECTION_SEED_BASE,
        calibration=calibration,
    )
    source_gate = generate_partition_v1(
        partition="source-gate",
        count_per_topology=SOURCE_GATE_COUNT_PER_TOPOLOGY,
        seed_base=SOURCE_GATE_SEED_BASE,
        calibration=calibration,
    )
    arms: dict[str, object] = {}
    for name, selected_features in risk_model_feature_sets_v1().items():
        model = fit_logistic_risk_model_v1(
            training,
            model_name=name,
            selected_feature_names=selected_features,
        )
        selection_scores = _model_scores(model, threshold_selection)
        selection = select_threshold_v1(threshold_selection, selection_scores)
        threshold = selection["selected_threshold"]
        source_evaluation: dict[str, object] | None = None
        if threshold is not None:
            source_evaluation = evaluate_selective_policy_v1(
                source_gate,
                _model_scores(model, source_gate),
                float(threshold),
                bootstrap_seed=SOURCE_BOOTSTRAP_SEED,
            )
        arms[name] = {
            "model": model.to_record(),
            "threshold_selection": selection,
            "source_gate": source_evaluation,
        }

    mean_model_losses = np.mean(
        np.stack([outcome.model_losses for outcome in training]), axis=0
    )
    global_best_index = int(np.argmin(mean_model_losses))
    primary = arms["full_query_conditional"]
    primary_source_gate = primary["source_gate"]
    source_gate_passed = bool(
        primary["threshold_selection"]["selection_passed"]
        and primary_source_gate is not None
        and primary_source_gate["gate_passed"]
    )
    result = {
        "schema": "bayesian-phystwin.controlled-query-competence-source-result",
        "schema_version": 1,
        "protocol": experiment_protocol_v1(),
        "calibration": calibration.to_record(),
        "risk_training_group_count": len(training),
        "threshold_selection_group_count": len(threshold_selection),
        "source_gate_group_count": len(source_gate),
        "risk_model_arms": arms,
        "global_source_best_fixed_model": _fixed_model_summary(
            training, global_best_index
        ),
        "source_gate_passed": source_gate_passed,
        "confirmation_authorized": source_gate_passed,
        "confirmation_outcomes_opened": False,
        "prob4d_used": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    return {**result, "source_result_id": content_id(result)}


def _validate_source_result(source_result: dict[str, Any]) -> None:
    supplied = source_result.get("source_result_id")
    identity = dict(source_result)
    identity.pop("source_result_id", None)
    if supplied != content_id(identity):
        raise ValueError("source result identity changed")
    if source_result.get("schema") != (
        "bayesian-phystwin.controlled-query-competence-source-result"
    ):
        raise ValueError("source result schema changed")
    if source_result.get("protocol") != experiment_protocol_v1():
        raise ValueError("source result protocol changed")
    if source_result.get("source_gate_passed") is not True:
        raise ValueError("source gate did not pass")
    if source_result.get("confirmation_authorized") is not True:
        raise ValueError("confirmation is not authorized")
    if source_result.get("confirmation_outcomes_opened") is not False:
        raise ValueError("source result already opened confirmation outcomes")


def run_confirmation_stage_v1(
    source_result: dict[str, Any],
) -> tuple[dict[str, object], tuple[QueryOutcomeV1, ...]]:
    """Run the one frozen confirmation partition after a passing source gate."""

    _validate_source_result(source_result)
    calibration_record = source_result["calibration"]
    calibration = ScreenCalibrationV1(
        prior=np.asarray(calibration_record["prior"], dtype=np.float64),
        residual_variance=np.asarray(
            calibration_record["residual_variance"], dtype=np.float64
        ),
    )
    confirmation = generate_partition_v1(
        partition="confirmation",
        count_per_topology=CONFIRMATION_COUNT_PER_TOPOLOGY,
        seed_base=CONFIRMATION_SEED_BASE,
        calibration=calibration,
    )
    arm_results: dict[str, object] = {}
    for name, arm in source_result["risk_model_arms"].items():
        model = FrozenLogisticRiskModelV1.from_record(dict(arm["model"]))
        threshold = arm["threshold_selection"]["selected_threshold"]
        if threshold is None or arm["source_gate"] is None:
            arm_results[name] = {
                "confirmation_evaluated": False,
                "reason": "source-threshold-or-gate-unavailable",
            }
            continue
        arm_results[name] = {
            "confirmation_evaluated": True,
            "model_id": model.artifact_id,
            "threshold": threshold,
            "evaluation": evaluate_selective_policy_v1(
                confirmation,
                _model_scores(model, confirmation),
                float(threshold),
                bootstrap_seed=CONFIRMATION_BOOTSTRAP_SEED,
            ),
        }

    primary = arm_results["full_query_conditional"]
    primary_passed = bool(
        primary["confirmation_evaluated"] and primary["evaluation"]["gate_passed"]
    )
    global_index = int(source_result["global_source_best_fixed_model"]["model_index"])
    always_candidate_regret = np.asarray(
        [
            outcome.regret if outcome.candidate_available else 0.0
            for outcome in confirmation
        ]
    )
    always_candidate_harm = np.asarray(
        [
            outcome.candidate_available and outcome.harmful_candidate
            for outcome in confirmation
        ]
    )
    result = {
        "schema": "bayesian-phystwin.controlled-query-competence-confirmation-result",
        "schema_version": 1,
        "protocol_id": source_result["protocol"]["protocol_id"],
        "source_result_id": source_result["source_result_id"],
        "confirmation_group_count": len(confirmation),
        "risk_model_arms": arm_results,
        "always_fallback": {
            "mean_regret": 0.0,
            "harmful_count": 0,
            "exact_fallback_identity": True,
        },
        "always_query_selector": {
            "mean_regret": float(np.mean(always_candidate_regret)),
            "harmful_count": int(np.sum(always_candidate_harm)),
            "harmful_rate_among_all_queries": float(np.mean(always_candidate_harm)),
        },
        "global_source_best_fixed_model": _fixed_model_summary(
            confirmation, global_index
        ),
        "decision": (
            "controlled-query-competence-pass"
            if primary_passed
            else "controlled-query-competence-gate-failed"
        ),
        "primary_gate_passed": primary_passed,
        "confirmation_attempt_count": 1,
        "confirmation_reselection_or_retry": False,
        "prob4d_used": False,
        "protected_artifacts_used": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    return {**result, "confirmation_result_id": content_id(result)}, confirmation


def outcome_records_v1(
    outcomes: tuple[QueryOutcomeV1, ...],
    source_result: dict[str, Any],
) -> list[dict[str, object]]:
    """Return compact confirmation rows after the one-shot result is frozen."""

    _validate_source_result(source_result)
    models = {
        name: FrozenLogisticRiskModelV1.from_record(dict(arm["model"]))
        for name, arm in source_result["risk_model_arms"].items()
    }
    records: list[dict[str, object]] = []
    for outcome in outcomes:
        record: dict[str, object] = {
            "group_id": outcome.group_id,
            "topology": outcome.topology,
            "action": outcome.action,
            "horizon_step_count": outcome.horizon_step_count,
            "query_name": outcome.query_name,
            "true_hypothesis": HYPOTHESIS_NAMES[outcome.true_hypothesis_index],
            "candidate_model": HYPOTHESIS_NAMES[outcome.candidate_model_index],
            "candidate_available": outcome.candidate_available,
            "candidate_loss": outcome.candidate_loss,
            "fallback_loss": outcome.fallback_loss,
            "regret": outcome.regret,
            "harmful_candidate": outcome.harmful_candidate,
        }
        for name, model in models.items():
            threshold = source_result["risk_model_arms"][name]["threshold_selection"][
                "selected_threshold"
            ]
            score = model.score(outcome.feature_vector)
            record[f"{name}_risk_score"] = score
            record[f"{name}_accepted"] = bool(
                threshold is not None
                and outcome.candidate_available
                and score <= float(threshold)
            )
        records.append(record)
    return records


__all__ = [
    "ACTION_NAMES",
    "CLAIM_BOUNDARY",
    "CONFIRMATION_COUNT_PER_TOPOLOGY",
    "CONFIRMATION_SEED_BASE",
    "HARM_MARGIN",
    "HORIZONS",
    "HYPOTHESIS_NAMES",
    "QUERY_NAMES",
    "SOURCE_BUNDLE_SHA256",
    "TOPOLOGY_NAMES",
    "FrozenLogisticRiskModelV1",
    "QueryOutcomeV1",
    "ScreenCalibrationV1",
    "build_actions_v1",
    "build_hypotheses_v1",
    "build_objects_v1",
    "evaluate_selective_policy_v1",
    "experiment_protocol_v1",
    "feature_names_v1",
    "fit_logistic_risk_model_v1",
    "fit_screen_calibration_v1",
    "generate_partition_v1",
    "outcome_records_v1",
    "preoutcome_route_v1",
    "query_value_v1",
    "risk_model_feature_sets_v1",
    "run_confirmation_stage_v1",
    "run_source_stage_v1",
    "screen_posterior_v1",
    "select_threshold_v1",
    "simulate_trajectory_v1",
]
