"""Leakage-safe TrackDeform3D known-action capacity smoke.

The predictor consumes only frame-zero geometry, a sparse prefix identity
panel, and released robot trajectories.  Hidden future identities live in a
separate evaluator carrier and are never accepted by the prediction API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from bayesian_phystwin.trackdeform3d_adapter import (
    deterministic_observed_identity_ids,
)

_NORMAL_90 = 1.6448536269514722


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _readonly(value: np.ndarray, *, dtype: Any = float) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class TrackDeform3DSmokeConfig:
    """Frozen hyperparameters for the public sample smoke."""

    prefix_frames: int = 60
    future_frames: int = 60
    observed_identity_fraction: float = 0.25
    pbd_iterations: int = 40
    graph_rank: int = 4
    validation_frames: int = 20
    minimum_validation_improvement: float = 0.10
    observation_std_m: float = 0.005
    coefficient_prior_std_m: float = 0.030
    minimum_action_scale_m: float = 0.005
    maximum_correction_m: float = 0.100
    nominal_coverage: float = 0.90
    constant_velocity_history: int = 6

    def __post_init__(self) -> None:
        _require(self.prefix_frames >= 3, "prefix_frames must be at least three")
        _require(self.future_frames >= 1, "future_frames must be positive")
        _require(0.0 < self.observed_identity_fraction < 1.0, "invalid fraction")
        _require(self.pbd_iterations >= 1, "pbd_iterations must be positive")
        _require(self.graph_rank >= 1, "graph_rank must be positive")
        _require(
            1 <= self.validation_frames < self.prefix_frames,
            "validation_frames must lie inside the prefix",
        )
        _require(
            0.0 <= self.minimum_validation_improvement < 1.0,
            "minimum validation improvement is invalid",
        )
        positive = (
            self.observation_std_m,
            self.coefficient_prior_std_m,
            self.minimum_action_scale_m,
            self.maximum_correction_m,
        )
        _require(all(np.isfinite(v) and v > 0.0 for v in positive), "invalid scale")
        _require(0.0 < self.nominal_coverage < 1.0, "invalid nominal coverage")
        _require(
            2 <= self.constant_velocity_history <= self.prefix_frames,
            "constant velocity history is invalid",
        )


@dataclass(frozen=True)
class TrackDeform3DPredictionInput:
    """All information available to the predictor."""

    frame_zero_points_m: np.ndarray
    edges: np.ndarray
    rest_lengths_m: np.ndarray
    end_effector_positions_m: np.ndarray
    observed_identity_ids: np.ndarray
    observed_prefix_points_m: np.ndarray

    def __post_init__(self) -> None:
        frame_zero = _readonly(self.frame_zero_points_m)
        edges = _readonly(self.edges, dtype=np.int64)
        rest = _readonly(self.rest_lengths_m)
        action = _readonly(self.end_effector_positions_m)
        observed_ids = _readonly(self.observed_identity_ids, dtype=np.int64)
        observed = _readonly(self.observed_prefix_points_m)

        node_count = len(frame_zero)
        _require(frame_zero.shape == (node_count, 3), "frame zero must be (N, 3)")
        _require(node_count >= 3, "at least three graph nodes are required")
        _require(edges.ndim == 2 and edges.shape[1] == 2, "edges must be (E, 2)")
        _require(rest.shape == (len(edges),), "rest lengths must match edges")
        _require(action.ndim == 3 and action.shape[1:] == (2, 3), "action changed")
        _require(observed_ids.ndim == 1, "observed IDs must be a vector")
        _require(len(observed_ids) >= 2, "at least two observed IDs are required")
        _require(len(np.unique(observed_ids)) == len(observed_ids), "duplicate IDs")
        _require(
            np.all((observed_ids >= 0) & (observed_ids < node_count)),
            "observed ID is outside the graph",
        )
        _require(
            observed.shape[1:] == (len(observed_ids), 3),
            "observed prefix shape changed",
        )
        _require(
            len(action) > len(observed),
            "known action must include at least one future frame",
        )
        _require(
            np.all(np.isfinite(frame_zero))
            and np.all(np.isfinite(rest))
            and np.all(np.isfinite(action))
            and np.all(np.isfinite(observed)),
            "prediction input contains non-finite values",
        )
        _require(np.all(rest > 0.0), "rest lengths must be positive")
        _require(np.all((edges >= 0) & (edges < node_count)), "edge is outside graph")

        object.__setattr__(self, "frame_zero_points_m", frame_zero)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "rest_lengths_m", rest)
        object.__setattr__(self, "end_effector_positions_m", action)
        object.__setattr__(self, "observed_identity_ids", observed_ids)
        object.__setattr__(self, "observed_prefix_points_m", observed)


@dataclass(frozen=True)
class TrackDeform3DEvaluatorTarget:
    """Hidden future identity carrier, unavailable to the predictor."""

    hidden_identity_ids: np.ndarray
    hidden_future_points_m: np.ndarray

    def __post_init__(self) -> None:
        hidden = _readonly(self.hidden_identity_ids, dtype=np.int64)
        future = _readonly(self.hidden_future_points_m)
        _require(hidden.ndim == 1 and len(hidden) > 0, "hidden IDs must be nonempty")
        _require(len(np.unique(hidden)) == len(hidden), "hidden IDs are duplicated")
        _require(future.ndim == 3, "hidden future must be (T, H, 3)")
        _require(future.shape[1:] == (len(hidden), 3), "hidden future shape changed")
        _require(np.all(np.isfinite(future)), "hidden future is not finite")
        object.__setattr__(self, "hidden_identity_ids", hidden)
        object.__setattr__(self, "hidden_future_points_m", future)


@dataclass(frozen=True)
class ActionConditionedGraphBelief:
    """Bayesian linear graph-mode discrepancy conditioned on known actions."""

    graph_basis: np.ndarray
    feature_center: np.ndarray
    feature_scale: np.ndarray
    coefficient_mean: np.ndarray
    coefficient_covariance: np.ndarray
    observation_variance_m2: float
    maximum_correction_m: float

    def __post_init__(self) -> None:
        basis = _readonly(self.graph_basis)
        center = _readonly(self.feature_center)
        scale = _readonly(self.feature_scale)
        mean = _readonly(self.coefficient_mean)
        covariance = _readonly(self.coefficient_covariance)
        feature_count = len(center) + 1
        parameter_count = basis.shape[1] * feature_count
        _require(basis.ndim == 2, "graph basis must be a matrix")
        _require(center.ndim == 1 and scale.shape == center.shape, "feature shape")
        _require(np.all(scale > 0.0), "feature scale must be positive")
        _require(mean.shape == (parameter_count, 3), "coefficient shape changed")
        _require(
            covariance.shape == (parameter_count, parameter_count),
            "coefficient covariance shape changed",
        )
        _require(
            np.isfinite(self.observation_variance_m2)
            and self.observation_variance_m2 > 0.0,
            "observation variance must be positive",
        )
        _require(
            np.isfinite(self.maximum_correction_m) and self.maximum_correction_m > 0.0,
            "maximum correction must be positive",
        )
        object.__setattr__(self, "graph_basis", basis)
        object.__setattr__(self, "feature_center", center)
        object.__setattr__(self, "feature_scale", scale)
        object.__setattr__(self, "coefficient_mean", mean)
        object.__setattr__(self, "coefficient_covariance", covariance)


@dataclass(frozen=True)
class TrackDeform3DSmokePrediction:
    """Future trajectories and the causal admission decision."""

    observed_identity_ids: np.ndarray
    persistence_m: np.ndarray
    constant_velocity_m: np.ndarray
    physical_m: np.ndarray
    guarded_bayesian_m: np.ndarray
    guarded_variance_m2: np.ndarray | None
    gate: dict[str, Any]

    def __post_init__(self) -> None:
        observed_ids = _readonly(self.observed_identity_ids, dtype=np.int64)
        trajectories = {
            name: _readonly(getattr(self, name))
            for name in (
                "persistence_m",
                "constant_velocity_m",
                "physical_m",
                "guarded_bayesian_m",
            )
        }
        shape = trajectories["physical_m"].shape
        _require(len(shape) == 3 and shape[2] == 3, "trajectory shape changed")
        _require(all(value.shape == shape for value in trajectories.values()), "shape")
        variance = self.guarded_variance_m2
        if variance is not None:
            variance = _readonly(variance)
            _require(variance.shape == shape, "variance shape changed")
            _require(np.all(variance > 0.0), "variance must be positive")
        object.__setattr__(self, "observed_identity_ids", observed_ids)
        for name, value in trajectories.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "guarded_variance_m2", variance)


def split_trackdeform3d_carriers(
    keypoint_trajectory_m: np.ndarray,
    edges: np.ndarray,
    rest_lengths_m: np.ndarray,
    end_effector_positions_m: np.ndarray,
    *,
    config: TrackDeform3DSmokeConfig,
) -> tuple[TrackDeform3DPredictionInput, TrackDeform3DEvaluatorTarget]:
    """Create disjoint predictor and hidden-future evaluator carriers."""

    trajectory = np.asarray(keypoint_trajectory_m, dtype=float)
    required = config.prefix_frames + config.future_frames
    _require(
        trajectory.ndim == 3 and trajectory.shape[2] == 3,
        "keypoint trajectory must be (T, N, 3)",
    )
    _require(len(trajectory) >= required, "keypoint trajectory is too short")
    _require(
        len(end_effector_positions_m) >= required, "action trajectory is too short"
    )
    node_count = trajectory.shape[1]
    observed_count = max(
        2,
        int(np.ceil(config.observed_identity_fraction * node_count)),
    )
    observed_count = min(observed_count, node_count - 1)
    observed_ids = deterministic_observed_identity_ids(trajectory[0], observed_count)
    hidden_ids = np.setdiff1d(np.arange(node_count), observed_ids)
    prediction_input = TrackDeform3DPredictionInput(
        frame_zero_points_m=trajectory[0],
        edges=edges,
        rest_lengths_m=rest_lengths_m,
        end_effector_positions_m=end_effector_positions_m[:required],
        observed_identity_ids=observed_ids,
        observed_prefix_points_m=trajectory[: config.prefix_frames, observed_ids],
    )
    target = TrackDeform3DEvaluatorTarget(
        hidden_identity_ids=hidden_ids,
        hidden_future_points_m=trajectory[config.prefix_frames : required, hidden_ids],
    )
    return prediction_input, target


def _chain_coordinates(
    frame_zero_m: np.ndarray,
    edges: np.ndarray,
    rest_lengths_m: np.ndarray,
    end_effector_zero_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    node_count = len(frame_zero_m)
    _require(len(edges) == node_count - 1, "physical smoke requires a tree")
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(node_count)]
    for (left, right), length in zip(edges, rest_lengths_m, strict=True):
        adjacency[int(left)].append((int(right), float(length)))
        adjacency[int(right)].append((int(left), float(length)))
    endpoints = np.asarray(
        [index for index, neighbors in enumerate(adjacency) if len(neighbors) == 1],
        dtype=np.int64,
    )
    _require(len(endpoints) == 2, "physical smoke requires one open chain")

    distance = np.full(node_count, np.inf)
    distance[int(endpoints[0])] = 0.0
    stack = [int(endpoints[0])]
    parent = np.full(node_count, -1, dtype=np.int64)
    while stack:
        node = stack.pop()
        for neighbor, length in adjacency[node]:
            if neighbor == parent[node]:
                continue
            parent[neighbor] = node
            distance[neighbor] = distance[node] + length
            stack.append(neighbor)
    _require(np.all(np.isfinite(distance)), "graph is disconnected")
    total = float(distance[int(endpoints[1])])
    _require(total > 0.0, "chain length must be positive")
    endpoint_zero = frame_zero_m[endpoints]
    assignment_cost = np.linalg.norm(
        endpoint_zero[:, None, :] - end_effector_zero_m[None, :, :],
        axis=2,
    )
    if assignment_cost[0, 0] + assignment_cost[1, 1] <= (
        assignment_cost[0, 1] + assignment_cost[1, 0]
    ):
        endpoint_to_effector = np.asarray([0, 1], dtype=np.int64)
    else:
        endpoint_to_effector = np.asarray([1, 0], dtype=np.int64)
    endpoint_zero_weight = 1.0 - distance / total
    return endpoints, endpoint_to_effector, endpoint_zero_weight


def rollout_known_action_chain(
    frame_zero_points_m: np.ndarray,
    edges: np.ndarray,
    rest_lengths_m: np.ndarray,
    end_effector_positions_m: np.ndarray,
    *,
    pbd_iterations: int,
) -> np.ndarray:
    """Transfer known endpoint actions through an inextensible open chain."""

    frame_zero = np.asarray(frame_zero_points_m, dtype=float)
    edge_array = np.asarray(edges, dtype=np.int64)
    rest = np.asarray(rest_lengths_m, dtype=float)
    action = np.asarray(end_effector_positions_m, dtype=float)
    endpoints, endpoint_to_effector, weight = _chain_coordinates(
        frame_zero,
        edge_array,
        rest,
        action[0],
    )
    fixed = np.zeros(len(frame_zero), dtype=bool)
    fixed[endpoints] = True
    trajectory = np.empty((len(action), len(frame_zero), 3), dtype=float)
    trajectory[0] = frame_zero
    for frame in range(1, len(action)):
        action_delta = action[frame] - action[frame - 1]
        endpoint_zero_delta = action_delta[int(endpoint_to_effector[0])]
        endpoint_one_delta = action_delta[int(endpoint_to_effector[1])]
        current = (
            trajectory[frame - 1]
            + weight[:, None] * endpoint_zero_delta[None]
            + (1.0 - weight)[:, None] * endpoint_one_delta[None]
        )
        for _ in range(pbd_iterations):
            for edge_index, (left, right) in enumerate(edge_array):
                difference = current[right] - current[left]
                length = float(np.linalg.norm(difference))
                if length <= 1e-15:
                    continue
                correction = (length - rest[edge_index]) * difference / length
                left_weight = 0.0 if fixed[left] else 1.0
                right_weight = 0.0 if fixed[right] else 1.0
                total_weight = left_weight + right_weight
                if total_weight > 0.0:
                    current[left] += left_weight * correction / total_weight
                    current[right] -= right_weight * correction / total_weight
        trajectory[frame] = current
    return trajectory


def _graph_basis(node_count: int, edges: np.ndarray, rank: int) -> np.ndarray:
    laplacian = np.zeros((node_count, node_count), dtype=float)
    for left, right in edges:
        laplacian[left, left] += 1.0
        laplacian[right, right] += 1.0
        laplacian[left, right] -= 1.0
        laplacian[right, left] -= 1.0
    _, eigenvectors = np.linalg.eigh(laplacian)
    return eigenvectors[:, : min(rank, node_count)]


def _raw_action_features(end_effector_positions_m: np.ndarray) -> np.ndarray:
    action = np.asarray(end_effector_positions_m, dtype=float)
    displacement = action - action[0]
    velocity = np.concatenate(
        [np.zeros((1, 2, 3)), np.diff(action, axis=0)],
        axis=0,
    )
    return np.concatenate(
        [displacement.reshape(len(action), -1), velocity.reshape(len(action), -1)],
        axis=1,
    )


def _normalized_action_features(
    raw_features: np.ndarray,
    fit_frames: np.ndarray,
    *,
    minimum_scale_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.mean(raw_features[fit_frames], axis=0)
    scale = np.maximum(np.std(raw_features[fit_frames], axis=0), minimum_scale_m)
    normalized = (raw_features - center) / scale
    return np.column_stack([np.ones(len(normalized)), normalized]), center, scale


def _design_matrix(
    features: np.ndarray,
    basis: np.ndarray,
    frames: np.ndarray,
    node_ids: np.ndarray,
) -> np.ndarray:
    selected_features = features[frames]
    selected_basis = basis[node_ids]
    design = np.einsum(
        "tf,ir->tifr",
        selected_features,
        selected_basis,
        optimize=True,
    )
    return design.reshape(len(frames) * len(node_ids), -1)


def fit_action_conditioned_graph_belief(
    physical_trajectory_m: np.ndarray,
    observed_prefix_points_m: np.ndarray,
    observed_identity_ids: np.ndarray,
    edges: np.ndarray,
    end_effector_positions_m: np.ndarray,
    fit_frames: np.ndarray,
    *,
    config: TrackDeform3DSmokeConfig,
) -> ActionConditionedGraphBelief:
    """Fit the fixed Bayesian graph-action regression on prefix evidence."""

    physical = np.asarray(physical_trajectory_m, dtype=float)
    observed = np.asarray(observed_prefix_points_m, dtype=float)
    ids = np.asarray(observed_identity_ids, dtype=np.int64)
    fit = np.asarray(fit_frames, dtype=np.int64)
    _require(fit.ndim == 1 and len(fit) > 0, "fit frames must be nonempty")
    basis = _graph_basis(
        physical.shape[1], np.asarray(edges, dtype=np.int64), config.graph_rank
    )
    raw_features = _raw_action_features(end_effector_positions_m)
    features, center, scale = _normalized_action_features(
        raw_features,
        fit,
        minimum_scale_m=config.minimum_action_scale_m,
    )
    design = _design_matrix(features, basis, fit, ids)
    target = (observed[fit] - physical[fit[:, None], ids[None, :]]).reshape(-1, 3)
    observation_variance = config.observation_std_m**2
    prior_variance = config.coefficient_prior_std_m**2
    precision = (
        design.T @ design / observation_variance
        + np.eye(design.shape[1]) / prior_variance
    )
    covariance = np.linalg.inv(precision)
    coefficient_mean = covariance @ (design.T @ target / observation_variance)
    return ActionConditionedGraphBelief(
        graph_basis=basis,
        feature_center=center,
        feature_scale=scale,
        coefficient_mean=coefficient_mean,
        coefficient_covariance=covariance,
        observation_variance_m2=observation_variance,
        maximum_correction_m=config.maximum_correction_m,
    )


def decode_action_conditioned_graph_belief(
    belief: ActionConditionedGraphBelief,
    end_effector_positions_m: np.ndarray,
    frames: np.ndarray,
    node_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode correction means and marginal variances for selected nodes."""

    raw = _raw_action_features(end_effector_positions_m)
    normalized = (raw - belief.feature_center) / belief.feature_scale
    features = np.column_stack([np.ones(len(normalized)), normalized])
    frame_ids = np.asarray(frames, dtype=np.int64)
    nodes = np.asarray(node_ids, dtype=np.int64)
    design = _design_matrix(features, belief.graph_basis, frame_ids, nodes)
    mean = design @ belief.coefficient_mean
    norm = np.linalg.norm(mean, axis=1, keepdims=True)
    mean *= np.minimum(
        1.0,
        belief.maximum_correction_m / np.maximum(norm, 1e-15),
    )
    latent_variance = np.einsum(
        "ij,jk,ik->i",
        design,
        belief.coefficient_covariance,
        design,
        optimize=True,
    )
    variance = latent_variance + belief.observation_variance_m2
    shape = (len(frame_ids), len(nodes), 3)
    return mean.reshape(shape), np.broadcast_to(
        variance.reshape(len(frame_ids), len(nodes), 1),
        shape,
    ).copy()


def _rmse_m(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.square(prediction - target), axis=-1))))


def _finite_frame_calibration_scale(
    error_m: np.ndarray,
    variance_m2: np.ndarray,
    *,
    nominal_coverage: float,
) -> tuple[float, int]:
    standardized = np.abs(error_m) / np.sqrt(variance_m2)
    frame_scores = np.max(standardized, axis=(1, 2))
    rank = min(
        len(frame_scores),
        int(np.ceil((len(frame_scores) + 1) * nominal_coverage)),
    )
    quantile = float(np.partition(frame_scores, rank - 1)[rank - 1])
    variance_scale = max(1.0, (quantile / _NORMAL_90) ** 2)
    return variance_scale, rank


def predict_trackdeform3d_smoke(
    prediction_input: TrackDeform3DPredictionInput,
    *,
    config: TrackDeform3DSmokeConfig,
) -> TrackDeform3DSmokePrediction:
    """Predict without accepting any future object trajectory."""

    prefix = config.prefix_frames
    future = config.future_frames
    total = prefix + future
    _require(
        len(prediction_input.observed_prefix_points_m) == prefix,
        "prefix length does not match config",
    )
    _require(
        len(prediction_input.end_effector_positions_m) == total,
        "action length does not match config",
    )
    physical_all = rollout_known_action_chain(
        prediction_input.frame_zero_points_m,
        prediction_input.edges,
        prediction_input.rest_lengths_m,
        prediction_input.end_effector_positions_m,
        pbd_iterations=config.pbd_iterations,
    )
    physical_future = physical_all[prefix:total]
    persistence = np.broadcast_to(
        prediction_input.frame_zero_points_m,
        physical_future.shape,
    ).copy()

    observed = prediction_input.observed_prefix_points_m
    translation_history = np.median(observed - observed[0], axis=1)
    velocity = np.median(
        np.diff(translation_history[-config.constant_velocity_history :], axis=0),
        axis=0,
    )
    horizon = np.arange(1, future + 1, dtype=float)[:, None, None]
    constant_velocity = (
        prediction_input.frame_zero_points_m[None]
        + translation_history[-1][None, None]
        + horizon * velocity[None, None]
    )

    fit_stop = prefix - config.validation_frames
    fit_frames = np.arange(fit_stop, dtype=np.int64)
    validation_frames = np.arange(fit_stop, prefix, dtype=np.int64)
    validation_belief = fit_action_conditioned_graph_belief(
        physical_all,
        observed,
        prediction_input.observed_identity_ids,
        prediction_input.edges,
        prediction_input.end_effector_positions_m,
        fit_frames,
        config=config,
    )
    validation_correction, validation_variance = decode_action_conditioned_graph_belief(
        validation_belief,
        prediction_input.end_effector_positions_m,
        validation_frames,
        prediction_input.observed_identity_ids,
    )
    validation_physical = physical_all[
        validation_frames[:, None], prediction_input.observed_identity_ids[None, :]
    ]
    validation_target = observed[validation_frames]
    validation_candidate = validation_physical + validation_correction
    baseline_rmse = _rmse_m(validation_physical, validation_target)
    candidate_rmse = _rmse_m(validation_candidate, validation_target)
    if baseline_rmse <= 1e-15:
        validation_improvement = 0.0
    else:
        validation_improvement = 1.0 - candidate_rmse / baseline_rmse
    tail_count = min(5, config.validation_frames)
    baseline_tail_rmse = _rmse_m(
        validation_physical[-tail_count:], validation_target[-tail_count:]
    )
    candidate_tail_rmse = _rmse_m(
        validation_candidate[-tail_count:], validation_target[-tail_count:]
    )
    admitted = bool(
        validation_improvement >= config.minimum_validation_improvement
        and candidate_tail_rmse <= baseline_tail_rmse
    )
    calibration_scale, calibration_rank = _finite_frame_calibration_scale(
        validation_target - validation_candidate,
        validation_variance,
        nominal_coverage=config.nominal_coverage,
    )

    gate: dict[str, Any] = {
        "admitted": admitted,
        "fit_frames": [0, fit_stop],
        "validation_frames": [fit_stop, prefix],
        "minimum_validation_improvement": config.minimum_validation_improvement,
        "validation_physical_rmse_m": baseline_rmse,
        "validation_candidate_rmse_m": candidate_rmse,
        "validation_improvement_fraction": validation_improvement,
        "validation_physical_last5_rmse_m": baseline_tail_rmse,
        "validation_candidate_last5_rmse_m": candidate_tail_rmse,
        "calibration_variance_scale": calibration_scale,
        "calibration_frame_rank": calibration_rank,
        "fallback": "bit_exact_physical",
    }
    if not admitted:
        guarded = physical_future.copy()
        _require(np.array_equal(guarded, physical_future), "fallback changed physical")
        variance = None
    else:
        final_belief = fit_action_conditioned_graph_belief(
            physical_all,
            observed,
            prediction_input.observed_identity_ids,
            prediction_input.edges,
            prediction_input.end_effector_positions_m,
            np.arange(prefix, dtype=np.int64),
            config=config,
        )
        future_frames = np.arange(prefix, total, dtype=np.int64)
        all_nodes = np.arange(len(prediction_input.frame_zero_points_m), dtype=np.int64)
        correction, variance = decode_action_conditioned_graph_belief(
            final_belief,
            prediction_input.end_effector_positions_m,
            future_frames,
            all_nodes,
        )
        variance *= calibration_scale
        guarded = physical_future + correction

    return TrackDeform3DSmokePrediction(
        observed_identity_ids=prediction_input.observed_identity_ids,
        persistence_m=persistence,
        constant_velocity_m=constant_velocity,
        physical_m=physical_future,
        guarded_bayesian_m=guarded,
        guarded_variance_m2=variance,
        gate=gate,
    )


def _symmetric_chamfer_m(prediction: np.ndarray, target: np.ndarray) -> float:
    values = []
    for predicted_frame, target_frame in zip(prediction, target, strict=True):
        distance = np.linalg.norm(
            predicted_frame[:, None, :] - target_frame[None, :, :],
            axis=2,
        )
        values.append(
            0.5
            * (
                float(np.mean(np.min(distance, axis=1)))
                + float(np.mean(np.min(distance, axis=0)))
            )
        )
    return float(np.mean(values))


def evaluate_trackdeform3d_smoke(
    prediction: TrackDeform3DSmokePrediction,
    target: TrackDeform3DEvaluatorTarget,
    *,
    nominal_coverage: float,
) -> dict[str, Any]:
    """Score only identities absent from the prediction prefix panel."""

    _require(
        not np.intersect1d(
            prediction.observed_identity_ids,
            target.hidden_identity_ids,
        ).size,
        "observed and hidden identity sets overlap",
    )
    hidden = target.hidden_identity_ids
    truth = target.hidden_future_points_m
    _require(
        prediction.physical_m.shape[0] == truth.shape[0],
        "future horizon changed",
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "hidden_identity_count": len(hidden),
        "observed_identity_count": len(prediction.observed_identity_ids),
        "gate": prediction.gate,
        "arms": {},
    }
    horizon_indices = np.array_split(np.arange(len(truth)), 3)
    for name, trajectory in (
        ("persistence", prediction.persistence_m),
        ("constant_velocity", prediction.constant_velocity_m),
        ("physical", prediction.physical_m),
        ("guarded_bayesian", prediction.guarded_bayesian_m),
    ):
        hidden_prediction = trajectory[:, hidden]
        arm = {
            "hidden_identity_rmse_m": _rmse_m(hidden_prediction, truth),
            "hidden_final_displacement_error_m": _rmse_m(
                hidden_prediction[-1:], truth[-1:]
            ),
            "hidden_symmetric_chamfer_m": _symmetric_chamfer_m(
                hidden_prediction, truth
            ),
            "horizon_resolved_identity_rmse_m": [
                _rmse_m(hidden_prediction[index], truth[index])
                for index in horizon_indices
            ],
        }
        if name == "guarded_bayesian" and prediction.guarded_variance_m2 is not None:
            variance = prediction.guarded_variance_m2[:, hidden]
            error = hidden_prediction - truth
            z_value = _NORMAL_90 if nominal_coverage == 0.90 else _NORMAL_90
            arm["marginal_coordinate_coverage"] = float(
                np.mean(np.abs(error) <= z_value * np.sqrt(variance))
            )
            arm["mean_nees"] = float(
                np.mean(np.sum(np.square(error) / variance, axis=-1))
            )
            arm["mean_interval_width_m"] = float(
                np.mean(2.0 * z_value * np.sqrt(variance))
            )
            arm["coverage_scope"] = "descriptive_correlated_point_frames"
        else:
            arm["marginal_coordinate_coverage"] = None
            arm["mean_nees"] = None
            arm["mean_interval_width_m"] = None
        result["arms"][name] = arm
    physical_rmse = result["arms"]["physical"]["hidden_identity_rmse_m"]
    guarded_rmse = result["arms"]["guarded_bayesian"]["hidden_identity_rmse_m"]
    result["guarded_improvement_vs_physical_fraction"] = (
        0.0 if physical_rmse <= 1e-15 else 1.0 - guarded_rmse / physical_rmse
    )
    result["information_boundary"] = {
        "future_object_trajectory_accepted_by_predictor": False,
        "observed_identities_scored": False,
        "known_future_robot_action_used": True,
        "upstream_tracker_is_pseudo_observation_not_ground_truth": True,
    }
    return result


__all__ = [
    "ActionConditionedGraphBelief",
    "TrackDeform3DEvaluatorTarget",
    "TrackDeform3DPredictionInput",
    "TrackDeform3DSmokeConfig",
    "TrackDeform3DSmokePrediction",
    "decode_action_conditioned_graph_belief",
    "evaluate_trackdeform3d_smoke",
    "fit_action_conditioned_graph_belief",
    "predict_trackdeform3d_smoke",
    "rollout_known_action_chain",
    "split_trackdeform3d_carriers",
]
