"""Backend-neutral Bayesian inference over finite physical rollout banks."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Sequence

import numpy as np


def _normalized_weights(values: np.ndarray, *, name: str) -> np.ndarray:
    weights = np.asarray(values, dtype=float)
    if weights.ndim != 1 or not len(weights):
        raise ValueError(f"{name} must be a nonempty vector")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError(f"{name} must have positive mass")
    return weights / total


def _normalize_joint_log_weights(log_weights: np.ndarray) -> np.ndarray:
    values = np.asarray(log_weights, dtype=float)
    if values.ndim != 2 or not np.any(np.isfinite(values)):
        raise ValueError("joint log weights must contain finite support")
    maximum = float(np.max(values[np.isfinite(values)]))
    weights = np.exp(np.where(np.isfinite(values), values - maximum, -np.inf))
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("joint posterior normalization failed")
    return weights / total


def _weighted_component_quantile(
    components: np.ndarray,
    weights: np.ndarray,
    probability: float,
) -> np.ndarray:
    order = np.argsort(components, axis=0, kind="mergesort")
    sorted_components = np.take_along_axis(components, order, axis=0)
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights, axis=0)
    indices = np.argmax(cumulative >= probability, axis=0)
    return sorted_components[indices, np.arange(components.shape[1])]


def _coordinate_mask(
    observations: np.ndarray,
    mask: np.ndarray | None,
) -> np.ndarray:
    finite = np.isfinite(observations)
    if mask is None:
        return finite
    supplied = np.asarray(mask, dtype=bool)
    if supplied.shape == observations.shape[:2]:
        supplied = np.repeat(supplied[:, :, None], observations.shape[2], axis=2)
    if supplied.shape != observations.shape:
        raise ValueError("observation mask must have shape (T, N) or (T, N, C)")
    return finite & supplied


def _student_t_mean_log_score(
    residual: np.ndarray,
    valid: np.ndarray,
    *,
    scale_m: np.ndarray | float,
    degrees_of_freedom: float,
    reduction_axes: tuple[int, ...],
) -> np.ndarray:
    if degrees_of_freedom <= 0.0:
        raise ValueError("degrees_of_freedom must be positive")
    scale = np.asarray(scale_m, dtype=float)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("likelihood scales must be finite and positive")
    standardized = residual / scale
    terms = -0.5 * (degrees_of_freedom + 1.0) * np.log1p(
        np.square(standardized) / degrees_of_freedom
    )
    valid_float = np.asarray(valid, dtype=float)
    while valid_float.ndim < terms.ndim:
        valid_float = valid_float[None]
    count = np.sum(valid_float, axis=reduction_axes)
    if np.any(count <= 0.0):
        raise ValueError("likelihood update has no valid coordinates")
    return np.sum(np.where(valid_float > 0.0, terms, 0.0), axis=reduction_axes) / count


@dataclass(frozen=True)
class PhysicalTrajectoryDistribution:
    """Mean, marginal variance, and optional empirical interval for a rollout."""

    method: str
    mean: np.ndarray
    variance: np.ndarray
    interval_lower: np.ndarray | None = None
    interval_upper: np.ndarray | None = None

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=float)
        variance = np.asarray(self.variance, dtype=float)
        if mean.ndim != 3 or mean.shape[2] not in {2, 3}:
            raise ValueError("trajectory mean must have shape (T, N, 2|3)")
        if variance.shape != mean.shape:
            raise ValueError("trajectory variance must match the mean")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(variance)):
            raise ValueError("trajectory moments must be finite")
        if np.any(variance <= 0.0):
            raise ValueError("trajectory variances must be positive")
        if (self.interval_lower is None) != (self.interval_upper is None):
            raise ValueError("both interval bounds must be supplied together")
        if self.interval_lower is not None and self.interval_upper is not None:
            lower = np.asarray(self.interval_lower, dtype=float)
            upper = np.asarray(self.interval_upper, dtype=float)
            if lower.shape != mean.shape or upper.shape != mean.shape:
                raise ValueError("trajectory interval bounds must match the mean")
            if np.any(lower > upper):
                raise ValueError("trajectory interval lower bound exceeds upper bound")
            object.__setattr__(self, "interval_lower", lower)
            object.__setattr__(self, "interval_upper", upper)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "variance", variance)


@dataclass(frozen=True)
class SparseTrajectoryEvidence:
    """Sparse external trajectory evidence aligned to physical rollout frames."""

    positions_m: np.ndarray
    node_indices: np.ndarray
    rollout_frame_indices: np.ndarray
    scale_m: float = 0.10
    degrees_of_freedom: float = 3.0
    likelihood_weight: float = 1.0
    compare_displacements: bool = True
    anchor_positions_m: np.ndarray | None = None
    anchor_rollout_frame: int = 0
    valid: np.ndarray | None = None
    source: str = "external_trajectory"

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions_m, dtype=float)
        nodes = np.asarray(self.node_indices, dtype=int)
        frames = np.asarray(self.rollout_frame_indices, dtype=float)
        if positions.ndim != 3 or positions.shape[2] not in {2, 3}:
            raise ValueError("evidence positions must have shape (F, Q, 2|3)")
        if nodes.shape != (positions.shape[1],):
            raise ValueError("node_indices must identify every evidence point")
        if frames.shape != (positions.shape[0],):
            raise ValueError("rollout_frame_indices must identify every evidence frame")
        if np.any(nodes < 0) or not np.all(np.isfinite(frames)):
            raise ValueError("evidence nodes and frame indices must be valid")
        if self.scale_m <= 0.0 or self.degrees_of_freedom <= 0.0:
            raise ValueError("evidence scale and degrees of freedom must be positive")
        if self.likelihood_weight < 0.0:
            raise ValueError("evidence likelihood_weight must be nonnegative")
        if self.valid is not None:
            valid = np.asarray(self.valid, dtype=bool)
            if valid.shape == positions.shape[:2]:
                valid = np.repeat(valid[:, :, None], positions.shape[2], axis=2)
            if valid.shape != positions.shape:
                raise ValueError("evidence validity must have shape (F, Q) or (F, Q, C)")
            object.__setattr__(self, "valid", valid)
        if self.compare_displacements:
            if self.anchor_positions_m is None:
                raise ValueError("displacement evidence requires anchor_positions_m")
            anchor = np.asarray(self.anchor_positions_m, dtype=float)
            if anchor.shape != positions.shape[1:]:
                raise ValueError("anchor_positions_m must have shape (Q, C)")
            if not np.all(np.isfinite(anchor)):
                raise ValueError("anchor_positions_m must be finite")
            object.__setattr__(self, "anchor_positions_m", anchor)
        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(self, "node_indices", nodes)
        object.__setattr__(self, "rollout_frame_indices", frames)


@dataclass(frozen=True)
class JointRolloutBank:
    """Finite support over rollout hypotheses and physical parameter particles."""

    hypothesis_ids: tuple[str, ...]
    hypothesis_metadata: tuple[dict[str, Any], ...]
    hypothesis_prior_weights: np.ndarray
    parameter_particles: np.ndarray
    parameter_weights: np.ndarray
    trajectories: np.ndarray
    variance_floor_m2: float = 1e-6
    confidence_level: float = 0.90

    def __post_init__(self) -> None:
        hypothesis_weights = _normalized_weights(
            self.hypothesis_prior_weights,
            name="hypothesis_prior_weights",
        )
        parameter_weights = _normalized_weights(
            self.parameter_weights,
            name="parameter_weights",
        )
        particles = np.asarray(self.parameter_particles, dtype=float)
        trajectories = np.asarray(self.trajectories, dtype=np.float32)
        hypothesis_count = len(self.hypothesis_ids)
        if len(set(self.hypothesis_ids)) != hypothesis_count:
            raise ValueError("hypothesis_ids must be unique")
        if len(self.hypothesis_metadata) != hypothesis_count:
            raise ValueError("hypothesis metadata must match hypothesis ids")
        if hypothesis_weights.shape != (hypothesis_count,):
            raise ValueError("hypothesis weights must match hypothesis ids")
        if particles.ndim != 2 or particles.shape[0] != len(parameter_weights):
            raise ValueError("parameter_particles must have shape (P, D)")
        if trajectories.ndim != 5 or trajectories.shape[:2] != (
            hypothesis_count,
            len(parameter_weights),
        ):
            raise ValueError("trajectories must have shape (H, P, T, N, C)")
        if trajectories.shape[-1] not in {2, 3}:
            raise ValueError("rollout coordinates must be 2D or 3D")
        if not np.all(np.isfinite(particles)) or not np.all(np.isfinite(trajectories)):
            raise ValueError("rollout bank arrays must be finite")
        if self.variance_floor_m2 <= 0.0:
            raise ValueError("variance_floor_m2 must be positive")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must lie in (0, 1)")
        object.__setattr__(self, "hypothesis_prior_weights", hypothesis_weights)
        object.__setattr__(self, "parameter_weights", parameter_weights)
        object.__setattr__(self, "parameter_particles", particles)
        object.__setattr__(self, "trajectories", trajectories)

    @property
    def prior_joint_weights(self) -> np.ndarray:
        return self.hypothesis_prior_weights[:, None] * self.parameter_weights[None]

    @property
    def frame_count(self) -> int:
        return int(self.trajectories.shape[2])

    @property
    def node_count(self) -> int:
        return int(self.trajectories.shape[3])

    @property
    def coordinate_count(self) -> int:
        return int(self.trajectories.shape[4])

    def _base_log_weights(self, base_weights: np.ndarray | None) -> np.ndarray:
        weights = (
            self.prior_joint_weights
            if base_weights is None
            else np.asarray(base_weights, dtype=float)
        )
        if weights.shape != self.prior_joint_weights.shape:
            raise ValueError("base_weights must match the joint rollout support")
        if np.any(weights < 0.0) or not np.isclose(np.sum(weights), 1.0):
            raise ValueError("base_weights must be nonnegative and sum to one")
        return np.log(np.maximum(weights, 1e-300))

    def update_from_observations(
        self,
        observations_m: np.ndarray,
        *,
        prefix_frame_count: int,
        scale_m: float,
        likelihood_power: float = 1.0,
        dynamic_likelihood_weight: float = 0.0,
        degrees_of_freedom: float = 4.0,
        mask: np.ndarray | None = None,
        observed_nodes: Sequence[int] | None = None,
        base_weights: np.ndarray | None = None,
        particle_discrepancy_m: np.ndarray | None = None,
        particle_discrepancy_variance_m2: np.ndarray | None = None,
    ) -> np.ndarray:
        """Update only from frames before ``prefix_frame_count``."""

        observations = np.asarray(observations_m, dtype=float)
        expected = self.trajectories.shape[2:]
        if observations.shape != expected:
            raise ValueError(f"observations must have shape {expected}")
        if not 2 <= prefix_frame_count < self.frame_count:
            raise ValueError("prefix_frame_count must leave at least one future frame")
        if scale_m <= 0.0 or likelihood_power <= 0.0:
            raise ValueError("observation scale and likelihood power must be positive")
        if dynamic_likelihood_weight < 0.0:
            raise ValueError("dynamic_likelihood_weight must be nonnegative")
        nodes = np.asarray(
            tuple(range(self.node_count))
            if observed_nodes is None
            else tuple(observed_nodes),
            dtype=int,
        )
        if nodes.ndim != 1 or not len(nodes) or np.any(nodes < 0) or np.any(nodes >= self.node_count):
            raise ValueError("observed_nodes must identify available rollout nodes")
        coordinate_valid = _coordinate_mask(observations, mask)
        selected_observed = observations[1:prefix_frame_count, nodes]
        selected_valid = coordinate_valid[1:prefix_frame_count, nodes]
        predicted = self.trajectories[:, :, 1:prefix_frame_count, nodes]
        likelihood_scale: np.ndarray | float = scale_m
        if particle_discrepancy_m is not None:
            discrepancy = np.asarray(particle_discrepancy_m, dtype=float)
            expected_discrepancy_shape = (
                len(self.parameter_weights),
                self.node_count,
                self.coordinate_count,
            )
            if discrepancy.shape != expected_discrepancy_shape:
                raise ValueError(
                    "particle_discrepancy_m must have shape "
                    f"{expected_discrepancy_shape}"
                )
            if not np.all(np.isfinite(discrepancy)):
                raise ValueError("particle discrepancy must be finite")
            predicted = predicted + discrepancy[None, :, None, nodes]
        if particle_discrepancy_variance_m2 is not None:
            discrepancy_variance = np.asarray(
                particle_discrepancy_variance_m2,
                dtype=float,
            )
            expected_variance_shape = (
                len(self.parameter_weights),
                self.node_count,
                self.coordinate_count,
            )
            if discrepancy_variance.shape != expected_variance_shape:
                raise ValueError(
                    "particle_discrepancy_variance_m2 must have shape "
                    f"{expected_variance_shape}"
                )
            if not np.all(np.isfinite(discrepancy_variance)) or np.any(
                discrepancy_variance < 0.0
            ):
                raise ValueError("particle discrepancy variance must be nonnegative")
            likelihood_scale = np.sqrt(
                scale_m**2 + discrepancy_variance[None, :, None, nodes]
            )
        score = _student_t_mean_log_score(
            predicted - selected_observed[None, None],
            selected_valid,
            scale_m=likelihood_scale,
            degrees_of_freedom=degrees_of_freedom,
            reduction_axes=(2, 3, 4),
        )
        if dynamic_likelihood_weight and prefix_frame_count >= 4:
            predicted_velocity = np.diff(predicted, axis=2)
            observed_velocity = np.diff(selected_observed, axis=0)
            velocity_valid = selected_valid[1:] & selected_valid[:-1]
            dynamic_score = _student_t_mean_log_score(
                predicted_velocity - observed_velocity[None, None],
                velocity_valid,
                scale_m=scale_m,
                degrees_of_freedom=degrees_of_freedom,
                reduction_axes=(2, 3, 4),
            )
            score += dynamic_likelihood_weight * dynamic_score
        return _normalize_joint_log_weights(
            self._base_log_weights(base_weights) + likelihood_power * score
        )

    def _interpolated_nodes(self, evidence: SparseTrajectoryEvidence) -> np.ndarray:
        frames = evidence.rollout_frame_indices
        if np.min(frames) < 0.0 or np.max(frames) > self.frame_count - 1:
            raise ValueError("evidence frame indices fall outside the rollout bank")
        if np.any(evidence.node_indices >= self.node_count):
            raise ValueError("evidence references an unavailable rollout node")
        lower = np.floor(frames).astype(int)
        upper = np.ceil(frames).astype(int)
        alpha = (frames - lower).reshape(1, 1, -1, 1, 1)
        selected_lower = self.trajectories[:, :, lower][:, :, :, evidence.node_indices]
        selected_upper = self.trajectories[:, :, upper][:, :, :, evidence.node_indices]
        return (1.0 - alpha) * selected_lower + alpha * selected_upper

    def update_from_sparse_evidence(
        self,
        evidence: SparseTrajectoryEvidence,
        *,
        base_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply robust product-of-experts evidence over physical rollouts."""

        predicted = self._interpolated_nodes(evidence)
        target = evidence.positions_m
        if target.shape[2] != self.coordinate_count:
            raise ValueError("evidence coordinate count differs from the rollout bank")
        if evidence.compare_displacements:
            predicted_anchor = self.trajectories[
                :, :, evidence.anchor_rollout_frame, evidence.node_indices
            ]
            predicted = predicted - predicted_anchor[:, :, None]
            target = target - evidence.anchor_positions_m[None]
        valid = np.isfinite(target)
        if evidence.valid is not None:
            valid &= evidence.valid
        score = _student_t_mean_log_score(
            predicted - target[None, None],
            valid,
            scale_m=evidence.scale_m,
            degrees_of_freedom=evidence.degrees_of_freedom,
            reduction_axes=(2, 3, 4),
        )
        return _normalize_joint_log_weights(
            self._base_log_weights(base_weights)
            + evidence.likelihood_weight * score
        )

    def predictive_distribution(
        self,
        joint_weights: np.ndarray | None = None,
        *,
        method: str,
        variance_multiplier: float = 1.0,
        include_intervals: bool = True,
    ) -> PhysicalTrajectoryDistribution:
        weights = (
            self.prior_joint_weights
            if joint_weights is None
            else np.asarray(joint_weights, dtype=float)
        )
        if weights.shape != self.prior_joint_weights.shape:
            raise ValueError("joint_weights must match the rollout bank")
        if np.any(weights < 0.0) or not np.isclose(np.sum(weights), 1.0):
            raise ValueError("joint_weights must be nonnegative and sum to one")
        if variance_multiplier <= 0.0:
            raise ValueError("variance_multiplier must be positive")
        mean = np.zeros(self.trajectories.shape[2:], dtype=np.float64)
        second_moment = np.zeros_like(mean)
        for hypothesis in range(self.trajectories.shape[0]):
            for particle in range(self.trajectories.shape[1]):
                weight = float(weights[hypothesis, particle])
                if weight <= 0.0:
                    continue
                trajectory = self.trajectories[hypothesis, particle]
                mean += weight * trajectory
                second_moment += weight * np.square(trajectory, dtype=np.float64)
        variance = np.maximum(second_moment - np.square(mean), 0.0)
        variance = variance_multiplier * (variance + self.variance_floor_m2)
        if not include_intervals:
            return PhysicalTrajectoryDistribution(method, mean, variance)
        components = mean[None, None] + np.sqrt(variance_multiplier) * (
            self.trajectories - mean[None, None]
        )
        flat_components = components.reshape(-1, int(np.prod(mean.shape)))
        flat_weights = weights.reshape(-1)
        tail = 0.5 * (1.0 - self.confidence_level)
        lower = _weighted_component_quantile(
            flat_components, flat_weights, tail
        ).reshape(mean.shape)
        upper = _weighted_component_quantile(
            flat_components, flat_weights, 1.0 - tail
        ).reshape(mean.shape)
        quantile = NormalDist().inv_cdf(0.5 * (1.0 + self.confidence_level))
        margin = quantile * np.sqrt(variance_multiplier * self.variance_floor_m2)
        return PhysicalTrajectoryDistribution(
            method,
            mean,
            variance,
            interval_lower=lower - margin,
            interval_upper=upper + margin,
        )

    def hypothesis_marginal(self, joint_weights: np.ndarray) -> np.ndarray:
        weights = np.asarray(joint_weights, dtype=float)
        if weights.shape != self.prior_joint_weights.shape:
            raise ValueError("joint_weights must match the rollout bank")
        marginal = np.sum(weights, axis=1)
        return marginal / np.sum(marginal)

    def parameter_marginal(self, joint_weights: np.ndarray) -> np.ndarray:
        weights = np.asarray(joint_weights, dtype=float)
        if weights.shape != self.prior_joint_weights.shape:
            raise ValueError("joint_weights must match the rollout bank")
        marginal = np.sum(weights, axis=0)
        return marginal / np.sum(marginal)
