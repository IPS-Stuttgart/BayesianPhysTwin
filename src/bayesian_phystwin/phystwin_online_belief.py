"""Recursive robust belief fields for sparse online PhysTwin observations.

The state is deliberately small: one global translation and one local residual
vector for each point selected at the beginning of an episode.  A Gaussian RBF
decoder turns that state into a correction for any point in the physical
rollout.  Missing centres retain their previous posterior, so an occlusion does
not delete the forecast.

This module does not read trajectories or targets.  The evaluator owns the
causal boundary and passes only measurements available at an update frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def finite_sample_absolute_residual_quantile_m(
    measured_residual_m: np.ndarray,
    available: np.ndarray,
    nominal_coverage: float,
) -> float:
    """Return a causal finite-sample absolute-residual half-width.

    The score pool contains the three absolute coordinate residuals for every
    currently available centre.  The selected one-indexed order statistic is
    ``min(n, ceil((n + 1) * nominal_coverage))``.  Coordinates within a frame
    are dependent, so callers must describe this as a conformal-style interval
    rather than claiming a formal iid conformal guarantee.
    """

    residual = np.asarray(measured_residual_m, dtype=float)
    mask = np.asarray(available, dtype=bool)
    if residual.ndim != 2 or residual.shape[1] != 3:
        raise ValueError("measured_residual_m must have shape (K, 3)")
    if mask.shape != (len(residual),):
        raise ValueError("available must have shape (K,)")
    if not np.isfinite(nominal_coverage) or not 0.0 < nominal_coverage < 1.0:
        raise ValueError("nominal_coverage must lie in (0, 1)")
    mask &= np.all(np.isfinite(residual), axis=1)
    scores = np.abs(residual[mask]).reshape(-1)
    if not len(scores):
        raise ValueError("no finite available residual coordinate")
    rank = min(len(scores), int(np.ceil((len(scores) + 1) * nominal_coverage)))
    return float(np.partition(scores, rank - 1)[rank - 1])


def robust_huber_continuation_gain(
    physical_displacement_m: np.ndarray,
    observed_displacement_m: np.ndarray,
    *,
    minimum_point_count: int = 3,
    fallback: float = 0.0,
) -> float:
    """Estimate how much of the physical continuation was observed.

    This is a scalar point-vector projection, robustified by Huber IRLS.  It
    drops the lowest prior-motion quartile to avoid unstable ratios at nearly
    stationary points and clips the result to ``[0, 1]``.  Both displacement
    arrays must end at the current observation, so the estimate is causal.
    The default zero fallback freezes motion when fewer than
    ``minimum_point_count`` overlapping observations support a projection.
    """

    physical = np.asarray(physical_displacement_m, dtype=float)
    observed = np.asarray(observed_displacement_m, dtype=float)
    if physical.ndim != 2 or physical.shape[1] != 3:
        raise ValueError("physical_displacement_m must have shape (N, 3)")
    if observed.shape != physical.shape:
        raise ValueError("observed_displacement_m must match physical displacement")
    if minimum_point_count < 1:
        raise ValueError("minimum_point_count must be positive")
    if not np.isfinite(fallback) or not 0.0 <= fallback <= 1.0:
        raise ValueError("fallback must lie in [0, 1]")

    finite = np.all(np.isfinite(physical), axis=1) & np.all(
        np.isfinite(observed), axis=1
    )
    physical = physical[finite]
    observed = observed[finite]
    if len(physical) < minimum_point_count:
        return float(fallback)
    motion = np.linalg.norm(physical, axis=1)
    positive = motion[motion > 1e-6]
    if len(positive) < minimum_point_count:
        return float(fallback)
    motion_threshold = max(1e-5, float(np.quantile(positive, 0.25)))
    keep = motion >= motion_threshold
    physical = physical[keep]
    observed = observed[keep]
    if len(physical) < minimum_point_count:
        return float(fallback)

    denominator = float(np.sum(np.square(physical)))
    if denominator <= 1e-12:
        return float(fallback)
    gain = float(np.sum(physical * observed) / denominator)
    for _ in range(20):
        radial = np.linalg.norm(observed - gain * physical, axis=1)
        center = float(np.median(radial))
        mad = float(np.median(np.abs(radial - center)))
        scale = max(1e-4, 1.4826 * mad)
        cutoff = 1.345 * scale
        weight = np.minimum(1.0, cutoff / np.maximum(radial, 1e-12))
        weighted_denominator = float(np.sum(weight[:, None] * np.square(physical)))
        if weighted_denominator <= 1e-12:
            break
        updated = float(
            np.sum(weight[:, None] * physical * observed) / weighted_denominator
        )
        if abs(updated - gain) <= 1e-8:
            gain = updated
            break
        gain = updated
    return float(np.clip(gain, 0.0, 1.0))


def deterministic_farthest_point_ids(
    positions_m: np.ndarray,
    candidate_ids: np.ndarray,
    count: int,
) -> np.ndarray:
    """Select deterministic, geometry-spanning point IDs.

    The smallest source ID is the first centre.  Distance and ID ties are
    resolved deterministically, making selection independent of array order.
    """

    positions = np.asarray(positions_m, dtype=float)
    candidates = np.asarray(candidate_ids, dtype=np.int64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions_m must have shape (N, 3)")
    if candidates.ndim != 1 or len(candidates) == 0:
        raise ValueError("candidate_ids must be a nonempty vector")
    if count < 1:
        raise ValueError("count must be positive")
    if np.any(candidates < 0) or np.any(candidates >= len(positions)):
        raise ValueError("candidate ID exceeds positions_m")
    if len(np.unique(candidates)) != len(candidates):
        raise ValueError("candidate_ids must be unique")
    if not np.all(np.isfinite(positions[candidates])):
        raise ValueError("candidate positions must be finite")

    ordered = np.sort(candidates, kind="mergesort")
    selected = [int(ordered[0])]
    selected_mask = ordered == selected[0]
    minimum_distance = np.linalg.norm(
        positions[ordered] - positions[selected[0]], axis=1
    )
    while len(selected) < min(count, len(ordered)):
        candidate_distance = minimum_distance.copy()
        candidate_distance[selected_mask] = -np.inf
        maximum = float(np.max(candidate_distance))
        tied = ordered[
            (~selected_mask)
            & np.isclose(candidate_distance, maximum, rtol=0.0, atol=1e-15)
        ]
        next_id = int(np.min(tied))
        selected.append(next_id)
        selected_mask |= ordered == next_id
        minimum_distance = np.minimum(
            minimum_distance,
            np.linalg.norm(positions[ordered] - positions[next_id], axis=1),
        )
    result = np.asarray(selected, dtype=np.int64)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class RecursiveRbfBeliefConfig:
    """Fixed hyperparameters for a recursive RBF discrepancy belief."""

    length_scale_fraction: float = 0.10
    local_blend: float = 0.25
    observation_std_m: float = 0.005
    process_std_m_per_sqrt_frame: float = 0.003
    global_prior_std_m: float = 0.10
    local_prior_std_m: float = 0.02
    degrees_of_freedom: float = 4.0
    minimum_reliability: float = 0.02
    maximum_correction_m: float = 0.10
    minimum_length_scale_m: float = 1e-4

    def __post_init__(self) -> None:
        positive = (
            self.length_scale_fraction,
            self.observation_std_m,
            self.process_std_m_per_sqrt_frame,
            self.global_prior_std_m,
            self.local_prior_std_m,
            self.degrees_of_freedom,
            self.maximum_correction_m,
            self.minimum_length_scale_m,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("belief scales and degrees of freedom must be positive")
        if not 0.0 <= self.local_blend <= 1.0:
            raise ValueError("local_blend must lie in [0, 1]")
        if not 0.0 < self.minimum_reliability <= 1.0:
            raise ValueError("minimum_reliability must lie in (0, 1]")


@dataclass(frozen=True)
class RecursiveRbfBeliefSnapshot:
    """Immutable state after one or more causal updates."""

    center_ids: np.ndarray
    center_positions_m: np.ndarray
    global_mean_m: np.ndarray
    global_variance_m2: np.ndarray
    local_mean_m: np.ndarray
    local_variance_m2: np.ndarray
    update_count: np.ndarray
    last_update_frame: int | None
    object_scale_m: float

    def __post_init__(self) -> None:
        arrays = {
            "center_ids": np.asarray(self.center_ids, dtype=np.int64).copy(),
            "center_positions_m": np.asarray(
                self.center_positions_m, dtype=float
            ).copy(),
            "global_mean_m": np.asarray(self.global_mean_m, dtype=float).copy(),
            "global_variance_m2": np.asarray(
                self.global_variance_m2, dtype=float
            ).copy(),
            "local_mean_m": np.asarray(self.local_mean_m, dtype=float).copy(),
            "local_variance_m2": np.asarray(self.local_variance_m2, dtype=float).copy(),
            "update_count": np.asarray(self.update_count, dtype=np.int64).copy(),
        }
        center_count = len(arrays["center_ids"])
        if arrays["center_ids"].shape != (center_count,):
            raise ValueError("center_ids must be a vector")
        if arrays["center_positions_m"].shape != (center_count, 3):
            raise ValueError("center_positions_m must have shape (K, 3)")
        if arrays["global_mean_m"].shape != (3,) or arrays[
            "global_variance_m2"
        ].shape != (3,):
            raise ValueError("global moments must have shape (3,)")
        if arrays["local_mean_m"].shape != (center_count, 3) or arrays[
            "local_variance_m2"
        ].shape != (center_count, 3):
            raise ValueError("local moments must have shape (K, 3)")
        if arrays["update_count"].shape != (center_count,):
            raise ValueError("update_count must have shape (K,)")
        if any(not np.all(np.isfinite(value)) for value in arrays.values()):
            raise ValueError("belief state must be finite")
        if np.any(arrays["global_variance_m2"] <= 0.0) or np.any(
            arrays["local_variance_m2"] <= 0.0
        ):
            raise ValueError("belief variances must be positive")
        if np.any(arrays["update_count"] < 0):
            raise ValueError("update counts must be nonnegative")
        if not np.isfinite(self.object_scale_m) or self.object_scale_m <= 0.0:
            raise ValueError("object_scale_m must be positive")
        if self.last_update_frame is not None and self.last_update_frame < 0:
            raise ValueError("last_update_frame must be nonnegative")
        for name, value in arrays.items():
            value.setflags(write=False)
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class BeliefFieldPrediction:
    """Decoded correction mean and coordinate-wise marginal variance."""

    mean_m: np.ndarray
    variance_m2: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean_m, dtype=float)
        variance = np.asarray(self.variance_m2, dtype=float)
        if mean.ndim != 2 or mean.shape[1] != 3 or variance.shape != mean.shape:
            raise ValueError("field moments must have shape (N, 3)")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(variance)):
            raise ValueError("field moments must be finite")
        if np.any(variance <= 0.0):
            raise ValueError("field variance must be positive")


def initialize_recursive_rbf_belief(
    center_ids: np.ndarray,
    center_positions_m: np.ndarray,
    object_positions_m: np.ndarray,
    *,
    config: RecursiveRbfBeliefConfig,
) -> RecursiveRbfBeliefSnapshot:
    """Create a diffuse, zero-correction prior without reading future data."""

    centers = np.asarray(center_ids, dtype=np.int64)
    center_positions = np.asarray(center_positions_m, dtype=float)
    object_positions = np.asarray(object_positions_m, dtype=float)
    if centers.ndim != 1 or len(centers) == 0:
        raise ValueError("center_ids must be a nonempty vector")
    if center_positions.shape != (len(centers), 3):
        raise ValueError("center_positions_m must have shape (K, 3)")
    if object_positions.ndim != 2 or object_positions.shape[1] != 3:
        raise ValueError("object_positions_m must have shape (N, 3)")
    finite = np.all(np.isfinite(object_positions), axis=1)
    if not np.any(finite) or not np.all(np.isfinite(center_positions)):
        raise ValueError("initial geometry must contain finite positions")
    lower = np.quantile(object_positions[finite], 0.05, axis=0)
    upper = np.quantile(object_positions[finite], 0.95, axis=0)
    scale = max(float(np.linalg.norm(upper - lower)), config.minimum_length_scale_m)
    return RecursiveRbfBeliefSnapshot(
        center_ids=centers,
        center_positions_m=center_positions,
        global_mean_m=np.zeros(3),
        global_variance_m2=np.full(3, config.global_prior_std_m**2),
        local_mean_m=np.zeros((len(centers), 3)),
        local_variance_m2=np.full((len(centers), 3), config.local_prior_std_m**2),
        update_count=np.zeros(len(centers), dtype=np.int64),
        last_update_frame=None,
        object_scale_m=scale,
    )


def _student_t_reliability(
    residual: np.ndarray,
    scale: np.ndarray,
    *,
    degrees_of_freedom: float,
    minimum: float,
) -> np.ndarray:
    standardized = residual / scale[None]
    squared_radius = np.sum(np.square(standardized), axis=1)
    dimension = residual.shape[1]
    reliability = (degrees_of_freedom + dimension) / (
        degrees_of_freedom + squared_radius
    )
    return np.clip(reliability, minimum, 1.0)


def update_recursive_rbf_belief(
    prior: RecursiveRbfBeliefSnapshot,
    frame_index: int,
    center_positions_m: np.ndarray,
    measured_residual_m: np.ndarray,
    available: np.ndarray,
    *,
    config: RecursiveRbfBeliefConfig,
) -> tuple[RecursiveRbfBeliefSnapshot, np.ndarray]:
    """Apply one robust causal measurement update.

    Returns the posterior and the Student-t reliability assigned to each
    centre.  Unavailable centres receive zero reliability and retain their
    predicted prior.
    """

    if frame_index < 0:
        raise ValueError("frame_index must be nonnegative")
    if prior.last_update_frame is not None and frame_index <= prior.last_update_frame:
        raise ValueError("updates must have strictly increasing frame indices")
    positions = np.asarray(center_positions_m, dtype=float)
    residual = np.asarray(measured_residual_m, dtype=float)
    mask = np.asarray(available, dtype=bool)
    center_count = len(prior.center_ids)
    if positions.shape != (center_count, 3) or residual.shape != (center_count, 3):
        raise ValueError("centre positions and residuals must have shape (K, 3)")
    if mask.shape != (center_count,):
        raise ValueError("available must have shape (K,)")
    mask &= np.all(np.isfinite(positions), axis=1)
    mask &= np.all(np.isfinite(residual), axis=1)

    elapsed = (
        0 if prior.last_update_frame is None else frame_index - prior.last_update_frame
    )
    process_variance = elapsed * config.process_std_m_per_sqrt_frame**2
    global_variance = prior.global_variance_m2 + process_variance
    local_variance = prior.local_variance_m2 + process_variance
    global_mean = prior.global_mean_m.copy()
    local_mean = prior.local_mean_m.copy()
    update_count = prior.update_count.copy()
    reliability = np.zeros(center_count, dtype=float)

    if np.any(mask):
        selected = residual[mask]
        robust_location = np.median(selected, axis=0)
        absolute_deviation = np.abs(selected - robust_location)
        robust_scale = 1.4826 * np.median(absolute_deviation, axis=0)
        robust_scale = np.maximum(robust_scale, config.observation_std_m)

        # A coordinate-wise median is the high-breakdown observation of the
        # global component.  Its variance shrinks with available point count.
        global_observation_variance = np.square(robust_scale) / np.sum(mask)
        global_gain = global_variance / (global_variance + global_observation_variance)
        global_mean += global_gain * (robust_location - global_mean)
        global_variance *= 1.0 - global_gain

        local_observation = selected - global_mean
        local_scale = np.maximum(robust_scale, config.observation_std_m)
        selected_reliability = _student_t_reliability(
            local_observation,
            local_scale,
            degrees_of_freedom=config.degrees_of_freedom,
            minimum=config.minimum_reliability,
        )
        selected_ids = np.flatnonzero(mask)
        reliability[selected_ids] = selected_reliability
        for local_index, centre_index in enumerate(selected_ids):
            observation_variance = (
                config.observation_std_m**2 / selected_reliability[local_index]
            )
            gain = local_variance[centre_index] / (
                local_variance[centre_index] + observation_variance
            )
            local_mean[centre_index] += gain * (
                local_observation[local_index] - local_mean[centre_index]
            )
            local_variance[centre_index] *= 1.0 - gain
            update_count[centre_index] += 1

    posterior = RecursiveRbfBeliefSnapshot(
        center_ids=prior.center_ids,
        center_positions_m=np.where(mask[:, None], positions, prior.center_positions_m),
        global_mean_m=global_mean,
        global_variance_m2=np.maximum(global_variance, 1e-12),
        local_mean_m=local_mean,
        local_variance_m2=np.maximum(local_variance, 1e-12),
        update_count=update_count,
        last_update_frame=frame_index,
        object_scale_m=prior.object_scale_m,
    )
    reliability.setflags(write=False)
    return posterior, reliability


def decode_recursive_rbf_belief(
    belief: RecursiveRbfBeliefSnapshot,
    query_positions_m: np.ndarray,
    *,
    forecast_frames: int,
    config: RecursiveRbfBeliefConfig,
) -> BeliefFieldPrediction:
    """Decode the same latent belief at arbitrary point queries."""

    query = np.asarray(query_positions_m, dtype=float)
    if query.ndim != 2 or query.shape[1] != 3 or not np.all(np.isfinite(query)):
        raise ValueError("query_positions_m must have finite shape (N, 3)")
    if forecast_frames < 0:
        raise ValueError("forecast_frames must be nonnegative")
    active = belief.update_count > 0
    mean = np.repeat(belief.global_mean_m[None], len(query), axis=0)
    variance = np.repeat(belief.global_variance_m2[None], len(query), axis=0)
    process_variance = forecast_frames * config.process_std_m_per_sqrt_frame**2
    variance += process_variance

    if np.any(active) and config.local_blend > 0.0:
        centres = belief.center_positions_m[active]
        distance = np.linalg.norm(query[:, None] - centres[None], axis=2)
        length_scale = max(
            belief.object_scale_m * config.length_scale_fraction,
            config.minimum_length_scale_m,
        )
        weight = np.exp(-0.5 * np.square(distance / length_scale))
        weight_sum = np.sum(weight, axis=1, keepdims=True)
        normalized = weight / np.maximum(weight_sum, 1e-15)
        unsupported = weight_sum[:, 0] < 1e-12
        normalized[unsupported] = 0.0
        mean += config.local_blend * (normalized @ belief.local_mean_m[active])
        local_variance = belief.local_variance_m2[active] + process_variance
        variance += config.local_blend**2 * (np.square(normalized) @ local_variance)

    norm = np.linalg.norm(mean, axis=1, keepdims=True)
    mean *= np.minimum(1.0, config.maximum_correction_m / np.maximum(norm, 1e-15))
    return BeliefFieldPrediction(
        mean_m=mean,
        variance_m2=np.maximum(variance, 1e-12),
    )
