"""Export full Bayesian-PhysTwin endpoint particles for Causal4D."""

from __future__ import annotations

import gc
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from bayesian_phystwin.phystwin_additional_bayesian_confirmation import (
    FIXED_INITIAL_STD_M,
    FIXED_INLIER_PRIOR,
    FIXED_OBSERVATION_STD_M,
    FIXED_OUTLIER_VARIANCE_MULTIPLIER,
    FIXED_PROCESS_STD_M,
)
from bayesian_phystwin.phystwin_bayesian_anchor import robust_random_walk_endpoint
from bayesian_phystwin.phystwin_residual_dynamics import (
    _lift_map,
    _lift_residual,
    _target_validity,
)
from causal4d.contracts import CausalContext, TwinBelief

if TYPE_CHECKING:
    from causal4d.phystwin_backend import OfficialPhysTwinBackend


@dataclass(frozen=True)
class BPTBeliefExportConfig:
    """Fixed, label-free settings for a full endpoint belief export."""

    process_std_m: float = FIXED_PROCESS_STD_M
    observation_std_m: float = FIXED_OBSERVATION_STD_M
    initial_std_m: float = FIXED_INITIAL_STD_M
    inlier_prior: float = FIXED_INLIER_PRIOR
    outlier_variance_multiplier: float = FIXED_OUTLIER_VARIANCE_MULTIPLIER
    interpolation_neighbors: int = 4
    maximum_discrepancy_m: float = 0.01

    def __post_init__(self) -> None:
        if self.process_std_m < 0.0:
            raise ValueError("process_std_m must be nonnegative")
        if self.observation_std_m <= 0.0 or self.initial_std_m <= 0.0:
            raise ValueError("observation and initial scales must be positive")
        if not 0.0 < self.inlier_prior < 1.0:
            raise ValueError("inlier_prior must lie in (0, 1)")
        if self.outlier_variance_multiplier <= 1.0:
            raise ValueError("outlier_variance_multiplier must exceed one")
        if self.interpolation_neighbors < 1 or self.maximum_discrepancy_m <= 0.0:
            raise ValueError("lifting settings must be positive")


def lift_isotropic_discrepancy_variance(
    tracked_variance_m2: np.ndarray,
    state_count: int,
    indices: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Propagate independent tracked variances through a fixed kNN readout."""

    tracked = np.asarray(tracked_variance_m2, dtype=float)
    neighbor_indices = np.asarray(indices, dtype=np.int64)
    neighbor_weights = np.asarray(weights, dtype=float)
    if tracked.ndim != 1 or not np.all(np.isfinite(tracked)) or np.any(tracked < 0.0):
        raise ValueError("tracked_variance_m2 must be a finite nonnegative vector")
    if state_count < len(tracked):
        raise ValueError("state_count cannot be smaller than the tracked state")
    extra_count = state_count - len(tracked)
    if neighbor_indices.shape != neighbor_weights.shape or neighbor_indices.shape[0] != extra_count:
        raise ValueError("lift map must identify every untracked state node")
    if np.any(neighbor_indices < 0) or np.any(neighbor_indices >= len(tracked)):
        raise ValueError("lift map references an unavailable tracked node")
    if extra_count and not np.allclose(np.sum(neighbor_weights, axis=1), 1.0):
        raise ValueError("lift weights must sum to one")
    scalar = np.empty(state_count, dtype=float)
    scalar[: len(tracked)] = tracked
    if extra_count:
        scalar[len(tracked) :] = np.sum(
            np.square(neighbor_weights) * tracked[neighbor_indices],
            axis=1,
        )
    return np.repeat(scalar[:, None], 3, axis=1)


def build_twin_belief_from_replays(
    *,
    context: CausalContext,
    replay_positions_m: np.ndarray,
    replay_velocities_mps: np.ndarray,
    observed_positions_m: np.ndarray,
    observed_valid: np.ndarray,
    theta: np.ndarray,
    theta_names: tuple[str, ...],
    weights: np.ndarray,
    particle_ids: tuple[str, ...] | None = None,
    metadata: dict[str, Any] | None = None,
    config: BPTBeliefExportConfig | None = None,
) -> TwinBelief:
    """Build a belief using only the declared pre-intervention prefix."""

    settings = config or BPTBeliefExportConfig()
    positions = np.asarray(replay_positions_m, dtype=float)
    velocities = np.asarray(replay_velocities_mps, dtype=float)
    observed = np.asarray(observed_positions_m, dtype=float)
    valid = np.asarray(observed_valid, dtype=bool)
    particle_values = np.asarray(theta, dtype=float)
    particle_weights = np.asarray(weights, dtype=float)
    train_end = context.o_minus.frame_stop
    if positions.ndim != 4 or positions.shape[-1] != 3:
        raise ValueError("replay_positions_m must have shape (P, T, N, 3)")
    if velocities.shape != positions.shape:
        raise ValueError("replay velocities must match replay positions")
    particle_count, frame_count, state_count, _ = positions.shape
    if frame_count < train_end:
        raise ValueError("replays do not cover O-")
    if observed.ndim != 3 or observed.shape[2] != 3 or len(observed) < train_end:
        raise ValueError("observed_positions_m must cover O- with shape (T, N, 3)")
    tracked_count = observed.shape[1]
    if tracked_count > state_count or valid.shape != observed.shape[:2]:
        raise ValueError("observed validity or tracked state size is inconsistent")
    if particle_values.shape != (particle_count, len(theta_names)):
        raise ValueError("theta does not identify every replay particle")
    if particle_weights.shape != (particle_count,):
        raise ValueError("weights do not identify every replay particle")
    if not 1 <= settings.interpolation_neighbors <= tracked_count:
        raise ValueError("interpolation_neighbors exceeds the tracked point count")

    # Material associations are fixed from the common initial graph geometry.
    lift_indices, lift_weights = _lift_map(
        positions[0, 0],
        tracked_count,
        settings.interpolation_neighbors,
    )
    discrepancy_means = np.empty((particle_count, state_count, 3), dtype=float)
    discrepancy_variances = np.empty_like(discrepancy_means)
    update_counts: list[int] = []
    final_inlier_probabilities: list[float] = []
    for particle_index in range(particle_count):
        residual = (
            observed[:train_end]
            - positions[particle_index, :train_end, :tracked_count]
        )
        posterior = robust_random_walk_endpoint(
            residual,
            valid[:train_end],
            end_frame=train_end,
            process_variance=settings.process_std_m**2,
            observation_variance=settings.observation_std_m**2,
            initial_variance=settings.initial_std_m**2,
            inlier_prior=settings.inlier_prior,
            outlier_variance_multiplier=settings.outlier_variance_multiplier,
        )
        discrepancy_means[particle_index] = _lift_residual(
            posterior.mean[None],
            state_count,
            lift_indices,
            lift_weights,
            maximum_norm=settings.maximum_discrepancy_m,
        )[0]
        discrepancy_variances[particle_index] = lift_isotropic_discrepancy_variance(
            posterior.variance,
            state_count,
            lift_indices,
            lift_weights,
        )
        update_counts.append(int(np.sum(posterior.update_count)))
        supported = posterior.update_count > 0
        final_inlier_probabilities.append(
            float(np.mean(posterior.final_inlier_probability[supported]))
            if np.any(supported)
            else 0.0
        )

    endpoint = train_end - 1
    endpoint_positions = positions[:, endpoint].copy()
    endpoint_velocities = velocities[:, endpoint].copy()
    pairwise_rmse = []
    for first in range(particle_count):
        for second in range(first + 1, particle_count):
            pairwise_rmse.append(
                float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                endpoint_positions[first]
                                - endpoint_positions[second]
                            )
                        )
                    )
                )
            )
    diagnostics = {
        "causal_fit_window": [context.o_minus.frame_start, train_end],
        "future_frames_read_by_estimator": 0,
        "particle_state_source": "official PhysTwin replay through O-",
        "discrepancy_role": "separate readout/process discrepancy; not injected into state",
        "discrepancy_filter": asdict(settings),
        "particle_update_counts": update_counts,
        "particle_mean_final_inlier_probability": final_inlier_probabilities,
        "maximum_pairwise_endpoint_rmse_m": max(pairwise_rmse, default=0.0),
    }
    diagnostics.update(metadata or {})
    identifiers = particle_ids or tuple(
        f"theta_{index:04d}" for index in range(particle_count)
    )
    return TwinBelief(
        context=context,
        endpoint_frame=endpoint,
        particle_ids=identifiers,
        theta_names=theta_names,
        endpoint_position_m=endpoint_positions,
        endpoint_velocity_mps=endpoint_velocities,
        theta=particle_values,
        discrepancy_mean_m=discrepancy_means,
        discrepancy_variance_m2=discrepancy_variances,
        weights=particle_weights,
        metadata=diagnostics,
    )


def export_official_phystwin_twin_belief(
    backend: OfficialPhysTwinBackend,
    *,
    context: CausalContext,
    config: BPTBeliefExportConfig | None = None,
) -> TwinBelief:
    """Replay every selected theta particle through O- in official Warp."""

    if context.case_id != backend.case_name:
        raise ValueError("causal context case does not match the PhysTwin backend")
    if context.o_minus.frame_start != 0 or context.o_minus.frame_stop != backend.train_end_frame:
        raise ValueError("causal context O- does not match the backend training split")
    from bayesian_phystwin.phystwin_state_injection import (
        _initialize_simulator,
        _released_self_collision_for_case,
        _rollout_initial,
    )

    self_collision = (
        _released_self_collision_for_case(backend.case_name)
        if backend.config.self_collision is None
        else backend.config.self_collision
    )
    simulator, torch, wp, _ = _initialize_simulator(
        backend.official_repo,
        backend.data,
        backend.optimal,
        backend.checkpoint_path,
        backend.graph,
        num_surface_points=backend.original_count + len(backend.surface_points),
        original_count=backend.original_count,
        dt=backend.config.dt,
        num_substeps=backend.config.num_substeps,
        self_collision=bool(self_collision),
        deterministic_spring_forces=backend.config.deterministic_spring_forces,
        spring_parameterization="grouped",
        device=backend.config.device,
    )
    replay_positions = []
    replay_velocities = []
    try:
        for particle in backend.particles.log_scales:
            with torch.no_grad():
                simulator.group_log_scale_tensor.copy_(
                    torch.as_tensor(
                        particle,
                        dtype=torch.float32,
                        device=backend.config.device,
                    )
                )
            positions, velocities = _rollout_initial(
                simulator,
                wp,
                frame_count=backend.train_end_frame,
            )
            replay_positions.append(positions)
            replay_velocities.append(velocities)
    finally:
        del simulator
        gc.collect()
        torch.cuda.empty_cache()

    valid = _target_validity(backend.visible, backend.motion_valid)
    particle_ids = tuple(
        "grid_" + "_".join(map(str, grid_index))
        for grid_index in backend.particles.grid_indices
    )
    return build_twin_belief_from_replays(
        context=context,
        replay_positions_m=np.stack(replay_positions),
        replay_velocities_mps=np.stack(replay_velocities),
        observed_positions_m=backend.object_points,
        observed_valid=valid,
        theta=backend.particles.log_scales,
        theta_names=("object_spring_log_scale", "controller_spring_log_scale"),
        weights=backend.particles.weights,
        particle_ids=particle_ids,
        metadata={
            "profile_path": str(backend.profile_path.resolve()),
            "profile_weight_key": backend.particles.source_weight_key,
            "profile_retained_probability_mass": backend.particles.retained_probability_mass,
            "official_backend": backend.default_manifest(),
        },
        config=config,
    )
