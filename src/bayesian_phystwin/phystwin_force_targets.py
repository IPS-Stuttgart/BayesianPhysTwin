"""Robust source-only inverse-dynamics targets for generalized-force learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .phystwin_graph_discrepancy import graph_smoothed_discrepancy_posterior


@dataclass(frozen=True)
class ResidualAccelerationEstimate:
    """Local-polynomial residual acceleration and its scalar uncertainty."""

    mean_mps2: np.ndarray
    variance_m2ps4: np.ndarray
    observed: np.ndarray
    robust_weight: np.ndarray
    temporal_support: np.ndarray
    end_frame: int
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class GeneralizedForceTargets:
    """Mass-scaled residual forces in the native simulator unit system."""

    mean_sim: np.ndarray
    variance_sim2: np.ndarray
    observed: np.ndarray
    training_weight: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class SimulatorForceScale:
    """Prefix-only robust force scale in native Warp simulator units."""

    value_sim: float
    diagnostics: dict[str, Any]


def _validate_trajectory(
    observed_m: np.ndarray,
    baseline_m: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = np.asarray(observed_m, dtype=float)
    baseline = np.asarray(baseline_m, dtype=float)
    mask = np.asarray(valid, dtype=bool)
    if (
        observed.ndim != 3
        or observed.shape[-1] != 3
        or baseline.shape != observed.shape
        or mask.shape != observed.shape[:2]
    ):
        raise ValueError(
            "observed/baseline/valid must have shapes (T,N,3)/(T,N,3)/(T,N)"
        )
    if not np.all(np.isfinite(baseline)):
        raise ValueError("baseline trajectory must be finite")
    if not np.all(np.isfinite(observed[mask])):
        raise ValueError("valid observations must be finite")
    return observed, baseline, mask


def estimate_residual_acceleration(
    observed_m: np.ndarray,
    baseline_m: np.ndarray,
    valid: np.ndarray,
    *,
    frame_dt_s: float,
    end_frame: int | None = None,
    prior_reliability: np.ndarray | None = None,
    window_radius: int = 3,
    huber_delta_m: float = 0.003,
    ridge: float = 1.0e-10,
    robust_iterations: int = 4,
    variance_floor_m2ps4: float = 1.0e-6,
    causal_window: bool = False,
) -> ResidualAccelerationEstimate:
    """Estimate residual acceleration without reading beyond ``end_frame``.

    A quadratic local polynomial is fitted independently at every node and
    frame. The state innovation enters this robust fit exactly once. Supplied
    prior reliability only controls the initial observation weights and must
    therefore be constructed from residual-independent perception cues.
    """

    observed, baseline, mask = _validate_trajectory(
        observed_m, baseline_m, valid
    )
    stop = len(observed) if end_frame is None else int(end_frame)
    if not 3 <= stop <= len(observed):
        raise ValueError("end_frame must leave at least three frames")
    if frame_dt_s <= 0.0 or window_radius < 1:
        raise ValueError("frame_dt_s and window_radius must be positive")
    if (
        huber_delta_m <= 0.0
        or ridge <= 0.0
        or robust_iterations < 1
        or variance_floor_m2ps4 <= 0.0
    ):
        raise ValueError("robust fit scales must be positive")
    reliability = (
        np.ones(mask.shape, dtype=float)
        if prior_reliability is None
        else np.asarray(prior_reliability, dtype=float)
    )
    if reliability.shape != mask.shape or not np.all(
        np.isfinite(reliability)
    ):
        raise ValueError("prior_reliability must be a finite (T,N) array")
    if np.any((reliability < 0.0) | (reliability > 1.0)):
        raise ValueError("prior_reliability must lie in [0,1]")

    residual = observed[:stop] - baseline[:stop]
    selected_mask = mask[:stop]
    selected_reliability = reliability[:stop]
    frames, nodes = selected_mask.shape
    acceleration = np.zeros((frames, nodes, 3), dtype=float)
    variance = np.full((frames, nodes), np.inf, dtype=float)
    robust_weight = np.zeros((frames, nodes), dtype=float)
    support = np.zeros((frames, nodes), dtype=np.int32)
    fitted = np.zeros((frames, nodes), dtype=bool)

    identity = np.eye(3, dtype=float)
    inverse_rhs = np.broadcast_to(
        np.array([0.0, 0.0, 1.0], dtype=float),
        (nodes, 3),
    )
    for center in range(frames):
        start = max(0, center - window_radius)
        stop_local = center + 1 if causal_window else min(
            frames,
            center + window_radius + 1,
        )
        local = np.arange(start, stop_local)
        time = (local - center).astype(float) * frame_dt_s
        design = np.stack(
            (np.ones_like(time), time, 0.5 * np.square(time)),
            axis=1,
        )
        base_weight = (
            selected_reliability[local]
            * selected_mask[local].astype(float)
        )
        support_count = np.sum(base_weight > 0.0, axis=0)
        eligible = support_count >= 3
        if not np.any(eligible):
            continue

        local_residual = residual[local]
        weights = base_weight.copy()
        coefficients = np.zeros((nodes, 3, 3), dtype=float)
        fit_residual = np.zeros_like(local_residual)
        for _ in range(robust_iterations):
            normal = np.einsum(
                "la,ln,lb->nab",
                design,
                weights,
                design,
                optimize=True,
            )
            normal += ridge * identity[None]
            right_hand_side = np.einsum(
                "la,ln,lnc->nac",
                design,
                weights,
                local_residual,
                optimize=True,
            )
            coefficients = np.linalg.solve(normal, right_hand_side)
            fit_residual = local_residual - np.einsum(
                "la,nac->lnc",
                design,
                coefficients,
                optimize=True,
            )
            magnitude = np.linalg.norm(fit_residual, axis=2)
            robust = np.minimum(
                1.0,
                huber_delta_m / np.maximum(magnitude, 1.0e-12),
            )
            weights = base_weight * robust

        effective = np.sum(weights, axis=0)
        eligible &= effective > 0.0
        if not np.any(eligible):
            continue
        normal = np.einsum(
            "la,ln,lb->nab",
            design,
            weights,
            design,
            optimize=True,
        )
        normal += ridge * identity[None]
        inverse_acceleration_variance = np.linalg.solve(
            normal,
            inverse_rhs[:, :, None],
        )[:, 2, 0]
        degrees = np.maximum(effective - 3.0, 1.0)
        residual_variance = np.sum(
            weights[:, :, None] * np.square(fit_residual),
            axis=(0, 2),
        ) / (3.0 * degrees)
        estimated_variance = np.maximum(
            residual_variance * inverse_acceleration_variance,
            variance_floor_m2ps4,
        )
        acceleration[center, eligible] = coefficients[eligible, 2]
        variance[center, eligible] = estimated_variance[eligible]
        robust_weight[center, eligible] = (
            effective[eligible] / np.sum(base_weight[:, eligible], axis=0)
        )
        support[center, eligible] = support_count[eligible]
        fitted[center, eligible] = True

    finite_variance = variance[fitted]
    diagnostics = {
        "fit_fraction": float(np.mean(fitted)),
        "fit_count": int(np.sum(fitted)),
        "coordinate_count": int(frames * nodes),
        "median_temporal_support": (
            float(np.median(support[fitted])) if np.any(fitted) else 0.0
        ),
        "median_robust_weight": (
            float(np.median(robust_weight[fitted])) if np.any(fitted) else 0.0
        ),
        "median_standard_deviation_mps2": (
            float(np.median(np.sqrt(finite_variance)))
            if len(finite_variance)
            else float("inf")
        ),
        "future_frames_used": False,
        "future_frames_used_per_target": not causal_window,
        "causal_window": causal_window,
        "innovation_used_once": True,
        "solver": "batched_nodes_v1",
    }
    return ResidualAccelerationEstimate(
        mean_mps2=acceleration,
        variance_m2ps4=variance,
        observed=fitted,
        robust_weight=robust_weight,
        temporal_support=support,
        end_frame=stop,
        diagnostics=diagnostics,
    )


def graph_smooth_residual_acceleration(
    estimate: ResidualAccelerationEstimate,
    laplacian: Any,
    *,
    prior_strength: float,
    ridge: float = 1.0e-8,
    covariance_probes: int = 8,
    covariance_seed: int = 20260724,
) -> ResidualAccelerationEstimate:
    """Lift noisy acceleration targets through one fixed spring-graph prior."""

    node_count = int(laplacian.shape[0])
    if laplacian.shape != (node_count, node_count):
        raise ValueError("laplacian must be square")
    if estimate.mean_mps2.shape[1] > node_count:
        raise ValueError("laplacian does not cover observed target nodes")
    if covariance_probes < 1:
        raise ValueError("covariance_probes must be positive")
    frames = len(estimate.mean_mps2)
    mean = np.zeros((frames, node_count, 3), dtype=float)
    variance = np.full((frames, node_count), np.inf, dtype=float)
    observed = np.zeros((frames, node_count), dtype=bool)
    robust_weight = np.zeros((frames, node_count), dtype=float)
    support = np.zeros((frames, node_count), dtype=np.int32)
    solved = 0
    for frame in range(frames):
        selected = estimate.observed[frame]
        if not np.any(selected):
            continue
        if isinstance(laplacian, np.ndarray):
            observed_count = estimate.mean_mps2.shape[1]
            reference = float(
                np.median(estimate.variance_m2ps4[frame, selected])
            )
            weights = np.zeros(node_count, dtype=float)
            observed_indices = np.flatnonzero(selected)
            weights[observed_indices] = (
                reference / estimate.variance_m2ps4[frame, selected]
            )
            full_mean = np.zeros((node_count, 3), dtype=float)
            full_mean[:observed_count] = estimate.mean_mps2[frame]
            precision = (
                np.diag(weights)
                + 2.0 * prior_strength * (laplacian.T @ laplacian)
                + ridge * np.eye(node_count)
            )
            mean[frame] = np.linalg.solve(
                precision, weights[:, None] * full_mean
            )
            covariance = reference * np.linalg.inv(precision)
            variance[frame] = np.maximum(
                np.diag(covariance), np.finfo(float).eps
            )
        else:
            posterior = graph_smoothed_discrepancy_posterior(
                estimate.mean_mps2[frame],
                estimate.variance_m2ps4[frame],
                selected,
                laplacian,
                prior_strength=prior_strength,
                ridge=ridge,
                covariance_probes=covariance_probes,
                covariance_seed=covariance_seed + frame,
            )
            mean[frame] = posterior.mean
            if posterior.marginal_variance is None:
                raise RuntimeError("graph target covariance was not estimated")
            variance[frame] = np.maximum(
                posterior.marginal_variance, np.finfo(float).eps
            )
        observed[frame] = np.isfinite(variance[frame])
        robust_value = float(np.mean(estimate.robust_weight[frame, selected]))
        support_value = int(np.median(estimate.temporal_support[frame, selected]))
        robust_weight[frame, observed[frame]] = robust_value
        support[frame, observed[frame]] = support_value
        solved += 1
    diagnostics = dict(estimate.diagnostics)
    diagnostics.update(
        {
            "graph_smoothed": True,
            "graph_node_count": node_count,
            "graph_solved_frames": solved,
            "graph_prior_strength": float(prior_strength),
            "graph_covariance_probes": covariance_probes,
        }
    )
    return ResidualAccelerationEstimate(
        mean_mps2=mean,
        variance_m2ps4=variance,
        observed=observed,
        robust_weight=robust_weight,
        temporal_support=support,
        end_frame=estimate.end_frame,
        diagnostics=diagnostics,
    )


def robust_prefix_force_scale(
    estimate: ResidualAccelerationEstimate,
    masses_sim: np.ndarray,
    *,
    prefix_end_frame: int,
    node_quantile: float = 0.95,
    temporal_quantile: float = 0.90,
    minimum_scale_sim: float = 0.10,
    maximum_scale_sim: float = 50.0,
) -> SimulatorForceScale:
    """Estimate one case scale without reading its held-out suffix.

    PhysTwin assigns every released graph node unit simulation mass. Those
    values do not establish a kilogram scale, so generalized forces remain in
    Warp's native simulator units. A high spatial quantile is computed per
    prefix frame and then robustly aggregated over time.
    """

    masses = np.asarray(masses_sim, dtype=float)
    nodes = estimate.mean_mps2.shape[1]
    if masses.shape != (nodes,) or np.any(masses <= 0.0) or not np.all(
        np.isfinite(masses)
    ):
        raise ValueError("masses_sim must be a finite positive N-vector")
    if not 3 <= prefix_end_frame <= len(estimate.mean_mps2):
        raise ValueError("prefix_end_frame must cover at least three frames")
    if not 0.0 < node_quantile < 1.0 or not 0.0 < temporal_quantile < 1.0:
        raise ValueError("force-scale quantiles must lie in (0,1)")
    if (
        minimum_scale_sim <= 0.0
        or maximum_scale_sim < minimum_scale_sim
        or not np.isfinite(minimum_scale_sim)
        or not np.isfinite(maximum_scale_sim)
    ):
        raise ValueError("force-scale bounds must be finite and ordered")

    raw_force = estimate.mean_mps2 * masses[None, :, None]
    magnitude = np.linalg.norm(raw_force, axis=2)
    selected = (
        estimate.observed
        & np.isfinite(estimate.variance_m2ps4)
        & (estimate.variance_m2ps4 > 0.0)
    )
    frame_scales = []
    for frame in range(prefix_end_frame):
        frame_selected = selected[frame]
        if np.any(frame_selected):
            frame_scales.append(
                float(
                    np.quantile(
                        magnitude[frame, frame_selected],
                        node_quantile,
                    )
                )
            )
    if not frame_scales:
        raise ValueError("prefix contains no supported simulator-force targets")
    unclipped = float(np.quantile(frame_scales, temporal_quantile))
    scale = float(np.clip(unclipped, minimum_scale_sim, maximum_scale_sim))
    prefix_selected = selected[:prefix_end_frame]
    cap_fraction = float(
        np.mean(
            magnitude[:prefix_end_frame][prefix_selected]
            >= scale * (1.0 - 1.0e-6)
        )
    )
    return SimulatorForceScale(
        value_sim=scale,
        diagnostics={
            "unit_contract": "warp_simulator_generalized_force_not_newtons",
            "source": "allowed_prefix_residual_acceleration",
            "prefix_end_frame": int(prefix_end_frame),
            "node_quantile": float(node_quantile),
            "temporal_quantile": float(temporal_quantile),
            "minimum_scale_sim": float(minimum_scale_sim),
            "maximum_scale_sim": float(maximum_scale_sim),
            "unclipped_scale_sim": unclipped,
            "selected_scale_sim": scale,
            "prefix_cap_fraction": cap_fraction,
            "prefix_frame_scale_min_sim": float(np.min(frame_scales)),
            "prefix_frame_scale_median_sim": float(np.median(frame_scales)),
            "prefix_frame_scale_max_sim": float(np.max(frame_scales)),
            "future_frames_used": False,
            "innovation_reused_as_prior_reliability": False,
        },
    )


def acceleration_to_force_targets(
    estimate: ResidualAccelerationEstimate,
    masses_sim: np.ndarray,
    *,
    maximum_force_sim: float,
    minimum_training_weight: float = 1.0e-4,
) -> GeneralizedForceTargets:
    """Convert acceleration to bounded native-simulator force targets."""

    masses = np.asarray(masses_sim, dtype=float)
    nodes = estimate.mean_mps2.shape[1]
    if masses.shape != (nodes,) or np.any(masses <= 0.0) or not np.all(
        np.isfinite(masses)
    ):
        raise ValueError("masses_sim must be a finite positive N-vector")
    if maximum_force_sim <= 0.0 or minimum_training_weight <= 0.0:
        raise ValueError("force and training-weight scales must be positive")
    force = estimate.mean_mps2 * masses[None, :, None]
    variance = estimate.variance_m2ps4 * np.square(masses)[None]
    norm = np.linalg.norm(force, axis=2, keepdims=True)
    scale = np.minimum(
        1.0,
        maximum_force_sim / np.maximum(norm, 1.0e-12),
    )
    force = force * scale
    finite = estimate.observed & np.isfinite(variance) & (variance > 0.0)
    finite_values = variance[finite]
    unsupported_variance = max(
        maximum_force_sim**2,
        float(np.max(finite_values, initial=0.0)),
    )
    variance = variance.copy()
    variance[~finite] = unsupported_variance
    inverse_variance = np.zeros_like(variance)
    inverse_variance[finite] = 1.0 / variance[finite]
    if np.any(finite):
        reference = float(np.median(inverse_variance[finite]))
        inverse_variance[finite] /= max(reference, 1.0e-12)
    training_weight = (
        inverse_variance
        * estimate.robust_weight
        * np.minimum(1.0, estimate.temporal_support / 5.0)
    )
    training_weight[~finite] = 0.0
    training_weight[
        finite & (training_weight < minimum_training_weight)
    ] = minimum_training_weight
    diagnostics = dict(estimate.diagnostics)
    diagnostics.update(
        {
            "force_unit_contract": (
                "warp_simulator_generalized_force_not_newtons"
            ),
            "maximum_target_force_sim": float(
                np.max(np.linalg.norm(force, axis=2), initial=0.0)
            ),
            "median_target_force_sim": (
                float(np.median(np.linalg.norm(force[finite], axis=1)))
                if np.any(finite)
                else 0.0
            ),
            "nonzero_training_weight_fraction": float(
                np.mean(training_weight > 0.0)
            ),
            "target_cap_sim": float(maximum_force_sim),
            "unsupported_target_variance_sim2": unsupported_variance,
        }
    )
    return GeneralizedForceTargets(
        mean_sim=force,
        variance_sim2=variance,
        observed=finite,
        training_weight=training_weight,
        diagnostics=diagnostics,
    )
