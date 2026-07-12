"""Predictive convergence audit for reduced Bayesian-PhysTwin support."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any

import numpy as np

from causal4d.contracts import TwinBelief
from causal4d.intervention_abduction import FactualAbductionConfig
from causal4d.parameter_support import SupportMethod, reduce_parameter_support
from causal4d.rollout_bank import JointRolloutBank


@dataclass(frozen=True)
class ParameterSupportAuditConfig:
    counts: tuple[int, ...] = (4, 8, 16, 32, 81)
    methods: tuple[SupportMethod, ...] = ("top_mass", "weighted_coreset")
    prefix_frame_count: int = 7
    confidence_level: float = 0.90
    energy_samples: int = 16
    energy_seed: int = 20260712
    mean_stability_tolerance_m: float = 5e-4
    variance_relative_stability_tolerance: float = 0.05

    def __post_init__(self) -> None:
        if not self.counts or any(count < 1 for count in self.counts):
            raise ValueError("support counts must be positive")
        if tuple(sorted(set(self.counts))) != self.counts:
            raise ValueError("support counts must be unique and increasing")
        if not self.methods or len(set(self.methods)) != len(self.methods):
            raise ValueError("support methods must be unique")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must lie in (0, 1)")
        if self.energy_samples < 2:
            raise ValueError("energy_samples must be at least two")
        if self.mean_stability_tolerance_m <= 0.0:
            raise ValueError("mean stability tolerance must be positive")
        if self.variance_relative_stability_tolerance <= 0.0:
            raise ValueError("variance stability tolerance must be positive")


def _reduced_bank(
    bank: JointRolloutBank,
    indices: np.ndarray,
    weights: np.ndarray,
) -> JointRolloutBank:
    if len(indices) == len(bank.parameter_weights) and np.array_equal(
        indices,
        np.arange(len(indices)),
    ):
        return bank
    return JointRolloutBank(
        hypothesis_ids=bank.hypothesis_ids,
        hypothesis_metadata=bank.hypothesis_metadata,
        hypothesis_prior_weights=bank.hypothesis_prior_weights,
        parameter_particles=bank.parameter_particles[indices],
        parameter_weights=weights,
        trajectories=bank.trajectories[:, indices],
        variance_floor_m2=bank.variance_floor_m2,
        confidence_level=bank.confidence_level,
    )


def _posterior_moments(
    bank: JointRolloutBank,
    joint_weights: np.ndarray,
    discrepancy_mean_m: np.ndarray,
    discrepancy_variance_m2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    shape = bank.trajectories.shape[2:]
    mean = np.zeros(shape, dtype=np.float64)
    second = np.zeros(shape, dtype=np.float64)
    conditional = np.zeros(shape[1:], dtype=np.float64)
    for hypothesis_index in range(len(bank.hypothesis_ids)):
        for particle_index in range(len(bank.parameter_weights)):
            weight = float(joint_weights[hypothesis_index, particle_index])
            if weight <= 0.0:
                continue
            readout = (
                bank.trajectories[hypothesis_index, particle_index].astype(float)
                + discrepancy_mean_m[particle_index][None]
            )
            mean += weight * readout
            second += weight * np.square(readout)
            conditional += weight * (
                discrepancy_variance_m2[particle_index] + bank.variance_floor_m2
            )
    epistemic = np.maximum(second - np.square(mean), 0.0)
    variance = epistemic + conditional[None]
    return mean, np.maximum(variance, np.finfo(float).tiny)


def _gaussian_energy_score(
    mean: np.ndarray,
    variance: np.ndarray,
    truth: np.ndarray,
    valid: np.ndarray,
    *,
    sample_count: int,
    seed: int,
) -> float:
    selected_mean = mean[valid]
    selected_std = np.sqrt(variance[valid])
    selected_truth = truth[valid]
    rng = np.random.default_rng(seed)
    first_noise = rng.standard_normal((sample_count, 1, 3))
    second_noise = rng.standard_normal((sample_count, 1, 3))
    first = selected_mean[None] + first_noise * selected_std[None]
    second = selected_mean[None] + second_noise * selected_std[None]
    observation_distance = np.linalg.norm(
        first - selected_truth[None],
        axis=2,
    )
    pair_distance = np.linalg.norm(first - second, axis=2)
    return float(np.mean(observation_distance) - 0.5 * np.mean(pair_distance))


def _horizon_groups(start: int, stop: int) -> tuple[tuple[str, int, int], ...]:
    edges = np.linspace(start, stop, 4, dtype=int)
    return tuple(
        (name, int(edges[index]), int(edges[index + 1]))
        for index, name in enumerate(("early", "middle", "late"))
    )


def _predictive_metrics(
    mean: np.ndarray,
    variance: np.ndarray,
    truth: np.ndarray,
    valid: np.ndarray,
    *,
    start_frame: int,
    confidence_level: float,
    energy_samples: int,
    energy_seed: int,
) -> dict[str, Any]:
    selected = valid.copy()
    selected[:start_frame] = False
    if not np.any(selected):
        raise ValueError("support audit has no valid held-out point frames")
    coordinate_valid = np.repeat(selected[:, :, None], 3, axis=2)
    residual = mean - truth
    selected_residual = residual[coordinate_valid]
    selected_variance = variance[coordinate_valid]
    z_score = NormalDist().inv_cdf(0.5 * (1.0 + confidence_level))
    lower = mean - z_score * np.sqrt(variance)
    upper = mean + z_score * np.sqrt(variance)
    covered = (truth >= lower) & (truth <= upper)

    by_horizon = {}
    for name, group_start, group_stop in _horizon_groups(start_frame, len(truth)):
        group = selected[group_start:group_stop]
        group_coordinates = np.repeat(group[:, :, None], 3, axis=2)
        group_covered = covered[group_start:group_stop][group_coordinates]
        group_residual = residual[group_start:group_stop][group_coordinates]
        group_variance = variance[group_start:group_stop][group_coordinates]
        by_horizon[name] = {
            "frame_interval": [group_start, group_stop],
            "coverage": float(np.mean(group_covered)),
            "nees": float(np.mean(np.square(group_residual) / group_variance)),
            "coordinate_rmse_m": float(np.sqrt(np.mean(np.square(group_residual)))),
        }

    return {
        "coordinate_rmse_m": float(np.sqrt(np.mean(np.square(selected_residual)))),
        "track_error_m": float(np.mean(np.linalg.norm(residual[selected], axis=1))),
        "coverage": float(np.mean(covered[coordinate_valid])),
        "coverage_error": float(
            abs(np.mean(covered[coordinate_valid]) - confidence_level)
        ),
        "nees": float(np.mean(np.square(selected_residual) / selected_variance)),
        "gaussian_nll": float(
            np.mean(
                0.5
                * (
                    np.log(2.0 * np.pi * selected_variance)
                    + np.square(selected_residual) / selected_variance
                )
            )
        ),
        "gaussian_energy_score_m": _gaussian_energy_score(
            mean,
            variance,
            truth,
            selected,
            sample_count=energy_samples,
            seed=energy_seed,
        ),
        "mean_interval_width_m": float(np.mean((upper - lower)[coordinate_valid])),
        "valid_point_frames": int(np.sum(selected)),
        "by_horizon": by_horizon,
    }


def audit_parameter_support(
    bank: JointRolloutBank,
    belief: TwinBelief,
    observations_from_endpoint_m: np.ndarray,
    observation_mask: np.ndarray,
    *,
    config: ParameterSupportAuditConfig | None = None,
    abduction_config: FactualAbductionConfig | None = None,
) -> dict[str, Any]:
    """Compare parameter reductions against one full-support rollout bank."""

    settings = config or ParameterSupportAuditConfig()
    likelihood = abduction_config or FactualAbductionConfig()
    observations = np.asarray(observations_from_endpoint_m, dtype=float)
    mask = np.asarray(observation_mask, dtype=bool)
    if observations.shape != bank.trajectories.shape[2:]:
        raise ValueError("observations must match the rollout bank")
    if mask.shape != observations.shape[:2]:
        raise ValueError("observation_mask must have shape (T, N)")
    if belief.theta.shape != bank.parameter_particles.shape or not np.array_equal(
        belief.theta,
        bank.parameter_particles,
    ):
        raise ValueError("full TwinBelief theta must match the rollout bank")
    if not np.allclose(
        belief.weights,
        bank.parameter_weights,
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError("full TwinBelief weights must match the rollout bank")
    if max(settings.counts) > len(bank.parameter_weights):
        raise ValueError("support audit count exceeds the full rollout bank")
    if settings.prefix_frame_count >= bank.frame_count:
        raise ValueError("prefix must leave held-out rollout frames")

    candidates: list[dict[str, Any]] = []
    moments: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for method in settings.methods:
        for requested_count in settings.counts:
            start_time = time.perf_counter()
            reduction = reduce_parameter_support(
                bank.parameter_particles,
                bank.parameter_weights,
                maximum_count=requested_count,
                method=method,
            )
            selected = reduction.indices
            reduced = _reduced_bank(bank, selected, reduction.weights)
            discrepancy_mean = belief.discrepancy_mean_m[
                selected,
                : bank.node_count,
            ]
            discrepancy_variance = belief.discrepancy_variance_m2[
                selected,
                : bank.node_count,
            ]
            joint_weights = reduced.update_from_observations(
                observations,
                prefix_frame_count=settings.prefix_frame_count,
                scale_m=likelihood.observation_scale_m,
                likelihood_power=likelihood.likelihood_power,
                dynamic_likelihood_weight=likelihood.dynamic_likelihood_weight,
                degrees_of_freedom=likelihood.degrees_of_freedom,
                mask=mask,
                particle_discrepancy_m=discrepancy_mean,
                particle_discrepancy_variance_m2=discrepancy_variance,
            )
            mean, variance = _posterior_moments(
                reduced,
                joint_weights,
                discrepancy_mean,
                discrepancy_variance,
            )
            metrics = _predictive_metrics(
                mean,
                variance,
                observations,
                mask,
                start_frame=settings.prefix_frame_count,
                confidence_level=settings.confidence_level,
                energy_samples=settings.energy_samples,
                energy_seed=settings.energy_seed,
            )
            key = (method, requested_count)
            moments[key] = (mean, variance)
            candidates.append(
                {
                    **reduction.as_dict(),
                    "requested_count": requested_count,
                    "posterior_joint_effective_support": float(
                        1.0 / np.sum(np.square(joint_weights))
                    ),
                    "posterior_parameter_effective_support": float(
                        1.0 / np.sum(np.square(np.sum(joint_weights, axis=0)))
                    ),
                    "runtime_seconds": float(time.perf_counter() - start_time),
                    "predictive": metrics,
                }
            )

    full_key = ("top_mass", len(bank.parameter_weights))
    if full_key not in moments:
        raise ValueError("top_mass full support must be included as the reference")
    full_mean, full_variance = moments[full_key]
    stable_counts: dict[str, int | None] = {}
    for method in settings.methods:
        stable_counts[method] = None
        for candidate in candidates:
            if candidate["method"] != method:
                continue
            mean, variance = moments[(method, int(candidate["requested_count"]))]
            mean_error = float(np.sqrt(np.mean(np.square(mean - full_mean))))
            variance_error = float(
                np.linalg.norm(variance - full_variance)
                / max(np.linalg.norm(full_variance), np.finfo(float).tiny)
            )
            candidate["predictive_mean_rmse_vs_full_m"] = mean_error
            candidate["predictive_variance_relative_l2_vs_full"] = variance_error
            candidate["label_free_stable_vs_full"] = bool(
                mean_error <= settings.mean_stability_tolerance_m
                and variance_error <= settings.variance_relative_stability_tolerance
            )
            if stable_counts[method] is None and candidate["label_free_stable_vs_full"]:
                stable_counts[method] = int(candidate["requested_count"])

    return {
        "schema_version": 1,
        "audit": "causal4d_parameter_support_convergence_v1",
        "label_use": {
            "support_selection": "parameter posterior only",
            "stability_selection": "full-support predictive moments only",
            "held_out_targets": "diagnostic metrics only",
        },
        "config": asdict(settings),
        "abduction_likelihood": asdict(likelihood),
        "full_support_count": len(bank.parameter_weights),
        "full_support_prior_effective_support": float(
            1.0 / np.sum(np.square(bank.parameter_weights))
        ),
        "stable_counts": stable_counts,
        "candidates": candidates,
    }
