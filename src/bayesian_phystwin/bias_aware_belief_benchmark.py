"""Controlled benchmark for bias-aware guarded state updates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .bias_aware_belief import (
    BiasAwareStateUpdateConfig,
    apply_regret_guard,
    fit_source_regret_certificate,
    update_bias_aware_state,
)


PROTOCOL_ID = "bias-aware-guarded-belief-synthetic-v1"


@dataclass(frozen=True)
class BiasAwareBeliefBenchmarkConfig:
    """Fixed synthetic controls for identifiability and safe routing."""

    seed: int = 20260720
    trial_count: int = 128
    point_count: int = 12
    view_count: int = 4
    source_group_count: int = 12
    source_samples_per_group: int = 12
    target_sample_count: int = 512

    def __post_init__(self) -> None:
        if self.trial_count < 16:
            raise ValueError("trial_count must be at least 16")
        if self.point_count < 6:
            raise ValueError("point_count must be at least 6")
        if self.view_count < 2:
            raise ValueError("view_count must be at least two")
        if self.source_group_count < 3:
            raise ValueError("source_group_count must be at least three")
        if self.source_samples_per_group < 3:
            raise ValueError("source group size must be at least three")
        if self.target_sample_count < 32:
            raise ValueError("target sample count must be at least 32")


def _state_update_config() -> BiasAwareStateUpdateConfig:
    return BiasAwareStateUpdateConfig(
        observation_std_m=0.0015,
        anchor_std_m=0.0005,
        state_prior_std_m=0.05,
        shared_bias_prior_std_m=0.05,
        camera_bias_prior_std_m=0.02,
        effective_samples_per_view=12.0,
    )


def _camera_observation(
    state_field_m: np.ndarray,
    shared_bias_m: np.ndarray,
    *,
    view_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    point_count = len(state_field_m)
    common_noise = rng.normal(scale=0.0005, size=(point_count, 3))
    observation = np.empty((view_count, point_count, 3), dtype=np.float64)
    for view in range(view_count):
        view_bias = rng.normal(scale=0.0003, size=3)
        independent_noise = rng.normal(scale=0.0002, size=(point_count, 3))
        observation[view] = (
            state_field_m
            + shared_bias_m
            + common_noise
            + view_bias
            + independent_noise
        )
    return observation


def _run_identifiability_controls(
    config: BiasAwareBeliefBenchmarkConfig,
    rng: np.random.Generator,
) -> dict[str, Any]:
    point_count = config.point_count
    available = np.ones((config.view_count, point_count), dtype=bool)
    global_basis = np.ones((point_count, 1), dtype=np.float64)
    local_basis = np.linspace(-1.0, 1.0, point_count)[:, None]
    shared_basis = global_basis.copy()
    update_config = _state_update_config()
    confounded_fallbacks = 0
    anchored_error = []
    anchored_camera_only_error = []
    local_error = []

    for _ in range(config.trial_count):
        state_vector = rng.normal(scale=0.008, size=3)
        bias_vector = rng.normal(scale=0.015, size=3)
        global_observation = _camera_observation(
            np.repeat(state_vector[None], point_count, axis=0),
            np.repeat(bias_vector[None], point_count, axis=0),
            view_count=config.view_count,
            rng=rng,
        )
        confounded = update_bias_aware_state(
            global_observation,
            available,
            global_basis,
            shared_basis,
            config=update_config,
        )
        confounded_fallbacks += int(
            not confounded.accepted
            and confounded.reason == "unanchored-common-mode-ambiguity"
        )
        anchor = state_vector + rng.normal(scale=0.0005, size=3)
        anchored = update_bias_aware_state(
            global_observation,
            available,
            global_basis,
            shared_basis,
            anchor_innovation_m=anchor[None],
            anchor_state_basis=np.ones((1, 1)),
            config=update_config,
        )
        anchored_error.append(
            np.linalg.norm(anchored.state_coefficients_m[0] - state_vector)
        )
        anchored_camera_only_error.append(
            np.linalg.norm(np.mean(global_observation, axis=(0, 1)) - state_vector)
        )

        local_coefficient = rng.normal(scale=0.008, size=3)
        local_state = local_basis * local_coefficient[None]
        local_observation = _camera_observation(
            local_state,
            np.repeat(bias_vector[None], point_count, axis=0),
            view_count=config.view_count,
            rng=rng,
        )
        local = update_bias_aware_state(
            local_observation,
            available,
            local_basis,
            shared_basis,
            config=update_config,
        )
        local_error.append(
            np.linalg.norm(local.state_coefficients_m[0] - local_coefficient)
        )

    return {
        "confounded_exact_fallback_rate": confounded_fallbacks / config.trial_count,
        "anchored_state_rmse_m": float(
            np.sqrt(np.mean(np.square(anchored_error)))
        ),
        "camera_as_state_rmse_m": float(
            np.sqrt(np.mean(np.square(anchored_camera_only_error)))
        ),
        "action_local_state_rmse_m": float(
            np.sqrt(np.mean(np.square(local_error)))
        ),
    }


def _run_correlation_control(
    config: BiasAwareBeliefBenchmarkConfig,
) -> dict[str, float]:
    basis = np.linspace(-1.0, 1.0, config.point_count)[:, None]
    one = np.zeros((1, config.point_count, 3), dtype=np.float64)
    one[0, :, 0] = 0.01 * basis[:, 0]
    duplicated = np.repeat(one, config.view_count, axis=0)
    update_config = _state_update_config()
    single = update_bias_aware_state(
        one,
        np.ones(one.shape[:2], dtype=bool),
        basis,
        np.zeros((config.point_count, 0)),
        config=update_config,
    )
    repeated = update_bias_aware_state(
        duplicated,
        np.ones(duplicated.shape[:2], dtype=bool),
        basis,
        np.zeros((config.point_count, 0)),
        config=update_config,
    )
    single_variance = float(single.posterior_covariance_m2[0, 0])
    repeated_variance = float(repeated.posterior_covariance_m2[0, 0])
    return {
        "single_view_state_variance_m2": single_variance,
        "duplicated_view_state_variance_m2": repeated_variance,
        "duplicated_to_single_variance_ratio": repeated_variance / single_variance,
    }


def _regret_function(
    features: np.ndarray,
    group_effect: np.ndarray,
    noise: np.ndarray,
) -> np.ndarray:
    anchor, identifiable_support, physical_snr = features.T
    return (
        0.008
        - 0.016 * anchor
        - 0.012 * identifiable_support
        - 0.002 * physical_snr
        + group_effect
        + noise
    )


def _run_regret_guard_control(
    config: BiasAwareBeliefBenchmarkConfig,
    rng: np.random.Generator,
) -> dict[str, Any]:
    source_count = config.source_group_count * config.source_samples_per_group
    source_groups = np.repeat(
        np.arange(config.source_group_count), config.source_samples_per_group
    )
    source_features = np.column_stack(
        (
            rng.integers(0, 2, size=source_count),
            rng.integers(0, 2, size=source_count),
            rng.uniform(0.5, 3.0, size=source_count),
        )
    )
    group_offsets = rng.normal(scale=0.001, size=config.source_group_count)
    source_regret = _regret_function(
        source_features,
        group_offsets[source_groups],
        rng.normal(scale=0.001, size=source_count),
    )
    certificate = fit_source_regret_certificate(
        source_features,
        source_regret,
        [f"source-{group}" for group in source_groups],
        ridge_penalty=0.5,
    )

    target_features = np.column_stack(
        (
            rng.integers(0, 2, size=config.target_sample_count),
            rng.integers(0, 2, size=config.target_sample_count),
            rng.uniform(0.6, 2.9, size=config.target_sample_count),
        )
    )
    target_regret = _regret_function(
        target_features,
        rng.normal(scale=0.001, size=config.target_sample_count),
        rng.normal(scale=0.001, size=config.target_sample_count),
    )
    accepted = np.zeros(config.target_sample_count, dtype=bool)
    upper = np.empty(config.target_sample_count, dtype=np.float64)
    for index, feature in enumerate(target_features):
        decision = apply_regret_guard(
            np.asarray([0.0]),
            np.asarray([1.0]),
            feature,
            certificate,
        )
        accepted[index] = decision.candidate_accepted
        upper[index] = decision.upper_regret
    selected_regret = np.where(accepted, target_regret, 0.0)
    out_of_support = apply_regret_guard(
        np.asarray([0.0]),
        np.asarray([1.0]),
        np.asarray([1.0, 1.0, 6.0]),
        certificate,
    )
    return {
        "source_group_count": certificate.source_group_count,
        "upper_residual_quantile_m": certificate.upper_residual_quantile,
        "target_acceptance_rate": float(np.mean(accepted)),
        "target_upper_bound_coverage": float(np.mean(target_regret <= upper)),
        "accepted_harmful_rate": (
            None if not np.any(accepted) else float(np.mean(target_regret[accepted] > 0.0))
        ),
        "selected_mean_regret_m": float(np.mean(selected_regret)),
        "out_of_support_exact_fallback": bool(
            not out_of_support.candidate_accepted
            and out_of_support.reason
            == "outside-source-support-exact-baseline-fallback"
        ),
    }


def run_bias_aware_belief_benchmark(
    config: BiasAwareBeliefBenchmarkConfig | None = None,
) -> dict[str, Any]:
    """Run all controls without reading empirical trajectories or outcomes."""

    cfg = config or BiasAwareBeliefBenchmarkConfig()
    rng = np.random.default_rng(cfg.seed)
    identifiability = _run_identifiability_controls(cfg, rng)
    correlation = _run_correlation_control(cfg)
    guard = _run_regret_guard_control(cfg, rng)
    gates = {
        "confounded_worlds_fall_back": (
            identifiability["confounded_exact_fallback_rate"] == 1.0
        ),
        "independent_anchor_beats_camera_as_state": (
            identifiability["anchored_state_rmse_m"]
            < 0.25 * identifiability["camera_as_state_rmse_m"]
        ),
        "action_local_state_is_recovered": (
            identifiability["action_local_state_rmse_m"] < 0.002
        ),
        "duplicated_views_do_not_shrink_variance": (
            correlation["duplicated_to_single_variance_ratio"] >= 0.999
        ),
        "regret_bound_covers_at_nominal_rate": (
            guard["target_upper_bound_coverage"] >= 0.90
        ),
        "accepted_updates_are_nonharmful": guard["accepted_harmful_rate"] == 0.0,
        "out_of_support_is_exact_fallback": guard[
            "out_of_support_exact_fallback"
        ],
    }
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "synthetic-mechanism-control-only",
        "config": asdict(cfg),
        "identifiability": identifiability,
        "correlation": correlation,
        "regret_guard": guard,
        "gates": gates,
        "all_gates_pass": bool(all(gates.values())),
        "claim_boundary": (
            "Controls implementation and identifiability logic only; no empirical "
            "accuracy, calibration, or state-of-the-art claim."
        ),
    }


def write_bias_aware_belief_benchmark(
    result: dict[str, Any], output_path: str | Path
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "BiasAwareBeliefBenchmarkConfig",
    "PROTOCOL_ID",
    "run_bias_aware_belief_benchmark",
    "write_bias_aware_belief_benchmark",
]
