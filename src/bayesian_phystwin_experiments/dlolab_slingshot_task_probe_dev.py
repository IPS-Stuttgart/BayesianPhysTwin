"""Bounded development screen for task-valued Slingshot diagnostic probes."""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .dlolab_slingshot_belief import (
    BIAS_STD_M,
    NOISE_STD_M,
    prior_weights,
)

Array: TypeAlias = NDArray[Any]

NEW_FRACTIONS = (0.60, 0.70)
CANDIDATE_NAMES = (
    "original_passive",
    "frontload_50_existing",
    "frontload_60_new",
    "frontload_70_new",
)
WORLD_INDICES = tuple(range(9, 18))
DRAW_COUNT = 8192
DRAW_SEED = 261206
ZERO_REWARD = 6.900000095367432


def protocol() -> dict[str, Any]:
    return {
        "schema": "dlolab-slingshot-task-probe-development-v1",
        "role": "bounded_open_source_development_screen_not_scientific_evidence",
        "candidate_names": list(CANDIDATE_NAMES),
        "new_frontload_fractions": list(NEW_FRACTIONS),
        "world_indices": list(WORLD_INDICES),
        "world_selection": "already-open x-offset-zero material slice",
        "task_rewards": "already-open frozen seven-action model-bank rewards",
        "native_work": "prefix-only new 60% and 70% probes",
        "prefix_steps": 300,
        "observation_frames": [139, 219, 299],
        "observed_rod_nodes": [3, 6, 8],
        "sphere_center_observed": True,
        "independent_noise_std_m": NOISE_STD_M,
        "shared_bias_std_m": BIAS_STD_M,
        "draw_count": DRAW_COUNT,
        "draw_seed": DRAW_SEED,
        "task_selector": "maximum expected posterior Bayes reward",
        "generic_information_selector": "maximum mutual information",
        "feasibility_gate": {
            "all_new_prefixes_native_qualified": True,
            "selected_probe_is_new": True,
            "minimum_gain_over_blind": 0.005,
            "minimum_gain_over_existing_50pct_probe": 0.001,
            "minimum_gain_fraction_of_oracle_headroom": 0.25,
        },
        "future_protocol_automatically_authorized": False,
        "truth_future_generated": False,
        "protected_data_read": False,
        "held_v8_read": False,
        "dlo4_dlo5_read": False,
        "official_dlo3_evaluation": False,
        "new_recordings": False,
        "gpu_work": False,
        "retry_authorized": False,
    }


def frontloaded_controls(original: Array, fraction: float) -> Array:
    value = np.asarray(original, dtype=np.float64)
    if (
        value.shape != (8, 3, 6)
        or original.dtype != np.float64
        or not np.isfinite(value).all()
        or not np.array_equal(value[5], value[7])
        or not np.all(value[:, 0] == value[5, 0])
        or not np.isfinite(fraction)
        or fraction not in NEW_FRACTIONS
    ):
        raise ValueError("registered shared-prefix controls and fraction required")
    result = value.copy()
    shifted = fraction * value[5, 1, :3]
    result[:, 0, :3] += shifted
    result[:, 1, :3] -= shifted
    if (
        not np.allclose(
            result[:, 0, :3] + result[:, 1, :3],
            value[:, 0, :3] + value[:, 1, :3],
            rtol=0,
            atol=1e-15,
        )
        or np.max(np.linalg.norm(result[:, :, :3], axis=-1)) > 0.1 + 1e-12
        or np.max(np.abs(result[:, :, 3:])) > 1.0
    ):
        raise ValueError("development probe exceeds the registered action envelope")
    return result


def new_probe_task(probe: int, batch: int) -> dict[str, Any]:
    if (
        type(probe) is not int
        or type(batch) is not int
        or probe not in range(len(NEW_FRACTIONS))
        or batch not in range(2)
    ):
        raise ValueError("registered development prefix task required")
    indices = list(range(9 + 8 * batch, min(17 + 8 * batch, 18)))
    return {
        "kind": "prefix_only",
        "name": f"frontload-{int(100 * NEW_FRACTIONS[probe])}-prefix-{batch}",
        "probe": probe,
        "fraction": NEW_FRACTIONS[probe],
        "batch": batch,
        "world_indices": indices,
    }


def conditional_prior() -> Array:
    result = np.asarray(prior_weights()[list(WORLD_INDICES)], dtype=np.float64)
    result /= result.sum()
    return result


def _posterior_weights(observations: Array, predictions: Array, prior: Array) -> Array:
    observed = np.asarray(observations, dtype=np.float64)
    predicted = np.asarray(predictions, dtype=np.float64)
    weight = np.asarray(prior, dtype=np.float64)
    if (
        observed.ndim != 4
        or predicted.ndim != 4
        or observed.shape[1:] != predicted.shape[1:]
        or predicted.shape[0] != weight.shape[0]
        or np.any(weight <= 0)
        or not np.isclose(weight.sum(), 1.0)
    ):
        raise ValueError("aligned finite observations, particles, and prior required")
    residual = (observed[:, None] - predicted[None]).reshape(
        len(observed), len(predicted), -1, 3
    )
    mean = residual.mean(axis=2)
    centered = residual - mean[:, :, None]
    noise_var = NOISE_STD_M**2
    shared_var = BIAS_STD_M**2
    count = residual.shape[2]
    distance = np.sum(centered**2, axis=(2, 3)) / noise_var
    distance += count * np.sum(mean**2, axis=2) / (noise_var + count * shared_var)
    log_weight = np.log(weight)[None] - 0.5 * distance
    log_weight -= np.max(log_weight, axis=1, keepdims=True)
    posterior = np.exp(log_weight)
    posterior /= posterior.sum(axis=1, keepdims=True)
    return np.asarray(posterior, dtype=np.float64)


def evaluate_candidates(
    histories: Array,
    rewards: Array,
    prior: Array,
    *,
    draws: int = DRAW_COUNT,
    seed: int = DRAW_SEED,
) -> dict[str, Any]:
    feature = np.asarray(histories, dtype=np.float64)
    reward = np.asarray(rewards, dtype=np.float64)
    weight = np.asarray(prior, dtype=np.float64)
    if (
        feature.shape != (len(CANDIDATE_NAMES), 9, 3, 4, 3)
        or reward.shape != (9, 7)
        or weight.shape != (9,)
        or any(not np.isfinite(row).all() for row in (feature, reward, weight))
        or np.any(weight <= 0)
        or not np.isclose(weight.sum(), 1.0)
        or type(draws) is not int
        or draws < 1
        or type(seed) is not int
    ):
        raise ValueError("complete development histories, rewards, and draws required")
    rng = np.random.default_rng(seed)
    bias = rng.normal(0, BIAS_STD_M, (draws, 1, 1, 3))
    noise = rng.normal(0, NOISE_STD_M, (draws, 3, 4, 3))
    prior_entropy = float(-np.sum(weight * np.log(weight)))
    blind_reward = float(np.max(weight @ reward))
    oracle_reward = float(weight @ np.max(reward, axis=1))
    rows: list[dict[str, float]] = []
    for candidate in range(len(CANDIDATE_NAMES)):
        bayes_reward = 0.0
        map_reward = 0.0
        map_accuracy = 0.0
        posterior_entropy = 0.0
        for world in range(9):
            for start in range(0, draws, 256):
                stop = min(start + 256, draws)
                observation = (
                    feature[candidate, world]
                    + bias[start:stop]
                    + noise[start:stop]
                )
                posterior = _posterior_weights(
                    observation, feature[candidate], weight
                )
                bayes_choice = np.argmax(posterior @ reward, axis=1)
                map_world = np.argmax(posterior, axis=1)
                map_choice = np.argmax(reward[map_world], axis=1)
                scale = weight[world] / draws
                bayes_reward += scale * float(np.sum(reward[world, bayes_choice]))
                map_reward += scale * float(np.sum(reward[world, map_choice]))
                map_accuracy += scale * float(np.sum(map_world == world))
                posterior_entropy += scale * float(
                    np.sum(-posterior * np.log(np.maximum(posterior, 1e-300)))
                )
        rows.append(
            {
                "expected_bayes_reward": bayes_reward,
                "expected_map_reward": map_reward,
                "gain_over_blind": bayes_reward - blind_reward,
                "map_accuracy": map_accuracy,
                "posterior_entropy_nats": posterior_entropy,
                "mutual_information_nats": prior_entropy - posterior_entropy,
            }
        )
    task_index = int(np.argmax([row["expected_bayes_reward"] for row in rows]))
    mi_index = int(np.argmax([row["mutual_information_nats"] for row in rows]))
    task_gain = rows[task_index]["gain_over_blind"]
    oracle_headroom = oracle_reward - blind_reward
    checks = {
        "selected_probe_is_new": task_index >= 2,
        "gain_over_blind_at_least_0_005": task_gain >= 0.005,
        "gain_over_existing_50pct_at_least_0_001": rows[task_index][
            "expected_bayes_reward"
        ]
        - rows[1]["expected_bayes_reward"]
        >= 0.001,
        "captures_at_least_25pct_oracle_headroom": task_gain
        >= 0.25 * oracle_headroom,
    }
    return {
        "candidate_names": list(CANDIDATE_NAMES),
        "candidates": rows,
        "best_blind_action": int(np.argmax(weight @ reward)),
        "best_blind_reward": blind_reward,
        "oracle_reward": oracle_reward,
        "oracle_headroom": oracle_headroom,
        "task_aware_probe_index": task_index,
        "task_aware_probe_name": CANDIDATE_NAMES[task_index],
        "generic_mi_probe_index": mi_index,
        "generic_mi_probe_name": CANDIDATE_NAMES[mi_index],
        "checks": checks,
        "value_feasibility_passed": all(checks.values()),
        "future_protocol_automatically_authorized": False,
        "truth_future_generated": False,
    }
