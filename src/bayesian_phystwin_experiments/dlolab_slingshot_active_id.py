"""Full-particle source qualification for matched-reset Slingshot probing."""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .dlolab_slingshot_belief import BIAS_STD_M, NOISE_STD_M, prior_weights
from .dlolab_slingshot_task_probe_dev import _posterior_weights

Array: TypeAlias = NDArray[Any]

PARTICLE_GROUPS = ((0, 9), (18, 27))
ACTIVE_FRACTION = 0.70
DRAW_COUNT = 4096
DRAW_SEED = 261209


def protocol() -> dict[str, Any]:
    return {
        "schema": "dlolab-slingshot-active-id-particle-source-v1",
        "role": "prospective_full_particle_probe_qualification_before_truth_worlds",
        "active_frontload_fraction": ACTIVE_FRACTION,
        "particle_count": 27,
        "reused_development_world_indices": list(range(9, 18)),
        "new_particle_groups": [list(group) for group in PARTICLE_GROUPS],
        "new_prefix_batch_count": 4,
        "prefix_steps": 300,
        "observation_frames": [139, 219, 299],
        "observed_rod_nodes": [3, 6, 8],
        "sphere_center_observed": True,
        "independent_noise_std_m": NOISE_STD_M,
        "shared_bias_std_m": BIAS_STD_M,
        "draw_count": DRAW_COUNT,
        "draw_seed": DRAW_SEED,
        "candidate_names": ["passive_original", "active_frontload_70"],
        "reward_table": "unchanged opened 27-particle seven-action source bank",
        "gate": {
            "all_new_prefixes_native_qualified": True,
            "minimum_active_gain_over_blind": 0.005,
            "minimum_active_gain_over_passive": 0.003,
            "minimum_oracle_headroom_captured": 0.25,
            "active_mutual_information_not_below_passive": True,
        },
        "continuous_truth_protocol_automatically_authorized": False,
        "truth_probe_generated": False,
        "truth_future_generated": False,
        "protected_data_read": False,
        "held_v8_read": False,
        "dlo4_dlo5_read": False,
        "official_dlo3_evaluation": False,
        "new_recordings": False,
        "gpu_work": False,
        "retry_authorized": False,
    }


def particle_task(group: int, batch: int) -> dict[str, Any]:
    if (
        type(group) is not int
        or type(batch) is not int
        or group not in range(len(PARTICLE_GROUPS))
        or batch not in range(2)
    ):
        raise ValueError("registered particle prefix task required")
    start, stop = PARTICLE_GROUPS[group]
    first = start + 8 * batch
    indices = list(range(first, min(first + 8, stop)))
    return {
        "kind": "particle_prefix_only",
        "name": f"particle-group-{group}-batch-{batch}",
        "group": group,
        "batch": batch,
        "world_indices": indices,
        "active_fraction": ACTIVE_FRACTION,
    }


def expected_value_screen(
    histories: Array,
    rewards: Array,
    *,
    draws: int = DRAW_COUNT,
    seed: int = DRAW_SEED,
) -> dict[str, Any]:
    feature = np.asarray(histories, dtype=np.float64)
    reward = np.asarray(rewards, dtype=np.float64)
    prior = np.asarray(prior_weights(), dtype=np.float64)
    if (
        feature.shape != (2, 27, 3, 4, 3)
        or reward.shape != (27, 7)
        or any(not np.isfinite(value).all() for value in (feature, reward))
        or type(draws) is not int
        or draws < 1
        or type(seed) is not int
    ):
        raise ValueError("complete full-particle histories and reward bank required")
    rng = np.random.default_rng(seed)
    bias = rng.normal(0, BIAS_STD_M, (draws, 1, 1, 3))
    noise = rng.normal(0, NOISE_STD_M, (draws, 3, 4, 3))
    prior_entropy = float(-np.sum(prior * np.log(prior)))
    blind_reward = float(np.max(prior @ reward))
    oracle_reward = float(prior @ np.max(reward, axis=1))
    rows: list[dict[str, float]] = []
    for candidate in range(2):
        bayes_reward = 0.0
        map_reward = 0.0
        posterior_entropy = 0.0
        action_changes = 0.0
        blind_action = int(np.argmax(prior @ reward))
        for world in range(27):
            for start in range(0, draws, 128):
                stop = min(start + 128, draws)
                observation = (
                    feature[candidate, world]
                    + bias[start:stop]
                    + noise[start:stop]
                )
                posterior = _posterior_weights(
                    observation, feature[candidate], prior
                )
                bayes_choice = np.argmax(posterior @ reward, axis=1)
                map_world = np.argmax(posterior, axis=1)
                map_choice = np.argmax(reward[map_world], axis=1)
                scale = float(prior[world]) / draws
                bayes_reward += scale * float(np.sum(reward[world, bayes_choice]))
                map_reward += scale * float(np.sum(reward[world, map_choice]))
                action_changes += scale * float(np.sum(bayes_choice != blind_action))
                posterior_entropy += scale * float(
                    np.sum(-posterior * np.log(np.maximum(posterior, 1e-300)))
                )
        bayes = float(bayes_reward)
        entropy = float(posterior_entropy)
        rows.append(
            {
                "expected_bayes_reward": bayes,
                "expected_map_reward": float(map_reward),
                "gain_over_blind": float(bayes - blind_reward),
                "probability_action_differs_from_blind": float(action_changes),
                "posterior_entropy_nats": entropy,
                "mutual_information_nats": float(prior_entropy - entropy),
            }
        )
    active_gain = rows[1]["gain_over_blind"]
    oracle_headroom = oracle_reward - blind_reward
    checks = {
        "active_gain_over_blind_at_least_0_005": bool(active_gain >= 0.005),
        "active_gain_over_passive_at_least_0_003": bool(
            rows[1]["expected_bayes_reward"]
            - rows[0]["expected_bayes_reward"]
            >= 0.003
        ),
        "captures_at_least_25pct_oracle_headroom": bool(
            active_gain >= 0.25 * oracle_headroom
        ),
        "active_mutual_information_not_below_passive": bool(
            rows[1]["mutual_information_nats"]
            >= rows[0]["mutual_information_nats"]
        ),
    }
    return {
        "candidate_names": ["passive_original", "active_frontload_70"],
        "candidates": rows,
        "best_blind_action": int(np.argmax(prior @ reward)),
        "best_blind_reward": blind_reward,
        "oracle_reward": oracle_reward,
        "oracle_headroom": oracle_headroom,
        "checks": checks,
        "particle_value_gate_passed": bool(all(checks.values())),
        "continuous_truth_protocol_automatically_authorized": False,
        "truth_future_generated": False,
    }
