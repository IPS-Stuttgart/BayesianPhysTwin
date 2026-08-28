"""Source-only value of a later prefix with a frozen final-motion bank."""

from __future__ import annotations

from typing import Any

import numpy as np

from .dlolab_native import array_digest
from .dlolab_slingshot_batch import split_batch
from .dlolab_slingshot_belief import particle_worlds, prior_weights
from .dlolab_slingshot_cmaes import task_metrics
from .dlolab_slingshot_contact import POSITION_FIELDS
from .dlolab_slingshot_probe import material_information

FRAMES = (139, 319, 499)
WORLD_ORDER = (4, 4, 4, 0, 1, 2, 3, 5, 6, 7, 8)
REFERENCE_PAIRS = ((0, 5), (5, 6), (7, 5))
NAMES = (
    "retained_yaw_plus",
    "zero_final_translation",
    "half_final_translation",
    "final_steering_minus20deg",
    "final_steering_plus20deg",
    "retained_yaw_minus",
    "final_yaw_plus0_4",
    "incumbent_duplicate",
)
REWARD_BUDGET = 0.00025
POSITION_BUDGET_M = 0.001
PAIR_MARGIN = 2 * REWARD_BUDGET
ZERO_REWARD = 6.900000095367432


def prior() -> np.ndarray:
    value = prior_weights()[9:18]
    return value / value.sum()


def task(index: int) -> dict[str, Any]:
    if type(index) is not int or index not in range(11):
        raise ValueError("unregistered late-branch task")
    world_index = WORLD_ORDER[index]
    return {
        "index": index,
        "name": f"batch-{index:02d}-material-{world_index}",
        "source_world_index": world_index,
        "world": particle_worlds()[9 + world_index],
        "qualification_repeat": index in (1, 2),
    }


def protocol() -> dict[str, Any]:
    return {
        "schema": "dlolab-slingshot-late-branch-source-v1",
        "role": "finite_source_feasibility_not_independent_method_evaluation",
        "tasks": [task(i) for i in range(11)],
        "action_names": list(NAMES),
        "unique_actions": 7,
        "native_batches": 11,
        "native_trajectories": 88,
        "branch_native_step": 500,
        "observation_frames": list(FRAMES),
        "observed_rod_nodes": [3, 6, 8],
        "sphere_center_observed": True,
        "observed_cube": False,
        "same_first_two_native_macros": True,
        "native_horizon_steps": 900,
        "native_release_step": 700,
        "native_force_physics_reward_unchanged": True,
        "source_material_prior": prior().tolist(),
        "independent_noise_sd_m": 0.002,
        "shared_bias_sd_m": 0.005,
        "noise_draws_per_world": 8192,
        "noise_seed": 260912,
        "minimum_whitened_stretching_secant_norm": 1.0,
        "reference_action_pairs": [list(p) for p in REFERENCE_PAIRS],
        "reference_and_duplicate_position_budget_m": POSITION_BUDGET_M,
        "reference_and_duplicate_reward_budget": REWARD_BUDGET,
        "entire_common_prefix_budget_m": 1e-6,
        "new_context_repeat_policy": "first_three_nominal_batches_all_actions",
        "nominal_source_value_uses_first_batch_not_best_or_average": True,
        "numeric_pair_margin": PAIR_MARGIN,
        "numeric_budget_is_population_bound": False,
        "minimum_adjusted_information_gain": 0.005,
        "minimum_adjusted_relative_excess_gain": 0.1,
        "minimum_adjusted_gain_over_map": 0.002,
        "minimum_adjusted_gain_over_ignored_bias": 0.0,
        "require_not_worse_than_original_best_fixed_within_pair_margin": True,
        "fallback_is_cached_artifact_not_bit_exact_native_replay": True,
        "earlier_failed_studies_reopened": False,
        "method_evaluation_authorized": False,
        "new_recordings": False,
        "gpu_work": False,
        "protected_data_read": False,
        "retry_authorized": False,
    }


def controls(source: np.ndarray) -> np.ndarray:
    if (
        source.shape != (8, 3, 6)
        or source.dtype != np.float64
        or not np.isfinite(source).all()
        or not np.array_equal(source[5, :2], source[6, :2])
    ):
        raise ValueError("exact shared-two-macro source controls required")
    value = np.repeat(source[5:6], 8, axis=0)
    value[1, 2, :3] = 0
    value[2, 2, :3] *= 0.5
    for slot, angle in ((3, -np.pi / 9), (4, np.pi / 9)):
        rotation = np.asarray(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        value[slot, 2, :2] = rotation @ source[5, 2, :2]
    value[5] = source[6]
    value[6, 2, 5] += 0.4
    if (
        np.max(np.linalg.norm(value[:, :, :3], axis=-1)) > 0.1 + 1e-12
        or np.max(np.abs(value[:, :, 3:])) > 1
        or not np.all(value[:, :2] == source[5, :2])
    ):
        raise ValueError("native limits or two-macro prefix changed")
    return value


def observations(prefix: dict[str, np.ndarray]) -> np.ndarray:
    if set(prefix) != {"rod_pos_m", "sphere_pos_m"}:
        raise ValueError("only registered prefix observations permitted")
    rod, sphere = prefix["rod_pos_m"], prefix["sphere_pos_m"]
    if (
        rod.shape != (500, 8, 12, 3)
        or sphere.shape != (500, 8, 3)
        or not np.isfinite(rod).all()
        or not np.isfinite(sphere).all()
    ):
        raise ValueError("complete finite 500-frame prefix required")
    selected = np.concatenate(
        [rod[list(FRAMES)][:, :, [3, 6, 8]], sphere[list(FRAMES), :, None]], axis=2
    )
    return selected.transpose(1, 0, 2, 3)


def native_checks(data, native, source, reference, reference_rewards, world):
    rows = split_batch(data, 8)
    if array_digest(data["controls"]) != array_digest(controls(source)):
        raise ValueError("frozen late-branch controls changed")
    metrics = [task_metrics(row) for row in rows]
    rewards = [m["native_reward"] for m in metrics]
    if rewards != native["native_cumulative_reward"]:
        raise ValueError("native reward arithmetic changed")
    realization = native["world_realization"]
    for field, key in (("bending", "bending_E"), ("stretching", "stretching_K")):
        if realization[field] != [[world[key]] * 8]:
            raise ValueError("native material realization changed")
    for field, xyz in (("sphere", [0.12, 0.06, 0.2]), ("cube", [0.12, 0.23, 0.22])):
        if not np.allclose(
            realization[f"{field}_initial_position_m"], [xyz] * 8, rtol=0, atol=1e-15
        ):
            raise ValueError("native source placement changed")
    common = max(
        float(np.max(np.abs(data[k][:500] - reference[k][:500, 5:6])))
        for k in POSITION_FIELDS
    )
    duplicate = max(
        float(np.max(np.abs(data[k][:, 0] - data[k][:, 7]))) for k in POSITION_FIELDS
    )
    reference_error = max(
        float(np.max(np.abs(data[k][:, a] - reference[k][:, b])))
        for k in POSITION_FIELDS
        for a, b in REFERENCE_PAIRS
    )
    reward_error = max(
        abs(rewards[a] - reference_rewards[b]) for a, b in REFERENCE_PAIRS
    )
    fixed = float(
        np.max(
            np.abs(
                data["rod_pos_m"][:, :, [0, 1, 10, 11]]
                - reference["rod_pos_m"][:1, :, [0, 1, 10, 11]]
            )
        )
    )
    checks = {
        "entire_common_prefix": common <= 1e-6,
        "fixed_endpoints": fixed <= 1e-9,
        "duplicate_positions": duplicate <= POSITION_BUDGET_M,
        "duplicate_rewards": abs(rewards[0] - rewards[7]) <= REWARD_BUDGET,
        "retained_reference_positions": reference_error <= POSITION_BUDGET_M,
        "retained_reference_rewards": reward_error <= REWARD_BUDGET,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "metrics": metrics,
        "prefix_error_m": common,
        "duplicate_error_m": duplicate,
        "reference_error_m": reference_error,
        "reference_reward_error": reward_error,
        "fixed_error_m": fixed,
    }


def repeat_checks(
    data: list[dict[str, np.ndarray]], rewards: np.ndarray
) -> dict[str, Any]:
    if len(data) != 3 or rewards.shape != (3, 8) or not np.isfinite(rewards).all():
        raise ValueError("all three nominal repetitions required")
    position = max(
        float(np.max(np.ptp(np.stack([row[name] for row in data]), axis=0)))
        for name in POSITION_FIELDS
    )
    reward = float(np.max(np.ptp(rewards, axis=0)))
    return {
        "maximum_coordinate_span_m": position,
        "maximum_same_action_reward_span": reward,
        "passed": position <= POSITION_BUDGET_M and reward <= REWARD_BUDGET,
        "population_bound_claimed": False,
    }


def information_value(prefix: np.ndarray, reward: np.ndarray, original: np.ndarray):
    if (
        prefix.shape != (9, 3, 4, 3)
        or reward.shape != (9, 7)
        or original.shape != (9, 7)
        or any(not np.isfinite(v).all() for v in (prefix, reward, original))
    ):
        raise ValueError("complete nine-world source bank required")
    p = prior()
    rng = np.random.default_rng(260912)
    noise = rng.normal(0, 0.005, (8192, 1, 3)) + rng.normal(0, 0.002, (8192, 12, 3))
    h = prefix.reshape(9, 12, 3) - prefix[4].reshape(1, 12, 3)
    covariance = 0.002**2 * np.eye(12) + 0.005**2 * np.ones((12, 12))
    chol = np.linalg.cholesky(covariance)
    versions = (
        (
            np.linalg.solve(chol, h).reshape(9, 36),
            np.linalg.solve(chol, noise).reshape(8192, 36),
        ),
        (h.reshape(9, 36) / 0.002, noise.reshape(8192, 36) / 0.002),
    )
    realized = np.zeros((8192, 3))
    per_world = np.zeros((9, 3))
    selection = np.zeros((3, 7))
    for world in range(9):
        for start in range(0, 8192, 256):
            decisions = []
            for white, error in versions:
                distance = np.sum(
                    (white[world] + error[start : start + 256, None] - white) ** 2,
                    axis=-1,
                )
                log_weight = np.log(p) - 0.5 * distance
                weight = np.exp(log_weight - log_weight.max(axis=1, keepdims=True))
                weight /= weight.sum(axis=1, keepdims=True)
                decisions.append(np.argmax(weight @ reward, axis=1))
                if len(decisions) == 1:
                    decisions.append(
                        np.argmax(reward[np.argmax(weight, axis=1)], axis=1)
                    )
            selected = np.stack(decisions, axis=1)
            values = reward[world, selected]
            realized[start : start + 256] += p[world] * values
            per_world[world] += values.sum(axis=0) / 8192
            for arm in range(3):
                selection[arm] += (
                    p[world] * np.bincount(selected[:, arm], minlength=7) / 8192
                )
    mean = realized.mean(axis=0)
    fixed = float(np.max(p @ reward))
    old_fixed = float(np.max(p @ original))
    adjusted = float(mean[0] - fixed - PAIR_MARGIN)
    map_gain = float(np.mean(realized[:, 0] - realized[:, 1]))
    bias_gain = float(np.mean(realized[:, 0] - realized[:, 2]))
    checks = {
        "adjusted_information_gain_at_least_0_005": adjusted >= 0.005,
        "adjusted_gain_at_least_10pct_fixed_excess": adjusted
        >= 0.1 * max(0.01, fixed - ZERO_REWARD),
        "adjusted_gain_over_map_at_least_0_002": map_gain - PAIR_MARGIN >= 0.002,
        "adjusted_gain_over_ignored_bias_nonnegative": bias_gain - PAIR_MARGIN >= 0,
        "not_below_original_best_fixed_within_margin": float(mean[0])
        >= old_fixed - PAIR_MARGIN,
    }
    names = ("bias_aware_posterior_mean", "bias_aware_map", "ignored_shared_bias")
    return {
        "arms": {
            name: {
                "expected_native_reward": float(mean[i]),
                "gain_over_best_fixed": float(mean[i] - fixed),
                "monte_carlo_standard_error": float(
                    realized[:, i].std(ddof=1) / np.sqrt(8192)
                ),
                "source_world_expected_rewards": per_world[:, i].tolist(),
                "action_probability": selection[i].tolist(),
            }
            for i, name in enumerate(names)
        },
        "best_fixed_action": int(np.argmax(p @ reward)),
        "best_fixed_reward": fixed,
        "original_best_fixed_reward": old_fixed,
        "perfect_information_reward": float(p @ reward.max(axis=1)),
        "oracle_actions": np.argmax(reward, axis=1).tolist(),
        "adjusted_information_gain": adjusted,
        "posterior_gain_over_map": map_gain,
        "posterior_gain_over_ignored_bias": bias_gain,
        "material_information": material_information(prefix),
        "checks": checks,
        "source_information_value_passed": all(checks.values()),
        "integration_only_not_independent_control_performance": True,
        "numerical_margin_is_not_statistical_bound": True,
    }
