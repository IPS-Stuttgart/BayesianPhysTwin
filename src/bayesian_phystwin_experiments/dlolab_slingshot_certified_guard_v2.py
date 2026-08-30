"""Prospective public-simulator replication of the Slingshot mean-regret guard."""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from bayesian_phystwin.guard_harm_risk import one_sided_binomial_upper_bound

from .dlolab_slingshot_active_bayes import continuous_worlds as active_v1_worlds
from .dlolab_slingshot_active_bayes_v2 import continuous_worlds as active_v2_worlds
from .dlolab_slingshot_belief import (
    BASELINE,
    BIAS_STD_M,
    NOISE_STD_M,
    ORDER,
    REWARD_MARGIN,
    particle_worlds,
    prior_weights,
    sample_worlds,
)
from .dlolab_slingshot_value import worlds as source_worlds

Array: TypeAlias = NDArray[Any]

WORLD_COUNT = 288
PREFIX_BATCH_COUNT = 36
SENSOR_DRAWS = 4096
WORLD_SEED = 261920
SENSOR_SEED = 261921
BOOTSTRAP_SEED = 261922
BOOTSTRAP_REPLICATES = 20000
ARM_NAMES = (
    "incumbent",
    "posterior_predictive_mean",
    "mean_regret_guard",
)
PARENT_ROOT = (
    "/home/florianpfaff/source-only/dlolab-slingshot-belief-control-source-v1-compact"
)
PARENT_LOCK_ID = "015e6d84aa68a2a4310552ef4880752b972890f02d3e09e333ff575c92b8df25"
PARENT_RESULT_ID = "9b8ff0817744392e0584c9b59936dd1b0e9331d3b0fa2d021f5a361947d32ee9"
PARENT_CALIBRATOR_ID = (
    "3d33e111a95f9e504c83c1938cba63a77009e900f3aaf4e137b54563f892d3eb"
)
PARENT_BANK_ID = "8ebf9c91322faf0658c84a2dcaa6895a98b1ff857e49e6714a2a2dad0c88d882"
PARENT_FILE_SHA256 = {
    "lock.json": "6dce35441588c2a5eff9c0ae08d85c8b41ff660403541dd489b8d9161bffcc8d",
    "result.json": "1df6afe4832a9c35bc65543255f5ce2c5830e6d58cfaa23d1140f8c867767e0b",
    "calibrator.json": "26a00b934dd91b9c121242858756b7a44fa58d61163db53a3ebdebf229de6725",
    "model-bank/arrays.npz": (
        "ef627e16490c0974d4c34fc82c16aae884fe6dd2a8dc0a80983e89b6d5e50832"
    ),
    "model-bank/seal.json": (
        "f4a9331d552fe8f9715d222327c3f5c41cd7fc81a006e0f9a2fc55dd2223a3ae"
    ),
}
MEAN_CALIBRATION_OFFSET = 0.7285524030751176
HARM_RISK_BUDGET = 0.05


def continuous_worlds() -> list[dict[str, Any]]:
    """Return the fixed fresh roster from the parent's continuous distribution."""
    rng = np.random.default_rng(WORLD_SEED)
    x = rng.uniform(-0.02, 0.02, WORLD_COUNT)
    bending = 1e5 * np.exp(rng.uniform(np.log(0.5), np.log(2.0), WORLD_COUNT))
    stretching = 8e5 * np.exp(rng.uniform(np.log(0.5), np.log(2.0), WORLD_COUNT))
    return [
        {
            "index": index,
            "x_offset_m": float(x[index]),
            "bending_E": float(bending[index]),
            "stretching_K": float(stretching[index]),
        }
        for index in range(WORLD_COUNT)
    ]


def _world_key(world: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(world["x_offset_m"]),
        float(world["bending_E"]),
        float(world["stretching_K"]),
    )


def _opened_world_keys() -> set[tuple[float, float, float]]:
    opened = [
        *source_worlds(),
        *particle_worlds(),
        *sample_worlds("calibration"),
        *sample_worlds("evaluation"),
        *active_v1_worlds(),
        *active_v2_worlds(),
    ]
    return {_world_key(world) for world in opened}


def validate_world(world: dict[str, Any]) -> None:
    if (
        set(world) != {"index", "x_offset_m", "bending_E", "stretching_K"}
        or type(world["index"]) is not int
        or world["index"] not in range(WORLD_COUNT)
        or world != continuous_worlds()[world["index"]]
        or any(
            type(world[name]) is not float or not np.isfinite(world[name])
            for name in ("x_offset_m", "bending_E", "stretching_K")
        )
    ):
        raise ValueError("registered fresh Slingshot world required")


def prefix_task(batch: int) -> dict[str, Any]:
    if type(batch) is not int or batch not in range(PREFIX_BATCH_COUNT):
        raise ValueError("registered Slingshot prefix batch required")
    indices = list(range(8 * batch, min(8 * batch + 8, WORLD_COUNT)))
    native_indices = indices + [indices[-1]] * (8 - len(indices))
    return {
        "kind": "prefix_only",
        "name": f"prefix-{batch:02d}",
        "batch": batch,
        "world_indices": indices,
        "native_world_indices": native_indices,
    }


def future_task(index: int) -> dict[str, Any]:
    if type(index) is not int or index not in range(WORLD_COUNT):
        raise ValueError("registered Slingshot future required")
    return {
        "kind": "all_action_future",
        "name": f"future-{index:03d}",
        "world_index": index,
    }


def protocol() -> dict[str, Any]:
    fresh = continuous_worlds()
    fresh_keys = {_world_key(world) for world in fresh}
    if len(fresh_keys) != WORLD_COUNT or fresh_keys & _opened_world_keys():
        raise ValueError("fresh Slingshot replication roster changed")
    return {
        "schema": "dlolab-slingshot-certified-guard-source-v2",
        "role": "prospective_public_simulator_cross_task_guard_replication",
        "parent_root": PARENT_ROOT,
        "parent_lock_id": PARENT_LOCK_ID,
        "parent_result_id": PARENT_RESULT_ID,
        "parent_source_gate_passed": False,
        "parent_gate_reclassified": False,
        "parent_mean_guard_used_for_candidate_selection": True,
        "parent_evaluation_is_development_evidence_only": True,
        "parent_calibrator_id": PARENT_CALIBRATOR_ID,
        "parent_model_bank_id": PARENT_BANK_ID,
        "parent_file_sha256": PARENT_FILE_SHA256,
        "method": "frozen_split_conformal_mean_regret_guard_with_exact_fallback",
        "mean_calibration_offset": MEAN_CALIBRATION_OFFSET,
        "fallback_action": BASELINE,
        "action_order_for_regret": list(ORDER),
        "worlds": fresh,
        "world_seed": WORLD_SEED,
        "world_count": WORLD_COUNT,
        "world_distribution": {
            "x_offset_m": "uniform[-0.02,0.02]",
            "bending_E": "log_uniform[50000,200000]",
            "stretching_K": "log_uniform[400000,1600000]",
        },
        "worlds_disjoint_from_all_registered_slingshot_source_and_development": True,
        "sensor_draws_per_world": SENSOR_DRAWS,
        "sensor_seed": SENSOR_SEED,
        "shared_xyz_bias_std_m": BIAS_STD_M,
        "independent_noise_std_m": NOISE_STD_M,
        "observation_frames": [139, 219, 299],
        "observation_rod_nodes": [3, 6, 8],
        "observation_sphere_center": True,
        "observation_units": "world_frame_metres",
        "statistical_unit": "fresh_continuous_world_after_averaging_sensor_draws",
        "arms": list(ARM_NAMES),
        "primary_arm": "mean_regret_guard",
        "harm_event": "world_mean_reward_below_incumbent_by_more_than_0_002",
        "harm_risk_interval": "one_sided_exact_clopper_pearson",
        "harm_risk_budget": HARM_RISK_BUDGET,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "pre_future_gate": {
            "all_36_prefix_batches_native_qualified": True,
            "guard_nonfallback_sensor_decisions_at_least": (
                WORLD_COUNT * SENSOR_DRAWS // 100
            ),
            "guard_updates_at_least_worlds": 32,
            "guard_differs_from_posterior_mean_at_least": (
                WORLD_COUNT * SENSOR_DRAWS // 100
            ),
            "distinct_guard_actions_at_least": 2,
        },
        "source_gate": {
            "all_288_worlds_and_native_qa": True,
            "distinct_oracle_actions_at_least": 2,
            "guard_gain_over_incumbent_at_least": 0.001,
            "positive_paired_ci95_vs_incumbent": True,
            "guard_one_sided_95pct_harm_risk_upper_at_most": HARM_RISK_BUDGET,
            "posterior_mean_harmed_worlds_at_least": 10,
            "guard_reduces_harmed_worlds_by_at_least": 5,
            "guard_reduces_mean_downside_vs_posterior_by_fraction": 0.75,
            "guard_retains_at_least_fraction_of_posterior_gain": 0.10,
            "oracle_headroom_fraction_at_least": 0.05,
        },
        "stage_order": [
            "all_prefix_only_batches",
            "all_noisy_observations_and_decisions",
            "decision_barrier_and_pre_future_gate",
            "all_action_futures",
            "score",
        ],
        "future_before_decision_barrier": False,
        "retry_authorized": False,
        "replacement_authorized": False,
        "fresh_successor_automatically_authorized": False,
        "official_benchmark_or_sota_claim": False,
        "real_robot_or_physical_safety_claim": False,
        "protected_data_read": False,
        "held_v8_read": False,
        "dlo4_dlo5_read": False,
        "official_dlo3_evaluation": False,
        "new_recordings": False,
        "gpu_work": False,
        "push_or_merge": False,
    }


def _posterior_weights(observations: Array, bank_prefix: Array) -> Array:
    observed = np.asarray(observations, dtype=np.float64)
    predicted = np.asarray(bank_prefix, dtype=np.float64)
    if (
        observed.ndim != 4
        or observed.shape[1:] != (3, 4, 3)
        or predicted.shape != (27, 3, 4, 3)
        or not np.isfinite(observed).all()
        or not np.isfinite(predicted).all()
    ):
        raise ValueError("complete finite Slingshot observations and bank required")
    residual = (observed[:, None] - predicted[None]).reshape(len(observed), 27, 12, 3)
    mean = residual.mean(axis=2)
    centered = residual - mean[:, :, None]
    noise_var = NOISE_STD_M**2
    shared_var = BIAS_STD_M**2
    distance = np.sum(centered**2, axis=(2, 3)) / noise_var
    distance += 12 * np.sum(mean**2, axis=2) / (noise_var + 12 * shared_var)
    with np.errstate(divide="ignore"):
        log_weight = np.log(prior_weights())[None] - 0.5 * distance
    log_weight -= np.max(log_weight, axis=1, keepdims=True)
    weight = np.exp(log_weight)
    weight /= weight.sum(axis=1, keepdims=True)
    if not np.isfinite(weight).all():
        raise ValueError("invalid posterior weights")
    return weight


def _decisions_for_observations(
    observations: Array,
    bank_prefix: Array,
    bank_reward: Array,
) -> Array:
    reward = np.asarray(bank_reward, dtype=np.float64)
    if reward.shape != (27, 7) or not np.isfinite(reward).all():
        raise ValueError("complete finite Slingshot reward bank required")
    expected = _posterior_weights(observations, bank_prefix) @ (-reward[:, ORDER])
    posterior = np.argmin(expected, axis=1)
    raw_upper = expected - expected[:, :1]
    allowed = raw_upper + MEAN_CALIBRATION_OFFSET < 0
    allowed[:, 0] = True
    guarded = np.argmin(np.where(allowed, expected, np.inf), axis=1)
    order = np.asarray(ORDER, dtype=np.int64)
    result: Array = np.empty((len(expected), len(ARM_NAMES)), dtype=np.int64)
    result[:, 0] = BASELINE
    result[:, 1] = order[posterior]
    result[:, 2] = order[guarded]
    return result


def infer_decisions(
    truth_prefix: Array,
    bank_prefix: Array,
    bank_reward: Array,
) -> dict[str, Array]:
    """Apply the frozen guard to registered sensor draws before future simulation."""
    truth = np.asarray(truth_prefix, dtype=np.float64)
    model_prefix = np.asarray(bank_prefix, dtype=np.float64)
    reward = np.asarray(bank_reward, dtype=np.float64)
    if (
        truth.shape != (WORLD_COUNT, 3, 4, 3)
        or model_prefix.shape != (27, 3, 4, 3)
        or reward.shape != (27, 7)
        or any(not np.isfinite(value).all() for value in (truth, model_prefix, reward))
    ):
        raise ValueError("complete finite Slingshot decision inputs required")
    decisions: Array = np.empty(
        (WORLD_COUNT, SENSOR_DRAWS, len(ARM_NAMES)), dtype=np.int64
    )
    decisions[:, :, 0] = BASELINE
    rng = np.random.default_rng(SENSOR_SEED)
    for world in range(WORLD_COUNT):
        bias = rng.normal(0, BIAS_STD_M, (SENSOR_DRAWS, 1, 1, 3))
        noise = rng.normal(0, NOISE_STD_M, (SENSOR_DRAWS, 3, 4, 3))
        observation = truth[world, None] + bias + noise
        decisions[world] = _decisions_for_observations(
            observation, model_prefix, reward
        )
    return {"decision": decisions}


def pre_future_checks(decision: Array, *, all_prefix_qa: bool) -> dict[str, Any]:
    values = np.asarray(decision)
    if (
        values.shape != (WORLD_COUNT, SENSOR_DRAWS, len(ARM_NAMES))
        or values.dtype.kind not in "iu"
        or np.any((values < 0) | (values > 6))
        or np.any(values[:, :, 0] != BASELINE)
    ):
        raise ValueError("complete valid pre-future decisions required")
    guard = values[:, :, 2]
    posterior = values[:, :, 1]
    nonfallback = int(np.count_nonzero(guard != BASELINE))
    updated_worlds = int(np.count_nonzero(np.any(guard != BASELINE, axis=1)))
    difference = int(np.count_nonzero(guard != posterior))
    distinct = int(len(np.unique(guard)))
    threshold = WORLD_COUNT * SENSOR_DRAWS // 100
    checks = {
        "all_36_prefix_batches_native_qualified": bool(all_prefix_qa),
        "guard_nonfallback_sensor_decisions_at_least_registered": (
            nonfallback >= threshold
        ),
        "guard_updates_at_least_32_worlds": updated_worlds >= 32,
        "guard_differs_from_posterior_mean_at_least_registered": (
            difference >= threshold
        ),
        "distinct_guard_actions_at_least_two": distinct >= 2,
    }
    return {
        "guard_nonfallback_sensor_decisions": nonfallback,
        "guard_updated_worlds": updated_worlds,
        "guard_posterior_decision_differences": difference,
        "distinct_guard_actions": distinct,
        "checks": checks,
        "pre_future_gate_passed": bool(all(checks.values())),
    }


def score(
    decision: Array,
    rewards: Array,
    *,
    all_native_qa: bool,
    pre_future_gate_passed: bool,
) -> dict[str, Any]:
    values = np.asarray(decision)
    reward = np.asarray(rewards, dtype=np.float64)
    if (
        values.shape != (WORLD_COUNT, SENSOR_DRAWS, len(ARM_NAMES))
        or reward.shape != (WORLD_COUNT, 7)
        or values.dtype.kind not in "iu"
        or np.any((values < 0) | (values > 6))
        or not np.isfinite(reward).all()
    ):
        raise ValueError("complete Slingshot result denominator required")
    selected = np.take_along_axis(reward[:, None, :], values, axis=2)
    world_reward = selected.mean(axis=1)
    incumbent = reward[:, BASELINE]
    all_fallback = np.all(values == BASELINE, axis=1)
    world_reward = np.where(all_fallback, incumbent[:, None], world_reward)
    gain = world_reward - incumbent[:, None]
    bootstrap = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, WORLD_COUNT, (BOOTSTRAP_REPLICATES, WORLD_COUNT)
    )
    arms: dict[str, Any] = {}
    for index, name in enumerate(ARM_NAMES):
        harms = int(np.count_nonzero(gain[:, index] < -REWARD_MARGIN))
        arms[name] = {
            "mean_native_reward": float(world_reward[:, index].mean()),
            "mean_gain_over_incumbent": float(gain[:, index].mean()),
            "mean_gain_ci95": np.quantile(
                gain[bootstrap, index].mean(axis=1), [0.025, 0.975]
            ).tolist(),
            "nonfallback_sensor_decisions": int(
                np.count_nonzero(values[:, :, index] != BASELINE)
            ),
            "updated_worlds": int(
                np.count_nonzero(np.any(values[:, :, index] != BASELINE, axis=1))
            ),
            "harmful_worlds_beyond_numeric_margin": harms,
            "harm_probability_upper95": one_sided_binomial_upper_bound(
                harms, WORLD_COUNT, 0.95
            ),
            "mean_downside_below_incumbent": float(
                np.maximum(-gain[:, index], 0).mean()
            ),
        }
    guard = arms["mean_regret_guard"]
    posterior = arms["posterior_predictive_mean"]
    oracle_gain = float(reward.max(axis=1).mean() - incumbent.mean())
    posterior_gain = float(posterior["mean_gain_over_incumbent"])
    guard_gain = float(guard["mean_gain_over_incumbent"])
    posterior_downside = float(posterior["mean_downside_below_incumbent"])
    guard_downside = float(guard["mean_downside_below_incumbent"])
    retained = guard_gain / posterior_gain if posterior_gain > 0 else 0.0
    headroom = guard_gain / oracle_gain if oracle_gain > 0 else 0.0
    downside_reduction = (
        1.0 - guard_downside / posterior_downside if posterior_downside > 0 else 0.0
    )
    harm_reduction = int(
        posterior["harmful_worlds_beyond_numeric_margin"]
        - guard["harmful_worlds_beyond_numeric_margin"]
    )
    checks = {
        "complete_denominator": True,
        "pre_future_gate_passed": bool(pre_future_gate_passed),
        "all_native_qa": bool(all_native_qa),
        "distinct_oracle_actions_at_least_two": (
            len(np.unique(np.argmax(reward, axis=1))) >= 2
        ),
        "guard_gain_at_least_0_001": guard_gain >= 0.001,
        "positive_paired_ci95_vs_incumbent": guard["mean_gain_ci95"][0] > 0,
        "guard_harm_upper_at_most_0_05": (
            guard["harm_probability_upper95"] <= HARM_RISK_BUDGET
        ),
        "posterior_mean_harmed_worlds_at_least_10": (
            posterior["harmful_worlds_beyond_numeric_margin"] >= 10
        ),
        "guard_reduces_harmed_worlds_by_at_least_5": harm_reduction >= 5,
        "guard_reduces_downside_by_at_least_75pct": downside_reduction >= 0.75,
        "guard_retains_at_least_10pct_posterior_gain": retained >= 0.10,
        "guard_captures_at_least_5pct_oracle_headroom": headroom >= 0.05,
    }
    return {
        "schema": "dlolab-slingshot-certified-guard-result-v2",
        "arms": arms,
        "mean_guard_harm_reduction_vs_posterior": harm_reduction,
        "mean_guard_downside_reduction_fraction": downside_reduction,
        "mean_guard_fraction_of_posterior_gain": retained,
        "mean_guard_fraction_of_oracle_headroom": headroom,
        "oracle_mean_native_reward": float(reward.max(axis=1).mean()),
        "oracle_gain_over_incumbent": oracle_gain,
        "distinct_oracle_actions": int(len(np.unique(np.argmax(reward, axis=1)))),
        "checks": checks,
        "source_gate_passed": bool(all(checks.values())),
        "ordinary_evaluations": WORLD_COUNT,
        "technical_failures": 0,
        "replacements": 0,
        "official_benchmark_or_sota_claim": False,
        "real_robot_or_physical_safety_claim": False,
        "protected_data_read": False,
        "held_v8_read": False,
        "dlo4_dlo5_read": False,
        "new_recordings": False,
    }
