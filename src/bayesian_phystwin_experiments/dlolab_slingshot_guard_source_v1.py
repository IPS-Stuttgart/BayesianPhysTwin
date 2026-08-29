"""Cross-fitted exact-fallback guard for Slingshot active Bayes control."""

from __future__ import annotations

import math
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .dlolab_slingshot_belief import BIAS_STD_M, NOISE_STD_M, prior_weights
from .dlolab_slingshot_task_probe_dev import _posterior_weights

Array: TypeAlias = NDArray[Any]

WORLD_COUNT = 19
SENSOR_DRAWS = 8
SENSOR_SEED = 261411
BOOTSTRAP_SEED = 261412
BOOTSTRAP_REPLICATES = 20000
ACTIVE_FRACTION = 0.70
REWARD_MARGIN = 0.002
MIN_DELETE_ONE_GAIN = 0.00025
SPREAD_PENALTIES = (0.0, 0.5, 1.0, 2.0)
GAIN_MARGINS = (0.0, 0.001, 0.002)


def candidates() -> list[dict[str, Any]]:
    """Return the frozen guard bank, including the exact fallback."""

    result: list[dict[str, Any]] = [
        {
            "index": 0,
            "name": "exact_blind_fallback",
            "spread_penalty": None,
            "minimum_gain": None,
        }
    ]
    for penalty in SPREAD_PENALTIES:
        for margin in GAIN_MARGINS:
            result.append(
                {
                    "index": len(result),
                    "name": f"mean_minus_{penalty:g}spread_margin_{margin:g}",
                    "spread_penalty": penalty,
                    "minimum_gain": margin,
                }
            )
    return result


def protocol(worlds: list[dict[str, Any]]) -> dict[str, Any]:
    if len(worlds) != WORLD_COUNT:
        raise ValueError("complete calibration-world roster required")
    return {
        "schema": "dlolab-slingshot-guard-source-v1",
        "role": "already_open_source_development_only",
        "worlds": worlds,
        "world_count": WORLD_COUNT,
        "sensor_draws_per_world": SENSOR_DRAWS,
        "sensor_seed": SENSOR_SEED,
        "shared_bias_std_m": BIAS_STD_M,
        "independent_noise_std_m": NOISE_STD_M,
        "active_frontload_fraction": ACTIVE_FRACTION,
        "probe_role": "separate_matched_reset_identification_episode",
        "probe_cost_in_task_reward": False,
        "candidate_family": candidates(),
        "guard_score": "posterior_mean_gain_minus_lambda_times_posterior_std",
        "exact_fallback": "blind_prior_action",
        "selection": {
            "outer_evaluation": "leave_one_world_out",
            "fit_worlds_per_fold": 18,
            "minimum_updated_draw_fraction": 0.10,
            "minimum_updated_world_fraction": 0.20,
            "minimum_delete_one_mean_gain": MIN_DELETE_ONE_GAIN,
            "maximum_harmed_world_fraction": 0.20,
            "fallback_when_no_candidate_admissible": True,
            "selection_order": [
                "maximum_delete_one_mean_gain",
                "maximum_mean_gain",
                "fewest_harmed_worlds",
                "larger_spread_penalty",
                "larger_gain_margin",
            ],
        },
        "pre_outcome_gate": {
            "all_three_active_prefix_batches_native_qualified": True,
            "unguarded_active_bayes_nonblind_draws_at_least": 16,
            "at_least_three_distinct_candidate_update_counts": True,
        },
        "source_gate": {
            "complete_19_world_cross_fit": True,
            "cross_fitted_nonblind_draws_at_least": 16,
            "cross_fitted_updated_worlds_at_least": 4,
            "full_fit_candidate_is_not_fallback": True,
            "cross_fitted_gain_over_blind_at_least": 0.001,
            "paired_world_bootstrap_ci95_lower_vs_blind_above": 0.0,
            "gain_over_unguarded_active_bayes_at_least": 0.001,
            "harmed_worlds_fewer_than_unguarded_active_bayes": True,
            "harmed_worlds_no_more_than": 3,
        },
        "stage_order": [
            "three_active_prefix_batches",
            "all_observations_posteriors_and_candidate_decisions",
            "decision_barrier_and_pre_outcome_gate",
            "read_already_open_parent_calibration_rewards",
            "leave_one_world_out_selection_and_score",
        ],
        "parent_calibration_reward_before_decision_barrier": False,
        "fresh_world_automatically_authorized": False,
        "retry_authorized": False,
        "replacement_authorized": False,
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


def sensor_errors() -> tuple[Array, Array]:
    rng = np.random.default_rng(SENSOR_SEED)
    bias = rng.normal(0, BIAS_STD_M, (WORLD_COUNT, SENSOR_DRAWS, 1, 1, 3))
    noise = rng.normal(0, NOISE_STD_M, (WORLD_COUNT, SENSOR_DRAWS, 3, 4, 3))
    return np.asarray(bias, dtype=np.float64), np.asarray(noise, dtype=np.float64)


def infer_candidates(history: Array, rewards: Array, truth_prefixes: Array) -> dict[str, Array]:
    """Infer posterior actions and every frozen guard decision before outcomes."""

    model_history = np.asarray(history, dtype=np.float64)
    model_reward = np.asarray(rewards, dtype=np.float64)
    truth = np.asarray(truth_prefixes, dtype=np.float64)
    prior = np.asarray(prior_weights(), dtype=np.float64)
    if (
        model_history.shape != (27, 3, 4, 3)
        or model_reward.shape != (27, 7)
        or truth.shape != (WORLD_COUNT, 3, 4, 3)
        or any(
            not value.all()
            for value in (
                np.isfinite(model_history),
                np.isfinite(model_reward),
                np.isfinite(truth),
            )
        )
    ):
        raise ValueError("complete finite active histories, rewards, and prefixes required")
    bias, noise = sensor_errors()
    observations = truth[:, None] + bias + noise
    posterior: Array = np.empty(
        (WORLD_COUNT, SENSOR_DRAWS, 27), dtype=np.float64
    )
    for world in range(WORLD_COUNT):
        posterior[world] = _posterior_weights(
            observations[world], model_history, prior
        )
    expected_reward = posterior @ model_reward
    bayes_action = np.argmax(expected_reward, axis=-1).astype(np.int64)
    map_world = np.argmax(posterior, axis=-1)
    map_action = np.argmax(model_reward[map_world], axis=-1).astype(np.int64)
    blind_action = int(np.argmax(prior @ model_reward))

    particle_gain: Array = np.empty(
        (WORLD_COUNT, SENSOR_DRAWS, 27), dtype=np.float64
    )
    for world in range(WORLD_COUNT):
        for draw in range(SENSOR_DRAWS):
            particle_gain[world, draw] = (
                model_reward[:, bayes_action[world, draw]]
                - model_reward[:, blind_action]
            )
    mean_gain = np.sum(posterior * particle_gain, axis=-1)
    centered = particle_gain - mean_gain[..., None]
    spread = np.sqrt(np.maximum(0.0, np.sum(posterior * centered**2, axis=-1)))
    positive_probability = np.sum(posterior * (particle_gain > 0), axis=-1)

    candidate_decisions: Array = np.full(
        (WORLD_COUNT, SENSOR_DRAWS, len(candidates())),
        blind_action,
        dtype=np.int64,
    )
    for spec in candidates()[1:]:
        penalty = float(spec["spread_penalty"])
        margin = float(spec["minimum_gain"])
        admitted = (mean_gain - penalty * spread >= margin) & (
            bayes_action != blind_action
        )
        candidate_decisions[:, :, spec["index"]] = np.where(
            admitted, bayes_action, blind_action
        )
    return {
        "truth_prefix_m": truth,
        "shared_bias_m": bias,
        "independent_noise_m": noise,
        "observation_m": observations,
        "posterior_weights": posterior,
        "posterior_expected_reward": expected_reward,
        "posterior_expected_gain": mean_gain,
        "posterior_gain_std": spread,
        "posterior_positive_gain_probability": positive_probability,
        "blind_action": np.asarray(blind_action, dtype=np.int64),
        "active_map_action": map_action,
        "active_bayes_action": bayes_action,
        "candidate_decisions": candidate_decisions,
    }


def pre_outcome_checks(candidate_data: dict[str, Array], *, all_prefix_qa: bool) -> dict[str, Any]:
    decision = np.asarray(candidate_data["candidate_decisions"])
    bayes = np.asarray(candidate_data["active_bayes_action"])
    blind = int(np.asarray(candidate_data["blind_action"]))
    if (
        decision.shape != (WORLD_COUNT, SENSOR_DRAWS, len(candidates()))
        or bayes.shape != (WORLD_COUNT, SENSOR_DRAWS)
        or decision.dtype.kind not in "iu"
        or bayes.dtype.kind not in "iu"
        or np.any((decision < 0) | (decision > 6))
        or np.any((bayes < 0) | (bayes > 6))
        or np.any(decision[:, :, 0] != blind)
    ):
        raise ValueError("complete valid pre-outcome guard decisions required")
    counts = [
        int(np.count_nonzero(decision[:, :, index] != blind))
        for index in range(len(candidates()))
    ]
    nonblind = int(np.count_nonzero(bayes != blind))
    checks = {
        "all_three_active_prefix_batches_native_qualified": bool(all_prefix_qa),
        "unguarded_active_bayes_nonblind_draws_at_least_16": nonblind >= 16,
        "at_least_three_distinct_candidate_update_counts": len(set(counts)) >= 3,
    }
    return {
        "unguarded_active_bayes_nonblind_draws": nonblind,
        "candidate_nonblind_draws": counts,
        "checks": checks,
        "pre_outcome_gate_passed": bool(all(checks.values())),
    }


def _candidate_world_gains(decisions: Array, rewards: Array) -> tuple[Array, Array]:
    decision = np.asarray(decisions)
    reward = np.asarray(rewards, dtype=np.float64)
    if (
        decision.shape != (WORLD_COUNT, SENSOR_DRAWS, len(candidates()))
        or reward.shape != (WORLD_COUNT, 7)
        or decision.dtype.kind not in "iu"
        or not np.isfinite(reward).all()
        or np.any((decision < 0) | (decision > 6))
    ):
        raise ValueError("complete candidate decisions and source rewards required")
    selected = np.take_along_axis(reward[:, None, :], decision, axis=2)
    world_reward = selected.mean(axis=1)
    return world_reward, world_reward - world_reward[:, :1]


def _fit_stats(
    candidate_index: int,
    fit_indices: Array,
    decisions: Array,
    world_gain: Array,
) -> dict[str, Any]:
    indices = np.asarray(fit_indices, dtype=np.int64)
    if (
        indices.ndim != 1
        or len(indices) < 2
        or len(np.unique(indices)) != len(indices)
        or np.any((indices < 0) | (indices >= WORLD_COUNT))
        or candidate_index not in range(len(candidates()))
    ):
        raise ValueError("valid source-fit indices and candidate required")
    blind = decisions[0, 0, 0]
    update = decisions[indices, :, candidate_index] != blind
    gains = world_gain[indices, candidate_index]
    delete_one = [float(np.delete(gains, row).mean()) for row in range(len(gains))]
    updated_draws = int(np.count_nonzero(update))
    updated_worlds = int(np.count_nonzero(np.any(update, axis=1)))
    harmed = int(np.count_nonzero(gains < -REWARD_MARGIN))
    minimum_draws = math.ceil(0.10 * len(indices) * SENSOR_DRAWS)
    minimum_worlds = max(3, math.ceil(0.20 * len(indices)))
    maximum_harms = math.floor(0.20 * len(indices))
    admissible = bool(
        candidate_index > 0
        and updated_draws >= minimum_draws
        and updated_worlds >= minimum_worlds
        and min(delete_one) >= MIN_DELETE_ONE_GAIN
        and harmed <= maximum_harms
    )
    return {
        "candidate_index": candidate_index,
        "fit_worlds": int(len(indices)),
        "mean_gain": float(gains.mean()),
        "minimum_delete_one_mean_gain": min(delete_one),
        "updated_draws": updated_draws,
        "updated_worlds": updated_worlds,
        "harmed_worlds": harmed,
        "minimum_updated_draws": minimum_draws,
        "minimum_updated_worlds": minimum_worlds,
        "maximum_harmed_worlds": maximum_harms,
        "admissible": admissible,
    }


def select_candidate(fit_indices: Array, decisions: Array, rewards: Array) -> dict[str, Any]:
    """Select a guard on fit worlds, falling back exactly when none is stable."""

    _, world_gain = _candidate_world_gains(decisions, rewards)
    rows = [
        _fit_stats(index, fit_indices, decisions, world_gain)
        for index in range(len(candidates()))
    ]
    eligible = [row for row in rows if row["admissible"]]
    if eligible:
        selected = max(
            eligible,
            key=lambda row: (
                row["minimum_delete_one_mean_gain"],
                row["mean_gain"],
                -row["harmed_worlds"],
                float(candidates()[row["candidate_index"]]["spread_penalty"]),
                float(candidates()[row["candidate_index"]]["minimum_gain"]),
            ),
        )
    else:
        selected = rows[0]
    return {
        "selected_candidate_index": selected["candidate_index"],
        "selected_candidate": candidates()[selected["candidate_index"]],
        "selected_stats": selected,
        "candidate_stats": rows,
        "exact_fallback_selected": selected["candidate_index"] == 0,
    }


def cross_fitted_decisions(decisions: Array, rewards: Array) -> tuple[Array, list[dict[str, Any]], dict[str, Any]]:
    value = np.asarray(decisions)
    reward = np.asarray(rewards, dtype=np.float64)
    _candidate_world_gains(value, reward)
    selected: Array = np.empty((WORLD_COUNT, SENSOR_DRAWS), dtype=np.int64)
    folds: list[dict[str, Any]] = []
    all_indices: Array = np.arange(WORLD_COUNT, dtype=np.int64)
    for held_out in range(WORLD_COUNT):
        fit = all_indices[all_indices != held_out]
        choice = select_candidate(fit, value, reward)
        index = int(choice["selected_candidate_index"])
        selected[held_out] = value[held_out, :, index]
        folds.append({"held_out_world": held_out, **choice})
    full = select_candidate(all_indices, value, reward)
    return selected, folds, full


def _bootstrap_ci(values: Array) -> list[float]:
    difference = np.asarray(values, dtype=np.float64)
    if difference.shape != (WORLD_COUNT,) or not np.isfinite(difference).all():
        raise ValueError("one finite value per source world required")
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, WORLD_COUNT, size=(BOOTSTRAP_REPLICATES, WORLD_COUNT)
    )
    quantiles: Array = np.asarray(
        np.quantile(difference[indices].mean(axis=1), [0.025, 0.975])
    )
    return [float(quantiles[0]), float(quantiles[1])]


def score(candidate_data: dict[str, Array], rewards: Array, *, all_native_qa: bool) -> dict[str, Any]:
    decisions = np.asarray(candidate_data["candidate_decisions"])
    reward = np.asarray(rewards, dtype=np.float64)
    crossfit, folds, full = cross_fitted_decisions(decisions, reward)
    blind_action = int(np.asarray(candidate_data["blind_action"]))
    active_bayes = np.asarray(candidate_data["active_bayes_action"])
    active_map = np.asarray(candidate_data["active_map_action"])
    blind_reward = reward[:, blind_action]

    def arm(action: Array) -> tuple[Array, dict[str, Any]]:
        selected = np.take_along_axis(reward[:, None, :], action[..., None], axis=2)[
            :, :, 0
        ]
        world = selected.mean(axis=1)
        gain = world - blind_reward
        return world, {
            "mean_native_reward": float(world.mean()),
            "mean_gain_over_blind": float(gain.mean()),
            "gain_ci95": _bootstrap_ci(gain),
            "nonblind_sensor_decisions": int(np.count_nonzero(action != blind_action)),
            "updated_worlds": int(np.count_nonzero(np.any(action != blind_action, axis=1))),
            "worlds_harmed_beyond_numeric_margin": int(
                np.count_nonzero(gain < -REWARD_MARGIN)
            ),
        }

    crossfit_world, crossfit_stats = arm(crossfit)
    active_world, active_stats = arm(active_bayes)
    map_world, map_stats = arm(active_map)
    gain_vs_active = crossfit_world - active_world
    full_index = int(full["selected_candidate_index"])
    full_world, full_stats = arm(decisions[:, :, full_index])
    checks = {
        "complete_19_world_cross_fit": len(folds) == WORLD_COUNT,
        "all_native_qa": bool(all_native_qa),
        "cross_fitted_nonblind_draws_at_least_16": crossfit_stats[
            "nonblind_sensor_decisions"
        ]
        >= 16,
        "cross_fitted_updated_worlds_at_least_4": crossfit_stats["updated_worlds"]
        >= 4,
        "full_fit_candidate_is_not_fallback": full_index != 0,
        "cross_fitted_gain_over_blind_at_least_0_001": crossfit_stats[
            "mean_gain_over_blind"
        ]
        >= 0.001,
        "positive_paired_ci95_vs_blind": crossfit_stats["gain_ci95"][0] > 0,
        "gain_over_unguarded_active_bayes_at_least_0_001": float(
            gain_vs_active.mean()
        )
        >= 0.001,
        "harmed_worlds_fewer_than_unguarded_active_bayes": crossfit_stats[
            "worlds_harmed_beyond_numeric_margin"
        ]
        < active_stats["worlds_harmed_beyond_numeric_margin"],
        "harmed_worlds_no_more_than_3": crossfit_stats[
            "worlds_harmed_beyond_numeric_margin"
        ]
        <= 3,
    }
    return {
        "schema": "dlolab-slingshot-guard-source-score-v1",
        "arms": {
            "blind_prior": {
                "mean_native_reward": float(blind_reward.mean()),
                "mean_gain_over_blind": 0.0,
                "gain_ci95": [0.0, 0.0],
                "nonblind_sensor_decisions": 0,
                "updated_worlds": 0,
                "worlds_harmed_beyond_numeric_margin": 0,
            },
            "active_map": map_stats,
            "active_bayes": active_stats,
            "cross_fitted_guard": crossfit_stats,
            "full_fit_guard_in_sample_diagnostic": full_stats,
        },
        "cross_fitted_gain_over_active_bayes": {
            "mean_gain": float(gain_vs_active.mean()),
            "ci95": _bootstrap_ci(gain_vs_active),
        },
        "folds": folds,
        "full_fit_selection": full,
        "checks": checks,
        "source_gate_passed": bool(all(checks.values())),
        "ordinary_worlds": WORLD_COUNT,
        "sensor_draws_per_world": SENSOR_DRAWS,
        "technical_failures": 0,
        "replacements": 0,
        "fresh_world_automatically_authorized": False,
        "official_benchmark_or_sota_claim": False,
        "real_robot_or_physical_safety_claim": False,
        "protected_data_read": False,
        "new_recordings": False,
    }
