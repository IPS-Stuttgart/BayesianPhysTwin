"""Frozen matched belief/control comparison on fresh public-simulator worlds.

The posterior, weighted quantiles, and split-conformal calibrator are standard
methods. The experiment tests their baseline-relative decision value; it does
not introduce a coverage theorem or identify real material parameters.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import Any

import numpy as np

from bayesian_phystwin.guard_harm_risk import one_sided_binomial_upper_bound

from .coupled_action_regret import (
    RegretCalibration,
    action_regret_upper,
    bias_marginalized_weights,
    calibrate_simultaneous_regret,
    guarded_action,
    selected_commands,
)
from .dlolab_native import array_digest
from .dlolab_slingshot_batch import split_batch
from .dlolab_slingshot_cmaes import task_metrics
from .dlolab_slingshot_value import action_bank

OBSERVATION_FRAMES = (139, 219, 299)
OBSERVATION_NODES = (3, 6, 8)
NOISE_STD_M = 0.002
BIAS_STD_M = 0.005
COUNTS = {"calibration": 19, "evaluation": 32}
SEEDS = {"calibration": 260901, "evaluation": 260902}
SENSOR_SEEDS = {"calibration": 260903, "evaluation": 260904}
BASELINE = 5
ORDER = (5, 0, 1, 2, 3, 4, 6)
MODES = ("mean", "independent", "joint")
ARMS = (
    "incumbent",
    "nominal_point",
    "prior_predictive_mean",
    "map_point",
    "posterior_predictive_mean",
    "posterior_iid_bias",
    "mean_regret_guard",
    "independent_regret_guard",
    "joint_regret_guard",
)
POSITION_ENVELOPE_M = 0.0005
REWARD_MARGIN = 0.002
ZERO_REWARD = 6.900000095367432


def particle_worlds() -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "x_offset_m": float(x),
            "bending_E": float(1e5 * e),
            "stretching_K": float(8e5 * k),
        }
        for index, (x, e, k) in enumerate(
            itertools.product((-0.02, 0.0, 0.02), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0))
        )
    ]


def prior_weights() -> np.ndarray:
    # Tensor trapezoidal quadrature for uniform x and log-uniform E/K.
    return np.asarray(
        [a * b * c for a, b, c in itertools.product((0.25, 0.5, 0.25), repeat=3)]
    )


def sample_worlds(role: str) -> list[dict[str, Any]]:
    if role not in COUNTS:
        raise ValueError("unknown simulated partition")
    rng = np.random.default_rng(SEEDS[role])
    count = COUNTS[role]
    x = rng.uniform(-0.02, 0.02, count)
    e = 1e5 * np.exp(rng.uniform(np.log(0.5), np.log(2.0), count))
    k = 8e5 * np.exp(rng.uniform(np.log(0.5), np.log(2.0), count))
    return [
        {
            "index": i,
            "x_offset_m": float(x[i]),
            "bending_E": float(e[i]),
            "stretching_K": float(k[i]),
        }
        for i in range(count)
    ]


def sensor_errors(role: str) -> np.ndarray:
    if role not in COUNTS:
        raise ValueError("unknown simulated sensor partition")
    rng = np.random.default_rng(SENSOR_SEEDS[role])
    bias = rng.normal(0, BIAS_STD_M, (COUNTS[role], 1, 1, 3))
    noise = rng.normal(0, NOISE_STD_M, (COUNTS[role], 3, 4, 3))
    return bias + noise


def controls(incumbent: np.ndarray) -> np.ndarray:
    values = action_bank(incumbent)
    values[7] = values[BASELINE]
    if values[7].tobytes() != values[BASELINE].tobytes():
        raise ValueError("exact source-selected baseline duplicate required")
    return values


def protocol() -> dict[str, Any]:
    return {
        "schema": "dlolab-slingshot-belief-control-source-v1",
        "role": "prospectively_frozen_public_simulator_source_experiment",
        "particle_worlds": particle_worlds(),
        "prior_weights": prior_weights().tolist(),
        "partitions": {role: sample_worlds(role) for role in COUNTS},
        "sensor_seeds": SENSOR_SEEDS,
        "observation_frames": list(OBSERVATION_FRAMES),
        "observation_rod_nodes": list(OBSERVATION_NODES),
        "observation_sphere_center": True,
        "observations_per_episode": 12,
        "known_metric_3d_identities": True,
        "automatic_perception_claim": False,
        "noise_std_m": NOISE_STD_M,
        "shared_xyz_bias_std_m": BIAS_STD_M,
        "baseline_action": BASELINE,
        "baseline_selection": "best_fixed_action_on_previously_opened_nine_world_screen",
        "action_order_for_regret": list(ORDER),
        "unique_actions": 7,
        "slot_7": "byte_identical_action_5_duplicate",
        "prefix_steps": 300,
        "native_steps": 900,
        "fresh_process_per_native_batch": True,
        "hidden_state_restart": False,
        "native_reward_unchanged": True,
        "calibration_count": 19,
        "calibration_coverage": 0.9,
        "calibration_rank": 18,
        "posterior_quantile": 0.9,
        "calibration_score": "maximum_over_six_alternative_action_regret_errors_per_world",
        "calibration_offset": "nonnegative_rank_18_of_19",
        "numerical_reward_margin": REWARD_MARGIN,
        "position_envelope_m": POSITION_ENVELOPE_M,
        "numerical_envelope_is_coverage_guarantee": False,
        "evaluation_count": 32,
        "arms": list(ARMS),
        "primary_arm": "joint_regret_guard",
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 260905,
        "statistical_unit": "one_fresh_continuous_world_and_one_sensor_draw",
        "stage_order": [
            "prefix_qualification",
            "model_bank",
            "calibration_prefix_predictions",
            "calibration_futures",
            "calibrator",
            "all_evaluation_prefix_decisions",
            "all_evaluation_futures",
            "score",
        ],
        "evaluation_future_before_decision_seal": False,
        "prior_failed_gates": "retained_failed_without_reclassification",
        "gate": {
            "all_32_evaluations_and_19_calibrations": True,
            "all_native_qa": True,
            "nonfallback_at_least": 8,
            "mean_reward_gain_at_least": 0.005,
            "gain_fraction_of_incumbent_excess_over_zero_at_least": 0.1,
            "paired_gain_ci95_lower_vs_incumbent_mean_guard_independent_guard": 0.0,
            "simultaneous_action_coverage_at_least": 0.875,
            "harm_probability_upper95_at_most": 0.1,
            "harm_tolerance_native_reward": REWARD_MARGIN,
            "distinct_oracle_actions_at_least": 2,
        },
        "retry_authorized": False,
        "replacement_authorized": False,
        "real_robot_or_physical_safety_claim": False,
        "official_benchmark_or_sota_claim": False,
        "protected_data_read": False,
        "new_recordings": False,
        "gpu_work": False,
        "push_or_merge": False,
    }


def prefix_observations(prefix: dict[str, np.ndarray]) -> np.ndarray:
    rod = np.asarray(prefix["rod_pos_m"])
    sphere = np.asarray(prefix["sphere_pos_m"])
    if rod.shape != (300, 8, 12, 3) or sphere.shape != (300, 8, 3):
        raise ValueError("only the complete permitted 300-frame prefix is accepted")
    if not np.isfinite(rod).all() or not np.isfinite(sphere).all():
        raise ValueError("nonfinite prefix")
    selected = rod[list(OBSERVATION_FRAMES)][:, :, list(OBSERVATION_NODES)]
    observed = np.concatenate(
        (selected, sphere[list(OBSERVATION_FRAMES), :, None]), axis=2
    )
    return observed.transpose(1, 0, 2, 3).copy()


def infer(
    observation: np.ndarray, bank_prefix: np.ndarray, bank_reward: np.ndarray
) -> dict[str, np.ndarray]:
    if (
        observation.shape != (3, 4, 3)
        or bank_prefix.shape != (27, 3, 4, 3)
        or bank_reward.shape != (27, 7)
    ):
        raise ValueError("frozen observation and model bank layout required")
    if not np.isfinite(bank_reward).all():
        raise ValueError("invalid model reward")
    weights = bias_marginalized_weights(
        observation,
        bank_prefix,
        noise_std_m=NOISE_STD_M,
        shared_bias_std_m=BIAS_STD_M,
        prior_weights=prior_weights(),
    )
    iid = bias_marginalized_weights(
        observation,
        bank_prefix,
        noise_std_m=NOISE_STD_M,
        shared_bias_std_m=0,
        prior_weights=prior_weights(),
    )
    losses = -bank_reward[:, ORDER]
    means = weights @ losses
    raw = np.stack(
        [
            means - means[0],
            action_regret_upper(losses, weights, coupling="independent"),
            action_regret_upper(losses, weights, coupling="joint"),
        ]
    )
    raw[:, 1:] += REWARD_MARGIN
    raw[:, 0] = 0
    return {
        "weights": weights,
        "iid_weights": iid,
        "expected_losses": means,
        "iid_expected_losses": iid @ losses,
        "map_losses": losses[int(np.argmax(weights))],
        "nominal_losses": losses[13],
        "prior_losses": prior_weights() @ losses,
        "raw_upper": raw,
    }


def calibrate(
    parts: list[dict[str, np.ndarray]], rewards: np.ndarray
) -> dict[str, RegretCalibration]:
    if len(parts) != 19 or rewards.shape != (19, 7):
        raise ValueError("all 19 calibration worlds required")
    return {
        mode: calibrate_simultaneous_regret(
            np.stack([p["raw_upper"][i] for p in parts]), -rewards[:, ORDER]
        )
        for i, mode in enumerate(MODES)
    }


def validate_calibration(calibrations: dict[str, RegretCalibration]) -> None:
    if set(calibrations) != set(MODES) or any(
        (c.coverage, c.count, c.rank) != (0.9, 19, 18) for c in calibrations.values()
    ):
        raise ValueError("registered calibration partition required")


def decide(
    parts: dict[str, np.ndarray], calibrations: dict[str, RegretCalibration]
) -> np.ndarray:
    validate_calibration(calibrations)
    result = [BASELINE]
    for key in (
        "nominal_losses",
        "prior_losses",
        "map_losses",
        "expected_losses",
        "iid_expected_losses",
    ):
        values = np.asarray(parts[key])
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ValueError("invalid action means")
        result.append(ORDER[int(np.argmin(values))])
    if parts["raw_upper"].shape != (3, 7):
        raise ValueError("invalid action bounds")
    for index, mode in enumerate(MODES):
        chosen = guarded_action(
            parts["expected_losses"], parts["raw_upper"][index], calibrations[mode]
        )
        result.append(ORDER[chosen])
    return np.asarray(result, dtype=np.int64)


def commands_for_decisions(
    bank: np.ndarray, decisions: np.ndarray
) -> tuple[np.ndarray, ...]:
    if (
        bank.shape != (8, 3, 6)
        or decisions.shape != (len(ARMS),)
        or decisions.dtype.kind not in "iu"
        or np.any((decisions < 0) | (decisions > 6))
    ):
        raise ValueError("invalid command selection")
    originals = tuple(bank[i : i + 1] for i in range(7))
    return tuple(selected_commands(originals, int(index)) for index in decisions)


def native_qa(
    values: dict[str, np.ndarray],
    native: dict,
    expected_controls: np.ndarray,
    prefix: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    rows = split_batch(values, 8)
    if array_digest(values["controls"]) != array_digest(expected_controls):
        raise ValueError("native candidate controls changed")
    metrics = [task_metrics(row) for row in rows]
    if [m["native_reward"] for m in metrics] != native["native_cumulative_reward"]:
        raise ValueError("native reward mismatch")
    fields = ("rod_pos_m", "sphere_pos_m", "cube_pos_m", "gripper_pos_m")
    common = max(
        float(np.max(np.abs(values[name][:300] - values[name][:300, :1])))
        for name in fields
    )
    duplicate = max(
        float(np.max(np.abs(values[name][:, 5] - values[name][:, 7])))
        for name in fields
    )
    prefix_error = (
        None
        if prefix is None
        else max(
            float(np.max(np.abs(values[name][:300, 0] - prefix[name])))
            for name in fields
        )
    )
    fixed = float(
        np.max(
            np.abs(
                values["rod_pos_m"][:, :, [0, 1, 10, 11]]
                - values["rod_pos_m"][:1, :, [0, 1, 10, 11]]
            )
        )
    )
    checks = {
        "common_prefix": common <= POSITION_ENVELOPE_M,
        "duplicate_positions": duplicate <= POSITION_ENVELOPE_M,
        "duplicate_rewards": abs(
            metrics[5]["native_reward"] - metrics[7]["native_reward"]
        )
        <= 0.001,
        "fixed_endpoints": fixed <= 1e-9,
        "sealed_prefix_replay": prefix_error is None
        or prefix_error <= POSITION_ENVELOPE_M,
    }
    return {
        "checks": checks,
        "qa_passed": all(checks.values()),
        "common_prefix_error_m": common,
        "duplicate_error_m": duplicate,
        "prefix_replay_error_m": prefix_error,
        "metrics": metrics,
    }


def score(
    decisions: np.ndarray,
    parts: list[dict[str, np.ndarray]],
    rewards: np.ndarray,
    calibrations: dict[str, RegretCalibration],
    *,
    all_native_qa: bool,
) -> dict[str, Any]:
    validate_calibration(calibrations)
    n = COUNTS["evaluation"]
    if (
        decisions.shape != (n, len(ARMS))
        or decisions.dtype.kind not in "iu"
        or rewards.shape != (n, 7)
        or len(parts) != n
    ):
        raise ValueError("complete 32-world denominator required")
    if (
        not np.isfinite(rewards).all()
        or np.any((decisions < 0) | (decisions > 6))
        or np.any(decisions[:, 0] != BASELINE)
    ):
        raise ValueError("invalid sealed decision or reward")
    if not np.array_equal(
        decisions, np.stack([decide(p, calibrations) for p in parts])
    ):
        raise ValueError("decisions do not reproduce from sealed prefix predictions")
    selected = rewards[np.arange(n)[:, None], decisions]
    baseline = rewards[:, BASELINE]
    gains = selected - baseline[:, None]
    boot = np.random.default_rng(260905).integers(0, n, size=(10000, n))
    arms = {}
    for i, name in enumerate(ARMS):
        harm = int(np.sum(gains[:, i] < -REWARD_MARGIN))
        arms[name] = {
            "mean_native_reward": float(selected[:, i].mean()),
            "mean_gain_over_incumbent": float(gains[:, i].mean()),
            "mean_gain_ci95": np.quantile(
                gains[boot, i].mean(axis=1), [0.025, 0.975]
            ).tolist(),
            "nonfallback_decisions": int(np.sum(decisions[:, i] != BASELINE)),
            "harmful_decisions_beyond_numeric_margin": harm,
            "harm_probability_upper95": one_sided_binomial_upper_bound(harm, n, 0.95),
            "worst_four_mean_regret": float(np.sort(-gains[:, i])[-4:].mean()),
        }
    realized_regret = baseline[:, None] - rewards[:, ORDER]
    coverage = {}
    for i, mode in enumerate(MODES):
        raw = np.stack([p["raw_upper"][i] for p in parts])
        offset = calibrations[mode].offset
        coverage[mode] = (
            1.0
            if offset is None
            else float(
                np.mean(np.all(realized_regret[:, 1:] <= raw[:, 1:] + offset, axis=1))
            )
        )
    paired = {}
    for i, name in enumerate(ARMS[:-1]):
        difference = selected[:, -1] - selected[:, i]
        paired[name] = {
            "mean_gain": float(difference.mean()),
            "ci95": np.quantile(difference[boot].mean(axis=1), [0.025, 0.975]).tolist(),
        }
    primary = arms["joint_regret_guard"]
    excess = float(baseline.mean() - ZERO_REWARD)
    checks = {
        "complete_denominator": True,
        "all_native_qa": bool(all_native_qa),
        "at_least_eight_nonfallback": primary["nonfallback_decisions"] >= 8,
        "absolute_reward_gain": primary["mean_gain_over_incumbent"] >= 0.005,
        "relative_reward_gain": primary["mean_gain_over_incumbent"]
        >= 0.1 * max(0.01, excess),
        "harm_upper_at_most_10pct": primary["harm_probability_upper95"] <= 0.1,
        "action_coverage_at_least_87_5pct": coverage["joint"] >= 0.875,
        "distinct_oracle_actions": len(np.unique(np.argmax(rewards, axis=1))) >= 2,
    }
    for control in ("incumbent", "mean_regret_guard", "independent_regret_guard"):
        checks[f"positive_paired_ci_vs_{control}"] = paired[control]["ci95"][0] > 0
    return {
        "schema": "dlolab-slingshot-belief-control-result-v1",
        "arms": arms,
        "paired_primary_gain": paired,
        "simultaneous_action_coverage": coverage,
        "calibrations": {k: dataclasses.asdict(v) for k, v in calibrations.items()},
        "oracle_mean_native_reward": float(rewards.max(axis=1).mean()),
        "distinct_oracle_actions": int(len(np.unique(np.argmax(rewards, axis=1)))),
        "incumbent_excess_reward_over_zero": excess,
        "posterior_ess_mean": float(
            np.mean([1 / np.sum(p["weights"] ** 2) for p in parts])
        ),
        "checks": checks,
        "source_gate_passed": all(checks.values()),
        "ordinary_evaluations": n,
        "calibration_worlds": 19,
        "technical_failures": 0,
        "replacements": 0,
        "official_benchmark_or_sota_claim": False,
        "real_robot_or_physical_safety_claim": False,
        "protected_data_read": False,
        "new_recordings": False,
    }
