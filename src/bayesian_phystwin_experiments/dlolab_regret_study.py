"""Frozen synthetic action-choice design, separate from public-data targets."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.guard_harm_risk import one_sided_binomial_upper_bound

from .coupled_action_regret import (
    RegretCalibration,
    action_regret_upper,
    bias_marginalized_weights,
    guarded_action,
)
from .dlolab_native import DloLabConfig, DloLabRuntime, NativeSnapshot

MODES = ("joint", "independent", "mean")
ARMS = (
    "hold",
    "nominal_model",
    "posterior_iid_sensor",
    "posterior_shared_bias",
    "mean_regret_guard",
    "independent_regret_guard",
    "joint_regret_guard",
)
OBSERVATION_TIMES = (4, 14, 24)
OBSERVATION_NODES = (3, 6, 10, 15)
PREFIX_STEPS = 25
HORIZON_STEPS = 40
CALIBRATION_COUNT = 39
EVALUATION_COUNT = 64
SEEDS = {"calibration": 260829, "evaluation": 260830}
NOISE_STD_M = 0.003
BIAS_STD_M = 0.012
ACTION_DISTANCE_M = 0.025
EFFORT_WEIGHT = 0.02


def action_offsets() -> np.ndarray:
    values = [np.zeros(3)]
    for y in (-ACTION_DISTANCE_M, 0.0, ACTION_DISTANCE_M):
        for z in (-ACTION_DISTANCE_M, 0.0, ACTION_DISTANCE_M):
            if y != 0 or z != 0:
                values.append(np.array([0.0, y, z]))
    return np.stack(values)


def protocol() -> dict[str, Any]:
    return {
        "schema": "dlolab-coupled-action-regret-source-v1",
        "config": dataclasses.asdict(DloLabConfig()),
        "prefix_steps": PREFIX_STEPS,
        "horizon_steps": HORIZON_STEPS,
        "observation_times": list(OBSERVATION_TIMES),
        "observation_nodes": list(OBSERVATION_NODES),
        "observation_count_per_episode": 12,
        "noise_std_m": NOISE_STD_M,
        "shared_bias_std_m": BIAS_STD_M,
        "particle_bending_scales": [0.5, 1.0, 2.0],
        "particle_lateral_velocities_m_per_s": [-0.30, -0.15, 0.0, 0.15, 0.30],
        "truth_bending_scale": "log-uniform[0.5,2.0]",
        "truth_initial_lateral_velocity_m_per_s": "uniform[-0.25,0.25]",
        "target_offset_m": {
            "x": [-0.005, 0.005],
            "y": [-0.02, 0.02],
            "z": [-0.05, -0.02],
        },
        "action_offsets_m": action_offsets().tolist(),
        "action_profile": "cubic zero-endpoint-velocity ramp",
        "loss": "squared terminal tip-to-target distance + 0.02 squared clamp displacement",
        "effort_weight": EFFORT_WEIGHT,
        "posterior_quantile": 0.90,
        "calibration_coverage": 0.90,
        "calibration_count": CALIBRATION_COUNT,
        "calibration_rank": 36,
        "evaluation_count": EVALUATION_COUNT,
        "seeds": SEEDS,
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 260831,
        "arms": list(ARMS),
        "primary_arm": "joint_regret_guard",
        "unit": "independent simulated world, noise draw, and goal",
        "model_class_misspecification_test": False,
        "continuous_truth_not_restricted_to_particle_grid": True,
        "closed_loop_study": False,
        "physical_safety_claim": False,
        "official_benchmark_claim": False,
        "protected_data_read": False,
        "new_physical_recordings": False,
        "evaluation_futures_before_decision_seal": False,
        "automatic_target_authorization": False,
        "gate": {
            "all_64_successful_no_replacements": True,
            "minimum_nonhold_decisions": 16,
            "minimum_mean_loss_gain_over_hold": 0.10,
            "minimum_gain_relative_to_each_calibrated_control": 1.10,
            "paired_mean_gain_ci_lower_vs_hold_and_controls": 0.0,
            "one_sided_95pct_harm_upper_at_most": 0.10,
            "minimum_simultaneous_action_bound_coverage": 0.85,
            "minimum_distinct_oracle_actions": 3,
            "harm_numerical_tolerance_m2": 1e-12,
        },
    }


def particle_parameters() -> tuple[np.ndarray, np.ndarray]:
    return (
        DloLabConfig().bending_modulus * np.repeat([0.5, 1.0, 2.0], 5),
        np.tile([-0.30, -0.15, 0.0, 0.15, 0.30], 3),
    )


def sample_worlds(role: str) -> dict[str, np.ndarray]:
    if role not in SEEDS:
        raise ValueError("unknown synthetic partition")
    n = CALIBRATION_COUNT if role == "calibration" else EVALUATION_COUNT
    rng = np.random.default_rng(SEEDS[role])
    config = DloLabConfig()
    bending = config.bending_modulus * np.exp(rng.uniform(np.log(0.5), np.log(2.0), n))
    velocity = rng.uniform(-0.25, 0.25, n)
    goal = np.column_stack(
        [
            (config.node_count - 1) * config.interval_m + rng.uniform(-0.005, 0.005, n),
            rng.uniform(-0.02, 0.02, n),
            config.height_m + rng.uniform(-0.05, -0.02, n),
        ]
    )
    bias = rng.normal(0, BIAS_STD_M, (n, 1, 1, 3))
    noise = rng.normal(0, NOISE_STD_M, (n, 3, 4, 3))
    return {
        "bending": bending,
        "velocity": velocity,
        "goals": goal,
        "sensor_error": bias + noise,
    }


def observe_prefix(prefix: np.ndarray, sensor_error: np.ndarray) -> np.ndarray:
    value = np.asarray(prefix)
    expected = (len(sensor_error), PREFIX_STEPS, DloLabConfig().node_count, 3)
    if value.shape != expected or sensor_error.shape != (len(value), 3, 4, 3):
        raise ValueError("only the registered causal prefix is accepted")
    if not np.isfinite(value).all() or not np.isfinite(sensor_error).all():
        raise ValueError("nonfinite prefix observation")
    return value[:, OBSERVATION_TIMES][:, :, OBSERVATION_NODES] + sensor_error


def commands_for_action(clamps: np.ndarray, offset: np.ndarray) -> np.ndarray:
    if clamps.ndim != 3 or clamps.shape[1:] != (2, 3) or offset.shape != (3,):
        raise ValueError("invalid action carrier")
    if not np.isfinite(clamps).all() or not np.isfinite(offset).all():
        raise ValueError("nonfinite action carrier")
    phase = np.linspace(0.0, 1.0, HORIZON_STEPS)
    ramp = 3 * phase**2 - 2 * phase**3
    return clamps[None] + ramp[:, None, None, None] * offset[None, None, None]


def start_prefix(
    upstream: Path,
    bending: np.ndarray,
    velocity: np.ndarray,
) -> tuple[DloLabRuntime, np.ndarray, NativeSnapshot]:
    runtime = DloLabRuntime(
        upstream,
        DloLabConfig(),
        batch_size=len(bending),
        bending_moduli=bending,
        lateral_velocities=velocity,
    )
    commands = np.broadcast_to(
        runtime.initial_positions[:, :2], (PREFIX_STEPS, len(bending), 2, 3)
    ).copy()
    try:
        prefix = runtime.rollout(commands).transpose(1, 0, 2, 3)
        qualify_geometry(prefix)
        return runtime, prefix, runtime.capture()
    except Exception:
        runtime.close()
        raise


def continue_all_actions(
    runtime: DloLabRuntime, snapshot: NativeSnapshot
) -> np.ndarray:
    futures = []
    clamps = runtime.initial_positions[:, :2]
    for index, offset in enumerate(action_offsets()):
        runtime.restore(snapshot)
        commands = commands_for_action(clamps, offset)
        future = runtime.rollout(commands).transpose(1, 0, 2, 3)
        qualify_geometry(future)
        if np.max(np.abs(future[:, :, :2] - commands.transpose(1, 0, 2, 3))) > 1e-10:
            raise RuntimeError("native clamp contract failed")
        futures.append(future)
        print(f"generated native action candidate {index + 1}/9", flush=True)
    snapshot.validate(runtime.config, runtime.model_id)
    return np.stack(futures, axis=1)


def qualify_geometry(trajectory: np.ndarray) -> None:
    value = np.asarray(trajectory)
    if value.ndim != 4 or value.shape[-2:] != (16, 3) or not np.isfinite(value).all():
        raise ValueError("invalid native trajectory")
    lengths = np.linalg.norm(np.diff(value, axis=2), axis=-1)
    if np.max(np.abs(lengths / DloLabConfig().interval_m - 1.0)) > 0.1:
        raise ValueError("native segment-length gate failed")


def loss_table(futures: np.ndarray, goals: np.ndarray) -> np.ndarray:
    if futures.ndim != 5 or futures.shape[1:] != (9, HORIZON_STEPS, 16, 3):
        raise ValueError("invalid frozen action predictions")
    if goals.ndim != 2 or goals.shape[1] != 3:
        raise ValueError("invalid task goals")
    if not np.isfinite(futures).all() or not np.isfinite(goals).all():
        raise ValueError("nonfinite task input")
    error = futures[None, :, :, -1, -1] - goals[:, None, None]
    effort = EFFORT_WEIGHT * np.sum(action_offsets() ** 2, axis=1)
    return np.sum(error**2, axis=-1) + effort[None, None]


def realized_losses(futures: np.ndarray, goals: np.ndarray) -> np.ndarray:
    if goals.ndim != 2 or goals.shape[1] != 3 or len(futures) != len(goals):
        raise ValueError("truth and goal denominator differ")
    if futures.shape[1:] != (9, HORIZON_STEPS, 16, 3):
        raise ValueError("invalid truth action bank")
    if not np.isfinite(futures).all() or not np.isfinite(goals).all():
        raise ValueError("nonfinite truth input")
    error = futures[:, :, -1, -1] - goals[:, None]
    effort = EFFORT_WEIGHT * np.sum(action_offsets() ** 2, axis=1)
    return np.sum(error**2, axis=-1) + effort[None]


def infer_parts(
    observations: np.ndarray,
    goals: np.ndarray,
    bank_prefix: np.ndarray,
    bank_future: np.ndarray,
) -> dict[str, np.ndarray]:
    if observations.shape != (len(goals), 3, 4, 3) or bank_prefix.shape != (
        15,
        PREFIX_STEPS,
        16,
        3,
    ):
        raise ValueError("incorrect permitted observation budget")
    if bank_future.shape != (15, 9, HORIZON_STEPS, 16, 3):
        raise ValueError("incorrect particle bank")
    predicted = bank_prefix[:, OBSERVATION_TIMES][:, :, OBSERVATION_NODES]
    losses = loss_table(bank_future, goals)
    weights, iid_weights, uppers = [], [], []
    for observation, particle_losses in zip(observations, losses, strict=True):
        weight = bias_marginalized_weights(
            observation,
            predicted,
            noise_std_m=NOISE_STD_M,
            shared_bias_std_m=BIAS_STD_M,
        )
        iid_weight = bias_marginalized_weights(
            observation,
            predicted,
            noise_std_m=NOISE_STD_M,
            shared_bias_std_m=0,
        )
        mean = weight @ particle_losses
        uppers.append(
            np.stack(
                [
                    action_regret_upper(particle_losses, weight, coupling="joint"),
                    action_regret_upper(
                        particle_losses, weight, coupling="independent"
                    ),
                    mean - mean[0],
                ]
            )
        )
        weights.append(weight)
        iid_weights.append(iid_weight)
    weight_array, iid_array = np.stack(weights), np.stack(iid_weights)
    return {
        "weights": weight_array,
        "iid_weights": iid_array,
        "expected_losses": np.einsum("np,npa->na", weight_array, losses),
        "iid_expected_losses": np.einsum("np,npa->na", iid_array, losses),
        "nominal_losses": losses[:, 7],
        "raw_upper": np.stack(uppers),
    }


def make_decisions(
    parts: dict[str, np.ndarray], calibrations: dict[str, RegretCalibration]
) -> np.ndarray:
    validate_calibrations(calibrations)
    n = len(parts["expected_losses"])
    for name in ("expected_losses", "iid_expected_losses", "nominal_losses"):
        if parts[name].shape != (n, 9) or not np.isfinite(parts[name]).all():
            raise ValueError("invalid action means")
    if parts["raw_upper"].shape != (n, len(MODES), 9):
        raise ValueError("invalid regret bound array")
    decisions = np.zeros((n, len(ARMS)), dtype=np.int64)
    decisions[:, 1] = np.argmin(parts["nominal_losses"], axis=1)
    decisions[:, 2] = np.argmin(parts["iid_expected_losses"], axis=1)
    decisions[:, 3] = np.argmin(parts["expected_losses"], axis=1)
    for case in range(n):
        for arm, mode in ((4, "mean"), (5, "independent"), (6, "joint")):
            decisions[case, arm] = guarded_action(
                parts["expected_losses"][case],
                parts["raw_upper"][case, MODES.index(mode)],
                calibrations[mode],
            )
    return decisions


def validate_calibrations(calibrations: dict[str, RegretCalibration]) -> None:
    if set(calibrations) != set(MODES):
        raise ValueError("all three registered calibrators are required")
    for value in calibrations.values():
        if (value.coverage, value.count, value.rank) != (0.9, CALIBRATION_COUNT, 36):
            raise ValueError("calibration partition or nominal level changed")


def score_decisions(
    decisions: np.ndarray,
    losses: np.ndarray,
    raw_upper: np.ndarray,
    calibrations: dict[str, RegretCalibration],
) -> dict[str, Any]:
    validate_calibrations(calibrations)
    if decisions.shape != (EVALUATION_COUNT, len(ARMS)) or losses.shape != (
        EVALUATION_COUNT,
        9,
    ):
        raise ValueError("all 64 episodes and all arms are required")
    if decisions.dtype.kind not in "iu" or np.any((decisions < 0) | (decisions >= 9)):
        raise ValueError("invalid sealed decision")
    if not np.isfinite(losses).all() or np.any(decisions[:, 0] != 0):
        raise ValueError("invalid truth or hold fallback")
    if (
        raw_upper.shape != (EVALUATION_COUNT, len(MODES), 9)
        or not np.isfinite(raw_upper).all()
    ):
        raise ValueError("invalid sealed regret bounds")
    if np.any(raw_upper[:, :, 0] != 0) or np.any(losses < 0):
        raise ValueError("invalid baseline regret or task loss")
    if losses[:, 0].mean() <= 0:
        raise ValueError(
            "zero baseline loss makes the registered relative gate undefined"
        )
    deployed = losses[np.arange(EVALUATION_COUNT)[:, None], decisions]
    baseline = losses[:, 0]
    gains = baseline[:, None] - deployed
    rng = np.random.default_rng(260831)
    indices = rng.integers(0, EVALUATION_COUNT, (10000, EVALUATION_COUNT))
    report: dict[str, Any] = {}
    for index, arm in enumerate(ARMS):
        harmful = deployed[:, index] > baseline + 1e-12
        nonhold = decisions[:, index] != 0
        report[arm] = {
            "mean_task_loss_m2": float(deployed[:, index].mean()),
            "mean_gain_over_hold_m2": float(gains[:, index].mean()),
            "mean_loss_change_percent": float(
                100 * (deployed[:, index].mean() / baseline.mean() - 1)
            ),
            "nonhold_decisions": int(nonhold.sum()),
            "harmful_decisions": int(harmful.sum()),
            "harm_probability_upper_95": one_sided_binomial_upper_bound(
                int(harmful.sum()), EVALUATION_COUNT, 0.95
            ),
            "mean_gain_ci95_m2": np.quantile(
                gains[indices, index].mean(axis=1), [0.025, 0.975]
            ).tolist(),
        }
    coverage = {}
    true_regret = losses[:, 1:] - losses[:, :1]
    for mode_index, mode in enumerate(MODES):
        offset = calibrations[mode].offset
        coverage[mode] = (
            1.0
            if offset is None
            else float(
                np.mean(
                    np.all(
                        true_regret <= raw_upper[:, mode_index, 1:] + offset + 1e-12,
                        axis=1,
                    )
                )
            )
        )
    pairwise = {}
    for index in (0, 4, 5):
        difference = deployed[:, index] - deployed[:, 6]
        pairwise[ARMS[index]] = np.quantile(
            difference[indices].mean(axis=1), [0.025, 0.975]
        ).tolist()
    primary = report["joint_regret_guard"]
    checks = {
        "complete_64_episode_denominator": True,
        "at_least_16_nonhold_decisions": primary["nonhold_decisions"] >= 16,
        "at_least_10pct_task_gain_over_hold": primary["mean_loss_change_percent"]
        <= -10.0,
        "harm_probability_upper_95_at_most_10pct": primary["harm_probability_upper_95"]
        <= 0.1,
        "simultaneous_action_coverage_at_least_85pct": coverage["joint"] >= 0.85,
        "at_least_three_distinct_oracle_actions": len(
            np.unique(np.argmin(losses, axis=1))
        )
        >= 3,
    }
    for control in ("mean_regret_guard", "independent_regret_guard"):
        checks[f"gain_10pct_larger_than_{control}"] = primary[
            "mean_gain_over_hold_m2"
        ] >= 1.1 * max(0.0, report[control]["mean_gain_over_hold_m2"])
    for control, interval in pairwise.items():
        checks[f"paired_gain_lower_ci_positive_vs_{control}"] = interval[0] > 0
    return {
        "schema": "dlolab-coupled-action-regret-source-result-v1",
        "arms": report,
        "oracle_mean_task_loss_m2": float(np.min(losses, axis=1).mean()),
        "distinct_oracle_actions": int(len(np.unique(np.argmin(losses, axis=1)))),
        "simultaneous_action_bound_coverage": coverage,
        "paired_gain_ci95_m2": pairwise,
        "checks": checks,
        "source_gate_passed": all(checks.values()),
        "ordinary_evaluation_episodes": EVALUATION_COUNT,
        "technical_failures": 0,
        "replacements": 0,
        "calibration_episodes": CALIBRATION_COUNT,
        "statistical_unit": "independent synthetic world, noise, and goal draw",
        "real_world_confirmation": False,
        "official_benchmark_result": False,
        "closed_loop_result": False,
        "physical_safety_claim": False,
        "automatic_target_authorization": False,
    }
