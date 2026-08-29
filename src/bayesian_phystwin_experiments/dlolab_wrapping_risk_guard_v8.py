"""Prospective native-Linux replication of the wrapping chance guard."""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .dlolab_wrapping_continuous_bayes_v1 import (
    continuous_worlds as failed_v1_worlds,
)
from .dlolab_wrapping_continuous_interp_v2 import (
    continuous_worlds as development_v2_worlds,
)
from .dlolab_wrapping_resolution_ensemble_v3 import (
    continuous_worlds as development_v3_worlds,
)
from .dlolab_wrapping_risk_guard_v4 import continuous_worlds as terminal_v4_worlds
from .dlolab_wrapping_source import (
    FRAMES,
    MEMORY_NAMES,
    N_ACTIONS,
    N_ENVS,
    NODES,
    POSITION_FIELDS,
    POSTS,
    PREFIX_STEPS,
    action_bank,
    native_reward,
    worlds,
)

Array: TypeAlias = NDArray[Any]

WORLD_COUNT = 144
PREFIX_BATCH_COUNT = 16
SENSOR_DRAWS = 4096
WORLD_SEED = 261810
SENSOR_SEED = 261811
BOOTSTRAP_SEED = 261812
BOOTSTRAP_REPLICATES = 20000
QUADRATURE_POINTS_PER_AXIS = 9
SHARED_BIAS_STD_M = 0.005
INDEPENDENT_NOISE_STD_M = 0.002
REWARD_MARGIN = 0.002
PRIMARY_PROBABILITY = 0.975
SENSITIVITY_PROBABILITIES = (0.90, 0.99)
NORMALIZED_LOG_STRETCHING_RANGE = (0.60, 0.995)
NORMALIZED_LOG_BENDING_RANGE = (0.02, 0.70)
ARM_NAMES = (
    "continuous_prior_best_fixed",
    "continuous_bayes",
    "posterior_975_guard",
    "posterior_90_guard",
    "posterior_99_guard",
    "finite_particle_bayes",
    "continuous_map",
)
FAILED_V1_FAILURE_ID = (
    "32f1da52f18bcddc1697931b139b1222692f8eb7b9839b2997b60b9328837692"
)
DEVELOPMENT_V2_RESULT_ID = (
    "2f95e41f51753881cdfe8ca77774cdd60c5d1f92a0552d5570e7299a3244b5bf"
)
DEVELOPMENT_V3_RESULT_ID = (
    "c187d1002f9c0244cea0356a5daac7cf987d3bb754ebb1021d9329f15ac47b19"
)
DEVELOPMENT_V4_DIAGNOSTIC_ID = (
    "281ca205c351d5dbe624e02b127784dc7f8be833e9d6be18d3c75af38b2785ed"
)
TERMINAL_V4_SUMMARY_ID = (
    "ef75f43b46654530ed8a788303feee13c36a3d448566041b42707fe898e07873"
)
TERMINAL_V4_FAILURE_ID = (
    "003be585e995ad8e38818cbb341fe9d39c8344d2dd8bc59d4bd6ace61945443f"
)
RUNTIME_V7_SUMMARY_ID = (
    "24bc06374ff8e5c392304b1b3091e346172b41e1ac8a22081d1efdaa52ff611e"
)


def continuous_worlds() -> list[dict[str, Any]]:
    rng = np.random.default_rng(WORLD_SEED)
    normalized_stretching = rng.uniform(*NORMALIZED_LOG_STRETCHING_RANGE, WORLD_COUNT)
    normalized_bending = rng.uniform(*NORMALIZED_LOG_BENDING_RANGE, WORLD_COUNT)
    stretching = 2e4 * np.exp(np.log(25.0) * normalized_stretching)
    bending = 1e3 * np.exp(np.log(100.0) * normalized_bending)
    return [
        {
            "index": index,
            "stretching_K": float(stretching[index]),
            "bending_E": float(bending[index]),
        }
        for index in range(WORLD_COUNT)
    ]


def preflight_world() -> dict[str, Any]:
    return {
        "index": -1,
        "stretching_K": 100000.0,
        "bending_E": 10000.0,
    }


def validate_continuous_world(world: dict[str, Any]) -> None:
    if (
        set(world) != {"index", "stretching_K", "bending_E"}
        or type(world["index"]) is not int
        or any(
            type(world[name]) is not float or not np.isfinite(world[name])
            for name in ("stretching_K", "bending_E")
        )
        or (
            world != preflight_world()
            and (
                world["index"] not in range(WORLD_COUNT)
                or world != continuous_worlds()[world["index"]]
            )
        )
    ):
        raise ValueError("registered continuous wrapping world required")


def prefix_task(batch: int) -> dict[str, Any]:
    if type(batch) is not int or batch not in range(PREFIX_BATCH_COUNT):
        raise ValueError("registered continuous wrapping prefix batch required")
    indices = list(range(9 * batch, min(9 * batch + 9, WORLD_COUNT)))
    native_indices = indices + [indices[-1]] * (9 - len(indices))
    return {
        "kind": "prefix_only",
        "name": f"prefix-{batch}",
        "batch": batch,
        "world_indices": indices,
        "native_world_indices": native_indices,
    }


def future_task(index: int) -> dict[str, Any]:
    if type(index) is not int or index not in range(WORLD_COUNT):
        raise ValueError("registered continuous wrapping future required")
    return {
        "kind": "all_action_future",
        "name": f"future-{index:03d}",
        "world_index": index,
    }


def protocol() -> dict[str, Any]:
    source_keys = {
        (float(row["stretching_K"]), float(row["bending_E"])) for row in worlds()
    }
    new_keys = {(row["stretching_K"], row["bending_E"]) for row in continuous_worlds()}
    failed_keys = {
        (row["stretching_K"], row["bending_E"]) for row in failed_v1_worlds()
    }
    development_v2_keys = {
        (row["stretching_K"], row["bending_E"]) for row in development_v2_worlds()
    }
    development_v3_keys = {
        (row["stretching_K"], row["bending_E"]) for row in development_v3_worlds()
    }
    terminal_v4_keys = {
        (row["stretching_K"], row["bending_E"]) for row in terminal_v4_worlds()
    }
    if (
        len(new_keys) != WORLD_COUNT
        or new_keys & source_keys
        or new_keys & failed_keys
        or new_keys & development_v2_keys
        or new_keys & development_v3_keys
        or new_keys & terminal_v4_keys
    ):
        raise ValueError("fresh wrapping risk-stress roster changed")
    return {
        "schema": "dlolab-wrapping-risk-guard-source-v8",
        "role": "prospective_native_linux_public_simulator_replication",
        "parent_source_gate_passed": False,
        "parent_gate_reclassified": False,
        "terminal_v1_failure_id": FAILED_V1_FAILURE_ID,
        "terminal_v1_retried_or_scored": False,
        "development_v2_result_id": DEVELOPMENT_V2_RESULT_ID,
        "development_v2_source_gate_passed": False,
        "development_v2_reclassified": False,
        "development_v2_used_for_method_selection": True,
        "development_v3_result_id": DEVELOPMENT_V3_RESULT_ID,
        "development_v3_source_gate_passed": False,
        "development_v3_reclassified": False,
        "development_v3_used_for_method_selection": True,
        "development_v4_diagnostic_id": DEVELOPMENT_V4_DIAGNOSTIC_ID,
        "terminal_v4_summary_id": TERMINAL_V4_SUMMARY_ID,
        "terminal_v4_failure_id": TERMINAL_V4_FAILURE_ID,
        "terminal_v4_retried_or_scored": False,
        "runtime_v7_summary_id": RUNTIME_V7_SUMMARY_ID,
        "runtime_v7_qualification_passed": True,
        "method_class_changed_from_v4": False,
        "replication_of_terminal_v4_method": True,
        "method": "baseline_relative_posterior_probability_guard",
        "posterior_improvement_probability_threshold": PRIMARY_PROBABILITY,
        "posterior_improvement_margin": REWARD_MARGIN,
        "fallback": "continuous_prior_best_fixed_action",
        "registered_fixed_action_index": 4,
        "candidate_selection_used_open_development_outcomes": True,
        "development_lead": {
            "opened_worlds": 80,
            "selected_guard_mean_native_reward": 0.902287226799513,
            "selected_guard_gain_over_fixed": 0.011390258144828434,
            "selected_guard_harmed_worlds": 0,
            "selected_guard_mean_downside_below_fixed": 2.932055162441216e-06,
            "selection_rule": (
                "smallest_registered_probability_with_zero_harmed_worlds_across_v2_v3"
            ),
            "lead_is_not_evidence": True,
        },
        "worlds": continuous_worlds(),
        "world_seed": WORLD_SEED,
        "world_count": WORLD_COUNT,
        "world_distribution": {
            "kind": "source_and_development_defined_action_transition_stress_panel",
            "normalized_log_stretching": list(NORMALIZED_LOG_STRETCHING_RANGE),
            "normalized_log_bending": list(NORMALIZED_LOG_BENDING_RANGE),
            "stretching_K": "20000*25**normalized_log_stretching",
            "bending_E": "1000*100**normalized_log_bending",
            "independent_uniform_draws": True,
        },
        "worlds_disjoint_from_source_v1_v2_v3_and_terminal_v4": True,
        "source_particle_count": 9,
        "interpolation": {
            "coordinates": "normalized_log_stretching_log_bending",
            "source_knots_per_axis": 3,
            "quadrature_points_per_axis": QUADRATURE_POINTS_PER_AXIS,
            "quadrature_count": QUADRATURE_POINTS_PER_AXIS**2,
            "basis": "piecewise_bilinear",
            "prior": "normalized_tensor_trapezoid",
            "interpolation_hyperparameters_fit_from_outcomes": False,
        },
        "task_action_count": N_ACTIONS,
        "native_action_slots": 9,
        "duplicate_native_slot": 8,
        "prefix_native_steps": 600,
        "future_native_steps": 2200,
        "prefix_frames_zero_based": [199, 399, 599],
        "observed_material_identities": list(NODES),
        "observation_units": "world_frame_metres",
        "sensor_draws_per_world": SENSOR_DRAWS,
        "sensor_seed": SENSOR_SEED,
        "shared_translation_bias_sd_m": SHARED_BIAS_STD_M,
        "independent_noise_sd_m": INDEPENDENT_NOISE_STD_M,
        "arms": list(ARM_NAMES),
        "primary_arm": "posterior_975_guard",
        "primary_hypothesis": (
            "posterior_chance_guard_reduces_baseline_relative_downside_while_retaining_value"
        ),
        "statistical_unit": "continuous_world_after_averaging_sensor_draws",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "pre_future_gate": {
            "all_sixteen_prefix_batches_native_qualified": True,
            "guard_nonfixed_sensor_decisions_at_least": 256,
            "guard_differs_from_continuous_bayes_at_least": 256,
            "distinct_guard_actions_at_least": 2,
            "every_nonfixed_guard_decision_has_probability_at_least": (
                PRIMARY_PROBABILITY
            ),
        },
        "source_gate": {
            "all_144_worlds_and_native_qa": True,
            "distinct_oracle_actions_at_least": 2,
            "guard_gain_over_best_fixed_at_least": 0.005,
            "positive_paired_ci95_vs_fixed": True,
            "guard_harmed_worlds": 0,
            "continuous_bayes_harmed_worlds_at_least": 2,
            "guard_reduces_harmed_worlds_by_at_least": 2,
            "guard_reduces_mean_downside_vs_continuous_by_fraction": 0.50,
            "guard_retains_at_least_fraction_of_continuous_gain": 0.50,
            "guard_mean_loss_vs_continuous_at_most": 0.012,
            "oracle_headroom_fraction_at_least": 0.20,
        },
        "stage_order": [
            "all_prefix_only_batches",
            "all_noisy_observations_and_decisions",
            "decision_barrier_and_pre_future_gate",
            "all_action_futures",
            "score",
        ],
        "future_before_decision_barrier": False,
        "failed_v1_payload_used_for_method_or_threshold_selection": False,
        "development_v2_v3_payload_used_for_threshold_and_panel_selection": True,
        "terminal_v4_partial_future_payload_read": False,
        "terminal_v4_retried": False,
        "runtime_v7_arrays_used_for_method_or_threshold_selection": False,
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


def _base_native_qa(
    data: dict[str, Array],
    native: dict[str, Any],
    expected_worlds: list[dict[str, Any]],
    *,
    prefix_only: bool,
) -> dict[str, Any]:
    if len(expected_worlds) != N_ENVS:
        raise ValueError("nine continuous worlds required for native QA")
    for world in expected_worlds:
        validate_continuous_world(world)
    steps = PREFIX_STEPS if prefix_only else 2200
    macro = 3 if prefix_only else 11
    shapes = {
        "rod_pos_m": (steps, N_ENVS, 50, 3),
        "rod_vel_m_s": (steps, N_ENVS, 50, 3),
        "post_pos_m": (steps, N_ENVS, 3, 3),
        "gripper_pos_m": (steps, N_ENVS, 2, 3),
        "robot_qpos": (steps, N_ENVS, 18),
    }
    controls = action_bank()[:, :macro]
    if (
        set(data)
        != set(shapes)
        | set(MEMORY_NAMES)
        | {"controls", "joint_targets", "initial_rod_pos_m"}
        or any(
            data[name].shape != shape or not np.isfinite(data[name]).all()
            for name, shape in shapes.items()
        )
        or data["controls"].dtype != np.float64
        or not np.array_equal(data["controls"], controls)
        or data["joint_targets"].shape != (N_ENVS, macro * 10 + 1, 18)
        or data["initial_rod_pos_m"].shape != (N_ENVS, 50, 3)
        or any(not np.isfinite(value).all() for value in data.values())
        or native.get("native_steps") != steps
        or native.get("worlds") != expected_worlds
        or native.get("prefix_only") is not prefix_only
        or native.get("future_simulated") is prefix_only
        or native.get("reward_exposed") is prefix_only
        or native.get("prefix_reward_excluded") is not prefix_only
        or native.get("device") != "cpu"
        or native.get("twisting_stiffness_zero_preserved") is not True
        or native.get("runtime_camera_rendered") is not False
        or native.get("native_source_modified") is not False
        or native.get("world_realization", {}).get("bending")
        != [world["bending_E"] for world in expected_worlds]
        or native.get("world_realization", {}).get("stretching")
        != [world["stretching_K"] for world in expected_worlds]
    ):
        raise ValueError("continuous native execution contract changed")
    initial = data["initial_rod_pos_m"]
    rest = np.linalg.norm(np.roll(initial, -1, axis=1) - initial, axis=-1)
    if np.any(rest <= 0):
        raise ValueError("initial loop contains collapsed segments")
    ratios = (
        np.linalg.norm(
            np.roll(data["rod_pos_m"], -1, axis=2) - data["rod_pos_m"], axis=-1
        )
        / rest
    )
    attachment = float(
        np.linalg.norm(
            data["rod_pos_m"][:, :, [17, 33]] - data["gripper_pos_m"], axis=-1
        ).max()
    )
    fixed = float(np.abs(data["post_pos_m"] - POSTS).max())
    checks = {
        "finite_extensible_segments": bool(ratios.min() >= 0.25 and ratios.max() <= 3),
        "above_floor": bool(float(data["rod_pos_m"][..., 2].min()) >= -0.01),
        "attached_material_points": bool(attachment <= 0.01),
        "fixed_posts": bool(fixed <= 1e-9),
    }
    return {
        "checks": checks,
        "segment_length_ratio_range": [float(ratios.min()), float(ratios.max())],
        "minimum_rod_height_m": float(data["rod_pos_m"][..., 2].min()),
        "maximum_attachment_distance_m": attachment,
        "fixed_post_error_m": fixed,
    }


def prefix_native_qa(
    data: dict[str, Array],
    native: dict[str, Any],
    expected_worlds: list[dict[str, Any]],
) -> dict[str, Any]:
    result = _base_native_qa(data, native, expected_worlds, prefix_only=True)
    checks = {
        **result["checks"],
        "no_future_reward_exposed": "native_final_reward" not in native
        and "native_cumulative_reward" not in native,
    }
    return {**result, "checks": checks, "qa_passed": bool(all(checks.values()))}


def future_native_qa(
    data: dict[str, Array],
    native: dict[str, Any],
    world: dict[str, Any],
) -> dict[str, Any]:
    expected_worlds = [world] * N_ENVS
    result = _base_native_qa(data, native, expected_worlds, prefix_only=False)
    final = native_reward(data["rod_pos_m"][-1], data["post_pos_m"][-1])
    reported = np.asarray(native.get("native_final_reward"), dtype=np.float64)
    cumulative_reported = np.asarray(
        native.get("native_cumulative_reward"), dtype=np.float32
    )
    if reported.shape != (N_ENVS,) or cumulative_reported.shape != (N_ENVS,):
        raise ValueError("complete native wrapping rewards required")
    cumulative = np.zeros(N_ENVS, dtype=np.float32)
    for frame in range(19, 2200, 20):
        cumulative += native_reward(
            data["rod_pos_m"][frame], data["post_pos_m"][frame]
        ).astype(np.float32) + np.float32(1)
    prefix_error = max(
        float(np.abs(data[name][:PREFIX_STEPS] - data[name][:PREFIX_STEPS, 1:2]).max())
        for name in POSITION_FIELDS
    )
    duplicate_error = max(
        float(np.abs(data[name][:, 1] - data[name][:, 8]).max())
        for name in POSITION_FIELDS
    )
    final_error = float(np.abs(final - reported).max())
    checks = {
        **result["checks"],
        "ordinary_native_success": bool(np.all(reported > -98)),
        "common_prefix": bool(prefix_error <= 1e-5),
        "duplicate_positions": bool(duplicate_error <= 0.001),
        "duplicate_rewards": bool(abs(final[1] - final[8]) <= 0.001),
        "native_final_reward": bool(final_error <= 1e-7),
        "native_cumulative_reward": bool(
            np.array_equal(cumulative, cumulative_reported)
        ),
    }
    return {
        **result,
        "checks": checks,
        "qa_passed": bool(all(checks.values())),
        "maximum_prefix_error_m": prefix_error,
        "maximum_duplicate_coordinate_error_m": duplicate_error,
        "final_reward_reconstruction_error": final_error,
        "final_rewards": final.tolist(),
    }


def prefix_observation(positions: Array) -> Array:
    value = np.asarray(positions, dtype=np.float64)
    if value.shape != (PREFIX_STEPS, N_ENVS, 50, 3) or not np.isfinite(value).all():
        raise ValueError("complete finite wrapping prefix required")
    return value[list(FRAMES)][:, :, list(NODES)].transpose(1, 0, 2, 3).copy()


def _covariance_cholesky() -> Array:
    count = 3 * len(NODES)
    covariance = INDEPENDENT_NOISE_STD_M**2 * np.eye(
        count
    ) + SHARED_BIAS_STD_M**2 * np.ones((count, count))
    return np.linalg.cholesky(covariance)


def _whiten(values: Array, chol: Array) -> Array:
    value = np.asarray(values, dtype=np.float64)
    if value.shape[-2:] != (3 * len(NODES), 3) or not np.isfinite(value).all():
        raise ValueError("finite wrapping prefix coordinates required")
    flat = value.reshape((-1, 3 * len(NODES), 3))
    result = np.empty_like(flat)
    for coordinate in range(3):
        result[:, :, coordinate] = np.linalg.solve(chol, flat[:, :, coordinate].T).T
    return result.reshape(value.shape[:-2] + (9 * len(NODES),))


def _bilinear(values: Array, coordinates: Array) -> Array:
    source = np.asarray(values, dtype=np.float64)
    point = np.asarray(coordinates, dtype=np.float64)
    if (
        source.shape[:2] != (3, 3)
        or point.ndim != 2
        or point.shape[1] != 2
        or not np.isfinite(source).all()
        or not np.isfinite(point).all()
        or np.any((point < 0) | (point > 1))
    ):
        raise ValueError("complete finite log-material interpolation inputs required")
    scaled = point * 2
    low = np.minimum(np.floor(scaled).astype(np.int64), 1)
    fraction = scaled - low
    k0, e0 = low[:, 0], low[:, 1]
    fk = fraction[:, 0].reshape((-1,) + (1,) * (source.ndim - 2))
    fe = fraction[:, 1].reshape((-1,) + (1,) * (source.ndim - 2))
    return (
        (1 - fk) * (1 - fe) * source[k0, e0]
        + fk * (1 - fe) * source[k0 + 1, e0]
        + (1 - fk) * fe * source[k0, e0 + 1]
        + fk * fe * source[k0 + 1, e0 + 1]
    )


def interpolated_bank(source_prefix: Array, source_rewards: Array) -> dict[str, Array]:
    prefix = np.asarray(source_prefix, dtype=np.float64)
    reward = np.asarray(source_rewards, dtype=np.float64)
    if (
        prefix.shape != (9, 3, len(NODES), 3)
        or reward.shape != (9, N_ACTIONS)
        or not np.isfinite(prefix).all()
        or not np.isfinite(reward).all()
    ):
        raise ValueError("complete finite wrapping source bank required")
    axis = np.linspace(0, 1, QUADRATURE_POINTS_PER_AXIS)
    k, e = np.meshgrid(axis, axis, indexing="ij")
    coordinates = np.column_stack((k.ravel(), e.ravel()))
    weights: Array = np.ones(QUADRATURE_POINTS_PER_AXIS, dtype=np.float64)
    weights[[0, -1]] = 0.5
    prior = np.outer(weights, weights).ravel()
    prior /= prior.sum()
    return {
        "normalized_log_material_coordinates": coordinates,
        "prior_weight": prior,
        "prefix_m": _bilinear(prefix.reshape(3, 3, 3, len(NODES), 3), coordinates),
        "native_reward": _bilinear(reward.reshape(3, 3, N_ACTIONS), coordinates),
    }


def posterior_guard_actions(
    expected_reward: Array,
    improvement_probability: Array,
    *,
    threshold: float,
    fixed_action: int,
) -> Array:
    expected = np.asarray(expected_reward, dtype=np.float64)
    probability = np.asarray(improvement_probability, dtype=np.float64)
    if (
        expected.ndim != 2
        or expected.shape[1] != N_ACTIONS
        or probability.shape != expected.shape
        or not np.isfinite(expected).all()
        or not np.isfinite(probability).all()
        or np.any((probability < 0) | (probability > 1))
        or type(threshold) is not float
        or not 0 < threshold < 1
        or type(fixed_action) is not int
        or fixed_action not in range(N_ACTIONS)
    ):
        raise ValueError("complete posterior chance-guard inputs required")
    eligible = probability >= threshold
    eligible[:, fixed_action] = True
    score = np.where(eligible, expected, -np.inf)
    return np.argmax(score, axis=1)


def risk_actions(
    finite_expected: Array,
    continuous_expected: Array,
    improvement_probability: Array,
    map_action: Array,
    fixed_action: int,
) -> Array:
    finite = np.asarray(finite_expected, dtype=np.float64)
    continuous = np.asarray(continuous_expected, dtype=np.float64)
    probability = np.asarray(improvement_probability, dtype=np.float64)
    plugin = np.asarray(map_action)
    if (
        finite.ndim != 2
        or finite.shape[1] != N_ACTIONS
        or continuous.shape != finite.shape
        or probability.shape != finite.shape
        or plugin.shape != (finite.shape[0],)
        or plugin.dtype.kind not in "iu"
        or type(fixed_action) is not int
        or fixed_action not in range(N_ACTIONS)
        or not np.isfinite(finite).all()
        or not np.isfinite(continuous).all()
        or not np.isfinite(probability).all()
        or np.any((probability < 0) | (probability > 1))
        or np.any((plugin < 0) | (plugin >= N_ACTIONS))
    ):
        raise ValueError("complete posterior chance-guard utilities required")
    actions: Array = np.empty((finite.shape[0], len(ARM_NAMES)), dtype=np.int64)
    actions[:, 0] = fixed_action
    actions[:, 1] = np.argmax(continuous, axis=1)
    actions[:, 2] = posterior_guard_actions(
        continuous,
        probability,
        threshold=PRIMARY_PROBABILITY,
        fixed_action=fixed_action,
    )
    actions[:, 3] = posterior_guard_actions(
        continuous,
        probability,
        threshold=SENSITIVITY_PROBABILITIES[0],
        fixed_action=fixed_action,
    )
    actions[:, 4] = posterior_guard_actions(
        continuous,
        probability,
        threshold=SENSITIVITY_PROBABILITIES[1],
        fixed_action=fixed_action,
    )
    actions[:, 5] = np.argmax(finite, axis=1)
    actions[:, 6] = plugin
    return actions


def infer_risk_decisions(
    source_prefix: Array,
    source_rewards: Array,
    truth_prefix: Array,
    *,
    sensor_draws: int,
    sensor_seed: int,
) -> dict[str, Array]:
    source_model = np.asarray(source_prefix, dtype=np.float64)
    source_reward = np.asarray(source_rewards, dtype=np.float64)
    truth = np.asarray(truth_prefix, dtype=np.float64)
    world_count = truth.shape[0] if truth.ndim == 4 else 0
    if (
        source_model.shape != (9, 3, len(NODES), 3)
        or source_reward.shape != (9, N_ACTIONS)
        or truth.shape != (world_count, 3, len(NODES), 3)
        or world_count < 1
        or type(sensor_draws) is not int
        or sensor_draws < 1
        or type(sensor_seed) is not int
        or any(
            not value.all()
            for value in (
                np.isfinite(source_model),
                np.isfinite(source_reward),
                np.isfinite(truth),
            )
        )
    ):
        raise ValueError("complete finite wrapping model and fresh prefixes required")
    interpolated = interpolated_bank(source_model, source_reward)
    continuous_model = interpolated["prefix_m"].reshape(
        QUADRATURE_POINTS_PER_AXIS**2, 3 * len(NODES), 3
    )
    continuous_reward = interpolated["native_reward"]
    continuous_prior = interpolated["prior_weight"]
    finite_model = source_model.reshape(9, 3 * len(NODES), 3)
    truth_flat = truth.reshape(world_count, 3 * len(NODES), 3)
    chol = _covariance_cholesky()
    white_continuous = _whiten(continuous_model, chol)
    white_finite = _whiten(finite_model, chol)
    fixed_action = int(np.argmax(continuous_prior @ continuous_reward))
    decisions: Array = np.empty(
        (world_count, sensor_draws, len(ARM_NAMES)), dtype=np.int64
    )
    decisions[:, :, 0] = fixed_action
    guarded_gain: Array = np.empty((world_count, sensor_draws), dtype=np.float64)
    guarded_probability: Array = np.empty_like(guarded_gain)
    entropy: Array = np.empty_like(guarded_gain)
    continuous_expected_all: Array = np.empty(
        (world_count, sensor_draws, N_ACTIONS), dtype=np.float64
    )
    improvement_probability_all: Array = np.empty_like(continuous_expected_all)
    gain_indicator = (
        continuous_reward - continuous_reward[:, [fixed_action]] >= REWARD_MARGIN
    ).astype(np.float64)
    log_prior = np.log(continuous_prior)
    rng = np.random.default_rng(sensor_seed)
    for world in range(world_count):
        bias = rng.normal(0, SHARED_BIAS_STD_M, (sensor_draws, 1, 3))
        noise = rng.normal(
            0,
            INDEPENDENT_NOISE_STD_M,
            (sensor_draws, 3 * len(NODES), 3),
        )
        for start in range(0, sensor_draws, 128):
            stop = min(start + 128, sensor_draws)
            observation = truth_flat[world] + bias[start:stop] + noise[start:stop]
            white_observation = _whiten(observation, chol)

            finite_delta = white_observation[:, None] - white_finite[None]
            finite_log = -0.5 * np.sum(finite_delta**2, axis=-1)
            finite_weight = np.exp(finite_log - finite_log.max(axis=1, keepdims=True))
            finite_weight /= finite_weight.sum(axis=1, keepdims=True)
            finite_expected = finite_weight @ source_reward

            delta = white_observation[:, None] - white_continuous[None]
            log_weight = -0.5 * np.sum(delta**2, axis=-1) + log_prior
            weight = np.exp(log_weight - log_weight.max(axis=1, keepdims=True))
            weight /= weight.sum(axis=1, keepdims=True)
            map_index = np.argmax(weight, axis=1)
            map_action = np.argmax(continuous_reward[map_index], axis=1)
            expected = weight @ continuous_reward
            probability = np.clip(weight @ gain_indicator, 0.0, 1.0)
            actions = risk_actions(
                finite_expected,
                expected,
                probability,
                map_action,
                fixed_action,
            )
            decisions[world, start:stop] = actions
            continuous_expected_all[world, start:stop] = expected
            improvement_probability_all[world, start:stop] = probability
            guarded_action = actions[:, 2]
            guarded_gain[world, start:stop] = (
                expected[np.arange(stop - start), guarded_action]
                - expected[:, fixed_action]
            )
            guarded_probability[world, start:stop] = probability[
                np.arange(stop - start), guarded_action
            ]
            entropy[world, start:stop] = -np.sum(
                weight * np.log(np.maximum(weight, np.finfo(np.float64).tiny)),
                axis=1,
            )
    return {
        "truth_prefix_m": truth,
        "decisions": decisions,
        "guarded_posterior_expected_gain_over_fixed": guarded_gain,
        "guarded_posterior_improvement_probability": guarded_probability,
        "continuous_posterior_entropy_nats": entropy,
        "continuous_posterior_expected_reward": continuous_expected_all,
        "continuous_posterior_improvement_probability": (improvement_probability_all),
        "continuous_prior_best_fixed_action": np.asarray(fixed_action, dtype=np.int64),
        "quadrature_normalized_log_material_coordinates": interpolated[
            "normalized_log_material_coordinates"
        ],
        "quadrature_prior_weight": continuous_prior,
    }


def infer_decisions(
    source_prefix: Array, source_rewards: Array, truth_prefix: Array
) -> dict[str, Array]:
    truth = np.asarray(truth_prefix, dtype=np.float64)
    if truth.shape != (WORLD_COUNT, 3, len(NODES), 3):
        raise ValueError("complete finite wrapping model and fresh prefixes required")
    return infer_risk_decisions(
        source_prefix,
        source_rewards,
        truth,
        sensor_draws=SENSOR_DRAWS,
        sensor_seed=SENSOR_SEED,
    )


def pre_future_checks(
    decisions: Array,
    guarded_probability: Array,
    *,
    all_prefix_qa: bool,
) -> dict[str, Any]:
    value = np.asarray(decisions)
    probability = np.asarray(guarded_probability, dtype=np.float64)
    if (
        value.shape != (WORLD_COUNT, SENSOR_DRAWS, len(ARM_NAMES))
        or value.dtype.kind not in "iu"
        or probability.shape != (WORLD_COUNT, SENSOR_DRAWS)
        or not np.isfinite(probability).all()
        or np.any((probability < 0) | (probability > 1))
        or np.any((value < 0) | (value >= N_ACTIONS))
        or np.any(value[:, :, 0] != value[0, 0, 0])
    ):
        raise ValueError("complete valid fresh wrapping decisions required")
    primary = ARM_NAMES.index("posterior_975_guard")
    continuous = ARM_NAMES.index("continuous_bayes")
    nonfixed = int(np.count_nonzero(value[:, :, primary] != value[:, :, 0]))
    differs_continuous = int(
        np.count_nonzero(value[:, :, primary] != value[:, :, continuous])
    )
    distinct = int(len(np.unique(value[:, :, primary])))
    nonfixed_mask = value[:, :, primary] != value[:, :, 0]
    minimum_probability = (
        float(probability[nonfixed_mask].min()) if np.any(nonfixed_mask) else 0.0
    )
    checks = {
        "all_sixteen_prefix_batches_native_qualified": bool(all_prefix_qa),
        "guard_nonfixed_sensor_decisions_at_least_256": nonfixed >= 256,
        "guard_differs_from_continuous_bayes_at_least_256": (differs_continuous >= 256),
        "distinct_guard_actions_at_least_2": distinct >= 2,
        "every_nonfixed_guard_decision_has_probability_at_least_0_975": (
            minimum_probability >= PRIMARY_PROBABILITY
        ),
    }
    return {
        "guard_nonfixed_sensor_decisions": nonfixed,
        "guard_differs_from_continuous_bayes": differs_continuous,
        "distinct_guard_actions": distinct,
        "minimum_nonfixed_guard_improvement_probability": minimum_probability,
        "checks": checks,
        "pre_future_gate_passed": bool(all(checks.values())),
    }


def _bootstrap_ci(values: Array) -> list[float]:
    difference = np.asarray(values, dtype=np.float64)
    if difference.shape != (WORLD_COUNT,) or not np.isfinite(difference).all():
        raise ValueError("one finite value per registered wrapping world required")
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, WORLD_COUNT, size=(BOOTSTRAP_REPLICATES, WORLD_COUNT)
    )
    quantiles: Array = np.asarray(
        np.quantile(difference[indices].mean(axis=1), [0.025, 0.975])
    )
    return [float(quantiles[0]), float(quantiles[1])]


def score(decisions: Array, rewards: Array, *, all_native_qa: bool) -> dict[str, Any]:
    decision = np.asarray(decisions)
    reward = np.asarray(rewards, dtype=np.float64)
    if (
        decision.shape != (WORLD_COUNT, SENSOR_DRAWS, len(ARM_NAMES))
        or decision.dtype.kind not in "iu"
        or reward.shape != (WORLD_COUNT, N_ACTIONS)
        or not np.isfinite(reward).all()
        or np.any((decision < 0) | (decision >= N_ACTIONS))
    ):
        raise ValueError("complete fresh wrapping decisions and rewards required")
    selected = np.take_along_axis(reward[:, None, :], decision, axis=2)
    world_reward = selected.mean(axis=1)
    fixed = world_reward[:, 0]
    arms: dict[str, Any] = {}
    for index, name in enumerate(ARM_NAMES):
        gain = world_reward[:, index] - fixed
        arms[name] = {
            "mean_native_reward": float(world_reward[:, index].mean()),
            "mean_gain_over_continuous_prior_best_fixed": float(gain.mean()),
            "gain_ci95": _bootstrap_ci(gain),
            "action_probability": [
                float(np.mean(decision[:, :, index] == action))
                for action in range(N_ACTIONS)
            ],
            "nonfixed_sensor_decisions": int(
                np.count_nonzero(decision[:, :, index] != decision[:, :, 0])
            ),
            "worlds_harmed_beyond_numeric_margin": int(
                np.count_nonzero(gain < -REWARD_MARGIN)
            ),
            "mean_downside_below_fixed": float(np.maximum(-gain, 0).mean()),
            "oracle_action_rate": float(
                np.mean(decision[:, :, index] == np.argmax(reward, axis=1)[:, None])
            ),
        }
    primary_index = ARM_NAMES.index("posterior_975_guard")
    primary = world_reward[:, primary_index]
    paired: dict[str, Any] = {}
    for index, name in enumerate(ARM_NAMES):
        if index == primary_index:
            continue
        difference = primary - world_reward[:, index]
        paired[name] = {
            "mean_gain": float(difference.mean()),
            "ci95": _bootstrap_ci(difference),
        }
    oracle = np.max(reward, axis=1)
    headroom = float(np.mean(oracle - fixed))
    gain = paired["continuous_prior_best_fixed"]["mean_gain"]
    fraction = gain / headroom if headroom > 0 else 0.0
    continuous_gain = arms["continuous_bayes"][
        "mean_gain_over_continuous_prior_best_fixed"
    ]
    retained = gain / continuous_gain if continuous_gain > 0 else 0.0
    guard_harms = arms["posterior_975_guard"]["worlds_harmed_beyond_numeric_margin"]
    continuous_harms = arms["continuous_bayes"]["worlds_harmed_beyond_numeric_margin"]
    guard_downside = np.maximum(fixed - primary, 0)
    continuous_downside = np.maximum(fixed - world_reward[:, 1], 0)
    continuous_downside_mean = float(continuous_downside.mean())
    downside_reduction = (
        float(
            (continuous_downside_mean - guard_downside.mean())
            / continuous_downside_mean
        )
        if continuous_downside_mean > 0
        else 0.0
    )
    harmed_reduction = continuous_harms - guard_harms
    checks = {
        "complete_144_world_denominator": True,
        "all_native_qa": bool(all_native_qa),
        "distinct_oracle_actions_at_least_2": len(np.unique(np.argmax(reward, axis=1)))
        >= 2,
        "guard_gain_over_best_fixed_at_least_0_005": gain >= 0.005,
        "positive_paired_ci95_vs_fixed": paired["continuous_prior_best_fixed"]["ci95"][
            0
        ]
        > 0,
        "guard_harmed_worlds_zero": guard_harms == 0,
        "continuous_bayes_harmed_worlds_at_least_2": continuous_harms >= 2,
        "guard_reduces_harmed_worlds_by_at_least_2": harmed_reduction >= 2,
        "guard_reduces_mean_downside_vs_continuous_by_at_least_50pct": (
            downside_reduction >= 0.50
        ),
        "guard_retains_at_least_50pct_continuous_gain": retained >= 0.50,
        "guard_mean_loss_vs_continuous_at_most_0_012": paired["continuous_bayes"][
            "mean_gain"
        ]
        >= -0.012,
        "captures_at_least_20pct_oracle_headroom": fraction >= 0.20,
    }
    return {
        "schema": "dlolab-wrapping-risk-guard-score-v8",
        "arms": arms,
        "paired_guard_gain": paired,
        "oracle_mean_native_reward": float(oracle.mean()),
        "oracle_headroom_over_continuous_prior_best_fixed": headroom,
        "oracle_headroom_fraction_captured": float(fraction),
        "continuous_gain_fraction_retained": float(retained),
        "continuous_mean_downside_below_fixed": continuous_downside_mean,
        "guard_mean_downside_below_fixed": float(guard_downside.mean()),
        "guard_downside_reduction_fraction_vs_continuous": downside_reduction,
        "guard_downside_reduction_ci95": _bootstrap_ci(
            continuous_downside - guard_downside
        ),
        "continuous_harmed_worlds": continuous_harms,
        "guard_harmed_worlds": guard_harms,
        "harmed_world_reduction": harmed_reduction,
        "distinct_oracle_actions": int(len(np.unique(np.argmax(reward, axis=1)))),
        "checks": checks,
        "source_gate_passed": bool(all(checks.values())),
        "ordinary_worlds": WORLD_COUNT,
        "sensor_draws_per_world": SENSOR_DRAWS,
        "technical_failures": 0,
        "replacements": 0,
        "fresh_successor_automatically_authorized": False,
        "official_benchmark_or_sota_claim": False,
        "real_robot_or_physical_safety_claim": False,
        "protected_data_read": False,
        "new_recordings": False,
    }
