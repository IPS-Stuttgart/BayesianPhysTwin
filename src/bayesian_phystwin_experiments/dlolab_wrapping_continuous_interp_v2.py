"""Continuous-material interpolation and Bayes control for native wrapping."""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .dlolab_wrapping_continuous_bayes_v1 import (
    continuous_worlds as failed_v1_worlds,
)
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

WORLD_COUNT = 32
SENSOR_DRAWS = 4096
WORLD_SEED = 261510
SENSOR_SEED = 261511
BOOTSTRAP_SEED = 261512
BOOTSTRAP_REPLICATES = 20000
QUADRATURE_POINTS_PER_AXIS = 9
SHARED_BIAS_STD_M = 0.005
INDEPENDENT_NOISE_STD_M = 0.002
REWARD_MARGIN = 0.002
ARM_NAMES = (
    "continuous_prior_best_fixed",
    "finite_particle_bayes",
    "continuous_map",
    "continuous_bayes",
    "ignored_shared_bias_continuous_bayes",
)
FAILED_V1_FAILURE_ID = (
    "32f1da52f18bcddc1697931b139b1222692f8eb7b9839b2997b60b9328837692"
)


def continuous_worlds() -> list[dict[str, Any]]:
    rng = np.random.default_rng(WORLD_SEED)
    stretching = np.exp(
        rng.uniform(np.log(2e4), np.log(5e5), WORLD_COUNT)
    )
    bending = np.exp(rng.uniform(np.log(1e3), np.log(1e5), WORLD_COUNT))
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
            type(world[name]) is not float
            or not np.isfinite(world[name])
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
    if type(batch) is not int or batch not in range(4):
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
        "name": f"future-{index:02d}",
        "world_index": index,
    }


def protocol() -> dict[str, Any]:
    source_keys = {
        (float(row["stretching_K"]), float(row["bending_E"])) for row in worlds()
    }
    new_keys = {
        (row["stretching_K"], row["bending_E"]) for row in continuous_worlds()
    }
    failed_keys = {
        (row["stretching_K"], row["bending_E"])
        for row in failed_v1_worlds()
    }
    if (
        len(new_keys) != WORLD_COUNT
        or new_keys & source_keys
        or new_keys & failed_keys
    ):
        raise ValueError("fresh continuous wrapping roster changed")
    return {
        "schema": "dlolab-wrapping-continuous-interp-source-v2",
        "role": "prospective_public_simulator_source_test",
        "parent_source_gate_passed": False,
        "parent_gate_reclassified": False,
        "terminal_v1_failure_id": FAILED_V1_FAILURE_ID,
        "terminal_v1_retried_or_scored": False,
        "method_class_changed": True,
        "method": "bilinear_log_material_interpolation_and_posterior_expected_reward",
        "worlds": continuous_worlds(),
        "world_seed": WORLD_SEED,
        "world_count": WORLD_COUNT,
        "world_distribution": {
            "stretching_K": "log_uniform[20000,500000]",
            "bending_E": "log_uniform[1000,100000]",
        },
        "worlds_disjoint_from_nine_particle_support_and_failed_v1": True,
        "source_particle_count": 9,
        "interpolation": {
            "coordinates": "normalized_log_stretching_log_bending",
            "source_knots_per_axis": 3,
            "quadrature_points_per_axis": QUADRATURE_POINTS_PER_AXIS,
            "quadrature_count": QUADRATURE_POINTS_PER_AXIS**2,
            "basis": "piecewise_bilinear",
            "prior": "normalized_tensor_trapezoid",
            "hyperparameters_fit_from_failed_v1_outcomes": False,
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
        "primary_arm": "continuous_bayes",
        "primary_hypothesis": (
            "continuous_interpolation_improves_off_grid_posterior_decisions"
        ),
        "statistical_unit": "continuous_world_after_averaging_sensor_draws",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "pre_future_gate": {
            "all_four_prefix_batches_native_qualified": True,
            "continuous_bayes_nonfixed_sensor_decisions_at_least": 256,
            "continuous_bayes_differs_from_finite_bayes_at_least": 256,
            "continuous_bayes_differs_from_map_at_least": 256,
            "distinct_continuous_bayes_actions_at_least": 2,
        },
        "source_gate": {
            "all_32_worlds_and_native_qa": True,
            "distinct_oracle_actions_at_least": 2,
            "continuous_bayes_gain_over_best_fixed_at_least": 0.01,
            "continuous_bayes_gain_over_finite_bayes_at_least": 0.002,
            "continuous_bayes_gain_over_map_at_least": 0.001,
            "continuous_bayes_gain_over_ignored_bias_at_least": 0.003,
            "paired_ci95_lower_for_all_four_comparisons_above": 0.0,
            "oracle_headroom_fraction_at_least": 0.30,
            "continuous_bayes_harms_no_more_worlds_than_finite_or_map": True,
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
        "finite_extensible_segments": bool(
            ratios.min() >= 0.25 and ratios.max() <= 3
        ),
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
    result = _base_native_qa(
        data, native, expected_worlds, prefix_only=True
    )
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
    result = _base_native_qa(
        data, native, expected_worlds, prefix_only=False
    )
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
        float(
            np.abs(
                data[name][:PREFIX_STEPS]
                - data[name][:PREFIX_STEPS, 1:2]
            ).max()
        )
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
    covariance = (
        INDEPENDENT_NOISE_STD_M**2 * np.eye(count)
        + SHARED_BIAS_STD_M**2 * np.ones((count, count))
    )
    return np.linalg.cholesky(covariance)


def _whiten(values: Array, chol: Array) -> Array:
    value = np.asarray(values, dtype=np.float64)
    if value.shape[-2:] != (3 * len(NODES), 3) or not np.isfinite(value).all():
        raise ValueError("finite wrapping prefix coordinates required")
    flat = value.reshape((-1, 3 * len(NODES), 3))
    result = np.empty_like(flat)
    for coordinate in range(3):
        result[:, :, coordinate] = np.linalg.solve(
            chol, flat[:, :, coordinate].T
        ).T
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
        "native_reward": _bilinear(
            reward.reshape(3, 3, N_ACTIONS), coordinates
        ),
    }


def infer_decisions(
    source_prefix: Array, source_rewards: Array, truth_prefix: Array
) -> dict[str, Array]:
    source_model = np.asarray(source_prefix, dtype=np.float64)
    source_reward = np.asarray(source_rewards, dtype=np.float64)
    truth = np.asarray(truth_prefix, dtype=np.float64)
    if (
        source_model.shape != (9, 3, len(NODES), 3)
        or source_reward.shape != (9, N_ACTIONS)
        or truth.shape != (WORLD_COUNT, 3, len(NODES), 3)
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
    truth_flat = truth.reshape(WORLD_COUNT, 3 * len(NODES), 3)
    chol = _covariance_cholesky()
    white_continuous = _whiten(continuous_model, chol)
    white_finite = _whiten(finite_model, chol)
    fixed_action = int(np.argmax(continuous_prior @ continuous_reward))
    decisions: Array = np.empty(
        (WORLD_COUNT, SENSOR_DRAWS, len(ARM_NAMES)), dtype=np.int64
    )
    decisions[:, :, 0] = fixed_action
    expected_gain: Array = np.empty((WORLD_COUNT, SENSOR_DRAWS), dtype=np.float64)
    entropy: Array = np.empty_like(expected_gain)
    log_prior = np.log(continuous_prior)
    rng = np.random.default_rng(SENSOR_SEED)
    for world in range(WORLD_COUNT):
        bias = rng.normal(0, SHARED_BIAS_STD_M, (SENSOR_DRAWS, 1, 3))
        noise = rng.normal(
            0,
            INDEPENDENT_NOISE_STD_M,
            (SENSOR_DRAWS, 3 * len(NODES), 3),
        )
        for start in range(0, SENSOR_DRAWS, 128):
            stop = min(start + 128, SENSOR_DRAWS)
            count = stop - start
            observation = truth_flat[world] + bias[start:stop] + noise[start:stop]
            white_observation = _whiten(observation, chol)

            finite_delta = white_observation[:, None] - white_finite[None]
            finite_log = -0.5 * np.sum(finite_delta**2, axis=-1)
            finite_weight = np.exp(
                finite_log - finite_log.max(axis=1, keepdims=True)
            )
            finite_weight /= finite_weight.sum(axis=1, keepdims=True)
            finite_action = np.argmax(finite_weight @ source_reward, axis=1)

            delta = white_observation[:, None] - white_continuous[None]
            log_weight = -0.5 * np.sum(delta**2, axis=-1) + log_prior
            weight = np.exp(log_weight - log_weight.max(axis=1, keepdims=True))
            weight /= weight.sum(axis=1, keepdims=True)
            map_index = np.argmax(weight, axis=1)
            map_action = np.argmax(continuous_reward[map_index], axis=1)
            expected = weight @ continuous_reward
            bayes_action = np.argmax(expected, axis=1)

            ignored_delta = (
                observation[:, None] - continuous_model[None]
            ).reshape(count, QUADRATURE_POINTS_PER_AXIS**2, -1)
            ignored_delta /= INDEPENDENT_NOISE_STD_M
            ignored_log = (
                -0.5 * np.sum(ignored_delta**2, axis=-1) + log_prior
            )
            ignored_weight = np.exp(
                ignored_log - ignored_log.max(axis=1, keepdims=True)
            )
            ignored_weight /= ignored_weight.sum(axis=1, keepdims=True)
            ignored_action = np.argmax(
                ignored_weight @ continuous_reward, axis=1
            )

            decisions[world, start:stop, 1] = finite_action
            decisions[world, start:stop, 2] = map_action
            decisions[world, start:stop, 3] = bayes_action
            decisions[world, start:stop, 4] = ignored_action
            expected_gain[world, start:stop] = (
                expected[np.arange(count), bayes_action] - expected[:, fixed_action]
            )
            entropy[world, start:stop] = -np.sum(
                weight * np.log(np.maximum(weight, np.finfo(np.float64).tiny)),
                axis=1,
            )
    return {
        "truth_prefix_m": truth,
        "decisions": decisions,
        "posterior_expected_gain_over_fixed": expected_gain,
        "posterior_entropy_nats": entropy,
        "continuous_prior_best_fixed_action": np.asarray(
            fixed_action, dtype=np.int64
        ),
        "quadrature_normalized_log_material_coordinates": interpolated[
            "normalized_log_material_coordinates"
        ],
        "quadrature_prior_weight": continuous_prior,
    }


def pre_future_checks(decisions: Array, *, all_prefix_qa: bool) -> dict[str, Any]:
    value = np.asarray(decisions)
    if (
        value.shape != (WORLD_COUNT, SENSOR_DRAWS, len(ARM_NAMES))
        or value.dtype.kind not in "iu"
        or np.any((value < 0) | (value >= N_ACTIONS))
        or np.any(value[:, :, 0] != value[0, 0, 0])
    ):
        raise ValueError("complete valid fresh wrapping decisions required")
    nonfixed = int(np.count_nonzero(value[:, :, 3] != value[:, :, 0]))
    differs_finite = int(np.count_nonzero(value[:, :, 3] != value[:, :, 1]))
    differs_map = int(np.count_nonzero(value[:, :, 3] != value[:, :, 2]))
    distinct = int(len(np.unique(value[:, :, 3])))
    checks = {
        "all_four_prefix_batches_native_qualified": bool(all_prefix_qa),
        "continuous_bayes_nonfixed_sensor_decisions_at_least_256": nonfixed >= 256,
        "continuous_bayes_differs_from_finite_bayes_at_least_256": (
            differs_finite >= 256
        ),
        "continuous_bayes_differs_from_map_at_least_256": differs_map >= 256,
        "distinct_continuous_bayes_actions_at_least_2": distinct >= 2,
    }
    return {
        "continuous_bayes_nonfixed_sensor_decisions": nonfixed,
        "continuous_bayes_differs_from_finite_bayes": differs_finite,
        "continuous_bayes_differs_from_map": differs_map,
        "distinct_continuous_bayes_actions": distinct,
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
            "oracle_action_rate": float(
                np.mean(
                    decision[:, :, index]
                    == np.argmax(reward, axis=1)[:, None]
                )
            ),
        }
    primary_index = ARM_NAMES.index("continuous_bayes")
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
    checks = {
        "complete_32_world_denominator": True,
        "all_native_qa": bool(all_native_qa),
        "distinct_oracle_actions_at_least_2": len(
            np.unique(np.argmax(reward, axis=1))
        )
        >= 2,
        "continuous_bayes_gain_over_best_fixed_at_least_0_01": gain >= 0.01,
        "continuous_bayes_gain_over_finite_bayes_at_least_0_002": paired[
            "finite_particle_bayes"
        ]["mean_gain"]
        >= 0.002,
        "continuous_bayes_gain_over_map_at_least_0_001": paired[
            "continuous_map"
        ]["mean_gain"]
        >= 0.001,
        "continuous_bayes_gain_over_ignored_bias_at_least_0_003": paired[
            "ignored_shared_bias_continuous_bayes"
        ]["mean_gain"]
        >= 0.003,
        "positive_paired_ci95_vs_fixed": paired[
            "continuous_prior_best_fixed"
        ]["ci95"][0]
        > 0,
        "positive_paired_ci95_vs_finite_bayes": paired["finite_particle_bayes"][
            "ci95"
        ][0]
        > 0,
        "positive_paired_ci95_vs_map": paired["continuous_map"]["ci95"][0] > 0,
        "positive_paired_ci95_vs_ignored_bias": paired[
            "ignored_shared_bias_continuous_bayes"
        ]["ci95"][0]
        > 0,
        "captures_at_least_30pct_oracle_headroom": fraction >= 0.30,
        "continuous_bayes_harms_no_more_worlds_than_finite": arms[
            "continuous_bayes"
        ]["worlds_harmed_beyond_numeric_margin"]
        <= arms["finite_particle_bayes"]["worlds_harmed_beyond_numeric_margin"],
        "continuous_bayes_harms_no_more_worlds_than_map": arms[
            "continuous_bayes"
        ]["worlds_harmed_beyond_numeric_margin"]
        <= arms["continuous_map"]["worlds_harmed_beyond_numeric_margin"],
    }
    return {
        "schema": "dlolab-wrapping-continuous-interp-score-v2",
        "arms": arms,
        "paired_continuous_bayes_gain": paired,
        "oracle_mean_native_reward": float(oracle.mean()),
        "oracle_headroom_over_continuous_prior_best_fixed": headroom,
        "oracle_headroom_fraction_captured": float(fraction),
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
