"""Native gripper-coupling source screen, separate from frozen material studies."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from .dlolab_native import array_digest
from .dlolab_slingshot_belief_native import run_registered_worlds

COUPLINGS = (0.3, 0.6, 0.9)
RUN_ORDER = (2, 0, 1)
POSITION_FIELDS = ("rod_pos_m", "sphere_pos_m", "cube_pos_m", "gripper_pos_m")


def protocol() -> dict[str, Any]:
    return {
        "schema": "dlolab-slingshot-contact-realization-source-v1",
        "role": "finite_source_mechanism_screen_not_method_confirmation",
        "native_franka_coupling_coefficients": list(COUPLINGS),
        "execution_order": list(RUN_ORDER),
        "source_world_count": 3,
        "x_offset_m": 0.0,
        "bending_E": 100000.0,
        "stretching_K": 800000.0,
        "prior_weights": [1 / 3] * 3,
        "native_actions_reward_controller_and_release_unchanged": True,
        "coefficient_is_native_tangential_coupling_not_measured_coulomb_friction": True,
        "nominal_replay_position_atol_m": 1e-6,
        "nominal_replay_reward_exact": True,
        "native_steps": 900,
        "branch_frame": 299,
        "observation_frames": [139, 219, 299],
        "observed_nodes": [3, 6, 8],
        "sphere_center_observed": True,
        "independent_noise_sd_m": 0.002,
        "shared_bias_sd_m": 0.005,
        "integration_draws_per_world": 8192,
        "integration_seed": 260909,
        "minimum_information_gain": 0.005,
        "minimum_relative_excess_information_gain": 0.1,
        "minimum_gain_over_map": 0.002,
        "require_not_worse_than_ignored_bias": True,
        "new_recordings": False,
        "gpu_work": False,
        "protected_data_read": False,
        "calibration_or_evaluation_worlds_read": False,
        "retry_authorized": False,
        "method_evaluation_authorized": False,
    }


def task(index: int) -> dict[str, Any]:
    if type(index) is not int or index not in range(3):
        raise ValueError("unregistered contact world")
    return {
        "name": f"coupling-{index}",
        "index": index,
        "franka_coupling": COUPLINGS[index],
        "worlds": [{"x_offset_m": 0.0, "bending_E": 100000.0, "stretching_K": 800000.0}]
        * 8,
    }


def geometry_binding(env: Any, coupling: float) -> dict[str, Any]:
    solver = env.scene.sim.rigid_solver
    values = np.asarray(solver.geoms_info.coup_friction.to_numpy())
    geometries = list(solver.geoms)
    robot_indices = [g.idx for g in env.franka1.geoms]
    if (
        values.ndim != 1
        or not np.isfinite(values).all()
        or not robot_indices
        or len(set(robot_indices)) != len(robot_indices)
        or any(i < 0 or i >= len(values) for i in robot_indices)
        or [g.idx for g in geometries] != list(range(len(values)))
    ):
        raise ValueError("invalid native coupling geometry layout")
    expected = np.asarray([g.coup_friction for g in geometries])
    if not np.array_equal(values, expected) or not np.all(
        values[robot_indices] == coupling
    ):
        raise ValueError("native gripper coupling did not reach the solver")
    others = [g for g in geometries if g.idx not in robot_indices]
    if any(g.coup_friction == 0.9 for g in others):
        raise ValueError("unregistered nominal gripper material outside Franka")
    return {
        "robot_geometry_indices": robot_indices,
        "robot_coupling_values": values[robot_indices].tolist(),
        "all_geometry_coupling_sha256": array_digest(values),
        "nonrobot_geometry": [
            {"index": g.idx, "coupling": float(g.coup_friction)} for g in others
        ],
        "verified_before_native_action": True,
    }


@contextmanager
def contact_adapter(
    material_class: Any, environment_class: Any, coupling: float
) -> Iterator[dict[str, Any]]:
    if type(coupling) is not float or coupling not in COUPLINGS:
        raise ValueError("unregistered native contact coefficient")
    original_material = material_class.__init__
    original_initialize = environment_class.init_cmaes_env
    captured: dict[str, Any] = {"modified_material_count": 0}

    def material_initialize(self: Any, *args: Any, **kwargs: Any) -> None:
        requested = dict(kwargs)
        if requested.get("coup_friction") == 0.9:
            if args or requested != {"needs_coup": True, "coup_friction": 0.9}:
                raise ValueError("native gripper material constructor changed")
            captured["modified_material_count"] += 1
            if captured["modified_material_count"] != 1:
                raise ValueError("more than one registered gripper material")
            requested["coup_friction"] = coupling
        original_material(self, *args, **requested)

    def initialize(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_initialize(self, *args, **kwargs)
        if captured["modified_material_count"] != 1 or "geometry" in captured:
            raise ValueError(
                "contact adapter requires exactly one native initialization"
            )
        captured["geometry"] = geometry_binding(self, coupling)
        return result

    with (
        patch.object(material_class, "__init__", material_initialize),
        patch.object(environment_class, "init_cmaes_env", initialize),
    ):
        yield captured
    if captured["modified_material_count"] != 1 or "geometry" not in captured:
        raise ValueError("incomplete native coupling binding")


def run_contact_world(
    upstream: Path, output: Path, controls: np.ndarray, index: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    spec = task(index)
    sys.path.insert(0, str(upstream))
    sys.path.insert(0, str(upstream / "experiments"))
    gs = importlib.import_module("genesis")
    native = importlib.import_module("envs.env_slingshot")
    with contact_adapter(
        gs.materials.Rigid, native.Train_Env_Slingshot, COUPLINGS[index]
    ) as binding:
        arrays, report = run_registered_worlds(
            upstream, output, controls, spec["worlds"], prefix_only=False
        )
    return arrays, {**report, "contact_realization": binding}


def nominal_replay(
    arrays: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    rewards: list[float],
    expected_rewards: list[float],
) -> dict[str, Any]:
    if (
        any(arrays[name].shape != reference[name].shape for name in POSITION_FIELDS)
        or array_digest(arrays["controls"]) != array_digest(reference["controls"])
        or len(rewards) != 8
        or len(expected_rewards) != 8
    ):
        raise ValueError("complete matched nominal reference required")
    error = max(
        float(np.max(np.abs(arrays[k] - reference[k]))) for k in POSITION_FIELDS
    )
    checks = {
        "nominal_positions_within_1um": error <= 1e-6,
        "nominal_native_reward_exact": rewards == expected_rewards,
    }
    return {
        "checks": checks,
        "maximum_position_error_m": error,
        "passed": all(checks.values()),
    }


def information_value(prefix: np.ndarray, rewards: np.ndarray) -> dict[str, Any]:
    if (
        prefix.shape != (3, 3, 4, 3)
        or rewards.shape != (3, 7)
        or not np.isfinite(prefix).all()
        or not np.isfinite(rewards).all()
    ):
        raise ValueError("complete three-world contact source bank required")
    rng = np.random.default_rng(260909)
    noise = rng.normal(0, 0.005, (8192, 1, 3)) + rng.normal(0, 0.002, (8192, 12, 3))
    history = prefix.reshape(3, 12, 3) - prefix[2].reshape(1, 12, 3)
    covariance = 0.002**2 * np.eye(12) + 0.005**2 * np.ones((12, 12))
    chol = np.linalg.cholesky(covariance)
    history_white = np.linalg.solve(chol, history).reshape(3, 36)
    noise_white = np.linalg.solve(chol, noise).reshape(8192, 36)
    history_iid = history.reshape(3, 36) / 0.002
    noise_iid = noise.reshape(8192, 36) / 0.002
    realized = np.zeros((8192, 3))
    selection = np.zeros((3, 7))
    per_world = np.zeros((3, 3))
    for world in range(3):
        for start in range(0, 8192, 256):
            stop = min(start + 256, 8192)
            decisions = []
            for h, n in ((history_white, noise_white), (history_iid, noise_iid)):
                distance = np.sum((h[world] + n[start:stop, None] - h) ** 2, axis=-1)
                log_weight = -0.5 * distance
                weight = np.exp(log_weight - np.max(log_weight, axis=1, keepdims=True))
                weight /= weight.sum(axis=1, keepdims=True)
                decisions.append(np.argmax(weight @ rewards, axis=1))
                if len(decisions) == 1:
                    decisions.append(
                        np.argmax(rewards[np.argmax(weight, axis=1)], axis=1)
                    )
            chosen = np.stack(decisions, axis=1)
            value = rewards[world, chosen]
            realized[start:stop] += value / 3
            per_world[world] += value.sum(axis=0) / 8192
            for arm in range(3):
                selection[arm] += np.bincount(chosen[:, arm], minlength=7) / (3 * 8192)
    blind_rewards = rewards.mean(axis=0)
    blind = float(np.max(blind_rewards))
    names = ("bias_aware_posterior_mean", "bias_aware_map", "ignored_shared_bias")
    arms = {
        name: {
            "expected_native_reward": float(realized[:, i].mean()),
            "gain_over_best_blind": float(realized[:, i].mean() - blind),
            "monte_carlo_standard_error": float(
                realized[:, i].std(ddof=1) / np.sqrt(8192)
            ),
            "source_world_expected_rewards": per_world[:, i].tolist(),
            "action_probability": selection[i].tolist(),
        }
        for i, name in enumerate(names)
    }
    gain = arms[names[0]]["gain_over_best_blind"]
    over_map = float(np.mean(realized[:, 0] - realized[:, 1]))
    over_ignored = float(np.mean(realized[:, 0] - realized[:, 2]))
    checks = {
        "information_gain_at_least_0_005": gain >= 0.005,
        "information_gain_at_least_10pct_blind_excess": gain
        >= 0.1 * max(0.01, blind - 6.900000095367432),
        "posterior_gain_over_map_at_least_0_002": over_map >= 0.002,
        "not_worse_than_ignored_shared_bias": over_ignored >= 0,
    }
    pair_distance = np.linalg.norm(
        history_white[:, None] - history_white[None], axis=-1
    )
    return {
        "arms": arms,
        "blind_expected_rewards": blind_rewards.tolist(),
        "best_blind_action": int(np.argmax(blind_rewards)),
        "best_blind_reward": blind,
        "perfect_information_reward": float(np.max(rewards, axis=1).mean()),
        "oracle_actions": np.argmax(rewards, axis=1).tolist(),
        "mahalanobis_prefix_distances": pair_distance.tolist(),
        "posterior_gain_over_map": over_map,
        "posterior_gain_over_ignored_bias": over_ignored,
        "checks": checks,
        "source_information_value_passed": all(checks.values()),
        "integration_only_not_independent_control_performance": True,
    }
