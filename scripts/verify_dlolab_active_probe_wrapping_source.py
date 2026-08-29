#!/usr/bin/env python3
"""Independent arithmetic verification of the frozen active-wrapping source study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin_experiments.dlolab_active_wrapping_source import (
    DECISION_DRAWS,
    N_ACTIONS,
    N_PROBES,
    NOISE_SEED,
    OBSERVATION_FRAMES,
    OBSERVED_NODES,
    PROBE_DRAWS,
    active_decision_gate,
    native_qa,
    prefix_observation,
    protocol,
    task,
)
from bayesian_phystwin_experiments.dlolab_native import file_digest
from bayesian_phystwin_experiments.dlolab_regret_artifacts import (
    read_record,
    write_record,
)
from bayesian_phystwin_experiments.dlolab_slingshot_process import load_native_bundle

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("/home/fpfaff/source-only/dlolab-active-probe-wrapping-source-v1")
PASSIVE_RESULT = ROOT / "results/sota/dlolab_wrapping_belief_source_v1/result.json"


def angular_reward(positions: np.ndarray, posts: np.ndarray) -> np.ndarray:
    relative = positions[..., :, None, :2] - posts[..., None, :, :2]
    angle = np.unwrap(np.arctan2(relative[..., 1], relative[..., 0]), axis=-2)
    closing = np.arctan2(relative[..., 0, 1], relative[..., 0, 0])
    last = np.arctan2(relative[..., -1, 1], relative[..., -1, 0])
    closing = last + np.angle(np.exp(1j * (closing - last)))
    turns = (closing - angle[..., 0, :]) / (2 * np.pi)
    distance = np.linalg.norm(
        positions[..., :, None, :] - posts[..., None, :, :], axis=-1
    ).min(axis=-2)
    return (
        1
        - np.mean((np.abs(turns) - 1) ** 2, axis=-1)
        - np.maximum(distance - 0.015, 0).sum(axis=-1)
    )


def precision(count: int) -> np.ndarray:
    independent = 0.002**2
    shared = 0.005**2
    return np.eye(count) / independent - (
        shared / (independent * (independent + count * shared))
    ) * np.ones((count, count))


def log_weights(delta: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    value = -0.5 * np.einsum("...ic,ij,...jc->...", delta, matrix, delta)
    weight = np.exp(value - value.max(axis=-1, keepdims=True))
    return weight / weight.sum(axis=-1, keepdims=True)


def probe_information(prefix: np.ndarray) -> dict[str, Any]:
    count = len(OBSERVATION_FRAMES) * len(OBSERVED_NODES)
    matrix = precision(count)
    rng = np.random.default_rng(NOISE_SEED)
    shared = rng.normal(0, 0.005, (PROBE_DRAWS, 1, 3))
    independent = rng.normal(0, 0.002, (N_PROBES, PROBE_DRAWS, count, 3))
    entropy = np.zeros(N_PROBES)
    correct = np.zeros(N_PROBES)
    for probe in range(N_PROBES):
        means = prefix[:, probe].reshape(9, count, 3)
        for world in range(9):
            for start in range(0, PROBE_DRAWS, 256):
                observation = (
                    means[world]
                    + shared[start : start + 256]
                    + independent[probe, start : start + 256]
                )
                weight = log_weights(observation[:, None] - means, matrix)
                entropy[probe] -= np.sum(
                    weight * np.log(np.maximum(weight, 1e-300))
                ) / (9 * PROBE_DRAWS)
                correct[probe] += np.sum(np.argmax(weight, axis=1) == world) / (
                    9 * PROBE_DRAWS
                )
    information = np.log(9.0) - entropy
    selected = int(np.argmax(information))
    checks = {
        "selected_probe_is_not_null": selected != 0,
        "mutual_information_gain": float(information[selected] - information[0])
        >= 0.05,
        "material_classification_gain": float(correct[selected] - correct[0]) >= 0.05,
    }
    return {
        "selected_probe_index": selected,
        "selected_probe_name": protocol()["probe_names"][selected],
        "expected_posterior_entropy_nats": entropy.tolist(),
        "mutual_information_nats": information.tolist(),
        "material_classification_accuracy": correct.tolist(),
        "mutual_information_gain_over_null_nats": float(
            information[selected] - information[0]
        ),
        "classification_gain_over_null": float(correct[selected] - correct[0]),
        "checks": checks,
        "passed": all(checks.values()),
        "future_reward_used": False,
    }


def decision_value(
    prefix: np.ndarray, rewards: np.ndarray, seed_offset: int
) -> dict[str, Any]:
    count = len(OBSERVATION_FRAMES) * len(OBSERVED_NODES)
    matrix = precision(count)
    rng = np.random.default_rng(NOISE_SEED + seed_offset)
    noise = rng.normal(0, 0.005, (DECISION_DRAWS, 1, 3)) + rng.normal(
        0, 0.002, (DECISION_DRAWS, count, 3)
    )
    means = prefix.reshape(9, count, 3)
    independent_matrix = np.eye(count) / 0.002**2
    realized = np.zeros((DECISION_DRAWS, 3))
    per_world = np.zeros((9, 3))
    selection = np.zeros((3, N_ACTIONS))
    for world in range(9):
        for start in range(0, DECISION_DRAWS, 256):
            observation = means[world] + noise[start : start + 256]
            aware = log_weights(observation[:, None] - means, matrix)
            ignored = log_weights(observation[:, None] - means, independent_matrix)
            selected = np.stack(
                [
                    np.argmax(aware @ rewards, axis=1),
                    np.argmax(rewards[np.argmax(aware, axis=1)], axis=1),
                    np.argmax(ignored @ rewards, axis=1),
                ],
                axis=1,
            )
            values = rewards[world, selected]
            realized[start : start + 256] += values / 9
            per_world[world] += values.sum(axis=0) / DECISION_DRAWS
            for arm in range(3):
                selection[arm] += np.bincount(selected[:, arm], minlength=N_ACTIONS) / (
                    9 * DECISION_DRAWS
                )
    fixed_action = int(np.argmax(rewards.mean(axis=0)))
    fixed = float(rewards[:, fixed_action].mean())
    means_value = realized.mean(axis=0)
    oracle = rewards.max(axis=1)
    names = ("bias_aware_bayes", "bias_aware_map", "ignored_shared_bias")
    return {
        "arms": {
            name: {
                "expected_native_final_reward": float(means_value[index]),
                "gain_over_best_fixed": float(means_value[index] - fixed),
                "monte_carlo_standard_error": float(
                    realized[:, index].std(ddof=1) / np.sqrt(DECISION_DRAWS)
                ),
                "source_world_expected_rewards": per_world[:, index].tolist(),
                "action_probability": selection[index].tolist(),
            }
            for index, name in enumerate(names)
        },
        "best_fixed_action": fixed_action,
        "best_fixed_reward": fixed,
        "prefix_hold_reward": float(rewards[:, 0].mean()),
        "oracle_reward": float(oracle.mean()),
        "oracle_actions": np.argmax(rewards, axis=1).tolist(),
        "source_worlds_are_prior_support_not_independent_evaluation": True,
        "monte_carlo_only_integrates_assumed_sensor_noise": True,
    }


def maximum_difference(left: Any, right: Any) -> float:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise ValueError("verification dictionary fields changed")
        return max(
            (maximum_difference(left[key], right[key]) for key in left), default=0
        )
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ValueError("verification list length changed")
        return max(
            (maximum_difference(a, b) for a, b in zip(left, right, strict=True)),
            default=0,
        )
    if (
        isinstance(left, bool)
        or isinstance(right, bool)
        or isinstance(left, str)
        or isinstance(right, str)
        or left is None
        or right is None
    ):
        if left != right:
            raise ValueError(f"verification scalar changed: {left!r} != {right!r}")
        return 0.0
    return abs(float(left) - float(right))


def load_stage(lock: dict, stage: str, probe_index: int | None):
    rows = []
    maximum_reward_difference = 0.0
    for index in range(11):
        spec = task(stage, index, probe_index)
        directory = OUTPUT / spec["name"]
        claim = read_record(directory / "claim.json")
        seal = read_record(directory / "seal.json")
        if (
            claim["lock_id"] != lock["artifact_id"]
            or seal["claim_id"] != claim["artifact_id"]
            or seal["task"] != spec
        ):
            raise ValueError("native custody changed")
        data = load_native_bundle(directory, seal["bundle"])
        independent_reward = angular_reward(
            data["rod_pos_m"][-1], data["post_pos_m"][-1]
        )
        maximum_reward_difference = max(
            maximum_reward_difference,
            float(
                np.abs(
                    independent_reward
                    - np.asarray(seal["native"]["native_final_reward"])
                ).max()
            ),
        )
        qa = native_qa(data, seal["native"], spec["world"], stage, probe_index)
        if not qa["passed"] or read_record(directory / "qa.json")["qa"] != qa:
            raise ValueError("stored native qualification changed")
        if index not in (1, 2):
            rows.append((seal, data))
    rows.sort(key=lambda row: row[0]["task"]["world"]["index"])
    return rows, maximum_reward_difference


def verify() -> dict[str, Any]:
    if OUTPUT.is_symlink() or not OUTPUT.is_dir():
        raise ValueError("registered completed active-wrapping output required")
    lock = read_record(OUTPUT / "lock.json")
    result = read_record(OUTPUT / "result.json")
    if (
        lock["protocol"] != protocol()
        or lock["output_root"] != str(OUTPUT)
        or result["lock_id"] != lock["artifact_id"]
        or result["retry_authorized"] is not False
        or result["protected_data_read"] is not False
        or len(lock["source_sha256"]) != 18
        or not {
            "src/bayesian_phystwin_experiments/dlolab_active_wrapping_source.py",
            "src/bayesian_phystwin_experiments/dlolab_active_wrapping_native.py",
            "scripts/remote/run_dlolab_active_probe_wrapping_source.py",
            "scripts/verify_dlolab_active_probe_wrapping_source.py",
        }.issubset(lock["source_sha256"])
        or any(
            file_digest(ROOT / path) != digest
            for path, digest in lock["source_sha256"].items()
        )
    ):
        raise ValueError("frozen active-wrapping lock/result changed")
    probe_rows, reward_difference = load_stage(lock, "probe", None)
    prefix = np.stack(
        [prefix_observation(row[1]["rod_pos_m"], "probe") for row in probe_rows]
    )
    probe_metrics = probe_information(prefix)
    maximum = maximum_difference(probe_metrics, result["probe_metrics"])
    if result["status"] == "probe_information_gate_failed":
        if result["source_gate_passed"] is not False or probe_metrics["passed"]:
            raise ValueError("failed probe gate accounting changed")
        return {
            "schema": "dlolab-active-probe-wrapping-second-arithmetic-v1",
            "lock_id": lock["artifact_id"],
            "result_id": result["artifact_id"],
            "verified_native_trajectories": 99,
            "verified_source_files": len(lock["source_sha256"]),
            "maximum_reward_formula_difference": reward_difference,
            "maximum_decision_arithmetic_difference": maximum,
            "probe_selection_recomputed_without_future_reward": True,
            "native_physics_reexecuted": False,
            "independent_human_review": False,
            "source_gate_passed": False,
            "passed": reward_difference <= 1e-7 and maximum <= 1e-10,
            "protected_data_read": False,
        }
    if result["status"] != "complete":
        raise ValueError("verifier currently requires completed matched study")
    selected = probe_metrics["selected_probe_index"]
    null_rows, difference = load_stage(lock, "baseline", 0)
    reward_difference = max(reward_difference, difference)
    active_rows, difference = load_stage(lock, "active", selected)
    reward_difference = max(reward_difference, difference)

    def bank(rows, stage):
        return (
            np.stack([prefix_observation(row[1]["rod_pos_m"], stage) for row in rows]),
            np.asarray(
                [row[0]["native"]["native_final_reward"][:N_ACTIONS] for row in rows]
            ),
        )

    null_prefix, null_reward = bank(null_rows, "baseline")
    active_prefix, active_reward = bank(active_rows, "active")
    null_metrics = decision_value(null_prefix, null_reward, 100)
    active_metrics = decision_value(active_prefix, active_reward, 100)
    maximum = max(
        maximum,
        maximum_difference(null_metrics, result["null_metrics"]),
        maximum_difference(active_metrics, result["active_metrics"]),
    )
    passive = read_record(PASSIVE_RESULT)
    decision = active_decision_gate(active_metrics, null_metrics, passive)
    maximum = max(maximum, maximum_difference(decision, result["decision"]))
    if result["source_gate_passed"] != decision["passed"]:
        raise ValueError("source decision changed")
    return {
        "schema": "dlolab-active-probe-wrapping-second-arithmetic-v1",
        "lock_id": lock["artifact_id"],
        "result_id": result["artifact_id"],
        "verified_native_trajectories": 297,
        "verified_source_files": len(lock["source_sha256"]),
        "maximum_reward_formula_difference": reward_difference,
        "maximum_decision_arithmetic_difference": maximum,
        "probe_selection_recomputed_without_future_reward": True,
        "native_physics_reexecuted": False,
        "independent_human_review": False,
        "source_gate_passed": decision["passed"],
        "passed": reward_difference <= 1e-7 and maximum <= 1e-10,
        "protected_data_read": False,
    }


if __name__ == "__main__":
    output = OUTPUT / "second-arithmetic.json"
    if output.exists() or output.is_symlink():
        raise ValueError("write-once second arithmetic already exists")
    row = verify()
    written = write_record(output, row)
    print(json.dumps(written, indent=2))
