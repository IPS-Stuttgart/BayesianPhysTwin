#!/usr/bin/env python3
"""Second arithmetic check, with no production experiment imports or native run.

This is an alternate implementation, not an independent human review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

POSITION_FIELDS = ("rod_pos_m", "gripper_pos_m", "post_pos_m", "hidden_post_pos_m")
TARGET_ARRAY_ID = "0240f00bcd164a03478a786e4fc488e4adac2d23c6c720289ce81cf714c1d641"


def canonical(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if (
        canonical({k: v for k, v in value.items() if k != "artifact_id"})
        != value["artifact_id"]
    ):
        raise ValueError(f"record digest differs: {path.name}")
    return value


def array_id(value: np.ndarray) -> str:
    header = {"dtype": value.dtype.str, "shape": list(value.shape), "order": "C"}
    raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw + b"\0" + value.tobytes(order="C")).hexdigest()


def bundle(root: Path, spec: dict) -> dict[str, np.ndarray]:
    if set(spec) != {"file", "file_sha256", "arrays"} or spec["file"] != "arrays.npz":
        raise ValueError("noncanonical native bundle")
    path = root / spec["file"]
    if path.is_symlink() or digest(path) != spec["file_sha256"]:
        raise ValueError("bundle file differs")
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)) or set(archive.files) != set(
            spec["arrays"]
        ):
            raise ValueError("bundle field set differs")
        data = {name: archive[name].copy() for name in archive.files}
    for key, value in data.items():
        if (
            value.dtype.kind not in "bifu"
            or not np.isfinite(value).all()
            or array_id(value) != spec["arrays"][key]
        ):
            raise ValueError("native array differs")
    return data


def reward(points: np.ndarray, goal: np.ndarray) -> np.ndarray:
    flat = points.reshape(-1, 30, 3)
    values = []
    for row in flat:
        distances = np.sqrt(((row[:, None] - goal[None]) ** 2).sum(axis=2))
        cost = distances.min(axis=0).sum() / 30 + distances.min(axis=1).sum() / 30
        cost += np.clip(row[:, 2] - 0.04, 0, None).sum() / 30
        center = np.sqrt(np.sum((row[15, :2] - [0.198, 0.198]) ** 2))
        cost += 5 * max(0.0, float(center) - 0.02)
        values.append(np.exp(-cost))
    return np.asarray(values).reshape(points.shape[:-2])


def qa(data: dict[str, np.ndarray], native: dict, p: dict) -> dict:
    q = p["qualification"]
    final = reward(data["rod_pos_m"][-1], data["target_pos_m"])
    cumulative = np.zeros(8, dtype=np.float32)
    for row in reward(data["rod_pos_m"][19::20], data["target_pos_m"]):
        cumulative += row.astype(np.float32)
    prefix = max(
        float(np.abs(data[k][:600] - data[k][:600, 1:2]).max()) for k in POSITION_FIELDS
    )
    duplicate = max(
        float(np.abs(data[k][:, 1] - data[k][:, 7]).max()) for k in POSITION_FIELDS
    )
    posts = np.asarray([[0.28, 0.14, 0.02], [0.1, 0.275, 0.02]])
    hidden = np.asarray(
        [[[x, y, z] for z in (-0.02, 0, 0.02)] for x, y in ((0.28, 0.14), (0.1, 0.275))]
    )
    fixed = max(
        float(np.abs(data["post_pos_m"] - posts).max()),
        float(np.abs(data["hidden_post_pos_m"] - hidden).max()),
    )
    lengths = np.sqrt(np.sum(np.diff(data["rod_pos_m"], axis=2) ** 2, axis=3))
    length_error = float(np.abs(lengths / 0.02 - 1).max())
    attachment = float(
        np.sqrt(
            np.sum((data["rod_pos_m"][:, :, 3] - data["gripper_pos_m"]) ** 2, axis=2)
        ).max()
    )
    final_error = float(np.abs(final - native["native_final_reward"]).max())
    checks = {
        "native_final_reward": final_error
        <= q["native_final_reward_reconstruction_atol"],
        "native_cumulative_reward": np.array_equal(
            cumulative, native["native_cumulative_reward"]
        ),
        "ordinary_native_success": bool(
            np.all(np.asarray(native["native_final_reward"]) > 0)
        ),
        "common_prefix": prefix <= q["all_action_prefix_error_m"],
        "duplicate_positions": duplicate <= q["duplicate_and_repeat_position_budget_m"],
        "duplicate_rewards": abs(final[1] - final[7])
        <= q["duplicate_and_repeat_reward_budget"],
        "fixed_posts": fixed <= q["fixed_post_error_m"],
        "segment_length": length_error <= q["maximum_segment_relative_error"],
        "above_floor": float(data["rod_pos_m"][..., 2].min())
        >= q["minimum_rod_height_m"],
        "attached_material_point": attachment <= q["maximum_attachment_distance_m"],
    }
    return {
        "passed": all(checks.values()),
        "checks": {k: bool(v) for k, v in checks.items()},
        "maximum_prefix_error_m": prefix,
        "maximum_duplicate_coordinate_error_m": duplicate,
        "fixed_post_error_m": fixed,
        "maximum_segment_relative_error": length_error,
        "maximum_attachment_distance_m": attachment,
        "final_reward_reconstruction_error": final_error,
        "final_rewards": final.tolist(),
    }


def compare(actual: Any, expected: Any, path="root") -> float:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"field set differs: {path}")
        return max(
            (compare(actual[k], expected[k], f"{path}.{k}") for k in actual),
            default=0.0,
        )
    if isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"list differs: {path}")
        return max(
            (compare(a, b, path) for a, b in zip(actual, expected, strict=True)),
            default=0.0,
        )
    if isinstance(expected, bool) or isinstance(expected, (str, int)):
        if actual != expected:
            raise ValueError(f"value differs: {path}")
        return 0.0
    error = abs(float(actual) - float(expected))
    if not np.isfinite(error) or error > 1e-10:
        raise ValueError(f"arithmetic differs: {path}, {error}")
    return error


def belief_value(prefix: np.ndarray, returns: np.ndarray, p: dict) -> dict:
    draws = p["noise_draws_per_world"]
    rng = np.random.default_rng(p["noise_seed"])
    noise = rng.normal(0, 0.005, (draws, 1, 3)) + rng.normal(0, 0.002, (draws, 15, 3))
    signals = prefix.reshape(9, 15, 3) - prefix[4].reshape(1, 15, 3)
    per_draw = np.zeros((draws, 3))
    per_world = np.zeros((9, 3))
    frequencies = np.zeros((3, 7))
    diagonal, shared = 0.002**2, 0.005**2
    coefficient = shared / (diagonal + 15 * shared)
    for world in range(9):
        observed = signals[world] + noise
        decisions = np.zeros((draws, 3), dtype=np.int64)
        for start in range(0, draws, 128):
            delta = observed[start : start + 128, None] - signals
            total = np.sum(delta * delta, axis=(2, 3))
            common = np.sum(delta.sum(axis=2) ** 2, axis=2)
            # Sherman-Morrison precision, not the production Cholesky whitening.
            for column, squared in (
                (0, (total - coefficient * common) / diagonal),
                (2, total / diagonal),
            ):
                log_weight = -0.5 * squared
                log_weight -= log_weight.max(axis=1, keepdims=True)
                weights = np.exp(log_weight)
                weights /= weights.sum(axis=1, keepdims=True)
                decisions[start : start + 128, column] = (weights @ returns).argmax(
                    axis=1
                )
                if column == 0:
                    decisions[start : start + 128, 1] = returns[
                        weights.argmax(axis=1)
                    ].argmax(axis=1)
        value = returns[world, decisions]
        per_world[world] = value.mean(axis=0)
        per_draw += value / 9
        for arm in range(3):
            frequencies[arm] += np.bincount(decisions[:, arm], minlength=7) / (
                draws * 9
            )
    mean = per_draw.mean(axis=0)
    fixed_action = int(returns.mean(axis=0).argmax())
    fixed = float(returns[:, fixed_action].mean())
    oracle = returns.max(axis=1)
    gates = p["source_gates"]
    margin = gates["numeric_pair_margin"]
    adjusted = float(mean[0] - fixed - margin)
    checks = {
        "best_fixed_beats_prefix_hold": fixed - float(returns[:, 0].mean())
        >= gates["minimum_best_fixed_gain_over_prefix_hold"],
        "adjusted_oracle_headroom": float(oracle.mean()) - fixed - margin
        >= gates["minimum_adjusted_oracle_gain"],
        "distinct_oracle_actions": len(set(returns.argmax(axis=1)))
        >= gates["minimum_distinct_oracle_actions"],
        "at_least_three_worlds_with_useful_oracle_headroom": int(
            np.sum(oracle - returns[:, fixed_action] > 0.01)
        )
        >= gates["minimum_worlds_with_oracle_gain_above_0_01"],
        "adjusted_bayes_gain_over_best_fixed": adjusted
        >= gates["minimum_adjusted_bayes_gain_over_best_fixed"],
        "adjusted_bayes_gain_fraction_of_fixed_deficit": adjusted
        >= gates["minimum_adjusted_fraction_of_best_fixed_reward_deficit"]
        * (1 - fixed),
        "adjusted_bayes_gain_over_map": float(mean[0] - mean[1] - margin)
        >= gates["minimum_adjusted_bayes_gain_over_map"],
        "adjusted_bayes_gain_over_ignored_bias": float(mean[0] - mean[2] - margin)
        >= gates["minimum_adjusted_bayes_gain_over_ignored_bias"],
    }
    return {
        "arms": {
            name: {
                "expected_native_final_reward": float(mean[i]),
                "gain_over_best_fixed": float(mean[i] - fixed),
                "monte_carlo_standard_error": float(
                    per_draw[:, i].std(ddof=1) / np.sqrt(draws)
                ),
                "source_world_expected_rewards": per_world[:, i].tolist(),
                "action_probability": frequencies[i].tolist(),
            }
            for i, name in enumerate(
                ("bias_aware_bayes", "bias_aware_map", "ignored_shared_bias")
            )
        },
        "best_fixed_action": fixed_action,
        "best_fixed_reward": fixed,
        "nominal_world_best_action": int(returns[4].argmax()),
        "nominal_world_action_expected_reward": float(
            returns[:, returns[4].argmax()].mean()
        ),
        "prefix_hold_reward": float(returns[:, 0].mean()),
        "oracle_reward": float(oracle.mean()),
        "oracle_actions": returns.argmax(axis=1).tolist(),
        "adjusted_bayes_gain": adjusted,
        "checks": {k: bool(v) for k, v in checks.items()},
        "source_gate_passed": all(checks.values()),
        "monte_carlo_only_integrates_assumed_sensor_noise": True,
        "source_worlds_are_prior_support_not_independent_evaluation": True,
        "method_promotion_authorized": False,
    }


def verify(root: Path, repo: Path) -> dict:
    lock, result = read(root / "lock.json"), read(root / "result.json")
    p = lock["protocol"]
    if (
        result["lock_id"] != lock["artifact_id"]
        or p["schema"] != "dlolab-wiring-belief-source-v1"
    ):
        raise ValueError("result/protocol binding differs")
    for path, expected in lock["source_sha256"].items():
        raw = subprocess.check_output(
            ["git", "show", f"{lock['revision']}:{path}"], cwd=repo
        )
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("committed source differs")
    if (
        lock["native_source"]["asset_sha256"][
            "dlo-lab/target_pos/wiring_post_finalpos.npy"
        ]
        != "4e852122d01b4d0e0e1aa59cd286f749031f22a962ef375a8ab1c3a4a9953072"
    ):
        raise ValueError("public task goal differs")
    rows = []
    max_error = 0.0
    for index in range(result["completed_batches"]):
        spec = p["tasks"][index]
        directory = root / spec["name"]
        claim, seal, receipt = (
            read(directory / name) for name in ("claim.json", "seal.json", "qa.json")
        )
        if (
            claim["task"] != spec
            or seal["task"] != spec
            or seal["claim_id"] != claim["artifact_id"]
            or seal["lock_id"] != lock["artifact_id"]
            or claim["lock_id"] != lock["artifact_id"]
            or receipt["lock_id"] != lock["artifact_id"]
            or receipt["seal_id"] != seal["artifact_id"]
            or result["completed_seal_ids"][index] != seal["artifact_id"]
        ):
            raise ValueError("task claim/seal/result differs")
        data = bundle(directory, seal["bundle"])
        if (
            array_id(data["controls"]) != p["controls_array_sha256"]
            or array_id(data["target_pos_m"]) != TARGET_ARRAY_ID
        ):
            raise ValueError("frozen control or target geometry differs")
        if (
            data["rod_pos_m"].shape != (1800, 8, 30, 3)
            or seal["native"]["native_steps"] != 1800
        ):
            raise ValueError("native rollout dimension differs")
        for kind, key in (("bending", "bending_E"), ("twisting", "twisting_G")):
            if seal["native"]["world_realization"][kind] != [spec["world"][key]] * 8:
                raise ValueError("native material differs")
        checked = qa(data, seal["native"], p)
        max_error = max(max_error, compare(checked, receipt["qa"]))
        rows.append((seal, data, checked))
    passed_count = sum(row[2]["passed"] for row in rows)
    if passed_count != result["admitted_batches"] or result[
        "ordinary_trajectories"
    ] != 8 * len(rows):
        raise ValueError("failure accounting differs")
    if len(rows) >= 3:
        span = max(
            float(np.ptp(np.stack([r[1][k] for r in rows[:3]]), axis=0).max())
            for k in POSITION_FIELDS
        )
        reward_span = float(
            np.ptp(
                np.asarray([r[0]["native"]["native_final_reward"] for r in rows[:3]]),
                axis=0,
            ).max()
        )
        repeated = {
            "maximum_coordinate_span_m": span,
            "maximum_same_action_reward_span": reward_span,
            "passed": span <= 0.001 and reward_span <= 0.001,
            "population_bound_claimed": False,
        }
        repeat_receipt = read(root / "repeat-qualification.json")
        if repeat_receipt["lock_id"] != lock["artifact_id"]:
            raise ValueError("repeat receipt binding differs")
        max_error = max(max_error, compare(repeated, repeat_receipt["repeat_qa"]))
    if result["status"] == "complete":
        if len(rows) != 11 or passed_count != 11 or not repeated["passed"]:
            raise ValueError("value analysis without full native qualification")
        ordered = sorted(
            (r for i, r in enumerate(rows) if i not in (1, 2)),
            key=lambda r: r[0]["task"]["world"]["index"],
        )
        bank_seal = read(root / "source-bank/seal.json")
        bank = bundle(root / "source-bank", bank_seal["bundle"])
        if bank_seal["artifact_id"] != result["source_bank_id"] or bank_seal[
            "source_seal_ids"
        ] != [r[0]["artifact_id"] for r in ordered]:
            raise ValueError("source bank lineage differs")
        expected_prefix = np.stack(
            [
                r[1]["rod_pos_m"][[199, 399, 599], 1][:, [6, 12, 18, 24, 29]]
                for r in ordered
            ]
        )
        expected_return = np.asarray(
            [r[0]["native"]["native_final_reward"][:7] for r in ordered]
        )
        if not np.array_equal(bank["prefix"], expected_prefix) or not np.array_equal(
            bank["reward"], expected_return
        ):
            raise ValueError("prefix/future information split differs")
        recomputed = belief_value(bank["prefix"], bank["reward"], p)
        max_error = max(max_error, compare(recomputed, result["metrics"]))
        if recomputed["source_gate_passed"] != result["source_gate_passed"]:
            raise ValueError("source gate differs")
    elif result["source_gate_passed"] is not False:
        raise ValueError("incomplete study cannot pass")
    answer = {
        "schema": "dlolab-wiring-second-arithmetic-v1",
        "lock_id": lock["artifact_id"],
        "result_id": result["artifact_id"],
        "result_sha256": digest(root / "result.json"),
        "verifier_sha256": digest(Path(__file__)),
        "native_trajectories_checked": 8 * len(rows),
        "native_micro_reward_rows_checked": 8 * len(rows) * 90,
        "source_value_analyzed": result["status"] == "complete",
        "maximum_arithmetic_difference": max_error,
        "source_gate_passed": result["source_gate_passed"],
        "passed": True,
        "independent_human_review": False,
        "new_empirical_execution": False,
        "protected_data_read": False,
    }
    answer["artifact_id"] = canonical(answer)
    return answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.root, args.repo)
    with args.output.open("x") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True))
