#!/usr/bin/env python3
"""Alternate arithmetic and custody checks, without a native simulation.

This is a second implementation, not independent human review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

POSITION_FIELDS = ("rod_pos_m", "gripper_pos_m", "post_pos_m")


def canonical(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    if path.is_symlink():
        raise ValueError("symlinked record")
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


def reward(points: np.ndarray, posts: np.ndarray) -> np.ndarray:
    if points.shape[-2:] != (50, 3) or posts.shape != points.shape[:-2] + (3, 3):
        raise ValueError("wrapping geometry differs")
    values = []
    for rod, objects in zip(
        points.reshape(-1, 50, 3), posts.reshape(-1, 3, 3), strict=True
    ):
        loss, distance = [], []
        for post in objects:
            relative = rod - post
            angles = np.arctan2(relative[:, 1], relative[:, 0])
            # Recover winding by unwrapping absolute angles, not cross/dot sums.
            unwrapped = np.unwrap(np.r_[angles, angles[0]])
            turns = (unwrapped[-1] - unwrapped[0]) / (2 * np.pi)
            loss.append((abs(turns) - 1) ** 2)
            distance.append(
                max(0, float(np.sqrt(np.sum(relative**2, axis=1)).min()) - 0.015)
            )
        values.append(1 - sum(loss) / 3 - sum(distance))
    return np.asarray(values).reshape(points.shape[:-2])


def qa(data: dict[str, np.ndarray], native: dict, p: dict) -> dict:
    q = p["qualification"]
    final = reward(data["rod_pos_m"][-1], data["post_pos_m"][-1])
    cumulative = np.zeros(9, dtype=np.float32)
    for row in reward(data["rod_pos_m"][19::20], data["post_pos_m"][19::20]):
        cumulative += row.astype(np.float32) + np.float32(1)
    prefix = max(
        float(np.abs(data[k][:600] - data[k][:600, 1:2]).max()) for k in POSITION_FIELDS
    )
    duplicate = max(
        float(np.abs(data[k][:, 1] - data[k][:, 8]).max()) for k in POSITION_FIELDS
    )
    fixed = float(
        np.abs(
            data["post_pos_m"]
            - [[0.45, -0.115, 0.02], [0.45, 0.115, 0.02], [0.25, 0, 0.02]]
        ).max()
    )
    initial = data["initial_rod_pos_m"]
    rest = np.sqrt(
        np.sum(
            (np.concatenate([initial[:, 1:], initial[:, :1]], axis=1) - initial) ** 2,
            axis=2,
        )
    )
    if np.any(rest <= 0):
        raise ValueError("collapsed initial segment")
    rod = data["rod_pos_m"]
    edges = np.concatenate([rod[:, :, 1:], rod[:, :, :1]], axis=2) - rod
    ratios = np.sqrt(np.sum(edges**2, axis=3)) / rest
    attachment = float(
        np.sqrt(
            np.sum((rod[:, :, [17, 33]] - data["gripper_pos_m"]) ** 2, axis=3)
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
            np.all(np.asarray(native["native_final_reward"]) > -98)
        ),
        "common_prefix": prefix <= q["all_action_prefix_error_m"],
        "duplicate_positions": duplicate <= q["duplicate_and_repeat_position_budget_m"],
        "duplicate_rewards": abs(final[1] - final[8])
        <= q["duplicate_and_repeat_reward_budget"],
        "fixed_posts": fixed <= q["fixed_post_error_m"],
        "finite_extensible_segments": bool(
            ratios.min() >= q["segment_length_ratio_range"][0]
            and ratios.max() <= q["segment_length_ratio_range"][1]
        ),
        "above_floor": float(rod[..., 2].min()) >= q["minimum_rod_height_m"],
        "attached_material_points": attachment <= q["maximum_attachment_distance_m"],
    }
    return {
        "passed": all(checks.values()),
        "checks": {k: bool(v) for k, v in checks.items()},
        "maximum_prefix_error_m": prefix,
        "maximum_duplicate_coordinate_error_m": duplicate,
        "fixed_post_error_m": fixed,
        "segment_length_ratio_range": [float(ratios.min()), float(ratios.max())],
        "maximum_attachment_distance_m": attachment,
        "final_reward_reconstruction_error": final_error,
        "final_rewards": final.tolist(),
    }


def compare(actual: Any, expected: Any, path: str = "root") -> float:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"field set differs: {path}")
        return max(
            (compare(actual[k], expected[k], f"{path}.{k}") for k in expected),
            default=0,
        )
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"list differs: {path}")
        return max(
            (compare(a, b, path) for a, b in zip(actual, expected, strict=True)),
            default=0,
        )
    if isinstance(expected, (bool, str, int)):
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"value differs: {path}")
        return 0
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
    frequencies = np.zeros((3, 8))
    diagonal, shared = 0.002**2, 0.005**2
    coefficient = shared / (diagonal + 15 * shared)
    for world in range(9):
        observed = signals[world] + noise
        decisions = np.zeros((draws, 3), dtype=np.int64)
        for start in range(0, draws, 128):
            delta = observed[start : start + 128, None] - signals
            total = np.sum(delta * delta, axis=(2, 3))
            common = np.sum(delta.sum(axis=2) ** 2, axis=2)
            # Sherman-Morrison precision instead of production Cholesky whitening.
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
            frequencies[arm] += np.bincount(decisions[:, arm], minlength=8) / (
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
        "three_worlds_with_useful_oracle_headroom": int(
            np.sum(oracle - returns[:, fixed_action] > 0.05)
        )
        >= gates["minimum_worlds_with_oracle_gain_above_0_05"],
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
        or p["schema"] != "dlolab-wrapping-belief-source-v1"
        or result["schema"] != "dlolab-wrapping-source-result-v1"
        or any(
            result[k] is not False
            for k in (
                "method_promotion_authorized",
                "retry_authorized",
                "protected_data_read",
                "new_recordings",
                "gpu_work",
            )
        )
    ):
        raise ValueError("result/protocol/boundary binding differs")
    for path, expected in lock["source_sha256"].items():
        raw = subprocess.check_output(
            ["git", "show", f"{lock['revision']}:{path}"], cwd=repo
        )
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("committed source differs")
    verifier_path = "scripts/verify_dlolab_wrapping_source.py"
    if lock["source_sha256"][verifier_path] != digest(Path(__file__)):
        raise ValueError("alternate verifier differs from the frozen code")
    if (
        lock["native_source"]["asset_archive_sha256"]
        != "acd483e232f1bb1fbf34078b154825fab3d2ee63b0aa4efc253c4411b368e421"
    ):
        raise ValueError("public source assets differ")
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
            or any(
                record["lock_id"] != lock["artifact_id"]
                for record in (claim, seal, receipt)
            )
            or receipt["seal_id"] != seal["artifact_id"]
            or result["completed_seal_ids"][index] != seal["artifact_id"]
            or seal["belief_value_analyzed"] is not False
            or seal["protected_data_read"] is not False
        ):
            raise ValueError("task claim/seal/result differs")
        data = bundle(directory, seal["bundle"])
        native = seal["native"]
        if (
            array_id(data["controls"]) != p["controls_array_sha256"]
            or data["rod_pos_m"].shape != (2200, 9, 50, 3)
            or native["native_steps"] != 2200
            or native["world"] != spec["world"]
            or native["device"] != "cpu"
            or native["native_source_modified"] is not False
            or native["runtime_camera_rendered"] is not False
            or native["twisting_stiffness_zero_preserved"] is not True
        ):
            raise ValueError("native runtime or array contract differs")
        for kind, key in (("bending", "bending_E"), ("stretching", "stretching_K")):
            if native["world_realization"][kind] != [spec["world"][key]] * 9:
                raise ValueError("native material differs")
        checked = qa(data, native, p)
        max_error = max(max_error, compare(checked, receipt["qa"]))
        rows.append((seal, data, checked))
    passed_count = sum(row[2]["passed"] for row in rows)
    if (
        passed_count != result["admitted_batches"]
        or result["ordinary_trajectories"] != 9 * passed_count
        or result["qualified_trajectories"] != 9 * passed_count
        or result["completed_native_trajectories"] != 9 * len(rows)
        or not len(rows) <= result["attempted_batches"] <= 11
        or result["unrun_batches"] != 11 - result["attempted_batches"]
    ):
        raise ValueError("failure accounting differs")
    repeated: dict[str, Any] = {"passed": False}
    if len(rows) >= 3 and all(r[2]["passed"] for r in rows[:3]):
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
    status = result["status"]
    if status == "complete":
        if len(rows) != 11 or passed_count != 11 or not repeated["passed"]:
            raise ValueError("value analysis without full native qualification")
        ordered = sorted(
            (r for i, r in enumerate(rows) if i not in (1, 2)),
            key=lambda r: r[0]["task"]["world"]["index"],
        )
        bank_seal = read(root / "source-bank/seal.json")
        bank = bundle(root / "source-bank", bank_seal["bundle"])
        if (
            bank_seal["artifact_id"] != result["source_bank_id"]
            or bank_seal["lock_id"] != lock["artifact_id"]
            or bank_seal["source_seal_ids"] != [r[0]["artifact_id"] for r in ordered]
        ):
            raise ValueError("source bank lineage differs")
        expected_prefix = np.stack(
            [
                r[1]["rod_pos_m"][[199, 399, 599], 1][:, [0, 8, 25, 41, 49]]
                for r in ordered
            ]
        )
        expected_return = np.asarray(
            [r[0]["native"]["native_final_reward"][:8] for r in ordered]
        )
        if not np.array_equal(bank["prefix"], expected_prefix) or not np.array_equal(
            bank["reward"], expected_return
        ):
            raise ValueError("prefix/future information split differs")
        recomputed = belief_value(bank["prefix"], bank["reward"], p)
        max_error = max(max_error, compare(recomputed, result["metrics"]))
        if recomputed["source_gate_passed"] != result["source_gate_passed"]:
            raise ValueError("source gate differs")
    elif status == "native_qualification_failed":
        if (
            not rows
            or rows[-1][2]["passed"]
            or passed_count != len(rows) - 1
            or result["failed_batch"] != len(rows) - 1
        ):
            raise ValueError("native qualification failure not retained correctly")
        if result["failed_checks"] != [
            k for k, v in rows[-1][2]["checks"].items() if not v
        ]:
            raise ValueError("failed checks differ")
    elif status == "native_repeatability_failed":
        if len(rows) != 3 or passed_count != 3 or repeated["passed"]:
            raise ValueError("repeat failure differs")
    elif status == "technical_failure":
        failure = read(root / "failure.json")
        if (
            failure["lock_id"] != lock["artifact_id"]
            or failure["stage"] != result["failed_stage"]
            or failure["error"] != result["error"]
            or failure["retry_authorized"] is not False
        ):
            raise ValueError("technical failure receipt differs")
    else:
        raise ValueError("unknown terminal status")
    if status != "complete" and result["source_gate_passed"] is not False:
        raise ValueError("incomplete study cannot pass")
    answer = {
        "schema": "dlolab-wrapping-second-arithmetic-v1",
        "lock_id": lock["artifact_id"],
        "result_id": result["artifact_id"],
        "result_sha256": digest(root / "result.json"),
        "verifier_sha256": digest(Path(__file__)),
        "native_trajectories_checked": 9 * len(rows),
        "native_micro_reward_rows_checked": 9 * len(rows) * 110,
        "source_files_checked": len(lock["source_sha256"]),
        "source_value_analyzed": status == "complete",
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
