#!/usr/bin/env python3
"""Verify the complete public DLO-Lab Slingshot guard replication."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import subprocess
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from bayesian_phystwin_experiments.dlolab_native import file_digest
from bayesian_phystwin_experiments.dlolab_regret_artifacts import read_record
from bayesian_phystwin_experiments.dlolab_slingshot_process import load_native_bundle

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("/home/fpfaff/source-only/dlolab-slingshot-certified-guard-source-v2")
FROZEN_REVISION = "7da610e3c321f605be29682d1360357496693c7e"
LOCK_ID = "7008acbe9ab7fd805832df4e97794f5c6924d00153bb25b6a5b6a2aa9abd54ef"
DECISION_ID = "46504a73df7f77e46b0d252657b6948ca23a301884a48a6e0e108e4f4243f490"
BARRIER_ID = "d2c1893755f86e6cca4210ef362ad1cb46b0ad5defc90a6c91c6a083df895432"
GENERATION_ID = "0514d18e98bac0904d258916da5c24d929926497dfb1e92183ffcb4860922fa1"
RESULT_ID = "35388657b9d3e162a5dcadeb003f6943123b3f19a9d8ac04b2eccd1cdec32ba1"
CONTINUATION_ID = "70b14a0ba4b2b3b9449be954675e94b62617864483be066f542816951b97d5d2"
CONTINUATION_COMPLETE_ID = (
    "bf549331975d60a982f4efda17749fc9c1ff9ebe0ace9f261c670b7b40e48086"
)
EXPECTED_TREE_FILES = 1_305
EXPECTED_TREE_BYTES = 677_043_088
EXPECTED_TREE_SHA256 = (
    "9c66f1a3f241465966d1ba37e0de8fe91622a9d7f87d910436d18c021803424f"
)
WORLD_COUNT = 288
PREFIX_BATCH_COUNT = 36
SENSOR_DRAWS = 4096
SENSOR_SEED = 261921
BOOTSTRAP_SEED = 261922
BOOTSTRAP_REPLICATES = 20_000
BASELINE = 5
ORDER = (5, 0, 1, 2, 3, 4, 6)
ARM_NAMES = ("incumbent", "posterior_predictive_mean", "mean_regret_guard")
OBSERVATION_FRAMES = (139, 219, 299)
OBSERVATION_NODES = (3, 6, 8)
BIAS_STD_M = 0.005
NOISE_STD_M = 0.002
MEAN_CALIBRATION_OFFSET = 0.7285524030751176
REWARD_MARGIN = 0.002
POSITION_ENVELOPE_M = 0.0005
POSITION_FIELDS = ("rod_pos_m", "sphere_pos_m", "cube_pos_m", "gripper_pos_m")
Array: TypeAlias = NDArray[Any]


def _tree_identity(output: Path) -> tuple[int, int, str]:
    files = sorted(path for path in output.rglob("*") if path.is_file())
    if any(path.is_symlink() for path in files):
        raise ValueError("symlinked evidence file is not admitted")
    digest = hashlib.sha256()
    byte_count = 0
    for path in files:
        relative = path.relative_to(output).as_posix()
        identity = file_digest(path)
        digest.update(relative.encode() + b"\0" + identity.encode() + b"\n")
        byte_count += path.stat().st_size
    return len(files), byte_count, digest.hexdigest()


def _expected_paths() -> set[str]:
    paths = {
        "lock.json",
        "continuation-receipt.json",
        "continuation.log",
        "decision-barrier.json",
        "decisions/arrays.npz",
        "decisions/seal.json",
        "generation.json",
        "result.json",
        "continuation-complete.json",
    }
    for batch in range(PREFIX_BATCH_COUNT):
        name = f"prefix-{batch:02d}"
        paths.update(
            {
                f"{name}.log",
                f"{name}/arrays.npz",
                f"{name}/claim.json",
                f"{name}/seal.json",
            }
        )
    for index in range(WORLD_COUNT):
        name = f"future-{index:03d}"
        paths.update(
            {
                f"{name}.log",
                f"{name}/arrays.npz",
                f"{name}/claim.json",
                f"{name}/seal.json",
            }
        )
    return paths


def _frozen_sources_match(lock: dict[str, Any]) -> bool:
    if lock.get("source_revision") != FROZEN_REVISION:
        return False
    for name, expected in lock["source_sha256"].items():
        blob = subprocess.check_output(
            ["git", "show", f"{FROZEN_REVISION}:{name}"], cwd=ROOT
        )
        if hashlib.sha256(blob).hexdigest() != expected:
            return False
    return True


def _prefix_task(batch: int) -> dict[str, Any]:
    indices = list(range(8 * batch, 8 * batch + 8))
    return {
        "kind": "prefix_only",
        "name": f"prefix-{batch:02d}",
        "batch": batch,
        "world_indices": indices,
        "native_world_indices": indices,
    }


def _future_task(index: int) -> dict[str, Any]:
    return {
        "kind": "all_action_future",
        "name": f"future-{index:03d}",
        "world_index": index,
    }


def _prefix_observation(arrays: dict[str, Array]) -> Array:
    rod = arrays["rod_pos_m"]
    sphere = arrays["sphere_pos_m"]
    selected = rod[list(OBSERVATION_FRAMES)][:, :, list(OBSERVATION_NODES)]
    return np.concatenate(
        (selected, sphere[list(OBSERVATION_FRAMES), :, None]), axis=2
    ).transpose(1, 0, 2, 3)


def _prior_weights() -> Array:
    return np.asarray(
        [a * b * c for a, b, c in itertools.product((0.25, 0.5, 0.25), repeat=3)]
    )


def _decisions_for_observations(
    observations: Array, bank_prefix: Array, bank_reward: Array
) -> Array:
    residual = (observations[:, None] - bank_prefix[None]).reshape(
        len(observations), 27, 12, 3
    )
    mean = residual.mean(axis=2)
    centered = residual - mean[:, :, None]
    distance = np.sum(centered**2, axis=(2, 3)) / NOISE_STD_M**2
    distance += 12 * np.sum(mean**2, axis=2) / (
        NOISE_STD_M**2 + 12 * BIAS_STD_M**2
    )
    with np.errstate(divide="ignore"):
        log_weight = np.log(_prior_weights())[None] - 0.5 * distance
    log_weight -= np.max(log_weight, axis=1, keepdims=True)
    weight = np.exp(log_weight)
    weight /= weight.sum(axis=1, keepdims=True)
    expected = weight @ (-bank_reward[:, ORDER])
    posterior = np.argmin(expected, axis=1)
    allowed = expected - expected[:, :1] + MEAN_CALIBRATION_OFFSET < 0
    allowed[:, 0] = True
    guarded = np.argmin(np.where(allowed, expected, np.inf), axis=1)
    order = np.asarray(ORDER, dtype=np.int64)
    result: Array = np.empty((len(observations), 3), dtype=np.int64)
    result[:, 0] = BASELINE
    result[:, 1] = order[posterior]
    result[:, 2] = order[guarded]
    return result


def _independent_decisions(truth: Array, bank_prefix: Array, bank_reward: Array) -> Array:
    result: Array = np.empty((WORLD_COUNT, SENSOR_DRAWS, 3), dtype=np.int64)
    rng = np.random.default_rng(SENSOR_SEED)
    for world in range(WORLD_COUNT):
        bias = rng.normal(0, BIAS_STD_M, (SENSOR_DRAWS, 1, 1, 3))
        noise = rng.normal(0, NOISE_STD_M, (SENSOR_DRAWS, 3, 4, 3))
        result[world] = _decisions_for_observations(
            truth[world, None] + bias + noise, bank_prefix, bank_reward
        )
    return result


def _native_reward(cube: Array) -> float:
    frames = list(range(119, 700, 20))
    frames[-1] = 899
    total = np.float32(0)
    for frame in frames:
        total += np.float32(np.clip(cube[frame, 1], 0, 5))
    return float(total)


def _future_qa(
    arrays: dict[str, Array], seal: dict[str, Any], prefix: dict[str, Array]
) -> Array:
    common = max(
        float(np.max(np.abs(arrays[name][:300] - arrays[name][:300, :1])))
        for name in POSITION_FIELDS
    )
    duplicate = max(
        float(np.max(np.abs(arrays[name][:, 5] - arrays[name][:, 7])))
        for name in POSITION_FIELDS
    )
    replay = max(
        float(np.max(np.abs(arrays[name][:300, 0] - prefix[name])))
        for name in POSITION_FIELDS
    )
    fixed = float(
        np.max(
            np.abs(
                arrays["rod_pos_m"][:, :, [0, 1, 10, 11]]
                - arrays["rod_pos_m"][:1, :, [0, 1, 10, 11]]
            )
        )
    )
    rewards = np.asarray(
        [_native_reward(arrays["cube_pos_m"][:, index]) for index in range(8)]
    )
    checks = {
        "common_prefix": common <= POSITION_ENVELOPE_M,
        "duplicate_positions": duplicate <= POSITION_ENVELOPE_M,
        "duplicate_rewards": abs(rewards[5] - rewards[7]) <= 0.001,
        "fixed_endpoints": fixed <= 1e-9,
        "sealed_prefix_replay": replay <= POSITION_ENVELOPE_M,
    }
    reported = seal["qa"]
    if (
        not all(checks.values())
        or reported["checks"] != checks
        or reported["qa_passed"] is not True
        or not np.array_equal(
            rewards,
            np.asarray([row["native_reward"] for row in reported["metrics"]]),
        )
        or not np.array_equal(
            rewards, np.asarray(seal["native"]["native_cumulative_reward"])
        )
    ):
        raise ValueError("future native QA or reward replay changed")
    return rewards[:7]


def _binomial_cdf(successes: int, trials: int, probability: float) -> float:
    logs = np.asarray(
        [
            math.lgamma(trials + 1.0)
            - math.lgamma(index + 1.0)
            - math.lgamma(trials - index + 1.0)
            + index * math.log(probability)
            + (trials - index) * math.log1p(-probability)
            for index in range(successes + 1)
        ]
    )
    maximum = float(np.max(logs))
    return float(math.exp(maximum) * np.sum(np.exp(logs - maximum)))


def _clopper_pearson_upper(harms: int) -> float:
    tail_probability = 1.0 - 0.95
    if harms == 0:
        return float(-math.expm1(math.log(tail_probability) / WORLD_COUNT))
    lower = harms / WORLD_COUNT
    upper = 1.0
    for _ in range(160):
        midpoint = 0.5 * (lower + upper)
        if _binomial_cdf(harms, WORLD_COUNT, midpoint) > tail_probability:
            lower = midpoint
        else:
            upper = midpoint
    return float(upper)


def _independent_arithmetic(decisions: Array, rewards: Array) -> dict[str, Any]:
    selected = np.take_along_axis(rewards[:, None, :], decisions, axis=2)
    world_reward = selected.mean(axis=1)
    incumbent = rewards[:, BASELINE]
    world_reward = np.where(
        np.all(decisions == BASELINE, axis=1), incumbent[:, None], world_reward
    )
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
                np.count_nonzero(decisions[:, :, index] != BASELINE)
            ),
            "updated_worlds": int(
                np.count_nonzero(
                    np.any(decisions[:, :, index] != BASELINE, axis=1)
                )
            ),
            "harmful_worlds_beyond_numeric_margin": harms,
            "harm_probability_upper95": _clopper_pearson_upper(harms),
            "mean_downside_below_incumbent": float(
                np.maximum(-gain[:, index], 0).mean()
            ),
        }
    guard = arms["mean_regret_guard"]
    posterior = arms["posterior_predictive_mean"]
    oracle_gain = float(rewards.max(axis=1).mean() - incumbent.mean())
    guard_gain = guard["mean_gain_over_incumbent"]
    posterior_gain = posterior["mean_gain_over_incumbent"]
    guard_downside = guard["mean_downside_below_incumbent"]
    posterior_downside = posterior["mean_downside_below_incumbent"]
    return {
        "arms": arms,
        "mean_guard_harm_reduction_vs_posterior": (
            posterior["harmful_worlds_beyond_numeric_margin"]
            - guard["harmful_worlds_beyond_numeric_margin"]
        ),
        "mean_guard_downside_reduction_fraction": (
            1.0 - guard_downside / posterior_downside
            if posterior_downside > 0
            else 0.0
        ),
        "mean_guard_fraction_of_posterior_gain": (
            guard_gain / posterior_gain if posterior_gain > 0 else 0.0
        ),
        "mean_guard_fraction_of_oracle_headroom": (
            guard_gain / oracle_gain if oracle_gain > 0 else 0.0
        ),
        "oracle_mean_native_reward": float(rewards.max(axis=1).mean()),
        "oracle_gain_over_incumbent": oracle_gain,
        "distinct_oracle_actions": int(len(np.unique(np.argmax(rewards, axis=1)))),
    }


def _assert_same(actual: object, expected: object, *, name: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"{name} mapping changed")
        for key, value in expected.items():
            _assert_same(actual[key], value, name=f"{name}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or not np.array_equal(actual, expected):
            raise ValueError(f"{name} sequence changed")
        return
    if actual != expected:
        raise ValueError(f"{name} changed")


def verify(output: Path) -> dict[str, Any]:
    expected_paths = _expected_paths()
    actual_paths = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    }
    if actual_paths != expected_paths or _tree_identity(output) != (
        EXPECTED_TREE_FILES,
        EXPECTED_TREE_BYTES,
        EXPECTED_TREE_SHA256,
    ):
        raise ValueError("complete frozen result tree identity changed")

    lock = read_record(output / "lock.json")
    decision = read_record(output / "decisions/seal.json")
    barrier = read_record(output / "decision-barrier.json")
    generation = read_record(output / "generation.json")
    result = read_record(output / "result.json")
    continuation = read_record(output / "continuation-receipt.json")
    complete = read_record(output / "continuation-complete.json")
    if (
        lock["artifact_id"] != LOCK_ID
        or decision["artifact_id"] != DECISION_ID
        or barrier["artifact_id"] != BARRIER_ID
        or generation["artifact_id"] != GENERATION_ID
        or result["artifact_id"] != RESULT_ID
        or continuation["artifact_id"] != CONTINUATION_ID
        or complete["artifact_id"] != CONTINUATION_COMPLETE_ID
        or not _frozen_sources_match(lock)
        or continuation["retried_task_indices"] != []
        or complete["retried_task_indices"] != []
        or continuation["retry_authorized"] is not False
        or complete["retry_authorized"] is not False
        or complete["result_id"] != RESULT_ID
    ):
        raise ValueError("root custody or frozen source identity changed")

    controls = np.asarray(lock["controls"], dtype=np.float64)
    prefix_ids: list[str] = []
    prefix_arrays: list[dict[str, Array]] = []
    truth: Array = np.empty((WORLD_COUNT, 3, 4, 3), dtype=np.float64)
    for batch in range(PREFIX_BATCH_COUNT):
        task = _prefix_task(batch)
        directory = output / task["name"]
        claim = read_record(directory / "claim.json")
        seal = read_record(directory / "seal.json")
        if (
            claim["task"] != task
            or seal["task"] != task
            or claim["lock_id"] != LOCK_ID
            or seal["lock_id"] != LOCK_ID
            or seal["claim_id"] != claim["artifact_id"]
            or claim["retry_authorized"] is not False
            or claim["replacement_authorized"] is not False
            or seal["qa"]["qa_passed"] is not True
        ):
            raise ValueError("prefix custody changed")
        arrays = load_native_bundle(directory, seal["bundle"])
        if not np.array_equal(
            arrays["controls"], np.repeat(controls[BASELINE : BASELINE + 1], 8, axis=0)
        ):
            raise ValueError("prefix action changed")
        observed = _prefix_observation(arrays)
        truth[task["world_indices"]] = observed
        prefix_arrays.extend(
            {
                name: arrays[name][:, slot]
                for name in POSITION_FIELDS
            }
            for slot in range(8)
        )
        prefix_ids.append(seal["artifact_id"])
    if prefix_ids != barrier["prefix_seal_ids"] or prefix_ids != decision[
        "prefix_seal_ids"
    ]:
        raise ValueError("prefix lineage changed")

    decision_arrays = load_native_bundle(output / "decisions", decision["bundle"])
    parent = Path(lock["parent"]["root"])
    parent_seal = read_record(parent / "model-bank/seal.json")
    parent_bank = load_native_bundle(parent / "model-bank", parent_seal["bundle"])
    regenerated = _independent_decisions(
        truth, parent_bank["prefix"], parent_bank["reward"]
    )
    if (
        not np.array_equal(decision_arrays["truth_prefix_m"], truth)
        or not np.array_equal(decision_arrays["decision"], regenerated)
        or barrier["pre_future_gate_passed"] is not True
    ):
        raise ValueError("sealed pre-future decisions do not reproduce")

    authorization = {
        "gate": "complete_passing_decision_barrier",
        "decision_seal_id": DECISION_ID,
        "barrier_id": BARRIER_ID,
    }
    rewards: Array = np.empty((WORLD_COUNT, 7), dtype=np.float64)
    future_ids: list[str] = []
    for index in range(WORLD_COUNT):
        task = _future_task(index)
        directory = output / task["name"]
        claim = read_record(directory / "claim.json")
        seal = read_record(directory / "seal.json")
        if (
            claim["task"] != task
            or seal["task"] != task
            or claim["authorization"] != authorization
            or claim["lock_id"] != LOCK_ID
            or seal["lock_id"] != LOCK_ID
            or seal["claim_id"] != claim["artifact_id"]
            or claim["retry_authorized"] is not False
            or claim["replacement_authorized"] is not False
        ):
            raise ValueError("future custody changed")
        arrays = load_native_bundle(directory, seal["bundle"])
        if not np.array_equal(arrays["controls"], controls):
            raise ValueError("future action bank changed")
        rewards[index] = _future_qa(arrays, seal, prefix_arrays[index])
        future_ids.append(seal["artifact_id"])
    if (
        future_ids != generation["future_seal_ids"]
        or future_ids != result["future_seal_ids"]
        or generation["ordinary_native_worlds"] != WORLD_COUNT
        or generation["technical_failures"] != 0
        or generation["replacements"] != 0
    ):
        raise ValueError("complete future denominator changed")

    arithmetic = _independent_arithmetic(regenerated, rewards)
    for name, expected in arithmetic.items():
        _assert_same(result[name], expected, name=name)
    failed_checks = {name for name, passed in result["checks"].items() if not passed}
    expected_failed = {
        "guard_gain_at_least_0_001",
        "positive_paired_ci95_vs_incumbent",
        "guard_harm_upper_at_most_0_05",
        "guard_retains_at_least_10pct_posterior_gain",
        "guard_captures_at_least_5pct_oracle_headroom",
    }
    if (
        failed_checks != expected_failed
        or result["source_gate_passed"] is not False
        or result["ordinary_evaluations"] != WORLD_COUNT
        or result["technical_failures"] != 0
        or result["replacements"] != 0
    ):
        raise ValueError("frozen source gate decision changed")
    return {
        "verification": "PASS",
        "tree_files": EXPECTED_TREE_FILES,
        "tree_bytes": EXPECTED_TREE_BYTES,
        "tree_sha256": EXPECTED_TREE_SHA256,
        "prefix_batches": PREFIX_BATCH_COUNT,
        "future_worlds": WORLD_COUNT,
        "technical_failures": 0,
        "replacements": 0,
        "retries": 0,
        "result_id": RESULT_ID,
        "source_gate_passed": False,
        "independent_human_review": False,
        "independent_implementation_replay": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(verify(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
