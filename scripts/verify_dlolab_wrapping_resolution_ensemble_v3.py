#!/usr/bin/env python3
"""Verify the model-resolution ensemble wrapping source result."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from bayesian_phystwin_experiments.dlolab_native import array_digest
from bayesian_phystwin_experiments.dlolab_regret_artifacts import read_record
from bayesian_phystwin_experiments.dlolab_slingshot_process import load_native_bundle
from bayesian_phystwin_experiments.dlolab_wrapping_resolution_ensemble_v3 import (
    ARM_NAMES,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    PREFIX_BATCH_COUNT,
    REWARD_MARGIN,
    SENSOR_DRAWS,
    WORLD_COUNT,
    continuous_worlds,
    future_native_qa,
    future_task,
    infer_decisions,
    pre_future_checks,
    prefix_native_qa,
    prefix_observation,
    prefix_task,
    protocol,
    score,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("/home/fpfaff/source-only/dlolab-wrapping-resolution-ensemble-source-v3")
ATTEMPT = Path(
    "/home/fpfaff/source-only/"
    "dlolab-wrapping-resolution-ensemble-source-v3.attempt.json"
)
PREFLIGHT = Path(
    "/home/fpfaff/source-only/dlolab-wrapping-resolution-ensemble-runtime-preflight-v3"
)
PARENT = Path("/home/fpfaff/source-only/dlolab-wrapping-belief-source-v1")
SUMMARY = (
    ROOT / "results/sota/dlolab_wrapping_resolution_ensemble_source_v3/summary.json"
)
FROZEN_REVISION = "78cc16b61e80f7745b3c7783739f965c3dcdf0e2"
ATTEMPT_ID = "9a8628dcd7ef253a46441607250a0c0664e9aa6d73e7d67bcd2ea9cddb1c509e"
PREFLIGHT_ID = "7537a19b82da8d478073466783d30e7d5a626ce5e1f149a39705c6aaef7fc43a"
LOCK_ID = "a783f8148dbdf1a570b2a060b526468c1924bd86d89b1574aa72845344a4026a"
DECISION_ID = "adb61ee81eb735881b844e8257979eeb88739e98635f024b77ce39195834b01c"
BARRIER_ID = "5d73917f1dc12bfec474433bf9b7c273481e9da8147cef93e705caa483565a42"
GENERATION_ID = "a03062d2ffab0daa2ff656743800bdea65ac17fce4c00170541b93878bd96b58"
RESULT_ID = "c187d1002f9c0244cea0356a5daac7cf987d3bb754ebb1021d9329f15ac47b19"
POSITION_FIELDS = ("rod_pos_m", "gripper_pos_m", "post_pos_m")
Array: TypeAlias = NDArray[Any]


def _frozen_source_matches(lock: dict[str, Any]) -> bool:
    for name, expected in lock["source_sha256"].items():
        blob = subprocess.check_output(
            ["git", "show", f"{FROZEN_REVISION}:{name}"], cwd=ROOT
        )
        if hashlib.sha256(blob).hexdigest() != expected:
            return False
    return True


def _prefixes(
    output: Path, lock: dict[str, Any]
) -> tuple[Array, list[str], list[bool], list[dict[str, Array]]]:
    truth: Array = np.empty((WORLD_COUNT, 3, 5, 3), dtype=np.float64)
    ids: list[str] = []
    passed: list[bool] = []
    arrays: list[dict[str, Array]] = []
    roster = continuous_worlds()
    for batch in range(PREFIX_BATCH_COUNT):
        task = prefix_task(batch)
        directory = output / task["name"]
        claim = read_record(directory / "claim.json")
        seal = read_record(directory / "seal.json")
        worlds = [roster[index] for index in task["native_world_indices"]]
        if (
            claim.get("schema") != "dlolab-wrapping-resolution-ensemble-claim-v3"
            or claim.get("lock_id") != lock["artifact_id"]
            or claim.get("task") != task
            or claim.get("authorization") != {"gate": "prefix_only_before_futures"}
            or claim.get("retry_authorized") is not False
            or seal.get("schema") != "dlolab-wrapping-resolution-ensemble-seal-v3"
            or seal.get("lock_id") != lock["artifact_id"]
            or seal.get("claim_id") != claim["artifact_id"]
            or seal.get("task") != task
        ):
            raise ValueError("prefix custody changed")
        data = load_native_bundle(directory, seal["bundle"])
        qa = prefix_native_qa(data, seal["native"], worlds)
        count = len(task["world_indices"])
        truth[task["world_indices"]] = prefix_observation(data["rod_pos_m"])[:count]
        ids.append(seal["artifact_id"])
        passed.append(bool(qa["qa_passed"]))
        arrays.append(data)
    return truth, ids, passed, arrays


def _independent_arithmetic(decisions: Array, rewards: Array) -> dict[str, Any]:
    decision = np.asarray(decisions, dtype=np.int64)
    reward = np.asarray(rewards, dtype=np.float64)
    selected = np.empty((WORLD_COUNT, SENSOR_DRAWS, len(ARM_NAMES)), dtype=np.float64)
    for world in range(WORLD_COUNT):
        for draw in range(SENSOR_DRAWS):
            for arm in range(len(ARM_NAMES)):
                selected[world, draw, arm] = reward[
                    world, int(decision[world, draw, arm])
                ]
    world_reward = selected.mean(axis=1)
    samples = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, WORLD_COUNT, size=(BOOTSTRAP_REPLICATES, WORLD_COUNT)
    )

    def interval(values: Array) -> list[float]:
        quantile: Array = np.asarray(
            np.quantile(values[samples].mean(axis=1), [0.025, 0.975])
        )
        return [float(quantile[0]), float(quantile[1])]

    fixed = world_reward[:, 0]
    primary_index = ARM_NAMES.index("equal_resolution_ensemble")
    oracle_action = np.argmax(reward, axis=1)
    arms: dict[str, Any] = {}
    for index, name in enumerate(ARM_NAMES):
        gain = world_reward[:, index] - fixed
        arms[name] = {
            "mean_native_reward": float(world_reward[:, index].mean()),
            "mean_gain_over_continuous_prior_best_fixed": float(gain.mean()),
            "gain_ci95": interval(gain),
            "action_probability": [
                float(np.mean(decision[:, :, index] == action))
                for action in range(reward.shape[1])
            ],
            "nonfixed_sensor_decisions": int(
                np.count_nonzero(decision[:, :, index] != decision[:, :, 0])
            ),
            "worlds_harmed_beyond_numeric_margin": int(
                np.count_nonzero(gain < -REWARD_MARGIN)
            ),
            "oracle_action_rate": float(
                np.mean(decision[:, :, index] == oracle_action[:, None])
            ),
        }
    paired: dict[str, dict[str, Any]] = {
        name: {
            "mean_gain": float(
                (world_reward[:, primary_index] - world_reward[:, index]).mean()
            ),
            "ci95": interval(world_reward[:, primary_index] - world_reward[:, index]),
        }
        for index, name in enumerate(ARM_NAMES)
        if index != primary_index
    }
    oracle = np.max(reward, axis=1)
    headroom = float(np.mean(oracle - fixed))
    fixed_gain = float((world_reward[:, primary_index] - world_reward[:, 0]).mean())
    finite_gain = arms["finite_particle_bayes"][
        "mean_gain_over_continuous_prior_best_fixed"
    ]
    retained = fixed_gain / finite_gain if finite_gain > 0 else 0.0
    oracle_fraction = fixed_gain / headroom if headroom > 0 else 0.0
    ensemble_harms = arms["equal_resolution_ensemble"][
        "worlds_harmed_beyond_numeric_margin"
    ]
    finite_harms = arms["finite_particle_bayes"]["worlds_harmed_beyond_numeric_margin"]
    continuous_harms = arms["continuous_bayes"]["worlds_harmed_beyond_numeric_margin"]
    distinct_oracle = int(len(np.unique(oracle_action)))
    checks = {
        "complete_48_world_denominator": True,
        "all_native_qa": True,
        "distinct_oracle_actions_at_least_2": distinct_oracle >= 2,
        "ensemble_gain_over_best_fixed_at_least_0_015": fixed_gain >= 0.015,
        "positive_paired_ci95_vs_fixed": paired["continuous_prior_best_fixed"]["ci95"][
            0
        ]
        > 0,
        "ensemble_mean_loss_vs_finite_at_most_0_001": paired["finite_particle_bayes"][
            "mean_gain"
        ]
        >= -0.001,
        "ensemble_ci95_lower_vs_finite_above_minus_0_002": paired[
            "finite_particle_bayes"
        ]["ci95"][0]
        > -0.002,
        "ensemble_retains_at_least_95pct_finite_gain": retained >= 0.95,
        "ensemble_mean_gain_over_continuous_nonnegative": paired["continuous_bayes"][
            "mean_gain"
        ]
        >= 0,
        "ensemble_harms_no_more_worlds_than_continuous": (
            ensemble_harms <= continuous_harms
        ),
        "ensemble_harms_fewer_worlds_than_finite": ensemble_harms < finite_harms,
        "captures_at_least_50pct_oracle_headroom": oracle_fraction >= 0.50,
    }
    return {
        "arms": arms,
        "paired": paired,
        "oracle_mean_native_reward": float(oracle.mean()),
        "oracle_headroom": headroom,
        "oracle_fraction": oracle_fraction,
        "finite_gain_fraction_retained": retained,
        "distinct_oracle_actions": distinct_oracle,
        "checks": checks,
        "source_gate_passed": bool(all(checks.values())),
    }


def verify(output: Path) -> dict[str, Any]:
    if output.resolve() != OUTPUT or output.is_symlink():
        raise ValueError("registered continuous interpolation root required")
    if (output / "failure.json").exists() or list(output.glob("*/failure.json")):
        raise ValueError("result and failure cannot coexist")
    attempt = read_record(ATTEMPT)
    lock = read_record(output / "lock.json")
    result = read_record(output / "result.json")
    preflight = read_record(PREFLIGHT / "result.json")
    if (
        attempt.get("schema") != "dlolab-wrapping-resolution-ensemble-attempt-v3"
        or attempt.get("artifact_id") != ATTEMPT_ID
        or attempt.get("revision") != FROZEN_REVISION
        or attempt.get("protocol") != protocol()
        or attempt.get("output_root") != str(OUTPUT)
        or lock.get("schema") != "dlolab-wrapping-resolution-ensemble-lock-v3"
        or lock.get("artifact_id") != LOCK_ID
        or lock.get("revision") != FROZEN_REVISION
        or lock.get("attempt_id") != attempt["artifact_id"]
        or lock.get("source_sha256") != attempt.get("source_sha256")
        or lock.get("protocol") != attempt.get("protocol")
        or lock.get("output_root") != str(OUTPUT)
        or lock.get("preflight_result_id") != PREFLIGHT_ID
        or preflight.get("artifact_id") != PREFLIGHT_ID
        or preflight.get("schema")
        != "dlolab-wrapping-resolution-ensemble-preflight-result-v3"
        or preflight.get("runtime_preflight_passed") is not True
        or preflight.get("study_attempt_consumed") is not False
        or result.get("artifact_id") != RESULT_ID
        or result.get("schema") != "dlolab-wrapping-resolution-ensemble-score-v3"
        or result.get("lock_id") != LOCK_ID
        or result.get("status") != "complete"
        or result.get("source_gate_passed") is not False
        or result.get("task_future_generated") is not True
        or result.get("technical_failures") != 0
        or result.get("replacements") != 0
        or not _frozen_source_matches(lock)
        or any(
            record.get("retry_authorized") is not False
            for record in (attempt, lock, result)
        )
        or any(
            record.get("protected_data_read") is not False
            for record in (attempt, lock, result)
        )
    ):
        raise ValueError("frozen result custody changed")

    parent_seal = read_record(PARENT / "source-bank" / "seal.json")
    bank = load_native_bundle(PARENT / "source-bank", parent_seal["bundle"])
    if (
        parent_seal.get("artifact_id") != lock.get("parent_source_bank_id")
        or array_digest(bank["prefix"]) != lock.get("source_prefix_sha256")
        or array_digest(bank["reward"]) != lock.get("source_reward_sha256")
    ):
        raise ValueError("source bank changed")

    truth, prefix_ids, prefix_qa, prefix_arrays = _prefixes(output, lock)
    expected = infer_decisions(bank["prefix"], bank["reward"], truth)
    decision_seal = read_record(output / "decisions" / "seal.json")
    decision = load_native_bundle(output / "decisions", decision_seal["bundle"])
    if (
        decision_seal.get("schema")
        != "dlolab-wrapping-resolution-ensemble-decision-seal-v3"
        or decision_seal.get("artifact_id") != DECISION_ID
        or decision_seal.get("artifact_id") != result.get("decision_seal_id")
        or decision_seal.get("lock_id") != LOCK_ID
        or decision_seal.get("prefix_seal_ids") != prefix_ids
        or decision_seal.get("parent_source_bank_id") != parent_seal["artifact_id"]
        or decision_seal.get("future_simulated") is not False
        or decision_seal.get("future_read") is not False
        or set(decision) != set(expected)
        or any(not np.array_equal(decision[name], expected[name]) for name in expected)
    ):
        raise ValueError("sealed decisions do not reconstruct")
    pre_future = pre_future_checks(decision["decisions"], all_prefix_qa=all(prefix_qa))
    barrier = read_record(output / "decision-barrier.json")
    if (
        barrier.get("schema")
        != "dlolab-wrapping-resolution-ensemble-decision-barrier-v3"
        or barrier.get("artifact_id") != BARRIER_ID
        or barrier.get("artifact_id") != result.get("barrier_id")
        or barrier.get("lock_id") != LOCK_ID
        or barrier.get("decision_seal_id") != DECISION_ID
        or barrier.get("pre_future") != pre_future
        or barrier.get("future_simulated") is not False
        or barrier.get("future_read") is not False
        or pre_future.get("pre_future_gate_passed") is not True
    ):
        raise ValueError("decision barrier does not reconstruct")

    rewards: list[list[float]] = []
    future_ids: list[str] = []
    future_qa: list[dict[str, Any]] = []
    prefix_match: list[float] = []
    roster = continuous_worlds()
    for index in range(WORLD_COUNT):
        task = future_task(index)
        directory = output / task["name"]
        claim = read_record(directory / "claim.json")
        seal = read_record(directory / "seal.json")
        if (
            claim.get("schema") != "dlolab-wrapping-resolution-ensemble-claim-v3"
            or claim.get("lock_id") != LOCK_ID
            or claim.get("task") != task
            or claim.get("authorization")
            != {"gate": "all_decisions_sealed", "barrier_id": BARRIER_ID}
            or claim.get("retry_authorized") is not False
            or seal.get("schema") != "dlolab-wrapping-resolution-ensemble-seal-v3"
            or seal.get("lock_id") != LOCK_ID
            or seal.get("claim_id") != claim["artifact_id"]
            or seal.get("task") != task
        ):
            raise ValueError("future custody changed")
        data = load_native_bundle(directory, seal["bundle"])
        qa = future_native_qa(data, seal["native"], roster[index])
        slot = index % 9
        prefix = prefix_arrays[index // 9]
        mismatch = max(
            float(np.abs(prefix[name][:, slot] - data[name][:600, 1]).max())
            for name in POSITION_FIELDS
        )
        if not qa["qa_passed"] or mismatch > 0.001:
            raise ValueError("future native QA changed")
        rewards.append([float(value) for value in qa["final_rewards"][:8]])
        future_ids.append(seal["artifact_id"])
        future_qa.append(qa)
        prefix_match.append(mismatch)

    reward = np.asarray(rewards, dtype=np.float64)
    generation = read_record(output / "generation" / "seal.json")
    generation_data = load_native_bundle(output / "generation", generation["bundle"])
    if (
        generation.get("schema") != "dlolab-wrapping-resolution-ensemble-generation-v3"
        or generation.get("artifact_id") != GENERATION_ID
        or generation.get("artifact_id") != result.get("generation_id")
        or generation.get("lock_id") != LOCK_ID
        or generation.get("barrier_id") != BARRIER_ID
        or generation.get("future_seal_ids") != future_ids
        or generation.get("native_qa") != future_qa
        or generation.get("prefix_match_error_m") != prefix_match
        or generation.get("ordinary_worlds") != WORLD_COUNT
        or generation.get("technical_failures") != 0
        or generation.get("replacements") != 0
        or set(generation_data) != {"reward"}
        or not np.array_equal(generation_data["reward"], reward)
    ):
        raise ValueError("generation bundle does not reconstruct")

    expected_score = score(decision["decisions"], reward, all_native_qa=all(prefix_qa))
    result_without_links = {
        key: value
        for key, value in result.items()
        if key
        not in {
            "artifact_id",
            "lock_id",
            "decision_seal_id",
            "barrier_id",
            "generation_id",
            "pre_future",
            "status",
            "task_future_generated",
            "retry_authorized",
        }
    }
    if result_without_links != expected_score or result.get("pre_future") != pre_future:
        raise ValueError("registered score does not reconstruct")

    arithmetic = _independent_arithmetic(decision["decisions"], reward)
    if (
        arithmetic["arms"] != result["arms"]
        or arithmetic["paired"] != result["paired_ensemble_gain"]
        or arithmetic["oracle_mean_native_reward"]
        != result["oracle_mean_native_reward"]
        or arithmetic["oracle_headroom"]
        != result["oracle_headroom_over_continuous_prior_best_fixed"]
        or arithmetic["oracle_fraction"] != result["oracle_headroom_fraction_captured"]
        or arithmetic["finite_gain_fraction_retained"]
        != result["finite_gain_fraction_retained"]
        or arithmetic["distinct_oracle_actions"] != result["distinct_oracle_actions"]
        or arithmetic["checks"] != result["checks"]
        or arithmetic["source_gate_passed"] != result["source_gate_passed"]
    ):
        raise ValueError("second arithmetic implementation disagrees")

    summary = read_record(SUMMARY)
    if (
        summary.get("schema") != "dlolab-wrapping-resolution-ensemble-summary-v3"
        or summary.get("result_id") != RESULT_ID
        or summary.get("source_gate_passed") is not False
        or summary.get("ensemble_gain_over_fixed")
        != result["paired_ensemble_gain"]["continuous_prior_best_fixed"]["mean_gain"]
        or summary.get("ensemble_gain_over_finite_bayes")
        != result["paired_ensemble_gain"]["finite_particle_bayes"]["mean_gain"]
        or summary.get("finite_gain_fraction_retained")
        != result["finite_gain_fraction_retained"]
    ):
        raise ValueError("compact summary changed")
    return {
        "schema": "dlolab-wrapping-resolution-ensemble-verification-v3",
        "attempt_id": attempt["artifact_id"],
        "lock_id": LOCK_ID,
        "decision_seal_id": DECISION_ID,
        "barrier_id": BARRIER_ID,
        "generation_id": GENERATION_ID,
        "result_id": RESULT_ID,
        "summary_id": summary["artifact_id"],
        "prefix_batches": PREFIX_BATCH_COUNT,
        "ordinary_worlds": WORLD_COUNT,
        "technical_failures": 0,
        "source_gate_passed": False,
        "second_arithmetic_implementation_passed": True,
        "independent_human_review": False,
        "protected_data_read": False,
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(verify(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
