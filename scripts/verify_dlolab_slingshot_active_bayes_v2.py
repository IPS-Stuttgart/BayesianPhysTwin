#!/usr/bin/env python3
"""Verify the fresh-world Slingshot active-Bayes v2 source result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from bayesian_phystwin_experiments.dlolab_native import array_digest, file_digest
from bayesian_phystwin_experiments.dlolab_regret_artifacts import read_record
from bayesian_phystwin_experiments.dlolab_slingshot_active_bayes_v2 import (
    ARM_NAMES,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    SENSOR_DRAWS,
    WORLD_COUNT,
    continuous_worlds,
    future_task,
    infer_decisions,
    pre_future_checks,
    prefix_task,
    protocol,
    score,
)
from bayesian_phystwin_experiments.dlolab_slingshot_batch import TRACE_NAMES
from bayesian_phystwin_experiments.dlolab_slingshot_belief import (
    native_qa,
    prefix_observations,
)
from bayesian_phystwin_experiments.dlolab_slingshot_process import (
    load_native_bundle,
)
from bayesian_phystwin_experiments.dlolab_slingshot_task_probe_dev import (
    frontloaded_controls,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("/home/fpfaff/source-only/dlolab-slingshot-active-bayes-source-v2")
ATTEMPT = Path(
    "/home/fpfaff/source-only/dlolab-slingshot-active-bayes-source-v2.attempt.json"
)
ACTIVE = Path(
    "/home/fpfaff/source-only/dlolab-slingshot-active-id-particle-source-v1"
)
SUMMARY = ROOT / "results/sota/dlolab_slingshot_active_bayes_source_v2/summary.json"
FROZEN_REVISION = "56e20c1d5fb1209f16cb71d5e21de2e1a90425aa"
ACTIVE_BANK_ID = "17b96572a07a3d20818e19f3f31fec4afff98429aea8628f0872e70a3788c22a"
PREFLIGHT_RESULT_ID = "53ab35dce6629dd1b2ab2b28e6756e14cd6100bc50bba46e45d02f71023aac25"
V1_RESULT_ID = "ba3d015ad95806f040c88b0f8548ee238edbef0ebbcc0b1fe1b4858d91547e5b"
POSITION_FIELDS = ("rod_pos_m", "sphere_pos_m", "cube_pos_m", "gripper_pos_m")
Array: TypeAlias = NDArray[Any]


def _world_realization(native: dict[str, Any], worlds: list[dict[str, Any]]) -> bool:
    expected = {
        "bending": [[row["bending_E"] for row in worlds]],
        "stretching": [[row["stretching_K"] for row in worlds]],
        "sphere_initial_position_m": [
            [0.12 + row["x_offset_m"], 0.06, 0.2] for row in worlds
        ],
        "cube_initial_position_m": [
            [0.12 + row["x_offset_m"], 0.23, 0.22] for row in worlds
        ],
    }
    return bool(native.get("world_realization") == expected)


def _prefixes(
    output: Path, lock: dict[str, Any], controls: Array
) -> tuple[Array, list[str], list[bool], list[dict[str, Array]]]:
    truth: Array = np.empty((2, WORLD_COUNT, 3, 4, 3), dtype=np.float64)
    ids: list[str] = []
    qa: list[bool] = []
    passive: list[dict[str, Array]] = []
    worlds = continuous_worlds()
    for probe in range(2):
        candidate = controls if probe == 0 else frontloaded_controls(controls, 0.70)
        expected_controls = np.repeat(candidate[5:6], 8, axis=0)
        for batch in range(4):
            task = prefix_task(probe, batch)
            directory = output / task["name"]
            claim = read_record(directory / "claim.json")
            seal = read_record(directory / "seal.json")
            native = seal.get("native", {})
            task_worlds = [worlds[index] for index in task["world_indices"]]
            if (
                claim.get("schema") != "dlolab-slingshot-active-bayes-claim-v2"
                or claim.get("lock_id") != lock["artifact_id"]
                or claim.get("task") != task
                or claim.get("authorization")
                != {"gate": "prefix_only_before_outcomes"}
                or claim.get("retry_authorized") is not False
                or seal.get("schema") != "dlolab-slingshot-active-bayes-seal-v2"
                or seal.get("lock_id") != lock["artifact_id"]
                or seal.get("claim_id") != claim["artifact_id"]
                or seal.get("task") != task
                or native.get("native_steps") != 300
                or native.get("future_simulated") is not False
                or native.get("reward_scored") is not False
                or not _world_realization(native, task_worlds)
            ):
                raise ValueError("prefix custody changed")
            data = load_native_bundle(directory, seal["bundle"])
            fixed = float(
                np.max(
                    np.abs(
                        data["rod_pos_m"][:, :, [0, 1, 10, 11]]
                        - data["rod_pos_m"][:1, :, [0, 1, 10, 11]]
                    )
                )
            )
            passed = bool(
                set(data) == set(TRACE_NAMES + ("controls",))
                and all(data[name].shape[:2] == (300, 8) for name in TRACE_NAMES)
                and array_digest(data["controls"]) == array_digest(expected_controls)
                and fixed <= 1e-9
            )
            indices = task["world_indices"]
            truth[probe, indices] = prefix_observations(data)
            ids.append(seal["artifact_id"])
            qa.append(passed)
            if probe == 0:
                passive.append(data)
    return truth, ids, qa, passive


def _independent_arithmetic(decisions: Array, rewards: Array) -> dict[str, Any]:
    selected = np.empty(
        (WORLD_COUNT, SENSOR_DRAWS, len(ARM_NAMES)), dtype=np.float64
    )
    for world in range(WORLD_COUNT):
        for draw in range(SENSOR_DRAWS):
            for arm in range(len(ARM_NAMES)):
                selected[world, draw, arm] = rewards[
                    world, int(decisions[world, draw, arm])
                ]
    world_reward = selected.mean(axis=1)
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, WORLD_COUNT, size=(BOOTSTRAP_REPLICATES, WORLD_COUNT)
    )

    def interval(values: Array) -> list[float]:
        means = values[indices].mean(axis=1)
        values_ci = np.quantile(means, [0.025, 0.975])
        return [float(values_ci[0]), float(values_ci[1])]

    return {
        "world_reward": world_reward,
        "gain_ci": [
            interval(world_reward[:, index] - world_reward[:, 0])
            for index in range(len(ARM_NAMES))
        ],
        "paired_ci": [
            interval(world_reward[:, 4] - world_reward[:, index])
            for index in range(4)
        ],
    }


def verify(output: Path) -> dict[str, Any]:
    if output.resolve() != OUTPUT or output.is_symlink():
        raise ValueError("registered Slingshot active-Bayes v2 root required")
    if (output / "failure.json").exists():
        raise ValueError("v2 result and failure cannot coexist")
    attempt = read_record(ATTEMPT)
    lock = read_record(output / "lock.json")
    result = read_record(output / "result.json")
    if (
        attempt.get("schema") != "dlolab-slingshot-active-bayes-attempt-v2"
        or attempt.get("revision") != FROZEN_REVISION
        or attempt.get("protocol") != protocol()
        or attempt.get("output_root") != str(OUTPUT)
        or lock.get("schema") != "dlolab-slingshot-active-bayes-lock-v2"
        or lock.get("revision") != FROZEN_REVISION
        or lock.get("attempt_id") != attempt["artifact_id"]
        or lock.get("source_sha256") != attempt.get("source_sha256")
        or lock.get("protocol") != attempt.get("protocol")
        or lock.get("output_root") != str(OUTPUT)
        or lock.get("runtime_preflight_result_id") != PREFLIGHT_RESULT_ID
        or lock.get("terminal_v1_result_id") != V1_RESULT_ID
        or result.get("lock_id") != lock["artifact_id"]
        or result.get("source_gate_passed") is not False
        or result.get("task_future_generated") is not True
        or result.get("technical_failures") != 0
        or result.get("replacements") != 0
        or any(
            record.get("retry_authorized") is not False
            for record in (attempt, lock, result)
        )
        or any(
            record.get("protected_data_read") is not False
            for record in (attempt, lock, result)
        )
    ):
        raise ValueError("v2 root custody changed")
    if any(
        file_digest(ROOT / name) != digest
        for name, digest in lock["source_sha256"].items()
    ):
        raise ValueError("frozen v2 source changed")

    active_seal = read_record(ACTIVE / "particle-bank" / "seal.json")
    active = load_native_bundle(ACTIVE / "particle-bank", active_seal["bundle"])
    if (
        active_seal.get("artifact_id") != ACTIVE_BANK_ID
        or active["history"].shape != (2, 27, 3, 4, 3)
        or active["reward"].shape != (27, 7)
    ):
        raise ValueError("registered particle bank changed")
    controls = np.asarray(
        read_record(
            Path(
                "/home/fpfaff/source-only/"
                "dlolab-benchmark-source-v1/belief-control-source-v1/lock.json"
            )
        )["controls"],
        dtype=np.float64,
    )
    truth, prefix_ids, prefix_qa, passive = _prefixes(output, lock, controls)
    expected_decisions = infer_decisions(active["history"], active["reward"], truth)
    decision_seal = read_record(output / "decisions" / "seal.json")
    decision_data = load_native_bundle(output / "decisions", decision_seal["bundle"])
    if (
        decision_seal.get("artifact_id") != result.get("decision_seal_id")
        or decision_seal.get("lock_id") != lock["artifact_id"]
        or decision_seal.get("prefix_seal_ids") != prefix_ids
        or decision_seal.get("particle_bank_id") != ACTIVE_BANK_ID
        or decision_seal.get("future_read") is not False
        or decision_seal.get("future_generated") is not False
        or set(decision_data) != set(expected_decisions)
        or any(
            not np.array_equal(decision_data[name], expected_decisions[name])
            for name in expected_decisions
        )
    ):
        raise ValueError("sealed decisions do not reconstruct")
    pre_future = pre_future_checks(
        decision_data["decisions"], all_prefix_qa=all(prefix_qa)
    )
    barrier = read_record(output / "decision-barrier.json")
    if (
        barrier.get("artifact_id") != result.get("barrier_id")
        or barrier.get("lock_id") != lock["artifact_id"]
        or barrier.get("decision_seal_id") != decision_seal["artifact_id"]
        or barrier.get("pre_future") != pre_future
        or barrier.get("future_read") is not False
        or barrier.get("future_generated") is not False
        or pre_future.get("pre_future_gate_passed") is not True
    ):
        raise ValueError("decision barrier does not reconstruct")

    rewards: list[list[float]] = []
    future_ids: list[str] = []
    future_qa: list[dict[str, Any]] = []
    worlds = continuous_worlds()
    for index in range(WORLD_COUNT):
        task = future_task(index)
        directory = output / task["name"]
        claim = read_record(directory / "claim.json")
        seal = read_record(directory / "seal.json")
        if (
            claim.get("schema") != "dlolab-slingshot-active-bayes-claim-v2"
            or claim.get("lock_id") != lock["artifact_id"]
            or claim.get("task") != task
            or claim.get("authorization")
            != {"gate": "all_decisions_sealed", "barrier_id": barrier["artifact_id"]}
            or seal.get("schema") != "dlolab-slingshot-active-bayes-seal-v2"
            or seal.get("lock_id") != lock["artifact_id"]
            or seal.get("claim_id") != claim["artifact_id"]
            or seal.get("task") != task
            or seal.get("native", {}).get("native_steps") != 900
            or not _world_realization(seal.get("native", {}), [worlds[index]] * 8)
        ):
            raise ValueError("future custody changed")
        data = load_native_bundle(directory, seal["bundle"])
        batch = index // 8
        slot = index % 8
        prefix = {name: passive[batch][name][:, slot] for name in POSITION_FIELDS}
        qa = native_qa(data, seal["native"], controls, prefix)
        rewards.append([float(row["native_reward"]) for row in qa["metrics"][:7]])
        future_ids.append(seal["artifact_id"])
        future_qa.append(qa)
    reward = np.asarray(rewards, dtype=np.float64)
    generation = read_record(output / "generation" / "seal.json")
    generation_data = load_native_bundle(output / "generation", generation["bundle"])
    if (
        generation.get("artifact_id") != result.get("generation_id")
        or generation.get("lock_id") != lock["artifact_id"]
        or generation.get("barrier_id") != barrier["artifact_id"]
        or generation.get("future_seal_ids") != future_ids
        or generation.get("native_qa") != future_qa
        or generation.get("ordinary_worlds") != WORLD_COUNT
        or generation.get("technical_failures") != 0
        or generation.get("replacements") != 0
        or set(generation_data) != {"reward"}
        or not np.array_equal(generation_data["reward"], reward)
    ):
        raise ValueError("generation bundle does not reconstruct")

    expected_score = score(
        decision_data["decisions"],
        reward,
        all_native_qa=all(row["qa_passed"] for row in future_qa),
    )
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
            "task_future_generated",
            "retry_authorized",
        }
    }
    if result_without_links != expected_score or result.get("pre_future") != pre_future:
        raise ValueError("registered score does not reconstruct")

    arithmetic = _independent_arithmetic(decision_data["decisions"], reward)
    for index, name in enumerate(ARM_NAMES):
        arm = result["arms"][name]
        if not np.isclose(
            arm["mean_native_reward"],
            arithmetic["world_reward"][:, index].mean(),
            rtol=0,
            atol=1e-15,
        ) or not np.allclose(
            arm["gain_ci95"], arithmetic["gain_ci"][index], rtol=0, atol=1e-15
        ):
            raise ValueError("independent arm arithmetic changed")
    for index, name in enumerate(ARM_NAMES[:4]):
        if not np.allclose(
            result["paired_active_bayes_gain"][name]["ci95"],
            arithmetic["paired_ci"][index],
            rtol=0,
            atol=1e-15,
        ):
            raise ValueError("independent paired arithmetic changed")

    summary = read_record(SUMMARY)
    if (
        summary.get("schema") != "dlolab-slingshot-active-bayes-summary-v2"
        or summary.get("result_id") != result["artifact_id"]
        or summary.get("source_gate_passed") is not False
        or summary.get("active_bayes_gain_over_active_map")
        != result["paired_active_bayes_gain"]["active_map"]["mean_gain"]
        or summary.get("active_bayes_gain_over_blind")
        != result["paired_active_bayes_gain"]["blind_prior"]["mean_gain"]
        or summary.get("active_bayes_gain_over_passive_bayes")
        != result["paired_active_bayes_gain"]["passive_bayes"]["mean_gain"]
    ):
        raise ValueError("compact result changed")
    return {
        "schema": "dlolab-slingshot-active-bayes-verification-v2",
        "attempt_id": attempt["artifact_id"],
        "lock_id": lock["artifact_id"],
        "decision_seal_id": decision_seal["artifact_id"],
        "barrier_id": barrier["artifact_id"],
        "generation_id": generation["artifact_id"],
        "result_id": result["artifact_id"],
        "summary_id": summary["artifact_id"],
        "prefix_batches": 8,
        "ordinary_worlds": 32,
        "technical_failures": 0,
        "source_gate_passed": False,
        "independent_arithmetic_passed": True,
        "protected_data_read": False,
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(verify(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
