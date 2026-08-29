#!/usr/bin/env python3
"""Verify the complete native-Linux wrapping chance-guard replication."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from bayesian_phystwin_experiments.dlolab_native import array_digest, file_digest
from bayesian_phystwin_experiments.dlolab_regret_artifacts import read_record
from bayesian_phystwin_experiments.dlolab_slingshot_process import load_native_bundle
from bayesian_phystwin_experiments.dlolab_wrapping_risk_guard_v8 import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    PREFIX_BATCH_COUNT,
    REWARD_MARGIN,
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
from bayesian_phystwin_experiments.dlolab_wrapping_source import POSITION_FIELDS

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("/home/fpfaff/source-only/dlolab-wrapping-risk-guard-source-v8")
RECORDED_OUTPUT = Path(
    "/home/florianpfaff/source-only/dlolab-wrapping-risk-guard-source-v8"
)
RECORDED_ATTEMPT = Path(
    "/home/florianpfaff/source-only/dlolab-wrapping-risk-guard-source-v8.attempt.json"
)
LOCAL_ATTEMPT = Path(
    "/home/fpfaff/source-only/dlolab-wrapping-risk-guard-source-v8.attempt.json"
)
FROZEN_REVISION = "5c6a163d664114126bacba5865f8b1597cf8684b"
ATTEMPT_ID = "8f20c4c8ca03d56daed26519d31b34f7409b939ce02edcbd31c7476a9acf67a5"
LOCK_ID = "1a99a36263f2613ba3af61ccb382a678f0a0df52997c0af667a2c8e0ea7e04e2"
DECISION_ID = "25a7ce96aa82bb475df753b3cf4d89b85dc2e665e1eceae0ea47304fe4e86301"
BARRIER_ID = "3740dcc207310b7c3696df03c283fefa34daf21f61505036e293fc34d90b599e"
GENERATION_ID = "2dc04f43d60ec1f44660f9242ac32b1a6c6fa90c6c87c5cb2e0c81508af9fc8f"
RESULT_ID = "61c97165b9277790ac0e4e5374659fb9eab598dc0ff96ef3503859d6198ae5b2"
EXPECTED_TREE_FILES = 647
EXPECTED_TREE_BYTES = 4_936_658_734
POSITION_FIELDS_FOR_PARITY = tuple(POSITION_FIELDS)
Array: TypeAlias = NDArray[Any]


def _frozen_sources_match(lock: dict[str, Any]) -> bool:
    for name, expected in lock["source_sha256"].items():
        blob = subprocess.check_output(
            ["git", "show", f"{FROZEN_REVISION}:{name}"], cwd=ROOT
        )
        if hashlib.sha256(blob).hexdigest() != expected:
            return False
    return True


def _tree_manifest(output: Path) -> tuple[dict[str, str], int]:
    files = sorted(path for path in output.rglob("*") if path.is_file())
    if any(path.is_symlink() for path in files):
        raise ValueError("symlinked result file is not admitted")
    return (
        {str(path.relative_to(output)): file_digest(path) for path in files},
        sum(path.stat().st_size for path in files),
    )


def _expected_paths() -> set[str]:
    paths = {
        "lock.json",
        "decision-barrier.json",
        "result.json",
        "decisions/arrays.npz",
        "decisions/seal.json",
        "generation/arrays.npz",
        "generation/seal.json",
    }
    for batch in range(PREFIX_BATCH_COUNT):
        name = prefix_task(batch)["name"]
        paths.update(
            {
                f"{name}.log",
                f"{name}/arrays.npz",
                f"{name}/claim.json",
                f"{name}/seal.json",
            }
        )
    for index in range(WORLD_COUNT):
        name = future_task(index)["name"]
        paths.update(
            {
                f"{name}.log",
                f"{name}/arrays.npz",
                f"{name}/claim.json",
                f"{name}/seal.json",
            }
        )
    return paths


def _read_and_check_records(output: Path, attempt: Path) -> None:
    records = [attempt, *sorted(output.rglob("*.json"))]
    for path in records:
        record = read_record(path)
        for flag in ("protected_data_read", "retry_authorized", "replacement_authorized"):
            if flag in record and record[flag] is not False:
                raise ValueError(f"forbidden {flag} in {path}")


def _prefixes(
    output: Path, lock: dict[str, Any]
) -> tuple[Array, list[str], list[dict[str, Any]], list[dict[str, Array]]]:
    roster = continuous_worlds()
    truth: Array = np.empty((WORLD_COUNT, 3, 5, 3), dtype=np.float64)
    seal_ids: list[str] = []
    qas: list[dict[str, Any]] = []
    arrays: list[dict[str, Array]] = []
    for batch in range(PREFIX_BATCH_COUNT):
        task = prefix_task(batch)
        directory = output / task["name"]
        claim = read_record(directory / "claim.json")
        seal = read_record(directory / "seal.json")
        expected_worlds = [roster[index] for index in task["native_world_indices"]]
        if (
            claim.get("schema") != "dlolab-wrapping-risk-guard-claim-v8"
            or claim.get("lock_id") != lock["artifact_id"]
            or claim.get("task") != task
            or claim.get("authorization") != {"gate": "prefix_only_before_futures"}
            or seal.get("schema") != "dlolab-wrapping-risk-guard-seal-v8"
            or seal.get("lock_id") != lock["artifact_id"]
            or seal.get("claim_id") != claim["artifact_id"]
            or seal.get("task") != task
        ):
            raise ValueError("prefix custody changed")
        data = load_native_bundle(directory, seal["bundle"])
        qa = prefix_native_qa(data, seal["native"], expected_worlds)
        count = len(task["world_indices"])
        truth[task["world_indices"]] = prefix_observation(data["rod_pos_m"])[:count]
        seal_ids.append(seal["artifact_id"])
        qas.append(qa)
        arrays.append(data)
    return truth, seal_ids, qas, arrays


def _futures(
    output: Path,
    lock: dict[str, Any],
    barrier: dict[str, Any],
    prefixes: list[dict[str, Array]],
) -> tuple[Array, list[str], list[dict[str, Any]], list[float]]:
    roster = continuous_worlds()
    reward: Array = np.empty((WORLD_COUNT, 8), dtype=np.float64)
    seal_ids: list[str] = []
    qas: list[dict[str, Any]] = []
    parity: list[float] = []
    for index in range(WORLD_COUNT):
        task = future_task(index)
        directory = output / task["name"]
        claim = read_record(directory / "claim.json")
        seal = read_record(directory / "seal.json")
        if (
            claim.get("schema") != "dlolab-wrapping-risk-guard-claim-v8"
            or claim.get("lock_id") != lock["artifact_id"]
            or claim.get("task") != task
            or claim.get("authorization")
            != {"gate": "all_decisions_sealed", "barrier_id": barrier["artifact_id"]}
            or seal.get("schema") != "dlolab-wrapping-risk-guard-seal-v8"
            or seal.get("lock_id") != lock["artifact_id"]
            or seal.get("claim_id") != claim["artifact_id"]
            or seal.get("task") != task
        ):
            raise ValueError("future custody changed")
        data = load_native_bundle(directory, seal["bundle"])
        qa = future_native_qa(data, seal["native"], roster[index])
        slot = index % 9
        prefix = prefixes[index // 9]
        difference = max(
            float(np.abs(prefix[name][:, slot] - data[name][:600, 1]).max())
            for name in POSITION_FIELDS_FOR_PARITY
        )
        if not qa["qa_passed"] or difference > 0.001:
            raise ValueError("future native QA or reset parity changed")
        reward[index] = np.asarray(qa["final_rewards"][:8], dtype=np.float64)
        seal_ids.append(seal["artifact_id"])
        qas.append(qa)
        parity.append(difference)
    return reward, seal_ids, qas, parity


def _independent_arithmetic(decisions: Array, rewards: Array) -> dict[str, Any]:
    selected = np.take_along_axis(rewards[:, None, :], decisions, axis=2)
    world_reward = selected.mean(axis=1)
    fixed = world_reward[:, 0]
    guard = world_reward[:, 2]
    continuous = world_reward[:, 1]
    guard_gain = guard - fixed
    continuous_gain = continuous - fixed
    continuous_downside = np.maximum(fixed - continuous, 0)
    guard_downside = np.maximum(fixed - guard, 0)
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, WORLD_COUNT, size=(BOOTSTRAP_REPLICATES, WORLD_COUNT)
    )

    def ci(values: Array) -> list[float]:
        values = np.asarray(values, dtype=np.float64)
        quantiles: Array = np.asarray(
            np.quantile(values[indices].mean(axis=1), [0.025, 0.975])
        )
        return [float(quantiles[0]), float(quantiles[1])]

    return {
        "fixed_mean_reward": float(fixed.mean()),
        "guard_mean_reward": float(guard.mean()),
        "continuous_mean_reward": float(continuous.mean()),
        "guard_gain": float(guard_gain.mean()),
        "guard_gain_ci95": ci(guard_gain),
        "continuous_gain": float(continuous_gain.mean()),
        "guard_vs_continuous": float((guard - continuous).mean()),
        "guard_vs_continuous_ci95": ci(guard - continuous),
        "guard_harms": int(np.count_nonzero(guard_gain < -REWARD_MARGIN)),
        "continuous_harms": int(np.count_nonzero(continuous_gain < -REWARD_MARGIN)),
        "guard_downside": float(guard_downside.mean()),
        "continuous_downside": float(continuous_downside.mean()),
        "downside_reduction_ci95": ci(continuous_downside - guard_downside),
    }


def verify(output: Path) -> dict[str, Any]:
    resolved = output.resolve()
    if resolved not in {OUTPUT, RECORDED_OUTPUT} or output.is_symlink():
        raise ValueError("registered native-Linux chance-guard root required")
    if (output / "failure.json").exists():
        raise ValueError("result and terminal failure cannot coexist")
    manifest, tree_bytes = _tree_manifest(output)
    expected_paths = _expected_paths()
    if set(manifest) != expected_paths:
        raise ValueError("complete exact 647-file denominator required")
    if len(manifest) != EXPECTED_TREE_FILES or tree_bytes != EXPECTED_TREE_BYTES:
        raise ValueError("registered native result tree size changed")
    attempt_path = LOCAL_ATTEMPT if resolved == OUTPUT else RECORDED_ATTEMPT
    parent_path = output.parent / "dlolab-wrapping-belief-source-v1-compact"
    _read_and_check_records(output, attempt_path)

    attempt = read_record(attempt_path)
    lock = read_record(output / "lock.json")
    result = read_record(output / "result.json")
    if (
        attempt.get("schema") != "dlolab-wrapping-risk-guard-attempt-v8"
        or attempt.get("artifact_id") != ATTEMPT_ID
        or attempt.get("revision") != FROZEN_REVISION
        or attempt.get("protocol") != protocol()
        or attempt.get("output_root") != str(RECORDED_OUTPUT)
        or lock.get("schema") != "dlolab-wrapping-risk-guard-lock-v8"
        or lock.get("artifact_id") != LOCK_ID
        or lock.get("revision") != FROZEN_REVISION
        or lock.get("attempt_id") != attempt["artifact_id"]
        or lock.get("source_sha256") != attempt.get("source_sha256")
        or lock.get("protocol") != attempt.get("protocol")
        or lock.get("output_root") != str(RECORDED_OUTPUT)
        or result.get("artifact_id") != RESULT_ID
        or result.get("lock_id") != lock["artifact_id"]
        or result.get("status") != "complete"
        or result.get("task_future_generated") is not True
        or result.get("source_gate_passed") is not False
        or result.get("ordinary_worlds") != WORLD_COUNT
        or result.get("technical_failures") != 0
        or result.get("replacements") != 0
        or not _frozen_sources_match(lock)
    ):
        raise ValueError("frozen v8 root custody changed")

    parent_lock = read_record(parent_path / "lock.json")
    parent_seal = read_record(parent_path / "source-bank/seal.json")
    parent = load_native_bundle(
        parent_path / "source-bank", parent_seal["bundle"]
    )
    if (
        parent_lock.get("artifact_id") != lock.get("parent_lock_id")
        or parent_seal.get("artifact_id") != lock.get("parent_source_bank_id")
        or array_digest(parent["prefix"]) != lock.get("source_prefix_sha256")
        or array_digest(parent["reward"]) != lock.get("source_reward_sha256")
        or any(
            file_digest(parent_path / name) != digest
            for name, digest in lock["parent_file_sha256"].items()
        )
    ):
        raise ValueError("registered source bank changed")

    truth, prefix_ids, prefix_qas, prefix_arrays = _prefixes(output, lock)
    expected_decisions = infer_decisions(parent["prefix"], parent["reward"], truth)
    decision_seal = read_record(output / "decisions/seal.json")
    decisions = load_native_bundle(output / "decisions", decision_seal["bundle"])
    if (
        decision_seal.get("schema")
        != "dlolab-wrapping-risk-guard-decision-seal-v8"
        or decision_seal.get("artifact_id") != DECISION_ID
        or decision_seal.get("lock_id") != lock["artifact_id"]
        or decision_seal.get("prefix_seal_ids") != prefix_ids
        or decision_seal.get("parent_source_bank_id") != parent_seal["artifact_id"]
        or decision_seal.get("future_simulated") is not False
        or decision_seal.get("future_read") is not False
        or set(decisions) != set(expected_decisions)
        or any(
            not np.array_equal(decisions[name], expected_decisions[name])
            for name in expected_decisions
        )
    ):
        raise ValueError("sealed decisions do not reconstruct")

    pre_future = pre_future_checks(
        decisions["decisions"],
        decisions["guarded_posterior_improvement_probability"],
        all_prefix_qa=all(qa["qa_passed"] for qa in prefix_qas),
    )
    barrier = read_record(output / "decision-barrier.json")
    if (
        barrier.get("schema") != "dlolab-wrapping-risk-guard-decision-barrier-v8"
        or barrier.get("artifact_id") != BARRIER_ID
        or barrier.get("lock_id") != lock["artifact_id"]
        or barrier.get("decision_seal_id") != decision_seal["artifact_id"]
        or barrier.get("pre_future") != pre_future
        or barrier.get("future_simulated") is not False
        or barrier.get("future_read") is not False
        or pre_future.get("pre_future_gate_passed") is not True
    ):
        raise ValueError("decision barrier does not reconstruct")

    reward, future_ids, future_qas, prefix_parity = _futures(
        output, lock, barrier, prefix_arrays
    )
    generation = read_record(output / "generation/seal.json")
    generation_data = load_native_bundle(output / "generation", generation["bundle"])
    if (
        generation.get("schema") != "dlolab-wrapping-risk-guard-generation-v8"
        or generation.get("artifact_id") != GENERATION_ID
        or generation.get("lock_id") != lock["artifact_id"]
        or generation.get("barrier_id") != barrier["artifact_id"]
        or generation.get("future_seal_ids") != future_ids
        or generation.get("native_qa") != future_qas
        or generation.get("prefix_match_error_m") != prefix_parity
        or generation.get("ordinary_worlds") != WORLD_COUNT
        or generation.get("technical_failures") != 0
        or generation.get("replacements") != 0
        or set(generation_data) != {"reward"}
        or not np.array_equal(generation_data["reward"], reward)
    ):
        raise ValueError("source generation does not reconstruct")

    metrics = score(decisions["decisions"], reward, all_native_qa=True)
    expected_result = {
        **metrics,
        "status": "complete",
        "lock_id": lock["artifact_id"],
        "decision_seal_id": decision_seal["artifact_id"],
        "barrier_id": barrier["artifact_id"],
        "generation_id": generation["artifact_id"],
        "pre_future": pre_future,
        "task_future_generated": True,
        "retry_authorized": False,
        "replacement_authorized": False,
        "protected_data_read": False,
    }
    if {key: value for key, value in result.items() if key != "artifact_id"} != expected_result:
        raise ValueError("registered v8 score does not reconstruct")

    arithmetic = _independent_arithmetic(decisions["decisions"], reward)
    expected_arithmetic = {
        "fixed_mean_reward": result["arms"]["continuous_prior_best_fixed"][
            "mean_native_reward"
        ],
        "guard_mean_reward": result["arms"]["posterior_975_guard"][
            "mean_native_reward"
        ],
        "continuous_mean_reward": result["arms"]["continuous_bayes"][
            "mean_native_reward"
        ],
        "guard_gain": result["paired_guard_gain"]["continuous_prior_best_fixed"][
            "mean_gain"
        ],
        "guard_gain_ci95": result["paired_guard_gain"][
            "continuous_prior_best_fixed"
        ]["ci95"],
        "continuous_gain": result["arms"]["continuous_bayes"][
            "mean_gain_over_continuous_prior_best_fixed"
        ],
        "guard_vs_continuous": result["paired_guard_gain"]["continuous_bayes"][
            "mean_gain"
        ],
        "guard_vs_continuous_ci95": result["paired_guard_gain"]["continuous_bayes"][
            "ci95"
        ],
        "guard_harms": result["guard_harmed_worlds"],
        "continuous_harms": result["continuous_harmed_worlds"],
        "guard_downside": result["guard_mean_downside_below_fixed"],
        "continuous_downside": result["continuous_mean_downside_below_fixed"],
        "downside_reduction_ci95": result["guard_downside_reduction_ci95"],
    }
    for key, expected in expected_arithmetic.items():
        actual = arithmetic[key]
        if isinstance(expected, list):
            if not np.allclose(actual, expected, rtol=0, atol=1e-15):
                raise ValueError(f"independent arithmetic changed: {key}")
        elif isinstance(expected, float):
            if not np.isclose(actual, expected, rtol=0, atol=1e-15):
                raise ValueError(f"independent arithmetic changed: {key}")
        elif actual != expected:
            raise ValueError(f"independent arithmetic changed: {key}")

    tree_id = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema": "dlolab-wrapping-risk-guard-verification-v8",
        "attempt_id": attempt["artifact_id"],
        "lock_id": lock["artifact_id"],
        "decision_seal_id": decision_seal["artifact_id"],
        "barrier_id": barrier["artifact_id"],
        "generation_id": generation["artifact_id"],
        "result_id": result["artifact_id"],
        "tree_id": tree_id,
        "tree_files": len(manifest),
        "tree_bytes": tree_bytes,
        "prefix_batches": PREFIX_BATCH_COUNT,
        "ordinary_worlds": WORLD_COUNT,
        "technical_failures": 0,
        "replacements": 0,
        "source_gate_passed": False,
        "guard_gain": arithmetic["guard_gain"],
        "guard_gain_ci95": arithmetic["guard_gain_ci95"],
        "guard_harmed_worlds": arithmetic["guard_harms"],
        "continuous_harmed_worlds": arithmetic["continuous_harms"],
        "downside_reduction_fraction": result[
            "guard_downside_reduction_fraction_vs_continuous"
        ],
        "independent_arithmetic_passed": True,
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
