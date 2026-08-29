#!/usr/bin/env python3
"""Verify the terminal wrapping chance-guard failure without scoring it."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin_experiments.dlolab_native import array_digest
from bayesian_phystwin_experiments.dlolab_regret_artifacts import read_record
from bayesian_phystwin_experiments.dlolab_slingshot_process import (
    load_native_bundle,
)
from bayesian_phystwin_experiments.dlolab_wrapping_risk_guard_v4 import (
    PREFIX_BATCH_COUNT,
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
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("/home/fpfaff/source-only/dlolab-wrapping-risk-guard-source-v4")
ATTEMPT = Path(
    "/home/fpfaff/source-only/dlolab-wrapping-risk-guard-source-v4.attempt.json"
)
PREFLIGHT = Path(
    "/home/fpfaff/source-only/dlolab-wrapping-risk-guard-runtime-preflight-v4"
)
PARENT = Path("/home/fpfaff/source-only/dlolab-wrapping-belief-source-v1")
SUMMARY = ROOT / "results/sota/dlolab_wrapping_risk_guard_source_v4/summary.json"
FROZEN_REVISION = "5d5150a794653c212d3a11f086d0ff845a427448"
ATTEMPT_ID = "17f4e1e6f140222b1d9c725c3921861aebae7dc9a59454c8938a3af94ddf43c8"
PREFLIGHT_ID = "095084fae2e256097acc0dafc6dde86ec2c065b36cfe376fcc2ef0e5dc1a60d2"
LOCK_ID = "ddd6724de1b0dc3303df6c5aba6906f4c41f8f896f27ed96f70ca24e89da5f18"
DECISION_ID = "198041b3f9f480802e8a529a94a740650b311c9a6e7375bbcf714c9d47238ea8"
BARRIER_ID = "519407e4590ffd61b901683c1a3d0f824e3ce608b15633b2012fecd5b6d07a07"
FAILURE_ID = "003be585e995ad8e38818cbb341fe9d39c8344d2dd8bc59d4bd6ace61945443f"
SUMMARY_ID = "ef75f43b46654530ed8a788303feee13c36a3d448566041b42707fe898e07873"
ORDINARY_FUTURES = 69
FAILED_FUTURE_INDEX = 69
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
            claim.get("schema") != "dlolab-wrapping-risk-guard-claim-v4"
            or claim.get("lock_id") != lock["artifact_id"]
            or claim.get("task") != task
            or claim.get("authorization") != {"gate": "prefix_only_before_futures"}
            or claim.get("retry_authorized") is not False
            or seal.get("schema") != "dlolab-wrapping-risk-guard-seal-v4"
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


def _completed_futures(
    output: Path,
    lock: dict[str, Any],
    prefix_arrays: list[dict[str, Array]],
) -> int:
    roster = continuous_worlds()
    ordinary = 0
    for index in range(ORDINARY_FUTURES):
        task = future_task(index)
        directory = output / task["name"]
        claim = read_record(directory / "claim.json")
        seal = read_record(directory / "seal.json")
        if (
            claim.get("schema") != "dlolab-wrapping-risk-guard-claim-v4"
            or claim.get("lock_id") != LOCK_ID
            or claim.get("task") != task
            or claim.get("authorization")
            != {"gate": "all_decisions_sealed", "barrier_id": BARRIER_ID}
            or claim.get("retry_authorized") is not False
            or seal.get("schema") != "dlolab-wrapping-risk-guard-seal-v4"
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
        ordinary += 1
    return ordinary


def _failed_future_is_unsealed(output: Path) -> bool:
    task = future_task(FAILED_FUTURE_INDEX)
    directory = output / task["name"]
    claim = read_record(directory / "claim.json")
    if (
        claim.get("schema") != "dlolab-wrapping-risk-guard-claim-v4"
        or claim.get("lock_id") != LOCK_ID
        or claim.get("task") != task
        or claim.get("authorization")
        != {"gate": "all_decisions_sealed", "barrier_id": BARRIER_ID}
        or claim.get("retry_authorized") is not False
        or (directory / "seal.json").exists()
        or (directory / "failure.json").exists()
        or (directory / "arrays.npz").exists()
    ):
        return False
    present = sorted(path.name for path in output.glob("future-*") if path.is_dir())
    expected = [f"future-{index:02d}" for index in range(FAILED_FUTURE_INDEX + 1)]
    return present == expected


def verify(output: Path) -> dict[str, Any]:
    if output.resolve() != OUTPUT or output.is_symlink():
        raise ValueError("registered wrapping chance-guard root required")
    if (output / "result.json").exists() or (output / "generation/seal.json").exists():
        raise ValueError("incomplete denominator cannot have a score")
    attempt = read_record(ATTEMPT)
    lock = read_record(output / "lock.json")
    failure = read_record(output / "failure.json")
    preflight = read_record(PREFLIGHT / "result.json")
    if (
        attempt.get("schema") != "dlolab-wrapping-risk-guard-attempt-v4"
        or attempt.get("artifact_id") != ATTEMPT_ID
        or attempt.get("revision") != FROZEN_REVISION
        or attempt.get("protocol") != protocol()
        or attempt.get("output_root") != str(OUTPUT)
        or lock.get("schema") != "dlolab-wrapping-risk-guard-lock-v4"
        or lock.get("artifact_id") != LOCK_ID
        or lock.get("revision") != FROZEN_REVISION
        or lock.get("attempt_id") != ATTEMPT_ID
        or lock.get("source_sha256") != attempt.get("source_sha256")
        or lock.get("protocol") != attempt.get("protocol")
        or lock.get("output_root") != str(OUTPUT)
        or lock.get("preflight_result_id") != PREFLIGHT_ID
        or preflight.get("artifact_id") != PREFLIGHT_ID
        or preflight.get("schema") != "dlolab-wrapping-risk-guard-preflight-result-v4"
        or preflight.get("runtime_preflight_passed") is not True
        or preflight.get("study_attempt_consumed") is not False
        or not _frozen_source_matches(lock)
        or any(
            record.get("retry_authorized") is not False
            for record in (attempt, lock, failure)
        )
        or any(
            record.get("protected_data_read") is not False
            for record in (attempt, lock, failure)
        )
    ):
        raise ValueError("frozen wrapping chance-guard custody changed")

    parent_seal = read_record(PARENT / "source-bank/seal.json")
    bank = load_native_bundle(PARENT / "source-bank", parent_seal["bundle"])
    if (
        parent_seal.get("artifact_id") != lock.get("parent_source_bank_id")
        or array_digest(bank["prefix"]) != lock.get("source_prefix_sha256")
        or array_digest(bank["reward"]) != lock.get("source_reward_sha256")
    ):
        raise ValueError("source bank changed")

    truth, prefix_ids, prefix_qa, prefix_arrays = _prefixes(output, lock)
    expected = infer_decisions(bank["prefix"], bank["reward"], truth)
    decision_seal = read_record(output / "decisions/seal.json")
    decision = load_native_bundle(output / "decisions", decision_seal["bundle"])
    if (
        decision_seal.get("schema") != "dlolab-wrapping-risk-guard-decision-seal-v4"
        or decision_seal.get("artifact_id") != DECISION_ID
        or decision_seal.get("lock_id") != LOCK_ID
        or decision_seal.get("prefix_seal_ids") != prefix_ids
        or decision_seal.get("parent_source_bank_id") != parent_seal["artifact_id"]
        or decision_seal.get("future_simulated") is not False
        or decision_seal.get("future_read") is not False
        or set(decision) != set(expected)
        or any(not np.array_equal(decision[name], expected[name]) for name in expected)
    ):
        raise ValueError("sealed decisions do not reconstruct")
    pre_future = pre_future_checks(
        decision["decisions"],
        decision["guarded_posterior_improvement_probability"],
        all_prefix_qa=all(prefix_qa),
    )
    barrier = read_record(output / "decision-barrier.json")
    if (
        barrier.get("schema") != "dlolab-wrapping-risk-guard-decision-barrier-v4"
        or barrier.get("artifact_id") != BARRIER_ID
        or barrier.get("lock_id") != LOCK_ID
        or barrier.get("decision_seal_id") != DECISION_ID
        or barrier.get("pre_future") != pre_future
        or barrier.get("future_simulated") is not False
        or barrier.get("future_read") is not False
        or pre_future.get("pre_future_gate_passed") is not True
    ):
        raise ValueError("decision barrier does not reconstruct")

    ordinary = _completed_futures(output, lock, prefix_arrays)
    if ordinary != ORDINARY_FUTURES or not _failed_future_is_unsealed(output):
        raise ValueError("partial future accounting changed")

    summary = read_record(SUMMARY)
    summary_payload = {
        key: value for key, value in summary.items() if key != "artifact_id"
    }
    if (
        failure.get("artifact_id") != FAILURE_ID
        or failure.get("schema") != "dlolab-wrapping-risk-guard-failure-v4"
        or failure.get("lock_id") != LOCK_ID
        or failure.get("terminal_stage") != "futures"
        or failure.get("completed_prefix_batches") != PREFIX_BATCH_COUNT
        or failure.get("completed_future_worlds") != ORDINARY_FUTURES
        or failure.get("error_type") != "RuntimeError"
        or failure.get("message") != "future-69 exited -11; no retry"
        or failure.get("retry_authorized") is not False
        or failure.get("replacement_authorized") is not False
        or summary.get("artifact_id") != SUMMARY_ID
        or content_id(summary_payload) != SUMMARY_ID
        or summary.get("ordinary_future_worlds") != ORDINARY_FUTURES
        or summary.get("registered_future_worlds") != WORLD_COUNT
        or summary.get("task_value_scored") is not False
        or summary.get("scientific_result_available") is not False
        or summary.get("source_gate_passed") is not False
        or summary.get("retry_authorized") is not False
        or summary.get("replacement_authorized") is not False
    ):
        raise ValueError("terminal failure accounting changed")
    return {
        "schema": "dlolab-wrapping-risk-guard-failure-verification-v4",
        "passed": True,
        "frozen_revision": FROZEN_REVISION,
        "attempt_id": ATTEMPT_ID,
        "lock_id": LOCK_ID,
        "decision_seal_id": DECISION_ID,
        "barrier_id": BARRIER_ID,
        "failure_id": FAILURE_ID,
        "summary_id": SUMMARY_ID,
        "prefix_batches": PREFIX_BATCH_COUNT,
        "ordinary_future_worlds": ordinary,
        "registered_future_worlds": WORLD_COUNT,
        "task_value_scored": False,
        "scientific_result_available": False,
        "retry_authorized": False,
        "independent_human_review": False,
        "second_implementation_only": True,
        "protected_data_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(verify(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
