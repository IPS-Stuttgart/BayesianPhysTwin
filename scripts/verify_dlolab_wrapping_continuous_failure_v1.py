#!/usr/bin/env python3
"""Verify the terminal continuous-material wrapping failure without scoring it."""

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
from bayesian_phystwin_experiments.dlolab_regret_artifacts import read_record
from bayesian_phystwin_experiments.dlolab_slingshot_process import load_native_bundle
from bayesian_phystwin_experiments.dlolab_wrapping_continuous_bayes_v1 import (
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
OUTPUT = Path("/home/fpfaff/source-only/dlolab-wrapping-continuous-bayes-source-v1")
ATTEMPT = Path(
    "/home/fpfaff/source-only/dlolab-wrapping-continuous-bayes-source-v1.attempt.json"
)
PREFLIGHT = Path(
    "/home/fpfaff/source-only/dlolab-wrapping-continuous-bayes-runtime-preflight-v1"
)
PARENT = Path("/home/fpfaff/source-only/dlolab-wrapping-belief-source-v1")
SUMMARY = ROOT / "results/sota/dlolab_wrapping_continuous_bayes_source_v1/summary.json"
FROZEN_REVISION = "7ddd623b89867348bb4b8635bea03bc6e32f8421"
PREFLIGHT_ID = "019c36ae8d28b0814ecdc3439a113f4780c25e03800772988fe79999c113818f"
LOCK_ID = "3afda0772f04f3ef7850b1ada4304ade752f462e4b08ce1100fef6ed50768534"
BARRIER_ID = "00c7002573419dd987aeb04d3f158eded35468676cb15e957c85eded522c8214"
FAILURE_ID = "32f1da52f18bcddc1697931b139b1222692f8eb7b9839b2997b60b9328837692"
SUMMARY_ID = "d37bc996f643030ca0a2b1a5a2c3528a39827825c2c09e68590ff3b3386dc4d9"
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
    qa_passed: list[bool] = []
    arrays: list[dict[str, Array]] = []
    roster = continuous_worlds()
    for batch in range(4):
        task = prefix_task(batch)
        directory = output / task["name"]
        claim = read_record(directory / "claim.json")
        seal = read_record(directory / "seal.json")
        worlds = [roster[index] for index in task["native_world_indices"]]
        if (
            claim.get("schema") != "dlolab-wrapping-continuous-claim-v1"
            or claim.get("lock_id") != lock["artifact_id"]
            or claim.get("task") != task
            or claim.get("authorization") != {"gate": "prefix_only_before_futures"}
            or claim.get("retry_authorized") is not False
            or seal.get("schema") != "dlolab-wrapping-continuous-seal-v1"
            or seal.get("lock_id") != lock["artifact_id"]
            or seal.get("claim_id") != claim["artifact_id"]
            or seal.get("task") != task
        ):
            raise ValueError("prefix custody changed")
        data = load_native_bundle(directory, seal["bundle"])
        qa = prefix_native_qa(data, seal["native"], worlds)
        count = len(task["world_indices"])
        truth[task["world_indices"]] = prefix_observation(data["rod_pos_m"])[
            :count
        ]
        ids.append(seal["artifact_id"])
        qa_passed.append(bool(qa["qa_passed"]))
        arrays.append(data)
    return truth, ids, qa_passed, arrays


def verify(output: Path) -> dict[str, Any]:
    if output.resolve() != OUTPUT or output.is_symlink():
        raise ValueError("registered continuous wrapping root required")
    attempt = read_record(ATTEMPT)
    lock = read_record(output / "lock.json")
    failure = read_record(output / "failure.json")
    preflight = read_record(PREFLIGHT / "result.json")
    if (
        attempt.get("schema") != "dlolab-wrapping-continuous-bayes-attempt-v1"
        or attempt.get("revision") != FROZEN_REVISION
        or attempt.get("protocol") != protocol()
        or attempt.get("output_root") != str(OUTPUT)
        or lock.get("schema") != "dlolab-wrapping-continuous-bayes-lock-v1"
        or lock.get("artifact_id") != LOCK_ID
        or lock.get("revision") != FROZEN_REVISION
        or lock.get("attempt_id") != attempt["artifact_id"]
        or lock.get("source_sha256") != attempt.get("source_sha256")
        or lock.get("protocol") != attempt.get("protocol")
        or lock.get("preflight_result_id") != PREFLIGHT_ID
        or preflight.get("artifact_id") != PREFLIGHT_ID
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
        raise ValueError("frozen continuous wrapping custody changed")

    parent_seal = read_record(PARENT / "source-bank" / "seal.json")
    bank = load_native_bundle(PARENT / "source-bank", parent_seal["bundle"])
    truth, prefix_ids, prefix_qa, prefix_arrays = _prefixes(output, lock)
    expected = infer_decisions(bank["prefix"], bank["reward"], truth)
    decision_seal = read_record(output / "decisions" / "seal.json")
    decision = load_native_bundle(output / "decisions", decision_seal["bundle"])
    if (
        decision_seal.get("lock_id") != LOCK_ID
        or decision_seal.get("prefix_seal_ids") != prefix_ids
        or decision_seal.get("parent_source_bank_id") != parent_seal["artifact_id"]
        or decision_seal.get("future_simulated") is not False
        or decision_seal.get("future_read") is not False
        or set(decision) != set(expected)
        or any(not np.array_equal(decision[name], expected[name]) for name in expected)
    ):
        raise ValueError("sealed decisions do not reconstruct")
    pre_future = pre_future_checks(
        decision["decisions"], all_prefix_qa=all(prefix_qa)
    )
    barrier = read_record(output / "decision-barrier.json")
    if (
        barrier.get("artifact_id") != BARRIER_ID
        or barrier.get("lock_id") != LOCK_ID
        or barrier.get("decision_seal_id") != decision_seal["artifact_id"]
        or barrier.get("pre_future") != pre_future
        or barrier.get("future_simulated") is not False
        or barrier.get("future_read") is not False
        or pre_future.get("pre_future_gate_passed") is not True
    ):
        raise ValueError("decision barrier does not reconstruct")

    ordinary = 0
    roster = continuous_worlds()
    for index in range(WORLD_COUNT):
        task = future_task(index)
        directory = output / task["name"]
        claim = read_record(directory / "claim.json")
        seal = read_record(directory / "seal.json")
        if (
            claim.get("lock_id") != LOCK_ID
            or claim.get("task") != task
            or claim.get("authorization")
            != {"gate": "all_decisions_sealed", "barrier_id": BARRIER_ID}
            or claim.get("retry_authorized") is not False
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

    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    summary_payload = {key: value for key, value in summary.items() if key != "artifact_id"}
    if (
        failure.get("artifact_id") != FAILURE_ID
        or failure.get("terminal_stage") != "futures"
        or failure.get("completed_prefix_batches") != 4
        or failure.get("completed_future_worlds") != WORLD_COUNT
        or failure.get("error_type") != "TypeError"
        or failure.get("message") != "Object of type bool is not JSON serializable"
        or failure.get("replacement_authorized") is not False
        or (output / "result.json").exists()
        or (output / "generation" / "seal.json").exists()
        or summary.get("artifact_id") != SUMMARY_ID
        or content_id(summary_payload) != SUMMARY_ID
        or summary.get("task_value_scored") is not False
        or summary.get("source_gate_passed") is not False
    ):
        raise ValueError("terminal failure accounting changed")
    return {
        "schema": "dlolab-wrapping-continuous-failure-verification-v1",
        "passed": True,
        "frozen_revision": FROZEN_REVISION,
        "prefix_batches": 4,
        "ordinary_future_worlds": ordinary,
        "decision_barrier_reconstructed": True,
        "task_value_scored": False,
        "scientific_result_available": False,
        "retry_authorized": False,
        "independent_human_review": False,
        "second_implementation_only": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(verify(args.output), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
