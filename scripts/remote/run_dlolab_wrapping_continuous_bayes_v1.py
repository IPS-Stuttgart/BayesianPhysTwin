#!/usr/bin/env python3
"""Run the frozen fresh continuous-material wrapping Bayes study."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from bayesian_phystwin_experiments.dlolab_benchmark import (
    source_identity,
    write_native_bundle,
)
from bayesian_phystwin_experiments.dlolab_native import array_digest, file_digest
from bayesian_phystwin_experiments.dlolab_regret_artifacts import (
    clean_revision,
    read_record,
    runtime_identity,
    write_record,
)
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
    preflight_world,
    protocol,
    score,
)
from bayesian_phystwin_experiments.dlolab_wrapping_continuous_native import (
    run_worlds,
)

ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path("/home/fpfaff/source-only/dlolab-benchmark-source-v1-assets")
PARENT = Path("/home/fpfaff/source-only/dlolab-wrapping-belief-source-v1")
OUTPUT = Path(
    "/home/fpfaff/source-only/dlolab-wrapping-continuous-bayes-source-v1"
)
ATTEMPT = Path(
    "/home/fpfaff/source-only/dlolab-wrapping-continuous-bayes-source-v1.attempt.json"
)
PREFLIGHT = Path(
    "/home/fpfaff/source-only/dlolab-wrapping-continuous-bayes-runtime-preflight-v1"
)
PREFLIGHT_ATTEMPT = Path(
    "/home/fpfaff/source-only/"
    "dlolab-wrapping-continuous-bayes-runtime-preflight-v1.attempt.json"
)
EXPECTED_PYTHON = Path(
    "/home/fpfaff/source-only/dlolab-benchmark-source-v1-assets/venv/bin/python"
)
PARENT_FILE_SHA256 = {
    "lock.json": "b689b17db607d79bb9b7642a5ad76a25591f7e85902ccd08ac01d7e6dc970bbc",
    "source-bank/arrays.npz": "914bd948df92e8b829ac65ca8c075c789d122a63a9ec32807a302bef16e2271d",
    "source-bank/seal.json": "143686ee40ddfb8456e23cded5c8225015e60bd678a4f77bd4074946c33fe14f",
    "result.json": "550b04bceab58d14f78f020a3870841ffae06e6ce8946d996f8418e868bacf9c",
}
PARENT_LOCK_ID = "70e6054141a5652957590f5b173c36ccff99cc167b48a3f8b4f085ba4be20a31"
PARENT_RESULT_ID = "5be8f1a54ac38e9dfc0745a5722a9490d8fa41299ca66080714caa8612a09ff0"
NEW_SOURCES = (
    "src/bayesian_phystwin_experiments/dlolab_wrapping_continuous_bayes_v1.py",
    "src/bayesian_phystwin_experiments/dlolab_wrapping_continuous_native.py",
    "scripts/remote/run_dlolab_wrapping_continuous_bayes_v1.py",
    "tests/test_dlolab_wrapping_continuous_bayes_v1.py",
    "tests/test_dlolab_wrapping_continuous_bayes_v1_custody.py",
    "docs/dlolab_wrapping_continuous_bayes_source_v1.md",
    "src/bayesian_phystwin_experiments/dlolab_wrapping_source.py",
    "src/bayesian_phystwin_experiments/dlolab_benchmark.py",
    "src/bayesian_phystwin_experiments/dlolab_native.py",
    "src/bayesian_phystwin_experiments/dlolab_regret_artifacts.py",
    "src/bayesian_phystwin_experiments/dlolab_slingshot_process.py",
    "src/bayesian_phystwin_experiments/deform_state_restart.py",
    "src/bayesian_phystwin/_portable_contracts.py",
    "src/bayesian_phystwin/_canonical_contracts.py",
)
POSITION_FIELDS = ("rod_pos_m", "gripper_pos_m", "post_pos_m")
Array: TypeAlias = NDArray[Any]


def runtime() -> dict[str, Any]:
    result = cast(dict[str, Any], runtime_identity())
    result["benchmark_packages"] = {
        name: importlib.metadata.version(name)
        for name in (
            "pin",
            "pin-pink",
            "proxsuite",
            "qpsolvers",
            "quadprog",
            "mushroom-rl",
            "omegaconf",
        )
    }
    return result


def native_source() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        source_identity(
            ASSETS / "upstream",
            ASSETS / "mushroom-rl",
            ASSETS / "dlo-lab.zip",
        ),
    )


def _source_hashes() -> dict[str, str]:
    if any(not (ROOT / name).is_file() for name in NEW_SOURCES):
        raise ValueError("complete wrapping continuous source required")
    return {name: file_digest(ROOT / name) for name in NEW_SOURCES}


def _parent() -> tuple[dict[str, Any], dict[str, Array]]:
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise ValueError("registered benchmark interpreter required")
    if any(
        file_digest(PARENT / name) != digest
        for name, digest in PARENT_FILE_SHA256.items()
    ):
        raise ValueError("registered wrapping source evidence changed")
    lock = read_record(PARENT / "lock.json")
    seal = read_record(PARENT / "source-bank" / "seal.json")
    result = read_record(PARENT / "result.json")
    bank = load_native_bundle(PARENT / "source-bank", seal["bundle"])
    metrics = result.get("metrics", {})
    if (
        lock.get("artifact_id") != PARENT_LOCK_ID
        or result.get("artifact_id") != PARENT_RESULT_ID
        or result.get("source_gate_passed") is not False
        or result.get("status") != "complete"
        or result.get("source_bank_id") != seal.get("artifact_id")
        or metrics.get("source_gate_passed") is not False
        or metrics.get("arms", {}).get("bias_aware_bayes", {}).get(
            "gain_over_best_fixed", 0
        )
        <= 0
        or bank.get("prefix", np.empty(0)).shape != (9, 3, 5, 3)
        or bank.get("reward", np.empty(0)).shape != (9, 8)
        or lock.get("runtime") != runtime()
        or lock.get("native_source") != native_source()
    ):
        raise ValueError("complete stopped wrapping source signal required")
    return lock, bank


def _preflight_files() -> dict[str, str]:
    paths = {
        "attempt.json": PREFLIGHT_ATTEMPT,
        "lock.json": PREFLIGHT / "lock.json",
        "claim.json": PREFLIGHT / "claim.json",
        "arrays.npz": PREFLIGHT / "arrays.npz",
        "seal.json": PREFLIGHT / "seal.json",
        "result.json": PREFLIGHT / "result.json",
    }
    if any(not path.is_file() or path.is_symlink() for path in paths.values()):
        raise ValueError("complete write-once wrapping runtime preflight required")
    return {name: file_digest(path) for name, path in paths.items()}


def _load_preflight() -> tuple[dict[str, Any], dict[str, str]]:
    hashes = _preflight_files()
    attempt = read_record(PREFLIGHT_ATTEMPT)
    lock = read_record(PREFLIGHT / "lock.json")
    claim = read_record(PREFLIGHT / "claim.json")
    seal = read_record(PREFLIGHT / "seal.json")
    result = read_record(PREFLIGHT / "result.json")
    data = load_native_bundle(PREFLIGHT, seal["bundle"])
    world_list = [preflight_world()] * 9
    qa = prefix_native_qa(data, seal["native"], world_list)
    if (
        attempt.get("schema") != "dlolab-wrapping-continuous-preflight-attempt-v1"
        or attempt.get("revision") != lock.get("revision")
        or attempt.get("source_sha256") != lock.get("source_sha256")
        or attempt.get("output_root") != str(PREFLIGHT)
        or lock.get("schema") != "dlolab-wrapping-continuous-preflight-lock-v1"
        or lock.get("source_sha256") != _source_hashes()
        or lock.get("runtime") != runtime()
        or lock.get("native_source") != native_source()
        or lock.get("attempt_id") != attempt.get("artifact_id")
        or claim.get("lock_id") != lock.get("artifact_id")
        or claim.get("prefix_only") is not True
        or seal.get("lock_id") != lock.get("artifact_id")
        or seal.get("claim_id") != claim.get("artifact_id")
        or result.get("lock_id") != lock.get("artifact_id")
        or result.get("seal_id") != seal.get("artifact_id")
        or result.get("qa") != qa
        or result.get("runtime_preflight_passed") is not True
        or result.get("study_attempt_consumed") is not False
        or any(
            record.get("retry_authorized") is not False
            for record in (attempt, lock, claim, result)
        )
        or not qa["qa_passed"]
    ):
        raise ValueError("registered wrapping runtime preflight changed")
    return result, hashes


def _validate(output: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Array]]:
    if output.resolve() != OUTPUT or output.is_symlink():
        raise ValueError("only the registered continuous wrapping root is permitted")
    lock = read_record(output / "lock.json")
    attempt = read_record(ATTEMPT)
    parent_lock, bank = _parent()
    preflight, preflight_hashes = _load_preflight()
    if (
        lock.get("schema") != "dlolab-wrapping-continuous-bayes-lock-v1"
        or lock.get("revision") != clean_revision(ROOT)
        or lock.get("source_sha256") != _source_hashes()
        or lock.get("protocol") != protocol()
        or lock.get("output_root") != str(OUTPUT)
        or lock.get("attempt_id") != attempt.get("artifact_id")
        or attempt.get("schema") != "dlolab-wrapping-continuous-bayes-attempt-v1"
        or attempt.get("revision") != lock.get("revision")
        or attempt.get("source_sha256") != lock.get("source_sha256")
        or attempt.get("protocol") != lock.get("protocol")
        or attempt.get("output_root") != str(OUTPUT)
        or lock.get("parent_file_sha256") != PARENT_FILE_SHA256
        or lock.get("parent_lock_id") != PARENT_LOCK_ID
        or lock.get("parent_result_id") != PARENT_RESULT_ID
        or lock.get("preflight_result_id") != preflight.get("artifact_id")
        or lock.get("preflight_file_sha256") != preflight_hashes
        or lock.get("source_prefix_sha256") != array_digest(bank["prefix"])
        or lock.get("source_reward_sha256") != array_digest(bank["reward"])
        or lock.get("runtime") != runtime()
        or lock.get("native_source") != native_source()
        or any(record.get("retry_authorized") is not False for record in (lock, attempt))
        or any(record.get("protected_data_read") is not False for record in (lock, attempt))
        or parent_lock.get("artifact_id") != PARENT_LOCK_ID
    ):
        raise ValueError("clean frozen continuous wrapping lock required")
    return lock, parent_lock, bank


def _worlds_for_task(task: dict[str, Any]) -> list[dict[str, Any]]:
    roster = continuous_worlds()
    if task["kind"] == "prefix_only":
        return [roster[index] for index in task["native_world_indices"]]
    return [roster[task["world_index"]]] * 9


def _load_task(
    output: Path,
    lock: dict[str, Any],
    task: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Array], dict[str, Any]]:
    directory = output / task["name"]
    worlds_for_native = _worlds_for_task(task)
    expected_authorization: dict[str, Any] = {"gate": "prefix_only_before_futures"}
    if task["kind"] == "all_action_future":
        barrier = _require_barrier(output, lock)
        expected_authorization = {
            "gate": "all_decisions_sealed",
            "barrier_id": barrier["artifact_id"],
        }
    claim = read_record(directory / "claim.json")
    seal = read_record(directory / "seal.json")
    if (
        claim.get("schema") != "dlolab-wrapping-continuous-claim-v1"
        or claim.get("lock_id") != lock["artifact_id"]
        or claim.get("task") != task
        or claim.get("authorization") != expected_authorization
        or claim.get("retry_authorized") is not False
        or seal.get("schema") != "dlolab-wrapping-continuous-seal-v1"
        or seal.get("lock_id") != lock["artifact_id"]
        or seal.get("claim_id") != claim["artifact_id"]
        or seal.get("task") != task
    ):
        raise ValueError("continuous wrapping task custody changed")
    data = load_native_bundle(directory, seal["bundle"])
    qa = (
        prefix_native_qa(data, seal["native"], worlds_for_native)
        if task["kind"] == "prefix_only"
        else future_native_qa(data, seal["native"], worlds_for_native[0])
    )
    return seal, data, qa


def _decision_contents(
    output: Path,
    lock: dict[str, Any],
    bank: dict[str, Array],
) -> tuple[dict[str, Array], list[str], list[dict[str, Any]]]:
    truth: Array = np.empty((WORLD_COUNT, 3, 5, 3), dtype=np.float64)
    ids: list[str] = []
    qas: list[dict[str, Any]] = []
    for batch in range(4):
        task = prefix_task(batch)
        seal, data, qa = _load_task(output, lock, task)
        count = len(task["world_indices"])
        truth[task["world_indices"]] = prefix_observation(data["rod_pos_m"])[:count]
        ids.append(seal["artifact_id"])
        qas.append(qa)
    return infer_decisions(bank["prefix"], bank["reward"], truth), ids, qas


def _load_decisions(
    output: Path,
    lock: dict[str, Any],
    bank: dict[str, Array],
) -> tuple[dict[str, Any], dict[str, Array], dict[str, Any]]:
    expected, prefix_ids, qas = _decision_contents(output, lock, bank)
    directory = output / "decisions"
    seal = read_record(directory / "seal.json")
    data = load_native_bundle(directory, seal["bundle"])
    if (
        seal.get("schema") != "dlolab-wrapping-continuous-decision-seal-v1"
        or seal.get("lock_id") != lock["artifact_id"]
        or seal.get("prefix_seal_ids") != prefix_ids
        or seal.get("parent_source_bank_id") != lock["parent_source_bank_id"]
        or seal.get("future_simulated") is not False
        or seal.get("future_read") is not False
        or set(data) != set(expected)
        or any(not np.array_equal(data[name], expected[name]) for name in expected)
    ):
        raise ValueError("sealed continuous wrapping decisions changed")
    gate = pre_future_checks(
        data["decisions"], all_prefix_qa=all(qa["qa_passed"] for qa in qas)
    )
    return seal, data, gate


def _barrier_contents(
    output: Path,
    lock: dict[str, Any],
) -> dict[str, Any]:
    _, _, bank = _validate(output)
    seal, _, gate = _load_decisions(output, lock, bank)
    return {
        "schema": "dlolab-wrapping-continuous-decision-barrier-v1",
        "lock_id": lock["artifact_id"],
        "decision_seal_id": seal["artifact_id"],
        "pre_future": gate,
        "future_simulated": False,
        "future_read": False,
    }


def _require_barrier(output: Path, lock: dict[str, Any]) -> dict[str, Any]:
    barrier: dict[str, Any] = read_record(output / "decision-barrier.json")
    expected = _barrier_contents(output, lock)
    if any(barrier.get(key) != value for key, value in expected.items()):
        raise ValueError("continuous wrapping decision barrier changed")
    if barrier["pre_future"]["pre_future_gate_passed"] is not True:
        raise ValueError("continuous wrapping pre-future gate did not pass")
    return barrier


def _worker(output: Path, kind: str, index: int) -> None:
    lock, parent_lock, _ = _validate(output)
    task = prefix_task(index) if kind == "prefix" else future_task(index)
    worlds_for_native = _worlds_for_task(task)
    authorization: dict[str, Any] = {"gate": "prefix_only_before_futures"}
    if kind == "future":
        barrier = _require_barrier(output, lock)
        authorization = {
            "gate": "all_decisions_sealed",
            "barrier_id": barrier["artifact_id"],
        }
    directory = output / task["name"]
    directory.mkdir()
    claim = write_record(
        directory / "claim.json",
        {
            "schema": "dlolab-wrapping-continuous-claim-v1",
            "lock_id": lock["artifact_id"],
            "task": task,
            "authorization": authorization,
            "retry_authorized": False,
        },
    )
    try:
        data, native = run_worlds(
            ASSETS / "upstream",
            directory,
            worlds_for_native,
            prefix_only=kind == "prefix",
        )
        bundle = write_native_bundle(directory, data)
        write_record(
            directory / "seal.json",
            {
                "schema": "dlolab-wrapping-continuous-seal-v1",
                "lock_id": lock["artifact_id"],
                "claim_id": claim["artifact_id"],
                "task": task,
                "native": native,
                "bundle": bundle,
            },
        )
    except Exception as error:
        write_record(
            directory / "failure.json",
            {
                "schema": "dlolab-wrapping-continuous-failure-v1",
                "lock_id": lock["artifact_id"],
                "claim_id": claim["artifact_id"],
                "task": task,
                "error_type": type(error).__name__,
                "message": str(error),
                "retry_authorized": False,
                "protected_data_read": False,
            },
        )
        raise


def _execute(output: Path, kind: str, index: int) -> None:
    task = prefix_task(index) if kind == "prefix" else future_task(index)
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--output",
        str(output),
        "--worker-kind",
        kind,
        "--worker-index",
        str(index),
    ]
    with (output / f"{task['name']}.log").open("x") as stream:
        run = subprocess.run(
            command,
            cwd=ROOT,
            env=os.environ.copy(),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if run.returncode:
        raise RuntimeError(f"{task['name']} exited {run.returncode}; no retry")


def _load_future(
    output: Path,
    lock: dict[str, Any],
    index: int,
) -> tuple[dict[str, Any], list[float], dict[str, Any], float]:
    task = future_task(index)
    seal, data, qa = _load_task(output, lock, task)
    prefix_spec = prefix_task(index // 9)
    _, prefix_data, _ = _load_task(output, lock, prefix_spec)
    slot = index % 9
    prefix_difference = max(
        float(
            np.abs(
                prefix_data[name][:, slot]
                - data[name][:600, 1]
            ).max()
        )
        for name in POSITION_FIELDS
    )
    if prefix_difference > 0.001:
        raise ValueError("prefix-only and full-future reset mismatch")
    return seal, [float(value) for value in qa["final_rewards"][:8]], qa, prefix_difference


def _run_preflight() -> None:
    if (
        PREFLIGHT.exists()
        or PREFLIGHT.is_symlink()
        or PREFLIGHT_ATTEMPT.exists()
        or PREFLIGHT_ATTEMPT.is_symlink()
    ):
        raise ValueError("one fresh wrapping runtime preflight required")
    revision = clean_revision(ROOT)
    parent_lock, _ = _parent()
    sources = _source_hashes()
    attempt = write_record(
        PREFLIGHT_ATTEMPT,
        {
            "schema": "dlolab-wrapping-continuous-preflight-attempt-v1",
            "revision": revision,
            "source_sha256": sources,
            "output_root": str(PREFLIGHT),
            "retry_authorized": False,
            "protected_data_read": False,
        },
    )
    PREFLIGHT.mkdir()
    lock = write_record(
        PREFLIGHT / "lock.json",
        {
            "schema": "dlolab-wrapping-continuous-preflight-lock-v1",
            "revision": revision,
            "source_sha256": sources,
            "output_root": str(PREFLIGHT),
            "attempt_id": attempt["artifact_id"],
            "runtime": runtime(),
            "native_source": native_source(),
            "parent_lock_id": parent_lock["artifact_id"],
            "retry_authorized": False,
            "protected_data_read": False,
        },
    )
    claim = write_record(
        PREFLIGHT / "claim.json",
        {
            "schema": "dlolab-wrapping-continuous-preflight-claim-v1",
            "lock_id": lock["artifact_id"],
            "prefix_only": True,
            "retry_authorized": False,
        },
    )
    try:
        world_list = [preflight_world()] * 9
        data, native = run_worlds(
            ASSETS / "upstream", PREFLIGHT, world_list, prefix_only=True
        )
        seal = write_record(
            PREFLIGHT / "seal.json",
            {
                "schema": "dlolab-wrapping-continuous-preflight-seal-v1",
                "lock_id": lock["artifact_id"],
                "claim_id": claim["artifact_id"],
                "bundle": write_native_bundle(PREFLIGHT, data),
                "native": native,
            },
        )
        qa = prefix_native_qa(data, native, world_list)
        result = write_record(
            PREFLIGHT / "result.json",
            {
                "schema": "dlolab-wrapping-continuous-preflight-result-v1",
                "lock_id": lock["artifact_id"],
                "seal_id": seal["artifact_id"],
                "qa": qa,
                "runtime_preflight_passed": qa["qa_passed"],
                "study_attempt_consumed": False,
                "retry_authorized": False,
                "protected_data_read": False,
            },
        )
        if not result["runtime_preflight_passed"]:
            raise ValueError("wrapping runtime preflight failed")
        print(f"runtime preflight passed; id={result['artifact_id']}", flush=True)
    except Exception as error:
        write_record(
            PREFLIGHT / "failure.json",
            {
                "schema": "dlolab-wrapping-continuous-preflight-failure-v1",
                "lock_id": lock["artifact_id"],
                "error_type": type(error).__name__,
                "message": str(error),
                "retry_authorized": False,
                "protected_data_read": False,
            },
        )
        raise


def _run(output: Path) -> None:
    if (
        output.resolve() != OUTPUT
        or output.exists()
        or output.is_symlink()
        or ATTEMPT.exists()
        or ATTEMPT.is_symlink()
    ):
        raise ValueError("one fresh continuous wrapping attempt required")
    revision = clean_revision(ROOT)
    parent_lock, bank = _parent()
    preflight, preflight_hashes = _load_preflight()
    sources = _source_hashes()
    parent_seal = read_record(PARENT / "source-bank" / "seal.json")
    attempt = write_record(
        ATTEMPT,
        {
            "schema": "dlolab-wrapping-continuous-bayes-attempt-v1",
            "revision": revision,
            "source_sha256": sources,
            "protocol": protocol(),
            "output_root": str(OUTPUT),
            "retry_authorized": False,
            "protected_data_read": False,
        },
    )
    output.mkdir()
    lock = write_record(
        output / "lock.json",
        {
            "schema": "dlolab-wrapping-continuous-bayes-lock-v1",
            "revision": revision,
            "source_sha256": sources,
            "protocol": protocol(),
            "output_root": str(OUTPUT),
            "attempt_id": attempt["artifact_id"],
            "parent_lock_id": PARENT_LOCK_ID,
            "parent_result_id": PARENT_RESULT_ID,
            "parent_source_bank_id": parent_seal["artifact_id"],
            "parent_file_sha256": PARENT_FILE_SHA256,
            "preflight_result_id": preflight["artifact_id"],
            "preflight_file_sha256": preflight_hashes,
            "source_prefix_sha256": array_digest(bank["prefix"]),
            "source_reward_sha256": array_digest(bank["reward"]),
            "runtime": runtime(),
            "native_source": native_source(),
            "retry_authorized": False,
            "protected_data_read": False,
        },
    )
    stage = "prefixes"
    try:
        for batch in range(4):
            _execute(output, "prefix", batch)
        decision_data, prefix_ids, qas = _decision_contents(output, lock, bank)
        directory = output / "decisions"
        directory.mkdir()
        decision_seal = write_record(
            directory / "seal.json",
            {
                "schema": "dlolab-wrapping-continuous-decision-seal-v1",
                "lock_id": lock["artifact_id"],
                "prefix_seal_ids": prefix_ids,
                "parent_source_bank_id": parent_seal["artifact_id"],
                "bundle": write_native_bundle(directory, decision_data),
                "future_simulated": False,
                "future_read": False,
            },
        )
        gate = pre_future_checks(
            decision_data["decisions"],
            all_prefix_qa=all(qa["qa_passed"] for qa in qas),
        )
        barrier = write_record(
            output / "decision-barrier.json",
            {
                "schema": "dlolab-wrapping-continuous-decision-barrier-v1",
                "lock_id": lock["artifact_id"],
                "decision_seal_id": decision_seal["artifact_id"],
                "pre_future": gate,
                "future_simulated": False,
                "future_read": False,
            },
        )
        if not gate["pre_future_gate_passed"]:
            write_record(
                output / "result.json",
                {
                    "schema": "dlolab-wrapping-continuous-bayes-result-v1",
                    "status": "pre_future_gate_failed",
                    "lock_id": lock["artifact_id"],
                    "decision_seal_id": decision_seal["artifact_id"],
                    "barrier_id": barrier["artifact_id"],
                    "pre_future": gate,
                    "task_future_generated": False,
                    "source_gate_passed": False,
                    "retry_authorized": False,
                    "protected_data_read": False,
                },
            )
            return
        stage = "futures"
        for index in range(WORLD_COUNT):
            _execute(output, "future", index)
        rewards: list[list[float]] = []
        future_ids: list[str] = []
        future_qa: list[dict[str, Any]] = []
        prefix_match: list[float] = []
        for index in range(WORLD_COUNT):
            seal, row, qa, difference = _load_future(output, lock, index)
            rewards.append(row)
            future_ids.append(seal["artifact_id"])
            future_qa.append(qa)
            prefix_match.append(difference)
        reward = np.asarray(rewards, dtype=np.float64)
        stage = "generation"
        generation_dir = output / "generation"
        generation_dir.mkdir()
        generation = write_record(
            generation_dir / "seal.json",
            {
                "schema": "dlolab-wrapping-continuous-generation-v1",
                "lock_id": lock["artifact_id"],
                "barrier_id": barrier["artifact_id"],
                "future_seal_ids": future_ids,
                "native_qa": future_qa,
                "prefix_match_error_m": prefix_match,
                "bundle": write_native_bundle(generation_dir, {"reward": reward}),
                "ordinary_worlds": WORLD_COUNT,
                "technical_failures": 0,
                "replacements": 0,
            },
        )
        stage = "score"
        metrics = score(
            decision_data["decisions"],
            reward,
            all_native_qa=all(qa["qa_passed"] for qa in future_qa),
        )
        result = write_record(
            output / "result.json",
            {
                **metrics,
                "status": "complete",
                "lock_id": lock["artifact_id"],
                "decision_seal_id": decision_seal["artifact_id"],
                "barrier_id": barrier["artifact_id"],
                "generation_id": generation["artifact_id"],
                "pre_future": gate,
                "task_future_generated": True,
                "retry_authorized": False,
            },
        )
        print(
            f"continuous wrapping gate={result['source_gate_passed']}; "
            f"id={result['artifact_id']}",
            flush=True,
        )
    except Exception as error:
        write_record(
            output / "failure.json",
            {
                "schema": "dlolab-wrapping-continuous-failure-v1",
                "lock_id": lock["artifact_id"],
                "terminal_stage": stage,
                "completed_prefix_batches": sum(
                    (output / prefix_task(batch)["name"] / "seal.json").is_file()
                    for batch in range(4)
                ),
                "completed_future_worlds": sum(
                    (output / future_task(index)["name"] / "seal.json").is_file()
                    for index in range(WORLD_COUNT)
                ),
                "error_type": type(error).__name__,
                "message": str(error),
                "retry_authorized": False,
                "replacement_authorized": False,
                "protected_data_read": False,
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--worker-kind", choices=("prefix", "future"))
    parser.add_argument("--worker-index", type=int)
    args = parser.parse_args()
    worker = args.worker_kind is not None or args.worker_index is not None
    if args.preflight:
        if worker or args.output != OUTPUT:
            raise ValueError("preflight cannot combine with study arguments")
        _run_preflight()
    elif worker:
        if args.worker_kind is None or args.worker_index is None:
            raise ValueError("complete registered worker specification required")
        _worker(args.output, args.worker_kind, args.worker_index)
    else:
        _run(args.output)


if __name__ == "__main__":
    main()
