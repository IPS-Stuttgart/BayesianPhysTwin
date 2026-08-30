#!/usr/bin/env python3
"""Run the one-attempt public DLO-Lab Slingshot guard replication."""

from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin_experiments.dlolab_benchmark import write_native_bundle
from bayesian_phystwin_experiments.dlolab_native import array_digest, file_digest
from bayesian_phystwin_experiments.dlolab_regret_artifacts import (
    clean_revision,
    read_record,
    write_record,
)
from bayesian_phystwin_experiments.dlolab_slingshot_batch import TRACE_NAMES
from bayesian_phystwin_experiments.dlolab_slingshot_belief import (
    BASELINE,
    native_qa,
    prefix_observations,
)
from bayesian_phystwin_experiments.dlolab_slingshot_belief_native import (
    run_registered_worlds,
)
from bayesian_phystwin_experiments.dlolab_slingshot_certified_guard_v2 import (
    PARENT_BANK_ID,
    PARENT_CALIBRATOR_ID,
    PARENT_FILE_SHA256,
    PARENT_LOCK_ID,
    PARENT_RESULT_ID,
    PARENT_ROOT,
    PREFIX_BATCH_COUNT,
    WORLD_COUNT,
    continuous_worlds,
    future_task,
    infer_decisions,
    pre_future_checks,
    prefix_task,
    protocol,
    score,
)
from bayesian_phystwin_experiments.dlolab_slingshot_cmaes import worker_environment
from bayesian_phystwin_experiments.dlolab_slingshot_process import (
    load_native_bundle,
    runtime,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(
    "/home/fpfaff/source-only/dlolab-slingshot-certified-guard-source-v2"
)
FUTURE_WORKERS = 4
SOURCES = (
    "src/bayesian_phystwin_experiments/dlolab_slingshot_certified_guard_v2.py",
    "scripts/remote/run_dlolab_slingshot_certified_guard_v2.py",
    "tests/test_dlolab_slingshot_certified_guard_v2.py",
    "tests/test_dlolab_slingshot_certified_guard_v2_custody.py",
    "docs/dlolab_slingshot_certified_guard_source_v2.md",
    "src/bayesian_phystwin_experiments/dlolab_slingshot_belief.py",
    "src/bayesian_phystwin_experiments/dlolab_slingshot_belief_native.py",
    "src/bayesian_phystwin_experiments/dlolab_slingshot_batch.py",
    "src/bayesian_phystwin_experiments/dlolab_slingshot_process.py",
    "src/bayesian_phystwin_experiments/dlolab_slingshot_value.py",
    "src/bayesian_phystwin_experiments/coupled_action_regret.py",
    "src/bayesian_phystwin_experiments/dlolab_benchmark.py",
    "src/bayesian_phystwin_experiments/dlolab_regret_artifacts.py",
    "src/bayesian_phystwin/guard_harm_risk.py",
)
POSITION_FIELDS = ("rod_pos_m", "sphere_pos_m", "cube_pos_m", "gripper_pos_m")


def _parent_root() -> Path:
    return Path(PARENT_ROOT)


def load_parent() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    parent = _parent_root()
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("exact frozen parent root required")
    for name, digest in PARENT_FILE_SHA256.items():
        path = parent / name
        if path.is_symlink() or file_digest(path) != digest:
            raise ValueError(f"frozen parent artifact changed: {name}")
    lock = read_record(parent / "lock.json")
    result = read_record(parent / "result.json")
    calibrator = read_record(parent / "calibrator.json")
    bank_seal = read_record(parent / "model-bank/seal.json")
    if (
        lock["artifact_id"] != PARENT_LOCK_ID
        or result["artifact_id"] != PARENT_RESULT_ID
        or result["lock_id"] != PARENT_LOCK_ID
        or result["source_gate_passed"] is not False
        or calibrator["artifact_id"] != PARENT_CALIBRATOR_ID
        or calibrator["lock_id"] != PARENT_LOCK_ID
        or calibrator["evaluation_futures_read"] is not False
        or bank_seal["artifact_id"] != PARENT_BANK_ID
        or bank_seal["lock_id"] != PARENT_LOCK_ID
        or result["arms"]["mean_regret_guard"]["mean_gain_over_incumbent"]
        != 0.0022203773260116577
        or result["arms"]["mean_regret_guard"][
            "harmful_decisions_beyond_numeric_margin"
        ]
        != 0
        or result["arms"]["mean_regret_guard"]["nonfallback_decisions"] != 3
    ):
        raise ValueError("frozen parent identity or development lead changed")
    for name, digest in lock["source_sha256"].items():
        if file_digest(ROOT / name) != digest:
            raise ValueError(f"parent implementation changed: {name}")
    arrays = load_native_bundle(parent / "model-bank", bank_seal["bundle"])
    if (
        set(arrays) != {"prefix", "reward"}
        or arrays["prefix"].shape != (27, 3, 4, 3)
        or arrays["reward"].shape != (27, 7)
        or calibrator["calibrations"]["mean"]
        != {
            "coverage": 0.9,
            "count": 19,
            "rank": 18,
            "offset": 0.7285524030751176,
        }
    ):
        raise ValueError("frozen parent bank or calibrator changed")
    identity = {
        "root": str(parent.resolve()),
        "lock_id": lock["artifact_id"],
        "result_id": result["artifact_id"],
        "calibrator_id": calibrator["artifact_id"],
        "bank_id": bank_seal["artifact_id"],
        "file_sha256": PARENT_FILE_SHA256,
        "assets_root": lock["assets_root"],
        "runtime": lock["screen"]["source"]["controller"]["runtime"],
        "controls": lock["controls"],
    }
    return identity, arrays


def _task(kind: str, index: int) -> dict[str, Any]:
    if kind == "prefix":
        return prefix_task(index)
    if kind == "future":
        return future_task(index)
    raise ValueError("unregistered Slingshot replication task")


def _task_worlds(spec: dict[str, Any]) -> list[dict[str, Any]]:
    roster = continuous_worlds()
    if spec["kind"] == "prefix_only":
        return [roster[index] for index in spec["native_world_indices"]]
    return [roster[spec["world_index"]]] * 8


def _expected_controls(lock: dict[str, Any], spec: dict[str, Any]) -> np.ndarray:
    bank = np.asarray(lock["controls"], dtype=np.float64)
    if bank.shape != (8, 3, 6):
        raise ValueError("frozen native action bank changed")
    if spec["kind"] == "prefix_only":
        return np.repeat(bank[BASELINE : BASELINE + 1], 8, axis=0)
    return bank


def _validate_realization(
    native: dict[str, Any], expected_worlds: list[dict[str, Any]]
) -> None:
    realization = native.get("world_realization", {})
    if realization.get("bending") != [
        [world["bending_E"] for world in expected_worlds]
    ] or realization.get("stretching") != [
        [world["stretching_K"] for world in expected_worlds]
    ]:
        raise ValueError("realized material parameters changed")
    for name, y, z in (("sphere", 0.06, 0.2), ("cube", 0.23, 0.22)):
        expected = np.asarray(
            [[0.12 + world["x_offset_m"], y, z] for world in expected_worlds]
        )
        actual = np.asarray(realization.get(f"{name}_initial_position_m"))
        if actual.shape != (8, 3) or not np.allclose(
            actual, expected, rtol=0, atol=1e-15
        ):
            raise ValueError("realized object placement changed")


def prefix_native_qa(
    arrays: dict[str, np.ndarray],
    native: dict[str, Any],
    expected_controls: np.ndarray,
    expected_worlds: list[dict[str, Any]],
) -> dict[str, Any]:
    if (
        set(arrays) != set(TRACE_NAMES + ("controls",))
        or any(arrays[name].shape[:2] != (300, 8) for name in TRACE_NAMES)
        or any(not np.isfinite(value).all() for value in arrays.values())
        or array_digest(arrays["controls"]) != array_digest(expected_controls)
        or native.get("native_steps") != 300
        or native.get("future_simulated") is not False
        or native.get("reward_scored") is not False
        or native.get("hidden_state_restart") is not False
    ):
        raise ValueError("causal-prefix native contract changed")
    _validate_realization(native, expected_worlds)
    prefix_observations(arrays)
    fixed = float(
        np.max(
            np.abs(
                arrays["rod_pos_m"][:, :, [0, 1, 10, 11]]
                - arrays["rod_pos_m"][:1, :, [0, 1, 10, 11]]
            )
        )
    )
    checks = {
        "complete_causal_prefix": True,
        "no_future_simulated": True,
        "no_reward_scored": True,
        "fixed_endpoints": fixed <= 1e-9,
    }
    return {
        "checks": checks,
        "fixed_endpoint_error_m": fixed,
        "qa_passed": bool(all(checks.values())),
    }


def _task_qa(
    arrays: dict[str, np.ndarray],
    native: dict[str, Any],
    expected_controls: np.ndarray,
    expected_worlds: list[dict[str, Any]],
    spec: dict[str, Any],
) -> dict[str, Any]:
    _validate_realization(native, expected_worlds)
    if spec["kind"] == "prefix_only":
        return prefix_native_qa(arrays, native, expected_controls, expected_worlds)
    return native_qa(arrays, native, expected_controls)


def freeze(output: Path) -> dict[str, Any]:
    if output.resolve() != OUTPUT_ROOT:
        raise ValueError("only the registered one-attempt root is authorized")
    revision = clean_revision(ROOT)
    parent, _ = load_parent()
    if runtime() != parent["runtime"]:
        raise ValueError("exact parent-qualified runtime required")
    output.mkdir(parents=True, exist_ok=False)
    return write_record(
        output / "lock.json",
        {
            "schema": "dlolab-slingshot-certified-guard-lock-v2",
            "source_revision": revision,
            "source_sha256": {name: file_digest(ROOT / name) for name in SOURCES},
            "protocol": protocol(),
            "parent": parent,
            "controls": parent["controls"],
            "assets_root": parent["assets_root"],
            "runtime": parent["runtime"],
            "future_workers": FUTURE_WORKERS,
            "output_root": str(output.resolve()),
            "protected_data_read": False,
        },
    )


def validate_lock(output: Path) -> dict[str, Any]:
    lock = read_record(output / "lock.json")
    parent, _ = load_parent()
    if (
        lock.get("schema") != "dlolab-slingshot-certified-guard-lock-v2"
        or output.resolve() != OUTPUT_ROOT
        or lock.get("output_root") != str(output.resolve())
        or lock.get("source_revision") != clean_revision(ROOT)
        or lock.get("source_sha256")
        != {name: file_digest(ROOT / name) for name in SOURCES}
        or lock.get("protocol") != protocol()
        or lock.get("parent") != parent
        or lock.get("controls") != parent["controls"]
        or lock.get("assets_root") != parent["assets_root"]
        or lock.get("runtime") != runtime()
        or lock.get("future_workers") != FUTURE_WORKERS
    ):
        raise ValueError("frozen Slingshot replication lock changed")
    return lock


def load_task(
    output: Path, lock: dict[str, Any], kind: str, index: int
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    spec = _task(kind, index)
    directory = output / spec["name"]
    claim = read_record(directory / "claim.json")
    seal = read_record(directory / "seal.json")
    if (
        claim.get("schema") != "dlolab-slingshot-certified-guard-claim-v2"
        or seal.get("schema") != "dlolab-slingshot-certified-guard-seal-v2"
        or claim.get("lock_id") != lock["artifact_id"]
        or seal.get("lock_id") != lock["artifact_id"]
        or claim.get("task") != spec
        or seal.get("task") != spec
        or seal.get("claim_id") != claim["artifact_id"]
        or claim.get("retry_authorized") is not False
    ):
        raise ValueError("native task custody changed")
    arrays = load_native_bundle(directory, seal["bundle"])
    controls_expected = _expected_controls(lock, spec)
    worlds_expected = _task_worlds(spec)
    qa = _task_qa(arrays, seal["native"], controls_expected, worlds_expected, spec)
    if not qa["qa_passed"] or seal.get("qa") != qa:
        raise ValueError("native task QA changed or failed")
    return seal, arrays


def _barrier_records(output: Path, lock: dict[str, Any]) -> tuple[dict, dict]:
    decision = read_record(output / "decisions/seal.json")
    barrier = read_record(output / "decision-barrier.json")
    if (
        decision.get("schema") != "dlolab-slingshot-certified-guard-decisions-v2"
        or decision.get("lock_id") != lock["artifact_id"]
        or decision.get("parent_bank_id") != PARENT_BANK_ID
        or decision.get("parent_calibrator_id") != PARENT_CALIBRATOR_ID
        or decision.get("protected_data_read") is not False
        or barrier.get("schema") != "dlolab-slingshot-certified-guard-barrier-v2"
        or barrier.get("lock_id") != lock["artifact_id"]
        or barrier.get("decision_seal_id") != decision["artifact_id"]
        or barrier.get("pre_future_gate_passed") is not True
        or barrier.get("future_simulated") is not False
        or barrier.get("future_read") is not False
        or barrier.get("protected_data_read") is not False
        or file_digest(output / "decisions/arrays.npz")
        != decision["bundle"]["file_sha256"]
    ):
        raise ValueError("complete passing decision barrier required")
    prefix_ids = [
        read_record(output / prefix_task(i)["name"] / "seal.json")["artifact_id"]
        for i in range(PREFIX_BATCH_COUNT)
    ]
    if barrier.get("prefix_seal_ids") != prefix_ids:
        raise ValueError("decision barrier prefix lineage changed")
    return decision, barrier


def worker(output: Path, kind: str, index: int) -> None:
    lock = validate_lock(output)
    spec = _task(kind, index)
    authorization: dict[str, Any] = {"gate": "registered_causal_prefix"}
    if spec["kind"] == "all_action_future":
        decision, barrier = _barrier_records(output, lock)
        authorization = {
            "gate": "complete_passing_decision_barrier",
            "decision_seal_id": decision["artifact_id"],
            "barrier_id": barrier["artifact_id"],
        }
    directory = output / spec["name"]
    directory.mkdir(exist_ok=False)
    claim = write_record(
        directory / "claim.json",
        {
            "schema": "dlolab-slingshot-certified-guard-claim-v2",
            "lock_id": lock["artifact_id"],
            "task": spec,
            "authorization": authorization,
            "retry_authorized": False,
            "replacement_authorized": False,
            "protected_data_read": False,
        },
    )
    try:
        expected_controls = _expected_controls(lock, spec)
        expected_worlds = _task_worlds(spec)
        arrays, native = run_registered_worlds(
            Path(lock["assets_root"]) / "upstream",
            directory,
            expected_controls,
            expected_worlds,
            prefix_only=spec["kind"] == "prefix_only",
        )
        qa = _task_qa(arrays, native, expected_controls, expected_worlds, spec)
        if not qa["qa_passed"]:
            raise ValueError("native task QA failed")
        bundle = write_native_bundle(directory, arrays)
        write_record(
            directory / "seal.json",
            {
                "schema": "dlolab-slingshot-certified-guard-seal-v2",
                "lock_id": lock["artifact_id"],
                "claim_id": claim["artifact_id"],
                "task": spec,
                "native": native,
                "qa": qa,
                "bundle": bundle,
            },
        )
    except Exception as error:
        write_record(
            directory / "failure.json",
            {
                "schema": "dlolab-slingshot-certified-guard-failure-v2",
                "lock_id": lock["artifact_id"],
                "claim_id": claim["artifact_id"],
                "task": spec,
                "error_type": type(error).__name__,
                "message": str(error),
                "retry_authorized": False,
                "replacement_authorized": False,
                "protected_data_read": False,
            },
        )
        raise


def execute(output: Path, lock: dict[str, Any], kind: str, index: int) -> None:
    spec = _task(kind, index)
    print(f"native Slingshot replication stage: {spec['name']}", flush=True)
    with (output / f"{spec['name']}.log").open("x") as stream:
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "--output",
                str(output.resolve()),
                "--worker-kind",
                kind,
                "--worker-index",
                str(index),
            ],
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=worker_environment(lock["runtime"]),
            check=True,
        )


def execute_many(
    output: Path,
    lock: dict[str, Any],
    kind: str,
    indices: range,
    *,
    workers: int,
) -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(execute, output, lock, kind, index) for index in indices
        ]
        for future in futures:
            future.result()


def _decision_artifact(
    output: Path, lock: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    _, bank = load_parent()
    truth: list[np.ndarray] = []
    prefix_ids: list[str] = []
    for batch in range(PREFIX_BATCH_COUNT):
        seal, arrays = load_task(output, lock, "prefix", batch)
        prefix_ids.append(seal["artifact_id"])
        observation = prefix_observations(arrays)
        truth.extend(observation[: len(prefix_task(batch)["world_indices"])])
    truth_array = np.stack(truth)
    inferred = infer_decisions(truth_array, bank["prefix"], bank["reward"])
    arrays = {"truth_prefix_m": truth_array, **inferred}
    preflight = pre_future_checks(inferred["decision"], all_prefix_qa=True)
    metadata = {
        "schema": "dlolab-slingshot-certified-guard-decisions-v2",
        "lock_id": lock["artifact_id"],
        "parent_bank_id": PARENT_BANK_ID,
        "parent_calibrator_id": PARENT_CALIBRATOR_ID,
        "prefix_seal_ids": prefix_ids,
        "pre_future": preflight,
        "future_simulated": False,
        "future_read": False,
        "protected_data_read": False,
    }
    return metadata, arrays, preflight


def seal_decisions(output: Path, lock: dict[str, Any]) -> dict[str, Any]:
    metadata, arrays, preflight = _decision_artifact(output, lock)
    directory = output / "decisions"
    directory.mkdir(exist_ok=False)
    bundle = write_native_bundle(directory, arrays)
    decision = write_record(directory / "seal.json", {**metadata, "bundle": bundle})
    barrier = write_record(
        output / "decision-barrier.json",
        {
            "schema": "dlolab-slingshot-certified-guard-barrier-v2",
            "lock_id": lock["artifact_id"],
            "decision_seal_id": decision["artifact_id"],
            "prefix_seal_ids": metadata["prefix_seal_ids"],
            "pre_future": preflight,
            "pre_future_gate_passed": preflight["pre_future_gate_passed"],
            "future_simulated": False,
            "future_read": False,
            "protected_data_read": False,
        },
    )
    if not preflight["pre_future_gate_passed"]:
        raise ValueError("registered pre-future gate failed")
    return barrier


def load_decisions(
    output: Path, lock: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    decision, barrier = _barrier_records(output, lock)
    arrays = load_native_bundle(output / "decisions", decision["bundle"])
    metadata, expected, preflight = _decision_artifact(output, lock)
    if any(decision.get(key) != value for key, value in metadata.items()) or (
        set(arrays) != set(expected)
        or any(
            array_digest(arrays[name]) != array_digest(expected[name])
            for name in arrays
        )
    ):
        raise ValueError("sealed decisions do not reproduce")
    if barrier["pre_future"] != preflight:
        raise ValueError("pre-future arithmetic changed")
    return decision, arrays, barrier


def future_table(
    output: Path, lock: dict[str, Any]
) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    decision, _, barrier = load_decisions(output, lock)
    rewards: list[list[float]] = []
    seals: list[str] = []
    qas: list[dict[str, Any]] = []
    for index in range(WORLD_COUNT):
        seal, _ = load_task(output, lock, "future", index)
        claim = read_record(output / future_task(index)["name"] / "claim.json")
        if claim["authorization"] != {
            "gate": "complete_passing_decision_barrier",
            "decision_seal_id": decision["artifact_id"],
            "barrier_id": barrier["artifact_id"],
        }:
            raise ValueError("future was not bound to sealed prefix decisions")
        rewards.append([row["native_reward"] for row in seal["qa"]["metrics"][:7]])
        seals.append(seal["artifact_id"])
        qas.append(seal["qa"])
    return np.asarray(rewards, dtype=np.float64), seals, qas


def run(output: Path) -> dict[str, Any]:
    lock = freeze(output)
    stage = "prefixes"
    try:
        execute_many(
            output,
            lock,
            "prefix",
            range(PREFIX_BATCH_COUNT),
            workers=FUTURE_WORKERS,
        )
        stage = "decision_barrier"
        seal_decisions(output, lock)
        stage = "futures"
        execute_many(
            output,
            lock,
            "future",
            range(WORLD_COUNT),
            workers=FUTURE_WORKERS,
        )
        stage = "score"
        decision_seal, decisions, barrier = load_decisions(output, lock)
        rewards, future_seals, qas = future_table(output, lock)
        generation = write_record(
            output / "generation.json",
            {
                "schema": "dlolab-slingshot-certified-guard-generation-v2",
                "lock_id": lock["artifact_id"],
                "decision_seal_id": decision_seal["artifact_id"],
                "barrier_id": barrier["artifact_id"],
                "ordinary_native_worlds": WORLD_COUNT,
                "technical_failures": 0,
                "replacements": 0,
                "future_seal_ids": future_seals,
            },
        )
        result = write_record(
            output / "result.json",
            {
                **score(
                    decisions["decision"],
                    rewards,
                    all_native_qa=all(qa["qa_passed"] for qa in qas),
                    pre_future_gate_passed=barrier["pre_future_gate_passed"],
                ),
                "lock_id": lock["artifact_id"],
                "decision_seal_id": decision_seal["artifact_id"],
                "barrier_id": barrier["artifact_id"],
                "generation_id": generation["artifact_id"],
                "future_seal_ids": future_seals,
            },
        )
        print(
            f"Slingshot guard gate={result['source_gate_passed']}; "
            f"id={result['artifact_id']}",
            flush=True,
        )
        return result
    except Exception as error:
        completed_prefix = sum(
            (output / prefix_task(index)["name"] / "seal.json").is_file()
            for index in range(PREFIX_BATCH_COUNT)
        )
        completed_future = sum(
            (output / future_task(index)["name"] / "seal.json").is_file()
            for index in range(WORLD_COUNT)
        )
        write_record(
            output / "failure.json",
            {
                "schema": "dlolab-slingshot-certified-guard-run-failure-v2",
                "lock_id": lock["artifact_id"],
                "terminal_stage": stage,
                "completed_prefix_batches": completed_prefix,
                "completed_future_worlds": completed_future,
                "error_type": type(error).__name__,
                "message": str(error),
                "retry_authorized": False,
                "replacement_authorized": False,
                "protected_data_read": False,
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-kind", choices=("prefix", "future"))
    parser.add_argument("--worker-index", type=int)
    args = parser.parse_args()
    if args.worker_kind is not None and args.worker_index is not None:
        worker(args.output, args.worker_kind, args.worker_index)
    elif args.worker_kind is not None or args.worker_index is not None:
        parser.error("both registered worker arguments are required")
    else:
        run(args.output)
