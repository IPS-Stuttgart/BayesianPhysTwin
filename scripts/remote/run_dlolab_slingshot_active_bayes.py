#!/usr/bin/env python3
"""Run the frozen fresh-world Slingshot active-Bayes source study."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from bayesian_phystwin_experiments.dlolab_benchmark import write_native_bundle
from bayesian_phystwin_experiments.dlolab_native import array_digest, file_digest
from bayesian_phystwin_experiments.dlolab_regret_artifacts import (
    clean_revision,
    read_record,
    write_record,
)
from bayesian_phystwin_experiments.dlolab_slingshot_active_bayes import (
    ACTIVE_FRACTION,
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
    particle_worlds,
    prefix_observations,
    sample_worlds,
)
from bayesian_phystwin_experiments.dlolab_slingshot_belief_native import (
    run_registered_worlds,
)
from bayesian_phystwin_experiments.dlolab_slingshot_cmaes import worker_environment
from bayesian_phystwin_experiments.dlolab_slingshot_process import (
    load_native_bundle,
    runtime,
)
from bayesian_phystwin_experiments.dlolab_slingshot_task_probe_dev import (
    frontloaded_controls,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path("/home/fpfaff/source-only/dlolab-slingshot-active-bayes-source-v1")
ATTEMPT = Path(
    "/home/fpfaff/source-only/dlolab-slingshot-active-bayes-source-v1.attempt.json"
)
PARENT = Path(
    "/home/fpfaff/source-only/dlolab-benchmark-source-v1/belief-control-source-v1"
)
ACTIVE = Path(
    "/home/fpfaff/source-only/dlolab-slingshot-active-id-particle-source-v1"
)
PARENT_FILE_SHA256 = {
    "lock.json": "6dce35441588c2a5eff9c0ae08d85c8b41ff660403541dd489b8d9161bffcc8d",
    "model-bank/arrays.npz": "ef627e16490c0974d4c34fc82c16aae884fe6dd2a8dc0a80983e89b6d5e50832",
    "model-bank/seal.json": "f4a9331d552fe8f9715d222327c3f5c41cd7fc81a006e0f9a2fc55dd2223a3ae",
}
ACTIVE_FILE_SHA256 = {
    "lock.json": "b239ae43c443fce6cdf7910dfec20bb3410b88fbe49e69eec0bfd1a647cec989",
    "particle-bank/arrays.npz": "d2ec1f6fc9e8495a1eb99c20d2c8815868dadd7ce9f4449904fb3daf39d15e20",
    "particle-bank/seal.json": "3d8d8bf60e50de58de55ed99351545390f2222bc7a4278e2c1223f8cb3e87afc",
    "result.json": "12a28ebe74e9fac2743e0e8c02363e994d1b8dbf3e758ec81555d4f82e979724",
}
PARENT_LOCK_ID = "015e6d84aa68a2a4310552ef4880752b972890f02d3e09e333ff575c92b8df25"
PARENT_BANK_ID = "8ebf9c91322faf0658c84a2dcaa6895a98b1ff857e49e6714a2a2dad0c88d882"
ACTIVE_LOCK_ID = "3dde6f7ec8aed5a68f040f387eb54dfc11a117341c82a282213169abe20d50ed"
ACTIVE_BANK_ID = "17b96572a07a3d20818e19f3f31fec4afff98429aea8628f0872e70a3788c22a"
ACTIVE_RESULT_ID = "b202020e4e9e73a92b83a416a09d252394890b7ab02bcd188ac73889e92c3005"
NEW_SOURCES = (
    "src/bayesian_phystwin_experiments/dlolab_slingshot_active_bayes.py",
    "scripts/remote/run_dlolab_slingshot_active_bayes.py",
    "tests/test_dlolab_slingshot_active_bayes.py",
    "tests/test_dlolab_slingshot_active_bayes_custody.py",
    "docs/dlolab_slingshot_active_bayes_source_v1.md",
)
POSITION_FIELDS = ("rod_pos_m", "sphere_pos_m", "cube_pos_m", "gripper_pos_m")
Array: TypeAlias = NDArray[Any]


def _source() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Array],
    Array,
]:
    if any(
        file_digest(PARENT / name) != digest
        for name, digest in PARENT_FILE_SHA256.items()
    ) or any(
        file_digest(ACTIVE / name) != digest
        for name, digest in ACTIVE_FILE_SHA256.items()
    ):
        raise ValueError("registered parent evidence changed")
    parent_lock = read_record(PARENT / "lock.json")
    parent_seal = read_record(PARENT / "model-bank" / "seal.json")
    active_lock = read_record(ACTIVE / "lock.json")
    active_seal = read_record(ACTIVE / "particle-bank" / "seal.json")
    active_result = read_record(ACTIVE / "result.json")
    parent = load_native_bundle(PARENT / "model-bank", parent_seal["bundle"])
    active = load_native_bundle(ACTIVE / "particle-bank", active_seal["bundle"])
    if (
        parent_lock.get("artifact_id") != PARENT_LOCK_ID
        or parent_seal.get("artifact_id") != PARENT_BANK_ID
        or parent_seal.get("lock_id") != PARENT_LOCK_ID
        or active_lock.get("artifact_id") != ACTIVE_LOCK_ID
        or active_seal.get("artifact_id") != ACTIVE_BANK_ID
        or active_result.get("artifact_id") != ACTIVE_RESULT_ID
        or active_result.get("particle_bank_id") != ACTIVE_BANK_ID
        or active_result.get("particle_value_gate_passed") is not False
        or active_result.get("truth_probe_generated") is not False
        or active_result.get("truth_future_generated") is not False
        or active_result.get("retry_authorized") is not False
        or parent["prefix"].shape != (27, 3, 4, 3)
        or parent["reward"].shape != (27, 7)
        or active["history"].shape != (2, 27, 3, 4, 3)
        or active["reward"].shape != (27, 7)
        or not np.array_equal(active["history"][0], parent["prefix"])
        or not np.array_equal(active["reward"], parent["reward"])
    ):
        raise ValueError("registered stopped active-identification bank required")
    controls: Array = np.asarray(parent_lock["controls"], dtype=np.float64)
    if (
        controls.shape != (8, 3, 6)
        or not np.array_equal(controls[5], controls[7])
        or runtime() != parent_lock["screen"]["source"]["controller"]["runtime"]
    ):
        raise ValueError("registered public native runtime and actions required")
    old_worlds = {
        (row["x_offset_m"], row["bending_E"], row["stretching_K"])
        for role in ("calibration", "evaluation")
        for row in sample_worlds(role)
    }
    new_worlds = {
        (row["x_offset_m"], row["bending_E"], row["stretching_K"])
        for row in continuous_worlds()
    }
    particles = {
        (row["x_offset_m"], row["bending_E"], row["stretching_K"])
        for row in particle_worlds()
    }
    if (
        len(new_worlds) != WORLD_COUNT
        or new_worlds & old_worlds
        or new_worlds & particles
    ):
        raise ValueError("fresh continuous-world roster changed")
    return parent_lock, active_lock, active, controls


def _source_hashes(active_lock: dict[str, Any]) -> dict[str, str]:
    names = sorted(set(active_lock["source_sha256"]) | set(NEW_SOURCES))
    if any(not (ROOT / name).is_file() for name in names):
        raise ValueError("complete registered active-Bayes source required")
    return {name: file_digest(ROOT / name) for name in names}


def _validate(
    output: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Array], Array]:
    if output.resolve() != OUTPUT or output.is_symlink():
        raise ValueError("only the registered active-Bayes root is permitted")
    lock = read_record(output / "lock.json")
    attempt = read_record(ATTEMPT)
    parent_lock, active_lock, active, controls = _source()
    if (
        lock.get("schema") != "dlolab-slingshot-active-bayes-lock-v1"
        or lock.get("revision") != clean_revision(ROOT)
        or lock.get("source_sha256") != _source_hashes(active_lock)
        or lock.get("protocol") != protocol()
        or lock.get("output_root") != str(OUTPUT)
        or lock.get("attempt_id") != attempt.get("artifact_id")
        or attempt.get("schema") != "dlolab-slingshot-active-bayes-attempt-v1"
        or attempt.get("revision") != lock.get("revision")
        or attempt.get("source_sha256") != lock.get("source_sha256")
        or attempt.get("protocol") != lock.get("protocol")
        or attempt.get("output_root") != str(OUTPUT)
        or lock.get("parent_lock_id") != PARENT_LOCK_ID
        or lock.get("parent_bank_id") != PARENT_BANK_ID
        or lock.get("active_lock_id") != ACTIVE_LOCK_ID
        or lock.get("active_bank_id") != ACTIVE_BANK_ID
        or lock.get("active_result_id") != ACTIVE_RESULT_ID
        or lock.get("parent_file_sha256") != PARENT_FILE_SHA256
        or lock.get("active_file_sha256") != ACTIVE_FILE_SHA256
        or lock.get("controls_sha256") != array_digest(controls)
        or lock.get("histories_sha256") != array_digest(active["history"])
        or lock.get("reward_sha256") != array_digest(active["reward"])
        or any(record.get("retry_authorized") is not False for record in (lock, attempt))
        or any(
            record.get("protected_data_read") is not False
            for record in (lock, attempt)
        )
        or parent_lock.get("artifact_id") != PARENT_LOCK_ID
    ):
        raise ValueError("clean frozen active-Bayes lock required")
    return lock, parent_lock, active, controls


def _expected_prefix_controls(controls: Array, probe: int) -> Array:
    candidate = (
        controls
        if probe == 0
        else frontloaded_controls(controls, ACTIVE_FRACTION)
    )
    return np.asarray(np.repeat(candidate[5:6], 8, axis=0))


def _world_realization_matches(
    native: dict[str, Any], worlds: list[dict[str, Any]]
) -> bool:
    realized = native.get("world_realization", {})
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
    return bool(realized == expected)


def _load_prefix(
    output: Path,
    lock: dict[str, Any],
    controls: Array,
    probe: int,
    batch: int,
) -> tuple[dict[str, Any], dict[str, Array], bool]:
    task = prefix_task(probe, batch)
    directory = output / task["name"]
    claim = read_record(directory / "claim.json")
    seal = read_record(directory / "seal.json")
    native = seal.get("native", {})
    worlds = [continuous_worlds()[index] for index in task["world_indices"]]
    if (
        claim.get("schema") != "dlolab-slingshot-active-bayes-claim-v1"
        or claim.get("lock_id") != lock["artifact_id"]
        or claim.get("task") != task
        or claim.get("authorization") != {"gate": "prefix_only_before_outcomes"}
        or claim.get("retry_authorized") is not False
        or seal.get("schema") != "dlolab-slingshot-active-bayes-seal-v1"
        or seal.get("claim_id") != claim["artifact_id"]
        or seal.get("lock_id") != lock["artifact_id"]
        or seal.get("task") != task
        or native.get("native_steps") != 300
        or native.get("future_simulated") is not False
        or native.get("reward_scored") is not False
        or not _world_realization_matches(native, worlds)
    ):
        raise ValueError("invalid prefix-only active-Bayes artifact")
    data = load_native_bundle(directory, seal["bundle"])
    expected = _expected_prefix_controls(controls, probe)
    if (
        set(data) != set(TRACE_NAMES + ("controls",))
        or any(data[name].shape[:2] != (300, 8) for name in TRACE_NAMES)
        or array_digest(data["controls"]) != array_digest(expected)
    ):
        raise ValueError("active-Bayes prefix payload changed")
    fixed = float(
        np.max(
            np.abs(
                data["rod_pos_m"][:, :, [0, 1, 10, 11]]
                - data["rod_pos_m"][:1, :, [0, 1, 10, 11]]
            )
        )
    )
    return seal, data, bool(fixed <= 1e-9)


def _prefix_bank(
    output: Path,
    lock: dict[str, Any],
    controls: Array,
) -> tuple[Array, list[str], list[bool]]:
    truth: Array = np.empty((2, WORLD_COUNT, 3, 4, 3), dtype=np.float64)
    ids: list[str] = []
    qa: list[bool] = []
    for probe in range(2):
        for batch in range(4):
            seal, data, passed = _load_prefix(
                output, lock, controls, probe, batch
            )
            indices = prefix_task(probe, batch)["world_indices"]
            truth[probe, indices] = prefix_observations(data)
            ids.append(seal["artifact_id"])
            qa.append(passed)
    return truth, ids, qa


def _decision_contents(
    output: Path,
    lock: dict[str, Any],
    active: dict[str, Array],
    controls: Array,
) -> tuple[dict[str, Array], list[str], list[bool]]:
    truth, ids, qa = _prefix_bank(output, lock, controls)
    return infer_decisions(active["history"], active["reward"], truth), ids, qa


def _load_decisions(
    output: Path,
    lock: dict[str, Any],
    active: dict[str, Array],
    controls: Array,
) -> tuple[dict[str, Any], dict[str, Array], dict[str, Any]]:
    expected, prefix_ids, qa = _decision_contents(output, lock, active, controls)
    directory = output / "decisions"
    seal = read_record(directory / "seal.json")
    data = load_native_bundle(directory, seal["bundle"])
    if (
        seal.get("schema") != "dlolab-slingshot-active-bayes-decision-seal-v1"
        or seal.get("lock_id") != lock["artifact_id"]
        or seal.get("prefix_seal_ids") != prefix_ids
        or seal.get("particle_bank_id") != ACTIVE_BANK_ID
        or seal.get("future_read") is not False
        or seal.get("future_generated") is not False
        or set(data) != set(expected)
        or any(not np.array_equal(data[name], expected[name]) for name in expected)
    ):
        raise ValueError("sealed active-Bayes decisions changed")
    gate = pre_future_checks(data["decisions"], all_prefix_qa=all(qa))
    return seal, data, gate


def _barrier_contents(
    output: Path,
    lock: dict[str, Any],
    active: dict[str, Array],
    controls: Array,
) -> dict[str, Any]:
    seal, _, gate = _load_decisions(output, lock, active, controls)
    return {
        "schema": "dlolab-slingshot-active-bayes-decision-barrier-v1",
        "lock_id": lock["artifact_id"],
        "decision_seal_id": seal["artifact_id"],
        "pre_future": gate,
        "future_read": False,
        "future_generated": False,
    }


def _require_barrier(
    output: Path,
    lock: dict[str, Any],
    active: dict[str, Array],
    controls: Array,
) -> dict[str, Any]:
    barrier: dict[str, Any] = read_record(output / "decision-barrier.json")
    expected = _barrier_contents(output, lock, active, controls)
    if any(barrier.get(key) != value for key, value in expected.items()):
        raise ValueError("active-Bayes decision barrier changed")
    if barrier["pre_future"]["pre_future_gate_passed"] is not True:
        raise ValueError("pre-future active-Bayes gate did not pass")
    return barrier


def _worker(output: Path, kind: str, index: int, probe: int | None) -> None:
    lock, parent_lock, active, controls = _validate(output)
    if kind == "prefix":
        if probe is None:
            raise ValueError("prefix probe required")
        task = prefix_task(probe, index)
        authorization = {"gate": "prefix_only_before_outcomes"}
        native_controls = _expected_prefix_controls(controls, probe)
        worlds = [continuous_worlds()[i] for i in task["world_indices"]]
        prefix_only = True
    elif kind == "future":
        if probe is not None:
            raise ValueError("future task cannot specify a probe")
        task = future_task(index)
        barrier = _require_barrier(output, lock, active, controls)
        authorization = {
            "gate": "all_decisions_sealed",
            "barrier_id": barrier["artifact_id"],
        }
        native_controls = controls
        worlds = [continuous_worlds()[index]] * 8
        prefix_only = False
    else:
        raise ValueError("registered active-Bayes worker kind required")
    directory = output / task["name"]
    directory.mkdir()
    claim = write_record(
        directory / "claim.json",
        {
            "schema": "dlolab-slingshot-active-bayes-claim-v1",
            "lock_id": lock["artifact_id"],
            "task": task,
            "authorization": authorization,
            "retry_authorized": False,
        },
    )
    try:
        data, native = run_registered_worlds(
            Path(parent_lock["assets_root"]) / "upstream",
            directory,
            native_controls,
            worlds,
            prefix_only=prefix_only,
        )
        bundle = write_native_bundle(directory, data)
        write_record(
            directory / "seal.json",
            {
                "schema": "dlolab-slingshot-active-bayes-seal-v1",
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
                "schema": "dlolab-slingshot-active-bayes-failure-v1",
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


def _execute(
    output: Path,
    parent_lock: dict[str, Any],
    kind: str,
    index: int,
    probe: int | None = None,
) -> None:
    task = prefix_task(probe, index) if probe is not None else future_task(index)
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
    if probe is not None:
        command += ["--worker-probe", str(probe)]
    with (output / f"{task['name']}.log").open("x") as stream:
        run = subprocess.run(
            command,
            cwd=ROOT,
            env=worker_environment(
                parent_lock["screen"]["source"]["controller"]["runtime"]
            ),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if run.returncode:
        raise RuntimeError(f"{task['name']} exited {run.returncode}; no retry")


def _load_future(
    output: Path,
    lock: dict[str, Any],
    active: dict[str, Array],
    controls: Array,
    index: int,
) -> tuple[dict[str, Any], list[float], dict[str, Any]]:
    task = future_task(index)
    barrier = _require_barrier(output, lock, active, controls)
    directory = output / task["name"]
    claim = read_record(directory / "claim.json")
    seal = read_record(directory / "seal.json")
    worlds = [continuous_worlds()[index]] * 8
    if (
        claim.get("schema") != "dlolab-slingshot-active-bayes-claim-v1"
        or claim.get("lock_id") != lock["artifact_id"]
        or claim.get("task") != task
        or claim.get("authorization")
        != {"gate": "all_decisions_sealed", "barrier_id": barrier["artifact_id"]}
        or claim.get("retry_authorized") is not False
        or seal.get("schema") != "dlolab-slingshot-active-bayes-seal-v1"
        or seal.get("claim_id") != claim["artifact_id"]
        or seal.get("lock_id") != lock["artifact_id"]
        or seal.get("task") != task
        or seal.get("native", {}).get("native_steps") != 900
        or not _world_realization_matches(seal.get("native", {}), worlds)
    ):
        raise ValueError("future custody changed")
    data = load_native_bundle(directory, seal["bundle"])
    _, passive, _ = _load_prefix(output, lock, controls, 0, index // 8)
    prefix = {name: passive[name][:, index % 8] for name in POSITION_FIELDS}
    qa = native_qa(data, seal["native"], controls, prefix)
    rewards = [float(row["native_reward"]) for row in qa["metrics"][:7]]
    return seal, rewards, qa


def _run(output: Path) -> None:
    if output.resolve() != OUTPUT or output.exists() or ATTEMPT.exists():
        raise ValueError("one fresh active-Bayes attempt required")
    revision = clean_revision(ROOT)
    parent_lock, active_lock, active, controls = _source()
    sources = _source_hashes(active_lock)
    attempt = write_record(
        ATTEMPT,
        {
            "schema": "dlolab-slingshot-active-bayes-attempt-v1",
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
            "schema": "dlolab-slingshot-active-bayes-lock-v1",
            "revision": revision,
            "source_sha256": sources,
            "protocol": protocol(),
            "output_root": str(OUTPUT),
            "attempt_id": attempt["artifact_id"],
            "parent_lock_id": PARENT_LOCK_ID,
            "parent_bank_id": PARENT_BANK_ID,
            "active_lock_id": ACTIVE_LOCK_ID,
            "active_bank_id": ACTIVE_BANK_ID,
            "active_result_id": ACTIVE_RESULT_ID,
            "parent_file_sha256": PARENT_FILE_SHA256,
            "active_file_sha256": ACTIVE_FILE_SHA256,
            "controls_sha256": array_digest(controls),
            "histories_sha256": array_digest(active["history"]),
            "reward_sha256": array_digest(active["reward"]),
            "retry_authorized": False,
            "protected_data_read": False,
        },
    )
    stage = "prefixes"
    try:
        for probe in range(2):
            for batch in range(4):
                _execute(output, parent_lock, "prefix", batch, probe)
        decision_data, prefix_ids, qa = _decision_contents(
            output, lock, active, controls
        )
        directory = output / "decisions"
        directory.mkdir()
        decision_bundle = write_native_bundle(directory, decision_data)
        decision_seal = write_record(
            directory / "seal.json",
            {
                "schema": "dlolab-slingshot-active-bayes-decision-seal-v1",
                "lock_id": lock["artifact_id"],
                "prefix_seal_ids": prefix_ids,
                "particle_bank_id": ACTIVE_BANK_ID,
                "bundle": decision_bundle,
                "future_read": False,
                "future_generated": False,
            },
        )
        gate = pre_future_checks(
            decision_data["decisions"], all_prefix_qa=all(qa)
        )
        barrier = write_record(
            output / "decision-barrier.json",
            {
                "schema": "dlolab-slingshot-active-bayes-decision-barrier-v1",
                "lock_id": lock["artifact_id"],
                "decision_seal_id": decision_seal["artifact_id"],
                "pre_future": gate,
                "future_read": False,
                "future_generated": False,
            },
        )
        if not gate["pre_future_gate_passed"]:
            write_record(
                output / "result.json",
                {
                    "schema": "dlolab-slingshot-active-bayes-result-v1",
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
            _execute(output, parent_lock, "future", index)
        rewards: list[list[float]] = []
        future_ids: list[str] = []
        future_qa: list[dict[str, Any]] = []
        for index in range(WORLD_COUNT):
            seal, row, values = _load_future(
                output, lock, active, controls, index
            )
            future_ids.append(seal["artifact_id"])
            rewards.append(row)
            future_qa.append(values)
        reward_array = np.asarray(rewards, dtype=np.float64)
        generation_dir = output / "generation"
        generation_dir.mkdir()
        generation_bundle = write_native_bundle(
            generation_dir, {"reward": reward_array}
        )
        generation = write_record(
            generation_dir / "seal.json",
            {
                "schema": "dlolab-slingshot-active-bayes-generation-v1",
                "lock_id": lock["artifact_id"],
                "barrier_id": barrier["artifact_id"],
                "future_seal_ids": future_ids,
                "native_qa": future_qa,
                "bundle": generation_bundle,
                "ordinary_worlds": WORLD_COUNT,
                "technical_failures": 0,
                "replacements": 0,
            },
        )
        stage = "score"
        metrics = score(
            decision_data["decisions"],
            reward_array,
            all_native_qa=all(row["qa_passed"] for row in future_qa),
        )
        result = write_record(
            output / "result.json",
            {
                **metrics,
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
            f"active-Bayes source gate={result['source_gate_passed']}; "
            f"id={result['artifact_id']}",
            flush=True,
        )
    except Exception as error:
        write_record(
            output / "failure.json",
            {
                "schema": "dlolab-slingshot-active-bayes-failure-v1",
                "lock_id": lock["artifact_id"],
                "terminal_stage": stage,
                "completed_prefix_batches": sum(
                    (output / prefix_task(probe, batch)["name"] / "seal.json").is_file()
                    for probe in range(2)
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
    parser.add_argument("--worker-kind", choices=("prefix", "future"))
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--worker-probe", type=int)
    args = parser.parse_args()
    supplied = args.worker_kind is not None or args.worker_index is not None
    if supplied:
        if args.worker_kind is None or args.worker_index is None:
            raise ValueError("complete registered worker specification required")
        _worker(args.output, args.worker_kind, args.worker_index, args.worker_probe)
    else:
        if args.worker_probe is not None:
            raise ValueError("worker probe supplied without worker")
        _run(args.output)


if __name__ == "__main__":
    main()
