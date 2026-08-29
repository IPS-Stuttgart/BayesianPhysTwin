#!/usr/bin/env python3
"""Run the frozen source-only Slingshot exact-fallback guard study."""

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
from bayesian_phystwin_experiments.dlolab_slingshot_batch import TRACE_NAMES
from bayesian_phystwin_experiments.dlolab_slingshot_belief import (
    particle_worlds,
    prefix_observations,
    sample_worlds,
)
from bayesian_phystwin_experiments.dlolab_slingshot_belief_native import (
    run_registered_worlds,
)
from bayesian_phystwin_experiments.dlolab_slingshot_cmaes import worker_environment
from bayesian_phystwin_experiments.dlolab_slingshot_guard_source_v1 import (
    ACTIVE_FRACTION,
    WORLD_COUNT,
    infer_candidates,
    pre_outcome_checks,
    protocol,
    score,
)
from bayesian_phystwin_experiments.dlolab_slingshot_process import (
    load_native_bundle,
    runtime,
)
from bayesian_phystwin_experiments.dlolab_slingshot_task_probe_dev import (
    frontloaded_controls,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path("/home/fpfaff/source-only/dlolab-slingshot-guard-source-v1")
ATTEMPT = Path("/home/fpfaff/source-only/dlolab-slingshot-guard-source-v1.attempt.json")
PARENT = Path(
    "/home/fpfaff/source-only/dlolab-benchmark-source-v1/belief-control-source-v1"
)
ACTIVE = Path("/home/fpfaff/source-only/dlolab-slingshot-active-id-particle-source-v1")
PREFLIGHT = Path(
    "/home/fpfaff/source-only/dlolab-slingshot-active-bayes-runtime-preflight-v2"
)
PREFLIGHT_ATTEMPT = Path(
    "/home/fpfaff/source-only/dlolab-slingshot-active-bayes-runtime-preflight-v2.attempt.json"
)
V2_RESULT = ROOT / "results/sota/dlolab_slingshot_active_bayes_source_v2/summary.json"
EXPECTED_PYTHON = Path(
    "/home/fpfaff/source-only/dlolab-benchmark-source-v1-assets/venv/bin/python"
)
PARENT_FILE_SHA256 = {
    "lock.json": "6dce35441588c2a5eff9c0ae08d85c8b41ff660403541dd489b8d9161bffcc8d",
    "model-bank/arrays.npz": "ef627e16490c0974d4c34fc82c16aae884fe6dd2a8dc0a80983e89b6d5e50832",
    "model-bank/seal.json": "f4a9331d552fe8f9715d222327c3f5c41cd7fc81a006e0f9a2fc55dd2223a3ae",
    "calibration-prediction-barrier.json": "cface546bfe070a98fb5d7a2daec4ae73b5fd2c162a12a699691dd9365d44e8b",
    "calibrator.json": "26a00b934dd91b9c121242858756b7a44fa58d61163db53a3ebdebf229de6725",
}
ACTIVE_FILE_SHA256 = {
    "lock.json": "b239ae43c443fce6cdf7910dfec20bb3410b88fbe49e69eec0bfd1a647cec989",
    "particle-bank/arrays.npz": "d2ec1f6fc9e8495a1eb99c20d2c8815868dadd7ce9f4449904fb3daf39d15e20",
    "particle-bank/seal.json": "3d8d8bf60e50de58de55ed99351545390f2222bc7a4278e2c1223f8cb3e87afc",
    "result.json": "12a28ebe74e9fac2743e0e8c02363e994d1b8dbf3e758ec81555d4f82e979724",
}
PREFLIGHT_FILE_SHA256 = {
    "attempt.json": "e9d6156a3995fab8c91168e28a35ac8607d569f618721f83f1047264e803dde5",
    "lock.json": "0b17837daee8bd8e8101d6a447094cef4b53ea36a4556fa30aafaf850028d415",
    "claim.json": "1755ddca6bdc807db4529020a695d6f6dd52cb3819a12e481c29e5d7b07b5f3c",
    "arrays.npz": "706bfb031fd5dca15ddbd7525c2cab34c7358109fcf0ba671253ee01ab4be37c",
    "seal.json": "30410024eb7f5f1d00f639748361241bb3798e9bcc5b69b43c681b0e57cb3038",
    "result.json": "d1a16dc3b5586733e95f22bd25a2a4b1ebc22c1f94d85f7eb94ac4985d02a19a",
}
PARENT_LOCK_ID = "015e6d84aa68a2a4310552ef4880752b972890f02d3e09e333ff575c92b8df25"
PARENT_BANK_ID = "8ebf9c91322faf0658c84a2dcaa6895a98b1ff857e49e6714a2a2dad0c88d882"
ACTIVE_LOCK_ID = "3dde6f7ec8aed5a68f040f387eb54dfc11a117341c82a282213169abe20d50ed"
ACTIVE_BANK_ID = "17b96572a07a3d20818e19f3f31fec4afff98429aea8628f0872e70a3788c22a"
ACTIVE_RESULT_ID = "b202020e4e9e73a92b83a416a09d252394890b7ab02bcd188ac73889e92c3005"
PREFLIGHT_RESULT_ID = "53ab35dce6629dd1b2ab2b28e6756e14cd6100bc50bba46e45d02f71023aac25"
V2_SUMMARY_ID = "e38a4e963c77297b2469f2e923b8bfd2118da205eb0d886c157f3c92322352d7"
V2_SUMMARY_SHA256 = "a2a8857e32fd01fd8287d933551e111a1c9db05cf0001b87b64389fdd28f3ad2"
NEW_SOURCES = (
    "src/bayesian_phystwin_experiments/dlolab_slingshot_guard_source_v1.py",
    "scripts/remote/run_dlolab_slingshot_guard_source_v1.py",
    "tests/test_dlolab_slingshot_guard_source_v1.py",
    "tests/test_dlolab_slingshot_guard_source_v1_custody.py",
    "docs/dlolab_slingshot_guard_source_v1.md",
)
POSITION_FIELDS = ("rod_pos_m", "sphere_pos_m", "cube_pos_m", "gripper_pos_m")
Array: TypeAlias = NDArray[Any]


def _preflight_path(name: str) -> Path:
    return PREFLIGHT_ATTEMPT if name == "attempt.json" else PREFLIGHT / name


def prefix_task(batch: int) -> dict[str, Any]:
    if type(batch) is not int or batch not in range(3):
        raise ValueError("registered guard prefix batch required")
    indices = list(range(8 * batch, min(8 * batch + 8, WORLD_COUNT)))
    native_indices = indices + [indices[-1]] * (8 - len(indices))
    return {
        "kind": "active_prefix_only",
        "name": f"active-prefix-{batch}",
        "batch": batch,
        "world_indices": indices,
        "native_world_indices": native_indices,
    }


def _source() -> tuple[dict[str, Any], dict[str, Any], dict[str, Array], Array, list[dict[str, Any]]]:
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise ValueError("registered parent benchmark interpreter required")
    if any(
        file_digest(PARENT / name) != digest
        for name, digest in PARENT_FILE_SHA256.items()
    ) or any(
        file_digest(ACTIVE / name) != digest
        for name, digest in ACTIVE_FILE_SHA256.items()
    ) or any(
        file_digest(_preflight_path(name)) != digest
        for name, digest in PREFLIGHT_FILE_SHA256.items()
    ):
        raise ValueError("registered source or runtime evidence changed")
    if file_digest(V2_RESULT) != V2_SUMMARY_SHA256:
        raise ValueError("registered v2 negative result changed")
    parent_lock = read_record(PARENT / "lock.json")
    parent_seal = read_record(PARENT / "model-bank" / "seal.json")
    active_lock = read_record(ACTIVE / "lock.json")
    active_seal = read_record(ACTIVE / "particle-bank" / "seal.json")
    active_result = read_record(ACTIVE / "result.json")
    preflight_result = read_record(PREFLIGHT / "result.json")
    v2 = read_record(V2_RESULT)
    parent = load_native_bundle(PARENT / "model-bank", parent_seal["bundle"])
    active = load_native_bundle(ACTIVE / "particle-bank", active_seal["bundle"])
    controls = np.asarray(parent_lock["controls"], dtype=np.float64)
    worlds = sample_worlds("calibration")
    if (
        parent_lock.get("artifact_id") != PARENT_LOCK_ID
        or parent_seal.get("artifact_id") != PARENT_BANK_ID
        or active_lock.get("artifact_id") != ACTIVE_LOCK_ID
        or active_seal.get("artifact_id") != ACTIVE_BANK_ID
        or active_result.get("artifact_id") != ACTIVE_RESULT_ID
        or active_result.get("particle_value_gate_passed") is not False
        or preflight_result.get("artifact_id") != PREFLIGHT_RESULT_ID
        or preflight_result.get("runtime_preflight_passed") is not True
        or v2.get("artifact_id") != V2_SUMMARY_ID
        or v2.get("source_gate_passed") is not False
        or parent["prefix"].shape != (27, 3, 4, 3)
        or parent["reward"].shape != (27, 7)
        or active["history"].shape != (2, 27, 3, 4, 3)
        or active["reward"].shape != (27, 7)
        or not np.array_equal(active["history"][0], parent["prefix"])
        or not np.array_equal(active["reward"], parent["reward"])
        or controls.shape != (8, 3, 6)
        or not np.array_equal(controls[5], controls[7])
        or len(worlds) != WORLD_COUNT
        or runtime() != parent_lock["screen"]["source"]["controller"]["runtime"]
    ):
        raise ValueError("registered stopped active-identification source required")
    particle_keys = {
        (row["x_offset_m"], row["bending_E"], row["stretching_K"])
        for row in particle_worlds()
    }
    world_keys = {
        (row["x_offset_m"], row["bending_E"], row["stretching_K"])
        for row in worlds
    }
    if len(world_keys) != WORLD_COUNT or world_keys & particle_keys:
        raise ValueError("registered calibration-world roster changed")
    return parent_lock, active_lock, active, controls, worlds


def _source_hashes(active_lock: dict[str, Any]) -> dict[str, str]:
    names = sorted(set(active_lock["source_sha256"]) | set(NEW_SOURCES))
    if any(not (ROOT / name).is_file() for name in names):
        raise ValueError("complete registered guard source required")
    return {name: file_digest(ROOT / name) for name in names}


def _validate(output: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Array], Array, list[dict[str, Any]]]:
    if output.resolve() != OUTPUT or output.is_symlink():
        raise ValueError("only the registered guard root is permitted")
    lock = read_record(output / "lock.json")
    attempt = read_record(ATTEMPT)
    parent_lock, active_lock, active, controls, worlds = _source()
    if (
        lock.get("schema") != "dlolab-slingshot-guard-lock-v1"
        or lock.get("revision") != clean_revision(ROOT)
        or lock.get("source_sha256") != _source_hashes(active_lock)
        or lock.get("protocol") != protocol(worlds)
        or lock.get("output_root") != str(OUTPUT)
        or lock.get("attempt_id") != attempt.get("artifact_id")
        or attempt.get("schema") != "dlolab-slingshot-guard-attempt-v1"
        or attempt.get("revision") != lock.get("revision")
        or attempt.get("source_sha256") != lock.get("source_sha256")
        or attempt.get("protocol") != lock.get("protocol")
        or attempt.get("output_root") != str(OUTPUT)
        or lock.get("parent_file_sha256") != PARENT_FILE_SHA256
        or lock.get("active_file_sha256") != ACTIVE_FILE_SHA256
        or lock.get("preflight_file_sha256") != PREFLIGHT_FILE_SHA256
        or lock.get("v2_summary_sha256") != V2_SUMMARY_SHA256
        or lock.get("controls_sha256") != array_digest(controls)
        or lock.get("active_history_sha256") != array_digest(active["history"][1])
        or lock.get("model_reward_sha256") != array_digest(active["reward"])
        or any(record.get("retry_authorized") is not False for record in (lock, attempt))
        or any(record.get("protected_data_read") is not False for record in (lock, attempt))
        or parent_lock.get("artifact_id") != PARENT_LOCK_ID
    ):
        raise ValueError("clean frozen guard lock required")
    return lock, parent_lock, active, controls, worlds


def _expected_controls(controls: Array) -> Array:
    active = frontloaded_controls(controls, ACTIVE_FRACTION)
    return np.asarray(np.repeat(active[5:6], 8, axis=0))


def _world_realization_matches(native: dict[str, Any], worlds: list[dict[str, Any]]) -> bool:
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
    worlds: list[dict[str, Any]],
    batch: int,
) -> tuple[dict[str, Any], dict[str, Array], bool]:
    task = prefix_task(batch)
    directory = output / task["name"]
    claim = read_record(directory / "claim.json")
    seal = read_record(directory / "seal.json")
    native = seal.get("native", {})
    native_worlds = [worlds[index] for index in task["native_world_indices"]]
    if (
        claim.get("schema") != "dlolab-slingshot-guard-claim-v1"
        or claim.get("lock_id") != lock["artifact_id"]
        or claim.get("task") != task
        or claim.get("authorization") != {"gate": "prefix_only_before_source_outcomes"}
        or claim.get("retry_authorized") is not False
        or seal.get("schema") != "dlolab-slingshot-guard-seal-v1"
        or seal.get("claim_id") != claim["artifact_id"]
        or seal.get("lock_id") != lock["artifact_id"]
        or seal.get("task") != task
        or native.get("native_steps") != 300
        or native.get("future_simulated") is not False
        or native.get("reward_scored") is not False
        or not _world_realization_matches(native, native_worlds)
    ):
        raise ValueError("invalid prefix-only guard artifact")
    data = load_native_bundle(directory, seal["bundle"])
    if (
        set(data) != set(TRACE_NAMES + ("controls",))
        or any(data[name].shape[:2] != (300, 8) for name in TRACE_NAMES)
        or array_digest(data["controls"]) != array_digest(_expected_controls(controls))
    ):
        raise ValueError("guard prefix payload changed")
    fixed = float(
        np.max(
            np.abs(
                data["rod_pos_m"][:, :, [0, 1, 10, 11]]
                - data["rod_pos_m"][:1, :, [0, 1, 10, 11]]
            )
        )
    )
    return seal, data, bool(fixed <= 1e-9)


def _decision_contents(
    output: Path,
    lock: dict[str, Any],
    active: dict[str, Array],
    controls: Array,
    worlds: list[dict[str, Any]],
) -> tuple[dict[str, Array], list[str], list[bool]]:
    truth: Array = np.empty((WORLD_COUNT, 3, 4, 3), dtype=np.float64)
    ids: list[str] = []
    qa: list[bool] = []
    for batch in range(3):
        seal, data, passed = _load_prefix(output, lock, controls, worlds, batch)
        task = prefix_task(batch)
        count = len(task["world_indices"])
        truth[task["world_indices"]] = prefix_observations(data)[:count]
        ids.append(seal["artifact_id"])
        qa.append(passed)
    return infer_candidates(active["history"][1], active["reward"], truth), ids, qa


def _load_decisions(
    output: Path,
    lock: dict[str, Any],
    active: dict[str, Array],
    controls: Array,
    worlds: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Array], dict[str, Any]]:
    expected, prefix_ids, qa = _decision_contents(output, lock, active, controls, worlds)
    directory = output / "decisions"
    seal = read_record(directory / "seal.json")
    data = load_native_bundle(directory, seal["bundle"])
    if (
        seal.get("schema") != "dlolab-slingshot-guard-decision-seal-v1"
        or seal.get("lock_id") != lock["artifact_id"]
        or seal.get("prefix_seal_ids") != prefix_ids
        or seal.get("particle_bank_id") != ACTIVE_BANK_ID
        or seal.get("source_outcome_read") is not False
        or set(data) != set(expected)
        or any(not np.array_equal(data[name], expected[name]) for name in expected)
    ):
        raise ValueError("sealed guard decisions changed")
    return seal, data, pre_outcome_checks(data, all_prefix_qa=all(qa))


def _barrier_contents(
    output: Path,
    lock: dict[str, Any],
    active: dict[str, Array],
    controls: Array,
    worlds: list[dict[str, Any]],
) -> dict[str, Any]:
    seal, _, gate = _load_decisions(output, lock, active, controls, worlds)
    return {
        "schema": "dlolab-slingshot-guard-decision-barrier-v1",
        "lock_id": lock["artifact_id"],
        "decision_seal_id": seal["artifact_id"],
        "pre_outcome": gate,
        "source_outcome_read": False,
    }


def _require_barrier(
    output: Path,
    lock: dict[str, Any],
    active: dict[str, Array],
    controls: Array,
    worlds: list[dict[str, Any]],
) -> dict[str, Any]:
    recorded: dict[str, Any] = read_record(output / "decision-barrier.json")
    expected = _barrier_contents(output, lock, active, controls, worlds)
    if any(recorded.get(key) != value for key, value in expected.items()):
        raise ValueError("guard decision barrier changed")
    if recorded["pre_outcome"]["pre_outcome_gate_passed"] is not True:
        raise ValueError("guard pre-outcome gate did not pass")
    return recorded


def _load_parent_rewards(
    output: Path,
    lock: dict[str, Any],
    active: dict[str, Array],
    controls: Array,
    worlds: list[dict[str, Any]],
) -> tuple[Array, dict[str, Any], list[dict[str, Any]]]:
    _require_barrier(output, lock, active, controls, worlds)
    if file_digest(PARENT / "calibrator.json") != PARENT_FILE_SHA256["calibrator.json"]:
        raise ValueError("parent calibration outcome changed")
    record = read_record(PARENT / "calibrator.json")
    qas = record.get("native_qa", [])
    if (
        record.get("schema") != "dlolab-slingshot-belief-calibrator-v1"
        or record.get("lock_id") != PARENT_LOCK_ID
        or record.get("count") != WORLD_COUNT
        or record.get("evaluation_futures_read") is not False
        or len(record.get("future_seals", [])) != WORLD_COUNT
        or len(qas) != WORLD_COUNT
        or not all(row.get("qa_passed") is True for row in qas)
    ):
        raise ValueError("complete qualified parent calibration outcome required")
    rewards = np.asarray(
        [[metric["native_reward"] for metric in row["metrics"][:7]] for row in qas],
        dtype=np.float64,
    )
    if rewards.shape != (WORLD_COUNT, 7) or not np.isfinite(rewards).all():
        raise ValueError("complete finite parent calibration rewards required")
    return rewards, record, qas


def _worker(output: Path, batch: int) -> None:
    lock, parent_lock, _, controls, worlds = _validate(output)
    task = prefix_task(batch)
    native_worlds = [worlds[index] for index in task["native_world_indices"]]
    directory = output / task["name"]
    directory.mkdir()
    claim = write_record(
        directory / "claim.json",
        {
            "schema": "dlolab-slingshot-guard-claim-v1",
            "lock_id": lock["artifact_id"],
            "task": task,
            "authorization": {"gate": "prefix_only_before_source_outcomes"},
            "retry_authorized": False,
        },
    )
    try:
        data, native = run_registered_worlds(
            Path(parent_lock["assets_root"]) / "upstream",
            directory,
            _expected_controls(controls),
            native_worlds,
            prefix_only=True,
        )
        bundle = write_native_bundle(directory, data)
        write_record(
            directory / "seal.json",
            {
                "schema": "dlolab-slingshot-guard-seal-v1",
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
                "schema": "dlolab-slingshot-guard-failure-v1",
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


def _execute(output: Path, parent_lock: dict[str, Any], batch: int) -> None:
    task = prefix_task(batch)
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--output",
        str(output),
        "--worker-batch",
        str(batch),
    ]
    with (output / f"{task['name']}.log").open("x") as stream:
        run = subprocess.run(
            command,
            cwd=ROOT,
            env=worker_environment(parent_lock["screen"]["source"]["controller"]["runtime"]),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if run.returncode:
        raise RuntimeError(f"{task['name']} exited {run.returncode}; no retry")


def _run(output: Path) -> None:
    if (
        output.resolve() != OUTPUT
        or output.exists()
        or output.is_symlink()
        or ATTEMPT.exists()
        or ATTEMPT.is_symlink()
    ):
        raise ValueError("one fresh guard source attempt required")
    revision = clean_revision(ROOT)
    parent_lock, active_lock, active, controls, worlds = _source()
    sources = _source_hashes(active_lock)
    attempt = write_record(
        ATTEMPT,
        {
            "schema": "dlolab-slingshot-guard-attempt-v1",
            "revision": revision,
            "source_sha256": sources,
            "protocol": protocol(worlds),
            "output_root": str(OUTPUT),
            "retry_authorized": False,
            "protected_data_read": False,
        },
    )
    output.mkdir()
    lock = write_record(
        output / "lock.json",
        {
            "schema": "dlolab-slingshot-guard-lock-v1",
            "revision": revision,
            "source_sha256": sources,
            "protocol": protocol(worlds),
            "output_root": str(OUTPUT),
            "attempt_id": attempt["artifact_id"],
            "parent_lock_id": PARENT_LOCK_ID,
            "parent_bank_id": PARENT_BANK_ID,
            "active_lock_id": ACTIVE_LOCK_ID,
            "active_bank_id": ACTIVE_BANK_ID,
            "active_result_id": ACTIVE_RESULT_ID,
            "preflight_result_id": PREFLIGHT_RESULT_ID,
            "v2_summary_id": V2_SUMMARY_ID,
            "parent_file_sha256": PARENT_FILE_SHA256,
            "active_file_sha256": ACTIVE_FILE_SHA256,
            "preflight_file_sha256": PREFLIGHT_FILE_SHA256,
            "v2_summary_sha256": V2_SUMMARY_SHA256,
            "controls_sha256": array_digest(controls),
            "active_history_sha256": array_digest(active["history"][1]),
            "model_reward_sha256": array_digest(active["reward"]),
            "expected_python": str(EXPECTED_PYTHON),
            "parent_calibration_reward_read": False,
            "retry_authorized": False,
            "protected_data_read": False,
        },
    )
    stage = "active_prefixes"
    source_outcome_read = False
    try:
        for batch in range(3):
            _execute(output, parent_lock, batch)
        decision_data, prefix_ids, qa = _decision_contents(output, lock, active, controls, worlds)
        decision_dir = output / "decisions"
        decision_dir.mkdir()
        decision_bundle = write_native_bundle(decision_dir, decision_data)
        decision_seal = write_record(
            decision_dir / "seal.json",
            {
                "schema": "dlolab-slingshot-guard-decision-seal-v1",
                "lock_id": lock["artifact_id"],
                "prefix_seal_ids": prefix_ids,
                "particle_bank_id": ACTIVE_BANK_ID,
                "bundle": decision_bundle,
                "source_outcome_read": False,
            },
        )
        gate = pre_outcome_checks(decision_data, all_prefix_qa=all(qa))
        barrier = write_record(
            output / "decision-barrier.json",
            {
                "schema": "dlolab-slingshot-guard-decision-barrier-v1",
                "lock_id": lock["artifact_id"],
                "decision_seal_id": decision_seal["artifact_id"],
                "pre_outcome": gate,
                "source_outcome_read": False,
            },
        )
        if not gate["pre_outcome_gate_passed"]:
            write_record(
                output / "result.json",
                {
                    "schema": "dlolab-slingshot-guard-result-v1",
                    "status": "pre_outcome_gate_failed",
                    "lock_id": lock["artifact_id"],
                    "decision_seal_id": decision_seal["artifact_id"],
                    "barrier_id": barrier["artifact_id"],
                    "pre_outcome": gate,
                    "parent_calibration_reward_read": False,
                    "source_gate_passed": False,
                    "fresh_world_automatically_authorized": False,
                    "retry_authorized": False,
                    "protected_data_read": False,
                },
            )
            return
        stage = "parent_calibration_outcome"
        reward, parent_calibrator, parent_qa = _load_parent_rewards(
            output, lock, active, controls, worlds
        )
        source_outcome_read = True
        generation_dir = output / "generation"
        generation_dir.mkdir()
        generation_bundle = write_native_bundle(generation_dir, {"reward": reward})
        generation = write_record(
            generation_dir / "seal.json",
            {
                "schema": "dlolab-slingshot-guard-generation-v1",
                "lock_id": lock["artifact_id"],
                "barrier_id": barrier["artifact_id"],
                "parent_calibrator_id": parent_calibrator["artifact_id"],
                "parent_future_seal_ids": parent_calibrator["future_seals"],
                "bundle": generation_bundle,
                "ordinary_worlds": WORLD_COUNT,
                "technical_failures": 0,
                "replacements": 0,
            },
        )
        stage = "score"
        metrics = score(
            decision_data,
            reward,
            all_native_qa=all(row["qa_passed"] for row in parent_qa),
        )
        result = write_record(
            output / "result.json",
            {
                **metrics,
                "lock_id": lock["artifact_id"],
                "decision_seal_id": decision_seal["artifact_id"],
                "barrier_id": barrier["artifact_id"],
                "generation_id": generation["artifact_id"],
                "parent_calibrator_id": parent_calibrator["artifact_id"],
                "pre_outcome": gate,
                "parent_calibration_reward_read": True,
                "retry_authorized": False,
            },
        )
        print(
            f"guard source gate={result['source_gate_passed']}; id={result['artifact_id']}",
            flush=True,
        )
    except Exception as error:
        write_record(
            output / "failure.json",
            {
                "schema": "dlolab-slingshot-guard-failure-v1",
                "lock_id": lock["artifact_id"],
                "terminal_stage": stage,
                "completed_prefix_batches": sum(
                    (output / prefix_task(batch)["name"] / "seal.json").is_file()
                    for batch in range(3)
                ),
                "decision_barrier_written": (output / "decision-barrier.json").is_file(),
                "parent_calibration_reward_read": source_outcome_read,
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
    parser.add_argument("--worker-batch", type=int)
    args = parser.parse_args()
    if args.worker_batch is None:
        _run(args.output)
    else:
        _worker(args.output, args.worker_batch)


if __name__ == "__main__":
    main()
