#!/usr/bin/env python3
"""Qualify one-world/one-action Slingshot execution in fresh processes."""

from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from bayesian_phystwin_experiments.dlolab_benchmark import write_native_bundle
from bayesian_phystwin_experiments.dlolab_native import array_digest, file_digest
from bayesian_phystwin_experiments.dlolab_regret_artifacts import (
    clean_revision,
    read_record,
    write_record,
)
from bayesian_phystwin_experiments.dlolab_slingshot_cmaes import (
    task_metrics,
    worker_environment,
)
from bayesian_phystwin_experiments.dlolab_slingshot_independent_native_v3 import (
    ACTION_COUNT,
    PROCESS_COUNT,
    WORLD_COUNT,
    independent_world_qa,
    protocol,
    qualification_worlds,
    run_registered_world,
    task,
    validate_singleton_arrays,
    validate_world_realization,
)
from bayesian_phystwin_experiments.dlolab_slingshot_process import (
    load_native_bundle,
    runtime,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(
    "/home/fpfaff/source-only/dlolab-slingshot-independent-native-qualification-v1"
)
ATTEMPT = Path(
    "/home/fpfaff/source-only/"
    "dlolab-slingshot-independent-native-qualification-v1.attempt.json"
)
PARENT_ROOT = Path(
    "/home/fpfaff/source-only/dlolab-slingshot-policy-certificate-source-v2"
)
PARENT_SUMMARY = (
    ROOT / "results/source/dlolab_slingshot_policy_certificate_source_v2/summary.json"
)
PARENT_LOCK_ID = "d29448148194279e739c1c5b0127f13871ac558eecab8233118058d534f7c7dc"
PARENT_LOCK_SHA256 = "65276bab350a7f7902f14a030100f69c1c5bc65a6f85fce0f4a4f13584a2e00f"
PARENT_SUMMARY_ID = "e00de6c8b7a82fa13ee1076e05e52ac1ea472de6a042e395a8f3490e8703c016"
PARENT_SUMMARY_SHA256 = (
    "9a7d2da87fb49f92162a263d3a77c7767433d83cd3bc41b197467abe13a87c5e"
)
PARENT_FAILURE_SHA256 = (
    "04433a4926404f9ae40fcbb888b516a4fa18991ddf1df63d248a190f2641de87"
)
CONTROLS_SHA256 = "af88f1c5299c2daf67e06325a2c2487dee9d007315cfce00944e491f3b411203"
WORKERS = 8
Array: TypeAlias = NDArray[Any]
SOURCES = (
    "src/bayesian_phystwin_experiments/dlolab_slingshot_independent_native_v3.py",
    "scripts/remote/run_dlolab_slingshot_independent_native_qualification_v1.py",
    "tests/test_dlolab_slingshot_independent_native_v3.py",
    "tests/test_dlolab_slingshot_independent_native_qualification_v1.py",
    "docs/dlolab_slingshot_independent_native_qualification_v1.md",
    "results/source/dlolab_slingshot_policy_certificate_source_v2/summary.json",
    "src/bayesian_phystwin_experiments/dlolab_slingshot_belief.py",
    "src/bayesian_phystwin_experiments/dlolab_slingshot_batch.py",
    "src/bayesian_phystwin_experiments/dlolab_slingshot_process.py",
    "src/bayesian_phystwin_experiments/dlolab_benchmark.py",
    "src/bayesian_phystwin_experiments/dlolab_regret_artifacts.py",
)


def load_parent() -> tuple[dict[str, Any], Array]:
    lock_path = PARENT_ROOT / "lock.json"
    failure_path = PARENT_ROOT / "failure.json"
    if (
        lock_path.is_symlink()
        or failure_path.is_symlink()
        or PARENT_SUMMARY.is_symlink()
        or file_digest(lock_path) != PARENT_LOCK_SHA256
        or file_digest(failure_path) != PARENT_FAILURE_SHA256
        or file_digest(PARENT_SUMMARY) != PARENT_SUMMARY_SHA256
    ):
        raise ValueError("terminal v2 source evidence changed")
    lock = read_record(lock_path)
    summary = read_record(PARENT_SUMMARY)
    controls = np.asarray(lock.get("controls"), dtype=np.float64)
    if (
        lock.get("artifact_id") != PARENT_LOCK_ID
        or summary.get("artifact_id") != PARENT_SUMMARY_ID
        or summary.get("status") != "retained_evaluation_native_qa_failure"
        or summary.get("ordinary_evaluation_futures") != 286
        or summary.get("technical_failures") != 2
        or summary.get("complete_288_world_denominator_scored") is not False
        or summary.get("retry_authorized") is not False
        or summary.get("replacement_authorized") is not False
        or controls.shape != (ACTION_COUNT, 3, 6)
        or controls.dtype != np.float64
        or not np.isfinite(controls).all()
        or array_digest(controls) != CONTROLS_SHA256
    ):
        raise ValueError("exact terminal v2 source lineage required")
    return cast(dict[str, Any], lock), controls


def _source_hashes() -> dict[str, str]:
    if any(not (ROOT / name).is_file() for name in SOURCES):
        raise ValueError("complete independent-native qualification source required")
    return {name: file_digest(ROOT / name) for name in SOURCES}


def freeze(output: Path) -> dict[str, Any]:
    if output.resolve() != OUTPUT or output.exists() or ATTEMPT.exists():
        raise ValueError("only the fresh registered one-attempt root is authorized")
    revision = clean_revision(ROOT)
    parent, controls = load_parent()
    if runtime() != parent.get("runtime"):
        raise ValueError("exact parent-qualified runtime required")
    output.mkdir(parents=True, exist_ok=False)
    lock = write_record(
        output / "lock.json",
        {
            "schema": "dlolab-slingshot-independent-native-lock-v1",
            "source_revision": revision,
            "source_sha256": _source_hashes(),
            "protocol": protocol(),
            "output_root": str(output.resolve()),
            "attempt_ledger": str(ATTEMPT.resolve()),
            "parent_root": str(PARENT_ROOT.resolve()),
            "parent_lock_id": PARENT_LOCK_ID,
            "parent_summary_id": PARENT_SUMMARY_ID,
            "assets_root": parent["assets_root"],
            "runtime": parent["runtime"],
            "controls": controls.tolist(),
            "controls_sha256": CONTROLS_SHA256,
            "worker_count": WORKERS,
            "protected_data_read": False,
        },
    )
    write_record(
        ATTEMPT,
        {
            "schema": "dlolab-slingshot-independent-native-attempt-v1",
            "lock_id": lock["artifact_id"],
            "source_revision": revision,
            "output_root": str(output.resolve()),
            "attempt_number": 1,
            "retry_authorized": False,
            "replacement_authorized": False,
            "protected_data_read": False,
        },
    )
    return cast(dict[str, Any], lock)


def validate_lock(output: Path) -> dict[str, Any]:
    if output.resolve() != OUTPUT:
        raise ValueError("only the registered qualification root is authorized")
    lock = read_record(output / "lock.json")
    attempt = read_record(ATTEMPT)
    parent, controls = load_parent()
    if (
        lock.get("schema") != "dlolab-slingshot-independent-native-lock-v1"
        or lock.get("source_revision") != clean_revision(ROOT)
        or lock.get("source_sha256") != _source_hashes()
        or lock.get("protocol") != protocol()
        or lock.get("output_root") != str(output.resolve())
        or lock.get("attempt_ledger") != str(ATTEMPT.resolve())
        or lock.get("parent_root") != str(PARENT_ROOT.resolve())
        or lock.get("parent_lock_id") != PARENT_LOCK_ID
        or lock.get("parent_summary_id") != PARENT_SUMMARY_ID
        or lock.get("assets_root") != parent["assets_root"]
        or lock.get("runtime") != runtime()
        or lock.get("controls") != controls.tolist()
        or lock.get("controls_sha256") != CONTROLS_SHA256
        or lock.get("worker_count") != WORKERS
        or attempt.get("schema") != "dlolab-slingshot-independent-native-attempt-v1"
        or attempt.get("lock_id") != lock.get("artifact_id")
        or attempt.get("source_revision") != lock.get("source_revision")
        or attempt.get("output_root") != str(output.resolve())
        or attempt.get("attempt_number") != 1
        or attempt.get("retry_authorized") is not False
    ):
        raise ValueError("frozen independent-native qualification changed")
    return cast(dict[str, Any], lock)


def _single_task_qa(
    arrays: dict[str, Array],
    native: dict[str, Any],
    expected_control: Array,
    world: dict[str, Any],
) -> dict[str, Any]:
    validate_world_realization(native, world)
    validate_singleton_arrays(arrays)
    if (
        array_digest(arrays["controls"]) != array_digest(expected_control)
        or native.get("native_steps") != 900
        or native.get("environment_count") != 1
        or native.get("fresh_python_process") is not True
        or native.get("world") != world
        or native.get("native_cumulative_reward")
        != [task_metrics(arrays)["native_reward"]]
    ):
        raise ValueError("independent native task arithmetic changed")
    fixed = float(
        np.max(
            np.abs(
                arrays["rod_pos_m"][:, :, [0, 1, 10, 11]]
                - arrays["rod_pos_m"][:1, :, [0, 1, 10, 11]]
            )
        )
    )
    return {
        "checks": {
            "complete_finite_singleton": True,
            "exact_control": True,
            "exact_world_realization": True,
            "exact_reward_arithmetic": True,
            "fixed_endpoints": fixed <= 1e-9,
        },
        "fixed_endpoint_error_m": fixed,
        "qa_passed": fixed <= 1e-9,
    }


def worker(output: Path, index: int) -> None:
    lock = validate_lock(output)
    spec = task(index)
    world = qualification_worlds()[spec["world_index"]]
    controls = np.asarray(lock["controls"], dtype=np.float64)
    command = controls[spec["action_index"] : spec["action_index"] + 1]
    directory = output / spec["name"]
    directory.mkdir(exist_ok=False)
    claim = write_record(
        directory / "claim.json",
        {
            "schema": "dlolab-slingshot-independent-native-claim-v1",
            "lock_id": lock["artifact_id"],
            "task": spec,
            "world": world,
            "control_sha256": array_digest(command),
            "retry_authorized": False,
            "replacement_authorized": False,
            "protected_data_read": False,
        },
    )
    try:
        arrays, native = run_registered_world(
            Path(lock["assets_root"]) / "upstream", directory, command, world
        )
        qa = _single_task_qa(arrays, native, command, world)
        if not qa["qa_passed"]:
            raise ValueError("single independent native task QA failed")
        write_record(
            directory / "seal.json",
            {
                "schema": "dlolab-slingshot-independent-native-seal-v1",
                "lock_id": lock["artifact_id"],
                "claim_id": claim["artifact_id"],
                "task": spec,
                "world": world,
                "native": native,
                "qa": qa,
                "bundle": write_native_bundle(directory, arrays),
            },
        )
    except Exception as error:
        write_record(
            directory / "failure.json",
            {
                "schema": "dlolab-slingshot-independent-native-task-failure-v1",
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


def load_task(
    output: Path, lock: dict[str, Any], index: int
) -> tuple[dict[str, Any], dict[str, Array]]:
    spec = task(index)
    world = qualification_worlds()[spec["world_index"]]
    directory = output / spec["name"]
    claim = read_record(directory / "claim.json")
    seal = read_record(directory / "seal.json")
    arrays = load_native_bundle(directory, seal["bundle"])
    controls = np.asarray(lock["controls"], dtype=np.float64)
    command = controls[spec["action_index"] : spec["action_index"] + 1]
    qa = _single_task_qa(arrays, seal["native"], command, world)
    if (
        claim.get("schema") != "dlolab-slingshot-independent-native-claim-v1"
        or seal.get("schema") != "dlolab-slingshot-independent-native-seal-v1"
        or claim.get("lock_id") != lock["artifact_id"]
        or seal.get("lock_id") != lock["artifact_id"]
        or claim.get("task") != spec
        or seal.get("task") != spec
        or claim.get("world") != world
        or seal.get("world") != world
        or claim.get("control_sha256") != array_digest(command)
        or seal.get("claim_id") != claim["artifact_id"]
        or claim.get("retry_authorized") is not False
        or claim.get("replacement_authorized") is not False
        or seal.get("qa") != qa
        or not qa["qa_passed"]
    ):
        raise ValueError("independent native task custody changed")
    return cast(dict[str, Any], seal), arrays


def validate_task_failure(output: Path, lock: dict[str, Any], index: int) -> None:
    spec = task(index)
    world = qualification_worlds()[spec["world_index"]]
    controls = np.asarray(lock["controls"], dtype=np.float64)
    command = controls[spec["action_index"] : spec["action_index"] + 1]
    directory = output / spec["name"]
    claim = read_record(directory / "claim.json")
    failure = read_record(directory / "failure.json")
    if (
        (directory / "seal.json").exists()
        or claim.get("schema") != "dlolab-slingshot-independent-native-claim-v1"
        or failure.get("schema")
        != "dlolab-slingshot-independent-native-task-failure-v1"
        or claim.get("lock_id") != lock["artifact_id"]
        or failure.get("lock_id") != lock["artifact_id"]
        or claim.get("task") != spec
        or failure.get("task") != spec
        or claim.get("world") != world
        or claim.get("control_sha256") != array_digest(command)
        or failure.get("claim_id") != claim["artifact_id"]
        or failure.get("retry_authorized") is not False
        or failure.get("replacement_authorized") is not False
        or failure.get("protected_data_read") is not False
        or not isinstance(failure.get("error_type"), str)
        or not isinstance(failure.get("message"), str)
    ):
        raise ValueError("independent native task failure custody changed")


def execute(output: Path, lock: dict[str, Any], index: int) -> int:
    spec = task(index)
    with (output / f"{spec['name']}.log").open("x") as stream:
        completed = subprocess.run(
            [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "--output",
                str(output.resolve()),
                "--worker",
                str(index),
            ],
            cwd=ROOT,
            env=worker_environment(lock["runtime"]),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(completed.returncode)


def run(output: Path) -> dict[str, Any]:
    lock = freeze(output)
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = [
            executor.submit(execute, output, lock, index)
            for index in range(PROCESS_COUNT)
        ]
        returncodes = [future.result() for future in futures]
    ordinary: list[int] = []
    failed: list[int] = []
    custody_errors: dict[str, str] = {}
    for index, code in enumerate(returncodes):
        try:
            if code == 0:
                load_task(output, lock, index)
                ordinary.append(index)
            else:
                validate_task_failure(output, lock, index)
                failed.append(index)
        except Exception as error:
            failed.append(index)
            custody_errors[str(index)] = f"{type(error).__name__}: {error}"
    world_qa_ids: list[str] = []
    qualified = 0
    if not failed:
        controls = np.asarray(lock["controls"], dtype=np.float64)
        for world_index, world in enumerate(qualification_worlds()):
            rows: list[dict[str, Array]] = []
            reports: list[dict[str, Any]] = []
            seal_ids: list[str] = []
            for action_index in range(ACTION_COUNT):
                index = world_index * ACTION_COUNT + action_index
                seal, arrays = load_task(output, lock, index)
                rows.append(arrays)
                reports.append(seal["native"])
                seal_ids.append(seal["artifact_id"])
            qa = independent_world_qa(rows, reports, controls, world)
            record = write_record(
                output / f"world-{world_index:02d}-qualification.json",
                {
                    "schema": "dlolab-slingshot-independent-world-qualification-v1",
                    "lock_id": lock["artifact_id"],
                    "world": world,
                    "source_seal_ids": seal_ids,
                    "qa": qa,
                    "protected_data_read": False,
                },
            )
            world_qa_ids.append(record["artifact_id"])
            qualified += int(qa["qa_passed"])
    passed = len(ordinary) == PROCESS_COUNT and qualified == WORLD_COUNT
    result = write_record(
        output / "result.json",
        {
            "schema": "dlolab-slingshot-independent-native-result-v1",
            "lock_id": lock["artifact_id"],
            "status": "passed" if passed else "terminal_qualification_failure",
            "planned_processes": PROCESS_COUNT,
            "ordinary_processes": len(ordinary),
            "failed_process_indices": failed,
            "custody_validation_errors": custody_errors,
            "qualified_worlds": qualified,
            "world_qualification_ids": world_qa_ids,
            "qualification_passed": passed,
            "v3_protocol_freeze_authorized": passed,
            "v3_scientific_execution_authorized": False,
            "retry_authorized": False,
            "replacement_authorized": False,
            "scientific_policy_value_scored": False,
            "protected_data_read": False,
            "held_v8_read": False,
            "dlo4_dlo5_read": False,
            "new_recordings": False,
        },
    )
    print(
        f"independent native qualification={passed}; artifact={result['artifact_id']}",
        flush=True,
    )
    return cast(dict[str, Any], result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--worker", type=int, choices=range(PROCESS_COUNT))
    args = parser.parse_args()
    if args.worker is None:
        run(args.output)
    else:
        worker(args.output, args.worker)
