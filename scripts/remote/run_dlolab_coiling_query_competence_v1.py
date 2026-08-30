#!/usr/bin/env python3
"""Run the frozen public DLO-Lab coiling development screen once."""

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

from bayesian_phystwin._portable_contracts import load_strict_json_object
from bayesian_phystwin_experiments.deform_state_restart import file_digest
from bayesian_phystwin_experiments.dlolab_benchmark import (
    source_identity,
    write_native_bundle,
)
from bayesian_phystwin_experiments.dlolab_coiling_native_v1 import run_world
from bayesian_phystwin_experiments.dlolab_coiling_query_competence_v1 import (
    PARENT_FAILURE_ID,
    PARENT_LOCK_ID,
    PARENT_TASK_FAILURE_ID,
    native_qa,
    protocol_v1_1,
    source_value,
    task,
    worlds,
)
from bayesian_phystwin_experiments.dlolab_regret_artifacts import (
    clean_revision,
    read_record,
    runtime_identity,
    write_record,
)
from bayesian_phystwin_experiments.dlolab_slingshot_process import (
    load_native_bundle,
)

ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path("/home/fpfaff/source-only/dlolab-benchmark-source-v1-assets")
OUTPUT = Path(
    "/home/fpfaff/source-only/dlolab-coiling-query-competence-development-v1-1"
)
PARENT_SUMMARY = (
    ROOT / "results/source/dlolab_coiling_query_competence_development_v1/summary.json"
)
PARENT_SUMMARY_SHA256 = (
    "c89ce240fe6931f0f12540ec2b91ddd00a6843ad16ffc0177340195c7c3dce05"
)
SOURCES = (
    "src/bayesian_phystwin_experiments/dlolab_coiling_query_competence_v1.py",
    "src/bayesian_phystwin_experiments/dlolab_coiling_native_v1.py",
    "scripts/remote/run_dlolab_coiling_query_competence_v1.py",
    "tests/test_dlolab_coiling_query_competence_v1.py",
    "tests/test_dlolab_coiling_runner_v1.py",
    "docs/dlolab_coiling_query_competence_development_v1.md",
    "docs/dlolab_coiling_query_competence_development_v1_result.md",
    "docs/dlolab_coiling_query_competence_development_v1_1.md",
    "results/source/dlolab_coiling_query_competence_development_v1/summary.json",
    "src/bayesian_phystwin_experiments/dlolab_benchmark.py",
    "src/bayesian_phystwin_experiments/dlolab_native.py",
    "src/bayesian_phystwin_experiments/dlolab_regret_artifacts.py",
    "src/bayesian_phystwin_experiments/dlolab_slingshot_process.py",
    "src/bayesian_phystwin_experiments/deform_state_restart.py",
    "src/bayesian_phystwin/_portable_contracts.py",
    "src/bayesian_phystwin/_canonical_contracts.py",
)

Array: TypeAlias = NDArray[Any]


def parent_failure() -> dict[str, Any]:
    value = dict(
        load_strict_json_object(PARENT_SUMMARY, label="coiling parent failure")
    )
    if (
        file_digest(PARENT_SUMMARY) != PARENT_SUMMARY_SHA256
        or value.get("lock_id") != PARENT_LOCK_ID
        or value.get("failure_id") != PARENT_FAILURE_ID
        or value.get("first_task_failure_id") != PARENT_TASK_FAILURE_ID
        or value.get("native_scene_steps_completed") != 0
        or value.get("native_rewards_generated") is not False
        or value.get("value_analysis_executed") is not False
        or value.get("parent_root_retry_authorized") is not False
        or value.get("replacement_may_change_scientific_fields") is not False
    ):
        raise ValueError("exact zero-step parent failure required")
    return value


def runtime() -> dict[str, Any]:
    value = runtime_identity()
    value["benchmark_packages"] = {
        name: importlib.metadata.version(name)
        for name in (
            "pin",
            "pin-pink",
            "qpsolvers",
            "proxsuite",
            "quadprog",
            "mushroom-rl",
            "omegaconf",
        )
    }
    return cast(dict[str, Any], value)


def source() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        source_identity(
            ASSETS / "upstream", ASSETS / "mushroom-rl", ASSETS / "dlo-lab.zip"
        ),
    )


def validate_lock(output: Path) -> dict[str, Any]:
    if output.resolve() != OUTPUT or output.is_symlink():
        raise ValueError("only the registered write-once coiling root is permitted")
    if (output / "failure.json").exists() or (output / "result.json").exists():
        raise ValueError("terminal coiling development study; no retry")
    lock = read_record(output / "lock.json")
    expected_source = {path: file_digest(ROOT / path) for path in SOURCES}
    if (
        lock.get("schema") != "dlolab-coiling-development-lock-v1-1"
        or lock.get("revision") != clean_revision(ROOT)
        or lock.get("protocol") != protocol_v1_1()
        or lock.get("parent_failure") != parent_failure()
        or lock.get("output_root") != str(OUTPUT)
        or lock.get("runtime") != runtime()
        or lock.get("native_source") != source()
        or lock.get("source_sha256") != expected_source
    ):
        raise ValueError("clean frozen coiling implementation/runtime/source required")
    return cast(dict[str, Any], lock)


def load_task(
    output: Path, lock: dict[str, Any], index: int
) -> tuple[dict[str, Any], dict[str, Array], dict[str, Any]]:
    spec = task(index)
    directory = output / spec["name"]
    claim = read_record(directory / "claim.json")
    seal = read_record(directory / "seal.json")
    if (
        claim.get("lock_id") != lock["artifact_id"]
        or seal.get("lock_id") != lock["artifact_id"]
        or seal.get("claim_id") != claim["artifact_id"]
        or claim.get("task") != spec
        or seal.get("task") != spec
    ):
        raise ValueError("native coiling task custody changed")
    arrays = load_native_bundle(directory, seal["bundle"])
    qa = native_qa(arrays, seal["native"], spec["world"])
    return seal, arrays, qa


def worker(output: Path, index: int) -> None:
    lock = validate_lock(output)
    spec = task(index)
    for previous in range(index):
        seal, _, qa = load_task(output, lock, previous)
        receipt = read_record(output / task(previous)["name"] / "qa.json")
        if (
            not qa["passed"]
            or receipt.get("lock_id") != lock["artifact_id"]
            or receipt.get("seal_id") != seal["artifact_id"]
            or receipt.get("qa") != qa
        ):
            raise ValueError("all earlier coiling worlds must requalify")
    directory = output / spec["name"]
    directory.mkdir(exist_ok=False)
    claim = write_record(
        directory / "claim.json",
        {"lock_id": lock["artifact_id"], "task": spec, "retry_authorized": False},
    )
    try:
        arrays, native = run_world(ASSETS / "upstream", directory, spec["world"])
        write_record(
            directory / "seal.json",
            {
                "lock_id": lock["artifact_id"],
                "claim_id": claim["artifact_id"],
                "task": spec,
                "bundle": write_native_bundle(directory, arrays),
                "native": native,
                "value_analyzed": False,
                "protected_data_read": False,
            },
        )
    except Exception as exc:
        write_record(
            directory / "failure.json",
            {
                "lock_id": lock["artifact_id"],
                "claim_id": claim["artifact_id"],
                "task": spec,
                "error": f"{type(exc).__name__}: {exc}",
                "retry_authorized": False,
                "protected_data_read": False,
            },
        )
        raise


def run(output: Path) -> None:
    if output.resolve() != OUTPUT or output.exists() or output.is_symlink():
        raise ValueError("the registered coiling source root must be fresh")
    revision = clean_revision(ROOT)
    output.mkdir(parents=True, exist_ok=False)
    lock = write_record(
        output / "lock.json",
        {
            "schema": "dlolab-coiling-development-lock-v1-1",
            "revision": revision,
            "protocol": protocol_v1_1(),
            "parent_failure": parent_failure(),
            "source_sha256": {path: file_digest(ROOT / path) for path in SOURCES},
            "native_source": source(),
            "runtime": runtime(),
            "output_root": str(OUTPUT),
            "retry_authorized": False,
            "protected_data_read": False,
        },
    )
    completed: list[str] = []
    attempted = 0
    stage = "native_worlds"

    def terminal(status: str, **extra: Any) -> dict[str, Any]:
        result = write_record(
            output / "result.json",
            {
                "schema": "dlolab-coiling-development-result-v1-1",
                "lock_id": lock["artifact_id"],
                "status": status,
                "completed_worlds": len(completed),
                "attempted_worlds": attempted,
                "unrun_worlds": len(worlds()) - attempted,
                "completed_seal_ids": completed,
                "development_gate_passed": False,
                "prospective_replication_automatically_authorized": False,
                "retry_authorized": False,
                "protected_data_read": False,
                "new_recordings": False,
                "gpu_work": False,
                **extra,
            },
        )
        print(
            f"terminal={status}; gate={result['development_gate_passed']}; "
            f"id={result['artifact_id']}",
            flush=True,
        )
        return cast(dict[str, Any], result)

    try:
        for index in range(len(worlds())):
            stage = f"world_{index:02d}"
            attempted += 1
            print(
                f"starting frozen coiling world {index + 1}/{len(worlds())}", flush=True
            )
            with (output / f"worker-{index:02d}.log").open("xb") as stream:
                subprocess.run(
                    [
                        sys.executable,
                        "-u",
                        str(Path(__file__).resolve()),
                        "--worker",
                        str(index),
                    ],
                    cwd=ROOT,
                    env=os.environ.copy(),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            seal, _, qa = load_task(output, lock, index)
            completed.append(seal["artifact_id"])
            write_record(
                output / task(index)["name"] / "qa.json",
                {
                    "lock_id": lock["artifact_id"],
                    "seal_id": seal["artifact_id"],
                    "qa": qa,
                },
            )
            if not qa["passed"]:
                terminal(
                    "native_qualification_failed",
                    failed_world=index,
                    failed_checks=[
                        key for key, value in qa["checks"].items() if not value
                    ],
                )
                return
            print(f"sealed and qualified world {index + 1}/{len(worlds())}", flush=True)
        stage = "source_bank_seal"
        rows = [load_task(output, lock, index) for index in range(len(worlds()))]
        bank = {
            "prefix": np.stack([row[1]["prefix_positions_m"][:, 1] for row in rows]),
            "reward": np.asarray([row[2]["final_rewards"][:7] for row in rows]),
        }
        directory = output / "source-bank"
        directory.mkdir()
        bank_seal = write_record(
            directory / "seal.json",
            {
                "lock_id": lock["artifact_id"],
                "source_seal_ids": [row[0]["artifact_id"] for row in rows],
                "bundle": write_native_bundle(directory, bank),
            },
        )
        stage = "development_value"
        metrics = source_value(bank["prefix"], bank["reward"])
        terminal(
            "complete",
            source_bank_id=bank_seal["artifact_id"],
            metrics=metrics,
            development_gate_passed=metrics["development_gate_passed"],
        )
    except Exception as exc:
        write_record(
            output / "failure.json",
            {
                "lock_id": lock["artifact_id"],
                "stage": stage,
                "completed_worlds": len(completed),
                "attempted_worlds": attempted,
                "unrun_worlds": len(worlds()) - attempted,
                "error": f"{type(exc).__name__}: {exc}",
                "retry_authorized": False,
                "protected_data_read": False,
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=int, choices=range(len(worlds())))
    arguments = parser.parse_args()
    if arguments.worker is None:
        run(OUTPUT)
    else:
        worker(OUTPUT, arguments.worker)
