#!/usr/bin/env python3
"""Run the frozen public DLO-Lab separation headroom screen once."""

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

from bayesian_phystwin_experiments.deform_state_restart import file_digest
from bayesian_phystwin_experiments.dlolab_benchmark import (
    source_identity,
    write_native_bundle,
)
from bayesian_phystwin_experiments.dlolab_regret_artifacts import (
    clean_revision,
    read_record,
    runtime_identity,
    write_record,
)
from bayesian_phystwin_experiments.dlolab_separation_headroom_v1 import (
    UNIQUE_ACTION_COUNT,
    development_metrics,
    native_qa,
    protocol,
    task,
    worlds,
)
from bayesian_phystwin_experiments.dlolab_separation_native_v1 import run_world
from bayesian_phystwin_experiments.dlolab_slingshot_process import (
    load_native_bundle,
)

ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path("/home/fpfaff/source-only/dlolab-benchmark-source-v1-assets")
OUTPUT = Path("/home/fpfaff/source-only/dlolab-separation-headroom-development-v1")
SOURCES = (
    "src/bayesian_phystwin_experiments/dlolab_separation_headroom_v1.py",
    "src/bayesian_phystwin_experiments/dlolab_separation_native_v1.py",
    "scripts/remote/run_dlolab_separation_headroom_v1.py",
    "tests/test_dlolab_separation_headroom_v1.py",
    "tests/test_dlolab_separation_headroom_runner_v1.py",
    "docs/dlolab_separation_headroom_development_v1.md",
    "src/bayesian_phystwin_experiments/dlolab_benchmark.py",
    "src/bayesian_phystwin_experiments/dlolab_native.py",
    "src/bayesian_phystwin_experiments/dlolab_regret_artifacts.py",
    "src/bayesian_phystwin_experiments/dlolab_slingshot_process.py",
    "src/bayesian_phystwin_experiments/deform_state_restart.py",
)

Array: TypeAlias = NDArray[Any]


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
        raise ValueError("only the registered write-once separation root is permitted")
    if (output / "failure.json").exists() or (output / "result.json").exists():
        raise ValueError("terminal separation development study; no retry")
    lock = read_record(output / "lock.json")
    if (
        lock.get("schema") != "dlolab-separation-headroom-lock-v1"
        or lock.get("revision") != clean_revision(ROOT)
        or lock.get("protocol") != protocol()
        or lock.get("output_root") != str(OUTPUT)
        or lock.get("runtime") != runtime()
        or lock.get("native_source") != source()
        or lock.get("source_sha256")
        != {path: file_digest(ROOT / path) for path in SOURCES}
    ):
        raise ValueError(
            "clean frozen separation implementation/runtime/source required"
        )
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
        or seal.get("value_analyzed") is not False
        or seal.get("protected_data_read") is not False
    ):
        raise ValueError("native separation task custody changed")
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
            raise ValueError("all earlier separation worlds must requalify")
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
        raise ValueError("the registered separation development root must be fresh")
    revision = clean_revision(ROOT)
    output.mkdir(parents=True, exist_ok=False)
    lock = write_record(
        output / "lock.json",
        {
            "schema": "dlolab-separation-headroom-lock-v1",
            "revision": revision,
            "protocol": protocol(),
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
    stage = "native_development_worlds"

    def terminal(status: str, **extra: Any) -> dict[str, Any]:
        result = write_record(
            output / "result.json",
            {
                "schema": "dlolab-separation-headroom-result-v1",
                "lock_id": lock["artifact_id"],
                "status": status,
                "completed_worlds": len(completed),
                "attempted_worlds": attempted,
                "unrun_worlds": len(worlds()) - attempted,
                "completed_seal_ids": completed,
                "development_gate_passed": False,
                "source_transfer_automatically_authorized": False,
                "prospective_execution_authorized": False,
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
            stage = f"development_world_{index:02d}"
            attempted += 1
            print(
                f"starting frozen separation world {index + 1}/{len(worlds())}",
                flush=True,
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
            print(
                f"sealed and qualified world {index + 1}/{len(worlds())}",
                flush=True,
            )
        stage = "value_analysis"
        rows = [load_task(output, lock, index) for index in range(len(worlds()))]
        reward_bank = np.asarray(
            [row[2]["final_rewards_m"][:UNIQUE_ACTION_COUNT] for row in rows],
            dtype=np.float64,
        )
        directory = output / "development-bank"
        directory.mkdir()
        bank = write_record(
            directory / "seal.json",
            {
                "lock_id": lock["artifact_id"],
                "source_seal_ids": [row[0]["artifact_id"] for row in rows],
                "bundle": write_native_bundle(directory, {"reward_m": reward_bank}),
            },
        )
        metrics = development_metrics(reward_bank)
        terminal(
            "complete",
            development_bank_id=bank["artifact_id"],
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
