#!/usr/bin/env python3
"""One frozen CPU-only wrapping source screen, with no task retries."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin_experiments.dlolab_benchmark import (
    source_identity,
    write_native_bundle,
)
from bayesian_phystwin_experiments.dlolab_native import file_digest
from bayesian_phystwin_experiments.dlolab_regret_artifacts import (
    clean_revision,
    read_record,
    runtime_identity,
    write_record,
)
from bayesian_phystwin_experiments.dlolab_slingshot_process import load_native_bundle
from bayesian_phystwin_experiments.dlolab_wrapping_native import run_world
from bayesian_phystwin_experiments.dlolab_wrapping_source import (
    N_ACTIONS,
    N_ENVS,
    PREFIX_STEPS,
    information_value,
    native_qa,
    prefix_observation,
    protocol,
    repeat_qa,
    task,
)

ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path("/home/fpfaff/source-only/dlolab-benchmark-source-v1-assets")
OUTPUT = Path("/home/fpfaff/source-only/dlolab-wrapping-belief-source-v1")
SOURCES = (
    "src/bayesian_phystwin_experiments/dlolab_wrapping_source.py",
    "src/bayesian_phystwin_experiments/dlolab_wrapping_native.py",
    "scripts/remote/run_dlolab_wrapping_source.py",
    "scripts/verify_dlolab_wrapping_source.py",
    "tests/test_dlolab_wrapping_source.py",
    "tests/test_dlolab_wrapping_runner.py",
    "docs/dlolab_wrapping_belief_source_v1.md",
    "src/bayesian_phystwin_experiments/dlolab_benchmark.py",
    "src/bayesian_phystwin_experiments/dlolab_native.py",
    "src/bayesian_phystwin_experiments/dlolab_regret_artifacts.py",
    "src/bayesian_phystwin_experiments/dlolab_slingshot_process.py",
    "src/bayesian_phystwin_experiments/dlolab_regret_study.py",
    "src/bayesian_phystwin_experiments/deform_state_restart.py",
    "src/bayesian_phystwin/_portable_contracts.py",
    "src/bayesian_phystwin/_canonical_contracts.py",
)


def runtime() -> dict[str, Any]:
    result = runtime_identity()
    result["benchmark_packages"] = {
        p: importlib.metadata.version(p)
        for p in (
            "pin",
            "pin-pink",
            "qpsolvers",
            "proxsuite",
            "quadprog",
            "mushroom-rl",
            "omegaconf",
        )
    }
    return result


def source() -> dict[str, Any]:
    return source_identity(
        ASSETS / "upstream", ASSETS / "mushroom-rl", ASSETS / "dlo-lab.zip"
    )


def validate_lock(output: Path) -> dict:
    if output.resolve() != OUTPUT or output.is_symlink():
        raise ValueError("only the registered write-once output root is permitted")
    if (output / "failure.json").exists() or (output / "result.json").exists():
        raise ValueError("terminal wrapping source study; no retry")
    lock = read_record(output / "lock.json")
    if (
        lock["schema"] != "dlolab-wrapping-source-lock-v1"
        or lock["revision"] != clean_revision(ROOT)
        or lock["protocol"] != protocol()
        or lock["output_root"] != str(OUTPUT)
        or lock["runtime"] != runtime()
        or lock["native_source"] != source()
        or set(lock["source_sha256"]) != set(SOURCES)
        or any(
            file_digest(ROOT / p) != digest
            for p, digest in lock["source_sha256"].items()
        )
    ):
        raise ValueError("clean frozen implementation/runtime/source required")
    return lock


def load_task(output: Path, lock: dict, index: int):
    spec = task(index)
    directory = output / spec["name"]
    claim = read_record(directory / "claim.json")
    seal = read_record(directory / "seal.json")
    if (
        claim["lock_id"] != lock["artifact_id"]
        or seal["lock_id"] != lock["artifact_id"]
        or seal["claim_id"] != claim["artifact_id"]
        or claim["task"] != spec
        or seal["task"] != spec
        or seal["belief_value_analyzed"] is not False
        or seal["protected_data_read"] is not False
    ):
        raise ValueError("native task custody changed")
    data = load_native_bundle(directory, seal["bundle"])
    return seal, data, native_qa(data, seal["native"], spec["world"])


def prerequisites(output: Path, lock: dict, index: int) -> None:
    task(index)
    nominal = []
    for previous in range(index):
        seal, data, qa = load_task(output, lock, previous)
        receipt = read_record(output / task(previous)["name"] / "qa.json")
        if (
            not qa["passed"]
            or receipt["qa"] != qa
            or receipt["seal_id"] != seal["artifact_id"]
            or receipt["lock_id"] != lock["artifact_id"]
        ):
            raise ValueError(
                "previous native task did not pass rederived qualification"
            )
        if previous < 3:
            nominal.append((data, seal["native"]["native_final_reward"]))
    if index >= 3:
        repeat = repeat_qa([r[0] for r in nominal], np.asarray([r[1] for r in nominal]))
        stored = read_record(output / "repeat-qualification.json")
        if (
            not repeat["passed"]
            or stored["repeat_qa"] != repeat
            or stored["lock_id"] != lock["artifact_id"]
        ):
            raise ValueError(
                "material study requires passing repeated native qualification"
            )


def worker(output: Path, index: int) -> None:
    lock = validate_lock(output)
    prerequisites(output, lock, index)
    spec = task(index)
    directory = output / spec["name"]
    directory.mkdir(exist_ok=False)
    claim = write_record(
        directory / "claim.json",
        {"lock_id": lock["artifact_id"], "task": spec, "retry_authorized": False},
    )
    try:
        data, native = run_world(ASSETS / "upstream", directory, spec["world"])
        write_record(
            directory / "seal.json",
            {
                "lock_id": lock["artifact_id"],
                "claim_id": claim["artifact_id"],
                "task": spec,
                "bundle": write_native_bundle(directory, data),
                "native": native,
                "belief_value_analyzed": False,
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
        raise ValueError("the registered source root must be fresh")
    revision = clean_revision(ROOT)
    native_source, native_runtime = source(), runtime()
    output.mkdir(parents=True, exist_ok=False)
    lock = write_record(
        output / "lock.json",
        {
            "schema": "dlolab-wrapping-source-lock-v1",
            "revision": revision,
            "protocol": protocol(),
            "source_sha256": {p: file_digest(ROOT / p) for p in SOURCES},
            "native_source": native_source,
            "runtime": native_runtime,
            "output_root": str(OUTPUT),
            "retry_authorized": False,
            "protected_data_read": False,
        },
    )
    completed: list[str] = []
    admitted: list[str] = []
    attempted = 0
    stage = "native_qualification"

    def terminal(status: str, **extra: Any) -> dict:
        value = write_record(
            output / "result.json",
            {
                "schema": "dlolab-wrapping-source-result-v1",
                "lock_id": lock["artifact_id"],
                "status": status,
                "completed_batches": len(completed),
                "admitted_batches": len(admitted),
                "ordinary_trajectories": N_ENVS * len(admitted),
                "completed_native_trajectories": N_ENVS * len(completed),
                "qualified_trajectories": N_ENVS * len(admitted),
                "unrun_batches": 11 - attempted,
                "attempted_batches": attempted,
                "completed_seal_ids": completed,
                "source_gate_passed": False,
                "method_promotion_authorized": False,
                "retry_authorized": False,
                "protected_data_read": False,
                "new_recordings": False,
                "gpu_work": False,
                **extra,
            },
        )
        print(
            f"terminal={status}; gate={value['source_gate_passed']}; id={value['artifact_id']}",
            flush=True,
        )
        return value

    try:
        for index in range(11):
            stage = f"worker_{index:02d}"
            attempted += 1
            print(f"starting frozen wrapping batch {index + 1}/11", flush=True)
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
                    failed_batch=index,
                    failed_checks=[k for k, v in qa["checks"].items() if not v],
                )
                return
            admitted.append(seal["artifact_id"])
            print(f"sealed and admitted batch {index + 1}/11", flush=True)
            if index == 2:
                nominal = [load_task(output, lock, i) for i in range(3)]
                repeat = repeat_qa(
                    [r[1] for r in nominal],
                    np.asarray(
                        [r[0]["native"]["native_final_reward"] for r in nominal]
                    ),
                )
                write_record(
                    output / "repeat-qualification.json",
                    {"lock_id": lock["artifact_id"], "repeat_qa": repeat},
                )
                if not repeat["passed"]:
                    terminal("native_repeatability_failed")
                    return
        stage = "source_bank_seal"
        rows = [load_task(output, lock, i) for i in range(11) if i not in (1, 2)]
        rows.sort(key=lambda row: row[0]["task"]["world"]["index"])
        bank = {
            "prefix": np.stack(
                [prefix_observation(row[1]["rod_pos_m"][:PREFIX_STEPS]) for row in rows]
            ),
            "reward": np.asarray(
                [row[0]["native"]["native_final_reward"][:N_ACTIONS] for row in rows]
            ),
        }
        directory = output / "source-bank"
        directory.mkdir()
        seal = write_record(
            directory / "seal.json",
            {
                "lock_id": lock["artifact_id"],
                "source_seal_ids": [r[0]["artifact_id"] for r in rows],
                "bundle": write_native_bundle(directory, bank),
            },
        )
        stage = "source_information_value"
        metrics = information_value(bank["prefix"], bank["reward"])
        terminal(
            "complete",
            source_bank_id=seal["artifact_id"],
            metrics=metrics,
            source_gate_passed=metrics["source_gate_passed"],
        )
    except Exception as exc:
        write_record(
            output / "failure.json",
            {
                "lock_id": lock["artifact_id"],
                "stage": stage,
                "completed_batches": len(completed),
                "admitted_batches": len(admitted),
                "attempted_batches": attempted,
                "unrun_batches": 11 - attempted,
                "error": f"{type(exc).__name__}: {exc}",
                "retry_authorized": False,
                "protected_data_read": False,
            },
        )
        terminal(
            "technical_failure",
            failed_stage=stage,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=int, choices=range(11))
    args = parser.parse_args()
    if args.worker is None:
        run(OUTPUT)
    else:
        worker(OUTPUT, args.worker)
