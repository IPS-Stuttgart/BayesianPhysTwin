#!/usr/bin/env python3
"""Run one frozen CPU-only active-probe wrapping source study."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin_experiments.dlolab_active_wrapping_native import run_world
from bayesian_phystwin_experiments.dlolab_active_wrapping_source import (
    N_ACTIONS,
    N_ENVS,
    active_decision_gate,
    decision_value,
    native_qa,
    prefix_observation,
    probe_information,
    protocol,
    repeat_qa,
    task,
)
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

ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path("/home/fpfaff/source-only/dlolab-benchmark-source-v1-assets")
OUTPUT = Path("/home/fpfaff/source-only/dlolab-active-probe-wrapping-source-v1-1")
PASSIVE_RESULT = ROOT / "results/sota/dlolab_wrapping_belief_source_v1/result.json"
SOURCES = (
    "src/bayesian_phystwin_experiments/dlolab_active_wrapping_source.py",
    "src/bayesian_phystwin_experiments/dlolab_active_wrapping_native.py",
    "scripts/remote/run_dlolab_active_probe_wrapping_source.py",
    "scripts/verify_dlolab_active_probe_wrapping_source.py",
    "tests/test_dlolab_active_wrapping_source.py",
    "tests/test_dlolab_active_wrapping_runner.py",
    "docs/dlolab_active_probe_wrapping_source_v1.md",
    "configs/sota/dlolab_active_probe_wrapping_source_v1.json",
    "results/sota/dlolab_wrapping_belief_source_v1/result.json",
    "results/sota/dlolab_active_probe_wrapping_prelock_v1/prelock-failure.json",
    "results/sota/dlolab_active_probe_wrapping_prelock_v1/prelock-failure-correction.json",
    "src/bayesian_phystwin_experiments/dlolab_wrapping_source.py",
    "src/bayesian_phystwin_experiments/dlolab_benchmark.py",
    "src/bayesian_phystwin_experiments/dlolab_native.py",
    "src/bayesian_phystwin_experiments/dlolab_regret_artifacts.py",
    "src/bayesian_phystwin_experiments/dlolab_slingshot_process.py",
    "src/bayesian_phystwin_experiments/dlolab_regret_study.py",
    "src/bayesian_phystwin_experiments/deform_state_restart.py",
    "src/bayesian_phystwin/_portable_contracts.py",
    "src/bayesian_phystwin/_canonical_contracts.py",
)
STAGES = ("probe", "baseline", "active")


def runtime() -> dict[str, Any]:
    result = runtime_identity()
    result["benchmark_packages"] = {
        package: importlib.metadata.version(package)
        for package in (
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


def _selected_probe(output: Path) -> int:
    row = read_record(output / "probe-selection.json")
    if (
        row["schema"] != "dlolab-active-wrapping-probe-selection-v1"
        or row["metrics"]["passed"] is not True
        or row["future_reward_read"] is not False
        or type(row["metrics"]["selected_probe_index"]) is not int
    ):
        raise ValueError("passing reward-blind probe selection required")
    return row["metrics"]["selected_probe_index"]


def validate_lock(output: Path) -> dict:
    if output.resolve() != OUTPUT or output.is_symlink():
        raise ValueError("only the registered write-once output root is permitted")
    if (output / "failure.json").exists() or (output / "result.json").exists():
        raise ValueError("terminal active-wrapping source study; no retry")
    lock = read_record(output / "lock.json")
    if (
        lock["schema"] != "dlolab-active-probe-wrapping-source-lock-v1-1"
        or lock["revision"] != clean_revision(ROOT)
        or lock["protocol"] != protocol()
        or lock["output_root"] != str(OUTPUT)
        or lock["runtime"] != runtime()
        or lock["native_source"] != source()
        or set(lock["source_sha256"]) != set(SOURCES)
        or any(
            file_digest(ROOT / path) != digest
            for path, digest in lock["source_sha256"].items()
        )
    ):
        raise ValueError("clean frozen active-wrapping implementation required")
    return lock


def load_task(
    output: Path, lock: dict, stage: str, index: int, probe_index: int | None
):
    spec = task(stage, index, probe_index)
    directory = output / spec["name"]
    claim = read_record(directory / "claim.json")
    seal = read_record(directory / "seal.json")
    if (
        claim["lock_id"] != lock["artifact_id"]
        or seal["lock_id"] != lock["artifact_id"]
        or seal["claim_id"] != claim["artifact_id"]
        or claim["task"] != spec
        or seal["task"] != spec
        or seal["decision_analyzed"] is not False
        or seal["protected_data_read"] is not False
    ):
        raise ValueError("active-wrapping native task custody changed")
    data = load_native_bundle(directory, seal["bundle"])
    return (
        seal,
        data,
        native_qa(data, seal["native"], spec["world"], stage, probe_index),
    )


def prerequisites(
    output: Path,
    lock: dict,
    stage: str,
    index: int,
    probe_index: int | None,
) -> None:
    task(stage, index, probe_index)
    if stage == "baseline" and probe_index != 0:
        raise ValueError("baseline stage must use the null probe")
    if stage == "active" and probe_index != _selected_probe(output):
        raise ValueError("active stage must use the sealed selected probe")
    if stage != "probe":
        _selected_probe(output)
    nominal = []
    for previous in range(index):
        seal, data, qa = load_task(output, lock, stage, previous, probe_index)
        receipt = read_record(
            output / task(stage, previous, probe_index)["name"] / "qa.json"
        )
        if (
            not qa["passed"]
            or receipt["qa"] != qa
            or receipt["seal_id"] != seal["artifact_id"]
            or receipt["lock_id"] != lock["artifact_id"]
        ):
            raise ValueError("previous active-wrapping task did not qualify")
        if previous < 3:
            nominal.append((data, seal["native"]["native_final_reward"]))
    if index >= 3:
        repeated = repeat_qa(
            [row[0] for row in nominal], np.asarray([row[1] for row in nominal])
        )
        stored = read_record(output / f"{stage}-repeat-qualification.json")
        if (
            not repeated["passed"]
            or stored["repeat_qa"] != repeated
            or stored["lock_id"] != lock["artifact_id"]
        ):
            raise ValueError("stage requires passing repeat qualification")


def worker(output: Path, stage: str, index: int, probe_index: int | None) -> None:
    lock = validate_lock(output)
    prerequisites(output, lock, stage, index, probe_index)
    spec = task(stage, index, probe_index)
    directory = output / spec["name"]
    directory.mkdir(exist_ok=False)
    claim = write_record(
        directory / "claim.json",
        {"lock_id": lock["artifact_id"], "task": spec, "retry_authorized": False},
    )
    try:
        data, native = run_world(
            ASSETS / "upstream", directory, spec["world"], stage, probe_index
        )
        write_record(
            directory / "seal.json",
            {
                "lock_id": lock["artifact_id"],
                "claim_id": claim["artifact_id"],
                "task": spec,
                "bundle": write_native_bundle(directory, data),
                "native": native,
                "decision_analyzed": False,
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


def _execute_stage(
    output: Path,
    lock: dict,
    stage: str,
    probe_index: int | None,
    completed: list[str],
    admitted: list[str],
) -> tuple[bool, int]:
    attempted = 0
    for index in range(11):
        attempted += 1
        print(f"starting frozen {stage} batch {index + 1}/11", flush=True)
        with (output / f"{stage}-worker-{index:02d}.log").open("xb") as stream:
            command = [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "--worker-stage",
                stage,
                "--worker-index",
                str(index),
            ]
            if probe_index is not None:
                command += ["--probe-index", str(probe_index)]
            subprocess.run(
                command,
                cwd=ROOT,
                env=os.environ.copy(),
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        seal, _, qa = load_task(output, lock, stage, index, probe_index)
        completed.append(seal["artifact_id"])
        write_record(
            output / task(stage, index, probe_index)["name"] / "qa.json",
            {"lock_id": lock["artifact_id"], "seal_id": seal["artifact_id"], "qa": qa},
        )
        if not qa["passed"]:
            return False, attempted
        admitted.append(seal["artifact_id"])
        if index == 2:
            nominal = [load_task(output, lock, stage, i, probe_index) for i in range(3)]
            repeated = repeat_qa(
                [row[1] for row in nominal],
                np.asarray(
                    [row[0]["native"]["native_final_reward"] for row in nominal]
                ),
            )
            write_record(
                output / f"{stage}-repeat-qualification.json",
                {"lock_id": lock["artifact_id"], "repeat_qa": repeated},
            )
            if not repeated["passed"]:
                return False, attempted
        print(f"sealed and admitted {stage} batch {index + 1}/11", flush=True)
    return True, attempted


def _unique_rows(output: Path, lock: dict, stage: str, probe_index: int | None):
    rows = [
        load_task(output, lock, stage, index, probe_index)
        for index in range(11)
        if index not in (1, 2)
    ]
    rows.sort(key=lambda row: row[0]["task"]["world"]["index"])
    return rows


def _seal_bank(
    output: Path, lock: dict, stage: str, probe_index: int | None
) -> tuple[dict, dict[str, np.ndarray]]:
    rows = _unique_rows(output, lock, stage, probe_index)
    if stage == "probe":
        bank = {
            "prefix": np.stack(
                [prefix_observation(row[1]["rod_pos_m"], stage) for row in rows]
            )
        }
    else:
        bank = {
            "prefix": np.stack(
                [prefix_observation(row[1]["rod_pos_m"], stage) for row in rows]
            ),
            "reward": np.asarray(
                [row[0]["native"]["native_final_reward"][:N_ACTIONS] for row in rows]
            ),
        }
    directory = output / f"{stage}-source-bank"
    directory.mkdir()
    seal = write_record(
        directory / "seal.json",
        {
            "lock_id": lock["artifact_id"],
            "source_seal_ids": [row[0]["artifact_id"] for row in rows],
            "probe_index": probe_index,
            "future_reward_present": stage != "probe",
            "bundle": write_native_bundle(directory, bank),
        },
    )
    return seal, bank


def run(output: Path) -> None:
    if output.resolve() != OUTPUT or output.exists() or output.is_symlink():
        raise ValueError("registered active-wrapping output root must be fresh")
    revision = clean_revision(ROOT)
    native_source = source()
    native_runtime = runtime()
    output.mkdir(parents=True, exist_ok=False)
    lock = write_record(
        output / "lock.json",
        {
            "schema": "dlolab-active-probe-wrapping-source-lock-v1-1",
            "revision": revision,
            "protocol": protocol(),
            "source_sha256": {path: file_digest(ROOT / path) for path in SOURCES},
            "native_source": native_source,
            "runtime": native_runtime,
            "passive_result_sha256": file_digest(PASSIVE_RESULT),
            "output_root": str(OUTPUT),
            "retry_authorized": False,
            "protected_data_read": False,
        },
    )
    completed: list[str] = []
    admitted: list[str] = []
    attempted = 0
    stage = "probe_native_qualification"

    def terminal(status: str, **extra: Any) -> dict:
        value = write_record(
            output / "result.json",
            {
                "schema": "dlolab-active-probe-wrapping-source-result-v1-1",
                "lock_id": lock["artifact_id"],
                "status": status,
                "completed_batches": len(completed),
                "admitted_batches": len(admitted),
                "ordinary_trajectories": N_ENVS * len(admitted),
                "completed_native_trajectories": N_ENVS * len(completed),
                "qualified_trajectories": N_ENVS * len(admitted),
                "attempted_batches": attempted,
                "unrun_batches": 33 - attempted,
                "completed_seal_ids": completed,
                "source_gate_passed": False,
                "method_promotion_authorized": False,
                "fresh_evaluation_authorized": False,
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
        passed, count = _execute_stage(output, lock, "probe", None, completed, admitted)
        attempted += count
        if not passed:
            terminal("probe_native_qualification_failed")
            return
        stage = "reward_blind_probe_selection"
        probe_seal, probe_bank = _seal_bank(output, lock, "probe", None)
        probe_metrics = probe_information(probe_bank["prefix"])
        selection = write_record(
            output / "probe-selection.json",
            {
                "schema": "dlolab-active-wrapping-probe-selection-v1",
                "lock_id": lock["artifact_id"],
                "probe_bank_id": probe_seal["artifact_id"],
                "metrics": probe_metrics,
                "future_reward_read": False,
            },
        )
        if not probe_metrics["passed"]:
            terminal(
                "probe_information_gate_failed",
                probe_selection_id=selection["artifact_id"],
                probe_metrics=probe_metrics,
            )
            return
        selected = probe_metrics["selected_probe_index"]
        stage = "null_continuation_qualification"
        passed, count = _execute_stage(output, lock, "baseline", 0, completed, admitted)
        attempted += count
        if not passed:
            terminal("null_continuation_qualification_failed")
            return
        stage = "active_continuation_qualification"
        passed, count = _execute_stage(
            output, lock, "active", selected, completed, admitted
        )
        attempted += count
        if not passed:
            terminal("active_continuation_qualification_failed")
            return
        stage = "matched_decision_analysis"
        null_seal, null_bank = _seal_bank(output, lock, "baseline", 0)
        active_seal, active_bank = _seal_bank(output, lock, "active", selected)
        null_metrics = decision_value(null_bank["prefix"], null_bank["reward"], 100)
        active_metrics = decision_value(
            active_bank["prefix"], active_bank["reward"], 100
        )
        passive = read_record(PASSIVE_RESULT)
        if (
            passive["schema"] != "dlolab-wrapping-source-result-v1"
            or file_digest(PASSIVE_RESULT) != lock["passive_result_sha256"]
        ):
            raise ValueError("frozen passive wrapping comparator changed")
        decision = active_decision_gate(active_metrics, null_metrics, passive)
        terminal(
            "complete",
            probe_selection_id=selection["artifact_id"],
            probe_metrics=probe_metrics,
            null_source_bank_id=null_seal["artifact_id"],
            active_source_bank_id=active_seal["artifact_id"],
            null_metrics=null_metrics,
            active_metrics=active_metrics,
            decision=decision,
            source_gate_passed=decision["passed"],
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
                "unrun_batches": 33 - attempted,
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
    parser.add_argument("--worker-stage", choices=STAGES)
    parser.add_argument("--worker-index", type=int, choices=range(11))
    parser.add_argument("--probe-index", type=int, choices=range(4))
    arguments = parser.parse_args()
    if arguments.worker_stage is None and arguments.worker_index is None:
        if arguments.probe_index is not None:
            raise ValueError("top-level run cannot preselect a probe")
        run(OUTPUT)
    elif arguments.worker_stage is not None and arguments.worker_index is not None:
        worker(
            OUTPUT,
            arguments.worker_stage,
            arguments.worker_index,
            arguments.probe_index,
        )
    else:
        raise ValueError("worker stage and index must be supplied together")
