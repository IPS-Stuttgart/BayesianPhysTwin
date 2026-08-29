#!/usr/bin/env python3
"""Run the bounded source-development task-valued Slingshot probe screen."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin_experiments.dlolab_benchmark import (
    source_identity,
    write_native_bundle,
)
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
    NEW_FRACTIONS,
    conditional_prior,
    evaluate_candidates,
    frontloaded_controls,
    new_probe_task,
    protocol,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path("/home/fpfaff/source-only/dlolab-slingshot-task-probe-dev-v1")
PARENT = Path(
    "/home/fpfaff/source-only/dlolab-benchmark-source-v1/belief-control-source-v1"
)
EXISTING = Path(
    "/home/fpfaff/source-only/dlolab-benchmark-source-v1/informative-prefix-source-v1"
)
PARENT_LOCK_ID = "015e6d84aa68a2a4310552ef4880752b972890f02d3e09e333ff575c92b8df25"
BANK_ID = "8ebf9c91322faf0658c84a2dcaa6895a98b1ff857e49e6714a2a2dad0c88d882"
EXISTING_LOCK_ID = "9224e18b38e06c24c1e89e0de503aeeb58b8fa383819f47d3a981cbc3c720cf9"
EXISTING_PREFIX_ID = "c6c78ad9f81d5440a020d7ec21f63e6d23204b3aed3f21306221d2f4bbd63303"
ADDITIONAL_SOURCES = (
    "src/bayesian_phystwin_experiments/dlolab_slingshot_task_probe_dev.py",
    "scripts/remote/run_dlolab_slingshot_task_probe_dev.py",
    "tests/test_dlolab_slingshot_task_probe_dev.py",
    "docs/dlolab_slingshot_task_probe_development_v1.md",
)


def _source() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    old = read_record(PARENT / "lock.json")
    bank_seal = read_record(PARENT / "model-bank/seal.json")
    if (
        old.get("artifact_id") != PARENT_LOCK_ID
        or bank_seal.get("artifact_id") != BANK_ID
        or bank_seal.get("lock_id") != PARENT_LOCK_ID
        or any(
            file_digest(ROOT / name) != digest
            for name, digest in old["source_sha256"].items()
        )
    ):
        raise ValueError("exact opened Slingshot source bank required")
    assets = Path(old["assets_root"])
    native = old["screen"]["source"]["controller"]
    if (
        runtime() != native["runtime"]
        or source_identity(
            assets / "upstream", assets / "mushroom-rl", assets / "dlo-lab.zip"
        )
        != native["native_source"]
    ):
        raise ValueError("qualified public Slingshot runtime changed")
    bank = load_native_bundle(PARENT / "model-bank", bank_seal["bundle"])
    if bank["prefix"].shape != (27, 3, 4, 3) or bank["reward"].shape != (27, 7):
        raise ValueError("complete opened model bank required")
    return old, bank


def _existing_50_history() -> np.ndarray:
    lock = read_record(EXISTING / "lock.json")
    decision = read_record(EXISTING / "prefix-result.json")
    if (
        lock.get("artifact_id") != EXISTING_LOCK_ID
        or decision.get("artifact_id") != EXISTING_PREFIX_ID
        or decision.get("lock_id") != EXISTING_LOCK_ID
        or decision.get("source_bank_authorized") is not False
    ):
        raise ValueError("exact closed 50% prefix evidence required")
    rows: list[np.ndarray] = []
    for batch in range(2):
        directory = EXISTING / f"probe-1-prefix-{batch}"
        seal = read_record(directory / "seal.json")
        task = seal.get("task", {})
        if (
            seal.get("lock_id") != EXISTING_LOCK_ID
            or task.get("probe") != 1
            or task.get("index") != batch
            or task.get("prefix_only") is not True
        ):
            raise ValueError("closed 50% prefix task changed")
        data = load_native_bundle(directory, seal["bundle"])
        observed = prefix_observations(data)
        rows.extend(observed[: 8 if batch == 0 else 1])
    result = np.stack(rows)
    if result.shape != (9, 3, 4, 3):
        raise ValueError("complete closed 50% history required")
    return result


def _worlds(indices: list[int]) -> list[dict[str, Any]]:
    rows = [particle_worlds()[index] for index in indices]
    return rows + [rows[-1]] * (8 - len(rows))


def _validate(output: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    if output.resolve() != OUTPUT or output.is_symlink():
        raise ValueError("only the registered development root is permitted")
    lock = read_record(output / "lock.json")
    old, bank = _source()
    if (
        lock.get("schema") != "dlolab-slingshot-task-probe-development-lock-v1"
        or lock.get("revision") != clean_revision(ROOT)
        or lock.get("protocol") != protocol()
        or lock.get("output_root") != str(OUTPUT)
        or lock.get("parent_lock_id") != PARENT_LOCK_ID
        or lock.get("bank_id") != BANK_ID
        or lock.get("existing_prefix_id") != EXISTING_PREFIX_ID
        or lock.get("retry_authorized") is not False
        or lock.get("protected_data_read") is not False
        or any(
            file_digest(ROOT / name) != digest
            for name, digest in lock["source_sha256"].items()
        )
    ):
        raise ValueError("clean frozen development lock required")
    original = np.asarray(old["controls"], dtype=np.float64)
    expected = [array_digest(frontloaded_controls(original, f)) for f in NEW_FRACTIONS]
    if lock.get("new_probe_control_sha256") != expected:
        raise ValueError("development probe commands changed")
    return lock, old, bank


def _load_task(
    output: Path,
    lock: dict[str, Any],
    old: dict[str, Any],
    probe: int,
    batch: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], bool]:
    task = new_probe_task(probe, batch)
    directory = output / task["name"]
    claim = read_record(directory / "claim.json")
    seal = read_record(directory / "seal.json")
    if (
        claim.get("lock_id") != lock["artifact_id"]
        or claim.get("task") != task
        or seal.get("claim_id") != claim["artifact_id"]
        or seal.get("lock_id") != lock["artifact_id"]
        or seal.get("task") != task
    ):
        raise ValueError("registered development task changed")
    data = load_native_bundle(directory, seal["bundle"])
    expected = frontloaded_controls(
        np.asarray(old["controls"], dtype=np.float64), NEW_FRACTIONS[probe]
    )
    expected = np.repeat(expected[5:6], 8, axis=0)
    if array_digest(data["controls"]) != array_digest(expected):
        raise ValueError("realized development commands changed")
    if set(data) != set(TRACE_NAMES + ("controls",)) or any(
        data[name].shape[:2] != (300, 8) for name in TRACE_NAMES
    ):
        raise ValueError("unexpected prefix-only development payload")
    native = seal["native"]
    if (
        native.get("native_steps") != 300
        or native.get("future_simulated") is not False
        or native.get("reward_scored") is not False
    ):
        raise ValueError("development prefix crossed the future boundary")
    fixed = float(
        np.max(
            np.abs(
                data["rod_pos_m"][:, :, [0, 1, 10, 11]]
                - data["rod_pos_m"][:1, :, [0, 1, 10, 11]]
            )
        )
    )
    duplicate = (
        0.0
        if batch == 0
        else max(
            float(np.max(np.abs(data[name] - data[name][:, :1])))
            for name in ("rod_pos_m", "sphere_pos_m", "cube_pos_m", "gripper_pos_m")
        )
    )
    return seal, data, bool(fixed <= 1e-9 and duplicate <= 0.0005)


def _worker(output: Path, probe: int, batch: int) -> None:
    lock, old, _ = _validate(output)
    task = new_probe_task(probe, batch)
    directory = output / task["name"]
    directory.mkdir()
    claim = write_record(
        directory / "claim.json", {"lock_id": lock["artifact_id"], "task": task}
    )
    controls = frontloaded_controls(
        np.asarray(old["controls"], dtype=np.float64), NEW_FRACTIONS[probe]
    )
    controls = np.repeat(controls[5:6], 8, axis=0)
    data, native = run_registered_worlds(
        Path(old["assets_root"]) / "upstream",
        directory,
        controls,
        _worlds(task["world_indices"]),
        prefix_only=True,
    )
    bundle = write_native_bundle(directory, data)
    write_record(
        directory / "seal.json",
        {
            "lock_id": lock["artifact_id"],
            "claim_id": claim["artifact_id"],
            "task": task,
            "bundle": bundle,
            "native": native,
        },
    )
    _load_task(output, lock, old, probe, batch)


def _launch(
    output: Path, old: dict[str, Any], probe: int, batch: int
) -> None:
    task = new_probe_task(probe, batch)
    command = [
        sys.executable,
        str(Path(__file__)),
        "--output",
        str(output),
        "--worker",
        "--probe",
        str(probe),
        "--batch",
        str(batch),
    ]
    with (output / f"{task['name']}.log").open("x") as handle:
        run = subprocess.run(
            command,
            cwd=ROOT,
            env=worker_environment(old["screen"]["source"]["controller"]["runtime"]),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if run.returncode:
        raise RuntimeError(f"{task['name']} exited {run.returncode}; no retry")


def _run(output: Path) -> None:
    if output.resolve() != OUTPUT or output.exists():
        raise ValueError("fresh registered development root required")
    old, bank = _source()
    existing = _existing_50_history()
    source_paths = sorted(set(old["source_sha256"]) | set(ADDITIONAL_SOURCES))
    original = np.asarray(old["controls"], dtype=np.float64)
    output.mkdir()
    lock = write_record(
        output / "lock.json",
        {
            "schema": "dlolab-slingshot-task-probe-development-lock-v1",
            "revision": clean_revision(ROOT),
            "source_sha256": {name: file_digest(ROOT / name) for name in source_paths},
            "protocol": protocol(),
            "output_root": str(OUTPUT),
            "parent_lock_id": PARENT_LOCK_ID,
            "bank_id": BANK_ID,
            "existing_lock_id": EXISTING_LOCK_ID,
            "existing_prefix_id": EXISTING_PREFIX_ID,
            "new_probe_control_sha256": [
                array_digest(frontloaded_controls(original, fraction))
                for fraction in NEW_FRACTIONS
            ],
            "retry_authorized": False,
            "protected_data_read": False,
        },
    )
    try:
        new_histories: list[np.ndarray] = []
        qa: list[bool] = []
        seal_ids: list[str] = []
        for probe in range(len(NEW_FRACTIONS)):
            rows: list[np.ndarray] = []
            for batch in range(2):
                _launch(output, old, probe, batch)
                seal, data, passed = _load_task(output, lock, old, probe, batch)
                seal_ids.append(seal["artifact_id"])
                qa.append(passed)
                observed = prefix_observations(data)
                rows.extend(observed[: 8 if batch == 0 else 1])
            new_histories.append(np.stack(rows))
        histories = np.stack(
            [bank["prefix"][9:18], existing, *new_histories]
        )
        rewards = np.asarray(bank["reward"][9:18], dtype=np.float64)
        prior = conditional_prior()
        metrics = evaluate_candidates(histories, rewards, prior)
        checks = {
            "all_new_prefixes_native_qualified": len(qa) == 4 and all(qa),
            **metrics["checks"],
        }
        bank_directory = output / "development-bank"
        bank_directory.mkdir()
        bundle = write_native_bundle(
            bank_directory,
            {"history": histories, "reward": rewards, "prior": prior},
        )
        bank_seal = write_record(
            bank_directory / "seal.json",
            {
                "lock_id": lock["artifact_id"],
                "new_prefix_seal_ids": seal_ids,
                "bundle": bundle,
                "new_prefix_qa": qa,
                "truth_future_generated": False,
            },
        )
        write_record(
            output / "result.json",
            {
                "schema": "dlolab-slingshot-task-probe-development-result-v1",
                "lock_id": lock["artifact_id"],
                "development_bank_id": bank_seal["artifact_id"],
                "metrics": metrics,
                "checks": checks,
                "value_feasibility_passed": all(checks.values()),
                "future_protocol_automatically_authorized": False,
                "truth_future_generated": False,
                "protected_data_read": False,
                "retry_authorized": False,
            },
        )
    except Exception as exc:
        write_record(
            output / "failure.json",
            {
                "schema": "dlolab-slingshot-task-probe-development-failure-v1",
                "lock_id": lock["artifact_id"],
                "error_type": type(exc).__name__,
                "error": str(exc),
                "truth_future_generated": False,
                "protected_data_read": False,
                "retry_authorized": False,
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--probe", type=int)
    parser.add_argument("--batch", type=int)
    args = parser.parse_args()
    if args.worker:
        if args.probe is None or args.batch is None:
            raise ValueError("worker probe and batch required")
        _worker(args.output, args.probe, args.batch)
    else:
        if args.probe is not None or args.batch is not None:
            raise ValueError("worker-only task arguments supplied")
        _run(args.output)


if __name__ == "__main__":
    main()
