#!/usr/bin/env python3
"""Run the frozen full-particle active-identification source qualification."""

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
from bayesian_phystwin_experiments.dlolab_slingshot_active_id import (
    ACTIVE_FRACTION,
    expected_value_screen,
    particle_task,
    protocol,
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
    frontloaded_controls,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(
    "/home/fpfaff/source-only/dlolab-slingshot-active-id-particle-source-v1"
)
ATTEMPT = Path(
    "/home/fpfaff/source-only/dlolab-slingshot-active-id-particle-source-v1.attempt.json"
)
PARENT = Path(
    "/home/fpfaff/source-only/dlolab-benchmark-source-v1/belief-control-source-v1"
)
DEVELOPMENT = Path(
    "/home/fpfaff/source-only/dlolab-slingshot-task-probe-dev-v1"
)
PARENT_LOCK_ID = "015e6d84aa68a2a4310552ef4880752b972890f02d3e09e333ff575c92b8df25"
BANK_ID = "8ebf9c91322faf0658c84a2dcaa6895a98b1ff857e49e6714a2a2dad0c88d882"
DEVELOPMENT_LOCK_ID = "fe82b961ad6634a4eaa7bde1a7034cfc81c0331087c0b897e691778f693fcb1c"
DEVELOPMENT_FAILURE_ID = "97724cd07621e6a6f63c8aa2da78f69a5308f370d64114544b0a75dff1cd35d7"
DEVELOPMENT_BANK_ID = "e7ef7d214006b74273cddd4eed65b2b79b157f6611af3dce18fcd359e0cb5947"
DEVELOPMENT_FILE_SHA256 = {
    "lock.json": "256d2577aba61d411cdab72fb5d691f4362e68082f5d0d46f152cba34eaaaa44",
    "failure.json": "89a349994decbcdc963d58643feea849ba7198960e3ac132e510e4cca69d24f7",
    "development-bank/arrays.npz": "464bd19ff2b3c7a1e271fb5d8fe005fa53c87166452334fb18ee4a988c49b804",
    "development-bank/seal.json": "6a25c83d48230e0afc9162cc4d831e34c040a0f42cafe507cef4c183b165f8fe",
}
ADDITIONAL_SOURCES = (
    "src/bayesian_phystwin_experiments/dlolab_slingshot_active_id.py",
    "src/bayesian_phystwin_experiments/dlolab_slingshot_task_probe_dev.py",
    "scripts/remote/run_dlolab_slingshot_active_id_particle.py",
    "tests/test_dlolab_slingshot_active_id.py",
    "docs/dlolab_slingshot_active_id_particle_source_v1.md",
)


def _source() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    old = read_record(PARENT / "lock.json")
    seal = read_record(PARENT / "model-bank/seal.json")
    if (
        old.get("artifact_id") != PARENT_LOCK_ID
        or seal.get("artifact_id") != BANK_ID
        or seal.get("lock_id") != PARENT_LOCK_ID
        or any(
            file_digest(ROOT / name) != digest
            for name, digest in old["source_sha256"].items()
        )
    ):
        raise ValueError("exact opened Slingshot model bank required")
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
    bank = load_native_bundle(PARENT / "model-bank", seal["bundle"])
    if bank["prefix"].shape != (27, 3, 4, 3) or bank["reward"].shape != (27, 7):
        raise ValueError("complete opened model bank required")
    return old, bank


def _development_history() -> np.ndarray:
    if any(
        file_digest(DEVELOPMENT / name) != digest
        for name, digest in DEVELOPMENT_FILE_SHA256.items()
    ):
        raise ValueError("sealed development root changed")
    lock = read_record(DEVELOPMENT / "lock.json")
    failure = read_record(DEVELOPMENT / "failure.json")
    seal = read_record(DEVELOPMENT / "development-bank/seal.json")
    if (
        lock.get("artifact_id") != DEVELOPMENT_LOCK_ID
        or failure.get("artifact_id") != DEVELOPMENT_FAILURE_ID
        or seal.get("artifact_id") != DEVELOPMENT_BANK_ID
        or failure.get("retry_authorized") is not False
        or failure.get("truth_future_generated") is not False
        or seal.get("new_prefix_qa") != [True, True, True, True]
    ):
        raise ValueError("terminal qualified development carrier required")
    data = load_native_bundle(DEVELOPMENT / "development-bank", seal["bundle"])
    if data["history"].shape != (4, 9, 3, 4, 3):
        raise ValueError("complete development histories required")
    return np.asarray(data["history"][3], dtype=np.float64)


def _worlds(indices: list[int]) -> list[dict[str, Any]]:
    rows = [particle_worlds()[index] for index in indices]
    return rows + [rows[-1]] * (8 - len(rows))


def _source_hashes(old: dict[str, Any]) -> dict[str, str]:
    paths = sorted(set(old["source_sha256"]) | set(ADDITIONAL_SOURCES))
    if any(not (ROOT / name).is_file() for name in paths):
        raise ValueError("complete registered active-identification source required")
    return {name: file_digest(ROOT / name) for name in paths}


def _validate(
    output: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    if output.resolve() != OUTPUT or output.is_symlink():
        raise ValueError("only the registered active-identification root is permitted")
    lock = read_record(output / "lock.json")
    attempt = read_record(ATTEMPT)
    old, bank = _source()
    if (
        lock.get("schema") != "dlolab-slingshot-active-id-particle-lock-v1"
        or lock.get("revision") != clean_revision(ROOT)
        or lock.get("source_sha256") != _source_hashes(old)
        or lock.get("protocol") != protocol()
        or lock.get("output_root") != str(OUTPUT)
        or lock.get("attempt_id") != attempt.get("artifact_id")
        or attempt.get("schema") != "dlolab-slingshot-active-id-attempt-v1"
        or attempt.get("revision") != lock.get("revision")
        or attempt.get("source_sha256") != lock.get("source_sha256")
        or attempt.get("protocol") != lock.get("protocol")
        or attempt.get("output_root") != str(OUTPUT)
        or lock.get("parent_lock_id") != PARENT_LOCK_ID
        or lock.get("bank_id") != BANK_ID
        or lock.get("development_bank_id") != DEVELOPMENT_BANK_ID
        or lock.get("retry_authorized") is not False
        or attempt.get("retry_authorized") is not False
        or lock.get("protected_data_read") is not False
    ):
        raise ValueError("clean frozen active-identification lock required")
    expected = array_digest(
        frontloaded_controls(np.asarray(old["controls"], dtype=np.float64), 0.7)
    )
    if lock.get("active_control_sha256") != expected:
        raise ValueError("active probe command changed")
    _development_history()
    return lock, old, bank


def _load_task(
    output: Path,
    lock: dict[str, Any],
    old: dict[str, Any],
    group: int,
    batch: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], bool]:
    task = particle_task(group, batch)
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
        raise ValueError("registered active particle task changed")
    data = load_native_bundle(directory, seal["bundle"])
    expected = frontloaded_controls(
        np.asarray(old["controls"], dtype=np.float64), ACTIVE_FRACTION
    )
    expected = np.repeat(expected[5:6], 8, axis=0)
    if array_digest(data["controls"]) != array_digest(expected):
        raise ValueError("active particle command changed")
    if set(data) != set(TRACE_NAMES + ("controls",)) or any(
        data[name].shape[:2] != (300, 8) for name in TRACE_NAMES
    ):
        raise ValueError("unexpected active particle prefix payload")
    native = seal["native"]
    if (
        native.get("native_steps") != 300
        or native.get("future_simulated") is not False
        or native.get("reward_scored") is not False
    ):
        raise ValueError("particle qualification crossed the truth boundary")
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


def _worker(output: Path, group: int, batch: int) -> None:
    lock, old, _ = _validate(output)
    task = particle_task(group, batch)
    directory = output / task["name"]
    directory.mkdir()
    claim = write_record(
        directory / "claim.json", {"lock_id": lock["artifact_id"], "task": task}
    )
    controls = frontloaded_controls(
        np.asarray(old["controls"], dtype=np.float64), ACTIVE_FRACTION
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
    _load_task(output, lock, old, group, batch)


def _launch(
    output: Path, old: dict[str, Any], group: int, batch: int
) -> None:
    task = particle_task(group, batch)
    command = [
        sys.executable,
        str(Path(__file__)),
        "--output",
        str(output),
        "--worker",
        "--group",
        str(group),
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
    if output.resolve() != OUTPUT or output.exists() or ATTEMPT.exists():
        raise ValueError("one fresh active-identification attempt required")
    revision = clean_revision(ROOT)
    old, bank = _source()
    sources = _source_hashes(old)
    attempt = write_record(
        ATTEMPT,
        {
            "schema": "dlolab-slingshot-active-id-attempt-v1",
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
            "schema": "dlolab-slingshot-active-id-particle-lock-v1",
            "revision": revision,
            "source_sha256": sources,
            "protocol": protocol(),
            "output_root": str(OUTPUT),
            "attempt_id": attempt["artifact_id"],
            "parent_lock_id": PARENT_LOCK_ID,
            "bank_id": BANK_ID,
            "development_lock_id": DEVELOPMENT_LOCK_ID,
            "development_failure_id": DEVELOPMENT_FAILURE_ID,
            "development_bank_id": DEVELOPMENT_BANK_ID,
            "development_file_sha256": DEVELOPMENT_FILE_SHA256,
            "active_control_sha256": array_digest(
                frontloaded_controls(
                    np.asarray(old["controls"], dtype=np.float64), ACTIVE_FRACTION
                )
            ),
            "retry_authorized": False,
            "protected_data_read": False,
        },
    )
    try:
        active = np.empty_like(bank["prefix"])
        active[9:18] = _development_history()
        qa: list[bool] = []
        seal_ids: list[str] = []
        for group in range(2):
            for batch in range(2):
                _launch(output, old, group, batch)
                seal, data, passed = _load_task(output, lock, old, group, batch)
                qa.append(passed)
                seal_ids.append(seal["artifact_id"])
                observed = prefix_observations(data)
                indices = particle_task(group, batch)["world_indices"]
                active[indices] = observed[: len(indices)]
        histories = np.stack([bank["prefix"], active])
        metrics = expected_value_screen(histories, bank["reward"])
        checks = {
            "all_new_prefixes_native_qualified": len(qa) == 4 and all(qa),
            **metrics["checks"],
        }
        directory = output / "particle-bank"
        directory.mkdir()
        bundle = write_native_bundle(
            directory,
            {"history": histories, "reward": bank["reward"]},
        )
        bank_seal = write_record(
            directory / "seal.json",
            {
                "lock_id": lock["artifact_id"],
                "new_prefix_seal_ids": seal_ids,
                "new_prefix_qa": qa,
                "bundle": bundle,
                "truth_probe_generated": False,
                "truth_future_generated": False,
            },
        )
        write_record(
            output / "result.json",
            {
                "schema": "dlolab-slingshot-active-id-particle-result-v1",
                "lock_id": lock["artifact_id"],
                "particle_bank_id": bank_seal["artifact_id"],
                "metrics": metrics,
                "checks": checks,
                "particle_value_gate_passed": bool(all(checks.values())),
                "continuous_truth_protocol_automatically_authorized": False,
                "truth_probe_generated": False,
                "truth_future_generated": False,
                "protected_data_read": False,
                "retry_authorized": False,
            },
        )
    except Exception as exc:
        write_record(
            output / "failure.json",
            {
                "schema": "dlolab-slingshot-active-id-particle-failure-v1",
                "lock_id": lock["artifact_id"],
                "error_type": type(exc).__name__,
                "error": str(exc),
                "truth_probe_generated": False,
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
    parser.add_argument("--group", type=int)
    parser.add_argument("--batch", type=int)
    args = parser.parse_args()
    if args.worker:
        if args.group is None or args.batch is None:
            raise ValueError("worker group and batch required")
        _worker(args.output, args.group, args.batch)
    else:
        if args.group is not None or args.batch is not None:
            raise ValueError("worker-only task arguments supplied")
        _run(args.output)


if __name__ == "__main__":
    main()
