#!/usr/bin/env python3
"""Verify the frozen Slingshot active-identification particle screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from bayesian_phystwin_experiments.dlolab_native import array_digest, file_digest
from bayesian_phystwin_experiments.dlolab_regret_artifacts import read_record
from bayesian_phystwin_experiments.dlolab_slingshot_active_id import (
    ACTIVE_FRACTION,
    expected_value_screen,
    particle_task,
    protocol,
)
from bayesian_phystwin_experiments.dlolab_slingshot_batch import TRACE_NAMES
from bayesian_phystwin_experiments.dlolab_slingshot_belief import prefix_observations
from bayesian_phystwin_experiments.dlolab_slingshot_process import load_native_bundle
from bayesian_phystwin_experiments.dlolab_slingshot_task_probe_dev import (
    frontloaded_controls,
)

ROOT = Path(__file__).resolve().parents[1]
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
FORBIDDEN_STAGE_NAMES = (
    "truth-probes",
    "truth-futures",
    "decisions",
    "score",
    "continuous-truth",
)
Array: TypeAlias = NDArray[Any]


def _load_task(
    output: Path,
    lock: dict[str, Any],
    group: int,
    batch: int,
) -> tuple[Array, str, bool]:
    task = particle_task(group, batch)
    directory = output / task["name"]
    claim = read_record(directory / "claim.json")
    seal = read_record(directory / "seal.json")
    native = seal.get("native", {})
    if (
        claim.get("lock_id") != lock["artifact_id"]
        or claim.get("task") != task
        or seal.get("claim_id") != claim["artifact_id"]
        or seal.get("lock_id") != lock["artifact_id"]
        or seal.get("task") != task
        or native.get("native_steps") != 300
        or native.get("future_simulated") is not False
        or native.get("reward_scored") is not False
    ):
        raise ValueError("invalid prefix-only particle task")
    data = load_native_bundle(directory, seal["bundle"])
    if set(data) != set(TRACE_NAMES + ("controls",)) or any(
        data[name].shape[:2] != (300, 8) for name in TRACE_NAMES
    ):
        raise ValueError("particle task payload changed")
    fixed_error = float(
        np.max(
            np.abs(
                data["rod_pos_m"][:, :, [0, 1, 10, 11]]
                - data["rod_pos_m"][:1, :, [0, 1, 10, 11]]
            )
        )
    )
    duplicate_error = (
        0.0
        if batch == 0
        else max(
            float(np.max(np.abs(data[name] - data[name][:, :1])))
            for name in (
                "rod_pos_m",
                "sphere_pos_m",
                "cube_pos_m",
                "gripper_pos_m",
            )
        )
    )
    passed = fixed_error <= 1e-9 and duplicate_error <= 0.0005
    indices = task["world_indices"]
    return prefix_observations(data)[: len(indices)], seal["artifact_id"], passed


def verify(output: Path) -> dict[str, Any]:
    if output.resolve() != OUTPUT or output.is_symlink():
        raise ValueError("registered Slingshot particle root required")
    if any((output / name).exists() for name in FORBIDDEN_STAGE_NAMES):
        raise ValueError("continuous-truth stage exists in stopped particle root")
    if (output / "failure.json").exists():
        raise ValueError("particle result and failure cannot coexist")

    lock = read_record(output / "lock.json")
    attempt = read_record(ATTEMPT)
    result = read_record(output / "result.json")
    bank_seal = read_record(output / "particle-bank" / "seal.json")
    if (
        lock.get("schema") != "dlolab-slingshot-active-id-particle-lock-v1"
        or attempt.get("schema") != "dlolab-slingshot-active-id-attempt-v1"
        or result.get("schema")
        != "dlolab-slingshot-active-id-particle-result-v1"
        or lock.get("attempt_id") != attempt["artifact_id"]
        or attempt.get("revision") != lock.get("revision")
        or attempt.get("source_sha256") != lock.get("source_sha256")
        or attempt.get("protocol") != lock.get("protocol")
        or lock.get("protocol") != protocol()
        or attempt.get("output_root") != str(OUTPUT)
        or lock.get("output_root") != str(OUTPUT)
        or result.get("lock_id") != lock["artifact_id"]
        or bank_seal.get("lock_id") != lock["artifact_id"]
        or result.get("particle_bank_id") != bank_seal["artifact_id"]
        or any(
            record.get("retry_authorized") is not False
            or record.get("protected_data_read") is not False
            for record in (attempt, lock, result)
        )
        or result.get("continuous_truth_protocol_automatically_authorized")
        is not False
        or result.get("truth_probe_generated") is not False
        or result.get("truth_future_generated") is not False
        or bank_seal.get("truth_probe_generated") is not False
        or bank_seal.get("truth_future_generated") is not False
    ):
        raise ValueError("invalid Slingshot particle custody")
    if any(
        file_digest(ROOT / name) != digest
        for name, digest in lock["source_sha256"].items()
    ):
        raise ValueError("frozen particle source changed")

    parent_lock = read_record(PARENT / "lock.json")
    parent_seal = read_record(PARENT / "model-bank" / "seal.json")
    parent = load_native_bundle(PARENT / "model-bank", parent_seal["bundle"])
    if (
        parent_lock.get("artifact_id") != lock.get("parent_lock_id")
        or parent_seal.get("artifact_id") != lock.get("bank_id")
        or parent_seal.get("lock_id") != parent_lock["artifact_id"]
        or parent["prefix"].shape != (27, 3, 4, 3)
        or parent["reward"].shape != (27, 7)
    ):
        raise ValueError("registered parent particle bank changed")
    expected_control = frontloaded_controls(
        np.asarray(parent_lock["controls"], dtype=np.float64), ACTIVE_FRACTION
    )
    if array_digest(expected_control) != lock.get("active_control_sha256"):
        raise ValueError("active command changed")

    development_hashes = lock.get("development_file_sha256", {})
    if any(
        file_digest(DEVELOPMENT / name) != digest
        for name, digest in development_hashes.items()
    ):
        raise ValueError("sealed development carrier changed")
    development_seal = read_record(DEVELOPMENT / "development-bank" / "seal.json")
    development = load_native_bundle(
        DEVELOPMENT / "development-bank", development_seal["bundle"]
    )
    if (
        development_seal.get("artifact_id") != lock.get("development_bank_id")
        or development_seal.get("new_prefix_qa") != [True, True, True, True]
        or development["history"].shape != (4, 9, 3, 4, 3)
    ):
        raise ValueError("development history changed")

    active = np.empty_like(parent["prefix"])
    active[9:18] = development["history"][3]
    task_ids: list[str] = []
    qa: list[bool] = []
    for group in range(2):
        for batch in range(2):
            history, task_id, passed = _load_task(output, lock, group, batch)
            indices = particle_task(group, batch)["world_indices"]
            active[indices] = history
            task_ids.append(task_id)
            qa.append(passed)

    bank = load_native_bundle(output / "particle-bank", bank_seal["bundle"])
    expected_history = np.stack([parent["prefix"], active])
    if (
        set(bank) != {"history", "reward"}
        or not np.array_equal(bank["history"], expected_history)
        or not np.array_equal(bank["reward"], parent["reward"])
        or bank_seal.get("new_prefix_seal_ids") != task_ids
        or bank_seal.get("new_prefix_qa") != qa
        or qa != [True, True, True, True]
    ):
        raise ValueError("full particle bank does not reconstruct")

    metrics = expected_value_screen(bank["history"], bank["reward"])
    checks = {"all_new_prefixes_native_qualified": True, **metrics["checks"]}
    passed = bool(all(checks.values()))
    if (
        result.get("metrics") != metrics
        or result.get("checks") != checks
        or result.get("particle_value_gate_passed") is not passed
        or passed is not False
    ):
        raise ValueError("particle value arithmetic or frozen decision changed")
    return {
        "schema": "dlolab-slingshot-active-id-verification-v1",
        "attempt_id": attempt["artifact_id"],
        "lock_id": lock["artifact_id"],
        "particle_bank_id": bank_seal["artifact_id"],
        "result_id": result["artifact_id"],
        "new_prefix_count": len(task_ids),
        "all_new_prefixes_native_qualified": True,
        "particle_value_gate_passed": False,
        "truth_stage_verified_absent": True,
        "native_replay_performed": False,
        "protected_data_read": False,
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(verify(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
