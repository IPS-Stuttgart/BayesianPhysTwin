#!/usr/bin/env python3
"""Fresh-process and slot-permutation numerical audit of known native controls."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np

from bayesian_phystwin_experiments.dlolab_benchmark import write_native_bundle
from bayesian_phystwin_experiments.dlolab_native import array_digest, file_digest
from bayesian_phystwin_experiments.dlolab_regret_artifacts import (
    clean_revision,
    read_record,
    write_record,
)
from bayesian_phystwin_experiments.dlolab_slingshot_belief import native_qa
from bayesian_phystwin_experiments.dlolab_slingshot_cmaes import worker_environment
from bayesian_phystwin_experiments.dlolab_slingshot_process import load_native_bundle
from bayesian_phystwin_experiments.dlolab_slingshot_repeatability import (
    controls,
    protocol,
    run_repeat,
    summarize,
    task,
    validate_force_record,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(
    "/home/fpfaff/source-only/dlolab-benchmark-source-v1/numerical-repeatability-v1"
)
STOPPED_ROOT = OUTPUT.parent / "contact-path-source-v1"
STOPPED_ID = "baa9b3d5baca3853255b191d06f38f1797479409c192556c318dd49f54d6eaa9"
SOURCES = (
    "src/bayesian_phystwin_experiments/dlolab_slingshot_repeatability.py",
    "scripts/remote/run_dlolab_slingshot_repeatability.py",
    "tests/test_dlolab_slingshot_repeatability.py",
    "docs/dlolab_slingshot_numerical_repeatability_v1.md",
)
SPEC = importlib.util.spec_from_file_location(
    "contact_path_source", ROOT / "scripts/remote/run_dlolab_slingshot_path.py"
)
assert SPEC is not None and SPEC.loader is not None
parent = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(parent)


def source():
    old, _, grip_lock, grip_result, references = parent.source()
    stopped = read_record(STOPPED_ROOT / "result.json")
    if (
        stopped["artifact_id"] != STOPPED_ID
        or stopped["terminal_stage"] != "native-reference-qa"
        or stopped["source_gate_passed"] is not False
    ):
        raise ValueError("exact terminal path result required; no reopening")
    return old, grip_lock, grip_result, references


def validate(output):
    if output.resolve() != OUTPUT:
        raise ValueError("only registered write-once audit root permitted")
    if (output / "failure.json").exists():
        raise ValueError("terminal audit failure; no retry")
    lock = read_record(output / "lock.json")
    if (
        lock["revision"] != clean_revision(ROOT)
        or lock["protocol"] != protocol()
        or lock["output_root"] != str(OUTPUT)
        or lock["stopped_result_id"] != STOPPED_ID
    ):
        raise ValueError("frozen clean numerical audit required")
    if any(
        file_digest(ROOT / p) != value for p, value in lock["source_sha256"].items()
    ):
        raise ValueError("frozen numerical audit bytes changed")
    old, grip_lock, grip_result, references = source()
    if (
        lock["grip_lock_id"] != grip_lock["artifact_id"]
        or lock["grip_result_id"] != grip_result["artifact_id"]
        or lock["reference_ids"] != [references[i][0]["artifact_id"] for i in range(3)]
    ):
        raise ValueError("known-control source lineage changed")
    if lock["controls_sha256"] != [
        array_digest(controls(np.asarray(old["controls"], dtype=np.float64), i))
        for i in range(15)
    ]:
        raise ValueError("registered command schedule changed")
    return lock, old, references


def load_task(output, lock, old, references, index):
    spec = task(index)
    directory = output / spec["name"]
    seal, claim = (
        read_record(directory / "seal.json"),
        read_record(directory / "claim.json"),
    )
    if (
        seal["lock_id"] != lock["artifact_id"]
        or claim["lock_id"] != lock["artifact_id"]
        or seal["claim_id"] != claim["artifact_id"]
        or seal["task"] != spec
        or claim["task"] != spec
    ):
        raise ValueError("registered numerical task identity changed")
    data = load_native_bundle(directory, seal["bundle"])
    expected = controls(np.asarray(old["controls"], dtype=np.float64), index)
    validate_force_record(seal["native"]["grip_schedule"], index)
    native_reference = references[spec["contact_index"]][0]["native"]
    if any(
        seal["native"][key] != native_reference[key]
        for key in ("world_realization", "contact_realization")
    ):
        raise ValueError("native world/material realization changed")
    qa = native_qa(data, seal["native"], expected)
    admission = {
        "legacy_native_qa": qa,
        "measurement_admitted": qa["checks"]["fixed_endpoints"],
        "duplicate_checks_not_used_to_censor_replay_variation": True,
    }
    return seal, data, admission


def gate(output, lock, old, references, index):
    seal, _, admission = load_task(output, lock, old, references, index)
    return {
        "schema": "dlolab-numerical-measurement-admission-v1",
        "lock_id": lock["artifact_id"],
        "source_seal_id": seal["artifact_id"],
        "index": index,
        **admission,
    }


def require_previous(output, lock, old, references, index):
    for earlier in range(index):
        expected = gate(output, lock, old, references, earlier)
        stored = read_record(output / f"admission-{earlier:02d}.json")
        if not expected["measurement_admitted"] or any(
            stored[k] != value for k, value in expected.items()
        ):
            raise ValueError("earlier native admission must rederive before next task")


def worker(output, index):
    spec = task(index)
    lock, old, references = validate(output)
    require_previous(output, lock, old, references, index)
    directory = output / spec["name"]
    directory.mkdir()
    claim = write_record(
        directory / "claim.json", {"lock_id": lock["artifact_id"], "task": spec}
    )
    data, native = run_repeat(
        Path(old["assets_root"]) / "upstream",
        directory,
        controls(np.asarray(old["controls"], dtype=np.float64), index),
        index,
    )
    write_record(
        directory / "seal.json",
        {
            "lock_id": lock["artifact_id"],
            "claim_id": claim["artifact_id"],
            "task": spec,
            "native": native,
            "bundle": write_native_bundle(directory, data),
        },
    )
    load_task(output, lock, old, references, index)


def launch(output, old, index):
    command = [
        sys.executable,
        str(Path(__file__)),
        "--output",
        str(output),
        "--worker",
        str(index),
    ]
    with (output / f"{task(index)['name']}.log").open("x") as handle:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=worker_environment(old["screen"]["source"]["controller"]["runtime"]),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"numerical task {index} exited {completed.returncode}; no retry"
        )
    print(f"completed numerical batch {index + 1}/15", flush=True)


def run(output):
    if output.resolve() != OUTPUT:
        raise ValueError("only registered write-once audit root permitted")
    revision = clean_revision(ROOT)
    old, grip_lock, grip_result, references = source()
    paths = sorted(
        set(grip_lock["source_sha256"]) | set(parent.ADDITIONAL_SOURCES) | set(SOURCES)
    )
    output.mkdir()
    lock = write_record(
        output / "lock.json",
        {
            "schema": "dlolab-slingshot-numerical-lock-v1",
            "revision": revision,
            "source_sha256": {p: file_digest(ROOT / p) for p in paths},
            "protocol": protocol(),
            "output_root": str(OUTPUT),
            "stopped_result_id": STOPPED_ID,
            "grip_lock_id": grip_lock["artifact_id"],
            "grip_result_id": grip_result["artifact_id"],
            "reference_ids": [references[i][0]["artifact_id"] for i in range(3)],
            "controls_sha256": [
                array_digest(controls(np.asarray(old["controls"], dtype=np.float64), i))
                for i in range(15)
            ],
        },
    )
    completed, attempted, stage = 0, 0, "native-numerical-batches"
    try:
        for index in range(15):
            attempted += 1
            launch(output, old, index)
            completed += 1
            admission = gate(output, lock, old, references, index)
            write_record(output / f"admission-{index:02d}.json", admission)
            if not admission["measurement_admitted"]:
                write_record(
                    output / "result.json",
                    {
                        "lock_id": lock["artifact_id"],
                        "terminal_stage": "native-measurement-admission",
                        "planned_batches": 15,
                        "completed_batches": completed,
                        "admitted_batches": completed - 1,
                        "unrun_batches": 15 - attempted,
                        "observed_numerical_budget_passed": False,
                        "new_controller_evaluation_authorized": False,
                        "protected_data_read": False,
                    },
                )
                return
        stage = "numerical-variation-summary"
        records, source_ids = [], []
        for index in range(15):
            seal, data, admission = load_task(output, lock, old, references, index)
            records.append(
                {
                    "task": task(index),
                    "arrays": data,
                    "reward": seal["native"]["native_cumulative_reward"],
                    "measurement_admitted": admission["measurement_admitted"],
                }
            )
            source_ids.append(seal["artifact_id"])
        metrics = summarize(records)
        result = write_record(
            output / "result.json",
            {
                "schema": "dlolab-slingshot-numerical-result-v1",
                "lock_id": lock["artifact_id"],
                "source_seal_ids": source_ids,
                "planned_batches": 15,
                "completed_batches": 15,
                "admitted_batches": 15,
                "unrun_batches": 0,
                "native_trajectories": 120,
                "metrics": metrics,
                "observed_numerical_budget_passed": metrics[
                    "observed_numerical_budget_passed"
                ],
                "new_controller_evaluation_authorized": False,
                "protected_data_read": False,
            },
        )
        print(
            f"observed numerical budget={result['observed_numerical_budget_passed']}; {result['artifact_id']}",
            flush=True,
        )
    except Exception as exc:
        write_record(
            output / "failure.json",
            {
                "lock_id": lock["artifact_id"],
                "stage": stage,
                "planned_batches": 15,
                "completed_batches": completed,
                "worker_invocations_attempted": attempted,
                "unrun_batches": 15 - attempted,
                "error": f"{type(exc).__name__}: {exc}",
                "retry_authorized": False,
                "protected_data_read": False,
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--worker", type=int, choices=range(15))
    args = parser.parse_args()
    if args.worker is None:
        run(args.output)
    else:
        worker(args.output, args.worker)
