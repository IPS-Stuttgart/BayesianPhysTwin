#!/usr/bin/env python3
"""Bounded native contact-path recovery screen; local CPU source only."""

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
from bayesian_phystwin_experiments.dlolab_slingshot_batch import TRACE_NAMES
from bayesian_phystwin_experiments.dlolab_slingshot_belief import prefix_observations
from bayesian_phystwin_experiments.dlolab_slingshot_cmaes import worker_environment
from bayesian_phystwin_experiments.dlolab_slingshot_contact import (
    RUN_ORDER,
    information_value,
    task,
)
from bayesian_phystwin_experiments.dlolab_slingshot_path import (
    compare_previous_policy,
    controls,
    protocol,
    reference_checks,
    run_path_world,
    validate_force_record,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(
    "/home/fpfaff/source-only/dlolab-benchmark-source-v1/contact-path-source-v1"
)
GRIP_ROOT = OUTPUT.parent / "grip-recovery-source-v1"
GRIP_LOCK_ID = "5a6e51e361289121f3b07ac6a257ce5fc522689df63d288172c0e60623b1935b"
GRIP_RESULT_ID = "b97ff64f95db0cb76aecde90dc1eec504e9e5e8422fd95c280c0caac909651c2"
ADDITIONAL_SOURCES = (
    "src/bayesian_phystwin_experiments/dlolab_slingshot_path.py",
    "scripts/remote/run_dlolab_slingshot_path.py",
    "tests/test_dlolab_slingshot_path.py",
    "docs/dlolab_slingshot_contact_path_source_v1.md",
)
SPEC = importlib.util.spec_from_file_location(
    "grip_source_runner", ROOT / "scripts/remote/run_dlolab_slingshot_grip.py"
)
assert SPEC is not None and SPEC.loader is not None
parent = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(parent)


def source():
    old, _, contacts = parent.source()
    grip = read_record(GRIP_ROOT / "lock.json")
    result = read_record(GRIP_ROOT / "result.json")
    if (
        grip["artifact_id"] != GRIP_LOCK_ID
        or result["artifact_id"] != GRIP_RESULT_ID
        or result["lock_id"] != GRIP_LOCK_ID
        or any(
            file_digest(ROOT / p) != digest
            for p, digest in grip["source_sha256"].items()
        )
    ):
        raise ValueError("exact completed grip source and implementation required")
    references = {}
    for index in range(3):
        seal, data, qa = parent.load_task(GRIP_ROOT, grip, old, contacts, index)
        if not qa["passed"]:
            raise ValueError("unqualified grip source reference")
        references[index] = (seal, data)
    return old, contacts, grip, result, references


def validate(output):
    if output.resolve() != OUTPUT:
        raise ValueError("only the registered write-once root is permitted")
    if (output / "failure.json").exists():
        raise ValueError("retained terminal failure; no retry")
    lock = read_record(output / "lock.json")
    if (
        lock["revision"] != clean_revision(ROOT)
        or lock["protocol"] != protocol()
        or lock["output_root"] != str(OUTPUT)
        or lock["grip_lock_id"] != GRIP_LOCK_ID
        or lock["grip_result_id"] != GRIP_RESULT_ID
        or any(
            file_digest(ROOT / p) != digest
            for p, digest in lock["source_sha256"].items()
        )
    ):
        raise ValueError("clean frozen contact-path source and protocol required")
    old, contacts, _, result, references = source()
    if lock["reference_ids"] != [references[i][0]["artifact_id"] for i in range(3)]:
        raise ValueError("grip reference identities changed")
    if lock["controls_sha256"] != array_digest(
        controls(np.asarray(old["controls"], dtype=np.float64))
    ):
        raise ValueError("registered contact-path controls changed")
    return lock, old, contacts, result, references


def load_task(output, lock, old, contacts, references, index):
    changed = {
        **old,
        "controls": controls(np.asarray(old["controls"], dtype=np.float64)).tolist(),
    }
    seal, data, qa = parent.base.load_task(output, lock, changed, index)
    validate_force_record(seal["native"]["grip_schedule"])
    contact_seal, contact = contacts[index]
    previous_seal, previous = references[index]
    replay = reference_checks(
        data,
        contact,
        previous,
        seal["native"]["native_cumulative_reward"],
        contact_seal["native"]["native_cumulative_reward"],
        previous_seal["native"]["native_cumulative_reward"],
    )
    return (
        seal,
        data,
        {
            "native_qa": qa,
            "reference_replay": replay,
            "passed": qa["qa_passed"] and replay["passed"],
        },
    )


def gate(output, lock, old, contacts, references, index):
    seal, _, qa = load_task(output, lock, old, contacts, references, index)
    return {
        "schema": "dlolab-slingshot-contact-path-native-gate-v1",
        "lock_id": lock["artifact_id"],
        "source_seal_id": seal["artifact_id"],
        "index": index,
        **qa,
    }


def require_previous(output, lock, old, contacts, references, index):
    for earlier in RUN_ORDER[: RUN_ORDER.index(index)]:
        stored = read_record(output / f"gate-{earlier}.json")
        computed = gate(output, lock, old, contacts, references, earlier)
        if not computed["passed"] or any(
            stored[k] != value for k, value in computed.items()
        ):
            raise ValueError("next world requires rederived passing earlier native QA")


def worker(output, index):
    spec = task(index)
    lock, old, contacts, _, references = validate(output)
    require_previous(output, lock, old, contacts, references, index)
    directory = output / spec["name"]
    directory.mkdir()
    claim = write_record(
        directory / "claim.json", {"lock_id": lock["artifact_id"], "task": spec}
    )
    data, native = run_path_world(
        Path(old["assets_root"]) / "upstream",
        directory,
        controls(np.asarray(old["controls"], dtype=np.float64)),
        index,
    )
    write_record(
        directory / "seal.json",
        {
            "lock_id": lock["artifact_id"],
            "claim_id": claim["artifact_id"],
            "task": spec,
            "bundle": write_native_bundle(directory, data),
            "native": native,
        },
    )
    load_task(output, lock, old, contacts, references, index)


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
            f"contact-path world {index} exited {completed.returncode}; no retry"
        )
    print(f"completed contact-path world {index}", flush=True)


def run(output):
    if output.resolve() != OUTPUT:
        raise ValueError("only the registered write-once root is permitted")
    revision = clean_revision(ROOT)
    old, contacts, grip, previous_result, references = source()
    paths = sorted(set(grip["source_sha256"]) | set(ADDITIONAL_SOURCES))
    output.mkdir()
    lock = write_record(
        output / "lock.json",
        {
            "schema": "dlolab-slingshot-contact-path-lock-v1",
            "revision": revision,
            "source_sha256": {p: file_digest(ROOT / p) for p in paths},
            "protocol": protocol(),
            "output_root": str(OUTPUT),
            "grip_lock_id": GRIP_LOCK_ID,
            "grip_result_id": GRIP_RESULT_ID,
            "reference_ids": [references[i][0]["artifact_id"] for i in range(3)],
            "controls_sha256": array_digest(
                controls(np.asarray(old["controls"], dtype=np.float64))
            ),
        },
    )
    stage = "native-contact-path-worlds"
    try:
        for done, index in enumerate(RUN_ORDER, start=1):
            launch(output, old, index)
            qa = gate(output, lock, old, contacts, references, index)
            write_record(output / f"gate-{index}.json", qa)
            if not qa["passed"]:
                write_record(
                    output / "result.json",
                    {
                        "lock_id": lock["artifact_id"],
                        "terminal_stage": "native-reference-qa",
                        "native_worlds_completed": done,
                        "source_gate_passed": False,
                        "method_evaluation_authorized": False,
                        "protected_data_read": False,
                    },
                )
                print("native replay or QA failed; no further worlds", flush=True)
                return
        histories, rewards, source_ids, qa_rows = [], [], [], []
        for index in range(3):
            seal, data, qa = load_task(output, lock, old, contacts, references, index)
            histories.append(
                prefix_observations(
                    {k: v[:300] for k, v in data.items() if k in TRACE_NAMES}
                )[0]
            )
            rewards.append([m["native_reward"] for m in qa["native_qa"]["metrics"][:7]])
            source_ids.append(seal["artifact_id"])
            qa_rows.append(qa)
        directory = output / "source-bank"
        directory.mkdir()
        arrays = {"prefix": np.stack(histories), "reward": np.asarray(rewards)}
        seal = write_record(
            directory / "seal.json",
            {
                "lock_id": lock["artifact_id"],
                "source_seal_ids": source_ids,
                "bundle": write_native_bundle(directory, arrays),
                "native_qa": qa_rows,
            },
        )
        stage = "source-information-value"
        metrics = compare_previous_policy(
            information_value(arrays["prefix"], arrays["reward"]), previous_result
        )
        result = write_record(
            output / "result.json",
            {
                "schema": "dlolab-slingshot-contact-path-source-result-v1",
                "lock_id": lock["artifact_id"],
                "source_bank_id": seal["artifact_id"],
                "native_worlds_completed": 3,
                "metrics": metrics,
                "source_gate_passed": metrics["source_information_value_passed"],
                "method_evaluation_authorized": False,
                "protected_data_read": False,
                "claim_boundary": "Adaptive source design with new causal loading paths; not independent control or safety evidence.",
            },
        )
        print(
            f"contact-path source gate={result['source_gate_passed']}; {result['artifact_id']}",
            flush=True,
        )
    except Exception as exc:
        write_record(
            output / "failure.json",
            {
                "lock_id": lock["artifact_id"],
                "stage": stage,
                "error": f"{type(exc).__name__}: {exc}",
                "retry_authorized": False,
                "protected_data_read": False,
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--worker", type=int, choices=range(3))
    args = parser.parse_args()
    if args.worker is None:
        run(args.output)
    else:
        worker(args.output, args.worker)
