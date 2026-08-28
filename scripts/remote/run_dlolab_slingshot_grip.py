#!/usr/bin/env python3
"""One CPU-only source screen of post-prefix native finger-force decisions."""

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
from bayesian_phystwin_experiments.dlolab_slingshot_grip import (
    controls,
    protocol,
    reference_checks,
    run_grip_world,
    validate_force_record,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(
    "/home/fpfaff/source-only/dlolab-benchmark-source-v1/grip-recovery-source-v1"
)
CONTACT_ROOT = OUTPUT.parent / "contact-realization-source-v1"
CONTACT_ID = "362175cce80bce8eb5409a02bcdd0476b376241525293d5aa7e9d05316631f67"
CONTACT_RESULT_ID = "09cd8e3f9ab6e879a872c06fb9ee039dd61b1d0b597ef656f4137114ee0c5d1a"
ADDITIONAL_SOURCES = (
    "src/bayesian_phystwin_experiments/dlolab_slingshot_grip.py",
    "scripts/remote/run_dlolab_slingshot_grip.py",
    "tests/test_dlolab_slingshot_grip.py",
    "docs/dlolab_slingshot_grip_recovery_source_v1.md",
)
SPEC = importlib.util.spec_from_file_location(
    "contact_source_runner", ROOT / "scripts/remote/run_dlolab_slingshot_contact.py"
)
assert SPEC is not None and SPEC.loader is not None
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)


def source():
    old, _, _ = base.source()
    contact = read_record(CONTACT_ROOT / "lock.json")
    result = read_record(CONTACT_ROOT / "result.json")
    if (
        contact["artifact_id"] != CONTACT_ID
        or result["artifact_id"] != CONTACT_RESULT_ID
        or result["lock_id"] != CONTACT_ID
        or any(
            file_digest(ROOT / p) != digest
            for p, digest in contact["source_sha256"].items()
        )
    ):
        raise ValueError("exact completed contact source required")
    references = {}
    for index in range(3):
        seal, data, qa = base.load_task(CONTACT_ROOT, contact, old, index)
        if not qa["qa_passed"]:
            raise ValueError("unqualified contact source reference")
        references[index] = (seal, data)
    return old, contact, references


def validate(output):
    if output.resolve() != OUTPUT:
        raise ValueError("only registered write-once root permitted")
    if (output / "failure.json").exists():
        raise ValueError("terminal retained failure; no retry")
    lock = read_record(output / "lock.json")
    if (
        lock["revision"] != clean_revision(ROOT)
        or lock["protocol"] != protocol()
        or lock["output_root"] != str(OUTPUT)
        or lock["contact_lock_id"] != CONTACT_ID
        or any(
            file_digest(ROOT / p) != digest
            for p, digest in lock["source_sha256"].items()
        )
    ):
        raise ValueError("clean frozen grip source/protocol required")
    old, _, references = source()
    if lock["reference_ids"] != [
        references[i][0]["artifact_id"] for i in range(3)
    ] or lock["cartesian_controls_sha256"] != array_digest(
        controls(np.asarray(old["controls"], dtype=np.float64))
    ):
        raise ValueError("source reference or controls changed")
    return lock, old, references


def load_task(output, lock, old, references, index):
    changed = {
        **old,
        "controls": controls(np.asarray(old["controls"], dtype=np.float64)).tolist(),
    }
    seal, data, qa = base.load_task(output, lock, changed, index)
    validate_force_record(seal["native"]["grip_schedule"])
    reference_seal, reference = references[index]
    replay = reference_checks(
        data,
        reference,
        seal["native"]["native_cumulative_reward"],
        reference_seal["native"]["native_cumulative_reward"],
    )
    return (
        seal,
        data,
        {
            "native_qa": qa,
            "fallback_and_prefix": replay,
            "passed": qa["qa_passed"] and replay["passed"],
        },
    )


def gate(output, lock, old, references, index):
    seal, _, qa = load_task(output, lock, old, references, index)
    return {
        "schema": "dlolab-slingshot-grip-native-gate-v1",
        "lock_id": lock["artifact_id"],
        "source_seal_id": seal["artifact_id"],
        "index": index,
        **qa,
    }


def require_previous(output, lock, old, references, index):
    for earlier in RUN_ORDER[: RUN_ORDER.index(index)]:
        stored = read_record(output / f"gate-{earlier}.json")
        computed = gate(output, lock, old, references, earlier)
        if not computed["passed"] or any(
            stored[k] != value for k, value in computed.items()
        ):
            raise ValueError("next native world requires rederived passing earlier QA")


def worker(output, index):
    spec = task(index)
    lock, old, references = validate(output)
    require_previous(output, lock, old, references, index)
    directory = output / spec["name"]
    directory.mkdir()
    claim = write_record(
        directory / "claim.json", {"lock_id": lock["artifact_id"], "task": spec}
    )
    data, native = run_grip_world(
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
            f"grip world {index} exited {completed.returncode}; no retry"
        )
    print(f"completed grip world {index}", flush=True)


def run(output):
    if output.resolve() != OUTPUT:
        raise ValueError("only registered write-once root permitted")
    revision = clean_revision(ROOT)
    old, contact, references = source()
    paths = sorted(set(contact["source_sha256"]) | set(ADDITIONAL_SOURCES))
    output.mkdir()
    lock = write_record(
        output / "lock.json",
        {
            "schema": "dlolab-slingshot-grip-lock-v1",
            "revision": revision,
            "source_sha256": {p: file_digest(ROOT / p) for p in paths},
            "protocol": protocol(),
            "output_root": str(OUTPUT),
            "contact_lock_id": CONTACT_ID,
            "reference_ids": [references[i][0]["artifact_id"] for i in range(3)],
            "cartesian_controls_sha256": array_digest(
                controls(np.asarray(old["controls"], dtype=np.float64))
            ),
        },
    )
    stage = "native-grip-worlds"
    try:
        for done, index in enumerate(RUN_ORDER, start=1):
            launch(output, old, index)
            qa = gate(output, lock, old, references, index)
            write_record(output / f"gate-{index}.json", qa)
            if not qa["passed"]:
                write_record(
                    output / "result.json",
                    {
                        "lock_id": lock["artifact_id"],
                        "terminal_stage": "native-fallback-qa",
                        "native_worlds_completed": done,
                        "source_gate_passed": False,
                        "method_evaluation_authorized": False,
                        "protected_data_read": False,
                    },
                )
                print("native fallback/prefix QA failed; no further worlds", flush=True)
                return
        histories, rewards, source_ids, qa_rows = [], [], [], []
        for index in range(3):
            seal, data, qa = load_task(output, lock, old, references, index)
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
        metrics = information_value(arrays["prefix"], arrays["reward"])
        result = write_record(
            output / "result.json",
            {
                "schema": "dlolab-slingshot-grip-source-result-v1",
                "lock_id": lock["artifact_id"],
                "source_bank_id": seal["artifact_id"],
                "native_worlds_completed": 3,
                "metrics": metrics,
                "source_gate_passed": metrics["source_information_value_passed"],
                "method_evaluation_authorized": False,
                "protected_data_read": False,
                "claim_boundary": "Contact-source development with a new causal force action; not independent control performance or safety calibration.",
            },
        )
        print(
            f"grip source gate={result['source_gate_passed']}; {result['artifact_id']}",
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
