#!/usr/bin/env python3
"""Run the frozen later-prefix/final-motion source screen, CPU only."""

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
from bayesian_phystwin_experiments.dlolab_slingshot_belief_native import (
    run_registered_worlds,
)
from bayesian_phystwin_experiments.dlolab_slingshot_cmaes import worker_environment
from bayesian_phystwin_experiments.dlolab_slingshot_late import (
    controls,
    information_value,
    native_checks,
    observations,
    protocol,
    repeat_checks,
    task,
)
from bayesian_phystwin_experiments.dlolab_slingshot_probe import material_information
from bayesian_phystwin_experiments.dlolab_slingshot_process import load_native_bundle

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(
    "/home/fpfaff/source-only/dlolab-benchmark-source-v1/late-branch-value-source-v1"
)
NUMERICAL = OUTPUT.parent / "numerical-repeatability-v1/result.json"
NUMERICAL_ID = "f9cf9969ae66d244408971700f6f1d9c0b6b6167df1e78afc24cb16bcd45b0fe"
SOURCES = (
    "src/bayesian_phystwin_experiments/dlolab_slingshot_late.py",
    "scripts/remote/run_dlolab_slingshot_late.py",
    "tests/test_dlolab_slingshot_late.py",
    "docs/dlolab_slingshot_late_branch_source_v1.md",
    "src/bayesian_phystwin_experiments/dlolab_slingshot_probe.py",
)
SPEC = importlib.util.spec_from_file_location(
    "contact_source", ROOT / "scripts/remote/run_dlolab_slingshot_contact.py"
)
assert SPEC is not None and SPEC.loader is not None
parent = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(parent)


def source():
    old, nominal, nominal_rewards = parent.source()
    numerical = read_record(NUMERICAL)
    if (
        numerical["artifact_id"] != NUMERICAL_ID
        or numerical["admitted_batches"] != 15
        or numerical["observed_numerical_budget_passed"] is not True
    ):
        raise ValueError(
            "exact observed-budget audit required, without changing its scope"
        )
    bank = read_record(parent.PARENT / "model-bank/seal.json")
    references = []
    for index in range(9, 18):
        if index == 13:
            references.append((parent.REFERENCE_ID, nominal, nominal_rewards))
            continue
        directory = parent.PARENT / f"particle-{index:02d}"
        seal = read_record(directory / "seal.json")
        if (
            seal["artifact_id"] != bank["parents"][index]["seal_id"]
            or seal["lock_id"] != parent.PARENT_ID
            or seal["task"]["kind"] != "particle"
            or seal["task"]["index"] != index
        ):
            raise ValueError("opened source particle identity changed")
        data = load_native_bundle(directory, seal["bundle"])
        if not native_qa(data, seal["native"], np.asarray(old["controls"]))[
            "qa_passed"
        ]:
            raise ValueError("original source qualification changed")
        references.append(
            (seal["artifact_id"], data, seal["native"]["native_cumulative_reward"])
        )
    return old, references


def source_information(references):
    history = np.stack(
        [
            observations({k: data[k][:500] for k in ("rod_pos_m", "sphere_pos_m")})[5]
            for _, data, _ in references
        ]
    )
    information = material_information(history)
    return {
        **information,
        "source_bank_authorized": information["whitened_stretching_secant_norm"] >= 1,
        "new_final_motion_reward_read": False,
    }


def validate(output):
    if output.resolve() != OUTPUT:
        raise ValueError("only registered write-once root permitted")
    if (output / "failure.json").exists() or (output / "result.json").exists():
        raise ValueError("terminal late-branch study; no retry")
    lock = read_record(output / "lock.json")
    if (
        lock["revision"] != clean_revision(ROOT)
        or lock["protocol"] != protocol()
        or lock["output_root"] != str(OUTPUT)
        or lock["parent_lock_id"] != parent.PARENT_ID
        or lock["parent_bank_id"] != parent.BANK_ID
        or lock["numerical_audit_id"] != NUMERICAL_ID
        or any(file_digest(ROOT / p) != v for p, v in lock["source_sha256"].items())
    ):
        raise ValueError("clean frozen later-branch implementation required")
    old, references = source()
    if lock["reference_ids"] != [r[0] for r in references] or lock[
        "controls_sha256"
    ] != array_digest(controls(np.asarray(old["controls"]))):
        raise ValueError("registered source reference/control identity changed")
    required = source_information(references)
    stored = read_record(output / "prefix-information.json")
    if (
        stored["lock_id"] != lock["artifact_id"]
        or not required["source_bank_authorized"]
        or any(stored[k] != value for k, value in required.items())
    ):
        raise ValueError(
            "passing prefix information must rederive before native execution"
        )
    return lock, old, references


def load_task(output, lock, old, references, index):
    spec = task(index)
    directory = output / spec["name"]
    claim, seal = (
        read_record(directory / "claim.json"),
        read_record(directory / "seal.json"),
    )
    if (
        seal["claim_id"] != claim["artifact_id"]
        or seal["task"] != spec
        or claim["task"] != spec
        or seal["lock_id"] != lock["artifact_id"]
        or claim["lock_id"] != lock["artifact_id"]
    ):
        raise ValueError("registered native task identity changed")
    data = load_native_bundle(directory, seal["bundle"])
    _, reference, rewards = references[spec["source_world_index"]]
    qa = native_checks(
        data,
        seal["native"],
        np.asarray(old["controls"]),
        reference,
        rewards,
        spec["world"],
    )
    return seal, data, qa


def gate(output, lock, old, references, index):
    seal, _, qa = load_task(output, lock, old, references, index)
    return {
        "lock_id": lock["artifact_id"],
        "index": index,
        "seal_id": seal["artifact_id"],
        "qa": qa,
    }


def repeat_gate(output, lock, old, references):
    rows = [load_task(output, lock, old, references, i) for i in range(3)]
    return {
        "lock_id": lock["artifact_id"],
        "seal_ids": [row[0]["artifact_id"] for row in rows],
        "numerics": repeat_checks(
            [row[1] for row in rows],
            np.asarray([row[0]["native"]["native_cumulative_reward"] for row in rows]),
        ),
    }


def require_previous(output, lock, old, references, index):
    for earlier in range(index):
        computed = gate(output, lock, old, references, earlier)
        stored = read_record(output / f"admission-{earlier:02d}.json")
        if not computed["qa"]["passed"] or any(
            stored[k] != v for k, v in computed.items()
        ):
            raise ValueError("earlier admission must rederive")
    if index >= 3:
        computed = repeat_gate(output, lock, old, references)
        stored = read_record(output / "repeat-gate.json")
        if not computed["numerics"]["passed"] or any(
            stored[k] != v for k, v in computed.items()
        ):
            raise ValueError("new-context numerical qualification must rederive")


def worker(output, index):
    lock, old, references = validate(output)
    spec = task(index)
    require_previous(output, lock, old, references, index)
    directory = output / spec["name"]
    directory.mkdir()
    claim = write_record(
        directory / "claim.json", {"lock_id": lock["artifact_id"], "task": spec}
    )
    data, native = run_registered_worlds(
        Path(old["assets_root"]) / "upstream",
        directory,
        controls(np.asarray(old["controls"])),
        [spec["world"]] * 8,
        prefix_only=False,
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
            f"native batch {index} exited {completed.returncode}; no retry"
        )
    print(f"completed late-branch batch {index + 1}/11", flush=True)


def run(output):
    if output.resolve() != OUTPUT:
        raise ValueError("only registered write-once root permitted")
    revision = clean_revision(ROOT)
    old, references = source()
    paths = sorted(
        set(old["source_sha256"]) | set(parent.ADDITIONAL_SOURCES) | set(SOURCES)
    )
    output.mkdir()
    lock = write_record(
        output / "lock.json",
        {
            "schema": "dlolab-slingshot-late-lock-v1",
            "revision": revision,
            "source_sha256": {p: file_digest(ROOT / p) for p in paths},
            "protocol": protocol(),
            "output_root": str(OUTPUT),
            "parent_lock_id": parent.PARENT_ID,
            "parent_bank_id": parent.BANK_ID,
            "reference_ids": [r[0] for r in references],
            "numerical_audit_id": NUMERICAL_ID,
            "controls_sha256": array_digest(controls(np.asarray(old["controls"]))),
        },
    )
    completed, admitted, attempted, stage = 0, 0, 0, "source-prefix-information"

    def terminal(stage, **extra):
        return write_record(
            output / "result.json",
            {
                "schema": "dlolab-slingshot-late-result-v1",
                "lock_id": lock["artifact_id"],
                "terminal_stage": stage,
                "planned_batches": 11,
                "completed_batches": completed,
                "admitted_batches": admitted,
                "unrun_batches": 11 - attempted,
                "source_gate_passed": False,
                "method_evaluation_authorized": False,
                "protected_data_read": False,
                **extra,
            },
        )

    try:
        prefix_gate = source_information(references)
        write_record(
            output / "prefix-information.json",
            {"lock_id": lock["artifact_id"], **prefix_gate},
        )
        if not prefix_gate["source_bank_authorized"]:
            terminal(stage)
            return
        for index in range(11):
            stage = "native-admission"
            attempted += 1
            launch(output, old, index)
            completed += 1
            admission = gate(output, lock, old, references, index)
            write_record(output / f"admission-{index:02d}.json", admission)
            if not admission["qa"]["passed"]:
                terminal(stage)
                return
            admitted += 1
            if index == 2:
                stage = "new-context-repeatability"
                repeated = repeat_gate(output, lock, old, references)
                write_record(output / "repeat-gate.json", repeated)
                if not repeated["numerics"]["passed"]:
                    terminal(stage)
                    return
        stage = "source-information-value"
        require_previous(output, lock, old, references, 11)
        rows = [
            load_task(output, lock, old, references, i)
            for i in range(11)
            if i not in (1, 2)
        ]
        rows.sort(key=lambda row: row[0]["task"]["source_world_index"])
        arrays = {
            "prefix": np.stack(
                [
                    observations(
                        {k: row[1][k][:500] for k in ("rod_pos_m", "sphere_pos_m")}
                    )[0]
                    for row in rows
                ]
            ),
            "reward": np.asarray(
                [row[0]["native"]["native_cumulative_reward"][:7] for row in rows]
            ),
            "original_reward": np.asarray([r[2][:7] for r in references]),
        }
        directory = output / "source-bank"
        directory.mkdir()
        seal = write_record(
            directory / "seal.json",
            {
                "lock_id": lock["artifact_id"],
                "source_seal_ids": [row[0]["artifact_id"] for row in rows],
                "bundle": write_native_bundle(directory, arrays),
            },
        )
        metrics = information_value(
            arrays["prefix"], arrays["reward"], arrays["original_reward"]
        )
        result = terminal(
            "complete",
            source_bank_id=seal["artifact_id"],
            metrics=metrics,
            source_gate_passed=metrics["source_information_value_passed"],
        )
        print(
            f"source gate={result['source_gate_passed']}; {result['artifact_id']}",
            flush=True,
        )
    except Exception as exc:
        write_record(
            output / "failure.json",
            {
                "lock_id": lock["artifact_id"],
                "stage": stage,
                "completed_batches": completed,
                "admitted_batches": admitted,
                "worker_invocations_attempted": attempted,
                "unrun_batches": 11 - attempted,
                "error": f"{type(exc).__name__}: {exc}",
                "retry_authorized": False,
                "protected_data_read": False,
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--worker", type=int, choices=range(11))
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        if args.worker is not None or args.output.resolve() != OUTPUT:
            parser.error("preflight cannot launch a worker or use another root")
        old, refs = source()
        print(
            {
                "preflight": True,
                "information": source_information(refs),
                "controls_sha256": array_digest(controls(np.asarray(old["controls"]))),
            },
            flush=True,
        )
    elif args.worker is None:
        run(args.output)
    else:
        worker(args.output, args.worker)
