#!/usr/bin/env python3
"""Run one frozen three-world native contact-realization source screen."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

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
    native_qa,
    prefix_observations,
)
from bayesian_phystwin_experiments.dlolab_slingshot_cmaes import worker_environment
from bayesian_phystwin_experiments.dlolab_slingshot_contact import (
    COUPLINGS,
    RUN_ORDER,
    information_value,
    nominal_replay,
    protocol,
    run_contact_world,
    task,
)
from bayesian_phystwin_experiments.dlolab_slingshot_process import (
    load_native_bundle,
    runtime,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(
    "/home/fpfaff/source-only/dlolab-benchmark-source-v1/contact-realization-source-v1"
)
PARENT = OUTPUT.parent / "belief-control-source-v1"
REFERENCE = OUTPUT.parent / "decision-value-source-v1/world-00"
PARENT_ID = "015e6d84aa68a2a4310552ef4880752b972890f02d3e09e333ff575c92b8df25"
BANK_ID = "8ebf9c91322faf0658c84a2dcaa6895a98b1ff857e49e6714a2a2dad0c88d882"
REFERENCE_ID = "cd2f32fa143b10349f5e895680b011d953ff53099587b685539c3159e853fc67"
REFERENCE_ACTIONS = (0, 1, 2, 3, 4, 5, 6, 5)
ADDITIONAL_SOURCES = (
    "src/bayesian_phystwin_experiments/dlolab_slingshot_contact.py",
    "scripts/remote/run_dlolab_slingshot_contact.py",
    "tests/test_dlolab_slingshot_contact.py",
    "docs/dlolab_slingshot_contact_realization_source_v1.md",
)


def source():
    old = read_record(PARENT / "lock.json")
    bank = read_record(PARENT / "model-bank/seal.json")
    reference_seal = read_record(REFERENCE / "seal.json")
    if (
        old["artifact_id"] != PARENT_ID
        or bank["artifact_id"] != BANK_ID
        or bank["lock_id"] != PARENT_ID
        or reference_seal["artifact_id"] != REFERENCE_ID
        or bank["parents"][13]["kind"] != "reused_open_source"
        or bank["parents"][13]["world_index"] != 0
    ):
        raise ValueError("exact opened source lineage required")
    if any(
        file_digest(ROOT / name) != digest
        for name, digest in old["source_sha256"].items()
    ):
        raise ValueError("existing frozen source implementation changed")
    native = old["screen"]["source"]["controller"]
    assets = Path(old["assets_root"])
    if (
        runtime() != native["runtime"]
        or source_identity(
            assets / "upstream", assets / "mushroom-rl", assets / "dlo-lab.zip"
        )
        != native["native_source"]
    ):
        raise ValueError("qualified public assets or CPU runtime changed")
    reference_raw = load_native_bundle(REFERENCE, reference_seal["bundle"])
    reference = {
        name: np.take(value, REFERENCE_ACTIONS, axis=1 if name in TRACE_NAMES else 0)
        for name, value in reference_raw.items()
    }
    if array_digest(reference["controls"]) != array_digest(
        np.asarray(old["controls"], dtype=np.float64)
    ):
        raise ValueError("nominal source action identity changed")
    return (
        old,
        reference,
        [
            reference_seal["native"]["native_cumulative_reward"][i]
            for i in REFERENCE_ACTIONS
        ],
    )


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
        or lock["parent_lock_id"] != PARENT_ID
        or lock["bank_id"] != BANK_ID
        or lock["reference_id"] != REFERENCE_ID
        or lock["reference_action_indices"] != list(REFERENCE_ACTIONS)
        or any(
            file_digest(ROOT / p) != digest
            for p, digest in lock["source_sha256"].items()
        )
    ):
        raise ValueError("clean frozen source/protocol required")
    old, reference, expected_reward = source()
    if lock["controls_sha256"] != array_digest(
        np.asarray(old["controls"], dtype=np.float64)
    ):
        raise ValueError("native command bytes changed")
    return lock, old, reference, expected_reward


def load_task(output, lock, old, index):
    spec = task(index)
    directory = output / spec["name"]
    claim = read_record(directory / "claim.json")
    seal = read_record(directory / "seal.json")
    if (
        claim["lock_id"] != lock["artifact_id"]
        or claim["task"] != spec
        or seal["claim_id"] != claim["artifact_id"]
        or seal["lock_id"] != lock["artifact_id"]
        or seal["task"] != spec
    ):
        raise ValueError("registered contact task identity changed")
    data = load_native_bundle(directory, seal["bundle"])
    binding = seal["native"]["contact_realization"]
    geometry = binding["geometry"]
    robot_indices = geometry["robot_geometry_indices"]
    if (
        binding["modified_material_count"] != 1
        or not robot_indices
        or geometry["verified_before_native_action"] is not True
        or geometry["robot_coupling_values"] != [COUPLINGS[index]] * len(robot_indices)
    ):
        raise ValueError("native gripper coupling realization changed")
    realization = seal["native"]["world_realization"]
    if realization["bending"] != [[100000.0] * 8] or realization["stretching"] != [
        [800000.0] * 8
    ]:
        raise ValueError("nominal material realization changed")
    for name, position in (("sphere", [0.12, 0.06, 0.2]), ("cube", [0.12, 0.23, 0.22])):
        if not np.allclose(
            realization[f"{name}_initial_position_m"],
            [position] * 8,
            rtol=0,
            atol=1e-15,
        ):
            raise ValueError("nominal placement changed")
    checks = native_qa(
        data, seal["native"], np.asarray(old["controls"], dtype=np.float64)
    )
    return seal, data, checks


def identity_result(output, lock, old, reference, expected_reward):
    seal, data, qa = load_task(output, lock, old, 2)
    replay = nominal_replay(
        data, reference, seal["native"]["native_cumulative_reward"], expected_reward
    )
    return {
        "schema": "dlolab-slingshot-contact-identity-v1",
        "lock_id": lock["artifact_id"],
        "nominal_seal_id": seal["artifact_id"],
        "native_qa": qa,
        "replay": replay,
        "contact_worlds_authorized": qa["qa_passed"] and replay["passed"],
    }


def require_identity(output, lock, old, reference, expected_reward):
    stored = read_record(output / "identity-result.json")
    computed = identity_result(output, lock, old, reference, expected_reward)
    if not computed["contact_worlds_authorized"] or any(
        stored[k] != value for k, value in computed.items()
    ):
        raise ValueError(
            "contact execution requires rederived passing nominal identity"
        )
    return computed


def worker(output, index):
    spec = task(index)
    lock, old, reference, expected_reward = validate(output)
    if index != 2:
        require_identity(output, lock, old, reference, expected_reward)
        for earlier in RUN_ORDER[1 : RUN_ORDER.index(index)]:
            _, _, qa = load_task(output, lock, old, earlier)
            if not qa["qa_passed"]:
                raise ValueError("previous contact world failed native QA")
    directory = output / spec["name"]
    directory.mkdir()
    claim = write_record(
        directory / "claim.json", {"lock_id": lock["artifact_id"], "task": spec}
    )
    values, native = run_contact_world(
        Path(old["assets_root"]) / "upstream",
        directory,
        np.asarray(old["controls"], dtype=np.float64),
        index,
    )
    write_record(
        directory / "seal.json",
        {
            "lock_id": lock["artifact_id"],
            "claim_id": claim["artifact_id"],
            "task": spec,
            "bundle": write_native_bundle(directory, values),
            "native": native,
        },
    )
    load_task(output, lock, old, index)


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
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=worker_environment(old["screen"]["source"]["controller"]["runtime"]),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(
            f"contact world {index} exited {result.returncode}; no retry"
        )
    print(f"completed contact world {index}", flush=True)


def run(output):
    if output.resolve() != OUTPUT:
        raise ValueError("only registered write-once root permitted")
    revision = clean_revision(ROOT)
    old, reference, expected_reward = source()
    output.mkdir()
    paths = sorted(set(old["source_sha256"]) | set(ADDITIONAL_SOURCES))
    lock = write_record(
        output / "lock.json",
        {
            "schema": "dlolab-slingshot-contact-lock-v1",
            "revision": revision,
            "source_sha256": {p: file_digest(ROOT / p) for p in paths},
            "protocol": protocol(),
            "output_root": str(OUTPUT),
            "parent_lock_id": PARENT_ID,
            "bank_id": BANK_ID,
            "reference_id": REFERENCE_ID,
            "reference_action_indices": list(REFERENCE_ACTIONS),
            "controls_sha256": array_digest(
                np.asarray(old["controls"], dtype=np.float64)
            ),
        },
    )
    stage = "nominal-identity"
    try:
        launch(output, old, 2)
        identity = identity_result(output, lock, old, reference, expected_reward)
        write_record(output / "identity-result.json", identity)
        if not identity["contact_worlds_authorized"]:
            write_record(
                output / "result.json",
                {
                    "lock_id": lock["artifact_id"],
                    "terminal_stage": stage,
                    "native_worlds_completed": 1,
                    "source_gate_passed": False,
                    "method_evaluation_authorized": False,
                    "protected_data_read": False,
                },
            )
            print("nominal identity failed; no contact alternatives run", flush=True)
            return
        stage = "contact-source-worlds"
        for index in (0, 1):
            launch(output, old, index)
            _, _, qa = load_task(output, lock, old, index)
            if not qa["qa_passed"]:
                raise ValueError(f"contact world {index} failed native QA")
        histories, rewards, ids, nonrobot, checks = [], [], [], [], []
        for index in range(3):
            seal, data, qa = load_task(output, lock, old, index)
            histories.append(
                prefix_observations(
                    {k: v[:300] for k, v in data.items() if k in TRACE_NAMES}
                )[0]
            )
            rewards.append([m["native_reward"] for m in qa["metrics"][:7]])
            ids.append(seal["artifact_id"])
            nonrobot.append(
                seal["native"]["contact_realization"]["geometry"]["nonrobot_geometry"]
            )
            checks.append(qa)
        if any(value != nonrobot[2] for value in nonrobot):
            raise ValueError("nonrobot material changed across contact worlds")
        directory = output / "source-bank"
        directory.mkdir()
        arrays = {"prefix": np.stack(histories), "reward": np.asarray(rewards)}
        seal = write_record(
            directory / "seal.json",
            {
                "lock_id": lock["artifact_id"],
                "source_seal_ids": ids,
                "bundle": write_native_bundle(directory, arrays),
                "native_qa": checks,
            },
        )
        metrics = information_value(arrays["prefix"], arrays["reward"])
        result = write_record(
            output / "result.json",
            {
                "schema": "dlolab-slingshot-contact-source-result-v1",
                "lock_id": lock["artifact_id"],
                "source_bank_id": seal["artifact_id"],
                "native_worlds_completed": 3,
                "metrics": metrics,
                "source_gate_passed": metrics["source_information_value_passed"],
                "method_evaluation_authorized": False,
                "protected_data_read": False,
                "claim_boundary": "Finite native-source information screen, not fresh control performance, calibrated safety, or published benchmark parity.",
            },
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
