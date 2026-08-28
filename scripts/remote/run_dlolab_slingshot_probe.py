#!/usr/bin/env python3
"""Run one frozen CPU-only early-pull source information screen."""

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
from bayesian_phystwin_experiments.dlolab_slingshot_belief_native import (
    run_registered_worlds,
)
from bayesian_phystwin_experiments.dlolab_slingshot_cmaes import worker_environment
from bayesian_phystwin_experiments.dlolab_slingshot_probe import (
    full_task,
    prefix_task,
    probe_controls,
    protocol,
    select_probe,
    source_information_value,
)
from bayesian_phystwin_experiments.dlolab_slingshot_process import (
    load_native_bundle,
    runtime,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(
    "/home/fpfaff/source-only/dlolab-benchmark-source-v1/informative-prefix-source-v1"
)
PARENT = Path(
    "/home/fpfaff/source-only/dlolab-benchmark-source-v1/belief-control-source-v1"
)
PARENT_LOCK_ID = "015e6d84aa68a2a4310552ef4880752b972890f02d3e09e333ff575c92b8df25"
BANK_ID = "8ebf9c91322faf0658c84a2dcaa6895a98b1ff857e49e6714a2a2dad0c88d882"
ADDITIONAL_SOURCES = (
    "src/bayesian_phystwin_experiments/dlolab_slingshot_probe.py",
    "scripts/remote/run_dlolab_slingshot_probe.py",
    "tests/test_dlolab_slingshot_probe.py",
    "docs/dlolab_slingshot_informative_prefix_source_v1.md",
)


def source():
    old = read_record(PARENT / "lock.json")
    bank_seal = read_record(PARENT / "model-bank/seal.json")
    if (
        old["artifact_id"] != PARENT_LOCK_ID
        or bank_seal["artifact_id"] != BANK_ID
        or bank_seal["lock_id"] != PARENT_LOCK_ID
    ):
        raise ValueError("exact opened source bank required")
    if any(
        file_digest(ROOT / name) != digest
        for name, digest in old["source_sha256"].items()
    ):
        raise ValueError("existing frozen source implementation changed")
    native_source = old["screen"]["source"]["controller"]
    assets = Path(old["assets_root"])
    if (
        runtime() != native_source["runtime"]
        or source_identity(
            assets / "upstream", assets / "mushroom-rl", assets / "dlo-lab.zip"
        )
        != native_source["native_source"]
    ):
        raise ValueError("qualified public assets or CPU runtime changed")
    bank = load_native_bundle(PARENT / "model-bank", bank_seal["bundle"])
    return old, bank


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
    ):
        raise ValueError("clean frozen implementation and protocol required")
    if any(
        file_digest(ROOT / name) != digest
        for name, digest in lock["source_sha256"].items()
    ):
        raise ValueError("source bytes changed")
    old, bank = source()
    if lock["parent_lock_id"] != old["artifact_id"] or lock["bank_id"] != BANK_ID:
        raise ValueError("source lineage changed")
    original = np.asarray(old["controls"], dtype=np.float64)
    for i in range(2):
        if (
            array_digest(probe_controls(original, i))
            != lock["probe_controls_sha256"][i]
        ):
            raise ValueError("front-loaded command changed")
    return lock, old, bank


def load_task(output, lock, old, spec):
    directory = output / spec["name"]
    claim, seal = (
        read_record(directory / "claim.json"),
        read_record(directory / "seal.json"),
    )
    if (
        claim["lock_id"] != lock["artifact_id"]
        or claim["task"] != spec
        or seal["claim_id"] != claim["artifact_id"]
        or seal["lock_id"] != lock["artifact_id"]
        or seal["task"] != spec
    ):
        raise ValueError("registered task/claim changed")
    data = load_native_bundle(directory, seal["bundle"])
    expected = probe_controls(
        np.asarray(old["controls"], dtype=np.float64), spec["probe"]
    )
    if spec["prefix_only"]:
        expected = np.repeat(expected[5:6], 8, axis=0)
    if array_digest(data["controls"]) != array_digest(expected):
        raise ValueError("native commands changed")
    native = seal["native"]
    realization = native["world_realization"]
    for kind, key in (("bending", "bending_E"), ("stretching", "stretching_K")):
        if realization[kind] != [[w[key] for w in spec["worlds"]]]:
            raise ValueError("material binding changed")
    for name, y, z in (("sphere", 0.06, 0.2), ("cube", 0.23, 0.22)):
        initial = np.asarray([[0.12 + w["x_offset_m"], y, z] for w in spec["worlds"]])
        if not np.allclose(
            initial, realization[f"{name}_initial_position_m"], rtol=0, atol=1e-15
        ):
            raise ValueError("placement binding changed")
    if spec["prefix_only"]:
        if set(data) != set(TRACE_NAMES + ("controls",)) or any(
            data[name].shape[:2] != (300, 8) for name in TRACE_NAMES
        ):
            raise ValueError("unexpected prefix members or frame budget")
        if (
            native["native_steps"] != 300
            or native["future_simulated"] is not False
            or native["reward_scored"] is not False
        ):
            raise ValueError("future access during prefix screen")
        prefix_observations(data)
    elif native["native_steps"] != 900:
        raise ValueError("incomplete source future")
    return seal, data


def prefix_result(output, lock, old):
    histories, all_qa, ids = [], [], []
    for probe in range(2):
        rows: list[np.ndarray] = []
        qa: list[bool] = []
        for batch in range(2):
            spec = prefix_task(probe, batch)
            seal, data = load_task(output, lock, old, spec)
            ids.append(seal["artifact_id"])
            observed = prefix_observations(data)
            rows.extend(observed[: len(spec["world_indices"])])
            fixed = np.max(
                np.abs(
                    data["rod_pos_m"][:, :, [0, 1, 10, 11]]
                    - data["rod_pos_m"][:1, :, [0, 1, 10, 11]]
                )
            )
            duplicate = 0.0
            if batch == 1:
                duplicate = max(
                    float(np.max(np.abs(data[name] - data[name][:, :1])))
                    for name in (
                        "rod_pos_m",
                        "sphere_pos_m",
                        "cube_pos_m",
                        "gripper_pos_m",
                    )
                )
            qa.append(bool(fixed <= 1e-9 and duplicate <= 0.0005))
        histories.append(np.stack(rows))
        all_qa.append(all(qa))
    return {
        "schema": "dlolab-slingshot-probe-prefix-decision-v1",
        "lock_id": lock["artifact_id"],
        "prefix_seal_ids": ids,
        **select_probe(histories, all_qa),
    }


def require_prefix(output, lock, old):
    stored = read_record(output / "prefix-result.json")
    computed = prefix_result(output, lock, old)
    if not computed["source_bank_authorized"] or any(
        stored[k] != v for k, v in computed.items()
    ):
        raise ValueError("source bank requires rederived passing prefix evidence")
    return computed


def worker(output, kind, probe, index):
    lock, old, _ = validate(output)
    if kind == "prefix":
        spec = prefix_task(probe, index)
    elif kind == "source":
        decision = require_prefix(output, lock, old)
        if decision["selected_probe"] != probe:
            raise ValueError("unselected probe cannot run source futures")
        spec = full_task(probe, index)
    else:
        raise ValueError("unregistered execution kind")
    directory = output / spec["name"]
    directory.mkdir()
    claim = write_record(
        directory / "claim.json", {"lock_id": lock["artifact_id"], "task": spec}
    )
    commands = probe_controls(np.asarray(old["controls"], dtype=np.float64), probe)
    if spec["prefix_only"]:
        commands = np.repeat(commands[5:6], 8, axis=0)
    data, native = run_registered_worlds(
        Path(old["assets_root"]) / "upstream",
        directory,
        commands,
        spec["worlds"],
        prefix_only=spec["prefix_only"],
    )
    bundle = write_native_bundle(directory, data)
    write_record(
        directory / "seal.json",
        {
            "lock_id": lock["artifact_id"],
            "claim_id": claim["artifact_id"],
            "task": spec,
            "bundle": bundle,
            "native": native,
        },
    )
    load_task(output, lock, old, spec)


def launch(output, old, kind, probe, index):
    name = (
        prefix_task(probe, index)["name"]
        if kind == "prefix"
        else full_task(probe, index)["name"]
    )
    command = [
        sys.executable,
        str(Path(__file__)),
        "--output",
        str(output),
        "--worker",
        kind,
        "--probe",
        str(probe),
        "--index",
        str(index),
    ]
    with (output / f"{name}.log").open("x") as handle:
        run = subprocess.run(
            command,
            cwd=ROOT,
            env=worker_environment(old["screen"]["source"]["controller"]["runtime"]),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if run.returncode:
        raise RuntimeError(f"{name} exited {run.returncode}; retained, no retry")
    print(f"completed {name}", flush=True)


def run(output):
    if output.resolve() != OUTPUT:
        raise ValueError("only registered write-once root permitted")
    revision = clean_revision(ROOT)
    old, bank = source()
    original = np.asarray(old["controls"], dtype=np.float64)
    source_paths = sorted(set(old["source_sha256"]) | set(ADDITIONAL_SOURCES))
    command_hashes = [array_digest(probe_controls(original, i)) for i in range(2)]
    output.mkdir()
    lock = write_record(
        output / "lock.json",
        {
            "schema": "dlolab-slingshot-probe-lock-v1",
            "revision": revision,
            "source_sha256": {p: file_digest(ROOT / p) for p in source_paths},
            "protocol": protocol(),
            "output_root": str(OUTPUT),
            "parent_lock_id": PARENT_LOCK_ID,
            "bank_id": BANK_ID,
            "probe_controls_sha256": command_hashes,
        },
    )
    stage = "prefix"
    try:
        for probe in range(2):
            for batch in range(2):
                launch(output, old, "prefix", probe, batch)
        decision = prefix_result(output, lock, old)
        write_record(output / "prefix-result.json", decision)
        if not decision["source_bank_authorized"]:
            result = write_record(
                output / "result.json",
                {
                    "schema": "dlolab-slingshot-probe-result-v1",
                    "lock_id": lock["artifact_id"],
                    "terminal_stage": "prefix_information_gate",
                    "source_gate_passed": False,
                    "source_worlds_generated": 0,
                    "method_evaluation_authorized": False,
                    "protected_data_read": False,
                },
            )
            print(
                f"prefix information gate failed; {result['artifact_id']}", flush=True
            )
            return
        selected = decision["selected_probe"]
        stage = "source-bank"
        rewards, histories, qa, ids = [], [], [], []
        for i in range(27):
            launch(output, old, "source", selected, i)
            seal, data = load_task(output, lock, old, full_task(selected, i))
            prefix = None
            if 9 <= i <= 17:
                spec = prefix_task(selected, (i - 9) // 8)
                _, earlier = load_task(output, lock, old, spec)
                prefix = {
                    name: earlier[name][:, (i - 9) % 8]
                    for name in (
                        "rod_pos_m",
                        "sphere_pos_m",
                        "cube_pos_m",
                        "gripper_pos_m",
                    )
                }
            checks = native_qa(
                data, seal["native"], probe_controls(original, selected), prefix
            )
            if not checks["qa_passed"]:
                raise ValueError(f"native source QA failed for {i}")
            qa.append(checks)
            ids.append(seal["artifact_id"])
            histories.append(
                prefix_observations(
                    {
                        name: value[:300]
                        for name, value in data.items()
                        if name in TRACE_NAMES
                    }
                )[0]
            )
            rewards.append([v["native_reward"] for v in checks["metrics"][:7]])
        directory = output / "source-bank"
        directory.mkdir()
        values = {"prefix": np.stack(histories), "reward": np.asarray(rewards)}
        bundle = write_native_bundle(directory, values)
        seal = write_record(
            directory / "seal.json",
            {
                "lock_id": lock["artifact_id"],
                "selected_probe": selected,
                "source_seal_ids": ids,
                "bundle": bundle,
                "native_qa": qa,
            },
        )
        prior = np.asarray(old["protocol"]["prior_weights"])
        metrics = source_information_value(values["prefix"], values["reward"], prior)
        old_blind = float(np.max(prior @ bank["reward"]))
        gate = {
            "complete_native_source": len(qa) == 27 and all(q["qa_passed"] for q in qa),
            "information_gain_at_least_0_005": metrics["information_gain"] >= 0.005,
            "information_gain_at_least_10pct_excess": metrics["information_gain"]
            >= 0.1 * max(0.01, metrics["best_blind_reward"] - 6.900000095367432),
            "posterior_gain_over_map_at_least_0_002": metrics["posterior_gain_over_map"]
            >= 0.002,
            "not_weaker_than_original_best_blind": metrics["posterior_mean_reward"]
            >= old_blind,
        }
        result = write_record(
            output / "result.json",
            {
                "schema": "dlolab-slingshot-probe-result-v1",
                "lock_id": lock["artifact_id"],
                "bank_id": seal["artifact_id"],
                "selected_probe": selected,
                "source_worlds_generated": 27,
                "metrics": metrics,
                "original_best_blind_source_reward": old_blind,
                "checks": gate,
                "source_gate_passed": all(gate.values()),
                "method_evaluation_authorized": False,
                "protected_data_read": False,
                "claim_boundary": "Finite opened-source-model information screen, not fresh control performance or published benchmark parity.",
            },
        )
        print(
            f"source information/value gate={result['source_gate_passed']}; {result['artifact_id']}",
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
    parser.add_argument("--worker", choices=("prefix", "source"))
    parser.add_argument("--probe", type=int)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.worker is None:
        if args.probe is not None or args.index is not None:
            parser.error("probe/index require worker mode")
        run(args.output)
    else:
        if args.probe is None or args.index is None:
            parser.error("worker mode requires probe/index")
        worker(args.output, args.worker, args.probe, args.index)
