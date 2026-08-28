#!/usr/bin/env python3
"""Frozen nine-world public simulator decision-value screen, CPU only."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from bayesian_phystwin_experiments.dlolab_benchmark import (
    source_identity,
    write_native_bundle,
)
from bayesian_phystwin_experiments.dlolab_native import file_digest
from bayesian_phystwin_experiments.dlolab_regret_artifacts import (
    clean_revision,
    read_record,
    write_record,
)
from bayesian_phystwin_experiments.dlolab_slingshot_cmaes import worker_environment
from bayesian_phystwin_experiments.dlolab_slingshot_process import (
    load_native_bundle,
    runtime,
)
from bayesian_phystwin_experiments.dlolab_slingshot_value import (
    action_bank,
    decision_value,
    protocol,
    run_world,
    verify_source,
    world_metrics,
    worlds,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    "src/bayesian_phystwin_experiments/dlolab_slingshot_value.py",
    "scripts/remote/run_dlolab_slingshot_value.py",
    "tests/test_dlolab_slingshot_value.py",
    "docs/dlolab_slingshot_decision_value_source_v1.md",
)


def worker(output, index):
    if type(index) is not int or index not in range(9):
        raise ValueError("unregistered world")
    lock = read_record(output / "lock.json")
    if (
        lock["source_revision"] != clean_revision(ROOT)
        or lock["source_sha256"] != {name: file_digest(ROOT / name) for name in SOURCES}
        or lock["protocol"] != protocol()
        or lock["output_root"] != str(output.resolve())
    ):
        raise ValueError("world source/lock changed")
    verified, reference = verify_source(
        Path(lock["source"]["controller"]["path"]),
        Path(lock["source"]["mechanism"]["path"]),
        ROOT,
    )
    if verified != lock["source"] or runtime() != verified["controller"]["runtime"]:
        raise ValueError("world source/runtime changed")
    assets = Path(lock["assets_root"])
    if (
        source_identity(
            assets / "upstream", assets / "mushroom-rl", assets / "dlo-lab.zip"
        )
        != verified["controller"]["native_source"]
    ):
        raise ValueError("native source changed")
    directory = output / f"world-{index:02d}"
    directory.mkdir(exist_ok=False)
    claim = write_record(
        directory / "claim.json",
        {
            "schema": "dlolab-slingshot-value-claim-v1",
            "lock_id": lock["artifact_id"],
            "index": index,
            "world": worlds()[index],
            "retry_authorized": False,
        },
    )
    try:
        arrays, native = run_world(
            assets / "upstream",
            directory,
            action_bank(reference["controls"]),
            worlds()[index],
        )
        bundle = write_native_bundle(directory, arrays)
        write_record(
            directory / "seal.json",
            {
                "schema": "dlolab-slingshot-value-seal-v1",
                "lock_id": lock["artifact_id"],
                "claim_id": claim["artifact_id"],
                "index": index,
                "world": worlds()[index],
                "bundle": bundle,
                "native": native,
            },
        )
    except Exception as error:
        write_record(
            directory / "failure.json",
            {
                "schema": "dlolab-slingshot-value-failure-v1",
                "claim_id": claim["artifact_id"],
                "error_type": type(error).__name__,
                "message": str(error),
                "retry_authorized": False,
            },
        )
        raise


def run(output, assets, controller, mechanism):
    revision = clean_revision(ROOT)
    verified, reference = verify_source(controller, mechanism, ROOT)
    if runtime() != verified["controller"]["runtime"]:
        raise ValueError("qualified runtime required")
    output.mkdir(parents=True, exist_ok=False)
    lock = write_record(
        output / "lock.json",
        {
            "schema": "dlolab-slingshot-value-lock-v1",
            "source_revision": revision,
            "source_sha256": {name: file_digest(ROOT / name) for name in SOURCES},
            "source": verified,
            "protocol": protocol(),
            "assets_root": str(assets.resolve()),
            "output_root": str(output.resolve()),
            "protected_data_read": False,
        },
    )
    seals = []
    try:
        for index in range(9):
            print(f"native decision-value world {index + 1}/9", flush=True)
            with (output / f"world-{index:02d}.log").open("x") as stream:
                subprocess.run(
                    [
                        sys.executable,
                        "-u",
                        str(Path(__file__).resolve()),
                        "--output",
                        str(output.resolve()),
                        "--worker-index",
                        str(index),
                    ],
                    cwd=ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                    env=worker_environment(verified["controller"]["runtime"]),
                )
            seals.append(
                read_record(output / f"world-{index:02d}/seal.json")["artifact_id"]
            )
        generation = write_record(
            output / "generation.json",
            {
                "schema": "dlolab-slingshot-value-generation-v1",
                "lock_id": lock["artifact_id"],
                "world_seals": seals,
                "ordinary_worlds": 9,
                "native_trajectories": 72,
                "technical_failures": 0,
            },
        )
        rows = []
        for index in range(9):
            directory = output / f"world-{index:02d}"
            seal = read_record(directory / "seal.json")
            if (
                seal["artifact_id"] != seals[index]
                or seal["lock_id"] != lock["artifact_id"]
                or seal["world"] != worlds()[index]
            ):
                raise ValueError("world seal binding changed")
            values = load_native_bundle(directory, seal["bundle"])
            rows.append(
                world_metrics(values, seal["native"], worlds()[index], reference)
            )
        result = write_record(
            output / "result.json",
            {
                "schema": "dlolab-slingshot-value-result-v1",
                "lock_id": lock["artifact_id"],
                "generation_id": generation["artifact_id"],
                "worlds": rows,
                **decision_value(rows),
            },
        )
        print(
            f"decision value={result['source_decision_value_passed']}; adjusted oracle gain={result['numeric_margin_adjusted_oracle_gain']}; id={result['artifact_id']}",
            flush=True,
        )
        return result
    except Exception as error:
        write_record(
            output / "failure.json",
            {
                "schema": "dlolab-slingshot-value-failure-v1",
                "lock_id": lock["artifact_id"],
                "completed_worlds": len(seals),
                "error_type": type(error).__name__,
                "message": str(error),
                "retry_authorized": False,
                "protected_data_read": False,
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--controller", type=Path)
    parser.add_argument("--mechanism", type=Path)
    parser.add_argument("--worker-index", type=int)
    args = parser.parse_args()
    if args.worker_index is not None:
        worker(args.output, args.worker_index)
    elif args.assets is None or args.controller is None or args.mechanism is None:
        parser.error("assets and both frozen source results are required")
    else:
        run(args.output, args.assets, args.controller, args.mechanism)
