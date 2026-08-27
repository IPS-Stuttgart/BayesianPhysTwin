#!/usr/bin/env python3
"""Qualify fresh-process native Slingshot resets without changing physics."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

from bayesian_phystwin_experiments.dlolab_benchmark import (
    slingshot_actions,
    source_identity,
    write_native_bundle,
)
from bayesian_phystwin_experiments.dlolab_native import file_digest
from bayesian_phystwin_experiments.dlolab_regret_artifacts import (
    clean_revision,
    read_record,
    write_record,
)
from bayesian_phystwin_experiments.dlolab_slingshot_process import (
    load_native_bundle,
    protocol,
    qualify,
    run_native,
    runtime,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    "src/bayesian_phystwin_experiments/dlolab_slingshot_process.py",
    "src/bayesian_phystwin_experiments/dlolab_benchmark.py",
    "src/bayesian_phystwin_experiments/dlolab_native.py",
    "src/bayesian_phystwin_experiments/dlolab_regret_artifacts.py",
    "src/bayesian_phystwin_experiments/deform_state_restart.py",
    "src/bayesian_phystwin/_portable_contracts.py",
    "scripts/remote/qualify_dlolab_slingshot_process.py",
    "tests/test_dlolab_slingshot_process.py",
    "docs/dlolab_slingshot_fresh_process_v2.md",
)


def validate_lock(output: Path) -> dict[str, Any]:
    lock = read_record(output / "lock.json")
    if (
        lock["protocol"] != protocol()
        or lock["output_root"] != str(output.resolve())
        or lock["source_revision"] != clean_revision(ROOT)
    ):
        raise ValueError("fresh-process qualification lock changed")
    if (
        lock["source_sha256"] != {name: file_digest(ROOT / name) for name in SOURCES}
        or lock["runtime"] != runtime()
    ):
        raise ValueError("qualified source/runtime changed")
    assets = Path(lock["assets_root"])
    if lock["native_source"] != source_identity(
        assets / "upstream", assets / "mushroom-rl", assets / "dlo-lab.zip"
    ):
        raise ValueError("native source or assets changed")
    return lock


def worker(output: Path, index: int) -> None:
    if index not in (0, 1, 2):
        raise ValueError("invalid registered rollout")
    lock = validate_lock(output)
    directory = output / f"run-{index}"
    directory.mkdir(exist_ok=False)
    claim = write_record(
        directory / "claim.json",
        {
            "schema": "dlolab-slingshot-fresh-process-claim-v2",
            "lock_id": lock["artifact_id"],
            "index": index,
            "action_index": [0, 1, 1][index],
            "retry_authorized": False,
        },
    )
    try:
        arrays, summary = run_native(
            Path(lock["assets_root"]) / "upstream",
            directory,
            slingshot_actions()[[0, 1, 1][index]][None],
        )
        bundle = write_native_bundle(directory, arrays)
        write_record(
            directory / "seal.json",
            {
                "schema": "dlolab-slingshot-fresh-process-seal-v2",
                "lock_id": lock["artifact_id"],
                "claim_id": claim["artifact_id"],
                "index": index,
                "summary": summary,
                "bundle": bundle,
                "protected_data_read": False,
                "method_evaluation_authorized": False,
            },
        )
    except Exception as error:
        write_record(
            directory / "failure.json",
            {
                "schema": "dlolab-slingshot-fresh-process-failure-v2",
                "lock_id": lock["artifact_id"],
                "claim_id": claim["artifact_id"],
                "index": index,
                "error_type": type(error).__name__,
                "message": str(error),
                "retry_authorized": False,
                "protected_data_read": False,
            },
        )
        raise


def run(output: Path, assets: Path, parent_result: Path) -> dict[str, Any]:
    revision = clean_revision(ROOT)
    if (
        file_digest(parent_result)
        != "33b293c808812f97fcd6833f0cf8daff3607d08a4bd1ed78767ed361e4efea65"
    ):
        raise ValueError("retained reused-reset result changed")
    parent = read_record(parent_result)
    parent_attempt = read_record(parent_result.parent / "attempt.json")
    native = source_identity(
        assets / "upstream", assets / "mushroom-rl", assets / "dlo-lab.zip"
    )
    current_runtime = runtime()
    if (
        parent["artifact_id"] != protocol()["retained_parent_result_id"]
        or parent["attempt_id"] != parent_attempt["artifact_id"]
        or parent["native_qualification_passed"] is not False
    ):
        raise ValueError("retained qualification failure required")
    if (
        parent_attempt["native_source"] != native
        or parent_attempt["runtime"] != current_runtime
    ):
        raise ValueError("fresh processes must use the same native source/runtime")
    output.mkdir(parents=True, exist_ok=False)
    lock = write_record(
        output / "lock.json",
        {
            "schema": "dlolab-slingshot-fresh-process-lock-v2",
            "protocol": protocol(),
            "source_revision": revision,
            "source_sha256": {name: file_digest(ROOT / name) for name in SOURCES},
            "output_root": str(output.resolve()),
            "assets_root": str(assets.resolve()),
            "native_source": native,
            "runtime": current_runtime,
            "parent_result_file_sha256": file_digest(parent_result),
            "protected_data_read": False,
            "method_evaluation_authorized": False,
        },
    )
    completed: list[str] = []
    try:
        for index in range(3):
            print(f"fresh-process native rollout {index + 1}/3", flush=True)
            with (output / f"run-{index}.log").open("x") as log:
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
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=ROOT,
                )
            completed.append(
                read_record(output / f"run-{index}/seal.json")["artifact_id"]
            )
        arrays = []
        summaries = []
        for index in range(3):
            seal = read_record(output / f"run-{index}/seal.json")
            if (
                seal["lock_id"] != lock["artifact_id"]
                or seal["index"] != index
                or seal["summary"]["native_steps"] != 900
            ):
                raise ValueError("native seal binding changed")
            arrays.append(load_native_bundle(output / f"run-{index}", seal["bundle"]))
            summaries.append(seal["summary"])
        result = write_record(
            output / "result.json",
            {
                "schema": "dlolab-slingshot-fresh-process-result-v2",
                "lock_id": lock["artifact_id"],
                "rollout_seals": completed,
                "rollouts": summaries,
                **qualify(arrays),
            },
        )
        print(
            f"native qualification={result['native_qualification_passed']}; id={result['artifact_id']}",
            flush=True,
        )
        return result
    except Exception as error:
        write_record(
            output / "failure.json",
            {
                "schema": "dlolab-slingshot-fresh-process-failure-v2",
                "lock_id": lock["artifact_id"],
                "completed_rollout_seals": completed,
                "error_type": type(error).__name__,
                "message": str(error),
                "retry_authorized": False,
                "protected_data_read": False,
                "method_evaluation_authorized": False,
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--parent-result", type=Path)
    parser.add_argument("--worker-index", type=int)
    args = parser.parse_args()
    if args.worker_index is not None:
        worker(args.output, args.worker_index)
    elif args.assets is None or args.parent_result is None:
        parser.error("--assets and --parent-result are required for qualification")
    else:
        run(args.output, args.assets, args.parent_result)
