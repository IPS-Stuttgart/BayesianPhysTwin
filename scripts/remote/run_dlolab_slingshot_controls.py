#!/usr/bin/env python3
"""Execute the fixed source-only native action bank, without a target study."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

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
from bayesian_phystwin_experiments.dlolab_slingshot_controls import (
    action_bank,
    candidate_metrics,
    protocol,
    summarize,
    verify_qualification,
)
from bayesian_phystwin_experiments.dlolab_slingshot_process import (
    load_native_bundle,
    run_native,
    runtime,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    "src/bayesian_phystwin_experiments/dlolab_slingshot_controls.py",
    "src/bayesian_phystwin_experiments/dlolab_slingshot_process.py",
    "src/bayesian_phystwin_experiments/dlolab_benchmark.py",
    "src/bayesian_phystwin_experiments/dlolab_native.py",
    "src/bayesian_phystwin_experiments/dlolab_regret_artifacts.py",
    "src/bayesian_phystwin_experiments/deform_state_restart.py",
    "scripts/remote/run_dlolab_slingshot_controls.py",
    "tests/test_dlolab_slingshot_controls.py",
    "docs/dlolab_slingshot_task_competence_v1.md",
)


def validate_lock(output: Path) -> dict[str, Any]:
    lock = read_record(output / "lock.json")
    if (
        lock["output_root"] != str(output.resolve())
        or lock["source_revision"] != clean_revision(ROOT)
        or lock["protocol"] != protocol()
    ):
        raise ValueError("registered action-bank lock changed")
    if (
        lock["source_sha256"] != {name: file_digest(ROOT / name) for name in SOURCES}
        or lock["qualification"]["runtime"] != runtime()
    ):
        raise ValueError("registered implementation/runtime changed")
    assets = Path(lock["assets_root"])
    if (
        source_identity(
            assets / "upstream", assets / "mushroom-rl", assets / "dlo-lab.zip"
        )
        != lock["qualification"]["native_source"]
    ):
        raise ValueError("native source changed")
    return lock


def worker(output: Path, index: int) -> None:
    actions, _ = action_bank()
    if not 0 <= index < len(actions):
        raise ValueError("unregistered action index")
    lock = validate_lock(output)
    directory = output / f"candidate-{index:02d}"
    directory.mkdir(exist_ok=False)
    claim = write_record(
        directory / "claim.json",
        {
            "schema": "dlolab-slingshot-control-claim-v1",
            "lock_id": lock["artifact_id"],
            "index": index,
            "retry_authorized": False,
        },
    )
    try:
        arrays, summary = run_native(
            Path(lock["assets_root"]) / "upstream", directory, actions[index][None]
        )
        bundle = write_native_bundle(directory, arrays)
        write_record(
            directory / "seal.json",
            {
                "schema": "dlolab-slingshot-control-seal-v1",
                "lock_id": lock["artifact_id"],
                "claim_id": claim["artifact_id"],
                "index": index,
                "bundle": bundle,
                "summary": summary,
                "protected_data_read": False,
            },
        )
    except Exception as error:
        write_record(
            directory / "failure.json",
            {
                "schema": "dlolab-slingshot-control-failure-v1",
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


def run(output: Path, assets: Path, qualification: Path) -> dict[str, Any]:
    revision = clean_revision(ROOT)
    verified = verify_qualification(qualification, ROOT)
    if verified["runtime"] != runtime() or verified["native_source"] != source_identity(
        assets / "upstream", assets / "mushroom-rl", assets / "dlo-lab.zip"
    ):
        raise ValueError("qualified runtime/native source changed")
    output.mkdir(parents=True, exist_ok=False)
    lock = write_record(
        output / "lock.json",
        {
            "schema": "dlolab-slingshot-control-lock-v1",
            "source_revision": revision,
            "source_sha256": {name: file_digest(ROOT / name) for name in SOURCES},
            "protocol": protocol(),
            "qualification": verified,
            "output_root": str(output.resolve()),
            "assets_root": str(assets.resolve()),
            "protected_data_read": False,
            "method_evaluation_authorized": False,
        },
    )
    dispositions = []
    for index in range(protocol()["candidate_count"]):
        print(
            f"native source candidate {index + 1}/{protocol()['candidate_count']}",
            flush=True,
        )
        with (output / f"candidate-{index:02d}.log").open("x") as log:
            process = subprocess.run(
                [
                    sys.executable,
                    "-u",
                    str(Path(__file__).resolve()),
                    "--output",
                    str(output.resolve()),
                    "--worker-index",
                    str(index),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=ROOT,
                check=False,
            )
        dispositions.append({"index": index, "returncode": process.returncode})
    generation = write_record(
        output / "generation.json",
        {
            "schema": "dlolab-slingshot-control-generation-v1",
            "lock_id": lock["artifact_id"],
            "dispositions": dispositions,
            "protected_data_read": False,
        },
    )
    rows, failures = [], []
    for item in dispositions:
        index = item["index"]
        directory = output / f"candidate-{index:02d}"
        if item["returncode"] != 0:
            failures.append(index)
            continue
        seal = read_record(directory / "seal.json")
        if seal["lock_id"] != lock["artifact_id"] or seal["index"] != index:
            raise ValueError("candidate seal binding changed")
        arrays = load_native_bundle(directory, seal["bundle"])
        rows.append(
            candidate_metrics(
                arrays, index, seal["summary"]["native_cumulative_reward"][0]
            )
        )
    result = write_record(
        output / "result.json",
        {
            "schema": "dlolab-slingshot-task-competence-result-v1",
            "lock_id": lock["artifact_id"],
            "generation_id": generation["artifact_id"],
            **summarize(rows, failures),
        },
    )
    print(
        f"task competence={result['task_competence_passed']}; capable={result['capable_candidate_count']}; id={result['artifact_id']}",
        flush=True,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--worker-index", type=int)
    args = parser.parse_args()
    if args.worker_index is not None:
        worker(args.output, args.worker_index)
    elif args.assets is None or args.qualification is None:
        parser.error("--assets and --qualification are required")
    else:
        run(args.output, args.assets, args.qualification)
