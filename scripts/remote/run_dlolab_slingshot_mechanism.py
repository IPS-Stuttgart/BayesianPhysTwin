#!/usr/bin/env python3
"""Three source-only fresh-process mechanism runs; no tuning or retries."""

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
from bayesian_phystwin_experiments.dlolab_slingshot_mechanism import (
    ARMS,
    assess,
    protocol,
    run_arm,
    verify_controller,
)
from bayesian_phystwin_experiments.dlolab_slingshot_process import (
    load_native_bundle,
    runtime,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    "src/bayesian_phystwin_experiments/dlolab_slingshot_mechanism.py",
    "scripts/remote/run_dlolab_slingshot_mechanism.py",
    "tests/test_dlolab_slingshot_mechanism.py",
    "docs/dlolab_slingshot_mechanism_source_v1.md",
)


def worker(output, index):
    if type(index) is not int or index not in range(3):
        raise ValueError("unregistered mechanism index")
    lock = read_record(output / "lock.json")
    if (
        lock["source_revision"] != clean_revision(ROOT)
        or lock["protocol"] != protocol()
        or lock["output_root"] != str(output.resolve())
    ):
        raise ValueError("mechanism lock changed")
    if lock["source_sha256"] != {name: file_digest(ROOT / name) for name in SOURCES}:
        raise ValueError("mechanism implementation changed")
    verified, reference = verify_controller(Path(lock["controller"]["path"]), ROOT)
    if verified != lock["controller"] or runtime() != verified["runtime"]:
        raise ValueError("mechanism controller/runtime changed")
    assets = Path(lock["assets_root"])
    if (
        source_identity(
            assets / "upstream", assets / "mushroom-rl", assets / "dlo-lab.zip"
        )
        != verified["native_source"]
    ):
        raise ValueError("mechanism native source changed")
    target = output / ARMS[index]
    target.mkdir(exist_ok=False)
    claim = write_record(
        target / "claim.json",
        {
            "schema": "dlolab-slingshot-mechanism-claim-v1",
            "lock_id": lock["artifact_id"],
            "index": index,
            "arm": ARMS[index],
            "retry_authorized": False,
        },
    )
    try:
        values, native = run_arm(
            assets / "upstream", target, reference["controls"], ARMS[index]
        )
        bundle = write_native_bundle(target, values)
        write_record(
            target / "seal.json",
            {
                "schema": "dlolab-slingshot-mechanism-seal-v1",
                "lock_id": lock["artifact_id"],
                "claim_id": claim["artifact_id"],
                "index": index,
                "arm": ARMS[index],
                "bundle": bundle,
                "native": native,
            },
        )
    except Exception as error:
        write_record(
            target / "failure.json",
            {
                "schema": "dlolab-slingshot-mechanism-failure-v1",
                "claim_id": claim["artifact_id"],
                "error_type": type(error).__name__,
                "message": str(error),
                "retry_authorized": False,
            },
        )
        raise


def run(output, assets, controller):
    revision = clean_revision(ROOT)
    verified, reference = verify_controller(controller, ROOT)
    if runtime() != verified["runtime"]:
        raise ValueError("qualified runtime required")
    output.mkdir(parents=True, exist_ok=False)
    lock = write_record(
        output / "lock.json",
        {
            "schema": "dlolab-slingshot-mechanism-lock-v1",
            "source_revision": revision,
            "source_sha256": {name: file_digest(ROOT / name) for name in SOURCES},
            "protocol": protocol(),
            "controller": verified,
            "assets_root": str(assets.resolve()),
            "output_root": str(output.resolve()),
            "protected_data_read": False,
        },
    )
    rows, seals = [], []
    try:
        for index, arm in enumerate(ARMS):
            print(f"native contact audit {index + 1}/3: {arm}", flush=True)
            with (output / f"{arm}.log").open("x") as stream:
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
                    env=worker_environment(verified["runtime"]),
                )
            seal = read_record(output / arm / "seal.json")
            if seal["lock_id"] != lock["artifact_id"] or seal["index"] != index:
                raise ValueError("mechanism seal binding changed")
            rows.append(load_native_bundle(output / arm, seal["bundle"]))
            seals.append(seal["artifact_id"])
        result = write_record(
            output / "result.json",
            {
                "schema": "dlolab-slingshot-mechanism-result-v1",
                "lock_id": lock["artifact_id"],
                "seals": seals,
                **assess(rows, reference),
            },
        )
        print(
            f"mechanism audit={result['mechanism_audit_passed']}; id={result['artifact_id']}",
            flush=True,
        )
        return result
    except Exception as error:
        write_record(
            output / "failure.json",
            {
                "schema": "dlolab-slingshot-mechanism-failure-v1",
                "lock_id": lock["artifact_id"],
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
    parser.add_argument("--worker-index", type=int)
    args = parser.parse_args()
    if args.worker_index is not None:
        worker(args.output, args.worker_index)
    elif args.assets is None or args.controller is None:
        parser.error("assets and controller are required")
    else:
        run(args.output, args.assets, args.controller)
