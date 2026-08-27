#!/usr/bin/env python3
"""One source-only native batch qualification, with all failures retained."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

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
from bayesian_phystwin_experiments.dlolab_slingshot_batch import (
    BATCH_INDICES,
    compare,
    protocol,
    run_batch,
    split_batch,
)
from bayesian_phystwin_experiments.dlolab_slingshot_controls import verify_qualification
from bayesian_phystwin_experiments.dlolab_slingshot_process import (
    load_native_bundle,
    runtime,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    "src/bayesian_phystwin_experiments/dlolab_slingshot_batch.py",
    "src/bayesian_phystwin_experiments/dlolab_slingshot_controls.py",
    "src/bayesian_phystwin_experiments/dlolab_slingshot_process.py",
    "src/bayesian_phystwin_experiments/dlolab_benchmark.py",
    "src/bayesian_phystwin_experiments/dlolab_native.py",
    "src/bayesian_phystwin_experiments/dlolab_regret_artifacts.py",
    "scripts/remote/qualify_dlolab_slingshot_batch.py",
    "tests/test_dlolab_slingshot_batch.py",
    "docs/dlolab_slingshot_batch_qualification_v1.md",
)


def run(output: Path, assets: Path, qualification: Path) -> dict[str, Any]:
    revision = clean_revision(ROOT)
    verified = verify_qualification(qualification, ROOT)
    if verified["runtime"] != runtime() or verified["native_source"] != source_identity(
        assets / "upstream", assets / "mushroom-rl", assets / "dlo-lab.zip"
    ):
        raise ValueError("qualified native runtime/source changed")
    references = []
    for index in range(2):
        directory = qualification.parent / f"run-{index}"
        references.append(
            load_native_bundle(
                directory, read_record(directory / "seal.json")["bundle"]
            )
        )
    output.mkdir(parents=True, exist_ok=False)
    lock = write_record(
        output / "lock.json",
        {
            "schema": "dlolab-slingshot-batch-lock-v1",
            "source_revision": revision,
            "source_sha256": {name: file_digest(ROOT / name) for name in SOURCES},
            "protocol": protocol(),
            "qualification": verified,
            "output_root": str(output.resolve()),
            "assets_root": str(assets.resolve()),
            "retry_authorized": False,
            "protected_data_read": False,
            "method_evaluation_authorized": False,
        },
    )
    stage = "native-batch"
    try:
        controls = np.stack([slingshot_actions()[index] for index in BATCH_INDICES])
        arrays, native = run_batch(assets / "upstream", output, controls)
        stage = "seal-generation"
        bundle = write_native_bundle(output, arrays)
        generation = write_record(
            output / "generation.json",
            {
                "schema": "dlolab-slingshot-batch-generation-v1",
                "lock_id": lock["artifact_id"],
                "bundle": bundle,
                "native": native,
                "protected_data_read": False,
            },
        )
        stage = "compare-isolated-reference"
        result = write_record(
            output / "result.json",
            {
                "schema": "dlolab-slingshot-batch-result-v1",
                "lock_id": lock["artifact_id"],
                "generation_id": generation["artifact_id"],
                "native": native,
                **compare(
                    split_batch(arrays, 8),
                    references,
                    native["native_cumulative_reward"],
                ),
            },
        )
        print(
            f"batch qualification={result['batch_qualification_passed']}; id={result['artifact_id']}",
            flush=True,
        )
        return result
    except Exception as error:
        write_record(
            output / "failure.json",
            {
                "schema": "dlolab-slingshot-batch-failure-v1",
                "lock_id": lock["artifact_id"],
                "terminal_stage": stage,
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
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.assets, args.qualification)
