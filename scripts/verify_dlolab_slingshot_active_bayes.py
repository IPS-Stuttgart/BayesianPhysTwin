#!/usr/bin/env python3
"""Verify the terminal pre-science Slingshot active-Bayes v1 attempt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from bayesian_phystwin_experiments.dlolab_native import file_digest
from bayesian_phystwin_experiments.dlolab_regret_artifacts import read_record
from bayesian_phystwin_experiments.dlolab_slingshot_active_bayes import (
    prefix_task,
    protocol,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("/home/fpfaff/source-only/dlolab-slingshot-active-bayes-source-v1")
ATTEMPT = Path(
    "/home/fpfaff/source-only/dlolab-slingshot-active-bayes-source-v1.attempt.json"
)
SUMMARY = ROOT / "results/sota/dlolab_slingshot_active_bayes_source_v1/summary.json"
FROZEN_REVISION = "d918ae7888f45491ebd1fb284a6d1da6fb51d0e7"
FILE_SHA256 = {
    "attempt.json": "da7129b00c5f2699edb724f28cd1c2d712f805d87ff4476b8550b51a4a11e0c5",
    "failure.json": "0433eeb588401d37b9c6798f79eeaefc45e5ad9bf3cc2c43f94635c1aab599ae",
    "lock.json": "019d06450fb0694bbc1eb506c0726c25eaafbf2a8b88666316b2efd1701768a3",
    "prefix-passive-0.log": "fa78ddb9b428fab2660b9d7f935fff4514842d98adf7def4f85d20b49d38e225",
    "prefix-passive-0/claim.json": "5bd571f0f3173c7bdd0d1ba3816322695ea4f6d3b9967c411a526e7399a851de",
    "prefix-passive-0/failure.json": "14a91e2bc9e7e385b027061d063c4832f5ff71174ba4572b6144e055d98ad633",
}
OUTPUT_FILES = set(FILE_SHA256) - {"attempt.json"}


def _record_links(output: Path) -> tuple[dict[str, Any], ...]:
    attempt = read_record(ATTEMPT)
    lock = read_record(output / "lock.json")
    failure = read_record(output / "failure.json")
    claim = read_record(output / "prefix-passive-0" / "claim.json")
    child = read_record(output / "prefix-passive-0" / "failure.json")
    return attempt, lock, failure, claim, child


def verify(output: Path) -> dict[str, Any]:
    if output.resolve() != OUTPUT or output.is_symlink():
        raise ValueError("registered Slingshot active-Bayes v1 root required")
    files = {
        str(path.relative_to(output))
        for path in output.rglob("*")
        if path.is_file()
    }
    if files != OUTPUT_FILES or any(path.is_symlink() for path in output.rglob("*")):
        raise ValueError("terminal v1 file set changed")
    observed = {
        "attempt.json": file_digest(ATTEMPT),
        **{name: file_digest(output / name) for name in sorted(OUTPUT_FILES)},
    }
    if observed != FILE_SHA256:
        raise ValueError("terminal v1 byte identity changed")

    attempt, lock, failure, claim, child = _record_links(output)
    expected_task = prefix_task(0, 0)
    records = (attempt, lock, failure, claim, child)
    if (
        attempt.get("schema") != "dlolab-slingshot-active-bayes-attempt-v1"
        or attempt.get("revision") != FROZEN_REVISION
        or attempt.get("protocol") != protocol()
        or attempt.get("output_root") != str(OUTPUT)
        or lock.get("schema") != "dlolab-slingshot-active-bayes-lock-v1"
        or lock.get("revision") != FROZEN_REVISION
        or lock.get("attempt_id") != attempt["artifact_id"]
        or lock.get("source_sha256") != attempt.get("source_sha256")
        or lock.get("protocol") != attempt.get("protocol")
        or failure.get("schema") != "dlolab-slingshot-active-bayes-failure-v1"
        or failure.get("lock_id") != lock["artifact_id"]
        or failure.get("terminal_stage") != "prefixes"
        or failure.get("completed_prefix_batches") != 0
        or failure.get("completed_future_worlds") != 0
        or failure.get("error_type") != "RuntimeError"
        or failure.get("message") != "prefix-passive-0 exited 1; no retry"
        or claim.get("schema") != "dlolab-slingshot-active-bayes-claim-v1"
        or claim.get("lock_id") != lock["artifact_id"]
        or claim.get("task") != expected_task
        or claim.get("authorization") != {"gate": "prefix_only_before_outcomes"}
        or child.get("schema") != "dlolab-slingshot-active-bayes-failure-v1"
        or child.get("lock_id") != lock["artifact_id"]
        or child.get("claim_id") != claim["artifact_id"]
        or child.get("task") != expected_task
        or child.get("error_type") != "ModuleNotFoundError"
        or child.get("message") != "No module named 'mediapy'"
        or any(record.get("retry_authorized") is not False for record in records)
        or any(
            record.get("protected_data_read") is not False
            for record in (attempt, lock, failure, child)
        )
    ):
        raise ValueError("terminal v1 custody or diagnosis changed")
    if any(
        file_digest(ROOT / name) != digest
        for name, digest in lock["source_sha256"].items()
    ):
        raise ValueError("frozen active-Bayes source changed")
    log = (output / "prefix-passive-0.log").read_text(encoding="utf-8")
    if "ModuleNotFoundError: No module named 'mediapy'" not in log:
        raise ValueError("registered import failure absent from execution log")

    summary = read_record(SUMMARY)
    if (
        summary.get("schema") != "dlolab-slingshot-active-bayes-result-v1"
        or summary.get("status") != "terminal_pre_science_runtime_failure"
        or summary.get("frozen_revision") != FROZEN_REVISION
        or summary.get("attempt_id") != attempt["artifact_id"]
        or summary.get("lock_id") != lock["artifact_id"]
        or summary.get("failure_id") != failure["artifact_id"]
        or summary.get("child_claim_id") != claim["artifact_id"]
        or summary.get("child_failure_id") != child["artifact_id"]
        or summary.get("file_sha256") != FILE_SHA256
        or summary.get("scientific_hypothesis_resolved") is not False
        or summary.get("source_gate_evaluated") is not False
        or summary.get("retry_authorized") is not False
    ):
        raise ValueError("compact v1 result changed")
    return {
        "schema": "dlolab-slingshot-active-bayes-verification-v1",
        "attempt_id": attempt["artifact_id"],
        "lock_id": lock["artifact_id"],
        "failure_id": failure["artifact_id"],
        "summary_id": summary["artifact_id"],
        "completed_prefix_batches": 0,
        "completed_future_worlds": 0,
        "scientific_hypothesis_resolved": False,
        "terminal_root_immutable": True,
        "protected_data_read": False,
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(verify(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
