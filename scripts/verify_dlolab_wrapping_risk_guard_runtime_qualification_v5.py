#!/usr/bin/env python3
"""Verify the terminal Python 3.11 runtime-qualification evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from bayesian_phystwin_experiments.dlolab_native import file_digest
from bayesian_phystwin_experiments.dlolab_regret_artifacts import read_record

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(
    "/home/fpfaff/source-only/dlolab-wrapping-risk-guard-runtime-qualification-v5"
)
ATTEMPT = Path(
    "/home/fpfaff/source-only/"
    "dlolab-wrapping-risk-guard-runtime-qualification-v5.attempt.json"
)
SUMMARY = (
    ROOT
    / "results/sota/dlolab_wrapping_risk_guard_runtime_qualification_v5/summary.json"
)
REVISION = "02b780795a968f14f44485e5fd0c1053e52bf025"
EXPECTED_FILE_SHA256 = {
    "attempt.json": "04b51f43f9eac33eb416dfc5112ffd173ee5b7b57126819a8cbcf9ae4a3c34aa",
    "lock.json": "3faf8e484e78c355c743b7e5cde0895c7c2edd146c72c2a8d2e5e25e38d301ce",
    "constructor-00/claim.json": (
        "dc1847cdd4d7da5ce909c7f9f1200e96610da05b0291ef302d0c5a7fb6f9134e"
    ),
    "constructor-00/failure.json": (
        "4afc7aec39db0534ccf51736923d1be3797530e631a78574f772ec3bcb2adbef"
    ),
    "constructor-00.log": (
        "ce737ea6d7e2c754666a7f8d161e4ff513d8a569fac14905f2fa7153ad379c61"
    ),
    "failure.json": "f973af071184667ebe59044b4b57c920c35a3b0967085bf790bbf02d083e3272",
}
EXPECTED_SUMMARY_ID = "9fbd92685e9417057f28c5165862dd000ef27ab35106854aaa93589910d3e7b0"


def _git_blob_sha256(revision: str, name: str) -> str:
    value = subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=ROOT)
    return hashlib.sha256(value).hexdigest()


def verify() -> dict[str, Any]:
    paths = {
        "attempt.json": ATTEMPT,
        "lock.json": OUTPUT / "lock.json",
        "constructor-00/claim.json": OUTPUT / "constructor-00/claim.json",
        "constructor-00/failure.json": OUTPUT / "constructor-00/failure.json",
        "constructor-00.log": OUTPUT / "constructor-00.log",
        "failure.json": OUTPUT / "failure.json",
    }
    actual_hashes = {name: file_digest(path) for name, path in paths.items()}
    if actual_hashes != EXPECTED_FILE_SHA256:
        raise ValueError("runtime qualification evidence bytes changed")
    files = {
        str(path.relative_to(OUTPUT)) for path in OUTPUT.rglob("*") if path.is_file()
    }
    if files != {
        "lock.json",
        "constructor-00.log",
        "constructor-00/claim.json",
        "constructor-00/failure.json",
        "failure.json",
    }:
        raise ValueError("runtime qualification artifact roster changed")
    attempt = read_record(ATTEMPT)
    lock = read_record(OUTPUT / "lock.json")
    claim = read_record(OUTPUT / "constructor-00/claim.json")
    task_failure = read_record(OUTPUT / "constructor-00/failure.json")
    failure = read_record(OUTPUT / "failure.json")
    summary = read_record(SUMMARY)
    if any(
        _git_blob_sha256(REVISION, name) != digest
        for name, digest in lock["source_sha256"].items()
    ):
        raise ValueError("qualified source differs from frozen revision")
    log = (OUTPUT / "constructor-00.log").read_text(encoding="utf-8")
    if (
        attempt.get("revision") != REVISION
        or attempt.get("artifact_id")
        != "7d18dfc0b07ac9e1b986191d42fae919f030029223c684276eedc625a5dd5d93"
        or lock.get("artifact_id")
        != "cf14f6b4533a4984988f24a699a9f9666e0290c5c7fb697a30e3d34be65dc9de"
        or lock.get("attempt_id") != attempt["artifact_id"]
        or lock.get("runtime", {}).get("python") != "3.11.15"
        or claim.get("artifact_id")
        != "d38dec6bbe0280b02e7f11bf6f831265cbdf720b277433be3f193fed340ab986"
        or task_failure.get("artifact_id")
        != "aee9011af51f11309a5f06733b692ee8543dae1f79a93fabe50207140e82400e"
        or task_failure.get("claim_id") != claim["artifact_id"]
        or task_failure.get("error_type") != "RuntimeError"
        or task_failure.get("message") != "native material realization was not captured"
        or failure.get("artifact_id")
        != "a83bfdece3cc2322a97a940ea74cd3232752bcb1ae7a688f15828141abccffab"
        or failure.get("constructor_successes") != 0
        or failure.get("full_rollout_successes") != 0
        or failure.get("qualification_passed") is not False
        or "Initial total length: 0.8791" not in log
        or "native material realization was not captured" not in log
        or (OUTPUT / "constructor-00/seal.json").exists()
        or (OUTPUT / "constructor-00/arrays.npz").exists()
        or (OUTPUT / "result.json").exists()
        or any(
            record.get("retry_authorized") is not False
            or record.get("replacement_authorized") is not False
            or record.get("protected_data_read") is not False
            for record in (attempt, lock, claim, task_failure, failure)
        )
        or summary.get("artifact_id") != EXPECTED_SUMMARY_ID
        or summary.get("status") != "terminal_technical_failure"
        or summary.get("failure_id") != failure["artifact_id"]
        or summary.get("task_failure_id") != task_failure["artifact_id"]
        or summary.get("constructor_successes") != 0
        or summary.get("full_rollout_successes") != 0
        or summary.get("qualification_passed") is not False
        or summary.get("retry_authorized") is not False
        or summary.get("scientific_outcome_scored") is not False
        or summary.get("v4_partial_future_artifacts_read") is not False
    ):
        raise ValueError("runtime qualification terminal decision changed")
    return {
        "schema": "dlolab-wrapping-risk-guard-runtime-qualification-verification-v5",
        "status": "verified_terminal_technical_failure",
        "summary_id": summary["artifact_id"],
        "failure_id": failure["artifact_id"],
        "file_sha256": actual_hashes,
        "ordinary_constructor_successes": 0,
        "ordinary_full_rollout_successes": 0,
        "retry_authorized": False,
        "scientific_result_available": False,
        "protected_data_read": False,
    }


def main() -> None:
    print(json.dumps(verify(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
