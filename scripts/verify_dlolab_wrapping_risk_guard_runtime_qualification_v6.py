#!/usr/bin/env python3
"""Verify the terminal corrected runtime-qualification evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin_experiments.dlolab_native import file_digest
from bayesian_phystwin_experiments.dlolab_regret_artifacts import read_record

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(
    "/home/fpfaff/source-only/dlolab-wrapping-risk-guard-runtime-qualification-v6"
)
ATTEMPT = Path(
    "/home/fpfaff/source-only/"
    "dlolab-wrapping-risk-guard-runtime-qualification-v6.attempt.json"
)
CORE = Path(
    "/mnt/c/Users/emper/AppData/Local/Temp/wsl-crashes/"
    "wsl-crash-1788009817-356313-_home_fpfaff_.local_share_uv_python_"
    "cpython-3.11.15-linux-x86_64-gnu_bin_python3.11-11.dmp"
)
SUMMARY = (
    ROOT
    / "results/sota/dlolab_wrapping_risk_guard_runtime_qualification_v6/summary.json"
)
REVISION = "50931b10651cf7d17210cc1ec8d81639225d7112"
EXPECTED_TREE_ID = "75553b1a7fced1fd39ded42dbf5fba0c272cb5cf8ada523f0813c342fd783f64"
EXPECTED_CORE_SHA256 = (
    "7c9a5497acc3864a873b976a675e6e84db94635ec292496c09c4f1a92f4e816f"
)
EXPECTED_SUMMARY_ID = "1e380f455e780988a3d9542c5e313a13dd0a47ca2c385f636722daa696932965"

SPEC = importlib.util.spec_from_file_location(
    "wrapping_runtime_qualification_v6_verifier_runner",
    ROOT / "scripts/remote/run_dlolab_wrapping_risk_guard_runtime_qualification_v6.py",
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _git_blob_sha256(revision: str, name: str) -> str:
    value = subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=ROOT)
    return hashlib.sha256(value).hexdigest()


def _tree() -> dict[str, str]:
    return {
        str(path.relative_to(OUTPUT)): file_digest(path)
        for path in sorted(OUTPUT.rglob("*"))
        if path.is_file()
    }


def verify() -> dict[str, Any]:
    tree = _tree()
    if len(tree) != 93 or content_id({"files": tree}) != EXPECTED_TREE_ID:
        raise ValueError("runtime v6 evidence tree changed")
    attempt = read_record(ATTEMPT)
    lock = read_record(OUTPUT / "lock.json")
    failure = read_record(OUTPUT / "failure.json")
    process_failure = read_record(OUTPUT / "constructor-22/process-failure.json")
    summary = read_record(SUMMARY)
    if any(
        _git_blob_sha256(REVISION, name) != digest
        for name, digest in lock["source_sha256"].items()
    ):
        raise ValueError("runtime v6 source differs from frozen revision")
    seal_ids: list[str] = []
    for index in range(22):
        task = runner._task("constructor", index)
        directory = OUTPUT / task["name"]
        seal, qa = runner._load_success(OUTPUT, lock, task)
        if not qa["qa_passed"] or (directory / "failure.json").exists():
            raise ValueError("completed runtime constructor changed")
        seal_ids.append(seal["artifact_id"])
    failed_claim = read_record(OUTPUT / "constructor-22/claim.json")
    failed_log = (OUTPUT / "constructor-22.log").read_text(encoding="utf-8")
    if (
        attempt.get("artifact_id")
        != "f97ed8c6950eba9d01a28992026df8818637edd85fa56bc5245081194ef5121c"
        or attempt.get("revision") != REVISION
        or lock.get("artifact_id")
        != "f7e06e6da9bad340c094e0a2e598dd9a4dfe1800f98b537bbcf9fada2caf7802"
        or lock.get("attempt_id") != attempt["artifact_id"]
        or lock.get("runtime", {}).get("python") != "3.11.15"
        or failure.get("artifact_id")
        != "0b2c9a8afacc8e4e309baa5683fc1b21d9e7ca9578cbd50a01264bd29af7df29"
        or failure.get("constructor_successes") != 22
        or failure.get("full_rollout_successes") != 0
        or failure.get("terminal_task") != runner._task("constructor", 22)
        or process_failure.get("artifact_id")
        != "f15761056601da7b135309d497a36fc9486d74ca706427839a3834de098907b5"
        or process_failure.get("claim_id") != failed_claim["artifact_id"]
        or process_failure.get("returncode") != -11
        or "'task': 'wrapping'" not in failed_log
        or "Initial total length" in failed_log
        or (OUTPUT / "constructor-22/seal.json").exists()
        or (OUTPUT / "constructor-22/arrays.npz").exists()
        or (OUTPUT / "constructor-23").exists()
        or any((OUTPUT / f"full-rollout-{index:02d}").exists() for index in range(4))
        or (OUTPUT / "result.json").exists()
        or any(
            record.get("retry_authorized") is not False
            or record.get("replacement_authorized") is not False
            or record.get("protected_data_read") is not False
            for record in (attempt, lock, failure, failed_claim, process_failure)
        )
        or summary.get("artifact_id") != EXPECTED_SUMMARY_ID
        or summary.get("tree_id") != EXPECTED_TREE_ID
        or summary.get("tree_file_count") != 93
        or summary.get("constructor_successes") != 22
        or summary.get("full_rollout_successes") != 0
        or summary.get("process_failure_id") != process_failure["artifact_id"]
        or summary.get("qualification_passed") is not False
        or summary.get("scientific_outcome_scored") is not False
        or summary.get("retry_authorized") is not False
        or summary.get("v4_partial_future_artifacts_read") is not False
        or summary.get("v5_runtime_artifacts_read") is not False
        or len(seal_ids) != 22
    ):
        raise ValueError("runtime v6 terminal decision changed")
    if not CORE.is_file() or file_digest(CORE) != EXPECTED_CORE_SHA256:
        raise ValueError("runtime v6 crash core changed")
    return {
        "schema": "dlolab-wrapping-risk-guard-runtime-qualification-verification-v6",
        "status": "verified_terminal_native_sigsegv",
        "summary_id": summary["artifact_id"],
        "failure_id": failure["artifact_id"],
        "tree_id": EXPECTED_TREE_ID,
        "tree_file_count": len(tree),
        "constructor_seal_ids": seal_ids,
        "core_sha256": EXPECTED_CORE_SHA256,
        "ordinary_constructor_successes": 22,
        "ordinary_full_rollout_successes": 0,
        "retry_authorized": False,
        "scientific_result_available": False,
        "protected_data_read": False,
    }


def main() -> None:
    print(json.dumps(verify(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
