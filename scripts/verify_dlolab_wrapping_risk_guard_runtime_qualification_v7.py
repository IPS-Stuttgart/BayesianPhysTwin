#!/usr/bin/env python3
"""Verify the successful native-Linux runtime-qualification evidence."""

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
    "/home/fpfaff/source-only/dlolab-wrapping-risk-guard-runtime-qualification-v7"
)
ATTEMPT = Path(
    "/home/fpfaff/source-only/"
    "dlolab-wrapping-risk-guard-runtime-qualification-v7.attempt.json"
)
SUMMARY = (
    ROOT
    / "results/sota/dlolab_wrapping_risk_guard_runtime_qualification_v7/summary.json"
)
REVISION = "cfc8e8e5533c4bc98104b7051c4eefd4586f2779"
EXPECTED_TREE_ID = "d0b772d13b92d09699e1942128c68d237815a736c54580f7bb2041a8a30c585a"
EXPECTED_SUMMARY_ID = "24bc06374ff8e5c392304b1b3091e346172b41e1ac8a22081d1efdaa52ff611e"
EXPECTED_RESULT_ID = "a147939df81acd11580f00405ae96a7b198909d00705b79d6636f228af0b0ee7"
EXPECTED_LOCK_ID = "59a62164c99d91713cbf64d0ed5f6bb5213748f4f4174b91efe0a3fffd4d5c81"
EXPECTED_ATTEMPT_ID = (
    "7b5b900037398c27e7c740fe82ac8db49bd516faf68e08404410ac1ab7239157"
)

SPEC = importlib.util.spec_from_file_location(
    "wrapping_runtime_qualification_v7_verifier_runner",
    ROOT / "scripts/remote/run_dlolab_wrapping_risk_guard_runtime_qualification_v7.py",
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
    if len(tree) != 114 or content_id({"files": tree}) != EXPECTED_TREE_ID:
        raise ValueError("runtime v7 evidence tree changed")
    attempt = read_record(ATTEMPT)
    lock = read_record(OUTPUT / "lock.json")
    result = read_record(OUTPUT / "result.json")
    summary = read_record(SUMMARY)
    if any(
        _git_blob_sha256(REVISION, name) != digest
        for name, digest in lock["source_sha256"].items()
    ):
        raise ValueError("runtime v7 source differs from frozen revision")
    constructor_ids: list[str] = []
    for index in range(24):
        task = runner._task("constructor", index)
        directory = OUTPUT / task["name"]
        seal, qa = runner._load_success(OUTPUT, lock, task)
        if not qa["qa_passed"] or any(
            (directory / name).exists()
            for name in ("failure.json", "process-failure.json")
        ):
            raise ValueError("completed runtime constructor changed")
        constructor_ids.append(seal["artifact_id"])
    rollout_ids: list[str] = []
    for index in range(4):
        task = runner._task("full", index)
        directory = OUTPUT / task["name"]
        seal, qa = runner._load_success(OUTPUT, lock, task)
        if not qa["qa_passed"] or any(
            (directory / name).exists()
            for name in ("failure.json", "process-failure.json")
        ):
            raise ValueError("completed runtime rollout changed")
        rollout_ids.append(seal["artifact_id"])
    boundary_records = (attempt, lock, result, summary)
    if (
        attempt.get("artifact_id") != EXPECTED_ATTEMPT_ID
        or attempt.get("revision") != REVISION
        or lock.get("artifact_id") != EXPECTED_LOCK_ID
        or lock.get("attempt_id") != attempt["artifact_id"]
        or lock.get("revision") != REVISION
        or lock.get("runtime", {}).get("host", {}).get("wsl") is not False
        or lock.get("runtime", {}).get("host", {}).get("hostname")
        != "workstation2"
        or result.get("artifact_id") != EXPECTED_RESULT_ID
        or result.get("status") != "complete"
        or result.get("constructor_seal_ids") != constructor_ids
        or result.get("full_rollout_seal_ids") != rollout_ids
        or result.get("constructor_successes") != 24
        or result.get("full_rollout_successes") != 4
        or result.get("qualification_passed") is not True
        or (OUTPUT / "failure.json").exists()
        or summary.get("artifact_id") != EXPECTED_SUMMARY_ID
        or summary.get("tree_id") != EXPECTED_TREE_ID
        or summary.get("tree_file_count") != 114
        or summary.get("result_id") != result["artifact_id"]
        or summary.get("constructor_successes") != 24
        or summary.get("full_rollout_successes") != 4
        or summary.get("qualification_passed") is not True
        or any(
            record.get("retry_authorized") is not False
            or record.get("replacement_authorized") is not False
            or record.get("protected_data_read") is not False
            for record in boundary_records
        )
        or result.get("fresh_scientific_worlds_defined") is not False
        or result.get("scientific_outcome_scored") is not False
        or result.get("study_automatically_authorized") is not False
    ):
        raise ValueError("runtime v7 qualification decision changed")
    return {
        "schema": "dlolab-wrapping-risk-guard-runtime-qualification-verification-v7",
        "status": "verified_complete",
        "summary_id": summary["artifact_id"],
        "result_id": result["artifact_id"],
        "tree_id": EXPECTED_TREE_ID,
        "tree_file_count": len(tree),
        "ordinary_constructor_successes": len(constructor_ids),
        "ordinary_full_rollout_successes": len(rollout_ids),
        "native_linux": True,
        "retry_authorized": False,
        "scientific_result_available": False,
        "protected_data_read": False,
    }


def main() -> None:
    print(json.dumps(verify(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
