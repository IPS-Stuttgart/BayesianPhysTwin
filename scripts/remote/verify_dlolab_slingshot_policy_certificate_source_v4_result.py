#!/usr/bin/env python3
"""Verify the positive reward-aligned Slingshot v4 source result."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import subprocess
from pathlib import Path
from typing import Any, cast

import numpy as np

from bayesian_phystwin._portable_contracts import content_id, load_strict_json_object
from bayesian_phystwin_experiments.dlolab_native import file_digest
from bayesian_phystwin_experiments.dlolab_regret_artifacts import read_record

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = (
    ROOT / "results/source/dlolab_slingshot_policy_certificate_source_v4/summary.json"
)
ATTEMPT = Path(
    "/home/fpfaff/source-only/dlolab-slingshot-policy-certificate-source-v4.attempt.json"
)
RUNNER_PATH = (
    ROOT / "scripts/remote/run_dlolab_slingshot_policy_certificate_source_v4.py"
)
SPEC = importlib.util.spec_from_file_location("slingshot_v4_result_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
entry: Any = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)
runner: Any = entry.runner


def _tree_identity(root: Path) -> dict[str, Any]:
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    byte_count = 0
    for path in paths:
        if path.is_symlink():
            raise ValueError("raw v4 tree must not contain symlinks")
        relative = path.relative_to(root).as_posix()
        digest.update(f"{file_digest(path)}  ./{relative}\n".encode())
        byte_count += path.stat().st_size
    return {
        "file_count": len(paths),
        "file_byte_count": byte_count,
        "canonical_tree_sha256": digest.hexdigest(),
    }


def _git_blob_digest(revision: str, name: str) -> str:
    value = subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=ROOT)
    return hashlib.sha256(value).hexdigest()


def _verify_frozen_source(lock: dict[str, Any]) -> None:
    revision = lock.get("source_revision")
    hashes = lock.get("source_sha256")
    if not isinstance(revision, str) or not isinstance(hashes, dict):
        raise ValueError("complete frozen v4 source identity required")
    subprocess.run(
        ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    if any(
        not isinstance(name, str)
        or not isinstance(expected, str)
        or _git_blob_digest(revision, name) != expected
        or file_digest(ROOT / name) != expected
        for name, expected in hashes.items()
    ):
        raise ValueError("frozen v4 source blob changed")


def _process_variability(root: Path, role: str, count: int) -> dict[str, Any]:
    qa = [
        read_record(root / f"{role}-future-{index:03d}-qualification.json")["qa"]
        for index in range(count)
    ]
    position = np.asarray([row["duplicate_error_m"] for row in qa])
    reward = np.asarray(
        [
            abs(
                row["metrics"][5]["native_reward"]
                - row["metrics"][7]["native_reward"]
            )
            for row in qa
        ]
    )
    return {
        "worlds": count,
        "reward_aligned_qa_passed": sum(bool(row["qa_passed"]) for row in qa),
        "position_deterministic_worlds": sum(
            bool(row["duplicate_position_deterministic"]) for row in qa
        ),
        "position_nondeterministic_worlds": sum(
            not bool(row["duplicate_position_deterministic"]) for row in qa
        ),
        "maximum_duplicate_position_error_m": float(position.max()),
        "maximum_duplicate_reward_error": float(reward.max()),
    }


def verify(raw_root: Path) -> dict[str, Any]:
    """Rehash the tree and replay every frozen v4 result dependency."""

    summary = dict(load_strict_json_object(SUMMARY, label="Slingshot v4 summary"))
    identity = summary.pop("artifact_id", None)
    if identity != content_id(summary):
        raise ValueError("compact v4 result identity changed")
    summary["artifact_id"] = identity
    if raw_root.resolve() != Path(summary["raw_tree"]["root"]):
        raise ValueError("only the registered raw v4 root is admitted")
    if _tree_identity(raw_root) != {
        key: summary["raw_tree"][key]
        for key in ("file_count", "file_byte_count", "canonical_tree_sha256")
    }:
        raise ValueError("raw v4 tree identity changed")

    key_paths = {
        "attempt.json": ATTEMPT,
        "lock.json": raw_root / "lock.json",
        "calibration/seal.json": raw_root / "calibration/seal.json",
        "evaluation-candidates/seal.json": (
            raw_root / "evaluation-candidates/seal.json"
        ),
        "evaluation-decisions/seal.json": (
            raw_root / "evaluation-decisions/seal.json"
        ),
        "evaluation-decision-barrier.json": (
            raw_root / "evaluation-decision-barrier.json"
        ),
        "result.json": raw_root / "result.json",
    }
    if any(
        file_digest(path) != summary["key_file_sha256"][name]
        for name, path in key_paths.items()
    ):
        raise ValueError("key v4 artifact changed")
    records = {name: read_record(path) for name, path in key_paths.items()}
    identity_names = {
        "attempt.json": "attempt_id",
        "lock.json": "lock_id",
        "calibration/seal.json": "calibration_seal_id",
        "evaluation-candidates/seal.json": "evaluation_candidate_seal_id",
        "evaluation-decisions/seal.json": "evaluation_decision_seal_id",
        "evaluation-decision-barrier.json": "evaluation_barrier_id",
        "result.json": "result_id",
    }
    if any(
        records[path_name]["artifact_id"] != summary["identities"][identity_name]
        for path_name, identity_name in identity_names.items()
    ):
        raise ValueError("raw v4 record identity changed")

    lock = records["lock.json"]
    attempt = records["attempt.json"]
    barrier = records["evaluation-decision-barrier.json"]
    if (
        lock.get("source_revision") != summary["source_revision"]
        or attempt.get("attempt_number") != 1
        or attempt.get("retry_authorized") is not False
        or attempt.get("replacement_authorized") is not False
        or barrier.get("pre_future_gate_passed") is not True
        or barrier.get("future_simulated") is not False
        or barrier.get("future_read") is not False
        or list(raw_root.rglob("failure.json"))
        or len(list(raw_root.glob("calibration-prefix-*/seal.json"))) != 16
        or len(list(raw_root.glob("calibration-future-*-action-*/seal.json")))
        != 1024
        or len(list(raw_root.glob("calibration-future-*-qualification.json"))) != 128
        or len(list(raw_root.glob("evaluation-prefix-*/seal.json"))) != 36
        or len(list(raw_root.glob("evaluation-future-*-action-*/seal.json"))) != 2304
        or len(list(raw_root.glob("evaluation-future-*-qualification.json"))) != 288
    ):
        raise ValueError("complete v4 execution contract changed")

    _verify_frozen_source(lock)
    original_clean_revision = runner.clean_revision
    runner.clean_revision = lambda root: lock["source_revision"]
    try:
        reproduced = runner.verify_result(raw_root)
    finally:
        runner.clean_revision = original_clean_revision
    if reproduced != records["result.json"]:
        raise ValueError("complete v4 result replay changed")
    if {
        "calibration": _process_variability(raw_root, "calibration", 128),
        "evaluation": _process_variability(raw_root, "evaluation", 288),
    } != summary["process_variability"]:
        raise ValueError("v4 process-variability diagnostic changed")
    if (
        reproduced.get("source_gate_passed") is not True
        or reproduced.get("technical_failures") != 0
        or reproduced.get("retries") != 0
        or reproduced.get("replacements") != 0
        or reproduced.get("protected_data_read") is not False
    ):
        raise ValueError("positive v4 source decision changed")
    return cast(dict[str, Any], summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.raw_root)
    print(f"verified positive Slingshot v4 {result['artifact_id']}", flush=True)
