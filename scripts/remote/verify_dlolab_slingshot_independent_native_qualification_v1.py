#!/usr/bin/env python3
"""Verify the complete independent-process Slingshot qualification."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import subprocess
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from bayesian_phystwin._portable_contracts import content_id, load_strict_json_object
from bayesian_phystwin_experiments.dlolab_native import file_digest
from bayesian_phystwin_experiments.dlolab_regret_artifacts import read_record
from bayesian_phystwin_experiments.dlolab_slingshot_independent_native_v3 import (
    ACTION_COUNT,
    PROCESS_COUNT,
    WORLD_COUNT,
    independent_world_qa,
    protocol,
    qualification_worlds,
)

ROOT = Path(__file__).resolve().parents[2]
Array: TypeAlias = NDArray[Any]
SUMMARY = (
    ROOT
    / "results/source/dlolab_slingshot_independent_native_qualification_v1/summary.json"
)
ATTEMPT = Path(
    "/home/fpfaff/source-only/"
    "dlolab-slingshot-independent-native-qualification-v1.attempt.json"
)
RUNNER_PATH = (
    ROOT / "scripts/remote/run_dlolab_slingshot_independent_native_qualification_v1.py"
)
SPEC = importlib.util.spec_from_file_location("independent_native_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _tree_identity(root: Path) -> dict[str, Any]:
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    byte_count = 0
    for path in paths:
        if path.is_symlink():
            raise ValueError("raw qualification tree must not contain symlinks")
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


def _verify_source(lock: dict[str, Any]) -> None:
    revision = lock.get("source_revision")
    hashes = lock.get("source_sha256")
    if not isinstance(revision, str) or not isinstance(hashes, dict):
        raise ValueError("complete frozen source identity required")
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
        for name, expected in hashes.items()
    ):
        raise ValueError("frozen source blob changed")


def verify(raw_root: Path) -> dict[str, Any]:
    summary = dict(load_strict_json_object(SUMMARY, label="qualification summary"))
    identity = summary.pop("artifact_id", None)
    if identity != content_id(summary):
        raise ValueError("compact qualification identity changed")
    summary["artifact_id"] = identity
    if raw_root.resolve() != Path(summary["raw_tree"]["root"]):
        raise ValueError("only the registered raw qualification root is admitted")
    if _tree_identity(raw_root) != {
        key: summary["raw_tree"][key]
        for key in ("file_count", "file_byte_count", "canonical_tree_sha256")
    }:
        raise ValueError("raw qualification tree identity changed")
    if (
        file_digest(raw_root / "lock.json") != summary["key_file_sha256"]["lock.json"]
        or file_digest(raw_root / "result.json")
        != summary["key_file_sha256"]["result.json"]
        or file_digest(ATTEMPT) != summary["key_file_sha256"]["attempt.json"]
    ):
        raise ValueError("key qualification artifact changed")

    lock = read_record(raw_root / "lock.json")
    attempt = read_record(ATTEMPT)
    result = read_record(raw_root / "result.json")
    _verify_source(lock)
    if (
        lock.get("artifact_id") != summary["identities"]["lock_id"]
        or lock.get("source_revision") != summary["source_revision"]
        or lock.get("protocol") != protocol()
        or attempt.get("artifact_id") != summary["identities"]["attempt_id"]
        or attempt.get("lock_id") != lock["artifact_id"]
        or attempt.get("attempt_number") != 1
        or attempt.get("retry_authorized") is not False
        or result.get("artifact_id") != summary["identities"]["result_id"]
        or result.get("status") != "passed"
        or result.get("ordinary_processes") != PROCESS_COUNT
        or result.get("failed_process_indices") != []
        or result.get("custody_validation_errors") != {}
        or result.get("qualified_worlds") != WORLD_COUNT
        or result.get("qualification_passed") is not True
        or result.get("v3_protocol_freeze_authorized") is not True
        or result.get("v3_scientific_execution_authorized") is not False
        or result.get("retry_authorized") is not False
        or result.get("replacement_authorized") is not False
    ):
        raise ValueError("qualification result contract changed")

    controls = np.asarray(lock["controls"], dtype=np.float64)
    world_ids: list[str] = []
    common: list[float] = []
    duplicate: list[float] = []
    duplicate_reward: list[float] = []
    fixed: list[float] = []
    forward: list[float] = []
    wall: list[float] = []
    for world_index, world in enumerate(qualification_worlds()):
        rows: list[dict[str, Array]] = []
        reports: list[dict[str, Any]] = []
        seal_ids: list[str] = []
        for action_index in range(ACTION_COUNT):
            index = world_index * ACTION_COUNT + action_index
            seal, arrays = runner.load_task(raw_root, lock, index)
            rows.append(arrays)
            reports.append(seal["native"])
            seal_ids.append(seal["artifact_id"])
            fixed.append(float(seal["qa"]["fixed_endpoint_error_m"]))
            forward.append(float(seal["native"]["native_forward_seconds"]))
            wall.append(float(seal["native"]["wall_seconds"]))
        qa = independent_world_qa(rows, reports, controls, world)
        stored = read_record(raw_root / f"world-{world_index:02d}-qualification.json")
        if (
            stored.get("lock_id") != lock["artifact_id"]
            or stored.get("world") != world
            or stored.get("source_seal_ids") != seal_ids
            or stored.get("qa") != qa
            or not qa["qa_passed"]
        ):
            raise ValueError("world qualification failed to rederive")
        world_ids.append(stored["artifact_id"])
        common.append(float(qa["common_prefix_error_m"]))
        duplicate.append(float(qa["duplicate_error_m"]))
        duplicate_reward.append(
            abs(
                float(qa["metrics"][5]["native_reward"])
                - float(qa["metrics"][7]["native_reward"])
            )
        )
    metrics = {
        "maximum_common_prefix_error_m": max(common),
        "maximum_duplicate_position_error_m": max(duplicate),
        "maximum_duplicate_reward_error": max(duplicate_reward),
        "maximum_fixed_endpoint_error_m": max(fixed),
        "sum_native_forward_seconds": sum(forward),
        "maximum_process_wall_seconds": max(wall),
    }
    if (
        world_ids != summary["identities"]["world_qualification_ids"]
        or world_ids != result["world_qualification_ids"]
        or metrics != summary["metrics"]
        or len(list(raw_root.glob("world-*/claim.json"))) != PROCESS_COUNT
        or len(list(raw_root.glob("world-*/seal.json"))) != PROCESS_COUNT
        or list(raw_root.glob("world-*/failure.json"))
    ):
        raise ValueError("complete qualification evidence changed")
    return cast(dict[str, Any], summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.raw_root)
    print(f"verified independent qualification {result['artifact_id']}", flush=True)
