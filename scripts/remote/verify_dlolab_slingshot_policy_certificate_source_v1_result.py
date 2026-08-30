#!/usr/bin/env python3
"""Verify the retained Slingshot policy-certificate pre-future result."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, cast

from bayesian_phystwin._portable_contracts import content_id, load_strict_json_object
from bayesian_phystwin_experiments.dlolab_native import file_digest
from bayesian_phystwin_experiments.dlolab_regret_artifacts import read_record

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = (
    ROOT
    / "results/source/dlolab_slingshot_policy_certificate_source_v1/summary.json"
)


def _tree_identity(root: Path) -> dict[str, Any]:
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    byte_count = 0
    for path in paths:
        if path.is_symlink():
            raise ValueError("raw result tree must not contain symlinks")
        relative = path.relative_to(root).as_posix()
        digest.update(f"{file_digest(path)}  ./{relative}\n".encode())
        byte_count += path.stat().st_size
    return {
        "file_count": len(paths),
        "file_byte_count": byte_count,
        "canonical_tree_sha256": digest.hexdigest(),
    }


def verify(raw_root: Path) -> dict[str, Any]:
    """Verify compact identity, retained failure, and zero-future custody."""

    summary = dict(load_strict_json_object(SUMMARY, label="policy certificate result"))
    identity = summary.pop("artifact_id", None)
    if identity != content_id(summary):
        raise ValueError("compact result content identity changed")
    summary["artifact_id"] = identity
    if raw_root.resolve() != Path(summary["raw_tree"]["root"]):
        raise ValueError("only the registered raw result root is admitted")
    if any(
        file_digest(raw_root / name) != expected
        for name, expected in summary["key_file_sha256"].items()
    ):
        raise ValueError("key raw result artifact changed")
    tree = _tree_identity(raw_root)
    if tree != {
        key: summary["raw_tree"][key]
        for key in ("file_count", "file_byte_count", "canonical_tree_sha256")
    }:
        raise ValueError("raw result tree identity changed")
    records = {
        "lock_id": read_record(raw_root / "lock.json"),
        "calibration_seal_id": read_record(raw_root / "calibration/seal.json"),
        "evaluation_candidate_seal_id": read_record(
            raw_root / "evaluation-candidates/seal.json"
        ),
        "evaluation_decision_seal_id": read_record(
            raw_root / "evaluation-decisions/seal.json"
        ),
        "evaluation_barrier_id": read_record(
            raw_root / "evaluation-decision-barrier.json"
        ),
        "failure_id": read_record(raw_root / "failure.json"),
    }
    if any(
        records[name]["artifact_id"] != expected
        for name, expected in summary["identities"].items()
    ):
        raise ValueError("raw record content identity changed")
    barrier = records["evaluation_barrier_id"]
    failure = records["failure_id"]
    if (
        barrier.get("pre_future") != summary["pre_future"]
        or barrier.get("pre_future_gate_passed") is not False
        or barrier.get("future_simulated") is not False
        or barrier.get("future_read") is not False
        or failure.get("terminal_stage") != "evaluation-decision-barrier"
        or failure.get("retry_authorized") is not False
        or failure.get("replacement_authorized") is not False
        or len(list(raw_root.glob("calibration-future-*/seal.json"))) != 96
        or len(list(raw_root.glob("evaluation-prefix-*/seal.json"))) != 36
        or list(raw_root.glob("evaluation-future-*"))
    ):
        raise ValueError("retained pre-future failure contract changed")
    return cast(dict[str, Any], summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.raw_root)
    print(f"verified retained result {result['artifact_id']}", flush=True)
