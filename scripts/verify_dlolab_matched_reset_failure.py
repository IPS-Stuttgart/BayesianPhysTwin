#!/usr/bin/env python3
"""Verify the terminal mixed-horizon transport failure without native replay."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from bayesian_phystwin._portable_contracts import content_id, load_strict_json_object
from bayesian_phystwin_experiments.deform_state_restart import file_digest

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("/home/fpfaff/source-only/dlolab-matched-reset-dual-control-source-v1")
REVISION = "953935820a842818e9db62caa5be7a5e603daf43"
EXPECTED = {
    "lock.json": "994db989a4b411d3f3eac6ca6d7cb5d078bb964bf56a68148aaf973630fff567",
    "failure.json": "f54693ea07f12f5c5e187c6d36d0f45212eae92b0133ecd20cfb9c740d4aad8c",
    "result.json": "780cc80e3b676a5a4cde5fa088f89ef5e6c394253236b4231347f50dfe75c752",
    "particle-bank.log": "1b47f4b8273193ac1ab8e211547a7dc7a10ccbce583b205aabd68eac495be2c6",
}


def _record(path: Path) -> dict[str, object]:
    value = dict(load_strict_json_object(path, label="matched-reset failure record"))
    identity = value.pop("artifact_id", None)
    if identity != content_id(value):
        raise ValueError("failure record content identity changed")
    return {**value, "artifact_id": identity}


def verify(output: Path) -> dict[str, object]:
    if output.resolve() != OUTPUT or output.is_symlink():
        raise ValueError("registered terminal output root required")
    files = {
        str(path.relative_to(output))
        for path in output.rglob("*")
        if path.is_file()
    }
    if files != set(EXPECTED) or any(
        file_digest(output / name) != digest for name, digest in EXPECTED.items()
    ):
        raise ValueError("terminal failure artifact set changed")
    lock = _record(output / "lock.json")
    failure = _record(output / "failure.json")
    result = _record(output / "result.json")
    if (
        lock["source_revision"] != REVISION
        or failure["lock_id"] != lock["artifact_id"]
        or failure["terminal_stage"] != "particle-bank"
        or failure["retry_authorized"] is not False
        or result["lock_id"] != lock["artifact_id"]
        or result["status"] != "technical_failure"
        or result["task_futures_generated"] is not False
        or result["truth_probe_observations_generated"] is not False
        or result["decisions_sealed"] is not False
        or result["retry_authorized"] is not False
    ):
        raise ValueError("terminal failure accounting changed")
    frozen_source = subprocess.check_output(
        [
            "git",
            "show",
            f"{REVISION}:src/bayesian_phystwin_experiments/dlolab_matched_reset_native.py",
        ],
        cwd=ROOT,
        text=True,
    )
    log = (output / "particle-bank.log").read_text()
    if (
        "return np.stack(trajectories)" not in frozen_source
        or "ValueError: all input arrays must have the same shape" not in log
        or "generate_particle_bank" not in log
    ):
        raise ValueError("mixed-horizon failure mechanism changed")
    return {
        "schema": "dlolab-matched-reset-failure-verification-v1",
        "revision": REVISION,
        "lock_id": lock["artifact_id"],
        "failure_id": failure["artifact_id"],
        "result_id": result["artifact_id"],
        "artifact_count": len(files),
        "mixed_horizon_failure_reproduced_from_source_and_log": True,
        "particle_bank_published": False,
        "probe_information_computed": False,
        "truth_stage_exists": False,
        "retry_authorized": False,
        "native_replay_performed": False,
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
