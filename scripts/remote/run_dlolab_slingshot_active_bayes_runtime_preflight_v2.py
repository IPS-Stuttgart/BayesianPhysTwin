#!/usr/bin/env python3
"""Qualify the exact native runtime for the active-Bayes v2 successor."""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin_experiments.dlolab_benchmark import write_native_bundle
from bayesian_phystwin_experiments.dlolab_native import array_digest, file_digest
from bayesian_phystwin_experiments.dlolab_regret_artifacts import (
    clean_revision,
    read_record,
    write_record,
)
from bayesian_phystwin_experiments.dlolab_slingshot_belief import particle_worlds
from bayesian_phystwin_experiments.dlolab_slingshot_belief_native import (
    run_registered_worlds,
)
from bayesian_phystwin_experiments.dlolab_slingshot_process import runtime

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(
    "/home/fpfaff/source-only/dlolab-slingshot-active-bayes-runtime-preflight-v2"
)
ATTEMPT = Path(
    "/home/fpfaff/source-only/"
    "dlolab-slingshot-active-bayes-runtime-preflight-v2.attempt.json"
)
PARENT = Path(
    "/home/fpfaff/source-only/dlolab-benchmark-source-v1/belief-control-source-v1"
)
V1 = Path("/home/fpfaff/source-only/dlolab-slingshot-active-bayes-source-v1")
V1_ATTEMPT = Path(
    "/home/fpfaff/source-only/dlolab-slingshot-active-bayes-source-v1.attempt.json"
)
EXPECTED_PYTHON = Path(
    "/home/fpfaff/source-only/dlolab-benchmark-source-v1-assets/venv/bin/python"
)
PARENT_LOCK_ID = "015e6d84aa68a2a4310552ef4880752b972890f02d3e09e333ff575c92b8df25"
PARENT_LOCK_SHA256 = "6dce35441588c2a5eff9c0ae08d85c8b41ff660403541dd489b8d9161bffcc8d"
V1_FILE_SHA256 = {
    "attempt.json": "da7129b00c5f2699edb724f28cd1c2d712f805d87ff4476b8550b51a4a11e0c5",
    "failure.json": "0433eeb588401d37b9c6798f79eeaefc45e5ad9bf3cc2c43f94635c1aab599ae",
    "lock.json": "019d06450fb0694bbc1eb506c0726c25eaafbf2a8b88666316b2efd1701768a3",
    "prefix-passive-0.log": "fa78ddb9b428fab2660b9d7f935fff4514842d98adf7def4f85d20b49d38e225",
    "prefix-passive-0/claim.json": "5bd571f0f3173c7bdd0d1ba3816322695ea4f6d3b9967c411a526e7399a851de",
    "prefix-passive-0/failure.json": "14a91e2bc9e7e385b027061d063c4832f5ff71174ba4572b6144e055d98ad633",
}
SOURCE_FILES = (
    "scripts/remote/run_dlolab_slingshot_active_bayes_runtime_preflight_v2.py",
    "tests/test_dlolab_slingshot_active_bayes_runtime_preflight_v2.py",
    "docs/dlolab_slingshot_active_bayes_runtime_preflight_v2.md",
)


def protocol() -> dict[str, Any]:
    return {
        "schema": "dlolab-slingshot-active-bayes-runtime-preflight-v2",
        "role": "pre_attempt_runtime_qualification_after_terminal_v1_import_failure",
        "v1_retried": False,
        "v1_scientific_result": False,
        "expected_python": str(EXPECTED_PYTHON),
        "required_packages": {"mediapy": "1.2.7", "genesis-world": "1.0.0"},
        "world": particle_worlds()[0],
        "batch_slots": 8,
        "control": "registered_parent_passive_action_5",
        "prefix_steps": 300,
        "future_simulated": False,
        "reward_scored": False,
        "study_attempt_consumed": False,
        "retry_authorized": False,
        "replacement_authorized": False,
        "protected_data_read": False,
        "held_v8_read": False,
        "dlo4_dlo5_read": False,
        "official_dlo3_evaluation": False,
        "new_recordings": False,
        "gpu_work": False,
    }


def _v1_path(name: str) -> Path:
    return V1_ATTEMPT if name == "attempt.json" else V1 / name


def _source() -> tuple[dict[str, Any], np.ndarray]:
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise ValueError("registered parent benchmark interpreter required")
    if file_digest(PARENT / "lock.json") != PARENT_LOCK_SHA256:
        raise ValueError("registered parent lock changed")
    if any(
        file_digest(_v1_path(name)) != digest
        for name, digest in V1_FILE_SHA256.items()
    ):
        raise ValueError("terminal active-Bayes v1 root changed")
    parent = read_record(PARENT / "lock.json")
    v1_attempt = read_record(V1_ATTEMPT)
    v1_failure = read_record(V1 / "failure.json")
    child_failure = read_record(V1 / "prefix-passive-0" / "failure.json")
    if (
        parent.get("artifact_id") != PARENT_LOCK_ID
        or runtime() != parent["screen"]["source"]["controller"]["runtime"]
        or importlib.metadata.version("mediapy") != "1.2.7"
        or importlib.metadata.version("genesis-world") != "1.0.0"
        or v1_attempt.get("protocol", {}).get("parent_output_retried") is not False
        or v1_failure.get("completed_prefix_batches") != 0
        or v1_failure.get("completed_future_worlds") != 0
        or v1_failure.get("retry_authorized") is not False
        or child_failure.get("error_type") != "ModuleNotFoundError"
        or child_failure.get("message") != "No module named 'mediapy'"
    ):
        raise ValueError("registered parent runtime or terminal v1 evidence changed")
    controls = np.asarray(parent["controls"], dtype=np.float64)
    if controls.shape != (8, 3, 6) or not np.array_equal(controls[5], controls[7]):
        raise ValueError("registered parent controls changed")
    return parent, controls


def _source_hashes() -> dict[str, str]:
    if any(not (ROOT / name).is_file() for name in SOURCE_FILES):
        raise ValueError("complete runtime-preflight source required")
    return {name: file_digest(ROOT / name) for name in SOURCE_FILES}


def _world_realization(native: dict[str, Any], world: dict[str, Any]) -> bool:
    expected = {
        "bending": [[world["bending_E"]] * 8],
        "stretching": [[world["stretching_K"]] * 8],
        "sphere_initial_position_m": [
            [0.12 + world["x_offset_m"], 0.06, 0.2]
        ]
        * 8,
        "cube_initial_position_m": [[0.12 + world["x_offset_m"], 0.23, 0.22]]
        * 8,
    }
    return bool(native.get("world_realization") == expected)


def run(output: Path) -> None:
    if (
        output.resolve() != OUTPUT
        or output.exists()
        or output.is_symlink()
        or ATTEMPT.exists()
        or ATTEMPT.is_symlink()
    ):
        raise ValueError("one fresh registered runtime preflight required")
    parent, controls = _source()
    revision = clean_revision(ROOT)
    sources = _source_hashes()
    attempt = write_record(
        ATTEMPT,
        {
            "schema": "dlolab-slingshot-active-bayes-runtime-attempt-v2",
            "revision": revision,
            "source_sha256": sources,
            "protocol": protocol(),
            "output_root": str(OUTPUT),
            "retry_authorized": False,
            "protected_data_read": False,
        },
    )
    output.mkdir()
    lock = write_record(
        output / "lock.json",
        {
            "schema": "dlolab-slingshot-active-bayes-runtime-lock-v2",
            "attempt_id": attempt["artifact_id"],
            "revision": revision,
            "source_sha256": sources,
            "protocol": protocol(),
            "parent_lock_id": PARENT_LOCK_ID,
            "parent_lock_sha256": PARENT_LOCK_SHA256,
            "v1_file_sha256": V1_FILE_SHA256,
            "controls_sha256": array_digest(controls),
            "output_root": str(OUTPUT),
            "retry_authorized": False,
            "protected_data_read": False,
        },
    )
    claim = write_record(
        output / "claim.json",
        {
            "schema": "dlolab-slingshot-active-bayes-runtime-claim-v2",
            "lock_id": lock["artifact_id"],
            "authorization": "prefix_only_runtime_qualification",
            "retry_authorized": False,
        },
    )
    try:
        batch_controls = np.repeat(controls[5:6], 8, axis=0)
        world = particle_worlds()[0]
        values, native = run_registered_worlds(
            Path(parent["assets_root"]) / "upstream",
            output,
            batch_controls,
            [world] * 8,
            prefix_only=True,
        )
        bundle = write_native_bundle(output, values)
        fixed = float(
            np.max(
                np.abs(
                    values["rod_pos_m"][:, :, [0, 1, 10, 11]]
                    - values["rod_pos_m"][:1, :, [0, 1, 10, 11]]
                )
            )
        )
        checks = {
            "native_steps_300": native.get("native_steps") == 300,
            "future_not_simulated": native.get("future_simulated") is False,
            "reward_not_scored": native.get("reward_scored") is False,
            "hidden_state_not_restarted": native.get("hidden_state_restart") is False,
            "world_realization_exact": _world_realization(native, world),
            "controls_exact": array_digest(values["controls"])
            == array_digest(batch_controls),
            "fixed_endpoints": fixed <= 1e-9,
        }
        seal = write_record(
            output / "seal.json",
            {
                "schema": "dlolab-slingshot-active-bayes-runtime-seal-v2",
                "lock_id": lock["artifact_id"],
                "claim_id": claim["artifact_id"],
                "native": native,
                "bundle": bundle,
                "checks": checks,
                "runtime_preflight_passed": bool(all(checks.values())),
                "study_attempt_consumed": False,
                "retry_authorized": False,
                "protected_data_read": False,
            },
        )
        if not seal["runtime_preflight_passed"]:
            raise ValueError("native runtime preflight did not qualify")
        write_record(
            output / "result.json",
            {
                "schema": "dlolab-slingshot-active-bayes-runtime-result-v2",
                "lock_id": lock["artifact_id"],
                "seal_id": seal["artifact_id"],
                "checks": checks,
                "runtime_preflight_passed": True,
                "study_attempt_consumed": False,
                "retry_authorized": False,
                "protected_data_read": False,
            },
        )
    except Exception as error:
        write_record(
            output / "failure.json",
            {
                "schema": "dlolab-slingshot-active-bayes-runtime-failure-v2",
                "lock_id": lock["artifact_id"],
                "claim_id": claim["artifact_id"],
                "error_type": type(error).__name__,
                "message": str(error),
                "study_attempt_consumed": False,
                "retry_authorized": False,
                "protected_data_read": False,
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
