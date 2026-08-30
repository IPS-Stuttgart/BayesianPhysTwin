#!/usr/bin/env python3
"""Verify the terminal independent-action Slingshot v3 source failure."""

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
    independent_world_qa,
)

Array: TypeAlias = NDArray[Any]
ROOT = Path(__file__).resolve().parents[2]
SUMMARY = (
    ROOT / "results/source/dlolab_slingshot_policy_certificate_source_v3/summary.json"
)
ATTEMPT = Path(
    "/home/fpfaff/source-only/dlolab-slingshot-policy-certificate-source-v3.attempt.json"
)
RUNNER_PATH = (
    ROOT / "scripts/remote/run_dlolab_slingshot_policy_certificate_source_v3.py"
)
SPEC = importlib.util.spec_from_file_location("slingshot_v3_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _tree_identity(root: Path) -> dict[str, Any]:
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    byte_count = 0
    for path in paths:
        if path.is_symlink():
            raise ValueError("raw v3 tree must not contain symlinks")
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
        raise ValueError("complete frozen v3 source identity required")
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
        raise ValueError("frozen v3 source blob changed")


def _localization(a: dict[str, Array], b: dict[str, Array]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for label, name in (
        ("sphere", "sphere_pos_m"),
        ("cube", "cube_pos_m"),
        ("rod", "rod_pos_m"),
        ("gripper", "gripper_pos_m"),
    ):
        difference = np.abs(np.asarray(a[name]) - np.asarray(b[name]))
        per_frame = np.max(difference.reshape(difference.shape[0], -1), axis=1)
        output[f"{label}_max_error_m"] = float(per_frame.max())
        if label in {"sphere", "cube"}:
            output[f"{label}_max_error_frame"] = int(np.argmax(per_frame))
            crossings = np.flatnonzero(per_frame > 0.0005)
            output[f"{label}_first_above_0_5mm_frame"] = int(crossings[0])
    output["interpretation"] = (
        "rare late rigid-contact bifurcation after an effectively identical prefix "
        "and command"
    )
    return output


def verify(raw_root: Path) -> dict[str, Any]:
    summary = dict(load_strict_json_object(SUMMARY, label="Slingshot v3 summary"))
    identity = summary.pop("artifact_id", None)
    if identity != content_id(summary):
        raise ValueError("compact v3 result identity changed")
    summary["artifact_id"] = identity
    if raw_root.resolve() != Path(summary["raw_tree"]["root"]):
        raise ValueError("only the registered raw v3 root is admitted")
    if _tree_identity(raw_root) != {
        key: summary["raw_tree"][key]
        for key in ("file_count", "file_byte_count", "canonical_tree_sha256")
    }:
        raise ValueError("raw v3 tree identity changed")
    key_paths = {
        "lock.json": raw_root / "lock.json",
        "calibration-candidates/seal.json": (
            raw_root / "calibration-candidates/seal.json"
        ),
        "failure.json": raw_root / "failure.json",
        "attempt.json": ATTEMPT,
    }
    if any(
        file_digest(path) != summary["key_file_sha256"][name]
        for name, path in key_paths.items()
    ):
        raise ValueError("key v3 artifact changed")

    lock = read_record(raw_root / "lock.json")
    attempt = read_record(ATTEMPT)
    failure = read_record(raw_root / "failure.json")
    candidate_seal, _ = runner.load_candidates(raw_root, lock, "calibration")
    _verify_source(lock)
    if (
        lock.get("artifact_id") != summary["identities"]["lock_id"]
        or lock.get("source_revision") != summary["source_revision"]
        or attempt.get("artifact_id") != summary["identities"]["attempt_id"]
        or attempt.get("lock_id") != lock["artifact_id"]
        or attempt.get("attempt_number") != 1
        or attempt.get("retry_authorized") is not False
        or candidate_seal.get("artifact_id")
        != summary["identities"]["calibration_candidate_seal_id"]
        or candidate_seal.get("future_simulated") is not False
        or candidate_seal.get("future_read") is not False
        or failure.get("artifact_id") != summary["identities"]["run_failure_id"]
        or failure.get("terminal_stage") != "calibration-world-qualification"
        or failure.get("retry_authorized") is not False
        or failure.get("replacement_authorized") is not False
        or failure.get("partial_score_authorized") is not False
    ):
        raise ValueError("terminal v3 record contract changed")

    authorization = {
        "gate": "reproduced_calibration_candidates",
        "candidate_seal_id": candidate_seal["artifact_id"],
    }
    controls = np.asarray(lock["controls"], dtype=np.float64)
    failures: list[int] = []
    common: list[float] = []
    position: list[float] = []
    reward: list[float] = []
    failed_rows: list[dict[str, Array]] = []
    for index, world in enumerate(runner.continuous_worlds("calibration")):
        rows: list[dict[str, Array]] = []
        reports: list[dict[str, Any]] = []
        seal_ids: list[str] = []
        for action in range(runner.ACTION_COUNT):
            seal, arrays = runner.load_future_action(
                raw_root,
                lock,
                "calibration",
                index,
                action,
                authorization=authorization,
            )
            rows.append(arrays)
            reports.append(seal["native"])
            seal_ids.append(seal["artifact_id"])
        qa = independent_world_qa(
            rows,
            reports,
            controls,
            world,
            world_count=runner.COUNTS["calibration"],
        )
        common.append(float(qa["common_prefix_error_m"]))
        position.append(float(qa["duplicate_error_m"]))
        reward.append(
            abs(
                float(qa["metrics"][5]["native_reward"])
                - float(qa["metrics"][7]["native_reward"])
            )
        )
        if qa["qa_passed"]:
            if index < 99:
                stored = read_record(
                    raw_root / f"calibration-future-{index:03d}-qualification.json"
                )
                if (
                    stored.get("role") != "calibration"
                    or stored.get("world") != world
                    or stored.get("action_seal_ids") != seal_ids
                    or stored.get("qa") != qa
                ):
                    raise ValueError("published v3 world qualification changed")
        else:
            failures.append(index)
            failed_rows = rows

    position_array = np.asarray(position, dtype=np.float64)
    reward_array = np.asarray(reward, dtype=np.float64)
    diagnostic = summary["post_terminal_read_only_diagnostic"]
    registered = summary["registered_failure"]
    if (
        failures != diagnostic["failed_world_indices"]
        or failures != [99]
        or len(list(raw_root.glob("calibration-prefix-*/seal.json"))) != 16
        or len(list(raw_root.glob("calibration-future-*-action-*/seal.json")))
        != 1024
        or list(raw_root.glob("calibration-future-*-action-*/failure.json"))
        or len(list(raw_root.glob("calibration-future-*-qualification.json"))) != 99
        or (raw_root / "calibration").exists()
        or list(raw_root.glob("evaluation-prefix-*"))
        or (raw_root / "result.json").exists()
        or max(common) != diagnostic["maximum_common_prefix_error_m"]
        or float(position_array.max())
        != diagnostic["maximum_duplicate_position_error_m"]
        or float(reward_array.max()) != diagnostic["maximum_duplicate_reward_error"]
        or np.quantile(position_array, [0.5, 0.9, 0.95, 0.99]).tolist()
        != diagnostic["position_error_quantiles_50_90_95_99_m"]
        or np.quantile(reward_array, [0.5, 0.9, 0.95, 0.99]).tolist()
        != diagnostic["reward_error_quantiles_50_90_95_99"]
        or registered["world_index"] != 99
        or registered["duplicate_position_error_m"] != position[99]
        or registered["duplicate_reward_error"] != reward[99]
        or len(failed_rows) != runner.ACTION_COUNT
        or _localization(failed_rows[5], failed_rows[7])
        != diagnostic["failure_localization"]
    ):
        raise ValueError("terminal v3 denominator diagnostic changed")
    return cast(dict[str, Any], summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.raw_root)
    print(f"verified terminal Slingshot v3 {result['artifact_id']}", flush=True)
