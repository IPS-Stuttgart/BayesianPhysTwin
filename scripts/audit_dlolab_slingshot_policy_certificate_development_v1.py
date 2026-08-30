#!/usr/bin/env python3
"""Build the opened-world Slingshot policy-certificate capacity diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin_experiments.dlolab_slingshot_batch import split_batch
from bayesian_phystwin_experiments.dlolab_slingshot_belief import BASELINE
from bayesian_phystwin_experiments.dlolab_slingshot_cmaes import task_metrics
from bayesian_phystwin_experiments.dlolab_slingshot_policy_certificate_v1 import (
    leave_one_out_capacity_diagnostic,
)

PARENT_LOCK_ID = "015e6d84aa68a2a4310552ef4880752b972890f02d3e09e333ff575c92b8df25"
PARENT_RESULT_ID = "9b8ff0817744392e0584c9b59936dd1b0e9331d3b0fa2d021f5a361947d32ee9"
ROLES = (("evaluation", 32), ("calibration", 19))
ROOT = Path(__file__).resolve().parents[1]
BOUND_SOURCES = (
    "src/bayesian_phystwin/policy_gain_certificate.py",
    "src/bayesian_phystwin_experiments/dlolab_slingshot_policy_certificate_v1.py",
    "scripts/audit_dlolab_slingshot_policy_certificate_development_v1.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _canonical_id(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "artifact_id"}
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _load_parent(root: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    lock_path = root / "lock.json"
    result_path = root / "result.json"
    lock = _load_json(lock_path)
    result = _load_json(result_path)
    if lock.get("artifact_id") != PARENT_LOCK_ID:
        raise ValueError("registered parent lock is required")
    if result.get("artifact_id") != PARENT_RESULT_ID:
        raise ValueError("registered parent result is required")
    protocol = lock.get("protocol")
    if (
        lock.get("protected_data_read") is not False
        or not isinstance(protocol, dict)
        or protocol.get("new_recordings") is not False
    ):
        raise ValueError("public-simulator parent boundary changed")

    case_ids: list[str] = []
    observations: list[np.ndarray] = []
    expected_losses: list[np.ndarray] = []
    action_gains: list[np.ndarray] = []
    for role, count in ROLES:
        for index in range(count):
            case_id = f"{role}-{index:02d}"
            prediction_path = root / f"{role}-predictions" / f"case-{index:02d}" / "arrays.npz"
            future_path = root / f"{role}-future-{index:02d}" / "arrays.npz"
            if not prediction_path.is_file() or not future_path.is_file():
                raise ValueError(f"missing opened parent record {case_id}")
            with np.load(prediction_path, allow_pickle=False) as archive:
                if "observation" not in archive.files or "expected_losses" not in archive.files:
                    raise ValueError(f"incomplete opened prediction {case_id}")
                observation = np.array(archive["observation"], dtype=np.float64, copy=True)
                losses = np.array(archive["expected_losses"], dtype=np.float64, copy=True)
            with np.load(future_path, allow_pickle=False) as archive:
                future = {name: np.array(archive[name], copy=True) for name in archive.files}
            rewards = np.asarray(
                [
                    task_metrics(row)["native_reward"]
                    for row in split_batch(future, 8)[:7]
                ],
                dtype=np.float64,
            )
            if observation.shape != (3, 4, 3) or losses.shape != (7,) or rewards.shape != (7,):
                raise ValueError(f"opened parent layout changed for {case_id}")
            case_ids.append(case_id)
            observations.append(observation)
            expected_losses.append(losses)
            action_gains.append(rewards - rewards[BASELINE])
    return (
        case_ids,
        np.stack(observations),
        np.stack(expected_losses),
        np.stack(action_gains),
    )


def build(source_root: Path) -> dict[str, Any]:
    case_ids, observations, expected_losses, action_gains = _load_parent(source_root)
    result = leave_one_out_capacity_diagnostic(
        case_ids=tuple(case_ids),
        observations=observations,
        expected_losses=expected_losses,
        action_gains=action_gains,
    )
    result.update(
        {
            "parent_lock_id": PARENT_LOCK_ID,
            "parent_result_id": PARENT_RESULT_ID,
            "parent_lock_file_sha256": _sha256(source_root / "lock.json"),
            "parent_result_file_sha256": _sha256(source_root / "result.json"),
            "source_root": str(source_root),
            "source_roles": [role for role, _ in ROLES],
            "source_sha256": {
                name: _sha256(ROOT / name) for name in BOUND_SOURCES
            },
            "source_outcomes_previously_opened": True,
            "source_gate_reclassified": False,
            "new_simulation_executed": False,
            "new_recordings": False,
            "protected_data_read": False,
            "held_v8_read": False,
            "dlo4_dlo5_read": False,
            "official_benchmark_or_sota_claim": False,
        }
    )
    result["artifact_id"] = _canonical_id(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.source_root.resolve())
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: result[key] for key in (
        "artifact_id",
        "accepted_count",
        "mean_guarded_gain",
        "harmful_guarded_count",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
