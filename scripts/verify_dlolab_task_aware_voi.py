#!/usr/bin/env python3
"""Verify task-aware VOI artifacts and arithmetic without native replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from bayesian_phystwin_experiments.dlolab_regret_artifacts import (
    load_bundle,
    read_record,
)
from bayesian_phystwin_experiments.dlolab_task_aware_voi import (
    GOAL_TIP_Z_M,
    PROBE_NAMES,
    TRUTH_COUNT,
    noisy_probe_observations,
    particle_count,
    probe_features,
    realized_task_losses,
    score_source,
    seal_decisions,
    selector_analysis,
    task_headroom,
    task_losses,
)

OUTPUT = Path("/home/fpfaff/source-only/dlolab-task-aware-voi-source-v1")
ATTEMPT_LEDGER = Path(
    "/home/fpfaff/source-only/dlolab-task-aware-voi-source-v1.attempt.json"
)
Array: TypeAlias = NDArray[Any]


def _stage(
    output: Path,
    lock: dict[str, Any],
    name: str,
    expected_count: int,
) -> tuple[dict[str, Any], dict[str, Array]]:
    seal = read_record(output / name / "seal.json")
    if (
        seal.get("schema") != "dlolab-task-aware-stage-seal-v1"
        or seal.get("stage") != name
        or seal.get("lock_id") != lock["artifact_id"]
        or seal.get("status") != "ordinary_success"
        or seal.get("count") != expected_count
        or seal.get("protected_data_read") is not False
        or seal.get("retry_authorized") is not False
    ):
        raise ValueError(f"invalid task-aware stage: {name}")
    return seal, load_bundle(output / name, seal["bundle"])


def _arrays_equal(expected: dict[str, Array], actual: dict[str, Array]) -> bool:
    return set(expected) == set(actual) and all(
        np.array_equal(expected[name], actual[name]) for name in expected
    )


def verify(output: Path) -> dict[str, Any]:
    if output.resolve() != OUTPUT or output.is_symlink():
        raise ValueError("registered task-aware source root required")
    lock = read_record(output / "lock.json")
    attempt = read_record(ATTEMPT_LEDGER)
    result = read_record(output / "result.json")
    if (
        lock.get("schema") != "dlolab-task-aware-voi-lock-v1"
        or result.get("schema") != "dlolab-task-aware-voi-result-v1"
        or attempt.get("schema") != "dlolab-task-aware-attempt-ledger-v1"
        or lock.get("attempt_id") != attempt.get("artifact_id")
        or attempt.get("source_revision") != lock.get("source_revision")
        or attempt.get("source_sha256") != lock.get("source_sha256")
        or attempt.get("retry_authorized") is not False
        or result.get("lock_id") != lock["artifact_id"]
        or result.get("retry_authorized") is not False
        or result.get("protected_data_read") is not False
    ):
        raise ValueError("invalid task-aware lock or result")
    bank_seal, bank = _stage(output, lock, "particle-bank", particle_count())
    if bank_seal["native"]["all_branches_qualified"] is not True:
        raise ValueError("particle bank did not pass native qualification")
    initial = bank["initial_position_m"]
    particle_probe = np.stack(
        [
            probe_features(bank["probe_trajectory_m"][index], initial)
            for index in range(len(PROBE_NAMES))
        ]
    )
    particle_loss = task_losses(
        bank["action_trajectory_m"].transpose(1, 0, 2, 3, 4),
        GOAL_TIP_Z_M,
    )
    selectors = selector_analysis(particle_probe, particle_loss)
    headroom = task_headroom(particle_loss)
    analysis = read_record(output / "particle-analysis.json")
    selection = read_record(output / "probe-selection.json")
    if (
        analysis.get("lock_id") != lock["artifact_id"]
        or analysis.get("particle_bank_id") != bank_seal["artifact_id"]
        or analysis.get("selector_analysis") != selectors
        or analysis.get("task_headroom") != headroom
        or selection.get("particle_analysis_id") != analysis["artifact_id"]
        or selection.get("metrics") != selectors
    ):
        raise ValueError("task-aware particle analysis changed")
    if not selectors["passed"] or not headroom["passed"]:
        if result.get("source_gate_passed") is not False:
            raise ValueError("failed source qualification mislabeled")
        return {
            "schema": "dlolab-task-aware-voi-verification-v1",
            "lock_id": lock["artifact_id"],
            "attempt_id": attempt["artifact_id"],
            "result_id": result["artifact_id"],
            "particle_bank_id": bank_seal["artifact_id"],
            "selector_gate_passed": selectors["passed"],
            "task_headroom_gate_passed": headroom["passed"],
            "truth_stage_verified": False,
            "native_replay_performed": False,
            "protected_data_read": False,
            "passed": True,
        }

    mi_probe = int(selectors["generic_mi_probe_index"])
    task_probe = int(selectors["task_aware_probe_index"])
    probe_seal, truth_probe = _stage(output, lock, "truth-probes", TRUTH_COUNT)
    future_seal, truth_future = _stage(output, lock, "truth-futures", TRUTH_COUNT)
    if (
        probe_seal["native"]["all_branches_qualified"] is not True
        or future_seal["native"]["all_branches_qualified"] is not True
        or any(
            not np.array_equal(truth_probe[name], truth_future[name])
            for name in ("bending", "twisting", "goal_index", "goal_tip_z_m", "initial_position_m")
        )
    ):
        raise ValueError("truth probe/future worlds changed")
    truth_features = np.stack(
        [
            probe_features(trajectory, truth_probe["initial_position_m"])
            for trajectory in truth_probe["probe_trajectory_m"]
        ]
    )
    nuisance = noisy_probe_observations(truth_features)
    expected_decisions = seal_decisions(
        nuisance["observation"],
        truth_probe["probe_indices"],
        particle_probe,
        particle_loss,
        truth_probe["goal_index"],
        task_probe,
        mi_probe,
    )
    decision_seal = read_record(output / "decisions" / "seal.json")
    actual_decisions = load_bundle(output / "decisions", decision_seal["bundle"])
    expected_bundle = {
        **expected_decisions,
        "probe_indices": truth_probe["probe_indices"],
        "probe_observation_m": nuisance["observation"],
        "shared_bias_m": nuisance["shared_bias"],
        "independent_noise_m": nuisance["independent_noise"],
    }
    if (
        decision_seal.get("lock_id") != lock["artifact_id"]
        or decision_seal.get("truth_probe_id") != probe_seal["artifact_id"]
        or decision_seal.get("task_futures_read") is not False
        or not _arrays_equal(expected_bundle, actual_decisions)
    ):
        raise ValueError("pre-outcome task-aware decisions changed")
    truth_loss = realized_task_losses(
        truth_future["action_trajectory_m"].transpose(1, 0, 2, 3, 4),
        truth_future["goal_tip_z_m"],
    )
    metrics = score_source(expected_decisions, truth_loss)
    score_seal = read_record(output / "score" / "seal.json")
    score_bundle = load_bundle(output / "score", score_seal["bundle"])
    if (
        score_seal.get("lock_id") != lock["artifact_id"]
        or score_seal.get("decision_id") != decision_seal["artifact_id"]
        or score_seal.get("truth_future_id") != future_seal["artifact_id"]
        or score_seal.get("metrics") != metrics
        or not np.array_equal(score_bundle.get("truth_loss_m2"), truth_loss)
        or result.get("score_id") != score_seal["artifact_id"]
        or result.get("source_gate_passed") is not metrics["source_gate_passed"]
    ):
        raise ValueError("task-aware score changed")
    return {
        "schema": "dlolab-task-aware-voi-verification-v1",
        "lock_id": lock["artifact_id"],
        "attempt_id": attempt["artifact_id"],
        "result_id": result["artifact_id"],
        "particle_bank_id": bank_seal["artifact_id"],
        "truth_probe_id": probe_seal["artifact_id"],
        "decision_id": decision_seal["artifact_id"],
        "truth_future_id": future_seal["artifact_id"],
        "score_id": score_seal["artifact_id"],
        "selector_gate_passed": True,
        "task_headroom_gate_passed": True,
        "source_value_gate_passed": metrics["source_gate_passed"],
        "truth_stage_verified": True,
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
