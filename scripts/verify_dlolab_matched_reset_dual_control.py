#!/usr/bin/env python3
"""Verify staged custody and arithmetic for the matched-reset source study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from run_dlolab_matched_reset_dual_control import (
    OUTPUT,
    _decision,
    _selection,
    _stage,
    _validate_lock,
)

from bayesian_phystwin_experiments.dlolab_matched_reset_dual_control import (
    FIXED_CONTROL_PROBE_INDEX,
    GOALS_Y_M,
    PROBE_NAMES,
    noisy_probe_observations,
    probe_features,
    probe_information,
    realized_task_losses,
    score_source,
    seal_decisions,
    task_headroom,
    task_losses,
)
from bayesian_phystwin_experiments.dlolab_regret_artifacts import read_record


def verify(output: Path) -> dict[str, object]:
    lock = _validate_lock(output)
    bank_seal, bank = _stage(output, lock, "particle-bank")
    initial = bank["initial_position_m"]
    particle_probe = np.stack(
        [
            probe_features(bank["probe_trajectory_m"][index], initial)
            for index in range(len(PROBE_NAMES))
        ]
    )
    particle_loss = task_losses(
        bank["action_trajectory_m"].transpose(1, 0, 2, 3, 4), GOALS_Y_M
    )
    information = probe_information(particle_probe)
    headroom = task_headroom(particle_loss)
    analysis = read_record(output / "particle-analysis.json")
    selection = _selection(output, lock)
    if (
        analysis["particle_bank_id"] != bank_seal["artifact_id"]
        or analysis["probe_information"] != information
        or analysis["task_headroom"] != headroom
        or selection["metrics"] != information
    ):
        raise ValueError("particle analysis or reward-blind selection changed")
    result = read_record(output / "result.json")
    summary: dict[str, object] = {
        "schema": "dlolab-matched-reset-source-verification-v1",
        "lock_id": lock["artifact_id"],
        "result_id": result["artifact_id"],
        "status": result["status"],
        "particle_analysis_reproduced": True,
        "probe_information_passed": information["passed"],
        "task_headroom_passed": headroom["passed"],
        "protected_data_read": False,
    }
    if not information["passed"] or not headroom["passed"]:
        if (output / "truth-probes").exists() or result["task_futures_generated"]:
            raise ValueError("downstream stage exists after failed source gate")
        return summary
    probe_seal, truth_probe = _stage(output, lock, "truth-probes")
    decision_seal, stored_decisions = _decision(output, lock)
    null_feature = probe_features(
        truth_probe["probe_trajectory_m"][0], truth_probe["initial_position_m"]
    )
    active_feature = probe_features(
        truth_probe["probe_trajectory_m"][2], truth_probe["initial_position_m"]
    )
    fixed_feature = probe_features(
        truth_probe["probe_trajectory_m"][1], truth_probe["initial_position_m"]
    )
    null_noise = noisy_probe_observations(null_feature)
    active_noise = noisy_probe_observations(active_feature)
    fixed_noise = noisy_probe_observations(fixed_feature)
    decisions = seal_decisions(
        active_noise["observation"],
        null_noise["observation"],
        fixed_noise["observation"],
        particle_probe[selection["selected_probe_index"]],
        particle_probe[0],
        particle_probe[FIXED_CONTROL_PROBE_INDEX],
        particle_loss,
        truth_probe["goal_index"],
    )
    for name, value in decisions.items():
        if not np.array_equal(stored_decisions[name], value):
            raise ValueError("sealed decision arithmetic changed")
    future_seal, truth_future = _stage(output, lock, "truth-futures")
    if (
        not np.array_equal(truth_probe["initial_position_m"], truth_future["initial_position_m"])
        or probe_seal["native"]["initial_state_sha256"]
        != future_seal["native"]["initial_state_sha256"]
    ):
        raise ValueError("probe and task initial native state changed")
    truth_loss = realized_task_losses(
        truth_future["action_trajectory_m"].transpose(1, 0, 2, 3, 4),
        truth_future["goal_y_m"],
    )
    metrics = score_source(decisions, truth_loss)
    score = read_record(output / "score" / "seal.json")
    if (
        score["decision_id"] != decision_seal["artifact_id"]
        or score["truth_future_id"] != future_seal["artifact_id"]
        or score["metrics"] != metrics
        or result["metrics"] != metrics
    ):
        raise ValueError("source-value arithmetic changed")
    summary.update(
        decisions_reproduced=True,
        exact_initial_state_matched=True,
        source_value_reproduced=True,
        source_gate_passed=metrics["source_gate_passed"],
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(verify(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
