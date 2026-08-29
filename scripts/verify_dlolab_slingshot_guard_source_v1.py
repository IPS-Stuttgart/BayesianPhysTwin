#!/usr/bin/env python3
"""Verify the Slingshot exact-fallback guard source result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from bayesian_phystwin_experiments.dlolab_native import array_digest, file_digest
from bayesian_phystwin_experiments.dlolab_regret_artifacts import read_record
from bayesian_phystwin_experiments.dlolab_slingshot_batch import TRACE_NAMES
from bayesian_phystwin_experiments.dlolab_slingshot_belief import (
    prefix_observations,
    sample_worlds,
)
from bayesian_phystwin_experiments.dlolab_slingshot_guard_source_v1 import (
    ACTIVE_FRACTION,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    SENSOR_DRAWS,
    WORLD_COUNT,
    infer_candidates,
    pre_outcome_checks,
    protocol,
    score,
)
from bayesian_phystwin_experiments.dlolab_slingshot_process import load_native_bundle
from bayesian_phystwin_experiments.dlolab_slingshot_task_probe_dev import (
    frontloaded_controls,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("/home/fpfaff/source-only/dlolab-slingshot-guard-source-v1")
ATTEMPT = Path("/home/fpfaff/source-only/dlolab-slingshot-guard-source-v1.attempt.json")
PARENT = Path(
    "/home/fpfaff/source-only/dlolab-benchmark-source-v1/belief-control-source-v1"
)
ACTIVE = Path("/home/fpfaff/source-only/dlolab-slingshot-active-id-particle-source-v1")
SUMMARY = ROOT / "results/sota/dlolab_slingshot_guard_source_v1/summary.json"
FROZEN_REVISION = "38c6b950e05ddbfd3524dc05d397b73c50ddb359"
PARENT_LOCK_ID = "015e6d84aa68a2a4310552ef4880752b972890f02d3e09e333ff575c92b8df25"
ACTIVE_BANK_ID = "17b96572a07a3d20818e19f3f31fec4afff98429aea8628f0872e70a3788c22a"
PARENT_CALIBRATOR_SHA256 = "26a00b934dd91b9c121242858756b7a44fa58d61163db53a3ebdebf229de6725"
RESULT_ID = "acc990372587512fd46c8dc485e5299718c9690490033c120a6efb4536259645"
SUMMARY_ID = "331af76ba0f37d602a978a7d94a1474970a3b4217e9c344762aeef0a833190e0"
Array: TypeAlias = NDArray[Any]


def _task(batch: int) -> dict[str, Any]:
    indices = list(range(8 * batch, min(8 * batch + 8, WORLD_COUNT)))
    return {
        "kind": "active_prefix_only",
        "name": f"active-prefix-{batch}",
        "batch": batch,
        "world_indices": indices,
        "native_world_indices": indices + [indices[-1]] * (8 - len(indices)),
    }


def _world_realization(native: dict[str, Any], worlds: list[dict[str, Any]]) -> bool:
    expected = {
        "bending": [[row["bending_E"] for row in worlds]],
        "stretching": [[row["stretching_K"] for row in worlds]],
        "sphere_initial_position_m": [
            [0.12 + row["x_offset_m"], 0.06, 0.2] for row in worlds
        ],
        "cube_initial_position_m": [
            [0.12 + row["x_offset_m"], 0.23, 0.22] for row in worlds
        ],
    }
    return bool(native.get("world_realization") == expected)


def _prefixes(
    output: Path,
    lock: dict[str, Any],
    controls: Array,
    worlds: list[dict[str, Any]],
) -> tuple[Array, list[str], list[bool]]:
    truth: Array = np.empty((WORLD_COUNT, 3, 4, 3), dtype=np.float64)
    ids: list[str] = []
    qa: list[bool] = []
    expected_controls = np.repeat(
        frontloaded_controls(controls, ACTIVE_FRACTION)[5:6], 8, axis=0
    )
    for batch in range(3):
        task = _task(batch)
        directory = output / task["name"]
        claim = read_record(directory / "claim.json")
        seal = read_record(directory / "seal.json")
        native = seal.get("native", {})
        task_worlds = [worlds[index] for index in task["native_world_indices"]]
        if (
            claim.get("schema") != "dlolab-slingshot-guard-claim-v1"
            or claim.get("lock_id") != lock["artifact_id"]
            or claim.get("task") != task
            or claim.get("authorization")
            != {"gate": "prefix_only_before_source_outcomes"}
            or claim.get("retry_authorized") is not False
            or seal.get("schema") != "dlolab-slingshot-guard-seal-v1"
            or seal.get("lock_id") != lock["artifact_id"]
            or seal.get("claim_id") != claim["artifact_id"]
            or seal.get("task") != task
            or native.get("native_steps") != 300
            or native.get("future_simulated") is not False
            or native.get("reward_scored") is not False
            or not _world_realization(native, task_worlds)
        ):
            raise ValueError("prefix custody changed")
        data = load_native_bundle(directory, seal["bundle"])
        fixed = float(
            np.max(
                np.abs(
                    data["rod_pos_m"][:, :, [0, 1, 10, 11]]
                    - data["rod_pos_m"][:1, :, [0, 1, 10, 11]]
                )
            )
        )
        passed = bool(
            set(data) == set(TRACE_NAMES + ("controls",))
            and all(data[name].shape[:2] == (300, 8) for name in TRACE_NAMES)
            and array_digest(data["controls"]) == array_digest(expected_controls)
            and fixed <= 1e-9
        )
        count = len(task["world_indices"])
        truth[task["world_indices"]] = prefix_observations(data)[:count]
        ids.append(seal["artifact_id"])
        qa.append(passed)
    return truth, ids, qa


def _independent_arithmetic(
    candidate_data: dict[str, Array],
    rewards: Array,
    result: dict[str, Any],
) -> dict[str, Any]:
    candidate = np.asarray(candidate_data["candidate_decisions"])
    active_bayes = np.asarray(candidate_data["active_bayes_action"])
    active_map = np.asarray(candidate_data["active_map_action"])
    blind = int(np.asarray(candidate_data["blind_action"]))
    crossfit: Array = np.empty((WORLD_COUNT, SENSOR_DRAWS), dtype=np.int64)
    for world, fold in enumerate(result["folds"]):
        if fold["held_out_world"] != world:
            raise ValueError("cross-fit fold order changed")
        crossfit[world] = candidate[
            world, :, int(fold["selected_candidate_index"])
        ]
    actions = {
        "active_map": active_map,
        "active_bayes": active_bayes,
        "cross_fitted_guard": crossfit,
    }
    blind_reward = rewards[:, blind]
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, WORLD_COUNT, size=(BOOTSTRAP_REPLICATES, WORLD_COUNT)
    )
    output: dict[str, Any] = {}
    for name, action in actions.items():
        selected = np.empty((WORLD_COUNT, SENSOR_DRAWS), dtype=np.float64)
        for world in range(WORLD_COUNT):
            for draw in range(SENSOR_DRAWS):
                selected[world, draw] = rewards[world, int(action[world, draw])]
        world_reward = selected.mean(axis=1)
        gain = world_reward - blind_reward
        boot = gain[indices].mean(axis=1)
        ci: Array = np.asarray(np.quantile(boot, [0.025, 0.975]))
        output[name] = {
            "world_reward": world_reward,
            "mean_gain": float(gain.mean()),
            "ci95": [float(ci[0]), float(ci[1])],
        }
    gain_vs_active: Array = np.asarray(
        output["cross_fitted_guard"]["world_reward"]
    ) - np.asarray(
        output["active_bayes"]["world_reward"]
    )
    boot = gain_vs_active[indices].mean(axis=1)
    ci = np.asarray(np.quantile(boot, [0.025, 0.975]))
    output["guard_vs_active_bayes"] = {
        "mean_gain": float(gain_vs_active.mean()),
        "ci95": [float(ci[0]), float(ci[1])],
    }
    return output


def verify(output: Path) -> dict[str, Any]:
    if output.resolve() != OUTPUT or output.is_symlink():
        raise ValueError("registered Slingshot guard root required")
    if (output / "failure.json").exists():
        raise ValueError("guard result and failure cannot coexist")
    attempt = read_record(ATTEMPT)
    lock = read_record(output / "lock.json")
    result = read_record(output / "result.json")
    worlds = sample_worlds("calibration")
    if (
        attempt.get("schema") != "dlolab-slingshot-guard-attempt-v1"
        or attempt.get("revision") != FROZEN_REVISION
        or attempt.get("protocol") != protocol(worlds)
        or attempt.get("output_root") != str(OUTPUT)
        or lock.get("schema") != "dlolab-slingshot-guard-lock-v1"
        or lock.get("revision") != FROZEN_REVISION
        or lock.get("attempt_id") != attempt["artifact_id"]
        or lock.get("source_sha256") != attempt.get("source_sha256")
        or lock.get("protocol") != attempt.get("protocol")
        or lock.get("output_root") != str(OUTPUT)
        or result.get("artifact_id") != RESULT_ID
        or result.get("lock_id") != lock["artifact_id"]
        or result.get("source_gate_passed") is not False
        or result.get("parent_calibration_reward_read") is not True
        or result.get("technical_failures") != 0
        or result.get("replacements") != 0
        or any(
            record.get("retry_authorized") is not False
            for record in (attempt, lock, result)
        )
        or any(
            record.get("protected_data_read") is not False
            for record in (attempt, lock, result)
        )
    ):
        raise ValueError("guard root custody changed")
    if any(
        file_digest(ROOT / name) != digest
        for name, digest in lock["source_sha256"].items()
    ):
        raise ValueError("frozen guard source changed")

    active_seal = read_record(ACTIVE / "particle-bank" / "seal.json")
    active = load_native_bundle(ACTIVE / "particle-bank", active_seal["bundle"])
    parent_lock = read_record(PARENT / "lock.json")
    controls = np.asarray(parent_lock["controls"], dtype=np.float64)
    if (
        active_seal.get("artifact_id") != ACTIVE_BANK_ID
        or active["history"].shape != (2, 27, 3, 4, 3)
        or active["reward"].shape != (27, 7)
        or parent_lock.get("artifact_id") != PARENT_LOCK_ID
    ):
        raise ValueError("registered source bank changed")
    truth, prefix_ids, prefix_qa = _prefixes(output, lock, controls, worlds)
    expected_data = infer_candidates(active["history"][1], active["reward"], truth)
    decision_seal = read_record(output / "decisions" / "seal.json")
    decision_data = load_native_bundle(output / "decisions", decision_seal["bundle"])
    if (
        decision_seal.get("artifact_id") != result.get("decision_seal_id")
        or decision_seal.get("lock_id") != lock["artifact_id"]
        or decision_seal.get("prefix_seal_ids") != prefix_ids
        or decision_seal.get("particle_bank_id") != ACTIVE_BANK_ID
        or decision_seal.get("source_outcome_read") is not False
        or set(decision_data) != set(expected_data)
        or any(
            not np.array_equal(decision_data[name], expected_data[name])
            for name in expected_data
        )
    ):
        raise ValueError("sealed guard decisions do not reconstruct")
    pre_outcome = pre_outcome_checks(decision_data, all_prefix_qa=all(prefix_qa))
    barrier = read_record(output / "decision-barrier.json")
    if (
        barrier.get("artifact_id") != result.get("barrier_id")
        or barrier.get("lock_id") != lock["artifact_id"]
        or barrier.get("decision_seal_id") != decision_seal["artifact_id"]
        or barrier.get("pre_outcome") != pre_outcome
        or barrier.get("source_outcome_read") is not False
        or pre_outcome.get("pre_outcome_gate_passed") is not True
    ):
        raise ValueError("decision barrier does not reconstruct")

    if file_digest(PARENT / "calibrator.json") != PARENT_CALIBRATOR_SHA256:
        raise ValueError("parent calibrator changed")
    parent = read_record(PARENT / "calibrator.json")
    qas = parent["native_qa"]
    rewards = np.asarray(
        [[metric["native_reward"] for metric in row["metrics"][:7]] for row in qas],
        dtype=np.float64,
    )
    generation = read_record(output / "generation" / "seal.json")
    generation_data = load_native_bundle(output / "generation", generation["bundle"])
    if (
        parent.get("lock_id") != PARENT_LOCK_ID
        or parent.get("count") != WORLD_COUNT
        or parent.get("evaluation_futures_read") is not False
        or not all(row["qa_passed"] for row in qas)
        or generation.get("artifact_id") != result.get("generation_id")
        or generation.get("lock_id") != lock["artifact_id"]
        or generation.get("barrier_id") != barrier["artifact_id"]
        or generation.get("parent_calibrator_id") != parent["artifact_id"]
        or generation.get("parent_future_seal_ids") != parent["future_seals"]
        or generation.get("ordinary_worlds") != WORLD_COUNT
        or generation.get("technical_failures") != 0
        or generation.get("replacements") != 0
        or set(generation_data) != {"reward"}
        or not np.array_equal(generation_data["reward"], rewards)
    ):
        raise ValueError("source generation does not reconstruct")

    expected_score = score(decision_data, rewards, all_native_qa=True)
    linked = {
        key: value
        for key, value in result.items()
        if key
        not in {
            "artifact_id",
            "lock_id",
            "decision_seal_id",
            "barrier_id",
            "generation_id",
            "parent_calibrator_id",
            "pre_outcome",
            "parent_calibration_reward_read",
            "retry_authorized",
        }
    }
    if linked != expected_score or result.get("pre_outcome") != pre_outcome:
        raise ValueError("registered guard score does not reconstruct")

    arithmetic = _independent_arithmetic(decision_data, rewards, result)
    for name in ("active_map", "active_bayes", "cross_fitted_guard"):
        if not np.isclose(
            result["arms"][name]["mean_gain_over_blind"],
            arithmetic[name]["mean_gain"],
            rtol=0,
            atol=1e-15,
        ) or not np.allclose(
            result["arms"][name]["gain_ci95"],
            arithmetic[name]["ci95"],
            rtol=0,
            atol=1e-15,
        ):
            raise ValueError("independent guard arithmetic changed")
    if not np.isclose(
        result["cross_fitted_gain_over_active_bayes"]["mean_gain"],
        arithmetic["guard_vs_active_bayes"]["mean_gain"],
        rtol=0,
        atol=1e-15,
    ) or not np.allclose(
        result["cross_fitted_gain_over_active_bayes"]["ci95"],
        arithmetic["guard_vs_active_bayes"]["ci95"],
        rtol=0,
        atol=1e-15,
    ):
        raise ValueError("independent paired arithmetic changed")

    updated = [
        row
        for row in result["full_fit_selection"]["candidate_stats"]
        if row["updated_draws"] > 0
    ]
    if not updated or any(row["mean_gain"] >= 0 for row in updated):
        raise ValueError("nontrivial-candidate conclusion changed")
    summary = read_record(SUMMARY)
    if (
        summary.get("artifact_id") != SUMMARY_ID
        or summary.get("result_id") != result["artifact_id"]
        or summary.get("source_revision") != FROZEN_REVISION
        or summary.get("source_gate_passed") is not False
        or summary.get("cross_fitted_exact_fallback_folds") != WORLD_COUNT
        or summary.get("full_fit_exact_fallback") is not True
        or summary.get("all_nontrivial_candidates_negative") is not True
        or summary.get("cross_fitted_guard_gain_over_active_bayes")
        != result["cross_fitted_gain_over_active_bayes"]["mean_gain"]
    ):
        raise ValueError("compact guard result changed")
    return {
        "schema": "dlolab-slingshot-guard-verification-v1",
        "attempt_id": attempt["artifact_id"],
        "lock_id": lock["artifact_id"],
        "decision_seal_id": decision_seal["artifact_id"],
        "barrier_id": barrier["artifact_id"],
        "generation_id": generation["artifact_id"],
        "result_id": result["artifact_id"],
        "summary_id": summary["artifact_id"],
        "prefix_batches": 3,
        "ordinary_worlds": WORLD_COUNT,
        "technical_failures": 0,
        "source_gate_passed": False,
        "exact_fallback_folds": WORLD_COUNT,
        "all_nontrivial_candidates_negative": True,
        "independent_arithmetic_passed": True,
        "independent_human_review": False,
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
