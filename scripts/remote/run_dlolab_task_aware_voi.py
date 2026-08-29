#!/usr/bin/env python3
"""Run one frozen matched-reset task-aware value-of-information source study."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin_experiments.deform_state_restart import file_digest
from bayesian_phystwin_experiments.dlolab_native import verify_upstream
from bayesian_phystwin_experiments.dlolab_regret_artifacts import (
    clean_revision,
    load_bundle,
    read_record,
    runtime_identity,
    write_bundle,
    write_record,
)
from bayesian_phystwin_experiments.dlolab_task_aware_native import (
    generate_particle_bank,
    generate_truth_futures,
    generate_truth_probes,
)
from bayesian_phystwin_experiments.dlolab_task_aware_voi import (
    GOAL_TIP_Z_M,
    PROBE_NAMES,
    TRUTH_COUNT,
    noisy_probe_observations,
    particle_count,
    probe_features,
    protocol,
    realized_task_losses,
    score_source,
    seal_decisions,
    selector_analysis,
    task_headroom,
    task_losses,
)

ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path("/home/fpfaff/source-only/dlolab-benchmark-source-v1-assets")
UPSTREAM = ASSETS / "upstream"
OUTPUT = Path("/home/fpfaff/source-only/dlolab-task-aware-voi-source-v1")
ATTEMPT_LEDGER = Path(
    "/home/fpfaff/source-only/dlolab-task-aware-voi-source-v1.attempt.json"
)
SOURCES = (
    "src/bayesian_phystwin_experiments/dlolab_task_aware_voi.py",
    "src/bayesian_phystwin_experiments/dlolab_task_aware_native.py",
    "scripts/remote/run_dlolab_task_aware_voi.py",
    "scripts/verify_dlolab_task_aware_voi.py",
    "tests/test_dlolab_task_aware_voi.py",
    "tests/test_dlolab_task_aware_runner.py",
    "docs/dlolab_task_aware_voi_source_v1.md",
    "configs/sota/dlolab_task_aware_voi_source_v1.json",
    "results/sota/dlolab_task_aware_voi_prelock_v1/null_probe_native_smoke.json",
    "src/bayesian_phystwin_experiments/dlolab_native.py",
    "src/bayesian_phystwin_experiments/dlolab_regret_artifacts.py",
    "src/bayesian_phystwin_experiments/coupled_action_regret.py",
    "src/bayesian_phystwin_experiments/deform_state_restart.py",
    "src/bayesian_phystwin/_portable_contracts.py",
    "src/bayesian_phystwin/_canonical_contracts.py",
)


def _source_hashes() -> dict[str, str]:
    if any(not (ROOT / name).is_file() for name in SOURCES):
        raise ValueError("complete task-aware registered source set required")
    return {name: file_digest(ROOT / name) for name in SOURCES}


def _validate_lock(output: Path) -> dict[str, Any]:
    if output.resolve() != OUTPUT or output.is_symlink():
        raise ValueError("only the registered task-aware output root is permitted")
    lock = read_record(output / "lock.json")
    attempt = read_record(ATTEMPT_LEDGER)
    if (
        lock.get("schema") != "dlolab-task-aware-voi-lock-v1"
        or lock.get("source_revision") != clean_revision(ROOT)
        or lock.get("source_sha256") != _source_hashes()
        or lock.get("protocol") != protocol()
        or lock.get("runtime") != runtime_identity()
        or lock.get("upstream") != verify_upstream(UPSTREAM)
        or lock.get("output_root") != str(OUTPUT)
        or lock.get("attempt_id") != attempt.get("artifact_id")
        or attempt.get("schema") != "dlolab-task-aware-attempt-ledger-v1"
        or attempt.get("source_revision") != lock.get("source_revision")
        or attempt.get("source_sha256") != lock.get("source_sha256")
        or attempt.get("protocol") != lock.get("protocol")
        or attempt.get("output_root") != str(OUTPUT)
        or attempt.get("retry_authorized") is not False
        or lock.get("retry_authorized") is not False
        or lock.get("protected_data_read") is not False
    ):
        raise ValueError("clean frozen task-aware lock required")
    return lock


def _stage(
    output: Path, lock: dict[str, Any], name: str
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    seal = read_record(output / name / "seal.json")
    expected = {
        "particle-bank": particle_count(),
        "truth-probes": TRUTH_COUNT,
        "truth-futures": TRUTH_COUNT,
    }
    if (
        seal.get("schema") != "dlolab-task-aware-stage-seal-v1"
        or seal.get("stage") != name
        or seal.get("lock_id") != lock["artifact_id"]
        or seal.get("status") != "ordinary_success"
        or seal.get("count") != expected[name]
        or seal.get("protected_data_read") is not False
        or seal.get("retry_authorized") is not False
    ):
        raise ValueError("ordinary complete task-aware stage required")
    if name != "truth-futures" and seal.get("task_futures_generated") is not False:
        raise ValueError("task futures crossed the task-aware decision boundary")
    if name == "truth-futures" and seal.get("task_futures_generated") is not True:
        raise ValueError("task-aware truth-future seal mislabeled")
    return seal, load_bundle(output / name, seal["bundle"])


def _selection(output: Path, lock: dict[str, Any]) -> dict[str, Any]:
    value = read_record(output / "probe-selection.json")
    if (
        value.get("schema") != "dlolab-task-aware-probe-selection-v1"
        or value.get("lock_id") != lock["artifact_id"]
        or value.get("metrics", {}).get("passed") is not True
        or value.get("truth_futures_read") is not False
        or type(value.get("task_aware_probe_index")) is not int
        or type(value.get("generic_mi_probe_index")) is not int
        or value["task_aware_probe_index"] == 0
        or value["generic_mi_probe_index"] == 0
        or value["task_aware_probe_index"] == value["generic_mi_probe_index"]
    ):
        raise ValueError("passing distinct task-aware and MI probe selection required")
    return value


def _decision(
    output: Path, lock: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    seal = read_record(output / "decisions" / "seal.json")
    if (
        seal.get("schema") != "dlolab-task-aware-decision-seal-v1"
        or seal.get("lock_id") != lock["artifact_id"]
        or seal.get("count") != TRUTH_COUNT
        or seal.get("task_futures_read") is not False
        or seal.get("protected_data_read") is not False
    ):
        raise ValueError("complete pre-outcome task-aware decision seal required")
    return seal, load_bundle(output / "decisions", seal["bundle"])


def _worker(
    output: Path,
    stage: str,
    generic_mi_probe: int | None,
    task_aware_probe: int | None,
) -> None:
    lock = _validate_lock(output)
    if (output / "result.json").exists() or (output / "failure.json").exists():
        raise ValueError("terminal task-aware source study cannot be retried")
    if stage == "particle-bank":
        if generic_mi_probe is not None or task_aware_probe is not None or (output / stage).exists():
            raise ValueError("fresh task-aware particle-bank stage required")
        arrays, native = generate_particle_bank(UPSTREAM)
        task_futures = False
    elif stage == "truth-probes":
        selection = _selection(output, lock)
        if (
            generic_mi_probe != selection["generic_mi_probe_index"]
            or task_aware_probe != selection["task_aware_probe_index"]
            or (output / stage).exists()
        ):
            raise ValueError("registered task-aware selector probes required")
        _stage(output, lock, "particle-bank")
        arrays, native = generate_truth_probes(
            UPSTREAM,
            generic_mi_probe,
            task_aware_probe,
        )
        task_futures = False
    elif stage == "truth-futures":
        selection = _selection(output, lock)
        if (
            generic_mi_probe != selection["generic_mi_probe_index"]
            or task_aware_probe != selection["task_aware_probe_index"]
            or (output / stage).exists()
        ):
            raise ValueError("registered task-aware selector probes required")
        _stage(output, lock, "truth-probes")
        _decision(output, lock)
        arrays, native = generate_truth_futures(UPSTREAM)
        task_futures = True
    else:
        raise ValueError("unknown task-aware worker stage")
    directory = output / stage
    directory.mkdir(exist_ok=False)
    write_record(
        directory / "seal.json",
        {
            "schema": "dlolab-task-aware-stage-seal-v1",
            "stage": stage,
            "lock_id": lock["artifact_id"],
            "status": "ordinary_success",
            "count": len(arrays["bending"]),
            "bundle": write_bundle(directory, arrays),
            "native": native,
            "task_futures_generated": task_futures,
            "protected_data_read": False,
            "retry_authorized": False,
        },
    )


def _execute_worker(
    output: Path,
    stage: str,
    generic_mi_probe: int | None = None,
    task_aware_probe: int | None = None,
) -> None:
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", stage]
    if generic_mi_probe is not None:
        command += ["--generic-mi-probe", str(generic_mi_probe)]
    if task_aware_probe is not None:
        command += ["--task-aware-probe", str(task_aware_probe)]
    with (output / f"{stage}.log").open("xb") as log:
        subprocess.run(
            command,
            cwd=ROOT,
            env=os.environ.copy(),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _terminal(
    output: Path, lock: dict[str, Any], status: str, **extra: Any
) -> dict[str, Any]:
    value = write_record(
        output / "result.json",
        {
            "schema": "dlolab-task-aware-voi-result-v1",
            "lock_id": lock["artifact_id"],
            "status": status,
            "source_gate_passed": status == "source_value_gate_passed",
            "method_promotion_authorized": False,
            "fresh_evaluation_authorized": False,
            "retry_authorized": False,
            "protected_data_read": False,
            "new_recordings": False,
            "gpu_work": False,
            **extra,
        },
    )
    print(
        f"terminal={status}; gate={value['source_gate_passed']}; id={value['artifact_id']}",
        flush=True,
    )
    return value


def run(output: Path) -> None:
    if (
        output.resolve() != OUTPUT
        or output.exists()
        or output.is_symlink()
        or ATTEMPT_LEDGER.exists()
        or ATTEMPT_LEDGER.is_symlink()
    ):
        raise ValueError("registered task-aware output root must be fresh")
    revision = clean_revision(ROOT)
    source_sha256 = _source_hashes()
    runtime = runtime_identity()
    upstream = verify_upstream(UPSTREAM)
    attempt = write_record(
        ATTEMPT_LEDGER,
        {
            "schema": "dlolab-task-aware-attempt-ledger-v1",
            "source_revision": revision,
            "source_sha256": source_sha256,
            "protocol": protocol(),
            "output_root": str(OUTPUT),
            "attempt_number": 1,
            "retry_authorized": False,
            "protected_data_read": False,
        },
    )
    output.mkdir(parents=True, exist_ok=False)
    lock = write_record(
        output / "lock.json",
        {
            "schema": "dlolab-task-aware-voi-lock-v1",
            "source_revision": revision,
            "source_sha256": source_sha256,
            "protocol": protocol(),
            "runtime": runtime,
            "upstream": upstream,
            "output_root": str(OUTPUT),
            "attempt_id": attempt["artifact_id"],
            "stage_order": [
                "particle-bank",
                "particle-selector-and-headroom",
                "truth-probes",
                "decisions",
                "truth-futures",
                "score",
            ],
            "retry_authorized": False,
            "protected_data_read": False,
        },
    )
    stage = "particle-bank"
    try:
        _execute_worker(output, stage)
        bank_seal, bank = _stage(output, lock, stage)
        if bank_seal["native"]["all_branches_qualified"] is not True:
            _terminal(
                output,
                lock,
                "particle_native_qualification_failed",
                particle_bank_id=bank_seal["artifact_id"],
                task_futures_generated=False,
                truth_probe_observations_generated=False,
                decisions_sealed=False,
            )
            return
        probe_trajectory = bank["probe_trajectory_m"]
        initial = bank["initial_position_m"]
        particle_probe = np.stack(
            [
                probe_features(probe_trajectory[index], initial)
                for index in range(len(PROBE_NAMES))
            ]
        )
        action_world_first = bank["action_trajectory_m"].transpose(1, 0, 2, 3, 4)
        particle_loss = task_losses(action_world_first, GOAL_TIP_Z_M)
        selectors = selector_analysis(particle_probe, particle_loss)
        headroom = task_headroom(particle_loss)
        analysis = write_record(
            output / "particle-analysis.json",
            {
                "schema": "dlolab-task-aware-particle-analysis-v1",
                "lock_id": lock["artifact_id"],
                "particle_bank_id": bank_seal["artifact_id"],
                "selector_analysis": selectors,
                "task_headroom": headroom,
                "truth_futures_read": False,
            },
        )
        selection = write_record(
            output / "probe-selection.json",
            {
                "schema": "dlolab-task-aware-probe-selection-v1",
                "lock_id": lock["artifact_id"],
                "particle_analysis_id": analysis["artifact_id"],
                "task_aware_probe_index": selectors["task_aware_probe_index"],
                "task_aware_probe_name": selectors["task_aware_probe_name"],
                "generic_mi_probe_index": selectors["generic_mi_probe_index"],
                "generic_mi_probe_name": selectors["generic_mi_probe_name"],
                "metrics": selectors,
                "truth_futures_read": False,
            },
        )
        if not selectors["passed"]:
            _terminal(
                output,
                lock,
                "selector_gate_failed",
                particle_analysis_id=analysis["artifact_id"],
                probe_selection_id=selection["artifact_id"],
                task_futures_generated=False,
                truth_probe_observations_generated=False,
                decisions_sealed=False,
            )
            return
        if not headroom["passed"]:
            _terminal(
                output,
                lock,
                "task_headroom_gate_failed",
                particle_analysis_id=analysis["artifact_id"],
                probe_selection_id=selection["artifact_id"],
                task_futures_generated=False,
                truth_probe_observations_generated=False,
                decisions_sealed=False,
            )
            return

        mi_probe = int(selectors["generic_mi_probe_index"])
        task_probe = int(selectors["task_aware_probe_index"])
        stage = "truth-probes"
        _execute_worker(output, stage, mi_probe, task_probe)
        probe_seal, truth_probe = _stage(output, lock, stage)
        if probe_seal["native"]["all_branches_qualified"] is not True:
            _terminal(
                output,
                lock,
                "truth_probe_native_qualification_failed",
                particle_analysis_id=analysis["artifact_id"],
                probe_selection_id=selection["artifact_id"],
                truth_probe_id=probe_seal["artifact_id"],
                task_futures_generated=False,
                truth_probe_observations_generated=True,
                decisions_sealed=False,
            )
            return
        truth_features = np.stack(
            [
                probe_features(trajectory, truth_probe["initial_position_m"])
                for trajectory in truth_probe["probe_trajectory_m"]
            ]
        )
        nuisance = noisy_probe_observations(truth_features)
        decisions = seal_decisions(
            nuisance["observation"],
            truth_probe["probe_indices"],
            particle_probe,
            particle_loss,
            truth_probe["goal_index"],
            task_probe,
            mi_probe,
        )
        decision_dir = output / "decisions"
        decision_dir.mkdir()
        decision_seal = write_record(
            decision_dir / "seal.json",
            {
                "schema": "dlolab-task-aware-decision-seal-v1",
                "lock_id": lock["artifact_id"],
                "probe_selection_id": selection["artifact_id"],
                "truth_probe_id": probe_seal["artifact_id"],
                "count": TRUTH_COUNT,
                "bundle": write_bundle(
                    decision_dir,
                    {
                        **decisions,
                        "probe_indices": truth_probe["probe_indices"],
                        "probe_observation_m": nuisance["observation"],
                        "shared_bias_m": nuisance["shared_bias"],
                        "independent_noise_m": nuisance["independent_noise"],
                    },
                ),
                "task_futures_read": False,
                "protected_data_read": False,
            },
        )

        stage = "truth-futures"
        _execute_worker(output, stage, mi_probe, task_probe)
        future_seal, truth_future = _stage(output, lock, stage)
        if future_seal["native"]["all_branches_qualified"] is not True:
            _terminal(
                output,
                lock,
                "truth_future_native_qualification_failed",
                particle_analysis_id=analysis["artifact_id"],
                probe_selection_id=selection["artifact_id"],
                truth_probe_id=probe_seal["artifact_id"],
                decision_id=decision_seal["artifact_id"],
                truth_future_id=future_seal["artifact_id"],
                task_futures_generated=True,
                truth_probe_observations_generated=True,
                decisions_sealed=True,
            )
            return
        if (
            not np.array_equal(truth_probe["bending"], truth_future["bending"])
            or not np.array_equal(truth_probe["twisting"], truth_future["twisting"])
            or not np.array_equal(truth_probe["goal_index"], truth_future["goal_index"])
            or not np.array_equal(
                truth_probe["goal_tip_z_m"], truth_future["goal_tip_z_m"]
            )
            or not np.array_equal(
                truth_probe["initial_position_m"], truth_future["initial_position_m"]
            )
            or probe_seal["native"]["initial_state_sha256"]
            != future_seal["native"]["initial_state_sha256"]
        ):
            raise RuntimeError("truth probe and task branches do not share exact world state")
        future_world_first = truth_future["action_trajectory_m"].transpose(1, 0, 2, 3, 4)
        truth_loss = realized_task_losses(
            future_world_first,
            truth_future["goal_tip_z_m"],
        )
        metrics = score_source(decisions, truth_loss)
        score_dir = output / "score"
        score_dir.mkdir()
        score_seal = write_record(
            score_dir / "seal.json",
            {
                "schema": "dlolab-task-aware-score-seal-v1",
                "lock_id": lock["artifact_id"],
                "decision_id": decision_seal["artifact_id"],
                "truth_future_id": future_seal["artifact_id"],
                "count": TRUTH_COUNT,
                "bundle": write_bundle(score_dir, {"truth_loss_m2": truth_loss}),
                "metrics": metrics,
                "protected_data_read": False,
            },
        )
        _terminal(
            output,
            lock,
            "source_value_gate_passed"
            if metrics["source_gate_passed"]
            else "source_value_gate_failed",
            particle_analysis_id=analysis["artifact_id"],
            probe_selection_id=selection["artifact_id"],
            truth_probe_id=probe_seal["artifact_id"],
            decision_id=decision_seal["artifact_id"],
            truth_future_id=future_seal["artifact_id"],
            score_id=score_seal["artifact_id"],
            task_futures_generated=True,
            truth_probe_observations_generated=True,
            decisions_sealed=True,
            metrics=metrics,
        )
    except Exception as exc:
        if not (output / "failure.json").exists():
            write_record(
                output / "failure.json",
                {
                    "schema": "dlolab-task-aware-runtime-failure-v1",
                    "lock_id": lock["artifact_id"],
                    "terminal_stage": stage,
                    "error": f"{type(exc).__name__}: {exc}",
                    "retry_authorized": False,
                    "protected_data_read": False,
                },
            )
        if not (output / "result.json").exists():
            _terminal(
                output,
                lock,
                "technical_failure",
                terminal_stage=stage,
                task_futures_generated=(output / "truth-futures" / "seal.json").exists(),
                truth_probe_observations_generated=(output / "truth-probes" / "seal.json").exists(),
                decisions_sealed=(output / "decisions" / "seal.json").exists(),
            )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--worker",
        choices=("particle-bank", "truth-probes", "truth-futures"),
    )
    parser.add_argument("--generic-mi-probe", type=int)
    parser.add_argument("--task-aware-probe", type=int)
    args = parser.parse_args()
    if args.worker:
        _worker(
            args.output,
            args.worker,
            args.generic_mi_probe,
            args.task_aware_probe,
        )
    else:
        if args.generic_mi_probe is not None or args.task_aware_probe is not None:
            raise ValueError("top-level run cannot preselect task-aware probes")
        run(args.output)


if __name__ == "__main__":
    main()
