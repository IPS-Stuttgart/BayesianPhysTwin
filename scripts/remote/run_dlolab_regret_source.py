"""Explicit bank -> calibration -> decision seal -> outcome stages; no retries."""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin_experiments.coupled_action_regret import (
    RegretCalibration,
    calibrate_simultaneous_regret,
    selected_commands,
)
from bayesian_phystwin_experiments.deform_state_restart import array_digest
from bayesian_phystwin_experiments.dlolab_regret_artifacts import (
    freeze,
    read_stage,
    validate_lock,
    write_bundle,
    write_record,
)
from bayesian_phystwin_experiments.dlolab_regret_study import (
    ARMS,
    MODES,
    action_offsets,
    commands_for_action,
    continue_all_actions,
    infer_parts,
    make_decisions,
    observe_prefix,
    particle_parameters,
    realized_losses,
    sample_worlds,
    score_decisions,
    start_prefix,
    validate_calibrations,
)

ROOT = Path(__file__).resolve().parents[2]


def calibration_from_seal(seal: dict[str, Any]) -> dict[str, RegretCalibration]:
    result = {
        name: RegretCalibration(**value) for name, value in seal["calibrators"].items()
    }
    validate_calibrations(result)
    return result


def command_identities(clamps: np.ndarray, decisions: np.ndarray) -> list[list[str]]:
    actions = tuple(commands_for_action(clamps, offset) for offset in action_offsets())
    for command in actions:
        command.flags.writeable = False
    result = []
    for case, row in enumerate(decisions):
        hashes = []
        for index in row:
            action = selected_commands(actions, int(index))
            if int(index) == 0 and action is not actions[0]:
                raise RuntimeError("fallback command object changed")
            hashes.append(array_digest(action[:, case]))
        result.append(hashes)
    return result


def execute(output: Path, stage: str) -> dict[str, Any]:
    lock = validate_lock(ROOT, output)
    upstream = Path(lock["upstream_root"])
    dependencies: dict[str, Any] = {}
    bank: dict[str, np.ndarray] = {}
    calibrations: dict[str, RegretCalibration] = {}
    sealed_prediction: dict[str, np.ndarray] = {}
    prediction_seal: dict[str, Any] = {}
    if stage != "bank":
        bank_seal, bank = read_stage(output, "bank", lock)
        dependencies["bank"] = bank_seal["artifact_id"]
    if stage in ("predict", "score"):
        cal_seal, _ = read_stage(output, "calibrate", lock)
        if cal_seal["dependencies"] != dependencies:
            raise ValueError("calibration bank binding changed")
        dependencies["calibrate"] = cal_seal["artifact_id"]
        calibrations = calibration_from_seal(cal_seal)
    if stage == "score":
        prediction_seal, sealed_prediction = read_stage(output, "predict", lock)
        if prediction_seal["dependencies"] != dependencies:
            raise ValueError("decision seal dependency changed")
        dependencies["predict"] = prediction_seal["artifact_id"]
        recalculated = infer_parts(
            sealed_prediction["observations"],
            sealed_prediction["goals"],
            bank["prefix"],
            bank["future"],
        )
        for name, value in recalculated.items():
            if array_digest(value) != array_digest(sealed_prediction[name]):
                raise ValueError(
                    "sealed inference changed before outcome authorization"
                )
        if array_digest(make_decisions(recalculated, calibrations)) != array_digest(
            sealed_prediction["decisions"]
        ):
            raise ValueError("sealed decisions changed before outcome authorization")
    directory = output / stage
    directory.mkdir(exist_ok=False)
    attempt = write_record(
        directory / "attempt.json",
        {
            "schema": "dlolab-regret-stage-attempt-v1",
            "stage": stage,
            "lock_id": lock["artifact_id"],
            "dependencies": dependencies,
            "evaluation_outcomes_generated": False,
            "protected_data_read": False,
            "retry_authorized": False,
        },
    )
    started = time.monotonic()
    runtime = None
    try:
        extras: dict[str, Any] = {}
        if stage == "bank":
            bending, velocity = particle_parameters()
            runtime, prefix, snapshot = start_prefix(upstream, bending, velocity)
            future = continue_all_actions(runtime, snapshot)
            arrays = {
                "prefix": prefix,
                "future": future,
                "bending": bending,
                "velocity": velocity,
            }
            count = 15
        else:
            role = "calibration" if stage == "calibrate" else "evaluation"
            world = sample_worlds(role)
            runtime, prefix, snapshot = start_prefix(
                upstream, world["bending"], world["velocity"]
            )
            observations = observe_prefix(prefix, world["sensor_error"])
            parts = infer_parts(
                observations, world["goals"], bank["prefix"], bank["future"]
            )
            arrays = {"observations": observations, "goals": world["goals"], **parts}
            count = len(observations)
            extras["prefix_sha256"] = array_digest(prefix)
            extras["model_identity"] = runtime.model_id
            extras["prefix_native_field_sha256"] = snapshot.field_digests
            if stage == "calibrate":
                future = continue_all_actions(runtime, snapshot)
                truth = realized_losses(future, world["goals"])
                calibration_values = {
                    mode: calibrate_simultaneous_regret(
                        parts["raw_upper"][:, index], truth
                    )
                    for index, mode in enumerate(MODES)
                }
                validate_calibrations(calibration_values)
                extras["calibrators"] = {
                    name: dataclasses.asdict(value)
                    for name, value in calibration_values.items()
                }
                arrays.update({"future": future, "losses": truth})
            elif stage == "predict":
                decisions = make_decisions(parts, calibrations)
                arrays["decisions"] = decisions
                extras["command_sha256"] = command_identities(
                    runtime.initial_positions[:, :2], decisions
                )
                extras["arms"] = list(ARMS)
            elif stage == "score":
                if (
                    extras["prefix_sha256"] != prediction_seal["prefix_sha256"]
                    or extras["prefix_native_field_sha256"]
                    != prediction_seal["prefix_native_field_sha256"]
                ):
                    raise RuntimeError(
                        "evaluation prefix replay changed before future generation"
                    )
                if extras["model_identity"] != prediction_seal["model_identity"]:
                    raise RuntimeError("evaluation world parameters changed")
                for name, value in arrays.items():
                    if array_digest(value) != array_digest(sealed_prediction[name]):
                        raise RuntimeError(
                            "permitted observation or inference replay changed"
                        )
                decisions = sealed_prediction["decisions"]
                commands = command_identities(
                    runtime.initial_positions[:, :2], decisions
                )
                if commands != prediction_seal["command_sha256"]:
                    raise RuntimeError("registered actions changed")
                future = continue_all_actions(runtime, snapshot)
                truth = realized_losses(future, world["goals"])
                result = score_decisions(
                    decisions, truth, parts["raw_upper"], calibrations
                )
                extras["result"] = result
                arrays = {"future": future, "losses": truth}
            else:
                raise ValueError("unknown stage")
        runtime.close()
        runtime = None
        bundle = write_bundle(directory, arrays)
        seal = write_record(
            directory / "seal.json",
            {
                "schema": "dlolab-regret-stage-seal-v1",
                "stage": stage,
                "lock_id": lock["artifact_id"],
                "attempt_id": attempt["artifact_id"],
                "dependencies": dependencies,
                "status": "ordinary_success",
                "count": count,
                "bundle": bundle,
                "wall_seconds": time.monotonic() - started,
                "protected_data_read": False,
                "physical_execution": False,
                "evaluation_outcomes_generated": stage == "score",
                **extras,
            },
        )
        print(f"{stage}: sealed {count}; artifact_id={seal['artifact_id']}", flush=True)
        if stage == "score":
            print(
                f"source_gate_passed={seal['result']['source_gate_passed']}", flush=True
            )
        return seal
    except Exception as error:
        write_record(
            directory / "failure.json",
            {
                "schema": "dlolab-regret-stage-failure-v1",
                "stage": stage,
                "lock_id": lock["artifact_id"],
                "attempt_id": attempt["artifact_id"],
                "error_type": type(error).__name__,
                "message": str(error),
                "source_gate_passed": False,
                "technical_failures": 1,
                "replacements": 0,
                "retry_authorized": False,
                "protected_data_read": False,
                "evaluation_outcome_access_stage": stage == "score",
            },
        )
        raise
    finally:
        if runtime is not None:
            runtime.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("freeze", "bank", "calibrate", "predict", "score")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--qualification", type=Path)
    args = parser.parse_args()
    if args.stage == "freeze":
        if args.upstream is None or args.qualification is None:
            parser.error("freeze requires --upstream and --qualification")
        value = freeze(ROOT, args.output, args.upstream, args.qualification)
        print(f"source_lock={value['artifact_id']}")
    else:
        if args.upstream is not None or args.qualification is not None:
            parser.error("execution paths come only from the lock")
        execute(args.output, args.stage)


if __name__ == "__main__":
    main()
