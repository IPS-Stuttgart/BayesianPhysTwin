#!/usr/bin/env python3
"""Run the one-attempt independent-action Slingshot policy certificate."""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from bayesian_phystwin.policy_gain_certificate import (
    LocalPolicyGainPredictor,
    PolicyGainCalibration,
)
from bayesian_phystwin_experiments.coupled_action_regret import RegretCalibration
from bayesian_phystwin_experiments.dlolab_benchmark import write_native_bundle
from bayesian_phystwin_experiments.dlolab_native import array_digest, file_digest
from bayesian_phystwin_experiments.dlolab_regret_artifacts import (
    clean_revision,
    read_record,
    write_record,
)
from bayesian_phystwin_experiments.dlolab_slingshot_batch import TRACE_NAMES
from bayesian_phystwin_experiments.dlolab_slingshot_belief import (
    BASELINE,
    prefix_observations,
)
from bayesian_phystwin_experiments.dlolab_slingshot_belief_native import (
    run_registered_worlds,
)
from bayesian_phystwin_experiments.dlolab_slingshot_cmaes import (
    task_metrics,
)
from bayesian_phystwin_experiments.dlolab_slingshot_cmaes import (
    worker_environment as native_worker_environment,
)
from bayesian_phystwin_experiments.dlolab_slingshot_independent_native_v3 import (
    independent_world_qa,
    run_registered_world,
    validate_singleton_arrays,
    validate_world_realization,
)
from bayesian_phystwin_experiments.dlolab_slingshot_policy_certificate_source_v3 import (
    ACTION_COUNT,
    CALIBRATION_RANK,
    COUNTS,
    QUALIFICATION_RESULT_ID,
    QUALIFICATION_RESULT_SHA256,
    calibrate,
    calibrate_simultaneous_guard,
    candidate_predictions,
    continuous_worlds,
    future_action_task,
    guarded_decisions,
    pre_future_checks,
    prefix_batch_count,
    prefix_task,
    protocol,
    score,
)
from bayesian_phystwin_experiments.dlolab_slingshot_process import (
    load_native_bundle,
    runtime,
)

Array: TypeAlias = NDArray[Any]
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(
    "/home/fpfaff/source-only/dlolab-slingshot-policy-certificate-source-v3"
)
ATTEMPT_LEDGER = Path(
    "/home/fpfaff/source-only/dlolab-slingshot-policy-certificate-source-v3.attempt.json"
)
QUALIFICATION_SUMMARY = (
    ROOT
    / "results/source/dlolab_slingshot_independent_native_qualification_v1/summary.json"
)
V2_RUNNER_PATH = (
    ROOT / "scripts/remote/run_dlolab_slingshot_policy_certificate_source_v2.py"
)
V2_SPEC = importlib.util.spec_from_file_location("slingshot_policy_v2", V2_RUNNER_PATH)
assert V2_SPEC is not None and V2_SPEC.loader is not None
V2_RUNNER = importlib.util.module_from_spec(V2_SPEC)
V2_SPEC.loader.exec_module(V2_RUNNER)

PREFIX_WORKERS = 1
FUTURE_WORKERS = 8
SOURCES = (
    "src/bayesian_phystwin_experiments/dlolab_slingshot_independent_native_v3.py",
    "src/bayesian_phystwin_experiments/dlolab_slingshot_policy_certificate_source_v3.py",
    "scripts/remote/run_dlolab_slingshot_policy_certificate_source_v3.py",
    "tests/test_dlolab_slingshot_independent_native_v3.py",
    "tests/test_dlolab_slingshot_policy_certificate_source_v3.py",
    "tests/test_dlolab_slingshot_policy_certificate_source_v3_custody.py",
    "docs/dlolab_slingshot_policy_certificate_source_v3.md",
    "results/source/dlolab_slingshot_independent_native_qualification_v1/summary.json",
    "scripts/remote/run_dlolab_slingshot_policy_certificate_source_v2.py",
    "src/bayesian_phystwin_experiments/dlolab_slingshot_policy_certificate_source_v2.py",
    "src/bayesian_phystwin_experiments/dlolab_slingshot_policy_certificate_v2.py",
    "src/bayesian_phystwin_experiments/dlolab_slingshot_policy_certificate_v1.py",
    "src/bayesian_phystwin_experiments/dlolab_slingshot_belief.py",
    "src/bayesian_phystwin_experiments/dlolab_slingshot_belief_native.py",
    "src/bayesian_phystwin_experiments/dlolab_slingshot_batch.py",
    "src/bayesian_phystwin_experiments/dlolab_slingshot_process.py",
    "src/bayesian_phystwin_experiments/dlolab_benchmark.py",
    "src/bayesian_phystwin_experiments/dlolab_regret_artifacts.py",
    "src/bayesian_phystwin/policy_gain_certificate.py",
    "src/bayesian_phystwin/guard_harm_risk.py",
    "src/bayesian_phystwin_experiments/coupled_action_regret.py",
)


def load_parent() -> tuple[dict[str, Any], dict[str, Array], LocalPolicyGainPredictor]:
    """Load the exact v2 reference bank without opening its failed futures."""

    return cast(
        tuple[dict[str, Any], dict[str, Array], LocalPolicyGainPredictor],
        V2_RUNNER.load_parent(),
    )


def load_qualification() -> dict[str, Any]:
    if (
        QUALIFICATION_SUMMARY.is_symlink()
        or file_digest(QUALIFICATION_SUMMARY) != QUALIFICATION_RESULT_SHA256
    ):
        raise ValueError("independent native qualification file changed")
    value = read_record(QUALIFICATION_SUMMARY)
    if (
        value.get("artifact_id") != QUALIFICATION_RESULT_ID
        or value.get("status") != "passed"
        or value.get("ordinary_processes") != 64
        or value.get("failed_processes") != 0
        or value.get("qualified_worlds") != 8
        or value.get("qualification_passed") is not True
        or value.get("v3_protocol_freeze_authorized") is not True
        or value.get("v3_scientific_execution_authorized") is not False
        or value.get("retry_authorized") is not False
        or value.get("replacement_authorized") is not False
        or value.get("protected_data_read") is not False
    ):
        raise ValueError("passing independent native qualification required")
    return cast(dict[str, Any], value)


def _source_hashes() -> dict[str, str]:
    if any(not (ROOT / name).is_file() for name in SOURCES):
        raise ValueError("complete v3 source tree required")
    return {name: file_digest(ROOT / name) for name in SOURCES}


def freeze(output: Path) -> dict[str, Any]:
    if (
        output.resolve() != OUTPUT_ROOT
        or output.exists()
        or ATTEMPT_LEDGER.exists()
    ):
        raise ValueError("only the fresh registered one-attempt root is authorized")
    revision = clean_revision(ROOT)
    parent, _, _ = load_parent()
    qualification = load_qualification()
    if runtime() != parent["runtime"]:
        raise ValueError("exact parent-qualified runtime required")
    output.mkdir(parents=True, exist_ok=False)
    lock = write_record(
        output / "lock.json",
        {
            "schema": "dlolab-slingshot-policy-certificate-lock-v3",
            "source_revision": revision,
            "source_sha256": _source_hashes(),
            "protocol": protocol(),
            "parent": parent,
            "qualification_result_id": qualification["artifact_id"],
            "qualification_result_sha256": QUALIFICATION_RESULT_SHA256,
            "controls": parent["controls"],
            "assets_root": parent["assets_root"],
            "runtime": parent["runtime"],
            "prefix_workers": PREFIX_WORKERS,
            "future_workers": FUTURE_WORKERS,
            "output_root": str(output.resolve()),
            "attempt_ledger": str(ATTEMPT_LEDGER.resolve()),
            "protected_data_read": False,
        },
    )
    write_record(
        ATTEMPT_LEDGER,
        {
            "schema": "dlolab-slingshot-policy-certificate-attempt-v3",
            "lock_id": lock["artifact_id"],
            "source_revision": revision,
            "output_root": str(output.resolve()),
            "attempt_number": 1,
            "retry_authorized": False,
            "replacement_authorized": False,
            "protected_data_read": False,
        },
    )
    return cast(dict[str, Any], lock)


def validate_lock(output: Path) -> dict[str, Any]:
    if output.resolve() != OUTPUT_ROOT:
        raise ValueError("only the registered v3 root is authorized")
    lock = read_record(output / "lock.json")
    attempt = read_record(ATTEMPT_LEDGER)
    parent, _, _ = load_parent()
    qualification = load_qualification()
    if (
        lock.get("schema") != "dlolab-slingshot-policy-certificate-lock-v3"
        or lock.get("source_revision") != clean_revision(ROOT)
        or lock.get("source_sha256") != _source_hashes()
        or lock.get("protocol") != protocol()
        or lock.get("parent") != parent
        or lock.get("qualification_result_id") != qualification["artifact_id"]
        or lock.get("qualification_result_sha256") != QUALIFICATION_RESULT_SHA256
        or lock.get("controls") != parent["controls"]
        or lock.get("assets_root") != parent["assets_root"]
        or lock.get("runtime") != runtime()
        or lock.get("prefix_workers") != PREFIX_WORKERS
        or lock.get("future_workers") != FUTURE_WORKERS
        or lock.get("output_root") != str(output.resolve())
        or lock.get("attempt_ledger") != str(ATTEMPT_LEDGER.resolve())
        or attempt.get("schema")
        != "dlolab-slingshot-policy-certificate-attempt-v3"
        or attempt.get("lock_id") != lock.get("artifact_id")
        or attempt.get("source_revision") != lock.get("source_revision")
        or attempt.get("output_root") != str(output.resolve())
        or attempt.get("attempt_number") != 1
        or attempt.get("retry_authorized") is not False
        or attempt.get("replacement_authorized") is not False
    ):
        raise ValueError("frozen v3 policy-certificate lock changed")
    return cast(dict[str, Any], lock)


def _prefix_worlds(spec: dict[str, Any]) -> list[dict[str, Any]]:
    roster = continuous_worlds(spec["role"])
    return [roster[index] for index in spec["world_indices"]]


def _prefix_controls(lock: dict[str, Any]) -> Array:
    bank = np.asarray(lock["controls"], dtype=np.float64)
    if bank.shape != (ACTION_COUNT, 3, 6):
        raise ValueError("frozen action bank changed")
    return np.repeat(bank[BASELINE : BASELINE + 1], 8, axis=0)


def _validate_prefix_realization(
    native: dict[str, Any], expected_worlds: list[dict[str, Any]]
) -> None:
    realization = native.get("world_realization", {})
    if realization.get("bending") != [
        [world["bending_E"] for world in expected_worlds]
    ] or realization.get("stretching") != [
        [world["stretching_K"] for world in expected_worlds]
    ]:
        raise ValueError("realized prefix material parameters changed")
    for name, y, z in (("sphere", 0.06, 0.2), ("cube", 0.23, 0.22)):
        expected = np.asarray(
            [[0.12 + world["x_offset_m"], y, z] for world in expected_worlds]
        )
        actual = np.asarray(realization.get(f"{name}_initial_position_m"))
        if actual.shape != (8, 3) or not np.allclose(
            actual, expected, rtol=0.0, atol=1e-15
        ):
            raise ValueError("realized prefix object placement changed")


def _prefix_qa(
    arrays: dict[str, Array],
    native: dict[str, Any],
    expected_controls: Array,
    expected_worlds: list[dict[str, Any]],
) -> dict[str, Any]:
    if (
        set(arrays) != set(TRACE_NAMES + ("controls",))
        or any(arrays[name].shape[:2] != (300, 8) for name in TRACE_NAMES)
        or any(not np.isfinite(value).all() for value in arrays.values())
        or array_digest(arrays["controls"]) != array_digest(expected_controls)
        or native.get("native_steps") != 300
        or native.get("future_simulated") is not False
        or native.get("reward_scored") is not False
        or native.get("hidden_state_restart") is not False
    ):
        raise ValueError("causal-prefix native contract changed")
    _validate_prefix_realization(native, expected_worlds)
    prefix_observations(arrays)
    fixed = float(
        np.max(
            np.abs(
                arrays["rod_pos_m"][:, :, [0, 1, 10, 11]]
                - arrays["rod_pos_m"][:1, :, [0, 1, 10, 11]]
            )
        )
    )
    checks = {
        "complete_causal_prefix": True,
        "no_future_simulated": True,
        "no_reward_scored": True,
        "fixed_endpoints": fixed <= 1e-9,
    }
    return {
        "checks": checks,
        "fixed_endpoint_error_m": fixed,
        "qa_passed": bool(all(checks.values())),
    }


def _single_action_qa(
    arrays: dict[str, Array],
    native: dict[str, Any],
    expected_control: Array,
    world: dict[str, Any],
    *,
    world_count: int,
) -> dict[str, Any]:
    validate_world_realization(native, world, world_count=world_count)
    validate_singleton_arrays(arrays)
    if (
        array_digest(arrays["controls"]) != array_digest(expected_control)
        or native.get("native_steps") != 900
        or native.get("environment_count") != 1
        or native.get("fresh_python_process") is not True
        or native.get("world") != world
        or native.get("native_cumulative_reward")
        != [task_metrics(arrays)["native_reward"]]
    ):
        raise ValueError("independent future action arithmetic changed")
    fixed = float(
        np.max(
            np.abs(
                arrays["rod_pos_m"][:, :, [0, 1, 10, 11]]
                - arrays["rod_pos_m"][:1, :, [0, 1, 10, 11]]
            )
        )
    )
    checks = {
        "complete_finite_singleton": True,
        "exact_control": True,
        "exact_world_realization": True,
        "exact_reward_arithmetic": True,
        "fixed_endpoints": fixed <= 1e-9,
    }
    return {
        "checks": checks,
        "fixed_endpoint_error_m": fixed,
        "qa_passed": bool(all(checks.values())),
    }


def _candidate_seal(output: Path, lock: dict[str, Any], role: str) -> dict[str, Any]:
    seal = read_record(output / f"{role}-candidates/seal.json")
    if (
        seal.get("schema") != "dlolab-slingshot-policy-candidates-v3"
        or seal.get("lock_id") != lock["artifact_id"]
        or seal.get("role") != role
        or seal.get("future_simulated") is not False
        or seal.get("future_read") is not False
    ):
        raise ValueError("complete sealed v3 candidates required")
    return cast(dict[str, Any], seal)


def _calibration_seal(output: Path, lock: dict[str, Any]) -> dict[str, Any]:
    seal = read_record(output / "calibration/seal.json")
    if (
        seal.get("schema") != "dlolab-slingshot-policy-calibration-v3"
        or seal.get("lock_id") != lock["artifact_id"]
        or seal.get("policy_calibration", {}).get("calibration_count")
        != COUNTS["calibration"]
        or seal.get("policy_calibration", {}).get("rank") != CALIBRATION_RANK
        or seal.get("simultaneous_calibration", {}).get("count")
        != COUNTS["calibration"]
        or seal.get("simultaneous_calibration", {}).get("rank")
        != CALIBRATION_RANK
        or seal.get("all_native_qa") is not True
    ):
        raise ValueError("complete registered v3 calibration required")
    return cast(dict[str, Any], seal)


def _evaluation_barrier(
    output: Path, lock: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    decision = read_record(output / "evaluation-decisions/seal.json")
    barrier = read_record(output / "evaluation-decision-barrier.json")
    if (
        decision.get("schema") != "dlolab-slingshot-policy-decisions-v3"
        or decision.get("lock_id") != lock["artifact_id"]
        or barrier.get("schema") != "dlolab-slingshot-policy-barrier-v3"
        or barrier.get("lock_id") != lock["artifact_id"]
        or barrier.get("decision_seal_id") != decision["artifact_id"]
        or barrier.get("pre_future_gate_passed") is not True
        or barrier.get("future_simulated") is not False
        or barrier.get("future_read") is not False
    ):
        raise ValueError("passing v3 evaluation decision barrier required")
    return cast(dict[str, Any], decision), cast(dict[str, Any], barrier)


def _expected_authorization(
    output: Path, lock: dict[str, Any], spec: dict[str, Any]
) -> dict[str, Any]:
    if spec["kind"] == "prefix_only":
        return {"gate": "registered_causal_prefix"}
    if spec["role"] == "calibration":
        candidate, _ = load_candidates(output, lock, "calibration")
        return {
            "gate": "reproduced_calibration_candidates",
            "candidate_seal_id": candidate["artifact_id"],
        }
    decision, _, barrier = load_evaluation_decisions(output, lock)
    return {
        "gate": "reproduced_passing_evaluation_decision_barrier",
        "decision_seal_id": decision["artifact_id"],
        "barrier_id": barrier["artifact_id"],
    }


def _spec(role: str, kind: str, index: int, action: int | None) -> dict[str, Any]:
    if kind == "prefix" and action is None:
        return prefix_task(role, index)
    if kind == "future" and action is not None:
        return future_action_task(role, index, action)
    raise ValueError("complete registered v3 worker coordinates required")


def worker(
    output: Path,
    role: str,
    kind: str,
    index: int,
    action: int | None = None,
) -> None:
    lock = validate_lock(output)
    spec = _spec(role, kind, index, action)
    authorization = _expected_authorization(output, lock, spec)
    directory = output / spec["name"]
    directory.mkdir(exist_ok=False)
    claim = write_record(
        directory / "claim.json",
        {
            "schema": "dlolab-slingshot-policy-task-claim-v3",
            "lock_id": lock["artifact_id"],
            "task": spec,
            "authorization": authorization,
            "retry_authorized": False,
            "replacement_authorized": False,
            "protected_data_read": False,
        },
    )
    try:
        if kind == "prefix":
            expected_controls = _prefix_controls(lock)
            expected_worlds = _prefix_worlds(spec)
            arrays, native = run_registered_worlds(
                Path(lock["assets_root"]) / "upstream",
                directory,
                expected_controls,
                expected_worlds,
                prefix_only=True,
            )
            qa = _prefix_qa(arrays, native, expected_controls, expected_worlds)
        else:
            if action is None:
                raise AssertionError("future action coordinate disappeared")
            controls = np.asarray(lock["controls"], dtype=np.float64)
            expected_control = controls[action : action + 1]
            world = continuous_worlds(role)[index]
            arrays, native = run_registered_world(
                Path(lock["assets_root"]) / "upstream",
                directory,
                expected_control,
                world,
                world_count=COUNTS[role],
            )
            qa = _single_action_qa(
                arrays,
                native,
                expected_control,
                world,
                world_count=COUNTS[role],
            )
        if not qa["qa_passed"]:
            raise ValueError("native v3 task QA failed")
        write_record(
            directory / "seal.json",
            {
                "schema": "dlolab-slingshot-policy-task-seal-v3",
                "lock_id": lock["artifact_id"],
                "claim_id": claim["artifact_id"],
                "task": spec,
                "native": native,
                "qa": qa,
                "bundle": write_native_bundle(directory, arrays),
            },
        )
    except Exception as error:
        write_record(
            directory / "failure.json",
            {
                "schema": "dlolab-slingshot-policy-task-failure-v3",
                "lock_id": lock["artifact_id"],
                "claim_id": claim["artifact_id"],
                "task": spec,
                "error_type": type(error).__name__,
                "message": str(error),
                "retry_authorized": False,
                "replacement_authorized": False,
                "protected_data_read": False,
            },
        )
        raise


def _task_records(
    output: Path,
    lock: dict[str, Any],
    spec: dict[str, Any],
    *,
    authorization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    directory = output / spec["name"]
    claim = read_record(directory / "claim.json")
    seal = read_record(directory / "seal.json")
    expected_authorization = (
        _expected_authorization(output, lock, spec)
        if authorization is None
        else authorization
    )
    if (
        claim.get("schema") != "dlolab-slingshot-policy-task-claim-v3"
        or seal.get("schema") != "dlolab-slingshot-policy-task-seal-v3"
        or claim.get("lock_id") != lock["artifact_id"]
        or seal.get("lock_id") != lock["artifact_id"]
        or claim.get("task") != spec
        or seal.get("task") != spec
        or claim.get("authorization") != expected_authorization
        or seal.get("claim_id") != claim["artifact_id"]
        or claim.get("retry_authorized") is not False
        or claim.get("replacement_authorized") is not False
        or claim.get("protected_data_read") is not False
        or seal.get("qa", {}).get("qa_passed") is not True
    ):
        raise ValueError("native v3 task custody changed")
    return cast(dict[str, Any], seal)


def load_prefix_task(
    output: Path, lock: dict[str, Any], role: str, batch: int
) -> tuple[dict[str, Any], dict[str, Array]]:
    spec = prefix_task(role, batch)
    seal = _task_records(
        output,
        lock,
        spec,
        authorization={"gate": "registered_causal_prefix"},
    )
    arrays = load_native_bundle(output / spec["name"], seal["bundle"])
    qa = _prefix_qa(arrays, seal["native"], _prefix_controls(lock), _prefix_worlds(spec))
    if not qa["qa_passed"] or seal.get("qa") != qa:
        raise ValueError("native v3 prefix QA changed")
    return seal, arrays


def load_future_action(
    output: Path,
    lock: dict[str, Any],
    role: str,
    world_index: int,
    action_index: int,
    *,
    authorization: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Array]]:
    spec = future_action_task(role, world_index, action_index)
    seal = _task_records(output, lock, spec, authorization=authorization)
    arrays = load_native_bundle(output / spec["name"], seal["bundle"])
    controls = np.asarray(lock["controls"], dtype=np.float64)
    world = continuous_worlds(role)[world_index]
    qa = _single_action_qa(
        arrays,
        seal["native"],
        controls[action_index : action_index + 1],
        world,
        world_count=COUNTS[role],
    )
    if not qa["qa_passed"] or seal.get("qa") != qa:
        raise ValueError("native v3 future action QA changed")
    return seal, arrays


def validate_task_failure(
    output: Path, lock: dict[str, Any], spec: dict[str, Any]
) -> None:
    directory = output / spec["name"]
    claim = read_record(directory / "claim.json")
    failure = read_record(directory / "failure.json")
    if (
        (directory / "seal.json").exists()
        or claim.get("schema") != "dlolab-slingshot-policy-task-claim-v3"
        or failure.get("schema") != "dlolab-slingshot-policy-task-failure-v3"
        or claim.get("lock_id") != lock["artifact_id"]
        or failure.get("lock_id") != lock["artifact_id"]
        or claim.get("task") != spec
        or failure.get("task") != spec
        or claim.get("authorization")
        != _expected_authorization(output, lock, spec)
        or failure.get("claim_id") != claim["artifact_id"]
        or failure.get("retry_authorized") is not False
        or failure.get("replacement_authorized") is not False
        or failure.get("protected_data_read") is not False
        or not isinstance(failure.get("error_type"), str)
        or not isinstance(failure.get("message"), str)
    ):
        raise ValueError("native v3 task failure custody changed")


def execute(
    output: Path,
    lock: dict[str, Any],
    role: str,
    kind: str,
    index: int,
    action: int | None = None,
) -> int:
    spec = _spec(role, kind, index, action)
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--output",
        str(output.resolve()),
        "--worker-role",
        role,
        "--worker-kind",
        kind,
        "--worker-index",
        str(index),
    ]
    if action is not None:
        command.extend(("--worker-action", str(action)))
    with (output / f"{spec['name']}.log").open("x") as stream:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=native_worker_environment(lock["runtime"]),
            check=False,
        )
    return int(completed.returncode)


def execute_many(
    output: Path,
    lock: dict[str, Any],
    role: str,
    kind: str,
    *,
    workers: int,
) -> None:
    coordinates = (
        [(index, None) for index in range(prefix_batch_count(role))]
        if kind == "prefix"
        else [
            (world_index, action_index)
            for world_index in range(COUNTS[role])
            for action_index in range(ACTION_COUNT)
        ]
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(execute, output, lock, role, kind, index, action)
            for index, action in coordinates
        ]
        returncodes = [future.result() for future in futures]
    authorization = (
        {"gate": "registered_causal_prefix"}
        if kind == "prefix"
        else _role_authorization(output, lock, role)
    )
    failures: list[str] = []
    for (index, action), code in zip(coordinates, returncodes, strict=True):
        spec = _spec(role, kind, index, action)
        try:
            if code == 0 and kind == "prefix":
                load_prefix_task(output, lock, role, index)
            elif code == 0 and action is not None:
                load_future_action(
                    output,
                    lock,
                    role,
                    index,
                    action,
                    authorization=authorization,
                )
            else:
                validate_task_failure(output, lock, spec)
                failures.append(spec["name"])
        except Exception as error:
            failures.append(f"{spec['name']}:{type(error).__name__}")
    if failures:
        raise RuntimeError(f"terminal v3 native task failures: {failures}")


def _candidate_artifact(
    output: Path, lock: dict[str, Any], role: str
) -> tuple[dict[str, Any], dict[str, Array]]:
    parent, bank, predictor = load_parent()
    truth: list[Array] = []
    prefix_ids: list[str] = []
    for batch in range(prefix_batch_count(role)):
        seal, arrays = load_prefix_task(output, lock, role, batch)
        prefix_ids.append(seal["artifact_id"])
        truth.extend(prefix_observations(arrays))
    candidate = candidate_predictions(
        role,
        np.stack(truth),
        bank["prefix"],
        bank["reward"],
        predictor,
    )
    metadata = {
        "schema": "dlolab-slingshot-policy-candidates-v3",
        "lock_id": lock["artifact_id"],
        "role": role,
        "parent_bank_id": parent["bank_id"],
        "development_artifact_id": parent["development_artifact_id"],
        "prefix_seal_ids": prefix_ids,
        "future_simulated": False,
        "future_read": False,
        "protected_data_read": False,
    }
    return metadata, candidate


def seal_candidates(output: Path, lock: dict[str, Any], role: str) -> dict[str, Any]:
    metadata, arrays = _candidate_artifact(output, lock, role)
    directory = output / f"{role}-candidates"
    directory.mkdir(exist_ok=False)
    bundle = write_native_bundle(directory, arrays)
    return cast(
        dict[str, Any],
        write_record(directory / "seal.json", {**metadata, "bundle": bundle}),
    )


def load_candidates(
    output: Path, lock: dict[str, Any], role: str
) -> tuple[dict[str, Any], dict[str, Array]]:
    seal = _candidate_seal(output, lock, role)
    arrays = load_native_bundle(output / f"{role}-candidates", seal["bundle"])
    metadata, expected = _candidate_artifact(output, lock, role)
    if any(seal.get(key) != value for key, value in metadata.items()) or (
        set(arrays) != set(expected)
        or any(
            array_digest(arrays[name]) != array_digest(expected[name]) for name in arrays
        )
    ):
        raise ValueError("sealed v3 candidate predictions do not reproduce")
    return seal, arrays


def _role_authorization(
    output: Path, lock: dict[str, Any], role: str
) -> dict[str, Any]:
    if role == "calibration":
        candidate, _ = load_candidates(output, lock, role)
        return {
            "gate": "reproduced_calibration_candidates",
            "candidate_seal_id": candidate["artifact_id"],
        }
    decision, _, barrier = load_evaluation_decisions(output, lock)
    return {
        "gate": "reproduced_passing_evaluation_decision_barrier",
        "decision_seal_id": decision["artifact_id"],
        "barrier_id": barrier["artifact_id"],
    }


def _world_qualification(
    output: Path,
    lock: dict[str, Any],
    role: str,
    index: int,
    *,
    write: bool,
    authorization: dict[str, Any],
) -> tuple[dict[str, Any], Array, list[str]]:
    rows: list[dict[str, Array]] = []
    reports: list[dict[str, Any]] = []
    seal_ids: list[str] = []
    for action_index in range(ACTION_COUNT):
        seal, arrays = load_future_action(
            output,
            lock,
            role,
            index,
            action_index,
            authorization=authorization,
        )
        rows.append(arrays)
        reports.append(seal["native"])
        seal_ids.append(seal["artifact_id"])
    world = continuous_worlds(role)[index]
    controls = np.asarray(lock["controls"], dtype=np.float64)
    qa = independent_world_qa(
        rows,
        reports,
        controls,
        world,
        world_count=COUNTS[role],
    )
    if not qa["qa_passed"]:
        raise ValueError("independent v3 world QA failed")
    rewards = np.asarray(
        [task_metrics(row)["native_reward"] for row in rows[:7]],
        dtype=np.float64,
    )
    metadata = {
        "schema": "dlolab-slingshot-policy-world-qualification-v3",
        "lock_id": lock["artifact_id"],
        "role": role,
        "world": world,
        "action_seal_ids": seal_ids,
        "qa": qa,
        "rewards": rewards.tolist(),
        "retry_authorized": False,
        "replacement_authorized": False,
        "protected_data_read": False,
    }
    path = output / f"{role}-future-{index:03d}-qualification.json"
    record = write_record(path, metadata) if write else read_record(path)
    expected = {**metadata, "artifact_id": record.get("artifact_id")}
    if record != expected or record.get("artifact_id") is None:
        raise ValueError("v3 world qualification does not reproduce")
    return cast(dict[str, Any], record), rewards, seal_ids


def _future_rewards(
    output: Path, lock: dict[str, Any], role: str, *, write: bool
) -> tuple[Array, list[str], list[str], bool]:
    authorization = _role_authorization(output, lock, role)
    rewards: list[Array] = []
    world_ids: list[str] = []
    action_ids: list[str] = []
    for index in range(COUNTS[role]):
        qualification, reward, seals = _world_qualification(
            output,
            lock,
            role,
            index,
            write=write,
            authorization=authorization,
        )
        rewards.append(reward)
        world_ids.append(qualification["artifact_id"])
        action_ids.extend(seals)
    return np.stack(rewards), world_ids, action_ids, True


def _calibration_artifact(
    output: Path, lock: dict[str, Any], *, write_qualifications: bool
) -> tuple[
    dict[str, Any],
    dict[str, Array],
    PolicyGainCalibration,
    RegretCalibration,
]:
    candidate_seal, candidate = load_candidates(output, lock, "calibration")
    rewards, world_ids, action_ids, all_qa = _future_rewards(
        output, lock, "calibration", write=write_qualifications
    )
    calibration, realized = calibrate(candidate, rewards)
    simultaneous = calibrate_simultaneous_guard(candidate, rewards)
    metadata = {
        "schema": "dlolab-slingshot-policy-calibration-v3",
        "lock_id": lock["artifact_id"],
        "candidate_seal_id": candidate_seal["artifact_id"],
        "future_action_seal_ids": action_ids,
        "world_qualification_ids": world_ids,
        "all_native_qa": all_qa,
        "policy_calibration": dataclasses.asdict(calibration),
        "simultaneous_calibration": dataclasses.asdict(simultaneous),
        "evaluation_prefix_read": False,
        "evaluation_future_simulated": False,
        "evaluation_future_read": False,
        "protected_data_read": False,
    }
    return (
        metadata,
        {"rewards": rewards, "realized_candidate_gain": realized},
        calibration,
        simultaneous,
    )


def seal_calibration(output: Path, lock: dict[str, Any]) -> dict[str, Any]:
    metadata, arrays, _, _ = _calibration_artifact(
        output, lock, write_qualifications=True
    )
    directory = output / "calibration"
    directory.mkdir(exist_ok=False)
    bundle = write_native_bundle(directory, arrays)
    return cast(
        dict[str, Any],
        write_record(directory / "seal.json", {**metadata, "bundle": bundle}),
    )


def load_calibration(
    output: Path, lock: dict[str, Any]
) -> tuple[dict[str, Any], PolicyGainCalibration, RegretCalibration]:
    seal = _calibration_seal(output, lock)
    arrays = load_native_bundle(output / "calibration", seal["bundle"])
    metadata, expected, calibration, simultaneous = _calibration_artifact(
        output, lock, write_qualifications=False
    )
    if set(arrays) != set(expected) or any(
        array_digest(arrays[name]) != array_digest(expected[name]) for name in arrays
    ):
        raise ValueError("sealed v3 calibration arrays changed")
    if any(seal.get(key) != value for key, value in metadata.items()):
        raise ValueError("sealed v3 calibration does not reproduce")
    return seal, calibration, simultaneous


def seal_evaluation_decisions(
    output: Path, lock: dict[str, Any]
) -> dict[str, Any]:
    candidate_seal, candidate = load_candidates(output, lock, "evaluation")
    calibration_seal, calibration, simultaneous = load_calibration(output, lock)
    guarded = guarded_decisions(candidate, calibration, simultaneous)
    preflight = pre_future_checks(guarded, all_prefix_qa=True)
    directory = output / "evaluation-decisions"
    directory.mkdir(exist_ok=False)
    bundle = write_native_bundle(directory, {**candidate, **guarded})
    decision = write_record(
        directory / "seal.json",
        {
            "schema": "dlolab-slingshot-policy-decisions-v3",
            "lock_id": lock["artifact_id"],
            "candidate_seal_id": candidate_seal["artifact_id"],
            "calibration_seal_id": calibration_seal["artifact_id"],
            "pre_future": preflight,
            "bundle": bundle,
            "future_simulated": False,
            "future_read": False,
            "protected_data_read": False,
        },
    )
    barrier = write_record(
        output / "evaluation-decision-barrier.json",
        {
            "schema": "dlolab-slingshot-policy-barrier-v3",
            "lock_id": lock["artifact_id"],
            "decision_seal_id": decision["artifact_id"],
            "calibration_seal_id": calibration_seal["artifact_id"],
            "pre_future": preflight,
            "pre_future_gate_passed": preflight["pre_future_gate_passed"],
            "future_simulated": False,
            "future_read": False,
            "protected_data_read": False,
        },
    )
    if not preflight["pre_future_gate_passed"]:
        raise ValueError("registered v3 evaluation pre-future gate failed")
    return cast(dict[str, Any], barrier)


def load_evaluation_decisions(
    output: Path, lock: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Array], dict[str, Any]]:
    decision, barrier = _evaluation_barrier(output, lock)
    arrays = load_native_bundle(output / "evaluation-decisions", decision["bundle"])
    _, candidate = load_candidates(output, lock, "evaluation")
    _, calibration, simultaneous = load_calibration(output, lock)
    expected_guarded = guarded_decisions(candidate, calibration, simultaneous)
    expected = {**candidate, **expected_guarded}
    if set(arrays) != set(expected) or any(
        array_digest(arrays[name]) != array_digest(expected[name]) for name in arrays
    ):
        raise ValueError("v3 evaluation decisions do not reproduce")
    preflight = pre_future_checks(expected_guarded, all_prefix_qa=True)
    if barrier.get("pre_future") != preflight:
        raise ValueError("v3 evaluation pre-future checks changed")
    return decision, arrays, barrier


def _score_inputs(
    decisions: dict[str, Array],
) -> tuple[dict[str, Array], dict[str, Array]]:
    candidate = {
        name: decisions[name]
        for name in (
            "truth_prefix_m",
            "observation_m",
            "features",
            "expected_losses",
            "mean_raw_upper",
            "candidate_actions",
            "predicted_gain",
            "neighbor_indices",
            "neighbor_squared_distances",
        )
    }
    guarded = {
        name: decisions[name]
        for name in (
            "decisions",
            "accepted_mask",
            "simultaneous_accepted_mask",
            "lower_gain_bound",
        )
    }
    return candidate, guarded


def verify_result(output: Path) -> dict[str, Any]:
    lock = validate_lock(output)
    if (output / "failure.json").exists():
        raise ValueError("retained v3 failure cannot verify as an ordinary result")
    result = read_record(output / "result.json")
    decision, decisions, barrier = load_evaluation_decisions(output, lock)
    calibration_seal, calibration, simultaneous = load_calibration(output, lock)
    candidate, guarded = _score_inputs(decisions)
    rewards, world_ids, action_ids, all_qa = _future_rewards(
        output, lock, "evaluation", write=False
    )
    expected = {
        **score(
            candidate,
            guarded,
            rewards,
            calibration,
            simultaneous,
            all_native_qa=all_qa,
            pre_future_gate_passed=barrier["pre_future_gate_passed"],
        ),
        "lock_id": lock["artifact_id"],
        "calibration_seal_id": calibration_seal["artifact_id"],
        "decision_seal_id": decision["artifact_id"],
        "barrier_id": barrier["artifact_id"],
        "future_action_seal_ids": action_ids,
        "world_qualification_ids": world_ids,
        "ordinary_evaluation_worlds": COUNTS["evaluation"],
        "ordinary_evaluation_action_processes": COUNTS["evaluation"] * ACTION_COUNT,
        "technical_failures": 0,
        "replacements": 0,
        "retries": 0,
    }
    observed = {key: value for key, value in result.items() if key != "artifact_id"}
    if observed != expected:
        raise ValueError("v3 policy-certificate result does not reproduce")
    return cast(dict[str, Any], result)


def run(output: Path) -> dict[str, Any]:
    stage = "freeze"
    lock: dict[str, Any] | None = None
    try:
        lock = freeze(output)
        stage = "calibration-prefixes"
        execute_many(
            output, lock, "calibration", "prefix", workers=PREFIX_WORKERS
        )
        stage = "calibration-candidates"
        seal_candidates(output, lock, "calibration")
        stage = "calibration-independent-action-futures"
        execute_many(
            output, lock, "calibration", "future", workers=FUTURE_WORKERS
        )
        stage = "calibration-world-qualification"
        seal_calibration(output, lock)
        stage = "evaluation-prefixes"
        execute_many(output, lock, "evaluation", "prefix", workers=PREFIX_WORKERS)
        stage = "evaluation-candidates"
        seal_candidates(output, lock, "evaluation")
        stage = "evaluation-decision-barrier"
        seal_evaluation_decisions(output, lock)
        stage = "evaluation-independent-action-futures"
        execute_many(output, lock, "evaluation", "future", workers=FUTURE_WORKERS)
        stage = "evaluation-world-qualification"
        rewards, world_ids, action_ids, all_qa = _future_rewards(
            output, lock, "evaluation", write=True
        )
        stage = "score"
        decision, decisions, barrier = load_evaluation_decisions(output, lock)
        calibration_seal, calibration, simultaneous = load_calibration(output, lock)
        candidate, guarded = _score_inputs(decisions)
        result = write_record(
            output / "result.json",
            {
                **score(
                    candidate,
                    guarded,
                    rewards,
                    calibration,
                    simultaneous,
                    all_native_qa=all_qa,
                    pre_future_gate_passed=barrier["pre_future_gate_passed"],
                ),
                "lock_id": lock["artifact_id"],
                "calibration_seal_id": calibration_seal["artifact_id"],
                "decision_seal_id": decision["artifact_id"],
                "barrier_id": barrier["artifact_id"],
                "future_action_seal_ids": action_ids,
                "world_qualification_ids": world_ids,
                "ordinary_evaluation_worlds": COUNTS["evaluation"],
                "ordinary_evaluation_action_processes": (
                    COUNTS["evaluation"] * ACTION_COUNT
                ),
                "technical_failures": 0,
                "replacements": 0,
                "retries": 0,
            },
        )
        print(
            f"Slingshot v3 policy-certificate gate={result['source_gate_passed']}; "
            f"id={result['artifact_id']}",
            flush=True,
        )
        return cast(dict[str, Any], result)
    except Exception as error:
        if output.is_dir():
            write_record(
                output / "failure.json",
                {
                    "schema": "dlolab-slingshot-policy-run-failure-v3",
                    "lock_id": None if lock is None else lock["artifact_id"],
                    "terminal_stage": stage,
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "retry_authorized": False,
                    "replacement_authorized": False,
                    "partial_score_authorized": False,
                    "protected_data_read": False,
                },
            )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-role", choices=("calibration", "evaluation"))
    parser.add_argument("--worker-kind", choices=("prefix", "future"))
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--worker-action", type=int)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    worker_values = (args.worker_role, args.worker_kind, args.worker_index)
    if args.verify_only and any(value is not None for value in worker_values):
        parser.error("verification cannot be combined with worker execution")
    if args.verify_only:
        verified = verify_result(args.output)
        print(f"verified Slingshot v3 result {verified['artifact_id']}", flush=True)
    elif all(value is not None for value in worker_values):
        worker(
            args.output,
            args.worker_role,
            args.worker_kind,
            args.worker_index,
            args.worker_action,
        )
    elif any(value is not None for value in (*worker_values, args.worker_action)):
        parser.error("all registered worker arguments are required")
    else:
        run(args.output)
