#!/usr/bin/env python3
"""Run the one-attempt public DLO-Lab policy-gain certificate study."""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np

from bayesian_phystwin.policy_gain_certificate import (
    LocalPolicyGainPredictor,
    PolicyGainCalibration,
    fit_local_policy_gain_predictor,
)
from bayesian_phystwin_experiments.coupled_action_regret import RegretCalibration
from bayesian_phystwin_experiments.dlolab_benchmark import write_native_bundle
from bayesian_phystwin_experiments.dlolab_native import array_digest, file_digest
from bayesian_phystwin_experiments.dlolab_regret_artifacts import (
    clean_revision,
    read_record,
    write_record,
)
from bayesian_phystwin_experiments.dlolab_slingshot_batch import (
    TRACE_NAMES,
    split_batch,
)
from bayesian_phystwin_experiments.dlolab_slingshot_belief import (
    BASELINE,
    infer,
    native_qa,
    prefix_observations,
)
from bayesian_phystwin_experiments.dlolab_slingshot_belief_native import (
    run_registered_worlds,
)
from bayesian_phystwin_experiments.dlolab_slingshot_cmaes import (
    task_metrics,
    worker_environment,
)
from bayesian_phystwin_experiments.dlolab_slingshot_policy_certificate_source_v2 import (
    CALIBRATION_RANK,
    COUNTS,
    calibrate,
    calibrate_simultaneous_guard,
    candidate_predictions,
    continuous_worlds,
    future_task,
    guarded_decisions,
    pre_future_checks,
    prefix_batch_count,
    prefix_task,
    protocol,
    score,
)
from bayesian_phystwin_experiments.dlolab_slingshot_policy_certificate_v2 import (
    NEIGHBOR_COUNT,
    combined_competence_features,
)
from bayesian_phystwin_experiments.dlolab_slingshot_process import (
    load_native_bundle,
    runtime,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(
    "/home/fpfaff/source-only/dlolab-slingshot-policy-certificate-source-v2"
)
PARENT_ROOT = Path(
    "/home/fpfaff/source-only/dlolab-benchmark-source-v1/belief-control-source-v1"
)
PARENT_LOCK_ID = "015e6d84aa68a2a4310552ef4880752b972890f02d3e09e333ff575c92b8df25"
PARENT_RESULT_ID = "9b8ff0817744392e0584c9b59936dd1b0e9331d3b0fa2d021f5a361947d32ee9"
PARENT_BANK_ID = "8ebf9c91322faf0658c84a2dcaa6895a98b1ff857e49e6714a2a2dad0c88d882"
DEVELOPMENT_SUMMARY = (
    ROOT / "results/source/dlolab_slingshot_policy_certificate_development_v2/summary.json"
)
DEVELOPMENT_ARTIFACT_ID = (
    "5b8e50986f1f7dc7785389fa840a2e0993cc8bcaa5a5c3d8095567ff4c81e682"
)
DEVELOPMENT_FILE_SHA256 = (
    "c1bb2bdaaee1fa5529e0ab85ab6a980dc41bf483eb49d6da39510d13ef383749"
)
POLICY_V1_ROOT = Path(
    "/home/fpfaff/source-only/dlolab-slingshot-policy-certificate-source-v1"
)
POLICY_V1_LOCK_ID = (
    "9401705f7d11f2acae32b4307eeff4e044aeba3e3e2a6403a568a999ee33a550"
)
POLICY_V1_CANDIDATE_ID = (
    "107cf3e04778c45d5e1417b14be5922facb5b1cb602cdb91a392097e74281ead"
)
POLICY_V1_CALIBRATION_ID = (
    "1ba8bebc54d799dc0eb11abcf940934b5d598275df272b6bf7a622fb8d35478d"
)
POLICY_V1_RESULT_ID = (
    "f0ac1753c92630bcc738db30f466f0745ec726d7aff74b99a0198e5aca6fb25b"
)
POLICY_V1_FILE_SHA256 = {
    "calibration-candidates/arrays.npz": (
        "fba91490c2041199f0022de451d8f6468d813bf8277276957116db5ce125f790"
    ),
    "calibration-candidates/seal.json": (
        "73754e062e68070e071fa9c672a37b36d3b77e316fcd730df3762b5e0708db47"
    ),
    "calibration/arrays.npz": (
        "64e92ff3983c67766bbd130d74c40eaccd18fe5689d800a300e7cdc2876d7c4d"
    ),
    "calibration/seal.json": (
        "62a0d94aca413977a5fc797b2097ce502e3779f215401a0d3f76b17eb60e8b54"
    ),
}
POLICY_V1_RESULT = (
    ROOT / "results/source/dlolab_slingshot_policy_certificate_source_v1/summary.json"
)
POLICY_V1_RESULT_FILE_SHA256 = (
    "93315c5988572c38cf638ffe7f34a7e927e16ce840f4323ff97f462dba5340e7"
)
PARENT_FILE_SHA256 = {
    "lock.json": "6dce35441588c2a5eff9c0ae08d85c8b41ff660403541dd489b8d9161bffcc8d",
    "result.json": "1df6afe4832a9c35bc65543255f5ce2c5830e6d58cfaa23d1140f8c867767e0b",
    "model-bank/arrays.npz": (
        "ef627e16490c0974d4c34fc82c16aae884fe6dd2a8dc0a80983e89b6d5e50832"
    ),
    "model-bank/seal.json": (
        "f4a9331d552fe8f9715d222327c3f5c41cd7fc81a006e0f9a2fc55dd2223a3ae"
    ),
}
FUTURE_WORKERS = 4
REFERENCE_ROLES = (("evaluation", 32), ("calibration", 19))
POSITION_FIELDS = ("rod_pos_m", "sphere_pos_m", "cube_pos_m", "gripper_pos_m")
SOURCES = (
    "src/bayesian_phystwin/policy_gain_certificate.py",
    "src/bayesian_phystwin_experiments/dlolab_slingshot_policy_certificate_v1.py",
    "src/bayesian_phystwin_experiments/dlolab_slingshot_policy_certificate_v2.py",
    "src/bayesian_phystwin_experiments/dlolab_slingshot_policy_certificate_source_v2.py",
    "scripts/remote/run_dlolab_slingshot_policy_certificate_source_v2.py",
    "scripts/remote/verify_dlolab_slingshot_policy_certificate_source_v2.py",
    "tests/test_policy_gain_certificate.py",
    "tests/test_dlolab_slingshot_policy_certificate_v1.py",
    "tests/test_dlolab_slingshot_policy_certificate_v2.py",
    "tests/test_dlolab_slingshot_policy_certificate_development_v2.py",
    "tests/test_dlolab_slingshot_policy_certificate_source_v2.py",
    "tests/test_dlolab_slingshot_policy_certificate_source_v2_custody.py",
    "docs/dlolab_slingshot_policy_certificate_development_v2.md",
    "docs/dlolab_slingshot_policy_certificate_source_v2.md",
    "results/source/dlolab_slingshot_policy_certificate_development_v2/summary.json",
    "results/source/dlolab_slingshot_policy_certificate_source_v1/summary.json",
    "src/bayesian_phystwin_experiments/dlolab_slingshot_belief.py",
    "src/bayesian_phystwin_experiments/dlolab_slingshot_belief_native.py",
    "src/bayesian_phystwin_experiments/dlolab_slingshot_batch.py",
    "src/bayesian_phystwin_experiments/dlolab_slingshot_process.py",
    "src/bayesian_phystwin_experiments/dlolab_benchmark.py",
    "src/bayesian_phystwin_experiments/dlolab_regret_artifacts.py",
    "src/bayesian_phystwin/guard_harm_risk.py",
    "src/bayesian_phystwin_experiments/coupled_action_regret.py",
)


def _reference_paths() -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    for role, count in REFERENCE_ROLES:
        for index in range(count):
            paths.extend(
                (
                    (
                        f"parent/{role}-predictions/case-{index:02d}/arrays.npz",
                        PARENT_ROOT
                        / f"{role}-predictions"
                        / f"case-{index:02d}"
                        / "arrays.npz",
                    ),
                    (
                        f"parent/{role}-future-{index:02d}/arrays.npz",
                        PARENT_ROOT / f"{role}-future-{index:02d}" / "arrays.npz",
                    ),
                )
            )
    paths.extend(
        (f"policy-v1/{name}", POLICY_V1_ROOT / name)
        for name in POLICY_V1_FILE_SHA256
    )
    return paths


def _reference_tree_identity() -> dict[str, Any]:
    digest = hashlib.sha256()
    byte_count = 0
    paths = _reference_paths()
    for relative, path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError("complete nonsymlinked reference tree required")
        identity = file_digest(path)
        digest.update(relative.encode() + b"\0" + identity.encode() + b"\n")
        byte_count += path.stat().st_size
    return {
        "file_count": len(paths),
        "byte_count": byte_count,
        "tree_sha256": digest.hexdigest(),
    }


def load_parent() -> tuple[
    dict[str, Any], dict[str, np.ndarray], LocalPolicyGainPredictor
]:
    if PARENT_ROOT.is_symlink() or not PARENT_ROOT.is_dir():
        raise ValueError("exact frozen parent root required")
    for name, expected in PARENT_FILE_SHA256.items():
        path = PARENT_ROOT / name
        if path.is_symlink() or file_digest(path) != expected:
            raise ValueError(f"frozen parent artifact changed: {name}")
    if file_digest(DEVELOPMENT_SUMMARY) != DEVELOPMENT_FILE_SHA256:
        raise ValueError("development evidence file changed")
    if file_digest(POLICY_V1_RESULT) != POLICY_V1_RESULT_FILE_SHA256:
        raise ValueError("terminal policy-v1 result file changed")
    if POLICY_V1_ROOT.is_symlink() or not POLICY_V1_ROOT.is_dir():
        raise ValueError("exact terminal policy-v1 source root required")
    for name, expected in POLICY_V1_FILE_SHA256.items():
        path = POLICY_V1_ROOT / name
        if path.is_symlink() or not path.is_file() or file_digest(path) != expected:
            raise ValueError(f"terminal policy-v1 artifact changed: {name}")
    development = read_record(DEVELOPMENT_SUMMARY)
    policy_v1_result = read_record(POLICY_V1_RESULT)
    policy_v1_lock = read_record(POLICY_V1_ROOT / "lock.json")
    policy_v1_candidate_seal = read_record(
        POLICY_V1_ROOT / "calibration-candidates/seal.json"
    )
    policy_v1_calibration_seal = read_record(POLICY_V1_ROOT / "calibration/seal.json")
    lock = read_record(PARENT_ROOT / "lock.json")
    result = read_record(PARENT_ROOT / "result.json")
    bank_seal = read_record(PARENT_ROOT / "model-bank/seal.json")
    if (
        development.get("artifact_id") != DEVELOPMENT_ARTIFACT_ID
        or development.get("advancement_gate_passed") is not True
        or development.get("selected_model") != "combined_distance_k7"
        or development.get("prospective_coverage_claim") is not False
        or policy_v1_result.get("artifact_id") != POLICY_V1_RESULT_ID
        or policy_v1_result.get("ordinary_evaluation_futures") != 0
        or policy_v1_lock.get("artifact_id") != POLICY_V1_LOCK_ID
        or policy_v1_candidate_seal.get("artifact_id") != POLICY_V1_CANDIDATE_ID
        or policy_v1_candidate_seal.get("schema")
        != "dlolab-slingshot-policy-candidates-v1"
        or policy_v1_candidate_seal.get("lock_id") != POLICY_V1_LOCK_ID
        or policy_v1_candidate_seal.get("future_simulated") is not False
        or policy_v1_candidate_seal.get("future_read") is not False
        or policy_v1_calibration_seal.get("artifact_id")
        != POLICY_V1_CALIBRATION_ID
        or policy_v1_calibration_seal.get("schema")
        != "dlolab-slingshot-policy-calibration-v1"
        or policy_v1_calibration_seal.get("lock_id") != POLICY_V1_LOCK_ID
        or policy_v1_calibration_seal.get("candidate_seal_id")
        != POLICY_V1_CANDIDATE_ID
        or policy_v1_calibration_seal.get("all_native_qa") is not True
        or policy_v1_calibration_seal.get("evaluation_future_simulated") is not False
        or policy_v1_calibration_seal.get("evaluation_future_read") is not False
        or lock.get("artifact_id") != PARENT_LOCK_ID
        or result.get("artifact_id") != PARENT_RESULT_ID
        or result.get("source_gate_passed") is not False
        or bank_seal.get("artifact_id") != PARENT_BANK_ID
    ):
        raise ValueError("frozen parent or development identity changed")
    bank = load_native_bundle(PARENT_ROOT / "model-bank", bank_seal["bundle"])
    if set(bank) != {"prefix", "reward"}:
        raise ValueError("frozen parent model bank changed")

    ids: list[str] = []
    features: list[np.ndarray] = []
    gains: list[np.ndarray] = []
    for role, count in REFERENCE_ROLES:
        for index in range(count):
            case_id = f"parent-{role}-{index:02d}"
            with np.load(
                PARENT_ROOT
                / f"{role}-predictions"
                / f"case-{index:02d}"
                / "arrays.npz",
                allow_pickle=False,
            ) as archive:
                observation = np.array(archive["observation"], copy=True)
                inferred = {
                    name: np.array(archive[name], copy=True)
                    for name in (
                        "weights",
                        "iid_weights",
                        "expected_losses",
                        "iid_expected_losses",
                        "map_losses",
                        "nominal_losses",
                        "prior_losses",
                        "raw_upper",
                    )
                }
            with np.load(
                PARENT_ROOT / f"{role}-future-{index:02d}" / "arrays.npz",
                allow_pickle=False,
            ) as archive:
                future = {name: np.array(archive[name], copy=True) for name in archive.files}
            rewards = np.asarray(
                [task_metrics(row)["native_reward"] for row in split_batch(future, 8)[:7]],
                dtype=np.float64,
            )
            ids.append(case_id)
            features.append(combined_competence_features(observation, inferred))
            gains.append(rewards - rewards[BASELINE])

    policy_v1_candidate = load_native_bundle(
        POLICY_V1_ROOT / "calibration-candidates",
        policy_v1_candidate_seal["bundle"],
    )
    policy_v1_calibration = load_native_bundle(
        POLICY_V1_ROOT / "calibration", policy_v1_calibration_seal["bundle"]
    )
    observations = np.asarray(policy_v1_candidate.get("observation_m"))
    rewards = np.asarray(policy_v1_calibration.get("rewards"))
    if (
        observations.shape != (96, 3, 4, 3)
        or rewards.shape != (96, 7)
        or not np.all(np.isfinite(observations))
        or not np.all(np.isfinite(rewards))
    ):
        raise ValueError("complete finite policy-v1 source training rows required")
    for index, observation in enumerate(observations):
        inferred = infer(observation, bank["prefix"], bank["reward"])
        ids.append(f"policy-v1-calibration-{index:03d}")
        features.append(combined_competence_features(observation, inferred))
        gains.append(rewards[index] - rewards[index, BASELINE])

    predictor = fit_local_policy_gain_predictor(
        reference_ids=tuple(ids),
        reference_features=np.stack(features),
        reference_action_gains=np.stack(gains),
        neighbor_count=NEIGHBOR_COUNT,
    )
    identity = {
        "root": str(PARENT_ROOT.resolve()),
        "lock_id": lock["artifact_id"],
        "result_id": result["artifact_id"],
        "bank_id": bank_seal["artifact_id"],
        "development_artifact_id": development["artifact_id"],
        "development_file_sha256": DEVELOPMENT_FILE_SHA256,
        "policy_v1_result_id": policy_v1_result["artifact_id"],
        "policy_v1_result_file_sha256": POLICY_V1_RESULT_FILE_SHA256,
        "policy_v1_file_sha256": POLICY_V1_FILE_SHA256,
        "parent_file_sha256": PARENT_FILE_SHA256,
        "reference_tree": _reference_tree_identity(),
        "reference_ids": list(predictor.reference_ids),
        "assets_root": lock["assets_root"],
        "runtime": lock["screen"]["source"]["controller"]["runtime"],
        "controls": lock["controls"],
    }
    return identity, bank, predictor


def _task(role: str, kind: str, index: int) -> dict[str, Any]:
    if kind == "prefix":
        return prefix_task(role, index)
    if kind == "future":
        return future_task(role, index)
    raise ValueError("unregistered policy-certificate task")


def _task_worlds(spec: dict[str, Any]) -> list[dict[str, Any]]:
    roster = continuous_worlds(spec["role"])
    if spec["kind"] == "prefix_only":
        return [roster[index] for index in spec["native_world_indices"]]
    return [roster[spec["world_index"]]] * 8


def _expected_controls(lock: dict[str, Any], spec: dict[str, Any]) -> np.ndarray:
    bank = np.asarray(lock["controls"], dtype=np.float64)
    if bank.shape != (8, 3, 6):
        raise ValueError("frozen action bank changed")
    if spec["kind"] == "prefix_only":
        return np.repeat(bank[BASELINE : BASELINE + 1], 8, axis=0)
    return bank


def _validate_realization(
    native: dict[str, Any], expected_worlds: list[dict[str, Any]]
) -> None:
    realization = native.get("world_realization", {})
    if realization.get("bending") != [
        [world["bending_E"] for world in expected_worlds]
    ] or realization.get("stretching") != [
        [world["stretching_K"] for world in expected_worlds]
    ]:
        raise ValueError("realized material parameters changed")
    for name, y, z in (("sphere", 0.06, 0.2), ("cube", 0.23, 0.22)):
        expected = np.asarray(
            [[0.12 + world["x_offset_m"], y, z] for world in expected_worlds]
        )
        actual = np.asarray(realization.get(f"{name}_initial_position_m"))
        if actual.shape != (8, 3) or not np.allclose(
            actual, expected, rtol=0.0, atol=1e-15
        ):
            raise ValueError("realized object placement changed")


def _prefix_qa(
    arrays: dict[str, np.ndarray],
    native: dict[str, Any],
    expected_controls: np.ndarray,
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
    _validate_realization(native, expected_worlds)
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


def _task_qa(
    arrays: dict[str, np.ndarray],
    native: dict[str, Any],
    expected_controls: np.ndarray,
    expected_worlds: list[dict[str, Any]],
    spec: dict[str, Any],
) -> dict[str, Any]:
    _validate_realization(native, expected_worlds)
    if spec["kind"] == "prefix_only":
        return _prefix_qa(arrays, native, expected_controls, expected_worlds)
    return cast(dict[str, Any], native_qa(arrays, native, expected_controls))


def freeze(output: Path) -> dict[str, Any]:
    if output.resolve() != OUTPUT_ROOT:
        raise ValueError("only the registered one-attempt root is authorized")
    revision = clean_revision(ROOT)
    parent, _, _ = load_parent()
    if runtime() != parent["runtime"]:
        raise ValueError("exact parent-qualified runtime required")
    output.mkdir(parents=True, exist_ok=False)
    return cast(
        dict[str, Any],
        write_record(
            output / "lock.json",
            {
                "schema": "dlolab-slingshot-policy-certificate-lock-v2",
                "source_revision": revision,
                "source_sha256": {name: file_digest(ROOT / name) for name in SOURCES},
                "protocol": protocol(),
                "parent": parent,
                "controls": parent["controls"],
                "assets_root": parent["assets_root"],
                "runtime": parent["runtime"],
                "future_workers": FUTURE_WORKERS,
                "output_root": str(output.resolve()),
                "protected_data_read": False,
            },
        ),
    )


def validate_lock(output: Path) -> dict[str, Any]:
    lock = read_record(output / "lock.json")
    parent, _, _ = load_parent()
    if (
        lock.get("schema") != "dlolab-slingshot-policy-certificate-lock-v2"
        or output.resolve() != OUTPUT_ROOT
        or lock.get("output_root") != str(output.resolve())
        or lock.get("source_revision") != clean_revision(ROOT)
        or lock.get("source_sha256")
        != {name: file_digest(ROOT / name) for name in SOURCES}
        or lock.get("protocol") != protocol()
        or lock.get("parent") != parent
        or lock.get("controls") != parent["controls"]
        or lock.get("assets_root") != parent["assets_root"]
        or lock.get("runtime") != runtime()
        or lock.get("future_workers") != FUTURE_WORKERS
    ):
        raise ValueError("frozen policy-certificate lock changed")
    return cast(dict[str, Any], lock)


def _candidate_seal(output: Path, lock: dict[str, Any], role: str) -> dict[str, Any]:
    seal = read_record(output / f"{role}-candidates/seal.json")
    if (
        seal.get("schema") != "dlolab-slingshot-policy-candidates-v2"
        or seal.get("lock_id") != lock["artifact_id"]
        or seal.get("role") != role
        or seal.get("future_simulated") is not False
        or seal.get("future_read") is not False
    ):
        raise ValueError("complete sealed candidates required")
    return cast(dict[str, Any], seal)


def _calibration_seal(output: Path, lock: dict[str, Any]) -> dict[str, Any]:
    seal = read_record(output / "calibration/seal.json")
    if (
        seal.get("schema") != "dlolab-slingshot-policy-calibration-v2"
        or seal.get("lock_id") != lock["artifact_id"]
        or seal.get("policy_calibration", {}).get("calibration_count")
        != COUNTS["calibration"]
        or seal.get("policy_calibration", {}).get("rank") != CALIBRATION_RANK
        or seal.get("simultaneous_calibration", {}).get("count")
        != COUNTS["calibration"]
        or seal.get("simultaneous_calibration", {}).get("rank") != CALIBRATION_RANK
    ):
        raise ValueError("complete registered calibration required")
    return cast(dict[str, Any], seal)


def _evaluation_barrier(
    output: Path, lock: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    decision = read_record(output / "evaluation-decisions/seal.json")
    barrier = read_record(output / "evaluation-decision-barrier.json")
    if (
        decision.get("schema") != "dlolab-slingshot-policy-decisions-v2"
        or decision.get("lock_id") != lock["artifact_id"]
        or barrier.get("schema") != "dlolab-slingshot-policy-barrier-v2"
        or barrier.get("lock_id") != lock["artifact_id"]
        or barrier.get("decision_seal_id") != decision["artifact_id"]
        or barrier.get("pre_future_gate_passed") is not True
        or barrier.get("future_simulated") is not False
        or barrier.get("future_read") is not False
    ):
        raise ValueError("passing evaluation decision barrier required")
    return decision, barrier


def worker(output: Path, role: str, kind: str, index: int) -> None:
    lock = validate_lock(output)
    spec = _task(role, kind, index)
    authorization: dict[str, Any] = {"gate": "registered_causal_prefix"}
    if spec["kind"] == "all_action_future" and role == "calibration":
        candidate, _ = load_candidates(output, lock, role)
        authorization = {
            "gate": "reproduced_calibration_candidates",
            "candidate_seal_id": candidate["artifact_id"],
        }
    elif spec["kind"] == "all_action_future":
        decision, _, barrier = load_evaluation_decisions(output, lock)
        authorization = {
            "gate": "reproduced_passing_evaluation_decision_barrier",
            "decision_seal_id": decision["artifact_id"],
            "barrier_id": barrier["artifact_id"],
        }
    directory = output / spec["name"]
    directory.mkdir(exist_ok=False)
    claim = write_record(
        directory / "claim.json",
        {
            "schema": "dlolab-slingshot-policy-task-claim-v2",
            "lock_id": lock["artifact_id"],
            "task": spec,
            "authorization": authorization,
            "retry_authorized": False,
            "replacement_authorized": False,
            "protected_data_read": False,
        },
    )
    try:
        expected_controls = _expected_controls(lock, spec)
        expected_worlds = _task_worlds(spec)
        arrays, native = run_registered_worlds(
            Path(lock["assets_root"]) / "upstream",
            directory,
            expected_controls,
            expected_worlds,
            prefix_only=spec["kind"] == "prefix_only",
        )
        qa = _task_qa(arrays, native, expected_controls, expected_worlds, spec)
        if not qa["qa_passed"]:
            raise ValueError("native task QA failed")
        bundle = write_native_bundle(directory, arrays)
        write_record(
            directory / "seal.json",
            {
                "schema": "dlolab-slingshot-policy-task-seal-v2",
                "lock_id": lock["artifact_id"],
                "claim_id": claim["artifact_id"],
                "task": spec,
                "native": native,
                "qa": qa,
                "bundle": bundle,
            },
        )
    except Exception as error:
        write_record(
            directory / "failure.json",
            {
                "schema": "dlolab-slingshot-policy-task-failure-v2",
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


def execute(
    output: Path, lock: dict[str, Any], role: str, kind: str, index: int
) -> None:
    spec = _task(role, kind, index)
    print(f"native policy-certificate stage: {spec['name']}", flush=True)
    with (output / f"{spec['name']}.log").open("x") as stream:
        subprocess.run(
            [
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
            ],
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=worker_environment(lock["runtime"]),
            check=True,
        )


def execute_many(
    output: Path,
    lock: dict[str, Any],
    role: str,
    kind: str,
    count: int,
    *,
    workers: int,
) -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(execute, output, lock, role, kind, index)
            for index in range(count)
        ]
        for future in futures:
            future.result()


def load_task(
    output: Path, lock: dict[str, Any], role: str, kind: str, index: int
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    seal = _task_records(output, lock, role, kind, index)
    spec = _task(role, kind, index)
    arrays = load_native_bundle(output / spec["name"], seal["bundle"])
    qa = _task_qa(
        arrays,
        seal["native"],
        _expected_controls(lock, spec),
        _task_worlds(spec),
        spec,
    )
    if not qa["qa_passed"] or seal.get("qa") != qa:
        raise ValueError("native task QA changed")
    return seal, arrays


def _task_records(
    output: Path, lock: dict[str, Any], role: str, kind: str, index: int
) -> dict[str, Any]:
    spec = _task(role, kind, index)
    directory = output / spec["name"]
    claim = read_record(directory / "claim.json")
    seal = read_record(directory / "seal.json")
    if (
        claim.get("schema") != "dlolab-slingshot-policy-task-claim-v2"
        or seal.get("schema") != "dlolab-slingshot-policy-task-seal-v2"
        or claim.get("lock_id") != lock["artifact_id"]
        or seal.get("lock_id") != lock["artifact_id"]
        or claim.get("task") != spec
        or seal.get("task") != spec
        or seal.get("claim_id") != claim["artifact_id"]
        or claim.get("retry_authorized") is not False
        or claim.get("replacement_authorized") is not False
        or claim.get("protected_data_read") is not False
        or seal.get("qa", {}).get("qa_passed") is not True
    ):
        raise ValueError("native task custody changed")
    return cast(dict[str, Any], seal)


def _candidate_artifact(
    output: Path, lock: dict[str, Any], role: str
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    parent, bank, predictor = load_parent()
    truth: list[np.ndarray] = []
    prefix_ids: list[str] = []
    for batch in range(prefix_batch_count(role)):
        seal, arrays = load_task(output, lock, role, "prefix", batch)
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
        "schema": "dlolab-slingshot-policy-candidates-v2",
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
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    seal = _candidate_seal(output, lock, role)
    arrays = load_native_bundle(output / f"{role}-candidates", seal["bundle"])
    metadata, expected = _candidate_artifact(output, lock, role)
    if any(seal.get(key) != value for key, value in metadata.items()) or (
        set(arrays) != set(expected)
        or any(array_digest(arrays[name]) != array_digest(expected[name]) for name in arrays)
    ):
        raise ValueError("sealed candidate predictions do not reproduce")
    return seal, arrays


def _future_rewards(
    output: Path, lock: dict[str, Any], role: str
) -> tuple[np.ndarray, list[str], bool]:
    rewards: list[np.ndarray] = []
    seal_ids: list[str] = []
    all_qa = True
    for index in range(COUNTS[role]):
        seal, arrays = load_task(output, lock, role, "future", index)
        seal_ids.append(seal["artifact_id"])
        all_qa = all_qa and bool(seal["qa"]["qa_passed"])
        rewards.append(
            np.asarray(
                [task_metrics(row)["native_reward"] for row in split_batch(arrays, 8)[:7]],
                dtype=np.float64,
            )
        )
    return np.stack(rewards), seal_ids, all_qa


def _calibration_artifact(
    output: Path, lock: dict[str, Any]
) -> tuple[
    dict[str, Any],
    dict[str, np.ndarray],
    PolicyGainCalibration,
    RegretCalibration,
]:
    candidate_seal, candidate = load_candidates(output, lock, "calibration")
    rewards, future_ids, all_qa = _future_rewards(output, lock, "calibration")
    calibration, realized = calibrate(candidate, rewards)
    simultaneous = calibrate_simultaneous_guard(candidate, rewards)
    metadata = {
        "schema": "dlolab-slingshot-policy-calibration-v2",
        "lock_id": lock["artifact_id"],
        "candidate_seal_id": candidate_seal["artifact_id"],
        "future_seal_ids": future_ids,
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
    metadata, arrays, _, _ = _calibration_artifact(output, lock)
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
    candidate_seal, candidate = load_candidates(output, lock, "calibration")
    if set(arrays) != {"rewards", "realized_candidate_gain"}:
        raise ValueError("calibration array members changed")
    calibration, realized = calibrate(candidate, arrays["rewards"])
    simultaneous = calibrate_simultaneous_guard(candidate, arrays["rewards"])
    future_ids = [
        _task_records(output, lock, "calibration", "future", index)["artifact_id"]
        for index in range(COUNTS["calibration"])
    ]
    metadata = {
        "schema": "dlolab-slingshot-policy-calibration-v2",
        "lock_id": lock["artifact_id"],
        "candidate_seal_id": candidate_seal["artifact_id"],
        "future_seal_ids": future_ids,
        "all_native_qa": True,
        "policy_calibration": dataclasses.asdict(calibration),
        "simultaneous_calibration": dataclasses.asdict(simultaneous),
        "evaluation_prefix_read": False,
        "evaluation_future_simulated": False,
        "evaluation_future_read": False,
        "protected_data_read": False,
    }
    if (
        any(seal.get(key) != value for key, value in metadata.items())
        or array_digest(arrays["realized_candidate_gain"]) != array_digest(realized)
    ):
        raise ValueError("sealed calibration does not reproduce")
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
            "schema": "dlolab-slingshot-policy-decisions-v2",
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
            "schema": "dlolab-slingshot-policy-barrier-v2",
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
        raise ValueError("registered evaluation pre-future gate failed")
    return cast(dict[str, Any], barrier)


def load_evaluation_decisions(
    output: Path, lock: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    decision, barrier = _evaluation_barrier(output, lock)
    arrays = load_native_bundle(output / "evaluation-decisions", decision["bundle"])
    _, candidate = load_candidates(output, lock, "evaluation")
    _, calibration, simultaneous = load_calibration(output, lock)
    expected_guarded = guarded_decisions(candidate, calibration, simultaneous)
    expected = {**candidate, **expected_guarded}
    if set(arrays) != set(expected) or any(
        array_digest(arrays[name]) != array_digest(expected[name]) for name in arrays
    ):
        raise ValueError("evaluation decisions do not reproduce")
    preflight = pre_future_checks(expected_guarded, all_prefix_qa=True)
    if barrier.get("pre_future") != preflight:
        raise ValueError("evaluation pre-future checks changed")
    return decision, arrays, barrier


def _score_inputs(
    decisions: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
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
    """Recompute the complete result from the frozen write-once artifacts."""

    lock = validate_lock(output)
    if (output / "failure.json").exists():
        raise ValueError("retained run failure cannot verify as an ordinary result")
    result = read_record(output / "result.json")
    decision_seal, decisions, barrier = load_evaluation_decisions(output, lock)
    calibration_seal, calibration, simultaneous = load_calibration(output, lock)
    candidate, guarded = _score_inputs(decisions)
    rewards, future_ids, all_qa = _future_rewards(output, lock, "evaluation")
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
        "decision_seal_id": decision_seal["artifact_id"],
        "barrier_id": barrier["artifact_id"],
        "future_seal_ids": future_ids,
        "ordinary_evaluation_worlds": COUNTS["evaluation"],
        "technical_failures": 0,
        "replacements": 0,
    }
    observed = {key: value for key, value in result.items() if key != "artifact_id"}
    if observed != expected:
        raise ValueError("policy-certificate result does not reproduce")
    return cast(dict[str, Any], result)


def run(output: Path) -> dict[str, Any]:
    stage = "freeze"
    lock: dict[str, Any] | None = None
    try:
        lock = freeze(output)
        stage = "calibration-prefixes"
        execute_many(
            output,
            lock,
            "calibration",
            "prefix",
            prefix_batch_count("calibration"),
            workers=1,
        )
        stage = "calibration-candidates"
        seal_candidates(output, lock, "calibration")
        stage = "calibration-futures"
        execute_many(
            output,
            lock,
            "calibration",
            "future",
            COUNTS["calibration"],
            workers=FUTURE_WORKERS,
        )
        stage = "calibration"
        seal_calibration(output, lock)
        stage = "evaluation-prefixes"
        execute_many(
            output,
            lock,
            "evaluation",
            "prefix",
            prefix_batch_count("evaluation"),
            workers=1,
        )
        stage = "evaluation-candidates"
        seal_candidates(output, lock, "evaluation")
        stage = "evaluation-decision-barrier"
        seal_evaluation_decisions(output, lock)
        stage = "evaluation-futures"
        execute_many(
            output,
            lock,
            "evaluation",
            "future",
            COUNTS["evaluation"],
            workers=FUTURE_WORKERS,
        )
        stage = "score"
        decision_seal, decisions, barrier = load_evaluation_decisions(output, lock)
        calibration_seal, calibration, simultaneous = load_calibration(output, lock)
        candidate, guarded = _score_inputs(decisions)
        rewards, future_ids, all_qa = _future_rewards(output, lock, "evaluation")
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
                "decision_seal_id": decision_seal["artifact_id"],
                "barrier_id": barrier["artifact_id"],
                "future_seal_ids": future_ids,
                "ordinary_evaluation_worlds": COUNTS["evaluation"],
                "technical_failures": 0,
                "replacements": 0,
            },
        )
        print(
            f"Slingshot policy-certificate gate={result['source_gate_passed']}; "
            f"id={result['artifact_id']}",
            flush=True,
        )
        return cast(dict[str, Any], result)
    except Exception as error:
        if output.is_dir():
            write_record(
                output / "failure.json",
                {
                    "schema": "dlolab-slingshot-policy-run-failure-v2",
                    "lock_id": None if lock is None else lock["artifact_id"],
                    "terminal_stage": stage,
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "retry_authorized": False,
                    "replacement_authorized": False,
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
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    worker_values = (args.worker_role, args.worker_kind, args.worker_index)
    if args.verify_only and any(value is not None for value in worker_values):
        parser.error("verification cannot be combined with worker execution")
    if args.verify_only:
        verified = verify_result(args.output)
        print(
            f"verified Slingshot policy-certificate result {verified['artifact_id']}",
            flush=True,
        )
    elif all(value is not None for value in worker_values):
        worker(args.output, args.worker_role, args.worker_kind, args.worker_index)
    elif any(value is not None for value in worker_values):
        parser.error("all registered worker arguments are required")
    else:
        run(args.output)
