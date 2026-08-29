#!/usr/bin/env python3
"""Run the frozen wrapping posterior chance-guard study."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from bayesian_phystwin_experiments.dlolab_benchmark import (
    source_identity,
    write_native_bundle,
)
from bayesian_phystwin_experiments.dlolab_native import array_digest, file_digest
from bayesian_phystwin_experiments.dlolab_regret_artifacts import (
    clean_revision,
    read_record,
    write_record,
)
from bayesian_phystwin_experiments.dlolab_slingshot_process import load_native_bundle
from bayesian_phystwin_experiments.dlolab_wrapping_risk_guard_native_v8 import (
    run_worlds,
)
from bayesian_phystwin_experiments.dlolab_wrapping_risk_guard_v8 import (
    DEVELOPMENT_V2_RESULT_ID,
    DEVELOPMENT_V3_RESULT_ID,
    DEVELOPMENT_V4_DIAGNOSTIC_ID,
    PREFIX_BATCH_COUNT,
    WORLD_COUNT,
    continuous_worlds,
    future_native_qa,
    future_task,
    infer_decisions,
    pre_future_checks,
    prefix_native_qa,
    prefix_observation,
    prefix_task,
    protocol,
    score,
)

ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path("/home/florianpfaff/source-only/dlolab-runtime-linux-v7-assets")
PARENT = Path(
    "/home/florianpfaff/source-only/dlolab-wrapping-belief-source-v1-compact"
)
TERMINAL_V1_SUMMARY = (
    ROOT / "results/sota/dlolab_wrapping_continuous_bayes_source_v1/summary.json"
)
DEVELOPMENT_V2_SUMMARY = (
    ROOT / "results/sota/dlolab_wrapping_continuous_interp_source_v2/summary.json"
)
DEVELOPMENT_V3_SUMMARY = (
    ROOT / "results/sota/dlolab_wrapping_resolution_ensemble_source_v3/summary.json"
)
DEVELOPMENT_V4_DIAGNOSTIC = (
    ROOT / "results/sota/dlolab_wrapping_risk_guard_development_v4/summary.json"
)
TERMINAL_V4_SUMMARY = (
    ROOT / "results/sota/dlolab_wrapping_risk_guard_source_v4/summary.json"
)
RUNTIME_V7_SUMMARY = (
    ROOT
    / "results/sota/dlolab_wrapping_risk_guard_runtime_qualification_v7/summary.json"
)
OUTPUT = Path(
    "/home/florianpfaff/source-only/dlolab-wrapping-risk-guard-source-v8"
)
ATTEMPT = Path(
    "/home/florianpfaff/source-only/dlolab-wrapping-risk-guard-source-v8.attempt.json"
)
EXPECTED_PYTHON = ASSETS / "venv/bin/python"
RUNTIME_LOCK = ASSETS / "runtime-lock.txt"
OSMESA_DIRECTORY = ASSETS / "native-libs"
EXPECTED_RUNTIME_LOCK_SHA256 = (
    "80bc64e1280519404f034872e471c43809269ebbf56902a2a7b45c51da14688c"
)
EXPECTED_PYTHON_SHA256 = (
    "9e7f0dd93c77a32d07aa66631b48116101db6266701b292ebdc56a30d6cc7924"
)
EXPECTED_NATIVE_LIBRARY_TREE_ID = (
    "9e96ed5c5e969262aaa69e80e028b3893e1ee41f3a2052a034df9b062347f1f2"
)
EXPECTED_OS_RELEASE_SHA256 = (
    "3e5851448bae5b36f351becde037a8b13b77307279f484eda808f8177d9a4293"
)
PARENT_FILE_SHA256 = {
    "lock.json": "b689b17db607d79bb9b7642a5ad76a25591f7e85902ccd08ac01d7e6dc970bbc",
    "source-bank/arrays.npz": "914bd948df92e8b829ac65ca8c075c789d122a63a9ec32807a302bef16e2271d",
    "source-bank/seal.json": "143686ee40ddfb8456e23cded5c8225015e60bd678a4f77bd4074946c33fe14f",
    "result.json": "550b04bceab58d14f78f020a3870841ffae06e6ce8946d996f8418e868bacf9c",
}
PARENT_LOCK_ID = "70e6054141a5652957590f5b173c36ccff99cc167b48a3f8b4f085ba4be20a31"
PARENT_RESULT_ID = "5be8f1a54ac38e9dfc0745a5722a9490d8fa41299ca66080714caa8612a09ff0"
TERMINAL_V1_FAILURE_ID = (
    "32f1da52f18bcddc1697931b139b1222692f8eb7b9839b2997b60b9328837692"
)
TERMINAL_V1_SUMMARY_SHA256 = (
    "ec4f7b423d304cffa6cb7796d269da050513fc0364631b885c256fe4a28eba1a"
)
DEVELOPMENT_V2_SUMMARY_SHA256 = (
    "a2b367ac3e927e2218e56613ad1c0627b486fa3dd9fbcd9567e3567465590343"
)
DEVELOPMENT_V3_SUMMARY_SHA256 = (
    "569c73fdfc7edea572148747df110edaada8583d5ff6a7d0908c05b86f161936"
)
DEVELOPMENT_V4_DIAGNOSTIC_SHA256 = (
    "235523b678f96cd9893c1be4cc16e44bf46b38106569134e6424abdacdde7080"
)
TERMINAL_V4_SUMMARY_SHA256 = (
    "128a81f30ec9bc6050eec65bb04a68c04ae54092134dc12f0f103292af27c145"
)
RUNTIME_V7_SUMMARY_SHA256 = (
    "0f90622ebc02c0c33ee967e43fe0c16a62c2f801e00ba157ab91876f92e2612d"
)
NEW_SOURCES = (
    "src/bayesian_phystwin_experiments/dlolab_wrapping_risk_guard_v8.py",
    "src/bayesian_phystwin_experiments/dlolab_wrapping_risk_guard_native_v8.py",
    "scripts/remote/run_dlolab_wrapping_risk_guard_v8.py",
    "tests/test_dlolab_wrapping_risk_guard_v8.py",
    "tests/test_dlolab_wrapping_risk_guard_v8_custody.py",
    "docs/dlolab_wrapping_risk_guard_source_v8.md",
    "scripts/audit_dlolab_wrapping_risk_guard_development_v4.py",
    "results/sota/dlolab_wrapping_risk_guard_development_v4/summary.json",
    "src/bayesian_phystwin_experiments/dlolab_wrapping_resolution_ensemble_v3.py",
    "docs/dlolab_wrapping_resolution_ensemble_source_v3_result.md",
    "results/sota/dlolab_wrapping_resolution_ensemble_source_v3/summary.json",
    "src/bayesian_phystwin_experiments/dlolab_wrapping_continuous_interp_v2.py",
    "docs/dlolab_wrapping_continuous_interp_source_v2_result.md",
    "results/sota/dlolab_wrapping_continuous_interp_source_v2/summary.json",
    "src/bayesian_phystwin_experiments/dlolab_wrapping_continuous_bayes_v1.py",
    "results/sota/dlolab_wrapping_continuous_bayes_source_v1/summary.json",
    "src/bayesian_phystwin_experiments/dlolab_wrapping_risk_guard_v4.py",
    "results/sota/dlolab_wrapping_risk_guard_source_v4/summary.json",
    "results/sota/dlolab_wrapping_risk_guard_runtime_qualification_v7/summary.json",
    "src/bayesian_phystwin_experiments/dlolab_wrapping_source.py",
    "src/bayesian_phystwin_experiments/dlolab_benchmark.py",
    "src/bayesian_phystwin_experiments/dlolab_native.py",
    "src/bayesian_phystwin_experiments/dlolab_regret_artifacts.py",
    "src/bayesian_phystwin_experiments/dlolab_slingshot_process.py",
    "src/bayesian_phystwin_experiments/deform_state_restart.py",
    "src/bayesian_phystwin/_portable_contracts.py",
    "src/bayesian_phystwin/_canonical_contracts.py",
)
POSITION_FIELDS = ("rod_pos_m", "gripper_pos_m", "post_pos_m")
Array: TypeAlias = NDArray[Any]


def _native_library_tree_id() -> str:
    entries = sorted(OSMESA_DIRECTORY.iterdir())
    if len(entries) != 39 or any(path.is_dir() for path in entries):
        raise ValueError("registered native library closure changed")
    values = {
        path.name: (
            f"symlink:{path.readlink()}"
            if path.is_symlink()
            else hashlib.sha256(path.read_bytes()).hexdigest()
        )
        for path in entries
    }
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def runtime() -> dict[str, Any]:
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise ValueError("registered Python 3.11 interpreter required")
    if platform.python_version() != "3.11.15":
        raise ValueError("registered Python version changed")
    if (
        RUNTIME_LOCK.is_symlink()
        or file_digest(RUNTIME_LOCK) != EXPECTED_RUNTIME_LOCK_SHA256
        or file_digest(EXPECTED_PYTHON.resolve(strict=True)) != EXPECTED_PYTHON_SHA256
        or _native_library_tree_id() != EXPECTED_NATIVE_LIBRARY_TREE_ID
    ):
        raise ValueError("registered native-Linux runtime changed")
    expected_environment = {
        "CUDA_VISIBLE_DEVICES": "",
        "PYOPENGL_PLATFORM": "osmesa",
        "LIBGL_ALWAYS_SOFTWARE": "1",
        "LD_LIBRARY_PATH": str(OSMESA_DIRECTORY),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    actual_environment = {name: os.environ.get(name) for name in expected_environment}
    if actual_environment != expected_environment:
        raise ValueError("registered CPU/software-rendering environment required")
    os_release = Path("/etc/os-release")
    host = {
        "hostname": socket.gethostname(),
        "system": platform.system(),
        "kernel_release": platform.release(),
        "architecture": platform.machine(),
        "os_release_sha256": file_digest(os_release),
        "wsl": "microsoft" in platform.release().lower(),
    }
    if host != {
        "hostname": "workstation2",
        "system": "Linux",
        "kernel_release": "7.0.0-28-generic",
        "architecture": "x86_64",
        "os_release_sha256": EXPECTED_OS_RELEASE_SHA256,
        "wsl": False,
    }:
        raise ValueError("registered native Linux host changed")
    return {
        "python": platform.python_version(),
        "python_binary_sha256": EXPECTED_PYTHON_SHA256,
        "runtime_lock_sha256": EXPECTED_RUNTIME_LOCK_SHA256,
        "native_library_tree_id": EXPECTED_NATIVE_LIBRARY_TREE_ID,
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "genesis-world",
                "mushroom-rl",
                "numpy",
                "omegaconf",
                "pin",
                "pin-pink",
                "PyOpenGL",
                "scipy",
                "torch",
            )
        },
        "device": "cpu",
        "precision": "float64",
        "torch_threads": 1,
        "osmesa_sha256": file_digest(
            (OSMESA_DIRECTORY / "libOSMesa.so.8").resolve(strict=True)
        ),
        "environment": actual_environment,
        "host": host,
    }


def native_source() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        source_identity(
            ASSETS / "upstream",
            ASSETS / "mushroom-rl",
            ASSETS / "dlo-lab.zip",
        ),
    )


def _source_hashes() -> dict[str, str]:
    if any(not (ROOT / name).is_file() for name in NEW_SOURCES):
        raise ValueError("complete wrapping continuous source required")
    return {name: file_digest(ROOT / name) for name in NEW_SOURCES}


def _summary(path: Path, digest: str, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or file_digest(path) != digest:
        raise ValueError(f"registered {label} summary changed")
    return cast(dict[str, Any], read_record(path))


def _terminal_v1() -> tuple[dict[str, Any], dict[str, str]]:
    value = _summary(
        TERMINAL_V1_SUMMARY,
        TERMINAL_V1_SUMMARY_SHA256,
        label="terminal v1",
    )
    if (
        value.get("failure_id") != TERMINAL_V1_FAILURE_ID
        or value.get("status") != "terminal_technical_failure"
        or value.get("completed_prefix_batches") != 4
        or value.get("completed_future_worlds") != 32
        or value.get("retry_authorized") is not False
        or value.get("replacement_authorized") is not False
        or value.get("protected_data_read") is not False
    ):
        raise ValueError("terminal v1 summary lineage changed")
    return value, {"summary.json": TERMINAL_V1_SUMMARY_SHA256}


def _development_v2() -> tuple[dict[str, Any], dict[str, str]]:
    result = _summary(
        DEVELOPMENT_V2_SUMMARY,
        DEVELOPMENT_V2_SUMMARY_SHA256,
        label="development v2",
    )
    if (
        result.get("result_id") != DEVELOPMENT_V2_RESULT_ID
        or result.get("status") != "complete_source_gate_failed"
        or result.get("source_gate_passed") is not False
        or result.get("ordinary_worlds") != 32
        or result.get("technical_failures") != 0
        or result.get("replacements") != 0
        or result.get("retry_authorized") is not False
        or result.get("protected_data_read") is not False
    ):
        raise ValueError("wrapping development v2 summary lineage changed")
    return result, {"summary.json": DEVELOPMENT_V2_SUMMARY_SHA256}


def _development_v3() -> tuple[dict[str, Any], dict[str, str]]:
    result = _summary(
        DEVELOPMENT_V3_SUMMARY,
        DEVELOPMENT_V3_SUMMARY_SHA256,
        label="development v3",
    )
    if (
        result.get("result_id") != DEVELOPMENT_V3_RESULT_ID
        or result.get("status") != "complete_source_gate_failed"
        or result.get("source_gate_passed") is not False
        or result.get("ordinary_worlds") != 48
        or result.get("technical_failures") != 0
        or result.get("replacements") != 0
        or result.get("retry_authorized") is not False
        or result.get("protected_data_read") is not False
    ):
        raise ValueError("wrapping development v3 summary lineage changed")
    return result, {"summary.json": DEVELOPMENT_V3_SUMMARY_SHA256}


def _development_v4_diagnostic() -> dict[str, Any]:
    if (
        not DEVELOPMENT_V4_DIAGNOSTIC.is_file()
        or DEVELOPMENT_V4_DIAGNOSTIC.is_symlink()
        or file_digest(DEVELOPMENT_V4_DIAGNOSTIC) != DEVELOPMENT_V4_DIAGNOSTIC_SHA256
    ):
        raise ValueError("registered risk-guard development diagnostic required")
    result = cast(dict[str, Any], read_record(DEVELOPMENT_V4_DIAGNOSTIC))
    selected = result.get("selected_development_metrics", {})
    if (
        result.get("artifact_id") != DEVELOPMENT_V4_DIAGNOSTIC_ID
        or result.get("status") != "post_open_development_diagnostic"
        or result.get("development_result_ids")
        != {
            "v2": DEVELOPMENT_V2_RESULT_ID,
            "v3": DEVELOPMENT_V3_RESULT_ID,
        }
        or result.get("selected_probability") != 0.975
        or selected.get("worlds") != 80
        or selected.get("worlds_harmed_beyond_numeric_margin") != 0
        or result.get("lead_is_not_prospective_evidence") is not True
        or result.get("future_experiment_automatically_authorized") is not False
        or result.get("development_v2_v3_results_reclassified") is not False
        or result.get("protected_data_read") is not False
    ):
        raise ValueError("risk-guard development diagnostic changed")
    return result


def _terminal_v4() -> dict[str, Any]:
    result = _summary(
        TERMINAL_V4_SUMMARY,
        TERMINAL_V4_SUMMARY_SHA256,
        label="terminal v4",
    )
    if (
        result.get("artifact_id")
        != "ef75f43b46654530ed8a788303feee13c36a3d448566041b42707fe898e07873"
        or result.get("failure_id")
        != "003be585e995ad8e38818cbb341fe9d39c8344d2dd8bc59d4bd6ace61945443f"
        or result.get("status") != "terminal_technical_failure"
        or result.get("ordinary_future_worlds") != 69
        or result.get("registered_future_worlds") != 72
        or result.get("task_value_scored") is not False
        or result.get("scientific_result_available") is not False
        or result.get("retry_authorized") is not False
        or result.get("replacement_authorized") is not False
        or result.get("protected_data_read") is not False
    ):
        raise ValueError("terminal v4 summary lineage changed")
    return result


def _runtime_v7() -> dict[str, Any]:
    result = _summary(
        RUNTIME_V7_SUMMARY,
        RUNTIME_V7_SUMMARY_SHA256,
        label="runtime v7",
    )
    if (
        result.get("artifact_id")
        != "24bc06374ff8e5c392304b1b3091e346172b41e1ac8a22081d1efdaa52ff611e"
        or result.get("status") != "complete"
        or result.get("constructor_successes") != 24
        or result.get("full_rollout_successes") != 4
        or result.get("qualification_passed") is not True
        or result.get("scientific_outcome_scored") is not False
        or result.get("retry_authorized") is not False
        or result.get("protected_data_read") is not False
    ):
        raise ValueError("runtime v7 qualification summary changed")
    return result


def _parent() -> tuple[dict[str, Any], dict[str, Array]]:
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise ValueError("registered benchmark interpreter required")
    if any(
        file_digest(PARENT / name) != digest
        for name, digest in PARENT_FILE_SHA256.items()
    ):
        raise ValueError("registered wrapping source evidence changed")
    lock = read_record(PARENT / "lock.json")
    seal = read_record(PARENT / "source-bank" / "seal.json")
    result = read_record(PARENT / "result.json")
    bank = load_native_bundle(PARENT / "source-bank", seal["bundle"])
    metrics = result.get("metrics", {})
    if (
        lock.get("artifact_id") != PARENT_LOCK_ID
        or result.get("artifact_id") != PARENT_RESULT_ID
        or result.get("source_gate_passed") is not False
        or result.get("status") != "complete"
        or result.get("source_bank_id") != seal.get("artifact_id")
        or metrics.get("source_gate_passed") is not False
        or metrics.get("arms", {})
        .get("bias_aware_bayes", {})
        .get("gain_over_best_fixed", 0)
        <= 0
        or bank.get("prefix", np.empty(0)).shape != (9, 3, 5, 3)
        or bank.get("reward", np.empty(0)).shape != (9, 8)
    ):
        raise ValueError("complete stopped wrapping source signal required")
    return lock, bank


def _runtime_qualification() -> tuple[dict[str, Any], dict[str, str]]:
    return _runtime_v7(), {"summary.json": RUNTIME_V7_SUMMARY_SHA256}


def _validate(output: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Array]]:
    if output.resolve() != OUTPUT or output.is_symlink():
        raise ValueError("only the registered continuous wrapping root is permitted")
    lock = read_record(output / "lock.json")
    attempt = read_record(ATTEMPT)
    parent_lock, bank = _parent()
    qualification, qualification_hashes = _runtime_qualification()
    terminal_v1, terminal_hashes = _terminal_v1()
    development_v2, development_hashes = _development_v2()
    development_v3, development_v3_hashes = _development_v3()
    development_v4 = _development_v4_diagnostic()
    terminal_v4 = _terminal_v4()
    if (
        lock.get("schema") != "dlolab-wrapping-risk-guard-lock-v8"
        or lock.get("revision") != clean_revision(ROOT)
        or lock.get("source_sha256") != _source_hashes()
        or lock.get("protocol") != protocol()
        or lock.get("output_root") != str(OUTPUT)
        or lock.get("attempt_id") != attempt.get("artifact_id")
        or attempt.get("schema") != "dlolab-wrapping-risk-guard-attempt-v8"
        or attempt.get("revision") != lock.get("revision")
        or attempt.get("source_sha256") != lock.get("source_sha256")
        or attempt.get("protocol") != lock.get("protocol")
        or attempt.get("output_root") != str(OUTPUT)
        or attempt.get("terminal_v4_partial_future_payload_read") is not False
        or attempt.get("runtime_v7_arrays_read") is not False
        or lock.get("parent_file_sha256") != PARENT_FILE_SHA256
        or lock.get("parent_lock_id") != PARENT_LOCK_ID
        or lock.get("parent_result_id") != PARENT_RESULT_ID
        or lock.get("runtime_v7_summary_id") != qualification.get("artifact_id")
        or lock.get("runtime_v7_file_sha256") != qualification_hashes
        or lock.get("source_prefix_sha256") != array_digest(bank["prefix"])
        or lock.get("source_reward_sha256") != array_digest(bank["reward"])
        or lock.get("runtime") != runtime()
        or lock.get("native_source") != native_source()
        or lock.get("terminal_v1_summary_id") != terminal_v1.get("artifact_id")
        or lock.get("terminal_v1_failure_id") != terminal_v1.get("failure_id")
        or lock.get("terminal_v1_summary_sha256") != terminal_hashes
        or lock.get("development_v2_summary_id") != development_v2.get("artifact_id")
        or lock.get("development_v2_result_id") != development_v2.get("result_id")
        or lock.get("development_v2_summary_sha256") != development_hashes
        or lock.get("development_v3_summary_id") != development_v3.get("artifact_id")
        or lock.get("development_v3_result_id") != development_v3.get("result_id")
        or lock.get("development_v3_summary_sha256") != development_v3_hashes
        or lock.get("development_v4_diagnostic_id") != development_v4.get("artifact_id")
        or lock.get("terminal_v4_summary_id") != terminal_v4.get("artifact_id")
        or lock.get("terminal_v4_failure_id") != terminal_v4.get("failure_id")
        or lock.get("terminal_v4_summary_sha256") != TERMINAL_V4_SUMMARY_SHA256
        or lock.get("terminal_v4_partial_future_payload_read") is not False
        or lock.get("terminal_v4_retried") is not False
        or lock.get("runtime_v7_arrays_read") is not False
        or any(
            record.get("retry_authorized") is not False for record in (lock, attempt)
        )
        or any(
            record.get("replacement_authorized") is not False
            for record in (lock, attempt)
        )
        or any(
            record.get("protected_data_read") is not False for record in (lock, attempt)
        )
        or parent_lock.get("artifact_id") != PARENT_LOCK_ID
    ):
        raise ValueError("clean frozen continuous wrapping lock required")
    return lock, parent_lock, bank


def _worlds_for_task(task: dict[str, Any]) -> list[dict[str, Any]]:
    roster = continuous_worlds()
    if task["kind"] == "prefix_only":
        return [roster[index] for index in task["native_world_indices"]]
    return [roster[task["world_index"]]] * 9


def _load_task(
    output: Path,
    lock: dict[str, Any],
    task: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Array], dict[str, Any]]:
    directory = output / task["name"]
    worlds_for_native = _worlds_for_task(task)
    expected_authorization: dict[str, Any] = {"gate": "prefix_only_before_futures"}
    if task["kind"] == "all_action_future":
        barrier = _require_barrier(output, lock)
        expected_authorization = {
            "gate": "all_decisions_sealed",
            "barrier_id": barrier["artifact_id"],
        }
    claim = read_record(directory / "claim.json")
    seal = read_record(directory / "seal.json")
    if (
        claim.get("schema") != "dlolab-wrapping-risk-guard-claim-v8"
        or claim.get("lock_id") != lock["artifact_id"]
        or claim.get("task") != task
        or claim.get("authorization") != expected_authorization
        or claim.get("retry_authorized") is not False
        or claim.get("replacement_authorized") is not False
        or claim.get("protected_data_read") is not False
        or seal.get("schema") != "dlolab-wrapping-risk-guard-seal-v8"
        or seal.get("lock_id") != lock["artifact_id"]
        or seal.get("claim_id") != claim["artifact_id"]
        or seal.get("task") != task
        or seal.get("protected_data_read") is not False
    ):
        raise ValueError("continuous wrapping task custody changed")
    data = load_native_bundle(directory, seal["bundle"])
    qa = (
        prefix_native_qa(data, seal["native"], worlds_for_native)
        if task["kind"] == "prefix_only"
        else future_native_qa(data, seal["native"], worlds_for_native[0])
    )
    return seal, data, qa


def _decision_contents(
    output: Path,
    lock: dict[str, Any],
    bank: dict[str, Array],
) -> tuple[dict[str, Array], list[str], list[dict[str, Any]]]:
    truth: Array = np.empty((WORLD_COUNT, 3, 5, 3), dtype=np.float64)
    ids: list[str] = []
    qas: list[dict[str, Any]] = []
    for batch in range(PREFIX_BATCH_COUNT):
        task = prefix_task(batch)
        seal, data, qa = _load_task(output, lock, task)
        count = len(task["world_indices"])
        truth[task["world_indices"]] = prefix_observation(data["rod_pos_m"])[:count]
        ids.append(seal["artifact_id"])
        qas.append(qa)
    return infer_decisions(bank["prefix"], bank["reward"], truth), ids, qas


def _load_decisions(
    output: Path,
    lock: dict[str, Any],
    bank: dict[str, Array],
) -> tuple[dict[str, Any], dict[str, Array], dict[str, Any]]:
    expected, prefix_ids, qas = _decision_contents(output, lock, bank)
    directory = output / "decisions"
    seal = read_record(directory / "seal.json")
    data = load_native_bundle(directory, seal["bundle"])
    if (
        seal.get("schema") != "dlolab-wrapping-risk-guard-decision-seal-v8"
        or seal.get("lock_id") != lock["artifact_id"]
        or seal.get("prefix_seal_ids") != prefix_ids
        or seal.get("parent_source_bank_id") != lock["parent_source_bank_id"]
        or seal.get("future_simulated") is not False
        or seal.get("future_read") is not False
        or seal.get("protected_data_read") is not False
        or set(data) != set(expected)
        or any(not np.array_equal(data[name], expected[name]) for name in expected)
    ):
        raise ValueError("sealed continuous wrapping decisions changed")
    gate = pre_future_checks(
        data["decisions"],
        data["guarded_posterior_improvement_probability"],
        all_prefix_qa=all(qa["qa_passed"] for qa in qas),
    )
    return seal, data, gate


def _barrier_contents(
    output: Path,
    lock: dict[str, Any],
) -> dict[str, Any]:
    _, _, bank = _validate(output)
    seal, _, gate = _load_decisions(output, lock, bank)
    return {
        "schema": "dlolab-wrapping-risk-guard-decision-barrier-v8",
        "lock_id": lock["artifact_id"],
        "decision_seal_id": seal["artifact_id"],
        "pre_future": gate,
        "future_simulated": False,
        "future_read": False,
        "protected_data_read": False,
    }


def _require_barrier(output: Path, lock: dict[str, Any]) -> dict[str, Any]:
    barrier: dict[str, Any] = read_record(output / "decision-barrier.json")
    expected = _barrier_contents(output, lock)
    if any(barrier.get(key) != value for key, value in expected.items()):
        raise ValueError("continuous wrapping decision barrier changed")
    if barrier["pre_future"]["pre_future_gate_passed"] is not True:
        raise ValueError("continuous wrapping pre-future gate did not pass")
    return barrier


def _worker(output: Path, kind: str, index: int) -> None:
    lock, parent_lock, _ = _validate(output)
    task = prefix_task(index) if kind == "prefix" else future_task(index)
    worlds_for_native = _worlds_for_task(task)
    authorization: dict[str, Any] = {"gate": "prefix_only_before_futures"}
    if kind == "future":
        barrier = _require_barrier(output, lock)
        authorization = {
            "gate": "all_decisions_sealed",
            "barrier_id": barrier["artifact_id"],
        }
    directory = output / task["name"]
    directory.mkdir()
    claim = write_record(
        directory / "claim.json",
        {
            "schema": "dlolab-wrapping-risk-guard-claim-v8",
            "lock_id": lock["artifact_id"],
            "task": task,
            "authorization": authorization,
            "retry_authorized": False,
            "replacement_authorized": False,
            "protected_data_read": False,
        },
    )
    try:
        data, native = run_worlds(
            ASSETS / "upstream",
            directory,
            worlds_for_native,
            prefix_only=kind == "prefix",
        )
        bundle = write_native_bundle(directory, data)
        write_record(
            directory / "seal.json",
            {
                "schema": "dlolab-wrapping-risk-guard-seal-v8",
                "lock_id": lock["artifact_id"],
                "claim_id": claim["artifact_id"],
                "task": task,
                "native": native,
                "bundle": bundle,
                "protected_data_read": False,
            },
        )
    except Exception as error:
        write_record(
            directory / "failure.json",
            {
                "schema": "dlolab-wrapping-risk-guard-failure-v8",
                "lock_id": lock["artifact_id"],
                "claim_id": claim["artifact_id"],
                "task": task,
                "error_type": type(error).__name__,
                "message": str(error),
                "retry_authorized": False,
                "replacement_authorized": False,
                "protected_data_read": False,
            },
        )
        raise


def _execute(output: Path, kind: str, index: int) -> None:
    task = prefix_task(index) if kind == "prefix" else future_task(index)
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--output",
        str(output),
        "--worker-kind",
        kind,
        "--worker-index",
        str(index),
    ]
    with (output / f"{task['name']}.log").open("x") as stream:
        run = subprocess.run(
            command,
            cwd=ROOT,
            env=os.environ.copy(),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if run.returncode:
        directory = output / task["name"]
        if (directory / "claim.json").is_file() and not (
            directory / "failure.json"
        ).exists():
            claim = read_record(directory / "claim.json")
            write_record(
                directory / "process-failure.json",
                {
                    "schema": "dlolab-wrapping-risk-guard-process-failure-v8",
                    "lock_id": claim["lock_id"],
                    "claim_id": claim["artifact_id"],
                    "task": task,
                    "returncode": run.returncode,
                    "retry_authorized": False,
                    "replacement_authorized": False,
                    "protected_data_read": False,
                },
            )
        raise RuntimeError(f"{task['name']} exited {run.returncode}; no retry")


def _load_future(
    output: Path,
    lock: dict[str, Any],
    index: int,
) -> tuple[dict[str, Any], list[float], dict[str, Any], float]:
    task = future_task(index)
    seal, data, qa = _load_task(output, lock, task)
    prefix_spec = prefix_task(index // 9)
    _, prefix_data, _ = _load_task(output, lock, prefix_spec)
    slot = index % 9
    prefix_difference = max(
        float(np.abs(prefix_data[name][:, slot] - data[name][:600, 1]).max())
        for name in POSITION_FIELDS
    )
    if prefix_difference > 0.001:
        raise ValueError("prefix-only and full-future reset mismatch")
    return (
        seal,
        [float(value) for value in qa["final_rewards"][:8]],
        qa,
        prefix_difference,
    )


def _run(output: Path) -> None:
    if (
        output.resolve() != OUTPUT
        or output.exists()
        or output.is_symlink()
        or ATTEMPT.exists()
        or ATTEMPT.is_symlink()
    ):
        raise ValueError("one fresh continuous wrapping attempt required")
    revision = clean_revision(ROOT)
    parent_lock, bank = _parent()
    qualification, qualification_hashes = _runtime_qualification()
    terminal_v1, terminal_hashes = _terminal_v1()
    development_v2, development_hashes = _development_v2()
    development_v3, development_v3_hashes = _development_v3()
    development_v4 = _development_v4_diagnostic()
    terminal_v4 = _terminal_v4()
    sources = _source_hashes()
    parent_seal = read_record(PARENT / "source-bank" / "seal.json")
    attempt = write_record(
        ATTEMPT,
        {
            "schema": "dlolab-wrapping-risk-guard-attempt-v8",
            "revision": revision,
            "source_sha256": sources,
            "protocol": protocol(),
            "output_root": str(OUTPUT),
            "retry_authorized": False,
            "replacement_authorized": False,
            "terminal_v4_partial_future_payload_read": False,
            "runtime_v7_arrays_read": False,
            "protected_data_read": False,
        },
    )
    output.mkdir()
    lock = write_record(
        output / "lock.json",
        {
            "schema": "dlolab-wrapping-risk-guard-lock-v8",
            "revision": revision,
            "source_sha256": sources,
            "protocol": protocol(),
            "output_root": str(OUTPUT),
            "attempt_id": attempt["artifact_id"],
            "parent_lock_id": PARENT_LOCK_ID,
            "parent_result_id": PARENT_RESULT_ID,
            "parent_source_bank_id": parent_seal["artifact_id"],
            "parent_file_sha256": PARENT_FILE_SHA256,
            "terminal_v1_summary_id": terminal_v1["artifact_id"],
            "terminal_v1_failure_id": terminal_v1["failure_id"],
            "terminal_v1_summary_sha256": terminal_hashes,
            "development_v2_summary_id": development_v2["artifact_id"],
            "development_v2_result_id": development_v2["result_id"],
            "development_v2_summary_sha256": development_hashes,
            "development_v3_summary_id": development_v3["artifact_id"],
            "development_v3_result_id": development_v3["result_id"],
            "development_v3_summary_sha256": development_v3_hashes,
            "development_v4_diagnostic_id": development_v4["artifact_id"],
            "terminal_v4_summary_id": terminal_v4["artifact_id"],
            "terminal_v4_failure_id": terminal_v4["failure_id"],
            "terminal_v4_summary_sha256": TERMINAL_V4_SUMMARY_SHA256,
            "terminal_v4_partial_future_payload_read": False,
            "terminal_v4_retried": False,
            "runtime_v7_summary_id": qualification["artifact_id"],
            "runtime_v7_file_sha256": qualification_hashes,
            "runtime_v7_arrays_read": False,
            "source_prefix_sha256": array_digest(bank["prefix"]),
            "source_reward_sha256": array_digest(bank["reward"]),
            "runtime": runtime(),
            "native_source": native_source(),
            "retry_authorized": False,
            "replacement_authorized": False,
            "protected_data_read": False,
        },
    )
    stage = "prefixes"
    try:
        for batch in range(PREFIX_BATCH_COUNT):
            _execute(output, "prefix", batch)
        decision_data, prefix_ids, qas = _decision_contents(output, lock, bank)
        directory = output / "decisions"
        directory.mkdir()
        decision_seal = write_record(
            directory / "seal.json",
            {
                "schema": "dlolab-wrapping-risk-guard-decision-seal-v8",
                "lock_id": lock["artifact_id"],
                "prefix_seal_ids": prefix_ids,
                "parent_source_bank_id": parent_seal["artifact_id"],
                "bundle": write_native_bundle(directory, decision_data),
                "future_simulated": False,
                "future_read": False,
                "protected_data_read": False,
            },
        )
        gate = pre_future_checks(
            decision_data["decisions"],
            decision_data["guarded_posterior_improvement_probability"],
            all_prefix_qa=all(qa["qa_passed"] for qa in qas),
        )
        barrier = write_record(
            output / "decision-barrier.json",
            {
                "schema": "dlolab-wrapping-risk-guard-decision-barrier-v8",
                "lock_id": lock["artifact_id"],
                "decision_seal_id": decision_seal["artifact_id"],
                "pre_future": gate,
                "future_simulated": False,
                "future_read": False,
                "protected_data_read": False,
            },
        )
        if not gate["pre_future_gate_passed"]:
            write_record(
                output / "result.json",
                {
                    "schema": "dlolab-wrapping-risk-guard-result-v8",
                    "status": "pre_future_gate_failed",
                    "lock_id": lock["artifact_id"],
                    "decision_seal_id": decision_seal["artifact_id"],
                    "barrier_id": barrier["artifact_id"],
                    "pre_future": gate,
                    "task_future_generated": False,
                    "source_gate_passed": False,
                    "retry_authorized": False,
                    "replacement_authorized": False,
                    "protected_data_read": False,
                },
            )
            return
        stage = "futures"
        for index in range(WORLD_COUNT):
            _execute(output, "future", index)
        rewards: list[list[float]] = []
        future_ids: list[str] = []
        future_qa: list[dict[str, Any]] = []
        prefix_match: list[float] = []
        for index in range(WORLD_COUNT):
            seal, row, qa, difference = _load_future(output, lock, index)
            rewards.append(row)
            future_ids.append(seal["artifact_id"])
            future_qa.append(qa)
            prefix_match.append(difference)
        reward = np.asarray(rewards, dtype=np.float64)
        stage = "generation"
        generation_dir = output / "generation"
        generation_dir.mkdir()
        generation = write_record(
            generation_dir / "seal.json",
            {
                "schema": "dlolab-wrapping-risk-guard-generation-v8",
                "lock_id": lock["artifact_id"],
                "barrier_id": barrier["artifact_id"],
                "future_seal_ids": future_ids,
                "native_qa": future_qa,
                "prefix_match_error_m": prefix_match,
                "bundle": write_native_bundle(generation_dir, {"reward": reward}),
                "ordinary_worlds": WORLD_COUNT,
                "technical_failures": 0,
                "replacements": 0,
                "protected_data_read": False,
            },
        )
        stage = "score"
        metrics = score(
            decision_data["decisions"],
            reward,
            all_native_qa=all(qa["qa_passed"] for qa in future_qa),
        )
        result = write_record(
            output / "result.json",
            {
                **metrics,
                "status": "complete",
                "lock_id": lock["artifact_id"],
                "decision_seal_id": decision_seal["artifact_id"],
                "barrier_id": barrier["artifact_id"],
                "generation_id": generation["artifact_id"],
                "pre_future": gate,
                "task_future_generated": True,
                "retry_authorized": False,
                "replacement_authorized": False,
                "protected_data_read": False,
            },
        )
        print(
            f"continuous wrapping gate={result['source_gate_passed']}; "
            f"id={result['artifact_id']}",
            flush=True,
        )
    except Exception as error:
        write_record(
            output / "failure.json",
            {
                "schema": "dlolab-wrapping-risk-guard-failure-v8",
                "lock_id": lock["artifact_id"],
                "terminal_stage": stage,
                "completed_prefix_batches": sum(
                    (output / prefix_task(batch)["name"] / "seal.json").is_file()
                    for batch in range(PREFIX_BATCH_COUNT)
                ),
                "completed_future_worlds": sum(
                    (output / future_task(index)["name"] / "seal.json").is_file()
                    for index in range(WORLD_COUNT)
                ),
                "error_type": type(error).__name__,
                "message": str(error),
                "retry_authorized": False,
                "replacement_authorized": False,
                "protected_data_read": False,
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--worker-kind", choices=("prefix", "future"))
    parser.add_argument("--worker-index", type=int)
    args = parser.parse_args()
    worker = args.worker_kind is not None or args.worker_index is not None
    if worker:
        if args.worker_kind is None or args.worker_index is None:
            raise ValueError("complete registered worker specification required")
        _worker(args.output, args.worker_kind, args.worker_index)
    else:
        _run(args.output)


if __name__ == "__main__":
    main()
