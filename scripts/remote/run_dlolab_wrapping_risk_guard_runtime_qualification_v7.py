#!/usr/bin/env python3
"""Run the write-once Python 3.11 qualification for the wrapping lineage."""

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

from bayesian_phystwin._portable_contracts import (
    content_id,
    load_strict_json_object,
)
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
from bayesian_phystwin_experiments.dlolab_slingshot_process import (
    load_native_bundle,
)
from bayesian_phystwin_experiments.dlolab_wrapping_risk_guard_native_v6 import (
    run_constructor_probe,
    run_worlds,
)
from bayesian_phystwin_experiments.dlolab_wrapping_risk_guard_v4 import (
    future_native_qa,
    preflight_world,
)

ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = (
    ROOT / "configs/sota/dlolab_wrapping_risk_guard_runtime_qualification_v7.json"
)
ASSETS = Path("/home/florianpfaff/source-only/dlolab-runtime-linux-v7-assets")
RUNTIME_ROOT = ASSETS
EXPECTED_PYTHON = RUNTIME_ROOT / "venv/bin/python"
RUNTIME_LOCK = RUNTIME_ROOT / "runtime-lock.txt"
OUTPUT = Path(
    "/home/florianpfaff/source-only/dlolab-wrapping-risk-guard-runtime-qualification-v7"
)
ATTEMPT = Path(
    "/home/florianpfaff/source-only/"
    "dlolab-wrapping-risk-guard-runtime-qualification-v7.attempt.json"
)
OSMESA_DIRECTORY = RUNTIME_ROOT / "native-libs"
V4_SUMMARY = ROOT / "results/sota/dlolab_wrapping_risk_guard_source_v4/summary.json"
V5_SUMMARY = (
    ROOT
    / "results/sota/dlolab_wrapping_risk_guard_runtime_qualification_v5/summary.json"
)
V6_SUMMARY = (
    ROOT
    / "results/sota/dlolab_wrapping_risk_guard_runtime_qualification_v6/summary.json"
)
EXPECTED_RUNTIME_LOCK_SHA256 = (
    "80bc64e1280519404f034872e471c43809269ebbf56902a2a7b45c51da14688c"
)
EXPECTED_PYTHON_SHA256 = (
    "9e7f0dd93c77a32d07aa66631b48116101db6266701b292ebdc56a30d6cc7924"
)
EXPECTED_V4_SUMMARY_SHA256 = (
    "128a81f30ec9bc6050eec65bb04a68c04ae54092134dc12f0f103292af27c145"
)
EXPECTED_V4_SUMMARY_ID = (
    "ef75f43b46654530ed8a788303feee13c36a3d448566041b42707fe898e07873"
)
EXPECTED_V4_FAILURE_ID = (
    "003be585e995ad8e38818cbb341fe9d39c8344d2dd8bc59d4bd6ace61945443f"
)
EXPECTED_V5_SUMMARY_SHA256 = (
    "20cc80d5c3adf4f3ac26fef964563d28194c2f812aa08b4ed7936496bae9c871"
)
EXPECTED_V5_SUMMARY_ID = (
    "9fbd92685e9417057f28c5165862dd000ef27ab35106854aaa93589910d3e7b0"
)
EXPECTED_V5_FAILURE_ID = (
    "a83bfdece3cc2322a97a940ea74cd3232752bcb1ae7a688f15828141abccffab"
)
EXPECTED_V6_SUMMARY_SHA256 = (
    "419d6640eb96b0b698289516eb1c988b618b55b4002c3e465a683ce5a5b1a7ec"
)
EXPECTED_V6_SUMMARY_ID = (
    "1e380f455e780988a3d9542c5e313a13dd0a47ca2c385f636722daa696932965"
)
EXPECTED_V6_FAILURE_ID = (
    "0b2c9a8afacc8e4e309baa5683fc1b21d9e7ca9578cbd50a01264bd29af7df29"
)
EXPECTED_NATIVE_LIBRARY_TREE_ID = (
    "9e96ed5c5e969262aaa69e80e028b3893e1ee41f3a2052a034df9b062347f1f2"
)
EXPECTED_OS_RELEASE_SHA256 = (
    "3e5851448bae5b36f351becde037a8b13b77307279f484eda808f8177d9a4293"
)
CONSTRUCTOR_COUNT = 24
FULL_ROLLOUT_COUNT = 4
SOURCE_PATHS = (
    "configs/sota/dlolab_wrapping_risk_guard_runtime_qualification_v7.json",
    "src/bayesian_phystwin_experiments/dlolab_wrapping_risk_guard_native_v6.py",
    "scripts/remote/run_dlolab_wrapping_risk_guard_runtime_qualification_v7.py",
    "tests/test_dlolab_wrapping_risk_guard_runtime_qualification_v7.py",
    "docs/dlolab_wrapping_risk_guard_runtime_qualification_v7.md",
    "src/bayesian_phystwin_experiments/dlolab_wrapping_risk_guard_v4.py",
    "src/bayesian_phystwin_experiments/dlolab_wrapping_source.py",
    "src/bayesian_phystwin_experiments/dlolab_benchmark.py",
    "src/bayesian_phystwin_experiments/dlolab_native.py",
    "src/bayesian_phystwin_experiments/dlolab_regret_artifacts.py",
    "src/bayesian_phystwin_experiments/dlolab_slingshot_process.py",
    "src/bayesian_phystwin_experiments/deform_state_restart.py",
    "src/bayesian_phystwin/_portable_contracts.py",
    "src/bayesian_phystwin/_canonical_contracts.py",
    "results/sota/dlolab_wrapping_risk_guard_source_v4/summary.json",
    "results/sota/dlolab_wrapping_risk_guard_runtime_qualification_v5/summary.json",
    "results/sota/dlolab_wrapping_risk_guard_runtime_qualification_v6/summary.json",
)
Array: TypeAlias = NDArray[Any]


def plan() -> dict[str, Any]:
    value = dict(load_strict_json_object(PLAN_PATH, label="runtime v7 plan"))
    workload = value.get("native_workload", {})
    gate = value.get("gate", {})
    runtime_spec = value.get("runtime", {})
    host = value.get("host", {})
    lineage = value.get("lineage", {})
    output = value.get("output", {})
    boundaries = value.get("boundaries", {})
    expected_boundaries = {
        "fresh_scientific_worlds_defined": False,
        "physical_execution": False,
        "protected_data_read": False,
        "scientific_outcome_scored": False,
        "v4_partial_future_artifacts_read": False,
        "v4_retry": False,
        "v5_runtime_artifacts_read": False,
        "v5_retry": False,
        "v6_runtime_artifacts_read": False,
        "v6_retry": False,
    }
    if (
        value.get("schema") != "dlolab-wrapping-risk-guard-runtime-qualification-v7"
        or workload.get("constructor_processes") != CONSTRUCTOR_COUNT
        or workload.get("full_rollout_processes") != FULL_ROLLOUT_COUNT
        or workload.get("worlds_per_process") != 9
        or workload.get("constructor_stops_after_init_cmaes_env") is not True
        or workload.get("constructor_requires_deferred_material_randomization")
        is not True
        or workload.get("full_rollout_requires_exact_post_reset_material_realization")
        is not True
        or workload.get("full_macro_steps") != 11
        or workload.get("micro_steps_per_macro") != 200
        or workload.get("preflight_world") != preflight_world()
        or gate.get("constructor_successes_required") != CONSTRUCTOR_COUNT
        or gate.get("full_rollout_successes_required") != FULL_ROLLOUT_COUNT
        or gate.get("separate_process_per_probe") is not True
        or gate.get("retry_authorized") is not False
        or gate.get("replacement_authorized") is not False
        or gate.get("study_automatically_authorized") is not False
        or runtime_spec.get("python_version") != "3.11.15"
        or runtime_spec.get("python_binary") != str(EXPECTED_PYTHON)
        or runtime_spec.get("python_binary_sha256") != EXPECTED_PYTHON_SHA256
        or runtime_spec.get("runtime_lock") != str(RUNTIME_LOCK)
        or runtime_spec.get("runtime_lock_sha256") != EXPECTED_RUNTIME_LOCK_SHA256
        or runtime_spec.get("native_library_tree_id")
        != EXPECTED_NATIVE_LIBRARY_TREE_ID
        or runtime_spec.get("device") != "cpu"
        or runtime_spec.get("precision") != "float64"
        or runtime_spec.get("software_rendering") != "osmesa"
        or output.get("root") != str(OUTPUT)
        or output.get("attempt_ledger") != str(ATTEMPT)
        or host
        != {
            "architecture": "x86_64",
            "hostname": "workstation2",
            "kernel_release": "7.0.0-28-generic",
            "native_linux": True,
            "os_release_sha256": EXPECTED_OS_RELEASE_SHA256,
            "wsl": False,
        }
        or lineage.get("v4_summary_sha256") != EXPECTED_V4_SUMMARY_SHA256
        or lineage.get("v4_summary_artifact_id") != EXPECTED_V4_SUMMARY_ID
        or lineage.get("v4_failure_id") != EXPECTED_V4_FAILURE_ID
        or lineage.get("v5_summary_sha256") != EXPECTED_V5_SUMMARY_SHA256
        or lineage.get("v5_summary_artifact_id") != EXPECTED_V5_SUMMARY_ID
        or lineage.get("v5_failure_id") != EXPECTED_V5_FAILURE_ID
        or lineage.get("v6_summary_sha256") != EXPECTED_V6_SUMMARY_SHA256
        or lineage.get("v6_summary_artifact_id") != EXPECTED_V6_SUMMARY_ID
        or lineage.get("v6_failure_id") != EXPECTED_V6_FAILURE_ID
        or boundaries != expected_boundaries
    ):
        raise ValueError("registered runtime v7 qualification plan changed")
    return cast(dict[str, Any], value)


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


def host_identity() -> dict[str, Any]:
    os_release = Path("/etc/os-release")
    value = {
        "hostname": socket.gethostname(),
        "system": platform.system(),
        "kernel_release": platform.release(),
        "architecture": platform.machine(),
        "os_release_sha256": file_digest(os_release),
        "wsl": "microsoft" in platform.release().lower(),
    }
    if value != {
        "hostname": "workstation2",
        "system": "Linux",
        "kernel_release": "7.0.0-28-generic",
        "architecture": "x86_64",
        "os_release_sha256": EXPECTED_OS_RELEASE_SHA256,
        "wsl": False,
    }:
        raise ValueError("registered native Linux host changed")
    return value


def runtime_identity_v7() -> dict[str, Any]:
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise ValueError("registered Python 3.11 interpreter required")
    if platform.python_version() != "3.11.15":
        raise ValueError("registered Python version changed")
    if (
        RUNTIME_LOCK.is_symlink()
        or file_digest(RUNTIME_LOCK) != EXPECTED_RUNTIME_LOCK_SHA256
        or file_digest(EXPECTED_PYTHON.resolve(strict=True)) != EXPECTED_PYTHON_SHA256
    ):
        raise ValueError("registered Python 3.11 runtime changed")
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
    library = OSMESA_DIRECTORY / "libOSMesa.so.8"
    resolved_library = library.resolve(strict=True)
    if not resolved_library.is_file() or not resolved_library.is_relative_to(
        OSMESA_DIRECTORY.resolve(strict=True)
    ):
        raise ValueError("registered OSMesa library missing")
    return {
        "python": platform.python_version(),
        "python_binary_sha256": EXPECTED_PYTHON_SHA256,
        "runtime_lock_sha256": EXPECTED_RUNTIME_LOCK_SHA256,
        "native_library_tree_id": _native_library_tree_id(),
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
        "osmesa_sha256": file_digest(resolved_library),
        "environment": actual_environment,
        "host": host_identity(),
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
    if any(not (ROOT / name).is_file() for name in SOURCE_PATHS):
        raise ValueError("complete runtime v7 qualification source required")
    return {name: file_digest(ROOT / name) for name in SOURCE_PATHS}


def _v4_failure_lineage() -> dict[str, Any]:
    if V4_SUMMARY.is_symlink() or file_digest(V4_SUMMARY) != EXPECTED_V4_SUMMARY_SHA256:
        raise ValueError("v4 terminal summary changed")
    value = read_record(V4_SUMMARY)
    if (
        value.get("artifact_id") != EXPECTED_V4_SUMMARY_ID
        or value.get("failure_id") != EXPECTED_V4_FAILURE_ID
        or value.get("status") != "terminal_technical_failure"
        or value.get("scientific_result_available") is not False
        or value.get("task_value_scored") is not False
        or value.get("retry_authorized") is not False
        or value.get("replacement_authorized") is not False
        or value.get("protected_data_read") is not False
    ):
        raise ValueError("v4 terminal failure lineage changed")
    return cast(dict[str, Any], value)


def _v5_failure_lineage() -> dict[str, Any]:
    if V5_SUMMARY.is_symlink() or file_digest(V5_SUMMARY) != EXPECTED_V5_SUMMARY_SHA256:
        raise ValueError("v5 terminal summary changed")
    value = read_record(V5_SUMMARY)
    if (
        value.get("artifact_id") != EXPECTED_V5_SUMMARY_ID
        or value.get("failure_id") != EXPECTED_V5_FAILURE_ID
        or value.get("status") != "terminal_technical_failure"
        or value.get("qualification_passed") is not False
        or value.get("scientific_outcome_scored") is not False
        or value.get("retry_authorized") is not False
        or value.get("replacement_authorized") is not False
        or value.get("protected_data_read") is not False
        or value.get("v4_partial_future_artifacts_read") is not False
    ):
        raise ValueError("v5 terminal failure lineage changed")
    return cast(dict[str, Any], value)


def _v6_failure_lineage() -> dict[str, Any]:
    if V6_SUMMARY.is_symlink() or file_digest(V6_SUMMARY) != EXPECTED_V6_SUMMARY_SHA256:
        raise ValueError("v6 terminal summary changed")
    value = read_record(V6_SUMMARY)
    if (
        value.get("artifact_id") != EXPECTED_V6_SUMMARY_ID
        or value.get("failure_id") != EXPECTED_V6_FAILURE_ID
        or value.get("status") != "terminal_technical_failure"
        or value.get("constructor_successes") != 22
        or value.get("full_rollout_successes") != 0
        or value.get("qualification_passed") is not False
        or value.get("scientific_outcome_scored") is not False
        or value.get("retry_authorized") is not False
        or value.get("replacement_authorized") is not False
        or value.get("protected_data_read") is not False
        or value.get("v5_runtime_artifacts_read") is not False
    ):
        raise ValueError("v6 terminal failure lineage changed")
    return cast(dict[str, Any], value)


def _task(kind: str, index: int) -> dict[str, Any]:
    count = CONSTRUCTOR_COUNT if kind == "constructor" else FULL_ROLLOUT_COUNT
    if (
        kind not in {"constructor", "full"}
        or type(index) is not int
        or index not in range(count)
    ):
        raise ValueError("registered runtime qualification task required")
    prefix = "constructor" if kind == "constructor" else "full-rollout"
    return {
        "kind": kind,
        "index": index,
        "name": f"{prefix}-{index:02d}",
        "worlds": [preflight_world()] * 9,
    }


def _validate_output(output: Path) -> None:
    if output.resolve() != OUTPUT or output.is_symlink():
        raise ValueError("only the registered runtime qualification root is permitted")


def _validate_lock(output: Path) -> dict[str, Any]:
    _validate_output(output)
    lock = read_record(output / "lock.json")
    attempt = read_record(ATTEMPT)
    revision = clean_revision(ROOT)
    sources = _source_hashes()
    current_plan = plan()
    current_runtime = runtime_identity_v7()
    current_native = native_source()
    v4_lineage = _v4_failure_lineage()
    v5_lineage = _v5_failure_lineage()
    v6_lineage = _v6_failure_lineage()
    expected_plan_id = content_id(current_plan)
    if (
        lock.get("schema") != "dlolab-wrapping-risk-guard-runtime-qualification-lock-v7"
        or lock.get("revision") != revision
        or lock.get("source_sha256") != sources
        or lock.get("plan") != current_plan
        or lock.get("plan_id") != expected_plan_id
        or lock.get("runtime") != current_runtime
        or lock.get("native_source") != current_native
        or lock.get("v4_summary_id") != v4_lineage["artifact_id"]
        or lock.get("v4_failure_id") != v4_lineage["failure_id"]
        or lock.get("v5_summary_id") != v5_lineage["artifact_id"]
        or lock.get("v5_failure_id") != v5_lineage["failure_id"]
        or lock.get("v6_summary_id") != v6_lineage["artifact_id"]
        or lock.get("v6_failure_id") != v6_lineage["failure_id"]
        or lock.get("output_root") != str(OUTPUT)
        or lock.get("attempt_id") != attempt.get("artifact_id")
        or lock.get("retry_authorized") is not False
        or lock.get("replacement_authorized") is not False
        or lock.get("protected_data_read") is not False
        or attempt.get("schema")
        != "dlolab-wrapping-risk-guard-runtime-qualification-attempt-v7"
        or attempt.get("revision") != revision
        or attempt.get("source_sha256") != sources
        or attempt.get("plan_id") != expected_plan_id
        or attempt.get("output_root") != str(OUTPUT)
        or attempt.get("retry_authorized") is not False
        or attempt.get("protected_data_read") is not False
    ):
        raise ValueError("clean frozen runtime qualification lock required")
    return cast(dict[str, Any], lock)


def _constructor_qa(data: dict[str, Array], native: dict[str, Any]) -> dict[str, Any]:
    required_shapes = {
        "rod_pos_m": (9, 50, 3),
        "rod_vel_m_s": (9, 50, 3),
        "post_pos_m": (9, 3, 3),
        "gripper_pos_m": (9, 2, 3),
    }
    finite = bool(data) and all(
        value.dtype.kind in "bifu" and np.isfinite(value).all()
        for value in data.values()
    )
    checks = {
        "complete_core_shapes": all(
            name in data and data[name].shape == shape
            for name, shape in required_shapes.items()
        ),
        "finite_state": finite,
        "memory_state_present": any(name.startswith("memory_") for name in data),
        "constructor_completed": native.get("constructor_completed") is True,
        "init_cmaes_env_completed": native.get("init_cmaes_env_completed") is True,
        "zero_steps": native.get("native_steps") == 0,
        "no_future": native.get("future_simulated") is False,
        "no_reward": native.get("reward_exposed") is False,
        "registered_worlds": native.get("worlds") == [preflight_world()] * 9,
        "material_randomization_deferred": native.get(
            "parameter_randomization_deferred"
        )
        is True
        and native.get("world_realization") == {},
        "state_identity": native.get("state_sha256")
        == {name: array_digest(value) for name, value in sorted(data.items())},
    }
    return {"checks": checks, "qa_passed": bool(all(checks.values()))}


def _expected_claim(lock: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "dlolab-wrapping-risk-guard-runtime-qualification-claim-v7",
        "lock_id": lock["artifact_id"],
        "task": task,
        "retry_authorized": False,
        "replacement_authorized": False,
        "protected_data_read": False,
    }


def _load_success(
    output: Path, lock: dict[str, Any], task: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    directory = output / task["name"]
    claim = read_record(directory / "claim.json")
    seal = read_record(directory / "seal.json")
    data = load_native_bundle(directory, seal["bundle"])
    qa = (
        _constructor_qa(data, seal["native"])
        if task["kind"] == "constructor"
        else future_native_qa(data, seal["native"], preflight_world())
    )
    expected_claim = _expected_claim(lock, task)
    if (
        {key: claim.get(key) for key in expected_claim} != expected_claim
        or set(claim) != {*expected_claim, "artifact_id"}
        or seal.get("schema")
        != "dlolab-wrapping-risk-guard-runtime-qualification-seal-v7"
        or seal.get("lock_id") != lock["artifact_id"]
        or seal.get("claim_id") != claim["artifact_id"]
        or seal.get("task") != task
        or seal.get("qa") != qa
        or not qa["qa_passed"]
    ):
        raise ValueError("runtime qualification task seal changed")
    return seal, qa


def _worker(output: Path, kind: str, index: int) -> None:
    lock = _validate_lock(output)
    task = _task(kind, index)
    directory = output / task["name"]
    claim = read_record(directory / "claim.json")
    expected_claim = _expected_claim(lock, task)
    if (
        {key: claim.get(key) for key in expected_claim} != expected_claim
        or set(claim) != {*expected_claim, "artifact_id"}
        or (directory / "seal.json").exists()
        or (directory / "failure.json").exists()
    ):
        raise ValueError("fresh registered runtime qualification claim required")
    try:
        worlds = [preflight_world()] * 9
        if kind == "constructor":
            data, native = run_constructor_probe(ASSETS / "upstream", directory, worlds)
            qa = _constructor_qa(data, native)
        else:
            data, native = run_worlds(
                ASSETS / "upstream", directory, worlds, prefix_only=False
            )
            qa = future_native_qa(data, native, preflight_world())
        if not qa["qa_passed"]:
            raise ValueError("runtime qualification native QA failed")
        bundle = write_native_bundle(directory, data)
        write_record(
            directory / "seal.json",
            {
                "schema": "dlolab-wrapping-risk-guard-runtime-qualification-seal-v7",
                "lock_id": lock["artifact_id"],
                "claim_id": claim["artifact_id"],
                "task": task,
                "native": native,
                "qa": qa,
                "bundle": bundle,
            },
        )
    except Exception as error:
        write_record(
            directory / "failure.json",
            {
                "schema": "dlolab-wrapping-risk-guard-runtime-qualification-task-failure-v7",
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


def _execute(output: Path, lock: dict[str, Any], kind: str, index: int) -> None:
    task = _task(kind, index)
    directory = output / task["name"]
    directory.mkdir()
    claim = write_record(directory / "claim.json", _expected_claim(lock, task))
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
        if not (directory / "failure.json").exists():
            write_record(
                directory / "process-failure.json",
                {
                    "schema": "dlolab-wrapping-risk-guard-runtime-process-failure-v7",
                    "lock_id": lock["artifact_id"],
                    "claim_id": claim["artifact_id"],
                    "task": task,
                    "returncode": run.returncode,
                    "retry_authorized": False,
                    "replacement_authorized": False,
                    "protected_data_read": False,
                },
            )
        raise RuntimeError(f"{task['name']} exited {run.returncode}; no retry")
    _load_success(output, lock, task)


def _completed(output: Path, kind: str) -> int:
    count = CONSTRUCTOR_COUNT if kind == "constructor" else FULL_ROLLOUT_COUNT
    return sum(
        (output / _task(kind, index)["name"] / "seal.json").is_file()
        for index in range(count)
    )


def _run(output: Path) -> None:
    _validate_output(output)
    if (
        output.exists()
        or output.is_symlink()
        or ATTEMPT.exists()
        or ATTEMPT.is_symlink()
    ):
        raise ValueError("one fresh runtime qualification attempt required")
    revision = clean_revision(ROOT)
    sources = _source_hashes()
    current_plan = plan()
    current_runtime = runtime_identity_v7()
    current_native = native_source()
    v4_lineage = _v4_failure_lineage()
    v5_lineage = _v5_failure_lineage()
    v6_lineage = _v6_failure_lineage()
    plan_id = content_id(current_plan)
    attempt = write_record(
        ATTEMPT,
        {
            "schema": "dlolab-wrapping-risk-guard-runtime-qualification-attempt-v7",
            "revision": revision,
            "source_sha256": sources,
            "plan_id": plan_id,
            "output_root": str(OUTPUT),
            "retry_authorized": False,
            "replacement_authorized": False,
            "protected_data_read": False,
        },
    )
    output.mkdir()
    lock = write_record(
        output / "lock.json",
        {
            "schema": "dlolab-wrapping-risk-guard-runtime-qualification-lock-v7",
            "revision": revision,
            "source_sha256": sources,
            "plan": current_plan,
            "plan_id": plan_id,
            "runtime": current_runtime,
            "native_source": current_native,
            "v4_summary_id": v4_lineage["artifact_id"],
            "v4_failure_id": v4_lineage["failure_id"],
            "v5_summary_id": v5_lineage["artifact_id"],
            "v5_failure_id": v5_lineage["failure_id"],
            "v6_summary_id": v6_lineage["artifact_id"],
            "v6_failure_id": v6_lineage["failure_id"],
            "output_root": str(OUTPUT),
            "attempt_id": attempt["artifact_id"],
            "retry_authorized": False,
            "replacement_authorized": False,
            "protected_data_read": False,
        },
    )
    stage = "constructors"
    active: dict[str, Any] | None = None
    try:
        for index in range(CONSTRUCTOR_COUNT):
            active = _task("constructor", index)
            _execute(output, lock, "constructor", index)
        stage = "full-rollouts"
        for index in range(FULL_ROLLOUT_COUNT):
            active = _task("full", index)
            _execute(output, lock, "full", index)
        constructor_records = [
            _load_success(output, lock, _task("constructor", index))
            for index in range(CONSTRUCTOR_COUNT)
        ]
        full_records = [
            _load_success(output, lock, _task("full", index))
            for index in range(FULL_ROLLOUT_COUNT)
        ]
        result = write_record(
            output / "result.json",
            {
                "schema": "dlolab-wrapping-risk-guard-runtime-qualification-result-v7",
                "status": "complete",
                "lock_id": lock["artifact_id"],
                "constructor_seal_ids": [
                    row[0]["artifact_id"] for row in constructor_records
                ],
                "full_rollout_seal_ids": [
                    row[0]["artifact_id"] for row in full_records
                ],
                "constructor_successes": len(constructor_records),
                "full_rollout_successes": len(full_records),
                "qualification_passed": True,
                "fresh_scientific_worlds_defined": False,
                "scientific_outcome_scored": False,
                "study_automatically_authorized": False,
                "retry_authorized": False,
                "replacement_authorized": False,
                "protected_data_read": False,
            },
        )
        print(f"runtime qualification passed; id={result['artifact_id']}", flush=True)
    except Exception as error:
        write_record(
            output / "failure.json",
            {
                "schema": "dlolab-wrapping-risk-guard-runtime-qualification-failure-v7",
                "status": "terminal_technical_failure",
                "lock_id": lock["artifact_id"],
                "terminal_stage": stage,
                "terminal_task": active,
                "constructor_successes": _completed(output, "constructor"),
                "full_rollout_successes": _completed(output, "full"),
                "error_type": type(error).__name__,
                "message": str(error),
                "qualification_passed": False,
                "study_automatically_authorized": False,
                "retry_authorized": False,
                "replacement_authorized": False,
                "protected_data_read": False,
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--worker-kind", choices=("constructor", "full"))
    parser.add_argument("--worker-index", type=int)
    args = parser.parse_args()
    worker = args.worker_kind is not None or args.worker_index is not None
    if worker:
        if args.worker_kind is None or args.worker_index is None:
            raise ValueError("complete registered qualification worker required")
        _worker(args.output, args.worker_kind, args.worker_index)
    else:
        _run(args.output)


if __name__ == "__main__":
    main()
