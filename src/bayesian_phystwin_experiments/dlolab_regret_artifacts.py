"""Write-once custody for the isolated, procedural DLO-Lab source study."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin._portable_contracts import content_id, load_strict_json_object

from .deform_state_restart import array_digest, file_digest, write_json_once
from .dlolab_native import DloLabConfig, verify_upstream
from .dlolab_regret_study import protocol

SOURCE_PATHS = (
    "src/bayesian_phystwin_experiments/dlolab_native.py",
    "src/bayesian_phystwin_experiments/dlolab_regret_study.py",
    "src/bayesian_phystwin_experiments/dlolab_regret_artifacts.py",
    "src/bayesian_phystwin_experiments/coupled_action_regret.py",
    "src/bayesian_phystwin_experiments/deform_state_restart.py",
    "src/bayesian_phystwin/guard_harm_risk.py",
    "src/bayesian_phystwin/_portable_contracts.py",
    "src/bayesian_phystwin/_canonical_contracts.py",
    "scripts/remote/qualify_dlolab_native.py",
    "scripts/remote/run_dlolab_regret_source.py",
    "scripts/remote/verify_dlolab_regret_source.py",
    "tests/test_dlolab_native.py",
    "tests/test_dlolab_regret_study.py",
    "tests/test_coupled_action_regret.py",
    "tests/test_dlolab_regret_artifacts.py",
    "tests/test_dlolab_regret_verifier.py",
    "docs/dlolab_coupled_action_regret_source_v1.md",
)
QUALIFICATION_CHECKS = (
    "finite_native_trajectories",
    "replay_trajectory_byte_identity",
    "replay_all_memory_byte_identity",
    "monolithic_trajectory_byte_identity",
    "monolithic_all_memory_byte_identity",
    "clamp_tracking_error_at_most_1e_minus_10_m",
    "alternative_action_moves_free_nodes_above_1e_minus_5_m",
    "segment_length_relative_error_at_most_10pct",
    "snapshot_unmodified",
)


def clean_revision(root: Path) -> str:
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True
    ).strip():
        raise ValueError("study execution requires clean committed source")
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def runtime_identity() -> dict[str, Any]:
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or os.environ.get("PYOPENGL_PLATFORM") != "osmesa"
    ):
        raise ValueError("registered CPU/software-rendering environment required")
    libraries = Path(os.environ.get("LD_LIBRARY_PATH", ""))
    library = libraries / "libOSMesa.so.8"
    if not library.is_file():
        raise ValueError("registered local OSMesa library missing")
    return {
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "torch",
                "quadrants",
                "scipy",
                "PyOpenGL",
                "genesis-world",
            )
        },
        "device": "cpu",
        "precision": "float64",
        "torch_threads": 1,
        "osmesa_sha256": file_digest(library.resolve(strict=True)),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "PYOPENGL_PLATFORM",
                "LIBGL_ALWAYS_SOFTWARE",
                "LD_LIBRARY_PATH",
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
            )
        },
    }


def write_record(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    if "artifact_id" in value:
        raise ValueError("caller must not supply a content identity")
    result = {**value, "artifact_id": content_id(value)}
    write_json_once(path, result)
    return result


def read_record(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError("symlinked record is not admitted")
    result = dict(load_strict_json_object(path, label="DLO-Lab study record"))
    identity = result.pop("artifact_id", None)
    if identity != content_id(result):
        raise ValueError("study record content identity mismatch")
    return {**result, "artifact_id": identity}


def write_bundle(directory: Path, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    if not arrays or any(
        x.dtype.kind not in "fiu" or not np.isfinite(x).all() for x in arrays.values()
    ):
        raise ValueError("artifact arrays must be finite numeric values")
    path = directory / "arrays.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return {
        "file": "arrays.npz",
        "file_sha256": file_digest(path),
        "arrays": {name: array_digest(value) for name, value in sorted(arrays.items())},
    }


def load_bundle(directory: Path, manifest: dict[str, Any]) -> dict[str, np.ndarray]:
    if (
        set(manifest) != {"file", "file_sha256", "arrays"}
        or manifest["file"] != "arrays.npz"
    ):
        raise ValueError("invalid array manifest or noncanonical artifact path")
    path = directory / "arrays.npz"
    if path.is_symlink() or file_digest(path) != manifest["file_sha256"]:
        raise ValueError("array artifact bytes changed")
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != set(manifest["arrays"]) or len(data.files) != len(
            set(data.files)
        ):
            raise ValueError("array artifact member set changed")
        result = {
            name: np.array(data[name], order="C", copy=True) for name in data.files
        }
    for name, value in result.items():
        if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError("artifact contains invalid numeric values")
        if array_digest(value) != manifest["arrays"][name]:
            raise ValueError("array identity changed")
    return result


def validate_qualification(path: Path, root: Path) -> dict[str, Any]:
    value = dict(load_strict_json_object(path, label="native qualification"))
    if (
        value.get("schema") != "dlolab-native-qualification-result-v1"
        or value.get("world_bank") is not True
        or value.get("qualification_passed") is not True
        or value.get("config_id") != DloLabConfig().identity
        or value.get("checks") != {key: True for key in QUALIFICATION_CHECKS}
        or value.get("protected_data_read") is not False
        or value.get("method_outcomes_read") is not False
    ):
        raise ValueError("complete passing world-bank qualification required")
    name = "src/bayesian_phystwin_experiments/dlolab_native.py"
    if value["source_sha256"][name] != file_digest(root / name):
        raise ValueError("qualified native source changed")
    if file_digest(path.parent / "trajectories.npz") != value["trajectories_sha256"]:
        raise ValueError("qualification trajectories changed")
    return {
        "path": str(path.resolve()),
        "sha256": file_digest(path),
        "source_revision": value["source_revision"],
    }


def freeze(
    root: Path, output: Path, upstream: Path, qualification: Path
) -> dict[str, Any]:
    revision = clean_revision(root)
    qual = validate_qualification(qualification, root)
    current_runtime = runtime_identity()
    source = {name: file_digest(root / name) for name in SOURCE_PATHS}
    upstream_identity = verify_upstream(upstream)
    output.mkdir(parents=True, exist_ok=False)
    return write_record(
        output / "lock.json",
        {
            "schema": "dlolab-regret-source-lock-v1",
            "source_revision": revision,
            "source_sha256": source,
            "output_root": str(output.resolve()),
            "upstream_root": str(upstream.resolve()),
            "upstream": upstream_identity,
            "runtime": current_runtime,
            "qualification": qual,
            "protocol": protocol(),
            "stage_order": ["bank", "calibrate", "predict", "score"],
            "protected_data_read": False,
            "physical_execution": False,
            "evaluation_outcomes_generated": False,
        },
    )


def validate_lock(root: Path, output: Path) -> dict[str, Any]:
    lock = read_record(output / "lock.json")
    if (
        lock.get("schema") != "dlolab-regret-source-lock-v1"
        or lock.get("protocol") != protocol()
    ):
        raise ValueError("registered protocol changed")
    if str(output.resolve()) != lock["output_root"]:
        raise ValueError("execution root differs from the one-attempt lock")
    if clean_revision(root) != lock["source_revision"]:
        raise ValueError("execution revision changed")
    if lock["source_sha256"] != {
        name: file_digest(root / name) for name in SOURCE_PATHS
    }:
        raise ValueError("registered source bytes changed")
    if (
        runtime_identity() != lock["runtime"]
        or verify_upstream(Path(lock["upstream_root"])) != lock["upstream"]
    ):
        raise ValueError("registered native runtime changed")
    if (
        validate_qualification(Path(lock["qualification"]["path"]), root)
        != lock["qualification"]
    ):
        raise ValueError("qualification identity changed")
    return lock


def read_stage(
    output: Path, stage: str, lock: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    value = read_record(output / stage / "seal.json")
    if (
        value.get("schema") != "dlolab-regret-stage-seal-v1"
        or value.get("stage") != stage
    ):
        raise ValueError("wrong stage seal")
    if (
        value.get("lock_id") != lock["artifact_id"]
        or value.get("status") != "ordinary_success"
    ):
        raise ValueError("stage is not an ordinary success under this lock")
    expected_count = {"bank": 15, "calibrate": 39, "predict": 64, "score": 64}[stage]
    if (
        value.get("count") != expected_count
        or value.get("protected_data_read") is not False
    ):
        raise ValueError("incomplete stage or information-boundary violation")
    if (
        stage in ("bank", "calibrate", "predict")
        and value.get("evaluation_outcomes_generated") is not False
    ):
        raise ValueError("evaluation outcomes preceded decision sealing")
    return value, load_bundle(output / stage, value["bundle"])
