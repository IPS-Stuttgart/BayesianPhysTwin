"""Prefix-only physical-source adapter for the public Deform360 v5 panel."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np

from . import deform360_bias_aware_prospective_artifacts as artifacts
from . import deform360_bias_aware_prospective_physical as physical
from ._portable_contracts import load_strict_json_object
from .deform360_bias_aware_prospective_artifacts import canonical_sha256, file_sha256
from .deform360_bias_aware_prospective_staging import select_action_only_window
from .deform360_joint_sparse_source_gate_v5 import (
    load_deform360_joint_sparse_source_execution_lock_v5,
)

PROTOCOL_ID = "deform360-joint-sparse-physical-source-v5"
SOURCE_PREPARATION_FILENAME = "bias_aware_source_preparation_manifest.json"
SOURCE_PREPARATION_KIND = "Deform360BiasAwareSourcePreparation"
PREPARED_INVENTORY_SCHEMA = (
    "bayesian-phystwin.deform360-calibration-prepared-source-inventory"
)
REQUIRED_STAGE_FILES = frozenset(
    {
        "scripts/remote/build_deform360_bias_aware_automatic_twin.py",
        "scripts/remote/run_deform360_bias_aware_frame_zero.py",
        "scripts/remote/run_deform360_bias_aware_physical_prior.py",
        "scripts/remote/run_deform360_joint_sparse_physical_source_v5.py",
        "scripts/remote/stage_deform360_bias_aware_prediction_prefix.py",
        "scripts/science/materialize_deform360_joint_sparse_physical_source_v5.py",
        "src/bayesian_phystwin/deform360_bias_aware_prospective_physical.py",
        "src/bayesian_phystwin/deform360_frame_zero_depth_initializer.py",
        "src/bayesian_phystwin/deform360_joint_sparse_physical_source_v5.py",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, *, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return cast(Sequence[Any], value)


def _nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _sha256(value: object, *, name: str) -> str:
    digest = _nonempty_string(value, name=name)
    if len(digest) != 64:
        raise ValueError(f"{name} must be a SHA-256 digest")
    return digest


def _git_output(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _cohort(lock: Mapping[str, Any]) -> dict[str, tuple[int, str]]:
    cohort = _mapping(lock.get("cohort"), name="cohort")
    rows = _sequence(cohort.get("development_objects"), name="development_objects")
    result: dict[str, tuple[int, str]] = {}
    for index, raw in enumerate(rows):
        row = _mapping(raw, name=f"development_objects[{index}]")
        object_id = _nonempty_string(row.get("object_id"), name="development object_id")
        episode_id = row.get("episode_id")
        stratum = row.get("stratum")
        _require(
            object_id == object_id.strip() and "\x00" not in object_id,
            "invalid development object_id",
        )
        _require(
            type(episode_id) is int and episode_id >= 0,
            "invalid development episode_id",
        )
        _require(stratum in {"sheet", "volumetric"}, "invalid source stratum")
        _require(object_id not in result, "development object repeats")
        result[object_id] = (cast(int, episode_id), cast(str, stratum))
    _require(len(result) == 10, "physical source requires ten development objects")
    return result


def load_joint_sparse_physical_execution_protocol_v5(
    path: str | Path,
) -> dict[str, Any]:
    """Expose the source lock through the normalized historical interface."""

    lock = load_deform360_joint_sparse_source_execution_lock_v5(path)
    cohort = _cohort(lock)
    normalized: dict[str, dict[str, tuple[int, ...]]] = {
        "sheet": {},
        "volumetric": {},
    }
    for object_id, (episode_id, stratum) in cohort.items():
        normalized[stratum][object_id] = (episode_id,)
    return {
        "payload": dict(lock),
        "config": {
            "protocol_id": PROTOCOL_ID,
            "dataset": dict(
                _mapping(lock.get("public_measurements"), name="public_measurements")
            ),
        },
        "config_sha256": str(lock["execution_lock_id"]),
        "calibration_cohort": normalized,
        "target_cohort": {"sheet": {}, "volumetric": {}},
    }


def joint_sparse_physical_case_records_v5(
    protocol_path: str | Path,
    *,
    role: str | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return the exact ten source cases and no confirmation case."""

    _require(role in {None, "calibration"}, "v5 physical adapter has no target role")
    lock = load_deform360_joint_sparse_source_execution_lock_v5(protocol_path)
    rows = tuple(
        {
            "case": f"{object_id}-ep{episode_id:04d}",
            "object_id": object_id,
            "episode_id": episode_id,
            "episode_key": f"{object_id}/{episode_id}",
            "stratum": stratum,
            "role": "calibration",
        }
        for object_id, (episode_id, stratum) in sorted(_cohort(lock).items())
    )
    _require(len(rows) == 10, "v5 physical case panel is incomplete")
    return rows


def joint_sparse_physical_case_record_v5(
    protocol_path: str | Path,
    *,
    object_id: str,
    episode_id: int,
) -> dict[str, Any]:
    """Resolve one exact public source object and episode."""

    matches = [
        row
        for row in joint_sparse_physical_case_records_v5(protocol_path)
        if row["object_id"] == object_id and row["episode_id"] == int(episode_id)
    ]
    _require(len(matches) == 1, "object/episode is outside the v5 source lock")
    return matches[0]


def validate_joint_sparse_physical_execution_v5(
    path: str | Path,
    *,
    repository: str | Path,
    require_clean_repository: bool = True,
) -> Mapping[str, Any]:
    """Validate every source file used by the process-local adapter."""

    lock_path = Path(path).resolve(strict=True)
    repo = Path(repository).resolve(strict=True)
    lock = load_deform360_joint_sparse_source_execution_lock_v5(lock_path)
    baseline = _mapping(lock.get("physical_baseline"), name="physical_baseline")
    _require(
        baseline.get("process_local_adapter_protocol_id") == PROTOCOL_ID,
        "physical source adapter protocol changed",
    )
    inventory = _mapping(
        baseline.get("prepared_source_inventory"), name="prepared_source_inventory"
    )
    _sha256(inventory.get("file_sha256"), name="prepared inventory file digest")
    _sha256(inventory.get("inventory_id"), name="prepared inventory identity")
    source_files = _mapping(
        baseline.get("source_files_sha256"), name="physical source files"
    )
    _require(
        REQUIRED_STAGE_FILES <= set(source_files),
        "physical source adapter file roster is incomplete",
    )
    for relative, expected in source_files.items():
        source = repo / str(relative)
        _require(source.is_file(), f"physical source file is missing: {relative}")
        _require(
            file_sha256(source) == expected,
            f"physical source file changed: {relative}",
        )
    if require_clean_repository:
        _require(not _git_output(repo, "status", "--porcelain"), "repository is dirty")
        _git_output(
            repo, "ls-files", "--error-unmatch", str(lock_path.relative_to(repo))
        )
    return lock


def _set(
    module: ModuleType, name: str, value: Any, changes: list[tuple[Any, str, Any]]
) -> None:
    if hasattr(module, name):
        changes.append((module, name, getattr(module, name)))
        setattr(module, name, value)


@contextmanager
def activate_joint_sparse_physical_runtime_v5() -> Iterator[None]:
    """Bind unchanged physical builders to v5 identities for one process."""

    changes: list[tuple[Any, str, Any]] = []
    for module in (artifacts, physical):
        _set(module, "PROTOCOL_ID", PROTOCOL_ID, changes)
    _set(
        artifacts,
        "load_bias_aware_prospective_protocol",
        load_joint_sparse_physical_execution_protocol_v5,
        changes,
    )
    _set(
        artifacts,
        "prospective_case_records",
        joint_sparse_physical_case_records_v5,
        changes,
    )
    _set(
        artifacts,
        "prospective_case_record",
        joint_sparse_physical_case_record_v5,
        changes,
    )
    try:
        yield
    finally:
        for module, name, value in reversed(changes):
            setattr(module, name, value)


def patch_joint_sparse_physical_stage_v5(
    module: ModuleType,
    *,
    stage: str,
    repository: Path,
    execution_lock: Path,
) -> None:
    """Patch aliases imported by one checksum-bound historical stage."""

    common = {
        "PROTOCOL_ID": PROTOCOL_ID,
        "load_bias_aware_prospective_protocol": (
            load_joint_sparse_physical_execution_protocol_v5
        ),
        "prospective_case_record": joint_sparse_physical_case_record_v5,
        "prospective_case_records": joint_sparse_physical_case_records_v5,
    }
    for name, value in common.items():
        if hasattr(module, name):
            setattr(module, name, value)
    if stage == "physical-prior":
        dynamic_module = cast(Any, module)
        original_run_logged = dynamic_module._run_logged

        def run_logged(command: Sequence[str], **kwargs: Any):
            rewritten = list(command)
            if len(rewritten) >= 2 and Path(rewritten[1]).name == (
                "build_deform360_bias_aware_automatic_twin.py"
            ):
                rewritten = [
                    rewritten[0],
                    str(
                        repository
                        / "scripts/remote/run_deform360_joint_sparse_physical_source_v5.py"
                    ),
                    "--execution-repo",
                    str(repository),
                    "--execution-lock",
                    str(execution_lock),
                    "--stage",
                    "automatic-twin",
                    *rewritten[2:],
                ]
            return original_run_logged(rewritten, **kwargs)

        dynamic_module._run_logged = run_logged


def _verified_source(root: Path, record: Mapping[str, Any], *, name: str) -> Path:
    relative = _nonempty_string(record.get("path"), name=f"{name} path")
    expected = _sha256(record.get("sha256"), name=f"{name} digest")
    requested = root / relative
    _require(requested.is_file() and not requested.is_symlink(), f"{name} is missing")
    source = requested.resolve(strict=True)
    try:
        source.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} escapes the processed root") from error
    _require(file_sha256(source) == expected, f"{name} SHA-256 changed")
    return source


def _copy_verified(source: Path, destination: Path, expected: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    _require(file_sha256(destination) == expected, "source copy changed")


def materialize_joint_sparse_physical_source_v5(
    *,
    execution_lock_path: str | Path,
    prepared_source_inventory_path: str | Path,
    processed_root: str | Path,
    object_id: str,
    output_root: str | Path,
) -> dict[str, Any]:
    """Copy and attest one released source episode without reading an outcome."""

    lock_path = Path(execution_lock_path).resolve(strict=True)
    lock = load_deform360_joint_sparse_source_execution_lock_v5(lock_path)
    cohort = _cohort(lock)
    if object_id not in cohort:
        raise ValueError("object is outside the v5 source lock")
    record = joint_sparse_physical_case_record_v5(
        lock_path,
        object_id=object_id,
        episode_id=cohort[object_id][0],
    )
    baseline = _mapping(lock.get("physical_baseline"), name="physical_baseline")
    inventory_lock = _mapping(
        baseline.get("prepared_source_inventory"), name="prepared_source_inventory"
    )
    inventory_path = Path(prepared_source_inventory_path).resolve(strict=True)
    _require(
        file_sha256(inventory_path) == inventory_lock.get("file_sha256"),
        "prepared source inventory file changed",
    )
    inventory = load_strict_json_object(
        inventory_path, label="prepared source inventory"
    )
    _require(
        inventory.get("schema") == PREPARED_INVENTORY_SCHEMA
        and inventory.get("schema_version") == 1
        and inventory.get("inventory_id") == inventory_lock.get("inventory_id")
        and inventory.get("object_count") == 10,
        "prepared source inventory identity changed",
    )
    boundary = _mapping(
        inventory.get("information_boundary"), name="inventory information boundary"
    )
    _require(
        boundary.get("calibration_target_metrics_computed") is False
        and boundary.get("confirmation_payloads_opened") is False
        and boundary.get("target_outcomes_used") is False,
        "prepared source inventory crossed its information boundary",
    )
    inventory_rows = [
        _mapping(item, name="prepared source object")
        for item in _sequence(inventory.get("objects"), name="prepared source objects")
    ]
    _require(
        {str(item.get("object_id")) for item in inventory_rows} == set(_cohort(lock)),
        "prepared source inventory cohort changed",
    )
    matches = [item for item in inventory_rows if item.get("object_id") == object_id]
    _require(len(matches) == 1, "prepared source object is missing or repeated")
    source_row = matches[0]
    _require(
        source_row.get("episode_id") == record["episode_id"],
        "prepared source episode changed",
    )

    root = Path(processed_root).resolve(strict=True)
    _require(root.is_dir() and not root.is_symlink(), "processed root is invalid")
    destination = (
        Path(output_root).absolute() / object_id / f"episode_{record['episode_id']:04d}"
    )
    _require(not destination.exists(), "physical source episode already exists")
    scratch = destination.with_name(f".{destination.name}.incomplete-{os.getpid()}")
    _require(not scratch.exists(), "physical source scratch path already exists")
    scratch.mkdir(parents=True)
    copied: dict[str, str] = {}
    try:
        episode_files = _mapping(source_row.get("episode_files"), name="episode_files")
        destinations = {
            "alignment": Path("alignment.json"),
            "extrinsics": Path("extrinsics.npy"),
            "robot": Path("robot/robot.npz"),
            "undistorted_intrinsics": Path("undistorted_intrinsics.npy"),
        }
        for key, relative_destination in destinations.items():
            source_record = _mapping(episode_files.get(key), name=f"episode {key}")
            source = _verified_source(root, source_record, name=f"episode {key}")
            expected = cast(str, source_record["sha256"])
            _copy_verified(source, scratch / relative_destination, expected)
            copied[relative_destination.as_posix()] = expected

        camera_rows = [
            _mapping(item, name="prepared source camera")
            for item in _sequence(source_row.get("cameras"), name="prepared cameras")
        ]
        _require(len(camera_rows) >= 8, "physical source needs at least eight cameras")
        cameras: list[str] = []
        for camera_row in camera_rows:
            camera = _nonempty_string(
                camera_row.get("camera"), name="prepared camera identity"
            )
            _require(
                camera not in cameras,
                "prepared camera identity changed",
            )
            cameras.append(camera)
            for key, filename in (
                ("alignment", "alignment.json"),
                ("metadata", "metadata.json"),
                ("timestamps", "aligned_timestamps.txt"),
                ("video", "undistorted.mp4"),
            ):
                source_record = _mapping(
                    camera_row.get(key), name=f"camera {camera} {key}"
                )
                source = _verified_source(
                    root, source_record, name=f"camera {camera} {key}"
                )
                expected = cast(str, source_record["sha256"])
                relative_destination = Path(camera) / filename
                _copy_verified(source, scratch / relative_destination, expected)
                copied[relative_destination.as_posix()] = expected
        _require(cameras == sorted(cameras), "prepared cameras are not sorted")

        robot_path = scratch / "robot/robot.npz"
        with np.load(robot_path, allow_pickle=False) as stored:
            selection = select_action_only_window(
                np.asarray(stored["actions"]), np.asarray(stored["openings"])
            )
        expected_window = _mapping(
            source_row.get("action_window"), name="action_window"
        )
        _require(
            selection["selected_raw_frame_range_half_open"]
            == expected_window.get("selected_raw_frame_range_half_open")
            and selection["prefix_raw_frame_range_half_open"]
            == expected_window.get("prefix_raw_frame_range_half_open")
            and selection["prediction_raw_frame_range_half_open"]
            == expected_window.get("prediction_raw_frame_range_half_open"),
            "released action-only source window changed",
        )
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "artifact_kind": SOURCE_PREPARATION_KIND,
            "protocol_id": PROTOCOL_ID,
            "protocol_config_sha256": lock["execution_lock_id"],
            **record,
            "target_access_authorization": None,
            "prepared_source_inventory_id": inventory["inventory_id"],
            "source_files_sha256": dict(sorted(copied.items())),
            "inputs_sha256": {
                "execution_lock": file_sha256(lock_path),
                "prepared_source_inventory": file_sha256(inventory_path),
            },
            "information_boundary": {
                "released_calibration_recordings_copied": True,
                "development_suffix_scored": False,
                "confirmation_payloads_opened": False,
                "target_outcomes_used": False,
                "new_measurements_collected": False,
                "human_approval_used": False,
            },
        }
        manifest["result_sha256"] = canonical_sha256(
            manifest, digest_key="result_sha256"
        )
        manifest_path = scratch / SOURCE_PREPARATION_FILENAME
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(scratch, destination)
    except Exception:
        shutil.rmtree(scratch, ignore_errors=True)
        raise
    return manifest


__all__ = [
    "PREPARED_INVENTORY_SCHEMA",
    "PROTOCOL_ID",
    "REQUIRED_STAGE_FILES",
    "SOURCE_PREPARATION_FILENAME",
    "activate_joint_sparse_physical_runtime_v5",
    "joint_sparse_physical_case_record_v5",
    "joint_sparse_physical_case_records_v5",
    "load_joint_sparse_physical_execution_protocol_v5",
    "materialize_joint_sparse_physical_source_v5",
    "patch_joint_sparse_physical_stage_v5",
    "validate_joint_sparse_physical_execution_v5",
]
