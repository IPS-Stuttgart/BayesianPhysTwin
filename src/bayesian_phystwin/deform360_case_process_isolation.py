"""One-case process boundary for official Deform360 reconstruction.

The parent process owns cohort capabilities, barriers, x0 queries, and scores.
After a target-reconstruction capability has been consumed, it may launch one
child with only the paths needed to reconstruct that case. The child uses the
pinned default Deform360 trainer and exits after publishing one sealed result,
so process-global Nerfstudio resources cannot accumulate across cases.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np

from . import deform360_held_outcome_reconstruction as numerical

ISOLATION_ID = "deform360-official-case-process-isolation-v1"
RESULT_KIND = "Deform360IsolatedOfficialReconstructionResult"
SCHEMA_VERSION = 1
RESULT_ARRAY_NAMES = frozenset(
    {
        "object_points",
        "object_visibilities",
        "object_motions_valid",
    }
)
_ROLE_VALUES = frozenset({"calibration", "confirmation"})
_CANONICAL_FLOAT32 = np.dtype("<f4")
_CANONICAL_BOOL = np.dtype("|b1")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _artifact_sha256(value: Mapping[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("artifact_sha256", None)
    return hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()


def _sha256_file(path: str | Path) -> str:
    source = Path(os.path.abspath(os.fspath(path)))
    before = os.lstat(source)
    _require(
        stat.S_ISREG(before.st_mode) and not stat.S_ISLNK(before.st_mode),
        f"not a regular file: {source}",
    )
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        _require(
            (before.st_dev, before.st_ino) == (opened.st_dev, opened.st_ino),
            f"file changed while opening: {source}",
        )
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        after = os.fstat(descriptor)
        _require(
            (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            f"file changed while hashing: {source}",
        )
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _bound_file(path: str | Path) -> dict[str, Any]:
    source = Path(os.path.abspath(os.fspath(path)))
    return {
        "path": str(source),
        "sha256": _sha256_file(source),
        "size_bytes": os.lstat(source).st_size,
    }


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    descriptor = _canonical_bytes(
        {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
        }
    )
    digest = hashlib.sha256()
    digest.update(descriptor)
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _array_record(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": _array_sha256(array),
    }


def _validate_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    _require(
        set(arrays) == RESULT_ARRAY_NAMES,
        "isolated reconstruction array set changed",
    )
    points = np.asarray(arrays["object_points"])
    visible = np.asarray(arrays["object_visibilities"])
    valid = np.asarray(arrays["object_motions_valid"])
    _require(
        points.dtype == _CANONICAL_FLOAT32
        and points.ndim == 3
        and points.shape[0] == numerical.FRAME_COUNT
        and points.shape[1] > 0
        and points.shape[2] == 3
        and np.all(np.isfinite(points)),
        "isolated object points must be finite float32 (76, M, 3)",
    )
    _require(
        visible.dtype == valid.dtype == _CANONICAL_BOOL
        and visible.shape == valid.shape == points.shape[:2],
        "isolated masks must be bool (76, M)",
    )


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(path, 0o400, follow_symlinks=False)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _write_new_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(path, 0o400, follow_symlinks=False)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def write_isolated_reconstruction_result(
    archive_path: str | Path,
    manifest_path: str | Path,
    *,
    case_name: str,
    role: Literal["calibration", "confirmation"],
    lock_path: str | Path,
    cohort_barrier_sha256: str,
    reconstruction: Mapping[str, Any],
    worker_source_path: str | Path,
) -> dict[str, Any]:
    """Publish one write-once child result after official reconstruction."""

    _require(isinstance(case_name, str) and case_name, "case name is missing")
    _require(role in _ROLE_VALUES, "role must be calibration or confirmation")
    _require(
        isinstance(cohort_barrier_sha256, str)
        and len(cohort_barrier_sha256) == 64
        and all(value in "0123456789abcdef" for value in cohort_barrier_sha256),
        "cohort barrier digest is invalid",
    )
    archive = Path(os.path.abspath(os.fspath(archive_path)))
    manifest = Path(os.path.abspath(os.fspath(manifest_path)))
    _require(
        archive.parent == manifest.parent
        and archive.parent.is_dir()
        and not archive.parent.is_symlink(),
        "isolated result parent is absent, linked, or split",
    )
    _require(
        not os.path.lexists(archive) and not os.path.lexists(manifest),
        "isolated result already exists",
    )
    arrays = {
        "object_points": np.asarray(reconstruction.get("object_points")),
        "object_visibilities": np.asarray(
            reconstruction.get("object_visibilities")
        ),
        "object_motions_valid": np.asarray(
            reconstruction.get("object_motions_valid")
        ),
    }
    _validate_arrays(arrays)
    provenance = reconstruction.get("provenance")
    _require(isinstance(provenance, Mapping), "reconstruction provenance is absent")
    # Check JSON serializability before creating either output.
    json.loads(
        json.dumps(provenance, sort_keys=True, ensure_ascii=True, allow_nan=False)
    )

    try:
        _write_new_npz(archive, arrays)
        value: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": RESULT_KIND,
            "isolation_id": ISOLATION_ID,
            "case_name": case_name,
            "role": role,
            "lock": _bound_file(lock_path),
            "cohort_barrier_sha256": cohort_barrier_sha256,
            "archive": _bound_file(archive),
            "array_records": {
                name: _array_record(array) for name, array in sorted(arrays.items())
            },
            "reconstruction_provenance": dict(provenance),
            "resource_lifecycle_policy": dict(
                numerical.PROCESS_ISOLATED_RESOURCE_LIFECYCLE_POLICY
            ),
            "isolation_source": _bound_file(Path(__file__).resolve()),
            "worker_source": _bound_file(worker_source_path),
            "information_boundary": {
                "one_official_case_per_child_process": True,
                "parent_capability_consumed_before_child_launch": True,
                "x0_query_path_received": False,
                "queried_prediction_path_received": False,
                "future_score_capability_received": False,
                "score_path_received": False,
                "gate_path_received": False,
            },
        }
        value["artifact_sha256"] = _artifact_sha256(value)
        _write_new_json(manifest, value)
        return validate_isolated_reconstruction_result(
            manifest,
            expected_case_name=case_name,
            expected_role=role,
            expected_lock_path=lock_path,
            expected_cohort_barrier_sha256=cohort_barrier_sha256,
            expected_worker_source_path=worker_source_path,
        )
    except BaseException:
        if os.path.lexists(manifest):
            os.chmod(manifest, 0o600, follow_symlinks=False)
            manifest.unlink(missing_ok=True)
        if os.path.lexists(archive):
            os.chmod(archive, 0o600, follow_symlinks=False)
            archive.unlink(missing_ok=True)
        raise


def validate_isolated_reconstruction_result(
    manifest_path: str | Path,
    *,
    expected_case_name: str | None = None,
    expected_role: Literal["calibration", "confirmation"] | None = None,
    expected_lock_path: str | Path | None = None,
    expected_cohort_barrier_sha256: str | None = None,
    expected_worker_source_path: str | Path | None = None,
    expected_gsplat_runtime_smoke_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate and rehash one child result and every contained array."""

    manifest = Path(os.path.abspath(os.fspath(manifest_path)))
    observed = os.lstat(manifest)
    _require(
        stat.S_ISREG(observed.st_mode)
        and not stat.S_ISLNK(observed.st_mode)
        and stat.S_IMODE(observed.st_mode) == 0o400,
        "isolated result manifest must be a mode-0400 regular file",
    )
    value = json.loads(manifest.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "isolated result manifest is not an object")
    _require(
        set(value)
        == {
            "schema_version",
            "artifact_kind",
            "isolation_id",
            "case_name",
            "role",
            "lock",
            "cohort_barrier_sha256",
            "archive",
            "array_records",
            "reconstruction_provenance",
            "resource_lifecycle_policy",
            "isolation_source",
            "worker_source",
            "information_boundary",
            "artifact_sha256",
        },
        "isolated result manifest fields changed",
    )
    _require(
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("artifact_kind") == RESULT_KIND
        and value.get("isolation_id") == ISOLATION_ID
        and value.get("artifact_sha256") == _artifact_sha256(value),
        "isolated result identity or checksum changed",
    )
    if expected_case_name is not None:
        _require(
            value.get("case_name") == expected_case_name,
            "isolated result binds another case",
        )
    if expected_role is not None:
        _require(value.get("role") == expected_role, "isolated result role changed")
    if expected_lock_path is not None:
        _require(
            value.get("lock") == _bound_file(expected_lock_path),
            "isolated result binds another lock",
        )
    if expected_cohort_barrier_sha256 is not None:
        _require(
            value.get("cohort_barrier_sha256")
            == expected_cohort_barrier_sha256,
            "isolated result binds another cohort barrier",
        )
    if expected_worker_source_path is not None:
        _require(
            value.get("worker_source") == _bound_file(expected_worker_source_path),
            "isolated result binds another worker",
        )
    _require(
        value.get("resource_lifecycle_policy")
        == dict(numerical.PROCESS_ISOLATED_RESOURCE_LIFECYCLE_POLICY),
        "isolated result lifecycle policy changed",
    )
    _require(
        value.get("isolation_source") == _bound_file(Path(__file__).resolve()),
        "isolated result implementation source changed",
    )
    information = value.get("information_boundary")
    _require(
        information
        == {
            "one_official_case_per_child_process": True,
            "parent_capability_consumed_before_child_launch": True,
            "x0_query_path_received": False,
            "queried_prediction_path_received": False,
            "future_score_capability_received": False,
            "score_path_received": False,
            "gate_path_received": False,
        },
        "isolated result information boundary changed",
    )
    archive_record = value.get("archive")
    _require(
        isinstance(archive_record, Mapping)
        and Path(str(archive_record.get("path", ""))).parent == manifest.parent
        and dict(archive_record)
        == _bound_file(str(archive_record.get("path", ""))),
        "isolated result archive binding changed",
    )
    archive = Path(str(archive_record["path"]))
    archive_mode = os.lstat(archive)
    _require(
        stat.S_ISREG(archive_mode.st_mode)
        and not stat.S_ISLNK(archive_mode.st_mode)
        and stat.S_IMODE(archive_mode.st_mode) == 0o400,
        "isolated result archive must be a mode-0400 regular file",
    )
    with np.load(archive, allow_pickle=False) as stored:
        _require(
            set(stored.files) == RESULT_ARRAY_NAMES,
            "isolated result archive arrays changed",
        )
        arrays = {name: np.asarray(stored[name]).copy() for name in stored.files}
    _validate_arrays(arrays)
    _require(
        value.get("array_records")
        == {name: _array_record(array) for name, array in sorted(arrays.items())},
        "isolated result array records changed",
    )
    provenance = value.get("reconstruction_provenance")
    _require(
        isinstance(provenance, Mapping),
        "isolated reconstruction provenance changed",
    )
    if expected_gsplat_runtime_smoke_artifact_sha256 is not None:
        _require(
            isinstance(expected_gsplat_runtime_smoke_artifact_sha256, str)
            and len(expected_gsplat_runtime_smoke_artifact_sha256) == 64
            and all(
                character in "0123456789abcdef"
                for character in expected_gsplat_runtime_smoke_artifact_sha256
            ),
            "expected gsplat runtime smoke digest is invalid",
        )
        smoke = provenance.get("isolated_gsplat_runtime_smoke")
        _require(
            isinstance(smoke, Mapping)
            and smoke.get("artifact_sha256")
            == expected_gsplat_runtime_smoke_artifact_sha256
            and smoke.get("extension_loaded_and_retained") is True
            and smoke.get("target_or_outcome_path_accessed") is False,
            "isolated gsplat runtime smoke differs from the locked runtime",
        )
    return value


def load_isolated_reconstruction(
    manifest_path: str | Path,
    **validation_kwargs: Any,
) -> dict[str, Any]:
    """Load the exact mapping expected by the held target canonicalizer."""

    value = validate_isolated_reconstruction_result(
        manifest_path, **validation_kwargs
    )
    archive = Path(str(value["archive"]["path"]))
    with np.load(archive, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]).copy() for name in stored.files}
    return {
        **arrays,
        "provenance": dict(value["reconstruction_provenance"]),
    }


def build_isolated_reconstruction_subprocess(
    *,
    python_executable: str | Path,
    deployed_code: str | Path,
    lock_path: str | Path,
    role: Literal["calibration", "confirmation"],
    case_name: str,
    online_prediction_seal_path: str | Path,
    aligned_episode_dir: str | Path,
    reconstruction_output_dir: str | Path,
    result_archive_path: str | Path,
    result_manifest_path: str | Path,
    cohort_barrier_sha256: str,
    deform360_repo: str | Path,
    sam2_repository: str | Path,
    sam2_checkpoint: str | Path,
    cotracker_repo: str | Path,
    cotracker_checkpoint: str | Path,
    device: str,
    ffmpeg: str,
    pycache_prefix: str,
    path_environment: str,
) -> tuple[tuple[str, ...], dict[str, str], str]:
    """Build the one-case worker command without query or score paths."""

    code = Path(os.path.abspath(os.fspath(deployed_code)))
    worker = code / "scripts" / "held" / "run_deform360_isolated_reconstruction.py"
    argv = (
        os.fspath(python_executable),
        "-I",
        "-B",
        "-X",
        f"pycache_prefix={pycache_prefix}",
        str(worker),
        "--lock",
        os.fspath(lock_path),
        "--role",
        role,
        "--case-name",
        case_name,
        "--online-prediction-seal",
        os.fspath(online_prediction_seal_path),
        "--aligned-episode",
        os.fspath(aligned_episode_dir),
        "--reconstruction-output-dir",
        os.fspath(reconstruction_output_dir),
        "--result-archive",
        os.fspath(result_archive_path),
        "--result-manifest",
        os.fspath(result_manifest_path),
        "--cohort-barrier-sha256",
        cohort_barrier_sha256,
        "--deform360-repo",
        os.fspath(deform360_repo),
        "--sam2-repository",
        os.fspath(sam2_repository),
        "--sam2-checkpoint",
        os.fspath(sam2_checkpoint),
        "--cotracker-repo",
        os.fspath(cotracker_repo),
        "--cotracker-checkpoint",
        os.fspath(cotracker_checkpoint),
        "--device",
        device,
        "--ffmpeg",
        ffmpeg,
    )
    environment = {
        "BPT_HELD_V8_OUTCOME_ENV_NORMALIZED": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_HUB_OFFLINE": "1",
        "HOME": "/home/florianpfaff",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "LOGNAME": "florianpfaff",
        "PATH": path_environment,
        "PYNPUT_BACKEND": "dummy",
        "PYOPENGL_PLATFORM": "egl",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPYCACHEPREFIX": pycache_prefix,
        "PYTHONSAFEPATH": "1",
        "TMPDIR": "/tmp",
        "TRANSFORMERS_OFFLINE": "1",
        "USER": "florianpfaff",
        "WANDB_MODE": "disabled",
    }
    return argv, environment, str(Path(result_manifest_path).parent)


def run_isolated_reconstruction_subprocess(
    *,
    stdout_log_path: str | Path,
    stderr_log_path: str | Path,
    **build_kwargs: Any,
) -> dict[str, Any]:
    """Run one child, seal its logs, and load its validated result."""

    stdout_path = Path(os.path.abspath(os.fspath(stdout_log_path)))
    stderr_path = Path(os.path.abspath(os.fspath(stderr_log_path)))
    _require(
        stdout_path.parent == stderr_path.parent
        and not os.path.lexists(stdout_path)
        and not os.path.lexists(stderr_path),
        "isolated worker logs are split or already exist",
    )
    subprocess_kwargs = dict(build_kwargs)
    expected_smoke = subprocess_kwargs.pop(
        "expected_gsplat_runtime_smoke_artifact_sha256", None
    )
    argv, environment, safe_cwd = build_isolated_reconstruction_subprocess(
        **subprocess_kwargs
    )
    stdout_descriptor = os.open(
        stdout_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        stderr_descriptor = os.open(
            stderr_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except BaseException:
        os.close(stdout_descriptor)
        os.chmod(stdout_path, 0o400, follow_symlinks=False)
        raise
    try:
        with (
            os.fdopen(stdout_descriptor, "wb") as stdout_stream,
            os.fdopen(stderr_descriptor, "wb") as stderr_stream,
        ):
            completed = subprocess.run(
                argv,
                check=False,
                env=environment,
                cwd=safe_cwd,
                stdin=subprocess.DEVNULL,
                stdout=stdout_stream,
                stderr=stderr_stream,
                close_fds=True,
            )
            stdout_stream.flush()
            stderr_stream.flush()
            os.fsync(stdout_stream.fileno())
            os.fsync(stderr_stream.fileno())
        os.chmod(stdout_path, 0o400, follow_symlinks=False)
        os.chmod(stderr_path, 0o400, follow_symlinks=False)
        _require(
            completed.returncode == 0,
            f"isolated reconstruction worker failed with exit code "
            f"{completed.returncode}",
        )
        validation_kwargs = {
            "expected_case_name": build_kwargs["case_name"],
            "expected_role": build_kwargs["role"],
            "expected_lock_path": build_kwargs["lock_path"],
            "expected_cohort_barrier_sha256": build_kwargs[
                "cohort_barrier_sha256"
            ],
            "expected_worker_source_path": Path(build_kwargs["deployed_code"])
            / "scripts"
            / "held"
            / "run_deform360_isolated_reconstruction.py",
        }
        if expected_smoke is not None:
            validation_kwargs[
                "expected_gsplat_runtime_smoke_artifact_sha256"
            ] = expected_smoke
        return load_isolated_reconstruction(
            build_kwargs["result_manifest_path"], **validation_kwargs
        )
    except BaseException:
        for path in (stdout_path, stderr_path):
            if os.path.lexists(path):
                os.chmod(path, 0o400, follow_symlinks=False)
        raise


__all__ = [
    "ISOLATION_ID",
    "RESULT_KIND",
    "build_isolated_reconstruction_subprocess",
    "load_isolated_reconstruction",
    "run_isolated_reconstruction_subprocess",
    "validate_isolated_reconstruction_result",
    "write_isolated_reconstruction_result",
]
