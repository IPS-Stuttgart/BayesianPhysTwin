"""Two-barrier operator for fresh Deform360 held-v8 outcomes.

This module intentionally imports no target, reconstruction, scorer, or CUDA
module at import time.  Production entry points first run a source-only
deployment verifier in a separate isolated interpreter, then smoke the pinned
CUDA runtime, cross barrier one, and only then load the future-bearing APIs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import socket
import stat
import subprocess
import sys
from typing import Any, Callable, Literal, Mapping, Sequence


PROTOCOL_ID = "deform360-held-online-belief-v8.1"
NO_GO_EXIT_CODE = 3
NOT_CONFIRMED_EXIT_CODE = 4
POST_CASE_FD_GROWTH_LIMIT = 32
QUALIFIED_RLIMIT_NOFILE_SOFT = 1024
ROLE_EXECUTION_COMPLETION_KIND = "Deform360HeldV8RoleExecutionCompletion"
ROLE_EXECUTION_COMPLETION_STATUS = "role-execution-complete"
ROLE_EXECUTION_COMPLETION_SUFFIX = "-execution-completion.json"
CANONICAL_HELD_ROOT = Path("/mnt/corsair/florianpfaff/bpt-online-belief-v1/held-v8")
CANONICAL_ALIGNED_ROOT = Path(
    "/mnt/lexar4tb/datasets/deform360/data-7fea8e2/replication-v1/aligned"
)
LOCKED_GSPLAT_SMOKE_EVIDENCE = Path(
    "/mnt/corsair/florianpfaff/bpt-online-belief-v1/held-v7/"
    "gsplat-runtime-smoke-evidence.json"
)
PINNED_PYTHON = Path(
    "/mnt/corsair/florianpfaff/bpt-held-v5-runtimes/"
    "bpt-gpu-pip-4948737892f77c6a9496795e6c3f25b92fcea466ddb7b5f1e9c1b0de1137f004/"
    "bin/python"
)
PYCACHE_PREFIX = "/nonexistent/bpt-held-v8-pycache"
PINNED_PATH = "/usr/local/bin:/usr/bin:/bin"

_DEPLOYMENT_BINDINGS = {
    "held_v81_attempt5_launcher_source": ("scripts/held/run_deform360_v81_attempt5.py"),
    "held_v8_outcome_driver_source": (
        "src/bayesian_phystwin/deform360_held_v8_outcome_driver.py"
    ),
    "held_v8_outcome_integrity_source": (
        "src/bayesian_phystwin/deform360_held_v8_outcome_integrity.py"
    ),
    "held_v8_role_outcome_integrity_sealer_source": (
        "scripts/held/seal_deform360_v8_role_outcome.py"
    ),
    "held_v8_outcome_reconstruction_adapter_source": (
        "src/bayesian_phystwin/deform360_held_v8_outcome_reconstruction.py"
    ),
    "held_v8_x0_query_worker_source": ("scripts/held/run_deform360_v8_x0_query.py"),
    "held_v8_gsplat_runtime_adapter_source": (
        "src/bayesian_phystwin/deform360_held_v8_gsplat_runtime.py"
    ),
    "held_v8_protocol_source": ("src/bayesian_phystwin/deform360_held_v8_protocol.py"),
    "held_v8_confirmation_source_operator_source": (
        "src/bayesian_phystwin/deform360_held_v8_confirmation_source.py"
    ),
    "held_v8_confirmation_source_materialization_launcher_source": (
        "scripts/held/run_deform360_v8_confirmation_source.py"
    ),
    "held_v8_confirmation_camera_selection_lineage": (
        "milestones/deform360-replication-source-qa-v1/artifacts/"
        "source_geometry_qa.json"
    ),
    "held_v8_confirmation_preregistration_lineage": (
        "configs/causal4d_public/deform360_replication_v1.json"
    ),
    "held_v8_query_artifacts_source": (
        "src/bayesian_phystwin/deform360_held_v8_query_artifacts.py"
    ),
    "held_v8_outcome_artifacts_source": (
        "src/bayesian_phystwin/deform360_held_v8_outcome_artifacts.py"
    ),
    "held_v8_scoring_source": ("src/bayesian_phystwin/deform360_held_v8_scoring.py"),
    "held_v8_score_artifacts_source": (
        "src/bayesian_phystwin/deform360_held_v8_score_artifacts.py"
    ),
    "held_v8_frozen_query_field_source": (
        "src/bayesian_phystwin/deform360_frozen_query_field.py"
    ),
    "held_official_reconstruction_numerical_source": (
        "src/bayesian_phystwin/deform360_held_outcome_reconstruction.py"
    ),
    "held_gsplat_runtime_source": (
        "src/bayesian_phystwin/deform360_held_gsplat_runtime.py"
    ),
}
_RECONSTRUCTION_RUNTIME_BINDINGS = frozenset(
    {
        "cotracker_checkpoint",
        "cotracker_commit_object",
        "cotracker_git_tree_manifest",
        "cotracker_revision_literal",
        "deform360_code_commit_object",
        "deform360_code_git_tree_manifest",
        "deform360_code_revision_literal",
        "ffmpeg_executable",
        "ffmpeg_version_literal",
        "sam2_checkpoint",
        "sam2_commit_object",
        "sam2_git_tree_manifest",
        "sam2_model_config",
        "sam2_revision_literal",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _emit(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def canonical_role_execution_completion_path(
    held_root: str | Path,
    role: Literal["calibration", "confirmation"],
) -> Path:
    _require(role in {"calibration", "confirmation"}, "outcome role is invalid")
    root = Path(os.path.abspath(os.fspath(held_root)))
    return root / role / f"{role}{ROLE_EXECUTION_COMPLETION_SUFFIX}"


def canonical_role_source_manifest_path(
    held_root: str | Path,
    role: Literal["calibration", "confirmation"],
) -> Path:
    root = Path(os.path.abspath(os.fspath(held_root)))
    if role == "calibration":
        return root / "replacement-source" / "manifests" / "aligned-source.json"
    return root / "confirmation-source" / "manifests" / "aligned-source-cohort.json"


def _open_file_descriptor_count() -> int:
    """Count process descriptors through procfs without opening a payload."""

    root = Path("/proc/self/fd")
    _require(
        root.is_dir() and not root.is_symlink(),
        "procfs file-descriptor census is unavailable",
    )
    count = len(os.listdir(root))
    _require(count > 0, "procfs file-descriptor census is empty")
    return count


def _rlimit_nofile_pair() -> tuple[int, int]:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    return int(soft), int(hard)


def _validate_no_writable_held_descriptors(held_root: str | Path) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(held_root)))
    _require(
        root.is_dir() and not root.is_symlink() and root.resolve() == root,
        "writable-FD guard received an invalid held root",
    )
    proc_path = Path("/proc/self/fd")
    descriptor = os.open(
        proc_path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    inspected: list[int] = []
    violations: list[dict[str, Any]] = []
    try:
        names = sorted(
            (name for name in os.listdir(descriptor) if name.isdecimal()),
            key=int,
        )
        for name in names:
            candidate = int(name)
            if candidate == descriptor:
                continue
            try:
                flags = fcntl.fcntl(candidate, fcntl.F_GETFL)
                target = os.readlink(name, dir_fd=descriptor)
            except (FileNotFoundError, OSError) as error:
                if isinstance(error, OSError) and error.errno == 9:
                    continue
                raise
            inspected.append(candidate)
            if flags & os.O_ACCMODE not in {os.O_WRONLY, os.O_RDWR}:
                continue
            normalized_target = target.removesuffix(" (deleted)")
            if not normalized_target.startswith("/"):
                continue
            target_path = Path(os.path.abspath(normalized_target))
            try:
                target_path.relative_to(root)
            except ValueError:
                continue
            violations.append(
                {
                    "file_descriptor": candidate,
                    "target": target,
                    "access_mode": flags & os.O_ACCMODE,
                }
            )
    finally:
        os.close(descriptor)
    _require(
        not violations,
        "parent retains a writable descriptor into the held root before sealing",
    )
    return {
        "procfs_path": str(proc_path),
        "fcntl_operation": "F_GETFL",
        "held_root": str(root),
        "checked_numeric_descriptor_count": len(inspected),
        "writable_held_root_or_descendant_descriptors": [],
        "validated_before_execution_completion_marker_open": True,
    }


def _validate_qualified_rlimit_nofile(
    observed: object,
    *,
    reference: tuple[int, int] | None = None,
    phase: str,
) -> tuple[int, int]:
    _require(
        isinstance(observed, tuple)
        and len(observed) == 2
        and all(type(value) is int for value in observed),
        f"{phase} RLIMIT_NOFILE pair is invalid",
    )
    pair = (observed[0], observed[1])
    if reference is not None:
        _require(
            pair == reference,
            f"RLIMIT_NOFILE soft/hard pair changed during {phase}",
        )
    soft, hard = pair
    _require(
        soft == QUALIFIED_RLIMIT_NOFILE_SOFT,
        f"{phase} RLIMIT_NOFILE soft limit differs from qualified value",
    )
    _require(
        hard == resource.RLIM_INFINITY or hard >= soft,
        f"{phase} RLIMIT_NOFILE hard limit is invalid",
    )
    return pair


def _sha256_file(path: str | Path) -> str:
    source = Path(os.path.abspath(os.fspath(path)))
    before = os.lstat(source)
    _require(stat.S_ISREG(before.st_mode), f"not a regular file: {source}")
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


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("execution completion is not canonical finite JSON") from error


def _execution_completion_artifact_sha256(value: Mapping[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("artifact_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(unsigned).rstrip(b"\n")).hexdigest()


def _stable_regular_state(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        stat.S_IFMT(value.st_mode),
        stat.S_IMODE(value.st_mode),
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _bound_regular_file(
    path: str | Path,
    *,
    role: str,
    required_mode: int | None = None,
) -> tuple[dict[str, Any], bytes]:
    source = Path(os.path.abspath(os.fspath(path)))
    before = os.lstat(source)
    _require(
        stat.S_ISREG(before.st_mode)
        and not stat.S_ISLNK(before.st_mode)
        and before.st_nlink == 1
        and source.resolve() == source,
        f"{role} is not a canonical single-link regular file",
    )
    if required_mode is not None:
        _require(
            stat.S_IMODE(before.st_mode) == required_mode,
            f"{role} mode is not {required_mode:04o}",
        )
    descriptor = os.open(
        source,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        _require(
            _stable_regular_state(opened) == _stable_regular_state(before),
            f"{role} changed while opening",
        )
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
        after = os.fstat(descriptor)
        current = os.lstat(source)
        _require(
            _stable_regular_state(opened)
            == _stable_regular_state(after)
            == _stable_regular_state(current),
            f"{role} changed while reading",
        )
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    return (
        {
            "path": str(source),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        },
        payload,
    )


def _json_object(payload: bytes, *, role: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{role} is not canonical JSON") from error
    _require(isinstance(value, dict), f"{role} is not a JSON object")
    return value


def _artifact_file_record(
    path: str | Path,
    artifact: Mapping[str, Any],
    *,
    role: str,
) -> dict[str, Any]:
    record, payload = _bound_regular_file(path, role=role, required_mode=0o400)
    observed = _json_object(payload, role=role)
    artifact_sha256 = artifact.get("artifact_sha256")
    _require(
        _valid_sha256(artifact_sha256)
        and observed.get("artifact_sha256") == artifact_sha256
        and _execution_completion_artifact_sha256(observed) == artifact_sha256,
        f"{role} artifact binding changed",
    )
    return {**record, "artifact_sha256": artifact_sha256}


def _semantic_return_code(
    role: Literal["calibration", "confirmation"], decision: object
) -> int:
    expected = {
        ("calibration", "GO"): 0,
        ("calibration", "NO-GO"): NO_GO_EXIT_CODE,
        ("confirmation", "CONFIRMED"): 0,
        ("confirmation", "NOT-CONFIRMED"): NOT_CONFIRMED_EXIT_CODE,
    }
    key = (role, decision)
    _require(key in expected, "gate returned an invalid semantic outcome")
    return expected[key]


def _git(
    code: Path, *arguments: str, check: bool = True
) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(
        ["git", "-C", str(code), *arguments],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "HOME": "/home/florianpfaff",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": PINNED_PATH,
        },
    )
    if check and completed.returncode != 0:
        raise ValueError(
            f"git {' '.join(arguments)} failed: "
            + completed.stderr.decode("utf-8", errors="replace").strip()
        )
    return completed


def _git_tree_records(raw: bytes) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for encoded in raw.split(b"\0"):
        if not encoded:
            continue
        header, separator, path_bytes = encoded.partition(b"\t")
        fields = header.split(b" ")
        _require(bool(separator) and len(fields) == 3, "malformed Git tree record")
        mode, kind, object_id = (field.decode("ascii") for field in fields)
        path = path_bytes.decode("utf-8")
        _require(
            mode in {"100644", "100755"}
            and kind == "blob"
            and len(object_id) in {40, 64}
            and all(character in "0123456789abcdef" for character in object_id)
            and path
            and not path.startswith("/")
            and ".." not in Path(path).parts,
            f"unsafe or unsupported tracked entry: {path}",
        )
        records.append(
            {"mode": mode, "type": kind, "object_id": object_id, "path": path}
        )
    _require(
        bool(records)
        and [record["path"] for record in records]
        == sorted(record["path"] for record in records),
        "Git tree is empty or unsorted",
    )
    return records


def _deployed_repository_evidence(code: Path) -> dict[str, str]:
    top = _git(code, "rev-parse", "--show-toplevel").stdout.decode().strip()
    _require(top == str(code), "deployed Git top level changed")
    head = _git(code, "rev-parse", "HEAD").stdout.decode().strip().lower()
    _require(
        len(head) in {40, 64}
        and all(character in "0123456789abcdef" for character in head)
        and code.name == f"code-{head}",
        "deployment path is not exact code-$HEAD",
    )
    symbolic = _git(code, "symbolic-ref", "-q", "HEAD", check=False)
    _require(
        symbolic.returncode == 1 and symbolic.stdout == b"",
        "deployed checkout is not detached",
    )
    _require(
        _git(code, "status", "--porcelain=v1", "--untracked-files=all").stdout == b"",
        "deployed worktree is not completely clean",
    )
    _git(code, "fsck", "--full", "--no-dangling")
    records = _git_tree_records(_git(code, "ls-tree", "-r", "-z", "HEAD").stdout)
    canonical = json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return {
        "head": head,
        "head_text_sha256": hashlib.sha256(head.encode("utf-8")).hexdigest(),
        "tree_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _validate_deployed_repository(
    code: Path, bindings: Mapping[str, Any]
) -> dict[str, str]:
    evidence = _deployed_repository_evidence(code)
    _require(
        bindings.get("method_head_text_sha256") == evidence["head_text_sha256"]
        and bindings.get("method_deployed_snapshot_tree") == evidence["tree_sha256"],
        "deployed HEAD or canonical Git tree differs from the lock",
    )
    return evidence


def _require_regular_mode(path: Path, mode: int, role: str) -> None:
    observed = os.lstat(path)
    _require(
        stat.S_ISREG(observed.st_mode)
        and not stat.S_ISLNK(observed.st_mode)
        and stat.S_IMODE(observed.st_mode) == mode,
        f"{role} is absent, linked, or not mode {mode:04o}",
    )


def _locked_gpu0_smoke_artifact(lock: Mapping[str, Any]) -> str:
    path = LOCKED_GSPLAT_SMOKE_EVIDENCE
    _require_regular_mode(path, 0o400, "locked gsplat smoke evidence")
    bindings = lock.get("immutable_bindings")
    _require(
        isinstance(bindings, Mapping)
        and bindings.get("v7_gsplat_runtime_smoke_evidence") == _sha256_file(path),
        "held-v8 lock does not bind the frozen gsplat smoke evidence",
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    smokes = value.get("smokes") if isinstance(value, Mapping) else None
    _require(isinstance(smokes, list), "locked gsplat smoke list is absent")
    matches = [
        smoke
        for smoke in smokes
        if isinstance(smoke, Mapping) and smoke.get("physical_gpu_index") == 0
    ]
    _require(len(matches) == 1, "locked gsplat evidence lacks unique GPU-0 smoke")
    artifact_sha256 = matches[0].get("artifact_sha256")
    _require(_valid_sha256(artifact_sha256), "locked GPU-0 smoke digest is invalid")
    return str(artifact_sha256)


@dataclass(frozen=True)
class DriverArguments:
    role: Literal["calibration", "confirmation"]
    deployed_code: str
    lock_path: str
    replacement_source_manifest_path: str | None
    dry_run_barrier_only: bool
    confirmation_source_manifest_path: str | None = None
    aligned_root: str | None = None
    deform360_repo: str | None = None
    sam2_repository: str | None = None
    sam2_checkpoint: str | None = None
    cotracker_repo: str | None = None
    cotracker_checkpoint: str | None = None
    device: str = "cuda:0"
    ffmpeg: str = "ffmpeg"


@dataclass(frozen=True)
class CasePaths:
    case_name: str
    physical_seal: Path
    online_seal: Path
    frozen_field_manifest: Path
    reconstruction_dir: Path
    isolated_reconstruction_archive: Path
    isolated_reconstruction_manifest: Path
    isolated_reconstruction_stdout: Path
    isolated_reconstruction_stderr: Path
    target_archive: Path
    target_manifest: Path
    official_query_archive: Path
    official_query_manifest: Path
    queried_archive: Path
    queried_seal: Path


@dataclass(frozen=True)
class OutcomeLayout:
    root: Path
    role: Literal["calibration", "confirmation"]
    role_root: Path
    protected_future_root: Path
    x0_query_root: Path
    queried_prediction_root: Path
    evidence_path: Path
    decision_path: Path
    execution_completion_path: Path
    cases: Mapping[str, CasePaths]


@dataclass(frozen=True)
class PostBarrierApi:
    backend_type: Callable[..., Any]
    reconstruct: Callable[..., Mapping[str, Any]]
    write_target_and_query: Callable[..., Mapping[str, Any]]
    validate_target: Callable[..., Mapping[str, Any]]
    validate_frozen_field: Callable[..., Mapping[str, Any]]
    validate_queried_prediction: Callable[..., Mapping[str, Any]]
    load_scoring_inputs: Callable[..., Any]
    score_case: Callable[..., dict[str, Any]]
    create_score_evidence_and_decision: Callable[
        ..., tuple[dict[str, Any], dict[str, Any]]
    ]
    validate_replacement_source: Callable[..., Mapping[str, Any]]
    validate_confirmation_source: Callable[..., Mapping[str, Any]]


def _case_paths(root: Path, role: str, case_name: str) -> CasePaths:
    case_root = root / role / "cases" / case_name
    protected = root / role / "private-targets" / case_name
    x0_query = root / role / "query-inputs" / case_name
    queried = root / role / "query-outputs" / case_name
    return CasePaths(
        case_name=case_name,
        physical_seal=case_root / "physical" / "physical_prior_seal.json",
        online_seal=case_root / "online" / "online_prediction_seal.json",
        frozen_field_manifest=(
            case_root / "frozen-field" / "preoutcome-frozen-field-manifest.json"
        ),
        reconstruction_dir=protected / "fresh-official-reconstruction",
        isolated_reconstruction_archive=(
            protected / "isolated-official-reconstruction.npz"
        ),
        isolated_reconstruction_manifest=(
            protected / "isolated-official-reconstruction.json"
        ),
        isolated_reconstruction_stdout=(
            protected / "isolated-official-reconstruction.stdout.log"
        ),
        isolated_reconstruction_stderr=(
            protected / "isolated-official-reconstruction.stderr.log"
        ),
        target_archive=protected / "official-target.npz",
        target_manifest=protected / "official-target-manifest.json",
        official_query_archive=x0_query / "official-frame-zero-query.npz",
        official_query_manifest=(x0_query / "official-frame-zero-query-manifest.json"),
        queried_archive=queried / "queried-prediction.npz",
        queried_seal=queried / "queried-prediction-seal.json",
    )


def build_layout(
    *, root: Path, role: Literal["calibration", "confirmation"], cases: Sequence[str]
) -> OutcomeLayout:
    role_root = root / role
    return OutcomeLayout(
        root=root,
        role=role,
        role_root=role_root,
        protected_future_root=role_root / "private-targets",
        x0_query_root=role_root / "query-inputs",
        queried_prediction_root=role_root / "query-outputs",
        evidence_path=role_root / f"{role}-score-evidence.json",
        decision_path=role_root / f"{role}-gate-decision.json",
        execution_completion_path=canonical_role_execution_completion_path(root, role),
        cases={case_name: _case_paths(root, role, case_name) for case_name in cases},
    )


def _validate_artifact_binding(
    record: object,
    *,
    expected_path: Path,
    role: str,
) -> dict[str, Any]:
    _require(
        isinstance(record, Mapping)
        and set(record) == {"path", "sha256", "size_bytes", "artifact_sha256"},
        f"{role} record fields changed",
    )
    bound, payload = _bound_regular_file(expected_path, role=role, required_mode=0o400)
    artifact = _json_object(payload, role=role)
    artifact_sha256 = artifact.get("artifact_sha256")
    _require(
        {key: record.get(key) for key in bound} == bound
        and record.get("artifact_sha256") == artifact_sha256
        and _valid_sha256(artifact_sha256)
        and _execution_completion_artifact_sha256(artifact) == artifact_sha256,
        f"{role} binding changed",
    )
    return artifact


def _validate_execution_resource_boundary(
    value: object,
    *,
    ordered_case_names: Sequence[str],
) -> None:
    _require(
        isinstance(value, Mapping)
        and set(value)
        == {
            "qualified_rlimit_nofile_soft",
            "post_case_fd_growth_limit",
            "initial_nofile",
            "pre_outcome",
            "post_cases",
            "end_outcome",
            "parent_writable_fd_guard",
            "publication",
        },
        "execution resource-boundary fields changed",
    )
    _require(
        value.get("qualified_rlimit_nofile_soft") == QUALIFIED_RLIMIT_NOFILE_SOFT
        and value.get("post_case_fd_growth_limit") == POST_CASE_FD_GROWTH_LIMIT,
        "execution resource-boundary contract changed",
    )
    initial = value.get("initial_nofile")
    _require(
        isinstance(initial, Mapping)
        and set(initial) == {"rlimit_nofile_soft", "rlimit_nofile_hard"}
        and type(initial.get("rlimit_nofile_soft")) is int
        and type(initial.get("rlimit_nofile_hard")) is int,
        "initial NOFILE observation changed",
    )
    expected_pair = _validate_qualified_rlimit_nofile(
        (initial["rlimit_nofile_soft"], initial["rlimit_nofile_hard"]),
        phase="sealed execution initial boundary",
    )
    pre = value.get("pre_outcome")
    _require(
        isinstance(pre, Mapping)
        and set(pre)
        == {
            "file_descriptor_count",
            "rlimit_nofile_soft",
            "rlimit_nofile_hard",
        }
        and type(pre.get("file_descriptor_count")) is int
        and pre["file_descriptor_count"] > 0,
        "pre-outcome resource observation changed",
    )
    reference = pre["file_descriptor_count"]
    observations = value.get("post_cases")
    _require(
        isinstance(observations, list) and len(observations) == len(ordered_case_names),
        "post-case resource observation cohort changed",
    )
    for index, (case_name, observation) in enumerate(
        zip(ordered_case_names, observations, strict=True)
    ):
        _require(
            isinstance(observation, Mapping)
            and set(observation)
            == {
                "case_name",
                "case_index",
                "file_descriptor_count",
                "reference_file_descriptor_count",
                "maximum_growth",
                "rlimit_nofile_soft",
                "rlimit_nofile_hard",
            }
            and observation.get("case_name") == case_name
            and observation.get("case_index") == index,
            f"post-case resource observation changed for {case_name}",
        )
    end = value.get("end_outcome")
    guard = value.get("parent_writable_fd_guard")
    publication = value.get("publication")
    standard_fields = {
        "file_descriptor_count",
        "reference_file_descriptor_count",
        "maximum_growth",
        "rlimit_nofile_soft",
        "rlimit_nofile_hard",
    }
    _require(
        isinstance(end, Mapping)
        and set(end) == standard_fields
        and isinstance(publication, Mapping)
        and set(publication)
        == standard_fields
        | {
            "marker_fd_open",
            "parent_directory_fd_open",
            "open_descriptor_delta_from_end",
        },
        "terminal resource observations changed",
    )
    _require(
        isinstance(guard, Mapping)
        and set(guard)
        == {
            "procfs_path",
            "fcntl_operation",
            "held_root",
            "checked_numeric_descriptor_count",
            "writable_held_root_or_descendant_descriptors",
            "validated_before_execution_completion_marker_open",
        }
        and guard.get("procfs_path") == "/proc/self/fd"
        and guard.get("fcntl_operation") == "F_GETFL"
        and type(guard.get("checked_numeric_descriptor_count")) is int
        and guard["checked_numeric_descriptor_count"] > 0
        and guard.get("writable_held_root_or_descendant_descriptors") == []
        and guard.get("validated_before_execution_completion_marker_open") is True,
        "parent writable-FD guard evidence changed",
    )
    all_bounded = [pre, *observations, end, publication]
    for observation in all_bounded:
        pair = (
            observation.get("rlimit_nofile_soft"),
            observation.get("rlimit_nofile_hard"),
        )
        _require(
            all(type(component) is int for component in pair),
            "NOFILE observation is not an exact integer pair",
        )
        _validate_qualified_rlimit_nofile(
            pair,
            reference=expected_pair,
            phase="sealed execution resource boundary",
        )
        if observation is pre:
            continue
        count = observation.get("file_descriptor_count")
        _require(
            type(count) is int
            and count > 0
            and observation.get("reference_file_descriptor_count") == reference
            and observation.get("maximum_growth") == POST_CASE_FD_GROWTH_LIMIT
            and count <= reference + POST_CASE_FD_GROWTH_LIMIT,
            "file-descriptor observation exceeds the frozen boundary",
        )
    _require(
        publication.get("marker_fd_open") is True
        and publication.get("parent_directory_fd_open") is True
        and publication.get("open_descriptor_delta_from_end") == 2
        and publication.get("file_descriptor_count")
        == end.get("file_descriptor_count") + 2,
        "publication was not observed with both completion descriptors open",
    )


def validate_role_execution_completion(
    completion_path: str | Path,
    *,
    lock_path: str | Path,
    expected_role: Literal["calibration", "confirmation"],
    expected_ordered_case_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate the driver's canonical pre-sealer execution completion."""

    path = Path(os.path.abspath(os.fspath(completion_path)))
    held_root = path.parent.parent
    _require(
        path == canonical_role_execution_completion_path(held_root, expected_role),
        "role execution completion path is not canonical",
    )
    _, payload = _bound_regular_file(
        path, role="role execution completion", required_mode=0o400
    )
    completion = _json_object(payload, role="role execution completion")
    _require(
        set(completion)
        == {
            "schema_version",
            "artifact_kind",
            "protocol_id",
            "status",
            "role",
            "lock",
            "ordered_case_names",
            "source_manifest",
            "score_evidence",
            "semantic_decision",
            "resource_boundary",
            "publication_contract",
            "self_hash_contract",
            "artifact_sha256",
        }
        and completion.get("schema_version") == 1
        and completion.get("artifact_kind") == ROLE_EXECUTION_COMPLETION_KIND
        and completion.get("protocol_id") == PROTOCOL_ID
        and completion.get("status") == ROLE_EXECUTION_COMPLETION_STATUS
        and completion.get("role") == expected_role,
        "role execution completion identity changed",
    )
    lock_record, lock_payload = _bound_regular_file(
        lock_path, role="execution completion lock", required_mode=0o400
    )
    _require(completion.get("lock") == lock_record, "execution lock binding changed")
    ordered = completion.get("ordered_case_names")
    _require(
        isinstance(ordered, list)
        and all(isinstance(case_name, str) and case_name for case_name in ordered),
        "execution completion case order changed",
    )
    if expected_ordered_case_names is None:
        lock = _json_object(lock_payload, role="execution completion lock")
        key = (
            "calibration_case_whitelist"
            if expected_role == "calibration"
            else "case_whitelist"
        )
        expected_ordered_case_names = lock.get(key)
        _require(
            lock.get("protocol_id") == PROTOCOL_ID
            and lock.get("stage") == expected_role
            and isinstance(expected_ordered_case_names, list),
            "execution completion lock role or cohort changed",
        )
    _require(
        ordered == list(expected_ordered_case_names),
        "execution completion ordered cohort changed",
    )
    role_root = held_root / expected_role
    source_manifest = _validate_artifact_binding(
        completion.get("source_manifest"),
        expected_path=canonical_role_source_manifest_path(held_root, expected_role),
        role="execution role source manifest",
    )
    _require(
        source_manifest.get("protocol_id") == PROTOCOL_ID,
        "execution role source manifest protocol changed",
    )
    evidence = _validate_artifact_binding(
        completion.get("score_evidence"),
        expected_path=role_root / f"{expected_role}-score-evidence.json",
        role="execution score evidence",
    )
    decision_record = completion.get("semantic_decision")
    _require(
        isinstance(decision_record, Mapping)
        and set(decision_record)
        == {
            "path",
            "sha256",
            "size_bytes",
            "artifact_sha256",
            "semantic_outcome",
            "semantic_return_code",
        },
        "semantic decision record fields changed",
    )
    decision = _validate_artifact_binding(
        {
            key: decision_record.get(key)
            for key in ("path", "sha256", "size_bytes", "artifact_sha256")
        },
        expected_path=role_root / f"{expected_role}-gate-decision.json",
        role="execution semantic decision",
    )
    semantic_outcome = decision_record.get("semantic_outcome")
    semantic_rc = _semantic_return_code(expected_role, semantic_outcome)
    _require(
        decision.get("decision") == semantic_outcome
        and decision_record.get("semantic_return_code") == semantic_rc
        and evidence.get("gate_result") == decision.get("gate_result"),
        "semantic outcome differs from the sealed evidence or decision",
    )
    resource_boundary = completion.get("resource_boundary")
    _validate_execution_resource_boundary(resource_boundary, ordered_case_names=ordered)
    _require(
        resource_boundary["parent_writable_fd_guard"]["held_root"] == str(held_root),
        "parent writable-FD guard binds another held root",
    )
    _require(
        completion.get("publication_contract")
        == {
            "marker_created_with_o_excl": True,
            "marker_and_parent_directory_descriptors_open_during_final_boundary": True,
            "marker_fsync_before_seal": True,
            "marker_mode_octal": "0400",
            "parent_directory_fsync_before_return": True,
            "partial_marker_unlinked_on_publication_failure": True,
        }
        and completion.get("self_hash_contract")
        == "artifact_sha256-is-sha256-of-canonical-json-with-artifact_sha256-omitted-v1"
        and completion.get("artifact_sha256")
        == _execution_completion_artifact_sha256(completion),
        "execution completion publication contract or self-hash changed",
    )
    return completion


def _publish_role_execution_completion(
    layout: OutcomeLayout,
    *,
    lock_path: str | Path,
    source_manifest_path: str | Path,
    ordered_case_names: Sequence[str],
    evidence: Mapping[str, Any],
    decision: Mapping[str, Any],
    semantic_return_code: int,
    resource_boundary: Mapping[str, Any],
    fd_counter: Callable[[], int],
    rlimit_nofile_getter: Callable[[], tuple[int, int]],
    rlimit_nofile_reference: tuple[int, int],
) -> dict[str, Any]:
    path = layout.execution_completion_path
    _require(
        path == canonical_role_execution_completion_path(layout.root, layout.role)
        and not os.path.lexists(path),
        "canonical role execution completion is not fresh",
    )
    lock_record, _ = _bound_regular_file(
        lock_path, role="execution completion lock", required_mode=0o400
    )
    evidence_record = _artifact_file_record(
        layout.evidence_path, evidence, role="execution score evidence"
    )
    decision_record = _artifact_file_record(
        layout.decision_path, decision, role="execution semantic decision"
    )
    source_path = Path(os.path.abspath(os.fspath(source_manifest_path)))
    _require(
        source_path == canonical_role_source_manifest_path(layout.root, layout.role),
        "execution source manifest path changed",
    )
    _, source_payload = _bound_regular_file(
        source_path, role="execution source manifest", required_mode=0o400
    )
    source_artifact = _json_object(source_payload, role="execution source manifest")
    source_record = _artifact_file_record(
        source_path, source_artifact, role="execution source manifest"
    )
    semantic_outcome = decision.get("decision")
    _require(
        semantic_return_code == _semantic_return_code(layout.role, semantic_outcome),
        "semantic return code changed before execution completion",
    )
    parent_before = os.lstat(path.parent)
    _require(
        stat.S_ISDIR(parent_before.st_mode)
        and not stat.S_ISLNK(parent_before.st_mode)
        and path.parent.resolve() == path.parent,
        "execution completion parent is absent, linked, or non-canonical",
    )
    parent_descriptor = os.open(
        path.parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    marker_descriptor: int | None = None
    marker_identity: tuple[int, int] | None = None
    try:
        parent_opened = os.fstat(parent_descriptor)
        _require(
            (parent_opened.st_dev, parent_opened.st_ino)
            == (parent_before.st_dev, parent_before.st_ino),
            "execution completion parent changed while opening",
        )
        marker_descriptor = os.open(
            path.name,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_descriptor,
        )
        marker_opened = os.fstat(marker_descriptor)
        _require(
            stat.S_ISREG(marker_opened.st_mode) and marker_opened.st_nlink == 1,
            "new execution completion is not a single-link regular file",
        )
        marker_identity = (marker_opened.st_dev, marker_opened.st_ino)
        publication_rlimit = _validate_qualified_rlimit_nofile(
            rlimit_nofile_getter(),
            reference=rlimit_nofile_reference,
            phase="execution completion publication boundary",
        )
        publication_fd_count = fd_counter()
        _require(
            type(publication_fd_count) is int and publication_fd_count > 0,
            "execution completion publication FD census is invalid",
        )
        end_observation = resource_boundary.get("end_outcome")
        pre_observation = resource_boundary.get("pre_outcome")
        _require(
            isinstance(end_observation, Mapping)
            and isinstance(pre_observation, Mapping)
            and type(end_observation.get("file_descriptor_count")) is int
            and type(pre_observation.get("file_descriptor_count")) is int
            and publication_fd_count == end_observation["file_descriptor_count"] + 2
            and publication_fd_count
            <= pre_observation["file_descriptor_count"] + POST_CASE_FD_GROWTH_LIMIT,
            "execution completion descriptors violate the final FD boundary",
        )
        completed_resource_boundary = {
            **resource_boundary,
            "publication": {
                "file_descriptor_count": publication_fd_count,
                "reference_file_descriptor_count": pre_observation[
                    "file_descriptor_count"
                ],
                "maximum_growth": POST_CASE_FD_GROWTH_LIMIT,
                "rlimit_nofile_soft": publication_rlimit[0],
                "rlimit_nofile_hard": publication_rlimit[1],
                "marker_fd_open": True,
                "parent_directory_fd_open": True,
                "open_descriptor_delta_from_end": 2,
            },
        }
        completion: dict[str, Any] = {
            "schema_version": 1,
            "artifact_kind": ROLE_EXECUTION_COMPLETION_KIND,
            "protocol_id": PROTOCOL_ID,
            "status": ROLE_EXECUTION_COMPLETION_STATUS,
            "role": layout.role,
            "lock": lock_record,
            "ordered_case_names": list(ordered_case_names),
            "source_manifest": source_record,
            "score_evidence": evidence_record,
            "semantic_decision": {
                **decision_record,
                "semantic_outcome": semantic_outcome,
                "semantic_return_code": semantic_return_code,
            },
            "resource_boundary": completed_resource_boundary,
            "publication_contract": {
                "marker_created_with_o_excl": True,
                "marker_and_parent_directory_descriptors_open_during_final_boundary": True,
                "marker_fsync_before_seal": True,
                "marker_mode_octal": "0400",
                "parent_directory_fsync_before_return": True,
                "partial_marker_unlinked_on_publication_failure": True,
            },
            "self_hash_contract": (
                "artifact_sha256-is-sha256-of-canonical-json-with-"
                "artifact_sha256-omitted-v1"
            ),
        }
        completion["artifact_sha256"] = _execution_completion_artifact_sha256(
            completion
        )
        encoded = _canonical_json_bytes(completion)
        offset = 0
        while offset < len(encoded):
            written = os.write(marker_descriptor, encoded[offset:])
            _require(written > 0, "execution completion write made no progress")
            offset += written
        os.fsync(marker_descriptor)
        os.fchmod(marker_descriptor, 0o400)
        os.fsync(marker_descriptor)
        sealed_state = os.fstat(marker_descriptor)
        current = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        _require(
            (sealed_state.st_dev, sealed_state.st_ino) == marker_identity
            and (current.st_dev, current.st_ino) == marker_identity
            and stat.S_IMODE(sealed_state.st_mode) == 0o400
            and sealed_state.st_size == len(encoded),
            "execution completion changed before publication",
        )
        os.lseek(marker_descriptor, 0, os.SEEK_SET)
        observed_payload = bytearray()
        while block := os.read(marker_descriptor, 1024 * 1024):
            observed_payload.extend(block)
        _require(
            bytes(observed_payload) == encoded, "execution completion write changed"
        )
        os.fsync(parent_descriptor)
        validated = validate_role_execution_completion(
            path,
            lock_path=lock_path,
            expected_role=layout.role,
            expected_ordered_case_names=ordered_case_names,
        )
        _require(validated == completion, "execution completion validation changed")
        return completion
    except BaseException:
        if marker_descriptor is not None and marker_identity is not None:
            try:
                current = os.stat(
                    path.name, dir_fd=parent_descriptor, follow_symlinks=False
                )
            except FileNotFoundError:
                current = None
            if (
                current is not None
                and (current.st_dev, current.st_ino) == marker_identity
            ):
                os.unlink(path.name, dir_fd=parent_descriptor)
                os.fsync(parent_descriptor)
        raise
    finally:
        if marker_descriptor is not None:
            os.close(marker_descriptor)
        os.close(parent_descriptor)


def build_role_outcome_sealer_subprocess(
    arguments: DriverArguments,
) -> tuple[tuple[str, ...], dict[str, str], str]:
    deployed_code = Path(os.path.abspath(arguments.deployed_code))
    sealer = deployed_code / "scripts" / "held" / "seal_deform360_v8_role_outcome.py"
    argv = (
        str(PINNED_PYTHON),
        "-I",
        "-B",
        "-X",
        f"pycache_prefix={PYCACHE_PREFIX}",
        str(sealer),
        "--role",
        arguments.role,
        "--lock",
        str(Path(os.path.abspath(arguments.lock_path))),
        "--deployed-code",
        str(deployed_code),
    )
    return argv, _normalized_environment(), str(deployed_code)


def run_role_outcome_sealer(arguments: DriverArguments) -> None:
    root = Path(os.path.abspath(arguments.lock_path)).parent
    marker = canonical_role_execution_completion_path(root, arguments.role)
    completion = validate_role_execution_completion(
        marker,
        lock_path=arguments.lock_path,
        expected_role=arguments.role,
    )
    initial = completion["resource_boundary"]["initial_nofile"]
    _validate_qualified_rlimit_nofile(
        _rlimit_nofile_pair(),
        reference=(
            initial["rlimit_nofile_soft"],
            initial["rlimit_nofile_hard"],
        ),
        phase="immediately before role outcome integrity sealer",
    )
    _validate_no_writable_held_descriptors(root)
    argv, environment, cwd = build_role_outcome_sealer_subprocess(arguments)
    completed = subprocess.run(
        argv,
        check=False,
        env=environment,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True,
    )
    _require(
        completed.returncode == 0 and completed.stderr == b"",
        f"role outcome integrity sealer failed with code {completed.returncode}",
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    _require(len(lines) == 1, "role outcome integrity sealer output changed")
    try:
        result = json.loads(lines[0].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("role outcome integrity sealer output is invalid") from error
    _require(
        isinstance(result, Mapping)
        and result.get("event") == "DEFORM360_V8_ROLE_OUTCOME_INTEGRITY_COMPLETE"
        and result.get("role") == arguments.role
        and result.get("terminal_outcome")
        == completion["semantic_decision"]["semantic_outcome"]
        and result.get("role_completion_path")
        == str(
            root
            / arguments.role
            / f"{arguments.role}-outcome-integrity-completion.json"
        )
        and _valid_sha256(result.get("role_completion_artifact_sha256"))
        and result.get("terminal_root_finalized")
        is (completion["semantic_decision"]["semantic_outcome"] != "GO"),
        "role outcome integrity sealer completion event changed",
    )


def build_query_subprocess(
    *,
    deployed_code: str | Path,
    lock_path: str | Path,
    official_query_manifest_path: str | Path,
    frozen_field_manifest_path: str | Path,
    output_archive_path: str | Path,
    output_seal_path: str | Path,
) -> tuple[tuple[str, ...], dict[str, str], str]:
    """Return the isolated query argv/env; no future path is accepted."""

    code = Path(os.path.abspath(os.fspath(deployed_code)))
    worker = code / "scripts" / "held" / "run_deform360_v8_x0_query.py"
    argv = (
        str(PINNED_PYTHON),
        "-I",
        "-B",
        "-X",
        f"pycache_prefix={PYCACHE_PREFIX}",
        str(worker),
        "--lock",
        os.fspath(lock_path),
        "--official-query-manifest",
        os.fspath(official_query_manifest_path),
        "--frozen-field-manifest",
        os.fspath(frozen_field_manifest_path),
        "--output-archive",
        os.fspath(output_archive_path),
        "--output-seal",
        os.fspath(output_seal_path),
    )
    environment = {
        "HOME": "/home/florianpfaff",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "LOGNAME": "florianpfaff",
        "PATH": PINNED_PATH,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPYCACHEPREFIX": PYCACHE_PREFIX,
        "PYTHONSAFEPATH": "1",
        "TMPDIR": "/tmp",
        "USER": "florianpfaff",
    }
    safe_cwd = str(Path(os.path.abspath(os.fspath(output_seal_path))).parent)
    return argv, environment, safe_cwd


def run_query_subprocess(**kwargs: Any) -> None:
    argv, environment, safe_cwd = build_query_subprocess(**kwargs)
    subprocess.run(
        argv,
        check=True,
        env=environment,
        cwd=safe_cwd,
        stdin=subprocess.DEVNULL,
        close_fds=True,
    )


def run_process_isolated_reconstruction(
    *,
    arguments: DriverArguments,
    paths: CasePaths,
    aligned_episode_dir: str | Path,
    cohort_barrier_sha256: str,
) -> Mapping[str, Any]:
    """Run one original-trainer reconstruction in a fresh child process."""

    from bayesian_phystwin import deform360_case_process_isolation as isolation

    required = {
        "deform360_repo": arguments.deform360_repo,
        "sam2_repository": arguments.sam2_repository,
        "sam2_checkpoint": arguments.sam2_checkpoint,
        "cotracker_repo": arguments.cotracker_repo,
        "cotracker_checkpoint": arguments.cotracker_checkpoint,
    }
    missing = sorted(name for name, value in required.items() if value is None)
    _require(not missing, f"isolated reconstruction runtime is missing: {missing}")
    return isolation.run_isolated_reconstruction_subprocess(
        stdout_log_path=paths.isolated_reconstruction_stdout,
        stderr_log_path=paths.isolated_reconstruction_stderr,
        python_executable=PINNED_PYTHON,
        deployed_code=arguments.deployed_code,
        lock_path=arguments.lock_path,
        role=arguments.role,
        case_name=paths.case_name,
        online_prediction_seal_path=paths.online_seal,
        aligned_episode_dir=aligned_episode_dir,
        reconstruction_output_dir=paths.reconstruction_dir,
        result_archive_path=paths.isolated_reconstruction_archive,
        result_manifest_path=paths.isolated_reconstruction_manifest,
        cohort_barrier_sha256=cohort_barrier_sha256,
        deform360_repo=str(required["deform360_repo"]),
        sam2_repository=str(required["sam2_repository"]),
        sam2_checkpoint=str(required["sam2_checkpoint"]),
        cotracker_repo=str(required["cotracker_repo"]),
        cotracker_checkpoint=str(required["cotracker_checkpoint"]),
        device=arguments.device,
        ffmpeg=arguments.ffmpeg,
        pycache_prefix=PYCACHE_PREFIX,
        path_environment=PINNED_PATH,
    )


def _source_verifier_environment() -> dict[str, str]:
    return {
        "GIT_OPTIONAL_LOCKS": "0",
        "HOME": "/home/florianpfaff",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "LOGNAME": "florianpfaff",
        "PATH": PINNED_PATH,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPYCACHEPREFIX": PYCACHE_PREFIX,
        "PYTHONSAFEPATH": "1",
        "TMPDIR": "/tmp",
        "USER": "florianpfaff",
    }


def run_source_only_deployment_verifier(arguments: DriverArguments) -> None:
    wrapper = (
        Path(arguments.deployed_code)
        / "scripts"
        / "held"
        / f"run_deform360_v8_{arguments.role}_outcomes.py"
    )
    subprocess.run(
        [
            str(PINNED_PYTHON),
            "-I",
            "-B",
            "-X",
            f"pycache_prefix={PYCACHE_PREFIX}",
            str(wrapper),
            "--source-only-verifier",
            "--deployed-code",
            arguments.deployed_code,
            "--lock",
            arguments.lock_path,
        ],
        check=True,
        env=_source_verifier_environment(),
    )


def verify_source_only_deployment(
    *,
    role: Literal["calibration", "confirmation"],
    deployed_code: str | Path,
    lock_path: str | Path,
    protocol: Any,
) -> None:
    """Verify only immutable source/lock state; never inspect a case payload."""

    _require(
        not os.path.lexists("/nonexistent") and not os.path.lexists(PYCACHE_PREFIX),
        "reserved held-v8 bytecode prefix is available",
    )
    root = CANONICAL_HELD_ROOT
    code = Path(os.path.abspath(os.fspath(deployed_code)))
    lock = Path(os.path.abspath(os.fspath(lock_path)))
    _require(
        root.is_dir() and not root.is_symlink() and root.resolve() == root,
        "held-v8 root is absent, linked, or non-canonical",
    )
    _require(
        code.is_dir()
        and not code.is_symlink()
        and code.resolve() == code
        and code.parent == root
        and re.fullmatch(r"code-[0-9a-f]{40}|code-[0-9a-f]{64}", code.name) is not None,
        "deployed code is outside the canonical immutable held-v8 snapshot",
    )
    expected_lock = root / f"{role}-lock.json"
    _require(lock == expected_lock, "source verifier received a non-canonical lock")
    _require_regular_mode(lock, 0o400, "held-v8 role lock")
    artifact = protocol.validate_protocol_lock(lock)
    _require(
        artifact.get("protocol_id") == PROTOCOL_ID and artifact.get("stage") == role,
        "source verifier received another protocol or role",
    )
    bindings = artifact.get("immutable_bindings")
    _require(isinstance(bindings, Mapping), "lock has no immutable bindings")
    repository_evidence = _validate_deployed_repository(code, bindings)
    required = dict(_DEPLOYMENT_BINDINGS)
    required[f"held_v8_{role}_outcome_driver_source"] = (
        f"scripts/held/run_deform360_v8_{role}_outcomes.py"
    )
    for name, relative in required.items():
        path = code / relative
        _require(
            path.is_file() and not path.is_symlink() and path.resolve() == path,
            f"deployed source is absent or linked: {relative}",
        )
        _require(
            bindings.get(name) == _sha256_file(path),
            f"deployed source differs from lock: {name}",
        )
    _require(
        bindings.get("held_v8_confirmation_source_contract")
        == protocol.confirmation_source.confirmation_source_contract_sha256(),
        "confirmation source contract differs from the lock",
    )
    for path in code.rglob("*"):
        observed = os.lstat(path)
        _require(not stat.S_ISLNK(observed.st_mode), "deployed tree contains a link")
        _require(
            stat.S_IMODE(observed.st_mode) & 0o222 == 0,
            f"deployed tree is writable: {path}",
        )
    _emit(
        "SOURCE_ONLY_DEPLOYMENT_VERIFIED",
        role=role,
        lock_file_sha256=_sha256_file(lock),
        deployed_head=repository_evidence["head"],
        deployed_tree_sha256=repository_evidence["tree_sha256"],
        source_count=len(required),
        current_role_case_or_future_payload_inspected=False,
        calibration_go_lineage_validated=role == "confirmation",
    )


def _physical_paths(layout: OutcomeLayout) -> dict[str, Path]:
    return {case: paths.physical_seal for case, paths in layout.cases.items()}


def _online_paths(layout: OutcomeLayout) -> dict[str, Path]:
    return {case: paths.online_seal for case, paths in layout.cases.items()}


def _field_paths(layout: OutcomeLayout) -> dict[str, Path]:
    return {case: paths.frozen_field_manifest for case, paths in layout.cases.items()}


def _query_paths(layout: OutcomeLayout) -> dict[str, Path]:
    return {case: paths.official_query_manifest for case, paths in layout.cases.items()}


def _queried_paths(layout: OutcomeLayout) -> dict[str, Path]:
    return {case: paths.queried_seal for case, paths in layout.cases.items()}


def _prepare_fresh_outputs(layout: OutcomeLayout) -> None:
    _require(
        layout.role_root.is_dir()
        and not layout.role_root.is_symlink()
        and layout.role_root.resolve() == layout.role_root,
        "role output root is absent, linked, or non-canonical",
    )
    claim = layout.role_root / ".v8-outcome-phase.claim"
    for path, role in (
        (claim, "outcome claim"),
        (layout.protected_future_root, "protected future root"),
        (layout.x0_query_root, "frame-zero query root"),
        (layout.queried_prediction_root, "queried prediction root"),
        (layout.evidence_path, "score evidence"),
        (layout.decision_path, "gate decision"),
        (layout.execution_completion_path, "role execution completion"),
    ):
        _require(not os.path.lexists(path), f"fresh {role} already exists: {path}")
    os.mkdir(claim, mode=0o500)
    os.mkdir(layout.protected_future_root, mode=0o700)
    os.mkdir(layout.x0_query_root, mode=0o700)
    os.mkdir(layout.queried_prediction_root, mode=0o700)
    for case_name, paths in layout.cases.items():
        case_directories = (
            paths.target_manifest.parent,
            paths.official_query_manifest.parent,
            paths.queried_seal.parent,
        )
        _require(
            len(set(case_directories)) == 3,
            f"case artifact roots are not disjoint: {case_name}",
        )
        for directory in case_directories:
            _require(
                not os.path.lexists(directory),
                f"fresh per-case artifact root already exists: {directory}",
            )
            os.mkdir(directory, mode=0o700)
            observed = os.lstat(directory)
            _require(
                stat.S_ISDIR(observed.st_mode)
                and not stat.S_ISLNK(observed.st_mode)
                and stat.S_IMODE(observed.st_mode) == 0o700
                and directory.resolve() == directory,
                f"per-case artifact root is unsafe: {directory}",
            )


def _validate_runtime(arguments: DriverArguments) -> None:
    _require(
        socket.gethostname() == "workstation2",
        "formal held-v8 outcomes must run on gpuserver6000/workstation2",
    )
    _require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "0" and arguments.device == "cuda:0",
        "formal held-v8 outcomes are bound to physical GPU 0",
    )
    _require(arguments.ffmpeg == "ffmpeg", "formal ffmpeg spelling changed")
    _require(
        arguments.aligned_root is not None
        and Path(arguments.aligned_root) == CANONICAL_ALIGNED_ROOT
        and CANONICAL_ALIGNED_ROOT.is_dir()
        and not CANONICAL_ALIGNED_ROOT.is_symlink()
        and CANONICAL_ALIGNED_ROOT.resolve() == CANONICAL_ALIGNED_ROOT,
        "aligned target root is not the canonical pinned root",
    )
    required = {
        "deform360_repo": arguments.deform360_repo,
        "sam2_repository": arguments.sam2_repository,
        "sam2_checkpoint": arguments.sam2_checkpoint,
        "cotracker_repo": arguments.cotracker_repo,
        "cotracker_checkpoint": arguments.cotracker_checkpoint,
    }
    missing = sorted(name for name, value in required.items() if not value)
    _require(not missing, f"outcome runtime paths are missing: {missing}")
    for name in ("deform360_repo", "sam2_repository", "cotracker_repo"):
        path = Path(str(required[name]))
        _require(
            path.is_dir() and not path.is_symlink() and path.resolve() == path,
            f"pinned runtime directory is absent or linked: {name}",
        )
    for name in ("sam2_checkpoint", "cotracker_checkpoint"):
        path = Path(str(required[name]))
        _require(
            path.is_file() and not path.is_symlink() and path.resolve() == path,
            f"pinned runtime file is absent or linked: {name}",
        )


def _aligned_episode_path(
    arguments: DriverArguments,
    protocol: Any,
    post: PostBarrierApi,
    case_name: str,
) -> Path:
    if arguments.role == "confirmation":
        _require(
            arguments.confirmation_source_manifest_path is not None,
            "confirmation source cohort is unavailable",
        )
        manifest = post.validate_confirmation_source(
            arguments.confirmation_source_manifest_path,
            expected_source_permit=protocol.confirmation_source_permit_evidence(
                arguments.lock_path
            ),
            verify_content=True,
        )
        matches = [
            record
            for record in manifest["cases"]
            if record.get("case_name") == case_name
        ]
        _require(
            len(matches) == 1,
            "confirmation source lacks one exact aligned episode",
        )
        return Path(str(manifest["source_root"])) / str(
            matches[0]["aligned_episode_relative_path"]
        )
    if case_name == protocol.FRESH_REPLACEMENT_CASE_NAME:
        _require(
            arguments.role == "calibration"
            and arguments.replacement_source_manifest_path is not None,
            "fresh replacement source is unavailable",
        )
        manifest = post.validate_replacement_source(
            arguments.replacement_source_manifest_path,
            expected_source_permit=protocol.replacement_source_permit_evidence(
                arguments.lock_path
            ),
        )
        return Path(str(manifest["aligned_episode_dir"]))
    object_id, separator, episode = case_name.rpartition("-ep")
    _require(separator == "-ep" and len(episode) == 4, "locked case is malformed")
    _require(
        arguments.role == "calibration",
        "confirmation may not fall back to the shared aligned dataset",
    )
    return Path(str(arguments.aligned_root)) / object_id / f"episode_{episode}"


def _source_node_count(field_manifest: Mapping[str, Any]) -> int:
    records = field_manifest.get("source_array_records")
    _require(isinstance(records, Mapping), "frozen field source records are absent")
    record = records.get("frame_zero_points_m")
    _require(isinstance(record, Mapping), "frozen field has no source point record")
    shape = record.get("shape")
    _require(
        isinstance(shape, list)
        and len(shape) == 2
        and isinstance(shape[0], int)
        and shape[0] > 0,
        "frozen source point count is invalid",
    )
    return int(shape[0])


def execute_outcomes(
    arguments: DriverArguments,
    *,
    protocol: Any,
    deployment_verifier: Callable[[DriverArguments], None],
    smoke_gsplat_runtime: Callable[[], Mapping[str, Any]],
    load_post_barrier_api: Callable[[], PostBarrierApi],
    query_runner: Callable[..., None] = run_query_subprocess,
    reconstruction_runner: Callable[..., Mapping[str, Any]] | None = None,
    validate_runtime: Callable[[DriverArguments], None] = _validate_runtime,
    fd_counter: Callable[[], int] = _open_file_descriptor_count,
    rlimit_nofile_getter: Callable[[], tuple[int, int]] = _rlimit_nofile_pair,
    role_sealer: Callable[[DriverArguments], None] = run_role_outcome_sealer,
    formal_paths: bool = True,
) -> int:
    """Execute the fresh two-barrier outcome flow with injectable operators."""

    _require(protocol.PROTOCOL_ID == PROTOCOL_ID, "deployed protocol is not v8")
    rlimit_nofile_reference: tuple[int, int] | None = None
    resource_boundary: dict[str, Any] | None = None
    if not arguments.dry_run_barrier_only:
        rlimit_nofile_reference = _validate_qualified_rlimit_nofile(
            rlimit_nofile_getter(), phase="initial outcome"
        )
        resource_boundary = {
            "qualified_rlimit_nofile_soft": QUALIFIED_RLIMIT_NOFILE_SOFT,
            "post_case_fd_growth_limit": POST_CASE_FD_GROWTH_LIMIT,
            "initial_nofile": {
                "rlimit_nofile_soft": rlimit_nofile_reference[0],
                "rlimit_nofile_hard": rlimit_nofile_reference[1],
            },
            "pre_outcome": None,
            "post_cases": [],
            "end_outcome": None,
            "parent_writable_fd_guard": None,
        }
        _emit(
            "QUALIFIED_RLIMIT_NOFILE_CAPTURED",
            soft_limit=rlimit_nofile_reference[0],
            hard_limit=rlimit_nofile_reference[1],
            qualified_soft_limit=QUALIFIED_RLIMIT_NOFILE_SOFT,
        )
    deployment_verifier(arguments)
    _emit("SOURCE_ONLY_VERIFIER_COMPLETE", role=arguments.role)
    lock = protocol.validate_protocol_lock(arguments.lock_path)
    _require(lock.get("stage") == arguments.role, "lock stage and driver role differ")
    expected = tuple(
        protocol.locked_case_names(arguments.lock_path, role=arguments.role)
    )
    expected_count = 15 if arguments.role == "calibration" else 6
    _require(len(expected) == expected_count, "locked cohort cardinality changed")
    if arguments.role == "calibration":
        _require(
            arguments.replacement_source_manifest_path is not None,
            "calibration requires the exact replacement-source manifest",
        )
        _require(
            arguments.confirmation_source_manifest_path is None,
            "calibration may not accept a confirmation source manifest",
        )
    else:
        _require(
            arguments.replacement_source_manifest_path is None,
            "confirmation may not accept a replacement-source manifest",
        )
        _require(
            arguments.confirmation_source_manifest_path is not None,
            "confirmation requires the exact six-case source manifest",
        )
    root = CANONICAL_HELD_ROOT if formal_paths else Path(arguments.lock_path).parent
    layout = build_layout(root=root, role=arguments.role, cases=expected)

    smoke = smoke_gsplat_runtime()
    smoke_sha = smoke.get("artifact_sha256") if isinstance(smoke, Mapping) else None
    _require(_valid_sha256(smoke_sha), "gsplat runtime smoke returned bad evidence")
    if formal_paths:
        _require(
            smoke_sha == _locked_gpu0_smoke_artifact(lock),
            "live gsplat smoke differs from locked physical-GPU-0 evidence",
        )
    _emit("GSPLAT_RUNTIME_SMOKE_VALIDATED", artifact_sha256=smoke_sha)

    barrier_one_kwargs = {
        "physical_seal_paths": _physical_paths(layout),
        "online_seal_paths": _online_paths(layout),
        "frozen_field_manifest_paths": _field_paths(layout),
        "replacement_aligned_source_manifest_path": (
            arguments.replacement_source_manifest_path
        ),
        "confirmation_aligned_source_manifest_path": (
            arguments.confirmation_source_manifest_path
        ),
        "role": arguments.role,
    }
    barrier_one = protocol.validate_first_cohort_barrier(
        arguments.lock_path, **barrier_one_kwargs
    )
    target_capabilities = protocol.authorize_target_reconstruction_capabilities(
        arguments.lock_path, **barrier_one_kwargs
    )
    _require(
        set(target_capabilities) == set(expected), "target capability cohort changed"
    )
    _emit(
        "FIRST_COHORT_BARRIER_VALIDATED",
        role=arguments.role,
        case_count=len(expected),
        barrier_sha256=barrier_one.barrier_sha256,
    )
    if arguments.dry_run_barrier_only:
        _emit("DRY_RUN_STOPPED_AFTER_FIRST_BARRIER", target_created=False)
        return 0

    if formal_paths:
        bindings = lock.get("immutable_bindings")
        _require(
            isinstance(bindings, Mapping)
            and _RECONSTRUCTION_RUNTIME_BINDINGS.issubset(bindings),
            "held-v8 lock lacks the inherited pinned reconstruction runtime",
        )
    validate_runtime(arguments)
    post = load_post_barrier_api()
    _prepare_fresh_outputs(layout)
    backend = (
        post.backend_type(
            deform360_repo=arguments.deform360_repo,
            sam2_repository=arguments.sam2_repository,
            sam2_checkpoint=arguments.sam2_checkpoint,
            cotracker_repo=arguments.cotracker_repo,
            cotracker_checkpoint=arguments.cotracker_checkpoint,
            device=arguments.device,
            ffmpeg=arguments.ffmpeg,
        )
        if reconstruction_runner is None
        else None
    )
    lock_sha256 = _sha256_file(arguments.lock_path)
    consumed_target_evidence: dict[str, Mapping[str, Any]] = {}
    post_case_fd_reference = fd_counter()
    _require(
        type(post_case_fd_reference) is int and post_case_fd_reference > 0,
        "pre-outcome file-descriptor census is invalid",
    )
    _require(
        rlimit_nofile_reference is not None,
        "full outcome execution lacks an RLIMIT_NOFILE reference",
    )
    pre_outcome_rlimit = _validate_qualified_rlimit_nofile(
        rlimit_nofile_getter(),
        reference=rlimit_nofile_reference,
        phase="pre-outcome boundary",
    )
    _require(resource_boundary is not None, "resource evidence was not initialized")
    resource_boundary["pre_outcome"] = {
        "file_descriptor_count": post_case_fd_reference,
        "rlimit_nofile_soft": pre_outcome_rlimit[0],
        "rlimit_nofile_hard": pre_outcome_rlimit[1],
    }
    _emit(
        "PRE_OUTCOME_RESOURCE_BOUNDARY_CAPTURED",
        file_descriptor_count=post_case_fd_reference,
        maximum_growth=POST_CASE_FD_GROWTH_LIMIT,
        rlimit_nofile_soft=pre_outcome_rlimit[0],
        rlimit_nofile_hard=pre_outcome_rlimit[1],
    )
    for case_index, case_name in enumerate(expected):
        paths = layout.cases[case_name]
        aligned = _aligned_episode_path(arguments, protocol, post, case_name)

        def consume_target(
            permit: object,
            *,
            case_name: str,
            operation: str,
        ) -> Mapping[str, Any]:
            evidence = protocol.consume_case_capability(
                permit, case_name=case_name, operation=operation
            )
            consumed_target_evidence[case_name] = dict(evidence)
            return evidence

        def reconstruct(
            *,
            case_name: str = case_name,
            paths: CasePaths = paths,
            aligned: Path = aligned,
        ) -> Mapping[str, Any]:
            _require(
                case_name in consumed_target_evidence,
                "reconstruction loader ran before target capability consumption",
            )
            if reconstruction_runner is not None:
                return reconstruction_runner(
                    arguments=arguments,
                    paths=paths,
                    aligned_episode_dir=aligned,
                    cohort_barrier_sha256=barrier_one.barrier_sha256,
                )
            _require(backend is not None, "in-process backend was not constructed")
            return post.reconstruct(
                lock_path=arguments.lock_path,
                role=arguments.role,
                case_name=case_name,
                online_prediction_seal_path=paths.online_seal,
                aligned_episode_dir=aligned,
                output_dir=paths.reconstruction_dir,
                cohort_barrier_sha256=barrier_one.barrier_sha256,
                backend=backend,
            )

        post.write_target_and_query(
            paths.target_archive,
            paths.target_manifest,
            paths.official_query_archive,
            paths.official_query_manifest,
            lock_path=arguments.lock_path,
            lock_sha256=lock_sha256,
            case_name=case_name,
            role=arguments.role,
            target_reconstruction_permit=target_capabilities[case_name],
            consume_target_reconstruction_permit=consume_target,
            reconstruction_loader=reconstruct,
        )
        post.validate_target(
            paths.target_manifest,
            lock_path=arguments.lock_path,
            expected_case_name=case_name,
            expected_role=arguments.role,
        )
        _emit("OFFICIAL_TARGET_AND_X0_SEALED", case_name=case_name)

        # The x0 worker is launched immediately, before any later case target
        # is created.  Its paths live under disjoint safe roots and expose no
        # protected-future path or directory name.  The locked, audited worker
        # is the enforcement boundary; this is not an OS namespace boundary.
        query_runner(
            deployed_code=arguments.deployed_code,
            lock_path=arguments.lock_path,
            official_query_manifest_path=paths.official_query_manifest,
            frozen_field_manifest_path=paths.frozen_field_manifest,
            output_archive_path=paths.queried_archive,
            output_seal_path=paths.queried_seal,
        )
        post.validate_queried_prediction(
            paths.queried_seal,
            lock_path=arguments.lock_path,
            expected_case_name=case_name,
        )
        _emit("ISOLATED_X0_QUERY_SEALED", case_name=case_name)
        observed_fd_count = fd_counter()
        _require(
            type(observed_fd_count) is int and observed_fd_count > 0,
            "post-case file-descriptor census is invalid",
        )
        _require(
            observed_fd_count <= post_case_fd_reference + POST_CASE_FD_GROWTH_LIMIT,
            "post-case file-descriptor growth exceeded the frozen safety limit",
        )
        post_case_rlimit = _validate_qualified_rlimit_nofile(
            rlimit_nofile_getter(),
            reference=rlimit_nofile_reference,
            phase=f"post-case boundary for {case_name}",
        )
        post_case_observation = {
            "case_name": case_name,
            "case_index": case_index,
            "file_descriptor_count": observed_fd_count,
            "reference_file_descriptor_count": post_case_fd_reference,
            "maximum_growth": POST_CASE_FD_GROWTH_LIMIT,
            "rlimit_nofile_soft": post_case_rlimit[0],
            "rlimit_nofile_hard": post_case_rlimit[1],
        }
        resource_boundary["post_cases"].append(post_case_observation)
        _emit(
            "POST_CASE_RESOURCE_BOUNDARY_VALIDATED",
            **post_case_observation,
        )
    _require(
        tuple(consumed_target_evidence) == expected
        and all(
            evidence.get("single_use_consumed") is True
            and evidence.get("operation") == protocol.TARGET_RECONSTRUCTION_OPERATION
            for evidence in consumed_target_evidence.values()
        ),
        "not every reconstruction capability was consumed exactly once",
    )

    barrier_two_kwargs = {
        "official_query_manifest_paths": _query_paths(layout),
        "queried_prediction_seal_paths": _queried_paths(layout),
        "role": arguments.role,
    }
    barrier_two = protocol.validate_second_cohort_barrier(
        arguments.lock_path, **barrier_two_kwargs
    )
    future_capabilities = protocol.authorize_future_score_capabilities(
        arguments.lock_path, **barrier_two_kwargs
    )
    _require(
        set(future_capabilities) == set(expected), "score capability cohort changed"
    )
    _emit(
        "SECOND_COHORT_BARRIER_VALIDATED",
        role=arguments.role,
        case_count=len(expected),
        barrier_sha256=barrier_two.barrier_sha256,
    )

    records: dict[str, dict[str, Any]] = {}
    for case_name in expected:
        paths = layout.cases[case_name]
        field = post.validate_frozen_field(
            paths.frozen_field_manifest,
            lock_path=arguments.lock_path,
            expected_case_name=case_name,
        )
        inputs = post.load_scoring_inputs(
            case_name=case_name,
            queried_prediction_seal_path=paths.queried_seal,
            target_manifest_path=paths.target_manifest,
            lock_path=arguments.lock_path,
            future_score_permit=future_capabilities[case_name],
            consume_future_score_permit=protocol.consume_case_capability,
            source_node_count=_source_node_count(field),
        )
        record = post.score_case(**inputs.scoring_kwargs())
        record["future_score_permit_evidence"] = dict(inputs.permit_evidence)
        records[case_name] = record
    evidence, decision = post.create_score_evidence_and_decision(
        layout.evidence_path,
        layout.decision_path,
        lock_path=arguments.lock_path,
        role=arguments.role,
        barrier_two_sha256=barrier_two.barrier_sha256,
        case_records=records,
    )
    _emit(
        "SCORE_EVIDENCE_AND_DECISION_SEALED",
        role=arguments.role,
        evidence_artifact_sha256=evidence["artifact_sha256"],
        decision_artifact_sha256=decision["artifact_sha256"],
        decision=decision["decision"],
    )
    end_rlimit = _validate_qualified_rlimit_nofile(
        rlimit_nofile_getter(),
        reference=rlimit_nofile_reference,
        phase="end outcome boundary",
    )
    end_fd_count = fd_counter()
    _require(
        type(end_fd_count) is int and end_fd_count > 0,
        "end-outcome file-descriptor census is invalid",
    )
    _require(
        end_fd_count <= post_case_fd_reference + POST_CASE_FD_GROWTH_LIMIT,
        "end-outcome file-descriptor growth exceeded the frozen safety limit",
    )
    end_observation = {
        "file_descriptor_count": end_fd_count,
        "reference_file_descriptor_count": post_case_fd_reference,
        "maximum_growth": POST_CASE_FD_GROWTH_LIMIT,
        "rlimit_nofile_soft": end_rlimit[0],
        "rlimit_nofile_hard": end_rlimit[1],
    }
    resource_boundary["end_outcome"] = end_observation
    _emit(
        "END_OUTCOME_RESOURCE_BOUNDARY_VALIDATED",
        **end_observation,
    )
    semantic_return_code = _semantic_return_code(
        arguments.role, decision.get("decision")
    )
    resource_boundary["parent_writable_fd_guard"] = (
        _validate_no_writable_held_descriptors(layout.root)
    )
    completion = _publish_role_execution_completion(
        layout,
        lock_path=arguments.lock_path,
        source_manifest_path=(
            arguments.replacement_source_manifest_path
            if arguments.role == "calibration"
            else arguments.confirmation_source_manifest_path
        ),
        ordered_case_names=expected,
        evidence=evidence,
        decision=decision,
        semantic_return_code=semantic_return_code,
        resource_boundary=resource_boundary,
        fd_counter=fd_counter,
        rlimit_nofile_getter=rlimit_nofile_getter,
        rlimit_nofile_reference=rlimit_nofile_reference,
    )
    _emit(
        "ROLE_EXECUTION_COMPLETION_PUBLISHED",
        role=arguments.role,
        semantic_outcome=decision["decision"],
        semantic_return_code=semantic_return_code,
        path=str(layout.execution_completion_path),
        artifact_sha256=completion["artifact_sha256"],
    )
    _emit(
        "ROLE_OUTCOME_INTEGRITY_SEALER_STARTING",
        role=arguments.role,
        semantic_outcome=decision["decision"],
        semantic_return_code=semantic_return_code,
    )
    if semantic_return_code == NO_GO_EXIT_CODE:
        _emit("CALIBRATION_NO_GO_CONFIRMATION_REMAINS_INACCESSIBLE")
    # The sealer may freeze the entire held root.  It captures its own output
    # into memory, and this parent performs no output or filesystem work after
    # the call succeeds.
    role_sealer(arguments)
    return semantic_return_code


def _load_protocol(deployed_code: str) -> tuple[Any, Path]:
    code = Path(deployed_code).resolve()
    source = code / "src"
    _require(source.is_dir() and not source.is_symlink(), "deployed source is invalid")
    preloaded = sorted(
        name
        for name in sys.modules
        if name == "bayesian_phystwin" or name.startswith("bayesian_phystwin.")
    )
    forbidden_preloaded = {
        "bayesian_phystwin.deform360_held_outcome_reconstruction",
        "bayesian_phystwin.deform360_held_outcome_scoring",
        "bayesian_phystwin.deform360_held_v8_outcome_artifacts",
        "bayesian_phystwin.deform360_held_v8_outcome_reconstruction",
        "bayesian_phystwin.deform360_held_v8_score_artifacts",
        "bayesian_phystwin.deform360_held_v8_scoring",
    }
    _require(
        set(preloaded).isdisjoint(forbidden_preloaded),
        f"future-bearing module was preloaded before barrier one: {preloaded}",
    )
    sys.path.insert(0, str(source))
    from bayesian_phystwin import deform360_held_v8_protocol as protocol

    expected = source / "bayesian_phystwin" / "deform360_held_v8_protocol.py"
    _require(Path(protocol.__file__).resolve() == expected, "protocol source escaped")
    return protocol, source


def _load_post_barrier_api(source: Path) -> PostBarrierApi:
    from bayesian_phystwin import deform360_held_outcome_reconstruction as numerical
    from bayesian_phystwin import deform360_held_v8_outcome_artifacts as outcomes
    from bayesian_phystwin import deform360_held_v8_outcome_reconstruction as adapter
    from bayesian_phystwin import deform360_held_v8_query_artifacts as queries
    from bayesian_phystwin import (
        deform360_held_v8_confirmation_source as confirmation_source,
    )
    from bayesian_phystwin import deform360_held_v8_replacement_source as replacement
    from bayesian_phystwin import deform360_held_v8_score_artifacts as score_artifacts
    from bayesian_phystwin import deform360_held_v8_scoring as scoring

    modules = (
        numerical,
        outcomes,
        adapter,
        queries,
        confirmation_source,
        replacement,
        score_artifacts,
        scoring,
    )
    for module in modules:
        path = Path(module.__file__).resolve()
        _require(
            source in path.parents, f"post-barrier module escaped deployment: {path}"
        )
    return PostBarrierApi(
        backend_type=numerical.PinnedOfficialPipelineBackend,
        reconstruct=adapter.reconstruct_fresh_official_target,
        write_target_and_query=(
            outcomes.write_official_target_and_frame_zero_query_artifacts
        ),
        validate_target=outcomes.validate_official_target_artifact,
        validate_frozen_field=queries.validate_preoutcome_frozen_field_manifest,
        validate_queried_prediction=queries.validate_queried_prediction_artifact,
        load_scoring_inputs=(
            outcomes.load_direct_scoring_inputs_after_future_score_permit
        ),
        score_case=scoring.score_direct_official_identity_case,
        create_score_evidence_and_decision=(
            score_artifacts.create_score_evidence_and_decision
        ),
        validate_replacement_source=replacement.validate_aligned_source_manifest,
        validate_confirmation_source=(
            confirmation_source.validate_confirmation_source_cohort_manifest
        ),
    )


def _load_smoke(source: Path) -> Callable[[], Mapping[str, Any]]:
    from bayesian_phystwin import deform360_held_v8_gsplat_runtime as runtime

    expected = source / "bayesian_phystwin" / "deform360_held_v8_gsplat_runtime.py"
    _require(
        Path(runtime.__file__).resolve() == expected, "runtime smoke escaped source"
    )
    return runtime.load_and_smoke_gsplat_runtime


def _normalized_environment() -> dict[str, str]:
    return {
        "BPT_HELD_V8_OUTCOME_ENV_NORMALIZED": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_HUB_OFFLINE": "1",
        "HOME": "/home/florianpfaff",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "LOGNAME": "florianpfaff",
        "PATH": PINNED_PATH,
        "PYNPUT_BACKEND": "dummy",
        "PYOPENGL_PLATFORM": "egl",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPYCACHEPREFIX": PYCACHE_PREFIX,
        "PYTHONSAFEPATH": "1",
        "TMPDIR": "/tmp",
        "TRANSFORMERS_OFFLINE": "1",
        "USER": "florianpfaff",
        "WANDB_MODE": "disabled",
    }


def _normalize_or_reexec() -> None:
    if os.environ.get("BPT_HELD_V8_OUTCOME_ENV_NORMALIZED") == "1":
        _require(sys.flags.isolated == 1, "normalized outcome process is not isolated")
        _require(dict(os.environ) == _normalized_environment(), "outcome env changed")
        _require(
            sys.flags.dont_write_bytecode == 1 and sys.pycache_prefix == PYCACHE_PREFIX,
            "outcome process may consult adjacent bytecode",
        )
        _require(
            not os.path.lexists("/nonexistent") and not os.path.lexists(PYCACHE_PREFIX),
            "reserved held-v8 bytecode prefix is available",
        )
        return
    environment = _normalized_environment()
    os.execve(
        PINNED_PYTHON,
        [
            str(PINNED_PYTHON),
            "-I",
            "-B",
            "-X",
            f"pycache_prefix={PYCACHE_PREFIX}",
            str(Path(sys.argv[0]).resolve()),
            *sys.argv[1:],
        ],
        environment,
    )


def _parse_args(role: Literal["calibration", "confirmation"]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Fresh held-v8 {role} outcomes")
    parser.add_argument(
        "--source-only-verifier", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument("--promote-only", action="store_true")
    parser.add_argument("--dry-run-barrier-only", action="store_true")
    parser.add_argument("--deployed-code", required=True)
    parser.add_argument("--lock")
    parser.add_argument("--replacement-source-manifest")
    parser.add_argument("--confirmation-source-manifest")
    parser.add_argument("--aligned-root")
    parser.add_argument("--deform360-repo")
    parser.add_argument("--sam2-repository")
    parser.add_argument("--sam2-checkpoint")
    parser.add_argument("--cotracker-repo")
    parser.add_argument("--cotracker-checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    namespace = parser.parse_args()
    if namespace.lock is None:
        namespace.lock = str(CANONICAL_HELD_ROOT / f"{role}-lock.json")
    if role == "calibration" and namespace.promote_only:
        parser.error("calibration has no promotion mode")
    if role == "confirmation" and namespace.promote_only:
        forbidden = [
            namespace.replacement_source_manifest,
            namespace.confirmation_source_manifest,
            namespace.aligned_root,
            namespace.deform360_repo,
            namespace.sam2_repository,
            namespace.sam2_checkpoint,
            namespace.cotracker_repo,
            namespace.cotracker_checkpoint,
        ]
        if (
            any(value is not None for value in forbidden)
            or namespace.dry_run_barrier_only
        ):
            parser.error("promotion accepts no prediction, source, or outcome runtime")
    return namespace


def main_for_role(
    role: Literal["calibration", "confirmation"],
    *,
    process_isolated_reconstruction: bool = False,
) -> int:
    namespace = _parse_args(role)
    protocol, source = _load_protocol(namespace.deployed_code)
    if namespace.source_only_verifier:
        verify_source_only_deployment(
            role=role,
            deployed_code=namespace.deployed_code,
            lock_path=namespace.lock,
            protocol=protocol,
        )
        return 0
    _normalize_or_reexec()
    if namespace.promote_only:
        _require(role == "confirmation", "only confirmation can be promoted")
        calibration_lock = CANONICAL_HELD_ROOT / "calibration-lock.json"
        calibration_decision = (
            CANONICAL_HELD_ROOT / "calibration" / "calibration-gate-decision.json"
        )
        arguments = DriverArguments(
            role="calibration",
            deployed_code=namespace.deployed_code,
            lock_path=str(calibration_lock),
            replacement_source_manifest_path=None,
            dry_run_barrier_only=False,
            confirmation_source_manifest_path=None,
        )
        run_source_only_deployment_verifier(arguments)
        protocol.create_confirmation_protocol_lock(
            CANONICAL_HELD_ROOT / "confirmation-lock.json",
            calibration_lock,
            calibration_decision,
        )
        _emit("CONFIRMATION_LOCK_PROMOTED_FROM_CALIBRATION_GO")
        return 0
    arguments = DriverArguments(
        role=role,
        deployed_code=namespace.deployed_code,
        lock_path=namespace.lock,
        replacement_source_manifest_path=namespace.replacement_source_manifest,
        dry_run_barrier_only=namespace.dry_run_barrier_only,
        confirmation_source_manifest_path=namespace.confirmation_source_manifest,
        aligned_root=namespace.aligned_root,
        deform360_repo=namespace.deform360_repo,
        sam2_repository=namespace.sam2_repository,
        sam2_checkpoint=namespace.sam2_checkpoint,
        cotracker_repo=namespace.cotracker_repo,
        cotracker_checkpoint=namespace.cotracker_checkpoint,
        device=namespace.device,
        ffmpeg=namespace.ffmpeg,
    )
    return execute_outcomes(
        arguments,
        protocol=protocol,
        deployment_verifier=run_source_only_deployment_verifier,
        smoke_gsplat_runtime=_load_smoke(source),
        load_post_barrier_api=lambda: _load_post_barrier_api(source),
        reconstruction_runner=(
            run_process_isolated_reconstruction
            if process_isolated_reconstruction
            else None
        ),
    )


__all__ = [
    "CANONICAL_HELD_ROOT",
    "DriverArguments",
    "NO_GO_EXIT_CODE",
    "NOT_CONFIRMED_EXIT_CODE",
    "OutcomeLayout",
    "PostBarrierApi",
    "PROTOCOL_ID",
    "QUALIFIED_RLIMIT_NOFILE_SOFT",
    "ROLE_EXECUTION_COMPLETION_KIND",
    "ROLE_EXECUTION_COMPLETION_STATUS",
    "ROLE_EXECUTION_COMPLETION_SUFFIX",
    "build_layout",
    "build_query_subprocess",
    "build_role_outcome_sealer_subprocess",
    "canonical_role_execution_completion_path",
    "canonical_role_source_manifest_path",
    "execute_outcomes",
    "main_for_role",
    "run_query_subprocess",
    "run_process_isolated_reconstruction",
    "run_role_outcome_sealer",
    "run_source_only_deployment_verifier",
    "validate_role_execution_completion",
    "verify_source_only_deployment",
]
