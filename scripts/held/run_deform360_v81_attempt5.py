#!/usr/bin/env python3
"""Run the immutable, phased Deform360 held-v8.1 attempt-5 workflow.

This operator owns orchestration only.  Numerical work and formal artifact
publication remain in the separately bound low-level operators.  Every child
log is written below the external orchestration root, never below the held
root, and every phase is intentionally a fresh one-shot operation.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
from datetime import datetime, timezone
import difflib
import fcntl
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import resource
import socket
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence


EXPECTED_HOST = "workstation2"
PROTOCOL_ID = "deform360-held-online-belief-v8.1"
EXECUTION_ATTEMPT = 5
TECHNICAL_FAILURE_EXIT_CODE = 2
QUALIFICATION_INCONCLUSIVE_EXIT_CODE = 3
CALIBRATION_NO_GO_EXIT_CODE = 3
CONFIRMATION_NOT_CONFIRMED_EXIT_CODE = 4
QUALIFIED_RLIMIT_NOFILE_SOFT = 1024

ORCHESTRATION_ROOT = Path("/mnt/corsair/florianpfaff/bpt-held-v81-orchestration")
PINNED_PYTHON = Path(
    "/mnt/corsair/florianpfaff/bpt-held-v5-runtimes/"
    "bpt-gpu-pip-4948737892f77c6a9496795e6c3f25b92fcea466ddb7b5f1e9c1b0de1137f004/"
    "bin/python"
)
PYCACHE_PREFIX = "/nonexistent/bpt-held-v8-pycache"
HELD_ROOT = Path("/mnt/corsair/florianpfaff/bpt-online-belief-v1/held-v8")
TERMINAL_COMPLETION = Path(f"{HELD_ROOT}-terminal-integrity-completion.json")
CONFIRMATION_SOURCE_ROOT = HELD_ROOT / "confirmation-source"
CONFIRMATION_SOURCE_RUNTIME_ROOT = HELD_ROOT / ".confirmation-source-runtime"
CONFIRMATION_SOURCE_MANIFEST = (
    CONFIRMATION_SOURCE_ROOT / "manifests/aligned-source-cohort.json"
)
ALIGNED_ROOT = Path(
    "/mnt/lexar4tb/datasets/deform360/data-7fea8e2/replication-v1/aligned"
)
DEFORM360_HEAD = "0fe36f0b7a7a917ba62b5f8cee707299a9a4a317"
DEFORM360_REPO = Path(
    "/mnt/corsair/florianpfaff/bpt-held-v81-runtimes/"
    f"Deform360-processing-{DEFORM360_HEAD}"
)
DEVELOPMENT_DATASET = Path(
    "/mnt/corsair/florianpfaff/deform360-reusable-sota-v1/"
    "processing-sam2-dev-smoke/004-rubber-band/episode_0001/"
    "splatfacto/.scratch_000000"
)
SAM2_HEAD = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
SAM2_REPOSITORY = Path("/mnt/lexar4tb/datasets/deform360/sam2-2b90b9f5")
SAM2_CHECKPOINT = SAM2_REPOSITORY / "checkpoints/sam2.1_hiera_small.pt"
SAM2_CHECKPOINT_SHA256 = (
    "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38"
)
COTRACKER_HEAD = "82e02e8029753ad4ef13cf06be7f4fc5facdda4d"
COTRACKER_REPOSITORY = Path(
    "/mnt/corsair/florianpfaff/deform360-processing-deps/co-tracker"
)
COTRACKER_CHECKPOINT = Path(
    "/home/florianpfaff/.cache/torch/hub/checkpoints/scaled_offline.pth"
)
COTRACKER_CHECKPOINT_SHA256 = (
    "2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834"
)

QUALIFICATION_BASE = Path("/mnt/corsair/florianpfaff")
QUALIFICATION_ROOT_PREFIX = "bpt-resource-lifecycle-qualification-"
REPLAY_ROOT = Path(
    "/mnt/corsair/florianpfaff/"
    "bpt-held-v8.1-attempt-5-admission-wrapper-scratch-20260722"
)

UPSTREAM = Path(
    "/mnt/corsair/florianpfaff/bpt-held-v5-runtimes/"
    "Bayesian-PhysTwin-upstream-58ab4808e59d"
)
OFFICIAL_PHYSTWIN = Path(
    "/mnt/corsair/florianpfaff/bpt-held-v5-runtimes/PhysTwin-upstream-2b6630528141"
)
SEMANTIC_MODEL = Path(
    "/mnt/corsair/florianpfaff/model-cache/siglip2-base-patch16-224-75de2d55"
)
SEMANTIC_MODEL_LOCK = Path(
    "/mnt/corsair/florianpfaff/bpt-framezero-field-dev-20260720/"
    "scratch_siglip2_model_lock.json"
)
ALLTRACKER = Path("/mnt/corsair/florianpfaff/alltracker-molmomotion-61f5b21")
ALLTRACKER_CHECKPOINT = Path("/mnt/corsair/florianpfaff/model-cache/alltracker.pth")
DEVELOPMENT_DECISION = Path(
    "/mnt/corsair/florianpfaff/bpt-online-belief-v1/runs/"
    "deform360-query-field-open27-v1-development/decision.json"
)
DEVELOPMENT_DECISION_SHA256 = (
    "110b3c1831898ff6b333f35236401761222f85eafac1dcbcea7b7183d5b434bd"
)

NVIDIA_SMI = Path("/usr/bin/nvidia-smi")
GIT = Path("/usr/bin/git")
BASH = Path("/bin/bash")
FFMPEG = Path("/usr/bin/ffmpeg")
LAUNCHER_RELATIVE = Path("scripts/held/run_deform360_v81_attempt5.py")
H2_PREPARER_RELATIVE = Path("scripts/held/prepare_deform360_v8_lock.py")
H2_LINEAGE_TEST_RELATIVE = Path("tests/test_deform360_held_v8_lock_preparer.py")
H1_LINEAGE_TEST_FUNCTION = "test_attempt_five_execution_hash_placeholders_fail_closed"
H2_LINEAGE_TEST_FUNCTION = "test_attempt_five_execution_pins_match_sealed_h1"
H2_PIN_ASSIGNMENT_NAMES = (
    "_RESOURCE_LIFECYCLE_QUALIFICATION_ROOT",
    "_RESOURCE_LIFECYCLE_QUALIFICATION_EVIDENCE",
    "_RESOURCE_LIFECYCLE_QUALIFICATION_COMPLETION",
    "_RESOURCE_LIFECYCLE_QUALIFICATION_EVIDENCE_FILE_SHA256",
    "_RESOURCE_LIFECYCLE_QUALIFICATION_EVIDENCE_ARTIFACT_SHA256",
    "_RESOURCE_LIFECYCLE_QUALIFICATION_COMPLETION_FILE_SHA256",
    "_RESOURCE_LIFECYCLE_QUALIFICATION_COMPLETION_ARTIFACT_SHA256",
    "_RESOURCE_LIFECYCLE_QUALIFICATION_ATTEMPT_FILE_SHA256",
    "_RESOURCE_LIFECYCLE_QUALIFICATION_ATTEMPT_ARTIFACT_SHA256",
    "_RESOURCE_LIFECYCLE_QUALIFICATION_MANIFEST_FILE_SHA256",
    "_RESOURCE_LIFECYCLE_QUALIFICATION_MANIFEST_ARTIFACT_SHA256",
    "_RESOURCE_LIFECYCLE_QUALIFICATION_ANALYSIS_FILE_SHA256",
    "_RESOURCE_LIFECYCLE_QUALIFICATION_ANALYSIS_ARTIFACT_SHA256",
    "_V8_ADMISSION_REPLAY_REPORT_FILE_SHA256",
    "_V8_ADMISSION_REPLAY_REPORT_ARTIFACT_SHA256",
    "_V8_ADMISSION_REPLAY_CODE_BINDING_FILE_SHA256",
    "_V8_ADMISSION_REPLAY_CODE_BINDING_ARTIFACT_SHA256",
)
H2_PATH_ASSIGNMENT_NAMES = frozenset(H2_PIN_ASSIGNMENT_NAMES[:3])
H2_CHANGED_PATHS = frozenset(
    {
        H2_PREPARER_RELATIVE.as_posix(),
        H2_LINEAGE_TEST_RELATIVE.as_posix(),
    }
)
_HEAD_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_BASE_CHILD_ENVIRONMENT = {
    "HOME": "/home/florianpfaff",
    "USER": "florianpfaff",
    "LOGNAME": "florianpfaff",
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "TMPDIR": "/tmp",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONPYCACHEPREFIX": PYCACHE_PREFIX,
    "PYTHONSAFEPATH": "1",
}


@dataclass
class PhaseContext:
    phase: str
    source_head: str
    root: Path
    started_utc: str
    steps: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    payload: Mapping[str, Any]


_ACTIVE_PHASE: PhaseContext | None = None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _signed_artifact(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("artifact_sha256", None)
    result["artifact_sha256"] = hashlib.sha256(_canonical_bytes(result)).hexdigest()
    return result


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _absolute(path: str | os.PathLike[str], *, label: str) -> Path:
    value = Path(os.path.abspath(os.fspath(path)))
    _require(Path(path).is_absolute(), f"{label} is not absolute")
    return value


def _canonical_directory(
    path: str | os.PathLike[str],
    *,
    label: str,
    mode: int | None = None,
) -> Path:
    value = _absolute(path, label=label)
    observed = os.lstat(value)
    _require(
        stat.S_ISDIR(observed.st_mode)
        and not stat.S_ISLNK(observed.st_mode)
        and value.resolve(strict=True) == value,
        f"{label} is absent, linked, or non-canonical",
    )
    if mode is not None:
        _require(
            stat.S_IMODE(observed.st_mode) == mode,
            f"{label} mode is not {mode:04o}",
        )
    return value


def _read_regular(
    path: str | os.PathLike[str],
    *,
    label: str,
    mode: int | None = None,
) -> bytes:
    value = _absolute(path, label=label)
    before = os.lstat(value)
    _require(
        stat.S_ISREG(before.st_mode)
        and not stat.S_ISLNK(before.st_mode)
        and before.st_nlink == 1
        and value.resolve(strict=True) == value,
        f"{label} is absent, linked, multiply linked, or non-canonical",
    )
    if mode is not None:
        _require(
            stat.S_IMODE(before.st_mode) == mode,
            f"{label} mode is not {mode:04o}",
        )
    descriptor = os.open(
        value,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        _require(
            (opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino),
            f"{label} changed while opening",
        )
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = os.lstat(value)
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    _require(
        identity
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        == (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns),
        f"{label} changed while reading",
    )
    return b"".join(chunks)


def _read_json_artifact(
    path: str | os.PathLike[str],
    *,
    label: str,
    mode: int = 0o400,
) -> tuple[dict[str, Any], bytes]:
    payload = _read_regular(path, label=label, mode=mode)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not valid JSON") from error
    _require(isinstance(value, dict), f"{label} is not a JSON object")
    expected = value.get("artifact_sha256")
    unsigned = dict(value)
    unsigned.pop("artifact_sha256", None)
    _require(
        _is_sha256(expected)
        and hashlib.sha256(_canonical_bytes(unsigned)).hexdigest() == expected,
        f"{label} artifact SHA-256 is invalid",
    )
    return value, payload


def _write_new(path: Path, payload: bytes, *, final_mode: int = 0o400) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            _require(written > 0, f"short write for {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, final_mode, follow_symlinks=False)


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_new(path, _json_bytes(value))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _child_environment(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    result = dict(_BASE_CHILD_ENVIRONMENT)
    if extra is not None:
        result.update(extra)
    return result


def _git_environment() -> dict[str, str]:
    result = _child_environment()
    result.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
        }
    )
    return result


def _capture_command(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> bytes:
    completed = subprocess.run(
        tuple(argv),
        check=False,
        cwd=None if cwd is None else os.fspath(cwd),
        env=dict(_child_environment() if environment is None else environment),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True,
    )
    _require(
        completed.returncode == 0,
        f"command failed with code {completed.returncode}: {argv[0]}",
    )
    return completed.stdout


def _git(code: Path, arguments: Sequence[str]) -> bytes:
    return _capture_command(
        [os.fspath(GIT), "-C", os.fspath(code), *arguments],
        cwd=code,
        environment=_git_environment(),
    )


def _verify_repository(
    code_root: str | os.PathLike[str],
    *,
    require_launcher_location: bool = True,
) -> tuple[Path, str]:
    code = _canonical_directory(code_root, label="source repository")
    _require((code / ".git").is_dir(), "source repository has no Git metadata")
    top = _git(code, ["rev-parse", "--show-toplevel"]).decode("utf-8").strip()
    head = _git(code, ["rev-parse", "HEAD"]).decode("ascii").strip().lower()
    _require(top == os.fspath(code), "source repository top level changed")
    _require(_HEAD_RE.fullmatch(head) is not None, "source HEAD is invalid")
    _require(
        _git(code, ["status", "--porcelain=v1", "--untracked-files=all"]) == b""
        and _git(code, ["ls-files", "--others", "--exclude-standard"]) == b""
        and _git(
            code,
            ["ls-files", "--others", "--ignored", "--exclude-standard"],
        )
        == b"",
        "source repository is dirty or contains untracked content",
    )
    shallow = _git(code, ["rev-parse", "--is-shallow-repository"])
    _require(shallow.strip() == b"false", "source repository is shallow")
    _git(code, ["fsck", "--full", "--no-dangling"])
    launcher = code / LAUNCHER_RELATIVE
    _read_regular(launcher, label="attempt-5 launcher source")
    if require_launcher_location:
        _require(
            launcher.resolve(strict=True) == Path(__file__).resolve(strict=True),
            "launcher is not executing from the selected source repository",
        )
    return code, head


def _require_deployed_repository(code_root: Path) -> tuple[Path, str]:
    code, head = _verify_repository(code_root)
    _require(
        code == HELD_ROOT / f"code-{head}",
        "deployed code path does not match its exact HEAD",
    )
    _require(
        os.lstat(code).st_mode & 0o222 == 0,
        "deployed code root is writable",
    )
    for current, directories, files in os.walk(code):
        for name in [*directories, *files]:
            path = Path(current) / name
            observed = os.lstat(path)
            _require(not stat.S_ISLNK(observed.st_mode), "deployment contains a link")
            _require(
                observed.st_mode & 0o222 == 0,
                f"deployment contains a writable entry: {path}",
            )
    return code, head


def _verify_pinned_git_repository(
    path: Path,
    head: str,
    *,
    label: str,
    allowed_ignored: frozenset[str] = frozenset(),
    allow_external_pycache: bool = False,
    require_nonwritable: bool = False,
) -> None:
    repository = _canonical_directory(path, label=label)
    observed_head = _git(repository, ["rev-parse", "HEAD"])
    observed_top = _git(repository, ["rev-parse", "--show-toplevel"])
    _require(
        observed_head.decode("ascii").strip().lower() == head,
        f"{label} HEAD changed",
    )
    _require(
        observed_top.decode("utf-8").strip() == os.fspath(repository),
        f"{label} Git top level changed",
    )
    ordinary_payload = _git(
        repository,
        ["ls-files", "-z", "--others", "--exclude-standard"],
    )
    ignored = _git(
        repository,
        ["ls-files", "-z", "--others", "--ignored", "--exclude-standard"],
    )
    try:
        ordinary_paths = frozenset(
            value.decode("utf-8") for value in ordinary_payload.split(b"\0") if value
        )
        ignored_paths = frozenset(
            value.decode("utf-8") for value in ignored.split(b"\0") if value
        )
    except UnicodeDecodeError as error:
        raise RuntimeError(f"{label} has a non-UTF-8 ignored path") from error
    _require(
        _git(repository, ["status", "--porcelain=v1", "--untracked-files=no"]) == b""
        and allowed_ignored <= ignored_paths
        and not (allowed_ignored & ordinary_paths),
        f"{label} has modified tracked files or misplaced allowed artifacts",
    )
    extras = (ordinary_paths | ignored_paths) - allowed_ignored
    if not allow_external_pycache:
        _require(not extras, f"{label} has unexpected untracked or ignored files")
    for relative_text in sorted(extras):
        relative = PurePosixPath(relative_text)
        _require(
            not relative.is_absolute()
            and ".." not in relative.parts
            and len(relative.parts) >= 2
            and relative.parts[-2] == "__pycache__"
            and relative.suffix == ".pyc",
            f"{label} extra entry is not an adjacent __pycache__/*.pyc: {relative_text}",
        )
        _read_regular(
            repository.joinpath(*relative.parts),
            label=f"{label} non-importable external bytecode",
        )
    if extras:
        # Every formal Python child is launched with ``-I -B`` and an explicit
        # ``-X pycache_prefix=...``.  Under that configuration CPython searches
        # only the reserved prefix for cache candidates, so these pre-existing
        # adjacent cache files are structurally inventoried but not importable.
        _require(
            _BASE_CHILD_ENVIRONMENT.get("PYTHONPYCACHEPREFIX") == PYCACHE_PREFIX
            and not os.path.lexists("/nonexistent")
            and not os.path.lexists(PYCACHE_PREFIX),
            f"{label} bytecode is not isolated by the reserved pycache prefix",
        )
    if require_nonwritable:
        for current, directories, files in os.walk(repository):
            for name in [".", *directories, *files]:
                candidate = Path(current) if name == "." else Path(current) / name
                _require(
                    os.lstat(candidate).st_mode & 0o222 == 0,
                    f"{label} contains a writable entry: {candidate}",
                )


def _require_executable(path: Path, *, label: str) -> None:
    _require(
        os.path.isfile(path) and os.access(path, os.X_OK),
        f"{label} is absent or not executable: {path}",
    )


def _sha256_file(path: Path, *, label: str, mode: int | None = None) -> str:
    return hashlib.sha256(_read_regular(path, label=label, mode=mode)).hexdigest()


def _require_host() -> None:
    _require(
        socket.gethostname() == EXPECTED_HOST,
        f"attempt-5 phases require {EXPECTED_HOST}",
    )


def _require_gpu_idle(indices: Sequence[int]) -> None:
    _require_executable(NVIDIA_SMI, label="nvidia-smi")
    output = _capture_command(
        [
            os.fspath(NVIDIA_SMI),
            "--query-gpu=index",
            "--format=csv,noheader,nounits",
        ]
    )
    try:
        available = {
            int(line.strip())
            for line in output.decode("ascii").splitlines()
            if line.strip()
        }
    except (UnicodeDecodeError, ValueError) as error:
        raise RuntimeError("nvidia-smi returned invalid GPU indices") from error
    for index in indices:
        _require(index in available, f"physical GPU {index} is absent")
        processes = _capture_command(
            [
                os.fspath(NVIDIA_SMI),
                f"--id={index}",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ]
        )
        _require(
            processes.strip() == b"",
            f"physical GPU {index} already has a compute process",
        )


def _set_soft_nofile(limit: int = QUALIFIED_RLIMIT_NOFILE_SOFT) -> dict[str, int]:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    infinity = resource.RLIM_INFINITY
    _require(hard == infinity or hard >= limit, "RLIMIT_NOFILE hard limit is too low")
    resource.setrlimit(resource.RLIMIT_NOFILE, (limit, hard))
    observed_soft, observed_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    _require(
        observed_soft == limit and observed_hard == hard,
        "failed to set the qualified RLIMIT_NOFILE soft limit",
    )
    return {
        "before_soft": int(soft),
        "hard": int(hard),
        "after_soft": int(observed_soft),
    }


def _under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_no_writable_held_descriptors() -> dict[str, Any]:
    if not Path("/proc/self/fd").is_dir():
        raise RuntimeError("/proc/self/fd is unavailable")
    held = HELD_ROOT.resolve(strict=os.path.lexists(HELD_ROOT))
    checked = 0
    for entry in os.scandir("/proc/self/fd"):
        try:
            descriptor = int(entry.name)
            flags = fcntl.fcntl(descriptor, fcntl.F_GETFL)
            target_text = os.readlink(entry.path)
        except (OSError, ValueError):
            continue
        checked += 1
        if flags & os.O_ACCMODE not in {os.O_WRONLY, os.O_RDWR}:
            continue
        if not target_text.startswith("/") or target_text.endswith(" (deleted)"):
            continue
        target = Path(os.path.abspath(target_text))
        _require(
            not _under(target, held),
            f"launcher holds a writable descriptor below the held root: fd {descriptor}",
        )
    return {"checked_descriptor_count": checked, "writable_held_descriptor_count": 0}


def _ensure_orchestration_root() -> Path:
    if not os.path.lexists(ORCHESTRATION_ROOT):
        os.mkdir(ORCHESTRATION_ROOT, 0o700)
    root = _canonical_directory(
        ORCHESTRATION_ROOT,
        label="orchestration root",
        mode=0o700,
    )
    _require(
        not _under(root, HELD_ROOT) and not _under(HELD_ROOT, root),
        "orchestration and held roots overlap",
    )
    return root


def _new_phase(phase: str, head: str) -> PhaseContext:
    global _ACTIVE_PHASE
    _require(_ACTIVE_PHASE is None, "another launcher phase is active")
    root = _ensure_orchestration_root()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    phase_root = root / f"{phase}-{head}-{timestamp}-{os.getpid()}"
    os.mkdir(phase_root, 0o700)
    _fsync_directory(root)
    context = PhaseContext(
        phase=phase,
        source_head=head,
        root=phase_root,
        started_utc=timestamp,
    )
    _ACTIVE_PHASE = context
    return context


def _finish_phase(
    context: PhaseContext,
    *,
    returncode: int,
    status: str,
    details: Mapping[str, Any] | None = None,
) -> Path:
    global _ACTIVE_PHASE
    summary = _signed_artifact(
        {
            "schema_version": 1,
            "artifact_kind": "Deform360HeldV81Attempt5LauncherPhase",
            "protocol_id": PROTOCOL_ID,
            "execution_attempt": EXECUTION_ATTEMPT,
            "phase": context.phase,
            "source_head": context.source_head,
            "started_utc": context.started_utc,
            "status": status,
            "returncode": returncode,
            "steps": context.steps,
            "details": {} if details is None else dict(details),
        }
    )
    path = context.root / "phase-summary.json"
    _write_new_json(path, summary)
    _fsync_directory(context.root)
    _ACTIVE_PHASE = None
    return path


def _finish_failed_phase(error: BaseException) -> Path | None:
    global _ACTIVE_PHASE
    context = _ACTIVE_PHASE
    if context is None:
        return None
    try:
        return _finish_phase(
            context,
            returncode=TECHNICAL_FAILURE_EXIT_CODE,
            status="technical-failure",
            details={
                "error_type": type(error).__name__,
                "message": str(error),
            },
        )
    except BaseException:
        _ACTIVE_PHASE = None
        return None


def _open_log(path: Path) -> int:
    return os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _record_exit(context: PhaseContext, name: str, returncode: int) -> None:
    exit_path = context.root / f"{name}.exit.code"
    _write_new(exit_path, f"{returncode}\n".encode("ascii"))
    context.steps.append(
        {
            "name": name,
            "log": str(context.root / f"{name}.log"),
            "exit_code_file": str(exit_path),
            "returncode": returncode,
        }
    )


def _run_logged_step(
    context: PhaseContext,
    name: str,
    argv: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str] | None = None,
) -> int:
    log_path = context.root / f"{name}.log"
    descriptor = _open_log(log_path)
    process: subprocess.Popen[bytes] | None = None
    returncode = -1
    try:
        try:
            process = subprocess.Popen(
                tuple(argv),
                cwd=os.fspath(cwd),
                env=dict(_child_environment() if environment is None else environment),
                stdin=subprocess.DEVNULL,
                stdout=descriptor,
                stderr=subprocess.STDOUT,
                close_fds=True,
            )
            returncode = process.wait()
        except BaseException:
            if process is not None:
                _terminate_process(process)
                returncode = (
                    process.returncode if process.returncode is not None else -1
                )
            raise
        finally:
            os.fsync(descriptor)
    finally:
        os.close(descriptor)
        os.chmod(log_path, 0o400, follow_symlinks=False)
        _record_exit(context, name, returncode)
    return returncode


def _run_concurrent_steps(
    context: PhaseContext,
    commands: Sequence[tuple[str, Sequence[str]]],
    *,
    cwd: Path,
    environment: Mapping[str, str],
) -> dict[str, int]:
    _require(len(commands) == 2, "formal execution requires exactly two shards")
    descriptors: dict[str, int] = {}
    processes: dict[str, subprocess.Popen[bytes]] = {}
    returncodes: dict[str, int] = {name: -1 for name, _ in commands}
    try:
        for name, _ in commands:
            descriptors[name] = _open_log(context.root / f"{name}.log")
        try:
            for name, argv in commands:
                processes[name] = subprocess.Popen(
                    tuple(argv),
                    cwd=os.fspath(cwd),
                    env=dict(environment),
                    stdin=subprocess.DEVNULL,
                    stdout=descriptors[name],
                    stderr=subprocess.STDOUT,
                    close_fds=True,
                )
            for name, _ in commands:
                returncodes[name] = processes[name].wait()
        except BaseException:
            for process in processes.values():
                _terminate_process(process)
            for name, process in processes.items():
                if process.returncode is not None:
                    returncodes[name] = process.returncode
            raise
    finally:
        for name, _ in commands:
            descriptor = descriptors.get(name)
            if descriptor is not None:
                os.fsync(descriptor)
                os.close(descriptor)
                os.chmod(context.root / f"{name}.log", 0o400, follow_symlinks=False)
                _record_exit(context, name, returncodes[name])
    _require(
        all(returncodes[name] == 0 for name, _ in commands),
        f"both formal shards are required to succeed: {returncodes}",
    )
    return returncodes


def _qualification_root(head: str) -> Path:
    return QUALIFICATION_BASE / f"{QUALIFICATION_ROOT_PREFIX}{head}"


def _qualification_commands(code: Path, head: str) -> tuple[list[str], list[str]]:
    root = _qualification_root(head)
    qualify = [
        os.fspath(PINNED_PYTHON),
        "-I",
        "-B",
        os.fspath(code / "scripts/development/qualify_deform360_resource_lifecycle.py"),
        "run",
        "--code-root",
        os.fspath(code),
        "--python",
        os.fspath(PINNED_PYTHON),
        "--deform360-repo",
        os.fspath(DEFORM360_REPO),
        "--dataset",
        os.fspath(DEVELOPMENT_DATASET),
        "--output-dir",
        os.fspath(root),
        "--phase",
        "all",
        "--cuda-device",
        "1",
        "--seed",
        "0",
        "--ab-iterations",
        "250",
        "--ab-repeat-count",
        "5",
        "--soak-fit-count",
        "243",
        "--soak-iterations",
        "1",
        "--first-fit-fd-growth-limit",
        "32",
        "--steady-fd-growth-limit",
        "4",
        "--steady-task-growth-limit",
        "4",
        "--fit-timeout-seconds",
        "3600",
        "--analyzer-timeout-seconds",
        "86400",
        "--soak-timeout-seconds",
        "86400",
    ]
    seal = [
        os.fspath(PINNED_PYTHON),
        "-I",
        "-B",
        os.fspath(
            code / "scripts/held/seal_deform360_resource_lifecycle_qualification.py"
        ),
        "--qualification-root",
        os.fspath(root),
    ]
    return qualify, seal


def _require_fresh(path: Path, *, label: str) -> None:
    _require(not os.path.lexists(path), f"{label} is not fresh: {path}")


def _validate_qualification_completion(head: str, *, admitted: bool) -> None:
    root = _qualification_root(head)
    completion_path = Path(f"{root}-integrity-completion.json")
    _canonical_directory(root, label="qualification root", mode=0o500)
    evidence, _ = _read_json_artifact(
        root / "resource-lifecycle-qualification.json",
        label="qualification evidence",
    )
    completion, _ = _read_json_artifact(
        completion_path,
        label="qualification integrity completion",
    )
    expected_status = "qualified" if admitted else "admission-inconclusive"
    _require(
        evidence.get("status") == expected_status
        and evidence.get("passed") is admitted
        and completion.get("status") == "qualification-integrity-complete"
        and completion.get("terminal_outcome") == expected_status
        and completion.get("admission_eligible") is admitted,
        "qualification semantic result and sealed completion disagree",
    )


def _collect_h2_pins(head: str) -> dict[str, Any]:
    qualification_root = _qualification_root(head)
    completion_path = Path(f"{qualification_root}-integrity-completion.json")
    _canonical_directory(qualification_root, label="qualification root", mode=0o500)
    _canonical_directory(REPLAY_ROOT, label="admission replay root", mode=0o500)
    paths = {
        "qualification_evidence": (
            qualification_root / "resource-lifecycle-qualification.json"
        ),
        "qualification_completion": completion_path,
        "qualification_attempt": qualification_root / "qualification-attempt.json",
        "qualification_manifest": qualification_root
        / "equivalence/repeat-manifest.json",
        "qualification_analysis": qualification_root
        / "equivalence/analysis-result.json",
        "replay_report": REPLAY_ROOT / "metadata-only-replay-report.json",
        "replay_code_binding": (REPLAY_ROOT / "metadata-only-replay-code-binding.json"),
    }
    artifacts: dict[str, dict[str, Any]] = {}
    payloads: dict[str, bytes] = {}
    for name, path in paths.items():
        artifact, payload = _read_json_artifact(path, label=name.replace("_", " "))
        artifacts[name] = artifact
        payloads[name] = payload

    evidence = artifacts["qualification_evidence"]
    completion = artifacts["qualification_completion"]
    source_code = completion.get("source_code")
    _require(
        evidence.get("status") == "qualified"
        and evidence.get("passed") is True
        and isinstance(evidence.get("admission"), Mapping)
        and evidence["admission"].get("decision") == "admitted"
        and completion.get("status") == "qualification-integrity-complete"
        and completion.get("terminal_outcome") == "qualified"
        and completion.get("admission_eligible") is True
        and completion.get("qualification_root") == os.fspath(qualification_root)
        and isinstance(source_code, Mapping)
        and source_code.get("source_head") == head,
        "qualification does not admit the exact H1 source",
    )
    completion_bindings = {
        "qualification_evidence": "qualification_evidence",
        "qualification_attempt": "qualification_attempt",
        "qualification_manifest": "repeat_manifest",
        "qualification_analysis": "equivalence_result",
    }
    for local_name, completion_name in completion_bindings.items():
        binding = completion.get(completion_name)
        _require(
            isinstance(binding, Mapping)
            and binding.get("sha256")
            == hashlib.sha256(payloads[local_name]).hexdigest()
            and binding.get("artifact_sha256")
            == artifacts[local_name].get("artifact_sha256"),
            f"qualification completion does not bind {local_name}",
        )

    replay_report = artifacts["replay_report"]
    replay_binding = artifacts["replay_code_binding"]
    report_source = replay_report.get("local_source_at_replay")
    binding_source = replay_binding.get("local_worktree_at_replay")
    report_record = replay_binding.get("replay_report")
    _require(
        replay_report.get("artifact_kind")
        == "Deform360HeldV81ExternalAdmissionMetadataOnlyReplay"
        and replay_report.get("protocol_id") == PROTOCOL_ID
        and replay_report.get("execution_attempt") == EXECUTION_ATTEMPT
        and replay_report.get("development_replay_only") is True
        and replay_report.get("formal_outcome_evidence") is False
        and isinstance(report_source, Mapping)
        and report_source.get("git_head") == head,
        "admission replay report identity changed",
    )
    _require(
        replay_binding.get("artifact_kind")
        == "Deform360HeldV81ExternalAdmissionReplayCodeBinding"
        and replay_binding.get("protocol_id") == PROTOCOL_ID
        and replay_binding.get("execution_attempt") == EXECUTION_ATTEMPT
        and replay_binding.get("formal_outcome_evidence") is False
        and replay_binding.get("target_query_score_or_outcome_accessed") is False
        and binding_source == report_source
        and isinstance(binding_source, Mapping)
        and binding_source.get("git_head") == head
        and isinstance(report_record, Mapping)
        and report_record.get("path") == os.fspath(paths["replay_report"])
        and report_record.get("sha256")
        == hashlib.sha256(payloads["replay_report"]).hexdigest()
        and report_record.get("artifact_sha256")
        == replay_report.get("artifact_sha256"),
        "admission replay code binding changed",
    )

    file_sha = {
        name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()
    }
    assignments: dict[str, str] = {
        "_RESOURCE_LIFECYCLE_QUALIFICATION_ROOT": os.fspath(qualification_root),
        "_RESOURCE_LIFECYCLE_QUALIFICATION_EVIDENCE": os.fspath(
            paths["qualification_evidence"]
        ),
        "_RESOURCE_LIFECYCLE_QUALIFICATION_COMPLETION": os.fspath(completion_path),
        "_RESOURCE_LIFECYCLE_QUALIFICATION_EVIDENCE_FILE_SHA256": file_sha[
            "qualification_evidence"
        ],
        "_RESOURCE_LIFECYCLE_QUALIFICATION_EVIDENCE_ARTIFACT_SHA256": artifacts[
            "qualification_evidence"
        ]["artifact_sha256"],
        "_RESOURCE_LIFECYCLE_QUALIFICATION_COMPLETION_FILE_SHA256": file_sha[
            "qualification_completion"
        ],
        "_RESOURCE_LIFECYCLE_QUALIFICATION_COMPLETION_ARTIFACT_SHA256": artifacts[
            "qualification_completion"
        ]["artifact_sha256"],
        "_RESOURCE_LIFECYCLE_QUALIFICATION_ATTEMPT_FILE_SHA256": file_sha[
            "qualification_attempt"
        ],
        "_RESOURCE_LIFECYCLE_QUALIFICATION_ATTEMPT_ARTIFACT_SHA256": artifacts[
            "qualification_attempt"
        ]["artifact_sha256"],
        "_RESOURCE_LIFECYCLE_QUALIFICATION_MANIFEST_FILE_SHA256": file_sha[
            "qualification_manifest"
        ],
        "_RESOURCE_LIFECYCLE_QUALIFICATION_MANIFEST_ARTIFACT_SHA256": artifacts[
            "qualification_manifest"
        ]["artifact_sha256"],
        "_RESOURCE_LIFECYCLE_QUALIFICATION_ANALYSIS_FILE_SHA256": file_sha[
            "qualification_analysis"
        ],
        "_RESOURCE_LIFECYCLE_QUALIFICATION_ANALYSIS_ARTIFACT_SHA256": artifacts[
            "qualification_analysis"
        ]["artifact_sha256"],
        "_V8_ADMISSION_REPLAY_REPORT_FILE_SHA256": file_sha["replay_report"],
        "_V8_ADMISSION_REPLAY_REPORT_ARTIFACT_SHA256": artifacts["replay_report"][
            "artifact_sha256"
        ],
        "_V8_ADMISSION_REPLAY_CODE_BINDING_FILE_SHA256": file_sha[
            "replay_code_binding"
        ],
        "_V8_ADMISSION_REPLAY_CODE_BINDING_ARTIFACT_SHA256": artifacts[
            "replay_code_binding"
        ]["artifact_sha256"],
    }
    _require(
        tuple(assignments) == H2_PIN_ASSIGNMENT_NAMES,
        "H2 pin assignment names or order changed",
    )
    return _signed_artifact(
        {
            "schema_version": 1,
            "artifact_kind": "Deform360HeldV81Attempt5H2Pins",
            "protocol_id": PROTOCOL_ID,
            "execution_attempt": EXECUTION_ATTEMPT,
            "h1_source_head": head,
            "preparer_assignments": assignments,
        }
    )


def _python_tree(payload: bytes, *, label: str) -> ast.Module:
    try:
        source = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError(f"{label} is not UTF-8") from error
    try:
        return ast.parse(source)
    except SyntaxError as error:
        raise RuntimeError(f"{label} is not valid Python") from error


def _node_byte_span(payload: bytes, node: ast.AST, *, label: str) -> tuple[int, int]:
    lineno = getattr(node, "lineno", None)
    col_offset = getattr(node, "col_offset", None)
    end_lineno = getattr(node, "end_lineno", None)
    end_col_offset = getattr(node, "end_col_offset", None)
    _require(
        all(
            isinstance(value, int)
            for value in (lineno, col_offset, end_lineno, end_col_offset)
        ),
        f"{label} has no exact source span",
    )
    lines = payload.splitlines(keepends=True)
    _require(
        1 <= lineno <= end_lineno <= len(lines),
        f"{label} source span is outside its file",
    )
    start = sum(len(line) for line in lines[: lineno - 1]) + col_offset
    end = sum(len(line) for line in lines[: end_lineno - 1]) + end_col_offset
    _require(0 <= start < end <= len(payload), f"{label} source span is invalid")
    return start, end


def _replace_source_spans(
    payload: bytes,
    replacements: Sequence[tuple[tuple[int, int], bytes]],
    *,
    label: str,
) -> bytes:
    result = payload
    previous_start = len(payload)
    for (start, end), replacement in sorted(
        replacements, key=lambda item: item[0][0], reverse=True
    ):
        _require(
            0 <= start < end <= previous_start,
            f"{label} replacement spans overlap or are invalid",
        )
        result = result[:start] + replacement + result[end:]
        previous_start = start
    return result


def _expected_h2_preparer_source(
    h1_payload: bytes, assignments: Mapping[str, str]
) -> bytes:
    _require(
        tuple(assignments) == H2_PIN_ASSIGNMENT_NAMES,
        "H2 preparer assignments differ from the exact 17-pin contract",
    )
    tree = _python_tree(h1_payload, label="H1 lock preparer")
    nodes: dict[str, ast.AnnAssign] = {}
    for statement in tree.body:
        if (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id in H2_PIN_ASSIGNMENT_NAMES
        ):
            _require(
                statement.target.id not in nodes,
                f"H1 lock preparer assigns {statement.target.id} more than once",
            )
            nodes[statement.target.id] = statement
    _require(
        tuple(name for name in H2_PIN_ASSIGNMENT_NAMES if name in nodes)
        == H2_PIN_ASSIGNMENT_NAMES
        and set(nodes) == set(H2_PIN_ASSIGNMENT_NAMES),
        "H1 lock preparer does not contain the exact 17 pin assignments",
    )
    replacements: list[tuple[tuple[int, int], bytes]] = []
    for name in H2_PIN_ASSIGNMENT_NAMES:
        node = nodes[name]
        _require(
            isinstance(node.value, ast.Constant) and node.value.value is None,
            f"H1 lock preparer pin {name} is not fail-closed None",
        )
        value = assignments[name]
        _require(isinstance(value, str) and value, f"H2 pin {name} is invalid")
        if name in H2_PATH_ASSIGNMENT_NAMES:
            replacement = f"Path({json.dumps(value, ensure_ascii=True)})"
        else:
            _require(_is_sha256(value), f"H2 digest pin {name} is invalid")
            replacement = json.dumps(value, ensure_ascii=True)
        replacements.append(
            (
                _node_byte_span(
                    h1_payload,
                    node.value,
                    label=f"H1 lock preparer pin {name}",
                ),
                replacement.encode("utf-8"),
            )
        )
    return _replace_source_spans(
        h1_payload,
        replacements,
        label="H2 lock preparer",
    )


def _render_h2_lineage_test(assignments: Mapping[str, str]) -> bytes:
    _require(
        tuple(assignments) == H2_PIN_ASSIGNMENT_NAMES,
        "H2 lineage-test assignments differ from the exact 17-pin contract",
    )
    lines = [
        f"def {H2_LINEAGE_TEST_FUNCTION}() -> None:",
        "    expected = {",
    ]
    for name in H2_PIN_ASSIGNMENT_NAMES:
        lines.append(
            f"        {json.dumps(name)}: "
            f"{json.dumps(assignments[name], ensure_ascii=True)},"
        )
    lines.extend(
        [
            "    }",
            "    observed = {}",
            "    for name in expected:",
            "        value = getattr(preparer, name)",
            "        observed[name] = str(value) if isinstance(value, Path) else value",
            "    assert observed == expected",
            "    root, evidence, completion = preparer._require_h2_execution_pins()",
            "    assert (str(root), str(evidence), str(completion)) == (",
            '        expected["_RESOURCE_LIFECYCLE_QUALIFICATION_ROOT"],',
            '        expected["_RESOURCE_LIFECYCLE_QUALIFICATION_EVIDENCE"],',
            '        expected["_RESOURCE_LIFECYCLE_QUALIFICATION_COMPLETION"],',
            "    )",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _expected_h2_lineage_test_source(
    h1_payload: bytes, assignments: Mapping[str, str]
) -> bytes:
    tree = _python_tree(h1_payload, label="H1 lock-preparer lineage test")
    candidates = [
        statement
        for statement in tree.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
        and statement.name == H1_LINEAGE_TEST_FUNCTION
    ]
    _require(
        len(candidates) == 1
        and isinstance(candidates[0], ast.FunctionDef)
        and not candidates[0].decorator_list,
        "H1 lineage test does not contain one undecorated placeholder test",
    )
    function = candidates[0]
    return _replace_source_spans(
        h1_payload,
        [
            (
                _node_byte_span(
                    h1_payload,
                    function,
                    label="H1 placeholder lineage test",
                ),
                _render_h2_lineage_test(assignments),
            )
        ],
        label="H2 lock-preparer lineage test",
    )


def _h2_transition_material(
    code: Path,
    h1_head: str,
    assignments: Mapping[str, str],
) -> dict[str, Any]:
    h1_preparer = _git(code, ["show", f"{h1_head}:{H2_PREPARER_RELATIVE.as_posix()}"])
    h1_test = _git(code, ["show", f"{h1_head}:{H2_LINEAGE_TEST_RELATIVE.as_posix()}"])
    h2_preparer = _expected_h2_preparer_source(h1_preparer, assignments)
    h2_test = _expected_h2_lineage_test_source(h1_test, assignments)
    patch_lines: list[str] = []
    for relative, before, after in (
        (H2_PREPARER_RELATIVE, h1_preparer, h2_preparer),
        (H2_LINEAGE_TEST_RELATIVE, h1_test, h2_test),
    ):
        try:
            before_lines = before.decode("utf-8").splitlines(keepends=True)
            after_lines = after.decode("utf-8").splitlines(keepends=True)
        except UnicodeDecodeError as error:
            raise RuntimeError("H2 transition source is not UTF-8") from error
        patch_lines.extend(
            difflib.unified_diff(
                before_lines,
                after_lines,
                fromfile=f"a/{relative.as_posix()}",
                tofile=f"b/{relative.as_posix()}",
            )
        )
    patch = "".join(patch_lines).encode("utf-8")
    _require(patch, "H2 transition patch is empty")
    return {
        "h1_head": h1_head,
        "changed_paths": sorted(H2_CHANGED_PATHS),
        "expected_sources": {
            H2_PREPARER_RELATIVE.as_posix(): {
                "sha256": hashlib.sha256(h2_preparer).hexdigest(),
                "size_bytes": len(h2_preparer),
            },
            H2_LINEAGE_TEST_RELATIVE.as_posix(): {
                "sha256": hashlib.sha256(h2_test).hexdigest(),
                "size_bytes": len(h2_test),
            },
        },
        "patch_sha256": hashlib.sha256(patch).hexdigest(),
        "patch": patch,
        "_expected_payloads": {
            H2_PREPARER_RELATIVE.as_posix(): h2_preparer,
            H2_LINEAGE_TEST_RELATIVE.as_posix(): h2_test,
        },
    }


def _validate_h2_acceptance(code: Path, h2_head: str) -> dict[str, Any]:
    parent_record = (
        _git(code, ["rev-list", "--parents", "-n", "1", h2_head])
        .decode("ascii")
        .strip()
        .split()
    )
    _require(
        len(parent_record) == 2
        and parent_record[0] == h2_head
        and _HEAD_RE.fullmatch(parent_record[1]) is not None,
        "formal source is not a single-parent H2 commit",
    )
    h1_head = parent_record[1]
    changed_payload = _git(
        code,
        ["diff", "--no-renames", "--name-only", "-z", f"{h1_head}..{h2_head}"],
    )
    try:
        changed_paths = frozenset(
            value.decode("utf-8") for value in changed_payload.split(b"\0") if value
        )
    except UnicodeDecodeError as error:
        raise RuntimeError("H2 changed path is not UTF-8") from error
    _require(
        changed_paths == H2_CHANGED_PATHS,
        "H2 changed paths differ from the exact preparer-plus-lineage-test contract",
    )
    pins = _collect_h2_pins(h1_head)
    _require(
        pins.get("h1_source_head") == h1_head,
        "sealed pin artifact is not bound to the H1 parent",
    )
    assignments = pins.get("preparer_assignments")
    _require(isinstance(assignments, Mapping), "sealed H1 pins are absent")
    transition = _h2_transition_material(code, h1_head, assignments)
    transition.pop("patch")
    expected_payloads = transition.pop("_expected_payloads")
    for relative, expected in expected_payloads.items():
        for commit, label in ((h1_head, "H1"), (h2_head, "H2")):
            tree_entry = _git(code, ["ls-tree", "-z", commit, "--", relative])
            metadata, separator, observed_path = tree_entry.rstrip(b"\0").partition(
                b"\t"
            )
            fields = metadata.split()
            _require(
                separator == b"\t"
                and observed_path == relative.encode("utf-8")
                and len(fields) == 3
                and fields[0] == b"100644"
                and fields[1] == b"blob"
                and _HEAD_RE.fullmatch(fields[2].decode("ascii")) is not None,
                f"{label} transition path is not one regular non-executable blob: "
                f"{relative}",
            )
        observed = _git(code, ["show", f"{h2_head}:{relative}"])
        _require(
            observed == expected,
            f"H2 source differs from the exact generated pin transition: {relative}",
        )
    return {
        **transition,
        "h2_head": h2_head,
        "pin_artifact_sha256": pins["artifact_sha256"],
        "exact_17_assignment_transition": True,
        "no_other_preparer_source_change": True,
        "lineage_test_exact_generated_replacement": True,
        "single_parent_h2": True,
    }


def _require_runtime_resources(
    cotracker_repo: Path, cotracker_checkpoint: Path
) -> None:
    for path, label in (
        (ALIGNED_ROOT, "aligned Deform360 root"),
        (UPSTREAM, "Bayesian-PhysTwin upstream"),
        (OFFICIAL_PHYSTWIN, "official PhysTwin repository"),
        (SEMANTIC_MODEL, "semantic model"),
        (ALLTRACKER, "AllTracker source"),
    ):
        _canonical_directory(path, label=label)
    for path, label in (
        (SEMANTIC_MODEL_LOCK, "semantic model lock"),
        (ALLTRACKER_CHECKPOINT, "AllTracker checkpoint"),
        (SAM2_CHECKPOINT, "SAM2 checkpoint"),
    ):
        _read_regular(path, label=label)
    _require(
        _sha256_file(
            DEVELOPMENT_DECISION,
            label="development decision",
            mode=0o400,
        )
        == DEVELOPMENT_DECISION_SHA256,
        "development decision SHA-256 changed",
    )
    _verify_pinned_git_repository(
        DEFORM360_REPO,
        DEFORM360_HEAD,
        label="Deform360 processing repository",
        require_nonwritable=True,
    )
    _verify_pinned_git_repository(
        SAM2_REPOSITORY,
        SAM2_HEAD,
        label="SAM2 repository",
        allowed_ignored=frozenset({"checkpoints/sam2.1_hiera_small.pt"}),
        allow_external_pycache=True,
    )
    _verify_pinned_git_repository(
        cotracker_repo,
        COTRACKER_HEAD,
        label="CoTracker repository",
        allow_external_pycache=True,
    )
    _require(
        _sha256_file(SAM2_CHECKPOINT, label="SAM2 checkpoint")
        == SAM2_CHECKPOINT_SHA256,
        "SAM2 checkpoint SHA-256 changed",
    )
    _require(
        _sha256_file(cotracker_checkpoint, label="CoTracker checkpoint")
        == COTRACKER_CHECKPOINT_SHA256,
        "CoTracker checkpoint SHA-256 changed",
    )
    for path, label in (
        (PINNED_PYTHON, "pinned Python"),
        (BASH, "bash"),
        (FFMPEG, "ffmpeg"),
    ):
        _require_executable(path, label=label)


def _outcome_command(
    code: Path,
    role: str,
    *,
    cotracker_repo: Path,
    cotracker_checkpoint: Path,
    replacement_manifest: Path | None = None,
    confirmation_source_manifest: Path | None = None,
) -> list[str]:
    command = [
        os.fspath(PINNED_PYTHON),
        "-I",
        "-B",
        "-X",
        f"pycache_prefix={PYCACHE_PREFIX}",
        os.fspath(code / f"scripts/held/run_deform360_v8_{role}_outcomes.py"),
        "--deployed-code",
        os.fspath(code),
    ]
    if replacement_manifest is not None:
        command.extend(
            ["--replacement-source-manifest", os.fspath(replacement_manifest)]
        )
    if confirmation_source_manifest is not None:
        command.extend(
            [
                "--confirmation-source-manifest",
                os.fspath(confirmation_source_manifest),
            ]
        )
    command.extend(
        [
            "--aligned-root",
            os.fspath(ALIGNED_ROOT),
            "--deform360-repo",
            os.fspath(DEFORM360_REPO),
            "--sam2-repository",
            os.fspath(SAM2_REPOSITORY),
            "--sam2-checkpoint",
            os.fspath(SAM2_CHECKPOINT),
            "--cotracker-repo",
            os.fspath(cotracker_repo),
            "--cotracker-checkpoint",
            os.fspath(cotracker_checkpoint),
            "--device",
            "cuda:0",
            "--ffmpeg",
            "ffmpeg",
        ]
    )
    return command


def _require_role_completion(
    role: str,
    *,
    terminal_expected: bool,
    expected_outcome: str,
) -> None:
    completion = HELD_ROOT / role / f"{role}-outcome-integrity-completion.json"
    artifact, _ = _read_json_artifact(completion, label=f"{role} outcome completion")
    _require(
        artifact.get("schema_version") == 1
        and artifact.get("artifact_kind")
        == "Deform360HeldV8RoleOutcomeIntegrityCompletion"
        and artifact.get("protocol_id") == PROTOCOL_ID
        and artifact.get("status") == "role-outcome-integrity-complete"
        and artifact.get("role") == role
        and artifact.get("held_root") == str(HELD_ROOT)
        and artifact.get("role_root") == str(HELD_ROOT / role)
        and artifact.get("terminal_outcome") == expected_outcome,
        f"{role} auto-sealer completion identity changed",
    )
    if terminal_expected:
        terminal, _ = _read_json_artifact(
            TERMINAL_COMPLETION, label="terminal integrity completion"
        )
        _require(
            terminal.get("schema_version") == 1
            and terminal.get("artifact_kind")
            == "Deform360HeldV8TerminalRootIntegrityCompletion"
            and terminal.get("protocol_id") == PROTOCOL_ID
            and terminal.get("status") == "terminal-root-integrity-complete"
            and terminal.get("held_root") == str(HELD_ROOT)
            and terminal.get("terminal_role") == role
            and terminal.get("terminal_outcome") == expected_outcome,
            "terminal integrity completion disagrees with the semantic outcome",
        )
    else:
        _require_fresh(
            TERMINAL_COMPLETION,
            label="terminal integrity completion after calibration GO",
        )


def run_qualify(arguments: argparse.Namespace) -> CommandResult:
    _require_host()
    code, head = _verify_repository(arguments.code_root)
    _verify_pinned_git_repository(
        DEFORM360_REPO,
        DEFORM360_HEAD,
        label="Deform360 processing repository",
        require_nonwritable=True,
    )
    _canonical_directory(DEVELOPMENT_DATASET, label="qualification dataset")
    _require_executable(PINNED_PYTHON, label="pinned Python")
    root = _qualification_root(head)
    completion = Path(f"{root}-integrity-completion.json")
    _require_fresh(root, label="qualification root")
    _require_fresh(completion, label="qualification completion")
    _require_gpu_idle((1,))
    nofile = _set_soft_nofile()
    context = _new_phase("qualify", head)
    qualify, seal = _qualification_commands(code, head)
    qualification_rc = _run_logged_step(
        context,
        "qualification",
        qualify,
        cwd=code,
    )
    _require(
        qualification_rc in {0, QUALIFICATION_INCONCLUSIVE_EXIT_CODE},
        f"qualification failed technically with code {qualification_rc}",
    )
    sealer_rc = _run_logged_step(
        context,
        "qualification-sealer",
        seal,
        cwd=code,
    )
    _require(sealer_rc == 0, f"qualification sealer failed with code {sealer_rc}")
    admitted = qualification_rc == 0
    _validate_qualification_completion(head, admitted=admitted)
    summary = _finish_phase(
        context,
        returncode=qualification_rc,
        status="qualified" if admitted else "admission-inconclusive",
        details={"qualification_root": str(root), "rlimit_nofile": nofile},
    )
    return CommandResult(
        qualification_rc,
        {
            "phase": "qualify",
            "returncode": qualification_rc,
            "qualification_root": str(root),
            "phase_summary": str(summary),
        },
    )


def run_replay(arguments: argparse.Namespace) -> CommandResult:
    _require_host()
    code, head = _verify_repository(arguments.code_root)
    _validate_qualification_completion(head, admitted=True)
    _require_fresh(REPLAY_ROOT, label="attempt-5 admission replay root")
    context = _new_phase("replay", head)
    command = [
        os.fspath(PINNED_PYTHON),
        "-I",
        "-B",
        os.fspath(code / "scripts/held/replay_deform360_v81_external_admission.py"),
    ]
    returncode = _run_logged_step(context, "admission-replay", command, cwd=code)
    _require(returncode == 0, f"admission replay failed with code {returncode}")
    _collect_h2_pins(head)
    summary = _finish_phase(context, returncode=0, status="complete")
    return CommandResult(
        0,
        {
            "phase": "replay",
            "returncode": 0,
            "replay_root": str(REPLAY_ROOT),
            "phase_summary": str(summary),
        },
    )


def run_emit_h2_pins(arguments: argparse.Namespace) -> CommandResult:
    _require_host()
    code, head = _verify_repository(arguments.code_root)
    pins = _collect_h2_pins(head)
    transition = _h2_transition_material(
        code,
        head,
        pins["preparer_assignments"],
    )
    patch = transition.pop("patch")
    transition.pop("_expected_payloads")
    context = _new_phase("emit-h2-pins", head)
    report_path = context.root / "h2-pins.json"
    transition_path = context.root / "h2-source-transition.json"
    patch_path = context.root / "h2-source-transition.patch"
    _write_new_json(report_path, pins)
    transition_artifact = _signed_artifact(
        {
            "schema_version": 1,
            "artifact_kind": "Deform360HeldV81Attempt5H2SourceTransition",
            "protocol_id": PROTOCOL_ID,
            "execution_attempt": EXECUTION_ATTEMPT,
            "pin_artifact_sha256": pins["artifact_sha256"],
            **transition,
        }
    )
    _write_new_json(transition_path, transition_artifact)
    _write_new(patch_path, patch)
    summary = _finish_phase(
        context,
        returncode=0,
        status="complete",
        details={
            "pin_report": str(report_path),
            "pin_report_artifact_sha256": pins["artifact_sha256"],
            "source_transition": str(transition_path),
            "source_transition_artifact_sha256": transition_artifact["artifact_sha256"],
            "source_transition_patch": str(patch_path),
            "source_transition_patch_sha256": transition["patch_sha256"],
            "repository_edited": False,
        },
    )
    return CommandResult(
        0,
        {
            "phase": "emit-h2-pins",
            "returncode": 0,
            "pin_report": str(report_path),
            "pin_report_artifact_sha256": pins["artifact_sha256"],
            "source_transition": str(transition_path),
            "source_transition_artifact_sha256": transition_artifact["artifact_sha256"],
            "source_transition_patch": str(patch_path),
            "source_transition_patch_sha256": transition["patch_sha256"],
            "repository_edited": False,
            "phase_summary": str(summary),
        },
    )


def run_prepare(arguments: argparse.Namespace) -> CommandResult:
    _require_host()
    code, head = _verify_repository(arguments.code_root)
    h2_acceptance = _validate_h2_acceptance(code, head)
    _require_executable(PINNED_PYTHON, label="pinned Python")
    _require_fresh(HELD_ROOT, label="held-v8 root")
    _require_fresh(TERMINAL_COMPLETION, label="held-v8 terminal completion")
    context = _new_phase("prepare", head)
    base = [
        os.fspath(PINNED_PYTHON),
        "-I",
        "-B",
        os.fspath(code / "scripts/held/prepare_deform360_v8_lock.py"),
        "--source-code",
        os.fspath(code),
    ]
    preflight_rc = _run_logged_step(
        context,
        "lock-preflight",
        [*base, "--preflight"],
        cwd=code,
    )
    _require(preflight_rc == 0, f"lock preflight failed with code {preflight_rc}")
    _require_fresh(HELD_ROOT, label="held-v8 root after preflight")
    create_rc = _run_logged_step(
        context,
        "lock-create",
        [*base, "--create"],
        cwd=code,
    )
    _require(create_rc == 0, f"lock creation failed with code {create_rc}")
    deployed = HELD_ROOT / f"code-{head}"
    _canonical_directory(HELD_ROOT, label="held-v8 root")
    _read_json_artifact(
        HELD_ROOT / "calibration-lock.json",
        label="calibration protocol lock",
    )
    _require_deployed_repository(deployed)
    summary = _finish_phase(
        context,
        returncode=0,
        status="complete",
        details={
            "deployed_code": str(deployed),
            "h2_acceptance": h2_acceptance,
        },
    )
    return CommandResult(
        0,
        {
            "phase": "prepare",
            "returncode": 0,
            "deployed_code": str(deployed),
            "phase_summary": str(summary),
        },
    )


def _runtime_arguments(arguments: argparse.Namespace) -> tuple[Path, Path]:
    repository_argument = _absolute(
        arguments.cotracker_repo, label="CoTracker repository argument"
    )
    checkpoint_argument = _absolute(
        arguments.cotracker_checkpoint,
        label="CoTracker checkpoint argument",
    )
    _require(
        repository_argument == COTRACKER_REPOSITORY
        and checkpoint_argument == COTRACKER_CHECKPOINT,
        "formal CoTracker paths differ from the frozen workstation2 paths",
    )
    cotracker_repo = _canonical_directory(
        repository_argument,
        label="CoTracker repository",
    )
    checkpoint = checkpoint_argument
    _read_regular(checkpoint, label="CoTracker checkpoint")
    return cotracker_repo, checkpoint


def _precheck_calibration_freshness(code: Path) -> None:
    _canonical_directory(HELD_ROOT, label="held-v8 root")
    _read_json_artifact(
        HELD_ROOT / "calibration-lock.json",
        label="calibration protocol lock",
    )
    _require(code.parent == HELD_ROOT, "calibration code is outside held-v8")
    for path, label in (
        (HELD_ROOT / "replacement-source", "replacement source"),
        (HELD_ROOT / "calibration", "calibration result root"),
        (CONFIRMATION_SOURCE_ROOT, "confirmation source"),
        (CONFIRMATION_SOURCE_RUNTIME_ROOT, "confirmation source runtime"),
        (HELD_ROOT / "confirmation-lock.json", "confirmation lock"),
        (HELD_ROOT / "confirmation", "confirmation result root"),
        (TERMINAL_COMPLETION, "terminal integrity completion"),
    ):
        _require_fresh(path, label=label)


def _precheck_confirmation_go(code: Path) -> None:
    _canonical_directory(HELD_ROOT, label="held-v8 root")
    _read_json_artifact(
        HELD_ROOT / "calibration-lock.json",
        label="calibration protocol lock",
    )
    decision, _ = _read_json_artifact(
        HELD_ROOT / "calibration/calibration-gate-decision.json",
        label="calibration GO decision",
    )
    completion, _ = _read_json_artifact(
        HELD_ROOT / "calibration/calibration-outcome-integrity-completion.json",
        label="calibration outcome completion",
    )
    _require(
        decision.get("protocol_id") == PROTOCOL_ID
        and decision.get("role") == "calibration"
        and decision.get("decision") == "GO"
        and completion.get("status") == "role-outcome-integrity-complete"
        and completion.get("role") == "calibration"
        and completion.get("terminal_outcome") == "GO",
        "confirmation requires an integrity-complete calibration GO",
    )
    _require(code.parent == HELD_ROOT, "confirmation code is outside held-v8")
    for path, label in (
        (CONFIRMATION_SOURCE_ROOT, "confirmation source"),
        (CONFIRMATION_SOURCE_RUNTIME_ROOT, "confirmation source runtime"),
        (HELD_ROOT / "confirmation-lock.json", "confirmation lock"),
        (HELD_ROOT / "confirmation", "confirmation result root"),
        (TERMINAL_COMPLETION, "terminal integrity completion"),
    ):
        _require_fresh(path, label=label)


def run_calibrate(arguments: argparse.Namespace) -> CommandResult:
    _require_host()
    code, head = _require_deployed_repository(arguments.code_root)
    cotracker_repo, cotracker_checkpoint = _runtime_arguments(arguments)
    _require_runtime_resources(cotracker_repo, cotracker_checkpoint)
    _precheck_calibration_freshness(code)
    _validate_no_writable_held_descriptors()
    _require_gpu_idle((0, 1))
    context = _new_phase("calibrate", head)
    child_environment = _child_environment({"BPT_HELD_V8_CODE": str(code)})
    source_command = [
        os.fspath(PINNED_PYTHON),
        "-I",
        "-B",
        "-X",
        f"pycache_prefix={PYCACHE_PREFIX}",
        os.fspath(code / "scripts/held/run_deform360_v8_replacement_source.py"),
    ]
    source_rc = _run_logged_step(
        context,
        "replacement-source",
        source_command,
        cwd=code,
        environment=child_environment,
    )
    _require(source_rc == 0, f"replacement source failed with code {source_rc}")
    manifest = HELD_ROOT / "replacement-source/manifests/aligned-source.json"
    _read_json_artifact(manifest, label="aligned replacement-source manifest")
    _require_gpu_idle((0, 1))
    commands = [
        (
            "calibration-shard-0",
            [
                os.fspath(BASH),
                os.fspath(code / "scripts/held/run_deform360_v8_calibration_shard.sh"),
                "0",
                "0",
                os.fspath(manifest),
            ],
        ),
        (
            "calibration-shard-1",
            [
                os.fspath(BASH),
                os.fspath(code / "scripts/held/run_deform360_v8_calibration_shard.sh"),
                "1",
                "1",
                os.fspath(manifest),
            ],
        ),
    ]
    _run_concurrent_steps(
        context,
        commands,
        cwd=code,
        environment=child_environment,
    )
    _require_gpu_idle((0,))
    nofile = _set_soft_nofile()
    _validate_no_writable_held_descriptors()
    outcome_rc = _run_logged_step(
        context,
        "calibration-outcome",
        _outcome_command(
            code,
            "calibration",
            cotracker_repo=cotracker_repo,
            cotracker_checkpoint=cotracker_checkpoint,
            replacement_manifest=manifest,
        ),
        cwd=code,
    )
    _require(
        outcome_rc in {0, CALIBRATION_NO_GO_EXIT_CODE},
        f"calibration outcome failed technically with code {outcome_rc}",
    )
    status = "GO" if outcome_rc == 0 else "NO-GO"
    _require_role_completion(
        "calibration",
        terminal_expected=outcome_rc == CALIBRATION_NO_GO_EXIT_CODE,
        expected_outcome=status,
    )
    summary = _finish_phase(
        context,
        returncode=outcome_rc,
        status=status,
        details={
            "replacement_source_manifest": str(manifest),
            "auto_sealer_invoked_by_outcome_driver": True,
            "rlimit_nofile": nofile,
        },
    )
    return CommandResult(
        outcome_rc,
        {
            "phase": "calibrate",
            "returncode": outcome_rc,
            "decision": status,
            "phase_summary": str(summary),
        },
    )


def run_confirm(arguments: argparse.Namespace) -> CommandResult:
    _require_host()
    code, head = _require_deployed_repository(arguments.code_root)
    cotracker_repo, cotracker_checkpoint = _runtime_arguments(arguments)
    _require_runtime_resources(cotracker_repo, cotracker_checkpoint)
    _precheck_confirmation_go(code)
    _validate_no_writable_held_descriptors()
    _require_gpu_idle((0, 1))
    context = _new_phase("confirm", head)
    promotion = [
        os.fspath(PINNED_PYTHON),
        "-I",
        "-B",
        "-X",
        f"pycache_prefix={PYCACHE_PREFIX}",
        os.fspath(code / "scripts/held/run_deform360_v8_confirmation_outcomes.py"),
        "--promote-only",
        "--deployed-code",
        os.fspath(code),
    ]
    promotion_rc = _run_logged_step(
        context,
        "confirmation-promotion",
        promotion,
        cwd=code,
    )
    _require(
        promotion_rc == 0, f"confirmation promotion failed with code {promotion_rc}"
    )
    _read_json_artifact(
        HELD_ROOT / "confirmation-lock.json",
        label="confirmation protocol lock",
    )
    _require_fresh(HELD_ROOT / "confirmation", label="confirmation result root")
    child_environment = _child_environment({"BPT_HELD_V8_CODE": str(code)})
    source_command = [
        os.fspath(PINNED_PYTHON),
        "-I",
        "-B",
        "-X",
        f"pycache_prefix={PYCACHE_PREFIX}",
        os.fspath(code / "scripts/held/run_deform360_v8_confirmation_source.py"),
        "--deployed-code",
        os.fspath(code),
    ]
    source_rc = _run_logged_step(
        context,
        "confirmation-source",
        source_command,
        cwd=code,
        environment=child_environment,
    )
    _require(source_rc == 0, f"confirmation source failed with code {source_rc}")
    source_manifest, _ = _read_json_artifact(
        CONFIRMATION_SOURCE_MANIFEST,
        label="confirmation source cohort manifest",
    )
    _require(
        source_manifest.get("schema_version") == 1
        and source_manifest.get("artifact_kind")
        == "Deform360HeldV8ConfirmationAlignedSourceCohort"
        and source_manifest.get("protocol_id") == PROTOCOL_ID
        and source_manifest.get("status") == "confirmation-source-cohort-complete"
        and source_manifest.get("role") == "confirmation",
        "confirmation source cohort manifest identity changed",
    )
    _require_gpu_idle((0, 1))
    commands = [
        (
            "confirmation-shard-0",
            [
                os.fspath(BASH),
                os.fspath(code / "scripts/held/run_deform360_v8_confirmation_shard.sh"),
                "0",
                "0",
                os.fspath(CONFIRMATION_SOURCE_MANIFEST),
            ],
        ),
        (
            "confirmation-shard-1",
            [
                os.fspath(BASH),
                os.fspath(code / "scripts/held/run_deform360_v8_confirmation_shard.sh"),
                "1",
                "1",
                os.fspath(CONFIRMATION_SOURCE_MANIFEST),
            ],
        ),
    ]
    _run_concurrent_steps(
        context,
        commands,
        cwd=code,
        environment=child_environment,
    )
    _require_gpu_idle((0,))
    nofile = _set_soft_nofile()
    _validate_no_writable_held_descriptors()
    outcome_rc = _run_logged_step(
        context,
        "confirmation-outcome",
        _outcome_command(
            code,
            "confirmation",
            cotracker_repo=cotracker_repo,
            cotracker_checkpoint=cotracker_checkpoint,
            confirmation_source_manifest=CONFIRMATION_SOURCE_MANIFEST,
        ),
        cwd=code,
    )
    _require(
        outcome_rc in {0, CONFIRMATION_NOT_CONFIRMED_EXIT_CODE},
        f"confirmation outcome failed technically with code {outcome_rc}",
    )
    status = "CONFIRMED" if outcome_rc == 0 else "NOT-CONFIRMED"
    _require_role_completion(
        "confirmation", terminal_expected=True, expected_outcome=status
    )
    summary = _finish_phase(
        context,
        returncode=outcome_rc,
        status=status,
        details={
            "confirmation_source_manifest": str(CONFIRMATION_SOURCE_MANIFEST),
            "auto_sealer_invoked_by_outcome_driver": True,
            "rlimit_nofile": nofile,
        },
    )
    return CommandResult(
        outcome_rc,
        {
            "phase": "confirm",
            "returncode": outcome_rc,
            "decision": status,
            "phase_summary": str(summary),
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    handlers = {
        "qualify": run_qualify,
        "replay": run_replay,
        "emit-h2-pins": run_emit_h2_pins,
        "prepare": run_prepare,
        "calibrate": run_calibrate,
        "confirm": run_confirm,
    }
    for name, handler in handlers.items():
        phase = subparsers.add_parser(name)
        phase.add_argument("--code-root", type=Path, required=True)
        if name in {"calibrate", "confirm"}:
            phase.add_argument("--cotracker-repo", type=Path, required=True)
            phase.add_argument("--cotracker-checkpoint", type=Path, required=True)
        phase.set_defaults(handler=handler)
    return parser


def _require_launcher_runtime() -> None:
    executable = Path(os.path.abspath(sys.executable))
    _require(
        executable == PINNED_PYTHON
        and sys.flags.isolated == 1
        and sys.flags.ignore_environment == 1
        and sys.flags.no_user_site == 1
        and sys.flags.dont_write_bytecode == 1,
        "run the attempt-5 launcher with the pinned Python using -I -B",
    )


def main(argv: Sequence[str] | None = None) -> int:
    os.umask(0o077)
    try:
        _require_launcher_runtime()
        arguments = _parser().parse_args(argv)
        result = arguments.handler(arguments)
        print(json.dumps(dict(result.payload), sort_keys=True), flush=True)
        return result.returncode
    except BaseException as error:
        failure_summary = _finish_failed_phase(error)
        print(
            json.dumps(
                {
                    "event": "FAIL_CLOSED",
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "phase_summary": (
                        None if failure_summary is None else str(failure_summary)
                    ),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return TECHNICAL_FAILURE_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
