#!/usr/bin/env python3
"""Run the permit-gated held-v5 Deform360 confirmation outcome phase.

This operator driver has one deliberately narrow job:

1. in --promote-only mode, derive the confirmation lock only from the
   immutable, validated calibration GO decision and then stop;
2. otherwise validate the exact six-case online-prediction cohort and obtain
   the live confirmation outcome capability;
3. only after that barrier, create each locked official target
   through the pinned reconstruction adapter;
4. score the fixed primary method and comparator; and
5. write immutable score evidence and the frozen final confirmation decision.

The promotion path accepts no prediction seal or target-bearing runtime path.
The ``--dry-run-barrier-only`` mode stops immediately after the six-seal
barrier and does not
load the post-barrier reconstruction or scoring API into the outcome process.
The source-only deployment verifier runs in a separate isolated subprocess.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence


EXPECTED_PROTOCOL_ID = "deform360-held-online-belief-v5"
EXPECTED_V5_BINDING_COUNT = 112
NOT_CONFIRMED_EXIT_CODE = 4
_CANONICAL_HELD_BASE = Path("/mnt/corsair/florianpfaff/bpt-online-belief-v1")
_CANONICAL_HELD_ROOT = _CANONICAL_HELD_BASE / "held-v5"
_CANONICAL_V1_LOCK = _CANONICAL_HELD_BASE / "held-v1" / "calibration-lock.json"
_CANONICAL_V1_REPORT = (
    _CANONICAL_HELD_BASE / "held-v1" / "v1-preoutcome-feasibility-report.json"
)
_CANONICAL_V2_WITHDRAWAL_REPORT = (
    _CANONICAL_HELD_BASE / "held-v4" / "v2-design-withdrawal-report.json"
)
_CANONICAL_V3_BOUNDARY_INCIDENT_REPORT = (
    _CANONICAL_HELD_BASE / "held-v4" / "v3-prelock-boundary-incident-report.json"
)
_CANONICAL_V4_LOCK = _CANONICAL_HELD_BASE / "held-v4" / "calibration-lock.json"
_CANONICAL_V4_EXECUTION_WITHDRAWAL_REPORT = (
    _CANONICAL_HELD_BASE / "held-v4" / "v4-execution-withdrawal-report.json"
)
_CANONICAL_CALIBRATION_LOCK = _CANONICAL_HELD_ROOT / "calibration-lock.json"
_CANONICAL_CALIBRATION_DECISION = (
    _CANONICAL_HELD_ROOT / "calibration" / "calibration-gate-decision.json"
)
_CANONICAL_CONFIRMATION_LOCK = _CANONICAL_HELD_ROOT / "confirmation-lock.json"
_CANONICAL_ALIGNED_ROOT = Path(
    "/mnt/lexar4tb/datasets/deform360/data-7fea8e2/replication-v1/aligned"
)
_PINNED_PYTHON = "/mnt/corsair/florianpfaff/bpt-held-v5-runtimes/bpt-gpu-pip-4948737892f77c6a9496795e6c3f25b92fcea466ddb7b5f1e9c1b0de1137f004/bin/python"
_PINNED_PATH = "/usr/local/bin:/usr/bin:/bin"
_PYCACHE_PREFIX = "/nonexistent/bpt-held-v5-pycache"
# Updated mechanically after the final lock-operator source is frozen.
EXPECTED_LOCK_OPERATOR_SHA256 = (
    "13fc045d98cae39e83023fe9cfd1b9c34d53e4285000c7fd8f3779728f4853e2"
)
_NORMALIZED_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "PYTHONSAFEPATH": "1",
    "PYTHONHASHSEED": "0",
    "PYNPUT_BACKEND": "dummy",
    "PYOPENGL_PLATFORM": "egl",
    "PYTHONPYCACHEPREFIX": _PYCACHE_PREFIX,
    "WANDB_MODE": "disabled",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _emit(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def _sha256_regular_file(path: Path, role: str) -> str:
    absolute = Path(os.path.abspath(os.fspath(path)))
    before = os.lstat(absolute)
    _require(stat.S_ISREG(before.st_mode), f"{role} is not a regular file")
    descriptor = os.open(
        absolute,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        after = os.fstat(descriptor)
        _require(
            stat.S_ISREG(after.st_mode)
            and (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino),
            f"{role} changed while opening",
        )
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _require_mode_0400(path: Path, role: str) -> None:
    _require(
        stat.S_IMODE(os.lstat(path).st_mode) == 0o400,
        f"{role} mode is not exactly 0400",
    )


def _minimal_environment(*, include_outcome_runtime: bool) -> dict[str, str]:
    environment = {
        "HOME": "/home/florianpfaff",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "LOGNAME": "florianpfaff",
        "PATH": _PINNED_PATH,
        "TMPDIR": "/tmp",
        "USER": "florianpfaff",
        **_NORMALIZED_ENVIRONMENT,
    }
    if include_outcome_runtime:
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_devices is not None:
            _require(
                bool(visible_devices)
                and all(
                    component.isdigit() for component in visible_devices.split(",")
                ),
                "CUDA_VISIBLE_DEVICES is not a numeric device list",
            )
            environment["CUDA_VISIBLE_DEVICES"] = visible_devices
    return environment


def _lock_verifier_environment() -> dict[str, str]:
    return {
        "GIT_OPTIONAL_LOCKS": "0",
        "HOME": "/home/florianpfaff",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "LOGNAME": "florianpfaff",
        "PATH": _PINNED_PATH,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPYCACHEPREFIX": _PYCACHE_PREFIX,
        "TMPDIR": "/tmp",
        "USER": "florianpfaff",
    }


def _verify_locked_deployment(arguments: "DriverArguments") -> Path:
    """Verify source-only deployment state without touching outcome paths."""

    _require(
        not os.path.lexists("/nonexistent")
        and not os.path.lexists(_PYCACHE_PREFIX),
        "reserved held-v5 pycache prefix is no longer unavailable",
    )
    root_argument = Path(arguments.held_root)
    _require(
        root_argument == _CANONICAL_HELD_ROOT
        and root_argument.is_dir()
        and not root_argument.is_symlink()
        and root_argument.resolve() == _CANONICAL_HELD_ROOT,
        "held root is not the canonical fresh held-v5 root",
    )
    code_argument = Path(arguments.deployed_code)
    _require(
        code_argument.is_dir()
        and not code_argument.is_symlink()
        and code_argument.resolve() == code_argument,
        "deployed v5 code is absent, linked, or non-canonical",
    )
    _require(
        code_argument.parent == _CANONICAL_HELD_ROOT
        and code_argument.name.startswith("code-")
        and len(code_argument.name.removeprefix("code-")) in {40, 64}
        and all(
            character in "0123456789abcdef"
            for character in code_argument.name.removeprefix("code-")
        ),
        "deployed code is outside canonical held-v5 code-$HEAD",
    )
    lock = Path(arguments.calibration_lock)
    _require(
        lock == _CANONICAL_CALIBRATION_LOCK
        and not lock.is_symlink(),
        "calibration lock is outside the canonical held-v5 root",
    )
    _require_mode_0400(lock, "held-v5 calibration lock")
    for path, role in (
        (_CANONICAL_V1_LOCK, "sealed v1 parent lock"),
        (_CANONICAL_V1_REPORT, "sealed v1 pre-outcome report"),
        (_CANONICAL_V2_WITHDRAWAL_REPORT, "sealed v2 withdrawal report"),
        (
            _CANONICAL_V3_BOUNDARY_INCIDENT_REPORT,
            "sealed v3 boundary-incident report",
        ),
        (_CANONICAL_V4_LOCK, "sealed v4 calibration lock"),
        (
            _CANONICAL_V4_EXECUTION_WITHDRAWAL_REPORT,
            "sealed v4 execution-withdrawal report",
        ),
    ):
        _require(
            path.is_file() and not path.is_symlink(), f"{role} is absent or linked"
        )
        _require(path.resolve() == path, f"{role} path is non-canonical")
        _require_mode_0400(path, role)

    operator_dir = code_argument / "scripts" / "held"
    lock_operator = operator_dir / "prepare_deform360_v5_lock.py"
    expected_driver = operator_dir / "run_deform360_v5_confirmation_outcomes.py"
    observed_driver = Path(__file__)
    _require(
        not observed_driver.is_symlink()
        and observed_driver.resolve() == expected_driver,
        "outcome driver is not the tracked deployed v5 source",
    )
    _require(
        lock_operator.is_file() and not lock_operator.is_symlink(),
        "tracked v5 lock operator is absent or linked",
    )
    for path in (lock_operator, expected_driver):
        _require(
            os.lstat(path).st_mode & 0o222 == 0,
            f"tracked operator is writable: {path.name}",
        )
    _require(
        _sha256_regular_file(lock_operator, "tracked v5 lock operator")
        == EXPECTED_LOCK_OPERATOR_SHA256,
        "lock operator checksum differs from the audited v5 verifier",
    )

    command = [
        _PINNED_PYTHON,
        "-B",
        "-X",
        f"pycache_prefix={_PYCACHE_PREFIX}",
        "-I",
        os.fspath(lock_operator),
        "--v1-lock",
        os.fspath(_CANONICAL_V1_LOCK),
        "--v1-report",
        os.fspath(_CANONICAL_V1_REPORT),
        "--v2-withdrawal-report",
        os.fspath(_CANONICAL_V2_WITHDRAWAL_REPORT),
        "--v3-boundary-incident-report",
        os.fspath(_CANONICAL_V3_BOUNDARY_INCIDENT_REPORT),
        "--v4-lock",
        os.fspath(_CANONICAL_V4_LOCK),
        "--v4-execution-withdrawal-report",
        os.fspath(_CANONICAL_V4_EXECUTION_WITHDRAWAL_REPORT),
        "--deployed-code",
        os.fspath(code_argument),
        "--output-lock",
        os.fspath(lock),
        "--verify-existing-lock",
    ]
    completed = subprocess.run(
        command,
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_lock_verifier_environment(),
        timeout=300,
    )
    _require(
        completed.returncode == 0,
        "source-only v5 deployment verification failed: "
        + completed.stderr.decode("utf-8", "replace").strip(),
    )
    try:
        verification = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("v5 deployment verifier returned malformed JSON") from error
    _require(
        isinstance(verification, Mapping)
        and verification.get("operation") == "verified_existing_lock"
        and verification.get("protocol_id") == EXPECTED_PROTOCOL_ID
        and verification.get("binding_count") == EXPECTED_V5_BINDING_COUNT
        and verification.get("method_head") == code_argument.name.removeprefix("code-")
        and verification.get("lock_file_sha256")
        == _sha256_regular_file(lock, "held-v5 calibration lock"),
        "v5 deployment-verifier identity disagrees with requested deployment",
    )
    return code_argument / "src"


def _require_package_provenance(source: Path) -> None:
    source = source.resolve()
    for name, module in tuple(sys.modules.items()):
        if name != "bayesian_phystwin" and not name.startswith("bayesian_phystwin."):
            continue
        module_file = getattr(module, "__file__", None)
        _require(bool(module_file), f"deployed module has no source path: {name}")
        observed = Path(str(module_file)).resolve()
        try:
            observed.relative_to(source)
        except ValueError as error:
            raise ValueError(
                f"deployed module imported outside snapshot: {name}: {observed}"
            ) from error


@dataclass(frozen=True)
class DriverArguments:
    deployed_code: str
    held_root: str
    calibration_lock: str
    calibration_decision: str
    confirmation_lock: str
    online_seals: tuple[str, ...]
    promote_only: bool
    dry_run_barrier_only: bool
    aligned_root: str | None = None
    deform360_repo: str | None = None
    sam2_repository: str | None = None
    sam2_checkpoint: str | None = None
    cotracker_repo: str | None = None
    cotracker_checkpoint: str | None = None
    device: str = "cuda:0"
    ffmpeg: str = "ffmpeg"


@dataclass(frozen=True)
class HeldLayout:
    root: Path
    lock: Path
    seal_paths: Mapping[str, Path]
    outcomes_root: Path
    evidence_path: Path
    decision_path: Path


@dataclass(frozen=True)
class PostBarrierApi:
    backend_type: Callable[..., Any]
    plan_target_operation: Callable[..., Any]
    target_operation_type: Callable[..., Any]
    score_and_create_gate: Callable[..., Any]


def _validate_lineage_paths(arguments: DriverArguments) -> None:
    _require(
        Path(arguments.calibration_lock) == _CANONICAL_CALIBRATION_LOCK,
        "calibration-lock spelling is not canonical",
    )
    decision = Path(arguments.calibration_decision)
    _require(
        decision == _CANONICAL_CALIBRATION_DECISION,
        "calibration-decision spelling is not canonical",
    )
    decision_stat = os.lstat(decision)
    _require(
        stat.S_ISREG(decision_stat.st_mode)
        and decision.resolve() == _CANONICAL_CALIBRATION_DECISION
        and decision_stat.st_mode & 0o222 == 0,
        "calibration decision is linked, writable, or non-canonical",
    )
    _require(
        Path(arguments.confirmation_lock) == _CANONICAL_CONFIRMATION_LOCK,
        "confirmation-lock spelling is not canonical",
    )


def _promote_confirmation_lock(arguments: DriverArguments, protocol: Any) -> int:
    """Create the write-once confirmation lock from the exact validated GO."""

    _validate_lineage_paths(arguments)
    _require(not arguments.online_seals, "promotion accepts no online seals")
    _require(
        not arguments.dry_run_barrier_only,
        "promotion and barrier-only modes are mutually exclusive",
    )
    forbidden_runtime = {
        "aligned_root": arguments.aligned_root,
        "deform360_repo": arguments.deform360_repo,
        "sam2_repository": arguments.sam2_repository,
        "sam2_checkpoint": arguments.sam2_checkpoint,
        "cotracker_repo": arguments.cotracker_repo,
        "cotracker_checkpoint": arguments.cotracker_checkpoint,
    }
    _require(
        all(value is None for value in forbidden_runtime.values()),
        "promotion accepts no target-bearing runtime paths",
    )
    destination = Path(arguments.confirmation_lock)
    _require(
        not os.path.lexists(destination),
        "confirmation lock already exists; promotion is write-once",
    )
    created = protocol.create_confirmation_protocol_lock(
        destination,
        arguments.calibration_lock,
        arguments.calibration_decision,
    )
    _require(
        created.get("stage") == "confirmation"
        and created.get("confirmation_access_authorized") is True,
        "protocol did not create a confirmation lock",
    )
    os.chmod(destination, 0o400, follow_symlinks=False)
    _require_mode_0400(destination, "held-v5 confirmation lock")
    validated = protocol.load_held_protocol_lock(destination)
    _require(
        validated.get("artifact_sha256") == created.get("artifact_sha256"),
        "confirmation lock changed while freezing",
    )
    _emit(
        "CONFIRMATION_LOCK_PROMOTED_FROM_VALIDATED_CALIBRATION_GO",
        path=str(destination),
        artifact_sha256=validated["artifact_sha256"],
        file_sha256=_sha256_regular_file(destination, "confirmation lock"),
        confirmation_payload_read=False,
    )
    return 0


def _validate_confirmation_lock_lineage(
    arguments: DriverArguments, protocol: Any
) -> None:
    """Validate the derived child without opening a confirmation payload."""

    _validate_lineage_paths(arguments)
    confirmation_lock = Path(arguments.confirmation_lock)
    _require(
        confirmation_lock.is_file()
        and not confirmation_lock.is_symlink()
        and confirmation_lock.resolve() == _CANONICAL_CONFIRMATION_LOCK,
        "confirmation lock is absent, linked, or non-canonical",
    )
    _require_mode_0400(confirmation_lock, "held-v5 confirmation lock")
    confirmation = protocol.load_held_protocol_lock(confirmation_lock)
    _require(
        confirmation.get("stage") == "confirmation"
        and confirmation["parent_calibration_lock"]["path"]
        == str(_CANONICAL_CALIBRATION_LOCK)
        and confirmation["calibration_gate_evidence"]["path"]
        == str(_CANONICAL_CALIBRATION_DECISION),
        "confirmation lock lineage changed",
    )


@dataclass(frozen=True)
class _ProgressCallback:
    case_name: str
    operation: str
    callback: Callable[[], Any]
    outcomes_root: Path
    outcomes_root_identity: tuple[int, int]

    def __call__(self) -> Any:
        _require_directory_identity(
            self.outcomes_root,
            self.outcomes_root_identity,
            "confirmation outcomes root",
        )
        _emit(
            "CONFIRMATION_TARGET_OPERATION_START",
            case_name=self.case_name,
            operation=self.operation,
        )
        value = self.callback()
        _require_directory_identity(
            self.outcomes_root,
            self.outcomes_root_identity,
            "confirmation outcomes root",
        )
        _emit(
            "CONFIRMATION_TARGET_OPERATION_COMPLETE",
            case_name=self.case_name,
            operation=self.operation,
        )
        return value


def _parse_seal_assignments(
    assignments: Sequence[str], expected_cases: Sequence[str]
) -> dict[str, str]:
    seals: dict[str, str] = {}
    for assignment in assignments:
        case_name, separator, path = assignment.partition("=")
        _require(
            bool(separator) and bool(case_name) and bool(path),
            "each --online-seal must be CASE=PATH",
        )
        _require(case_name not in seals, f"duplicate online seal: {case_name}")
        seals[case_name] = path
    expected = set(expected_cases)
    observed = set(seals)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    _require(
        not missing and not extra and len(seals) == len(expected_cases),
        "online seals must be the exact locked confirmation cohort; "
        f"missing={missing}, extra={extra}",
    )
    return seals


def _prevalidate_canonical_online_seals(
    held_root: str,
    seal_strings: Mapping[str, str],
    expected_cases: Sequence[str],
) -> None:
    """Reject path substitution without opening any seal or protected payload."""

    root = Path(held_root)
    _require(root.is_absolute(), "held root spelling must be absolute")
    for case_name in expected_cases:
        canonical = (
            root
            / "confirmation"
            / "cases"
            / case_name
            / "online"
            / "online_prediction_seal.json"
        )
        supplied = Path(seal_strings[case_name])
        _require(
            supplied == canonical,
            f"online seal spelling is not the canonical held path: {case_name}",
        )
        before = os.lstat(canonical)
        _require(
            stat.S_ISREG(before.st_mode),
            f"online seal is not a regular non-symlink file: {case_name}",
        )
        parent = canonical.parent
        while True:
            observed = os.lstat(parent)
            _require(
                stat.S_ISDIR(observed.st_mode),
                f"online seal parent is linked or non-directory: {case_name}",
            )
            if parent == root:
                break
            _require(
                root in parent.parents,
                f"online seal parent escaped the held root: {case_name}",
            )
            parent = parent.parent


def _validate_held_layout_after_barrier(
    arguments: DriverArguments,
    seal_strings: Mapping[str, str],
    expected_cases: Sequence[str],
) -> HeldLayout:
    """Validate operator layout only after outcome authorization succeeds."""

    root_argument = Path(arguments.held_root)
    _require(
        root_argument.is_dir() and not root_argument.is_symlink(),
        "held-v5 root must be an existing non-symlink directory",
    )
    root = root_argument.resolve()
    lock = Path(arguments.confirmation_lock).resolve()
    _require(
        lock == root / "confirmation-lock.json",
        "confirmation lock is outside the canonical held-v5 root",
    )
    normalized: dict[str, Path] = {}
    for case_name in expected_cases:
        path = Path(seal_strings[case_name]).resolve()
        canonical = (
            root
            / "confirmation"
            / "cases"
            / case_name
            / "online"
            / "online_prediction_seal.json"
        )
        _require(
            path == canonical,
            f"online seal is outside the canonical held-v5 case path: {case_name}",
        )
        normalized[case_name] = path
    confirmation_root = root / "confirmation"
    return HeldLayout(
        root=root,
        lock=lock,
        seal_paths=normalized,
        outcomes_root=confirmation_root / "outcomes",
        evidence_path=confirmation_root / "confirmation-score-evidence.json",
        decision_path=confirmation_root / "confirmation-final-decision.json",
    )


def _require_full_runtime_arguments(arguments: DriverArguments) -> None:
    required = {
        "aligned_root": arguments.aligned_root,
        "deform360_repo": arguments.deform360_repo,
        "sam2_repository": arguments.sam2_repository,
        "sam2_checkpoint": arguments.sam2_checkpoint,
        "cotracker_repo": arguments.cotracker_repo,
        "cotracker_checkpoint": arguments.cotracker_checkpoint,
    }
    missing = sorted(name for name, value in required.items() if not value)
    _require(not missing, f"full outcome run lacks runtime arguments: {missing}")


def _validate_outcome_runtime_after_barrier(arguments: DriverArguments) -> None:
    """Validate target-bearing runtime paths only after authorization."""

    _require(
        os.uname().nodename == "workstation2",
        "formal outcome reconstruction must run on gpuserver6000/workstation2",
    )
    _require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "0",
        "formal outcome reconstruction is bound to physical GPU 0",
    )
    aligned = Path(str(arguments.aligned_root))
    _require(
        aligned == _CANONICAL_ALIGNED_ROOT
        and aligned.is_dir()
        and not aligned.is_symlink()
        and aligned.resolve() == _CANONICAL_ALIGNED_ROOT,
        "aligned target root is not the exact canonical dataset root",
    )
    _require(arguments.device == "cuda:0", "formal outcome device must be cuda:0")
    _require(arguments.ffmpeg == "ffmpeg", "formal ffmpeg spelling changed")


def _prepare_fresh_gate_outputs_after_barrier(layout: HeldLayout) -> None:
    confirmation_root = layout.outcomes_root.parent
    _require(
        confirmation_root.is_dir()
        and not confirmation_root.is_symlink()
        and confirmation_root.resolve() == confirmation_root,
        "confirmation output root is absent, linked, or non-canonical",
    )
    _require(
        not os.path.lexists(layout.evidence_path),
        f"immutable score evidence already exists: {layout.evidence_path}",
    )
    _require(
        not os.path.lexists(layout.decision_path),
        f"immutable gate decision already exists: {layout.decision_path}",
    )
    _require(
        not os.path.lexists(layout.outcomes_root),
        f"outcome root already exists; exact v5 outcome resume is forbidden: {layout.outcomes_root}",
    )
    claim = confirmation_root / ".outcome-phase.claim"
    _require(
        not os.path.lexists(claim),
        "confirmation outcome phase was already claimed",
    )
    os.mkdir(claim, mode=0o500)
    os.mkdir(layout.outcomes_root, mode=0o700)
    observed = os.lstat(layout.outcomes_root)
    _require(
        stat.S_ISDIR(observed.st_mode)
        and not layout.outcomes_root.is_symlink()
        and layout.outcomes_root.resolve() == layout.outcomes_root,
        "fresh outcome root is linked or non-canonical",
    )


def _require_directory_identity(
    path: Path,
    expected: tuple[int, int] | None,
    role: str,
) -> tuple[int, int]:
    observed = os.lstat(path)
    identity = (observed.st_dev, observed.st_ino)
    _require(
        stat.S_ISDIR(observed.st_mode)
        and path.resolve() == path
        and (expected is None or identity == expected),
        f"{role} inode changed",
    )
    return identity


def _aligned_episode_path(aligned_root: str, case_name: str) -> str:
    object_id, separator, episode = case_name.rpartition("-ep")
    _require(
        separator == "-ep" and len(episode) == 4 and episode.isdigit(),
        f"invalid locked case name: {case_name}",
    )
    root = Path(os.path.abspath(aligned_root))
    object_dir = root / object_id
    episode_dir = object_dir / f"episode_{episode}"
    for path, role in (
        (root, "aligned root"),
        (object_dir, "aligned object directory"),
        (episode_dir, "aligned episode directory"),
    ):
        observed = os.lstat(path)
        _require(
            stat.S_ISDIR(observed.st_mode) and path.resolve() == path,
            f"{role} is linked, absent, or non-canonical: {path}",
        )
    _require(
        object_dir.parent == root
        and episode_dir.parent == object_dir
        and episode_dir.name == f"episode_{episode}",
        f"aligned episode escaped its exact lexical ancestry: {case_name}",
    )
    return os.fspath(episode_dir)


def _execute_driver(
    arguments: DriverArguments,
    protocol: Any,
    load_post_barrier_api: Callable[[], PostBarrierApi],
    validate_post_barrier_runtime: Callable[[DriverArguments], None] = (
        _validate_outcome_runtime_after_barrier
    ),
    validate_prebarrier_confirmation: Callable[[DriverArguments, Any], None] = (
        _validate_confirmation_lock_lineage
    ),
) -> int:
    """Execute with injectable APIs so barrier ordering can be self-checked."""

    _require(
        protocol.PROTOCOL_ID == EXPECTED_PROTOCOL_ID,
        "deployed protocol is not held-v5",
    )
    _require(not arguments.promote_only, "outcome path cannot run in promotion mode")
    validate_prebarrier_confirmation(arguments, protocol)
    expected_cases = tuple(protocol.CONFIRMATION_CASE_NAMES)
    _require(len(expected_cases) == 6, "held-v5 confirmation cohort is not six cases")
    seal_strings = _parse_seal_assignments(arguments.online_seals, expected_cases)
    _prevalidate_canonical_online_seals(
        arguments.held_root,
        seal_strings,
        expected_cases,
    )
    if not arguments.dry_run_barrier_only:
        _require_full_runtime_arguments(arguments)

    # This is intentionally the first artifact API call.  No target/outcome
    # path has been derived, inspected, created, imported, or opened above.
    permit = protocol.authorize_outcome_phase(
        arguments.confirmation_lock,
        seal_strings,
        role="confirmation",
    )
    layout = _validate_held_layout_after_barrier(
        arguments,
        seal_strings,
        expected_cases,
    )
    _emit(
        "CONFIRMATION_COHORT_BARRIER_VALIDATED",
        protocol_id=protocol.PROTOCOL_ID,
        role="confirmation",
        case_count=len(expected_cases),
        cohort_barrier_sha256=permit.cohort_barrier_sha256,
        lock=str(layout.lock),
    )
    if arguments.dry_run_barrier_only:
        _emit(
            "DRY_RUN_COMPLETE_OUTCOMES_UNOPENED",
            confirmation_payload_read=False,
            target_or_outcome_path_inspected=False,
        )
        return 0

    # From here onward the complete cohort permit exists.  Importing and using
    # the reconstruction/scoring adapters is now allowed.
    validate_post_barrier_runtime(arguments)
    post = load_post_barrier_api()
    _prepare_fresh_gate_outputs_after_barrier(layout)
    outcomes_root_identity = _require_directory_identity(
        layout.outcomes_root,
        None,
        "confirmation outcomes root",
    )
    backend = post.backend_type(
        deform360_repo=arguments.deform360_repo,
        sam2_repository=arguments.sam2_repository,
        sam2_checkpoint=arguments.sam2_checkpoint,
        cotracker_repo=arguments.cotracker_repo,
        cotracker_checkpoint=arguments.cotracker_checkpoint,
        device=arguments.device,
        ffmpeg=arguments.ffmpeg,
    )
    operations: dict[str, Any] = {}
    for case_name in expected_cases:
        output_dir = layout.outcomes_root / case_name
        _require(
            output_dir.parent == layout.outcomes_root
            and not os.path.lexists(output_dir),
            f"outcome case path is not fresh and contained: {case_name}",
        )
        aligned_episode = _aligned_episode_path(str(arguments.aligned_root), case_name)
        planned = post.plan_target_operation(
            permit,
            case_name=case_name,
            aligned_episode_dir=aligned_episode,
            output_dir=output_dir,
            backend=backend,
        )
        _require(
            getattr(planned, "operation", None) == "create"
            and callable(getattr(planned, "callback", None)),
            f"official adapter did not return a fresh CREATE operation: {case_name}",
        )
        operations[case_name] = post.target_operation_type(
            operation=planned.operation,
            callback=_ProgressCallback(
                case_name=case_name,
                operation=planned.operation,
                callback=planned.callback,
                outcomes_root=layout.outcomes_root,
                outcomes_root_identity=outcomes_root_identity,
            ),
        )
        _emit(
            "CONFIRMATION_TARGET_OPERATION_PLANNED",
            case_name=case_name,
            operation=planned.operation,
        )
    _require(
        tuple(operations) == expected_cases,
        "TargetOperations are not in exact locked confirmation order",
    )

    decision, evidence, records = post.score_and_create_gate(
        layout.decision_path,
        permit,
        operations,
        evidence_path=layout.evidence_path,
    )
    _require(
        tuple(records) == expected_cases,
        "scorer did not return all cases in exact locked order",
    )
    _require(
        decision.get("decision") in {"CONFIRMED", "NOT_CONFIRMED"},
        "frozen confirmation gate returned an invalid decision",
    )
    _emit(
        "CONFIRMATION_SCORE_EVIDENCE_WRITTEN",
        path=str(layout.evidence_path),
        artifact_sha256=evidence.get("artifact_sha256"),
    )
    _emit(
        "CONFIRMATION_GATE_DECISION_WRITTEN",
        path=str(layout.decision_path),
        artifact_sha256=decision.get("artifact_sha256"),
        decision=decision["decision"],
        summary=decision.get("summary"),
        confirmation_payload_read=True,
    )
    if decision["decision"] == "NOT_CONFIRMED":
        _emit(
            "CONFIRMATION_GATE_NOT_CONFIRMED",
            exit_code=NOT_CONFIRMED_EXIT_CODE,
        )
        return NOT_CONFIRMED_EXIT_CODE
    _emit(
        "CONFIRMATION_GATE_CONFIRMED",
        one_sided_sign_test_p=decision["summary"]["one_sided_sign_test_p"],
    )
    return 0


def _load_protocol(deployed_code: str) -> tuple[Any, Path]:
    code_argument = Path(deployed_code)
    _require(
        code_argument.is_dir() and not code_argument.is_symlink(),
        "deployed v5 code must be an existing non-symlink directory",
    )
    code = code_argument.resolve()
    source = code / "src"
    _require(source.is_dir() and not source.is_symlink(), "deployed src is invalid")
    preloaded = sorted(
        name
        for name in sys.modules
        if name == "bayesian_phystwin" or name.startswith("bayesian_phystwin.")
    )
    _require(
        not preloaded, f"bayesian_phystwin was preloaded before audit: {preloaded}"
    )
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    from bayesian_phystwin import deform360_held_protocol as protocol  # noqa: PLC0415

    expected_file = source / "bayesian_phystwin" / "deform360_held_protocol.py"
    _require(
        Path(protocol.__file__).resolve() == expected_file.resolve(),
        "held protocol imported from outside the deployed v5 snapshot",
    )
    _require_package_provenance(source)
    return protocol, source


def _load_post_barrier_api(source: Path) -> PostBarrierApi:
    from bayesian_phystwin import (  # noqa: PLC0415
        deform360_held_outcome_reconstruction as reconstruction,
    )
    from bayesian_phystwin import (  # noqa: PLC0415
        deform360_held_outcome_scoring as scoring,
    )

    expected = {
        reconstruction: (
            source / "bayesian_phystwin" / "deform360_held_outcome_reconstruction.py"
        ),
        scoring: source / "bayesian_phystwin" / "deform360_held_outcome_scoring.py",
    }
    for module, path in expected.items():
        _require(
            Path(module.__file__).resolve() == path.resolve(),
            f"post-barrier module imported outside deployed snapshot: {module.__name__}",
        )
    _require_package_provenance(source)
    return PostBarrierApi(
        backend_type=reconstruction.PinnedOfficialPipelineBackend,
        plan_target_operation=(
            reconstruction.plan_official_reconstruction_target_operation
        ),
        target_operation_type=scoring.TargetOperation,
        score_and_create_gate=scoring.score_and_create_confirmation_gate,
    )


def _normalized_environment_or_reexec() -> None:
    if (
        os.environ.get("BPT_HELD_CONFIRMATION_OUTCOME_ENV_NORMALIZED") == "1"
        and sys.flags.isolated == 1
    ):
        expected = _minimal_environment(include_outcome_runtime=True)
        expected["BPT_HELD_CONFIRMATION_OUTCOME_ENV_NORMALIZED"] = "1"
        _require(
            dict(os.environ) == expected,
            "normalized outcome environment contains an unapproved variable",
        )
        _require(
            sys.flags.dont_write_bytecode == 1
            and sys.pycache_prefix == _PYCACHE_PREFIX,
            "normalized outcome process may consult adjacent pyc",
        )
        return
    _require(
        os.environ.get("BPT_HELD_CONFIRMATION_OUTCOME_ENV_NORMALIZED") != "1",
        "failed to enter the normalized outcome runtime environment",
    )
    environment = _minimal_environment(include_outcome_runtime=True)
    environment["BPT_HELD_CONFIRMATION_OUTCOME_ENV_NORMALIZED"] = "1"
    os.execve(
        _PINNED_PYTHON,
        [
            _PINNED_PYTHON,
            "-B",
            "-X",
            f"pycache_prefix={_PYCACHE_PREFIX}",
            "-I",
            str(Path(__file__).resolve()),
            *sys.argv[1:],
        ],
        environment,
    )


def _driver_arguments(namespace: argparse.Namespace) -> DriverArguments:
    return DriverArguments(
        deployed_code=namespace.deployed_code,
        held_root=namespace.held_root,
        calibration_lock=namespace.calibration_lock,
        calibration_decision=namespace.calibration_decision,
        confirmation_lock=namespace.confirmation_lock,
        online_seals=tuple(namespace.online_seal),
        promote_only=namespace.promote_only,
        dry_run_barrier_only=namespace.dry_run_barrier_only,
        aligned_root=namespace.aligned_root,
        deform360_repo=namespace.deform360_repo,
        sam2_repository=namespace.sam2_repository,
        sam2_checkpoint=namespace.sam2_checkpoint,
        cotracker_repo=namespace.cotracker_repo,
        cotracker_checkpoint=namespace.cotracker_checkpoint,
        device=namespace.device,
        ffmpeg=namespace.ffmpeg,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Permit-gated official reconstruction and frozen scoring of the "
            "exact held-v5 six-case confirmation cohort."
        )
    )
    parser.add_argument("--self-check", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--promote-only", action="store_true")
    mode.add_argument("--dry-run-barrier-only", action="store_true")
    parser.add_argument("--deployed-code")
    parser.add_argument("--held-root")
    parser.add_argument("--calibration-lock")
    parser.add_argument("--calibration-decision")
    parser.add_argument("--confirmation-lock")
    parser.add_argument(
        "--online-seal",
        action="append",
        default=[],
        metavar="CASE=PATH",
        help="repeat exactly once for each of the six locked confirmation cases",
    )
    parser.add_argument("--aligned-root")
    parser.add_argument("--deform360-repo")
    parser.add_argument("--sam2-repository")
    parser.add_argument("--sam2-checkpoint")
    parser.add_argument("--cotracker-repo")
    parser.add_argument("--cotracker-checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    namespace = parser.parse_args()
    if not namespace.self_check:
        missing = [
            name
            for name in (
                "deployed_code",
                "held_root",
                "calibration_lock",
                "calibration_decision",
                "confirmation_lock",
            )
            if not getattr(namespace, name)
        ]
        parser.error(
            f"missing required normal-run arguments: {missing}"
        ) if missing else None
    return namespace


def _mock_arguments(root: Path, *, dry: bool) -> DriverArguments:
    cases = tuple(f"mock-object-{index:02d}-ep{index:04d}" for index in range(6))
    seals: list[str] = []
    for case_name in cases:
        seal = (
            root
            / "confirmation"
            / "cases"
            / case_name
            / "online"
            / "online_prediction_seal.json"
        )
        seal.parent.mkdir(parents=True, exist_ok=True)
        seal.write_text("mock seal\n", encoding="utf-8")
        seals.append(f"{case_name}={seal}")
    calibration_lock = root / "calibration-lock.json"
    calibration_lock.write_text("mock calibration lock\n", encoding="utf-8")
    calibration_decision = root / "confirmation" / "calibration-decision.json"
    calibration_decision.parent.mkdir(parents=True, exist_ok=True)
    calibration_decision.write_text("mock GO\n", encoding="utf-8")
    confirmation_lock = root / "confirmation-lock.json"
    confirmation_lock.write_text("mock confirmation lock\n", encoding="utf-8")
    aligned = root / "mock-aligned"
    for case_name in cases:
        object_id, _separator, episode = case_name.rpartition("-ep")
        (aligned / object_id / f"episode_{episode}").mkdir(parents=True)
    return DriverArguments(
        deployed_code="unused-in-injected-self-check",
        held_root=str(root),
        calibration_lock=str(calibration_lock),
        calibration_decision=str(calibration_decision),
        confirmation_lock=str(confirmation_lock),
        online_seals=tuple(seals),
        promote_only=False,
        dry_run_barrier_only=dry,
        aligned_root=str(aligned),
        deform360_repo="mock-deform360",
        sam2_repository="mock-sam2",
        sam2_checkpoint="mock-sam2-checkpoint",
        cotracker_repo="mock-cotracker",
        cotracker_checkpoint="mock-cotracker-checkpoint",
    )


def _self_check() -> int:
    """Mock the barrier-first pass and failed-confirmation paths."""

    @dataclass(frozen=True)
    class FakeTargetOperation:
        operation: str
        callback: Callable[[], Any]

    class FakeBackend:
        def __init__(self, **_kwargs: Any) -> None:
            events.append("backend")

    class FakeProtocol:
        PROTOCOL_ID = EXPECTED_PROTOCOL_ID

        def __init__(self, cases: tuple[str, ...]) -> None:
            self.CONFIRMATION_CASE_NAMES = cases

        @staticmethod
        def authorize_outcome_phase(
            _lock: str, seals: Mapping[str, str], *, role: str
        ) -> Any:
            events.append("authorize")
            _require(role == "confirmation" and len(seals) == 6, "bad mock permit")
            return SimpleNamespace(cohort_barrier_sha256="b" * 64)

    def fake_plan(_permit: Any, **kwargs: Any) -> FakeTargetOperation:
        case_name = kwargs["case_name"]
        events.append(f"plan:{case_name}")
        return FakeTargetOperation(
            operation="create",
            callback=lambda case_name=case_name: {"case_name": case_name},
        )

    def post_api(decision: str) -> PostBarrierApi:
        def fake_score(
            decision_path: Path,
            _permit: Any,
            operations: Mapping[str, FakeTargetOperation],
            *,
            evidence_path: Path,
        ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
            events.append("score")
            for operation in operations.values():
                operation.callback()
            evidence_path.write_text("mock immutable evidence\n", encoding="utf-8")
            decision_path.write_text("mock immutable decision\n", encoding="utf-8")
            records = {case: {} for case in operations}
            return (
                {
                    "decision": decision,
                    "artifact_sha256": "d" * 64,
                    "summary": {
                        "passed": decision == "CONFIRMED",
                        "one_sided_sign_test_p": 1.0 / 64.0,
                    },
                },
                {"artifact_sha256": "e" * 64},
                records,
            )

        return PostBarrierApi(
            backend_type=FakeBackend,
            plan_target_operation=fake_plan,
            target_operation_type=FakeTargetOperation,
            score_and_create_gate=fake_score,
        )

    with tempfile.TemporaryDirectory(prefix="bpt-outcome-driver-self-check-") as temp:
        base = Path(temp)
        cases = tuple(f"mock-object-{index:02d}-ep{index:04d}" for index in range(6))

        events: list[str] = []
        dry_args = _mock_arguments(base / "dry", dry=True)

        def forbidden_post_import() -> PostBarrierApi:
            raise AssertionError("dry-run imported a post-barrier API")

        dry_code = _execute_driver(
            dry_args,
            FakeProtocol(cases),
            forbidden_post_import,
            validate_prebarrier_confirmation=lambda _arguments, _protocol: None,
        )
        _require(dry_code == 0 and events == ["authorize"], "dry barrier order failed")

        events = []
        confirmed_args = _mock_arguments(base / "confirmed", dry=False)
        confirmed_code = _execute_driver(
            confirmed_args,
            FakeProtocol(cases),
            lambda: post_api("CONFIRMED"),
            lambda _arguments: None,
            lambda _arguments, _protocol: None,
        )
        _require(
            confirmed_code == 0 and events[0] == "authorize",
            "confirmation pass was not barrier-first",
        )
        _require(
            events.count("score") == 1 and events[-1] == "score",
            "confirmation-pass orchestration continued after scoring",
        )

        events = []
        failed_args = _mock_arguments(base / "not-confirmed", dry=False)
        failed_code = _execute_driver(
            failed_args,
            FakeProtocol(cases),
            lambda: post_api("NOT_CONFIRMED"),
            lambda _arguments: None,
            lambda _arguments, _protocol: None,
        )
        _require(
            failed_code == NOT_CONFIRMED_EXIT_CODE and events[0] == "authorize",
            "failed confirmation was not barrier-first and non-zero",
        )
        _require(
            events.count("score") == 1 and events[-1] == "score",
            "failed confirmation orchestration did not stop at the gate",
        )
    _emit(
        "SELF_CHECK_PASSED",
        checks=[
            "dry-run authorizes barrier and imports no post-barrier API",
            "full runs authorize before planning any target operation",
            "exactly six TargetOperations are planned in locked order",
            "failed confirmation exits non-zero after immutable gate writing",
            "fresh outcome roots forbid confirmation resume reads",
        ],
    )
    return 0


def main() -> int:
    namespace = _parse_args()
    if namespace.self_check:
        return _self_check()
    _normalized_environment_or_reexec()
    arguments = _driver_arguments(namespace)
    _verify_locked_deployment(arguments)
    protocol, source = _load_protocol(arguments.deployed_code)
    if arguments.promote_only:
        return _promote_confirmation_lock(arguments, protocol)
    return _execute_driver(
        arguments,
        protocol,
        lambda: _load_post_barrier_api(source),
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as error:
        _emit(
            "FAIL_CLOSED",
            error_type=type(error).__name__,
            message=str(error),
            confirmation_payload_read_status="NOT_CLAIMED",
        )
        raise SystemExit(2) from error
