#!/usr/bin/env python3
"""Seal the held-v8.1 attempt-4 technical outcome failure.

This one-purpose forensic operator does not deserialize a target, query,
prediction, point cloud, mask, image, video, or score payload.  It validates
the already-observed execution boundary using names and metadata, hashes every
non-code byte through stable ``O_NOFOLLOW`` reads, binds the durable launcher
log, atomically renames the formal root, seals both evidence trees, verifies
the renamed archive, and writes immutable completion and pointer artifacts.

The operation is restart-idempotent.  It never overwrites an existing report,
completion, or pointer and can finish after an interruption following the
atomic rename.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
from typing import Any


BASE = Path("/mnt/corsair/florianpfaff/bpt-online-belief-v1")
ACTIVE = BASE / "held-v8"
ARCHIVE = BASE / "held-v8-attempt-4-withdrawn-postbarrier"
REPORT_NAME = "execution-withdrawal-postbarrier-attempt4.json"
POINTER = BASE / "held-v8-attempt-4-withdrawal-pointer.json"
COMPLETION = BASE / "held-v8-attempt-4-withdrawal-integrity-completion.json"
LAUNCHER = Path(
    "/mnt/corsair/florianpfaff/bpt-held-v81-orchestration/"
    "calibration-outcome-c88168c-20260722T1847"
)
OPERATOR_SOURCE = Path(__file__).resolve()

PROTOCOL_ID = "deform360-held-online-belief-v8.1"
EXECUTION_ATTEMPT = 4
STATUS = (
    "withdrawn-postbarrier-during-third-target-reconstruction-before-barrier2-or-score"
)
DISPOSITION = (
    "WITHDRAWN_AFTER_TWO_TARGET_X0_QUERY_PAIRS_DURING_THIRD_TARGET_"
    "RECONSTRUCTION_BEFORE_SECOND_BARRIER_OR_SCORE"
)

EXPECTED_DEPLOYED_HEAD = "c88168cd88be37aa403929c5323da7a29eafa20a"
EXPECTED_DEPLOYED_CODE_NAME = f"code-{EXPECTED_DEPLOYED_HEAD}"
EXPECTED_DEPLOYED_TREE_SHA256 = (
    "e1baaa61aca75f7e3a8d9f51d5fd47feca113a761071f86e3ec6c96d15243cc4"
)
EXPECTED_DEPLOYED_TREE_RECORD_COUNT = 954
EXPECTED_LOCK_FILE_SHA256 = (
    "80fb448f2aa8aa654624b70a5b07ebf8a59eacba62b81dd13079715d5bbf5037"
)
EXPECTED_LOCK_ARTIFACT_SHA256 = (
    "da8ff292b80e64b1d235af3adcc11c5fd31e0b5109827b6cc7cc68113c954437"
)
EXPECTED_LAUNCHER_LOG_SIZE = 1_168_519_909
EXPECTED_LAUNCHER_LOG_SHA256 = (
    "9153b50771d8818384d96a77f3502dbbc9494136f679fd25aa6e8208f73bd3e8"
)
EXPECTED_EXIT_SIZE = 2
EXPECTED_EXIT_SHA256 = (
    "53c234e5e8472b6ac51c1ae1cab3fe06fad053beb8ebfd8977b010655bfdd3c3"
)
EXPECTED_EXIT_BYTES = b"2\n"

EXPECTED_CASES = (
    "072-cotton-clohesline-ep0003",
    "002-rope-silk-ep0004",
    "002-rope-silk-ep0008",
    "083-blanket-cloth-ep0000",
    "083-blanket-cloth-ep0003",
    "083-blanket-cloth-ep0006",
    "085-scarf-cloth-ep0000",
    "085-scarf-cloth-ep0005",
    "085-scarf-cloth-ep0007",
    "092-squirrel-ep0002",
    "092-squirrel-ep0003",
    "092-squirrel-ep0006",
    "170-spider-ep0002",
    "170-spider-ep0004",
    "170-spider-ep0007",
)
COMPLETED_OUTCOME_CASES = (
    "072-cotton-clohesline-ep0003",
    "002-rope-silk-ep0004",
)
COMPLETED_CAMERAS: Mapping[str, tuple[str, ...]] = {
    "072-cotton-clohesline-ep0003": (
        "brics-odroid-001_cam0",
        "brics-odroid-008_cam0",
        "brics-odroid-012_cam0",
        "brics-odroid-014_cam1",
        "brics-odroid-016_cam0",
        "brics-odroid-021_cam0",
        "brics-odroid-022_cam0",
        "brics-odroid-022_cam1",
    ),
    "002-rope-silk-ep0004": (
        "brics-odroid-001_cam0",
        "brics-odroid-006_cam0",
        "brics-odroid-007_cam0",
        "brics-odroid-010_cam0",
        "brics-odroid-013_cam0",
        "brics-odroid-014_cam1",
        "brics-odroid-015_cam1",
        "brics-odroid-019_cam1",
        "brics-odroid-021_cam1",
        "brics-odroid-025_cam1",
        "brics-odroid-027_cam0",
        "brics-odroid-028_cam0",
    ),
}
COMPLETED_PCD_FRAME_COUNT = 76
COMPLETED_SPLAT_FRAME_COUNT = 81
COMPLETED_STAGING_EXPECTATIONS: Mapping[str, Mapping[str, int]] = {
    "072-cotton-clohesline-ep0003": {
        "directory_count": 20,
        "regular_file_count": 235,
        "regular_file_bytes": 230_280_156,
    },
    "002-rope-silk-ep0004": {
        "directory_count": 28,
        "regular_file_count": 271,
        "regular_file_bytes": 352_494_692,
    },
}
FAILED_CASE = "002-rope-silk-ep0008"
FAILED_CAMERAS = (
    "brics-odroid-001_cam0",
    "brics-odroid-007_cam0",
    "brics-odroid-013_cam0",
    "brics-odroid-014_cam1",
    "brics-odroid-019_cam1",
    "brics-odroid-021_cam1",
    "brics-odroid-027_cam0",
    "brics-odroid-028_cam0",
)
FAILED_SCRATCH_TIMESTAMP = "2026-07-22_192624"

LOG_MARKERS: Mapping[str, tuple[bytes, int]] = {
    "first_cohort_barrier_validated": (
        b'"event": "FIRST_COHORT_BARRIER_VALIDATED"',
        1,
    ),
    "official_target_and_x0_sealed": (
        b'"event": "OFFICIAL_TARGET_AND_X0_SEALED"',
        2,
    ),
    "isolated_x0_query_sealed": (
        b'"event": "ISOLATED_X0_QUERY_SEALED"',
        2,
    ),
    "second_cohort_barrier_validated": (
        b'"event": "SECOND_COHORT_BARRIER_VALIDATED"',
        0,
    ),
    "fail_closed": (b'"event": "FAIL_CLOSED"', 1),
    "terminal_error_type": (b'"error_type": "OSError"', 1),
    "too_many_open_files": (b"Too many open files", 1),
}

_HEAD_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_FORMAL_PROCESS_MARKERS = (
    "run_deform360_v8_calibration_shard.sh",
    "run_deform360_v8_calibration_case.sh",
    "run_deform360_v8_calibration_outcomes.py",
    "run_deform360_v8_confirmation_shard.sh",
    "run_deform360_v8_confirmation_case.sh",
    "run_deform360_v8_confirmation_outcomes.py",
    "run_deform360_v8_replacement_source.py",
    "run_deform360_v8_x0_query.py",
    "deform360_held_v8_frame_zero_assets",
    "deform360_held_v8_physical_prior",
    "deform360_held_v8_online",
    "deform360_held_v8_outcome_driver",
)


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


def _artifact_sha256(value: Mapping[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("artifact_sha256", None)
    return hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()


def _signed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["artifact_sha256"] = _artifact_sha256(result)
    return result


def _pretty_json_bytes(value: Mapping[str, Any]) -> bytes:
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


def _stable_identity(value: os.stat_result) -> tuple[int, ...]:
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


def _stable_file(
    path: Path,
    *,
    collect: bool = False,
    markers: Mapping[str, bytes] | None = None,
    allowed_link_counts: frozenset[int] = frozenset({1}),
) -> dict[str, Any]:
    """Hash one regular file with an exact allowed link count."""

    before = os.lstat(path)
    _require(
        stat.S_ISREG(before.st_mode)
        and not stat.S_ISLNK(before.st_mode)
        and before.st_nlink in allowed_link_counts,
        (
            f"regular single-link file required: {path}"
            if allowed_link_counts == frozenset({1})
            else f"regular file with allowed link count required: {path}"
        ),
    )
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    digest = hashlib.sha256()
    payload = bytearray() if collect else None
    marker_values = {} if markers is None else dict(markers)
    marker_counts = {name: 0 for name in marker_values}
    overlap = b""
    maximum_marker = max((len(value) for value in marker_values.values()), default=1)
    try:
        opened = os.fstat(descriptor)
        _require(
            _stable_identity(opened) == _stable_identity(before),
            f"file changed while opening: {path}",
        )
        while block := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(block)
            if payload is not None:
                payload.extend(block)
            if marker_values:
                joined = overlap + block
                prefix = len(overlap)
                for name, marker in marker_values.items():
                    offset = 0
                    while True:
                        found = joined.find(marker, offset)
                        if found < 0:
                            break
                        if found + len(marker) > prefix:
                            marker_counts[name] += 1
                        offset = found + 1
                overlap = joined[-(maximum_marker - 1) :] if maximum_marker > 1 else b""
        after_open = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = os.lstat(path)
    _require(
        _stable_identity(before)
        == _stable_identity(after_open)
        == _stable_identity(after),
        f"file changed while hashing: {path}",
    )
    return {
        "size_bytes": before.st_size,
        "sha256": digest.hexdigest(),
        "mode_octal": f"{stat.S_IMODE(before.st_mode):04o}",
        **({"payload": bytes(payload)} if payload is not None else {}),
        **({"marker_counts": marker_counts} if marker_values else {}),
    }


def _read_metadata_json(path: Path, *, role: str) -> dict[str, Any]:
    """Read operator metadata; callers never pass a protected payload."""

    observed = _stable_file(path, collect=True)
    try:
        value = json.loads(observed["payload"].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{role} is not JSON metadata") from error
    _require(isinstance(value, dict), f"{role} is not a JSON object")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _pending_json_path(path: Path) -> Path:
    return BASE / f".{path.name}.attempt4-withdrawal.pending"


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> tuple[int, str]:
    """Materialize canonical JSON once, with restart-safe pending recovery.

    The final name is created exclusively with a hard link from a fully synced
    staging file placed directly under ``BASE``, outside ``ACTIVE`` but on the
    same filesystem.  A restart can finish either the one-link pending state or
    the two-link post-publication state.  A partial pending file is
    operator-owned scratch and is removed before a fresh write; an unexpected
    final artifact is never replaced.
    """

    payload = _pretty_json_bytes(value)
    pending = _pending_json_path(path)
    _require(
        BASE.is_dir()
        and not BASE.is_symlink()
        and BASE.resolve() == BASE
        and pending.parent == BASE
        and ACTIVE not in pending.parents
        and os.stat(BASE).st_dev == os.stat(path.parent).st_dev,
        "JSON staging must be a same-filesystem direct child of BASE outside ACTIVE",
    )
    target_exists = os.path.lexists(path)
    pending_exists = os.path.lexists(pending)

    if target_exists and pending_exists:
        target_state = os.lstat(path)
        pending_state = os.lstat(pending)
        _require(
            stat.S_ISREG(target_state.st_mode)
            and stat.S_ISREG(pending_state.st_mode)
            and not stat.S_ISLNK(target_state.st_mode)
            and not stat.S_ISLNK(pending_state.st_mode)
            and target_state.st_dev == pending_state.st_dev
            and target_state.st_ino == pending_state.st_ino
            and target_state.st_nlink == pending_state.st_nlink == 2,
            f"unexpected final/pending JSON pair: {path}",
        )
        linked = _stable_file(
            pending,
            collect=True,
            allowed_link_counts=frozenset({2}),
        )
        _require(
            linked["mode_octal"] == "0400" and linked["payload"] == payload,
            f"linked pending JSON changed: {pending}",
        )
        _fsync_directory(path.parent)
        os.unlink(pending)
        _fsync_directory(pending.parent)
        pending_exists = False

    if target_exists:
        _require(not pending_exists, f"unexpected pending JSON exists: {pending}")
        observed = _stable_file(path, collect=True)
        _require(
            observed["mode_octal"] == "0400" and observed["payload"] == payload,
            f"existing final JSON changed: {path}",
        )
        _fsync_directory(path.parent)
        if pending.parent != path.parent:
            _fsync_directory(pending.parent)
        return int(observed["size_bytes"]), str(observed["sha256"])

    if pending_exists:
        pending_observed = _stable_file(pending, collect=True)
        if not (
            pending_observed["mode_octal"] == "0400"
            and pending_observed["payload"] == payload
        ):
            os.unlink(pending)
            _fsync_directory(pending.parent)
            pending_exists = False

    if not pending_exists:
        descriptor = os.open(
            pending,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o400,
        )
        try:
            offset = 0
            while offset < len(payload):
                offset += os.write(descriptor, payload[offset:])
            os.fchmod(descriptor, 0o400)
            os.fsync(descriptor)
        except BaseException:
            os.close(descriptor)
            pending.unlink(missing_ok=True)
            _fsync_directory(pending.parent)
            raise
        else:
            os.close(descriptor)
        _fsync_directory(pending.parent)

    pending_observed = _stable_file(pending, collect=True)
    _require(
        pending_observed["mode_octal"] == "0400"
        and pending_observed["payload"] == payload,
        f"pending JSON changed: {pending}",
    )
    os.link(pending, path, follow_symlinks=False)
    linked_target = os.lstat(path)
    linked_pending = os.lstat(pending)
    _require(
        linked_target.st_dev == linked_pending.st_dev
        and linked_target.st_ino == linked_pending.st_ino
        and linked_target.st_nlink == linked_pending.st_nlink == 2,
        f"final JSON publication link changed: {path}",
    )
    _fsync_directory(path.parent)
    os.unlink(pending)
    _fsync_directory(pending.parent)
    observed = _stable_file(path)
    _require(
        observed["mode_octal"] == "0400" and observed["size_bytes"] == len(payload),
        f"sealed JSON changed: {path}",
    )
    return int(observed["size_bytes"]), str(observed["sha256"])


def _validate_signed_metadata(path: Path, *, role: str) -> dict[str, Any]:
    observed = _stable_file(path)
    _require(observed["mode_octal"] == "0400", f"{role} is not mode 0400")
    value = _read_metadata_json(path, role=role)
    _require(
        value.get("artifact_sha256") == _artifact_sha256(value),
        f"{role} artifact hash changed",
    )
    return value


def _operator_source_binding() -> dict[str, Any]:
    source = OPERATOR_SOURCE
    _require(
        source.is_absolute()
        and source.resolve() == source
        and source.name == "seal_deform360_v8_attempt4_outcome_failure.py",
        "executed attempt-4 withdrawal operator path changed",
    )
    observed = _stable_file(source)
    return {"path": os.fspath(source), **observed}


def _process_ancestry() -> set[int]:
    result: set[int] = set()
    pid = os.getpid()
    while pid > 1 and pid not in result:
        result.add(pid)
        try:
            fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
            pid = int(fields[3])
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            break
    return result


def _running_formal_processes() -> list[dict[str, Any]]:
    own = _process_ancestry()
    records: list[dict[str, Any]] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return records
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) in own:
            continue
        try:
            argv = [
                part.decode(errors="replace")
                for part in (entry / "cmdline").read_bytes().split(b"\0")
                if part
            ]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        basenames = {Path(value).name for value in argv}
        joined = " ".join(argv)
        if any(
            marker in basenames or marker in joined
            for marker in _FORMAL_PROCESS_MARKERS
        ):
            records.append({"pid": int(entry.name), "argv": argv})
    return sorted(records, key=lambda row: int(row["pid"]))


def _run_git(root: Path, arguments: Sequence[str]) -> bytes:
    completed = subprocess.run(
        [
            "git",
            "-c",
            "core.fileMode=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-C",
            str(root),
            *arguments,
        ],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_OPTIONAL_LOCKS": "0",
            "HOME": "/home/florianpfaff",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/local/bin:/usr/bin:/bin",
        },
    )
    _require(
        completed.returncode == 0,
        "git "
        + " ".join(arguments)
        + " failed: "
        + completed.stderr.decode(errors="replace").strip(),
    )
    return completed.stdout


def _parse_git_tree(raw: bytes) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
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
            and _HEAD_RE.fullmatch(object_id) is not None
            and path
            and not path.startswith("/")
            and ".." not in Path(path).parts,
            f"unsafe or unsupported deployed-code entry: {path}",
        )
        rows.append({"mode": mode, "type": kind, "object_id": object_id, "path": path})
    _require(
        bool(rows)
        and [row["path"] for row in rows] == sorted(row["path"] for row in rows),
        "deployed Git tree is empty or unsorted",
    )
    return rows


def _git_blob_oid(path: Path, *, object_id: str) -> str:
    before = os.lstat(path)
    _require(
        stat.S_ISREG(before.st_mode)
        and not stat.S_ISLNK(before.st_mode)
        and before.st_nlink == 1,
        f"deployed tracked file is linked or special: {path}",
    )
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    algorithm = "sha1" if len(object_id) == 40 else "sha256"
    digest = hashlib.new(algorithm)
    digest.update(f"blob {before.st_size}\0".encode("ascii"))
    try:
        opened = os.fstat(descriptor)
        _require(
            _stable_identity(opened) == _stable_identity(before),
            f"deployed tracked file changed while opening: {path}",
        )
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
        after_open = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = os.lstat(path)
    _require(
        _stable_identity(before)
        == _stable_identity(after_open)
        == _stable_identity(after),
        f"deployed tracked file changed while hashing: {path}",
    )
    return digest.hexdigest()


def _repository_binding(code: Path) -> dict[str, Any]:
    state = os.lstat(code)
    _require(
        stat.S_ISDIR(state.st_mode)
        and not stat.S_ISLNK(state.st_mode)
        and code.resolve() == code
        and (code / ".git").is_dir(),
        "deployed code is not a canonical Git directory",
    )
    top = _run_git(code, ["rev-parse", "--show-toplevel"]).decode().strip()
    head = _run_git(code, ["rev-parse", "HEAD"]).decode().strip().lower()
    _require(
        top == str(code) and _HEAD_RE.fullmatch(head) is not None,
        "bad deployed Git identity",
    )
    _require(
        _run_git(code, ["status", "--porcelain=v1", "--untracked-files=all"]) == b"",
        "deployed worktree content changed",
    )
    _require(
        _run_git(code, ["ls-files", "--others", "--exclude-standard", "-z"]) == b""
        and _run_git(
            code,
            ["ls-files", "--others", "--ignored", "--exclude-standard", "-z"],
        )
        == b"",
        "deployed worktree has ordinary or ignored untracked files",
    )
    _require(
        _run_git(code, ["rev-parse", "--is-shallow-repository"]).decode().strip()
        == "false",
        "deployed repository is shallow",
    )
    _run_git(code, ["fsck", "--full", "--no-dangling"])
    rows = _parse_git_tree(_run_git(code, ["ls-tree", "-r", "-z", "HEAD"]))
    tracked = {row["path"] for row in rows}
    working: set[str] = set()
    for current, directories, files in os.walk(code, topdown=True, followlinks=False):
        parent = Path(current)
        relative = parent.relative_to(code)
        if relative == Path("."):
            directories[:] = sorted(name for name in directories if name != ".git")
        else:
            directories[:] = sorted(directories)
        for name in directories:
            child = parent / name
            child_state = os.lstat(child)
            _require(
                stat.S_ISDIR(child_state.st_mode)
                and not stat.S_ISLNK(child_state.st_mode),
                f"deployed worktree directory is linked or special: {child}",
            )
        for name in sorted(files):
            child = parent / name
            relative_child = child.relative_to(code).as_posix()
            working.add(relative_child)
    _require(working == tracked, "deployed working tree has missing or extra files")
    for row in rows:
        _require(
            _git_blob_oid(code / row["path"], object_id=row["object_id"])
            == row["object_id"],
            f"deployed tracked file differs from Git: {row['path']}",
        )
    tree_sha256 = hashlib.sha256(_canonical_bytes(rows)).hexdigest()
    _require(
        head == EXPECTED_DEPLOYED_HEAD
        and code.name == EXPECTED_DEPLOYED_CODE_NAME
        and len(rows) == EXPECTED_DEPLOYED_TREE_RECORD_COUNT
        and tree_sha256 == EXPECTED_DEPLOYED_TREE_SHA256,
        "deployed attempt-4 HEAD or tree changed",
    )
    return {
        "path": code.name,
        "git_head": head,
        "head_text_sha256": hashlib.sha256(head.encode("ascii")).hexdigest(),
        "git_tree_record_count": len(rows),
        "git_tree_manifest_sha256": tree_sha256,
        "every_working_file_matches_bound_git_blob": True,
        "no_ordinary_or_ignored_untracked_files": True,
    }


def _code_directory(root: Path) -> Path:
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir() and not path.is_symlink() and path.name.startswith("code-")
    ]
    _require(
        len(candidates) == 1 and candidates[0].name == EXPECTED_DEPLOYED_CODE_NAME,
        "expected exact attempt-4 deployed-code directory",
    )
    return candidates[0]


def _validate_lock(root: Path, code: Mapping[str, Any]) -> dict[str, Any]:
    path = root / "calibration-lock.json"
    observed = _stable_file(path)
    _require(
        observed["mode_octal"] == "0400"
        and observed["sha256"] == EXPECTED_LOCK_FILE_SHA256,
        "attempt-4 calibration lock changed",
    )
    lock = _read_metadata_json(path, role="attempt-4 calibration lock")
    _require(
        lock.get("artifact_sha256") == EXPECTED_LOCK_ARTIFACT_SHA256
        and _artifact_sha256(lock) == EXPECTED_LOCK_ARTIFACT_SHA256,
        "attempt-4 lock artifact hash changed",
    )
    _require(
        lock.get("protocol_id") == PROTOCOL_ID
        and lock.get("execution_attempt") == EXECUTION_ATTEMPT
        and lock.get("stage") == "calibration"
        and lock.get("held_root") == os.fspath(ACTIVE)
        and tuple(lock.get("calibration_case_whitelist", ())) == EXPECTED_CASES,
        "attempt-4 lock identity or cohort changed",
    )
    bindings = lock.get("immutable_bindings")
    _require(isinstance(bindings, Mapping), "attempt-4 lock bindings are absent")
    _require(
        bindings.get("method_deployed_snapshot_tree")
        == code["git_tree_manifest_sha256"]
        == EXPECTED_DEPLOYED_TREE_SHA256
        and bindings.get("method_head_text_sha256") == code["head_text_sha256"],
        "attempt-4 deployed code differs from its lock",
    )
    return {
        "path": "calibration-lock.json",
        **observed,
        "artifact_sha256": EXPECTED_LOCK_ARTIFACT_SHA256,
    }


def _inventory(root: Path, *, code_name: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        parent = Path(current)
        relative_parent = parent.relative_to(root)
        directories[:] = sorted(
            name
            for name in directories
            if not (relative_parent == Path(".") and name == code_name)
        )
        for name in directories:
            path = parent / name
            observed = os.lstat(path)
            _require(
                stat.S_ISDIR(observed.st_mode) and not stat.S_ISLNK(observed.st_mode),
                f"directory symlink or special entry refused: {path}",
            )
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "type": "directory",
                    "mode_octal": f"{stat.S_IMODE(observed.st_mode):04o}",
                }
            )
        for name in sorted(files):
            path = parent / name
            relative = path.relative_to(root)
            if relative == Path(REPORT_NAME):
                continue
            observed = _stable_file(path)
            rows.append(
                {
                    "path": relative.as_posix(),
                    "type": "file",
                    **observed,
                }
            )
    rows.sort(key=lambda row: str(row["path"]))
    _require(
        len(rows) == len({str(row["path"]) for row in rows}),
        "duplicate inventory path",
    )
    return {
        "rows": rows,
        "entry_count": len(rows),
        "directory_count": sum(row["type"] == "directory" for row in rows),
        "regular_file_count": sum(row["type"] == "file" for row in rows),
        "regular_file_bytes": sum(
            int(row.get("size_bytes", 0)) for row in rows if row["type"] == "file"
        ),
        "inventory_sha256": hashlib.sha256(
            _canonical_bytes({"rows": rows})
        ).hexdigest(),
        "excluded_deployed_code_directory": code_name,
        "excluded_withdrawal_report": REPORT_NAME,
    }


def _content_projection(inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in row.items() if key != "mode_octal"}
        for row in inventory["rows"]
    ]


def _sealed_inventory(inventory: Mapping[str, Any]) -> dict[str, Any]:
    rows = [
        {
            **row,
            "mode_octal": "0500" if row["type"] == "directory" else "0400",
        }
        for row in inventory["rows"]
    ]
    return {
        **{key: value for key, value in inventory.items() if key != "rows"},
        "rows": rows,
        "inventory_sha256": hashlib.sha256(
            _canonical_bytes({"rows": rows})
        ).hexdigest(),
    }


def _require_directory(path: Path, *, mode: int | None, role: str) -> None:
    observed = os.lstat(path)
    _require(
        stat.S_ISDIR(observed.st_mode)
        and not stat.S_ISLNK(observed.st_mode)
        and (mode is None or stat.S_IMODE(observed.st_mode) == mode),
        f"{role} is absent, linked, special, or has the wrong mode: {path}",
    )


def _require_regular_mode(path: Path, *, mode: int, role: str) -> None:
    observed = _stable_file(path)
    _require(observed["mode_octal"] == f"{mode:04o}", f"{role} is not mode {mode:04o}")


def _require_regular_entry(path: Path, *, mode: int, role: str) -> None:
    observed = os.lstat(path)
    _require(
        stat.S_ISREG(observed.st_mode)
        and not stat.S_ISLNK(observed.st_mode)
        and observed.st_nlink == 1
        and stat.S_IMODE(observed.st_mode) == mode,
        f"{role} is absent, linked, special, or has the wrong mode: {path}",
    )


def _validate_completed_reconstruction(
    root: Path,
    *,
    case: str,
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    reconstruction = (
        root / "calibration/private-targets" / case / "fresh-official-reconstruction"
    )
    cameras = COMPLETED_CAMERAS.get(case)
    _require(cameras is not None and bool(cameras), f"completed cameras absent: {case}")
    _require_directory(
        reconstruction, mode=0o500, role=f"completed reconstruction {case}"
    )
    _require(
        {entry.name for entry in reconstruction.iterdir()}
        == {"held-v8-official-reconstruction-audit.json", "staged-aligned"},
        f"completed reconstruction inventory changed: {case}",
    )
    _require_regular_entry(
        reconstruction / "held-v8-official-reconstruction-audit.json",
        mode=0o400,
        role=f"completed reconstruction audit {case}",
    )
    staged = reconstruction / "staged-aligned"
    episode = staged / "episode_0000"
    _require_directory(staged, mode=0o500, role=f"completed staged root {case}")
    _require(
        {entry.name for entry in staged.iterdir()} == {"episode_0000"},
        f"completed staged inventory changed: {case}",
    )
    _require_directory(episode, mode=0o500, role=f"completed episode {case}")
    _require(
        {entry.name for entry in episode.iterdir()}
        == {
            *cameras,
            "extrinsics.npy",
            "undistorted_intrinsics.npy",
            "pcd_clean",
            "robot",
            "splatfacto",
        },
        f"completed episode inventory changed: {case}",
    )
    for name in ("extrinsics.npy", "undistorted_intrinsics.npy"):
        _require_regular_entry(
            episode / name, mode=0o400, role=f"completed {name} {case}"
        )

    camera_files = {
        "aligned_timestamps.txt",
        "mask_refined.h5",
        "metadata.json",
        "rendered_depth.h5",
        "rendered_depth.meta.json",
        "undistorted.mp4",
    }
    tracking_files = {"tracking.meta.json", "vel.h5", "visibility.h5"}
    for camera in cameras:
        camera_root = episode / camera
        tracking = camera_root / "tracking"
        _require_directory(
            camera_root, mode=0o500, role=f"completed camera {case}/{camera}"
        )
        _require(
            {entry.name for entry in camera_root.iterdir()}
            == {"tracking", *camera_files},
            f"completed camera inventory changed: {case}/{camera}",
        )
        _require_directory(
            tracking, mode=0o500, role=f"completed tracking {case}/{camera}"
        )
        _require(
            {entry.name for entry in tracking.iterdir()} == tracking_files,
            f"completed tracking inventory changed: {case}/{camera}",
        )
        for name in camera_files:
            _require_regular_entry(
                camera_root / name,
                mode=0o400,
                role=f"completed camera payload {case}/{camera}/{name}",
            )
        for name in tracking_files:
            _require_regular_entry(
                tracking / name,
                mode=0o400,
                role=f"completed tracking payload {case}/{camera}/{name}",
            )

    pcd = episode / "pcd_clean"
    expected_pcd = {
        *(f"{frame:06d}.npz" for frame in range(COMPLETED_PCD_FRAME_COUNT)),
        "pcd_clean.meta.json",
    }
    _require_directory(pcd, mode=0o500, role=f"completed clean point clouds {case}")
    _require(
        {entry.name for entry in pcd.iterdir()} == expected_pcd,
        f"completed clean point-cloud inventory changed: {case}",
    )
    for name in expected_pcd:
        _require_regular_entry(
            pcd / name, mode=0o400, role=f"completed clean point cloud {case}/{name}"
        )

    robot = episode / "robot"
    robot_files = {"robot.meta.json", "robot.npz"}
    _require_directory(robot, mode=0o500, role=f"completed robot root {case}")
    _require(
        {entry.name for entry in robot.iterdir()} == robot_files,
        f"completed robot inventory changed: {case}",
    )
    for name in robot_files:
        _require_regular_entry(
            robot / name, mode=0o400, role=f"completed robot payload {case}/{name}"
        )

    splat = episode / "splatfacto"
    expected_splats = {
        *(f"splat_{frame}.ply" for frame in range(COMPLETED_SPLAT_FRAME_COUNT)),
        "splatfacto.meta.json",
    }
    _require_directory(splat, mode=0o500, role=f"completed splat root {case}")
    _require(
        {entry.name for entry in splat.iterdir()} == expected_splats,
        f"completed splat inventory changed: {case}",
    )
    for name in expected_splats:
        _require_regular_entry(
            splat / name, mode=0o400, role=f"completed splat payload {case}/{name}"
        )

    relative = reconstruction.relative_to(root).as_posix()
    rows = [
        dict(row)
        for row in inventory["rows"]
        if str(row["path"]) == relative or str(row["path"]).startswith(f"{relative}/")
    ]
    staged_relative = staged.relative_to(root).as_posix()
    staged_rows = [
        dict(row)
        for row in inventory["rows"]
        if str(row["path"]).startswith(f"{staged_relative}/")
    ]
    expected_staging = COMPLETED_STAGING_EXPECTATIONS.get(case)
    observed_staging = {
        "directory_count": sum(row["type"] == "directory" for row in staged_rows),
        "regular_file_count": sum(row["type"] == "file" for row in staged_rows),
        "regular_file_bytes": sum(
            int(row.get("size_bytes", 0))
            for row in staged_rows
            if row["type"] == "file"
        ),
    }
    _require(
        expected_staging is not None and observed_staging == dict(expected_staging),
        f"completed staging metadata changed: {case}",
    )
    directory_count = 6 + 2 * len(cameras)
    regular_file_count = (
        7 + COMPLETED_PCD_FRAME_COUNT + COMPLETED_SPLAT_FRAME_COUNT + 9 * len(cameras)
    )
    _require(
        len(rows) == directory_count + regular_file_count
        and sum(row["type"] == "directory" for row in rows) == directory_count
        and sum(row["type"] == "file" for row in rows) == regular_file_count
        and all(
            row["mode_octal"] == ("0500" if row["type"] == "directory" else "0400")
            for row in rows
        ),
        f"completed reconstruction is not exactly represented in inventory: {case}",
    )
    return {
        "case": case,
        "camera_count": len(cameras),
        "pcd_clean_frame_count": COMPLETED_PCD_FRAME_COUNT,
        "splatfacto_frame_count": COMPLETED_SPLAT_FRAME_COUNT,
        "staged_descendant_directory_count": observed_staging["directory_count"],
        "staged_regular_file_count": observed_staging["regular_file_count"],
        "staged_regular_file_bytes": observed_staging["regular_file_bytes"],
        "staged_inventory_sha256": hashlib.sha256(
            _canonical_bytes({"rows": staged_rows})
        ).hexdigest(),
        "directory_count": directory_count,
        "regular_file_count": regular_file_count,
        "regular_file_bytes": sum(
            int(row.get("size_bytes", 0)) for row in rows if row["type"] == "file"
        ),
        "inventory_sha256": hashlib.sha256(
            _canonical_bytes({"rows": rows})
        ).hexdigest(),
        "all_directories_mode_0500": True,
        "all_files_mode_0400_and_single_link": True,
        "all_content_bound_by_stable_inventory_sha256": True,
    }


def _validate_failed_reconstruction(root: Path) -> dict[str, Any]:
    reconstruction = (
        root
        / "calibration/private-targets"
        / FAILED_CASE
        / "fresh-official-reconstruction"
    )
    _require_directory(reconstruction, mode=0o700, role="partial third reconstruction")
    _require(
        {entry.name for entry in reconstruction.iterdir()} == {"staged-aligned"},
        "partial third reconstruction root changed",
    )
    staged = reconstruction / "staged-aligned"
    episode = staged / "episode_0000"
    _require_directory(staged, mode=None, role="partial staged root")
    _require(
        {entry.name for entry in staged.iterdir()} == {"episode_0000"},
        "partial staged episode inventory changed",
    )
    _require_directory(episode, mode=None, role="partial staged episode")
    expected_episode = {
        *FAILED_CAMERAS,
        "extrinsics.npy",
        "undistorted_intrinsics.npy",
        "robot",
        "splatfacto",
    }
    _require(
        {entry.name for entry in episode.iterdir()} == expected_episode,
        "partial staged episode inventory changed",
    )
    for camera in FAILED_CAMERAS:
        camera_root = episode / camera
        _require_directory(camera_root, mode=None, role=f"partial camera {camera}")
        _require(
            {entry.name for entry in camera_root.iterdir()}
            == {
                "aligned_timestamps.txt",
                "mask_refined.h5",
                "metadata.json",
                "undistorted.mp4",
            },
            f"partial camera inventory changed: {camera}",
        )
    robot = episode / "robot"
    _require_directory(robot, mode=None, role="partial robot root")
    _require(
        {entry.name for entry in robot.iterdir()} == {"robot.meta.json", "robot.npz"},
        "partial robot inventory changed",
    )
    splat = episode / "splatfacto"
    _require_directory(splat, mode=None, role="partial splat root")
    expected_splats = {f"splat_{frame}.ply" for frame in range(81)}
    _require(
        {entry.name for entry in splat.iterdir()}
        == {".scratch_000080", *expected_splats},
        "partial splat inventory changed",
    )
    scratch = splat / ".scratch_000080"
    timestamp = scratch / "outputs/splat_80/splatfacto" / FAILED_SCRATCH_TIMESTAMP
    expected_scratch_directories = {
        "outputs",
        "outputs/splat_80",
        "outputs/splat_80/splatfacto",
        f"outputs/splat_80/splatfacto/{FAILED_SCRATCH_TIMESTAMP}",
        f"outputs/splat_80/splatfacto/{FAILED_SCRATCH_TIMESTAMP}/nerfstudio_models",
    }
    observed_scratch_directories = {
        path.relative_to(scratch).as_posix()
        for path in scratch.rglob("*")
        if path.is_dir() and not path.is_symlink()
    }
    expected_scratch_files = {
        f"outputs/splat_80/splatfacto/{FAILED_SCRATCH_TIMESTAMP}/config.yml",
        f"outputs/splat_80/splatfacto/{FAILED_SCRATCH_TIMESTAMP}/dataparser_transforms.json",
        (
            f"outputs/splat_80/splatfacto/{FAILED_SCRATCH_TIMESTAMP}/"
            "nerfstudio_models/step-000000249.ckpt"
        ),
    }
    observed_scratch_files = {
        path.relative_to(scratch).as_posix()
        for path in scratch.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    _require(
        observed_scratch_directories == expected_scratch_directories
        and observed_scratch_files == expected_scratch_files
        and timestamp.is_dir(),
        "partial final-frame scratch inventory changed",
    )
    return {
        "failed_case": FAILED_CASE,
        "failed_case_evidence": {
            "staged_camera_count": len(FAILED_CAMERAS),
            "splatfacto_ply_count": 81,
            "partial_final_frame_scratch_present": True,
            "final_frame_checkpoint_present": True,
            "official_reconstruction_audit_present": False,
            "official_target_archive_present": False,
            "official_target_manifest_present": False,
        },
    }


def _validate_launcher(*, allowed_modes: set[int]) -> dict[str, Any]:
    observed_root = os.lstat(LAUNCHER)
    _require(
        stat.S_ISDIR(observed_root.st_mode)
        and not stat.S_ISLNK(observed_root.st_mode)
        and LAUNCHER.resolve() == LAUNCHER
        and stat.S_IMODE(observed_root.st_mode) in allowed_modes,
        "attempt-4 launcher directory is absent, linked, or has the wrong mode",
    )
    _require(
        {entry.name for entry in LAUNCHER.iterdir()} == {"output.log", "exit.code"},
        "attempt-4 launcher allowlist changed",
    )
    marker_values = {name: marker for name, (marker, _count) in LOG_MARKERS.items()}
    log = _stable_file(LAUNCHER / "output.log", markers=marker_values)
    observed_counts = log.pop("marker_counts")
    expected_counts = {name: count for name, (_marker, count) in LOG_MARKERS.items()}
    _require(
        log["mode_octal"] == "0400"
        and log["size_bytes"] == EXPECTED_LAUNCHER_LOG_SIZE
        and log["sha256"] == EXPECTED_LAUNCHER_LOG_SHA256
        and observed_counts == expected_counts,
        "attempt-4 launcher log identity or terminal markers changed",
    )
    exit_record = _stable_file(LAUNCHER / "exit.code", collect=True)
    exit_payload = exit_record.pop("payload")
    _require(
        exit_record["mode_octal"] == "0400"
        and exit_record["size_bytes"] == EXPECTED_EXIT_SIZE
        and exit_record["sha256"] == EXPECTED_EXIT_SHA256
        and exit_payload == EXPECTED_EXIT_BYTES,
        "attempt-4 launcher exit evidence changed",
    )
    return {
        "path": os.fspath(LAUNCHER),
        "exact_file_allowlist": ["exit.code", "output.log"],
        "output_log": log,
        "exit_code": exit_record,
        "terminal_marker_counts": observed_counts,
        "log_scanned_for_fixed_markers_only": True,
        "log_numerical_payload_parsed": False,
    }


def _validate_execution_boundary(
    root: Path,
    inventory: Mapping[str, Any],
    launcher: Mapping[str, Any],
) -> dict[str, Any]:
    root_names = {entry.name for entry in root.iterdir()}
    root_names.discard(REPORT_NAME)
    _require(
        root_names
        == {
            "calibration",
            "calibration-lock.json",
            "post-withdrawal-development-use-disclosure.json",
            "replacement-source",
            EXPECTED_DEPLOYED_CODE_NAME,
        },
        "attempt-4 formal root inventory changed",
    )
    calibration = root / "calibration"
    expected_calibration_children = {
        ".shard-0.claim",
        ".shard-1.claim",
        ".v8-outcome-phase.claim",
        "cases",
        "logs",
        "private-targets",
        "query-inputs",
        "query-outputs",
        "shard-0.lock-verify.log",
        "shard-1.lock-verify.log",
    }
    _require(
        {entry.name for entry in calibration.iterdir()}
        == expected_calibration_children,
        "attempt-4 calibration root inventory changed",
    )
    for name in (".shard-0.claim", ".shard-1.claim", ".v8-outcome-phase.claim"):
        path = calibration / name
        expected_mode = 0o500 if name == ".v8-outcome-phase.claim" else 0o700
        _require_directory(path, mode=expected_mode, role=name)
        _require(not any(path.iterdir()), f"claim directory is not empty: {name}")

    cases_root = calibration / "cases"
    observed_cases = sorted(path.name for path in cases_root.iterdir() if path.is_dir())
    _require(
        observed_cases == sorted(EXPECTED_CASES), "calibration case inventory changed"
    )
    stage_paths = {
        "frame_zero_manifest": "frame-zero/frame_zero_bundle.manifest.json",
        "physical_prior_seal": "physical/physical_prior_seal.json",
        "prefix_authorization": "prefix-authorization.json",
        "online_prediction_seal": "online/online_prediction_seal.json",
        "frozen_field_manifest": "frozen-field/preoutcome-frozen-field-manifest.json",
    }
    stage_counts: dict[str, int] = {}
    for stage, relative in stage_paths.items():
        paths = [cases_root / case / relative for case in EXPECTED_CASES]
        _require(
            all(path.is_file() and not path.is_symlink() for path in paths),
            f"{stage} is incomplete",
        )
        _require(
            all(stat.S_IMODE(os.lstat(path).st_mode) == 0o400 for path in paths),
            f"{stage} is not fully sealed",
        )
        stage_counts[stage] = len(paths)

    private = calibration / "private-targets"
    query_inputs = calibration / "query-inputs"
    query_outputs = calibration / "query-outputs"
    for path, role in (
        (private, "private target root"),
        (query_inputs, "query input root"),
        (query_outputs, "query output root"),
    ):
        _require_directory(path, mode=0o700, role=role)
        _require(
            sorted(child.name for child in path.iterdir() if child.is_dir())
            == sorted(EXPECTED_CASES),
            f"{role} case inventory changed",
        )
        _require(
            all(
                stat.S_IMODE(os.lstat(path / case).st_mode) == 0o700
                for case in EXPECTED_CASES
            ),
            f"{role} case-directory mode changed",
        )

    completed_reconstruction_inventories: dict[str, dict[str, Any]] = {}
    for case in COMPLETED_OUTCOME_CASES:
        completed_private = private / case
        _require(
            {entry.name for entry in completed_private.iterdir()}
            == {
                "fresh-official-reconstruction",
                "official-target-manifest.json",
                "official-target.npz",
            },
            f"completed private target inventory changed: {case}",
        )
        completed_reconstruction_inventories[case] = _validate_completed_reconstruction(
            root,
            case=case,
            inventory=inventory,
        )
        for path, role in (
            (completed_private / "official-target-manifest.json", "target manifest"),
            (completed_private / "official-target.npz", "target archive"),
            (
                query_inputs / case / "official-frame-zero-query-manifest.json",
                "x0 manifest",
            ),
            (query_inputs / case / "official-frame-zero-query.npz", "x0 archive"),
            (query_outputs / case / "queried-prediction-seal.json", "query seal"),
            (query_outputs / case / "queried-prediction.npz", "queried archive"),
        ):
            _require_regular_mode(path, mode=0o400, role=f"{role} {case}")
        _require(
            {entry.name for entry in (query_inputs / case).iterdir()}
            == {
                "official-frame-zero-query-manifest.json",
                "official-frame-zero-query.npz",
            }
            and {entry.name for entry in (query_outputs / case).iterdir()}
            == {"queried-prediction-seal.json", "queried-prediction.npz"},
            f"completed query-pair inventory changed: {case}",
        )

    failed_private = private / FAILED_CASE
    _require(
        {entry.name for entry in failed_private.iterdir()}
        == {"fresh-official-reconstruction"},
        "failed third private target inventory changed",
    )
    _require(
        not any((query_inputs / FAILED_CASE).iterdir()), "failed third x0 query exists"
    )
    _require(
        not any((query_outputs / FAILED_CASE).iterdir()),
        "failed third queried prediction exists",
    )
    partial = _validate_failed_reconstruction(root)

    later = set(EXPECTED_CASES) - set(COMPLETED_OUTCOME_CASES) - {FAILED_CASE}
    for case in later:
        _require(
            not any((private / case).iterdir()), f"later private target exists: {case}"
        )
        _require(
            not any((query_inputs / case).iterdir()), f"later x0 query exists: {case}"
        )
        _require(
            not any((query_outputs / case).iterdir()),
            f"later queried prediction exists: {case}",
        )

    forbidden = (
        calibration / "calibration-score-evidence.json",
        calibration / "calibration-gate-decision.json",
        root / "confirmation-lock.json",
        root / "confirmation",
    )
    _require(
        not any(os.path.lexists(path) for path in forbidden),
        "second-barrier score, decision, or confirmation evidence exists",
    )
    forbidden_inventory = [
        str(row["path"])
        for row in inventory["rows"]
        if str(row["path"]).endswith(".failed.log")
        or str(row["path"]).startswith("confirmation")
    ]
    _require(
        not forbidden_inventory,
        f"forbidden terminal artifacts exist: {forbidden_inventory}",
    )
    marker_counts = launcher["terminal_marker_counts"]
    _require(
        marker_counts
        == {name: expected for name, (_marker, expected) in LOG_MARKERS.items()},
        "launcher terminal event boundary changed",
    )
    return {
        "calibration_case_directory_count": 15,
        **{f"{stage}_count": count for stage, count in stage_counts.items()},
        "first_cohort_barrier_validated_count": 1,
        "outcome_phase_claim_count": 1,
        "official_target_archive_count": 2,
        "official_target_manifest_count": 2,
        "official_x0_archive_count": 2,
        "official_x0_manifest_count": 2,
        "queried_prediction_archive_count": 2,
        "queried_prediction_seal_count": 2,
        "completed_reconstruction_inventories": completed_reconstruction_inventories,
        "partial_reconstruction_count": 1,
        **partial,
        "second_cohort_barrier_validated_count": 0,
        "score_evidence_count": 0,
        "gate_decision_count": 0,
        "confirmation_lock_count": 0,
        "confirmation_root_count": 0,
        "launcher_terminal_marker_counts": dict(marker_counts),
    }


def _build_report(
    *,
    code: Mapping[str, Any],
    lock: Mapping[str, Any],
    launcher: Mapping[str, Any],
    boundary: Mapping[str, Any],
    inventory: Mapping[str, Any],
    operator_source: Mapping[str, Any],
) -> dict[str, Any]:
    failure_path = (
        ACTIVE
        / "calibration/private-targets"
        / FAILED_CASE
        / "fresh-official-reconstruction/staged-aligned/episode_0000/"
        "splatfacto/.scratch_000080/outputs/splat_80/splatfacto"
        / FAILED_SCRATCH_TIMESTAMP
    )
    return _signed(
        {
            "schema_version": 1,
            "artifact_kind": "Deform360HeldV81Attempt4PostBarrierWithdrawalReport",
            "protocol_id": PROTOCOL_ID,
            "execution_attempt": EXECUTION_ATTEMPT,
            "status": STATUS,
            "disposition": DISPOSITION,
            "result_status": "NO_CALIBRATION_RESULT",
            "date": "2026-07-22",
            "formal_root_before_withdrawal": os.fspath(ACTIVE),
            "immutable_archive_path": os.fspath(ARCHIVE),
            "executed_withdrawal_operator_source": dict(operator_source),
            "deployed_code": dict(code),
            "calibration_lock": dict(lock),
            "durable_launcher_evidence": dict(launcher),
            "terminal_failure": {
                "evidence_origin": "durable-launcher-log-fixed-marker-scan",
                "outer_outcome_driver_exit_code": 2,
                "exception_type": "OSError",
                "errno": 24,
                "exception_message_class": "Too many open files",
                "failed_case": FAILED_CASE,
                "failure_path": os.fspath(failure_path),
                "failure_phase": (
                    "third target reconstruction after final-frame training and "
                    "before reconstruction audit, target seal, second barrier, or score"
                ),
            },
            "execution_boundary": dict(boundary),
            "stable_noncode_inventory": {
                **inventory,
                "stable_pre_report_hash_pass_count": 2,
                "hash_method": "O_NOFOLLOW stable SHA-256 byte streams",
                "payload_deserialization_performed": False,
            },
            "expected_postseal_inventory": _sealed_inventory(inventory),
            "information_boundary": {
                "first_complete_cohort_barrier_crossed": True,
                "completed_target_x0_queried_pairs": 2,
                "third_target_reconstruction_partial": True,
                "third_target_archive_or_manifest_created": False,
                "second_complete_cohort_barrier_crossed": False,
                "future_score_capability_issued": False,
                "score_created_or_read": False,
                "gate_decision_created_or_read": False,
                "confirmation_created_or_read": False,
                "forensic_operator_deserialized_target_query_prediction_or_score": False,
                "forensic_operator_decoded_image_video_pointcloud_or_mask": False,
                "forensic_operator_hashed_all_noncode_bytes": True,
                "launcher_log_examined_only_for_fixed_operational_markers": True,
                "held_values_or_method_performance_used_for_recovery": False,
            },
            "successor_disposition": {
                "reuse_attempt4_predictions": False,
                "reuse_attempt4_targets_queries_or_queried_predictions": False,
                "reuse_attempt4_source_manifests_or_partial_reconstruction": False,
                "reuse_attempt4_score_gate_or_confirmation_artifacts": False,
                "full_fresh_calibration_rerun_required": True,
                "active_held_v8_root_must_remain_absent_until_successor_lock": True,
                "archive_and_launcher_must_remain_fully_nonwritable": True,
                "post_rename_integrity_verification_required_before_pointer": True,
            },
        }
    )


def _validate_report_identity(report: Mapping[str, Any]) -> None:
    _require(
        report.get("artifact_kind")
        == "Deform360HeldV81Attempt4PostBarrierWithdrawalReport"
        and report.get("protocol_id") == PROTOCOL_ID
        and report.get("execution_attempt") == EXECUTION_ATTEMPT
        and report.get("status") == STATUS
        and report.get("disposition") == DISPOSITION
        and report.get("artifact_sha256") == _artifact_sha256(report),
        "attempt-4 withdrawal report identity changed",
    )


def _make_tree_read_only(root: Path) -> None:
    paths: list[Path] = []
    for current, directories, files in os.walk(root, topdown=False, followlinks=False):
        parent = Path(current)
        paths.extend(parent / name for name in sorted(files))
        paths.extend(parent / name for name in sorted(directories))
    for path in paths:
        observed = os.lstat(path)
        _require(not stat.S_ISLNK(observed.st_mode), f"archive symlink refused: {path}")
        if stat.S_ISREG(observed.st_mode):
            _require(observed.st_nlink == 1, f"archive hardlink refused: {path}")
            os.chmod(path, 0o400, follow_symlinks=False)
        elif stat.S_ISDIR(observed.st_mode):
            os.chmod(path, 0o500, follow_symlinks=False)
        else:
            raise RuntimeError(f"archive special file refused: {path}")
    os.chmod(root, 0o500, follow_symlinks=False)


def _seal_launcher(expected: Mapping[str, Any]) -> dict[str, Any]:
    observed = _validate_launcher(allowed_modes={0o700, 0o500})
    _require(observed == expected, "launcher evidence changed before sealing")
    os.chmod(LAUNCHER, 0o500, follow_symlinks=False)
    sealed = _validate_launcher(allowed_modes={0o500})
    _require(sealed == expected, "sealed launcher evidence changed")
    return {**sealed, "root_mode_octal": "0500", "fully_nonwritable": True}


def _verify_archive(report: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        ARCHIVE.is_dir()
        and not ARCHIVE.is_symlink()
        and ARCHIVE.resolve() == ARCHIVE
        and not os.path.lexists(ACTIVE),
        "attempt-4 archive/root state is invalid",
    )
    code_path = _code_directory(ARCHIVE)
    code = _repository_binding(code_path)
    _require(code == report.get("deployed_code"), "archived deployed code changed")
    lock = _validate_lock(ARCHIVE, code)
    _require(lock == report.get("calibration_lock"), "archived lock binding changed")
    observed = _inventory(ARCHIVE, code_name=code_path.name)
    recorded = report.get("stable_noncode_inventory")
    _require(isinstance(recorded, Mapping), "report inventory is absent")
    _require(
        _content_projection(observed) == _content_projection(recorded),
        "archive content differs from the pre-rename inventory",
    )
    _make_tree_read_only(ARCHIVE)
    sealed_code = _repository_binding(code_path)
    _require(sealed_code == code, "deployed code changed while sealing")
    sealed = _inventory(ARCHIVE, code_name=code_path.name)
    _require(
        sealed == report.get("expected_postseal_inventory"),
        "sealed archive inventory changed",
    )
    _require(
        stat.S_IMODE(os.lstat(ARCHIVE).st_mode) == 0o500
        and not any(os.lstat(path).st_mode & 0o222 for path in ARCHIVE.rglob("*")),
        "attempt-4 archive remains writable",
    )
    launcher = _seal_launcher(report["durable_launcher_evidence"])
    report_path = ARCHIVE / REPORT_NAME
    report_record = _stable_file(report_path)
    _require(report_record["mode_octal"] == "0400", "archived report is not mode 0400")
    return {
        "archive_path": os.fspath(ARCHIVE),
        "archive_root_mode_octal": "0500",
        "archive_fully_nonwritable": True,
        "postseal_noncode_inventory_sha256": sealed["inventory_sha256"],
        "postseal_noncode_entry_count": sealed["entry_count"],
        "withdrawal_report_path": os.fspath(report_path),
        "withdrawal_report_size_bytes": report_record["size_bytes"],
        "withdrawal_report_file_sha256": report_record["sha256"],
        "withdrawal_report_artifact_sha256": report["artifact_sha256"],
        "deployed_code": sealed_code,
        "durable_launcher_evidence": launcher,
        "independent_post_rename_integrity_verified": True,
    }


def _build_completion(
    verification: Mapping[str, Any], *, operator_source: Mapping[str, Any]
) -> dict[str, Any]:
    return _signed(
        {
            "schema_version": 1,
            "artifact_kind": "Deform360HeldV81Attempt4WithdrawalIntegrityCompletion",
            "protocol_id": PROTOCOL_ID,
            "execution_attempt": EXECUTION_ATTEMPT,
            "status": "withdrawal-integrity-complete",
            "disposition": DISPOSITION,
            "date": "2026-07-22",
            **verification,
            "executed_withdrawal_operator_source": dict(operator_source),
            "pointer_contract": {
                "path": os.fspath(POINTER),
                "artifact_kind": "Deform360HeldV81Attempt4WithdrawalPointer",
                "pointer_must_bind_this_completion": True,
                "completion_does_not_predict_pointer_hash_to_avoid_circularity": True,
            },
        }
    )


def _completion_binding(completion: Mapping[str, Any]) -> dict[str, Any]:
    observed = _stable_file(COMPLETION)
    _require(observed["mode_octal"] == "0400", "attempt-4 completion is not mode 0400")
    return {
        "path": os.fspath(COMPLETION),
        **observed,
        "artifact_sha256": completion["artifact_sha256"],
    }


def _build_pointer(
    verification: Mapping[str, Any],
    *,
    operator_source: Mapping[str, Any],
    completion_binding: Mapping[str, Any],
) -> dict[str, Any]:
    return _signed(
        {
            "schema_version": 1,
            "artifact_kind": "Deform360HeldV81Attempt4WithdrawalPointer",
            "protocol_id": PROTOCOL_ID,
            "execution_attempt": EXECUTION_ATTEMPT,
            "status": STATUS,
            "disposition": DISPOSITION,
            "date": "2026-07-22",
            **verification,
            "executed_withdrawal_operator_source": dict(operator_source),
            "withdrawal_integrity_completion": dict(completion_binding),
            "active_held_v8_root_absent_after_archive": not os.path.lexists(ACTIVE),
            "outer_outcome_driver_exit_code": 2,
            "first_cohort_barrier_crossed": True,
            "second_cohort_barrier_crossed": False,
            "completed_target_x0_queried_pairs": 2,
            "score_evidence_count": 0,
            "gate_decision_count": 0,
            "confirmation_accessed": False,
        }
    )


def _prepare_active_report() -> dict[str, Any]:
    _require(
        ACTIVE.is_dir()
        and not ACTIVE.is_symlink()
        and ACTIVE.resolve() == ACTIVE
        and stat.S_IMODE(os.lstat(ACTIVE).st_mode) == 0o700,
        "active held-v8 root is not a canonical mode-0700 directory",
    )
    code_path = _code_directory(ACTIVE)
    code = _repository_binding(code_path)
    lock = _validate_lock(ACTIVE, code)
    launcher = _validate_launcher(allowed_modes={0o700, 0o500})
    first = _inventory(ACTIVE, code_name=code_path.name)
    second = _inventory(ACTIVE, code_name=code_path.name)
    _require(first == second, "attempt-4 evidence changed across hash passes")
    boundary = _validate_execution_boundary(ACTIVE, first, launcher)
    operator_source = _operator_source_binding()
    expected = _build_report(
        code=code,
        lock=lock,
        launcher=launcher,
        boundary=boundary,
        inventory=first,
        operator_source=operator_source,
    )
    report_path = ACTIVE / REPORT_NAME
    _exclusive_json(report_path, expected)
    existing = _validate_signed_metadata(
        report_path, role="attempt-4 withdrawal report"
    )
    _require(existing == expected, "existing active withdrawal report changed")
    third = _inventory(ACTIVE, code_name=code_path.name)
    _require(first == third, "attempt-4 evidence changed while report was written")
    return expected


def _finish_archive() -> dict[str, Any]:
    report_path = ARCHIVE / REPORT_NAME
    report = _validate_signed_metadata(report_path, role="attempt-4 withdrawal report")
    _validate_report_identity(report)
    operator_source = _operator_source_binding()
    _require(
        report.get("executed_withdrawal_operator_source") == operator_source,
        "executed withdrawal operator source changed",
    )
    verification = _verify_archive(report)
    expected_completion = _build_completion(
        verification, operator_source=operator_source
    )
    _exclusive_json(COMPLETION, expected_completion)
    completion = _validate_signed_metadata(
        COMPLETION, role="attempt-4 withdrawal integrity completion"
    )
    _require(completion == expected_completion, "completion changed")
    completion_binding = _completion_binding(completion)
    expected_pointer = _build_pointer(
        verification,
        operator_source=operator_source,
        completion_binding=completion_binding,
    )
    _exclusive_json(POINTER, expected_pointer)
    pointer = _validate_signed_metadata(POINTER, role="attempt-4 withdrawal pointer")
    _require(pointer == expected_pointer, "attempt-4 pointer changed")
    return {
        "archive": os.fspath(ARCHIVE),
        "report_file_sha256": verification["withdrawal_report_file_sha256"],
        "report_artifact_sha256": report["artifact_sha256"],
        "pointer_file_sha256": _stable_file(POINTER)["sha256"],
        "pointer_artifact_sha256": pointer["artifact_sha256"],
        "completion_file_sha256": completion_binding["sha256"],
        "completion_artifact_sha256": completion["artifact_sha256"],
        "operator_source_sha256": operator_source["sha256"],
        "postseal_noncode_inventory_sha256": verification[
            "postseal_noncode_inventory_sha256"
        ],
        "launcher_log_sha256": verification["durable_launcher_evidence"]["output_log"][
            "sha256"
        ],
        "independent_post_rename_integrity_verified": True,
        "idempotent_recovery_complete": True,
    }


def main() -> int:
    _require(socket.gethostname() == "workstation2", "must run on gpuserver6000")
    running = _running_formal_processes()
    _require(not running, f"formal held-v8 processes remain: {running}")
    active_exists = os.path.lexists(ACTIVE)
    archive_exists = os.path.lexists(ARCHIVE)
    _require(
        not (active_exists and archive_exists), "active and archived roots coexist"
    )
    _require(active_exists or archive_exists, "attempt-4 active/archive root is absent")
    if active_exists:
        _require(
            not os.path.lexists(POINTER) and not os.path.lexists(COMPLETION),
            "pointer/completion exists before atomic archive",
        )
        report = _prepare_active_report()
        os.rename(ACTIVE, ARCHIVE)
        _fsync_directory(BASE)
        _require(not os.path.lexists(ACTIVE), "active root survived atomic rename")
        archived = _validate_signed_metadata(
            ARCHIVE / REPORT_NAME, role="renamed attempt-4 withdrawal report"
        )
        _require(archived == report, "withdrawal report changed across rename")
    result = _finish_archive()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
