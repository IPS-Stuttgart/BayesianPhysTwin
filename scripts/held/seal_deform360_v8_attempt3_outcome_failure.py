#!/usr/bin/env python3
"""Seal and archive the terminal held-v8 attempt-3 post-barrier failure.

This forensic operator never deserializes a target, query, prediction, image,
point cloud, or other outcome payload.  It inventories every non-code byte via
``O_NOFOLLOW`` SHA-256 streams, validates the exact filesystem boundary that
was observed after the failed x0 worker, writes a signed withdrawal report,
atomically renames the formal root, seals the archive, independently verifies
the renamed evidence, and writes an immutable pointer outside the archive.

The operation is recoverable after interruption.  Re-running it completes and
validates an existing report, renamed archive, or external pointer; it never
replaces an existing artifact.
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
ARCHIVE = BASE / "held-v8-attempt-3-withdrawn-postbarrier"
REPORT_NAME = "execution-withdrawal-postbarrier-attempt3.json"
POINTER = BASE / "held-v8-attempt-3-withdrawal-pointer.json"

PROTOCOL_ID = "deform360-held-online-belief-v8"
STATUS = "withdrawn-postbarrier-before-queried-prediction-or-score"
DISPOSITION = (
    "WITHDRAWN_AFTER_TARGET_AND_X0_BEFORE_ANY_QUERIED_PREDICTION_SEAL_OR_SCORE"
)
FAILED_CASE = "072-cotton-clohesline-ep0003"

EXPECTED_DEPLOYED_HEAD = "9ad7ad2b385f7abc5e8c42081a41018980dd3827"
EXPECTED_DEPLOYED_TREE_SHA256 = (
    "445f325dca5710c9873951445cb26107966e5344333edd8a69ac380e50e09546"
)
EXPECTED_LOCK_FILE_SHA256 = (
    "285639feb459857e822067252add52b80b99309d66444ebe02b978ef452cd0da"
)
EXPECTED_LOCK_ARTIFACT_SHA256 = (
    "1922166ac792ec6ac8360c4edd1ec6cc29b069beb1cdc3cff11e2fac0a426d6a"
)

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

STAGED_CAMERAS = (
    "brics-odroid-001_cam0",
    "brics-odroid-008_cam0",
    "brics-odroid-012_cam0",
    "brics-odroid-014_cam1",
    "brics-odroid-016_cam0",
    "brics-odroid-021_cam0",
    "brics-odroid-022_cam0",
    "brics-odroid-022_cam1",
)
PCD_CLEAN_FRAME_COUNT = 76
SPLATFACTO_FRAME_COUNT = 81

EXPECTED_FORMAL_METADATA: Mapping[str, tuple[int, str]] = {
    (
        "calibration/private-targets/072-cotton-clohesline-ep0003/"
        "fresh-official-reconstruction/held-v8-official-reconstruction-audit.json"
    ): (
        243456,
        "3f061372ec0949e11b8f4f3f9ffb5c0a17b9b29f1f6e6fc3e4e822e6b6b12a69",
    ),
    (
        "calibration/private-targets/072-cotton-clohesline-ep0003/"
        "official-target-manifest.json"
    ): (
        4489,
        "d4a94135cd88e92f512fbed1cf6b716a7aeccdd35df38f80963a2feee2b0a135",
    ),
    (
        "calibration/query-inputs/072-cotton-clohesline-ep0003/"
        "official-frame-zero-query-manifest.json"
    ): (
        1603,
        "1e09724cfdcf4542af8f0d73d90a701521288ab4e4b37ae4e9946202d15cb624",
    ),
}
EXPECTED_FORMAL_ARRAY_SIZES: Mapping[str, int] = {
    (
        "calibration/private-targets/072-cotton-clohesline-ep0003/"
        "official-target.npz"
    ): 552054,
    (
        "calibration/query-inputs/072-cotton-clohesline-ep0003/"
        "official-frame-zero-query.npz"
    ): 8818,
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


def _sha256_regular_file(path: Path) -> tuple[int, str, int]:
    """Hash one stable regular file without following a symlink."""

    before = os.lstat(path)
    _require(
        stat.S_ISREG(before.st_mode) and not stat.S_ISLNK(before.st_mode),
        f"regular non-symlink file required: {path}",
    )
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        _require(
            (before.st_dev, before.st_ino, before.st_size)
            == (opened.st_dev, opened.st_ino, opened.st_size),
            f"file changed before hashing: {path}",
        )
        while block := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(block)
        after_open = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = os.lstat(path)
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    _require(
        identity
        == (
            after_open.st_dev,
            after_open.st_ino,
            after_open.st_size,
            after_open.st_mtime_ns,
            after_open.st_ctime_ns,
        )
        == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ),
        f"file changed while hashing: {path}",
    )
    return before.st_size, digest.hexdigest(), stat.S_IMODE(before.st_mode)


def _read_metadata_json(path: Path, *, role: str) -> dict[str, Any]:
    """Deserialize operator metadata only, never a target/query payload."""

    _size, _digest, _mode = _sha256_regular_file(path)
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        payload = bytearray()
        while block := os.read(descriptor, 1024 * 1024):
            payload.extend(block)
    finally:
        os.close(descriptor)
    try:
        value = json.loads(bytes(payload).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{role} is not JSON metadata") from error
    _require(isinstance(value, dict), f"{role} is not a JSON object")
    return value


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> tuple[int, str]:
    payload = _pretty_json_bytes(value)
    descriptor = os.open(
        path,
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
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)
    os.chmod(path, 0o400, follow_symlinks=False)
    size, digest, mode = _sha256_regular_file(path)
    _require(mode == 0o400 and size == len(payload), f"sealed JSON changed: {path}")
    return size, digest


def _validate_signed_metadata(path: Path, *, role: str) -> dict[str, Any]:
    state = os.lstat(path)
    _require(
        stat.S_ISREG(state.st_mode)
        and not stat.S_ISLNK(state.st_mode)
        and stat.S_IMODE(state.st_mode) == 0o400,
        f"{role} is not a sealed mode-0400 file",
    )
    value = _read_metadata_json(path, role=role)
    observed = value.get("artifact_sha256")
    _require(
        isinstance(observed, str)
        and observed == _artifact_sha256(value),
        f"{role} artifact hash changed",
    )
    return value


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
        joined = " ".join(argv)
        if any(marker in joined for marker in _FORMAL_PROCESS_MARKERS):
            records.append({"pid": int(entry.name), "argv": argv})
    return sorted(records, key=lambda row: int(row["pid"]))


def _run_git(root: Path, arguments: Sequence[str]) -> bytes:
    environment = {
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_OPTIONAL_LOCKS": "0",
        "HOME": "/home/florianpfaff",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
    }
    completed = subprocess.run(
        ["git", "-c", "core.fileMode=false", "-C", str(root), *arguments],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
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
        _require(bool(separator) and bool(path_bytes), "malformed Git tree record")
        fields = header.split(b" ")
        _require(len(fields) == 3, "malformed Git tree header")
        mode, kind, object_id = (field.decode("ascii") for field in fields)
        path = path_bytes.decode("utf-8")
        _require(
            mode in {"100644", "100755"}
            and kind == "blob"
            and _HEAD_RE.fullmatch(object_id) is not None,
            f"unsupported deployed-code entry: {path}",
        )
        _require(
            path and not path.startswith("/") and ".." not in Path(path).parts,
            "unsafe deployed-code path",
        )
        rows.append({"mode": mode, "type": kind, "object_id": object_id, "path": path})
    _require(bool(rows), "deployed Git tree is empty")
    _require(
        [row["path"] for row in rows] == sorted(row["path"] for row in rows),
        "deployed Git tree is not sorted",
    )
    return rows


def _repository_binding(code: Path) -> dict[str, Any]:
    state = os.lstat(code)
    _require(
        stat.S_ISDIR(state.st_mode)
        and not stat.S_ISLNK(state.st_mode)
        and code.resolve() == code,
        "deployed code is not a canonical directory",
    )
    _require((code / ".git").is_dir(), "deployed code is not a Git repository")
    top = _run_git(code, ["rev-parse", "--show-toplevel"]).decode().strip()
    _require(top == str(code), "deployed Git top level changed")
    head = _run_git(code, ["rev-parse", "HEAD"]).decode().strip().lower()
    _require(_HEAD_RE.fullmatch(head) is not None, "deployed HEAD is invalid")
    _require(
        _run_git(code, ["status", "--porcelain=v1", "--untracked-files=all"]) == b"",
        "deployed worktree content changed",
    )
    _require(
        _run_git(code, ["rev-parse", "--is-shallow-repository"]).decode().strip()
        == "false",
        "deployed repository is shallow",
    )
    _run_git(code, ["fsck", "--full", "--no-dangling"])
    rows = _parse_git_tree(_run_git(code, ["ls-tree", "-r", "-z", "HEAD"]))
    _require(
        all((code / row["path"]).is_file() for row in rows),
        "deployed tracked content is absent",
    )
    tree_sha256 = hashlib.sha256(_canonical_bytes(rows)).hexdigest()
    return {
        "path": code.name,
        "git_head": head,
        "head_text_sha256": hashlib.sha256(head.encode("ascii")).hexdigest(),
        "git_tree_record_count": len(rows),
        "git_tree_manifest_sha256": tree_sha256,
    }


def _code_directory(root: Path) -> Path:
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and not path.is_symlink()
        and path.name.startswith("code-")
        and _HEAD_RE.fullmatch(path.name.removeprefix("code-")) is not None
    ]
    _require(len(candidates) == 1, "expected exactly one deployed-code directory")
    return candidates[0]


def _validate_lock(root: Path, code: Mapping[str, Any]) -> dict[str, Any]:
    path = root / "calibration-lock.json"
    size, file_sha256, mode = _sha256_regular_file(path)
    _require(mode == 0o400, "calibration lock is not mode 0400")
    _require(file_sha256 == EXPECTED_LOCK_FILE_SHA256, "attempt-3 lock changed")
    lock = _read_metadata_json(path, role="calibration lock")
    _require(
        lock.get("artifact_sha256") == EXPECTED_LOCK_ARTIFACT_SHA256
        and _artifact_sha256(lock) == EXPECTED_LOCK_ARTIFACT_SHA256,
        "attempt-3 lock artifact hash changed",
    )
    _require(
        lock.get("protocol_id") == PROTOCOL_ID
        and lock.get("stage") == "calibration"
        and lock.get("execution_attempt") == 3
        and lock.get("held_root") == os.fspath(ACTIVE),
        "attempt-3 lock identity changed",
    )
    _require(
        tuple(lock.get("calibration_case_whitelist", ())) == EXPECTED_CASES,
        "attempt-3 calibration cohort changed",
    )
    bindings = lock.get("immutable_bindings")
    _require(isinstance(bindings, Mapping), "lock immutable bindings are absent")
    _require(
        bindings.get("method_deployed_snapshot_tree")
        == code["git_tree_manifest_sha256"]
        == EXPECTED_DEPLOYED_TREE_SHA256,
        "deployed snapshot tree differs from the lock",
    )
    _require(
        code["git_head"] == EXPECTED_DEPLOYED_HEAD
        and code["path"] == f"code-{EXPECTED_DEPLOYED_HEAD}"
        and bindings.get("method_head_text_sha256") == code["head_text_sha256"],
        "deployed HEAD differs from the lock",
    )
    return {
        "path": "calibration-lock.json",
        "mode_octal": "0400",
        "size_bytes": size,
        "file_sha256": file_sha256,
        "artifact_sha256": EXPECTED_LOCK_ARTIFACT_SHA256,
    }


def _inventory(root: Path, *, code_name: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        relative_parent = current_path.relative_to(root)
        directories[:] = sorted(
            name
            for name in directories
            if not (relative_parent == Path(".") and name == code_name)
        )
        for name in directories:
            path = current_path / name
            state = os.lstat(path)
            _require(
                stat.S_ISDIR(state.st_mode) and not stat.S_ISLNK(state.st_mode),
                f"directory symlink or special entry refused: {path}",
            )
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "type": "directory",
                    "mode_octal": f"{stat.S_IMODE(state.st_mode):04o}",
                }
            )
        for name in sorted(files):
            path = current_path / name
            relative = path.relative_to(root)
            if relative == Path(REPORT_NAME):
                continue
            size, digest, mode = _sha256_regular_file(path)
            rows.append(
                {
                    "path": relative.as_posix(),
                    "type": "file",
                    "mode_octal": f"{mode:04o}",
                    "size_bytes": size,
                    "sha256": digest,
                }
            )
    rows.sort(key=lambda row: str(row["path"]))
    _require(
        len({str(row["path"]) for row in rows}) == len(rows),
        "duplicate inventory path",
    )
    payload = {"rows": rows}
    return {
        "rows": rows,
        "entry_count": len(rows),
        "directory_count": sum(row["type"] == "directory" for row in rows),
        "regular_file_count": sum(row["type"] == "file" for row in rows),
        "regular_file_bytes": sum(
            int(row.get("size_bytes", 0)) for row in rows if row["type"] == "file"
        ),
        "inventory_sha256": hashlib.sha256(_canonical_bytes(payload)).hexdigest(),
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
    payload = {"rows": rows}
    return {
        **{key: value for key, value in inventory.items() if key != "rows"},
        "rows": rows,
        "inventory_sha256": hashlib.sha256(_canonical_bytes(payload)).hexdigest(),
    }


def _expected_reconstruction_paths() -> tuple[set[str], set[str]]:
    episode = (
        "calibration/private-targets/"
        f"{FAILED_CASE}/fresh-official-reconstruction/staged-aligned/episode_0000"
    )
    reconstruction = (
        f"calibration/private-targets/{FAILED_CASE}/fresh-official-reconstruction"
    )
    directories = {
        reconstruction,
        f"{reconstruction}/staged-aligned",
        episode,
        f"{episode}/pcd_clean",
        f"{episode}/robot",
        f"{episode}/splatfacto",
    }
    files = {
        f"{reconstruction}/held-v8-official-reconstruction-audit.json",
        f"{episode}/extrinsics.npy",
        f"{episode}/undistorted_intrinsics.npy",
        f"{episode}/pcd_clean/pcd_clean.meta.json",
        f"{episode}/robot/robot.meta.json",
        f"{episode}/robot/robot.npz",
        f"{episode}/splatfacto/splatfacto.meta.json",
    }
    for camera in STAGED_CAMERAS:
        camera_root = f"{episode}/{camera}"
        tracking = f"{camera_root}/tracking"
        directories.update({camera_root, tracking})
        files.update(
            {
                f"{camera_root}/aligned_timestamps.txt",
                f"{camera_root}/mask_refined.h5",
                f"{camera_root}/metadata.json",
                f"{camera_root}/rendered_depth.h5",
                f"{camera_root}/rendered_depth.meta.json",
                f"{camera_root}/undistorted.mp4",
                f"{tracking}/tracking.meta.json",
                f"{tracking}/vel.h5",
                f"{tracking}/visibility.h5",
            }
        )
    files.update(
        f"{episode}/pcd_clean/{frame:06d}.npz"
        for frame in range(PCD_CLEAN_FRAME_COUNT)
    )
    files.update(
        f"{episode}/splatfacto/splat_{frame}.ply"
        for frame in range(SPLATFACTO_FRAME_COUNT)
    )
    return directories, files


def _require_directory(path: Path, *, mode: int, role: str) -> None:
    state = os.lstat(path)
    _require(
        stat.S_ISDIR(state.st_mode)
        and not stat.S_ISLNK(state.st_mode)
        and stat.S_IMODE(state.st_mode) == mode,
        f"{role} is not a real mode-{mode:04o} directory: {path}",
    )


def _validate_execution_boundary(root: Path, inventory: Mapping[str, Any]) -> dict[str, Any]:
    calibration = root / "calibration"
    _require_directory(calibration / ".v8-outcome-phase.claim", mode=0o500, role="outcome claim")
    _require(
        not any((calibration / ".v8-outcome-phase.claim").iterdir()),
        "outcome claim is not empty",
    )
    cases_root = calibration / "cases"
    case_names = sorted(path.name for path in cases_root.iterdir() if path.is_dir())
    _require(case_names == sorted(EXPECTED_CASES), "calibration case inventory changed")
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
        _require(all(path.is_file() and not path.is_symlink() for path in paths), f"{stage} is incomplete")
        _require(
            all(stat.S_IMODE(os.lstat(path).st_mode) == 0o400 for path in paths),
            f"{stage} is not fully sealed",
        )
        stage_counts[stage] = len(paths)
    _require(all(count == 15 for count in stage_counts.values()), "pre-outcome seal count changed")

    private = calibration / "private-targets"
    query_inputs = calibration / "query-inputs"
    query_outputs = calibration / "query-outputs"
    for path, role in ((private, "private target root"), (query_inputs, "query input root"), (query_outputs, "query output root")):
        _require_directory(path, mode=0o700, role=role)
        observed = sorted(child.name for child in path.iterdir() if child.is_dir())
        _require(observed == sorted(EXPECTED_CASES), f"{role} case inventory changed")
        _require(
            all(stat.S_IMODE(os.lstat(path / case).st_mode) == 0o700 for case in EXPECTED_CASES),
            f"{role} case-directory mode changed",
        )

    for case in EXPECTED_CASES:
        if case != FAILED_CASE:
            _require(not any((private / case).iterdir()), f"later private target exists: {case}")
            _require(not any((query_inputs / case).iterdir()), f"later x0 query exists: {case}")
        _require(not any((query_outputs / case).iterdir()), f"queried prediction exists: {case}")

    failed_private = private / FAILED_CASE
    failed_query = query_inputs / FAILED_CASE
    _require(
        {entry.name for entry in failed_private.iterdir()}
        == {
            "fresh-official-reconstruction",
            "official-target-manifest.json",
            "official-target.npz",
        },
        "first-case private target inventory changed",
    )
    _require(
        {entry.name for entry in failed_query.iterdir()}
        == {
            "official-frame-zero-query-manifest.json",
            "official-frame-zero-query.npz",
        },
        "first-case x0 query inventory changed",
    )

    indexed = {str(row["path"]): row for row in inventory["rows"]}
    reconstruction_directories, reconstruction_files = _expected_reconstruction_paths()
    observed_reconstruction_directories = {
        path
        for path, row in indexed.items()
        if row["type"] == "directory"
        and path.startswith(
            f"calibration/private-targets/{FAILED_CASE}/fresh-official-reconstruction"
        )
    }
    observed_reconstruction_files = {
        path
        for path, row in indexed.items()
        if row["type"] == "file"
        and path.startswith(
            f"calibration/private-targets/{FAILED_CASE}/fresh-official-reconstruction"
        )
    }
    _require(
        observed_reconstruction_directories == reconstruction_directories,
        "reconstruction directory inventory changed",
    )
    _require(
        observed_reconstruction_files == reconstruction_files,
        "reconstruction file inventory changed",
    )
    _require(
        all(indexed[path]["mode_octal"] == "0500" for path in reconstruction_directories)
        and all(indexed[path]["mode_octal"] == "0400" for path in reconstruction_files),
        "reconstruction evidence is not sealed",
    )

    metadata_records: dict[str, Any] = {}
    for relative, (expected_size, expected_sha256) in EXPECTED_FORMAL_METADATA.items():
        row = indexed.get(relative)
        _require(
            row
            == {
                "path": relative,
                "type": "file",
                "mode_octal": "0400",
                "size_bytes": expected_size,
                "sha256": expected_sha256,
            },
            f"formal metadata identity changed: {relative}",
        )
        metadata_records[relative] = dict(row)
    array_records: dict[str, Any] = {}
    for relative, expected_size in EXPECTED_FORMAL_ARRAY_SIZES.items():
        row = indexed.get(relative)
        _require(
            isinstance(row, Mapping)
            and row.get("type") == "file"
            and row.get("mode_octal") == "0400"
            and row.get("size_bytes") == expected_size
            and isinstance(row.get("sha256"), str),
            f"formal payload metadata changed: {relative}",
        )
        array_records[relative] = dict(row)

    forbidden_exact = (
        calibration / "calibration-score-evidence.json",
        calibration / "calibration-gate-decision.json",
        root / "confirmation-lock.json",
        root / "confirmation",
    )
    _require(
        not any(os.path.lexists(path) for path in forbidden_exact),
        "score, decision, or confirmation evidence exists",
    )
    forbidden_names = [
        str(row["path"])
        for row in inventory["rows"]
        if str(row["path"]).endswith(".failed.log")
        or "queried-prediction" in Path(str(row["path"])).name
        or "score" in Path(str(row["path"])).name
        or "decision" in Path(str(row["path"])).name
        or str(row["path"]).startswith("confirmation")
    ]
    _require(not forbidden_names, f"forbidden terminal evidence exists: {forbidden_names}")
    return {
        "calibration_case_directory_count": 15,
        **{f"{stage}_count": count for stage, count in stage_counts.items()},
        "outcome_phase_claim_count": 1,
        "private_target_case_directory_count": 15,
        "private_target_nonempty_case_count": 1,
        "query_input_case_directory_count": 15,
        "query_input_nonempty_case_count": 1,
        "query_output_case_directory_count": 15,
        "query_output_file_count": 0,
        "official_target_archive_count": 1,
        "official_target_manifest_count": 1,
        "official_x0_archive_count": 1,
        "official_x0_manifest_count": 1,
        "queried_prediction_archive_count": 0,
        "queried_prediction_seal_count": 0,
        "score_evidence_count": 0,
        "gate_decision_count": 0,
        "confirmation_lock_count": 0,
        "confirmation_root_count": 0,
        "failed_log_count": 0,
        "reconstruction_directory_count": len(reconstruction_directories),
        "reconstruction_file_count": len(reconstruction_files),
        "formal_metadata_records": metadata_records,
        "formal_payload_byte_stream_records": array_records,
    }


def _build_report(
    *,
    code: Mapping[str, Any],
    lock: Mapping[str, Any],
    boundary: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    return _signed(
        {
            "schema_version": 1,
            "artifact_kind": "Deform360HeldV8Attempt3PostBarrierWithdrawalReport",
            "protocol_id": PROTOCOL_ID,
            "execution_attempt": 3,
            "status": STATUS,
            "disposition": DISPOSITION,
            "date": "2026-07-22",
            "formal_root_before_withdrawal": os.fspath(ACTIVE),
            "immutable_archive_path": os.fspath(ARCHIVE),
            "deployed_code": dict(code),
            "calibration_lock": dict(lock),
            "terminal_failure": {
                "evidence_origin": "launcher-observed-not-filesystem-persisted",
                "console_log_persisted": False,
                "outer_outcome_driver_exit_code": 2,
                "driver_terminal_event": {
                    "event": "FAIL_CLOSED",
                    "error_type": "CalledProcessError",
                    "message_suffix": "returned non-zero exit status 1.",
                },
                "inner_x0_worker_exit_code": 1,
                "inner_exception_type": "ValueError",
                "inner_exception_message": (
                    "an assimilation center has no query identity within the exclusion radius"
                ),
                "traceback_locus": {
                    "module": "deform360_frozen_query_field.py",
                    "function": "map_assimilation_centers_to_queries",
                    "boundary": "before _queried_arrays returned",
                },
                "failed_case": FAILED_CASE,
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
                "first_case_official_target_created": True,
                "first_case_official_x0_query_created": True,
                "queried_prediction_created_or_read": False,
                "score_created_or_read": False,
                "gate_decision_created_or_read": False,
                "confirmation_created_or_read": False,
                "forensic_operator_deserialized_target_query_or_prediction": False,
                "forensic_operator_decoded_image_video_pointcloud_or_mask": False,
                "forensic_operator_hashed_all_noncode_bytes": True,
            },
            "successor_disposition": {
                "reuse_attempt3_predictions_targets_queries_or_partial_artifacts": False,
                "active_held_v8_root_must_remain_absent": True,
                "archive_must_remain_fully_nonwritable": True,
                "post_rename_integrity_verification_required_before_pointer": True,
            },
        }
    )


def _validate_report_identity(report: Mapping[str, Any]) -> None:
    _require(
        report.get("artifact_kind")
        == "Deform360HeldV8Attempt3PostBarrierWithdrawalReport"
        and report.get("protocol_id") == PROTOCOL_ID
        and report.get("execution_attempt") == 3
        and report.get("status") == STATUS
        and report.get("disposition") == DISPOSITION,
        "attempt-3 withdrawal report identity changed",
    )
    _require(
        report.get("artifact_sha256") == _artifact_sha256(report),
        "attempt-3 withdrawal report artifact hash changed",
    )


def _make_tree_read_only(root: Path) -> None:
    paths: list[Path] = []
    for current, directories, files in os.walk(root, topdown=False, followlinks=False):
        current_path = Path(current)
        paths.extend(current_path / name for name in sorted(files))
        paths.extend(current_path / name for name in sorted(directories))
    for path in paths:
        state = os.lstat(path)
        _require(not stat.S_ISLNK(state.st_mode), f"archive symlink refused: {path}")
        if stat.S_ISREG(state.st_mode):
            os.chmod(path, 0o400, follow_symlinks=False)
        elif stat.S_ISDIR(state.st_mode):
            os.chmod(path, 0o500, follow_symlinks=False)
        else:
            raise RuntimeError(f"archive special file refused: {path}")
    os.chmod(root, 0o500, follow_symlinks=False)


def _verify_archive(report: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        ARCHIVE.is_dir()
        and not ARCHIVE.is_symlink()
        and ARCHIVE.resolve() == ARCHIVE
        and not os.path.lexists(ACTIVE),
        "attempt-3 archive/root state is invalid",
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
    sealed = _inventory(ARCHIVE, code_name=code_path.name)
    expected_sealed = report.get("expected_postseal_inventory")
    _require(sealed == expected_sealed, "sealed archive inventory changed")
    _require(
        stat.S_IMODE(os.lstat(ARCHIVE).st_mode) == 0o500
        and not any(
            os.lstat(path).st_mode & 0o222 for path in ARCHIVE.rglob("*")
        ),
        "attempt-3 archive remains writable",
    )
    report_path = ARCHIVE / REPORT_NAME
    report_size, report_sha256, report_mode = _sha256_regular_file(report_path)
    _require(report_mode == 0o400, "archived withdrawal report is not mode 0400")
    return {
        "archive_path": os.fspath(ARCHIVE),
        "archive_root_mode_octal": "0500",
        "archive_fully_nonwritable": True,
        "postseal_noncode_inventory_sha256": sealed["inventory_sha256"],
        "postseal_noncode_entry_count": sealed["entry_count"],
        "withdrawal_report_path": os.fspath(report_path),
        "withdrawal_report_size_bytes": report_size,
        "withdrawal_report_file_sha256": report_sha256,
        "withdrawal_report_artifact_sha256": report["artifact_sha256"],
        "deployed_code": code,
        "independent_post_rename_integrity_verified": True,
    }


def _build_pointer(verification: Mapping[str, Any]) -> dict[str, Any]:
    return _signed(
        {
            "schema_version": 1,
            "artifact_kind": "Deform360HeldV8Attempt3WithdrawalPointer",
            "protocol_id": PROTOCOL_ID,
            "execution_attempt": 3,
            "status": STATUS,
            "disposition": DISPOSITION,
            "date": "2026-07-22",
            **verification,
            "active_held_v8_root_absent_after_archive": not os.path.lexists(ACTIVE),
            "outer_outcome_driver_exit_code": 2,
            "failure_evidence_origin": "launcher-observed-not-filesystem-persisted",
            "queried_prediction_seal_count": 0,
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
    first = _inventory(ACTIVE, code_name=code_path.name)
    second = _inventory(ACTIVE, code_name=code_path.name)
    _require(first == second, "attempt-3 evidence changed across hash passes")
    boundary = _validate_execution_boundary(ACTIVE, first)
    expected = _build_report(code=code, lock=lock, boundary=boundary, inventory=first)
    report_path = ACTIVE / REPORT_NAME
    if os.path.lexists(report_path):
        existing = _validate_signed_metadata(report_path, role="attempt-3 withdrawal report")
        _require(existing == expected, "existing active withdrawal report changed")
    else:
        _exclusive_json(report_path, expected)
    third = _inventory(ACTIVE, code_name=code_path.name)
    _require(first == third, "attempt-3 evidence changed while report was written")
    return expected


def _finish_archive() -> dict[str, Any]:
    report_path = ARCHIVE / REPORT_NAME
    report = _validate_signed_metadata(report_path, role="attempt-3 withdrawal report")
    _validate_report_identity(report)
    verification = _verify_archive(report)
    expected_pointer = _build_pointer(verification)
    if os.path.lexists(POINTER):
        pointer = _validate_signed_metadata(POINTER, role="attempt-3 withdrawal pointer")
        _require(pointer == expected_pointer, "existing attempt-3 pointer changed")
    else:
        _exclusive_json(POINTER, expected_pointer)
        pointer = _validate_signed_metadata(POINTER, role="attempt-3 withdrawal pointer")
        _require(pointer == expected_pointer, "written attempt-3 pointer changed")
    return {
        "archive": os.fspath(ARCHIVE),
        "report_file_sha256": verification["withdrawal_report_file_sha256"],
        "report_artifact_sha256": report["artifact_sha256"],
        "pointer_file_sha256": _sha256_regular_file(POINTER)[1],
        "pointer_artifact_sha256": pointer["artifact_sha256"],
        "postseal_noncode_inventory_sha256": verification[
            "postseal_noncode_inventory_sha256"
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
    _require(not (active_exists and archive_exists), "active and archived roots coexist")
    _require(active_exists or archive_exists, "attempt-3 active/archive root is absent")
    if active_exists:
        _require(not os.path.lexists(POINTER), "pointer exists before atomic archive")
        report = _prepare_active_report()
        os.rename(ACTIVE, ARCHIVE)
        _require(not os.path.lexists(ACTIVE), "active root survived atomic rename")
        archived = _validate_signed_metadata(
            ARCHIVE / REPORT_NAME, role="renamed attempt-3 withdrawal report"
        )
        _require(archived == report, "withdrawal report changed across rename")
    result = _finish_archive()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
