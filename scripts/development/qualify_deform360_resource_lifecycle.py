#!/usr/bin/env python3
"""Qualify bounded per-fit Nerfstudio resources on non-held data.

This development operator has two independent gates:

* an isolated-process A/B fit compares the released Deform360 trainer with
  the resource-bounded wrapper, using the same copied dataset and seeds; and
* a same-process wrapped soak performs 243 short real fits by default, which
  crosses the three-case failure region that motivated the lifecycle fix.

The operator accepts no formal held path.  It copies only the files referenced
by a Nerfstudio ``transforms.json``, rewrites the absolute seed-Ply path into
each private copy, and runs every GPU worker with the pinned Python runtime.
The final canonical JSON evidence is self-signed with ``artifact_sha256``.
It is development evidence and must be frozen separately before a held lock
may bind it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import shutil
import socket
import stat
import subprocess
import sys
import traceback
from typing import Any, Mapping, Sequence


QUALIFICATION_ID = "deform360-nerfstudio-resource-lifecycle-qualification-v1"
PINNED_PYTHON = Path(
    "/mnt/corsair/florianpfaff/bpt-held-v5-runtimes/"
    "bpt-gpu-pip-4948737892f77c6a9496795e6c3f25b92fcea466ddb7b5f1e9c1b0de1137f004/"
    "bin/python"
)
PINNED_PYTHON_RUNTIME = PINNED_PYTHON.parent.parent
PINNED_PYTHON_FREEZE = Path(f"{PINNED_PYTHON_RUNTIME}.freeze.sorted.txt")
PINNED_PYTHON_TREE_MANIFEST = Path(f"{PINNED_PYTHON_RUNTIME}.tree-manifest.json")
PINNED_PYTHON_FREEZE_SHA256 = (
    "4948737892f77c6a9496795e6c3f25b92fcea466ddb7b5f1e9c1b0de1137f004"
)
PINNED_PYTHON_TREE_MANIFEST_SHA256 = (
    "8147db39bc3ab30943951ae5f304de48ffc819625d30a382d5305528b6601b61"
)
PINNED_PYTHON_SYMLINK_TARGET = "/usr/bin/python3"
PINNED_PYTHON_RESOLVED = Path("/usr/bin/python3.12")
PINNED_PYTHON_BASE_PREFIX = Path("/usr")
PINNED_PYTHON_RESOLVED_SHA256 = (
    "e1efa562c2cc2e35521a5c9c9b9939921001ff8ca9708a13ef15ace68cc2ccd7"
)
PINNED_DEFORM360 = Path(
    "/mnt/corsair/florianpfaff/bpt-held-v81-runtimes/"
    "Deform360-processing-0fe36f0b7a7a917ba62b5f8cee707299a9a4a317"
)
PINNED_DEFORM360_REVISION = "0fe36f0b7a7a917ba62b5f8cee707299a9a4a317"
DEFAULT_PUBLIC_DEV_DATASET = Path(
    "/mnt/corsair/florianpfaff/deform360-reusable-sota-v1/"
    "processing-sam2-dev-smoke/004-rubber-band/episode_0001/"
    "splatfacto/.scratch_000000"
)
FORMAL_HELD_PARENT = Path("/mnt/corsair/florianpfaff/bpt-online-belief-v1")
RELATIVE_WRAPPER_SOURCE = Path(
    "src/bayesian_phystwin/deform360_held_outcome_reconstruction.py"
)
RELATIVE_QUALIFICATION_SOURCE = Path(
    "scripts/development/qualify_deform360_resource_lifecycle.py"
)
AB_ITERATIONS = 250
SOAK_FIT_COUNT = 243
SOAK_ITERATIONS = 1
SOAK_TRAINER_REINITIALIZATION_INTERVAL = 81
FIRST_FIT_FD_GROWTH_LIMIT = 32
STEADY_FD_GROWTH_LIMIT = 4
STEADY_TASK_GROWTH_LIMIT = 4


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


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
    result.pop("artifact_sha256", None)
    result["artifact_sha256"] = _artifact_sha256(result)
    return result


def _canonical_json(value: Mapping[str, Any]) -> bytes:
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


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _is_formal_held_path(path: str | Path) -> bool:
    candidate = _absolute(path)
    try:
        relative = candidate.relative_to(FORMAL_HELD_PARENT)
    except ValueError:
        return False
    return bool(relative.parts) and relative.parts[0].startswith("held-")


def _assert_nonheld_path(
    path: str | Path,
    *,
    label: str,
    must_exist: bool,
) -> Path:
    """Resolve a path while rejecting formal held roots lexically and really."""

    absolute = _absolute(path)
    _require(
        not _is_formal_held_path(absolute),
        f"{label} points into a formal held root",
    )
    if must_exist:
        try:
            lexical = os.lstat(absolute)
        except OSError as error:
            raise ValueError(f"{label} is unavailable: {absolute}") from error
        _require(not stat.S_ISLNK(lexical.st_mode), f"{label} is a symlink")
    try:
        resolved = absolute.resolve(strict=must_exist)
    except OSError as error:
        raise ValueError(f"{label} is unavailable: {absolute}") from error
    _require(
        not _is_formal_held_path(resolved),
        f"{label} resolves into a formal held root",
    )
    return resolved


def _stable_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_regular_nofollow(path: str | Path, *, label: str) -> bytes:
    source = _assert_nonheld_path(path, label=label, must_exist=True)
    before = os.lstat(source)
    _require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file")
    descriptor = os.open(
        source,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        _require(
            _stable_identity(opened) == _stable_identity(before),
            f"{label} changed while opening",
        )
        digest_input: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest_input.append(block)
        after = os.fstat(descriptor)
        current = os.lstat(source)
        _require(
            _stable_identity(after) == _stable_identity(opened)
            and _stable_identity(current) == _stable_identity(opened),
            f"{label} changed while reading",
        )
        return b"".join(digest_input)
    finally:
        os.close(descriptor)


def _sha256_file(path: str | Path, *, label: str = "file") -> str:
    return hashlib.sha256(_read_regular_nofollow(path, label=label)).hexdigest()


def _bound_file(path: str | Path, *, label: str = "file") -> dict[str, Any]:
    source = _assert_nonheld_path(path, label=label, must_exist=True)
    observed = os.lstat(source)
    _require(stat.S_ISREG(observed.st_mode), f"{label} is not a regular file")
    return {
        "path": os.fspath(source),
        "sha256": _sha256_file(source, label=label),
        "size_bytes": observed.st_size,
        "mode_octal": f"{stat.S_IMODE(observed.st_mode):04o}",
    }


def _write_new_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    destination = _assert_nonheld_path(
        path, label="JSON evidence output", must_exist=False
    )
    _require(not os.path.lexists(destination), "JSON evidence output already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json(value)
    descriptor = os.open(
        destination,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(destination, 0o444, follow_symlinks=False)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination


def _write_new_regular(path: str | Path, payload: bytes) -> Path:
    destination = _assert_nonheld_path(
        path, label="materialized regular file", must_exist=False
    )
    _require(not os.path.lexists(destination), "materialized file already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        destination,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(destination, 0o444, follow_symlinks=False)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination


def _load_signed_json(path: str | Path, *, label: str) -> dict[str, Any]:
    payload = _read_regular_nofollow(path, label=label)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not UTF-8 JSON") from error
    _require(isinstance(value, dict), f"{label} is not a JSON object")
    _require(
        value.get("artifact_sha256") == _artifact_sha256(value),
        f"{label} artifact signature is invalid",
    )
    return value


def _relative_source_file(root: Path, value: str, *, label: str) -> tuple[Path, Path]:
    relative = Path(value)
    _require(not relative.is_absolute(), f"{label} must be relative")
    normalized = Path(os.path.normpath(os.fspath(relative)))
    _require(
        normalized.parts
        and normalized.parts[0] not in ("", ".", "..")
        and ".." not in normalized.parts,
        f"{label} escapes the dataset",
    )
    source = (root / normalized).resolve(strict=True)
    _require(root == source.parent or root in source.parents, f"{label} escapes root")
    _require(source.is_file() and not source.is_symlink(), f"{label} is not regular")
    return normalized, source


def _materialize_dataset(source: str | Path, destination: str | Path) -> dict[str, Any]:
    """Copy only referenced inputs and rewrite the absolute seed-Ply path."""

    source_root = _assert_nonheld_path(
        source, label="public development dataset", must_exist=True
    )
    _require(
        source_root.is_dir() and not source_root.is_symlink(),
        "public development dataset is not a canonical directory",
    )
    output = _assert_nonheld_path(
        destination, label="materialized development dataset", must_exist=False
    )
    _require(
        not os.path.lexists(output), "materialized development dataset already exists"
    )
    transforms_path = source_root / "transforms.json"
    transforms_bytes = _read_regular_nofollow(
        transforms_path, label="source transforms"
    )
    try:
        transforms = json.loads(transforms_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("source transforms are not UTF-8 JSON") from error
    _require(isinstance(transforms, dict), "source transforms are not an object")
    frames = transforms.get("frames")
    _require(isinstance(frames, list) and frames, "source transforms have no frames")

    raw_seed = transforms.get("ply_file_path")
    _require(isinstance(raw_seed, str) and raw_seed, "seed-Ply path is missing")
    seed_path = _assert_nonheld_path(raw_seed, label="source seed Ply", must_exist=True)
    _require(
        source_root == seed_path.parent or source_root in seed_path.parents,
        "source seed Ply is outside the development dataset",
    )
    seed_relative = seed_path.relative_to(source_root)
    _require(seed_path.is_file() and not seed_path.is_symlink(), "seed Ply is linked")

    referenced: dict[Path, Path] = {seed_relative: seed_path}
    for index, frame in enumerate(frames):
        _require(isinstance(frame, dict), f"frame {index} is not an object")
        file_path = frame.get("file_path")
        _require(isinstance(file_path, str), f"frame {index} has no file path")
        relative, frame_source = _relative_source_file(
            source_root, file_path, label=f"frame {index} file"
        )
        referenced[relative] = frame_source

    output.mkdir(parents=True, exist_ok=False)
    try:
        for relative, input_path in sorted(
            referenced.items(), key=lambda item: os.fspath(item[0])
        ):
            copied = output / relative
            payload = _read_regular_nofollow(
                input_path, label="referenced source dataset input"
            )
            _write_new_regular(copied, payload)
        rewritten = json.loads(json.dumps(transforms, allow_nan=False))
        rewritten["ply_file_path"] = os.fspath((output / seed_relative).resolve())
        portable = json.loads(json.dumps(rewritten, allow_nan=False))
        portable["ply_file_path"] = "<MATERIALIZED-SEED-PLY>"
        transformed = (
            json.dumps(rewritten, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        destination_transforms = output / "transforms.json"
        _write_new_regular(destination_transforms, transformed)
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise

    source_records = {
        os.fspath(relative): _bound_file(path, label="source dataset input")
        for relative, path in sorted(
            referenced.items(), key=lambda item: os.fspath(item[0])
        )
    }
    materialized_records = {
        os.fspath(relative): _bound_file(
            output / relative, label="materialized dataset input"
        )
        for relative in sorted(referenced, key=os.fspath)
    }
    materialized_records["transforms.json"] = _bound_file(
        output / "transforms.json", label="materialized transforms"
    )
    source_transforms_binding = _bound_file(transforms_path, label="source transforms")
    _require(
        source_transforms_binding["sha256"]
        == hashlib.sha256(transforms_bytes).hexdigest()
        and source_transforms_binding["size_bytes"] == len(transforms_bytes),
        "source transforms changed during materialization",
    )
    referenced_source_content = _record_content_identity(source_records)
    referenced_materialized_content = _record_content_identity(
        {
            name: value
            for name, value in materialized_records.items()
            if name != "transforms.json"
        }
    )
    _require(
        referenced_source_content == referenced_materialized_content,
        "source and materialized referenced content differ",
    )
    return {
        "source_root": os.fspath(source_root),
        "destination_root": os.fspath(output),
        "source_transforms": source_transforms_binding,
        "source_transforms_sha256": hashlib.sha256(transforms_bytes).hexdigest(),
        "materialized_transforms_sha256": hashlib.sha256(transformed).hexdigest(),
        "portable_transforms_sha256": hashlib.sha256(
            _canonical_bytes(portable)
        ).hexdigest(),
        "rewritten_field": "ply_file_path",
        "source_seed_ply_path": os.fspath(seed_path),
        "materialized_seed_ply_path": os.fspath(output / seed_relative),
        "frame_count": len(frames),
        "copied_regular_file_count": len(referenced) + 1,
        "source_records": source_records,
        "materialized_records": materialized_records,
        "referenced_source_content": referenced_source_content,
        "referenced_materialized_content": referenced_materialized_content,
        "referenced_source_materialized_content_equal": True,
        "unreferenced_outputs_copied": False,
    }


def _record_content_identity(records: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: {
            "sha256": value["sha256"],
            "size_bytes": value["size_bytes"],
        }
        for name, value in sorted(records.items())
    }


def _materialized_dataset_identity(audit: Mapping[str, Any]) -> dict[str, Any]:
    records = dict(audit["materialized_records"])
    records.pop("transforms.json", None)
    return {
        "portable_transforms_sha256": audit["portable_transforms_sha256"],
        "referenced_file_content": _record_content_identity(records),
        "frame_count": audit["frame_count"],
    }


def _materialized_inputs_stable(audit: Mapping[str, Any]) -> bool:
    records = audit.get("materialized_records")
    if not isinstance(records, Mapping):
        return False
    for record in records.values():
        if not isinstance(record, Mapping):
            return False
        path = record.get("path")
        if not isinstance(path, str):
            return False
        try:
            observed = _bound_file(path, label="post-qualification dataset input")
        except (OSError, ValueError):
            return False
        if observed != dict(record):
            return False
    return True


def _source_inputs_stable(audit: Mapping[str, Any]) -> bool:
    records = audit.get("source_records")
    transforms = audit.get("source_transforms")
    if not isinstance(records, Mapping) or not isinstance(transforms, Mapping):
        return False
    all_records = [*records.values(), transforms]
    for record in all_records:
        if not isinstance(record, Mapping):
            return False
        path = record.get("path")
        if not isinstance(path, str):
            return False
        try:
            observed = _bound_file(path, label="post-qualification source input")
        except (OSError, ValueError):
            return False
        if observed != dict(record):
            return False
    return True


def _assert_canonical_run_parameters(
    arguments: argparse.Namespace, dataset: Path
) -> dict[str, Any]:
    canonical_dataset = _assert_nonheld_path(
        DEFAULT_PUBLIC_DEV_DATASET,
        label="canonical public development dataset",
        must_exist=True,
    )
    expected = {
        "phase": "all",
        "cuda_device": 1,
        "seed": 0,
        "ab_iterations": AB_ITERATIONS,
        "soak_fit_count": SOAK_FIT_COUNT,
        "soak_iterations": SOAK_ITERATIONS,
        "first_fit_fd_growth_limit": FIRST_FIT_FD_GROWTH_LIMIT,
        "steady_fd_growth_limit": STEADY_FD_GROWTH_LIMIT,
        "steady_task_growth_limit": STEADY_TASK_GROWTH_LIMIT,
    }
    for name, expected_value in expected.items():
        observed = getattr(arguments, name)
        _require(
            observed == expected_value,
            f"canonical qualification requires {name}={expected_value!r}; "
            f"observed {observed!r}",
        )
    _require(
        dataset == canonical_dataset,
        "canonical qualification requires the exact resolved public development "
        "dataset",
    )
    return {"dataset": os.fspath(canonical_dataset), **expected}


def _python_runtime_binding(python: Path) -> dict[str, Any]:
    lexical = _absolute(python)
    _require(lexical == PINNED_PYTHON, "Python launcher path is not pinned")
    try:
        lexical_stat = os.lstat(lexical)
    except OSError as error:
        raise ValueError(f"pinned Python is unavailable: {lexical}") from error
    _require(stat.S_ISLNK(lexical_stat.st_mode), "pinned Python is not a symlink")
    symlink_target = os.readlink(lexical)
    _require(
        symlink_target == PINNED_PYTHON_SYMLINK_TARGET,
        "pinned Python symlink target changed",
    )
    resolved = lexical.resolve(strict=True)
    _require(resolved == PINNED_PYTHON_RESOLVED, "resolved pinned Python changed")
    resolved_stat = os.lstat(resolved)
    _require(
        stat.S_ISREG(resolved_stat.st_mode),
        "resolved pinned Python is not a regular file",
    )
    _require(os.access(lexical, os.X_OK), "pinned Python is not executable")
    environment = {
        "HOME": "/home/florianpfaff",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PIP_CONFIG_FILE": os.devnull,
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INPUT": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }
    command = [
        os.fspath(lexical),
        "-I",
        "-B",
        "-m",
        "pip",
        "freeze",
        "--all",
    ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        timeout=120,
        env=environment,
    )
    _require(result.returncode == 0, "pinned Python pip freeze --all failed")
    try:
        decoded = result.stdout.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("pinned Python pip freeze output is not UTF-8") from error
    lines = sorted(line.strip() for line in decoded.splitlines() if line.strip())
    _require(lines, "pinned Python pip freeze --all returned no distributions")
    normalized = ("\n".join(lines) + "\n").encode("utf-8")
    environment_root = lexical.parent.parent
    pyvenv_config = environment_root / "pyvenv.cfg"
    resolved_binding = _bound_file(resolved, label="resolved pinned Python executable")
    freeze_binding = _bound_file(
        PINNED_PYTHON_FREEZE, label="pinned Python frozen package inventory"
    )
    tree_binding = _bound_file(
        PINNED_PYTHON_TREE_MANIFEST,
        label="pinned Python frozen runtime tree manifest",
    )
    _require(
        resolved_binding["sha256"] == PINNED_PYTHON_RESOLVED_SHA256,
        "resolved pinned Python digest changed",
    )
    _require(
        freeze_binding["sha256"] == PINNED_PYTHON_FREEZE_SHA256
        and freeze_binding["mode_octal"] == "0400",
        "pinned Python frozen package inventory binding changed",
    )
    _require(
        tree_binding["sha256"] == PINNED_PYTHON_TREE_MANIFEST_SHA256
        and tree_binding["mode_octal"] == "0400",
        "pinned Python frozen tree manifest binding changed",
    )
    live_freeze_sha256 = hashlib.sha256(normalized).hexdigest()
    _require(
        live_freeze_sha256 == PINNED_PYTHON_FREEZE_SHA256,
        "live pip freeze differs from the frozen package inventory",
    )
    return {
        "lexical_path": os.fspath(lexical),
        "lexical_mode_octal": f"{stat.S_IMODE(lexical_stat.st_mode):04o}",
        "lexical_symlink_target": symlink_target,
        "resolved_executable": resolved_binding,
        "pyvenv_config": _bound_file(pyvenv_config, label="pinned Python pyvenv.cfg"),
        "frozen_package_inventory": freeze_binding,
        "frozen_runtime_tree_manifest": tree_binding,
        "pip_freeze_all": {
            "normalized_sha256": live_freeze_sha256,
            "normalized_line_count": len(lines),
            "normalized_size_bytes": len(normalized),
            "equals_frozen_package_inventory": True,
        },
    }


def _current_python_process_binding() -> dict[str, str]:
    executable = _absolute(sys.executable)
    base_executable_value = getattr(sys, "_base_executable", None)
    _require(
        isinstance(base_executable_value, str) and base_executable_value,
        "canonical parent Python has no base executable",
    )
    base_executable = _absolute(base_executable_value)
    prefix = _absolute(sys.prefix)
    base_prefix = _absolute(sys.base_prefix)
    _require(
        executable == PINNED_PYTHON,
        "canonical parent is not executing through the pinned Python launcher",
    )
    _require(
        base_executable == PINNED_PYTHON_RESOLVED,
        "canonical parent Python base executable changed",
    )
    _require(prefix == PINNED_PYTHON_RUNTIME, "canonical parent Python prefix changed")
    _require(
        base_prefix == PINNED_PYTHON_BASE_PREFIX,
        "canonical parent Python base prefix changed",
    )
    return {
        "sys_executable": os.fspath(executable),
        "sys_base_executable": os.fspath(base_executable),
        "sys_prefix": os.fspath(prefix),
        "sys_base_prefix": os.fspath(base_prefix),
    }


def _git_binding(root: Path, *, expected_head: str | None = None) -> dict[str, Any]:
    repository = _assert_nonheld_path(root, label="Git repository", must_exist=True)
    _require(repository.is_dir() and not repository.is_symlink(), "bad Git root")
    environment = {
        "GIT_OPTIONAL_LOCKS": "0",
        "HOME": "/home/florianpfaff",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
    }

    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["/usr/bin/git", "-C", repository, *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
            env=environment,
        )
        _require(result.returncode == 0, f"Git command failed: {' '.join(arguments)}")
        return result.stdout.strip()

    head = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD^{tree}")
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    ordinary_untracked = git("ls-files", "--others", "--exclude-standard")
    ignored_untracked = git("ls-files", "--others", "--ignored", "--exclude-standard")
    _require(not status, f"qualification repository is dirty: {repository}")
    _require(
        not ordinary_untracked,
        f"qualification repository has ordinary untracked files: {repository}",
    )
    _require(
        not ignored_untracked,
        f"qualification repository has ignored untracked files: {repository}",
    )
    if expected_head is not None:
        _require(head == expected_head, f"pinned repository HEAD changed: {repository}")
    return {
        "path": os.fspath(repository),
        "head": head,
        "tree": tree,
        "clean": True,
        "ordinary_untracked_file_count": 0,
        "ignored_untracked_file_count": 0,
    }


def _process_boundary() -> dict[str, int]:
    fd_root = Path("/proc/self/fd")
    task_root = Path("/proc/self/task")
    _require(fd_root.is_dir() and task_root.is_dir(), "procfs resource census missing")
    rss_kib: int | None = None
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            fields = line.split()
            _require(len(fields) >= 2 and fields[1].isdigit(), "VmRSS is malformed")
            rss_kib = int(fields[1])
            break
    _require(rss_kib is not None and rss_kib > 0, "VmRSS is unavailable")
    rlimit_soft, rlimit_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    _require(
        isinstance(rlimit_soft, int)
        and isinstance(rlimit_hard, int)
        and rlimit_soft > 0
        and rlimit_hard >= rlimit_soft,
        "RLIMIT_NOFILE is invalid",
    )
    return {
        "file_descriptor_count": len(os.listdir(fd_root)),
        "task_count": len(os.listdir(task_root)),
        "rss_kib": rss_kib,
        "rlimit_nofile_soft": rlimit_soft,
        "rlimit_nofile_hard": rlimit_hard,
    }


@dataclass(frozen=True)
class _GlobalStateSnapshot:
    event_writers_object_id: int
    event_writer_ids: tuple[int, ...]
    event_storage_object_id: int
    event_storage_ids: tuple[int, ...]
    global_buffer_object_id: int
    global_buffer_items: tuple[tuple[str, int], ...]
    profiler_object_id: int
    profiler_ids: tuple[int, ...]
    pytorch_profiler_id: int | None

    def record(self) -> dict[str, Any]:
        return {
            "event_writers_object_id": self.event_writers_object_id,
            "event_writer_ids": list(self.event_writer_ids),
            "event_storage_object_id": self.event_storage_object_id,
            "event_storage_ids": list(self.event_storage_ids),
            "global_buffer_object_id": self.global_buffer_object_id,
            "global_buffer_items": [list(value) for value in self.global_buffer_items],
            "profiler_object_id": self.profiler_object_id,
            "profiler_ids": list(self.profiler_ids),
            "pytorch_profiler_id": self.pytorch_profiler_id,
        }


def _global_state_snapshot(writer: Any, profiler: Any) -> _GlobalStateSnapshot:
    return _GlobalStateSnapshot(
        event_writers_object_id=id(writer.EVENT_WRITERS),
        event_writer_ids=tuple(id(value) for value in writer.EVENT_WRITERS),
        event_storage_object_id=id(writer.EVENT_STORAGE),
        event_storage_ids=tuple(id(value) for value in writer.EVENT_STORAGE),
        global_buffer_object_id=id(writer.GLOBAL_BUFFER),
        global_buffer_items=tuple(
            sorted(
                (repr(key), id(value)) for key, value in writer.GLOBAL_BUFFER.items()
            )
        ),
        profiler_object_id=id(profiler.PROFILER),
        profiler_ids=tuple(id(value) for value in profiler.PROFILER),
        pytorch_profiler_id=(
            None if profiler.PYTORCH_PROFILER is None else id(profiler.PYTORCH_PROFILER)
        ),
    )


def _seed_runtime(seed: int) -> dict[str, Any]:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    _require(torch.cuda.is_available(), "pinned CUDA runtime is unavailable")
    return {
        "seed": seed,
        "python_random_seeded": True,
        "numpy_seeded": True,
        "torch_cpu_seeded": True,
        "torch_cuda_seeded": True,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "cuda_device_count": torch.cuda.device_count(),
        "python_version": sys.version,
    }


def _import_trainers(
    code_root: Path, deform360_root: Path
) -> tuple[Any, Any, Any, Any]:
    code_source = code_root / "src"
    for path in (code_source, deform360_root):
        value = os.fspath(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    from deform360.processing import reconstruct_stage
    from nerfstudio.utils import profiler, writer
    from bayesian_phystwin import (
        deform360_held_outcome_reconstruction as reconstruction,
    )

    _require(
        Path(reconstruct_stage.__file__).resolve(strict=True)
        == deform360_root / "deform360/processing/reconstruct_stage.py",
        "Deform360 trainer loaded from another checkout",
    )
    _require(
        Path(reconstruction.__file__).resolve(strict=True)
        == code_root / RELATIVE_WRAPPER_SOURCE,
        "resource wrapper loaded from another checkout",
    )
    return (
        reconstruct_stage.NerfstudioSplatTrainer,
        reconstruction._ResourceBoundedSplatTrainer,
        writer,
        profiler,
    )


def _child_fit(arguments: argparse.Namespace) -> int:
    result_path = _assert_nonheld_path(
        arguments.result, label="fit child result", must_exist=False
    )
    try:
        code = _assert_nonheld_path(
            arguments.code_root, label="fit code root", must_exist=True
        )
        deform360 = _assert_nonheld_path(
            arguments.deform360_repo, label="fit Deform360 root", must_exist=True
        )
        dataset = _assert_nonheld_path(
            arguments.dataset, label="fit dataset", must_exist=True
        )
        output = _assert_nonheld_path(
            arguments.output_dir, label="fit output", must_exist=True
        )
        _require(arguments.variant in ("original", "wrapped"), "bad fit variant")
        runtime = _seed_runtime(arguments.seed)
        trainer_type, wrapper_type, writer, profiler = _import_trainers(code, deform360)
        delegate = trainer_type()
        trainer = wrapper_type(delegate) if arguments.variant == "wrapped" else delegate
        before_globals = _global_state_snapshot(writer, profiler)
        before = _process_boundary()
        output_filename = "splat.ply"
        produced = Path(
            trainer.train(dataset, output, output_filename, arguments.iterations)
        ).resolve(strict=True)
        _require(produced == (output / output_filename).resolve(), "fit output escaped")
        after = _process_boundary()
        after_globals = _global_state_snapshot(writer, profiler)
        globals_restored = before_globals == after_globals
        predicates = {
            "output_created": produced.is_file() and not produced.is_symlink(),
            "wrapped_fit_requires_global_restoration": (
                arguments.variant != "wrapped" or globals_restored
            ),
            "rlimit_nofile_soft_is_1024": before["rlimit_nofile_soft"] == 1024,
            "rlimit_nofile_unchanged": (
                after["rlimit_nofile_soft"] == before["rlimit_nofile_soft"]
                and after["rlimit_nofile_hard"] == before["rlimit_nofile_hard"]
            ),
        }
        passed = all(predicates.values())
        result = _signed(
            {
                "schema_version": 1,
                "artifact_kind": "Deform360ResourceLifecycleFitChildEvidence",
                "qualification_id": QUALIFICATION_ID,
                "variant": arguments.variant,
                "passed": passed,
                "parameters": {
                    "iterations": arguments.iterations,
                    "seed": arguments.seed,
                },
                "runtime": runtime,
                "dataset": os.fspath(dataset),
                "output": _bound_file(produced, label="fit output Ply"),
                "resource_boundary": {"before": before, "after": after},
                "global_state": {
                    "before": before_globals.record(),
                    "after": after_globals.record(),
                    "restored": globals_restored,
                },
                "predicates": predicates,
                "formal_held_path_supplied": False,
            }
        )
        _write_new_json(result_path, result)
        return 0 if passed else 2
    except BaseException as error:
        failure = _signed(
            {
                "schema_version": 1,
                "artifact_kind": "Deform360ResourceLifecycleFitChildEvidence",
                "qualification_id": QUALIFICATION_ID,
                "variant": getattr(arguments, "variant", "unknown"),
                "passed": False,
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
                "formal_held_path_supplied": False,
            }
        )
        if not os.path.lexists(result_path):
            _write_new_json(result_path, failure)
        return 2


def _evaluate_soak_boundaries(
    before: Mapping[str, int],
    fits: Sequence[Mapping[str, Any]],
    *,
    expected_fit_count: int,
    first_fd_growth_limit: int,
    steady_fd_growth_limit: int,
    steady_task_growth_limit: int,
) -> dict[str, Any]:
    valid = bool(fits) and len(fits) == expected_fit_count
    first = fits[0]["resource_boundary"] if fits else {}
    first_fd = int(first.get("file_descriptor_count", -1))
    first_tasks = int(first.get("task_count", -1))
    fd_values = [
        int(value["resource_boundary"]["file_descriptor_count"]) for value in fits
    ]
    task_values = [int(value["resource_boundary"]["task_count"]) for value in fits]
    rss_values = [int(value["resource_boundary"]["rss_kib"]) for value in fits]
    rlimit_values = [
        (
            int(value["resource_boundary"]["rlimit_nofile_soft"]),
            int(value["resource_boundary"]["rlimit_nofile_hard"]),
        )
        for value in fits
    ]
    before_rlimit = (
        int(before["rlimit_nofile_soft"]),
        int(before["rlimit_nofile_hard"]),
    )
    trainer_reinitialization_indices = [
        int(value["fit_index"])
        for value in fits
        if value.get("trainer_reinitialized") is True
    ]
    expected_trainer_reinitialization_indices = list(
        range(0, expected_fit_count, SOAK_TRAINER_REINITIALIZATION_INTERVAL)
    )
    predicates = {
        "fit_count_exact": valid,
        "trainer_reinitialization_indices_exact": (
            trainer_reinitialization_indices
            == expected_trainer_reinitialization_indices
        ),
        "all_fits_created_outputs": bool(fits)
        and all(value.get("output_created") is True for value in fits),
        "dataset_outputs_created_after_every_fit": bool(fits)
        and all(value.get("dataset_outputs_created") is True for value in fits),
        "cleanup_completed_after_every_fit": bool(fits)
        and all(value.get("cleanup_completed") is True for value in fits),
        "output_ply_absent_after_every_fit": bool(fits)
        and all(value.get("output_ply_absent_after_cleanup") is True for value in fits),
        "dataset_outputs_absent_after_every_fit": bool(fits)
        and all(
            value.get("dataset_outputs_absent_after_cleanup") is True for value in fits
        ),
        "resource_boundary_recorded_after_cleanup": bool(fits)
        and all(
            value.get("resource_boundary_stage") == "after_cleanup" for value in fits
        ),
        "globals_restored_after_every_fit": bool(fits)
        and all(value.get("globals_restored") is True for value in fits),
        "first_fit_fd_growth_within_limit": bool(fits)
        and first_fd <= int(before["file_descriptor_count"]) + first_fd_growth_limit,
        "steady_fd_growth_within_limit": bool(fits)
        and max(fd_values) <= first_fd + steady_fd_growth_limit,
        "steady_task_growth_within_limit": bool(fits)
        and max(task_values) <= first_tasks + steady_task_growth_limit,
        "resource_counts_positive": bool(fits)
        and min(fd_values) > 0
        and min(task_values) > 0
        and min(rss_values) > 0,
        "rlimit_nofile_soft_is_1024": before_rlimit[0] == 1024,
        "rlimit_nofile_unchanged": bool(fits)
        and all(value == before_rlimit for value in rlimit_values),
    }
    return {
        "passed": all(predicates.values()),
        "predicates": predicates,
        "reference": {
            "pre_fit": dict(before),
            "first_post_cleanup_fit": dict(first),
        },
        "observed": {
            "minimum_fd_count": min(fd_values) if fd_values else None,
            "maximum_fd_count": max(fd_values) if fd_values else None,
            "final_fd_count": fd_values[-1] if fd_values else None,
            "maximum_fd_growth_from_first_post_fit": (
                max(fd_values) - first_fd if fd_values else None
            ),
            "minimum_task_count": min(task_values) if task_values else None,
            "maximum_task_count": max(task_values) if task_values else None,
            "final_task_count": task_values[-1] if task_values else None,
            "maximum_task_growth_from_first_post_fit": (
                max(task_values) - first_tasks if task_values else None
            ),
            "minimum_rss_kib": min(rss_values) if rss_values else None,
            "maximum_rss_kib": max(rss_values) if rss_values else None,
            "final_rss_kib": rss_values[-1] if rss_values else None,
        },
        "limits": {
            "first_fit_fd_growth": first_fd_growth_limit,
            "steady_fd_growth": steady_fd_growth_limit,
            "steady_task_growth": steady_task_growth_limit,
        },
        "trainer_reinitialization": {
            "interval": SOAK_TRAINER_REINITIALIZATION_INTERVAL,
            "expected_indices": expected_trainer_reinitialization_indices,
            "observed_indices": trainer_reinitialization_indices,
        },
    }


def _child_soak(arguments: argparse.Namespace) -> int:
    result_path = _assert_nonheld_path(
        arguments.result, label="soak child result", must_exist=False
    )
    try:
        code = _assert_nonheld_path(
            arguments.code_root, label="soak code root", must_exist=True
        )
        deform360 = _assert_nonheld_path(
            arguments.deform360_repo, label="soak Deform360 root", must_exist=True
        )
        dataset = _assert_nonheld_path(
            arguments.dataset, label="soak dataset", must_exist=True
        )
        output = _assert_nonheld_path(
            arguments.output_dir, label="soak output", must_exist=True
        )
        runtime = _seed_runtime(arguments.seed)
        trainer_type, wrapper_type, writer, profiler = _import_trainers(code, deform360)
        initial_globals = _global_state_snapshot(writer, profiler)
        before = _process_boundary()
        fits: list[dict[str, Any]] = []
        trainer: Any | None = None
        for index in range(arguments.fit_count):
            trainer_reinitialized = index % SOAK_TRAINER_REINITIALIZATION_INTERVAL == 0
            if trainer_reinitialized:
                trainer = wrapper_type(trainer_type())
            _require(trainer is not None, "soak trainer was not initialized")
            output_filename = f"splat-{index:04d}.ply"
            produced = Path(
                trainer.train(
                    dataset,
                    output,
                    output_filename,
                    arguments.iterations,
                )
            ).resolve(strict=True)
            _require(
                produced == (output / output_filename).resolve(),
                "soak output escaped",
            )
            output_created = produced.is_file() and not produced.is_symlink()
            _require(output_created, "soak output Ply is not a regular file")
            output_size = produced.stat().st_size
            produced.unlink()
            dataset_outputs = dataset / "outputs"
            _require(
                os.path.lexists(dataset_outputs),
                "soak dataset outputs were not created",
            )
            dataset_outputs_stat = os.lstat(dataset_outputs)
            dataset_outputs_created = stat.S_ISDIR(dataset_outputs_stat.st_mode)
            _require(
                dataset_outputs_created
                and not stat.S_ISLNK(dataset_outputs_stat.st_mode),
                "soak dataset outputs are not a real directory",
            )
            shutil.rmtree(dataset_outputs)
            output_absent = not os.path.lexists(produced)
            dataset_outputs_absent = not os.path.lexists(dataset_outputs)
            _require(output_absent, "soak output Ply remains after cleanup")
            _require(
                dataset_outputs_absent,
                "soak dataset outputs remain after cleanup",
            )
            boundary = _process_boundary()
            current_globals = _global_state_snapshot(writer, profiler)
            fits.append(
                {
                    "fit_index": index,
                    "trainer_reinitialized": trainer_reinitialized,
                    "output_created": output_created,
                    "dataset_outputs_created": dataset_outputs_created,
                    "output_size_bytes": output_size,
                    "cleanup_completed": True,
                    "output_ply_absent_after_cleanup": output_absent,
                    "dataset_outputs_absent_after_cleanup": dataset_outputs_absent,
                    "resource_boundary_stage": "after_cleanup",
                    "resource_boundary": boundary,
                    "globals_restored": current_globals == initial_globals,
                    "global_state": current_globals.record(),
                }
            )
        evaluation = _evaluate_soak_boundaries(
            before,
            fits,
            expected_fit_count=arguments.fit_count,
            first_fd_growth_limit=arguments.first_fd_growth_limit,
            steady_fd_growth_limit=arguments.steady_fd_growth_limit,
            steady_task_growth_limit=arguments.steady_task_growth_limit,
        )
        result = _signed(
            {
                "schema_version": 1,
                "artifact_kind": "Deform360ResourceLifecycleSoakChildEvidence",
                "qualification_id": QUALIFICATION_ID,
                "passed": evaluation["passed"],
                "parameters": {
                    "fit_count": arguments.fit_count,
                    "iterations_per_fit": arguments.iterations,
                    "seed": arguments.seed,
                    "trainer_reinitialization_interval": (
                        SOAK_TRAINER_REINITIALIZATION_INTERVAL
                    ),
                },
                "runtime": runtime,
                "dataset": os.fspath(dataset),
                "initial_global_state": initial_globals.record(),
                "fits": fits,
                "evaluation": evaluation,
                "formal_held_path_supplied": False,
            }
        )
        _write_new_json(result_path, result)
        return 0 if evaluation["passed"] else 2
    except BaseException as error:
        failure = _signed(
            {
                "schema_version": 1,
                "artifact_kind": "Deform360ResourceLifecycleSoakChildEvidence",
                "qualification_id": QUALIFICATION_ID,
                "passed": False,
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
                "formal_held_path_supplied": False,
            }
        )
        if not os.path.lexists(result_path):
            _write_new_json(result_path, failure)
        return 2


def _compare_structured_arrays(left: Any, right: Any) -> dict[str, Any]:
    import numpy as np

    left_names = tuple(left.dtype.names or ())
    right_names = tuple(right.dtype.names or ())
    names_equal = left_names == right_names and bool(left_names)
    shape_equal = tuple(left.shape) == tuple(right.shape)
    records: dict[str, Any] = {}
    common = (
        left_names
        if names_equal
        else tuple(name for name in left_names if name in set(right_names))
    )
    for name in common:
        left_value = np.asarray(left[name])
        right_value = np.asarray(right[name])
        dtype_equal = left_value.dtype == right_value.dtype
        field_shape_equal = left_value.shape == right_value.shape
        left_finite = bool(np.all(np.isfinite(left_value)))
        right_finite = bool(np.all(np.isfinite(right_value)))
        exact = bool(
            dtype_equal
            and field_shape_equal
            and np.array_equal(left_value, right_value)
        )
        max_abs: float | None = None
        max_abs_finite = False
        if field_shape_equal and left_finite and right_finite and left_value.size:
            difference = np.abs(
                left_value.astype(np.float64) - right_value.astype(np.float64)
            )
            observed = float(np.max(difference))
            if np.isfinite(observed):
                max_abs = observed
                max_abs_finite = True
        records[name] = {
            "left_dtype": left_value.dtype.str,
            "right_dtype": right_value.dtype.str,
            "dtype_equal": dtype_equal,
            "shape_equal": field_shape_equal,
            "left_finite": left_finite,
            "right_finite": right_finite,
            "exact": exact,
            "finite_max_abs_difference": max_abs,
            "max_abs_difference_is_finite": max_abs_finite,
        }
    fields_exact = bool(records) and all(value["exact"] for value in records.values())
    finite = bool(records) and all(
        value["left_finite"]
        and value["right_finite"]
        and value["max_abs_difference_is_finite"]
        for value in records.values()
    )
    return {
        "passed": names_equal and shape_equal and fields_exact and finite,
        "left_shape": list(left.shape),
        "right_shape": list(right.shape),
        "shape_equal": shape_equal,
        "left_field_order": list(left_names),
        "right_field_order": list(right_names),
        "field_order_equal": names_equal,
        "all_fields_exact": fields_exact,
        "all_values_and_max_abs_differences_finite": finite,
        "fields": records,
    }


def _compare_ply(left_path: Path, right_path: Path) -> dict[str, Any]:
    from plyfile import PlyData

    left = PlyData.read(left_path)
    right = PlyData.read(right_path)
    left_elements = [value.name for value in left.elements]
    right_elements = [value.name for value in right.elements]
    _require(
        "vertex" in left_elements and "vertex" in right_elements, "Ply has no vertex"
    )
    comparison = _compare_structured_arrays(left["vertex"].data, right["vertex"].data)
    file_sha_equal = _sha256_file(left_path) == _sha256_file(right_path)
    result = {
        **comparison,
        "left_elements": left_elements,
        "right_elements": right_elements,
        "element_order_equal": left_elements == right_elements,
        "file_sha256_equal": file_sha_equal,
    }
    result["passed"] = bool(
        comparison["passed"]
        and left_elements == right_elements
        and left_elements == ["vertex"]
    )
    return result


def _child_environment(cuda_device: int, temporary: Path) -> dict[str, str]:
    return {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": str(cuda_device),
        "HF_HUB_OFFLINE": "1",
        "HOME": "/home/florianpfaff",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "LOGNAME": "florianpfaff",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYNPUT_BACKEND": "dummy",
        "PYOPENGL_PLATFORM": "egl",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
        "TMPDIR": os.fspath(temporary),
        "TRANSFORMERS_OFFLINE": "1",
        "USER": "florianpfaff",
        "WANDB_MODE": "disabled",
    }


def _invoke_child(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    log_path: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    _require(not os.path.lexists(log_path), "child log already exists")
    timed_out = False
    timeout_error: dict[str, Any] | None = None
    with log_path.open("xb") as log:
        try:
            result = subprocess.run(
                list(command),
                check=False,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=dict(environment),
                timeout=timeout_seconds,
            )
            return_code: int | None = result.returncode
        except subprocess.TimeoutExpired as error:
            timed_out = True
            return_code = None
            timeout_error = {
                "type": type(error).__name__,
                "message": str(error),
                "timeout_seconds": timeout_seconds,
            }
        log.flush()
        os.fsync(log.fileno())
    return {
        "command": list(command),
        "return_code": return_code,
        "timed_out": timed_out,
        "timeout_error": timeout_error,
        "log": _bound_file(log_path, label="qualification child log"),
    }


def _load_optional_child_evidence(
    path: Path, *, label: str
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    try:
        evidence = _load_signed_json(path, label=label)
    except (OSError, ValueError) as error:
        return None, {
            "loaded_and_signature_valid": False,
            "error": {"type": type(error).__name__, "message": str(error)},
        }
    return evidence, {"loaded_and_signature_valid": True, "error": None}


def _compare_optional_fit_outputs(
    original: Mapping[str, Any] | None,
    wrapped: Mapping[str, Any] | None,
) -> dict[str, Any]:
    integrity_predicates = {
        "original_binding_matches_child_evidence_before_parse": False,
        "wrapped_binding_matches_child_evidence_before_parse": False,
        "original_binding_stable_across_parse": False,
        "wrapped_binding_stable_across_parse": False,
    }
    try:
        _require(original is not None, "original fit evidence is unavailable")
        _require(wrapped is not None, "wrapped fit evidence is unavailable")
        original_output = original.get("output")
        wrapped_output = wrapped.get("output")
        _require(isinstance(original_output, Mapping), "original output is unavailable")
        _require(isinstance(wrapped_output, Mapping), "wrapped output is unavailable")
        original_path = original_output.get("path")
        wrapped_path = wrapped_output.get("path")
        _require(isinstance(original_path, str), "original output path is unavailable")
        _require(isinstance(wrapped_path, str), "wrapped output path is unavailable")
        original_before = _bound_file(
            original_path, label="original fit output before comparison"
        )
        wrapped_before = _bound_file(
            wrapped_path, label="wrapped fit output before comparison"
        )
        integrity_predicates["original_binding_matches_child_evidence_before_parse"] = (
            original_before == dict(original_output)
        )
        integrity_predicates["wrapped_binding_matches_child_evidence_before_parse"] = (
            wrapped_before == dict(wrapped_output)
        )
        _require(
            integrity_predicates[
                "original_binding_matches_child_evidence_before_parse"
            ],
            "original output binding differs from child evidence",
        )
        _require(
            integrity_predicates["wrapped_binding_matches_child_evidence_before_parse"],
            "wrapped output binding differs from child evidence",
        )
        comparison = _compare_ply(Path(original_path), Path(wrapped_path))
        original_after = _bound_file(
            original_path, label="original fit output after comparison"
        )
        wrapped_after = _bound_file(
            wrapped_path, label="wrapped fit output after comparison"
        )
        integrity_predicates["original_binding_stable_across_parse"] = (
            original_after == original_before
        )
        integrity_predicates["wrapped_binding_stable_across_parse"] = (
            wrapped_after == wrapped_before
        )
        comparison["output_integrity_predicates"] = integrity_predicates
        comparison["passed"] = bool(
            comparison.get("passed") and all(integrity_predicates.values())
        )
        return comparison
    except Exception as error:
        return {
            "passed": False,
            "output_integrity_predicates": integrity_predicates,
            "error": {"type": type(error).__name__, "message": str(error)},
        }


def _add_common_child_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--deform360-repo", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run the non-held qualification")
    run.add_argument("--code-root", type=Path, required=True)
    run.add_argument("--python", type=Path, default=PINNED_PYTHON)
    run.add_argument("--deform360-repo", type=Path, default=PINNED_DEFORM360)
    run.add_argument("--dataset", type=Path, default=DEFAULT_PUBLIC_DEV_DATASET)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--phase", choices=("all", "ab", "soak"), default="all")
    run.add_argument("--cuda-device", type=int, choices=(0, 1), default=1)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--ab-iterations", type=int, default=AB_ITERATIONS)
    run.add_argument("--soak-fit-count", type=int, default=SOAK_FIT_COUNT)
    run.add_argument("--soak-iterations", type=int, default=SOAK_ITERATIONS)
    run.add_argument(
        "--first-fit-fd-growth-limit", type=int, default=FIRST_FIT_FD_GROWTH_LIMIT
    )
    run.add_argument(
        "--steady-fd-growth-limit", type=int, default=STEADY_FD_GROWTH_LIMIT
    )
    run.add_argument(
        "--steady-task-growth-limit", type=int, default=STEADY_TASK_GROWTH_LIMIT
    )
    run.add_argument("--fit-timeout-seconds", type=int, default=3600)
    run.add_argument("--soak-timeout-seconds", type=int, default=86400)

    fit = subparsers.add_parser("_fit-child")
    _add_common_child_arguments(fit)
    fit.add_argument("--variant", choices=("original", "wrapped"), required=True)

    soak = subparsers.add_parser("_soak-child")
    _add_common_child_arguments(soak)
    soak.add_argument("--fit-count", type=int, required=True)
    soak.add_argument("--first-fd-growth-limit", type=int, required=True)
    soak.add_argument("--steady-fd-growth-limit", type=int, required=True)
    soak.add_argument("--steady-task-growth-limit", type=int, required=True)
    return parser


def _run(arguments: argparse.Namespace) -> int:
    code = _assert_nonheld_path(
        arguments.code_root, label="qualification code root", must_exist=True
    )
    deform360 = _assert_nonheld_path(
        arguments.deform360_repo,
        label="qualification Deform360 root",
        must_exist=True,
    )
    dataset = _assert_nonheld_path(
        arguments.dataset, label="qualification source dataset", must_exist=True
    )
    output = _assert_nonheld_path(
        arguments.output_dir, label="qualification output", must_exist=False
    )
    canonical_run_parameters = _assert_canonical_run_parameters(arguments, dataset)
    python = _absolute(arguments.python)
    _require(python == PINNED_PYTHON, "qualification Python is not the pinned runtime")
    parent_python_process = _current_python_process_binding()
    python_binding = _python_runtime_binding(python)
    _require(deform360 == PINNED_DEFORM360, "Deform360 runtime is not pinned")
    _require(socket.gethostname() == "workstation2", "qualification host is not pinned")
    for protected, label in (
        (code, "code root"),
        (deform360, "Deform360 root"),
        (dataset, "source dataset"),
    ):
        _require(
            protected not in output.parents,
            f"qualification output is nested inside the {label}",
        )
    _require(arguments.ab_iterations > 0, "A/B iterations must be positive")
    _require(arguments.soak_fit_count > 0, "soak fit count must be positive")
    _require(arguments.soak_iterations > 0, "soak iterations must be positive")
    _require(
        min(
            arguments.first_fit_fd_growth_limit,
            arguments.steady_fd_growth_limit,
            arguments.steady_task_growth_limit,
        )
        >= 0,
        "resource growth limits must be nonnegative",
    )
    _require(not os.path.lexists(output), "qualification output already exists")
    script = (code / RELATIVE_QUALIFICATION_SOURCE).resolve(strict=True)
    _require(
        script == Path(__file__).resolve(strict=True),
        "qualification operator is outside the clean code root",
    )
    code_binding = _git_binding(code)
    deform360_binding = _git_binding(deform360, expected_head=PINNED_DEFORM360_REVISION)
    output.mkdir(parents=True, exist_ok=False)
    temporary = output / "tmp"
    temporary.mkdir()
    environment = _child_environment(arguments.cuda_device, temporary)
    invocations: dict[str, Any] = {}
    datasets: dict[str, Any] = {}
    ab: dict[str, Any] | None = None
    soak: dict[str, Any] | None = None
    soak_evidence_validation: dict[str, Any] | None = None

    if arguments.phase in ("all", "ab"):
        children: dict[str, dict[str, Any] | None] = {}
        child_evidence_validation: dict[str, Any] = {}
        for variant in ("original", "wrapped"):
            root = output / "ab" / variant
            root.mkdir(parents=True)
            materialized = root / "dataset"
            export = root / "export"
            export.mkdir()
            datasets[f"ab_{variant}"] = _materialize_dataset(dataset, materialized)
            result_path = root / "fit-evidence.json"
            command = [
                os.fspath(python),
                "-I",
                "-B",
                os.fspath(script),
                "_fit-child",
                "--code-root",
                os.fspath(code),
                "--deform360-repo",
                os.fspath(deform360),
                "--dataset",
                os.fspath(materialized),
                "--output-dir",
                os.fspath(export),
                "--result",
                os.fspath(result_path),
                "--iterations",
                str(arguments.ab_iterations),
                "--seed",
                str(arguments.seed),
                "--variant",
                variant,
            ]
            invocations[f"ab_{variant}"] = _invoke_child(
                command,
                environment=environment,
                log_path=root / "fit.log",
                timeout_seconds=arguments.fit_timeout_seconds,
            )
            children[variant], child_evidence_validation[variant] = (
                _load_optional_child_evidence(
                    result_path, label=f"{variant} fit child evidence"
                )
            )
        comparison = _compare_optional_fit_outputs(
            children["original"], children["wrapped"]
        )
        ab_predicates = {
            "original_child_evidence_valid": child_evidence_validation["original"][
                "loaded_and_signature_valid"
            ],
            "wrapped_child_evidence_valid": child_evidence_validation["wrapped"][
                "loaded_and_signature_valid"
            ],
            "original_child_passed": children["original"] is not None
            and children["original"].get("passed") is True,
            "wrapped_child_passed": children["wrapped"] is not None
            and children["wrapped"].get("passed") is True,
            "isolated_child_processes": True,
            "same_seed_and_iterations": (
                children["original"] is not None
                and children["wrapped"] is not None
                and children["original"].get("parameters")
                == children["wrapped"].get("parameters")
            ),
            "materialized_dataset_content_equal": (
                _materialized_dataset_identity(datasets["ab_original"])
                == _materialized_dataset_identity(datasets["ab_wrapped"])
            ),
            "structured_ply_fields_exact_and_finite": comparison["passed"],
        }
        ab = {
            "passed": all(ab_predicates.values()),
            "predicates": ab_predicates,
            "children": children,
            "child_evidence_validation": child_evidence_validation,
            "ply_comparison": comparison,
        }

    if arguments.phase in ("all", "soak"):
        root = output / "soak"
        root.mkdir(parents=True)
        materialized = root / "dataset"
        export = root / "export"
        export.mkdir()
        datasets["soak"] = _materialize_dataset(dataset, materialized)
        result_path = root / "soak-evidence.json"
        command = [
            os.fspath(python),
            "-I",
            "-B",
            os.fspath(script),
            "_soak-child",
            "--code-root",
            os.fspath(code),
            "--deform360-repo",
            os.fspath(deform360),
            "--dataset",
            os.fspath(materialized),
            "--output-dir",
            os.fspath(export),
            "--result",
            os.fspath(result_path),
            "--iterations",
            str(arguments.soak_iterations),
            "--seed",
            str(arguments.seed),
            "--fit-count",
            str(arguments.soak_fit_count),
            "--first-fd-growth-limit",
            str(arguments.first_fit_fd_growth_limit),
            "--steady-fd-growth-limit",
            str(arguments.steady_fd_growth_limit),
            "--steady-task-growth-limit",
            str(arguments.steady_task_growth_limit),
        ]
        invocations["soak"] = _invoke_child(
            command,
            environment=environment,
            log_path=root / "soak.log",
            timeout_seconds=arguments.soak_timeout_seconds,
        )
        soak, soak_evidence_validation = _load_optional_child_evidence(
            result_path, label="soak child evidence"
        )

    code_binding_after = _git_binding(code)
    deform360_binding_after = _git_binding(
        deform360, expected_head=PINNED_DEFORM360_REVISION
    )
    python_binding_after = _python_runtime_binding(python)
    materialized_inputs_stable = all(
        _materialized_inputs_stable(value) for value in datasets.values()
    )
    source_inputs_stable = all(
        _source_inputs_stable(value) for value in datasets.values()
    )
    source_materialized_content_equal = all(
        value.get("referenced_source_materialized_content_equal") is True
        and value.get("referenced_source_content")
        == value.get("referenced_materialized_content")
        for value in datasets.values()
    )
    predicates = {
        "formal_held_paths_rejected": True,
        "code_checkout_clean": code_binding["clean"],
        "deform360_checkout_clean_and_pinned": (
            deform360_binding["clean"]
            and deform360_binding["head"] == PINNED_DEFORM360_REVISION
        ),
        "code_checkout_stable_across_qualification": (
            code_binding_after == code_binding
        ),
        "deform360_checkout_stable_across_qualification": (
            deform360_binding_after == deform360_binding
        ),
        "python_runtime_stable_across_qualification": (
            python_binding_after == python_binding
        ),
        "canonical_parent_process_is_pinned": True,
        "materialized_inputs_stable_across_qualification": (materialized_inputs_stable),
        "source_inputs_stable_across_qualification": source_inputs_stable,
        "referenced_source_materialized_content_equal": (
            source_materialized_content_equal
        ),
        "ab_passed_when_requested": arguments.phase not in ("all", "ab")
        or (ab is not None and ab["passed"]),
        "soak_passed_when_requested": arguments.phase not in ("all", "soak")
        or (
            soak is not None
            and soak_evidence_validation is not None
            and soak_evidence_validation["loaded_and_signature_valid"]
            and soak.get("passed") is True
        ),
        "all_children_exited_zero": all(
            value["return_code"] == 0 for value in invocations.values()
        ),
    }
    evidence = _signed(
        {
            "schema_version": 1,
            "artifact_kind": "Deform360ResourceLifecycleQualificationEvidence",
            "qualification_id": QUALIFICATION_ID,
            "passed": all(predicates.values()),
            "host": socket.gethostname(),
            "phase": arguments.phase,
            "canonical_run_parameters": canonical_run_parameters,
            "parameters": {
                "cuda_device": arguments.cuda_device,
                "seed": arguments.seed,
                "ab_iterations": arguments.ab_iterations,
                "soak_fit_count": arguments.soak_fit_count,
                "soak_iterations": arguments.soak_iterations,
                "first_fit_fd_growth_limit": arguments.first_fit_fd_growth_limit,
                "steady_fd_growth_limit": arguments.steady_fd_growth_limit,
                "steady_task_growth_limit": arguments.steady_task_growth_limit,
            },
            "runtime_bindings": {
                "python_path": os.fspath(python),
                "python": python_binding,
                "python_after": python_binding_after,
                "parent_python_process": parent_python_process,
                "code": code_binding,
                "code_after": code_binding_after,
                "deform360": deform360_binding,
                "deform360_after": deform360_binding_after,
                "qualification_source": _bound_file(
                    script, label="qualification operator"
                ),
                "wrapper_source": _bound_file(
                    code / RELATIVE_WRAPPER_SOURCE, label="resource wrapper source"
                ),
            },
            "source_dataset": os.fspath(dataset),
            "materialized_datasets": datasets,
            "invocations": invocations,
            "ab": ab,
            "soak": soak,
            "soak_evidence_validation": soak_evidence_validation,
            "predicates": predicates,
            "information_boundary": {
                "formal_held_path_accepted": False,
                "formal_target_or_outcome_array_read": False,
                "development_dataset_only": True,
                "unreferenced_source_outputs_copied": False,
                "rlimit_nofile_changed": False,
            },
        }
    )
    evidence_path = _write_new_json(
        output / "resource-lifecycle-qualification.json", evidence
    )
    print(
        json.dumps(
            {
                "passed": evidence["passed"],
                "evidence": os.fspath(evidence_path),
                "artifact_sha256": evidence["artifact_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if evidence["passed"] else 2


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "run":
        return _run(arguments)
    if arguments.command == "_fit-child":
        return _child_fit(arguments)
    if arguments.command == "_soak-child":
        return _child_soak(arguments)
    raise AssertionError(f"unknown command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
