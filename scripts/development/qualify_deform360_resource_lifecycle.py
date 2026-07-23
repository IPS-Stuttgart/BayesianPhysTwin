#!/usr/bin/env python3
"""Qualify bounded per-fit Nerfstudio resources on non-held data.

This development operator has three ordered gates:

* five isolated original fits and five isolated wrapped fits produce a fresh
  paired A/B cohort on physical GPU 1;
* the frozen equivalence analyzer checks exact equality first and the
  predeclared distributional envelope second; and
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
import importlib
import json
import os
from pathlib import Path
import random
import resource
import socket
import stat
import subprocess
import sys
import traceback
from typing import Any, Mapping, Sequence


QUALIFICATION_ID = "deform360-nerfstudio-resource-lifecycle-qualification-v2"
QUALIFICATION_KIND = "Deform360ResourceLifecycleQualificationEvidenceV2"
QUALIFICATION_ATTEMPT_KIND = "Deform360ResourceLifecycleQualificationAttemptV2"
GENERATOR_PROFILE = "same-as-analyzer"
PHYSICAL_GPU_INDEX = 1
AB_REPEAT_COUNT = 5
QUALIFICATION_BASE = Path("/mnt/corsair/florianpfaff")
FIT_TIMEOUT_SECONDS = 3_600
ANALYZER_TIMEOUT_SECONDS = 86_400
SOAK_TIMEOUT_SECONDS = 86_400
ANALYSIS_ID = "deform360-resource-lifecycle-distributional-equivalence-v1"
ANALYSIS_MANIFEST_KIND = "Deform360ResourceLifecycleRepeatManifestV1"
ANALYSIS_RESULT_KIND = "Deform360ResourceLifecycleDistributionalEquivalenceV1"
ANALYZER_NO_GO_INTERPRETATION = (
    "admission-inconclusive; the frozen analyzer did not admit this single fresh "
    "cohort, which is not proof of wrapper inequivalence"
)
FROZEN_ANALYZER_SOURCE_SHA256 = (
    "43056e39ff7ea5f760f18420784db0edbb75523031dba7f3a19eca0c6951c128"
)
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
RELATIVE_ANALYZER_SOURCE = Path(
    "scripts/development/analyze_deform360_resource_lifecycle_equivalence.py"
)
AB_ITERATIONS = 250
SOAK_FIT_COUNT = 243
SOAK_ITERATIONS = 1
SOAK_TRAINER_REINITIALIZATION_INTERVAL = 81
V8_PYCACHE_PREFIX = "/nonexistent/bpt-held-v8-pycache"
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


def _root_consumption_policy() -> dict[str, Any]:
    return {
        "canonical_root_consumed_at_creation": True,
        "same_root_retry_permitted": False,
        "same_revision_retry_permitted": False,
        "in_place_reuse_permitted": False,
        "incomplete_root_sealable_or_replayable": False,
        "technical_fix_in_later_disclosed_revision_may_use_new_root": True,
        "replacement_requires_different_canonical_root": True,
        "replacement_may_change_frozen_analyzer_or_numerical_gate": False,
    }


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
        if os.path.lexists(output):
            _remove_owned_tree(
                output,
                parent=output.parent,
                label="failed materialized dataset",
            )
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
        "cuda_device": PHYSICAL_GPU_INDEX,
        "seed": 0,
        "ab_iterations": AB_ITERATIONS,
        "ab_repeat_count": AB_REPEAT_COUNT,
        "soak_fit_count": SOAK_FIT_COUNT,
        "soak_iterations": SOAK_ITERATIONS,
        "first_fit_fd_growth_limit": FIRST_FIT_FD_GROWTH_LIMIT,
        "steady_fd_growth_limit": STEADY_FD_GROWTH_LIMIT,
        "steady_task_growth_limit": STEADY_TASK_GROWTH_LIMIT,
        "fit_timeout_seconds": FIT_TIMEOUT_SECONDS,
        "analyzer_timeout_seconds": ANALYZER_TIMEOUT_SECONDS,
        "soak_timeout_seconds": SOAK_TIMEOUT_SECONDS,
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
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "HOME": "/home/florianpfaff",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
    }

    def git(*arguments: str) -> str:
        result = subprocess.run(
            [
                "/usr/bin/git",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.untrackedCache=false",
                "-C",
                repository,
                *arguments,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
            env=environment,
        )
        _require(result.returncode == 0, f"Git command failed: {' '.join(arguments)}")
        return result.stdout.strip()

    replacement_refs = git(
        "for-each-ref",
        "--format=%(refname)",
        "refs/replace",
    )
    _require(
        not replacement_refs,
        f"qualification repository has replacement refs: {repository}",
    )
    git_directory_raw = git("rev-parse", "--absolute-git-dir")
    _require(git_directory_raw, f"qualification Git directory is absent: {repository}")
    git_directory = Path(git_directory_raw)
    _require(
        git_directory.is_absolute(),
        f"qualification Git directory is not absolute: {repository}",
    )
    git_directory = _assert_nonheld_path(
        git_directory,
        label="qualification Git directory",
        must_exist=True,
    )
    _require(
        git_directory.is_dir() and not git_directory.is_symlink(),
        f"qualification Git directory is not a real directory: {repository}",
    )
    _require(
        not os.path.lexists(git_directory / "info/grafts"),
        f"qualification repository has a grafts file: {repository}",
    )
    tracked_index_records = [
        record for record in git("ls-files", "-v", "-z").split("\0") if record
    ]
    _require(
        tracked_index_records
        and all(record.startswith("H ") for record in tracked_index_records),
        f"qualification repository has non-ordinary tracked index entries: "
        f"{repository}",
    )

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


def _gsplat_smoke_artifact_sha256(value: Mapping[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("artifact_sha256", None)
    return hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _load_and_smoke_gsplat_runtime(code_root: Path) -> dict[str, Any]:
    code_source = code_root / "src"
    source_value = os.fspath(code_source)
    if source_value not in sys.path:
        sys.path.insert(0, source_value)
    runtime = importlib.import_module(
        "bayesian_phystwin.deform360_held_v8_gsplat_runtime"
    )
    expected = (
        code_source / "bayesian_phystwin" / "deform360_held_v8_gsplat_runtime.py"
    ).resolve(strict=True)
    runtime_file = Path(runtime.__file__).resolve(strict=True)
    _require(runtime_file == expected, "gsplat runtime smoke escaped the code root")
    smoke_function = getattr(runtime, "load_and_smoke_gsplat_runtime", None)
    _require(callable(smoke_function), "gsplat runtime smoke entry point is absent")
    smoke = smoke_function()
    _require(isinstance(smoke, Mapping), "gsplat runtime smoke evidence is not a map")
    evidence = dict(smoke)
    artifact_sha256 = evidence.get("artifact_sha256")
    _require(
        isinstance(artifact_sha256, str)
        and len(artifact_sha256) == 64
        and artifact_sha256 == _gsplat_smoke_artifact_sha256(evidence),
        "gsplat runtime smoke evidence signature is invalid",
    )
    _require(
        evidence.get("artifact_kind") == "Deform360HeldGsplatRuntimeSmokeV1",
        "gsplat runtime smoke evidence kind changed",
    )
    _require(
        evidence.get("extension_loaded_and_retained") is True,
        "gsplat runtime supplement was not retained",
    )
    _require(
        evidence.get("target_or_outcome_path_accessed") is False,
        "gsplat runtime smoke crossed the information boundary",
    )
    return {
        "adapter_source": _bound_file(
            expected, label="held-v8 gsplat runtime adapter source"
        ),
        "evidence": evidence,
        "evidence_artifact_sha256": artifact_sha256,
    }


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
        gsplat_runtime_smoke = _load_and_smoke_gsplat_runtime(code)
        runtime = _seed_runtime(arguments.seed)
        trainer_type, wrapper_type, writer, profiler = _import_trainers(code, deform360)
        delegate = trainer_type()
        trainer = wrapper_type(delegate) if arguments.variant == "wrapped" else delegate
        before_globals = _global_state_snapshot(writer, profiler)
        before = _process_boundary()
        output_filename = "splat.ply"
        produced = _absolute(
            Path(trainer.train(dataset, output, output_filename, arguments.iterations))
        )
        expected_output = _absolute(output / output_filename)
        _require(produced == expected_output, "fit output escaped")
        produced_state = os.lstat(produced)
        _require(
            stat.S_ISREG(produced_state.st_mode)
            and not stat.S_ISLNK(produced_state.st_mode)
            and produced_state.st_nlink == 1,
            "fit output is linked or not a regular file",
        )
        after = _process_boundary()
        after_globals = _global_state_snapshot(writer, profiler)
        globals_restored = before_globals == after_globals
        predicates = {
            "output_created": True,
            "wrapped_fit_requires_global_restoration": (
                arguments.variant != "wrapped" or globals_restored
            ),
            "rlimit_nofile_soft_is_1024": before["rlimit_nofile_soft"] == 1024,
            "rlimit_nofile_unchanged": (
                after["rlimit_nofile_soft"] == before["rlimit_nofile_soft"]
                and after["rlimit_nofile_hard"] == before["rlimit_nofile_hard"]
            ),
            "gsplat_runtime_smoke_validated_and_retained": True,
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
                "gsplat_runtime_smoke": gsplat_runtime_smoke,
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
        gsplat_runtime_smoke = _load_and_smoke_gsplat_runtime(code)
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
            produced = _absolute(
                Path(
                    trainer.train(
                        dataset,
                        output,
                        output_filename,
                        arguments.iterations,
                    )
                )
            )
            _require(
                produced == _absolute(output / output_filename),
                "soak output escaped",
            )
            produced_state = os.lstat(produced)
            output_created = bool(
                stat.S_ISREG(produced_state.st_mode)
                and not stat.S_ISLNK(produced_state.st_mode)
                and produced_state.st_nlink == 1
            )
            _require(output_created, "soak output Ply is not a regular file")
            output_size = produced_state.st_size
            output_cleanup = _remove_owned_file(
                produced,
                parent=output,
                label=f"soak fit {index} output Ply",
            )
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
            dataset_outputs_cleanup = _remove_owned_tree(
                dataset_outputs,
                parent=dataset,
                label=f"soak fit {index} generated outputs",
            )
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
                    "cleanup": {
                        "output_ply": output_cleanup,
                        "dataset_outputs": dataset_outputs_cleanup,
                    },
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
                "gsplat_runtime_smoke": gsplat_runtime_smoke,
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
        "CUDA_MODULE_LOADING": "LAZY",
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
        "PYTHONPYCACHEPREFIX": V8_PYCACHE_PREFIX,
        "PYTHONSAFEPATH": "1",
        "TMPDIR": os.fspath(temporary),
        "TRANSFORMERS_OFFLINE": "1",
        "USER": "florianpfaff",
        "WANDB_MODE": "disabled",
    }


def _child_python_argv_prefix(python: Path, script: Path) -> list[str]:
    return [
        os.fspath(python),
        "-I",
        "-B",
        "-X",
        f"pycache_prefix={V8_PYCACHE_PREFIX}",
        os.fspath(script),
    ]


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
        "environment": dict(environment),
        "return_code": return_code,
        "timed_out": timed_out,
        "timeout_error": timeout_error,
        "timeout_seconds": timeout_seconds,
        "log": _bound_file(log_path, label="qualification child log"),
    }


_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
)
_REGULAR_FILE_OPEN_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


def _object_identity(value: os.stat_result) -> tuple[int, int, int]:
    """Return the fields that remain stable while an opened tree is emptied."""

    return value.st_dev, value.st_ino, stat.S_IFMT(value.st_mode)


def _open_verified_directory_path(path: Path, *, label: str) -> int:
    """Open a directory and prove the descriptor names the lstat'ed object."""

    try:
        before = os.lstat(path)
    except OSError as error:
        raise ValueError(f"{label} is unavailable") from error
    _require(
        stat.S_ISDIR(before.st_mode) and not stat.S_ISLNK(before.st_mode),
        f"{label} is not a real directory",
    )
    descriptor = os.open(path, _DIRECTORY_OPEN_FLAGS)
    try:
        opened = os.fstat(descriptor)
        current = os.lstat(path)
        _require(
            _stable_identity(opened)
            == _stable_identity(before)
            == _stable_identity(current),
            f"{label} changed while opening",
        )
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _open_verified_directory_at(
    parent_descriptor: int,
    name: str,
    expected: os.stat_result,
    *,
    label: str,
) -> int:
    """Open one child directory relative to an already verified parent."""

    _require(
        stat.S_ISDIR(expected.st_mode) and not stat.S_ISLNK(expected.st_mode),
        f"{label} contains a linked directory",
    )
    descriptor = os.open(
        name,
        _DIRECTORY_OPEN_FLAGS,
        dir_fd=parent_descriptor,
    )
    try:
        opened = os.fstat(descriptor)
        current = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        _require(
            _stable_identity(opened)
            == _stable_identity(expected)
            == _stable_identity(current),
            f"{label} changed while opening a directory",
        )
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _open_verified_regular_at(
    parent_descriptor: int,
    name: str,
    expected: os.stat_result,
    *,
    label: str,
) -> int:
    """Open one single-link regular file relative to a verified directory."""

    _require(
        stat.S_ISREG(expected.st_mode)
        and not stat.S_ISLNK(expected.st_mode)
        and expected.st_nlink == 1,
        f"{label} is linked or not a regular file",
    )
    descriptor = os.open(
        name,
        _REGULAR_FILE_OPEN_FLAGS,
        dir_fd=parent_descriptor,
    )
    try:
        opened = os.fstat(descriptor)
        current = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        _require(
            _stable_identity(opened)
            == _stable_identity(expected)
            == _stable_identity(current),
            f"{label} changed while opening",
        )
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _bound_regular_descriptor(
    descriptor: int,
    *,
    parent_descriptor: int,
    name: str,
    path: Path,
    expected: os.stat_result,
    label: str,
) -> dict[str, Any]:
    """Hash an opened file and retain its directory-entry identity proof."""

    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            break
        digest.update(block)
    opened = os.fstat(descriptor)
    current = os.stat(
        name,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    _require(
        _stable_identity(opened)
        == _stable_identity(expected)
        == _stable_identity(current),
        f"{label} changed while reading",
    )
    return {
        "path": os.fspath(path),
        "sha256": digest.hexdigest(),
        "size_bytes": opened.st_size,
        "mode_octal": f"{stat.S_IMODE(opened.st_mode):04o}",
    }


def _require_directory_path_matches_descriptor(
    path: Path,
    descriptor: int,
    *,
    label: str,
) -> None:
    """Detect an ancestor/parent substitution without directing deletion by path."""

    try:
        current = os.lstat(path)
    except OSError as error:
        raise ValueError(f"{label} changed during cleanup") from error
    opened = os.fstat(descriptor)
    _require(
        stat.S_ISDIR(current.st_mode)
        and not stat.S_ISLNK(current.st_mode)
        and _object_identity(current) == _object_identity(opened),
        f"{label} changed during cleanup",
    )


def _entry_absent_at(parent_descriptor: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return True
    return False


def _inventory_directory_descriptor(
    descriptor: int,
    *,
    root: Path,
    relative: tuple[str, ...],
    expected: os.stat_result,
    label: str,
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], int]:
    """Inventory a directory using only descendants of its opened descriptor."""

    _require(
        _stable_identity(os.fstat(descriptor)) == _stable_identity(expected),
        f"{label} changed while inventorying",
    )
    entries: dict[str, Any] = {}
    regular_file_bytes = 0
    for name in sorted(os.listdir(descriptor)):
        observed = os.stat(
            name,
            dir_fd=descriptor,
            follow_symlinks=False,
        )
        child_relative = (*relative, name)
        relative_path = Path(*child_relative).as_posix()
        if stat.S_ISDIR(observed.st_mode) and not stat.S_ISLNK(observed.st_mode):
            child_descriptor = _open_verified_directory_at(
                descriptor,
                name,
                observed,
                label=label,
            )
            try:
                child_entries, child_bytes = _inventory_directory_descriptor(
                    child_descriptor,
                    root=root,
                    relative=child_relative,
                    expected=observed,
                    label=label,
                    rows=rows,
                )
            finally:
                os.close(child_descriptor)
            entries[name] = {
                "type": "directory",
                "identity": _stable_identity(observed),
                "entries": child_entries,
            }
            regular_file_bytes += child_bytes
            rows.append(
                {
                    "path": relative_path,
                    "type": "directory",
                    "mode_octal": f"{stat.S_IMODE(observed.st_mode):04o}",
                }
            )
            continue
        _require(
            stat.S_ISREG(observed.st_mode)
            and not stat.S_ISLNK(observed.st_mode)
            and observed.st_nlink == 1,
            f"{label} contains a linked or special file",
        )
        child_descriptor = _open_verified_regular_at(
            descriptor,
            name,
            observed,
            label=f"{label} file",
        )
        try:
            binding = _bound_regular_descriptor(
                child_descriptor,
                parent_descriptor=descriptor,
                name=name,
                path=root.joinpath(*child_relative),
                expected=observed,
                label=f"{label} file",
            )
        finally:
            os.close(child_descriptor)
        entries[name] = {
            "type": "file",
            "identity": _stable_identity(observed),
        }
        regular_file_bytes += int(binding["size_bytes"])
        rows.append(
            {
                "path": relative_path,
                "type": "file",
                "mode_octal": binding["mode_octal"],
                "size_bytes": binding["size_bytes"],
                "sha256": binding["sha256"],
            }
        )
    _require(
        _stable_identity(os.fstat(descriptor)) == _stable_identity(expected),
        f"{label} changed while inventorying",
    )
    return entries, regular_file_bytes


def _open_owned_tree(
    root: Path,
    *,
    parent: Path,
    label: str,
) -> tuple[int, int, os.stat_result]:
    _require(root.parent == parent, f"{label} escaped its bounded parent")
    parent_descriptor = _open_verified_directory_path(
        parent,
        label=f"{label} bounded parent",
    )
    try:
        root_state = os.stat(
            root.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        _require(
            stat.S_ISDIR(root_state.st_mode) and not stat.S_ISLNK(root_state.st_mode),
            f"{label} is not a real directory",
        )
        root_descriptor = _open_verified_directory_at(
            parent_descriptor,
            root.name,
            root_state,
            label=label,
        )
    except BaseException:
        os.close(parent_descriptor)
        raise
    return parent_descriptor, root_descriptor, root_state


def _owned_tree_snapshot(
    root: Path,
    *,
    root_descriptor: int,
    root_state: os.stat_result,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    entries, regular_file_bytes = _inventory_directory_descriptor(
        root_descriptor,
        root=root,
        relative=(),
        expected=root_state,
        label=label,
        rows=rows,
    )
    rows.sort(key=lambda value: str(value["path"]))
    inventory = {
        "root": os.fspath(root),
        "entry_count": len(rows),
        "regular_file_bytes": regular_file_bytes,
        "inventory_sha256": hashlib.sha256(
            _canonical_bytes({"rows": rows})
        ).hexdigest(),
    }
    snapshot = {
        "type": "directory",
        "identity": _stable_identity(root_state),
        "entries": entries,
    }
    return inventory, snapshot


def _delete_inventory_descriptor(
    descriptor: int,
    snapshot: Mapping[str, Any],
    *,
    label: str,
) -> None:
    """Delete a frozen inventory relative to its still-open root descriptor."""

    _require(
        _stable_identity(os.fstat(descriptor)) == tuple(snapshot["identity"]),
        f"{label} changed before cleanup",
    )
    entries = snapshot["entries"]
    _require(isinstance(entries, Mapping), f"{label} inventory is invalid")
    _require(
        sorted(os.listdir(descriptor)) == sorted(entries),
        f"{label} changed before cleanup",
    )
    for name in sorted(entries):
        child = entries[name]
        _require(isinstance(child, Mapping), f"{label} inventory is invalid")
        expected_identity = tuple(child["identity"])
        observed = os.stat(
            name,
            dir_fd=descriptor,
            follow_symlinks=False,
        )
        _require(
            _stable_identity(observed) == expected_identity,
            f"{label} changed before cleanup",
        )
        if child.get("type") == "file":
            child_descriptor = _open_verified_regular_at(
                descriptor,
                name,
                observed,
                label=f"{label} file",
            )
            try:
                current = os.stat(
                    name,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                _require(
                    _stable_identity(current) == expected_identity,
                    f"{label} changed before cleanup",
                )
                os.unlink(name, dir_fd=descriptor)
                _require(
                    _entry_absent_at(descriptor, name),
                    f"{label} file remains after cleanup",
                )
                removed = os.fstat(child_descriptor)
                _require(
                    _object_identity(removed) == _object_identity(observed)
                    and removed.st_nlink == 0,
                    f"{label} changed during cleanup",
                )
            finally:
                os.close(child_descriptor)
            continue
        _require(child.get("type") == "directory", f"{label} inventory is invalid")
        child_descriptor = _open_verified_directory_at(
            descriptor,
            name,
            observed,
            label=label,
        )
        try:
            _delete_inventory_descriptor(child_descriptor, child, label=label)
            current = os.stat(
                name,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
            _require(
                _object_identity(current)
                == _object_identity(os.fstat(child_descriptor)),
                f"{label} changed during cleanup",
            )
            _require(not os.listdir(child_descriptor), f"{label} is not empty")
            os.rmdir(name, dir_fd=descriptor)
            _require(
                _entry_absent_at(descriptor, name),
                f"{label} directory remains after cleanup",
            )
        finally:
            os.close(child_descriptor)


def _remove_owned_file(path: Path, *, parent: Path, label: str) -> dict[str, Any]:
    """Remove one exact generated file through its verified parent descriptor."""

    source = _absolute(path)
    expected_parent = _absolute(parent)
    _require(source.parent == expected_parent, f"{label} escaped its bounded parent")
    parent_descriptor = _open_verified_directory_path(
        expected_parent,
        label=f"{label} bounded parent",
    )
    try:
        observed = os.stat(
            source.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        child_descriptor = _open_verified_regular_at(
            parent_descriptor,
            source.name,
            observed,
            label=label,
        )
        try:
            binding = _bound_regular_descriptor(
                child_descriptor,
                parent_descriptor=parent_descriptor,
                name=source.name,
                path=source,
                expected=observed,
                label=label,
            )
            _require_directory_path_matches_descriptor(
                expected_parent,
                parent_descriptor,
                label=f"{label} bounded parent",
            )
            os.unlink(source.name, dir_fd=parent_descriptor)
            _require(
                _entry_absent_at(parent_descriptor, source.name),
                f"{label} remains after cleanup",
            )
            removed = os.fstat(child_descriptor)
            _require(
                _object_identity(removed) == _object_identity(observed)
                and removed.st_nlink == 0,
                f"{label} changed during cleanup",
            )
        finally:
            os.close(child_descriptor)
        _require_directory_path_matches_descriptor(
            expected_parent,
            parent_descriptor,
            label=f"{label} bounded parent",
        )
    finally:
        os.close(parent_descriptor)
    return {
        "bounded_parent": os.fspath(expected_parent),
        "pre_cleanup_binding": binding,
        "pre_cleanup_link_count": 1,
        "removed": True,
        "post_cleanup_absent": True,
    }


def _owned_tree_inventory(path: Path, *, parent: Path, label: str) -> dict[str, Any]:
    """Hash an owned tree through verified, no-follow directory descriptors."""

    root = _absolute(path)
    expected_parent = _absolute(parent)
    parent_descriptor, root_descriptor, root_state = _open_owned_tree(
        root,
        parent=expected_parent,
        label=label,
    )
    try:
        inventory, _ = _owned_tree_snapshot(
            root,
            root_descriptor=root_descriptor,
            root_state=root_state,
            label=label,
        )
        _require_directory_path_matches_descriptor(
            expected_parent,
            parent_descriptor,
            label=f"{label} bounded parent",
        )
    finally:
        os.close(root_descriptor)
        os.close(parent_descriptor)
    return inventory


def _remove_owned_tree(path: Path, *, parent: Path, label: str) -> dict[str, Any]:
    """Remove an inventoried tree through verified, no-follow descriptors."""

    root = _absolute(path)
    expected_parent = _absolute(parent)
    parent_descriptor, root_descriptor, root_state = _open_owned_tree(
        root,
        parent=expected_parent,
        label=label,
    )
    try:
        inventory, snapshot = _owned_tree_snapshot(
            root,
            root_descriptor=root_descriptor,
            root_state=root_state,
            label=label,
        )
        _require_directory_path_matches_descriptor(
            expected_parent,
            parent_descriptor,
            label=f"{label} bounded parent",
        )
        _delete_inventory_descriptor(root_descriptor, snapshot, label=label)
        current = os.stat(
            root.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        _require(
            _object_identity(current) == _object_identity(os.fstat(root_descriptor)),
            f"{label} changed during cleanup",
        )
        _require(not os.listdir(root_descriptor), f"{label} is not empty")
        os.rmdir(root.name, dir_fd=parent_descriptor)
        _require(
            _entry_absent_at(parent_descriptor, root.name),
            f"{label} remains after cleanup",
        )
        _require_directory_path_matches_descriptor(
            expected_parent,
            parent_descriptor,
            label=f"{label} bounded parent",
        )
    finally:
        os.close(root_descriptor)
        os.close(parent_descriptor)
    return {
        "bounded_parent": os.fspath(expected_parent),
        "pre_cleanup_inventory": inventory,
        "removed": True,
        "post_cleanup_absent": True,
    }


def _reset_owned_temporary(
    temporary: Path, *, output: Path, label: str
) -> dict[str, Any]:
    cleanup = _remove_owned_tree(temporary, parent=output, label=label)
    temporary.mkdir(mode=0o700)
    _require(
        temporary.parent == output and not any(temporary.iterdir()),
        "qualification temporary directory was not reset",
    )
    return {**cleanup, "recreated_empty": True}


def _validate_fit_child_before_cleanup(
    evidence: Mapping[str, Any],
    *,
    variant: str,
    dataset: Path,
    output: Path,
) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "artifact_kind",
        "qualification_id",
        "variant",
        "passed",
        "parameters",
        "runtime",
        "gsplat_runtime_smoke",
        "dataset",
        "output",
        "resource_boundary",
        "global_state",
        "predicates",
        "formal_held_path_supplied",
        "artifact_sha256",
    }
    _require(set(evidence) == expected_fields, f"{variant} fit evidence fields changed")
    _require(
        evidence.get("schema_version") == 1
        and evidence.get("artifact_kind")
        == "Deform360ResourceLifecycleFitChildEvidence"
        and evidence.get("qualification_id") == QUALIFICATION_ID
        and evidence.get("variant") == variant
        and evidence.get("passed") is True
        and evidence.get("parameters") == {"iterations": AB_ITERATIONS, "seed": 0}
        and evidence.get("dataset") == os.fspath(dataset)
        and evidence.get("formal_held_path_supplied") is False,
        f"{variant} fit evidence identity changed",
    )
    output_record = evidence.get("output")
    _require(isinstance(output_record, Mapping), f"{variant} fit output is absent")
    output_state = os.lstat(output)
    _require(
        output.parent.is_dir()
        and stat.S_ISREG(output_state.st_mode)
        and not stat.S_ISLNK(output_state.st_mode)
        and output_state.st_nlink == 1,
        f"{variant} retained fit output is linked or not regular",
    )
    observed_output = _bound_file(output, label=f"{variant} retained fit output")
    _require(
        dict(output_record) == observed_output,
        f"{variant} retained fit output binding changed",
    )
    predicates = evidence.get("predicates")
    _require(
        isinstance(predicates, Mapping)
        and predicates
        and all(value is True for value in predicates.values()),
        f"{variant} fit predicate failed",
    )
    smoke = evidence.get("gsplat_runtime_smoke")
    _require(isinstance(smoke, Mapping), f"{variant} gsplat smoke is absent")
    smoke_evidence = smoke.get("evidence")
    _require(
        isinstance(smoke_evidence, Mapping)
        and smoke_evidence.get("physical_gpu_index") == PHYSICAL_GPU_INDEX
        and smoke_evidence.get("logical_device") == "cuda:0"
        and smoke_evidence.get("target_or_outcome_path_accessed") is False,
        f"{variant} fit GPU or information boundary changed",
    )
    return {
        "loaded_and_signature_valid": True,
        "identity_and_output_binding_valid": True,
        "artifact_sha256": evidence["artifact_sha256"],
    }


def _signed_file_record(path: Path, *, label: str) -> dict[str, Any]:
    observed = os.lstat(path)
    _require(
        stat.S_ISREG(observed.st_mode)
        and not stat.S_ISLNK(observed.st_mode)
        and observed.st_nlink == 1,
        f"{label} is linked or not a regular file",
    )
    artifact = _load_signed_json(path, label=label)
    return {
        **_bound_file(path, label=label),
        "artifact_sha256": artifact["artifact_sha256"],
    }


def _validate_soak_child_before_cleanup(
    evidence: Mapping[str, Any],
    *,
    dataset: Path,
    output: Path,
) -> dict[str, Any]:
    _require(
        set(evidence)
        == {
            "schema_version",
            "artifact_kind",
            "qualification_id",
            "passed",
            "parameters",
            "runtime",
            "gsplat_runtime_smoke",
            "dataset",
            "initial_global_state",
            "fits",
            "evaluation",
            "formal_held_path_supplied",
            "artifact_sha256",
        },
        "resource soak evidence fields changed",
    )
    fits = evidence.get("fits")
    evaluation = evidence.get("evaluation")
    runtime = evidence.get("runtime")
    smoke = evidence.get("gsplat_runtime_smoke")
    smoke_evidence = smoke.get("evidence") if isinstance(smoke, Mapping) else None
    _require(
        evidence.get("schema_version") == 1
        and evidence.get("artifact_kind")
        == "Deform360ResourceLifecycleSoakChildEvidence"
        and evidence.get("qualification_id") == QUALIFICATION_ID
        and evidence.get("passed") is True
        and evidence.get("parameters")
        == {
            "fit_count": SOAK_FIT_COUNT,
            "iterations_per_fit": SOAK_ITERATIONS,
            "seed": 0,
            "trainer_reinitialization_interval": (
                SOAK_TRAINER_REINITIALIZATION_INTERVAL
            ),
        }
        and evidence.get("dataset") == os.fspath(dataset)
        and evidence.get("formal_held_path_supplied") is False
        and isinstance(runtime, Mapping)
        and runtime.get("seed") == 0
        and runtime.get("cuda_device_count") == 1
        and isinstance(smoke_evidence, Mapping)
        and smoke_evidence.get("physical_gpu_index") == PHYSICAL_GPU_INDEX
        and smoke_evidence.get("logical_device") == "cuda:0"
        and smoke_evidence.get("target_or_outcome_path_accessed") is False
        and isinstance(fits, list)
        and len(fits) == SOAK_FIT_COUNT
        and isinstance(evaluation, Mapping)
        and evaluation.get("passed") is True,
        "resource soak evidence identity or runtime changed",
    )
    expected_fit_fields = {
        "fit_index",
        "trainer_reinitialized",
        "output_created",
        "dataset_outputs_created",
        "output_size_bytes",
        "cleanup_completed",
        "cleanup",
        "output_ply_absent_after_cleanup",
        "dataset_outputs_absent_after_cleanup",
        "resource_boundary_stage",
        "resource_boundary",
        "globals_restored",
        "global_state",
    }
    for index, fit in enumerate(fits):
        _require(
            isinstance(fit, Mapping)
            and set(fit) == expected_fit_fields
            and fit.get("fit_index") == index
            and fit.get("trainer_reinitialized")
            is (index % SOAK_TRAINER_REINITIALIZATION_INTERVAL == 0)
            and fit.get("output_created") is True
            and fit.get("dataset_outputs_created") is True
            and isinstance(fit.get("output_size_bytes"), int)
            and not isinstance(fit.get("output_size_bytes"), bool)
            and int(fit["output_size_bytes"]) > 0
            and fit.get("cleanup_completed") is True
            and fit.get("output_ply_absent_after_cleanup") is True
            and fit.get("dataset_outputs_absent_after_cleanup") is True
            and fit.get("resource_boundary_stage") == "after_cleanup"
            and fit.get("globals_restored") is True,
            f"resource soak fit {index} changed or failed",
        )
        cleanup = fit.get("cleanup")
        _require(
            isinstance(cleanup, Mapping)
            and set(cleanup) == {"output_ply", "dataset_outputs"},
            f"resource soak fit {index} cleanup fields changed",
        )
        output_cleanup = cleanup["output_ply"]
        dataset_cleanup = cleanup["dataset_outputs"]
        _require(
            isinstance(output_cleanup, Mapping)
            and output_cleanup.get("bounded_parent") == os.fspath(output)
            and output_cleanup.get("pre_cleanup_link_count") == 1
            and output_cleanup.get("removed") is True
            and output_cleanup.get("post_cleanup_absent") is True
            and isinstance(output_cleanup.get("pre_cleanup_binding"), Mapping)
            and output_cleanup["pre_cleanup_binding"].get("path")
            == os.fspath(output / f"splat-{index:04d}.ply")
            and output_cleanup["pre_cleanup_binding"].get("size_bytes")
            == fit.get("output_size_bytes"),
            f"resource soak fit {index} output cleanup changed",
        )
        _require(
            isinstance(dataset_cleanup, Mapping)
            and dataset_cleanup.get("bounded_parent") == os.fspath(dataset)
            and dataset_cleanup.get("removed") is True
            and dataset_cleanup.get("post_cleanup_absent") is True
            and isinstance(dataset_cleanup.get("pre_cleanup_inventory"), Mapping)
            and dataset_cleanup["pre_cleanup_inventory"].get("root")
            == os.fspath(dataset / "outputs"),
            f"resource soak fit {index} dataset cleanup changed",
        )
    _require(
        evaluation.get("predicates")
        and all(value is True for value in evaluation["predicates"].values()),
        "resource soak evaluation predicate failed",
    )
    return {
        "loaded_and_signature_valid": True,
        "identity_sequence_resource_and_cleanup_valid": True,
        "artifact_sha256": evidence["artifact_sha256"],
    }


def _validate_analysis_artifacts(
    manifest_path: Path,
    result_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = _load_signed_json(manifest_path, label="equivalence repeat manifest")
    result = _load_signed_json(result_path, label="equivalence analysis result")
    expected_environment = manifest.get("expected_environment")
    _require(
        manifest.get("schema_version") == 1
        and manifest.get("artifact_kind") == ANALYSIS_MANIFEST_KIND
        and manifest.get("analysis_id") == ANALYSIS_ID
        and isinstance(expected_environment, Mapping)
        and expected_environment.get("generator_profile") == GENERATOR_PROFILE
        and expected_environment.get("physical_gpu_index") == PHYSICAL_GPU_INDEX,
        "equivalence manifest identity changed",
    )
    decision = result.get("decision")
    _require(
        result.get("schema_version") == 1
        and result.get("artifact_kind") == ANALYSIS_RESULT_KIND
        and result.get("analysis_id") == ANALYSIS_ID
        and result.get("generator_profile") == GENERATOR_PROFILE
        and result.get("physical_gpu_index") == PHYSICAL_GPU_INDEX
        and result.get("development_only") is True
        and result.get("formal_path_accessed") is False
        and isinstance(decision, Mapping),
        "equivalence result identity changed",
    )
    _require(
        set(decision)
        == {
            "exact_matched_structured_array_equality_primary_passed",
            "exact_matched_file_bytes_equal",
            "secondary_distributional_equivalence_passed",
            "accepted",
            "acceptance_basis",
        }
        and isinstance(decision.get("accepted"), bool)
        and decision.get("acceptance_basis")
        in {
            "exact-structured-array-equality",
            "secondary-distributional-envelope",
            "rejected",
        }
        and (
            decision.get("accepted") is True
            and decision.get("acceptance_basis")
            in {
                "exact-structured-array-equality",
                "secondary-distributional-envelope",
            }
            or decision.get("accepted") is False
            and decision.get("acceptance_basis") == "rejected"
        ),
        "equivalence analysis decision changed",
    )
    manifest_binding = result.get("input_manifest")
    observed_manifest = _bound_file(
        manifest_path, label="result-bound equivalence manifest"
    )
    _require(
        isinstance(manifest_binding, Mapping)
        and all(
            manifest_binding.get(key) == observed_manifest[key]
            for key in ("path", "sha256", "size_bytes", "mode_octal")
        )
        and manifest_binding.get("artifact_sha256") == manifest["artifact_sha256"],
        "equivalence result binds another manifest",
    )
    return manifest, result, dict(decision)


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
    run.add_argument("--ab-repeat-count", type=int, default=AB_REPEAT_COUNT)
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
    run.add_argument("--fit-timeout-seconds", type=int, default=FIT_TIMEOUT_SECONDS)
    run.add_argument(
        "--analyzer-timeout-seconds", type=int, default=ANALYZER_TIMEOUT_SECONDS
    )
    run.add_argument("--soak-timeout-seconds", type=int, default=SOAK_TIMEOUT_SECONDS)

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
    _require(not os.path.lexists(output), "qualification output already exists")
    script = (code / RELATIVE_QUALIFICATION_SOURCE).resolve(strict=True)
    analyzer_script = (code / RELATIVE_ANALYZER_SOURCE).resolve(strict=True)
    _require(
        script == Path(__file__).resolve(strict=True),
        "qualification operator is outside the clean code root",
    )
    _require(analyzer_script.is_file(), "equivalence analyzer source is absent")
    analyzer_source_binding = _bound_file(
        analyzer_script, label="frozen equivalence analyzer source"
    )
    _require(
        analyzer_source_binding["sha256"] == FROZEN_ANALYZER_SOURCE_SHA256,
        "frozen equivalence analyzer source digest changed",
    )
    code_binding = _git_binding(code)
    deform360_binding = _git_binding(deform360, expected_head=PINNED_DEFORM360_REVISION)
    expected_output = QUALIFICATION_BASE / (
        f"bpt-resource-lifecycle-qualification-{code_binding['head']}"
    )
    _require(output == expected_output, "qualification output root is not canonical")
    output.mkdir(parents=True, exist_ok=False)
    attempt_path = _write_new_json(
        output / "qualification-attempt.json",
        _signed(
            {
                "schema_version": 2,
                "artifact_kind": QUALIFICATION_ATTEMPT_KIND,
                "qualification_id": QUALIFICATION_ID,
                "state": "canonical-root-consumed-at-creation",
                "output_root": os.fspath(output),
                "code_revision": code_binding["head"],
                "generator_profile": GENERATOR_PROFILE,
                "physical_gpu_index": PHYSICAL_GPU_INDEX,
                "frozen_analyzer_source": analyzer_source_binding,
                "root_consumption_policy": _root_consumption_policy(),
                "formal_held_path_supplied": False,
            }
        ),
    )
    attempt_record_before = _signed_file_record(
        attempt_path, label="qualification attempt marker before execution"
    )
    temporary = output / "tmp"
    temporary.mkdir(mode=0o700)
    environment = _child_environment(arguments.cuda_device, temporary)
    invocations: dict[str, Any] = {}
    datasets: dict[str, Any] = {}
    cleanup_events: list[dict[str, Any]] = []
    pairing_ids = [f"repeat-{index:03d}" for index in range(AB_REPEAT_COUNT)]
    repeats: dict[str, list[dict[str, Any]]] = {"original": [], "wrapped": []}

    for variant in ("original", "wrapped"):
        for pairing_id in pairing_ids:
            repeat_root = output / "ab" / variant / pairing_id
            repeat_root.mkdir(parents=True)
            materialized = repeat_root / "dataset"
            export = repeat_root / "export"
            export.mkdir()
            dataset_key = f"ab_{variant}_{pairing_id.replace('-', '_')}"
            datasets[dataset_key] = _materialize_dataset(dataset, materialized)
            result_path = repeat_root / "fit-evidence.json"
            output_path = export / "splat.ply"
            command = [
                *_child_python_argv_prefix(python, script),
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
            invocation_key = f"ab_{variant}_{pairing_id.replace('-', '_')}"
            invocation = _invoke_child(
                command,
                environment=environment,
                log_path=repeat_root / "fit.log",
                timeout_seconds=arguments.fit_timeout_seconds,
            )
            invocations[invocation_key] = invocation
            _require(
                invocation["return_code"] == 0
                and invocation["timed_out"] is False
                and invocation["timeout_error"] is None,
                f"{variant} {pairing_id} fit child failed",
            )
            child = _load_signed_json(
                result_path, label=f"{variant} {pairing_id} fit evidence"
            )
            child_validation = _validate_fit_child_before_cleanup(
                child,
                variant=variant,
                dataset=materialized,
                output=output_path,
            )
            generated_outputs = materialized / "outputs"
            _require(
                os.path.lexists(generated_outputs),
                f"{variant} {pairing_id} generated outputs are absent",
            )
            generated_cleanup = _remove_owned_tree(
                generated_outputs,
                parent=materialized,
                label=f"{variant} {pairing_id} generated outputs",
            )
            temporary_cleanup = _reset_owned_temporary(
                temporary,
                output=output,
                label=f"{variant} {pairing_id} temporary cache",
            )
            cleanup = {
                "generated_dataset_outputs": generated_cleanup,
                "qualification_temporary_cache": temporary_cleanup,
            }
            cleanup_events.extend((generated_cleanup, temporary_cleanup))
            repeats[variant].append(
                {
                    "pairing_id": pairing_id,
                    "dataset_key": dataset_key,
                    "invocation_key": invocation_key,
                    "invocation": invocation,
                    "child_evidence": child,
                    "child_evidence_record": _signed_file_record(
                        result_path,
                        label=f"{variant} {pairing_id} fit evidence",
                    ),
                    "child_evidence_validation": child_validation,
                    "retained_output": _bound_file(
                        output_path,
                        label=f"{variant} {pairing_id} retained PLY",
                    ),
                    "cleanup": cleanup,
                }
            )

    identities = [
        _materialized_dataset_identity(datasets[record["dataset_key"]])
        for variant in ("original", "wrapped")
        for record in repeats[variant]
    ]
    _require(
        len(identities) == 2 * AB_REPEAT_COUNT
        and all(value == identities[0] for value in identities[1:]),
        "A/B materialized dataset identities differ",
    )

    equivalence_root = output / "equivalence"
    equivalence_root.mkdir()
    manifest_path = equivalence_root / "repeat-manifest.json"
    manifest_command = [
        *_child_python_argv_prefix(python, analyzer_script),
        "prepare-manifest",
    ]
    for variant in ("original", "wrapped"):
        option = f"--{variant}"
        for record in repeats[variant]:
            manifest_command.extend(
                [
                    option,
                    str(record["pairing_id"]),
                    str(record["retained_output"]["path"]),
                    str(record["child_evidence_record"]["path"]),
                ]
            )
    manifest_command.extend(
        [
            "--canonical-transforms",
            os.fspath(dataset / "transforms.json"),
            "--code-root",
            os.fspath(code),
            "--generator-code-root",
            os.fspath(code),
            "--generator-profile",
            GENERATOR_PROFILE,
            "--deform360-root",
            os.fspath(deform360),
            "--output",
            os.fspath(manifest_path),
        ]
    )
    manifest_invocation = _invoke_child(
        manifest_command,
        environment=environment,
        log_path=equivalence_root / "prepare-manifest.log",
        timeout_seconds=arguments.analyzer_timeout_seconds,
    )
    invocations["equivalence_prepare_manifest"] = manifest_invocation
    _require(
        manifest_invocation["return_code"] == 0
        and manifest_invocation["timed_out"] is False
        and manifest_invocation["timeout_error"] is None,
        "equivalence manifest preparation failed",
    )
    manifest = _load_signed_json(manifest_path, label="equivalence repeat manifest")
    manifest_temp_cleanup = _reset_owned_temporary(
        temporary,
        output=output,
        label="equivalence manifest temporary cache",
    )
    cleanup_events.append(manifest_temp_cleanup)

    result_path = equivalence_root / "analysis-result.json"
    analysis_command = [
        *_child_python_argv_prefix(python, analyzer_script),
        "analyze",
        "--manifest",
        os.fspath(manifest_path),
        "--code-root",
        os.fspath(code),
        "--generator-code-root",
        os.fspath(code),
        "--deform360-root",
        os.fspath(deform360),
        "--output",
        os.fspath(result_path),
    ]
    analysis_invocation = _invoke_child(
        analysis_command,
        environment=environment,
        log_path=equivalence_root / "analyze.log",
        timeout_seconds=arguments.analyzer_timeout_seconds,
    )
    invocations["equivalence_analyze"] = analysis_invocation
    _require(
        analysis_invocation["timed_out"] is False
        and analysis_invocation["timeout_error"] is None
        and analysis_invocation["return_code"] in (0, 3)
        and result_path.is_file(),
        "equivalence analyzer failed technically",
    )
    manifest_after, analysis_result, analysis_decision = _validate_analysis_artifacts(
        manifest_path, result_path
    )
    _require(manifest_after == manifest, "equivalence manifest changed")
    analysis_accepted = analysis_decision.get("accepted") is True
    _require(
        (analysis_invocation["return_code"] == 0 and analysis_accepted)
        or (analysis_invocation["return_code"] == 3 and not analysis_accepted),
        "equivalence analyzer exit code differs from its signed decision",
    )
    analysis_temp_cleanup = _reset_owned_temporary(
        temporary,
        output=output,
        label="equivalence analysis temporary cache",
    )
    cleanup_events.append(analysis_temp_cleanup)
    equivalence = {
        "manifest": {
            **_bound_file(manifest_path, label="equivalence repeat manifest"),
            "artifact_sha256": manifest["artifact_sha256"],
        },
        "prepare_manifest_invocation": manifest_invocation,
        "result": {
            **_bound_file(result_path, label="equivalence analysis result"),
            "artifact_sha256": analysis_result["artifact_sha256"],
        },
        "analysis_invocation": analysis_invocation,
        "decision": analysis_decision,
        "cleanup": {
            "after_manifest": manifest_temp_cleanup,
            "after_analysis": analysis_temp_cleanup,
        },
        "passed": analysis_accepted,
    }

    soak: dict[str, Any] | None = None
    if analysis_accepted:
        soak_root = output / "soak"
        soak_root.mkdir(parents=True)
        materialized = soak_root / "dataset"
        export = soak_root / "export"
        export.mkdir()
        datasets["soak"] = _materialize_dataset(dataset, materialized)
        soak_result_path = soak_root / "soak-evidence.json"
        command = [
            *_child_python_argv_prefix(python, script),
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
            os.fspath(soak_result_path),
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
        soak_invocation = _invoke_child(
            command,
            environment=environment,
            log_path=soak_root / "soak.log",
            timeout_seconds=arguments.soak_timeout_seconds,
        )
        invocations["soak"] = soak_invocation
        _require(
            soak_invocation["return_code"] == 0
            and soak_invocation["timed_out"] is False
            and soak_invocation["timeout_error"] is None,
            "resource soak child failed",
        )
        soak_child = _load_signed_json(soak_result_path, label="soak child evidence")
        soak_child_validation = _validate_soak_child_before_cleanup(
            soak_child,
            dataset=materialized,
            output=export,
        )
        _require(
            not os.path.lexists(materialized / "outputs") and not any(export.iterdir()),
            "resource soak left generated outputs",
        )
        export_cleanup = _remove_owned_tree(
            export, parent=soak_root, label="resource soak empty export"
        )
        final_temp_cleanup = _remove_owned_tree(
            temporary, parent=output, label="final qualification temporary cache"
        )
        cleanup_events.extend((export_cleanup, final_temp_cleanup))
        soak = {
            "passed": True,
            "invocation": soak_invocation,
            "child_evidence": soak_child,
            "child_evidence_record": _signed_file_record(
                soak_result_path, label="soak child evidence"
            ),
            "child_evidence_validation": soak_child_validation,
            "cleanup": {
                "generated_outputs_absent_after_every_fit": True,
                "empty_export_removed": export_cleanup,
                "final_temporary_cache_removed": final_temp_cleanup,
            },
        }
    else:
        final_temp_cleanup = _remove_owned_tree(
            temporary,
            parent=output,
            label="scientific no-go temporary cache",
        )
        cleanup_events.append(final_temp_cleanup)

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
    attempt_record_after = _signed_file_record(
        attempt_path, label="qualification attempt marker after execution"
    )
    ab_predicates = {
        "repeat_count_exact": all(
            len(repeats[variant]) == AB_REPEAT_COUNT
            for variant in ("original", "wrapped")
        ),
        "pairing_ids_exact_and_shared": all(
            [record["pairing_id"] for record in repeats[variant]] == pairing_ids
            for variant in ("original", "wrapped")
        ),
        "all_fit_children_exited_zero": all(
            record["invocation"]["return_code"] == 0
            for variant in ("original", "wrapped")
            for record in repeats[variant]
        ),
        "all_fit_evidence_and_outputs_valid": all(
            record["child_evidence_validation"]["identity_and_output_binding_valid"]
            is True
            for variant in ("original", "wrapped")
            for record in repeats[variant]
        ),
        "all_generated_outputs_and_caches_removed": all(
            record["cleanup"]["generated_dataset_outputs"]["post_cleanup_absent"]
            is True
            and record["cleanup"]["qualification_temporary_cache"][
                "post_cleanup_absent"
            ]
            is True
            for variant in ("original", "wrapped")
            for record in repeats[variant]
        ),
        "materialized_dataset_content_equal": bool(identities)
        and all(value == identities[0] for value in identities),
        "manifest_preparation_passed": manifest_invocation["return_code"] == 0,
        "analyzer_exit_matches_signed_decision": (
            (analysis_invocation["return_code"] == 0 and analysis_accepted)
            or (analysis_invocation["return_code"] == 3 and not analysis_accepted)
        ),
        "equivalence_analysis_accepted": analysis_accepted,
    }
    ab = {
        "passed": all(ab_predicates.values()),
        "repeat_count_per_mode": AB_REPEAT_COUNT,
        "pairing_ids": pairing_ids,
        "repeats": repeats,
        "equivalence": equivalence,
        "predicates": ab_predicates,
    }
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
        "fresh_ten_fit_ab_cohort_passed": ab["passed"],
        "equivalence_analyzer_accepted": analysis_accepted,
        "soak_started_only_after_analyzer_acceptance": (
            (analysis_accepted and soak is not None)
            or (not analysis_accepted and soak is None)
        ),
        "resource_soak_passed": soak is not None and soak["passed"] is True,
        "qualification_temporary_root_absent": not os.path.lexists(temporary),
        "fresh_output_root_was_required": True,
        "analyzer_no_go_skips_soak_and_is_terminal": (
            analysis_accepted or soak is None
        ),
        "retry_or_in_place_reuse_forbidden": True,
        "attempt_marker_stable_across_qualification": (
            attempt_record_after == attempt_record_before
        ),
        "frozen_analyzer_source_digest_exact": (
            analyzer_source_binding["sha256"] == FROZEN_ANALYZER_SOURCE_SHA256
        ),
    }
    analyzer_outcome_predicates = {
        "fresh_ten_fit_ab_cohort_passed",
        "equivalence_analyzer_accepted",
        "resource_soak_passed",
    }
    _require(
        all(
            value is True
            for name, value in predicates.items()
            if name not in analyzer_outcome_predicates
        ),
        "qualification failed technically after canonical root creation",
    )
    if analysis_accepted:
        _require(
            all(value is True for value in predicates.values()),
            "accepted analyzer outcome did not complete every qualification gate",
        )
    else:
        _require(
            all(predicates[name] is False for name in analyzer_outcome_predicates),
            "analyzer no-go is not the sole cause of non-admission",
        )
    passed = analysis_accepted
    admission = {
        "decision": "admitted" if passed else "inconclusive",
        "terminal": True,
        "analyzer_outcome": ("accepted" if analysis_accepted else "scientific-no-go"),
        "analyzer_no_go_interpretation": (
            None if analysis_accepted else ANALYZER_NO_GO_INTERPRETATION
        ),
        "wrapper_inequivalence_proven": False,
        "retry_permitted": False,
        "in_place_reuse_permitted": False,
    }
    evidence = _signed(
        {
            "schema_version": 2,
            "artifact_kind": QUALIFICATION_KIND,
            "qualification_id": QUALIFICATION_ID,
            "status": "qualified" if passed else "admission-inconclusive",
            "passed": passed,
            "admission": admission,
            "attempt": attempt_record_after,
            "root_consumption_policy": _root_consumption_policy(),
            "host": socket.gethostname(),
            "phase": arguments.phase,
            "generator_profile": GENERATOR_PROFILE,
            "physical_gpu_index": PHYSICAL_GPU_INDEX,
            "canonical_run_parameters": canonical_run_parameters,
            "parameters": {
                "cuda_device": arguments.cuda_device,
                "seed": arguments.seed,
                "ab_iterations": arguments.ab_iterations,
                "ab_repeat_count": arguments.ab_repeat_count,
                "soak_fit_count": arguments.soak_fit_count,
                "soak_iterations": arguments.soak_iterations,
                "first_fit_fd_growth_limit": arguments.first_fit_fd_growth_limit,
                "steady_fd_growth_limit": arguments.steady_fd_growth_limit,
                "steady_task_growth_limit": arguments.steady_task_growth_limit,
                "fit_timeout_seconds": arguments.fit_timeout_seconds,
                "analyzer_timeout_seconds": arguments.analyzer_timeout_seconds,
                "soak_timeout_seconds": arguments.soak_timeout_seconds,
            },
            "execution_order": [
                "fresh-five-original-and-five-wrapped-fits",
                "equivalence-analyzer",
                "243-fit-soak-only-after-analyzer-acceptance",
            ],
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
                "analyzer_source": analyzer_source_binding,
            },
            "source_dataset": os.fspath(dataset),
            "materialized_datasets": datasets,
            "invocations": invocations,
            "ab": ab,
            "soak": soak,
            "cleanup_events": cleanup_events,
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
    return 0 if evidence["passed"] else 3


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
