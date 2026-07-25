#!/usr/bin/env python3
"""Materialize the exact six-case Deform360 confirmation source after GO."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import sys
from typing import Any, Mapping


HELD_ROOT = Path("/mnt/corsair/florianpfaff/bpt-online-belief-v1/held-v82")
LOCK = HELD_ROOT / "confirmation-lock.json"
SOURCE_ROOT = HELD_ROOT / "confirmation-source"
PROCESSING_CODE = Path(
    "/mnt/corsair/florianpfaff/bpt-held-v81-runtimes/"
    "Deform360-processing-0fe36f0b7a7a917ba62b5f8cee707299a9a4a317"
)
PINNED_PYTHON = Path(
    "/mnt/corsair/florianpfaff/bpt-held-v5-runtimes/"
    "bpt-gpu-pip-4948737892f77c6a9496795e6c3f25b92fcea466ddb7b5f1e9c1b0de1137f004/"
    "bin/python"
)
PYCACHE_PREFIX = "/nonexistent/bpt-held-v82-pycache"
NORMALIZED_MARKER = "BPT_HELD_V8_CONFIRMATION_SOURCE_ENV_NORMALIZED"
EXPECTED_HOST = "workstation2"
_SOURCE_BINDINGS = {
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
        "configs/sota/deform360_replication_v1.json"
    ),
}
_GIT_ENVIRONMENT = {
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "HOME": "/nonexistent",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": "/usr/bin:/bin",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _stable_file_state(value: os.stat_result) -> tuple[int, ...]:
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


def _sha256_file(path: Path) -> str:
    observed = os.lstat(path)
    _require(
        stat.S_ISREG(observed.st_mode) and not stat.S_ISLNK(observed.st_mode),
        f"source is absent, linked, or not a regular file: {path}",
    )
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        _require(
            _stable_file_state(opened) == _stable_file_state(observed),
            f"source changed while opening: {path}",
        )
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
        after = os.fstat(descriptor)
        current = os.lstat(path)
        _require(
            _stable_file_state(after)
            == _stable_file_state(opened)
            == _stable_file_state(current),
            f"source changed while hashing: {path}",
        )
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _read_stable_regular_bytes(path: Path) -> bytes:
    observed = os.lstat(path)
    _require(
        stat.S_ISREG(observed.st_mode) and not stat.S_ISLNK(observed.st_mode),
        f"source is absent, linked, or not a regular file: {path}",
    )
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    payload = bytearray()
    try:
        opened = os.fstat(descriptor)
        _require(
            _stable_file_state(opened) == _stable_file_state(observed),
            f"source changed while opening: {path}",
        )
        while block := os.read(descriptor, 1024 * 1024):
            payload.extend(block)
        after = os.fstat(descriptor)
        current = os.lstat(path)
        _require(
            _stable_file_state(after)
            == _stable_file_state(opened)
            == _stable_file_state(current),
            f"source changed while reading: {path}",
        )
    finally:
        os.close(descriptor)
    return bytes(payload)


def _runtime_root() -> Path:
    return HELD_ROOT / ".confirmation-source-runtime"


def _remove_owned_runtime_tree(
    root: Path, *, expected_identity: tuple[int, int]
) -> None:
    observed_root = os.lstat(root)
    _require(
        stat.S_ISDIR(observed_root.st_mode)
        and not stat.S_ISLNK(observed_root.st_mode)
        and (observed_root.st_dev, observed_root.st_ino) == expected_identity,
        "refusing to clean up a replaced confirmation runtime root",
    )
    directories: list[Path] = []
    for current, names, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        observed_current = os.lstat(current_path)
        _require(
            stat.S_ISDIR(observed_current.st_mode)
            and not stat.S_ISLNK(observed_current.st_mode),
            "confirmation runtime cleanup encountered a non-directory",
        )
        os.chmod(current_path, 0o700, follow_symlinks=False)
        retained: list[str] = []
        for name in sorted(names):
            child = current_path / name
            observed = os.lstat(child)
            if stat.S_ISLNK(observed.st_mode):
                child.unlink()
                continue
            _require(
                stat.S_ISDIR(observed.st_mode),
                "confirmation runtime cleanup encountered a non-directory child",
            )
            os.chmod(child, 0o700, follow_symlinks=False)
            retained.append(name)
            directories.append(child)
        names[:] = retained
        for name in sorted(files):
            child = current_path / name
            _require(
                not stat.S_ISDIR(os.lstat(child).st_mode),
                "confirmation runtime cleanup encountered an unexpected directory",
            )
            child.unlink()
    for directory in sorted(
        directories, key=lambda value: len(value.parts), reverse=True
    ):
        directory.rmdir()
    current_root = os.lstat(root)
    _require(
        (current_root.st_dev, current_root.st_ino) == expected_identity,
        "confirmation runtime root changed during cleanup",
    )
    root.rmdir()


def _cleanup_runtime_or_rollback_source(
    *,
    runtime: Path,
    runtime_identity: tuple[int, int],
    published_source_identity: tuple[int, int] | None,
    source_module: Any,
) -> None:
    try:
        _remove_owned_runtime_tree(
            runtime,
            expected_identity=runtime_identity,
        )
    except BaseException:
        if published_source_identity is not None and os.path.lexists(SOURCE_ROOT):
            source_module._remove_owned_tree(
                SOURCE_ROOT,
                expected_identity=published_source_identity,
            )
            parent_descriptor = os.open(
                SOURCE_ROOT.parent,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0),
            )
            try:
                os.fsync(parent_descriptor)
            finally:
                os.close(parent_descriptor)
        raise


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _git(code: Path, *arguments: str, check: bool = True) -> tuple[int, bytes, bytes]:
    import subprocess

    completed = subprocess.run(
        [
            "/usr/bin/git",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            "-C",
            str(code),
            *arguments,
        ],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_GIT_ENVIRONMENT,
        timeout=120,
    )
    if check and completed.returncode != 0:
        raise ValueError(
            f"git {' '.join(arguments)} failed: "
            + completed.stderr.decode("utf-8", errors="replace").strip()
        )
    return completed.returncode, completed.stdout, completed.stderr


def _git_tree_records(raw: bytes) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for encoded in raw.split(b"\0"):
        if not encoded:
            continue
        header, separator, path_bytes = encoded.partition(b"\t")
        fields = header.split(b" ")
        _require(bool(separator) and len(fields) == 3, "malformed Git tree record")
        try:
            mode, kind, object_id = (field.decode("ascii") for field in fields)
            path = path_bytes.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError("deployed Git tree is not canonical text") from error
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
        "deployed Git tree is empty or unsorted",
    )
    return records


def _validate_deployed_repository(
    code: Path, bindings: Mapping[str, Any]
) -> dict[str, str]:
    _, top_raw, _ = _git(code, "rev-parse", "--show-toplevel")
    _, head_raw, _ = _git(code, "rev-parse", "HEAD")
    top = top_raw.decode("utf-8").strip()
    head = head_raw.decode("ascii").strip().lower()
    _require(
        top == str(code)
        and len(head) in {40, 64}
        and all(character in "0123456789abcdef" for character in head)
        and code.name == f"code-{head}",
        "confirmation source deployment is not exact code-$HEAD",
    )
    symbolic_rc, symbolic_output, _ = _git(
        code, "symbolic-ref", "-q", "HEAD", check=False
    )
    _require(
        symbolic_rc == 1 and symbolic_output == b"",
        "confirmation source deployment is not detached",
    )
    _, status, _ = _git(code, "status", "--porcelain=v1", "--untracked-files=all")
    _, untracked, _ = _git(code, "ls-files", "--others", "--exclude-standard", "-z")
    _, ignored, _ = _git(
        code,
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
        "-z",
    )
    _require(
        status == b"" and untracked == b"" and ignored == b"",
        "confirmation source deployment is not a clean exact worktree",
    )
    _git(code, "fsck", "--full", "--no-dangling")
    _, tree_raw, _ = _git(code, "ls-tree", "-r", "-z", "HEAD")
    records = _git_tree_records(tree_raw)
    evidence = {
        "head": head,
        "head_text_sha256": hashlib.sha256(head.encode("utf-8")).hexdigest(),
        "tree_sha256": hashlib.sha256(_canonical_bytes(records)).hexdigest(),
    }
    _require(
        bindings.get("method_head_text_sha256") == evidence["head_text_sha256"]
        and bindings.get("method_deployed_snapshot_tree") == evidence["tree_sha256"],
        "confirmation source deployed HEAD or canonical Git tree differs from lock",
    )
    return evidence


def _normalized_environment(code: Path) -> dict[str, str]:
    temporary = _runtime_root()
    return {
        "HOME": "/home/florianpfaff",
        "USER": "florianpfaff",
        "LOGNAME": "florianpfaff",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "TMPDIR": str(temporary),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONPYCACHEPREFIX": PYCACHE_PREFIX,
        "HF_HOME": str(temporary / "hf"),
        "HF_HUB_CACHE": str(temporary / "hf" / "hub"),
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "DO_NOT_TRACK": "1",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_OPTIONAL_LOCKS": "0",
        "BPT_HELD_V8_CODE": str(code),
        NORMALIZED_MARKER: "1",
    }


def _normalize_or_reexec(code: Path) -> None:
    if os.environ.get(NORMALIZED_MARKER) == "1":
        _require(sys.flags.isolated == 1, "source operator is not isolated")
        _require(
            dict(os.environ) == _normalized_environment(code),
            "source operator environment changed",
        )
        _require(
            sys.flags.dont_write_bytecode == 1 and sys.pycache_prefix == PYCACHE_PREFIX,
            "source operator may consult adjacent bytecode",
        )
        return
    environment = _normalized_environment(code)
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


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(_read_stable_regular_bytes(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"canonical JSON expected: {path}") from error
    _require(isinstance(value, dict), f"JSON object expected: {path}")
    return value


def _validate_confirmation_selection_lineage(
    *,
    geometry_qa: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    source: Any,
) -> None:
    cases = tuple(source.CONFIRMATION_SOURCE_CASES)
    expected_object_ids = [case.object_id for case in cases]
    _require(
        len(cases) == 6 and len(set(expected_object_ids)) == 6,
        "confirmation source cases are not the exact six unique objects",
    )

    qa_objects = geometry_qa.get("objects")
    _require(
        isinstance(qa_objects, list)
        and len(qa_objects) == len(cases)
        and all(isinstance(record, Mapping) for record in qa_objects),
        "camera-selection QA does not contain the exact six object records",
    )
    _require(
        [record.get("object_id") for record in qa_objects] == expected_object_ids,
        "camera-selection QA object order differs from confirmation cases",
    )
    for case, record in zip(cases, qa_objects, strict=True):
        selected_cameras = record.get("selected_cameras")
        _require(
            isinstance(selected_cameras, list)
            and all(isinstance(camera, str) and camera for camera in selected_cameras)
            and tuple(selected_cameras) == tuple(case.cameras),
            f"camera-selection QA differs from confirmation case {case.case_name}",
        )

    preregistration_config = preregistration.get("config")
    _require(
        isinstance(preregistration_config, Mapping),
        "preregistration lacks its config object",
    )
    cohort = preregistration_config.get("cohort")
    _require(
        isinstance(cohort, list)
        and len(cohort) == len(cases)
        and all(isinstance(record, Mapping) for record in cohort),
        "preregistration does not contain the exact six cohort records",
    )
    _require(
        [record.get("object_id") for record in cohort] == expected_object_ids,
        "preregistration cohort order differs from confirmation cases",
    )
    for case, record in zip(cases, cohort, strict=True):
        _require(
            type(record.get("target_episode_id")) is int
            and record["target_episode_id"] == case.episode_id
            and type(record.get("target_bimanual")) is bool
            and record["target_bimanual"] is case.bimanual,
            f"preregistration target differs from confirmation case {case.case_name}",
        )


def _validate_deployment(code: Path, lock: Mapping[str, Any], source: Any) -> None:
    expected_wrapper = (
        code / "scripts" / "held" / "run_deform360_v8_confirmation_source.py"
    )
    _require(
        code.is_dir()
        and not code.is_symlink()
        and code.resolve() == code
        and code.parent == HELD_ROOT
        and code.name.startswith("code-"),
        "confirmation source code is outside the immutable deployment",
    )
    _require(
        Path(__file__).resolve(strict=True) == expected_wrapper,
        "executing confirmation source wrapper escaped the immutable deployment",
    )
    _require(
        not any(
            stat.S_IMODE(os.lstat(path).st_mode) & 0o222 for path in code.rglob("*")
        ),
        "confirmation source deployment is writable",
    )
    bindings = lock.get("immutable_bindings")
    _require(isinstance(bindings, Mapping), "confirmation lock lacks bindings")
    _validate_deployed_repository(code, bindings)
    lineage_payloads: dict[str, bytes] = {}
    for key, relative in _SOURCE_BINDINGS.items():
        path = code / relative
        payload = (
            _read_stable_regular_bytes(path)
            if key
            in {
                "held_v8_confirmation_camera_selection_lineage",
                "held_v8_confirmation_preregistration_lineage",
            }
            else None
        )
        observed_sha256 = (
            hashlib.sha256(payload).hexdigest()
            if payload is not None
            else _sha256_file(path)
        )
        _require(
            path.is_file()
            and not path.is_symlink()
            and path.resolve() == path
            and bindings.get(key) == observed_sha256,
            f"confirmation source binding changed: {key}",
        )
        if payload is not None:
            lineage_payloads[key] = payload
    _require(
        bindings.get("held_v8_confirmation_source_contract")
        == source.confirmation_source_contract_sha256(),
        "confirmation source contract is not locked",
    )
    qa_payload = lineage_payloads["held_v8_confirmation_camera_selection_lineage"]
    preregistration_payload = lineage_payloads[
        "held_v8_confirmation_preregistration_lineage"
    ]
    _require(
        hashlib.sha256(qa_payload).hexdigest() == source.SOURCE_GEOMETRY_QA_FILE_SHA256
        and hashlib.sha256(preregistration_payload).hexdigest()
        == source.REPLICATION_PREREGISTRATION_FILE_SHA256,
        "confirmation selection lineage differs from its source contract",
    )
    try:
        geometry_qa = json.loads(qa_payload.decode("utf-8"))
        preregistration = json.loads(preregistration_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("confirmation selection lineage is not JSON") from error
    _require(
        isinstance(geometry_qa, Mapping) and isinstance(preregistration, Mapping),
        "confirmation selection lineage lacks JSON objects",
    )
    _validate_confirmation_selection_lineage(
        geometry_qa=geometry_qa,
        preregistration=preregistration,
        source=source,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deployed-code", required=True)
    namespace = parser.parse_args()
    code = Path(os.path.abspath(namespace.deployed_code))
    _normalize_or_reexec(code)
    _require(
        socket.gethostname() == EXPECTED_HOST,
        "confirmation source must run on gpuserver6000/workstation2",
    )
    _require(
        not os.path.lexists("/nonexistent") and not os.path.lexists(PYCACHE_PREFIX),
        "reserved held-v8 bytecode prefix is available",
    )
    sys.path.insert(0, str(code / "src"))
    from bayesian_phystwin import (  # noqa: PLC0415
        deform360_held_v8_confirmation_source as source,
    )
    from bayesian_phystwin import deform360_held_v8_protocol as protocol  # noqa: PLC0415

    expected_package = code / "src" / "bayesian_phystwin"
    _require(
        Path(source.__file__).resolve()
        == expected_package / "deform360_held_v8_confirmation_source.py"
        and Path(protocol.__file__).resolve()
        == expected_package / "deform360_held_v8_protocol.py",
        "confirmation source modules escaped the immutable deployment",
    )

    # Protocol authorization recursively validates the sealed calibration GO.
    # No source/provider/payload path is touched before this returns.
    permit = protocol.authorize_confirmation_source_materialization(LOCK)
    expected = protocol.confirmation_source_permit_evidence(LOCK)
    lock = protocol.validate_protocol_lock(LOCK)
    _validate_deployment(code, lock, source)
    runtime = _runtime_root()
    _require(
        not os.path.lexists(runtime), "confirmation source runtime root is not fresh"
    )
    runtime.mkdir(mode=0o700)
    runtime_state = os.lstat(runtime)
    runtime_identity = (runtime_state.st_dev, runtime_state.st_ino)
    published_source_identity: tuple[int, int] | None = None
    try:
        manifest = source.materialize_confirmation_source_cohort(
            source.ConfirmationSourcePaths(
                source_root=SOURCE_ROOT,
                processing_code_root=PROCESSING_CODE,
                python_executable=PINNED_PYTHON,
            ),
            source_permit=permit,
            consume_source_permit=(
                protocol.consume_confirmation_source_materialization_capability
            ),
            expected_source_permit=expected,
        )
        published_state = os.lstat(SOURCE_ROOT)
        published_source_identity = (
            published_state.st_dev,
            published_state.st_ino,
        )
    finally:
        if os.path.lexists(runtime):
            _cleanup_runtime_or_rollback_source(
                runtime=runtime,
                runtime_identity=runtime_identity,
                published_source_identity=published_source_identity,
                source_module=source,
            )
    print(
        json.dumps(
            {
                "event": "CONFIRMATION_SOURCE_COHORT_PUBLISHED",
                "manifest": str(manifest),
                "manifest_sha256": _sha256_file(manifest),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
