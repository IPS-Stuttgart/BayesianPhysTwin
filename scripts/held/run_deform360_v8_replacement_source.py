#!/usr/bin/env python3
"""Acquire the one fresh Deform360 held-v8 calibration replacement source."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence


HELD_ROOT = Path("/mnt/corsair/florianpfaff/bpt-online-belief-v1/held-v83")
CALIBRATION_LOCK = HELD_ROOT / "calibration-lock.json"
SOURCE_ROOT = HELD_ROOT / "replacement-source"
PROCESSING_CODE = Path(
    "/mnt/corsair/florianpfaff/bpt-held-v81-runtimes/"
    "Deform360-processing-0fe36f0b7a7a917ba62b5f8cee707299a9a4a317"
)
PINNED_PYTHON = Path(
    "/mnt/corsair/florianpfaff/bpt-held-v5-runtimes/"
    "bpt-gpu-pip-4948737892f77c6a9496795e6c3f25b92fcea466ddb7b5f1e9c1b0de1137f004/"
    "bin/python"
)
PINNED_PYTHON_LINK_TARGET = "/usr/bin/python3"
PINNED_PYTHON_RESOLVED = Path("/usr/bin/python3.12")
PINNED_PYTHON_TARGET_SHA256 = (
    "e1efa562c2cc2e35521a5c9c9b9939921001ff8ca9708a13ef15ace68cc2ccd7"
)
PYCACHE_PREFIX = Path("/nonexistent/bpt-held-v83-pycache")
PROCESSING_REVISION = "0fe36f0b7a7a917ba62b5f8cee707299a9a4a317"
PROCESSING_TREE = "c566ed29db7e0fd6a4cb768d840a4aa662864680"
NORMALIZED_MARKER = "BPT_HELD_V8_REPLACEMENT_SOURCE_ENV_NORMALIZED"
CODE_ENVIRONMENT_KEY = "BPT_HELD_V8_CODE"
EXPECTED_HOST = "workstation2"

_GIT = Path("/usr/bin/git")
_HEAD_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_BINDING_KEYS = frozenset(
    {
        "held_v8_protocol_source",
        "held_v8_replacement_source_operator_source",
        "held_v8_replacement_source_acquisition_launcher_source",
        "replacement_source_inventory_contract",
        "method_deployed_snapshot_tree",
        "method_head_text_sha256",
    }
)


@dataclass(frozen=True)
class FormalSourcePaths:
    source_root: Path
    download_root: Path
    aligned_root: Path
    manifest_root: Path
    inventory_manifest: Path
    content_manifest: Path
    aligned_source_manifest: Path
    temporary_root: Path
    hf_home: Path
    cache_root: Path
    matplotlib_root: Path


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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    observed = os.lstat(path)
    _require(stat.S_ISREG(observed.st_mode), f"not a regular file: {path}")
    _require(not stat.S_ISLNK(observed.st_mode), f"linked file is forbidden: {path}")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        _require(
            (observed.st_dev, observed.st_ino) == (opened.st_dev, opened.st_ino),
            f"file changed while opening: {path}",
        )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        after = os.fstat(descriptor)
        current = os.lstat(path)
        _require(
            (after.st_dev, after.st_ino) == (opened.st_dev, opened.st_ino)
            and (current.st_dev, current.st_ino) == (opened.st_dev, opened.st_ino)
            and after.st_size == opened.st_size
            and after.st_mtime_ns == opened.st_mtime_ns
            and after.st_ctime_ns == opened.st_ctime_ns,
            f"file changed while hashing: {path}",
        )
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _canonical_existing_directory(path: Path, *, label: str) -> Path:
    _require(path.is_absolute(), f"{label} is not absolute")
    observed = os.lstat(path)
    _require(
        stat.S_ISDIR(observed.st_mode) and not stat.S_ISLNK(observed.st_mode),
        f"{label} is absent, linked, or not a directory",
    )
    _require(path.resolve(strict=True) == path, f"{label} is not canonical")
    return path


def formal_source_paths() -> FormalSourcePaths:
    manifests = SOURCE_ROOT / "manifests"
    return FormalSourcePaths(
        source_root=SOURCE_ROOT,
        download_root=SOURCE_ROOT / "download",
        aligned_root=SOURCE_ROOT / "aligned",
        manifest_root=manifests,
        inventory_manifest=manifests / "remote-inventory.json",
        content_manifest=manifests / "downloaded-content.json",
        aligned_source_manifest=manifests / "aligned-source.json",
        temporary_root=SOURCE_ROOT / "tmp",
        hf_home=SOURCE_ROOT / "hf-home",
        cache_root=SOURCE_ROOT / "cache",
        matplotlib_root=SOURCE_ROOT / "matplotlib",
    )


def normalized_environment(code: Path) -> dict[str, str]:
    paths = formal_source_paths()
    return {
        "HOME": "/home/florianpfaff",
        "USER": "florianpfaff",
        "LOGNAME": "florianpfaff",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "TMPDIR": str(paths.temporary_root),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONPYCACHEPREFIX": str(PYCACHE_PREFIX),
        "HF_HOME": str(paths.hf_home),
        "HF_HUB_CACHE": str(paths.hf_home / "hub"),
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "DO_NOT_TRACK": "1",
        "XDG_CACHE_HOME": str(paths.cache_root),
        "MPLCONFIGDIR": str(paths.matplotlib_root),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_OPTIONAL_LOCKS": "0",
        CODE_ENVIRONMENT_KEY: str(code),
        NORMALIZED_MARKER: "1",
    }


def _run_git(
    code: Path,
    arguments: Sequence[str],
    *,
    returns: frozenset[int] = frozenset({0}),
) -> bytes:
    environment = {
        "HOME": "/nonexistent",
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_OPTIONAL_LOCKS": "0",
    }
    completed = subprocess.run(
        [str(_GIT), "-C", str(code), *arguments],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=120,
        env=environment,
    )
    _require(
        completed.returncode in returns,
        "Git provenance command failed: "
        f"git {' '.join(arguments)}: "
        f"{completed.stderr.decode('utf-8', 'replace').strip()}",
    )
    return completed.stdout


def parse_git_tree(raw: bytes) -> list[dict[str, str]]:
    """Parse the exact ``git ls-tree -r -z HEAD`` provenance representation."""

    records: list[dict[str, str]] = []
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        try:
            metadata, path_bytes = entry.split(b"\t", 1)
            mode_bytes, type_bytes, object_bytes = metadata.split(b" ", 2)
            mode = mode_bytes.decode("ascii")
            kind = type_bytes.decode("ascii")
            object_id = object_bytes.decode("ascii").lower()
            path = path_bytes.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError("deployed Git tree record is malformed") from error
        parts = path.split("/")
        _require(
            path
            and not path.startswith("/")
            and all(part not in {"", ".", ".."} for part in parts),
            "deployed Git tree contains an unsafe path",
        )
        _require(kind == "blob", f"deployed Git entry is not a blob: {path}")
        _require(mode in {"100644", "100755"}, f"unsafe Git mode: {path}")
        _require(
            _HEAD_RE.fullmatch(object_id) is not None, f"invalid Git object: {path}"
        )
        records.append(
            {"mode": mode, "type": kind, "object_id": object_id, "path": path}
        )
    _require(bool(records), "deployed Git tree is empty")
    paths = [record["path"] for record in records]
    _require(paths == sorted(paths), "deployed Git tree order changed")
    _require(len(paths) == len(set(paths)), "deployed Git tree has duplicate paths")
    return records


def git_tree_records_sha256(records: Sequence[Mapping[str, str]]) -> str:
    return _sha256_bytes(_canonical_bytes(list(records)))


def _validate_nonwritable_tree(code: Path) -> None:
    stack = [code]
    while stack:
        directory = stack.pop()
        observed_directory = os.lstat(directory)
        _require(
            stat.S_ISDIR(observed_directory.st_mode)
            and not stat.S_ISLNK(observed_directory.st_mode)
            and observed_directory.st_mode & 0o222 == 0,
            f"deployed directory is linked, special, or writable: {directory}",
        )
        with os.scandir(directory) as iterator:
            entries = sorted(iterator, key=lambda item: item.name)
        for entry in entries:
            observed = entry.stat(follow_symlinks=False)
            path = Path(entry.path)
            _require(not stat.S_ISLNK(observed.st_mode), f"deployed symlink: {path}")
            _require(
                observed.st_mode & 0o222 == 0, f"deployed path is writable: {path}"
            )
            if stat.S_ISDIR(observed.st_mode):
                stack.append(path)
            else:
                _require(
                    stat.S_ISREG(observed.st_mode), f"deployed special file: {path}"
                )


def deployed_git_provenance(code: Path) -> dict[str, Any]:
    _canonical_existing_directory(code, label="deployed held-v8 code")
    _require(stat.S_ISREG(os.lstat(_GIT).st_mode), "pinned Git executable is absent")
    _canonical_existing_directory(code / ".git", label="deployed held-v8 Git directory")
    _validate_nonwritable_tree(code)
    symbolic = subprocess.run(
        [str(_GIT), "-C", str(code), "symbolic-ref", "-q", "HEAD"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=120,
        env={
            "HOME": "/nonexistent",
            "PATH": "/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_OPTIONAL_LOCKS": "0",
        },
    )
    _require(symbolic.returncode == 1, "deployed held-v8 HEAD is not detached")
    head = (
        _run_git(code, ("rev-parse", "--verify", "HEAD"))
        .decode("ascii")
        .strip()
        .lower()
    )
    _require(_HEAD_RE.fullmatch(head) is not None, "deployed held-v8 HEAD is invalid")
    _require(code.name == f"code-{head}", "deployed held-v8 path is not code-$HEAD")
    _require(
        _run_git(code, ("status", "--porcelain", "--untracked-files=all")) == b"",
        "deployed held-v8 worktree is dirty",
    )
    records = parse_git_tree(_run_git(code, ("ls-tree", "-r", "-z", "HEAD")))
    return {
        "head": head,
        "head_text_sha256": _sha256_bytes(head.encode("ascii")),
        "tree_records": records,
        "tree_records_sha256": git_tree_records_sha256(records),
    }


def required_deployment_bindings(
    *,
    code: Path,
    launcher: Path,
    head_text_sha256: str,
    tree_records_sha256: str,
    protocol_module: ModuleType,
    source_module: ModuleType,
) -> dict[str, str]:
    protocol_path = code / "src/bayesian_phystwin/deform360_held_v8_protocol.py"
    source_path = code / "src/bayesian_phystwin/deform360_held_v8_replacement_source.py"
    _require(
        Path(str(protocol_module.__file__)).resolve(strict=True) == protocol_path,
        "held-v8 protocol imported outside deployed code",
    )
    _require(
        Path(str(source_module.__file__)).resolve(strict=True) == source_path,
        "replacement-source operator imported outside deployed code",
    )
    return {
        "held_v8_protocol_source": _sha256_file(protocol_path),
        "held_v8_replacement_source_operator_source": _sha256_file(source_path),
        "held_v8_replacement_source_acquisition_launcher_source": _sha256_file(
            launcher
        ),
        "replacement_source_inventory_contract": protocol_module.held_contract_sha256(
            source_module.REPLACEMENT_SOURCE_INVENTORY_CONTRACT
        ),
        "method_deployed_snapshot_tree": tree_records_sha256,
        "method_head_text_sha256": head_text_sha256,
    }


def validate_required_bindings(
    lock: Mapping[str, Any], expected: Mapping[str, str]
) -> None:
    bindings = lock.get("immutable_bindings")
    _require(isinstance(bindings, Mapping), "calibration lock bindings are invalid")
    _require(set(expected) == _REQUIRED_BINDING_KEYS, "launcher binding set changed")
    _require(
        all(_SHA256_RE.fullmatch(value) is not None for value in expected.values()),
        "launcher expected binding is not SHA-256",
    )
    missing = sorted(key for key in expected if bindings.get(key) != expected[key])
    _require(not missing, f"calibration lock deployment binding changed: {missing}")


def validate_pinned_python(
    lock: Mapping[str, Any],
    *,
    launcher: Path = PINNED_PYTHON,
    expected_link_target: str = PINNED_PYTHON_LINK_TARGET,
    expected_resolved: Path = PINNED_PYTHON_RESOLVED,
    expected_sha256: str = PINNED_PYTHON_TARGET_SHA256,
) -> None:
    observed = os.lstat(launcher)
    _require(
        stat.S_ISLNK(observed.st_mode)
        and os.readlink(launcher) == expected_link_target,
        "pinned Python launcher symlink changed",
    )
    resolved = launcher.resolve(strict=True)
    _require(
        resolved == expected_resolved
        and stat.S_ISREG(os.lstat(resolved).st_mode)
        and not resolved.is_symlink()
        and os.access(resolved, os.X_OK),
        "pinned Python executable target changed",
    )
    digest = _sha256_file(resolved)
    bindings = lock.get("immutable_bindings")
    _require(
        digest == expected_sha256
        and isinstance(bindings, Mapping)
        and bindings.get("pinned_python_executable_target") == digest,
        "pinned Python executable binding changed",
    )


def validate_runtime_bindings(
    lock: Mapping[str, Any], *, source_module: ModuleType
) -> None:
    bindings = lock.get("immutable_bindings")
    _require(isinstance(bindings, Mapping), "runtime lock bindings are invalid")
    expected = {
        "deform360_processing_head_text_sha256": _sha256_bytes(
            PROCESSING_REVISION.encode("ascii")
        ),
        "hf_dataset_revision_text_sha256": _sha256_bytes(
            source_module.HF_DATASET_REVISION.encode("ascii")
        ),
    }
    _require(
        all(bindings.get(key) == value for key, value in expected.items()),
        "replacement-source runtime binding changed",
    )


def validate_processing_revision(code: Path) -> str:
    _require(code == PROCESSING_CODE, "Deform360 processing path changed")
    _canonical_existing_directory(code, label="pinned Deform360 processing code")
    _require(
        not any(
            os.lstat(Path(current) / name).st_mode & 0o222
            for current, directories, files in os.walk(code, followlinks=False)
            for name in [*directories, *files]
        )
        and os.lstat(code).st_mode & 0o222 == 0,
        "pinned Deform360 processing snapshot is writable",
    )
    head = (
        _run_git(code, ("rev-parse", "--verify", "HEAD"))
        .decode("ascii")
        .strip()
        .lower()
    )
    _require(
        head == PROCESSING_REVISION, "pinned Deform360 processing revision changed"
    )
    tree = (
        _run_git(code, ("rev-parse", "--verify", "HEAD^{tree}"))
        .decode("ascii")
        .strip()
        .lower()
    )
    _require(tree == PROCESSING_TREE, "pinned Deform360 processing tree changed")
    _require(
        _run_git(code, ("status", "--porcelain", "--untracked-files=all")) == b"",
        "pinned Deform360 processing worktree is dirty",
    )
    _require(
        _run_git(
            code,
            ("ls-files", "--others", "--ignored", "--exclude-standard"),
        )
        == b"",
        "pinned Deform360 processing snapshot contains ignored files",
    )
    return head


def invoke_formal_operator(
    *,
    protocol_module: ModuleType,
    source_module: ModuleType,
    paths: FormalSourcePaths,
) -> Path:
    """Issue the source capability and invoke the operator in one process."""

    permit = protocol_module.authorize_replacement_source_acquisition(CALIBRATION_LOCK)
    expected_permit = protocol_module.replacement_source_permit_evidence(
        CALIBRATION_LOCK
    )
    result = source_module.acquire_and_align_replacement_source(
        source_module.ReplacementSourcePaths(
            download_root=paths.download_root,
            aligned_root=paths.aligned_root,
            inventory_manifest=paths.inventory_manifest,
            content_manifest=paths.content_manifest,
            aligned_source_manifest=paths.aligned_source_manifest,
            processing_code_root=PROCESSING_CODE,
            python_executable=PINNED_PYTHON,
        ),
        source_permit=permit,
        consume_source_permit=(
            protocol_module.consume_replacement_source_acquisition_capability
        ),
        expected_source_permit=expected_permit,
        revision_reader=validate_processing_revision,
    )
    _require(
        result == paths.aligned_source_manifest, "source operator output path changed"
    )
    return result


def _reexec_normalized(script: Path, code: Path) -> None:
    environment = normalized_environment(code)
    os.execve(
        PINNED_PYTHON,
        [
            str(PINNED_PYTHON),
            "-I",
            "-B",
            "-X",
            f"pycache_prefix={PYCACHE_PREFIX}",
            str(script),
        ],
        environment,
    )


def main() -> None:
    _require(len(sys.argv) == 1, "replacement-source launcher takes no arguments")
    script = Path(os.path.abspath(__file__))
    _require(script.is_absolute(), "launcher path is not absolute")
    _require(
        stat.S_ISREG(os.lstat(script).st_mode) and not script.is_symlink(),
        "replacement-source launcher is absent or linked",
    )
    code_from_script = script.parents[2]
    supplied_code = os.environ.get(CODE_ENVIRONMENT_KEY)
    _require(
        bool(supplied_code), f"set {CODE_ENVIRONMENT_KEY} to deployed held-v8 code"
    )
    code = Path(str(supplied_code))
    _require(code.is_absolute(), "deployed held-v8 code path is not absolute")
    _require(
        code_from_script == code, "launcher is outside supplied held-v8 deployment"
    )

    if os.environ.get(NORMALIZED_MARKER) != "1":
        _reexec_normalized(script, code)
        raise AssertionError("execve unexpectedly returned")
    _require(
        dict(os.environ) == normalized_environment(code),
        "normalized replacement-source environment changed",
    )
    os.umask(0o077)
    _require(
        socket.gethostname() == EXPECTED_HOST, "formal source host is not workstation2"
    )
    _require(
        not os.path.lexists(Path("/nonexistent"))
        and not os.path.lexists(PYCACHE_PREFIX),
        "reserved pycache path is available",
    )
    _canonical_existing_directory(HELD_ROOT, label="held-v8 root")
    _require(
        script == code / "scripts/held/run_deform360_v8_replacement_source.py",
        "launcher canonical deployed path changed",
    )

    provenance = deployed_git_provenance(code)
    source_path = code / "src"
    sys.path.insert(0, str(source_path))
    import bayesian_phystwin.deform360_held_v8_protocol as protocol_module
    import bayesian_phystwin.deform360_held_v8_replacement_source as source_module

    lock = protocol_module.validate_protocol_lock(CALIBRATION_LOCK)
    _require(
        lock.get("stage") == "calibration",
        "source acquisition requires calibration lock",
    )
    _require(Path(str(lock.get("held_root"))) == HELD_ROOT, "lock held root changed")
    expected_bindings = required_deployment_bindings(
        code=code,
        launcher=script,
        head_text_sha256=provenance["head_text_sha256"],
        tree_records_sha256=provenance["tree_records_sha256"],
        protocol_module=protocol_module,
        source_module=source_module,
    )
    validate_required_bindings(lock, expected_bindings)
    validate_runtime_bindings(lock, source_module=source_module)
    validate_processing_revision(PROCESSING_CODE)
    validate_pinned_python(lock)

    paths = formal_source_paths()
    _require(
        not os.path.lexists(paths.source_root),
        "formal replacement source root already exists; reuse is forbidden",
    )
    paths.source_root.mkdir(mode=0o700)
    for directory in (
        paths.manifest_root,
        paths.temporary_root,
        paths.hf_home,
        paths.cache_root,
        paths.matplotlib_root,
    ):
        directory.mkdir(mode=0o700)
        _canonical_existing_directory(directory, label="fresh source support directory")

    manifest = invoke_formal_operator(
        protocol_module=protocol_module,
        source_module=source_module,
        paths=paths,
    )
    expected_permit = protocol_module.replacement_source_permit_evidence(
        CALIBRATION_LOCK
    )
    source_module.validate_aligned_source_manifest(
        manifest,
        expected_source_permit=expected_permit,
    )
    observed = os.lstat(manifest)
    _require(
        stat.S_ISREG(observed.st_mode)
        and not stat.S_ISLNK(observed.st_mode)
        and stat.S_IMODE(observed.st_mode) == 0o400,
        "aligned-source manifest is not sealed mode 0400",
    )
    print(
        f"ALIGNED_SOURCE_MANIFEST path={manifest} sha256={_sha256_file(manifest)}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise
