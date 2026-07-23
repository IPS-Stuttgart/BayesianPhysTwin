"""H2 adapter for the frozen Deform360 source/physical execution engine.

The expensive source alignment, prefix staging, frame-zero reconstruction, and
PhysTwin rollout were already implemented and audited in a separate clean
Bayesian-PhysTwin revision.  This module does not copy or alter those numerical
sources.  It checksum-validates the clean external checkout and installs only
process-local protocol/case adapters for the exact H2 cohort.

No outcome loader is installed.  Every adapted case is deliberately exposed to
the old prediction engine as a calibration-role case so its target-access gate
remains closed; target opening is governed later by this protocol's complete
prediction barrier.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib
from importlib.machinery import SourceFileLoader
import inspect
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import stat
import subprocess
import sys
from types import ModuleType
from typing import Any

from .deform360_adaptive_covariance_confirmation_lock import (
    DATASET_REPOSITORY,
    DATASET_REVISION,
    EXPECTED_STRATA,
    PROTOCOL_ID,
    load_confirmation_cohort_lock,
)


EXTERNAL_EXECUTION_COMMIT = "29091daa1a984ae97a1722011b2039fba708d8ed"
DEFORM360_EXECUTION_COMMIT = "0fe36f0b7a7a917ba62b5f8cee707299a9a4a317"
EXTERNAL_FILE_SHA256: Mapping[str, str] = {
    "scripts/remote/prepare_deform360_bias_aware_source.py": (
        "fddb5a24ab90bbe292f4544d93ed73b5e1a4b7d12283f3e4ddf1c1029ec84046"
    ),
    "scripts/remote/stage_deform360_bias_aware_prediction_prefix.py": (
        "a90578e8a83e5a72388b86f25c6b7b9dee872b75e2919c352e3a3a3ea431e5d6"
    ),
    "scripts/remote/run_deform360_bias_aware_frame_zero.py": (
        "d51410521cd8a894a653d930c2e80257b27099f033fe45951a2d091ec142fdec"
    ),
    "scripts/remote/build_deform360_bias_aware_automatic_twin.py": (
        "27db1f8cf3e35e603e33117d01f0f8c35abb304a064a44e88bf41de8e7eac017"
    ),
    "scripts/remote/run_deform360_bias_aware_physical_prior.py": (
        "e5d81207e89ccaa170a3711708d8ee5ba6b4b181fb08ab17819a35e8c0a9a4ff"
    ),
    "src/bayesian_phystwin/deform360_bias_aware_prospective_artifacts.py": (
        "15ff825257780c13ad0bbb218b894a5b50086be19e3f0914440cc609afef2941"
    ),
    "src/bayesian_phystwin/deform360_bias_aware_prospective_physical.py": (
        "9f87a8aec1bd8bd8ae35865b819bae75d2a1d8d97d4950ca90b96ca90fa21226"
    ),
    "src/bayesian_phystwin/deform360_bias_aware_prospective_staging.py": (
        "259a0ad68c9d0027874c26cfb4bffcaec463c22b3179177827a7388673681a91"
    ),
    "src/bayesian_phystwin/deform360_bias_aware_prospective_uncertainty.py": (
        "a184116e71434d10ac6e0611336705d66fb5e47db6f9b8ecf1ee84ab47ddd57b"
    ),
    "src/bayesian_phystwin/deform360_bias_aware_prospective_protocol.py": (
        "096e645e498c1a5595555f4f3db84c32916a59470fb451532cfc4e6bdc4f77b0"
    ),
    "src/bayesian_phystwin/deform360_bias_aware_prospective_download.py": (
        "50cc8fbf9cae585e18b3a86761828015e83c63b4eeb83fe301856b7001259aab"
    ),
    "src/bayesian_phystwin/deform360_frame_zero_initializer.py": (
        "fdc5fefed2487e2a0f49377a5e518679686e09038c9aff7849c75e8490624f7b"
    ),
    "src/bayesian_phystwin/deform360_frame_zero_depth_initializer.py": (
        "ff92c1531a4e4bd492408f626dc5515d084d5a9a538123c99d32f087b66cc6d8"
    ),
}
DOWNLOAD_ARTIFACT_KIND = "Deform360AdaptiveCovarianceConfirmationDownloadV1"
_DOWNLOAD_MANIFEST_KEYS = {
    "schema_version",
    "artifact_kind",
    "protocol_id",
    "dataset_repository",
    "dataset_revision",
    "implementation_commit_h1",
    "cohort_lock_commit_h2",
    "cohort_lock_artifact_sha256",
    "audio_included",
    "object_count",
    "objects",
    "information_boundary",
    "artifact_sha256",
}
_DOWNLOAD_BOUNDARY = {
    "locked_object_directories_only": True,
    "directory_id_used_as_identity": True,
    "released_metadata_label_used_for_identity": False,
    "target_or_outcome_opened": False,
    "metric_computed": False,
    "pinned_remote_inventory_captured": True,
    "every_non_audio_file_content_hashed": True,
}
_AUDIO_SUFFIXES = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".wma",
}
COHORT_LOCK_REPOSITORY_PATH = (
    "configs/sota/deform360_adaptive_covariance_confirmation_cohort_lock_v1.json"
)
ADAPTER_RUNTIME_REPOSITORY_PATH = (
    "src/bayesian_phystwin/"
    "deform360_adaptive_covariance_confirmation_external_runtime.py"
)
ADAPTER_LOCK_MODULE_REPOSITORY_PATH = (
    "src/bayesian_phystwin/deform360_adaptive_covariance_confirmation_lock.py"
)
ADAPTER_PACKAGE_REPOSITORY_PATH = "src/bayesian_phystwin/__init__.py"

STAGE_SCRIPTS: Mapping[str, str] = {
    "prepare-source": "prepare_deform360_bias_aware_source.py",
    "stage-prefix": "stage_deform360_bias_aware_prediction_prefix.py",
    "frame-zero": "run_deform360_bias_aware_frame_zero.py",
    "automatic-twin": "build_deform360_bias_aware_automatic_twin.py",
    "physical-prior": "run_deform360_bias_aware_physical_prior.py",
}

# Every Bayesian-PhysTwin module reachable from the frozen stage roots at the
# execution commit.  The adapter package is imported first so it can expose the
# confirmation lock; these names must nevertheless resolve from the frozen
# execution checkout rather than silently reusing newer same-named modules.
EXTERNAL_MODULE_SUFFIXES = (
    "bias_aware_belief",
    "cpd_registration",
    "deform360_bias_aware_belief_development",
    "deform360_bias_aware_prospective_artifacts",
    "deform360_bias_aware_prospective_download",
    "deform360_bias_aware_prospective_evaluation",
    "deform360_bias_aware_prospective_physical",
    "deform360_bias_aware_prospective_protocol",
    "deform360_bias_aware_prospective_staging",
    "deform360_bias_aware_prospective_uncertainty",
    "deform360_cpd_diagnostic",
    "deform360_frame_zero_depth_initializer",
    "deform360_frame_zero_initializer",
    "deform360_online_belief_evaluation",
    "deform360_raw_camera_observation",
    "phystwin_official_evaluation",
    "phystwin_online_belief",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _stable_regular_file_record(path: Path) -> dict[str, Any]:
    observed = os.lstat(path)
    _require(
        stat.S_ISREG(observed.st_mode)
        and not stat.S_ISLNK(observed.st_mode)
        and observed.st_nlink == 1,
        f"downloaded path is not a single-link regular file: {path}",
    )
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    digest = hashlib.sha256()
    git_blob = hashlib.sha1(usedforsecurity=False)
    try:
        opened = os.fstat(descriptor)
        _require(
            _stable_file_state(opened) == _stable_file_state(observed),
            f"downloaded file changed while opening: {path}",
        )
        git_blob.update(f"blob {opened.st_size}\0".encode("ascii"))
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
            git_blob.update(block)
        after = os.fstat(descriptor)
        current = os.lstat(path)
        _require(
            _stable_file_state(after)
            == _stable_file_state(opened)
            == _stable_file_state(current),
            f"downloaded file changed while hashing: {path}",
        )
    finally:
        os.close(descriptor)
    return {
        "size_bytes": opened.st_size,
        "sha256": digest.hexdigest(),
        "git_blob_id": git_blob.hexdigest(),
    }


def _read_stable_regular_bytes(path: Path) -> bytes:
    observed = os.lstat(path)
    _require(
        stat.S_ISREG(observed.st_mode)
        and not stat.S_ISLNK(observed.st_mode)
        and observed.st_nlink == 1,
        f"path is not a single-link regular file: {path}",
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
            f"file changed while opening: {path}",
        )
        while block := os.read(descriptor, 1024 * 1024):
            payload.extend(block)
        after = os.fstat(descriptor)
        current = os.lstat(path)
        _require(
            _stable_file_state(after)
            == _stable_file_state(opened)
            == _stable_file_state(current),
            f"file changed while reading: {path}",
        )
    finally:
        os.close(descriptor)
    return bytes(payload)


def _scan_downloaded_object(
    object_root: Path,
    *,
    download_root: Path,
) -> tuple[dict[str, Path], set[str]]:
    observed_root = os.lstat(object_root)
    _require(
        stat.S_ISDIR(observed_root.st_mode) and not stat.S_ISLNK(observed_root.st_mode),
        f"downloaded object root is invalid: {object_root}",
    )
    files: dict[str, Path] = {}
    directories: set[str] = set()
    pending = [object_root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                observed = entry.stat(follow_symlinks=False)
                relative = path.relative_to(download_root).as_posix()
                _require(
                    not stat.S_ISLNK(observed.st_mode),
                    f"downloaded object tree contains a symlink: {relative}",
                )
                if stat.S_ISDIR(observed.st_mode):
                    directories.add(relative)
                    pending.append(path)
                    continue
                _require(
                    stat.S_ISREG(observed.st_mode) and observed.st_nlink == 1,
                    f"downloaded object tree contains a special or linked file: "
                    f"{relative}",
                )
                _require(
                    PurePosixPath(relative).suffix.lower() not in _AUDIO_SUFFIXES,
                    f"downloaded object tree contains forbidden audio: {relative}",
                )
                files[relative] = path
    return files, directories


def _git_output(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _reject_adapter_python_caches(repository: Path) -> None:
    """Require a source-only adapter tree before any production authorization."""

    roots = tuple(
        path for path in (repository / "src", repository / "scripts") if path.exists()
    )
    for root in roots:
        _require(
            root.is_dir() and not root.is_symlink(),
            f"adapter Python source root is invalid: {root}",
        )
        pending = [root]
        while pending:
            directory = pending.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    observed = entry.stat(follow_symlinks=False)
                    _require(
                        not stat.S_ISLNK(observed.st_mode),
                        f"adapter Python tree contains a symlink: {path}",
                    )
                    if stat.S_ISDIR(observed.st_mode):
                        _require(
                            entry.name != "__pycache__",
                            f"adapter Python bytecode cache is forbidden: {path}",
                        )
                        pending.append(path)
                        continue
                    if stat.S_ISREG(observed.st_mode):
                        _require(
                            path.suffix.lower() not in {".pyc", ".pyo"},
                            f"adapter Python bytecode is forbidden: {path}",
                        )
                        continue
                    _require(
                        False,
                        f"adapter Python tree contains a special file: {path}",
                    )


def validate_external_execution_repository(path: str | Path) -> dict[str, Any]:
    """Validate the exact clean numerical execution checkout and source bytes."""

    root = Path(path).absolute()
    _require(
        root.is_dir() and not root.is_symlink() and root.resolve(strict=True) == root,
        "external execution repository is invalid",
    )
    _require(
        _git_output(root, "rev-parse", "HEAD") == EXTERNAL_EXECUTION_COMMIT,
        "external execution commit changed",
    )
    _require(
        not _git_output(root, "status", "--porcelain", "--untracked-files=all"),
        "external execution repository is dirty",
    )
    observed: dict[str, str] = {}
    for relative, expected in EXTERNAL_FILE_SHA256.items():
        source = root / relative
        _require(
            source.is_file() and not source.is_symlink(),
            f"external execution file is missing: {relative}",
        )
        digest = _sha256(source)
        _require(digest == expected, f"external execution file changed: {relative}")
        observed[relative] = digest
    return {
        "commit": EXTERNAL_EXECUTION_COMMIT,
        "file_sha256": observed,
    }


def validate_deform360_execution_repository(path: str | Path) -> dict[str, str]:
    """Bind imports to one clean official Deform360 checkout."""

    root = Path(path).absolute()
    _require(
        root.is_dir() and not root.is_symlink() and root.resolve(strict=True) == root,
        "Deform360 execution repository is invalid",
    )
    _require(
        _git_output(root, "rev-parse", "HEAD") == DEFORM360_EXECUTION_COMMIT,
        "Deform360 execution commit changed",
    )
    _require(
        not _git_output(root, "status", "--porcelain", "--untracked-files=all"),
        "Deform360 execution repository is dirty",
    )
    return {"commit": DEFORM360_EXECUTION_COMMIT}


def validate_two_commit_execution_repository(
    repository: str | Path,
    lock_path: str | Path,
    *,
    h1_commit: str,
    h2_commit: str,
) -> dict[str, Any]:
    """Verify H2 is the one-lock-file child of H1 and binds exact lock bytes."""

    root = Path(repository).absolute()
    lock = Path(lock_path).absolute()
    _require(
        root.is_dir()
        and not root.is_symlink()
        and root.resolve(strict=True) == root
        and lock.is_file()
        and not lock.is_symlink()
        and lock.resolve(strict=True) == lock,
        "two-commit execution repository is invalid",
    )
    expected_lock = root / COHORT_LOCK_REPOSITORY_PATH
    _require(lock == expected_lock, "H2 lock is not at its canonical repository path")
    _require(_git_output(root, "rev-parse", "HEAD") == h2_commit, "adapter is not H2")
    _require(
        not _git_output(root, "status", "--porcelain", "--untracked-files=all"),
        "adapter checkout is dirty",
    )
    parents = _git_output(root, "rev-list", "--parents", "-n", "1", h2_commit).split()
    _require(
        parents == [h2_commit, h1_commit],
        "H2 must be the single-parent direct child of H1",
    )
    changed = _git_output(
        root,
        "diff-tree",
        "--no-commit-id",
        "--name-status",
        "--diff-filter=ACDMRTUXB",
        "-r",
        h2_commit,
    ).splitlines()
    _require(
        changed == [f"A\t{COHORT_LOCK_REPOSITORY_PATH}"],
        "H2 must add only the canonical cohort-lock artifact",
    )
    h1_lock = subprocess.run(
        ["git", "cat-file", "-e", f"{h1_commit}:{COHORT_LOCK_REPOSITORY_PATH}"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    _require(
        h1_lock.returncode != 0,
        "canonical cohort-lock artifact already exists at H1",
    )
    blob = subprocess.run(
        ["git", "show", f"{h2_commit}:{COHORT_LOCK_REPOSITORY_PATH}"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    payload = lock.read_bytes()
    _require(blob == payload, "working H2 lock bytes differ from the committed blob")
    return {
        "implementation_commit_h1": h1_commit,
        "cohort_lock_commit_h2": h2_commit,
        "cohort_lock_repository_path": COHORT_LOCK_REPOSITORY_PATH,
        "cohort_lock_file_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _valid_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and value != "0" * 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _module_repository_path(name: str) -> str:
    parts = name.split(".")
    _require(
        parts[0] == "bayesian_phystwin",
        f"adapter module name is invalid: {name}",
    )
    if len(parts) == 1:
        return ADAPTER_PACKAGE_REPOSITORY_PATH
    return f"src/{'/'.join(parts)}.py"


def _validate_committed_source(
    repository: Path,
    source_value: object,
    repository_path: str,
    *,
    label: str,
    commit: str,
) -> str:
    """Bind one live Python source to the exact tracked blob at a commit."""

    _require(
        isinstance(source_value, (str, os.PathLike)),
        f"{label} has no filesystem source",
    )
    source = Path(source_value).absolute()
    expected = repository / repository_path
    _require(
        source.is_file()
        and not source.is_symlink()
        and source.resolve(strict=True) == source
        and source == expected,
        f"{label} is not executing from the canonical adapter checkout",
    )
    tracked = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}:{repository_path}"],
        cwd=repository,
        check=False,
        capture_output=True,
    )
    _require(
        tracked.returncode == 0,
        f"{label} is not tracked by the declared adapter commit",
    )
    committed = subprocess.run(
        ["git", "show", f"{commit}:{repository_path}"],
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout
    observed = source.read_bytes()
    _require(
        observed == committed,
        f"{label} bytes differ from the declared adapter commit",
    )
    return hashlib.sha256(observed).hexdigest()


def _validate_adapter_python_provenance(
    repository: Path,
    *,
    commit: str,
    entrypoint_file: str | os.PathLike[str],
    entrypoint_repository_path: str,
    require_source_only_cache: bool = False,
) -> dict[str, str]:
    """Reject an outside/dirty Python runtime pointed at a clean checkout."""

    relative = PurePosixPath(entrypoint_repository_path)
    _require(
        bool(entrypoint_repository_path)
        and not relative.is_absolute()
        and ".." not in relative.parts
        and "\\" not in entrypoint_repository_path,
        "adapter entrypoint repository path is invalid",
    )
    observed = {
        entrypoint_repository_path: _validate_committed_source(
            repository,
            entrypoint_file,
            entrypoint_repository_path,
            label="production entrypoint",
            commit=commit,
        ),
        ADAPTER_RUNTIME_REPOSITORY_PATH: _validate_committed_source(
            repository,
            __file__,
            ADAPTER_RUNTIME_REPOSITORY_PATH,
            label="confirmation runtime module",
            commit=commit,
        ),
    }

    package = sys.modules.get("bayesian_phystwin")
    _require(package is not None, "adapter package is not loaded")
    package_paths = tuple(
        Path(path).absolute() for path in getattr(package, "__path__", ())
    )
    expected_package = repository / "src" / "bayesian_phystwin"
    _require(
        package_paths == (expected_package,),
        "adapter package search path escaped the canonical checkout",
    )

    # Every confirmation module already participating in this process must be
    # the ordinary source module in this checkout.  This catches a dirty
    # installed CLI/runtime that merely supplies --adapter-repo pointing at a
    # separate clean H2 tree.
    prefixes = (
        "bayesian_phystwin.deform360_adaptive_covariance_confirmation",
        "bayesian_phystwin.cli.deform360_adaptive_covariance_confirmation",
    )
    required_names = {
        "bayesian_phystwin",
        "bayesian_phystwin.deform360_adaptive_covariance_confirmation_lock",
        __name__,
    }
    participating = {
        name: module
        for name, module in tuple(sys.modules.items())
        if name in required_names or name.startswith(prefixes)
    }
    _require(
        required_names <= set(participating),
        "required adapter runtime modules are not loaded",
    )
    for name, module in sorted(participating.items()):
        repository_path = _module_repository_path(name)
        expected_source = repository / repository_path
        specification = getattr(module, "__spec__", None)
        loader = getattr(specification, "loader", None)
        origin = getattr(specification, "origin", None)
        _require(
            isinstance(loader, SourceFileLoader)
            and Path(loader.path).absolute() == expected_source
            and isinstance(origin, str)
            and Path(origin).absolute() == expected_source,
            f"adapter module was not loaded directly from source: {name}",
        )
        cached = getattr(module, "__cached__", None)
        if require_source_only_cache:
            _require(
                not isinstance(cached, str) or not Path(cached).exists(),
                f"loaded adapter bytecode cache is forbidden: {name}",
            )
        digest = _validate_committed_source(
            repository,
            getattr(module, "__file__", None),
            repository_path,
            label=f"adapter module {name}",
            commit=commit,
        )
        previous = observed.get(repository_path)
        _require(
            previous in {None, digest},
            f"adapter source digest changed during provenance validation: {name}",
        )
        observed[repository_path] = digest
    return dict(sorted(observed.items()))


def validate_confirmation_h2_loaded_runtime(
    repository: str | Path,
    lock_path: str | Path,
    h2_commit: str,
    *,
    expected_h1: str,
    source_file: str | os.PathLike[str],
    source_repository_path: str,
) -> dict[str, Any]:
    """Authorize a direct production API from the exact source-only H2 tree.

    Unlike the entrypoint validator, this function deliberately has no
    direct-caller assertion: it is used by production factory/evaluator APIs.
    It still independently verifies the direct-child/add-only lock history,
    the clean checkout, absence of adapter bytecode caches, and every loaded
    confirmation module against its committed H2 blob.
    """

    _require(_valid_commit(expected_h1), "declared H1 commit is invalid")
    _require(
        _valid_commit(h2_commit) and h2_commit != expected_h1,
        "declared H2 commit is invalid",
    )
    root = Path(repository).absolute()
    binding = validate_two_commit_execution_repository(
        root,
        lock_path,
        h1_commit=expected_h1,
        h2_commit=h2_commit,
    )
    _reject_adapter_python_caches(root)
    # The clean-source check above authorizes this process.  Keep later
    # deferred imports from creating a cache after that authorization.
    sys.dont_write_bytecode = True
    payload = load_confirmation_cohort_lock(
        lock_path,
        expected_implementation_commit_h1=expected_h1,
    )
    provenance = _validate_adapter_python_provenance(
        root,
        commit=h2_commit,
        entrypoint_file=source_file,
        entrypoint_repository_path=source_repository_path,
        require_source_only_cache=True,
    )
    return {
        **binding,
        "cohort_lock_artifact_sha256": payload["artifact_sha256"],
        "python_source_sha256": provenance,
        "source_only_runtime": True,
        "adapter_python_bytecode_cache_absent": True,
    }


def _require_direct_entrypoint_caller(
    entrypoint_file: str | os.PathLike[str],
    *,
    source_bootstrap_file: str | os.PathLike[str] | None = None,
) -> None:
    """Ensure a caller cannot substitute a clean file for its own code file."""

    frame = inspect.currentframe()
    try:
        caller = None if frame is None else frame.f_back
        caller = None if caller is None else caller.f_back
        _require(caller is not None, "production entrypoint caller is unavailable")
        _require(
            Path(caller.f_code.co_filename).absolute()
            == Path(entrypoint_file).absolute(),
            "declared production entrypoint does not match its direct caller",
        )
        if source_bootstrap_file is not None:
            bootstrap_caller = caller.f_back
            _require(
                bootstrap_caller is not None
                and Path(bootstrap_caller.f_code.co_filename).absolute()
                == Path(source_bootstrap_file).absolute(),
                "production CLI was not invoked by its declared source bootstrap",
            )
    finally:
        del frame


def validate_confirmation_h1_lock_generation_entrypoint(
    repository: str | Path,
    output_path: str | Path,
    h1_commit: str,
    *,
    entrypoint_file: str | os.PathLike[str],
    entrypoint_repository_path: str,
) -> dict[str, Any]:
    """Authorize metadata-only lock creation from the exact clean H1 tree."""

    _require_direct_entrypoint_caller(entrypoint_file)
    root = Path(repository).absolute()
    output = Path(output_path).absolute()
    _require(_valid_commit(h1_commit), "declared H1 commit is invalid")
    _require(
        root.is_dir() and not root.is_symlink() and root.resolve(strict=True) == root,
        "H1 adapter repository is invalid",
    )
    _require(_git_output(root, "rev-parse", "HEAD") == h1_commit, "adapter is not H1")
    h1_lock = subprocess.run(
        ["git", "cat-file", "-e", f"{h1_commit}:{COHORT_LOCK_REPOSITORY_PATH}"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    _require(
        h1_lock.returncode != 0,
        "canonical cohort-lock artifact already exists at H1",
    )
    _require(
        not _git_output(root, "status", "--porcelain", "--untracked-files=all"),
        "H1 adapter checkout is dirty",
    )
    expected_output = root / COHORT_LOCK_REPOSITORY_PATH
    _require(
        output == expected_output,
        "cohort lock output is not the canonical repository path",
    )
    _require(
        output.parent.is_dir()
        and not output.parent.is_symlink()
        and output.parent.resolve(strict=True) == output.parent,
        "canonical cohort lock parent is invalid",
    )
    _require(
        not output.exists() and not output.is_symlink(),
        "canonical cohort lock output must be absent at H1",
    )
    provenance = _validate_adapter_python_provenance(
        root,
        commit=h1_commit,
        entrypoint_file=entrypoint_file,
        entrypoint_repository_path=entrypoint_repository_path,
    )
    return {
        "implementation_commit_h1": h1_commit,
        "cohort_lock_repository_path": COHORT_LOCK_REPOSITORY_PATH,
        "python_source_sha256": provenance,
    }


def validate_confirmation_h2_production_entrypoint(
    repository: str | Path,
    lock_path: str | Path,
    h2_commit: str,
    *,
    expected_h1: str,
    entrypoint_file: str | os.PathLike[str],
    entrypoint_repository_path: str,
    source_bootstrap_file: str | os.PathLike[str] | None = None,
    source_bootstrap_repository_path: str | None = None,
) -> dict[str, Any]:
    """Authorize one production process from the exact clean canonical H2."""

    _require(
        (source_bootstrap_file is None) == (source_bootstrap_repository_path is None),
        "source bootstrap binding is incomplete",
    )
    _require_direct_entrypoint_caller(
        entrypoint_file,
        source_bootstrap_file=source_bootstrap_file,
    )
    _require(_valid_commit(expected_h1), "declared H1 commit is invalid")
    _require(
        _valid_commit(h2_commit) and h2_commit != expected_h1,
        "declared H2 commit is invalid",
    )
    result = validate_confirmation_h2_loaded_runtime(
        repository,
        lock_path,
        h2_commit,
        expected_h1=expected_h1,
        source_file=entrypoint_file,
        source_repository_path=entrypoint_repository_path,
    )
    if source_bootstrap_file is not None:
        bootstrap_digest = _validate_committed_source(
            Path(repository).absolute(),
            source_bootstrap_file,
            str(source_bootstrap_repository_path),
            label="production source bootstrap",
            commit=h2_commit,
        )
        result["python_source_sha256"][str(source_bootstrap_repository_path)] = (
            bootstrap_digest
        )
    return result


def load_confirmation_execution_protocol(path: str | Path) -> dict[str, Any]:
    """Expose the H2 lock through the normalized interface used by old runners."""

    lock = load_confirmation_cohort_lock(path)
    calibration: dict[str, dict[str, tuple[int, ...]]] = {
        stratum: {} for stratum in EXPECTED_STRATA
    }
    for stratum in EXPECTED_STRATA:
        for record in lock["cohort"][stratum]:
            calibration[stratum][record["object_id"]] = tuple(
                int(episode["episode_id"]) for episode in record["episodes"]
            )
    return {
        "payload": lock,
        "config": {
            "protocol_id": PROTOCOL_ID,
            "target_access_authorized": False,
            "source_execution_role": "target-free-calibration-compatible-adapter",
        },
        "config_sha256": lock["artifact_sha256"],
        "calibration_cohort": calibration,
        "target_cohort": {stratum: {} for stratum in EXPECTED_STRATA},
    }


def confirmation_case_records(
    protocol_path: str | Path,
    *,
    role: str | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return all exact H2 cases as target-closed execution-role records."""

    _require(role in {None, "calibration"}, "confirmation execution has no target role")
    protocol = load_confirmation_execution_protocol(protocol_path)
    rows: list[dict[str, Any]] = []
    for stratum in EXPECTED_STRATA:
        for object_id, episodes in protocol["calibration_cohort"][stratum].items():
            for episode_id in episodes:
                rows.append(
                    {
                        "case": f"{object_id}-ep{episode_id:04d}",
                        "object_id": object_id,
                        "episode_id": int(episode_id),
                        "episode_key": f"{object_id}/{episode_id}",
                        "stratum": stratum,
                        "role": "calibration",
                    }
                )
    _require(
        len(rows) == len({row["case"] for row in rows}) == 34,
        "confirmation execution case panel changed",
    )
    return tuple(rows)


def confirmation_case_record(
    protocol_path: str | Path,
    *,
    object_id: str,
    episode_id: int,
) -> dict[str, Any]:
    matches = [
        row
        for row in confirmation_case_records(protocol_path)
        if row["object_id"] == object_id and row["episode_id"] == int(episode_id)
    ]
    _require(len(matches) == 1, "object/episode is outside the exact H2 lock")
    return matches[0]


@dataclass(frozen=True)
class ConfirmationDownloadPlan:
    """Minimal plan interface consumed by the frozen source-preparation stage."""

    object_ids: tuple[str, ...]


def confirmation_download_plan(protocol_path: str | Path) -> ConfirmationDownloadPlan:
    records = confirmation_case_records(protocol_path)
    objects = tuple(dict.fromkeys(record["object_id"] for record in records))
    _require(len(objects) == 17, "confirmation download plan changed")
    return ConfirmationDownloadPlan(objects)


def _canonical_sha256(payload: Mapping[str, Any], *, key: str) -> str:
    value = dict(payload)
    value.pop(key, None)
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _value_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


_DOWNLOAD_OBJECT_KEYS = {
    "object_id",
    "selected_episode_ids",
    "released_metadata_object_label",
    "directory_id_is_identity",
    "file_count",
    "total_bytes",
    "metadata_sha256",
    "remote_inventory_sha256",
    "content_inventory_sha256",
    "files",
}
_DOWNLOAD_FILE_KEYS = {
    "path",
    "remote_blob_id",
    "remote_lfs_sha256",
    "size_bytes",
    "sha256",
}


def _valid_lower_hex(value: object, *, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_download_object_record(row: Mapping[str, Any]) -> None:
    _require(set(row) == _DOWNLOAD_OBJECT_KEYS, "download object fields changed")
    object_id = row.get("object_id")
    episodes = row.get("selected_episode_ids")
    _require(
        isinstance(object_id, str)
        and object_id
        and "/" not in object_id
        and "\\" not in object_id
        and object_id not in {".", ".."},
        "download object identity is invalid",
    )
    _require(
        isinstance(episodes, list)
        and len(episodes) == len(set(episodes)) == 2
        and all(type(episode) is int and 0 <= episode < 10 for episode in episodes),
        f"download selected episodes changed: {object_id}",
    )
    _require(
        isinstance(row.get("released_metadata_object_label"), str)
        and bool(row["released_metadata_object_label"].strip())
        and row.get("directory_id_is_identity") is True,
        f"download object identity fields changed: {object_id}",
    )
    files = row.get("files")
    _require(
        isinstance(files, list) and bool(files),
        f"download file inventory is empty: {object_id}",
    )
    normalized: list[Mapping[str, Any]] = []
    paths: list[str] = []
    for record in files:
        _require(
            isinstance(record, Mapping) and set(record) == _DOWNLOAD_FILE_KEYS,
            f"download file fields changed: {object_id}",
        )
        relative = record.get("path")
        candidate = PurePosixPath(relative) if isinstance(relative, str) else None
        _require(
            isinstance(relative, str)
            and candidate is not None
            and not candidate.is_absolute()
            and ".." not in candidate.parts
            and "\\" not in relative
            and len(candidate.parts) >= 3
            and candidate.parts[:2] == ("raw", object_id),
            f"download file path escaped its object: {object_id}",
        )
        _require(
            candidate.suffix.lower() not in _AUDIO_SUFFIXES,
            f"download manifest contains forbidden audio: {relative}",
        )
        _require(
            _valid_lower_hex(record.get("remote_blob_id"), length=40)
            and (
                record.get("remote_lfs_sha256") is None
                or _valid_lower_hex(record.get("remote_lfs_sha256"), length=64)
            )
            and type(record.get("size_bytes")) is int
            and record["size_bytes"] >= 0
            and _valid_lower_hex(record.get("sha256"), length=64),
            f"download file binding is invalid: {relative}",
        )
        normalized.append(record)
        paths.append(relative)
    _require(
        paths == sorted(paths) and len(paths) == len(set(paths)),
        f"download file ordering or uniqueness changed: {object_id}",
    )
    metadata_relative = f"raw/{object_id}/metadata.json"
    metadata_records = [
        record for record in normalized if record["path"] == metadata_relative
    ]
    _require(
        len(metadata_records) == 1,
        f"download metadata inventory changed: {object_id}",
    )
    remote_inventory = [
        {
            "path": record["path"],
            "blob_id": record["remote_blob_id"],
            "lfs_sha256": record["remote_lfs_sha256"],
        }
        for record in normalized
    ]
    _require(
        _valid_lower_hex(row.get("remote_inventory_sha256"), length=64)
        and row["remote_inventory_sha256"] == _value_sha256(remote_inventory),
        f"download remote blob/path binding changed: {object_id}",
    )
    _require(
        _valid_lower_hex(row.get("content_inventory_sha256"), length=64)
        and row["content_inventory_sha256"] == _value_sha256(normalized),
        f"download content inventory binding changed: {object_id}",
    )
    _require(
        type(row.get("file_count")) is int
        and row["file_count"] == len(normalized)
        and type(row.get("total_bytes")) is int
        and row["total_bytes"]
        == sum(int(record["size_bytes"]) for record in normalized)
        and _valid_lower_hex(row.get("metadata_sha256"), length=64)
        and row["metadata_sha256"] == metadata_records[0]["sha256"],
        f"download file summary changed: {object_id}",
    )


def validate_confirmation_download_manifest(
    path: Path,
    *,
    protocol_config_sha256: str,
    object_id: str,
    episode_id: int,
    metadata_path: Path,
    expected_h1: str,
    expected_h2: str,
) -> dict[str, Any]:
    """Replay one H2 object's exact inventory before its media can be consumed."""

    try:
        payload = json.loads(_read_stable_regular_bytes(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("confirmation download manifest is not UTF-8 JSON") from error
    _require(isinstance(payload, Mapping), "confirmation download manifest is invalid")
    _require(
        set(payload) == _DOWNLOAD_MANIFEST_KEYS
        and payload.get("schema_version") == 2
        and payload.get("artifact_kind") == DOWNLOAD_ARTIFACT_KIND
        and payload.get("protocol_id") == PROTOCOL_ID
        and payload.get("implementation_commit_h1") == expected_h1
        and payload.get("cohort_lock_commit_h2") == expected_h2
        and payload.get("cohort_lock_artifact_sha256") == protocol_config_sha256
        and payload.get("dataset_repository") == DATASET_REPOSITORY
        and payload.get("dataset_revision") == DATASET_REVISION,
        "confirmation download manifest is incompatible",
    )
    _require(
        payload.get("artifact_sha256")
        == _canonical_sha256(payload, key="artifact_sha256"),
        "confirmation download manifest checksum changed",
    )
    all_rows = payload.get("objects")
    _require(
        isinstance(all_rows, list)
        and payload.get("object_count") == len(all_rows) == 17
        and payload.get("audio_included") is False,
        "confirmation download object panel changed",
    )
    for row in all_rows:
        _require(
            isinstance(row, Mapping),
            "confirmation download object record is invalid",
        )
        _validate_download_object_record(row)
    object_ids = [row["object_id"] for row in all_rows]
    _require(
        len(object_ids) == len(set(object_ids)),
        "confirmation download object identities are not unique",
    )
    rows = [row for row in all_rows if row["object_id"] == object_id]
    _require(len(rows) == 1, "confirmation download object changed")
    _require(
        episode_id in rows[0].get("selected_episode_ids", []),
        "confirmation download omitted the selected episode",
    )
    boundary = payload.get("information_boundary")
    _require(
        boundary == _DOWNLOAD_BOUNDARY,
        "confirmation download content boundary changed",
    )

    selected = rows[0]
    metadata = Path(metadata_path).absolute()
    object_root = metadata.parent
    _require(
        metadata.name == "metadata.json"
        and object_root.name == object_id
        and object_root.parent.name == "raw"
        and object_root.parent.parent.is_dir()
        and object_root.parent.parent.resolve(strict=True) == object_root.parent.parent,
        "downloaded object path is noncanonical",
    )
    download_root = object_root.parent.parent
    observed_files, observed_directories = _scan_downloaded_object(
        object_root,
        download_root=download_root,
    )
    expected_files = {str(record["path"]): record for record in selected["files"]}
    missing = sorted(set(expected_files) - set(observed_files))
    extras = sorted(set(observed_files) - set(expected_files))
    _require(not missing, f"downloaded files are missing: {missing}")
    _require(not extras, f"downloaded object has extra files: {extras}")
    expected_directories: set[str] = set()
    object_prefix = PurePosixPath("raw") / object_id
    for relative in expected_files:
        parent = PurePosixPath(relative).parent
        while parent != object_prefix:
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    _require(
        observed_directories <= expected_directories,
        "downloaded object has extra directories: "
        f"{sorted(observed_directories - expected_directories)}",
    )
    for relative, expected in expected_files.items():
        observed = _stable_regular_file_record(observed_files[relative])
        _require(
            observed["size_bytes"] == expected["size_bytes"]
            and observed["sha256"] == expected["sha256"],
            f"downloaded file content changed: {relative}",
        )
        if expected["remote_lfs_sha256"] is not None:
            _require(
                observed["sha256"] == expected["remote_lfs_sha256"],
                f"downloaded LFS content differs from pinned remote: {relative}",
            )
        else:
            _require(
                observed["git_blob_id"] == expected["remote_blob_id"],
                f"downloaded Git blob differs from pinned remote: {relative}",
            )
    _require(
        selected["metadata_sha256"]
        == expected_files[f"raw/{object_id}/metadata.json"]["sha256"],
        "downloaded object metadata changed",
    )
    return payload


def _set(
    module: ModuleType,
    name: str,
    value: Any,
    changes: list[tuple[ModuleType, str, Any]],
) -> None:
    if hasattr(module, name):
        changes.append((module, name, getattr(module, name)))
        setattr(module, name, value)


def _external_modules(repository: Path) -> dict[str, ModuleType]:
    import bayesian_phystwin

    external_package = repository / "src" / "bayesian_phystwin"
    package_path = str(external_package)
    _require(
        package_path in bayesian_phystwin.__path__,
        "external execution package path is not active",
    )
    names = {
        "artifacts": "deform360_bias_aware_prospective_artifacts",
        "physical": "deform360_bias_aware_prospective_physical",
        "uncertainty": "deform360_bias_aware_prospective_uncertainty",
    }
    result: dict[str, ModuleType] = {}
    for key, suffix in names.items():
        module = importlib.import_module(f"bayesian_phystwin.{suffix}")
        source = Path(str(module.__file__)).resolve(strict=True)
        _require(
            source == (external_package / f"{suffix}.py").resolve(strict=True),
            f"external {key} module provenance changed",
        )
        result[key] = module
    return result


def validate_external_module_provenance(
    repository: str | Path,
) -> dict[str, str]:
    """Verify that every loaded frozen dependency came from the exact checkout."""

    external_package = Path(repository).absolute() / "src" / "bayesian_phystwin"
    observed: dict[str, str] = {}
    for suffix in EXTERNAL_MODULE_SUFFIXES:
        name = f"bayesian_phystwin.{suffix}"
        module = sys.modules.get(name)
        if module is None:
            continue
        source_value = getattr(module, "__file__", None)
        _require(
            isinstance(source_value, str), f"external module has no source: {name}"
        )
        source = Path(source_value).resolve(strict=True)
        _require(
            source == (external_package / f"{suffix}.py").resolve(strict=True),
            f"external module provenance changed: {name}",
        )
        observed[name] = _sha256(source)
    for required in (
        "bayesian_phystwin.deform360_bias_aware_prospective_artifacts",
        "bayesian_phystwin.deform360_bias_aware_prospective_physical",
        "bayesian_phystwin.deform360_bias_aware_prospective_uncertainty",
    ):
        _require(
            required in observed, f"required external module is absent: {required}"
        )
    return observed


@contextmanager
def activate_confirmation_external_runtime(
    repository: str | Path,
) -> Iterator[dict[str, ModuleType]]:
    """Install only target-free H2 identity hooks for one isolated process."""

    root = Path(repository).absolute()
    validate_external_execution_repository(root)
    import bayesian_phystwin

    external_package = root / "src" / "bayesian_phystwin"
    package_path = str(external_package)
    original_package_path = list(bayesian_phystwin.__path__)
    displaced_modules: dict[str, ModuleType] = {}
    displaced_attributes: dict[str, object] = {}
    for suffix in EXTERNAL_MODULE_SUFFIXES:
        name = f"bayesian_phystwin.{suffix}"
        loaded = sys.modules.pop(name, None)
        if loaded is not None:
            displaced_modules[name] = loaded
        if hasattr(bayesian_phystwin, suffix):
            displaced_attributes[suffix] = getattr(bayesian_phystwin, suffix)
            delattr(bayesian_phystwin, suffix)
    bayesian_phystwin.__path__.insert(0, package_path)
    changes: list[tuple[ModuleType, str, Any]] = []
    try:
        modules = _external_modules(root)
        validate_external_module_provenance(root)
        for module in modules.values():
            _set(module, "PROTOCOL_ID", PROTOCOL_ID, changes)
        artifacts = modules["artifacts"]
        _set(
            artifacts,
            "load_bias_aware_prospective_protocol",
            load_confirmation_execution_protocol,
            changes,
        )
        _set(artifacts, "prospective_case_records", confirmation_case_records, changes)
        _set(artifacts, "prospective_case_record", confirmation_case_record, changes)
        yield modules
    finally:
        for module, name, original in reversed(changes):
            setattr(module, name, original)
        for suffix in EXTERNAL_MODULE_SUFFIXES:
            name = f"bayesian_phystwin.{suffix}"
            loaded = sys.modules.get(name)
            source_value = getattr(loaded, "__file__", None)
            if isinstance(source_value, str):
                source = Path(source_value).resolve(strict=True)
                if external_package in source.parents:
                    sys.modules.pop(name, None)
        sys.modules.update(displaced_modules)
        for suffix in EXTERNAL_MODULE_SUFFIXES:
            if hasattr(bayesian_phystwin, suffix):
                delattr(bayesian_phystwin, suffix)
        for suffix, value in displaced_attributes.items():
            setattr(bayesian_phystwin, suffix, value)
        bayesian_phystwin.__path__[:] = original_package_path


def patch_confirmation_stage_module(
    module: ModuleType,
    *,
    stage: str,
    adapter_repository: Path,
    execution_repository: Path,
    deform360_repository: Path,
    lock_path: Path,
    h2_commit: str,
    expected_h1: str,
) -> None:
    """Patch aliases imported by one checksum-bound external prediction stage."""

    _require(stage in STAGE_SCRIPTS, "unknown confirmation execution stage")
    common = {
        "PROTOCOL_ID": PROTOCOL_ID,
        "load_bias_aware_prospective_protocol": (load_confirmation_execution_protocol),
        "prospective_case_record": confirmation_case_record,
        "prospective_case_records": confirmation_case_records,
    }
    for name, value in common.items():
        if hasattr(module, name):
            setattr(module, name, value)
    if stage == "prepare-source":
        module.bias_aware_prospective_download_plan = confirmation_download_plan

        def validate_download_manifest(
            path: Path,
            *,
            protocol_config_sha256: str,
            object_id: str,
            episode_id: int,
            metadata_path: Path,
        ) -> dict[str, Any]:
            return validate_confirmation_download_manifest(
                path,
                protocol_config_sha256=protocol_config_sha256,
                object_id=object_id,
                episode_id=episode_id,
                metadata_path=metadata_path,
                expected_h1=expected_h1,
                expected_h2=h2_commit,
            )

        module._validate_download_manifest = validate_download_manifest
    if stage == "physical-prior":
        original_run_logged = module._run_logged

        def run_logged(command: Sequence[str], **kwargs: Any):
            rewritten = list(command)
            if len(rewritten) >= 2 and Path(rewritten[1]).name == (
                "build_deform360_bias_aware_automatic_twin.py"
            ):
                wrapper = (
                    adapter_repository
                    / "scripts"
                    / "remote"
                    / "run_deform360_adaptive_confirmation_external_stage.py"
                )
                rewritten = [
                    rewritten[0],
                    str(wrapper),
                    "--adapter-repo",
                    str(adapter_repository),
                    "--execution-repo",
                    str(execution_repository),
                    "--deform360-repo",
                    str(deform360_repository),
                    "--lock",
                    str(lock_path),
                    "--h2-commit",
                    h2_commit,
                    "--expected-h1",
                    expected_h1,
                    "--stage",
                    "automatic-twin",
                    *rewritten[2:],
                ]
                environment = dict(kwargs.get("env", os.environ))
                environment["PYTHONPATH"] = os.pathsep.join(
                    (
                        str(adapter_repository / "src"),
                        str(execution_repository / "src"),
                        environment.get("PYTHONPATH", ""),
                    )
                ).rstrip(os.pathsep)
                kwargs["env"] = environment
            return original_run_logged(rewritten, **kwargs)

        module._run_logged = run_logged


__all__ = [
    "ADAPTER_LOCK_MODULE_REPOSITORY_PATH",
    "ADAPTER_PACKAGE_REPOSITORY_PATH",
    "ADAPTER_RUNTIME_REPOSITORY_PATH",
    "DOWNLOAD_ARTIFACT_KIND",
    "DEFORM360_EXECUTION_COMMIT",
    "COHORT_LOCK_REPOSITORY_PATH",
    "EXTERNAL_EXECUTION_COMMIT",
    "EXTERNAL_FILE_SHA256",
    "EXTERNAL_MODULE_SUFFIXES",
    "STAGE_SCRIPTS",
    "activate_confirmation_external_runtime",
    "confirmation_case_record",
    "confirmation_case_records",
    "confirmation_download_plan",
    "load_confirmation_execution_protocol",
    "patch_confirmation_stage_module",
    "validate_confirmation_download_manifest",
    "validate_confirmation_h1_lock_generation_entrypoint",
    "validate_confirmation_h2_production_entrypoint",
    "validate_external_module_provenance",
    "validate_deform360_execution_repository",
    "validate_external_execution_repository",
    "validate_two_commit_execution_repository",
]
