"""Bind the narrow Viser process-churn guard used by held-v8.3 workers."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import stat
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


UPSTREAM_CLIENT_AUTOBUILD_SHA256 = (
    "3c657a8baa49498e372234f05e4d9baf4c8a45b1aead45276e881987bf9da506"
)
ARTIFACT_KIND = "Deform360HeldViserProcessChurnGuardV1"
_INSTALLED_EVIDENCE: dict[str, Any] | None = None
_INSTALLED_HELPER: Any = None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _artifact(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["artifact_sha256"] = hashlib.sha256(_canonical_bytes(value)).hexdigest()
    return result


def _sha256_file(path: Path) -> str:
    source = Path(os.path.abspath(os.fspath(path)))
    before = os.lstat(source)
    _require(
        stat.S_ISREG(before.st_mode) and not stat.S_ISLNK(before.st_mode),
        "Viser source is not a regular file",
    )
    descriptor = os.open(
        source,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        opened = os.fstat(descriptor)
        _require(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns),
            "Viser source changed while opening",
        )
        digest = hashlib.sha256()
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
        after = os.fstat(descriptor)
        _require(
            (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            "Viser source changed while hashing",
        )
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _bound_file(path: Path) -> dict[str, Any]:
    source = Path(os.path.abspath(os.fspath(path)))
    return {
        "path": os.fspath(source),
        "sha256": _sha256_file(source),
        "size_bytes": os.lstat(source).st_size,
    }


def _check_viser_yarn_running(
    processes: Iterable[Any],
    *,
    ignored_exceptions: Sequence[type[BaseException]],
) -> bool:
    """Replicate the pinned helper while tolerating exited process handles."""

    ignored = tuple(ignored_exceptions)
    for process in processes:
        try:
            cwd = Path(process.cwd()).as_posix()
            command = tuple(str(part) for part in process.cmdline())
        except ignored:
            continue
        if cwd.endswith("viser/client") and any(
            part.endswith("yarn") or part.endswith("yarn.js") for part in command
        ):
            return True
    return False


def install_viser_process_churn_guard() -> Mapping[str, Any]:
    """Install the byte-bound process-enumeration guard before trainer import."""

    global _INSTALLED_EVIDENCE, _INSTALLED_HELPER

    if _INSTALLED_EVIDENCE is not None:
        autobuild = importlib.import_module("viser._client_autobuild")
        _require(
            getattr(autobuild, "_check_viser_yarn_running", None)
            is _INSTALLED_HELPER,
            "installed Viser process guard was replaced",
        )
        return dict(_INSTALLED_EVIDENCE)

    trainer_module = "deform360.processing.reconstruct_stage"
    _require(
        trainer_module not in sys.modules,
        "Viser process guard must be installed before original trainer import",
    )
    autobuild = importlib.import_module("viser._client_autobuild")
    psutil = importlib.import_module("psutil")
    source = Path(str(autobuild.__file__)).resolve(strict=True)
    source_sha256 = _sha256_file(source)
    _require(
        source_sha256 == UPSTREAM_CLIENT_AUTOBUILD_SHA256,
        "pinned Viser client-autobuild source changed",
    )
    original = getattr(autobuild, "_check_viser_yarn_running", None)
    _require(callable(original), "pinned Viser yarn helper is absent")
    ignored_exceptions = (
        psutil.AccessDenied,
        psutil.ZombieProcess,
        psutil.NoSuchProcess,
    )

    def guarded_check() -> bool:
        return _check_viser_yarn_running(
            psutil.process_iter(),
            ignored_exceptions=ignored_exceptions,
        )

    guarded_check.__name__ = "_check_viser_yarn_running"
    guarded_check.__module__ = str(autobuild.__name__)
    setattr(autobuild, "_check_viser_yarn_running", guarded_check)
    _INSTALLED_HELPER = guarded_check
    _INSTALLED_EVIDENCE = _artifact(
        {
            "schema_version": 1,
            "artifact_kind": ARTIFACT_KIND,
            "guard_source": _bound_file(Path(__file__).resolve(strict=True)),
            "upstream_client_autobuild_source": _bound_file(source),
            "ignored_process_exceptions": [
                "AccessDenied",
                "ZombieProcess",
                "NoSuchProcess",
            ],
            "original_helper_module": str(getattr(original, "__module__", "")),
            "original_helper_name": str(getattr(original, "__name__", "")),
            "guard_installed_before_original_trainer_import": True,
            "target_or_outcome_path_accessed": False,
        }
    )
    return dict(_INSTALLED_EVIDENCE)


__all__ = [
    "ARTIFACT_KIND",
    "UPSTREAM_CLIENT_AUTOBUILD_SHA256",
    "install_viser_process_churn_guard",
]
