from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import BinaryIO

import pytest

import bayesian_phystwin.legacy_artifacts as legacy_artifacts
from bayesian_phystwin.legacy_artifacts import (
    load_trusted_legacy_phystwin_pickle,
)


def _write_pickle(path: Path, value: object) -> str:
    payload = pickle.dumps(value)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_deserialization_uses_verified_snapshot_when_source_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "artifact.pkl"
    trusted = {"value": "trusted"}
    replacement = {"value": "replacement"}
    digest = _write_pickle(path, trusted)
    original_load = legacy_artifacts.pickle.load

    def replace_source_then_load(stream: BinaryIO) -> object:
        _write_pickle(path, replacement)
        return original_load(stream)

    monkeypatch.setattr(
        legacy_artifacts.pickle,
        "load",
        replace_source_then_load,
    )

    loaded = load_trusted_legacy_phystwin_pickle(
        path,
        expected_sha256=digest,
        artifact_kind="mapping",
        required_keys=("value",),
    )

    assert loaded == trusted
    assert pickle.loads(path.read_bytes()) == replacement


def test_verified_snapshot_can_spool_to_disk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "artifact.pkl"
    trusted = {"payload": "x" * 4096}
    digest = _write_pickle(path, trusted)
    monkeypatch.setattr(legacy_artifacts, "_SNAPSHOT_MEMORY_LIMIT_BYTES", 1)

    loaded = load_trusted_legacy_phystwin_pickle(
        path,
        expected_sha256=digest,
        artifact_kind="mapping",
        required_keys=("payload",),
    )

    assert loaded == trusted
