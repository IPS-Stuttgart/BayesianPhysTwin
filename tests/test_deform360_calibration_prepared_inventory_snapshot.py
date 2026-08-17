"""Descriptor-bound snapshot regressions for the prepared-source inventory."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import test_deform360_calibration_prepared_inventory as inventory_tests

CLI = inventory_tests.CLI


def _replace_path_after_open(
    monkeypatch: pytest.MonkeyPatch,
    *,
    target: Path,
    replacement: bytes,
) -> None:
    original_open = CLI.os.open
    replaced = False

    def open_then_replace(path: object, flags: int, *args: object) -> int:
        nonlocal replaced
        descriptor = original_open(path, flags, *args)
        if not replaced and Path(path) == target:
            staged = target.with_name(f".{target.name}.replacement")
            staged.write_bytes(replacement)
            os.replace(staged, target)
            replaced = True
        return descriptor

    monkeypatch.setattr(CLI.os, "open", open_then_replace)


def test_stable_file_record_hashes_the_opened_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = b"reviewed calibration bytes"
    replacement = b"replacement bytes"
    source = tmp_path / "payload.bin"
    source.write_bytes(original)
    _replace_path_after_open(
        monkeypatch,
        target=source,
        replacement=replacement,
    )

    with tempfile.SpooledTemporaryFile(mode="w+b") as snapshot:
        record = CLI._read_stable_file(
            source,
            root=tmp_path.resolve(),
            name="payload",
            sink=snapshot,
        )
        assert snapshot.read() == original

    assert record["sha256"] == hashlib.sha256(original).hexdigest()
    assert record["byte_count"] == len(original)
    assert source.read_bytes() == replacement


def test_npy_contract_is_parsed_from_the_verified_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "tactile.npy"
    np.save(source, np.arange(12, dtype=np.float32).reshape(3, 4), allow_pickle=False)
    original = source.read_bytes()
    staged = tmp_path / "replacement.npy"
    np.save(staged, np.arange(5, dtype=np.int64), allow_pickle=False)
    replacement = staged.read_bytes()
    staged.unlink()
    _replace_path_after_open(
        monkeypatch,
        target=source,
        replacement=replacement,
    )

    record = CLI._npy_record(
        source,
        root=tmp_path.resolve(),
        expected_sha256=hashlib.sha256(original).hexdigest(),
    )

    assert record["shape"] == [3, 4]
    assert record["dtype"] == np.dtype(np.float32).str
    assert source.read_bytes() == replacement


def test_npz_contract_is_parsed_from_the_verified_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "robot.npz"
    np.savez(
        source,
        actions=np.zeros((2, 5, 3)),
        T_worlds=np.tile(np.eye(4), (2, 1, 1)),
        openings=np.zeros(2),
        bimanual=np.asarray(False),
    )
    original = source.read_bytes()
    staged = tmp_path / "replacement.npz"
    np.savez(staged, unexpected=np.ones(1))
    replacement = staged.read_bytes()
    staged.unlink()
    _replace_path_after_open(
        monkeypatch,
        target=source,
        replacement=replacement,
    )

    record = CLI._npz_record(
        source,
        root=tmp_path.resolve(),
        expected_sha256=hashlib.sha256(original).hexdigest(),
    )

    assert set(record["arrays"]) >= {"actions", "T_worlds", "openings", "bimanual"}
    assert source.read_bytes() == replacement
