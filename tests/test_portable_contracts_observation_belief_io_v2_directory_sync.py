from __future__ import annotations

from pathlib import Path

import pytest

import bayesian_phystwin.observation_belief_io_v2 as io_v2


def test_directory_sync_failure_after_open_is_best_effort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = 17
    closed: list[int] = []

    monkeypatch.setattr(io_v2.os, "open", lambda *_args: descriptor)

    def fail_fsync(_: int) -> None:
        raise OSError("directory fsync is unsupported")

    monkeypatch.setattr(io_v2.os, "fsync", fail_fsync)
    monkeypatch.setattr(io_v2.os, "close", closed.append)

    io_v2._fsync_directory(tmp_path)

    assert closed == [descriptor]
