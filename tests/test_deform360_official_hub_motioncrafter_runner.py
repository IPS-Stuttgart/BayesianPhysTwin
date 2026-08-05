from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "remote"
    / "run_deform360_official_hub_motioncrafter_jobs.py"
)
SPEC = importlib.util.spec_from_file_location(
    "_deform360_official_hub_motioncrafter_runner",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class _Runner:
    def __init__(self, *, result: Path | None = None, error: Exception | None = None):
        self.result = result
        self.error = error
        self.resume_calls: list[bool] = []

    def run(self, *, resume: bool) -> Path:
        self.resume_calls.append(resume)
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def test_release_job_memory_clears_cuda_cache() -> None:
    calls: list[str] = []
    torch_module = SimpleNamespace(
        cuda=SimpleNamespace(empty_cache=lambda: calls.append("empty_cache"))
    )

    MODULE._release_job_memory(torch_module=torch_module)

    assert calls == ["empty_cache"]


def test_memory_barrier_runs_after_success(monkeypatch: pytest.MonkeyPatch) -> None:
    result = Path("predictions.json")
    runner = _Runner(result=result)
    releases: list[str] = []
    monkeypatch.setattr(MODULE, "_release_job_memory", lambda: releases.append("release"))

    assert MODULE._run_with_memory_barrier(runner, resume=True) == result
    assert runner.resume_calls == [True]
    assert releases == ["release"]


def test_memory_barrier_runs_after_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _Runner(error=RuntimeError("inference failed"))
    releases: list[str] = []
    monkeypatch.setattr(MODULE, "_release_job_memory", lambda: releases.append("release"))

    with pytest.raises(RuntimeError, match="inference failed"):
        MODULE._run_with_memory_barrier(runner, resume=False)

    assert runner.resume_calls == [False]
    assert releases == ["release"]
