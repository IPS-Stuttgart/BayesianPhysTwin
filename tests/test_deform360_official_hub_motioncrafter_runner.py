from __future__ import annotations

import importlib.util
import sys
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


def test_isolated_worker_command_binds_paths_and_resume(tmp_path: Path) -> None:
    args = SimpleNamespace(
        job_manifest=tmp_path / "jobs.json",
        processed_root=tmp_path / "processed",
        output_root=tmp_path / "outputs",
        prob4d_root=tmp_path / "prob4d",
        motioncrafter_root=tmp_path / "motioncrafter",
        cache_dir=tmp_path / "cache",
        repository_root=tmp_path / "bpt",
    )
    runner_source = tmp_path / "runner.py"

    command = MODULE._isolated_worker_command(
        args,
        runner_source=runner_source,
        job_id="job-001",
        resume=True,
    )

    assert command[:2] == [sys.executable, str(runner_source)]
    assert command[-3:] == ["--worker-job-id", "job-001", "--resume"]
    for path in (
        args.job_manifest,
        args.processed_root,
        args.output_root,
        args.prob4d_root,
        args.motioncrafter_root,
        args.cache_dir,
        args.repository_root,
    ):
        assert str(path.resolve()) in command
