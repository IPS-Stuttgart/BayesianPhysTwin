from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bayesian_phystwin_experiments.dlolab_regret_artifacts import read_record

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "slingshot_certified_guard_v2_runner",
    ROOT / "scripts/remote/run_dlolab_slingshot_certified_guard_v2.py",
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _lock() -> dict[str, Any]:
    return {
        "artifact_id": "lock",
        "controls": np.zeros((8, 3, 6), dtype=np.float64).tolist(),
        "assets_root": "/unused",
        "runtime": {},
    }


def test_alternate_root_is_rejected_before_parent_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "OUTPUT_ROOT", tmp_path / "registered")
    monkeypatch.setattr(
        runner,
        "load_parent",
        lambda: pytest.fail("parent read before output-root rejection"),
    )
    with pytest.raises(ValueError, match="registered one-attempt root"):
        runner.freeze(tmp_path / "alternate")


def test_future_requires_barrier_before_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    monkeypatch.setattr(runner, "validate_lock", lambda path: _lock())
    monkeypatch.setattr(
        runner,
        "_barrier_records",
        lambda *args: (_ for _ in ()).throw(ValueError("missing decision barrier")),
    )
    monkeypatch.setattr(
        runner,
        "run_registered_worlds",
        lambda *args, **kwargs: pytest.fail("native future entered"),
    )
    with pytest.raises(ValueError, match="decision barrier"):
        runner.worker(output, "future", 0)
    assert list(output.iterdir()) == []


def test_prefix_failure_consumes_claim_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    monkeypatch.setattr(runner, "validate_lock", lambda path: _lock())

    def fail(*args: object, **kwargs: object) -> object:
        directory = Path(str(args[1]))
        claim = read_record(directory / "claim.json")
        assert claim["retry_authorized"] is False
        assert claim["replacement_authorized"] is False
        assert claim["protected_data_read"] is False
        raise RuntimeError("synthetic native failure")

    monkeypatch.setattr(runner, "run_registered_worlds", fail)
    with pytest.raises(RuntimeError, match="synthetic native failure"):
        runner.worker(output, "prefix", 0)
    failure = read_record(output / "prefix-00/failure.json")
    assert failure["retry_authorized"] is False
    assert failure["replacement_authorized"] is False
    assert failure["protected_data_read"] is False
    with pytest.raises(FileExistsError):
        runner.worker(output, "prefix", 0)


def test_complete_roster_has_no_replacement_slots() -> None:
    assert runner.PREFIX_BATCH_COUNT == 36
    assert runner.WORLD_COUNT == 288
    assert runner.prefix_task(35)["world_indices"] == list(range(280, 288))
    assert runner.prefix_task(35)["native_world_indices"] == list(range(280, 288))


def test_runner_binds_parent_and_future_information_boundary() -> None:
    source = (
        ROOT / "scripts/remote/run_dlolab_slingshot_certified_guard_v2.py"
    ).read_text()
    for expression in (
        'result["source_gate_passed"] is not False',
        'calibrator["evaluation_futures_read"] is not False',
        'barrier.get("future_simulated") is not False',
        'barrier.get("future_read") is not False',
        'claim.get("retry_authorized") is not False',
        "directory.mkdir(exist_ok=False)",
    ):
        assert expression in source


def test_parent_artifact_hashes_are_complete() -> None:
    assert runner.PARENT_FILE_SHA256 == {
        "lock.json": "6dce35441588c2a5eff9c0ae08d85c8b41ff660403541dd489b8d9161bffcc8d",
        "result.json": "1df6afe4832a9c35bc65543255f5ce2c5830e6d58cfaa23d1140f8c867767e0b",
        "calibrator.json": (
            "26a00b934dd91b9c121242858756b7a44fa58d61163db53a3ebdebf229de6725"
        ),
        "model-bank/arrays.npz": (
            "ef627e16490c0974d4c34fc82c16aae884fe6dd2a8dc0a80983e89b6d5e50832"
        ),
        "model-bank/seal.json": (
            "f4a9331d552fe8f9715d222327c3f5c41cd7fc81a006e0f9a2fc55dd2223a3ae"
        ),
    }
