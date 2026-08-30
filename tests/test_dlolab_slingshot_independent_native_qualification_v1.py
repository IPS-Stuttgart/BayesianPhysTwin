"""Custody contracts for the independent native Slingshot qualification."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT / "scripts/remote/run_dlolab_slingshot_independent_native_qualification_v1.py"
)
SPEC = importlib.util.spec_from_file_location(
    "independent_native_qualification", RUNNER
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_runner_binds_terminal_parent_and_complete_source_contract():
    assert len(runner.PARENT_LOCK_ID) == 64
    assert len(runner.PARENT_SUMMARY_ID) == 64
    assert len(runner.CONTROLS_SHA256) == 64
    assert runner.WORKERS == 8
    for name in runner.SOURCES:
        assert (ROOT / name).is_file()
    source = RUNNER.read_text()
    assert "fresh registered one-attempt root" in source
    assert '"retry_authorized": False' in source
    assert '"replacement_authorized": False' in source
    assert '"v3_scientific_execution_authorized": False' in source


def test_unregistered_output_is_rejected_before_parent_access(tmp_path, monkeypatch):
    def forbidden():
        raise AssertionError("parent must not be touched")

    monkeypatch.setattr(runner, "load_parent", forbidden)
    with pytest.raises(ValueError, match="registered one-attempt"):
        runner.freeze(tmp_path)


def test_any_process_failure_prevents_world_scoring_and_v3_authorization(
    tmp_path, monkeypatch
):
    tmp_path.mkdir(exist_ok=True)
    monkeypatch.setattr(
        runner,
        "freeze",
        lambda output: {"artifact_id": "a" * 64, "runtime": {}},
    )
    monkeypatch.setattr(
        runner,
        "execute",
        lambda output, lock, index: int(index == 3),
    )
    monkeypatch.setattr(runner, "load_task", lambda output, lock, index: ({}, {}))
    monkeypatch.setattr(
        runner,
        "validate_task_failure",
        lambda output, lock, index: None,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("incomplete denominator must not be aggregated")

    monkeypatch.setattr(runner, "independent_world_qa", forbidden)
    result = runner.run(tmp_path)
    assert result["status"] == "terminal_qualification_failure"
    assert result["ordinary_processes"] == 63
    assert result["failed_process_indices"] == [3]
    assert result["custody_validation_errors"] == {}
    assert result["qualified_worlds"] == 0
    assert result["qualification_passed"] is False
    assert result["v3_protocol_freeze_authorized"] is False
    assert result["v3_scientific_execution_authorized"] is False
