"""Fail-closed custody tests for the reward-aligned v4 runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from bayesian_phystwin_experiments.dlolab_native import file_digest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "slingshot_policy_certificate_source_v4_runner",
    ROOT / "scripts/remote/run_dlolab_slingshot_policy_certificate_source_v4.py",
)
assert SPEC is not None and SPEC.loader is not None
runner_entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner_entry)
runner = runner_entry.runner


def test_reward_aligned_qualification_is_exactly_bound() -> None:
    assert file_digest(runner_entry.QUALIFICATION_SUMMARY) == (
        runner_entry.method.QUALIFICATION_RESULT_SHA256
    )
    value = runner_entry.load_qualification()
    assert value["artifact_id"] == runner_entry.method.QUALIFICATION_RESULT_ID
    assert value["denominator"]["reward_aligned_qualified_worlds"] == 128


def test_v4_configuration_binds_worker_entry_and_schemas() -> None:
    assert runner_entry.__file__ is not None
    assert runner.OUTPUT_ROOT == runner_entry.OUTPUT_ROOT
    assert runner.WORKER_RUNNER_PATH == Path(runner_entry.__file__).resolve()
    assert runner.ATTEMPT_LEDGER == runner_entry.ATTEMPT_LEDGER
    assert runner.LOCK_SCHEMA.endswith("-v4")
    assert runner.WORLD_QUALIFICATION_SCHEMA.endswith("-v4")
    assert runner.independent_world_qa is runner_entry.method.reward_aligned_world_qa
    assert runner.world_rewards is runner_entry.method.reward_aligned_world_rewards


def test_alternate_root_rejected_before_parent_or_qualification_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "OUTPUT_ROOT", tmp_path / "registered")
    monkeypatch.setattr(
        runner, "load_parent", lambda: pytest.fail("parent read before root rejection")
    )
    monkeypatch.setattr(
        runner,
        "load_qualification",
        lambda: pytest.fail("qualification read before root rejection"),
    )
    with pytest.raises(ValueError, match="fresh registered"):
        runner.freeze(tmp_path / "alternate")


def test_runner_binds_complete_denominator_no_retry_and_no_partial_score() -> None:
    source = (
        ROOT / "scripts/remote/run_dlolab_slingshot_policy_certificate_source_v3.py"
    ).read_text()
    entry = (
        ROOT / "scripts/remote/run_dlolab_slingshot_policy_certificate_source_v4.py"
    ).read_text()
    for expression in (
        '"attempt_number": 1',
        '"retry_authorized": False',
        '"replacement_authorized": False',
        "for action_index in range(ACTION_COUNT)",
        '"partial_score_authorized": False',
    ):
        assert expression in source
    assert "runner.WORKER_RUNNER_PATH = Path(__file__).resolve()" in entry
    assert runner.COUNTS == {"calibration": 128, "evaluation": 288}
