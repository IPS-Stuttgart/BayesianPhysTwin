from __future__ import annotations

import ast
from pathlib import Path

from bayesian_phystwin_experiments.dlolab_matched_reset_dual_control import protocol

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/remote/run_dlolab_matched_reset_dual_control.py"


def test_registered_runner_has_one_write_once_source_root() -> None:
    source = RUNNER.read_text()
    assert "/home/fpfaff/source-only/dlolab-matched-reset-dual-control-source-v1" in source
    assert "retry_authorized\": False" in source
    ast.parse(source)


def test_truth_futures_are_dispatched_after_decision_seal() -> None:
    source = RUNNER.read_text()
    decision = source.index('decision_seal = write_record(')
    future = source.index('stage = "truth-futures"')
    assert decision < future
    assert protocol()["staged_information_boundary"].index("decision_seal") < protocol()[
        "staged_information_boundary"
    ].index("truth_task_futures")


def test_probe_selection_call_has_no_task_loss_argument() -> None:
    source = RUNNER.read_text()
    call = source[source.index("information = probe_information(") :]
    call = call[: call.index("\n")]
    assert "particle_loss" not in call
    assert protocol()["probe_selection_uses_task_reward_or_future"] is False
