from __future__ import annotations

import ast
from pathlib import Path

from bayesian_phystwin_experiments.dlolab_task_aware_voi import protocol

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/remote/run_dlolab_task_aware_voi.py"


def test_registered_runner_has_one_new_write_once_source_root() -> None:
    source = RUNNER.read_text()
    assert "/home/fpfaff/source-only/dlolab-task-aware-voi-source-v1" in source
    assert "/home/fpfaff/source-only/dlolab-task-aware-voi-source-v1.attempt.json" in source
    assert "/home/fpfaff/source-only/dlolab-matched-reset-dual-control-source-v1" not in source
    assert '"retry_authorized": False' in source
    assert "ATTEMPT_LEDGER.exists()" in source
    ast.parse(source)


def test_truth_futures_are_dispatched_after_task_aware_decision_seal() -> None:
    source = RUNNER.read_text()
    decision = source.index('decision_seal = write_record(')
    future = source.index('stage = "truth-futures"')
    assert decision < future
    stages = protocol()["staged_information_boundary"]
    assert stages.index("decision_seal") < stages.index("truth_task_futures")


def test_task_aware_selector_uses_particle_task_table_but_not_truth() -> None:
    source = RUNNER.read_text()
    call = source[source.index("selectors = selector_analysis(") :]
    call = call[: call.index("\n")]
    assert "particle_loss" in call
    assert protocol()["primary_probe_uses_particle_task_table"] is True
    assert protocol()["primary_probe_uses_truth_futures"] is False


def test_generic_mi_and_fixed_probe_remain_explicit_controls() -> None:
    source = RUNNER.read_text()
    assert 'selectors["generic_mi_probe_index"]' in source
    assert '"fixed_probe_bayes"' not in source  # Decisions are produced by the bound core.
    assert "mi_probe_bayes" not in source
    assert "score_source(decisions, truth_loss)" in source


def test_runner_binds_verifier_protocol_and_prelock_sources() -> None:
    source = RUNNER.read_text()
    for name in (
        "scripts/verify_dlolab_task_aware_voi.py",
        "docs/dlolab_task_aware_voi_source_v1.md",
        "configs/sota/dlolab_task_aware_voi_source_v1.json",
        "results/sota/dlolab_task_aware_voi_prelock_v1/null_probe_native_smoke.json",
    ):
        assert name in source
