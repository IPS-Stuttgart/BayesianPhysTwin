from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from bayesian_phystwin_experiments import dlolab_slingshot_certified_guard_v2 as study
from bayesian_phystwin_experiments.dlolab_regret_artifacts import read_record

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "slingshot_certified_guard_source_v2_verifier",
    ROOT / "scripts/verify_dlolab_slingshot_certified_guard_source_v2.py",
)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def test_exact_result_tree_contract_covers_continuation_and_every_task() -> None:
    paths = verifier._expected_paths()
    assert len(paths) == verifier.EXPECTED_TREE_FILES == 1_305
    assert "continuation-receipt.json" in paths
    assert "continuation-complete.json" in paths
    assert "prefix-35/seal.json" in paths
    assert "future-287/seal.json" in paths
    assert not any("failure" in path for path in paths)


def test_independent_decision_rule_matches_registered_rule() -> None:
    rng = np.random.default_rng(260830)
    observations = rng.normal(0, 0.02, (19, 3, 4, 3))
    bank_prefix = rng.normal(0, 0.02, (27, 3, 4, 3))
    bank_reward = rng.normal(7, 0.3, (27, 7))
    expected = study._decisions_for_observations(
        observations, bank_prefix, bank_reward
    )
    actual = verifier._decisions_for_observations(
        observations, bank_prefix, bank_reward
    )
    np.testing.assert_array_equal(actual, expected)


def test_independent_score_matches_registered_score() -> None:
    rng = np.random.default_rng(260831)
    rewards = rng.normal(7, 0.03, (study.WORLD_COUNT, 7))
    decisions = rng.integers(
        0,
        7,
        (study.WORLD_COUNT, study.SENSOR_DRAWS, len(study.ARM_NAMES)),
    )
    decisions[:, :, 0] = study.BASELINE
    expected = study.score(
        decisions,
        rewards,
        all_native_qa=True,
        pre_future_gate_passed=True,
    )
    actual = verifier._independent_arithmetic(decisions, rewards)
    for name, value in actual.items():
        verifier._assert_same(expected[name], value, name=name)


def test_compact_summary_preserves_failed_cross_task_gate() -> None:
    summary = read_record(
        ROOT
        / "results/source/dlolab_slingshot_certified_guard_source_v2/summary.json"
    )
    assert summary["status"] == "complete_source_gate_failed"
    assert summary["source_gate_passed"] is False
    assert summary["ordinary_future_worlds"] == 288
    assert summary["technical_failures"] == 0
    assert summary["posterior_harmed_worlds"] == 62
    assert summary["guard_harmed_worlds"] == 14
    assert summary["guard_downside_reduction_fraction"] > 0.92
    assert summary["guard_harm_risk_upper"] > summary["harm_risk_budget"]
    assert summary["independent_human_review"] is False
    assert summary["official_benchmark_or_sota_claim"] is False

    verification = read_record(
        ROOT
        / "results/source/dlolab_slingshot_certified_guard_source_v2/verification.json"
    )
    assert verification["verification"] == "PASS"
    assert verification["result_id"] == verifier.RESULT_ID
    assert verification["tree_sha256"] == verifier.EXPECTED_TREE_SHA256
    assert verification["independent_human_review"] is False
