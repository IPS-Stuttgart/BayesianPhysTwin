from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from bayesian_phystwin_experiments import dlolab_wrapping_risk_guard_v8 as study
from bayesian_phystwin_experiments.dlolab_regret_artifacts import read_record

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "wrapping_risk_guard_source_v8_verifier",
    ROOT / "scripts/verify_dlolab_wrapping_risk_guard_source_v8.py",
)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def test_exact_result_tree_contract_covers_every_registered_task() -> None:
    paths = verifier._expected_paths()
    assert len(paths) == verifier.EXPECTED_TREE_FILES == 647
    assert "decisions/arrays.npz" in paths
    assert "generation/arrays.npz" in paths
    assert "prefix-15/seal.json" in paths
    assert "future-143/seal.json" in paths
    assert not any("failure" in path for path in paths)


def test_independent_arithmetic_matches_registered_score_definition() -> None:
    decisions = np.full(
        (study.WORLD_COUNT, study.SENSOR_DRAWS, len(study.ARM_NAMES)),
        4,
        dtype=np.int64,
    )
    decisions[:, :, 1] = 5
    decisions[:, :, 2] = np.where(
        np.arange(study.WORLD_COUNT)[:, None] % 3 == 0, 5, 4
    )
    reward = np.zeros((study.WORLD_COUNT, study.N_ACTIONS), dtype=np.float64)
    reward[:, 4] = 0.8
    reward[:, 5] = np.where(np.arange(study.WORLD_COUNT) % 5 == 0, 0.75, 0.85)
    registered = study.score(decisions, reward, all_native_qa=True)
    independent = verifier._independent_arithmetic(decisions, reward)
    assert independent["fixed_mean_reward"] == registered["arms"][
        "continuous_prior_best_fixed"
    ]["mean_native_reward"]
    assert independent["guard_gain"] == registered["paired_guard_gain"][
        "continuous_prior_best_fixed"
    ]["mean_gain"]
    assert independent["guard_gain_ci95"] == registered["paired_guard_gain"][
        "continuous_prior_best_fixed"
    ]["ci95"]
    assert independent["guard_harms"] == registered["guard_harmed_worlds"]
    assert independent["continuous_harms"] == registered["continuous_harmed_worlds"]
    assert independent["downside_reduction_ci95"] == registered[
        "guard_downside_reduction_ci95"
    ]


def test_compact_summary_preserves_failed_gate_and_positive_risk_result() -> None:
    summary = read_record(
        ROOT / "results/sota/dlolab_wrapping_risk_guard_source_v8/summary.json"
    )
    assert summary["status"] == "complete_source_gate_failed"
    assert summary["source_gate_passed"] is False
    assert summary["strict_gate_reclassified"] is False
    assert summary["guard_gain_ci95"][0] > 0
    assert summary["continuous_bayes_harmed_worlds"] == 10
    assert summary["guard_harmed_worlds"] == 2
    assert summary["guard_downside_reduction_fraction"] > 0.94
    assert summary["independent_human_review"] is False
    assert summary["official_benchmark_or_sota_claim"] is False
    assert summary["protected_data_read"] is False
