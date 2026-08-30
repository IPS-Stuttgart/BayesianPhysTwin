from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np

from bayesian_phystwin_experiments import dlolab_wrapping_certified_guard_v9 as study
from bayesian_phystwin_experiments.dlolab_regret_artifacts import read_record

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "wrapping_certified_guard_source_v9_verifier",
    ROOT / "scripts/verify_dlolab_wrapping_certified_guard_source_v9.py",
)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def test_exact_result_tree_contract_covers_every_registered_task() -> None:
    paths = verifier._expected_paths()
    assert len(paths) == verifier.EXPECTED_TREE_FILES == 1_287
    assert "decisions/arrays.npz" in paths
    assert "generation/arrays.npz" in paths
    assert "prefix-31/seal.json" in paths
    assert "future-287/seal.json" in paths
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
    registered = study.score(
        decisions,
        reward,
        all_native_qa=True,
        calibration_certificate_valid=True,
    )
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
    assert independent["guard_harm_risk_upper"] == registered[
        "guard_one_sided_95pct_harm_risk_upper"
    ]
    assert independent["downside_reduction_ci95"] == registered[
        "guard_downside_reduction_ci95"
    ]


def test_protocol_verification_is_portable_only_within_four_ulps() -> None:
    recorded = copy.deepcopy(study.protocol())
    recorded["worlds"][5]["stretching_K"] = float(
        np.nextafter(recorded["worlds"][5]["stretching_K"], np.inf)
    )
    assert verifier._portable_protocol_matches(recorded)

    recorded["worlds"][5]["stretching_K"] += 1e-6
    assert not verifier._portable_protocol_matches(recorded)

    changed_contract = copy.deepcopy(study.protocol())
    changed_contract["retry_authorized"] = True
    assert not verifier._portable_protocol_matches(changed_contract)


def test_decision_verification_requires_exact_actions_and_tight_float_agreement(
) -> None:
    regenerated = {
        "decisions": np.array([[1, 2]], dtype=np.int64),
        "probability": np.array([[0.2, 0.8]], dtype=np.float64),
    }
    recorded = {name: value.copy() for name, value in regenerated.items()}
    recorded["probability"][0, 0] = np.nextafter(
        recorded["probability"][0, 0], np.inf
    )
    assert verifier._decision_arrays_match(recorded, regenerated)

    changed_action = {name: value.copy() for name, value in recorded.items()}
    changed_action["decisions"][0, 0] = 0
    assert not verifier._decision_arrays_match(changed_action, regenerated)

    changed_probability = {name: value.copy() for name, value in recorded.items()}
    changed_probability["probability"][0, 0] += 1e-9
    assert not verifier._decision_arrays_match(changed_probability, regenerated)


def test_compact_summary_preserves_positive_certified_guard_result() -> None:
    summary = read_record(
        ROOT
        / "results/sota/dlolab_wrapping_risk_certified_guard_source_v9/summary.json"
    )
    assert summary["status"] == "complete_source_gate_passed"
    assert summary["source_gate_passed"] is True
    assert summary["guard_gain_ci95"][0] > 0
    assert summary["continuous_bayes_harmed_worlds"] == 15
    assert summary["guard_harmed_worlds"] == 1
    assert summary["guard_harm_risk_upper"] < summary["harm_risk_budget"]
    assert summary["guard_downside_reduction_fraction"] > 0.97
    assert summary["independent_human_review"] is False
    assert summary["official_benchmark_or_sota_claim"] is False
    assert summary["protected_data_read"] is False
