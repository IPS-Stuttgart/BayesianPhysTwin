import hashlib
import json
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (
    REPOSITORY_ROOT
    / "results"
    / "sota"
    / "deform360_bias_aware_guarded_belief_v4"
)
SUMMARY = RESULT_ROOT / "summary.json"
LOCK = RESULT_ROOT / "prospective_lock.json"


def test_bias_aware_source_result_bundle_is_locked() -> None:
    assert hashlib.sha256(SUMMARY.read_bytes()).hexdigest() == (
        "dbad5fd3b4d572d515d38b9bb31df84a2f036c223aaed3aa0810c25fbec3e015"
    )
    assert hashlib.sha256(LOCK.read_bytes()).hexdigest() == (
        "5f5672d35aa41e276f1dd5ace54b6694b0139ff2a562e3c3a24558fa555c9dd6"
    )
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["prospective_candidate_lock"]["sha256"] == (
        "5f5672d35aa41e276f1dd5ace54b6694b0139ff2a562e3c3a24558fa555c9dd6"
    )


def test_bias_aware_source_result_preserves_transfer_and_claim_boundaries() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    comparison = summary["comparisons_to_selected_raw_baseline"][
        "bias_aware_group_bound_guarded_cross_fit"
    ]
    identity = comparison["post_update_hidden_identity_rmse_m"]
    chamfer = comparison["post_update_hidden_symmetric_chamfer_m"]

    assert summary["case_count"] == 27
    assert summary["object_count"] == 5
    assert identity["object_balanced_relative_change"] == pytest.approx(
        -0.014138112336178444
    )
    assert chamfer["object_balanced_relative_change"] == pytest.approx(
        -0.013296050017297525
    )
    assert identity["episode_win_count"] == 7
    assert identity["episode_tie_count"] == 20
    assert chamfer["episode_win_count"] == 7
    assert chamfer["episode_tie_count"] == 20
    guard = summary["cross_fitted_regret_guard"]["source_group_bound"]
    assert guard["accepted_count"] == 10
    assert guard["accepted_harmful_rate"] == 0.0
    assert guard["exact_fallback_count"] == 71
    assert guard["minimum_finite_sample_coverage"] == pytest.approx(0.75)
    assert all(summary["source_transfer_gates"].values())
    assert summary["larger_preregistered_run_justified"]
    assert not summary["calibrated_90_percent_claim_ready"]


def test_prospective_lock_allows_only_fresh_accuracy_evaluation() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))

    assert lock["candidate_interval_count"] == 10
    assert lock["source_group_count"] == 4
    assert lock["finite_sample_rank"] == 4
    assert lock["finite_sample_coverage"] == pytest.approx(0.8)
    assert lock["upper_regret_m"] < -lock["minimum_improvement_m"]
    assert lock["candidate_certified"]
    assert lock["fresh_accuracy_evaluation_allowed"]
    assert not lock["calibrated_90_percent_claim_allowed"]
    assert lock["information_boundary"] == {
        "eligibility_is_target_free": True,
        "future_observations_used_to_construct_candidate": False,
        "prospective_outcomes_used_to_construct_candidate": False,
        "source_outcomes_used_to_fit_lock": True,
    }
