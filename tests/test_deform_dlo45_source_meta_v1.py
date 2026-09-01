from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin_experiments.deform_dlo45_source_meta_v1 import (
    evaluate_source_meta_analysis,
    exact_upper_sign_probability,
    load_source_meta_protocol,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols" / "deform_dlo45_source_meta_v1.json"
RUNNER = ROOT / "scripts" / "science" / "summarize_deform_dlo45_source_meta_v1.py"


def _source_result(dlo: str, *, scale: float) -> dict[str, object]:
    baseline = np.linspace(1.0, 1.7, 8) * scale
    candidate = baseline * np.linspace(0.80, 0.94, 8)
    free_baseline = 1.5 * baseline
    free_candidate = 1.5 * candidate
    difference = baseline - candidate
    distributions = {
        name: {
            "coordinate_coverage_90": coverage,
            "coordinate_nees": nees,
            "multivariate_nees": nees + 0.1,
            "energy_score": 0.02,
            "gaussian_nll": -7.0,
            "interval_width_m": 0.05,
        }
        for name, coverage, nees in (
            ("trajectory-clustered-full-coordinate-covariance-v1", 0.88, 1.3),
            ("calibrated-full-coordinate-covariance-v1", 0.94, 0.7),
        )
    }
    return {
        "schema_version": 1,
        "contract": "deform-dlo45-source-result-v1",
        "dlo": dlo,
        "source_test_opened": True,
        "target_eval_enumerated": False,
        "target_eval_read": False,
        "target_authorized": False,
        "retry_authorized": False,
        "prob4d_used": False,
        "source_gate": {
            "metric": "official-mean-coordinate-l1-all-nodes",
            "case_names": [f"{dlo}-{index}.pkl" for index in range(8)],
            "baseline_case_l1_m": baseline.tolist(),
            "candidate_case_l1_m": candidate.tolist(),
            "baseline_mean_l1_m": float(np.mean(baseline)),
            "candidate_mean_l1_m": float(np.mean(candidate)),
            "relative_improvement": float(1.0 - np.mean(candidate) / np.mean(baseline)),
            "wins": int(np.sum(difference > 1e-15)),
            "ties": int(np.sum(np.abs(difference) <= 1e-15)),
            "worst_candidate_to_baseline_ratio": float(np.max(candidate / baseline)),
            "passed": True,
            "free_node_diagnostic": {
                "baseline_case_l1_m": free_baseline.tolist(),
                "candidate_case_l1_m": free_candidate.tolist(),
            },
        },
        "bayesian_distributions": distributions,
    }


def test_protocol_freezes_post_source_pre_target_analysis() -> None:
    protocol = load_source_meta_protocol(PROTOCOL)

    assert protocol["blocking_run"]["run_id"] == 33361441865
    assert protocol["evaluation"]["bootstrap_repetitions"] == 10000
    assert (
        protocol["evaluation"]["decision_rule"] == "both-original-source-gates-passed"
    )
    assert protocol["information_boundary"]["target_scores_used"] is False


def test_joint_source_meta_analysis_reports_complete_directional_consistency() -> None:
    protocol = load_source_meta_protocol(PROTOCOL)
    result = evaluate_source_meta_analysis(
        protocol=protocol,
        source_results={
            "DLO4": _source_result("DLO4", scale=0.01),
            "DLO5": _source_result("DLO5", scale=0.02),
        },
    )

    assert result["both_original_source_gates_passed"] is True
    assert result["pooled_equal_trajectory"]["case_count"] == 16
    assert result["pooled_equal_trajectory"]["wins"] == 16
    assert result["pooled_equal_trajectory"]["losses"] == 0
    assert result["pooled_equal_trajectory"][
        "exact_upper_sign_probability"
    ] == pytest.approx(2.0**-16)
    assert (
        result["dlo_stratified_bootstrap"]["relative_improvement_95_interval"][0] > 0.0
    )
    assert result["information_boundary"]["target_scores_used"] is False


def test_target_access_in_source_record_is_rejected() -> None:
    protocol = load_source_meta_protocol(PROTOCOL)
    dlo5 = _source_result("DLO5", scale=0.02)
    dlo5["target_eval_read"] = True

    with pytest.raises(ValueError, match="boundary"):
        evaluate_source_meta_analysis(
            protocol=protocol,
            source_results={
                "DLO4": _source_result("DLO4", scale=0.01),
                "DLO5": dlo5,
            },
        )


def test_exact_sign_probability_excludes_ties() -> None:
    assert exact_upper_sign_probability(wins=16, losses=0) == pytest.approx(2.0**-16)
    assert exact_upper_sign_probability(wins=3, losses=1) == pytest.approx(5.0 / 16.0)
    with pytest.raises(ValueError, match="nonempty"):
        exact_upper_sign_probability(wins=0, losses=0)


def test_runner_seals_method_before_loading_source_payloads() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    seal = source.index('method_seal_path = output_root / "method_seal.json"')
    load = source.index(
        "source_results = {dlo: _read_json(path) for dlo, path in result_paths.items()}"
    )

    assert seal < load
    assert "target_scores_used" in source
    assert "source_payloads_loaded_after_method_seal" in source
