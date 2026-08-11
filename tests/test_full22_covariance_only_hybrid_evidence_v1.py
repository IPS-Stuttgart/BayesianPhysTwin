from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT / "results" / "science" / "full22_covariance_only_hybrid_v1" / "summary.json"
)
RESULT_MARKDOWN = SUMMARY.with_name("result.md")


def _summary() -> dict[str, object]:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_compact_evidence_binds_exact_execution_and_source() -> None:
    payload = _summary()

    assert payload["contract"] == (
        "bayesian-phystwin-full22-covariance-only-hybrid-compact-evidence-v1"
    )
    assert payload["status"] == "completed"
    assert payload["analysis_status"] == (
        "retrospective-cross-fitted-development-only"
    )
    assert payload["report_id"] == (
        "5fc777163fd6173c9669b497309d883e2780a5ebe23da5dbe4cdaf682ad8806a"
    )

    workflow = payload["result_workflow"]
    assert isinstance(workflow, dict)
    assert workflow == {
        "analyzer_revision": "9ac5344036331a40e9029a1aa5814f601c0eaf15",
        "artifact_digest": (
            "sha256:945dd5ed5db9b4119d81cf15d6d6c6da304b9421ab5938e71ad65390bbca1676"
        ),
        "artifact_id": 9090224528,
        "artifact_name": "full22-covariance-only-hybrid-v1-31461910994-1",
        "artifact_size_bytes": 10229,
        "result_json_sha256": (
            "31ed0828b0c024a508467904c56c040f4e38374c4688a06167f2e584b13111a0"
        ),
        "run_attempt": 1,
        "run_id": 31461910994,
    }

    source = payload["source"]
    assert isinstance(source, dict)
    assert source["run_id"] == 31410594302
    assert source["artifact_id"] == 9074451004
    assert source["prefix_manifest_id"] == (
        "1195884a383b2f2f690a4ea7a0fce9bb82a1dc755d276f85450174fcd9bdc25c"
    )
    assert source["public_source_files_verified"] is True


def test_compact_evidence_preserves_result_and_fresh_candidate() -> None:
    payload = _summary()

    mean_identity = payload["mean_identity"]
    assert isinstance(mean_identity, dict)
    assert mean_identity == {
        "chamfer_distance_difference_m": 0.0,
        "exact_identity_case_count": 22,
        "reference_predictor": "last_residual",
        "track_error_difference_m": 0.0,
    }

    primary = payload["primary_effect"]
    assert isinstance(primary, dict)
    assert primary["arm"] == "crossfit_selected_scaled_covariance"
    assert primary["better_worse_tie_cases"] == [17, 5, 0]
    assert primary["familywise_decision"] == "hybrid_better"
    assert primary["mean_gaussian_nll_difference"] == pytest.approx(
        -9.136379254487014
    )
    assert primary["simultaneous_95_ci"] == pytest.approx(
        [-13.96117966524658, -4.31157884372745]
    )

    selection = payload["crossfit_selection"]
    assert selection == {
        "dynamic_endpoint_v2_fold_count": 1,
        "independent_endpoint_v1_fold_count": 21,
    }
    fresh = payload["full_source_fit_for_separate_fresh_study"]
    assert fresh == {
        "covariance_donor": "independent_endpoint_v1",
        "early_middle_late_scales": [8.0, 16.0, 16.0],
        "mean_predictor": "last_residual",
    }

    coverage = payload["coverage_and_width"]
    assert isinstance(coverage, dict)
    assert coverage["reference_marginal_coverage"] == pytest.approx(
        0.7057813694762896
    )
    assert coverage["hybrid_marginal_coverage"] == pytest.approx(
        0.9097997892386086
    )
    assert coverage["hybrid_to_reference_width_ratio"] == pytest.approx(
        3.0971912419525798
    )


def test_compact_evidence_remains_nonclaim_bearing() -> None:
    payload = _summary()

    assert payload["claim_authorized"] is False
    assert payload["selection_authorized"] is False
    assert payload["promotion_authorized"] is False
    assert "Already-open full-22 cohort" in str(payload["scientific_boundary"])

    markdown = RESULT_MARKDOWN.read_text(encoding="utf-8")
    assert "better in `17/22`" in markdown
    assert "`independent_endpoint_v1` in `21/22` folds" in markdown
    assert "track and Chamfer effects\nare exactly zero" in markdown
    assert "`claim_authorized=false`" in markdown
