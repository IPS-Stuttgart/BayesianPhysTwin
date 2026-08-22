import hashlib
import json
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (
    REPOSITORY_ROOT / "results" / "sota" / "deform_dlo2_local_residual_official_v7"
)
SUMMARY = RESULT_ROOT / "summary.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_dlo2_local_residual_official_v7_result() -> None:
    result = json.loads(SUMMARY.read_text(encoding="utf-8"))

    assert (
        _sha256(SUMMARY)
        == "4ad06dc81e7ec144b62fe0d8a3f8b85a8ac4658233ef95d0048e217aa6083310"
    )
    assert result["contract"] == (
        "deform-dlo2-local-residual-official-v7-compact-evidence"
    )
    assert result["decision"]["validation_passed"] is True
    assert result["decision"]["benchmark_specific_claim_gate_passed"] is True

    evaluation = result["evaluation"]
    assert evaluation["case_count"] == 14
    assert evaluation["scored_future_frames"] == 498
    assert evaluation["all_expected_cases_evaluated_once"] is True
    assert evaluation["target_selection_performed"] is False
    assert evaluation["target_calibration_performed"] is False
    assert evaluation["target_retry_performed"] is False
    assert evaluation["case_replacement_performed"] is False
    assert evaluation["candidate_mean_l1_m"] == pytest.approx(0.007860559253359958)
    assert evaluation["comparison_baseline_mean_l1_m"] == pytest.approx(
        0.008746962326219684
    )
    assert evaluation["relative_improvement"] == pytest.approx(0.10133838923744597)
    assert evaluation["case_wins"] == 14
    assert evaluation["maximum_case_ratio"] == pytest.approx(0.9419249083983519)

    reference = result["published_reference"]
    assert reference["mean_l1_m"] == pytest.approx(0.0097)
    assert reference["candidate_all_unique_mean_l1_m"] < reference["mean_l1_m"]
    assert reference["candidate_canonical_draw_mean_l1_m"] < reference["mean_l1_m"]
    assert result["claim_gate"]["passed"] is True
    assert result["method"]["prob4d_used"] is False
    assert result["uncertainty"]["variance_scale"] == 1.0

    lineage = result["lineage"]
    for key, value in lineage.items():
        if key.endswith("_sha256"):
            assert len(value) == 64
            int(value, 16)
