from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/diagnostics/cloth_sim2real_prob4d_covariance_v1"
RESULT_ID = "e4fb7b7bd7f7455cf5e97def84db6f82da1360f5d00c344ca0a55c8d37219670"
REPORT_ID = "bbed9a588ed1f5d2ba076131c2ec9f89bee1f4624b5eaa902ee5fa16a9ca3ff1"
ARTIFACT_SHA256 = (
    "04f98659a45f5b6b64d7b9feb865c29085c7be18dd99b1684a7a99a439d18955"
)
DATASET_SHA256 = (
    "268d07d94396f6f4ca277b6da0e8acf43512747fea6d40327eb33166da972c7f"
)


def _load(name: str) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads((RESULTS / name).read_text(encoding="utf-8")),
    )


def test_checksums_bind_every_capsule_file() -> None:
    expected: dict[str, str] = {}
    for line in (RESULTS / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", maxsplit=1)
        expected[name] = digest

    assert expected == {
        "artifact_receipt.json": (
            "484ea943c0c487b707c5f9e36fb812d1b96fa56ffd76047d4fb53a774e76ec5c"
        ),
        "calibration_domain_guard_diagnostic.json": (
            "ddcf6f7f82a9a0429fbaa651b85d9ce8df3fbab62a0c56ef9d95ce9c8e63490a"
        ),
        "method_summary.csv": (
            "67f84ef8b5cf42854d9288af817547332a0549bf3f0f4197a6f1006a7ff81168"
        ),
        "pairwise_diagnostic.json": (
            "6751fb95aaf7966987df0e3ac413e838f4e351d9aafb247fbb101be27f50b4ea"
        ),
        "scientific_result_compact.json": (
            "cc126f3e009838a2e6c87706cf9fb95c598d9ff90e9431d41504879038cb8692"
        ),
        "workflow_provenance.json": (
            "411a2186f2099e19d9ffe7a7f6b1596cdafde75200b9e1035d73f6d63828a3d7"
        ),
    }
    for name, digest in expected.items():
        assert hashlib.sha256((RESULTS / name).read_bytes()).hexdigest() == digest


def test_receipt_and_provenance_bind_authoritative_workflow_artifact() -> None:
    receipt = _load("artifact_receipt.json")
    provenance = _load("workflow_provenance.json")

    assert receipt["workflow_run_id"] == 31384479427
    assert receipt["artifact_id"] == 9061934079
    assert receipt["artifact_sha256"] == ARTIFACT_SHA256
    assert receipt["artifact_size_bytes"] == 220238
    assert receipt["artifact_file_count"] == 124
    assert receipt["result_id"] == RESULT_ID
    assert receipt["report_id"] == REPORT_ID
    assert receipt["claim_authorized"] is False
    assert receipt["fresh_confirmation"] is False
    assert provenance["source_revision"] == (
        "7019d4e6effa88addba40cf6faeaeabc6d285c02"
    )
    assert provenance["dataset_sha256"] == DATASET_SHA256
    assert provenance["dataset_size_bytes"] == 3762021195
    assert provenance["result_id"] == RESULT_ID
    assert provenance["report_id"] == REPORT_ID


def test_real_cloth_result_has_uncertainty_signal_not_a_point_loss_win() -> None:
    result = _load("scientific_result_compact.json")
    variants = cast(dict[str, Any], result["treatments"])
    full = variants["full_joint"]
    independent = variants["independent_rows"]

    assert result["result_id"] == RESULT_ID
    assert result["report_id"] == REPORT_ID
    assert result["dataset_sha256"] == DATASET_SHA256
    assert result["retrospective_target_rerun"] is True
    assert result["fresh_confirmation"] is False
    assert result["claim_authorized"] is False
    assert full["dynamic_primary"]["candidate_symmetric_l1_chamfer_m"] == (
        pytest.approx(0.07860009797701045)
    )
    assert full["dynamic_primary"]["raw_90_coordinate_coverage"] == (
        pytest.approx(0.44355256934115217)
    )
    assert full["calibration_std_multiplier"] == pytest.approx(4.816265434408154)
    assert full["quasi_static_secondary"][
        "object_balanced_symmetric_relative_improvement"
    ] == pytest.approx(-0.08133441633042779)
    assert independent["dynamic_primary"]["candidate_symmetric_l1_chamfer_m"] < (
        full["dynamic_primary"]["candidate_symmetric_l1_chamfer_m"]
    )
    assert full["dynamic_primary"]["raw_90_coordinate_coverage"] == max(
        variant["dynamic_primary"]["raw_90_coordinate_coverage"]
        for variant in variants.values()
    )
    assert full["calibration_std_multiplier"] == min(
        variant["calibration_std_multiplier"] for variant in variants.values()
    )
    assert full["all_target_trials"]["harmful_accepted_count"] == 2


def test_method_table_contains_the_five_canonical_treatments() -> None:
    with (RESULTS / "method_summary.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert [row["treatment"] for row in rows] == [
        "full_joint",
        "shared_uncertainty_underreported",
        "block_diagonal",
        "independent_rows",
        "shared_uncertainty_removed",
    ]
    full = rows[0]
    assert float(full["dynamic_candidate_symmetric_l1_chamfer_mm"]) == (
        pytest.approx(78.600097977)
    )
    assert float(full["dynamic_raw_90_coverage_percent"]) == pytest.approx(
        44.3552569341
    )
    assert float(full["quasi_static_relative_improvement_percent"]) < 0.0


def test_pairwise_diagnostic_rejects_a_full_joint_accuracy_claim() -> None:
    diagnostic = _load("pairwise_diagnostic.json")
    comparisons = cast(dict[str, Any], diagnostic["comparisons"])
    independent_dynamic = comparisons["independent_rows"]["metrics"][
        "future_symmetric_l1_chamfer"
    ]["dynamic_trials"]
    removed_all = comparisons["shared_uncertainty_removed"]["metrics"][
        "future_symmetric_l1_chamfer"
    ]["all_target_trials"]

    assert diagnostic["retrospective_post_outcome_diagnostic"] is True
    assert diagnostic["claim_authorized"] is False
    assert independent_dynamic["mean_difference_mm"] > 0.0
    assert removed_all["percentile_95_interval_mm"][0] < 0.0
    assert removed_all["percentile_95_interval_mm"][1] > 0.0
    assert comparisons["shared_uncertainty_removed"]["dynamic_uncertainty"][
        "raw_coverage_percentage_point_difference"
    ] == pytest.approx(4.065582739224837)


def test_calibration_domain_guard_is_retrospective_and_harm_free() -> None:
    diagnostic = _load("calibration_domain_guard_diagnostic.json")
    full = diagnostic["treatments"]["full_joint"]

    assert diagnostic["information_boundary"]["fresh_confirmation"] is False
    assert diagnostic["information_boundary"]["claim_authorized"] is False
    assert full["calibration"]["dynamic"]["authorized"] is True
    assert full["calibration"]["quasi_static"]["authorized"] is False
    assert full["target"]["accepted_count"] == 3
    assert full["target"]["fallback_count"] == 3
    assert full["target"]["harmful_accepted_count"] == 0
    assert full["target"]["relative_improvement"] == pytest.approx(
        0.044756357929444594
    )
