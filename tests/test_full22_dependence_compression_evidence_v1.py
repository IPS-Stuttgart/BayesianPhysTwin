from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "results" / "science" / "full22_dependence_compression_diagnostic_v1"
)


def _load(name: str) -> dict[str, object]:
    value = json.loads((EVIDENCE / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def test_result_is_content_addressed_and_uses_all_cases() -> None:
    result = _load("result.json")
    result_id = result.pop("result_id")

    assert result_id == _canonical_sha256(result)
    assert result["status"] == "completed-target-free"
    aggregate = result["aggregate"]
    assert isinstance(aggregate, dict)
    assert aggregate["case_count"] == 22
    assert aggregate["block_count"] == 3638553
    assert aggregate["equal_case_mean_total_correlation_nats"] == pytest.approx(
        0.0015531253193047767
    )
    assert aggregate["case_bootstrap_95_interval_nats"] == pytest.approx(
        [0.0010322945390123276, 0.0020976713899215032]
    )


def test_frozen_gates_reject_the_fused_headline_and_outcome_access() -> None:
    result = _load("result.json")
    gates = result["gates"]
    assert gates == {
        "dependence_signal_supported": False,
        "headline_fused_claim_supported": False,
        "local_rank1_fidelity_supported": True,
        "realized_outcome_comparison_authorized": False,
        "strict_compression_supported": False,
        "whole_object_dependence_testable": False,
    }

    representation = result["representation"]
    assert isinstance(representation, dict)
    assert representation["dense_cross_track_covariance_available"] is False
    assert representation["full_symmetric_parameters_per_block"] == 6
    assert representation["diagonal_plus_rank1_parameters_per_block"] == 6


def test_independent_recomputation_and_source_custody_are_bound() -> None:
    verification = _load("verification.json")
    verification_id = verification.pop("verification_id")

    assert verification_id == _canonical_sha256(verification)
    assert verification["status"] == "verified-negative"
    independent = verification["independent_recomputation"]
    assert isinstance(independent, dict)
    assert independent["scored_mean_matches_registered_result"] is True
    assert independent[
        "intrinsic_donor_equal_case_mean_total_correlation_nats"
    ] == pytest.approx(0.0027418518630065603)
    assert independent["intrinsic_donor_case_range_nats"] == pytest.approx(
        [0.000022575076322922625, 0.0072111116558115085]
    )

    source = verification["source_extraction"]
    assert isinstance(source, dict)
    assert source["member_count"] == 23
    assert source["all_prediction_case_sha256_verified"] is True


def test_recorded_file_hashes_match_verification() -> None:
    verification = _load("verification.json")
    result = verification["result"]
    assert isinstance(result, dict)

    assert hashlib.sha256((EVIDENCE / "result.json").read_bytes()).hexdigest() == (
        result["file_sha256"]
    )
    assert hashlib.sha256((EVIDENCE / "report.md").read_bytes()).hexdigest() == (
        result["report_file_sha256"]
    )
