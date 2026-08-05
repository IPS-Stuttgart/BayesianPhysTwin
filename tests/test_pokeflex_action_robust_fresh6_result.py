import json
from pathlib import Path

import pytest

from bayesian_phystwin.pokeflex_conservative_shrinkage_target import (
    canonical_payload_sha256,
    file_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "sota" / "pokeflex_action_robust_fresh6_v3"


def _load(name: str) -> dict[str, object]:
    return json.loads((RESULT_ROOT / name).read_text(encoding="utf-8"))


def test_action_robust_fresh6_result_is_bound_and_passes_all_gates() -> None:
    result_path = RESULT_ROOT / "target_result.json"
    barrier_path = RESULT_ROOT / "prediction_barrier.json"
    summary = _load("summary.json")
    result = _load("target_result.json")

    assert file_sha256(result_path) == (
        "1e8fcae19d618d52a05762ebd039e92098b52725459bf8d320124fffcaead204"
    )
    assert file_sha256(barrier_path) == (
        "7a0fdf988edfc81ffb80675d363430d45265c717fc7f9b0dcb25a832be5434cb"
    )
    assert file_sha256(RESULT_ROOT / "summary.json") == (
        "41dae21e0c5d92bb79835da92fb7fa5e479bae52b667eac4b48757d2fdb01b9b"
    )
    assert summary["summary_sha256"] == canonical_payload_sha256(
        summary,
        digest_field="summary_sha256",
    )
    assert summary["summary_sha256"] == (
        "3bdf93d6f939d87a4ad971d64589095553fd3df59b4016767479b664ba0fb945"
    )
    assert summary["implementation_revision"] == (
        "7882fc449e33f12a577ad2cfcec3d24651bfba79"
    )
    assert summary["target_result_file_sha256"] == file_sha256(result_path)
    assert summary["prediction_barrier_file_sha256"] == file_sha256(barrier_path)
    assert result["aggregate"]["checkpoint_pairing"]["passed"] is True
    assert result["aggregate"]["global_scale_advancement"]["passed"] is True
    assert result["aggregate"]["all_target_gates_passed"] is True
    assert result["aggregate"]["published_direct_comparison_authorized"] is False


def test_action_robust_fresh6_advancement_is_no_regression() -> None:
    summary = _load("summary.json")
    direct = summary["action_robust_vs_checkpoint"]
    advancement = summary["action_robust_vs_global"]

    assert direct["relative_CD_UL1_improvement"] == pytest.approx(
        0.02240165696467543
    )
    assert direct["win_count"] == 6
    assert direct["tie_count"] == 0
    assert direct["minimum_per_object_relative_improvement"] > 0
    assert direct["bootstrap_upper_candidate_minus_reference_CD_UL1_mm"] < 0
    assert advancement["relative_CD_UL1_improvement"] == pytest.approx(
        0.012273361865912684
    )
    assert advancement["win_count"] == 6
    assert advancement["tie_count"] == 0
    assert advancement["minimum_per_object_relative_improvement"] > 0
    assert advancement["bootstrap_upper_candidate_minus_reference_CD_UL1_mm"] < 0
    assert summary["scored_frame_count"] == 436
    assert summary["supported_scored_frame_count"] == 402

    assert all(
        row["action_robust_vs_checkpoint_relative_improvement"] > 0
        and row["action_robust_vs_global_relative_improvement"] > 0
        for row in summary["objects"]
    )
    pizza = next(
        row for row in summary["objects"] if row["object_name"] == "3dPrintedPizza"
    )
    assert pizza["global_vs_checkpoint_relative_improvement"] < 0
    assert pizza["action_robust_vs_checkpoint_relative_improvement"] > 0


def test_action_robust_fresh6_scoring_provenance_is_canonical() -> None:
    payload_path = RESULT_ROOT / "scoring_provenance.json"
    payload = _load("scoring_provenance.json")
    barrier = _load("prediction_barrier.json")
    barrier_seals = {
        row["take_id"]: row["seal_file_sha256"]
        for row in barrier["predictions"]
    }

    assert file_sha256(payload_path) == (
        "1e3e149a8eada32750a05dd1ba28a2b4eb8688773b2f61409cb4d0d83ac8d5c4"
    )
    assert payload["provenance_sha256"] == canonical_payload_sha256(
        payload,
        digest_field="provenance_sha256",
    )
    assert payload["provenance_sha256"] == (
        "6efb36c8f2ccf25efd3d667803c39b01dcc9ee88eae6b7b1c5f8329a8e8ea326"
    )
    assert payload["transfer"]["direct_lan"] is True
    assert payload["transfer"]["jump_server_used_for_data"] is False
    assert payload["transfer"]["one_time_transfer_credential_removed"] is True
    assert payload["transfer"]["post_transfer_member_rehash_passed"] is True
    assert payload["transfer"]["staged_member_count"] == 2865
    assert payload["transfer"]["staged_byte_count"] == 2711727518
    assert payload["target_mesh_geometry_decoded_before_barrier"] is False
    assert payload["target_metric_computed_before_barrier"] is False
    assert len(payload["takes"]) == 6

    for row in payload["takes"]:
        seal_path = RESULT_ROOT / "prediction_seals" / row["take_id"] / "seal.json"
        assert file_sha256(seal_path) == row["prediction_seal_file_sha256"]
        assert row["prediction_seal_file_sha256"] == barrier_seals[row["take_id"]]
        assert row["fallback_mismatch_count"] == 0
        assert row["global_fallback_mismatch_count"] == 0
        assert row["future_mesh_read_before_barrier"] is False
