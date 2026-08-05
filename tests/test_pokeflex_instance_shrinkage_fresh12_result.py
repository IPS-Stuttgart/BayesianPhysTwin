import json
from pathlib import Path

import pytest

from bayesian_phystwin.pokeflex_conservative_shrinkage_target import (
    canonical_payload_sha256,
    file_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "sota" / "pokeflex_instance_shrinkage_fresh12_v2"


def _load(name: str) -> dict[str, object]:
    return json.loads((RESULT_ROOT / name).read_text(encoding="utf-8"))


def test_instance_fresh12_result_is_bound_and_preserves_failed_gate() -> None:
    result_path = RESULT_ROOT / "target_result.json"
    barrier_path = RESULT_ROOT / "prediction_barrier.json"
    summary = _load("summary.json")
    result = _load("target_result.json")

    assert file_sha256(result_path) == (
        "defc3c8cf78ccdea0c5b4d03fcb5d3e983dff1a63c4c08b29111a004bfdb0d3b"
    )
    assert file_sha256(barrier_path) == (
        "243ee31e5f1f8fd94d8d2fcb601f9ccb4a2afde1fd4527f017578d037d0a5e51"
    )
    assert summary["summary_sha256"] == canonical_payload_sha256(
        summary,
        digest_field="summary_sha256",
    )
    assert summary["target_result_file_sha256"] == file_sha256(result_path)
    assert summary["prediction_barrier_file_sha256"] == file_sha256(barrier_path)
    assert summary["implementation_revision"] == (
        "d51eca193ca1762b95c0802a1a428c09d036d92f"
    )
    assert result["aggregate"]["checkpoint_pairing"]["passed"] is True
    assert result["aggregate"]["global_scale_checkpoint_pairing"]["passed"] is True
    assert result["aggregate"]["global_scale_advancement"]["passed"] is False
    assert result["aggregate"]["all_target_gates_passed"] is False
    assert result["aggregate"]["published_direct_comparison_authorized"] is False


def test_instance_fresh12_partial_positive_is_scoped_exactly() -> None:
    summary = _load("summary.json")
    direct = summary["instance_vs_checkpoint"]
    advancement = summary["instance_vs_global"]

    assert direct["relative_CD_UL1_improvement"] == pytest.approx(0.01722968243835599)
    assert direct["win_count"] == 11
    assert direct["tie_count"] == 1
    assert advancement["relative_CD_UL1_improvement"] == pytest.approx(
        0.006123286928793878
    )
    assert advancement["bootstrap_upper_candidate_minus_reference_CD_UL1_mm"] < 0
    assert advancement["minimum_per_object_relative_improvement"] == pytest.approx(
        -0.011402171146111119
    )
    assert summary["scored_frame_count"] == 875
    assert summary["supported_scored_frame_count"] == 741

    regressions = [
        row
        for row in summary["objects"]
        if row["instance_vs_global_relative_improvement"] < 0
    ]
    assert [row["object_name"] for row in regressions] == ["3dPrintedPyramid"]


def test_instance_fresh12_scoring_provenance_is_canonical() -> None:
    payload = _load("scoring_provenance.json")

    assert payload["provenance_sha256"] == canonical_payload_sha256(
        payload,
        digest_field="provenance_sha256",
    )
    assert payload["transfer"]["direct_lan"] is True
    assert payload["transfer"]["jump_server_used_for_data"] is False
    assert payload["transfer"]["post_transfer_member_rehash_passed"] is True
    assert payload["transfer"]["staged_member_count"] == 5583
    assert payload["transfer"]["staged_byte_count"] == 5416488098
    assert payload["target_mesh_geometry_decoded_before_barrier"] is False
    assert payload["target_metric_computed_before_barrier"] is False
    assert len(payload["takes"]) == 12
