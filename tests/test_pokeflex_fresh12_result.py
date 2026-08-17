import json
from pathlib import Path

from bayesian_phystwin.pokeflex_conservative_shrinkage_target import (
    canonical_payload_sha256,
    file_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (
    ROOT / "results" / "sota" / "pokeflex_conservative_shrinkage_fresh12_public_v1"
)


def test_fresh12_result_is_bound_and_passes_locked_gates() -> None:
    result_path = RESULT_ROOT / "target_result.json"
    barrier_path = RESULT_ROOT / "prediction_barrier.json"
    summary = json.loads((RESULT_ROOT / "summary.json").read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))

    assert file_sha256(result_path) == (
        "3355769d8994ea70c421f4c009fb180a33bf0a72b7aaa8cf4efa37aecced902f"
    )
    assert file_sha256(barrier_path) == (
        "e9ba6abba60c3ac0b4bfb34003cc05053f940879996704c11ecaac8f6bdbff25"
    )
    assert summary["target_result_file_sha256"] == file_sha256(result_path)
    assert summary["prediction_barrier_file_sha256"] == file_sha256(barrier_path)
    assert summary["implementation_revision"] == (
        "a93e88edd1a19e0ccaf6afdf9e0c9b4ba78c7cde"
    )
    assert result["aggregate"]["paired_transfer_passed"] is True
    assert result["aggregate"]["all_target_gates_passed"] is True
    assert result["aggregate"]["fresh12_object_win_count"] == 11
    assert result["aggregate"]["fresh12_exact_fallback_tie_count"] == 1
    assert result["aggregate"]["published_direct_comparison_authorized"] is False


def test_fresh12_scoring_provenance_is_canonical() -> None:
    path = RESULT_ROOT / "scoring_provenance.json"
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["provenance_sha256"] == canonical_payload_sha256(
        payload,
        digest_field="provenance_sha256",
    )
    assert payload["transfer"]["direct_lan"] is True
    assert payload["transfer"]["jump_server_used_for_data"] is False
    assert payload["transfer"]["post_transfer_member_rehash_passed"] is True
    assert len(payload["takes"]) == 12
    assert all(row["staged_member_count"] > 0 for row in payload["takes"])
