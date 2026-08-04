import json
from pathlib import Path

import pytest

from bayesian_phystwin.pokeflex_conservative_shrinkage_target import (
    canonical_payload_sha256,
    evaluate_target_metrics,
    file_sha256,
    load_pokeflex_shrinkage_target_protocol,
    prediction_seal_sha256,
    validate_prediction_barrier,
)

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "sota" / "pokeflex_action_robust_official13_public_v1"
PROTOCOL_PATH = (
    ROOT / "configs" / "sota" / "pokeflex_action_robust_official13_public_v1.json"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_action_robust_official13_result_is_bound_and_reproducible() -> None:
    protocol = load_pokeflex_shrinkage_target_protocol(PROTOCOL_PATH)
    result_path = RESULT_ROOT / "target_result.json"
    barrier_path = RESULT_ROOT / "prediction_barrier.json"
    summary_path = RESULT_ROOT / "summary.json"
    result = _load(result_path)
    barrier = _load(barrier_path)
    summary = _load(summary_path)

    assert file_sha256(result_path) == (
        "619c46726aab0f7e81d2e943bd44820e521c9fe6285906add28af87203c15ebd"
    )
    assert file_sha256(barrier_path) == (
        "666ffe0df7f485ee48864598337e5671fcc43169a97a223f12ef6442208cd3e5"
    )
    assert file_sha256(summary_path) == (
        "9e4ff075432806c1d9b16da54d31242468c2bb5aa047a5950bcb399b3bbf8ece"
    )
    assert summary["summary_sha256"] == canonical_payload_sha256(
        summary,
        digest_field="summary_sha256",
    )
    assert summary["summary_sha256"] == (
        "a7797cb5b318cb54e84c3cee16f33206a39fed81d5de0090409e2dc4ca00e6cf"
    )
    assert validate_prediction_barrier(barrier, protocol)["passed"] is True
    assert evaluate_target_metrics(result["objects"], protocol) == result["aggregate"]
    assert result["aggregate"]["all_target_gates_passed"] is True
    assert result["aggregate"]["published_direct_comparison_authorized"] is False


def test_action_robust_official13_improves_both_physical_references() -> None:
    summary = _load(RESULT_ROOT / "summary.json")
    frame = summary["frame_balanced"]
    direct = summary["action_robust_vs_checkpoint"]
    advancement = summary["action_robust_vs_global"]

    assert frame["action_robust_CD_UL1_mm"] == pytest.approx(6.447848304173961)
    assert frame["global_scale_CD_UL1_mm"] == pytest.approx(6.4993172114797195)
    assert frame["checkpoint_CD_UL1_mm"] == pytest.approx(6.5694225302682865)
    assert direct["relative_CD_UL1_improvement"] == pytest.approx(0.01862530999033738)
    assert direct["win_count"] == 12
    assert direct["tie_count"] == 1
    assert direct["bootstrap_upper_candidate_minus_reference_CD_UL1_mm"] < 0
    assert advancement["relative_CD_UL1_improvement"] == pytest.approx(
        0.008307443325929541
    )
    assert advancement["win_count"] == 6
    assert advancement["tie_count"] == 7
    assert advancement["bootstrap_upper_candidate_minus_reference_CD_UL1_mm"] < 0
    assert summary["global_scale_reproduction_passed"] is True
    assert summary["public_subset_numeric_reference_passed"] is True
    assert summary["published_direct_comparison_authorized"] is False
    assert summary["prospective_take_count"] == 0
    assert summary["retrospective_take_count"] == 13


def test_action_robust_official13_seals_match_barrier_and_fallbacks() -> None:
    protocol = _load(PROTOCOL_PATH)
    barrier = _load(RESULT_ROOT / "prediction_barrier.json")
    multipliers = protocol["method"]["action_robust_scale_calibration"]["multipliers"]

    for row in barrier["predictions"]:
        take_id = row["take_id"]
        seal_path = RESULT_ROOT / "prediction_seals" / take_id / "seal.json"
        seal = _load(seal_path)
        assert file_sha256(seal_path) == row["seal_file_sha256"]
        assert seal["seal_sha256"] == prediction_seal_sha256(seal)
        assert seal["future_mesh_read"] is False
        assert seal["implementation_clean"] is True
        assert seal["fallback_mismatch_count"] == 0
        assert seal["global_fallback_mismatch_count"] == 0
        assert seal["correction_multiplier"] == multipliers[seal["object_name"]]


def test_action_robust_official13_provenance_is_canonical() -> None:
    path = RESULT_ROOT / "scoring_provenance.json"
    payload = _load(path)

    assert file_sha256(path) == (
        "79fb1ffae8b2e37415af933ef2b231cbc75fd35b56f927f8e922ad92e8a28821"
    )
    assert payload["provenance_sha256"] == canonical_payload_sha256(
        payload,
        digest_field="provenance_sha256",
    )
    assert payload["provenance_sha256"] == (
        "a6ab77204215e1e7691cdf34399b28d6db753e278392796e3d83c02b29541835"
    )
    assert payload["transfer"]["direct_lan"] is True
    assert payload["transfer"]["jump_server_used_for_data"] is False
    assert payload["transfer"]["post_transfer_rehash_passed"] is True
    assert (
        payload["historical_exposure"][
            "all_thirteen_target_outcomes_previously_opened_under_prior_method"
        ]
        is True
    )
    assert payload["historical_exposure"]["new_method_is_prospective"] is False
