from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bayesian_phystwin.pokeflex_conservative_shrinkage_target import (
    OFFICIAL13_PUBLIC_PROSPECTIVE_TAKE_IDS,
    OFFICIAL13_PUBLIC_TARGET_TAKE_IDS,
    evaluate_target_metrics,
    validate_pokeflex_shrinkage_target_protocol,
    validate_prediction_barrier,
)

ROOT = Path("results/sota/pokeflex_conservative_shrinkage_official13_public_v1")
RESULT = ROOT / "target_result.json"
BARRIER = ROOT / "prediction_barrier.json"
PROTOCOL = Path(
    "configs/sota/pokeflex_conservative_shrinkage_official13_public_v1.json"
)


def test_official13_public_result_recomputes_and_passes_frozen_gates() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    validate_pokeflex_shrinkage_target_protocol(protocol)
    assert evaluate_target_metrics(result["objects"], protocol) == result["aggregate"]
    assert result["aggregate"]["all_target_gates_passed"] is True
    assert result["aggregate"]["paired_transfer_passed"] is True
    assert result["aggregate"]["prospective_object_win_count"] == 9
    assert result["aggregate"][
        "prospective_object_balanced_relative_CD_UL1_improvement"
    ] == pytest.approx(0.01045989693087063)
    assert result["aggregate"][
        "prospective_bootstrap_upper_candidate_minus_baseline_CD_UL1_mm"
    ] == pytest.approx(-0.043774239785478744)


def test_official13_public_result_preserves_barrier_and_claim_boundary() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    barrier = json.loads(BARRIER.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == (
        "0bc3a16f2f91a556d624b1b45fa8aca4063736085d474ee889ff99018901dbe4"
    )
    assert hashlib.sha256(BARRIER.read_bytes()).hexdigest() == (
        "caa72b62f99b836d386d96b668797564f26d4b4629552ca8aec45e0caa71651c"
    )
    assert validate_prediction_barrier(barrier, protocol)["passed"] is True
    assert result["barrier_sha256"] == barrier["barrier_sha256"]
    assert result["target_meshes_opened_after_complete_barrier"] is True
    assert result["aggregate"]["published_reference_is_contextual_only"] is True
    assert result["aggregate"]["published_direct_comparison_authorized"] is False


def test_official13_public_result_has_complete_nonregressing_accounting() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    by_take = {row["take_id"]: row for row in result["objects"]}
    frames = [frame for row in result["objects"] for frame in row["frames"]]

    assert tuple(by_take) == OFFICIAL13_PUBLIC_TARGET_TAKE_IDS
    assert len(frames) == 970
    assert sum(int(frame["update_supported"]) for frame in frames) == 835
    assert {
        frame["candidate_jaccard_error"] for frame in frames
    } == {"ValueError: Not all meshes are volumes!"}
    assert all(
        by_take[take_id]["candidate_mean_CD_UL1_mm"]
        <= by_take[take_id]["baseline_mean_CD_UL1_mm"]
        for take_id in OFFICIAL13_PUBLIC_PROSPECTIVE_TAKE_IDS
    )
    exact_ties = [
        take_id
        for take_id in OFFICIAL13_PUBLIC_PROSPECTIVE_TAKE_IDS
        if by_take[take_id]["candidate_mean_CD_UL1_mm"]
        == by_take[take_id]["baseline_mean_CD_UL1_mm"]
    ]
    assert exact_ties == ["PlushVolleyball_T4"]
    assert by_take["PlushVolleyball_T4"]["supported_frame_count"] == 0
