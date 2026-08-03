from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bayesian_phystwin.pokeflex_conservative_shrinkage_target import (
    evaluate_target_metrics,
    validate_prediction_barrier,
)

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = (
    ROOT / "results" / "sota" / "pokeflex_conservative_shrinkage_public_paired_v1"
)
PROTOCOL = (
    ROOT / "configs" / "sota" / "pokeflex_conservative_shrinkage_public_paired_v1.json"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_public_paired_result_reproduces_registered_near_pass() -> None:
    protocol = _load(PROTOCOL)
    result_path = RESULT_DIR / "target_result.json"
    result = _load(result_path)
    aggregate = evaluate_target_metrics(result["objects"], protocol)

    assert _sha256(result_path) == (
        "d3a03ca0c5f834cb5ae8fff840027e4ae919d3f94451a11847ab0b319221bb3c"
    )
    assert aggregate == result["aggregate"]
    assert aggregate["baseline_object_balanced_CD_UL1_mm"] == pytest.approx(
        4.703036315664229
    )
    assert aggregate["candidate_object_balanced_CD_UL1_mm"] == pytest.approx(
        4.656542200240883
    )
    assert aggregate["object_balanced_relative_CD_UL1_improvement"] == pytest.approx(
        0.009885978398357111
    )
    assert aggregate[
        "bootstrap_upper_candidate_minus_baseline_CD_UL1_mm"
    ] == pytest.approx(-0.029266977982326744)
    assert aggregate["object_win_count"] == 13
    assert aggregate["supported_object_count"] == 14
    assert aggregate["minimum_per_object_relative_improvement"] == pytest.approx(
        -0.004585143854373038
    )
    assert aggregate["paired_transfer_passed"] is False
    assert aggregate["all_target_gates_passed"] is False


def test_public_paired_result_has_one_loss_and_one_exact_fallback() -> None:
    result = _load(RESULT_DIR / "target_result.json")
    objects = result["objects"]
    losses = [
        row
        for row in objects
        if row["candidate_mean_CD_UL1_mm"] > row["baseline_mean_CD_UL1_mm"]
    ]
    ties = [
        row
        for row in objects
        if row["candidate_mean_CD_UL1_mm"] == row["baseline_mean_CD_UL1_mm"]
    ]

    assert [row["take_id"] for row in losses] == ["Sponge_T1"]
    assert [row["take_id"] for row in ties] == ["PlushVolleyball_T1"]
    assert ties[0]["supported_frame_count"] == 0
    assert sum(row["scored_frame_count"] for row in objects) == 1165
    assert sum(row["supported_frame_count"] for row in objects) == 1010
    assert sum(row["candidate_jaccard_valid_count"] for row in objects) == 868


def test_public_paired_custody_artifacts_are_bound_and_barrier_valid() -> None:
    protocol = _load(PROTOCOL)
    barrier_path = RESULT_DIR / "prediction_barrier.json"
    barrier = _load(barrier_path)
    causal = _load(RESULT_DIR / "causal_input_manifest.json")
    target = _load(RESULT_DIR / "authorized_target_manifest.json")

    assert _sha256(barrier_path) == (
        "2752852e4c80e2444b70864d2df598e8efbc8e5aee84684c863d5b126a800e66"
    )
    assert validate_prediction_barrier(barrier, protocol)["passed"]
    assert barrier["implementation_revision"] == (
        "c015534497c2a37aacd5c059ad48bed3721b6e3f"
    )
    assert causal["future_mesh_member_read_count"] == 0
    assert causal["target_outcome_scored"] is False
    assert target["target_access_after_complete_barrier"] is True
    assert target["barrier_file_sha256"] == _sha256(barrier_path)
    assert target["target_mesh_member_read_count"] == 1165
    assert target["non_scored_mesh_member_read_count"] == 0


def test_public_paired_retry_preserves_technical_failure_accounting() -> None:
    first_text = (RESULT_DIR / "prediction_attempt1.log").read_text(encoding="utf-8")
    payload_text, marker = first_text.rsplit("\nprediction failures: ", maxsplit=1)
    first = json.loads(payload_text)
    second = _load(RESULT_DIR / "prediction_attempt2.json")

    assert marker.strip() == "9"
    assert sum(row["returncode"] != 0 for row in first) == 9
    assert len(second) == 9
    assert all(row["returncode"] == 0 for row in second)
    assert _sha256(RESULT_DIR / "prediction_attempt1.log") == (
        "d3ccb6ef9d67abd033f2177d63bd6fb89493f794c5adf80e7071ab0b55eedf45"
    )
    assert _sha256(RESULT_DIR / "prediction_attempt2.json") == (
        "baf7bb6348b7edd9d1bca957fe020942c9121636c61eabe3e35a8753ec63ae4a"
    )
