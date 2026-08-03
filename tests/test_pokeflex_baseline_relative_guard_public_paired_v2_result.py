from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (
    ROOT
    / "results"
    / "sota"
    / "pokeflex_baseline_relative_guard_public_paired_v2"
)
RESULT = RESULT_ROOT / "target_result.json"
BARRIER = RESULT_ROOT / "prediction_barrier.json"
EXECUTION = RESULT_ROOT / "execution_manifest.json"
POSTOPEN_AUDIT = RESULT_ROOT / "postopen_audit.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_registered_result_is_frozen_and_fails_only_declared_transfer_gates() -> None:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    aggregate = payload["aggregate"]

    assert _sha256(RESULT) == (
        "aa2680cbe0d7a6c9e342c9093ff4045e25a6952191fc9033376516d468329685"
    )
    assert payload["target_meshes_opened_after_complete_barrier"] is True
    assert aggregate["baseline_object_balanced_CD_UL1_mm"] == pytest.approx(
        5.569258167533781
    )
    assert aggregate["candidate_object_balanced_CD_UL1_mm"] == pytest.approx(
        5.548629659593054
    )
    assert aggregate["object_balanced_relative_CD_UL1_improvement"] == pytest.approx(
        0.0037039956346397907
    )
    assert aggregate["bootstrap_upper_candidate_minus_baseline_CD_UL1_mm"] < 0.0
    assert aggregate["supported_object_count"] == 10
    assert aggregate["object_win_count"] == 9
    assert aggregate["object_tie_count"] == 2
    assert aggregate["object_loss_count"] == 1
    assert aggregate["paired_transfer_passed"] is False
    assert aggregate["all_target_gates_passed"] is False


def test_barrier_and_execution_manifest_preserve_preoutcome_custody() -> None:
    barrier = json.loads(BARRIER.read_text(encoding="utf-8"))
    execution = json.loads(EXECUTION.read_text(encoding="utf-8"))

    assert _sha256(BARRIER) == (
        "904ce0b994b5dd5ad4b3cc943709497d187b0cef7c26cbfea796c2663c6a2ec4"
    )
    assert barrier["prediction_count"] == 12
    assert barrier["implementation_revision"] == (
        "1185d9fb5e4a8c8f08df8d315bacbc64fbccf2f1"
    )
    assert barrier["target_mesh_opened"] is False
    assert barrier["scoring_authorized"] is True
    assert execution["initial_preoutcome_attempt"]["prediction_seal_created"] is False
    assert execution["causal_staging"]["future_target_mesh_read_count"] == 0
    assert execution["prediction"]["prediction_seal_count"] == 12
    assert execution["prediction"]["fallback_mismatch_count"] == 0
    assert execution["target_scoring"]["paired_transfer_passed"] is False
    assert execution["held_v8_accessed"] is False


def test_postopen_audit_proves_threshold_tuning_cannot_pass() -> None:
    payload = json.loads(POSTOPEN_AUDIT.read_text(encoding="utf-8"))
    audit = payload["audit"]

    assert _sha256(POSTOPEN_AUDIT) == (
        "3cec4bf48332be6ae164a6dfdd47f56dec4b186d65a7b74af614d71e176101a2"
    )
    assert audit["accepted_scored_frame_count"] == 137
    assert audit["accepted_improving_frame_count"] == 126
    assert audit["accepted_harmful_frame_count"] == 11
    assert audit["accepted_false_safe_rate"] == pytest.approx(
        0.08029197080291971
    )
    assert audit["accepted_upper_bound_coverage"] == pytest.approx(
        0.8978102189781022
    )
    assert audit["maximum_zero_loss_win_count_from_sealed_candidates"] == 8
    oracle = audit["frame_oracle_within_sealed_candidates"]
    assert oracle["object_win_count"] == 9
    assert oracle["object_tie_count"] == 3
    assert oracle["object_loss_count"] == 0
