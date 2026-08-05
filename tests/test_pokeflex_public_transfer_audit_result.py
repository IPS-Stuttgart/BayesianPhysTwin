import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from bayesian_phystwin.pokeflex_public_transfer_audit import result_sha256

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (
    ROOT
    / "results"
    / "sota"
    / "pokeflex_action_robust_public78_retrospective_v6"
)
RESULT_PATH = RESULT_ROOT / "result.json"
SUMMARY_PATH = RESULT_ROOT / "summary.json"
SUMMARY_BUILDER = (
    ROOT
    / "scripts"
    / "development"
    / "build_pokeflex_public_transfer_audit_summary.py"
)


def test_public_transfer_result_is_exact_and_passes_only_retrospective_gate() -> None:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert hashlib.sha256(RESULT_PATH.read_bytes()).hexdigest() == (
        "2e1e0ff91abd432461b198917d14059da38ec0254727b1ccddb2aa3d0a2a4340"
    )
    assert result["result_sha256"] == (
        "5cb382596d9e29fa1b1be20c8c40549c5e9c43769c3fab3c6168d5af8f7bcaaa"
    )
    assert result["result_sha256"] == result_sha256(result)
    assert result["decision"] == {
        "retrospective_interpretation_gate_passed": True,
        "prospective_v5_strict_advancement_passed": False,
        "current_method_establishes_strict_superiority_over_global": False,
        "retuning_from_public_outcomes_authorized": False,
        "independent_fresh_evaluation_required_for_advancement": True,
    }

    summary = result["retrospective"]["summary"]
    assert summary["object_count"] == 18
    assert summary["take_count"] == 78
    assert summary["frame_count"] == 5823
    assert summary["object_balanced"]["candidate_CD_UL1_mm"] == pytest.approx(
        5.268236710997882
    )
    assert summary["candidate_vs_checkpoint"][
        "object_balanced_relative_improvement"
    ] == pytest.approx(0.023988905722933634)
    assert summary["candidate_vs_global"][
        "object_balanced_relative_improvement"
    ] == pytest.approx(0.013196306558725942)
    assert summary["candidate_vs_checkpoint"]["object_win_count"] == 17
    assert summary["candidate_vs_global"]["object_win_count"] == 15
    assert summary["candidate_vs_global"]["object_loss_count"] == 1


def test_public_transfer_summary_rebuilds_and_accounts_for_exact_fallbacks() -> None:
    spec = importlib.util.spec_from_file_location(
        "public_transfer_summary_builder",
        SUMMARY_BUILDER,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    recorded = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    rebuilt = module.build_summary(
        RESULT_ROOT,
        ROOT
        / "configs"
        / "sota"
        / "pokeflex_action_robust_public78_retrospective_v6.json",
    )
    assert rebuilt == recorded
    assert hashlib.sha256(SUMMARY_PATH.read_bytes()).hexdigest() == (
        "8d6d30afc7773ec49cdf58248396fad10aa47a71176f1f11a297dd62ae7135e1"
    )
    assert recorded["summary_sha256"] == (
        "0e56c165d0bacf94ce950c9c636532cf045adea5992b363d7200bfa1e8f7d1f5"
    )
    assert len(recorded["files"]) == 237

    execution = recorded["execution_accounting"]
    assert execution["locked_take_count"] == 78
    assert execution["ordinary_prediction_count"] == 74
    assert execution["full_exact_checkpoint_fallback_count"] == 4
    assert execution["partial_exact_checkpoint_fallback_count"] == 0
    assert execution["unsealable_count"] == 0
    assert execution["fallback_target_frame_count"] == 337
    assert execution["total_scored_target_frame_count"] == 5823
    assert execution["all_fallback_candidates_equal_checkpoint"] is True
    assert execution["pose_imputation_used_by_prediction"] is False
    assert execution["source_robot_bytes_modified"] is False
    assert recorded["server_transfer"]["jump_server_in_payload_path"] is False
