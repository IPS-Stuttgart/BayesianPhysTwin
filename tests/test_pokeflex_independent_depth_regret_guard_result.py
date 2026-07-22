import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
SOURCE = (
    ROOT
    / "results"
    / "sota"
    / "pokeflex_independent_depth_regret_guard_source_v1"
    / "source_cross_object_evaluation.json"
)
CURRENT_STATE = (
    ROOT
    / "results"
    / "sota"
    / "pokeflex_independent_depth_state_update_kill_v1"
    / "current_state_two_take_summary.json"
)
DIRECT_STATE = CURRENT_STATE.with_name("direct_d405_foamdice_summary.json")
PROSPECTIVE = (
    ROOT
    / "results"
    / "sota"
    / "pokeflex_independent_depth_regret_guard_prospective_v1"
    / "prospective_evaluation.json"
)
EXECUTION_MANIFEST = PROSPECTIVE.with_name("execution_manifest.json")
PROTOCOL = (
    ROOT
    / "configs"
    / "sota"
    / "pokeflex_independent_depth_regret_guard_prospective_v1.json"
)


def test_source_guard_and_rejected_state_branches_are_frozen() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    current = json.loads(CURRENT_STATE.read_text(encoding="utf-8"))
    direct = json.loads(DIRECT_STATE.read_text(encoding="utf-8"))

    assert source["cross_object"]["gate_passed"] is True
    assert source["cross_object"]["object_balanced_relative_improvement"] > 0.01
    assert source["cross_object"]["object_wins"] == 4
    assert source["cross_object"]["maximum_object_regression"] == 0.0
    assert current["object_balanced_selector"]["relative_improvement"] < 0.0
    assert current["objects"][0]["relative_improvement"] < -0.07
    assert direct["result"]["relative_improvement"] < -0.05
    assert direct["decision"].startswith("stop")


def test_prospective_protocol_references_exact_source_result() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()

    assert protocol["source_evidence"]["result_sha256"] == digest
    assert protocol["prospective_cohort"]["target_objects_remain_sealed"] is True


def test_prospective_replication_passed_without_opening_target_objects() -> None:
    result = json.loads(PROSPECTIVE.read_text(encoding="utf-8"))
    manifest = json.loads(EXECUTION_MANIFEST.read_text(encoding="utf-8"))

    assert result["gate_passed"] is True
    assert result["object_balanced_relative_improvement"] > 0.03
    assert result["object_wins"] == 2
    assert result["object_losses"] == 0
    assert result["accepted_frame_wins"] == 77
    assert result["accepted_frame_losses"] == 10
    assert result["exact_fallback_frame_count"] == 154
    assert manifest["replacement_performed"] is False
    assert manifest["calibration_objects_opened"] is False
    assert manifest["target_objects_opened"] is False
    assert manifest["prospective_evaluation"]["sha256"] == hashlib.sha256(
        PROSPECTIVE.read_bytes()
    ).hexdigest()
