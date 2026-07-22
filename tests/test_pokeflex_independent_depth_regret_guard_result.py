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
