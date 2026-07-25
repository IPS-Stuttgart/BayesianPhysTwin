import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    REPOSITORY_ROOT
    / "results"
    / "sota"
    / "pokeflex_action_guard_development_v1"
    / "summary.json"
)


def test_pokeflex_action_guard_result_preserves_failed_gate() -> None:
    result = json.loads(SUMMARY.read_text(encoding="utf-8"))
    prospective = result["prospective_validation"]

    assert result["candidate_lock_sha256"] == (
        "4796ca2cf1f45d9e6cd810de13650126ef3f9dba48087f3d839754d2c37630c6"
    )
    assert prospective["take_count"] == 20
    assert prospective["frame_count"] == 1418
    assert prospective["gates"]["minimum_relative_improvement"]["passed"] is False
    assert prospective["gates"]["all_passed"] is False
    assert prospective["decision"].startswith("FAILED")


def test_pokeflex_action_guard_result_keeps_unopened_boundaries() -> None:
    result = json.loads(SUMMARY.read_text(encoding="utf-8"))
    boundaries = result["unopened_boundaries"]

    assert boundaries == {
        "calibration_objects": True,
        "development_take_T2": True,
        "development_takes": ["T7", "T8"],
        "target_objects": True,
    }
    artifacts = result["source_artifacts"]
    assert len(artifacts) == 60
    assert len({(item["take"], item["role"]) for item in artifacts}) == 60
    assert all(len(item["sha256"]) == 64 for item in artifacts)
