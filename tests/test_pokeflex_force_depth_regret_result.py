import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
RESULT_ROOT = ROOT / "results" / "sota" / "pokeflex_force_depth_regret_development_v1"
SOURCE = RESULT_ROOT / "source_only.json"
ALL_OPENED = RESULT_ROOT / "all_opened.json"
MANIFEST = RESULT_ROOT / "execution_manifest.json"


def test_source_result_passes_but_independent_object_transfer_fails() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    all_opened = json.loads(ALL_OPENED.read_text(encoding="utf-8"))

    assert source["cross_object"]["gate_passed"] is True
    assert source["cross_object"]["object_balanced_relative_improvement"] > 0.029
    assert source["cross_object"]["false_safe_rate"] < 0.061
    assert all_opened["cross_object"]["gate_passed"] is False
    assert all_opened["cross_object"]["object_balanced_relative_improvement"] < 0.0
    assert all_opened["cross_object"]["false_safe_rate"] > 0.28
    assert all_opened["cross_object"]["maximum_object_regression"] > 0.10


def test_selector_failure_is_not_only_conservative_calibration() -> None:
    result = json.loads(ALL_OPENED.read_text(encoding="utf-8"))

    assert (
        result["selector_controls"]["candidate_ucb"][
            "object_balanced_relative_improvement"
        ]
        < 0.0
    )
    assert (
        result["selector_controls"]["predicted_mean"][
            "object_balanced_relative_improvement"
        ]
        < 0.0
    )
    assert (
        result["candidate_bank_oracle"]["object_balanced_relative_improvement"] > 0.07
    )
    assert result["fixed_arm_controls"]["maximin"]["gate_passed"] is False


def test_manifest_freezes_evidence_and_keeps_target_sealed() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert (
        manifest["evaluations"]["source_only"]["sha256"]
        == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    )
    assert (
        manifest["evaluations"]["all_opened"]["sha256"]
        == hashlib.sha256(ALL_OPENED.read_bytes()).hexdigest()
    )
    assert manifest["replacement_performed"] is False
    assert manifest["target_protocol_authorized"] is False
    assert manifest["target_objects_opened"] is False
