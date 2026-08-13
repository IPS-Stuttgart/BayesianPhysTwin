import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
PROTOCOL = ROOT / "configs/sota/matphys_cotracker_selected_overlay_source_v1.json"
RESULT = (
    ROOT
    / "results/sota/diagnostics/matphys_cotracker_selected_overlay_source_v1/decision.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_source_decision_and_provenance() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))

    assert _sha256(PROTOCOL) == (
        "8b044af71530cda3c374c03f31a1c7eb4d8891d8ba3686b76f372b88c0ce6678"
    )
    assert _sha256(RESULT) == (
        "87ebe1aa94785e7e7f50bd1f0975ea65417786e6fa4cb79330d8d6d77ef03201"
    )
    assert protocol["decision_arms"]["primary"] == result["primary_candidate"]
    assert protocol["observation"]["manual_prefix_override"] is False
    assert result["input"]["sha256"] == (
        "67d4d222172610d571f0691ea0c60abd4f6fd34e68552a4750a38838afc12927"
    )
    assert result["case_count"] == 22
    assert result["gate_passed"] is False
    assert result["decision"] == "stop-selected-overlay-cotracker-composition"


def test_registered_arm_fails_track_and_regression_gates() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    arm = result["arms"]["primary_causal_temporal"]

    assert arm["candidate_equal_case_mean"]["chamfer_distance_m"] == (
        0.008574563780290666
    )
    assert arm["candidate_equal_case_mean"]["track_error_m"] == (0.022808115881904453)
    assert arm["relative_improvement_percent"]["chamfer_distance_m"] > 5.0
    assert arm["relative_improvement_percent"]["track_error_m"] < 0.0
    assert arm["joint_case_wins"] == 7
    assert arm["maximum_case_metric_regression_percent"] > 100.0
