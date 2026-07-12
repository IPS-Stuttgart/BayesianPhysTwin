import hashlib
import json
from pathlib import Path

import pytest


def _result_root() -> Path:
    return (
        Path(__file__).resolve().parents[1] / "runs" / "causal4d-real-undercoverage-v1"
    )


def test_real_undercoverage_result_bundle_is_checksummed() -> None:
    root = _result_root()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment"] == "causal4d-real-undercoverage-v1"
    assert not manifest["claim_ready"]
    for name, descriptor in manifest["artifacts"].items():
        content = (root / name).read_bytes()
        assert len(content) == descriptor["bytes"]
        assert hashlib.sha256(content).hexdigest() == descriptor["sha256"]


def test_support_and_graph_results_match_the_registered_diagnosis() -> None:
    root = _result_root()
    support = json.loads(
        (root / "parameter_support_audit.json").read_text(encoding="utf-8")
    )
    assert support["stable_counts"] == {
        "top_mass": 32,
        "weighted_coreset": 16,
    }
    top4 = next(
        row
        for row in support["candidates"]
        if row["method"] == "top_mass" and row["count"] == 4
    )
    full = next(
        row
        for row in support["candidates"]
        if row["method"] == "top_mass" and row["count"] == 81
    )
    assert top4["predictive"]["coverage"] == pytest.approx(0.5058833209907019)
    assert full["predictive"]["coverage"] == pytest.approx(0.5504571613883766)

    graph = json.loads(
        (root / "single_lift_graph_discrepancy.json").read_text(encoding="utf-8")
    )
    variants = {value["method"]: value for value in graph["variants"]}
    current = variants["current_random_walk_readout"]["groups"]["all"]
    persistence = variants["graph_persistence"]["groups"]["all"]
    assert persistence["track_error_m"] < current["track_error_m"]
    assert persistence["coverage"] > current["coverage"]
    assert variants["graph_persistence"]["groups"]["horizon:late"]["coverage"] < 0.90


def test_harmful_source_calibration_remains_rejected() -> None:
    root = _result_root()
    calibration = json.loads(
        (root / "affine_graph_persistence_calibration.json").read_text(encoding="utf-8")
    )
    evaluation = json.loads(
        (root / "affine_graph_persistence_evaluation.json").read_text(encoding="utf-8")
    )
    assert not calibration["claim_ready"]
    assert calibration["calibration_trial_count"] == 1
    target = evaluation["cases"][0]
    assert target["calibrated"]["all"]["coverage"] < target["raw"]["all"]["coverage"]
    assert evaluation["target_labels_used_for_calibration"] is False
