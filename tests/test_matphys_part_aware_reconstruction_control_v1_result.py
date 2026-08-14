from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = (
    Path(__file__).parents[1]
    / "results"
    / "sota"
    / "diagnostics"
    / "matphys_part_aware_reconstruction_control_v1"
)


def _load(name: str) -> dict[str, object]:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def _sha256(name: str) -> str:
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def test_terminal_capacity_result_is_bound_and_nonpredictive() -> None:
    result = _load("result.json")

    assert result["contract"] == "matphys-all-frame-part-aware-reconstruction-result-v1"
    assert result["future_observations_used"] is True
    assert result["predictive_use_authorized"] is False
    assert result["published_matphys_method"] is False
    assert result["decision"] == {
        "advance_to_source_only_causal_design": True,
        "authorizes_predictive_use_of_checkpoint": False,
        "backend_export_pass": True,
        "capacity_pass": True,
    }
    assert result["terminal_test_metrics_mm"] == pytest.approx(
        {"chamfer_distance": 11.31706337061313, "track_error": 17.356220613624455}
    )
    assert result["released_phystwin_test_metrics_mm"] == pytest.approx(
        {"chamfer_distance": 17.8600324887817, "track_error": 24.926414339721696}
    )
    assert result["improvement_percent"] == pytest.approx(
        {
            "chamfer_distance_m": 36.63469885778965,
            "track_error_m": 30.37016725680314,
        }
    )
    assert result["backend_export"] == {
        "complete_positive_spring_field": True,
        "finite_global_parameter_count": 7,
        "finite_moved_part_adapter": True,
        "spring_count": 110833,
    }
    assert _sha256("result.json") == (
        "4a47e5f41a8838dde5aa0abfb405f508ad169c2056c07d80e836408ebb4f7af3"
    )


def test_terminal_checkpoint_and_spatial_adapter_audits_pass() -> None:
    finiteness = _load("checkpoint_finiteness.json")
    adapter = _load("part_adapter_terminal_audit.json")
    spatial = _load("part_spring_field_audit.json")
    export = _load("export_manifest.json")
    metrics = _load("official_reconstruction_metrics.json")
    training = _load("training_audit.json")

    checkpoint_sha256 = (
        "c8407722279b04022804e032fb433b2f6156863057363d656ead970628116f26"
    )
    assert finiteness["finite"] is True
    assert finiteness["model_nonfinite_count"] == 0
    assert finiteness["optimizer_nonfinite_count"] == 0
    assert finiteness["optimizer_rank_summaries"] == [
        {
            "accepted_steps": 16800,
            "attempted_steps": 16800,
            "rejected_post_step": 0,
            "rejected_pre_step": 0,
        }
    ]
    assert finiteness["checkpoint"]["sha256"] == checkpoint_sha256
    assert training["checkpoint"]["sha256"] == checkpoint_sha256
    assert training["future_observations_used"] is True
    assert training["predictive_use_authorized"] is False
    assert training["training_configuration"]["epochs"] == 200

    assert adapter["finite"] is True
    assert adapter["adapter_moved_from_zero"] is True
    weight = adapter["tensors"]["part_feature_encoder.1.weight"]
    assert weight["finite"] is True
    assert weight["nonzero_count"] == 32768

    assert export["checkpoint"]["sha256"] == checkpoint_sha256
    assert export["case"]["spring_field"]["count"] == 110833
    assert export["predictive_use_authorized"] is False
    variation = spatial["summary"]["spatial_variation"]
    assert variation["distinct_part_count"] == 5
    assert variation["maximum_to_minimum_part_geometric_mean_ratio"] == pytest.approx(
        66.24784832997477
    )
    assert metrics["evaluation"]["test"] == pytest.approx(
        {
            "chamfer_distance_m": 0.011317063370613131,
            "frame_count": 26,
            "frame_end_exclusive": 85,
            "frame_start": 59,
            "track_error_m": 0.017356220613624455,
        }
    )
    assert _sha256("part_spring_field_audit.json") == (
        "329945a6fa2cfc13ab1ce9e9dc6bddfbb5562c75f6245d4053ff7fd3bdcb52eb"
    )
