from pathlib import Path

from bayesian_phystwin.deform360_bias_aware_prospective_v2_protocol import (
    load_bias_aware_prospective_v2_protocol,
)
from bayesian_phystwin.deform360_bias_aware_prospective_v2_runtime import (
    prospective_v2_case_records,
)
from bayesian_phystwin.deform360_bias_aware_prospective_v2_support import (
    calibration_support_summary,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/sota/deform360_bias_aware_guarded_belief_prospective_v2.json"


def _rows(automatic_objects: set[str]) -> list[dict[str, object]]:
    fresh = {"078-fishing-line", "161-tube", "088-snake"}
    return [
        {
            **record,
            "origin": ("fresh_v2" if record["object_id"] in fresh else "inherited_v1"),
            "automatic_twin": record["object_id"] in automatic_objects,
        }
        for record in prospective_v2_case_records(PROTOCOL, role="calibration")
    ]


def _summary(automatic_objects: set[str]) -> dict[str, object]:
    config = load_bias_aware_prospective_v2_protocol(PROTOCOL)["config"]
    return calibration_support_summary(
        _rows(automatic_objects),
        config["calibration_support_gate"],
        source_groups=config["method"]["source_group_count"],
    )


def test_v2_support_passes_with_inherited_five_and_fresh_three() -> None:
    summary = _summary(
        {
            "076-rubber-bands",
            "011-green-cloth",
            "175-plastic-bag-cloth",
            "163-bear",
            "168-cat-big",
            "078-fishing-line",
            "161-tube",
            "088-snake",
        }
    )

    assert summary["support_passed"] is True
    assert summary["automatic_twin_object_count"] == 8
    assert summary["automatic_twin_object_count_by_stratum"] == {
        "filament": 4,
        "sheet": 2,
        "volumetric": 2,
    }
    assert summary["fresh_filament_automatic_twin_count"] == 3
    assert summary["combined_eligible_object_group_count"] == 12
    assert summary["finite_sample_rank"] == 12
    assert summary["finite_sample_coverage"] == 12 / 13


def test_v2_support_fails_when_fresh_filament_repair_is_insufficient() -> None:
    summary = _summary(
        {
            "076-rubber-bands",
            "011-green-cloth",
            "175-plastic-bag-cloth",
            "163-bear",
            "168-cat-big",
            "078-fishing-line",
        }
    )

    assert summary["support_passed"] is False
    assert summary["fresh_filament_automatic_twin_count"] == 1
    assert "fresh_filament_automatic_twins_required" in summary["failed_support_gates"]
    assert "minimum_automatic_twin_objects" in summary["failed_support_gates"]


def test_persistence_fallback_does_not_count_as_support() -> None:
    automatic = {
        "076-rubber-bands",
        "011-green-cloth",
        "175-plastic-bag-cloth",
        "163-bear",
        "168-cat-big",
        "078-fishing-line",
        "161-tube",
    }
    rows = _rows(automatic)
    fallback = next(row for row in rows if row["object_id"] == "088-snake")
    fallback["disposition"] = "prediction"
    fallback["physical_mode"] = "persistence_fallback"
    fallback["automatic_twin"] = False
    config = load_bias_aware_prospective_v2_protocol(PROTOCOL)["config"]
    summary = calibration_support_summary(
        rows,
        config["calibration_support_gate"],
        source_groups=config["method"]["source_group_count"],
    )

    assert summary["automatic_twin_object_count"] == 7
    assert summary["fresh_filament_automatic_twin_count"] == 2
    assert summary["support_passed"] is True
