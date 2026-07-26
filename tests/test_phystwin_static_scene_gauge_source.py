from __future__ import annotations

import json
from pathlib import Path

import pytest

from bayesian_phystwin.phystwin_static_scene_gauge_source import (
    StaticSceneGaugeSourceGate,
    aggregate_phystwin_static_scene_gauge_source,
)


def _write_result(
    path: Path,
    *,
    case: str,
    raw: tuple[float, float, float],
    corrected: tuple[float, float, float],
) -> None:
    metric_names = ("mean_error_mm", "rmse_mm", "late_mean_error_mm")
    payload = {
        "artifact_kind": "PhysTwinStaticSceneGaugePrefixCompetenceV1",
        "case": case,
        "raw": {
            **dict(zip(metric_names, raw, strict=True)),
            "point_frame_count": 20,
        },
        "static_scene_gauge": {
            **dict(zip(metric_names, corrected, strict=True)),
            "point_frame_count": 20,
        },
        "support": {
            "common_point_frame_count": 20,
            "common_point_frame_fraction": 0.5,
            "gauge_supported_dense_fraction": 0.4,
        },
        "inputs": {"gauge_sha256": f"{case}-gauge"},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_source_aggregate_passes_fixed_gate(tmp_path: Path) -> None:
    paths = []
    cases = [f"case-{index}" for index in range(3)]
    for case in cases:
        path = tmp_path / f"{case}.json"
        _write_result(
            path,
            case=case,
            raw=(10.0, 12.0, 14.0),
            corrected=(9.0, 10.8, 12.6),
        )
        paths.append(path)

    result = aggregate_phystwin_static_scene_gauge_source(
        paths,
        expected_cases=cases,
        gate=StaticSceneGaugeSourceGate(minimum_mean_error_wins=3),
    )

    assert result["gate_passed"] is True
    assert result["aggregate"]["mean_error_mm"]["case_wins"] == 3
    assert result["aggregate"]["mean_error_mm"][
        "relative_improvement"
    ] == pytest.approx(0.1)


def test_source_aggregate_rejects_missing_or_support_changed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "case.json"
    _write_result(
        path,
        case="case",
        raw=(10.0, 12.0, 14.0),
        corrected=(9.0, 10.8, 12.6),
    )

    with pytest.raises(ValueError, match="missing expected cases"):
        aggregate_phystwin_static_scene_gauge_source(
            [path],
            expected_cases=["case", "missing"],
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["static_scene_gauge"]["point_frame_count"] = 19
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="common raw/corrected support"):
        aggregate_phystwin_static_scene_gauge_source(
            [path],
            expected_cases=["case"],
        )
