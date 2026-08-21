from __future__ import annotations

from typing import cast

import pytest

from bayesian_phystwin_experiments.matphys_residual_overlay_audit import (
    METRICS,
    summarize_matphys_overlay_rows,
)


def _row(
    case: str,
    *,
    backbone: tuple[float, float],
    bayesian: tuple[float, float],
    persistence: tuple[float, float],
    selected: tuple[float, float],
) -> dict[str, object]:
    def metrics(values: tuple[float, float]) -> dict[str, float]:
        return dict(zip(METRICS, values, strict=True))

    return {
        "case": case,
        "methods": {
            "backbone": metrics(backbone),
            "bayesian_anchor": metrics(bayesian),
            "last_residual": metrics(persistence),
            "validation_selected": metrics(selected),
        },
    }


def test_summary_reports_matched_changes_wins_and_worst_case() -> None:
    rows = [
        _row(
            "one",
            backbone=(0.010, 0.020),
            bayesian=(0.008, 0.018),
            persistence=(0.009, 0.017),
            selected=(0.008, 0.018),
        ),
        _row(
            "two",
            backbone=(0.020, 0.010),
            bayesian=(0.021, 0.009),
            persistence=(0.020, 0.011),
            selected=(0.021, 0.009),
        ),
    ]

    result = summarize_matphys_overlay_rows(rows)

    assert result["case_count"] == 2
    means = cast(dict[str, dict[str, float]], result["equal_case_mean"])
    assert means["backbone"] == pytest.approx(
        {"chamfer_distance_m": 0.015, "track_error_m": 0.015}
    )
    comparisons = cast(dict[str, dict[str, object]], result["comparisons_vs_backbone"])
    bayesian = comparisons["bayesian_anchor"]
    assert bayesian["joint_case_wins"] == 1
    assert bayesian["case_wins"] == {
        "chamfer_distance_m": 1,
        "track_error_m": 2,
    }
    assert bayesian["worst_case_ratio"] == pytest.approx(
        {"chamfer_distance_m": 1.05, "track_error_m": 0.9}
    )


def test_summary_rejects_empty_cohort() -> None:
    with pytest.raises(ValueError, match="at least one case"):
        summarize_matphys_overlay_rows([])


def test_summary_rejects_nonpositive_backbone_metric() -> None:
    row = _row(
        "bad",
        backbone=(0.0, 0.01),
        bayesian=(0.01, 0.01),
        persistence=(0.01, 0.01),
        selected=(0.01, 0.01),
    )

    with pytest.raises(ValueError, match="backbone metric"):
        summarize_matphys_overlay_rows([row])
