"""Aggregate opened-source transfer results for the PhysTwin static-scene gauge."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class StaticSceneGaugeSourceGate:
    """Frozen transfer criteria for the opened 21-case source cohort."""

    minimum_mean_error_gain: float = 0.03
    minimum_rmse_gain: float = 0.02
    minimum_late_error_gain: float = 0.02
    minimum_mean_error_wins: int = 14
    maximum_case_mean_error_regression: float = 0.10

    def __post_init__(self) -> None:
        for name in (
            "minimum_mean_error_gain",
            "minimum_rmse_gain",
            "minimum_late_error_gain",
            "maximum_case_mean_error_regression",
        ):
            value = float(getattr(self, name))
            _require(0.0 <= value < 1.0, f"{name} must lie in [0, 1)")
        _require(
            self.minimum_mean_error_wins >= 1,
            "minimum_mean_error_wins must be positive",
        )


def _load_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(
        payload.get("artifact_kind")
        == "PhysTwinStaticSceneGaugePrefixCompetenceV1",
        f"{path} is not a static-scene gauge competence result",
    )
    return payload


def aggregate_phystwin_static_scene_gauge_source(
    result_paths: Iterable[str | Path],
    *,
    expected_cases: Iterable[str],
    gate: StaticSceneGaugeSourceGate | None = None,
) -> dict[str, Any]:
    """Aggregate one sealed prefix-competence result per expected source case."""

    resolved_gate = gate or StaticSceneGaugeSourceGate()
    expected = tuple(str(case) for case in expected_cases)
    _require(len(expected) > 0, "expected_cases cannot be empty")
    _require(
        len(set(expected)) == len(expected),
        "expected_cases must be unique",
    )

    by_case: dict[str, dict[str, Any]] = {}
    for raw_path in result_paths:
        path = Path(raw_path)
        payload = _load_result(path)
        case = str(payload["case"])
        _require(case not in by_case, f"duplicate result for {case}")
        payload["_result_path"] = str(path)
        by_case[case] = payload

    missing = sorted(set(expected) - set(by_case))
    extra = sorted(set(by_case) - set(expected))
    _require(not missing, f"missing expected cases: {missing}")
    _require(not extra, f"unexpected cases: {extra}")

    metric_names = ("mean_error_mm", "rmse_mm", "late_mean_error_mm")
    rows: list[dict[str, Any]] = []
    for case in expected:
        payload = by_case[case]
        raw = payload["raw"]
        corrected = payload["static_scene_gauge"]
        improvements = {
            name: float(
                1.0
                - float(corrected[name])
                / max(float(raw[name]), 1e-12)
            )
            for name in metric_names
        }
        _require(
            int(raw["point_frame_count"])
            == int(corrected["point_frame_count"])
            == int(payload["support"]["common_point_frame_count"]),
            f"{case} does not use common raw/corrected support",
        )
        rows.append(
            {
                "case": case,
                "raw": {name: float(raw[name]) for name in metric_names},
                "static_scene_gauge": {
                    name: float(corrected[name]) for name in metric_names
                },
                "relative_improvement": improvements,
                "common_point_frame_count": int(
                    payload["support"]["common_point_frame_count"]
                ),
                "common_point_frame_fraction": float(
                    payload["support"]["common_point_frame_fraction"]
                ),
                "gauge_supported_dense_fraction": float(
                    payload["support"]["gauge_supported_dense_fraction"]
                ),
                "inputs": payload["inputs"],
                "result_path": payload["_result_path"],
            }
        )

    aggregate: dict[str, Any] = {}
    for name in metric_names:
        raw_values = np.asarray(
            [row["raw"][name] for row in rows],
            dtype=np.float64,
        )
        corrected_values = np.asarray(
            [row["static_scene_gauge"][name] for row in rows],
            dtype=np.float64,
        )
        relative = 1.0 - corrected_values / np.maximum(raw_values, 1e-12)
        aggregate[name] = {
            "raw_equal_case_mean": float(np.mean(raw_values)),
            "static_scene_gauge_equal_case_mean": float(
                np.mean(corrected_values)
            ),
            "relative_improvement": float(
                1.0
                - np.mean(corrected_values)
                / max(float(np.mean(raw_values)), 1e-12)
            ),
            "case_wins": int(np.sum(corrected_values < raw_values)),
            "case_ties": int(np.sum(corrected_values == raw_values)),
            "worst_case_relative_improvement": float(np.min(relative)),
        }

    gate_checks = {
        "mean_error_gain": (
            aggregate["mean_error_mm"]["relative_improvement"]
            >= resolved_gate.minimum_mean_error_gain
        ),
        "rmse_gain": (
            aggregate["rmse_mm"]["relative_improvement"]
            >= resolved_gate.minimum_rmse_gain
        ),
        "late_error_gain": (
            aggregate["late_mean_error_mm"]["relative_improvement"]
            >= resolved_gate.minimum_late_error_gain
        ),
        "mean_error_wins": (
            aggregate["mean_error_mm"]["case_wins"]
            >= resolved_gate.minimum_mean_error_wins
        ),
        "maximum_case_regression": (
            aggregate["mean_error_mm"]["worst_case_relative_improvement"]
            >= -resolved_gate.maximum_case_mean_error_regression
        ),
    }
    return {
        "schema_version": 1,
        "artifact_kind": "PhysTwinStaticSceneGaugeSourceAggregateV1",
        "case_count": len(rows),
        "expected_cases": list(expected),
        "gate": asdict(resolved_gate),
        "aggregate": aggregate,
        "gate_checks": gate_checks,
        "gate_passed": bool(all(gate_checks.values())),
        "cases": rows,
        "claim_boundary": (
            "Opened-source automatic-observation competence only; this is "
            "neither independent confirmation nor a state-of-the-art claim."
        ),
    }


__all__ = [
    "StaticSceneGaugeSourceGate",
    "aggregate_phystwin_static_scene_gauge_source",
]
