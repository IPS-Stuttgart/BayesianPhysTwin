"""Independent validation of portable Deform360 Prob4D source-gate decisions.

The source-gate result is content-addressed, but content addressing alone does not
prove that stored decision booleans agree with the stored scientific evidence.
This module reconstructs every frozen gate check that can be derived from the
portable result and copied gate lock, and rejects internally inconsistent
artifacts before they can authorize confirmation access.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

_FACTOR_NAMES = frozenset(
    {
        "point_parallel",
        "point_lateral",
        "gauge_scale",
        "gauge_rotation",
        "gauge_translation",
    }
)
_METRIC_NAMES = frozenset({"coverage_90", "nees_per_dof"})
_AGGREGATE_SECTIONS = (
    "point_before",
    "point_after",
    "gauge_before",
    "gauge_after",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, *, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return cast(Sequence[Any], value)


def _literal_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty literal string")
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _real(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    _require(math.isfinite(result), f"{name} must be finite")
    if minimum is not None:
        _require(result >= minimum, f"{name} must be >= {minimum}")
    if maximum is not None:
        _require(result <= maximum, f"{name} must be <= {maximum}")
    return result


def _ordered_mean(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    _require(bool(ordered), "mean requires observations")
    _require(
        all(math.isfinite(value) for value in ordered), "mean values must be finite"
    )
    return math.fsum(ordered) / len(ordered)


def _same_real(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-15)


def _metric(value: object, *, name: str) -> dict[str, float]:
    metric = _mapping(value, name=name)
    _require(set(metric) == _METRIC_NAMES, f"{name} fields changed")
    return {
        "coverage_90": _real(
            metric["coverage_90"], name=f"{name}.coverage_90", minimum=0.0, maximum=1.0
        ),
        "nees_per_dof": _real(
            metric["nees_per_dof"], name=f"{name}.nees_per_dof", minimum=0.0
        ),
    }


def _factors(value: object, *, name: str) -> dict[str, float]:
    factors = _mapping(value, name=name)
    _require(set(factors) == _FACTOR_NAMES, f"{name} fields changed")
    return {
        key: _real(factors[key], name=f"{name}.{key}", minimum=0.0)
        for key in sorted(_FACTOR_NAMES)
    }


def _factor_ratio(left: float, right: float) -> float:
    _require(left > 0.0 and right > 0.0, "calibration factor must be positive")
    return max(left / right, right / left)


def _require_same_metric(
    stored: Mapping[str, float], expected: Mapping[str, float], *, name: str
) -> None:
    for field in _METRIC_NAMES:
        _require(
            _same_real(float(stored[field]), float(expected[field])),
            f"{name}.{field} differs from stored fold evidence",
        )


def validate_source_gate_decision_evidence(
    result: Mapping[str, Any], gate_lock: Mapping[str, Any]
) -> None:
    """Reconstruct portable gate checks from stored evidence and the frozen lock.

    This is intentionally independent of the source-gate evaluator.  It does not
    need access to target data or confirmation payloads; it validates only the
    decision evidence already present in the portable result.
    """

    cohort = _mapping(gate_lock.get("cohort"), name="source gate cohort")
    thresholds = _mapping(gate_lock.get("thresholds"), name="source gate thresholds")
    expected_object_count = _integer(
        cohort.get("exact_object_count"), name="exact_object_count", minimum=2
    )
    expected_strata_raw = _mapping(
        cohort.get("exact_stratum_counts"), name="exact_stratum_counts"
    )
    expected_stratum_counts = {
        _literal_string(stratum, name="expected stratum"): _integer(
            count, name=f"exact_stratum_counts.{stratum}", minimum=1
        )
        for stratum, count in expected_strata_raw.items()
    }

    minimum_streams = _integer(
        cohort.get("minimum_metric_streams_per_object"),
        name="minimum_metric_streams_per_object",
        minimum=1,
    )
    minimum_points = _integer(
        cohort.get("minimum_point_clusters_per_object"),
        name="minimum_point_clusters_per_object",
        minimum=1,
    )
    minimum_gauges = _integer(
        cohort.get("minimum_gauge_rows_per_object"),
        name="minimum_gauge_rows_per_object",
        minimum=1,
    )
    minimum_passing_folds = _integer(
        thresholds.get("minimum_passing_folds"), name="minimum_passing_folds", minimum=1
    )
    minimum_point_fold = _real(
        thresholds.get("minimum_point_coverage_90_per_passing_fold"),
        name="minimum_point_coverage_90_per_passing_fold",
        minimum=0.0,
        maximum=1.0,
    )
    minimum_gauge_fold = _real(
        thresholds.get("minimum_gauge_coverage_90_per_passing_fold"),
        name="minimum_gauge_coverage_90_per_passing_fold",
        minimum=0.0,
        maximum=1.0,
    )
    maximum_factor_ratio = _real(
        thresholds.get("maximum_factor_ratio_to_full_fit"),
        name="maximum_factor_ratio_to_full_fit",
        minimum=1.0,
    )
    target_coverage = _real(
        thresholds.get("target_coverage"),
        name="target_coverage",
        minimum=0.0,
        maximum=1.0,
    )
    worsening_tolerance = _real(
        thresholds.get("calibrated_coverage_error_worsening_tolerance"),
        name="calibrated_coverage_error_worsening_tolerance",
        minimum=0.0,
        maximum=1.0,
    )

    object_count = _integer(
        result.get("physical_object_count"), name="physical_object_count", minimum=2
    )
    full_factors = _factors(result.get("full_fit_factors"), name="full_fit_factors")
    support_rows = _sequence(result.get("support"), name="support")
    fold_rows = _sequence(result.get("folds"), name="folds")
    _require(
        len(support_rows) == object_count and len(fold_rows) == object_count,
        "source gate object evidence count changed",
    )

    support: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(support_rows):
        row = _mapping(raw, name=f"support {index}")
        _require(
            set(row)
            == {
                "object_id",
                "stratum",
                "metric_stream_count",
                "point_cluster_count",
                "gauge_row_count",
            },
            "source gate support fields changed",
        )
        object_id = _literal_string(row.get("object_id"), name="support object_id")
        _require(object_id not in support, "source gate support repeats an object")
        support[object_id] = {
            "stratum": _literal_string(row.get("stratum"), name="support stratum"),
            "metric_stream_count": _integer(
                row.get("metric_stream_count"), name="metric_stream_count"
            ),
            "point_cluster_count": _integer(
                row.get("point_cluster_count"), name="point_cluster_count"
            ),
            "gauge_row_count": _integer(
                row.get("gauge_row_count"), name="gauge_row_count"
            ),
        }

    folds: list[dict[str, Any]] = []
    seen_fold_objects: set[str] = set()
    for index, raw in enumerate(fold_rows):
        row = _mapping(raw, name=f"fold {index}")
        _require(
            set(row)
            == {
                "object_id",
                "stratum",
                "training_object_count",
                "factors",
                "maximum_factor_ratio_to_full_fit",
                "point_before",
                "point_after",
                "gauge_before",
                "gauge_after",
                "fold_passed",
            },
            "source gate fold fields changed",
        )
        object_id = _literal_string(row.get("object_id"), name="fold object_id")
        _require(
            object_id not in seen_fold_objects, "source gate folds repeat an object"
        )
        seen_fold_objects.add(object_id)
        stratum = _literal_string(row.get("stratum"), name="fold stratum")
        _require(object_id in support, "source gate fold has no matching support row")
        _require(
            support[object_id]["stratum"] == stratum,
            "source gate support and fold strata differ",
        )
        _require(
            _integer(row.get("training_object_count"), name="training_object_count")
            == object_count - 1,
            "source gate fold training-object count changed",
        )
        factors = _factors(row.get("factors"), name=f"fold {index} factors")
        recomputed_factor_ratio = max(
            _factor_ratio(factors[name], full_factors[name]) for name in _FACTOR_NAMES
        )
        stored_factor_ratio = _real(
            row.get("maximum_factor_ratio_to_full_fit"),
            name="maximum_factor_ratio_to_full_fit",
            minimum=1.0,
        )
        _require(
            _same_real(stored_factor_ratio, recomputed_factor_ratio),
            "source gate fold factor ratio differs from stored factors",
        )
        point_before = _metric(
            row.get("point_before"), name=f"fold {index} point_before"
        )
        point_after = _metric(row.get("point_after"), name=f"fold {index} point_after")
        gauge_before = _metric(
            row.get("gauge_before"), name=f"fold {index} gauge_before"
        )
        gauge_after = _metric(row.get("gauge_after"), name=f"fold {index} gauge_after")
        expected_fold_passed = (
            point_after["coverage_90"] >= minimum_point_fold
            and gauge_after["coverage_90"] >= minimum_gauge_fold
            and stored_factor_ratio <= maximum_factor_ratio
        )
        _require(
            _boolean(row.get("fold_passed"), name="fold_passed")
            is expected_fold_passed,
            "source gate fold decision differs from stored evidence",
        )
        folds.append(
            {
                "object_id": object_id,
                "stratum": stratum,
                "point_before": point_before,
                "point_after": point_after,
                "gauge_before": gauge_before,
                "gauge_after": gauge_after,
                "fold_passed": expected_fold_passed,
            }
        )

    _require(
        set(support) == seen_fold_objects, "source gate support and fold rosters differ"
    )
    observed_stratum_counts: dict[str, int] = {}
    for fold in folds:
        stratum = cast(str, fold["stratum"])
        observed_stratum_counts[stratum] = observed_stratum_counts.get(stratum, 0) + 1
    stored_strata = _mapping(result.get("stratum_counts"), name="stratum_counts")
    normalized_stored_strata = {
        _literal_string(stratum, name="stored stratum"): _integer(
            count, name=f"stratum_counts.{stratum}"
        )
        for stratum, count in stored_strata.items()
    }
    _require(
        normalized_stored_strata == observed_stratum_counts,
        "source gate stratum counts differ from fold evidence",
    )

    def balanced(section: str, metric: str, selected: Sequence[int]) -> float:
        return _ordered_mean(
            [
                cast(dict[str, float], folds[index][section])[metric]
                for index in selected
            ]
        )

    all_indices = tuple(range(object_count))
    expected_aggregate: dict[str, Any] = {
        section: {
            "coverage_90": balanced(section, "coverage_90", all_indices),
            "nees_per_dof": balanced(section, "nees_per_dof", all_indices),
        }
        for section in _AGGREGATE_SECTIONS
    }
    expected_aggregate["strata"] = {}
    for stratum in sorted(observed_stratum_counts):
        selected = tuple(
            index for index, fold in enumerate(folds) if fold["stratum"] == stratum
        )
        expected_aggregate["strata"][stratum] = {
            "object_count": len(selected),
            "point_coverage_90": balanced("point_after", "coverage_90", selected),
            "gauge_coverage_90": balanced("gauge_after", "coverage_90", selected),
        }

    aggregate = _mapping(result.get("aggregate"), name="aggregate")
    _require(
        set(aggregate) == {*_AGGREGATE_SECTIONS, "strata"},
        "source gate aggregate fields changed",
    )
    normalized_aggregate: dict[str, Any] = {}
    for section in _AGGREGATE_SECTIONS:
        normalized_aggregate[section] = _metric(
            aggregate.get(section), name=f"aggregate.{section}"
        )
        _require_same_metric(
            normalized_aggregate[section],
            expected_aggregate[section],
            name=f"aggregate.{section}",
        )
    aggregate_strata = _mapping(aggregate.get("strata"), name="aggregate.strata")
    _require(
        set(aggregate_strata) == set(observed_stratum_counts),
        "source gate aggregate stratum roster changed",
    )
    normalized_aggregate["strata"] = {}
    for stratum, expected in cast(
        dict[str, dict[str, float]], expected_aggregate["strata"]
    ).items():
        row = _mapping(aggregate_strata[stratum], name=f"aggregate.strata.{stratum}")
        _require(
            set(row) == {"object_count", "point_coverage_90", "gauge_coverage_90"},
            "source gate aggregate stratum fields changed",
        )
        object_total = _integer(
            row.get("object_count"), name="aggregate stratum object_count"
        )
        point_coverage = _real(
            row.get("point_coverage_90"),
            name="aggregate stratum point_coverage_90",
            minimum=0.0,
            maximum=1.0,
        )
        gauge_coverage = _real(
            row.get("gauge_coverage_90"),
            name="aggregate stratum gauge_coverage_90",
            minimum=0.0,
            maximum=1.0,
        )
        _require(
            object_total == int(expected["object_count"])
            and _same_real(point_coverage, float(expected["point_coverage_90"]))
            and _same_real(gauge_coverage, float(expected["gauge_coverage_90"])),
            "source gate aggregate stratum differs from fold evidence",
        )
        normalized_aggregate["strata"][stratum] = {
            "object_count": object_total,
            "point_coverage_90": point_coverage,
            "gauge_coverage_90": gauge_coverage,
        }

    checks: dict[str, bool] = {
        "exact_object_count": object_count == expected_object_count,
        "exact_stratum_counts": observed_stratum_counts == expected_stratum_counts,
        "metric_stream_support": all(
            row["metric_stream_count"] >= minimum_streams for row in support.values()
        ),
        "point_cluster_support": all(
            row["point_cluster_count"] >= minimum_points for row in support.values()
        ),
        "gauge_row_support": all(
            row["gauge_row_count"] >= minimum_gauges for row in support.values()
        ),
        "minimum_passing_folds": sum(bool(fold["fold_passed"]) for fold in folds)
        >= minimum_passing_folds,
        "aggregate_point_coverage": _real(
            thresholds.get("aggregate_point_coverage_90_minimum"),
            name="aggregate_point_coverage_90_minimum",
            minimum=0.0,
            maximum=1.0,
        )
        <= normalized_aggregate["point_after"]["coverage_90"]
        <= _real(
            thresholds.get("aggregate_point_coverage_90_maximum"),
            name="aggregate_point_coverage_90_maximum",
            minimum=0.0,
            maximum=1.0,
        ),
        "aggregate_gauge_coverage": normalized_aggregate["gauge_after"]["coverage_90"]
        >= _real(
            thresholds.get("aggregate_gauge_coverage_90_minimum"),
            name="aggregate_gauge_coverage_90_minimum",
            minimum=0.0,
            maximum=1.0,
        ),
        "aggregate_point_nees": _real(
            thresholds.get("aggregate_point_nees_per_dof_minimum"),
            name="aggregate_point_nees_per_dof_minimum",
            minimum=0.0,
        )
        <= normalized_aggregate["point_after"]["nees_per_dof"]
        <= _real(
            thresholds.get("aggregate_point_nees_per_dof_maximum"),
            name="aggregate_point_nees_per_dof_maximum",
            minimum=0.0,
        ),
        "aggregate_gauge_nees": _real(
            thresholds.get("aggregate_gauge_nees_per_dof_minimum"),
            name="aggregate_gauge_nees_per_dof_minimum",
            minimum=0.0,
        )
        <= normalized_aggregate["gauge_after"]["nees_per_dof"]
        <= _real(
            thresholds.get("aggregate_gauge_nees_per_dof_maximum"),
            name="aggregate_gauge_nees_per_dof_maximum",
            minimum=0.0,
        ),
        "point_calibration_nonworsening": abs(
            normalized_aggregate["point_after"]["coverage_90"] - target_coverage
        )
        <= abs(normalized_aggregate["point_before"]["coverage_90"] - target_coverage)
        + worsening_tolerance,
        "gauge_calibration_nonworsening": abs(
            normalized_aggregate["gauge_after"]["coverage_90"] - target_coverage
        )
        <= abs(normalized_aggregate["gauge_before"]["coverage_90"] - target_coverage)
        + worsening_tolerance,
    }
    for stratum in sorted(observed_stratum_counts):
        row = normalized_aggregate["strata"][stratum]
        checks[f"{stratum}_point_transfer"] = (
            row["point_coverage_90"] >= minimum_point_fold
        )
        checks[f"{stratum}_gauge_transfer"] = (
            row["gauge_coverage_90"] >= minimum_gauge_fold
        )

    stored_checks = _mapping(result.get("checks"), name="checks")
    _require(
        set(stored_checks) == set(checks)
        and all(type(value) is bool for value in stored_checks.values())
        and dict(stored_checks) == checks,
        "source gate checks differ from stored decision evidence",
    )
    expected_passed = all(checks.values())
    _require(
        _integer(result.get("passed_check_count"), name="passed_check_count")
        == sum(checks.values())
        and _integer(result.get("total_check_count"), name="total_check_count")
        == len(checks),
        "source gate check counts differ from stored decision evidence",
    )
    _require(
        _boolean(result.get("gate_passed"), name="gate_passed") is expected_passed,
        "source gate decision differs from stored decision evidence",
    )
    _require(
        _boolean(
            result.get("confirmation_access_authorized"),
            name="confirmation_access_authorized",
        )
        is expected_passed,
        "source gate authorization differs from stored decision evidence",
    )
    expected_status = "source-gate-passed" if expected_passed else "source-gate-failed"
    _require(
        result.get("status") == expected_status,
        "source gate status differs from stored decision evidence",
    )


__all__ = ["validate_source_gate_decision_evidence"]
