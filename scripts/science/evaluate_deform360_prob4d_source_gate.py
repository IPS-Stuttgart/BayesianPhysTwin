#!/usr/bin/env python3
"""Evaluate the frozen public Deform360 Prob4D source-calibration gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

import numpy as np

from bayesian_phystwin._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    genuine_integer,
    plain_json,
)
from bayesian_phystwin._portable_contracts import (
    content_id,
    exact_revision,
    nonempty_string,
    require_exact_fields,
    sha256_digest,
)
from bayesian_phystwin.deform360_prob4d_source_calibration import (
    RESULT_SCHEMA as SOURCE_CALIBRATION_RESULT_SCHEMA,
)
from bayesian_phystwin.deform360_prob4d_source_calibration import (
    RESULT_SEMANTICS as SOURCE_CALIBRATION_RESULT_SEMANTICS,
)
from bayesian_phystwin.deform360_prob4d_source_calibration import (
    RESULT_VERSION as SOURCE_CALIBRATION_RESULT_VERSION,
)
from bayesian_phystwin.deform360_prob4d_source_calibration import (
    Deform360Prob4DCalibrationSamplesV2,
    _regularized_inverse_psd,
    _upper_winsorized_mean,
    collapse_point_correlation_clusters,
    load_deform360_prob4d_calibration_samples,
)
from bayesian_phystwin.deform360_public_contact_prefix import (
    _ordinary_directory,
    _ordinary_file,
)

SOURCE_GATE_SCHEMA: Final = "bayesian-phystwin.deform360-prob4d-source-gate-result"
SOURCE_GATE_VERSION: Final = 1
SOURCE_GATE_SEMANTICS: Final = (
    "object-balanced-leave-one-physical-object-out-calibration-gate-v1"
)
SOURCE_GATE_RESULT_FILENAME: Final = "source-gate-result.json"
SOURCE_GATE_LOCK_FILENAME: Final = "source-gate-lock.json"
SOURCE_GATE_CLAIM_BOUNDARY: Final = (
    "A passing result authorizes only independent evaluation on the locked public "
    "confirmation objects. It does not establish confirmation performance, "
    "state-estimation benefit, physical-query benefit, safety, official benchmark "
    "parity, or state of the art."
)

_LOCK_SCHEMA = "bayesian-phystwin.deform360-prob4d-source-gate-v1"
_LOCK_FIELDS = frozenset(
    {
        "artifact_id",
        "schema",
        "schema_version",
        "semantics",
        "protocol_id",
        "cohort",
        "thresholds",
        "information_boundary",
        "claim_boundary",
    }
)
_COHORT_FIELDS = frozenset(
    {
        "exact_object_count",
        "exact_stratum_counts",
        "minimum_metric_streams_per_object",
        "minimum_point_clusters_per_object",
        "minimum_gauge_rows_per_object",
    }
)
_THRESHOLD_FIELDS = frozenset(
    {
        "target_coverage",
        "trim_quantile",
        "point_chi_square_90",
        "gauge_chi_square_90",
        "point_degrees_of_freedom",
        "gauge_degrees_of_freedom",
        "aggregate_point_coverage_90_minimum",
        "aggregate_point_coverage_90_maximum",
        "aggregate_gauge_coverage_90_minimum",
        "aggregate_point_nees_per_dof_minimum",
        "aggregate_point_nees_per_dof_maximum",
        "aggregate_gauge_nees_per_dof_minimum",
        "aggregate_gauge_nees_per_dof_maximum",
        "minimum_point_coverage_90_per_passing_fold",
        "minimum_gauge_coverage_90_per_passing_fold",
        "maximum_factor_ratio_to_full_fit",
        "minimum_passing_folds",
        "calibrated_coverage_error_worsening_tolerance",
    }
)
_LOCK_BOUNDARY = {
    "confirmation_payloads_opened": False,
    "future_frames_used": False,
    "human_approval_required": False,
    "new_measurements_required": False,
    "replacement_allowed": False,
    "target_outcomes_used": False,
}
_SOURCE_RESULT_FIELDS = frozenset(
    {
        "result_id",
        "schema",
        "schema_version",
        "semantics",
        "protocol_id",
        "calibration_sample_bundle_id",
        "calibration_sample_manifest_sha256",
        "visual_production_result_id",
        "prob4d_revision",
        "motioncrafter_revision",
        "physical_object_count",
        "stratum_counts",
        "point_effective_cluster_count",
        "gauge_raw_row_count",
        "artifacts",
        "reports",
        "information_boundary",
        "claim_boundary",
    }
)
_SOURCE_BOUNDARY_FIELDS = frozenset(
    {
        "calibration_payloads_opened",
        "confirmation_payloads_opened",
        "target_outcomes_used",
        "future_frames_used",
        "replacement_allowed",
        "confirmation_access_authorized",
        "calibration_gate_evaluated",
    }
)
_RESULT_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
        "result_id",
        "implementation_revision",
        "protocol_id",
        "gate_lock_id",
        "gate_lock_file_sha256",
        "calibration_sample_bundle_id",
        "calibration_sample_manifest_sha256",
        "source_calibration_result_id",
        "source_calibration_result_file_sha256",
        "physical_object_count",
        "stratum_counts",
        "support",
        "full_fit_factors",
        "folds",
        "aggregate",
        "checks",
        "passed_check_count",
        "total_check_count",
        "gate_passed",
        "confirmation_access_authorized",
        "status",
        "information_boundary",
        "claim_boundary",
    }
)


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: str | Path, *, name: str) -> dict[str, Any]:
    source = _ordinary_file(path, name=name)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return cast(dict[str, Any], value)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            plain_json(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _finite_real(value: object, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    _require(np.isfinite(result), f"{name} must be finite")
    if minimum is not None:
        _require(result >= minimum, f"{name} must be >= {minimum}")
    return result


def load_source_gate_lock(path: str | Path) -> dict[str, Any]:
    """Load and content-verify the frozen source-gate thresholds."""

    lock = _load_json(path, name="source gate lock")
    require_exact_fields(lock, expected=_LOCK_FIELDS, name="source gate lock")
    _require(
        lock["schema"] == _LOCK_SCHEMA
        and lock["schema_version"] == 1
        and lock["semantics"] == SOURCE_GATE_SEMANTICS,
        "source gate lock contract changed",
    )
    identity = dict(lock)
    declared_id = sha256_digest(identity.pop("artifact_id"), name="gate lock ID")
    _require(content_id(identity) == declared_id, "source gate lock ID changed")
    nonempty_string(lock["protocol_id"], name="protocol_id")
    cohort = _mapping(lock["cohort"], name="cohort")
    require_exact_fields(cohort, expected=_COHORT_FIELDS, name="cohort")
    genuine_integer(cohort["exact_object_count"], name="exact_object_count", minimum=2)
    stratum_counts = _mapping(cohort["exact_stratum_counts"], name="stratum counts")
    _require(len(stratum_counts) >= 2, "source gate requires at least two strata")
    for stratum, count in stratum_counts.items():
        nonempty_string(stratum, name="stratum")
        genuine_integer(count, name=f"stratum_counts.{stratum}", minimum=1)
    for field in (
        "minimum_metric_streams_per_object",
        "minimum_point_clusters_per_object",
        "minimum_gauge_rows_per_object",
    ):
        genuine_integer(cohort[field], name=field, minimum=1)
    thresholds = _mapping(lock["thresholds"], name="thresholds")
    require_exact_fields(thresholds, expected=_THRESHOLD_FIELDS, name="thresholds")
    for field, value in thresholds.items():
        if field in {
            "point_degrees_of_freedom",
            "gauge_degrees_of_freedom",
            "minimum_passing_folds",
        }:
            genuine_integer(value, name=field, minimum=1)
        else:
            _finite_real(value, name=field, minimum=0.0)
    expected_objects = genuine_integer(
        cohort["exact_object_count"], name="exact_object_count", minimum=2
    )
    _require(
        sum(int(count) for count in stratum_counts.values()) == expected_objects,
        "stratum counts do not sum to the exact object count",
    )
    for field in (
        "target_coverage",
        "aggregate_point_coverage_90_minimum",
        "aggregate_point_coverage_90_maximum",
        "aggregate_gauge_coverage_90_minimum",
        "minimum_point_coverage_90_per_passing_fold",
        "minimum_gauge_coverage_90_per_passing_fold",
        "calibrated_coverage_error_worsening_tolerance",
    ):
        _require(float(thresholds[field]) <= 1.0, f"{field} must be <= 1")
    _require(
        float(thresholds["aggregate_point_coverage_90_minimum"])
        <= float(thresholds["aggregate_point_coverage_90_maximum"]),
        "point coverage bounds are reversed",
    )
    _require(
        int(thresholds["minimum_passing_folds"]) <= expected_objects,
        "minimum passing folds exceeds the cohort",
    )
    _require(lock["information_boundary"] == _LOCK_BOUNDARY, "gate boundary changed")
    nonempty_string(lock["claim_boundary"], name="claim_boundary")
    return cast(dict[str, Any], plain_json(lock))


def _ordered_mean(values: Sequence[float] | np.ndarray) -> float:
    array = np.sort(np.asarray(values, dtype=np.float64))
    _require(array.ndim == 1 and len(array) > 0, "mean requires observations")
    _require(np.all(np.isfinite(array)), "mean observations are not finite")
    return math.fsum(map(float, array)) / len(array)


def _object_balanced_factor(
    ratios: np.ndarray,
    case_index: np.ndarray,
    train_cases: Sequence[int],
    *,
    trim_quantile: float,
) -> float:
    values = []
    for case in train_cases:
        selected = np.asarray(ratios[case_index == case], dtype=np.float64)
        _require(len(selected) > 0, "training object has no calibration ratio")
        values.append(_upper_winsorized_mean(selected, quantile=trim_quantile))
    return _ordered_mean(values)


def _point_ratios(
    samples: Deform360Prob4DCalibrationSamplesV2 | Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = samples.arrays
    collapsed = collapse_point_correlation_clusters(
        errors=arrays["point_errors_m"],
        ray_directions=arrays["point_ray_directions"],
        parallel_variance=arrays["point_parallel_variance_m2"],
        lateral_variance=arrays["point_lateral_variance_m2"],
        case_index=arrays["point_case_index"],
        cluster_index=arrays["point_correlation_cluster_index"],
        valid=arrays["point_valid"],
    )
    errors, rays, parallel_variance, lateral_variance, case_index, _report = collapsed
    rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
    parallel_error = np.sum(errors * rays, axis=1)
    total_squared = np.sum(errors**2, axis=1)
    lateral_squared = np.maximum(total_squared - parallel_error**2, 0.0)
    return (
        parallel_error**2 / parallel_variance,
        lateral_squared / (2.0 * lateral_variance),
        np.asarray(case_index, dtype=np.int64),
    )


def _gauge_ratios(
    samples: Deform360Prob4DCalibrationSamplesV2 | Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    errors = np.asarray(samples.arrays["gauge_errors"], dtype=np.float64)
    covariance = np.asarray(samples.arrays["gauge_covariance"], dtype=np.float64)
    case_index = np.asarray(samples.arrays["gauge_case_index"], dtype=np.int64)
    scale: np.ndarray = np.empty(len(errors), dtype=np.float64)
    rotation: np.ndarray = np.empty(len(errors), dtype=np.float64)
    translation: np.ndarray = np.empty(len(errors), dtype=np.float64)
    for row, (error, matrix) in enumerate(zip(errors, covariance, strict=True)):
        scale[row] = error[0] ** 2 / max(matrix[0, 0], 1e-12)
        for destination, block in (
            (rotation, slice(1, 4)),
            (translation, slice(4, 7)),
        ):
            block_error = error[block]
            information = _regularized_inverse_psd(matrix[block, block])
            destination[row] = float(block_error @ information @ block_error / 3.0)
    return scale, rotation, translation, case_index


def _coverage_metrics(
    nees: np.ndarray, *, threshold: float, dof: int
) -> dict[str, float]:
    values = np.asarray(nees, dtype=np.float64)
    _require(len(values) > 0 and np.all(np.isfinite(values)), "invalid NEES values")
    return {
        "coverage_90": float(np.mean(values <= threshold)),
        "nees_per_dof": _ordered_mean(values) / dof,
    }


def _gauge_nees(
    samples: Deform360Prob4DCalibrationSamplesV2 | Any,
    selected: np.ndarray,
    *,
    scale_factor: float,
    rotation_factor: float,
    translation_factor: float,
) -> np.ndarray:
    errors = np.asarray(samples.arrays["gauge_errors"], dtype=np.float64)[selected]
    covariance = np.asarray(samples.arrays["gauge_covariance"], dtype=np.float64)[
        selected
    ]
    scaling = np.sqrt(
        np.asarray(
            [
                scale_factor,
                rotation_factor,
                rotation_factor,
                rotation_factor,
                translation_factor,
                translation_factor,
                translation_factor,
            ],
            dtype=np.float64,
        )
    )
    values: np.ndarray = np.empty(len(errors), dtype=np.float64)
    for row, (error, matrix) in enumerate(zip(errors, covariance, strict=True)):
        calibrated = matrix * np.outer(scaling, scaling)
        values[row] = float(error @ _regularized_inverse_psd(calibrated) @ error)
    return values


def _factor_ratio(left: float, right: float) -> float:
    _require(left > 0.0 and right > 0.0, "calibration factor must be positive")
    return max(left / right, right / left)


def evaluate_source_gate(
    samples: Deform360Prob4DCalibrationSamplesV2 | Any,
    gate_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the frozen object-balanced LOO gate without target access."""

    cohort = _mapping(gate_lock["cohort"], name="cohort")
    thresholds = _mapping(gate_lock["thresholds"], name="thresholds")
    object_ids = tuple(samples.object_ids)
    cases = tuple(samples.cases)
    object_count = len(object_ids)
    _require(len(cases) == object_count, "sample object identities changed")
    trim_quantile = float(thresholds["trim_quantile"])
    point_parallel, point_lateral, point_case = _point_ratios(samples)
    gauge_scale, gauge_rotation, gauge_translation, gauge_case = _gauge_ratios(samples)
    all_cases = tuple(range(object_count))

    full_factors = {
        "point_parallel": _object_balanced_factor(
            point_parallel, point_case, all_cases, trim_quantile=trim_quantile
        ),
        "point_lateral": _object_balanced_factor(
            point_lateral, point_case, all_cases, trim_quantile=trim_quantile
        ),
        "gauge_scale": _object_balanced_factor(
            gauge_scale, gauge_case, all_cases, trim_quantile=trim_quantile
        ),
        "gauge_rotation": _object_balanced_factor(
            gauge_rotation, gauge_case, all_cases, trim_quantile=trim_quantile
        ),
        "gauge_translation": _object_balanced_factor(
            gauge_translation, gauge_case, all_cases, trim_quantile=trim_quantile
        ),
    }
    point_threshold = float(thresholds["point_chi_square_90"])
    gauge_threshold = float(thresholds["gauge_chi_square_90"])
    point_dof = int(thresholds["point_degrees_of_freedom"])
    gauge_dof = int(thresholds["gauge_degrees_of_freedom"])
    max_factor_ratio = float(thresholds["maximum_factor_ratio_to_full_fit"])
    minimum_point_fold = float(thresholds["minimum_point_coverage_90_per_passing_fold"])
    minimum_gauge_fold = float(thresholds["minimum_gauge_coverage_90_per_passing_fold"])

    folds: list[dict[str, Any]] = []
    support: list[dict[str, Any]] = []
    for held_out, (object_id, case) in enumerate(zip(object_ids, cases, strict=True)):
        train_cases = tuple(case_id for case_id in all_cases if case_id != held_out)
        factors = {
            "point_parallel": _object_balanced_factor(
                point_parallel,
                point_case,
                train_cases,
                trim_quantile=trim_quantile,
            ),
            "point_lateral": _object_balanced_factor(
                point_lateral,
                point_case,
                train_cases,
                trim_quantile=trim_quantile,
            ),
            "gauge_scale": _object_balanced_factor(
                gauge_scale, gauge_case, train_cases, trim_quantile=trim_quantile
            ),
            "gauge_rotation": _object_balanced_factor(
                gauge_rotation, gauge_case, train_cases, trim_quantile=trim_quantile
            ),
            "gauge_translation": _object_balanced_factor(
                gauge_translation,
                gauge_case,
                train_cases,
                trim_quantile=trim_quantile,
            ),
        }
        point_rows = point_case == held_out
        gauge_rows = gauge_case == held_out
        point_nees_before = point_parallel[point_rows] + 2.0 * point_lateral[point_rows]
        point_nees_after = (
            point_parallel[point_rows] / factors["point_parallel"]
            + 2.0 * point_lateral[point_rows] / factors["point_lateral"]
        )
        gauge_nees_before = _gauge_nees(
            samples,
            gauge_rows,
            scale_factor=1.0,
            rotation_factor=1.0,
            translation_factor=1.0,
        )
        gauge_nees_after = _gauge_nees(
            samples,
            gauge_rows,
            scale_factor=factors["gauge_scale"],
            rotation_factor=factors["gauge_rotation"],
            translation_factor=factors["gauge_translation"],
        )
        ratios = {
            name: _factor_ratio(value, full_factors[name])
            for name, value in factors.items()
        }
        point_after = _coverage_metrics(
            point_nees_after, threshold=point_threshold, dof=point_dof
        )
        gauge_after = _coverage_metrics(
            gauge_nees_after, threshold=gauge_threshold, dof=gauge_dof
        )
        fold_passed = (
            point_after["coverage_90"] >= minimum_point_fold
            and gauge_after["coverage_90"] >= minimum_gauge_fold
            and max(ratios.values()) <= max_factor_ratio
        )
        stratum = nonempty_string(case["stratum"], name="case stratum")
        stream_count = len(cast(Sequence[object], case["metric_references"]))
        support_row = {
            "object_id": object_id,
            "stratum": stratum,
            "metric_stream_count": stream_count,
            "point_cluster_count": int(np.sum(point_rows)),
            "gauge_row_count": int(np.sum(gauge_rows)),
        }
        support.append(support_row)
        folds.append(
            {
                "object_id": object_id,
                "stratum": stratum,
                "training_object_count": len(train_cases),
                "factors": factors,
                "maximum_factor_ratio_to_full_fit": max(ratios.values()),
                "point_before": _coverage_metrics(
                    point_nees_before, threshold=point_threshold, dof=point_dof
                ),
                "point_after": point_after,
                "gauge_before": _coverage_metrics(
                    gauge_nees_before, threshold=gauge_threshold, dof=gauge_dof
                ),
                "gauge_after": gauge_after,
                "fold_passed": fold_passed,
            }
        )

    strata = sorted({cast(str, case["stratum"]) for case in cases})
    stratum_counts = {
        stratum: sum(case["stratum"] == stratum for case in cases) for stratum in strata
    }

    def balanced(section: str, metric: str, selected: Sequence[int]) -> float:
        return _ordered_mean(
            [float(folds[index][section][metric]) for index in selected]
        )

    aggregate: dict[str, Any] = {
        "point_before": {
            "coverage_90": balanced("point_before", "coverage_90", all_cases),
            "nees_per_dof": balanced("point_before", "nees_per_dof", all_cases),
        },
        "point_after": {
            "coverage_90": balanced("point_after", "coverage_90", all_cases),
            "nees_per_dof": balanced("point_after", "nees_per_dof", all_cases),
        },
        "gauge_before": {
            "coverage_90": balanced("gauge_before", "coverage_90", all_cases),
            "nees_per_dof": balanced("gauge_before", "nees_per_dof", all_cases),
        },
        "gauge_after": {
            "coverage_90": balanced("gauge_after", "coverage_90", all_cases),
            "nees_per_dof": balanced("gauge_after", "nees_per_dof", all_cases),
        },
        "strata": {
            stratum: {
                "object_count": len(
                    selected := tuple(
                        index
                        for index, fold in enumerate(folds)
                        if fold["stratum"] == stratum
                    )
                ),
                "point_coverage_90": balanced("point_after", "coverage_90", selected),
                "gauge_coverage_90": balanced("gauge_after", "coverage_90", selected),
            }
            for stratum in strata
        },
    }
    target_coverage = float(thresholds["target_coverage"])
    tolerance = float(thresholds["calibrated_coverage_error_worsening_tolerance"])
    minimum_streams = int(cohort["minimum_metric_streams_per_object"])
    minimum_points = int(cohort["minimum_point_clusters_per_object"])
    minimum_gauges = int(cohort["minimum_gauge_rows_per_object"])
    checks: dict[str, bool] = {
        "exact_object_count": object_count == int(cohort["exact_object_count"]),
        "exact_stratum_counts": stratum_counts
        == dict(cast(Mapping[str, int], cohort["exact_stratum_counts"])),
        "metric_stream_support": all(
            row["metric_stream_count"] >= minimum_streams for row in support
        ),
        "point_cluster_support": all(
            row["point_cluster_count"] >= minimum_points for row in support
        ),
        "gauge_row_support": all(
            row["gauge_row_count"] >= minimum_gauges for row in support
        ),
        "minimum_passing_folds": sum(fold["fold_passed"] for fold in folds)
        >= int(thresholds["minimum_passing_folds"]),
        "aggregate_point_coverage": float(
            thresholds["aggregate_point_coverage_90_minimum"]
        )
        <= aggregate["point_after"]["coverage_90"]
        <= float(thresholds["aggregate_point_coverage_90_maximum"]),
        "aggregate_gauge_coverage": aggregate["gauge_after"]["coverage_90"]
        >= float(thresholds["aggregate_gauge_coverage_90_minimum"]),
        "aggregate_point_nees": float(
            thresholds["aggregate_point_nees_per_dof_minimum"]
        )
        <= aggregate["point_after"]["nees_per_dof"]
        <= float(thresholds["aggregate_point_nees_per_dof_maximum"]),
        "aggregate_gauge_nees": float(
            thresholds["aggregate_gauge_nees_per_dof_minimum"]
        )
        <= aggregate["gauge_after"]["nees_per_dof"]
        <= float(thresholds["aggregate_gauge_nees_per_dof_maximum"]),
        "point_calibration_nonworsening": abs(
            aggregate["point_after"]["coverage_90"] - target_coverage
        )
        <= abs(aggregate["point_before"]["coverage_90"] - target_coverage) + tolerance,
        "gauge_calibration_nonworsening": abs(
            aggregate["gauge_after"]["coverage_90"] - target_coverage
        )
        <= abs(aggregate["gauge_before"]["coverage_90"] - target_coverage) + tolerance,
    }
    for stratum in strata:
        checks[f"{stratum}_point_transfer"] = (
            aggregate["strata"][stratum]["point_coverage_90"] >= minimum_point_fold
        )
        checks[f"{stratum}_gauge_transfer"] = (
            aggregate["strata"][stratum]["gauge_coverage_90"] >= minimum_gauge_fold
        )
    gate_passed = all(checks.values())
    return cast(
        dict[str, Any],
        frozen_finite_json_mapping(
            {
                "physical_object_count": object_count,
                "stratum_counts": stratum_counts,
                "support": support,
                "full_fit_factors": full_factors,
                "folds": folds,
                "aggregate": aggregate,
                "checks": checks,
                "passed_check_count": sum(checks.values()),
                "total_check_count": len(checks),
                "gate_passed": gate_passed,
            },
            name="source gate evaluation",
        ),
    )


def _validate_source_calibration_result(
    path: Path,
    *,
    root: Path,
    samples: Deform360Prob4DCalibrationSamplesV2,
) -> dict[str, Any]:
    result = _load_json(path, name="source calibration result")
    require_exact_fields(
        result,
        expected=_SOURCE_RESULT_FIELDS,
        name="source calibration result",
    )
    _require(
        result["schema"] == SOURCE_CALIBRATION_RESULT_SCHEMA
        and result["schema_version"] == SOURCE_CALIBRATION_RESULT_VERSION
        and result["semantics"] == SOURCE_CALIBRATION_RESULT_SEMANTICS,
        "source calibration result contract changed",
    )
    identity = dict(result)
    declared_id = sha256_digest(identity.pop("result_id"), name="source result ID")
    _require(
        content_id(identity) == declared_id, "source calibration result ID changed"
    )
    _require(
        result["protocol_id"] == samples.protocol_id
        and result["calibration_sample_bundle_id"] == samples.bundle_id
        and result["calibration_sample_manifest_sha256"] == samples.manifest_file_sha256
        and result["physical_object_count"] == len(samples.object_ids),
        "source calibration result differs from samples",
    )
    boundary = _mapping(result["information_boundary"], name="source boundary")
    require_exact_fields(
        boundary,
        expected=_SOURCE_BOUNDARY_FIELDS,
        name="source boundary",
    )
    _require(
        boundary.get("confirmation_access_authorized") is False
        and boundary.get("calibration_gate_evaluated") is False
        and boundary.get("confirmation_payloads_opened") is False
        and boundary.get("target_outcomes_used") is False,
        "source calibration result crossed its information boundary",
    )
    checksums = _ordinary_file(root / "SHA256SUMS", name="source SHA256SUMS")
    expected = []
    for line in checksums.read_text(encoding="ascii").splitlines():
        digest, separator, relative = line.partition("  ")
        _require(bool(separator), "source checksum line changed")
        source = _ordinary_file(root / relative, name="source calibration artifact")
        _require(
            root == source or root in source.parents, "source artifact escapes root"
        )
        _require(_sha256_file(source) == digest, "source calibration checksum changed")
        expected.append(relative)
    actual = sorted(
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file() and item.name != "SHA256SUMS"
    )
    _require(sorted(expected) == actual, "source calibration artifact roster changed")
    return result


def publish_source_gate_result(
    *,
    samples: Deform360Prob4DCalibrationSamplesV2,
    source_calibration_result_path: str | Path,
    source_calibration_root: str | Path,
    gate_lock_path: str | Path,
    implementation_revision: str,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Evaluate and atomically publish one source-only decision artifact."""

    source_root = _ordinary_directory(
        source_calibration_root, name="source calibration root"
    )
    source_path = _ordinary_file(
        source_calibration_result_path, name="source calibration result"
    )
    _require(
        source_root == source_path.parent or source_root in source_path.parents,
        "source calibration result escapes its root",
    )
    source = _validate_source_calibration_result(
        source_path, root=source_root, samples=samples
    )
    lock_path = _ordinary_file(gate_lock_path, name="source gate lock")
    gate_lock = load_source_gate_lock(lock_path)
    _require(
        gate_lock["protocol_id"] == samples.protocol_id,
        "source gate protocol differs from samples",
    )
    evaluation = evaluate_source_gate(samples, gate_lock)
    passed = cast(bool, evaluation["gate_passed"])
    boundary = {
        "calibration_payloads_opened": True,
        "calibration_gate_evaluated": True,
        "confirmation_payloads_opened": False,
        "target_outcomes_used": False,
        "future_frames_used": False,
        "replacement_allowed": False,
        "human_approval_required": False,
        "new_measurements_required": False,
    }
    identity: dict[str, Any] = {
        "schema": SOURCE_GATE_SCHEMA,
        "schema_version": SOURCE_GATE_VERSION,
        "semantics": SOURCE_GATE_SEMANTICS,
        "implementation_revision": exact_revision(
            implementation_revision, name="implementation_revision"
        ),
        "protocol_id": samples.protocol_id,
        "gate_lock_id": gate_lock["artifact_id"],
        "gate_lock_file_sha256": _sha256_file(lock_path),
        "calibration_sample_bundle_id": samples.bundle_id,
        "calibration_sample_manifest_sha256": samples.manifest_file_sha256,
        "source_calibration_result_id": source["result_id"],
        "source_calibration_result_file_sha256": _sha256_file(source_path),
        **evaluation,
        "confirmation_access_authorized": passed,
        "status": "source-gate-passed" if passed else "source-gate-failed",
        "information_boundary": boundary,
        "claim_boundary": SOURCE_GATE_CLAIM_BOUNDARY,
    }
    result = {**identity, "result_id": content_id(identity)}
    output = Path(output_directory).absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    _ordinary_directory(output.parent, name="source gate output parent")
    _require(not os.path.lexists(output), "source gate output already exists")
    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.partial"
    temporary.mkdir(mode=0o700)
    try:
        _write_json(temporary / SOURCE_GATE_RESULT_FILENAME, result)
        shutil.copyfile(lock_path, temporary / SOURCE_GATE_LOCK_FILENAME)
        paths = sorted(path for path in temporary.rglob("*") if path.is_file())
        (temporary / "SHA256SUMS").write_text(
            "".join(
                f"{_sha256_file(path)}  {path.relative_to(temporary).as_posix()}\n"
                for path in paths
            ),
            encoding="ascii",
        )
        validate_source_gate_result(temporary)
        _require(not os.path.lexists(output), "source gate output already exists")
        os.rename(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return validate_source_gate_result(output)


def validate_source_gate_result(directory: str | Path) -> dict[str, Any]:
    """Strictly reload one portable source-gate decision."""

    root = _ordinary_directory(directory, name="source gate result")
    result = _load_json(root / SOURCE_GATE_RESULT_FILENAME, name="source gate result")
    require_exact_fields(result, expected=_RESULT_FIELDS, name="source gate result")
    _require(
        result["schema"] == SOURCE_GATE_SCHEMA
        and result["schema_version"] == SOURCE_GATE_VERSION
        and result["semantics"] == SOURCE_GATE_SEMANTICS,
        "source gate result contract changed",
    )
    identity = dict(result)
    declared_id = sha256_digest(identity.pop("result_id"), name="result_id")
    _require(content_id(identity) == declared_id, "source gate result ID changed")
    gate_passed = genuine_boolean(result["gate_passed"], name="gate_passed")
    authorized = genuine_boolean(
        result["confirmation_access_authorized"],
        name="confirmation_access_authorized",
    )
    _require(
        gate_passed is authorized, "source gate authorization differs from decision"
    )
    _require(
        result["status"]
        == ("source-gate-passed" if gate_passed else "source-gate-failed"),
        "source gate status changed",
    )
    checks = _mapping(result["checks"], name="checks")
    _require(
        all(type(value) is bool for value in checks.values())
        and sum(checks.values()) == result["passed_check_count"]
        and len(checks) == result["total_check_count"]
        and all(checks.values()) is gate_passed,
        "source gate check accounting changed",
    )
    boundary = _mapping(result["information_boundary"], name="information_boundary")
    _require(
        boundary.get("calibration_gate_evaluated") is True
        and boundary.get("confirmation_payloads_opened") is False
        and boundary.get("target_outcomes_used") is False
        and boundary.get("future_frames_used") is False
        and boundary.get("replacement_allowed") is False
        and boundary.get("human_approval_required") is False
        and boundary.get("new_measurements_required") is False,
        "source gate information boundary changed",
    )
    lock_path = _ordinary_file(
        root / SOURCE_GATE_LOCK_FILENAME, name="copied gate lock"
    )
    lock = load_source_gate_lock(lock_path)
    _require(
        lock["artifact_id"] == result["gate_lock_id"]
        and _sha256_file(lock_path) == result["gate_lock_file_sha256"],
        "copied source gate lock changed",
    )
    checksum_path = _ordinary_file(root / "SHA256SUMS", name="source gate SHA256SUMS")
    expected = "".join(
        f"{_sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
        for path in sorted(
            item
            for item in root.rglob("*")
            if item.is_file() and item.name != "SHA256SUMS"
        )
    )
    _require(checksum_path.read_text(encoding="ascii") == expected, "checksums changed")
    return cast(dict[str, Any], plain_json(result))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--visual-provider-spec", type=Path, required=True)
    parser.add_argument("--metric-prior-policy", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--source-calibration-root", type=Path, required=True)
    parser.add_argument("--source-calibration-result", type=Path, required=True)
    parser.add_argument("--gate-lock", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    samples = load_deform360_prob4d_calibration_samples(
        arguments.samples,
        selection_path=arguments.selection,
        visual_provider_spec_path=arguments.visual_provider_spec,
        metric_prior_policy_path=arguments.metric_prior_policy,
        prediction_root=arguments.prediction_root,
    )
    result = publish_source_gate_result(
        samples=samples,
        source_calibration_result_path=arguments.source_calibration_result,
        source_calibration_root=arguments.source_calibration_root,
        gate_lock_path=arguments.gate_lock,
        implementation_revision=arguments.implementation_revision,
        output_directory=arguments.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
