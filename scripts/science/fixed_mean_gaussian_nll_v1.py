"""Retrospective fixed-mean Gaussian NLL decomposition.

The diagnostic compares two covariance matrices attached to one shared predictive
mean. It is source-only infrastructure and never authorizes target access or a
scientific claim.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from statistics import NormalDist
from typing import Final, cast

import numpy as np

INPUT_CONTRACT: Final = "bayesian-phystwin.fixed-mean-gaussian-nll-diagnostic-input-v1"
REPORT_CONTRACT: Final = (
    "bayesian-phystwin.fixed-mean-gaussian-nll-diagnostic-report-v1"
)
ANALYSIS_STATUS: Final = "retrospective-source-only-non-claim-bearing"
SCHEMA_VERSION: Final = 1
CLAIM_BOUNDARY: Final = (
    "Retrospective explanatory diagnostic only. It does not select or calibrate "
    "a covariance, authorize target access, identify physical state, establish "
    "transfer or intervention value, or support a scientific claim."
)
ROOT_FIELDS: Final = frozenset(
    {
        "analysis_id",
        "analysis_status",
        "candidate_arm_id",
        "claim_authorized",
        "contract",
        "horizon_order",
        "maximum_condition_number",
        "nominal_coverage",
        "observation_model_id",
        "protocol_id",
        "query_id",
        "records",
        "reference_arm_id",
        "schema_version",
        "scientific_boundary",
        "source_artifact_id",
        "statistical_unit",
    }
)
RECORD_FIELDS: Final = frozenset(
    {
        "candidate_covariance",
        "group_id",
        "horizon",
        "mean",
        "observation",
        "reference_covariance",
        "unit_id",
    }
)
METRICS: Final = (
    "nll_difference_per_dimension",
    "sharpness_difference_per_dimension",
    "standardized_error_difference_per_dimension",
    "reference_marginal_coverage",
    "candidate_marginal_coverage",
    "reference_mean_full_interval_width",
    "candidate_mean_full_interval_width",
)


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return cast(Mapping[str, object], value)


def _sequence(value: object, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return cast(Sequence[object], value)


def _exact_fields(
    value: Mapping[str, object], expected: frozenset[str], name: str
) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise ValueError(f"{name} fields changed: missing={missing}, extra={extra}")


def _text(value: object, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    if any(character in value for character in "\x00\n\r"):
        raise ValueError(f"{name} must be a single canonical line")
    return value


def _finite(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    if minimum is not None and (result <= minimum if exclusive else result < minimum):
        raise ValueError(f"{name} is below its registered minimum")
    if maximum is not None and (result >= maximum if exclusive else result > maximum):
        raise ValueError(f"{name} exceeds its registered maximum")
    return result


def _array(value: object, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise ValueError(f"{name} must contain real numeric values")
    result = np.asarray(raw, dtype=np.float64)
    if result.ndim < 1 or result.size < 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a nonempty finite numeric array")
    return result


def _covariance(
    value: object,
    dimension: int,
    name: str,
    maximum_condition_number: float,
) -> np.ndarray:
    matrix = _array(value, name)
    if matrix.shape != (dimension, dimension):
        raise ValueError(f"{name} must have shape ({dimension}, {dimension})")
    if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    matrix = 0.5 * (matrix + matrix.T)
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be positive definite") from error
    condition_number = float(np.linalg.cond(matrix))
    if not np.isfinite(condition_number) or (
        condition_number > maximum_condition_number
    ):
        raise ValueError(f"{name} exceeds the maximum condition number")
    return matrix


def _content_id(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def gaussian_nll_terms(
    mean: object,
    covariance: object,
    observation: object,
    *,
    maximum_condition_number: float = 1e14,
) -> dict[str, float | int]:
    """Return normalization, sharpness, and standardized-error NLL terms."""

    location = _array(mean, "mean")
    target = _array(observation, "observation")
    if location.shape != target.shape:
        raise ValueError("mean shape differs from observation shape")
    limit = _finite(
        maximum_condition_number,
        "maximum_condition_number",
        minimum=1.0,
    )
    dimension = target.size
    matrix = _covariance(covariance, dimension, "covariance", limit)
    factor = np.linalg.cholesky(matrix)
    whitened = np.linalg.solve(factor, np.asarray(target - location).reshape(-1))
    log_determinant = 2.0 * float(np.sum(np.log(np.diag(factor))))
    mahalanobis_squared = float(whitened @ whitened)
    normalization = 0.5 * math.log(2.0 * math.pi)
    sharpness = 0.5 * log_determinant / dimension
    standardized_error = 0.5 * mahalanobis_squared / dimension
    total = math.fsum((normalization, sharpness, standardized_error))
    values = (
        log_determinant,
        mahalanobis_squared,
        normalization,
        sharpness,
        standardized_error,
        total,
    )
    if not all(np.isfinite(value) for value in values):
        raise FloatingPointError("Gaussian NLL decomposition became nonfinite")
    return {
        "dimension": int(dimension),
        "log_determinant": log_determinant,
        "mahalanobis_squared": mahalanobis_squared,
        "normalization_per_dimension": normalization,
        "sharpness_per_dimension": sharpness,
        "standardized_error_per_dimension": standardized_error,
        "total_per_dimension": total,
    }


def _mean_metrics(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot average an empty metric collection")
    result = {
        key: float(math.fsum(float(row[key]) for row in rows) / len(rows))
        for key in METRICS
    }
    reference_width = result["reference_mean_full_interval_width"]
    if reference_width <= 0.0:
        raise FloatingPointError("reference interval width became nonpositive")
    result["candidate_to_reference_width_ratio"] = (
        result["candidate_mean_full_interval_width"] / reference_width
    )
    residual = result["nll_difference_per_dimension"] - math.fsum(
        (
            result["sharpness_difference_per_dimension"],
            result["standardized_error_difference_per_dimension"],
        )
    )
    tolerance = (
        128.0
        * np.finfo(np.float64).eps
        * max(1.0, *(abs(value) for value in result.values()))
    )
    if not all(np.isfinite(value) for value in result.values()):
        raise FloatingPointError("aggregated diagnostic became nonfinite")
    if abs(residual) > tolerance:
        raise FloatingPointError("aggregated Gaussian NLL identity failed")
    result["decomposition_residual"] = float(residual)
    return result


def _record_metrics(
    value: object,
    index: int,
    horizons: tuple[str, ...],
    nominal_coverage: float,
    maximum_condition_number: float,
) -> dict[str, object]:
    name = f"records[{index}]"
    record = _mapping(value, name)
    _exact_fields(record, RECORD_FIELDS, name)
    mean = _array(record["mean"], f"{name}.mean")
    observation = _array(record["observation"], f"{name}.observation")
    if mean.shape != observation.shape:
        raise ValueError(f"{name} mean shape differs from observation shape")
    horizon = _text(record["horizon"], f"{name}.horizon")
    if horizon not in horizons:
        raise ValueError(f"{name}.horizon is outside horizon_order")
    reference_covariance = _covariance(
        record["reference_covariance"],
        mean.size,
        f"{name}.reference_covariance",
        maximum_condition_number,
    )
    candidate_covariance = _covariance(
        record["candidate_covariance"],
        mean.size,
        f"{name}.candidate_covariance",
        maximum_condition_number,
    )
    reference = gaussian_nll_terms(
        mean,
        reference_covariance,
        observation,
        maximum_condition_number=maximum_condition_number,
    )
    candidate = gaussian_nll_terms(
        mean,
        candidate_covariance,
        observation,
        maximum_condition_number=maximum_condition_number,
    )
    quantile = NormalDist().inv_cdf(0.5 + 0.5 * nominal_coverage)
    error = np.abs(np.asarray(observation - mean).reshape(-1))

    def coverage_width(covariance: np.ndarray) -> tuple[float, float]:
        standard_deviation = np.sqrt(np.diag(covariance))
        coverage = float(np.mean(error <= quantile * standard_deviation))
        width = float(np.mean(2.0 * quantile * standard_deviation))
        if not np.isfinite(coverage) or not np.isfinite(width) or width <= 0.0:
            raise FloatingPointError("coverage or interval width became invalid")
        return coverage, width

    reference_coverage, reference_width = coverage_width(reference_covariance)
    candidate_coverage, candidate_width = coverage_width(candidate_covariance)
    metrics = _mean_metrics(
        [
            {
                "candidate_marginal_coverage": candidate_coverage,
                "candidate_mean_full_interval_width": candidate_width,
                "nll_difference_per_dimension": float(
                    candidate["total_per_dimension"] - reference["total_per_dimension"]
                ),
                "reference_marginal_coverage": reference_coverage,
                "reference_mean_full_interval_width": reference_width,
                "sharpness_difference_per_dimension": float(
                    candidate["sharpness_per_dimension"]
                    - reference["sharpness_per_dimension"]
                ),
                "standardized_error_difference_per_dimension": float(
                    candidate["standardized_error_per_dimension"]
                    - reference["standardized_error_per_dimension"]
                ),
            }
        ]
    )
    return {
        "group_id": _text(record["group_id"], f"{name}.group_id"),
        "horizon": horizon,
        "metrics": metrics,
        "unit_id": _text(record["unit_id"], f"{name}.unit_id"),
    }


def analyze_fixed_mean_gaussian_nll(payload: object) -> dict[str, object]:
    """Validate and analyze one retrospective fixed-mean covariance bundle."""

    root = _mapping(payload, "input")
    _exact_fields(root, ROOT_FIELDS, "input")
    if root["contract"] != INPUT_CONTRACT or root["schema_version"] != 1:
        raise ValueError("input contract or schema version changed")
    if root["analysis_status"] != ANALYSIS_STATUS:
        raise ValueError("analysis must remain retrospective and non-claim-bearing")
    if type(root["claim_authorized"]) is not bool or root["claim_authorized"]:
        raise ValueError("claim_authorized must remain false")
    labels = {
        field: _text(root[field], field)
        for field in (
            "analysis_id",
            "candidate_arm_id",
            "observation_model_id",
            "protocol_id",
            "query_id",
            "reference_arm_id",
            "scientific_boundary",
            "source_artifact_id",
            "statistical_unit",
        )
    }
    if labels["candidate_arm_id"] == labels["reference_arm_id"]:
        raise ValueError("reference and candidate arm identities must differ")
    nominal_coverage = _finite(
        root["nominal_coverage"],
        "nominal_coverage",
        minimum=0.5,
        maximum=1.0,
        exclusive=True,
    )
    maximum_condition_number = _finite(
        root["maximum_condition_number"],
        "maximum_condition_number",
        minimum=1.0,
    )
    horizons = tuple(
        _text(value, f"horizon_order[{index}]")
        for index, value in enumerate(_sequence(root["horizon_order"], "horizon_order"))
    )
    if not horizons or len(set(horizons)) != len(horizons):
        raise ValueError("horizon_order must contain unique labels")
    records = tuple(
        _record_metrics(
            value,
            index,
            horizons,
            nominal_coverage,
            maximum_condition_number,
        )
        for index, value in enumerate(_sequence(root["records"], "records"))
    )
    if not records:
        raise ValueError("records must not be empty")
    horizon_index = {horizon: index for index, horizon in enumerate(horizons)}
    keys = tuple(
        (
            row["group_id"],
            horizon_index[cast(str, row["horizon"])],
            row["unit_id"],
        )
        for row in records
    )
    if keys != tuple(sorted(keys)):
        raise ValueError("records must be sorted by group, horizon_order, and unit")
    unit_ids = [row["unit_id"] for row in records]
    if len(set(unit_ids)) != len(unit_ids):
        raise ValueError("records repeat a unit_id")
    groups = tuple(sorted({cast(str, row["group_id"]) for row in records}))
    roster = {(row["group_id"], row["horizon"]) for row in records}
    expected = {(group, horizon) for group in groups for horizon in horizons}
    if roster != expected:
        raise ValueError("records must contain every group-by-horizon cell")

    cells = {
        (group, horizon): _mean_metrics(
            [
                cast(Mapping[str, float], row["metrics"])
                for row in records
                if row["group_id"] == group and row["horizon"] == horizon
            ]
        )
        for group in groups
        for horizon in horizons
    }
    group_rows = [
        {
            "by_horizon": {horizon: cells[(group, horizon)] for horizon in horizons},
            "group_id": group,
            "overall": _mean_metrics([cells[(group, horizon)] for horizon in horizons]),
        }
        for group in groups
    ]
    group_metrics = [cast(Mapping[str, float], row["overall"]) for row in group_rows]
    differences = [row["nll_difference_per_dimension"] for row in group_metrics]
    tie_tolerance = float(
        128.0
        * np.finfo(np.float64).eps
        * max(1.0, *(abs(value) for value in differences))
    )
    better = int(sum(value < -tie_tolerance for value in differences))
    worse = int(sum(value > tie_tolerance for value in differences))
    report: dict[str, object] = {
        **labels,
        "analysis_status": ANALYSIS_STATUS,
        "claim_authorized": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "component_identity": (
            "NLL difference equals sharpness difference plus standardized-error "
            "difference because both arms use one shared mean."
        ),
        "contract": REPORT_CONTRACT,
        "difference_semantics": (
            "candidate covariance minus reference covariance; lower NLL is better"
        ),
        "fixed_mean_by_construction": True,
        "group_analyses": group_rows,
        "group_count": len(groups),
        "horizon_order": list(horizons),
        "input_id": _content_id(root),
        "maximum_condition_number": maximum_condition_number,
        "nominal_coverage": nominal_coverage,
        "record_analyses": list(records),
        "record_count": len(records),
        "schema_version": SCHEMA_VERSION,
        "summary": {
            "better_worse_tie_groups": [
                better,
                worse,
                len(groups) - better - worse,
            ],
            "by_horizon": {
                horizon: _mean_metrics([cells[(group, horizon)] for group in groups])
                for horizon in horizons
            },
            "numerical_tie_tolerance": tie_tolerance,
            "overall": _mean_metrics(group_metrics),
        },
        "weighting": {
            "across_groups": "equal-group",
            "within_group": "equal-horizon",
            "within_group_horizon": "equal-record",
        },
    }
    report["report_id"] = _content_id(report)
    return report
