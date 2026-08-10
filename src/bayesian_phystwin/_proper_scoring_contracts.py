"""Fail-closed predictive-distribution contracts for proper scoring."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

_FORECAST_FIELDS: Final = frozenset(
    {"samples", "gaussian", "scalar_intervals"}
)
_GAUSSIAN_FIELDS: Final = frozenset({"mean", "covariance"})
_INTERVAL_FORECAST_FIELDS: Final = frozenset(
    {"median", "central_intervals"}
)
_INTERVAL_FIELDS: Final = frozenset(
    {"nominal_coverage", "lower", "upper"}
)
_PAIR_FIELDS: Final = frozenset({"left", "right", "weight"})


@dataclass(frozen=True, slots=True)
class _ScalarInterval:
    nominal_coverage: float
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class _ScalarIntervalForecast:
    median: float
    intervals: tuple[_ScalarInterval, ...]


@dataclass(frozen=True, slots=True)
class _Forecast:
    dimension: int
    samples: np.ndarray | None
    gaussian_mean: np.ndarray | None
    gaussian_covariance: np.ndarray | None
    scalar_intervals: _ScalarIntervalForecast | None
    identity: Mapping[str, Any]

    @property
    def families(self) -> tuple[str, ...]:
        result: list[str] = []
        if self.samples is not None:
            result.append("energy_score")
        if self.gaussian_mean is not None:
            result.append("gaussian_log_score_shifted")
        if self.scalar_intervals is not None:
            result.append("weighted_interval_score")
        return tuple(result)


@dataclass(frozen=True, slots=True)
class _Pair:
    left: int
    right: int
    weight: float


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _sequence(value: object, *, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _closed_fields(
    value: Mapping[str, object],
    allowed: frozenset[str],
    *,
    name: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{name} contains unknown fields: {unknown}")


def _text(value: object, *, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _number(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be boolean")
    return value


def _horizon(value: object, *, name: str) -> str:
    if type(value) is str:
        return _text(value, name=name)
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a label or nonnegative number")
    number = _number(value, name=name, minimum=0.0)
    return f"{number:.12g}"


def _numeric_array(value: object, *, name: str, ndim: int) -> np.ndarray:
    raw = np.asarray(value)
    _require(raw.dtype.kind in {"i", "u", "f"}, f"{name} must be real numeric")
    result = np.asarray(raw, dtype=np.float64)
    _require(result.ndim == ndim, f"{name} must have {ndim} dimensions")
    _require(np.all(np.isfinite(result)), f"{name} must be finite")
    return result


def _vector(
    value: object,
    *,
    name: str,
    maximum_dimension: int,
) -> np.ndarray:
    result = _numeric_array(value, name=name, ndim=1)
    _require(len(result) >= 1, f"{name} must not be empty")
    _require(
        len(result) <= maximum_dimension,
        f"{name} exceeds the dimension budget",
    )
    return result


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_identity(value: np.ndarray) -> list[object]:
    return np.asarray(value, dtype=np.float64).tolist()


def _parse_scalar_intervals(
    value: object,
    *,
    name: str,
) -> _ScalarIntervalForecast:
    interval_forecast = _mapping(value, name=name)
    _closed_fields(
        interval_forecast,
        _INTERVAL_FORECAST_FIELDS,
        name=name,
    )
    median = _number(interval_forecast.get("median"), name=f"{name}.median")
    raw_intervals = _sequence(
        interval_forecast.get("central_intervals"),
        name=f"{name}.central_intervals",
    )
    _require(bool(raw_intervals), f"{name}.central_intervals must not be empty")
    intervals: list[_ScalarInterval] = []
    seen: set[float] = set()
    for index, raw_interval in enumerate(raw_intervals):
        item_name = f"{name}.central_intervals[{index}]"
        item = _mapping(raw_interval, name=item_name)
        _closed_fields(item, _INTERVAL_FIELDS, name=item_name)
        nominal = _number(
            item.get("nominal_coverage"),
            name=f"{item_name}.nominal_coverage",
            minimum=0.0,
            maximum=1.0,
        )
        _require(
            nominal not in {0.0, 1.0},
            f"{item_name}.nominal_coverage must lie in (0, 1)",
        )
        _require(
            nominal not in seen,
            f"{name} repeats nominal coverage {nominal}",
        )
        seen.add(nominal)
        lower = _number(item.get("lower"), name=f"{item_name}.lower")
        upper = _number(item.get("upper"), name=f"{item_name}.upper")
        _require(lower <= upper, f"{item_name} has lower > upper")
        _require(
            lower <= median <= upper,
            f"{item_name} does not contain the registered median",
        )
        intervals.append(_ScalarInterval(nominal, lower, upper))
    intervals.sort(key=lambda item: item.nominal_coverage)
    for narrower, wider in zip(intervals[:-1], intervals[1:], strict=True):
        _require(
            wider.lower <= narrower.lower and wider.upper >= narrower.upper,
            f"{name}.central_intervals must be nested by nominal coverage",
        )
    return _ScalarIntervalForecast(median, tuple(intervals))


def _parse_forecast(
    value: object,
    *,
    name: str,
    dimension: int,
    maximum_samples_per_forecast: int,
    maximum_dimension: int,
    maximum_array_elements: int,
) -> _Forecast:
    forecast = _mapping(value, name=name)
    _closed_fields(forecast, _FORECAST_FIELDS, name=name)
    _require(bool(forecast), f"{name} must contain a predictive representation")

    samples: np.ndarray | None = None
    if "samples" in forecast:
        samples = _numeric_array(
            forecast["samples"],
            name=f"{name}.samples",
            ndim=2,
        )
        _require(len(samples) >= 1, f"{name}.samples must not be empty")
        _require(
            len(samples) <= maximum_samples_per_forecast,
            f"{name}.samples exceeds the sample-count budget",
        )
        _require(
            samples.shape[1] == dimension,
            f"{name}.samples has changed query dimension",
        )
        _require(
            samples.size <= maximum_array_elements,
            f"{name}.samples exceeds the array-element budget",
        )

    gaussian_mean: np.ndarray | None = None
    gaussian_covariance: np.ndarray | None = None
    if "gaussian" in forecast:
        gaussian = _mapping(forecast["gaussian"], name=f"{name}.gaussian")
        _closed_fields(gaussian, _GAUSSIAN_FIELDS, name=f"{name}.gaussian")
        gaussian_mean = _vector(
            gaussian.get("mean"),
            name=f"{name}.gaussian.mean",
            maximum_dimension=maximum_dimension,
        )
        _require(
            len(gaussian_mean) == dimension,
            f"{name}.gaussian.mean has changed query dimension",
        )
        gaussian_covariance = _numeric_array(
            gaussian.get("covariance"),
            name=f"{name}.gaussian.covariance",
            ndim=2,
        )
        _require(
            gaussian_covariance.shape == (dimension, dimension),
            f"{name}.gaussian.covariance has changed shape",
        )
        _require(
            gaussian_covariance.size <= maximum_array_elements,
            f"{name}.gaussian.covariance exceeds the array-element budget",
        )
        _require(
            np.allclose(
                gaussian_covariance,
                gaussian_covariance.T,
                atol=1e-12,
                rtol=1e-12,
            ),
            f"{name}.gaussian.covariance must be symmetric",
        )
        try:
            np.linalg.cholesky(
                0.5 * (gaussian_covariance + gaussian_covariance.T)
            )
        except np.linalg.LinAlgError as error:
            raise ValueError(
                f"{name}.gaussian.covariance must be positive definite"
            ) from error

    scalar_intervals: _ScalarIntervalForecast | None = None
    if "scalar_intervals" in forecast:
        _require(
            dimension == 1,
            f"{name}.scalar_intervals requires a scalar query",
        )
        scalar_intervals = _parse_scalar_intervals(
            forecast["scalar_intervals"],
            name=f"{name}.scalar_intervals",
        )

    identity: dict[str, Any] = {}
    if samples is not None:
        identity["samples"] = _array_identity(samples)
    if gaussian_mean is not None and gaussian_covariance is not None:
        identity["gaussian"] = {
            "mean": _array_identity(gaussian_mean),
            "covariance": _array_identity(gaussian_covariance),
        }
    if scalar_intervals is not None:
        identity["scalar_intervals"] = {
            "median": scalar_intervals.median,
            "central_intervals": [
                {
                    "nominal_coverage": interval.nominal_coverage,
                    "lower": interval.lower,
                    "upper": interval.upper,
                }
                for interval in scalar_intervals.intervals
            ],
        }
    return _Forecast(
        dimension=dimension,
        samples=samples,
        gaussian_mean=gaussian_mean,
        gaussian_covariance=gaussian_covariance,
        scalar_intervals=scalar_intervals,
        identity=identity,
    )


def _parse_pairs(
    value: object,
    *,
    name: str,
    dimension: int,
    maximum_variogram_pairs: int,
) -> tuple[_Pair, ...]:
    if value is None:
        return ()
    raw_pairs = _sequence(value, name=name)
    _require(
        len(raw_pairs) <= maximum_variogram_pairs,
        f"{name} exceeds the pair-count budget",
    )
    pairs: list[_Pair] = []
    seen: set[tuple[int, int]] = set()
    for index, raw_pair in enumerate(raw_pairs):
        item_name = f"{name}[{index}]"
        item = _mapping(raw_pair, name=item_name)
        _closed_fields(item, _PAIR_FIELDS, name=item_name)
        left = _integer(item.get("left"), name=f"{item_name}.left")
        right = _integer(item.get("right"), name=f"{item_name}.right")
        _require(left < dimension, f"{item_name}.left is outside the query")
        _require(right < dimension, f"{item_name}.right is outside the query")
        _require(left != right, f"{item_name} must connect distinct coordinates")
        canonical = (min(left, right), max(left, right))
        _require(canonical not in seen, f"{name} repeats pair {canonical}")
        seen.add(canonical)
        weight = _number(
            item.get("weight"),
            name=f"{item_name}.weight",
            minimum=0.0,
        )
        _require(weight > 0.0, f"{item_name}.weight must be positive")
        pairs.append(_Pair(canonical[0], canonical[1], weight))
    pairs.sort(key=lambda pair: (pair.left, pair.right))
    return tuple(pairs)


def _pair_identity(pairs: Sequence[_Pair]) -> list[dict[str, object]]:
    return [
        {"left": pair.left, "right": pair.right, "weight": pair.weight}
        for pair in pairs
    ]


def _query_signature(
    forecast: _Forecast,
    pairs: Sequence[_Pair],
) -> Mapping[str, object]:
    interval_coverages: list[float] = []
    if forecast.scalar_intervals is not None:
        interval_coverages = [
            item.nominal_coverage for item in forecast.scalar_intervals.intervals
        ]
    families = list(forecast.families)
    if pairs:
        families.append("variogram_score")
    return {
        "dimension": forecast.dimension,
        "families": sorted(families),
        "interval_coverages": interval_coverages,
        "variogram_pairs": _pair_identity(pairs),
    }


__all__ = [
    "_Forecast",
    "_Pair",
    "_ScalarIntervalForecast",
    "_array_identity",
    "_boolean",
    "_canonical_json_sha256",
    "_closed_fields",
    "_horizon",
    "_integer",
    "_mapping",
    "_number",
    "_pair_identity",
    "_parse_forecast",
    "_parse_pairs",
    "_query_signature",
    "_require",
    "_sequence",
    "_text",
    "_vector",
]
