"""Source-calibrated horizon dynamics for model-averaged discrepancy endpoints.

The historical endpoint predictor remains unchanged. This additive module fits a
compact mean-retention and variance-growth model from independent source groups
and propagates a ``ModelAveragedEndpointPosteriorV1`` without target outcomes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from .endpoint_model_average import ModelAveragedEndpointPosteriorV1

HORIZON_DISCREPANCY_CALIBRATION_SCHEMA = (
    "bayesian-phystwin.horizon-discrepancy-calibration"
)
HORIZON_DISCREPANCY_CALIBRATION_VERSION = 1
HORIZON_DISCREPANCY_CALIBRATION_SEMANTICS = (
    "source-group-selected-mean-retention-and-process-growth-v1"
)

_CALIBRATION_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
        "artifact_id",
        "source_group_ids",
        "source_summary_sha256",
        "horizon_steps",
        "mean_reversion_half_life_steps",
        "minimum_mean_retention",
        "stationary_std_m",
        "process_std_m_per_sqrt_step",
        "component_process_variance_scale",
        "source_outcomes_used",
        "interval_calibration_outcomes_used",
        "confirmation_outcomes_used",
        "target_outcomes_used",
        "metadata",
    }
)


def _plain_json(value: Any) -> Any:
    """Return a finite JSON-compatible deep copy with string mapping keys."""

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("metadata mapping keys must be literal strings")
            result[key] = _plain_json(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain_json(value.tolist())
    if isinstance(value, np.generic):
        return _plain_json(value.item())
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not np.isfinite(value):
            raise ValueError("metadata must contain only finite numbers")
        return value
    raise ValueError(f"unsupported metadata value: {type(value).__name__}")


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        _plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _content_id(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a literal lowercase SHA-256 digest")
    return cast(str, value)


def _genuine_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be Boolean")
    return cast(bool, value)


def _genuine_int(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return cast(int, value)


def _finite_scalar(value: object, *, name: str, minimum: float = 0.0) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite real number >= {minimum}")
    result = float(value)
    if not np.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be a finite real number >= {minimum}")
    return result


def _probability(value: object, *, name: str) -> float:
    result = _finite_scalar(value, name=name)
    if result > 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _readonly(value: np.ndarray, *, dtype: np.dtype | type = np.float64) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _axis_vector(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite nonnegative length-three vector")
    result = np.asarray(raw, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError(f"{name} must be a finite nonnegative length-three vector")
    return _readonly(result)


def _canonical_groups(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("source_group_ids must be a sequence of strings")
    groups = tuple(values)
    if len(groups) < 2:
        raise ValueError("at least two independent source groups are required")
    if any(type(value) is not str or not value for value in groups):
        raise ValueError("source_group_ids must contain nonempty literal strings")
    if len(set(groups)) != len(groups):
        raise ValueError("source_group_ids must be unique")
    return tuple(sorted(groups))


def _horizon_vector(values: Sequence[int]) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind not in "iu":
        raise ValueError("horizon_steps must contain genuine integers")
    horizon = np.asarray(raw, dtype=np.int64)
    if (
        horizon.ndim != 1
        or len(horizon) < 1
        or np.any(horizon <= 0)
        or np.any(np.diff(horizon) <= 0)
    ):
        raise ValueError("horizon_steps must be a strictly increasing positive vector")
    return _readonly(horizon, dtype=np.int64)


def _array_digest(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(json.dumps(contiguous.shape, separators=(",", ":")).encode("ascii"))
    digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _source_summary_id(
    groups: Sequence[str],
    horizon: np.ndarray,
    endpoint: np.ndarray,
    future: np.ndarray,
) -> str:
    descriptor = {
        "schema": "bayesian-phystwin.horizon-discrepancy-source-summary",
        "schema_version": 1,
        "source_group_ids": list(groups),
        "horizon_steps": [int(value) for value in horizon],
        "endpoint_mean_sha256": _array_digest(endpoint),
        "future_mean_sha256": _array_digest(future),
    }
    return _content_id(descriptor)


@dataclass(frozen=True, slots=True)
class HorizonDiscrepancyCalibrationV1:
    """Source-only mean-retention and variance-growth calibration."""

    source_group_ids: Sequence[str]
    source_summary_sha256: str
    horizon_steps: Sequence[int]
    mean_reversion_half_life_steps: float | None
    minimum_mean_retention: float
    stationary_std_m: np.ndarray
    process_std_m_per_sqrt_step: np.ndarray
    component_process_variance_scale: float = 1.0
    source_outcomes_used: bool = True
    interval_calibration_outcomes_used: bool = False
    confirmation_outcomes_used: bool = False
    target_outcomes_used: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        groups = _canonical_groups(self.source_group_ids)
        source_summary = _sha256(
            self.source_summary_sha256,
            name="source_summary_sha256",
        )
        horizon = _horizon_vector(self.horizon_steps)
        half_life = self.mean_reversion_half_life_steps
        if half_life is not None:
            half_life = _finite_scalar(
                half_life,
                name="mean_reversion_half_life_steps",
                minimum=np.finfo(np.float64).tiny,
            )
        minimum_retention = _probability(
            self.minimum_mean_retention,
            name="minimum_mean_retention",
        )
        if half_life is None and minimum_retention != 1.0:
            raise ValueError(
                "no-reversion calibration requires minimum_mean_retention=1"
            )
        if half_life is not None and minimum_retention >= 1.0:
            raise ValueError(
                "finite mean reversion requires minimum_mean_retention < 1"
            )
        stationary = _axis_vector(self.stationary_std_m, name="stationary_std_m")
        process = _axis_vector(
            self.process_std_m_per_sqrt_step,
            name="process_std_m_per_sqrt_step",
        )
        if not np.any(process > 0.0):
            raise ValueError("process_std_m_per_sqrt_step requires a positive floor")
        component_scale = _finite_scalar(
            self.component_process_variance_scale,
            name="component_process_variance_scale",
        )
        source_used = _genuine_bool(
            self.source_outcomes_used,
            name="source_outcomes_used",
        )
        interval_used = _genuine_bool(
            self.interval_calibration_outcomes_used,
            name="interval_calibration_outcomes_used",
        )
        confirmation_used = _genuine_bool(
            self.confirmation_outcomes_used,
            name="confirmation_outcomes_used",
        )
        target_used = _genuine_bool(
            self.target_outcomes_used,
            name="target_outcomes_used",
        )
        if not source_used:
            raise ValueError("calibration must identify its source-only outcomes")
        if interval_used:
            raise ValueError("interval-calibration outcomes cannot select dynamics")
        if confirmation_used or target_used:
            raise ValueError("horizon dynamics must be frozen before target outcomes")
        metadata = MappingProxyType(
            cast(dict[str, Any], _plain_json(dict(self.metadata)))
        )

        object.__setattr__(self, "source_group_ids", groups)
        object.__setattr__(self, "source_summary_sha256", source_summary)
        object.__setattr__(
            self,
            "horizon_steps",
            tuple(int(value) for value in horizon),
        )
        object.__setattr__(self, "mean_reversion_half_life_steps", half_life)
        object.__setattr__(self, "minimum_mean_retention", minimum_retention)
        object.__setattr__(self, "stationary_std_m", stationary)
        object.__setattr__(self, "process_std_m_per_sqrt_step", process)
        object.__setattr__(self, "component_process_variance_scale", component_scale)
        object.__setattr__(self, "source_outcomes_used", source_used)
        object.__setattr__(self, "interval_calibration_outcomes_used", interval_used)
        object.__setattr__(self, "confirmation_outcomes_used", confirmation_used)
        object.__setattr__(self, "target_outcomes_used", target_used)
        object.__setattr__(self, "metadata", metadata)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": HORIZON_DISCREPANCY_CALIBRATION_SCHEMA,
            "schema_version": HORIZON_DISCREPANCY_CALIBRATION_VERSION,
            "semantics": HORIZON_DISCREPANCY_CALIBRATION_SEMANTICS,
            "source_group_ids": list(self.source_group_ids),
            "source_summary_sha256": self.source_summary_sha256,
            "horizon_steps": list(self.horizon_steps),
            "mean_reversion_half_life_steps": self.mean_reversion_half_life_steps,
            "minimum_mean_retention": self.minimum_mean_retention,
            "stationary_std_m": self.stationary_std_m.tolist(),
            "process_std_m_per_sqrt_step": (
                self.process_std_m_per_sqrt_step.tolist()
            ),
            "component_process_variance_scale": (
                self.component_process_variance_scale
            ),
            "source_outcomes_used": self.source_outcomes_used,
            "interval_calibration_outcomes_used": (
                self.interval_calibration_outcomes_used
            ),
            "confirmation_outcomes_used": self.confirmation_outcomes_used,
            "target_outcomes_used": self.target_outcomes_used,
            "metadata": _plain_json(self.metadata),
        }

    @property
    def artifact_id(self) -> str:
        return _content_id(self.descriptor())

    def to_record(self) -> dict[str, object]:
        return {"artifact_id": self.artifact_id, **self.descriptor()}

    @classmethod
    def from_mapping(cls, value: object) -> HorizonDiscrepancyCalibrationV1:
        if not isinstance(value, Mapping):
            raise ValueError("horizon calibration must be a JSON object")
        missing = sorted(_CALIBRATION_FIELDS - set(value))
        extra = sorted(set(value) - _CALIBRATION_FIELDS)
        if missing or extra:
            raise ValueError(
                f"horizon calibration fields changed: missing={missing}, extra={extra}"
            )
        if value["schema"] != HORIZON_DISCREPANCY_CALIBRATION_SCHEMA:
            raise ValueError("unsupported horizon calibration schema")
        version = _genuine_int(
            value["schema_version"],
            name="schema_version",
            minimum=1,
        )
        if version != HORIZON_DISCREPANCY_CALIBRATION_VERSION:
            raise ValueError("unsupported horizon calibration version")
        if value["semantics"] != HORIZON_DISCREPANCY_CALIBRATION_SEMANTICS:
            raise ValueError("horizon calibration semantics changed")
        result = cls(
            source_group_ids=cast(Sequence[str], value["source_group_ids"]),
            source_summary_sha256=cast(str, value["source_summary_sha256"]),
            horizon_steps=cast(Sequence[int], value["horizon_steps"]),
            mean_reversion_half_life_steps=cast(
                float | None,
                value["mean_reversion_half_life_steps"],
            ),
            minimum_mean_retention=cast(float, value["minimum_mean_retention"]),
            stationary_std_m=np.asarray(value["stationary_std_m"]),
            process_std_m_per_sqrt_step=np.asarray(
                value["process_std_m_per_sqrt_step"]
            ),
            component_process_variance_scale=cast(
                float,
                value["component_process_variance_scale"],
            ),
            source_outcomes_used=cast(bool, value["source_outcomes_used"]),
            interval_calibration_outcomes_used=cast(
                bool,
                value["interval_calibration_outcomes_used"],
            ),
            confirmation_outcomes_used=cast(
                bool,
                value["confirmation_outcomes_used"],
            ),
            target_outcomes_used=cast(bool, value["target_outcomes_used"]),
            metadata=cast(Mapping[str, Any], value["metadata"]),
        )
        declared_id = _sha256(value["artifact_id"], name="artifact_id")
        if declared_id != result.artifact_id:
            raise ValueError("horizon calibration artifact_id does not match content")
        return result


@dataclass(frozen=True, slots=True)
class HorizonConditionedEndpointPredictionV1:
    """Horizon-propagated endpoint moments and their calibration lineage."""

    mean_m: np.ndarray
    covariance_m2: np.ndarray
    component_weights: np.ndarray
    component_mean_m: np.ndarray
    component_variance_m2: np.ndarray
    additional_axis_variance_m2: np.ndarray
    horizon_steps: int
    mean_retention: float
    calibration_id: str

    def __post_init__(self) -> None:
        horizon = _genuine_int(
            self.horizon_steps,
            name="horizon_steps",
            minimum=0,
        )
        retention = _probability(self.mean_retention, name="mean_retention")
        calibration_id = _sha256(self.calibration_id, name="calibration_id")
        mean = np.asarray(self.mean_m, dtype=np.float64)
        covariance = np.asarray(self.covariance_m2, dtype=np.float64)
        weights = np.asarray(self.component_weights, dtype=np.float64)
        component_mean = np.asarray(self.component_mean_m, dtype=np.float64)
        component_variance = np.asarray(
            self.component_variance_m2,
            dtype=np.float64,
        )
        additional = np.asarray(
            self.additional_axis_variance_m2,
            dtype=np.float64,
        )
        if mean.ndim != 2 or mean.shape[1] != 3 or len(mean) < 1:
            raise ValueError("mean_m must have shape (N>=1, 3)")
        track_count = len(mean)
        if covariance.shape != (track_count, 3, 3):
            raise ValueError("covariance_m2 must have shape (N, 3, 3)")
        if weights.ndim != 2 or weights.shape[0] != track_count:
            raise ValueError("component_weights shape changed")
        component_count = weights.shape[1]
        if component_mean.shape != (component_count, track_count, 3):
            raise ValueError("component_mean_m shape changed")
        if component_variance.shape != (component_count, track_count):
            raise ValueError("component_variance_m2 shape changed")
        if additional.shape != (3,):
            raise ValueError("additional_axis_variance_m2 must have shape (3,)")
        arrays = (mean, covariance, weights, component_mean, component_variance, additional)
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("prediction contains non-finite values")
        if not np.allclose(covariance, covariance.transpose(0, 2, 1)):
            raise ValueError("covariance_m2 must be symmetric")
        if np.min(np.linalg.eigvalsh(covariance), initial=0.0) < -1e-12:
            raise ValueError("covariance_m2 must be positive semidefinite")
        if np.any(weights < 0.0) or not np.allclose(
            np.sum(weights, axis=1),
            1.0,
            atol=1e-12,
            rtol=1e-12,
        ):
            raise ValueError("component_weights must be row-normalized")
        if np.any(component_variance < 0.0) or np.any(additional < 0.0):
            raise ValueError("prediction variances must be nonnegative")
        object.__setattr__(self, "mean_m", _readonly(mean))
        object.__setattr__(self, "covariance_m2", _readonly(covariance))
        object.__setattr__(self, "component_weights", _readonly(weights))
        object.__setattr__(self, "component_mean_m", _readonly(component_mean))
        object.__setattr__(
            self,
            "component_variance_m2",
            _readonly(component_variance),
        )
        object.__setattr__(
            self,
            "additional_axis_variance_m2",
            _readonly(additional),
        )
        object.__setattr__(self, "horizon_steps", horizon)
        object.__setattr__(self, "mean_retention", retention)
        object.__setattr__(self, "calibration_id", calibration_id)


def mean_retention_at_horizon(
    calibration: HorizonDiscrepancyCalibrationV1,
    horizon_steps: int,
) -> float:
    """Return the calibrated endpoint-mean retention at one future horizon."""

    if not isinstance(calibration, HorizonDiscrepancyCalibrationV1):
        raise TypeError("calibration must be a HorizonDiscrepancyCalibrationV1")
    horizon = _genuine_int(horizon_steps, name="horizon_steps", minimum=0)
    half_life = calibration.mean_reversion_half_life_steps
    if half_life is None:
        return 1.0
    transient = 2.0 ** (-float(horizon) / half_life)
    floor = calibration.minimum_mean_retention
    return float(floor + (1.0 - floor) * transient)


def predict_horizon_conditioned_endpoint(
    posterior: ModelAveragedEndpointPosteriorV1,
    calibration: HorizonDiscrepancyCalibrationV1,
    *,
    horizon_steps: int,
) -> HorizonConditionedEndpointPredictionV1:
    """Propagate endpoint moments with source-frozen horizon dynamics."""

    if not isinstance(posterior, ModelAveragedEndpointPosteriorV1):
        raise TypeError("posterior must be a ModelAveragedEndpointPosteriorV1")
    if not isinstance(calibration, HorizonDiscrepancyCalibrationV1):
        raise TypeError("calibration must be a HorizonDiscrepancyCalibrationV1")
    horizon = _genuine_int(horizon_steps, name="horizon_steps", minimum=0)
    retention = mean_retention_at_horizon(calibration, horizon)
    retention_squared = retention * retention

    component_mean = retention * posterior.component_mean_m
    component_variance = (
        retention_squared * posterior.component_variance_m2
        + horizon
        * calibration.component_process_variance_scale
        * posterior.component_process_variance_m2[:, None]
    )
    stationary_variance = np.square(calibration.stationary_std_m)
    process_variance = np.square(calibration.process_std_m_per_sqrt_step)
    additional = (
        (1.0 - retention_squared) * stationary_variance
        + horizon * process_variance
    )
    mean = np.einsum(
        "nk,knc->nc",
        posterior.component_weights,
        component_mean,
    )
    centered = component_mean - mean[None, :, :]
    between = centered[:, :, :, None] * centered[:, :, None, :]
    within = (
        component_variance[:, :, None, None] * np.eye(3)
        + np.diag(additional)[None, None, :, :]
    )
    covariance = np.einsum(
        "nk,knij->nij",
        posterior.component_weights,
        within + between,
    )
    covariance = 0.5 * (covariance + covariance.transpose(0, 2, 1))
    return HorizonConditionedEndpointPredictionV1(
        mean_m=mean,
        covariance_m2=covariance,
        component_weights=posterior.component_weights,
        component_mean_m=component_mean,
        component_variance_m2=component_variance,
        additional_axis_variance_m2=additional,
        horizon_steps=horizon,
        mean_retention=retention,
        calibration_id=calibration.artifact_id,
    )


def _candidate_retention(
    half_life: float | None,
    floor: float,
    horizon: np.ndarray,
) -> np.ndarray:
    if half_life is None:
        return np.ones(len(horizon), dtype=np.float64)
    return floor + (1.0 - floor) * np.power(
        2.0,
        -horizon.astype(np.float64) / half_life,
    )


def _fit_nonnegative_two_term(
    design: np.ndarray,
    target: np.ndarray,
    *,
    process_floor: float,
) -> np.ndarray:
    candidates: list[np.ndarray] = []
    unconstrained = np.linalg.lstsq(design, target, rcond=None)[0]
    if unconstrained[0] >= 0.0 and unconstrained[1] >= process_floor:
        candidates.append(unconstrained)

    first = design[:, 0]
    second = design[:, 1]
    first_denominator = float(first @ first)
    second_denominator = float(second @ second)

    residual_at_floor = target - process_floor * second
    stationary = (
        max(0.0, float(first @ residual_at_floor) / first_denominator)
        if first_denominator > 0.0
        else 0.0
    )
    candidates.append(np.asarray([stationary, process_floor]))

    process = (
        max(process_floor, float(second @ target) / second_denominator)
        if second_denominator > 0.0
        else process_floor
    )
    candidates.append(np.asarray([0.0, process]))

    return min(
        candidates,
        key=lambda candidate: float(
            np.sum(np.square(design @ candidate - target))
        ),
    )


def fit_horizon_discrepancy_calibration(
    source_group_ids: Sequence[str],
    endpoint_mean_m: np.ndarray,
    future_mean_m: np.ndarray,
    horizon_steps: Sequence[int],
    *,
    half_life_candidates: Sequence[float | None] = (
        None,
        4.0,
        8.0,
        16.0,
        32.0,
        64.0,
        128.0,
    ),
    minimum_retention_candidates: Sequence[float] = (0.0, 0.25, 0.5, 0.75),
    component_process_variance_scale: float = 1.0,
    minimum_process_std_m_per_sqrt_step: float = 1e-6,
    metadata: Mapping[str, Any] | None = None,
) -> HorizonDiscrepancyCalibrationV1:
    """Fit compact horizon dynamics with equal weight per independent group."""

    input_groups = tuple(source_group_ids)
    groups = _canonical_groups(input_groups)
    endpoint = np.asarray(endpoint_mean_m, dtype=np.float64)
    future = np.asarray(future_mean_m, dtype=np.float64)
    horizon = _horizon_vector(horizon_steps)
    if endpoint.shape != (len(input_groups), 3):
        raise ValueError("endpoint_mean_m must have shape (group, 3)")
    if future.shape != (len(input_groups), len(horizon), 3):
        raise ValueError("future_mean_m must have shape (group, horizon, 3)")
    if not np.all(np.isfinite(endpoint)) or not np.all(np.isfinite(future)):
        raise ValueError("source discrepancy summaries must be finite")
    order = np.argsort(np.asarray(input_groups), kind="stable")
    endpoint = np.ascontiguousarray(endpoint[order])
    future = np.ascontiguousarray(future[order])

    component_scale = _finite_scalar(
        component_process_variance_scale,
        name="component_process_variance_scale",
    )
    process_floor_std = _finite_scalar(
        minimum_process_std_m_per_sqrt_step,
        name="minimum_process_std_m_per_sqrt_step",
        minimum=np.finfo(np.float64).tiny,
    )
    process_floor_variance = process_floor_std**2

    normalized_half_lives: list[float | None] = []
    for value in half_life_candidates:
        if value is None:
            normalized_half_lives.append(None)
        else:
            normalized_half_lives.append(
                _finite_scalar(
                    value,
                    name="half_life_candidates entry",
                    minimum=np.finfo(np.float64).tiny,
                )
            )
    if not normalized_half_lives:
        raise ValueError("at least one half-life candidate is required")
    floors = sorted(
        {
            _probability(value, name="minimum_retention_candidates entry")
            for value in minimum_retention_candidates
        }
    )
    if not floors:
        raise ValueError("at least one minimum-retention candidate is required")

    candidates: list[tuple[float | None, float, np.ndarray, float]] = []
    for half_life in normalized_half_lives:
        candidate_floors = (1.0,) if half_life is None else tuple(
            floor for floor in floors if floor < 1.0
        )
        for floor in candidate_floors:
            retention = _candidate_retention(half_life, floor, horizon)
            error = future - retention[None, :, None] * endpoint[:, None, :]
            group_loss = np.mean(np.linalg.norm(error, axis=2), axis=1)
            score = float(np.mean(group_loss))
            candidates.append((half_life, floor, retention, score))
    if not candidates:
        raise ValueError("horizon candidate grid is empty")
    selected_half_life, selected_floor, retention, _ = min(
        candidates,
        key=lambda candidate: candidate[3],
    )

    residual = future - retention[None, :, None] * endpoint[:, None, :]
    second_moment = np.mean(np.square(residual), axis=0)
    decay = 1.0 - np.square(retention)
    design = np.column_stack(
        [decay, horizon.astype(np.float64)]
    )
    stationary_variance = np.empty(3, dtype=np.float64)
    process_variance = np.empty(3, dtype=np.float64)
    for axis in range(3):
        coefficients = _fit_nonnegative_two_term(
            design,
            second_moment[:, axis],
            process_floor=process_floor_variance,
        )
        stationary_variance[axis] = coefficients[0]
        process_variance[axis] = coefficients[1]

    scores = {
        (
            "half_life=infinite"
            if half_life is None
            else f"half_life={half_life:.12g}"
        )
        + f";minimum_retention={floor:.12g}": score
        for half_life, floor, _, score in candidates
    }
    source_summary = _source_summary_id(
        groups,
        horizon,
        endpoint,
        future,
    )
    return HorizonDiscrepancyCalibrationV1(
        source_group_ids=groups,
        source_summary_sha256=source_summary,
        horizon_steps=tuple(int(value) for value in horizon),
        mean_reversion_half_life_steps=selected_half_life,
        minimum_mean_retention=selected_floor,
        stationary_std_m=np.sqrt(stationary_variance),
        process_std_m_per_sqrt_step=np.sqrt(process_variance),
        component_process_variance_scale=component_scale,
        metadata={
            "selection_objective": "equal-group-mean-euclidean-error",
            "candidate_scores_m": scores,
            "minimum_process_std_m_per_sqrt_step": process_floor_std,
            **({} if metadata is None else cast(dict[str, Any], _plain_json(metadata))),
        },
    )


def save_horizon_discrepancy_calibration(
    path: str | Path,
    calibration: HorizonDiscrepancyCalibrationV1,
) -> None:
    """Write a canonical human-readable calibration artifact."""

    if not isinstance(calibration, HorizonDiscrepancyCalibrationV1):
        raise TypeError("calibration must be a HorizonDiscrepancyCalibrationV1")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(
            calibration.to_record(),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not permitted: {value}")


def load_horizon_discrepancy_calibration(
    path: str | Path,
) -> HorizonDiscrepancyCalibrationV1:
    """Strictly load and independently revalidate a calibration artifact."""

    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read horizon calibration: {path}") from error
    return HorizonDiscrepancyCalibrationV1.from_mapping(value)


__all__ = [
    "HORIZON_DISCREPANCY_CALIBRATION_SCHEMA",
    "HORIZON_DISCREPANCY_CALIBRATION_SEMANTICS",
    "HORIZON_DISCREPANCY_CALIBRATION_VERSION",
    "HorizonConditionedEndpointPredictionV1",
    "HorizonDiscrepancyCalibrationV1",
    "fit_horizon_discrepancy_calibration",
    "load_horizon_discrepancy_calibration",
    "mean_retention_at_horizon",
    "predict_horizon_conditioned_endpoint",
    "save_horizon_discrepancy_calibration",
]
