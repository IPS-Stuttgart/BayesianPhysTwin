"""Immutable contracts for source-calibrated horizon discrepancy prediction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

from ._canonical_contracts import (
    canonical_string_tuple,
    frozen_finite_json_mapping,
    genuine_boolean,
    genuine_integer,
    plain_json,
)
from ._horizon_discrepancy_common import (
    CALIBRATION_FIELDS,
    HORIZON_DISCREPANCY_CALIBRATION_SCHEMA,
    HORIZON_DISCREPANCY_CALIBRATION_SEMANTICS,
    HORIZON_DISCREPANCY_CALIBRATION_VERSION,
    axis_vector,
    finite_real,
    horizon_vector,
    probability,
    readonly,
)
from ._portable_contracts import (
    content_id,
    load_strict_json_object,
    require_exact_fields,
    sha256_digest,
    write_atomic_json,
)


@dataclass(frozen=True, slots=True)
class HorizonDiscrepancyCalibrationV1:
    """Source-only horizon dynamics frozen before interval and target outcomes."""

    source_group_ids: Sequence[str]
    source_summary_sha256: str
    horizon_steps: Sequence[int]
    mean_reversion_half_life_steps: float | None
    minimum_mean_retention: float
    stationary_std_m: np.ndarray
    additional_process_std_m_per_sqrt_step: np.ndarray
    component_process_variance_scale: float = 1.0
    source_outcomes_used: bool = True
    interval_calibration_outcomes_used: bool = False
    confirmation_outcomes_used: bool = False
    target_outcomes_used: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        groups = canonical_string_tuple(
            self.source_group_ids, name="source_group_ids", allow_empty=False
        )
        if len(groups) < 2:
            raise ValueError("at least two independent source groups are required")
        if len(set(groups)) != len(groups):
            raise ValueError("source_group_ids must be unique")
        groups = tuple(sorted(groups))
        summary_id = sha256_digest(
            self.source_summary_sha256, name="source_summary_sha256"
        )
        horizons = horizon_vector(self.horizon_steps, allow_zero=False)
        half_life = self.mean_reversion_half_life_steps
        if half_life is not None:
            half_life = finite_real(
                half_life,
                name="mean_reversion_half_life_steps",
                minimum=float(np.finfo(np.float64).tiny),
            )
        retention = probability(
            self.minimum_mean_retention, name="minimum_mean_retention"
        )
        if half_life is None and retention != 1.0:
            raise ValueError("no mean reversion requires minimum_mean_retention=1")
        if half_life is not None and retention >= 1.0:
            raise ValueError(
                "finite mean reversion requires minimum_mean_retention < 1"
            )
        stationary = axis_vector(self.stationary_std_m, name="stationary_std_m")
        process = axis_vector(
            self.additional_process_std_m_per_sqrt_step,
            name="additional_process_std_m_per_sqrt_step",
        )
        if not np.any(process > 0):
            raise ValueError("additional process uncertainty requires a positive floor")
        component_scale = finite_real(
            self.component_process_variance_scale,
            name="component_process_variance_scale",
        )
        source_used = genuine_boolean(
            self.source_outcomes_used, name="source_outcomes_used"
        )
        interval_used = genuine_boolean(
            self.interval_calibration_outcomes_used,
            name="interval_calibration_outcomes_used",
        )
        confirmation_used = genuine_boolean(
            self.confirmation_outcomes_used, name="confirmation_outcomes_used"
        )
        target_used = genuine_boolean(
            self.target_outcomes_used, name="target_outcomes_used"
        )
        if not source_used:
            raise ValueError("horizon dynamics must identify their source outcomes")
        if interval_used:
            raise ValueError(
                "interval-calibration outcomes cannot select horizon dynamics"
            )
        if confirmation_used or target_used:
            raise ValueError("horizon dynamics must be frozen before target outcomes")

        object.__setattr__(self, "source_group_ids", groups)
        object.__setattr__(self, "source_summary_sha256", summary_id)
        object.__setattr__(self, "horizon_steps", horizons)
        object.__setattr__(self, "mean_reversion_half_life_steps", half_life)
        object.__setattr__(self, "minimum_mean_retention", retention)
        object.__setattr__(self, "stationary_std_m", stationary)
        object.__setattr__(self, "additional_process_std_m_per_sqrt_step", process)
        object.__setattr__(self, "component_process_variance_scale", component_scale)
        object.__setattr__(self, "source_outcomes_used", source_used)
        object.__setattr__(self, "interval_calibration_outcomes_used", interval_used)
        object.__setattr__(self, "confirmation_outcomes_used", confirmation_used)
        object.__setattr__(self, "target_outcomes_used", target_used)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata, name="horizon discrepancy calibration metadata"
            ),
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": HORIZON_DISCREPANCY_CALIBRATION_SCHEMA,
            "schema_version": HORIZON_DISCREPANCY_CALIBRATION_VERSION,
            "semantics": HORIZON_DISCREPANCY_CALIBRATION_SEMANTICS,
            "source_group_ids": self.source_group_ids,
            "source_summary_sha256": self.source_summary_sha256,
            "horizon_steps": [int(value) for value in self.horizon_steps],
            "mean_reversion_half_life_steps": self.mean_reversion_half_life_steps,
            "minimum_mean_retention": self.minimum_mean_retention,
            "stationary_std_m": self.stationary_std_m.tolist(),
            "additional_process_std_m_per_sqrt_step": (
                self.additional_process_std_m_per_sqrt_step.tolist()
            ),
            "component_process_variance_scale": (self.component_process_variance_scale),
            "source_outcomes_used": self.source_outcomes_used,
            "interval_calibration_outcomes_used": (
                self.interval_calibration_outcomes_used
            ),
            "confirmation_outcomes_used": self.confirmation_outcomes_used,
            "target_outcomes_used": self.target_outcomes_used,
            "metadata": plain_json(self.metadata),
        }

    @property
    def artifact_id(self) -> str:
        return content_id(self.descriptor())

    def to_record(self) -> dict[str, object]:
        return {"artifact_id": self.artifact_id, **self.descriptor()}

    @classmethod
    def from_mapping(cls, value: object) -> HorizonDiscrepancyCalibrationV1:
        if not isinstance(value, Mapping):
            raise ValueError("horizon discrepancy calibration must be a JSON object")
        require_exact_fields(
            value, expected=CALIBRATION_FIELDS, name="horizon discrepancy calibration"
        )
        if value["schema"] != HORIZON_DISCREPANCY_CALIBRATION_SCHEMA:
            raise ValueError("unsupported horizon discrepancy calibration schema")
        version = genuine_integer(
            value["schema_version"], name="schema_version", minimum=1
        )
        if version != HORIZON_DISCREPANCY_CALIBRATION_VERSION:
            raise ValueError("unsupported horizon discrepancy calibration version")
        if value["semantics"] != HORIZON_DISCREPANCY_CALIBRATION_SEMANTICS:
            raise ValueError("horizon discrepancy calibration semantics changed")
        result = cls(
            source_group_ids=cast(Sequence[str], value["source_group_ids"]),
            source_summary_sha256=cast(str, value["source_summary_sha256"]),
            horizon_steps=cast(Sequence[int], value["horizon_steps"]),
            mean_reversion_half_life_steps=cast(
                float | None, value["mean_reversion_half_life_steps"]
            ),
            minimum_mean_retention=cast(float, value["minimum_mean_retention"]),
            stationary_std_m=np.asarray(value["stationary_std_m"]),
            additional_process_std_m_per_sqrt_step=np.asarray(
                value["additional_process_std_m_per_sqrt_step"]
            ),
            component_process_variance_scale=cast(
                float, value["component_process_variance_scale"]
            ),
            source_outcomes_used=cast(bool, value["source_outcomes_used"]),
            interval_calibration_outcomes_used=cast(
                bool, value["interval_calibration_outcomes_used"]
            ),
            confirmation_outcomes_used=cast(bool, value["confirmation_outcomes_used"]),
            target_outcomes_used=cast(bool, value["target_outcomes_used"]),
            metadata=cast(Mapping[str, Any], value["metadata"]),
        )
        declared = sha256_digest(value["artifact_id"], name="artifact_id")
        if declared != result.artifact_id:
            raise ValueError("horizon discrepancy calibration artifact_id changed")
        return result


@dataclass(frozen=True, slots=True)
class HorizonConditionedEndpointPredictionV1:
    """Immutable horizon-conditioned endpoint moments and diagnostics."""

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
        mean = np.asarray(self.mean_m, dtype=np.float64)
        covariance = np.asarray(self.covariance_m2, dtype=np.float64)
        weights = np.asarray(self.component_weights, dtype=np.float64)
        component_mean = np.asarray(self.component_mean_m, dtype=np.float64)
        component_variance = np.asarray(self.component_variance_m2, dtype=np.float64)
        additional = np.asarray(self.additional_axis_variance_m2, dtype=np.float64)
        horizon = genuine_integer(self.horizon_steps, name="horizon_steps", minimum=0)
        retention = probability(self.mean_retention, name="mean_retention")
        calibration_id = sha256_digest(self.calibration_id, name="calibration_id")
        if mean.ndim != 2 or mean.shape[1] != 3 or len(mean) < 1:
            raise ValueError("mean_m must have shape (N>=1, 3)")
        if covariance.shape != (len(mean), 3, 3):
            raise ValueError("covariance_m2 must have shape (N, 3, 3)")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
            raise ValueError("prediction moments must be finite")
        if not np.allclose(covariance, covariance.transpose(0, 2, 1)):
            raise ValueError("covariance_m2 must be symmetric")
        if np.min(np.linalg.eigvalsh(covariance), initial=0.0) < -1e-12:
            raise ValueError("covariance_m2 must be positive semidefinite")
        if weights.ndim != 2 or weights.shape[0] != len(mean):
            raise ValueError("component_weights shape changed")
        if np.any(weights < 0) or not np.allclose(np.sum(weights, axis=1), 1.0):
            raise ValueError("component_weights must be row-normalized")
        component_count = weights.shape[1]
        if component_mean.shape != (component_count, len(mean), 3):
            raise ValueError("component_mean_m shape changed")
        if component_variance.shape != (component_count, len(mean)):
            raise ValueError("component_variance_m2 shape changed")
        if additional.shape != (3,) or np.any(additional < 0):
            raise ValueError("additional_axis_variance_m2 must be nonnegative length 3")
        if not all(
            np.all(np.isfinite(value))
            for value in (weights, component_mean, component_variance, additional)
        ):
            raise ValueError("prediction component values must be finite")
        if np.any(component_variance < 0):
            raise ValueError("component_variance_m2 must be nonnegative")
        object.__setattr__(self, "mean_m", readonly(mean))
        object.__setattr__(self, "covariance_m2", readonly(covariance))
        object.__setattr__(self, "component_weights", readonly(weights))
        object.__setattr__(self, "component_mean_m", readonly(component_mean))
        object.__setattr__(self, "component_variance_m2", readonly(component_variance))
        object.__setattr__(self, "additional_axis_variance_m2", readonly(additional))
        object.__setattr__(self, "horizon_steps", horizon)
        object.__setattr__(self, "mean_retention", retention)
        object.__setattr__(self, "calibration_id", calibration_id)


def save_horizon_discrepancy_calibration(
    calibration: HorizonDiscrepancyCalibrationV1,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(calibration, HorizonDiscrepancyCalibrationV1):
        raise TypeError("calibration must be a HorizonDiscrepancyCalibrationV1")
    write_atomic_json(calibration.to_record(), path, overwrite=overwrite)


def load_horizon_discrepancy_calibration(
    path: str | Path,
) -> HorizonDiscrepancyCalibrationV1:
    return HorizonDiscrepancyCalibrationV1.from_mapping(
        load_strict_json_object(path, label="horizon discrepancy calibration")
    )
