"""Independent Causal4D observation clock-prior consumption.

A compact Gaussian payload cannot prove that its mean and standard deviation
belong to the content-addressed Causal4D artifact named by the payload. This
module reconstructs the complete source-only prior record, recomputes its
predictive summary and content identity, and only then exposes the scalar timing
prior used by BayesianPhysTwin.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._canonical_contracts import genuine_integer
from ._portable_contracts import (
    content_id,
    exact_revision,
    nonempty_string,
    require_exact_fields,
    sha256_digest,
)
from .observation_timing_interchange import (
    OBSERVATION_TIME_CORRECTION_CONVENTION,
)
from .observation_timing_nuisance import ObservationTimingPrior

CAUSAL4D_OBSERVATION_CLOCK_OFFSET_PRIOR_SCHEMA = (
    "causal4d.observation-clock-offset-prior"
)
CAUSAL4D_OBSERVATION_CLOCK_OFFSET_PRIOR_VERSION = 1

_INFORMATION_BOUNDARY: dict[str, bool] = {
    "source_or_dry_run_only": True,
    "target_outcomes_used": False,
    "hardware_timestamps_authoritative": True,
    "equal_weight_per_execution": True,
}
_CLAIM_BOUNDARY = (
    "This predictive timing prior does not identify contact slip, material "
    "relaxation, controller-frame physics, or downstream physical-query benefit."
)
_RECORD_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "artifact_id",
        "clock_domain",
        "reference_clock_domain",
        "time_scale",
        "offset_convention",
        "source_revision",
        "source_artifact_ids",
        "execution_ids",
        "source_offsets_s",
        "source_group_count",
        "mean_offset_s",
        "sample_standard_deviation_s",
        "grid_quantization_standard_deviation_s",
        "minimum_predictive_standard_deviation_s",
        "predictive_standard_deviation_s",
        "information_boundary",
        "claim_boundary",
    }
)


def _canonical_string(value: object, *, name: str) -> str:
    result = nonempty_string(value, name=name)
    if result != result.strip():
        raise ValueError(f"{name} must not contain surrounding whitespace")
    return result


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite")
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be finite")
    result = float(array.item())
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _sequence(value: object, *, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    return value


def _ordered_unique_strings(value: object, *, name: str) -> tuple[str, ...]:
    values = _sequence(value, name=name)
    result = tuple(
        _canonical_string(item, name=f"{name} entry") for item in values
    )
    if not result:
        raise ValueError(f"{name} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    return result


def _ordered_sha256s(value: object, *, name: str) -> tuple[str, ...]:
    values = _sequence(value, name=name)
    result = tuple(
        sha256_digest(item, name=f"{name} entry") for item in values
    )
    if not result:
        raise ValueError(f"{name} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    return result


def _ordered_finite_floats(value: object, *, name: str) -> tuple[float, ...]:
    values = _sequence(value, name=name)
    result = tuple(_finite_float(item, name=f"{name} entry") for item in values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _predictive_summary(
    source_offsets_s: tuple[float, ...],
    *,
    grid_standard_deviation_s: float,
    minimum_standard_deviation_s: float,
) -> tuple[float, float, float]:
    values = np.asarray(source_offsets_s, dtype=np.float64)
    if len(values) < 3:
        raise ValueError("at least three source executions are required")
    mean = float(np.mean(values))
    sample_standard_deviation = float(np.std(values, ddof=1))
    predictive_variance = (
        1.0 + 1.0 / len(values)
    ) * sample_standard_deviation**2 + grid_standard_deviation_s**2
    predictive_standard_deviation = max(
        minimum_standard_deviation_s,
        math.sqrt(max(0.0, predictive_variance)),
    )
    return mean, sample_standard_deviation, predictive_standard_deviation


@dataclass(frozen=True, slots=True)
class Causal4DObservationClockOffsetPriorV1:
    """Independent reconstruction of one Causal4D predictive timing prior."""

    clock_domain: str
    reference_clock_domain: str
    time_scale: str
    source_revision: str
    source_artifact_ids: Sequence[str]
    execution_ids: Sequence[str]
    source_offsets_s: Sequence[float]
    source_group_count: int
    mean_offset_s: float
    sample_standard_deviation_s: float
    grid_quantization_standard_deviation_s: float
    minimum_predictive_standard_deviation_s: float
    predictive_standard_deviation_s: float
    offset_convention: str = OBSERVATION_TIME_CORRECTION_CONVENTION
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        clock_domain = _canonical_string(self.clock_domain, name="clock_domain")
        reference_clock_domain = _canonical_string(
            self.reference_clock_domain,
            name="reference_clock_domain",
        )
        if clock_domain == reference_clock_domain:
            raise ValueError("clock and reference clock domains must differ")
        time_scale = _canonical_string(self.time_scale, name="time_scale")
        source_revision = exact_revision(
            self.source_revision,
            name="source_revision",
        )
        source_artifact_ids = _ordered_sha256s(
            self.source_artifact_ids,
            name="source_artifact_ids",
        )
        execution_ids = _ordered_unique_strings(
            self.execution_ids,
            name="execution_ids",
        )
        source_offsets = _ordered_finite_floats(
            self.source_offsets_s,
            name="source_offsets_s",
        )
        if len(source_artifact_ids) != len(execution_ids) or len(
            execution_ids
        ) != len(source_offsets):
            raise ValueError("source timing evidence counts differ")
        if execution_ids != tuple(sorted(execution_ids)):
            raise ValueError("execution IDs must use deterministic sorted order")
        source_group_count = genuine_integer(
            self.source_group_count,
            name="source_group_count",
            minimum=3,
        )
        if source_group_count != len(execution_ids):
            raise ValueError(
                "source_group_count must equal at least three executions"
            )
        if self.offset_convention != OBSERVATION_TIME_CORRECTION_CONVENTION:
            raise ValueError("observation time-correction convention changed")
        grid_standard_deviation = _finite_float(
            self.grid_quantization_standard_deviation_s,
            name="grid_quantization_standard_deviation_s",
        )
        minimum_standard_deviation = _finite_float(
            self.minimum_predictive_standard_deviation_s,
            name="minimum_predictive_standard_deviation_s",
        )
        if grid_standard_deviation <= 0.0:
            raise ValueError("grid quantization standard deviation must be positive")
        if minimum_standard_deviation <= 0.0:
            raise ValueError("minimum predictive standard deviation must be positive")
        expected_summary = _predictive_summary(
            source_offsets,
            grid_standard_deviation_s=grid_standard_deviation,
            minimum_standard_deviation_s=minimum_standard_deviation,
        )
        supplied_summary = (
            _finite_float(self.mean_offset_s, name="mean_offset_s"),
            _finite_float(
                self.sample_standard_deviation_s,
                name="sample_standard_deviation_s",
            ),
            _finite_float(
                self.predictive_standard_deviation_s,
                name="predictive_standard_deviation_s",
            ),
        )
        if not all(
            np.isclose(actual, expected, rtol=1e-13, atol=1e-15)
            for actual, expected in zip(
                supplied_summary,
                expected_summary,
                strict=True,
            )
        ):
            raise ValueError(
                "Causal4D clock-offset prior summary does not match source offsets"
            )
        if supplied_summary[2] < minimum_standard_deviation:
            raise ValueError(
                "predictive standard deviation is below its declared floor"
            )

        object.__setattr__(self, "clock_domain", clock_domain)
        object.__setattr__(
            self,
            "reference_clock_domain",
            reference_clock_domain,
        )
        object.__setattr__(self, "time_scale", time_scale)
        object.__setattr__(self, "source_revision", source_revision)
        object.__setattr__(self, "source_artifact_ids", source_artifact_ids)
        object.__setattr__(self, "execution_ids", execution_ids)
        object.__setattr__(self, "source_offsets_s", source_offsets)
        object.__setattr__(self, "source_group_count", source_group_count)
        object.__setattr__(self, "mean_offset_s", supplied_summary[0])
        object.__setattr__(
            self,
            "sample_standard_deviation_s",
            supplied_summary[1],
        )
        object.__setattr__(
            self,
            "grid_quantization_standard_deviation_s",
            grid_standard_deviation,
        )
        object.__setattr__(
            self,
            "minimum_predictive_standard_deviation_s",
            minimum_standard_deviation,
        )
        object.__setattr__(
            self,
            "predictive_standard_deviation_s",
            supplied_summary[2],
        )

        expected_id = content_id(self.identity_record())
        supplied_id = self.artifact_id
        if supplied_id is not None:
            supplied_id = sha256_digest(supplied_id, name="artifact_id")
            if supplied_id != expected_id:
                raise ValueError(
                    "Causal4D observation clock-offset prior artifact ID mismatch"
                )
        object.__setattr__(self, "artifact_id", expected_id)

    def identity_record(self) -> dict[str, Any]:
        return {
            "schema": CAUSAL4D_OBSERVATION_CLOCK_OFFSET_PRIOR_SCHEMA,
            "schema_version": CAUSAL4D_OBSERVATION_CLOCK_OFFSET_PRIOR_VERSION,
            "clock_domain": self.clock_domain,
            "reference_clock_domain": self.reference_clock_domain,
            "time_scale": self.time_scale,
            "offset_convention": self.offset_convention,
            "source_revision": self.source_revision,
            "source_artifact_ids": list(self.source_artifact_ids),
            "execution_ids": list(self.execution_ids),
            "source_offsets_s": list(self.source_offsets_s),
            "source_group_count": self.source_group_count,
            "mean_offset_s": self.mean_offset_s,
            "sample_standard_deviation_s": self.sample_standard_deviation_s,
            "grid_quantization_standard_deviation_s": (
                self.grid_quantization_standard_deviation_s
            ),
            "minimum_predictive_standard_deviation_s": (
                self.minimum_predictive_standard_deviation_s
            ),
            "predictive_standard_deviation_s": (
                self.predictive_standard_deviation_s
            ),
            "information_boundary": dict(_INFORMATION_BOUNDARY),
            "claim_boundary": _CLAIM_BOUNDARY,
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> Causal4DObservationClockOffsetPriorV1:
        require_exact_fields(
            value,
            expected=_RECORD_FIELDS,
            name="Causal4D observation clock-offset prior",
        )
        if value["schema"] != CAUSAL4D_OBSERVATION_CLOCK_OFFSET_PRIOR_SCHEMA:
            raise ValueError("unsupported Causal4D clock-offset prior schema")
        version = genuine_integer(
            value["schema_version"],
            name="schema_version",
            minimum=1,
        )
        if version != CAUSAL4D_OBSERVATION_CLOCK_OFFSET_PRIOR_VERSION:
            raise ValueError("unsupported Causal4D clock-offset prior version")
        if value["information_boundary"] != _INFORMATION_BOUNDARY:
            raise ValueError("Causal4D clock-offset information boundary changed")
        if value["claim_boundary"] != _CLAIM_BOUNDARY:
            raise ValueError("Causal4D clock-offset claim boundary changed")
        return cls(
            clock_domain=value["clock_domain"],
            reference_clock_domain=value["reference_clock_domain"],
            time_scale=value["time_scale"],
            source_revision=value["source_revision"],
            source_artifact_ids=value["source_artifact_ids"],
            execution_ids=value["execution_ids"],
            source_offsets_s=value["source_offsets_s"],
            source_group_count=value["source_group_count"],
            mean_offset_s=value["mean_offset_s"],
            sample_standard_deviation_s=value[
                "sample_standard_deviation_s"
            ],
            grid_quantization_standard_deviation_s=value[
                "grid_quantization_standard_deviation_s"
            ],
            minimum_predictive_standard_deviation_s=value[
                "minimum_predictive_standard_deviation_s"
            ],
            predictive_standard_deviation_s=value[
                "predictive_standard_deviation_s"
            ],
            offset_convention=value["offset_convention"],
            artifact_id=value["artifact_id"],
        )

    def observation_timing_prior(self) -> ObservationTimingPrior:
        artifact_id = self.artifact_id
        if artifact_id is None:
            raise AssertionError("validated Causal4D prior lacks an artifact ID")
        return ObservationTimingPrior(
            clock_domain=self.clock_domain,
            mean_offset_s=self.mean_offset_s,
            standard_deviation_s=self.predictive_standard_deviation_s,
            source_artifact_id=artifact_id,
        )


def causal4d_observation_timing_prior_from_record(
    value: Mapping[str, Any],
    *,
    expected_artifact_id: str,
    expected_clock_domain: str,
    expected_time_scale: str,
) -> ObservationTimingPrior:
    """Reconstruct and bind one full Causal4D timing-prior record."""

    expected_id = sha256_digest(
        expected_artifact_id,
        name="expected_artifact_id",
    )
    clock_domain = _canonical_string(
        expected_clock_domain,
        name="expected_clock_domain",
    )
    time_scale = _canonical_string(
        expected_time_scale,
        name="expected_time_scale",
    )
    prior = Causal4DObservationClockOffsetPriorV1.from_mapping(value)
    if prior.artifact_id != expected_id:
        raise ValueError("Causal4D clock prior artifact ID differs from lineage")
    if prior.clock_domain != clock_domain:
        raise ValueError("Causal4D clock prior domain differs from lineage")
    if prior.time_scale != time_scale:
        raise ValueError("Causal4D clock prior time scale differs from lineage")
    return prior.observation_timing_prior()


__all__ = [
    "CAUSAL4D_OBSERVATION_CLOCK_OFFSET_PRIOR_SCHEMA",
    "CAUSAL4D_OBSERVATION_CLOCK_OFFSET_PRIOR_VERSION",
    "Causal4DObservationClockOffsetPriorV1",
    "causal4d_observation_timing_prior_from_record",
]
