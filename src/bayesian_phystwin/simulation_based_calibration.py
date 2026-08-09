"""Content-addressed simulation-based calibration diagnostics.

The statistical units are independent simulation replicates. Frames, views,
tracks, points, particles, and taxels are repeated observations within one
replicate and must not be presented as extra groups. These diagnostics test an
implemented inference chain under its own generator; they do not establish
calibration or accuracy on real physical data.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._canonical_contracts import (
    canonical_string_tuple,
    frozen_finite_json_mapping,
    genuine_integer,
    immutable_array,
    immutable_integer_array,
    literal_lower_hex,
    plain_json,
)
from ._portable_contracts import content_id
from ._simulation_based_calibration_core import (
    finite_float_array,
    posterior_pit_matrix,
    weighted_randomized_pit,
)

SIMULATION_BASED_CALIBRATION_SCHEMA = (
    "bayesian_phystwin.simulation_based_calibration_summary"
)
SIMULATION_BASED_CALIBRATION_VERSION = 1


def _array_record(values: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(values)
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _diagnostics(
    pit_values: np.ndarray,
    *,
    bin_count: int,
) -> tuple[np.ndarray, ...]:
    replicate_count, parameter_count = pit_values.shape
    histogram = np.zeros((parameter_count, bin_count), dtype=np.int64)
    mean = np.empty(parameter_count, dtype=np.float64)
    ks = np.empty(parameter_count, dtype=np.float64)
    cramer_von_mises = np.empty(parameter_count, dtype=np.float64)
    central_50 = np.empty(parameter_count, dtype=np.float64)
    central_90 = np.empty(parameter_count, dtype=np.float64)
    central_95 = np.empty(parameter_count, dtype=np.float64)
    tail_5 = np.empty((parameter_count, 2), dtype=np.float64)
    positions = (2.0 * np.arange(1, replicate_count + 1, dtype=np.float64) - 1.0) / (
        2.0 * replicate_count
    )

    for parameter in range(parameter_count):
        values = pit_values[:, parameter]
        histogram[parameter], _ = np.histogram(
            values,
            bins=bin_count,
            range=(0.0, 1.0),
        )
        ordered = np.sort(values)
        upper: np.ndarray = np.arange(
            1,
            replicate_count + 1,
            dtype=np.float64,
        )
        lower: np.ndarray = np.arange(replicate_count, dtype=np.float64)
        mean[parameter] = float(np.mean(values))
        ks[parameter] = float(
            max(
                np.max(upper / replicate_count - ordered),
                np.max(ordered - lower / replicate_count),
            )
        )
        cramer_von_mises[parameter] = float(
            1.0 / (12.0 * replicate_count) + np.sum(np.square(ordered - positions))
        )
        central_50[parameter] = float(np.mean((values >= 0.25) & (values <= 0.75)))
        central_90[parameter] = float(np.mean((values >= 0.05) & (values <= 0.95)))
        central_95[parameter] = float(np.mean((values >= 0.025) & (values <= 0.975)))
        tail_5[parameter, 0] = float(np.mean(values < 0.05))
        tail_5[parameter, 1] = float(np.mean(values > 0.95))

    return (
        histogram,
        mean,
        ks,
        cramer_von_mises,
        central_50,
        central_90,
        central_95,
        tail_5,
    )


@dataclass(frozen=True, slots=True)
class SimulationBasedCalibrationSummaryV1:
    """PIT histograms and uniformity diagnostics for independent groups."""

    group_ids: Sequence[str]
    parameter_names: Sequence[str]
    pit_values: np.ndarray
    bin_count: int = 10
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None
    histogram_counts: np.ndarray = field(init=False, repr=False)
    mean_pit: np.ndarray = field(init=False)
    ks_distance: np.ndarray = field(init=False)
    cramer_von_mises: np.ndarray = field(init=False)
    central_50_coverage: np.ndarray = field(init=False)
    central_90_coverage: np.ndarray = field(init=False)
    central_95_coverage: np.ndarray = field(init=False)
    tail_5_rates: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        groups = canonical_string_tuple(
            self.group_ids,
            name="group_ids",
            allow_empty=False,
        )
        parameters = canonical_string_tuple(
            self.parameter_names,
            name="parameter_names",
            allow_empty=False,
        )
        if len(set(groups)) != len(groups):
            raise ValueError("group_ids must be unique independent units")
        if len(set(parameters)) != len(parameters):
            raise ValueError("parameter_names must be unique")
        values = finite_float_array(self.pit_values, name="pit_values", ndim=2)
        if values.shape != (len(groups), len(parameters)):
            raise ValueError("pit_values must match group and parameter counts")
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("pit_values must lie in [0, 1]")
        bins = genuine_integer(self.bin_count, name="bin_count", minimum=2)
        if bins > 10_000:
            raise ValueError("bin_count must not exceed 10000")
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="simulation-based calibration metadata",
        )

        diagnostics = _diagnostics(values, bin_count=bins)
        frozen_values = immutable_array(values, dtype=np.dtype("<f8"))
        histogram = immutable_integer_array(
            diagnostics[0],
            name="histogram_counts",
        )
        derived = tuple(
            immutable_array(array, dtype=np.dtype("<f8")) for array in diagnostics[1:]
        )
        object.__setattr__(self, "group_ids", groups)
        object.__setattr__(self, "parameter_names", parameters)
        object.__setattr__(self, "pit_values", frozen_values)
        object.__setattr__(self, "bin_count", bins)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "histogram_counts", histogram)
        object.__setattr__(self, "mean_pit", derived[0])
        object.__setattr__(self, "ks_distance", derived[1])
        object.__setattr__(self, "cramer_von_mises", derived[2])
        object.__setattr__(self, "central_50_coverage", derived[3])
        object.__setattr__(self, "central_90_coverage", derived[4])
        object.__setattr__(self, "central_95_coverage", derived[5])
        object.__setattr__(self, "tail_5_rates", derived[6])

        computed = content_id(self.descriptor())
        if self.artifact_id is not None:
            supplied = literal_lower_hex(
                self.artifact_id,
                name="artifact_id",
                lengths={64},
            )
            if supplied != computed:
                raise ValueError("artifact_id does not match summary content")
        object.__setattr__(self, "artifact_id", computed)

    @property
    def independent_group_count(self) -> int:
        return len(self.group_ids)

    @property
    def parameter_count(self) -> int:
        return len(self.parameter_names)

    @property
    def expected_histogram_count(self) -> float:
        return self.independent_group_count / self.bin_count

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": SIMULATION_BASED_CALIBRATION_SCHEMA,
            "schema_version": SIMULATION_BASED_CALIBRATION_VERSION,
            "group_ids": list(self.group_ids),
            "parameter_names": list(self.parameter_names),
            "bin_count": self.bin_count,
            "pit_values": _array_record(self.pit_values),
            "histogram_counts": _array_record(self.histogram_counts),
            "mean_pit": _array_record(self.mean_pit),
            "ks_distance": _array_record(self.ks_distance),
            "cramer_von_mises": _array_record(self.cramer_von_mises),
            "central_50_coverage": _array_record(self.central_50_coverage),
            "central_90_coverage": _array_record(self.central_90_coverage),
            "central_95_coverage": _array_record(self.central_95_coverage),
            "tail_5_rates": _array_record(self.tail_5_rates),
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {
            **self.descriptor(),
            "artifact_id": self.artifact_id,
            "independent_group_count": self.independent_group_count,
            "parameter_count": self.parameter_count,
            "expected_histogram_count": self.expected_histogram_count,
            "histogram_counts_values": self.histogram_counts.tolist(),
            "mean_pit_values": self.mean_pit.tolist(),
            "ks_distance_values": self.ks_distance.tolist(),
            "cramer_von_mises_values": self.cramer_von_mises.tolist(),
            "central_50_coverage_values": self.central_50_coverage.tolist(),
            "central_90_coverage_values": self.central_90_coverage.tolist(),
            "central_95_coverage_values": self.central_95_coverage.tolist(),
            "tail_5_rate_values": self.tail_5_rates.tolist(),
        }


__all__ = [
    "SIMULATION_BASED_CALIBRATION_SCHEMA",
    "SIMULATION_BASED_CALIBRATION_VERSION",
    "SimulationBasedCalibrationSummaryV1",
    "posterior_pit_matrix",
    "weighted_randomized_pit",
]
