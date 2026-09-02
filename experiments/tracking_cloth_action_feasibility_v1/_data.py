"""Actual-timestamp source views for the Tracking Cloth action audit.

The publisher's nominal capture rate is metadata, not an exact timestamp grid.
This module uses the recorded monotone timestamps, causally fills isolated
missing marker observations after one complete initialization frame, and
resamples repetitions 1 and 2 onto the registered analysis grid. Repetition 3 is
rejected before numeric trajectory rows are opened.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

from experiments.tracking_cloth_action_feasibility_v1._metrics import (
    object_digest,
    physical_action_metrics,
)
from experiments.tracking_cloth_self_collision_selective_twin_v1.data import (
    Case,
    _partition_markers,
    _positions,
    _row_stream,
    audit_dataset,
)

FloatArray: TypeAlias = npt.NDArray[np.float64]


@dataclass(frozen=True)
class ActualTimeSourceView:
    """One source trajectory on the registered actual-time analysis grid."""

    times: FloatArray
    cloth: FloatArray
    cutoff: int
    initial_diameter_m: float
    missing_coordinate_fraction_before_carry: float
    native_dt_median_seconds: float
    native_dt_min_seconds: float
    native_dt_max_seconds: float


def _positive_float(value: object, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive finite number") from error
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _causal_fill(values: FloatArray) -> tuple[FloatArray, float]:
    if values.ndim != 3 or values.shape[1:] != (20, 3):
        raise ValueError("cloth samples must have shape (time, 20, 3)")
    finite_marker = np.isfinite(values).all(axis=2)
    missing_fraction = float(1.0 - np.mean(finite_marker))
    output = np.array(values, dtype=np.float64, copy=True)
    if not finite_marker[0].all():
        raise ValueError("the selected initialization sample must be complete")
    for index in range(1, output.shape[0]):
        missing = ~finite_marker[index]
        output[index, missing] = output[index - 1, missing]
    if not np.isfinite(output).all():
        raise ValueError("source trajectory remains nonfinite after causal carry")
    return output, missing_fraction


def _analysis_grid(
    start_time: float,
    *,
    prefix_seconds: float,
    forecast_seconds: float,
    period_seconds: float,
) -> tuple[FloatArray, int]:
    duration = prefix_seconds + forecast_seconds
    interval_count = int(round(duration / period_seconds))
    if interval_count < 2 or not np.isclose(
        interval_count * period_seconds,
        duration,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError(
            "prefix plus forecast must be an integer multiple of the "
            "analysis period"
        )
    cutoff = int(round(prefix_seconds / period_seconds))
    if cutoff < 1 or not np.isclose(
        cutoff * period_seconds,
        prefix_seconds,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("prefix must be an integer multiple of the analysis period")
    times = start_time + period_seconds * np.arange(
        interval_count + 1, dtype=np.float64
    )
    times.setflags(write=False)
    return times, cutoff


def _linear_resample(
    source_times: FloatArray,
    source_values: FloatArray,
    target_times: FloatArray,
) -> FloatArray:
    if source_times.ndim != 1 or source_values.shape[0] != source_times.size:
        raise ValueError("source timestamps and cloth samples are inconsistent")
    flat = source_values.reshape(source_times.size, -1)
    output = np.empty((target_times.size, flat.shape[1]), dtype=np.float64)
    for column in range(flat.shape[1]):
        output[:, column] = np.interp(
            target_times,
            source_times,
            flat[:, column],
        )
    result = output.reshape(target_times.size, 20, 3)
    result.setflags(write=False)
    return result


def source_trajectory(
    case: Case,
    protocol: dict[str, Any],
) -> ActualTimeSourceView:
    """Read one authorized source trajectory using its recorded timestamps."""

    allowed = {int(value) for value in protocol["source_repetitions"]}
    if case.repetition not in allowed:
        raise ValueError("numeric trajectory access is restricted to source repetitions")

    rows = list(_row_stream(case.path))
    marker_count = rows[0][3]
    first_time = rows[0][1]
    deadline = first_time + _positive_float(
        protocol["initial_complete_frame_deadline_seconds"],
        name="initial_complete_frame_deadline_seconds",
    )
    all_indices = np.arange(marker_count, dtype=np.int64)
    initial_time: float | None = None
    initial: FloatArray | None = None
    for _, time, cells, count in rows:
        if count != marker_count:
            raise ValueError("marker count changed within a recording")
        if time > deadline + 1e-9:
            break
        values = _positions(cells, all_indices)
        if np.isfinite(values).all():
            initial_time = float(time)
            initial = np.asarray(values, dtype=np.float64)
            break
    if initial_time is None or initial is None:
        raise ValueError(f"no complete initialization frame: {case.path.name}")

    cloth_indices, cloth_order, _, scale = _partition_markers(initial)
    prefix_seconds = _positive_float(
        protocol["prefix_seconds"],
        name="prefix_seconds",
    )
    forecast_seconds = _positive_float(
        protocol["forecast_seconds"],
        name="forecast_seconds",
    )
    period_seconds = _positive_float(
        protocol["analysis_sample_period_seconds"],
        name="analysis_sample_period_seconds",
    )
    target_times, cutoff = _analysis_grid(
        initial_time,
        prefix_seconds=prefix_seconds,
        forecast_seconds=forecast_seconds,
        period_seconds=period_seconds,
    )

    source_times: list[float] = []
    source_cloth: list[FloatArray] = []
    for _, time, cells, count in rows:
        if count != marker_count:
            raise ValueError("marker count changed within a recording")
        if time < initial_time - 1e-9:
            continue
        if time > target_times[-1] + 0.25:
            break
        source_times.append(float(time))
        cloth = _positions(cells, cloth_indices)[cloth_order] * scale
        source_cloth.append(np.asarray(cloth, dtype=np.float64))

    raw_times = np.asarray(source_times, dtype=np.float64)
    raw_cloth = np.asarray(source_cloth, dtype=np.float64)
    if raw_times.size < 3 or raw_cloth.shape != (raw_times.size, 20, 3):
        raise ValueError("insufficient source trajectory samples")
    native_dt = np.diff(raw_times)
    if not np.all(np.isfinite(native_dt)) or np.any(native_dt <= 0.0):
        raise ValueError(
            "recorded source timestamps must be finite and increasing"
        )
    median_dt = float(np.median(native_dt))
    coverage_slack = max(2.0 * median_dt, 0.05)
    if raw_times[-1] < target_times[-1] - coverage_slack:
        raise ValueError("recording does not cover the registered analysis horizon")

    filled, missing_fraction = _causal_fill(raw_cloth)
    cloth = _linear_resample(raw_times, filled, target_times)
    initial_diameter = float(
        np.max(
            np.linalg.norm(
                cloth[0, :, None] - cloth[0, None, :],
                axis=2,
            )
        )
    )
    if not np.isfinite(initial_diameter) or initial_diameter <= 0.0:
        raise ValueError("initial cloth diameter is invalid")

    return ActualTimeSourceView(
        times=target_times,
        cloth=cloth,
        cutoff=cutoff,
        initial_diameter_m=initial_diameter,
        missing_coordinate_fraction_before_carry=missing_fraction,
        native_dt_median_seconds=median_dt,
        native_dt_min_seconds=float(np.min(native_dt)),
        native_dt_max_seconds=float(np.max(native_dt)),
    )


def source_rows(
    root: Path,
    protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build compact physical-action rows without numerically opening rep3."""

    cases, inventory = audit_dataset(root, protocol)
    allowed = {int(value) for value in protocol["source_repetitions"]}
    source_cases = [case for case in cases if case.repetition in allowed]
    expected_count = (
        len(protocol["materials"])
        * len(protocol["interactions"])
        * len(allowed)
    )
    if len(source_cases) != expected_count:
        raise ValueError("source roster is incomplete")

    rows: list[dict[str, Any]] = []
    for case in source_cases:
        view = source_trajectory(case, protocol)
        metrics = physical_action_metrics(
            view.cloth,
            cutoff=view.cutoff,
            contact_distance_m=float(protocol["self_contact_distance_m"]),
            edge_strain_weight=float(protocol["edge_strain_weight"]),
            edge_strain_quantile=float(protocol["edge_strain_quantile"]),
            initial_diameter_m=view.initial_diameter_m,
        )
        rows.append(
            {
                "case_id": case.case_id,
                "material": case.material,
                "interaction": case.interaction,
                "repetition": case.repetition,
                "native_dt_seconds": view.native_dt_median_seconds,
                "native_dt_min_seconds": view.native_dt_min_seconds,
                "native_dt_max_seconds": view.native_dt_max_seconds,
                "analysis_dt_seconds": float(
                    protocol["analysis_sample_period_seconds"]
                ),
                "sample_count": int(view.times.size),
                "prefix_sample_count": int(view.cutoff + 1),
                "missing_coordinate_fraction_before_carry": (
                    view.missing_coordinate_fraction_before_carry
                ),
                **metrics,
            }
        )
    rows.sort(
        key=lambda item: (
            item["material"],
            item["repetition"],
            item["interaction"],
        )
    )
    inventory = dict(inventory)
    inventory["numeric_source_repetitions_read"] = sorted(allowed)
    inventory["numeric_rep3_outcomes_read"] = False
    inventory["actual_timestamp_resampling"] = {
        "period_seconds": float(protocol["analysis_sample_period_seconds"]),
        "interpolation": "linear-after-causal-marker-carry",
        "nominal_capture_rate_assumed_exact": False,
    }
    inventory["inventory_id"] = object_digest(inventory)
    return rows, inventory


__all__ = [
    "ActualTimeSourceView",
    "source_rows",
    "source_trajectory",
]
