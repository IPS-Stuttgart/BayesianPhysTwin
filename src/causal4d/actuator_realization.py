"""Source-only actuator synchronization and bias diagnostics using PyRecEst."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ACTUATOR_REALIZATION_SCHEMA_VERSION = 1
LOCKED_PYRECEST_VERSION = "2.4.1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_id(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("artifact_id", None)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _finite_or_none(values: np.ndarray) -> list[float | None]:
    return [float(value) if np.isfinite(value) else None for value in values]


def _load_pyrecest_calibration() -> tuple[Any, ...]:
    try:
        import pyrecest
        from pyrecest.calibration import (
            apply_time_offset,
            fit_sensor_bias_correction,
            fit_time_offset,
            interpolate_reference_values,
            make_offset_grid,
        )
    except ImportError as error:
        raise RuntimeError(
            "PyRecEst 2.4.1 is required; install the 'pyrecest' optional extra "
            "in a Python 3.11 or newer environment"
        ) from error
    if pyrecest.__version__ != LOCKED_PYRECEST_VERSION:
        raise RuntimeError(
            f"PyRecEst {LOCKED_PYRECEST_VERSION} is locked, found "
            f"{pyrecest.__version__}"
        )
    return (
        apply_time_offset,
        fit_sensor_bias_correction,
        fit_time_offset,
        interpolate_reference_values,
        make_offset_grid,
    )


def _trajectory(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError(f"{name} must have shape (sample, 3)")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _times(values: np.ndarray, count: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float).reshape(-1)
    if len(result) != count or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain one finite timestamp per sample")
    if np.any(np.diff(result) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return result


def fit_actuator_realization_calibration(
    command_times_s: np.ndarray,
    command_positions_m: np.ndarray,
    measured_times_s: np.ndarray,
    measured_positions_m: np.ndarray,
    *,
    execution_id: str,
    minimum_offset_s: float = -0.150,
    maximum_offset_s: float = 0.150,
    offset_step_s: float = 0.001,
    maximum_time_delta_s: float = 0.010,
) -> dict[str, Any]:
    """Fit source-only timing and affine-bias diagnostics for one execution.

    The returned time offset follows PyRecEst's convention: it is added to each
    measured timestamp before querying the command trajectory. Consequently,
    positive physical actuation delay is reported as the negative fitted offset.
    """

    if not execution_id:
        raise ValueError("execution_id must be nonempty")
    command_positions = _trajectory(command_positions_m, "command_positions_m")
    measured_positions = _trajectory(measured_positions_m, "measured_positions_m")
    command_times = _times(command_times_s, len(command_positions), "command_times_s")
    measured_times = _times(
        measured_times_s, len(measured_positions), "measured_times_s"
    )
    (
        apply_time_offset,
        fit_sensor_bias_correction,
        fit_time_offset,
        interpolate_reference_values,
        make_offset_grid,
    ) = _load_pyrecest_calibration()
    offsets = make_offset_grid(
        minimum_offset_s,
        maximum_offset_s,
        offset_step_s,
    )
    # Remove constant frame translation before timing estimation so spatial bias
    # is not misreported as latency. Gain remains visible but does not move phase.
    centered_measured = measured_positions - np.mean(measured_positions, axis=0)
    centered_command = command_positions - np.mean(command_positions, axis=0)
    offset_fit = fit_time_offset(
        measured_times,
        centered_measured,
        command_times,
        centered_command,
        offsets,
        metric="rmse",
        max_time_delta_s=maximum_time_delta_s,
        metadata={"execution_id": execution_id},
    )
    if offset_fit.best_offset_s is None:
        raise ValueError("no timestamp offset has valid trajectory overlap")
    aligned_times = apply_time_offset(measured_times, offset_fit.best_offset_s)
    command_at_measurement, valid = interpolate_reference_values(
        command_times,
        command_positions,
        aligned_times,
        max_time_delta_s=maximum_time_delta_s,
    )
    valid &= np.all(np.isfinite(measured_positions), axis=1)
    if int(np.sum(valid)) < 4:
        raise ValueError("fewer than four aligned actuator samples are valid")
    aligned_measured = measured_positions[valid]
    aligned_command = command_at_measurement[valid]
    aligned_valid_times = aligned_times[valid]
    features = aligned_command - aligned_command[0]
    bias_model = fit_sensor_bias_correction(
        aligned_valid_times,
        aligned_measured,
        aligned_valid_times,
        aligned_command,
        feature_values=features,
        max_time_delta_s=0.0,
        ridge_alpha=1.0e-2,
        min_samples=4,
        metadata={
            "execution_id": execution_id,
            "role": "diagnostic_affine_residual_not_physical_frame_posterior",
        },
    )
    corrected = bias_model.apply(aligned_measured, features)
    raw_error = aligned_measured - aligned_command
    corrected_error = corrected - aligned_command
    raw_rmse = float(np.sqrt(np.mean(np.square(raw_error))))
    corrected_rmse = float(np.sqrt(np.mean(np.square(corrected_error))))
    offset_summary = offset_fit.summary()
    result: dict[str, Any] = {
        "schema_version": ACTUATOR_REALIZATION_SCHEMA_VERSION,
        "artifact_kind": "ActuatorRealizationCalibration",
        "execution_id": execution_id,
        "pyrecest_version": LOCKED_PYRECEST_VERSION,
        "timestamp_alignment": {
            "convention": "aligned_measurement_time_s = measurement_time_s + offset_s",
            "best_offset_s": float(offset_fit.best_offset_s),
            "estimated_positive_actuation_delay_s": float(-offset_fit.best_offset_s),
            "metric": offset_fit.metric,
            "best_metric_value_m": float(offset_summary["best_metric_value"]),
            "offset_grid_s": offset_fit.offsets_s.tolist(),
            "metric_values_m": _finite_or_none(offset_fit.metric_values),
            "valid_counts": offset_fit.counts.tolist(),
            "maximum_time_delta_s": float(maximum_time_delta_s),
        },
        "bias_diagnostic": bias_model.to_dict(),
        "fit_metrics": {
            "aligned_sample_count": int(np.sum(valid)),
            "aligned_sample_fraction": float(np.mean(valid)),
            "raw_coordinate_rmse_m": raw_rmse,
            "bias_corrected_coordinate_rmse_m": corrected_rmse,
            "rmse_reduction_fraction": (
                float(1.0 - corrected_rmse / raw_rmse) if raw_rmse > 0.0 else 0.0
            ),
        },
        "information_boundary": {
            "source_or_dry_run_only": True,
            "target_outcomes_used": False,
            "hardware_timestamps_authoritative": True,
            "bias_model_is_not_a_physical_frame_or_gain_posterior": True,
        },
        "claim_boundary": (
            "PyRecEst timing and bias fits are synchronization diagnostics. They "
            "do not identify contact slip, material lag, or controller-frame physics."
        ),
    }
    result["artifact_id"] = _artifact_id(result)
    return result


def calibrate_actuator_npz(
    input_npz: str | Path,
    output_json: str | Path,
    *,
    execution_id: str,
    minimum_offset_s: float = -0.150,
    maximum_offset_s: float = 0.150,
    offset_step_s: float = 0.001,
    maximum_time_delta_s: float = 0.010,
) -> dict[str, Any]:
    """Fit and serialize a checksummed actuator-realization diagnostic."""

    input_path = Path(input_npz)
    with np.load(input_path, allow_pickle=False) as archive:
        required = {
            "command_times_s",
            "command_positions_m",
            "measured_times_s",
            "measured_positions_m",
        }
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"actuator NPZ is missing arrays: {sorted(missing)}")
        result = fit_actuator_realization_calibration(
            archive["command_times_s"],
            archive["command_positions_m"],
            archive["measured_times_s"],
            archive["measured_positions_m"],
            execution_id=execution_id,
            minimum_offset_s=minimum_offset_s,
            maximum_offset_s=maximum_offset_s,
            offset_step_s=offset_step_s,
            maximum_time_delta_s=maximum_time_delta_s,
        )
    result["source_npz"] = {
        "path": str(input_path),
        "sha256": _file_sha256(input_path),
    }
    result["artifact_id"] = _artifact_id(result)
    output = Path(output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result
