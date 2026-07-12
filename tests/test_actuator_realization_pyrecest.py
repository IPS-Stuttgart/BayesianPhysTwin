from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyrecest")

from causal4d.actuator_realization import (
    calibrate_actuator_npz,
    fit_actuator_realization_calibration,
)


def _trajectory(times_s: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            0.04 * np.sin(2.0 * np.pi * 0.7 * times_s),
            0.03 * np.cos(2.0 * np.pi * 0.4 * times_s),
            0.02 * times_s,
        ]
    )


def test_pyrecest_actuator_calibration_recovers_delay_and_reduces_bias() -> None:
    times = np.linspace(0.0, 2.0, 401)
    delay_s = 0.035
    command = _trajectory(times)
    measured = 1.04 * _trajectory(times - delay_s) + np.array([0.003, -0.002, 0.001])

    result = fit_actuator_realization_calibration(
        times,
        command,
        times,
        measured,
        execution_id="dry-run-1",
        minimum_offset_s=-0.080,
        maximum_offset_s=0.080,
        offset_step_s=0.001,
        maximum_time_delta_s=0.006,
    )

    assert result["timestamp_alignment"][
        "estimated_positive_actuation_delay_s"
    ] == pytest.approx(delay_s, abs=0.003)
    assert (
        result["fit_metrics"]["bias_corrected_coordinate_rmse_m"]
        < result["fit_metrics"]["raw_coordinate_rmse_m"]
    )
    assert result["information_boundary"]["target_outcomes_used"] is False


def test_actuator_npz_artifact_includes_source_checksum(tmp_path) -> None:
    times = np.linspace(0.0, 1.0, 101)
    input_path = tmp_path / "actuator.npz"
    output_path = tmp_path / "calibration.json"
    np.savez(
        input_path,
        command_times_s=times,
        command_positions_m=_trajectory(times),
        measured_times_s=times,
        measured_positions_m=_trajectory(times - 0.010),
    )

    result = calibrate_actuator_npz(
        input_path,
        output_path,
        execution_id="dry-run-checksum",
        minimum_offset_s=-0.030,
        maximum_offset_s=0.030,
        offset_step_s=0.001,
        maximum_time_delta_s=0.011,
    )

    assert result["source_npz"]["sha256"]
    assert len(result["artifact_id"]) == 64
    assert output_path.is_file()
