from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from experiments.tracking_cloth_action_feasibility_v1 import _data as action_data
from experiments.tracking_cloth_self_collision_selective_twin_v1.data import (
    Case,
)


def _protocol() -> dict[str, Any]:
    return {
        "source_repetitions": [1, 2],
        "initial_complete_frame_deadline_seconds": 0.05,
        "prefix_seconds": 0.1,
        "forecast_seconds": 0.2,
        "analysis_sample_period_seconds": 0.05,
    }


def _points(time: float) -> np.ndarray:
    x, y = np.meshgrid(np.linspace(0.0, 0.3, 4), np.linspace(0.0, 0.4, 5))
    cloth = np.column_stack(
        [
            x.ravel(),
            y.ravel(),
            np.full(20, 0.01 * time),
        ]
    )
    rod = np.array([[-0.5, 0.0, 0.0], [0.9, 0.0, 0.0]])
    return np.vstack([cloth, rod])


def test_source_trajectory_uses_recorded_irregular_timestamps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded_times = [
        0.0,
        0.009,
        0.018,
        0.035,
        0.052,
        0.078,
        0.102,
        0.139,
        0.171,
        0.208,
        0.247,
        0.281,
        0.322,
    ]

    def fake_rows(path: Path):
        del path
        for frame, time in enumerate(recorded_times):
            yield float(frame), time, [str(time)], 22

    def fake_positions(cells: list[str], indices: np.ndarray) -> np.ndarray:
        return _points(float(cells[0]))[np.asarray(indices, dtype=np.int64)]

    def fake_partition(initial: np.ndarray):
        del initial
        return (
            np.arange(20, dtype=np.int64),
            np.arange(20, dtype=np.int64),
            np.array([20, 21], dtype=np.int64),
            1.0,
        )

    monkeypatch.setattr(action_data, "_row_stream", fake_rows)
    monkeypatch.setattr(action_data, "_positions", fake_positions)
    monkeypatch.setattr(action_data, "_partition_markers", fake_partition)

    case = Case(
        path=Path("cotton_a2_four_corners_normal_rep1.csv"),
        material="cotton",
        interaction="four_corners_normal",
        repetition=1,
    )
    view = action_data.source_trajectory(case, _protocol())

    assert view.times.tolist() == pytest.approx([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    assert view.cutoff == 2
    assert np.diff(view.times).tolist() == pytest.approx([0.05] * 6)
    assert view.native_dt_max_seconds > 0.03
    assert view.native_dt_median_seconds != pytest.approx(1.0 / 120.0)
    assert view.cloth.shape == (7, 20, 3)
    assert np.isfinite(view.cloth).all()


def test_reserved_target_is_rejected_before_numeric_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened = False

    def forbidden_rows(path: Path):
        nonlocal opened
        del path
        opened = True
        return iter(())

    monkeypatch.setattr(action_data, "_row_stream", forbidden_rows)
    case = Case(
        path=Path("cotton_a2_four_corners_normal_rep3.csv"),
        material="cotton",
        interaction="four_corners_normal",
        repetition=3,
    )

    with pytest.raises(ValueError, match="restricted to source repetitions"):
        action_data.source_trajectory(case, _protocol())
    assert not opened


def test_analysis_grid_rejects_nonintegral_horizon() -> None:
    protocol = _protocol()
    protocol["analysis_sample_period_seconds"] = 0.07

    with pytest.raises(ValueError, match="integer multiple"):
        action_data._analysis_grid(
            0.0,
            prefix_seconds=float(protocol["prefix_seconds"]),
            forecast_seconds=float(protocol["forecast_seconds"]),
            period_seconds=float(protocol["analysis_sample_period_seconds"]),
        )
