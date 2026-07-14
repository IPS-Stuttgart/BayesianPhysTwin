from pathlib import Path

import pytest

from causal4d_public.deform360_replication_tactile import (
    select_causal_tactile_baseline,
    timestamp_from_path,
)


def test_timestamp_from_path_uses_terminal_integer() -> None:
    assert timestamp_from_path("sensor_1769566118603178.npy") == 1769566118603178
    with pytest.raises(ValueError, match="terminal timestamp"):
        timestamp_from_path("sensor.npy")


def test_baseline_selection_prefers_latest_causal_value(tmp_path: Path) -> None:
    recording = tmp_path / "sensor_1000000000200.npy"
    recording.touch()
    candidates = [
        tmp_path / name
        for name in (
            "median_1000000000100.npy",
            "median_1000000000190.npy",
            "median_1000000000220.npy",
        )
    ]
    for candidate in candidates:
        candidate.touch()
    selected, diagnostic = select_causal_tactile_baseline(recording, candidates)
    assert selected == candidates[1].resolve()
    assert diagnostic["selection_rule"] == "latest-baseline-at-or-before-recording"
    assert diagnostic["signed_baseline_age"] == 10


def test_baseline_selection_falls_back_to_nearest(tmp_path: Path) -> None:
    recording = tmp_path / "sensor_1000000000100.npy"
    recording.touch()
    candidates = [
        tmp_path / name
        for name in ("median_1000000000120.npy", "median_1000000000160.npy")
    ]
    for candidate in candidates:
        candidate.touch()
    selected, diagnostic = select_causal_tactile_baseline(recording, candidates)
    assert selected == candidates[0].resolve()
    assert diagnostic["selection_rule"] == "nearest-baseline-when-no-causal-baseline-exists"
    assert diagnostic["signed_baseline_age"] == -20
