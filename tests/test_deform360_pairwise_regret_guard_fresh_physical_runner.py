from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/remote/run_deform360_pairwise_regret_guard_fresh_physical.py"


def _module():
    spec = importlib.util.spec_from_file_location("fresh_physical_runner", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _robot(path: Path, frame_count: int) -> None:
    np.savez_compressed(
        path,
        format_version=np.asarray(1, dtype=np.uint16),
        actions=np.zeros((frame_count, 5, 3), dtype=np.float64),
        T_worlds=np.repeat(np.eye(4)[None], frame_count, axis=0),
        openings=np.linspace(0.04, 0.08, frame_count),
        bimanual=np.asarray(False),
    )


def test_known_action_slice_materializes_exact_prediction_window(
    tmp_path: Path,
) -> None:
    source = tmp_path / "robot.npz"
    destination = tmp_path / "known_action_76.npz"
    _robot(source, 81)
    _module()._slice_known_action(source, destination)
    with np.load(destination, allow_pickle=False) as stored:
        assert len(stored["actions"]) == 76
        assert len(stored["T_worlds"]) == 76
        assert len(stored["openings"]) == 76
        assert np.asarray(stored["format_version"]).shape == ()
        assert np.asarray(stored["bimanual"]).shape == ()


def test_known_action_slice_rejects_nonfrozen_window(tmp_path: Path) -> None:
    source = tmp_path / "robot.npz"
    _robot(source, 80)
    with pytest.raises(ValueError, match="raw window"):
        _module()._slice_known_action(source, tmp_path / "out.npz")
