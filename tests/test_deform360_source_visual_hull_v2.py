from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "science"
    / "audit_deform360_source_visual_hull_v2.py"
)
SPEC = importlib.util.spec_from_file_location(
    "audit_deform360_source_visual_hull_v2", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_carrier(path: Path, shape: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "data",
            shape=shape,
            chunks=(1, shape[1], shape[2]),
            dtype=np.uint8,
        )


def test_episode_frame_quantiles_are_content_independent() -> None:
    first = MODULE._episode_frame_indices(
        np.zeros(264, dtype=np.uint8), [0.1, 0.5, 0.9]
    )
    second = MODULE._episode_frame_indices(
        np.arange(264, dtype=np.int64), [0.1, 0.5, 0.9]
    )
    assert first == [26, 132, 237]
    assert second == first


def test_duplicate_episode_frame_quantiles_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        MODULE._episode_frame_indices(
            np.zeros(3, dtype=np.uint8), [0.1, 0.2]
        )


def test_mask_timeline_population_reads_headers_without_tactile_files(
    tmp_path: Path,
) -> None:
    episode = tmp_path / "038-mat-cloth" / "episode_0003"
    _write_carrier(episode / "camera_00" / "mask_refined.h5", (17, 8, 9))
    _write_carrier(episode / "camera_01" / "mask_refined.h5", (17, 8, 9))
    population, records = MODULE._mask_timeline_population(episode, [])
    assert population.shape == (17,)
    assert np.count_nonzero(population) == 0
    assert len(records) == 2
    assert all(record["payload_frames_opened_for_selection"] == 0 for record in records)
    assert {record["camera"] for record in records} == {"camera_00", "camera_01"}


def test_mask_timeline_population_rejects_inconsistent_lengths(
    tmp_path: Path,
) -> None:
    episode = tmp_path / "038-mat-cloth" / "episode_0003"
    _write_carrier(episode / "camera_00" / "mask_refined.h5", (17, 8, 9))
    _write_carrier(episode / "camera_01" / "mask_refined.h5", (18, 8, 9))
    with pytest.raises(ValueError, match="timelines disagree"):
        MODULE._mask_timeline_population(episode, [])
