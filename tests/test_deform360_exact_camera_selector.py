from __future__ import annotations

from pathlib import Path

import numpy as np

from bayesian_phystwin.deform360_exact_camera_selector import (
    compile_exact_camera_subset_solver,
    select_frame_zero_observation_plan_exact_fast,
)
from bayesian_phystwin.deform360_raw_camera_observation import (
    RawCameraObservationConfig,
    select_frame_zero_observation_plan,
)


def _fixture(seed: int = 9):
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(18, 3)) * 0.05
    points[:, 2] += 0.4
    cameras = tuple(f"camera-{index:02d}" for index in range(10))
    support = rng.random((len(points), len(cameras))) > 0.32
    support[:, :3] = True
    projected = {
        camera: rng.uniform(0.0, 512.0, size=(len(points), 2)) for camera in cameras
    }
    extrinsics = {}
    for index, camera in enumerate(cameras):
        angle = 2.0 * np.pi * index / len(cameras)
        transform = np.eye(4)
        transform[:3, 3] = [0.3 * np.cos(angle), 0.3 * np.sin(angle), 0.05]
        extrinsics[camera] = transform
    config = RawCameraObservationConfig(
        center_count=6,
        selected_camera_count=4,
    )
    return points, cameras, support, projected, extrinsics, config


def test_native_solver_is_checksum_addressed(tmp_path: Path) -> None:
    executable, provenance = compile_exact_camera_subset_solver(tmp_path)
    second, repeated = compile_exact_camera_subset_solver(tmp_path)

    assert executable == second
    assert provenance == repeated
    assert len(provenance["native_source_sha256"]) == 64
    assert len(provenance["native_executable_sha256"]) == 64


def test_exact_accelerator_matches_frozen_exhaustive_selector(
    tmp_path: Path,
) -> None:
    points, cameras, support, projected, extrinsics, config = _fixture()

    expected = select_frame_zero_observation_plan(
        points,
        cameras,
        support,
        projected,
        extrinsics,
        config=config,
    )
    actual = select_frame_zero_observation_plan_exact_fast(
        points,
        cameras,
        support,
        projected,
        extrinsics,
        config=config,
        cache_dir=tmp_path,
    )

    assert actual["selected_cameras"] == expected["selected_cameras"]
    assert actual["selection_score"] == expected["selection_score"]
    assert np.array_equal(actual["candidate_ids"], expected["candidate_ids"])
    assert np.array_equal(actual["center_ids"], expected["center_ids"])
    assert np.array_equal(
        actual["selected_camera_indices"], expected["selected_camera_indices"]
    )
    for camera in actual["selected_cameras"]:
        assert np.array_equal(
            actual["query_ids"][camera], expected["query_ids"][camera]
        )
        assert np.array_equal(
            actual["query_pixels"][camera], expected["query_pixels"][camera]
        )


def test_exact_accelerator_preserves_lexicographic_ties(tmp_path: Path) -> None:
    points, cameras, _, projected, extrinsics, config = _fixture(seed=12)
    support = np.ones((len(points), len(cameras)), dtype=bool)

    expected = select_frame_zero_observation_plan(
        points,
        cameras,
        support,
        projected,
        extrinsics,
        config=config,
    )
    actual = select_frame_zero_observation_plan_exact_fast(
        points,
        cameras,
        support,
        projected,
        extrinsics,
        config=config,
        cache_dir=tmp_path,
    )

    assert actual["selected_cameras"] == expected["selected_cameras"]
    assert actual["selection_score"] == expected["selection_score"]
