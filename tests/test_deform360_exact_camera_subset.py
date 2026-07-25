from __future__ import annotations

import hashlib
import itertools
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.deform360_exact_camera_subset import (
    FROZEN_RAW_CAMERA_BUILDER_SHA256,
    frozen_builder_path,
    select_exact_camera_subset,
    select_frame_zero_observation_plan_exact_accelerated,
)
from bayesian_phystwin.deform360_raw_camera_observation import (
    RawCameraObservationConfig,
    _maximum_ray_angle_degrees,
    select_frame_zero_observation_plan,
)


def _reference_subset(
    points: np.ndarray,
    centers: np.ndarray,
    support: np.ndarray,
    origins: np.ndarray,
    *,
    selected_camera_count: int,
    minimum_initial_view_count: int,
) -> tuple[tuple[int, ...], tuple[int, int, int, float]]:
    best_subset: tuple[int, ...] | None = None
    best_score: tuple[int, int, int, float] | None = None
    for subset in itertools.combinations(
        range(support.shape[1]), selected_camera_count
    ):
        counts = np.sum(support[centers][:, subset], axis=1)
        angles = [
            _maximum_ray_angle_degrees(
                points[point_id],
                [index for index in subset if support[point_id, index]],
                origins,
            )
            for center_index, point_id in enumerate(centers)
            if counts[center_index] >= 2
        ]
        score = (
            int(np.sum(counts >= minimum_initial_view_count)),
            int(np.sum(counts >= 3)),
            int(np.sum(counts)),
            0.0 if not angles else float(np.median(angles)),
        )
        if best_score is None or score > best_score:
            best_subset = subset
            best_score = score
    if best_subset is None or best_score is None:
        raise AssertionError("reference camera selector produced no subset")
    return best_subset, best_score


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("batch_size", [1, 7, 64])
def test_exact_camera_subset_matches_frozen_reference(
    seed: int,
    batch_size: int,
) -> None:
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(9, 3))
    points[:, 2] += 3.0
    origins = rng.normal(scale=0.5, size=(9, 3))
    support = rng.random((len(points), len(origins))) > 0.45
    support[:, :2] = True
    centers = np.arange(7, dtype=np.int64)

    expected = _reference_subset(
        points,
        centers,
        support,
        origins,
        selected_camera_count=4,
        minimum_initial_view_count=2,
    )
    actual = select_exact_camera_subset(
        points,
        centers,
        support,
        origins,
        selected_camera_count=4,
        minimum_initial_view_count=2,
        batch_size=batch_size,
    )

    assert actual == expected


def test_exact_camera_subset_preserves_first_lexicographic_tie() -> None:
    points = np.column_stack((np.linspace(0.0, 0.4, 6), np.zeros(6), np.full(6, 2.0)))
    origins = np.zeros((8, 3))
    support = np.ones((len(points), len(origins)), dtype=bool)
    centers = np.arange(len(points), dtype=np.int64)

    subset, score = select_exact_camera_subset(
        points,
        centers,
        support,
        origins,
        selected_camera_count=4,
        minimum_initial_view_count=2,
        batch_size=3,
    )

    assert subset == (0, 1, 2, 3)
    assert score == (6, 6, 24, 0.0)


def test_accelerated_full_plan_matches_frozen_plan() -> None:
    point_count = 20
    camera_count = 9
    angle = np.linspace(0.0, 2.0 * np.pi, point_count, endpoint=False)
    points = np.stack(
        (0.2 * np.cos(angle), 0.15 * np.sin(angle), np.full(point_count, 2.0)),
        axis=1,
    )
    cameras = tuple(f"camera-{index}" for index in range(camera_count))
    extrinsics = {}
    for index, camera in enumerate(cameras):
        matrix = np.eye(4)
        matrix[:3, 3] = (
            0.4 * np.cos(2.0 * np.pi * index / camera_count),
            0.4 * np.sin(2.0 * np.pi * index / camera_count),
            0.0,
        )
        extrinsics[camera] = matrix
    rng = np.random.default_rng(17)
    support = rng.random((point_count, camera_count)) > 0.35
    support[:, :2] = True
    projected = {camera: np.zeros((point_count, 2)) for camera in cameras}
    config = RawCameraObservationConfig(center_count=8, selected_camera_count=4)

    expected = select_frame_zero_observation_plan(
        points,
        cameras,
        support,
        projected,
        extrinsics,
        config=config,
    )
    actual = select_frame_zero_observation_plan_exact_accelerated(
        points,
        cameras,
        support,
        projected,
        extrinsics,
        config=config,
    )

    assert actual["selected_cameras"] == expected["selected_cameras"]
    assert actual["selection_score"] == expected["selection_score"]
    for key in (
        "candidate_ids",
        "center_ids",
        "selected_camera_indices",
        "support",
    ):
        np.testing.assert_array_equal(actual[key], expected[key])
    assert actual["query_ids"].keys() == expected["query_ids"].keys()
    for camera in actual["query_ids"]:
        np.testing.assert_array_equal(
            actual["query_ids"][camera],
            expected["query_ids"][camera],
        )
        np.testing.assert_array_equal(
            actual["query_pixels"][camera],
            expected["query_pixels"][camera],
        )


def test_accelerator_is_bound_to_frozen_builder_hash() -> None:
    path = Path(frozen_builder_path())

    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        FROZEN_RAW_CAMERA_BUILDER_SHA256
    )
