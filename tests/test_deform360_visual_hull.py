from __future__ import annotations

import numpy as np

from causal4d_public.deform360_visual_hull import (
    AdaptiveRopeHullConfig,
    adaptive_rope_visual_hull,
    carve_candidate_points,
    regular_grid_in_bounds,
)


def _camera(translation: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    intrinsics = np.asarray([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    transform = np.eye(4)
    transform[:3, 3] = translation
    return intrinsics, transform


def test_regular_grid_coarsens_to_the_declared_cap() -> None:
    grid, diagnostics = regular_grid_in_bounds(
        np.asarray((-0.3, -0.2, 0.8)),
        np.asarray((0.3, 0.2, 1.2)),
        requested_voxel_size_m=0.002,
        maximum_point_count=200_000,
    )

    assert len(grid) <= 200_000
    assert diagnostics["coarsened_for_grid_cap"] is True
    assert diagnostics["grid_point_count"] == len(grid)


def test_candidate_carving_uses_peak_relative_multiview_votes() -> None:
    candidates = np.asarray(
        [
            (-0.10, 0.0, 1.0),
            (0.00, 0.0, 1.0),
            (0.10, 0.0, 1.0),
            (0.35, 0.0, 1.0),
        ]
    )
    masks = {}
    intrinsics = {}
    extrinsics = {}
    for name, translation in (
        ("a", (0.0, 0.0, 0.0)),
        ("b", (0.01, 0.0, 0.0)),
        ("c", (-0.01, 0.0, 0.0)),
    ):
        mask = np.zeros((100, 100), dtype=bool)
        mask[47:54, 35:66] = True
        masks[name] = mask
        intrinsics[name], extrinsics[name] = _camera(translation)

    hull, diagnostics = carve_candidate_points(
        candidates,
        masks,
        intrinsics,
        extrinsics,
        consensus_fraction_of_peak=0.6,
        minimum_consensus_votes=2,
    )

    assert len(hull) == 3
    assert diagnostics["peak_vote_count"] == 3
    assert diagnostics["required_vote_count"] == 2


def test_adaptive_hull_recovers_a_local_rope_volume() -> None:
    prior = np.column_stack((np.linspace(-0.12, 0.12, 13), np.zeros(13), np.ones(13)))
    masks = {}
    intrinsics = {}
    extrinsics = {}
    for name, translation in (
        ("a", (0.0, 0.0, 0.0)),
        ("b", (0.015, 0.0, 0.0)),
        ("c", (-0.015, 0.0, 0.0)),
    ):
        mask = np.zeros((100, 100), dtype=bool)
        mask[48:53, 35:66] = True
        masks[name] = mask
        intrinsics[name], extrinsics[name] = _camera(translation)
    config = AdaptiveRopeHullConfig(
        local_voxel_size_m=0.006,
        initial_margin_m=0.03,
        minimum_consensus_votes=2,
        minimum_hull_point_count=20,
        maximum_grid_point_count=200_000,
    )

    hull, diagnostics = adaptive_rope_visual_hull(
        prior,
        masks,
        intrinsics,
        extrinsics,
        config=config,
    )

    assert len(hull) >= 20
    assert diagnostics["attempt_count"] == 1
    assert diagnostics["final_hull_world_m"]["q01_to_q99_span"][0] > 0.20
