from __future__ import annotations

import numpy as np

from bayesian_phystwin.phystwin_tapnextpp_cotracker_complement import (
    TAPNextPPCoTrackerComplementConfig,
    build_complementary_prediction_arrays,
)


def _fixture(
    *,
    tap_offset_m: float = 0.0,
    tap_supported: bool = True,
    neighbor_count: int = 3,
) -> tuple[dict[str, np.ndarray], TAPNextPPCoTrackerComplementConfig]:
    frame_count = 8
    query = np.zeros((1, 3), dtype=np.float32)
    tap = np.zeros((frame_count, 1, 3), dtype=np.float32)
    tap[:, 0, 0] = np.arange(frame_count) * 0.001 + tap_offset_m
    support = np.zeros((frame_count, 1), dtype=bool)
    if tap_supported:
        support[:5] = True
    covariance = np.repeat(
        np.eye(3, dtype=np.float32)[None, None],
        frame_count,
        axis=0,
    )
    covariance *= 1e-6
    reliability = np.ones((frame_count, 1), dtype=np.float32)
    base_offset = np.linspace(-0.002, 0.002, neighbor_count)
    points = np.zeros((frame_count, neighbor_count, 3), dtype=np.float32)
    points[0, :, 1] = base_offset
    for frame in range(1, frame_count):
        points[frame] = points[0]
        points[frame, :, 0] += frame * 0.001
    valid = np.ones((frame_count, neighbor_count), dtype=bool)
    camera_count = np.full((frame_count, neighbor_count), 2, dtype=np.int16)
    reprojection = np.ones((frame_count, neighbor_count), dtype=np.float32)
    quality = np.full((frame_count, neighbor_count), 0.9, dtype=np.float32)
    config = TAPNextPPCoTrackerComplementConfig(
        source_frame_start=0,
        source_frame_end_exclusive=frame_count,
        minimum_prior_overlap_rows=3,
        endpoint_frame_count=2,
    )
    return (
        {
            "query_points_world_m": query,
            "tapnextpp_trajectory_m": tap,
            "tapnextpp_support": support,
            "tapnextpp_covariance_m2": covariance,
            "tapnextpp_reliability": reliability,
            "cotracker_points_world_m": points,
            "cotracker_valid": valid,
            "cotracker_camera_count": camera_count,
            "cotracker_reprojection_error_px": reprojection,
            "cotracker_quality_probability": quality,
        },
        config,
    )


def test_complement_fills_only_after_causal_provider_agreement() -> None:
    inputs, config = _fixture()
    result = build_complementary_prediction_arrays(**inputs, config=config)

    np.testing.assert_array_equal(
        result["trajectory_world_m"][:5],
        inputs["tapnextpp_trajectory_m"][:5],
    )
    np.testing.assert_array_equal(result["provider_code"][:5], 1)
    np.testing.assert_array_equal(result["provider_code"][5:], 2)
    np.testing.assert_allclose(
        result["trajectory_world_m"][:, 0, 0],
        np.arange(8) * 0.001,
        atol=1e-8,
    )
    assert np.all(result["bridge_anchor_frame"][5:] == 4)


def test_complement_refuses_identity_without_tapnextpp_history() -> None:
    inputs, config = _fixture(tap_supported=False)
    result = build_complementary_prediction_arrays(**inputs, config=config)

    assert not np.any(result["accepted_support"])
    assert not np.any(result["provider_code"])
    np.testing.assert_array_equal(
        result["trajectory_world_m"],
        inputs["tapnextpp_trajectory_m"],
    )


def test_complement_refuses_inconsistent_provider() -> None:
    inputs, config = _fixture(tap_offset_m=0.020)
    result = build_complementary_prediction_arrays(**inputs, config=config)

    assert np.all(result["provider_code"][:5] == 1)
    assert not np.any(result["provider_code"][5:] == 2)
    assert not np.any(result["accepted_support"][5:])


def test_duplicate_correlated_neighbors_do_not_remove_bias_floor() -> None:
    inputs_three, config = _fixture(neighbor_count=3)
    inputs_many, _ = _fixture(neighbor_count=30)
    result_three = build_complementary_prediction_arrays(
        **inputs_three,
        config=config,
    )
    result_many = build_complementary_prediction_arrays(
        **inputs_many,
        config=config,
    )

    floor = 2.0 * config.two_view_shared_bias_standard_deviation_m**2
    assert np.min(np.linalg.eigvalsh(result_three["observation_covariance_m2"][5, 0])) >= floor
    assert np.min(np.linalg.eigvalsh(result_many["observation_covariance_m2"][5, 0])) >= floor


def test_tapnextpp_supported_rows_are_bit_identical() -> None:
    inputs, config = _fixture()
    tap = inputs["tapnextpp_trajectory_m"]
    tap[2, 0] = np.array([0.002, -0.0, np.float32(-0.0)])
    result = build_complementary_prediction_arrays(**inputs, config=config)

    support = inputs["tapnextpp_support"]
    np.testing.assert_array_equal(result["trajectory_world_m"][support], tap[support])
