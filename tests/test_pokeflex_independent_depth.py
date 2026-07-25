import hashlib

import numpy as np
import pytest

from bayesian_phystwin.pokeflex_independent_depth import (
    PokeFlexIndependentDepthAnchor,
    apply_independent_depth_guard,
    build_independent_depth_anchor,
    calibrate_depth_translation,
    load_independent_depth_anchor,
    realsense_depth_to_world_points,
    save_independent_depth_anchor,
    select_points_near_geometry,
)


CALIBRATION_SHA256 = hashlib.sha256(b"calibration").hexdigest()


def _anchor(points_by_sensor: tuple[np.ndarray, ...]) -> PokeFlexIndependentDepthAnchor:
    return build_independent_depth_anchor(
        take_id="FoamDice_T3",
        frame_id=7,
        causal_cutoff_frame=7,
        sensor_points_m=points_by_sensor,
        sensor_names=tuple(f"realsense{index}" for index in range(len(points_by_sensor))),
        calibration_sha256=(CALIBRATION_SHA256,) * len(points_by_sensor),
        sensor_variance_m2=0.002**2,
        voxel_size_m=0.004,
        maximum_clusters_per_sensor=32,
    )


def test_realsense_depth_uses_metric_scale_and_inverse_extrinsic() -> None:
    depth = np.array([[1000, 65535], [2000, 3000]], dtype=np.uint16)
    intrinsics = np.array(
        [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]]
    )
    world_to_camera = np.eye(4)
    world_to_camera[0, 3] = 0.5

    points = realsense_depth_to_world_points(
        depth,
        intrinsics,
        world_to_camera,
        depth_scale=10000.0,
        minimum_depth_m=0.05,
        maximum_depth_m=0.40,
    )

    assert points.shape == (3, 3)
    assert np.allclose(points[0], [-0.5, 0.0, 0.1])
    assert np.allclose(points[1], [-0.5, 0.002, 0.2])
    assert np.allclose(points[2], [-0.497, 0.003, 0.3])


def test_duplicate_pixel_block_does_not_increase_anchor_information() -> None:
    points = np.array([[0.0, 0.0, 0.0], [0.001, 0.001, 0.001]])
    duplicate = np.repeat(points, 100, axis=0)

    original = _anchor((points, points + 0.02))
    repeated = _anchor((duplicate, np.repeat(points + 0.02, 100, axis=0)))

    assert len(original.points_m) == len(repeated.points_m) == 2
    assert np.array_equal(original.variance_m2, repeated.variance_m2)


def test_anchor_round_trip_preserves_metric_covariance(tmp_path) -> None:
    anchor = _anchor(
        (
            np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]]),
            np.array([[0.0, 0.02, 0.0], [0.01, 0.02, 0.0]]),
        )
    )
    path = save_independent_depth_anchor(anchor, tmp_path / "anchor.npz")

    loaded = load_independent_depth_anchor(path)

    assert loaded.take_id == anchor.take_id
    assert loaded.frame_id == anchor.frame_id
    assert np.array_equal(loaded.points_m, anchor.points_m)
    assert np.array_equal(loaded.variance_m2, anchor.variance_m2)
    assert loaded.metadata_dict()["variance_unit"] == "m^2"


def test_anchor_rejects_observation_after_causal_cutoff() -> None:
    with pytest.raises(ValueError, match="causal cutoff"):
        PokeFlexIndependentDepthAnchor(
            take_id="FoamDice_T3",
            frame_id=8,
            causal_cutoff_frame=7,
            points_m=np.zeros((2, 3)),
            variance_m2=np.ones(2) * 1e-6,
            sensor_index=np.array([0, 0]),
            sensor_names=("realsense0",),
            calibration_sha256=(CALIBRATION_SHA256,),
        )


def test_translation_calibration_recovers_known_source_bias() -> None:
    template = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.02, 0.0, 0.0],
            [0.0, 0.02, 0.0],
            [0.0, 0.0, 0.02],
        ]
    )
    template = np.repeat(template, 16, axis=0)
    bias = np.array([0.006, -0.004, 0.003])

    result = calibrate_depth_translation(
        template + bias,
        template,
        maximum_association_m=0.02,
        minimum_inliers=16,
    )

    assert np.allclose(result.translation_m, -bias, atol=1e-9)
    assert result.median_residual_m < 1e-9


def test_static_geometry_support_removes_background_without_state_residual() -> None:
    geometry = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
    points = np.array(
        [[0.001, 0.0, 0.0], [0.011, 0.0, 0.0], [0.0, 0.03, 0.0]]
    )

    selected = select_points_near_geometry(
        points,
        geometry,
        maximum_distance_m=0.005,
    )

    assert np.array_equal(selected, points[:2])


def test_unknown_correlation_guard_uses_worst_sensor_and_exact_fallback() -> None:
    baseline = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
    candidate = baseline + np.array([0.002, 0.0, 0.0])
    anchor = _anchor(
        (
            candidate.copy(),
            baseline.copy(),
        )
    )

    result = apply_independent_depth_guard(baseline, candidate, anchor)

    assert not result.accepted
    assert np.array_equal(result.selected_vertices_m, baseline)
    assert result.covariance_intersection_upper_regret_mm == pytest.approx(
        np.max(result.per_sensor_regret_mm)
    )
    assert result.covariance_intersection_upper_regret_mm >= float(
        np.mean(result.per_sensor_regret_mm)
    )


def test_guard_accepts_only_when_every_sensor_supports_candidate() -> None:
    baseline = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
    candidate = baseline + np.array([0.003, 0.0, 0.0])
    anchor = _anchor((candidate.copy(), candidate.copy()))

    result = apply_independent_depth_guard(
        baseline,
        candidate,
        anchor,
        minimum_improvement_mm=0.5,
    )

    assert result.accepted
    assert np.array_equal(result.selected_vertices_m, candidate)
