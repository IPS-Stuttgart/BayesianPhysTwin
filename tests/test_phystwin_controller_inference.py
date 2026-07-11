import numpy as np

from bayesian_phystwin.phystwin_controller_inference import (
    latent_controller_objective,
    persistent_group_bias_trajectory,
    point_weighted_group_bias_rms,
    scale_group_bias_direction,
)


def test_persistent_group_bias_ramps_then_holds() -> None:
    controls = np.zeros((10, 4, 3), dtype=float)
    groups = np.array([0, 0, 1, 1], dtype=np.int32)
    bias = np.array([[0.003, 0.0, 0.0], [0.0, -0.002, 0.0]])

    jitter = persistent_group_bias_trajectory(
        controls,
        groups,
        bias,
        start_frame=3,
        ramp_frames=3,
    )

    np.testing.assert_array_equal(jitter[:3], 0.0)
    np.testing.assert_allclose(jitter[3, 0], bias[0] / 3.0)
    np.testing.assert_allclose(jitter[4, 2], 2.0 * bias[1] / 3.0)
    np.testing.assert_allclose(jitter[5:, 0], np.repeat(bias[[0]], 5, axis=0))
    np.testing.assert_allclose(jitter[5:, 3], np.repeat(bias[[1]], 5, axis=0))


def test_group_bias_scaling_respects_point_rms_and_group_cap() -> None:
    groups = np.array([0, 0, 0, 1], dtype=np.int32)
    direction = np.array([[1.0, 0.0, 0.0], [0.0, 4.0, 0.0]])

    uncapped = scale_group_bias_direction(
        direction,
        groups,
        target_rms_m=0.002,
        maximum_group_norm_m=0.01,
    )
    capped = scale_group_bias_direction(
        direction,
        groups,
        target_rms_m=0.002,
        maximum_group_norm_m=0.0025,
    )

    np.testing.assert_allclose(point_weighted_group_bias_rms(uncapped, groups), 0.002)
    assert np.max(np.linalg.norm(capped, axis=1)) <= 0.0025 + 1e-15
    assert point_weighted_group_bias_rms(capped, groups) < 0.002


def test_latent_controller_objective_penalizes_nonzero_rough_bias() -> None:
    chamfer = np.full(5, 0.01)
    zero = np.zeros((9, 2, 3), dtype=float)
    rough = zero.copy()
    rough[4:, :, 0] = 0.002

    baseline = latent_controller_objective(
        chamfer,
        zero,
        start_frame=4,
        stop_frame=9,
        observation_sigma_m=0.005,
        controller_sigma_m=0.002,
        smoothness_sigma_m=0.0005,
    )
    candidate = latent_controller_objective(
        chamfer,
        rough,
        start_frame=4,
        stop_frame=9,
        observation_sigma_m=0.005,
        controller_sigma_m=0.002,
        smoothness_sigma_m=0.0005,
    )

    assert baseline["objective"] == baseline["data_term"]
    assert candidate["proximity_term"] > 0.0
    assert candidate["smoothness_term"] > 0.0
    assert candidate["objective"] > baseline["objective"]
