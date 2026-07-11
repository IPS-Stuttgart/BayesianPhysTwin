import numpy as np

from bayesian_phystwin.phystwin_controller_sensitivity import (
    controller_hand_count,
    controller_jitter_diagnostics,
    controller_jitter_id,
    infer_controller_groups,
    smooth_group_controller_jitter,
)


def test_controller_groups_separate_two_spatial_hands() -> None:
    first = np.array(
        [[-0.12, 0.00, 0.00], [-0.10, 0.01, 0.00], [-0.11, -0.01, 0.00]]
    )
    second = np.array(
        [[0.10, 0.00, 0.00], [0.12, 0.01, 0.00], [0.11, -0.01, 0.00]]
    )
    labels = infer_controller_groups(np.concatenate((first, second)), group_count=2)

    assert len(set(labels[:3])) == 1
    assert len(set(labels[3:])) == 1
    assert labels[0] != labels[3]
    assert controller_hand_count("double_lift_cloth_1") == 2
    assert controller_hand_count("single_lift_cloth_1") == 1


def test_smooth_controller_jitter_is_matched_and_group_coherent() -> None:
    controls = np.zeros((12, 6, 3), dtype=float)
    groups = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    jitter = smooth_group_controller_jitter(
        controls,
        groups,
        start_frame=5,
        target_rms_m=0.002,
        correlation_frames=4.0,
        seed=17,
    )
    repeated = smooth_group_controller_jitter(
        controls,
        groups,
        start_frame=5,
        target_rms_m=0.002,
        correlation_frames=4.0,
        seed=17,
    )

    np.testing.assert_array_equal(jitter, repeated)
    np.testing.assert_array_equal(jitter[:5], 0.0)
    np.testing.assert_allclose(jitter[:, 0], jitter[:, 1])
    np.testing.assert_allclose(jitter[:, 3], jitter[:, 5])
    rms = np.sqrt(np.mean(np.sum(np.square(jitter[5:]), axis=2)))
    np.testing.assert_allclose(rms, 0.002, atol=1e-15)
    np.testing.assert_allclose(
        (controls + jitter) + (controls - jitter), 2.0 * controls
    )


def test_controller_jitter_diagnostics_keep_vector_units() -> None:
    controls = np.zeros((8, 2, 3), dtype=float)
    jitter = np.zeros_like(controls)
    jitter[3:, :, 0] = 0.003
    diagnostics = controller_jitter_diagnostics(
        controls, jitter, start_frame=3
    )

    np.testing.assert_allclose(diagnostics["jitter_vector_rms_m"], 0.003)
    assert controller_jitter_id(0.002) == "jitter_2mm"
