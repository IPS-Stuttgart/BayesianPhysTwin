import numpy as np
import pytest

from bayesian_phystwin.phystwin_graph_discrepancy import (
    normalized_spring_laplacian,
)
from causal4d.rest_geometry import (
    apply_frame_correction,
    corrected_spring_rest_lengths,
    fit_weighted_frame_correction,
    infer_graph_rest_geometry_correction,
    reattach_controller_rest_lengths,
    rotate_vectors,
    scaled_frame_correction,
)


def _row_rotation_z(angle: float) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.array(
        [[cosine, sine, 0.0], [-sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )


def test_weighted_se3_recovers_rigid_correspondence() -> None:
    source = np.array(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.3]]
    )
    linear = _row_rotation_z(np.deg2rad(3.0))
    translation = np.array([0.004, -0.003, 0.002])
    target = source @ linear + translation

    correction = fit_weighted_frame_correction(
        source,
        target,
        np.array([1.0, 0.5, 2.0, 1.5]),
        mode="se3",
    )

    np.testing.assert_allclose(correction.linear, linear, atol=1e-12)
    np.testing.assert_allclose(correction.translation, translation, atol=1e-12)
    np.testing.assert_allclose(apply_frame_correction(source, correction), target)
    np.testing.assert_allclose(
        rotate_vectors(source, correction), source @ linear, atol=1e-15
    )


def test_frame_fit_obeys_bounds_and_scales_on_se3() -> None:
    source = np.array(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]
    )
    target = source @ _row_rotation_z(np.deg2rad(15.0)) + np.array([0.1, 0.0, 0.0])

    bounded = fit_weighted_frame_correction(
        source,
        target,
        np.ones(3),
        maximum_rotation_rad=np.deg2rad(5.0),
        maximum_translation_m=0.02,
    )
    half = scaled_frame_correction(bounded, 0.5)

    assert bounded.rotation_angle_rad == pytest.approx(np.deg2rad(5.0))
    assert np.linalg.norm(bounded.translation) == pytest.approx(0.02)
    assert half.rotation_angle_rad == pytest.approx(np.deg2rad(2.5))
    np.testing.assert_allclose(half.translation, 0.5 * bounded.translation)
    assert np.linalg.det(half.linear) == pytest.approx(1.0)


def test_rest_length_update_changes_only_object_springs_and_clips_ratio() -> None:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    springs = np.array([[0, 1], [1, 2], [3, 0]])
    released = np.array([1.0, 1.0, 0.4])
    field = np.array(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )

    _, corrected, raw_ratio, ratio = corrected_spring_rest_lengths(
        vertices,
        springs,
        released,
        num_object_springs=2,
        nonrigid_field=field,
        correction_scale=1.0,
        maximum_log_ratio=np.log(1.1),
    )

    np.testing.assert_allclose(raw_ratio, [1.5, 1.5])
    np.testing.assert_allclose(ratio, [1.1, 1.1])
    np.testing.assert_allclose(corrected, [1.1, 1.1, 0.4])


def test_controller_reattachment_recomputes_only_attachment_tail() -> None:
    object_vertices = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]])
    controllers = np.array([[0.0, 1.0, 0.0]])
    springs = np.array([[0, 1], [2, 0], [2, 1]])
    rest = np.array([1.1, 1.0, np.sqrt(2.0)])

    corrected, raw_ratio, ratio = reattach_controller_rest_lengths(
        object_vertices,
        controllers,
        springs,
        rest,
        num_object_springs=1,
        maximum_log_ratio=np.log(1.05),
    )

    assert corrected[0] == rest[0]
    assert raw_ratio[0] == pytest.approx(1.0)
    assert raw_ratio[1] > 1.0
    np.testing.assert_allclose(ratio, [1.0, 1.05])


def test_rigid_endpoint_discrepancy_is_removed_before_graph_smoothing() -> None:
    scipy = pytest.importorskip("scipy")
    assert scipy is not None
    reference = np.array(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.1, 0.0], [0.0, 0.1, 0.0]]
    )
    springs = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]], dtype=np.int32)
    released = np.linalg.norm(
        reference[springs[:, 0]] - reference[springs[:, 1]], axis=1
    )
    linear = _row_rotation_z(np.deg2rad(2.0))
    translation = np.array([0.004, -0.002, 0.001])
    discrepancy = reference @ linear + translation - reference
    laplacian = normalized_spring_laplacian(len(reference), springs)

    correction = infer_graph_rest_geometry_correction(
        reference,
        reference,
        springs,
        released,
        num_object_springs=len(springs),
        endpoint_mean=discrepancy,
        endpoint_variance=np.full(len(reference), 1e-6),
        observed=np.ones(len(reference), dtype=bool),
        laplacian=laplacian,
        graph_prior_strength=0.1,
    )

    np.testing.assert_allclose(correction.nonrigid_field, 0.0, atol=1e-10)
    np.testing.assert_allclose(correction.endpoint_correction, discrepancy, atol=1e-10)
    np.testing.assert_allclose(
        correction.corrected_reference_vertices,
        reference @ linear + translation,
        atol=1e-10,
    )
    np.testing.assert_allclose(correction.corrected_rest_lengths, released, atol=1e-10)


def test_frame_none_leaves_translation_in_graph_nullspace_without_straining_springs() -> None:
    pytest.importorskip("scipy")
    reference = np.array(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]
    )
    springs = np.array([[0, 1], [1, 2]], dtype=np.int32)
    released = np.array([0.1, 0.1])
    laplacian = normalized_spring_laplacian(len(reference), springs)

    correction = infer_graph_rest_geometry_correction(
        reference,
        reference,
        springs,
        released,
        num_object_springs=2,
        endpoint_mean=np.tile([0.005, 0.0, 0.0], (3, 1)),
        endpoint_variance=np.full(3, 1e-6),
        observed=np.ones(3, dtype=bool),
        laplacian=laplacian,
        graph_prior_strength=0.1,
        frame_mode="none",
    )

    np.testing.assert_allclose(
        correction.nonrigid_field,
        np.tile([0.005, 0.0, 0.0], (3, 1)),
        atol=1e-8,
    )
    np.testing.assert_allclose(correction.corrected_rest_lengths, released, atol=1e-8)
