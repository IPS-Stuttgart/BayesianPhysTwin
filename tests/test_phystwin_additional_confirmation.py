import pickle
from pathlib import Path

import numpy as np

from bayesian_phystwin.phystwin_additional_confirmation import (
    apply_endpoint_transform,
    apply_persistent_residual_anchor,
    fit_endpoint_transform,
)


def test_endpoint_transform_controls_recover_known_transforms() -> None:
    source = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.02, 0.00, 0.00],
            [0.00, 0.03, 0.00],
            [0.00, 0.00, 0.04],
            [0.01, 0.015, 0.02],
            [-0.01, 0.01, 0.03],
        ]
    )
    angle = 0.37
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.array([0.004, -0.006, 0.003])

    rigid_target = source @ rotation + translation
    rigid = fit_endpoint_transform(source, rigid_target, mode="se3")
    np.testing.assert_allclose(
        apply_endpoint_transform(source, rigid), rigid_target, atol=1e-12
    )
    assert rigid["scale"] == 1.0
    assert np.linalg.det(np.asarray(rigid["rotation"])) > 0.0

    similarity_target = 1.15 * source @ rotation + translation
    similarity = fit_endpoint_transform(source, similarity_target, mode="sim3")
    np.testing.assert_allclose(
        apply_endpoint_transform(source, similarity),
        similarity_target,
        atol=1e-12,
    )
    np.testing.assert_allclose(similarity["scale"], 1.15, atol=1e-12)

    affine_linear = np.array(
        [
            [1.05, 0.10, -0.02],
            [-0.04, 0.95, 0.03],
            [0.01, -0.06, 1.10],
        ]
    )
    affine_target = source @ affine_linear + translation
    affine = fit_endpoint_transform(source, affine_target, mode="affine")
    np.testing.assert_allclose(
        apply_endpoint_transform(source, affine), affine_target, atol=1e-12
    )
    assert affine["rotation"] is None
    assert affine["scale"] is None


def test_label_free_anchor_uses_only_training_residual(tmp_path: Path) -> None:
    frame_count = 8
    original = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]])
    observed = np.repeat(original[None], frame_count, axis=0)
    observed[1:, :, 0] += 0.008
    baseline = np.repeat(original[None], frame_count, axis=0)
    data = {
        "object_points": observed.astype(np.float32),
        "object_visibilities": np.ones((frame_count, 2), dtype=bool),
        "object_motions_valid": np.ones((frame_count - 1, 2), dtype=bool),
        "controller_points": np.zeros((frame_count, 1, 3), dtype=np.float32),
        "surface_points": np.empty((0, 3), dtype=np.float32),
        "interior_points": np.empty((0, 3), dtype=np.float32),
    }
    final_path = tmp_path / "final.pkl"
    baseline_path = tmp_path / "baseline.pkl"
    for path, value in (
        (final_path, data),
        (baseline_path, baseline.astype(np.float32)),
    ):
        with path.open("wb") as handle:
            pickle.dump(value, handle)

    summary = apply_persistent_residual_anchor(
        final_path,
        baseline_path,
        tmp_path / "output",
        train_end_frame=5,
        maximum_residual_m=0.01,
        interpolation_neighbors=1,
    )

    assert summary["contract"]["manual_labels"] == "none"
    assert summary["contract"]["selection"] == "none"
    assert summary["future"]["percent_change"] < -99.0


def test_global_translation_mode_is_explicit(tmp_path: Path) -> None:
    frame_count = 6
    original = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]])
    observed = np.repeat(original[None], frame_count, axis=0)
    observed[1:, :, 1] += 0.004
    baseline = np.repeat(original[None], frame_count, axis=0)
    data = {
        "object_points": observed.astype(np.float32),
        "object_visibilities": np.ones((frame_count, 2), dtype=bool),
        "object_motions_valid": np.ones((frame_count - 1, 2), dtype=bool),
        "controller_points": np.zeros((frame_count, 1, 3), dtype=np.float32),
        "surface_points": np.empty((0, 3), dtype=np.float32),
        "interior_points": np.empty((0, 3), dtype=np.float32),
    }
    final_path = tmp_path / "global_final.pkl"
    baseline_path = tmp_path / "global_baseline.pkl"
    for path, value in (
        (final_path, data),
        (baseline_path, baseline.astype(np.float32)),
    ):
        with path.open("wb") as handle:
            pickle.dump(value, handle)

    summary = apply_persistent_residual_anchor(
        final_path,
        baseline_path,
        tmp_path / "global_output",
        train_end_frame=4,
        interpolation_neighbors=1,
        spatial_mode="global_translation",
    )

    assert summary["contract"]["spatial_model"] == "global_translation"
