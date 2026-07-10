import pickle
from pathlib import Path

import numpy as np

from bayesian_phystwin.phystwin_additional_confirmation import (
    apply_persistent_residual_anchor,
)


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
