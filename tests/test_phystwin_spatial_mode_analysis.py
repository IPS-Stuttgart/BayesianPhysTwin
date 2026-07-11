import json
import pickle
from pathlib import Path

import numpy as np

from bayesian_phystwin.phystwin_spatial_mode_analysis import (
    field_localization_diagnostics,
    run_spatial_mode_analysis,
)


def _write_pickle(path: Path, value: object) -> None:
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def test_field_localization_detects_controller_concentration() -> None:
    points = np.array(
        [
            [0.00, 0.00, 0.02],
            [0.02, 0.00, 0.02],
            [0.00, 0.02, 0.02],
            [0.02, 0.02, 0.02],
        ]
    )
    field = np.zeros_like(points)
    field[0, 0] = 0.01
    diagnostics = field_localization_diagnostics(
        points,
        np.array([[0.0, 0.0, 0.02]]),
        field,
        controller_radius_m=0.005,
        ground_band_m=0.001,
    )

    near = diagnostics["near_controller"]
    assert near["point_fraction"] == 0.25
    assert near["residual_energy_fraction"] == 1.0
    assert near["energy_concentration_ratio"] == 4.0


def test_main_spatial_mode_runner_scores_both_metrics(tmp_path: Path) -> None:
    root = tmp_path / "data"
    case = root / "single_lift_cloth"
    case.mkdir(parents=True)
    frame_count = 8
    original = np.array(
        [
            [0.000, 0.000, -0.020],
            [0.020, 0.000, -0.020],
            [0.000, 0.020, -0.020],
            [0.000, 0.000, -0.040],
        ]
    )
    baseline = np.repeat(original[None], frame_count, axis=0)
    observed = baseline.copy()
    observed[1:5, :, 0] += 0.004
    observed[5:, :, 0] += 0.005
    final_data = {
        "object_points": observed,
        "object_visibilities": np.ones((frame_count, len(original)), dtype=bool),
        "object_motions_valid": np.ones(
            (frame_count - 1, len(original)), dtype=bool
        ),
        "controller_points": np.repeat(
            np.array([[[0.0, 0.0, -0.02]]]), frame_count, axis=0
        ),
        "surface_points": np.empty((0, 3)),
        "interior_points": np.empty((0, 3)),
    }
    optimal = {
        "object_radius": 0.05,
        "object_max_neighbours": 8,
        "controller_radius": 0.01,
        "controller_max_neighbours": 4,
    }
    _write_pickle(case / "final_data.pkl", final_data)
    _write_pickle(case / "inference.pkl", baseline)
    _write_pickle(case / "optimal_params.pkl", optimal)
    _write_pickle(case / "gt_track_3d.pkl", observed)
    (case / "split.json").write_text(
        json.dumps({"train": [0, 5], "test": [5, 8], "frame_len": 8}),
        encoding="utf-8",
    )
    (root / "evaluation_subset_manifest.json").write_text(
        json.dumps({"selected_cases": ["single_lift_cloth"]}),
        encoding="utf-8",
    )

    result = run_spatial_mode_analysis(
        root,
        tmp_path / "output",
        cohort="all",
        bootstrap_samples=20,
        bootstrap_block_length=2,
        interpolation_neighbors=4,
    )

    case_result = result["case_results"]["single_lift_cloth"]
    translation = case_result["methods"]["global_translation"]
    assert set(translation["future"]) == {
        "chamfer_distance_m",
        "track_error_m",
    }
    assert translation["endpoint_fit"]["endpoint_sse_explained_fraction"] > 0.999
    np.testing.assert_allclose(
        translation["geometry"]["translation_norm_m"], 0.004, atol=1e-12
    )
    assert result["case_count"] == 1
    assert "affine" in result["comparisons_vs_per_point"]
