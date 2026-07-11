import json
import math
import pickle
from pathlib import Path

import numpy as np

from bayesian_phystwin.phystwin_calibration import (
    PhysTwinCalibrationProtocol,
    conformal_upper_bounds,
    finite_sample_conformal_quantile,
    lift_diagonal_anchor_variance,
    run_phystwin_calibration_audit,
    summarize_nees,
)


def test_finite_sample_quantile_refuses_an_impossible_coverage_level() -> None:
    finite, rank = finite_sample_conformal_quantile(np.arange(9.0), 0.9)
    infinite, impossible_rank = finite_sample_conformal_quantile(np.arange(7.0), 0.9)

    assert finite == 8.0
    assert rank == 9
    assert math.isinf(infinite)
    assert impossible_rank == 8


def test_scaled_conformal_bound_uses_the_posterior_scale() -> None:
    upper, quantile, rank = conformal_upper_bounds(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([2.0, 4.0]),
        coverage=0.5,
        score="scaled",
    )

    assert rank == 2
    assert quantile == 2.0
    assert np.allclose(upper, [4.0, 8.0])


def test_diagonal_variance_lift_squares_interpolation_weights() -> None:
    lifted = lift_diagonal_anchor_variance(
        np.array([1.0, 4.0]),
        3,
        np.array([[0, 1]]),
        np.array([[0.5, 0.5]]),
    )

    assert np.allclose(lifted, [1.0, 4.0, 1.25])


def test_nees_summary_uses_three_dimensional_expectation() -> None:
    summary = summarize_nees(np.array([3.0, 3.0]))

    assert summary["mean_3d"] == 3.0
    assert summary["mean_per_coordinate"] == 1.0
    assert summary["covariance_multiplier_for_mean_nees_3"] == 1.0


def test_calibration_audit_keeps_validation_out_of_state_fit(tmp_path: Path) -> None:
    root = tmp_path / "data"
    case = root / "toy_case"
    case.mkdir(parents=True)
    frame_count = 16
    train_end = 12
    original = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]])
    baseline = np.repeat(original[None], frame_count, axis=0)
    observed = baseline.copy()
    observed[1:, :, 0] += 0.005
    tracks = observed[:, :1].copy()
    data = {
        "object_points": observed.astype(np.float32),
        "object_visibilities": np.ones((frame_count, 2), dtype=bool),
        "object_motions_valid": np.ones((frame_count - 1, 2), dtype=bool),
        "surface_points": np.empty((0, 3), dtype=np.float32),
    }
    for path, value in (
        (case / "final_data.pkl", data),
        (case / "inference.pkl", baseline.astype(np.float32)),
        (case / "gt_track_3d.pkl", tracks.astype(np.float32)),
    ):
        with path.open("wb") as handle:
            pickle.dump(value, handle)
    (case / "split.json").write_text(
        json.dumps(
            {
                "frame_len": frame_count,
                "train": [0, train_end],
                "test": [train_end, frame_count],
            }
        ),
        encoding="utf-8",
    )
    (root / "evaluation_subset_manifest.json").write_text(
        json.dumps({"selected_cases": ["toy_case"]}),
        encoding="utf-8",
    )

    result = run_phystwin_calibration_audit(
        root,
        tmp_path / "output",
        protocol=PhysTwinCalibrationProtocol(
            interpolation_neighbors=1,
            coverage_levels=(0.5,),
            bootstrap_samples=20,
            development_cases=(),
        ),
    )

    case_result = result["case_results"]["toy_case"]
    assert case_result["fit_end_frame"] == 9
    assert case_result["calibration_frame_count"] == 3
    assert case_result["future_frame_count"] == 4
    assert case_result["conformal"]["track_error_m"]["posterior_scaled"]["50"][
        "finite_bound"
    ]
    assert set(
        case_result["conformal"]["track_error_m"]["posterior_scaled"]["50"][
            "future_by_horizon"
        ]
    ) == {"early", "middle", "late"}
    assert case_result["nees"]["strict_future_nees_3d"]["count"] == 4
    assert (
        result["confirmation"]["future_point_metrics"]["track_error_m"]["case_count"]
        == 1
    )
    assert Path(result["summary_path"]).exists()
