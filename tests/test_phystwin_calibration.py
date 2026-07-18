import hashlib
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pytest

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


def test_calibration_audit_accepts_hash_validated_external_backbone(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    case = root / "toy_case"
    case.mkdir(parents=True)
    frame_count = 16
    train_end = 12
    initial = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]])
    released = np.repeat(initial[None], frame_count, axis=0)
    external = released.copy()
    external[1:, :, 1] = 0.002
    observed = released.copy()
    observed[1:, :, 0] = 0.005
    tracks = observed[:, :1].copy()
    data = {
        "object_points": observed.astype(np.float32),
        "object_visibilities": np.ones((frame_count, 2), dtype=bool),
        "object_motions_valid": np.ones((frame_count - 1, 2), dtype=bool),
        "surface_points": np.empty((0, 3), dtype=np.float32),
    }
    for path, value in (
        (case / "final_data.pkl", data),
        (case / "inference.pkl", released.astype(np.float32)),
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
        json.dumps({"selected_cases": ["toy_case"]}), encoding="utf-8"
    )
    trajectory = tmp_path / "external.pkl"
    with trajectory.open("wb") as handle:
        pickle.dump(external.astype(np.float32), handle)
    digest = hashlib.sha256(trajectory.read_bytes()).hexdigest()
    manifest = tmp_path / "external_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backbone": {
                    "name": "toy external backbone",
                    "source_repository": "https://example.test/backbone",
                    "source_commit": "a" * 40,
                    "future_observations_used": False,
                    "coordinate_frame": "phystwin-world-metres-v1",
                    "vertex_contract": "phystwin-observed-prefix-then-surface-v1",
                },
                "cases": [
                    {
                        "name": "toy_case",
                        "trajectory": str(trajectory),
                        "sha256": digest,
                        "evidence_end_frame_exclusive": train_end,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = run_phystwin_calibration_audit(
        root,
        tmp_path / "external_output",
        external_backbone_manifest=manifest,
        protocol=PhysTwinCalibrationProtocol(
            interpolation_neighbors=1,
            coverage_levels=(0.5,),
            bootstrap_samples=20,
            development_cases=(),
        ),
    )

    assert result["claim_boundary"]["predictor_update_after_fit"] == "none"
    locked = json.loads(
        (tmp_path / "external_output" / "locked_protocol.json").read_text()
    )
    assert locked["specification"]["baseline"]["kind"] == "external_backbone"
    assert locked["specification"]["baseline"]["manifest"][
        "sha256"
    ] == hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()
    metrics = result["case_results"]["toy_case"]["future_point_metrics"]
    assert metrics["track_error_m"]["baseline_mean_m"] > 0.005

    mismatched_overlay = tmp_path / "mismatched_overlay"
    mismatched_overlay.mkdir()
    (mismatched_overlay / "locked_protocol.json").write_text(
        json.dumps(
            {
                "protocol_id": "b" * 64,
                "specification": {
                    "backbone_manifest": {"sha256": "0" * 64}
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="different hashes"):
        run_phystwin_calibration_audit(
            root,
            tmp_path / "mismatched_output",
            external_backbone_manifest=manifest,
            external_overlay_dir=mismatched_overlay,
            protocol=PhysTwinCalibrationProtocol(
                interpolation_neighbors=1,
                coverage_levels=(0.5,),
                bootstrap_samples=20,
                development_cases=(),
            ),
        )

    external_overlay = tmp_path / "external_overlay"
    anchor_case = external_overlay / "cases" / "toy_case" / "bayesian_anchor"
    anchor_case.mkdir(parents=True)
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (external_overlay / "locked_protocol.json").write_text(
        json.dumps(
            {
                "protocol_id": "c" * 64,
                "specification": {
                    "backbone_manifest": {"sha256": manifest_sha256}
                },
            }
        ),
        encoding="utf-8",
    )
    (anchor_case / "summary.json").write_text(
        json.dumps(
            {
                "config": {
                    "train_end_frame": train_end,
                    "maximum_residual_m": 0.01,
                },
                "selection": {
                    "accepted": True,
                    "selected_candidate": {
                        "process_std_m": 0.005,
                        "observation_std_m": 0.001,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        anchor_case / "posterior.npz",
        mean=np.zeros((2, 3), dtype=float),
        variance=np.full(2, 1e-4, dtype=float),
        lift_indices=np.empty((0, 1), dtype=np.int64),
        lift_weights=np.empty((0, 1), dtype=float),
    )

    operational = run_phystwin_calibration_audit(
        root,
        tmp_path / "operational_output",
        external_backbone_manifest=manifest,
        external_overlay_dir=external_overlay,
        protocol=PhysTwinCalibrationProtocol(
            interpolation_neighbors=1,
            coverage_levels=(0.5,),
            bootstrap_samples=20,
            development_cases=(),
        ),
    )

    future_nees = operational["case_results"]["toy_case"]["nees"][
        "operational_future_nees_3d"
    ]
    assert future_nees["accepted_on_validation"] is True
    assert future_nees["count"] == 4
