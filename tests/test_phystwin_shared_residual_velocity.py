import json
import pickle
from pathlib import Path

import numpy as np

from bayesian_phystwin.phystwin_shared_residual_velocity import (
    SharedResidualVelocityConfig,
    fit_shared_residual_velocity_development,
)


def _write_manifest(
    root: Path,
    *,
    future_offset: float = 0.0,
) -> Path:
    episodes = []
    for index, slope in enumerate((0.001, 0.002, 0.003)):
        case_root = root / f"case_{index}"
        case_root.mkdir(parents=True, exist_ok=True)
        frame_count = 12
        original = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]])
        baseline = np.repeat(original[None], frame_count, axis=0)
        displacement = slope * np.arange(frame_count)
        observed = baseline.copy()
        observed[:, :, 0] += displacement[:, None]
        if index == 0:
            observed[9:, :, 1] += future_offset
        controllers = np.zeros((frame_count, 1, 3))
        controllers[:, 0, 0] = displacement
        data = {
            "object_points": observed.astype(np.float32),
            "object_visibilities": np.ones((frame_count, 2), dtype=bool),
            "object_motions_valid": np.ones((frame_count - 1, 2), dtype=bool),
            "controller_points": controllers.astype(np.float32),
            "surface_points": np.empty((0, 3), dtype=np.float32),
            "interior_points": np.empty((0, 3), dtype=np.float32),
        }
        paths = {
            "final_data": case_root / "final.pkl",
            "baseline_trajectory": case_root / "baseline.pkl",
            "gt_track_3d": case_root / "track.pkl",
        }
        for key, value in (
            ("final_data", data),
            ("baseline_trajectory", baseline.astype(np.float32)),
            ("gt_track_3d", observed[:, :1].astype(np.float32)),
        ):
            with paths[key].open("wb") as handle:
                pickle.dump(value, handle)
        episodes.append(
            {
                "case": f"case_{index}",
                **{key: str(path) for key, path in paths.items()},
                "fit_end_frame": 6,
                "train_end_frame": 9,
            }
        )
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"episodes": episodes}), encoding="utf-8")
    return manifest


def _config(**overrides: object) -> SharedResidualVelocityConfig:
    values = {
        "smoothing_candidates": (1.0,),
        "global_ridge": 1e-6,
        "local_prior_strength_candidates": (100.0,),
        "maximum_training_points": 2,
        "interpolation_neighbors": 1,
        "maximum_velocity_multiplier": 3.0,
        "maximum_residual_m": 0.1,
        "minimum_fold_improvement": 0.001,
        "minimum_development_improvement": 0.001,
    }
    values.update(overrides)
    return SharedResidualVelocityConfig(**values)


def test_shared_velocity_passes_synthetic_cross_episode_gate(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "data")

    summary = fit_shared_residual_velocity_development(
        manifest,
        tmp_path / "output",
        config=_config(),
    )

    assert summary["selection"]["development_gate_passed"]
    assert summary["future_metrics_opened"]
    assert len(summary["future_results"]) == 3
    assert all(
        result["selected_method"] == "shared_residual_velocity"
        for result in summary["future_results"]
    )


def test_failed_shared_gate_keeps_development_future_closed(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "data")

    summary = fit_shared_residual_velocity_development(
        manifest,
        tmp_path / "output",
        config=_config(minimum_development_improvement=2.0),
    )

    assert not summary["selection"]["development_gate_passed"]
    assert not summary["future_metrics_opened"]
    assert summary["future_results"] == []


def test_shared_prediction_does_not_use_future_observations(tmp_path: Path) -> None:
    first_manifest = _write_manifest(tmp_path / "first_data")
    second_manifest = _write_manifest(
        tmp_path / "second_data", future_offset=10.0
    )

    first = fit_shared_residual_velocity_development(
        first_manifest, tmp_path / "first_output", config=_config()
    )
    second = fit_shared_residual_velocity_development(
        second_manifest, tmp_path / "second_output", config=_config()
    )
    first_paths = {
        result["case"]: result["trajectory"] for result in first["future_results"]
    }
    second_paths = {
        result["case"]: result["trajectory"] for result in second["future_results"]
    }
    assert first["selection"] == second["selection"]
    for case in first_paths:
        with Path(first_paths[case]).open("rb") as handle:
            first_trajectory = pickle.load(handle)
        with Path(second_paths[case]).open("rb") as handle:
            second_trajectory = pickle.load(handle)
        np.testing.assert_array_equal(first_trajectory, second_trajectory)
