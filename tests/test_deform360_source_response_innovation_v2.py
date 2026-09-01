from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "science"
    / "run_deform360_source_response_innovation_v2.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_deform360_source_response_innovation_v2", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_robust_dynamic_features_remove_static_preload() -> None:
    values = np.full((40, 16, 32), 2.0, dtype=np.float64)
    values[25:, 4:8, 10:14] += 0.5
    features, activity, names, records = (
        MODULE._robust_dynamic_tactile_features(
            {"sensor": values},
            baseline_frame_count=12,
            mad_threshold=3.0,
            minimum_noise_scale=0.01,
            maximum_standardized_residual=25.0,
        )
    )
    assert features.shape == (40, 4)
    assert len(names) == 4
    assert np.all(activity[:25] == 0.0)
    assert np.all(activity[25:] > 0.0)
    assert records[0]["dynamic_frame_count"] == 15
    assert np.all(features[:25] == 0.0)
    assert np.all(features[25:, 0] > 0.0)


def test_action_history_features_use_causal_past_and_horizon_exposure() -> None:
    primitives = np.arange(20, dtype=np.float64)[:, None]
    openings = np.column_stack([np.arange(21, dtype=np.float64)])
    times = np.array([8, 10], dtype=np.int64)
    action, exposure, names, gated_names = MODULE._action_history_features(
        primitives,
        openings,
        times,
        horizon=2,
        history_windows=[1, 2, 4, 8],
    )
    np.testing.assert_allclose(exposure[:, 0], [17.0, 21.0])
    # Row layout: exposure, w1, w2, w4, w8, current opening.
    np.testing.assert_allclose(
        action[0], [17.0, 7.0, 6.5, 5.5, 3.5, 8.0]
    )
    np.testing.assert_allclose(
        action[1], [21.0, 9.0, 8.5, 7.5, 5.5, 10.0]
    )
    assert len(names) == action.shape[1]
    assert len(gated_names) == exposure.shape[1]


def test_block_permutation_preserves_rows_within_each_split() -> None:
    values = np.arange(120, dtype=np.float64).reshape(60, 2)
    splits = [slice(0, 20), slice(20, 40), slice(40, 60)]
    permuted = MODULE._block_permute(
        values, splits, block_size=4, seed=20260902
    )
    assert not np.array_equal(permuted, values)
    for split in splits:
        assert sorted(map(tuple, permuted[split].tolist())) == sorted(
            map(tuple, values[split].tolist())
        )


def test_full_covariance_metrics_use_cross_output_dependence() -> None:
    rng = np.random.default_rng(4)
    design = MODULE._with_intercept(rng.normal(size=(180, 5)))
    coefficient = rng.normal(size=(design.shape[1], 3))
    covariance = np.array(
        [[0.3, 0.22, 0.0], [0.22, 0.3, 0.04], [0.0, 0.04, 0.2]]
    )
    targets = design @ coefficient + rng.multivariate_normal(
        np.zeros(3), covariance, size=len(design)
    )
    model = MODULE._fit(
        design[:120], targets[:120], ridge=0.01, eigenvalue_floor=1e-8
    )
    means, predicted_covariance = MODULE._predict(
        model, design[120:], covariance_scale=1.0
    )
    full = MODULE._probabilistic_metrics(
        targets[120:], means, predicted_covariance, diagonal=False
    )
    diagonal = MODULE._probabilistic_metrics(
        targets[120:], means, predicted_covariance, diagonal=True
    )
    assert full["nll_per_dimension"] < diagonal["nll_per_dimension"]
    assert np.all(np.linalg.eigvalsh(predicted_covariance) > 0.0)


def _write_synthetic_episode(root: Path, frame_count: int = 180) -> None:
    robot_root = root / "robot"
    robot_root.mkdir(parents=True)
    rng = np.random.default_rng(18)
    action = rng.normal(scale=0.002, size=frame_count - 1)
    action = np.convolve(action, np.ones(3) / 3.0, mode="same")
    positions = np.concatenate([[0.0], np.cumsum(action)])
    transforms = np.repeat(np.eye(4)[None, :, :], frame_count, axis=0)
    transforms[:, 0, 3] = positions
    openings = 0.04 + 0.01 * np.sin(
        np.linspace(0.0, 8.0, frame_count)
    )
    np.savez(
        robot_root / "robot.npz",
        T_worlds=transforms,
        openings=openings,
        bimanual=np.asarray(False),
        format_version=np.asarray(1),
    )

    latent = np.zeros(frame_count)
    for frame in range(1, frame_count):
        latent[frame] = (
            0.88 * latent[frame - 1] + 120.0 * action[frame - 1]
        )
    sensors = ["s0", "s1", "s2", "s3"]
    for sensor_index, sensor in enumerate(sensors):
        directory = root / sensor
        directory.mkdir()
        values = np.full(
            (frame_count, 16, 32), 0.25 + 0.02 * sensor_index
        )
        patch = np.maximum(
            0.0, latent + 0.15 + 0.02 * sensor_index
        )
        values[
            :, 3 + sensor_index : 6 + sensor_index, 8:12
        ] += patch[:, None, None]
        values += rng.normal(scale=0.001, size=values.shape)
        np.save(
            directory / "synced_tactile.npy",
            values.astype(np.float32),
        )


def test_end_to_end_synthetic_source_keeps_target_closed(
    tmp_path: Path,
) -> None:
    episode = tmp_path / "038-mat-cloth" / "episode_0003"
    _write_synthetic_episode(episode)
    protocol = {
        "schema": "test/deform360-source-response-innovation-v2",
        "source_object": "038-mat-cloth",
        "source_episode": 3,
        "tactile_preprocessing": {
            "sensors": ["s0", "s1", "s2", "s3"],
            "baseline_prefix_fraction_of_train": 0.2,
            "minimum_baseline_frames": 16,
            "mad_threshold": 3.0,
            "minimum_noise_scale": 0.001,
            "maximum_standardized_residual": 25.0,
            "minimum_increment_train_std": 1e-5,
        },
        "design": {
            "train_fraction": 0.6,
            "calibration_fraction": 0.2,
            "prediction_horizon_grid_frames": [1, 2],
            "history_windows": [1, 2, 4],
            "ridge_grid": [0.01, 1.0, 100.0],
            "covariance_scale_grid": [0.5, 1.0, 2.0, 4.0],
            "covariance_eigenvalue_floor": 1e-6,
            "standardization_minimum_scale": 1e-8,
            "permutation_block_frames": 6,
            "permutation_seed": 7,
        },
        "qualification_gates": {
            "minimum_selected_output_dimension": 4,
            "minimum_dynamic_frame_fraction": 0.05,
            "maximum_dynamic_frame_fraction": 1.0,
            "minimum_train_transitions": 80,
            "minimum_calibration_transitions": 20,
            "minimum_test_transitions": 20,
            "minimum_gated_action_vs_state_nll_gain_per_dimension": 0.0,
            "minimum_gated_vs_ungated_nll_gain_per_dimension": 0.0,
            "minimum_full_vs_diagonal_nll_gain_per_dimension": 0.0,
            "minimum_true_vs_permuted_action_nll_gain_per_dimension": 0.0,
            "maximum_gated_rmse_ratio_to_zero_innovation": 1.05,
            "minimum_normalized_joint_nees": 0.5,
            "maximum_normalized_joint_nees": 2.0,
            "minimum_marginal_coverage": 0.8,
            "maximum_marginal_coverage": 0.98,
        },
        "statistical_scope": "synthetic test",
        "claim_boundary": "no target",
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    output = tmp_path / "output"
    subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "--source-episode-root",
            str(episode),
            "--protocol",
            str(protocol_path),
            "--output-dir",
            str(output),
            "--repository",
            "test/repository",
            "--revision",
            "0" * 40,
            "--workflow-run-id",
            "1",
            "--workflow-run-attempt",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads((output / "result.json").read_text())
    assert result["decision"] in {
        "source-response-innovation-qualified",
        "source-response-innovation-not-qualified",
    }
    assert result["horizon_selection"]["selected_horizon_frames"] in {1, 2}
    assert result["tactile_preprocessing"]["selected_output_dimension"] >= 4
    boundary = result["information_boundary"]
    assert boundary["source_robot_payload_opened"] is True
    assert boundary["source_tactile_payloads_opened"] is True
    assert boundary["source_camera_pixels_opened"] is False
    assert boundary["target_directory_contents_listed"] is False
    assert boundary["target_numeric_payload_opened"] is False
    assert boundary["target_scoring_performed"] is False
    assert boundary["paper_claim_authorized"] is False
