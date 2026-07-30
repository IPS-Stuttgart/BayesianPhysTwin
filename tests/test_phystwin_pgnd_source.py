import hashlib
import json
import pickle
import subprocess
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.phystwin_pgnd_source import (
    PGNDMetricTransform,
    build_pgnd_gripper_actions,
    evaluate_pgnd_source_prediction,
    interpolate_model_steps,
    physically_supported_contact_trajectory,
    prepare_pgnd_source_input,
    select_pgnd_frames,
    verify_clean_git_checkout,
    verify_pgnd_assets,
)


def test_frame_selection_is_prefix_only_and_covers_final_frame() -> None:
    selection = select_pgnd_frames(
        train_end_exclusive=59,
        frame_count=85,
    )

    assert selection.initialization_frame == 57
    assert selection.history_frames == (51, 54)
    assert selection.prediction_frames == tuple(range(60, 85, 3))
    assert max(selection.history_frames) < 59
    assert selection.prediction_frames[-1] == 84


def test_frame_selection_rejects_uncovered_episode_tail() -> None:
    with pytest.raises(ValueError, match="land exactly"):
        select_pgnd_frames(train_end_exclusive=43, frame_count=62)


def test_metric_transform_is_metric_and_prefix_determined() -> None:
    current = np.array(
        [
            [-0.2, -0.1, -0.3],
            [0.2, 0.1, 0.0],
            [0.0, 0.0, -0.1],
        ]
    )
    transform = PGNDMetricTransform.fit(current)
    encoded = transform.positions_to_model(current)
    decoded = transform.positions_to_world(encoded)

    np.testing.assert_allclose(decoded, current, atol=1e-15)
    np.testing.assert_allclose(
        np.linalg.norm(encoded[0] - encoded[1]),
        np.linalg.norm(current[0] - current[1]),
    )
    assert np.min(encoded[:, 1]) == pytest.approx(0.04001)


def test_contact_selection_uses_physical_prior_not_observation() -> None:
    controller = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ]
    )
    physical = np.array(
        [
            [[0.9, 0.0, 0.0]],
            [[0.1, 0.0, 0.0]],
        ]
    )

    selected, indices = physically_supported_contact_trajectory(controller, physical)

    np.testing.assert_array_equal(indices, [1, 0])
    np.testing.assert_array_equal(selected, [controller[0, 1], controller[1, 0]])


def test_gripper_action_rows_preserve_metric_velocity() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
    actions = build_pgnd_gripper_actions(positions, dt_s=0.1, radius_m=0.04)

    assert actions.shape == (2, 1, 15)
    np.testing.assert_allclose(actions[:, 0, :3], positions)
    np.testing.assert_allclose(actions[:, 0, 3], 0.1)
    np.testing.assert_allclose(actions[:, 0, 13], 0.04)
    np.testing.assert_allclose(actions[:, 0, 14], 0.0)


def test_model_step_interpolation_preserves_prefix_and_anchors() -> None:
    physical = np.zeros((7, 2, 3), dtype=float)
    physical[:, :, 0] = np.arange(7)[:, None]
    predictions = np.array(
        [
            np.full((2, 3), 30.0),
            np.full((2, 3), 60.0),
        ]
    )

    result = interpolate_model_steps(
        physical_prefix=physical,
        model_prediction_frames=(3, 6),
        model_predictions=predictions,
        initialization_frame=0,
        frame_count=7,
    )

    np.testing.assert_array_equal(result[0], physical[0])
    np.testing.assert_array_equal(result[3], predictions[0])
    np.testing.assert_array_equal(result[6], predictions[1])
    np.testing.assert_allclose(result[1], 10.0)
    np.testing.assert_allclose(result[5], 50.0)


def test_asset_verification_pins_commit_checkpoint_and_config(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "pgnd"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.name", "Test"], check=True
    )
    (checkout / "source").write_text("pinned\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "source"], check=True)
    subprocess.run(["git", "-C", str(checkout), "commit", "-qm", "Pinned"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = tmp_path / "hydra.yaml"
    config.write_bytes(b"config")

    result = verify_pgnd_assets(
        checkout,
        checkpoint,
        config,
        expected_commit=commit,
        expected_checkpoint_sha256=hashlib.sha256(b"checkpoint").hexdigest(),
        expected_config_sha256=hashlib.sha256(b"config").hexdigest(),
    )

    assert result["commit"] == commit
    assert result["clean"]
    with pytest.raises(ValueError, match="config mismatch"):
        verify_pgnd_assets(
            checkout,
            checkpoint,
            config,
            expected_commit=commit,
            expected_checkpoint_sha256=hashlib.sha256(b"checkpoint").hexdigest(),
            expected_config_sha256="0" * 64,
        )
    (checkout / "source").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dirty"):
        verify_clean_git_checkout(checkout, expected_commit=commit)


def _write_source_case(root: Path, future_offset: float) -> tuple[Path, ...]:
    frame_count = 10
    controller = np.zeros((frame_count, 2, 3), dtype=np.float32)
    physical = np.zeros((frame_count, 6, 3), dtype=np.float32)
    object_points = np.zeros((frame_count, 5, 3), dtype=np.float32)
    object_points[6:, :, 0] = future_offset
    final_data = {
        "controller_points": controller,
        "object_points": object_points,
        "object_visibilities": np.ones((frame_count, 5), dtype=bool),
        "surface_points": np.empty((0, 3), dtype=np.float32),
    }
    final_path = root / "final.pkl"
    physical_path = root / "physical.pkl"
    split_path = root / "split.json"
    root.mkdir(parents=True)
    with final_path.open("wb") as handle:
        pickle.dump(final_data, handle)
    with physical_path.open("wb") as handle:
        pickle.dump(physical, handle)
    split_path.write_text(
        json.dumps({"train": [0, 7], "test": [7, 10]}),
        encoding="utf-8",
    )
    return final_path, physical_path, split_path


def test_prepared_carrier_excludes_future_observations(tmp_path: Path) -> None:
    first = _write_source_case(tmp_path / "first", future_offset=0.0)
    second = _write_source_case(tmp_path / "second", future_offset=100.0)
    first_output = tmp_path / "first.npz"
    second_output = tmp_path / "second.npz"

    prepare_pgnd_source_input(
        final_data_path=first[0],
        physical_trajectory_path=first[1],
        split_path=first[2],
        output_path=first_output,
    )
    prepare_pgnd_source_input(
        final_data_path=second[0],
        physical_trajectory_path=second[1],
        split_path=second[2],
        output_path=second_output,
    )

    with (
        np.load(first_output) as first_archive,
        np.load(second_output) as second_archive,
    ):
        assert "object_points" not in first_archive.files
        assert "object_visibilities" not in first_archive.files
        assert int(first_archive["num_surface_points"]) == 5
        assert set(first_archive.files) == set(second_archive.files)
        for name in first_archive.files:
            np.testing.assert_array_equal(first_archive[name], second_archive[name])


def test_source_gate_requires_both_metrics_to_beat_full_physical() -> None:
    frame_count = 10
    node_count = 4
    query = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.1, 0.1, 0.0],
        ]
    )
    truth = np.repeat(query[None], frame_count, axis=0)
    full_physical = truth.copy()
    full_physical[7:, :, 0] += 0.01
    candidate = truth.copy()
    equal_physical = full_physical.copy()
    persistence = np.repeat(truth[6:7], frame_count, axis=0)
    final_data = {
        "object_points": truth.copy(),
        "object_visibilities": np.ones((frame_count, node_count), dtype=bool),
        "surface_points": np.empty((0, 3)),
    }

    result = evaluate_pgnd_source_prediction(
        candidate_trajectory=candidate,
        equal_support_physical=equal_physical,
        equal_support_surface_count=node_count,
        full_physical=full_physical,
        persistence_trajectory=persistence,
        final_data=final_data,
        gt_track_3d=truth[:, :1],
        train_end_exclusive=7,
        test_end_exclusive=10,
        required_relative_improvement=0.02,
    )

    assert result["gate"]["passed"]
