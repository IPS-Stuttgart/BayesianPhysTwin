import hashlib
import pickle
import subprocess
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.phystwin_pgrd_adapter import (
    MetricNormalizer,
    PhysTwinPGRDAdapterConfig,
    compose_dense_endpoint_with_sampled_dynamics,
    deterministic_farthest_point_sample,
    fit_pgrd_residual_adapter,
    rollout_pgrd_correction,
    verify_pgrd_assets,
)


class _ZeroPredictor:
    def reset(self) -> None:
        return None

    def predict(
        self,
        x: np.ndarray,
        v: np.ndarray,
        x_history: np.ndarray,
        v_history: np.ndarray,
        x_sim: np.ndarray,
        v_sim: np.ndarray,
    ) -> np.ndarray:
        del v, x_history, v_history, x_sim, v_sim
        return np.zeros_like(x)


class _ConstantPredictor(_ZeroPredictor):
    def predict(
        self,
        x: np.ndarray,
        v: np.ndarray,
        x_history: np.ndarray,
        v_history: np.ndarray,
        x_sim: np.ndarray,
        v_sim: np.ndarray,
    ) -> np.ndarray:
        del v, x_history, v_history, x_sim, v_sim
        return np.full_like(x, 100.0)


def test_farthest_point_sampling_is_deterministic_and_spans_geometry() -> None:
    points = np.array(
        [
            [-2.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )

    first = deterministic_farthest_point_sample(points, 3)
    second = deterministic_farthest_point_sample(points, 3)

    np.testing.assert_array_equal(first, second)
    assert set(first[:2]) == {0, 4}
    assert len(set(first)) == 3


def test_metric_normalization_round_trip() -> None:
    points = np.array([[-0.2, 0.1, 0.0], [0.2, 0.0, 0.1]])
    normalizer = MetricNormalizer.fit(
        points, normalized_extent=0.5, yaw_degrees=90.0
    )

    encoded = normalizer.positions_to_model(points)
    decoded = normalizer.positions_to_metric(encoded)

    np.testing.assert_allclose(decoded, points, atol=1e-15)
    assert np.max(np.ptp(encoded, axis=0)) == pytest.approx(0.5)


def test_asset_verification_pins_commit_and_checkpoint(tmp_path: Path) -> None:
    checkout = tmp_path / "pgrd"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.name", "Test"], check=True
    )
    source = checkout / "source.txt"
    source.write_text("pinned\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "source.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "commit", "-q", "-m", "Pinned"], check=True
    )
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    checksum = hashlib.sha256(b"checkpoint").hexdigest()

    provenance = verify_pgrd_assets(
        checkout,
        checkpoint,
        expected_commit=commit,
        expected_checkpoint_sha256=checksum,
    )

    assert provenance["commit"] == commit
    assert provenance["checkpoint_sha256"] == checksum
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        verify_pgrd_assets(
            checkout,
            checkpoint,
            expected_commit=commit,
            expected_checkpoint_sha256="0" * 64,
        )


def test_zero_trust_is_exact_endpoint_persistence() -> None:
    baseline = np.zeros((8, 4, 3), dtype=float)
    baseline[:, :, 0] = np.arange(8)[:, None] * 0.001
    baseline[:, :, 1] = np.arange(4)[None] * 0.01
    prefix = baseline[:5].copy()
    prefix[:, :, 2] += 0.004
    sample_indices = np.arange(4)
    normalizer = MetricNormalizer.fit(baseline[0], normalized_extent=0.5)

    correction = rollout_pgrd_correction(
        baseline,
        prefix,
        sample_indices,
        _ConstantPredictor(),
        normalizer,
        start_frame=5,
        end_frame=8,
        history_length=2,
        temporal_warmup_steps=1,
        simulation_dt=0.1,
        model_frame_stride=1,
        trust=0.0,
        maximum_residual_m=0.01,
    )

    expected = np.zeros_like(correction)
    expected[:, :, 2] = 0.004
    np.testing.assert_array_equal(correction, expected)


def test_dense_endpoint_is_preserved_when_sampled_dynamics_are_zero() -> None:
    endpoint = np.arange(18, dtype=float).reshape(6, 3) * 0.001
    sample_indices = np.array([0, 3, 5])
    sampled = np.repeat(endpoint[sample_indices][None], 4, axis=0)
    interpolation_indices = np.array([[0], [0], [1], [1], [2], [2]])
    interpolation_weights = np.ones((6, 1))

    dense = compose_dense_endpoint_with_sampled_dynamics(
        sampled,
        endpoint,
        sample_indices,
        interpolation_indices,
        interpolation_weights,
    )

    np.testing.assert_array_equal(dense, np.repeat(endpoint[None], 4, axis=0))


def _write_case(
    root: Path, *, future_observation_offset: float = 0.0
) -> tuple[Path, Path, Path]:
    frame_count = 12
    original = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.02, 0.00, 0.00],
            [0.00, 0.02, 0.00],
            [0.02, 0.02, 0.00],
        ]
    )
    baseline = np.repeat(original[None], frame_count, axis=0)
    observed = baseline.copy()
    observed[:, :, 2] += 0.004
    observed[8:, :, 1] += future_observation_offset
    data = {
        "object_points": observed.astype(np.float32),
        "object_visibilities": np.ones((frame_count, 4), dtype=bool),
        "object_motions_valid": np.ones((frame_count - 1, 4), dtype=bool),
        "controller_points": np.zeros((frame_count, 1, 3), dtype=np.float32),
        "surface_points": np.empty((0, 3), dtype=np.float32),
        "interior_points": np.empty((0, 3), dtype=np.float32),
    }
    gt_track = np.repeat((original[:1] + [0.0, 0.0, 0.004])[None], frame_count, axis=0)
    root.mkdir(parents=True)
    paths = (root / "final.pkl", root / "baseline.pkl", root / "track.pkl")
    for path, value in zip(paths, (data, baseline.astype(np.float32), gt_track)):
        with path.open("wb") as handle:
            pickle.dump(value, handle)
    return paths


def _config() -> PhysTwinPGRDAdapterConfig:
    return PhysTwinPGRDAdapterConfig(
        fit_end_frame=5,
        train_end_frame=8,
        normalized_extent_candidates=(0.5,),
        yaw_candidates_degrees=(0.0,),
        trust_candidates=(0.5,),
        number_of_points=4,
        interpolation_neighbors=1,
        minimum_dynamic_improvement=0.01,
    )


def test_future_observation_mutation_cannot_change_prediction(tmp_path: Path) -> None:
    first = _write_case(tmp_path / "first")
    second = _write_case(tmp_path / "second", future_observation_offset=10.0)

    first_summary = fit_pgrd_residual_adapter(
        *first,
        tmp_path / "first_output",
        config=_config(),
        predictor=_ZeroPredictor(),
    )
    second_summary = fit_pgrd_residual_adapter(
        *second,
        tmp_path / "second_output",
        config=_config(),
        predictor=_ZeroPredictor(),
    )
    with Path(first_summary["outputs"]["trajectory"]).open("rb") as handle:
        first_trajectory = pickle.load(handle)
    with Path(second_summary["outputs"]["trajectory"]).open("rb") as handle:
        second_trajectory = pickle.load(handle)

    assert first_summary["selection"] == second_summary["selection"]
    assert first_summary["selection"]["selected_method"] == "persistence"
    assert not first_summary["test"]["future_metrics_opened"]
    np.testing.assert_array_equal(first_trajectory, second_trajectory)
