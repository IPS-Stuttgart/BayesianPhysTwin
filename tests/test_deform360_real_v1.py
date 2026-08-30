from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "deform360_real_v1"
    / "run.py"
)
SPEC = importlib.util.spec_from_file_location("deform360_real_v1_run", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def protocol(root: Path) -> dict:
    return {
        "schema": "bayesian-phystwin/deform360-real-evaluation-protocol-v1",
        "schema_version": 1,
        "protocol_id": "test-deform360-real-v1",
        "dataset_root": str(root.resolve()),
        "official_processing_repository": "lhy0807/deform360",
        "official_processing_revision": "d8522a4403b766aeb387510c04e89032a56fdf35",
        "development_objects": ["001-rope", "002-rope-silk"],
        "reserved_objects": ["066-glove-half-black-cloth"],
        "profiles": {
            "pilot": {
                "max_cases": 4,
                "max_frames": 32,
                "max_points": 32,
                "max_tactile_channels": 32,
                "max_candidate_archives": 16,
            }
        },
        "model": {
            "velocity_lags": [1, 2, 4, 8],
            "minimum_prefix_steps": 10,
            "gibbs_temperature_floor_fraction": 0.05,
            "variance_floor_fraction": 1e-6,
            "marginal_coverage_probability": 0.9,
        },
        "analysis": {
            "bootstrap_repetitions": 100,
            "bootstrap_seed": 4,
            "aggregation_unit": "carrier",
            "selection_rule": "names only",
        },
        "information_boundary": {
            "public_real_measurements": True,
            "rolling_prefix_only_prediction": True,
            "future_used_for_scoring_only": True,
            "reserved_objects_opened": False,
            "method_parameters_fit_to_evaluated_future": False,
            "raw_payload_uploaded": False,
            "fresh_confirmation_authorized": False,
            "paper_claim_authorized": False,
        },
    }


def save_protocol(tmp_path: Path, root: Path) -> Path:
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol(root)), encoding="utf-8")
    return path


def moving_points(frames: int = 24, points: int = 8) -> np.ndarray:
    rng = np.random.default_rng(7)
    initial = rng.normal(scale=0.02, size=(points, 3))
    velocity = rng.normal(scale=0.001, size=(points, 3))
    time = np.arange(frames)[:, None, None]
    return initial[None] + time * velocity[None]


def test_prediction_is_invariant_to_future_poisoning() -> None:
    clean = moving_points()
    poisoned = clean.copy()
    poisoned[13:] += 1000.0
    valid = np.ones(clean.shape[:2], dtype=bool)
    before = module.prediction_for_step(
        clean,
        valid,
        frame=12,
        lags=(1, 2, 4, 8),
        floor_fraction=1e-6,
        temperature_floor_fraction=0.05,
    )
    after = module.prediction_for_step(
        poisoned,
        valid,
        frame=12,
        lags=(1, 2, 4, 8),
        floor_fraction=1e-6,
        temperature_floor_fraction=0.05,
    )
    np.testing.assert_array_equal(before.bayesian, after.bayesian)
    np.testing.assert_array_equal(before.weights, after.weights)
    np.testing.assert_array_equal(before.diagonal, after.diagonal)
    np.testing.assert_array_equal(before.factors, after.factors)


def test_fixed_identity_npz_end_to_end(tmp_path: Path) -> None:
    root = tmp_path / "deform360"
    carrier = root / "processed-repository" / "001-rope" / "episode_0000"
    carrier.mkdir(parents=True)
    np.savez_compressed(
        carrier / "particle_trajectory.npz",
        positions_world_m=moving_points(),
        valid_mask=np.ones((24, 8), dtype=bool),
    )
    result = module.run(
        data_root=root,
        protocol_path=save_protocol(tmp_path, root),
        output_dir=tmp_path / "output",
        profile_name="pilot",
        revision="a" * 40,
    )
    assert result["selection"]["evaluated_count"] == 1
    summary = result["summary"]["fixed_identity_3d"]
    assert summary["case_count"] == 1
    assert summary["mean_primary_error"]["bayesian"] < 1e-8
    assert result["claim_authorized"] is False


def test_pcd_clean_sequence_is_scored(tmp_path: Path) -> None:
    root = tmp_path / "deform360"
    pcd = (
        root
        / "processed-repository"
        / "001-rope"
        / "episode_0000"
        / "pcd_clean"
    )
    pcd.mkdir(parents=True)
    base = moving_points(frames=1, points=20)[0]
    for frame in range(24):
        points = base + np.array([frame * 0.001, 0.0, 0.0])
        np.savez_compressed(pcd / f"{frame:06d}.npz", pts=points)
    result = module.run(
        data_root=root,
        protocol_path=save_protocol(tmp_path, root),
        output_dir=tmp_path / "output",
        profile_name="pilot",
        revision=None,
    )
    case = result["cases"][0]
    assert case["representation"] == "pcd_clean_centroid_3d"
    assert case["metrics"]["last_residual_chamfer_mm"] < 1e-8
    assert case["metrics"]["bayesian_chamfer_mm"] < 1e-8


def test_headerless_tactile_fallback_is_real_measurement_carrier(
    tmp_path: Path,
) -> None:
    root = tmp_path / "deform360"
    tactile = (
        root
        / "raw-repository"
        / "raw"
        / "001-rope"
        / "brics-odroid_tactilel_left"
    )
    tactile.mkdir(parents=True)
    values = np.zeros((40, 16, 32), dtype=np.float32)
    values[10:] = np.arange(30, dtype=np.float32)[:, None, None] * 0.01
    values.tofile(tactile / "sensor_123.npy")
    result = module.run(
        data_root=root,
        protocol_path=save_protocol(tmp_path, root),
        output_dir=tmp_path / "output",
        profile_name="pilot",
        revision=None,
    )
    case = result["cases"][0]
    assert case["representation"] == "raw_tactile_field"
    assert case["unit"] == "normalized_tactile"
    assert 0.0 <= case["metrics"]["bayesian_marginal_90_coverage"] <= 1.0


def test_reserved_objects_are_not_opened(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "deform360"
    reserved = (
        root
        / "processed-repository"
        / "066-glove-half-black-cloth"
        / "episode_0000"
    )
    allowed = root / "processed-repository" / "001-rope" / "episode_0000"
    reserved.mkdir(parents=True)
    allowed.mkdir(parents=True)
    np.savez_compressed(reserved / "trajectory.npz", positions_world_m=moving_points())
    np.savez_compressed(allowed / "trajectory.npz", positions_world_m=moving_points())

    opened: list[Path] = []
    original = module.load_trajectory_npz

    def recording_loader(carrier, profile_value, root_value):
        opened.append(carrier.path)
        return original(carrier, profile_value, root_value)

    monkeypatch.setattr(module, "load_trajectory_npz", recording_loader)
    result = module.run(
        data_root=root,
        protocol_path=save_protocol(tmp_path, root),
        output_dir=tmp_path / "output",
        profile_name="pilot",
        revision=None,
    )
    assert all("066-glove-half-black-cloth" not in str(path) for path in opened)
    assert result["selection"]["reserved_object_overlap"] == []
