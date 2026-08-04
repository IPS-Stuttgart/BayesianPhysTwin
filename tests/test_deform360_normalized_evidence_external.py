from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bayesian_phystwin.contracts.fixed_anchor import FixedBayesianAnchorConfigV1
from bayesian_phystwin.endpoint_model_average import (
    ModelAveragedEndpointConfigV1,
    ModelAveragedEndpointPosteriorV1,
)
from scripts.science.run_deform360_normalized_evidence_external import (
    EvaluationLimits,
    _chamfer_rmse,
    _evaluate_hulls,
    _normalized_weights,
    _strict_scale_to_meters,
    _verify_seal,
    evaluate_selection,
    seal_selection,
)

PROTOCOL = Path("protocols/deform360_normalized_evidence_external_v1.json")


def _git_repository(path: Path, tracked_text: str = "") -> Path:
    path.mkdir(parents=True)
    subprocess.run(("git", "init", "-q"), cwd=path, check=True)
    subprocess.run(("git", "config", "user.name", "Test"), cwd=path, check=True)
    subprocess.run(
        ("git", "config", "user.email", "test@example.invalid"),
        cwd=path,
        check=True,
    )
    evidence = path / "docs" / "deform360_history.md"
    evidence.parent.mkdir(parents=True)
    evidence.write_text(tracked_text, encoding="utf-8")
    subprocess.run(("git", "add", "."), cwd=path, check=True)
    subprocess.run(("git", "commit", "-qm", "fixture"), cwd=path, check=True)
    return path


def _write_fixed_archive(path: Path, offset: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.arange(8, dtype=float)[:, None, None]
    track = np.arange(4, dtype=float)[None, :, None]
    direction = np.array([[[0.001, 0.0005, -0.00025]]])
    positions = offset + track * np.array([[[0.002, 0.0, 0.0]]]) + frames * direction
    np.savez_compressed(
        path,
        positions_world_m=positions,
        valid_mask=np.ones(positions.shape[:2], dtype=bool),
    )


def _fresh_ids() -> tuple[str, ...]:
    return (
        "201-alpha",
        "202-beta",
        "203-gamma",
        "204-delta",
        "205-epsilon",
        "206-zeta",
    )


def test_seal_uses_names_only_and_excludes_evidence_mentions(tmp_path: Path) -> None:
    repository = _git_repository(
        tmp_path / "repository",
        "Previously opened object: 207-mentioned-object\n",
    )
    data = tmp_path / "data"
    for object_id in (*_fresh_ids(), "207-mentioned-object", "002-rope-silk"):
        path = data / object_id / "episode-a" / "trajectory.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not an npz; sealing must not open this payload")
    output = tmp_path / "selection.json"

    seal = seal_selection(
        data,
        PROTOCOL,
        output,
        repository_root=repository,
    )

    selected = {entry["object_id"] for entry in seal["selected"]}
    assert selected == set(_fresh_ids())
    assert seal["support_passed"] is True
    assert seal["information_boundary"]["dataset_payload_opened"] is False
    assert "207-mentioned-object" in seal["excluded_object_ids"]
    assert "002-rope-silk" in seal["excluded_object_ids"]
    assert output.is_file()


def test_complete_fixed_identity_evaluation_is_hash_bound(tmp_path: Path) -> None:
    repository = _git_repository(tmp_path / "repository")
    data = tmp_path / "data"
    for index, object_id in enumerate(_fresh_ids()):
        _write_fixed_archive(
            data / object_id / "episode-a" / "trajectory.npz",
            offset=0.01 * index,
        )
    seal_path = tmp_path / "selection.json"
    seal = seal_selection(
        data,
        PROTOCOL,
        seal_path,
        repository_root=repository,
    )
    assert len(seal["selected"]) == 6
    result_path = tmp_path / "result.json"

    result = evaluate_selection(
        seal_path,
        PROTOCOL,
        result_path,
        repository_root=repository,
        limits=EvaluationLimits(
            max_frames_per_archive=8,
            max_tracks=4,
            chamfer_points=4,
            bootstrap_samples=100,
            bootstrap_seed=7,
        ),
    )

    assert result["supported_object_count"] == 6
    assert result["summary"]["gates"]["support_passed"] is True
    assert set(result["summary"]["predictive"]) == {
        "cumulative_evidence_model_average_v1",
        "per_observation_normalized_evidence_model_average_v1",
    }
    assert all(
        case["representation"] == "fixed_identity_trajectory"
        for case in result["cases"]
    )
    stored = json.loads(result_path.read_text(encoding="utf-8"))
    digest = stored.pop("result_sha256")
    from scripts.science.run_deform360_normalized_evidence_external import (
        _canonical_sha256,
    )

    assert digest == _canonical_sha256(stored)


def test_seal_tampering_fails_before_evaluation(tmp_path: Path) -> None:
    repository = _git_repository(tmp_path / "repository")
    data = tmp_path / "data"
    for object_id in _fresh_ids():
        _write_fixed_archive(data / object_id / "episode-a" / "trajectory.npz")
    seal_path = tmp_path / "selection.json"
    seal_selection(data, PROTOCOL, seal_path, repository_root=repository)
    payload = json.loads(seal_path.read_text(encoding="utf-8"))
    payload["selected"][0]["archive_path"] = "201-alpha/replacement/trajectory.npz"
    seal_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="digest changed"):
        _verify_seal(seal_path, PROTOCOL, repository)


def test_normalized_evidence_prevents_synthetic_component_collapse() -> None:
    config = ModelAveragedEndpointConfigV1(
        components=(
            FixedBayesianAnchorConfigV1(process_std_m=0.0),
            FixedBayesianAnchorConfigV1(process_std_m=0.001),
        )
    )
    posterior = ModelAveragedEndpointPosteriorV1(
        mean_m=np.zeros((1, 3)),
        covariance_m2=np.eye(3)[None],
        final_nominal_probability=np.array([0.5]),
        update_count=np.array([100], dtype=np.int64),
        component_weights=np.array([[1.0 - 1e-12, 1e-12]]),
        component_log_evidence=np.array([[0.0, -100.0]]),
        component_mean_m=np.zeros((2, 1, 3)),
        component_variance_m2=np.ones((2, 1)),
        component_process_variance_m2=np.array([0.0, 1e-6]),
        config=config,
        end_frame=100,
    )

    normalized = _normalized_weights(posterior)

    assert normalized.shape == (1, 2)
    assert normalized[0, 1] > 0.25
    assert np.isclose(np.sum(normalized), 1.0)


def test_packed_hull_contract_evaluates_centroid_translation() -> None:
    frames = np.arange(8, dtype=np.int64)
    base = np.array(
        [[-0.01, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]]
    )
    hulls = tuple(base + np.array([0.001 * frame, 0.0, 0.0]) for frame in frames)

    result = _evaluate_hulls(
        (frames, hulls),
        path="201-alpha/episode-a/hulls.npz",
        object_id="201-alpha",
        limits=EvaluationLimits(
            max_frames_per_archive=8,
            max_tracks=16,
            chamfer_points=16,
            bootstrap_samples=10,
            bootstrap_seed=3,
        ),
    )

    assert result is not None
    assert result["representation"] == "packed_visual_hulls"
    assert len(result["steps"]) == 5
    assert result["steps"][0]["centroid_error_m"]["last_supported_residual"] < 1e-12


def test_strict_units_and_chamfer_contracts() -> None:
    assert _strict_scale_to_meters("positions_world_m") == (1.0, "declared_m")
    assert _strict_scale_to_meters("positions_mm") == (1e-3, "declared_mm")
    with pytest.raises(ValueError, match="no declared"):
        _strict_scale_to_meters("positions")

    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert _chamfer_rmse(points, points, maximum_points=2) == 0.0
    with pytest.raises(ValueError, match="nonempty"):
        _chamfer_rmse(np.empty((0, 3)), points, maximum_points=2)


def test_evaluation_limits_fail_closed() -> None:
    with pytest.raises(ValueError, match="max_tracks"):
        EvaluationLimits(max_tracks=0)
    with pytest.raises(ValueError, match="bootstrap_seed"):
        EvaluationLimits(bootstrap_seed=True)  # type: ignore[arg-type]


def test_normalized_weights_preserve_prior_without_updates() -> None:
    config = ModelAveragedEndpointConfigV1(
        components=(
            FixedBayesianAnchorConfigV1(process_std_m=0.0),
            FixedBayesianAnchorConfigV1(process_std_m=0.001),
        ),
        component_prior_probability=(0.25, 0.75),
    )
    posterior = SimpleNamespace(
        config=config,
        update_count=np.array([0], dtype=np.int64),
        component_log_evidence=np.zeros((1, 2)),
    )

    weights = _normalized_weights(posterior)  # type: ignore[arg-type]

    assert np.allclose(weights, [[0.25, 0.75]])
