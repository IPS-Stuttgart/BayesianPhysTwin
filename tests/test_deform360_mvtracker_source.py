import io
from pathlib import Path

import numpy as np

from bayesian_phystwin.deform360_mvtracker_source import (
    _NumpyCompatibilityUnpickler,
    _compatible_pickle_module,
    MVTrackerSourceConfig,
    exact_anchor_trajectory,
    metric_observation_variance_m2,
    score_competence_arrays,
    seal_prediction,
    write_prediction_artifact,
)


def test_numpy_two_private_pickle_namespace_resolves_when_available() -> None:
    unpickler = _NumpyCompatibilityUnpickler(io.BytesIO())

    resolved = unpickler.find_class("numpy._core.numeric", "_frombuffer")

    assert resolved is np._core.numeric._frombuffer


def test_numpy_two_private_pickle_namespace_has_legacy_fallback(
    monkeypatch,
) -> None:
    def missing_private_module(module: str) -> None:
        assert module == "numpy._core.numeric"
        raise ModuleNotFoundError(module)

    monkeypatch.setattr(
        "bayesian_phystwin.deform360_mvtracker_source.importlib.import_module",
        missing_private_module,
    )

    assert (
        _compatible_pickle_module("numpy._core.numeric")
        == "numpy.core.numeric"
    )


def _small_config() -> MVTrackerSourceConfig:
    return MVTrackerSourceConfig(
        prefix_frame_count=3,
        update_frame=2,
        center_ids=(0, 1, 2),
        selected_cameras=("camera-a", "camera-b"),
    )


def test_exact_anchor_preserves_predicted_displacements() -> None:
    raw = np.asarray(
        [
            [[0.01, 0.0, 0.0], [1.0, 0.02, 0.0]],
            [[0.02, 0.0, 0.0], [1.0, 0.03, 0.0]],
        ],
        dtype=np.float32,
    )
    initial = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)

    anchored, correction = exact_anchor_trajectory(raw, initial)

    np.testing.assert_array_equal(anchored[0], initial)
    np.testing.assert_allclose(anchored[1] - anchored[0], raw[1] - raw[0])
    np.testing.assert_allclose(correction, initial - raw[0])


def test_metric_variance_uses_visibility_not_state_innovation() -> None:
    visibility = np.asarray([[1.0, 0.5], [0.25, 0.1]])
    correction = np.asarray([[0.0, 0.0, 0.0], [0.003, 0.0, 0.0]])

    first = metric_observation_variance_m2(
        visibility,
        correction,
        standard_deviation_floor_m=0.005,
    )
    second = metric_observation_variance_m2(
        visibility,
        correction,
        standard_deviation_floor_m=0.005,
    )

    np.testing.assert_array_equal(first, second)
    assert first[1, 0] > first[0, 0]
    assert first[1, 1] > first[0, 1]
    assert np.all(first >= 0.005**2)


def test_prediction_artifact_is_sealed_without_target_input(tmp_path: Path) -> None:
    config = _small_config()
    raw = np.zeros((3, 3, 3), dtype=np.float32)
    raw[:, :, 0] = np.arange(3)[:, None] * 0.01
    frame_zero = np.zeros((3, 3), dtype=np.float32)
    physical = np.zeros((4, 3, 3), dtype=np.float32)
    persistence = np.zeros_like(physical)
    output = tmp_path / "prediction"

    report = write_prediction_artifact(
        output,
        raw_tracker_m=raw,
        visibility_probability=np.ones((3, 3), dtype=np.float32),
        physical_prior_m=physical,
        persistence_m=persistence,
        frame_zero_points_m=frame_zero,
        input_provenance={"rgb_prefix_sha256": "a" * 64},
        runtime_provenance={"device": "test"},
        config=config,
    )
    seal = seal_prediction(output, config=config)

    assert report["information_boundary"]["source_target_read"] is False
    assert report["information_boundary"]["deployable_predictive_observation"] is False
    assert seal["information_boundary"][
        "prediction_hashed_before_source_target_loading"
    ]


def test_competence_gate_passes_accurate_well_supported_tracker() -> None:
    config = _small_config()
    target = np.zeros((3, 3, 3), dtype=np.float32)
    target[1:, :, 0] = 0.02
    tracker = target.copy()
    tracker[1:, :, 0] += 0.001
    physical = np.zeros_like(target)
    persistence = np.zeros_like(target)

    result = score_competence_arrays(
        tracker_m=tracker,
        tracker_visibility_probability=np.ones((3, 3), dtype=np.float32),
        physical_prior_centers_m=physical,
        persistence_centers_m=persistence,
        target_centers_m=target,
        target_visibility=np.ones((3, 3), dtype=bool),
        target_validity=np.ones((3, 3), dtype=bool),
        config=config,
    )

    assert result["passed"]
    assert result["supported_fraction"] == 1.0
    assert result["relative_gain_over_best_baseline"] > 0.9


def test_competence_gate_rejects_visibility_selection() -> None:
    config = _small_config()
    target = np.zeros((3, 3, 3), dtype=np.float32)
    visibility = np.zeros((3, 3), dtype=np.float32)
    visibility[1:, 0] = 1.0

    result = score_competence_arrays(
        tracker_m=target,
        tracker_visibility_probability=visibility,
        physical_prior_centers_m=np.ones_like(target) * 0.01,
        persistence_centers_m=np.ones_like(target) * 0.01,
        target_centers_m=target,
        target_visibility=np.ones((3, 3), dtype=bool),
        target_validity=np.ones((3, 3), dtype=bool),
        config=config,
    )

    assert not result["passed"]
    assert not result["gates"]["supported_fraction"]


def test_competence_gate_records_zero_tracker_support() -> None:
    config = _small_config()
    target = np.zeros((3, 3, 3), dtype=np.float32)

    result = score_competence_arrays(
        tracker_m=target,
        tracker_visibility_probability=np.zeros((3, 3), dtype=np.float32),
        physical_prior_centers_m=np.ones_like(target) * 0.01,
        persistence_centers_m=np.ones_like(target) * 0.01,
        target_centers_m=target,
        target_visibility=np.ones((3, 3), dtype=bool),
        target_validity=np.ones((3, 3), dtype=bool),
        config=config,
    )

    assert not result["passed"]
    assert result["supported_identity_frames"] == 0
    assert result["scores"]["mvtracker_identity_rmse_m"] is None
