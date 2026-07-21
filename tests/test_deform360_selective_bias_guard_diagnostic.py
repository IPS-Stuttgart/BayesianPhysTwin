import copy
import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.bias_aware_belief import BiasAwareStateUpdateConfig
from bayesian_phystwin.deform360_bias_aware_belief_development import (
    Deform360BiasAwareDevelopmentConfig,
)
from bayesian_phystwin.deform360_selective_bias_guard_diagnostic import (
    apply_frozen_selective_bias_guard_arrays,
    config_from_source_lock,
    selective_reliability_and_variance,
)


def _lock() -> dict[str, object]:
    config = Deform360BiasAwareDevelopmentConfig(
        update_frames=(3, 6),
        minimum_available_center_count=4,
        minimum_motion_center_count=3,
        physical_response_rank=2,
        minimum_physical_response_m=0.0005,
        minimum_observed_motion_m=0.0005,
        state_update=BiasAwareStateUpdateConfig(
            observation_std_m=0.002,
            state_prior_std_m=0.05,
            shared_bias_prior_std_m=0.05,
            camera_bias_prior_std_m=0.05,
        ),
    )
    from dataclasses import asdict

    return {
        "protocol_id": "frozen-source-v4-test",
        "candidate_certified": True,
        "upper_regret_m": -1e-6,
        "config": asdict(config),
    }


def _arrays() -> dict[str, np.ndarray]:
    frame_count = 8
    point_count = 8
    angle = np.linspace(0.0, 2.0 * np.pi, point_count, endpoint=False)
    frame_zero = np.column_stack(
        (0.1 * np.cos(angle), 0.1 * np.sin(angle), np.zeros(point_count))
    ).astype(np.float32)
    persistence = np.repeat(frame_zero[None], frame_count, axis=0)
    measurement = persistence.copy()
    measurement[:, :, 0] += np.linspace(0.0, 0.01, frame_count)[:, None]
    return {
        "persistence": persistence,
        "frame_zero": frame_zero,
        "measurement": measurement,
        "visibility": np.ones((frame_count, point_count), dtype=bool),
        "validity": np.ones((frame_count, point_count), dtype=bool),
        "center_ids": np.arange(point_count),
        "updates": np.asarray([3, 6]),
        "inlier_count": np.full((2, point_count), 3, dtype=np.int16),
        "reprojection": np.ones((2, point_count), dtype=np.float32),
    }


def test_source_lock_config_round_trips_exactly() -> None:
    lock = _lock()

    config = config_from_source_lock(lock)

    assert config.update_frames == (3, 6)
    assert config.state_update.observation_std_m == 0.002


def test_repository_source_v4_lock_round_trips_exactly() -> None:
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "results"
        / "sota"
        / "deform360_bias_aware_guarded_belief_v4"
        / "prospective_lock.json"
    )
    lock = json.loads(path.read_text(encoding="utf-8"))

    config = config_from_source_lock(lock)

    assert config.update_frames == (19, 38, 57)
    assert config.minimum_physical_agreement_gain == 0.4


def test_zero_physical_response_rejects_camera_update_bit_exactly() -> None:
    arrays = _arrays()

    report, selected = apply_frozen_selective_bias_guard_arrays(
        arrays["persistence"],
        arrays["persistence"].copy(),
        arrays["frame_zero"],
        arrays["measurement"],
        arrays["visibility"],
        arrays["validity"],
        center_ids=arrays["center_ids"],
        update_frames=arrays["updates"],
        selected_camera_count=4,
        triangulation_inlier_view_count=arrays["inlier_count"],
        triangulation_median_reprojection_px=arrays["reprojection"],
        source_lock=_lock(),
    )

    assert report["candidate_available_count"] == 0
    assert report["accepted_count"] == 0
    assert report["exact_fallback_interval_count"] == 2
    assert report["driven_backbone_bit_exact_persistence"] is True
    assert report["selected_bit_exact_persistence"] is True
    assert selected.tobytes() == arrays["persistence"].tobytes()


def test_reliability_does_not_depend_on_state_innovation() -> None:
    arrays = _arrays()
    first, first_variance = selective_reliability_and_variance(
        arrays["inlier_count"],
        arrays["reprojection"],
        selected_camera_count=4,
        observation_variance_floor_m2=0.005**2,
        reprojection_scale_px=3.0,
    )
    arrays["measurement"] += 10.0
    second, second_variance = selective_reliability_and_variance(
        arrays["inlier_count"],
        arrays["reprojection"],
        selected_camera_count=4,
        observation_variance_floor_m2=0.005**2,
        reprojection_scale_px=3.0,
    )

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first_variance, second_variance)


def test_source_lock_mutation_is_rejected() -> None:
    lock = copy.deepcopy(_lock())
    lock["candidate_certified"] = False

    with pytest.raises(ValueError, match="did not certify"):
        config_from_source_lock(lock)
