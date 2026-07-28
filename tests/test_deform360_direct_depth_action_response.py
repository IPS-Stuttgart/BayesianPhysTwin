from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.deform360_direct_depth_action_response import (
    DirectDepthActionResponseConfig,
    evaluate_direct_depth_action_response,
)
from bayesian_phystwin.deform360_dynamic_query import CameraPanel
from bayesian_phystwin.deform360_dynamic_tapnextpp_assimilation import (
    BirthAnchoredMeasurements,
)
from bayesian_phystwin.deform360_sentinel_query_schedule import (
    DYNAMIC_DEPTH_ADMISSION_PROTOCOL_ID,
    Deform360SentinelQueryConfig,
    Deform360SentinelQuerySchedule,
)
from bayesian_phystwin.phystwin_sentinel_queries import (
    ACTIVE_QUERY_ROLE,
    SENTINEL_QUERY_ROLE,
)

BIRTH_FRAME = 13
UPDATE_FRAME = 19


def _schedule() -> Deform360SentinelQuerySchedule:
    config = Deform360SentinelQueryConfig(
        selected_camera_count=3,
        minimum_eligible_camera_count=3,
        total_query_count=12,
        sentinel_query_count=3,
        minimum_camera_support=3,
        graph_basis_rank=2,
        query_birth_frame=BIRTH_FRAME,
        query_update_frame=UPDATE_FRAME,
        protocol_id=DYNAMIC_DEPTH_ADMISSION_PROTOCOL_ID,
    )
    return Deform360SentinelQuerySchedule(
        update_frames=np.full(12, UPDATE_FRAME, dtype=np.int64),
        birth_frames=np.full(12, BIRTH_FRAME, dtype=np.int64),
        entity_ids=np.arange(12, dtype=np.int64),
        query_roles=np.asarray(
            [ACTIVE_QUERY_ROLE] * 9 + [SENTINEL_QUERY_ROLE] * 3
        ),
        predicted_motion_m=np.asarray([0.01] * 9 + [0.0001] * 3),
        predicted_visible_views=np.full(12, 3, dtype=np.int64),
        information_gain=np.ones(12),
        config=config,
        camera_panel=CameraPanel(
            camera_indices=np.arange(3),
            camera_names=("a", "b", "c"),
            frame_zero_coverage=np.ones(3),
            selection_scores=np.ones(3),
        ),
        physical_prefix_sha256="0" * 64,
        graph_basis_sha256="1" * 64,
        artifact_sha256="2" * 64,
    )


def _inputs(
    *,
    observed_scale: float = 1.0,
    orthogonal: bool = False,
    covariance_scale: float = 1.0,
) -> tuple[np.ndarray, BirthAnchoredMeasurements]:
    physical = np.zeros((76, 12, 3), dtype=np.float64)
    physical[BIRTH_FRAME] = np.asarray(
        [
            [0.00, 0.00, 0.00],
            [0.10, 0.00, 0.00],
            [0.20, 0.00, 0.00],
            [0.00, 0.10, 0.00],
            [0.10, 0.10, 0.00],
            [0.20, 0.10, 0.00],
            [0.00, 0.20, 0.00],
            [0.10, 0.20, 0.00],
            [0.20, 0.20, 0.00],
            [0.05, 0.05, 0.00],
            [0.15, 0.05, 0.00],
            [0.10, 0.15, 0.00],
        ]
    )
    physical[: BIRTH_FRAME + 1] = physical[BIRTH_FRAME]
    displacement = np.asarray(
        [
            [0.010, 0.000, 0.000],
            [0.010, 0.002, 0.000],
            [0.010, -0.002, 0.000],
            [0.000, 0.010, 0.000],
            [0.002, 0.010, 0.000],
            [-0.002, 0.010, 0.000],
            [0.007, 0.007, 0.000],
            [0.006, 0.008, 0.000],
            [0.008, 0.006, 0.000],
            [0.0001, 0.0000, 0.0000],
            [0.0000, 0.0001, 0.0000],
            [0.0001, 0.0001, 0.0000],
        ]
    )
    physical[UPDATE_FRAME:] = physical[BIRTH_FRAME] + displacement

    measurement = np.full_like(physical, np.nan)
    covariance = np.full((*physical.shape[:2], 3, 3), np.nan)
    reliability = np.zeros(physical.shape[:2])
    association = np.zeros(physical.shape[:2])
    available = np.zeros(physical.shape[:2], dtype=bool)
    observed = observed_scale * displacement[:9]
    if orthogonal:
        observed = np.column_stack((-observed[:, 1], observed[:, 0], observed[:, 2]))
    measurement[UPDATE_FRAME, :9] = physical[BIRTH_FRAME, :9] + observed
    covariance[UPDATE_FRAME, :9] = covariance_scale * 1e-8 * np.eye(3)
    reliability[UPDATE_FRAME, :9] = 1.0
    association[UPDATE_FRAME, :9] = 1.0
    available[UPDATE_FRAME, :9] = True
    return physical, BirthAnchoredMeasurements(
        measurement_m=measurement,
        covariance_m2=covariance,
        prior_reliability=reliability,
        association_probability=association,
        available=available,
        entity_ids=np.arange(9),
    )


def test_dynamic_depth_schedule_accepts_only_registered_endpoint_pairs() -> None:
    assert _schedule().config.query_update_frame == UPDATE_FRAME
    with pytest.raises(ValueError, match="endpoint pair"):
        Deform360SentinelQueryConfig(
            query_birth_frame=12,
            query_update_frame=19,
            protocol_id=DYNAMIC_DEPTH_ADMISSION_PROTOCOL_ID,
        )


def test_aligned_direct_depth_response_is_admitted() -> None:
    physical, measurements = _inputs()
    result = evaluate_direct_depth_action_response(
        "source",
        physical,
        measurements,
        _schedule(),
        sentinel_applied=True,
        actuator_displacement_m=0.02,
    )

    assert result.admitted
    assert result.reason == "action-aligned-direct-depth-response"
    assert result.supported_active_count == 9
    assert result.passing_group_count == 3
    assert all(group.effective_count <= 3.0 for group in result.groups)
    assert result.descriptor()["information_boundary"]["future_metric_read"] is False


def test_missing_sentinel_bias_forces_rejection() -> None:
    physical, measurements = _inputs()
    result = evaluate_direct_depth_action_response(
        "source",
        physical,
        measurements,
        _schedule(),
        sentinel_applied=False,
        actuator_displacement_m=0.02,
    )

    assert not result.admitted
    assert result.reason == "sentinel-common-bias-unavailable"


def test_orthogonal_metric_motion_fails_action_response_gate() -> None:
    physical, measurements = _inputs(orthogonal=True)
    result = evaluate_direct_depth_action_response(
        "source",
        physical,
        measurements,
        _schedule(),
        sentinel_applied=True,
        actuator_displacement_m=0.02,
    )

    assert not result.admitted
    assert result.reason == "insufficient-action-aligned-response"
    assert result.passing_group_count == 0


def test_tiny_actuator_motion_rejects_an_otherwise_aligned_response() -> None:
    physical, measurements = _inputs()
    result = evaluate_direct_depth_action_response(
        "source",
        physical,
        measurements,
        _schedule(),
        sentinel_applied=True,
        actuator_displacement_m=0.001,
    )

    assert not result.admitted
    assert result.reason == "insufficient-actuator-displacement"


def test_effective_information_is_capped_for_correlated_identities() -> None:
    physical, measurements = _inputs()
    result = evaluate_direct_depth_action_response(
        "source",
        physical,
        measurements,
        _schedule(),
        sentinel_applied=True,
        actuator_displacement_m=0.02,
        config=DirectDepthActionResponseConfig(maximum_effective_count=1.0),
    )

    assert all(group.effective_count <= 1.0 for group in result.groups)
    assert all(group.response_gain_std > 0.0 for group in result.groups)


def test_artifact_changes_when_prefix_measurement_changes() -> None:
    physical, measurements = _inputs()
    first = evaluate_direct_depth_action_response(
        "source",
        physical,
        measurements,
        _schedule(),
        sentinel_applied=True,
        actuator_displacement_m=0.02,
    )
    physical_again, changed = _inputs(observed_scale=0.9)
    second = evaluate_direct_depth_action_response(
        "source",
        physical_again,
        changed,
        _schedule(),
        sentinel_applied=True,
        actuator_displacement_m=0.02,
    )

    assert first.artifact_sha256 != second.artifact_sha256

    physical_third, changed_covariance = _inputs(covariance_scale=2.0)
    third = evaluate_direct_depth_action_response(
        "source",
        physical_third,
        changed_covariance,
        _schedule(),
        sentinel_applied=True,
        actuator_displacement_m=0.02,
    )
    assert first.measurement_prefix_sha256 != third.measurement_prefix_sha256
    assert first.artifact_sha256 != third.artifact_sha256
