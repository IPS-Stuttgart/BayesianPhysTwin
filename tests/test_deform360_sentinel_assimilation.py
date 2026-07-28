from __future__ import annotations

import numpy as np

from bayesian_phystwin.deform360_dynamic_query import CameraPanel
from bayesian_phystwin.deform360_dynamic_tapnextpp_assimilation import (
    CANDIDATE_ARM,
    PERSISTENCE_ARM,
    BirthAnchoredMeasurements,
    predict_dynamic_tapnextpp_candidate,
)
from bayesian_phystwin.deform360_sentinel_assimilation import (
    build_sentinel_debiased_measurements,
)
from bayesian_phystwin.deform360_sentinel_query_schedule import (
    PREFIX_END_FRAME,
    PROTOCOL_ID,
    SHORT_HORIZON_PROTOCOL_ID,
    Deform360SentinelQueryConfig,
    Deform360SentinelQuerySchedule,
)
from bayesian_phystwin.phystwin_sentinel_queries import (
    ACTIVE_QUERY_ROLE,
    SENTINEL_QUERY_ROLE,
)


def _schedule(
    *,
    birth_frame: int = 0,
) -> Deform360SentinelQuerySchedule:
    config = Deform360SentinelQueryConfig(
        selected_camera_count=3,
        minimum_eligible_camera_count=3,
        total_query_count=4,
        sentinel_query_count=2,
        minimum_camera_support=3,
        graph_basis_rank=2,
        query_birth_frame=birth_frame,
        protocol_id=(
            SHORT_HORIZON_PROTOCOL_ID
            if birth_frame
            else PROTOCOL_ID
        ),
    )
    return Deform360SentinelQuerySchedule(
        update_frames=np.full(4, PREFIX_END_FRAME, dtype=np.int64),
        birth_frames=np.full(4, birth_frame, dtype=np.int64),
        entity_ids=np.arange(4, dtype=np.int64),
        query_roles=np.asarray(
            [
                ACTIVE_QUERY_ROLE,
                ACTIVE_QUERY_ROLE,
                SENTINEL_QUERY_ROLE,
                SENTINEL_QUERY_ROLE,
            ]
        ),
        predicted_motion_m=np.asarray([0.02, 0.03, 0.0002, 0.0003]),
        predicted_visible_views=np.full(4, 3, dtype=np.int64),
        information_gain=np.ones(4),
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
    birth_frame: int = 0,
    missing_sentinel: bool = False,
    inconsistent_sentinels: bool = False,
) -> tuple[np.ndarray, BirthAnchoredMeasurements]:
    physical: np.ndarray = np.zeros((76, 6, 3), dtype=np.float64)
    physical[PREFIX_END_FRAME:, 0, 0] = 0.02
    physical[PREFIX_END_FRAME:, 1, 1] = 0.03
    physical[PREFIX_END_FRAME:, 2, 2] = 0.0002
    physical[PREFIX_END_FRAME:, 3, 2] = 0.0003
    measurement = np.full_like(physical, np.nan)
    covariance = np.full((*physical.shape[:2], 3, 3), np.nan)
    reliability = np.zeros(physical.shape[:2])
    association = np.zeros(physical.shape[:2])
    available = np.zeros(physical.shape[:2], dtype=bool)
    bias = np.asarray([0.01, -0.005, 0.002])
    for entity in range(4):
        displacement = (
            physical[PREFIX_END_FRAME, entity]
            - physical[birth_frame, entity]
        )
        measurement[PREFIX_END_FRAME, entity] = (
            physical[birth_frame, entity] + displacement + bias
        )
        covariance[PREFIX_END_FRAME, entity] = 1e-6 * np.eye(3)
        reliability[PREFIX_END_FRAME, entity] = 1.0
        association[PREFIX_END_FRAME, entity] = 1.0
        available[PREFIX_END_FRAME, entity] = True
    if inconsistent_sentinels:
        measurement[PREFIX_END_FRAME, 3, 0] += 0.1
    if missing_sentinel:
        measurement[PREFIX_END_FRAME, 3] = np.nan
        covariance[PREFIX_END_FRAME, 3] = np.nan
        reliability[PREFIX_END_FRAME, 3] = 0.0
        association[PREFIX_END_FRAME, 3] = 0.0
        available[PREFIX_END_FRAME, 3] = False
    return physical, BirthAnchoredMeasurements(
        measurement_m=measurement,
        covariance_m2=covariance,
        prior_reliability=reliability,
        association_probability=association,
        available=available,
        entity_ids=np.arange(4),
    )


def test_sentinel_debias_recovers_physical_active_displacements() -> None:
    physical, measurements = _inputs()
    result = build_sentinel_debiased_measurements(
        measurements,
        _schedule(),
        physical,
    )

    assert result.applied
    assert result.decision == "sentinel-common-mode-debiased"
    np.testing.assert_allclose(
        result.measurements.measurement_m[PREFIX_END_FRAME, :2],
        physical[PREFIX_END_FRAME, :2],
        atol=1e-12,
    )
    assert np.all(result.measurements.available[PREFIX_END_FRAME, :2])
    assert not np.any(result.measurements.available[:, 2:])
    assert np.all(
        np.linalg.eigvalsh(
            result.measurements.covariance_m2[PREFIX_END_FRAME, :2]
        )
        > 1e-6
    )
    assert result.report()["method_contract"]["sentinels_used_as_state_measurements"] is False


def test_missing_sentinel_forces_exact_persistence_measurements() -> None:
    physical, measurements = _inputs(missing_sentinel=True)
    result = build_sentinel_debiased_measurements(
        measurements,
        _schedule(),
        physical,
    )

    assert not result.applied
    assert result.decision == "exact-persistence-incomplete-sentinel-support"
    assert result.supported_sentinel_count == 1
    assert not np.any(result.measurements.available)
    assert np.all(np.isnan(result.measurements.measurement_m))


def test_inconsistent_sentinel_bias_forces_exact_persistence() -> None:
    physical, measurements = _inputs(inconsistent_sentinels=True)
    result = build_sentinel_debiased_measurements(
        measurements,
        _schedule(),
        physical,
    )

    assert not result.applied
    assert result.estimate.decision == "sentinel-common-mode-inconsistent"
    assert not np.any(result.measurements.available)


def test_rejected_sentinel_arm_is_exact_future_persistence() -> None:
    physical, measurements = _inputs(missing_sentinel=True)
    persistence = np.repeat(physical[0][None], len(physical), axis=0)
    result = build_sentinel_debiased_measurements(
        measurements,
        _schedule(),
        physical,
    )
    _, arrays = predict_dynamic_tapnextpp_candidate(
        physical,
        persistence,
        result.measurements,
    )

    assert not result.applied
    assert np.array_equal(
        arrays[CANDIDATE_ARM][PREFIX_END_FRAME + 1 :],
        arrays[PERSISTENCE_ARM][PREFIX_END_FRAME + 1 :],
    )


def test_short_horizon_debias_uses_the_declared_birth_state() -> None:
    birth_frame = 51
    physical, measurements = _inputs(birth_frame=birth_frame)
    physical[birth_frame, :4] = np.asarray(
        [
            [0.10, 0.00, 0.00],
            [0.00, 0.10, 0.00],
            [0.00, 0.00, 0.10],
            [0.10, 0.10, 0.00],
        ]
    )
    bias = np.asarray([0.01, -0.005, 0.002])
    for entity in range(4):
        displacement = (
            physical[PREFIX_END_FRAME, entity]
            - physical[birth_frame, entity]
        )
        measurements.measurement_m.setflags(write=True)
        measurements.measurement_m[PREFIX_END_FRAME, entity] = (
            physical[birth_frame, entity] + displacement + bias
        )
        measurements.measurement_m.setflags(write=False)
    result = build_sentinel_debiased_measurements(
        measurements,
        _schedule(birth_frame=birth_frame),
        physical,
    )

    assert result.applied
    np.testing.assert_allclose(
        result.measurements.measurement_m[PREFIX_END_FRAME, :2],
        physical[PREFIX_END_FRAME, :2],
        atol=1e-12,
    )
