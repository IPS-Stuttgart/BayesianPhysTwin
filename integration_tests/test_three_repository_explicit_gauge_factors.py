from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from prob4d.gauge import GaugeEstimate
from prob4d.provider_v2_factors import (
    PROVIDER_FACTOR_API_VERSION,
    ObservationFactor,
    ObservationFactorBundle,
    stack_sparse_observation_factors,
)
from prob4d.sim3 import Sim3

from bayesian_phystwin.explicit_gauge_prob4d import (
    build_claim_bearing_explicit_gauge_batch,
    update_claim_bearing_explicit_gauge_from_artifacts,
)
from bayesian_phystwin.physical_linearization import PhysicalLinearizationV1

ARTIFACT_ID = "a" * 64
PROVIDER_MANIFEST_ID = "b" * 64
SOURCE_REVISION = "e" * 40
CALIBRATION_IDS = {
    "gauge_artifact_id": "c" * 64,
    "point_artifact_id": "d" * 64,
}


def _producer_bundle() -> ObservationFactorBundle:
    covariance_0 = np.eye(7, dtype=np.float64) * 2.0e-4
    covariance_1 = np.eye(7, dtype=np.float64) * 3.0e-4
    cross = np.eye(7, dtype=np.float64) * 5.0e-5
    gauges = (
        GaugeEstimate("window-0", Sim3.identity(), covariance_0),
        GaugeEstimate("window-1", Sim3.identity(), covariance_1),
    )
    common = {
        "valid_mask": np.asarray([True, True]),
        "local_covariance_m2": np.repeat(
            np.eye(3, dtype=np.float64)[None] * 1.0e-3,
            2,
            axis=0,
        ),
        "association_probability": np.asarray([0.9, 0.8]),
        "prior_reliability": np.asarray([0.85, 0.75]),
        "prior_nominal_probability": 0.95,
        "composite_weight": 0.5,
        "causal_frame_stop": 6,
    }
    factors = (
        ObservationFactor(
            factor_id="factor-0",
            frame_index=2,
            view_id="camera-0",
            window_id="window-0",
            gauge_id="window-0",
            point_ids=np.asarray([10, 11], dtype=np.int64),
            points_local_m=np.asarray(
                [[0.0, 0.0, 1.0], [0.2, 0.0, 1.1]],
                dtype=np.float64,
            ),
            correlation_group_id="camera-0:frame-2",
            **common,
        ),
        ObservationFactor(
            factor_id="factor-1",
            frame_index=4,
            view_id="camera-0",
            window_id="window-1",
            gauge_id="window-1",
            point_ids=np.asarray([20, 21], dtype=np.int64),
            points_local_m=np.asarray(
                [[0.1, 0.2, 1.2], [0.3, 0.1, 1.3]],
                dtype=np.float64,
            ),
            correlation_group_id="camera-0:frame-4",
            **common,
        ),
    )
    return ObservationFactorBundle(
        sequence_id="sequence-a",
        case_id="case-a",
        stream_id="prob4d:explicit-gauge:camera-0",
        factors=factors,
        gauges=gauges,
        source_repository="FlorianPfaff/Prob4D",
        source_revision=SOURCE_REVISION,
        causal_frame_stop=6,
        joint_gauge_covariance=np.block(
            [
                [covariance_0, cross],
                [cross, covariance_1],
            ]
        ),
        gauge_covariance_semantics="joint-cross-window",
    )


def _claim_wrapper(bundle: ObservationFactorBundle) -> SimpleNamespace:
    runtime = {
        "source": "source_checkout",
        "independently_verified": True,
    }
    attestation = {
        "claim_bearing": True,
        "export_mode": "calibrated",
        "provider_revision": SOURCE_REVISION,
        "provider_manifest_id": PROVIDER_MANIFEST_ID,
        "calibration_artifact_ids": CALIBRATION_IDS,
        "runtime_revision": runtime,
    }
    envelope = SimpleNamespace(
        artifact_id=ARTIFACT_ID,
        bundle_schema_version=4,
        sequence_id=bundle.sequence_id,
        case_id=bundle.case_id,
        stream_id=bundle.stream_id,
        source_repository=bundle.source_repository,
        source_revision=bundle.source_revision,
        causal_frame_stop=bundle.causal_frame_stop,
        factor_count=len(bundle.factors),
        observation_count=sum(len(factor.point_ids) for factor in bundle.factors),
        gauge_ids=tuple(gauge.window_id for gauge in bundle.gauges),
        gauge_covariance_semantics=bundle.gauge_covariance_semantics,
        cross_window_gauge_covariance_preserved=(
            bundle.cross_window_gauge_covariance_preserved
        ),
        provider_manifest_id=PROVIDER_MANIFEST_ID,
        calibration_artifact_ids=CALIBRATION_IDS,
        runtime_revision_source=runtime["source"],
        runtime_revision_independently_verified=True,
        provider_attestation=attestation,
    )
    return SimpleNamespace(
        bundle=bundle,
        envelope=envelope,
        artifact_id=ARTIFACT_ID,
    )


def test_three_repository_explicit_gauge_factor_bridge() -> None:
    assert PROVIDER_FACTOR_API_VERSION == 2
    bundle = _producer_bundle()
    sparse = stack_sparse_observation_factors(bundle)
    validated = _claim_wrapper(bundle)
    count = sparse.observation_count
    state_jacobian = np.zeros((count, 3, 1), dtype=np.float64)
    state_jacobian[:, 2, 0] = 1.0
    query_jacobian = np.zeros((1, 3, 1), dtype=np.float64)
    query_jacobian[0, 2, 0] = 1.0
    linearization = PhysicalLinearizationV1(
        observation_artifact_id=ARTIFACT_ID,
        baseline_belief_id="f" * 64,
        action_prefix_id="1" * 64,
        simulator_revision="installed-wheel-fixture",
        frame_ids=sparse.frame_indices,
        entity_ids=sparse.point_ids,
        view_indices=np.zeros(count, dtype=np.int64),
        window_indices=sparse.gauge_indices,
        state_jacobian=state_jacobian,
        query_state_jacobian=query_jacobian,
        physical_response_m=np.asarray([[0.0, 0.0, 0.02]]),
        metadata={"scope": "three-repository-explicit-gauge-fixture"},
    )
    physical_prediction = sparse.world_mean_m.copy()
    physical_prediction[:, 2] -= 0.003

    adapted = build_claim_bearing_explicit_gauge_batch(
        validated,
        sparse,
        linearization,
        physical_prediction_xyz_m=physical_prediction,
    )

    np.testing.assert_allclose(
        adapted.batch.gauge_jacobian,
        sparse.dense_gauge_jacobian(),
    )
    np.testing.assert_allclose(
        adapted.batch.observation_covariance_m2,
        sparse.conditional_world_covariance_m2,
    )
    np.testing.assert_allclose(
        adapted.batch.gauge_prior_covariance,
        sparse.gauge_prior_covariance,
    )
    assert (
        sparse.sparse_gauge_design_nbytes
        < adapted.dense_gauge_design_bytes
    )
    update = update_claim_bearing_explicit_gauge_from_artifacts(
        validated,
        sparse,
        linearization,
        physical_prediction_xyz_m=physical_prediction,
    )
    assert update.observation_artifact_id == ARTIFACT_ID
    assert update.result.input_lineage[
        "prob4d_claim_bearing_factor_bundle_envelope_artifact_id"
    ] == ARTIFACT_ID
    assert update.result.input_lineage[
        "prob4d_marginal_point_covariance_consumed"
    ] is False
