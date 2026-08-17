from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from bayesian_phystwin.explicit_gauge_prob4d import (
    build_claim_bearing_explicit_gauge_batch,
    update_claim_bearing_explicit_gauge_from_artifacts,
)
from bayesian_phystwin.physical_linearization import PhysicalLinearizationV1
from bayesian_phystwin.prior_aware_gauge_belief import PriorAwareGaugeConfigV1
from bayesian_phystwin.sparse_explicit_gauge_prob4d import (
    NativeSparseExplicitGaugeFactorAdapterResult,
    build_claim_bearing_native_sparse_explicit_gauge_batch,
    update_claim_bearing_native_sparse_explicit_gauge_from_artifacts,
)

ARTIFACT_ID = "a" * 64
PROVIDER_MANIFEST_ID = "b" * 64
GAUGE_CALIBRATION_ID = "c" * 64
POINT_CALIBRATION_ID = "d" * 64
SOURCE_REVISION = "e" * 40


def _fixture() -> tuple[
    SimpleNamespace,
    SimpleNamespace,
    PhysicalLinearizationV1,
    np.ndarray,
]:
    gauge_ids = ("window-0", "window-1")
    frame_indices = np.asarray([0, 0, 1, 1], dtype=np.int64)
    point_ids = np.asarray([0, 1, 0, 1], dtype=np.int64)
    gauge_indices = np.asarray([0, 0, 1, 1], dtype=np.int64)
    view_ids = ("camera-b", "camera-a", "camera-b", "camera-a")
    factor_ids = ("factor-0", "factor-1", "factor-2", "factor-3")
    groups = ("group-0", "group-0", "group-1", "group-1")
    world_mean = np.asarray(
        [
            [0.0, 0.0, 1.000],
            [0.1, 0.0, 1.000],
            [0.0, 0.1, 1.005],
            [0.1, 0.1, 1.005],
        ]
    )
    conditional = np.repeat((np.eye(3) * 1.0e-4)[None], 4, axis=0)
    marginal = conditional + np.eye(3)[None] * 1.0e-5
    local_gauge = np.zeros((4, 3, 7))
    local_gauge[:, 0, 4] = 1.0
    local_gauge[:, 1, 5] = 1.0
    association = np.asarray([1.0, 0.8, 0.9, 0.7])
    reliability = np.asarray([0.95, 0.90, 0.85, 0.80])
    nominal = np.asarray([0.98, 0.98, 0.90, 0.90])
    composite = np.asarray([0.8, 1.0, 0.7, 0.9])
    gauge_prior = np.eye(14) * 1.0e-6
    gauge_prior[4, 11] = gauge_prior[11, 4] = 2.0e-7

    factors: list[SimpleNamespace] = []
    linearized: dict[str, SimpleNamespace] = {}
    for index in range(4):
        gauge_id = gauge_ids[int(gauge_indices[index])]
        factor = SimpleNamespace(
            factor_id=factor_ids[index],
            frame_index=int(frame_indices[index]),
            view_id=view_ids[index],
            window_id=gauge_id,
            gauge_id=gauge_id,
            correlation_group_id=groups[index],
            point_ids=point_ids[index : index + 1],
            valid_mask=np.asarray([True]),
            association_probability=association[index : index + 1],
            prior_reliability=reliability[index : index + 1],
            prior_nominal_probability=float(nominal[index]),
            composite_weight=float(composite[index]),
        )
        factors.append(factor)
        linearized[factor.factor_id] = SimpleNamespace(
            factor_id=factor.factor_id,
            frame_index=factor.frame_index,
            view_id=factor.view_id,
            window_id=factor.window_id,
            gauge_id=factor.gauge_id,
            correlation_group_id=factor.correlation_group_id,
            point_ids=factor.point_ids,
            world_mean_m=world_mean[index : index + 1],
            conditional_world_covariance_m2=conditional[index : index + 1],
            marginal_world_covariance_m2=marginal[index : index + 1],
            gauge_jacobian=local_gauge[index : index + 1],
            valid_mask=factor.valid_mask,
            association_probability=factor.association_probability,
            prior_reliability=factor.prior_reliability,
            prior_nominal_probability=factor.prior_nominal_probability,
            composite_weight=factor.composite_weight,
        )

    calibration_ids = {
        "gauge_artifact_id": GAUGE_CALIBRATION_ID,
        "point_artifact_id": POINT_CALIBRATION_ID,
    }
    attestation = {
        "claim_bearing": True,
        "export_mode": "calibrated",
        "provider_revision": SOURCE_REVISION,
        "provider_manifest_id": PROVIDER_MANIFEST_ID,
        "calibration_artifact_ids": calibration_ids,
        "runtime_revision": {
            "source": "source_checkout",
            "independently_verified": True,
        },
    }
    envelope = SimpleNamespace(
        artifact_id=ARTIFACT_ID,
        bundle_schema_version=4,
        sequence_id="sequence-a",
        case_id="case-a",
        stream_id="prob4d:explicit-gauge-factors",
        source_repository="FlorianPfaff/Prob4D",
        source_revision=SOURCE_REVISION,
        causal_frame_stop=2,
        factor_count=4,
        observation_count=4,
        gauge_ids=gauge_ids,
        gauge_covariance_semantics="joint-cross-window",
        cross_window_gauge_covariance_preserved=True,
        provider_manifest_id=PROVIDER_MANIFEST_ID,
        calibration_artifact_ids=calibration_ids,
        runtime_revision_source="source_checkout",
        runtime_revision_independently_verified=True,
        provider_attestation=attestation,
    )
    bundle = SimpleNamespace(
        sequence_id=envelope.sequence_id,
        case_id=envelope.case_id,
        stream_id=envelope.stream_id,
        source_repository=envelope.source_repository,
        source_revision=envelope.source_revision,
        causal_frame_stop=envelope.causal_frame_stop,
        factors=tuple(factors),
        gauges=tuple(SimpleNamespace(window_id=value) for value in gauge_ids),
        joint_gauge_covariance=gauge_prior,
        linearize=lambda factor: linearized[factor.factor_id],
    )
    validated = SimpleNamespace(
        bundle=bundle,
        envelope=envelope,
        artifact_id=ARTIFACT_ID,
    )
    stack = SimpleNamespace(
        world_mean_m=world_mean,
        conditional_world_covariance_m2=conditional,
        marginal_world_covariance_m2=marginal,
        local_gauge_jacobian=local_gauge,
        gauge_indices=gauge_indices,
        gauge_prior_covariance=gauge_prior,
        association_probability=association,
        prior_reliability=reliability,
        prior_nominal_probability=nominal,
        composite_weight=composite,
        point_ids=point_ids,
        frame_indices=frame_indices,
        view_ids=view_ids,
        factor_ids=factor_ids,
        correlation_group_ids=groups,
        gauge_ids=gauge_ids,
        causal_frame_stop=2,
    )
    state_jacobian = np.zeros((4, 3, 1))
    state_jacobian[:, 2, 0] = 1.0
    query = np.zeros((1, 3, 1))
    query[0, 2, 0] = 1.0
    view_positions = {"camera-a": 0, "camera-b": 1}
    linearization_contract = PhysicalLinearizationV1(
        observation_artifact_id=ARTIFACT_ID,
        baseline_belief_id="f" * 64,
        action_prefix_id="1" * 64,
        simulator_revision="simulator-revision-a",
        frame_ids=frame_indices,
        entity_ids=point_ids,
        view_indices=np.asarray([view_positions[value] for value in view_ids]),
        window_indices=gauge_indices,
        state_jacobian=state_jacobian,
        query_state_jacobian=query,
        physical_response_m=np.asarray([[0.0, 0.0, 0.02]]),
    )
    physical_prediction = world_mean.copy()
    physical_prediction[:, 2] -= 0.005
    return validated, stack, linearization_contract, physical_prediction


def _config() -> PriorAwareGaugeConfigV1:
    return PriorAwareGaugeConfigV1(
        minimum_conditional_information_fraction=0.0,
        minimum_identifiable_fraction=1.0e-8,
        minimum_query_sensitivity_fraction=0.0,
        maximum_state_update_m=1.0,
        maximum_update_to_physical_response_ratio=100.0,
    )


def test_native_sparse_adapter_preserves_strict_factor_semantics() -> None:
    validated, stack, linearization, physical_prediction = _fixture()

    adapted = build_claim_bearing_native_sparse_explicit_gauge_batch(
        validated,
        stack,
        linearization,
        physical_prediction_xyz_m=physical_prediction,
    )

    assert adapted.batch.gauge_jacobian.shape == (4, 3, 0)
    assert adapted.batch.gauge_prior_covariance.shape == (0, 0)
    np.testing.assert_array_equal(
        adapted.sparse_gauge_design.local_gauge_jacobian,
        stack.local_gauge_jacobian,
    )
    np.testing.assert_array_equal(
        adapted.sparse_gauge_design.gauge_prior_covariance,
        stack.gauge_prior_covariance,
    )
    assert adapted.batch.metadata["prob4d_dense_compatibility_bridge"] is False
    assert adapted.batch.metadata["prob4d_native_sparse_gauge_solver"] is True
    assert adapted.batch.metadata["prob4d_dense_gauge_design_materialized"] is False
    assert adapted.dense_gauge_design_avoided_bytes == 4 * 3 * 14 * 8


def test_native_sparse_claim_update_matches_dense_reference() -> None:
    validated, stack, linearization, physical_prediction = _fixture()
    config = _config()

    dense = update_claim_bearing_explicit_gauge_from_artifacts(
        validated,
        stack,
        linearization,
        physical_prediction_xyz_m=physical_prediction,
        config=config,
    )
    sparse = update_claim_bearing_native_sparse_explicit_gauge_from_artifacts(
        validated,
        stack,
        linearization,
        physical_prediction_xyz_m=physical_prediction,
        config=config,
    )

    assert dense.inference_admissible
    assert sparse.inference_admissible
    for name in (
        "state_coefficients",
        "gauge_delta",
        "shared_bias_coefficients",
        "view_bias_coefficients",
        "anchor_bias_coefficients",
        "posterior_covariance",
        "robust_weights",
    ):
        np.testing.assert_allclose(
            getattr(sparse.result, name),
            getattr(dense.result, name),
            atol=2.0e-10,
            rtol=2.0e-9,
        )
    assert (
        sparse.result.input_lineage["prob4d_claim_bearing_sparse_stack_sha256"]
        == dense.result.input_lineage["prob4d_claim_bearing_sparse_stack_sha256"]
    )


def test_native_sparse_path_succeeds_below_dense_memory_limit() -> None:
    validated, stack, linearization, physical_prediction = _fixture()

    with pytest.raises(MemoryError, match="dense compatibility design"):
        build_claim_bearing_explicit_gauge_batch(
            validated,
            stack,
            linearization,
            physical_prediction_xyz_m=physical_prediction,
            maximum_dense_gauge_design_bytes=1,
        )

    sparse = update_claim_bearing_native_sparse_explicit_gauge_from_artifacts(
        validated,
        stack,
        linearization,
        physical_prediction_xyz_m=physical_prediction,
        config=_config(),
    )
    assert sparse.inference_admissible
    assert sparse.result.diagnostics["native_sparse_gauge_design_materialized"] is False


def test_native_sparse_adapter_accepts_optional_bias_and_anchor_inputs() -> None:
    validated, stack, linearization, physical_prediction = _fixture()
    shared = np.zeros((4, 3, 1))
    view = np.zeros((4, 3, 1))
    anchor_innovation = np.asarray([[0.0, 0.0, 0.004]])
    anchor_covariance = np.asarray([np.eye(3) * 1.0e-4])
    anchor_state = np.zeros((1, 3, 1))
    anchor_state[0, 2, 0] = 1.0
    anchor_bias = np.zeros((1, 3, 1))
    anchor_bias[0, 0, 0] = 1.0

    adapted = build_claim_bearing_native_sparse_explicit_gauge_batch(
        validated,
        stack,
        linearization,
        physical_prediction_xyz_m=physical_prediction,
        shared_bias_jacobian=shared,
        view_bias_jacobian=view,
        anchor_innovation_m=anchor_innovation,
        anchor_covariance_m2=anchor_covariance,
        anchor_state_jacobian=anchor_state,
        anchor_correlation_group_ids=("anchor-0",),
        anchor_prior_reliability=np.asarray([1.0]),
        anchor_prior_nominal_probability=np.asarray([0.95]),
        anchor_composite_weight=np.asarray([0.8]),
        anchor_bias_jacobian=anchor_bias,
        anchor_bias_prior_covariance=np.asarray([[1.0e-5]]),
        metadata={"registered_arm": "native-sparse"},
    )

    np.testing.assert_array_equal(adapted.batch.shared_bias_jacobian, shared)
    np.testing.assert_array_equal(adapted.batch.view_bias_jacobian, view)
    assert adapted.batch.metadata["registered_arm"] == "native-sparse"


def test_native_sparse_adapter_rejects_reserved_metadata_override() -> None:
    validated, stack, linearization, physical_prediction = _fixture()
    with pytest.raises(ValueError, match="reserved native-sparse"):
        build_claim_bearing_native_sparse_explicit_gauge_batch(
            validated,
            stack,
            linearization,
            physical_prediction_xyz_m=physical_prediction,
            metadata={"prob4d_native_sparse_gauge_solver": False},
        )


def test_native_sparse_adapter_result_rejects_invalid_accounting() -> None:
    validated, stack, linearization, physical_prediction = _fixture()
    adapted = build_claim_bearing_native_sparse_explicit_gauge_batch(
        validated,
        stack,
        linearization,
        physical_prediction_xyz_m=physical_prediction,
    )
    values = dict(
        batch=adapted.batch,
        sparse_gauge_design=adapted.sparse_gauge_design,
        observation_artifact_id=adapted.observation_artifact_id,
        linearization_artifact_id=adapted.linearization_artifact_id,
        provider_manifest_id=adapted.provider_manifest_id,
        calibration_artifact_ids=adapted.calibration_artifact_ids,
        runtime_revision_source=adapted.runtime_revision_source,
        dense_gauge_design_avoided_bytes=0,
        gauge_ids=adapted.gauge_ids,
        view_ids=adapted.view_ids,
    )
    with pytest.raises(ValueError, match="accounting"):
        NativeSparseExplicitGaugeFactorAdapterResult(**values)
    values["batch"] = object()
    values["dense_gauge_design_avoided_bytes"] = (
        adapted.dense_gauge_design_avoided_bytes
    )
    with pytest.raises(TypeError, match="GaugeAwareObservationBatch"):
        NativeSparseExplicitGaugeFactorAdapterResult(**values)  # type: ignore[arg-type]


def test_native_sparse_adapter_result_rejects_invalid_design_and_gauge_ids() -> None:
    validated, stack, linearization, physical_prediction = _fixture()
    adapted = build_claim_bearing_native_sparse_explicit_gauge_batch(
        validated,
        stack,
        linearization,
        physical_prediction_xyz_m=physical_prediction,
    )
    with pytest.raises(TypeError, match="sparse_gauge_design"):
        replace(adapted, sparse_gauge_design=object())
    with pytest.raises(ValueError, match="gauge IDs differ"):
        replace(adapted, gauge_ids=("other-0", "other-1"))


def test_native_sparse_adapter_rejects_envelope_observation_count_drift() -> None:
    validated, stack, linearization, physical_prediction = _fixture()
    validated.envelope.observation_count += 1
    with pytest.raises(ValueError, match="envelope observation_count"):
        build_claim_bearing_native_sparse_explicit_gauge_batch(
            validated,
            stack,
            linearization,
            physical_prediction_xyz_m=physical_prediction,
        )


@pytest.mark.parametrize(
    ("physical_prediction", "match"),
    (
        (np.zeros((3, 3)), "must have shape"),
        (
            np.asarray(
                [
                    [0.0, 0.0, 1.0],
                    [0.1, 0.0, 1.0],
                    [0.0, 0.1, 1.0],
                    [0.1, 0.1, np.nan],
                ]
            ),
            "must be finite",
        ),
    ),
)
def test_native_sparse_adapter_rejects_invalid_physical_prediction(
    physical_prediction: np.ndarray,
    match: str,
) -> None:
    validated, stack, linearization, _ = _fixture()
    with pytest.raises(ValueError, match=match):
        build_claim_bearing_native_sparse_explicit_gauge_batch(
            validated,
            stack,
            linearization,
            physical_prediction_xyz_m=physical_prediction,
        )


def test_native_sparse_explicit_gauge_module_does_not_import_prob4d() -> None:
    code = """
import sys
import bayesian_phystwin.sparse_explicit_gauge_prob4d
if "prob4d" in sys.modules:
    raise SystemExit("consumer imported the producer implementation")
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
